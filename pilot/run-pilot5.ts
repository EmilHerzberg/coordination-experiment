// Pilot v2 entrypoint — three-rung ladder on the five-bucket ledger.
//
//   node --experimental-transform-types --import ./register.mjs run-pilot5.ts estimate
//   node --experimental-transform-types --import ./register.mjs run-pilot5.ts probe     (1 model × 1 event, real calls)
//   node --experimental-transform-types --import ./register.mjs run-pilot5.ts full      (all models × events)
//   node --experimental-transform-types --import ./register.mjs run-pilot5.ts full --fresh   (throw the level away and restart)
//   PILOT5_EVENT_LIMIT=50 ... run-pilot5.ts full   (D32: only the first 50 events of the shuffled order; resume lifts it)
//
// `estimate` makes NO network calls: it prices the prompts with the local
// tokenizer and the per-model $/M rates from models_verified.md, assuming an
// output length per call. Use it before any paid run.
//
// Output dir: thesis/scaffold/pilot5/ — every run-output file is written under
// the level prefix it belongs to (L1_ledger5.jsonl, L1_outcomes.jsonl,
// L1_runs_raw.jsonl, L1_run_status.txt, plus the four artefacts below).
// Nothing is copied by hand afterwards. The old scheme wrote fixed names and
// relied on the operator renaming them: that produced six pairs of byte-
// identical duplicates in pilot5/ and left `ledger_report_v2.md` describing a
// DIFFERENT run than the `ledger5.jsonl` sitting next to it.
import { execFileSync } from 'node:child_process';
import { appendFileSync, writeFileSync, readFileSync, renameSync, unlinkSync, mkdirSync, existsSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import { hostname } from 'node:os';
import { fileURLToPath } from 'node:url';
import { OpenRouterClient, CostCapError, Throttle, withDeadline, type ReasoningMode, type CostState } from './openrouter-client.js';
import { getKey } from './env.js';
import { EVIDENCE_BLOCKS } from './evidence.js';
import { runLadder, structureRecords, STRUCTURES, COMBINER_ROLES, type LadderParams } from './structures5.js';
import { summarize, localTokens, renderSegments, measureTemplateOverhead, type CallLedgerRecord } from './ledger5.js';
import { workerPrompt, aggregatorPrompt, debateRevisePrompt, type EvidencePack } from './prompts.js';
import { buildReport } from './make-report5.js';
import { buildTranscript } from './make-transcript.js';
import { buildStructureTokens, buildMeta } from './make-structure-tokens.js';
import type { LLMClient } from '../src/types.js';

const HERE = dirname(fileURLToPath(import.meta.url));
// PILOT5_OUTDIR exists so the tests can drive the whole resume path without
// writing a single byte into pilot5/, which holds the committed pilot data.
const OUTDIR = process.env.PILOT5_OUTDIR
  ? resolve(process.env.PILOT5_OUTDIR)
  : resolve(HERE, '..', '..', 'pilot5');

const MODE = (process.argv[2] ?? 'estimate').toLowerCase();

/**
 * Opt-out from resume. It DESTROYS whatever the level already contains, so it
 * has to be asked for explicitly — and only exactly; a bare `--fres` or
 * `PILOT5_FRESH=0` must never be read as consent to throw ~$22 of work away.
 */
export function wantsFresh(argv: readonly string[], env: Record<string, string | undefined>): boolean {
  return argv.includes('--fresh') || env.PILOT5_FRESH === '1';
}
const FRESH = wantsFresh(process.argv.slice(3), process.env);
const CAP_USD = Number(process.env.PILOT5_CAP_USD ?? '2.0');   // hard stop; confirm with the user before raising
// Event-level parallelism. Each (model, event) pair is an independent ladder run,
// so they can proceed concurrently; the per-model Throttle keeps the request rate
// safe. Wall-clock, not money, is the binding constraint on the real run.
const CONCURRENCY = Number(process.env.PILOT5_CONCURRENCY ?? '6');
// Repeat the whole event set N times, to get sample size on rare failures.
const REPEATS = Math.max(1, Number(process.env.PILOT5_REPEATS ?? '1'));
// D32 (2026-09-06): the pre-declared operational checkpoint. When set, the
// work list is cut to the first N DISTINCT events of the seed-shuffled order of
// the FULL task list (both models; the same N events at both levels), so the
// real run can be paused after its first N events for a BLIND operational
// check — do the APIs answer, does anything break, what does a ladder cost in
// money and minutes. Lifting the variable is an ordinary resume: the N events
// are on disk and skipped, the remaining ones run. See checkpointEvents().
// D32 deviation 3 (2026-09-08): a ladder that does not complete within this
// wall-clock budget is treated as an OUTAGE (FAIL line, no outcomes line,
// retried on the next resume). Found after 21 h of the real run: 8 of 10 slots
// per level were silently hung. Real ladders take 3-40 min; 60 min is generous.
const LADDER_TIMEOUT_MS = Number(process.env.PILOT5_LADDER_TIMEOUT_MS ?? String(60 * 60 * 1000));
const EVENT_LIMIT = process.env.PILOT5_EVENT_LIMIT ? Math.max(1, Math.floor(Number(process.env.PILOT5_EVENT_LIMIT))) : null;

export interface ModelSpec {
  key: string; slug: string; provider: string; mode: ReasoningMode;
  arm: 'reasoning' | 'non-reasoning';
  evidence: 'all' | 'first';
  /** $/M tokens in → out, from models_verified.md (2026-07-24). */
  priceIn: number; priceOut: number;
  /** Minimum spacing between request starts (ms). See the 429 note on MODELS. */
  minIntervalMs?: number;
  /**
   * Per-provider override of minIntervalMs for a split model (D29). Added
   * 2026-09-07 as D32 deviation 1: 15 h into the real run DeepInfra throttled
   * qwen into a retry spiral (12 outages/h, ~half of its calls needing 429
   * retries) while SiliconFlow ran clean at 3 s, and the retrying ladders
   * clogged the concurrency slots of BOTH levels. A rate setting only.
   */
  minIntervalMsByProvider?: Record<string, number>;
}

/** The request spacing for one model on one provider: the provider override, else the model's, else none. */
export function throttleIntervalFor(m: ModelSpec, provider: string): number {
  return m.minIntervalMsByProvider?.[provider] ?? m.minIntervalMs ?? 0;
}

// EXPERIMENT ROSTER — decided 2026-08-26 (D20). Two open-weight families, one
// reasoning arm and one non-reasoning arm, exactly as the exposé's Scope section
// requires. Both served by DeepInfra; prices and quantisation re-verified live
// against the OpenRouter endpoints API on 2026-08-26.
//
// The two arms differ in family, parameter scale (32B dense vs 671B MoE) AND
// serving precision (fp8 vs fp4). This is a DELIBERATE, author-approved choice:
// RQ3's model contrast is explicitly exploratory and the models are declared
// not directly comparable in the limitations (see DECISIONS_PROMPT_SESSION.md
// D20). RQ1 and RQ2 are unaffected — they are within-model comparisons.
//
// Prices here are for the offline estimate only; real cost comes from usage.cost.
export const MODELS: ModelSpec[] = [
  // minIntervalMs raised 300 -> 3000 on 2026-09-04: the end-to-end test run
  // showed qwen at 0.3s spacing failing 32%% of calls on DeepInfra (429), against
  // 0%% for deepseek at 3s. One failed call drops a whole ~18-call event, so every
  // event was lost. The throttle is per-model per-PROCESS, so two concurrent
  // levels halve the effective spacing again. Match deepseek. See testrun10.
  { key: 'qwen3-32b@on',          slug: 'qwen/qwen3-32b',                 provider: 'DeepInfra', mode: 'reason_on', arm: 'reasoning',     evidence: 'all', priceIn: 0.08, priceOut: 0.28, minIntervalMs: 3000, minIntervalMsByProvider: { DeepInfra: 6000 } },
  // RE-PINNED 2026-08-28 (D24) from DeepInfra fp4 to SiliconFlow fp8.
  //
  // DeepInfra returned BYTE-IDENTICAL text for concurrent identical prompts —
  // 6 responses, 2 distinct, same SHA-256 within each round — which at
  // temperature 0.7 cannot come from independent sampling. It also produced
  // degenerate output (whitespace, prompt echoes, runaway repetition). Both
  // are fatal here: the communication-free rung IS independent repeated
  // sampling, so a host that collapses identical concurrent requests destroys
  // the baseline. Measured A/B on the real MSFT prompt, interleaved in one
  // session: DeepInfra 0/6 well-formed, SiliconFlow 6/6 and 6 distinct.
  // Qwen on DeepInfra was checked and is NOT affected (3/3 and 8/8 distinct).
  //
  // minIntervalMs is retained from the 2026-08-26 upstream-429 incident.
  { key: 'deepseek-chat-v3-0324', slug: 'deepseek/deepseek-chat-v3-0324', provider: 'SiliconFlow', mode: 'none',      arm: 'non-reasoning', evidence: 'all', priceIn: 0.25, priceOut: 1.00, minIntervalMs: 3000 },
];

/**
 * Verified alternatives kept out of the roster but available for a robustness
 * check or a substitution if a host drops a model (D9 requires that the served
 * checkpoint be pinned, so any swap must be re-verified and recorded):
 *   llama-3.3-70b   meta-llama/llama-3.3-70b-instruct  DeepInfra fp8  0.10/0.32
 *   deepseek-r1     deepseek/deepseek-r1-0528          SiliconFlow fp8 0.50/2.18
 *                   (NOTE: DeepInfra serves r1 at fp4 for the same price —
 *                    SiliconFlow's fp8 is strictly better value if r1 is used.)
 *   qwen3-32b@off   qwen/qwen3-32b                     SiliconFlow fp8 0.14/0.57
 *                   (the on/off toggle only works on SiliconFlow; DeepInfra
 *                    ignores reasoning:{enabled:false} and keeps thinking on.)
 */
export const RESERVE_MODELS: ModelSpec[] = [
  { key: 'llama-3.3-70b',  slug: 'meta-llama/llama-3.3-70b-instruct', provider: 'DeepInfra',   mode: 'none',       arm: 'non-reasoning', evidence: 'first', priceIn: 0.10, priceOut: 0.32 },
  { key: 'deepseek-r1',    slug: 'deepseek/deepseek-r1-0528',         provider: 'SiliconFlow', mode: 'reason_on',  arm: 'reasoning',     evidence: 'first', priceIn: 0.50, priceOut: 2.18 },
  { key: 'qwen3-32b@off',  slug: 'qwen/qwen3-32b',                    provider: 'SiliconFlow', mode: 'reason_off', arm: 'non-reasoning', evidence: 'first', priceIn: 0.14, priceOut: 0.57 },
];

// D29 (2026-09-05): qwen3-32b@on is LOAD-SPLIT across two fp8 endpoints. During
// the end-to-end test run DeepInfra throttled qwen on a sustained quota (a single
// event booked in ~50 min at minIntervalMs=3000). SiliconFlow serves the same
// model at the same fp8 precision, so the two are ONE arm, not two.
//
// probe-qwen-providers.ts, 2026-09-05, 8 concurrent identical calls per provider,
// interleaved, real worker prompt, temp 0.7:
//   - DISTINCT content 8/8 AND distinct gen_id 8/8 on BOTH -> no concurrent
//     collapse either side (this is the D24 failure that re-pinned deepseek;
//     qwen is clean on both). RQ1's independent-sampling assumption holds.
//   - reasoning fires 8/8 on both (the @on toggle works on SiliconFlow too).
//   - prompt_tokens 629 (DeepInfra) vs 628 (SiliconFlow): a 1-token chat-template
//     offset, NOT a content-tokenization difference. The ledger treats the
//     provider's prompt_tokens as authoritative and rescales the D/A/B/B_SELF
//     buckets to it (ledger5.ts:12-13, 307-311), so D+A+B+B_SELF=prompt_tokens
//     holds for 628 or 629 alike; no identity breaks. The offset lands only on
//     the shared D baseline (content-identical across conditions), 0.16%.
//
// Events are pinned to a provider by a hash of their BASE id (level- and
// repeat-independent), so (a) every call in one ladder run shares one token
// basis and (b) L1 and L2 of an event use the same provider. The offset is then
// constant within every event and CANCELS in the within-event comparisons that
// RQ1-paired / RQ2 / RQ3 make. modelKey stays 'qwen3-32b@on' (one arm); the
// provider actually used is recorded per call as provider_pinned/returned.
const PROVIDER_SPLIT: Record<string, string[]> = {
  'qwen3-32b@on': ['DeepInfra', 'SiliconFlow'],
};

/**
 * Strip the "#rN" repeat suffix AND the "_L1"/"_L2" level suffix, so all
 * repeats and BOTH levels of an event share its provider.
 *
 * The level strip was missing until 2026-09-05: the runner builds ev.id as
 * `${pack.id}_L${LEVEL}` (see the events mapping below), so hashing ev.id
 * directly sent ~half the events to a DIFFERENT provider at L2 than at L1 —
 * the exact opposite of what D29 documents, and it would have confounded the
 * qwen level contrast with an endpoint change. Caught by the adversarial
 * review of the analysis plan (REVIEW_ANALYSIS_PLAN_2026-09-05.md, L2-03);
 * order matters: the repeat suffix is appended after the level suffix.
 */
export function baseEventId(id: string): string {
  return id.replace(/#r\d+$/, '').replace(/_L\d+$/, '');
}

/** djb2, stable across processes/runs (no Math.random, no Date). */
function hash32(s: string): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h * 33) ^ s.charCodeAt(i)) >>> 0;
  return h >>> 0;
}

/**
 * Deterministic event->provider assignment for split models; identity for the
 * rest. Same base id -> same provider on every run and at both levels, so the
 * split is fully reproducible and never mixes two providers inside one ladder.
 */
export function providerForEvent(m: ModelSpec, evId: string): string {
  const split = PROVIDER_SPLIT[m.key];
  if (!split || split.length < 2) return m.provider;
  return split[hash32(baseEventId(evId)) % split.length];
}

export const PARAMS: LadderParams = {
  agentCount: 3,
  debateRounds: 2,        // round 0 (= workers) + 2 revisions
  temperature: 0.7,
  maxTokensPerCall: 8000,   // r1 traces run 1.5-3k hidden tokens; leave headroom so truncation never causes "malformed"
  maxParseAttempts: 2,    // D16: pre-registered MINIMUM (1 retry); may be raised, every attempt logged
  aggregate: 'mean',      // B5 CLOSED (D19): expose mandates averaging; Brier convexity guarantees mean >= typical individual
};

// --------------------------------------------------------------- file layout
/**
 * Every output file of one level, in one place.
 *
 * The level prefix is applied HERE and never afterwards. Writing to a fixed
 * name and renaming later was a human convention, and the human forgot: the
 * level-2 run of 2026-08-28 has transcripts but no report, and the surviving
 * unprefixed `ledger_report_v2.md` belongs to the level-1 run while the
 * unprefixed `ledger5.jsonl` next to it belongs to level 2.
 *
 * `mode` adds a second, coarser prefix: a PROBE writes `probe_L{level}_*` and
 * shares not one run-output file with `full`. A probe is a one-event smoke test
 * that deliberately skips the resume/purge logic — and it used to append to the
 * very files `full` treats as authoritative. Reproduced before this fix: two
 * probe runs on ONE event left 2 outcomes lines and 40 ledger records where an
 * honest single pass writes 20, and the next `full` then reported "resuming: 1
 * of 2 done" and never purged the duplication, because the event looked
 * complete. Separate names make that structurally impossible instead of a rule
 * the operator has to remember at 3am.
 *
 * The calibration cache stays UNPREFIXED, and is the ONE thing a probe and a
 * full run still share, on purpose: it is a per-model measurement (C2 / D18)
 * holding the model's fixed chat-template overhead, not run output. It is a
 * property of the model, identical whichever mode measured it, and sharing it
 * is what lets the cheap probe pay for it once. It is now one file PER MODEL —
 * see `overheadPathFor` for why a single shared object could not survive two
 * sessions.
 */
export function outputPaths(outDir: string, level: number, mode: string = 'full') {
  const prefix = mode === 'probe' ? `probe_L${level}_` : `L${level}_`;
  const p = (name: string) => resolve(outDir, `${prefix}${name}`);
  return {
    /** Filename prefix of this (level, mode) — quoted on the status line. */
    prefix,
    ledger: p('ledger5.jsonl'),
    outcomes: p('outcomes.jsonl'),
    raw: p('runs_raw.jsonl'),
    status: p('run_status.txt'),
    report: p('report.md'),
    transcript: p('transcript.md'),
    transcriptFull: p('transcript_full.md'),
    structureTokens: p('structure_tokens.jsonl'),
    lock: p('run.lock'),
    /**
     * The pre-2026-09-01 single shared cache. Read ONLY to migrate it into the
     * per-model files, then renamed aside; nothing writes it any more.
     */
    overheadLegacy: resolve(outDir, 'template_overhead.json'),
  };
}

// -------------------------------------------------- calibration cache (C2/D18)
/**
 * Where one model's measured chat-template overhead lives.
 *
 * It used to be one `template_overhead.json` holding every model: each session
 * read it into memory once and later rewrote the WHOLE object, so a concurrent
 * session's entries were silently dropped. Reproduced: a full run calibrating
 * qwen3-32b@on and deepseek-chat-v3-0324 on level 1, concurrent with a probe
 * calibrating llama-3.3-70b on level 2. The probe measured and wrote llama
 * first; the full session then rewrote the file from its own in-memory copy and
 * the file ended as {"qwen3-32b@on": 7, "deepseek-chat-v3-0324": 7}. The llama
 * measurement — a paid call — was gone, and would be paid for again.
 *
 * ONE FILE PER MODEL, chosen over a read-modify-write under a lock: a per-model
 * write is independent and idempotent by construction, so there is no window to
 * lose, nothing to serialise, and no second lock that can go stale and block a
 * $31.50 run. It also confines a torn write to the one model it belongs to.
 *
 * The filename carries the model key, percent-escaped down to [a-z0-9._-]:
 * `qwen3-32b@on` becomes `template_overhead.qwen3-32b%40on.json`. That is
 * filesystem-safe (none of Windows' < > : " / \ | ? * survive the escape) and
 * exactly reversible, which matters because the key is the join back to the
 * ledger. Upper case is escaped too, and the escapes use lower-case hex, so
 * the whole encoded key is lower case: Windows compares filenames
 * case-insensitively, and `Qwen` and `qwen` would otherwise be ONE file with
 * one model silently reading the other's calibration. Every key in MODELS and
 * RESERVE_MODELS is lower case today, so this costs nothing now and cannot
 * bite later.
 */
export function encodeModelKey(key: string): string {
  return key.replace(/[^a-z0-9._-]/g, c =>
    Array.from(new TextEncoder().encode(c))
      .map(b => '%' + b.toString(16).padStart(2, '0')).join(''));
}

/** Inverse of `encodeModelKey`. `%` itself escapes to `%25`, so this is exact. */
export function decodeModelKey(encoded: string): string {
  return decodeURIComponent(encoded);
}

export function overheadPathFor(outDir: string, modelKey: string): string {
  return resolve(outDir, `template_overhead.${encodeModelKey(modelKey)}.json`);
}

/**
 * This model's cached overhead, or undefined if it has never been measured.
 *
 * An unreadable cache file returns undefined rather than throwing: re-measuring
 * costs one cheap call, and aborting a 1.7-day run over a torn 60-byte cache
 * would be the expensive answer. It is said out loud, though — a silent
 * re-measurement is how two disagreeing numbers stay invisible.
 */
export function readOverhead(outDir: string, modelKey: string, status?: Status): number | undefined {
  const p = overheadPathFor(outDir, modelKey);
  if (!existsSync(p)) return undefined;
  try {
    const o = JSON.parse(readFileSync(p, 'utf-8'));
    if (typeof o?.overhead === 'number') return o.overhead;
    status?.(`[calib] ${modelKey}: ${basename(p)} holds no numeric overhead; re-measuring`);
  } catch (e: any) {
    status?.(`[calib] ${modelKey}: ${basename(p)} is unreadable (${e?.message}); re-measuring`);
  }
  return undefined;
}

/**
 * Record one model's measurement. Independent of every other model's file, so
 * two sessions writing two models cannot interact at all.
 *
 * Temp file then rename, for the same reason as `purgeFile`: a concurrent
 * session reading this file must never see half of it. The temp name carries
 * the pid so two sessions re-measuring the SAME model cannot collide on the
 * temp either; the rename is atomic and both write the same number anyway.
 */
export function writeOverhead(
  outDir: string,
  modelKey: string,
  overhead: number,
  /** When this number was MEASURED. A migrated value was not measured now, and
   *  saying it was would be a small lie in the one file whose whole job is to
   *  say where a number came from. */
  measuredAt: string = new Date().toISOString(),
): string {
  const p = overheadPathFor(outDir, modelKey);
  const tmp = `${p}.${process.pid}.tmp`;
  writeFileSync(tmp, JSON.stringify({ modelKey, overhead, measuredAt }, null, 2) + '\n');
  renameSync(tmp, p);
  return p;
}

/**
 * Split a pre-existing `template_overhead.json` into per-model files.
 *
 * It holds REAL measurements that were paid for, so it is migrated rather than
 * orphaned — and then renamed aside rather than deleted, so the evidence
 * survives if the split ever turns out to be wrong. An entry that already has a
 * per-model file is left alone: that file is at least as new as this one.
 */
export function migrateOverheadCache(outDir: string, legacyPath: string, status: Status): number {
  if (!existsSync(legacyPath)) return 0;
  const obj = JSON.parse(readFileSync(legacyPath, 'utf-8')) as Record<string, unknown>;
  let moved = 0;
  for (const [key, overhead] of Object.entries(obj)) {
    if (typeof overhead !== 'number') continue;
    if (existsSync(overheadPathFor(outDir, key))) continue;
    writeOverhead(outDir, key, overhead, `unknown — migrated from ${basename(legacyPath)}, which recorded no measurement time`);
    moved++;
  }
  const keep = `${legacyPath}.migrated`;
  // A concurrent session may have renamed it already; that is success, not failure.
  try { renameSync(legacyPath, keep); } catch { /* already migrated by the other session */ }
  status(`[calib] migrated ${moved} measurement(s) out of ${basename(legacyPath)} into per-model files; the original is kept as ${basename(keep)}`);
  return moved;
}

// ------------------------------------------------------- resume, pure part
/**
 * The natural key of one ladder run.
 *
 * The event id already carries everything needed to make this unique: the
 * level is appended when the packs are loaded (`_L2`) and the repeat index by
 * the REPEATS loop (`#r2`), so two levels or two repeats of the same event can
 * never collide. NUL is the separator because it is the one character that
 * neither an event id nor a model key (`qwen3-32b@on`) can contain — with a
 * printable separator `a|b` + `c` and `a` + `b|c` would be the same key.
 */
export function taskKey(eventId: string, modelKey: string): string {
  return `${eventId}\u0000${modelKey}`;
}

/** Key of a ledger record (`ledger5.jsonl`). */
export const ledgerKeyOf = (r: { eventId?: string; modelKey?: string }): string | null =>
  r.eventId && r.modelKey ? taskKey(r.eventId, r.modelKey) : null;

/**
 * Key of a raw client-log record (`runs_raw.jsonl`), which names the same two
 * fields differently. The calibration probe is logged before any event is
 * begun and so has no evidence_id; it returns null and is therefore kept —
 * a record attributable to no event cannot be double-counted by anything.
 */
export const rawKeyOf = (r: { evidence_id?: string; model_key?: string }): string | null =>
  r.evidence_id && r.model_key ? taskKey(r.evidence_id, r.model_key) : null;

/**
 * The set of (eventId, modelKey) that finished. One outcomes line is appended
 * only after a whole ladder run returns, so its presence is the only proof
 * that the event is complete.
 */
export function completedKeys(outcomeLines: readonly string[]): Set<string> {
  const keys = new Set<string>();
  for (const raw of outcomeLines) {
    const line = raw.trim();
    if (!line) continue;
    let o: { eventId?: unknown; modelKey?: unknown };
    try {
      o = JSON.parse(line);
    } catch {
      // A kill mid-append can leave the final line half-written. That is not
      // completed work, so dropping it is the correct reading: the event is
      // re-run and its orphaned ledger records are purged below.
      continue;
    }
    if (typeof o.eventId === 'string' && typeof o.modelKey === 'string') {
      keys.add(taskKey(o.eventId, o.modelKey));
    }
  }
  return keys;
}

/**
 * Split the work list into what still has to run and what is already on disk.
 *
 * Order is preserved in both halves. The compute-matching prefix rule (D23)
 * and `buildStructureTokens` both read the ledger in generation order, so a
 * resumed run must EXTEND that order rather than reshuffle it.
 */
/**
 * Deterministic in-place Fisher-Yates shuffle (mulberry32 PRNG). Seeded so the
 * order is reproducible across resumes and machines; Math.random would make
 * every resume see a different order, which is exactly what determinism rules
 * out. Exported for the tests.
 */
export function shuffleInPlace<T>(arr: T[], seed: number): T[] {
  let s = seed >>> 0;
  const rand = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/** The one seed of the work order (D31/II-8). Also the seed the checkpoint set is derived from (D32). */
export const WORK_ORDER_SEED = 20260905;

/**
 * The base events of the first `n` distinct events in the seed-shuffled order of
 * the FULL task list (D32). Level-invariant by construction: the task lists of
 * L1 and L2 have the same length and the same (model, pack) layout, ids differ
 * only by the level suffix that baseEventId() strips, and the shuffle permutes
 * indices — so both levels pick the same companies and quarters. Independent
 * of what is already on disk, so a resume sees the same set.
 */
export function checkpointEvents<T extends { m: { key: string }; ev: { id: string } }>(
  tasks: readonly T[],
  n: number,
): Set<string> {
  const keep = new Set<string>();
  for (const t of shuffleInPlace([...tasks], WORK_ORDER_SEED)) {
    if (keep.size >= n) break;
    keep.add(baseEventId(t.ev.id));
  }
  return keep;
}

export function planWork<T extends { m: { key: string }; ev: { id: string } }>(
  tasks: readonly T[],
  doneKeys: ReadonlySet<string>,
): { todo: T[]; skipped: T[] } {
  const todo: T[] = [];
  const skipped: T[] = [];
  for (const t of tasks) {
    if (doneKeys.has(taskKey(t.ev.id, t.m.key))) skipped.push(t);
    else todo.push(t);
  }
  return { todo, skipped };
}

/**
 * Drop records belonging to an event that never finished.
 *
 * THIS IS THE SUBTLE PART OF RESUMING. One event writes ~20 ledger lines in a
 * loop and then ONE outcomes line; a kill anywhere inside that window leaves
 * ledger records with no outcomes line. On resume the event runs again and
 * appends a SECOND set of records — and every downstream identity still holds
 * (D+A+B+B_SELF = prompt_tokens, C1+C2 = completion, `commCost` well-formed
 * per structure), because each record is individually valid. Only the totals
 * are wrong, by however much of the half-run survived. Nothing in the report
 * or the token summary would flag it, which is why the half-run is removed
 * before anything new is appended rather than reconciled afterwards.
 */
export function purgeOrphans<T>(
  records: readonly T[],
  doneKeys: ReadonlySet<string>,
  keyOf: (r: T) => string | null = ledgerKeyOf as unknown as (r: T) => string | null,
): { kept: T[]; removed: T[] } {
  const kept: T[] = [];
  const removed: T[] = [];
  for (const r of records) {
    const k = keyOf(r);
    if (k === null || doneKeys.has(k)) kept.push(r);
    else removed.push(r);
  }
  return { kept, removed };
}

// --------------------------------------------------- resume, filesystem part
/**
 * Split a JSONL file into its lines WITHOUT ever holding the file as one
 * string. D32 deviation 4 (2026-09-09): `L2_runs_raw.jsonl` reached 554 MB and
 * `readFileSync(path, 'utf-8')` threw "Cannot create a string longer than
 * 0x1fffffe8 characters" (Node's string ceiling, ≈ 512 MB) — the resume died
 * at startup. A Buffer has no such ceiling on 64-bit Node; each line is decoded
 * on its own. `\r\n` is accepted exactly as the old `split(/\r?\n/)` did.
 */
function readLines(path: string): string[] {
  if (!existsSync(path)) return [];
  const buf = readFileSync(path);
  const lines: string[] = [];
  let start = 0;
  for (;;) {
    const nl = buf.indexOf(0x0a, start);
    const end = nl === -1 ? buf.length : nl;
    const cut = end > start && buf[end - 1] === 0x0d ? end - 1 : end;
    if (cut > start) lines.push(buf.toString('utf-8', start, cut));
    if (nl === -1) break;
    start = nl + 1;
  }
  return lines;
}

/**
 * Rewrite a JSONL file keeping only the lines of completed runs.
 *
 * Two deliberate properties:
 *  - The ORIGINAL line text is kept for survivors, never a re-serialisation.
 *    A purge is then a pure deletion, so a resumed run that has nothing to do
 *    leaves the files byte-identical.
 *  - Temp file then rename. A crash halfway through the purge would otherwise
 *    leave a truncated ledger, which is strictly worse than the orphans it was
 *    cleaning up: orphans are detectable, a half-written line is not.
 */
export interface PurgeIO {
  /** Create (or truncate) `path` with the first chunk. */
  write(path: string, data: string): void;
  /**
   * Continue `path` with a further chunk. Optional only for the test fakes
   * whose files fit in one chunk; the real io always has it (deviation 4).
   */
  append?(path: string, data: string): void;
  rename(from: string, to: string): void;
}
/** The real thing. Injectable only so a test can make either half fail. */
const REAL_PURGE_IO: PurgeIO = {
  write: (p, d) => writeFileSync(p, d),
  append: (p, d) => appendFileSync(p, d),
  rename: (f, t) => renameSync(f, t),
};
/**
 * Survivors are written in chunks of at most this many characters (plus the
 * line that crosses the boundary), so the temp file of a 554 MB raw log is
 * never one joined string. 64 MB is far below the ceiling and large enough
 * that a purge is still a handful of writes.
 */
const PURGE_CHUNK_CHARS = 64 * 1024 * 1024;

export function purgeFile(
  path: string,
  doneKeys: ReadonlySet<string>,
  keyOf: (r: any) => string | null,
  io: PurgeIO = REAL_PURGE_IO,
  chunkChars: number = PURGE_CHUNK_CHARS,
): number {
  const lines = readLines(path);
  if (!lines.length) return 0;
  const parsed = lines.map(line => {
    try { return { line, rec: JSON.parse(line) as any }; } catch { return { line, rec: null }; }
  });
  // An unparseable line is a torn write, which by definition belongs to the run
  // that was killed — so it is removed with the rest of that half-run.
  const { kept, removed } = purgeOrphans(parsed, doneKeys, p => (p.rec === null ? '\u0000torn' : keyOf(p.rec)));
  if (!removed.length) return 0;               // untouched file, untouched bytes
  const tmp = `${path}.purge.tmp`;
  try {
    // First chunk creates the temp file (an empty file if nothing survives),
    // every further chunk is appended. A chunk is closed as soon as it reaches
    // `chunkChars`, so it overshoots by at most one line.
    let chunk: string[] = [];
    let size = 0;
    let first = true;
    const flush = () => {
      const data = chunk.length ? chunk.join('\n') + '\n' : '';
      if (first) { io.write(tmp, data); first = false; }
      else {
        if (!io.append) throw new Error('purgeFile: survivors exceed one chunk but the PurgeIO has no append()');
        io.append(tmp, data);
      }
      chunk = []; size = 0;
    };
    for (const k of kept) {
      chunk.push(k.line);
      size += k.line.length + 1;
      if (size >= chunkChars) flush();
    }
    if (first || chunk.length) flush();
    io.rename(tmp, path);
  } catch (e) {
    // Whatever failed, `path` is still the ORIGINAL complete file — that is the
    // entire point of writing beside it and renaming. Remove the fragment we
    // left behind so the next run does not inherit a mystery, then rethrow:
    // failing loudly here is right, continuing on a ledger we no longer
    // understand is not.
    try { if (existsSync(tmp)) unlinkSync(tmp); } catch { /* nothing to clean up */ }
    throw e;
  }
  return removed.length;
}

// ---------------------------------------------------------------------- lock
/**
 * Is `pid` a live process? `kill(pid, 0)` sends no signal, it only asks.
 * EPERM means the pid exists but belongs to another user — still alive.
 */
export function isPidAlive(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; } catch (e: any) { return e?.code === 'EPERM'; }
}

export interface LockInfo { pid: number; host: string; started: string; mode: string; level: number }

/**
 * Read a lock file. A file that exists but cannot be parsed is reported as an
 * UNKNOWN holder rather than as "no lock": something wrote it, and guessing
 * that it is dead is exactly the guess that duplicates 800 events.
 */
export function readLockInfo(text: string | null): LockInfo | null {
  if (text === null) return null;
  try {
    const o = JSON.parse(text);
    if (typeof o?.pid === 'number') {
      return {
        pid: o.pid, host: String(o.host ?? ''), started: String(o.started ?? '?'),
        mode: String(o.mode ?? '?'), level: Number(o.level ?? 0),
      };
    }
  } catch { /* torn write, or hand-edited */ }
  // The host is a sentinel no real hostname can equal (spaces, parentheses),
  // so the 'other host -> do not adjudicate' branch below always takes it.
  return { pid: 0, host: '(unreadable lock file)', started: 'unknown', mode: 'unknown', level: 0 };
}

/**
 * Advisory single-writer lock, one per (level, mode).
 *
 * `doneKeys` is a snapshot taken once at process start and `appendFileSync`
 * offers no cross-process dedup, so two `full` processes on one level each see
 * the other's work as not done and run all 1,600 ladder runs twice — ~$31 and
 * a ledger in which every event appears twice while every per-record identity
 * still holds. Nothing downstream can detect that. So it must not happen.
 *
 * A crashed run must stay recoverable: the lock records the pid and the host,
 * and a lock whose pid is gone is STALE — removed automatically with a loud
 * status line rather than left to block the restart. A lock from another host,
 * or one we cannot parse, is not something we can adjudicate, so the operator
 * is told the exact file to delete. Ctrl-C kills the process before the
 * `finally` below can run; that leaves precisely such a stale lock, and the
 * next start clears it by itself.
 */
export interface LockHooks {
  /**
   * Called after the "no lock present" observation and BEFORE the exclusive
   * write. Injectable only so a test can slide a second, complete claim into
   * exactly that window — the window the `wx` flag below exists for. Nothing
   * in production passes it. Without it the race can only be hoped for: both
   * claims are synchronous, so process timing decides whether it ever happens.
   */
  beforeClaim?: () => void;
}

export function acquireLock(path: string, info: LockInfo, status: Status, hooks: LockHooks = {}): () => void {
  const held = readLockInfo(existsSync(path) ? readFileSync(path, 'utf-8') : null);
  if (held) {
    if (held.host !== info.host || isPidAlive(held.pid)) {
      status(`!! REFUSING TO START: ${basename(path)} is held by pid ${held.pid} on ${held.host} (${held.mode}, level ${held.level}, since ${held.started})`);
      throw new Error(
        `another session holds ${path} (pid ${held.pid} on ${held.host}, started ${held.started}). ` +
        `Two processes writing one level duplicate every event they both run. ` +
        `If that process is really gone, delete the lock and start again:  rm "${path}"`,
      );
    }
    status(`!! STALE LOCK cleared: ${basename(path)} named pid ${held.pid} (started ${held.started}), which no longer exists — a previous session was killed. Whatever it left half-finished is cleaned up by the resume step below.`);
    try { unlinkSync(path); } catch { /* lost the race; the exclusive write below will refuse */ }
  }
  hooks.beforeClaim?.();
  try {
    // `wx` is what makes this a CLAIM and not an overwrite. Both sessions above
    // can legitimately observe "no lock present" — the check and the write are
    // two syscalls — and without the flag both would then write, both would
    // believe they hold the level, and every event they share would be run and
    // billed twice. EEXIST means we lost; losing is the correct outcome.
    writeFileSync(path, JSON.stringify(info) + '\n', { flag: 'wx' });
  } catch (e: any) {
    if (e?.code === 'EEXIST') {
      const winner = readLockInfo(existsSync(path) ? readFileSync(path, 'utf-8') : null);
      const who = winner
        ? `pid ${winner.pid} on ${winner.host} (${winner.mode}, level ${winner.level}, since ${winner.started})`
        : 'a session that has already released it';
      status(`!! LOST THE RACE for ${basename(path)}: ${who} claimed it in the same instant`);
      throw new Error(`another session claimed ${path} at the same instant, held by ${who}; not starting`);
    }
    throw e;
  }
  let released = false;
  return () => {
    if (released) return;
    released = true;
    try {
      // Remove only OUR lock. If a stale-lock recovery has already handed the
      // file to a different session, deleting it would unlock a live run.
      const now = readLockInfo(existsSync(path) ? readFileSync(path, 'utf-8') : null);
      if (now && now.pid === info.pid && now.started === info.started) unlinkSync(path);
    } catch { /* already gone */ }
  };
}

// -------------------------------------------------------------------- status
/** Writes one line to ONE session's status file, and to the console. */
export type Status = (line: string) => void;

/**
 * A status logger bound to a single status file.
 *
 * This used to be a module-level `let statusPath`, reassigned synchronously at
 * the top of every run() and read by every later status() call. Two run() calls
 * in one process therefore shared one variable — and running level 1 and
 * level 2 side by side in one process is exactly how the 1.7-day wall clock
 * gets halved. The second reassignment won and the first session's log stopped
 * dead after its header. Reproduced: L1 and L2 concurrently, three events each,
 * left a 2-line L1_run_status.txt (its two header lines, nothing else) and a
 * 20-line L2_run_status.txt holding L1's calibration line, L1's three progress
 * lines, L1's "FULL done" and L1's four artefact lines. The runs themselves
 * were fine; the record of them was gone, and over 1.7 days that record IS the
 * diagnostic.
 *
 * A closure per run() call has nothing a second call can reassign.
 */
export function makeStatus(statusPath: string): Status {
  return (line: string) => {
    mkdirSync(dirname(statusPath), { recursive: true });
    // APPEND, never truncate. Across a 1.7-day run interruptions are certain, and
    // overwriting the log on restart would erase the record of what happened
    // exactly at the moment it matters.
    appendFileSync(statusPath, line + '\n');
    console.log(line);
  };
}

// main()'s FATAL handler fires outside any session, so its path has to be fixed
// at import time. It is its OWN logger, not a variable a session can reassign.
const fatalStatus = makeStatus(
  outputPaths(OUTDIR, (process.env.PILOT5_LEVEL ?? '1') === '2' ? 2 : 1, MODE).status,
);

// Evidence source. By default the synthetic pilot blocks; with PILOT5_PACKS set,
// the real level-1 / level-2 packs exported from thesis.db by
// scripts/20_build_evidence_packs.py. PILOT5_LEVEL picks the tier (1 or 2).
const PACKS_PATH = process.env.PILOT5_PACKS ? resolve(OUTDIR, process.env.PILOT5_PACKS) : null;
const LEVEL = (process.env.PILOT5_LEVEL ?? '1') === '2' ? 2 : 1;

interface RealPack {
  id: string; symbol: string; reportDate: string; t0: string;
  label: number; abnormalReturn: number; level1: string; level2: string;
}

const realPacks: RealPack[] = PACKS_PATH ? (JSON.parse(readFileSync(PACKS_PATH, 'utf-8')) as RealPack[]) : [];

const events: EvidencePack[] = PACKS_PATH
  ? realPacks.map(p => ({
      // The {id, text} object handed to the ladder never carries the label or
      // the realised return, so no model can read the answer. Be precise about
      // where the truth DOES live at run time: in the packs JSON on disk (the
      // pack builder writes label/abnormalReturn there) — not in any file this
      // runner writes. The analysis plan's blinding wording follows from that.
      id: `${p.id}_L${LEVEL}`,
      text: LEVEL === 2 ? p.level2 : p.level1,
    }))
  : EVIDENCE_BLOCKS.map(b => ({ id: b.id, text: b.description }));

// ------------------------------------------------------------------ estimate
function estimate(): void {
  // Assumed output per call (visible + hidden), by arm. Pilot v1 medians were
  // ~250 visible tokens; reasoning arms added ~750 (qwen) to ~1700 (r1) hidden.
  const OUT = { 'non-reasoning': 300, reasoning: 1500 } as const;
  const N = PARAMS.agentCount, R = PARAMS.debateRounds;
  let grand = 0;
  console.log(`estimate: N=${N}, debate rounds=${R}, assumed out/call: ${JSON.stringify(OUT)}\n`);
  console.log('| model | events | calls | ~prompt tok | ~completion tok | ~$ |');
  console.log('|---|---:|---:|---:|---:|---:|');
  // D32: with PILOT5_EVENT_LIMIT set, price exactly the checkpoint set.
  const keep = EVENT_LIMIT !== null
    ? checkpointEvents(MODELS.flatMap(m => events.map(ev => ({ m, ev }))), EVENT_LIMIT)
    : null;
  if (keep) console.log(`event limit ${EVENT_LIMIT} (D32 checkpoint): pricing ${keep.size} of ${events.length} events
`);
  for (const m of MODELS) {
    const pool = keep ? events.filter(ev => keep.has(baseEventId(ev.id))) : events;
    const evs = m.evidence === 'all' ? pool : pool.slice(0, 1);
    let pTok = 0, cTok = 0, calls = 0;
    const out = OUT[m.arm];
    for (const ev of evs) {
      const w = workerPrompt(ev);
      const wTok = localTokens(renderSegments(w.system)) + localTokens(renderSegments(w.user)) + 10;
      pTok += N * wTok; cTok += N * out; calls += N;
      // aggregator: reasons ≈ visible out each
      const fakeReason = 'x'.repeat(0) + ' word'.repeat(200);
      const a = aggregatorPrompt(ev, Array.from({ length: N }, () => fakeReason));
      pTok += localTokens(renderSegments(a.system)) + localTokens(renderSegments(a.user)) + 10; cTok += out; calls += 1;
      // debate: each round every agent reads N-1 peers (≈ visible out each) + own
      const fakeFull = `<reason>${fakeReason}</reason><decision>0.5</decision>`;
      const d = debateRevisePrompt(ev, { agentId: '1', fullText: fakeFull },
        Array.from({ length: N - 1 }, (_, i) => ({ agentId: String(i + 2), fullText: fakeFull })), 1);
      const dTok = localTokens(renderSegments(d.system)) + localTokens(renderSegments(d.user)) + 10;
      pTok += R * N * dTok; cTok += R * N * out; calls += R * N;
    }
    const usd = (pTok * m.priceIn + cTok * m.priceOut) / 1e6;
    grand += usd;
    console.log(`| ${m.key} | ${evs.length} | ${calls} | ${pTok} | ${cTok} | ${usd.toFixed(3)} |`);
  }
  console.log(`\nTOTAL ≈ $${grand.toFixed(3)}  (cap set to $${CAP_USD}; retries and long reasoning chains push this up)`);
}

// ----------------------------------------------------------------- artefacts
/**
 * The four derived files, from the run that just finished.
 *
 * They used to be four separate manual commands, which is precisely why the
 * level-2 run of 2026-08-28 has transcripts but NO report — the step was
 * forgotten. Generating them here means the artefacts always describe the
 * ledger they sit next to.
 */
function writeArtefacts(P: ReturnType<typeof outputPaths>, packs: RealPack[], status: Status): boolean {
  const records = readLines(P.ledger).map(l => JSON.parse(l) as CallLedgerRecord);
  const outcomes = readLines(P.outcomes).map(l => JSON.parse(l) as any);
  if (!records.length) {
    status(`[report] SKIPPED: ${P.ledger} has no records — nothing to report on`);
    return false;
  }
  let ok = true;
  // Each artefact is attempted independently: one builder throwing must not
  // cost the author the other three.
  const step = (label: string, fn: () => string) => {
    try {
      const written = fn();
      status(`[report] wrote ${written}`);
    } catch (e: any) {
      // The expensive part already succeeded and the raw data is on disk, so
      // this is NOT a failed run — but it must not slip past either, so main()
      // exits non-zero on the strength of this flag.
      ok = false;
      status(`[report] FAILED to build ${label}: ${e?.stack ?? e}`);
    }
  };
  step(P.report, () => { writeFileSync(P.report, buildReport(records, outcomes)); return P.report; });
  step(P.transcript, () => { writeFileSync(P.transcript, buildTranscript(records, outcomes, packs, false)); return P.transcript; });
  step(P.transcriptFull, () => { writeFileSync(P.transcriptFull, buildTranscript(records, outcomes, packs, true)); return P.transcriptFull; });
  step(P.structureTokens, () => {
    const meta = buildMeta(P.ledger, records);
    const lines = buildStructureTokens(records);
    writeFileSync(P.structureTokens, [JSON.stringify({ _meta: meta }), ...lines.map(l => JSON.stringify(l))].join('\n') + '\n');
    return P.structureTokens;
  });
  return ok;
}

// ---------------------------------------------------------------------- run
/** A client the ladder can drive. Injectable so tests need no key and no network. */
export type LadderClient = LLMClient & { beginStructureRun(configName: string, evidenceId: string): void };

export interface RunOptions {
  outDir?: string;
  level?: number;
  mode?: string;
  fresh?: boolean;
  capUsd?: number;
  concurrency?: number;
  repeats?: number;
  /** D32 checkpoint: run only the first N distinct events of the shuffled order. */
  eventLimit?: number | null;
  /** D32 deviation 3: wall-clock watchdog per ladder, ms. */
  ladderTimeoutMs?: number;
  events?: EvidencePack[];
  packs?: RealPack[];
  models?: ModelSpec[];
  /** Injected client factory; when given, no API key is read and no host is contacted. */
  makeClient?: (m: ModelSpec) => LadderClient;
}

export interface RunResult {
  planned: number; skipped: number; completed: number; dropped: number;
  purgedLedger: number; purgedRaw: number;
  artefactsOk: boolean;
}

async function run(modelKeys: string[], evLimit: number | null, opts: RunOptions = {}): Promise<RunResult> {
  const outDir = opts.outDir ?? OUTDIR;
  const level = opts.level ?? LEVEL;
  const mode = opts.mode ?? MODE;
  const fresh = opts.fresh ?? FRESH;
  const capUsd = opts.capUsd ?? CAP_USD;
  const concurrency = opts.concurrency ?? CONCURRENCY;
  const repeats = opts.repeats ?? REPEATS;
  const eventLimit = opts.eventLimit === undefined ? EVENT_LIMIT : opts.eventLimit;
  const ladderTimeoutMs = opts.ladderTimeoutMs ?? LADDER_TIMEOUT_MS;
  const evs0 = opts.events ?? events;
  const roster = (opts.models ?? MODELS).filter(m => modelKeys.includes(m.key));

  mkdirSync(outDir, { recursive: true });
  const P = outputPaths(outDir, level, mode);
  // Local to THIS call. A module-level path shared by two concurrent run()
  // calls is what silently emptied one level's status log — see makeStatus.
  const status = makeStatus(P.status);

  // One writer per (level, mode). Released in the `finally` at the end of the
  // session, including when the session throws.
  const releaseLock = acquireLock(P.lock, {
    pid: process.pid, host: hostname(), started: new Date().toISOString(), mode, level,
  }, status);
  try {
    const cost: CostState = { spentUsd: 0, capUsd };

    // One header line per SESSION, appended. Reading the status file top to
    // bottom now tells you how many times the run was interrupted and restarted.
    // IV-2 (D31): the harness version is part of the run record. Same lookup
    // as make-structure-tokens' provenance header; 'unknown' must never block
    // a run (version recording is bookkeeping, not a gate).
    let harnessCommit = 'unknown';
    try {
      harnessCommit = execFileSync('git', ['rev-parse', '--short', 'HEAD'],
        { cwd: dirname(fileURLToPath(import.meta.url)), encoding: 'utf-8' }).trim();
    } catch { /* not a git checkout; recorded as unknown */ }
    status(`=== ${mode.toUpperCase()} session ${new Date().toISOString()} cap=$${capUsd} level=${level} evidence=${PACKS_PATH ? 'REAL' : 'synthetic'} concurrency=${concurrency} repeats=${repeats} harness=${harnessCommit} ===`);

    // A cache written by an older build is split into per-model files before
    // anything reads it, so no measurement that was paid for is orphaned.
    migrateOverheadCache(outDir, P.overheadLegacy, status);

    // ---- one shared throttle per MODEL×PROVIDER. Concurrent events each get
    // their own client (so per-event logging stays correct) but must share the
    // rate limit, or six parallel events would each get their own budget and
    // burst again. For a split model (qwen@on, D29) each provider gets its own
    // throttle, so the two fp8 endpoints are rate-limited independently and the
    // aggregate qwen rate is ~2x that of a single endpoint.
    const throttleKey = (mkey: string, provider: string) => `${mkey} ${provider}`;
    const throttles = new Map<string, Throttle>();
    for (const m of roster) {
      for (const provider of PROVIDER_SPLIT[m.key] ?? [m.provider]) {
        throttles.set(throttleKey(m.key, provider), new Throttle(throttleIntervalFor(m, provider)));
      }
    }

    const apiKey = opts.makeClient ? '' : getKey();
    // provider is resolved per event (D29 split); the test hook ignores it.
    const mkClient: (m: ModelSpec, provider: string) => LadderClient = opts.makeClient
      ? (m, _provider) => opts.makeClient!(m)
      : (m, provider) => new OpenRouterClient({
          modelKey: m.key, modelSlug: m.slug, provider, mode: m.mode,
          apiKey, logPath: P.raw, cost, throttle: throttles.get(throttleKey(m.key, provider)),
        }) as unknown as LadderClient;

    // ---- resume or restart. Never both, and never silently.
    let doneKeys = new Set<string>();
    let purgedLedger = 0, purgedRaw = 0;
    if (fresh) {
      const discarded = completedKeys(readLines(P.outcomes)).size;
      writeFileSync(P.ledger, ''); writeFileSync(P.outcomes, ''); writeFileSync(P.raw, '');
      status(`!! --fresh: TRUNCATED ${basename(P.ledger)}, ${basename(P.outcomes)}, ${basename(P.raw)} — ${discarded} completed ladder runs discarded`);
    } else if (mode === 'full') {
      doneKeys = completedKeys(readLines(P.outcomes));
      // The outcomes file is purged against its OWN keys, so nothing complete can
      // be lost; the only thing this removes is a final line torn by the kill.
      // It has to go, because the artefact builders JSON.parse every line and
      // would otherwise die at the end of a 1.7-day run over half a line.
      const purgedOutcomes = purgeFile(P.outcomes, doneKeys, ledgerKeyOf);
      purgedLedger = purgeFile(P.ledger, doneKeys, ledgerKeyOf);
      purgedRaw = purgeFile(P.raw, doneKeys, rawKeyOf);
      if (purgedLedger || purgedRaw || purgedOutcomes) {
        status(`purged ${purgedLedger} orphan ledger records, ${purgedRaw} orphan raw records and ${purgedOutcomes} torn outcomes lines left by an interrupted run`);
      }
    } else if (mode === 'probe') {
      // A probe is a disposable smoke test, so its own files start empty every
      // time: a probe file always shows exactly one pass, never the sum of the
      // last three. Nothing else reads them — `full` neither reads nor writes a
      // `probe_` file — so there is nothing here to lose.
      writeFileSync(P.ledger, ''); writeFileSync(P.outcomes, ''); writeFileSync(P.raw, '');
      status(`[probe] writing to ${P.prefix}* in ${outDir} — smoke-test files, truncated at session start. The authoritative ${`L${level}_`}* files are neither read nor written by a probe.`);
    }

    // ---- the work list: every (model, event) pair is an independent ladder run.
    // PILOT5_REPEATS repeats the whole event set. Format failures are rare (the one
    // observed was 1 in 6 aggregator calls and did not reproduce), so a single pass
    // over three events cannot distinguish a fixed prompt from a lucky draw.
    // Repeats buy sample size cheaply; repeated events carry an "#rN" id suffix.
    const allTasks = roster.flatMap(m =>
      Array.from({ length: repeats }, (_, rep) =>
        (m.evidence === 'all' ? evs0 : evs0.slice(0, 1))
          .slice(0, evLimit ?? undefined)
          .map(ev => ({ m, ev: rep === 0 ? ev : { ...ev, id: `${ev.id}#r${rep + 1}` } })),
      ).flat(),
    );
    // D32: the checkpoint cut, if one is set. Applied to the FULL list before
    // the done-set is removed, so the set does not depend on what is on disk.
    let work = allTasks;
    if (eventLimit !== null) {
      const keep = checkpointEvents(allTasks, eventLimit);
      work = allTasks.filter(t => keep.has(baseEventId(t.ev.id)));
      status(`event limit ${eventLimit} (D32 checkpoint): ${keep.size} events = ${work.length} of ${allTasks.length} ladder runs; lift PILOT5_EVENT_LIMIT to resume the rest`);
    }
    const { todo: tasks, skipped } = planWork(work, doneKeys);
    // D31/II-8 (2026-09-05): the work list is SHUFFLED with a fixed seed, so an
    // early close (cost cap, kill) leaves an ~random subset of events instead
    // of an alphabetical prefix "A to somewhere" — and, as a bonus, the models
    // are interleaved, which removes the starvation the 2026-09-04 test run
    // documented (all of one model's events queued before the other's).
    //
    // NOT to be confused with the sample draw: WHICH events run is decided
    // solely by the pinned event_sample.txt (its own seed, 20260903). This
    // shuffle only changes the ORDER they are worked through. It is seeded and
    // deterministic, so resumes see the same order; and it does not touch the
    // within-event generation order of the repetition pool, which is what
    // compute matching consumes.
    shuffleInPlace(tasks, WORK_ORDER_SEED);
    if (skipped.length) status(`resuming: ${skipped.length} of ${work.length} done, ${tasks.length} to go`);
    status(`${tasks.length} ladder runs over ${roster.length} models, concurrency ${concurrency}, order seed-shuffled (D31)`);

    // ---- calibration first, sequentially: one cheap probe per model (D18).
    const paramsByModel = new Map<string, LadderParams>();
    for (const m of roster) {
      // Read PER MODEL, here, rather than snapshotting one whole file at session
      // start: a concurrent session that measured this model in the meantime has
      // already paid for it, and this picks that up instead of paying again.
      let overhead = readOverhead(outDir, m.key, status);
      if (overhead === undefined) {
        try {
          // Template overhead is a model-level constant; the 1-token cross-provider
          // offset (D29) is far below the proportional-residual noise the ledger
          // already spreads, so one calibration on the default provider suffices.
          const probe = await measureTemplateOverhead(mkClient(m, m.provider));
          overhead = probe.overhead;
          writeOverhead(outDir, m.key, overhead);
          status(`[calib] ${m.key}: template overhead ${overhead} tok/call (provider ${probe.providerPrompt} vs local ${probe.localCount})`);
        } catch (e: any) {
          overhead = 0;
          status(`[calib] ${m.key}: probe FAILED (${e?.message}); falling back to purely proportional allocation`);
        }
      }
      paramsByModel.set(m.key, { ...PARAMS, templateOverhead: overhead });
    }

    let done = 0, dropped = 0, aborted = false;
    await pool(tasks, concurrency, async ({ m, ev }) => {
      if (aborted) return;
      if (cost.spentUsd >= capUsd) {
        if (!aborted) { aborted = true; status(`!! cost cap $${capUsd} reached; stopping`); }
        return;
      }
      // Fresh client per task: beginStructureRun sets per-run logging state, which
      // concurrent tasks would otherwise clobber. The throttle is still shared.
      // provider is resolved per event so a split model's ladder runs entirely on
      // one endpoint (D29); non-split models get their fixed provider.
      const provider = providerForEvent(m, ev.id);
      const client = mkClient(m, provider);
      client.beginStructureRun('ladder', ev.id);
      const t0 = Date.now();
      try {
        // Watchdog (D32 deviation 3): a hung ladder must fail like an outage,
        // not hold a slot forever. The abandoned ladder's promise keeps running
        // in the background but nothing below ever reads it, so it can never
        // write a ledger record or an outcomes line.
        const res = await withDeadline(
          runLadder(client, ev, m.key, paramsByModel.get(m.key)!),
          ladderTimeoutMs,
        ).catch((e: unknown) => {
          if (e instanceof Error && /\(deadline\)/.test(e.message)) {
            throw new Error(`ladder watchdog: no completion within ${Math.round(ladderTimeoutMs / 60000)} min`);
          }
          throw e;
        });
        // appendFileSync is synchronous, so concurrent writers cannot interleave a line.
        // The ledger records go down first and the outcomes line LAST: that ordering is
        // what makes a missing outcomes line mean "interrupted", which resume relies on.
        for (const r of res.ledger.records) appendFileSync(P.ledger, JSON.stringify(r) + '\n');
        // `provider` makes the D29 split auditable per event (the analysis
        // plan's provider-invariance check needs it in the outcomes, not only
        // in the raw log); `aggregatorNumberLeaks` persists the D27 residual
        // leak count, which was computed per event but discarded until
        // 2026-09-05 (review finding L2-12).
        appendFileSync(P.outcomes, JSON.stringify({ modelKey: m.key, eventId: ev.id, provider, outcomes: res.outcomes, roundProbabilities: res.roundProbabilities, matched: res.matched, repetitionRuns: res.repetitionRuns, aggregatorNumberLeaks: res.aggregatorNumberLeaks }) + '\n');
        const line = STRUCTURES.map(s => {
          const o = res.outcomes[s];
          const t = summarize(structureRecords(res.ledger.records, s), COMBINER_ROLES);
          if (o.dropped) dropped++;
          const mm = res.matched.find(x => x.structure === s);
          const vs = mm && mm.probability !== null ? ` vs rep@${mm.runsUsed}run=${mm.probability.toFixed(2)}` : '';
          return `${s}=${o.dropped ? 'DROP' : o.probability!.toFixed(2)}(comm ${t.commCost})${vs}`;
        }).join(' ');
        done++;
        status(`[${mode}] (${done}/${tasks.length}) ${m.key} | ${ev.id} | ${line} | ${((Date.now() - t0) / 1000).toFixed(0)}s | $${cost.spentUsd.toFixed(4)}`);
      } catch (e: any) {
        // Reaching here means runLadder did NOT return a result: an outage, or
        // the cost cap. No outcomes line has been written for this event and none
        // will be, so it is not done and the next resume runs it again. Say so on
        // the line — a wall of bare FAILs at 3am reads like lost data, and the
        // whole point of the fix is that this data is not lost.
        if (e instanceof CostCapError) {
          aborted = true;
          status(`!! ${e.message} — ${ev.id} wrote no outcomes line; it will be retried on the next resume`);
          return;
        }
        status(`[${mode}] ${m.key} | ${ev.id} | FAIL ${e?.message} (no outcomes line written; retried on the next resume)`);
      }
    });
    status(`${mode.toUpperCase()} done. ${done}/${tasks.length} ladder runs this session (${skipped.length} already on disk), ${dropped} structure drops, total=$${cost.spentUsd.toFixed(4)}`);

    // Artefacts only for a `full` run. Since the probe now writes under its own
    // prefix it could no longer overwrite the level's report — but a one-event
    // smoke test is not a dataset: every sanity row and every spread diagnostic
    // in the report is meaningless on a single event, and a probe_L1_report.md
    // would exist only to be mistaken for one.
    const artefactsOk = mode === 'full' ? writeArtefacts(P, opts.packs ?? realPacks, status) : true;

    return { planned: work.length, skipped: skipped.length, completed: done, dropped, purgedLedger, purgedRaw, artefactsOk };
  } finally {
    releaseLock();
  }
}

/** Exported so tests can drive the whole resume path with a mock client. */
export { run };

/** Run `fn` over `items` with at most `limit` in flight. */
async function pool<T>(items: T[], limit: number, fn: (item: T) => Promise<void>): Promise<void> {
  let next = 0;
  const workers = Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, async () => {
    for (;;) {
      const i = next++;
      if (i >= items.length) return;
      await fn(items[i]);
    }
  });
  await Promise.all(workers);
}

async function main(): Promise<void> {
  if (MODE === 'estimate') { estimate(); return; }
  let r: RunResult;
  if (MODE === 'probe') r = await run([process.env.PILOT5_PROBE_MODEL ?? 'qwen3-32b@on'], 1);
  else if (MODE === 'full') r = await run(MODELS.map(m => m.key), null);
  else throw new Error(`unknown mode ${MODE}`);
  // The ladder runs succeeded and the data is on disk; only a derived file is
  // missing. Non-zero so a wrapper or a CI step cannot mistake it for clean.
  if (!r.artefactsOk) process.exit(1);
}

// Guarded so `run-pilot5.ts` can be imported (by the tests, and by anything
// wanting MODELS or the resume helpers) without launching a run.
const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  main().catch(e => { fatalStatus(`FATAL: ${e?.stack ?? e}`); process.exit(1); });
}
