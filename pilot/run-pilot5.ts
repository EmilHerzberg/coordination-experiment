// Pilot v2 entrypoint — three-rung ladder on the five-bucket ledger.
//
//   node --experimental-transform-types --import ./register.mjs run-pilot5.ts estimate
//   node --experimental-transform-types --import ./register.mjs run-pilot5.ts probe     (1 model × 1 event, real calls)
//   node --experimental-transform-types --import ./register.mjs run-pilot5.ts full      (all models × events)
//
// `estimate` makes NO network calls: it prices the prompts with the local
// tokenizer and the per-model $/M rates from models_verified.md, assuming an
// output length per call. Use it before any paid run.
//
// Output dir: thesis/scaffold/pilot5/  (ledger5.jsonl, outcomes.jsonl, run_status.txt)
import { appendFileSync, writeFileSync, readFileSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { OpenRouterClient, CostCapError, Throttle, type ReasoningMode, type CostState } from './openrouter-client.js';
import { getKey } from './env.js';
import { EVIDENCE_BLOCKS } from './evidence.js';
import { runLadder, structureRecords, STRUCTURES, COMBINER_ROLES, type LadderParams } from './structures5.js';
import { summarize, localTokens, renderSegments, measureTemplateOverhead } from './ledger5.js';
import { workerPrompt, aggregatorPrompt, debateRevisePrompt, type EvidencePack } from './prompts.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUTDIR = resolve(HERE, '..', '..', 'pilot5');
mkdirSync(OUTDIR, { recursive: true });
const LEDGER_PATH = resolve(OUTDIR, 'ledger5.jsonl');
const OUTCOMES_PATH = resolve(OUTDIR, 'outcomes.jsonl');
const RAW_PATH = resolve(OUTDIR, 'runs_raw.jsonl');
const STATUS_PATH = resolve(OUTDIR, 'run_status.txt');
const OVERHEAD_PATH = resolve(OUTDIR, 'template_overhead.json');   // C2 / D18 calibration cache

const MODE = (process.argv[2] ?? 'estimate').toLowerCase();
const CAP_USD = Number(process.env.PILOT5_CAP_USD ?? '2.0');   // hard stop; confirm with the user before raising
// Event-level parallelism. Each (model, event) pair is an independent ladder run,
// so they can proceed concurrently; the per-model Throttle keeps the request rate
// safe. Wall-clock, not money, is the binding constraint on the real run.
const CONCURRENCY = Number(process.env.PILOT5_CONCURRENCY ?? '6');
// Repeat the whole event set N times, to get sample size on rare failures.
const REPEATS = Math.max(1, Number(process.env.PILOT5_REPEATS ?? '1'));

interface ModelSpec {
  key: string; slug: string; provider: string; mode: ReasoningMode;
  arm: 'reasoning' | 'non-reasoning';
  evidence: 'all' | 'first';
  /** $/M tokens in → out, from models_verified.md (2026-07-24). */
  priceIn: number; priceOut: number;
  /** Minimum spacing between request starts (ms). See the 429 note on MODELS. */
  minIntervalMs?: number;
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
  { key: 'qwen3-32b@on',          slug: 'qwen/qwen3-32b',                 provider: 'DeepInfra', mode: 'reason_on', arm: 'reasoning',     evidence: 'all', priceIn: 0.08, priceOut: 0.28, minIntervalMs: 300 },
  // minIntervalMs: the 2026-08-26 pilot lost 7 calls to upstream 429s on this
  // model — the ladder bursts N concurrent workers at a shared, rate-limited
  // endpoint. Spacing the starts turns the burst into a queue.
  { key: 'deepseek-chat-v3-0324', slug: 'deepseek/deepseek-chat-v3-0324', provider: 'DeepInfra', mode: 'none',      arm: 'non-reasoning', evidence: 'all', priceIn: 0.24, priceOut: 0.90, minIntervalMs: 3000 },
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

export const PARAMS: LadderParams = {
  agentCount: 3,
  debateRounds: 2,        // round 0 (= workers) + 2 revisions
  temperature: 0.7,
  maxTokensPerCall: 8000,   // r1 traces run 1.5-3k hidden tokens; leave headroom so truncation never causes "malformed"
  maxParseAttempts: 2,    // D16: pre-registered MINIMUM (1 retry); may be raised, every attempt logged
  aggregate: 'mean',      // B5 CLOSED (D19): expose mandates averaging; Brier convexity guarantees mean >= typical individual
};

function status(line: string): void {
  appendFileSync(STATUS_PATH, line + '\n');
  console.log(line);
}

// Evidence source. By default the synthetic pilot blocks; with PILOT5_PACKS set,
// the real level-1 / level-2 packs exported from thesis.db by
// scripts/20_build_evidence_packs.py. PILOT5_LEVEL picks the tier (1 or 2).
const PACKS_PATH = process.env.PILOT5_PACKS ? resolve(OUTDIR, process.env.PILOT5_PACKS) : null;
const LEVEL = (process.env.PILOT5_LEVEL ?? '1') === '2' ? 2 : 1;

interface RealPack {
  id: string; symbol: string; reportDate: string; t0: string;
  label: number; abnormalReturn: number; level1: string; level2: string;
}

const events: EvidencePack[] = PACKS_PATH
  ? (JSON.parse(readFileSync(PACKS_PATH, 'utf-8')) as RealPack[]).map(p => ({
      // The label and the realised return are NEVER put into an evidence pack:
      // they live only in the outcome file, so no model can read its own answer.
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
  for (const m of MODELS) {
    const evs = m.evidence === 'all' ? events : events.slice(0, 1);
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

// ---------------------------------------------------------------------- run
async function run(modelKeys: string[], evLimit: number | null): Promise<void> {
  const apiKey = getKey();
  const cost: CostState = { spentUsd: 0, capUsd: CAP_USD };
  const overheadCache: Record<string, number> = existsSync(OVERHEAD_PATH)
    ? JSON.parse(readFileSync(OVERHEAD_PATH, 'utf-8'))
    : {};
  writeFileSync(STATUS_PATH, `${MODE.toUpperCase()} start ${new Date().toISOString()} cap=$${CAP_USD} level=${LEVEL} evidence=${PACKS_PATH ? 'REAL' : 'synthetic'}\n`);
  if (!existsSync(LEDGER_PATH) || MODE === 'full') { writeFileSync(LEDGER_PATH, ''); writeFileSync(OUTCOMES_PATH, ''); writeFileSync(RAW_PATH, ''); }

  const roster = MODELS.filter(m => modelKeys.includes(m.key));

  // ---- one shared throttle per MODEL. Concurrent events each get their own
  // client (so per-event logging stays correct) but must share the rate limit,
  // or six parallel events would each get their own budget and burst again.
  const throttles = new Map<string, Throttle>(
    roster.map(m => [m.key, new Throttle(m.minIntervalMs ?? 0)]),
  );
  const mkClient = (m: ModelSpec) => new OpenRouterClient({
    modelKey: m.key, modelSlug: m.slug, provider: m.provider, mode: m.mode,
    apiKey, logPath: RAW_PATH, cost, throttle: throttles.get(m.key),
  });

  // ---- calibration first, sequentially: one cheap probe per model (D18).
  const paramsByModel = new Map<string, LadderParams>();
  for (const m of roster) {
    let overhead = overheadCache[m.key];
    if (overhead === undefined) {
      try {
        const probe = await measureTemplateOverhead(mkClient(m));
        overhead = probe.overhead;
        overheadCache[m.key] = overhead;
        writeFileSync(OVERHEAD_PATH, JSON.stringify(overheadCache, null, 2));
        status(`[calib] ${m.key}: template overhead ${overhead} tok/call (provider ${probe.providerPrompt} vs local ${probe.localCount})`);
      } catch (e: any) {
        overhead = 0;
        status(`[calib] ${m.key}: probe FAILED (${e?.message}); falling back to purely proportional allocation`);
      }
    }
    paramsByModel.set(m.key, { ...PARAMS, templateOverhead: overhead });
  }

  // ---- the work list: every (model, event) pair is an independent ladder run.
  // PILOT5_REPEATS repeats the whole event set. Format failures are rare (the one
  // observed was 1 in 6 aggregator calls and did not reproduce), so a single pass
  // over three events cannot distinguish a fixed prompt from a lucky draw.
  // Repeats buy sample size cheaply; repeated events carry an "#rN" id suffix.
  const tasks = roster.flatMap(m =>
    Array.from({ length: REPEATS }, (_, rep) =>
      (m.evidence === 'all' ? events : events.slice(0, 1))
        .slice(0, evLimit ?? undefined)
        .map(ev => ({ m, ev: rep === 0 ? ev : { ...ev, id: `${ev.id}#r${rep + 1}` } })),
    ).flat(),
  );
  status(`${tasks.length} ladder runs over ${roster.length} models, concurrency ${CONCURRENCY}`);

  let done = 0, dropped = 0, aborted = false;
  await pool(tasks, CONCURRENCY, async ({ m, ev }) => {
    if (aborted) return;
    if (cost.spentUsd >= CAP_USD) {
      if (!aborted) { aborted = true; status(`!! cost cap $${CAP_USD} reached; stopping`); }
      return;
    }
    // Fresh client per task: beginStructureRun sets per-run logging state, which
    // concurrent tasks would otherwise clobber. The throttle is still shared.
    const client = mkClient(m);
    client.beginStructureRun('ladder', ev.id);
    const t0 = Date.now();
    try {
      const res = await runLadder(client, ev, m.key, paramsByModel.get(m.key)!);
      // appendFileSync is synchronous, so concurrent writers cannot interleave a line.
      for (const r of res.ledger.records) appendFileSync(LEDGER_PATH, JSON.stringify(r) + '\n');
      appendFileSync(OUTCOMES_PATH, JSON.stringify({ modelKey: m.key, eventId: ev.id, outcomes: res.outcomes, roundProbabilities: res.roundProbabilities, matched: res.matched, repetitionRuns: res.repetitionRuns }) + '\n');
      const line = STRUCTURES.map(s => {
        const o = res.outcomes[s];
        const t = summarize(structureRecords(res.ledger.records, s), COMBINER_ROLES);
        if (o.dropped) dropped++;
        const mm = res.matched.find(x => x.structure === s);
        const vs = mm && mm.probability !== null ? ` vs rep@${mm.runsUsed}run=${mm.probability.toFixed(2)}` : '';
        return `${s}=${o.dropped ? 'DROP' : o.probability!.toFixed(2)}(comm ${t.commCost})${vs}`;
      }).join(' ');
      done++;
      status(`[${MODE}] (${done}/${tasks.length}) ${m.key} | ${ev.id} | ${line} | ${((Date.now() - t0) / 1000).toFixed(0)}s | $${cost.spentUsd.toFixed(4)}`);
    } catch (e: any) {
      if (e instanceof CostCapError) { aborted = true; status(`!! ${e.message}`); return; }
      status(`[${MODE}] ${m.key} | ${ev.id} | FAIL ${e?.message}`);
    }
  });
  status(`${MODE.toUpperCase()} done. ${done}/${tasks.length} ladder runs, ${dropped} structure drops, total=$${cost.spentUsd.toFixed(4)}`);
}

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
  if (MODE === 'probe') { await run([process.env.PILOT5_PROBE_MODEL ?? 'qwen3-32b@on'], 1); return; }
  if (MODE === 'full') { await run(MODELS.map(m => m.key), null); return; }
  throw new Error(`unknown mode ${MODE}`);
}

main().catch(e => { status(`FATAL: ${e?.stack ?? e}`); process.exit(1); });
