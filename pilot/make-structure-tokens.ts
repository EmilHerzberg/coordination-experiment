// Re-shapes pilot5/ledger5.jsonl into one line per (event x model x structure),
// so the Python analysis can join token cost onto the outcomes without ever
// re-implementing the accounting rules. Run:
//   node --experimental-transform-types --import ./register.mjs make-structure-tokens.ts \
//        [--ledger <path>] [--out <path>]
//
// WHY THIS FILE IS TYPESCRIPT AND NOT PYTHON. Every figure below comes from
// `summarize(structureRecords(...), COMBINER_ROLES)` — the same two functions
// `make-report5.ts` calls. Nothing here adds, subtracts or re-derives a token.
// The rule for what counts as communication cost (A + B for an ordinary agent,
// the WHOLE call for a combiner, working doc §5.2) was audited on 2026-08-29
// and exists in exactly one place; a Python re-implementation would be a second
// place that could drift from it silently while still printing plausible
// numbers. This file is a projection, not a calculation.
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { summarize, type CallLedgerRecord } from './ledger5.js';
import { structureRecords, STRUCTURES, COMBINER_ROLES, type StructureName } from './structures5.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUTDIR = process.env.PILOT5_OUTDIR ? resolve(process.env.PILOT5_OUTDIR) : resolve(HERE, '..', '..', 'pilot5');
// Level-prefixed, matching what `run-pilot5.ts` writes. See buildMeta below for
// why an unprefixed default was actively dangerous here.
const LEVEL = (process.env.PILOT5_LEVEL ?? '1') === '2' ? 2 : 1;
const LEDGER_PATH = resolve(OUTDIR, `L${LEVEL}_ledger5.jsonl`);
const TOKENS_PATH = resolve(OUTDIR, `L${LEVEL}_structure_tokens.jsonl`);

function readJsonl<T>(p: string): T[] {
  if (!existsSync(p)) return [];
  return readFileSync(p, 'utf-8').split(/\r?\n/).filter(Boolean).map(l => JSON.parse(l) as T);
}

/** One emitted data line. The key list is fixed: the Python side reads by name. */
export interface StructureTokenLine {
  eventId: string;
  modelKey: string;
  structure: StructureName;
  D: number; A: number; B: number; B_SELF: number; C1: number; C2: number; C2_hidden: number;
  commCost: number; commCostInclSelf: number;
  billedCommCost: number; billedCommCostInclSelf: number;
  total: number; billedTotal: number;
  prompt: number; completion: number;
  calls: number; malformedAttempts: number;
  costUsd: number;
}

/** The provenance header written as line 1. See buildMeta for why each field is here. */
export interface StructureTokenMeta {
  ledgerPath: string;
  ledgerLines: number;
  ledgerSha256: string;
  harnessCommit: string | null;
  generatedAt: string;
}

/**
 * One line per (event, model, structure), in the order the calls were made.
 *
 * Takes records rather than a path so a test can drive it from a mock run with
 * no files involved.
 */
export function buildStructureTokens(records: CallLedgerRecord[]): StructureTokenLine[] {
  // First-appearance order, never sorted. The ledger is written in generation
  // order, and the rest of the pipeline (repetitionRuns, the compute-matching
  // prefix rule) depends on that same order not being rearranged after the
  // fact, so this file does not introduce a different one.
  const pairs: { eventId: string; modelKey: string }[] = [];
  const seen = new Set<string>();
  for (const r of records) {
    const key = `${r.eventId}|${r.modelKey}`;
    if (seen.has(key)) continue;
    seen.add(key);
    pairs.push({ eventId: r.eventId, modelKey: r.modelKey });
  }

  const lines: StructureTokenLine[] = [];
  for (const { eventId, modelKey } of pairs) {
    // Narrow to ONE event and ONE model before `structureRecords` sees the
    // records. `structureRecords` filters on the structure field alone, by
    // design: `make-report5.ts` hands it one model's whole ledger and wants the
    // model-level total. Handing it the raw file here would pool every event's
    // shared workers into every line — the totals would still be internally
    // consistent and would still add up, which is exactly the kind of wrong
    // number this project keeps finding.
    const mine = records.filter(r => r.eventId === eventId && r.modelKey === modelKey);
    for (const s of STRUCTURES) {
      const t = summarize(structureRecords(mine, s), COMBINER_ROLES);
      // All three structures are emitted for every (event, model) even when one
      // was dropped, so a missing line means a missing EVENT and never a
      // quietly absent structure. A dropped structure shows up as its own
      // malformedAttempts > 0, which the Python side can act on.
      lines.push({
        eventId, modelKey, structure: s,
        D: t.D, A: t.A, B: t.B, B_SELF: t.B_SELF, C1: t.C1, C2: t.C2, C2_hidden: t.C2_hidden,
        commCost: t.commCost, commCostInclSelf: t.commCostInclSelf,
        billedCommCost: t.billedCommCost, billedCommCostInclSelf: t.billedCommCostInclSelf,
        // `total` covers usable calls only; `billedTotal` adds the attempts that
        // were discarded and retried. Compute matching in `structures5.ts` uses
        // the billed figure, because a structure that fails more often would
        // otherwise look cheaper than it was. Both are emitted so the Python
        // side never has to guess which one a column means. They are equal
        // whenever nothing was retried, which is the case for both pilot runs
        // (L1 and L2 contain 0 malformed attempts).
        total: t.total, billedTotal: t.billedTotal,
        prompt: t.prompt, completion: t.completion,
        calls: t.calls, malformedAttempts: t.malformedAttempts,
        costUsd: t.costUsd,
      });
    }
  }
  return lines;
}

/**
 * The provenance header.
 *
 * It exists so the Python side can refuse a summary belonging to a different
 * ledger. Until 2026-09-01 the harness wrote fixed filenames (`ledger5.jsonl`)
 * and the runs were archived afterwards, BY HAND, as `L1_ledger5.jsonl` /
 * `L2_ledger5.jsonl` — so a stale `structure_tokens.jsonl` sitting next to a
 * freshly re-run ledger looked exactly like a current one, and the hash was the
 * only thing that told them apart. The runner now writes the level prefix from
 * the start and regenerates this file itself at the end of every run, which
 * removes the way that mismatch was created; the hash stays, because it is the
 * only check that survives a file being copied or re-run out of band.
 */
export function buildMeta(ledgerPath: string, records: CallLedgerRecord[]): StructureTokenMeta {
  // Hash the raw BYTES, not the parsed text. These files are CRLF, and reading
  // them as utf-8 first would let a line-ending change slip past unnoticed —
  // the hash has to answer "is this the same file", not "does it mean the same".
  const sha = createHash('sha256').update(readFileSync(ledgerPath)).digest('hex');
  let harnessCommit: string | null = null;
  try {
    harnessCommit = execFileSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: HERE, encoding: 'utf-8' }).trim();
  } catch {
    // No git, a detached checkout, an archive download: write null. Inventing a
    // placeholder here would put an unverifiable string in the one field whose
    // whole job is to be verifiable.
    harnessCommit = null;
  }
  return {
    ledgerPath,
    ledgerLines: records.length,
    ledgerSha256: sha,
    harnessCommit,
    generatedAt: new Date().toISOString(),
  };
}

const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const argv = process.argv.slice(2);
  const arg = (flag: string) => {
    const i = argv.indexOf(flag);
    if (i < 0) return undefined;
    const v = argv[i + 1];
    // A flag with nothing after it used to fall through to the default path and
    // silently overwrite the wrong file, so it is an error rather than a shrug.
    if (v === undefined || v.startsWith('--')) { console.error(`${flag} needs a path`); process.exit(1); }
    return v;
  };
  const ledgerPath = resolve(arg('--ledger') ?? LEDGER_PATH);
  const outPath = resolve(arg('--out') ?? TOKENS_PATH);

  const records = readJsonl<CallLedgerRecord>(ledgerPath);
  if (!records.length) { console.error(`no records in ${ledgerPath}`); process.exit(1); }
  const meta = buildMeta(ledgerPath, records);
  const lines = buildStructureTokens(records);
  writeFileSync(outPath, [JSON.stringify({ _meta: meta }), ...lines.map(l => JSON.stringify(l))].join('\n') + '\n');
  console.log(`wrote ${outPath}: ${lines.length} rows from ${records.length} ledger lines (sha ${meta.ledgerSha256.slice(0, 12)})`);
}
