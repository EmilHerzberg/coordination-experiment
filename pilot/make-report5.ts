// Builds pilot5/ledger_report_v2.md from pilot5/ledger5.jsonl + outcomes.jsonl.
//   node --experimental-transform-types --import ./register.mjs make-report5.ts
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { summarize, localTokens, type CallLedgerRecord, type StructureTotals } from './ledger5.js';
import { structureRecords, STRUCTURES, COMBINER_ROLES, spread, sd, type StructureName } from './structures5.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUTDIR = resolve(HERE, '..', '..', 'pilot5');
const LEDGER_PATH = resolve(OUTDIR, 'ledger5.jsonl');
const OUTCOMES_PATH = resolve(OUTDIR, 'outcomes.jsonl');
const REPORT_PATH = resolve(OUTDIR, 'ledger_report_v2.md');

function readJsonl<T>(p: string): T[] {
  if (!existsSync(p)) return [];
  return readFileSync(p, 'utf-8').split(/\r?\n/).filter(Boolean).map(l => JSON.parse(l) as T);
}

interface OutcomeLine {
  modelKey: string;
  eventId: string;
  outcomes: Record<StructureName, { probability: number | null; dropped: boolean; dropReason?: string }>;
  /** [round0(workers), round1, ...] — D14 spread/collapse diagnostic. */
  roundProbabilities?: number[][];
  /** D23 compute matching: one entry per communicating structure. */
  matched?: { structure: StructureName; structureTokens: number; runsUsed: number;
              matchedTokens: number; residualTokens: number;
              probability: number | null; structureProbability: number | null }[];
  repetitionRuns?: { probability: number; totalTokens: number }[];
}

export function buildReport(records: CallLedgerRecord[], outcomes: OutcomeLine[]): string {
  const models = [...new Set(records.map(r => r.modelKey))];
  const L: string[] = [];
  L.push('# ledger_report_v2.md — five-bucket token accounting per structure × model');
  L.push('');
  L.push('_Buckets: D evidence+base, A coordination instruction, B received messages (per read), B_SELF own prior re-read (debate), C1 final answer, C2 everything else written (incl. hidden reasoning). Communication cost = A + B + whole combiner call(s); B_SELF shown separately (open item). Absolute tokens, not shares. Malformed attempts excluded from buckets, included in $._');
  L.push('');
  L.push(`Records: ${records.length} (ok ${records.filter(r => r.ok).length}, malformed attempts ${records.filter(r => r.malformed).length}). Total cost: $${records.reduce((a, r) => a + r.costUsd, 0).toFixed(4)}.`);
  L.push('');
  L.push('| model | structure | calls | D | A | B | B_SELF | C1 | C2 (hidden) | total | **comm cost** | comm incl. self | malformed | wasted | read gap | visible discrep. (tok) | $ |');
  L.push('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|');
  const perModelStruct = new Map<string, StructureTotals>();
  for (const m of models) {
    const mine = records.filter(r => r.modelKey === m);
    for (const s of STRUCTURES) {
      const t = summarize(structureRecords(mine, s), COMBINER_ROLES);
      perModelStruct.set(`${m}|${s}`, t);
      L.push(`| ${m} | ${s} | ${t.calls} | ${t.D} | ${t.A} | ${t.B} | ${t.B_SELF} | ${t.C1} | ${t.C2} (${t.C2_hidden}) | ${t.total} | **${t.commCost}** | ${t.commCostInclSelf} | ${t.malformedAttempts} | ${t.wastedTokens} | ${(t.maxReadGap * 100).toFixed(1)}% | ${t.maxVisibleDiscrepancy} | ${t.costUsd.toFixed(4)} |`);
    }
  }
  L.push('');
  L.push('## Chat-template overhead (C2 / D18)');
  L.push('');
  L.push('_Fixed per-call tokens the provider bills that our local tokenizer never sees. Measured once per model with a minimal two-message probe and booked entirely to D, because a lone forecaster pays exactly the same amount. The residual column is the length-proportional tokenizer mismatch, which alone is spread across the buckets._');
  L.push('');
  L.push('| model | measured overhead (tok/call) | max read residual | max visible discrepancy (tok) |');
  L.push('|---|---:|---:|---:|');
  for (const m of models) {
    const mine = records.filter(r => r.modelKey === m && r.ok);
    const oh = mine.length ? mine[0].read.templateOverhead : 0;
    const maxRes = mine.reduce((a, r) => Math.max(a, Math.abs(r.read.gapShare)), 0);
    const maxVis = mine.reduce((a, r) => Math.max(a, Math.abs(r.write.visibleDiscrepancy ?? 0)), 0);
    L.push(`| ${m} | ${oh} | ${(maxRes * 100).toFixed(2)}% | ${maxVis} |`);
  }
  L.push('');
  L.push("_**Visible discrepancy** is `localVisible - (completion_tokens - reasoning_tokens)`. It mixes two tokenizers, so read it as a reconciliation diagnostic rather than an error: our cl100k_base count and the model's own tokenizer differ by roughly 5-13 % on Qwen text. See the reconciliation table below for the part that survives that correction._");
  L.push('');
  L.push('### Reconciliation of the provider figures (working doc §5.4)');
  L.push('');
  L.push('| model | local reasoning / provider reasoning | local total / completion | implied visible vs provider budget |');
  L.push('|---|---:|---:|---:|');
  for (const m of models) {
    const mine = records.filter(r => r.modelKey === m && r.ok);
    const lr = mine.reduce((a, r) => a + localTokens(r.reasoningText || ''), 0);
    const pr = mine.reduce((a, r) => a + r.write.providerReasoning, 0);
    const lv = mine.reduce((a, r) => a + localTokens(r.responseText || ''), 0);
    const pc = mine.reduce((a, r) => a + r.write.providerCompletion, 0);
    const budget = pc - pr;
    const scale = pc > 0 ? (lr + lv) / pc : 1;          // local tokens per provider token
    const impliedVisible = scale > 0 ? lv / scale : 0;   // our visible text in provider units
    L.push(`| ${m} | ${pr > 0 ? (lr / pr).toFixed(3) : 'n/a'} | ${(scale).toFixed(3)} | ${budget > 0 ? (budget / Math.max(impliedVisible, 1)).toFixed(3) : 'n/a'} |`);
  }
  L.push('');
  L.push("_Third column below 1.00 means the provider's `completion_tokens` minus `reasoning_tokens` leaves less room than the visible answer actually occupies, even after rescaling into the provider's own units. Since D22 the hidden/visible split is NOT taken from this subtraction: both texts are counted with the same local tokenizer and rescaled to `completion_tokens`, so the instrument bias cancels per call. Cross-checking against the prompt (text we wrote, so known exactly) puts that bias at ~3.3 % on Qwen, which implies the returned trace is ~91 % of what was billed — the hidden share is therefore likely understated by ~9 %, disclosed rather than corrected. **Nothing the thesis reports depends on this:** the read side is exact shares of `prompt_tokens`, compute matching uses `total_tokens`, and C1 + C2 = `completion_tokens` holds exactly. Only the hidden/visible split within C2 is approximate, and it feeds no research question._");
  L.push('');
  L.push('## Sanity checks (NEXT_STEP_EXPLAINED.md §7)');
  L.push('');
  L.push('| model | repetition A=B=0 | aggregator B = Σ worker reasons (local) | debate B grows per round | identities hold on every call |');
  L.push('|---|---|---|---|---|');
  for (const m of models) {
    const mine = records.filter(r => r.modelKey === m && r.ok);
    const rep = perModelStruct.get(`${m}|repetition`)!;
    const repOk = rep.A === 0 && rep.B === 0;
    // aggregator: its B should equal the local size of the reasons it was shown (segments origin worker:*:reason)
    const aggCalls = mine.filter(r => r.agentRole === 'aggregator');
    const aggOk = aggCalls.every(r => {
      // Only the residual (providerPrompt - fixed overhead) is spread proportionally (C2/D18).
      const bLocal = r.segments.filter(s => s.bucket === 'B').reduce((a, s) => a + s.localTokens, 0);
      const scale = (r.read.providerPrompt - r.read.templateOverhead) / Math.max(r.read.localSum, 1);
      return bLocal > 0 && Math.abs(r.read.B - bLocal * scale) <= 2;
    });
    const rounds = [...new Set(mine.filter(r => r.structure === 'debate').map(r => r.round))].sort((a, b) => (a ?? 0) - (b ?? 0));
    const bByRound = rounds.map(rd => mine.filter(r => r.structure === 'debate' && r.round === rd).reduce((a, r) => a + r.read.B, 0));
    const debOk = bByRound.every(b => b > 0);
    const idOk = mine.every(r => r.read.D + r.read.A + r.read.B + r.read.B_SELF === r.read.providerPrompt && r.write.C1 + r.write.C2 === r.write.providerCompletion);
    L.push(`| ${m} | ${repOk ? 'ok' : 'FAIL'} | ${aggCalls.length ? (aggOk ? 'ok' : 'FAIL') : 'n/a'} | ${bByRound.length ? bByRound.join(' → ') : 'n/a'} ${debOk ? '' : 'FAIL'} | ${idOk ? 'ok' : 'FAIL'} |`);
  }
  L.push('');
  L.push('## Communication cost, absolute tokens (the number compute-matching needs)');
  L.push('');
  L.push('| model | repetition | aggregator | debate | (debate incl. B_SELF) |');
  L.push('|---|---:|---:|---:|---:|');
  for (const m of models) {
    const g = (s: StructureName) => perModelStruct.get(`${m}|${s}`)!;
    L.push(`| ${m} | ${g('repetition').commCost} | ${g('aggregator').commCost} | ${g('debate').commCost} | ${g('debate').commCostInclSelf} |`);
  }
  L.push('');
  L.push('## Outcomes per event');
  L.push('');
  L.push('| model | event | repetition | aggregator | debate |');
  L.push('|---|---|---:|---:|---:|');
  for (const o of outcomes) {
    const f = (s: StructureName) => o.outcomes[s].dropped ? `DROP (${o.outcomes[s].dropReason?.slice(0, 40)})` : o.outcomes[s].probability!.toFixed(2);
    L.push(`| ${o.modelKey} | ${o.eventId} | ${f('repetition')} | ${f('aggregator')} | ${f('debate')} |`);
  }
  L.push('');
  L.push('## Compute-matched comparisons (D23) — the numbers RQ1 is computed from');
  L.push('');
  L.push("_Each communicating structure is paired with the communication-free rung given the SAME budget. Budget = total tokens (prompt + completion, as billed) that the structure spent, including the three shared workers. The baseline is drawn by repeating single-agent runs until the cumulative spend covers the largest budget; each structure is then compared against the PREFIX of those runs that fits its own budget, taken in generation order so no favourable subset can be selected afterwards. `residual` is the shortfall left by the fits-rule and is reported, never hidden (working doc §4.2)._");
  L.push('');
  const withMatched = outcomes.filter(o => o.matched && o.matched.length);
  if (!withMatched.length) L.push('no matched comparisons recorded');
  else {
    L.push('| model | structure | events | mean budget (tok) | mean runs matched | mean residual (tok) | residual % |');
    L.push('|---|---|---:|---:|---:|---:|---:|');
    for (const m of [...new Set(withMatched.map(o => o.modelKey))]) {
      for (const s of STRUCTURES) {
        const rows = withMatched.flatMap(o => (o.modelKey === m ? o.matched!.filter(x => x.structure === s) : []));
        if (!rows.length) continue;
        const mean = (f: (x: typeof rows[number]) => number) => rows.reduce((a, x) => a + f(x), 0) / rows.length;
        const res = mean(x => x.residualTokens), bud = mean(x => x.structureTokens);
        L.push(`| ${m} | ${s} | ${rows.length} | ${bud.toFixed(0)} | ${mean(x => x.runsUsed).toFixed(1)} | ${res.toFixed(0)} | ${(res / bud * 100).toFixed(1)}% |`);
      }
    }
  }
  L.push('');
  L.push('## Agent spread per debate round (D14 diagnostic)');
  L.push('');
  L.push("_Round 0 = the shared workers. Spread = max minus min of the agents' probabilities in that round; SD in brackets. Two things to read here. (a) If round-0 spread is near zero, temperature 0.7 is not producing a real ensemble: averaging buys nothing and the debate has nothing to argue about. (b) If spread shrinks across rounds, the agents converged DESPITE the explicit instruction not to converge merely to converge (D11) — the stronger of the two possible findings._");
  L.push('');
  const withRounds = outcomes.filter(o => o.roundProbabilities && o.roundProbabilities.length);
  if (!withRounds.length) {
    L.push('no round data');
  } else {
    const maxR = Math.max(...withRounds.map(o => o.roundProbabilities!.length));
    L.push('| model | events | ' + Array.from({ length: maxR }, (_, r) => `round ${r} spread (SD)`).join(' | ') + ' |');
    L.push('|---|---:|' + Array.from({ length: maxR }, () => '---:').join('|') + '|');
    for (const m of [...new Set(withRounds.map(o => o.modelKey))]) {
      const mine = withRounds.filter(o => o.modelKey === m);
      const cells: string[] = [];
      for (let r = 0; r < maxR; r++) {
        const vecs = mine.map(o => o.roundProbabilities![r]).filter(v => Array.isArray(v) && v.length > 0);
        if (!vecs.length) { cells.push('n/a'); continue; }
        const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
        cells.push(`${mean(vecs.map(spread)).toFixed(3)} (${mean(vecs.map(sd)).toFixed(3)})`);
      }
      L.push(`| ${m} | ${mine.length} | ${cells.join(' | ')} |`);
    }
  }
  L.push('');
  L.push('## Malformed attempts');
  L.push('');
  L.push("_Split by cause (D16). `truncated` means the answer hit our per-call token cap so no closing tag was written — that is the cap being too low, not the model failing to comply, and the remedy is raising the cap rather than dropping the event. `format` is the only genuine model failure. Infrastructure errors (5xx, timeouts, rate limits) never appear here; the client retries them itself._");
  L.push('');
  const bad = records.filter(r => r.malformed);
  const okRecords = records.filter(r => r.ok);
  const retried = okRecords.filter(r => r.attempt > 1);
  L.push(`Attempts: ${okRecords.length} usable calls, of which ${retried.length} needed more than one attempt (max attempt used: ${Math.max(0, ...okRecords.map(r => r.attempt))}). Failed attempts: ${bad.filter(r => r.failureKind === 'format').length} format, ${bad.filter(r => r.failureKind === 'truncated').length} truncated.`);
  L.push('');
  if (!bad.length) L.push('none');
  else {
    L.push('| model | event | role | attempt | cause | finish_reason | error | tokens |');
    L.push('|---|---|---|---:|---|---|---|---:|');
    for (const r of bad) L.push(`| ${r.modelKey} | ${r.eventId} | ${r.agentRole} | ${r.attempt} | ${r.failureKind ?? '?'} | ${r.finishReason ?? '-'} | ${r.parseError} | ${r.totalTokens} |`);
  }
  L.push('');
  return L.join('\n');
}

const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const records = readJsonl<CallLedgerRecord>(LEDGER_PATH);
  const outcomes = readJsonl<OutcomeLine>(OUTCOMES_PATH);
  if (!records.length) { console.error(`no records in ${LEDGER_PATH}`); process.exit(1); }
  writeFileSync(REPORT_PATH, buildReport(records, outcomes));
  console.log(`wrote ${REPORT_PATH}`);
}
