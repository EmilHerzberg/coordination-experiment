// Builds pilot5/ledger_report_v2.md from pilot5/ledger5.jsonl + outcomes.jsonl.
//   node --experimental-transform-types --import ./register.mjs make-report5.ts
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { summarize, type CallLedgerRecord, type StructureTotals } from './ledger5.js';
import { structureRecords, STRUCTURES, COMBINER_ROLES, type StructureName } from './structures5.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUTDIR = resolve(HERE, '..', '..', 'pilot5');
const LEDGER_PATH = resolve(OUTDIR, 'ledger5.jsonl');
const OUTCOMES_PATH = resolve(OUTDIR, 'outcomes.jsonl');
const REPORT_PATH = resolve(OUTDIR, 'ledger_report_v2.md');

function readJsonl<T>(p: string): T[] {
  if (!existsSync(p)) return [];
  return readFileSync(p, 'utf-8').split(/\r?\n/).filter(Boolean).map(l => JSON.parse(l) as T);
}

interface OutcomeLine { modelKey: string; eventId: string; outcomes: Record<StructureName, { probability: number | null; dropped: boolean; dropReason?: string }> }

export function buildReport(records: CallLedgerRecord[], outcomes: OutcomeLine[]): string {
  const models = [...new Set(records.map(r => r.modelKey))];
  const L: string[] = [];
  L.push('# ledger_report_v2.md — five-bucket token accounting per structure × model');
  L.push('');
  L.push('_Buckets: D evidence+base, A coordination instruction, B received messages (per read), B_SELF own prior re-read (debate), C1 final answer, C2 everything else written (incl. hidden reasoning). Communication cost = A + B + whole combiner call(s); B_SELF shown separately (open item). Absolute tokens, not shares. Malformed attempts excluded from buckets, included in $._');
  L.push('');
  L.push(`Records: ${records.length} (ok ${records.filter(r => r.ok).length}, malformed attempts ${records.filter(r => r.malformed).length}). Total cost: $${records.reduce((a, r) => a + r.costUsd, 0).toFixed(4)}.`);
  L.push('');
  L.push('| model | structure | calls | D | A | B | B_SELF | C1 | C2 (hidden) | total | **comm cost** | comm incl. self | malformed | wasted | max gap read/write | $ |');
  L.push('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|');
  const perModelStruct = new Map<string, StructureTotals>();
  for (const m of models) {
    const mine = records.filter(r => r.modelKey === m);
    for (const s of STRUCTURES) {
      const t = summarize(structureRecords(mine, s), COMBINER_ROLES);
      perModelStruct.set(`${m}|${s}`, t);
      L.push(`| ${m} | ${s} | ${t.calls} | ${t.D} | ${t.A} | ${t.B} | ${t.B_SELF} | ${t.C1} | ${t.C2} (${t.C2_hidden}) | ${t.total} | **${t.commCost}** | ${t.commCostInclSelf} | ${t.malformedAttempts} | ${t.wastedTokens} | ${(t.maxReadGap * 100).toFixed(1)}% / ${(t.maxWriteGap * 100).toFixed(1)}% | ${t.costUsd.toFixed(4)} |`);
    }
  }
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
      const bLocal = r.segments.filter(s => s.bucket === 'B').reduce((a, s) => a + s.localTokens, 0);
      return bLocal > 0 && Math.abs(r.read.B - bLocal * (r.read.providerPrompt / Math.max(r.read.localSum, 1))) <= 2;
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
  L.push('## Malformed attempts');
  L.push('');
  const bad = records.filter(r => r.malformed);
  if (!bad.length) L.push('none');
  else {
    L.push('| model | event | role | attempt | error | tokens |');
    L.push('|---|---|---|---:|---|---:|');
    for (const r of bad) L.push(`| ${r.modelKey} | ${r.eventId} | ${r.agentRole} | ${r.attempt} | ${r.parseError} | ${r.totalTokens} |`);
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
