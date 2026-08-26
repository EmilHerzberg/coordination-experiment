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
import { appendFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { OpenRouterClient, CostCapError, type ReasoningMode, type CostState } from './openrouter-client.js';
import { getKey } from './env.js';
import { EVIDENCE_BLOCKS } from './evidence.js';
import { runLadder, structureRecords, STRUCTURES, COMBINER_ROLES, type LadderParams } from './structures5.js';
import { summarize, localTokens, renderSegments } from './ledger5.js';
import { workerPrompt, aggregatorPrompt, debateRevisePrompt, type EvidencePack } from './prompts-draft.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUTDIR = resolve(HERE, '..', '..', 'pilot5');
mkdirSync(OUTDIR, { recursive: true });
const LEDGER_PATH = resolve(OUTDIR, 'ledger5.jsonl');
const OUTCOMES_PATH = resolve(OUTDIR, 'outcomes.jsonl');
const RAW_PATH = resolve(OUTDIR, 'runs_raw.jsonl');
const STATUS_PATH = resolve(OUTDIR, 'run_status.txt');

const MODE = (process.argv[2] ?? 'estimate').toLowerCase();
const CAP_USD = Number(process.env.PILOT5_CAP_USD ?? '2.0');   // hard stop; confirm with the user before raising

interface ModelSpec {
  key: string; slug: string; provider: string; mode: ReasoningMode;
  arm: 'reasoning' | 'non-reasoning';
  evidence: 'all' | 'first';
  /** $/M tokens in → out, from models_verified.md (2026-07-24). */
  priceIn: number; priceOut: number;
}

// Same roster as pilot v1 (models_verified.md). Prices are for the estimate only;
// real cost comes from the provider's usage.cost.
export const MODELS: ModelSpec[] = [
  { key: 'qwen3-32b@on',          slug: 'qwen/qwen3-32b',                     provider: 'SiliconFlow', mode: 'reason_on',  arm: 'reasoning',     evidence: 'all',   priceIn: 0.14, priceOut: 0.57 },
  { key: 'qwen3-32b@off',         slug: 'qwen/qwen3-32b',                     provider: 'SiliconFlow', mode: 'reason_off', arm: 'non-reasoning', evidence: 'all',   priceIn: 0.14, priceOut: 0.57 },
  { key: 'llama-3.3-70b',         slug: 'meta-llama/llama-3.3-70b-instruct',  provider: 'Crusoe',      mode: 'none',       arm: 'non-reasoning', evidence: 'all',   priceIn: 0.25, priceOut: 0.75 },
  { key: 'deepseek-r1-0528',      slug: 'deepseek/deepseek-r1-0528',          provider: 'DeepInfra',   mode: 'reason_on',  arm: 'reasoning',     evidence: 'first', priceIn: 0.50, priceOut: 2.15 },
  { key: 'deepseek-chat-v3-0324', slug: 'deepseek/deepseek-chat-v3-0324',     provider: 'Crusoe',      mode: 'none',       arm: 'non-reasoning', evidence: 'first', priceIn: 0.50, priceOut: 1.50 },
];

export const PARAMS: LadderParams = {
  agentCount: 3,
  debateRounds: 2,        // round 0 (= workers) + 2 revisions
  temperature: 0.7,
  maxTokensPerCall: 8000,   // r1 traces run 1.5-3k hidden tokens; leave headroom so truncation never causes "malformed"
  aggregate: 'mean',      // [SESSION]
};

function status(line: string): void {
  appendFileSync(STATUS_PATH, line + '\n');
  console.log(line);
}

const events: EvidencePack[] = EVIDENCE_BLOCKS.map(b => ({ id: b.id, text: b.description }));

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
  writeFileSync(STATUS_PATH, `${MODE.toUpperCase()} start ${new Date().toISOString()} cap=$${CAP_USD}\n`);
  if (!existsSync(LEDGER_PATH) || MODE === 'full') { writeFileSync(LEDGER_PATH, ''); writeFileSync(OUTCOMES_PATH, ''); writeFileSync(RAW_PATH, ''); }

  outer: for (const m of MODELS.filter(m => modelKeys.includes(m.key))) {
    const client = new OpenRouterClient({
      modelKey: m.key, modelSlug: m.slug, provider: m.provider, mode: m.mode,
      apiKey, logPath: RAW_PATH, cost,
    });
    const evs = (m.evidence === 'all' ? events : events.slice(0, 1)).slice(0, evLimit ?? undefined);
    for (const ev of evs) {
      if (cost.spentUsd >= CAP_USD) { status(`!! cost cap $${CAP_USD} reached; stopping`); break outer; }
      client.beginStructureRun('ladder', ev.id);
      const t0 = Date.now();
      try {
        const res = await runLadder(client, ev, m.key, PARAMS);
        for (const r of res.ledger.records) appendFileSync(LEDGER_PATH, JSON.stringify(r) + '\n');
        appendFileSync(OUTCOMES_PATH, JSON.stringify({ modelKey: m.key, eventId: ev.id, outcomes: res.outcomes, roundProbabilities: res.roundProbabilities }) + '\n');
        const line = STRUCTURES.map(s => {
          const o = res.outcomes[s];
          const t = summarize(structureRecords(res.ledger.records, s), COMBINER_ROLES);
          return `${s}=${o.dropped ? 'DROP' : o.probability!.toFixed(2)}(comm ${t.commCost})`;
        }).join(' ');
        status(`[${MODE}] ${m.key} | ${ev.id} | ${line} | ${((Date.now() - t0) / 1000).toFixed(0)}s | $${cost.spentUsd.toFixed(4)}`);
      } catch (e: any) {
        if (e instanceof CostCapError) { status(`!! ${e.message}`); break outer; }
        status(`[${MODE}] ${m.key} | ${ev.id} | FAIL ${e?.message}`);
      }
    }
  }
  status(`${MODE.toUpperCase()} done. total=$${cost.spentUsd.toFixed(4)}`);
}

async function main(): Promise<void> {
  if (MODE === 'estimate') { estimate(); return; }
  if (MODE === 'probe') { await run([process.env.PILOT5_PROBE_MODEL ?? 'qwen3-32b@off'], 1); return; }
  if (MODE === 'full') { await run(MODELS.map(m => m.key), null); return; }
  throw new Error(`unknown mode ${MODE}`);
}

main().catch(e => { status(`FATAL: ${e?.stack ?? e}`); process.exit(1); });
