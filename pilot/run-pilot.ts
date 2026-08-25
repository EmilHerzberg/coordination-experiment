// Pilot entrypoint. Runs the 5 coordination structures across the pilot models
// on synthetic evidence, provider-pinned, with a hard $10 cost cap.
//   node --experimental-transform-types --import ./register.mjs run-pilot.ts probe
//   node --experimental-transform-types --import ./register.mjs run-pilot.ts full
import { appendFileSync, writeFileSync, mkdirSync } from 'node:fs';
import type { AgentTools, CoordinationConfig, CoordinationConfigParams } from '../src/types.js';
import { IndependentEnsemble } from '../src/configs/independent-ensemble.js';
import { PeerCritiqueDebate } from '../src/configs/peer-critique-debate.js';
import { OrchestratorSpecialist } from '../src/configs/orchestrator-specialist.js';
import { SingleCall, AggregatorEnsemble } from './configs.js';
import { OpenRouterClient, CostCapError, type ReasoningMode, type CostState } from './openrouter-client.js';
import { getKey } from './env.js';
import { EVIDENCE_BLOCKS, makeMarket } from './evidence.js';

const OUTDIR = 'c:\\Users\\herzb\\Documents\\Privat\\Uni\\thesis\\check reasoning tokens are countable\\pilot';
mkdirSync(OUTDIR, { recursive: true });
const RUNS_PATH = `${OUTDIR}\\runs.jsonl`;
const STATUS_PATH = `${OUTDIR}\\run_status.txt`;

const MODE = (process.argv[2] ?? 'probe').toLowerCase();
const QWEN_OFF_NOTHINK = process.env.QWEN_OFF_NOTHINK === '1';
const CAP_USD = 9.0; // hard internal stop, margin below the $10 briefing cap

interface ModelSpec {
  key: string;
  slug: string;
  provider: string;
  mode: ReasoningMode;
  arm: 'reasoning' | 'non-reasoning';
  evidence: 'all' | 'first';
  noThink?: boolean;
}

const MODELS: ModelSpec[] = [
  { key: 'qwen3-32b@on', slug: 'qwen/qwen3-32b', provider: 'SiliconFlow', mode: 'reason_on', arm: 'reasoning', evidence: 'all' },
  { key: 'qwen3-32b@off', slug: 'qwen/qwen3-32b', provider: 'SiliconFlow', mode: 'reason_off', arm: 'non-reasoning', evidence: 'all', noThink: QWEN_OFF_NOTHINK },
  { key: 'llama-3.3-70b', slug: 'meta-llama/llama-3.3-70b-instruct', provider: 'Crusoe', mode: 'none', arm: 'non-reasoning', evidence: 'all' },
  { key: 'deepseek-r1-0528', slug: 'deepseek/deepseek-r1-0528', provider: 'DeepInfra', mode: 'reason_on', arm: 'reasoning', evidence: 'first' },
  { key: 'deepseek-chat-v3-0324', slug: 'deepseek/deepseek-chat-v3-0324', provider: 'Crusoe', mode: 'none', arm: 'non-reasoning', evidence: 'first' },
];

function structures(): CoordinationConfig[] {
  return [
    new SingleCall(),
    new IndependentEnsemble(),
    new AggregatorEnsemble(),
    new PeerCritiqueDebate(),
    new OrchestratorSpecialist(),
  ];
}

const PARAMS: CoordinationConfigParams = {
  agentCount: 3,
  maxInternalRounds: 1, // debate: round 0 + 1 revision = 2 rounds total (pilot)
  convergenceTolerance: 0.05,
  maxTokensPerMarket: 20000,
  temperature: 0.7,
};

const noTools: AgentTools = {
  async getMarketDetails() { return { question: '', description: '' }; },
  async getPriceHistory() { return []; },
  async searchWeb() { return []; },
};

function status(line: string): void {
  appendFileSync(STATUS_PATH, line + '\n');
  console.log(line);
}

function makeClient(m: ModelSpec, apiKey: string, cost: CostState): OpenRouterClient {
  return new OpenRouterClient({
    modelKey: m.key, modelSlug: m.slug, provider: m.provider, mode: m.mode,
    noThinkInjection: m.noThink, apiKey, logPath: RUNS_PATH, cost,
  });
}

async function main(): Promise<void> {
  const apiKey = getKey();
  const cost: CostState = { spentUsd: 0, capUsd: CAP_USD };

  if (MODE === 'probe') {
    writeFileSync(STATUS_PATH, `PROBE start ${new Date().toISOString()}  (QWEN_OFF_NOTHINK=${QWEN_OFF_NOTHINK})\n`);
    writeFileSync(RUNS_PATH, ''); // fresh log for probe
    const single = new SingleCall();
    const market = makeMarket(0);
    for (const m of MODELS) {
      const client = makeClient(m, apiKey, cost);
      client.beginStructureRun('PROBE_single', EVIDENCE_BLOCKS[0].id);
      try {
        const res = await single.predict({ market, tools: noTools, llm: client, params: PARAMS });
        status(`[probe] ${m.key} (${m.provider}) -> prob=${res.probability.toFixed(3)} costSoFar=$${cost.spentUsd.toFixed(4)}`);
      } catch (e: any) {
        status(`[probe] ${m.key} (${m.provider}) -> ERROR: ${e?.message}`);
      }
    }
    status(`PROBE done. total cost=$${cost.spentUsd.toFixed(4)}`);
    return;
  }

  // FULL run
  writeFileSync(STATUS_PATH, `FULL start ${new Date().toISOString()}  (QWEN_OFF_NOTHINK=${QWEN_OFF_NOTHINK}, cap=$${CAP_USD})\n`);
  writeFileSync(RUNS_PATH, '');
  const structs = structures();
  let runCount = 0;
  let failCount = 0;

  outer: for (const m of MODELS) {
    const client = makeClient(m, apiKey, cost);
    const blocks = m.evidence === 'all' ? EVIDENCE_BLOCKS.length : 1;
    for (let bi = 0; bi < blocks; bi++) {
      const market = makeMarket(bi);
      for (const st of structs) {
        if (cost.spentUsd >= CAP_USD) { status(`!! COST CAP hit ($${cost.spentUsd.toFixed(4)}); stopping.`); break outer; }
        client.beginStructureRun(st.name, EVIDENCE_BLOCKS[bi].id);
        const t0 = Date.now();
        try {
          const res = await st.predict({ market, tools: noTools, llm: client, params: PARAMS });
          runCount++;
          status(`[full] ${m.key} | ${EVIDENCE_BLOCKS[bi].id} | ${st.name} -> prob=${res.probability.toFixed(3)} | ${((Date.now() - t0) / 1000).toFixed(1)}s | cost=$${cost.spentUsd.toFixed(4)}`);
        } catch (e: any) {
          if (e instanceof CostCapError) { status(`!! ${e.message}; stopping.`); break outer; }
          failCount++;
          status(`[full] ${m.key} | ${EVIDENCE_BLOCKS[bi].id} | ${st.name} -> FAIL: ${e?.message}`);
        }
      }
    }
  }
  status(`FULL done. structure-runs ok=${runCount} fail=${failCount} totalCost=$${cost.spentUsd.toFixed(4)}`);
}

main().catch(e => { status(`FATAL: ${e?.stack ?? e}`); process.exit(1); });
