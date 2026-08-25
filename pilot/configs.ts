// Structures 1 and 3 — NOT in the N&S repo; they carry the thesis contribution
// (the single-call anchor and the communication-free-vs-aggregator null).
// Built on the same TraceLedger + prompt scaffold as their configs.
import type { CoordinationConfig, PredictArgs, PredictResult } from '../src/types.js';
import { TraceLedger } from '../src/configs/base.js';
import {
  COMMON_SYSTEM_HEADER,
  COMMON_OUTPUT_FORMAT,
  buildSystemPrompt,
  buildForecasterUserPrompt,
} from '../src/prompts.js';
import { parseFinalProbability } from '../src/parsing.js';

// ---- Structure 1: single call -------------------------------------------
export class SingleCall implements CoordinationConfig {
  readonly name = 'single_call';
  async predict(args: PredictArgs): Promise<PredictResult> {
    const { market, llm, params } = args;
    const ledger = new TraceLedger(this.name);
    const systemPrompt = buildSystemPrompt('forecaster');
    const userPrompt = buildForecasterUserPrompt(market);
    const r = await ledger.call(llm, 'forecaster_solo', systemPrompt, userPrompt, {
      maxTokens: params.maxTokensPerMarket,
      temperature: params.temperature,
    });
    const probability = parseFinalProbability(r.text);
    return { probability, trace: ledger.build() };
  }
}

// ---- Structure 3: aggregator-ensemble -----------------------------------
// Same N independent worker runs as the arithmetic ensemble, PLUS one
// aggregator agent that receives ONLY the workers' outputs (never the
// evidence) and produces the final probability. This is the cheapest
// nonzero dose of communication (Definition B): worker outputs become the
// aggregator's input.
const AGGREGATOR_INSTRUCTION = `\
Role: Aggregator. You will be shown the full responses of ${'${N}'} independent forecasters \
who each estimated the same probability. You do NOT see the underlying evidence — \
only their responses. Combine their estimates and reasoning into a single, better \
probability for the YES outcome. Weigh their arguments; do not simply average \
blindly if some reasoning is stronger. Note where they disagree and resolve it.`;

function buildAggregatorSystemPrompt(n: number): string {
  const instr = AGGREGATOR_INSTRUCTION.replace('${N}', String(n));
  // No tool reminder: the aggregator has no evidence and no tools by design.
  return [COMMON_SYSTEM_HEADER, '', instr, '', COMMON_OUTPUT_FORMAT].join('\n');
}

export class AggregatorEnsemble implements CoordinationConfig {
  readonly name = 'aggregator_ensemble';
  async predict(args: PredictArgs): Promise<PredictResult> {
    const { market, llm, params } = args;
    const ledger = new TraceLedger(this.name);
    const N = params.agentCount;
    const perCallBudget = Math.floor(params.maxTokensPerMarket / (N + 1));

    const sys = buildSystemPrompt('forecaster');
    const user = buildForecasterUserPrompt(market);
    const workers = await Promise.all(
      Array.from({ length: N }, (_, i) =>
        ledger.call(llm, `agg_worker_${i + 1}`, sys, user, {
          maxTokens: perCallBudget,
          temperature: params.temperature,
        }),
      ),
    );

    // Aggregator sees ONLY worker outputs.
    const workerBlocks = workers
      .map((r, i) => `--- Forecaster ${i + 1} ---\n${r.text.trim()}`)
      .join('\n\n');
    const aggSys = buildAggregatorSystemPrompt(N);
    const aggUser = `The ${N} independent forecasters produced the following full responses. Combine them into one probability.\n\n${workerBlocks}\n\nProduce your integrated probability estimate.`;

    const aggR = await ledger.call(llm, 'aggregator', aggSys, aggUser, {
      maxTokens: perCallBudget,
      temperature: params.temperature,
    });
    const probability = parseFinalProbability(aggR.text);
    return { probability, trace: ledger.build() };
  }
}
