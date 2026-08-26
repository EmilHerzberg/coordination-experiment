// The three-rung ladder (working document §3) on the five-bucket ledger.
//
//   rung 1  repetition   N independent workers, probabilities combined by formula
//   rung 2  aggregator   the SAME N workers' <reason> texts (numbers withheld)
//                        → one aggregating agent
//   rung 3  debate       the SAME N workers as round 0, then R revision rounds
//                        in which every agent reads the peers' FULL answers
//
// Shared worker runs (working doc §3.3): one set of worker calls serves as
// rung 1, as the input of rung 2, and as round 0 of rung 3. In the ledger the
// worker records carry structure='workers'; a structure's totals are
// workers + its own extra calls (see `structureRecords`).

import type { LLMClient } from '../src/types.js';
import { Ledger5, MalformedCallError, type CallLedgerRecord, type LedgerCallResult } from './ledger5.js';
import { workerPrompt, aggregatorPrompt, debateRevisePrompt, type EvidencePack } from './prompts.js';

export interface LadderParams {
  agentCount: number;          // N
  debateRounds: number;        // R revision rounds after round 0
  temperature: number;
  maxTokensPerCall?: number;
  /**
   * D16 amendment: pre-registered as AT LEAST one retry, i.e. >= 2 total attempts.
   * Raising it is legitimate (a malformed call holds no probability, so there is
   * nothing to select on) provided the retry stays identical and mechanical.
   * Every attempt is logged, so the distribution is auditable.
   */
  maxParseAttempts?: number;
  /** C2 (D18): the model's measured fixed chat-template overhead, booked to D. */
  templateOverhead?: number;
  /**
   * B5 CLOSED 2026-08-26 (D19): 'mean'. The exposé commits to averaging, and the
   * Brier score's convexity guarantees Brier(mean) <= mean(individual Briers),
   * with the gap equal to the diversity the D14 spread diagnostic measures.
   * The median (used by N&S and pilot v1) has no such guarantee and discards
   * two of three values at N=3. Deviation from the scaffold study is deliberate.
   */
  aggregate: 'mean' | 'median';
}

export const COMBINER_ROLES = ['aggregator'];
export const STRUCTURES = ['repetition', 'aggregator', 'debate'] as const;
export type StructureName = typeof STRUCTURES[number];

export interface StructureOutcome {
  probability: number | null;   // null = dropped (malformed)
  dropped: boolean;
  dropReason?: string;
}

export interface EventRunResult {
  eventId: string;
  modelKey: string;
  outcomes: Record<StructureName, StructureOutcome>;
  ledger: Ledger5;
  /**
   * D14 diagnostic — the agents' probabilities per debate round.
   * roundProbabilities[0] = the shared workers (round 0), [1] = after revision 1, …
   *
   * Two questions this answers, both decided in this session:
   *  - B7/temperature: does temperature 0.7 produce a real ensemble at round 0,
   *    or do all N workers land on the same number (then averaging buys nothing
   *    and the debate has nothing to argue about)?
   *  - D11/anti-convergence: does the spread SHRINK across rounds despite the
   *    explicit instruction not to converge merely to converge? Collapse observed
   *    against that instruction is the stronger of the two possible findings.
   */
  roundProbabilities: number[][];
}

/** Max-minus-min of a probability vector; 0 means the agents agree exactly. */
export function spread(ps: number[]): number {
  return ps.length ? Math.max(...ps) - Math.min(...ps) : 0;
}

/** Population standard deviation; reported next to the spread. */
export function sd(ps: number[]): number {
  if (ps.length < 2) return 0;
  const m = ps.reduce((a, b) => a + b, 0) / ps.length;
  return Math.sqrt(ps.reduce((a, b) => a + (b - m) ** 2, 0) / ps.length);
}

export function combine(ps: number[], how: 'mean' | 'median'): number {
  if (ps.length === 0) throw new Error('combine(): empty');
  if (how === 'mean') return ps.reduce((a, b) => a + b, 0) / ps.length;
  const s = ps.slice().sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Records that belong to a structure's cost = shared workers + own calls. */
export function structureRecords(all: CallLedgerRecord[], s: StructureName): CallLedgerRecord[] {
  return all.filter(r => r.structure === 'workers' || r.structure === s);
}

/**
 * Run all three rungs for one event on one model. Malformed calls (after the
 * single retry) drop the affected structure(s); a malformed WORKER drops all
 * three, since all three depend on the worker set.
 */
export async function runLadder(
  llm: LLMClient,
  ev: EvidencePack,
  modelKey: string,
  p: LadderParams,
): Promise<EventRunResult> {
  const ledger = new Ledger5('workers', ev.id, modelKey, [], p.templateOverhead ?? 0);
  const N = p.agentCount;
  const outcomes = {} as Record<StructureName, StructureOutcome>;
  const drop = (why: string): StructureOutcome => ({ probability: null, dropped: true, dropReason: why });
  const roundProbabilities: number[][] = [];   // [round0(workers), round1, ...] — D14 diagnostic

  // ---- shared workers (rung 1 / rung 2 input / debate round 0)
  // allSettled: wait for ALL sibling calls so no record lands after we return.
  const workerSettled = await Promise.allSettled(
    Array.from({ length: N }, (_, i) =>
      ledger.call(llm, `worker_${i + 1}`, workerPrompt(ev), {
        maxTokens: p.maxTokensPerCall, temperature: p.temperature, maxAttempts: p.maxParseAttempts, round: 0,
      }),
    ),
  );
  const workerFail = firstFailure(workerSettled, 'worker');
  if (workerFail) {
    for (const s of STRUCTURES) outcomes[s] = drop(workerFail);
    return { eventId: ev.id, modelKey, outcomes, ledger, roundProbabilities };
  }
  const workers = workerSettled.map(r => (r as PromiseFulfilledResult<LedgerCallResult>).value);
  roundProbabilities.push(workers.map(w => w.parsed.probability));   // round 0 = the shared workers
  outcomes.repetition = {
    probability: combine(workers.map(w => w.parsed.probability), p.aggregate),
    dropped: false,
  };

  // ---- rung 2: aggregator reads <reason> only
  {
    const sub = withStructure(ledger, 'aggregator');
    try {
      const r = await sub.call(llm, 'aggregator', aggregatorPrompt(ev, workers.map(w => w.parsed.reason)), {
        maxTokens: p.maxTokensPerCall, temperature: p.temperature, maxAttempts: p.maxParseAttempts,
      });
      outcomes.aggregator = { probability: r.parsed.probability, dropped: false };
    } catch (e) {
      outcomes.aggregator = drop(e instanceof MalformedCallError ? e.message : `aggregator error: ${(e as Error).message}`);
    }
  }

  // ---- rung 3: debate, round 0 = workers, then R revision rounds with FULL answers
  {
    const sub = withStructure(ledger, 'debate');
    let states = workers.map((w, i) => ({ agentId: String(i + 1), fullText: w.fullVisible, probability: w.parsed.probability }));
    let failed: string | null = null;
    for (let round = 1; round <= p.debateRounds && !failed; round++) {
      const settled = await Promise.allSettled(
        states.map((own, i) => {
          const peers = states.filter((_, j) => j !== i).map(s => ({ agentId: s.agentId, fullText: s.fullText }));
          return sub
            .call(llm, `debater_${own.agentId}`, debateRevisePrompt(ev, own, peers, round), {
              maxTokens: p.maxTokensPerCall, temperature: p.temperature, maxAttempts: p.maxParseAttempts, round,
            })
            .then(r => ({ agentId: own.agentId, fullText: r.fullVisible, probability: r.parsed.probability }));
        }),
      );
      failed = firstFailure(settled, 'debate');
      if (!failed) {
        states = settled.map(r => (r as PromiseFulfilledResult<typeof states[number]>).value);
        roundProbabilities.push(states.map(s => s.probability));
      }
    }
    outcomes.debate = failed
      ? drop(failed)
      : { probability: combine(states.map(s => s.probability), p.aggregate), dropped: false };
  }

  return { eventId: ev.id, modelKey, outcomes, ledger, roundProbabilities };
}

function firstFailure(settled: PromiseSettledResult<unknown>[], what: string): string | null {
  for (const r of settled) {
    if (r.status === 'rejected') {
      const e = r.reason;
      return e instanceof MalformedCallError ? e.message : `${what} error: ${(e as Error)?.message ?? String(e)}`;
    }
  }
  return null;
}

/** A view on the same ledger that stamps a different structure name on new records. */
function withStructure(base: Ledger5, structure: StructureName): Ledger5 {
  return new Ledger5(structure, base.eventId, base.modelKey, base.records, base.templateOverhead);
}
