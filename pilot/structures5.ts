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
import { workerPrompt, aggregatorPrompt, debateRevisePrompt, reasonStatesOwnNumber, type EvidencePack } from './prompts.js';

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

/**
 * One compute-matched comparison: a communicating structure against the
 * communication-free rung given the same budget (D23).
 *
 * The exposé's central claim is that structures are compared "at the same cost",
 * so this is the object RQ1 is actually computed from.
 */
export interface MatchedComparison {
  structure: StructureName;
  /** Total tokens the structure itself spent, including the shared workers. */
  structureTokens: number;
  /** How many repetition runs fit inside that budget, in generation order. */
  runsUsed: number;
  /** Total tokens those runs spent. Always <= structureTokens (fits-rule). */
  matchedTokens: number;
  /** structureTokens - matchedTokens. Reported, never hidden (working doc §4.2). */
  residualTokens: number;
  /** The repetition rung's answer at this budget: the combined probability. */
  probability: number | null;
  /** The structure's own answer, repeated here so the pair travels together. */
  structureProbability: number | null;
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
  /**
   * Compute matching (D23). One entry per communicating structure, each pairing
   * it with the communication-free rung at its own budget.
   */
  matched: MatchedComparison[];
  /**
   * Every repetition run's probability and token cost, in GENERATION order.
   * Order is preserved and never re-sorted, so no favourable subset can be
   * selected after the fact (working doc §4.2). The first `agentCount` entries
   * are the shared workers.
   */
  repetitionRuns: { probability: number; totalTokens: number }[];
  /**
   * How many of the workers' reasoning sections still stated their own
   * probability in prose, despite the format instruction telling them not to
   * (D27). The aggregator is described to the model as seeing reasoning without
   * final numbers, so any count above zero is a partial breach of rung 2's
   * defining condition and must be reported rather than assumed away.
   */
  aggregatorNumberLeaks: number;
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

/**
 * Round a probability to 9 decimals.
 *
 * Not cosmetic. The models answer on a 0.05 grid, and multiples of 0.05 are not
 * exactly representable in binary, so an average of them lands 1-2 ULP off the
 * decimal it means. Three agents who all say 0.7 produce 0.6999999999999998,
 * which the analysis then bins one decile TOO LOW - the very bug that
 * `21_metrics.py` rounds its bin edges to prevent, reintroduced here where that
 * fix cannot reach. The floating-point sum is also order-dependent: 26% of
 * three-agent answer multisets gave a different double depending on which
 * worker replied first, so the recorded forecast was not a function of the
 * agents' answers alone. And because `murphy(n_bins=None)` groups by DISTINCT
 * value, identical forecasts split into separate groups and invent resolution
 * out of noise.
 *
 * Nine decimals is far finer than any probability these models can mean and far
 * coarser than the 1e-16 noise being removed. `21_metrics.py` canonicalises
 * with the same rule (FORECAST_DECIMALS = 9) at the point it scores, so both
 * sides of the pipeline agree on when two forecasts are the same forecast.
 */
export function canon(p: number): number {
  return Math.round(p * 1e9) / 1e9;
}

export function combine(ps: number[], how: 'mean' | 'median'): number {
  if (ps.length === 0) throw new Error('combine(): empty');
  if (how === 'mean') return canon(ps.reduce((a, b) => a + b, 0) / ps.length);
  const s = ps.slice().sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  // The two-element median is an average too, so it needs the same treatment.
  return s.length % 2 ? canon(s[m]) : canon((s[m - 1] + s[m]) / 2);
}

/**
 * Records that belong to a structure's cost = the shared workers + its own calls.
 * For 'repetition' this now also picks up the extra runs drawn for compute
 * matching (D23), which are logged under structure='repetition'.
 */
export function structureRecords(all: CallLedgerRecord[], s: StructureName): CallLedgerRecord[] {
  return all.filter(r => r.structure === 'workers' || r.structure === s);
}

/**
 * Run all three rungs for one event on one model.
 *
 * TWO kinds of failure, and the distinction is the whole difference between a
 * result and an outage (2026-09-01):
 *
 *  - MalformedCallError — the model did not produce a parseable answer after
 *    its pre-registered retries (D16). That IS a measurement: the affected
 *    structure is dropped, the drop is recorded, and it belongs in the D16
 *    failure taxonomy. A malformed WORKER drops all three rungs, since all
 *    three are built on the worker set.
 *
 *  - anything else — a CostCapError, or a plain Error the client raises once
 *    ITS own retry budget for 5xx / timeout / connection reset is spent. That
 *    is NOT a measurement, so it PROPAGATES out of this function: run-pilot5
 *    then writes no outcomes line, and the event is retried on the next
 *    resume. This is not a nicety. Swallowing it recorded `dropped=true` for a
 *    provider outage, and because the outcomes line IS the resume marker, the
 *    event counted as done and was never retried — a five-minute outage
 *    permanently converted every event in flight into a fake failure. Measured
 *    on the reproduction before this fix: one injected ECONNRESET on
 *    `worker_2` left the event on disk with 2 ledger records, all three rungs
 *    `dropped=true`, and the resume reporting `completed=0 skipped=2`. It also
 *    contradicted the caption the thesis quotes (make-report5.ts D16:
 *    "Infrastructure errors ... never appear here; the client retries them
 *    itself").
 *
 * The half-finished ledger of a propagating run is deliberately discarded with
 * it: nothing is appended to L*_ledger5.jsonl, and the raw client log's
 * records for that event are orphans, which the resume purge removes.
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
  let aggregatorNumberLeaks = 0;              // D27 residual leak, counted below

  // ---- shared workers (rung 1 / rung 2 input / debate round 0)
  // allSettled: wait for ALL sibling calls so no record lands after we return.
  const workerSettled = await Promise.allSettled(
    Array.from({ length: N }, (_, i) =>
      ledger.call(llm, `worker_${i + 1}`, workerPrompt(ev), {
        maxTokens: p.maxTokensPerCall, temperature: p.temperature, maxAttempts: p.maxParseAttempts, round: 0,
      }),
    ),
  );
  const workerFail = firstMalformed(workerSettled);
  if (workerFail) {
    for (const s of STRUCTURES) outcomes[s] = drop(workerFail);
    return { eventId: ev.id, modelKey, outcomes, ledger, roundProbabilities, matched: [], repetitionRuns: [], aggregatorNumberLeaks };
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
      // Counted on the text actually handed to the aggregator, not on the
      // template, because the template is exactly what looked correct before.
      aggregatorNumberLeaks = workers
        .filter(w => reasonStatesOwnNumber(w.parsed.reason, w.parsed.probability)).length;
      const r = await sub.call(llm, 'aggregator', aggregatorPrompt(ev, workers.map(w => w.parsed.reason)), {
        maxTokens: p.maxTokensPerCall, temperature: p.temperature, maxAttempts: p.maxParseAttempts,
      });
      outcomes.aggregator = { probability: r.parsed.probability, dropped: false };
    } catch (e) {
      // Only a format failure is the aggregator's own result. An outage (or the
      // cost cap) is rethrown, so this event produces no outcomes line at all
      // and is retried whole — see the two-kinds-of-failure note on runLadder.
      if (!(e instanceof MalformedCallError)) throw e;
      outcomes.aggregator = drop(e.message);
    }
  }

  // ---- rung 3: debate, round 0 = workers, then R revision rounds with FULL answers
  {
    const sub = withStructure(ledger, 'debate');
    const roundsBeforeDebate = roundProbabilities.length;   // == 1: round 0 is the shared workers
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
      failed = firstMalformed(settled);   // an outage in round r throws out of the loop and out of runLadder
      if (!failed) {
        states = settled.map(r => (r as PromiseFulfilledResult<typeof states[number]>).value);
        roundProbabilities.push(states.map(s => s.probability));
      }
    }
    // A debate that fails in round 2 has already pushed its round-1 vector.
    // Left there, the convergence table (D14) averages each round column over a
    // DIFFERENT set of events while reporting a single event count, and the
    // event is meanwhile excluded from every other debate result because it is
    // marked dropped. Rolling back to round 0 keeps the two consistent: a
    // debate contributes rounds only if it finished them all.
    if (failed) roundProbabilities.length = roundsBeforeDebate;
    outcomes.debate = failed
      ? drop(failed)
      : { probability: combine(states.map(s => s.probability), p.aggregate), dropped: false };
  }

  // ---- compute matching (D23). The communicating structures have now run and
  // their spend is known, so the communication-free rung is drawn up to the
  // largest of those budgets and each structure is paired with the prefix of
  // runs that fits its own.
  //
  // What one logical call COST is every attempt it took, not just the attempt
  // that happened to parse. A retry re-sends the whole prompt, and a truncated
  // attempt burns the entire per-call cap before being thrown away; both are
  // billed. Counting only the surviving attempt made whichever structure fails
  // more often look cheaper than it was, and its compute-matched baseline was
  // then drawn to that understated budget - the exact confound this matching
  // exists to rule out. In one reproduction a debate that really spent 13,888
  // tokens was compared against a baseline handed 5,056 (2.75x).
  //
  // It cuts both ways: the draw loop below used to skip an unusable extra run
  // entirely, so the BASELINE got its retries for free and kept drawing. Both
  // sides now pay for every attempt.
  const billedFor = (structure: string, role: string) =>
    ledger.records
      .filter(r => r.structure === structure && r.agentRole === role)
      .reduce((a, r) => a + r.totalTokens, 0);

  const repetitionRuns = workers.map((w, i) => ({
    probability: w.parsed.probability,
    totalTokens: billedFor('workers', `worker_${i + 1}`),
  }));
  const matched: MatchedComparison[] = [];
  if (!outcomes.repetition.dropped) {
    // Budget per structure = the shared workers + that structure's own calls.
    // Measured in TOTAL tokens (prompt + completion), as billed: the exposé asks
    // which topology used the most tokens, and read tokens are real cost here
    // (communication cost is A + B, both read-side).
    const workerTokens = repetitionRuns.reduce((a, r) => a + r.totalTokens, 0);
    // No `r.ok` filter: a discarded attempt was billed, so it is part of what
    // this structure spent. See the note on billedFor above.
    const ownTokens = (s: StructureName) =>
      ledger.records
        .filter(r => r.structure === s)
        .reduce((a, r) => a + r.totalTokens, 0);
    const budgets = new Map<StructureName, number>();
    for (const s of STRUCTURES) {
      if (s === 'repetition' || outcomes[s].dropped) continue;
      budgets.set(s, workerTokens + ownTokens(s));
    }

    if (budgets.size > 0) {
      const ceiling = Math.max(...budgets.values());
      // Draw further single-agent runs until the cumulative spend covers the
      // ceiling. They are identical to the workers: same prompt, no coordination.
      const sub = withStructure(ledger, 'repetition');
      let spent = workerTokens;
      let guard = 0;
      const drawOne = (i: number) =>
        sub.call(llm, `rep_${i}`, workerPrompt(ev), {
          maxTokens: p.maxTokensPerCall, temperature: p.temperature,
          maxAttempts: p.maxParseAttempts, round: 0,
        });

      // Draw in BATCHES rather than one at a time. The runs are independent, so
      // serialising them was pure latency: 13-16 sequential calls took 370-760s
      // per ladder against 165s for the rest of it. The batch size is estimated
      // from the average cost of the runs already drawn, then topped up if the
      // estimate fell short. Overshoot is harmless — a run beyond the ceiling is
      // simply never used by any prefix, since every prefix must FIT its budget.
      while (spent < ceiling && guard < MAX_EXTRA_RUNS) {
        const avgRun = repetitionRuns.reduce((a, r) => a + r.totalTokens, 0) / repetitionRuns.length;
        const need = Math.max(1, Math.min(
          Math.ceil((ceiling - spent) / Math.max(avgRun, 1)),
          MAX_EXTRA_RUNS - guard,
          MAX_BATCH,
        ));
        const startIdx = repetitionRuns.length;
        const settled = await Promise.allSettled(
          Array.from({ length: need }, (_, k) => drawOne(startIdx + k + 1)),
        );
        guard += need;
        let anyOk = false;
        for (let k = 0; k < settled.length; k++) {
          const billed = billedFor('repetition', `rep_${startIdx + k + 1}`);
          const r = settled[k];
          if (r.status !== 'fulfilled') {
            // Same rule as the three rungs: only a malformed draw is a result.
            // An outage here does not drop anything — it silently SHORTENS the
            // communication-free baseline, which is the one confound this
            // matching exists to rule out. Measured on the reproduction: with
            // the extra draws failing, debate was compared against a 3-run
            // baseline (945 of 3,948 tokens, residual 3,003) instead of the
            // 12-run one it should have had (3,765/3,948, residual 183) — a
            // 4x understated null hypothesis, written to disk with no drop
            // flag and no way to tell afterwards.
            if (!(r.reason instanceof MalformedCallError)) throw r.reason;
            // An extra run that produced nothing usable still cost money. Not
            // charging for it let the baseline keep drawing on a budget it had
            // already spent, which is the same bug as above with the sign
            // flipped: it would hand the null hypothesis free compute.
            spent += billed;
            continue;
          }
          anyOk = true;
          repetitionRuns.push({ probability: r.value.parsed.probability, totalTokens: billed });
          spent += billed;
        }
        if (!anyOk) break;   // nothing usable came back; report the shortfall
      }

      for (const [s, budget] of budgets) {
        // Prefix that FITS: take runs in generation order while they stay inside
        // the budget. Never re-sorted, so no favourable subset can be chosen
        // afterwards. The resulting shortfall is reported, not hidden.
        let used = 0, tokens = 0;
        for (const run of repetitionRuns) {
          if (tokens + run.totalTokens > budget) break;
          tokens += run.totalTokens; used++;
        }
        matched.push({
          structure: s,
          structureTokens: budget,
          runsUsed: used,
          matchedTokens: tokens,
          residualTokens: budget - tokens,
          probability: used > 0 ? combine(repetitionRuns.slice(0, used).map(r => r.probability), p.aggregate) : null,
          structureProbability: outcomes[s].probability,
        });
      }
    }
  }

  return { eventId: ev.id, modelKey, outcomes, ledger, roundProbabilities, matched, repetitionRuns, aggregatorNumberLeaks };
}

/** Safety stop on the matching draw, so a cheap-run/expensive-structure ratio cannot loop away. */
const MAX_EXTRA_RUNS = 200;
/**
 * Cap on one parallel batch of matching draws. Keeps the burst bounded so the
 * per-model throttle stays the thing that governs request rate, not this.
 */
const MAX_BATCH = 8;

/**
 * Triage a batch of settled sibling calls: return the first MalformedCallError's
 * message (a result the caller records as a drop) and RETHROW anything else (an
 * outage the caller must not record at all).
 *
 * The whole batch is scanned before returning, so an infrastructure error wins
 * over a malformed sibling no matter which order they settled in: "the provider
 * fell over" is not made less true by one agent also having produced garbage,
 * and the two orders must not give different data.
 *
 * The original error object is rethrown, never a wrapped copy — run-pilot5
 * distinguishes the cost cap from everything else with `instanceof
 * CostCapError`, which a re-wrapped message would defeat.
 */
function firstMalformed(settled: PromiseSettledResult<unknown>[]): string | null {
  let malformed: string | null = null;
  for (const r of settled) {
    if (r.status !== 'rejected') continue;
    if (r.reason instanceof MalformedCallError) malformed ??= r.reason.message;
    else throw r.reason;
  }
  return malformed;
}

/** A view on the same ledger that stamps a different structure name on new records. */
function withStructure(base: Ledger5, structure: StructureName): Ledger5 {
  return new Ledger5(structure, base.eventId, base.modelKey, base.records, base.templateOverhead);
}
