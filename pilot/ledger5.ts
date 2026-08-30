// Five-bucket token ledger (working document §5; NEXT_STEP_EXPLAINED.md §5).
//
// Read side  (prompt tokens):   D  evidence + base instruction
//                               A  coordination instruction
//                               B  received messages from OTHER agents (per read)
//                               B_SELF  the agent's own earlier output re-read (debate)
//                                       — recorded separately; whether it counts as
//                                       communication is an open prompt-session item.
// Write side (completion tokens): C1 final answer (<decision>), C2 everything else
//                               (visible <reason> + tags + hidden reasoning trace).
//
// Identities enforced per call: D+A+B+B_SELF = prompt_tokens, C1+C2 = completion_tokens.
// The provider count is the truth for the totals; the local tokenizer only splits.
// The pre-rescale local-vs-provider gap is logged, never hidden.

import { getEncoding, type Tiktoken } from 'js-tiktoken';
import type { GenerateRequest, GenerateResponse, LLMClient } from '../src/types.js';
import { parseTagged, stripThink, TaggedParseError, type TaggedOutput } from './tagged.js';

export type ReadBucket = 'D' | 'A' | 'B' | 'B_SELF';

export interface Segment {
  bucket: ReadBucket;
  text: string;
  /** Free-form origin note, e.g. "evidence", "peer:2:round1", "role:aggregator". */
  origin: string;
}

/** A prompt is an ordered list of typed segments; system and user are kept apart. */
export interface SegmentedPrompt {
  system: Segment[];
  user: Segment[];
}

export function renderSegments(segs: Segment[]): string {
  return segs.map(s => s.text).join('');
}

// ---------------------------------------------------------------- tokenizer
let enc: Tiktoken | null = null;
export function localTokens(text: string): number {
  if (!text) return 0;
  if (!enc) enc = getEncoding('cl100k_base');
  return enc.encode(text, 'all').length;
}

/**
 * Split one message's segments into token counts that sum EXACTLY to the token
 * count of the concatenated message.
 *
 * Tokenizing each segment on its own and adding up over-counts, because a token
 * may span a segment boundary and splitting forces an extra boundary there. The
 * error is roughly one token per boundary, so it falls hardest on short, numerous
 * segments — which is exactly what bucket A consists of (role headers such as
 * "--- Analyst 2 ---"). Left uncorrected it inflates A, i.e. the communication
 * cost RQ2 reports, in the same direction as the chat-template overhead did.
 *
 * Counting cumulative prefixes attributes each boundary token to the segment
 * that completes it and makes the parts add up to the whole by construction.
 */
export function splitTokens(segs: Segment[]): number[] {
  const out: number[] = [];
  let acc = '';
  let prev = 0;
  for (const s of segs) {
    acc += s.text;
    const now = localTokens(acc);
    out.push(Math.max(0, now - prev));
    prev = now;
  }
  // The whole point of this function is that the parts add up to the whole.
  // The clamp above can break that: a cumulative token count is not strictly
  // monotone (appending text can merge two tokens into one, so `now` may be
  // BELOW `prev`), and clamping that difference to zero throws the deficit away
  // instead of charging it to anyone. The total then exceeds localTokens(whole)
  // and the read-side identity D + A + B + B_SELF = prompt_tokens drifts.
  //
  // The excess is taken back from the largest segments first, because that is
  // where a merged token most plausibly came from and because it cannot push a
  // small segment negative.
  const total = prev;
  let excess = out.reduce((a, b) => a + b, 0) - total;
  while (excess > 0) {
    let big = 0;
    for (let i = 1; i < out.length; i++) if (out[i] > out[big]) big = i;
    if (out[big] === 0) break;                 // nothing left to take
    const take = Math.min(excess, out[big]);
    out[big] -= take;
    excess -= take;
  }
  return out;
}

// ------------------------------------------------------------------ records
export interface ReadSideCounts {
  D: number; A: number; B: number; B_SELF: number;
  /** Σ local counts before rescale. */
  localSum: number;
  /** provider prompt_tokens. */
  providerPrompt: number;
  /**
   * C2 (D18): the model's measured fixed chat-template overhead, assigned in
   * FULL to D. A lone forecaster pays it too, so spreading it over A and B
   * would inflate the communication cost that RQ2 reports.
   */
  templateOverhead: number;
  /**
   * Residual gap AFTER removing the fixed overhead, as a share of the provider
   * count: (providerPrompt - overhead - localSum) / providerPrompt. This is the
   * length-proportional tokenizer mismatch, and it alone is spread across the
   * buckets in proportion to their local size.
   */
  gapShare: number;
}

export interface WriteSideCounts {
  C1: number; C2: number;
  /** hidden reasoning tokens (subset of C2), straight from the provider. */
  C2_hidden: number;
  /** visible C2 = C2 - C2_hidden. */
  C2_visible: number;
  providerCompletion: number;
  providerReasoning: number;
  /** Our local token count of the visible answer text. */
  localVisible: number;
  /**
   * localVisible - (providerCompletion - providerReasoning), in TOKENS.
   *
   * NOTE ON UNITS: this mixes two tokenizers, so it is a reconciliation
   * DIAGNOSTIC, not an error term. localVisible is counted with cl100k_base,
   * while the provider's figures are in the model's own tokenizer. Part of any
   * gap is simply that difference (measured at roughly 5-13 % for Qwen).
   *
   * A real residual survives that correction, though. Over the 30 Qwen calls of
   * the 2026-08-27 pilot, rescaling the visible answer into the provider's units
   * still implies ~15 % more visible output than `completion - reasoning` leaves
   * room for. The most plausible reading is that `message.reasoning` returns an
   * INCOMPLETE rendering of the trace that was billed, since the shortfall runs
   * in that direction. It cannot be settled from outside the provider.
   *
   * Why the accounting is nonetheless safe (working doc §5.4):
   *  - every figure the thesis reports is a PROVIDER count; the local tokenizer
   *    only supplies proportions, which are near scale-invariant;
   *  - the read side is exact (shares of `prompt_tokens`), and compute matching
   *    uses `total_tokens`, so neither RQ1 nor RQ2 touches this at all;
   *  - C1 + C2 = completion_tokens still holds exactly, being shares of it.
   * Only the hidden/visible split WITHIN C2 is approximate. It is a descriptive
   * statistic and an input to no research question; the ratio is reported per
   * model rather than smoothed away.
   */
  visibleDiscrepancy: number;
  /** Kept for continuity; see visibleDiscrepancy for the interpretable figure. */
  gapShare: number;
}

export interface CallLedgerRecord {
  structure: string;
  eventId: string;
  modelKey: string;
  agentRole: string;
  round: number | null;
  attempt: number;             // 1 = first try, 2.. = retries
  ok: boolean;                 // parsed fine
  malformed: boolean;          // this attempt failed to parse
  parseError?: string;
  /** Set on failed attempts only. See FailureKind. */
  failureKind?: FailureKind;
  /** Provider finish_reason for this attempt, verbatim. */
  finishReason?: string | null;
  read: ReadSideCounts;
  write: WriteSideCounts;
  totalTokens: number;
  costUsd: number;
  durationMs: number;
  probability?: number;
  /** Segment-level detail so the split can be re-audited offline. */
  segments: { side: 'system' | 'user'; bucket: ReadBucket; origin: string; localTokens: number }[];
  responseText: string;
  reasoningText: string;
}

// ------------------------------------------------------------- the ledger
export interface LedgerCallResult {
  parsed: TaggedOutput;
  raw: GenerateResponse;
  record: CallLedgerRecord;
  /** The verbatim visible answer (tags included) — what a peer would receive in a debate. */
  fullVisible: string;
}

export class MalformedCallError extends Error {
  constructor(message: string, public readonly attempts: CallLedgerRecord[]) {
    super(message);
    this.name = 'MalformedCallError';
  }
}

/** Provider-side extras the OpenRouter client attaches to GenerateResponse. */
export interface ProviderExtras {
  reasoningTokens: number;
  reasoningText: string;
  content: string;
  /** OpenRouter finish_reason: 'stop' | 'length' | … — used for the failure taxonomy. */
  finishReason?: string | null;
}

/**
 * Why a call failed to yield a usable answer (D16 amendment, 2026-08-26).
 * Kept apart because only 'format' is a genuine model failure:
 *  - 'truncated'  the answer hit our per-call token cap, so the closing tag was
 *                 never written. That is OUR cap being too low, not the model
 *                 failing; the remedy is raising the cap, not dropping the event.
 *  - 'format'     the model produced no parseable tagged answer within the cap.
 * Infrastructure failures (HTTP 5xx, timeouts, rate limits) never reach here —
 * the OpenRouter client retries them itself and throws if they persist.
 */
export type FailureKind = 'format' | 'truncated';

export class Ledger5 {
  constructor(
    public readonly structure: string,
    public readonly eventId: string,
    public readonly modelKey: string,
    /** Pass a shared array so several structure views write into one ledger. */
    public readonly records: CallLedgerRecord[] = [],
    /**
     * C2 (D18): this model's fixed per-call chat-template overhead in tokens,
     * measured once by `measureTemplateOverhead`. Booked entirely to D.
     * 0 falls back to the old purely proportional behaviour.
     */
    public readonly templateOverhead: number = 0,
  ) {}

  /**
   * One logical call: send, parse; on parse failure re-send the IDENTICAL prompt
   * and try again, up to `maxAttempts` times in total.
   *
   * D16 amendment (2026-08-26): the pre-registered rule is AT LEAST one retry
   * (maxAttempts >= 2), not exactly one. Raising it is legitimate because a
   * malformed call contains no parseable probability, so there is nothing to
   * select on — retrying cannot be answer-shopping. Two properties keep that
   * true and must not be broken:
   *   1. the retry re-sends the IDENTICAL prompt (no corrective hint, which
   *      would be a different prompt and would favour format-flaky models);
   *   2. the decision to retry is mechanical (triggered by parse failure), never
   *      taken by a human after inspecting the failed content.
   * Every attempt is recorded, so the attempt distribution is auditable.
   */
  async call(
    llm: LLMClient,
    agentRole: string,
    prompt: SegmentedPrompt,
    opts: { maxTokens?: number; temperature: number; round?: number; maxAttempts?: number },
  ): Promise<LedgerCallResult> {
    const maxAttempts = Math.max(2, opts.maxAttempts ?? 2);
    const req: GenerateRequest = {
      systemPrompt: renderSegments(prompt.system),
      userPrompt: renderSegments(prompt.user),
      tools: [],
      maxTokens: opts.maxTokens,
      temperature: opts.temperature,
      metadata: { agentRole, configName: this.structure },
    };
    const attempts: CallLedgerRecord[] = [];
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      const raw = await llm.generate(req);
      const extras = (raw as GenerateResponse & { extras?: ProviderExtras }).extras
        ?? { reasoningTokens: 0, reasoningText: '', content: raw.text };
      const { visible } = stripThink(extras.content);
      let parsed: TaggedOutput | null = null;
      let parseError: string | undefined;
      try {
        parsed = parseTagged(visible);
      } catch (e) {
        parseError = e instanceof TaggedParseError ? e.message : String(e);
      }
      const record = this.book(agentRole, opts.round ?? null, attempt, prompt, raw, extras, visible, parsed, parseError);
      this.records.push(record);
      attempts.push(record);
      if (parsed) {
        return { parsed, raw, record, fullVisible: visible.trim() };
      }
    }
    const kinds = [...new Set(attempts.map(a => a.failureKind))].join('/');
    throw new MalformedCallError(
      `${agentRole}: unusable after ${maxAttempts} attempts [${kinds}] (${attempts.map(a => a.parseError).join(' | ')})`,
      attempts,
    );
  }

  private book(
    agentRole: string, round: number | null, attempt: number,
    prompt: SegmentedPrompt, raw: GenerateResponse, extras: ProviderExtras,
    visible: string, parsed: TaggedOutput | null, parseError?: string,
  ): CallLedgerRecord {
    // ---- read side
    // Prefix-based split PER MESSAGE: system and user are separate messages, so
    // their boundary is a real one, but boundaries inside a message are ours and
    // must not create phantom tokens (see splitTokens).
    const sysTok = splitTokens(prompt.system);
    const usrTok = splitTokens(prompt.user);
    const segs = [
      ...prompt.system.map((s, i) => ({ side: 'system' as const, bucket: s.bucket, origin: s.origin, localTokens: sysTok[i] })),
      ...prompt.user.map((s, i) => ({ side: 'user' as const, bucket: s.bucket, origin: s.origin, localTokens: usrTok[i] })),
    ];
    const localByBucket: Record<ReadBucket, number> = { D: 0, A: 0, B: 0, B_SELF: 0 };
    for (const s of segs) localByBucket[s.bucket] += s.localTokens;
    const localSum = segs.reduce((a, s) => a + s.localTokens, 0);
    const providerPrompt = raw.usage.promptTokens;
    // C2 (D18): peel off the fixed template overhead first and give it to D,
    // then distribute only the length-proportional residual across the buckets.
    const overhead = Math.max(0, Math.min(this.templateOverhead, providerPrompt));
    const readScaled = rescale(localByBucket, localSum, providerPrompt - overhead);
    readScaled.D += overhead;
    const read: ReadSideCounts = {
      ...readScaled,
      localSum,
      providerPrompt,
      templateOverhead: overhead,
      gapShare: providerPrompt > 0 ? (providerPrompt - overhead - localSum) / providerPrompt : 0,
    };

    // ---- write side
    const providerCompletion = raw.usage.completionTokens;
    const providerReasoning = Math.min(extras.reasoningTokens ?? 0, providerCompletion);
    const visibleBudget = providerCompletion - providerReasoning;
    const decisionLocal = parsed ? localTokens(parsed.decisionRaw) : 0;
    const visibleLocal = localTokens(visible);
    const restLocal = Math.max(visibleLocal - decisionLocal, 0);
    const reasoningLocal = localTokens(extras.reasoningText || '');

    // Splitting the completion (measured 2026-08-28, 120 Qwen calls).
    //
    // We do NOT derive the visible part as `completion - reasoning_tokens`. That
    // subtraction is unusable per call: localVisible / (completion - reasoning)
    // has a coefficient of variation of 47 % and reaches 5.0, i.e. the implied
    // "budget" is sometimes a fifth of the text that demonstrably exists.
    //
    // Instead both parts are counted with the SAME local tokenizer and rescaled
    // to the provider's completion total — the identical principle used on the
    // read side. One ruler measures both parts, so its bias cancels in the ratio,
    // and the parts sum to the provider total by construction.
    //
    // This is sound because localReasoning / reasoning_tokens is a TIGHT constant
    // (mean 0.880, CV 6.6 %): a stable multiplicative factor is what a tokenizer
    // bias looks like, and it also shows the returned reasoning text is complete —
    // a trace truncated at varying points could not produce a stable ratio.
    //
    // Fallback: if the provider reports reasoning tokens but returns no text, the
    // proportion is unknowable, so we fall back to the subtraction for that call.
    const haveTrace = reasoningLocal > 0;
    const w = haveTrace
      ? rescale(
          { C1: decisionLocal, C2v: restLocal, C2h: reasoningLocal },
          decisionLocal + restLocal + reasoningLocal,
          providerCompletion,
        )
      : visibleBudget > 0
        ? { ...rescale({ C1: decisionLocal, C2v: restLocal }, decisionLocal + restLocal, visibleBudget), C2h: providerReasoning }
        // Providers sometimes report reasoning_tokens equal to (or above) the
        // whole completion. Taken at face value that leaves no room for the
        // visible answer, and the rescale above returned C1 = 0 for a call that
        // demonstrably produced one - the final answer, the one thing RQ2 has to
        // count, priced at nothing.
        //
        // Between the provider's reasoning count and text we can actually read,
        // the text is the better evidence, so the whole completion is split by
        // the visible parts and the hidden share is recorded as 0. That is
        // certainly not right either, but it is wrong in a direction a reader
        // can see: `visibleDiscrepancy` on such a call equals the entire visible
        // length, and `maxVisibleDiscrepancy` carries it into the report.
        : { ...rescale({ C1: decisionLocal, C2v: restLocal }, decisionLocal + restLocal, providerCompletion), C2h: 0 };
    const write: WriteSideCounts = {
      C1: w.C1,
      C2: w.C2v + w.C2h,
      C2_hidden: w.C2h,
      C2_visible: w.C2v,
      providerCompletion,
      providerReasoning,
      localVisible: visibleLocal,
      visibleDiscrepancy: visibleLocal - visibleBudget,
      gapShare: visibleBudget > 0 ? (visibleBudget - visibleLocal) / visibleBudget : 0,
    };

    // Failure taxonomy (D16 amendment): a cut-off answer is our cap being too
    // low, not the model failing to comply — kept apart so the two are never
    // conflated in the drop statistics.
    const finishReason = extras.finishReason ?? null;
    const failureKind: FailureKind | undefined = parsed
      ? undefined
      : (finishReason === 'length' ? 'truncated' : 'format');

    return {
      structure: this.structure, eventId: this.eventId, modelKey: this.modelKey,
      agentRole, round, attempt,
      ok: !!parsed, malformed: !parsed, parseError, failureKind, finishReason,
      read, write,
      totalTokens: raw.usage.totalTokens,
      costUsd: raw.costUsd ?? 0,
      durationMs: raw.durationMs,
      probability: parsed?.probability,
      segments: segs,
      responseText: visible,
      reasoningText: extras.reasoningText,
    };
  }
}

/**
 * C2 (D18) — measure a model's FIXED per-call chat-template overhead.
 *
 * The provider wraps the messages in control markers (role delimiters, BOS and
 * similar). Those are billed tokens that our local tokenizer never sees, and
 * their number does not depend on how long the prompt is — a lone forecaster
 * pays exactly the same amount as a debating one. Attributing them to D rather
 * than smearing them over A and B removes a small but systematic upward bias on
 * the communication cost that RQ2 reports.
 *
 * Method: send the shortest possible two-message prompt and subtract our local
 * count from what the provider billed. One call per model, a fraction of a cent.
 * Uses the same message shape (system + user) as every real call, which is what
 * makes the measured constant transferable.
 *
 * The residual is floored at 0: a negative result would mean the model's own
 * tokenizer is more compact than cl100k_base on this text, which is a
 * proportional effect and belongs in the residual gap, not here.
 */
export async function measureTemplateOverhead(
  llm: LLMClient,
  probe: { system: string; user: string } = { system: 'a', user: 'b' },
): Promise<{ overhead: number; providerPrompt: number; localCount: number }> {
  const res = await llm.generate({
    systemPrompt: probe.system,
    userPrompt: probe.user,
    tools: [],
    maxTokens: 1,
    temperature: 0,
    metadata: { agentRole: 'calibration_probe', configName: 'calibration' },
  });
  const localCount = localTokens(probe.system) + localTokens(probe.user);
  const providerPrompt = res.usage.promptTokens;
  return { overhead: Math.max(0, providerPrompt - localCount), providerPrompt, localCount };
}

/** Largest-remainder rescale of integer parts so they sum EXACTLY to `target`. */
function rescale<K extends string>(parts: Record<K, number>, localSum: number, target: number): Record<K, number> {
  const keys = Object.keys(parts) as K[];
  const out = {} as Record<K, number>;
  if (target <= 0 || localSum <= 0) { for (const k of keys) out[k] = 0; return out; }
  const exact = keys.map(k => (parts[k] / localSum) * target);
  const floors = exact.map(Math.floor);
  let rem = target - floors.reduce((a, b) => a + b, 0);
  const order = keys.map((_, i) => i).sort((i, j) => (exact[j] - floors[j]) - (exact[i] - floors[i]));
  for (const i of order) { if (rem <= 0) break; floors[i]++; rem--; }
  keys.forEach((k, i) => (out[k] = floors[i]));
  return out;
}

// ------------------------------------------------------ structure summary
export interface StructureTotals {
  D: number; A: number; B: number; B_SELF: number; C1: number; C2: number; C2_hidden: number;
  prompt: number; completion: number; total: number; costUsd: number;
  calls: number; malformedAttempts: number; wastedTokens: number;
  /** Failed attempts split by cause; only `format` is a genuine model failure. */
  failFormat: number; failTruncated: number;
  /** Successful calls that needed more than one attempt, and the max seen. */
  retriedCalls: number; maxAttemptUsed: number;
  /** Worst |localVisible - (completion - reasoning)| in tokens; see visibleDiscrepancy. */
  maxVisibleDiscrepancy: number;
  /** A + B + whole combiner call(s). B_SELF reported separately (open item). */
  commCost: number; commCostInclSelf: number;
  maxReadGap: number; maxWriteGap: number;
  /**
   * The same three totals AS BILLED, i.e. including the attempts that were
   * discarded and retried.
   *
   * The five buckets above deliberately cover usable calls only - a malformed
   * answer has no parseable decision, so there is no honest C1 to book. But the
   * discarded attempt was still paid for, and leaving it out of the totals made
   * the money and the token count disagree (one reproduction reported $0.0042
   * against 920 tokens for a call that really cost 9,540). Whichever structure
   * fails more often then looks cheaper than it was, which is precisely the
   * confound compute matching exists to remove, so `structures5.ts` matches on
   * these figures rather than on `total` / `commCost`.
   */
  billedPrompt: number; billedCompletion: number; billedTotal: number;
  billedCommCost: number; billedCommCostInclSelf: number;
}

/**
 * Sum a structure's records. `combinerRoles` are agents that exist only to
 * combine others (their ENTIRE cost is communication cost, working doc §5.2).
 * Malformed attempts are excluded from the buckets but counted as waste;
 * their dollar cost stays in costUsd so billing is honest.
 */
export function summarize(records: CallLedgerRecord[], combinerRoles: string[] = []): StructureTotals {
  const t: StructureTotals = {
    D: 0, A: 0, B: 0, B_SELF: 0, C1: 0, C2: 0, C2_hidden: 0, prompt: 0, completion: 0, total: 0, costUsd: 0,
    calls: 0, malformedAttempts: 0, wastedTokens: 0, failFormat: 0, failTruncated: 0,
    retriedCalls: 0, maxAttemptUsed: 0, maxVisibleDiscrepancy: 0,
    commCost: 0, commCostInclSelf: 0, maxReadGap: 0, maxWriteGap: 0,
    billedPrompt: 0, billedCompletion: 0, billedTotal: 0,
    billedCommCost: 0, billedCommCostInclSelf: 0,
  };
  for (const r of records) {
    t.costUsd += r.costUsd;
    // Billed first, for EVERY record. A discarded attempt re-sent the whole
    // prompt and was charged for it; the read-side buckets are known for it too,
    // because they are computed from the prompt rather than from the answer.
    t.billedPrompt += r.read.providerPrompt;
    t.billedCompletion += r.write.providerCompletion;
    t.billedTotal += r.totalTokens;
    {
      const isCombinerBilled = combinerRoles.includes(r.agentRole);
      const ownBilled = isCombinerBilled ? r.totalTokens : r.read.A + r.read.B;
      t.billedCommCost += ownBilled;
      t.billedCommCostInclSelf += ownBilled + (isCombinerBilled ? 0 : r.read.B_SELF);
    }
    if (r.malformed) {
      t.malformedAttempts++;
      t.wastedTokens += r.totalTokens;
      if (r.failureKind === 'truncated') t.failTruncated++; else t.failFormat++;
      continue;
    }
    t.calls++;
    t.maxAttemptUsed = Math.max(t.maxAttemptUsed, r.attempt);
    if (r.attempt > 1) t.retriedCalls++;
    t.D += r.read.D; t.A += r.read.A; t.B += r.read.B; t.B_SELF += r.read.B_SELF;
    t.C1 += r.write.C1; t.C2 += r.write.C2; t.C2_hidden += r.write.C2_hidden;
    t.prompt += r.read.providerPrompt; t.completion += r.write.providerCompletion; t.total += r.totalTokens;
    t.maxReadGap = Math.max(t.maxReadGap, Math.abs(r.read.gapShare));
    t.maxWriteGap = Math.max(t.maxWriteGap, Math.abs(r.write.gapShare));
    t.maxVisibleDiscrepancy = Math.max(t.maxVisibleDiscrepancy, Math.abs(r.write.visibleDiscrepancy ?? 0));
    const isCombiner = combinerRoles.includes(r.agentRole);
    const own = isCombiner ? r.totalTokens : r.read.A + r.read.B;
    t.commCost += own;
    t.commCostInclSelf += own + (isCombiner ? 0 : r.read.B_SELF);
  }
  return t;
}
