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

// ------------------------------------------------------------------ records
export interface ReadSideCounts {
  D: number; A: number; B: number; B_SELF: number;
  /** Σ local counts before rescale. */
  localSum: number;
  /** provider prompt_tokens. */
  providerPrompt: number;
  /** (providerPrompt - localSum) / providerPrompt. Chat-template overhead + tokenizer mismatch. */
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
  /** local count of visible text vs (completion - reasoning); share. */
  gapShare: number;
}

export interface CallLedgerRecord {
  structure: string;
  eventId: string;
  modelKey: string;
  agentRole: string;
  round: number | null;
  attempt: number;             // 1 = first try, 2 = the single retry
  ok: boolean;                 // parsed fine
  malformed: boolean;          // this attempt failed to parse
  parseError?: string;
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
}

export class Ledger5 {
  constructor(
    public readonly structure: string,
    public readonly eventId: string,
    public readonly modelKey: string,
    /** Pass a shared array so several structure views write into one ledger. */
    public readonly records: CallLedgerRecord[] = [],
  ) {}

  /**
   * One logical call: send, parse; on parse failure re-send the IDENTICAL prompt once.
   * Both attempts are recorded (attempt 1 flagged malformed if it failed). If the
   * retry also fails, throws MalformedCallError; the caller drops the event.
   */
  async call(
    llm: LLMClient,
    agentRole: string,
    prompt: SegmentedPrompt,
    opts: { maxTokens?: number; temperature: number; round?: number },
  ): Promise<LedgerCallResult> {
    const req: GenerateRequest = {
      systemPrompt: renderSegments(prompt.system),
      userPrompt: renderSegments(prompt.user),
      tools: [],
      maxTokens: opts.maxTokens,
      temperature: opts.temperature,
      metadata: { agentRole, configName: this.structure },
    };
    const attempts: CallLedgerRecord[] = [];
    for (let attempt = 1; attempt <= 2; attempt++) {
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
    throw new MalformedCallError(
      `${agentRole}: malformed after retry (${attempts.map(a => a.parseError).join(' | ')})`,
      attempts,
    );
  }

  private book(
    agentRole: string, round: number | null, attempt: number,
    prompt: SegmentedPrompt, raw: GenerateResponse, extras: ProviderExtras,
    visible: string, parsed: TaggedOutput | null, parseError?: string,
  ): CallLedgerRecord {
    // ---- read side
    const segs = [
      ...prompt.system.map(s => ({ side: 'system' as const, ...s })),
      ...prompt.user.map(s => ({ side: 'user' as const, ...s })),
    ].map(s => ({ side: s.side, bucket: s.bucket, origin: s.origin, localTokens: localTokens(s.text) }));
    const localByBucket: Record<ReadBucket, number> = { D: 0, A: 0, B: 0, B_SELF: 0 };
    for (const s of segs) localByBucket[s.bucket] += s.localTokens;
    const localSum = segs.reduce((a, s) => a + s.localTokens, 0);
    const providerPrompt = raw.usage.promptTokens;
    const readScaled = rescale(localByBucket, localSum, providerPrompt);
    const read: ReadSideCounts = {
      ...readScaled,
      localSum,
      providerPrompt,
      gapShare: providerPrompt > 0 ? (providerPrompt - localSum) / providerPrompt : 0,
    };

    // ---- write side
    const providerCompletion = raw.usage.completionTokens;
    const providerReasoning = Math.min(extras.reasoningTokens ?? 0, providerCompletion);
    const visibleBudget = providerCompletion - providerReasoning;
    const decisionLocal = parsed ? localTokens(parsed.decisionRaw) : 0;
    const visibleLocal = localTokens(visible);
    const restLocal = Math.max(visibleLocal - decisionLocal, 0);
    const w = rescale({ C1: decisionLocal, C2v: restLocal }, decisionLocal + restLocal, visibleBudget);
    const write: WriteSideCounts = {
      C1: w.C1,
      C2: w.C2v + providerReasoning,
      C2_hidden: providerReasoning,
      C2_visible: w.C2v,
      providerCompletion,
      providerReasoning,
      gapShare: visibleBudget > 0 ? (visibleBudget - visibleLocal) / visibleBudget : 0,
    };

    return {
      structure: this.structure, eventId: this.eventId, modelKey: this.modelKey,
      agentRole, round, attempt,
      ok: !!parsed, malformed: !parsed, parseError,
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
  /** A + B + whole combiner call(s). B_SELF reported separately (open item). */
  commCost: number; commCostInclSelf: number;
  maxReadGap: number; maxWriteGap: number;
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
    calls: 0, malformedAttempts: 0, wastedTokens: 0, commCost: 0, commCostInclSelf: 0, maxReadGap: 0, maxWriteGap: 0,
  };
  for (const r of records) {
    t.costUsd += r.costUsd;
    if (r.malformed) { t.malformedAttempts++; t.wastedTokens += r.totalTokens; continue; }
    t.calls++;
    t.D += r.read.D; t.A += r.read.A; t.B += r.read.B; t.B_SELF += r.read.B_SELF;
    t.C1 += r.write.C1; t.C2 += r.write.C2; t.C2_hidden += r.write.C2_hidden;
    t.prompt += r.read.providerPrompt; t.completion += r.write.providerCompletion; t.total += r.totalTokens;
    t.maxReadGap = Math.max(t.maxReadGap, Math.abs(r.read.gapShare));
    t.maxWriteGap = Math.max(t.maxWriteGap, Math.abs(r.write.gapShare));
    const isCombiner = combinerRoles.includes(r.agentRole);
    const own = isCombiner ? r.totalTokens : r.read.A + r.read.B;
    t.commCost += own;
    t.commCostInclSelf += own + (isCombiner ? 0 : r.read.B_SELF);
  }
  return t;
}
