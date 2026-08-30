// Tagged output format for the thesis structures (decision (B), 2026-08-13).
//
// Every probability-emitting agent must answer as
//   <reason> ...free text... </reason>
//   <decision> 0.35 </decision>
// The boundary between reasoning and answer is thereby declared by the model,
// not inferred by the experimenter. See NEXT_STEP_EXPLAINED.md §4.

export class TaggedParseError extends Error {
  constructor(message: string, public readonly rawText: string) {
    super(message);
    this.name = 'TaggedParseError';
  }
}

export interface TaggedOutput {
  /** Text inside <reason>…</reason>, trimmed. */
  reason: string;
  /** Raw text inside <decision>…</decision>, trimmed. */
  decisionRaw: string;
  /** Parsed probability in [0,1]. */
  probability: number;
  /** Anything outside the two tags (leading/trailing chatter). Kept for the ledger. */
  outside: string;
}

const REASON_RE = /<reason>([\s\S]*?)<\/reason>/i;
const DECISION_RE = /<decision>([\s\S]*?)<\/decision>/i;

/** Strip an inline <think>…</think> block some providers put in `content`. */
export function stripThink(text: string): { visible: string; think: string } {
  const m = text.match(/<think>([\s\S]*?)<\/think>/i);
  if (!m) return { visible: text, think: '' };
  return { visible: text.replace(m[0], ''), think: m[1] };
}

/**
 * Strict parse. Throws TaggedParseError when either tag is missing, appears
 * more than once, or the decision is not a probability in [0,1].
 */
export function parseTagged(text: string): TaggedOutput {
  const reasonMatches = text.match(new RegExp(REASON_RE.source, 'gi')) ?? [];
  const decisionMatches = text.match(new RegExp(DECISION_RE.source, 'gi')) ?? [];
  if (reasonMatches.length !== 1) {
    throw new TaggedParseError(`expected exactly one <reason> block, found ${reasonMatches.length}`, text);
  }
  if (decisionMatches.length !== 1) {
    throw new TaggedParseError(`expected exactly one <decision> block, found ${decisionMatches.length}`, text);
  }
  const reason = REASON_RE.exec(text)![1].trim();
  const decisionRaw = DECISION_RE.exec(text)![1].trim();

  // Accept "0.35", ".35", "0.35 " and also "35%" (converted); reject anything else.
  let probability: number;
  const pct = /^(\d{1,3}(?:\.\d+)?)\s*%$/.exec(decisionRaw);
  if (pct) {
    // "0.35%" cannot be read safely. As written it is 0.0035, but a model that
    // meant 0.35 and appended a percent sign out of habit writes exactly this,
    // and the two readings differ by a factor of 100 - the difference between
    // "roughly even odds" and "practically impossible". Guessing either way
    // would put a fabricated number into a proper scoring rule, so an ambiguous
    // answer is refused. The caller retries, and if the model keeps writing it
    // the attempt is counted as a genuine format failure, which is the honest
    // record.
    //
    // A whole number with a percent sign has no second reading: "35%" is 0.35,
    // "1%" is 0.01. Only a value at or below 1 written WITH decimals is
    // ambiguous.
    const written = Number(pct[1]);
    if (written <= 1 && pct[1].includes('.')) {
      throw new TaggedParseError(
        `ambiguous decision "${decisionRaw}": as written this is ${written / 100}, ` +
        `but it may mean ${written}; state a bare probability such as "${written}" instead`,
        text,
      );
    }
    probability = written / 100;
  } else if (/^(1(?:\.0+)?|0(?:\.\d+)?|\.\d+)$/.test(decisionRaw)) {
    probability = Number(decisionRaw);
  } else {
    throw new TaggedParseError(`decision is not a bare probability: "${decisionRaw.slice(0, 40)}"`, text);
  }
  if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
    throw new TaggedParseError(`decision out of [0,1]: ${decisionRaw}`, text);
  }

  const outside = text.replace(REASON_RE, '').replace(DECISION_RE, '').trim();
  return { reason, decisionRaw, probability, outside };
}
