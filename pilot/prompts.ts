// =====================================================================
//  PROMPTS — FROZEN 2026-08-26. The prompt session is closed; every item of
//  the agenda (A1-A6, B1-B8, C1-C4) is decided, and the reasoning on both
//  sides plus the rejected alternatives is recorded in
//  ../../DECISIONS_PROMPT_SESSION.md (D1-D19).
//  Do not change wording here without adding a numbered entry there.
//
//  Everything a model READS lives in this one file, so wording can change
//  without touching the ledger or the structures.
//
//  Every function returns SEGMENTS, each tagged with its ledger bucket:
//    D       evidence + base instruction (what a lone forecaster needs anyway)
//    A       coordination instruction (exists only because this is a team)
//    B       text written by another agent, read here
//    B_SELF  the agent's own earlier output, read again
//  The bucket tag is the ONLY thing the ledger looks at. If a sentence
//  moves between D and A, its tokens move between buckets. That is the
//  point: the classification is made here, once, in the open.
//
// =====================================================================

import type { Segment, SegmentedPrompt } from './ledger5.js';

// ---------------------------------------------------------------- D: base
// A1 FULLY CLOSED 2026-08-25 — sentence-by-sentence rationale in
// DECISIONS_PROMPT_SESSION.md: D2 (leakage stance), D3 (each sentence),
// D5 (calibration, anchor-free + Brier), D8 (AR definition = minus market,
// beta item resolved; working doc §2.2 amended to match).
export const BASE_SYSTEM = `\
You are a financial analyst. A company has just published its quarterly earnings. \
Using only the information given to you, estimate the probability that the stock's \
abnormal return over the next five trading days is positive. Abnormal return means \
the stock's five-day return minus the five-day return of the market (S&P 500). \
Your reasoning must consist of clearly followable arguments. Each argument must \
state the specific data from your input that it builds on, so that your reasoning \
can be followed and the data you use is documented. \
Your probability is a frequency statement, not a figure of speech, and it is graded \
with a proper scoring rule (the Brier score): if you state a probability p, such \
events should occur in about a fraction p of comparable cases. \
Do not write disclaimers about uncertainty being inherent; quantify the uncertainty \
in your probability instead.`;

// A2 CLOSED 2026-08-25 (D6): empty template only — no filled example (anchoring),
// percent safety net stays unadvertised, strict "and nothing else".
//
// D27 AMENDMENT 2026-08-29: the sentence keeping the probability out of the
// reasoning text. AGGREGATOR_ROLE tells the aggregator it is shown the analysts'
// reasoning "without their final numbers", but the code only ever stripped the
// <decision> TAG - a number stated in prose passed straight through. Measured on
// the pilot logs: 3 of 12 worker answers stated their own probability in the
// reasoning (all qwen3-32b, 3 of its 6 worker calls), both qwen aggregator calls
// received at least one, and in L2 the aggregator wrote "arriving at 45%" about
// Analyst 2 and returned exactly 0.45. Rung 2 was therefore not blind
// aggregation for those events, and the prompt was asserting something false to
// the model, which is worse than asserting nothing.
//
// Fixed at the SOURCE rather than by filtering the models' text afterwards: no
// output is altered, and the line sits in FORMAT_INSTRUCTION, which all three
// rungs and the single-agent baseline share, so bucket A stays identical across
// conditions. The residual leak rate is still measured per run (see
// reasonStatesOwnNumber) because compliance will not be perfect.
export const FORMAT_INSTRUCTION = `\
Answer in exactly two tagged sections and nothing else:
<reason>
Your reasoning, as long or short as you need. State your probability only in the decision section below, not here.
</reason>
<decision>
A single number between 0 and 1.
</decision>`;

// A3 CLOSED 2026-08-25 (D7): question restated after the evidence (long-prompt
// recency) PLUS a format reminder sentence (author's addition).
export const QUESTION_LINE = `\
Question: what is the probability that this stock's abnormal return over the next five trading days is positive?
Answer in exactly the two tagged sections specified above.`;

// ---------------------------------------------------------- A: coordination
// A4 CLOSED 2026-08-25 (D10): explicit asymmetry + numbers withheld + data-
// provenance bridge. No "do not simply average" line — "weigh / note conflicts"
// already carries that job without hinting that numbers might be hidden in the text.
// NOTE: the aggregator's system prompt is BASE_SYSTEM + this role + FORMAT, so it
// inherits task, AR definition, grounding, calibration and anti-hedging verbatim.
export const AGGREGATOR_ROLE = (n: number) => `\
In this task you are aggregating the work of ${n} analysts who have each read the \
source material and written their reasoning about the same earnings release. You do \
not see the source material yourself. You are shown their reasoning sections verbatim, without \
their final numbers. The data cited inside their reasoning is taken from that source \
material, so you can reason both about their arguments and about the data those \
arguments rest on. Weigh their arguments, note where they conflict, and form the \
final probability from their reasoning.`;

// A5 CLOSED 2026-08-26 (D11): N&S anti-convergence wording chosen by the author
// over the assistant's neutral recommendation. Trade-off logged in
// DECISIONS_PROMPT_SESSION.md D11 — preserved diversity would be confounded with
// the instruction, but a collapse observed DESPITE this instruction is a stronger
// finding, and it keeps the debate comparable with N&S, whose harness this is.
// No persona restatement (D10 amendment). Mentions the own prior => requires B2=true.
export const DEBATE_REVISE_ROLE = `\
In this task you are one of several analysts answering the same question. Below are \
the other analysts' full answers and your own earlier answer. Revise your estimate \
where their reasoning convinces you, and keep it where it does not. State \
disagreement openly. Do not converge merely to converge.`;

// ------------------------------------------------- structural toggles (closed)
/** DECIDED 2026-08-25 (D1, DECISIONS_PROMPT_SESSION.md): blind aggregator per exposé p.3.
 *  Mitigation lives in BASE_SYSTEM (data-grounded reasoning). Do not flip. */
export const AGGREGATOR_SEES_EVIDENCE = false;
/** B2 CLOSED 2026-08-26 (D12): YES — Du et al. standard; "keep your estimate" (D11)
 *  presupposes knowing it, and models carry no memory between calls. Logged as
 *  B_SELF; whether B_SELF counts toward headline comm cost is deferred to the
 *  analysis plan (both figures are always recorded). Do not flip. */
export const DEBATE_SHOWS_OWN_PRIOR = true;

// ============================================================ builders
export interface EvidencePack {
  id: string;
  /** The level-1 (or level-1+2) evidence text, verbatim. */
  text: string;
}

const seg = (bucket: Segment['bucket'], text: string, origin: string): Segment => ({ bucket, text, origin });

/** Rung 1/2 worker and debate round 0: base instruction + evidence only. Zero A, zero B. */
export function workerPrompt(ev: EvidencePack): SegmentedPrompt {
  return {
    system: [
      seg('D', BASE_SYSTEM, 'base_system'),
      seg('D', '\n\n', 'sep'),
      seg('D', FORMAT_INSTRUCTION, 'format'),
    ],
    user: [
      seg('D', ev.text, `evidence:${ev.id}`),
      seg('D', '\n\n' + QUESTION_LINE, 'question'),
    ],
  };
}

/** Rung 2 aggregator: role (A) + the workers' <reason> texts (B). Optionally evidence (D). */
export function aggregatorPrompt(ev: EvidencePack, workerReasons: string[]): SegmentedPrompt {
  const user: Segment[] = [];
  if (AGGREGATOR_SEES_EVIDENCE) {
    user.push(seg('D', ev.text, `evidence:${ev.id}`), seg('D', '\n\n', 'sep'));
  }
  user.push(seg('A', 'The analysts\' reasoning sections:\n\n', 'agg_frame'));
  workerReasons.forEach((r, i) => {
    user.push(seg('A', `Analyst ${i + 1} wrote:\n`, `agg_header:${i + 1}`));
    user.push(seg('B', r.trim(), `worker:${i + 1}:reason`));
    user.push(seg('A', '\n\n', 'sep'));
  });
  user.push(seg('D', QUESTION_LINE, 'question'));
  return {
    system: [
      seg('D', BASE_SYSTEM, 'base_system'),
      seg('A', '\n\n' + AGGREGATOR_ROLE(workerReasons.length), 'agg_role'),
      seg('D', '\n\n' + FORMAT_INSTRUCTION, 'format'),
    ],
    user,
  };
}

/** Rung 3 debate revision: base + revise role (A), evidence (D), peers' full answers (B), own prior (B_SELF).
 *  A6 CLOSED 2026-08-26 (D13): navigation headers kept as drafted, including the
 *  round number — the TOTAL round count is never revealed, so it carries no
 *  strategic signal about being near the end.
 *
 *  AMENDED 2026-08-27 (D21): peer headers changed from "--- Analyst N ---" to the
 *  prose form "Analyst N wrote:", in both the aggregator and the debate prompt.
 *  A DeepSeek aggregator once answered in "--- reasoning --- / --- numeric ---"
 *  sections, i.e. it appeared to imitate the delimiter style of its own input.
 *  The failure did NOT reproduce on a re-run of the identical call, so this is a
 *  cheap precaution against a plausible-but-unconfirmed cause, not a proven fix. */
export function debateRevisePrompt(
  ev: EvidencePack,
  own: { agentId: string; fullText: string },
  peers: { agentId: string; fullText: string }[],
  round: number,
): SegmentedPrompt {
  const user: Segment[] = [
    seg('D', ev.text, `evidence:${ev.id}`),
    seg('D', '\n\n' + QUESTION_LINE, 'question'),
    seg('A', `\n\nRound ${round}. Other analysts' answers:\n\n`, 'debate_frame'),
  ];
  for (const p of peers) {
    user.push(seg('A', `Analyst ${p.agentId} wrote:\n`, `peer_header:${p.agentId}`));
    user.push(seg('B', p.fullText.trim(), `peer:${p.agentId}:round${round - 1}`));
    user.push(seg('A', '\n\n', 'sep'));
  }
  if (DEBATE_SHOWS_OWN_PRIOR) {
    user.push(seg('A', 'Your own previous answer:\n', 'own_header'));
    user.push(seg('B_SELF', own.fullText.trim(), `self:${own.agentId}:round${round - 1}`));
    user.push(seg('A', '\n\n', 'sep'));
  }
  user.push(seg('A', 'Give your revised answer.', 'debate_ask'));
  return {
    system: [
      seg('D', BASE_SYSTEM, 'base_system'),
      seg('A', '\n\n' + DEBATE_REVISE_ROLE, 'debate_role'),
      seg('D', '\n\n' + FORMAT_INSTRUCTION, 'format'),
    ],
    user,
  };
}


/**
 * Did this reasoning text state the answer's OWN probability in prose?
 *
 * Deliberately narrow: it looks for the model's own final number, not for any
 * percentage. Reasoning legitimately cites figures ("revenue up 10%", "the
 * stock fell 3.2%"), so a general percent detector would fire constantly and
 * measure nothing. Matching the answer's own value catches exactly the leak
 * that matters - the number the aggregator could copy, and in the pilot did.
 *
 * Known limits, stated because this figure is reported: a reason that says
 * "roughly even odds" without a numeral is not caught, and a cited figure that
 * genuinely coincides with the forecast - a stock down 45% in an answer of 0.45
 * - is still a false positive, because no amount of pattern matching can tell
 * those apart. It is a floor on the leak rate, not an exact count.
 *
 * What it is NOT any more is a substring artefact: see the boundaries below.
 */
export function reasonStatesOwnNumber(reason: string, probability: number): boolean {
  const hay = reason.toLowerCase();
  const pct = Math.round(probability * 100);
  const patterns: RegExp[] = [];

  // The boundaries are the whole point, and the first version did not have them.
  // It tested `hay.includes('.65')`, which is true of "the S&P has lagged 9.65%"
  // - an unrelated market figure - so an answer of 0.65 was flagged as leaking
  // its own number when it had done nothing of the sort. Measured on the pilot
  // logs the loose version flagged 4 of 12 worker answers where 3 are genuine.
  // Financial reasoning is made of percentages, so this fires often, and the
  // leak rate is a number the thesis reports (D27).
  //
  // `(?<![DIGIT.])` refuses a match preceded by a digit or a decimal point, so
  // "9.65" and "10.35" cannot supply the fraction, and "165%" cannot supply the
  // percentage.
  const left = '(?<![0-9\\.])';

  // "65 %", "65%", "65 percent"
  patterns.push(new RegExp(left + pct + '\\s*(?:%|percent)'));

  // "0.65" and ".65", and the two-decimal spelling "0.50" of 0.5. A trailing
  // digit is refused too, so 0.65 is not found inside "0.653".
  const spellings = new Set([String(probability), probability.toFixed(2)]);
  for (const dec of spellings) {
    const dot = dec.indexOf('.');
    if (dot < 0) continue;   // 0 and 1 have no fraction; a bare "1" would match everywhere
    const frac = '\\.' + dec.slice(dot + 1);
    patterns.push(new RegExp(left + '0?' + frac + '(?![0-9])'));
  }

  return patterns.some(re => re.test(hay));
}