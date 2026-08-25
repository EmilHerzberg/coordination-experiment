// =====================================================================
//  DRAFT PROMPTS — NOT FINAL. To be worked out together in the prompt
//  session before the pilot rerun. Everything that a model READS lives in
//  this one file so that the wording can be changed without touching the
//  ledger or the structures.
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
//  Open items for the prompt session are marked  // [SESSION] .
// =====================================================================

import type { Segment, SegmentedPrompt } from './ledger5.js';

// ---------------------------------------------------------------- D: base
// [SESSION] wording of the base instruction; whether to mention "calibrated";
//           whether to forbid disclaimers; anything task-specific (5-day AR).
export const BASE_SYSTEM = `\
You are a financial analyst. A company has just published its quarterly earnings. \
Using only the information given to you, estimate the probability that the stock's \
abnormal return over the next five trading days is positive. Abnormal return means \
the stock's return minus what the market would have implied for it. \
Be calibrated: when you say 70%, such events should occur about 70% of the time.`;

// [SESSION] exact wording of the format contract; whether examples are given.
export const FORMAT_INSTRUCTION = `\
Answer in exactly two tagged sections and nothing else:
<reason>
Your reasoning, as long or short as you need.
</reason>
<decision>
A single number between 0 and 1.
</decision>`;

/** The question line placed after the evidence. */
export const QUESTION_LINE = `\
Question: what is the probability that this stock's abnormal return over the next five trading days is positive?`;

// ---------------------------------------------------------- A: coordination
// [SESSION] aggregator role text; whether it is told how many workers; whether
//           it is told that the workers' probabilities were withheld.
export const AGGREGATOR_ROLE = (n: number) => `\
You are the aggregating analyst. ${n} independent analysts have each written their \
reasoning about the same earnings release. You are shown their reasoning sections \
verbatim, without their final numbers. Weigh their arguments, note where they \
conflict, and form your own probability from the reasoning.`;

// [SESSION] debate revise text; how strongly to push back on convergence.
export const DEBATE_REVISE_ROLE = `\
You are one analyst in a small group. Below are the other analysts' full answers \
to the same question, and your own earlier answer. Revise your estimate where their \
reasoning convinces you and hold it where it does not. State disagreement openly. \
Do not converge merely to converge.`;

// ---------------------------------------------------------- toggles [SESSION]
/** Does the aggregator ALSO see the evidence? Pilot: no (workers' text only). */
export const AGGREGATOR_SEES_EVIDENCE = false;
/** Does a debating agent see its own previous answer again? (Du et al.: yes.) */
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
    user.push(seg('A', `--- Analyst ${i + 1} ---\n`, `agg_header:${i + 1}`));
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

/** Rung 3 debate revision: base + revise role (A), evidence (D), peers' full answers (B), own prior (B_SELF). */
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
    user.push(seg('A', `--- Analyst ${p.agentId} ---\n`, `peer_header:${p.agentId}`));
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
