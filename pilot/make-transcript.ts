// Renders the agents' conversation into two readable Markdown files, so the run
// can be read like a meeting transcript rather than inspected as JSON.
//
//   node --experimental-transform-types --import ./register.mjs make-transcript.ts \
//        [--ledger <path>] [--outcomes <path>] [--packs <path>] [--out <path>] [--out-full <path>]
//
//   pilot5/L1_transcript.md         what each agent SAID (answers only)
//   pilot5/L1_transcript_full.md    the same, plus each agent's private reasoning
//
// The split exists because the reasoning traces are long: the short file is for
// following the argument, the long one for auditing how a conclusion was reached.
//
// `run-pilot5.ts` imports `buildTranscript` and writes both files at the end of
// every full run. Running this file by hand is now only for re-rendering an
// archived ledger — which is why the paths are explicit flags.
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { CallLedgerRecord } from './ledger5.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUTDIR = process.env.PILOT5_OUTDIR ? resolve(process.env.PILOT5_OUTDIR) : resolve(HERE, '..', '..', 'pilot5');
// Defaults follow the level prefix the runner writes. The old unprefixed
// defaults were how `ledger_report_v2.md` ended up describing a different run
// than the `ledger5.jsonl` beside it.
const LEVEL = (process.env.PILOT5_LEVEL ?? '1') === '2' ? 2 : 1;
const LEDGER = resolve(OUTDIR, `L${LEVEL}_ledger5.jsonl`);
const OUTCOMES = resolve(OUTDIR, `L${LEVEL}_outcomes.jsonl`);
const PACKS = resolve(OUTDIR, 'evidence_packs.json');

export interface OutcomeLine {
  modelKey: string; eventId: string;
  outcomes: Record<string, { probability: number | null; dropped: boolean; dropReason?: string }>;
  roundProbabilities?: number[][];
  matched?: { structure: string; structureTokens: number; runsUsed: number;
              matchedTokens: number; residualTokens: number;
              probability: number | null; structureProbability: number | null }[];
  repetitionRuns?: { probability: number; totalTokens: number }[];
}

const readJsonl = <T,>(p: string): T[] =>
  existsSync(p) ? readFileSync(p, 'utf-8').split(/\r?\n/).filter(Boolean).map(l => JSON.parse(l) as T) : [];

/** Pull the text inside a tag, for display. Falls back to the raw answer. */
function section(text: string, tag: 'reason' | 'decision'): string {
  const m = new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`, 'i').exec(text);
  return (m ? m[1] : text).trim();
}

function indent(text: string, by = '> '): string {
  return text.split('\n').map(l => by + l).join('\n');
}

/**
 * Takes records rather than paths so the runner can render straight from the
 * run it just finished, and so a test can drive it from a mock run with no
 * files involved.
 */
export function buildTranscript(
  records: CallLedgerRecord[],
  outcomes: OutcomeLine[],
  packs: any[],
  withReasoning: boolean,
  /**
   * Ground truth is OFF by default since 2026-09-05. The runner regenerates
   * the transcripts at the end of every full session — resumes included — so
   * with the label block unconditional, an open run's directory contained a
   * file joining every booked forecast to its outcome, which contradicted the
   * analysis plan's blinding rule (REVIEW_ANALYSIS_PLAN_2026-09-05.md, L2-01).
   * The truth-bearing rendering still exists, but only via the explicit
   * `--reveal-truth` CLI flag, to be used after the run is closed and scored.
   */
  revealTruth = false,
): string {
  const L: string[] = [];

  L.push(withReasoning ? '# Agent transcript — answers and private reasoning' : '# Agent transcript — answers only');
  L.push('');
  L.push(withReasoning
    ? '_Every call, in the order the protocol makes it. Each agent\'s private reasoning trace is included in a collapsed block; the visible answer follows. The reasoning is what the model wrote to itself, and for the debate it is also what its peers were shown._'
    : '_Every call, in the order the protocol makes it, showing only what each agent produced as its answer. For the reasoning traces see `transcript_full.md`._');
  L.push('');

  // group by (model, event)
  const keys = [...new Set(records.filter(r => r.ok).map(r => `${r.modelKey} ${r.eventId}`))];
  for (const key of keys) {
    const [modelKey, eventId] = key.split(' ');
    const rs = records.filter(r => r.ok && r.modelKey === modelKey && r.eventId === eventId);
    const o = outcomes.find(x => x.modelKey === modelKey && x.eventId === eventId);
    const pack = packs.find(p => eventId.startsWith(p.id));

    L.push('---');
    L.push('');
    L.push(`## ${eventId} — ${modelKey}`);
    L.push('');
    if (pack) {
      L.push(`**${pack.symbol}**, results released ${pack.reportDate}; prediction made after the close of ${pack.t0}.`);
      L.push('');
      if (revealTruth) {
        L.push(`> **Ground truth (never shown to any agent):** the five-day abnormal return was **${(pack.abnormalReturn * 100).toFixed(2)}%**, so the correct answer is **${pack.label === 1 ? 'POSITIVE (label 1)' : 'NEGATIVE (label 0)'}**. A good forecast puts its probability ${pack.label === 1 ? 'above' : 'below'} 0.5.`);
        L.push('');
      }
    }

    // ---------- round 0: the shared workers
    const workers = rs.filter(r => r.agentRole.startsWith('worker_')).sort((a, b) => a.agentRole.localeCompare(b.agentRole));
    if (workers.length) {
      L.push('### Round 0 — the three analysts work independently');
      L.push('');
      L.push('_They cannot see each other. These same three answers serve as the communication-free rung, as the aggregator\'s input, and as round 0 of the debate._');
      L.push('');
      for (const w of workers) {
        L.push(`**Analyst ${w.agentRole.replace('worker_', '')}** → **${section(w.responseText, 'decision')}**`);
        L.push('');
        if (withReasoning && w.reasoningText) {
          L.push('<details><summary>private reasoning trace</summary>');
          L.push('');
          L.push(indent(w.reasoningText.trim()));
          L.push('');
          L.push('</details>');
          L.push('');
        }
        L.push(indent(section(w.responseText, 'reason')));
        L.push('');
      }
    }

    // ---------- the aggregator
    const agg = rs.find(r => r.agentRole === 'aggregator');
    if (agg) {
      L.push('### Rung 2 — the aggregator');
      L.push('');
      L.push('_It sees the three reasoning sections above, verbatim and **without their numbers**, and never sees the source material._');
      L.push('');
      L.push(`**Aggregator** → **${section(agg.responseText, 'decision')}**`);
      L.push('');
      if (withReasoning && agg.reasoningText) {
        L.push('<details><summary>private reasoning trace</summary>');
        L.push('');
        L.push(indent(agg.reasoningText.trim()));
        L.push('');
        L.push('</details>');
        L.push('');
      }
      L.push(indent(section(agg.responseText, 'reason')));
      L.push('');
    }

    // ---------- debate rounds
    const rounds = [...new Set(rs.filter(r => r.structure === 'debate' && r.round).map(r => r.round!))].sort();
    for (const round of rounds) {
      const rr = rs.filter(r => r.structure === 'debate' && r.round === round)
                   .sort((a, b) => a.agentRole.localeCompare(b.agentRole));
      L.push(`### Rung 3 — debate, revision round ${round}`);
      L.push('');
      L.push('_Each analyst has now read the other two answers in full (reasoning **and** number) plus its own previous answer, under an instruction to state disagreement openly and not to converge merely to converge._');
      L.push('');
      for (const r of rr) {
        L.push(`**Analyst ${r.agentRole.replace('debater_', '')}** → **${section(r.responseText, 'decision')}**`);
        L.push('');
        if (withReasoning && r.reasoningText) {
          L.push('<details><summary>private reasoning trace</summary>');
          L.push('');
          L.push(indent(r.reasoningText.trim()));
          L.push('');
          L.push('</details>');
          L.push('');
        }
        L.push(indent(section(r.responseText, 'reason')));
        L.push('');
      }
      if (o?.roundProbabilities?.[round]) {
        const v = o.roundProbabilities[round];
        L.push(`_After this round: ${v.map(x => x.toFixed(2)).join(' / ')} — spread ${(Math.max(...v) - Math.min(...v)).toFixed(3)}._`);
        L.push('');
      }
    }

    // ---------- what each rung concluded
    if (o) {
      L.push('### What each rung concluded');
      L.push('');
      L.push('| rung | probability | vs the matched baseline |');
      L.push('|---|---:|---|');
      for (const s of ['repetition', 'aggregator', 'debate']) {
        const oc = o.outcomes[s];
        const m = o.matched?.find(x => x.structure === s);
        const vs = m && m.probability !== null
          ? `${m.probability.toFixed(2)} from ${m.runsUsed} runs at the same budget (${m.structureTokens.toLocaleString()} tok, residual ${m.residualTokens})`
          : s === 'repetition' ? '— (it is the baseline)' : '—';
        L.push(`| ${s} | ${oc?.dropped ? 'DROPPED' : oc?.probability?.toFixed(2) ?? '—'} | ${vs} |`);
      }
      L.push('');
      if (o.repetitionRuns?.length) {
        L.push(`_Baseline reference curve — all ${o.repetitionRuns.length} runs in generation order: ${o.repetitionRuns.map(r => r.probability.toFixed(2)).join(', ')}_`);
        L.push('');
      }
    }
  }
  return L.join('\n');
}

// Importing this file must not write anything: `run-pilot5.ts` imports the
// builder, and the previous top-level writeFileSync calls would have fired on
// import and clobbered the transcripts of the run in progress.
const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const argv = process.argv.slice(2);
  const arg = (flag: string) => {
    const i = argv.indexOf(flag);
    if (i < 0) return undefined;
    const v = argv[i + 1];
    // A flag with nothing after it must not fall through to the default path
    // and silently overwrite the wrong file.
    if (v === undefined || v.startsWith('--')) { console.error(`${flag} needs a path`); process.exit(1); }
    return v;
  };
  const ledgerPath = resolve(arg('--ledger') ?? LEDGER);
  const outcomesPath = resolve(arg('--outcomes') ?? OUTCOMES);
  const packsPath = resolve(arg('--packs') ?? PACKS);
  const outPath = resolve(arg('--out') ?? resolve(OUTDIR, `L${LEVEL}_transcript.md`));
  const outFullPath = resolve(arg('--out-full') ?? resolve(OUTDIR, `L${LEVEL}_transcript_full.md`));
  // Ground truth only on explicit request, and only after the run is closed:
  // the flag exists so the readable truth-annotated transcript is still
  // producible for the write-up, without ever existing during a live run.
  const revealTruth = argv.includes('--reveal-truth');

  const records = readJsonl<CallLedgerRecord>(ledgerPath);
  if (!records.length) { console.error(`no records in ${ledgerPath}`); process.exit(1); }
  const outcomes = readJsonl<OutcomeLine>(outcomesPath);
  const packs: any[] = existsSync(packsPath) ? JSON.parse(readFileSync(packsPath, 'utf-8')) : [];
  writeFileSync(outPath, buildTranscript(records, outcomes, packs, false, revealTruth));
  writeFileSync(outFullPath, buildTranscript(records, outcomes, packs, true, revealTruth));
  console.log(`wrote ${outPath}${revealTruth ? ' (ground truth REVEALED)' : ' (ground truth withheld)'}`);
  console.log(`wrote ${outFullPath}${revealTruth ? ' (ground truth REVEALED)' : ' (ground truth withheld)'}`);
}
