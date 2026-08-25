// Synthetic, fictional earnings evidence — NO real market data (pilot §4).
// Made-up tickers, made-up numbers. Outcomes do not exist and do not matter;
// this validates plumbing, labels, and logs, not predictions.
import type { Market } from '../src/types.js';

export const QUESTION =
  'Probability that this stock\'s abnormal return over the next 5 trading days is positive? Answer with a number in [0,1].';

const BLOCK_A = `FICTIONAL COMPANY — Nuvora Dynamics (ticker: NVRD), Q3 FY25 earnings release (synthetic).
Nuvora Dynamics, a mid-cap industrial-automation supplier, reported quarterly revenue of $1.42B, up 11% year over year and modestly ahead of the $1.38B consensus. Diluted EPS came in at $2.31 versus a $2.02 consensus, a 14% beat, helped by a one-time supplier rebate of roughly $0.12 per share that management flagged as non-recurring. Gross margin expanded 180 basis points to 41.6% on a richer mix of software-attached hardware. Management raised full-year EPS guidance to a range of $8.90–$9.10 from a prior $8.40–$8.70, and lifted the revenue outlook to approximately $5.75B. The order backlog grew to $3.1B, up 7% sequentially, with book-to-bill above 1.0 for the fourth consecutive quarter. On the call, the CFO noted that data-center cooling and grid-automation demand remained "resilient," while short-cycle factory orders in Europe were "softer than we would like." Free cash flow was $410M, and the company repurchased $250M of stock during the quarter. The board authorized a new $1.0B buyback. Net leverage stands at 1.3x EBITDA. Analysts on the call pressed management on pricing durability given normalizing component costs; the CEO responded that roughly 60% of the margin gain was structural (mix and software) and 40% was transitory (rebates and freight). Guidance assumes no improvement in European short-cycle demand for the remainder of the year. Two sell-side analysts noted after the call that the raise was "high quality despite the one-timer." The stock had underperformed its sector by about 6% into the print.`;

const BLOCK_B = `FICTIONAL COMPANY — Halcyon Therapeutics (ticker: HLCN), Q2 FY25 earnings release (synthetic).
Halcyon Therapeutics, a commercial-stage biopharma, reported revenue of $612M, up 4% year over year but below the $655M consensus, and EPS of $0.71 against a $0.88 consensus — a clear miss. The shortfall was driven by slower-than-expected uptake of its lead therapy in the second-line setting and higher rebates to payers. Management cut full-year revenue guidance to $2.45B–$2.55B from $2.7B–$2.8B and lowered EPS guidance to $3.10–$3.30 from $3.80–$4.00. Gross-to-net erosion accelerated to 34% from 29% a year ago. On a more constructive note, a Phase III readout for a follow-on indication is expected within the next 90 days, and management characterized the trial as "on track" with enrollment complete. The company ended the quarter with $1.8B in cash and no debt maturities until 2029. R&D spend rose 18% as two programs advanced into pivotal trials. During the call, several analysts questioned whether the payer rebate pressure was structural; the Chief Commercial Officer conceded that "the competitive dynamics in second line have intensified faster than we modeled." The company withdrew its prior long-term margin target, saying it would "revisit the framework at the investor day in the fall." The stock had outperformed the biotech index by roughly 9% over the prior month heading into the report, and short interest was elevated at about 11% of float.`;

const BLOCK_C = `FICTIONAL COMPANY — Meridian Grid Solutions (ticker: MRGS), Q4 FY25 earnings release (synthetic).
Meridian Grid Solutions, a utility-scale energy-storage integrator, reported revenue of $2.08B, up 22% year over year and roughly in line with the $2.05B consensus. EPS was $1.44 versus $1.41 expected — a narrow beat. However, the composition was mixed: revenue growth was strong, but gross margin compressed 120 basis points to 19.8% as the company absorbed higher battery-cell costs on legacy fixed-price contracts. Management reaffirmed, but did not raise, full-year guidance of $8.6B–$8.9B revenue and $5.70–$6.05 EPS, disappointing some investors who had expected an increase given the top-line momentum. New bookings were a record $3.4B, and management highlighted a "structural shift" toward higher-margin services and long-duration storage. The CFO guided to margin recovery in the second half as legacy contracts roll off, but declined to quantify the exit-rate margin. Net debt rose to $2.9B (2.4x EBITDA) to fund working capital for a large project pipeline; the company drew on its revolver during the quarter. On the call, an analyst flagged that days-sales-outstanding had risen to 78 from 61, raising questions about collection timing on large projects; the CFO attributed it to "milestone timing, not credit quality." Guidance embeds an assumption that cell costs stabilize. The stock traded roughly in line with its sector over the prior quarter, and options markets implied an unusually large post-earnings move.`;

export const EVIDENCE_BLOCKS: { id: string; ticker: string; description: string }[] = [
  { id: 'A_NVRD_beat', ticker: 'NVRD', description: BLOCK_A },
  { id: 'B_HLCN_miss', ticker: 'HLCN', description: BLOCK_B },
  { id: 'C_MRGS_mixed', ticker: 'MRGS', description: BLOCK_C },
];

export function makeMarket(blockIndex: number): Market {
  const b = EVIDENCE_BLOCKS[blockIndex];
  return {
    index: blockIndex,
    question: `${QUESTION} (${b.ticker})`,
    description: b.description,
  };
}
