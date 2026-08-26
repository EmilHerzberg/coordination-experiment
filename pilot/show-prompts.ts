// Renders the exact prompts every agent receives, with ledger-bucket tags and
// local token counts. No network calls. Use it to audit the prompt design and
// as the source for the thesis appendix.
//   node --experimental-transform-types --import ./register.mjs show-prompts.ts [worker|aggregator|debate|all]
import { workerPrompt, aggregatorPrompt, debateRevisePrompt, type EvidencePack } from './prompts.js';
import { localTokens, type SegmentedPrompt } from './ledger5.js';

const EV: EvidencePack = { id: 'DEMO', text: '<<EVIDENCE BLOCK — the event\'s level-1 (or level-1+2) data goes here>>' };

const REASON_1 = 'The EPS beat of 14 % is inflated by a one-time supplier rebate of about $0.12 per share, so the underlying beat is closer to 8 %. Management raised full-year EPS guidance to $8.90-$9.10 from $8.40-$8.70, which is a larger raise than the one-timer explains. Against that, European short-cycle orders were called "softer than we would like", and the stock had already underperformed its sector by 6 % into the print, so part of the bad news was priced in.';
const REASON_2 = 'Backlog grew to $3.1B, up 7 % sequentially, with book-to-bill above 1.0 for the fourth consecutive quarter, which points to demand rather than a one-quarter effect. Gross margin expanded 180 basis points and management attributed 60 % of that to mix and software, i.e. structural. The new $1.0B buyback and 1.3x net leverage leave room to support the stock.';
const REASON_3 = 'Guidance assumes no improvement in European short-cycle demand, so the raise is not built on optimism about the weak segment. The quality of the beat was questioned on the call regarding pricing durability, and the CEO conceded 40 % of the margin gain is transitory. Two sell-side analysts nevertheless called the raise "high quality despite the one-timer".';

function render(title: string, p: SegmentedPrompt): void {
  console.log('\n' + '='.repeat(78));
  console.log(title);
  console.log('='.repeat(78));
  for (const [side, segs] of [['SYSTEM', p.system], ['USER', p.user]] as const) {
    console.log(`\n--- ${side} ` + '-'.repeat(72 - side.length));
    for (const s of segs) {
      const t = localTokens(s.text);
      if (!s.text.trim()) { continue; }
      console.log(`\n[${s.bucket.padEnd(6)}|${String(t).padStart(4)} tok| ${s.origin}]`);
      console.log(s.text.trim());
    }
  }
  const tot = (segs: typeof p.system) => segs.reduce((a, s) => a + localTokens(s.text), 0);
  const byBucket: Record<string, number> = {};
  for (const s of [...p.system, ...p.user]) byBucket[s.bucket] = (byBucket[s.bucket] ?? 0) + localTokens(s.text);
  console.log('\n--- TOTALS ' + '-'.repeat(66));
  console.log(`system ${tot(p.system)} tok + user ${tot(p.user)} tok = ${tot(p.system) + tot(p.user)} tok`);
  console.log('by bucket: ' + Object.entries(byBucket).map(([k, v]) => `${k}=${v}`).join('  '));
}

const which = (process.argv[2] ?? 'all').toLowerCase();
if (which === 'worker' || which === 'all') {
  render('WORKER  (rung 1 run / rung 2 input / debate round 0)', workerPrompt(EV));
}
if (which === 'aggregator' || which === 'all') {
  render('AGGREGATOR  (rung 2)', aggregatorPrompt(EV, [REASON_1, REASON_2, REASON_3]));
}
if (which === 'debate' || which === 'all') {
  const full = (r: string, p: string) => `<reason>\n${r}\n</reason>\n<decision>\n${p}\n</decision>`;
  render('DEBATER  (rung 3, revision round 1)', debateRevisePrompt(
    EV,
    { agentId: '1', fullText: full(REASON_1, '0.44') },
    [{ agentId: '2', fullText: full(REASON_2, '0.61') }, { agentId: '3', fullText: full(REASON_3, '0.52') }],
    1,
  ));
}
