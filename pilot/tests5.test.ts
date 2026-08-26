// Mock-LLM tests for the tagged format, the five-bucket ledger and the routing
// rules of the three-rung ladder. Run:
//   node --experimental-transform-types --import ./register.mjs --test tests5.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { GenerateRequest, GenerateResponse, LLMClient } from '../src/types.js';
import { parseTagged, stripThink, TaggedParseError } from './tagged.js';
import { Ledger5, localTokens, summarize, MalformedCallError, type ProviderExtras } from './ledger5.js';
import { workerPrompt, aggregatorPrompt, debateRevisePrompt } from './prompts-draft.js';
import { runLadder, structureRecords, COMBINER_ROLES } from './structures5.js';

// ------------------------------------------------------------------ mock
type Script = (req: GenerateRequest, n: number) => { content: string; reasoning?: string; reasoningTokens?: number; finishReason?: string };

/** Mock client: token counts derived from the local tokenizer plus a fixed template overhead. */
function mockClient(script: Script, overhead = 7): LLMClient & { calls: GenerateRequest[] } {
  const calls: GenerateRequest[] = [];
  return {
    calls,
    async generate(req) {
      calls.push(req);
      const out = script(req, calls.length);
      const promptTokens = localTokens(req.systemPrompt) + localTokens(req.userPrompt) + overhead;
      const reasoningTokens = out.reasoningTokens ?? 0;
      const completionTokens = localTokens(out.content) + reasoningTokens;
      const extras: ProviderExtras = { reasoningTokens, reasoningText: out.reasoning ?? '', content: out.content, finishReason: out.finishReason ?? 'stop' };
      const res: GenerateResponse & { extras: ProviderExtras } = {
        text: out.content, toolCalls: [],
        usage: { promptTokens, completionTokens, totalTokens: promptTokens + completionTokens },
        costUsd: 0.001, modelId: 'mock', durationMs: 1, extras,
      };
      return res;
    },
  };
}

const good = (p: number, reason = 'Revenue beat, guidance cut, drift likely negative.') =>
  `<reason>\n${reason}\n</reason>\n<decision>\n${p}\n</decision>`;

const EV = { id: 'E1', text: 'FICTIONAL COMPANY X reported revenue of $1B, up 10%.' };

// --------------------------------------------------------------- parser
test('parseTagged: happy path, percent, leading dot', () => {
  const a = parseTagged(good(0.35));
  assert.equal(a.probability, 0.35);
  assert.equal(a.reason, 'Revenue beat, guidance cut, drift likely negative.');
  assert.equal(parseTagged('<reason>x</reason><decision>35%</decision>').probability, 0.35);
  assert.equal(parseTagged('<reason>x</reason><decision>.5</decision>').probability, 0.5);
  assert.equal(parseTagged('<reason>x</reason><decision>1</decision>').probability, 1);
});

test('parseTagged: rejects missing / duplicate / non-numeric', () => {
  assert.throws(() => parseTagged('no tags here 0.4'), TaggedParseError);
  assert.throws(() => parseTagged('<reason>a</reason>'), TaggedParseError);
  assert.throws(() => parseTagged('<reason>a</reason><decision>0.4</decision><decision>0.5</decision>'), TaggedParseError);
  assert.throws(() => parseTagged('<reason>a</reason><decision>about 0.4</decision>'), TaggedParseError);
  assert.throws(() => parseTagged('<reason>a</reason><decision>1.4</decision>'), TaggedParseError);
});

test('stripThink removes an inline think block', () => {
  const { visible, think } = stripThink('<think>hmm</think>' + good(0.2));
  assert.equal(think, 'hmm');
  assert.equal(parseTagged(visible).probability, 0.2);
});

// --------------------------------------------------------------- ledger
test('ledger identities hold on every call and buckets follow segment tags', async () => {
  const llm = mockClient(() => ({ content: good(0.6), reasoning: 'thinking...', reasoningTokens: 40 }));
  const led = new Ledger5('workers', 'E1', 'mock');
  const r = await led.call(llm, 'worker_1', workerPrompt(EV), { temperature: 0.7 });
  const rec = r.record;
  assert.equal(rec.read.D + rec.read.A + rec.read.B + rec.read.B_SELF, rec.read.providerPrompt);
  assert.equal(rec.write.C1 + rec.write.C2, rec.write.providerCompletion);
  assert.equal(rec.read.A, 0); assert.equal(rec.read.B, 0); assert.equal(rec.read.B_SELF, 0);
  assert.equal(rec.write.C2_hidden, 40);
  assert.ok(rec.write.C1 >= 1 && rec.write.C1 <= 3, `C1 should be the bare number, got ${rec.write.C1}`);
  assert.ok(rec.read.gapShare > 0 && rec.read.gapShare < 0.2, 'template overhead shows up as a small positive gap');
});

test('malformed → identical retry once → success is booked as attempt 2, attempt 1 flagged', async () => {
  const llm = mockClient((_, n) => ({ content: n === 1 ? 'I think 0.4' : good(0.4) }));
  const led = new Ledger5('workers', 'E1', 'mock');
  const r = await led.call(llm, 'worker_1', workerPrompt(EV), { temperature: 0.7 });
  assert.equal(r.parsed.probability, 0.4);
  assert.equal(led.records.length, 2);
  assert.equal(led.records[0].malformed, true);
  assert.equal(led.records[1].attempt, 2);
  assert.deepEqual(llm.calls[0], llm.calls[1], 'retry must re-send the identical prompt');
  const t = summarize(led.records);
  assert.equal(t.malformedAttempts, 1);
  assert.equal(t.calls, 1);
  assert.ok(t.wastedTokens > 0);
  assert.ok(Math.abs(t.costUsd - 0.002) < 1e-9, 'billing includes the failed attempt');
});

test('malformed twice → MalformedCallError, both attempts recorded', async () => {
  const llm = mockClient(() => ({ content: 'no tags' }));
  const led = new Ledger5('workers', 'E1', 'mock');
  await assert.rejects(led.call(llm, 'worker_1', workerPrompt(EV), { temperature: 0.7 }), MalformedCallError);
  assert.equal(led.records.length, 2);
  assert.ok(led.records.every(r => r.malformed));
});

// -------------------------------------------------------------- routing
test('aggregator receives ONLY <reason> texts (no numbers); B equals their local size', () => {
  const reasons = ['alpha reasoning', 'beta reasoning'];
  const p = aggregatorPrompt(EV, reasons);
  const user = p.user.map(s => s.text).join('');
  assert.ok(user.includes('alpha reasoning') && user.includes('beta reasoning'));
  assert.ok(!/0\.\d/.test(user), 'no probabilities may leak to the aggregator');
  const bLocal = p.user.filter(s => s.bucket === 'B').reduce((a, s) => a + localTokens(s.text), 0);
  assert.equal(bLocal, localTokens('alpha reasoning') + localTokens('beta reasoning'));
  assert.ok(p.system.some(s => s.bucket === 'A'), 'aggregator role is coordination instruction');
});

test('debate peers receive FULL answers (reason + decision) as B; own prior as B_SELF', () => {
  const own = { agentId: '1', fullText: good(0.3) };
  const peers = [{ agentId: '2', fullText: good(0.7, 'peer two') }];
  const p = debateRevisePrompt(EV, own, peers, 1);
  const b = p.user.filter(s => s.bucket === 'B').map(s => s.text).join('');
  assert.ok(b.includes('<decision>') && b.includes('0.7'), 'peers see the number too');
  const bs = p.user.filter(s => s.bucket === 'B_SELF').map(s => s.text).join('');
  assert.ok(bs.includes('0.3'));
});

// --------------------------------------------------------------- ladder
test('runLadder: repetition has A=B=0; aggregator commCost = whole aggregator call; debate B grows with peers×rounds', async () => {
  const llm = mockClient(req => {
    const role = req.metadata?.agentRole ?? '';
    const p = role.startsWith('worker') ? 0.4 : role === 'aggregator' ? 0.55 : 0.5;
    return { content: good(p, `${role} says something moderately long about the earnings.`) };
  });
  const res = await runLadder(llm, EV, 'mock', { agentCount: 3, debateRounds: 2, temperature: 0.7, aggregate: 'mean' });
  assert.ok(Math.abs(res.outcomes.repetition.probability! - 0.4) < 1e-9);
  assert.equal(res.outcomes.aggregator.probability, 0.55);
  assert.equal(res.outcomes.debate.probability, 0.5);
  // 3 workers + 1 aggregator + 3×2 debate = 10 calls
  assert.equal(res.ledger.records.length, 10);

  const rep = summarize(structureRecords(res.ledger.records, 'repetition'), COMBINER_ROLES);
  assert.equal(rep.A, 0); assert.equal(rep.B, 0); assert.equal(rep.commCost, 0);

  const agg = summarize(structureRecords(res.ledger.records, 'aggregator'), COMBINER_ROLES);
  const aggCall = res.ledger.records.find(r => r.agentRole === 'aggregator')!;
  assert.equal(agg.commCost, aggCall.totalTokens, 'combiner counts in full');
  assert.ok(aggCall.read.B > 0 && aggCall.read.A > 0);

  const deb = summarize(structureRecords(res.ledger.records, 'debate'), COMBINER_ROLES);
  const r1 = res.ledger.records.filter(r => r.structure === 'debate' && r.round === 1);
  const r2 = res.ledger.records.filter(r => r.structure === 'debate' && r.round === 2);
  assert.equal(r1.length, 3); assert.equal(r2.length, 3);
  for (const r of [...r1, ...r2]) {
    assert.ok(r.read.B > 0, 'each debater reads two peers');
    assert.ok(r.read.B_SELF > 0, 'own prior recorded separately');
    assert.equal(r.read.D + r.read.A + r.read.B + r.read.B_SELF, r.read.providerPrompt);
  }
  assert.equal(deb.B, [...r1, ...r2].reduce((a, r) => a + r.read.B, 0));
  assert.ok(deb.commCost > agg.commCost, 'debate talks more than the aggregator');
  assert.equal(deb.commCostInclSelf, deb.commCost + deb.B_SELF);
});

test('runLadder: a malformed worker drops all three structures, aggregator failure drops only itself', async () => {
  const bad = mockClient(req => (req.metadata?.agentRole === 'worker_2' ? { content: 'nope' } : { content: good(0.5) }));
  const r1 = await runLadder(bad, EV, 'mock', { agentCount: 3, debateRounds: 1, temperature: 0.7, aggregate: 'mean' });
  assert.ok(r1.outcomes.repetition.dropped && r1.outcomes.aggregator.dropped && r1.outcomes.debate.dropped);

  const badAgg = mockClient(req => (req.metadata?.agentRole === 'aggregator' ? { content: 'nope' } : { content: good(0.5) }));
  const r2 = await runLadder(badAgg, EV, 'mock', { agentCount: 3, debateRounds: 1, temperature: 0.7, aggregate: 'mean' });
  assert.equal(r2.outcomes.repetition.dropped, false);
  assert.equal(r2.outcomes.aggregator.dropped, true);
  assert.equal(r2.outcomes.debate.dropped, false);
  const t = summarize(structureRecords(r2.ledger.records, 'aggregator'), COMBINER_ROLES);
  assert.equal(t.malformedAttempts, 2);
});

// ---------------------------------------------------------------- report
test('buildReport: sanity rows all ok on a clean mock run', async () => {
  const { buildReport } = await import('./make-report5.js');
  const llm = mockClient(req => ({ content: good(0.5, `${req.metadata?.agentRole} reasoning text here.`), reasoningTokens: 20 }));
  const res = await runLadder(llm, EV, 'mockA', { agentCount: 3, debateRounds: 2, temperature: 0.7, aggregate: 'mean' });
  const md = buildReport(res.ledger.records, [{ modelKey: 'mockA', eventId: EV.id, outcomes: res.outcomes }]);
  const sanity = md.split('\n').find(l => l.startsWith('| mockA | ok'))!;
  assert.ok(sanity, 'sanity row present\n' + md);
  assert.ok(!sanity.includes('FAIL'), sanity);
  assert.ok(md.includes('| mockA | repetition |'));
  assert.ok(md.includes('none'), 'no malformed attempts');
});

// ------------------------------------------- D14 spread / collapse diagnostic
test('roundProbabilities capture per-round spread, and the report detects collapse', async () => {
  const { buildReport } = await import('./make-report5.js');
  const { spread, sd } = await import('./structures5.js');
  assert.ok(Math.abs(spread([0.3, 0.5, 0.7]) - 0.4) < 1e-9);
  assert.equal(spread([0.5, 0.5, 0.5]), 0);
  assert.ok(sd([0.5, 0.5, 0.5]) === 0);

  // Call order in runLadder: 3 workers, 1 aggregator, then 3 per debate round.
  const seq = [0.30, 0.50, 0.70,   // round 0 workers  -> spread 0.40
               0.55,               // aggregator
               0.50, 0.51, 0.52,   // round 1          -> spread 0.02
               0.50, 0.50, 0.51];  // round 2          -> spread 0.01
  let i = 0;
  const llm = mockClient(() => ({ content: good(seq[i++] ?? 0.5, 'reasoning text of adequate length here.') }));
  const res = await runLadder(llm, EV, 'collapsy', { agentCount: 3, debateRounds: 2, temperature: 0.7, aggregate: 'mean' });

  assert.equal(res.roundProbabilities.length, 3, 'round 0 + 2 revision rounds');
  assert.deepEqual(res.roundProbabilities[0], [0.30, 0.50, 0.70]);
  assert.ok(Math.abs(spread(res.roundProbabilities[0]) - 0.40) < 1e-9);
  assert.ok(spread(res.roundProbabilities[1]) < 0.05, 'round 1 collapsed');
  assert.ok(spread(res.roundProbabilities[2]) < spread(res.roundProbabilities[0]), 'collapse persists');

  const md = buildReport(res.ledger.records, [
    { modelKey: 'collapsy', eventId: EV.id, outcomes: res.outcomes, roundProbabilities: res.roundProbabilities },
  ]);
  assert.ok(md.includes('Agent spread per debate round'), 'section present');
  const row = md.split('\n').find(l => l.startsWith('| collapsy | 1 |'))!;
  assert.ok(row, 'spread row present\n' + md);
  assert.ok(row.includes('0.400'), 'round 0 spread shown: ' + row);
  assert.ok(row.includes('0.020'), 'round 1 spread shown: ' + row);
});

test('a worker failure still returns a usable (empty) roundProbabilities', async () => {
  const bad = mockClient(req => (req.metadata?.agentRole === 'worker_2' ? { content: 'nope' } : { content: good(0.5) }));
  const r = await runLadder(bad, EV, 'mock', { agentCount: 3, debateRounds: 1, temperature: 0.7, aggregate: 'mean' });
  assert.deepEqual(r.roundProbabilities, []);
});

// ------------------------------ D16 amendment: configurable retries + taxonomy
test('maxParseAttempts allows more than one retry; every attempt is recorded', async () => {
  let n = 0;
  const llm = mockClient(() => { n++; return { content: n < 4 ? 'no tags at all' : good(0.42) }; });
  const led = new Ledger5('workers', 'E1', 'mock');
  const r = await led.call(llm, 'worker_1', workerPrompt(EV), { temperature: 0.7, maxAttempts: 4 });
  assert.equal(r.parsed.probability, 0.42);
  assert.equal(led.records.length, 4, 'all four attempts booked');
  assert.equal(led.records[3].attempt, 4);
  assert.ok(led.records.slice(0, 3).every(x => x.malformed));
  // identical prompt on every retry - the property that keeps retrying legitimate
  assert.deepEqual(llm.calls[0], llm.calls[3], 'retries must re-send the identical prompt');
  const t = summarize(led.records);
  assert.equal(t.calls, 1);
  assert.equal(t.malformedAttempts, 3);
  assert.equal(t.retriedCalls, 1);
  assert.equal(t.maxAttemptUsed, 4);
});

test('maxAttempts is floored at 2 - the pre-registered minimum of one retry', async () => {
  let n = 0;
  const llm = mockClient(() => { n++; return { content: n === 1 ? 'nope' : good(0.5) }; });
  const led = new Ledger5('workers', 'E1', 'mock');
  const r = await led.call(llm, 'worker_1', workerPrompt(EV), { temperature: 0.7, maxAttempts: 1 });
  assert.equal(r.parsed.probability, 0.5, 'a retry happened despite maxAttempts=1');
  assert.equal(led.records.length, 2);
});

test('failure taxonomy separates truncation from a genuine format failure', async () => {
  const truncated = mockClient(() => ({ content: '<reason>I was cut off mid sen', finishReason: 'length' }));
  const led1 = new Ledger5('workers', 'E1', 'mock');
  await assert.rejects(led1.call(truncated, 'worker_1', workerPrompt(EV), { temperature: 0.7 }), MalformedCallError);
  assert.ok(led1.records.every(r => r.failureKind === 'truncated'), 'cap hit => truncated, not a model failure');
  assert.ok(led1.records.every(r => r.finishReason === 'length'));
  const t1 = summarize(led1.records);
  assert.equal(t1.failTruncated, 2);
  assert.equal(t1.failFormat, 0);

  const prose = mockClient(() => ({ content: 'I think it is about 60 percent.', finishReason: 'stop' }));
  const led2 = new Ledger5('workers', 'E1', 'mock');
  await assert.rejects(led2.call(prose, 'worker_1', workerPrompt(EV), { temperature: 0.7 }), MalformedCallError);
  const t2 = summarize(led2.records);
  assert.equal(t2.failFormat, 2);
  assert.equal(t2.failTruncated, 0);
});
