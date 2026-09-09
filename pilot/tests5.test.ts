// Mock-LLM tests for the tagged format, the five-bucket ledger and the routing
// rules of the three-rung ladder. Run:
//   node --experimental-transform-types --import ./register.mjs --test tests5.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { GenerateRequest, GenerateResponse, LLMClient } from '../src/types.js';
import { parseTagged, stripThink, TaggedParseError } from './tagged.js';
import { Ledger5, localTokens, summarize, splitTokens, MalformedCallError, type ProviderExtras } from './ledger5.js';
import { workerPrompt, aggregatorPrompt, debateRevisePrompt, reasonStatesOwnNumber } from './prompts.js';
import { runLadder, structureRecords, combine, COMBINER_ROLES } from './structures5.js';
import { buildStructureTokens } from './make-structure-tokens.js';
import { buildTranscript } from './make-transcript.js';
import { appendFileSync, existsSync, mkdtempSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { hostname, tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import {
  run, planWork, completedKeys, purgeOrphans, purgeFile, taskKey, outputPaths, ledgerKeyOf,
  checkpointEvents, WORK_ORDER_SEED, shuffleInPlace, baseEventId, MODELS, throttleIntervalFor,
  rawKeyOf, wantsFresh, isPidAlive, acquireLock, readLockInfo,
  encodeModelKey, decodeModelKey, overheadPathFor, readOverhead,
  type ModelSpec, type LadderClient, type RunOptions, type PurgeIO,
} from './run-pilot5.js';
import { CostCapError, Throttle, OpenRouterClient } from './openrouter-client.js';

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
      // Default to a SELF-CONSISTENT provider: the reported reasoning token count
      // matches the reasoning text actually returned. Tests that need the provider's
      // figures to disagree set reasoningTokens explicitly.
      const reasoningTokens = out.reasoningTokens ?? localTokens(out.reasoning ?? '');
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
  const TRACE = 'weighing the guidance raise against the one-time rebate and the soft European orders. '.repeat(3);
  const llm = mockClient(() => ({ content: good(0.6), reasoning: TRACE }));
  const led = new Ledger5('workers', 'E1', 'mock');
  const r = await led.call(llm, 'worker_1', workerPrompt(EV), { temperature: 0.7 });
  const rec = r.record;
  assert.equal(rec.read.D + rec.read.A + rec.read.B + rec.read.B_SELF, rec.read.providerPrompt);
  assert.equal(rec.write.C1 + rec.write.C2, rec.write.providerCompletion);
  assert.equal(rec.read.A, 0); assert.equal(rec.read.B, 0); assert.equal(rec.read.B_SELF, 0);
  assert.equal(rec.write.C2_hidden, localTokens(TRACE), 'a self-consistent trace is attributed exactly');
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
  // 3 workers + 1 aggregator + 3x2 debate = 10 structure calls, PLUS the extra
  // single-agent runs drawn for compute matching (D23), which live under
  // structure='repetition'.
  const structureCalls = res.ledger.records.filter(r => r.structure !== 'repetition');
  assert.equal(structureCalls.length, 10);
  assert.ok(res.ledger.records.length > 10, 'compute matching must add repetition runs');

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
// FINDING 1 (2026-09-01). The taxonomy the thesis quotes (make-report5.ts D16:
// "Infrastructure errors ... never appear here; the client retries them itself")
// was not what the code did: EVERY rejection became a `dropped` outcome and
// runLadder returned normally, so run-pilot5 wrote an outcomes line — the very
// marker resume reads as "done". A provider outage was therefore recorded as a
// model failure AND made permanent. Reproduced before the fix: one injected
// ECONNRESET on worker_2 left the event on disk with all three rungs
// dropped=true and 2 ledger records, and the resume said completed=0 skipped=2.
test('runLadder: a malformed answer is a result (drop), an outage is not (it propagates)', async () => {
  const P = { agentCount: 3, debateRounds: 1, temperature: 0.7, aggregate: 'mean' as const };
  const failingAt = (role: string, err: () => Error) => mockClient(req => {
    if ((req.metadata?.agentRole ?? '') === role || role === 'rep_*' && (req.metadata?.agentRole ?? '').startsWith('rep_')) throw err();
    return { content: good(0.4) };
  });
  const net = () => new Error('fetch failed: ECONNRESET after 4 attempts');

  // All three rungs, plus the compute-matching draws — every place a rejection
  // used to be swallowed. `rep_*` covers the extra baseline runs, where an
  // outage does not even produce a drop: it silently SHORTENS the
  // communication-free baseline (measured: debate compared against 945 of
  // 3,948 tokens instead of 3,765, a 4x understated null hypothesis).
  for (const role of ['worker_2', 'aggregator', 'debater_2', 'rep_*']) {
    await assert.rejects(
      () => runLadder(failingAt(role, net), EV, 'mock', P),
      /ECONNRESET/,
      `an outage at ${role} must reach the caller, not be booked as a drop`,
    );
  }
  // The cost cap must keep its IDENTITY across the ladder: run-pilot5 stops the
  // whole session on `instanceof CostCapError`, which a re-wrapped message
  // (`aggregator error: ...`, as before) silently defeated.
  await assert.rejects(
    () => runLadder(failingAt('aggregator', () => new CostCapError('cost cap $2.0 reached')), EV, 'mock', P),
    (e: unknown) => e instanceof CostCapError,
    'CostCapError must arrive as itself',
  );

  // And the other half of the distinction: a MALFORMED answer is still a
  // result. It drops its structure, keeps the others, and returns normally.
  const malformed = mockClient(req => ({
    content: (req.metadata?.agentRole ?? '') === 'aggregator' ? 'no tags at all' : good(0.4),
  }));
  const res = await runLadder(malformed, EV, 'mock', P);
  assert.equal(res.outcomes.aggregator.dropped, true);
  assert.match(res.outcomes.aggregator.dropReason!, /unusable after 2 attempts/);
  assert.equal(res.outcomes.repetition.dropped, false);
  assert.equal(res.outcomes.debate.dropped, false);
});

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

// --------------------- C2 (D18): fixed template overhead is booked to D only
test('template overhead goes entirely to D; identity still holds; A and B unaffected', async () => {
  const OVERHEAD = 9;
  // mock bills local tokens + OVERHEAD, exactly like a chat template would
  const llm = mockClient(() => ({ content: good(0.5, 'peer reasoning text of some length here.') }), OVERHEAD);
  const withOh = new Ledger5('debate', 'E1', 'mock', [], OVERHEAD);
  const withoutOh = new Ledger5('debate', 'E1', 'mock', [], 0);
  const own = { agentId: '1', fullText: good(0.4, 'my earlier reasoning') };
  const peers = [{ agentId: '2', fullText: good(0.6, 'peer two reasoning') }];
  const p = debateRevisePrompt(EV, own, peers, 1);

  const a = (await withOh.call(llm, 'debater_1', p, { temperature: 0.7, round: 1 })).record;
  const b = (await withoutOh.call(llm, 'debater_1', p, { temperature: 0.7, round: 1 })).record;

  // identity holds in both cases
  for (const r of [a, b]) {
    assert.equal(r.read.D + r.read.A + r.read.B + r.read.B_SELF, r.read.providerPrompt);
  }
  assert.equal(a.read.templateOverhead, OVERHEAD);
  assert.equal(b.read.templateOverhead, 0);
  // the overhead lands in D, and A/B/B_SELF are NOT inflated by it
  assert.ok(a.read.D > b.read.D, 'D absorbs the fixed overhead');
  assert.ok(a.read.A <= b.read.A, 'A must not grow when overhead is attributed to D');
  assert.ok(a.read.B <= b.read.B, 'B must not grow when overhead is attributed to D');
  // and the residual gap is ~0 once the true overhead is removed
  assert.ok(Math.abs(a.read.gapShare) < 1e-9, `residual should vanish, got ${a.read.gapShare}`);
  assert.ok(b.read.gapShare > 0, 'without calibration the whole overhead shows up as gap');
});

test('measureTemplateOverhead recovers the mock template cost', async () => {
  const { measureTemplateOverhead } = await import('./ledger5.js');
  const llm = mockClient(() => ({ content: 'x' }), 11);
  const r = await measureTemplateOverhead(llm);
  assert.equal(r.overhead, 11);
  assert.equal(r.providerPrompt - r.localCount, 11);
});

test('splitTokens sums exactly to the whole message and avoids boundary over-counting', async () => {
  const { splitTokens, localTokens: lt } = await import('./ledger5.js');
  const segs = [
    { bucket: 'A' as const, text: '--- Analyst 2 ---\n', origin: 'h2' },
    { bucket: 'B' as const, text: 'Backlog grew to $3.1B, up 7 % sequentially.', origin: 'p2' },
    { bucket: 'A' as const, text: '\n\n', origin: 'sep' },
    { bucket: 'A' as const, text: '--- Analyst 3 ---\n', origin: 'h3' },
    { bucket: 'B' as const, text: 'Guidance assumes no improvement in Europe.', origin: 'p3' },
  ];
  const whole = segs.map(s => s.text).join('');
  const parts = splitTokens(segs);
  assert.equal(parts.reduce((a, b) => a + b, 0), lt(whole), 'parts must sum to the whole');
  const naive = segs.reduce((a, s) => a + lt(s.text), 0);
  assert.ok(naive >= lt(whole), 'per-segment tokenisation over-counts at boundaries');
});

// ---- hidden/visible split from local proportions, not the provider subtraction
test('C2 hidden/visible comes from local proportions and sums to completion', async () => {
  // Provider reports a reasoning count that does NOT decompose with the visible
  // text (the pattern measured on Qwen/DeepInfra): reasoning is nearly the whole
  // completion, leaving an impossible 5 tokens for a much longer answer.
  const reasoning = 'thinking step '.repeat(60);          // ~120 local tokens
  const content = good(0.42, 'visible reasoning text here that is clearly longer than five tokens.');
  const llm: LLMClient = {
    async generate() {
      const res: GenerateResponse & { extras: ProviderExtras } = {
        text: content, toolCalls: [],
        usage: { promptTokens: 100, completionTokens: 200, totalTokens: 300 },
        costUsd: 0, modelId: 'mock', durationMs: 1,
        extras: { reasoningTokens: 195, reasoningText: reasoning, content, finishReason: 'stop' },
      };
      return res;
    },
  };
  const led = new Ledger5('workers', 'E1', 'mock');
  const r = await led.call(llm, 'worker_1', workerPrompt(EV), { temperature: 0.7 });
  const w = r.record.write;
  assert.equal(w.C1 + w.C2, w.providerCompletion, 'identity must still hold exactly');
  assert.equal(w.C2_hidden + w.C2_visible, w.C2);
  // the naive subtraction would have left only 200-195 = 5 tokens for the answer
  assert.ok(w.C2_visible > 20, `visible must not be squeezed to the subtraction residual, got ${w.C2_visible}`);
  // and the split should track the local proportion of the two texts
  const { localTokens: lt } = await import('./ledger5.js');
  const expectedHiddenShare = lt(reasoning) / (lt(reasoning) + lt(content));
  const actualHiddenShare = w.C2_hidden / w.providerCompletion;
  assert.ok(Math.abs(expectedHiddenShare - actualHiddenShare) < 0.02,
    `hidden share ${actualHiddenShare.toFixed(3)} should track local proportion ${expectedHiddenShare.toFixed(3)}`);
});

test('non-reasoning models are unaffected: no trace means the old path', async () => {
  const llm = mockClient(() => ({ content: good(0.5, 'plain answer'), reasoningTokens: 0 }));
  const led = new Ledger5('workers', 'E1', 'mock');
  const r = await led.call(llm, 'worker_1', workerPrompt(EV), { temperature: 0.7 });
  const w = r.record.write;
  assert.equal(w.C2_hidden, 0);
  assert.equal(w.C1 + w.C2, w.providerCompletion);
});

// ------------------------------------------------- D23 compute matching
test('compute matching: baseline is drawn up to the largest budget, prefix FITS, residual reported', async () => {
  const llm = mockClient(req => ({
    content: good(0.5, `${req.metadata?.agentRole} reasoning of some length to spend tokens.`),
  }));
  const res = await runLadder(llm, EV, 'mock', {
    agentCount: 3, debateRounds: 2, temperature: 0.7, aggregate: 'mean',
  });
  assert.equal(res.matched.length, 2, 'one comparison per communicating structure');

  const byName = Object.fromEntries(res.matched.map(m => [m.structure, m]));
  const debate = byName['debate'], agg = byName['aggregator'];

  // the debate is the most expensive, so it sets the ceiling
  assert.ok(debate.structureTokens > agg.structureTokens, 'debate should cost more than the aggregator');

  // extra runs were actually drawn beyond the three shared workers
  assert.ok(res.repetitionRuns.length > 3, `expected extra runs, got ${res.repetitionRuns.length}`);

  for (const m of res.matched) {
    // FITS rule: never over budget, and the residual is exactly the shortfall
    assert.ok(m.matchedTokens <= m.structureTokens, `${m.structure}: matched must not exceed budget`);
    assert.equal(m.residualTokens, m.structureTokens - m.matchedTokens);
    // the prefix really is a prefix, in generation order
    const expected = res.repetitionRuns.slice(0, m.runsUsed).reduce((a, r) => a + r.totalTokens, 0);
    assert.equal(m.matchedTokens, expected, `${m.structure}: matched tokens must be the prefix sum`);
    // adding one more run would have exceeded the budget (the prefix is maximal)
    const next = res.repetitionRuns[m.runsUsed];
    if (next) assert.ok(m.matchedTokens + next.totalTokens > m.structureTokens, `${m.structure}: prefix must be maximal`);
    assert.equal(m.probability, combine(res.repetitionRuns.slice(0, m.runsUsed).map(r => r.probability), 'mean'));
  }
  // a cheaper structure gets a shorter prefix than a more expensive one
  assert.ok(agg.runsUsed <= debate.runsUsed, 'the cheaper structure is matched with fewer runs');
});

test('compute matching: the first three repetition runs ARE the shared workers', async () => {
  let n = 0;
  const probs = [0.10, 0.20, 0.30];
  const llm = mockClient(req => {
    const role = req.metadata?.agentRole ?? '';
    const p = role.startsWith('worker') ? probs[n++] : 0.5;
    return { content: good(p, 'text of a reasonable length for token accounting purposes.') };
  });
  const res = await runLadder(llm, EV, 'mock', { agentCount: 3, debateRounds: 1, temperature: 0.7, aggregate: 'mean' });
  assert.deepEqual(res.repetitionRuns.slice(0, 3).map(r => r.probability), probs,
    'the shared workers must head the repetition sequence, in order');
});

test('compute matching: a dropped structure gets no comparison', async () => {
  const badAgg = mockClient(req => (req.metadata?.agentRole === 'aggregator' ? { content: 'nope' } : { content: good(0.5, 'some reasoning text here') }));
  const res = await runLadder(badAgg, EV, 'mock', { agentCount: 3, debateRounds: 1, temperature: 0.7, aggregate: 'mean' });
  assert.equal(res.outcomes.aggregator.dropped, true);
  assert.ok(!res.matched.some(m => m.structure === 'aggregator'), 'no matched comparison for a dropped structure');
  assert.ok(res.matched.some(m => m.structure === 'debate'), 'the surviving structure still gets one');
});

/** The row from the SANITY CHECKS table — several tables start with the model name. */
function sanityRow(md: string, model: string): string {
  const lines = md.split('\n');
  const start = lines.findIndex(l => l.startsWith('## Sanity checks'));
  assert.ok(start >= 0, 'sanity section missing');
  const row = lines.slice(start).find(l => l.startsWith(`| ${model} |`));
  assert.ok(row, `no sanity row for ${model}`);
  return row!;
}

// ---- D24: a host that collapses identical concurrent requests must be caught
test('report FAILS the independent-samples check when responses are identical', async () => {
  const { buildReport } = await import('./make-report5.js');
  // every call returns the SAME text, as DeepInfra did for DeepSeek V3
  const llm = mockClient(() => ({ content: good(0.5, 'an identical answer every single time') }));
  const res = await runLadder(llm, EV, 'dupmodel', { agentCount: 3, debateRounds: 1, temperature: 0.7, aggregate: 'mean' });
  const md = buildReport(res.ledger.records, [{ modelKey: 'dupmodel', eventId: EV.id, outcomes: res.outcomes } as any]);
  const row = sanityRow(md, 'dupmodel');
  assert.ok(row.includes('FAIL'), 'identical worker answers must fail the check: ' + row);
  assert.ok(/\d+\/\d+ distinct/.test(row), 'the row should report how many were distinct: ' + row);
});

test('report PASSES the independent-samples check when responses differ', async () => {
  const { buildReport } = await import('./make-report5.js');
  let n = 0;
  const llm = mockClient(() => ({ content: good(0.4 + 0.01 * n, `distinct answer number ${n++}`) }));
  const res = await runLadder(llm, EV, 'okmodel', { agentCount: 3, debateRounds: 1, temperature: 0.7, aggregate: 'mean' });
  const md = buildReport(res.ledger.records, [{ modelKey: 'okmodel', eventId: EV.id, outcomes: res.outcomes } as any]);
  const row = sanityRow(md, 'okmodel');
  assert.ok(!row.includes('distinct'), 'distinct answers must pass: ' + row);
});


// ==========================================================================
// Regression tests for the 2026-08-29 edge-case audit. Each one reproduces a
// bug that ran without error and returned a plausible wrong number. See
// scaffold/EDGE_AUDIT_2026-08-29.md for the full finding.
// ==========================================================================

// ---- pct-ambiguity-100x
test('audit: a probability written with BOTH decimals and a percent sign is refused', () => {
  const wrap = (d: string) => '<reason>because</reason><decision>' + d + '</decision>';
  // Unambiguous, must parse.
  const okCases: [string, number][] = [['35%', 0.35], ['100%', 1], ['0%', 0], ['1%', 0.01], ['50 %', 0.5]];
  for (const [written, expected] of okCases) {
    assert.equal(parseTagged(wrap(written)).probability, expected, written);
  }
  // Ambiguous by a factor of 100: "0.35%" is 0.0035 as written, but a model that
  // meant 0.35 writes exactly this. Guessing either way fabricates a forecast.
  for (const written of ['0.35%', '0.5%', '1.0%']) {
    assert.throws(() => parseTagged(wrap(written)), TaggedParseError, written + ' must be refused');
  }
});

// ---- combine-unrounded-mean
test('audit: combine() returns one canonical double, independent of agent order', () => {
  // Three agents unanimously saying 0.7 used to produce 0.6999999999999998,
  // which the analysis then binned one decile TOO LOW.
  assert.equal(combine([0.7, 0.7, 0.7], 'mean'), 0.7);
  assert.equal(combine([0.65, 0.7, 0.75], 'mean'), 0.7);
  assert.equal(combine([0.4, 0.5], 'median'), 0.45);

  // Order-independence over the whole answer grid the models actually use.
  const grid: number[] = [];
  for (let k = 6; k <= 14; k++) grid.push(k * 0.05);
  const byMultiset = new Map<string, Set<number>>();
  for (const a of grid) for (const b of grid) for (const c of grid) {
    const key = [a, b, c].slice().sort((x, y) => x - y).join(',');
    if (!byMultiset.has(key)) byMultiset.set(key, new Set<number>());
    byMultiset.get(key)!.add(combine([a, b, c], 'mean'));
  }
  let orderDependent = 0;
  let belowEdge = 0;
  for (const [, vals] of byMultiset) {
    if (vals.size > 1) orderDependent++;
    const v = [...vals][0];
    const decile = Math.round(v * 10) / 10;
    if (Math.abs(v - decile) < 1e-9 && v < decile) belowEdge++;
  }
  assert.equal(orderDependent, 0, 'the forecast must be a function of the answers alone');
  assert.equal(belowEdge, 0, 'a mean on a decile edge must never be emitted below it');
});

// ---- splittokens-clamp-overcounts
test('audit: splitTokens parts always add up to the whole', () => {
  const cases: string[][] = [
    ['hello ', 'world'],
    ['', 'only the second segment'],
    ['   ', 'leading whitespace segment'],
    ['a', 'b', 'c', 'd', 'e'],
    ['--- Analyst 1 ---', ' some reasoning text here', '--- Analyst 2 ---', ' more reasoning'],
    ['one'],
  ];
  for (const texts of cases) {
    const segs = texts.map((text, i) => ({ bucket: 'D' as const, text, label: 'seg' + i }));
    const parts = splitTokens(segs);
    const whole = localTokens(texts.join(''));
    assert.equal(parts.reduce((a, b) => a + b, 0), whole, JSON.stringify(texts));
    assert.ok(parts.every(x => x >= 0), 'no negative segment: ' + JSON.stringify(texts));
    assert.equal(parts.length, texts.length);
  }
});

// ---- malformed-attempt-tokens-dropped
test('audit: a discarded retry stays out of the buckets but inside the billed totals', async () => {
  // Attempt 1 is cut off by the token cap, attempt 2 succeeds.
  let n = 0;
  const llm = mockClient(() => (++n === 1
    ? { content: '<reason>still writing when the cap hit', finishReason: 'length' }
    : { content: good(0.55) }));
  const led = new Ledger5('aggregator', EV.id, 'mock');
  await led.call(llm, 'aggregator', aggregatorPrompt(EV, ['alpha', 'beta', 'gamma']), { temperature: 0.7 });

  const billed = led.records.reduce((a, r) => a + r.totalTokens, 0);
  assert.equal(led.records.length, 2, 'both attempts are recorded');
  const t = summarize(led.records, ['aggregator']);

  // The buckets cover usable calls only: a malformed answer has no parseable
  // decision, so there is no honest C1 to book.
  assert.equal(t.calls, 1);
  assert.ok(t.total < billed, 'the bucket total excludes the discarded attempt');
  // The money was spent either way, so the billed figures must match reality.
  assert.equal(t.billedTotal, billed);
  assert.equal(t.billedCommCost, billed, 'the whole combiner call is communication cost');
  assert.equal(t.billedPrompt + t.billedCompletion, billed);
});

// ---- budget-excludes-retries
test('audit: the compute-match budget counts every attempt that was billed', async () => {
  let burned = false;
  const llm = mockClient((req) => {
    const role = String((req as any).metadata?.agentRole ?? '');
    if (role.startsWith('debater_') && !burned) {
      burned = true;
      return { content: '<reason>cut off mid sentence', finishReason: 'length' };
    }
    return { content: good(0.45, 'a reasoning section of a realistic length for this test') };
  });
  const res = await runLadder(llm, EV, 'mock', { agentCount: 3, debateRounds: 1, temperature: 0.7, aggregate: 'mean' });

  const debateMatch = res.matched.find(mm => mm.structure === 'debate');
  assert.ok(debateMatch, 'the debate should have a matched comparison');
  const billedDebate = res.ledger.records
    .filter(r => r.structure === 'workers' || r.structure === 'debate')
    .reduce((a, r) => a + r.totalTokens, 0);
  assert.equal(debateMatch!.structureTokens, billedDebate,
    'the budget must be what the structure really spent, retries included');
  assert.ok(res.ledger.records.some(r => r.malformed), 'the test must actually produce a discarded attempt');
});

// ---- dropped-debate-round-data
test('audit: a debate that drops leaves no partial rounds behind', async () => {
  // Round 1 succeeds for everyone, round 2 fails: the event is dropped, so its
  // half-finished rounds must not enter the convergence table.
  // The request metadata carries the agent role but not the round number, so
  // the round is identified by counting: with 3 agents, calls 1-3 are round 1
  // and calls 4-6 are round 2.
  let debaterCalls = 0;
  const llm = mockClient((req) => {
    const role = String((req as any).metadata?.agentRole ?? '');
    if (role.startsWith('debater_')) {
      debaterCalls++;
      if (debaterCalls > 3) return { content: 'no tags at all' };
    }
    return { content: good(0.5, 'a reasoning section for this round') };
  });
  const res = await runLadder(llm, EV, 'mock', { agentCount: 3, debateRounds: 2, temperature: 0.7, aggregate: 'mean' });
  assert.ok(debaterCalls > 3, 'round 1 must have completed before round 2 failed');
  assert.equal(res.outcomes.debate.dropped, true, 'the debate must be dropped');
  assert.equal(res.roundProbabilities.length, 1,
    'only round 0 (the shared workers) may survive a dropped debate');
});

// ---- structure-tokens emitter
test('structure_tokens: combiner lines book the whole call, worker lines book A + B', async () => {
  const llm = mockClient(() => ({ content: good(0.5, 'a reasoning section of a realistic length for this test') }));
  const res = await runLadder(llm, EV, 'mock', { agentCount: 3, debateRounds: 1, temperature: 0.7, aggregate: 'mean' });
  const rows = buildStructureTokens(res.ledger.records);
  const line = (s: string) => rows.find(r => r.eventId === EV.id && r.modelKey === 'mock' && r.structure === s)!;

  // The aggregator exists ONLY to combine other agents, so its entire call is
  // communication cost (working doc §5.2) — the evidence it re-reads is not a
  // cost a lone forecaster would ever have paid. Every other role contributes
  // A + B, the coordination instruction plus the peer messages it read.
  const aggCalls = res.ledger.records.filter(r => r.agentRole === 'aggregator');
  assert.equal(aggCalls.length, 1, 'one aggregating call per event');
  const aggWholeCall = aggCalls[0].totalTokens;
  const aggAB = aggCalls[0].read.A + aggCalls[0].read.B;
  // Without this the two rules would be indistinguishable and the test could
  // not fail: it is the D evidence block that makes the whole call larger.
  assert.ok(aggWholeCall > aggAB, 'the whole combiner call must exceed its own A + B');

  const sumAB = (structures: string[]) => res.ledger.records
    .filter(r => structures.includes(r.structure))
    .reduce((a, r) => a + r.read.A + r.read.B, 0);

  // repetition = the shared workers alone. They receive no coordination
  // instruction and read no peer text, so A + B is 0 — that zero is what makes
  // this rung the communication-free baseline the other two are measured against.
  assert.equal(sumAB(['workers']), 0, 'workers must read no coordination text at all');
  assert.equal(line('repetition').commCost, 0, 'a worker-only line carries A + B');

  // aggregator = the same workers (A + B = 0) plus the whole combiner call.
  assert.equal(line('aggregator').commCost, sumAB(['workers']) + aggWholeCall,
    'the aggregator line carries the whole combiner call');

  // debate = the same workers plus debaters, who are NOT combiners: real,
  // non-zero A + B, so this is the branch the repetition line cannot exercise.
  assert.ok(sumAB(['debate']) > 0, 'debaters must actually read peer messages');
  assert.equal(line('debate').commCost, sumAB(['workers', 'debate']),
    'a non-combiner line carries A + B even when A + B is large');

  // Second event, same model, one ledger. `structureRecords` filters on the
  // structure field alone, so if the emitter did not narrow to one event first
  // it would fold E2's shared workers into E1's totals. The result would still
  // be internally consistent and would still satisfy every identity above —
  // just roughly twice the true cost.
  const res2 = await runLadder(llm, { id: 'E2', text: EV.text }, 'mock',
    { agentCount: 3, debateRounds: 1, temperature: 0.7, aggregate: 'mean' });
  const both = buildStructureTokens([...res.ledger.records, ...res2.ledger.records]);
  assert.equal(both.length, 6, 'two events x one model x three structures');
  assert.deepEqual(both.find(r => r.eventId === EV.id && r.structure === 'aggregator'), line('aggregator'),
    'a second event must not change the first event numbers');
});

// ==========================================================================
// RESUME / LEVEL-PREFIX / SELF-REPORTING  (run-pilot5.ts)
//
// The real run is 800 events x 2 models x 2 evidence levels = 3,200 ladder
// runs, ~$31.50 and ~1.7 days of wall clock at concurrency 6. Over that span a
// sleep, a network blip, a provider outage or a Ctrl-C is a certainty, so the
// behaviour under interruption is a property of the instrument, not an edge
// case. These tests pin it down. None of them touches the network: the ladder
// is driven by the same mock client as everything above.
// ==========================================================================

/** Scratch root. NEVER pilot5/ — that holds the committed pilot data. */
const SCRATCH = process.env.PILOT5_TEST_OUTDIR ?? tmpdir();
const scratchDir = (label: string) => mkdtempSync(join(SCRATCH, `pilot5-${label}-`));

/** A model spec whose slug and provider are never used, because the client is injected. */
const MOCK_MODEL: ModelSpec = {
  key: 'mock', slug: 'mock/mock', provider: 'mock', mode: 'none',
  arm: 'non-reasoning', evidence: 'all', priceIn: 0, priceOut: 0,
};

/**
 * A client the runner can drive: the mock generator above plus the two things
 * `run()` needs from a real OpenRouterClient — `beginStructureRun`, and the
 * raw request log it writes itself. The raw log matters here because
 * `runs_raw.jsonl` has to be purged on resume exactly like the ledger, and it
 * names its key fields `evidence_id` / `model_key`, not `eventId` / `modelKey`.
 */
function mockLadderClient(
  rawPath: string,
  modelKey = 'mock',
  /** Optional outage injector: return an Error to make THIS call fail. */
  failOn: (evidenceId: string, agentRole: string) => Error | null = () => null,
): LadderClient {
  let evidenceId: string | undefined;
  let n = 0;
  const inner = mockClient(req => ({
    // Distinct text per call: identical answers at temperature > 0 are the
    // D24 request-collapsing failure, and the report checks for exactly that.
    content: good(0.4 + (n % 5) / 100, `${req.metadata?.agentRole ?? 'x'} reasoning number ${n} of a realistic length.`),
  }));
  return {
    beginStructureRun(_configName: string, evId: string) { evidenceId = evId; },
    async generate(req: GenerateRequest) {
      n++;
      const boom = failOn(evidenceId ?? '', req.metadata?.agentRole ?? '');
      if (boom) throw boom;
      const res = await inner.generate(req);
      appendFileSync(rawPath, JSON.stringify({
        ok: true, evidence_id: evidenceId, model_key: modelKey,
        agent_role: req.metadata?.agentRole ?? null, call: n,
      }) + '\n');
      return res;
    },
  };
}

const EVENTS = [
  { id: 'AAPL_2024Q1_L1', text: 'FICTIONAL COMPANY A reported revenue of $1B, up 10%.' },
  { id: 'MSFT_2024Q1_L1', text: 'FICTIONAL COMPANY B reported revenue of $2B, down 4%.' },
];

const runMock = (dir: string, opts: Partial<RunOptions> = {}) =>
  run(['mock'], null, {
    outDir: dir, level: 1, mode: 'full', concurrency: 2, repeats: 1,
    events: EVENTS, models: [MOCK_MODEL], packs: [],
    makeClient: () => mockLadderClient(outputPaths(dir, opts.level ?? 1, opts.mode ?? 'full').raw),
    ...opts,
  });

const lines = (p: string) => (existsSync(p) ? readFileSync(p, 'utf-8').split(/\r?\n/).filter(Boolean) : []);

// ---- 1. completedKeys
test('resume: completedKeys treats level and repeat as part of the key', () => {
  const keys = completedKeys([
    JSON.stringify({ modelKey: 'qwen3-32b@on', eventId: 'AAPL_2024Q1_L1' }),
    JSON.stringify({ modelKey: 'qwen3-32b@on', eventId: 'AAPL_2024Q1_L2' }),
    JSON.stringify({ modelKey: 'qwen3-32b@on', eventId: 'AAPL_2024Q1_L1#r2' }),
    JSON.stringify({ modelKey: 'deepseek-chat-v3-0324', eventId: 'AAPL_2024Q1_L1' }),
    '',                                    // trailing newline of an appended file
    '{"modelKey":"qwen3-32b@on","eventId"',  // a kill mid-append: torn last line
  ]);
  // 4 well-formed lines, and NOT ONE of them may collapse into another. The
  // level suffix (_L2) and the repeat suffix (#r2) are the whole reason the
  // pair (eventId, modelKey) is a complete natural key.
  assert.equal(keys.size, 4);
  assert.ok(keys.has(taskKey('AAPL_2024Q1_L1', 'qwen3-32b@on')));
  assert.ok(keys.has(taskKey('AAPL_2024Q1_L2', 'qwen3-32b@on')));
  assert.ok(keys.has(taskKey('AAPL_2024Q1_L1#r2', 'qwen3-32b@on')));
  assert.ok(keys.has(taskKey('AAPL_2024Q1_L1', 'deepseek-chat-v3-0324')));
  // The torn line is not completed work. Counting it would skip an event that
  // never produced an outcome, i.e. silently lose it from the dataset.
  assert.equal([...keys].filter(k => k.includes('AAPL_2024Q1_L1')).length, 3);
});

// ---- 2. planWork
test('resume: planWork skips exactly the done tasks and keeps the rest in order', () => {
  const mk = (key: string, id: string) => ({ m: { key }, ev: { id } });
  const tasks = [
    mk('qwen', 'E1'), mk('qwen', 'E2'), mk('qwen', 'E3'),
    mk('deepseek', 'E1'), mk('deepseek', 'E2'), mk('deepseek', 'E3'),
  ];
  const done = new Set([taskKey('E2', 'qwen'), taskKey('E1', 'deepseek'), taskKey('E3', 'deepseek')]);
  const { todo, skipped } = planWork(tasks, done);

  assert.deepEqual(todo.map(t => `${t.ev.id}/${t.m.key}`), ['E1/qwen', 'E3/qwen', 'E2/deepseek']);
  assert.deepEqual(skipped.map(t => `${t.ev.id}/${t.m.key}`), ['E2/qwen', 'E1/deepseek', 'E3/deepseek']);
  assert.equal(todo.length + skipped.length, tasks.length, 'no task may be lost or duplicated');
  // Same event id under a different model is a DIFFERENT run. If the key were
  // the event alone, E1/qwen would be skipped because E1/deepseek is done.
  assert.ok(todo.some(t => t.ev.id === 'E1' && t.m.key === 'qwen'));
});

// ---- 3. purgeOrphans
test('resume: purgeOrphans drops a half-run and keeps every record of a finished one', () => {
  // E1 finished (5 records). E2 was killed after 3 of its records were written
  // and before its outcomes line, so it has NO outcomes line.
  const records = [
    { eventId: 'E1', modelKey: 'mock', agentRole: 'worker_1' },
    { eventId: 'E2', modelKey: 'mock', agentRole: 'worker_1' },
    { eventId: 'E1', modelKey: 'mock', agentRole: 'worker_2' },
    { eventId: 'E2', modelKey: 'mock', agentRole: 'worker_2' },
    { eventId: 'E1', modelKey: 'mock', agentRole: 'worker_3' },
    { eventId: 'E2', modelKey: 'mock', agentRole: 'aggregator' },
    { eventId: 'E1', modelKey: 'mock', agentRole: 'aggregator' },
    { eventId: 'E1', modelKey: 'mock', agentRole: 'debater_1' },
    // Same event id, other model: finished there, so it must survive.
    { eventId: 'E2', modelKey: 'other', agentRole: 'worker_1' },
  ];
  const done = new Set([taskKey('E1', 'mock'), taskKey('E2', 'other')]);
  const { kept, removed } = purgeOrphans(records, done);

  // The counts are chosen so a no-op (`kept = records, removed = []`) gives
  // 9/0 and a purge-everything gives 0/9: neither can pass by accident, and
  // neither matches on length alone.
  assert.equal(removed.length, 3);
  assert.equal(kept.length, 6);
  assert.equal(kept.length + removed.length, records.length);
  assert.ok(removed.every(r => r.eventId === 'E2' && r.modelKey === 'mock'));
  assert.equal(kept.filter(r => r.eventId === 'E1').length, 5, 'a finished event keeps ALL its records');
  assert.equal(kept.filter(r => r.modelKey === 'other').length, 1, 'the other model finished E2 and is untouched');

  // The raw log names the same two fields differently, and the calibration
  // probe is logged before any event begins — no evidence_id at all. That
  // record belongs to no event, so it cannot be double-counted and is kept.
  const raw = [
    { evidence_id: 'E1', model_key: 'mock' },
    { evidence_id: 'E2', model_key: 'mock' },
    { model_key: 'mock' },
  ];
  const r2 = purgeOrphans(raw, done, rawKeyOf);
  assert.equal(r2.removed.length, 1);
  assert.equal(r2.kept.length, 2);
});

// ---- 4. the property that matters most: a finished run is a no-op
test('resume: re-running a finished run does nothing and leaves the files byte-identical', async () => {
  const dir = scratchDir('noop');
  const P = outputPaths(dir, 1);
  const first = await runMock(dir);
  assert.equal(first.completed, 2);
  assert.equal(first.skipped, 0);

  const before = [P.ledger, P.outcomes, P.raw].map(p => readFileSync(p));
  const second = await runMock(dir);

  assert.equal(second.completed, 0, 'not one paid call may be repeated');
  assert.equal(second.skipped, 2);
  assert.equal(second.purgedLedger, 0);
  assert.equal(second.purgedRaw, 0);
  // Bytes, not counts: a purge that re-serialised the survivors, or an append
  // of a re-ordered duplicate, would keep every count identical.
  const after = [P.ledger, P.outcomes, P.raw].map(p => readFileSync(p));
  for (let i = 0; i < before.length; i++) {
    assert.ok(before[i].equals(after[i]), `${[P.ledger, P.outcomes, P.raw][i]} changed on a no-op resume`);
  }
  // The status file is the ONE file that must grow: it is the record of how
  // often the run was interrupted, which is worthless if it is overwritten.
  assert.equal(lines(P.status).filter(l => l.startsWith('=== FULL session')).length, 2);
});

// ---- 5. resume after a crash: the event must appear exactly ONCE
test('resume: orphan records from a killed run are purged, so no event is counted twice', async () => {
  const dir = scratchDir('crash');
  const P = outputPaths(dir, 1);

  // A clean run of event 1 only — this is the reference shape.
  await runMock(dir, { events: [EVENTS[0]] });
  const cleanCount = lines(P.ledger).length;
  assert.ok(cleanCount > 5, `a ladder run writes many records, got ${cleanCount}`);

  // Now simulate SIGKILL in the window between the ledger append loop and the
  // outcomes append: some of event 2's records are on disk, its outcomes line
  // is not. Same for the raw log.
  const orphans = lines(P.ledger).slice(0, 4)
    .map(l => { const r = JSON.parse(l); r.eventId = EVENTS[1].id; return JSON.stringify(r); });
  appendFileSync(P.ledger, orphans.join('\n') + '\n');
  appendFileSync(P.raw, JSON.stringify({ ok: true, evidence_id: EVENTS[1].id, model_key: 'mock' }) + '\n');
  // The kill also caught the outcomes append mid-write. Note there is no
  // trailing newline: the next append would glue itself onto this fragment.
  appendFileSync(P.outcomes, '{"modelKey":"mock","eventId"');
  assert.equal(lines(P.ledger).length, cleanCount + 4);

  const res = await runMock(dir);   // both events; event 1 is done, event 2 is not
  assert.equal(res.skipped, 1);
  assert.equal(res.completed, 1);
  assert.equal(res.purgedLedger, 4, 'the four orphan ledger records must be removed');
  assert.equal(res.purgedRaw, 1);

  // Count records PER (eventId, modelKey). This is the assertion that catches
  // the double count: 4 stale records plus a full re-run still satisfies every
  // token identity (D+A+B+B_SELF = prompt, C1+C2 = completion) and every
  // sanity row in the report, while inflating the event's cost by ~40 %.
  const perKey = new Map<string, number>();
  for (const l of lines(P.ledger)) {
    const r = JSON.parse(l);
    perKey.set(taskKey(r.eventId, r.modelKey), (perKey.get(taskKey(r.eventId, r.modelKey)) ?? 0) + 1);
  }
  assert.equal(perKey.get(taskKey(EVENTS[0].id, 'mock')), cleanCount);
  assert.equal(perKey.get(taskKey(EVENTS[1].id, 'mock')), cleanCount,
    'event 2 must appear exactly once — the same record count as a clean run');
  assert.equal(lines(P.outcomes).length, 2, 'one outcomes line per event, never two');
  // The torn fragment is gone rather than fused with the line appended after
  // it — which would have taken the artefact builders down at the very end of
  // a 1.7-day run, over half a line.
  for (const l of lines(P.outcomes)) JSON.parse(l);
  assert.deepEqual(lines(P.outcomes).map(l => JSON.parse(l).eventId), [EVENTS[0].id, EVENTS[1].id]);
});

// ---- 6. --fresh
test('resume: --fresh truncates, and the status line says so', async () => {
  const dir = scratchDir('fresh');
  const P = outputPaths(dir, 1);
  await runMock(dir);
  assert.equal(lines(P.outcomes).length, 2);

  const res = await runMock(dir, { fresh: true, events: [EVENTS[0]] });
  assert.equal(res.skipped, 0, '--fresh must not resume anything');
  assert.equal(res.completed, 1);
  // Only the one event of the fresh run survives: the other event's records
  // were thrown away rather than resumed.
  assert.equal(lines(P.outcomes).length, 1);
  assert.equal(JSON.parse(lines(P.outcomes)[0]).eventId, EVENTS[0].id);
  assert.ok(lines(P.ledger).every(l => JSON.parse(l).eventId === EVENTS[0].id));

  const said = lines(P.status).find(l => l.includes('--fresh'));
  assert.ok(said, 'a run that destroys work must say so on the status line');
  assert.ok(/TRUNCATED/.test(said!), `status line must name the act, got: ${said}`);
  assert.ok(/2 completed ladder runs discarded/.test(said!), `status line must say what it cost, got: ${said}`);
});

// ---- 7. level separation
test('resume: an L1 run never touches L2 files, and vice versa', async () => {
  const dir = scratchDir('levels');
  const P1 = outputPaths(dir, 1), P2 = outputPaths(dir, 2);
  await runMock(dir, { level: 1 });

  for (const p of [P2.ledger, P2.outcomes, P2.raw, P2.status, P2.report, P2.structureTokens]) {
    assert.ok(!existsSync(p), `an L1 run wrote ${p}`);
  }
  // The pre-prefix scheme wrote `ledger5.jsonl` and relied on the operator
  // renaming it afterwards. No unprefixed run output may exist any more.
  for (const stale of ['ledger5.jsonl', 'outcomes.jsonl', 'runs_raw.jsonl', 'run_status.txt', 'ledger_report_v2.md', 'transcript.md']) {
    assert.ok(!existsSync(join(dir, stale)), `unprefixed ${stale} was left behind`);
  }

  const l1Before = readFileSync(P1.ledger);
  await runMock(dir, { level: 2, events: EVENTS.map(e => ({ ...e, id: e.id.replace('_L1', '_L2') })) });
  assert.ok(readFileSync(P1.ledger).equals(l1Before), 'the L2 run modified the L1 ledger');
  assert.ok(existsSync(P2.ledger));
  assert.ok(lines(P2.ledger).every(l => JSON.parse(l).eventId.endsWith('_L2')));
  // The calibration cache is the one deliberately SHARED thing: a per-model
  // measurement, not run output, so the cheap probe is paid for once. It is one
  // file PER MODEL now — a single shared object could not survive two sessions.
  assert.ok(existsSync(overheadPathFor(dir, 'mock')));
  assert.ok(!existsSync(join(dir, 'template_overhead.json')),
    'the clobberable single-object cache is back');
});

// ---- 8. the four artefacts
test('resume: a full run generates the four artefacts under its own level prefix', async () => {
  const dir = scratchDir('artefacts');
  const P = outputPaths(dir, 2);
  const res = await runMock(dir, { level: 2 });
  assert.equal(res.artefactsOk, true);

  for (const p of [P.report, P.transcript, P.transcriptFull, P.structureTokens]) {
    assert.ok(existsSync(p), `missing artefact ${p}`);
    assert.ok(readFileSync(p, 'utf-8').length > 100, `${p} is empty`);
  }
  // The L2 run of 2026-08-28 has transcripts but no report, because generating
  // it was a separate manual command and the step was forgotten.
  assert.ok(readFileSync(P.report, 'utf-8').includes('five-bucket token accounting'));
  // The summary must describe THIS ledger: its provenance header carries the
  // hash of the file it was built from.
  const meta = JSON.parse(lines(P.structureTokens)[0])._meta;
  assert.equal(meta.ledgerPath, P.ledger);
  assert.equal(meta.ledgerSha256, createHash('sha256').update(readFileSync(P.ledger)).digest('hex'));
  assert.equal(meta.ledgerLines, lines(P.ledger).length);
  // 2 events x 1 model x 3 structures, and every event that ran is present.
  assert.equal(lines(P.structureTokens).length - 1, 6);
  // IV-2 (D31): the session header records the harness version.
  assert.ok(lines(P.status).some(l => l.includes('harness=')),
    'the status header must record the harness commit');
  // Blinding (2026-09-05): no artefact a full run writes may join a forecast to
  // its outcome. The ground-truth block is opt-in via --reveal-truth only.
  assert.ok(!readFileSync(P.transcript, 'utf-8').includes('Ground truth'),
    'a run-time transcript must not contain the ground truth');
  assert.ok(!readFileSync(P.transcriptFull, 'utf-8').includes('Ground truth'));
});

// ---- 8b. the ground-truth gate itself, on a pack that WOULD reveal it
test('transcript: ground truth appears only with revealTruth, never by default', () => {
  const records = [{
    ok: true, modelKey: 'm', eventId: 'AAPL_2024Q1_L1', agentRole: 'worker_1',
    responseText: '<reason>r</reason><decision>0.6</decision>', reasoningText: '',
  }] as any;
  const packs = [{ id: 'AAPL_2024Q1', symbol: 'AAPL', reportDate: '2024-01-01',
    t0: '2024-01-01', label: 1, abnormalReturn: 0.0123 }];
  const closed = buildTranscript(records, [], packs, false);
  const revealed = buildTranscript(records, [], packs, false, true);
  assert.ok(!closed.includes('Ground truth'),
    'default rendering must withhold the label even when the pack carries it');
  assert.ok(revealed.includes('Ground truth') && revealed.includes('POSITIVE (label 1)'),
    'the post-run rendering must still be able to show it');
});

// ---- 9. a builder failure costs the artefact, never the run
test('resume: a failing report builder does not fail the run, but is not silent either', async () => {
  const dir = scratchDir('builderfail');
  const P = outputPaths(dir, 1);
  // A pack whose header field blows up when the transcript builder reads it.
  // What breaks does not matter; that the ladder runs are untouched by it
  // does — they already cost real money and are already on disk. (The trap sat
  // on `abnormalReturn` until 2026-09-05; the run-time builder no longer reads
  // the ground-truth fields at all — that is the blinding fix — so the trap
  // moved to `reportDate`, which the header line still reads.)
  const boom = [{ id: 'AAPL_2024Q1', symbol: 'AAPL', t0: '2024-01-01', label: 1, abnormalReturn: 0.01,
    get reportDate(): string { throw new Error('simulated builder failure'); } }];
  const res = await runMock(dir, { packs: boom as any });

  assert.equal(res.completed, 2, 'the expensive part succeeded and must be reported as such');
  assert.equal(res.artefactsOk, false, 'a broken artefact must not pass unnoticed');
  assert.equal(lines(P.ledger).length > 0, true, 'the ledger is on disk regardless');
  assert.equal(lines(P.outcomes).length, 2);
  // The other three artefacts are still produced: one builder throwing must not
  // cost the author the rest.
  assert.ok(existsSync(P.report));
  assert.ok(existsSync(P.structureTokens));
  assert.ok(!existsSync(P.transcript), 'the transcript is the one that failed here');
  assert.ok(!existsSync(P.transcriptFull));
  assert.ok(lines(P.status).some(l => l.startsWith('[report] FAILED')),
    'the failure has to be on the status line, in words');

  // And it is genuinely resumable afterwards: nothing was lost.
  const again = await runMock(dir);
  assert.equal(again.completed, 0);
  assert.equal(again.skipped, 2);
});

// ---- 10. FINDING 1 end to end: an outage leaves the event to be retried
test('resume: a provider outage writes no outcomes line, and the resume re-runs exactly that event', async () => {
  const dir = scratchDir('outage');
  const P = outputPaths(dir, 1);
  const outage = (ev: string, role: string) =>
    ev === EVENTS[1].id && role === 'aggregator' ? new Error('fetch failed: ECONNRESET after 4 attempts') : null;

  const first = await runMock(dir, { makeClient: () => mockLadderClient(P.raw, 'mock', outage) });
  assert.equal(first.completed, 1, 'the healthy event completes');
  // THE assertion: nothing at all was banked for the broken event — no outcomes
  // line, so it is not done, and no ledger records, so nothing to purge either.
  assert.deepEqual(lines(P.outcomes).map(l => JSON.parse(l).eventId), [EVENTS[0].id]);
  assert.ok(!lines(P.ledger).some(l => JSON.parse(l).eventId === EVENTS[1].id));
  // The failure taxonomy must stay clean: an outage may not appear as a drop.
  assert.equal(first.dropped, 0);
  for (const l of lines(P.outcomes)) assert.ok(!/ECONNRESET/.test(l), 'an outage was recorded as a scientific result');
  // ... and it has to SAY so, or a wall of FAILs at 3am reads like lost data.
  const said = lines(P.status).find(l => l.includes('FAIL'));
  assert.ok(said && /retried on the next resume/.test(said), `status line must say the event is not lost, got: ${said}`);

  // The provider comes back: the resume runs exactly the missing event.
  const clean = await runMock(dir, { events: [EVENTS[0]] });   // reference shape
  const perEvent = new Map<string, number>();
  const again = await runMock(dir);
  assert.equal(again.skipped, 1);
  assert.equal(again.completed, 1);
  for (const l of lines(P.ledger)) {
    const r = JSON.parse(l);
    perEvent.set(r.eventId, (perEvent.get(r.eventId) ?? 0) + 1);
  }
  assert.equal(perEvent.get(EVENTS[0].id), perEvent.get(EVENTS[1].id),
    'the retried event must end up with the same record count as a clean one');
  assert.deepEqual(lines(P.outcomes).map(l => JSON.parse(l).eventId), [EVENTS[0].id, EVENTS[1].id]);
  assert.equal(clean.completed, 0, 'the reference run found event 1 already done');
});

// ---- 11. FINDING 2: a probe may not touch the authoritative files
test('probe: a smoke test writes probe_L1_* only, and never contaminates what full trusts', async () => {
  const dir = scratchDir('probe');
  const A = outputPaths(dir, 1);              // authoritative — what `full` reads
  const B = outputPaths(dir, 1, 'probe');     // the probe's own files
  assert.notEqual(A.ledger, B.ledger, 'probe and full must not share a path');

  // Two probe runs on ONE event. Before the fix this left the AUTHORITATIVE
  // files with 2 outcomes lines and 40 ledger records where one honest pass
  // writes 20, and the next `full` reported "resuming: 1 of 2 done".
  await runMock(dir, { mode: 'probe', events: [EVENTS[0]] });
  await runMock(dir, { mode: 'probe', events: [EVENTS[0]] });
  for (const p of [A.ledger, A.outcomes, A.raw, A.status, A.report]) {
    assert.ok(!existsSync(p), `a probe wrote the authoritative ${p}`);
  }
  // The probe's own files show ONE pass, not the sum of the last two runs.
  assert.equal(lines(B.outcomes).length, 1);
  const probeRecords = lines(B.ledger).length;
  assert.ok(lines(B.status).some(l => l.includes(B.prefix)),
    'the status line must say where the probe wrote');

  // `full` now finds nothing done and plans every event.
  const res = await runMock(dir);
  assert.equal(res.planned, 2);
  assert.equal(res.skipped, 0, 'a probe must never make an event look complete');
  assert.equal(res.completed, 2);
  const perEvent = new Map<string, number>();
  for (const l of lines(A.ledger)) {
    const r = JSON.parse(l);
    perEvent.set(r.eventId, (perEvent.get(r.eventId) ?? 0) + 1);
  }
  assert.equal(perEvent.size, 2);
  assert.equal(perEvent.get(EVENTS[0].id), probeRecords, 'one honest pass, not two');
  assert.equal(perEvent.get(EVENTS[1].id), probeRecords);
  assert.equal(lines(A.outcomes).length, 2);
});

// ---- 12. FINDING 3: one writer per level
test('lock: a second session on the same level refuses to start, and says how to clear it', async () => {
  const dir = scratchDir('lock');
  const P = outputPaths(dir, 1);
  // `run` takes the lock synchronously before its first await, so this is a
  // real overlap and not a timing race: A is mid-session when B starts.
  const a = runMock(dir);
  const b = runMock(dir).then(() => 'RAN').catch((e: Error) => e.message);
  const [ra, rb] = await Promise.all([a, b]);

  assert.equal(ra.completed, 2, 'the session that holds the lock runs normally');
  assert.notEqual(rb, 'RAN', 'two sessions ran against one level');
  assert.match(rb as string, /another session holds/);
  // A crashed run must be recoverable BY THE OPERATOR, so the message has to
  // name the file to delete — refusing without a way out is its own outage.
  assert.match(rb as string, /rm "/);
  assert.match(rb as string, /L1_run\.lock/);
  assert.ok(lines(P.status).some(l => l.includes('REFUSING TO START')));

  // Exactly one pass per event, and the lock is gone afterwards.
  assert.equal(lines(P.outcomes).length, 2);
  assert.ok(!existsSync(P.lock), 'the lock outlived the session that held it');
});

test('lock: a killed session leaves a STALE lock that clears itself, loudly', async () => {
  const dir = scratchDir('stalelock');
  const P = outputPaths(dir, 1);
  // What a SIGKILL leaves behind: a lock naming a pid that no longer exists.
  // Blocking the restart on it would make a crashed run unrecoverable — the
  // opposite of the point, on a 1.7-day run where a kill is a certainty.
  const deadPid = 999999;
  assert.equal(isPidAlive(deadPid), false, 'the test needs a pid that is really gone');
  writeFileSync(P.lock, JSON.stringify({
    pid: deadPid, host: hostname(), started: '2026-09-01T00:00:00.000Z', mode: 'full', level: 1,
  }) + '\n');

  const res = await runMock(dir);
  assert.equal(res.completed, 2, 'a stale lock must not block the restart');
  const said = lines(P.status).find(l => l.includes('STALE LOCK'));
  assert.ok(said, 'clearing another session\'s lock file may not be silent');
  assert.match(said!, new RegExp(String(deadPid)), 'the line must name the pid it decided was dead');
  assert.ok(!existsSync(P.lock));
});

test('lock: a lock that cannot be read is not adjudicated, and the pid check is honest', async () => {
  const dir = scratchDir('badlock');
  const P = outputPaths(dir, 1);
  writeFileSync(P.lock, 'this is not json\n');
  // Something wrote it; "I cannot read it" is not evidence that it is dead.
  await assert.rejects(() => runMock(dir), /another session holds/);
  assert.ok(existsSync(P.lock), 'an unreadable lock must not be silently discarded');
  assert.equal(lines(P.outcomes).length, 0, 'not one call may be made while refused');

  // The operator does what the message says.
  const fs = await import('node:fs');
  fs.unlinkSync(P.lock);
  assert.equal((await runMock(dir)).completed, 2);

  // isPidAlive is the whole basis of the stale/live decision, so pin both ends.
  assert.equal(isPidAlive(process.pid), true);
  assert.equal(isPidAlive(0), false);
  assert.equal(isPidAlive(-1), false);
  assert.equal(isPidAlive(999999), false);
});

test('lock: a session that throws still releases the lock', async () => {
  const dir = scratchDir('lockthrow');
  const P = outputPaths(dir, 1);
  // A corrupt legacy calibration cache makes the migration — and so run() —
  // throw AFTER the lock is taken. Whatever the failure, leaving the lock
  // behind would block every restart of a level that is otherwise perfectly
  // resumable.
  writeFileSync(P.overheadLegacy, '{ not json');
  await assert.rejects(() => runMock(dir));
  assert.ok(!existsSync(P.lock), 'the lock survived a failed session and would block the restart');

  writeFileSync(P.overheadLegacy, '{}');
  assert.equal((await runMock(dir)).completed, 2);
});

// ---- 13. FINDING 4: the purge is atomic, and now something checks it
test('purge: a crash between the temp write and the rename leaves the original file complete', async () => {
  const dir = scratchDir('purgeatomic');
  const path = join(dir, 'L1_ledger5.jsonl');
  // E1 is done, E2 is a half-run — so the purge really does rewrite the file.
  const rows = [
    { eventId: 'E1', modelKey: 'mock', agentRole: 'worker_1', totalTokens: 111 },
    { eventId: 'E2', modelKey: 'mock', agentRole: 'worker_1', totalTokens: 222 },
    { eventId: 'E1', modelKey: 'mock', agentRole: 'worker_2', totalTokens: 333 },
    { eventId: 'E2', modelKey: 'mock', agentRole: 'worker_2', totalTokens: 444 },
    { eventId: 'E1', modelKey: 'mock', agentRole: 'aggregator', totalTokens: 555 },
    { eventId: 'E1', modelKey: 'mock', agentRole: 'debater_1', totalTokens: 666 },
  ];
  writeFileSync(path, rows.map(r => JSON.stringify(r)).join('\n') + '\n');
  const done = new Set([taskKey('E1', 'mock')]);
  const before = readFileSync(path);
  const intact = (why: string) => {
    assert.ok(readFileSync(path).equals(before), why);
    assert.equal(lines(path).length, rows.length, why);
    for (const l of lines(path)) JSON.parse(l);          // every line still whole
    assert.ok(!existsSync(`${path}.purge.tmp`), 'a fragment was left behind');
  };

  // (a) the write dies halfway — the ENOSPC / power-cut case the comment on
  // purgeFile claims to survive. Nothing had ever exercised it: a mutant that
  // wrote the survivors straight back over `path` passed all 43 tests.
  const written: string[] = [];
  const halfWrite: PurgeIO = {
    write: (p, d) => {
      written.push(p);
      writeFileSync(p, d.slice(0, Math.max(1, Math.floor(d.length / 2))));   // torn
      throw Object.assign(new Error('ENOSPC: no space left on device'), { code: 'ENOSPC' });
    },
    rename: () => assert.fail('rename must not run after a failed write'),
  };
  assert.throws(() => purgeFile(path, done, ledgerKeyOf, halfWrite), /ENOSPC/);
  intact('a failed WRITE damaged the original ledger');
  assert.ok(written[0].endsWith('.purge.tmp'),
    `the purge wrote to ${written[0]}, i.e. straight over the file it was cleaning`);

  // (b) the write succeeds and the RENAME dies (EPERM: another handle on the
  // file, routine on Windows). Same requirement.
  const failRename: PurgeIO = {
    write: (p, d) => writeFileSync(p, d),
    rename: () => { throw Object.assign(new Error('EPERM: operation not permitted, rename'), { code: 'EPERM' }); },
  };
  assert.throws(() => purgeFile(path, done, ledgerKeyOf, failRename), /EPERM/);
  intact('a failed RENAME damaged the original ledger');

  // And with no failure injected it still does its job.
  assert.equal(purgeFile(path, done, ledgerKeyOf), 2);
  assert.deepEqual(lines(path).map(l => JSON.parse(l).eventId), ['E1', 'E1', 'E1', 'E1']);
  assert.ok(!existsSync(`${path}.purge.tmp`));
});

// ---- 13b. D32 deviation 4: the purge must survive a file past Node's string limit
// On 2026-09-09 the L2 resume died at startup: L2_runs_raw.jsonl had grown to
// 554 MB and readFileSync(path, 'utf-8') cannot build a string longer than
// 0x1fffffe8 (≈ 512 MB) characters. The read side now goes through a Buffer;
// the write side must never join the survivors into ONE string either, so the
// temp file is written in chunks: `write` creates it, `append` continues it.
test('purge: survivors are written in chunks, never as one joined string', () => {
  const dir = scratchDir('purgechunk');
  const path = join(dir, 'L2_runs_raw.jsonl');
  const rows = Array.from({ length: 60 }, (_, i) =>
    JSON.stringify({ evidence_id: i % 6 === 0 ? 'ORPHAN_L2' : 'E1_L2', model_key: 'mock', call_seq: i, pad: 'x'.repeat(40) }));
  writeFileSync(path, rows.join('\n') + '\n');
  const done = new Set([taskKey('E1_L2', 'mock')]);
  const calls: string[] = [];
  const io: PurgeIO = {
    write: (p, d) => { calls.push(`write:${d.length}`); writeFileSync(p, d); },
    append: (p, d) => { calls.push(`append:${d.length}`); appendFileSync(p, d); },
    rename: (f, t) => renameSync(f, t),
  };
  // 300 characters per chunk: a handful of lines each, so the 50 survivors need several chunks.
  const removed = purgeFile(path, done, rawKeyOf, io, 300);
  assert.equal(removed, 10);
  const expected = rows.filter(r => !r.includes('ORPHAN')).join('\n') + '\n';
  assert.equal(readFileSync(path, 'utf-8'), expected, 'chunking changed the bytes');
  assert.ok(calls[0].startsWith('write:'), `first chunk must CREATE the temp file, got ${calls[0]}`);
  assert.ok(calls.filter(c => c.startsWith('append:')).length >= 3, `expected chunked appends, got ${calls.join(' ')}`);
  for (const c of calls) assert.ok(Number(c.split(':')[1]) <= 300 + 200, `a chunk exceeded the limit by more than one line: ${c}`);
  assert.ok(!existsSync(`${path}.purge.tmp`));
  // CRLF input: lines are still recognised (the old regex split accepted CR LF).
  writeFileSync(path, rows.join('\r\n') + '\r\n');
  assert.equal(purgeFile(path, done, rawKeyOf), 10);
  assert.equal(readFileSync(path, 'utf-8'), expected);
});

// ---- 14. the --fresh opt-out is exact
test('resume: --fresh is recognised only when it is actually asked for', () => {
  assert.equal(wantsFresh(['--fresh'], {}), true);
  assert.equal(wantsFresh([], { PILOT5_FRESH: '1' }), true);
  assert.equal(wantsFresh([], {}), false);
  // Consent to destroy ~1.2 days of work must be exact, never inferred.
  assert.equal(wantsFresh(['--fres'], {}), false);
  assert.equal(wantsFresh(['--fresh-start'], {}), false);
  assert.equal(wantsFresh([], { PILOT5_FRESH: '0' }), false);
  assert.equal(wantsFresh([], { PILOT5_FRESH: 'true' }), false);
});

// ==========================================================================
// TWO SESSIONS IN ONE PROCESS
//
// The author will very likely run level 1 and level 2 side by side to halve
// the ~1.7 days of wall clock, and the lock deliberately permits that:
// different levels are different locks. Everything below is about what that
// concurrency does to state the two sessions share.
// ==========================================================================

// ---- 15. FINDING A: one status log per session, not per process
test('status: level 1 and level 2 side by side each keep their OWN status log', async () => {
  const dir = scratchDir('statuslogs');
  const evs = (lvl: number) => [1, 2, 3].map(i =>
    ({ id: `E${i}_L${lvl}`, text: `FICTIONAL COMPANY ${i} reported revenue of $${i}B, up 10%.` }));
  const at = (level: number) => run(['mock'], null, {
    outDir: dir, level, mode: 'full', concurrency: 2, repeats: 1,
    events: evs(level), models: [MOCK_MODEL], packs: [],
    makeClient: () => mockLadderClient(outputPaths(dir, level, 'full').raw),
  });

  // Both sessions really are in flight together: run() takes its lock and (in
  // the broken version) reassigned the shared status path synchronously, before
  // its first await.
  const [r1, r2] = await Promise.all([at(1), at(2)]);
  assert.equal(r1.completed, 3);
  assert.equal(r2.completed, 3);

  // Before the fix, L1_run_status.txt held its 2 header lines and NOTHING else:
  // no calibration line, no per-event progress, no "FULL done", no artefact
  // lines. All 18 of them were in L2_run_status.txt. The run was fine; the
  // record of it — the only diagnostic there is over 1.7 days — was gone.
  for (const level of [1, 2]) {
    const ls = lines(outputPaths(dir, level).status);
    const has = (re: RegExp) => ls.some(l => re.test(l));
    assert.ok(has(/=== FULL session/), `L${level}: no session header`);
    assert.ok(has(/\[calib\] mock: template overhead/), `L${level}: the calibration line went to the other session`);
    assert.equal(ls.filter(l => /^\[full\] \(\d\/3\)/.test(l)).length, 3, `L${level}: per-event progress is missing`);
    assert.ok(has(/FULL done\. 3\/3/), `L${level}: no completion line`);
    assert.equal(ls.filter(l => /^\[report\] wrote /.test(l)).length, 4, `L${level}: the artefact lines are missing`);
    const other = level === 1 ? 2 : 1;
    assert.ok(!ls.some(l => l.includes(`_L${other}`)), `L${level}'s log swallowed L${other}'s lines`);
  }
});

// ---- 16. FINDING B: the shared calibration cache cannot be clobbered
test('calibration: a model key survives the filename round trip, and the name is safe', () => {
  for (const key of ['qwen3-32b@on', 'qwen3-32b@off', 'deepseek-chat-v3-0324',
                     'meta-llama/llama-3.3-70b:free', 'weird key 50%?*', 'CamelCase', 'ünïcode']) {
    const enc = encodeModelKey(key);
    assert.equal(decodeModelKey(enc), key, `${key} did not survive the round trip`);
    // Lower case only, and none of the characters Windows refuses: the key is
    // the join back to the ledger, so it has to come out the way it went in.
    assert.match(enc, /^[a-z0-9._%-]*$/, `${enc} is not a safe, case-collision-free filename`);
  }
  const d = tmpdir();
  assert.notEqual(overheadPathFor(d, 'qwen3-32b@on'), overheadPathFor(d, 'qwen3-32b@off'));
});

test('calibration: a session cannot clobber a concurrent session\'s measurement', async () => {
  const dir = scratchDir('calibrace');
  const spec = (key: string): ModelSpec => ({ ...MOCK_MODEL, key, slug: `mock/${key}` });
  const X = spec('qwen3-32b@on'), Y = spec('deepseek-chat-v3-0324'), Z = spec('llama-3.3-70b');

  // The full session holds at its FIRST call — the calibration probe — until
  // the probe session has measured and written Z. That is exactly the
  // interleaving that lost Z: read the whole object once, rewrite it whole
  // later. Deterministic, not a hoped-for timing coincidence.
  let release!: () => void;
  const gate = new Promise<void>(r => { release = r; });
  const gated = (m: ModelSpec): LadderClient => {
    const inner = mockLadderClient(outputPaths(dir, 1, 'full').raw, m.key);
    let first = true;
    return {
      beginStructureRun: inner.beginStructureRun,
      async generate(req: GenerateRequest) {
        if (first) { first = false; await gate; }
        return inner.generate(req);
      },
    };
  };

  const full = run([X.key, Y.key], null, {
    outDir: dir, level: 1, mode: 'full', concurrency: 1, repeats: 1,
    events: [EVENTS[0]], models: [X, Y], packs: [], makeClient: gated,
  });
  const probe = run([Z.key], 1, {
    outDir: dir, level: 2, mode: 'probe', concurrency: 1, repeats: 1,
    events: [EVENTS[0]], models: [Z], packs: [],
    makeClient: (m: ModelSpec) => mockLadderClient(outputPaths(dir, 2, 'probe').raw, m.key),
  });

  await probe;
  assert.equal(readOverhead(dir, Z.key), 7, 'the probe did not record its own measurement');
  release();
  await full;

  // All three survive, each in its own file, each naming the model it belongs
  // to. Before the fix the file held only X and Y; Z's paid call was lost, and
  // would have been paid for again.
  for (const m of [X, Y, Z]) {
    const p = overheadPathFor(dir, m.key);
    assert.ok(existsSync(p), `${m.key} has no calibration file: its measurement was clobbered`);
    const o = JSON.parse(readFileSync(p, 'utf-8'));
    assert.equal(o.modelKey, m.key, `${p} holds ${o.modelKey}'s measurement, not ${m.key}'s`);
    assert.equal(o.overhead, 7);
  }
});

test('calibration: an existing template_overhead.json is migrated, not orphaned', async () => {
  const dir = scratchDir('calibmigrate');
  const legacy = outputPaths(dir, 1).overheadLegacy;
  // Two REAL measurements, in the pre-per-model format. 11 and 13 are not what
  // the mock would measure (7), so re-measurement is distinguishable from reuse.
  const before = JSON.stringify({ 'qwen3-32b@on': 11, 'deepseek-chat-v3-0324': 13 }, null, 2);
  writeFileSync(legacy, before);

  const M: ModelSpec = { ...MOCK_MODEL, key: 'qwen3-32b@on' };
  const go = () => run([M.key], null, {
    outDir: dir, level: 1, mode: 'full', concurrency: 1, repeats: 1,
    events: [EVENTS[0]], models: [M], packs: [],
    makeClient: () => mockLadderClient(outputPaths(dir, 1, 'full').raw, M.key),
  });
  assert.equal((await go()).completed, 1);

  assert.equal(readOverhead(dir, 'qwen3-32b@on'), 11,
    'the migrated measurement was thrown away and the model re-measured');
  assert.equal(readOverhead(dir, 'deepseek-chat-v3-0324'), 13,
    'a model that did not run in this session lost its measurement');
  // A migrated value was not measured now, and the file may not claim it was.
  const migrated = JSON.parse(readFileSync(overheadPathFor(dir, 'qwen3-32b@on'), 'utf-8'));
  assert.match(migrated.measuredAt, /migrated from template_overhead\.json/);
  // The original is kept, byte for byte: it holds real measurements, so if the
  // split ever turns out wrong the evidence is still on disk.
  assert.ok(!existsSync(legacy), 'the legacy file would be re-read on every start');
  assert.equal(readFileSync(`${legacy}.migrated`, 'utf-8'), before);
  assert.ok(lines(outputPaths(dir, 1).status).some(l => /migrated 2 measurement/.test(l)),
    'moving another run\'s calibration data may not be silent');

  // Idempotent: the next session finds nothing to migrate and changes nothing.
  await go();
  assert.equal(readOverhead(dir, 'qwen3-32b@on'), 11);
  assert.equal(readFileSync(`${legacy}.migrated`, 'utf-8'), before);
});

// ---- 17. FINDING C: the lock must ALLOW what it is meant to allow
test('lock: two different levels, and a probe beside a full run, all proceed', async () => {
  // Every other lock test here checks that it REFUSES. A defect that made
  // acquireLock scan sibling *_run.lock files and treat ANY of them as held —
  // a level-blind lock that blocks exactly the work this design permits —
  // passed all 51 of them. This is the other half.
  const dir = scratchDir('locklevels');
  const at = (level: number, mode = 'full') => run(['mock'], null, {
    outDir: dir, level, mode, concurrency: 2, repeats: 1,
    events: EVENTS.map(e => ({ ...e, id: e.id.replace('_L1', `_L${level}`) })),
    models: [MOCK_MODEL], packs: [],
    makeClient: () => mockLadderClient(outputPaths(dir, level, mode).raw),
  });
  const [a, b] = await Promise.all([at(1), at(2)]);
  assert.equal(a.completed, 2, 'the level-1 session was blocked by the level-2 lock');
  assert.equal(b.completed, 2, 'the level-2 session was blocked by the level-1 lock');
  assert.ok(!existsSync(outputPaths(dir, 1).lock) && !existsSync(outputPaths(dir, 2).lock),
    'a lock outlived the session that held it');

  // Same level, different mode: a probe has its own prefix, its own files and
  // its own lock, so it must be able to run beside the full session it is
  // smoke-testing for.
  const dir2 = scratchDir('lockprobe');
  const at2 = (mode: string) => run(['mock'], null, {
    outDir: dir2, level: 1, mode, concurrency: 2, repeats: 1,
    events: EVENTS, models: [MOCK_MODEL], packs: [],
    makeClient: () => mockLadderClient(outputPaths(dir2, 1, mode).raw),
  });
  const [f, p] = await Promise.all([at2('full'), at2('probe')]);
  assert.equal(f.completed, 2, 'a probe blocked the full run on the same level');
  assert.equal(p.completed, 2, 'the full run blocked the probe on the same level');
});

// ---- 18. FINDING D: the claim itself is exclusive
test('lock: two simultaneous claims — exactly one wins, and the loser is told who holds it', () => {
  const dir = scratchDir('lockrace');
  const P = outputPaths(dir, 1);
  const said: string[] = [];
  const info = (pid: number) => ({
    pid, host: hostname(), started: `2026-09-01T00:00:0${pid}.000Z`, mode: 'full', level: 1,
  });

  // Both attempts must OBSERVE "no lock present" and only then write — the
  // check and the write are two syscalls, and `flag: 'wx'` is what closes that
  // window. acquireLock is synchronous, so the window is opened DELIBERATELY
  // rather than hoped for: B's complete claim is slid into A's, after A has
  // looked and before A writes. Removing the flag and changing nothing else
  // passed all 51 subtests; nothing mentioned EEXIST or wx.
  let releaseB: (() => void) | null = null;
  let bFailed: unknown = null;
  let aFailed: any = null;
  let aRelease: (() => void) | null = null;
  try {
    aRelease = acquireLock(P.lock, info(1), l => said.push(l), {
      beforeClaim: () => {
        if (releaseB) return;                       // A's claim, once
        try { releaseB = acquireLock(P.lock, info(2), l => said.push(l)); }
        catch (e) { bFailed = e; }
      },
    });
  } catch (e) { aFailed = e; }

  assert.equal(bFailed, null, 'the session that got there first must not be refused');
  assert.ok(releaseB, 'nobody took the lock at all');
  assert.ok(aFailed, 'BOTH claims succeeded: two sessions would run one level and bill every event twice');
  assert.equal(aRelease, null);
  assert.match(aFailed.message, /at the same instant/);
  // Refusing without saying who won leaves the operator with nothing to act on.
  assert.match(aFailed.message, /pid 2/);

  // The file holds the WINNER, untouched by the loser.
  const holder = readLockInfo(readFileSync(P.lock, 'utf-8'))!;
  assert.equal(holder.pid, 2, 'the loser overwrote the winner\'s lock file');
  assert.equal(holder.started, info(2).started);
  const lost = said.find(l => l.includes('LOST THE RACE'));
  assert.ok(lost, 'losing the race was silent');
  assert.match(lost!, /pid 2/);

  releaseB!();
  assert.ok(!existsSync(P.lock), 'the winner could not release the lock it holds');
});

// ---- D27 leak detector: boundaries, not substrings
test('audit: reasonStatesOwnNumber matches whole numbers, not substrings of other figures', () => {
  // The first version tested `hay.includes('.65')`. Financial reasoning is made
  // of percentages, so "the S&P has lagged 9.65%" flagged an answer of 0.65 as
  // having leaked its own probability when it had done nothing of the sort.
  // Measured on the pilot logs, the loose version reported 4 of 12 worker
  // answers where 3 are genuine - and that rate is a number the thesis reports.
  const notALeak: [string, number][] = [
    ['the S&P 500 has lagged (1.70% and 9.65% over the same periods)', 0.65],
    ['revenue rose 10.35% year on year', 0.35],
    ['a 165% move is impossible', 0.65],
    ['my estimate is 0.653', 0.65],
    ['exactly 0.6500 here', 0.65],
    ['roughly even odds', 0.5],          // verbal, not caught by design
  ];
  for (const [text, p] of notALeak) {
    assert.equal(reasonStatesOwnNumber(text, p), false, `must NOT flag: ${text}`);
  }

  // What it must still catch, including the two spellings of one probability.
  const isALeak: [string, number][] = [
    ['A 55% probability balances the likelihood of a reversal', 0.55],
    ['implies a 45% chance of positive abnormal returns', 0.45],
    ['I put this at .65', 0.65],
    ['probability 0.50 seems fair', 0.5],
    ['about 50 percent', 0.5],
    ['I would say 50 % here', 0.5],
  ];
  for (const [text, p] of isALeak) {
    assert.equal(reasonStatesOwnNumber(text, p), true, `must flag: ${text}`);
  }

  // The property the boundaries exist for, stated once more as an invariant:
  // embedding the answer's own digits inside a LARGER number is never a leak.
  for (const p of [0.35, 0.45, 0.55, 0.65]) {
    const pct = Math.round(p * 100);
    assert.equal(reasonStatesOwnNumber(`the index moved 1.${pct}% last week`, p), false,
      `1.${pct}% must not flag ${p}`);
    assert.equal(reasonStatesOwnNumber(`a ${pct}% chance`, p), true,
      `a bare ${pct}% must flag ${p}`);
  }
});

// ==========================================================================
// D32 — the pre-declared operational checkpoint (PILOT5_EVENT_LIMIT)
//
// The real run may be PAUSED after its first N events for a blind operational
// check (APIs answer, nothing breaks, cost and time per ladder). "First N
// events" must mean the same thing at both levels, on both models, and on
// every resume: the first N DISTINCT base events in the seed-shuffled order of
// the FULL task list — not the first N packs by date, which evLimit gives.
// ==========================================================================

test('checkpoint: the first N distinct events of the shuffled order, both models, same set at both levels', () => {
  const mk = (key: string, id: string) => ({ m: { key }, ev: { id } });
  const ids = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6'];
  const tasksL1 = ['qwen', 'deepseek'].flatMap(k => ids.map(id => mk(k, `${id}_L1`)));
  const tasksL2 = ['qwen', 'deepseek'].flatMap(k => ids.map(id => mk(k, `${id}_L2`)));

  const keep = checkpointEvents(tasksL1, 3);
  assert.equal(keep.size, 3, 'exactly N base events');
  // Reference: walk the shuffled full list and take base ids as they appear.
  const order = shuffleInPlace([...tasksL1], WORK_ORDER_SEED).map(t => baseEventId(t.ev.id));
  const expected = [...new Set(order)].slice(0, 3);
  assert.deepEqual([...keep].sort(), [...expected].sort());
  // A set derived from the FULL list is not the alphabetical prefix.
  assert.notDeepEqual([...keep].sort(), ['E1', 'E2', 'E3'], 'must follow the shuffle, not the pack order');
  // Level-invariant: the two levels pick the same companies and quarters.
  assert.deepEqual([...checkpointEvents(tasksL2, 3)].sort(), [...keep].sort());
  // Restricting the task list keeps BOTH models for every kept event.
  const limited = tasksL1.filter(t => keep.has(baseEventId(t.ev.id)));
  assert.equal(limited.length, 6, '3 events x 2 models');
  for (const id of keep) assert.equal(limited.filter(t => baseEventId(t.ev.id) === id).length, 2);
  // Asking for more than there are events is the whole set, never an error.
  assert.equal(checkpointEvents(tasksL1, 99).size, 6);
});

test('checkpoint: eventLimit books only the checkpoint events, and lifting it resumes the rest', async () => {
  const dir = scratchDir('checkpoint');
  const P = outputPaths(dir, 1);
  const six = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF'].map(s => ({ id: `${s}_2025-01-01_L1`, text: `FICTIONAL ${s} reported.` }));

  const first = await runMock(dir, { events: six, eventLimit: 2 });
  assert.equal(first.planned, 2, 'the plan is the checkpoint, not the whole pool');
  assert.equal(first.completed, 2);
  const booked = lines(P.outcomes).map(l => baseEventId(JSON.parse(l).eventId));
  const expected = checkpointEvents(six.map(ev => ({ m: MOCK_MODEL, ev })), 2);
  assert.deepEqual(booked.sort(), [...expected].sort(), 'exactly the checkpoint set, nothing else');
  assert.ok(lines(P.status).some(l => /event limit 2/.test(l)), 'the status log states the limit');

  // Lifting the limit is an ordinary resume: nothing is re-run, the rest runs.
  const second = await runMock(dir, { events: six });
  assert.equal(second.skipped, 2);
  assert.equal(second.completed, 4);
  assert.equal(lines(P.outcomes).length, 6);
});

// ==========================================================================
// D32 deviation 1 (2026-09-07): DeepInfra throttled qwen into a retry spiral
// during the real run. Two rate-only fixes, both infrastructure:
//   (1) EVERY attempt takes a throttle slot, not only the first — a retry that
//       skips the queue is exactly the burst the throttle exists to prevent;
//   (2) the spacing can differ per provider for a split model (qwen@DeepInfra
//       gets 6 s, everything else keeps 3 s).
// ==========================================================================

test('throttle: a retried call takes a slot for EVERY attempt, not only the first', async () => {
  class CountingThrottle extends Throttle {
    takes = 0;
    override async take() { this.takes++; await super.take(); }
  }
  const throttle = new CountingThrottle(0);
  const dir = scratchDir('throttle');
  const okBody = JSON.stringify({
    id: 'gen-1', provider: 'mock',
    choices: [{ message: { content: '<reason>r</reason><decision>0.5</decision>' }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2, cost: 0 },
  });
  const realFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = (async () => {
    calls++;
    return calls === 1
      ? new Response('upstream busy', { status: 503 })
      : new Response(okBody, { status: 200, headers: { 'content-type': 'application/json' } });
  }) as typeof fetch;
  try {
    const client = new OpenRouterClient({
      modelKey: 'mock', modelSlug: 'mock/mock', provider: 'mock', mode: 'none', apiKey: 'x',
      logPath: join(dir, 'raw.jsonl'), cost: { spentUsd: 0, capUsd: 1 }, throttle,
    });
    client.beginStructureRun('ladder', 'E1');
    const res = await client.generate({ systemPrompt: 's', userPrompt: 'u', maxTokens: 10, temperature: 0 } as GenerateRequest);
    assert.ok(res.text.includes('0.5'));
    assert.equal(calls, 2, 'one 503 and one success');
    assert.equal(throttle.takes, 2, 'the retry must queue behind the throttle like any other request');
  } finally {
    globalThis.fetch = realFetch;
  }
});

// ==========================================================================
// D32 deviation 3 (2026-09-08): 8 of 10 slots per level were found HUNG after
// 21 h — ladders whose next call never settled, never logged, held no socket,
// and never timed out. Two independent guards, both wall-clock:
//   (1) every HTTP attempt has a hard deadline enforced by our own timer (a
//       Promise.race), not only by the signal handed to fetch;
//   (2) every ladder has a watchdog in the runner; a ladder that does not
//       complete in time is an OUTAGE (FAIL line, no outcomes line, retried on
//       the next resume) — never a silent dead slot.
// ==========================================================================

test('deadline: an HTTP attempt that never settles is abandoned by our own timer and retried', async () => {
  const dir = scratchDir('deadline');
  const realFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = (() => { calls++; return new Promise<Response>(() => { /* never settles, ignores the signal */ }); }) as typeof fetch;
  try {
    const client = new OpenRouterClient({
      modelKey: 'mock', modelSlug: 'mock/mock', provider: 'mock', mode: 'none', apiKey: 'x',
      logPath: join(dir, 'raw.jsonl'), cost: { spentUsd: 0, capUsd: 1 },
      attemptTimeoutMs: 40, rateLimitAttempts: 0,
    });
    client.beginStructureRun('ladder', 'E1');
    const t0 = Date.now();
    await assert.rejects(
      client.generate({ systemPrompt: 's', userPrompt: 'u', maxTokens: 10, temperature: 0 } as GenerateRequest),
      /OpenRouter call failed/,
    );
    assert.ok(Date.now() - t0 < 15000, 'must give up in bounded time');
    assert.equal(calls, 3, 'every attempt was tried and abandoned in turn (3 = base attempts with no 429 budget)');
    const failed = lines(join(dir, 'raw.jsonl')).map(l => JSON.parse(l)).filter(r => r.ok === false);
    assert.equal(failed.length, 1, 'the failure is logged once, like any other outage');
    assert.match(failed[0].error, /deadline/, 'and names the deadline as the cause');
  } finally {
    globalThis.fetch = realFetch;
  }
});

test('watchdog: a ladder that never completes becomes an outage, and the run goes on', async () => {
  const dir = scratchDir('watchdog');
  const P = outputPaths(dir, 1);
  const hung = EVENTS[1].id;
  // One wrapper PER ladder (run() calls makeClient per task); a shared one
  // would let the second task's beginStructureRun overwrite the first's id.
  const makeStuck = (): LadderClient => {
    const client = mockLadderClient(P.raw);
    let evId = '';
    return {
      beginStructureRun(c, e) { evId = e; client.beginStructureRun(c, e); },
      generate(req) { return evId === hung ? new Promise(() => { /* never */ }) : client.generate(req); },
    };
  };
  const t0 = Date.now();
  const res = await runMock(dir, { makeClient: makeStuck, ladderTimeoutMs: 3000 });
  assert.ok(Date.now() - t0 < 15000, 'the session must end; a hung ladder may not hold it open');
  assert.equal(res.completed, 1, 'the healthy event booked');
  assert.deepEqual(lines(P.outcomes).map(l => JSON.parse(l).eventId), [EVENTS[0].id], 'no outcomes line for the hung event');
  const fail = lines(P.status).find(l => l.includes(hung) && l.includes('FAIL'));
  assert.ok(fail && /watchdog/.test(fail), `the hung ladder is reported as an outage naming the watchdog, got: ${fail}`);
  assert.ok(/retried on the next resume/.test(fail!), 'and as retryable');
});

test('throttle: the spacing is per provider for a split model, per model otherwise', () => {
  const qwen = MODELS.find(m => m.key === 'qwen3-32b@on')!;
  const deepseek = MODELS.find(m => m.key === 'deepseek-chat-v3-0324')!;
  assert.equal(throttleIntervalFor(qwen, 'DeepInfra'), 6000, 'DeepInfra throttled the real run (D32 deviation 1)');
  assert.equal(throttleIntervalFor(qwen, 'SiliconFlow'), 3000, 'SiliconFlow ran clean at 3 s');
  assert.equal(throttleIntervalFor(deepseek, 'SiliconFlow'), 3000);
  assert.equal(throttleIntervalFor(MOCK_MODEL, 'mock'), 0, 'no spacing configured means none');
});
