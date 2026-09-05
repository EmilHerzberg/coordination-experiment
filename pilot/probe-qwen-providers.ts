// Decision probe: can qwen3-32b@on be load-split across DeepInfra AND SiliconFlow
// without breaking the repetition rung (RQ1) or the token accounting (RQ2)?
//
// The worry the split has to clear is the one that re-pinned deepseek in D24:
// DeepInfra collapsed CONCURRENT identical requests to byte-identical answers,
// which turns "N independent samples" into "1 sample counted N times". So the
// test fires N identical worker calls CONCURRENTLY at each provider and asks:
//   1. distinct?      sha256 over raw content — collapse shows up as duplicates
//   2. reasoning on?  reasoning_text present (the @on toggle must actually fire)
//   3. tokens equal?  same prompt -> same prompt_tokens across BOTH providers,
//                      else communication cost (RQ2) would depend on the provider
//   4. reachable?     SiliconFlow must actually serve qwen3-32b at all, pinned
//                      with allow_fallbacks:false (no silent substitution)
//
// Both providers are interleaved in one session so momentary load can't favour
// one. Real MSFT level-1 evidence, the real worker prompt, temperature 0.7 (the
// real run's value — collapse is a temp>0 phenomenon; at temp 0 identical output
// would be expected, not a defect).
//
//   PROBE_CONCURRENT=8 node --experimental-transform-types --import ./register.mjs probe-qwen-providers.ts
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, existsSync, unlinkSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { OpenRouterClient, type CostState } from './openrouter-client.js';
import { getKey } from './env.js';
import { workerPrompt } from './prompts.js';
import { renderSegments } from './ledger5.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUTDIR = resolve(HERE, '..', '..', 'pilot5');
const LOG = resolve(OUTDIR, 'qwen_provider_probe.jsonl');
if (existsSync(LOG)) unlinkSync(LOG); // fresh log per probe run

const pack = JSON.parse(readFileSync(resolve(OUTDIR, 'evidence_packs.json'), 'utf-8'))[0];
const prompt = workerPrompt({ id: pack.id, text: pack.level1 });
const SYS = renderSegments(prompt.system);
const USR = renderSegments(prompt.user);

const CONCURRENT = Number(process.env.PROBE_CONCURRENT ?? '8');
const MAXTOK = Number(process.env.PROBE_MAXTOK ?? '4000');

const cost: CostState = { spentUsd: 0, capUsd: 1.0 };
const PROVIDERS = ['DeepInfra', 'SiliconFlow'];

async function one(provider: string, i: number): Promise<void> {
  const c = new OpenRouterClient({
    modelKey: 'qwen3-32b@on', modelSlug: 'qwen/qwen3-32b',
    provider, mode: 'reason_on', apiKey: getKey(),
    logPath: LOG, cost,
  });
  c.beginStructureRun('probe', pack.id);
  try {
    await c.generate({
      systemPrompt: SYS, userPrompt: USR, tools: [], maxTokens: MAXTOK, temperature: 0.7,
      metadata: { agentRole: `${provider}#${i}`, configName: 'probe' },
    });
  } catch {
    // The client already logged ok:false with http_status + error; swallow so
    // Promise.all still resolves and the log stays the single source of truth.
  }
}

for (const p of PROVIDERS) {
  console.log(`firing ${CONCURRENT} CONCURRENT identical qwen3-32b@on calls at ${p} ...`);
  await Promise.all(Array.from({ length: CONCURRENT }, (_, i) => one(p, i)));
}

// ---- analyse the log (authoritative: gen_id, provider_returned, usage) --------
const rows = readFileSync(LOG, 'utf-8').split('\n').filter(Boolean).map(l => JSON.parse(l));
const sha = (s: string) => createHash('sha256').update(s, 'utf-8').digest('hex').slice(0, 12);
const num = (t: string) => {
  const m = String(t).match(/<decision>\s*([0-9]*\.?[0-9]+)\s*<\/decision>/i);
  if (!m) return null;
  const v = Number(m[1]);
  return Number.isFinite(v) && v >= 0 && v <= 1 ? v : null;
};
const med = (v: number[]) => (v.length ? [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)] : 0);

console.log('\n================ PER-PROVIDER ================');
const summary: Record<string, any> = {};
for (const p of PROVIDERS) {
  const rs = rows.filter(r => r.provider_pinned === p);
  const ok = rs.filter(r => r.ok);
  const bad = rs.filter(r => !r.ok);
  const contents: string[] = ok.map(r => r.content ?? '');
  const hashes = contents.map(sha);
  const distinctContent = new Set(hashes).size;
  const distinctGenId = new Set(ok.map(r => r.gen_id)).size;
  const withReasoning = ok.filter(r => (r.reasoning_text_len ?? 0) > 0).length;
  const promptTokSet = [...new Set(ok.map(r => r.prompt_tokens))];
  const completions = ok.map(r => r.completion_tokens ?? 0);
  const parsed = ok.map(r => num(r.content ?? '')).filter(v => v !== null) as number[];
  const returnedProviders = [...new Set(ok.map(r => r.provider_returned))];
  const truncated = ok.filter(r => r.finish_reason === 'length').length;
  const http429 = bad.filter(r => r.http_status === 429).length;

  // largest cluster of identical answers = the collapse signature
  const clusters: Record<string, number> = {};
  for (const h of hashes) clusters[h] = (clusters[h] ?? 0) + 1;
  const biggestCluster = Math.max(0, ...Object.values(clusters));

  summary[p] = {
    calls: rs.length, ok: ok.length, failed: bad.length, http429,
    distinctContent, biggestCluster, distinctGenId,
    reasoningPresent: `${withReasoning}/${ok.length}`,
    promptTokens: promptTokSet, completionTok: { min: Math.min(...completions), median: med(completions), max: Math.max(...completions) },
    wellFormedDecision: `${parsed.length}/${ok.length}`, decisionRange: parsed.length ? [Math.min(...parsed), Math.max(...parsed)] : [],
    providerReturned: returnedProviders, truncatedToCap: truncated,
  };

  console.log(`\n${p}`);
  console.log(`  reachable          : ${ok.length}/${rs.length} ok  (${bad.length} failed, ${http429}× 429)`);
  if (bad.length) console.log(`  first error        : status ${bad[0].http_status} — ${String(bad[0].error).slice(0, 120)}`);
  console.log(`  provider_returned  : ${returnedProviders.join(', ') || '(none)'}`);
  console.log(`  DISTINCT content   : ${distinctContent}/${ok.length}   (biggest identical cluster = ${biggestCluster}${biggestCluster > 1 ? '  <-- COLLAPSE' : ''})`);
  console.log(`  distinct gen_id    : ${distinctGenId}/${ok.length}`);
  console.log(`  reasoning present  : ${withReasoning}/${ok.length}   (the @on toggle firing)`);
  console.log(`  prompt_tokens      : ${promptTokSet.join(', ')}   (identical prompt -> expect one value)`);
  console.log(`  completion_tokens  : min ${Math.min(...completions)} / median ${med(completions)} / max ${Math.max(...completions)}   (spread = independent sampling)`);
  console.log(`  well-formed <dec>  : ${parsed.length}/${ok.length}   range ${parsed.length ? `[${Math.min(...parsed)}, ${Math.max(...parsed)}]` : '—'}`);
  if (truncated) console.log(`  truncated to cap   : ${truncated}  (max_tokens=${MAXTOK}; ok for this test)`);
}

// ---- cross-provider verdicts -------------------------------------------------
console.log('\n================ CROSS-PROVIDER ================');
const di = summary['DeepInfra'], sf = summary['SiliconFlow'];
const bothReachable = di.ok > 0 && sf.ok > 0;
const tokensMatch = bothReachable &&
  di.promptTokens.length === 1 && sf.promptTokens.length === 1 &&
  di.promptTokens[0] === sf.promptTokens[0];
const noCollapse = di.biggestCluster <= 1 && sf.biggestCluster <= 1;
const reasoningBoth = bothReachable &&
  Number(di.reasoningPresent.split('/')[0]) === di.ok &&
  Number(sf.reasoningPresent.split('/')[0]) === sf.ok;

console.log(`  both providers serve qwen3-32b@on   : ${bothReachable ? 'YES' : 'NO'}`);
console.log(`  no concurrent-collapse either side  : ${noCollapse ? 'YES' : 'NO'}`);
console.log(`  reasoning fires on both             : ${reasoningBoth ? 'YES' : 'NO'}`);
console.log(`  prompt_tokens identical across both : ${tokensMatch ? `YES (${di.promptTokens[0]})` : `NO (DeepInfra ${JSON.stringify(di.promptTokens)} vs SiliconFlow ${JSON.stringify(sf.promptTokens)})`}`);
const clean = bothReachable && noCollapse && reasoningBoth && tokensMatch;
console.log(`\n  VERDICT: split is ${clean ? 'CLEAN — safe to load-balance qwen@on across both' : 'NOT clean — see the NO above before splitting'}`);
console.log(`  probe cost $${cost.spentUsd.toFixed(4)}`);

writeFileSync(resolve(OUTDIR, 'qwen_provider_probe_summary.json'),
  JSON.stringify({ concurrent: CONCURRENT, maxTokens: MAXTOK, verdictClean: clean, summary }, null, 1));
console.log(`\n  full per-call log : pilot5/qwen_provider_probe.jsonl`);
console.log(`  summary written   : pilot5/qwen_provider_probe_summary.json`);
