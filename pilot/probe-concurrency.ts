// Does DeepSeek V3 survive CONCURRENT requests on SiliconFlow (fp8) where it
// breaks down on DeepInfra (fp4)?
//
// Both providers are probed in the same session and interleaved, so provider
// load at this moment cannot favour one of them. Real MSFT level-1 evidence,
// the real worker prompt. max_tokens is capped low so a runaway generation is
// still detected (finish_reason=length) without paying for 8,000 tokens of it.
//
//   node --experimental-transform-types --import ./register.mjs probe-concurrency.ts
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { OpenRouterClient, type CostState } from './openrouter-client.js';
import { getKey } from './env.js';
import { workerPrompt } from './prompts.js';
import { renderSegments } from './ledger5.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUTDIR = resolve(HERE, '..', '..', 'pilot5');
const pack = JSON.parse(readFileSync(resolve(OUTDIR, 'evidence_packs.json'), 'utf-8'))[0];
const prompt = workerPrompt({ id: pack.id, text: pack.level1 });
const SYS = renderSegments(prompt.system);
const USR = renderSegments(prompt.user);

const CONCURRENT = Number(process.env.PROBE_CONCURRENT ?? '3');
const ROUNDS = Number(process.env.PROBE_ROUNDS ?? '2');
const MAXTOK = Number(process.env.PROBE_MAXTOK ?? '2000');

const cost: CostState = { spentUsd: 0, capUsd: 1.0 };
const PROVIDERS = [
  { label: 'DeepInfra fp4', provider: 'DeepInfra' },
  { label: 'SiliconFlow fp8', provider: 'SiliconFlow' },
];

interface R { ok: boolean; finish: string; completion: number; chars: number; head: string }

async function one(provider: string, i: number): Promise<R> {
  const c = new OpenRouterClient({
    modelKey: `ds@${provider}`, modelSlug: 'deepseek/deepseek-chat-v3-0324',
    provider, mode: 'none', apiKey: getKey(),
    logPath: resolve(OUTDIR, 'concurrency_probe.jsonl'), cost,
  });
  c.beginStructureRun('probe', pack.id);
  try {
    const r = await c.generate({
      systemPrompt: SYS, userPrompt: USR, tools: [], maxTokens: MAXTOK, temperature: 0.7,
      metadata: { agentRole: `${provider}_${i}`, configName: 'probe' },
    });
    const t = r.text.trim();
    const ok = /<reason>[\s\S]*?<\/reason>[\s\S]*?<decision>[\s\S]*?<\/decision>/i.test(t);
    const extras = (r as any).extras;
    return { ok, finish: extras?.finishReason ?? '?', completion: r.usage.completionTokens,
             chars: t.length, head: t.slice(0, 55).replace(/\s+/g, ' ') };
  } catch (e: any) {
    return { ok: false, finish: 'ERROR', completion: 0, chars: 0, head: String(e?.message).slice(0, 55) };
  }
}

const tally: Record<string, { ok: number; n: number; runaway: number }> = {};
for (const p of PROVIDERS) tally[p.label] = { ok: 0, n: 0, runaway: 0 };

for (let round = 1; round <= ROUNDS; round++) {
  for (const p of PROVIDERS) {
    console.log(`\nround ${round} — ${p.label}: ${CONCURRENT} CONCURRENT requests`);
    const rs = await Promise.all(Array.from({ length: CONCURRENT }, (_, i) => one(p.provider, i)));
    for (const [i, r] of rs.entries()) {
      const verdict = r.ok ? 'OK        ' : (r.finish === 'length' ? 'RUNAWAY   ' : 'DEGENERATE');
      console.log(`   ${i + 1}: ${verdict} finish=${r.finish.padEnd(6)} completion=${String(r.completion).padStart(5)} chars=${String(r.chars).padStart(6)} | ${r.head}`);
      tally[p.label].n++;
      if (r.ok) tally[p.label].ok++;
      if (!r.ok && r.finish === 'length') tally[p.label].runaway++;
    }
  }
}

console.log('\n================ SUMMARY ================');
for (const p of PROVIDERS) {
  const t = tally[p.label];
  console.log(`  ${p.label.padEnd(18)} ${t.ok}/${t.n} well-formed   (${t.runaway} runaway to the cap)`);
}
console.log(`  probe cost $${cost.spentUsd.toFixed(4)}`);
writeFileSync(resolve(OUTDIR, 'concurrency_probe_summary.json'), JSON.stringify(tally, null, 1));
