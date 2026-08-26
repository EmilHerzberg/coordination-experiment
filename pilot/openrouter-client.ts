// OpenRouter adapter implementing the N&S `LLMClient` interface.
// - provider pinning (order + allow_fallbacks:false), logged provider actually used
// - reasoning visibility enabled for reasoning arms; toggle off for the qwen off-arm
// - NO tools forwarded to the API (satisfies "no retrieval/web tools attached")
// - every call appended as one JSONL line with full request/response/usage/cost
// - hard cost cap: throws once cumulative spend reaches the cap
import { appendFileSync } from 'node:fs';
import type { GenerateRequest, GenerateResponse, LLMClient } from '../src/types.js';

export type ReasoningMode = 'reason_on' | 'reason_off' | 'none';

export interface CostState {
  spentUsd: number;
  capUsd: number;
}

export class CostCapError extends Error {}

const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';

function sleep(ms: number): Promise<void> {
  return new Promise(res => setTimeout(res, ms));
}

export class OpenRouterClient implements LLMClient {
  private seq = 0;
  private curConfig = 'unknown';
  private curEvidence = 'unknown';

  constructor(
    private readonly opts: {
      modelKey: string; // e.g. "qwen3-32b@on"
      modelSlug: string;
      provider: string; // pinned provider name
      mode: ReasoningMode;
      noThinkInjection?: boolean; // append " /no_think" to user prompt (qwen off fallback)
      apiKey: string;
      logPath: string;
      cost: CostState;
      retryPass?: boolean;
    },
  ) {}

  /** Reset per-structure state so call_seq is meaningful within one structure run. */
  beginStructureRun(configName: string, evidenceId: string): void {
    this.seq = 0;
    this.curConfig = configName;
    this.curEvidence = evidenceId;
  }

  async generate(req: GenerateRequest): Promise<GenerateResponse> {
    if (this.opts.cost.spentUsd >= this.opts.cost.capUsd) {
      throw new CostCapError(
        `Cost cap reached: $${this.opts.cost.spentUsd.toFixed(4)} >= $${this.opts.cost.capUsd}`,
      );
    }
    const callSeq = this.seq++;
    const userPrompt =
      this.opts.noThinkInjection ? `${req.userPrompt} /no_think` : req.userPrompt;

    const body: Record<string, unknown> = {
      model: this.opts.modelSlug,
      messages: [
        { role: 'system', content: req.systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      temperature: req.temperature ?? 0.7,
      max_tokens: req.maxTokens,
      usage: { include: true },
      provider: { order: [this.opts.provider], allow_fallbacks: false },
      // NOTE: req.tools is deliberately ignored — no tools attached to any call.
    };
    if (this.opts.mode === 'reason_on') body.reasoning = { enabled: true };
    else if (this.opts.mode === 'reason_off') body.reasoning = { enabled: false };
    // mode 'none' -> omit reasoning entirely.

    const startedAt = Date.now();
    let status = 0;
    let data: any = null;
    let errText = '';
    let attempts = 0;
    for (let attempt = 0; attempt < 3; attempt++) {
      attempts = attempt + 1;
      try {
        const resp = await fetch(OPENROUTER_URL, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${this.opts.apiKey}`,
            'Content-Type': 'application/json',
            'X-Title': 'thesis-coordination-pilot',
          },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(180000), // 180s per attempt; covers long reasoning streams (deepseek-r1)
        });
        status = resp.status;
        if (resp.ok) {
          data = await resp.json();
          break;
        }
        errText = await resp.text();
        if (status === 429 || status >= 500) {
          await sleep(700 * (attempt + 1) * (attempt + 1));
          continue;
        }
        break; // non-retryable 4xx
      } catch (e: any) {
        errText = `${e?.name}: ${e?.message}`;
        await sleep(700 * (attempt + 1) * (attempt + 1));
      }
    }
    const durationMs = Date.now() - startedAt;

    if (!data) {
      // Log the failure and rethrow so the structure run records it.
      this.log({
        ok: false,
        call_seq: callSeq,
        config: this.curConfig,
        evidence_id: this.curEvidence,
        model_key: this.opts.modelKey,
        model_slug: this.opts.modelSlug,
        provider_pinned: this.opts.provider,
        http_status: status,
        error: errText.slice(0, 800),
        attempts,
        agent_role: req.metadata?.agentRole,
        system_prompt: req.systemPrompt,
        user_prompt: userPrompt,
        max_tokens: req.maxTokens,
        temperature: req.temperature,
        duration_ms: durationMs,
      });
      throw new Error(`OpenRouter call failed (status ${status}): ${errText.slice(0, 200)}`);
    }

    const choice = data.choices?.[0] ?? {};
    const msg = choice.message ?? {};
    const content: string = typeof msg.content === 'string' ? msg.content : '';
    const reasoning: string = typeof msg.reasoning === 'string' ? msg.reasoning : '';
    const reasoningDetails = msg.reasoning_details ?? null;
    const thinkMatch = content.match(/<think>([\s\S]*?)<\/think>/i);
    const thinkText = thinkMatch ? thinkMatch[1] : '';
    const usage = data.usage ?? {};
    const reasoningTokens =
      usage.completion_tokens_details?.reasoning_tokens ??
      usage.reasoning_tokens ??
      0;
    const costUsd = typeof usage.cost === 'number' ? usage.cost : 0;

    // Text for the harness's parser: prefer visible content; if it lacks the
    // FINAL_PROBABILITY trailer but the reasoning channel has it (e.g. some
    // providers emit the answer inside <reasoning>), append reasoning so the
    // config's parseFinalProbability can still succeed.
    let parseText = content;
    if (!/FINAL_PROBABILITY/i.test(content)) {
      const extra = reasoning || thinkText;
      if (extra) parseText = `${content}\n${extra}`;
    }

    this.opts.cost.spentUsd += costUsd;

    this.log({
      ok: true,
      call_seq: callSeq,
      config: this.curConfig,
      evidence_id: this.curEvidence,
      model_key: this.opts.modelKey,
      model_slug: this.opts.modelSlug,
      mode: this.opts.mode,
      no_think_injection: !!this.opts.noThinkInjection,
      provider_pinned: this.opts.provider,
      provider_returned: data.provider ?? null,
      gen_id: data.id ?? null,
      agent_role: req.metadata?.agentRole ?? null,
      temperature: req.temperature,
      max_tokens: req.maxTokens,
      finish_reason: choice.finish_reason ?? null,
      native_finish_reason: choice.native_finish_reason ?? null,
      system_prompt: req.systemPrompt,
      user_prompt: userPrompt,
      content,
      content_len: content.length,
      reasoning_text: reasoning,
      reasoning_text_len: reasoning.length,
      think_tag_len: thinkText.length,
      reasoning_details_present: !!reasoningDetails,
      has_final_probability_in_content: /FINAL_PROBABILITY/i.test(content),
      has_final_probability_in_reasoning: /FINAL_PROBABILITY/i.test(reasoning || thinkText),
      has_tagged_decision_in_content: /<decision>[\s\S]*?<\/decision>/i.test(content),
      has_tagged_decision_in_reasoning: /<decision>[\s\S]*?<\/decision>/i.test(reasoning || thinkText),
      usage,
      prompt_tokens: usage.prompt_tokens ?? null,
      completion_tokens: usage.completion_tokens ?? null,
      total_tokens: usage.total_tokens ?? null,
      reasoning_tokens: reasoningTokens,
      cost_usd: costUsd,
      cost_spent_cumulative: this.opts.cost.spentUsd,
      duration_ms: durationMs,
      attempts,
    });

    return {
      text: parseText,
      toolCalls: [],
      usage: {
        promptTokens: usage.prompt_tokens ?? 0,
        completionTokens: usage.completion_tokens ?? 0,
        totalTokens: usage.total_tokens ?? 0,
      },
      costUsd,
      modelId: `${this.opts.modelSlug}@${data.provider ?? this.opts.provider}`,
      durationMs,
      // Five-bucket ledger reads these (raw content, no parse hacks applied).
      extras: { reasoningTokens, reasoningText: reasoning || thinkText, content, finishReason: choice.finish_reason ?? null },
    } as GenerateResponse & { extras: { reasoningTokens: number; reasoningText: string; content: string; finishReason: string | null } };
  }

  private log(record: Record<string, unknown>): void {
    appendFileSync(this.opts.logPath, JSON.stringify({ ts: new Date().toISOString(), ...record }) + '\n');
  }
}
