// Pure Prometheus scraping/parsing for the vLLM and SGLang enrichment tier.
// No JSX, no OpenCode/solid-js imports — kept separate so it can be unit
// tested (test/prometheus.test.mjs) without pulling in the TUI runtime, which
// isn't installed locally (OpenCode provides it at load time).

export interface PromSpec {
  prefix: string
  promptTokens: string
  generationTokens: string
  cachedTokens: string
  ttftSum: string
  ttftCount: string
}

export const VLLM_SPEC: PromSpec = {
  prefix: "vllm:",
  promptTokens: "vllm:prompt_tokens_total",
  generationTokens: "vllm:generation_tokens_total",
  cachedTokens: "vllm:prompt_tokens_cached_total",
  ttftSum: "vllm:time_to_first_token_seconds_sum",
  ttftCount: "vllm:time_to_first_token_seconds_count",
}

export const SGLANG_SPEC: PromSpec = {
  prefix: "sglang:",
  promptTokens: "sglang:prompt_tokens_total",
  generationTokens: "sglang:generation_tokens_total",
  cachedTokens: "sglang:cached_tokens_total",
  ttftSum: "sglang:time_to_first_token_seconds_sum",
  ttftCount: "sglang:time_to_first_token_seconds_count",
}

export interface PromSample {
  prompt: number
  generation: number
  cached: number
  ttftSum: number
  ttftCount: number
}

/**
 * Sums every series sharing `name`, ignoring labels — both engines split some
 * counters by rank or cache source (sglang:cached_tokens_total{cache_source=…}),
 * and the total across them is what matters. A Prometheus client also emits a
 * `_created` line per counter holding a unix timestamp, so the match must not
 * accept a longer name this is only a prefix of.
 */
export function sumLabeledMetric(text: string, name: string): number {
  let total = 0
  for (const line of text.split("\n")) {
    if (line.startsWith("#") || !line.startsWith(name)) continue
    const rest = line.slice(name.length)
    if (rest.length > 0 && rest[0] !== "{" && rest[0] !== " ") continue // longer name, same prefix
    const sp = line.lastIndexOf(" ")
    if (sp === -1) continue
    const v = Number(line.slice(sp + 1))
    if (!Number.isNaN(v)) total += v
  }
  return total
}

export function parsePromSample(text: string, spec: PromSpec): PromSample | null {
  if (!text.includes(spec.prefix)) return null
  return {
    prompt: sumLabeledMetric(text, spec.promptTokens),
    generation: sumLabeledMetric(text, spec.generationTokens),
    cached: sumLabeledMetric(text, spec.cachedTokens),
    ttftSum: sumLabeledMetric(text, spec.ttftSum),
    ttftCount: sumLabeledMetric(text, spec.ttftCount),
  }
}

export async function fetchPromSample(base: string, spec: PromSpec): Promise<PromSample | null> {
  const ctrl = new AbortController()
  const t = setTimeout(() => ctrl.abort(), 2500)
  try {
    const res = await fetch(`${base}/metrics`, { signal: ctrl.signal, headers: { connection: "close" } })
    if (!res.ok) return null
    return parsePromSample(await res.text(), spec)
  } catch {
    return null
  } finally {
    clearTimeout(t)
  }
}

export interface PromDiff {
  completionTokens: number
  promptTokens: number
  cachedTokens: number
  ttftAvg?: number
}

/**
 * Diffs two samples taken across a turn boundary. Returns null when there is
 * nothing to attribute to this turn: no generation advanced (a cache hit
 * answered from a still-open connection, or a concurrent caller's turn beat
 * this one to the scrape) or the counters ran backwards (the server restarted).
 */
export function diffPromSamples(prev: PromSample, now: PromSample): PromDiff | null {
  if (now.generation < prev.generation) return null // counters reset
  const completionTokens = now.generation - prev.generation
  if (completionTokens <= 0) return null
  const dTtftCount = now.ttftCount - prev.ttftCount
  return {
    completionTokens,
    promptTokens: now.prompt - prev.prompt,
    cachedTokens: now.cached - prev.cached,
    ttftAvg: dTtftCount > 0 ? (now.ttftSum - prev.ttftSum) / dTtftCount : undefined,
  }
}
