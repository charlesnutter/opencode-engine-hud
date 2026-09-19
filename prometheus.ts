// Pure Prometheus scraping/parsing for the vLLM and SGLang enrichment tier.
// No JSX, no OpenCode/solid-js imports — kept separate so it can be unit
// tested (test/prometheus.test.mjs) without pulling in the TUI runtime, which
// isn't installed locally (OpenCode provides it at load time).

export interface PromSpec {
  prefix: string
  promptTokens: string
  generationTokens: string
  /** Absent on engines with no prefix/prompt-cache counter (vllm-mlx). */
  cachedTokens?: string
  ttftSum: string
  ttftCount: string
  /**
   * End-to-end request-duration histogram, where the engine publishes one.
   * With it (and a TTFT histogram) a single request's decode window can be
   * derived exactly — duration minus TTFT — instead of leaning on the
   * caller's own wall clock.
   */
  durationSum?: string
  durationCount?: string
  /**
   * Separate prefill and decode time histograms, where an engine publishes
   * them (LMDeploy). Better than duration-minus-TTFT: each phase is timed by
   * the engine directly, so both rates are its own measurement.
   */
  prefillTimeSum?: string
  prefillTimeCount?: string
  decodeTimeSum?: string
  decodeTimeCount?: string
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

/**
 * Aphrodite is a vLLM fork and inherits its metric shape verbatim, under its
 * own prefix — so it is the same spec with `vllm:` swapped for `aphrodite:`.
 */
export const APHRODITE_SPEC: PromSpec = {
  prefix: "aphrodite:",
  promptTokens: "aphrodite:prompt_tokens_total",
  generationTokens: "aphrodite:generation_tokens_total",
  cachedTokens: "aphrodite:prompt_tokens_cached_total",
  ttftSum: "aphrodite:time_to_first_token_seconds_sum",
  ttftCount: "aphrodite:time_to_first_token_seconds_count",
}

/**
 * vllm-mlx (the MLX-native Apple Silicon server, not vllm-metal, which runs
 * upstream vLLM itself and so uses VLLM_SPEC). Underscore-prefixed
 * prometheus_client names, labelled by endpoint/stream, and — unusually — it
 * publishes BOTH a TTFT and an end-to-end duration histogram, so a single
 * turn's decode window is recoverable exactly. No prompt-cache counter.
 */
export const VLLM_MLX_SPEC: PromSpec = {
  prefix: "vllm_mlx_",
  promptTokens: "vllm_mlx_prompt_tokens_total",
  generationTokens: "vllm_mlx_completion_tokens_total",
  ttftSum: "vllm_mlx_inference_ttft_seconds_sum",
  ttftCount: "vllm_mlx_inference_ttft_seconds_count",
  durationSum: "vllm_mlx_inference_request_duration_seconds_sum",
  durationCount: "vllm_mlx_inference_request_duration_seconds_count",
}

/**
 * LMDeploy publishes the richest surface of these engines: alongside the usual
 * counters it times prefill and decode as separate histograms, so both rates
 * are engine-measured rather than derived. Needs `--enable-metrics` (off by
 * default); default port 23333. Names verified against
 * lmdeploy/metrics/loggers.py.
 */
export const LMDEPLOY_SPEC: PromSpec = {
  prefix: "lmdeploy:",
  promptTokens: "lmdeploy:prompt_tokens_total",
  generationTokens: "lmdeploy:generation_tokens_total",
  ttftSum: "lmdeploy:time_to_first_token_seconds_sum",
  ttftCount: "lmdeploy:time_to_first_token_seconds_count",
  durationSum: "lmdeploy:e2e_request_latency_seconds_sum",
  durationCount: "lmdeploy:e2e_request_latency_seconds_count",
  prefillTimeSum: "lmdeploy:request_prefill_time_seconds_sum",
  prefillTimeCount: "lmdeploy:request_prefill_time_seconds_count",
  decodeTimeSum: "lmdeploy:request_decode_time_seconds_sum",
  decodeTimeCount: "lmdeploy:request_decode_time_seconds_count",
}

export interface PromSample {
  prompt: number
  generation: number
  cached: number
  ttftSum: number
  ttftCount: number
  durationSum: number
  durationCount: number
  prefillTimeSum: number
  prefillTimeCount: number
  decodeTimeSum: number
  decodeTimeCount: number
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
    cached: spec.cachedTokens ? sumLabeledMetric(text, spec.cachedTokens) : 0,
    ttftSum: sumLabeledMetric(text, spec.ttftSum),
    ttftCount: sumLabeledMetric(text, spec.ttftCount),
    durationSum: spec.durationSum ? sumLabeledMetric(text, spec.durationSum) : 0,
    durationCount: spec.durationCount ? sumLabeledMetric(text, spec.durationCount) : 0,
    prefillTimeSum: spec.prefillTimeSum ? sumLabeledMetric(text, spec.prefillTimeSum) : 0,
    prefillTimeCount: spec.prefillTimeCount ? sumLabeledMetric(text, spec.prefillTimeCount) : 0,
    decodeTimeSum: spec.decodeTimeSum ? sumLabeledMetric(text, spec.decodeTimeSum) : 0,
    decodeTimeCount: spec.decodeTimeCount ? sumLabeledMetric(text, spec.decodeTimeCount) : 0,
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
  /** Mean TTFT over the requests in this window; exact when `ttftExact`. */
  ttft?: number
  /**
   * True when exactly one request landed in the window, which makes `ttft`
   * and `durationS` that request's own values rather than an average over
   * several. OpenCode issues one request per turn, so this is the norm.
   */
  ttftExact: boolean
  durationS?: number
  /**
   * Decode rate measured by the engine itself: tokens over (duration - TTFT),
   * i.e. excluding prefill. Only when the engine publishes both histograms
   * and exactly one request landed, so it describes this turn alone.
   */
  decodeTokS?: number
  /**
   * Prefill rate, when the engine times prefill as its own phase (LMDeploy).
   * Nothing else here can produce this from Prometheus alone.
   */
  prefillTokS?: number
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
  const ttft = dTtftCount > 0 ? (now.ttftSum - prev.ttftSum) / dTtftCount : undefined
  const dDurCount = now.durationCount - prev.durationCount
  const durationS = dDurCount > 0 ? (now.durationSum - prev.durationSum) / dDurCount : undefined

  // Exactly one request in the window makes these figures this turn's own.
  const exact = dTtftCount === 1
  const promptTokens = now.prompt - prev.prompt

  let decodeTokS: number | undefined
  let prefillTokS: number | undefined

  // Best case: the engine timed decode as its own phase (LMDeploy).
  const dDecCount = now.decodeTimeCount - prev.decodeTimeCount
  if (dDecCount === 1) {
    const decodeS = now.decodeTimeSum - prev.decodeTimeSum
    if (decodeS > 0) decodeTokS = completionTokens / decodeS
  }
  // Otherwise derive the decode window as duration minus TTFT (vllm-mlx).
  if (decodeTokS === undefined && exact && dDurCount === 1 && ttft !== undefined && durationS !== undefined) {
    const decodeWindow = durationS - ttft
    if (decodeWindow > 0) decodeTokS = completionTokens / decodeWindow
  }
  const dPreCount = now.prefillTimeCount - prev.prefillTimeCount
  if (dPreCount === 1 && promptTokens > 0) {
    const prefillS = now.prefillTimeSum - prev.prefillTimeSum
    if (prefillS > 0) prefillTokS = promptTokens / prefillS
  }

  return {
    completionTokens,
    promptTokens,
    cachedTokens: now.cached - prev.cached,
    ttft,
    ttftExact: exact,
    durationS,
    decodeTokS,
    prefillTokS,
  }
}
