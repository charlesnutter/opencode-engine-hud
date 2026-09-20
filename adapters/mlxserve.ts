// mlx-serve enrichment — /v1/metrics/requests.
//
// mlx-serve (raspoli/mlx-serve) is an Apple-Silicon manager that hot-swaps MLX
// models, spawning `mlx_lm.server` as a subprocess and wrapping it with the
// observability that server lacks. Not to be confused with `mlx_lm.server`
// itself, which exposes nothing beyond per-response `usage`.
//
// Its shape is unlike any other engine here: instead of cumulative counters to
// difference, or a single last-request slot, it keeps a KEYED HISTORY of
// recent requests, newest first, each with its own `request_id`. That makes
// the freshness check exact — we report a record only once, by id — rather
// than inferred from a counter delta.
//
// Two traps, both found on a live v0.1.0 server and neither documented:
//
//   1. The unprefixed aliases (/health, /status, /metrics, /events,
//      /dashboard) are exact paths only. `/metrics/requests` 404s; the working
//      path is `/v1/metrics/requests`.
//   2. `ttft_ms` and `tokens_per_second` mean different things depending on
//      whether the request streamed. See `streamed` below.
//
// No JSX/solid-js imports, so it stays unit-testable (test/mlxserve.test.mjs).

import { httpJson, type HttpOptions } from "../http"

/** One record from /v1/metrics/requests, as the server names its fields. */
export interface MlxServeRequest {
  requestId: string
  model: string
  totalDurationMs: number
  ttftMs: number | null
  tokensPerSecond: number | null
  promptTokens: number | null
  completionTokens: number | null
  statusCode: number
  error: string | null
  coldStart: boolean
}

export interface MlxServeTurn {
  /** The record this describes; the caller stores it to avoid reporting twice. */
  requestId: string
  completionTokens: number
  /**
   * Absent on a streamed request: mlx_lm.server reports no prompt count when
   * streaming, so the field comes back null and there is nothing to show.
   */
  promptTokens?: number
  /**
   * True decode rate, excluding prefill. Only on a streamed request, where the
   * server computes it as completion / (duration - TTFT) — verified exactly
   * against a live record: 100 tokens, 463.5ms duration, 198.8ms TTFT, 377.8
   * tok/s reported, and 100/(0.4635-0.1988) == 377.8.
   */
  decodeTokS?: number
  /**
   * Whole-request rate including prefill, on a NON-streamed request, where the
   * server's own figure is completion / duration. Kept separate from
   * `decodeTokS` because the two are not comparable and must not be shown
   * under the same label.
   */
  overallTokS?: number
  /** Real TTFT, streamed requests only — see `streamed`. */
  ttft?: number
  totalS: number
  /**
   * Whether the request streamed, which decides what the two figures above
   * mean. A non-streamed request has no first-token event to observe, so the
   * server stamps TTFT at completion and `ttft_ms` comes back equal to
   * `total_duration_ms` (measured: 4778.7 and 4778.7). Its `tokens_per_second`
   * is then a whole-request rate, not a decode rate. OpenCode streams, so the
   * streamed branch is the normal one.
   */
  streamed: boolean
  /**
   * The model was loaded on demand during this request, so the duration
   * includes model load — 4.8s against 0.46s once warm, for the same server.
   * Worth showing, or the turn reads as a collapse in performance.
   */
  coldStart: boolean
}

/**
 * How close `ttft_ms` must get to `total_duration_ms` before we treat the two
 * as the same instant, i.e. a non-streamed request. A genuinely streamed turn
 * leaves a real decode window between them.
 */
const NON_STREAM_RATIO = 0.99

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null)

export function parseMlxServeRequests(raw: unknown): MlxServeRequest[] | null {
  const list = (raw as { requests?: unknown } | null | undefined)?.requests
  if (!Array.isArray(list)) return null // not this endpoint
  const out: MlxServeRequest[] = []
  for (const entry of list) {
    if (!entry || typeof entry !== "object") continue
    const r = entry as Record<string, unknown>
    if (typeof r.request_id !== "string") continue
    out.push({
      requestId: r.request_id,
      model: typeof r.model === "string" ? r.model : "",
      totalDurationMs: num(r.total_duration_ms) ?? 0,
      ttftMs: num(r.ttft_ms),
      tokensPerSecond: num(r.tokens_per_second),
      promptTokens: num(r.prompt_tokens),
      completionTokens: num(r.completion_tokens),
      statusCode: num(r.status_code) ?? 200,
      error: typeof r.error === "string" ? r.error : null,
      coldStart: r.cold_start === true,
    })
  }
  return out
}

/**
 * Picks the newest usable record and turns it into this turn's figures.
 *
 * `lastSeenId` is the record already reported for a previous turn. The history
 * is newest-first, so if the head still carries that id nothing new has
 * completed and this returns null — the universal line stands instead. Records
 * that failed, or produced no tokens, are skipped rather than reported.
 */
export function mlxServeTurn(
  records: MlxServeRequest[],
  lastSeenId: string | undefined
): MlxServeTurn | null {
  const r = records.find(
    (x) => x.statusCode < 400 && x.error === null && (x.completionTokens ?? 0) > 0
  )
  if (!r) return null
  if (lastSeenId !== undefined && r.requestId === lastSeenId) return null

  const totalS = r.totalDurationMs / 1000
  // TTFT at or above the whole duration means it was stamped at completion.
  const streamed =
    r.ttftMs !== null && r.totalDurationMs > 0 && r.ttftMs < r.totalDurationMs * NON_STREAM_RATIO

  const rate = r.tokensPerSecond !== null && r.tokensPerSecond > 0 ? r.tokensPerSecond : undefined

  return {
    requestId: r.requestId,
    completionTokens: r.completionTokens as number,
    promptTokens: r.promptTokens ?? undefined,
    decodeTokS: streamed ? rate : undefined,
    overallTokS: streamed ? undefined : rate,
    ttft: streamed ? (r.ttftMs as number) / 1000 : undefined,
    totalS,
    streamed,
    coldStart: r.coldStart,
  }
}

export async function fetchMlxServeRequests(
  base: string,
  model?: string,
  apiKey?: string,
  opts?: HttpOptions
): Promise<MlxServeRequest[] | null> {
  // Note the /v1 prefix: /metrics/requests without it is a 404.
  const q = new URLSearchParams({ last_n: "5" })
  if (model) q.set("model", model)
  // MLX_API_KEY, when set on the server, guards every route on this router.
  const headers = apiKey ? { ...opts?.headers, authorization: `Bearer ${apiKey}` } : opts?.headers
  return parseMlxServeRequests(
    await httpJson(`${base}/v1/metrics/requests?${q}`, { ...opts, headers })
  )
}
