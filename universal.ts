// The universal telemetry layer — everything derived from OpenCode's own
// per-turn events, with no engine endpoint involved. Every provider gets this,
// including ones with no adapter here at all.
//
// Kept out of tui.tsx (which imports the TUI runtime) so it can be unit
// tested. It was the one tier without tests, and that is exactly where a real
// bug hid: the decode rate divided only the VISIBLE output tokens by a window
// that also covered the model's thinking, understating reasoning-model rates
// several-fold. See test/universal.test.mjs.

export const nn = (v: unknown, d = 1) =>
  typeof v === "number" && isFinite(v) ? v.toFixed(d) : "?"
export const ni = (v: unknown) =>
  typeof v === "number" && isFinite(v) ? String(Math.round(v)) : "?"

export function short(model: string): string {
  const tail = model.split("/").pop() ?? model
  return tail.length > 24 ? tail.slice(0, 23) + "…" : tail
}

async function getJson(url: string, headers?: Record<string, string>): Promise<any | null> {
  const ctrl = new AbortController()
  const t = setTimeout(() => ctrl.abort(), 2500)
  try {
    const res = await fetch(url, { signal: ctrl.signal, headers })
    if (!res.ok) return null
    return await res.json()
  } catch {
    return null
  } finally {
    clearTimeout(t)
  }
}

// ---- Tier 1: universal, from OpenCode's own per-turn events -----------------
export interface Turn {
  startAt?: number // request start (message.time.created), for TTFT
  firstAt?: number // first streamed delta
  lastAt?: number // last streamed delta
  bytes: number // streamed bytes, for an estimate when usage is absent
}

/**
 * Decode rate, TTFT and total time from OpenCode's own per-turn timing —
 * `time.created`/`time.completed` on the message, and the streaming-delta
 * marks in `turn`. Shared by the universal line and by enrichment tiers that
 * have exact token counts but no per-request timing of their own (vLLM,
 * SGLang): their Prometheus counters need continuous polling to split decode
 * from prefill, which nothing here does, but OpenCode's own event stream
 * already has it for free.
 */
export function turnRate(tokens: number, info: any, turn?: Turn): { decodeTokS?: number; ttft?: number; total?: number } {
  const created = info?.time?.created
  const completed = info?.time?.completed
  const total = typeof created === "number" && typeof completed === "number" ? (completed - created) / 1000 : undefined

  let ttft: number | undefined
  let decodeTokS: number | undefined
  if (turn) {
    if (turn.firstAt && turn.startAt) ttft = (turn.firstAt - turn.startAt) / 1000
    if (turn.firstAt && turn.lastAt && turn.lastAt > turn.firstAt && tokens > 0) {
      decodeTokS = tokens / ((turn.lastAt - turn.firstAt) / 1000)
    }
  }
  // Fall back to whole-request rate if the stream window was too short to time.
  if (decodeTokS === undefined && tokens > 0 && total && total > 0) decodeTokS = tokens / total
  return { decodeTokS, ttft, total }
}

export function universalLine(provider: string, model: string, info: any, turn?: Turn): string {
  const out: number = info?.tokens?.output ?? 0
  const reason: number = info?.tokens?.reasoning ?? 0
  // Reasoning tokens are decoded tokens: they are produced one at a time
  // inside the very window this rate is measured over. Dividing only the
  // VISIBLE output by that window understates the rate by however much of the
  // turn was spent thinking. Measured against Splash (Qwen3.8-27B): 889 of
  // 1247 generated tokens were reasoning, and this line reported 11.4 tok/s
  // where the engine's own log said 39.7 over the same 31.4s window. The
  // visible/think split stays in the totals line below; only the rate's
  // numerator is corrected.
  const generated = out + reason
  const { decodeTokS, ttft, total } = turnRate(generated, info, turn)

  const think = reason > 0 ? ` (+${ni(reason)} think)` : ""
  const rate =
    decodeTokS !== undefined
      ? `${nn(decodeTokS)} tok/s${ttft !== undefined ? `  ttft ${nn(ttft, 2)}s` : ""}`
      : ttft !== undefined
        ? `ttft ${nn(ttft, 2)}s`
        : ""
  const totals = `${ni(out)} tok${think}${total !== undefined ? `  ${nn(total, 2)}s` : ""}`
  return [`${provider}  ${short(model)}`, rate, totals].filter(Boolean).join("\n")
}

