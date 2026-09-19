/** @jsxImportSource @opentui/solid */
// opencode-hud — persistent per-turn local-inference stats in the sidebar.
//
// Two tiers of data:
//   1. Universal layer — every provider OpenCode talks to. Built from OpenCode's
//      own events: `message.part.delta` gives streaming (first delta = TTFT,
//      byte deltas estimate live tokens) and `message.updated` gives the exact
//      final counts (tokens.output/input/reasoning) and wall time. No engine
//      endpoint needed, so Ollama, llama.cpp, MLX-LM, vLLM, SGLang and anything
//      OpenAI-compatible all work.
//   2. Per-engine enrichment — where an engine exposes richer server-side data,
//      fetch it and show that instead. MTPLX (/metrics) and oMLX (/api/status)
//      are wired; every other provider falls back to the universal line.
//
// The panel is keyed to provider+model: a switch replaces it, never blends.
//
// Config (plugin options in tui.json, or env fallback):
//   mtplxMetricsUrl (MTPLX_METRICS_URL)  default http://127.0.0.1:8000/metrics
//   omlxBaseUrl     (OMLX_BASE_URL)       default http://127.0.0.1:8099
//   omlxApiKey      (OMLX_API_KEY)        required to read oMLX; no default
//
//   "plugin": [["@charlesnutter/opencode-hud", { "omlxApiKey": "…" }]]
import type { TextRenderable } from "@opentui/core"
import type { TuiPlugin, TuiPluginModule } from "@opencode-ai/plugin/tui"
import { onCleanup } from "solid-js"
import { PromSpec, VLLM_SPEC, SGLANG_SPEC, APHRODITE_SPEC, VLLM_MLX_SPEC, LMDEPLOY_SPEC, fetchPromSample, diffPromSamples, PromSample } from "./prometheus"

interface Config {
  mtplxUrl: string
  omlxBase: string
  omlxKey: string
  llamacppBase: string
  vllmBase: string
  sglangBase: string
  vllmMlxBase: string
  aphroditeBase: string
  lmdeployBase: string
}

const nn = (v: unknown, d = 1) =>
  typeof v === "number" && isFinite(v) ? v.toFixed(d) : "?"
const ni = (v: unknown) =>
  typeof v === "number" && isFinite(v) ? String(Math.round(v)) : "?"

function short(model: string): string {
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
interface Turn {
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
function turnRate(tokens: number, info: any, turn?: Turn): { decodeTokS?: number; ttft?: number; total?: number } {
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

function universalLine(provider: string, model: string, info: any, turn?: Turn): string {
  const out: number = info?.tokens?.output ?? 0
  const reason: number = info?.tokens?.reasoning ?? 0
  const { decodeTokS, ttft, total } = turnRate(out, info, turn)

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

// ---- Tier 2: MTPLX enrichment — /metrics `latest`, per-request precise ------
async function mtplxLine(cfg: Config, model: string): Promise<string | null> {
  const body = await getJson(cfg.mtplxUrl)
  const l = body?.latest
  if (!l) return null
  let mtp = ""
  if (typeof l.verify_calls === "number" && l.verify_calls > 0 && typeof l.completion_tokens === "number") {
    const perPass = l.completion_tokens / l.verify_calls
    const acc = Array.isArray(l.mean_accept_probability_by_depth)
      ? l.mean_accept_probability_by_depth.map((p: number) => Math.round(p * 100)).join("/")
      : null
    mtp = `MTP ${nn(perPass, 2)}x${acc ? ` ${acc}%` : ""}`
  }
  const think =
    typeof l.reasoning_tokens === "number" && l.reasoning_tokens > 0 ? ` (+${ni(l.reasoning_tokens)} think)` : ""
  return [
    `MTPLX  ${short(model)}`,
    `${nn(l.decode_tok_s)} tok/s  ttft ${nn(l.ttft_s, 2)}s`,
    `prefill ${ni(l.prefill_tok_s)} tok/s`,
    `${ni(l.completion_tokens)} tok${think}  ${nn(l.request_elapsed_s, 2)}s`,
    mtp,
  ].filter(Boolean).join("\n")
}

// ---- Tier 2: oMLX enrichment — /api/status, differenced across the turn -----
interface OmlxSample {
  requests: number
  prompt: number
  completion: number
  cached: number
  avgGen: number
  avgPrefill: number
  model?: string
}
let omlxPrev: OmlxSample | undefined

async function omlxSample(cfg: Config): Promise<OmlxSample | null> {
  if (!cfg.omlxKey) return null
  const j = await getJson(`${cfg.omlxBase}/api/status`, { authorization: `Bearer ${cfg.omlxKey}` })
  if (!j) return null
  return {
    requests: j.total_requests ?? 0,
    prompt: j.total_prompt_tokens ?? 0,
    completion: j.total_completion_tokens ?? 0,
    cached: j.total_cached_tokens ?? 0,
    avgGen: j.avg_generation_tps ?? 0,
    avgPrefill: j.avg_prefill_tps ?? 0,
    model: j.loaded_models?.[0] ?? j.default_model,
  }
}

async function omlxLine(cfg: Config): Promise<string | null> {
  const s = await omlxSample(cfg)
  if (!s) return null
  const prev = omlxPrev
  omlxPrev = s
  if (!prev || prev.model !== s.model || s.requests <= prev.requests) {
    return [`oMLX  ${short(s.model ?? "")}`, `${nn(s.avgGen)} tok/s (server avg)`, `prefill ${ni(s.avgPrefill)} tok/s (avg)`]
      .join("\n")
  }
  const dReq = s.requests - prev.requests
  const comp = s.completion - prev.completion
  let decode = dReq === 1 ? s.avgGen * s.requests - prev.avgGen * prev.requests : NaN
  let prefill = dReq === 1 ? s.avgPrefill * s.requests - prev.avgPrefill * prev.requests : NaN
  if (!(decode > 0)) decode = s.avgGen
  if (!(prefill > 0)) prefill = s.avgPrefill
  const cached = s.cached - prev.cached
  return [
    `oMLX  ${short(s.model ?? "")}`,
    `${nn(decode)} tok/s`,
    `prefill ${ni(prefill)} tok/s`,
    `${ni(comp)} tok  (${ni(s.prompt - prev.prompt)} prompt${cached > 0 ? `, ${ni(cached)} cached` : ""})`,
  ].join("\n")
}

// ---- Tier 2: llama.cpp enrichment — /metrics, differenced across the turn --
// llama.cpp's counters are atomic at completion, exactly like oMLX's: they sit
// still while a request runs and jump once it lands (confirmed against a live
// server). So the same across-turn snapshot diff applies, with no need for the
// continuous /slots poll loop a true live ticker would require — the universal
// layer above already covers TTFT and a live estimate from OpenCode's own
// streaming events. Needs the server started with --metrics (off by default);
// unmetriced or unreachable servers just fail the fetch and fall back to the
// universal line.
interface LlamaCppCounters {
  promptTokens: number
  promptSeconds: number
  predictedTokens: number
  predictedSeconds: number
}
let llamacppPrev: LlamaCppCounters | undefined

async function getText(url: string): Promise<string | null> {
  const ctrl = new AbortController()
  const t = setTimeout(() => ctrl.abort(), 2500)
  try {
    const res = await fetch(url, { signal: ctrl.signal, headers: { connection: "close" } })
    if (!res.ok) return null
    return await res.text()
  } catch {
    return null
  } finally {
    clearTimeout(t)
  }
}

/** llama.cpp emits bare `name value` lines with no labels. */
function parsePrometheus(text: string): Record<string, number> {
  const out: Record<string, number> = {}
  for (const line of text.split("\n")) {
    if (line.startsWith("#") || !line.trim()) continue
    const sp = line.lastIndexOf(" ")
    if (sp === -1) continue
    const name = line.slice(0, sp).replace(/\{.*\}$/, "")
    const value = Number(line.slice(sp + 1))
    if (!Number.isNaN(value)) out[name] = value
  }
  return out
}

async function llamacppCounters(cfg: Config): Promise<LlamaCppCounters | null> {
  const text = await getText(`${cfg.llamacppBase}/metrics`)
  if (text === null) return null
  const v = parsePrometheus(text)
  return {
    promptTokens: v["llamacpp:prompt_tokens_total"] ?? 0,
    promptSeconds: v["llamacpp:prompt_seconds_total"] ?? 0,
    predictedTokens: v["llamacpp:tokens_predicted_total"] ?? 0,
    predictedSeconds: v["llamacpp:tokens_predicted_seconds_total"] ?? 0,
  }
}

async function llamacppLine(cfg: Config, model: string): Promise<string | null> {
  const now = await llamacppCounters(cfg)
  if (!now) return null // unreachable, or started without --metrics
  const prev = llamacppPrev
  llamacppPrev = now
  if (!prev || now.predictedTokens <= prev.predictedTokens) {
    // No baseline yet (first turn since launch), or nothing moved (answered
    // from cache faster than we could sample, or a concurrent caller's turn
    // already advanced the counters). The universal line still covers this
    // turn; the next one gets a clean diff.
    return null
  }
  const completionTokens = now.predictedTokens - prev.predictedTokens
  const decodeS = now.predictedSeconds - prev.predictedSeconds
  // The counter under-reports the prompt on a cache hit (it counts only what
  // was actually computed), but with no live /slots sample at hand to correct
  // it, this is what's available — same tradeoff the extension's adapter notes.
  const promptTokens = now.promptTokens - prev.promptTokens
  const prefillS = now.promptSeconds - prev.promptSeconds
  const decodeTokS = decodeS > 0 ? completionTokens / decodeS : undefined
  const prefillTokS = prefillS > 0 && promptTokens > 0 ? promptTokens / prefillS : undefined
  return [
    `llama.cpp  ${short(model)}`,
    decodeTokS !== undefined ? `${nn(decodeTokS)} tok/s` : "",
    prefillTokS !== undefined ? `prefill ${ni(prefillTokS)} tok/s` : "",
    `${ni(completionTokens)} tok  ${nn(decodeS + prefillS, 2)}s`,
  ].filter(Boolean).join("\n")
}

// ---- Tier 2: vLLM / SGLang enrichment — Prometheus, diffed across the turn -
// Both publish cumulative counters that, unlike llama.cpp/oMLX, advance DURING
// generation rather than only at completion. That only matters for a
// continuous poller, which this plugin does not run — the turn-boundary diff
// (prometheus.ts) still gives exact prompt/generation/cached token counts and
// the TTFT histogram's per-window average, none of which the universal layer
// can see at all. Decode rate and total time reuse OpenCode's own turn timing
// (`turnRate`, shared with the universal line): Prometheus alone needs
// mid-request polling to split decode from prefill, and OpenCode's streaming
// events already have that split for free.
//
// Not run against a live vLLM/SGLang server (both CUDA-only) — validated
// against real captured /metrics text in test/prometheus.test.mjs instead.
const promPrev = new Map<string, PromSample>() // keyed by provider id, not URL

async function prometheusLine(
  providerId: string,
  spec: PromSpec,
  base: string,
  label: string,
  model: string,
  info: any,
  turn?: Turn
): Promise<string | null> {
  const now = await fetchPromSample(base, spec)
  if (!now) return null
  const prev = promPrev.get(providerId)
  promPrev.set(providerId, now)
  if (!prev) return null // no baseline yet (first turn since launch)

  const diff = diffPromSamples(prev, now)
  if (!diff) return null

  // Prefer the engine's own measured decode rate (duration minus TTFT, one
  // request) where it publishes the histograms for it; otherwise fall back to
  // OpenCode's turn timing, which is all vLLM/SGLang can support.
  const fallback = turnRate(diff.completionTokens, info, turn)
  const decodeTokS = diff.decodeTokS ?? fallback.decodeTokS
  const total = diff.durationS ?? fallback.total
  const ttftLabel =
    diff.ttft !== undefined ? `  ttft ${nn(diff.ttft, 2)}s${diff.ttftExact ? "" : " (avg)"}` : ""
  return [
    `${label}  ${short(model)}`,
    decodeTokS !== undefined ? `${nn(decodeTokS)} tok/s${ttftLabel}` : ttftLabel.trim(),
    diff.prefillTokS !== undefined ? `prefill ${ni(diff.prefillTokS)} tok/s` : "",
    `${ni(diff.completionTokens)} tok  (${ni(diff.promptTokens)} prompt${diff.cachedTokens > 0 ? `, ${ni(diff.cachedTokens)} cached` : ""})${total !== undefined ? `  ${nn(total, 2)}s` : ""}`,
  ].filter(Boolean).join("\n")
}

interface Store { text: string; listeners: Set<() => void> }

function SidebarFooter(props: { api: Parameters<TuiPlugin>[0]; store: Store }) {
  let text: TextRenderable | undefined
  const sync = () => {
    if (!text) return
    text.content = props.store.text
    try {
      props.api.renderer?.requestRender?.()
    } catch {}
  }
  props.store.listeners.add(sync)
  onCleanup(() => props.store.listeners.delete(sync))
  let fg: unknown = undefined
  try {
    fg = props.api.theme?.current?.textMuted
  } catch {}
  return (
    <text
      ref={(ref: TextRenderable) => {
        text = ref
        sync()
      }}
      fg={fg as any}
    >
      {props.store.text}
    </text>
  )
}

const tui: TuiPlugin = async (api, options) => {
  const opts = (options ?? {}) as Record<string, unknown>
  const str = (v: unknown, envKey: string, fallback: string) =>
    typeof v === "string" && v ? v : process.env[envKey] || fallback
  const cfg: Config = {
    mtplxUrl: str(opts.mtplxMetricsUrl, "MTPLX_METRICS_URL", "http://127.0.0.1:8000/metrics"),
    omlxBase: str(opts.omlxBaseUrl, "OMLX_BASE_URL", "http://127.0.0.1:8099").replace(/\/+$/, ""),
    omlxKey: str(opts.omlxApiKey, "OMLX_API_KEY", ""),
    llamacppBase: str(opts.llamacppBaseUrl, "LLAMACPP_BASE_URL", "http://127.0.0.1:8080").replace(/\/+$/, ""),
    vllmBase: str(opts.vllmBaseUrl, "VLLM_BASE_URL", "http://127.0.0.1:8000").replace(/\/+$/, ""),
    sglangBase: str(opts.sglangBaseUrl, "SGLANG_BASE_URL", "http://127.0.0.1:30000").replace(/\/+$/, ""),
    vllmMlxBase: str(opts.vllmMlxBaseUrl, "VLLM_MLX_BASE_URL", "http://127.0.0.1:8000").replace(/\/+$/, ""),
    aphroditeBase: str(opts.aphroditeBaseUrl, "APHRODITE_BASE_URL", "http://127.0.0.1:2242").replace(/\/+$/, ""),
    lmdeployBase: str(opts.lmdeployBaseUrl, "LMDEPLOY_BASE_URL", "http://127.0.0.1:23333").replace(/\/+$/, ""),
  }

  const store: Store = { text: "inference · —", listeners: new Set() }
  const bump = () => {
    for (const l of store.listeners) l()
  }

  // Per-turn stream timing for the universal layer, keyed by message id.
  const turns = new Map<string, Turn>()
  const turn = (id: string): Turn => {
    let t = turns.get(id)
    if (!t) {
      t = { bytes: 0 }
      turns.set(id, t)
      if (turns.size > 64) {
        // Bound the map; drop the oldest insertion.
        const first = turns.keys().next().value
        if (first && first !== id) turns.delete(first)
      }
    }
    return t
  }

  omlxSample(cfg).then((s) => {
    if (s) omlxPrev = s
  }).catch(() => {})
  llamacppCounters(cfg).then((c) => {
    if (c) llamacppPrev = c
  }).catch(() => {})

  let lastKey = ""
  const refresh = async (info: any, provider: string, model: string) => {
    const key = `${provider}/${model}`
    if (key !== lastKey) {
      store.text = `${provider}  ${short(model)}\n…`
      bump()
      lastKey = key
    }
    const t = turns.get(info.id)
    // Prefer richer per-engine enrichment; fall back to the universal line.
    let line: string | null = null
    if (provider === "mtplx") line = await mtplxLine(cfg, model)
    else if (provider === "omlx") line = await omlxLine(cfg)
    else if (provider === "llamacpp") line = await llamacppLine(cfg, model)
    else if (provider === "vllm") line = await prometheusLine("vllm", VLLM_SPEC, cfg.vllmBase, "vLLM", model, info, t)
    else if (provider === "sglang") line = await prometheusLine("sglang", SGLANG_SPEC, cfg.sglangBase, "SGLang", model, info, t)
    else if (provider === "vllmmlx" || provider === "vllm-mlx")
      line = await prometheusLine("vllmmlx", VLLM_MLX_SPEC, cfg.vllmMlxBase, "vllm-mlx", model, info, t)
    else if (provider === "aphrodite")
      line = await prometheusLine("aphrodite", APHRODITE_SPEC, cfg.aphroditeBase, "Aphrodite", model, info, t)
    else if (provider === "lmdeploy")
      line = await prometheusLine("lmdeploy", LMDEPLOY_SPEC, cfg.lmdeployBase, "LMDeploy", model, info, t)
    if (!line) line = universalLine(provider, model, info, t)
    turns.delete(info.id)
    if (line) {
      store.text = line
      bump()
    }
  }

  const offs: Array<() => void> = []
  try {
    offs.push(
      api.event.on("message.part.delta", (evt: any) => {
        const p = evt?.properties
        if (!p || (p.field !== "text" && p.field !== "reasoning")) return
        const id = p.messageID
        if (!id) return
        const t = turn(id)
        const now = Date.now()
        if (!t.firstAt) t.firstAt = now
        t.lastAt = now
        if (typeof p.delta === "string") t.bytes += Buffer.byteLength(p.delta, "utf8")
      })
    )
    offs.push(
      api.event.on("message.updated", (evt: any) => {
        const info = evt?.properties?.info
        if (!info || info.role !== "assistant") return
        if (info.summary === true) return
        const id = String(info.id ?? "")
        if (!info.time?.completed) {
          // In-flight: capture the request start for TTFT.
          if (id && typeof info.time?.created === "number") {
            const t = turn(id)
            if (t.startAt === undefined) t.startAt = info.time.created
          }
          return
        }
        const provider = String(info.providerID ?? "")
        const model = String(info.modelID ?? "")
        setTimeout(() => {
          refresh(info, provider, model).catch(() => {})
        }, 120)
      })
    )
  } catch {}

  try {
    api.lifecycle?.onDispose?.(() => {
      for (const off of offs) {
        try {
          off()
        } catch {}
      }
    })
  } catch {}

  try {
    api.slots.register({
      slots: {
        sidebar_footer() {
          return <SidebarFooter api={api} store={store} />
        },
      },
    } as any)
  } catch {
    // registration failed; never crash the TUI over the HUD.
  }
}

const plugin: TuiPluginModule & { id: string } = {
  id: "opencode-hud",
  tui,
}

export default plugin
