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
import type { RGBA, TextRenderable } from "@opentui/core"
import type { TuiPlugin, TuiPluginModule } from "@opencode-ai/plugin/tui"
import { onCleanup } from "solid-js"
import { appendFileSync } from "node:fs"
import { fetchKoboldPerf, koboldTurn } from "./koboldcpp"
import { fetchSplashSample, diffSplashSamples } from "./splash"
import type { SplashSample } from "./splash"
import { fetchMlxServeRequests, mlxServeTurn } from "./mlxserve"
import { turnRate, universalLine, tokensLabel, short, nn, ni } from "./universal"
import type { Turn } from "./universal"
import type { AssistantMessage } from "@opencode-ai/sdk/v2"
import { VLLM_SPEC, SGLANG_SPEC, APHRODITE_SPEC, VLLM_MLX_SPEC, LMDEPLOY_SPEC, fetchPromSample, diffPromSamples } from "./prometheus"
import type { PromSpec, PromSample } from "./prometheus"

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
  llamafileBase: string
  koboldBase: string
  splashBase: string
  mlxServeBase: string
  mlxServeKey: string
}

/**
 * Diagnostics, off unless OPENCODE_HUD_DEBUG is set. Adapter failures are
 * caught so a broken engine never blanks the panel, which means they are
 * otherwise invisible; this is how you see them.
 */
const HUD_DEBUG = !!process.env.OPENCODE_HUD_DEBUG
function dbg(msg: string) {
  if (!HUD_DEBUG) return
  try {
    appendFileSync("/tmp/opencode-hud-debug.log", `${new Date().toISOString()} ${msg}\n`)
  } catch {}
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
  // MTPLX follows the OpenAI convention: completion_tokens already INCLUDES
  // reasoning_tokens, so it is the topline as-is (see tokensLabel).
  const reasoning = typeof l.reasoning_tokens === "number" ? l.reasoning_tokens : 0
  return [
    `MTPLX  ${short(model)}`,
    `${nn(l.decode_tok_s)} tok/s  ttft ${nn(l.ttft_s, 2)}s`,
    `prefill ${ni(l.prefill_tok_s)} tok/s`,
    `${tokensLabel(l.completion_tokens, reasoning)}  ${nn(l.request_elapsed_s, 2)}s`,
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
const llamacppPrev = new Map<string, LlamaCppCounters>()

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

async function llamacppCounters(base: string): Promise<LlamaCppCounters | null> {
  const text = await getText(`${base}/metrics`)
  if (text === null) return null
  const v = parsePrometheus(text)
  return {
    promptTokens: v["llamacpp:prompt_tokens_total"] ?? 0,
    promptSeconds: v["llamacpp:prompt_seconds_total"] ?? 0,
    predictedTokens: v["llamacpp:tokens_predicted_total"] ?? 0,
    predictedSeconds: v["llamacpp:tokens_predicted_seconds_total"] ?? 0,
  }
}

async function llamacppLine(
  key: string,
  base: string,
  label: string,
  model: string
): Promise<string | null> {
  const now = await llamacppCounters(base)
  if (!now) return null // unreachable, or started without --metrics
  const prev = llamacppPrev.get(key)
  llamacppPrev.set(key, now)
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
    `${label}  ${short(model)}`,
    decodeTokS !== undefined ? `${nn(decodeTokS)} tok/s` : "",
    prefillTokS !== undefined ? `prefill ${ni(prefillTokS)} tok/s` : "",
    `${ni(completionTokens)} tok  ${nn(decodeS + prefillS, 2)}s`,
  ].filter(Boolean).join("\n")
}

// ---- Tier 2: mlx-serve enrichment — /v1/metrics/requests, id-keyed --------
// Alone among these engines, mlx-serve keeps a history of recent requests each
// carrying its own id, so "is this our turn?" is answered by identity rather
// than by a counter delta. Note the /v1 prefix: /metrics/requests without it
// is a 404 even though /metrics itself resolves. Validated live against
// mlx-serve 0.1.0 on Apple Silicon.
const mlxServePrevId = new Map<string, string>()

async function mlxServeLine(cfg: Config, model: string): Promise<string | null> {
  const recs = await fetchMlxServeRequests(cfg.mlxServeBase, model, cfg.mlxServeKey || undefined)
  if (!recs) return null // unreachable, not mlx-serve, or the API key is wrong
  const t = mlxServeTurn(recs, mlxServePrevId.get(cfg.mlxServeBase))
  if (!t) return null // nothing newer than the record already reported
  mlxServePrevId.set(cfg.mlxServeBase, t.requestId)

  // decodeTokS and overallTokS are never both set; they are not comparable, so
  // the whole-request one is labelled rather than shown as a decode rate.
  const rate =
    t.decodeTokS !== undefined
      ? `${nn(t.decodeTokS)} tok/s${t.ttft !== undefined ? `  ttft ${nn(t.ttft, 2)}s` : ""}`
      : t.overallTokS !== undefined
        ? `${nn(t.overallTokS)} tok/s (whole request)`
        : ""
  return [
    `mlx-serve  ${short(model)}`,
    rate,
    `${ni(t.completionTokens)} tok${t.promptTokens !== undefined ? `  ${ni(t.promptTokens)} prompt` : ""}  ${nn(t.totalS, 2)}s`,
    // A cold start loaded the model mid-request; without this the turn reads
    // as a tenfold slowdown rather than a one-off load.
    t.coldStart ? "cold start (model loaded)" : "",
  ].filter(Boolean).join("\n")
}

// ---- Tier 2: Splash enrichment — /metrics, both phases engine-timed -------
// The fullest line this plugin draws: Splash counts tokens AND wall time for
// prefill and decode separately, so both rates are differenced straight from
// its own measurements, and it counts prefix-cache reuse and speculative
// drafting besides. Validated live against Splash 1.0 on Apple Silicon.
const splashPrev = new Map<string, SplashSample>()

async function splashLine(base: string, model: string): Promise<string | null> {
  const now = await fetchSplashSample(base)
  if (!now) return null // unreachable, or not a Splash server
  const prev = splashPrev.get(base)
  splashPrev.set(base, now)
  if (!prev) return null // no baseline yet (first turn since launch)
  const t = diffSplashSamples(prev, now)
  if (!t) return null

  // promptTokens is what was recomputed; cached is what the prefix cache
  // served. Showing both is the honest reading of a Splash prefill rate.
  const prompt = t.promptTokens + t.cachedTokens
  return [
    `Splash  ${short(model)}`,
    t.decodeTokS !== undefined ? `${nn(t.decodeTokS)} tok/s` : "",
    t.prefillTokS !== undefined ? `prefill ${ni(t.prefillTokS)} tok/s` : "",
    `${ni(t.completionTokens)} tok  ${nn(t.prefillS + t.decodeS, 2)}s`,
    `${ni(prompt)} prompt${t.cachedTokens > 0 ? `, ${ni(t.cachedTokens)} cached` : ""}`,
    t.draftAcceptRate !== undefined ? `draft ${ni(t.draftAcceptRate * 100)}% accepted` : "",
    // Only when a turn spanned several requests (tool round trips), so the
    // figures above read as sums rather than as one reply.
    t.requests > 1 ? `${ni(t.requests)} requests this turn` : "",
  ].filter(Boolean).join("\n")
}

// ---- Tier 2: KoboldCpp enrichment — /api/extra/perf, last-request ---------
// Unlike every Prometheus engine here, this one hands over the previous
// request already reduced, with prefill and decode timed separately, so there
// is no differencing to do. The only state kept is the generation counter,
// which is what distinguishes "this turn's numbers" from a stale sample (see
// koboldcpp.ts). Validated live against KoboldCpp v1.121 on Apple Silicon.
const koboldPrevGens = new Map<string, number>()

async function koboldLine(base: string, model: string): Promise<string | null> {
  const perf = await fetchKoboldPerf(base)
  if (!perf) return null // unreachable, or not a KoboldCpp server
  const prev = koboldPrevGens.get(base)
  koboldPrevGens.set(base, perf.total_gens)
  const t = koboldTurn(perf, prev)
  if (!t) return null // nothing new to attribute to this turn
  return [
    `KoboldCpp  ${short(model)}`,
    t.decodeTokS !== undefined ? `${nn(t.decodeTokS)} tok/s` : "",
    t.prefillTokS !== undefined ? `prefill ${ni(t.prefillTokS)} tok/s` : "",
    `${ni(t.completionTokens)} tok  ${nn(t.prefillS + t.decodeS, 2)}s`,
    t.draftAcceptRate !== undefined ? `draft ${ni(t.draftAcceptRate * 100)}% accepted` : "",
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
// SGLang is validated live on this machine via its MLX/Apple-Silicon backend
// (SGLANG_USE_MLX=1 --enable-metrics); vLLM likewise via vllm-metal. Aphrodite
// and LMDeploy need CUDA, so those are validated against real captured
// /metrics text in test/prometheus.test.mjs instead.
const promPrev = new Map<string, PromSample>() // keyed by provider id, not URL

async function prometheusLine(
  providerId: string,
  spec: PromSpec,
  base: string,
  label: string,
  model: string,
  info: AssistantMessage | undefined,
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
  // OpenCode's turn timing. The engine-derived figure is dropped when the
  // decode window is implausibly short (see MIN_DECODE_SHARE), so a
  // non-streaming caller falls back here rather than showing clock noise.
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
  // theme.current.textMuted is typed RGBA and non-optional; the guard stays
  // because this API is undocumented and has shifted before.
  let fg: RGBA | undefined
  try {
    fg = props.api.theme?.current?.textMuted
  } catch {}
  return (
    <text
      ref={(ref: TextRenderable) => {
        text = ref
        sync()
      }}
      fg={fg}
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
    llamafileBase: str(opts.llamafileBaseUrl, "LLAMAFILE_BASE_URL", "http://127.0.0.1:8003").replace(/\/+$/, ""),
    koboldBase: str(opts.koboldcppBaseUrl, "KOBOLDCPP_BASE_URL", "http://127.0.0.1:5001").replace(/\/+$/, ""),
    splashBase: str(opts.splashBaseUrl, "SPLASH_BASE_URL", "http://127.0.0.1:8000").replace(/\/+$/, ""),
    mlxServeBase: str(opts.mlxServeBaseUrl, "MLXSERVE_BASE_URL", "http://127.0.0.1:8095").replace(/\/+$/, ""),
    mlxServeKey: str(opts.mlxServeApiKey, "MLX_API_KEY", ""),
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
  llamacppCounters(cfg.llamacppBase).then((c) => {
    if (c) llamacppPrev.set("llamacpp", c)
  }).catch(() => {})
  llamacppCounters(cfg.llamafileBase).then((c) => {
    if (c) llamacppPrev.set("llamafile", c)
  }).catch(() => {})

  let lastKey = ""
  const refresh = async (info: AssistantMessage, provider: string, model: string) => {
    const key = `${provider}/${model}`
    if (key !== lastKey) {
      store.text = `${provider}  ${short(model)}\n…`
      bump()
      lastKey = key
    }
    const t = turns.get(info.id)
    // Prefer richer per-engine enrichment; fall back to the universal line.
    let line: string | null = null
    try {
    if (provider === "mtplx") line = await mtplxLine(cfg, model)
    else if (provider === "omlx") line = await omlxLine(cfg)
    else if (provider === "llamacpp") line = await llamacppLine("llamacpp", cfg.llamacppBase, "llama.cpp", model)
    // llamafile is llama.cpp-derived and publishes the identical metric names,
    // so it reuses this adapter verbatim — only the URL and baseline differ.
    else if (provider === "llamafile") line = await llamacppLine("llamafile", cfg.llamafileBase, "llamafile", model)
    else if (provider === "splash") line = await splashLine(cfg.splashBase, model)
    else if (provider === "mlxserve" || provider === "mlx-serve")
      line = await mlxServeLine(cfg, model)
    else if (provider === "koboldcpp" || provider === "kobold")
      line = await koboldLine(cfg.koboldBase, model)
    else if (provider === "vllm") line = await prometheusLine("vllm", VLLM_SPEC, cfg.vllmBase, "vLLM", model, info, t)
    else if (provider === "sglang") line = await prometheusLine("sglang", SGLANG_SPEC, cfg.sglangBase, "SGLang", model, info, t)
    else if (provider === "vllmmlx" || provider === "vllm-mlx")
      line = await prometheusLine("vllmmlx", VLLM_MLX_SPEC, cfg.vllmMlxBase, "vllm-mlx", model, info, t)
    else if (provider === "aphrodite")
      line = await prometheusLine("aphrodite", APHRODITE_SPEC, cfg.aphroditeBase, "Aphrodite", model, info, t)
    else if (provider === "lmdeploy")
      line = await prometheusLine("lmdeploy", LMDEPLOY_SPEC, cfg.lmdeployBase, "LMDeploy", model, info, t)
    } catch (e: unknown) {
      // An adapter failing must never blank the panel: fall through to the
      // universal line. Silent for users, visible with OPENCODE_HUD_DEBUG —
      // a swallowed ReferenceError here once broke the MTPLX and oMLX
      // adapters for several commits without any visible symptom.
      // `unknown` forces this: a throw is not guaranteed to be an Error.
      const err = e instanceof Error ? e : new Error(String(e))
      dbg(`${provider} adapter threw: ${err.name}: ${err.message}\n${err.stack ?? ""}`)
    }
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
      api.event.on("message.part.delta", (evt) => {
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
      api.event.on("message.updated", (evt) => {
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
    })
  } catch {
    // registration failed; never crash the TUI over the HUD.
  }
}

const plugin: TuiPluginModule & { id: string } = {
  id: "opencode-hud",
  tui,
}

export default plugin
