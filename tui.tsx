/** @jsxImportSource @opentui/solid */
// opencode-hud — persistent per-turn local-inference stats in the sidebar.
//
// Renders into the `sidebar_footer` slot. After each assistant turn it reads the
// serving engine's telemetry and shows a compact block, keyed to the model that
// produced it — on a model or provider change the panel is replaced, never
// blended, so a switch zeroes cleanly.
//
// Engines:
//   - MTPLX  (provider id `mtplx`) — /metrics `latest` receipt, per-request
//            precise: decode tok/s, TTFT, prefill, MTP speculative acceptance.
//   - oMLX   (provider id `omlx`)  — /api/status, differenced across the turn.
//            Poll, atomic-at-completion: exact tokens + per-request rates
//            recovered from the running average, no live ticker or TTFT.
//   - others — a name and a dash (no adapter yet).
//
// Config (plugin options in tui.json, or env fallback):
//   mtplxMetricsUrl  (MTPLX_METRICS_URL)  default http://127.0.0.1:8000/metrics
//   omlxBaseUrl      (OMLX_BASE_URL)       default http://127.0.0.1:8099
//   omlxApiKey       (OMLX_API_KEY)        required to read oMLX; no default
//
//   "plugin": [["@charlesnutter/opencode-hud", { "omlxApiKey": "…" }]]
import type { TextRenderable } from "@opentui/core"
import type { TuiPlugin, TuiPluginModule } from "@opencode-ai/plugin/tui"
import { onCleanup } from "solid-js"

interface Config {
  mtplxUrl: string
  omlxBase: string
  omlxKey: string
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

// ---- MTPLX: /metrics `latest` receipt, per-request precise -----------------
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

// ---- oMLX: /api/status, differenced across the turn -------------------------
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
  if (!cfg.omlxKey) return null // no key configured; oMLX cannot be read
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
  if (!s) return cfg.omlxKey ? null : "oMLX\nset omlxApiKey to read stats"
  const prev = omlxPrev
  omlxPrev = s
  // No baseline (first turn, or model just changed): show what this reading has
  // rather than a delta we cannot compute yet.
  if (!prev || prev.model !== s.model || s.requests <= prev.requests) {
    return [`oMLX  ${short(s.model ?? "")}`, `${nn(s.avgGen)} tok/s (server avg)`, `prefill ${ni(s.avgPrefill)} tok/s (avg)`]
      .join("\n")
  }
  const dReq = s.requests - prev.requests
  const comp = s.completion - prev.completion
  // avg_generation_tps is the arithmetic mean of per-request rates, so the last
  // request's rate is avg_new*n_new - avg_prev*n_prev — exact when one request
  // landed between polls. Fall back to the current average otherwise.
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
  }

  const store: Store = { text: "inference · —", listeners: new Set() }
  const bump = () => {
    for (const l of store.listeners) l()
  }

  // Seed an oMLX baseline so the first omlx turn can be differenced. Best
  // effort: if oMLX is not up or unkeyed, the first turn seeds it instead.
  omlxSample(cfg).then((s) => {
    if (s) omlxPrev = s
  }).catch(() => {})

  let lastKey = ""
  const refresh = async (provider: string, model: string) => {
    const key = `${provider}/${model}`
    if (key !== lastKey) {
      // Model or provider changed: zero the panel immediately rather than leave
      // the previous model's numbers up while the new reading is fetched.
      store.text = `${provider}  ${short(model)}\n…`
      bump()
      lastKey = key
    }
    let line: string | null = null
    if (provider === "mtplx") line = await mtplxLine(cfg, model)
    else if (provider === "omlx") line = await omlxLine(cfg)
    else line = `${provider}  ${short(model)}\n(no HUD adapter)`
    if (line) {
      store.text = line
      bump()
    }
  }

  let off: (() => void) | undefined
  try {
    off = api.event.on("message.updated", (evt: any) => {
      const info = evt?.properties?.info
      if (!info || info.role !== "assistant") return
      if (info.summary === true) return
      if (!info.time?.completed) return
      const provider = String(info.providerID ?? "")
      const model = String(info.modelID ?? "")
      setTimeout(() => {
        refresh(provider, model).catch(() => {})
      }, 120)
    })
  } catch {}

  try {
    api.lifecycle?.onDispose?.(() => {
      try {
        off?.()
      } catch {}
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
