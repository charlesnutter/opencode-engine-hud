# opencode-hud

Live local-inference stats pinned to the **OpenCode sidebar**. After each
assistant turn, it reads the serving engine's telemetry and shows a compact
block at the bottom of the sidebar, keyed to the model that produced it.

```
MTPLX  Qwen3.8-27B
39.6 tok/s  ttft 0.31s
prefill 234 tok/s
63 tok (+23 think)  3.73s
MTP 2.85x 81/60/43%
```

On a model or provider switch the panel is replaced, never blended — a change
zeroes cleanly rather than carrying a stale reading.

## How it works: two tiers

**Universal layer** — every provider gets this for free, no engine endpoint
needed. Built from OpenCode's own per-turn events (`message.part.delta` for
streaming/TTFT, `message.updated` for exact final token counts and wall time),
so it works for any OpenAI-compatible server: Ollama, MLX-LM, LM Studio,
anything.

**Enrichment** — five engines additionally get their own server-side telemetry
fetched and merged in, keyed off the OpenCode *provider id* (the key under
`.provider` in `opencode.json`). Get that id right (see below) and the richer
line replaces the universal one automatically; anything else, and you still
get the universal layer.

## Engines: what each one shows

| Provider id | tok/s | TTFT | Prefill tok/s | Exact tokens | Cache info | Extras | Validated |
|---|---|---|---|---|---|---|---|
| `mtplx` | ✅ | ✅ per-turn | ✅ | ✅ | — | MTP speculative accept %, reasoning tokens | live |
| `omlx` | ✅ (recovered, per-turn when 1 req/interval) | ❌ | ✅ (recovered) | ✅ | ✅ cached tokens | — | live |
| `llamacpp` | ✅ | ❌ (universal TTFT still shows) | ✅ | ✅ | — | — | live |
| `llamafile` | ✅ | ❌ (universal TTFT still shows) | ✅ | ✅ | — | — | live |
| `vllm` | ✅ (from OpenCode's own turn timing, not vLLM's own histogram) | ✅ per-turn when one request lands, else window average | ❌ | ✅ (prompt/generation/cached) | ✅ cached tokens | — | **live** (via vllm-metal on Apple Silicon) |
| `sglang` | same as vLLM | same as vLLM | ❌ | ✅ | ✅ | — | **synthetic fixtures** (CUDA-only engine) |
| `vllmmlx` | ✅ **engine-measured**, excludes prefill | ✅ **per-turn, engine-measured** | ❌ | ✅ | — | — | live |
| `aphrodite` | same as vLLM | same as vLLM | ❌ | ✅ | ✅ | — | **derived fixture** (CUDA-only engine) |
| `lmdeploy` | ✅ **engine-timed decode phase** | ✅ per-turn | ✅ **engine-timed prefill phase** | ✅ | — | — | **synthetic fixtures** (CUDA-only engine) |
| anything else (Ollama, MLX-LM, LM Studio, …) | ✅ | ✅ per-turn | ❌ | ✅ | ❌ | — | live |

Notes on what's *missing* and why, since that matters as much as what's shown:

- **oMLX has no TTFT at all.** Its counters are atomic at completion — frozen
  while a request runs, updated only once it lands — so there's nothing to
  time a first token against without a live endpoint it doesn't expose.
- **llama.cpp and oMLX have no prefill/decode split from their own data**
  beyond what's differenced from cumulative counters; the panel's "prefill
  tok/s" for these two *is* that differenced figure, not a separate timing.
- **vLLM/SGLang's tok/s is not vLLM's own number.** Splitting decode from
  prefill needs continuous polling mid-request, which this plugin doesn't do;
  the rate shown instead reuses OpenCode's own streaming-delta timing (the
  same source the universal layer uses for every other provider). Their TTFT
  *is* engine-reported, but it's a Prometheus histogram average over
  however many requests landed in the interval since the last turn — not this
  turn's own value — labelled `(avg)` in the panel to say so.
- **oMLX, vLLM and SGLang report cache-hit tokens; MTPLX and llama.cpp
  don't.** Not because the data isn't there for MTPLX — its `/metrics` reports
  `cached_tokens` too, this plugin just doesn't read that field yet. llama.cpp
  has no cache counter at all; its prompt-token count is silently *lower* on a
  cache hit, since the underlying counter only tracks what was actually
  computed.
- **LMDeploy is the richest surface of any engine here.** It times prefill and
  decode as separate histograms, so both rates are its own measurement rather
  than derived — nothing else can produce a real prefill rate from Prometheus.
  Its fixtures are **synthesized**, not captured: it needs CUDA. The metric
  names and label shape are verified against its source
  (`lmdeploy/metrics/loggers.py`), but no live server confirmed them.
- **vllm-mlx is the only *live-validated* engine with a true decode rate.** It
  publishes both a TTFT *and* an end-to-end duration histogram, so when exactly
  one request lands in the window — which is the norm, since OpenCode issues one
  per turn — the deltas are that turn's own values, and decode rate comes out as
  tokens ÷ (duration − TTFT), excluding prefill. vLLM/SGLang/Aphrodite publish
  no duration histogram, so they fall back to OpenCode's turn timing and their
  TTFT stays a window average (labelled `(avg)`). If several requests blend into
  one window, vllm-mlx drops the per-request rate rather than report a blended
  one.
- **vLLM is live-validated via [vllm-metal](https://github.com/vllm-project/vllm-metal)**,
  the official Apple Silicon plugin: it runs upstream vLLM's own API server with
  an MLX/Metal compute backend, so its `/metrics` *is* vLLM's. Every field name
  in our spec was confirmed against a live instance, with deltas cross-checked
  against the response's own `usage`.
- **SGLang, Aphrodite and LMDeploy are fixtures-only, and their fixtures are
  synthesized** — all three need CUDA. Their metric names come from each
  engine's source, but no live server has confirmed them, and the values are
  plausible rather than measured. A passing test proves the parser and the
  diff arithmetic are right; it does not prove the engine emits these names.
  See [`fixtures/README.md`](fixtures/README.md) for per-file provenance.

## Adding an engine to `opencode.json`

The HUD only ever reads what OpenCode already knows about, so an engine has to
exist as a **provider** in `opencode.json` before it can show up here at all —
that's a separate file from this plugin's own `tui.json` config (below). The
shape is the same for any OpenAI-compatible server:

```jsonc
// ~/.config/opencode/opencode.json
{
  "provider": {
    "<provider-id>": {
      "name": "Display name",
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:<port>/v1", "apiKey": "anything" },
      "models": {
        "<model id the server reports at /v1/models>": {
          "name": "Display name for the model",
          "limit": { "context": 32768, "output": 8192 },
          "modalities": { "input": ["text"], "output": ["text"] },
          "tool_call": true
        }
      }
    }
  }
}
```

**The provider id is what turns on enrichment** — use exactly `mtplx`, `omlx`,
`llamacpp`, `vllm` or `sglang` to get that engine's richer line; any other key
(e.g. `mlxlm`, `mystery-server`) still works fully, just with the universal
layer only.

Per-engine notes:

- **Ollama** — `baseURL: "http://127.0.0.1:11434/v1"`. Any provider id (no
  enrichment adapter for Ollama); its own `/api/*` telemetry is per-caller, not
  server-wide, so there's nothing for this plugin to fetch beyond what
  OpenCode's events already give it.
- **llama.cpp** — provider id `llamacpp`, `baseURL:
  "http://127.0.0.1:8080/v1"`. Start the **classic single-model binary**,
  not the newer multi-model router:
  ```bash
  llama-server --hf-repo <user>/<repo> --hf-file <file>.gguf \
    --host 127.0.0.1 --port 8080 --metrics
  ```
  `--metrics` is required — it's off by default, and without it this provider
  falls back to the universal layer silently. The router (`llama serve`)
  exposes a different `/props` shape (`model_path: "none"`, a `role: "router"`
  field) that this adapter doesn't read; use `llama-server` directly.
- **llamafile** — provider id `llamafile`, default port 8003. It is
  llama.cpp-derived and publishes the *identical* `llamacpp:` metric names in
  the same bare (unlabelled) format, so it reuses that adapter verbatim — only
  the URL and its own counter baseline differ, which lets llama.cpp and
  llamafile run side by side. The bare binary (`llamafile-<ver>-thin`, ~41MB)
  loads an external GGUF, so no bundled-weights download is needed:
  ```bash
  llamafile -m model.gguf --server --host 127.0.0.1 --port 8003 --metrics
  ```
- **MLX-LM** (`mlx_lm.server`) — any provider id; it has no server-wide
  `/metrics` of its own, so universal layer only.
- **vLLM** — provider id `vllm`, default port 8000. On a CUDA host, point
  `baseURL` at wherever it runs. **On Apple Silicon**, install
  [vllm-metal](https://github.com/vllm-project/vllm-metal)
  (`brew tap vllm-project/vllm-metal …`) and `vllm serve <model>` works
  normally — it's upstream vLLM, so this adapter needs no changes.
- **SGLang** — provider id `sglang`, default port 30000. Also needs
  `--enable-metrics` on the server (off by default) or `/metrics` won't exist
  at all.
- **vllm-mlx** — provider id `vllmmlx` (or `vllm-mlx`), default port 8000.
  The MLX-native Apple Silicon server, installable with `pip install vllm-mlx`.
  Start it with the metrics flag, which is **`--enable-metrics`**, not
  `--metrics` as some docs say:
  ```bash
  vllm-mlx serve mlx-community/Qwen2.5-0.5B-Instruct-4bit --port 8000 --enable-metrics
  ```
  Note it defaults to the same port as vLLM, so the two can't both run as-is.
- **LMDeploy** — provider id `lmdeploy`, default port **23333**. Needs
  `--enable-metrics` (off by default). CUDA host.
- **Aphrodite** — provider id `aphrodite`, default port **2242** (a holdover
  from its KoboldAI origins). A vLLM fork, so its metrics are vLLM's under an
  `aphrodite:` prefix. Needs a CUDA host.
- **MTPLX / oMLX** — provider ids `mtplx` / `omlx`. See their own docs for
  serving; oMLX additionally needs `omlxApiKey` set in this plugin's own
  config (next section) to read its telemetry.

## Install (this plugin)

Requires OpenCode ≥ 1.18.20. This is a **TUI plugin**, so it goes in
`~/.config/opencode/tui.json` (not `opencode.json`):

```jsonc
// ~/.config/opencode/tui.json
{
  "$schema": "https://opencode.ai/tui.json",
  "plugin": [
    ["@charlesnutter/opencode-hud", { "omlxApiKey": "<your oMLX /v1 API key>" }]
  ]
}
```

Then restart OpenCode with the sidebar open. (Once published, `opencode plugin
@charlesnutter/opencode-hud` can add it for you.)

## Configuration

Options are passed in the `tui.json` plugin entry; each also has an env
fallback. All are optional — an engine that isn't running or isn't configured
just falls back to the universal layer.

| Option | Env | Default |
|---|---|---|
| `mtplxMetricsUrl` | `MTPLX_METRICS_URL` | `http://127.0.0.1:8000/metrics` |
| `omlxBaseUrl` | `OMLX_BASE_URL` | `http://127.0.0.1:8099` |
| `omlxApiKey` | `OMLX_API_KEY` | *(none — required to read oMLX; without it the panel says so)* |
| `llamacppBaseUrl` | `LLAMACPP_BASE_URL` | `http://127.0.0.1:8080` |
| `vllmBaseUrl` | `VLLM_BASE_URL` | `http://127.0.0.1:8000` |
| `sglangBaseUrl` | `SGLANG_BASE_URL` | `http://127.0.0.1:30000` |
| `vllmMlxBaseUrl` | `VLLM_MLX_BASE_URL` | `http://127.0.0.1:8000` |
| `aphroditeBaseUrl` | `APHRODITE_BASE_URL` | `http://127.0.0.1:2242` |
| `lmdeployBaseUrl` | `LMDEPLOY_BASE_URL` | `http://127.0.0.1:23333` |
| `llamafileBaseUrl` | `LLAMAFILE_BASE_URL` | `http://127.0.0.1:8003` |

## Local development

The runtime (SolidJS / opentui) is provided by OpenCode, so no build step or
`npm install` is needed to run it — point `tui.json` at your working copy:

```jsonc
{ "plugin": [["/Users/you/dev/opencode-hud", { "omlxApiKey": "…" }]] }
```

Restart OpenCode to reload. For type-checking and the vLLM/SGLang fixture
tests (these don't need the OpenCode runtime, so `npm install` is needed only
for these):

```bash
npm install
npm run typecheck
npm test
```

## Roadmap

- LM Studio enrichment — low value; it only reports `stats.tokens_per_second`
  per response, which the universal layer already approximates as well.
- **Checked and ruled out** (universal layer only, no server-wide telemetry
  exists): **ExLlamaV3 / TabbyAPI** — despite third-party claims of a
  Prometheus endpoint, there is none in its source; **lightning-mlx** — no
  telemetry endpoint at all.
- **Worth a look, not yet built**: **Modular MAX serve** (rich `maxserve_*`
  metrics including TTFT and inter-token latency, but Apple Silicon support
  unconfirmed); **KoboldCpp** (`/api/extra/perf` carries a complete
  last-request set — `last_input_count`, `last_token_count`,
  `last_process_time`, `last_eval_time` — and runs on macOS arm64, so it could
  be live-validated, but needs a ~700MB download); (llamafile is done —
  it did work with the `llamacpp` adapter unchanged.)
- ~~**vllm-metal**~~ — done: confirmed the existing `vllm` adapter works
  against it unchanged, which moved the vLLM tier to live-validated.
- A live vLLM/SGLang server to validate the enrichment tier against real
  traffic, not just captured fixtures.
- An optional keybind to toggle the panel independently of the sidebar.
- Publish to npm (`@charlesnutter/opencode-hud`) and list in the [OpenCode
  ecosystem](https://opencode.ai/docs/ecosystem#plugins).

## License

MIT
