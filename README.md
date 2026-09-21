# opencode-hud

Live local-inference stats pinned to the **OpenCode sidebar**. After each
assistant turn, it reads the serving engine's telemetry and shows a compact
block at the bottom of the sidebar, keyed to the model that produced it.

```
MTPLX  Qwen3.8-27B
39.6 tok/s  ttft 0.31s
prefill 234 tok/s
63 tok  3.73s
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

**Enrichment** — the provider ids in the table below additionally get their
own server-side telemetry fetched and merged in, keyed off the OpenCode
*provider id* (the key under `.provider` in `opencode.json`). Get that id
right (see below) and the richer line replaces the universal one
automatically; anything else, and you still get the universal layer.

## Engines: what each one shows

| Provider id | tok/s | TTFT | Prefill tok/s | Exact tokens | Cache info | Extras | Validated |
|---|---|---|---|---|---|---|---|
| `mtplx` | ✅ | ✅ per-turn | ✅ | ✅ | — | MTP speculative accept % | live |
| `omlx` | ✅ (recovered, per-turn when 1 req/interval) | ❌ | ✅ (recovered) | ✅ | ✅ cached tokens | — | live |
| `llamacpp` | ✅ | ❌ (universal TTFT still shows) | ✅ | ✅ | — | — | live |
| `llamafile` | ✅ | ❌ (universal TTFT still shows) | ✅ | ✅ | — | — | live |
| `mlxserve` | ✅ **engine-measured**, excludes prefill (streamed turns) | ✅ **real per-request TTFT** (streamed turns) | ❌ | ✅ (prompt only on non-streamed) | — | cold-start flag | **live** |
| `splash` | ✅ **engine-timed decode phase** | ❌ (universal TTFT still shows) | ✅ **engine-timed prefill phase**, over recomputed tokens only | ✅ | ✅ cached tokens | speculative-draft accept % | **live** |
| `koboldcpp` | ✅ **engine-timed decode phase** | ❌ (universal TTFT still shows) | ✅ **engine-timed prefill phase**, when the prompt is big enough to time | ✅ | — | speculative-draft accept % (with a draft model) | **live** |
| `vllm` | ✅ (from OpenCode's own turn timing, not vLLM's own histogram) | ✅ per-turn when one request lands, else window average | ❌ | ✅ (prompt/generation/cached) | ✅ cached tokens | — | **live** (via vllm-metal on Apple Silicon) |
| `sglang` | ✅ **engine-measured** on streaming turns, excludes prefill | ✅ per-turn when one request lands, else window average | ❌ | ✅ | ✅ cached tokens | — | **live** (via its MLX backend on Apple Silicon) |
| `vllmmlx` | ✅ **engine-measured**, excludes prefill | ✅ **per-turn, engine-measured** | ❌ | ✅ | — | — | **live** |
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
- **vLLM's tok/s is not vLLM's own number.** Splitting decode from
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
- **MTPLX shows no think/answer split, and can't from this endpoint.** A live
  capture's `/metrics` `latest` receipt was searched key by key, nested
  objects included, against a turn whose own response reported 23 of 64
  completion tokens as reasoning — no field anywhere in the 342 keys held that
  number. The per-response `usage.completion_tokens_details.reasoning_tokens`
  MTPLX returns from `/v1/chat/completions` has it; `/metrics` doesn't. This
  adapter only ever polls the latter, so `completion_tokens` (which does
  already include reasoning, confirmed) is shown as a bare total.
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
  tokens ÷ (duration − TTFT), excluding prefill. vLLM and Aphrodite publish
  no duration histogram, so they fall back to OpenCode's turn timing and their
  TTFT stays a window average (labelled `(avg)`). If several requests blend into
  one window, vllm-mlx drops the per-request rate rather than report a blended
  one.
- **vLLM is live-validated via [vllm-metal](https://github.com/vllm-project/vllm-metal)**,
  the official Apple Silicon plugin: it runs upstream vLLM's own API server with
  an MLX/Metal compute backend, so its `/metrics` *is* vLLM's. Every field name
  in our spec was confirmed against a live instance, with deltas cross-checked
  against the response's own `usage`. Its tests also carry an older,
  weaker-provenance pair (`vllm-idle.prom`/`vllm-busy.prom`, inherited from a
  prior project and not cross-checked against a response body) — see
  [`fixtures/README.md`](fixtures/README.md) for the distinction.
- **Every figure is one turn, never a running total.** Each adapter samples at
  turn boundaries and subtracts, so nothing accumulates across a session. The
  one wrinkle: an OpenCode turn that calls tools issues a request per round
  trip, and those all land inside one window — the Splash line then says
  `N requests this turn` so its sums are not misread as a single reply.
- **mlx-serve is the only engine whose freshness check is exact.** Its
  `/v1/metrics/requests` returns a keyed history of recent requests, so a turn
  is matched by `request_id` rather than inferred from a counter delta. Four
  things the live server taught us, none documented: `/metrics/requests`
  without the `/v1` prefix is a 404 even though `/metrics` resolves; on a
  **non-streamed** request `ttft_ms` comes back equal to `total_duration_ms`
  and `tokens_per_second` is a whole-request rate, so neither is offered as a
  decode figure; a streamed request reports **no prompt count** at all; and
  because the endpoint keeps a *bounded* history rather than just the latest
  request (unlike KoboldCpp), a turn that fires several requests can have its
  tokens genuinely recovered by summing every record newer than the last one
  reported, rather than only ever showing the single newest — fixed after
  finding it silently dropped an intermediate request's tokens with no
  indication anything was missed. The rate is dropped rather than
  misattributed when more than one record is summed, since mlx-serve hands
  over a pre-computed per-record rate with no raw counters to aggregate the
  way Splash's do; the panel says `N requests this turn` instead, matching
  Splash's own convention for the identical real scenario. Cold starts are
  flagged, because a turn that loads the model runs ~10x longer (4.8s against
  0.46s warm) and would otherwise read as a collapse.
- **Splash draws the fullest line of any engine here.** It publishes
  cumulative token *and* wall-time counters for prefill and decode separately,
  so both rates are differenced straight from its own measurements, plus
  prefix-cache reuse and speculative-draft acceptance. Its prefill rate is also
  the only trustworthy one on a cache hit: `splash_prefill_input_tokens_total`
  counts only tokens actually recomputed, with the reused ones in
  `splash_cache_reused_tokens_total` — the two sum to the response's
  `prompt_tokens`. Compare KoboldCpp below, which divides the *whole* prompt by
  the uncached time and so overstates.
- **KoboldCpp is the only engine here that needs no arithmetic from us.**
  `/api/extra/perf` reports the previous request already reduced, with prefill
  and decode timed as separate phases, so both rates are the engine's own
  measurements. Four things the live server taught us, none of them in the
  docs: its phase timers quantise to about 1ms, so a short prompt yields
  nonsense like "16000 tok/s" prefill (suppressed below a 10ms floor); a *full*
  prefix-cache hit reports `process_time: 0.0` rather than a huge rate; a
  *partial* cache hit silently overstates prefill, because `last_input_count`
  counts the whole prompt while `last_process_time` covers only what was
  recomputed (documented, not fixed — the endpoint exposes no cached-token
  count to correct it with); and the endpoint keeps no history beyond the
  single most recent request, so a turn that fires several requests (an
  agentic turn's tool round trips) silently reports only the last one's
  numbers. That last one *was* fixed — `total_gens` still counts every
  request even though `last_*` doesn't, so the gap is now detected and the
  panel says "N generations this turn (last shown only)" instead of quietly
  under-reporting.
- **KoboldCpp's streaming emits no usage chunk**, so for a streamed turn the
  universal layer never sees token counts at all. This endpoint is the only
  source of them, which makes the enrichment tier load-bearing here rather
  than merely additive.
- **SGLang is live-validated via its MLX backend.** SGLang ships an opt-in
  Apple Silicon path (`SGLANG_USE_MLX=1`), and `--enable-metrics` does work
  there — its Apple Metal docs page never says so. Every spec field was
  confirmed against a live instance with deltas cross-checked against the
  response's own `usage`. Two things only a live server could have shown: its
  `cached_tokens_total` counter is registered lazily and is simply absent until
  the first prefix-cache hit, and on a **non-streaming** turn it stamps TTFT at
  completion, so TTFT and end-to-end latency collapse onto each other. The
  synthetic fixtures this replaced asserted a metric name the real server never
  emitted.
- **Aphrodite and LMDeploy are fixtures-only, and their fixtures are
  synthesized** — both need CUDA. Their metric names come from each engine's
  source, but no live server has confirmed them, and the values are plausible
  rather than measured. A passing test proves the parser and the diff
  arithmetic are right; it does not prove the engine emits these names.
  See [`fixtures/README.md`](fixtures/README.md) for per-file provenance.
- **llamafile has no fixture of its own**, and its "live" mark above rests on
  one thing: a real llamafile server was pointed at this unchanged adapter and
  its `/metrics` output matched. Every fixture and test exercising the parser
  and diff arithmetic (`llamacpp-*.prom`) is a llama.cpp capture, not a
  llamafile one — reused because the two publish identical metric names, not
  because a llamafile-specific capture exists.
- **The "anything else" row's "live" mark is about the mechanism, not those
  three products by name.** The universal layer (`message.part.delta` /
  `message.updated`) is the most-exercised code path in this plugin — every
  engine above sits on top of it — so it is thoroughly live-tested. Ollama,
  MLX-LM and LM Studio specifically have not each been run against this
  plugin; they're named as examples of what falls into this row, not as
  engines individually confirmed. LM Studio's own dedicated enrichment is
  still unbuilt (see Roadmap).

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
- **mlx-serve** — provider id `mlxserve` (or `mlx-serve`), default port 8095.
  Nothing to enable. This is [raspoli/mlx-serve](https://github.com/raspoli/mlx-serve),
  an Apple Silicon manager that hot-swaps MLX models and wraps `mlx_lm.server`
  with the observability that server lacks — **not** `mlx_lm.server` itself,
  which still gets the universal line only. Set `mlxServeApiKey` if the server
  runs with `MLX_API_KEY`.
- **Splash** — provider id `splash`, default port 8000. Nothing to enable:
  `/metrics` is always on. `splash serve --model <owner/repo>`, then
  `splash opencode` wires it up. It is Apple Silicon only and ships one
  packaged model per repo.
- **KoboldCpp** — provider id `koboldcpp` (or `kobold`), default port 5001.
  Nothing to enable: `/api/extra/perf` is always on. Point OpenCode at its
  OpenAI-compatible `/v1/` endpoint. On Apple Silicon grab the
  `koboldcpp-mac-arm64` release binary (64MB) and run
  `./koboldcpp --model <model.gguf> --port 5001`.
- **SGLang** — provider id `sglang`, default port 30000. Also needs
  `--enable-metrics` on the server (off by default) or `/metrics` won't exist
  at all. On Apple Silicon, use its opt-in MLX backend: swap in the alternate
  pyproject (`mv python/pyproject_other.toml python/pyproject.toml`), then
  `SGLANG_BUILD_RUST_EXTS=none uv pip install -e "python[srt_mps]"` and launch
  with `SGLANG_USE_MLX=1 python -m sglang.launch_server --model-path <mlx-model>
  --disable-cuda-graph --enable-metrics`. The extra is `srt_mps` and the
  procedure is the one in `.github/workflows/pr-test-mlx.yml`; the published
  docs page names an `all_mps` extra that the shipped pyproject does not have.
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

Requires OpenCode ≥ 1.18.0. This is a **TUI plugin**, so it goes in
`~/.config/opencode/tui.json` (not `opencode.json`):

```jsonc
// ~/.config/opencode/tui.json
{
  "$schema": "https://opencode.ai/tui.json",
  "plugin": [
    ["@banburist/opencode-hud", { "omlxApiKey": "<your oMLX /v1 API key>" }]
  ]
}
```

Then restart OpenCode with the sidebar open. (Once published, `opencode plugin
@banburist/opencode-hud` can add it for you.)

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
| `mlxServeBaseUrl` | `MLXSERVE_BASE_URL` | `http://127.0.0.1:8095` |
| `mlxServeApiKey` | `MLX_API_KEY` | *(unset)* |
| `splashBaseUrl` | `SPLASH_BASE_URL` | `http://127.0.0.1:8000` |
| `koboldcppBaseUrl` | `KOBOLDCPP_BASE_URL` | `http://127.0.0.1:5001` |
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

`@opencode-ai/plugin` and `@opencode-ai/sdk` are pinned in `devDependencies`
to exactly the version floor declared in `engines.opencode`, not to a `>=`
range. A range resolves to the newest published version, so the typecheck
would have silently validated against something newer than the floor being
claimed — which is how the floor came to be wrong: it was set to whatever
happened to be installed on the first commit, and this plugin was running
against OpenCode 1.18.31 (whose bundled `@opencode-ai/plugin` is 1.18.18)
the whole time, below the 1.18.20 the package declared. Pinning means
`npm run typecheck` fails if the floor is ever raised past what the code
needs, or lowered past what it supports.

The floor is 1.18.0 because `dist/tui.d.ts` is byte-identical across every
1.18.x release, so every API used here — `lifecycle.signal`,
`lifecycle.onDispose`, `slots.register`, the `sidebar_footer` slot and
`event.on` — is unchanged across the line. The same surface exists back to
1.16.2, but nothing here has been run against it, so it is not claimed.

## Roadmap

- LM Studio enrichment — low value; it only reports `stats.tokens_per_second`
  per response, which the universal layer already approximates as well.
- **Checked and ruled out** (universal layer only, no server-wide telemetry
  exists): **ExLlamaV3 / TabbyAPI** — despite third-party claims of a
  Prometheus endpoint, there is none in its source; **lightning-mlx** — no
  telemetry endpoint at all.
- **Worth a look, not yet built**: **Modular MAX serve** (rich `maxserve_*`
  metrics including TTFT and inter-token latency, but Apple Silicon support
  unconfirmed).
- ~~**llamafile**~~ — done: it works with the `llamacpp` adapter unchanged,
  confirmed live.
- ~~**KoboldCpp**~~ — done and live-validated. The Mac arm64 binary is 64MB,
  not the ~700MB guessed here earlier; `/api/extra/perf` needs no flag.
- ~~**vllm-metal**~~ — done: confirmed the existing `vllm` adapter works
  against it unchanged, which moved the vLLM tier to live-validated.
- ~~A live vLLM/SGLang server~~ — done: both are now live-validated
  (vLLM via vllm-metal, SGLang via its MLX backend).
- An optional keybind to toggle the panel independently of the sidebar.
- Publish to npm (`@banburist/opencode-hud`) and list in the [OpenCode
  ecosystem](https://opencode.ai/docs/ecosystem#plugins).

## License

MIT
