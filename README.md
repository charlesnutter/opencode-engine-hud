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

## Engines

Every provider gets the **universal layer** for free — tok/s, TTFT, and exact
tokens, read from OpenCode's own per-turn events (`message.part.delta` +
`message.updated`). No engine endpoint needed, so Ollama, MLX-LM and any other
OpenAI-compatible server all work out of the box. Some engines get richer
**enrichment** on top, from their own server-side telemetry:

| Provider | Source | What you get | Validated |
|---|---|---|---|
| **MTPLX** (`mtplx`) | `/metrics` `latest` receipt | Per-request precise: decode tok/s, TTFT, prefill, MTP speculative acceptance | live |
| **oMLX** (`omlx`) | `/api/status`, differenced across the turn | Exact tokens + per-request rates. Poll, atomic-at-completion: no live ticker, no TTFT | live |
| **llama.cpp** (`llamacpp`) | `/metrics`, differenced across the turn | Exact tokens, decode tok/s, prefill tok/s. Needs `--metrics` (off by default) | live |
| **vLLM** (`vllm`) | Prometheus `/metrics`, differenced across the turn | Exact tokens (prompt/generation/cached), TTFT histogram average. Decode rate from OpenCode's own turn timing | fixtures only (CUDA-only engine) |
| **SGLang** (`sglang`) | Prometheus `/metrics`, differenced across the turn | Same as vLLM | fixtures only (CUDA-only engine) |
| others (Ollama, MLX-LM, LM Studio, …) | — | Universal layer only | live |

vLLM and SGLang can't run on Apple Silicon, so that tier is validated against
real captured `/metrics` text (`fixtures/`, `test/prometheus.test.mjs`) rather
than a live server. Run it with `npm test`.

## Install

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
fallback.

| Option | Env | Default | Notes |
|---|---|---|---|
| `mtplxMetricsUrl` | `MTPLX_METRICS_URL` | `http://127.0.0.1:8000/metrics` | MTPLX metrics endpoint |
| `omlxBaseUrl` | `OMLX_BASE_URL` | `http://127.0.0.1:8099` | oMLX server base URL |
| `omlxApiKey` | `OMLX_API_KEY` | *(none)* | oMLX's `/v1` API key. Required to read oMLX; without it the panel says so. |

## Local development

The runtime (SolidJS / opentui) is provided by OpenCode, so no build step or
`npm install` is needed to run it — point `tui.json` at your working copy:

```jsonc
{ "plugin": [["/Users/you/dev/opencode-hud", { "omlxApiKey": "…" }]] }
```

Restart OpenCode to reload. For type-checking:

```bash
npm install      # pulls the type-only deps
npm run typecheck
```

## Roadmap

- More engines (Ollama, llama.cpp, vLLM, LM Studio) — the same normalized
  approach as the [inference-hud](https://github.com/charlesnutter/inference-hud)
  VS Code extension, which this shares telemetry techniques with.
- An optional keybind to toggle the panel independently of the sidebar.

## License

MIT
