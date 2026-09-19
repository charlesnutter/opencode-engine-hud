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

| Provider | Source | What you get |
|---|---|---|
| **MTPLX** (`mtplx`) | `/metrics` `latest` receipt | Per-request precise: decode tok/s, TTFT, prefill, MTP speculative acceptance |
| **oMLX** (`omlx`) | `/api/status`, differenced across the turn | Exact tokens + per-request rates (recovered from the running average). Poll, atomic-at-completion: **no live ticker, no TTFT** |
| others | — | Provider name + a dash (no adapter yet) |

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
