# Fixture provenance

Not all fixtures are equal evidence. Each file below is labelled with where it
came from, because "the test passes" means something different depending on it.

| File | Provenance |
|---|---|
| `vllm-mlx-idle.prom`, `vllm-mlx-after.prom` | **Live capture.** Taken from a local vllm-mlx server on Apple Silicon, bracketing one real generation whose response reported `usage: {prompt_tokens: 33, completion_tokens: 50}`. The test asserts against those numbers. |
| `vllm-metal-before.prom`, `vllm-metal-after.prom` | **Live capture.** Taken from upstream vLLM running under vllm-metal on Apple Silicon, bracketing one real generation (`usage: {prompt_tokens: 35, completion_tokens: 35}`). |
| `vllm-idle.prom`, `vllm-busy.prom` | **Real capture**, inherited from the inference-hud VS Code extension. Real bytes a real vLLM emitted, but not captured here and not cross-checked against a response body. |
| `sglang-stream-before.prom`, `sglang-stream-after.prom` | **Live capture.** SGLang on its MLX/Apple-Silicon backend (`SGLANG_USE_MLX=1 --enable-metrics`), bracketing one real **streaming** generation (`usage: {prompt_tokens: 35, completion_tokens: 25}`). This is the path OpenCode uses. |
| `sglang-before.prom`, `sglang-after.prom` | **Live capture**, same server, one **non-streaming** generation. Kept deliberately: SGLang stamps TTFT at completion when not streaming, so these are the regression case for the degenerate decode-window guard. |
| `koboldcpp-before.json`, `koboldcpp-after.json` | **Live capture.** KoboldCpp v1.121, macOS arm64, bracketing one generation (`usage: {prompt_tokens: 16, completion_tokens: 83}`). Short prompt, so `last_process_time` sits on the ~1ms timer floor and the server reports 16000 tok/s prefill — the regression case for the floor guard. |
| `koboldcpp-novel-*.json` | **Live capture.** A long prompt with no prefix-cache overlap, so the prefill timer measures real work (0.197s over 2818 tokens). The positive case: prefill survives the floor. |
| `koboldcpp-cachehit-*.json` | **Live capture.** The identical prompt re-sent, a full prefix-cache hit. The server reports `process_time: 0.0` and `speed: 0` — not a huge rate. |
| `koboldcpp-longprompt-*.json` | **Live capture.** A *partial* cache hit: 2016 prompt tokens but only the uncached suffix timed, reported as 18000 tok/s. Pins the known, undetectable overstatement. |
| `lmdeploy-before.prom`, `lmdeploy-after.prom` | **Synthesized.** LMDeploy is CUDA-only. Names and the `{model_name,engine}` label shape are verified against `lmdeploy/metrics/loggers.py`; values chosen to make the diff arithmetic checkable. |
| *(Aphrodite has no fixture)* | Its test **derives** input by swapping `vllm:` → `aphrodite:` in the real vLLM capture, which is precisely the documented difference between them. No real Aphrodite bytes exist here. |

KoboldCpp fixtures are JSON, not Prometheus text, and each carries a
`_provenance` object recording the host, server version, scenario and the
response body's own `usage` for the generation it brackets — the tests assert
against that block rather than against numbers typed into the test file.

## What a passing test does and doesn't tell you

Fixtures verify the parser and the diff arithmetic: label summing, the
`_created`-line trap, prefix boundaries, counter resets, blended windows.

They cannot verify that a current build of the engine actually emits those
names, that the endpoint exists, or that the flag to enable it is what the docs
say. Every one of those has been wrong at least once here — a flag documented
as `--metrics` was really `--enable-metrics`; an engine widely described as
having a Prometheus endpoint had none at all. Only running the thing catches
that.

Replace any synthesized pair with a real capture if a suitable machine becomes
available.

## Replacing a synthesized fixture with a real one

`scripts/capture-fixture.sh` captures a real before/after pair around exactly
one generation, and writes a provenance header recording the host, endpoint,
model and the response's own `usage` block:

```bash
./scripts/capture-fixture.sh http://127.0.0.1:30000 sglang: sglang
```

It refuses to write anything if `/metrics` is missing, carries no lines with
the expected prefix, or is byte-identical before and after a generation — the
three ways a capture would be worthless. Afterwards, assert the printed deltas
against the printed `usage` block in `test/prometheus.test.mjs` and update the
table above.
