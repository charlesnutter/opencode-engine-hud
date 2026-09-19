# Fixture provenance

Not all fixtures are equal evidence. Each file below is labelled with where it
came from, because "the test passes" means something different depending on it.

| File | Provenance |
|---|---|
| `vllm-mlx-idle.prom`, `vllm-mlx-after.prom` | **Live capture.** Taken from a local vllm-mlx server on Apple Silicon, bracketing one real generation whose response reported `usage: {prompt_tokens: 33, completion_tokens: 50}`. The test asserts against those numbers. |
| `vllm-metal-before.prom`, `vllm-metal-after.prom` | **Live capture.** Taken from upstream vLLM running under vllm-metal on Apple Silicon, bracketing one real generation (`usage: {prompt_tokens: 35, completion_tokens: 35}`). |
| `vllm-idle.prom`, `vllm-busy.prom` | **Real capture**, inherited from the inference-hud VS Code extension. Real bytes a real vLLM emitted, but not captured here and not cross-checked against a response body. |
| `sglang-idle.prom`, `sglang-busy.prom` | **Synthesized** — inherited from the same project, which documents them as hand-written from `python/sglang/srt/observability/metrics_collector.py`. SGLang has no macOS backend. Names/types/labels are from source; values are plausible, not measured. |
| `lmdeploy-before.prom`, `lmdeploy-after.prom` | **Synthesized.** LMDeploy is CUDA-only. Names and the `{model_name,engine}` label shape are verified against `lmdeploy/metrics/loggers.py`; values chosen to make the diff arithmetic checkable. |
| *(Aphrodite has no fixture)* | Its test **derives** input by swapping `vllm:` → `aphrodite:` in the real vLLM capture, which is precisely the documented difference between them. No real Aphrodite bytes exist here. |

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
