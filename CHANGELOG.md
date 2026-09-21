## [0.1.1] – 2026-09-21
### Changed
- Renamed `@banburist/opencode-hud` → `@banburist/opencode-engine-hud`
  (repo, plugin id, npm package). Published under the old scope for under a
  day with no adoption, so renamed now rather than later. The old package
  is deprecated on npm, pointing here.

## [0.1.0] – 2026-09-21
### Added
- Universal + Tier 2 telemetry for 12 provider ids across 7 engines
- Live-validated adapters for mtplx, omlx, llamacpp/llamafile,
  mlx-serve, splash, koboldcpp, vllm, sglang, vllm-mlx
- Fixture-verified support for aphrodite and lmdeploy (CUDA-only,
  not independently live-tested — see README)