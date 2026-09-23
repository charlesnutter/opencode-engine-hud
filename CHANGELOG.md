## [Unreleased]
### Added
- A turn whose engine figures were declined now says why:
  `engine telemetry from the next turn` when there is no baseline yet,
  `engine data skipped: overlapping requests` when the engine's window held
  other requests.

### Fixed
- A rate measured over the whole turn, used when the stream window cannot be
  timed, is now labelled `overall`. It could be ~10x lower than the decode
  rate and was shown as if it were one.
- A time-to-first-token at or past the end of its own turn, or negative, is
  now suppressed rather than shown.
- Two turns completing close together could render in the wrong order; only
  the latest turn now updates the panel.
- `vllm`, `sglang`, `vllmmlx`, `aphrodite` and `lmdeploy` rendered figures
  for several requests as one turn's whenever other requests reached the
  engine in the same window. Measured on the successor plugin: a 46-token
  answer showed `116135.1 tok/s` over `8594 tok`. Such a turn now shows the
  universal line with the notice above. A window is this turn's only when
  the engine's token count matches OpenCode's own: a non-streamed request
  records no time-to-first-token, so counting those alone missed it.
- On `sglang`, `vllmmlx` and `lmdeploy`, the turn's total and engine-derived
  decode rate were averaged with any other request in the window, such as a
  rejected ~0s title request, which halved the total. The engine's duration
  is now used only when exactly one was recorded; otherwise OpenCode's timing.

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