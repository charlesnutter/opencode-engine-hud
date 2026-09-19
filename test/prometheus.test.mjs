// Validates prometheus.ts against real captured /metrics text (fixtures/,
// taken from the inference-hud VS Code extension's own verified captures).
//
// vLLM, SGLang and Aphrodite are CUDA-only, so those are validated against
// captured fixtures — real bytes those engines produced, not hand-written
// text. The vllm-mlx fixtures are different: they were captured live from a
// local server on this machine, before and after a single real generation,
// so its assertions check values cross-checked against that response's own
// `usage`. Run with: bun test/prometheus.test.mjs
import { strict as assert } from "node:assert"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import path from "node:path"
import {
  sumLabeledMetric,
  parsePromSample,
  diffPromSamples,
  VLLM_SPEC,
  SGLANG_SPEC,
  APHRODITE_SPEC,
  VLLM_MLX_SPEC,
} from "../prometheus.ts"

const dir = path.dirname(fileURLToPath(import.meta.url))
const fixture = (name) => readFileSync(path.join(dir, "..", "fixtures", name), "utf8")

let passed = 0
function test(name, fn) {
  try {
    fn()
    passed++
    console.log(`  ok  ${name}`)
  } catch (e) {
    console.error(`FAIL  ${name}`)
    console.error(`      ${e.message}`)
    process.exitCode = 1
  }
}

// ---- vLLM: parse a real idle capture exactly -------------------------------
test("vLLM: parses the real idle capture's counters exactly", () => {
  const text = fixture("vllm-idle.prom")
  const s = parsePromSample(text, VLLM_SPEC)
  assert.ok(s, "expected a sample (prefix present)")
  assert.equal(s.prompt, 78)
  assert.equal(s.generation, 700)
  assert.equal(s.cached, 0)
  assert.equal(s.ttftCount, 2)
  assert.ok(Math.abs(s.ttftSum - 4.6046302318573) < 1e-9)
})

// ---- vLLM: a turn diffed across two synthetic-but-realistic snapshots -----
test("vLLM: diff across a turn gives exact tokens and a TTFT average", () => {
  const idle = fixture("vllm-idle.prom")
  // Bump generation +50, prompt +12, and one more TTFT sample (0.30s) landed —
  // the same technique the extension's own test suite uses: a real capture,
  // with only the counters a turn would advance edited by exact amounts.
  const after = idle
    .replace(/(vllm:generation_tokens_total\{[^}]*\}) 700\.0/, "$1 750.0")
    .replace(/(vllm:prompt_tokens_total\{[^}]*\}) 78\.0/, "$1 90.0")
    .replace(/(vllm:time_to_first_token_seconds_count\{[^}]*\}) 2\.0/, "$1 3.0")
    .replace(/(vllm:time_to_first_token_seconds_sum\{[^}]*\}) 4\.6046302318573/, "$1 4.9046302318573")
  assert.notEqual(after, idle, "the fixture lines this test relies on are present")

  const before = parsePromSample(idle, VLLM_SPEC)
  const now = parsePromSample(after, VLLM_SPEC)
  const diff = diffPromSamples(before, now)
  assert.ok(diff)
  assert.equal(diff.completionTokens, 50)
  assert.equal(diff.promptTokens, 12)
  assert.equal(diff.cachedTokens, 0)
  assert.ok(Math.abs(diff.ttft - 0.3) < 1e-9, `ttft=${diff.ttft}`)
})

// ---- vLLM: nothing landed -> no completion (not a zero-token turn) --------
test("vLLM: an unchanged read reports no completion", () => {
  const s = parsePromSample(fixture("vllm-idle.prom"), VLLM_SPEC)
  assert.equal(diffPromSamples(s, s), null)
})

// ---- vLLM: a server restart (counters go backwards) is not a negative rate
test("vLLM: counters running backwards are treated as a restart, not a turn", () => {
  const before = parsePromSample(fixture("vllm-idle.prom"), VLLM_SPEC)
  const after = { ...before, generation: before.generation - 10 }
  assert.equal(diffPromSamples(before, after), null)
})

// ---- SGLang: parses a real capture, summing the labelled cache counter ----
test("SGLang: parses the real idle capture's counters exactly", () => {
  const s = parsePromSample(fixture("sglang-idle.prom"), SGLANG_SPEC)
  assert.ok(s)
  assert.equal(s.prompt, 2199)
  assert.equal(s.generation, 400)
  // cached_tokens_total is labelled by cache_source; only "device" is present
  // in this capture, so the sum equals that one series.
  assert.equal(s.cached, 2198)
  assert.equal(s.ttftCount, 3)
  assert.ok(Math.abs(s.ttftSum - 0.8412) < 1e-9)
})

test("SGLang: diff across a turn gives exact tokens", () => {
  const idle = fixture("sglang-idle.prom")
  const after = idle.replace(/(sglang:generation_tokens_total\{[^}]*\}) 400\.0/, "$1 487.0")
  const before = parsePromSample(idle, SGLANG_SPEC)
  const now = parsePromSample(after, SGLANG_SPEC)
  const diff = diffPromSamples(before, now)
  assert.ok(diff)
  assert.equal(diff.completionTokens, 87)
})

// ---- sumLabeledMetric: the `_created` line trap ----------------------------
test("sumLabeledMetric: a name that is a prefix of another metric is not conflated with it", () => {
  const text = [
    'vllm:generation_tokens_total{engine="0"} 10.0',
    'vllm:generation_tokens_total_extra_metric{engine="0"} 99999.0',
    'vllm:generation_tokens_created{engine="0"} 1789775000.0',
  ].join("\n")
  assert.equal(sumLabeledMetric(text, "vllm:generation_tokens_total"), 10)
})

test("sumLabeledMetric: sums across multiple label sets (data-parallel ranks)", () => {
  const text = ['vllm:generation_tokens_total{engine="0"} 10.0', 'vllm:generation_tokens_total{engine="1"} 25.0'].join(
    "\n"
  )
  assert.equal(sumLabeledMetric(text, "vllm:generation_tokens_total"), 35)
})

// ---- an engine's own metrics text is rejected by the other's spec ---------
test("cross-check: vLLM text does not parse against the SGLang spec", () => {
  assert.equal(parsePromSample(fixture("vllm-idle.prom"), SGLANG_SPEC), null)
})
test("cross-check: SGLang text does not parse against the vLLM spec", () => {
  assert.equal(parsePromSample(fixture("sglang-idle.prom"), VLLM_SPEC), null)
})

// ---- vllm-mlx: captured LIVE, before/after one real generation ------------
// The turn between these two captures reported, in its own response body:
//   usage: { prompt_tokens: 33, completion_tokens: 50 }
// Both histograms advanced by exactly 1, so TTFT and duration are that single
// request's own values, not an average.
test("vllm-mlx: parses the live capture's counters and histograms", () => {
  const s = parsePromSample(fixture("vllm-mlx-idle.prom"), VLLM_MLX_SPEC)
  assert.ok(s, "expected a sample (vllm_mlx_ prefix present)")
  assert.equal(s.prompt, 72)
  assert.equal(s.generation, 124)
  assert.equal(s.cached, 0) // vllm-mlx publishes no prompt-cache counter
  assert.equal(s.ttftCount, 2)
  assert.equal(s.durationCount, 2)
})

test("vllm-mlx: diff matches the response's own usage, exactly", () => {
  const before = parsePromSample(fixture("vllm-mlx-idle.prom"), VLLM_MLX_SPEC)
  const now = parsePromSample(fixture("vllm-mlx-after.prom"), VLLM_MLX_SPEC)
  const diff = diffPromSamples(before, now)
  assert.ok(diff)
  // Cross-checked against the live response body's usage block.
  assert.equal(diff.completionTokens, 50)
  assert.equal(diff.promptTokens, 33)
  // One request in the window -> these are its own values, not an average.
  assert.equal(diff.ttftExact, true)
  assert.ok(Math.abs(diff.ttft - 0.07569008297287) < 1e-6, `ttft=${diff.ttft}`)
  assert.ok(Math.abs(diff.durationS - 0.192384666996076) < 1e-6, `duration=${diff.durationS}`)
  // Engine-measured decode rate: tokens / (duration - ttft), excluding prefill.
  assert.ok(diff.decodeTokS > 400 && diff.decodeTokS < 460, `decodeTokS=${diff.decodeTokS}`)
})

test("vllm-mlx: two requests in one window drops the exact flag and the decode rate", () => {
  const before = parsePromSample(fixture("vllm-mlx-idle.prom"), VLLM_MLX_SPEC)
  // Same generation delta, but both histograms advanced by 2 rather than 1.
  const now = {
    ...parsePromSample(fixture("vllm-mlx-after.prom"), VLLM_MLX_SPEC),
    ttftCount: before.ttftCount + 2,
    durationCount: before.durationCount + 2,
  }
  const diff = diffPromSamples(before, now)
  assert.ok(diff)
  assert.equal(diff.ttftExact, false)
  assert.equal(diff.decodeTokS, undefined, "no per-request rate when several requests blend")
  assert.ok(diff.ttft !== undefined, "still reports the window average")
})

// ---- Aphrodite: vLLM's shape under its own prefix -------------------------
test("Aphrodite: parses vLLM-shaped metrics under the aphrodite: prefix", () => {
  // Aphrodite is CUDA-only; this reuses the real vLLM capture with the prefix
  // swapped, which is precisely the documented difference between them.
  const text = fixture("vllm-idle.prom").replace(/vllm:/g, "aphrodite:")
  const s = parsePromSample(text, APHRODITE_SPEC)
  assert.ok(s)
  assert.equal(s.prompt, 78)
  assert.equal(s.generation, 700)
  assert.equal(s.ttftCount, 2)
})

test("cross-check: vLLM text does not parse against the vllm-mlx spec", () => {
  assert.equal(parsePromSample(fixture("vllm-idle.prom"), VLLM_MLX_SPEC), null)
})

// ---- vLLM: validated LIVE, via vllm-metal on Apple Silicon ---------------
// vllm-metal runs upstream vLLM's own server with an MLX/Metal compute
// backend, so its /metrics is vLLM's. These two captures bracket one real
// generation whose response reported usage: { prompt_tokens: 35,
// completion_tokens: 35 } — which is what makes this tier live-validated
// rather than fixtures-only.
test("vLLM (live via vllm-metal): diff matches the response's own usage", () => {
  const before = parsePromSample(fixture("vllm-metal-before.prom"), VLLM_SPEC)
  const now = parsePromSample(fixture("vllm-metal-after.prom"), VLLM_SPEC)
  const diff = diffPromSamples(before, now)
  assert.ok(diff)
  assert.equal(diff.completionTokens, 35)
  assert.equal(diff.promptTokens, 35)
  // One request landed, so the TTFT delta is that request's own value.
  assert.equal(diff.ttftExact, true)
  assert.ok(Math.abs(diff.ttft - 1.008126974105835) < 1e-6, `ttft=${diff.ttft}`)
  // vLLM publishes no duration histogram, so no engine-measured decode rate.
  assert.equal(diff.decodeTokS, undefined)
})

console.log(`\n${passed} passed`)
if (process.exitCode) {
  console.error("some tests failed")
}
