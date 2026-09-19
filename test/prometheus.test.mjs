// Validates prometheus.ts against real captured /metrics text (fixtures/,
// taken from the inference-hud VS Code extension's own verified captures).
//
// vLLM and SGLang are CUDA-only, so unlike the MTPLX/oMLX/llama.cpp tiers
// (each checked against a live local server) this tier can only be validated
// against these fixtures — real bytes an engine actually produced, not
// hand-written text. Run with: bun test/prometheus.test.mjs
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
  assert.ok(Math.abs(diff.ttftAvg - 0.3) < 1e-9, `ttftAvg=${diff.ttftAvg}`)
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

console.log(`\n${passed} passed`)
if (process.exitCode) {
  console.error("some tests failed")
}
