// Validates omlx.ts — the arithmetic-mean recovery and the fallbacks around it.
//
// oMLX publishes only running averages, so a turn's rate has to be recovered
// from how the mean moved. That arithmetic is easy to get subtly wrong and
// impossible to notice by eye: a plausible-looking tok/s that is actually the
// server's lifetime average, or a negative rate from a counter reset rendered
// as if measured.
// Run with: bun test/omlx.test.mjs
import { strict as assert } from "node:assert"
import { recoverLatest, formatOmlxLine, toOmlxSample } from "../adapters/omlx.ts"

let passed = 0
function test(name, fn) {
  try {
    fn()
    passed++
    console.log("  ok ", name)
  } catch (e) {
    console.log("  FAIL", name, "\n      ", e.message)
    process.exitCode = 1
  }
}

const base = {
  requests: 10, prompt: 1000, completion: 500, cached: 200,
  avgGen: 30, avgPrefill: 400, model: "Qwen3.8-27B-MTPLX-Optimized-Speed",
}

// ---- the recovery arithmetic ------------------------------------------------
test("recovers a single new observation from the running mean", () => {
  // Ten requests averaging 30 tok/s, then an eleventh at 85: the new mean is
  // (10*30 + 85) / 11 = 35. Recovery must return 85, not 35.
  const newAvg = (10 * 30 + 85) / 11
  assert.ok(Math.abs(recoverLatest(30, 10, newAvg, 11) - 85) < 1e-9)
})

test("refuses when more than one request landed in the window", () => {
  // Two requests cannot be separated from a mean, so there is no honest answer.
  assert.equal(recoverLatest(30, 10, 33, 12), undefined)
})

test("refuses when no request landed", () => {
  assert.equal(recoverLatest(30, 10, 30, 10), undefined)
})

test("refuses a non-positive result rather than reporting it", () => {
  // A mean that fell far enough implies a negative contribution — impossible
  // for a rate, so it is a reset or a concurrent writer, not a measurement.
  assert.equal(recoverLatest(30, 10, 20, 11), undefined)
})

test("refuses when counters ran backwards", () => {
  assert.equal(recoverLatest(30, 10, 30, 9), undefined)
})

// ---- the rendered panel -----------------------------------------------------
test("with no baseline it shows the server average, labelled as such", () => {
  const out = formatOmlxLine(base, undefined)
  assert.ok(out.includes("(server avg)"), out)
  assert.ok(out.includes("(avg)"), out)
  // No per-turn token counts are claimed, because none are known.
  assert.ok(!out.includes("tok  ("), out)
})

test("a model switch falls back to averages rather than differencing", () => {
  // Differencing across a model change would attribute one model's tokens to
  // another.
  const now = { ...base, requests: 11, model: "some-other-model" }
  assert.ok(formatOmlxLine(now, base).includes("(server avg)"))
})

test("one new request yields a per-turn line with recovered rates", () => {
  const now = {
    ...base,
    requests: 11,
    completion: 590,
    prompt: 1120,
    cached: 232,
    avgGen: (10 * 30 + 85) / 11,
    avgPrefill: (10 * 400 + 620) / 11,
  }
  const out = formatOmlxLine(now, base).split("\n")
  assert.equal(out.length, 4)
  assert.equal(out[1], "85.0 tok/s", out.join(" | "))
  assert.equal(out[2], "prefill 620 tok/s")
  // Token counts are the window's own deltas: 590-500, 1120-1000, 232-200.
  assert.equal(out[3], "90 tok  (120 prompt, 32 cached)")
  assert.ok(!out.some((l) => l.includes("avg")), "no avg label on a real turn")
})

test("cached is omitted when the prefix cache did not move", () => {
  const now = { ...base, requests: 11, completion: 590, prompt: 1120, avgGen: (10 * 30 + 85) / 11 }
  assert.ok(formatOmlxLine(now, base).endsWith("90 tok  (120 prompt)"))
})

test("several requests in one window fall back to the lifetime average", () => {
  // The token deltas are still this window's, so no avg label — but the rate
  // cannot be attributed to one turn and must not pretend otherwise.
  const now = { ...base, requests: 13, completion: 800, prompt: 1500, avgGen: 31, avgPrefill: 410 }
  const out = formatOmlxLine(now, base).split("\n")
  assert.equal(out[1], "31.0 tok/s")
  assert.ok(out[3].startsWith("300 tok"))
})

// ---- the status mapping -----------------------------------------------------
test("a status payload missing fields maps to zeros, not NaN", () => {
  const s = toOmlxSample({})
  assert.deepEqual(
    { r: s.requests, p: s.prompt, c: s.completion, g: s.avgGen },
    { r: 0, p: 0, c: 0, g: 0 }
  )
  assert.equal(s.model, undefined)
})

test("the loaded model wins over the configured default", () => {
  assert.equal(toOmlxSample({ loaded_models: ["a"], default_model: "b" }).model, "a")
  assert.equal(toOmlxSample({ default_model: "b" }).model, "b")
})

console.log(`\n${passed} passed`)
