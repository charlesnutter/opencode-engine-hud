// Validates mtplx.ts against both panel states observed live in the sidebar.
//
// The completed turn rendered five lines correctly. The INTERRUPTED turn
// rendered four, two of which carried placeholders:
//
//     MTPLX  arsis-dev-ukisai-swift-…
//     27.5 tok/s  ttft ?s
//     prefill ? tok/s
//     85 tok  3.35s
//
// MTPLX's receipt drops ttft_s and prefill_tok_s when a turn is interrupted,
// while keeping the decode rate and token count. Those "?" are the failure
// this module exists to prevent: a placeholder where a measurement belongs.
// Run with: bun test/mtplx.test.mjs
import { strict as assert } from "node:assert"
import { formatMtplxLine } from "../mtplx.ts"

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

const MODEL = "arsis-dev-ukisai-swift-qwen3-8-27b-mtplx"

// Values read off the completed-turn screenshot.
const COMPLETED = {
  decode_tok_s: 30.3,
  ttft_s: 0.89,
  prefill_tok_s: 401,
  completion_tokens: 22,
  reasoning_tokens: 0,
  request_elapsed_s: 1.62,
  verify_calls: 7,
  mean_accept_probability_by_depth: [0.79, 0.83, 0.67],
}

// The interrupted turn: same receipt minus the two timing fields.
const INTERRUPTED = {
  decode_tok_s: 27.5,
  ttft_s: null,
  prefill_tok_s: null,
  completion_tokens: 85,
  reasoning_tokens: 0,
  request_elapsed_s: 3.35,
  verify_calls: 0,
}

test("a completed turn renders all five lines", () => {
  const out = formatMtplxLine(COMPLETED, MODEL).split("\n")
  assert.equal(out.length, 5, out.join(" | "))
  assert.ok(out[0].startsWith("MTPLX  "))
  assert.ok(out[1].includes("30.3 tok/s") && out[1].includes("ttft 0.89s"))
  assert.equal(out[2], "prefill 401 tok/s")
  assert.ok(out[3].startsWith("22 tok"))
  assert.ok(out[4].startsWith("MTP 3.14x"), out[4])
})

test("an interrupted turn omits the missing figures instead of printing ?", () => {
  const out = formatMtplxLine(INTERRUPTED, MODEL)
  assert.ok(!out.includes("?"), `no placeholder may reach the panel:\n${out}`)
  // What it still knows is kept.
  assert.ok(out.includes("27.5 tok/s"))
  assert.ok(out.includes("85 tok"))
  assert.ok(out.includes("3.35s"))
  // What it does not know is absent entirely, not blanked.
  assert.ok(!out.includes("ttft"))
  assert.ok(!out.includes("prefill"))
})

test("an interrupted turn is three lines, not four with holes", () => {
  const out = formatMtplxLine(INTERRUPTED, MODEL).split("\n")
  assert.deepEqual(out, ["MTPLX  arsis-dev-ukisai-swift-…", "27.5 tok/s", "85 tok  3.35s"])
})

test("rate and TTFT are independently optional", () => {
  // TTFT without a rate is still worth showing.
  const out = formatMtplxLine({ ...INTERRUPTED, decode_tok_s: null, ttft_s: 1.2 }, MODEL)
  assert.ok(out.includes("ttft 1.20s"))
  assert.ok(!out.includes("tok/s"))
  assert.ok(!out.includes("?"))
})

test("no verify passes means no MTP line, not a division by zero", () => {
  // Match the line, not the substring: the header "MTPLX" contains "MTP".
  const mtpLine = (l) => formatMtplxLine(l, MODEL).split("\n").find((x) => x.startsWith("MTP "))
  assert.equal(mtpLine(INTERRUPTED), undefined)
  assert.equal(mtpLine({ ...COMPLETED, verify_calls: 0 }), undefined)
  assert.ok(mtpLine(COMPLETED)?.startsWith("MTP 3.14x"))
})

test("reasoning tokens show as a subset of the topline", () => {
  const out = formatMtplxLine({ ...COMPLETED, completion_tokens: 1247, reasoning_tokens: 889 }, MODEL)
  assert.ok(out.includes("1247 tok (889 think)"), out)
  assert.ok(!out.includes("(+"), "never the additive form")
})

test("an empty receipt renders the header alone, with no holes", () => {
  const out = formatMtplxLine({}, MODEL)
  assert.equal(out, "MTPLX  arsis-dev-ukisai-swift-…")
  assert.ok(!out.includes("?"))
})

test("NaN is treated as absent, not rendered", () => {
  const out = formatMtplxLine({ ...COMPLETED, decode_tok_s: NaN, prefill_tok_s: NaN }, MODEL)
  assert.ok(!out.includes("?"), out)
  assert.ok(!out.includes("prefill"))
})

console.log(`\n${passed} passed`)
