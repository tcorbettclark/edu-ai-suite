/* Compare the browser LSTM with the Python greedy reference outputs.
 *
 * Run it with: node examples/browser_lm/test_infer.js [corpus]
 */
"use strict";

const fs = require("fs");
const path = require("path");
const { createModel, generate } = require("./lm.js");

const here = __dirname;
const corpus = process.argv[2] || "dorian_gray";
const out = path.join(here, "out", corpus);
const reference = JSON.parse(fs.readFileSync(path.join(out, "reference.json"), "utf8"));

let failures = 0;
for (const name of ["char", "word"]) {
  const exportData = JSON.parse(
    fs.readFileSync(path.join(out, `${name}.compact.json`), "utf8")
  );
  const model = createModel(exportData);
  console.log(`\n${name}: vocab ${model.tokenizer.tokens.length}`);
  for (let i = 0; i < model.checkpoints.length; i++) {
    const snapshot = model.checkpoints[i];
    const started = performance.now();
    const got = generate(model, i, reference.prompt, {
      temperature: 0,
      maxNewTokens: reference.max_new_tokens,
      seed: 0,
    });
    const elapsed = performance.now() - started;
    const want = reference.models[name][String(snapshot.epoch)];
    const ok = got === want;
    if (!ok) failures++;
    console.log(
      `  epoch ${String(snapshot.epoch).padStart(2)}  ${ok ? "OK" : "MISMATCH"}  ${elapsed.toFixed(0)}ms`
    );
    if (!ok) {
      console.log(`    want: ${JSON.stringify(want)}`);
      console.log(`    got:  ${JSON.stringify(got)}`);
    }
  }
}

if (failures) {
  console.error(`\n${failures} mismatch(es)`);
  process.exitCode = 1;
} else {
  console.log("\nall greedy outputs match the Python reference");
}
