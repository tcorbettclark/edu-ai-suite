/* Browser inference for the tiny LSTM models exported by train_export.py.
 *
 * The exports are JSON: a tokenizer, an architecture, and a few checkpoints of Keras
 * weights. This file turns one checkpoint into text, using the same left-padded
 * context and temperature sampling as edu_ai_suite.lm.LanguageModel.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.BrowserLM = factory();
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const BOS_ID = 0;
  const UNK_ID = 1;
  const EOS_ID = 2;

  const ATTACH_PREV = new Set(".,;:!?…)]}%'’\"”»".split(""));
  const ATTACH_NEXT = new Set("‘“«([{".split(""));

  function splitSentences(text) {
    const out = [];
    for (const rawLine of String(text).split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line) continue;
      // A terminator ends a sentence only when whitespace follows it.
      for (const part of line.split(/(?<=[.!?…])\s+/u)) {
        const sentence = part.trim();
        if (sentence) out.push(sentence);
      }
    }
    return out;
  }

  function segment(text, kind) {
    if (kind === "char") return Array.from(text);
    return (
      text.match(/[\p{L}\p{N}_]+(?:['’][\p{L}\p{N}_]+)*|[^\p{L}\p{N}_\s]/gu) || []
    );
  }

  function decodeWord(tokens) {
    if (!tokens.length) return "";
    const pieces = [tokens[0]];
    for (let i = 1; i < tokens.length; i++) {
      const prev = tokens[i - 1];
      const token = tokens[i];
      const attach =
        ATTACH_NEXT.has(prev[prev.length - 1]) || ATTACH_PREV.has(token);
      pieces.push(attach ? token : " " + token);
    }
    return pieces.join("");
  }

  function decodeIds(model, ids) {
    const tokens = ids
      .filter((id) => id !== BOS_ID && id !== EOS_ID)
      .map((id) => model.tokenizer.tokens[id]);
    if (model.tokenization === "char") return tokens.join("");
    return decodeWord(tokens);
  }

  function encodePrompt(model, prompt) {
    const sentences = splitSentences(prompt);
    const unit = sentences.length ? sentences[sentences.length - 1] : "";
    return segment(unit, model.tokenization).map((token) =>
      model.tokenToId.has(token) ? model.tokenToId.get(token) : UNK_ID
    );
  }

  function contextFor(ids, width) {
    const tail = ids.slice(-width);
    return new Array(width - tail.length).fill(BOS_ID).concat(tail);
  }

  function decodeBase64(text) {
    const binary = atob(text);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  function weightData(weight) {
    if (weight.data_b64) {
      const bytes = decodeBase64(weight.data_b64);
      if (weight.dtype === "float32") {
        return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
      }
      const quantised = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const data = new Float32Array(quantised.length);
      if (weight.scales) {
        for (let row = 0; row < weight.rows; row++) {
          const scale = weight.scales[row];
          const start = row * weight.row_length;
          for (let i = 0; i < weight.row_length; i++) {
            data[start + i] = quantised[start + i] * scale;
          }
        }
      } else {
        for (let i = 0; i < quantised.length; i++) {
          data[i] = quantised[i] * weight.scale;
        }
      }
      return data;
    }
    return Float32Array.from(weight.data);
  }

  function createModel(exportData) {
    const tokenToId = new Map();
    exportData.tokenizer.tokens.forEach((token, id) => tokenToId.set(token, id));
    return {
      ...exportData,
      tokenToId,
      checkpoints: exportData.checkpoints.map((snapshot) => ({
        epoch: snapshot.epoch,
        loss: snapshot.loss,
        val_loss: snapshot.val_loss,
        weights: snapshot.weights.map((weight) => ({
          shape: weight.shape,
          data: weightData(weight),
        })),
      })),
    };
  }

  function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  function forward(weights, context, architecture) {
    const [embedding, kernel, recurrent, bias, logitsKernel, logitsBias] = weights;
    const units = architecture.units;
    const embeddingDim = architecture.embedding_dim;
    const vocabSize = logitsBias.data.length;
    const gates = 4 * units;

    let hidden = new Float32Array(units);
    let cell = new Float32Array(units);

    for (const id of context) {
      const z = new Float32Array(gates);
      z.set(bias.data);

      const embeddingOffset = id * embeddingDim;
      for (let j = 0; j < embeddingDim; j++) {
        const value = embedding.data[embeddingOffset + j];
        if (value === 0) continue;
        const kernelOffset = j * gates;
        for (let g = 0; g < gates; g++) {
          z[g] += value * kernel.data[kernelOffset + g];
        }
      }

      for (let k = 0; k < units; k++) {
        const value = hidden[k];
        if (value === 0) continue;
        const recurrentOffset = k * gates;
        for (let g = 0; g < gates; g++) {
          z[g] += value * recurrent.data[recurrentOffset + g];
        }
      }

      const nextHidden = new Float32Array(units);
      const nextCell = new Float32Array(units);
      for (let i = 0; i < units; i++) {
        const inputGate = sigmoid(z[i]);
        const forgetGate = sigmoid(z[units + i]);
        const candidate = Math.tanh(z[2 * units + i]);
        const outputGate = sigmoid(z[3 * units + i]);
        const value = forgetGate * cell[i] + inputGate * candidate;
        nextCell[i] = value;
        nextHidden[i] = outputGate * Math.tanh(value);
      }
      hidden = nextHidden;
      cell = nextCell;
    }

    const logits = new Float32Array(vocabSize);
    for (let v = 0; v < vocabSize; v++) {
      let value = logitsBias.data[v];
      for (let i = 0; i < units; i++) {
        value += hidden[i] * logitsKernel.data[i * vocabSize + v];
      }
      logits[v] = value;
    }
    return logits;
  }

  function sample(logits, temperature, random) {
    if (temperature <= 0) {
      let best = 0;
      for (let i = 1; i < logits.length; i++) {
        if (logits[i] > logits[best]) best = i;
      }
      return best;
    }

    let max = -Infinity;
    for (const value of logits) if (value > max) max = value;
    const weights = new Float32Array(logits.length);
    let total = 0;
    for (let i = 0; i < logits.length; i++) {
      const weight = Math.exp((logits[i] - max) / temperature);
      weights[i] = weight;
      total += weight;
    }
    let target = random() * total;
    for (let i = 0; i < weights.length; i++) {
      target -= weights[i];
      if (target <= 0) return i;
    }
    return weights.length - 1;
  }


  function probabilities(logits, temperature) {
    if (temperature <= 0) {
      const result = new Float32Array(logits.length);
      let best = 0;
      for (let i = 1; i < logits.length; i++) {
        if (logits[i] > logits[best]) best = i;
      }
      result[best] = 1;
      return result;
    }
    let max = -Infinity;
    for (const value of logits) if (value > max) max = value;
    const result = new Float32Array(logits.length);
    let total = 0;
    for (let i = 0; i < logits.length; i++) {
      const weight = Math.exp((logits[i] - max) / temperature);
      result[i] = weight;
      total += weight;
    }
    for (let i = 0; i < result.length; i++) result[i] /= total;
    return result;
  }

  function nextDistributionFromIds(model, checkpointIndex, ids, options) {
    const settings = { temperature: 0.7, topK: 10, ...options };
    const snapshot = model.checkpoints[checkpointIndex];
    if (!snapshot) throw new Error(`no checkpoint ${checkpointIndex}`);
    const context = contextFor(ids, model.architecture.context_width);
    const logits = forward(snapshot.weights, context, model.architecture);
    const probs = probabilities(logits, settings.temperature);
    return Array.from(probs, (p, id) => ({
      id,
      token: model.tokenizer.tokens[id],
      p,
    }))
      .filter((item) => item.id !== BOS_ID)
      .sort((a, b) => b.p - a.p)
      .slice(0, settings.topK);
  }

  function nextDistribution(model, checkpointIndex, prompt, options) {
    return nextDistributionFromIds(
      model,
      checkpointIndex,
      encodePrompt(model, prompt),
      options
    );
  }

  function sampleNextId(model, checkpointIndex, ids, options) {
    const settings = { temperature: 0.7, random: Math.random, ...options };
    const snapshot = model.checkpoints[checkpointIndex];
    if (!snapshot) throw new Error(`no checkpoint ${checkpointIndex}`);
    const context = contextFor(ids, model.architecture.context_width);
    const logits = forward(snapshot.weights, context, model.architecture);
    for (let attempt = 0; attempt < 8; attempt++) {
      const id = sample(logits, settings.temperature, settings.random);
      if (id !== BOS_ID) return id;
    }
    return EOS_ID;
  }

  function mulberry32(seed) {
    let state = seed >>> 0;
    return function () {
      state = (state + 0x6d2b79f5) >>> 0;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  function generate(model, checkpointIndex, prompt, options) {
    const settings = { temperature: 0.7, maxNewTokens: 48, seed: 0, ...options };
    const snapshot = model.checkpoints[checkpointIndex];
    if (!snapshot) throw new Error(`no checkpoint ${checkpointIndex}`);
    const promptIds = encodePrompt(model, prompt);
    const width = model.architecture.context_width;
    let context = contextFor(promptIds, width);
    const random = mulberry32(settings.seed);
    const output = promptIds.slice();

    for (let step = 0; step < settings.maxNewTokens; step++) {
      const logits = forward(snapshot.weights, context, model.architecture);
      const id = sample(logits, settings.temperature, random);
      if (id === EOS_ID) break;
      if (id === BOS_ID) continue;
      output.push(id);
      context = context.slice(1);
      context.push(id);
    }
    return decodeIds(model, output);
  }

  return {
    BOS_ID,
    UNK_ID,
    EOS_ID,
    createModel,
    generate,
    nextDistribution,
    nextDistributionFromIds,
    sampleNextId,
    mulberry32,
    splitSentences,
    segment,
    decodeIds,
    encodePrompt,
  };
});
