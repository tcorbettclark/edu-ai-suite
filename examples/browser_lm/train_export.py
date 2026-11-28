"""Train small language models on a text corpus and export browser checkpoints.

One character-level model and one word-level model are trained from the same corpus.
Each run keeps the random initial weights and a few later epochs, so a browser demo can
show the same prompt against different amounts of training.

The corpus is selected by ``BROWSER_LM_CORPUS``: ``dorian_gray`` (the default),
``happy_prince``, or ``tiny_stories``. The script prints loss, perplexity and token
accuracy for both training and validation at every saved checkpoint.

Run it with ``BROWSER_LM_CORPUS=tiny_stories uv run python examples/browser_lm/train_export.py``.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

# Keep progress visible when stdout is a pipe or a tee, not a terminal.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np

from edu_ai_suite.lm import (
    LSTM,
    LanguageModel,
    TrainingOptions,
    make_pairs,
    tokenizer_from_text,
)

from corpus import CORPORA, CORPUS_NAME, load_corpus
from manifest import write_manifest

# Settings.
OUTPUT_ROOT = Path(__file__).with_name("out")
SEED = 0
VALIDATION_FRACTION = 0.1
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
CHECKPOINT_EPOCHS = (5, 20, 50)
TEMPERATURES = (0.0, 0.7, 1.0)
MAX_NEW_TOKENS = 48
ACCURACY_PAIRS = 4_096

# The two tokenizations: the same task at two different granularities.
RUNS = (
    {
        "name": "char",
        "tokenization": "char",
        "context_width": 24,
        "embedding_dim": 16,
        "units": 64,
        "epochs": 50,
        "keep_top": None,
    },
    {
        "name": "word",
        "tokenization": "word",
        "context_width": 12,
        "embedding_dim": 16,
        "units": 48,
        "epochs": 50,
        "keep_top": None,
    },
)


def weight_records(model: keras.Model) -> list[dict]:
    """Every weight tensor as a flat list, with its name and shape."""
    records = []
    for layer in model.layers:
        for weight in layer.weights:
            records.append(
                {
                    "name": weight.name,
                    "shape": list(weight.shape),
                    "data": weight.numpy().reshape(-1).tolist(),
                }
            )
    return records


def token_accuracy(
    model: keras.Model,
    pairs,
    limit: int = ACCURACY_PAIRS,
    batch_size: int = 256,
) -> float:
    """Fraction of next-token predictions that are exactly right."""
    n = min(len(pairs), limit)
    if n == 0:
        return float("nan")
    correct = 0
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        logits = model(pairs.contexts[start:stop], training=False).numpy()
        correct += int((logits.argmax(axis=-1) == pairs.targets[start:stop]).sum())
    return correct / n


def checkpoint(
    model: keras.Model,
    epoch: int,
    loss: float | None,
    val_loss: float | None,
    train_accuracy: float,
    val_accuracy: float,
) -> dict:
    """One browser-loadable snapshot of the model's weights and measurements."""
    return {
        "epoch": epoch,
        "loss": loss,
        "val_loss": val_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "weights": weight_records(model),
    }


def train_run(run: dict, corpus: str, corpus_config: dict) -> dict:
    """Train one tokenization and return its browser export."""
    name = run["name"]
    prompt = corpus_config["prompt"]
    print(f"\n=== {name} ===")

    tokenizer = tokenizer_from_text(corpus, run["tokenization"])
    if run["keep_top"] is not None:
        tokenizer = tokenizer.keep_top(run["keep_top"])
    print(f"  vocabulary: {tokenizer.vocab_size:,}")
    print(f"  corpus tokens: {tokenizer.total_token_count:,}")

    architecture = LSTM(
        context_width=run["context_width"],
        embedding_dim=run["embedding_dim"],
        units=run["units"],
        layers=1,
    )
    model = LanguageModel(architecture, tokenizer)

    # The same split and seed that fit() will use, so the accuracies describe the
    # same training and validation sets the losses do.
    pairs = make_pairs(tokenizer.encode(corpus), architecture.context_width)
    train_pairs, val_pairs = pairs.split(VALIDATION_FRACTION, seed=SEED)

    # Build once with the training seed so the random checkpoint is the weights the
    # optimizer actually starts from. fit() builds the training model with the same
    # seed, so the two initial models agree.
    initial = architecture.build(vocab_size=tokenizer.vocab_size, seed=SEED)
    print(f"  parameters: {initial.count_params():,}")
    print(f"  training pairs: {len(train_pairs):,}, validation pairs: {len(val_pairs):,}")
    checkpoints = [
        checkpoint(
            initial,
            epoch=0,
            loss=None,
            val_loss=None,
            train_accuracy=token_accuracy(initial, train_pairs),
            val_accuracy=token_accuracy(initial, val_pairs),
        )
    ]
    print(
        f"  random checkpoint: train acc {checkpoints[0]['train_accuracy'] * 100:.1f}%, "
        f"val acc {checkpoints[0]['val_accuracy'] * 100:.1f}%"
    )

    started = time.time()

    def on_epoch(stats: dict) -> None:
        epoch = int(stats["epoch"])
        elapsed = time.time() - started
        loss = float(stats["loss"])
        val_loss = float(stats["val_loss"])
        if epoch not in CHECKPOINT_EPOCHS:
            print(
                f"  epoch {epoch:2d}: loss {loss:.3f}, val loss {val_loss:.3f}, "
                f"train ppl {math.exp(loss):.1f}, val ppl {math.exp(val_loss):.1f}, "
                f"{elapsed:.1f}s"
            )
            return
        snapshot = checkpoint(
            model.keras_model,
            epoch=epoch,
            loss=loss,
            val_loss=val_loss,
            train_accuracy=token_accuracy(model.keras_model, train_pairs),
            val_accuracy=token_accuracy(model.keras_model, val_pairs),
        )
        checkpoints.append(snapshot)
        print(
            f"  epoch {epoch:2d}: "
            f"loss {loss:.3f}, val loss {val_loss:.3f}, "
            f"train ppl {math.exp(loss):.1f}, val ppl {math.exp(val_loss):.1f}, "
            f"train acc {snapshot['train_accuracy'] * 100:.1f}%, "
            f"val acc {snapshot['val_accuracy'] * 100:.1f}%, "
            f"{elapsed:.1f}s  [checkpoint]"
        )

    model.fit(
        corpus,
        TrainingOptions(
            epochs=run["epochs"],
            batch_size=BATCH_SIZE,
            validation_fraction=VALIDATION_FRACTION,
            learning_rate=LEARNING_RATE,
            seed=SEED,
        ),
        on_epoch=on_epoch,
    )

    print(f"  finished in {time.time() - started:.1f}s")
    for snapshot in checkpoints:
        if snapshot["epoch"] == 0:
            continue
        print(f"  samples at epoch {snapshot['epoch']}:")
        # Rebuild a LanguageModel around the saved weights for generation. The
        # checkpoint holds Keras weights in the same layer order as the fitted model.
        probe = LanguageModel(architecture, tokenizer)
        probe.keras_model = architecture.build(vocab_size=tokenizer.vocab_size, seed=SEED)
        probe.keras_model.set_weights(
            [
                np.asarray(weight["data"], dtype=np.float32).reshape(weight["shape"])
                for weight in snapshot["weights"]
            ]
        )
        for temperature in TEMPERATURES:
            text = probe.generate(
                prompt,
                temperature=temperature,
                max_new_tokens=MAX_NEW_TOKENS,
                seed=SEED,
            )
            print(f"    T={temperature:.1f}: {text!r}")

    export = {
        "format": "edu-ai-suite-browser-lm-v1",
        "tokenization": name,
        "prompt": prompt,
        "max_new_tokens": MAX_NEW_TOKENS,
        "corpus": {
            "name": CORPUS_NAME,
            "title": corpus_config["title"],
            "author": corpus_config["author"],
            "source": corpus_config["source"],
            "licence": corpus_config["licence"],
            "characters": len(corpus),
            "tokens": tokenizer.total_token_count,
        },
        "architecture": {
            "name": "lstm",
            "context_width": run["context_width"],
            "embedding_dim": run["embedding_dim"],
            "units": run["units"],
            "layers": 1,
        },
        "tokenizer": tokenizer.to_dict(),
        "checkpoints": checkpoints,
    }
    return export


def main() -> None:
    corpus_config = CORPORA[CORPUS_NAME]
    output_dir = OUTPUT_ROOT / CORPUS_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(corpus_config)
    print(f"corpus: {corpus_config['title']} ({CORPUS_NAME})")
    print(f"  {len(corpus):,} characters")
    print(f"  output: {output_dir}")
    print("  this run trains two models; expect minutes to about half an hour on a laptop")

    for run in RUNS:
        export = train_run(run, corpus, corpus_config)
        path = output_dir / f"{run['name']}.json"
        path.write_text(json.dumps(export, separators=(",", ":")), encoding="utf-8")
        print(f"  wrote {path} ({path.stat().st_size:,} bytes)")

    print(f"\nwrote {write_manifest(output_dir, CORPUS_NAME, corpus_config)}")


if __name__ == "__main__":
    main()
