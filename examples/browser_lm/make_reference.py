"""Reconstruct the exported browser models and write greedy reference outputs.

The browser test compares its greedy generation with these strings, so a mismatch is
caught as a runtime bug rather than a difference in random sampling.

Run it with ``uv run python examples/browser_lm/make_reference.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from edu_ai_suite.lm import LSTM, LanguageModel
from edu_ai_suite.lm.tokenizer import tokenizer_from_dict

from corpus import CORPUS_NAME

# Settings.
OUTPUT_DIR = Path(__file__).with_name("out") / CORPUS_NAME
SEED = 0


def weights_from_records(records: list[dict]) -> list[np.ndarray]:
    """Keras weight arrays from the flat records in the browser export."""
    return [
        np.asarray(record["data"], dtype=np.float32).reshape(record["shape"])
        for record in records
    ]


def main() -> None:
    references = {"models": {}}
    for name in ("char", "word"):
        export = json.loads((OUTPUT_DIR / f"{name}.json").read_text(encoding="utf-8"))
        tokenizer = tokenizer_from_dict(export["tokenizer"])
        architecture = LSTM(
            context_width=export["architecture"]["context_width"],
            embedding_dim=export["architecture"]["embedding_dim"],
            units=export["architecture"]["units"],
            layers=export["architecture"]["layers"],
        )
        references["prompt"] = export["prompt"]
        references["max_new_tokens"] = export["max_new_tokens"]
        references["models"][name] = {}
        for snapshot in export["checkpoints"]:
            model = LanguageModel(architecture, tokenizer)
            model.keras_model = architecture.build(
                vocab_size=tokenizer.vocab_size, seed=SEED
            )
            model.keras_model.set_weights(weights_from_records(snapshot["weights"]))
            text = model.generate(
                export["prompt"],
                temperature=0.0,
                max_new_tokens=export["max_new_tokens"],
                seed=SEED,
            )
            references["models"][name][str(snapshot["epoch"])] = text
            print(f"{name} epoch {snapshot['epoch']}: {text!r}")

    path = OUTPUT_DIR / "reference.json"
    path.write_text(json.dumps(references, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
