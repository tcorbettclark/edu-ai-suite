"""The run manifest for one corpus: serialise what the browser assets contain."""

from __future__ import annotations

import json
from pathlib import Path


def corpus_record(name: str, config: dict) -> dict:
    """The corpus fields to write, without the local training path."""
    return {
        "name": name,
        "file": config["path"].name,
        "title": config["title"],
        "author": config["author"],
        "source": config["source"],
        "licence": config["licence"],
        "prompt": config["prompt"],
    }


def write_manifest(output_dir: Path, corpus_name: str, corpus_config: dict) -> Path:
    """Read the two exported models and write ``manifest.json`` beside them."""
    models = []
    for name in ("char", "word"):
        path = output_dir / f"{name}.json"
        export = json.loads(path.read_text(encoding="utf-8"))
        last = export["checkpoints"][-1]
        models.append(
            {
                "tokenization": name,
                "file": path.name,
                "bytes": path.stat().st_size,
                "checkpoints": [snapshot["epoch"] for snapshot in export["checkpoints"]],
                "vocab_size": len(export["tokenizer"]["tokens"]),
                "final_train_accuracy": last["train_accuracy"],
                "final_val_accuracy": last["val_accuracy"],
            }
        )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "corpus": corpus_record(corpus_name, corpus_config),
                "models": models,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path
