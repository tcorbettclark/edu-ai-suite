"""Pack the browser checkpoints densely: float32 for char, int8 for word.

The float exports are convenient to inspect; these files are what a page loads. Char is
small enough to keep exact float32, and the word model uses one int8 scale per matrix
row, which keeps the greedy reference outputs identical while cutting the download.

Run it with ``BROWSER_LM_CORPUS=tiny_stories uv run python examples/browser_lm/compact_export.py``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from corpus import CORPUS_NAME

# Settings.
OUTPUT_DIR = Path(__file__).with_name("out") / CORPUS_NAME
PRECISION = {"char": "float32", "word": "int8"}


def float32_record(record: dict) -> dict:
    """One weight tensor as base64 little-endian float32."""
    values = np.asarray(record["data"], dtype="<f4")
    return {
        "name": record["name"],
        "shape": record["shape"],
        "dtype": "float32",
        "data_b64": base64.b64encode(values.tobytes()).decode("ascii"),
    }


def int8_record(record: dict) -> tuple[dict, float]:
    """One weight tensor as base64 int8; 2D weights get a scale per row."""
    values = np.asarray(record["data"], dtype=np.float32).reshape(record["shape"])
    if values.ndim == 1:
        peak = float(np.max(np.abs(values))) if values.size else 0.0
        scale = peak / 127.0 if peak else 1.0
        quantised = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
        error = float(np.max(np.abs(values - quantised.astype(np.float32) * scale)))
        packed = {
            "name": record["name"],
            "shape": record["shape"],
            "dtype": "int8",
            "scale": scale,
            "data_b64": base64.b64encode(quantised.tobytes()).decode("ascii"),
        }
        return packed, error

    rows, row_length = values.shape
    scales = np.max(np.abs(values), axis=1) / 127.0
    scales[scales == 0] = 1.0
    quantised = np.clip(
        np.rint(values / scales[:, None]), -127, 127
    ).astype(np.int8)
    restored = quantised.astype(np.float32) * scales[:, None]
    error = float(np.max(np.abs(values - restored)))
    packed = {
        "name": record["name"],
        "shape": record["shape"],
        "dtype": "int8",
        "rows": rows,
        "row_length": row_length,
        "scales": scales.tolist(),
        "data_b64": base64.b64encode(quantised.tobytes()).decode("ascii"),
    }
    return packed, error


def main() -> None:
    manifest = []
    for name, precision in PRECISION.items():
        source = OUTPUT_DIR / f"{name}.json"
        export = json.loads(source.read_text(encoding="utf-8"))
        biggest_error = 0.0
        checkpoints = []
        for snapshot in export["checkpoints"]:
            weights = []
            for record in snapshot["weights"]:
                if precision == "int8":
                    packed, error = int8_record(record)
                    biggest_error = max(biggest_error, error)
                else:
                    packed = float32_record(record)
                weights.append(packed)
            checkpoints.append({**snapshot, "weights": weights})
        export = {
            **export,
            "quantization": f"{precision}-base64",
            "checkpoints": checkpoints,
        }
        path = OUTPUT_DIR / f"{name}.compact.json"
        path.write_text(json.dumps(export, separators=(",", ":")), encoding="utf-8")
        detail = (
            f", max rounding error {biggest_error:.2e}" if precision == "int8" else ""
        )
        manifest.append(
            {
                "tokenization": name,
                "file": path.name,
                "precision": precision,
                "bytes": path.stat().st_size,
                "vocab_size": len(export["tokenizer"]["tokens"]),
                "corpus": export["corpus"]["title"],
                "checkpoints": [snapshot["epoch"] for snapshot in export["checkpoints"]],
            }
        )
        print(f"{name}: {precision}, {path.stat().st_size:,} bytes{detail}")

    manifest_path = OUTPUT_DIR / "browser_manifest.json"
    manifest_path.write_text(json.dumps({"models": manifest}, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
