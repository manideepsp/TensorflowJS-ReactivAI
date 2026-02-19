#!/usr/bin/env python3
"""
Convert a Keras (.keras / .h5) model to TensorFlow.js Layers format.

Pure-Python implementation — no `tensorflowjs` package required.
Produces the same `model.json` + `group1-shard1of1.bin` that TFJS `loadLayersModel` expects.

Usage:
  python scripts/convert-keras-to-tfjs.py --keras artifacts/emotion_model.keras --out public/models/emotion_model_py
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_layer_config(layer: dict) -> dict:
    """Strip Keras 3 fields that TFJS loadLayersModel doesn't understand."""
    cleaned = {}
    # Fields to remove — TFJS only understands Keras 2 format
    SKIP = {"module", "registered_name", "build_config"}
    for k, v in layer.items():
        if k in SKIP:
            continue
        if k == "config" and isinstance(v, dict):
            # Recurse into nested layer configs (e.g. kernel_initializer)
            cleaned[k] = {
                ck: _clean_layer_config(cv) if isinstance(cv, dict) and "class_name" in cv else cv
                for ck, cv in v.items()
            }
        elif k == "layers" and isinstance(v, list):
            cleaned[k] = [_clean_layer_config(l) for l in v]
        else:
            cleaned[k] = v
    return cleaned


def _keras_config_to_tfjs(model: tf.keras.Model) -> dict:
    """Convert the Keras model JSON config into TFJS-compatible modelTopology."""
    cfg = json.loads(model.to_json())

    # Clean all layers recursively
    if "config" in cfg and "layers" in cfg["config"]:
        cfg["config"]["layers"] = [
            _clean_layer_config(l) for l in cfg["config"]["layers"]
        ]

    cfg["keras_version"] = f"keras {tf.__version__}"
    cfg["backend"] = "tensorflow"
    return cfg


def _collect_weights(model: tf.keras.Model) -> tuple[list[dict], bytes]:
    """Return (weights_manifest_entries, concatenated_binary_data)."""
    entries: list[dict] = []
    buffers: list[bytes] = []

    for w in model.weights:
        arr: np.ndarray = w.numpy().astype(np.float32)
        # Keras names end with `:0` (e.g. "conv2d/kernel:0") but TFJS
        # loadLayersModel expects names without the suffix.
        name = w.name
        if name.endswith(":0"):
            name = name[:-2]
        entries.append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": "float32",
        })
        buffers.append(arr.tobytes())

    return entries, b"".join(buffers)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(keras_path: Path, out_dir: Path) -> None:
    print(f"Loading {keras_path} ...")
    model = tf.keras.models.load_model(keras_path)

    topology = _keras_config_to_tfjs(model)
    weight_entries, weight_bytes = _collect_weights(model)

    shard_name = "group1-shard1of1.bin"

    model_json = {
        "format": "layers-model",
        "generatedBy": f"keras {tf.__version__} / custom converter",
        "convertedBy": "scripts/convert-keras-to-tfjs.py",
        "modelTopology": topology,
        "weightsManifest": [
            {
                "paths": [shard_name],
                "weights": weight_entries,
            }
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "model.json"
    json_path.write_text(json.dumps(model_json, indent=2), encoding="utf-8")

    bin_path = out_dir / shard_name
    bin_path.write_bytes(weight_bytes)

    size_mb = len(weight_bytes) / (1024 * 1024)
    print(f"Wrote {json_path}  ({len(json.dumps(model_json)):,} bytes)")
    print(f"Wrote {bin_path}  ({size_mb:.2f} MB)")
    print("Done ✓")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Keras model → TFJS Layers format")
    p.add_argument("--keras", required=True, help="Path to .keras, .h5, or SavedModel dir")
    p.add_argument("--out", default="public/models/emotion_model_py", help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    keras_path = Path(args.keras)
    if not keras_path.exists():
        raise FileNotFoundError(f"Not found: {keras_path}")
    convert(keras_path, Path(args.out))


if __name__ == "__main__":
    main()
