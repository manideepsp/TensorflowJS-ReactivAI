"""
Verify the TFJS-converted model by loading model.json + .bin in Python
and comparing predictions with the original Keras model.
"""
import json
import struct
from pathlib import Path
import numpy as np
import tensorflow as tf

LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# --- Load original Keras model ---
keras_model = tf.keras.models.load_model("artifacts/emotion_model.keras")

# --- Load converted TFJS weights from binary ---
tfjs_dir = Path("public/models/emotion_model")
with open(tfjs_dir / "model.json", "r") as f:
    model_json = json.load(f)

bin_data = (tfjs_dir / "group1-shard1of1.bin").read_bytes()

manifest_weights = model_json["weightsManifest"][0]["weights"]

# Parse weights from binary
offset = 0
tfjs_weights = {}
for wspec in manifest_weights:
    name = wspec["name"]
    shape = wspec["shape"]
    n_elements = int(np.prod(shape)) if shape else 1
    n_bytes = n_elements * 4  # float32
    arr = np.frombuffer(bin_data[offset:offset + n_bytes], dtype=np.float32).reshape(shape)
    tfjs_weights[name] = arr
    offset += n_bytes

print(f"Binary file: {len(bin_data)} bytes, parsed {offset} bytes ({len(tfjs_weights)} weights)")
print()

# --- Compare Keras weights vs TFJS-converted weights ---
print("=== Weight comparison (Keras vs TFJS) ===")
all_match = True
for w in keras_model.weights:
    keras_name = w.name.replace(":0", "")
    keras_arr = w.numpy().astype(np.float32)
    
    if keras_name in tfjs_weights:
        tfjs_arr = tfjs_weights[keras_name]
        if keras_arr.shape != tfjs_arr.shape:
            print(f"  SHAPE MISMATCH: {keras_name}: keras={keras_arr.shape} tfjs={tfjs_arr.shape}")
            all_match = False
        elif not np.allclose(keras_arr, tfjs_arr, atol=1e-7):
            diff = np.abs(keras_arr - tfjs_arr).max()
            print(f"  VALUE MISMATCH: {keras_name}: max_diff={diff:.2e}")
            all_match = False
        else:
            print(f"  OK: {keras_name} {keras_arr.shape}")
    else:
        print(f"  MISSING in TFJS: {keras_name}")
        all_match = False

if all_match:
    print("\n✓ All weights match perfectly!")
else:
    print("\n✗ Weight mismatches found!")

# --- Build a fresh Keras model with the TFJS weights and compare predictions ---
print("\n=== Prediction comparison ===")
# Reconstruct model from TFJS weights
test_model = tf.keras.models.clone_model(keras_model)

# Set weights from TFJS binary in the same order as the original
tfjs_weight_arrays = []
for w in test_model.weights:
    name = w.name.replace(":0", "")
    tfjs_weight_arrays.append(tfjs_weights[name])

test_model.set_weights(tfjs_weight_arrays)

# Test inputs
tests = {
    "Uniform 0.5": np.full((1, 48, 48, 1), 0.5, dtype=np.float32),
    "Random noise": np.random.default_rng(1).random((1, 48, 48, 1)).astype(np.float32),
    "All black": np.zeros((1, 48, 48, 1), dtype=np.float32),
    "All white": np.ones((1, 48, 48, 1), dtype=np.float32),
}

for name, inp in tests.items():
    p_keras = keras_model.predict(inp, verbose=0)[0]
    p_tfjs = test_model.predict(inp, verbose=0)[0]
    diff = np.abs(p_keras - p_tfjs).max()
    k_top = LABELS[np.argmax(p_keras)]
    t_top = LABELS[np.argmax(p_tfjs)]
    match = "✓" if k_top == t_top and diff < 1e-5 else "✗"
    print(f"  {match} {name}: keras={k_top}({p_keras.max():.4f}) tfjs={t_top}({p_tfjs.max():.4f}) max_diff={diff:.2e}")
