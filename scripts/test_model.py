"""Quick sanity check: does the Keras model always predict Sad?"""
import numpy as np
import tensorflow as tf

m = tf.keras.models.load_model("artifacts/emotion_model.keras")
labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

tests = {
    "Uniform 0.5": np.full((1, 48, 48, 1), 0.5, dtype=np.float32),
    "Random noise": np.random.default_rng(1).random((1, 48, 48, 1)).astype(np.float32),
    "All black": np.zeros((1, 48, 48, 1), dtype=np.float32),
    "All white": np.ones((1, 48, 48, 1), dtype=np.float32),
}

# Face-like gradient: bright center, dark edges
y, x = np.ogrid[:48, :48]
dist = ((x - 24) ** 2 + (y - 24) ** 2) / (24 ** 2)
tests["Face gradient"] = np.clip(1.0 - dist * 0.7, 0, 1).astype(np.float32).reshape(1, 48, 48, 1)

for name, inp in tests.items():
    p = m.predict(inp, verbose=0)[0]
    top_idx = int(np.argmax(p))
    print(f"\n{name}:")
    for i, (l, v) in enumerate(zip(labels, p)):
        bar = "#" * int(v * 40)
        marker = " <--" if i == top_idx else ""
        print(f"  {l:10s} {v:8.5f}  {bar}{marker}")
