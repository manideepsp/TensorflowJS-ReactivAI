#!/usr/bin/env python3
# pyright: reportAttributeAccessIssue=false
"""
Train emotion classifier in Python (TensorFlow/Keras) and export to TensorFlow.js format.

Usage:
  python scripts/train-emotion-model.py --data path/to/fer2013.csv
  python scripts/train-emotion-model.py --synthetic          # quick test with fake data

Output:
  - Keras model:  artifacts/emotion_model.keras
  - TFJS model:   public/models/emotion_model_py/model.json (+ shards)

Notes:
  - Keeps existing JS training pipeline intact (`scripts/train-emotion-model.mjs`).
  - App can load this model via URL: ?model=emotion_model_py
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Tuple

import sys

import numpy as np
import tensorflow as tf

# Use our own pure-Python converter (no tensorflowjs package needed)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_keras_to_tfjs import convert as convert_to_tfjs


NUM_CLASSES = 7
IMG_SIZE = 48


def generate_synthetic_data(n_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic 48x48 grayscale images with randomised class-correlated patterns.

    Each sample gets random position/size/intensity jitter so val set is not
    a carbon-copy of train set.
    """
    rng = np.random.default_rng(seed)
    LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    x = np.zeros((n_samples, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    yy_grid, xx_grid = np.ogrid[:IMG_SIZE, :IMG_SIZE]

    for i in range(n_samples):
        label = i % NUM_CLASSES
        y[i] = label

        # Random per-sample parameters
        bg_mean = rng.uniform(0.3, 0.55)
        bg_std = rng.uniform(0.08, 0.2)
        cx = IMG_SIZE // 2 + rng.integers(-5, 6)
        cy = IMG_SIZE // 2 + rng.integers(-5, 6)
        intensity = rng.uniform(0.35, 0.7)
        radius_scale = rng.uniform(0.7, 1.3)

        img = rng.normal(bg_mean, bg_std, (IMG_SIZE, IMG_SIZE)).astype(np.float32)

        if label == 0:    # angry: horizontal bars (brow-like)
            t1 = rng.integers(6, 16)
            t2 = rng.integers(28, 40)
            w = rng.integers(3, 6)
            img[t1:t1+w, 8:40] += intensity
            img[t2:t2+w, 8:40] += intensity
        elif label == 1:  # disgust: asymmetric blob
            r = int(12 * radius_scale)
            mask = ((xx_grid - cx)**2 + (yy_grid - cy)**2) < r**2
            img[mask] += intensity
            # add a smaller off-center spot
            ox = cx + rng.integers(-8, 9)
            mask2 = ((xx_grid - ox)**2 + (yy_grid - cy)**2) < (r//2)**2
            img[mask2] -= intensity * 0.5
        elif label == 2:  # fear: diagonal stripes
            spacing = rng.integers(4, 9)
            thickness = rng.integers(1, 3)
            for d in range(-48, 48, spacing):
                for j in range(IMG_SIZE):
                    for t in range(thickness):
                        k = j + d + t
                        if 0 <= k < IMG_SIZE:
                            img[j, k] += intensity * 0.7
        elif label == 3:  # happy: arc / smile
            r = int(12 * radius_scale)
            start_deg = rng.integers(190, 220)
            end_deg = rng.integers(320, 360)
            for angle in range(start_deg, end_deg):
                rad = np.radians(angle)
                ri, ci = int(cy + r*np.sin(rad)), int(cx + r*np.cos(rad))
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        rr, cc = ri+dr, ci+dc
                        if 0 <= rr < IMG_SIZE and 0 <= cc < IMG_SIZE:
                            img[rr, cc] += intensity * 0.4
        elif label == 4:  # sad: downward arcs + dark patches
            r = int(10 * radius_scale)
            for angle in range(10, 170):
                rad = np.radians(angle)
                ri, ci = int(cy + r*np.sin(rad)), int(cx + r*np.cos(rad))
                if 0 <= ri < IMG_SIZE and 0 <= ci < IMG_SIZE:
                    img[ri, ci] -= intensity * 0.6
            # dark patch below
            img[cy:cy+8, cx-6:cx+6] -= intensity * 0.3
        elif label == 5:  # surprise: open circle (mouth O)
            r_inner = int(8 * radius_scale)
            r_outer = r_inner + rng.integers(3, 6)
            dist = (xx_grid - cx)**2 + (yy_grid - cy)**2
            mask = (dist >= r_inner**2) & (dist <= r_outer**2)
            img[mask] += intensity
        elif label == 6:  # neutral: very uniform, minimal features
            img[:] = rng.normal(0.5, 0.05, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
            # faint horizontal line
            mid = IMG_SIZE // 2 + rng.integers(-3, 4)
            img[mid, 12:36] += rng.uniform(0.05, 0.15)

        # Random global perturbations
        if rng.random() < 0.5:
            img = np.fliplr(img)  # horizontal flip
        # brightness/contrast jitter
        img = img * rng.uniform(0.85, 1.15) + rng.uniform(-0.08, 0.08)
        # Gaussian noise
        img += rng.normal(0, rng.uniform(0.02, 0.08), img.shape).astype(np.float32)

        x[i, :, :, 0] = np.clip(img, 0, 1)

    perm = rng.permutation(n_samples)
    print(f"Generated {n_samples} synthetic samples ({', '.join(LABELS)})")
    return x[perm], y[perm]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to FER2013 CSV (emotion,pixels,Usage)")
    parser.add_argument("--data-dir", help="Path to FER2013 image directory (train/ and test/ subdirs with per-class folders)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for quick testing")
    parser.add_argument("--synthetic-n", type=int, default=7000, help="Number of synthetic samples (default 7000 = 1000/class)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for quick experiments (0 = all rows)")
    parser.add_argument(
        "--keras-out",
        default="artifacts/emotion_model.keras",
        help="Where to save Keras model",
    )
    parser.add_argument(
        "--tfjs-out",
        default="public/models/emotion_model",
        help="Directory for TFJS model.json + shards",
    )
    return parser.parse_args()


# Map directory names → label indices that match EMOTION_LABELS in the frontend:
#   ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# WITHOUT this explicit map, flow_from_directory uses alphabetical order
# which puts Neutral=4, Sad=5, Surprise=6 — wrong!
CLASS_TO_IDX = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6,
}


def load_fer2013_dir(data_dir: Path, limit: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load FER2013 from image directory structure (train/ and test/ with per-class subfolders).
    Returns (x_train, y_train, x_test, y_test) with images normalised to [0,1].
    """
    from PIL import Image

    def _load_split(split_dir: Path, max_samples: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for class_name, idx in CLASS_TO_IDX.items():
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                print(f"  Warning: missing class dir {class_dir}")
                continue
            files = sorted(class_dir.iterdir())
            count = 0
            for fp in files:
                try:
                    img = Image.open(fp).convert("L").resize((IMG_SIZE, IMG_SIZE))
                    arr = np.asarray(img, dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE, 1) / 255.0
                    xs.append(arr)
                    ys.append(idx)
                    count += 1
                except Exception:
                    continue  # skip corrupt files
                if max_samples and len(xs) >= max_samples:
                    break
            print(f"    {class_name}: {count}")
            if max_samples and len(xs) >= max_samples:
                break
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"train/ not found in {data_dir}")

    print("Loading training images...")
    x_train, y_train = _load_split(train_dir, limit)
    print(f"  Train: {len(x_train)} samples")

    if test_dir.is_dir():
        print("Loading test images...")
        x_test, y_test = _load_split(test_dir)
        print(f"  Test:  {len(x_test)} samples")
    else:
        x_test, y_test = np.array([]), np.array([])

    # Show class distribution
    for name, idx in CLASS_TO_IDX.items():
        n_tr = int((y_train == idx).sum())
        n_te = int((y_test == idx).sum()) if len(y_test) else 0
        print(f"  {name:10s}: train={n_tr:5d}  test={n_te:4d}")

    return x_train, y_train, x_test, y_test


def load_fer2013(csv_path: Path, limit: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            label = int(row["emotion"])
            pixels = np.fromstring(row["pixels"], dtype=np.float32, sep=" ")
            if pixels.size != IMG_SIZE * IMG_SIZE:
                continue
            img = (pixels / 255.0).reshape(IMG_SIZE, IMG_SIZE, 1)
            xs.append(img)
            ys.append(label)

            if limit and (i + 1) >= limit:
                break

    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return x, y


def stratified_split(
    x: np.ndarray, y: np.ndarray, val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for cls in range(NUM_CLASSES):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def build_model() -> tf.keras.Model:
    # No augmentation layers inside the model — TFJS doesn't support
    # RandomFlip / RandomRotation / RandomZoom.  Augmentation is applied
    # via ImageDataGenerator in the training loop instead.
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
            # Block 1
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(0.25),
            # Block 2
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(0.25),
            # Block 3
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(0.25),
            # Classifier
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    args = parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    has_separate_test = False

    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset dir not found: {data_dir}")
        print("\n=== IMAGE DIRECTORY MODE ===")
        x_train, y_train, x_val, y_val = load_fer2013_dir(data_dir, limit=args.limit)
        has_separate_test = len(x_val) > 0
        if not has_separate_test:
            x_train, y_train, x_val, y_val = stratified_split(x_train, y_train, args.val_ratio, args.seed)
    elif args.synthetic:
        print("\n=== SYNTHETIC DATA MODE (for testing pipeline) ===")
        x, y = generate_synthetic_data(args.synthetic_n, args.seed)
        x_train, y_train, x_val, y_val = stratified_split(x, y, args.val_ratio, args.seed)
    elif args.data:
        csv_path = Path(args.data)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        print("Loading dataset...")
        x, y = load_fer2013(csv_path, limit=args.limit)
        print(f"Loaded: {len(x)} samples")
        x_train, y_train, x_val, y_val = stratified_split(x, y, args.val_ratio, args.seed)
    else:
        raise SystemExit("Error: provide --data <path>, --data-dir <path>, or --synthetic")

    print(f"Train: {len(x_train)} | Val: {len(x_val)}")

    # Compute class weights to handle imbalanced classes (e.g. disgust has far fewer samples)
    # Cap the max weight to avoid destabilizing training
    from collections import Counter
    class_counts = Counter(y_train.tolist())
    total = len(y_train)
    class_weight = {
        cls: min(total / (NUM_CLASSES * count), 3.0)   # cap at 3x
        for cls, count in class_counts.items()
    }
    print("Class weights:", {k: f"{v:.2f}" for k, v in sorted(class_weight.items())})

    model = build_model()
    model.summary()

    # Training-time augmentation (not baked into the model graph)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=8,
        zoom_range=0.08,
        width_shift_range=0.06,
        height_shift_range=0.06,
    )
    datagen.fit(x_train)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    print("\nTraining...")
    model.fit(
        datagen.flow(x_train, y_train, batch_size=args.batch_size),
        steps_per_epoch=len(x_train) // args.batch_size,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    print("\nEvaluating...")
    loss, acc = model.evaluate(x_val, y_val, verbose=0)
    split_name = "Test" if has_separate_test else "Val"
    print(f"{split_name} loss: {loss:.4f} | {split_name} acc: {acc:.4f}")

    keras_out = Path(args.keras_out)
    keras_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(keras_out)
    print(f"Saved Keras model: {keras_out}")

    print("Converting to TensorFlow.js format...")
    convert_to_tfjs(keras_out, Path(args.tfjs_out))

    print("\nDone. Model is now the app default — just reload the page.")


if __name__ == "__main__":
    main()
