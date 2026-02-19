/**
 * Train Emotion Classification Model
 *
 * Generates a TensorFlow.js model that can distinguish 7 emotions from
 * 48×48 grayscale face crops.  Because we cannot bundle the full FER2013
 * dataset, this script synthesises training images with facial-feature-like
 * patterns that correlate with each emotion (mouth curvature, eye openness,
 * brow angle, etc.) and trains the same Conv2D architecture used at runtime.
 *
 * Run:  node scripts/train-emotion-model.mjs
 *
 * Output: public/models/emotion_model/model.json
 *         public/models/emotion_model/group1-shard1of1.bin
 */

import * as tf from '@tensorflow/tfjs';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { performance } from 'perf_hooks';

// Register CPU backend for Node.js (no native addon needed)
await tf.setBackend('cpu');
await tf.ready();
console.log('Using backend:', tf.getBackend());

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(__dirname, '..', 'public', 'models', 'emotion_model');

const EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];
const IMG_SIZE = 48;
const CHANNELS = 1;
const NUM_CLASSES = 7;

// ── Synthetic face image generator ─────────────────────────────────────
// Each emotion has a characteristic pattern:
//   - mouth shape (curvature, openness)
//   - eye openness
//   - brow position / angle
//   - overall brightness / contrast
// We draw simplified face components with noise & variation.

function drawEllipse(data, cx, cy, rx, ry, value, w, h) {
  for (let y = Math.max(0, Math.floor(cy - ry)); y <= Math.min(h - 1, Math.ceil(cy + ry)); y++) {
    for (let x = Math.max(0, Math.floor(cx - rx)); x <= Math.min(w - 1, Math.ceil(cx + rx)); x++) {
      const dx = (x - cx) / rx;
      const dy = (y - cy) / ry;
      if (dx * dx + dy * dy <= 1) {
        data[y * w + x] = Math.min(1, Math.max(0, data[y * w + x] + value));
      }
    }
  }
}

function drawLine(data, x1, y1, x2, y2, value, w, h, thickness = 1) {
  const steps = Math.max(Math.abs(x2 - x1), Math.abs(y2 - y1), 1) * 2;
  for (let s = 0; s <= steps; s++) {
    const t = s / steps;
    const x = Math.round(x1 + (x2 - x1) * t);
    const y = Math.round(y1 + (y2 - y1) * t);
    for (let dy = -thickness; dy <= thickness; dy++) {
      for (let dx = -thickness; dx <= thickness; dx++) {
        const px = x + dx;
        const py = y + dy;
        if (px >= 0 && px < w && py >= 0 && py < h) {
          data[py * w + px] = Math.min(1, Math.max(0, data[py * w + px] + value));
        }
      }
    }
  }
}

function generateFace(emotionIdx) {
  const data = new Float32Array(IMG_SIZE * IMG_SIZE);
  const r = () => (Math.random() - 0.5);  // ±0.5
  const rr = () => Math.random();

  // Base face — oval with slight noise
  const faceCx = 24 + r() * 3;
  const faceCy = 24 + r() * 2;
  const faceRx = 16 + r() * 2;
  const faceRy = 19 + r() * 2;
  const baseBright = 0.4 + rr() * 0.2;
  drawEllipse(data, faceCx, faceCy, faceRx, faceRy, baseBright, IMG_SIZE, IMG_SIZE);

  // Eyes (always present)
  const eyeY = faceCy - 4 + r() * 2;
  const eyeSpacing = 7 + r() * 1;
  const leftEyeX = faceCx - eyeSpacing / 2 - 2;
  const rightEyeX = faceCx + eyeSpacing / 2 + 2;

  // Emotion-specific features
  switch (emotionIdx) {
    case 0: // Angry — furrowed brows, tight mouth, high contrast
      // Thick dark brows angled down toward center
      drawLine(data, leftEyeX - 4, eyeY - 5 + r(), leftEyeX + 3, eyeY - 3 + r(), 0.6, IMG_SIZE, IMG_SIZE, 1);
      drawLine(data, rightEyeX - 3, eyeY - 3 + r(), rightEyeX + 4, eyeY - 5 + r(), 0.6, IMG_SIZE, IMG_SIZE, 1);
      // Narrow eyes
      drawEllipse(data, leftEyeX, eyeY, 3, 1.5, 0.7, IMG_SIZE, IMG_SIZE);
      drawEllipse(data, rightEyeX, eyeY, 3, 1.5, 0.7, IMG_SIZE, IMG_SIZE);
      // Tight straight mouth
      drawLine(data, faceCx - 6, faceCy + 8, faceCx + 6, faceCy + 8, 0.6, IMG_SIZE, IMG_SIZE, 1);
      // Wrinkle between brows
      drawLine(data, faceCx - 1, eyeY - 6, faceCx + 1, eyeY - 2, 0.4, IMG_SIZE, IMG_SIZE);
      break;

    case 1: // Disgust — raised upper lip, wrinkled nose, asymmetric
      drawEllipse(data, leftEyeX, eyeY, 3, 2, 0.6, IMG_SIZE, IMG_SIZE);
      drawEllipse(data, rightEyeX, eyeY, 2.5, 1.5, 0.6, IMG_SIZE, IMG_SIZE);
      // Wrinkled nose
      drawLine(data, faceCx - 2, faceCy + 1, faceCx, faceCy + 3, 0.5, IMG_SIZE, IMG_SIZE);
      drawLine(data, faceCx + 2, faceCy + 1, faceCx, faceCy + 3, 0.5, IMG_SIZE, IMG_SIZE);
      // Raised upper lip, asymmetric mouth
      drawLine(data, faceCx - 5, faceCy + 9, faceCx + 4, faceCy + 7, 0.5, IMG_SIZE, IMG_SIZE, 1);
      break;

    case 2: // Fear — wide eyes, raised brows, open mouth
      // High raised brows
      drawLine(data, leftEyeX - 3, eyeY - 7, leftEyeX + 3, eyeY - 7, 0.5, IMG_SIZE, IMG_SIZE);
      drawLine(data, rightEyeX - 3, eyeY - 7, rightEyeX + 3, eyeY - 7, 0.5, IMG_SIZE, IMG_SIZE);
      // Wide open eyes
      drawEllipse(data, leftEyeX, eyeY, 3.5, 3, 0.8, IMG_SIZE, IMG_SIZE);
      drawEllipse(data, rightEyeX, eyeY, 3.5, 3, 0.8, IMG_SIZE, IMG_SIZE);
      // Open mouth (oval)
      drawEllipse(data, faceCx, faceCy + 9, 4, 3, 0.6, IMG_SIZE, IMG_SIZE);
      break;

    case 3: // Happy — smile, crinkled eyes, raised cheeks
      // Slightly squinted eyes (smile crinkle)
      drawEllipse(data, leftEyeX, eyeY, 3, 2, 0.6, IMG_SIZE, IMG_SIZE);
      drawEllipse(data, rightEyeX, eyeY, 3, 2, 0.6, IMG_SIZE, IMG_SIZE);
      // Smile — curved up line
      for (let x = -7; x <= 7; x++) {
        const y = faceCy + 8 - (7 * 7 - x * x) / 20;
        const px = Math.round(faceCx + x);
        const py = Math.round(y);
        if (px >= 0 && px < IMG_SIZE && py >= 0 && py < IMG_SIZE) {
          data[py * IMG_SIZE + px] = Math.min(1, data[py * IMG_SIZE + px] + 0.7);
          if (py + 1 < IMG_SIZE) data[(py + 1) * IMG_SIZE + px] = Math.min(1, data[(py + 1) * IMG_SIZE + px] + 0.4);
        }
      }
      // Raised cheeks
      drawEllipse(data, leftEyeX - 1, faceCy + 3, 3, 2, 0.2, IMG_SIZE, IMG_SIZE);
      drawEllipse(data, rightEyeX + 1, faceCy + 3, 3, 2, 0.2, IMG_SIZE, IMG_SIZE);
      break;

    case 4: // Sad — drooping mouth, lowered brows, teardrop
      // Lowered inner brows
      drawLine(data, leftEyeX - 3, eyeY - 4, leftEyeX + 2, eyeY - 5, 0.5, IMG_SIZE, IMG_SIZE);
      drawLine(data, rightEyeX - 2, eyeY - 5, rightEyeX + 3, eyeY - 4, 0.5, IMG_SIZE, IMG_SIZE);
      // Normal eyes
      drawEllipse(data, leftEyeX, eyeY, 3, 2, 0.6, IMG_SIZE, IMG_SIZE);
      drawEllipse(data, rightEyeX, eyeY, 3, 2, 0.6, IMG_SIZE, IMG_SIZE);
      // Frown — curved down line
      for (let x = -6; x <= 6; x++) {
        const y = faceCy + 8 + (6 * 6 - x * x) / 18;
        const px = Math.round(faceCx + x);
        const py = Math.round(y);
        if (px >= 0 && px < IMG_SIZE && py >= 0 && py < IMG_SIZE) {
          data[py * IMG_SIZE + px] = Math.min(1, data[py * IMG_SIZE + px] + 0.6);
          if (py + 1 < IMG_SIZE) data[(py + 1) * IMG_SIZE + px] = Math.min(1, data[(py + 1) * IMG_SIZE + px] + 0.3);
        }
      }
      break;

    case 5: // Surprise — very wide eyes, raised brows, O-mouth
      // Very high brows (arched)
      for (let x = -3; x <= 3; x++) {
        const by = eyeY - 7 - (9 - x * x) / 6;
        drawLine(data, Math.round(leftEyeX + x), Math.round(by), Math.round(leftEyeX + x + 1), Math.round(by), 0.6, IMG_SIZE, IMG_SIZE);
        drawLine(data, Math.round(rightEyeX + x), Math.round(by), Math.round(rightEyeX + x + 1), Math.round(by), 0.6, IMG_SIZE, IMG_SIZE);
      }
      // Very wide eyes
      drawEllipse(data, leftEyeX, eyeY, 4, 3.5, 0.8, IMG_SIZE, IMG_SIZE);
      drawEllipse(data, rightEyeX, eyeY, 4, 3.5, 0.8, IMG_SIZE, IMG_SIZE);
      // O-shaped mouth
      drawEllipse(data, faceCx, faceCy + 9, 3.5, 4, 0.7, IMG_SIZE, IMG_SIZE);
      // Hollow center of mouth
      drawEllipse(data, faceCx, faceCy + 9, 2, 2.5, -0.3, IMG_SIZE, IMG_SIZE);
      break;

    case 6: // Neutral — relaxed features, straight mouth
      // Normal brows
      drawLine(data, leftEyeX - 3, eyeY - 4, leftEyeX + 3, eyeY - 4, 0.4, IMG_SIZE, IMG_SIZE);
      drawLine(data, rightEyeX - 3, eyeY - 4, rightEyeX + 3, eyeY - 4, 0.4, IMG_SIZE, IMG_SIZE);
      // Normal eyes
      drawEllipse(data, leftEyeX, eyeY, 3, 2, 0.6, IMG_SIZE, IMG_SIZE);
      drawEllipse(data, rightEyeX, eyeY, 3, 2, 0.6, IMG_SIZE, IMG_SIZE);
      // Straight mouth
      drawLine(data, faceCx - 5, faceCy + 8, faceCx + 5, faceCy + 8, 0.5, IMG_SIZE, IMG_SIZE);
      break;
  }

  // Add random noise
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.min(1, Math.max(0, data[i] + (Math.random() - 0.5) * 0.15));
  }

  return data;
}

// ── Build model ────────────────────────────────────────────────────────
// Lightweight architecture optimised for fast CPU training on 48×48 inputs

function buildModel() {
  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    inputShape: [IMG_SIZE, IMG_SIZE, CHANNELS],
    filters: 16, kernelSize: 3, padding: 'same', activation: 'relu',
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));  // → 24×24

  model.add(tf.layers.conv2d({
    filters: 32, kernelSize: 3, padding: 'same', activation: 'relu',
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));  // → 12×12

  model.add(tf.layers.conv2d({
    filters: 64, kernelSize: 3, padding: 'same', activation: 'relu',
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));  // → 6×6

  model.add(tf.layers.flatten());  // 6×6×64 = 2304
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

// ── Generate dataset ───────────────────────────────────────────────────

function generateDataset(numPerClass) {
  const total = numPerClass * NUM_CLASSES;
  const xs = new Float32Array(total * IMG_SIZE * IMG_SIZE);
  const ys = new Float32Array(total * NUM_CLASSES);

  let idx = 0;
  for (let cls = 0; cls < NUM_CLASSES; cls++) {
    for (let i = 0; i < numPerClass; i++) {
      const face = generateFace(cls);
      xs.set(face, idx * IMG_SIZE * IMG_SIZE);
      ys[idx * NUM_CLASSES + cls] = 1;
      idx++;
    }
  }

  // Shuffle
  const indices = Array.from({ length: total }, (_, i) => i);
  for (let i = total - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const shuffledXs = new Float32Array(xs.length);
  const shuffledYs = new Float32Array(ys.length);
  for (let i = 0; i < total; i++) {
    const src = indices[i];
    shuffledXs.set(xs.subarray(src * IMG_SIZE * IMG_SIZE, (src + 1) * IMG_SIZE * IMG_SIZE), i * IMG_SIZE * IMG_SIZE);
    shuffledYs.set(ys.subarray(src * NUM_CLASSES, (src + 1) * NUM_CLASSES), i * NUM_CLASSES);
  }

  return {
    x: tf.tensor4d(shuffledXs, [total, IMG_SIZE, IMG_SIZE, CHANNELS]),
    y: tf.tensor2d(shuffledYs, [total, NUM_CLASSES]),
  };
}

// ── Progress bar helper ────────────────────────────────────────────────

function progressBar(current, total, width = 30, extra = '') {
  const pct = current / total;
  const filled = Math.round(width * pct);
  const empty = width - filled;
  const bar = '█'.repeat(filled) + '░'.repeat(empty);
  const pctStr = (pct * 100).toFixed(1).padStart(5);
  process.stdout.write(`\r  [${bar}] ${pctStr}% ${current}/${total} ${extra}`);
}

function elapsedSince(startMs) {
  const sec = ((performance.now() - startMs) / 1000);
  if (sec < 60) return `${sec.toFixed(1)}s`;
  return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`;
}

function getArg(name, fallback) {
  const prefix = `--${name}=`;
  const found = process.argv.find((a) => a.startsWith(prefix));
  if (!found) return fallback;
  const raw = Number(found.slice(prefix.length));
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : fallback;
}

function metric(logs, ...keys) {
  for (const k of keys) {
    const v = logs?.[k];
    if (typeof v === 'number' && Number.isFinite(v)) return v;
  }
  return NaN;
}

// ── Main ───────────────────────────────────────────────────────────────

async function main() {
  const t0 = performance.now();

  console.log('Building model...');
  const model = buildModel();
  model.summary();

  // Faster defaults for pure JS CPU backend; override via CLI args.
  // Example: node scripts/train-emotion-model.mjs --train=300 --val=60 --epochs=15 --batch=64
  const TRAIN_PER_CLASS = getArg('train', 120);
  const VAL_PER_CLASS = getArg('val', 30);
  const EPOCHS = getArg('epochs', 8);
  const BATCH_SIZE = getArg('batch', 128);

  console.log(`\nGenerating training data (${TRAIN_PER_CLASS} per class = ${TRAIN_PER_CLASS * NUM_CLASSES} samples)...`);
  const train = generateDataset(TRAIN_PER_CLASS);
  console.log(`Generating validation data (${VAL_PER_CLASS} per class = ${VAL_PER_CLASS * NUM_CLASSES} samples)...`);
  const val = generateDataset(VAL_PER_CLASS);

  const totalTrainSamples = TRAIN_PER_CLASS * NUM_CLASSES;
  const batchesPerEpoch = Math.ceil(totalTrainSamples / BATCH_SIZE);

  console.log(`\nTraining: ${EPOCHS} epochs, batch_size=${BATCH_SIZE}, ${batchesPerEpoch} batches/epoch`);
  console.log(`Backend: ${tf.getBackend()} (CPU — expect ~1-3 minutes total)\n`);

  let epochStart = performance.now();
  let currentBatch = 0;
  let avgBatchMs = 0;
  let lastBatchTick = performance.now();

  await model.fit(train.x, train.y, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    validationData: [val.x, val.y],
    callbacks: {
      onEpochBegin: (epoch) => {
        epochStart = performance.now();
        currentBatch = 0;
        avgBatchMs = 0;
        lastBatchTick = performance.now();
      },
      onBatchEnd: (batch, logs) => {
        currentBatch = batch + 1;
        const now = performance.now();
        const batchMs = now - lastBatchTick;
        lastBatchTick = now;
        avgBatchMs = avgBatchMs === 0 ? batchMs : (avgBatchMs * 0.9 + batchMs * 0.1);
        const remainBatches = Math.max(0, batchesPerEpoch - currentBatch);
        const etaSec = (remainBatches * avgBatchMs) / 1000;
        const etaStr = etaSec < 60
          ? `${etaSec.toFixed(0)}s`
          : `${Math.floor(etaSec / 60)}m ${Math.round(etaSec % 60)}s`;
        const loss = metric(logs, 'loss');
        const acc = metric(logs, 'acc', 'accuracy');
        const batchInfo = `loss=${Number.isFinite(loss) ? loss.toFixed(4) : 'n/a'} acc=${Number.isFinite(acc) ? acc.toFixed(3) : 'n/a'} eta=${etaStr}`;
        progressBar(currentBatch, batchesPerEpoch, 25, batchInfo);
      },
      onEpochEnd: (epoch, logs) => {
        const elapsed = elapsedSince(epochStart);
        process.stdout.write('\r' + ' '.repeat(90) + '\r'); // Clear progress line
        const loss = metric(logs, 'loss');
        const acc = metric(logs, 'acc', 'accuracy');
        const valLoss = metric(logs, 'val_loss');
        const valAcc = metric(logs, 'val_acc', 'val_accuracy');
        console.log(
          `  Epoch ${String(epoch + 1).padStart(2)}/${EPOCHS}  ` +
          `loss=${Number.isFinite(loss) ? loss.toFixed(4) : 'n/a'}  ` +
          `acc=${Number.isFinite(acc) ? acc.toFixed(4) : 'n/a'}  ` +
          `val_loss=${Number.isFinite(valLoss) ? valLoss.toFixed(4) : 'n/a'}  ` +
          `val_acc=${Number.isFinite(valAcc) ? valAcc.toFixed(4) : 'n/a'}  ` +
          `(${elapsed})`
        );
      },
    },
  });

  console.log(`\nTraining complete in ${elapsedSince(t0)}`);

  // Clean up training tensors
  train.x.dispose();
  train.y.dispose();
  val.x.dispose();
  val.y.dispose();

  // Verify predictions
  console.log('\nVerifying predictions on one sample per class:');
  let correct = 0;
  for (let cls = 0; cls < NUM_CLASSES; cls++) {
    const face = generateFace(cls);
    const input = tf.tensor4d(face, [1, IMG_SIZE, IMG_SIZE, CHANNELS]);
    const pred = model.predict(input);
    const probs = await pred.data();
    const maxIdx = probs.indexOf(Math.max(...probs));
    const match = maxIdx === cls ? '✓' : '✗';
    if (maxIdx === cls) correct++;
    console.log(`  ${match} ${EMOTIONS[cls].padEnd(10)} → predicted: ${EMOTIONS[maxIdx].padEnd(10)} (${(probs[maxIdx] * 100).toFixed(1)}%)`);
    input.dispose();
    pred.dispose();
  }
  console.log(`  Score: ${correct}/${NUM_CLASSES} correct`);

  // Save using manual handler (file:// requires tfjs-node)
  if (!existsSync(OUT_DIR)) mkdirSync(OUT_DIR, { recursive: true });
  console.log(`\nSaving model to ${OUT_DIR}...`);

  const saveResult = await model.save(tf.io.withSaveHandler(async (artifacts) => {
    // Write model.json
    const modelJSON = {
      format: artifacts.format,
      generatedBy: artifacts.generatedBy,
      convertedBy: artifacts.convertedBy,
      modelTopology: artifacts.modelTopology,
      weightsManifest: [{
        paths: ['group1-shard1of1.bin'],
        weights: artifacts.weightSpecs,
      }],
    };
    writeFileSync(join(OUT_DIR, 'model.json'), JSON.stringify(modelJSON, null, 2));

    // Write weights binary
    const weightData = artifacts.weightData;
    let buffer;
    if (weightData instanceof ArrayBuffer) {
      buffer = Buffer.from(weightData);
    } else if (Array.isArray(weightData)) {
      const totalLen = weightData.reduce((s, ab) => s + ab.byteLength, 0);
      buffer = Buffer.alloc(totalLen);
      let offset = 0;
      for (const ab of weightData) {
        Buffer.from(ab).copy(buffer, offset);
        offset += ab.byteLength;
      }
    } else {
      buffer = Buffer.from(weightData);
    }
    writeFileSync(join(OUT_DIR, 'group1-shard1of1.bin'), buffer);

    return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
  }));

  console.log('✅ Done! Model saved.');
  console.log(`Total time: ${elapsedSince(t0)}`);
}

main().catch(console.error);
