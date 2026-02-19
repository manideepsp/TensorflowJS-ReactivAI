/**
 * One-time script to generate a placeholder emotion classification model
 * in TensorFlow.js GraphModel format.
 *
 * Architecture:
 *   Input  [1, 48, 48, 1]  (grayscale face crop)
 *   Conv2D 32 filters 3×3 → ReLU → MaxPool 2×2
 *   Conv2D 64 filters 3×3 → ReLU → MaxPool 2×2
 *   Flatten
 *   Dense 128 → ReLU
 *   Dense 7 → Softmax   (7 emotion classes)
 *
 * The weights are random – replace with a real trained model for production.
 *
 * Usage:
 *   node scripts/generate-model.mjs
 */

import * as tf from '@tensorflow/tfjs';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUTPUT_DIR = join(__dirname, '..', 'public', 'models', 'emotion_model');

async function generateModel() {
  // Force CPU backend for Node.js
  await tf.setBackend('cpu');
  await tf.ready();

  console.log('Building emotion classification model...');

  const model = tf.sequential();

  // Conv block 1
  model.add(tf.layers.conv2d({
    inputShape: [48, 48, 1],
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same',
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  // Conv block 2
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same',
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  // Dense head
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 7, activation: 'softmax' }));

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  model.summary();

  // Ensure output dir exists
  if (!existsSync(OUTPUT_DIR)) {
    mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  // ── Save as LayersModel first, then convert topology to GraphModel format ──
  // We save as a "layers model" because tf.sequential only supports that.
  // However our loader uses loadGraphModel, so we need to convert the format.

  // Step 1 – extract raw weight data
  const weightSpecs = [];
  const weightBuffers = [];

  for (const w of model.getWeights()) {
    const name = w.name;
    const data = await w.data();          // Float32Array
    const shape = w.shape;
    weightSpecs.push({ name, shape, dtype: 'float32' });
    weightBuffers.push(Buffer.from(data.buffer, data.byteOffset, data.byteLength));
  }

  // Concatenate all weights into one shard
  const totalBytes = weightBuffers.reduce((s, b) => s + b.byteLength, 0);
  const shard = Buffer.alloc(totalBytes);
  let offset = 0;
  for (const buf of weightBuffers) {
    buf.copy(shard, offset);
    offset += buf.byteLength;
  }

  // Step 2 – build a minimal TF.js Layers-model model.json
  const layersModelJSON = model.toJSON();

  const modelJSON = {
    format: 'layers-model',
    generatedBy: 'EdgePresence generate-model script',
    convertedBy: null,
    modelTopology: layersModelJSON,
    weightsManifest: [
      {
        paths: ['group1-shard1of1.bin'],
        weights: weightSpecs,
      },
    ],
  };

  // Step 3 – write files
  const modelJsonPath = join(OUTPUT_DIR, 'model.json');
  const shardPath = join(OUTPUT_DIR, 'group1-shard1of1.bin');

  writeFileSync(modelJsonPath, JSON.stringify(modelJSON));
  writeFileSync(shardPath, shard);

  console.log(`\n✓ model.json  → ${modelJsonPath} (${(JSON.stringify(modelJSON).length / 1024).toFixed(1)} KB)`);
  console.log(`✓ weights     → ${shardPath} (${(shard.byteLength / 1024).toFixed(1)} KB)`);
  console.log(`  Total params: ${model.countParams()}`);
  console.log('\nDone. Model files written to public/models/emotion_model/');

  model.dispose();
}

generateModel().catch(err => {
  console.error('Failed to generate model:', err);
  process.exit(1);
});
