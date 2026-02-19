/**
 * Emotion Classification Model
 * Loads and runs inference on static TensorFlow.js model
 * STRICT CLIENT-SIDE ONLY - No backend, no API calls
 */

// Use the unified TFJS bundle to avoid multiple runtime instances
import * as tf from '@tensorflow/tfjs';

let model: tf.LayersModel | null = null;

export const EMOTION_LABELS = [
  'Angry',
  'Disgust',
  'Fear',
  'Happy',
  'Sad',
  'Surprise',
  'Neutral'
];

function resolveEmotionModelDir(): string {
  const fallback = 'emotion_model';

  if (typeof window === 'undefined') return fallback;

  // Allow override via ?model=<dir> query param for A/B testing
  const value = new URLSearchParams(window.location.search).get('model');
  if (!value) return fallback;

  const safe = value.replace(/[^a-zA-Z0-9_-]/g, '');
  return safe || fallback;
}

/**
 * Load emotion classification model from static JSON files
 * Uses BASE_URL for GitHub Pages compatibility
 */
export async function loadEmotionModel(): Promise<void> {
  // Always dispose old model – it may be a zombie after React remount / HMR
  if (model) {
    try { model.dispose(); } catch { /* already disposed */ }
    model = null;
  }

  // Use import.meta.env.BASE_URL for GitHub Pages subpath compatibility.
  // Model can be switched via URL query:
  //   ?model=emotion_model      (default existing model)
  //   ?model=emotion_model_py   (Python-trained model)
  const modelDir = resolveEmotionModelDir();
  const modelUrl = `${import.meta.env.BASE_URL}models/${modelDir}/model.json`;

  console.log('Loading emotion model from:', modelUrl);

  model = await tf.loadLayersModel(modelUrl);

  console.log('Emotion model loaded successfully');
}

/**
 * Predict emotion from preprocessed face image tensor
 * Input shape must be [1, 48, 48, 1]
 * Returns probability distribution over emotion classes
 */
export async function predictEmotion(input: tf.Tensor4D): Promise<Float32Array> {
  if (!model) {
    throw new Error('Emotion model not loaded. Call loadEmotionModel() first.');
  }

  // Validate input shape
  const inputShape = input.shape;
  if (inputShape[0] !== 1 || inputShape[1] !== 48 || inputShape[2] !== 48 || inputShape[3] !== 1) {
    throw new Error(`Invalid input shape: expected [1, 48, 48, 1], got [${inputShape.join(', ')}]`);
  }

  const result = tf.tidy(() => {
    const prediction = model!.predict(input) as tf.Tensor;
    return prediction.clone();
  });

  const probabilities = await result.data() as Float32Array;

  // Dispose the result tensor
  result.dispose();

  return probabilities;
}

/**
 * Get the top emotion prediction
 */
export function getTopEmotion(probabilities: Float32Array): {
  label: string;
  confidence: number;
  index: number;
} {
  let maxIndex = 0;
  let maxProb = probabilities[0];

  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      maxIndex = i;
    }
  }

  return {
    label: EMOTION_LABELS[maxIndex] || 'Unknown',
    confidence: maxProb,
    index: maxIndex
  };
}

/**
 * Check if model is loaded
 */
export function isModelLoaded(): boolean {
  return model !== null;
}

/**
 * Dispose model and free memory
 */
export function disposeModel(): void {
  if (model) {
    model.dispose();
    model = null;
  }
}
