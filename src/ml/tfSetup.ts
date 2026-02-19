/**
 * TensorFlow.js Backend Setup
 * Ensures WebGL backend is initialized before any model operations
 */

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';

let initialized = false;

/**
 * Initialize TensorFlow.js with WebGL backend
 * Must be called before any model loading or inference
 */
export async function initializeTensorFlow(): Promise<void> {
  // Always re-initialize — the backend may have been reset by HMR / remount
  await tf.setBackend('webgl');
  await tf.ready();
  
  initialized = true;
  
  console.log('TensorFlow.js initialized');
  console.log('Backend:', tf.getBackend());
  console.log('WebGL version:', (tf.backend() as any).canvas?.getContext('webgl2') ? 'WebGL 2' : 'WebGL 1');
}

/**
 * Get current memory stats
 */
export function getMemoryInfo(): {
  numTensors: number;
  numBytes: number;
  numDataBuffers: number;
} {
  return tf.memory();
}

/**
 * Check if TensorFlow.js is initialized
 */
export function isInitialized(): boolean {
  return initialized;
}
