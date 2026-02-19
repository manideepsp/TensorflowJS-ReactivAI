/**
 * Face Detection Module
 * Uses TensorFlow.js Face Landmarks Detection model
 * Client-side only - no backend required
 *
 * IMPORTANT: We pass a canvas snapshot (not the raw video element) to
 * estimateFaces().  Passing the HTMLVideoElement directly fails when it
 * is rendered inside a React component tree because WebGL's texImage2D
 * cannot reliably read pixels from a video whose layout is controlled
 * by React (CSS transforms, positioned containers, etc.).
 */

import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import '@tensorflow/tfjs-backend-webgl';

let detector: faceLandmarksDetection.FaceLandmarksDetector | null = null;

// Reusable off-screen canvas to avoid allocation every frame
let _snapCanvas: HTMLCanvasElement | null = null;
let _snapCtx: CanvasRenderingContext2D | null = null;

export interface FaceDetectionResult {
  keypoints: Array<{ x: number; y: number; z?: number; name?: string }>;
  box: {
    xMin: number;
    yMin: number;
    xMax: number;
    yMax: number;
    width: number;
    height: number;
  };
  score: number;
}

/**
 * Initialize face landmarks detector.
 * Always disposes an existing detector before creating a new one so that
 * Vite HMR and React remounts get a fresh, working instance.
 */
export async function initializeFaceDetector(): Promise<void> {
  if (detector) {
    try { detector.dispose(); } catch { /* already disposed */ }
    detector = null;
  }

  const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
  detector = await faceLandmarksDetection.createDetector(model, {
    runtime: 'tfjs',
    maxFaces: 1,
    refineLandmarks: false,
  });
}

/**
 * Detect face in video frame.
 * Returns highest confidence face or null if no face detected.
 */
export async function detectFace(
  video: HTMLVideoElement
): Promise<FaceDetectionResult | null> {
  if (!detector) {
    throw new Error('Face detector not initialized. Call initializeFaceDetector() first.');
  }

  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return null;
  }

  // Snapshot video frame to an off-screen canvas
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!_snapCanvas || _snapCanvas.width !== w || _snapCanvas.height !== h) {
    _snapCanvas = document.createElement('canvas');
    _snapCanvas.width = w;
    _snapCanvas.height = h;
    _snapCtx = _snapCanvas.getContext('2d');
  }
  _snapCtx!.drawImage(video, 0, 0, w, h);

  const faces = await detector.estimateFaces(_snapCanvas!, {
    flipHorizontal: false,
  });

  if (faces.length === 0) {
    return null;
  }

  const face = faces[0];
  const faceScore = (face as { score?: number }).score ?? 1.0;

  // Compute bounding box from keypoints (always pixel coords)
  let bxMin = Infinity, byMin = Infinity, bxMax = -Infinity, byMax = -Infinity;
  for (const kp of face.keypoints) {
    if (kp.x < bxMin) bxMin = kp.x;
    if (kp.y < byMin) byMin = kp.y;
    if (kp.x > bxMax) bxMax = kp.x;
    if (kp.y > byMax) byMax = kp.y;
  }

  // Add ~10 % padding so the crop includes forehead / chin
  const pad = Math.max(bxMax - bxMin, byMax - byMin) * 0.1;
  bxMin = Math.max(0, bxMin - pad);
  byMin = Math.max(0, byMin - pad);
  bxMax = Math.min(w, bxMax + pad);
  byMax = Math.min(h, byMax + pad);

  return {
    keypoints: face.keypoints.map((kp) => ({
      x: kp.x, y: kp.y, z: kp.z, name: kp.name,
    })),
    box: {
      xMin: bxMin, yMin: byMin, xMax: bxMax, yMax: byMax,
      width: bxMax - bxMin, height: byMax - byMin,
    },
    score: faceScore,
  };
}

/**
 * Check if detector is initialized
 */
export function isDetectorInitialized(): boolean {
  return detector !== null;
}

/**
 * Dispose detector and free memory
 */
export function disposeFaceDetector(): void {
  if (detector) {
    detector.dispose();
    detector = null;
  }
}
