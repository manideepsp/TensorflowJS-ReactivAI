/**
 * Performance Monitoring
 * Tracks FPS, inference latency, and memory usage
 */

import * as tf from '@tensorflow/tfjs';

export interface PerformanceMetrics {
  fps: number;
  emotionLatency: number;
  faceDetectionLatency: number;
  audioLatency: number;
  numTensors: number;
  numBytes: number;
  timestamp: number;
}

export class PerformanceMonitor {
  private frameCount: number = 0;
  private lastFpsTime: number = 0;
  private currentFps: number = 0;

  private emotionLatencies: number[] = [];
  private faceDetectionLatencies: number[] = [];
  private audioLatencies: number[] = [];

  private readonly latencyBufferSize: number = 30;

  constructor() {
    this.lastFpsTime = performance.now();
  }

  /**
   * Update FPS counter
   * Call this once per frame
   */
  updateFps(): void {
    this.frameCount++;
    const now = performance.now();
    const elapsed = now - this.lastFpsTime;

    // Update FPS every second
    if (elapsed >= 1000) {
      this.currentFps = Math.round((this.frameCount * 1000) / elapsed);
      this.frameCount = 0;
      this.lastFpsTime = now;
    }
  }

  /**
   * Record emotion inference latency
   */
  recordEmotionLatency(latency: number): void {
    this.emotionLatencies.push(latency);
    if (this.emotionLatencies.length > this.latencyBufferSize) {
      this.emotionLatencies.shift();
    }
  }

  /**
   * Record face detection latency
   */
  recordFaceDetectionLatency(latency: number): void {
    this.faceDetectionLatencies.push(latency);
    if (this.faceDetectionLatencies.length > this.latencyBufferSize) {
      this.faceDetectionLatencies.shift();
    }
  }

  /**
   * Record audio processing latency
   */
  recordAudioLatency(latency: number): void {
    this.audioLatencies.push(latency);
    if (this.audioLatencies.length > this.latencyBufferSize) {
      this.audioLatencies.shift();
    }
  }

  /**
   * Get current performance metrics
   */
  getMetrics(): PerformanceMetrics {
    const memory = tf.memory();

    return {
      fps: this.currentFps,
      emotionLatency: this.getAverageLatency(this.emotionLatencies),
      faceDetectionLatency: this.getAverageLatency(this.faceDetectionLatencies),
      audioLatency: this.getAverageLatency(this.audioLatencies),
      numTensors: memory.numTensors,
      numBytes: memory.numBytes,
      timestamp: Date.now(),
    };
  }

  /**
   * Reset all metrics
   */
  reset(): void {
    this.frameCount = 0;
    this.lastFpsTime = performance.now();
    this.currentFps = 0;
    this.emotionLatencies = [];
    this.faceDetectionLatencies = [];
    this.audioLatencies = [];
  }

  /**
   * Check for memory growth (tensor leak detection)
   */
  checkMemoryGrowth(baselineTensors: number): boolean {
    const current = tf.memory().numTensors;
    const growth = current - baselineTensors;
    
    if (growth > 10) {
      console.warn(`Possible memory leak detected: ${growth} tensor(s) growth from baseline`);
      return true;
    }
    
    return false;
  }

  /**
   * Get average latency from buffer
   */
  private getAverageLatency(buffer: number[]): number {
    if (buffer.length === 0) {
      return 0;
    }
    const sum = buffer.reduce((acc, val) => acc + val, 0);
    return Math.round(sum / buffer.length);
  }
}
