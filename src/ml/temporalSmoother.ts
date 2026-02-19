/**
 * Temporal Smoothing for Emotion Predictions
 * Applies exponential moving average to reduce jitter
 */

export class TemporalSmoother {
  private buffer: Float32Array | null = null;
  private alpha: number;

  /**
   * @param alpha - Smoothing factor (0-1). Higher = less smoothing, more responsive
   */
  constructor(alpha: number = 0.3) {
    if (alpha < 0 || alpha > 1) {
      throw new Error('Alpha must be between 0 and 1');
    }
    this.alpha = alpha;
  }

  /**
   * Apply exponential moving average smoothing
   * Formula: smoothed = alpha * current + (1 - alpha) * previous
   */
  smooth(current: Float32Array): Float32Array {
    if (!this.buffer) {
      // First frame - initialize buffer
      this.buffer = new Float32Array(current);
      return new Float32Array(current);
    }

    if (this.buffer.length !== current.length) {
      throw new Error('Input length mismatch');
    }

    const smoothed = new Float32Array(current.length);

    for (let i = 0; i < current.length; i++) {
      smoothed[i] = this.alpha * current[i] + (1 - this.alpha) * this.buffer[i];
      this.buffer[i] = smoothed[i];
    }

    return smoothed;
  }

  /**
   * Reset the internal buffer
   */
  reset(): void {
    this.buffer = null;
  }

  /**
   * Get current smoothed values
   */
  getBuffer(): Float32Array | null {
    return this.buffer ? new Float32Array(this.buffer) : null;
  }

  /**
   * Update alpha dynamically
   */
  setAlpha(alpha: number): void {
    if (alpha < 0 || alpha > 1) {
      throw new Error('Alpha must be between 0 and 1');
    }
    this.alpha = alpha;
  }
}
