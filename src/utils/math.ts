/**
 * Mathematical utility functions
 */

/**
 * Clamp a value between min and max
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Linear interpolation
 */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Map value from one range to another
 */
export function mapRange(
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number
): number {
  return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
}

/**
 * Calculate mean of an array
 */
export function mean(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  const sum = values.reduce((acc, val) => acc + val, 0);
  return sum / values.length;
}

/**
 * Calculate standard deviation
 */
export function standardDeviation(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  const avg = mean(values);
  const squareDiffs = values.map(value => Math.pow(value - avg, 2));
  return Math.sqrt(mean(squareDiffs));
}

/**
 * Calculate root mean square (RMS)
 */
export function rms(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  const sumSquares = values.reduce((acc, val) => acc + val * val, 0);
  return Math.sqrt(sumSquares / values.length);
}

/**
 * Exponential moving average
 */
export function exponentialMovingAverage(
  current: number,
  previous: number,
  alpha: number
): number {
  return alpha * current + (1 - alpha) * previous;
}

/**
 * Softmax function
 */
export function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const expScores = logits.map(x => Math.exp(x - maxLogit));
  const sumExpScores = expScores.reduce((acc, val) => acc + val, 0);
  return expScores.map(x => x / sumExpScores);
}

/**
 * Find index of maximum value
 */
export function argmax(values: number[]): number {
  if (values.length === 0) {
    return -1;
  }
  let maxIndex = 0;
  let maxValue = values[0];
  for (let i = 1; i < values.length; i++) {
    if (values[i] > maxValue) {
      maxValue = values[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}
