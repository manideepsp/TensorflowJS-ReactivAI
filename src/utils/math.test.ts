import { describe, expect, it } from 'vitest';
import {
  clamp,
  lerp,
  mapRange,
  mean,
  standardDeviation,
  rms,
  exponentialMovingAverage,
  softmax,
  argmax,
} from './math';

describe('math utilities', () => {
  it('clamps values', () => {
    expect(clamp(5, 0, 3)).toBe(3);
    expect(clamp(-1, 0, 3)).toBe(0);
  });

  it('lerps between values', () => {
    expect(lerp(0, 10, 0.5)).toBe(5);
  });

  it('maps range correctly', () => {
    expect(mapRange(5, 0, 10, 0, 100)).toBe(50);
  });

  it('computes mean, std, rms', () => {
    const values = [1, 2, 3, 4];
    expect(mean(values)).toBe(2.5);
    expect(standardDeviation(values)).toBeCloseTo(1.118, 3);
    expect(rms(values)).toBeCloseTo(2.7386, 4);
  });

  it('computes exponential moving average', () => {
    expect(exponentialMovingAverage(1, 0, 0.5)).toBe(0.5);
  });

  it('computes softmax and argmax', () => {
    const scores = [1, 2, 3];
    const probs = softmax(scores);
    const sum = probs.reduce((a, b) => a + b, 0);

    expect(sum).toBeCloseTo(1, 6);
    expect(argmax(probs)).toBe(2);
  });
});
