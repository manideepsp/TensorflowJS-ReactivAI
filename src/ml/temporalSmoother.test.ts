import { describe, expect, it } from 'vitest';
import { TemporalSmoother } from './temporalSmoother';

describe('TemporalSmoother', () => {
  it('returns first input unchanged and stores buffer', () => {
    const smoother = new TemporalSmoother(0.5);
    const input = new Float32Array([0.2, 0.8]);
    const out = smoother.smooth(input);

    expect(out[0]).toBeCloseTo(0.2);
    expect(out[1]).toBeCloseTo(0.8);
    const buf = smoother.getBuffer();
    expect(buf).not.toBeNull();
    expect(buf![0]).toBeCloseTo(0.2);
    expect(buf![1]).toBeCloseTo(0.8);
  });

  it('smooths subsequent values using EMA', () => {
    const smoother = new TemporalSmoother(0.5);
    smoother.smooth(new Float32Array([0.0, 1.0])); // seed
    const out = smoother.smooth(new Float32Array([1.0, 0.0]));

    expect(Array.from(out)).toEqual([0.5, 0.5]);
  });

  it('throws on length mismatch', () => {
    const smoother = new TemporalSmoother(0.4);
    smoother.smooth(new Float32Array([0.1, 0.2, 0.3]));

    expect(() => smoother.smooth(new Float32Array([0.1, 0.2]))).toThrow();
  });

  it('resets buffer', () => {
    const smoother = new TemporalSmoother(0.3);
    smoother.smooth(new Float32Array([0.3]));
    smoother.reset();

    expect(smoother.getBuffer()).toBeNull();
  });
});
