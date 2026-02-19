import { describe, expect, it } from 'vitest';
import {
  normalize,
  standardize,
  normalizeArray,
  normalizePixels,
  rgbToGrayscale,
  extractGrayscale,
} from './normalization';

describe('normalization utilities', () => {
  it('normalizes and standardizes values', () => {
    expect(normalize(5, 0, 10)).toBe(0.5);
    expect(standardize(5, 5, 2)).toBe(0);
  });

  it('normalizes array with min-max', () => {
    expect(normalizeArray([2, 4, 6])).toEqual([0, 0.5, 1]);
  });

  it('normalizes pixels to expected ranges', () => {
    const pixels = new Uint8ClampedArray([0, 128, 255]);
    const norm01 = Array.from(normalizePixels(pixels, '0-1'));
    expect(norm01[0]).toBeCloseTo(0);
    expect(norm01[1]).toBeCloseTo(128 / 255, 6);
    expect(norm01[2]).toBeCloseTo(1);
    const neg = normalizePixels(pixels, '-1-1');
    expect(neg[0]).toBeCloseTo(-1);
    expect(neg[2]).toBeCloseTo(1);
  });

  it('converts RGB to grayscale and extracts from ImageData', () => {
    const gray = rgbToGrayscale(255, 255, 255);
    expect(gray).toBe(255);

    const data = new Uint8ClampedArray([
      // pixel 1 (red)
      255, 0, 0, 255,
      // pixel 2 (green)
      0, 255, 0, 255,
      // pixel 3 (blue)
      0, 0, 255, 255,
      // pixel 4 (white)
      255, 255, 255, 255,
    ]);

    const imageData: ImageData = typeof ImageData === 'function'
      ? new ImageData(data, 2, 2)
      : ({ data, width: 2, height: 2 } as unknown as ImageData);
    const grayscale = extractGrayscale(imageData);

    expect(grayscale.length).toBe(4);
    expect(grayscale[3]).toBeCloseTo(1); // white pixel normalized to 1
  });
});
