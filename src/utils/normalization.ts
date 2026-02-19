/**
 * Data normalization utilities
 */

/**
 * Normalize value to 0-1 range
 */
export function normalize(value: number, min: number, max: number): number {
  if (max === min) {
    return 0;
  }
  return (value - min) / (max - min);
}

/**
 * Standardize value using z-score normalization
 * (value - mean) / std
 */
export function standardize(value: number, mean: number, std: number): number {
  if (std === 0) {
    return 0;
  }
  return (value - mean) / std;
}

/**
 * Min-max normalization for array
 */
export function normalizeArray(values: number[]): number[] {
  if (values.length === 0) {
    return [];
  }

  const min = Math.min(...values);
  const max = Math.max(...values);

  if (max === min) {
    return values.map(() => 0);
  }

  return values.map(v => (v - min) / (max - min));
}

/**
 * Normalize image pixel values
 * Converts 0-255 range to 0-1 or -1 to 1
 */
export function normalizePixels(
  pixels: Uint8ClampedArray | number[],
  range: '0-1' | '-1-1' = '0-1'
): Float32Array {
  const normalized = new Float32Array(pixels.length);

  if (range === '0-1') {
    for (let i = 0; i < pixels.length; i++) {
      normalized[i] = pixels[i] / 255;
    }
  } else {
    for (let i = 0; i < pixels.length; i++) {
      normalized[i] = (pixels[i] / 255) * 2 - 1;
    }
  }

  return normalized;
}

/**
 * Convert RGB to grayscale using luminosity method
 */
export function rgbToGrayscale(r: number, g: number, b: number): number {
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

/**
 * Extract grayscale channel from RGBA ImageData
 */
export function extractGrayscale(imageData: ImageData): Float32Array {
  const { data, width, height } = imageData;
  const grayscale = new Float32Array(width * height);

  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    const gray = rgbToGrayscale(data[idx], data[idx + 1], data[idx + 2]);
    grayscale[i] = gray / 255; // Normalize to 0-1
  }

  return grayscale;
}
