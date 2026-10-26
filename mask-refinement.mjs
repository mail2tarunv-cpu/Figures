/**
 * Refine a person/background confidence matte without losing small gaps between
 * arms, legs, and the body. Live uses temporal hysteresis; stills additionally
 * use the photo itself as an edge guide. Entirely client-side.
 */
export function refineConfidence(input, width, height, { previous = null, guide = null, quality = 'live' } = {}) {
  if (!input || input.length !== width * height || width < 1 || height < 1) {
    throw new Error('Confidence mask dimensions do not match the image');
  }
  const size = width * height;
  const output = new Float32Array(size);
  const canBlend = previous?.length === size && quality === 'live';

  // Keep strong changes (movement) instantaneous, stabilize only nearly equal
  // frame-to-frame boundary pixels. This avoids long silhouette ghost trails.
  for (let i = 0; i < size; i++) {
    const current = Math.max(0, Math.min(1, Number(input[i]) || 0));
    const prior = canBlend ? previous[i] : current;
    const close = Math.abs(current - prior) < 0.14;
    output[i] = close && current > 0.07 && current < 0.93
      ? current * 0.76 + prior * 0.24
      : current;
  }

  if (width < 3 || height < 3) return output;

  // Remove isolated one-pixel errors without closing real large holes or
  // erasing fingers, straps, arms, and the gap between a person's legs.
  const cleaned = new Float32Array(output);
  for (let y = 1; y < height - 1; y++) {
    const row = y * width;
    for (let x = 1; x < width - 1; x++) {
      const i = row + x;
      const v = output[i];
      if (v > 0.80) {
        let n = 0;
        n += output[i - 1] < 0.22;
        n += output[i + 1] < 0.22;
        n += output[i - width] < 0.22;
        n += output[i + width] < 0.22;
        if (n === 4) cleaned[i] = 0.1;
      } else if (v < 0.20) {
        let n = 0;
        n += output[i - 1] > 0.82;
        n += output[i + 1] > 0.82;
        n += output[i - width] > 0.82;
        n += output[i + width] > 0.82;
        if (n === 4) cleaned[i] = 0.9;
      }
    }
  }

  // A small RGB-guided boundary adjustment is reserved for captured photos.
  // It will not turn a low-resolution AI mask into a hand-traced cutout, but it
  // reduces halos where the image has a clear color edge at the silhouette.
  if (quality !== 'capture' || !guide || guide.length !== size * 4) return cleaned;
  const guided = new Float32Array(cleaned);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const center = cleaned[i];
      if (center < 0.08 || center > 0.92) continue;
      const p = i * 4;
      let sum = center * 2.0;
      let total = 2.0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const ni = (y + dy) * width + x + dx;
          const q = ni * 4;
          const difference = (
            Math.abs(guide[p] - guide[q]) +
            Math.abs(guide[p + 1] - guide[q + 1]) +
            Math.abs(guide[p + 2] - guide[q + 2])
          ) / 765;
          const weight = (dx && dy ? 0.55 : 1.0) / (1 + 80 * difference * difference);
          sum += cleaned[ni] * weight;
          total += weight;
        }
      }
      guided[i] = sum / total;
    }
  }
  return guided;
}

export function alphaFromConfidence(confidence, quality = 'live') {
  // Tight, smooth threshold keeps body regions completely opaque, retains a
  // ~1px anti-aliasing transition at the mask resolution, not a faded person.
  const start = quality === 'capture' ? 0.39 : 0.38;
  const end = quality === 'capture' ? 0.61 : 0.62;
  const t = Math.max(0, Math.min(1, (confidence - start) / (end - start)));
  return Math.round((t * t * (3 - 2 * t)) * 255);
}
