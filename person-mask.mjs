/**
 * Normalize MediaPipe ImageSegmenter outputs to one owned foreground confidence map.
 *
 * The selfie segmenter may return a single foreground confidence channel, even
 * though its category IDs are 0 (background) and 1 (person). Other segmenters
 * can return separate confidence channels for both classes.
 * If only category output is available, convert label 1 to an opaque mask.
 */
export function extractPersonMask(result) {
  const masks = result?.confidenceMasks;
  if (masks?.length) {
    // 2+ channels => category index 1; 1 channel => foreground confidence.
    const mask = masks.length > 1 ? masks[1] : masks[0];
    if (mask && typeof mask.getAsFloat32Array === 'function') {
      const values = mask.getAsFloat32Array();
      if (values?.length === mask.width * mask.height) {
        // MediaPipe releases callback-owned masks after the callback returns.
        return {
          width: mask.width,
          height: mask.height,
          confidence: new Float32Array(values),
        };
      }
    }
  }

  // Robust fallback for models/runtimes that only return the category mask.
  const category = result?.categoryMask;
  if (category && typeof category.getAsUint8Array === 'function') {
    const labels = category.getAsUint8Array();
    if (labels?.length === category.width * category.height) {
      const confidence = new Float32Array(labels.length);
      for (let i = 0; i < labels.length; i++) {
        confidence[i] = labels[i] === 1 ? 1 : 0;
      }
      return { width: category.width, height: category.height, confidence };
    }
  }

  return null;
}
