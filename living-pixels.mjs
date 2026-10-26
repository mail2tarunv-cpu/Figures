/**
 * FIGURES living-pixel maths. Pure functions so mask/track behavior can be tested
 * without a camera, DOM or MediaPipe. All coordinates are in display-grid space.
 */
export const clamp01 = value => Math.max(0, Math.min(1, Number(value) || 0));
export const tileHash = (x, y) => {
  let h = Math.imul((x + 1) ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul((y + 1) ^ 0xc2b2ae35, 0x27d4eb2d);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
};

/** Sample genuine confidence values before binary thresholding. */
export function buildConfidenceTiles(confidence, maskWidth, maskHeight, cols, rows, crop) {
  if (!confidence || confidence.length !== maskWidth * maskHeight) throw new Error('Invalid confidence field');
  const tiles = new Float32Array(cols * rows);
  // Two by two samples preserve gaps but average out the noisy 256px boundary.
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      let sum = 0;
      for (const oy of [0.28, 0.72]) {
        for (const ox of [0.28, 0.72]) {
          const px = Math.max(0, Math.min(maskWidth - 1, Math.floor(crop.mx + (x + ox) / cols * crop.mw)));
          const py = Math.max(0, Math.min(maskHeight - 1, Math.floor(crop.my + (y + oy) / rows * crop.mh)));
          sum += clamp01(confidence[py * maskWidth + px]);
        }
      }
      tiles[y * cols + x] = sum / 4;
    }
  }
  return tiles;
}

/** Alpha/density based on the *probability*, not a hard mask edge. */
export function tileOpacity(confidence, noise = 0.5) {
  const c = clamp01(confidence);
  if (c < 0.11) return 0;
  const n = clamp01(noise);
  const density = Math.pow((c - 0.1) / 0.9, 0.78);
  // Stable blue-noise-ish dropout only along uncertain edges.
  if (n > Math.min(1, density * 1.36)) return 0;
  return Math.min(0.98, 0.24 + density * 0.72);
}

/** 8-connected groups on the coarse field, not person identity recognition. */
export function findFigures(tiles, cols, rows, cutoff = 0.48, minTiles = 5) {
  const seen = new Uint8Array(tiles.length);
  const figures = [];
  for (let start = 0; start < tiles.length; start++) {
    if (seen[start] || tiles[start] < cutoff) continue;
    seen[start] = 1;
    const queue = [start];
    const cells = [];
    let sx = 0, sy = 0;
    for (let j = 0; j < queue.length; j++) {
      const index = queue[j];
      const x = index % cols, y = Math.floor(index / cols);
      cells.push(index); sx += x + 0.5; sy += y + 0.5;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) continue;
        const next = ny * cols + nx;
        if (!seen[next] && tiles[next] >= cutoff) {
          seen[next] = 1;
          queue.push(next);
        }
      }
    }
    if (cells.length >= minTiles) {
      figures.push({ cells, x: sx / cells.length / cols, y: sy / cells.length / rows, area: cells.length / tiles.length });
    }
  }
  return figures.sort((a, b) => b.cells.length - a.cells.length);
}

/** Associate figure blobs between analyses using nearest motion-predicted center.
 * Stable during separation but merges/occlusions cannot be resolved by a single
 * foreground mask. Old IDs expire after occlusion or camera reset.
 */
export function trackFigures(previous, figures, nextId, now) {
  const live = previous.filter(t => now - t.seenAt < 750);
  const used = new Set();
  const tracks = [];
  const idsByCell = new Map();
  let movement = 0;
  for (const figure of figures) {
    let match = null, best = Infinity;
    for (const track of live) {
      if (used.has(track.id)) continue;
      const dt = Math.min(0.25, Math.max(0, (now - track.seenAt) / 1000));
      const px = track.x + track.vx * dt, py = track.y + track.vy * dt;
      const d = Math.hypot(figure.x - px, figure.y - py);
      const relativeArea = Math.abs(Math.log((figure.area + .005) / (track.area + .005)));
      const score = d + relativeArea * .035;
      if (d < .19 && score < best) { best = score; match = track; }
    }
    let track;
    if (match) {
      used.add(match.id);
      const dt = Math.max(0.045, (now - match.seenAt) / 1000);
      const dx = figure.x - match.x, dy = figure.y - match.y;
      movement = Math.max(movement, Math.hypot(dx, dy));
      track = { id: match.id, x: figure.x, y: figure.y, area: figure.area, vx: (dx / dt) * .6 + match.vx * .4, vy: (dy / dt) * .6 + match.vy * .4, seenAt: now };
    } else {
      track = { id: nextId++, x: figure.x, y: figure.y, area: figure.area, vx: 0, vy: 0, seenAt: now };
    }
    tracks.push(track);
    for (const cell of figure.cells) idsByCell.set(cell, track.id);
  }
  return { tracks, nextId, idsByCell, movement };
}

export function makeEchoPoints(tiles, cols, rows, maxPoints = 140) {
  const points = [];
  const stride = Math.max(2, Math.ceil(Math.sqrt(cols * rows / maxPoints)));
  for (let y = 0; y < rows; y += stride) {
    for (let x = 0; x < cols; x += stride) {
      const coverage = tiles[y * cols + x];
      if (coverage < 0.35) continue;
      const h = tileHash(x, y);
      if (tileOpacity(coverage, h) === 0) continue;
      points.push({ x: (x + .5) / cols, y: (y + .5) / rows, c: coverage, h });
    }
  }
  return points.slice(0, maxPoints);
}
