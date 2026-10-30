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
  // A weighted nine-point sample preserves joints and thin limbs more reliably
  // than four corner samples. Keep source coordinates aligned to the mask crop.
  const samples = [
    [.5, .5, .28],
    [.2, .2, .08], [.8, .2, .08], [.2, .8, .08], [.8, .8, .08],
    [.5, .2, .12], [.5, .8, .12], [.2, .5, .08], [.8, .5, .08]
  ];
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      let sum = 0, weightSum = 0;
      for (const [ox, oy, weight] of samples) {
        const px = Math.max(0, Math.min(maskWidth - 1, Math.floor(crop.mx + (x + ox) / cols * crop.mw)));
        const py = Math.max(0, Math.min(maskHeight - 1, Math.floor(crop.my + (y + oy) / rows * crop.mh)));
        sum += clamp01(confidence[py * maskWidth + px]) * weight;
        weightSum += weight;
      }
      tiles[y * cols + x] = sum / weightSum;
    }
  }
  return tiles;
}

/**
 * A dense, legible figure instead of a hollow cloud of dots.
 * Only the low-confidence edge has a few deterministic missing tiles.
 */
export function tileOpacity(confidence, noise = 0.5) {
  const c = clamp01(confidence);
  if (c < .16) return 0;
  if (c >= .76) return .98;
  if (c >= .52) return .9 + (c - .52) * .25;
  const t = (c - .16) / .36;
  if (clamp01(noise) > .90 + t * .05) return 0;
  return .43 + t * .39;
}

/**
 * Spatially stabilize the probability grid without smearing the mask between
 * frames. Deletes singleton segmentation specks and repairs tiny pinholes.
 * No temporal "ghost" accumulation, so limbs can change position immediately.
 */
export function stabilizeTileField(field, cols, rows) {
  if (!field || field.length !== cols * rows) throw new Error('Invalid tile field');
  const result = new Float32Array(field.length);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const i = y * cols + x;
      const center = clamp01(field[i]);
      let sum = center * 1.6, total = 1.6, strong = 0, nearby = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || xx >= cols || yy < 0 || yy >= rows) continue;
          const neighbor = clamp01(field[yy * cols + xx]);
          const weight = dx && dy ? .7 : 1;
          sum += neighbor * weight;
          total += weight;
          if (neighbor > .55) strong++;
          if (neighbor > .30) nearby++;
        }
      }
      const average = sum / total;
      // Isolated high-confidence noise also disappears; a true continuous thin
      // limb normally has two neighbors along its axis or its diagonal.
      if (nearby <= 1) { result[i] = 0; continue; }
      if (center < .43 && strong >= 5) {
        result[i] = Math.max(center, Math.min(.9, average * .96));
      } else {
        result[i] = center * .82 + average * .18;
      }
      // A confidence field must stay bounded even on unusual model output.
      result[i] = clamp01(result[i]);
    }
  }
  return result;
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
