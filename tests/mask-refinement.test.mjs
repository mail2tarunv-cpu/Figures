import test from 'node:test';
import assert from 'node:assert/strict';
import { refineConfidence, alphaFromConfidence } from '../mask-refinement.mjs';

test('solid human areas are fully opaque and background stays transparent', () => {
  assert.equal(alphaFromConfidence(1), 255);
  assert.equal(alphaFromConfidence(0), 0);
  assert.equal(alphaFromConfidence(0.5), 128);
});

test('single-pixel isolated false foreground is removed', () => {
  const input = Float32Array.from([0,0,0, 0,1,0, 0,0,0]);
  const mask = refineConfidence(input, 3, 3);
  assert.ok(mask[4] < 0.22);
});

test('a real 3-pixel-wide person band survives', () => {
  const input = new Float32Array(25);
  for (let y = 0; y < 5; y++) for (let x = 1; x < 4; x++) input[y*5+x] = 1;
  const mask = refineConfidence(input, 5, 5);
  assert.equal(mask[12], 1);
});

test('stabilization smooths small variations but follows new silhouette positions', () => {
  const previous = new Float32Array(9).fill(0.54);
  const input = new Float32Array(9).fill(0.56);
  const stable = refineConfidence(input, 3, 3, { previous });
  assert.ok(stable[4] < 0.56 && stable[4] > 0.54);
  const changed = refineConfidence(new Float32Array(9).fill(1), 3, 3, { previous });
  assert.equal(changed[4], 1);
});

test('capture guided pass respects mask dimensions and returns bounded alpha', () => {
  const input = new Float32Array(25).fill(0.5);
  const guide = new Uint8ClampedArray(25*4);
  for (let y=0; y<5; y++) for (let x=0; x<5; x++) {
    const p=(y*5+x)*4; guide[p]=guide[p+1]=guide[p+2]=x<2?0:255; guide[p+3]=255;
  }
  const mask=refineConfidence(input,5,5,{guide,quality:'capture'});
  assert.equal(mask.length,25);
  for(const v of mask) assert.ok(Number.isFinite(v) && v>=0 && v<=1);
});

test('invalid masks throw instead of rendering a shifted shape', () => {
  assert.throws(() => refineConfidence(new Float32Array(3),4,4),/dimensions/);
});
