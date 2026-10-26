import assert from 'node:assert/strict';
import { test } from 'node:test';
import { extractPersonMask } from '../person-mask.mjs';

function confidenceMask(values) {
  return { width: 2, height: 2, getAsFloat32Array: () => new Float32Array(values) };
}

test('single-channel selfie mask is the foreground confidence', () => {
  const result = { confidenceMasks: [confidenceMask([0, 0.2, 0.8, 1])] };
  const mask = extractPersonMask(result);
  assert.deepEqual(Array.from(mask.confidence), Array.from(new Float32Array([0, 0.2, 0.8, 1])));
  assert.equal(mask.width, 2);
  assert.equal(mask.height, 2);
});

test('dual-channel output selects foreground index 1', () => {
  const result = { confidenceMasks: [confidenceMask([1,1,1,1]), confidenceMask([0, 0, 0.5, 1])] };
  const mask = extractPersonMask(result);
  assert.deepEqual(Array.from(mask.confidence), [0, 0, 0.5, 1]);
});

test('category-mask fallback converts label 1 into filled person pixels', () => {
  const result = { confidenceMasks: [], categoryMask: {
    width: 2, height: 2, getAsUint8Array: () => Uint8Array.from([0, 1, 2, 1]),
  } };
  const mask = extractPersonMask(result);
  assert.deepEqual(Array.from(mask.confidence), [0, 1, 0, 1]);
});

test('copies callback-owned pixels so they survive disposal', () => {
  const original = new Float32Array([0, 1, 0, 1]);
  const mask = extractPersonMask({ confidenceMasks: [{width: 2,height: 2,getAsFloat32Array:()=> original}] });
  original.fill(0);
  assert.deepEqual(Array.from(mask.confidence), [0, 1, 0, 1]);
});

test('invalid outputs are rejected safely', () => {
  assert.equal(extractPersonMask({confidenceMasks: []}), null);
  assert.equal(extractPersonMask({confidenceMasks: [confidenceMask([1])]}), null);
  assert.equal(extractPersonMask(null), null);
});
