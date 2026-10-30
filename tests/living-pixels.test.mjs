import test from 'node:test';
import assert from 'node:assert/strict';
import { buildConfidenceTiles, stabilizeTileField, tileOpacity, tileHash, findFigures, trackFigures, makeEchoPoints } from '../living-pixels.mjs';

test('confidence tiles sample the mask without hard-thresholding', () => {
  const field = new Float32Array([0,.3,.6,1]);
  const tiles = buildConfidenceTiles(field,2,2,2,2,{mx:0,my:0,mw:2,mh:2});
  assert.ok(Math.abs(tiles[1] - .3) < 1e-5);
  assert.equal(tiles[3], 1);
});
test('low confidence fades out instead of making a hard filled pixel', () => {
  assert.equal(tileOpacity(0.05,0), 0);
  assert.equal(tileOpacity(0.4,0.95), 0);
  assert.ok(tileOpacity(1,.9) > tileOpacity(.5,0));
});
test('tile hash is deterministic and spatially varies', () => {
  assert.equal(tileHash(2,3),tileHash(2,3));
  assert.notEqual(tileHash(2,3),tileHash(3,2));
});
test('connected components separate people', () => {
  const grid = new Float32Array(7*5);
  for (let y=1;y<=3;y++) for (let x of [1,2,5,6]) grid[y*7+x]=1;
  const people=findFigures(grid,7,5,.48,3);
  assert.equal(people.length,2);
  assert.ok(people[0].x !== people[1].x);
});
test('moving blob retains ID and reports movement', () => {
  const mk=(x)=>[{x,y:.5,area:.08,cells:[2,3,4]}];
  const first=trackFigures([],mk(.2),1,100);
  const second=trackFigures(first.tracks,mk(.23),first.nextId,210);
  assert.equal(second.tracks[0].id,first.tracks[0].id);
  assert.ok(second.movement>0.02);
  assert.equal(second.idsByCell.get(2),1);
});
test('disappeared blob expires and gets new identity', () => {
  const first=trackFigures([],[{x:.3,y:.5,area:.1,cells:[5]}],1,100);
  const second=trackFigures(first.tracks,[{x:.3,y:.5,area:.1,cells:[5]}],first.nextId,1200);
  assert.notEqual(first.tracks[0].id,second.tracks[0].id);
});
test('echo samples stay bounded and normalized', () => {
  const points=makeEchoPoints(new Float32Array(10000).fill(1),100,100,60);
  assert.ok(points.length<=60 && points.length>0);
  assert.ok(points.every(p=>p.x>=0&&p.x<=1&&p.y>=0&&p.y<=1));
});

test('pixel-field cleanup removes isolated segmentation specks', () => {
  const raw = new Float32Array(25);
  raw[12] = 0.98;
  const cleaned = stabilizeTileField(raw, 5, 5);
  assert.equal(cleaned[12], 0);
});

test('dense interiors and narrow legs remain continuous', () => {
  const solid = stabilizeTileField(new Float32Array(36).fill(1), 6, 6);
  assert.ok(solid.every(v => v > 0.98));

  const limb = new Float32Array(49);
  for (let y = 1; y < 6; y++) limb[y * 7 + 3] = 0.93;
  const refined = stabilizeTileField(limb, 7, 7);
  assert.ok(refined[3 * 7 + 3] > 0.7);
});

test('tile cleanup fills a small hole without washing away details', () => {
  const raw = new Float32Array(25).fill(.96);
  raw[12] = .25;
  const refined = stabilizeTileField(raw, 5, 5);
  assert.ok(refined[12] > .5);
  assert.ok(refined.every(v => v >= 0 && v <= 1));
});

test('dense tile cores stay almost fully opaque while uncertain edges fade', () => {
  assert.ok(tileOpacity(1, .99) > .95);
  assert.equal(tileOpacity(.05, 0), 0);
  assert.equal(tileOpacity(.4, .95), 0);
});
