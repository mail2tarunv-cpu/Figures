/** FIGURES — one living-pixel view, with a fixed #121212 background. */
import { extractPersonMask } from './person-mask.mjs';
import { refineConfidence } from './mask-refinement.mjs';
import { buildConfidenceTiles, stabilizeTileField, tileOpacity, tileHash, findFigures, trackFigures, makeEchoPoints } from './living-pixels.mjs';


const PACKAGE_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite';
const STAGE_COLOR = '#121212';
const MAX_LIVE_ANALYSIS_SIDE = 640;
const MAX_IMPORTED_ANALYSIS_SIDE = 960;
const MIN_ANALYSIS_INTERVAL_MS = 70;
const ECHO_LIFETIME_MS = 700;
const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)');
const elements = {
  canvas: document.querySelector('#outputCanvas'), stage: document.querySelector('#stage'),
  video: document.querySelector('#cameraVideo'), intro: document.querySelector('#intro'),
  start: document.querySelector('#startButton'), cameraHint: document.querySelector('#cameraHint'),
  importStart: document.querySelector('#importVideoButton'),
  importActive: document.querySelector('#importVideoActive'),
  fileInput: document.querySelector('#videoFileInput'),
  activeSource: document.querySelector('#activeSource'),
  sourceLabel: document.querySelector('#sourceLabel'),
  liveIndicator: document.querySelector('#liveIndicator'), toast: document.querySelector('#toast'),
  colorDock: document.querySelector('#colorDock'), colorTrigger: document.querySelector('#colorTrigger'),
  palette: document.querySelector('#palettePanel'), colorPreview: document.querySelector('#colorPreview'),
  colorHex: document.querySelector('#colorHex'), svField: document.querySelector('#svField'),
  svCursor: document.querySelector('#svCursor'), hueSlider: document.querySelector('#hueSlider'),
  presets: document.querySelectorAll('.preset')
};
const ctx = elements.canvas.getContext('2d', { alpha: false });
const analysisCanvas = document.createElement('canvas');
const analysisCtx = analysisCanvas.getContext('2d', { willReadFrequently: true });
const overlayCanvas = document.createElement('canvas');
const overlayCtx = overlayCanvas.getContext('2d', { alpha: true });
let maskRevision = 0, occupancyCache = null, toastTimeout = null;
const state = {
  mode: 'idle', facing: 'user', color: '#F5FC66',
  importedVideoUrl: null,
  stream: null, mediaWidth: 0, mediaHeight: 0, model: null, modelPromise: null, modelMode: null,
  confidence: null, previousConfidence: null, maskWidth: 0, maskHeight: 0, hasMask: false,
  animationHandle: null, lastInferAt: 0, lastRenderAt: 0, analysisIntervalMs: 88,
  lastVideoTime: -1, lastTimestamp: 0, busy: false,
  tracks: [], nextTrackId: 1, echoes: [], lastEchoAt: 0
};
const picker = { h: 65, s: 0.6, v: 0.99, open: false };
function notify(message, duration = 4000) {
  clearTimeout(toastTimeout);
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  toastTimeout = setTimeout(() => { elements.toast.hidden = true; }, duration);
}
function syncCanvasResolution() {
  const bounds = elements.stage.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const factor = Math.min(1, 1920 / Math.max(bounds.width * ratio, bounds.height * ratio));
  const w = Math.max(2, Math.round(bounds.width * ratio * factor));
  const h = Math.max(2, Math.round(bounds.height * ratio * factor));
  if (elements.canvas.width !== w || elements.canvas.height !== h) {
    elements.canvas.width = w; elements.canvas.height = h;
    maskRevision++; occupancyCache = null; drawFrame();
  }
}
function stopSource() {
  if (state.animationHandle !== null) cancelAnimationFrame(state.animationHandle);
  state.animationHandle = null;
  state.stream?.getTracks().forEach(track => track.stop());
  state.stream = null;
  elements.video.pause();
  elements.video.srcObject = null;
  elements.video.removeAttribute('src');
  elements.video.load();
  if (state.importedVideoUrl) URL.revokeObjectURL(state.importedVideoUrl);
  state.importedVideoUrl = null;
}
function clearMask() {
  state.confidence = null; state.previousConfidence = null;
  state.maskWidth = state.maskHeight = 0;
  state.hasMask = false; maskRevision++; occupancyCache = null;
  state.tracks = []; state.nextTrackId = 1;
  state.echoes = []; state.lastEchoAt = 0;
  elements.cameraHint.hidden = state.mode !== 'camera';
}
function getCrop(srcW, srcH, dstW, dstH) {
  // Center-crop both the photo and segmentation mask by the SAME rectangle.
  const sourceRatio = srcW / srcH;
  const targetRatio = dstW / dstH;
  if (sourceRatio > targetRatio) {
    const croppedWidth = srcH * targetRatio;
    return { x: (srcW - croppedWidth) / 2, y: 0, width: croppedWidth, height: srcH };
  }
  const croppedHeight = srcW / targetRatio;
  return { x: 0, y: (srcH - croppedHeight) / 2, width: srcW, height: croppedHeight };
}

function hexRgb(color) {
  return [parseInt(color.slice(1, 3), 16), parseInt(color.slice(3, 5), 16), parseInt(color.slice(5, 7), 16)];
}

function mixChannel(a, b, amount) {
  return Math.round(a + (b - a) * amount);
}

function buildPixelPalette() {
  const base = hexRgb(state.color);
  return {
    light: base.map(v => mixChannel(v, 255, 0.12)),
    base,
    shadow: base.map(v => mixChannel(v, 0, 0.10)),
  };
}


function drawFrame() {
  const w = elements.canvas.width, h = elements.canvas.height;
  if (!w || !h) return;
  ctx.save();
  ctx.fillStyle = STAGE_COLOR; ctx.fillRect(0, 0, w, h);
  if (state.hasMask && state.mediaWidth && state.mediaHeight) {
    if (state.mode === 'video') {
      // Contain the whole imported shot instead of cropping a portrait actor
      // to the browser viewport. Everything outside the actor stays #121212.
      const scale = Math.min(w / state.mediaWidth, h / state.mediaHeight);
      const dw = Math.max(1, Math.round(state.mediaWidth * scale));
      const dh = Math.max(1, Math.round(state.mediaHeight * scale));
      renderPixelOverlay(dw, dh, {
        mx: 0, my: 0, mw: state.maskWidth, mh: state.maskHeight
      }, performance.now());
      ctx.drawImage(overlayCanvas, Math.round((w - dw) / 2), Math.round((h - dh) / 2), dw, dh);
    } else {
      const { x, y, width, height } = getCrop(state.mediaWidth, state.mediaHeight, w, h);
      if (state.facing === 'user') { ctx.translate(w, 0); ctx.scale(-1, 1); }
      renderPixelOverlay(w, h, {
        mx: x / state.mediaWidth * state.maskWidth,
        my: y / state.mediaHeight * state.maskHeight,
        mw: width / state.mediaWidth * state.maskWidth,
        mh: height / state.mediaHeight * state.maskHeight
      }, performance.now());
      ctx.drawImage(overlayCanvas, 0, 0, w, h);
    }
  }
  ctx.restore();
}
function getPixelOccupancy(width, height, crop) {
  // Imported footage is framed small on desktop; use finer, denser pixels to
  // preserve a readable face, arms, legs and the actor's physical performance.
  const detail = state.mode === 'video' ? 96 : 64;
  const minimum = state.mode === 'video' ? 5 : 7;
  const cell = Math.max(minimum, Math.round(Math.min(width, height) / detail));
  const cols = Math.ceil(width / cell), rows = Math.ceil(height / cell);
  const key = `${maskRevision}:${width}:${height}:${cell}:${crop.mx.toFixed(3)}:${crop.my.toFixed(3)}:${crop.mw.toFixed(3)}:${crop.mh.toFixed(3)}`;
  if (occupancyCache?.key === key) return occupancyCache;
  const previous = occupancyCache;
  const rawField = buildConfidenceTiles(state.confidence, state.maskWidth, state.maskHeight, cols, rows, crop);
  const field = stabilizeTileField(rawField, cols, rows);
  const components = findFigures(field, cols, rows, .44, 7);
  const now = performance.now();
  const tracked = trackFigures(state.mode !== 'idle' ? state.tracks : [], components, state.nextTrackId, now);
  const identities = new Int32Array(cols * rows);
  for (const [index, id] of tracked.idsByCell) identities[index] = id;
  if (state.mode !== 'idle') {
    // Capture only actual movement, not an echo of an unmoving figure.
    if (state.mode === 'camera' && previous && previous.width === width && previous.height === height &&
        tracked.movement > .008 && now - state.lastEchoAt > 150 && !(reducedMotion?.matches)) {
      const points = makeEchoPoints(previous.field, previous.cols, previous.rows, 125);
      if (points.length) state.echoes.push({ points, capturedAt: now, cell: previous.cell });
      state.echoes = state.echoes.slice(-7);
      state.lastEchoAt = now;
    }
    state.tracks = tracked.tracks;
    state.nextTrackId = tracked.nextId;
  }
  occupancyCache = { key, field, cols, rows, cell, identities, width, height };
  return occupancyCache;
}

function renderPixelOverlay(width, height, crop, timeMs) {
  if (overlayCanvas.width !== width) overlayCanvas.width = width;
  if (overlayCanvas.height !== height) overlayCanvas.height = height;
  overlayCtx.clearRect(0, 0, width, height);
  const { field, cols, rows, cell, identities } = getPixelOccupancy(width, height, crop);
  const { light, base, shadow } = buildPixelPalette();
  const colors = [light, base, shadow].map(rgb => `rgb(${rgb.join(',')})`);
  const motionless = Boolean(reducedMotion?.matches);
  const seconds = (motionless ? 0 : timeMs) * .001;
  const now = performance.now();
  // Independent of model inference cadence: an echo remains behind and fades.
  if (state.mode === 'camera' && !motionless) {
    state.echoes = state.echoes.filter(e => now - e.capturedAt < ECHO_LIFETIME_MS);
    overlayCtx.fillStyle = colors[0];
    for (const echo of state.echoes) {
      const age = (now - echo.capturedAt) / ECHO_LIFETIME_MS;
      const alpha = (1 - age) ** 1.6 * .12;
      const step = Math.max(3, echo.cell * (.34 + age * .10));
      overlayCtx.globalAlpha = alpha;
      for (const p of echo.points) {
        const x = p.x * width + age * cell * 1.5;
        const y = p.y * height - age * cell * 1.2;
        overlayCtx.fillRect(Math.round(x), Math.round(y), step, step);
      }
    }
  }
  let prevColor = '';
  // One consistent, small negative space keeps the silhouette cohesive.
  const gap = Math.max(1, Math.round(cell * .08));
  for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
    const i = y * cols + x, c = field[i];
    const hash = tileHash(x, y);
    const opacity = tileOpacity(c, hash);
    if (!opacity) continue;
    const id = identities[i];
    // Stable, slowly travelling tone bands. Figure IDs keep the pattern phase
    // associated with a moving separated figure rather than the camera frame.
    const phase = x * .31 + y * .21 + id * .2 - seconds * 1.45 + hash * .6;
    const wave = Math.sin(phase) + .24 * Math.cos(y * .41 - seconds * .58);
    const tone = wave > .56 ? 0 : wave < -.66 ? 2 : 1;
    if (prevColor !== colors[tone]) { overlayCtx.fillStyle = colors[tone]; prevColor = colors[tone]; }
    // Solid core, softer articulated edges. Small, coherent motion preserves
    // the human performance instead of making every tile swim independently.
    const sizeFraction = c >= .74 ? .96 : c >= .50 ? .84 : (.58 + c * .38);
    const pixelSize = Math.max(2, Math.round((cell - gap) * sizeFraction));
    const inset = (cell - pixelSize) / 2;
    const bodyX = motionless ? 0 : Math.sin(seconds * .72 + id * .67) * cell * .025;
    const bodyY = motionless ? 0 : Math.cos(seconds * .64 + id * .59) * cell * .025;
    const microX = motionless ? 0 : Math.sin(seconds * .45 + x * .18 + hash * 4) * cell * .035;
    const microY = motionless ? 0 : Math.cos(seconds * .42 + y * .16 + hash * 3.2) * cell * .035;
    overlayCtx.globalAlpha = opacity;
    overlayCtx.fillRect(
      Math.round(x * cell + inset + bodyX + microX),
      Math.round(y * cell + inset + bodyY + microY),
      pixelSize, pixelSize
    );
  }
  overlayCtx.globalAlpha = 1;
}


function applySegmentation(result) {
  const mask = extractPersonMask(result);
  if (!mask) { console.warn('Person mask unavailable'); return; }
  const previous = state.maskWidth === mask.width && state.maskHeight === mask.height ? state.previousConfidence : null;
  const refined = refineConfidence(mask.confidence, mask.width, mask.height, { previous, quality: 'live' });
  let count = 0;
  for (let i = 0; i < refined.length; i += 16) if (refined[i] > 0.61) count++;
  if (count < 3) { if (state.hasMask) clearMask(); return; }
  state.maskWidth = mask.width; state.maskHeight = mask.height;
  state.confidence = refined; state.previousConfidence = refined;
  state.hasMask = true; elements.cameraHint.hidden = true;
  maskRevision++;
}
async function loadModel() {
  if (state.model) return state.model;
  if (state.modelPromise) return state.modelPromise;
  state.modelPromise = (async () => {
    const { ImageSegmenter, FilesetResolver } = await import(`${PACKAGE_BASE}/vision_bundle.mjs`);
    const wasm = await FilesetResolver.forVisionTasks(`${PACKAGE_BASE}/wasm`);
    const segmenter = await ImageSegmenter.createFromOptions(wasm, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' },
      runningMode: 'VIDEO',
      outputConfidenceMasks: true,
      outputCategoryMask: true,
    });
    state.model = segmenter;
    state.modelMode = 'VIDEO';
    return segmenter;
  })();
  try {
    return await state.modelPromise;
  } catch (error) {
    state.modelPromise = null;
    throw error;
  }
}


function analyzeSource() {
  // More source detail for uploaded clips, while leaving realtime webcam
  // inference capped for responsiveness. The model's native resolution still
  // limits fine edges; this avoids adding premature downsampling artifacts.
  const maxSide = state.mode === 'video' ? MAX_IMPORTED_ANALYSIS_SIDE : MAX_LIVE_ANALYSIS_SIDE;
  const scale = Math.min(1, maxSide / Math.max(state.mediaWidth, state.mediaHeight));
  const w = Math.max(2, Math.round(state.mediaWidth * scale));
  const h = Math.max(2, Math.round(state.mediaHeight * scale));
  if (analysisCanvas.width !== w) analysisCanvas.width = w;
  if (analysisCanvas.height !== h) analysisCanvas.height = h;
  analysisCtx.drawImage(elements.video, 0, 0, w, h);
}
function tick(now) {
  if (state.mode !== 'camera' && state.mode !== 'video') return;
  if (elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    // Imported clips loop; discard stale segmentation/trails at the seam.
    if (state.mode === 'video' && elements.video.currentTime < state.lastVideoTime - .08) {
      clearMask();
      state.lastVideoTime = -1;
      state.lastInferAt = 0;
    }
    if (now - state.lastRenderAt >= 30) { drawFrame(); state.lastRenderAt = now; }
    if (state.model && state.modelMode === 'VIDEO' && !document.hidden &&
        now - state.lastInferAt >= state.analysisIntervalMs &&
        elements.video.currentTime !== state.lastVideoTime) {
      state.lastInferAt = now; state.lastVideoTime = elements.video.currentTime;
      analyzeSource();
      try {
        const timestamp = Math.max(Math.floor(now), state.lastTimestamp + 1);
        state.lastTimestamp = timestamp;
        const start = performance.now();
        state.model.segmentForVideo(analysisCanvas, timestamp, applySegmentation);
        state.analysisIntervalMs = Math.max(MIN_ANALYSIS_INTERVAL_MS,
          Math.min(220, (performance.now() - start) * 1.25));
      } catch (error) {
        console.error('Segmentation failed', error);
        state.modelMode = 'error';
        notify('Tracking stopped. Reload to retry.');
      }
    }
  }
  state.animationHandle = requestAnimationFrame(tick);
}
async function startCamera() {
  if (state.busy || state.mode === 'camera') return;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    notify('Camera requires HTTPS or localhost.'); return;
  }
  state.busy = true;
  stopSource(); clearMask();
  elements.start.disabled = true;
  elements.importStart.disabled = true;
  elements.importActive.disabled = true;
  elements.start.querySelector('span').textContent = 'Starting…';
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'user' }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    elements.video.srcObject = state.stream;
    await elements.video.play();
    state.mediaWidth = elements.video.videoWidth; state.mediaHeight = elements.video.videoHeight;
    if (!state.mediaWidth || !state.mediaHeight) throw new Error('No camera frames');
    elements.start.querySelector('span').textContent = 'Loading vision…';
    await loadModel();
    state.mode = 'camera';
    elements.intro.hidden = true;
    elements.activeSource.hidden = false;
    elements.sourceLabel.textContent = 'LIVE';
    elements.liveIndicator.hidden = false;
    elements.colorDock.hidden = false;
    elements.cameraHint.hidden = false;
    state.lastInferAt = 0; state.lastRenderAt = 0; state.lastVideoTime = -1;
    state.lastTimestamp = 0; state.analysisIntervalMs = 88;
    syncCanvasResolution(); state.animationHandle = requestAnimationFrame(tick);
  } catch (error) {
    console.error('Camera startup failed', error); stopSource(); state.mode = 'idle';
    elements.intro.hidden = false; elements.activeSource.hidden = true; elements.colorDock.hidden = true;
    const denied = error?.name === 'NotAllowedError' || error?.name === 'PermissionDeniedError';
    notify(denied ? 'Allow camera access in browser settings.' : 'Camera unavailable. Check permissions and connection.', 6000);
  } finally {
    state.busy = false; elements.start.disabled = false;
    elements.importStart.disabled = false;
    elements.importActive.disabled = false;
    elements.start.querySelector('span').textContent = 'Start camera';
  }
}
function isVideoFile(file) {
  return Boolean(file && (file.type.startsWith('video/') || /\.(mp4|m4v|mov|webm|ogv)$/i.test(file.name)));
}
async function loadImportedVideo(file) {
  elements.fileInput.value = ''; // Allow re-selecting the same file later.
  if (!file || state.busy) return;
  if (!isVideoFile(file)) { notify('Choose a video file (MP4, MOV or WebM).'); return; }

  state.busy = true;
  elements.start.disabled = true;
  elements.importStart.disabled = true;
  elements.importActive.disabled = true;
  stopSource();
  state.mode = 'idle';
  clearMask();
  elements.activeSource.hidden = true;
  elements.colorDock.hidden = true;
  elements.intro.hidden = false;
  elements.importStart.querySelector('span').textContent = 'Loading video…';
  drawFrame();

  try {
    // File stays on this device. No uploads; the browser decodes it locally.
    state.importedVideoUrl = URL.createObjectURL(file);
    elements.video.loop = true;
    elements.video.muted = true;
    elements.video.playsInline = true;
    elements.video.preload = 'auto';
    elements.video.src = state.importedVideoUrl;
    elements.video.load();

    // Start muted playback directly after file selection; load inference in parallel.
    await Promise.all([elements.video.play(), loadModel()]);
    state.mediaWidth = elements.video.videoWidth;
    state.mediaHeight = elements.video.videoHeight;
    if (!state.mediaWidth || !state.mediaHeight) throw new Error('No decoded video dimensions');

    // Start the preview at the beginning, even if loading the model took time.
    elements.video.currentTime = 0;
    state.mode = 'video';
    elements.intro.hidden = true;
    elements.activeSource.hidden = false;
    elements.sourceLabel.textContent = 'VIDEO LOOP';
    elements.liveIndicator.hidden = false;
    elements.colorDock.hidden = false;
    elements.cameraHint.hidden = true;
    state.lastInferAt = 0;
    state.lastRenderAt = 0;
    state.lastVideoTime = -1;
    state.lastTimestamp = 0;
    state.analysisIntervalMs = 88;
    syncCanvasResolution();
    state.animationHandle = requestAnimationFrame(tick);
  } catch (error) {
    console.error('Could not import video:', error);
    stopSource();
    state.mode = 'idle';
    clearMask();
    elements.intro.hidden = false;
    elements.activeSource.hidden = true;
    elements.colorDock.hidden = true;
    drawFrame();
    notify('Could not play this file. Try an H.264 MP4 or another browser.', 6500);
  } finally {
    state.busy = false;
    elements.start.disabled = false;
    elements.importStart.disabled = false;
    elements.importActive.disabled = false;
    elements.importStart.querySelector('span').textContent = 'Import video';
  }
}
function hsvToHex(h,s,v) {
  const f = n => {
    const k = (n + h / 60) % 6;
    return Math.round((v - v*s*Math.max(0,Math.min(k,4-k,1)))*255);
  };
  return '#' + [f(5),f(3),f(1)].map(n=>n.toString(16).padStart(2,'0')).join('').toUpperCase();
}
function hexToHsv(hex) {
  const [r,g,b] = hexRgb(hex).map(v=>v/255);
  const hi=Math.max(r,g,b),lo=Math.min(r,g,b),d=hi-lo;
  let h=0;
  if (d) {
    if (hi===r) h=((g-b)/d)%6;
    else if (hi===g) h=(b-r)/d+2;
    else h=(r-g)/d+4;
    h=(h*60+360)%360;
  }
  return {h,s:hi===0?0:d/hi,v:hi};
}
function updatePickerUI() {
  const hex=hsvToHex(picker.h,picker.s,picker.v);
  state.color=hex;
  elements.colorPreview.style.backgroundColor=hex;
  elements.colorHex.textContent=hex;
  elements.svField.style.setProperty('--picked-hue',`hsl(${picker.h} 100% 50%)`);
  elements.svCursor.style.left=`${picker.s*100}%`;
  elements.svCursor.style.top=`${(1-picker.v)*100}%`;
  elements.hueSlider.value=Math.round(picker.h);
  elements.svField.setAttribute('aria-valuetext', `Saturation ${Math.round(picker.s*100)} percent, brightness ${Math.round(picker.v*100)} percent`);
  elements.presets.forEach(button=>{
    const selected=button.dataset.color.toUpperCase()===hex;
    button.classList.toggle('is-selected',selected);
    button.setAttribute('aria-pressed',String(selected));
  });
  if (state.hasMask) drawFrame();
}
function setColor(hex) {
  const hsv=hexToHsv(hex); Object.assign(picker,hsv); updatePickerUI();
}
function togglePicker(force) {
  picker.open=typeof force==='boolean'?force:!picker.open;
  elements.palette.hidden=!picker.open;
  elements.colorTrigger.setAttribute('aria-expanded',String(picker.open));
  elements.colorDock.classList.toggle('is-open',picker.open);
}
function sampleSV(event) {
  const box=elements.svField.getBoundingClientRect();
  if (!box.width||!box.height) return;
  picker.s=Math.max(0,Math.min(1,(event.clientX-box.left)/box.width));
  picker.v=1-Math.max(0,Math.min(1,(event.clientY-box.top)/box.height));
  updatePickerUI();
}
function bindUI() {
  elements.start.addEventListener('click',startCamera);
  const chooseVideo = () => { if (!state.busy) elements.fileInput.click(); };
  elements.importStart.addEventListener('click',chooseVideo);
  elements.importActive.addEventListener('click',chooseVideo);
  elements.fileInput.addEventListener('change',event => {
    void loadImportedVideo(event.target.files?.[0]);
  });
  elements.colorTrigger.addEventListener('click',()=>togglePicker());
  elements.hueSlider.addEventListener('input',e=>{picker.h=Number(e.target.value);updatePickerUI();});
  elements.svField.addEventListener('pointerdown',e=>{
    elements.svField.setPointerCapture(e.pointerId);sampleSV(e);
  });
  elements.svField.addEventListener('pointermove',e=>{
    if(elements.svField.hasPointerCapture(e.pointerId))sampleSV(e);
  });
  elements.svField.addEventListener('pointerup',e=>{
    if(elements.svField.hasPointerCapture(e.pointerId))elements.svField.releasePointerCapture(e.pointerId);
  });
  elements.svField.addEventListener('keydown',e=>{
    const step=e.shiftKey?.1:.03;
    if(e.key==='ArrowLeft')picker.s=Math.max(0,picker.s-step);
    else if(e.key==='ArrowRight')picker.s=Math.min(1,picker.s+step);
    else if(e.key==='ArrowUp')picker.v=Math.min(1,picker.v+step);
    else if(e.key==='ArrowDown')picker.v=Math.max(0,picker.v-step);
    else return;
    e.preventDefault();updatePickerUI();
  });
  elements.presets.forEach(button=>button.addEventListener('click',()=>setColor(button.dataset.color)));
  document.addEventListener('pointerdown',e=>{
    if(picker.open&&!elements.colorDock.contains(e.target))togglePicker(false);
  });
  document.addEventListener('keydown',async e=>{
    if(e.key==='Escape'){togglePicker(false);return;}
    if(e.target instanceof HTMLInputElement)return;
    if(e.key.toLowerCase()==='c'&&state.mode!=='idle')togglePicker();
    if(e.key.toLowerCase()==='f'&&state.mode!=='idle') {
      try { if(document.fullscreenElement)await document.exitFullscreen();
        else await document.documentElement.requestFullscreen();
      } catch(err){console.warn('Fullscreen unavailable',err);}
    }
  });
  window.addEventListener('resize',syncCanvasResolution);
  window.addEventListener('pagehide',stopSource);
  elements.video.addEventListener('seeked', () => {
    if (state.mode === 'video') { clearMask(); state.lastVideoTime = -1; state.lastInferAt = 0; }
  });
  document.addEventListener('visibilitychange',()=>{
    if(!document.hidden)state.lastInferAt=0;
  });
}
bindUI();setColor(state.color);syncCanvasResolution();
