/**
 * FIGURES — Experiment 001
 * A compact, client-side live camera / photo segmenter.
 * MediaPipe performs on-device inference; camera imagery is not sent to our servers.
 * The JS/WASM bundle and model are fetched from public CDNs on first use.
 */

import { extractPersonMask } from './person-mask.mjs';
import { refineConfidence, alphaFromConfidence } from './mask-refinement.mjs';

const MEDIAPIPE_VERSION = '1.1.1';
const PACKAGE_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite';
const OUTPUT_HEIGHT = 1200;
// Camera inference is constrained by the 256px segmentation model itself.
// A larger analysis frame still helps preserve sampling/color boundaries.
const MAX_LIVE_ANALYSIS_SIDE = 640;
const MAX_CAPTURE_ANALYSIS_SIDE = 1400;
const MAX_EXPORT_SIDE = 2400;
const MIN_ANALYSIS_INTERVAL_MS = 70;

const elements = {
  canvas: document.querySelector('#outputCanvas'),
  video: document.querySelector('#cameraVideo'),
  empty: document.querySelector('#emptyState'),
  live: document.querySelector('#liveTag'),
  viewBottom: document.querySelector('#viewBottom'),
  viewfinder: document.querySelector('#viewfinder'),
  toast: document.querySelector('#toast'),
  start: document.querySelector('#startButton'),
  uploadStart: document.querySelector('#uploadStartButton'),
  upload: document.querySelector('#uploadButton'),
  file: document.querySelector('#fileInput'),
  shutter: document.querySelector('#shutterButton'),
  flip: document.querySelector('#flipButton'),
  compare: document.querySelector('#compareButton'),
  backToCamera: document.querySelector('#backToCameraButton'),
  colorButtons: document.querySelectorAll('.swatch'),
  info: document.querySelector('#infoButton'),
  modal: document.querySelector('#aboutModal'),
  closeAbout: document.querySelector('#closeAbout'),
};

const ctx = elements.canvas.getContext('2d', { alpha: false });
const analysisCanvas = document.createElement('canvas');
const analysisCtx = analysisCanvas.getContext('2d', { willReadFrequently: true });
const maskCanvas = document.createElement('canvas');
const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
const overlayCanvas = document.createElement('canvas');
const overlayCtx = overlayCanvas.getContext('2d', { alpha: true });
const occupancyCanvas = document.createElement('canvas');
const occupancyCtx = occupancyCanvas.getContext('2d', { willReadFrequently: true });
let maskRevision = 0;
let occupancyCache = null;

const state = {
  mode: 'idle',                    // idle | camera | photo
  facing: 'environment',
  color: '#F5FC66',
  stream: null,
  photo: null,
  mediaWidth: 0,
  mediaHeight: 0,
  model: null,
  modelPromise: null,
  modelMode: null,
  confidence: null,
  previousConfidence: null,
  maskQuality: 'live',
  maskWidth: 0,
  maskHeight: 0,
  hasMask: false,
  comparing: false,
  busy: false,
  animationHandle: null,
  lastInferAt: 0,
  analysisIntervalMs: 88,
  lastVideoTime: -1,
  lastTimestamp: 0,
  session: 0,
  frozenEffectTime: 0,
};

let toastTimeout = null;

function notify(message, duration = 3200) {
  clearTimeout(toastTimeout);
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  toastTimeout = setTimeout(() => { elements.toast.hidden = true; }, duration);
}

function syncCanvasResolution() {
  // Match the saved PNG crop exactly to the visible viewfinder at every size.
  const bounds = elements.viewfinder.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const w = Math.max(360, Math.round(OUTPUT_HEIGHT * bounds.width / bounds.height));
  if (elements.canvas.width !== w || elements.canvas.height !== OUTPUT_HEIGHT) {
    elements.canvas.width = w;
    elements.canvas.height = OUTPUT_HEIGHT;
    drawFrame();
  }
}

function hideEmpty() {
  elements.empty.hidden = true;
  elements.viewBottom.hidden = false;
  syncCanvasResolution();
}

function refreshControls() {
  const active = state.mode !== 'idle';
  elements.live.hidden = state.mode !== 'camera';
  elements.flip.disabled = state.mode !== 'camera' || state.busy;
  elements.shutter.disabled = !active || state.busy || (state.mode === 'photo' && !state.hasMask);
  elements.shutter.setAttribute('aria-label', state.mode === 'camera' ? 'Take photo and refine edges' : 'Save edited photo');
  elements.shutter.classList.toggle('save-mode', state.mode === 'photo');
  const sceneTip = elements.viewBottom.querySelector('.scene-tip');
  if (sceneTip) sceneTip.textContent = state.mode === 'photo' ? 'TAP ↓ TO SAVE' : 'PIXEL FLOW';
  elements.compare.disabled = !active || !state.hasMask || state.busy;
  elements.backToCamera.disabled = state.mode !== 'photo' || state.busy;
  elements.compare.hidden = !active || !state.hasMask;
  elements.backToCamera.hidden = state.mode !== 'photo';
  for (const button of elements.colorButtons) {
    const selected = button.dataset.color === state.color;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  }
}

function stopVideo() {
  if (state.animationHandle != null) cancelAnimationFrame(state.animationHandle);
  state.animationHandle = null;
  if (state.stream) {
    for (const track of state.stream.getTracks()) track.stop();
  }
  state.stream = null;
  elements.video.pause();
  elements.video.srcObject = null;
}

function releasePhoto() {
  if (state.photo && typeof state.photo.close === 'function') state.photo.close();
  state.photo = null;
}

function clearMask() {
  state.confidence = null;
  state.previousConfidence = null;
  state.maskQuality = 'live';
  state.hasMask = false;
  state.maskWidth = state.maskHeight = 0;
  state.frozenEffectTime = 0;
  maskRevision++;
  occupancyCache = null;
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  refreshControls();
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

function drawFrame(targetCanvas = elements.canvas, targetContext = ctx) {
  if (state.mode === 'idle') return;
  const source = state.mode === 'camera' ? elements.video : state.photo;
  if (!source || !state.mediaWidth || !state.mediaHeight) return;

  const { width: w, height: h } = targetCanvas;
  const { x, y, width, height } = getCrop(state.mediaWidth, state.mediaHeight, w, h);
  targetContext.clearRect(0, 0, w, h);
  targetContext.save();
  if (state.mode === 'camera' && state.facing === 'user') {
    targetContext.translate(w, 0);
    targetContext.scale(-1, 1);
  }
  targetContext.imageSmoothingEnabled = true;
  targetContext.imageSmoothingQuality = 'high';
  targetContext.drawImage(source, x, y, width, height, 0, 0, w, h);

  if (state.hasMask && !state.comparing) {
    const mx = x / state.mediaWidth * state.maskWidth;
    const my = y / state.mediaHeight * state.maskHeight;
    const mw = width / state.mediaWidth * state.maskWidth;
    const mh = height / state.mediaHeight * state.maskHeight;
    const effectTime = state.mode === 'camera' ? performance.now() : state.frozenEffectTime;
    renderPixelOverlay(w, h, { mx, my, mw, mh }, effectTime);
    targetContext.drawImage(overlayCanvas, 0, 0, w, h);
  }
  targetContext.restore();
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
    light: base.map(v => mixChannel(v, 255, 0.24)),
    base,
    shadow: base.map(v => mixChannel(v, 0, 0.18)),
  };
}

// Turn the soft neural-network cutout into an intentional grid of square tiles.
// Crucially, the grid *itself* defines the silhouette edge, rather than drawing
// a sharp pixel pattern under the same visibly inaccurate smooth outline.
function getPixelOccupancy(width, height, crop) {
  const cell = Math.max(12, Math.round(Math.min(width, height) / 37));
  const cols = Math.ceil(width / cell);
  const rows = Math.ceil(height / cell);
  const key = `${maskRevision}:${width}:${height}:${cell}:${crop.mx.toFixed(3)}:${crop.my.toFixed(3)}:${crop.mw.toFixed(3)}:${crop.mh.toFixed(3)}`;
  if (occupancyCache?.key === key) return occupancyCache;

  occupancyCanvas.width = cols;
  occupancyCanvas.height = rows;
  occupancyCtx.clearRect(0, 0, cols, rows);
  // Browser resampling here effectively averages each mask tile's coverage.
  occupancyCtx.drawImage(maskCanvas, crop.mx, crop.my, crop.mw, crop.mh, 0, 0, cols, rows);
  const rgba = occupancyCtx.getImageData(0, 0, cols, rows).data;
  const coverage = new Uint8Array(cols * rows);
  for (let i = 0; i < coverage.length; i++) coverage[i] = rgba[i * 4 + 3];
  occupancyCache = { key, coverage, cols, rows, cell };
  return occupancyCache;
}

// Subtle continuous directional color waves and breathing movement of each
// square make the field feel spatial and alive without a frantic glitch effect.
function renderPixelOverlay(width, height, crop, timeMs) {
  if (overlayCanvas.width !== width) overlayCanvas.width = width;
  if (overlayCanvas.height !== height) overlayCanvas.height = height;
  overlayCtx.clearRect(0, 0, width, height);
  const { coverage, cols, rows, cell } = getPixelOccupancy(width, height, crop);
  const { light, base, shadow } = buildPixelPalette();
  const colors = [
    `rgb(${light.join(',')})`,
    `rgb(${base.join(',')})`,
    `rgb(${shadow.join(',')})`,
  ];
  const t = timeMs * 0.001;
  const gap = Math.max(1, Math.round(cell * 0.1));
  const size = cell - gap;
  let previousStyle = '';

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const i = row * cols + col;
      const amount = coverage[i] / 255;
      // Strong inner pixels remain consistently filled; ambiguous boundary
      // pixels become sparse squares rather than smeared AI-cutout borders.
      const hash = ((col * 73 + row * 139 + col * row * 17) % 101) / 101;
      if (amount < (amount > 0.72 ? 0.16 : 0.28 + hash * 0.23)) continue;
      const flow = Math.sin(col * 0.38 + row * 0.26 - t * 1.35 + hash * 0.7);
      const shimmer = Math.cos(col * 0.31 - row * 0.19 + t * 0.55);
      const tone = flow + 0.25 * shimmer > 0.67 ? 0 : flow < -0.65 ? 2 : 1;
      if (colors[tone] !== previousStyle) {
        previousStyle = colors[tone];
        overlayCtx.fillStyle = previousStyle;
      }
      const xDrift = Math.sin(t * 0.85 + row * 0.33 + hash * 5.1) * cell * 0.11;
      const yDrift = Math.cos(t * 0.68 + col * 0.41 + hash * 4.2) * cell * 0.13;
      // Intentionally omit squares at some edges to create a floating breakup.
      const edge = amount < 0.7;
      const pixelSize = edge ? Math.round(size * 0.77) : size;
      const center = (cell - pixelSize) / 2;
      overlayCtx.globalAlpha = edge ? Math.min(0.92, 0.54 + amount * 0.6) : 0.96;
      overlayCtx.fillRect(
        Math.round(col * cell + center + xDrift),
        Math.round(row * cell + center + yDrift),
        pixelSize, pixelSize
      );
    }
  }
  overlayCtx.globalAlpha = 1;
}

function paintMask() {
  if (!state.confidence || !state.maskWidth || !state.maskHeight) return;
  const width = state.maskWidth, height = state.maskHeight;
  if (maskCanvas.width !== width) maskCanvas.width = width;
  if (maskCanvas.height !== height) maskCanvas.height = height;
  const image = maskCtx.createImageData(width, height);
  const data = image.data;
  for (let i = 0; i < state.confidence.length; i++) {
    const value = state.confidence[i];
    const p = i * 4;
    data[p] = 255;
    data[p + 1] = 255;
    data[p + 2] = 255;
    data[p + 3] = alphaFromConfidence(value, state.maskQuality);
  }
  maskCtx.putImageData(image, 0, 0);
  maskRevision++;
  occupancyCache = null;
  state.hasMask = true;
  refreshControls();
  drawFrame();
}

function applySegmentation(result, quality = 'live') {
  const mask = extractPersonMask(result);
  if (!mask) {
    // This means the model returned no usable mask, not that the scene lacks a person.
    if (!state.hasMask) notify('Detection output unavailable. Please reopen the camera.', 5200);
    console.warn('No usable segmentation mask:', {
      confidenceChannels: result?.confidenceMasks?.length ?? 0,
      categoryMask: Boolean(result?.categoryMask),
    });
    return;
  }
  const aligned = state.maskWidth === mask.width && state.maskHeight === mask.height;
  const previous = aligned ? state.previousConfidence : null;
  let guide = null;
  if (quality === 'capture' && analysisCanvas.width === mask.width && analysisCanvas.height === mask.height) {
    try { guide = analysisCtx.getImageData(0, 0, mask.width, mask.height).data; }
    catch (error) { console.warn('Photo edge guide unavailable:', error); }
  }
  const refined = refineConfidence(mask.confidence, mask.width, mask.height, { previous, guide, quality });
  let detectedPixels = 0;
  for (let i = 0; i < refined.length; i += 16) if (refined[i] > 0.61) detectedPixels++;
  if (detectedPixels < 3) {
    clearMask();
    drawFrame();
    return;
  }
  state.maskWidth = mask.width;
  state.maskHeight = mask.height;
  state.maskQuality = quality;
  state.confidence = refined;
  state.previousConfidence = quality === 'live' ? refined : null;
  paintMask();
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

async function setModelMode(mode) {
  await loadModel();
  if (state.modelMode !== mode) {
    await state.model.setOptions({ runningMode: mode });
    state.modelMode = mode;
  }
}

function analyzeSource(source, sourceWidth, sourceHeight, quality = 'live') {
  const maxSide = quality === 'capture' ? MAX_CAPTURE_ANALYSIS_SIDE : MAX_LIVE_ANALYSIS_SIDE;
  const factor = Math.min(1, maxSide / Math.max(sourceWidth, sourceHeight));
  const w = Math.max(2, Math.round(sourceWidth * factor));
  const h = Math.max(2, Math.round(sourceHeight * factor));
  if (analysisCanvas.width !== w) analysisCanvas.width = w;
  if (analysisCanvas.height !== h) analysisCanvas.height = h;
  analysisCtx.drawImage(source, 0, 0, w, h);
}

function tick(now) {
  if (state.mode !== 'camera') return;
  if (elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    drawFrame();
    if (state.model && state.modelMode === 'VIDEO' && !document.hidden &&
      (now - state.lastInferAt) >= state.analysisIntervalMs &&
      elements.video.currentTime !== state.lastVideoTime) {
      state.lastInferAt = now;
      state.lastVideoTime = elements.video.currentTime;
      analyzeSource(elements.video, state.mediaWidth, state.mediaHeight);
      try {
        // segmentForVideo is synchronous. Throttling prevents redundant inference.
        const timestamp = Math.max(Math.floor(now), state.lastTimestamp + 1);
        state.lastTimestamp = timestamp;
        const inferenceStart = performance.now();
        state.model.segmentForVideo(analysisCanvas, timestamp, result => applySegmentation(result, 'live'));
        // Inference is synchronous on Safari. Back off on slower phones rather
        // than permanently freezing the camera to chase an FPS number.
        const cost = performance.now() - inferenceStart;
        state.analysisIntervalMs = Math.min(200, Math.max(MIN_ANALYSIS_INTERVAL_MS, cost * 1.25));
      } catch (error) {
        console.error('Live segmentation error:', error);
        notify('Live segmentation stopped. Try reopening the camera.');
        state.modelMode = 'error';
      }
    }
  }
  state.animationHandle = requestAnimationFrame(tick);
}

async function startCamera() {
  if (state.busy) return;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    notify('Camera needs HTTPS or localhost. Try uploading a photo.', 5500);
    return;
  }
  state.busy = true;
  const session = ++state.session;
  stopVideo();
  releasePhoto();
  clearMask();
  elements.start.disabled = true;
  elements.start.textContent = 'Opening camera…';
  refreshControls();
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: state.facing },
        width: { ideal: 1280 },
        height: { ideal: 960 },
      },
      audio: false,
    });
    if (session !== state.session) return;
    elements.video.srcObject = state.stream;
    await elements.video.play();
    state.mediaWidth = elements.video.videoWidth;
    state.mediaHeight = elements.video.videoHeight;
    if (!state.mediaWidth || !state.mediaHeight) throw new Error('No camera frames received');
    state.mode = 'camera';
    hideEmpty();
    state.lastInferAt = 0;
    state.analysisIntervalMs = 88;
    state.lastVideoTime = -1;
    state.lastTimestamp = 0;
    state.animationHandle = requestAnimationFrame(tick);
    notify('Loading the person detector…', 4800);
    await setModelMode('VIDEO');
    if (session !== state.session) return;
    notify('Ready. Aim at someone.');
  } catch (error) {
    console.error('Cannot start camera:', error);
    const denied = error?.name === 'NotAllowedError' || error?.name === 'PermissionDeniedError';
    notify(denied ? 'Camera permission denied. You can still upload a photo.' :
      'Camera or model unavailable. Try a photo or check your connection.', 6500);
    if (state.mode !== 'camera') {
      stopVideo();
      state.mode = 'idle';
    }
  } finally {
    if (session === state.session) {
      state.busy = false;
      elements.start.disabled = false;
      elements.start.textContent = 'Open camera';
      refreshControls();
    }
  }
}

async function loadPhoto(file) {
  if (!file || !file.type.startsWith('image/')) {
    notify('Choose an image file.');
    return;
  }
  ++state.session;
  stopVideo();
  releasePhoto();
  clearMask();
  state.mode = 'idle';
  state.busy = true;
  refreshControls();
  notify('Preparing your photo…', 5000);
  try {
    let photo;
    try {
      photo = await createImageBitmap(file, { imageOrientation: 'from-image' });
    } catch {
      photo = await loadFallbackImage(file);
    }
    state.photo = photo;
    state.mediaWidth = photo.width;
    state.mediaHeight = photo.height;
    state.mode = 'photo';
    state.frozenEffectTime = performance.now();
    hideEmpty();
    drawFrame();
    await setModelMode('IMAGE');
    analyzeSource(photo, photo.width, photo.height, 'capture');
    state.model.segment(analysisCanvas, result => applySegmentation(result, 'capture'));
    notify(state.hasMask ? 'Edges refined. Tap the shutter to save.' : 'No person detected in this photo.');
  } catch (error) {
    console.error('Cannot segment photo:', error);
    notify('Could not process the image. Check your connection or try JPG/PNG.', 6000);
  } finally {
    state.busy = false;
    refreshControls();
    elements.file.value = '';
  }
}

function loadFallbackImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => { URL.revokeObjectURL(url); resolve(image); };
    image.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Could not decode image')); };
    image.src = url;
  });
}

// Freeze the original sensor frame, not the previously composited preview.
// Capture performs a separate image-mode segmentation pass before a PNG exists.
async function captureFrame() {
  if (state.busy || state.mode !== 'camera') return;
  const session = ++state.session;
  state.busy = true;
  refreshControls();
  try {
    const snapshot = document.createElement('canvas');
    snapshot.width = state.mediaWidth;
    snapshot.height = state.mediaHeight;
    snapshot.getContext('2d', { alpha: false }).drawImage(elements.video, 0, 0, snapshot.width, snapshot.height);
    stopVideo();
    releasePhoto();
    state.mode = 'photo';
    clearMask();
    state.frozenEffectTime = performance.now();
    state.photo = snapshot;
    drawFrame();
    notify('Refining the silhouette…', 6000);
    await setModelMode('IMAGE');
    if (session !== state.session) return;
    analyzeSource(snapshot, snapshot.width, snapshot.height, 'capture');
    // Callback form avoids an extra full mask copy and works on iOS Safari.
    state.model.segment(analysisCanvas, result => applySegmentation(result, 'capture'));
    notify(state.hasMask ? 'Refined. Tap the shutter to save.' : 'No person detected. Try another angle.', 4000);
  } catch (error) {
    console.error('Capture refinement failed:', error);
    notify('Could not refine this image. Return to camera and retry.', 5500);
  } finally {
    if (session === state.session) { state.busy = false; refreshControls(); }
  }
}

async function saveFrame() {
  if (!state.hasMask || state.busy || state.mode !== 'photo') return;
  state.busy = true;
  refreshControls();
  try {
    // Export from the original photo, not the low-res live preview canvas.
    // Keep the same crop and color the user composed in the viewfinder.
    const bounds = elements.canvas;
    const crop = getCrop(state.mediaWidth, state.mediaHeight, bounds.width, bounds.height);
    const scale = Math.min(1, MAX_EXPORT_SIDE / Math.max(crop.width, crop.height));
    const width = Math.max(1, Math.round(crop.width * scale));
    const height = Math.max(1, Math.round(crop.height * scale));
    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = width;
    exportCanvas.height = height;
    const exportCtx = exportCanvas.getContext('2d', { alpha: false });
    const wasComparing = state.comparing;
    state.comparing = false;
    try { drawFrame(exportCanvas, exportCtx); }
    finally { state.comparing = wasComparing; }
    const blob = await new Promise(resolve => exportCanvas.toBlob(resolve, 'image/png'));
    if (!blob) throw new Error('PNG encoder returned no image');
    const file = new File([blob], `FIGURES-${new Date().toISOString().replaceAll(':', '-')}.png`, { type: 'image/png' });
    if (navigator.canShare?.({ files: [file] }) && navigator.share) {
      try {
        await navigator.share({ files: [file], title: 'FIGURES' });
      } catch (error) {
        if (error?.name !== 'AbortError') throw error;
      }
    } else {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = file.name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 30000);
      notify('Image saved.');
    }
  } catch (error) {
    console.error('Saving failed:', error);
    notify('Save failed. Please try again.');
  } finally {
    state.busy = false;
    refreshControls();
  }
}

function setComparing(isComparing) {
  state.comparing = isComparing;
  if (state.mode === 'photo') drawFrame();
}

function initEvents() {
  elements.start.addEventListener('click', startCamera);
  elements.uploadStart.addEventListener('click', () => elements.file.click());
  elements.upload.addEventListener('click', () => elements.file.click());
  elements.file.addEventListener('change', event => loadPhoto(event.target.files?.[0]));
  elements.backToCamera.addEventListener('click', startCamera);
  elements.flip.addEventListener('click', () => {
    if (state.busy) return;
    state.facing = state.facing === 'environment' ? 'user' : 'environment';
    startCamera();
  });
  elements.shutter.addEventListener('click', () => {
    if (state.mode === 'camera') captureFrame();
    else if (state.mode === 'photo') saveFrame();
  });
  elements.compare.addEventListener('pointerdown', event => {
    if (elements.compare.disabled) return;
    elements.compare.setPointerCapture(event.pointerId);
    setComparing(true);
  });
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture', 'pointerleave']) {
    elements.compare.addEventListener(type, () => setComparing(false));
  }
  elements.compare.addEventListener('keydown', event => {
    if (event.key === ' ' || event.key === 'Enter') setComparing(true);
  });
  elements.compare.addEventListener('keyup', () => setComparing(false));
  for (const button of elements.colorButtons) {
    button.addEventListener('click', () => {
      state.color = button.dataset.color;
      if (state.confidence) paintMask();
      refreshControls();
    });
  }
  elements.info.addEventListener('click', () => { elements.modal.hidden = false; });
  elements.closeAbout.addEventListener('click', () => { elements.modal.hidden = true; });
  elements.modal.addEventListener('click', event => {
    if (event.target === elements.modal) elements.modal.hidden = true;
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') elements.modal.hidden = true;
  });
  window.addEventListener('pagehide', stopVideo);
  window.addEventListener('resize', syncCanvasResolution);
}

initEvents();
syncCanvasResolution();
refreshControls();
