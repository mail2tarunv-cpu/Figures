/**
 * FIGURES — Experiment 001
 * A compact, client-side live camera / photo segmenter.
 * MediaPipe performs on-device inference; camera imagery is not sent to our servers.
 * The JS/WASM bundle and model are fetched from public CDNs on first use.
 */

import { extractPersonMask } from './person-mask.mjs';
import { refineConfidence } from './mask-refinement.mjs';
import { buildConfidenceTiles, tileOpacity, tileHash, findFigures, trackFigures, makeEchoPoints } from './living-pixels.mjs';

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
const RECORD_DURATION_MS = 8000;
const ECHO_LIFETIME_MS = 950;
const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)');

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
  backdropToggle: document.querySelector('#backdropToggle'),
  record: document.querySelector('#recordButton'),
  recordClock: document.querySelector('#recordClock'),
  previewModal: document.querySelector('#videoPreviewModal'),
  previewVideo: document.querySelector('#clipPreview'),
  saveClip: document.querySelector('#saveClipButton'),
  closeClip: document.querySelector('#closeClipButton'),
  colorButtons: document.querySelectorAll('.swatch'),
  info: document.querySelector('#infoButton'),
  modal: document.querySelector('#aboutModal'),
  closeAbout: document.querySelector('#closeAbout'),
};

const ctx = elements.canvas.getContext('2d', { alpha: false });
const analysisCanvas = document.createElement('canvas');
const analysisCtx = analysisCanvas.getContext('2d', { willReadFrequently: true });
const guideCanvas = document.createElement('canvas');
const guideCtx = guideCanvas.getContext('2d', { willReadFrequently: true });
const overlayCanvas = document.createElement('canvas');
const overlayCtx = overlayCanvas.getContext('2d', { alpha: true });
// Occupancy is computed from the unclipped probabilities, not the binary mask.
const recording = { recorder: null, stream: null, chunks: [], timeout: null, startedAt: 0, blob: null, url: null, canceled: false };
let maskRevision = 0;
let occupancyCache = null;

const state = {
  mode: 'idle',                    // idle | camera | photo
  facing: 'environment',
  color: '#F5FC66',
  blackBackground: false,
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
  lastRenderAt: 0,
  tracks: [],
  nextTrackId: 1,
  echoes: [],
  lastEchoAt: 0,
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
  const isRecording = recording.recorder?.state === 'recording';
  elements.flip.disabled = state.mode !== 'camera' || state.busy || isRecording;
  elements.upload.disabled = state.busy || isRecording;
  elements.record.hidden = state.mode !== 'camera';
  elements.record.disabled = state.mode !== 'camera' || state.busy || (!isRecording && !supportsVideoRecording());
  elements.record.classList.toggle('is-recording', isRecording);
  elements.record.setAttribute('aria-pressed', String(isRecording));
  elements.record.setAttribute('aria-label', isRecording ? 'Stop video recording' : 'Record an 8-second clip');
  elements.shutter.disabled = !active || state.busy || isRecording || (state.mode === 'photo' && !state.hasMask);
  elements.shutter.setAttribute('aria-label', state.mode === 'camera' ? 'Take photo and refine edges' : 'Save edited photo');
  elements.shutter.classList.toggle('save-mode', state.mode === 'photo');
  const sceneTip = elements.viewBottom.querySelector('.scene-tip');
  if (sceneTip) sceneTip.textContent = state.mode === 'photo' ? 'TAP ↓ TO SAVE' : (isRecording ? 'RECORDING · 8 SEC MAX' : 'LIVING PIXELS');
  elements.compare.disabled = !active || !state.hasMask || state.busy;
  elements.backToCamera.disabled = state.mode !== 'photo' || state.busy;
  elements.compare.hidden = !active || !state.hasMask;
  elements.backToCamera.hidden = state.mode !== 'photo';
  // The toggle remains usable during a recording so the change is captured.
  elements.backdropToggle.hidden = !active;
  elements.backdropToggle.disabled = !active || state.busy;
  elements.backdropToggle.setAttribute('aria-pressed', String(state.blackBackground));
  elements.backdropToggle.classList.toggle('is-active', state.blackBackground);
  elements.backdropToggle.title = state.blackBackground ? 'Show original background' : 'Make the background black';
  for (const button of elements.colorButtons) {
    const selected = button.dataset.color === state.color;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  }
}

function stopVideo() {
  if (recording.recorder?.state === 'recording') stopRecording(true);
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
  state.tracks = []; state.nextTrackId = 1;
  state.echoes = []; state.lastEchoAt = 0;
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
  // Show the original viewfinder until a person is detected, so users can
  // still frame their shot. Holding Compare always reveals the raw scene.
  // This single compositor path feeds live preview, PNGs, and video recording.
  if (state.blackBackground && state.hasMask && !state.comparing) {
    targetContext.fillStyle = '#000000';
    targetContext.fillRect(0, 0, w, h);
  } else {
    targetContext.drawImage(source, x, y, width, height, 0, 0, w, h);
  }

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
  const cell = Math.max(12, Math.round(Math.min(width, height) / 38));
  const cols = Math.ceil(width / cell), rows = Math.ceil(height / cell);
  const key = `${maskRevision}:${width}:${height}:${cell}:${crop.mx.toFixed(3)}:${crop.my.toFixed(3)}:${crop.mw.toFixed(3)}:${crop.mh.toFixed(3)}`;
  if (occupancyCache?.key === key) return occupancyCache;
  const previous = occupancyCache;
  const field = buildConfidenceTiles(state.confidence, state.maskWidth, state.maskHeight, cols, rows, crop);
  const components = findFigures(field, cols, rows, .46, 5);
  const now = performance.now();
  const tracked = trackFigures(state.mode === 'camera' ? state.tracks : [], components, state.nextTrackId, now);
  const identities = new Int32Array(cols * rows);
  for (const [index, id] of tracked.idsByCell) identities[index] = id;
  if (state.mode === 'camera') {
    // Capture only actual movement, not an echo of an unmoving figure.
    if (previous && previous.width === width && previous.height === height &&
        tracked.movement > .008 && now - state.lastEchoAt > 135 && !(reducedMotion?.matches)) {
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
      const alpha = (1 - age) ** 1.6 * .27;
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
  const gap = Math.max(1, Math.round(cell * .09));
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
    const edge = c < .72;
    const pixelSize = Math.max(3, Math.round((cell - gap) * (edge ? .56 + .24 * c : .96)));
    const floating = motionless ? 0 : 1;
    const xDrift = floating * (Math.sin(seconds * .78 + y * .29 + hash * 6) * cell * .14 + seconds % 2 * cell * .015);
    const yDrift = floating * Math.cos(seconds * .62 + x * .35 + hash * 4) * cell * .15;
    const inset = (cell - pixelSize) / 2;
    overlayCtx.globalAlpha = opacity;
    overlayCtx.fillRect(Math.round(x * cell + inset + xDrift), Math.round(y * cell + inset + yDrift), pixelSize, pixelSize);
  }
  overlayCtx.globalAlpha = 1;
}

function paintMask() {
  if (!state.confidence || !state.maskWidth || !state.maskHeight) return;
  // No texture upload per inference: the particle renderer samples the raw
  // confidence field directly and runs independently in requestAnimationFrame.
  maskRevision++;
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
  if (quality === 'capture') {
    try {
      // The model produces a smaller mask than the photo: align the RGB guide to
      // the MASK, otherwise the matting step would never actually run.
      guideCanvas.width = mask.width; guideCanvas.height = mask.height;
      guideCtx.drawImage(analysisCanvas, 0, 0, mask.width, mask.height);
      guide = guideCtx.getImageData(0, 0, mask.width, mask.height).data;
    } catch (error) { console.warn('Photo edge guide unavailable:', error); }
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
    // Renderer remains smooth between relatively expensive AI segmentations.
    if (now - state.lastRenderAt >= 32) { drawFrame(); state.lastRenderAt = now; }
    if (recording.recorder?.state === 'recording') updateRecordingClock(now);
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
    state.lastInferAt = 0; state.lastRenderAt = 0;
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

function supportsVideoRecording() {
  return typeof MediaRecorder !== 'undefined' && typeof elements.canvas.captureStream === 'function';
}

function preferredVideoMime() {
  if (!supportsVideoRecording()) return null;
  const types = ['video/mp4;codecs=avc1.42E01E', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  return types.find(type => MediaRecorder.isTypeSupported(type)) || '';
}

function updateRecordingClock(now = performance.now()) {
  if (recording.recorder?.state !== 'recording') return;
  const elapsed = Math.min(RECORD_DURATION_MS, now - recording.startedAt);
  const secs = Math.ceil((RECORD_DURATION_MS - elapsed) / 1000);
  elements.recordClock.textContent = `${secs}s left`;
  if (elapsed >= RECORD_DURATION_MS) stopRecording();
}

function startRecording() {
  if (state.mode !== 'camera' || state.busy || recording.recorder?.state === 'recording') return;
  if (!supportsVideoRecording()) { notify('This browser cannot record canvas video. Try Safari or Chrome.'); return; }
  try {
    closeVideoPreview();
    const stream = elements.canvas.captureStream(30);
    if (!stream.getVideoTracks().length) throw new Error('Canvas video track unavailable');
    const mimeType = preferredVideoMime();
    const recorder = new MediaRecorder(stream, { ...(mimeType ? { mimeType } : {}), videoBitsPerSecond: 2_400_000 });
    recording.recorder = recorder;
    recording.stream = stream;
    recording.canceled = false;
    recording.chunks = [];
    recording.startedAt = performance.now();
    recorder.ondataavailable = event => { if (event.data?.size) recording.chunks.push(event.data); };
    recorder.onerror = event => { console.error('Video recorder error', event); notify('Recording failed on this browser.'); stopRecording(true); };
    recorder.onstop = () => {
      const cancelled = recording.canceled;
      const chunks = [...recording.chunks];
      const type = recorder.mimeType || mimeType || 'video/mp4';
      recording.stream?.getTracks().forEach(track => track.stop());
      recording.stream = null;
      if (recording.timeout) clearTimeout(recording.timeout);
      recording.timeout = null;
      recording.recorder = null;
      recording.chunks = [];
      elements.recordClock.hidden = true;
      refreshControls();
      if (cancelled) return;
      if (!chunks.length || chunks.every(blob => !blob.size)) {
        notify('The browser recorded no frames. Try again.'); return;
      }
      recording.blob = new Blob(chunks, { type });
      recording.url = URL.createObjectURL(recording.blob);
      elements.previewVideo.src = recording.url;
      elements.previewModal.hidden = false;
    };
    // Start in direct response to the user's tap (Safari user activation).
    recorder.start(250);
    recording.timeout = setTimeout(() => stopRecording(), RECORD_DURATION_MS);
    elements.recordClock.hidden = false;
    updateRecordingClock();
    refreshControls();
  } catch (error) {
    console.error('Could not record the processed canvas', error);
    recording.stream?.getTracks().forEach(track => track.stop());
    recording.stream = null; recording.recorder = null;
    notify('Video recording unavailable on this device. Photos still work.', 5000);
    refreshControls();
  }
}

function stopRecording(discard = false) {
  if (recording.recorder?.state !== 'recording') return;
  recording.canceled = discard;
  try { recording.recorder.stop(); } catch (error) { console.warn('Stopping recorder failed', error); }
  if (recording.timeout) clearTimeout(recording.timeout);
  recording.timeout = null;
  elements.recordClock.hidden = true;
  refreshControls();
}

function closeVideoPreview() {
  elements.previewModal.hidden = true;
  elements.previewVideo.pause();
  elements.previewVideo.removeAttribute('src');
  elements.previewVideo.load();
  if (recording.url) URL.revokeObjectURL(recording.url);
  recording.url = null; recording.blob = null;
}

async function saveRecordedClip() {
  const blob = recording.blob;
  if (!blob) return;
  const ext = blob.type.includes('webm') ? 'webm' : 'mp4';
  const file = new File([blob], `FIGURES-${new Date().toISOString().replaceAll(':','-')}.${ext}`, { type: blob.type });
  if (navigator.canShare?.({ files: [file] }) && navigator.share) {
    try { await navigator.share({ files: [file], title: 'FIGURES · Living Pixels' }); return; }
    catch (error) { if (error?.name === 'AbortError') return; console.warn('File sharing unavailable, downloading', error); }
  }
  const link = document.createElement('a');
  link.href = recording.url; link.download = file.name;
  document.body.appendChild(link); link.click(); link.remove();
  notify('Clip ready to save.');
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
  elements.record.addEventListener('click', () => recording.recorder?.state === 'recording' ? stopRecording() : startRecording());
  elements.closeClip.addEventListener('click', closeVideoPreview);
  elements.saveClip.addEventListener('click', saveRecordedClip);
  elements.previewModal.addEventListener('click', event => { if (event.target === elements.previewModal) closeVideoPreview(); });
  elements.shutter.addEventListener('click', () => {
    if (state.mode === 'camera') captureFrame();
    else if (state.mode === 'photo') saveFrame();
  });
  elements.backdropToggle.addEventListener('click', () => {
    if (state.mode === 'idle' || state.busy) return;
    state.blackBackground = !state.blackBackground;
    refreshControls();
    drawFrame();
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
  window.addEventListener('pagehide', () => { stopRecording(true); stopVideo(); closeVideoPreview(); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stopRecording(); });
  window.addEventListener('resize', syncCanvasResolution);
}

initEvents();
syncCanvasResolution();
refreshControls();
