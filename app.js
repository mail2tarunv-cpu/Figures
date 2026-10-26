/**
 * FIGURES — Experiment 001
 * A compact, client-side live camera / photo segmenter.
 * MediaPipe performs on-device inference; camera imagery is not sent to our servers.
 * The JS/WASM bundle and model are fetched from public CDNs on first use.
 */

import { extractPersonMask } from './person-mask.mjs';

const MEDIAPIPE_VERSION = '1.1.1';
const PACKAGE_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite';
const OUTPUT_HEIGHT = 1200;
const MAX_ANALYSIS_SIDE = 512;
const ANALYSIS_INTERVAL_MS = 95;

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
  maskWidth: 0,
  maskHeight: 0,
  hasMask: false,
  comparing: false,
  busy: false,
  animationHandle: null,
  lastInferAt: 0,
  lastVideoTime: -1,
  lastTimestamp: 0,
  session: 0,
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
  elements.shutter.disabled = !active || !state.hasMask;
  elements.compare.disabled = !active || !state.hasMask;
  elements.backToCamera.disabled = state.mode !== 'photo' || state.busy;
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
  state.hasMask = false;
  state.maskWidth = state.maskHeight = 0;
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

function drawFrame() {
  if (state.mode === 'idle') return;
  const source = state.mode === 'camera' ? elements.video : state.photo;
  if (!source || !state.mediaWidth || !state.mediaHeight) return;

  const { width: w, height: h } = elements.canvas;
  const { x, y, width, height } = getCrop(state.mediaWidth, state.mediaHeight, w, h);
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  if (state.mode === 'camera' && state.facing === 'user') {
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
  }
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, x, y, width, height, 0, 0, w, h);

  if (state.hasMask && !state.comparing) {
    // The segmentation image covers the *whole source frame*, not just its crop.
    // Matching source and mask crops prevents mismatched silhouettes.
    const mx = x / state.mediaWidth * state.maskWidth;
    const my = y / state.mediaHeight * state.maskHeight;
    const mw = width / state.mediaWidth * state.maskWidth;
    const mh = height / state.mediaHeight * state.maskHeight;
    ctx.drawImage(maskCanvas, mx, my, mw, mh, 0, 0, w, h);
  }
  ctx.restore();
}

function hexRgb(color) {
  return [parseInt(color.slice(1, 3), 16), parseInt(color.slice(3, 5), 16), parseInt(color.slice(5, 7), 16)];
}

function paintMask() {
  if (!state.confidence || !state.maskWidth || !state.maskHeight) return;
  const [r, g, b] = hexRgb(state.color);
  const width = state.maskWidth, height = state.maskHeight;
  if (maskCanvas.width !== width) maskCanvas.width = width;
  if (maskCanvas.height !== height) maskCanvas.height = height;
  const image = maskCtx.createImageData(width, height);
  const data = image.data;
  for (let i = 0; i < state.confidence.length; i++) {
    const value = state.confidence[i];
    // A narrow, anti-aliased confidence ramp: almost solid, slightly softened edges.
    const t = Math.max(0, Math.min(1, (value - 0.38) / 0.32));
    const smooth = t * t * (3 - 2 * t);
    const p = i * 4;
    data[p] = r;
    data[p + 1] = g;
    data[p + 2] = b;
    data[p + 3] = Math.round(smooth * 255);
  }
  maskCtx.putImageData(image, 0, 0);
  state.hasMask = true;
  refreshControls();
  drawFrame();
}

function applySegmentation(result) {
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
  state.maskWidth = mask.width;
  state.maskHeight = mask.height;
  state.confidence = mask.confidence;
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

function analyzeSource(source, sourceWidth, sourceHeight) {
  const factor = Math.min(1, MAX_ANALYSIS_SIDE / Math.max(sourceWidth, sourceHeight));
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
      (now - state.lastInferAt) >= ANALYSIS_INTERVAL_MS &&
      elements.video.currentTime !== state.lastVideoTime) {
      state.lastInferAt = now;
      state.lastVideoTime = elements.video.currentTime;
      analyzeSource(elements.video, state.mediaWidth, state.mediaHeight);
      try {
        // segmentForVideo is synchronous. Throttling prevents redundant inference.
        const timestamp = Math.max(Math.floor(now), state.lastTimestamp + 1);
        state.lastTimestamp = timestamp;
        state.model.segmentForVideo(analysisCanvas, timestamp, applySegmentation);
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
      elements.start.innerHTML = 'Open camera <span aria-hidden="true">↗</span>';
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
    hideEmpty();
    drawFrame();
    await setModelMode('IMAGE');
    analyzeSource(photo, photo.width, photo.height);
    state.model.segment(analysisCanvas, applySegmentation);
    notify('Photo ready. Choose a color and save.');
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

async function saveFrame() {
  if (!state.hasMask) return;
  const originallyComparing = state.comparing;
  state.comparing = false;
  drawFrame();
  state.comparing = originallyComparing;
  const blob = await new Promise(resolve => elements.canvas.toBlob(resolve, 'image/png'));
  if (!blob) return notify('Could not save the image.');
  const file = new File([blob], `FIGURES-${new Date().toISOString().replaceAll(':', '-')}.png`, { type: 'image/png' });
  try {
    if (navigator.canShare?.({ files: [file] }) && navigator.share) {
      await navigator.share({ files: [file], title: 'FIGURES' });
      notify('Image shared.');
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
    if (error?.name !== 'AbortError') {
      console.error('Saving failed:', error);
      notify('Save failed. Try again.');
    }
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
  elements.shutter.addEventListener('click', saveFrame);
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