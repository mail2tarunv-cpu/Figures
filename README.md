# FIGURES — Pixel Studio

FIGURES transforms people into living pixel figures on a permanent **#121212** background. The minimal studio separates a **fixed 9:16 portrait frame** from all controls. The browser preview and captured file are rendered from the same dedicated **1080 × 1920** canvas, so buttons, labels, and the color picker are never recorded.

## Use

1. Open [FIGURES](https://figures-delta.vercel.app) on a MacBook or another device with a supported browser.
2. Choose **Live camera** to request webcam permission, or **Import video** to open a local MP4, MOV, M4V, or WebM. Source switching stops the previous stream.
3. The person becomes stable, dense pixels. Adjust **Color** with a hue slider, a draggable saturation/brightness square, or presets. The figure sits on a dark #121212 background.
4. **Live camera:** choose **Record**, then **Stop recording**. The resulting **silent 9:16** video downloads locally. It stops automatically after **60 seconds**.
5. **Imported video:** choose **Download video**. FIGURES processes the entire video from the beginning and downloads a silent **9:16** video, up to **2 minutes** long. Leave the tab visible throughout the export.

The capture frame has no overlaid UI. Live camera video uses a centered 9:16 crop (to fill the portrait frame). Imported clips preserve their full original composition inside the 9:16 output (unused space is #121212).

## Browser formats and privacy

Exports use the best supported local MediaRecorder format: **MP4** when the browser provides it; otherwise **WebM**. The downloaded format is determined by browser/codec support rather than the source file extension. Audio is intentionally omitted. All camera and imported-video frames remain **on the user's device**; only the model/runtime assets are downloaded from MediaPipe-hosted infrastructure.

This is not a full offline renderer: imported clips are recorded at the browser's real-time playback speed, so slow/overloaded machines can drop rendered frames. See the limitations below.

## Tech

- MediaPipe ImageSegmenter (Selfie Segmenter float16 TFLite model) with browser-side CPU/WASM inference.
- Confidence sampling, spatial cleanup, stable denser tiles, temporal confidence stabilization, and constrained living-pixel shimmer.
- Separate segmentation and animation clocks; live and imported videos use the **same portrait output canvas**.
- Camera video is center-cropped in 9:16; imported-video content is contained instead.
- A canvas video track is encoded via the browser's MediaRecorder and saved locally.

## Local development

```sh
python3 -m http.server 8000
```

Open `http://localhost:8000`, then run tests with Node:

```sh
node --check app.js
node --test tests/*.test.mjs
```

## Known limits

- The selfie model's native segmentation resolution is far below 1080 × 1920; a higher-resolution output cannot invent exact hair/finger edges.
- Processing and encoding are hardware- and browser-dependent. Check real exports in Chrome and Safari on an actual MacBook before relying on the final film.
- The browser cannot guarantee exact 30-fps delivery under heavy processing. Imported files are processed as they play, rather than with frame-accurate offline transcoding.
- Multiple overlapping people are not individually tracked reliably; a segmentation mask can merge them.
- Camera capture requires a secure origin (HTTPS or localhost).
