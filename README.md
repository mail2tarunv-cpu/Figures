# FIGURES — Living Pixels

A **single-purpose live visual experience**: people become floating pixel figures on a permanent **#121212** background. No camera image is ever composited into the output.

## Experience

1. Open [FIGURES](https://figures-delta.vercel.app) and select **Start camera**.
2. Grant webcam access. The device runs MediaPipe person segmentation locally; once a figure is detected its pixels drift, shimmer, and leave subtle motion echoes.
3. Click the small **Color** control at the bottom. Drag inside the saturation/brightness field, scrub the hue spectrum, or choose a preset. Colors change instantly without resetting tracking.
4. Close the picker and just watch.

This is intentionally **viewing-only**. There is **no photo capture, video recording, uploading, sharing, or background toggle**. The background is always #121212, including while no person is detected.

Mac shortcuts: **C** opens or closes the color picker, **F** enters or exits fullscreen, **Esc** closes the picker. On mobile, tap the color control. Browser permission requires HTTPS or localhost.

## Technology

- MediaPipe ImageSegmenter Selfie Segmenter (TensorFlow Lite model, running via the browser's WASM CPU backend).
- Confidence-driven particles: more solid near strongly detected body regions, sparse and floating at uncertain mask edges.
- 2D Canvas rendering on a separate animation clock; person segmentation runs at an adaptive interval.
- Lightweight centroid-based association for separated figures, with short motion trails.
- Five palette presets plus continuous HSV color selection (hue, saturation, brightness).
- Camera stays local. The required MediaPipe runtime/model download from external CDNs on first use; there is no app-server camera upload.

## Local development

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000`. The site is static HTML, CSS, JavaScript and .mjs modules.

```bash
node --check app.js
node --test tests/*.test.mjs
```

## Current limitations

- The selfie segmentation model has roughly 256px-native mask resolution. Fine hair, distant pedestrians and overlapping people can still fail.
- Separate person tracking is approximate, not individual identity tracking.
- Segmentation is CPU-bound; the renderer stays fluid between inference frames but heavier computers or high-res monitors may experience dropped frames.
- Safari and Chrome webcam permission and inference performance vary across devices. Test on the actual MacBook and iPhone before assessing final visual quality.
- No LiDAR: browser implementations do not expose iPhone LiDAR depth data as a general-purpose camera API.
