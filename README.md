# FIGURES — Living Pixels

A **minimal living-pixel experience**: people become floating pixel figures on a permanent **#121212** background. Use either a webcam or a local video file. Neither source's original pixels are composited into the displayed scene.

## Experience

1. Open [FIGURES](https://figures-delta.vercel.app). Select **Start camera** for the live webcam, or **Import video** to select a local MP4, MOV, M4V or WebM clip.
2. The device runs MediaPipe person segmentation locally. The figure becomes floating pixels, with short motion echoes. Imported videos play in a loop and are **contained** within the dark stage instead of cropped, preserving the full actor.
3. Use the **Color** control at the bottom to drag through saturation/brightness, scrub the hue spectrum, or choose a preset. Colors change immediately.
4. With a video playing, click **Import video** in the top-right to replace the clip. Only one source plays at a time.

This is **viewing-only**. Importing a video is local-only: there is **no uploading to a server, photo capture, processed-video export, recording, sharing, or background toggle**. The background is always #121212, including while no person is detected. The generated pixel version can be previewed but is not yet saved as an MP4.

Mac shortcuts: **C** opens or closes the color picker, **F** enters or exits fullscreen, **Esc** closes the picker. On mobile, tap the color control. Webcam permission requires HTTPS or localhost. MP4 with H.264 is the most widely compatible format; MOV and WebM depend on browser and codec support.

## Technology

- MediaPipe ImageSegmenter Selfie Segmenter (TensorFlow Lite model, running via the browser's WASM CPU backend).
- Confidence-driven particles: more solid near strongly detected body regions, sparse and floating at uncertain mask edges.
- 2D Canvas rendering on a separate animation clock; person segmentation runs at an adaptive interval.
- Lightweight centroid-based association for separated figures, with short motion trails.
- Five palette presets plus continuous HSV color selection (hue, saturation, brightness).
- Camera and imported video files stay local. The required MediaPipe runtime/model download from external CDNs on first use; there is no app-server video upload.

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
- Safari and Chrome webcam permission, imported-video decoding and inference performance vary across devices. Test an actual imported MP4 on MacBook before assessing final visual quality.
- No LiDAR: browser implementations do not expose iPhone LiDAR depth data as a general-purpose camera API.
