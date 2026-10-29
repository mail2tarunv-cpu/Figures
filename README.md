# FIGURES — Living Pixels

A camera experiment that turns people into **floating confidence-driven pixel figures**, keeping the surrounding photo untouched. Designed as a fullscreen desktop camera workspace on MacBook, with a responsive iPhone layout and no account required.

## Implemented

- **P0 · Confidence density:** Sample MediaPipe's *raw* per-pixel person probabilities (before alpha thresholding). Dense bright pixels occupy high-confidence body regions; ambiguous edges become scattered and translucent instead of making a hard incorrect cutout.
- **P0 · Independent animation:** 2D Canvas renders continuously (target ~30 fps) even when segmentation runs more slowly (~5–14 analyses/s depending on phone). Slowly moving color waves and subpixel drifts have a stable time clock.
- **P1 · Motion Echo:** Store sparse historical figure-grid samples only when the segmented person's center moves. Echo particles float up and right and fade in under one second; capped history avoids unlimited memory growth. Reduced Motion disables floating/trails.
- **P1 · Record:** A prominent **Record video** control records the composited canvas silently with `captureStream()` + `MediaRecorder` on supporting browsers. Tap once to start and again to stop, with a 60-second safety limit to bound browser memory. Review the clip, then save/share MP4 or WebM depending on browser. If the device doesn't support these APIs, photo capture remains available. Recording is local; browser recording format/quality may vary. iOS Safari testing is still required.
- **P2 · Lightweight figure association:** Connected components in the segmentation confidence grid are matched across successive analyses by motion-predicted centroid and size. Identity IDs drive visual phase continuity and trigger trails. **This is not a true individual-person detector**: close, overlapping or occluded people merge and IDs may reset. Improving that requires a true instance segmentation/tracking model.
- Full-browser desktop viewfinder with floating camera controls, a dedicated fullscreen action, five palettes, hold-to-compare, the #121212 isolated background, uploaded-photo refinement, and photo PNG sharing/saving. The original compact layout remains on narrow screens.
- Original camera imagery remains on device. MediaPipe JS/WASM and the selfie segmentation model download from their public CDNs on first use.

## Start

```bash
python3 -m http.server 8000
```

Open http://localhost:8000 on a laptop or deploy the site to **HTTPS** for mobile camera permission. Tap **Open camera**, then use the shutter to capture a still (tap again to save), or **Record video** to start a silent recording. Tap **Stop recording** to finish, review it, and select **Save video**. On MacBook, use **Fullscreen** for an immersive browser view. Recording is capped at 60 seconds; browser codec support determines MP4/WebM output. Some browsers only offer download rather than the native share sheet.

## Architecture

- `app.js` — controls, MediaPipe selfie segmenter, adaptive inference, animation, recording, save/share
- `living-pixels.mjs` — pure confidence sampling, opacity, connected components, lightweight centroid association, sparse echo points
- `person-mask.mjs` — normalizes MediaPipe foreground outputs, single/dual channel plus category fallback
- `mask-refinement.mjs` — temporal smoothing, isolated-noise removal, guided photo edge refinement
- `styles.css`, `index.html` — camera-first UI
- `tests/*.test.mjs` — Node regression tests

```bash
node --test tests/*.test.mjs
node --check app.js
```

## Known limits

1. The MediaPipe selfie segmenter's native input is ~256×256. Enlarging the camera frame cannot generate missing hair, shoe or distant-person boundaries. Pixels are intentionally expressive rather than a perfect Photoshop cutout.
2. A single foreground mask cannot separate overlapping people. Our component tracker only associates already-separated regions.
3. Canvas + MediaRecorder support, codecs, FPS, battery and thermal throttling vary across iPhones. Confirm on-device before claiming production quality; test MP4/WebM output on the target iOS version.
4. Video recording is silent, capped to 60 seconds, and stops if the browser tab goes into the background. This is canvas output with the FIGURES pixel effect; it is not a screen recording.

## P3 · LiDAR / native spatial pixels — not part of the web build

Browser Safari has no standard API to access live iPhone LiDAR scene-depth buffers. A real depth-aware version would require a native iOS app. Proposed pipeline:

- Swift + ARKit `ARWorldTrackingConfiguration`, device capability checks for `.sceneDepth` / `.smoothedSceneDepth`
- ARKit `personSegmentationWithDepth` where supported, or Vision/Core ML person segmentation
- Register depth and RGB images; reject invalid/far depth samples, estimate per-tile world-space positions
- Metal instanced particles with depth-based parallax, occlusion, motion trails and camera motion compensation
- Fallback to current 2D treatment when LiDAR or person-segmentation-with-depth isn't supported

LiDAR *does not replace* high-quality person segmentation, and only supported iPhone Pro/iPad Pro devices have the sensor. Validate on devices before choosing native migration.
