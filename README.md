# FIGURES — camera experiment 001

A working browser MVP for the original idea: **turn people into solid-color silhouettes while preserving the photographic scene**. The application is intentionally built around one filter, with a small clean engine that can support more experiments later.

## What it can do

- Live camera feed (starts on rear camera when available)
- Real-time person vs. background segmentation
- Solid-color silhouettes with 5 different presets
- Upload and process an existing photo
- Press-and-hold comparison with the original
- Flip front/rear camera
- Save/share the processed image as PNG
- Runs inference on-device. No custom server, account, or photo upload endpoint.

**Important:** This version uses *binary person segmentation*. It paints *all people the same color*. Per-person colors, tracking, video capture, and halftone mode are intentionally out of scope. The model is optimized for selfie/person segmentation and may struggle with very distant pedestrians, overlapping figures, or motion blur.

## Run locally

1. Unzip the project and open Terminal inside the `figures-camera` directory.
2. Start a local HTTP server:

   ```bash
   python3 -m http.server 8000
   ```

3. Open `http://localhost:8000` in Chrome, Safari, or another modern browser.
4. Select **Open camera** and grant permission, or upload a photograph.

For mobile phones, deploy the three web files (`index.html`, `styles.css`, `app.js`) to **an HTTPS website**, e.g., Vercel/Netlify. Merely opening your laptop's `http://192.168...` URL on a phone will not grant camera access because it is not a secure context. Browsers require HTTPS or localhost for camera permission.

## Dependencies

- `@mediapipe/tasks-vision@1.1.1` — browser JS/WASM loaded from jsDelivr
- Google `selfie_segmenter` float16 model — loaded from Google Cloud Storage

Internet access is required to load the library and model, especially on first use. The app does not include any tracking analytics code; MediaPipe may send technical usage metrics according to Google's package privacy notice. Image processing happens locally in the browser.

## File map

- `index.html` — app UI and interaction controls
- `styles.css` — responsive camera-first visual design
- `app.js` — camera, segmentation, canvas compositor, save/share

## What to test on an actual device

- Rear camera resolution/orientation on an iPhone
- How closely the masks follow walking people; notably feet and hair
- Mask jitter while walking with the camera
- Low-light performance
- Safari's share sheet vs. Chrome PNG download

## Future filter architecture

`applySegmentation()` receives the person confidence mask; `paintMask()` is the renderer. Other filters (halftone, negative-space, multiple colors) can be implemented in new renderers without rebuilding the camera/photo pipeline. Distinct colors for each person would require a separate instance-segmentation or person-tracking pipeline.