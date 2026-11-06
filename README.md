# PixelLab – Digital Image Processing Studio

A responsive web dashboard for classic digital image processing. The browser (HTML, CSS,
JavaScript, Bootstrap 5) only collects input and shows results. **Every operation runs in the
Flask back end with OpenCV and NumPy**; Pillow validates and decodes uploads.

## Features

- Upload by file picker or drag and drop, with validation of type, content and size
- Original and processed images side by side, click either to open it full size
- 26 operations with parameter controls (sliders, number boxes, switches, drop-downs)
- **Sequential processing**: each operation is applied to the previous result
- History panel, **Undo**, **Reset to original**, and **Download** as PNG, JPEG or WebP
- Histogram plots (RGB and grayscale, original vs processed) with mean, std, min, max, median
- RGB channel separation (tinted or grayscale) and an "Extract RGB channel" operation
- Light and dark theme, keyboard-accessible drop zone, mobile-friendly layout

### Operations

| Category | Operations |
|---|---|
| Basic | Grayscale, Negative |
| Geometric | Resize (percent or exact size, 5 interpolation modes), Rotate (any angle, expand canvas), Flip |
| Intensity | Brightness, Contrast, Gamma correction, Logarithmic transform, Histogram equalization (global or CLAHE) |
| Filtering | Gaussian blur, Median blur, Mean filter, Sharpen (unsharp mask or kernel), Denoise (Non-Local Means or bilateral) |
| Edge Detection | Sobel (X, Y or magnitude), Canny, Laplacian |
| Thresholding | Binary, Inverse binary (both optionally Otsu), Adaptive (mean or Gaussian) |
| Morphology | Erosion, Dilation, Opening, Closing (rectangle, ellipse or cross kernel; iterations) |
| Color Channels | Extract Red, Green or Blue channel |

## Project structure

```
dip_app/
├── app.py               Flask app factory, all routes, error handling, security headers
├── image_ops.py         The 26 operations + parameter specs (single source of truth)
├── validators.py        Operation-name and parameter validation against those specs
├── image_io.py          Upload validation/decoding (Pillow), encoding, Unicode-safe file I/O
├── storage.py           Per-image workspace on disk: original, every step, undo, cleanup
├── analysis.py          Histogram rendering (OpenCV drawing) and RGB channel separation
├── config.py            Limits and settings (upload size, dimensions, step limit, TTL)
├── errors.py            ApiError / ValidationError
├── generate_sample.py   Creates samples/sample.png for testing
├── requirements.txt
├── templates/index.html
├── static/css/style.css
├── static/js/script.js
├── tests/test_app.py    41 automated tests (standard library unittest)
└── .vscode/             settings.json and launch.json for VS Code
```

`instance/` is created at run time (session secret and temporary image workspaces).

## Setup on Windows with VS Code

1. **Install Python 3.10 or newer** from <https://www.python.org/downloads/>. On the first
   installer screen tick **Add python.exe to PATH**. Check it in a new terminal: `py --version`.
2. **Install VS Code** and, in its Extensions panel, the **Python** extension (Microsoft).
3. **Open the project**: *File → Open Folder…* and choose the `dip_app` folder.
4. **Open a terminal** inside VS Code (`` Ctrl+` ``) and create a virtual environment:

   ```powershell
   py -m venv .venv
   ```

5. **Activate it**:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   The prompt now starts with `(.venv)`. If PowerShell says scripts are disabled, run
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then repeat the activation.
   Alternative without changing policy: `.venv\Scripts\activate.bat` in a Command Prompt.
6. **Select the interpreter**: `Ctrl+Shift+P` → *Python: Select Interpreter* → choose
   `.\.venv\Scripts\python.exe`. (If VS Code asks to use the new environment, accept.)
7. **Install dependencies**:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

8. **Run the app**:

   ```powershell
   python app.py
   ```

   Open <http://127.0.0.1:5000> in your browser. Stop the server with `Ctrl+C`.
   You can also press **F5** in VS Code (uses `.vscode/launch.json`).

   Optional: auto-reload while editing code: `$env:FLASK_DEBUG="1"; python app.py`
   (PowerShell). Do not use debug mode on a shared network.
9. The page loads Bootstrap from the jsDelivr CDN, so the computer needs internet access the
   first time. To work offline, download the Bootstrap and Bootstrap Icons files into
   `static/vendor/` and point the three `<link>`/`<script>` tags in `templates/index.html`
   at them (the Content-Security-Policy in `app.py` already allows files from `'self'`).

### Settings (optional environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | 5000 | Port to listen on |
| `DIP_MAX_UPLOAD_MB` | 10 | Maximum upload size |
| `DIP_MAX_IMAGE_DIM` | 2560 | Larger uploads are scaled down to this longest side |
| `DIP_SECRET_KEY` | generated in `instance/secret.key` | Session signing key |

PowerShell example: `$env:DIP_MAX_UPLOAD_MB="20"; python app.py`

## Testing

### Automated tests (no extra packages needed)

```powershell
python -m unittest -v
```

The 41 tests cover upload validation (fake images, bad extensions, oversize files), every
operation with default parameters, numerical checks of individual operations (negative equals
`255 - image`, brightness clips instead of wrapping, thresholds are strictly 0/255, rotate 90 is
counter-clockwise, ...), parameter validation, undo/reset, download formats, histograms and
channel separation. In VS Code you can also use the Testing panel (flask icon); unittest is
pre-configured.

### Manual test with the sample image

1. `python generate_sample.py` creates `samples/sample.png` (gradients, shapes, text, uneven
   lighting, a washed-out patch, Gaussian and salt-and-pepper noise).
2. Drag it onto the drop zone. Original and processed previews show 640 × 480 px.
3. Try these and watch the Processed preview:

| Do this | Expected result |
|---|---|
| Median blur, size 5 | The black and white salt-and-pepper dots disappear |
| Histogram equalization | The washed-out patch (bottom right) gains contrast |
| Gaussian blur, then Canny | Cleaner, thinner edges than Canny alone (History shows 2 steps) |
| Adaptive threshold | Text and shapes are separated despite the uneven lighting |
| Binary threshold with Otsu | Clean black/white image; the threshold slider is ignored |
| Rotate 45°, expand on | Canvas grows, corners are black (or white) |
| Resize, 25 % | Processed badge shows 160 × 120 px |
| Erosion then Dilation | Same result family as Opening |
| Histograms tab | Two plots per image; they change after every operation |
| RGB channels tab | Three tinted or gray previews of the current image |
| Undo / Reset to original | Previous step or the untouched original comes back |
| Download → PNG | `sample_processed.png` is saved and matches the Processed preview |

4. **Error checks**: rename a text file to `fake.png` and upload it (rejected: not a valid
   image); upload an `.exe` (rejected: unsupported type); upload a file larger than the limit
   (rejected before upload).

## API reference

All responses are JSON (except images and downloads). Errors look like
`{"ok": false, "error": "message"}` with status 400, 404, 409, 413 or 500. The image belongs to
the browser session (signed cookie), so no image id is sent by the client.

| Method and route | Purpose |
|---|---|
| `GET /` | Dashboard |
| `GET /api/operations` | Operation catalogue with parameter specs, plus limits |
| `GET /api/state` | Current image state (restores the page after a refresh) |
| `POST /api/upload` | Multipart field `image`. Returns state, status 201 |
| `POST /api/process` | `{"operation": "gaussian_blur", "params": {"ksize": 7}}` – applied to the current result |
| `POST /api/undo` | Remove the last step |
| `POST /api/reset` | Back to the original |
| `GET /api/image/original` or `/api/image/processed` | PNG bytes |
| `GET /api/download?format=png\|jpg\|webp&quality=1-100` | Processed image as a file |
| `POST /api/histogram` | Histogram plots (PNG data URLs) and statistics for original and processed |
| `POST /api/channels` | `{"source": "processed\|original", "mode": "color\|gray"}` |

Example with `curl` (Git Bash or WSL; the cookie jar keeps your session):

```bash
curl -c jar -b jar -F image=@samples/sample.png http://127.0.0.1:5000/api/upload
curl -c jar -b jar -H "Content-Type: application/json" \
     -d '{"operation":"canny","params":{"threshold1":80,"threshold2":160}}' \
     http://127.0.0.1:5000/api/process
curl -b jar -o result.png "http://127.0.0.1:5000/api/download?format=png"
```

## Security and robustness

- Uploads are validated by **content** (Pillow decodes them), not only by extension; only PNG,
  JPEG, BMP, WebP, TIFF and GIF are accepted. Size limit (default 10 MB) is enforced by Flask
  and again while reading. Images above 40 megapixels are refused (decompression-bomb guard);
  images over 2560 px on the longest side are scaled down.
- Client file names are never used for storage. Images live in `instance/workspaces/<random
  32-hex id>/`, and ids are re-validated with a strict pattern before touching the disk.
  Names are sanitised before being used in the download file name.
- Operation names must be in the registry; parameters are type-checked, range-checked
  (odd kernel sizes, enumerated choices, NaN/infinity and booleans-as-numbers rejected) and
  unknown parameters are refused. Results above 4096 px per side are refused.
- Transparent images are flattened onto white; EXIF orientation is applied; 16-bit images
  are scaled to 8-bit.
- Response headers: Content-Security-Policy, `X-Content-Type-Options`, `X-Frame-Options`,
  `Referrer-Policy`; API responses are `no-store`. Session cookie is HttpOnly and SameSite=Lax.
- Idle workspaces are deleted after 24 hours; each image keeps at most 30 steps (the undo depth).

## Notes and limits

- This is a single-user learning/demo server: the workspace lock is per process, so run one
  worker. For a more robust Windows server use Waitress:
  `pip install waitress` then `waitress-serve --listen=127.0.0.1:5000 app:app`.
- Non-Local Means denoising is CPU heavy and can take several seconds on large images.
- Internally every image is 8-bit BGR; single-channel results (edges, thresholds) are stored as
  gray BGR so any operation can follow any other.
- Operations that need intensity only (Sobel, Canny, Laplacian, thresholds) first convert to
  grayscale. Histogram equalization works on the luminance channel so colours are preserved.

## Troubleshooting

| Problem | Fix |
|---|---|
| `py` or `python` not recognised | Reinstall Python with *Add to PATH*, reopen the terminal |
| Activation script blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use `activate.bat` |
| `ModuleNotFoundError: cv2` or `flask` | The venv is not active, or run `pip install -r requirements.txt` inside it |
| Page is unstyled | No internet to reach the CDN (see step 9 for offline use) |
| `Address already in use` | Another program uses port 5000: `$env:PORT="5001"; python app.py` |
| "No image loaded" after restarting | Session workspaces expire or are deleted; upload the image again |
