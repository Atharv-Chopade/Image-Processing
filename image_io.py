"""Safe image decoding (Pillow) and encoding (OpenCV) helpers."""
from __future__ import annotations

import base64
import io
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename

from config import ALLOWED_EXTENSIONS, ALLOWED_FORMATS, MAX_IMAGE_DIM, MAX_INPUT_PIXELS
from errors import ApiError, ValidationError

# Pillow raises DecompressionBombError for images above 2x this value.
Image.MAX_IMAGE_PIXELS = MAX_INPUT_PIXELS

_ENCODERS = {
    "png": (".png", "image/png"),
    "jpg": (".jpg", "image/jpeg"),
    "jpeg": (".jpg", "image/jpeg"),
    "webp": (".webp", "image/webp"),
}


# --------------------------------------------------------------------------- #
# File names
# --------------------------------------------------------------------------- #
def clean_filename(raw_name: str) -> str:
    """Return a safe ``stem.ext`` name; raise if the extension is not allowed."""
    base = os.path.basename((raw_name or "").replace("\\", "/"))
    stem, ext = os.path.splitext(base)
    ext = ext.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        shown = f".{ext}" if ext else "(none)"
        raise ValidationError(f"Unsupported file type {shown}. Allowed types: {allowed}.")
    safe_stem = secure_filename(stem)[:80] or "image"
    return f"{safe_stem}.{ext}"


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert any Pillow image to an 8-bit BGR array (transparency is flattened onto white)."""
    mode = img.mode
    if mode in ("I;16", "I;16L", "I;16B", "I;16N", "I", "F"):
        data = np.asarray(img).astype(np.float64)
        low, high = float(data.min()), float(data.max())
        scaled = (data - low) / (high - low) * 255.0 if high > low else np.zeros_like(data)
        return cv2.cvtColor(scaled.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    has_alpha = (mode in ("RGBA", "LA", "PA", "La", "RGBa")
                 or "transparency" in img.info)
    if has_alpha:
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        rgb = background.convert("RGB")
    else:
        rgb = img.convert("RGB")
    return np.ascontiguousarray(cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))


def decode_image(raw: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Validate and decode uploaded bytes into a BGR array.

    The *content* is checked with Pillow (not just the file name): the format must be on the
    allow-list, the pixel count must be reasonable and the data must decode cleanly.
    Oversized images are scaled down to ``MAX_IMAGE_DIM`` to keep processing fast.
    """
    if not raw:
        raise ValidationError("The uploaded file is empty.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as probe:
                fmt = probe.format
                width, height = probe.size
                probe.verify()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValidationError("The image has too many pixels to process safely.")
    except Exception:
        raise ValidationError("The file is not a valid image, or it is corrupted.")

    if fmt not in ALLOWED_FORMATS:
        raise ValidationError(f"Unsupported image format ({fmt}). Use PNG, JPEG, BMP, WebP, TIFF or GIF.")
    if width * height > MAX_INPUT_PIXELS:
        raise ValidationError(
            f"The image is {width}x{height}px; the limit is {MAX_INPUT_PIXELS // 1_000_000} megapixels.")

    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            img = ImageOps.exif_transpose(img)      # respect camera orientation
            bgr = _pil_to_bgr(img)
    except Exception:
        raise ValidationError("The image could not be decoded.")

    h, w = bgr.shape[:2]
    downscaled = False
    longest = max(h, w)
    if longest > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / longest
        bgr = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
        downscaled = True
    return bgr, {"format": fmt, "source_size": (w, h), "downscaled": downscaled}


# --------------------------------------------------------------------------- #
# Encoding / files
# --------------------------------------------------------------------------- #
def encode_image(img: np.ndarray, fmt: str = "png", quality: int = 95) -> Tuple[bytes, str, str]:
    """Encode to PNG/JPEG/WebP. Returns ``(bytes, mimetype, extension_without_dot)``."""
    fmt = fmt.lower()
    if fmt not in _ENCODERS:
        raise ValidationError("Format must be one of: png, jpg, webp.")
    suffix, mimetype = _ENCODERS[fmt]
    params = []
    if suffix == ".jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif suffix == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
    elif suffix == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, buffer = cv2.imencode(suffix, img, params)
    if not ok:
        raise ApiError("The image could not be encoded.", 500)
    return buffer.tobytes(), mimetype, suffix.lstrip(".")


def to_data_url(img: np.ndarray) -> str:
    data, mimetype, _ = encode_image(img, "png")
    return f"data:{mimetype};base64,{base64.b64encode(data).decode('ascii')}"


def read_image_file(path: Path):
    """Read an image with a Unicode-safe path (cv2.imread fails on some Windows paths)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_png_file(path: Path, img: np.ndarray) -> None:
    """Atomically write a PNG file using a Unicode-safe path."""
    ok, buffer = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ApiError("The image could not be saved.", 500)
    temp = path.with_name(path.name + ".tmp")
    buffer.tofile(str(temp))
    os.replace(temp, path)
