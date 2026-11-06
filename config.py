"""Configuration and safety limits for PixelLab (Digital Image Processing Studio)."""
from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer from the environment, falling back to *default*."""
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(value, minimum)


# ---------------------------------------------------------------- upload limits
MAX_UPLOAD_MB = _env_int("DIP_MAX_UPLOAD_MB", 10)           # file size limit
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff", "gif"}
ALLOWED_FORMATS = {"PNG", "JPEG", "BMP", "WEBP", "TIFF", "GIF"}  # real format, checked by Pillow
MAX_INPUT_PIXELS = 40_000_000                                # rejects decompression bombs
MAX_IMAGE_DIM = _env_int("DIP_MAX_IMAGE_DIM", 2560)          # larger uploads are scaled down
MAX_RESULT_DIM = 4096                                        # max width/height after resize/rotate

# ------------------------------------------------------------ workspace limits
MAX_STEPS = 30                # sequential operations kept per image (also the undo depth)
WORKSPACE_TTL_HOURS = 24      # idle workspaces are deleted after this time
PREVIEW_MAX_DIM = 640         # channel-separation previews are scaled to this size


class Config:
    """Flask configuration object (see ``create_app`` in app.py)."""

    BASE_DIR = BASE_DIR
    INSTANCE_DIR = BASE_DIR / "instance"
    WORKSPACE_DIR = INSTANCE_DIR / "workspaces"

    MAX_UPLOAD_MB = MAX_UPLOAD_MB
    MAX_STEPS = MAX_STEPS
    WORKSPACE_TTL_HOURS = WORKSPACE_TTL_HOURS

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = timedelta(hours=WORKSPACE_TTL_HOURS)
