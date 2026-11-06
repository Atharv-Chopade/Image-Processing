"""Image analysis: histogram rendering (OpenCV drawing) and RGB channel separation."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from config import PREVIEW_MAX_DIM
from image_io import to_data_url
from image_ops import extract_channel

# BGR colours used on the histogram plots.
_COLOR_R = (80, 80, 240)
_COLOR_G = (90, 210, 90)
_COLOR_B = (240, 140, 60)
_COLOR_GRAY = (215, 215, 215)


def _histogram(channel: np.ndarray) -> np.ndarray:
    return cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()


def _render_plot(series: Sequence[Tuple[Tuple[int, int, int], np.ndarray]],
                 width: int = 512, height: int = 240) -> np.ndarray:
    """Draw one or more 256-bin histograms on a dark canvas."""
    left, right, top, bottom = 12, 12, 10, 24
    canvas = np.full((height, width, 3), (38, 34, 32), dtype=np.uint8)
    plot_w, plot_h = width - left - right, height - top - bottom
    base_y = top + plot_h

    for value in (0, 64, 128, 192, 255):                       # grid + axis labels
        x = left + int(round(value / 255.0 * plot_w))
        cv2.line(canvas, (x, top), (x, base_y), (70, 66, 64), 1)
        label_x = max(0, min(width - 26, x - 10))
        cv2.putText(canvas, str(value), (label_x, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.line(canvas, (left, base_y), (left + plot_w, base_y), (120, 120, 120), 1)

    peak = max((float(h.max()) for _, h in series), default=0.0)
    peak = peak if peak > 0 else 1.0
    xs = left + np.arange(256) / 255.0 * plot_w

    for color, hist in series:
        ys = base_y - hist / peak * plot_h
        line = np.stack([xs, ys], axis=1).round().astype(np.int32)
        closing = np.array([[line[-1, 0], base_y], [line[0, 0], base_y]], dtype=np.int32)
        polygon = np.vstack([line, closing])
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [polygon], color)
        canvas = cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0)
        cv2.polylines(canvas, [line.reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
    return canvas


def _channel_stats(channel: np.ndarray) -> Dict[str, Any]:
    return {"mean": round(float(channel.mean()), 2), "std": round(float(channel.std()), 2),
            "min": int(channel.min()), "max": int(channel.max()), "median": int(np.median(channel))}


def histogram_report(img: np.ndarray) -> Dict[str, Any]:
    """Return RGB and grayscale histogram plots (PNG data URLs) plus per-channel statistics."""
    blue, green, red = cv2.split(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb_plot = _render_plot([(_COLOR_B, _histogram(blue)), (_COLOR_G, _histogram(green)),
                             (_COLOR_R, _histogram(red))])
    gray_plot = _render_plot([(_COLOR_GRAY, _histogram(gray))])
    height, width = img.shape[:2]
    return {
        "rgb": to_data_url(rgb_plot),
        "gray": to_data_url(gray_plot),
        "stats": {"width": width, "height": height,
                  "red": _channel_stats(red), "green": _channel_stats(green),
                  "blue": _channel_stats(blue), "gray": _channel_stats(gray)},
    }


def channel_previews(img: np.ndarray, mode: str = "color",
                     max_dim: int = PREVIEW_MAX_DIM) -> List[Dict[str, str]]:
    """Separate the image into R, G and B previews (scaled down to keep responses small)."""
    height, width = img.shape[:2]
    longest = max(height, width)
    if longest > max_dim:
        scale = max_dim / longest
        img = cv2.resize(img, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA)
    return [{"key": key, "name": name, "image": to_data_url(extract_channel(img, key, mode))}
            for key, name in (("r", "Red"), ("g", "Green"), ("b", "Blue"))]
