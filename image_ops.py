"""
image_ops.py
============
Every image-processing operation offered by the application.

Each operation is a plain function ``func(img_bgr, **params) -> ndarray`` registered with
the ``@operation`` decorator together with a *parameter specification*.  That single
specification is used for:

1. building the parameter controls in the browser (``GET /api/operations``), and
2. validating incoming requests (``validators.py``).

All images inside the application are 8-bit, 3-channel BGR arrays (OpenCV's native
layout).  Operations that produce a single-channel result (edges, thresholds, ...) are
converted back to BGR by ``apply_operation`` so that operations can be chained freely.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import MAX_RESULT_DIM
from errors import ValidationError

CATEGORIES: List[str] = [
    "Basic",
    "Geometric",
    "Intensity",
    "Filtering",
    "Edge Detection",
    "Thresholding",
    "Morphology",
    "Color Channels",
]

OPERATIONS: Dict[str, Dict[str, Any]] = {}

INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}
MORPH_SHAPES = {
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
}


# --------------------------------------------------------------------------- #
# Parameter-spec builders
# --------------------------------------------------------------------------- #
def int_param(name: str, label: str, default: int, minimum: int, maximum: int, *,
              odd: bool = False, show_if: Optional[dict] = None, hint: Optional[str] = None,
              default_from: Optional[str] = None) -> Dict[str, Any]:
    return {"name": name, "label": label, "type": "int", "default": default,
            "min": minimum, "max": maximum, "step": 2 if odd else 1, "odd": odd,
            "show_if": show_if, "help": hint, "default_from": default_from}


def float_param(name: str, label: str, default: float, minimum: float, maximum: float,
                step: float = 0.1, *, show_if: Optional[dict] = None,
                hint: Optional[str] = None) -> Dict[str, Any]:
    return {"name": name, "label": label, "type": "float", "default": default,
            "min": minimum, "max": maximum, "step": step, "odd": False,
            "show_if": show_if, "help": hint, "default_from": None}


def choice_param(name: str, label: str, default: Any, choices: Sequence[Tuple[Any, str]], *,
                 show_if: Optional[dict] = None, hint: Optional[str] = None) -> Dict[str, Any]:
    return {"name": name, "label": label, "type": "choice", "default": default,
            "choices": [{"value": value, "label": text} for value, text in choices],
            "show_if": show_if, "help": hint, "default_from": None}


def bool_param(name: str, label: str, default: bool, *, show_if: Optional[dict] = None,
               hint: Optional[str] = None) -> Dict[str, Any]:
    return {"name": name, "label": label, "type": "bool", "default": default,
            "show_if": show_if, "help": hint, "default_from": None}


def operation(key: str, label: str, category: str, description: str,
              params: Optional[List[Dict[str, Any]]] = None) -> Callable:
    """Register an operation function together with its metadata."""
    def decorator(func: Callable) -> Callable:
        OPERATIONS[key] = {"key": key, "label": label, "category": category,
                           "description": description, "params": params or [], "func": func}
        return func
    return decorator


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _scale_to_u8(values: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Take |values|, scale so the maximum becomes 255 (times *gain*) and clip to uint8."""
    magnitude = np.abs(values.astype(np.float64))
    peak = float(magnitude.max()) if magnitude.size else 0.0
    if peak <= 0.0:
        return np.zeros(magnitude.shape, dtype=np.uint8)
    return np.clip(magnitude / peak * 255.0 * gain, 0, 255).astype(np.uint8)


def _lookup_table(fn: Callable[[int], float]) -> np.ndarray:
    table = np.array([fn(i) for i in range(256)], dtype=np.float64)
    return np.clip(np.round(table), 0, 255).astype(np.uint8)


def _check_result_size(width: int, height: int) -> None:
    if width > MAX_RESULT_DIM or height > MAX_RESULT_DIM:
        raise ValidationError(
            f"The result would be {width}x{height}px. The maximum is {MAX_RESULT_DIM}px per side.")


def _kernel(ksize: int, shape: str) -> np.ndarray:
    return cv2.getStructuringElement(MORPH_SHAPES[shape], (ksize, ksize))


def extract_channel(img: np.ndarray, channel: str, mode: str = "color") -> np.ndarray:
    """Return one of the B, G, R channels as a BGR image (tinted or gray)."""
    index = {"b": 0, "g": 1, "r": 2}[channel]
    plane = img[:, :, index]
    if mode == "gray":
        return _to_bgr(plane)
    out = np.zeros_like(img)
    out[:, :, index] = plane
    return out


# --------------------------------------------------------------------------- #
# Basic
# --------------------------------------------------------------------------- #
@operation("grayscale", "Grayscale", "Basic",
           "Convert to shades of gray using the luminance formula Y = 0.299 R + 0.587 G + 0.114 B.")
def op_grayscale(img: np.ndarray) -> np.ndarray:
    return _to_bgr(_to_gray(img))


@operation("negative", "Negative", "Basic",
           "Invert every intensity: s = 255 - r. Bright areas become dark and vice versa.")
def op_negative(img: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(img)


# --------------------------------------------------------------------------- #
# Geometric
# --------------------------------------------------------------------------- #
@operation(
    "resize", "Resize", "Geometric",
    "Change the image size by a percentage or to exact pixel dimensions.",
    params=[
        choice_param("mode", "Mode", "percent",
                     [("percent", "Scale by percentage"), ("dimensions", "Width and height in pixels")]),
        float_param("percent", "Scale (%)", 50, 5, 400, 1, show_if={"mode": ["percent"]}),
        int_param("width", "Width (px)", 512, 1, MAX_RESULT_DIM,
                  show_if={"mode": ["dimensions"]}, default_from="image_width"),
        int_param("height", "Height (px)", 512, 1, MAX_RESULT_DIM,
                  show_if={"mode": ["dimensions"]}, default_from="image_height"),
        bool_param("keep_aspect", "Keep aspect ratio (fit inside the box)", True,
                   show_if={"mode": ["dimensions"]}),
        choice_param("interpolation", "Interpolation", "linear",
                     [("nearest", "Nearest neighbour"), ("linear", "Bilinear"), ("cubic", "Bicubic"),
                      ("area", "Area (best for shrinking)"), ("lanczos", "Lanczos")]),
    ])
def op_resize(img: np.ndarray, mode: str, percent: float, width: int, height: int,
              keep_aspect: bool, interpolation: str) -> np.ndarray:
    h, w = img.shape[:2]
    if mode == "percent":
        new_w = max(1, int(round(w * percent / 100.0)))
        new_h = max(1, int(round(h * percent / 100.0)))
    elif keep_aspect:
        scale = min(width / w, height / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
    else:
        new_w, new_h = width, height
    _check_result_size(new_w, new_h)
    return cv2.resize(img, (new_w, new_h), interpolation=INTERPOLATIONS[interpolation])


@operation(
    "rotate", "Rotate", "Geometric",
    "Rotate around the centre. Positive angles turn the image counter-clockwise.",
    params=[
        float_param("angle", "Angle (degrees)", 90, -360, 360, 1),
        bool_param("expand", "Expand canvas so nothing is cropped", True),
        choice_param("background", "Background colour", "black", [("black", "Black"), ("white", "White")]),
    ])
def op_rotate(img: np.ndarray, angle: float, expand: bool, background: str) -> np.ndarray:
    h, w = img.shape[:2]
    normalized = angle % 360.0
    if expand and normalized in (0.0, 90.0, 180.0, 270.0):      # exact, loss-free rotations
        if normalized == 90.0:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if normalized == 180.0:
            return cv2.rotate(img, cv2.ROTATE_180)
        if normalized == 270.0:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img.copy()

    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_w, new_h = w, h
    if expand:
        cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
        new_w = int(math.ceil(h * sin + w * cos))
        new_h = int(math.ceil(h * cos + w * sin))
        _check_result_size(new_w, new_h)
        matrix[0, 2] += new_w / 2.0 - center[0]
        matrix[1, 2] += new_h / 2.0 - center[1]
    fill = (255, 255, 255) if background == "white" else (0, 0, 0)
    return cv2.warpAffine(img, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


@operation(
    "flip", "Flip", "Geometric", "Mirror the image horizontally, vertically or both.",
    params=[choice_param("direction", "Direction", "horizontal",
                         [("horizontal", "Horizontal (left-right)"), ("vertical", "Vertical (top-bottom)"),
                          ("both", "Both axes")])])
def op_flip(img: np.ndarray, direction: str) -> np.ndarray:
    code = {"horizontal": 1, "vertical": 0, "both": -1}[direction]
    return cv2.flip(img, code)


# --------------------------------------------------------------------------- #
# Intensity transformations
# --------------------------------------------------------------------------- #
@operation(
    "brightness", "Brightness", "Intensity",
    "Add a constant to every pixel: s = r + value (clipped to 0-255).",
    params=[int_param("value", "Brightness offset", 40, -255, 255)])
def op_brightness(img: np.ndarray, value: int) -> np.ndarray:
    # int16 arithmetic avoids uint8 wrap-around before clipping.
    return np.clip(img.astype(np.int16) + value, 0, 255).astype(np.uint8)


@operation(
    "contrast", "Contrast", "Intensity",
    "Stretch or compress intensities around mid-gray: s = factor (r - 128) + 128.",
    params=[float_param("factor", "Contrast factor", 1.5, 0.1, 3.0, 0.1,
                        hint="Below 1 lowers contrast, above 1 raises it.")])
def op_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    out = factor * (img.astype(np.float32) - 128.0) + 128.0
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


@operation(
    "gamma", "Gamma correction", "Intensity",
    "Power-law transform s = 255 (r / 255)^gamma. Gamma below 1 brightens, above 1 darkens.",
    params=[float_param("gamma", "Gamma", 0.5, 0.1, 5.0, 0.1)])
def op_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    table = _lookup_table(lambda i: 255.0 * (i / 255.0) ** gamma)
    return cv2.LUT(img, table)


@operation(
    "log_transform", "Logarithmic transform", "Intensity",
    "s = c log(1 + r). Expands dark values and compresses bright ones, revealing shadow detail.",
    params=[float_param("scale", "Scale factor", 1.0, 0.2, 2.0, 0.1,
                        hint="1.0 maps 255 exactly to 255; larger values brighten further.")])
def op_log_transform(img: np.ndarray, scale: float) -> np.ndarray:
    c = 255.0 / math.log1p(255.0) * scale
    table = _lookup_table(lambda i: c * math.log1p(i))
    return cv2.LUT(img, table)


@operation(
    "hist_equalization", "Histogram equalization", "Intensity",
    "Spread the intensity histogram to improve contrast. Applied to the luminance channel so colours are preserved.",
    params=[
        choice_param("method", "Method", "global",
                     [("global", "Global equalization"), ("clahe", "CLAHE (adaptive, tile based)")]),
        float_param("clip_limit", "CLAHE clip limit", 2.0, 1.0, 40.0, 0.5, show_if={"method": ["clahe"]}),
        int_param("tile_size", "CLAHE tile grid size", 8, 2, 32, show_if={"method": ["clahe"]}),
    ])
def op_hist_equalization(img: np.ndarray, method: str, clip_limit: float, tile_size: int) -> np.ndarray:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    luma = ycrcb[:, :, 0]
    if method == "global":
        luma = cv2.equalizeHist(luma)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        luma = clahe.apply(luma)
    ycrcb[:, :, 0] = luma
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
@operation(
    "gaussian_blur", "Gaussian blur", "Filtering",
    "Smooth the image with a Gaussian kernel. Good for reducing Gaussian-like noise.",
    params=[
        int_param("ksize", "Kernel size (odd)", 5, 1, 51, odd=True),
        float_param("sigma", "Sigma (0 = automatic)", 0.0, 0.0, 20.0, 0.1),
    ])
def op_gaussian_blur(img: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


@operation(
    "median_blur", "Median blur", "Filtering",
    "Replace each pixel by the median of its neighbourhood. Excellent against salt-and-pepper noise.",
    params=[int_param("ksize", "Kernel size (odd)", 5, 3, 51, odd=True)])
def op_median_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.medianBlur(img, ksize)


@operation(
    "mean_filter", "Mean filter", "Filtering",
    "Box (averaging) filter: every pixel becomes the mean of its neighbourhood.",
    params=[int_param("ksize", "Kernel size (odd)", 5, 3, 51, odd=True)])
def op_mean_filter(img: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.blur(img, (ksize, ksize))


@operation(
    "sharpen", "Sharpen", "Filtering",
    "Enhance edges and fine detail with unsharp masking or a Laplacian-style sharpening kernel.",
    params=[
        choice_param("method", "Method", "unsharp",
                     [("unsharp", "Unsharp mask"), ("kernel", "Sharpening kernel")]),
        float_param("amount", "Amount", 1.0, 0.1, 5.0, 0.1),
        float_param("sigma", "Blur sigma", 1.5, 0.5, 10.0, 0.1, show_if={"method": ["unsharp"]}),
    ])
def op_sharpen(img: np.ndarray, method: str, amount: float, sigma: float) -> np.ndarray:
    if method == "unsharp":
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    a = float(amount)
    kernel = np.array([[0, -a, 0], [-a, 1 + 4 * a, -a], [0, -a, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


@operation(
    "denoise", "Denoise", "Filtering",
    "Remove noise while keeping edges, using Non-Local Means or a bilateral filter. Non-Local Means can take a few seconds on large images.",
    params=[
        choice_param("method", "Method", "nlm",
                     [("nlm", "Non-Local Means"), ("bilateral", "Bilateral filter")]),
        float_param("strength", "Strength (h)", 10.0, 1.0, 30.0, 1.0, show_if={"method": ["nlm"]}),
        int_param("diameter", "Neighbourhood diameter", 9, 3, 25, show_if={"method": ["bilateral"]}),
        float_param("sigma_color", "Sigma colour", 75.0, 10.0, 200.0, 5.0, show_if={"method": ["bilateral"]}),
        float_param("sigma_space", "Sigma space", 75.0, 10.0, 200.0, 5.0, show_if={"method": ["bilateral"]}),
    ])
def op_denoise(img: np.ndarray, method: str, strength: float, diameter: int,
               sigma_color: float, sigma_space: float) -> np.ndarray:
    if method == "nlm":
        return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)
    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)


# --------------------------------------------------------------------------- #
# Edge detection
# --------------------------------------------------------------------------- #
_KERNEL_CHOICES = [(1, "1"), (3, "3"), (5, "5"), (7, "7")]


@operation(
    "sobel", "Sobel edges", "Edge Detection",
    "First-derivative edge detector. Shows the gradient magnitude (or one direction) of the gray image.",
    params=[
        choice_param("ksize", "Kernel size", 3, _KERNEL_CHOICES),
        choice_param("direction", "Direction", "both",
                     [("both", "Magnitude (X and Y)"), ("x", "Horizontal gradient (vertical edges)"),
                      ("y", "Vertical gradient (horizontal edges)")]),
        float_param("gain", "Edge brightness gain", 1.5, 0.5, 5.0, 0.1),
    ])
def op_sobel(img: np.ndarray, ksize: int, direction: str, gain: float) -> np.ndarray:
    gray = _to_gray(img)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    if direction == "x":
        magnitude = np.abs(grad_x)
    elif direction == "y":
        magnitude = np.abs(grad_y)
    else:
        magnitude = np.hypot(grad_x, grad_y)
    return _scale_to_u8(magnitude, gain)


@operation(
    "canny", "Canny edges", "Edge Detection",
    "Multi-stage edge detector with hysteresis thresholds. Gives thin, well-connected edges.",
    params=[
        int_param("threshold1", "Lower threshold", 100, 0, 500),
        int_param("threshold2", "Upper threshold", 200, 0, 500),
        choice_param("aperture", "Sobel aperture size", 3, [(3, "3"), (5, "5"), (7, "7")]),
        bool_param("l2_gradient", "Use L2 gradient (more accurate)", False),
        bool_param("pre_blur", "Smooth with 5x5 Gaussian first", True),
    ])
def op_canny(img: np.ndarray, threshold1: int, threshold2: int, aperture: int,
             l2_gradient: bool, pre_blur: bool) -> np.ndarray:
    if threshold1 > threshold2:
        raise ValidationError("Lower threshold (threshold1) must not be greater than the upper threshold (threshold2).")
    gray = _to_gray(img)
    if pre_blur:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture, L2gradient=l2_gradient)


@operation(
    "laplacian", "Laplacian edges", "Edge Detection",
    "Second-derivative edge detector that responds to rapid intensity change in all directions.",
    params=[
        choice_param("ksize", "Kernel size", 3, _KERNEL_CHOICES),
        float_param("gain", "Edge brightness gain", 2.0, 0.5, 5.0, 0.1),
        bool_param("smooth", "Smooth with 3x3 Gaussian first", True),
    ])
def op_laplacian(img: np.ndarray, ksize: int, gain: float, smooth: bool) -> np.ndarray:
    gray = _to_gray(img)
    if smooth:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return _scale_to_u8(cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize), gain)


# --------------------------------------------------------------------------- #
# Thresholding
# --------------------------------------------------------------------------- #
def _global_threshold(img: np.ndarray, threshold: int, use_otsu: bool, inverse: bool) -> np.ndarray:
    gray = _to_gray(img)
    kind = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    if use_otsu:
        _, out = cv2.threshold(gray, 0, 255, kind | cv2.THRESH_OTSU)
    else:
        _, out = cv2.threshold(gray, threshold, 255, kind)
    return out


@operation(
    "threshold_binary", "Binary threshold", "Thresholding",
    "Pixels brighter than the threshold become white, all others black.",
    params=[
        int_param("threshold", "Threshold", 127, 0, 255),
        bool_param("use_otsu", "Choose the threshold automatically (Otsu)", False),
    ])
def op_threshold_binary(img: np.ndarray, threshold: int, use_otsu: bool) -> np.ndarray:
    return _global_threshold(img, threshold, use_otsu, inverse=False)


@operation(
    "threshold_binary_inv", "Inverse binary threshold", "Thresholding",
    "Pixels brighter than the threshold become black, all others white.",
    params=[
        int_param("threshold", "Threshold", 127, 0, 255),
        bool_param("use_otsu", "Choose the threshold automatically (Otsu)", False),
    ])
def op_threshold_binary_inv(img: np.ndarray, threshold: int, use_otsu: bool) -> np.ndarray:
    return _global_threshold(img, threshold, use_otsu, inverse=True)


@operation(
    "threshold_adaptive", "Adaptive threshold", "Thresholding",
    "The threshold is computed per pixel from its neighbourhood, which copes with uneven lighting.",
    params=[
        choice_param("method", "Neighbourhood weighting", "gaussian",
                     [("gaussian", "Gaussian weighted mean"), ("mean", "Plain mean")]),
        int_param("block_size", "Block size (odd)", 11, 3, 99, odd=True),
        float_param("c", "Constant C (subtracted from the mean)", 2.0, -30.0, 30.0, 0.5),
        bool_param("invert", "Invert result", False),
    ])
def op_threshold_adaptive(img: np.ndarray, method: str, block_size: int, c: float,
                          invert: bool) -> np.ndarray:
    gray = _to_gray(img)
    flag = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    kind = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(gray, 255, flag, kind, block_size, c)


# --------------------------------------------------------------------------- #
# Morphology
# --------------------------------------------------------------------------- #
def _morph_params() -> List[Dict[str, Any]]:
    return [
        int_param("ksize", "Kernel size (odd)", 3, 1, 31, odd=True),
        choice_param("shape", "Kernel shape", "rect",
                     [("rect", "Rectangle"), ("ellipse", "Ellipse"), ("cross", "Cross")]),
        int_param("iterations", "Iterations", 1, 1, 10),
    ]


@operation("erosion", "Erosion", "Morphology",
           "Shrinks bright regions and removes small bright specks (minimum filter over the kernel).",
           params=_morph_params())
def op_erosion(img: np.ndarray, ksize: int, shape: str, iterations: int) -> np.ndarray:
    return cv2.erode(img, _kernel(ksize, shape), iterations=iterations)


@operation("dilation", "Dilation", "Morphology",
           "Grows bright regions and fills small dark gaps (maximum filter over the kernel).",
           params=_morph_params())
def op_dilation(img: np.ndarray, ksize: int, shape: str, iterations: int) -> np.ndarray:
    return cv2.dilate(img, _kernel(ksize, shape), iterations=iterations)


@operation("opening", "Opening", "Morphology",
           "Erosion followed by dilation. Removes small bright noise while keeping the size of larger shapes.",
           params=_morph_params())
def op_opening(img: np.ndarray, ksize: int, shape: str, iterations: int) -> np.ndarray:
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, _kernel(ksize, shape), iterations=iterations)


@operation("closing", "Closing", "Morphology",
           "Dilation followed by erosion. Closes small dark holes and gaps inside bright shapes.",
           params=_morph_params())
def op_closing(img: np.ndarray, ksize: int, shape: str, iterations: int) -> np.ndarray:
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, _kernel(ksize, shape), iterations=iterations)


# --------------------------------------------------------------------------- #
# Color channels
# --------------------------------------------------------------------------- #
@operation(
    "channel_extract", "Extract RGB channel", "Color Channels",
    "Keep a single colour channel. Use the 'RGB Channels' tab below the previews to see all three at once.",
    params=[
        choice_param("channel", "Channel", "r", [("r", "Red"), ("g", "Green"), ("b", "Blue")]),
        choice_param("mode", "Display as", "color",
                     [("color", "Tinted in its own colour"), ("gray", "Grayscale intensity")]),
    ])
def op_channel_extract(img: np.ndarray, channel: str, mode: str) -> np.ndarray:
    return extract_channel(img, channel, mode)


# --------------------------------------------------------------------------- #
# Public API used by app.py
# --------------------------------------------------------------------------- #
def _finalize(result: np.ndarray) -> np.ndarray:
    """Guarantee an 8-bit, 3-channel, contiguous BGR array."""
    if result.dtype != np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)
    if result.ndim == 2:
        result = _to_bgr(result)
    elif result.ndim == 3 and result.shape[2] == 1:
        result = _to_bgr(result[:, :, 0])
    return np.ascontiguousarray(result)


def apply_operation(key: str, img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Run a registered operation with already-validated parameters."""
    return _finalize(OPERATIONS[key]["func"](img, **params))


def public_specs() -> Dict[str, Any]:
    """Operation metadata (without the Python callables) for the front end."""
    operations = [{k: v for k, v in op.items() if k != "func"} for op in OPERATIONS.values()]
    used = {op["category"] for op in operations}
    return {"categories": [c for c in CATEGORIES if c in used], "operations": operations}
