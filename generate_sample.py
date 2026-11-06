"""Create samples/sample.png: a synthetic picture for trying every operation.

It contains smooth gradients, sharp shapes and text (good for edge detection and thresholding),
uneven lighting (adaptive threshold), a low-contrast corner (histogram equalization), plus
Gaussian and salt-and-pepper noise (blur, median and denoise filters).

Run:  python generate_sample.py
"""
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / "samples" / "sample.png"


def main() -> None:
    rng = np.random.default_rng(42)
    h, w = 480, 640
    img = np.zeros((h, w, 3), np.float32)
    img[:, :, 0] = np.linspace(40, 220, w)[None, :]            # blue ramps left to right
    img[:, :, 1] = np.linspace(200, 60, h)[:, None]            # green ramps top to bottom
    img[:, :, 2] = 110

    light = np.linspace(0.55, 1.15, w)[None, :, None]           # uneven illumination
    img *= light

    cv2.rectangle(img, (60, 60), (260, 200), (255, 255, 255), -1)
    cv2.circle(img, (440, 150), 80, (20, 20, 200), -1)
    cv2.line(img, (40, 300), (600, 420), (0, 0, 0), 4)
    cv2.putText(img, "DIP LAB", (90, 400), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 3, cv2.LINE_AA)
    img[300:450, 470:620] = 0.25 * img[300:450, 470:620] + 90   # washed-out, low-contrast patch

    img = np.clip(img + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
    salt = rng.random((h, w))
    img[salt < 0.01] = 0
    img[salt > 0.99] = 255

    OUT.parent.mkdir(exist_ok=True)
    ok, buffer = cv2.imencode(".png", img)
    if not ok:
        raise SystemExit("Could not encode the sample image.")
    buffer.tofile(str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
