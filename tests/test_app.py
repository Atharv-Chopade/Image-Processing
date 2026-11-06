"""Automated tests (standard library only).  Run from the project folder with:

    python -m unittest -v
"""
import io
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app import create_app
from image_ops import OPERATIONS, apply_operation
from validators import validate_params

EXPECTED_OPERATIONS = {
    "grayscale", "resize", "rotate", "flip", "brightness", "contrast", "hist_equalization",
    "gaussian_blur", "median_blur", "sharpen", "denoise", "sobel", "canny", "laplacian",
    "threshold_binary", "threshold_binary_inv", "threshold_adaptive", "negative",
    "log_transform", "gamma", "erosion", "dilation", "opening", "closing", "mean_filter",
    "channel_extract",
}


def make_image(height=96, width=128):
    """A small synthetic test picture: gradients, a rectangle, a circle and mild noise."""
    rng = np.random.default_rng(0)
    img = np.zeros((height, width, 3), np.uint8)
    img[:, :, 0] = np.linspace(0, 255, width).astype(np.uint8)[None, :]
    img[:, :, 1] = np.linspace(255, 0, height).astype(np.uint8)[:, None]
    img[:, :, 2] = 90
    cv2.rectangle(img, (20, 20), (70, 60), (255, 255, 255), -1)
    cv2.circle(img, (90, 50), 20, (0, 0, 0), -1)
    noise = rng.integers(-15, 16, img.shape)
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


def png_bytes(img):
    return cv2.imencode(".png", img)[1].tobytes()


def run(op, img, **params):
    return apply_operation(op, img, validate_params(op, params))


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.app = create_app({
            "TESTING": True, "SECRET_KEY": "test-key", "INSTANCE_DIR": root,
            "WORKSPACE_DIR": root / "workspaces", "MAX_UPLOAD_MB": 1,
        })
        self.client = self.app.test_client()

    def tearDown(self):
        self._tmp.cleanup()

    def upload(self, data=None, name="test.png"):
        data = png_bytes(make_image()) if data is None else data
        return self.client.post("/api/upload", data={"image": (io.BytesIO(data), name)},
                                content_type="multipart/form-data")

    def process(self, operation, params=None):
        return self.client.post("/api/process", json={"operation": operation, "params": params or {}})

    def processed_image(self):
        response = self.client.get("/api/image/processed")
        self.assertEqual(response.status_code, 200)
        return cv2.imdecode(np.frombuffer(response.data, np.uint8), cv2.IMREAD_COLOR)


class PageAndCatalogueTests(ApiTestCase):
    def test_index_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"PixelLab", response.data)
        self.assertIn("Content-Security-Policy", response.headers)

    def test_all_required_operations_are_registered(self):
        data = self.client.get("/api/operations").get_json()
        self.assertEqual({op["key"] for op in data["operations"]}, EXPECTED_OPERATIONS)
        self.assertEqual(set(OPERATIONS), EXPECTED_OPERATIONS)
        self.assertEqual(data["limits"]["max_upload_mb"], 1)


class UploadTests(ApiTestCase):
    def test_valid_upload(self):
        response = self.upload()
        self.assertEqual(response.status_code, 201)
        body = response.get_json()
        self.assertTrue(body["loaded"])
        self.assertEqual(body["original_size"], {"width": 128, "height": 96})

    def test_rejects_non_image_content_with_image_extension(self):
        response = self.upload(b"this is definitely not an image", "evil.png")
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["ok"])

    def test_rejects_disallowed_extension(self):
        self.assertEqual(self.upload(name="script.exe").status_code, 400)
        self.assertEqual(self.upload(name="noextension").status_code, 400)

    def test_rejects_missing_file(self):
        response = self.client.post("/api/upload", data={}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_rejects_oversized_file(self):
        response = self.upload(b"\x00" * (2 * 1024 * 1024), "big.png")
        self.assertEqual(response.status_code, 413)
        self.assertIn("too large", response.get_json()["error"])

    def test_filename_is_sanitised(self):
        body = self.upload(name="../../etc/my photo.PNG").get_json()
        self.assertEqual(body["filename"], "my_photo.png")

    def test_state_is_restored_for_same_session(self):
        self.upload()
        self.process("grayscale")
        state = self.client.get("/api/state").get_json()
        self.assertTrue(state["loaded"])
        self.assertEqual(len(state["history"]), 1)


class ProcessingTests(ApiTestCase):
    def test_process_requires_an_uploaded_image(self):
        response = self.process("grayscale")
        self.assertEqual(response.status_code, 409)

    def test_every_operation_works_with_default_parameters(self):
        self.upload()
        for key in sorted(EXPECTED_OPERATIONS):
            with self.subTest(operation=key):
                self.client.post("/api/reset")
                response = self.process(key)
                self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
                image = self.processed_image()
                self.assertEqual(image.ndim, 3)
                self.assertEqual(image.dtype, np.uint8)

    def test_sequential_processing_and_history(self):
        self.upload()
        self.process("grayscale")
        body = self.process("resize", {"mode": "percent", "percent": 50}).get_json()
        self.assertEqual([step["operation"] for step in body["history"]], ["grayscale", "resize"])
        self.assertEqual(body["current_size"], {"width": 64, "height": 48})
        image = self.processed_image()
        self.assertEqual(image.shape[:2], (48, 64))
        self.assertTrue(np.array_equal(image[:, :, 0], image[:, :, 1]))   # still gray after resize

    def test_undo_and_reset(self):
        self.upload()
        self.process("negative")
        self.process("flip", {"direction": "vertical"})
        body = self.client.post("/api/undo").get_json()
        self.assertEqual(len(body["history"]), 1)
        body = self.client.post("/api/reset").get_json()
        self.assertEqual(body["history"], [])
        original = cv2.imdecode(np.frombuffer(self.client.get("/api/image/original").data, np.uint8),
                                cv2.IMREAD_COLOR)
        self.assertTrue(np.array_equal(self.processed_image(), original))
        self.assertEqual(self.client.post("/api/undo").status_code, 400)

    def test_invalid_requests_are_rejected(self):
        self.upload()
        cases = [
            ({"operation": "does_not_exist"}, "Unknown operation"),
            ({"operation": 123}, "operation"),
            ({}, "operation"),
            ({"operation": "gaussian_blur", "params": {"ksize": 4}}, "odd"),
            ({"operation": "gaussian_blur", "params": {"ksize": 999}}, "between"),
            ({"operation": "gaussian_blur", "params": {"ksize": "abc"}}, "whole number"),
            ({"operation": "gaussian_blur", "params": {"ksize": 5.5}}, "whole number"),
            ({"operation": "gaussian_blur", "params": {"bogus": 1}}, "Unknown parameter"),
            ({"operation": "gaussian_blur", "params": [1, 2]}, "JSON object"),
            ({"operation": "flip", "params": {"direction": "diagonal"}}, "one of"),
            ({"operation": "brightness", "params": {"value": True}}, "whole number"),
            ({"operation": "brightness", "params": {"value": 1e9}}, "between"),
            ({"operation": "contrast", "params": {"factor": float("nan")}}, "number"),
            ({"operation": "canny", "params": {"threshold1": 300, "threshold2": 100}}, "threshold"),
            ({"operation": "resize", "params": {"mode": "dimensions", "width": 99999, "height": 10}}, "between"),
        ]
        for payload, fragment in cases:
            with self.subTest(payload=payload):
                response = self.client.post("/api/process", json=payload)
                self.assertEqual(response.status_code, 400)
                body = response.get_json()
                self.assertFalse(body["ok"])
                self.assertIn(fragment.lower(), body["error"].lower())

    def test_non_json_body_is_rejected(self):
        self.upload()
        response = self.client.post("/api/process", data="plain text", content_type="text/plain")
        self.assertEqual(response.status_code, 400)

    def test_step_limit(self):
        self.app.config["MAX_STEPS"] = 2
        self.client = self.app.test_client()
        # a fresh store reads the limit from app.config at call time
        self.upload()
        self.assertEqual(self.process("negative").status_code, 200)
        self.assertEqual(self.process("negative").status_code, 200)
        self.assertEqual(self.process("negative").status_code, 409)


class DownloadAndAnalysisTests(ApiTestCase):
    def test_download_formats(self):
        self.upload(name="holiday.png")
        self.process("grayscale")
        png = self.client.get("/api/download?format=png")
        self.assertEqual(png.status_code, 200)
        self.assertTrue(png.data.startswith(b"\x89PNG"))
        self.assertIn("holiday_processed.png", png.headers["Content-Disposition"])
        jpg = self.client.get("/api/download?format=jpg&quality=80")
        self.assertTrue(jpg.data.startswith(b"\xff\xd8"))
        webp = self.client.get("/api/download?format=webp")
        self.assertTrue(webp.data.startswith(b"RIFF"))

    def test_download_validation(self):
        self.upload()
        self.assertEqual(self.client.get("/api/download?format=exe").status_code, 400)
        self.assertEqual(self.client.get("/api/download?quality=500").status_code, 400)
        self.assertEqual(self.client.get("/api/download?quality=abc").status_code, 400)

    def test_download_without_image(self):
        self.assertEqual(self.client.get("/api/download").status_code, 409)

    def test_histogram(self):
        self.upload()
        body = self.client.post("/api/histogram").get_json()
        for side in ("original", "processed"):
            self.assertTrue(body[side]["rgb"].startswith("data:image/png;base64,"))
            self.assertTrue(body[side]["gray"].startswith("data:image/png;base64,"))
            self.assertEqual(body[side]["stats"]["width"], 128)
            self.assertIn("mean", body[side]["stats"]["red"])

    def test_channel_separation(self):
        self.upload()
        for mode in ("color", "gray"):
            body = self.client.post("/api/channels", json={"source": "original", "mode": mode}).get_json()
            self.assertEqual([c["key"] for c in body["channels"]], ["r", "g", "b"])
        self.assertEqual(self.client.post("/api/channels", json={"mode": "sepia"}).status_code, 400)
        self.assertEqual(self.client.post("/api/channels", json={"source": "x"}).status_code, 400)

    def test_unknown_image_kind(self):
        self.upload()
        self.assertEqual(self.client.get("/api/image/secret").status_code, 404)

    def test_api_404_is_json(self):
        response = self.client.get("/api/nothing-here")
        self.assertEqual(response.status_code, 404)
        self.assertFalse(response.get_json()["ok"])


class OperationCorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.img = make_image()

    def test_negative(self):
        np.testing.assert_array_equal(run("negative", self.img), 255 - self.img)

    def test_flip(self):
        np.testing.assert_array_equal(run("flip", self.img, direction="horizontal"), self.img[:, ::-1])
        np.testing.assert_array_equal(run("flip", self.img, direction="vertical"), self.img[::-1])
        np.testing.assert_array_equal(run("flip", self.img, direction="both"), self.img[::-1, ::-1])

    def test_brightness_clips_instead_of_wrapping(self):
        self.assertTrue((run("brightness", self.img, value=255) == 255).all())
        self.assertTrue((run("brightness", self.img, value=-255) == 0).all())

    def test_grayscale_channels_are_equal(self):
        out = run("grayscale", self.img)
        np.testing.assert_array_equal(out[:, :, 0], out[:, :, 1])
        np.testing.assert_array_equal(out[:, :, 1], out[:, :, 2])

    def test_identity_parameters(self):
        np.testing.assert_array_equal(run("gamma", self.img, gamma=1.0), self.img)
        np.testing.assert_array_equal(run("contrast", self.img, factor=1.0), self.img)
        np.testing.assert_array_equal(run("brightness", self.img, value=0), self.img)

    def test_gamma_direction(self):
        self.assertGreater(run("gamma", self.img, gamma=0.4).mean(), self.img.mean())
        self.assertLess(run("gamma", self.img, gamma=2.5).mean(), self.img.mean())

    def test_log_transform_brightens_dark_values(self):
        dark = np.full((8, 8, 3), 20, np.uint8)
        self.assertGreater(run("log_transform", dark, scale=1.0).mean(), 20)

    def test_resize_and_rotate_shapes(self):
        h, w = self.img.shape[:2]
        self.assertEqual(run("resize", self.img, mode="percent", percent=50).shape[:2], (h // 2, w // 2))
        self.assertEqual(run("resize", self.img, mode="dimensions", width=40, height=40,
                             keep_aspect=False).shape[:2], (40, 40))
        fitted = run("resize", self.img, mode="dimensions", width=64, height=64, keep_aspect=True)
        self.assertEqual(fitted.shape[:2], (48, 64))
        self.assertEqual(run("rotate", self.img, angle=90).shape[:2], (w, h))
        self.assertEqual(run("rotate", self.img, angle=180).shape[:2], (h, w))
        self.assertEqual(run("rotate", self.img, angle=45, expand=False).shape[:2], (h, w))
        self.assertGreater(run("rotate", self.img, angle=45, expand=True).shape[1], w)

    def test_rotate_90_is_counter_clockwise(self):
        np.testing.assert_array_equal(run("rotate", self.img, angle=90), np.rot90(self.img, 1))

    def test_thresholds_are_binary(self):
        for op in ("threshold_binary", "threshold_binary_inv"):
            out = run(op, self.img, threshold=100)
            self.assertTrue(set(np.unique(out)) <= {0, 255})
        binary = run("threshold_binary", self.img, threshold=100)
        inverse = run("threshold_binary_inv", self.img, threshold=100)
        np.testing.assert_array_equal(binary, 255 - inverse)
        self.assertTrue(set(np.unique(run("threshold_binary", self.img, use_otsu=True))) <= {0, 255})
        self.assertTrue(set(np.unique(run("threshold_adaptive", self.img))) <= {0, 255})

    def test_edge_detectors_find_the_rectangle(self):
        flat = np.zeros((60, 60, 3), np.uint8)
        flat[15:45, 15:45] = 255
        for op in ("sobel", "canny", "laplacian"):
            with self.subTest(op=op):
                out = run(op, flat)
                self.assertGreater(out.max(), 0)
                self.assertEqual(out[30, 30].tolist(), [0, 0, 0])      # flat interior has no edge

    def test_morphology_ordering(self):
        eroded = run("erosion", self.img, ksize=3)
        dilated = run("dilation", self.img, ksize=3)
        self.assertTrue((eroded <= self.img).all())
        self.assertTrue((dilated >= self.img).all())
        self.assertTrue((run("opening", self.img, ksize=3) <= self.img).all())
        self.assertTrue((run("closing", self.img, ksize=3) >= self.img).all())

    def test_median_removes_salt_noise(self):
        noisy = np.zeros((20, 20, 3), np.uint8)
        noisy[10, 10] = 255
        self.assertEqual(int(run("median_blur", noisy, ksize=3).max()), 0)

    def test_blur_filters_reduce_variance(self):
        noisy = np.random.default_rng(2).integers(90, 170, (64, 64, 3)).astype(np.uint8)
        for op in ("gaussian_blur", "mean_filter", "median_blur"):
            with self.subTest(op=op):
                self.assertLess(run(op, noisy, ksize=7).std(), noisy.std())

    def test_sharpen_increases_local_contrast(self):
        self.assertGreater(run("sharpen", self.img, method="unsharp", amount=2.0).std(), self.img.std() * 0.99)
        self.assertEqual(run("sharpen", self.img, method="kernel", amount=1.0).shape, self.img.shape)

    def test_denoise_methods(self):
        for method in ("nlm", "bilateral"):
            with self.subTest(method=method):
                self.assertLess(run("denoise", self.img, method=method).std(), self.img.std() + 1)

    def test_histogram_equalization_widens_range(self):
        low_contrast = np.random.default_rng(1).integers(100, 121, (64, 64, 3)).astype(np.uint8)
        for method in ("global", "clahe"):
            with self.subTest(method=method):
                out = run("hist_equalization", low_contrast, method=method)
                self.assertGreater(int(out.max()) - int(out.min()),
                                   int(low_contrast.max()) - int(low_contrast.min()))

    def test_channel_extract(self):
        red = run("channel_extract", self.img, channel="r", mode="color")
        self.assertTrue((red[:, :, 0] == 0).all() and (red[:, :, 1] == 0).all())
        np.testing.assert_array_equal(red[:, :, 2], self.img[:, :, 2])
        gray = run("channel_extract", self.img, channel="b", mode="gray")
        np.testing.assert_array_equal(gray[:, :, 0], self.img[:, :, 0])
        np.testing.assert_array_equal(gray[:, :, 1], self.img[:, :, 0])


if __name__ == "__main__":
    unittest.main()
