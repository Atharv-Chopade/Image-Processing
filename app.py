"""PixelLab - Digital Image Processing Studio (Flask back end).

API overview (all responses are JSON unless noted):

    GET  /                       the dashboard page
    GET  /api/operations         operation catalogue + parameter specs + limits
    GET  /api/state              current image state of this browser session
    POST /api/upload             multipart form field "image"
    POST /api/process            {"operation": "...", "params": {...}}   (sequential)
    POST /api/undo               undo the last operation
    POST /api/reset              back to the original image
    GET  /api/image/<kind>       PNG bytes, kind = original | processed
    GET  /api/download           processed image as a file (?format=png|jpg|webp&quality=1-100)
    POST /api/histogram          histogram plots + statistics (original and processed)
    POST /api/channels           {"source": "processed|original", "mode": "color|gray"}
"""
from __future__ import annotations

import io
import os
import secrets
from pathlib import Path
from typing import Any, Dict

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_file, session
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

import config
from analysis import channel_previews, histogram_report
from errors import ApiError, ValidationError
from image_io import clean_filename, decode_image, encode_image
from image_ops import OPERATIONS, apply_operation, public_specs
from storage import WorkspaceStore
from validators import validate_operation, validate_params

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self' https://cdn.jsdelivr.net; "
    "style-src 'self' https://cdn.jsdelivr.net; "
    "font-src 'self' https://cdn.jsdelivr.net data:; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'; "
    "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
)


def _load_secret_key(path: Path) -> str:
    """Use DIP_SECRET_KEY if set, otherwise create and reuse instance/secret.key."""
    env_key = os.environ.get("DIP_SECRET_KEY")
    if env_key:
        return env_key
    try:
        if path.is_file():
            stored = path.read_text(encoding="utf-8").strip()
            if len(stored) >= 32:
                return stored
        generated = secrets.token_hex(32)
        path.write_text(generated, encoding="utf-8")
        return generated
    except OSError:
        return secrets.token_hex(32)


def create_app(test_config: Dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config.Config)
    if test_config:
        app.config.update(test_config)
    app.config["MAX_CONTENT_LENGTH"] = int(app.config["MAX_UPLOAD_MB"]) * 1024 * 1024 + 64 * 1024
    app.json.sort_keys = False

    instance_dir = Path(app.config["INSTANCE_DIR"])
    instance_dir.mkdir(parents=True, exist_ok=True)
    app.secret_key = app.config.get("SECRET_KEY") or _load_secret_key(instance_dir / "secret.key")

    store = WorkspaceStore(app.config["WORKSPACE_DIR"], app.config["MAX_STEPS"],
                           app.config["WORKSPACE_TTL_HOURS"])
    store.cleanup_expired()

    max_upload_bytes = int(app.config["MAX_UPLOAD_MB"]) * 1024 * 1024

    # ------------------------------------------------------------------ helpers
    def require_image_id() -> str:
        image_id = session.get("image_id")
        if not image_id or not store.exists(image_id):
            session.pop("image_id", None)
            raise ApiError("No image loaded. Please upload an image first.", 409)
        return image_id

    def state_payload(image_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        orig_w, orig_h = meta["original_size"]
        cur_w, cur_h = store.current_size(meta)
        return {
            "ok": True,
            "loaded": True,
            "filename": meta["filename"],
            "token": f"{image_id[:8]}-{meta['version']}",
            "original_size": {"width": orig_w, "height": orig_h},
            "current_size": {"width": cur_w, "height": cur_h},
            "history": [{"index": i + 1, "operation": s["operation"], "label": s["label"],
                         "params": s["params"]} for i, s in enumerate(meta["steps"])],
            "can_undo": bool(meta["steps"]),
            "max_steps": app.config["MAX_STEPS"],
        }

    def png_response(data: bytes) -> Response:
        response = Response(data, mimetype="image/png")
        response.headers["Cache-Control"] = "no-store"
        return response

    # ------------------------------------------------------------------- pages
    @app.get("/")
    def index():
        return render_template("index.html")

    # --------------------------------------------------------------------- API
    @app.get("/api/operations")
    def list_operations():
        payload = public_specs()
        payload["ok"] = True
        payload["limits"] = {
            "max_upload_mb": app.config["MAX_UPLOAD_MB"],
            "allowed_extensions": sorted(config.ALLOWED_EXTENSIONS),
            "max_dimension": config.MAX_IMAGE_DIM,
            "max_result_dimension": config.MAX_RESULT_DIM,
            "max_steps": app.config["MAX_STEPS"],
        }
        return jsonify(payload)

    @app.get("/api/state")
    def get_state():
        image_id = session.get("image_id")
        if image_id and store.exists(image_id):
            return jsonify(state_payload(image_id, store.load_meta(image_id)))
        return jsonify({"ok": True, "loaded": False})

    @app.post("/api/upload")
    def upload():
        file = request.files.get("image")
        if file is None or not file.filename:
            raise ValidationError("No file received. Choose an image to upload.")
        filename = clean_filename(file.filename)
        raw = file.stream.read(max_upload_bytes + 1)
        if len(raw) > max_upload_bytes:
            raise ApiError(f"The file is too large. The limit is {app.config['MAX_UPLOAD_MB']} MB.", 413)
        bgr, info = decode_image(raw)

        with store.lock:
            previous = session.get("image_id")
            image_id, meta = store.create(filename, bgr)
            if previous and previous != image_id:
                store.delete(previous)
            session.permanent = True
            session["image_id"] = image_id
            store.cleanup_expired()

        payload = state_payload(image_id, meta)
        if info["downscaled"]:
            height, width = bgr.shape[:2]
            payload["notice"] = (f"The image was larger than {config.MAX_IMAGE_DIM}px, so it was "
                                 f"scaled down to {width} x {height} for faster processing.")
        return jsonify(payload), 201

    @app.post("/api/process")
    def process():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ValidationError("The request body must be a JSON object.")
        key = validate_operation(body.get("operation"))
        params = validate_params(key, body.get("params"))
        image_id = require_image_id()

        with store.lock:
            meta = store.load_meta(image_id)
            if len(meta["steps"]) >= app.config["MAX_STEPS"]:
                raise ApiError(f"Step limit reached ({app.config['MAX_STEPS']}). "
                               "Undo a step, reset, or download the result and upload it again.", 409)
            image = store.read_current(image_id, meta)
            try:
                result = apply_operation(key, image, params)
            except ApiError:
                raise
            except cv2.error:
                app.logger.exception("OpenCV failed for %s %s", key, params)
                raise ValidationError("OpenCV could not apply this operation with the given parameters.")
            meta = store.push_step(image_id, meta, result, key, OPERATIONS[key]["label"], params)
        return jsonify(state_payload(image_id, meta))

    @app.post("/api/undo")
    def undo():
        image_id = require_image_id()
        with store.lock:
            meta = store.undo(image_id, store.load_meta(image_id))
        return jsonify(state_payload(image_id, meta))

    @app.post("/api/reset")
    def reset():
        image_id = require_image_id()
        with store.lock:
            meta = store.reset(image_id, store.load_meta(image_id))
        return jsonify(state_payload(image_id, meta))

    @app.get("/api/image/<kind>")
    def get_image(kind: str):
        if kind not in ("original", "processed"):
            raise ApiError("Unknown image kind.", 404)
        image_id = require_image_id()
        with store.lock:
            meta = store.load_meta(image_id)
            path = (store.original_path(image_id) if kind == "original"
                    else store.current_path(image_id, meta))
            try:
                data = path.read_bytes()          # read fully so no file handle stays open (Windows)
            except OSError:
                raise ApiError("The image file is missing. Please upload it again.", 404)
        return png_response(data)

    @app.get("/api/download")
    def download():
        fmt = request.args.get("format", "png").lower()
        if fmt not in ("png", "jpg", "jpeg", "webp"):
            raise ValidationError("Format must be one of: png, jpg, webp.")
        try:
            quality = int(request.args.get("quality", "95"))
        except ValueError:
            raise ValidationError("Quality must be a whole number from 1 to 100.")
        if not 1 <= quality <= 100:
            raise ValidationError("Quality must be a whole number from 1 to 100.")

        image_id = require_image_id()
        with store.lock:
            meta = store.load_meta(image_id)
            image = store.read_current(image_id, meta)
        data, mimetype, extension = encode_image(image, fmt, quality)
        stem = Path(meta["filename"]).stem or "image"
        response = send_file(io.BytesIO(data), mimetype=mimetype, as_attachment=True,
                             download_name=f"{stem}_processed.{extension}")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/api/histogram")
    def histogram():
        image_id = require_image_id()
        with store.lock:
            meta = store.load_meta(image_id)
            original = store.read_original(image_id)
            processed = store.read_current(image_id, meta)
        return jsonify({"ok": True, "original": histogram_report(original),
                        "processed": histogram_report(processed)})

    @app.post("/api/channels")
    def channels():
        body = request.get_json(silent=True)
        if body is None:
            body = {}
        if not isinstance(body, dict):
            raise ValidationError("The request body must be a JSON object.")
        source = body.get("source", "processed")
        mode = body.get("mode", "color")
        if source not in ("processed", "original"):
            raise ValidationError("'source' must be 'processed' or 'original'.")
        if mode not in ("color", "gray"):
            raise ValidationError("'mode' must be 'color' or 'gray'.")

        image_id = require_image_id()
        with store.lock:
            meta = store.load_meta(image_id)
            image = (store.read_original(image_id) if source == "original"
                     else store.read_current(image_id, meta))
        return jsonify({"ok": True, "source": source, "mode": mode,
                        "channels": channel_previews(image, mode)})

    # ------------------------------------------------------------ error handling
    @app.errorhandler(ApiError)
    def handle_api_error(err: ApiError):
        return jsonify(ok=False, error=err.message), err.status_code

    @app.errorhandler(HTTPException)
    def handle_http_error(err: HTTPException):
        if not request.path.startswith("/api/"):
            return err
        if isinstance(err, RequestEntityTooLarge):
            message = f"The file is too large. The limit is {app.config['MAX_UPLOAD_MB']} MB."
        else:
            message = err.description or err.name
        return jsonify(ok=False, error=message), err.code

    @app.errorhandler(Exception)
    def handle_unexpected(err: Exception):
        app.logger.exception("Unhandled error")
        if request.path.startswith("/api/"):
            return jsonify(ok=False, error="An unexpected server error occurred."), 500
        return "Internal server error", 500

    @app.after_request
    def add_security_headers(response: Response) -> Response:
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        if request.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")),
            debug=os.environ.get("FLASK_DEBUG") == "1")
