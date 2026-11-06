"""On-disk workspaces: one folder per uploaded image holding the original and every step."""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from errors import ApiError
from image_io import read_image_file, write_png_file

ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class WorkspaceStore:
    """Creates, reads and deletes workspaces. Only server-generated hex ids ever touch the disk."""

    def __init__(self, root: Path, max_steps: int, ttl_hours: float):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.ttl_seconds = ttl_hours * 3600
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ paths
    def _dir(self, image_id: Any) -> Path:
        if not isinstance(image_id, str) or not ID_PATTERN.match(image_id):
            raise ApiError("Invalid image identifier.", 400)
        return self.root / image_id

    def exists(self, image_id: Any) -> bool:
        try:
            return (self._dir(image_id) / "meta.json").is_file()
        except ApiError:
            return False

    def original_path(self, image_id: str) -> Path:
        return self._dir(image_id) / "original.png"

    def current_path(self, image_id: str, meta: Dict[str, Any]) -> Path:
        steps = meta["steps"]
        return self._dir(image_id) / (steps[-1]["file"] if steps else "original.png")

    # ------------------------------------------------------------------- meta
    def load_meta(self, image_id: str) -> Dict[str, Any]:
        with self.lock:
            try:
                return json.loads((self._dir(image_id) / "meta.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise ApiError("Your image session has expired. Please upload the image again.", 409)

    def _save_meta(self, image_id: str, meta: Dict[str, Any]) -> None:
        path = self._dir(image_id) / "meta.json"
        temp = path.with_name("meta.json.tmp")
        temp.write_text(json.dumps(meta), encoding="utf-8")
        os.replace(temp, path)

    @staticmethod
    def current_size(meta: Dict[str, Any]) -> Tuple[int, int]:
        size = meta["steps"][-1]["size"] if meta["steps"] else meta["original_size"]
        return int(size[0]), int(size[1])

    # ------------------------------------------------------------ pixel data
    def read_original(self, image_id: str) -> np.ndarray:
        return self._read(self.original_path(image_id))

    def read_current(self, image_id: str, meta: Dict[str, Any]) -> np.ndarray:
        return self._read(self.current_path(image_id, meta))

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        img = read_image_file(path)
        if img is None:
            raise ApiError("The stored image is missing or damaged. Please upload it again.", 410)
        return img

    # --------------------------------------------------------------- mutation
    def create(self, filename: str, bgr: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        with self.lock:
            image_id = uuid.uuid4().hex
            folder = self._dir(image_id)
            folder.mkdir(parents=True)
            write_png_file(folder / "original.png", bgr)
            height, width = bgr.shape[:2]
            meta = {"filename": filename, "version": 1, "counter": 0,
                    "original_size": [width, height], "steps": []}
            self._save_meta(image_id, meta)
            return image_id, meta

    def push_step(self, image_id: str, meta: Dict[str, Any], bgr: np.ndarray,
                  operation: str, label: str, params: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            meta["counter"] += 1
            file_name = f"step_{meta['counter']:03d}.png"
            write_png_file(self._dir(image_id) / file_name, bgr)
            height, width = bgr.shape[:2]
            meta["steps"].append({"operation": operation, "label": label, "params": params,
                                  "file": file_name, "size": [width, height]})
            meta["version"] += 1
            self._save_meta(image_id, meta)
            return meta

    def undo(self, image_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if not meta["steps"]:
                raise ApiError("There is nothing to undo.", 400)
            step = meta["steps"].pop()
            self._remove_file(self._dir(image_id) / step["file"])
            meta["version"] += 1
            self._save_meta(image_id, meta)
            return meta

    def reset(self, image_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            for step in meta["steps"]:
                self._remove_file(self._dir(image_id) / step["file"])
            meta["steps"] = []
            meta["counter"] = 0
            meta["version"] += 1
            self._save_meta(image_id, meta)
            return meta

    def delete(self, image_id: str) -> None:
        with self.lock:
            try:
                shutil.rmtree(self._dir(image_id), ignore_errors=True)
            except ApiError:
                pass

    @staticmethod
    def _remove_file(path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            pass   # already gone, or briefly locked on Windows; harmless

    def cleanup_expired(self) -> None:
        """Delete workspaces that have been idle for longer than the time-to-live."""
        now = time.time()
        with self.lock:
            for folder in self.root.iterdir():
                if not folder.is_dir() or not ID_PATTERN.match(folder.name):
                    continue
                marker = folder / "meta.json"
                try:
                    last_used = marker.stat().st_mtime if marker.exists() else folder.stat().st_mtime
                except OSError:
                    continue
                if now - last_used > self.ttl_seconds:
                    shutil.rmtree(folder, ignore_errors=True)
