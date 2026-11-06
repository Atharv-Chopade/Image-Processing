"""Exceptions that are converted into JSON error responses by ``app.py``."""
from __future__ import annotations

from typing import Optional


class ApiError(Exception):
    """An error whose message is safe to show to the API client."""

    status_code = 400

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class ValidationError(ApiError):
    """The request (file, operation name or parameters) is invalid."""

    status_code = 400
