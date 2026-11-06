"""Validation of operation names and operation parameters against the specs in image_ops.py."""
from __future__ import annotations

import math
from typing import Any, Dict

from errors import ValidationError
from image_ops import OPERATIONS

_TRUE_STRINGS = {"true", "1", "yes", "on"}
_FALSE_STRINGS = {"false", "0", "no", "off"}


def _fmt(number: float) -> str:
    return f"{number:g}"


def validate_operation(name: Any) -> str:
    """Return the operation key if it is a known operation, otherwise raise."""
    if not isinstance(name, str) or not name.strip():
        raise ValidationError("An 'operation' name is required.")
    key = name.strip()
    if key not in OPERATIONS:
        raise ValidationError(f"Unknown operation: {key[:40]!r}.")
    return key


def _coerce_bool(value: Any, spec: Dict[str, Any]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
    raise ValidationError(f"'{spec['label']}' must be true or false.")


def _coerce_int(value: Any, spec: Dict[str, Any]) -> int:
    label = spec["label"]
    error = ValidationError(
        f"'{label}' must be a whole number between {_fmt(spec['min'])} and {_fmt(spec['max'])}.")
    if isinstance(value, bool):
        raise error
    number: float
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        number = value
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            raise error
    else:
        raise error
    if not math.isfinite(number) or not float(number).is_integer():
        raise error
    result = int(number)
    if result < spec["min"] or result > spec["max"]:
        raise ValidationError(f"'{label}' must be between {_fmt(spec['min'])} and {_fmt(spec['max'])}.")
    if spec.get("odd") and result % 2 == 0:
        raise ValidationError(f"'{label}' must be an odd number.")
    return result


def _coerce_float(value: Any, spec: Dict[str, Any]) -> float:
    label = spec["label"]
    error = ValidationError(
        f"'{label}' must be a number between {_fmt(spec['min'])} and {_fmt(spec['max'])}.")
    if isinstance(value, bool):
        raise error
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            raise error
    else:
        raise error
    if not math.isfinite(number):
        raise error
    if number < spec["min"] or number > spec["max"]:
        raise ValidationError(f"'{label}' must be between {_fmt(spec['min'])} and {_fmt(spec['max'])}.")
    return number


def _coerce_choice(value: Any, spec: Dict[str, Any]) -> Any:
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        for choice in spec["choices"]:
            if str(choice["value"]) == str(value).strip():
                return choice["value"]          # keep the original (typed) value
    allowed = ", ".join(str(c["value"]) for c in spec["choices"])
    raise ValidationError(f"'{spec['label']}' must be one of: {allowed}.")


_COERCERS = {"bool": _coerce_bool, "int": _coerce_int, "float": _coerce_float, "choice": _coerce_choice}


def validate_params(operation_key: str, raw: Any) -> Dict[str, Any]:
    """Validate *raw* parameters, apply defaults for missing ones and reject unknown ones."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValidationError("'params' must be a JSON object.")

    specs = OPERATIONS[operation_key]["params"]
    known = {spec["name"] for spec in specs}
    unknown = sorted(str(key)[:30] for key in raw if key not in known)
    if unknown:
        raise ValidationError(f"Unknown parameter(s) for '{operation_key}': {', '.join(unknown)}.")

    clean: Dict[str, Any] = {}
    for spec in specs:
        value = raw.get(spec["name"], spec["default"])
        clean[spec["name"]] = _COERCERS[spec["type"]](value, spec)
    return clean
