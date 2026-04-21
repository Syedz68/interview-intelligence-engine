"""
app/core/numpy_encoder.py
--------------------------
Utility to recursively convert numpy scalars/arrays to native Python
types so Pydantic v2 / FastAPI can serialize them without errors.

Usage
-----
Call ``sanitize(obj)`` on any dict/list before returning it from a
service, or register ``NumpyEncoder`` as a custom JSON encoder.
"""

from __future__ import annotations

from typing import Any


def sanitize(obj: Any) -> Any:
    """
    Recursively walk ``obj`` and convert any numpy scalar or array to a
    plain Python float/int/list. Safe to call on plain Python objects too.
    """
    try:
        import numpy as np  # lazy – no hard dependency at import time
    except ImportError:
        return obj  # numpy not installed; nothing to do

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [sanitize(v) for v in obj]
        return converted if isinstance(obj, list) else tuple(converted)
    return obj