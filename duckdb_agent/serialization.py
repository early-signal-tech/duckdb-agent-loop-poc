"""Convert DuckDB values to plain JSON."""

import json
import math
from decimal import Decimal


def to_jsonable(value):
    """Convert DuckDB result values to plain JSON types."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # NaN/Infinity are not valid JSON.
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, bytes):
        return value.hex()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)  # UUID, timedelta, and anything else


def json_dumps(obj) -> str:
    """Serialize query results as compact, valid JSON (non-ASCII kept as-is)."""
    return json.dumps(to_jsonable(obj), ensure_ascii=False, allow_nan=False)
