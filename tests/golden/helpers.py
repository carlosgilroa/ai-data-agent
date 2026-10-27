"""Shared helpers for the golden question tests."""

import json
import math
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, List, Sequence

GOLDEN_FILE = Path(__file__).parent / "questions.json"

# Money is compared to the peso; percentages and ratios to 0.01.
ABS_TOLERANCE = 0.01
REL_TOLERANCE = 1e-6


def load_golden() -> dict:
    return json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))


def normalize(value: Any) -> Any:
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and math.isnan(value) else float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    return str(value)


def values_match(a: Any, b: Any) -> bool:
    a, b = normalize(a), normalize(b)
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=REL_TOLERANCE, abs_tol=ABS_TOLERANCE)
    return a == b


def _sort_key(row: Sequence[Any]):
    return tuple((v is None, str(normalize(v))) for v in row)


def rows_match(actual: List[Sequence[Any]], expected: List[Sequence[Any]], ordered: bool) -> str:
    """Return an empty string when the row sets match, else a description of the first difference."""
    if len(actual) != len(expected):
        return f"row count differs: got {len(actual)}, expected {len(expected)}"
    if not ordered:
        actual, expected = sorted(actual, key=_sort_key), sorted(expected, key=_sort_key)
    for i, (a, e) in enumerate(zip(actual, expected)):
        if len(a) != len(e):
            return f"row {i}: got {len(a)} columns, expected {len(e)}"
        if not all(values_match(x, y) for x, y in zip(a, e)):
            return f"row {i}: got {list(a)}, expected {list(e)}"
    return ""
