"""Date filter expressions for time fields, resolved to half-open date ranges [start, end).

Supported expressions (case-insensitive, comma-separated values are OR-ed):

    2026            a calendar year
    2026-09         a calendar month
    2026-09-15      a single day
    2026-Q3         a calendar quarter
    A to B          from the start of A up to, but not including, the start of B (Looker semantics)
    A through B     from the start of A to the end of B (inclusive)
    before A        everything before the start of A
    after A         from the start of A onwards
    today, yesterday
    this week | this month | this quarter | this year     the current period to date
    last week | last month | last quarter | last year     the previous complete period
    last N days | weeks | months | quarters | years      the N previous complete periods
    past N days | weeks | months | quarters | years      the current period to date plus the N-1 before it

"last month" is always the previous full calendar month and "this month" the current month so far,
so a question asked on 2026-10-26 maps "last month" to September and "this month" to 1-26 October.
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional, Tuple

UNITS = {"day": "day", "week": "week", "month": "month", "quarter": "quarter", "year": "year"}


class DateFilterError(ValueError):
    pass


@dataclass(frozen=True)
class DateRange:
    start: Optional[date]  # None = unbounded
    end: Optional[date]  # exclusive; None = unbounded

    def describe(self) -> str:
        if self.start and self.end:
            last = self.end - timedelta(days=1)
            return f"{self.start.isoformat()} to {last.isoformat()}"
        if self.start:
            return f"from {self.start.isoformat()}"
        if self.end:
            return f"before {self.end.isoformat()}"
        return "all dates"


def _add_months(d: date, months: int) -> date:
    total = d.year * 12 + (d.month - 1) + months
    return date(total // 12, total % 12 + 1, 1)


def _period_start(d: date, unit: str) -> date:
    if unit == "day":
        return d
    if unit == "week":
        return d - timedelta(days=d.weekday())
    if unit == "month":
        return d.replace(day=1)
    if unit == "quarter":
        return date(d.year, 3 * ((d.month - 1) // 3) + 1, 1)
    return date(d.year, 1, 1)


def _shift(d: date, unit: str, n: int) -> date:
    if unit == "day":
        return d + timedelta(days=n)
    if unit == "week":
        return d + timedelta(weeks=n)
    if unit == "month":
        return _add_months(d, n)
    if unit == "quarter":
        return _add_months(d, 3 * n)
    return date(d.year + n, d.month, d.day)


def _absolute(token: str) -> Tuple[date, date]:
    """Parse an absolute period and return its [start, end)."""
    t = token.strip().replace("/", "-")
    m = re.fullmatch(r"(\d{4})", t)
    if m:
        y = int(m.group(1))
        return date(y, 1, 1), date(y + 1, 1, 1)
    m = re.fullmatch(r"(\d{4})[- ]?q([1-4])", t, re.I)
    if m:
        start = date(int(m.group(1)), 3 * (int(m.group(2)) - 1) + 1, 1)
        return start, _add_months(start, 3)
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", t)
    if m:
        start = date(int(m.group(1)), int(m.group(2)), 1)
        return start, _add_months(start, 1)
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        start = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return start, start + timedelta(days=1)
    raise DateFilterError(
        f"Unrecognised date '{token}'. Use YYYY, YYYY-MM, YYYY-MM-DD, YYYY-Qn or a relative phrase like 'last month'."
    )


def _relative(token: str, today: date) -> Optional[Tuple[date, date]]:
    t = re.sub(r"\s+", " ", token.strip().lower())
    tomorrow = today + timedelta(days=1)
    if t == "today":
        return today, tomorrow
    if t == "yesterday":
        return today - timedelta(days=1), today
    m = re.fullmatch(r"(this|last) (day|week|month|quarter|year)", t)
    if m:
        unit = UNITS[m.group(2)]
        current = _period_start(today, unit)
        if m.group(1) == "this":
            return current, tomorrow
        return _shift(current, unit, -1), current
    m = re.fullmatch(r"(last|past) (\d+) (day|week|month|quarter|year)s?", t)
    if m:
        n, unit = int(m.group(2)), UNITS[m.group(3)]
        if n < 1:
            raise DateFilterError("The number of periods must be at least 1.")
        current = _period_start(today, unit)
        if m.group(1) == "last":
            return _shift(current, unit, -n), current
        return _shift(current, unit, -(n - 1)), tomorrow
    return None


def _period(token: str, today: date) -> Tuple[date, date]:
    return _relative(token, today) or _absolute(token)


def parse_date_filter(expression: str, today: date) -> List[DateRange]:
    """Resolve a date filter expression into one or more date ranges."""
    ranges: List[DateRange] = []
    for part in [p.strip() for p in str(expression).split(",") if p.strip()]:
        low = part.lower()
        m = re.fullmatch(r"(.+?)\s+(to|through)\s+(.+)", low)
        if m:
            start, _ = _period(m.group(1), today)
            b_start, b_end = _period(m.group(3), today)
            end = b_end if m.group(2) == "through" else b_start
            if end <= start:
                raise DateFilterError(f"Empty date range '{part}'.")
            ranges.append(DateRange(start, end))
        elif low.startswith("before "):
            start, _ = _period(part[7:], today)
            ranges.append(DateRange(None, start))
        elif low.startswith("after "):
            start, _ = _period(part[6:], today)
            ranges.append(DateRange(start, None))
        else:
            start, end = _period(part, today)
            ranges.append(DateRange(start, end))
    if not ranges:
        raise DateFilterError("Empty date filter.")
    return ranges
