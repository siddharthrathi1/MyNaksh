"""Resolve relative time expressions against the message timestamp.

"next year" said in 2026 must be stored as 2027 — otherwise the memory silently becomes wrong
a year later. We store both the resolved value and the raw phrase (for explainability).
Resolution granularity matches what the user said: year → "YYYY", month → "YYYY-MM",
day/week → "YYYY-MM-DD".
"""

import calendar
import re
from datetime import datetime, timedelta

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "a": 1, "an": 1, "couple of": 2, "few": 3}

TIMEFRAME_PATTERN = re.compile(
    r"\b(next year|this year|next month|this month|next week|tomorrow|"
    r"in (?:\d+|one|two|three|four|five|six|a|an|a couple of|a few) (?:years?|months?|weeks?|days?)|"
    r"(?:by|in|before) (?:20\d{2})|"
    r"(?:by|in) (?:" + "|".join(_MONTHS) + r")(?: 20\d{2})?)\b",
    re.I,
)


def _add_months(dt: datetime, months: int) -> datetime:
    month_index = dt.month - 1 + months
    year, month = dt.year + month_index // 12, month_index % 12 + 1
    return dt.replace(year=year, month=month, day=min(dt.day, calendar.monthrange(year, month)[1]))


def find_timeframe(text: str) -> str | None:
    match = TIMEFRAME_PATTERN.search(text)
    return match.group(0) if match else None


def resolve_timeframe(raw: str, now: datetime) -> str | None:
    phrase = raw.lower().strip()
    if phrase == "next year":
        return str(now.year + 1)
    if phrase == "this year":
        return str(now.year)
    if phrase == "next month":
        return _add_months(now, 1).strftime("%Y-%m")
    if phrase == "this month":
        return now.strftime("%Y-%m")
    if phrase == "next week":
        return (now + timedelta(weeks=1)).date().isoformat()
    if phrase == "tomorrow":
        return (now + timedelta(days=1)).date().isoformat()

    if m := re.fullmatch(r"(?:by|in|before) (20\d{2})", phrase):
        return m.group(1)

    if m := re.fullmatch(r"in (\d+|one|two|three|four|five|six|a|an|a couple of|a few) (year|month|week|day)s?", phrase):
        qty_raw = m.group(1).replace("a couple of", "couple of").replace("a few", "few")
        qty = int(qty_raw) if qty_raw.isdigit() else _NUM_WORDS.get(qty_raw, 1)
        unit = m.group(2)
        if unit == "year":
            return str(now.year + qty)
        if unit == "month":
            return _add_months(now, qty).strftime("%Y-%m")
        days = qty * 7 if unit == "week" else qty
        return (now + timedelta(days=days)).date().isoformat()

    if m := re.fullmatch(r"(?:by|in) ([a-z]+)(?: (20\d{2}))?", phrase):
        month = _MONTHS.get(m.group(1))
        if month:
            year = int(m.group(2)) if m.group(2) else (now.year if month >= now.month else now.year + 1)
            return f"{year}-{month:02d}"
    return None
