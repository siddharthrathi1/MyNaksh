from collections.abc import Callable
from datetime import datetime, timezone

# Injected everywhere "now" matters (relative-date resolution, recency scoring),
# so tests can pin time and get deterministic results.
Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
