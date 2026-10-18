"""Pacing and scan-order helpers for cTrader historical trendbar requests.

Spotware documents ~5 ProtoOAGetTrendbarsReq per second. Bursting one
request per symbol back-to-back trips BLOCKED_PAYLOAD_TYPE; a 180s blanket
cooldown then starves every symbol after the first six in a fixed list.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Callable

from backend.services.multi_asset_bars import classify_symbol

TRENDBAR_CACHE_TTL_SEC = 45.0
DEFAULT_BAR_REQ_PER_SEC = 4.0
DEFAULT_RATE_LIMIT_COOLDOWN_SEC = 45.0
RATE_LIMIT_COOLDOWN_CAP_SEC = 300.0


def env_bar_req_per_sec() -> float:
    try:
        return max(0.1, float(os.getenv("CTRADER_BAR_REQ_PER_SEC", "4") or 4))
    except (TypeError, ValueError):
        return DEFAULT_BAR_REQ_PER_SEC


def env_rate_limit_cooldown_sec() -> float:
    try:
        return max(1.0, float(os.getenv("CTRADER_RATE_LIMIT_COOLDOWN_SEC", "45") or 45))
    except (TypeError, ValueError):
        return DEFAULT_RATE_LIMIT_COOLDOWN_SEC


def rate_limit_cooldown_sec(strikes: int, base: float | None = None, cap: float = RATE_LIMIT_COOLDOWN_CAP_SEC) -> float:
    """Exponential backoff: base, 2*base, 4*base, ... capped.

    ``strikes`` is 1-based (first BLOCKED_PAYLOAD_TYPE → base cooldown).
    """
    b = env_rate_limit_cooldown_sec() if base is None else max(1.0, float(base))
    n = max(1, int(strikes))
    return min(b * (2 ** (n - 1)), float(cap))


def rotate_symbols(symbols: Sequence[str], cursor: int) -> list[str]:
    items = list(symbols)
    n = len(items)
    if n == 0:
        return []
    i = int(cursor) % n
    if i == 0:
        return items
    return items[i:] + items[:i]


def fair_symbol_order(
    symbols: Sequence[str],
    last_ok_at: Mapping[str, float] | None = None,
    cursor: int = 0,
) -> list[str]:
    """Starved cTrader symbols first; crypto keeps its relative tail order.

    ``last_ok_at`` is last successful live trendbar receipt (not cache TTL).
    Never-fetched symbols sort before recently served ones so a rate-limit
    mid-scan cannot lock the book onto the same six majors. ``cursor`` rotates
    ties so two all-cold scans do not start on the same symbol.
    """
    served = {str(key).upper(): float(value) for key, value in (last_ok_at or {}).items()}
    ctrader: list[str] = []
    other: list[str] = []
    for symbol in symbols:
        if classify_symbol(symbol) == "crypto":
            other.append(symbol)
        else:
            ctrader.append(symbol)
    rotated = rotate_symbols(ctrader, cursor)
    rotated.sort(key=lambda sym: served.get(str(sym).upper(), 0.0))
    return rotated + other


class TrendbarRequestPacer:
    """Token-bucket pacer for ProtoOAGetTrendbarsReq (default <= 4/sec)."""

    def __init__(
        self,
        rate_per_sec: float | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        rate = DEFAULT_BAR_REQ_PER_SEC if rate_per_sec is None else float(rate_per_sec)
        self.rate_per_sec = max(0.1, rate)
        self.capacity = self.rate_per_sec
        self._tokens = self.capacity
        self._monotonic = monotonic
        self._sleep = sleep
        self._ts = monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Reserve one historical-data slot; sleep if the bucket is empty.

        Returns the seconds waited (0 if the token was immediately available).
        """
        with self._lock:
            now = self._monotonic()
            elapsed = max(0.0, now - self._ts)
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_sec)
            self._ts = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0
            wait = (1.0 - self._tokens) / self.rate_per_sec
            self._tokens -= 1.0
        if wait > 0:
            self._sleep(wait)
        return wait
