"""Token-bucket rate limiter for external data APIs.

Prevents free-tier bans (Alpha Vantage 25/day, CoinGecko 30/min, etc.)
and gives the feed scheduler honest backpressure instead of silent 429s.

Design guarantees:
  1. Waiting callers reserve their token inside the lock (``tokens`` may dip
     negative) so concurrent callers cannot overshoot the quota.
  2. Both synchronous (``acquire`` / ``cached_call``) and asynchronous
     (``acquire_async`` / ``cached_call_async``) entry points exist so the
     async trading loop is never blocked by ``time.sleep``.
  3. Skip beats retry-storm; stale cache beats nothing.
  4. Optional Telegram alert fires only after ``ALERT_AFTER_SKIPS`` consecutive
     skips per provider to prevent alert fatigue.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

ALERT_AFTER_SKIPS = int(os.getenv("RATE_LIMIT_ALERT_AFTER_SKIPS", "3"))


@dataclass
class Bucket:
    capacity: float
    refill_per_sec: float
    tokens: float | None = None
    ts: float = 0.0

    def __post_init__(self) -> None:
        if self.tokens is None:
            self.tokens = float(self.capacity)
        self.ts = time.monotonic()

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self.ts)
        self.tokens = min(
            self.capacity,
            float(self.tokens if self.tokens is not None else self.capacity) + elapsed * self.refill_per_sec,
        )
        self.ts = now

    def take(self, n: float = 1.0, max_wait: float | None = None) -> float:
        """Consume or reserve ``n`` tokens; returns seconds needed to wait (0 = immediate).

        When ``max_wait`` is provided and the required wait exceeds ``max_wait``,
        tokens are NOT deducted so a rejected caller does not starve subsequent callers.
        Otherwise tokens are deducted immediately (even if negative) to serialize
        concurrent waiters accurately.
        """
        now = time.monotonic()
        self._refill(now)
        avail = float(self.tokens or 0.0)
        if avail >= n:
            self.tokens = avail - n
            return 0.0
        needed = n - avail
        wait = needed / self.refill_per_sec if self.refill_per_sec > 0 else float("inf")
        if max_wait is not None and wait > max_wait:
            return wait
        self.tokens = avail - n
        return wait


_lock = threading.Lock()
_buckets: dict[str, Bucket] = {}
_skip_counts: dict[str, int] = {}

# (capacity, refill/sec) — conservative for free tiers
_LIMITS: dict[str, tuple[float, float]] = {
    "alphavantage":  (4.0,  4.0 / 86400.0),   # ~4/day headroom on 25/day
    "coingecko":     (10.0, 10.0 / 60.0),     # 10 burst, 10/min sustained
    "coinmarketcap": (10.0, 10.0 / 60.0),
    "twelvedata":    (4.0,  4.0 / 60.0),      # 8/min -> use half
    "polygon":       (2.0,  2.0 / 60.0),
    "newsapi":       (20.0, 20.0 / 86400.0),  # free tier is daily
    "binance":       (20.0, 20.0 / 60.0),
    "fred":          (60.0, 60.0 / 60.0),
    "omniroute":     (30.0, 30.0 / 60.0),
}


def _reserve(provider: str, n: float, max_wait: float) -> tuple[bool, float, int]:
    """Internal lock-protected reservation. Returns (ok, wait_sec, consecutive_skips)."""
    with _lock:
        b = _buckets.get(provider)
        if b is None:
            cap, rate = _LIMITS.get(provider, (5.0, 5.0 / 60.0))
            b = _buckets[provider] = Bucket(cap, rate)
        wait = b.take(n, max_wait=max_wait)
        if wait > max_wait:
            skips = _skip_counts[provider] = _skip_counts.get(provider, 0) + 1
            return False, wait, skips
        _skip_counts[provider] = 0
        return True, wait, 0


def _maybe_notify_skip(provider: str, wait: float, skips: int) -> None:
    if skips != ALERT_AFTER_SKIPS:
        return
    msg = (
        f"⚠️ *Rate Limiter*: provider `{provider}` skipped {skips} consecutive times "
        f"(required wait {wait:.0f}s). Serving stale cache / skipping feed."
    )
    try:
        from backend.utils.telegram import send_telegram_message
        loop = asyncio.get_running_loop()
        loop.create_task(send_telegram_message(msg))
    except Exception:
        logger.debug("rate_limit: telegram skip notification not dispatched for %s", provider)


def acquire(provider: str, n: float = 1.0, max_wait: float = 30.0) -> bool:
    """True if tokens acquired (having waited <= max_wait), else False.
    Caller should treat False as 'skip this provider this cycle', not error."""
    ok, wait, skips = _reserve(provider, n, max_wait)
    if not ok:
        logger.info("rate_limit: %s wait %.1fs > %ss, skipping (skips=%d)", provider, wait, max_wait, skips)
        _maybe_notify_skip(provider, wait, skips)
        return False
    if wait > 0:
        logger.debug("rate_limit: %s waiting %.2fs", provider, wait)
        time.sleep(wait)
    return True


async def acquire_async(provider: str, n: float = 1.0, max_wait: float = 30.0) -> bool:
    """Async equivalent of :func:`acquire` that never blocks the event loop."""
    ok, wait, skips = _reserve(provider, n, max_wait)
    if not ok:
        logger.info("rate_limit: %s wait %.1fs > %ss, skipping (skips=%d)", provider, wait, max_wait, skips)
        _maybe_notify_skip(provider, wait, skips)
        return False
    if wait > 0:
        logger.debug("rate_limit: %s awaiting %.2fs", provider, wait)
        await asyncio.sleep(wait)
    return True


def cached_call(
    provider: str,
    cache: dict[str, tuple[float, Any]],
    key: str,
    ttl_sec: float,
    fn: Callable[[], T],
    max_wait: float = 30.0,
) -> T | None:
    """Cache wrapper: serve fresh cache, else acquire token, else serve stale cache."""
    now = time.time()
    hit = cache.get(key)
    if hit and (now - hit[0]) < ttl_sec:
        return hit[1]
    if not acquire(provider, max_wait=max_wait):
        return hit[1] if hit else None
    try:
        val = fn()
        cache[key] = (time.time(), val)
        return val
    except Exception as e:
        logger.warning("rate_limit: %s call failed: %s", provider, e)
        return hit[1] if hit else None


async def cached_call_async(
    provider: str,
    cache: dict[str, tuple[float, Any]],
    key: str,
    ttl_sec: float,
    fn: Callable[[], Awaitable[T]],
    max_wait: float = 30.0,
) -> T | None:
    """Async cache wrapper: serve fresh cache, else acquire_async, else serve stale cache."""
    now = time.time()
    hit = cache.get(key)
    if hit and (now - hit[0]) < ttl_sec:
        return hit[1]
    if not await acquire_async(provider, max_wait=max_wait):
        return hit[1] if hit else None
    try:
        val = await fn()
        cache[key] = (time.time(), val)
        return val
    except Exception as e:
        logger.warning("rate_limit: %s async call failed: %s", provider, e)
        return hit[1] if hit else None


def get_rate_limiter_status() -> dict[str, Any]:
    """Snapshot of active token buckets and consecutive skip counters."""
    with _lock:
        now = time.monotonic()
        snapshot: dict[str, Any] = {}
        for name, b in _buckets.items():
            b._refill(now)
            snapshot[name] = {
                "tokens": round(float(b.tokens or 0.0), 2),
                "capacity": b.capacity,
                "refill_per_sec": round(b.refill_per_sec, 6),
                "consecutive_skips": _skip_counts.get(name, 0),
            }
        return snapshot
