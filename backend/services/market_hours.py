"""Venue session hours for entry filters.

Crypto is 24/7. cTrader FX / metals / oil follow the existing QuantumTrade
window (Sun 21:00 UTC – Fri 21:00 UTC). Used to skip weekend IC order
attempts without weakening weekday fail-closed entry filters.

Daily-break awareness (IC / most FX books pause ~21:00–22:00 UTC on
weekdays) and ``entry_cutoff_minutes_before_close`` live here. The
candidate engine only *enforces* the cutoff when
``CTRADER_ENTRY_CUTOFF_MODE=enforce``; default is ``shadow``.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from backend.services.multi_asset_bars import classify_symbol

_CRYPTO_YFINANCE = {"BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"}


def _is_ic_crypto_cfd(symbol: str) -> bool:
    """IC Markets crypto CFDs (BTCUSD) — 24/7 including weekend, daily break applies."""
    from backend.services.ctrader_crypto import is_ic_crypto_symbol

    return is_ic_crypto_symbol(symbol)

# Weekly window (UTC). Matches the historical QuantumTrade / IC session.
SESSION_OPEN_WEEKDAY = 6  # Sunday
SESSION_OPEN_HOUR = 21
SESSION_CLOSE_WEEKDAY = 4  # Friday
SESSION_CLOSE_HOUR = 21


def is_crypto_symbol(symbol: str) -> bool:
    """Match the trading-loop crypto set so weekend skip does not close perps."""
    if classify_symbol(symbol) == "crypto":
        return True
    upper = symbol.upper()
    if upper.endswith(("USDT", "USDC", "BUSD")):
        return True
    if upper in _CRYPTO_YFINANCE or (upper.endswith("-USD") and "=" not in upper):
        return True
    return False


def _as_utc(now: Optional[datetime]) -> datetime:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        return current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def _parse_hhmm(raw: str, default_hour: int, default_minute: int) -> tuple[int, int]:
    text = (raw or "").strip()
    if not text:
        return default_hour, default_minute
    try:
        parts = text.replace(".", ":").split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute
    except (TypeError, ValueError):
        pass
    return default_hour, default_minute


def daily_break_window() -> tuple[int, int, int, int]:
    """Return (start_hour, start_min, end_hour, end_min) in UTC.

    Default 21:00–22:00 UTC — the common IC / cTrader daily maintenance
    gap on weekdays. Friday 21:00 is already the weekly close.
    """
    start_h, start_m = _parse_hhmm(os.getenv("CTRADER_DAILY_BREAK_START_UTC", "21:00"), 21, 0)
    end_h, end_m = _parse_hhmm(os.getenv("CTRADER_DAILY_BREAK_END_UTC", "22:00"), 22, 0)
    return start_h, start_m, end_h, end_m


def is_daily_break(now: Optional[datetime] = None) -> bool:
    """True during the weekday daily maintenance window (not the weekend)."""
    current = _as_utc(now)
    weekday = current.weekday()
    if weekday >= 5:  # Sat/Sun handled by the weekly window
        return False
    if weekday == SESSION_CLOSE_WEEKDAY and current.hour >= SESSION_CLOSE_HOUR:
        return False  # already weekly-closed
    start_h, start_m, end_h, end_m = daily_break_window()
    start = current.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    end = current.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    if end <= start:
        end = end + timedelta(days=1)
    return start <= current < end


def is_venue_open(symbol: str, now: datetime | None = None) -> bool:
    """True when new entries may be sent for ``symbol``.

    Crypto (USDT/USDC perps): always open.
    FX, metals, oil, index: closed Fri 21:00 UTC through Sun 21:00 UTC.

    Daily-break is *not* folded into this predicate so existing weekend
    skip tests and live weekday filters stay unchanged. Use
    ``is_daily_break`` / ``entry_cutoff_blocks`` for the new cutoff.
    """
    if is_crypto_symbol(symbol):
        return True

    current = _as_utc(now)
    weekday = current.weekday()  # Mon=0 ... Sun=6
    if weekday == SESSION_CLOSE_WEEKDAY and current.hour >= SESSION_CLOSE_HOUR:
        return False
    if weekday == 5:  # Saturday
        return False
    if weekday == SESSION_OPEN_WEEKDAY and current.hour < SESSION_OPEN_HOUR:
        return False
    return True


def next_venue_close(symbol: str, now: Optional[datetime] = None) -> Optional[datetime]:
    """Next session close or daily-break start. None for 24/7 Binance crypto.

    IC crypto CFDs stay open on the weekend but still hit the weekday
    daily-break window, so the next close is that break start.
    """
    if is_crypto_symbol(symbol) and not _is_ic_crypto_cfd(symbol):
        return None
    current = _as_utc(now)
    if is_crypto_symbol(symbol) and _is_ic_crypto_cfd(symbol):
        start_h, start_m, _end_h, _end_m = daily_break_window()
        break_today = current.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        if current.weekday() < 5 and current < break_today:
            return break_today
        # Weekend / after today's break → next weekday break.
        days = 1
        cursor = current
        while days <= 7:
            cursor = current + timedelta(days=days)
            if cursor.weekday() < 5:
                return cursor.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
            days += 1
        return None
    if not is_venue_open(symbol, current):
        return current
    start_h, start_m, _end_h, _end_m = daily_break_window()
    break_today = current.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    # Weekday daily break that has not started yet, and is before Friday close.
    if current.weekday() < SESSION_CLOSE_WEEKDAY and current < break_today:
        return break_today
    if current.weekday() == SESSION_CLOSE_WEEKDAY:
        friday_close = current.replace(
            hour=SESSION_CLOSE_HOUR, minute=0, second=0, microsecond=0
        )
        if current < friday_close:
            # Daily-break start and Friday close often coincide at 21:00 UTC.
            if current < break_today:
                return min(break_today, friday_close)
            return friday_close
    # Walk forward to the coming Friday 21:00 UTC.
    days_ahead = (SESSION_CLOSE_WEEKDAY - current.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    friday = (current + timedelta(days=days_ahead)).replace(
        hour=SESSION_CLOSE_HOUR, minute=0, second=0, microsecond=0
    )
    return friday


def time_until_venue_close(symbol: str, now: Optional[datetime] = None) -> Optional[timedelta]:
    """timedelta until the next venue close, or None if the venue is 24/7."""
    close_at = next_venue_close(symbol, now)
    if close_at is None:
        return None
    current = _as_utc(now)
    delta = close_at - current
    return delta if delta.total_seconds() > 0 else timedelta(0)


def entry_cutoff_minutes() -> float:
    try:
        return max(0.0, float(os.getenv("CTRADER_ENTRY_CUTOFF_MIN", "30") or 30))
    except (TypeError, ValueError):
        return 30.0


def entry_cutoff_mode() -> str:
    raw = (os.getenv("CTRADER_ENTRY_CUTOFF_MODE", "shadow") or "shadow").strip().lower()
    if raw in {"off", "shadow", "enforce"}:
        return raw
    return "shadow"


def entry_cutoff_blocks(
    symbol: str,
    now: Optional[datetime] = None,
    minutes: Optional[float] = None,
) -> bool:
    """True when an entry would sit inside the pre-close / daily-break window.

    Binance crypto never blocks. IC crypto CFDs respect the weekday daily
    break (and the pre-break cutoff) but stay open on the weekend.
    Already-closed weekly FX sessions also block (the existing
    ``is_venue_open`` path is the primary weekend skip).
    """
    if is_crypto_symbol(symbol) and not _is_ic_crypto_cfd(symbol):
        return False
    current = _as_utc(now)
    if is_crypto_symbol(symbol) and _is_ic_crypto_cfd(symbol):
        if is_daily_break(current):
            return True
        remaining = time_until_venue_close(symbol, current)
        if remaining is None:
            return False
        cutoff = entry_cutoff_minutes() if minutes is None else max(0.0, float(minutes))
        return remaining.total_seconds() <= cutoff * 60.0
    if not is_venue_open(symbol, current):
        return True
    if is_daily_break(current):
        return True
    remaining = time_until_venue_close(symbol, current)
    if remaining is None:
        return False
    cutoff = entry_cutoff_minutes() if minutes is None else max(0.0, float(minutes))
    return remaining.total_seconds() <= cutoff * 60.0


def evaluate_entry_cutoff(
    symbol: str,
    now: Optional[datetime] = None,
    minutes: Optional[float] = None,
    mode: Optional[str] = None,
) -> dict:
    """Shadow/enforce wrapper used by the candidate filter."""
    resolved = (mode or entry_cutoff_mode()).strip().lower()
    if resolved not in {"off", "shadow", "enforce"}:
        resolved = "shadow"
    would_block = entry_cutoff_blocks(symbol, now, minutes)
    remaining = time_until_venue_close(symbol, now)
    result = {
        "mode": resolved,
        "would_block": would_block,
        "blocked": False,
        "daily_break": is_daily_break(now) if (not is_crypto_symbol(symbol) or _is_ic_crypto_cfd(symbol)) else False,
        "minutes_until_close": None if remaining is None else remaining.total_seconds() / 60.0,
        "reason": "cutoff_off" if resolved == "off" else "cutoff_ok",
    }
    if resolved == "off":
        return result
    if not would_block:
        return result
    reason = "daily_break" if result["daily_break"] else "entry_cutoff_before_close"
    result["reason"] = reason if resolved == "enforce" else f"shadow: would block ({reason})"
    result["blocked"] = resolved == "enforce"
    return result


LOOKBACK_CAP_MS = 7 * 86400 * 1000


def is_closed_for_lookback(symbol: str, now: datetime | None = None) -> bool:
    """True when FX/metals print no bars (weekend close or weekday daily break).

    Crypto is 24/7. Daily-break is included here so trendbar windows cover
    the requested open-market bar count; ``is_venue_open`` stays weekend-only
    so entry gating is unchanged.
    """
    if is_crypto_symbol(symbol):
        return False
    current = _as_utc(now)
    if not is_venue_open(symbol, current):
        return True
    return is_daily_break(current)


def _overlap_ms(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> int:
    lo = max(a0, b0)
    hi = min(a1, b1)
    if hi <= lo:
        return 0
    return int((hi - lo).total_seconds() * 1000)


def _weekend_closed_ms(start: datetime, end: datetime) -> int:
    """Milliseconds of ``[start, end)`` inside Fri 21:00–Sun 21:00 UTC."""
    start = _as_utc(start)
    end = _as_utc(end)
    if end <= start:
        return 0
    days_since_friday = (start.weekday() - SESSION_CLOSE_WEEKDAY) % 7
    friday = (start - timedelta(days=days_since_friday)).replace(
        hour=SESSION_CLOSE_HOUR, minute=0, second=0, microsecond=0
    )
    if friday > start:
        friday -= timedelta(days=7)
    weekend_span_days = (SESSION_OPEN_WEEKDAY - SESSION_CLOSE_WEEKDAY) % 7
    closed = 0
    cursor = friday
    while cursor < end:
        sun_open = (cursor + timedelta(days=weekend_span_days)).replace(
            hour=SESSION_OPEN_HOUR, minute=0, second=0, microsecond=0
        )
        closed += _overlap_ms(start, end, cursor, sun_open)
        cursor += timedelta(days=7)
    return closed


def closed_ms_in_range(
    start: datetime,
    end: datetime,
    symbol: str = "EURUSD",
) -> int:
    """Milliseconds of ``[start, end)`` with no FX/metals prints."""
    if is_crypto_symbol(symbol):
        return 0
    start = _as_utc(start)
    end = _as_utc(end)
    if end <= start:
        return 0

    closed = _weekend_closed_ms(start, end)

    day = start.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    last = end.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    while day <= last:
        start_h, start_m, end_h, end_m = daily_break_window()
        br0 = day.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        br1 = day.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
        if br1 <= br0:
            br1 += timedelta(days=1)
        if is_daily_break(br0):
            br_ms = _overlap_ms(start, end, br0, br1)
            # Custom break windows that spill into Friday 21:00+ are already
            # counted as weekend — do not double-count that intersection.
            already = _weekend_closed_ms(max(start, br0), min(end, br1))
            closed += max(0, br_ms - already)
        day += timedelta(days=1)
    return closed


def market_hours_lookback_ms(
    needed_open_ms: int,
    now: datetime | None = None,
    symbol: str = "EURUSD",
    cap_ms: int = LOOKBACK_CAP_MS,
) -> int:
    """Wall-clock ms covering ``needed_open_ms`` of venue-open time.

    When the naive window overlaps the weekend close (Fri 21:00–Sun 21:00 UTC)
    or the weekday daily break, the start is pushed back by that closed
    duration so a 40×M5 request still yields ~40 tradable bars.
    """
    needed = max(0, int(needed_open_ms))
    cap = max(0, int(cap_ms))
    if needed == 0:
        return 0
    if needed >= cap:
        return cap
    current = _as_utc(now)
    if is_crypto_symbol(symbol):
        return min(needed, cap)

    def open_ms_in(wall_ms: int) -> int:
        start = current - timedelta(milliseconds=wall_ms)
        return int(wall_ms) - closed_ms_in_range(start, current, symbol)

    if open_ms_in(needed) >= needed:
        return needed
    lo, hi = needed, cap
    best = cap
    while lo <= hi:
        mid = (lo + hi) // 2
        if open_ms_in(mid) >= needed:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return int(best)
