"""Past-only market state built from candles already in hand.

Indicators use only bars at or before the last row. The backtest harness
slices history before calling this so a forecast cannot see the next close.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.jev.config import MIN_BARS, STATE_BUILDER_VERSION


def _num(bar: dict, *keys: str) -> float | None:
    for key in keys:
        if key in bar and bar[key] is not None:
            try:
                value = float(bar[key])
            except (TypeError, ValueError):
                return None
            return value
    return None


def _ohlcv(bar: dict) -> dict[str, float] | None:
    open_ = _num(bar, "open", "Open")
    high = _num(bar, "high", "High")
    low = _num(bar, "low", "Low")
    close = _num(bar, "close", "Close")
    volume = _num(bar, "volume", "Volume")
    if None in (open_, high, low, close):
        return None
    if high < low or min(open_, high, low, close) < 0:
        return None
    row = {
        "open": round(open_, 8),
        "high": round(high, 8),
        "low": round(low, 8),
        "close": round(close, 8),
        "volume": round(volume or 0.0, 4),
    }
    stamp = bar_time(bar)
    if stamp is not None:
        row["time"] = stamp
    return row


def bar_time(bar: dict) -> int | None:
    for key in ("time", "timestamp", "timestamp_epoch", "date"):
        raw = bar.get(key)
        if raw is None or raw == "":
            continue
        if isinstance(raw, (int, float)):
            value = float(raw)
            if value > 1e12:
                value = value / 1000.0
            if value > 0:
                return int(value)
        text = str(raw)
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())
    return None


def assert_no_lookahead(rows: list[dict], as_of: int) -> None:
    """Nothing strictly after as_of may remain in a past-only state."""
    for row in rows:
        stamp = row.get("time")
        if stamp is not None and int(stamp) > int(as_of):
            raise ValueError(f"lookahead bar at {stamp} is after as_of {as_of}")


def rows_as_of(rows: list[dict], as_of: int) -> list[dict]:
    kept = [row for row in rows if row.get("time") is None or int(row["time"]) <= int(as_of)]
    assert_no_lookahead(kept, as_of)
    return kept


def rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    start = len(closes) - period
    for index in range(start, len(closes)):
        delta = closes[index] - closes[index - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(rows: list[dict[str, float]], period: int = 14) -> float | None:
    if len(rows) < period + 1:
        return None
    window = rows[-(period + 1) :]
    true_ranges: list[float] = []
    for index in range(1, len(window)):
        high = window[index]["high"]
        low = window[index]["low"]
        prev_close = window[index - 1]["close"]
        true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    if not true_ranges:
        return None
    return sum(true_ranges) / len(true_ranges)


def base_asset(symbol: str) -> str:
    cleaned = (
        symbol.upper()
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
        .replace("$", "")
        .replace("=X", "")
        .strip()
    )
    for quote in ("USDT", "USDC", "BUSD", "USD"):
        if cleaned.endswith(quote) and len(cleaned) > len(quote):
            return cleaned[: -len(quote)]
    return cleaned


def futures_symbol(symbol: str) -> str:
    cleaned = (
        symbol.upper()
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
        .replace("$", "")
        .replace("=X", "")
        .strip()
    )
    if cleaned.endswith(("USDT", "USDC", "BUSD")):
        return cleaned
    if cleaned.endswith("USD") and len(cleaned) > 3:
        return cleaned[:-3] + "USDT"
    return f"{cleaned}USDT"


def build_market_state(
    symbol: str,
    bars: list[dict],
    metrics: dict | None = None,
    sessions: int = 30,
) -> dict[str, Any] | None:
    """Return a past-only state, or None when history is too short to score."""
    rows: list[dict[str, float]] = []
    for bar in bars or []:
        parsed = _ohlcv(bar)
        if parsed is not None:
            rows.append(parsed)
    if len(rows) < MIN_BARS:
        return None
    window = rows[-sessions:]
    closes = [row["close"] for row in window]
    last = closes[-1]
    sma = sum(closes) / len(closes)
    ma_distance_pct = ((last - sma) / sma * 100.0) if sma else 0.0
    volumes = [row["volume"] for row in window]
    avg_volume = sum(volumes) / len(volumes) if volumes else 0.0
    volume_ratio = (volumes[-1] / avg_volume) if avg_volume else 1.0
    atr_value = atr(window)
    rsi_value = rsi(closes)
    change_pct = ((closes[-1] - closes[0]) / closes[0] * 100.0) if closes[0] else 0.0
    if change_pct > 1.5:
        momentum = "up"
    elif change_pct < -1.5:
        momentum = "down"
    else:
        momentum = "flat"

    metrics = metrics or {}
    funding = metrics.get("funding_rate", metrics.get("fundingRate"))
    try:
        funding_rate = float(funding) if funding is not None else None
    except (TypeError, ValueError):
        funding_rate = None
    open_interest = metrics.get("open_interest", metrics.get("openInterest"))
    try:
        open_interest_usd = float(open_interest) if open_interest is not None else None
    except (TypeError, ValueError):
        open_interest_usd = None

    return {
        "asset": base_asset(symbol),
        "symbol": futures_symbol(symbol),
        "market": {
            "sessions": len(window),
            "last_close": round(last, 8),
            "change_window_pct": round(change_pct, 4),
            "momentum_bucket": momentum,
            "rsi_14": None if rsi_value is None else round(rsi_value, 2),
            "atr_14": None if atr_value is None else round(atr_value, 8),
            "ma_distance_pct": round(ma_distance_pct, 4),
            "volume_ratio": round(volume_ratio, 4),
            "funding_rate": funding_rate,
            "open_interest_usd": open_interest_usd,
            "ohlcv": window,
            "state_builder_version": STATE_BUILDER_VERSION,
            **calendar_context(window),
        },
    }


def calendar_context(rows: list[dict[str, float]]) -> dict[str, Any]:
    """UTC month and week buckets using only bars that carry a timestamp.

    Crypto does not have an exchange close. Weeks are ISO weeks and months
    are calendar months in UTC. Bars without timestamps are omitted rather
    than assigned a guessed date.
    """
    stamped = [row for row in rows if row.get("time") is not None]
    if len(stamped) < 2:
        return {}
    as_of = max(int(row["time"]) for row in stamped)
    assert_no_lookahead(stamped, as_of)
    months: dict[str, list[float]] = {}
    weeks: dict[str, list[float]] = {}
    for row in stamped:
        moment = datetime.fromtimestamp(int(row["time"]), tz=timezone.utc)
        months.setdefault(moment.strftime("%Y-%m"), []).append(float(row["close"]))
        year, week, _day = moment.isocalendar()
        weeks.setdefault(f"{year}-W{week:02d}", []).append(float(row["close"]))
    return {
        "as_of": as_of,
        "monthly_context": _bucket_summary(months, limit=12),
        "weekly_context": _bucket_summary(weeks, limit=14),
    }


def _bucket_summary(groups: dict[str, list[float]], limit: int) -> list[dict[str, Any]]:
    keys = sorted(groups)[-limit:]
    summary = []
    for key in keys:
        closes = groups[key]
        first = closes[0]
        last = closes[-1]
        change = ((last - first) / first * 100.0) if first else 0.0
        summary.append({
            "bucket": key,
            "sessions": len(closes),
            "open": round(first, 8),
            "close": round(last, 8),
            "change_pct": round(change, 4),
            "complete": False,
        })
    if summary:
        summary[-1]["partial"] = True
        for item in summary[:-1]:
            item["complete"] = True
            item["partial"] = False
    return summary
