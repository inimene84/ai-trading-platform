"""Past-only market state built from candles already in hand.

Indicators use only bars at or before the last row. The backtest harness
slices history before calling this so a forecast cannot see the next close.
"""

from __future__ import annotations

from typing import Any

from backend.jev.config import MIN_BARS


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
    return {
        "open": round(open_, 8),
        "high": round(high, 8),
        "low": round(low, 8),
        "close": round(close, 8),
        "volume": round(volume or 0.0, 4),
    }


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
        },
    }
