"""Thin helpers shared by live /predict and trainers.

The canonical five-bucket list lives in `asset_universe.py` (GPU trainer).
These helpers only cover Jesse symbol shaping and candle timeframe fallback.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from asset_universe import ALL_CLASSES, DEFAULT_UNIVERSE, candle_timeframe_candidates, normalize_symbol, resolve_universe

JESSE_BINANCE_EXCHANGE = "Binance Perpetual Futures"
YFINANCE_EXCHANGE = "Yahoo Finance"


def jesse_symbol_from_compact(symbol: str) -> str:
    return normalize_symbol(symbol)


def candle_timeframe_candidates(requested: str) -> Tuple[str, ...]:
    """Prefer 1m (resample) then native requested timeframe."""
    if requested == "1m":
        return ("1m",)
    if requested == "1h":
        return ("1m", "1h")
    return (requested, "1m")


def all_asset_classes() -> Sequence[str]:
    return ALL_CLASSES


def universe_by_class() -> Dict[str, List[str]]:
    return {cls: [item.symbol for item in items] for cls, items in DEFAULT_UNIVERSE.items()}


CRYPTO_IMPORT = [
    {"symbol": item.symbol, "exchange": JESSE_BINANCE_EXCHANGE, "asset_class": "crypto"}
    for item in resolve_universe("crypto")
]
YFINANCE_IMPORT = [
    {
        "symbol": item.symbol,
        "yahoo": item.ticker,
        "asset_class": item.asset_class,
        "exchange": YFINANCE_EXCHANGE,
    }
    for item in resolve_universe("all")
    if item.source == "yfinance"
]
