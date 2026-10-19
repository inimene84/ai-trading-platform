"""Fetch OHLCV bars for crypto, forex, metals, oil, and equities."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Literal

logger = logging.getLogger(__name__)

AssetClass = Literal["crypto", "forex", "metal", "oil", "stock", "index"]

_CRYPTO_SUFFIXES = ("USDT", "USDC", "BUSD", "PERP")
_FOREX_MAJORS = {
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURCHF", "EURAUD",
}
_METALS = {"XAUUSD", "XAGUSD", "GOLD", "SILVER", "XPTUSD", "XPDUSD"}
_OIL = {"USOIL", "UKOIL", "WTI", "BRENT", "CL=F", "BZ=F"}
# IC Markets crypto CFDs are 6-letter XXXUSD names. Without this set they
# collapse to the forex heuristic (BTCUSD → "forex") and inherit FX pips.
_IC_CRYPTO_CFD = {
    "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "BNBUSD",
    "LNKUSD", "AVXUSD", "DOGUSD",
    "LINKUSD", "AVAXUSD", "DOGEUSD",
}


def classify_symbol(symbol: str) -> AssetClass:
    sym = symbol.upper().replace("/", "").replace("-", "").strip().rstrip(".")
    if sym.endswith("=X"):
        sym = sym[:-2]
    if sym in _METALS or sym.startswith("XAU") or sym.startswith("XAG"):
        return "metal"
    if sym in _OIL or (sym.endswith("=F") and sym.startswith(("CL", "BZ"))):
        return "oil"
    # Crypto suffixes must win before the 6-letter forex heuristic.
    # OPUSDT / ARBUSDT / APTUSDT are 6 alpha chars and were sent to
    # yfinance as OPUSDT=X (404 spam + wrong asset class).
    if any(sym.endswith(sfx) for sfx in _CRYPTO_SUFFIXES):
        return "crypto"
    if sym in _IC_CRYPTO_CFD:
        return "crypto"
    if sym.endswith("USD") and len(sym) > 6:
        return "crypto"
    if sym in _FOREX_MAJORS or (len(sym) == 6 and sym.isalpha()):
        return "forex"
    if sym in {"SPX", "SPY", "QQQ", "DIA", "NDX", "VIX"}:
        return "index"
    return "stock"


def _tf_to_ctrader_period(timeframe: str) -> str:
    mapping = {
        "1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
        "1h": "H1", "4h": "H4", "1d": "D1", "1w": "W1",
        "M1": "M1", "M5": "M5", "M15": "M15", "M30": "M30",
        "H1": "H1", "H4": "H4", "D1": "D1", "W1": "W1",
    }
    return mapping.get(timeframe, "H1")


def tf_to_binance_interval(timeframe: str) -> str:
    """Map cTrader-style (M5) or Binance-style (5m) timeframes to Binance kline interval."""
    mapping = {
        "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
        "H1": "1h", "H4": "4h", "D1": "1d", "W1": "1w",
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w",
    }
    key = timeframe if timeframe in mapping else timeframe.upper()
    return mapping.get(key, "5m")


def _normalize_bars(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for b in raw or []:
        out.append({
            "time": b.get("time") or b.get("timestamp"),
            "open": float(b.get("open", 0) or 0),
            "high": float(b.get("high", 0) or 0),
            "low": float(b.get("low", 0) or 0),
            "close": float(b.get("close", 0) or 0),
            "volume": float(b.get("volume", 0) or 0),
        })
    return out


async def fetch_bars(symbol: str, timeframe: str = "1h", limit: int = 100) -> Dict[str, Any]:
    """Return normalized OHLCV bars plus asset metadata."""
    sym = symbol.upper().replace("/", "")
    asset_class = classify_symbol(sym)
    period = _tf_to_ctrader_period(timeframe)

    if asset_class == "crypto":
        from backend.services.binance_market_data import binance_market_data
        # Callers pass cTrader-style timeframes (M5, H1); Binance rejects those
        # outright and the empty result used to look like a flat market.
        bars = await binance_market_data.get_klines(
            symbol=sym, interval=tf_to_binance_interval(timeframe), limit=limit
        )
        return {"symbol": sym, "asset_class": asset_class, "source": "binance", "data": _normalize_bars(bars)}

    if asset_class in ("forex", "metal"):
        from backend.services.ctrader_service import ctrader_service
        raw = await asyncio.to_thread(
            ctrader_service.get_trendbars,
            symbol=sym.replace("GOLD", "XAUUSD").replace("SILVER", "XAGUSD"),
            period=period,
            count=limit,
        )
        return {"symbol": sym, "asset_class": asset_class, "source": "ctrader", "data": _normalize_bars(raw)}

    # stocks / oil / index via yfinance
    import yfinance as yf

    yf_map = {
        "USOIL": "CL=F", "UKOIL": "BZ=F", "WTI": "CL=F", "BRENT": "BZ=F",
        "GOLD": "GC=F", "SILVER": "SI=F", "SPX": "^GSPC", "NDX": "^NDX", "VIX": "^VIX",
    }
    yf_sym = yf_map.get(sym, sym)
    interval = "1h" if timeframe in ("1h", "H1") else "1d" if timeframe in ("1d", "D1", "1w", "W1") else "15m"

    def _yf_fetch():
        ticker = yf.Ticker(yf_sym)
        hist = ticker.history(period="60d", interval=interval)
        if hist is None or hist.empty:
            return []
        rows = hist.tail(limit)
        bars = []
        for idx, row in rows.iterrows():
            bars.append({
                "time": int(idx.timestamp()),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume", 0) or 0),
            })
        return bars

    raw = await asyncio.to_thread(_yf_fetch)
    source = "yfinance"
    if asset_class == "oil":
        source = "yfinance-oil"
    elif asset_class == "stock":
        source = "yfinance-equity"
    return {"symbol": sym, "asset_class": asset_class, "source": source, "data": _normalize_bars(raw)}
