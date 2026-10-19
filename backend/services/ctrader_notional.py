"""Per-symbol USD notional for cTrader (IC Markets) positions.

Trade.quantity on cTrader rows is in *lots*. Spotware OpenAPI volume is in
cents of a unit and ``ProtoOASymbol.lotSize`` (also in cents) differs per
symbol, so "units per lot" is NOT always 100,000:

* FX:              lotSize 10,000,000 cents  -> 100,000 units of base
* XAUUSD:          100 oz per lot
* XAGUSD:          1,000 oz per lot
* IC crypto CFDs:  1 coin per lot (ETHUSD 0.1 lot = 0.1 ETH, volume=10)

The risk guard used to apply the FX 100,000 units/lot to every cTrader row,
so 0.1 lot ETHUSD read as 10,000 ETH (~$24.2M notional, $805k after /30
leverage) and stopped the loop on a ~$240 position.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from backend.services.multi_asset_bars import classify_symbol

logger = logging.getLogger(__name__)

FX_UNITS_PER_LOT = 100_000.0

# Static fallback when the live symbol catalog is not loaded (tests, cold start).
# Mirrors CTraderService.UNITS_PER_LOT; live ProtoOASymbol.lotSize wins.
_FALLBACK_UNITS_PER_LOT: dict[str, float] = {
    "XAUUSD": 100.0,
    "XAGUSD": 1_000.0,
    "GOLD": 100.0,
    "SILVER": 1_000.0,
    "BTCUSD": 1.0,
    "ETHUSD": 1.0,
    "SOLUSD": 1.0,
    "XRPUSD": 1.0,
    "ADAUSD": 1.0,
    "BNBUSD": 1.0,
    "LNKUSD": 1.0,
    "AVXUSD": 1.0,
    "DOGUSD": 1.0,
    "LINKUSD": 1.0,
    "AVAXUSD": 1.0,
    "DOGEUSD": 1.0,
}

CTRADER_BROKERS = {"ctrader", "ctrader:paper", "icmarkets", "ic"}


def _norm(symbol: str) -> str:
    return (
        str(symbol or "")
        .upper()
        .replace("/", "")
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .rstrip(".")
    )


def _ctrader_service_cls():
    try:
        from backend.services.ctrader_service import CTraderService

        return CTraderService
    except Exception:  # pragma: no cover - import guard only
        return None


def _ctrader_service_inst():
    try:
        from backend.services.ctrader_service import ctrader_service

        return ctrader_service
    except Exception:  # pragma: no cover
        return None


def ctrader_units_per_lot(symbol: str) -> float:
    """Base-asset units in 1.00 lot. Live lotSize (cents) > static table > FX."""
    sym = _norm(symbol)
    cls = _ctrader_service_cls()
    if cls is not None:
        try:
            spec = cls._live_spec(sym) or {}
            lot_cents = int(spec.get("lot_size") or 0)
            if lot_cents > 0:
                return float(lot_cents) / 100.0
            name = cls.normalize_symbol_name(sym)
            units = cls.UNITS_PER_LOT.get(name)
            if units:
                return float(units)
        except Exception:
            pass
    if sym in _FALLBACK_UNITS_PER_LOT:
        return _FALLBACK_UNITS_PER_LOT[sym]
    asset = classify_symbol(sym) if sym else "forex"
    if asset == "crypto":
        return 1.0
    if asset == "metal":
        return 100.0 if sym.startswith("XAU") else 1_000.0 if sym.startswith("XAG") else 100.0
    return FX_UNITS_PER_LOT


def _quote_to_usd(sym: str, price: float) -> Optional[float]:
    """Rate converting the quote currency to USD (None when unknown)."""
    if len(sym) != 6:
        return 1.0 if sym.endswith("USD") else None
    base, quote = sym[:3], sym[3:]
    if quote == "USD":
        return 1.0
    if base == "USD":
        return 1.0 / price if price > 0 else None
    inst = _ctrader_service_inst()
    if inst is not None:
        try:
            rate = inst.quote_to_usd_rate(sym, price)
            if rate and rate > 0:
                return float(rate)
        except Exception:
            pass
    return None


def ctrader_notional_usd(symbol: str, lots: float, price: float) -> float:
    """Full (unleveraged) USD notional of a cTrader position."""
    sym = _norm(symbol)
    lots = abs(float(lots or 0.0))
    px = abs(float(price or 0.0))
    if lots <= 0:
        return 0.0
    units = lots * ctrader_units_per_lot(sym)
    if len(sym) == 6 and sym.isalpha() and classify_symbol(sym) == "forex":
        base, quote = sym[:3], sym[3:]
        if base == "USD":
            return units  # USDXXX: base notional is already USD
        if quote == "USD":
            return units * (px if px > 0 else 1.0)
        rate = _quote_to_usd(sym, px)
        if quote == "JPY" and rate is None:
            rate = 1.0 / float(os.getenv("RISK_USDJPY_FALLBACK", "150") or 150)
        if rate is not None and px > 0:
            return units * px * rate
        # Legacy rough fallback for crosses without a live quote->USD mark.
        if 0 < px < 20:
            return units * px
        return units
    # Metals / crypto / other CFDs: units * price in the quote ccy (USD on IC).
    notional = units * px if px > 0 else units
    rate = _quote_to_usd(sym, px) if not sym.endswith("USD") else 1.0
    return notional * (rate if rate else 1.0)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def ctrader_risk_exposure_usd(symbol: str, lots: float, price: float) -> float:
    """Exposure the directional cap compares against equity*mult.

    FX / metals: margin-equivalent (notional / RISK_FX_LEVERAGE, default 30),
    unchanged semantics. Crypto CFDs: full notional by default, the same way
    the crypto loop counts Binance coins (|price*qty|); override with
    RISK_CTRADER_CRYPTO_LEVERAGE.
    """
    sym = _norm(symbol)
    notional = ctrader_notional_usd(sym, lots, price)
    if classify_symbol(sym) == "crypto":
        lev = _env_float("RISK_CTRADER_CRYPTO_LEVERAGE", 1.0)
    else:
        lev = _env_float("RISK_FX_LEVERAGE", _env_float("CTRADER_LEVERAGE", 30.0))
    return notional / max(lev, 1.0)


def is_ctrader_trade(trade: Any) -> bool:
    broker = str(getattr(trade, "broker", "") or getattr(trade, "exchange", "") or "").lower()
    return broker in CTRADER_BROKERS


def dedupe_open_ctrader_rows(trades: list[Any]) -> list[Any]:
    """One exposure row per cTrader broker_position_id.

    The false-close/duplicate-row bug can leave two OPEN rows for one broker
    position; counting both would double the exposure. Rows without a
    position id, and non-cTrader rows, are kept as-is.
    """
    seen: set[str] = set()
    out: list[Any] = []
    for t in trades or []:
        pid = str(getattr(t, "broker_position_id", "") or "").strip()
        if pid and is_ctrader_trade(t):
            key = pid
            if key in seen:
                logger.warning(
                    "[RISK GUARD] duplicate open cTrader row for position %s (%s) "
                    "excluded from exposure", pid, getattr(t, "symbol", "?"),
                )
                continue
            seen.add(key)
        out.append(t)
    return out
