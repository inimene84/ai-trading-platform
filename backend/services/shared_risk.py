"""Cross-venue open-risk budget (Binance + cTrader).

Existing per-venue caps (``MAX_POSITIONS``, ``CTRADER_MAX_OPEN_POSITIONS``,
``MAX_BINANCE_POSITIONS``, …) stay in ``risk_guard`` and are unchanged.
This module adds an *additional* combined book when both venues share
the same money mode (both live cash, or both paper/demo).

``SHARED_RISK_MODE``:
    off      — no-op
    shadow   — compute + log; never raises (default)
    enforce  — raise ``RiskBreach`` on combined breaches

Split books (cTrader demo + Binance live) are never silently unioned:
both books are reported with a warning and combined limits are skipped.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from backend.services.equity_scope import (
    binance_money_mode,
    compose_balance_payload,
    ctrader_money_mode,
    is_split_book,
)
from backend.services.multi_asset_bars import classify_symbol

logger = logging.getLogger(__name__)


def _is_fx_trade(trade: Any) -> bool:
    broker = str(getattr(trade, "broker", "") or getattr(trade, "exchange", "") or "").lower()
    if broker in {"ctrader", "ctrader:paper", "icmarkets", "ic"}:
        return True
    symbol = str(getattr(trade, "symbol", "") or "").upper().replace("/", "").replace("_", "")
    return len(symbol) == 6 and symbol.isalpha()


def _fx_notional_usdt(symbol: str, lots: float, price: float) -> float:
    sym = str(symbol or "").upper().replace("/", "").replace("_", "")
    lots = abs(float(lots or 0.0))
    px = abs(float(price or 0.0))
    if lots <= 0:
        return 0.0
    units = lots * 100_000.0
    if len(sym) != 6:
        return units * px if px > 0 else units
    base, quote = sym[:3], sym[3:]
    if base == "USD":
        return units
    if quote == "USD":
        return units * (px if px > 0 else 1.0)
    return units * px if px > 0 and px < 20 else units


def _trade_exposure_usdt(trade: Any) -> float:
    price = getattr(trade, "entry_price", None) or getattr(trade, "price", None)
    qty = getattr(trade, "quantity", None)
    try:
        px = float(price or 0.0)
        q = float(qty or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if _is_fx_trade(trade):
        notional = _fx_notional_usdt(str(getattr(trade, "symbol", "") or ""), q, px)
        lev = float(os.getenv("RISK_FX_LEVERAGE", os.getenv("CTRADER_LEVERAGE", "30")) or 30)
        return notional / max(lev, 1.0)
    return abs(px * q)


def _directional_exposure_usdt(open_trades: list[Any]) -> float:
    return sum(_trade_exposure_usdt(t) for t in open_trades)


def shared_risk_mode() -> str:
    raw = (os.getenv("SHARED_RISK_MODE", "shadow") or "shadow").strip().lower()
    if raw in {"off", "shadow", "enforce"}:
        return raw
    return "shadow"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def books_combinable() -> bool:
    """True only when both venues are in the same money mode."""
    if is_split_book():
        return False
    bn = binance_money_mode()
    ct = ctrader_money_mode()
    if bn == ct:
        return True
    paperish = {"paper", "demo", "testnet"}
    liveish = {"live_cash"}
    return (bn in paperish and ct in paperish) or (bn in liveish and ct in liveish)


def _broker_of(trade: Any) -> str:
    raw = str(getattr(trade, "broker", "") or getattr(trade, "exchange", "") or "").lower()
    if "binance" in raw:
        return "binance_futures"
    if raw in {"ctrader", "ctrader:paper", "ic", "icmarkets"}:
        return "ctrader"
    symbol = str(getattr(trade, "symbol", "") or "")
    asset = classify_symbol(symbol)
    if asset == "crypto":
        return "binance_futures"
    return "ctrader"


def _partition_trades(open_trades: list[Any]) -> dict[str, list[Any]]:
    books: dict[str, list[Any]] = {"binance_futures": [], "ctrader": []}
    for trade in open_trades or []:
        books.setdefault(_broker_of(trade), []).append(trade)
    return books


def _distinct_symbols(trades: list[Any]) -> set[str]:
    return {s for t in trades if (s := getattr(t, "symbol", None))}


def _open_risk_usd(trade: Any) -> float:
    """USD at risk to the stop (not full notional)."""
    entry = getattr(trade, "entry_price", None) or getattr(trade, "price", None)
    stop = getattr(trade, "stop_loss", None)
    qty = getattr(trade, "quantity", None)
    try:
        entry_f = float(entry or 0.0)
        stop_f = float(stop or 0.0)
        qty_f = abs(float(qty or 0.0))
    except (TypeError, ValueError):
        return 0.0
    if entry_f <= 0 or qty_f <= 0:
        return 0.0
    if stop_f <= 0:
        return _trade_exposure_usdt(trade) * 0.01
    distance = abs(entry_f - stop_f)
    if _is_fx_trade(trade):
        pip = 0.01 if str(getattr(trade, "symbol", "") or "").upper().endswith("JPY") else 0.0001
        stop_pips = distance / pip
        notional = _fx_notional_usdt(str(getattr(trade, "symbol", "") or ""), qty_f, entry_f)
        return abs(notional * (stop_pips * pip / entry_f)) if entry_f else 0.0
    return distance * qty_f


def net_usd_direction(trades: list[Any]) -> float:
    """Signed USD-direction exposure: +USD (long-dollar) / −USD (short-dollar).

    Crypto BUY (long coin / short USD) is negative. Crypto SELL is positive.
    XXXUSD BUY is short USD; USDXXX BUY is long USD.
    """
    net = 0.0
    for trade in trades:
        exposure = _trade_exposure_usdt(trade)
        side = str(getattr(trade, "direction", "") or getattr(trade, "side", "") or "").upper()
        sign = 1.0 if side in {"BUY", "LONG"} else -1.0 if side in {"SELL", "SHORT"} else 0.0
        symbol = str(getattr(trade, "symbol", "") or "").upper().replace("/", "").replace("_", "")
        asset = classify_symbol(symbol)
        if asset == "crypto":
            # Long coin = short USD.
            net += -sign * exposure
            continue
        if len(symbol) >= 6:
            base, quote = symbol[:3], symbol[3:6]
            if quote == "USD":
                net += -sign * exposure
            elif base == "USD":
                net += sign * exposure
            else:
                net += -sign * exposure
            continue
        net += -sign * exposure
    return net


def evaluate_shared_risk(
    open_trades: list[Any],
    *,
    books: Optional[list[dict[str, Any]]] = None,
    combined_equity: Optional[float] = None,
    combined_peak: Optional[float] = None,
) -> dict[str, Any]:
    """Compute combined-book risk. Never raises — caller decides enforce."""
    mode = shared_risk_mode()
    partitioned = _partition_trades(open_trades)
    binance_trades = partitioned.get("binance_futures") or []
    ctrader_trades = partitioned.get("ctrader") or []
    report = {
        "binance_futures": {
            "positions": len(_distinct_symbols(binance_trades)),
            "open_risk_usd": sum(_open_risk_usd(t) for t in binance_trades),
            "exposure_usd": _directional_exposure_usdt(binance_trades),
            "net_usd": net_usd_direction(binance_trades),
            "money_mode": binance_money_mode(),
        },
        "ctrader": {
            "positions": len(_distinct_symbols(ctrader_trades)),
            "open_risk_usd": sum(_open_risk_usd(t) for t in ctrader_trades),
            "exposure_usd": _directional_exposure_usdt(ctrader_trades),
            "net_usd": net_usd_direction(ctrader_trades),
            "money_mode": ctrader_money_mode(),
        },
    }
    result: dict[str, Any] = {
        "mode": mode,
        "combinable": books_combinable(),
        "split_book": is_split_book(),
        "books": report,
        "combined": None,
        "breaches": [],
        "would_breaches": [],
        "warning": None,
        "action": "allow",
    }
    if mode == "off":
        result["warning"] = "shared_risk_off"
        return result

    if not books_combinable():
        result["warning"] = (
            "SHARED_RISK: books are different money modes "
            f"(binance={binance_money_mode()}, ctrader={ctrader_money_mode()}). "
            "Reporting both books; not combining. Existing per-venue caps still apply."
        )
        logger.warning(result["warning"])
        return result

    all_trades = list(open_trades or [])
    equity = combined_equity
    if equity is None and books:
        payload = compose_balance_payload(books)
        equity = float(payload.get("equity") or 0.0) or None
        if payload.get("error"):
            result["warning"] = f"shared_risk_equity_unverified: {payload.get('error')}"
            return result

    max_positions = _env_int("SHARED_RISK_MAX_POSITIONS", 4)
    max_open_risk_pct = _env_float("SHARED_RISK_MAX_OPEN_RISK_PCT", 6.0)
    max_drawdown_pct = _env_float("SHARED_RISK_MAX_DRAWDOWN_PCT", 15.0)
    max_net_usd = _env_float("SHARED_RISK_MAX_NET_USD", 0.0)

    symbols = _distinct_symbols(all_trades)
    open_risk = sum(_open_risk_usd(t) for t in all_trades)
    net_usd = net_usd_direction(all_trades)
    drawdown_pct = None
    if combined_peak and combined_peak > 0 and equity is not None:
        if equity < combined_peak:
            drawdown_pct = ((combined_peak - equity) / combined_peak) * 100.0

    combined = {
        "positions": len(symbols),
        "symbols": sorted(symbols),
        "open_risk_usd": open_risk,
        "net_usd": net_usd,
        "equity": equity,
        "peak": combined_peak,
        "drawdown_pct": drawdown_pct,
        "max_positions": max_positions,
        "max_open_risk_pct": max_open_risk_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "max_net_usd": max_net_usd,
    }
    result["combined"] = combined

    would: list[str] = []
    if max_positions > 0 and len(symbols) > max_positions:
        would.append(
            f"shared max positions exceeded: {len(symbols)} > {max_positions} ({sorted(symbols)})"
        )
    if equity and equity > 0 and max_open_risk_pct > 0:
        risk_pct = (open_risk / equity) * 100.0
        combined["open_risk_pct"] = risk_pct
        if risk_pct > max_open_risk_pct:
            would.append(
                f"shared open-risk exceeded: {risk_pct:.2f}% > {max_open_risk_pct}% "
                f"(${open_risk:.2f} of ${equity:.2f})"
            )
    if drawdown_pct is not None and max_drawdown_pct < 100 and drawdown_pct > max_drawdown_pct:
        would.append(
            f"shared drawdown exceeded: {drawdown_pct:.2f}% > {max_drawdown_pct}%"
        )
    if max_net_usd > 0 and abs(net_usd) > max_net_usd:
        would.append(
            f"shared net USD exposure exceeded: {net_usd:.2f} (cap ±{max_net_usd:.2f})"
        )

    result["would_breaches"] = would
    if would:
        if mode == "enforce":
            result["breaches"] = would
            result["action"] = "block"
        else:
            result["action"] = "allow"
            logger.warning("[SHARED RISK][shadow] would block: %s", "; ".join(would))
    return result
