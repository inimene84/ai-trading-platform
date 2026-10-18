"""Flag-gated Binance maker TP / close / re-quote helpers.

All live order-placement changes stay behind env flags that default OFF
so merging this module does not change the book:

* ``BINANCE_MAKER_TP_ENABLED`` (default false) — reduce-only post-only
  (GTX) limit at the take-profit instead of ``TAKE_PROFIT_MARKET``.
  Stops stay ``STOP_MARKET``. Market flatten only when last/mark has
  already traded through the rest. Age / ``BINANCE_MAKER_TP_TIMEOUT_SEC``
  never flattens — a TP is meant to rest for hours.
* ``BINANCE_MAKER_CLOSE_ENABLED`` (default false) — maker-first GTX for
  discretionary / time-based reduce-only closes.
* ``BINANCE_MAKER_REQUOTE_ENABLED`` (default false) — cancel and re-join
  the touch when the book walks away, instead of a single GTX then MARKET.
"""

from __future__ import annotations

import os
from typing import Any, Optional


def _truthy(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default) or default).strip().lower() in {"1", "true", "yes", "on"}


def maker_tp_enabled() -> bool:
    return _truthy("BINANCE_MAKER_TP_ENABLED", "false")


def maker_close_enabled() -> bool:
    return _truthy("BINANCE_MAKER_CLOSE_ENABLED", "false")


def maker_requote_enabled() -> bool:
    return _truthy("BINANCE_MAKER_REQUOTE_ENABLED", "false")


def maker_tp_timeout_sec() -> float:
    try:
        return max(1.0, float(os.getenv("BINANCE_MAKER_TP_TIMEOUT_SEC", "120") or 120))
    except (TypeError, ValueError):
        return 120.0


def maker_requote_max() -> int:
    try:
        return max(1, int(os.getenv("BINANCE_MAKER_REQUOTE_MAX", "4") or 4))
    except (TypeError, ValueError):
        return 4


def maker_wait_sec() -> float:
    try:
        return max(0.0, float(os.getenv("MAKER_WAIT_SEC", "8") or 8))
    except (TypeError, ValueError):
        return 8.0


def gtx_rejected_as_taker(err: Exception | str) -> bool:
    text = str(err)
    return any(code in text for code in ("-2021", "-5022")) or "post only" in text.lower() or "would immediately" in text.lower()


def price_traded_through_tp(*, side: str, tp_price: float, last_price: float) -> bool:
    """True when last has crossed the reduce-only TP rest (fill-or-fallback).

    Long TP is a SELL rest above the market — last >= tp means it traded through.
    Short TP is a BUY rest below the market — last <= tp means it traded through.
    """
    if tp_price <= 0 or last_price <= 0:
        return False
    close_side = (side or "").upper()
    if close_side == "SELL":  # flattening a long
        return last_price >= float(tp_price)
    if close_side == "BUY":  # flattening a short
        return last_price <= float(tp_price)
    return False


def book_walked_away(*, side: str, rest_price: float, best_bid: float, best_ask: float, tick: float) -> bool:
    """True when the touch moved at least one tick away from our rest."""
    tick = tick if tick > 0 else 0.0
    if (side or "").upper() == "BUY":
        return best_bid > 0 and (best_bid - rest_price) >= max(tick, 0.0) - 1e-12 and best_bid > rest_price + 1e-12
    if (side or "").upper() == "SELL":
        return best_ask > 0 and (rest_price - best_ask) >= max(tick, 0.0) - 1e-12 and rest_price > best_ask + 1e-12
    return False


def maker_tp_params(
    *,
    futures_sym: str,
    close_side: str,
    position_side: str,
    quantity: float,
    tp_price: float,
    client_order_id: Optional[str] = None,
) -> dict[str, Any]:
    """Reduce-only post-only LIMIT at the take-profit (hedge-mode safe)."""
    params: dict[str, Any] = {
        "symbol": futures_sym,
        "side": close_side,
        "type": "LIMIT",
        "timeInForce": "GTX",
        "price": tp_price,
        "quantity": quantity,
        "positionSide": position_side,
    }
    if client_order_id:
        params["newClientOrderId"] = client_order_id
    return params


def take_profit_market_params(
    *,
    futures_sym: str,
    close_side: str,
    position_side: str,
    tp_price: float,
) -> dict[str, Any]:
    return {
        "symbol": futures_sym,
        "side": close_side,
        "type": "TAKE_PROFIT_MARKET",
        "stopPrice": tp_price,
        "closePosition": "true",
        "positionSide": position_side,
    }


def maker_tp_market_flatten_params(
    *,
    futures_sym: str,
    close_side: str,
    position_side: str,
    quantity: float,
) -> dict[str, Any]:
    """Immediate reduce-only MARKET flatten (hedge-mode: positionSide, no reduceOnly).

    Used when a GTX TP would take or last/mark has already traded through.
    Do not re-place TAKE_PROFIT_MARKET at a stop that has already printed —
    Binance rejects that with -2021.
    """
    return {
        "symbol": futures_sym,
        "side": close_side,
        "type": "MARKET",
        "quantity": quantity,
        "positionSide": position_side,
    }


def is_maker_tp_order(order: dict[str, Any]) -> bool:
    """Identify a resting maker TP (GTX reduce-only LIMIT, not an entry)."""
    otype = (order.get("type") or "").upper()
    tif = (order.get("timeInForce") or order.get("time_in_force") or "").upper()
    cid = str(order.get("clientOrderId") or order.get("newClientOrderId") or "")
    if cid.startswith("qtpMTP"):
        return True
    reduce_only = str(order.get("reduceOnly") or order.get("reduce_only") or "").lower() in {
        "true", "1", "yes",
    }
    return otype == "LIMIT" and tif == "GTX" and reduce_only
