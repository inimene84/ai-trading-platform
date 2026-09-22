"""Dry-run quote plans inspired by maker entries and IOC exits.

The function returns a plan and never submits it. Live order routers must
not call this module to place size. submit is always false.
"""

from __future__ import annotations

from typing import Any


def plan_quote(
    *,
    intent: str,
    bias: str,
    bid: float,
    ask: float,
    tick: float,
    quote_size: float = 0.0,
    position_size: float = 0.0,
) -> dict[str, Any]:
    """Plan an ALO entry, an IOC reduce-only exit, or a cancel-on-hold.

    A buy improves the bid by one tick and stays strictly below the ask.
    A sell improves the ask by one tick and stays strictly above the bid.
    """
    action = (intent or "hold").strip().lower()
    side_bias = (bias or "flat").strip().lower()
    base = {
        "intent": action,
        "bias": side_bias,
        "submit": False,
        "dry_run": True,
        "reason": "Jev quote plans are research-only and are not sent to a venue",
    }
    if tick <= 0 or ask <= bid or bid <= 0:
        return {**base, "order": None, "reason": "book is not usable; no plan"}
    if action == "close":
        if position_size == 0:
            return {**base, "order": None, "reason": "no position to close"}
        side = "sell" if position_size > 0 else "buy"
        return {
            **base,
            "order": {
                "type": "IOC",
                "side": side,
                "reduce_only": True,
                "size": abs(position_size),
                "price": ask if side == "buy" else bid,
            },
        }
    if action != "open":
        return {**base, "order": None, "cancel_resting": True}
    if side_bias == "long":
        price = min(bid + tick, ask - tick)
        side = "buy"
    elif side_bias == "short":
        price = max(ask - tick, bid + tick)
        side = "sell"
    else:
        return {**base, "order": None, "cancel_resting": True}
    if price <= bid and side == "sell":
        return {**base, "order": None, "reason": "post-only price would cross"}
    if price >= ask and side == "buy":
        return {**base, "order": None, "reason": "post-only price would cross"}
    return {
        **base,
        "order": {
            "type": "ALO",
            "side": side,
            "reduce_only": False,
            "size": max(0.0, quote_size),
            "price": price,
        },
    }


def live_submit_allowed() -> bool:
    """Hard stop. Jev plans are not authorized to reach an exchange."""
    return False
