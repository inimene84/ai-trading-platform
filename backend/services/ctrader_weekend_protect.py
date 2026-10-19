"""Friday pre-close flatten/tighten and swap-aware max hold for cTrader.

``CTRADER_WEEKEND_PROTECT_MODE``:
    off       — no-op
    shadow    — log what flatten/tighten would do (default)
    tighten   — move SL to break-even (in profit) or a hard stop (loser)
    flatten   — close positions that would gap over the weekend

``CTRADER_SWAP_HOLD_MODE`` (off|shadow|enforce, default shadow) counts
overnight swap in expected value for holds longer than one day. Triple
swap is Wednesday for FX/metals and Friday for crypto (see venue_costs).

Neither path runs unless the corresponding mode is not ``off``. Defaults
are shadow so merge does not touch the live IC book.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from backend.backtesting.venue_costs import default_cost_config, swap_cost_usd, swap_multiplier
from backend.services.market_hours import (
    SESSION_CLOSE_HOUR,
    SESSION_CLOSE_WEEKDAY,
    is_crypto_symbol,
    time_until_venue_close,
)
from backend.services.multi_asset_bars import classify_symbol

logger = logging.getLogger(__name__)


def weekend_protect_mode() -> str:
    raw = (os.getenv("CTRADER_WEEKEND_PROTECT_MODE", "shadow") or "shadow").strip().lower()
    if raw in {"off", "shadow", "tighten", "flatten"}:
        return raw
    return "shadow"


def swap_hold_mode() -> str:
    raw = (os.getenv("CTRADER_SWAP_HOLD_MODE", "shadow") or "shadow").strip().lower()
    if raw in {"off", "shadow", "enforce"}:
        return raw
    return "shadow"


def weekend_protect_lead_minutes() -> float:
    try:
        return max(0.0, float(os.getenv("CTRADER_WEEKEND_PROTECT_MINUTES", "60") or 60))
    except (TypeError, ValueError):
        return 60.0


def is_friday_pre_close(now: Optional[datetime] = None, lead_minutes: Optional[float] = None) -> bool:
    """True on Friday inside the lead window before 21:00 UTC close."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)
    if current.weekday() != SESSION_CLOSE_WEEKDAY:
        return False
    close = current.replace(hour=SESSION_CLOSE_HOUR, minute=0, second=0, microsecond=0)
    lead = weekend_protect_lead_minutes() if lead_minutes is None else float(lead_minutes)
    start = close - timedelta(minutes=lead)
    return start <= current < close


def _position_side(pos: dict[str, Any]) -> str:
    return str(pos.get("side") or pos.get("direction") or "").upper()


def _unrealized(pos: dict[str, Any]) -> float:
    entry = float(pos.get("entry_price") or 0.0)
    current = float(pos.get("current_price") or 0.0)
    side = _position_side(pos)
    if entry <= 0 or current <= 0:
        return 0.0
    return (current - entry) if side == "BUY" else (entry - current)


def evaluate_weekend_protect(
    positions: list[dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    mode: Optional[str] = None,
) -> dict[str, Any]:
    """Decide flatten/tighten actions. Does not talk to the broker."""
    resolved = (mode or weekend_protect_mode()).strip().lower()
    if resolved not in {"off", "shadow", "tighten", "flatten"}:
        resolved = "shadow"
    result: dict[str, Any] = {
        "mode": resolved,
        "active_window": False,
        "actions": [],
        "applied": [],
    }
    if resolved == "off":
        return result
    if not is_friday_pre_close(now):
        return result
    result["active_window"] = True

    for pos in positions or []:
        symbol = str(pos.get("symbol") or "")
        if not symbol or is_crypto_symbol(symbol):
            continue
        remaining = time_until_venue_close(symbol, now)
        if remaining is None:
            continue
        ur = _unrealized(pos)
        intended = "flatten" if resolved == "flatten" else "tighten"
        # tighten: BE if in profit, else keep/affirm hard stop
        detail = "flatten_before_weekend" if intended == "flatten" else (
            "tighten_to_break_even" if ur > 0 else "affirm_hard_stop"
        )
        action = {
            "symbol": symbol,
            "side": _position_side(pos),
            "position_id": str(pos.get("position_id") or ""),
            "unrealized": ur,
            "intended": intended,
            "detail": detail,
            "minutes_until_close": remaining.total_seconds() / 60.0,
        }
        result["actions"].append(action)
    return result


def apply_weekend_protect(positions: list[dict[str, Any]], ctrader, now: Optional[datetime] = None) -> dict[str, Any]:
    """Execute or shadow-log Friday pre-close actions."""
    plan = evaluate_weekend_protect(positions, now=now)
    mode = plan["mode"]
    if mode == "off" or not plan["active_window"]:
        return plan
    for action in plan["actions"]:
        pos = next(
            (p for p in positions if str(p.get("position_id") or "") == action["position_id"]),
            None,
        )
        if pos is None:
            continue
        if mode == "shadow":
            logger.warning(
                "[WEEKEND PROTECT][shadow] would %s %s %s (%s)",
                action["intended"], action["side"], action["symbol"], action["detail"],
            )
            continue
        if mode == "flatten":
            res = ctrader.close_position(
                action["position_id"],
                symbol=action["symbol"],
                volume=pos.get("quantity"),
            )
            action["result"] = res.get("status")
            plan["applied"].append(action)
            continue
        # tighten
        entry = float(pos.get("entry_price") or 0.0)
        side = action["side"]
        if entry <= 0 or side not in {"BUY", "SELL"}:
            continue
        if action["detail"] == "tighten_to_break_even":
            # Keep the live TP: the Spotware amend replaces both legs.
            res = ctrader.amend_position_sltp(
                action["position_id"], stop_loss=entry, take_profit=pos.get("take_profit"),
            )
            action["result"] = res.get("status")
            plan["applied"].append(action)
        else:
            logger.info(
                "[WEEKEND PROTECT] %s %s already on the loss side — leaving hard stop",
                side, action["symbol"],
            )
    return plan


def _hold_hours(pos: dict[str, Any], now: datetime) -> float:
    opened = pos.get("opened_at") or pos.get("open_time") or pos.get("timestamp")
    if opened is None:
        return 0.0
    if isinstance(opened, str):
        try:
            opened = datetime.fromisoformat(opened.replace("Z", "+00:00"))
        except ValueError:
            return 0.0
    if getattr(opened, "tzinfo", None) is None:
        opened = opened.replace(tzinfo=timezone.utc)
    return max(0.0, (now - opened).total_seconds() / 3600.0)


def _rollover_nights(opened: datetime, now: datetime) -> list[datetime]:
    nights = []
    cursor = opened.astimezone(timezone.utc).replace(hour=21, minute=0, second=0, microsecond=0)
    if cursor <= opened.astimezone(timezone.utc):
        cursor = cursor + timedelta(days=1)
    end = now.astimezone(timezone.utc)
    while cursor <= end:
        nights.append(cursor)
        cursor = cursor + timedelta(days=1)
    return nights


def evaluate_swap_hold(
    pos: dict[str, Any],
    *,
    now: Optional[datetime] = None,
    expected_remaining_usd: Optional[float] = None,
) -> dict[str, Any]:
    """Compare remaining edge vs swap already due + still expected."""
    mode = swap_hold_mode()
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    symbol = str(pos.get("symbol") or "")
    hours = _hold_hours(pos, current)
    result: dict[str, Any] = {
        "mode": mode,
        "symbol": symbol,
        "hold_hours": hours,
        "swap_usd": 0.0,
        "expected_remaining_usd": expected_remaining_usd,
        "would_flatten": False,
        "action": "hold",
        "reason": "swap_hold_off" if mode == "off" else "within_intraday",
    }
    if mode == "off":
        return result
    if hours <= 24.0:
        return result

    opened = pos.get("opened_at") or pos.get("open_time") or pos.get("timestamp")
    if isinstance(opened, str):
        try:
            opened = datetime.fromisoformat(opened.replace("Z", "+00:00"))
        except ValueError:
            opened = current - timedelta(hours=hours)
    if opened is None:
        opened = current - timedelta(hours=hours)
    nights = _rollover_nights(opened, current)
    asset = classify_symbol(symbol)
    cfg = default_cost_config("ctrader", symbol)
    lots = abs(float(pos.get("quantity") or pos.get("lots") or 0.01))
    pip_value = float(os.getenv("CTRADER_PIP_VALUE_PER_LOT", "10") or 10)
    swap = swap_cost_usd(
        lots=lots,
        pip_value_per_lot=pip_value,
        side=_position_side(pos),
        nights=nights,
        asset_class=asset,
        long_pips=cfg.swap_long_pips,
        short_pips=cfg.swap_short_pips,
    )
    # One more rollover if we keep holding through tonight.
    tonight = current.replace(hour=21, minute=0, second=0, microsecond=0)
    if tonight <= current:
        tonight = tonight + timedelta(days=1)
    extra = swap_multiplier(tonight, asset) * (
        cfg.swap_long_pips if _position_side(pos) in {"BUY", "LONG"} else cfg.swap_short_pips
    ) * lots * pip_value
    total_swap = swap + extra
    result["swap_usd"] = total_swap
    result["nights"] = len(nights)

    remaining = expected_remaining_usd
    if remaining is None:
        tp = float(pos.get("take_profit") or 0.0)
        entry = float(pos.get("entry_price") or 0.0)
        current_px = float(pos.get("current_price") or entry)
        if tp > 0 and current_px > 0:
            remaining = abs(tp - current_px) * lots * pip_value / max(cfg.pip_size, 1e-9) * cfg.pip_size
        else:
            remaining = 0.0
    result["expected_remaining_usd"] = remaining
    ev = float(remaining) - total_swap
    result["expected_value_after_swap"] = ev
    if ev < 0:
        result["would_flatten"] = True
        result["reason"] = f"swap {total_swap:.2f} exceeds remaining edge {remaining:.2f}"
        if mode == "enforce":
            result["action"] = "flatten"
        else:
            result["action"] = "hold"
            result["reason"] = f"shadow: would flatten — {result['reason']}"
    else:
        result["reason"] = f"hold EV {ev:.2f} after swap {total_swap:.2f}"
    return result


def apply_swap_holds(positions: list[dict[str, Any]], ctrader, now: Optional[datetime] = None) -> list[dict[str, Any]]:
    decisions = []
    for pos in positions or []:
        decision = evaluate_swap_hold(pos, now=now)
        decisions.append(decision)
        if decision.get("action") == "flatten":
            pid = pos.get("position_id")
            if pid:
                res = ctrader.close_position(pid, symbol=pos.get("symbol"), volume=pos.get("quantity"))
                decision["result"] = res.get("status")
        elif decision.get("would_flatten"):
            logger.warning(
                "[SWAP HOLD][shadow] would flatten %s after %.1fh: %s",
                pos.get("symbol"), decision.get("hold_hours"), decision.get("reason"),
            )
    return decisions
