"""Binance funding-rate entry filter.

Longs pay a positive funding rate; shorts pay a negative one. When the
next funding print strongly opposes the intended side, this module can
veto the entry or size it down.

Modes (``FUNDING_FILTER_MODE``):
    off      — no-op (default-safe companion; live default is shadow)
    shadow   — log / annotate what enforce would do; never blocks (default)
    enforce  — veto or size-down

Does not place or cancel orders. Fail-open on fetch errors so a stale
premiumIndex cannot halt the book.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

FundingMode = Literal["off", "shadow", "enforce"]
FundingAction = Literal["allow", "size_down", "veto"]


def funding_filter_mode() -> FundingMode:
    raw = (os.getenv("FUNDING_FILTER_MODE", "shadow") or "shadow").strip().lower()
    if raw in {"off", "shadow", "enforce"}:
        return raw  # type: ignore[return-value]
    return "shadow"


def funding_filter_threshold() -> float:
    """Absolute funding rate that counts as 'strongly opposing' (default 5 bps)."""
    try:
        return abs(float(os.getenv("FUNDING_FILTER_THRESHOLD", "0.0005") or 0.0005))
    except (TypeError, ValueError):
        return 0.0005


def funding_filter_size_mult() -> float:
    """Size multiplier when funding opposes but is below the veto threshold."""
    try:
        mult = float(os.getenv("FUNDING_FILTER_SIZE_MULT", "0.5") or 0.5)
    except (TypeError, ValueError):
        return 0.5
    return min(1.0, max(0.0, mult))


def funding_opposes_side(direction: str, funding_rate: float) -> bool:
    """True when the next funding payment is charged to this side.

    Positive rate: longs pay shorts. Negative rate: shorts pay longs.
    """
    side = (direction or "").strip().upper()
    rate = float(funding_rate or 0.0)
    if side in {"BUY", "LONG"}:
        return rate > 0.0
    if side in {"SELL", "SHORT"}:
        return rate < 0.0
    return False


def evaluate_funding_filter(
    *,
    direction: str,
    funding_rate: Optional[float],
    threshold: Optional[float] = None,
    size_mult: Optional[float] = None,
    mode: Optional[str] = None,
) -> dict[str, Any]:
    """Decide allow / size_down / veto for one entry.

    Missing/None funding is fail-open (allow). Size-down applies when
    funding opposes the side but ``|rate|`` is below the veto threshold
    and the size multiplier is < 1.
    """
    resolved_mode: FundingMode
    raw_mode = (mode or funding_filter_mode()).strip().lower()
    resolved_mode = raw_mode if raw_mode in {"off", "shadow", "enforce"} else "shadow"  # type: ignore[assignment]
    thresh = funding_filter_threshold() if threshold is None else abs(float(threshold))
    mult = funding_filter_size_mult() if size_mult is None else min(1.0, max(0.0, float(size_mult)))

    result: dict[str, Any] = {
        "mode": resolved_mode,
        "action": "allow",
        "would_action": "allow",
        "funding_rate": None if funding_rate is None else float(funding_rate),
        "threshold": thresh,
        "size_mult": 1.0,
        "opposes": False,
        "reason": "funding_filter_off" if resolved_mode == "off" else "funding_ok",
    }
    if resolved_mode == "off":
        return result
    if funding_rate is None:
        result["reason"] = "funding_unavailable_fail_open"
        return result

    rate = float(funding_rate)
    result["funding_rate"] = rate
    opposes = funding_opposes_side(direction, rate)
    result["opposes"] = opposes
    if not opposes or abs(rate) < 1e-12:
        result["reason"] = "funding_aligned_or_flat"
        return result

    if abs(rate) >= thresh:
        would: FundingAction = "veto"
        reason = (
            f"funding {rate:+.6f} opposes {direction} at/above threshold {thresh:.6f}"
        )
    elif mult < 1.0:
        would = "size_down"
        reason = (
            f"funding {rate:+.6f} opposes {direction}; size x{mult:.2f} "
            f"(below veto threshold {thresh:.6f})"
        )
    else:
        would = "allow"
        reason = f"funding {rate:+.6f} opposes {direction} but below threshold"

    result["would_action"] = would
    result["reason"] = reason
    if would == "size_down":
        result["size_mult"] = mult

    if resolved_mode == "enforce":
        result["action"] = would
    else:
        result["action"] = "allow"
        result["reason"] = f"shadow: would {would} — {reason}"
    return result


async def fetch_funding_rate(symbol: str) -> Optional[float]:
    """Best-effort premiumIndex / fundingRate fetch. None on any failure."""
    try:
        from backend.services.binance_market_data import binance_market_data

        data = await binance_market_data.get_funding_rate(symbol)
        if not data:
            return None
        return float(data.get("fundingRate", 0.0) or 0.0)
    except Exception as exc:
        logger.warning("[FUNDING] fetch failed for %s: %s", symbol, exc)
        return None


def apply_size_mult(size_data: dict[str, Any], mult: float) -> dict[str, Any]:
    """Scale quantity / lots / risk_usd when the filter size-downs."""
    if mult >= 1.0 or mult <= 0.0:
        return size_data
    scaled = dict(size_data)
    for key in ("quantity", "lots", "risk_usd", "notional", "margin_required"):
        if key in scaled and scaled[key] is not None:
            try:
                scaled[key] = float(scaled[key]) * mult
            except (TypeError, ValueError):
                pass
    scaled["funding_size_mult"] = mult
    return scaled
