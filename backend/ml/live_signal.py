"""Optional live four-number check for a promoted model (contract §7).

side, p_win, conformal_width, costed_edge_bps — skip the order if any live
gate fails. Missing p_win / width / edge telemetry is a veto (fail-closed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from backend.ml.geometry import (
    DEFAULT_GATES,
    DEFAULT_LIVE_SIGNAL,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_TAKER_FEE_RATE,
    HOUSE_SL_ATR_MULT,
)

# Jesse /predict does not ship ATR; use a conservative notional proxy for EV→bps.
_DEFAULT_ATR_NOTIONAL_FRAC = 0.015


@dataclass
class LiveFourNumberDecision:
    allowed: bool
    reason: str
    side: Optional[str] = None
    p_win: Optional[float] = None
    conformal_width: Optional[float] = None
    costed_edge_bps: Optional[float] = None
    applied: bool = False


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def expected_value_r_to_costed_edge_bps(
    ev_r: float,
    *,
    sl_atr_mult: float = HOUSE_SL_ATR_MULT,
    atr_notional_frac: float = _DEFAULT_ATR_NOTIONAL_FRAC,
) -> float:
    """Convert Jesse decision EV (R-multiples) to basis points after round-trip costs."""
    round_trip_bps = 2.0 * (DEFAULT_TAKER_FEE_RATE + DEFAULT_SLIPPAGE_RATE) * 10_000.0
    gross_bps = float(ev_r) * float(sl_atr_mult) * float(atr_notional_frac) * 10_000.0
    return gross_bps - round_trip_bps


def map_jesse_prediction_to_live_telemetry(
    payload: Mapping[str, Any],
) -> dict[str, Optional[float]]:
    """Map Jesse /predict fields to QTP four-number contract names.

    Jesse serves ``conformal_margin`` and ``decision.expected_value_r``; the
    promotion contract expects ``conformal_width`` and ``costed_edge_bps``.
    """
    side = str(payload.get("signal") or "").upper()
    probs = payload.get("probabilities") if isinstance(payload.get("probabilities"), Mapping) else {}

    p_win: Optional[float] = None
    if side == "BUY":
        p_win = _float_or_none(probs.get("bullish")) or _float_or_none(payload.get("confidence"))
    elif side == "SELL":
        p_win = _float_or_none(probs.get("bearish")) or _float_or_none(payload.get("confidence"))
    else:
        p_win = _float_or_none(payload.get("confidence"))

    margin = _float_or_none(payload.get("conformal_margin"))
    entropy = _float_or_none(payload.get("entropy"))
    conformal_width: Optional[float] = None
    if margin is not None and entropy is not None:
        conformal_width = max(0.0, entropy - margin)
    elif margin is not None:
        conformal_width = max(0.0, 1.0 - margin)
    elif entropy is not None:
        conformal_width = float(entropy)

    decision = payload.get("decision") if isinstance(payload.get("decision"), Mapping) else {}
    kelly = payload.get("kelly") if isinstance(payload.get("kelly"), Mapping) else {}
    ev_r = _float_or_none(decision.get("expected_value_r"))
    if ev_r is None:
        ev_r = _float_or_none(kelly.get("expected_value_r"))

    costed_edge_bps: Optional[float] = None
    if ev_r is not None:
        barrier = payload.get("barrier_geometry") if isinstance(payload.get("barrier_geometry"), Mapping) else {}
        sl_mult = _float_or_none(barrier.get("sl_atr_mult")) or HOUSE_SL_ATR_MULT
        costed_edge_bps = expected_value_r_to_costed_edge_bps(ev_r, sl_atr_mult=sl_mult)

    return {
        "p_win": p_win,
        "conformal_width": conformal_width,
        "costed_edge_bps": costed_edge_bps,
    }


def attach_jesse_live_telemetry(payload: dict[str, Any]) -> dict[str, Any]:
    """Fill QTP four-number fields on a Jesse /predict payload when absent."""
    mapped = map_jesse_prediction_to_live_telemetry(payload)
    for key, value in mapped.items():
        if payload.get(key) is None and value is not None:
            payload[key] = value
    return payload


def evaluate_live_four_numbers(
    signal: Mapping[str, Any],
    *,
    geometry: Mapping[str, Any] | None = None,
) -> LiveFourNumberDecision:
    """Return whether a promoted-model order may proceed.

    Skip (block) if p_win < live.p_win_min OR conformal_width > live.width_max
    OR costed_edge_bps <= gates.min_costed_edge_bps. Missing any of the three
    telemetry numbers is a veto — do not invent a pass from confidence alone.
    """
    geo = dict(geometry or {})
    live = dict(DEFAULT_LIVE_SIGNAL)
    live.update(dict(geo.get("live") or {}))
    gates = dict(DEFAULT_GATES)
    gates.update(dict(geo.get("gates") or {}))

    side = signal.get("side") or signal.get("signal")
    p_win = _float_or_none(signal.get("p_win"))
    width = _float_or_none(signal.get("conformal_width"))
    edge = _float_or_none(signal.get("costed_edge_bps"))

    missing = [name for name, value in (
        ("p_win", p_win),
        ("conformal_width", width),
        ("costed_edge_bps", edge),
    ) if value is None]
    if missing:
        return LiveFourNumberDecision(
            allowed=False,
            reason=f"missing live telemetry: {', '.join(missing)}",
            side=str(side) if side is not None else None,
            p_win=p_win,
            conformal_width=width,
            costed_edge_bps=edge,
            applied=True,
        )

    p_win_min = float(live.get("p_win_min", 0.55))
    width_max = float(live.get("width_max", 0.35))
    min_edge = float(gates.get("min_costed_edge_bps", 0.0))

    if p_win is not None and p_win < p_win_min:
        return LiveFourNumberDecision(
            allowed=False,
            reason=f"p_win {p_win:.4f} < live.p_win_min {p_win_min:.4f}",
            side=str(side) if side is not None else None,
            p_win=p_win,
            conformal_width=width,
            costed_edge_bps=edge,
            applied=True,
        )
    if width is not None and width > width_max:
        return LiveFourNumberDecision(
            allowed=False,
            reason=f"conformal_width {width:.4f} > live.width_max {width_max:.4f}",
            side=str(side) if side is not None else None,
            p_win=p_win,
            conformal_width=width,
            costed_edge_bps=edge,
            applied=True,
        )
    if edge is not None and edge <= min_edge:
        return LiveFourNumberDecision(
            allowed=False,
            reason=f"costed_edge_bps {edge:.4f} <= gates.min_costed_edge_bps {min_edge:.4f}",
            side=str(side) if side is not None else None,
            p_win=p_win,
            conformal_width=width,
            costed_edge_bps=edge,
            applied=True,
        )
    return LiveFourNumberDecision(
        allowed=True,
        reason="live four-number check passed",
        side=str(side) if side is not None else None,
        p_win=p_win,
        conformal_width=width,
        costed_edge_bps=edge,
        applied=True,
    )
