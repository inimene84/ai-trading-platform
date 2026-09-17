"""Optional live four-number check for a promoted model (contract §7).

side, p_win, conformal_width, costed_edge_bps — skip the order if any live
gate fails. Missing p_win / width / edge telemetry is a veto (fail-closed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from backend.ml.geometry import DEFAULT_GATES, DEFAULT_LIVE_SIGNAL


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
