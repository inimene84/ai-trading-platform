"""Reliability-diagram train/deploy control for CalibratedClassifierCV.

Kelly sizing uses the *absolute* probability, so a model that ranks well but
is miscalibrated will over- or under-bet. This module does not train; it
audits holdout probabilities after isotonic/Platt calibration.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np

SLOPE_TOL = 0.35
INTERCEPT_TOL = 0.15
MAX_BRIER = 0.30


def reliability_diagram(
    y_true: Iterable[Any],
    y_prob: Iterable[Any],
    *,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Bin predicted P(win) vs observed win rate. Returns slope / Brier / gate."""
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(y_prob), dtype=float)
    if y.size == 0 or p.size != y.size:
        return {
            "bins": [],
            "brier": None,
            "slope": None,
            "intercept": None,
            "near_diagonal": False,
            "n": 0,
            "reasons": ["empty or mismatched reliability sample"],
        }
    p = np.clip(p, 0.0, 1.0)
    y = (y > 0).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    xs: list[float] = []
    ys: list[float] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(mask.sum())
        pred = float(p[mask].mean()) if count else None
        obs = float(y[mask].mean()) if count else None
        bins.append(
            {
                "lo": lo,
                "hi": hi,
                "count": count,
                "predicted": pred,
                "observed": obs,
            }
        )
        if count >= 3 and pred is not None and obs is not None:
            xs.append(pred)
            ys.append(obs)
    brier = float(np.mean((p - y) ** 2))
    slope = None
    intercept = None
    if len(xs) >= 2 and float(np.std(xs)) > 1e-9:
        slope, intercept = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
        slope = float(slope)
        intercept = float(intercept)
    reasons: list[str] = []
    near = True
    if slope is None:
        near = False
        reasons.append("not enough populated bins to fit a reliability slope")
    else:
        if abs(slope - 1.0) > SLOPE_TOL:
            near = False
            reasons.append(f"|slope-1|={abs(slope - 1.0):.3f} > {SLOPE_TOL}")
        if abs(intercept) > INTERCEPT_TOL:
            near = False
            reasons.append(f"|intercept|={abs(intercept):.3f} > {INTERCEPT_TOL}")
    if brier > MAX_BRIER:
        near = False
        reasons.append(f"brier {brier:.3f} > {MAX_BRIER}")
    return {
        "bins": bins,
        "brier": round(brier, 6),
        "slope": None if slope is None else round(slope, 4),
        "intercept": None if intercept is None else round(intercept, 4),
        "near_diagonal": bool(near),
        "n": int(y.size),
        "n_bins": n_bins,
        "reasons": reasons,
    }


def calibration_deploy_control(diagram: Dict[str, Any]) -> Dict[str, Any]:
    """Train/deploy gate: Kelly may use p only when the curve is near diagonal."""
    ok = bool(diagram.get("near_diagonal"))
    return {
        "calibration_ok": ok,
        "method": "CalibratedClassifierCV",
        "reliability": diagram,
        "kelly_cap_if_miscalibrated": 0.01 if not ok else None,
        "reason": "near diagonal" if ok else "; ".join(diagram.get("reasons") or ["miscalibrated"]),
    }
