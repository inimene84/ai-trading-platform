"""
Institutional gates for Jesse ML artifacts and Fractional Kelly sizing.

The live QuantumAIStrategy geometry is 1.75 ATR stop / 5.5 ATR take-profit.
Models trained on any other barrier pair are predicting a trade the live book
will never take. Promotion additionally requires DSR > 0.95 and PBO < 0.30,
with Deflated Sharpe computed against the true trial count (hyperparameter
grid size), not a hardcoded n_trials=5 that collapses DSR to 1.0.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

# Live strategy geometry (jesse4 / optimizer-confirmed).
STRATEGY_SL_ATR = 1.75
STRATEGY_PT_ATR = 5.5
STRATEGY_PAYOFF_RATIO = STRATEGY_PT_ATR / STRATEGY_SL_ATR  # ~3.14

DSR_GATE = 0.95
PBO_GATE = 0.30
KELLY_FRACTION = 0.25
KELLY_BASELINE = 0.125  # quarter-Kelly at 55% win / 2:1 payoff
MIN_CLOSED_TRADES_FOR_EMPIRICAL_B = 30
MIN_CLASS_RECALL = 0.10


@dataclass(frozen=True)
class PromotionDecision:
    ok: bool
    reason: str
    dsr: Optional[float]
    pbo: Optional[float]
    pt_mult: Optional[float]
    sl_mult: Optional[float]


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def geometry_matches_live(
    pt_mult: Optional[float],
    sl_mult: Optional[float],
    tol: float = 0.05,
) -> bool:
    if pt_mult is None or sl_mult is None:
        return False
    return abs(pt_mult - STRATEGY_PT_ATR) <= tol and abs(sl_mult - STRATEGY_SL_ATR) <= tol


def evaluate_promotion(
    metrics: Optional[Mapping[str, Any]] = None,
    *,
    pt_mult: Optional[float] = None,
    sl_mult: Optional[float] = None,
    allow_overfit: Optional[bool] = None,
    require_geometry: bool = True,
) -> PromotionDecision:
    """
    Machine-enforce the documented deployment gates.

    allow_overfit: when True (or JESSE_ML_ALLOW_OVERFIT=true), DSR/PBO failures
    become warnings rather than hard blocks. Geometry mismatch still fails
    unless require_geometry is False — a model trained on the wrong trade is
    never a valid live meta-labeler.
    """
    metrics = metrics or {}
    dsr = _as_float(metrics.get("deflated_sharpe_ratio"))
    pbo = _as_float(metrics.get("prob_backtest_overfitting"))
    pt = _as_float(pt_mult if pt_mult is not None else metrics.get("pt_mult"))
    sl = _as_float(sl_mult if sl_mult is not None else metrics.get("sl_mult"))

    if allow_overfit is None:
        allow_overfit = os.getenv("JESSE_ML_ALLOW_OVERFIT", "false").lower() == "true"

    if require_geometry and not geometry_matches_live(pt, sl):
        return PromotionDecision(
            ok=False,
            reason=(
                f"triple-barrier geometry {pt}x/{sl}x ATR does not match live "
                f"{STRATEGY_PT_ATR}x/{STRATEGY_SL_ATR}x — model predicts an untaken trade"
            ),
            dsr=dsr,
            pbo=pbo,
            pt_mult=pt,
            sl_mult=sl,
        )

    bullish_recall = _as_float(metrics.get("bullish_recall"))
    bearish_recall = _as_float(metrics.get("bearish_recall"))
    if (
        bullish_recall is not None
        and bearish_recall is not None
        and bullish_recall < 0.05
        and bearish_recall > 0.90
    ):
        return PromotionDecision(
            ok=False,
            reason=(
                f"collapsed classifier (bullish recall {bullish_recall:.1%}, "
                f"bearish recall {bearish_recall:.1%}) — would veto every BUY"
            ),
            dsr=dsr,
            pbo=pbo,
            pt_mult=pt,
            sl_mult=sl,
        )
    if bullish_recall is not None and bullish_recall < MIN_CLASS_RECALL:
        return PromotionDecision(
            ok=False,
            reason=(
                f"insufficient bullish recall {bullish_recall:.1%} "
                f"< {MIN_CLASS_RECALL:.0%} both-class floor"
            ),
            dsr=dsr,
            pbo=pbo,
            pt_mult=pt,
            sl_mult=sl,
        )
    if bearish_recall is not None and bearish_recall < MIN_CLASS_RECALL:
        return PromotionDecision(
            ok=False,
            reason=(
                f"insufficient fail-class recall {bearish_recall:.1%} "
                f"< {MIN_CLASS_RECALL:.0%} both-class floor"
            ),
            dsr=dsr,
            pbo=pbo,
            pt_mult=pt,
            sl_mult=sl,
        )

    if dsr is None or pbo is None:
        if allow_overfit:
            return PromotionDecision(
                ok=True,
                reason="DSR/PBO missing; JESSE_ML_ALLOW_OVERFIT override enabled",
                dsr=dsr,
                pbo=pbo,
                pt_mult=pt,
                sl_mult=sl,
            )
        return PromotionDecision(
            ok=False,
            reason="model metadata missing DSR or PBO — fail closed",
            dsr=dsr,
            pbo=pbo,
            pt_mult=pt,
            sl_mult=sl,
        )

    if dsr < DSR_GATE:
        reason = f"DSR {dsr:.4f} < {DSR_GATE} gate"
        if not allow_overfit:
            return PromotionDecision(ok=False, reason=reason, dsr=dsr, pbo=pbo, pt_mult=pt, sl_mult=sl)
    elif pbo >= PBO_GATE:
        reason = f"PBO {pbo:.1%} >= {PBO_GATE:.0%} gate"
        if not allow_overfit:
            return PromotionDecision(ok=False, reason=reason, dsr=dsr, pbo=pbo, pt_mult=pt, sl_mult=sl)
    else:
        reason = f"PASS DSR={dsr:.4f} PBO={pbo:.1%} geometry={pt}x/{sl}x"

    return PromotionDecision(
        ok=True,
        reason=("OVERRIDE " if allow_overfit and (dsr < DSR_GATE or pbo >= PBO_GATE) else "") + reason,
        dsr=dsr,
        pbo=pbo,
        pt_mult=pt,
        sl_mult=sl,
    )


def payoff_ratio_from_geometry(
    pt_mult: Optional[float] = None,
    sl_mult: Optional[float] = None,
) -> float:
    pt = _as_float(pt_mult) or STRATEGY_PT_ATR
    sl = _as_float(sl_mult) or STRATEGY_SL_ATR
    if sl <= 0:
        return STRATEGY_PAYOFF_RATIO
    return max(0.5, min(pt / sl, 8.0))


def empirical_payoff_ratio(
    avg_win: float,
    avg_loss_abs: float,
    closed_count: int,
    *,
    fallback: float = STRATEGY_PAYOFF_RATIO,
    min_closed: int = MIN_CLOSED_TRADES_FOR_EMPIRICAL_B,
) -> float:
    """b = avg win / avg |loss|. Geometry fallback until the book has enough trades."""
    if closed_count < min_closed or avg_win <= 0 or avg_loss_abs <= 0:
        return fallback
    return max(0.5, min(avg_win / avg_loss_abs, 8.0))


def calculate_fractional_kelly(
    win_prob: float,
    payoff_ratio: float = STRATEGY_PAYOFF_RATIO,
    fraction: float = KELLY_FRACTION,
) -> Dict[str, float]:
    """
    f* = fraction * (p * b - (1 - p)) / b
    Size multiplier is scaled to a 0.125 quarter-Kelly baseline and clipped.
    """
    b = payoff_ratio if payoff_ratio > 0 else 1.0
    p = min(max(float(win_prob), 0.0), 1.0)
    full_kelly = (p * b - (1.0 - p)) / b
    fractional = max(0.0, full_kelly) * fraction
    if fractional <= 0:
        size_multiplier = 0.0
    else:
        size_multiplier = float(max(0.20, min(fractional / KELLY_BASELINE, 2.0)))
    return {
        "fractional_kelly": round(float(fractional), 4),
        "full_kelly": round(float(full_kelly), 4),
        "payoff_ratio": round(float(b), 4),
        "size_multiplier": round(size_multiplier, 3),
        "win_prob": round(p, 4),
    }


def annotate_ml_prediction(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Attach promotion decision; downgrade status to error when the gate fails."""
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    nested = {
        "deflated_sharpe_ratio": payload.get("deflated_sharpe_ratio", metrics.get("deflated_sharpe_ratio")),
        "prob_backtest_overfitting": payload.get("prob_backtest_overfitting", metrics.get("prob_backtest_overfitting")),
        "pt_mult": payload.get("pt_mult", metrics.get("pt_mult")),
        "sl_mult": payload.get("sl_mult", metrics.get("sl_mult")),
    }
    # Prefer nested metrics when top-level is absent.
    for key, value in list(nested.items()):
        if value is None:
            nested[key] = metrics.get(key)

    decision = evaluate_promotion(
        nested,
        pt_mult=nested.get("pt_mult"),
        sl_mult=nested.get("sl_mult"),
    )
    payload["promotion_ok"] = decision.ok
    payload["promotion_reason"] = decision.reason
    payload["deflated_sharpe_ratio"] = decision.dsr
    payload["prob_backtest_overfitting"] = decision.pbo
    if not decision.ok and payload.get("status") == "success":
        payload["status"] = "error"
        payload["error"] = f"Jesse ML promotion gate: {decision.reason}"
        payload["signal"] = "NEUTRAL"
        payload["gated"] = True
        payload["gated_reason"] = decision.reason
    return payload


def clip_kelly_for_thin_book(
    size_multiplier: float,
    closed_count: int,
    min_closed: int = MIN_CLOSED_TRADES_FOR_EMPIRICAL_B,
) -> Tuple[float, bool]:
    """Until 30 closed trades exist in the active partition, clip Kelly to [0.25, 1.0]."""
    if closed_count >= min_closed:
        return float(size_multiplier), False
    clipped = max(0.25, min(1.0, float(size_multiplier)))
    return clipped, clipped != float(size_multiplier)
