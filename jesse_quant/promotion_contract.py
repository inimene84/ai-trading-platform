#!/usr/bin/env python3
"""
QTP Promotion Contract v1.0.0 — GPU job ↔ Decision Engine.

The Decision Engine never recomputes DSR/PBO from raw prices. It re-reads
metrics.json, verifies hashes, and applies the boolean gates below.
Verdicts: REJECT | SHADOW | PROMOTE | ROLLBACK
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from barrier_config import (
    ATR_PERIOD,
    BASE_SLIPPAGE,
    DSR_MIN,
    FEE_RATE,
    FUNDING_INTERVAL_HOURS,
    FUNDING_RATE_8H,
    MAX_HOLDING_BARS,
    PBO_MAX,
    SL_ATR_MULT,
    TP_ATR_MULT,
)
from promotion_gates import (
    DSR_GATE,
    PBO_GATE,
    STRATEGY_PT_ATR,
    STRATEGY_SL_ATR,
    evaluate_promotion,
    geometry_matches_live,
)

SPEC_VERSION = "1.0.0"
MIN_CLASS_RECALL = 0.10
MIN_TRADES = 80
MIN_NET_SHARPE_AFTER_COSTS = 0.0

VERDICTS = ("REJECT", "SHADOW", "PROMOTE", "ROLLBACK")


@dataclass(frozen=True)
class ContractDecision:
    verdict: str
    reason: str
    code: str
    dsr: Optional[float]
    pbo: Optional[float]
    geometry_hash: Optional[str]
    feature_schema_hash: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_version": SPEC_VERSION,
            "verdict": self.verdict,
            "reason": self.reason,
            "code": self.code,
            "dsr": self.dsr,
            "pbo": self.pbo,
            "geometry_hash": self.geometry_hash,
            "feature_schema_hash": self.feature_schema_hash,
        }


def canonical_json(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(obj: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def geometry_hash(geometry: Mapping[str, Any]) -> str:
    payload = {k: v for k, v in geometry.items() if k != "hashes"}
    return sha256_hex(payload)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def evaluate_contract(
    geometry: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    live_geometry: Optional[Mapping[str, Any]] = None,
    promote_requested: bool = False,
    holdout_spent: bool = False,
    holdout_pass: bool = False,
) -> ContractDecision:
    """Walk the SPEC gate table in order. First hard failure wins REJECT."""
    live_geometry = live_geometry or geometry
    geo_hash = geometry_hash(geometry)
    live_hash = geometry_hash(live_geometry)
    feat_hash = metrics.get("feature_schema_hash") or metrics.get("feature_hash")
    dsr = _as_float(metrics.get("deflated_sharpe_ratio") or (metrics.get("dsr") or {}).get("dsr") if isinstance(metrics.get("dsr"), dict) else metrics.get("deflated_sharpe_ratio"))
    pbo = _as_float(metrics.get("prob_backtest_overfitting") or (metrics.get("pbo") or {}).get("pbo") if isinstance(metrics.get("pbo"), dict) else metrics.get("prob_backtest_overfitting"))

    spec = str(metrics.get("spec_version") or geometry.get("spec_version") or "")
    if spec != SPEC_VERSION:
        return ContractDecision("REJECT", f"spec_version {spec!r} != {SPEC_VERSION}", "SPEC_VERSION", dsr, pbo, geo_hash, feat_hash)

    if geo_hash != live_hash:
        return ContractDecision("REJECT", "geometry_hash mismatch vs live strategy", "GEOMETRY_HASH", dsr, pbo, geo_hash, feat_hash)

    sl = _as_float(geometry.get("triple_barrier", {}).get("sl_atr_mult") if isinstance(geometry.get("triple_barrier"), dict) else geometry.get("sl_atr_mult"))
    pt = _as_float(geometry.get("triple_barrier", {}).get("pt_atr_mult") if isinstance(geometry.get("triple_barrier"), dict) else geometry.get("pt_atr_mult"))
    if sl is None:
        sl = _as_float(metrics.get("sl_mult"))
    if pt is None:
        pt = _as_float(metrics.get("pt_mult"))
    if not geometry_matches_live(pt, sl):
        return ContractDecision(
            "REJECT",
            f"live lock failed: pt={pt} sl={sl} expected {STRATEGY_PT_ATR}/{STRATEGY_SL_ATR}",
            "GEOMETRY_LIVE_LOCK",
            dsr,
            pbo,
            geo_hash,
            feat_hash,
        )

    if metrics.get("used_raw_n_as_effective") is True:
        return ContractDecision("REJECT", "n_trials used raw Optuna count as effective", "N_TRIALS_NOT_EFFECTIVE", dsr, pbo, geo_hash, feat_hash)

    if dsr is None:
        return ContractDecision("REJECT", "DSR missing or non-finite", "DSR_MISSING", dsr, pbo, geo_hash, feat_hash)
    if dsr < DSR_GATE:
        return ContractDecision("REJECT", f"DSR {dsr:.4f} < {DSR_GATE}", "DSR_FLOOR", dsr, pbo, geo_hash, feat_hash)

    n_configs = metrics.get("n_grid") or metrics.get("n_trials") or (metrics.get("pbo") or {}).get("n_configs") if isinstance(metrics.get("pbo"), dict) else metrics.get("n_grid")
    try:
        n_configs_i = int(n_configs) if n_configs is not None else 0
    except (TypeError, ValueError):
        n_configs_i = 0
    if pbo is None:
        if n_configs_i < 2:
            return ContractDecision("REJECT", "PBO infeasible (too few configs)", "PBO_INFEASIBLE", dsr, pbo, geo_hash, feat_hash)
        return ContractDecision("REJECT", "PBO missing", "PBO_INFEASIBLE", dsr, pbo, geo_hash, feat_hash)
    if pbo >= PBO_GATE:
        return ContractDecision("REJECT", f"PBO {pbo:.1%} >= {PBO_GATE:.0%}", "PBO_CEILING", dsr, pbo, geo_hash, feat_hash)

    bullish = _as_float(metrics.get("bullish_recall"))
    bearish = _as_float(metrics.get("bearish_recall"))
    if bullish is not None and bullish < MIN_CLASS_RECALL:
        return ContractDecision(
            "REJECT",
            f"bullish recall {bullish:.1%} < {MIN_CLASS_RECALL:.0%} — collapsed/weak BUY class",
            "COLLAPSED_CLASSIFIER",
            dsr,
            pbo,
            geo_hash,
            feat_hash,
        )
    if bearish is not None and bearish < MIN_CLASS_RECALL:
        return ContractDecision(
            "REJECT",
            f"bearish/fail recall {bearish:.1%} < {MIN_CLASS_RECALL:.0%}",
            "COLLAPSED_CLASSIFIER",
            dsr,
            pbo,
            geo_hash,
            feat_hash,
        )

    n_trades = metrics.get("n_events") or metrics.get("n_trades") or (metrics.get("costed") or {}).get("n_trades") if isinstance(metrics.get("costed"), dict) else metrics.get("n_events")
    try:
        n_trades_i = int(n_trades) if n_trades is not None else 0
    except (TypeError, ValueError):
        n_trades_i = 0
    if n_trades_i and n_trades_i < MIN_TRADES:
        return ContractDecision("REJECT", f"n_trades {n_trades_i} < {MIN_TRADES}", "MIN_TRADES", dsr, pbo, geo_hash, feat_hash)

    holdout_sr = _as_float(metrics.get("holdout_sharpe") or (metrics.get("costed") or {}).get("sharpe_annualized") if isinstance(metrics.get("costed"), dict) else metrics.get("holdout_sharpe"))
    if holdout_sr is not None and holdout_sr < MIN_NET_SHARPE_AFTER_COSTS:
        return ContractDecision("REJECT", f"costed Sharpe {holdout_sr:.4f} < {MIN_NET_SHARPE_AFTER_COSTS}", "NET_SHARPE", dsr, pbo, geo_hash, feat_hash)

    # Authoritative overlap with the existing machine gate (geometry/DSR/PBO/collapse).
    promo = evaluate_promotion(metrics, pt_mult=pt, sl_mult=sl, allow_overfit=False)
    if not promo.ok:
        return ContractDecision("REJECT", promo.reason, "PROMOTION_GATES", dsr, pbo, geo_hash, feat_hash)

    if holdout_spent:
        return ContractDecision("REJECT", "holdout_id already spent", "HOLDOUT_SPENT", dsr, pbo, geo_hash, feat_hash)

    if promote_requested:
        if not holdout_pass:
            return ContractDecision("REJECT", "holdout required for PROMOTE and did not pass", "HOLDOUT_FAIL", dsr, pbo, geo_hash, feat_hash)
        return ContractDecision("PROMOTE", f"PASS DSR={dsr:.4f} PBO={pbo:.1%} geometry={pt}x/{sl}x", "PASS", dsr, pbo, geo_hash, feat_hash)

    return ContractDecision("SHADOW", f"hard gates pass; holdout unspent (shadow only) DSR={dsr:.4f} PBO={pbo:.1%}", "SHADOW", dsr, pbo, geo_hash, feat_hash)


def default_geometry() -> Dict[str, Any]:
    geo = {
        "spec_version": SPEC_VERSION,
        "strategy_id": "QuantumAIStrategy",
        "triple_barrier": {
            "sl_atr_mult": SL_ATR_MULT,
            "pt_atr_mult": TP_ATR_MULT,
            "atr_period": ATR_PERIOD,
            "vertical_timeout_bars": MAX_HOLDING_BARS,
        },
        "bar": {
            "timeframe": "1h",
            "primary_type": "time",
            "bars_per_year": 8760,
        },
        "costs": {
            "assume_fill": "taker",
            "fee_bps_per_side": FEE_RATE * 10_000.0,
            "base_slippage": BASE_SLIPPAGE,
            "funding_rate_8h": FUNDING_RATE_8H,
            "funding_interval_hours": FUNDING_INTERVAL_HOURS,
        },
        "gates": {
            "dsr_min": DSR_MIN,
            "pbo_max": PBO_MAX,
            "min_class_recall": MIN_CLASS_RECALL,
            "min_trades": MIN_TRADES,
            "min_net_sharpe_after_costs": MIN_NET_SHARPE_AFTER_COSTS,
        },
        "live": {
            "p_win_min": 0.45,
            "width_max": 0.50,
        },
        "validation": {
            "purge_horizon": MAX_HOLDING_BARS,
            "embargo_bars": max(MAX_HOLDING_BARS, 200),
            "embargo_fraction": 0.01,
            "cpcv_n_splits": 16,
            "note": "embargo_bars wins over embargo_fraction; must be >= max(48, longest lookback)",
        },
    }
    geo["hashes"] = {"geometry_hash": geometry_hash(geo)}
    return geo


def write_geometry(path: str, geometry: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    geo = dict(geometry or default_geometry())
    if "hashes" not in geo:
        geo["hashes"] = {"geometry_hash": geometry_hash(geo)}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(geo, fh, indent=2)
    return geo
