#!/usr/bin/env python3
"""
Jesse Machine Learning Real-Time Prediction Microservice (Institutional v3.0)
Caches loaded models in memory for fast probability scoring accessible via HTTP REST
API from ai-trading-backend, n8n, or AI agents.

Features:
  - Feature schema hash validation to eliminate train-serve skew
  - Machine-enforced promotion gate: DSR > 0.95, PBO < 0.30, both-class recall ≥ 10%
    (collapsed always-SELL artifacts are refused). ML_GATE_OVERRIDE is research-only.
  - Geometry-aware decision rule (signal_policy) using the deployed SL/TP ATR geometry
  - Fractional Kelly sizing with an *empirical* payoff ratio b taken from the artifact
    (realised avg win / avg loss of the training-set barrier outcomes), falling back to
    the theoretical TP/SL ratio of the barrier geometry
  - Isotonic probability calibration & split-conformal uncertainty gating
  - Model expiration monitoring (168-hour staleness guard)
"""

import glob
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # type: ignore[assignment]

from barrier_config import (
    DSR_MIN,
    PBO_MAX,
    breakeven_win_probability,
    geometry_dict,
    theoretical_payoff_ratio,
)
from feature_schema import FEATURE_HASH
from fracdiff import resolve_artifact_d
from ml_features import FEATURE_NAMES, compute_latest_features
from asset_universe import candle_timeframe_candidates, normalize_symbol as jesse_symbol_from_compact
from promotion_gates import MIN_CLASS_RECALL, annotate_ml_prediction, artifact_gate as evaluate_artifact_gate
from signal_policy import DEFAULT_DOMINANCE, DEFAULT_MIN_EV_R, decide_signal, expected_value_r

app = FastAPI(title="Jesse ML Inference Engine (Institutional)", version="3.0.0")

# Database settings
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_NAME = os.getenv("POSTGRES_NAME", "jesse_db")
DB_USER = os.getenv("POSTGRES_USERNAME", "jesse_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_SEARCH_DIRS = [d for d in [os.getenv("ML_MODELS_DIR"), os.path.join(REPO_ROOT, "storage", "models")] if d]
# QTP control plane (/root/qtp-training) may write SHADOW bundles under
# qtp-training/artifacts/. Do not add that directory here. Shadow artifacts
# must not size or send orders. Do not set ML_GATE_OVERRIDE for QTP ingest.
SERVER_PORT = int(os.getenv("ML_PORT", "9003"))

# Gate override: only for research. Live trading must never set this.
GATE_OVERRIDE = os.getenv("ML_GATE_OVERRIDE", "false").strip().lower() in ("1", "true", "yes", "on")

KELLY_FRACTION = 0.5
KELLY_ABS_CAP = 0.02
MIN_TRADES_FOR_EMPIRICAL_PAYOFF = 30

# Model cache in memory
_MODEL_CACHE: Dict[str, Any] = {}
_REJECTED_CACHE: Dict[str, Dict[str, Any]] = {}


def normalize_symbol(symbol: str) -> str:
    """FX/metals stay EUR-USD / XAU-USD; crypto USDT pairs stay BTC-USDT."""
    return jesse_symbol_from_compact(symbol)


def calculate_conformal_uncertainty(probs: np.ndarray) -> Dict[str, Any]:
    """
    Computes uncertainty bounds:
      - margin: Delta between highest and second highest probability
      - entropy: Normalized Shannon entropy across class distributions [0, 1]
      - uncertainty_level: LOW, MEDIUM, or HIGH
    """
    sorted_p = np.sort(probs)[::-1]
    margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0

    k = len(probs)
    safe_p = np.clip(probs, 1e-12, 1.0)
    entropy = -float(np.sum(safe_p * np.log(safe_p))) / math.log(max(k, 2))

    if margin < 0.08 or entropy > 0.92:
        level = "HIGH"
    elif margin < 0.15 or entropy > 0.80:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {"margin": round(margin, 4), "entropy": round(entropy, 4), "level": level}


def resolve_payoff_ratio(model_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Payoff ratio b for Kelly sizing.
      1. Realised avg-win / avg-loss of the model's trades (holdout) when >= 30 trades
      2. Realised avg-win / avg-loss of all training-set barrier attempts
      3. Theoretical TP / SL of the barrier geometry (e.g. 5.5 / 1.75 = 3.14)
    """
    geometry = (model_data or {}).get("barrier_geometry") or geometry_dict()
    theoretical = float(geometry.get("theoretical_payoff_ratio") or theoretical_payoff_ratio(
        geometry.get("tp_atr_mult", 5.5), geometry.get("sl_atr_mult", 1.75)
    ))
    payoff_meta = ((model_data or {}).get("metrics") or {}).get("payoff") or {}
    model_trades = payoff_meta.get("model_trades_holdout") or {}
    all_attempts = payoff_meta.get("all_attempts_train") or {}

    if model_trades.get("payoff_ratio") and int(model_trades.get("n_trades", 0)) >= MIN_TRADES_FOR_EMPIRICAL_PAYOFF:
        return {
            "payoff_ratio": float(model_trades["payoff_ratio"]),
            "source": "empirical_holdout_model_trades",
            "kind": "net_realized",
            "n": int(model_trades["n_trades"]),
            "theoretical": theoretical,
        }
    if all_attempts.get("payoff_ratio"):
        return {
            "payoff_ratio": float(all_attempts["payoff_ratio"]),
            "source": "empirical_training_attempts",
            "kind": "net_realized",
            "n": int(all_attempts.get("n_trades", 0)),
            "theoretical": theoretical,
        }
    emp = (model_data or {}).get("payoff_ratio_empirical")
    if emp:
        return {"payoff_ratio": float(emp), "source": str((model_data or {}).get("payoff_ratio_source", "artifact")), "n": None, "theoretical": theoretical}
    return {"payoff_ratio": theoretical, "source": "theoretical_geometry", "n": None, "theoretical": theoretical}


def calculate_fractional_kelly(
    win_prob: float,
    payoff_ratio: float = theoretical_payoff_ratio(),
    fraction: float = KELLY_FRACTION,
    abs_cap: float = KELLY_ABS_CAP,
    payoff_source: str = "theoretical_geometry",
) -> Dict[str, Any]:
    """
    Constrained Fractional Kelly position sizing.
      f = min(fraction * (p * b - (1 - p)) / b, abs_cap)
    Half-Kelly by default with a 2% equity ceiling on the wallet fraction.
    The size multiplier normalises the *uncapped* half-Kelly by a reference
    edge (+10pp above break-even) so the 2% cap does not flatten size to 0.20.
    """
    if payoff_ratio <= 0:
        payoff_ratio = 1.0
    b = payoff_ratio
    p = float(win_prob)
    full_kelly = (p * b - (1.0 - p)) / b
    uncapped = max(0.0, full_kelly) * fraction
    fractional_kelly = min(uncapped, float(abs_cap))

    p_ref = min(0.95, breakeven_win_probability(b) + 0.10)
    baseline_kelly = max(1e-6, ((p_ref * b - (1.0 - p_ref)) / b) * fraction)
    size_multiplier = (
        round(float(np.clip(uncapped / baseline_kelly, 0.20, 1.0)), 3) if uncapped > 0 else 0.0
    )

    return {
        "fractional_kelly": round(float(fractional_kelly), 4),
        "fractional_kelly_uncapped": round(float(uncapped), 4),
        "full_kelly": round(float(full_kelly), 4),
        "payoff_ratio": round(b, 4),
        "payoff_ratio_source": payoff_source,
        "breakeven_probability": round(breakeven_win_probability(b), 4),
        "expected_value_r": round(expected_value_r(p, b), 4),
        "kelly_fraction": fraction,
        "kelly_abs_cap": float(abs_cap),
        "baseline_kelly": round(float(baseline_kelly), 4),
        "size_multiplier": size_multiplier,
    }


def check_model_expiration(trained_at_str: Optional[str], expiration_hours: int = 168) -> Dict[str, Any]:
    """Checks if a model has exceeded its freshness SLA (default 7 days / 168 hours)."""
    if not trained_at_str:
        return {"expired": False, "age_hours": None}
    try:
        t_trained = datetime.fromisoformat(trained_at_str.replace("Z", "+00:00"))
        if t_trained.tzinfo is None:
            t_trained = t_trained.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - t_trained).total_seconds() / 3600.0
        return {"expired": age > expiration_hours, "age_hours": round(age, 1), "expiration_hours": expiration_hours}
    except Exception:
        return {"expired": False, "age_hours": None}


def artifact_gate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the promotion gate recorded in an artifact. Old PASS stamps are ignored."""
    return evaluate_artifact_gate(payload, allow_overfit=GATE_OVERRIDE)


def _is_production_artifact(path: str, norm_symbol: str, timeframe: str) -> bool:
    """Only `<SYMBOL>_<TF>_<model>.joblib` counts; rejected/collapsed/candidate variants never do."""
    stem = os.path.basename(path)[: -len(".joblib")]
    parts = stem.split("_")
    return len(parts) == 3 and parts[0] == norm_symbol and parts[1] == timeframe and "." not in parts[2]


def _find_artifact(norm_symbol: str, timeframe: str, model_type: str) -> Optional[str]:
    for sdir in MODEL_SEARCH_DIRS:
        exact = os.path.join(sdir, f"{norm_symbol}_{timeframe}_{model_type}.joblib")
        if os.path.exists(exact):
            return exact
        candidates = sorted(
            p for p in glob.glob(os.path.join(sdir, f"{norm_symbol}_{timeframe}_*.joblib"))
            if _is_production_artifact(p, norm_symbol, timeframe)
        )
        if candidates:
            return candidates[0]
    return None


def get_cached_model(symbol: str, timeframe: str = "1h", model_type: str = "lightgbm") -> Optional[Dict[str, Any]]:
    norm_symbol = normalize_symbol(symbol)
    cache_key = f"{norm_symbol}_{timeframe}_{model_type}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    if cache_key in _REJECTED_CACHE:
        return None

    found_path = _find_artifact(norm_symbol, timeframe, model_type)
    if not found_path:
        return None

    try:
        payload = joblib.load(found_path)
        payload["filename"] = os.path.basename(found_path)

        saved_hash = payload.get("feature_schema_hash") or payload.get("feature_hash")
        payload["feature_schema_hash"] = saved_hash
        payload["schema_parity_ok"] = saved_hash == FEATURE_HASH
        payload["expiration_info"] = check_model_expiration(payload.get("trained_at"), payload.get("expiration_hours", 168))

        gate = artifact_gate(payload)
        payload["gate_info"] = gate
        if gate["status"] == "FAIL" and not GATE_OVERRIDE:
            _REJECTED_CACHE[cache_key] = {"filename": payload["filename"], "gate": gate}
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] REFUSED model {found_path}: promotion gate failed "
                f"({'; '.join(gate['reasons'])}). Set ML_GATE_OVERRIDE=true to load anyway (research only).",
                file=sys.stderr,
            )
            return None

        payload["payoff_info"] = resolve_payoff_ratio(payload)
        _MODEL_CACHE[cache_key] = payload
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loaded and cached model: {found_path} "
            f"(Parity: {payload['schema_parity_ok']}, Hash: {saved_hash}, Gate: {gate['status']}, "
            f"b={payload['payoff_info']['payoff_ratio']:.3f} [{payload['payoff_info']['source']}])"
        )
        return payload
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error loading model from {found_path}: {e}", file=sys.stderr)
        return None


def fetch_recent_candles(symbol: str, timeframe: str = "1h", limit_bars: int = 250) -> np.ndarray:
    """Load OHLCV for inference.

    Crypto lives as 1m rows and is resampled. Yahoo FX/metal/stock rows are stored
    at native 1h (or 1d) so we fall back to that timeframe when 1m is empty.
    """
    norm_symbol = normalize_symbol(symbol)
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed")
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
    candidates = candle_timeframe_candidates(timeframe)
    df = pd.DataFrame()
    used_tf = timeframe
    try:
        for tf in candidates:
            mult = 60 if timeframe == "1h" and tf == "1m" else 1
            raw_limit = (limit_bars + 30) * mult
            query = (
                "SELECT timestamp, open, high, low, close, volume "
                "FROM candle WHERE symbol = %s AND timeframe = %s "
                "ORDER BY timestamp DESC LIMIT %s"
            )
            loaded = pd.read_sql(query, conn, params=(norm_symbol, tf, raw_limit))
            if len(loaded) > 0:
                df = loaded
                used_tf = tf
                break
        if len(df) == 0:
            # Legacy rows that predate the timeframe column filter
            query = (
                "SELECT timestamp, open, high, low, close, volume "
                "FROM candle WHERE symbol = %s ORDER BY timestamp DESC LIMIT %s"
            )
            df = pd.read_sql(query, conn, params=(norm_symbol, (limit_bars + 30) * 60))
            used_tf = "1m"
    finally:
        conn.close()

    if len(df) == 0:
        raise ValueError(f"No candles found for {norm_symbol}")

    df = df.iloc[::-1].copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)

    if timeframe != "1m" and used_tf == "1m":
        rule_map = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1d"}
        resample_rule = rule_map.get(timeframe, timeframe)
        df = df.resample(resample_rule).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()

    ts = (df.index.astype(int) // 10**6).to_numpy()
    candles = np.column_stack([
        ts,
        df["open"].to_numpy(),
        df["close"].to_numpy(),
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["volume"].to_numpy(),
    ])
    return candles


class PredictRequest(BaseModel):
    symbol: str = "BTC-USDT"
    timeframe: str = "1h"
    model_type: str = "lightgbm"
    threshold: Optional[float] = None       # optional absolute probability floor
    min_margin: float = 0.08
    min_ev_r: float = DEFAULT_MIN_EV_R


class MetaPredictRequest(BaseModel):
    symbol: str = "BTC-USDT"
    primary_signal: str  # "BUY" or "SELL"
    timeframe: str = "1h"
    model_type: str = "lightgbm"
    min_prob: Optional[float] = None        # optional absolute probability floor
    min_ev_r: float = DEFAULT_MIN_EV_R


def _model_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("metrics") or {}
    gate = payload.get("gate_info") or artifact_gate(payload)
    payoff = payload.get("payoff_info") or resolve_payoff_ratio(payload)
    return {
        "model": payload.get("filename", "unknown"),
        "trained_at": payload.get("trained_at"),
        "train_range": payload.get("train_range"),
        "barrier_geometry": payload.get("barrier_geometry"),
        "label_mode": payload.get("label_mode", "legacy_long_only" if payload.get("labeling_mode") == "triple_barrier" else payload.get("labeling_mode")),
        "n_trials": gate.get("n_trials"),
        "dsr": gate.get("dsr"),
        "pbo": gate.get("pbo"),
        "holdout_sharpe": payload.get("holdout_sharpe", metrics.get("holdout_sharpe")),
        "bullish_recall": metrics.get("bullish_recall"),
        "bearish_recall": metrics.get("bearish_recall"),
        "pt_mult": payload.get("pt_mult", metrics.get("pt_mult")),
        "sl_mult": payload.get("sl_mult", metrics.get("sl_mult")),
        "gate": gate,
        "payoff_ratio": payoff["payoff_ratio"],
        "payoff_ratio_source": payoff["source"],
        "payoff_ratio_theoretical": payoff["theoretical"],
        "feature_hash": payload.get("feature_schema_hash"),
        "feature_schema_parity": payload.get("schema_parity_ok", False),
        "model_expired": (payload.get("expiration_info") or {}).get("expired", False),
    }


@app.get("/health")
def health():
    available_models = []
    for sdir in MODEL_SEARCH_DIRS:
        for f in glob.glob(os.path.join(sdir, "*.joblib")):
            available_models.append(os.path.basename(f))

    return {
        "status": "ok",
        "version": app.version,
        "canonical_feature_hash": FEATURE_HASH,
        "model_dirs": MODEL_SEARCH_DIRS,
        "cached_models": list(_MODEL_CACHE.keys()),
        "rejected_models": {k: v["gate"]["reasons"] for k, v in _REJECTED_CACHE.items()},
        "available_models": sorted(set(available_models)),
        "gate_thresholds": {
            "dsr_min": DSR_MIN,
            "pbo_max": PBO_MAX,
            "min_class_recall": MIN_CLASS_RECALL,
            "override_active": GATE_OVERRIDE,
        },
        "deployed_geometry": geometry_dict(),
    }


@app.get("/model-metadata")
def model_metadata(symbol: str = "BTC-USDT", timeframe: str = "1h", model_type: str = "lightgbm"):
    """Return MLOps metadata (DSR, PBO, n_trials, geometry, payoff, gate) for a trained model artifact."""
    norm_symbol = normalize_symbol(symbol)
    meta_name = f"{norm_symbol}_{timeframe}_{model_type}_meta.json"
    for sdir in MODEL_SEARCH_DIRS:
        meta_path = os.path.join(sdir, meta_name)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            payload["status"] = "success"
            payload.setdefault("symbol", norm_symbol)
            payload.setdefault("timeframe", timeframe)
            payload.setdefault("model_type", model_type)
            payload["gate_evaluation"] = artifact_gate(payload)
            return payload
    return {"status": "error", "error": f"Metadata not found for {meta_name}"}


@app.post("/cache/clear")
def clear_cache():
    count = len(_MODEL_CACHE)
    _MODEL_CACHE.clear()
    _REJECTED_CACHE.clear()
    return {"status": "ok", "cleared_models_count": count}


@app.get("/predict")
def predict_get(
    symbol: str = Query("BTC-USDT"),
    timeframe: str = Query("1h"),
    model_type: str = Query("lightgbm"),
    threshold: Optional[float] = Query(None),
    min_margin: float = Query(0.08),
    min_ev_r: float = Query(DEFAULT_MIN_EV_R),
):
    return run_inference(symbol, timeframe, model_type, threshold, min_margin, min_ev_r)


@app.post("/predict")
def predict_post(body: PredictRequest):
    return run_inference(body.symbol, body.timeframe, body.model_type, body.threshold, body.min_margin, body.min_ev_r)


def _rejection_response(norm_symbol: str, timeframe: str, model_type: str) -> Dict[str, Any]:
    key = f"{norm_symbol}_{timeframe}_{model_type}"
    rej = _REJECTED_CACHE.get(key)
    if rej:
        return {
            "status": "error",
            "error": f"Model artifact {rej['filename']} refused: promotion gate failed ({'; '.join(rej['gate']['reasons'])})",
            "gate": rej["gate"],
        }
    return {"status": "error", "error": f"No model artifact found for {norm_symbol} ({timeframe}, {model_type})"}


@app.post("/meta-predict")
def meta_predict(body: MetaPredictRequest):
    """
    Evaluates a primary strategy directional signal using the calibrated ML model.
    Acts as the institutional meta-label filter before order dispatch.
    """
    norm_symbol = normalize_symbol(body.symbol)
    res = run_inference(norm_symbol, body.timeframe, body.model_type, body.min_prob, 0.08, body.min_ev_r)
    if res.get("status") != "success":
        return {"action": "VETO", "reason": res.get("error", "inference_failure"), "gate": res.get("gate")}

    primary = body.primary_signal.upper()
    probs = res.get("probabilities", {})
    p_bull = probs.get("bullish", 0.0)
    p_bear = probs.get("bearish", 0.0)
    uncertainty = res.get("uncertainty", "HIGH")
    b = res["kelly"]["payoff_ratio"]
    payoff_source = res["kelly"]["payoff_ratio_source"]

    if primary in ("BUY", "LONG"):
        calibrated_prob, opposing_prob = p_bull, p_bear
    elif primary in ("SELL", "SHORT"):
        calibrated_prob, opposing_prob = p_bear, p_bull
    else:
        return {"action": "VETO", "reason": f"Unknown primary signal '{primary}'"}

    ev = expected_value_r(calibrated_prob, b)
    p_be = breakeven_win_probability(b)
    min_prob = body.min_prob if body.min_prob is not None else 0.0

    if uncertainty == "HIGH":
        action, reason = "VETO", "CONFORMAL_UNCERTAINTY_HIGH"
    elif opposing_prob > calibrated_prob * DEFAULT_DOMINANCE and opposing_prob >= p_be:
        action, reason = "VETO", f"OPPOSING_DRIFT_DETECTED (p_opp={opposing_prob:.3f})"
    elif ev < body.min_ev_r or calibrated_prob < min_prob:
        action, reason = "VETO", f"CALIBRATED_EDGE_SUBPAR (p={calibrated_prob:.3f}, EV={ev:.3f}R < {body.min_ev_r}R, break-even p={p_be:.3f})"
    else:
        action, reason = "EXECUTE", "CALIBRATED_EDGE_VERIFIED"

    kelly = calculate_fractional_kelly(calibrated_prob, payoff_ratio=b, payoff_source=payoff_source)

    return {
        "action": action,
        "reason": reason,
        "primary_signal": primary,
        "calibrated_prob": round(calibrated_prob, 4),
        "opposing_prob": round(opposing_prob, 4),
        "expected_value_r": round(ev, 4),
        "breakeven_probability": round(p_be, 4),
        "uncertainty": uncertainty,
        "kelly_size_multiplier": kelly["size_multiplier"] if action == "EXECUTE" else 0.0,
        "fractional_kelly": kelly["fractional_kelly"],
        "payoff_ratio": kelly["payoff_ratio"],
        "payoff_ratio_source": payoff_source,
        "model_metadata": res.get("model_metadata"),
        "latency_ms": res.get("latency_ms", 0.0),
    }


def run_inference(
    symbol: str,
    timeframe: str = "1h",
    model_type: str = "lightgbm",
    threshold: Optional[float] = None,
    min_margin: float = 0.08,
    min_ev_r: float = DEFAULT_MIN_EV_R,
) -> Dict[str, Any]:
    t0 = time.time()
    norm_symbol = normalize_symbol(symbol)
    model_data = get_cached_model(norm_symbol, timeframe, model_type)
    if not model_data:
        return _rejection_response(norm_symbol, timeframe, model_type)

    pipeline = model_data["pipeline"]

    try:
        candles = fetch_recent_candles(norm_symbol, timeframe)
        feats = compute_latest_features(candles, fracdiff_d=resolve_artifact_d(model_data, symbol=norm_symbol))
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if np.isnan(feats).any():
        return {"status": "error", "error": "Insufficient candle history to compute complete feature vector"}

    feats_df = pd.DataFrame([feats], columns=FEATURE_NAMES)
    probs = pipeline.predict_proba(feats_df)[0]
    if len(probs) == 2:
        # Binary long meta-labeler (event_long): P(long fails), P(long works). A failed
        # long is *not* evidence for a short, so bearish is reported as 0.
        p_neutral = float(probs[0])
        p_bullish = float(probs[1])
        p_bearish = 0.0
    else:
        p_neutral = float(probs[0])
        p_bullish = float(probs[1])
        p_bearish = float(probs[2])

    payoff = model_data["payoff_info"]
    b = payoff["payoff_ratio"]
    # Decision rule uses the payoff the trainer validated the holdout with (realised avg
    # win / avg loss of training attempts); Kelly sizing uses the best empirical estimate.
    rule = ((model_data.get("metrics") or {}).get("decision_rule") or {})
    b_rule = float(rule.get("payoff_ratio_used") or b)
    uncertainty_info = calculate_conformal_uncertainty(probs)

    decision = decide_signal(
        p_neutral, p_bullish, p_bearish,
        payoff_ratio=b_rule, min_ev_r=min_ev_r, dominance=DEFAULT_DOMINANCE,
        min_probability=threshold if threshold is not None else 0.0,
    )
    raw_signal = decision["signal"]
    conf = decision["confidence"]
    gated = False
    gated_reason = None

    # Conformal Uncertainty Gate: Veto signal if margin is too low or uncertainty is HIGH
    if raw_signal != "NEUTRAL" and (uncertainty_info["margin"] < min_margin or uncertainty_info["level"] == "HIGH"):
        signal = "NEUTRAL"
        gated = True
        gated_reason = f"CONFORMAL_UNCERTAINTY_GATE (Margin: {uncertainty_info['margin']:.3f} < {min_margin})"
    else:
        signal = raw_signal

    # For NEUTRAL the Kelly block is reported for the strongest directional probability so
    # the EV/edge fields describe the actual (rejected) edge instead of a 50% placeholder.
    win_p = conf if signal in ("BUY", "SELL") else max(p_bullish, p_bearish)
    kelly_info = calculate_fractional_kelly(win_p, payoff_ratio=b, payoff_source=payoff["source"])
    if signal == "NEUTRAL":
        kelly_info["size_multiplier"] = 0.0
        kelly_info["fractional_kelly"] = 0.0

    latest_close = float(candles[-1, 2])
    summary = _model_summary(model_data)

    result = {
        "status": "success",
        "symbol": norm_symbol,
        "timeframe": timeframe,
        "signal": signal,
        "confidence": round(conf, 4),
        "raw_signal": raw_signal,
        "gated": gated,
        "gated_reason": gated_reason,
        "uncertainty": uncertainty_info["level"],
        "conformal_margin": uncertainty_info["margin"],
        "entropy": uncertainty_info["entropy"],
        "probabilities": {
            "bullish": round(p_bullish, 4),
            "bearish": round(p_bearish, 4),
            "neutral": round(p_neutral, 4),
        },
        "decision": decision,
        "kelly": kelly_info,
        "latest_close": latest_close,
        "model": summary["model"],
        "model_metadata": summary,
        "barrier_geometry": summary["barrier_geometry"],
        "n_trials": summary["n_trials"],
        "dsr": summary["dsr"],
        "pbo": summary["pbo"],
        "metrics": {
            "deflated_sharpe_ratio": summary["dsr"],
            "prob_backtest_overfitting": summary["pbo"],
            "bullish_recall": summary.get("bullish_recall"),
            "bearish_recall": summary.get("bearish_recall"),
            "pt_mult": summary.get("pt_mult"),
            "sl_mult": summary.get("sl_mult"),
            "n_trials": summary["n_trials"],
            "holdout_sharpe": summary.get("holdout_sharpe"),
        },
        "gate": summary["gate"],
        "promotion_ok": bool(summary["gate"].get("passed")),
        "promotion_reason": "; ".join(summary["gate"].get("reasons") or []) or summary["gate"].get("status"),
        "feature_schema_parity": summary["feature_schema_parity"],
        "feature_hash": summary["feature_hash"],
        "model_expired": summary["model_expired"],
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }
    return annotate_ml_prediction(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
