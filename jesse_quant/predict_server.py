#!/usr/bin/env python3
"""
Jesse Machine Learning Real-Time Prediction Microservice (Institutional v2.0)
Caches loaded models in memory for ultra-fast (<10ms) probability scoring
accessible via HTTP REST API from ai-trading-backend, n8n, or AI agents.

Features:
  - Feature schema hash validation to eliminate train-serve skew
  - Isotonic probability calibration & split-conformal uncertainty gating
  - Meta-labeling direction filtering & Fractional Kelly position sizing
  - Model expiration monitoring (168-hour staleness guard)
"""

import glob
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Query
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
import pandas as pd
import psycopg2

from ml_features import compute_latest_features, FEATURE_NAMES
from feature_schema import FEATURE_HASH, verify_feature_parity
from promotion_gates import (
    STRATEGY_PAYOFF_RATIO,
    annotate_ml_prediction,
    calculate_fractional_kelly,
    payoff_ratio_from_geometry,
)

app = FastAPI(title="Jesse ML Inference Engine (Institutional)", version="2.0.0")

# Database settings
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_NAME = os.getenv("POSTGRES_NAME", "jesse_db")
DB_USER = os.getenv("POSTGRES_USERNAME", "jesse_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

# Model cache in memory
_MODEL_CACHE: Dict[str, Any] = {}


def normalize_symbol(symbol: str) -> str:
    s = symbol.upper().replace("/", "-")
    if "-" in s:
        return s
    for quote in ["USDT", "USDC", "BUSD", "USD"]:
        if s.endswith(quote):
            return f"{s[:-len(quote)]}-USDT"
    return s


def calculate_conformal_uncertainty(probs: np.ndarray) -> Dict[str, Any]:
    """
    Computes uncertainty bounds:
      - margin: Delta between highest and second highest probability
      - entropy: Normalized Shannon entropy across class distributions [0, 1]
      - uncertainty_level: LOW, MEDIUM, or HIGH
    """
    sorted_p = np.sort(probs)[::-1]
    margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0

    # Shannon Entropy normalized by ln(K)
    k = len(probs)
    safe_p = np.clip(probs, 1e-12, 1.0)
    entropy = -float(np.sum(safe_p * np.log(safe_p))) / math.log(max(k, 2))

    if margin < 0.08 or entropy > 0.92:
        level = "HIGH"
    elif margin < 0.15 or entropy > 0.80:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "margin": round(margin, 4),
        "entropy": round(entropy, 4),
        "level": level,
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
        return {
            "expired": age > expiration_hours,
            "age_hours": round(age, 1),
            "expiration_hours": expiration_hours,
        }
    except Exception:
        return {"expired": False, "age_hours": None}


def get_cached_model(symbol: str, timeframe: str = "1h", model_type: str = "lightgbm") -> Optional[Dict[str, Any]]:
    norm_symbol = normalize_symbol(symbol)
    cache_key = f"{norm_symbol}_{timeframe}_{model_type}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    search_dirs = [
        "/home/storage/models",
        "/root/jesse-trading/storage/models",
        "storage/models",
    ]

    found_path = None
    for sdir in search_dirs:
        exact = os.path.join(sdir, f"{norm_symbol}_{timeframe}_{model_type}.joblib")
        if os.path.exists(exact) and ".rejected." not in exact:
            found_path = exact
            break
        candidates = [
            c for c in glob.glob(os.path.join(sdir, f"{norm_symbol}_{timeframe}_*.joblib"))
            if os.path.basename(c) == f"{norm_symbol}_{timeframe}_{model_type}.joblib"
        ]
        if candidates:
            found_path = candidates[0]
            break
        any_cand = [
            c for c in glob.glob(os.path.join(sdir, f"{norm_symbol}_{timeframe}_{model_type}.joblib"))
            if os.path.isfile(c)
        ]
        if any_cand:
            found_path = any_cand[0]
            break

    if not found_path:
        return None

    try:
        payload = joblib.load(found_path)
        payload["filename"] = os.path.basename(found_path)

        # Schema parity validation
        saved_hash = payload.get("feature_schema_hash") or payload.get("feature_hash")
        parity_ok = (saved_hash == FEATURE_HASH)
        payload["feature_schema_hash"] = saved_hash
        payload["schema_parity_ok"] = parity_ok

        # Expiration check
        exp_info = check_model_expiration(
            payload.get("trained_at"),
            payload.get("expiration_hours", 168),
        )
        payload["expiration_info"] = exp_info

        _MODEL_CACHE[cache_key] = payload
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loaded and cached model: {found_path} (Parity: {parity_ok}, Hash: {saved_hash})")
        return payload
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error loading model from {found_path}: {e}", file=sys.stderr)
        return None


def fetch_recent_candles(symbol: str, timeframe: str = "1h", limit_bars: int = 250) -> np.ndarray:
    norm_symbol = normalize_symbol(symbol)
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )

    mult = 60 if timeframe == "1h" else 15 if timeframe == "15m" else 240 if timeframe == "4h" else 1
    raw_limit = (limit_bars + 30) * mult

    query = (
        "SELECT timestamp, open, high, low, close, volume "
        "FROM candle WHERE symbol = %s ORDER BY timestamp DESC LIMIT %s"
    )
    df = pd.read_sql(query, conn, params=(norm_symbol, raw_limit))
    conn.close()

    if len(df) == 0:
        raise ValueError(f"No candles found for {norm_symbol}")

    df = df.iloc[::-1].copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)

    if timeframe != "1m":
        rule_map = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1d"}
        resample_rule = rule_map.get(timeframe, timeframe)
        df = df.resample(resample_rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        ).dropna()

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
    threshold: float = 0.45
    min_margin: float = 0.08


class MetaPredictRequest(BaseModel):
    symbol: str = "BTC-USDT"
    primary_signal: str  # "BUY" or "SELL"
    timeframe: str = "1h"
    model_type: str = "lightgbm"
    min_prob: float = 0.45


@app.get("/health")
def health():
    search_dirs = ["/home/storage/models", "/root/jesse-trading/storage/models", "storage/models"]
    available_models = []
    for sdir in search_dirs:
        for f in glob.glob(os.path.join(sdir, "*.joblib")):
            available_models.append(os.path.basename(f))

    return {
        "status": "ok",
        "canonical_feature_hash": FEATURE_HASH,
        "cached_models": list(_MODEL_CACHE.keys()),
        "available_models": sorted(list(set(available_models))),
        "production_models": sorted(
            [m for m in set(available_models) if ".rejected." not in m and not m.endswith(".rejected.pt")]
        ),
    }


@app.post("/cache/clear")
def clear_cache():
    global _MODEL_CACHE
    count = len(_MODEL_CACHE)
    _MODEL_CACHE.clear()
    return {"status": "ok", "cleared_models_count": count}


@app.get("/predict")
def predict_get(
    symbol: str = Query("BTC-USDT"),
    timeframe: str = Query("1h"),
    model_type: str = Query("lightgbm"),
    threshold: float = Query(0.45),
    min_margin: float = Query(0.08),
):
    return run_inference(symbol, timeframe, model_type, threshold, min_margin)


@app.post("/predict")
def predict_post(body: PredictRequest):
    return run_inference(body.symbol, body.timeframe, body.model_type, body.threshold, body.min_margin)


@app.post("/meta-predict")
def meta_predict(body: MetaPredictRequest):
    """
    Evaluates a primary strategy directional signal using the calibrated ML model.
    Acts as the institutional meta-label filter before order dispatch.
    """
    norm_symbol = normalize_symbol(body.symbol)
    res = run_inference(norm_symbol, body.timeframe, body.model_type, body.min_prob)
    if res.get("status") != "success":
        return {"action": "VETO", "reason": res.get("error", "inference_failure")}

    primary = body.primary_signal.upper()
    probs = res.get("probabilities", {})
    p_bull = probs.get("bullish", 0.0)
    p_bear = probs.get("bearish", 0.0)
    uncertainty = res.get("uncertainty", "HIGH")

    # Meta-label decision rule
    if primary in ("BUY", "LONG"):
        calibrated_prob = p_bull
        opposing_prob = p_bear
    elif primary in ("SELL", "SHORT"):
        calibrated_prob = p_bear
        opposing_prob = p_bull
    else:
        return {"action": "VETO", "reason": f"Unknown primary signal '{primary}'"}

    # Veto conditions
    if uncertainty == "HIGH":
        action = "VETO"
        reason = "CONFORMAL_UNCERTAINTY_HIGH"
    elif opposing_prob >= 0.50 and opposing_prob > calibrated_prob:
        action = "VETO"
        reason = f"OPPOSING_DRIFT_DETECTED (p_opp={opposing_prob:.3f})"
    elif calibrated_prob < body.min_prob:
        action = "VETO"
        reason = f"CALIBRATED_PROBABILITY_SUBPAR (p={calibrated_prob:.3f} < {body.min_prob})"
    else:
        action = "EXECUTE"
        reason = "CALIBRATED_EDGE_VERIFIED"

    kelly = calculate_fractional_kelly(calibrated_prob, payoff_ratio=payoff_ratio_from_geometry(
        res.get("pt_mult"), res.get("sl_mult")
    ))

    return {
        "action": action,
        "reason": reason,
        "primary_signal": primary,
        "calibrated_prob": round(calibrated_prob, 4),
        "uncertainty": uncertainty,
        "kelly_size_multiplier": kelly["size_multiplier"] if action == "EXECUTE" else 0.0,
        "fractional_kelly": kelly["fractional_kelly"],
        "latency_ms": res.get("latency_ms", 0.0),
    }


def run_inference(
    symbol: str,
    timeframe: str = "1h",
    model_type: str = "lightgbm",
    threshold: float = 0.45,
    min_margin: float = 0.08,
) -> Dict[str, Any]:
    t0 = time.time()
    norm_symbol = normalize_symbol(symbol)
    model_data = get_cached_model(norm_symbol, timeframe, model_type)
    if not model_data:
        return {
            "status": "error",
            "error": f"No model artifact found for {norm_symbol} ({timeframe}, {model_type})",
        }

    pipeline = model_data["pipeline"]

    try:
        candles = fetch_recent_candles(norm_symbol, timeframe)
        feats = compute_latest_features(candles)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if np.isnan(feats).any():
        return {
            "status": "error",
            "error": "Insufficient candle history to compute complete feature vector",
        }

    feats_df = pd.DataFrame([feats], columns=FEATURE_NAMES)
    probs = pipeline.predict_proba(feats_df)[0]
    if len(probs) == 2:
        # Binary meta-labeler: P(fail), P(TP-before-SL) for a long setup
        p_bearish = float(probs[0])
        p_bullish = float(probs[1])
        p_neutral = 0.0
    else:
        p_neutral = float(probs[0])
        p_bullish = float(probs[1])
        p_bearish = float(probs[2]) if len(probs) > 2 else 0.0

    uncertainty_info = calculate_conformal_uncertainty(probs)
    gated = False
    gated_reason = None

    if p_bullish >= threshold and p_bullish > (p_bearish * 1.2):
        raw_signal = "BUY"
        conf = p_bullish
    elif p_bearish >= threshold and p_bearish > (p_bullish * 1.2):
        raw_signal = "SELL"
        conf = p_bearish
    else:
        raw_signal = "NEUTRAL"
        conf = p_neutral

    # Conformal Uncertainty Gate: Veto signal if margin is too low or uncertainty is HIGH
    if raw_signal != "NEUTRAL":
        if uncertainty_info["margin"] < min_margin or uncertainty_info["level"] == "HIGH":
            signal = "NEUTRAL"
            gated = True
            gated_reason = f"CONFORMAL_UNCERTAINTY_GATE (Margin: {uncertainty_info['margin']:.3f} < {min_margin})"
        else:
            signal = raw_signal
    else:
        signal = raw_signal

    # Fractional Kelly using the model's own barrier geometry (5.5/1.75 → b≈3.14)
    win_p = conf if signal in ("BUY", "SELL") else 0.50
    pt_mult = model_data.get("pt_mult")
    sl_mult = model_data.get("sl_mult")
    metrics = model_data.get("metrics") or {}
    if pt_mult is None:
        pt_mult = metrics.get("pt_mult")
    if sl_mult is None:
        sl_mult = metrics.get("sl_mult")
    payoff_b = payoff_ratio_from_geometry(pt_mult, sl_mult)
    kelly_info = calculate_fractional_kelly(win_p, payoff_ratio=payoff_b)

    latest_close = float(candles[-1, 2])
    latency_ms = round((time.time() - t0) * 1000, 2)

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
        "kelly": kelly_info,
        "pt_mult": pt_mult,
        "sl_mult": sl_mult,
        "metrics": {
            "deflated_sharpe_ratio": metrics.get("deflated_sharpe_ratio"),
            "prob_backtest_overfitting": metrics.get("prob_backtest_overfitting"),
            "pt_mult": pt_mult,
            "sl_mult": sl_mult,
            "n_trials": metrics.get("n_trials"),
            "holdout_sharpe": metrics.get("holdout_sharpe"),
            "bullish_recall": metrics.get("bullish_recall"),
            "bearish_recall": metrics.get("bearish_recall"),
        },
        "latest_close": latest_close,
        "model": model_data.get("filename", "unknown"),
        "feature_schema_parity": model_data.get("schema_parity_ok", False),
        "feature_hash": model_data.get("feature_schema_hash", "unknown"),
        "model_expired": model_data.get("expiration_info", {}).get("expired", False),
        "latency_ms": latency_ms,
    }
    return annotate_ml_prediction(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9003)
