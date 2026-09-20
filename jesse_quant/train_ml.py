#!/usr/bin/env python3
"""
Jesse Machine Learning Quant Model Trainer
Extracts historical candles from PostgreSQL, computes non-linear predictive features,
labels forward returns, runs Purged Time-Series Cross-Validation, and serializes
the trained model pipeline to storage/models/.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # type: ignore[assignment]

from calibration import calibration_deploy_control, reliability_diagram
from monte_carlo import moving_block_bootstrap
from embargo_audit import (
    audit_purged_embargo,
    embargo_bars_for_n,
    longest_feature_lookback_bars,
)
from fracdiff import kernel_width, select_fracdiff_d
from ml_features import FEATURE_NAMES, LONGEST_FEATURE_LOOKBACK_BARS, compute_features_df
from validation_metrics import PurgedKFold, calculate_sharpe_ratio, deflated_sharpe_ratio, probability_of_backtest_overfitting
from feature_schema import FEATURE_HASH
from gpu_device import detect_training_device, lightgbm_device_kwargs
from promotion_contract import (
    SPEC_VERSION,
    default_geometry,
    evaluate_contract,
    write_geometry,
)
from trial_registry import record_trials
from asset_universe import candle_timeframe_candidates
from train_errors import TooFewEventsError
from barrier_config import (
    MAX_HOLDING_BARS,
    TRAIL_ACTIVATION_ATR,
    TRAIL_ATR_MULT,
    cost_model_dict,
    round_trip_cost_pct,
)
from promotion_gates import (
    DSR_GATE,
    PBO_GATE,
    STRATEGY_PT_ATR,
    STRATEGY_SL_ATR,
    evaluate_promotion,
    net_theoretical_payoff_ratio,
)
from triple_barrier import apply_triple_barrier, compute_sample_uniqueness, realized_payoff_stats

# Database settings — never hardcode credentials; the Jesse container injects these.
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_NAME = os.getenv("POSTGRES_NAME", "jesse_db")
DB_USER = os.getenv("POSTGRES_USERNAME", "jesse_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

MODELS_DIR = os.getenv("JESSE_MODELS_DIR", "")
if not MODELS_DIR:
    MODELS_DIR = "/root/jesse-trading/storage/models"
if not os.path.exists(MODELS_DIR):
    if os.path.exists("/home/storage"):
        MODELS_DIR = "/home/storage/models"
    else:
        MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_candles_from_path(path: str, symbol: str = "BTC-USDT", timeframe: str = "1h") -> pd.DataFrame:
    """Load OHLCV from parquet/csv (gzip ok) produced by a Jesse candle dump."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, compression="infer")
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp column")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    print(f"    Loaded {len(df):,} rows from {path} for {symbol}")
    if timeframe != "1m":
        return _resample_ohlcv(df, timeframe)
    return df


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule_map = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1d"}
    resample_rule = rule_map.get(timeframe, timeframe)
    df_resampled = df.resample(resample_rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()
    print(f"    Resampled to {len(df_resampled):,} {timeframe} candles ({df_resampled.index.min().date()} to {df_resampled.index.max().date()})")
    return df_resampled


def load_candles_from_db(symbol: str = "BTC-USDT", timeframe: str = "1h") -> pd.DataFrame:
    """Load 1m candles from PostgreSQL and resample to target timeframe."""
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed — pass --candles PATH on the GPU node")
    if not DB_PASS:
        raise RuntimeError("POSTGRES_PASSWORD is not set — refusing to connect")
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )

    query = (
        "SELECT timestamp, open, high, low, close, volume "
        "FROM candle WHERE symbol = %s AND timeframe = %s ORDER BY timestamp ASC"
    )
    t0 = time.time()
    df = pd.DataFrame()
    used_tf = "1m"
    for tf in candle_timeframe_candidates(timeframe):
        loaded = pd.read_sql(query, conn, params=(symbol, tf))
        if len(loaded) > 0:
            df = loaded
            used_tf = tf
            break
    if len(df) == 0:
        # Legacy rows
        df = pd.read_sql(
            "SELECT timestamp, open, high, low, close, volume FROM candle WHERE symbol = %s ORDER BY timestamp ASC",
            conn,
            params=(symbol,),
        )
        used_tf = "1m"
    conn.close()
    print(f"    Loaded {len(df):,} {used_tf} candles for {symbol} in {time.time()-t0:.2f}s")

    if len(df) == 0:
        raise ValueError(f"No candles found in database for symbol {symbol}")

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)

    if timeframe != "1m" and used_tf == "1m":
        return _resample_ohlcv(df, timeframe)
    return df


def quantum_ai_event_index(df: pd.DataFrame) -> pd.Index:
    """Bars where QuantumAIStrategy's long filter would fire (event-sampled labels)."""
    close = df["close"]
    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_mid = close.ewm(span=50, adjust=False).mean()
    ema_slow = close.ewm(span=200, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rsi = 100.0 - (100.0 / (1.0 + avg_gain / (avg_loss + 1e-12)))
    bull_regime = (close > ema_slow) & (ema_mid > ema_slow)
    pullback = (close <= ema_fast * 1.005) & (close >= ema_mid * 0.99)
    rsi_dip = (rsi >= 38.0) & (rsi <= 54.0)
    candle_bull = close > df["open"]
    mask = bull_regime & pullback & rsi_dip & candle_bull
    return df.index[mask]


def prepare_dataset(
    df: pd.DataFrame,
    labeling_mode: str = "triple_barrier",
    pt_mult: float = STRATEGY_PT_ATR,
    sl_mult: float = STRATEGY_SL_ATR,
    max_holding: int = MAX_HOLDING_BARS,
    forward_horizon: int = 6,
    threshold_pct: float = 0.75,
    fallback_every_bar: bool = True,
    min_events: int = 50,
    fracdiff_d: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series], Optional[pd.Series], pd.Series]:
    """
    Computes features and labels outcomes using either:
    1. Triple-Barrier Method (path-dependent PT, SL, timeout with sample uniqueness weights)
    2. Fixed Horizon returns (return[t+H] vs fixed threshold)

    Meta-labels are P(net outcome > 0 | primary signal) after fees/slip/funding,
    not the theoretical b≈3.14 TP-before-SL event.
    """
    print("[*] Computing quantitative technical indicators & FracDiff features...")
    t0 = time.time()
    X = compute_features_df(df, fracdiff_d=fracdiff_d)

    if labeling_mode == "triple_barrier":
        events_idx = quantum_ai_event_index(df)
        print(
            f"[*] Applying Triple-Barrier Method on {len(events_idx):,} QuantumAI entry events: "
            f"PT={pt_mult}x ATR, SL={sl_mult}x ATR, trail={TRAIL_ACTIVATION_ATR}/{TRAIL_ATR_MULT}, "
            f"Max Holding={max_holding} bars, net of costs..."
        )
        if len(events_idx) < min_events:
            if not fallback_every_bar:
                raise TooFewEventsError(len(events_idx), min_events)
            print("[!] Too few strategy events; falling back to every-bar labeling")
            events_idx = None
        tb_df = apply_triple_barrier(
            df,
            events_idx=events_idx,
            pt_multiplier=pt_mult,
            sl_multiplier=sl_mult,
            max_holding_bars=max_holding,
            trail_activation_atr=TRAIL_ACTIVATION_ATR,
            trail_atr_mult=TRAIL_ATR_MULT,
            apply_costs=True,
        )
        sample_weights = compute_sample_uniqueness(tb_df, df.index)

        net = tb_df["net_ret"] if "net_ret" in tb_df.columns else tb_df["ret"]
        y = pd.Series((net > 0).astype(int), index=tb_df.index, dtype=int)

        samples_info_sets = tb_df["t1"]
        common_idx = X.index.intersection(y.index)
        valid_mask = ~(X.loc[common_idx].isna().any(axis=1) | y.loc[common_idx].isna() | net.loc[common_idx].isna())
        final_idx = common_idx[valid_mask]

        X_clean = X.loc[final_idx]
        y_clean = y.loc[final_idx]
        w_clean = sample_weights.loc[final_idx]
        info_sets_clean = samples_info_sets.loc[final_idx]
        net_clean = net.loc[final_idx]
    else:
        # Fixed forward horizon
        forward_return = (df["close"].shift(-forward_horizon) / df["close"] - 1.0) * 100.0
        y = pd.Series(0, index=df.index, dtype=int)
        y[forward_return > threshold_pct] = 1
        y[forward_return < -threshold_pct] = 2

        valid_mask = ~(X.isna().any(axis=1) | forward_return.isna())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        w_clean = None
        info_sets_clean = None
        net_clean = forward_return[valid_mask]

    class_counts = y_clean.value_counts().to_dict()
    print(f"    Dataset prepared in {time.time()-t0:.2f}s: {len(X_clean):,} valid rows")
    print(f"    Class Distribution: Net-loss(0)={class_counts.get(0, 0):,}, Net-win(1)={class_counts.get(1, 0):,}, Bearish(2)={class_counts.get(2, 0):,}")
    return X_clean, y_clean, w_clean, info_sets_clean, net_clean



LGBM_GRID = [
    {"n_estimators": 80, "max_depth": 3, "num_leaves": 8, "learning_rate": 0.03,
     "min_child_samples": 80, "reg_lambda": 2.0, "colsample_bytree": 0.6, "subsample": 0.7},
    {"n_estimators": 100, "max_depth": 4, "num_leaves": 12, "learning_rate": 0.03,
     "min_child_samples": 60, "reg_lambda": 1.5, "colsample_bytree": 0.7, "subsample": 0.8},
    {"n_estimators": 120, "max_depth": 4, "num_leaves": 16, "learning_rate": 0.02,
     "min_child_samples": 50, "reg_lambda": 1.0, "colsample_bytree": 0.7, "subsample": 0.8},
    {"n_estimators": 80, "max_depth": 2, "num_leaves": 4, "learning_rate": 0.05,
     "min_child_samples": 120, "reg_lambda": 4.0, "colsample_bytree": 0.5, "subsample": 0.6},
    {"n_estimators": 140, "max_depth": 5, "num_leaves": 20, "learning_rate": 0.02,
     "min_child_samples": 50, "reg_lambda": 1.5, "colsample_bytree": 0.75, "subsample": 0.85},
    {"n_estimators": 160, "max_depth": 3, "num_leaves": 8, "learning_rate": 0.04,
     "min_child_samples": 100, "reg_lambda": 3.0, "colsample_bytree": 0.55, "subsample": 0.7},
]


def _load_trial_ledger() -> int:
    ledger_path = os.path.join(MODELS_DIR, "_trial_ledger.json")
    if not os.path.exists(ledger_path):
        return 0
    try:
        with open(ledger_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return int(data.get("cumulative_trials", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0


def _save_trial_ledger(cumulative_trials: int) -> None:
    ledger_path = os.path.join(MODELS_DIR, "_trial_ledger.json")
    with open(ledger_path, "w", encoding="utf-8") as fh:
        json.dump({"cumulative_trials": int(cumulative_trials)}, fh)


def _make_clf(model_type: str, params: Optional[Dict[str, Any]] = None, device_kwargs: Optional[Dict[str, Any]] = None):
    if model_type == "lightgbm":
        cfg = {
            "n_estimators": 120,
            "learning_rate": 0.03,
            "max_depth": 4,
            "num_leaves": 16,
            "min_child_samples": 60,
            "reg_lambda": 1.5,
            "colsample_bytree": 0.7,
            "subsample": 0.8,
            "bagging_freq": 1,
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1,
        }
        if device_kwargs:
            cfg.update(device_kwargs)
        if params:
            cfg.update(params)
        return LGBMClassifier(**cfg)
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.03, max_depth=5, class_weight="balanced", random_state=42,
    )


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[pd.Series] = None,
    samples_info_sets: Optional[pd.Series] = None,
    model_type: str = "lightgbm",
    test_size: float = 0.20,
    pt_mult: float = STRATEGY_PT_ATR,
    sl_mult: float = STRATEGY_SL_ATR,
    symbol: str = "BTC-USDT",
    timeframe: str = "1h",
    max_holding: int = MAX_HOLDING_BARS,
    fracdiff_d: Optional[float] = None,
    fracdiff_meta: Optional[Dict[str, Any]] = None,
    net_returns: Optional[pd.Series] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Train with a hyperparameter grid so DSR/PBO see the true trial count."""
    device = detect_training_device()
    device_kwargs = lightgbm_device_kwargs(device) if model_type == "lightgbm" else {}
    print(f"[*] Training device: {device.kind} ({device.name}) lightgbm={device.lightgbm_device}")

    split_idx = int(len(X) * (1.0 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    w_train = sample_weights.iloc[:split_idx].to_numpy() if sample_weights is not None else None

    print(f"\n[*] Training Window: {len(X_train):,} bars | Holdout Test Window: {len(X_test):,} bars")

    lookback = longest_feature_lookback_bars(kernel_width(float(fracdiff_d)) if fracdiff_d is not None else 0)
    lookback = max(lookback, LONGEST_FEATURE_LOOKBACK_BARS)
    embargo_bars = embargo_bars_for_n(
        len(X),
        max_holding_bars=int(max_holding),
        longest_lookback_bars=lookback,
    )
    purge_horizon = int(max_holding)
    embargo_audit = audit_purged_embargo(
        embargo_bars=embargo_bars,
        purge_horizon_bars=purge_horizon,
        max_holding_bars=int(max_holding),
        longest_lookback_bars=lookback,
        timeframe=timeframe,
    )
    print(
        f"[*] Embargo audit: embargo={embargo_bars} purge={purge_horizon} "
        f"required={embargo_audit.required_embargo_bars} ok={embargo_audit.ok}"
    )
    if not embargo_audit.ok:
        raise RuntimeError("Purged-CV embargo audit failed: " + "; ".join(embargo_audit.reasons))

    grid = LGBM_GRID if model_type == "lightgbm" else [{}]
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    holdout_columns = []
    trial_sharpes = []
    fitted = []
    used_device_kwargs = dict(device_kwargs)

    print(f"[*] Evaluating {len(grid)} candidate configurations for CPCV/PBO matrix...")
    for i, params in enumerate(grid, 1):
        try:
            clf = _make_clf(model_type, params, used_device_kwargs)
            if w_train is not None and model_type in ("lightgbm", "random_forest"):
                clf.fit(X_train_s, y_train, sample_weight=w_train)
            else:
                clf.fit(X_train_s, y_train)
        except Exception as exc:
            if used_device_kwargs:
                print(f"[!] GPU LightGBM failed ({exc}); falling back to CPU for remaining trials")
                used_device_kwargs = {}
                clf = _make_clf(model_type, params, used_device_kwargs)
                if w_train is not None and model_type in ("lightgbm", "random_forest"):
                    clf.fit(X_train_s, y_train, sample_weight=w_train)
                else:
                    clf.fit(X_train_s, y_train)
            else:
                raise
        preds = clf.predict(X_test_s)
        if net_returns is not None:
            holdout_net = net_returns.iloc[split_idx:].to_numpy(dtype=float) / 100.0
            rets = np.where(preds == 1, holdout_net, 0.0)
        else:
            pred_signal = np.where(preds == 1, 1.0, -1.0)
            actual_signal = np.where(y_test.to_numpy() == 1, 1.0, -1.0)
            rets = pred_signal * actual_signal * 0.01
        holdout_columns.append(rets)
        sr = calculate_sharpe_ratio(rets)
        trial_sharpes.append(sr)
        fitted.append((params, clf, sr, preds))
        print(f"    trial {i}/{len(grid)} Sharpe={sr:.3f} params={params}")

    trial_matrix = np.column_stack(holdout_columns)
    pbo_cv, med_rank, pbo_ranks = probability_of_backtest_overfitting(
        trial_matrix, n_blocks=min(16, max(4, (trial_matrix.shape[0] // 20) * 2))
    )
    if not pbo_ranks:
        # Unevaluable CSCV (too few periods) must not silently pass the 0.0 default.
        pbo_cv = 1.0
        med_rank = 1.0

    historical = _load_trial_ledger()
    scope = f"ml:{symbol}:{timeframe}"
    n_trials = record_trials(scope, len(grid), note="lightgbm_grid")
    n_trials = max(n_trials, historical + len(grid))
    _save_trial_ledger(n_trials)

    var_sharpe = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else None
    best_idx = int(np.argmax(trial_sharpes))
    best_params, _, best_sr, best_preds = fitted[best_idx]
    holdout_returns = holdout_columns[best_idx]
    holdout_sr = calculate_sharpe_ratio(holdout_returns)
    dsr_holdout = deflated_sharpe_ratio(holdout_returns, n_trials=n_trials, variance_of_trials=var_sharpe)

    print(f"[*] Best trial #{best_idx + 1} holdout Sharpe={holdout_sr:.3f}; n_trials={n_trials} (ledger+grid)")
    monte = moving_block_bootstrap(holdout_returns)
    print(f"[*] Monte Carlo: {monte.get('reason')} scenarios={monte.get('n_scenarios')}")
    if net_returns is not None:
        payoff = realized_payoff_stats(net_returns.to_numpy(dtype=float))
    else:
        payoff = realized_payoff_stats(holdout_returns * 100.0)
    fallback_b = net_theoretical_payoff_ratio(pt_mult, sl_mult, holding_bars=int(max_holding))
    if payoff["n_wins"] >= 30 and payoff["avg_loss_abs"] > 0:
        b_hat = float(payoff["payoff_ratio"])
        payoff_kind = "net_realized"
    else:
        b_hat = float(fallback_b)
        payoff_kind = "net_theoretical_geometry"

    base_clf = _make_clf(model_type, best_params, used_device_kwargs)
    info_train = None
    if samples_info_sets is not None:
        info_train = samples_info_sets.iloc[:split_idx]
    purged_cv = PurgedKFold(
        n_splits=3,
        samples_info_sets=info_train,
        embargo_bars=embargo_bars,
    )
    calibrated_clf = None
    calibration_method = "isotonic"
    print(f"[*] Fitting final {model_type} pipeline with CalibratedClassifierCV + PurgedKFold...")
    for method in ("isotonic", "sigmoid"):
        try:
            candidate = CalibratedClassifierCV(estimator=base_clf, method=method, cv=purged_cv)
            if w_train is not None and model_type in ("lightgbm", "random_forest"):
                candidate.fit(X_train_s, y_train, sample_weight=w_train)
            else:
                candidate.fit(X_train_s, y_train)
            calibrated_clf = candidate
            calibration_method = method
            break
        except Exception as exc:
            print(f"[!] CalibratedClassifierCV({method}) with PurgedKFold failed: {exc}")
    if calibrated_clf is None:
        raise RuntimeError(
            "CalibratedClassifierCV failed under PurgedKFold — refusing shuffled KFold (leakage)"
        )

    pipeline = Pipeline([
        ("scaler", scaler),
        ("classifier", calibrated_clf),
    ])

    test_preds = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
    test_prob = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_test)
        # Binary meta-labeler: column 1 is P(TP-before-SL).
        test_prob = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    reliability = reliability_diagram(y_test.to_numpy(), test_prob if test_prob is not None else [])
    calibration_ctl = calibration_deploy_control(reliability)

    importances = {}
    first_est = getattr(calibrated_clf.calibrated_classifiers_[0], "estimator", None)
    if first_est and hasattr(first_est, "feature_importances_"):
        for name, imp in zip(FEATURE_NAMES, first_est.feature_importances_):
            importances[name] = float(imp)
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    metrics = {
        "model_type": model_type,
        "cv_mean_accuracy": float(test_acc),
        "test_accuracy": float(test_acc),
        "deflated_sharpe_ratio": float(dsr_holdout),
        "prob_backtest_overfitting": float(pbo_cv),
        "holdout_sharpe": float(holdout_sr),
        "n_trials": int(n_trials),
        "n_grid": int(len(grid)),
        "best_params": best_params,
        "pt_mult": float(pt_mult),
        "sl_mult": float(sl_mult),
        "payoff_ratio": float(b_hat),
        "pbo_median_rank": float(med_rank),
        "bullish_precision": float(report.get("1", {}).get("precision", 0)),
        "bullish_recall": float(report.get("1", {}).get("recall", 0)),
        "bearish_precision": float(report.get("0", report.get("2", {})).get("precision", 0)),
        "bearish_recall": float(report.get("0", report.get("2", {})).get("recall", 0)),
        "feature_importances": importances,
        "n_events": int(len(X)),
        "spec_version": SPEC_VERSION,
        "feature_schema_hash": FEATURE_HASH,
        "used_raw_n_as_effective": False,
        "device": device.to_dict(),
        "lightgbm_device": used_device_kwargs.get("device", "cpu"),
        "fracdiff_d": None if fracdiff_d is None else float(fracdiff_d),
        "fracdiff": fracdiff_meta or {},
        "embargo_audit": embargo_audit.to_dict(),
        "calibration": calibration_ctl,
        "calibration_method": calibration_method,
        "payoff_ratio_kind": payoff_kind,
        "realized_payoff": payoff,
        "monte_carlo": monte,
        "max_holding_bars": int(max_holding),
        "trail_activation_atr": float(TRAIL_ACTIVATION_ATR),
        "trail_atr_mult": float(TRAIL_ATR_MULT),
        "costs": cost_model_dict(),
        "round_trip_cost_pct": round_trip_cost_pct(int(max_holding)),
    }

    promotion = evaluate_promotion(metrics, pt_mult=pt_mult, sl_mult=sl_mult)
    metrics["promotion_ok"] = promotion.ok
    metrics["promotion_reason"] = promotion.reason

    print("\n=========================================================")
    print("        HOLDOUT OUT-OF-SAMPLE TEST EVALUATION            ")
    print("=========================================================")
    print(f"Model:                     {model_type.upper()} (Isotonic Calibrated)")
    print(f"Overall Accuracy:          {test_acc:.2%}")
    print(f"Holdout Sharpe Ratio:      {holdout_sr:.2f}")
    print(f"Deflated Sharpe Ratio:     {dsr_holdout:.4f} (n_trials={n_trials}, gate > {DSR_GATE})")
    print(f"Prob. of Overfitting (PBO):{pbo_cv:.2%} (gate < {PBO_GATE:.0%})")
    print(f"Promotion:                 {'PASS' if promotion.ok else 'BLOCKED'} — {promotion.reason}")
    print(f"Bullish Precision:         {metrics['bullish_precision']:.2%} (Recall: {metrics['bullish_recall']:.2%})")
    print(f"Bearish Precision:         {metrics['bearish_precision']:.2%} (Recall: {metrics['bearish_recall']:.2%})")
    print("---------------------------------------------------------")
    print("TOP PREDICTIVE FEATURES:")
    for idx, (feat, score) in enumerate(list(importances.items())[:8], 1):
        print(f"  {idx}. {feat:<20} : {score:.4f}")
    print("=========================================================")

    return pipeline, metrics


def main():
    parser = argparse.ArgumentParser(description="Jesse Machine Learning Quant Trainer")
    parser.add_argument("--symbol", default="BTC-USDT", help="Pair symbol (default: BTC-USDT)")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe (default: 1h)")
    parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "hist_gradient_boosting", "random_forest"])
    parser.add_argument("--labeling", default="triple_barrier", choices=["triple_barrier", "fixed"], help="Labeling methodology")
    parser.add_argument("--pt-mult", type=float, default=STRATEGY_PT_ATR, help="Triple-barrier profit target ATR multiplier (live geometry 5.5)")
    parser.add_argument("--sl-mult", type=float, default=STRATEGY_SL_ATR, help="Triple-barrier stop loss ATR multiplier (live geometry 1.75)")
    parser.add_argument("--holding", type=int, default=MAX_HOLDING_BARS, help="Triple-barrier maximum holding period in bars (live 1h = 48)")
    parser.add_argument("--horizon", type=int, default=6, help="Fixed return forward horizon in bars (default: 6)")
    parser.add_argument("--threshold", type=float, default=0.75, help="Fixed return threshold pct (default: 0.75)")
    parser.add_argument("--candles", default="", help="Parquet/CSV path (GPU node). If empty, load from Postgres.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Training device preference")
    parser.add_argument("--force-promote", action="store_true", help="Overwrite production artifact even if DSR/PBO gates fail")
    args = parser.parse_args()
    if args.device != "auto":
        os.environ["JESSE_TRAIN_DEVICE"] = args.device

    print("=========================================================")
    print("         JESSE QUANT MACHINE LEARNING TRAINER            ")
    print("=========================================================")
    print(f"Symbol:        {args.symbol}")
    print(f"Timeframe:     {args.timeframe}")
    print(f"Model Engine:  {args.model.upper()} (Isotonic Calibrated)")
    print(f"Labeling Mode: {args.labeling.upper()} (PT={args.pt_mult}x, SL={args.sl_mult}x, MaxHold={args.holding}b)")
    print(f"Feature Hash:  {FEATURE_HASH}")
    print("=========================================================")

    if args.candles:
        df = load_candles_from_path(args.candles, args.symbol, args.timeframe)
    else:
        df = load_candles_from_db(args.symbol, args.timeframe)
    fracdiff_meta = select_fracdiff_d(df["close"])
    fracdiff_d = float(fracdiff_meta["d"])
    print(f"[*] FracDiff d={fracdiff_d} adf_p={fracdiff_meta['adf_pvalue']:.4f} source={fracdiff_meta['source']}")
    X, y, sample_weights, samples_info_sets, net_returns = prepare_dataset(
        df,
        labeling_mode=args.labeling,
        pt_mult=args.pt_mult,
        sl_mult=args.sl_mult,
        max_holding=args.holding,
        forward_horizon=args.horizon,
        threshold_pct=args.threshold,
        fracdiff_d=fracdiff_d,
    )
    pipeline, metrics = train_model(
        X,
        y,
        sample_weights=sample_weights,
        samples_info_sets=samples_info_sets,
        model_type=args.model,
        pt_mult=args.pt_mult,
        sl_mult=args.sl_mult,
        symbol=args.symbol,
        timeframe=args.timeframe,
        max_holding=args.holding,
        fracdiff_d=fracdiff_d,
        fracdiff_meta=fracdiff_meta,
        net_returns=net_returns,
    )

    # Save model artifact with MLOps metadata
    norm_symbol = args.symbol.replace("/", "-")
    model_filename = f"{norm_symbol}_{args.timeframe}_{args.model}.joblib"
    model_path = os.path.join(MODELS_DIR, model_filename)
    promoted = bool(metrics.get("promotion_ok")) or args.force_promote
    if not promoted:
        model_filename = f"{norm_symbol}_{args.timeframe}_{args.model}.rejected.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        print(f"\n[!] Promotion BLOCKED: {metrics.get('promotion_reason')}")
        print("[!] Writing rejected artifact only — production model left unchanged")

    save_payload = {
        "pipeline": pipeline,
        "feature_names": FEATURE_NAMES,
        "feature_hash": FEATURE_HASH,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "model_type": args.model,
        "labeling_mode": args.labeling,
        "pt_mult": args.pt_mult,
        "sl_mult": args.sl_mult,
        "calibrated": True,
        "fracdiff_d": fracdiff_d,
        "fracdiff_d_by_symbol": {args.symbol: fracdiff_d},
        "expiration_hours": 168,
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "promotion_ok": promoted,
        "promotion_reason": metrics.get("promotion_reason"),
    }

    joblib.dump(save_payload, model_path)
    print(f"\n[{'✓' if promoted else '!'}] Model pipeline saved to: {model_path}")

    run_dir = os.path.join(MODELS_DIR, "runs", datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + f"_{norm_symbol}_{args.timeframe}_{args.model}")
    os.makedirs(run_dir, exist_ok=True)
    geo = default_geometry()
    write_geometry(os.path.join(run_dir, "geometry.json"), geo)
    metrics_payload = {
        "spec_version": SPEC_VERSION,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "model_type": args.model,
        "feature_schema_hash": FEATURE_HASH,
        "pt_mult": args.pt_mult,
        "sl_mult": args.sl_mult,
        **metrics,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2, default=str)
    contract = evaluate_contract(
        geo,
        metrics_payload,
        live_geometry=geo,
        promote_requested=False,
        holdout_spent=False,
        holdout_pass=False,
    )
    with open(os.path.join(run_dir, "contract.json"), "w", encoding="utf-8") as fh:
        json.dump(contract.to_dict(), fh, indent=2)
    print(f"[*] Contract verdict: {contract.verdict} ({contract.code}) — {contract.reason}")
    print(f"[*] Run artifacts: {run_dir}")

    meta_name = f"{norm_symbol}_{args.timeframe}_{args.model}_meta.json"
    if not promoted:
        meta_name = f"{norm_symbol}_{args.timeframe}_{args.model}_rejected_meta.json"
    meta_path = os.path.join(MODELS_DIR, meta_name)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "symbol": args.symbol,
                "timeframe": args.timeframe,
                "model_type": args.model,
                "feature_hash": FEATURE_HASH,
                "labeling_mode": args.labeling,
                "pt_mult": args.pt_mult,
                "sl_mult": args.sl_mult,
                "calibrated": True,
                "fracdiff_d": fracdiff_d,
                "fracdiff_d_by_symbol": {args.symbol: fracdiff_d},
                "expiration_hours": 168,
                "metrics": metrics,
                "promotion_ok": promoted,
                "promotion_reason": metrics.get("promotion_reason"),
                "trained_at": save_payload["trained_at"],
            },
            f,
            indent=2,
        )
    print(f"[✓] Model MLOps metadata saved to: {meta_path}\n")
    if not promoted:
        sys.exit(2)


if __name__ == "__main__":
    main()

