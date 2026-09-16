#!/usr/bin/env python3
"""GPU training entrypoint for the Jesse / QuantumTrade meta-labeler.

Runs on the Hostinger GPU node:
  1. Detect CUDA (B200 / whatever is attached)
  2. Load dumped Jesse candles (parquet/csv) or Postgres
  3. QuantumAI-event triple-barrier labels at live 5.5 / 1.75 ATR geometry
  4. LightGBM (CUDA if available) + optional LSTM sequence model
  5. Write geometry.json / metrics.json / contract.json
  6. Promote to production *.joblib ONLY if DSR/PBO/recall gates pass

Huge binaries stay on the GPU/trading VPS. This script never talks to brokers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from ml_features import FEATURE_NAMES
from feature_schema import FEATURE_HASH
from gpu_device import detect_training_device
from promotion_contract import SPEC_VERSION, default_geometry, evaluate_contract, write_geometry
from promotion_gates import STRATEGY_PT_ATR, STRATEGY_SL_ATR, evaluate_promotion
from train_ml import (
    MODELS_DIR,
    load_candles_from_db,
    load_candles_from_path,
    prepare_dataset,
    train_model,
)
from train_sequence import train_lstm_meta
from trial_registry import record_trials
from validation_metrics import calculate_sharpe_ratio, deflated_sharpe_ratio, probability_of_backtest_overfitting


def _write_run_bundle(
    run_dir: str,
    metrics: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    os.makedirs(run_dir, exist_ok=True)
    geo = default_geometry()
    write_geometry(os.path.join(run_dir, "geometry.json"), geo)
    payload = {
        "spec_version": SPEC_VERSION,
        "feature_schema_hash": FEATURE_HASH,
        **metrics,
    }
    if extra:
        payload.update(extra)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    contract = evaluate_contract(
        geo,
        payload,
        live_geometry=geo,
        promote_requested=False,
        holdout_spent=False,
        holdout_pass=False,
    )
    with open(os.path.join(run_dir, "contract.json"), "w", encoding="utf-8") as fh:
        json.dump(contract.to_dict(), fh, indent=2)
    print(f"[*] Contract verdict: {contract.verdict} ({contract.code}) — {contract.reason}")
    return contract.to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU Jesse meta-label trainer")
    parser.add_argument("--symbol", default="BTC-USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--candles", default="", help="CSV/parquet dump; empty = Postgres")
    parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "lstm", "both"])
    parser.add_argument("--pt-mult", type=float, default=STRATEGY_PT_ATR)
    parser.add_argument("--sl-mult", type=float, default=STRATEGY_SL_ATR)
    parser.add_argument("--holding", type=int, default=24)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()
    if args.device != "auto":
        os.environ["JESSE_TRAIN_DEVICE"] = args.device

    t0 = time.time()
    device = detect_training_device(args.device)
    print("=========================================================")
    print("         JESSE GPU QUANT TRAINER                         ")
    print("=========================================================")
    print(f"Symbol:        {args.symbol}")
    print(f"Timeframe:     {args.timeframe}")
    print(f"Device:        {device.kind} {device.name} VRAM={device.vram_mb} MiB")
    print(f"Torch:         {device.torch_version} cuda={device.torch_cuda}")
    print(f"LightGBM:      {device.lightgbm_device}")
    print(f"Feature Hash:  {FEATURE_HASH}")
    print("=========================================================")

    if args.candles:
        df = load_candles_from_path(args.candles, args.symbol, args.timeframe)
    else:
        df = load_candles_from_db(args.symbol, args.timeframe)

    X, y, sample_weights, samples_info_sets = prepare_dataset(
        df,
        labeling_mode="triple_barrier",
        pt_mult=args.pt_mult,
        sl_mult=args.sl_mult,
        max_holding=args.holding,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    norm_symbol = args.symbol.replace("/", "-")
    run_root = os.path.join(MODELS_DIR, "runs", f"{stamp}_{norm_symbol}_{args.timeframe}_gpu")
    os.makedirs(run_root, exist_ok=True)

    lgbm_metrics: Dict[str, Any] = {}
    lstm_metrics: Dict[str, Any] = {}
    lgbm_cols: List[np.ndarray] = []
    lstm_cols: List[np.ndarray] = []

    if args.model in ("lightgbm", "both"):
        pipeline, lgbm_metrics = train_model(
            X,
            y,
            sample_weights=sample_weights,
            samples_info_sets=samples_info_sets,
            model_type="lightgbm",
            pt_mult=args.pt_mult,
            sl_mult=args.sl_mult,
            symbol=args.symbol,
            timeframe=args.timeframe,
        )
        promo = evaluate_promotion(lgbm_metrics, pt_mult=args.pt_mult, sl_mult=args.sl_mult)
        lgbm_metrics["promotion_ok"] = promo.ok
        lgbm_metrics["promotion_reason"] = promo.reason
        fname = f"{norm_symbol}_{args.timeframe}_lightgbm"
        if promo.ok:
            path = os.path.join(MODELS_DIR, f"{fname}.joblib")
        else:
            path = os.path.join(MODELS_DIR, f"{fname}.rejected.joblib")
            print(f"[!] LightGBM promotion BLOCKED: {promo.reason}")
        joblib.dump(
            {
                "pipeline": pipeline,
                "feature_names": FEATURE_NAMES,
                "feature_hash": FEATURE_HASH,
                "symbol": args.symbol,
                "timeframe": args.timeframe,
                "model_type": "lightgbm",
                "pt_mult": args.pt_mult,
                "sl_mult": args.sl_mult,
                "calibrated": True,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "metrics": lgbm_metrics,
                "promotion_ok": promo.ok,
                "promotion_reason": promo.reason,
                "device": device.to_dict(),
            },
            path,
        )
        print(f"[{'✓' if promo.ok else '!'}] LightGBM artifact: {path}")
        _write_run_bundle(os.path.join(run_root, "lightgbm"), lgbm_metrics, {"symbol": args.symbol, "timeframe": args.timeframe, "pt_mult": args.pt_mult, "sl_mult": args.sl_mult})

    if args.model in ("lstm", "both"):
        state, lstm_metrics, lstm_cols = train_lstm_meta(
            X, y, sample_weights=sample_weights, device_info=device,
        )
        if lstm_cols:
            holdout = lstm_cols[int(np.argmax([calculate_sharpe_ratio(c) for c in lstm_cols]))]
            n_lstm_trials = record_trials(f"ml:{args.symbol}:{args.timeframe}:lstm", len(lstm_cols), note="lstm_grid")
            var_s = float(np.var([calculate_sharpe_ratio(c) for c in lstm_cols], ddof=1)) if len(lstm_cols) > 1 else None
            trial_matrix = np.column_stack([c[: min(len(x) for x in lstm_cols)] for c in lstm_cols])
            pbo, med_rank, ranks = probability_of_backtest_overfitting(
                trial_matrix, n_blocks=min(16, max(4, (trial_matrix.shape[0] // 20) * 2)),
            )
            if not ranks:
                pbo = 1.0
            lstm_metrics["deflated_sharpe_ratio"] = float(deflated_sharpe_ratio(holdout, n_trials=n_lstm_trials, variance_of_trials=var_s))
            lstm_metrics["prob_backtest_overfitting"] = float(pbo)
            lstm_metrics["holdout_sharpe"] = float(calculate_sharpe_ratio(holdout))
            lstm_metrics["n_trials"] = int(n_lstm_trials)
            lstm_metrics["pbo_median_rank"] = float(med_rank)
            lstm_metrics["pt_mult"] = float(args.pt_mult)
            lstm_metrics["sl_mult"] = float(args.sl_mult)
            lstm_metrics["spec_version"] = SPEC_VERSION
            lstm_metrics["feature_schema_hash"] = FEATURE_HASH
            lstm_metrics["used_raw_n_as_effective"] = False
        promo = evaluate_promotion(lstm_metrics, pt_mult=args.pt_mult, sl_mult=args.sl_mult)
        lstm_metrics["promotion_ok"] = promo.ok
        lstm_metrics["promotion_reason"] = promo.reason
        lstm_name = f"{norm_symbol}_{args.timeframe}_lstm"
        if promo.ok and state is not None:
            path = os.path.join(MODELS_DIR, f"{lstm_name}.pt")
        else:
            path = os.path.join(MODELS_DIR, f"{lstm_name}.rejected.pt")
            print(f"[!] LSTM promotion BLOCKED: {promo.reason}")
        joblib.dump(
            {
                "state_dict": state,
                "feature_names": FEATURE_NAMES,
                "feature_hash": FEATURE_HASH,
                "metrics": lstm_metrics,
                "promotion_ok": promo.ok,
                "device": device.to_dict(),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "pt_mult": args.pt_mult,
                "sl_mult": args.sl_mult,
            },
            path,
        )
        print(f"[{'✓' if promo.ok else '!'}] LSTM artifact: {path}")
        _write_run_bundle(os.path.join(run_root, "lstm"), lstm_metrics, {"symbol": args.symbol, "timeframe": args.timeframe, "pt_mult": args.pt_mult, "sl_mult": args.sl_mult})

    summary = {
        "elapsed_sec": round(time.time() - t0, 1),
        "device": device.to_dict(),
        "lightgbm": {k: lgbm_metrics.get(k) for k in ("deflated_sharpe_ratio", "prob_backtest_overfitting", "bullish_recall", "bearish_recall", "promotion_ok", "promotion_reason", "n_events")},
        "lstm": {k: lstm_metrics.get(k) for k in ("deflated_sharpe_ratio", "prob_backtest_overfitting", "bullish_recall", "bearish_recall", "promotion_ok", "promotion_reason", "n_events")},
        "run_dir": run_root,
    }
    with open(os.path.join(run_root, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    lgbm_ok = bool(lgbm_metrics.get("promotion_ok"))
    lstm_ok = bool(lstm_metrics.get("promotion_ok"))
    if args.model == "lightgbm":
        return 0 if lgbm_ok else 2
    if args.model == "lstm":
        return 0 if lstm_ok else 2
    return 0 if (lgbm_ok or lstm_ok) else 2


if __name__ == "__main__":
    sys.exit(main())
