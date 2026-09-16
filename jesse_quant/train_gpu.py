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

Multi-asset:
  python train_gpu.py --list-universe
  python train_gpu.py --asset-class forex --candles-dir storage/candles --model both
  python train_gpu.py --asset-class all --candles-dir storage/candles --model both
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np

from asset_universe import (
    ALL_CLASSES,
    UniverseSymbol,
    candle_filename,
    lookup_symbol,
    normalize_symbol,
    resolve_universe,
)
from ml_features import FEATURE_NAMES
from feature_schema import FEATURE_HASH
from gpu_device import detect_training_device
from promotion_contract import SPEC_VERSION, default_geometry, evaluate_contract, write_geometry
from promotion_gates import STRATEGY_PT_ATR, STRATEGY_SL_ATR, evaluate_promotion
from train_ml import (
    MODELS_DIR,
    TooFewEventsError,
    load_candles_from_db,
    load_candles_from_path,
    prepare_dataset,
    train_model,
)
from train_sequence import train_lstm_meta
from trial_registry import record_trials
from validation_metrics import calculate_sharpe_ratio, deflated_sharpe_ratio, probability_of_backtest_overfitting

# Train on event samples this small, but never every-bar fallback on the GPU path.
MIN_TRAIN_EVENTS = 20
GATE_KEYS = (
    "deflated_sharpe_ratio",
    "prob_backtest_overfitting",
    "bullish_recall",
    "bearish_recall",
    "promotion_ok",
    "promotion_reason",
    "n_events",
    "error",
)


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


def find_candle_path(symbol: str, candles_dir: str, explicit: str = "") -> Optional[str]:
    if explicit:
        return explicit if os.path.exists(explicit) else None
    norm = normalize_symbol(symbol)
    candidates = [
        os.path.join(candles_dir, candle_filename(norm, "1m")),
        os.path.join(candles_dir, candle_filename(norm, "1h")),
        os.path.join(candles_dir, f"{norm}.csv.gz"),
        os.path.join(candles_dir, f"{norm}_1m.csv.gz"),
        os.path.join(candles_dir, f"{norm}_1h.csv.gz"),
    ]
    item = lookup_symbol(norm)
    if item is not None:
        candidates.insert(0, os.path.join(candles_dir, item.filename()))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _gate_slice(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {k: metrics.get(k) for k in GATE_KEYS}


def train_one_symbol(
    symbol: str,
    timeframe: str,
    candles: str,
    model: str,
    pt_mult: float,
    sl_mult: float,
    holding: int,
    device: Any,
    asset_class: str = "",
    events: str = "quantum_ai",
) -> Dict[str, Any]:
    """Train LightGBM and/or LSTM for one symbol.

    events=quantum_ai never falls back to every-bar labels (GPU promotion path).
    events=everybar is kept for the overlapping B200 campaign CLI.
    """
    t0 = time.time()
    summary: Dict[str, Any] = {
        "symbol": symbol,
        "asset_class": asset_class,
        "timeframe": timeframe,
        "candles": candles,
        "skipped": False,
        "skip_reason": "",
        "n_bars": 0,
        "n_events": 0,
        "lightgbm": {},
        "lstm": {},
        "promoted": False,
    }
    if candles:
        df = load_candles_from_path(candles, symbol, timeframe)
    else:
        df = load_candles_from_db(symbol, timeframe)
    summary["n_bars"] = int(len(df))
    if len(df) < 400:
        summary["skipped"] = True
        summary["skip_reason"] = f"too few bars ({len(df)})"
        print(f"[!] {symbol}: {summary['skip_reason']}")
        return summary

    everybar = events == "everybar"
    try:
        X, y, sample_weights, samples_info_sets = prepare_dataset(
            df,
            labeling_mode="triple_barrier",
            pt_mult=pt_mult,
            sl_mult=sl_mult,
            max_holding=holding,
            fallback_every_bar=everybar,
            min_events=10**9 if everybar else MIN_TRAIN_EVENTS,
        )
    except TooFewEventsError as exc:
        summary["skipped"] = True
        summary["n_events"] = exc.n_events
        summary["skip_reason"] = str(exc)
        print(f"[!] {symbol}: {exc} — not training, not promoting")
        return summary

    summary["n_events"] = int(len(y))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    norm_symbol = normalize_symbol(symbol)
    run_root = os.path.join(MODELS_DIR, "runs", f"{stamp}_{norm_symbol}_{timeframe}_gpu")
    os.makedirs(run_root, exist_ok=True)
    summary["run_dir"] = run_root

    extra_meta = {
        "symbol": symbol,
        "timeframe": timeframe,
        "pt_mult": pt_mult,
        "sl_mult": sl_mult,
        "asset_class": asset_class,
    }
    lgbm_metrics: Dict[str, Any] = {}
    lstm_metrics: Dict[str, Any] = {}
    any_ok = False

    if model in ("lightgbm", "both"):
        pipeline, lgbm_metrics = train_model(
            X,
            y,
            sample_weights=sample_weights,
            samples_info_sets=samples_info_sets,
            model_type="lightgbm",
            pt_mult=pt_mult,
            sl_mult=sl_mult,
            symbol=symbol,
            timeframe=timeframe,
        )
        promo = evaluate_promotion(lgbm_metrics, pt_mult=pt_mult, sl_mult=sl_mult)
        lgbm_metrics["promotion_ok"] = promo.ok
        lgbm_metrics["promotion_reason"] = promo.reason
        fname = f"{norm_symbol}_{timeframe}_lightgbm"
        if promo.ok:
            path = os.path.join(MODELS_DIR, f"{fname}.joblib")
            any_ok = True
        else:
            path = os.path.join(MODELS_DIR, f"{fname}.rejected.joblib")
            print(f"[!] LightGBM promotion BLOCKED: {promo.reason}")
        joblib.dump(
            {
                "pipeline": pipeline,
                "feature_names": FEATURE_NAMES,
                "feature_hash": FEATURE_HASH,
                "symbol": symbol,
                "timeframe": timeframe,
                "asset_class": asset_class,
                "model_type": "lightgbm",
                "pt_mult": pt_mult,
                "sl_mult": sl_mult,
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
        _write_run_bundle(os.path.join(run_root, "lightgbm"), lgbm_metrics, extra_meta)

    if model in ("lstm", "both"):
        state, lstm_metrics, lstm_cols = train_lstm_meta(
            X, y, sample_weights=sample_weights, device_info=device,
        )
        if lstm_cols:
            holdout = lstm_cols[int(np.argmax([calculate_sharpe_ratio(c) for c in lstm_cols]))]
            n_lstm_trials = record_trials(f"ml:{symbol}:{timeframe}:lstm", len(lstm_cols), note="lstm_grid")
            var_s = float(np.var([calculate_sharpe_ratio(c) for c in lstm_cols], ddof=1)) if len(lstm_cols) > 1 else None
            trial_matrix = np.column_stack([c[: min(len(x) for x in lstm_cols)] for c in lstm_cols])
            pbo, med_rank, ranks = probability_of_backtest_overfitting(
                trial_matrix, n_blocks=min(16, max(4, (trial_matrix.shape[0] // 20) * 2)),
            )
            if not ranks:
                pbo = 1.0
            lstm_metrics["deflated_sharpe_ratio"] = float(
                deflated_sharpe_ratio(holdout, n_trials=n_lstm_trials, variance_of_trials=var_s)
            )
            lstm_metrics["prob_backtest_overfitting"] = float(pbo)
            lstm_metrics["holdout_sharpe"] = float(calculate_sharpe_ratio(holdout))
            lstm_metrics["n_trials"] = int(n_lstm_trials)
            lstm_metrics["pbo_median_rank"] = float(med_rank)
            lstm_metrics["pt_mult"] = float(pt_mult)
            lstm_metrics["sl_mult"] = float(sl_mult)
            lstm_metrics["spec_version"] = SPEC_VERSION
            lstm_metrics["feature_schema_hash"] = FEATURE_HASH
            lstm_metrics["used_raw_n_as_effective"] = False
        promo = evaluate_promotion(lstm_metrics, pt_mult=pt_mult, sl_mult=sl_mult)
        lstm_metrics["promotion_ok"] = promo.ok
        lstm_metrics["promotion_reason"] = promo.reason
        lstm_name = f"{norm_symbol}_{timeframe}_lstm"
        if promo.ok and state is not None:
            path = os.path.join(MODELS_DIR, f"{lstm_name}.pt")
            any_ok = True
        else:
            path = os.path.join(MODELS_DIR, f"{lstm_name}.rejected.pt")
            print(f"[!] LSTM promotion BLOCKED: {promo.reason}")
        joblib.dump(
            {
                "state_dict": state,
                "feature_names": FEATURE_NAMES,
                "feature_hash": FEATURE_HASH,
                "symbol": symbol,
                "asset_class": asset_class,
                "metrics": lstm_metrics,
                "promotion_ok": promo.ok,
                "device": device.to_dict(),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "pt_mult": pt_mult,
                "sl_mult": sl_mult,
            },
            path,
        )
        print(f"[{'✓' if promo.ok else '!'}] LSTM artifact: {path}")
        _write_run_bundle(os.path.join(run_root, "lstm"), lstm_metrics, extra_meta)

    summary["elapsed_sec"] = round(time.time() - t0, 1)
    summary["lightgbm"] = _gate_slice(lgbm_metrics)
    summary["lstm"] = _gate_slice(lstm_metrics)
    summary["promoted"] = bool(any_ok)
    with open(os.path.join(run_root, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return summary


def train_universe(
    asset_class: str,
    timeframe: str,
    candles_dir: str,
    model: str,
    pt_mult: float,
    sl_mult: float,
    holding: int,
    device: Any,
    symbols: Optional[Sequence[str]] = None,
    events: str = "quantum_ai",
    exclude: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    rows: List[UniverseSymbol] = resolve_universe(asset_class, symbols=symbols)
    skip = {normalize_symbol(s) for s in (exclude or []) if s}
    batch: Dict[str, Any] = {
        "asset_class": asset_class,
        "timeframe": timeframe,
        "model": model,
        "pt_mult": pt_mult,
        "sl_mult": sl_mult,
        "n_symbols": len(rows),
        "results": [],
        "promoted": [],
        "blocked": [],
        "skipped": [],
        "missing_candles": [],
    }
    if not rows:
        print(f"[!] No symbols in asset class {asset_class} — skip (no fake labels)")
        return batch

    for item in rows:
        if skip and normalize_symbol(item.symbol) in skip:
            rec = {
                "symbol": item.symbol,
                "asset_class": item.asset_class,
                "skipped": True,
                "skip_reason": "excluded",
            }
            batch["skipped"].append(item.symbol)
            batch["results"].append(rec)
            continue
        path = find_candle_path(item.symbol, candles_dir)
        if not path:
            rec = {
                "symbol": item.symbol,
                "asset_class": item.asset_class,
                "skipped": True,
                "skip_reason": "missing candles",
            }
            batch["missing_candles"].append(item.symbol)
            batch["skipped"].append(item.symbol)
            batch["results"].append(rec)
            print(f"[!] {item.symbol}: no candle dump in {candles_dir}")
            continue
        print("=========================================================")
        print(f"  {item.asset_class}  {item.symbol}  candles={path}")
        print("=========================================================")
        rec = train_one_symbol(
            symbol=item.symbol,
            timeframe=timeframe,
            candles=path,
            model=model,
            pt_mult=pt_mult,
            sl_mult=sl_mult,
            holding=holding,
            device=device,
            asset_class=item.asset_class,
            events=events,
        )
        batch["results"].append(rec)
        if rec.get("skipped"):
            batch["skipped"].append(item.symbol)
        elif rec.get("promoted"):
            batch["promoted"].append(item.symbol)
        else:
            batch["blocked"].append(item.symbol)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(
        MODELS_DIR, "runs", f"{stamp}_{asset_class}_{timeframe}_batch.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(batch, fh, indent=2, default=str)
    batch["batch_path"] = out_path
    print(json.dumps({k: batch[k] for k in ("asset_class", "promoted", "blocked", "skipped", "missing_candles")}, indent=2))
    return batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU Jesse meta-label trainer")
    parser.add_argument("--symbol", default="BTC-USDT")
    parser.add_argument(
        "--asset-class",
        default="",
        help="crypto|forex|metals|minerals|stocks|all (batch). Empty = single --symbol",
    )
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--candles", default="", help="CSV/parquet dump; empty = Postgres")
    parser.add_argument(
        "--candles-dir",
        default="",
        help="Directory of {SYMBOL}_{1m|1h}.csv.gz dumps for --asset-class batches",
    )
    parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "lstm", "both"])
    parser.add_argument("--pt-mult", type=float, default=STRATEGY_PT_ATR)
    parser.add_argument("--sl-mult", type=float, default=STRATEGY_SL_ATR)
    parser.add_argument("--holding", type=int, default=24)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--events",
        default="quantum_ai",
        choices=["quantum_ai", "everybar"],
        help="quantum_ai = strategy-event labels (promotion path); everybar = campaign compatibility",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="comma-separated symbols to skip in a batch (e.g. BTC-USDT,ETH-USDT,SOL-USDT)",
    )
    parser.add_argument("--list-universe", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_universe:
        for cls in ALL_CLASSES:
            print(f"## {cls}")
            for item in resolve_universe(cls):
                print(f"  {item.symbol:12} {item.source:12} {item.ticker}")
        return 0

    if args.device != "auto":
        os.environ["JESSE_TRAIN_DEVICE"] = args.device

    t0 = time.time()
    device = detect_training_device(args.device)
    print("=========================================================")
    print("         JESSE GPU QUANT TRAINER                         ")
    print("=========================================================")
    print(f"Symbol:        {args.symbol}")
    print(f"Asset class:   {args.asset_class or '(single symbol)'}")
    print(f"Timeframe:     {args.timeframe}")
    print(f"Device:        {device.kind} {device.name} VRAM={device.vram_mb} MiB")
    print(f"Torch:         {device.torch_version} cuda={device.torch_cuda}")
    print(f"LightGBM:      {device.lightgbm_device}")
    print(f"Feature Hash:  {FEATURE_HASH}")
    print("=========================================================")

    candles_dir = args.candles_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "storage", "candles"
    )

    if args.asset_class:
        batch = train_universe(
            asset_class=args.asset_class,
            timeframe=args.timeframe,
            candles_dir=candles_dir,
            model=args.model,
            pt_mult=args.pt_mult,
            sl_mult=args.sl_mult,
            holding=args.holding,
            device=device,
            events=args.events,
            exclude=[s.strip() for s in args.exclude.split(",") if s.strip()],
        )
        print(f"[*] Batch elapsed {time.time() - t0:.1f}s")
        if batch.get("promoted"):
            return 0
        return 2

    candles = args.candles
    if not candles:
        found = find_candle_path(args.symbol, candles_dir)
        candles = found or ""
    rec = train_one_symbol(
        symbol=args.symbol,
        timeframe=args.timeframe,
        candles=candles,
        model=args.model,
        pt_mult=args.pt_mult,
        sl_mult=args.sl_mult,
        holding=args.holding,
        device=device,
        asset_class=(lookup_symbol(args.symbol).asset_class if lookup_symbol(args.symbol) else ""),
        events=args.events,
    )
    if rec.get("skipped"):
        return 2
    lgbm_ok = bool(rec.get("lightgbm", {}).get("promotion_ok"))
    lstm_ok = bool(rec.get("lstm", {}).get("promotion_ok"))
    if args.model == "lightgbm":
        return 0 if lgbm_ok else 2
    if args.model == "lstm":
        return 0 if lstm_ok else 2
    return 0 if (lgbm_ok or lstm_ok) else 2


if __name__ == "__main__":
    sys.exit(main())
