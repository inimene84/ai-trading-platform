"""GPU trainer CLI helpers with no LightGBM/sklearn/joblib imports.

Backend CI can import this module. Heavy training stays in train_gpu.py.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from asset_universe import candle_filename, lookup_symbol, normalize_symbol
from promotion_gates import STRATEGY_PT_ATR, STRATEGY_SL_ATR

# Train on event samples this small, but never every-bar fallback on the GPU path.
MIN_TRAIN_EVENTS = 20


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
