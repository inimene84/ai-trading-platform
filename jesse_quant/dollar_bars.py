#!/usr/bin/env python3
"""Dollar-bar resampling for research experiments.

Live serving still uses 1h time bars so train-serve parity with FEATURE_HASH
is preserved. Dollar bars are an optional sampling scheme for GPU research
jobs, not a production feature schema change.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def resample_dollar_bars(
    df: pd.DataFrame,
    dollar_threshold: Optional[float] = None,
    target_bars: int = 25_000,
) -> pd.DataFrame:
    """Aggregate 1m (or finer) OHLCV into dollar bars.

    If dollar_threshold is None, choose a threshold that yields roughly
    `target_bars` bars over the full sample.
    """
    if "close" not in df.columns or "volume" not in df.columns:
        raise ValueError("dollar bars require close and volume columns")
    notional = (df["close"].astype(float) * df["volume"].astype(float)).clip(lower=0.0)
    if dollar_threshold is None:
        total = float(notional.sum())
        dollar_threshold = max(total / max(target_bars, 1), 1.0)

    cum = 0.0
    rows = []
    bucket_open = None
    bucket_high = None
    bucket_low = None
    bucket_close = None
    bucket_vol = 0.0
    bucket_ts = None

    for ts, row in df.iterrows():
        o, h, l, c, v = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row["volume"])
        if bucket_open is None:
            bucket_open = o
            bucket_high = h
            bucket_low = l
            bucket_ts = ts
            bucket_vol = 0.0
        bucket_high = max(bucket_high, h)
        bucket_low = min(bucket_low, l)
        bucket_close = c
        bucket_vol += v
        cum += c * v
        if cum >= dollar_threshold:
            rows.append(
                {
                    "datetime": bucket_ts,
                    "open": bucket_open,
                    "high": bucket_high,
                    "low": bucket_low,
                    "close": bucket_close,
                    "volume": bucket_vol,
                }
            )
            cum = 0.0
            bucket_open = None

    out = pd.DataFrame(rows)
    if out.empty:
        return df.copy()
    out = out.set_index("datetime")
    return out
