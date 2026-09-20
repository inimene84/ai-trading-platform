"""ml_features.py must stay strictly causal (no future rows at t)."""

import numpy as np
import pandas as pd

from ml_features import FEATURE_NAMES, compute_features_df


def _ohlcv(n: int = 260, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.4, n))
    high = close + rng.uniform(0.1, 0.6, n)
    low = close - rng.uniform(0.1, 0.6, n)
    open_ = close + rng.normal(0.0, 0.15, n)
    volume = rng.uniform(100.0, 500.0, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def test_feature_names_include_fracdiff_pair():
    assert "fd_close" in FEATURE_NAMES
    assert "fd_logret" in FEATURE_NAMES


def test_features_at_t_ignore_future_rows():
    df = _ohlcv()
    full = compute_features_df(df, fracdiff_d=0.4)
    prefix = compute_features_df(df.iloc[:220].copy(), fracdiff_d=0.4)
    left = prefix.iloc[219].to_numpy(dtype=float)
    right = full.iloc[219].to_numpy(dtype=float)
    assert np.allclose(left, right, equal_nan=True, atol=1e-8)


def test_future_shock_does_not_leak_into_the_past():
    df = _ohlcv()
    shocked = df.copy()
    shocked.loc[shocked.index[-1], "close"] = shocked["close"].iloc[-1] * 1.25
    shocked.loc[shocked.index[-1], "high"] = shocked["close"].iloc[-1] * 1.30
    base = compute_features_df(df, fracdiff_d=0.4)
    after = compute_features_df(shocked, fracdiff_d=0.4)
    assert np.allclose(
        base.iloc[:-1].to_numpy(dtype=float),
        after.iloc[:-1].to_numpy(dtype=float),
        equal_nan=True,
        atol=1e-8,
    )
