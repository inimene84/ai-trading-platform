"""
Machine Learning Feature Engineering Module for Quant Trading
Provides unified, stationary, and leak-free feature extraction for both
training (train_ml.py) and live/backtest inference (QuantumMLStrategy).
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


FEATURE_NAMES = [
    # Momentum & Oscillators
    "rsi_14",
    "rsi_7",
    "stoch_k",
    "stoch_d",
    "macd_norm",
    "macd_signal_norm",
    "macd_hist_norm",
    "roc_3",
    "roc_6",
    "roc_12",
    "roc_24",
    # Trend & Moving Average Ratios
    "ema_ratio_9_21",
    "ema_ratio_21_50",
    "ema_ratio_50_200",
    "price_to_ema_21",
    "price_to_ema_50",
    # Volatility & Range
    "natr_14",
    "bb_percent_b",
    "bb_bandwidth",
    "high_low_ratio",
    "real_body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    # Volume Dynamics
    "volume_zscore_24",
    "volume_ratio_ema",
    "obv_slope_6",
]


def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes stationary technical indicators and price action features on an OHLCV DataFrame.
    Guarantees strict causality: all features at index t only use rows <= t.
    """
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"]

    features = pd.DataFrame(index=df.index)

    # 1. Momentum & Oscillators
    features["rsi_14"] = _calc_rsi(c, 14)
    features["rsi_7"] = _calc_rsi(c, 7)

    # Stochastic %K and %D
    low_14 = l.rolling(window=14, min_periods=14).min()
    high_14 = h.rolling(window=14, min_periods=14).max()
    features["stoch_k"] = ((c - low_14) / (high_14 - low_14 + 1e-12)) * 100.0
    features["stoch_d"] = features["stoch_k"].rolling(window=3, min_periods=3).mean()

    # MACD normalized by ATR
    ema_12 = c.ewm(span=12, adjust=False).mean()
    ema_26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_sig

    # ATR 14
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False).mean()

    features["macd_norm"] = macd_line / (atr_14 + 1e-12)
    features["macd_signal_norm"] = macd_sig / (atr_14 + 1e-12)
    features["macd_hist_norm"] = macd_hist / (atr_14 + 1e-12)

    # Rate of Change (log returns over varying lookbacks)
    features["roc_3"] = (c / c.shift(3) - 1.0) * 100.0
    features["roc_6"] = (c / c.shift(6) - 1.0) * 100.0
    features["roc_12"] = (c / c.shift(12) - 1.0) * 100.0
    features["roc_24"] = (c / c.shift(24) - 1.0) * 100.0

    # 2. Trend & Moving Average Ratios
    ema_9 = c.ewm(span=9, adjust=False).mean()
    ema_21 = c.ewm(span=21, adjust=False).mean()
    ema_50 = c.ewm(span=50, adjust=False).mean()
    ema_200 = c.ewm(span=200, adjust=False).mean()

    features["ema_ratio_9_21"] = (ema_9 / (ema_21 + 1e-12) - 1.0) * 100.0
    features["ema_ratio_21_50"] = (ema_21 / (ema_50 + 1e-12) - 1.0) * 100.0
    features["ema_ratio_50_200"] = (ema_50 / (ema_200 + 1e-12) - 1.0) * 100.0
    features["price_to_ema_21"] = (c / (ema_21 + 1e-12) - 1.0) * 100.0
    features["price_to_ema_50"] = (c / (ema_50 + 1e-12) - 1.0) * 100.0

    # 3. Volatility & Range
    features["natr_14"] = (atr_14 / (c + 1e-12)) * 100.0

    # Bollinger Bands (20 periods, 2 std)
    bb_mid = c.rolling(window=20, min_periods=20).mean()
    bb_std = c.rolling(window=20, min_periods=20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    features["bb_percent_b"] = (c - bb_lower) / (bb_upper - bb_lower + 1e-12)
    features["bb_bandwidth"] = (bb_upper - bb_lower) / (bb_mid + 1e-12)

    # Candle price action geometry
    candle_range = h - l + 1e-12
    body_range = (c - o).abs()
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l

    features["high_low_ratio"] = (candle_range / (c + 1e-12)) * 100.0
    features["real_body_ratio"] = body_range / candle_range
    features["upper_shadow_ratio"] = upper_wick / candle_range
    features["lower_shadow_ratio"] = lower_wick / candle_range

    # 4. Volume Dynamics
    v_mean_24 = v.rolling(window=24, min_periods=24).mean()
    v_std_24 = v.rolling(window=24, min_periods=24).std()
    features["volume_zscore_24"] = (v - v_mean_24) / (v_std_24 + 1e-12)
    v_ema_20 = v.ewm(span=20, adjust=False).mean()
    features["volume_ratio_ema"] = v / (v_ema_20 + 1e-12)

    # On-Balance Volume Slope (normalized)
    obv_direction = np.sign(c.diff().fillna(0))
    obv = (obv_direction * v).cumsum()
    features["obv_slope_6"] = (obv - obv.shift(6)) / (v_mean_24 * 6 + 1e-12)

    return features[FEATURE_NAMES]


def compute_latest_features(candles: np.ndarray) -> np.ndarray:
    """
    Extracts the feature vector for the latest candle from a Jesse candle numpy array:
    Format: [[timestamp, open, close, high, low, volume], ...]
    Returns: 1D numpy array of shape (n_features,).
    """
    if len(candles) < 220:
        return np.full((len(FEATURE_NAMES),), np.nan)

    # Jesse candles format: [timestamp, open, close, high, low, volume]
    recent = candles[-250:]
    df = pd.DataFrame(
        {
            "open": recent[:, 1],
            "close": recent[:, 2],
            "high": recent[:, 3],
            "low": recent[:, 4],
            "volume": recent[:, 5],
        }
    )

    feat_df = compute_features_df(df)
    latest = feat_df.iloc[-1].to_numpy(dtype=np.float32)
    return latest
