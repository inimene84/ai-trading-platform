#!/usr/bin/env python3
"""
Triple-Barrier Method and Meta-Labeling Module
Implements Marcos López de Prado's path-dependent labeling framework:
1. Triple-Barrier Labeling: Horizontal Profit Target, Horizontal Stop Loss, Vertical Holding Limit
2. Concurrent Label Uniqueness Weighting: Sample weights for LightGBM/GBM classifiers
3. Meta-Labeling: Secondary binary labels determining if a primary model signal hits profit before stop
"""

from typing import Dict, Optional, Union
import numpy as np
import pandas as pd

from barrier_config import ATR_PERIOD, MAX_HOLDING_BARS, SL_ATR_MULT, TP_ATR_MULT


def get_daily_volatility(close: pd.Series, lookback: int = 50) -> pd.Series:
    """Computes rolling exponential standard deviation of returns as volatility proxy."""
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1.0  # Daily returns
    df0 = df0.ewm(span=lookback).std()
    return df0.bfill()


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range. The first bar has no previous close, so it is high minus low."""
    high, low, close = df["high"], df["low"], df["close"]
    previous = close.shift(1)
    parts = pd.concat(
        [high - low, (high - previous).abs(), (low - previous).abs()],
        axis=1,
    )
    return parts.max(axis=1)


def get_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Mean of the last `period` true ranges, including the current bar.

    Bars before a full window are NaN. This is not an exponential average:
    `ewm(span=period)` depends on where the file started and is not Wilder ATR.
    """
    if period < 1:
        raise ValueError("ATR period must be positive")
    ranges = true_range(df)
    averaged = ranges.rolling(period, min_periods=period).mean()
    # Index `period` is the first bar whose window starts at index 1, so every
    # range in that window had a previous close. The bar at index 0 does not.
    averaged.iloc[:period] = np.nan
    return averaged


def _median_step(index: pd.Index) -> Optional[pd.Timedelta]:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return None
    deltas = index.to_series().diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return None
    return deltas.median()


def _event_location(df: pd.DataFrame, idx) -> Optional[int]:
    loc = df.index.get_loc(idx)
    if isinstance(loc, slice) or not isinstance(loc, (int, np.integer)):
        return None
    return int(loc)


def _resolve_side(sides: Optional[pd.Series], idx) -> Optional[int]:
    if sides is None:
        return 1
    raw = sides.loc[idx]
    if isinstance(raw, pd.Series):
        raw = raw.iloc[-1]
    try:
        side = int(raw)
    except (TypeError, ValueError):
        return None
    if side not in (1, -1):
        return None
    return side


def _walk_barriers(
    df: pd.DataFrame,
    loc: int,
    side: int,
    atr: float,
    pt_multiplier: float,
    sl_multiplier: float,
    max_holding_bars: int,
) -> Optional[dict]:
    """First barrier strictly after `loc`. None means the path cannot be labeled."""
    end = loc + 1 + max_holding_bars
    if end > len(df):
        return None
    window = df.iloc[loc:end]
    step = _median_step(df.index)
    if step is not None:
        deltas = window.index.to_series().diff().iloc[1:]
        if (deltas > step * 1.5).any() or (deltas <= pd.Timedelta(0)).any():
            return None

    entry = float(df["close"].iloc[loc])
    if side == 1:
        profit = entry + pt_multiplier * atr
        stop = entry - sl_multiplier * atr
    else:
        profit = entry - pt_multiplier * atr
        stop = entry + sl_multiplier * atr

    highs = window["high"].iloc[1:]
    lows = window["low"].iloc[1:]
    closes = window["close"].iloc[1:]
    for offset in range(len(closes)):
        bar_high = float(highs.iloc[offset])
        bar_low = float(lows.iloc[offset])
        if side == 1:
            hit_profit = bar_high >= profit
            hit_stop = bar_low <= stop
        else:
            hit_profit = bar_low <= profit
            hit_stop = bar_high >= stop
        # A bar that trades through both levels is a stop. Wick order is unknown.
        if hit_stop:
            return {
                "t1": closes.index[offset],
                "ret": (-sl_multiplier * atr / entry) * 100.0,
                "label": -1,
                "touch_type": "sl",
            }
        if hit_profit:
            return {
                "t1": closes.index[offset],
                "ret": (pt_multiplier * atr / entry) * 100.0,
                "label": 1,
                "touch_type": "pt",
            }

    expiry = float(closes.iloc[-1])
    favorable = (expiry - entry) if side == 1 else (entry - expiry)
    return {
        "t1": closes.index[-1],
        "ret": (favorable / entry) * 100.0,
        "label": 0,
        "touch_type": "timeout",
    }


def apply_triple_barrier(
    df: pd.DataFrame,
    events_idx: Optional[pd.Index] = None,
    pt_multiplier: float = TP_ATR_MULT,
    sl_multiplier: float = SL_ATR_MULT,
    max_holding_bars: int = MAX_HOLDING_BARS,
    use_atr: bool = True,
    sides: Optional[pd.Series] = None,
    atr_period: int = ATR_PERIOD,
) -> pd.DataFrame:
    """
    Path-dependent barrier outcome for each event.

    `sides` is +1 for a long and -1 for a short. The default is long, which
    matches the long-only QuantumAI event sampler. A short profit is a down
    move of `pt_multiplier` ATR, not an up move.

    label +1 means the profit barrier was touched first. label -1 means the
    stop, including a bar that touched both. label 0 is the vertical barrier
    and is never rewritten from the expiry return. Events without a full ATR
    window, without a full forward window, or with a missing bar are omitted.
    """
    if max_holding_bars < 1:
        raise ValueError("holding period must be positive")
    if events_idx is None:
        events_idx = df.index[:-max_holding_bars]

    if use_atr:
        vol = get_atr(df, period=atr_period)
    else:
        vol = df["close"] * df["close"].pct_change().rolling(20).std()

    rows = []
    for idx in events_idx:
        loc = _event_location(df, idx)
        if loc is None:
            continue
        side = _resolve_side(sides, idx)
        if side is None:
            continue
        atr = float(vol.iloc[loc])
        if not np.isfinite(atr) or atr <= 0:
            continue
        walked = _walk_barriers(
            df, loc, side, atr, pt_multiplier, sl_multiplier, max_holding_bars
        )
        if walked is None:
            continue
        rows.append({
            "datetime": idx,
            "t1": walked["t1"],
            "entry_price": float(df["close"].iloc[loc]),
            "trgt": atr,
            "ret": walked["ret"],
            "label": walked["label"],
            "touch_type": walked["touch_type"],
            "side": side,
        })

    columns = ["t1", "entry_price", "trgt", "ret", "label", "touch_type", "side"]
    if not rows:
        return pd.DataFrame(columns=columns).rename_axis("datetime")
    return pd.DataFrame(rows).set_index("datetime")


def realized_payoff_stats(returns: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """Win/loss payoff stats used by the promotion tests and Kelly fallback."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    n_wins = int(len(wins))
    n_losses = int(len(losses))
    avg_win = float(wins.mean()) if n_wins else 0.0
    avg_loss_abs = float(np.abs(losses).mean()) if n_losses else 0.0
    payoff = (avg_win / avg_loss_abs) if avg_loss_abs > 0 else 0.0
    return {
        "n": int(len(arr)),
        "n_wins": n_wins,
        "n_losses": n_losses,
        "avg_win": avg_win,
        "avg_loss_abs": avg_loss_abs,
        "payoff_ratio": float(payoff),
        "win_rate": float(n_wins / len(arr)) if len(arr) else 0.0,
    }


def compute_sample_uniqueness(df_events: pd.DataFrame, total_bars_index: pd.Index) -> pd.Series:
    """
    Computes average uniqueness weights for overlapping triple-barrier events.
    Guarantees that clustered signals do not over-weight gradient boosting loss functions.
    Reference: Advances in Financial Machine Learning, Ch. 4
    """
    # Build concurrency series across all timestamps
    concurrency = pd.Series(0, index=total_bars_index)

    for idx, row in df_events.iterrows():
        t0 = idx
        t1 = row["t1"]
        concurrency.loc[t0:t1] += 1

    concurrency = concurrency.replace(0, 1)

    # Compute uniqueness per event: average of 1 / concurrency across event lifetime
    uniqueness = pd.Series(index=df_events.index, dtype=float)

    for idx, row in df_events.iterrows():
        t0 = idx
        t1 = row["t1"]
        sub_c = concurrency.loc[t0:t1]
        u = (1.0 / sub_c).mean()
        uniqueness.loc[idx] = u

    # Normalize weights so sum equals number of samples
    norm_weights = uniqueness / uniqueness.mean()
    return norm_weights.fillna(1.0)


def generate_meta_labels(
    df: pd.DataFrame,
    primary_signals: pd.Series,
    pt_multiplier: float = TP_ATR_MULT,
    sl_multiplier: float = SL_ATR_MULT,
    max_holding_bars: int = MAX_HOLDING_BARS,
    atr_period: int = ATR_PERIOD,
) -> pd.DataFrame:
    """
    Meta-label for a primary signal (+1 long, -1 short).

    meta_label 1: the profit barrier was touched before the stop.
    meta_label 0: the stop was touched, both levels printed in one bar, or
    the vertical barrier expired. A profitable timeout is not a success.
    Events with no ATR or a hole in the forward bars are omitted.
    """
    events_idx = primary_signals[primary_signals != 0].index
    labeled = apply_triple_barrier(
        df,
        events_idx=events_idx,
        pt_multiplier=pt_multiplier,
        sl_multiplier=sl_multiplier,
        max_holding_bars=max_holding_bars,
        sides=primary_signals,
        atr_period=atr_period,
    )
    if labeled.empty:
        return pd.DataFrame(columns=["t1", "side", "entry_price", "meta_label"]).rename_axis("datetime")
    meta = pd.DataFrame(
        {
            "t1": labeled["t1"],
            "side": labeled["side"],
            "entry_price": labeled["entry_price"],
            "meta_label": (labeled["label"] == 1).astype(int),
        },
        index=labeled.index,
    )
    meta.index.name = "datetime"
    return meta


if __name__ == "__main__":
    print("[*] Self-testing triple_barrier module...")
    # Synthetic price series with random walk
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=1000, freq="1h")
    rets = np.random.normal(0.0002, 0.008, size=1000)
    prices = 40000.0 * np.exp(np.cumsum(rets))
    highs = prices * (1.0 + np.abs(np.random.normal(0, 0.003, size=1000)))
    lows = prices * (1.0 - np.abs(np.random.normal(0, 0.003, size=1000)))
    vols = np.random.uniform(10, 100, size=1000)

    df_synth = pd.DataFrame({
        "open": prices * 0.999,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": vols,
    }, index=dates)

    tb = apply_triple_barrier(df_synth, pt_multiplier=3.0, sl_multiplier=1.5, max_holding_bars=24)
    print(f"    Triple-Barrier Labels Generated: {len(tb)}")
    print(f"    Label counts:\n{tb['label'].value_counts()}")

    weights = compute_sample_uniqueness(tb, df_synth.index)
    print(f"    Average Uniqueness Weight: {weights.mean():.4f} (Min: {weights.min():.4f}, Max: {weights.max():.4f})")

    # Meta-label test
    signals = pd.Series(0, index=df_synth.index)
    signals.iloc[::20] = 1 # Long signal every 20 bars
    meta = generate_meta_labels(df_synth, signals, pt_multiplier=3.0, sl_multiplier=1.5)
    print(f"    Meta-Labels Generated: {len(meta)}, Success Rate: {meta['meta_label'].mean():.2%}")
    print("[✓] triple_barrier module tests passed successfully!")
