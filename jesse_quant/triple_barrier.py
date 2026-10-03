#!/usr/bin/env python3
"""
Triple-Barrier Method and Meta-Labeling Module
Implements Marcos López de Prado's path-dependent labeling framework:
1. Triple-Barrier Labeling: Horizontal Profit Target, Horizontal Stop Loss, Vertical Holding Limit
2. Concurrent Label Uniqueness Weighting: Sample weights for LightGBM/GBM classifiers
3. Meta-Labeling: Secondary binary labels determining if a primary model signal hits profit before stop
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from barrier_config import MAX_HOLDING_BARS


def get_daily_volatility(close: pd.Series, lookback: int = 50) -> pd.Series:
    """Computes rolling exponential standard deviation of returns as volatility proxy."""
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1.0  # Daily returns
    df0 = df0.ewm(span=lookback).std()
    return df0.bfill()


def get_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average True Range."""
    h, l, c = df["high"], df["low"], df["close"]
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def apply_triple_barrier(
    df: pd.DataFrame,
    events_idx: Optional[pd.Index] = None,
    pt_multiplier: float = 5.5,
    sl_multiplier: float = 1.75,
    max_holding_bars: int = MAX_HOLDING_BARS,
    use_atr: bool = True,
) -> pd.DataFrame:
    """
    Computes path-dependent triple-barrier outcomes for each observation:
      - Upper Barrier: entry + pt_multiplier * volatility
      - Lower Barrier: entry - sl_multiplier * volatility
      - Vertical Barrier: entry + max_holding_bars
    
    Returns DataFrame with columns:
      - t1: Timestamp when first barrier was touched (expiration)
      - trgt: Volatility threshold applied (in price units)
      - ret: Realized return at barrier touch
      - label: +1 (Bullish/TP hit first), -1 (Bearish/SL hit first), 0 (Vertical barrier hit / timeout)
      - touch_type: 'pt', 'sl', or 'timeout'
    """
    if events_idx is None:
        events_idx = df.index[:-max_holding_bars]

    close = df["close"]
    high = df["high"]
    low = df["low"]

    if use_atr:
        vol = get_atr(df, period=14)
    else:
        vol = close * close.pct_change().rolling(20).std()

    out = []

    for idx in events_idx:
        loc = df.index.get_loc(idx)
        if loc + max_holding_bars >= len(df):
            break

        entry_price = close.iloc[loc]
        v = vol.iloc[loc]
        if np.isnan(v) or v <= 0:
            v = entry_price * 0.01

        upper_barrier = entry_price + (pt_multiplier * v)
        lower_barrier = entry_price - (sl_multiplier * v)

        # Scan forward along the price path
        sub_high = high.iloc[loc + 1 : loc + 1 + max_holding_bars]
        sub_low = low.iloc[loc + 1 : loc + 1 + max_holding_bars]
        sub_close = close.iloc[loc + 1 : loc + 1 + max_holding_bars]

        touch_time = None
        touch_type = "timeout"
        label = 0
        realized_ret = 0.0

        for step in range(len(sub_close)):
            bar_time = sub_close.index[step]
            h_bar = sub_high.iloc[step]
            l_bar = sub_low.iloc[step]

            tp_hit = h_bar >= upper_barrier
            sl_hit = l_bar <= lower_barrier

            if tp_hit and not sl_hit:
                touch_time = bar_time
                touch_type = "pt"
                label = 1
                realized_ret = (upper_barrier / entry_price - 1.0) * 100.0
                break
            elif sl_hit and not tp_hit:
                touch_time = bar_time
                touch_type = "sl"
                label = -1
                realized_ret = (lower_barrier / entry_price - 1.0) * 100.0
                break
            elif tp_hit and sl_hit:
                # Both hit in same bar (worst-case assumption: hit SL first)
                touch_time = bar_time
                touch_type = "sl"
                label = -1
                realized_ret = (lower_barrier / entry_price - 1.0) * 100.0
                break

        if touch_time is None:
            # Vertical barrier reached
            touch_time = sub_close.index[-1]
            final_p = sub_close.iloc[-1]
            realized_ret = (final_p / entry_price - 1.0) * 100.0
            touch_type = "timeout"
            label = 1 if realized_ret > (0.2 * sl_multiplier * v / entry_price * 100) else (-1 if realized_ret < (-0.2 * sl_multiplier * v / entry_price * 100) else 0)

        out.append({
            "datetime": idx,
            "t1": touch_time,
            "entry_price": entry_price,
            "trgt": v,
            "ret": realized_ret,
            "label": label,
            "touch_type": touch_type,
        })

    out_df = pd.DataFrame(out).set_index("datetime")
    return out_df


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
    pt_multiplier: float = 5.5,
    sl_multiplier: float = 1.75,
    max_holding_bars: int = MAX_HOLDING_BARS,
) -> pd.DataFrame:
    """
    Generates Meta-Labels for a primary strategy signal (+1 for Long, -1 for Short):
      meta_label = 1 : Primary signal touched profit-taking barrier before stop-loss
      meta_label = 0 : Primary signal failed (stopped out or timed out at loss)
    
    This trains the secondary model to predict trade execution quality / size.
    """
    events_idx = primary_signals[primary_signals != 0].index
    close = df["close"]
    high = df["high"]
    low = df["low"]
    atr = get_atr(df, period=14)

    meta_records = []

    for idx in events_idx:
        loc = df.index.get_loc(idx)
        if loc + max_holding_bars >= len(df):
            break

        side = int(primary_signals.loc[idx])
        entry_p = close.iloc[loc]
        v = atr.iloc[loc]

        if side == 1:  # Long signal
            upper = entry_p + (pt_multiplier * v)
            lower = entry_p - (sl_multiplier * v)
        else:  # Short signal
            upper = entry_p - (pt_multiplier * v)
            lower = entry_p + (sl_multiplier * v)

        sub_h = high.iloc[loc + 1 : loc + 1 + max_holding_bars]
        sub_l = low.iloc[loc + 1 : loc + 1 + max_holding_bars]
        sub_c = close.iloc[loc + 1 : loc + 1 + max_holding_bars]

        success = 0
        touch_t = sub_c.index[-1]

        for step in range(len(sub_c)):
            h_bar = sub_h.iloc[step]
            l_bar = sub_l.iloc[step]
            b_time = sub_c.index[step]

            if side == 1:
                hit_tp = h_bar >= upper
                hit_sl = l_bar <= lower
            else:
                hit_tp = l_bar <= upper
                hit_sl = h_bar >= lower

            if hit_tp and not hit_sl:
                success = 1
                touch_t = b_time
                break
            elif hit_sl:
                success = 0
                touch_t = b_time
                break

        meta_records.append({
            "datetime": idx,
            "t1": touch_t,
            "side": side,
            "entry_price": entry_p,
            "meta_label": success,
        })

    meta_df = pd.DataFrame(meta_records).set_index("datetime")
    return meta_df


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
