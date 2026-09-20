"""Triple-barrier labels must use live hold/trail and net-of-cost returns."""

import inspect

import numpy as np
import pandas as pd

from barrier_config import MAX_HOLDING_BARS, TRAIL_ACTIVATION_ATR, TRAIL_ATR_MULT, round_trip_cost_pct
from triple_barrier import apply_triple_barrier, net_round_trip_return_pct


def test_apply_triple_barrier_defaults_are_live_geometry():
    sig = inspect.signature(apply_triple_barrier)
    assert sig.parameters["max_holding_bars"].default == MAX_HOLDING_BARS == 48
    assert sig.parameters["pt_multiplier"].default == 5.5
    assert sig.parameters["sl_multiplier"].default == 1.75
    assert sig.parameters["trail_activation_atr"].default == TRAIL_ACTIVATION_ATR == 2.2
    assert sig.parameters["trail_atr_mult"].default == TRAIL_ATR_MULT == 1.6
    assert sig.parameters["apply_costs"].default is True


def test_net_round_trip_is_below_gross():
    gross = 2.0
    net = net_round_trip_return_pct(gross, 48, apply_costs=True)
    assert net < gross
    assert abs((gross - net) - round_trip_cost_pct(48)) < 1e-9
    assert net_round_trip_return_pct(gross, 48, apply_costs=False) == gross


def test_trail_exits_before_giving_back_to_initial_stop():
    n = 300
    close = np.concatenate(
        [
            np.full(200, 100.0),
            np.linspace(100.0, 110.0, 20),
            np.linspace(110.0, 99.0, 80),
        ]
    )
    high = close + 0.05
    low = close - 0.05
    df = pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 10.0),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )
    events = df.index[[200]]
    tb = apply_triple_barrier(
        df,
        events_idx=events,
        pt_multiplier=5.5,
        sl_multiplier=1.75,
        max_holding_bars=48,
        trail_activation_atr=2.2,
        trail_atr_mult=1.6,
        apply_costs=True,
    )
    assert len(tb) == 1
    row = tb.iloc[0]
    assert row["touch_type"] in {"trail", "pt"}
    assert row["touch_type"] != "timeout"
    # Give-back to 99 would hit a 1.75 ATR stop if trail were off.
    assert float(row["net_ret"]) > -2.0
    assert "net_ret" in tb.columns
    assert float(row["net_ret"]) <= float(row["ret"])
