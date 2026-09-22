"""Purged CV embargo/purge must cover max_holding_bars (48 on 1h) and lookback."""

import numpy as np
import pandas as pd

from embargo_audit import (
    DEFAULT_FEATURE_LOOKBACK_BARS,
    audit_from_validation_block,
    audit_purged_embargo,
    embargo_bars_for_n,
    required_embargo_bars,
)
from validation_metrics import PurgedKFold


def test_required_embargo_is_max_of_holding_and_lookback():
    assert required_embargo_bars(48, 200) == 200
    assert required_embargo_bars(48, 24) == 48


def test_percent_embargo_of_1_percent_fails_48_bar_1h_book():
    # 2000 bars * 1% = 20 < 48. This is the leak the audit must catch.
    audit = audit_from_validation_block(
        {"embargo_fraction": 0.01, "purge_horizon": 48},
        max_holding_bars=48,
        longest_lookback_bars=48,
        n_samples=2000,
        timeframe="1h",
    )
    assert audit.ok is False
    assert any("embargo" in reason for reason in audit.reasons)


def test_audit_passes_when_embargo_and_purge_cover_horizons():
    required = required_embargo_bars(48, DEFAULT_FEATURE_LOOKBACK_BARS)
    audit = audit_purged_embargo(
        embargo_bars=required,
        purge_horizon_bars=48,
        max_holding_bars=48,
        longest_lookback_bars=DEFAULT_FEATURE_LOOKBACK_BARS,
        timeframe="1h",
    )
    assert audit.ok is True
    assert audit.required_embargo_bars >= 48


def test_purged_kfold_embargo_bars_excludes_buffer():
    n = 400
    embargo = 48
    pkf = PurgedKFold(n_splits=4, embargo_bars=embargo)
    X = np.arange(n)
    for train_idx, test_idx in pkf.split(X):
        assert set(train_idx).isdisjoint(set(test_idx))
        test_max = int(max(test_idx))
        buffer = set(range(test_max + 1, min(n, test_max + 1 + embargo)))
        assert set(train_idx).isdisjoint(buffer)


def test_datetime_embargo_is_bars_not_event_count():
    """Sparse events: 48 embargo bars = 48 hours, not 48 events."""
    idx = pd.date_range("2024-01-01", periods=20, freq="6h")
    frame = pd.DataFrame({"x": np.arange(20)}, index=idx)
    pkf = PurgedKFold(n_splits=2, embargo_bars=48, bar_timedelta=pd.Timedelta(hours=1))
    train_idx, test_idx = next(pkf.split(frame))
    test_end_ts = idx[int(max(test_idx))]
    cutoff = test_end_ts + pd.Timedelta(hours=48)
    embargoed = {i for i, ts in enumerate(idx) if test_end_ts < ts <= cutoff}
    assert len(embargoed) < 48
    assert set(train_idx).isdisjoint(embargoed)
    later = {i for i, ts in enumerate(idx) if ts > cutoff}
    assert later
    assert later.issubset(set(train_idx))


def test_embargo_bars_for_n_never_below_required():
    assert embargo_bars_for_n(500, max_holding_bars=48, longest_lookback_bars=200) == 200
    assert embargo_bars_for_n(
        50_000, max_holding_bars=48, longest_lookback_bars=200, embargo_pct=0.01
    ) == 500
