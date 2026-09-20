"""FracDiff is causal and persists per-symbol d on the artifact."""

import numpy as np
import pandas as pd

from fracdiff import (
    DEFAULT_D,
    fracdiff_series,
    fracdiff_weights,
    resolve_artifact_d,
    select_fracdiff_d,
)


def test_fracdiff_weights_are_finite_and_longest_at_current_bar():
    weights = fracdiff_weights(0.4)
    assert len(weights) >= 2
    assert np.isfinite(weights).all()
    assert abs(weights[-1]) == 1.0  # current observation coefficient


def test_fracdiff_is_causal_on_a_price_shock():
    prices = pd.Series(np.linspace(100.0, 110.0, 80))
    shocked = prices.copy()
    shocked.iloc[40] = 130.0
    base = fracdiff_series(prices, 0.4)
    after = fracdiff_series(shocked, 0.4)
    # Bars strictly before the shock must be unchanged.
    assert np.allclose(base.iloc[:40].to_numpy(), after.iloc[:40].to_numpy(), equal_nan=True)
    # The shock may affect t>=40 only.
    assert after.iloc[40] != base.iloc[40] or np.isnan(base.iloc[40])


def test_select_fracdiff_d_stays_in_research_band():
    rng = np.random.default_rng(7)
    # Integrated noise: non-stationary level that FFD can stationarize.
    walk = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 0.8, 400)))
    chosen = select_fracdiff_d(walk)
    assert 0.35 <= chosen["d"] <= 0.55
    assert chosen["d"] in {round(x, 4) for x in np.arange(0.35, 0.56, 0.05)}


def test_resolve_artifact_d_prefers_per_symbol_and_never_regrids():
    artifact = {
        "fracdiff_d": 0.40,
        "fracdiff_d_by_symbol": {"ETH-USDT": 0.50},
        "metrics": {"fracdiff_d": 0.35},
    }
    assert resolve_artifact_d(artifact, symbol="ETH-USDT") == 0.50
    assert resolve_artifact_d(artifact, symbol="BTC-USDT") == 0.40
    assert resolve_artifact_d({}, symbol="BTC-USDT") == DEFAULT_D
