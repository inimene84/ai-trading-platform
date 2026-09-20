"""Moving-block bootstrap G4 gate: >=500 scenarios, worst-5% ruin < 5%."""

import numpy as np

from monte_carlo import MC_MIN_SCENARIOS, moving_block_bootstrap


def test_monte_carlo_default_scenario_count():
    assert MC_MIN_SCENARIOS >= 500


def test_monte_carlo_rejects_tiny_sample():
    out = moving_block_bootstrap(np.array([0.01, -0.01, 0.0]), n_scenarios=500)
    assert out["gate_ok"] is False
    assert out["n_scenarios"] == 0


def test_monte_carlo_stable_positive_edge_passes():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.004, 0.003, size=400)
    out = moving_block_bootstrap(rets, n_scenarios=500, seed=0)
    assert out["n_scenarios"] == 500
    assert out["gate_ok"] is True
    assert out["worst5_ruin_rate"] < 0.05


def test_monte_carlo_ruinous_path_fails_gate():
    rng = np.random.default_rng(1)
    rets = rng.normal(-0.03, 0.04, size=200)
    out = moving_block_bootstrap(rets, n_scenarios=500, seed=1)
    assert out["gate_ok"] is False
    assert out["ruin_rate"] >= 0.05 or out["worst5_ruin_rate"] >= 0.05
