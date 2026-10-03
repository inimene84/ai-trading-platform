"""Moving-block bootstrap stress test for promotable GPU configs.

Master-plan G4: >=500 scenarios; ruin rate in the worst 5% of paths < 5%.
This module never talks to brokers and never promotes an artifact.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

MC_MIN_SCENARIOS = 500
RUIN_DRAWDOWN = 0.50
WORST5_RUIN_MAX = 0.05


def moving_block_bootstrap(
    returns: np.ndarray,
    *,
    n_scenarios: int = MC_MIN_SCENARIOS,
    block_size: Optional[int] = None,
    ruin_drawdown: float = RUIN_DRAWDOWN,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap equity curves from *net* trade returns (fractions, not percent)."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 8:
        return {
            "n_scenarios": 0,
            "n_returns": n,
            "block_size": 0,
            "ruin_rate": 1.0,
            "worst5_ruin_rate": 1.0,
            "median_max_drawdown": 1.0,
            "gate_ok": False,
            "reason": f"too few net returns for Monte Carlo ({n} < 8)",
        }
    if block_size is None:
        block_size = max(4, min(20, n // 10 or 4))
    block_size = int(max(2, min(int(block_size), n)))
    rng = np.random.default_rng(int(seed))
    n_blocks = int(np.ceil(n / block_size))
    max_start = max(1, n - block_size + 1)
    terminals = np.empty(int(n_scenarios), dtype=float)
    ruins = np.empty(int(n_scenarios), dtype=bool)
    max_dds = np.empty(int(n_scenarios), dtype=float)
    for i in range(int(n_scenarios)):
        starts = rng.integers(0, max_start, size=n_blocks)
        path = np.concatenate([arr[int(s) : int(s) + block_size] for s in starts])[:n]
        equity = np.cumprod(1.0 + path)
        peak = np.maximum.accumulate(np.maximum(equity, 1e-12))
        dd = 1.0 - equity / peak
        max_dd = float(np.nanmax(dd)) if dd.size else 1.0
        ruined = bool(float(np.nanmin(equity)) <= (1.0 - float(ruin_drawdown)) or max_dd >= float(ruin_drawdown))
        terminals[i] = float(equity[-1]) if equity.size else 0.0
        ruins[i] = ruined
        max_dds[i] = max_dd
    worst5_n = max(1, int(np.ceil(0.05 * int(n_scenarios))))
    worst_idx = np.argsort(terminals)[:worst5_n]
    worst5_ruin = float(np.mean(ruins[worst_idx]))
    ruin_rate = float(np.mean(ruins))
    median_dd = float(np.median(max_dds))
    gate_ok = worst5_ruin < float(WORST5_RUIN_MAX) and ruin_rate < float(WORST5_RUIN_MAX)
    reason = (
        f"PASS ruin={ruin_rate:.1%} worst5={worst5_ruin:.1%} median_dd={median_dd:.1%}"
        if gate_ok
        else f"ruin {ruin_rate:.1%} / worst-5% ruin {worst5_ruin:.1%} (gate < {WORST5_RUIN_MAX:.0%})"
    )
    return {
        "n_scenarios": int(n_scenarios),
        "n_returns": n,
        "block_size": int(block_size),
        "ruin_rate": ruin_rate,
        "worst5_ruin_rate": worst5_ruin,
        "median_max_drawdown": median_dd,
        "ruin_drawdown": float(ruin_drawdown),
        "gate_ok": bool(gate_ok),
        "reason": reason,
    }
