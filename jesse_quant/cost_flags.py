"""
Per-run cost-model toggles shared between the host-side runners (run_backtest.py) and the
strategies executing inside the Jesse container. Pure standard library so it imports on both.

storage/cost_model.json  -> {"zero_cost": true} disables fees/slippage/funding for the run
storage/cost_reports/    -> one JSON per strategy run with funding and slippage charged
"""

import json
import os
import time
from typing import Any, Dict

from barrier_config import (
    BASE_SLIPPAGE,
    FEE_RATE,
    FUNDING_INTERVAL_HOURS,
    FUNDING_RATE_8H,
    MAX_SLIPPAGE,
    ZERO_COST,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
COST_FLAG_PATH = os.getenv("JESSE_COST_FLAG_PATH") or os.path.join(REPO_ROOT, "storage", "cost_model.json")
COST_REPORT_DIR = os.getenv("JESSE_COST_REPORT_DIR") or os.path.join(REPO_ROOT, "storage", "cost_reports")


def read_cost_flags() -> Dict[str, Any]:
    """Merge barrier_config defaults with the optional per-run flag file."""
    flags: Dict[str, Any] = {
        "zero_cost": ZERO_COST,
        "funding_rate_8h": FUNDING_RATE_8H,
        "funding_interval_hours": FUNDING_INTERVAL_HOURS,
        "base_slippage": BASE_SLIPPAGE,
        "max_slippage": MAX_SLIPPAGE,
        "fee_rate": FEE_RATE,
    }
    if os.path.exists(COST_FLAG_PATH):
        try:
            with open(COST_FLAG_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for key in flags:
                if key in data and data[key] is not None:
                    flags[key] = data[key]
        except (OSError, ValueError):
            pass
    flags["zero_cost"] = bool(flags["zero_cost"])
    return flags


def write_cost_flags(zero_cost: bool, **overrides: Any) -> str:
    """Write the per-run flag file consumed by the strategies (used by run_backtest.py)."""
    os.makedirs(os.path.dirname(COST_FLAG_PATH), exist_ok=True)
    payload = {"zero_cost": bool(zero_cost), "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    payload.update(overrides)
    with open(COST_FLAG_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return COST_FLAG_PATH


def clear_cost_flags() -> None:
    if os.path.exists(COST_FLAG_PATH):
        os.remove(COST_FLAG_PATH)
