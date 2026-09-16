#!/usr/bin/env python3
"""
Persistent trial registry for Deflated Sharpe Ratio (DSR) corrections.

The DSR correction for selection bias needs the *total* number of strategy/model
configurations that were ever evaluated against the same data, not just the trials of
the current run (Bailey & Lopez de Prado, 2014). This module keeps a cumulative counter
per research scope in storage/trial_registry.json so the count survives across runs.

Scopes are free-form strings such as "ml:BTC-USDT:1h" or
"optimizer:QuantumAIStrategy:BTC-USDT:1h". `total_trials()` returns the scope total plus
a share of the global history when requested, so that the count is never smaller than
what the current run alone evaluated.
"""

import argparse
import json
import os
import threading
import time
from typing import Any, Dict, Optional

_LOCK = threading.Lock()


def _default_registry_path() -> str:
    # Keep the registry next to this checkout so worktrees do not share/clobber counters.
    repo_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(repo_root, "storage", "trial_registry.json")


REGISTRY_PATH = os.getenv("TRIAL_REGISTRY_PATH", _default_registry_path())

# Documented estimate of trials evaluated *before* the registry existed, derived from the
# project walkthroughs: ML v1 (fixed-horizon, 5 folds x 3 symbols), ML v2 triple-barrier
# (5 folds + 1 final fit per symbol, BTC trained twice), GA optimizer run of 16 trials.
# These are recorded once, tagged as "historical_estimate", so DSR does not restart at 1.
# Pre-registry trials reconstructed from the walkthrough documents plus the 18 grid
# trials recorded by the interim storage/models/_trial_ledger.json (3 BTC runs x 6 configs).
_HISTORICAL_SEED: Dict[str, int] = {
    "ml:BTC-USDT:1h": 5 + 6 + 6 + 18,
    "ml:ETH-USDT:1h": 5 + 6,
    "ml:SOL-USDT:1h": 5 + 6,
    "optimizer:QuantumAIStrategy:BTC-USDT:1h": 16,
}


def _empty() -> Dict[str, Any]:
    return {"version": 1, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "scopes": {}, "events": []}


def load_registry(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or REGISTRY_PATH
    if not os.path.exists(path):
        reg = _empty()
        for scope, n in _HISTORICAL_SEED.items():
            reg["scopes"][scope] = {"total_trials": int(n), "runs": 1}
            reg["events"].append({
                "ts": reg["created_at"],
                "scope": scope,
                "n_trials": int(n),
                "note": "historical_estimate (pre-registry runs documented in walkthroughs)",
            })
        return reg
    try:
        with open(path, "r", encoding="utf-8") as f:
            reg = json.load(f)
        reg.setdefault("scopes", {})
        reg.setdefault("events", [])
        return reg
    except (OSError, json.JSONDecodeError):
        return _empty()


def save_registry(reg: Dict[str, Any], path: Optional[str] = None) -> None:
    path = path or REGISTRY_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)
    os.replace(tmp, path)


def record_trials(scope: str, n_trials: int, note: str = "", path: Optional[str] = None) -> int:
    """Add `n_trials` to `scope` and return the new cumulative total for that scope."""
    path = path or REGISTRY_PATH
    n_trials = int(max(0, n_trials))
    with _LOCK:
        reg = load_registry(path)
        entry = reg["scopes"].setdefault(scope, {"total_trials": 0, "runs": 0})
        entry["total_trials"] = int(entry.get("total_trials", 0)) + n_trials
        entry["runs"] = int(entry.get("runs", 0)) + 1
        reg["events"].append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scope": scope,
            "n_trials": n_trials,
            "note": note,
        })
        # Keep the event log bounded.
        if len(reg["events"]) > 2000:
            reg["events"] = reg["events"][-2000:]
        save_registry(reg, path)
        return int(entry["total_trials"])


def total_trials(scope: str, path: Optional[str] = None) -> int:
    """Cumulative trials recorded for `scope` (0 if the scope is unknown)."""
    path = path or REGISTRY_PATH
    reg = load_registry(path)
    return int(reg["scopes"].get(scope, {}).get("total_trials", 0))


def global_total(path: Optional[str] = None) -> int:
    path = path or REGISTRY_PATH
    reg = load_registry(path)
    return int(sum(int(v.get("total_trials", 0)) for v in reg["scopes"].values()))


def effective_trials(scope: str, current_run_trials: int, path: Optional[str] = None) -> int:
    """
    Number of trials to feed into the DSR for the current run: everything previously
    recorded for the scope plus the trials of this run, never below the current run.
    """
    prior = total_trials(scope, path)
    return int(max(current_run_trials, prior + current_run_trials))


def scope_summary(path: Optional[str] = None) -> Dict[str, int]:
    reg = load_registry(path)
    return {k: int(v.get("total_trials", 0)) for k, v in reg["scopes"].items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect or update the DSR trial registry")
    parser.add_argument("--scope", help="Scope to add trials to")
    parser.add_argument("--add", type=int, default=0, help="Number of trials to add")
    parser.add_argument("--note", default="manual", help="Note for the event log")
    args = parser.parse_args()

    if args.scope and args.add:
        total = record_trials(args.scope, args.add, args.note)
        print(f"[✓] {args.scope}: cumulative trials = {total}")
    print(json.dumps({"registry": REGISTRY_PATH, "scopes": scope_summary(), "global_total": global_total()}, indent=2))
