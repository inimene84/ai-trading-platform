"""T+1 scoring template for Jev direction forecasts.

The source stock study published roughly 45 percent accuracy. This harness
only scores a supplied prediction series against the next bar. It does not
call Jev and it is not a promotion gate. Live capital still goes through
DSR, PBO, and the existing conformal veto.
"""

from __future__ import annotations

from typing import Any


def realized_direction(previous_close: float, next_close: float, flat_bps: float = 8.0) -> str:
    if previous_close <= 0:
        return "FLAT"
    change_bps = (next_close - previous_close) / previous_close * 10_000.0
    if change_bps > flat_bps:
        return "UP"
    if change_bps < -flat_bps:
        return "DOWN"
    return "FLAT"


def score_t_plus_one(
    bars: list[dict],
    predictions: list[str],
    flat_bps: float = 8.0,
) -> dict[str, Any]:
    """Score `predictions[i]` against the close move from bar i to bar i+1.

    `predictions` must be one shorter than `bars`. Each prediction is the
    forecast that was allowed to see only `bars[: i + 1]`.
    """
    if len(bars) < 2 or len(predictions) != len(bars) - 1:
        return {
            "samples": 0,
            "hits": 0,
            "accuracy": None,
            "by_direction": {},
            "reason": "predictions must align to bars[:-1]",
        }
    hits = 0
    counted = 0
    by_direction: dict[str, dict[str, int]] = {
        "UP": {"n": 0, "hit": 0},
        "FLAT": {"n": 0, "hit": 0},
        "DOWN": {"n": 0, "hit": 0},
    }
    for index, predicted in enumerate(predictions):
        label = str(predicted or "").upper()
        if label not in by_direction:
            continue
        previous = float(bars[index]["close"])
        nxt = float(bars[index + 1]["close"])
        actual = realized_direction(previous, nxt, flat_bps=flat_bps)
        by_direction[label]["n"] += 1
        counted += 1
        if actual == label:
            hits += 1
            by_direction[label]["hit"] += 1
    accuracy = (hits / counted) if counted else None
    return {
        "samples": counted,
        "hits": hits,
        "accuracy": None if accuracy is None else round(accuracy, 4),
        "by_direction": by_direction,
        "reason": "",
    }
