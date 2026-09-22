"""T+1 scoring template for Jev direction forecasts.

The source stock study published roughly 45 percent accuracy. This harness
only scores a supplied prediction series against the next bar. It does not
call Jev and it is not a promotion gate. Live capital still goes through
DSR, PBO, and the existing conformal veto.
"""

from __future__ import annotations

import math
from typing import Any

# Flat bands are fractions of price, fixed in code. Day ±0.3%, 30-day ±2%.
HORIZON_FLAT_FRACTION = {
    "day": 0.003,
    "30d": 0.02,
}


def horizon_flat_bps(horizon: str) -> float:
    if horizon not in HORIZON_FLAT_FRACTION:
        raise ValueError(f"unknown horizon: {horizon}")
    return HORIZON_FLAT_FRACTION[horizon] * 10_000.0


def wilson_interval(hits: int, samples: int, z: float = 1.96) -> tuple[float, float] | None:
    if samples <= 0:
        return None
    proportion = hits / samples
    z2 = z * z
    denominator = 1.0 + z2 / samples
    centre = proportion + z2 / (2.0 * samples)
    margin = z * math.sqrt((proportion * (1.0 - proportion) + z2 / (4.0 * samples)) / samples)
    low = max(0.0, (centre - margin) / denominator)
    high = min(1.0, (centre + margin) / denominator)
    return (low, high)


def multiclass_brier(rows: list[tuple[dict[str, float], str]]) -> float | None:
    if not rows:
        return None
    total = 0.0
    counted = 0
    for probabilities, outcome in rows:
        if outcome not in probabilities:
            continue
        weight = sum(float(value) for value in probabilities.values()) or 1.0
        for name, value in probabilities.items():
            target = 1.0 if name == outcome else 0.0
            predicted = float(value) / weight
            total += (predicted - target) ** 2
        counted += 1
    if not counted:
        return None
    return total / counted


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
    probability_rows: list[tuple[dict[str, float], str]] | None = None,
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
            "brier": None,
            "wilson95": None,
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
    interval = wilson_interval(hits, counted)
    brier = multiclass_brier(probability_rows or [])
    return {
        "samples": counted,
        "hits": hits,
        "accuracy": None if accuracy is None else round(accuracy, 4),
        "by_direction": by_direction,
        "brier": None if brier is None else round(brier, 4),
        "wilson95": None if interval is None else [round(interval[0], 4), round(interval[1], 4)],
        "reason": "Research score only. A hit rate is not a live-trading promise.",
    }
