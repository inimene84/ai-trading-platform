"""Display-only calibration for Jev probabilities.

Temperature scaling and a small isotonic map are fit from labeled journal
rows. Until the label count reaches JEV_CALIBRATION_MIN_LABELS, callers must
keep the raw probability as a display hint and must not size risk from it.
"""

from __future__ import annotations

import math
from typing import Any

from backend.jev.config import calibration_min_labels


def _clip(probability: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, probability))


def temperature_scale(probabilities: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = {key: _clip(float(value)) ** (1.0 / temperature) for key, value in probabilities.items()}
    total = sum(scaled.values()) or 1.0
    return {key: weight / total for key, weight in scaled.items()}


def fit_temperature(rows: list[tuple[dict[str, float], str]]) -> float | None:
    """Grid-search a temperature that minimizes negative log likelihood."""
    if len(rows) < 8:
        return None
    best_t = 1.0
    best_nll = math.inf
    step = 0.25
    candidate = 0.5
    while candidate <= 3.0 + 1e-9:
        nll = 0.0
        for probabilities, winner in rows:
            if winner not in probabilities:
                continue
            scaled = temperature_scale(probabilities, candidate)
            nll -= math.log(_clip(scaled[winner]))
        if nll < best_nll:
            best_nll = nll
            best_t = candidate
        candidate = round(candidate + step, 2)
    return best_t


def fit_isotonic(pairs: list[tuple[float, int]]) -> list[tuple[float, float]] | None:
    """Pool-adjacent-violators map from predicted probability to observed frequency."""
    usable = [(float(prob), 1 if int(label) > 0 else 0) for prob, label in pairs]
    if len(usable) < 8:
        return None
    usable.sort(key=lambda item: item[0])
    blocks: list[dict[str, float]] = [
        {"lo": prob, "hi": prob, "sum": float(label), "n": 1.0} for prob, label in usable
    ]
    index = 0
    while index < len(blocks) - 1:
        left = blocks[index]["sum"] / blocks[index]["n"]
        right = blocks[index + 1]["sum"] / blocks[index + 1]["n"]
        if left <= right + 1e-12:
            index += 1
            continue
        merged = {
            "lo": blocks[index]["lo"],
            "hi": blocks[index + 1]["hi"],
            "sum": blocks[index]["sum"] + blocks[index + 1]["sum"],
            "n": blocks[index]["n"] + blocks[index + 1]["n"],
        }
        blocks[index:index + 2] = [merged]
        index = max(0, index - 1)
    return [(block["hi"], block["sum"] / block["n"]) for block in blocks]


def apply_isotonic(probability: float, curve: list[tuple[float, float]]) -> float:
    if not curve:
        return probability
    if probability <= curve[0][0]:
        return curve[0][1]
    for index in range(1, len(curve)):
        right_x, right_y = curve[index]
        left_x, left_y = curve[index - 1]
        if probability <= right_x:
            span = right_x - left_x
            if span <= 0:
                return right_y
            weight = (probability - left_x) / span
            return left_y + weight * (right_y - left_y)
    return curve[-1][1]


def calibration_report(labeled: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether raw Jev probabilities may leave the display path."""
    minimum = calibration_min_labels()
    choice_rows: list[tuple[dict[str, float], str]] = []
    noul_pairs: list[tuple[float, int]] = []
    for row in labeled:
        answers = row.get("answers") or {}
        probs = answers.get("trade_probabilities") or answers.get("direction_probabilities")
        action = answers.get("trade_action") or answers.get("price_direction")
        if isinstance(probs, dict) and action in probs:
            choice_rows.append((probs, str(action)))
        squeeze = answers.get("squeeze_probability")
        if squeeze is not None:
            noul_pairs.append((float(squeeze), int(row.get("label") or 0)))
    ready = len(labeled) >= minimum
    temperature = fit_temperature(choice_rows) if ready else None
    isotonic = fit_isotonic(noul_pairs) if ready else None
    return {
        "labels": len(labeled),
        "min_labels": minimum,
        "ready": ready,
        "sizing_allowed": False,
        "display_only": not ready,
        "temperature": temperature,
        "isotonic_points": len(isotonic or []),
        "note": (
            "Raw Jev probabilities are model outputs, not historical win rates. "
            "Position sizing from these numbers stays disabled."
        ),
    }


def display_probability(
    probabilities: dict[str, float],
    temperature: float | None,
) -> dict[str, Any]:
    if temperature is None:
        return {"probabilities": probabilities, "calibrated": False}
    return {"probabilities": temperature_scale(probabilities, temperature), "calibrated": True}
