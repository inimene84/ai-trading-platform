"""Strict validation for TypeSafe System One responses.

Invalid schema is a hard failure. Callers must treat that as no signal.
"""

from __future__ import annotations

import math
from typing import Any

from backend.jev.questions import (
    CATALYST_LEVELS,
    PRICE_DIRECTIONS,
    SENTIMENT_LEVELS,
    TRADE_ACTIONS,
)


class JevSchemaError(ValueError):
    """The model response cannot be used as a trading advisory."""


def _finite(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise JevSchemaError(f"{field} is not a number") from exc
    if not math.isfinite(number):
        raise JevSchemaError(f"{field} is not finite")
    return number


def _unit_probability(value: Any, field: str) -> float:
    number = _finite(value, field)
    if number > 1.0 and number <= 100.0:
        number = number / 100.0
    if number < 0.0 or number > 1.0:
        raise JevSchemaError(f"{field} probability out of range")
    return number


def _probability_map(raw: Any, allowed: tuple[str, ...], field: str) -> dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        raise JevSchemaError(f"{field} probabilities missing")
    unknown = [key for key in raw if key not in allowed]
    if unknown:
        raise JevSchemaError(f"{field} has unknown keys: {unknown}")
    parsed = {key: _unit_probability(raw[key], f"{field}.{key}") for key in allowed if key in raw}
    if set(parsed) != set(allowed):
        raise JevSchemaError(f"{field} probabilities must cover {allowed}")
    total = sum(parsed.values())
    if total <= 0 or abs(total - 1.0) > 0.15:
        raise JevSchemaError(f"{field} probabilities do not sum to 1")
    scale = 1.0 / total
    return {key: parsed[key] * scale for key in allowed}


def _score_index(value: Any, levels: tuple[str, ...], field: str) -> tuple[float, str]:
    score = _finite(value, field)
    if score < -0.05 or score > (len(levels) - 1) + 0.05:
        raise JevSchemaError(f"{field} score out of range")
    index = min(len(levels) - 1, max(0, int(round(score))))
    return score, levels[index]


def _answer(answers: dict, name: str) -> dict:
    raw = answers.get(name)
    if not isinstance(raw, dict):
        raise JevSchemaError(f"missing answer {name}")
    return raw


def validate_system_one(payload: Any) -> dict[str, Any]:
    """Return a normalized answer dict or raise JevSchemaError."""
    if not isinstance(payload, dict):
        raise JevSchemaError("response is not an object")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise JevSchemaError("missing answers")

    trade = _answer(answers, "trade_action")
    action = str(trade.get("choice") or "").strip()
    if action not in TRADE_ACTIONS:
        raise JevSchemaError(f"trade_action choice invalid: {action!r}")
    trade_probs = _probability_map(trade.get("probabilities"), TRADE_ACTIONS, "trade_action")
    if "confidence" in trade and trade.get("confidence") is not None:
        confidence = _unit_probability(trade.get("confidence"), "trade_action.confidence")
    else:
        confidence = trade_probs[action]

    sentiment = _answer(answers, "sentiment_spectrum")
    sentiment_score, sentiment_label = _score_index(
        sentiment.get("score"), SENTIMENT_LEVELS, "sentiment_spectrum"
    )

    squeeze = _answer(answers, "is_short_squeeze_risk")
    squeeze_probability = _unit_probability(squeeze.get("noul"), "is_short_squeeze_risk.noul")

    catalyst = _answer(answers, "catalyst_impact")
    catalyst_score, catalyst_label = _score_index(
        catalyst.get("score"), CATALYST_LEVELS, "catalyst_impact"
    )

    direction_raw = _answer(answers, "price_direction")
    direction = str(direction_raw.get("choice") or "").strip()
    if direction not in PRICE_DIRECTIONS:
        raise JevSchemaError(f"price_direction choice invalid: {direction!r}")
    direction_probs = _probability_map(
        direction_raw.get("probabilities"), PRICE_DIRECTIONS, "price_direction"
    )
    ordered = sorted(direction_probs.values(), reverse=True)
    margin = ordered[0] - ordered[1]

    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {
        "model": payload.get("model"),
        "trade_action": action,
        "trade_confidence": confidence,
        "trade_probabilities": trade_probs,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "squeeze_probability": squeeze_probability,
        "catalyst_score": catalyst_score,
        "catalyst_label": catalyst_label,
        "price_direction": direction,
        "direction_probabilities": direction_probs,
        "direction_margin": margin,
        "usage": usage,
    }
