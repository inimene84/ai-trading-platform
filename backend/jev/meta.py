"""Research meta-label for Jev direction calls.

The triple barrier and the take/skip rule are offline labels. They do not
submit orders. order_size_fraction is fixed at zero so a future caller cannot
turn a Jev probability into a position.
"""

from __future__ import annotations

import math
from typing import Any


def distribution_entropy(probabilities: dict[str, float]) -> float:
    total = 0.0
    for probability in probabilities.values():
        value = float(probability)
        if value > 0:
            total -= value * math.log(value)
    return total


def triple_barrier(
    closes: list[float],
    entry: int,
    upper_frac: float,
    lower_frac: float,
    horizon: int,
) -> int:
    """+1 upper hit first, -1 lower hit first, 0 if the time barrier expires."""
    if entry < 0 or entry >= len(closes) or horizon < 1:
        return 0
    if upper_frac <= 0 or lower_frac <= 0:
        return 0
    start = float(closes[entry])
    if start <= 0:
        return 0
    upper = start * (1.0 + upper_frac)
    lower = start * (1.0 - lower_frac)
    end = min(len(closes) - 1, entry + horizon)
    for index in range(entry + 1, end + 1):
        price = float(closes[index])
        if price >= upper:
            return 1
        if price <= lower:
            return -1
    return 0


def meta_take(
    *,
    labels: int,
    min_labels: int,
    margin: float,
    min_margin: float,
    probabilities: dict[str, float],
    conformal_veto: bool,
) -> dict[str, Any]:
    """Whether a research note would act. Never a live sizing input."""
    if labels < min_labels:
        return {"take": False, "reason": "insufficient labeled outcomes", "suggested_fraction": 0.0}
    if conformal_veto:
        return {"take": False, "reason": "adaptive conformal set is too wide", "suggested_fraction": 0.0}
    if margin < min_margin:
        return {"take": False, "reason": "direction margin below floor", "suggested_fraction": 0.0}
    entropy = distribution_entropy(probabilities)
    if entropy > math.log(3):
        return {"take": False, "reason": "direction distribution is too diffuse", "suggested_fraction": 0.0}
    return {
        "take": True,
        "reason": "research gate passed; live sizing remains disabled",
        "suggested_fraction": 0.0,
    }


def order_size_fraction(**_ignored: Any) -> float:
    """Jev does not size positions. This stays zero regardless of inputs."""
    return 0.0
