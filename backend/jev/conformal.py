"""Adaptive conformal sets for Jev direction probabilities.

Split conformal assumes exchangeable residuals. Trade decisions are a time
series, so the threshold moves online: after each labeled outcome the
quantile steps by alpha minus the miss indicator. With too little history
the veto stays off and the separate calibration gate still blocks sizing.
"""

from __future__ import annotations

from typing import Any


class AdaptiveConformal:
    def __init__(self, alpha: float = 0.1, min_history: int = 30, step: float = 0.05, window: int = 500) -> None:
        self.alpha = alpha
        self.min_history = min_history
        self.step = step
        self.window = window
        self.threshold = 1.0
        self._scores: list[float] = []
        self._covered = 0
        self._total = 0

    def observe(self, nonconformity: float, covered: bool) -> None:
        self._scores.append(float(nonconformity))
        if len(self._scores) > self.window:
            self._scores = self._scores[-self.window:]
        self._total += 1
        if covered:
            self._covered += 1
        miss = 0.0 if covered else 1.0
        self.threshold = min(1.0, max(0.0, self.threshold + self.step * (self.alpha - miss)))

    def prediction_set(self, probabilities: dict[str, float]) -> list[str]:
        chosen = [
            name for name, probability in probabilities.items()
            if (1.0 - float(probability)) <= self.threshold + 1e-12
        ]
        return sorted(chosen)

    def veto(self, probabilities: dict[str, float]) -> bool:
        if len(self._scores) < self.min_history:
            return False
        return len(self.prediction_set(probabilities)) > 2

    def snapshot(self) -> dict[str, Any]:
        coverage = None if self._total == 0 else self._covered / self._total
        return {
            "history": len(self._scores),
            "min_history": self.min_history,
            "threshold": round(self.threshold, 4),
            "empirical_coverage": coverage,
            "ready": len(self._scores) >= self.min_history,
        }


_stores: dict[str, AdaptiveConformal] = {}


def conformal_for(symbol: str) -> AdaptiveConformal:
    key = symbol.upper()
    store = _stores.get(key)
    if store is None:
        store = AdaptiveConformal()
        _stores[key] = store
    return store


def reset_conformal() -> None:
    _stores.clear()
