"""Trainer errors that CI/unit tests can import without LightGBM/sklearn."""

from __future__ import annotations


class TooFewEventsError(RuntimeError):
    """Raised when QuantumAI-event sampling is below the trainer floor."""

    def __init__(self, n_events: int, min_events: int) -> None:
        self.n_events = int(n_events)
        self.min_events = int(min_events)
        super().__init__(
            f"only {self.n_events} QuantumAI events (min {self.min_events})"
        )
