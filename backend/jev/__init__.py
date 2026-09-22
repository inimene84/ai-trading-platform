"""Jev typed-evaluation advisory layer.

Jev (TypeSafe System One) is a cheaper market-analysis alternative to the
LLM persona agents. Outputs are advisory only. API failures, timeouts, and
invalid schemas produce no signal — never a default buy.
"""

from backend.jev.opinion import evaluate_opinion, jev_analysis_enabled, should_skip_personas
from backend.jev.service import evaluate_symbol

__all__ = [
    "evaluate_opinion",
    "evaluate_symbol",
    "jev_analysis_enabled",
    "should_skip_personas",
]
