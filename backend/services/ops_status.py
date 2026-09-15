"""Ops-facing snapshot for /trading/status and /sentry/status.

Combines broker env / equity scope (no secrets) with LLM router last-error
so split-book and LLM degradation are visible without SSH.
"""

from __future__ import annotations

from typing import Any

from backend.llm.router import get_llm_router_status
from backend.services.equity_scope import describe_equity_books
from backend.services.sentry_resume import auto_resume_policy


def trading_ops_snapshot() -> dict[str, Any]:
    return {
        "equity_books": describe_equity_books(),
        "llm_router": get_llm_router_status(),
        "sentry_auto_resume": auto_resume_policy(),
    }
