"""Market-research judgment card over already gathered evidence.

Jev classifies the evidence. It does not fetch it, and a missing or invalid
response is no classification rather than a default "higher". The card never
influences the trading book and never sizes an order.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import httpx

from backend.jev.client import JevUnavailable
from backend.jev.config import env_flag, env_int, jev_base_url, jev_model, jev_timeout_seconds, typesafe_api_key
from backend.jev.evidence import EVIDENCE_CATEGORIES, select_passages
from backend.jev.gateway import circuit_open, note_credit_failure
from backend.jev.meta import order_size_fraction
from backend.jev.schema import JevSchemaError

logger = logging.getLogger(__name__)

TICKER_RE = re.compile(r"^[A-Z0-9.\-=]{1,15}$")
HIGHER_OUTCOMES = ("HIGHER", "NOT_HIGHER")
CATEGORY_BIAS = ("BULLISH", "NEUTRAL", "BEARISH")
OUTLOOK_LEVELS = (
    "Clearly negative",
    "Leaning negative",
    "Balanced",
    "Leaning positive",
    "Clearly positive",
)
RESEARCH_SCHEMA_VERSION = "research-card-1"


def research_classify_enabled() -> bool:
    return env_flag("JEV_RESEARCH_CLASSIFY", False)


def evidence_token_budget() -> int:
    return max(50, env_int("JEV_EVIDENCE_TOKEN_BUDGET", 800))


def normalize_ticker(symbol: str) -> str:
    cleaned = symbol.upper().replace("/", "").replace(" ", "").strip()
    if not TICKER_RE.match(cleaned):
        raise ValueError(f"invalid research ticker: {symbol!r}")
    return cleaned


def lint_questions(questions: dict[str, Any]) -> list[str]:
    """Offline checks before a billable System One call. Empty means acceptable."""
    errors: list[str] = []
    if not isinstance(questions, dict) or not questions:
        return ["questions must be a non-empty object"]
    for name, spec in questions.items():
        if not isinstance(spec, dict):
            errors.append(f"{name} is not an object")
            continue
        kind = spec.get("type")
        instructions = str(spec.get("instructions") or "").strip()
        if kind not in {"choice", "score", "noul"}:
            errors.append(f"{name} has an unknown type")
            continue
        if len(instructions) < 12:
            errors.append(f"{name} instructions are too short to be a real question")
        criteria = spec.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                errors.append(f"{name} choice needs at least two criteria")
        elif kind == "score":
            if not isinstance(criteria, list) or len(criteria) < 2:
                errors.append(f"{name} score needs an ordered criteria list")
        elif kind == "noul" and criteria is not None and not isinstance(criteria, dict):
            errors.append(f"{name} noul criteria must be an object when present")
    return errors


def research_questions() -> dict[str, dict]:
    questions: dict[str, dict] = {
        "higher_30d": {
            "type": "choice",
            "instructions": (
                "Using only the cited evidence snippets in state, is the asset more likely "
                "than not to trade higher in 30 calendar days? This is research, not an order."
            ),
            "criteria": {
                "HIGHER": "The cited evidence leans toward a higher price in 30 days.",
                "NOT_HIGHER": "The cited evidence does not support a higher price in 30 days.",
            },
        },
        "outlook": {
            "type": "score",
            "instructions": "Rate the strength of the cited evidence outlook. Do not do arithmetic.",
            "criteria": list(OUTLOOK_LEVELS),
        },
        "evidence_quality": {
            "type": "noul",
            "instructions": "Is the cited evidence specific, sourced, and sufficient for a research note?",
            "criteria": {
                "true": "Multiple sourced snippets address the asset directly.",
                "false": "Snippets are thin, duplicated, or off-topic.",
            },
        },
    }
    for category in EVIDENCE_CATEGORIES:
        questions[f"bias_{category}"] = {
            "type": "choice",
            "instructions": (
                f"From the {category} snippets only, what is the research bias? "
                "Ignore snippets from other categories."
            ),
            "criteria": {
                "BULLISH": "Those snippets lean positive for the asset.",
                "NEUTRAL": "Those snippets are mixed, stale, or absent.",
                "BEARISH": "Those snippets lean negative for the asset.",
            },
        }
    return questions


def _unit(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise JevSchemaError(f"{field} is not a number") from exc
    if number > 1.0 and number <= 100.0:
        number = number / 100.0
    if number < 0.0 or number > 1.0 or number != number:
        raise JevSchemaError(f"{field} is out of range")
    return number


def _choice(raw: Any, allowed: tuple[str, ...], field: str) -> tuple[str, dict[str, float]]:
    if not isinstance(raw, dict):
        raise JevSchemaError(f"missing {field}")
    choice = str(raw.get("choice") or "").strip()
    if choice not in allowed:
        raise JevSchemaError(f"{field} choice invalid")
    probabilities = raw.get("probabilities")
    if not isinstance(probabilities, dict):
        raise JevSchemaError(f"{field} probabilities missing")
    parsed = {name: _unit(probabilities[name], f"{field}.{name}") for name in allowed if name in probabilities}
    if set(parsed) != set(allowed):
        raise JevSchemaError(f"{field} probabilities must cover {allowed}")
    total = sum(parsed.values())
    if total <= 0 or abs(total - 1.0) > 0.15:
        raise JevSchemaError(f"{field} probabilities do not sum to 1")
    scale = 1.0 / total
    return choice, {name: parsed[name] * scale for name in allowed}


def validate_research_response(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
        raise JevSchemaError("research response missing answers")
    answers = payload["answers"]
    higher, higher_probs = _choice(answers.get("higher_30d"), HIGHER_OUTCOMES, "higher_30d")
    outlook = answers.get("outlook")
    if not isinstance(outlook, dict):
        raise JevSchemaError("missing outlook")
    score = float(outlook.get("score"))
    if score < -0.05 or score > len(OUTLOOK_LEVELS) - 1 + 0.05:
        raise JevSchemaError("outlook score out of range")
    index = min(len(OUTLOOK_LEVELS) - 1, max(0, int(round(score))))
    quality = answers.get("evidence_quality")
    if not isinstance(quality, dict):
        raise JevSchemaError("missing evidence_quality")
    quality_p = _unit(quality.get("noul"), "evidence_quality.noul")
    category_signals = {}
    for category in EVIDENCE_CATEGORIES:
        choice, _probs = _choice(answers.get(f"bias_{category}"), CATEGORY_BIAS, f"bias_{category}")
        category_signals[category] = choice
    return {
        "model": payload.get("model"),
        "higher_30d": higher,
        "higher_probability": higher_probs["HIGHER"],
        "higher_probabilities": higher_probs,
        "outlook_score": score,
        "outlook_label": OUTLOOK_LEVELS[index],
        "evidence_quality": quality_p,
        "category_signals": category_signals,
        "usage": payload.get("usage") if isinstance(payload.get("usage"), dict) else {},
    }


def _no_classification(symbol: str, reason: str, passages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "status": "no_signal",
        "advisory": True,
        "influence_book": False,
        "sizing_allowed": False,
        "order_size_fraction": order_size_fraction(),
        "reason": reason,
        "classification": None,
        "passages": passages,
        "question_schema_version": RESEARCH_SCHEMA_VERSION,
    }


async def classify_research(
    symbol: str,
    categories: dict[str, list[dict[str, Any]]],
    *,
    client: Any | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    """Classify cited evidence. Disabled or failed calls return no classification."""
    ticker = normalize_ticker(symbol)
    passages = select_passages(categories, token_budget=evidence_token_budget())
    active = research_classify_enabled() if enabled is None else enabled
    if not active:
        return _no_classification(ticker, "JEV_RESEARCH_CLASSIFY is off", passages)
    questions = research_questions()
    problems = lint_questions(questions)
    if problems:
        return _no_classification(ticker, "; ".join(problems), passages)
    state = {
        "asset": ticker,
        "untrusted_evidence": True,
        "passages": [
            {
                "category": item.get("category"),
                "title": item.get("title"),
                "source": item.get("source"),
                "url": item.get("url"),
                "date": item.get("date"),
                "snippet": item.get("snippet"),
            }
            for item in passages
        ],
    }
    started = time.perf_counter()
    try:
        if circuit_open():
            raise JevUnavailable("Jev provider paused after credit errors")
        if client is None:
            payload = await _post_research(state, questions)
        else:
            payload = await client.system_one(state, questions)
        if isinstance(payload, dict) and "higher_30d" in payload and "answers" not in payload:
            parsed = payload
        else:
            parsed = validate_research_response(payload)
    except (JevUnavailable, JevSchemaError, ValueError, TypeError) as exc:
        logger.warning("Research classification unavailable for %s: %s", ticker, exc)
        return _no_classification(ticker, str(exc), passages)
    decision_ms = (time.perf_counter() - started) * 1000.0
    return {
        "symbol": ticker,
        "status": "ok",
        "advisory": True,
        "influence_book": False,
        "sizing_allowed": False,
        "order_size_fraction": order_size_fraction(),
        "reason": "research classification only",
        "classification": parsed,
        "passages": passages,
        "decision_ms": round(decision_ms, 2),
        "question_schema_version": RESEARCH_SCHEMA_VERSION,
        "mock": False,
    }


async def _post_research(state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
    key = typesafe_api_key()
    if not key:
        raise JevUnavailable("TYPESAFE_API_KEY not configured")
    url = f"{jev_base_url()}/v1/systemone"
    body = {"model": jev_model(), "state": state, "questions": questions}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=jev_timeout_seconds()) as http:
            response = await http.post(url, headers=headers, json=body)
            if response.status_code == 402:
                note_credit_failure("402 insufficient credits")
                raise JevUnavailable("Jev provider returned 402")
            response.raise_for_status()
            return response.json()
    except JevUnavailable:
        raise
    except httpx.HTTPError as exc:
        note_credit_failure(str(exc))
        raise JevUnavailable(f"research Jev HTTP error: {exc}") from exc


def signing_key_path() -> str:
    return os.getenv("JEV_TRUST_PRIVATE_KEY_PATH", "").strip()
