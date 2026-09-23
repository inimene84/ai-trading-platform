"""Fast Jev revaluator for the paper desk.

One short System One call per symbol: 30-day Noul, long/flat/short Choice,
conviction Score, and a risk Choice. Social pulls, the journal, and calibration
are skipped so a tape refresh stays one round trip. The card is advisory.
It never sizes or sends an order, and a mock provider is not treated as a
live revalue.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from backend.jev.client import JevUnavailable, log_unavailable
from backend.jev.gateway import build_client
from backend.jev.questions import analysis_questions
from backend.jev.research import lint_questions
from backend.jev.schema import (
    JevSchemaError,
    _answer,
    _probability_map,
    _score_index,
    _unit_probability,
)
from backend.jev.state import build_market_state, futures_symbol
from backend.services.binance_market_data import binance_market_data

logger = logging.getLogger(__name__)

SIDES = ("long", "flat", "short")
RISK_LEVELS = ("low", "watch", "high", "freeze")
CONVICTION_LEVELS = ("None", "Low", "Moderate", "High", "Extreme")
MIN_SIDE_CONFIDENCE = 0.55
LOSS_VETO = -0.025
GROSS_VETO = 0.8
CACHE_SECONDS = 30.0
MAX_SYMBOLS = 8
_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def clear_revalue_cache() -> None:
    _cache.clear()


def revalue_questions() -> dict[str, dict]:
    """Slimmer than the full advisory set. One call still answers every question."""
    return {
        "higher_30d": {
            "type": "noul",
            "instructions": (
                "Using only the supplied past-only market state, is the asset likely "
                "to trade higher over the next 30 days? This is a probability, not an order."
            ),
            "criteria": {
                "true": "The completed sessions support a higher price over the next 30 days.",
                "false": "The completed sessions do not support a higher price over the next 30 days.",
            },
        },
        "side": {
            "type": "choice",
            "instructions": (
                "For this round only, is the advisory stance long, flat, or short? "
                "Prefer flat when the edge is unclear. Do not size a position."
            ),
            "criteria": {
                "long": "Completed sessions support a long bias this round.",
                "flat": "No clear asymmetric edge this round.",
                "short": "Completed sessions support a short bias this round.",
            },
        },
        "conviction": {
            "type": "score",
            "instructions": (
                "How much conviction does the past-only state justify? "
                "Empty social data is not extra conviction."
            ),
            "criteria": list(CONVICTION_LEVELS),
        },
        "risk_level": {
            "type": "choice",
            "instructions": (
                "What deposit-style risk level fits this state? "
                "Choose freeze when volatility or disagreement makes a ticket unsafe."
            ),
            "criteria": {
                "low": "Quiet state, no reason to freeze the ticket.",
                "watch": "Edge exists but volatility or disagreement needs a closer look.",
                "high": "Wide range or unstable momentum. Size would be inappropriate.",
                "freeze": "State is unsafe for a new ticket.",
            },
        },
    }


def validate_revalue(payload: Any) -> dict[str, Any]:
    """Normalize a fast-revalue System One body or raise JevSchemaError."""
    if not isinstance(payload, dict):
        raise JevSchemaError("response is not an object")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise JevSchemaError("missing answers")

    higher = _unit_probability(_answer(answers, "higher_30d").get("noul"), "higher_30d.noul")

    side_raw = _answer(answers, "side")
    side = str(side_raw.get("choice") or "").strip()
    if side not in SIDES:
        raise JevSchemaError(f"side choice invalid: {side!r}")
    side_probs = _probability_map(side_raw.get("probabilities"), SIDES, "side")
    if side_raw.get("confidence") is not None:
        side_confidence = _unit_probability(side_raw.get("confidence"), "side.confidence")
    else:
        side_confidence = side_probs[side]

    conviction_score, _label = _score_index(
        _answer(answers, "conviction").get("score"),
        CONVICTION_LEVELS,
        "conviction",
    )
    conviction = conviction_score / (len(CONVICTION_LEVELS) - 1)

    risk_raw = _answer(answers, "risk_level")
    risk = str(risk_raw.get("choice") or "").strip()
    if risk not in RISK_LEVELS:
        raise JevSchemaError(f"risk_level choice invalid: {risk!r}")
    risk_probs = _probability_map(risk_raw.get("probabilities"), RISK_LEVELS, "risk_level")
    if risk_raw.get("confidence") is not None:
        risk_confidence = _unit_probability(risk_raw.get("confidence"), "risk_level.confidence")
    else:
        risk_confidence = risk_probs[risk]

    return {
        "model": payload.get("model"),
        "higher": higher,
        "side": side,
        "side_confidence": side_confidence,
        "conviction": conviction,
        "risk": risk,
        "risk_confidence": risk_confidence,
        "freeze_prob": risk_probs["freeze"],
    }


def _stand_aside(symbol: str, reason: str, source: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "status": "no_signal",
        "source": source,
        "side": "flat",
        "higher": 0.5,
        "conviction": 0.0,
        "sideConfidence": 0.0,
        "risk": "watch",
        "riskConfidence": 0.0,
        "freezeProb": 0.0,
        "verdict": "STAND ASIDE",
        "vetoes": [reason],
        "entry": None,
        "stop": None,
        "target": None,
        "evidence": [],
        "evidenceKept": 0,
        "evidenceDeduped": 0,
        "questions": [],
        "pattern": "Jev fast revaluator",
        "advisory": True,
        "sizing_allowed": False,
        "influence_book": False,
        "reason": reason,
    }


def _headline_lines(titles: list[str]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for title in titles:
        text = " ".join(str(title).split())[:180]
        key = text.lower()
        if len(text) < 8 or key in seen:
            continue
        seen.add(key)
        lines.append(f"Headline: {text}")
        if len(lines) >= 4:
            break
    return lines


def _compose(
    symbol: str,
    state: dict[str, Any],
    validated: dict[str, Any],
    *,
    loss_frac: float,
    gross_frac: float,
    headlines: list[str] | None = None,
) -> dict[str, Any]:
    market = state.get("market") or {}
    last = float(market.get("last_close") or 0.0)
    atr = market.get("atr_14")
    atr_value = float(atr) if isinstance(atr, (int, float)) and atr > 0 else last * 0.01
    side = validated["side"]
    risk = validated["risk"]
    vetoes: list[str] = []
    if loss_frac <= LOSS_VETO:
        vetoes.append("Session loss is past 2.5% of starting equity.")
    if gross_frac >= GROSS_VETO:
        vetoes.append("Gross exposure is past 80% of equity.")
    if risk == "freeze":
        vetoes.append("Risk Choice is freeze. The rule vetoes the ticket.")

    if vetoes:
        verdict = "VETO"
    elif side == "flat":
        verdict = "STAND ASIDE"
    elif float(validated["side_confidence"]) < MIN_SIDE_CONFIDENCE:
        verdict = "ESCALATE"
    else:
        verdict = "ALLOW"

    entry = None if side == "flat" or last <= 0 else last
    if entry is None:
        stop = None
        target = None
    elif side == "long":
        stop = entry - atr_value
        target = entry + atr_value * 1.8
    else:
        stop = entry + atr_value
        target = entry - atr_value * 1.8
    if stop is not None and stop <= 0:
        stop = entry * 0.5 if entry else None
    if target is not None and target <= 0:
        target = None

    titles = [str(title) for title in (headlines or []) if str(title).strip()][:6]
    evidence = _headline_lines(titles) + [
        (
            f"Last close {last:.6g}. RSI {market.get('rsi_14')}. "
            f"ATR {market.get('atr_14')}."
        ),
        (
            f"Window change {market.get('change_window_pct')}%. "
            f"Momentum {market.get('momentum_bucket')}."
        ),
        (
            f"MA distance {market.get('ma_distance_pct')}%. "
            f"Volume ratio {market.get('volume_ratio')}."
        ),
        "Social feed skipped. This pass is the fast revaluator.",
        "Advisory only. This card does not send a live order.",
    ]
    evidence_raw = len(titles) + 5
    higher = float(validated["higher"])
    conviction = float(validated["conviction"])
    side_confidence = float(validated["side_confidence"])
    risk_confidence = float(validated["risk_confidence"])
    questions = [
        {
            "question": "Likely higher over the next 30 days?",
            "type": "Noul",
            "answer": f"{higher:.2f}",
            "confidence": None,
        },
        {
            "question": "This round: long, flat, or short?",
            "type": "Choice",
            "answer": side,
            "confidence": round(side_confidence, 4),
        },
        {
            "question": "How much conviction is justified?",
            "type": "Score",
            "answer": f"{conviction:.2f}",
            "confidence": None,
        },
        {
            "question": "Deposit-style risk level?",
            "type": "Choice",
            "answer": risk,
            "confidence": round(risk_confidence, 4),
        },
    ]
    return {
        "symbol": symbol,
        "status": "ok",
        "source": "jev",
        "side": side,
        "higher": round(higher, 4),
        "conviction": round(conviction, 4),
        "sideConfidence": round(side_confidence, 4),
        "risk": risk,
        "riskConfidence": round(risk_confidence, 4),
        "freezeProb": round(float(validated["freeze_prob"]), 4),
        "verdict": verdict,
        "vetoes": vetoes,
        "entry": entry,
        "stop": None if stop is None else round(float(stop), 8),
        "target": None if target is None else round(float(target), 8),
        "evidence": evidence,
        "evidenceKept": len(evidence),
        "evidenceDeduped": evidence_raw,
        "questions": questions,
        "pattern": "Jev fast revaluator · headline evidence · no order",
        "model": validated.get("model"),
        "advisory": True,
        "sizing_allowed": False,
        "influence_book": False,
        "reason": f"Jev {side} ({verdict})",
    }


def _cache_key(
    symbol: str,
    state: dict[str, Any],
    loss_frac: float,
    gross_frac: float,
    headlines: list[str],
) -> str:
    market = state.get("market") or {}
    loss_flag = int(loss_frac <= LOSS_VETO)
    gross_flag = int(gross_frac >= GROSS_VETO)
    hint = "|".join(headlines[:3])
    return f"{symbol}:{market.get('last_close')}:{loss_flag}:{gross_flag}:{hint}"


async def _bars_for(symbol: str, bars: list[dict] | None) -> list[dict]:
    if bars is not None:
        return list(bars)
    try:
        return await binance_market_data.get_klines(futures_symbol(symbol), interval="1h", limit=40)
    except Exception as exc:
        logger.warning("Jev revalue bar load failed for %s: %s", symbol, exc)
        return []


async def revalue_symbol(
    symbol: str,
    *,
    bars: list[dict] | None = None,
    loss_frac: float = 0.0,
    gross_frac: float = 0.0,
    client: Any | None = None,
    fetch_bars: bool = True,
    headlines: list[str] | None = None,
) -> dict[str, Any]:
    """Revalue one symbol. A missing or invalid Jev answer is stand-aside."""
    requested = (symbol or "").strip().upper()
    if not requested:
        return _stand_aside("", "symbol is required", "unavailable")

    jev = client or build_client()
    if getattr(jev, "name", "") == "mock":
        return _stand_aside(requested, "mock provider is not a live revalue", "mock")

    if bars is not None:
        history = list(bars)
    elif fetch_bars:
        history = await _bars_for(requested, None)
    else:
        history = []
    state = build_market_state(requested, history)
    if state is None:
        return _stand_aside(requested, "insufficient past-only history", "unavailable")

    titles = [" ".join(str(title).split())[:180] for title in (headlines or []) if str(title).strip()][:6]
    key = _cache_key(requested, state, loss_frac, gross_frac, titles)
    cached = _cache.get(key)
    now = time.time()
    if cached and cached[0] > now:
        payload = dict(cached[1])
        payload["cached"] = True
        return payload

    questions = revalue_questions()
    problems = lint_questions(questions)
    if problems:
        return _stand_aside(requested, "question lint failed", "unavailable")
    # Guard the slim set against accidentally shipping the slow advisory questionnaire.
    if set(questions) == set(analysis_questions()):
        return _stand_aside(requested, "revalue questions collided with the full advisory set", "unavailable")

    jev_state = {
        "asset": state["asset"],
        "market": {item: value for item, value in state["market"].items()},
        "social_stats": {
            "sample_size": 0,
            "polarity_score": 0.0,
            "sentiment_label": "Neutral / Mixed",
        },
        "representative_posts": [],
        "headlines": titles,
        "book": {
            "loss_frac": loss_frac,
            "gross_frac": gross_frac,
            "note": "Paper desk context. Do not size or send an order.",
        },
    }
    try:
        validated = await jev.system_one(jev_state, questions, validator=validate_revalue)
    except (JevUnavailable, JevSchemaError) as exc:
        log_unavailable(requested, exc)
        return _stand_aside(requested, str(exc), "unavailable")
    except Exception as exc:
        log_unavailable(requested, exc)
        return _stand_aside(requested, "Jev revalue failed", "unavailable")

    card = _compose(
        requested,
        state,
        validated,
        loss_frac=loss_frac,
        gross_frac=gross_frac,
        headlines=titles,
    )
    card["cached"] = False
    _cache[key] = (now + CACHE_SECONDS, card)
    return card


async def revalue_tape(
    symbols: list[str],
    *,
    loss_frac: float = 0.0,
    gross_frac: float = 0.0,
    client: Any | None = None,
    bars_by_symbol: dict[str, list[dict]] | None = None,
    headlines_by_symbol: dict[str, list[str]] | None = None,
    fetch_bars: bool = True,
) -> dict[str, Any]:
    """Revalue up to eight symbols concurrently. One failure does not blank the tape."""
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        name = (raw or "").strip().upper()
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
        if len(ordered) >= MAX_SYMBOLS:
            break
    if not ordered:
        return {
            "advisory": True,
            "sizing_allowed": False,
            "influence_book": False,
            "called_jev": False,
            "cards": [],
            "elapsed_ms": 0.0,
        }

    started = time.perf_counter()
    semaphore = asyncio.Semaphore(4)
    supplied = bars_by_symbol or {}
    supplied_headlines = headlines_by_symbol or {}

    async def one(name: str) -> dict[str, Any]:
        async with semaphore:
            preset = supplied.get(name)
            return await revalue_symbol(
                name,
                bars=preset,
                loss_frac=loss_frac,
                gross_frac=gross_frac,
                client=client,
                fetch_bars=fetch_bars and preset is None,
                headlines=supplied_headlines.get(name) or [],
            )

    cards = list(await asyncio.gather(*(one(name) for name in ordered)))
    return {
        "advisory": True,
        "sizing_allowed": False,
        "influence_book": False,
        "called_jev": any(card.get("source") == "jev" for card in cards),
        "cards": cards,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }
