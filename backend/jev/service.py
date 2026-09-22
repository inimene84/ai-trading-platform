"""Orchestrate a read-only Jev evaluation.

The returned object is advisory. `action` is null whenever status is
`no_signal`. Nothing here places or sizes an order.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from backend.jev.calibration import calibration_report, display_probability
from backend.jev.client import JevClient, JevUnavailable, log_unavailable
from backend.jev.config import (
    jev_cache_seconds,
    jev_include_social,
    jev_influence_book,
    jev_min_prob_margin,
)
from backend.jev.conformal import conformal_for
from backend.jev.journal import get_journal
from backend.jev.meta import meta_take, order_size_fraction
from backend.jev.questions import analysis_questions
from backend.jev.research import lint_questions
from backend.jev.trust import allows_book_influence, trust_report
from backend.jev.sinks import persist_social
from backend.jev.state import base_asset, build_market_state, futures_symbol
from backend.jev.stats import process_posts
from backend.jev.twitter import TwitterIngestor
from backend.services.binance_market_data import binance_market_data

logger = logging.getLogger(__name__)

_BULLISH = {"STRONG_BUY", "BUY"}
_BEARISH = {"SELL", "STRONG_SELL"}
_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def clear_evaluation_cache() -> None:
    _cache.clear()


def vote_from_answers(validated: dict[str, Any], min_margin: float | None = None) -> dict[str, Any]:
    """Map a validated Jev payload to a single opinion-layer vote.

    A narrow up/flat/down margin, or a disagreement between the trade stance
    and the direction forecast, suppresses the vote. The evaluation itself
    still counts as successful so callers can skip the expensive persona path.
    """
    margin_floor = jev_min_prob_margin() if min_margin is None else min_margin
    action = validated["trade_action"]
    direction = validated["price_direction"]
    margin = float(validated["direction_margin"])
    action_sign = 1 if action in _BULLISH else (-1 if action in _BEARISH else 0)
    direction_sign = 1 if direction == "UP" else (-1 if direction == "DOWN" else 0)
    conflict = action_sign != 0 and direction_sign != 0 and action_sign != direction_sign
    vetoed = margin < margin_floor
    if vetoed or conflict:
        signal = None
        confidence = 0.0
    elif action_sign > 0:
        signal = "bullish"
        confidence = float(validated["trade_confidence"])
    elif action_sign < 0:
        signal = "bearish"
        confidence = float(validated["trade_confidence"])
    else:
        signal = "neutral"
        confidence = float(validated["trade_confidence"])
    reason = _reason(validated, vetoed=vetoed, conflict=conflict, margin_floor=margin_floor)
    return {
        "signal": signal,
        "confidence": round(confidence, 4),
        "vetoed": vetoed,
        "conflict": conflict,
        "reason": reason,
    }


def _reason(validated: dict[str, Any], *, vetoed: bool, conflict: bool, margin_floor: float) -> str:
    if vetoed:
        return (
            f"uncertainty veto: price-direction margin {validated['direction_margin']:.3f} "
            f"< {margin_floor:.3f}"
        )
    if conflict:
        return (
            f"no signal: trade stance {validated['trade_action']} disagrees with "
            f"price direction {validated['price_direction']}"
        )
    return (
        f"Jev {validated['trade_action']} "
        f"(direction {validated['price_direction']}, "
        f"sentiment {validated['sentiment_label']}, "
        f"squeeze {validated['squeeze_probability']:.0%})"
    )


def _no_signal(symbol: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "symbol": futures_symbol(symbol) if symbol else symbol,
        "base_asset": base_asset(symbol) if symbol else "",
        "status": "no_signal",
        "advisory": True,
        "signal": None,
        "action": None,
        "confidence": 0.0,
        "vetoed": False,
        "conflict": False,
        "reason": reason,
        "answers": None,
        "market": extra.get("market"),
        "social": extra.get("social"),
        "cached": False,
    }


def _cache_key(symbol: str, state: dict[str, Any], include_social: bool) -> str:
    market = state.get("market") or {}
    sessions = market.get("ohlcv") or []
    last = sessions[-1]["close"] if sessions else "na"
    return f"{state.get('symbol')}:{last}:{market.get('rsi_14')}:{include_social}"


async def _load_bars(symbol: str) -> list[dict]:
    return await binance_market_data.get_klines(futures_symbol(symbol), interval="1h", limit=40)


async def evaluate_symbol(
    symbol: str,
    bars: list[dict] | None = None,
    metrics: dict | None = None,
    include_social: bool | None = None,
    client: JevClient | None = None,
    twitter: TwitterIngestor | None = None,
    fetch_bars: bool = True,
    social_sample: int | None = None,
) -> dict[str, Any]:
    """Evaluate one symbol. Never returns a buy when Jev is unavailable."""
    asset_symbol = (symbol or "").strip()
    if not asset_symbol:
        return _no_signal("", "symbol is required")

    social_on = jev_include_social() if include_social is None else include_social
    history = list(bars) if bars is not None else []
    if not history and fetch_bars:
        try:
            history = await _load_bars(asset_symbol)
        except Exception as exc:
            logger.warning("Jev bar load failed for %s: %s", asset_symbol, exc)
            history = []

    state = build_market_state(asset_symbol, history, metrics)
    if state is None:
        failure = _no_signal(asset_symbol, "insufficient past-only history")
        _govern(failure, {"symbol": asset_symbol, "reason": "insufficient history"}, None, None, 0.0)
        return failure

    social_stats = process_posts([])
    social_meta: dict[str, Any] = {"status": "skipped", "sample_size": 0}
    if social_on:
        ingestor = twitter or TwitterIngestor()
        social_pull = await ingestor.fetch(state["asset"], target_count=social_sample)
        social_stats = process_posts(social_pull.get("tweets") or [])
        social_meta = {
            "status": social_pull.get("status"),
            "reason": social_pull.get("reason") or "",
            "sample_size": social_stats["sample_size"],
            "polarity_score": social_stats["polarity_score"],
            "sentiment_label": social_stats["sentiment_label"],
            "early_stopped": social_pull.get("early_stopped", False),
            "newly_fetched_count": social_pull.get("newly_fetched_count", 0),
        }
        try:
            await persist_social(state["symbol"], social_stats, social_pull.get("tweets") or [])
        except Exception as exc:
            logger.warning("Jev social persist skipped for %s: %s", asset_symbol, exc)

    jev_state = {
        "asset": state["asset"],
        "market": {key: value for key, value in state["market"].items()},
        "social_stats": {
            "sample_size": social_stats["sample_size"],
            "author_diversity_pct": social_stats["author_diversity_pct"],
            "polarity_score": social_stats["polarity_score"],
            "weighted_polarity_score": social_stats.get("weighted_polarity_score", 0.0),
            "bot_downweight_mean": social_stats.get("bot_downweight_mean", 1.0),
            "sentiment_label": social_stats["sentiment_label"],
        },
        "representative_posts": social_stats["stratified_sample"][:20],
    }
    cache_key = _cache_key(asset_symbol, state, social_on)
    ttl = jev_cache_seconds()
    cached = _cache.get(cache_key)
    if cached and cached[0] > time.time():
        payload = dict(cached[1])
        payload["cached"] = True
        return payload

    jev = client or JevClient()
    questions = analysis_questions()
    problems = lint_questions(questions)
    if problems:
        failure = _no_signal(asset_symbol, "question lint failed: " + "; ".join(problems), market=_summary(state), social=social_meta)
        _govern(failure, jev_state, None, client, 0.0)
        return failure
    started = time.perf_counter()
    try:
        validated = await jev.system_one(jev_state, questions)
    except JevUnavailable as exc:
        log_unavailable(asset_symbol, exc)
        failure = _no_signal(asset_symbol, str(exc), market=_summary(state), social=social_meta)
        _govern(failure, jev_state, None, client, (time.perf_counter() - started) * 1000.0)
        return failure
    except Exception as exc:
        log_unavailable(asset_symbol, exc)
        failure = _no_signal(asset_symbol, "Jev evaluation failed", market=_summary(state), social=social_meta)
        _govern(failure, jev_state, None, client, (time.perf_counter() - started) * 1000.0)
        return failure

    vote = vote_from_answers(validated)
    result = {
        "symbol": state["symbol"],
        "base_asset": state["asset"],
        "status": "ok",
        "advisory": True,
        "signal": vote["signal"],
        "action": None if vote["signal"] is None else validated["trade_action"],
        "confidence": vote["confidence"],
        "vetoed": vote["vetoed"],
        "conflict": vote["conflict"],
        "reason": vote["reason"],
        "answers": {
            "trade_action": validated["trade_action"],
            "trade_confidence": validated["trade_confidence"],
            "trade_probabilities": validated["trade_probabilities"],
            "sentiment_label": validated["sentiment_label"],
            "sentiment_score": validated["sentiment_score"],
            "squeeze_probability": validated["squeeze_probability"],
            "catalyst_label": validated["catalyst_label"],
            "catalyst_score": validated["catalyst_score"],
            "price_direction": validated["price_direction"],
            "direction_probabilities": validated["direction_probabilities"],
            "direction_margin": validated["direction_margin"],
        },
        "market": _summary(state),
        "social": social_meta,
        "cached": False,
        "model": validated.get("model"),
    }
    _govern(result, jev_state, validated, client, (time.perf_counter() - started) * 1000.0)
    if ttl > 0 and result["status"] == "ok":
        _cache[cache_key] = (time.time() + ttl, result)
    return result


def _govern(
    result: dict[str, Any],
    state: dict[str, Any],
    validated: dict[str, Any] | None,
    client: Any,
    latency_ms: float,
) -> None:
    """Attach journal, calibration, and the book-influence gate. Sizing stays zero."""
    provider = str(getattr(client, "name", "typesafe") or "typesafe")
    usage = (validated or {}).get("usage") if isinstance(validated, dict) else None
    tokens = None
    if isinstance(usage, dict) and usage.get("input_tokens") is not None:
        tokens = int(usage["input_tokens"])
    journal_meta: dict[str, Any] = {}
    try:
        journal_meta = get_journal().record(
            symbol=str(result.get("symbol") or state.get("asset") or ""),
            state=state,
            status=str(result.get("status") or "no_signal"),
            provider=provider,
            model_version=(validated or {}).get("model") if isinstance(validated, dict) else None,
            raw_answers=result.get("answers") if isinstance(result.get("answers"), dict) else None,
            input_tokens=tokens,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        logger.warning("Jev journal write skipped: %s", exc)
    try:
        labeled = get_journal().labeled_choices()
    except Exception as exc:
        logger.warning("Jev journal read skipped: %s", exc)
        labeled = []
    report = calibration_report(labeled)
    trust = trust_report(labeled)
    direction_probs = {}
    margin = 0.0
    if isinstance(validated, dict):
        direction_probs = dict(validated.get("direction_probabilities") or {})
        margin = float(validated.get("direction_margin") or 0.0)
    conformal_veto = False
    if direction_probs:
        conformal_veto = conformal_for(str(result.get("symbol") or "")).veto(direction_probs)
    if conformal_veto and result.get("status") == "ok":
        result["signal"] = None
        result["action"] = None
        result["confidence"] = 0.0
        result["vetoed"] = True
        result["reason"] = "adaptive conformal set is too wide"
    result["conformal_veto"] = conformal_veto
    meta = meta_take(
        labels=int(report["labels"]),
        min_labels=int(report["min_labels"]),
        margin=margin,
        min_margin=jev_min_prob_margin(),
        probabilities=direction_probs or {"FLAT": 1.0},
        conformal_veto=conformal_veto,
    )
    influence = bool(
        jev_influence_book()
        and report["ready"]
        and meta["take"]
        and provider != "mock"
        and result.get("status") == "ok"
        and result.get("signal") in {"bullish", "bearish", "neutral"}
        and not result.get("vetoed")
        and allows_book_influence(str(trust["verdict"]))
    )
    result["decision_id"] = journal_meta.get("decision_id")
    result["state_hash"] = journal_meta.get("state_hash")
    result["question_schema_version"] = journal_meta.get("question_schema_version")
    result["calibration"] = report
    result["trust"] = trust
    result["display_probabilities"] = display_probability(direction_probs, report.get("temperature")) if direction_probs else {}
    result["meta"] = meta
    result["influence_book"] = influence
    result["sizing_allowed"] = False
    result["order_size_fraction"] = order_size_fraction()


def _summary(state: dict[str, Any]) -> dict[str, Any]:
    market = state.get("market") or {}
    return {
        "sessions": market.get("sessions"),
        "last_close": market.get("last_close"),
        "rsi_14": market.get("rsi_14"),
        "atr_14": market.get("atr_14"),
        "ma_distance_pct": market.get("ma_distance_pct"),
        "volume_ratio": market.get("volume_ratio"),
        "momentum_bucket": market.get("momentum_bucket"),
    }
