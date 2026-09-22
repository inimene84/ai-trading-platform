"""Jev advisory path: schema, fail-closed client, state, social dedup, votes."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from backend.jev.backtest import realized_direction, score_t_plus_one
from backend.jev.client import JevClient, JevUnavailable
from backend.jev.questions import PRICE_DIRECTIONS, TRADE_ACTIONS
from backend.jev.schema import JevSchemaError, validate_system_one
from backend.jev.service import clear_evaluation_cache, evaluate_symbol, vote_from_answers
from backend.jev.state import build_market_state, rsi
from backend.jev.stats import process_posts
from backend.jev.store import TweetStore
from backend.jev.twitter import TwitterIngestor, clear_twitter_cache


def _probs(winner: str, allowed: tuple[str, ...], winner_p: float = 0.7) -> dict[str, float]:
    rest = (1.0 - winner_p) / (len(allowed) - 1)
    return {name: (winner_p if name == winner else rest) for name in allowed}


def _payload(
    action: str = "BUY",
    direction: str = "UP",
    direction_probs: dict[str, float] | None = None,
) -> dict:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "trade_action": {
                "type": "choice",
                "choice": action,
                "confidence": 0.81,
                "probabilities": _probs(action, TRADE_ACTIONS, 0.7),
            },
            "sentiment_spectrum": {"type": "score", "score": 3.1},
            "is_short_squeeze_risk": {"type": "noul", "noul": 0.22},
            "catalyst_impact": {"type": "score", "score": 1.0},
            "price_direction": {
                "type": "choice",
                "choice": direction,
                "probabilities": direction_probs or _probs(direction, PRICE_DIRECTIONS, 0.72),
            },
        },
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


def _bars(count: int = 20, drift: float = 1.0) -> list[dict]:
    rows = []
    price = 100.0
    for index in range(count):
        price += drift
        rows.append({
            "open": price - 0.4,
            "high": price + 0.6,
            "low": price - 0.8,
            "close": price,
            "volume": 10 + index,
        })
    return rows


def test_schema_accepts_a_complete_response():
    parsed = validate_system_one(_payload())
    assert parsed["trade_action"] == "BUY"
    assert parsed["price_direction"] == "UP"
    assert parsed["sentiment_label"] == "Optimistic / Bullish"
    assert abs(sum(parsed["direction_probabilities"].values()) - 1.0) < 1e-9


def test_schema_rejects_missing_probabilities_and_bad_choice():
    broken = _payload()
    del broken["answers"]["price_direction"]["probabilities"]
    with pytest.raises(JevSchemaError):
        validate_system_one(broken)
    broken = _payload()
    broken["answers"]["trade_action"]["choice"] = "YOLO"
    with pytest.raises(JevSchemaError):
        validate_system_one(broken)
    broken = _payload()
    broken["answers"]["is_short_squeeze_risk"]["noul"] = 2
    with pytest.raises(JevSchemaError):
        validate_system_one(broken)


def test_vote_is_suppressed_on_narrow_margin_or_conflict():
    wide = validate_system_one(_payload())
    vote = vote_from_answers(wide, min_margin=0.12)
    assert vote["signal"] == "bullish"
    assert vote["vetoed"] is False

    tight = validate_system_one(_payload(direction_probs={"UP": 0.4, "FLAT": 0.35, "DOWN": 0.25}))
    vetoed = vote_from_answers(tight, min_margin=0.12)
    assert vetoed["signal"] is None
    assert vetoed["vetoed"] is True
    assert vetoed["confidence"] == 0.0

    clash = validate_system_one(_payload(action="BUY", direction="DOWN"))
    conflicted = vote_from_answers(clash, min_margin=0.12)
    assert conflicted["signal"] is None
    assert conflicted["conflict"] is True


def test_hold_stays_neutral():
    parsed = validate_system_one(_payload(action="HOLD", direction="FLAT"))
    vote = vote_from_answers(parsed, min_margin=0.05)
    assert vote["signal"] == "neutral"


def test_state_is_past_only_and_refuses_short_history():
    assert build_market_state("BTC", _bars(5)) is None
    state = build_market_state("BTC", _bars(20, drift=0.5), {"funding_rate": -0.01})
    assert state is not None
    assert state["asset"] == "BTC"
    assert state["symbol"] == "BTCUSDT"
    assert state["market"]["sessions"] == 20
    assert state["market"]["rsi_14"] > 50
    assert state["market"]["funding_rate"] == -0.01
    assert rsi([float(i) for i in range(20)]) == 100.0


def test_polarity_scores_fear_and_greed():
    stats = process_posts([
        {"id": "1", "text": "panic dump crash", "likes": 5, "retweets": 1, "author_username": "a"},
        {"id": "2", "text": "moon pump breakout", "likes": 50, "retweets": 4, "author_username": "b"},
        {"id": "1", "text": "duplicate", "likes": 1, "retweets": 0, "author_username": "a"},
    ])
    assert stats["sample_size"] == 3
    assert stats["unique_authors_count"] == 2
    assert len(stats["stratified_sample"]) == 2
    assert process_posts([])["polarity_score"] == 0.0


@pytest.mark.asyncio
async def test_client_fail_closed_without_key_or_bad_schema():
    client = JevClient(api_key="", timeout=1)
    with pytest.raises(JevUnavailable, match="not configured"):
        await client.system_one({"asset": "BTC"}, {"trade_action": {}})

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answers": {}})

    bad = JevClient(api_key="test-key", timeout=1, transport=httpx.MockTransport(handler))
    with pytest.raises(JevUnavailable, match="schema"):
        await bad.system_one({"asset": "BTC"}, {"trade_action": {}})


@pytest.mark.asyncio
async def test_evaluate_symbol_never_buys_when_jev_is_down(monkeypatch):
    monkeypatch.setenv("JEV_CACHE_SECONDS", "0")
    clear_evaluation_cache()

    class Down:
        async def system_one(self, _state, _questions):
            raise JevUnavailable("timeout")

    result = await evaluate_symbol(
        "ETH",
        bars=_bars(),
        include_social=False,
        client=Down(),
        fetch_bars=False,
    )
    assert result["status"] == "no_signal"
    assert result["action"] is None
    assert result["signal"] is None
    assert result["action"] not in TRADE_ACTIONS


@pytest.mark.asyncio
async def test_evaluate_symbol_caches_a_validated_call(monkeypatch):
    monkeypatch.setenv("JEV_CACHE_SECONDS", "600")
    clear_evaluation_cache()
    parsed = validate_system_one(_payload())

    class Once:
        def __init__(self):
            self.calls = 0

        async def system_one(self, _state, _questions):
            self.calls += 1
            return parsed

    client = Once()
    first = await evaluate_symbol("SOL", bars=_bars(), include_social=False, client=client, fetch_bars=False)
    second = await evaluate_symbol("SOL", bars=_bars(), include_social=False, client=client, fetch_bars=False)
    assert first["status"] == "ok"
    assert first["signal"] == "bullish"
    assert first["advisory"] is True
    assert client.calls == 1
    assert second["cached"] is True
    clear_evaluation_cache()


@pytest.mark.asyncio
async def test_twitter_early_stop_and_no_simulated_posts(tmp_path):
    clear_twitter_cache()
    store = TweetStore(tmp_path / "tweets.db")
    store.save("BTC", [{
        "id": "known1",
        "text": "old fear dump",
        "likes": 1,
        "retweets": 0,
        "author_username": "old",
        "timestamp_epoch": 10,
    }])
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert request.headers["x-api-key"] == "tw-key"
        body = {
            "tweets": [
                {"id": "new1", "text": "$BTC moon pump", "likeCount": 12, "author": {"userName": "new"}},
                {"id": "known1", "text": "already stored", "likeCount": 1, "author": {"userName": "old"}},
            ],
            "next_cursor": "should-not-fetch",
        }
        return httpx.Response(200, json=body)

    ingestor = TwitterIngestor(
        api_key="tw-key",
        store=store,
        transport=httpx.MockTransport(handler),
        cache_seconds=0,
    )
    pulled = await ingestor.fetch("BTC", target_count=10)
    assert calls["n"] == 1
    assert pulled["early_stopped"] is True
    assert pulled["newly_fetched_count"] == 1
    assert "known1" in store.known_ids("BTC")
    assert "new1" in store.known_ids("BTC")

    empty = TwitterIngestor(api_key="", store=TweetStore(tmp_path / "empty.db"), cache_seconds=0)
    missing = await empty.fetch("BTC", target_count=5)
    assert missing["tweets"] == []
    assert missing["status"] == "unavailable"
    assert "mock" not in json.dumps(missing)


def test_t_plus_one_harness_scores_next_bar_only():
    bars = [
        {"close": 100},
        {"close": 102},
        {"close": 102.02},
    ]
    scored = score_t_plus_one(bars, ["UP", "DOWN"], flat_bps=8.0)
    assert realized_direction(100, 102) == "UP"
    assert realized_direction(102, 102.02) == "FLAT"
    assert scored["samples"] == 2
    assert scored["hits"] == 1
    assert scored["accuracy"] == 0.5
    assert score_t_plus_one(bars, ["UP"])["samples"] == 0


@pytest.mark.asyncio
async def test_opinion_layer_uses_jev_and_keeps_personas_on_failure(monkeypatch):
    from backend.services.opinion_layer import AgentOpinion, analyze_symbol

    monkeypatch.setenv("JEV_REPLACE_PERSONAS", "true")
    monkeypatch.setenv("FINMEM_ENABLED", "false")
    monkeypatch.setattr(
        "backend.services.opinion_layer._run_technical_opinion",
        lambda *_args, **_kwargs: AgentOpinion(agent="technical_analyst", signal="neutral", confidence=0.1),
    )
    monkeypatch.setattr(
        "backend.services.opinion_layer._get_trade_memory",
        lambda *_args, **_kwargs: {"count": 0, "summary": ""},
    )
    recall = MagicMock(samples=0, confidence=0.0, reasoning="", signal="neutral")
    recall.to_dict.return_value = {}
    monkeypatch.setattr("backend.services.opinion_layer.trade_memory.recall_similar", AsyncMock(return_value=recall))
    monkeypatch.setattr("backend.services.opinion_layer.skill_miner.match_skill", lambda *_args, **_kwargs: None)
    personas = AsyncMock(return_value=[])
    monkeypatch.setattr("backend.services.opinion_layer.run_all_personas", personas)

    jev_ok = {
        "status": "ok",
        "signal": "bullish",
        "confidence": 0.66,
        "reason": "Jev BUY",
        "action": "BUY",
        "answers": {"trade_action": "BUY"},
        "vetoed": False,
        "influence_book": True,
    }
    monkeypatch.setattr("backend.services.opinion_layer.evaluate_opinion", AsyncMock(return_value=jev_ok))
    opinion = await analyze_symbol(
        "BTCUSDT",
        _bars(3),
        include_kronos=False,
        include_social=False,
        include_alerts=False,
        include_personas=True,
        include_macro=False,
        include_news=False,
    )
    names = [item.agent for item in opinion.agent_opinions]
    assert "jev_analyst" in names
    personas.assert_not_awaited()

    personas.reset_mock()
    monkeypatch.setattr("backend.services.opinion_layer.evaluate_opinion", AsyncMock(return_value=None))
    await analyze_symbol(
        "BTCUSDT",
        _bars(3),
        include_kronos=False,
        include_social=False,
        include_alerts=False,
        include_personas=True,
        include_macro=False,
        include_news=False,
    )
    personas.assert_awaited()
