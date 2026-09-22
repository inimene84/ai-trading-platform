"""OpenRouter System One provider. HTTP is mocked; nothing is sent to the network."""

from __future__ import annotations

import json

import httpx
import pytest

from backend.jev.client import JevUnavailable
from backend.jev.gateway import OpenRouterJevClient, build_client, circuit_open, reset_circuit
from backend.jev.journal import reset_journal_cache
from backend.jev.meta import order_size_fraction
from backend.jev.questions import PRICE_DIRECTIONS, TRADE_ACTIONS
from backend.jev.quote import live_submit_allowed
from backend.jev.service import clear_evaluation_cache, evaluate_symbol


def _probs(winner: str, allowed: tuple[str, ...], winner_p: float = 0.7) -> dict[str, float]:
    rest = (1.0 - winner_p) / (len(allowed) - 1)
    return {name: (winner_p if name == winner else rest) for name in allowed}


def _body() -> dict:
    return {
        "model": "typesafe/jev-1.13-20260917",
        "answers": {
            "trade_action": {
                "choice": "HOLD",
                "confidence": 0.6,
                "probabilities": _probs("HOLD", TRADE_ACTIONS, 0.7),
            },
            "sentiment_spectrum": {"score": 2.0},
            "is_short_squeeze_risk": {"noul": 0.2},
            "catalyst_impact": {"score": 0.0},
            "price_direction": {
                "choice": "FLAT",
                "probabilities": _probs("FLAT", PRICE_DIRECTIONS, 0.72),
            },
        },
        "usage": {"input_tokens": 1200, "output_tokens": 20, "cost": 0.00005},
    }


def _bars() -> list[dict]:
    rows = []
    price = 100.0
    for index in range(20):
        price += 0.4
        rows.append({
            "open": price - 0.2,
            "high": price + 0.3,
            "low": price - 0.3,
            "close": price,
            "volume": 5 + index,
        })
    return rows


def test_provider_selection_uses_openrouter(monkeypatch):
    monkeypatch.setenv("JEV_PROVIDER", "openrouter")
    client = build_client()
    assert client.name == "openrouter"
    assert build_client("typesafe").name == "typesafe"
    assert build_client("mock").name == "mock"


@pytest.mark.asyncio
async def test_missing_openrouter_key_fails_closed(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    reset_circuit()
    client = OpenRouterJevClient()
    with pytest.raises(JevUnavailable, match="OPENROUTER_API_KEY"):
        await client.system_one({"asset": "BTC"}, {"trade_action": {}})


@pytest.mark.asyncio
async def test_schema_rejection_and_pinned_slug(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("JEV_OPENROUTER_MODEL", "typesafe/jev-1.13")
    reset_circuit()
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        body = json.loads(request.content)
        seen["model"] = body["model"]
        return httpx.Response(200, json={"answers": {}})

    client = OpenRouterJevClient(transport=httpx.MockTransport(handler))
    with pytest.raises(JevUnavailable, match="schema"):
        await client.system_one({"asset": "BTC"}, {"trade_action": {"type": "choice"}})
    assert seen["url"].endswith("/systemone")
    assert seen["auth"] == "Bearer test-openrouter-key"
    assert seen["model"] == "typesafe/jev-1.13"
    reset_circuit()


@pytest.mark.asyncio
async def test_openrouter_answer_cannot_size_or_submit(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("JEV_OPENROUTER_MODEL", "typesafe/jev-1.13")
    monkeypatch.setenv("JEV_INFLUENCE_BOOK", "true")
    monkeypatch.setenv("JEV_CACHE_SECONDS", "0")
    monkeypatch.setenv("JEV_JOURNAL_PATH", str(tmp_path / "journal.db"))
    reset_circuit()
    clear_evaluation_cache()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_body())

    reset_journal_cache()
    result = await evaluate_symbol(
        "BTC",
        bars=_bars(),
        include_social=False,
        fetch_bars=False,
        client=OpenRouterJevClient(transport=httpx.MockTransport(handler)),
    )
    assert result["status"] == "ok"
    assert result["signal"] == "neutral"
    assert result["order_size_fraction"] == 0.0
    assert result["sizing_allowed"] is False
    assert result["influence_book"] is False
    assert order_size_fraction(probability=0.99) == 0.0
    assert live_submit_allowed() is False
    reset_circuit()


@pytest.mark.asyncio
async def test_openrouter_credit_error_pauses(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("JEV_PAUSE_MS", "60000")
    reset_circuit()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": "insufficient credits"})

    client = OpenRouterJevClient(transport=httpx.MockTransport(handler))
    with pytest.raises(JevUnavailable, match="402"):
        await client.system_one({"asset": "BTC"}, {})
    assert circuit_open() is True
    reset_circuit()
