"""Fast Jev revaluator: slim questions, fail closed, no sizing."""

from __future__ import annotations

import pytest

from backend.jev.client import JevUnavailable
from backend.jev.gateway import MockJevClient
from backend.jev.research import lint_questions
from backend.jev.revalue import (
    clear_revalue_cache,
    revalue_questions,
    revalue_symbol,
    revalue_tape,
    validate_revalue,
)
from backend.jev.schema import JevSchemaError


def _bars(count: int = 20) -> list[dict]:
    rows = []
    price = 100.0
    for index in range(count):
        price += 1.0
        rows.append({
            "open": price - 0.4,
            "high": price + 0.6,
            "low": price - 0.8,
            "close": price,
            "volume": 10 + index,
        })
    return rows


def _probs(winner: str, allowed: tuple[str, ...], winner_p: float = 0.7) -> dict[str, float]:
    rest = (1.0 - winner_p) / (len(allowed) - 1)
    return {name: (winner_p if name == winner else rest) for name in allowed}


def _raw(side: str = "long", risk: str = "watch") -> dict:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "higher_30d": {"noul": 0.74},
            "side": {
                "choice": side,
                "confidence": 0.81,
                "probabilities": _probs(side, ("long", "flat", "short"), 0.7),
            },
            "conviction": {"score": 3},
            "risk_level": {
                "choice": risk,
                "confidence": 0.66,
                "probabilities": _probs(risk, ("low", "watch", "high", "freeze"), 0.64),
            },
        },
    }


class Live:
    name = "typesafe"

    def __init__(self, payload: dict | None = None, fail: Exception | None = None) -> None:
        self.payload = payload if payload is not None else _raw()
        self.fail = fail
        self.calls = 0
        self.questions: dict | None = None

    async def system_one(self, state, questions, validator=None):
        self.calls += 1
        self.questions = questions
        assert state["representative_posts"] == []
        assert state["social_stats"]["sample_size"] == 0
        if self.fail:
            raise self.fail
        if validator is None:
            raise AssertionError("revalue must pass its own validator")
        return validator(self.payload)


def test_revalue_questions_pass_lint_and_stay_slim():
    questions = revalue_questions()
    assert lint_questions(questions) == []
    assert set(questions) == {"higher_30d", "side", "conviction", "risk_level"}


def test_validate_revalue_rejects_a_bad_side():
    broken = _raw()
    broken["answers"]["side"]["choice"] = "BUY"
    with pytest.raises(JevSchemaError):
        validate_revalue(broken)


@pytest.mark.asyncio
async def test_revalue_keeps_a_headline_and_still_does_not_size():
    clear_revalue_cache()
    card = await revalue_symbol(
        "BTC",
        bars=_bars(),
        client=Live(),
        fetch_bars=False,
        headlines=["Bitcoin funding turns positive after the flush"],
    )
    assert card["sizing_allowed"] is False
    assert card["influence_book"] is False
    assert any("Bitcoin funding turns positive" in line for line in card["evidence"])
    assert card["pattern"].endswith("no order")


@pytest.mark.asyncio
async def test_revalue_symbol_returns_an_advisory_card():
    clear_revalue_cache()
    client = Live()
    card = await revalue_symbol("btc", bars=_bars(), client=client, fetch_bars=False)
    assert card["status"] == "ok"
    assert card["source"] == "jev"
    assert card["symbol"] == "BTC"
    assert card["side"] == "long"
    assert card["verdict"] == "ALLOW"
    assert card["sizing_allowed"] is False
    assert card["influence_book"] is False
    assert card["advisory"] is True
    assert card["entry"] is not None
    assert client.calls == 1
    assert "trade_action" not in (client.questions or {})


@pytest.mark.asyncio
async def test_revalue_vetoes_a_losing_book():
    clear_revalue_cache()
    card = await revalue_symbol(
        "ETH",
        bars=_bars(),
        client=Live(),
        fetch_bars=False,
        loss_frac=-0.03,
    )
    assert card["verdict"] == "VETO"
    assert card["side"] == "long"
    assert card["sizing_allowed"] is False
    assert any("Session loss" in line for line in card["vetoes"])


@pytest.mark.asyncio
async def test_revalue_stands_aside_when_jev_is_down():
    clear_revalue_cache()
    card = await revalue_symbol(
        "SOL",
        bars=_bars(),
        client=Live(fail=JevUnavailable("timeout")),
        fetch_bars=False,
    )
    assert card["status"] == "no_signal"
    assert card["source"] == "unavailable"
    assert card["side"] == "flat"
    assert card["verdict"] == "STAND ASIDE"
    assert card["entry"] is None
    assert card["sizing_allowed"] is False


@pytest.mark.asyncio
async def test_revalue_stands_aside_on_a_bad_schema():
    clear_revalue_cache()
    broken = _raw()
    del broken["answers"]["side"]["probabilities"]
    card = await revalue_symbol("SOL", bars=_bars(), client=Live(payload=broken), fetch_bars=False)
    assert card["source"] == "unavailable"
    assert card["side"] == "flat"
    assert card["verdict"] == "STAND ASIDE"


@pytest.mark.asyncio
async def test_mock_provider_is_not_a_live_revalue():
    clear_revalue_cache()
    card = await revalue_symbol("BTC", bars=_bars(), client=MockJevClient(), fetch_bars=False)
    assert card["source"] == "mock"
    assert card["side"] == "flat"
    assert card["sizing_allowed"] is False


@pytest.mark.asyncio
async def test_revalue_caches_the_same_close():
    clear_revalue_cache()
    client = Live()
    first = await revalue_symbol("BTC", bars=_bars(), client=client, fetch_bars=False)
    second = await revalue_symbol("BTC", bars=_bars(), client=client, fetch_bars=False)
    assert first["source"] == "jev"
    assert second["cached"] is True
    assert client.calls == 1


@pytest.mark.asyncio
async def test_tape_keeps_going_when_one_symbol_fails():
    clear_revalue_cache()

    class Split:
        name = "typesafe"

        async def system_one(self, state, questions, validator=None):
            if state["asset"] == "ETH":
                raise JevUnavailable("eth down")
            return validator(_raw())

    result = await revalue_tape(
        ["BTC", "ETH", "BTC"],
        client=Split(),
        bars_by_symbol={"BTC": _bars(), "ETH": _bars()},
        fetch_bars=False,
    )
    assert result["sizing_allowed"] is False
    assert result["influence_book"] is False
    assert result["called_jev"] is True
    by_symbol = {card["symbol"]: card for card in result["cards"]}
    assert set(by_symbol) == {"BTC", "ETH"}
    assert by_symbol["BTC"]["source"] == "jev"
    assert by_symbol["ETH"]["source"] == "unavailable"
    assert by_symbol["ETH"]["side"] == "flat"
