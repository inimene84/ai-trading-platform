"""Offline unit tests for the async LangGraph bear-veto debate layer."""
import asyncio
import json
from unittest.mock import patch

from backend.agents.debate import bear_veto


async def _fake_call_meta(task, prompt, system="", temperature=None, max_tokens=None, response_json=False):
    if "KILL this trade" in prompt:
        return json.dumps({"points": ["thin edge"], "strongest": "fees", "confidence": 0.8}), {
            "model": "mock-bear",
            "provider": "omniroute",
        }
    if "Chief Risk Judge" in system:
        return json.dumps({"veto": True, "bear_strength": 0.7, "reason": "fee churn"}), {
            "model": "mock-judge",
            "provider": "omniroute",
        }
    return json.dumps({"points": ["trend up"], "strongest": "momentum", "confidence": 0.6}), {
        "model": "mock-bull",
        "provider": "omniroute",
    }


async def _noop_persist(_row):
    return True


def test_bear_veto_blocks_in_enforce():
    with (
        patch.object(bear_veto, "call_llm_resilient_meta", _fake_call_meta),
        patch.object(bear_veto, "_persist_verdict", _noop_persist),
        patch.object(bear_veto, "BEAR_VETO_MODE", "enforce"),
    ):
        out = asyncio.run(
            bear_veto.evaluate_bear_veto(
                {"symbol": "BTCUSDC", "direction": "BUY", "signal_detail": {}, "market_snapshot": {}}
            )
        )
        assert out["veto"] is True
        assert out["veto_effective"] is True
        assert out["verdict"]["bear_strength"] == 0.7
        assert out["model"] == "mock-judge"


def test_shadow_never_blocks():
    with (
        patch.object(bear_veto, "call_llm_resilient_meta", _fake_call_meta),
        patch.object(bear_veto, "_persist_verdict", _noop_persist),
        patch.object(bear_veto, "BEAR_VETO_MODE", "shadow"),
    ):
        out = asyncio.run(
            bear_veto.evaluate_bear_veto(
                {"symbol": "BTCUSDC", "direction": "BUY", "signal_detail": {}, "market_snapshot": {}}
            )
        )
        assert out["veto"] is True
        assert out["veto_effective"] is False


def test_off_short_circuits():
    with patch.object(bear_veto, "BEAR_VETO_MODE", "off"):
        out = asyncio.run(bear_veto.evaluate_bear_veto({"symbol": "X", "direction": "BUY"}))
        assert out["mode"] == "off"
        assert out["veto_effective"] is False


def test_llm_failure_abstains_and_fails_open():
    async def _boom(*_a, **_kw):
        raise RuntimeError("provider down")

    with (
        patch.object(bear_veto, "call_llm_resilient_meta", _boom),
        patch.object(bear_veto, "_persist_verdict", _noop_persist),
        patch.object(bear_veto, "BEAR_VETO_MODE", "enforce"),
    ):
        out = asyncio.run(
            bear_veto.evaluate_bear_veto(
                {"symbol": "BTCUSDC", "direction": "BUY", "signal_detail": {}, "market_snapshot": {}}
            )
        )
        assert out["abstain"] is True
        assert out["veto"] is False
        assert out["veto_effective"] is False
