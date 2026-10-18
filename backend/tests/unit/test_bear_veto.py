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


def test_default_mode_is_shadow_and_never_blocks(monkeypatch):
    monkeypatch.delenv("BEAR_VETO_MODE", raising=False)
    monkeypatch.setattr(bear_veto, "BEAR_VETO_MODE", "shadow")
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
        assert out["mode"] == "shadow"
        assert out["veto"] is True
        assert out["veto_effective"] is False


def test_persist_never_posts_with_anon_key(monkeypatch):
    anon_jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJyb2xlIjoiYW5vbiJ9."
        "sig"
    )
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", anon_jwt)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    with patch("backend.agents.debate.bear_veto.httpx.AsyncClient") as client_cls:
        ok = asyncio.run(bear_veto._persist_verdict({"symbol": "BTCUSDC", "mode": "shadow"}))
    assert ok is False
    client_cls.assert_not_called()


def test_persist_exception_is_swallowed():
    async def _boom(_row):
        raise RuntimeError("supabase down")

    with (
        patch.object(bear_veto, "call_llm_resilient_meta", _fake_call_meta),
        patch.object(bear_veto, "_persist_verdict", _boom),
        patch.object(bear_veto, "BEAR_VETO_MODE", "shadow"),
    ):
        out = asyncio.run(
            bear_veto.evaluate_bear_veto(
                {"symbol": "BTCUSDC", "direction": "BUY", "signal_detail": {}, "market_snapshot": {}}
            )
        )
        assert out["veto_effective"] is False
        assert out["veto"] is True
