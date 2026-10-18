"""Postmortem → FinMem embedding path, including the OpenAI fail-soft backend."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.finmem_service import FinMemService
from backend.services.postmortem import embed_to_finmem, on_trade_closed
from backend.services.provider_credit_circuit import credit_circuit
from backend.utils import embeddings


def _pm(**overrides) -> dict:
    row = {
        "trade_id": "binance_futures:paper:99",
        "symbol": "BTCUSDT",
        "direction": "BUY",
        "cause": "stop_placement",
        "realized_pnl": -15.0,
        "exit_reason": "stop_loss",
        "reflection": "Widen stop outside the 1x ATR noise band.",
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_embed_to_finmem_calls_finmem_store_memory():
    store = AsyncMock(return_value="mem-123")
    with patch("backend.services.finmem_service.finmem_service") as svc:
        svc.store_memory = store
        ok = await embed_to_finmem(_pm())
    assert ok is True
    store.assert_awaited_once()
    kwargs = store.await_args.kwargs
    assert kwargs["symbol"] == "BTCUSDT"
    assert kwargs["layer"] == "deep"
    assert kwargs["source_type"] == "postmortem"
    assert kwargs["base_importance"] == 85.0
    assert "POSTMORTEM" in kwargs["content"]
    assert kwargs["sentiments"]["negative"] == 0.75


@pytest.mark.asyncio
async def test_embed_to_finmem_winner_uses_lower_importance():
    store = AsyncMock(return_value="mem-win")
    with patch("backend.services.finmem_service.finmem_service") as svc:
        svc.store_memory = store
        ok = await embed_to_finmem(_pm(realized_pnl=12.0, cause="winner"))
    assert ok is True
    assert store.await_args.kwargs["base_importance"] == 70.0
    assert store.await_args.kwargs["sentiments"]["positive"] == 0.75


@pytest.mark.asyncio
async def test_embed_to_finmem_fail_soft_on_missing_memory_store():
    """The live bug: FinMemService had no memory_store. Must not raise."""
    class _Broken:
        pass

    with patch("backend.services.finmem_service.finmem_service", _Broken()):
        assert await embed_to_finmem(_pm()) is False


@pytest.mark.asyncio
async def test_embed_to_finmem_fail_soft_on_store_error():
    store = AsyncMock(side_effect=RuntimeError("qdrant down"))
    with patch("backend.services.finmem_service.finmem_service") as svc:
        svc.store_memory = store
        assert await embed_to_finmem(_pm()) is False


@pytest.mark.asyncio
async def test_on_trade_closed_writes_even_when_embed_fails():
    upsert = AsyncMock(return_value=True)

    async def _fake_llm(*_a, **_kw):
        return "scratch", {"model": "mock"}

    with (
        patch("backend.llm.router.call_llm_resilient_meta", _fake_llm),
        patch("backend.services.postmortem._supabase_upsert", upsert),
        patch("backend.services.postmortem.embed_to_finmem", AsyncMock(return_value=False)),
    ):
        pm = await on_trade_closed(
            {
                "id": 7,
                "symbol": "ETHUSDT",
                "direction": "SELL",
                "broker": "binance_futures",
                "mode": "paper",
                "pnl": -4.0,
                "exit_reason": "stop_loss",
            }
        )
    assert pm is not None
    upsert.assert_awaited_once()
    assert upsert.await_args.args[0] == "postmortems"


def test_finmem_service_exposes_memory_store_and_store_memory(monkeypatch):
    monkeypatch.setattr("backend.services.finmem_service.QDRANT_AVAILABLE", False)
    svc = FinMemService()
    assert svc.memory_store is svc.memory
    assert callable(svc.store_memory)
    assert svc.memory._client is None


@pytest.mark.asyncio
async def test_finmem_service_store_memory_delegates(monkeypatch):
    monkeypatch.setattr("backend.services.finmem_service.QDRANT_AVAILABLE", False)
    svc = FinMemService()
    svc.memory.store_memory = AsyncMock(return_value="delegated-id")
    mem_id = await svc.store_memory(
        symbol="SOLUSDT",
        layer="deep",
        content="postmortem text",
        source_type="postmortem",
        base_importance=85.0,
        sentiments={"positive": 0.0, "neutral": 0.2, "negative": 0.8},
    )
    assert mem_id == "delegated-id"
    svc.memory.store_memory.assert_awaited_once()
    kwargs = svc.memory.store_memory.await_args.kwargs
    assert kwargs["layer"] == "deep"
    assert kwargs["source_type"] == "postmortem"


@pytest.mark.asyncio
async def test_layered_memory_uses_generated_embedding(monkeypatch):
    monkeypatch.setattr("backend.services.finmem_service.QDRANT_AVAILABLE", False)
    captured = {}

    async def _fake_embed(text, *, vector_size=1536, **_kw):
        captured["text"] = text
        captured["vector_size"] = vector_size
        return [0.01] * vector_size

    upsert = AsyncMock()
    monkeypatch.setattr("backend.services.finmem_service.generate_text_embedding", _fake_embed)
    svc = FinMemService()
    svc.memory._client = MagicMock()
    svc.memory._client.upsert = upsert
    svc.memory.ensure_collection = AsyncMock(return_value=True)

    mem_id = await svc.store_memory(
        symbol="BTCUSDT",
        layer="deep",
        content="[POSTMORTEM] BTCUSDT BUY cause=stop_placement",
        source_type="postmortem",
        base_importance=85.0,
    )
    assert mem_id
    assert captured["vector_size"] == svc.memory.vector_size
    assert "POSTMORTEM" in captured["text"]
    upsert.assert_awaited_once()
    point = upsert.await_args.kwargs["points"][0]
    assert len(point.vector) == svc.memory.vector_size
    assert point.payload["source_type"] == "postmortem"
    assert point.payload["layer"] == "deep"


def test_openai_model_id_strips_provider_prefix():
    assert embeddings._openai_model_id("openai/text-embedding-3-small") == "text-embedding-3-small"
    assert embeddings._openai_model_id("text-embedding-3-small") == "text-embedding-3-small"
    assert embeddings._openai_model_id("") == "text-embedding-3-small"


def test_usable_api_key_rejects_placeholders():
    assert embeddings._usable_api_key("sk-test") is True
    assert embeddings._usable_api_key("") is False
    assert embeddings._usable_api_key("your_openai_api_key_here") is False
    assert embeddings._usable_api_key(None) is False


@pytest.mark.asyncio
async def test_openai_embed_missing_key_returns_none(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert await embeddings._openai_embed("hello", "text-embedding-3-small") is None


def test_openai_embed_url_defaults_and_strips_blank(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert embeddings._openai_embed_url() == "https://api.openai.com/v1/embeddings"
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    assert embeddings._openai_embed_url() == "https://api.openai.com/v1/embeddings"
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1/")
    assert embeddings._openai_embed_url() == "https://proxy.example/v1/embeddings"


@pytest.mark.asyncio
async def test_openai_embed_401_trips_embed_circuit_not_chat(monkeypatch):
    credit_circuit.reset()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")

    class _Resp:
        status_code = 401

        def raise_for_status(self):
            raise RuntimeError("HTTP 401 invalid_api_key")

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr(embeddings.httpx, "AsyncClient", _Client)
    with pytest.raises(RuntimeError):
        await embeddings._openai_embed("hello", "text-embedding-3-small")
    assert credit_circuit.is_open("openai-embed") is True
    assert credit_circuit.is_open("openai") is False
    credit_circuit.reset()


@pytest.mark.asyncio
async def test_openai_embed_calls_openai_api(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("EMBED_DIMENSIONS", "1536")
    captured = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"embedding": [0.2] * 1536}]}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return _Resp()

    monkeypatch.setattr(embeddings.httpx, "AsyncClient", _Client)
    vec = await embeddings._openai_embed("pm text", "openai/text-embedding-3-small")
    assert vec is not None
    assert len(vec) == 1536
    assert captured["json"]["model"] == "text-embedding-3-small"
    assert captured["json"]["dimensions"] == 1536
    assert captured["json"]["input"] == "pm text"
    assert captured["headers"]["Authorization"] == "Bearer sk-test-openai"
    assert captured["url"].endswith("/embeddings")


@pytest.mark.asyncio
async def test_generate_text_embedding_uses_openai_when_openrouter_circuit_open(monkeypatch):
    credit_circuit.reset()
    credit_circuit.trip("openrouter", "402")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")

    async def fail_openrouter(_text, _model):
        raise AssertionError("OpenRouter should stay skipped while the circuit is open")

    async def fake_openai(_text, _model):
        return [0.5, 0.5, 0.0]

    async def fail_litellm(_text, _model):
        raise AssertionError("LiteLLM should not run after OpenAI succeeds")

    monkeypatch.setattr(embeddings, "_openrouter_embed", fail_openrouter)
    monkeypatch.setattr(embeddings, "_openai_embed", fake_openai)
    monkeypatch.setattr(embeddings, "_litellm_embed", fail_litellm)

    vec = await embeddings.generate_text_embedding("hello", vector_size=3, normalize=False)
    assert vec == [0.5, 0.5, 0.0]
    credit_circuit.reset()


@pytest.mark.asyncio
async def test_generate_text_embedding_falls_back_to_openai(monkeypatch):
    credit_circuit.reset()
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")

    async def fail_openrouter(_text, _model):
        raise RuntimeError("openrouter down")

    async def fail_litellm(_text, _model):
        raise AssertionError("LiteLLM should not run after OpenAI succeeds")

    async def fake_openai(_text, _model):
        return [0.3, 0.4, 0.3]

    monkeypatch.setattr(embeddings, "_openrouter_embed", fail_openrouter)
    monkeypatch.setattr(embeddings, "_openai_embed", fake_openai)
    monkeypatch.setattr(embeddings, "_litellm_embed", fail_litellm)

    vec = await embeddings.generate_text_embedding("hello", vector_size=3, normalize=False)
    assert vec == [0.3, 0.4, 0.3]
    credit_circuit.reset()


@pytest.mark.asyncio
async def test_generate_text_embedding_openai_error_is_fail_soft(monkeypatch):
    credit_circuit.reset()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")

    async def boom(_text, _model):
        raise RuntimeError("429 rate limit")

    async def no_litellm(_text, _model):
        raise RuntimeError("litellm down")

    monkeypatch.setattr(embeddings, "_openai_embed", boom)
    monkeypatch.setattr(embeddings, "_litellm_embed", no_litellm)
    vec = await embeddings.generate_text_embedding("hello", vector_size=3, normalize=False)
    assert vec is None
    credit_circuit.reset()
