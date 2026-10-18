"""Embedding helper must skip OpenRouter while the 402 circuit is open."""

import pytest

from backend.services.provider_credit_circuit import credit_circuit
from backend.utils import embeddings


@pytest.mark.asyncio
async def test_generate_text_embedding_skips_tripped_openrouter(monkeypatch):
    credit_circuit.reset()
    credit_circuit.trip("openrouter", "402")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    async def fail_openrouter(_text, _model):
        raise AssertionError("OpenRouter should not be called while the circuit is open")

    async def fake_litellm(_text, _model):
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(embeddings, "_openrouter_embed", fail_openrouter)
    monkeypatch.setattr(embeddings, "_litellm_embed", fake_litellm)

    vec = await embeddings.generate_text_embedding("hello", vector_size=3, normalize=False)
    assert vec == [0.1, 0.2, 0.3]
    credit_circuit.reset()


@pytest.mark.asyncio
async def test_openrouter_402_trips_circuit(monkeypatch):
    credit_circuit.reset()
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    class _Resp:
        status_code = 402

        def raise_for_status(self):
            raise RuntimeError("HTTP 402 insufficient credits")

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

    async def fake_litellm(_text, _model):
        return [1.0, 0.0]

    monkeypatch.setattr(embeddings, "_litellm_embed", fake_litellm)
    vec = await embeddings.generate_text_embedding("hello", vector_size=2, normalize=False)
    assert vec == [1.0, 0.0]
    assert credit_circuit.is_open("openrouter") is True
    credit_circuit.reset()
