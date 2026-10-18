"""402 / credit circuit must trip once and skip the provider afterwards."""

import httpx

from backend.services.provider_credit_circuit import (
    ProviderCreditCircuit,
    looks_like_credit_error,
)


def test_trip_on_402_and_skip_until_reset():
    circuit = ProviderCreditCircuit()
    assert circuit.is_open("openrouter") is False
    assert circuit.note_failure("openrouter", "HTTP 402 insufficient credits") is True
    assert circuit.is_open("openrouter") is True
    assert circuit.is_open("OPENROUTER-gemini") is True
    assert circuit.remaining_seconds("openrouter") > 0
    circuit.note_success("openrouter")
    assert circuit.is_open("openrouter") is False


def test_non_credit_errors_do_not_trip():
    circuit = ProviderCreditCircuit()
    assert circuit.note_failure("openrouter", "HTTP 500 server error") is False
    assert circuit.is_open("openrouter") is False


def test_looks_like_credit_error_reads_status():
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/embeddings")
    response = httpx.Response(402, text="payment required", request=request)
    exc = httpx.HTTPStatusError("402", request=request, response=response)
    assert looks_like_credit_error(exc) is True
    assert looks_like_credit_error(RuntimeError("timeout")) is False
