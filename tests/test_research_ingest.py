import pytest
from unittest.mock import AsyncMock, patch

from backend.services.research_ingest import (
    format_decision_doc,
    ingest_decision,
    is_research_ingest_enabled,
)


def test_format_decision_doc_fields():
    raw_payload = {
        "symbol": "BTCUSDC",
        "broker": "binance",
        "mode": "live",
        "signal": "BUY",
        "confidence": 0.85,
        "promotion_verdict": "PROMOTE",
        "promotion_reason": "passed all gates",
        "gate_id": "DSR_FLOOR",
        "reason": "entry decision",
        "shadow": False,
        "rejected": False,
        "prompt_version": "v1",
        "model_id": "gpt-5-6-luna",
    }
    doc = format_decision_doc(raw_payload)
    assert doc["symbol"] == "BTCUSDC"
    assert doc["broker"] == "binance"
    assert doc["mode"] == "live"
    assert doc["signal"] == "BUY"
    assert doc["confidence"] == 0.85
    assert doc["promotion_verdict"] == "PROMOTE"
    assert doc["promotion_reason"] == "passed all gates"
    assert doc["gate_id"] == "DSR_FLOOR"
    assert doc["reason"] == "entry decision"
    assert doc["shadow"] is False
    assert doc["rejected"] is False
    assert doc["prompt_version"] == "v1"
    assert doc["model_id"] == "gpt-5-6-luna"
    assert "ts" in doc


@pytest.mark.asyncio
async def test_ingest_decision_flag_off(monkeypatch):
    monkeypatch.setenv("RESEARCH_INGEST_ENABLED", "false")
    with patch("backend.services.opensearch_client.opensearch_client.bulk_index", new_callable=AsyncMock) as mock_bulk:
        await ingest_decision({"symbol": "BTCUSDC", "signal": "BUY"})
        mock_bulk.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_decision_flag_on_calls_bulk(monkeypatch):
    monkeypatch.setenv("RESEARCH_INGEST_ENABLED", "true")
    with patch("backend.services.opensearch_client.opensearch_client.bulk_index", new_callable=AsyncMock) as mock_bulk:
        mock_bulk.return_value = 1
        await ingest_decision({"symbol": "BTCUSDC", "signal": "BUY"})
        mock_bulk.assert_called_once()
        args, kwargs = mock_bulk.call_args
        assert args[0] == "qt-decisions"
        assert len(args[1]) == 1
        assert args[1][0]["symbol"] == "BTCUSDC"


@pytest.mark.asyncio
async def test_ingest_decision_error_swallowed(monkeypatch):
    monkeypatch.setenv("RESEARCH_INGEST_ENABLED", "true")
    with patch("backend.services.opensearch_client.opensearch_client.bulk_index", side_effect=RuntimeError("connection refused")):
        # Must not raise
        await ingest_decision({"symbol": "BTCUSDC", "signal": "BUY"})
