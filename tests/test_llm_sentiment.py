import json
import pytest
from unittest.mock import AsyncMock, patch

from backend.services.llm_sentiment import (
    SentimentLabel,
    classify_bundle,
    create_default_sentiment,
)


@pytest.mark.asyncio
async def test_classify_bundle_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RESEARCH_SENTIMENT_ENABLED", raising=False)
    result = await classify_bundle("BTCUSDC", ["Headline 1"])
    assert result.stance == "unclear"
    assert result.should_block_entry is False
    assert "disabled" in result.summary.lower()


@pytest.mark.asyncio
async def test_classify_bundle_empty_headlines(monkeypatch):
    monkeypatch.setenv("RESEARCH_SENTIMENT_ENABLED", "true")
    result = await classify_bundle("BTCUSDC", [])
    assert result.stance == "unclear"
    assert result.should_block_entry is False


@pytest.mark.asyncio
async def test_classify_bundle_timeout_fail_soft(monkeypatch):
    monkeypatch.setenv("RESEARCH_SENTIMENT_ENABLED", "true")
    with patch("backend.llm.router.call_llm_resilient", side_effect=TimeoutError("LLM timed out")):
        result = await classify_bundle("BTCUSDC", ["Headline 1", "Headline 2"])
        assert result.stance == "unclear"
        assert result.should_block_entry is False


@pytest.mark.asyncio
async def test_sec_etf_approval_not_should_block_entry(monkeypatch):
    monkeypatch.setenv("RESEARCH_SENTIMENT_ENABLED", "true")
    mock_payload = {
        "symbol": "BTCUSDC",
        "horizon": "intraday",
        "stance": "bullish",
        "event_type": "etf",
        "severity": 1,
        "confidence": 90,
        "should_block_entry": False,
        "summary": "SEC approves spot Bitcoin ETF applications.",
    }
    with patch("backend.llm.router.call_llm_resilient", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = json.dumps(mock_payload)
        result = await classify_bundle("BTCUSDC", ["SEC approves spot Bitcoin ETF listing"])
        assert result.stance == "bullish"
        assert result.event_type == "etf"
        assert result.should_block_entry is False
