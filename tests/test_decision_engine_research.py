import pytest
from unittest.mock import AsyncMock, patch

from backend.services.decision_engine import DecisionEngine
from backend.services.risk_config import RiskConfig



def _make_bars(n=60, base_price=50000.0):
    bars = []
    for i in range(n):
        bars.append({
            "timestamp": 1700000000 + i * 900,
            "open": base_price + i,
            "high": base_price + i + 10,
            "low": base_price + i - 10,
            "close": base_price + i + 2,
            "volume": 100.0,
        })
    return bars


@pytest.mark.asyncio
async def test_evaluate_symbol_still_returns_none_on_veto_when_ingest_raises(monkeypatch):
    monkeypatch.setenv("RESEARCH_INGEST_ENABLED", "true")
    config = RiskConfig()
    engine = DecisionEngine(config)

    # Ingest error should be swallowed, returning None as expected for cooldown/short bars
    with patch("backend.services.research_ingest.opensearch_client.bulk_index", side_effect=RuntimeError("OS is down")):
        # bars < 50 triggers immediate veto
        res = await engine.evaluate_symbol("BTCUSDC", _make_bars(10), None, 0, [], False)
        assert res is None


@pytest.mark.asyncio
async def test_evaluate_symbol_veto_produces_decision_doc_when_flag_on(monkeypatch):
    monkeypatch.setenv("RESEARCH_INGEST_ENABLED", "true")
    config = RiskConfig()
    engine = DecisionEngine(config)

    with patch("backend.services.research_ingest.opensearch_client.bulk_index", new_callable=AsyncMock) as mock_bulk:
        mock_bulk.return_value = 1
        # bars < 50 triggers veto "insufficient bars"
        res = await engine.evaluate_symbol("BTCUSDC", _make_bars(10), None, 0, [], False)
        assert res is None
        import asyncio
        await asyncio.sleep(0.01)
        assert mock_bulk.call_count == 1
        call_args = mock_bulk.call_args[0]
        assert call_args[0] == "qt-decisions"
        doc = call_args[1][0]
        assert doc["symbol"] == "BTCUSDC"
        assert doc["rejected"] is True
        assert "insufficient bars" in doc["reason"]


@pytest.mark.asyncio
async def test_evaluate_symbol_zero_calls_when_flag_off(monkeypatch):
    monkeypatch.setenv("RESEARCH_INGEST_ENABLED", "false")
    config = RiskConfig()
    engine = DecisionEngine(config)

    with patch("backend.services.research_ingest.opensearch_client.bulk_index", new_callable=AsyncMock) as mock_bulk:
        res = await engine.evaluate_symbol("BTCUSDC", _make_bars(10), None, 0, [], False)
        assert res is None
        mock_bulk.assert_not_called()
