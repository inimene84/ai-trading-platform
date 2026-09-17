"""n8n enrichment ingest: Influx measurements + market-data routes."""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.services.influxdb_writer import InfluxDBWriter


@pytest.fixture
def writer():
    w = InfluxDBWriter.__new__(InfluxDBWriter)
    w.url = "http://influx.test"
    w.token = "tok"
    w.org = "org"
    w._enabled = True
    w._write = AsyncMock()
    return w


@pytest.mark.asyncio
async def test_write_onchain_signal_measurement(writer):
    await writer.write_onchain_signal(
        symbol="btcusdt",
        score=0.4,
        direction="bullish",
        whale_notional=12.5,
        oi_change_pct=-1.2,
        funding_rate=0.0001,
        impact_score=0.3,
        source="on-chain-whale",
    )
    writer._write.assert_awaited_once()
    args = writer._write.await_args.args
    assert args[0] == InfluxDBWriter.BUCKET_NEWS
    assert args[1] == "onchain_signal"
    assert args[2]["symbol"] == "BTCUSDT"
    assert args[2]["direction"] == "BULLISH"
    assert args[3]["score"] == 0.4


@pytest.mark.asyncio
async def test_write_macro_technical_divergence_measurements(writer):
    await writer.write_macro_signal(symbol="BTCUSDT", vix=28.0, risk_regime="RISK_OFF")
    await writer.write_technical_signal(symbol="ETHUSDT", score=-0.7, direction="BEARISH", rsi=72.0)
    await writer.write_divergence_alert(symbol="SOLUSDT", score=0.2, signal="SELL_DUE_TO_SENTIMENT")
    names = [c.args[1] for c in writer._write.await_args_list]
    assert names == ["macro_signal", "technical_signal", "divergence_alert"]


def test_enrichment_routes_store_and_list(auth_headers, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    monkeypatch.setenv("CONFIRM_LIVE_DEPLOY", "true")
    from backend.main import app
    from backend.routes import market_data as md

    md._LAST_WRITES["on-chain"] = []
    md._LAST_WRITES["macro"] = []
    md._LAST_WRITES["technical"] = []
    md._LAST_WRITES["divergence"] = []

    client = TestClient(app, raise_server_exceptions=False)
    with patch("backend.routes.market_data.influx") as influx:
        influx.write_onchain_signal = AsyncMock()
        influx.write_macro_signal = AsyncMock()
        influx.write_technical_signal = AsyncMock()
        influx.write_divergence_alert = AsyncMock()

        r1 = client.post(
            "/api/market-data/on-chain",
            headers=auth_headers,
            json={"symbol": "BTCUSDT", "whale_sentiment": "NEUTRAL", "whale_score": 0},
        )
        assert r1.status_code == 200, r1.text
        assert r1.json()["kind"] == "on-chain"
        influx.write_onchain_signal.assert_awaited()

        r2 = client.post(
            "/api/market-data/macro",
            headers=auth_headers,
            json={"source": "macro-correlation", "vix": 14.0, "risk_regime": "RISK_ON"},
        )
        assert r2.status_code == 200
        assert r2.json()["data"][0]["direction"] == "BULLISH"

        r3 = client.post(
            "/api/market-data/technical",
            headers=auth_headers,
            json={
                "symbol": "ETHUSDT",
                "overall_signal": "BEARISH",
                "rsi": 71.2,
                "confidence": 0.7,
                "price_change_4h": 2.5,
            },
        )
        assert r3.status_code == 200
        tech = client.get("/api/market-data/technical")
        assert tech.status_code == 200
        body = tech.json()
        assert body["count"] == 1
        assert body["output"][0]["json"]["symbol"] == "ETHUSDT"

        r4 = client.post(
            "/api/alerts/divergence",
            headers=auth_headers,
            json={
                "symbol": "BTCUSDT",
                "direction": "BULLISH",
                "price_change": -0.03,
                "signal": "SELL_DUE_TO_SENTIMENT",
                "confidence": 0.03,
            },
        )
        assert r4.status_code == 200
        assert r4.json()["kind"] == "divergence"
        influx.write_divergence_alert.assert_awaited()

        listed = client.get("/openapi.json").json()["paths"]
        wanted = [
            "/api/market-data/on-chain",
            "/api/market-data/macro",
            "/api/market-data/technical",
            "/api/market-data/divergence",
            "/api/alerts/divergence",
        ]
        for path in wanted:
            assert path in listed
