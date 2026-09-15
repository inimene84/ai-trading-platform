import pytest
from starlette.testclient import TestClient
from unittest.mock import AsyncMock, patch

from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_research_routes_require_admin_auth_when_token_set(client, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "secret-test-token")
    
    # Missing token -> 401
    resp_news = client.get("/research/news")
    assert resp_news.status_code == 401

    resp_decisions = client.get("/research/decisions")
    assert resp_decisions.status_code == 401

    resp_events = client.get("/research/events")
    assert resp_events.status_code == 401

    resp_bars = client.get("/research/mt/bars")
    assert resp_bars.status_code == 401


def test_research_routes_empty_on_down(client, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "secret-test-token")
    headers = {"X-API-Key": "secret-test-token"}

    # Mock OpenSearch client returning [] when OS is down / times out
    with patch("backend.services.opensearch_client.opensearch_client.search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []

        resp = client.get("/research/decisions?symbol=BTCUSDC", headers=headers)
        assert resp.status_code == 200
        assert resp.json() == []

        resp_news = client.get("/research/news?q=bitcoin", headers=headers)
        assert resp_news.status_code == 200
        assert resp_news.json() == []

        resp_events = client.get("/research/events?currency=USD", headers=headers)
        assert resp_events.status_code == 200
        assert resp_events.json() == []

        resp_bars = client.get("/research/mt/bars?symbol=XAUUSD", headers=headers)
        assert resp_bars.status_code == 200
        assert resp_bars.json() == []


def test_research_health_status(client, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "secret-test-token")
    headers = {"X-API-Key": "secret-test-token"}
    with patch("backend.services.opensearch_client.opensearch_client.ping", new_callable=AsyncMock) as mock_os_ping, \
         patch("backend.services.searxng_client.searxng_client.ping", new_callable=AsyncMock) as mock_searx_ping, \
         patch("backend.services.mt_bridge_client.mt_bridge_client.ping", new_callable=AsyncMock) as mock_mt_ping:

        mock_os_ping.return_value = True
        mock_searx_ping.return_value = False
        mock_mt_ping.return_value = True

        resp = client.get("/research/health", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["opensearch"] is True
        assert data["searxng"] is False
        assert data["mt5_bridge"] is True
        assert data["status"] == "ok"

