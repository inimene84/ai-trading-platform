"""Unit tests for MCP server URL helpers and HTTP method support."""

import importlib
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _load_mcp_server(monkeypatch):
    # Avoid importing real fastmcp if missing — inject a stub.
    if "fastmcp" not in sys.modules:
        stub = ModuleType("fastmcp")

        class FastMCP:
            def __init__(self, *args, **kwargs):
                self.tools = []

            def tool(self, fn=None, **kwargs):
                def deco(f):
                    self.tools.append(f)
                    return f

                return deco if fn is None else deco(fn)

            @classmethod
            def from_openapi(cls, *args, **kwargs):
                return cls()

            def run(self, *args, **kwargs):
                return None

        stub.FastMCP = FastMCP
        sys.modules["fastmcp"] = stub

    monkeypatch.setenv("BACKEND_BASE_URL", "http://backend:8000")
    monkeypatch.setenv("BACKEND_API_PREFIX", "/api")
    monkeypatch.setenv("MCP_API_TOKEN", "test-token")
    if "mcp_server.server" in sys.modules:
        del sys.modules["mcp_server.server"]
    return importlib.import_module("mcp_server.server")


def test_url_prepends_api_prefix(monkeypatch):
    mod = _load_mcp_server(monkeypatch)
    assert mod._url("/trading/positions") == "http://backend:8000/api/trading/positions"
    assert mod._url("/api/trading/positions") == "http://backend:8000/api/trading/positions"
    # Sentry + health are dual-mounted under /api on the backend
    assert mod._url("/sentry/status") == "http://backend:8000/api/sentry/status"
    assert mod._url("/health") == "http://backend:8000/api/health"


def test_headers_include_bearer_and_api_key(monkeypatch):
    mod = _load_mcp_server(monkeypatch)
    h = mod._headers()
    assert h["Authorization"] == "Bearer test-token"
    assert h["X-API-Key"] == "test-token"


@pytest.mark.asyncio
async def test_call_backend_supports_put(monkeypatch):
    mod = _load_mcp_server(monkeypatch)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.request = AsyncMock(return_value=mock_resp)

    with patch.object(mod.httpx, "AsyncClient", return_value=mock_client):
        out = await mod.call_backend(
            "PUT",
            "/trading/positions/7/modify",
            body={"stop_loss": 1.0},
        )
    assert out["result"]["ok"] is True
    args, kwargs = mock_client.request.call_args
    assert args[0] == "PUT"
    assert kwargs["json"] == {"stop_loss": 1.0}


@pytest.mark.asyncio
async def test_get_trading_opinion_posts_analyze(monkeypatch):
    mod = _load_mcp_server(monkeypatch)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"direction": "HOLD"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.request = AsyncMock(return_value=mock_resp)

    with patch.object(mod.httpx, "AsyncClient", return_value=mock_client):
        out = await mod.get_trading_opinion("btcusdc")
    assert out["direction"] == "HOLD"
    args, kwargs = mock_client.request.call_args
    assert args[0] == "POST"
    assert args[1].endswith("/api/trading/opinion/analyze")
    assert kwargs["json"]["symbol"] == "BTCUSDC"


@pytest.mark.asyncio
async def test_research_fetch_posts_to_scrapling_sidecar(monkeypatch):
    mod = _load_mcp_server(monkeypatch)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"url": "https://example.com", "content": "ok"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.request = AsyncMock(return_value=mock_resp)

    with patch.object(mod.httpx, "AsyncClient", return_value=mock_client):
        out = await mod.research_fetch("https://example.com", css_selector="h1")
    assert out["content"] == "ok"
    args, kwargs = mock_client.request.call_args
    assert args[0] == "POST"
    assert args[1].endswith("/api/research/fetch")
    assert kwargs["json"]["url"] == "https://example.com"
    assert kwargs["json"]["css_selector"] == "h1"


@pytest.mark.asyncio
async def test_search_research_news_and_decisions(monkeypatch):
    mod = _load_mcp_server(monkeypatch)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"symbol": "BTCUSDC", "verdict": "REJECT"}]
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.request = AsyncMock(return_value=mock_resp)

    with patch.object(mod.httpx, "AsyncClient", return_value=mock_client):
        res = await mod.search_research(kind="decisions", symbol="BTCUSDC", verdict="REJECT")
    assert len(res) == 1
    assert res[0]["verdict"] == "REJECT"
    args, kwargs = mock_client.request.call_args
    assert args[0] == "GET"
    assert args[1].endswith("/api/research/decisions")
    assert kwargs["params"]["symbol"] == "BTCUSDC"
    assert kwargs["params"]["verdict"] == "REJECT"

