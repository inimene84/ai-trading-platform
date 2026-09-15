import json
import pytest
import httpx
from unittest.mock import AsyncMock, patch

from backend.services.opensearch_client import OpenSearchClient


@pytest.mark.asyncio
async def test_ping_success():
    client = OpenSearchClient(base_url="http://mock-os:9200")
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = httpx.Response(200, json={"version": {"number": "2.11.0"}})
        assert await client.ping() is True


@pytest.mark.asyncio
async def test_ping_timeout_returns_false():
    client = OpenSearchClient(base_url="http://mock-os:9200")
    with patch("httpx.AsyncClient.get", side_effect=httpx.TimeoutException("timeout")):
        assert await client.ping() is False


@pytest.mark.asyncio
async def test_search_timeout_returns_empty_list():
    client = OpenSearchClient(base_url="http://mock-os:9200")
    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("timed out")):
        res = await client.search("qt-decisions", {"query": {"match_all": {}}})
        assert res == []


@pytest.mark.asyncio
async def test_search_success_parses_hits():
    client = OpenSearchClient(base_url="http://mock-os:9200")
    payload = {
        "hits": {
            "total": {"value": 1},
            "hits": [
                {"_index": "qt-decisions", "_id": "1", "_source": {"symbol": "BTCUSDC", "verdict": "REJECT"}}
            ]
        }
    }
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = httpx.Response(200, json=payload)
        res = await client.search("qt-decisions", {"query": {"match_all": {}}})
        assert len(res) == 1
        assert res[0]["symbol"] == "BTCUSDC"
        assert res[0]["verdict"] == "REJECT"


@pytest.mark.asyncio
async def test_bulk_payload_shape():
    client = OpenSearchClient(base_url="http://mock-os:9200")
    docs = [
        {"symbol": "BTCUSDC", "signal": "BUY"},
        {"symbol": "ETHUSDC", "signal": "SELL"},
    ]
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = httpx.Response(
            200,
            json={"items": [{"index": {"status": 201}}, {"index": {"status": 201}}]}
        )
        indexed_count = await client.bulk_index("qt-decisions", docs)
        assert indexed_count == 2
        
        # Verify body passed to httpx.post
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["headers"]["Content-Type"] == "application/x-ndjson"
        content = call_kwargs["content"]
        lines = [line for line in content.split("\n") if line]
        assert len(lines) == 4
        assert json.loads(lines[0]) == {"index": {"_index": "qt-decisions"}}
        assert json.loads(lines[1]) == docs[0]
        assert json.loads(lines[2]) == {"index": {"_index": "qt-decisions"}}
        assert json.loads(lines[3]) == docs[1]
