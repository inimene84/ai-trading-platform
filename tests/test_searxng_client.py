import hashlib
import pytest
import httpx
from unittest.mock import patch

from backend.services.searxng_client import SearXNGClient, url_hash


def test_url_hash():
    url = "https://example.com/crypto-news-1"
    expected = hashlib.sha256(url.encode("utf-8")).hexdigest()
    assert url_hash(url) == expected


@pytest.mark.asyncio
async def test_searxng_search_parsing_and_dedup():
    client = SearXNGClient(base_url="http://mock-searxng:8080")
    fake_response = {
        "results": [
            {
                "title": "Bitcoin surges past 70k",
                "url": "https://example.com/btc-news",
                "content": "BTC hits new record...",
                "engine": "bing",
            },
            # Duplicate URL with different casing/whitespace or exact duplicate
            {
                "title": "Bitcoin surges past 70k (mirror)",
                "url": "https://example.com/btc-news",
                "content": "BTC hits new record...",
                "engine": "google",
            },
            {
                "title": "Ethereum ETF update",
                "url": "https://example.com/eth-news",
                "content": "SEC delays ETH ETF decision",
                "engine": "duckduckgo",
            }
        ]
    }

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = httpx.Response(200, json=fake_response)
        results = await client.search("crypto market", categories="news")
        
        # 3 items input, 2 unique URLs output
        assert len(results) == 2
        assert results[0]["title"] == "Bitcoin surges past 70k"
        assert results[0]["url"] == "https://example.com/btc-news"
        assert results[0]["engine"] == "bing"
        assert results[1]["title"] == "Ethereum ETF update"
        assert results[1]["url"] == "https://example.com/eth-news"


@pytest.mark.asyncio
async def test_searxng_timeout_returns_empty_list():
    client = SearXNGClient(base_url="http://mock-searxng:8080")
    with patch("httpx.AsyncClient.get", side_effect=httpx.TimeoutException("timeout")):
        results = await client.search("bitcoin")
        assert results == []


@pytest.mark.asyncio
async def test_searxng_500_error_returns_empty_list():
    client = SearXNGClient(base_url="http://mock-searxng:8080")
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = httpx.Response(500, text="Internal Server Error")
        results = await client.search("bitcoin")
        assert results == []
