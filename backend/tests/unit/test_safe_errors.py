"""Tests for safe error responses, log-injection filtering and the Binance proxy allowlist."""

import json
import logging

import pytest

from backend.utils.safe_errors import (
    LogInjectionFilter,
    internal_error_response,
    sanitize_log_text,
)


def test_internal_error_response_hides_exception_text(caplog):
    log = logging.getLogger("test.safe_errors")
    exc = RuntimeError("secret path /etc/app/.env and token abc123")
    with caplog.at_level(logging.ERROR, logger="test.safe_errors"):
        resp = internal_error_response(log, "Trending error", exc)
    body = json.loads(resp.body)
    assert resp.status_code == 500
    assert "secret path" not in json.dumps(body)
    assert body["error"] == "Trending error (internal error)"
    assert len(body["error_id"]) == 12
    # The full detail is still in the server log, tagged with the same id.
    assert body["error_id"] in caplog.text
    assert "secret path" in caplog.text


def test_sanitize_log_text_removes_line_breaks():
    forged = "BTCUSDT\nERROR:root:fake admin login"
    out = sanitize_log_text(forged)
    assert "\n" not in out
    assert "\\n" in out


def test_log_injection_filter_cleans_msg_and_args():
    record = logging.LogRecord(
        "x", logging.INFO, __file__, 1, "sym=%s\r\nforged", ("A\nB",), None
    )
    assert LogInjectionFilter().filter(record) is True
    assert "\n" not in record.getMessage()
    assert "\r" not in record.getMessage()


@pytest.mark.asyncio
async def test_binance_proxy_rejects_unlisted_endpoint():
    from starlette.requests import Request

    from backend.routes.trading import binance_proxy

    request = Request({"type": "http", "query_string": b"", "headers": []})
    resp = await binance_proxy("../../evil.example/x", request)
    assert resp.status_code == 400
    assert json.loads(resp.body) == {"error": "endpoint not allowed"}


def test_binance_proxy_endpoints_are_server_constants():
    from backend.routes.trading import _BINANCE_PROXY_ALLOWED, _BINANCE_PROXY_ENDPOINTS

    assert set(_BINANCE_PROXY_ENDPOINTS) == _BINANCE_PROXY_ALLOWED
    assert all(k == v for k, v in _BINANCE_PROXY_ENDPOINTS.items())
