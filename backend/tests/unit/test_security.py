import pytest
from fastapi import HTTPException
from starlette.requests import Request

from backend.security import admin_auth_enabled, is_sensitive_request, validate_admin_request
from backend.main import validate_live_startup_security


def _req(method: str, path: str) -> Request:
    scope = {"type": "http", "method": method, "path": path, "headers": []}
    return Request(scope)


def test_admin_auth_disabled_without_env(monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    assert admin_auth_enabled() is False
    validate_admin_request(_req("POST", "/trading/loop/start"))


def test_sensitive_trading_mutations(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "secret")
    assert is_sensitive_request(_req("POST", "/trading/loop/start")) is True
    assert is_sensitive_request(_req("GET", "/trading/status")) is True
    assert is_sensitive_request(_req("GET", "/trading/strategies")) is False


def test_validate_admin_request_rejects_missing_token(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "secret")
    with pytest.raises(HTTPException) as exc:
        validate_admin_request(_req("POST", "/trading/config/update"))
    assert exc.value.status_code == 401


def test_validate_admin_request_accepts_header(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "secret")
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/trading/loop/stop",
        "headers": [(b"x-api-key", b"secret")],
    }
    validate_admin_request(Request(scope))


def test_live_startup_refuses_missing_admin_token(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    for name in ("ADMIN_API_KEY", "API_AUTH_TOKEN", "BACKEND_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="Refusing LIVE startup"):
        validate_live_startup_security()


def test_live_startup_refuses_missing_confirm_live_deploy(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("ADMIN_API_KEY", "secret")
    monkeypatch.delenv("CONFIRM_LIVE_DEPLOY", raising=False)
    with pytest.raises(RuntimeError, match="CONFIRM_LIVE_DEPLOY=true is required"):
        validate_live_startup_security()


def test_live_startup_accepts_when_fully_configured(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("ADMIN_API_KEY", "secret")
    monkeypatch.setenv("CONFIRM_LIVE_DEPLOY", "true")
    validate_live_startup_security()


def test_paper_startup_allows_missing_admin_token(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "paper")
    for name in ("ADMIN_API_KEY", "API_AUTH_TOKEN", "BACKEND_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    validate_live_startup_security()


def test_jev_routes_are_sensitive():
    assert is_sensitive_request(_req("GET", "/jev/evaluate")) is True
    assert is_sensitive_request(_req("GET", "/api/jev/evaluate")) is True


def test_jesse_and_finmem_routes_are_sensitive():
    assert is_sensitive_request(_req("POST", "/jesse/sync")) is True
    assert is_sensitive_request(_req("POST", "/jesse/finmem/ingest")) is True
    assert is_sensitive_request(_req("POST", "/jesse/finmem/evaluate")) is True
    assert is_sensitive_request(_req("GET", "/jesse/ml-predict")) is True
    assert is_sensitive_request(_req("GET", "/api/jesse/status")) is True
    assert is_sensitive_request(_req("GET", "/api/jesse/models")) is True
    assert is_sensitive_request(_req("GET", "/api/jesse/promotion-status")) is True


def test_jesse_unauthenticated_requests_blocked(monkeypatch):
    from starlette.testclient import TestClient
    from backend.main import app

    monkeypatch.setenv("ADMIN_API_KEY", "test-secret-key")
    client = TestClient(app, raise_server_exceptions=False)

    # Missing token -> 401
    r_sync = client.post("/jesse/sync", json={})
    assert r_sync.status_code == 401

    r_ingest = client.post("/jesse/finmem/ingest", json={"symbol": "BTC-USDT", "layer": "shallow", "content": "news"})
    assert r_ingest.status_code == 401

    r_eval = client.post("/jesse/finmem/evaluate", json={"symbol": "BTC-USDT"})
    assert r_eval.status_code == 401

    r_predict = client.get("/jesse/ml-predict?symbol=BTC-USDT")
    assert r_predict.status_code == 401

    r_models = client.get("/api/jesse/models")
    assert r_models.status_code == 401

    r_promo = client.get("/api/jesse/promotion-status")
    assert r_promo.status_code == 401

    # Wrong token -> 403
    r_bad = client.post("/jesse/sync", json={}, headers={"X-API-Key": "wrong-key"})
    assert r_bad.status_code == 403

