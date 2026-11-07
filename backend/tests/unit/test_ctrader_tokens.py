"""Isolated cTrader token-store refresh regressions (no broker / OAuth network)."""
import threading
import time
from unittest.mock import MagicMock, patch

import httpx

from backend.services.ctrader_tokens import (
    DEFAULT_EXPIRES_IN_SEC,
    REFRESH_COOLDOWN_SEC,
    REFRESH_HTTP_TIMEOUT_SEC,
    CTraderTokenStore,
)

PRIMARY = "https://openapi.ctrader.com/apps/token"
FALLBACK = "https://connect.spotware.com/apps/token"


def _fresh_tokens(**overrides):
    data = {
        "access_token": "acc_old",
        "refresh_token": "ref_single_use",
        "client_id": "cid",
        "client_secret": "sec",
        "account_id": 1,
        "updated_at": int(time.time()),
        "expires_in": DEFAULT_EXPIRES_IN_SEC,
    }
    data.update(overrides)
    return data


def _store(tmp_path, **overrides) -> CTraderTokenStore:
    store = CTraderTokenStore(file_path=tmp_path / "ctrader_tokens.json")
    store.save_tokens(_fresh_tokens(**overrides))
    return store


def _ok_response(access="acc_new", refresh="ref_new"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "accessToken": access,
        "refreshToken": refresh,
        "expiresIn": DEFAULT_EXPIRES_IN_SEC,
    }
    resp.text = ""
    return resp


def test_env_bootstrap_keeps_thirty_day_expires_default(tmp_path, monkeypatch):
    monkeypatch.setenv("CTRADER_ACCESS_TOKEN", "acc_env")
    monkeypatch.setenv("CTRADER_REFRESH_TOKEN", "ref_env")
    monkeypatch.setenv("CTRADER_CLIENT_ID", "cid_env")
    monkeypatch.setenv("CTRADER_CLIENT_SECRET", "sec_env")
    monkeypatch.setenv("CTRADER_ACCOUNT_ID", "42")
    store = CTraderTokenStore(file_path=tmp_path / "missing.json")
    tokens = store.get_tokens()
    assert tokens["expires_in"] == 2592000
    assert tokens["expires_in"] == DEFAULT_EXPIRES_IN_SEC


def test_refresh_posts_with_thirty_second_timeout(tmp_path):
    store = _store(tmp_path)
    captured = {}

    def fake_post(url, data=None, timeout=None):
        captured["timeout"] = timeout
        captured["url"] = url
        return _ok_response()

    with patch("backend.services.ctrader_tokens._ctrader_token_urls", return_value=(PRIMARY, FALLBACK)), patch(
        "backend.services.ctrader_tokens.httpx.post", side_effect=fake_post
    ):
        assert store.refresh_if_needed(force=True) == "acc_new"
    assert captured["timeout"] == REFRESH_HTTP_TIMEOUT_SEC
    assert captured["timeout"] == 30.0
    assert captured["url"] == PRIMARY


def test_timeout_does_not_resubmit_refresh_token_to_fallback(tmp_path):
    store = _store(tmp_path)
    urls: list[str] = []

    def fake_post(url, data=None, timeout=None):
        urls.append(url)
        raise httpx.TimeoutException("read timed out")

    with patch("backend.services.ctrader_tokens._ctrader_token_urls", return_value=(PRIMARY, FALLBACK)), patch(
        "backend.services.ctrader_tokens.httpx.post", side_effect=fake_post
    ):
        assert store.refresh_if_needed(force=True) == "acc_old"
    assert urls == [PRIMARY]


def test_connect_error_still_tries_fallback_url(tmp_path):
    store = _store(tmp_path)
    urls: list[str] = []

    def fake_post(url, data=None, timeout=None):
        urls.append(url)
        if url == PRIMARY:
            raise httpx.ConnectError("dns failed")
        return _ok_response()

    with patch("backend.services.ctrader_tokens._ctrader_token_urls", return_value=(PRIMARY, FALLBACK)), patch(
        "backend.services.ctrader_tokens.httpx.post", side_effect=fake_post
    ):
        assert store.refresh_if_needed(force=True) == "acc_new"
    assert urls == [PRIMARY, FALLBACK]


def test_cooldown_skips_second_force_refresh(tmp_path):
    store = _store(tmp_path)
    posts = []

    def fake_post(url, data=None, timeout=None):
        posts.append(url)
        return _ok_response()

    with patch("backend.services.ctrader_tokens._ctrader_token_urls", return_value=(PRIMARY,)), patch(
        "backend.services.ctrader_tokens.httpx.post", side_effect=fake_post
    ):
        assert store.refresh_if_needed(force=True) == "acc_new"
        assert store.refresh_if_needed(force=True) == "acc_new"
    assert posts == [PRIMARY]
    assert REFRESH_COOLDOWN_SEC == 600.0


def test_cooldown_also_blocks_proactive_refresh_after_timeout(tmp_path):
    """has_credentials() refreshes without force; cooldown must still hold after timeout."""
    store = _store(tmp_path, updated_at=int(time.time()) - (DEFAULT_EXPIRES_IN_SEC - 1000))
    posts = []

    def fake_post(url, data=None, timeout=None):
        posts.append(url)
        raise httpx.TimeoutException("read timed out")

    with patch("backend.services.ctrader_tokens._ctrader_token_urls", return_value=(PRIMARY, FALLBACK)), patch(
        "backend.services.ctrader_tokens.httpx.post", side_effect=fake_post
    ):
        assert store.refresh_if_needed(force=True) == "acc_old"
        assert store.refresh_if_needed(force=False) == "acc_old"
    assert posts == [PRIMARY]


def test_concurrent_force_refresh_is_single_flight(tmp_path):
    store = _store(tmp_path)
    posts: list[str] = []
    started = threading.Event()
    release = threading.Event()

    def fake_post(url, data=None, timeout=None):
        posts.append(url)
        started.set()
        assert release.wait(timeout=2.0)
        return _ok_response()

    with patch("backend.services.ctrader_tokens._ctrader_token_urls", return_value=(PRIMARY, FALLBACK)), patch(
        "backend.services.ctrader_tokens.httpx.post", side_effect=fake_post
    ):
        t1 = threading.Thread(target=lambda: store.refresh_if_needed(force=True))
        t2 = threading.Thread(target=lambda: store.refresh_if_needed(force=True))
        t1.start()
        assert started.wait(timeout=2.0)
        t2.start()
        # t1 holds the lock across the POST; t2 waits, then cooldown skips a
        # second POST of the same refresh token.
        release.set()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

    assert posts == [PRIMARY]
    assert t1.is_alive() is False
    assert t2.is_alive() is False
