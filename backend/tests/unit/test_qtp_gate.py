from __future__ import annotations

import sys
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import qtp_gate  # noqa: E402


def test_cookie_roundtrip() -> None:
    secret = b"x" * 32
    token = qtp_gate.sign_cookie(secret, now=1_000_000, nonce="abc")
    assert qtp_gate.cookie_valid(secret, token, now=1_000_001)
    assert not qtp_gate.cookie_valid(secret, token, now=1_000_000 + qtp_gate.COOKIE_TTL_SEC + 1)
    assert not qtp_gate.cookie_valid(secret, "v1.1.n.deadbeef", now=1_000_001)
    assert not qtp_gate.cookie_valid(b"y" * 32, token, now=1_000_001)


def test_cookie_rejects_garbage() -> None:
    secret = b"x" * 32
    assert not qtp_gate.cookie_valid(secret, "")
    assert not qtp_gate.cookie_valid(secret, "nope")
    assert not qtp_gate.cookie_valid(secret, "v1.notanint.n.ab")
    assert not qtp_gate.cookie_valid(secret, "v1.9999999999.n.ééé")


def _start_server() -> ThreadingHTTPServer:
    qtp_gate.Handler.secret = b"s" * 32
    qtp_gate.Handler.expected_user = "admin"
    qtp_gate.Handler.expected_password = "secret"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), qtp_gate.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def _request(
    httpd: ThreadingHTTPServer,
    method: str,
    path: str,
    body: str | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes, dict[str, str]]:
    conn = HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=3)
    hdrs = headers or {}
    if body is not None:
        payload = body.encode()
        hdrs = {**hdrs, "Content-Length": str(len(payload))}
        conn.request(method, path, body=payload, headers=hdrs)
    else:
        conn.request(method, path, headers=hdrs)
    res = conn.getresponse()
    data = res.read()
    got = {k.lower(): v for k, v in res.getheaders()}
    conn.close()
    return res.status, data, got


def test_http_contract() -> None:
    qtp_gate._FAILS.clear()
    httpd = _start_server()
    try:
        status, body, hdrs = _request(httpd, "GET", "/login")
        assert status == 200
        assert b"Sign in" in body
        assert "www-authenticate" not in hdrs

        status, body, hdrs = _request(
            httpd,
            "POST",
            "/login",
            body="username=nope&password=nope",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Real-Ip": "203.0.113.9",
            },
        )
        assert status == 200, status
        assert b"Wrong username or password" in body
        assert "www-authenticate" not in hdrs

        status, body, hdrs = _request(
            httpd,
            "POST",
            "/login",
            body="username=admin&password=secret",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert status == 302
        cookie = hdrs["set-cookie"]
        assert "HttpOnly" in cookie and "Secure" in cookie
        token = cookie.split(";", 1)[0].split("=", 1)[1]

        status, _, hdrs = _request(httpd, "GET", "/verify", headers={"Cookie": f"qtp_gate={token}"})
        assert status == 200

        status, body, hdrs = _request(
            httpd,
            "GET",
            "/verify",
            headers={"X-Forwarded-Uri": "/api/backend/trading/positions", "Accept": "application/json"},
        )
        assert status == 401
        assert b"login required" in body
        assert "www-authenticate" not in hdrs

        status, _, hdrs = _request(httpd, "GET", "/verify", headers={"X-Forwarded-Uri": "/"})
        assert status == 302
        assert hdrs["location"].endswith("/login")

        conn = HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=3)
        conn.request("POST", "/login", body=b"x", headers={"Content-Length": "-1"})
        res = conn.getresponse()
        status = res.status
        res.read()
        conn.close()
        assert status == 400

        flipped = token[:-1] + ("0" if token[-1] != "0" else "1")
        status, _, hdrs = _request(
            httpd, "GET", "/verify", headers={"Cookie": f"qtp_gate={flipped}"}
        )
        assert status == 302
        assert hdrs["location"].endswith("/login")
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_login_throttle_uses_real_ip_not_xff() -> None:
    qtp_gate._FAILS.clear()
    httpd = _start_server()
    try:
        fail_headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Real-Ip": "203.0.113.50",
            "X-Forwarded-For": "198.51.100.7",
        }
        for _ in range(qtp_gate._FAIL_MAX):
            status, _, _ = _request(
                httpd,
                "POST",
                "/login",
                body="username=nope&password=nope",
                headers=fail_headers,
            )
            assert status == 200
        status, _, _ = _request(
            httpd,
            "POST",
            "/login",
            body="username=nope&password=nope",
            headers=fail_headers,
        )
        assert status == 429
        status, _, _ = _request(
            httpd,
            "POST",
            "/login",
            body="username=nope&password=nope",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Real-Ip": "203.0.113.51",
                "X-Forwarded-For": "203.0.113.50",
            },
        )
        assert status == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
