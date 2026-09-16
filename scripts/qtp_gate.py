#!/usr/bin/env python3
"""Cookie login gate for trading.thorai.cloud (replaces HTTP Basic Auth).

Traefik ForwardAuth calls GET /verify with the browser Cookie.
Public GET/POST /login and GET /logout are routed here without auth.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
import re
import secrets
import sys
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

COOKIE_NAME = "qtp_gate"
COOKIE_TTL_SEC = 7 * 24 * 3600
LISTEN = ("0.0.0.0", int(os.environ.get("QTP_GATE_PORT", "8089")))
PUBLIC_HOST = os.environ.get("QTP_PUBLIC_HOST", "trading.thorai.cloud")
CREDS_FILE = Path(os.environ.get("QTP_CREDS_FILE", "/run/secrets/qtp-ui.txt"))
SECRET_FILE = Path(os.environ.get("QTP_GATE_SECRET_FILE", "/run/secrets/qtp-gate-secret"))
MAX_POST = 4096
MAX_FAIL_KEYS = 4096

_FAILS: dict[str, list[float]] = {}
_FAIL_LOCK = threading.Lock()
_FAIL_WINDOW = 600.0
_FAIL_MAX = 8
_HEX_SIG = re.compile(r"^[0-9a-f]{64}$")


def _read_secret() -> bytes:
    raw = SECRET_FILE.read_bytes().strip()
    if len(raw) < 32:
        raise SystemExit("gate secret file is too short")
    return raw


def load_creds() -> tuple[str, str]:
    user = ""
    password = ""
    for line in CREDS_FILE.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "username":
            user = value.strip()
        elif key.strip() == "password":
            password = value.rstrip("\r\n")
    if not user or not password:
        raise SystemExit("qtp-ui.txt missing username or password")
    return user, password


def _mac(secret: bytes, msg: str) -> str:
    return hmac.new(secret, msg.encode("utf-8"), hashlib.sha256).hexdigest()


def _fixed_compare(left: str, right: str) -> bool:
    left_d = hashlib.sha256(left.encode("utf-8")).digest()
    right_d = hashlib.sha256(right.encode("utf-8")).digest()
    return hmac.compare_digest(left_d, right_d)


def sign_cookie(secret: bytes, now: int | None = None, nonce: str | None = None) -> str:
    exp = int(now if now is not None else time.time()) + COOKIE_TTL_SEC
    nonce = nonce or secrets.token_urlsafe(16)
    msg = f"v1.{exp}.{nonce}"
    return f"{msg}.{_mac(secret, msg)}"


def cookie_valid(secret: bytes, value: str, now: int | None = None) -> bool:
    parts = (value or "").split(".")
    if len(parts) != 4 or parts[0] != "v1":
        return False
    _, exp_s, nonce, sig = parts
    if not nonce or not _HEX_SIG.fullmatch(sig):
        return False
    try:
        exp = int(exp_s)
    except ValueError:
        return False
    if exp < int(now if now is not None else time.time()):
        return False
    msg = f"v1.{exp}.{nonce}"
    return hmac.compare_digest(_mac(secret, msg), sig)


def _too_many_fails(ip: str) -> bool:
    now = time.time()
    with _FAIL_LOCK:
        hits = [t for t in _FAILS.get(ip, []) if now - t < _FAIL_WINDOW]
        if hits:
            _FAILS[ip] = hits
        else:
            _FAILS.pop(ip, None)
        return len(hits) >= _FAIL_MAX


def _record_fail(ip: str) -> None:
    now = time.time()
    with _FAIL_LOCK:
        if ip not in _FAILS and len(_FAILS) >= MAX_FAIL_KEYS:
            oldest = next(iter(_FAILS))
            _FAILS.pop(oldest, None)
        hits = [t for t in _FAILS.get(ip, []) if now - t < _FAIL_WINDOW]
        hits.append(now)
        _FAILS[ip] = hits


def _html(error: str = "") -> bytes:
    err = ""
    if error:
        err = (
            '<p style="color:#fb7185;font-size:13px;margin:0 0 16px">'
            "Wrong username or password.</p>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>QuantumTrade — Sign in</title>
  <style>
    body {{ margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
      background:#09090b; color:#e4e4e7; font-family:Inter,system-ui,sans-serif; }}
    .card {{ width:min(380px,92vw); background:#18181b; border:1px solid #27272a; border-radius:16px; padding:28px; }}
    h1 {{ margin:0 0 6px; font-size:18px; }}
    p.sub {{ margin:0 0 22px; color:#a1a1aa; font-size:13px; }}
    label {{ display:block; font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:#71717a; margin:0 0 6px; }}
    input {{ width:100%; box-sizing:border-box; background:#09090b; border:1px solid #3f3f46;
      border-radius:10px; color:#fafafa; padding:10px 12px; margin:0 0 14px; font-size:14px; }}
    button {{ width:100%; border:0; border-radius:10px; padding:11px; background:#10b981; color:#052e16;
      font-weight:700; cursor:pointer; }}
  </style>
</head>
<body>
  <form class="card" method="post" action="/login" autocomplete="on">
    <h1>QuantumTrade</h1>
    <p class="sub">Sign in to open the trading desk. This is not your broker or API key.</p>
    {err}
    <label for="username">Username</label>
    <input id="username" name="username" autocomplete="username" required autofocus/>
    <label for="password">Password</label>
    <input id="password" name="password" type="password" autocomplete="current-password" required/>
    <button type="submit">Continue</button>
  </form>
</body>
</html>""".encode()


class Handler(BaseHTTPRequestHandler):
    timeout = 10
    secret = b""
    expected_user = ""
    expected_password = ""
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _client_ip(self) -> str:
        # Traefik sets X-Real-Ip to the TCP client. Ignore X-Forwarded-For
        # (leftmost hop is attacker-controlled on the public login POST).
        candidate = (self.headers.get("X-Real-Ip") or "").strip()
        try:
            ipaddress.ip_address(candidate)
            return candidate
        except ValueError:
            return self.client_address[0]

    def _cookie_value(self) -> str:
        raw = self.headers.get("Cookie", "")
        jar = SimpleCookie()
        try:
            jar.load(raw)
        except Exception:
            return ""
        morsel = jar.get(COOKIE_NAME)
        return morsel.value if morsel else ""

    def _send(self, code: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        extra = headers or {}
        extra.setdefault("Cache-Control", "no-store")
        extra.setdefault("Connection", "close")
        self.close_connection = True
        self.send_response(code)
        for key, val in extra.items():
            self.send_header(key, val)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/healthz":
            self._send(200, b"ok", {"Content-Type": "text/plain"})
            return
        if path == "/verify":
            self._verify()
            return
        if path == "/logout":
            self._send(
                302,
                b"",
                {
                    "Location": "/login",
                    "Set-Cookie": f"{COOKIE_NAME}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax",
                },
            )
            return
        if path == "/login":
            self._send(200, _html(), {"Content-Type": "text/html; charset=utf-8"})
            return
        self._send(404, b"not found", {"Content-Type": "text/plain"})

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path != "/login":
            self._send(404, b"not found", {"Content-Type": "text/plain"})
            return
        ip = self._client_ip()
        if _too_many_fails(ip):
            self._send(429, _html("x"), {"Content-Type": "text/html; charset=utf-8"})
            return
        raw_len = self.headers.get("Content-Length", "")
        if not raw_len.isdigit():
            self._send(400, b"bad request", {"Content-Type": "text/plain"})
            return
        length = int(raw_len)
        if length > MAX_POST:
            self._send(413, b"too large", {"Content-Type": "text/plain"})
            return
        raw = self.rfile.read(length).decode("utf-8", "replace")
        form = parse_qs(raw, keep_blank_values=True)
        user = (form.get("username") or [""])[0]
        password = (form.get("password") or [""])[0]
        if not (
            _fixed_compare(user, self.expected_user)
            and _fixed_compare(password, self.expected_password)
        ):
            _record_fail(ip)
            # 200, not 401: a 401 here re-opens the browser password popup.
            self._send(200, _html("x"), {"Content-Type": "text/html; charset=utf-8"})
            return
        token = sign_cookie(self.secret)
        self._send(
            302,
            b"",
            {
                "Location": "/",
                "Set-Cookie": (
                    f"{COOKIE_NAME}={token}; Path=/; Max-Age={COOKIE_TTL_SEC}; "
                    "HttpOnly; Secure; SameSite=Lax"
                ),
            },
        )

    def _verify(self) -> None:
        if cookie_valid(self.secret, self._cookie_value()):
            self._send(200, b"ok", {"Content-Type": "text/plain"})
            return
        uri = self.headers.get("X-Forwarded-Uri") or ""
        accept = self.headers.get("Accept", "")
        path = urlparse(uri).path
        apiish = (
            path.startswith("/api")
            or path.startswith("/health")
            or path.startswith("/docs")
            or path.startswith("/grafana")
            or "application/json" in accept
        )
        if apiish:
            self._send(401, b'{"error":"login required"}', {"Content-Type": "application/json"})
            return
        self._send(302, b"", {"Location": f"https://{PUBLIC_HOST}/login"})


def main() -> None:
    Handler.secret = _read_secret()
    Handler.expected_user, Handler.expected_password = load_creds()
    httpd = ThreadingHTTPServer(LISTEN, Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
