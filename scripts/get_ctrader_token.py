#!/usr/bin/env python3
"""
cTrader Open API OAuth helper — request fresh demo/live tokens.

Usage:
  python scripts/get_ctrader_token.py url
      Print the browser authorization URL (open it, approve access, copy ?code= from redirect).

  python scripts/get_ctrader_token.py exchange <authorization_code>
      Exchange the code for access + refresh tokens; writes .env and data/ctrader_tokens.json.

  python scripts/get_ctrader_token.py refresh
      Refresh using CTRADER_REFRESH_TOKEN (if still valid).

Requires in .env:
  CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET
Optional:
  CTRADER_REDIRECT_URI (default https://localhost/callback)
  CTRADER_ACCOUNT_ID
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv, set_key

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
TOKENS_FILE = ROOT / "data" / "ctrader_tokens.json"

from backend.services.ctrader_oauth import (
    auth_url as _ctrader_auth_url,
    ctrader_env as _ctrader_env,
    is_sandbox_app,
    protobuf_host as _protobuf_host,
    token_urls as _ctrader_token_urls,
)


def _load_env() -> None:
    load_dotenv(ENV_FILE)


def _redirect_uri() -> str:
    return os.getenv("CTRADER_REDIRECT_URI", "https://localhost/callback").strip()


def _client_creds() -> tuple[str, str]:
    cid = os.getenv("CTRADER_CLIENT_ID", "").strip()
    secret = os.getenv("CTRADER_CLIENT_SECRET", "").strip()
    if not cid or not secret:
        raise SystemExit("CTRADER_CLIENT_ID and CTRADER_CLIENT_SECRET must be set in .env")
    return cid, secret


def _normalize_token_response(data: dict) -> dict:
    if data.get("errorCode"):
        raise RuntimeError(f"{data.get('errorCode')}: {data.get('description')}")
    access = data.get("accessToken") or data.get("access_token")
    refresh = data.get("refreshToken") or data.get("refresh_token")
    if not access:
        raise RuntimeError(f"No access token in response: {data}")
    expires = int(data.get("expiresIn") or data.get("expires_in") or 2_628_000)
    return {
        "access_token": access,
        "refresh_token": refresh or "",
        "expires_in": expires,
        "token_type": data.get("tokenType") or data.get("token_type") or "bearer",
        "updated_at": int(time.time()),
    }


def _post_token(payload: dict) -> dict:
    last_err = None
    for url in _ctrader_token_urls():
        try:
            resp = httpx.post(url, data=payload, timeout=20.0)
            data = resp.json()
            if resp.status_code == 200 and not data.get("errorCode"):
                return _normalize_token_response(data)
            last_err = f"{url} -> {data.get('errorCode') or resp.status_code}: {data.get('description') or resp.text[:200]}"
        except Exception as exc:
            last_err = f"{url} -> {exc}"
    raise RuntimeError(last_err or "token request failed")


def build_auth_url() -> str:
    _load_env()
    cid, _ = _client_creds()
    redirect = _redirect_uri()
    try:
        from ctrader_open_api import Auth

        auth = Auth(cid, os.getenv("CTRADER_CLIENT_SECRET", ""), redirect)
        return auth.getAuthUri()
    except ImportError:
        from urllib.parse import urlencode

        params = {
            "client_id": cid,
            "redirect_uri": redirect,
            "scope": "trading",
            "response_type": "code",
        }
        return f"{_ctrader_auth_url()}?{urlencode(params)}"


def exchange_code(code: str) -> dict:
    _load_env()
    cid, secret = _client_creds()
    redirect = _redirect_uri()
    code = code.strip()
    if not code:
        raise SystemExit("authorization code is required")

    tokens = _post_token(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect,
            "client_id": cid,
            "client_secret": secret,
        }
    )
    return _persist(tokens)


def refresh_tokens() -> dict:
    _load_env()
    cid, secret = _client_creds()
    refresh = os.getenv("CTRADER_REFRESH_TOKEN", "").strip()
    if not refresh:
        raise SystemExit("CTRADER_REFRESH_TOKEN missing — run `url` + `exchange` flow instead")

    tokens = _post_token(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh,
            "client_id": cid,
            "client_secret": secret,
        }
    )
    return _persist(tokens)


def _persist(tokens: dict) -> dict:
    cid = os.getenv("CTRADER_CLIENT_ID", "").strip()
    secret = os.getenv("CTRADER_CLIENT_SECRET", "").strip()
    account_raw = os.getenv("CTRADER_ACCOUNT_ID", "0").strip()
    account_id = int(account_raw) if account_raw.isdigit() else 0

    record = {
        "client_id": cid,
        "client_secret": secret,
        "account_id": account_id,
        **tokens,
    }

    if ENV_FILE.exists():
        set_key(str(ENV_FILE), "CTRADER_ACCESS_TOKEN", tokens["access_token"])
        set_key(str(ENV_FILE), "CTRADER_REFRESH_TOKEN", tokens["refresh_token"])
        if account_id:
            set_key(str(ENV_FILE), "CTRADER_ACCOUNT_ID", str(account_id))
        print(f"Updated {ENV_FILE}")

    TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKENS_FILE.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved {TOKENS_FILE}")

    return record


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    cmd = sys.argv[1].lower()
    if cmd == "url":
        print(build_auth_url())
        print(f"\nRedirect URI registered in your Spotware app must match:\n  {_redirect_uri()}")
        print(f"Account type CTRADER_ENV={_ctrader_env()} → protobuf host {_protobuf_host()}")
        print("Approve the demo hedging account (same CTRADER_ACCOUNT_ID), not a live-money book.")
        print("\nAfter approving access, copy the `code` query param from the redirect URL.")
        print("Then run:  python scripts/get_ctrader_token.py exchange <code>")
        print("Then POST /trading/ctrader/enable — host must stay demo.ctraderapi.com for a demo account.")
        return

    if cmd == "exchange":
        if len(sys.argv) < 3:
            raise SystemExit("usage: get_ctrader_token.py exchange <authorization_code>")
        record = exchange_code(sys.argv[2])
        print("OK — new tokens saved.")
        print(json.dumps({k: ("***" if "token" in k else v) for k, v in record.items()}, indent=2))
        return

    if cmd == "refresh":
        record = refresh_tokens()
        print("OK — tokens refreshed.")
        print(json.dumps({k: ("***" if "token" in k else v) for k, v in record.items()}, indent=2))
        return

    raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
