"""
cTrader Open API OAuth 2.0 helper.

Spotware OAuth endpoints:
  Authorize: https://connect.spotware.com/apps/auth
  Token:     https://connect.spotware.com/apps/token

Full OAuth flow (one-time setup):
  1. Build the authorize URL (see get_auth_url())
  2. Open it in a browser, log in with your cTID
  3. Approve the demo hedging account (same CTRADER_ACCOUNT_ID) unless
     CTRADER_ENV=live is an explicit live-money confirmation
  4. Copy the `code` from the redirect URL
  5. Call exchange_code_for_token(code) → saves tokens to .env / file
  Connect/enable then uses demo.ctraderapi.com when CTRADER_ENV=demo.

Afterwards, use refresh_access_token() to renew silently.
"""

import os
import json
import httpx
from pathlib import Path
from urllib.parse import urlencode
from dotenv import load_dotenv, set_key

load_dotenv()

from backend.services.ctrader_oauth import auth_url as _ctrader_auth_url, token_urls as _ctrader_token_urls

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def _normalize_spotware_tokens(data: dict) -> dict:
    if data.get("errorCode"):
        raise ValueError(f"{data.get('errorCode')}: {data.get('description')}")
    access = data.get("accessToken") or data.get("access_token")
    refresh = data.get("refreshToken") or data.get("refresh_token")
    if not access:
        raise ValueError(f"No access token in Spotware response: {data}")
    return {
        "access_token": access,
        "refresh_token": refresh or "",
        "expires_in": int(data.get("expiresIn") or data.get("expires_in") or 2_628_000),
        "token_type": data.get("tokenType") or data.get("token_type") or "bearer",
    }


def _post_spotware_token(payload: dict) -> dict:
    last_err = None
    for url in _ctrader_token_urls():
        try:
            resp = httpx.post(url, data=payload, timeout=20.0)
            data = resp.json()
            if resp.status_code == 200 and not data.get("errorCode"):
                return _normalize_spotware_tokens(data)
            last_err = f"{url}: {data.get('errorCode') or resp.status_code} {data.get('description') or resp.text[:200]}"
        except Exception as exc:
            last_err = f"{url}: {exc}"
    raise RuntimeError(last_err or "Spotware token request failed")


def get_auth_url(
    client_id: str | None = None,
    redirect_uri: str = "https://localhost/callback",
    scope: str = "trading",
) -> str:
    """
    Build the OAuth authorization URL.
    Open this URL in a browser to get an authorization code.
    """
    client_id = client_id or os.getenv("CTRADER_CLIENT_ID")
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "response_type": "code",
    }
    return f"{_ctrader_auth_url()}?{urlencode(params)}"


def exchange_code_for_token(
    code: str,
    redirect_uri: str = "https://localhost/callback",
    client_id: str | None = None,
    client_secret: str | None = None,
    save_to_env: bool = True,
) -> dict:
    """
    Exchange authorization code for access + refresh tokens.
    Optionally writes them back to .env.
    """
    client_id = client_id or os.getenv("CTRADER_CLIENT_ID")
    client_secret = client_secret or os.getenv("CTRADER_CLIENT_SECRET")

    resp = httpx.post(
        _ctrader_token_urls()[0],
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "client_secret": client_secret,
        },
    )
    try:
        tokens = _normalize_spotware_tokens(resp.json())
    except ValueError:
        tokens = _post_spotware_token(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "client_secret": client_secret,
            }
        )

    if save_to_env and ENV_FILE.exists():
        set_key(str(ENV_FILE), "CTRADER_ACCESS_TOKEN", tokens["access_token"])
        set_key(str(ENV_FILE), "CTRADER_REFRESH_TOKEN", tokens.get("refresh_token", ""))
        print("Tokens saved to .env")

    return tokens


def refresh_access_token(
    refresh_token: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    save_to_env: bool = True,
) -> dict:
    """
    Use refresh token to get a new access token without user interaction.
    Call this on startup if the access token may have expired.
    """
    refresh_token = refresh_token or os.getenv("CTRADER_REFRESH_TOKEN")
    client_id = client_id or os.getenv("CTRADER_CLIENT_ID")
    client_secret = client_secret or os.getenv("CTRADER_CLIENT_SECRET")

    tokens = _post_spotware_token(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        }
    )

    if save_to_env and ENV_FILE.exists():
        set_key(str(ENV_FILE), "CTRADER_ACCESS_TOKEN", tokens["access_token"])
        if "refresh_token" in tokens:
            set_key(str(ENV_FILE), "CTRADER_REFRESH_TOKEN", tokens["refresh_token"])

    return tokens


# ── CLI helper ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("Auth URL (open in browser):")
        print(get_auth_url())
        print("\nThen run:  python -m src.brokers.auth <code>")
    elif sys.argv[1] == "refresh":
        tokens = refresh_access_token()
        print("Refreshed:", json.dumps(tokens, indent=2))
    else:
        code = sys.argv[1]
        tokens = exchange_code_for_token(code)
        print("Tokens:", json.dumps(tokens, indent=2))
