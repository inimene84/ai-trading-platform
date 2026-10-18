"""cTrader OAuth endpoint resolution (production vs sandbox apps).

Two independent knobs:

* ``CTRADER_ENV`` = account type / protobuf host.
  demo|sandbox → ``demo.ctraderapi.com``. live → ``live.ctraderapi.com``.
  A demo hedging account (virtual funds) only authenticates on the demo host.
* QT execute ("live mode") = ``TRADING_MODE=live`` + not dry-run.
  That sends real Open API orders. It does **not** mean live-money host.

Do not set ``CTRADER_ENV=live`` for a demo account. Live host + demo
accountId fails with CH_ACCESS_TOKEN_INVALID / refresh ACCESS_DENIED.
"""

from __future__ import annotations

import os

# Spotware portal "Sandbox" apps use sandbox-connect; demo/live trading apps use openapi/connect.
SANDBOX_AUTH_URL = "https://sandbox-connect.spotware.com/apps/auth"
SANDBOX_TOKEN_URL = "https://sandbox-connect.spotware.com/apps/token"
PROD_AUTH_URL = "https://connect.spotware.com/apps/auth"
PROD_TOKEN_URL = "https://openapi.ctrader.com/apps/token"
PROD_TOKEN_URL_FALLBACK = "https://connect.spotware.com/apps/token"

DEMO_PROTOBUF_HOST = "demo.ctraderapi.com"
LIVE_PROTOBUF_HOST = "live.ctraderapi.com"


def ctrader_env() -> str:
    raw = os.getenv("CTRADER_ENV", "demo").strip().lower()
    if raw in {"sandbox", "playground", "demo", "live"}:
        return raw
    return "demo"


def is_sandbox_app() -> bool:
    return ctrader_env() in {"sandbox", "playground"}


def auth_url() -> str:
    return SANDBOX_AUTH_URL if is_sandbox_app() else PROD_AUTH_URL


def token_urls() -> tuple[str, ...]:
    if is_sandbox_app():
        return (SANDBOX_TOKEN_URL,)
    return (PROD_TOKEN_URL, PROD_TOKEN_URL_FALLBACK)


def account_uses_live_host() -> bool:
    """True only for a live-money account that is explicitly confirmed.

    QT execute / TRADING_MODE=live is not enough. Demo hedging + executing
    loop stays on demo.ctraderapi.com.
    """
    if ctrader_env() != "live":
        return False
    paper = os.getenv("CTRADER_PAPER_MODE", os.getenv("CTRADE_PAPER_MODE", "true"))
    if (paper or "true").strip().lower() == "true":
        return False
    confirm = os.getenv("CTRADER_LIVE_CONFIRM", os.getenv("CTRADE_LIVE_CONFIRM", "")).strip()
    return confirm == "I_UNDERSTAND"


def protobuf_host(is_live: bool | None = None) -> str:
    """TCP/protobuf host — sandbox and demo both use demo.ctraderapi.com."""
    if is_live is None:
        is_live = account_uses_live_host()
    if is_live:
        return LIVE_PROTOBUF_HOST
    return DEMO_PROTOBUF_HOST
