import hmac
import os
from typing import Iterable

from fastapi import HTTPException, Request, status


AUTH_ENV_VARS = ("ADMIN_API_KEY", "API_AUTH_TOKEN", "BACKEND_API_KEY")
PUBLIC_PATHS = {
    "/",
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}
SENSITIVE_PREFIXES = (
    "/api-keys",
    "/api/api-keys",
    "/storage",
    "/api/storage",
    # Signal candidates expose live trade intent (entries, stops, sizing).
    "/signals",
    "/api/signals",
    # Research plane endpoints expose proprietary market research & decisions.
    "/research",
    "/api/research",
)
SENTRY_PREFIXES = ("/sentry",)
TRADING_PREFIXES = ("/trading", "/api/trading")
JESSE_PREFIXES = ("/jesse", "/api/jesse")
JEV_PREFIXES = ("/jev", "/api/jev")
PUBLIC_TRADING_PATHS = {
    "/trading/strategies",
    "/api/trading/strategies",
}


def _tokens_from_env(env_names: Iterable[str] = AUTH_ENV_VARS) -> list[str]:
    return [token for name in env_names if (token := os.getenv(name, "").strip())]


def admin_auth_enabled() -> bool:
    """Auth middleware is active only when an admin token is configured."""
    return bool(_tokens_from_env())


def is_sensitive_request(request: Request) -> bool:
    """Endpoints that mutate state, expose secrets, or control trading processes."""
    path = request.url.path.rstrip("/") or "/"
    method = request.method.upper()

    if method == "OPTIONS" or path in PUBLIC_PATHS:
        return False

    if path.startswith(SENSITIVE_PREFIXES):
        return True

    if path.startswith(SENTRY_PREFIXES):
        # Sentry POST routes use their dedicated X-Sentry-Token validator.
        return method not in {"GET", "HEAD", "POST", "OPTIONS"}

    if path.startswith(TRADING_PREFIXES):
        if path in PUBLIC_TRADING_PATHS:
            return False
        # Protect operational data (positions, balances, status, history) as well as loop controls.
        return True

    if path.startswith(JESSE_PREFIXES):
        # Protect quant parameters, FINMEM memory ingestion/evaluation, and inference.
        return True

    if path.startswith(JEV_PREFIXES):
        # Jev evaluations are advisory research, same auth surface as Jesse.
        return True

    return method not in {"GET", "HEAD"}


def validate_admin_request(request: Request) -> None:
    configured_tokens = _tokens_from_env()
    if not configured_tokens:
        return

    supplied_token = request.headers.get("x-api-key", "").strip()
    auth_header = request.headers.get("authorization", "").strip()
    if auth_header.lower().startswith("bearer "):
        supplied_token = auth_header[7:].strip()

    if not supplied_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing admin API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not any(hmac.compare_digest(supplied_token, token) for token in configured_tokens):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid admin API token.")
