"""Server-side Supabase credentials for privileged writes.

Postmortems, bear verdicts, and shadow replays must never use the anon key.
The PostgREST client is constructed from a service-role secret only; if that
secret is missing or is an anon JWT, callers skip the network write.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_SERVICE_KEY_ENVS = ("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_SERVICE_KEY")
_GENERIC_KEY_ENV = "SUPABASE_KEY"
_ANON_KEY_ENVS = ("SUPABASE_ANON_KEY",)


def _jwt_role(token: str) -> str:
    """Return the unverified JWT ``role`` claim, or ``""`` if not a JWT."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return ""
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
        if not isinstance(data, dict):
            return ""
        return str(data.get("role") or "").strip().lower()
    except Exception:
        return ""


def _looks_like_anon_key(key: str) -> bool:
    role = _jwt_role(key)
    return role == "anon"


def service_role_key() -> str:
    """Return a service-role key or ``""``. Never returns the anon key."""
    for env_name in _SERVICE_KEY_ENVS:
        raw = (os.getenv(env_name) or "").strip()
        if not raw:
            continue
        if _looks_like_anon_key(raw):
            logger.warning(
                "Ignoring %s for privileged Supabase writes: JWT role is anon",
                env_name,
            )
            continue
        return raw

    generic = (os.getenv(_GENERIC_KEY_ENV) or "").strip()
    if generic and _jwt_role(generic) == "service_role":
        return generic

    for env_name in _ANON_KEY_ENVS:
        if (os.getenv(env_name) or "").strip():
            logger.warning(
                "Refusing %s for privileged Supabase writes; set SUPABASE_SERVICE_ROLE_KEY",
                env_name,
            )
    return ""


def service_role_credentials() -> tuple[str, str]:
    """Return ``(url, service_role_key)`` or ``("", "")`` if not configured."""
    url = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    key = service_role_key()
    if not (url and key):
        return "", ""
    return url, key


def is_service_role_configured() -> bool:
    url, key = service_role_credentials()
    return bool(url and key)


def service_role_headers(prefer: Optional[str] = "return=minimal") -> dict[str, str]:
    """Authorization headers for a PostgREST write. Empty dict if unconfigured."""
    _url, key = service_role_credentials()
    if not key:
        return {}
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers
