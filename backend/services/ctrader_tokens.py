"""
cTrader OAuth Token Persistence and Auto-Refresh Manager.

Features:
- Encrypted / structured persistent store for rotating refresh tokens.
- Proactive token renewal before the 30-day expiration window (< 5 days remaining).
- Resilient fallback from persistent volume/file to environment variables.
- Cooldown, single-flight lock, and no fallback POST after an ambiguous timeout
  so a single-use Spotware refresh token is not reused.
"""
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from backend.services.ctrader_oauth import token_urls as _ctrader_token_urls

logger = logging.getLogger(__name__)

TOKENS_FILE = Path("data/ctrader_tokens.json")

# Spotware refresh tokens are single-use. After a timeout it is unknown whether
# the primary endpoint already rotated the token, so we must not POST it again.
REFRESH_COOLDOWN_SEC = 600.0
REFRESH_HTTP_TIMEOUT_SEC = 30.0
DEFAULT_EXPIRES_IN_SEC = 2592000  # ~30 days
PROACTIVE_REFRESH_REMAINING_SEC = 432000  # 5 days


def _ensure_dir():
    TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)


class CTraderTokenStore:
    """Manages secure token lifecycle and rotation persistence."""

    def __init__(self, file_path: Path = TOKENS_FILE):
        self.file_path = file_path
        self._memory_cache: Optional[Dict[str, Any]] = None
        self._last_refresh_attempt_ts: float = 0.0
        self._refresh_lock = threading.Lock()

    def get_tokens(self) -> Dict[str, Any]:
        """Load the latest valid token set."""
        if self._memory_cache:
            return dict(self._memory_cache)

        # 1. Try file storage
        if self.file_path.exists():
            try:
                data = json.loads(self.file_path.read_text(encoding="utf-8"))
                if data.get("access_token") or data.get("accessToken"):
                    self._memory_cache = data
                    return dict(data)
            except Exception as e:
                logger.warning(f"Failed to read cTrader token store file: {e}")

        # 2. Fall back to environment bootstrap
        access_token = os.getenv("CTRADER_ACCESS_TOKEN", "").strip()
        refresh_token = os.getenv("CTRADER_REFRESH_TOKEN", "").strip()
        client_id = os.getenv("CTRADER_CLIENT_ID", "").strip()
        client_secret = os.getenv("CTRADER_CLIENT_SECRET", "").strip()
        account_id = os.getenv("CTRADER_ACCOUNT_ID", "0").strip()

        tokens = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
            "account_id": int(account_id) if account_id.isdigit() else 0,
            "updated_at": int(time.time()),
            "expires_in": DEFAULT_EXPIRES_IN_SEC,  # default ~30 days
        }
        self._memory_cache = tokens
        return dict(tokens)

    def save_tokens(self, tokens: Dict[str, Any]) -> None:
        """Persist tokens atomically to disk and memory."""
        self._memory_cache = dict(tokens)
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.file_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(tokens, indent=2), encoding="utf-8")
            tmp_path.replace(self.file_path)
            logger.info("cTrader tokens saved to persistent token store")
        except Exception as e:
            logger.error(f"Failed to save cTrader tokens to disk: {e}")

    def _needs_refresh(self, tokens: Dict[str, Any], force: bool, now: int) -> bool:
        updated_at = tokens.get("updated_at", now)
        expires_in = tokens.get("expires_in", DEFAULT_EXPIRES_IN_SEC)
        time_elapsed = now - updated_at
        return bool(force) or (time_elapsed > (expires_in - PROACTIVE_REFRESH_REMAINING_SEC))

    def refresh_if_needed(self, force: bool = False) -> Optional[str]:
        """
        Check if token needs refresh (less than 5 days remaining or forced).
        Returns active access_token or None.

        Refresh POSTs are serialized on ``_refresh_lock`` (single-flight).
        The 600s cooldown starts only when a request may have reached Spotware
        (an HTTP response, or a read/write timeout). Confirmed pre-send
        failures such as ``ConnectError`` do not start the cooldown, so a
        brief outage does not block re-auth for ten minutes.
        """
        tokens = self.get_tokens()
        client_id = tokens.get("client_id") or os.getenv("CTRADER_CLIENT_ID", "")
        client_secret = tokens.get("client_secret") or os.getenv("CTRADER_CLIENT_SECRET", "")
        refresh_token = tokens.get("refresh_token") or os.getenv("CTRADER_REFRESH_TOKEN", "")

        if not refresh_token or not client_id or not client_secret:
            return tokens.get("access_token")

        now = int(time.time())
        if not self._needs_refresh(tokens, force, now):
            return tokens.get("access_token")

        with self._refresh_lock:
            tokens = self.get_tokens()
            client_id = tokens.get("client_id") or os.getenv("CTRADER_CLIENT_ID", "")
            client_secret = tokens.get("client_secret") or os.getenv("CTRADER_CLIENT_SECRET", "")
            refresh_token = tokens.get("refresh_token") or os.getenv("CTRADER_REFRESH_TOKEN", "")
            if not refresh_token or not client_id or not client_secret:
                return tokens.get("access_token")
            now = int(time.time())
            if not self._needs_refresh(tokens, force, now):
                return tokens.get("access_token")
            since_attempt = time.time() - self._last_refresh_attempt_ts
            if since_attempt < REFRESH_COOLDOWN_SEC:
                logger.warning(
                    "cTrader token refresh skipped (within %.0fs cooldown window)",
                    REFRESH_COOLDOWN_SEC,
                )
                return tokens.get("access_token")
            return self._refresh_via_oauth(tokens, client_id, client_secret, refresh_token, now)

    def _arm_refresh_cooldown(self) -> None:
        self._last_refresh_attempt_ts = time.time()

    @staticmethod
    def request_may_have_reached_spotware(
        *,
        got_response: bool = False,
        exc: BaseException | None = None,
    ) -> bool:
        """True when the single-use refresh grant may already have been consumed.

        An HTTP response reached Spotware. Read/write timeouts are ambiguous.
        Connect/pool timeouts and other connect errors never delivered the grant.
        """
        if got_response:
            return True
        # httpx 0.28: ConnectTimeout is a TimeoutException, not a ConnectError.
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
            return False
        return isinstance(exc, httpx.TimeoutException)

    def _refresh_via_oauth(
        self,
        tokens: Dict[str, Any],
        client_id: str,
        client_secret: str,
        refresh_token: str,
        now: int,
    ) -> Optional[str]:
        logger.info("cTrader token rotation: renewing access token via Spotware OAuth...")
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        }
        for token_url in _ctrader_token_urls():
            try:
                resp = httpx.post(token_url, data=payload, timeout=REFRESH_HTTP_TIMEOUT_SEC)
                self._arm_refresh_cooldown()
                data = resp.json()
                if resp.status_code == 200 and not data.get("errorCode"):
                    new_access = data.get("accessToken") or data.get("access_token")
                    new_refresh = data.get("refreshToken") or data.get("refresh_token") or refresh_token
                    expires = data.get("expiresIn") or data.get("expires_in") or DEFAULT_EXPIRES_IN_SEC
                    if not new_access:
                        continue
                    tokens["access_token"] = new_access
                    tokens["refresh_token"] = new_refresh
                    tokens["updated_at"] = now
                    tokens["expires_in"] = expires
                    self.save_tokens(tokens)
                    logger.info("cTrader token successfully renewed via %s", token_url)
                    return new_access
                logger.error(
                    "cTrader token refresh rejected via %s (%s): %s",
                    token_url,
                    resp.status_code,
                    data.get("description") or resp.text[:200],
                )
            except httpx.TimeoutException as e:
                # Do not POST the single-use refresh token to a fallback URL.
                # Arm cooldown only for read/write timeouts (request may have
                # reached Spotware). ConnectTimeout never delivered the grant.
                if self.request_may_have_reached_spotware(exc=e):
                    self._arm_refresh_cooldown()
                logger.error(
                    "cTrader token refresh timed out on %s (aborting fallback to avoid single-use token reuse): %s",
                    token_url,
                    e,
                )
                break
            except Exception as e:
                logger.error(f"cTrader token refresh network error ({token_url}): {e}")

        return tokens.get("access_token")


# Global token store instance
token_store = CTraderTokenStore()
ctrader_token_store = token_store
