"""Environment flags for the Jev advisory path. Defaults are fail-closed."""

from __future__ import annotations

import os

DEFAULT_MODEL = "jev-1.13.0"
DEFAULT_BASE_URL = "https://api.typesafe.ai"
# OpenRouter model page: https://openrouter.ai/typesafe/jev-1.13
# Not jev-1.13.0 — that patch id is the direct TypeSafe pin and is not the OpenRouter slug.
DEFAULT_OPENROUTER_MODEL = "typesafe/jev-1.13"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 12.0
DEFAULT_CACHE_SECONDS = 600
DEFAULT_MIN_PROB_MARGIN = 0.12
DEFAULT_TWEET_TARGET = 100
MIN_BARS = 16
QUESTION_SCHEMA_VERSION = "crypto-advisory-1"
STATE_BUILDER_VERSION = "perp-state-2"
JEV_INPUT_USD_PER_MILLION = 0.042
DEFAULT_CALIBRATION_MIN_LABELS = 200
DEFAULT_PAUSE_MS = 30_000


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def jev_analysis_enabled() -> bool:
    """When false, the opinion layer never calls Jev. Existing agents are unchanged."""
    return env_flag("JEV_ANALYSIS_ENABLED", False)


def jev_replace_personas() -> bool:
    """Skip LLM persona agents only after a validated Jev evaluation."""
    return env_flag("JEV_REPLACE_PERSONAS", False)


def jev_include_social() -> bool:
    """Twitter pulls are off on the trading path unless explicitly enabled."""
    return env_flag("JEV_INCLUDE_SOCIAL", False)


def jev_embed_tweets() -> bool:
    return env_flag("JEV_EMBED_TWEETS", False)


def typesafe_api_key() -> str:
    return os.getenv("TYPESAFE_API_KEY", "").strip()


def twitter_api_key() -> str:
    return (os.getenv("TWITTERAPI_KEY") or os.getenv("TWITTER_API_KEY") or "").strip()


def jev_model() -> str:
    return os.getenv("JEV_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def jev_base_url() -> str:
    return os.getenv("JEV_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/") or DEFAULT_BASE_URL


def jev_timeout_seconds() -> float:
    return max(1.0, env_float("JEV_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))


def jev_cache_seconds() -> int:
    return max(0, env_int("JEV_CACHE_SECONDS", DEFAULT_CACHE_SECONDS))


def jev_min_prob_margin() -> float:
    return min(1.0, max(0.0, env_float("JEV_MIN_PROB_MARGIN", DEFAULT_MIN_PROB_MARGIN)))


def jev_provider() -> str:
    """typesafe, openrouter, litellm, or mock. Mock never influences the book."""
    return os.getenv("JEV_PROVIDER", "typesafe").strip().lower() or "typesafe"


def openrouter_api_key() -> str:
    return os.getenv("OPENROUTER_API_KEY", "").strip()


def openrouter_base_url() -> str:
    return os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL).strip().rstrip("/") or DEFAULT_OPENROUTER_BASE_URL


def jev_openrouter_model() -> str:
    """Pinned OpenRouter slug. JEV_MODEL stays the direct TypeSafe id."""
    explicit = os.getenv("JEV_OPENROUTER_MODEL", "").strip()
    return explicit or DEFAULT_OPENROUTER_MODEL


def openrouter_headers(api_key: str) -> dict[str, str]:
    """Same header style as the embedding client in backend.utils.embeddings."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    title = os.getenv("OPENROUTER_APP_TITLE", "ai-trading-platform").strip()
    if title:
        headers["X-Title"] = title
    return headers


def jev_fallback_provider() -> str:
    return os.getenv("JEV_FALLBACK_PROVIDER", "").strip().lower()


def jev_influence_book() -> bool:
    """Uncalibrated Jev probabilities stay display-only unless this is explicitly on."""
    return env_flag("JEV_INFLUENCE_BOOK", False)


def calibration_min_labels() -> int:
    return max(1, env_int("JEV_CALIBRATION_MIN_LABELS", DEFAULT_CALIBRATION_MIN_LABELS))


def jev_pause_seconds() -> float:
    return max(0.0, env_float("JEV_PAUSE_MS", DEFAULT_PAUSE_MS) / 1000.0)


def estimate_cost_usd(input_tokens: int | None) -> float | None:
    if input_tokens is None:
        return None
    return round(int(input_tokens) * JEV_INPUT_USD_PER_MILLION / 1_000_000, 8)
