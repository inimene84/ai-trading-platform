"""
LLM Model Router — Task-Based Model Selection
===============================================
Central registry that maps task types to model configurations.
Eliminates scattered model name hardcoding across the codebase.

Usage:
    from backend.llm.router import pick_model, call_llm_resilient
    cfg = pick_model("persona_analysis")
    # cfg.name, cfg.provider, cfg.base_url, cfg.max_tokens, cfg.temperature
"""

from __future__ import annotations

import logging
import os
import time
import httpx
import asyncio
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Literal, Never, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""
    name: str
    provider: str
    tier: Literal["cheap", "balanced", "premium"] = "balanced"
    base_url: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.4
    api_key_env: str = ""  # env var name for the API key


# ── Default model registry ───────────────────────────────────────────────────
# These can be overridden via environment variables per task type.
#
# PRIMARY PROVIDER: OmniRoute (https://omni.allikas.online)
#   auto/fast is the QT default — Allikas measured HTTP 200 in ~7–8s
#   (kilocode/openrouter/free). auto/smart is slower and can miss a tight
#   client deadline from the QT VPS.
#   auto/cheap, auto/reasoning, auto/coding, auto/best-free also valid.
#
# Fallback chain: OmniRoute → KieAI → OpenRouter → xAI → OpenAI → Anthropic → Gemini

# OmniRoute config (OpenAI-compatible /v1/chat/completions — not Kie /codex)
_OMNIROUTE_CANONICAL_BASE = "https://omni.allikas.online/v1"
_OMNIROUTE_BASE_URL = os.getenv("OMNIROUTE_BASE_URL", _OMNIROUTE_CANONICAL_BASE)
_OMNIROUTE_DEFAULT_PRESET = "auto/fast"

# Kie.ai direct model IDs (fallback). Luna is the faster GPT 5.6 sibling;
# Terra 500'd in production on the same /codex/v1/responses path.
# Extra hops: KIE_FALLBACK_MODELS (Gemini OpenAI-compat, then Claude Haiku).
_KIE_MODEL = os.getenv("KIE_MODEL", "gpt-5-6-luna")
_KIE_OPUS_DIRECT_MODEL = os.getenv("KIE_OPUS_MODEL", "")
_KIE_BASE_URL = os.getenv("KIE_BASE_URL", "https://api.kie.ai")
_DEFAULT_KIE_FALLBACK_MODELS = "gemini-3-8-flash-openai,claude-haiku-4-5"
# Documented OpenAI-compat Gemini slugs (docs.kie.ai/market/gemini/*).
_KIE_GEMINI_SLUGS: dict[str, str] = {
    "gemini-3-8-flash": "gemini-3-8-flash-openai",
    "gemini-3-8-flash-openai": "gemini-3-8-flash-openai",
    "gemini-3-pro": "gemini-3-pro",
    "gemini-3-pro-openai": "gemini-3-pro",
    "gemini-3-flash": "gemini-3-flash",
    "gemini-3-flash-openai": "gemini-3-flash-openai",
}
_KIE_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9.-]{0,80}")
_LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", os.getenv("PERSONA_LLM_BASE_URL", "http://litellm:4000/v1"))

_DEFAULT_REGISTRY: dict[str, ModelConfig] = {
    # PRIMARY: OmniRoute auto/fast — Allikas-verified low-latency preset
    # Task-specific presets give the router hints for optimal model selection.
    "persona_analysis": ModelConfig(
        name=os.getenv("PERSONA_LLM_MODEL", "auto/fast"),
        provider="omniroute",
        tier="balanced",
        base_url=_OMNIROUTE_BASE_URL,
        max_tokens=1024,
        temperature=0.3,
        api_key_env="OMNIROUTE_API_KEY",
    ),

    # Deep trading analysis — OmniRoute auto/fast (measured healthy path)
    "deep_analysis": ModelConfig(
        name=os.getenv("DEEP_ANALYSIS_LLM_MODEL", "auto/fast"),
        provider="omniroute",
        tier="balanced",
        base_url=_OMNIROUTE_BASE_URL,
        max_tokens=1500,
        temperature=0.3,
        api_key_env="OMNIROUTE_API_KEY",
    ),

    # Premium/complex reasoning — still auto/fast unless env pins another preset
    "premium_analysis": ModelConfig(
        name=os.getenv("PREMIUM_ANALYSIS_LLM_MODEL", "auto/fast"),
        provider="omniroute",
        tier="premium",
        base_url=_OMNIROUTE_BASE_URL,
        max_tokens=2048,
        temperature=0.3,
        api_key_env="OMNIROUTE_API_KEY",
    ),

    # General LLM tasks (news scoring, alerts, etc.) — fast + cheap preset
    "general": ModelConfig(
        name=os.getenv("GENERAL_LLM_MODEL", "auto/fast"),
        provider="omniroute",
        tier="balanced",
        base_url=_OMNIROUTE_BASE_URL,
        max_tokens=1024,
        temperature=0.4,
        api_key_env="OMNIROUTE_API_KEY",
    ),

    # Dashboard assistant (Gemini UI → OmniRoute)
    "assistant_chat": ModelConfig(
        name=os.getenv("ASSISTANT_LLM_MODEL", "auto/fast"),
        provider="omniroute",
        tier="balanced",
        base_url=_OMNIROUTE_BASE_URL,
        max_tokens=1200,
        temperature=0.35,
        api_key_env="OMNIROUTE_API_KEY",
    ),

    # GrokBOT overseer summaries (prefers xAI when configured; chain falls back)
    "grok_overseer": ModelConfig(
        name=os.getenv("XAI_MODEL", "grok-beta"),
        provider="xai",
        tier="balanced",
        base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
        max_tokens=900,
        temperature=0.2,
        api_key_env="XAI_API_KEY",
    ),

    # ── Fallback chain entries (used if OmniRoute unavailable) ────────────────
    # KieAI fallback (direct Kie.ai; family endpoint is chosen per model id)
    "fallback_kie": ModelConfig(
        name=_KIE_MODEL,
        provider="kie",
        tier="balanced",
        base_url=_KIE_BASE_URL,
        max_tokens=1024,
        temperature=0.3,
        api_key_env="KIE_API_KEY",
    ),

    # OpenRouter fallback
    "fallback_1": ModelConfig(
        name=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-5"),
        provider="openrouter",
        tier="balanced",
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key_env="OPENROUTER_API_KEY",
    ),
    "fallback_2": ModelConfig(
        name=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        provider="anthropic",
        tier="balanced",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "fallback_3": ModelConfig(
        name=os.getenv("GEMINI_MODEL", "google/gemini-2.5-flash"),
        provider="openrouter-gemini",
        tier="balanced",
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key_env="OPENROUTER_API_KEY",
    ),
}


def pick_model(task_type: str) -> ModelConfig:
    """
    Select the appropriate model configuration for a given task type.

    Args:
        task_type: One of the keys in the registry
                   ('persona_analysis', 'deep_analysis', 'general', etc.)

    Returns:
        ModelConfig for the requested task, or the 'general' fallback.
    """
    config = _DEFAULT_REGISTRY.get(task_type)
    if config is None:
        logger.warning(f"Unknown task type '{task_type}', falling back to 'general'")
        config = _DEFAULT_REGISTRY["general"]
    return config


def get_api_key(config: ModelConfig) -> str:
    """Resolve the API key for a model config from environment variables."""
    if config.api_key_env:
        # Never substitute an unrelated provider's token. The old cascade sent
        # LiteLLM/Kie keys to Anthropic and Gemini, producing repeated 401/400s.
        return os.getenv(config.api_key_env, "")
    # Legacy configs without an explicit key name may use the local proxy key.
    # LITELLM_API_KEY is the LiteLLM master key (also used for KieAI proxy)
    return (
        os.getenv("LITELLM_API_KEY", "")
        or os.getenv("KIE_API_KEY", "")   # KieAI proxy key also accepted by LiteLLM
        or os.getenv("PERSONA_LLM_API_KEY", "")
        or os.getenv("GROQ_API_KEY", "")
    )


def list_models() -> dict[str, dict]:
    """Return a summary of all registered models (useful for debug/API)."""
    return {
        task: {
            "model": cfg.name,
            "provider": cfg.provider,
            "tier": cfg.tier,
        }
        for task, cfg in _DEFAULT_REGISTRY.items()
    }


class LLMChainExhausted(RuntimeError):
    """Raised when every configured LLM provider failed within the chain budget."""


# Last-error summary for /trading/status — no response bodies, no keys.
_llm_router_status: dict[str, Any] = {
    "degraded": False,
    "chain_exhausted": False,
    "last_success_at": None,
    "last_success_provider": None,
    "last_success_model": None,
    "last_error": None,
    "last_error_at": None,
    "last_error_provider": None,
    "last_error_task": None,
}


def get_llm_router_status() -> dict[str, Any]:
    """Ops snapshot of the last LLM success/failure (safe to expose)."""
    return dict(_llm_router_status)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_llm_success(task_type: str, provider: str, model: str) -> None:
    _llm_router_status["degraded"] = False
    _llm_router_status["chain_exhausted"] = False
    _llm_router_status["last_success_at"] = _utc_now_iso()
    _llm_router_status["last_success_provider"] = provider
    _llm_router_status["last_success_model"] = model
    _llm_router_status["last_error"] = None
    _llm_router_status["last_error_at"] = None
    _llm_router_status["last_error_provider"] = None
    _llm_router_status["last_error_task"] = task_type


def _record_llm_failure(task_type: str, provider: str, summary: str, *, exhausted: bool = False) -> None:
    _llm_router_status["last_error"] = summary
    _llm_router_status["last_error_at"] = _utc_now_iso()
    _llm_router_status["last_error_provider"] = provider
    _llm_router_status["last_error_task"] = task_type
    if exhausted:
        _llm_router_status["degraded"] = True
        _llm_router_status["chain_exhausted"] = True


# ── Timeouts, error classification, catalog sanitization ─────────────────────

# OmniRoute auto-router presets. Dead kie/* / LiteLLM catalog ids must never be
# sent to OmniRoute — they hang or 400 while the trading cycle waits.
_OMNIROUTE_PRESETS = {
    "auto/smart",
    "auto/fast",
    "auto/cheap",
    "auto/reasoning",
    "auto/coding",
    "auto/best-free",
    "auto/chat",
    "auto/best-coding",
}
_KIE_NATIVE_MODELS = {
    "gpt-5-6-terra",
    "gpt-5-6-luna",
    "gpt-5-6-sol",
    "gpt-5-5",
    "gpt-5-2",
    "gemini-3-8-flash",
    "gemini-3-8-flash-openai",
    "claude-haiku-4-5",
}
_DEAD_CATALOG_PREFIXES = ("kie/", "litellm/")

# Per-provider read timeouts.
# OmniRoute auto/fast is healthy (~7–8s on Allikas; plan 7–15s+ from QT VPS).
# 8s was too tight and clipped live successes. 25s is one attempt with margin,
# then fail over — not 90s × 3 retries. Override with
# LLM_<PROVIDER>_TIMEOUT_SECONDS (e.g. LLM_OMNIROUTE_TIMEOUT_SECONDS).
# LLM_PROVIDER_TIMEOUT_SECONDS is a fallback for unknown providers only; it
# cannot shrink OmniRoute below _OMNIROUTE_TIMEOUT_FLOOR.
_OMNIROUTE_TIMEOUT_FLOOR = 20.0
_PROVIDER_TIMEOUT_DEFAULTS = {
    "omniroute": 25.0,
    "kie": 10.0,
    "openrouter": 12.0,
    "openrouter-gemini": 12.0,
    "xai": 12.0,
    "openai": 12.0,
    "anthropic": 12.0,
    "google": 12.0,
    "gemini": 12.0,
    "groq": 10.0,
    "litellm": 12.0,
    "ollama": 20.0,
}

_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_TIMEOUT_ERRORS = (httpx.TimeoutException, TimeoutError, asyncio.TimeoutError)
_CONNECT_ERRORS = (httpx.ConnectError, httpx.NetworkError)


def _env_float(name: str, default: float, *, allow_zero: bool = False) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if allow_zero:
        return value if value >= 0 else default
    return value if value > 0 else default


def _provider_timeout(provider: str) -> float:
    """Bounded HTTP timeout for one provider attempt."""
    prov = provider.lower()
    default = _PROVIDER_TIMEOUT_DEFAULTS.get(prov, 12.0)
    key = f"LLM_{prov.upper().replace('-', '_')}_TIMEOUT_SECONDS"
    per = os.getenv(key)
    if per:
        value = _env_float(key, default)
        if prov == "omniroute":
            return max(value, _OMNIROUTE_TIMEOUT_FLOOR) if value > 0 else default
        return value
    # Global override does not apply to OmniRoute — a copied .env with
    # LLM_PROVIDER_TIMEOUT_SECONDS=8 would clip the measured 7–15s path.
    if prov != "omniroute":
        global_override = os.getenv("LLM_PROVIDER_TIMEOUT_SECONDS")
        if global_override:
            return _env_float("LLM_PROVIDER_TIMEOUT_SECONDS", default)
    return default


def _chain_budget_seconds(task_type: str) -> float:
    """Hard cap for the whole fallback chain so the trading loop is not stalled."""
    # Must exceed OmniRoute's 25s attempt, then Luna + one extra Kie family hop.
    if task_type == "persona_analysis":
        return _env_float("LLM_PERSONA_CHAIN_BUDGET_SECONDS", 48.0)
    return _env_float("LLM_CHAIN_BUDGET_SECONDS", 55.0)


def _retry_backoff_seconds() -> float:
    return _env_float("LLM_RETRY_BACKOFF_SECONDS", 0.35, allow_zero=True)


def _httpx_timeout(seconds: float, provider: str = "") -> httpx.Timeout:
    read = max(1.0, float(seconds))
    if provider.lower() == "omniroute":
        # QT VPS → omni.allikas.online may need more than 3s for TLS.
        connect = min(8.0, read)
        write = min(8.0, read)
    else:
        connect = min(3.0, read)
        write = min(5.0, read)
    return httpx.Timeout(connect=connect, read=read, write=write, pool=3.0)


def _http_status_code(exc: BaseException) -> Optional[int]:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        return exc.response.status_code
    match = re.search(r"HTTP\s+(\d{3})", str(exc))
    if match:
        return int(match.group(1))
    return None


def _error_summary(exc: BaseException) -> str:
    """Short, body-free error label for logs and raised chain failures."""
    status = _http_status_code(exc)
    if status is not None:
        return f"{type(exc).__name__} HTTP {status}"
    return type(exc).__name__


def _raise_http_status(resp: httpx.Response, provider: str) -> None:
    """Raise HTTPStatusError without embedding provider response bodies (keys, etc.)."""
    preview = (resp.text or "")[:200]
    if preview:
        logger.debug("LLM %s HTTP %s body: %s", provider, resp.status_code, preview)
    raise httpx.HTTPStatusError(
        f"HTTP {resp.status_code} from {provider}",
        request=resp.request,
        response=resp,
    )


def classify_llm_error(exc: BaseException) -> Literal["retry", "failover"]:
    """
    Decide whether to retry the current provider once, or fail over immediately.

    Timeouts and connection failures fail over immediately (do not burn the cycle
    on a dead primary). HTTP 5xx / 429 get one short retry, then the next provider.
    Other 4xx skip remaining retries for this provider.
    """
    if isinstance(exc, _TIMEOUT_ERRORS) or isinstance(exc, _CONNECT_ERRORS):
        return "failover"
    status = _http_status_code(exc)
    if status is None:
        return "retry"
    if status in _RETRYABLE_STATUS:
        return "retry"
    return "failover"


def _omniroute_preset_for_task(task_type: str) -> str:
    configured = (os.getenv("OMNIROUTE_DEFAULT_MODEL") or "").strip()
    if configured in _OMNIROUTE_PRESETS:
        return configured
    return _OMNIROUTE_DEFAULT_PRESET


def _sanitize_omniroute_base_url(url: Optional[str]) -> str:
    """Keep OmniRoute on the OpenAI-compatible Allikas /v1 host, not Kie/codex."""
    raw = (url or "").strip().rstrip("/")
    lower = raw.lower()
    looks_wrong = (
        not raw
        or "api.kie.ai" in lower
        or "openrouter.ai" in lower
        or "/codex/" in lower
        or ":4000" in raw
    )
    if looks_wrong:
        if raw and raw != _OMNIROUTE_CANONICAL_BASE:
            logger.warning(
                "LLM Router: remapping OmniRoute base_url %r → %s",
                url,
                _OMNIROUTE_CANONICAL_BASE,
            )
        return _OMNIROUTE_CANONICAL_BASE
    if "omni.allikas.online" in lower and not lower.endswith("/v1"):
        return raw + "/v1"
    return raw


def _looks_like_kie_native_id(name: str) -> bool:
    """True when a model id belongs on Kie, not OmniRoute auto/*."""
    lower = (name or "").strip().lower()
    if not lower or lower in _OMNIROUTE_PRESETS:
        return False
    if lower in _KIE_NATIVE_MODELS:
        return True
    if lower.startswith(("claude", "grok", "gemini", "gpt-5-6", "gpt-5-5", "gpt-5-4", "gpt-6")):
        return True
    if lower.startswith("gpt-") and "codex" in lower:
        return True
    if lower in {"gpt-5-2", "gpt-5.2"} or lower.startswith("gpt-5-2"):
        return True
    return False


def sanitize_provider_config(cfg: ModelConfig, task_type: str = "general") -> Optional[ModelConfig]:
    """
    Drop or remap unusable catalog ids.

    OmniRoute must receive auto/* (or a real OmniRoute model), never kie/* LiteLLM
    aliases or Kie-native ids like gpt-5-6-terra. Kie fallbacks keep native ids
    and strip a leading kie/ prefix; unknown catalog paths are skipped.
    """
    name = (cfg.name or "").strip()
    lower = name.lower()
    provider = cfg.provider.lower()

    if provider == "omniroute":
        looks_dead = (
            not name
            or lower.startswith(_DEAD_CATALOG_PREFIXES)
            or _looks_like_kie_native_id(lower)
        )
        preset_name = name
        if looks_dead:
            preset_name = _omniroute_preset_for_task(task_type)
            if name != preset_name:
                logger.warning(
                    "LLM Router: remapping model %r → OmniRoute %s (dead kie/catalog id)",
                    name,
                    preset_name,
                )
        base_url = _sanitize_omniroute_base_url(cfg.base_url or _OMNIROUTE_BASE_URL)
        if preset_name != cfg.name or base_url != cfg.base_url:
            return replace(cfg, name=preset_name, base_url=base_url)
        return cfg

    if provider == "kie":
        cleaned = name[4:] if lower.startswith("kie/") else name
        cleaned = cleaned.strip()
        if not cleaned or "/" in cleaned:
            logger.warning("LLM Router: skipping Kie fallback with unusable model %r", name)
            return None
        if not _KIE_SLUG_RE.fullmatch(cleaned.lower()):
            logger.warning("LLM Router: skipping Kie fallback with invalid model id %r", name)
            return None
        if cleaned != name:
            logger.info("LLM Router: stripping kie/ prefix → %s", cleaned)
            return replace(cfg, name=cleaned)
        return cfg

    return cfg


# ── Kie.ai family routing (docs.kie.ai Market / Chat Models) ──────────────────
# Different model families use different host paths and payloads:
#   Claude        POST /claude/v1/messages              Anthropic messages, stream:false
#   GPT 5.6/5.5   POST /codex/v1/responses              Responses API (input_text)
#   GPT Codex     POST /api/v1/responses                Responses API
#   Grok          POST /grok/v1/responses               Responses API
#   Gemini/GPT5.2 POST /{slug}/v1/chat/completions      OpenAI chat, stream:false

KieKind = Literal["claude", "codex_responses", "gpt_codex", "grok_responses", "openai_chat"]


@dataclass(frozen=True)
class KieRoute:
    """Resolved Kie.ai endpoint for a model id."""

    kind: KieKind
    path: str
    model: str


def _kie_gemini_slug(model: str) -> str:
    mid = model.strip().lower()
    if mid in _KIE_GEMINI_SLUGS:
        return _KIE_GEMINI_SLUGS[mid]
    if mid.endswith("-openai"):
        return mid
    openai_slug = f"{mid}-openai"
    if openai_slug in _KIE_GEMINI_SLUGS.values() or "flash" in mid:
        return openai_slug
    return mid


def resolve_kie_route(model: str) -> KieRoute:
    """Map a Kie model id onto the documented family endpoint.

    Public so unit tests can pin the routing table without HTTP.
    ``model`` is the canonical id sent in the JSON body (matches the URL slug
    for OpenAI-compat families).
    """
    mid = (model or "").strip().lower()
    if not mid:
        return KieRoute("codex_responses", "/codex/v1/responses", "gpt-5-6-luna")
    if mid.startswith("claude"):
        return KieRoute("claude", "/claude/v1/messages", mid)
    if mid.startswith("grok"):
        return KieRoute("grok_responses", "/grok/v1/responses", mid)
    if mid.startswith("gemini"):
        slug = _kie_gemini_slug(mid)
        return KieRoute("openai_chat", f"/{slug}/v1/chat/completions", slug)
    if "codex" in mid:
        return KieRoute("gpt_codex", "/api/v1/responses", mid)
    if (
        mid.startswith("gpt-5-6")
        or mid.startswith("gpt-5-5")
        or mid.startswith("gpt-5-4")
        or mid.startswith("gpt-6")
    ):
        return KieRoute("codex_responses", "/codex/v1/responses", mid)
    if mid in {"gpt-5-2", "gpt-5.2"} or mid.startswith("gpt-5-2"):
        return KieRoute("openai_chat", "/gpt-5-2/v1/chat/completions", "gpt-5-2")
    return KieRoute("codex_responses", "/codex/v1/responses", mid)


def _kie_fallback_model_ids() -> list[str]:
    """Extra Kie models tried after the primary ``KIE_MODEL`` (same API key)."""
    raw = os.getenv("KIE_FALLBACK_MODELS", _DEFAULT_KIE_FALLBACK_MODELS).strip()
    models: list[str] = []
    if raw:
        models.extend(part.strip() for part in raw.split(",") if part.strip())
    opus = os.getenv("KIE_OPUS_MODEL", _KIE_OPUS_DIRECT_MODEL).strip()
    if opus:
        models.append(opus)
    seen: set[str] = set()
    unique: list[str] = []
    for model in models:
        if model not in seen:
            unique.append(model)
            seen.add(model)
    return unique


def _kie_request_url(cfg: ModelConfig, path: str) -> str:
    raw = (cfg.base_url or _KIE_BASE_URL).strip().rstrip("/")
    if "api.kie.ai" not in raw.lower():
        raw = "https://api.kie.ai"
    if raw.endswith("/claude") and path.startswith("/claude/"):
        return raw + path[len("/claude"):]
    if raw.endswith("/codex") and path.startswith("/codex/"):
        return raw + path[len("/codex"):]
    return f"{raw}{path}"


def _kie_responses_payload(model: str, prompt: str, system: str) -> dict[str, Any]:
    input_msgs: list[dict[str, Any]] = []
    if system:
        input_msgs.append({
            "role": "system",
            "content": [{"type": "input_text", "text": system}],
        })
    input_msgs.append({
        "role": "user",
        "content": [{"type": "input_text", "text": prompt}],
    })
    return {
        "model": model,
        "stream": False,
        "input": input_msgs,
        "reasoning": {"effort": "low"},
    }


def _kie_chat_completions_payload(
    model: str,
    prompt: str,
    system: str,
    max_tokens: int,
    response_json: bool,
    temperature: Optional[float] = None,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if response_json:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _extract_kie_responses_text(data: dict[str, Any]) -> str:
    text = ""
    if isinstance(data.get("output"), list):
        for item in data["output"]:
            if not isinstance(item, dict):
                continue
            for part in item.get("content", []):
                if isinstance(part, dict) and part.get("type") in ("text", "output_text"):
                    text += part.get("text", "")
    elif isinstance(data.get("choices"), list) and data["choices"]:
        choice = data["choices"][0]
        if isinstance(choice, dict):
            msg = choice.get("message") or {}
            text = msg.get("content", "") if isinstance(msg, dict) else str(choice.get("text", ""))
    elif "data" in data and isinstance(data["data"], dict):
        text = data["data"].get("content", "") or data["data"].get("text", "")
    elif "text" in data and isinstance(data["text"], str):
        text = data["text"]
    elif "response" in data and isinstance(data["response"], str):
        text = data["response"]
    if not text:
        raise ValueError("Kie responses API returned no output text")
    return text


def _extract_kie_chat_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message") or {}
            if isinstance(msg, dict):
                content = msg.get("content") or msg.get("reasoning_content") or ""
                if content:
                    return content
    candidates = data.get("candidates")
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, dict):
            parts = (first.get("content") or {}).get("parts") or []
            texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
            if texts:
                return "".join(texts)
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    if data.get("output"):
        return _extract_kie_responses_text(data)
    raise ValueError("Kie chat completions returned no content")


def _extract_kie_claude_text(data: dict[str, Any]) -> str:
    text = ""
    for block in data.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            text += block.get("text", "")
    return text


def _assert_never_kie_kind(kind: Never) -> None:
    raise ValueError(f"Unhandled Kie route kind: {kind}")


def _kie_template_config() -> ModelConfig:
    """Prefer live ``KIE_MODEL``; otherwise the registry snapshot (tests may patch it)."""
    env_name = os.getenv("KIE_MODEL")
    if env_name is not None and env_name.strip():
        name = env_name.strip()
    else:
        name = _DEFAULT_REGISTRY["fallback_kie"].name or _KIE_MODEL
    return replace(_DEFAULT_REGISTRY["fallback_kie"], name=name)


def _expand_kie_model_hops(
    configs_to_try: list[tuple[str, ModelConfig]],
) -> list[tuple[str, ModelConfig]]:
    """Insert extra Kie family hops after the first Kie provider (same API key)."""
    extras = _kie_fallback_model_ids()
    template = _kie_template_config()
    if not extras:
        return configs_to_try
    if not _provider_configured(template) and not any(
        cfg.provider.lower() == "kie" for _, cfg in configs_to_try
    ):
        return configs_to_try

    out: list[tuple[str, ModelConfig]] = []
    seen: set[tuple[str, str]] = set()
    expanded = False

    def _append(label: str, cfg: ModelConfig) -> None:
        identity = (cfg.provider.lower(), cfg.name)
        if identity in seen:
            return
        seen.add(identity)
        out.append((label, cfg))

    def _append_extras(base: ModelConfig) -> None:
        for i, model in enumerate(extras, start=2):
            extra = replace(base, name=model)
            sanitized = sanitize_provider_config(extra, "general")
            if sanitized is None:
                continue
            _append(f"Fallback 1.{i} (KieAI {sanitized.name})", sanitized)

    for label, cfg in configs_to_try:
        _append(label, cfg)
        if cfg.provider.lower() == "kie" and not expanded:
            expanded = True
            _append_extras(cfg)

    if not expanded:
        insert_at = min(1, len(out))
        extra_rows: list[tuple[str, ModelConfig]] = []
        seen_before = set(seen)

        def _collect(label: str, cfg: ModelConfig) -> None:
            identity = (cfg.provider.lower(), cfg.name)
            if identity in seen_before:
                return
            seen_before.add(identity)
            extra_rows.append((label, cfg))

        for i, model in enumerate(extras, start=1):
            extra = replace(template, name=model)
            sanitized = sanitize_provider_config(extra, "general")
            if sanitized is None:
                continue
            _collect(f"Fallback 1.{i} (KieAI {sanitized.name})", sanitized)
        out[insert_at:insert_at] = extra_rows
        for _, cfg in extra_rows:
            seen.add((cfg.provider.lower(), cfg.name))
    return out


# ── Resilient LLM Execution Engine ────────────────────────────────────────────

_LLM_SEMAPHORE = asyncio.Semaphore(3)


def _clean_and_parse_json(content: str) -> dict:
    """Clean LLM output and parse it as JSON."""
    content = str(content).strip()
    
    # Remove <think>...</think> tags if present
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    
    # Try direct parsing first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
        
    # Try markdown json block extraction (with or without outer braces)
    m = re.search(r'```(?:json)?\s*\n?(.*?)\s*\n?```', content, re.DOTALL)
    if m:
        inner = m.group(1).strip()
        for candidate in (inner, f"{{{inner}}}"):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
            
    # Try searching for anything between first { and last }
    m = re.search(r'\{.*\}', content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
            
    logger.error(f"Failed to parse JSON. Raw LLM output: {content}")
    raise ValueError("Could not parse JSON from LLM response")


async def _invoke_provider(
    cfg: ModelConfig,
    api_key: str,
    prompt: str,
    system: str,
    temperature: Optional[float],
    max_tokens: Optional[int],
    response_json: bool,
    timeout: Optional[float] = None,
) -> str:
    prov = cfg.provider.lower()
    temp = temperature if temperature is not None else cfg.temperature
    tokens = max_tokens if max_tokens is not None else cfg.max_tokens
    client_timeout = _httpx_timeout(
        timeout if timeout is not None else _provider_timeout(prov),
        provider=prov,
    )

    if prov == "omniroute":
        # OmniRoute — OpenAI-compatible endpoint that auto-selects the best available model.
        # Supports all auto/* presets: auto/smart, auto/fast, auto/cheap, auto/reasoning,
        # auto/coding, auto/best-free, etc. Falls back internally if a model is unavailable.
        #
        # IMPORTANT: OmniRoute defaults to SSE streaming (text/event-stream).
        # We MUST set stream=False to get a standard JSON response body.
        base_url = cfg.base_url or _OMNIROUTE_BASE_URL
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": cfg.name,
            "messages": messages,
            "max_tokens": tokens,
            "temperature": temp,
            "stream": False,  # Force non-streaming JSON response
        }
        if response_json:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-trading-platform.local",
            "X-Title": "AI Trading Platform",
        }
        async with httpx.AsyncClient(timeout=client_timeout) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            if not resp.is_success:
                _raise_http_status(resp, "omniroute")

            content_type = resp.headers.get("content-type", "")

            # Handle SSE stream defensively (shouldn't happen with stream=False but guard anyway)
            if "text/event-stream" in content_type:
                text_pieces = []
                for line in resp.text.splitlines():
                    line = line.strip()
                    if line.startswith("data:"):
                        raw = line[5:].strip()
                        if raw and raw != "[DONE]":
                            try:
                                chunk = json.loads(raw)
                                choices = chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    piece = delta.get("content") or delta.get("reasoning_content", "")
                                    if piece:
                                        text_pieces.append(piece)
                            except json.JSONDecodeError:
                                pass
                content = "".join(text_pieces)
                if not content:
                    raise ValueError("OmniRoute SSE stream returned no content")
                return content

            # Standard JSON response
            try:
                data = resp.json()
            except Exception:
                raise ValueError(
                    f"OmniRoute returned non-JSON body (status={resp.status_code})"
                )
            choices = data.get("choices", [])
            if not choices:
                raise ValueError(f"OmniRoute returned no choices: {data}")
            msg = choices[0].get("message", {})
            content = msg.get("content") or msg.get("reasoning_content") or ""
            if not content:
                raise ValueError("OmniRoute returned empty content in message")
            used_model = data.get("model", cfg.name)
            logger.info(f"OmniRoute: success via model={used_model} provider={data.get('provider', '?')}")
            return content

    if prov in ("litellm", "xai", "groq", "openai", "openrouter", "openrouter-gemini"):
        # OpenAI chat completions format
        base_url = cfg.base_url or "https://api.openai.com/v1"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": cfg.name,
            "messages": messages,
            "max_tokens": tokens,
        }
        
        is_reasoning_model = any(x in cfg.name.lower() for x in ("o1-", "o3-", "reasoning"))
        if not is_reasoning_model:
            payload["temperature"] = temp
            if response_json:
                payload["response_format"] = {"type": "json_object"}
                
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        async with httpx.AsyncClient(timeout=client_timeout) as client:
            resp = await client.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
            if not resp.is_success:
                _raise_http_status(resp, prov)
            return resp.json()["choices"][0]["message"]["content"]
            
    elif prov == "kie":
        # docs.kie.ai: each family has its own path (Claude / Codex / Gemini / Grok / GPT-5.2).
        route = resolve_kie_route(cfg.name)
        url = _kie_request_url(cfg, route.path)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        logger.debug("Kie request: POST %s kind=%s model=%s", url, route.kind, route.model)

        if route.kind == "claude":
            messages = [{"role": "user", "content": prompt}]
            payload: dict[str, Any] = {
                "model": route.model,
                "messages": messages,
                "max_tokens": tokens,
                "stream": False,
            }
            if system:
                payload["system"] = [
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                ]
            async with httpx.AsyncClient(timeout=client_timeout) as client:
                for attempt_tokens in (tokens, tokens * 2):
                    payload["max_tokens"] = attempt_tokens
                    resp = await client.post(url, headers=headers, json=payload)
                    if not resp.is_success:
                        _raise_http_status(resp, prov)
                    data = resp.json()
                    text = _extract_kie_claude_text(data)
                    if response_json:
                        stripped = text.lstrip()
                        if not stripped.startswith("{") and not stripped.startswith("```"):
                            text = "{" + text
                    out_tokens = (data.get("usage") or {}).get("output_tokens", 0)
                    truncated = data.get("stop_reason") == "max_tokens" or out_tokens >= attempt_tokens
                    if truncated and response_json:
                        logger.warning(
                            f"LLM output truncated at max_tokens={attempt_tokens} for {cfg.name}; retrying with larger budget"
                        )
                        continue
                    return text
                raise ValueError(f"Output still truncated at max_tokens={tokens * 2} for {cfg.name}")

        if route.kind in ("codex_responses", "gpt_codex", "grok_responses"):
            payload = _kie_responses_payload(route.model, prompt, system)
            async with httpx.AsyncClient(timeout=client_timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if not resp.is_success:
                    _raise_http_status(resp, prov)
                return _extract_kie_responses_text(resp.json())

        if route.kind == "openai_chat":
            payload = _kie_chat_completions_payload(
                route.model, prompt, system, tokens, response_json, temperature=temp
            )
            async with httpx.AsyncClient(timeout=client_timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if not resp.is_success:
                    _raise_http_status(resp, prov)
                return _extract_kie_chat_text(resp.json())

        _assert_never_kie_kind(route.kind)

    elif prov == "anthropic":
        # Anthropic messages format
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        messages = [{"role": "user", "content": prompt}]
        if response_json:
            messages.append({"role": "assistant", "content": "{"})
        payload = {
            "model": cfg.name,
            "messages": messages,
            "max_tokens": tokens,
        }
        if system:
            payload["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
            
        async with httpx.AsyncClient(timeout=client_timeout) as client:
            for attempt_tokens in (tokens, tokens * 2):
                payload["max_tokens"] = attempt_tokens
                resp = await client.post(url, headers=headers, json=payload)
                if not resp.is_success:
                    _raise_http_status(resp, prov)
                data = resp.json()
                text = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        text += block.get("text", "")
                if response_json:
                    stripped = text.lstrip()
                    if not stripped.startswith("{") and not stripped.startswith("```"):
                        text = "{" + text
                out_tokens = (data.get("usage") or {}).get("output_tokens", 0)
                truncated = data.get("stop_reason") == "max_tokens" or out_tokens >= attempt_tokens
                if truncated and response_json:
                    logger.warning(
                        f"LLM output truncated at max_tokens={attempt_tokens} for {cfg.name}; retrying with larger budget"
                    )
                    continue
                return text
            raise ValueError(f"Output still truncated at max_tokens={tokens * 2} for {cfg.name}")
            
    elif prov in ("google", "gemini"):
        # Google Gemini generateContent format
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.name}:generateContent?key={api_key}"
        
        contents = []
        if system:
            contents.append({"role": "user", "parts": [{"text": f"System: {system}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temp,
                "maxOutputTokens": tokens,
            }
        }
        if response_json:
            payload["generationConfig"]["responseMimeType"] = "application/json"
            
        async with httpx.AsyncClient(timeout=client_timeout) as client:
            resp = await client.post(url, json=payload)
            if not resp.is_success:
                _raise_http_status(resp, prov)
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError("Gemini returned no candidates")
            return candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
    elif prov == "ollama":
        # Ollama local chat format
        base_url = cfg.base_url or "http://localhost:11434"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": cfg.name,
            "messages": messages,
            "stream": False,
        }
        
        async with httpx.AsyncClient(timeout=client_timeout) as client:
            resp = await client.post(f"{base_url.rstrip('/')}/api/chat", json=payload)
            if not resp.is_success:
                _raise_http_status(resp, prov)
            return resp.json().get("message", {}).get("content", "")
            
    else:
        raise ValueError(f"Unsupported LLM provider: {prov}")


def _provider_configured(cfg: ModelConfig) -> bool:
    if cfg.provider in ("ollama",):
        return True
    key = get_api_key(cfg)
    if not key:
        return False
    if len(key) < 20 and cfg.api_key_env in ("XAI_API_KEY", "GOOGLE_API_KEY"):
        return False
    if any(marker in key.lower() for marker in ("changeme", "placeholder", "your_", "xxx")):
        return False
    if cfg.provider == "anthropic" and not key.startswith("sk-ant-"):
        logger.warning("LLM Router: skipping malformed Anthropic API key")
        return False
    return True


def build_provider_chain(task_type: str) -> list[tuple[str, ModelConfig]]:
    """Primary + configured fallbacks, with dead catalog ids remapped/skipped."""
    primary_cfg = pick_model(task_type)
    raw_chain: list[tuple[str, ModelConfig]] = [
        ("Primary (OmniRoute)", primary_cfg),
        ("Fallback 1 (KieAI)", _kie_template_config()),
        ("Fallback 2 (OpenRouter)", _DEFAULT_REGISTRY["fallback_1"]),
        ("Fallback 3 (xAI)", ModelConfig(
            name=os.getenv("XAI_MODEL", "grok-4-1-fast-reasoning"),
            provider="xai",
            base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
            api_key_env="XAI_API_KEY",
        )),
        ("Fallback 4 (OpenAI)", ModelConfig(
            name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            provider="openai",
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key_env="OPENAI_API_KEY",
        )),
        ("Fallback 5 (Anthropic)", _DEFAULT_REGISTRY["fallback_2"]),
        ("Fallback 6 (Gemini)", _DEFAULT_REGISTRY["fallback_3"]),
    ]
    if os.getenv("OLLAMA_ENABLED", "false").lower() == "true":
        raw_chain.append(("Fallback 7 (Ollama)", ModelConfig(
            name=os.getenv("OLLAMA_PRIMARY_MODEL", "phi3.5"),
            provider="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )))

    configs_to_try: list[tuple[str, ModelConfig]] = []
    seen: set[tuple[str, str]] = set()

    for idx, (label, cfg) in enumerate(raw_chain):
        sanitized = sanitize_provider_config(cfg, task_type)
        if sanitized is None:
            continue
        is_primary = idx == 0
        if not is_primary and not _provider_configured(sanitized):
            continue
        identity = (sanitized.provider.lower(), sanitized.name)
        if identity in seen:
            continue
        seen.add(identity)
        configs_to_try.append((label, sanitized))
    return _expand_kie_model_hops(configs_to_try)


async def call_llm_resilient(
    task_type: str,
    prompt: str,
    system: str = "",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_json: bool = False,
) -> str:
    """
    Highly resilient LLM executor.

    1. Acquires a semaphore to limit concurrency.
    2. Tries the primary model once; timeouts/connect errors fail over immediately.
    3. HTTP 5xx / 429 get one short retry, then the next fallback.
    4. Stops when the chain budget is exhausted so the trading loop is not stalled.
    5. Cleans and parses output (removing <think> tags, extracting JSON if requested).
    """
    configs_to_try = build_provider_chain(task_type)
    budget = _chain_budget_seconds(task_type)
    started = time.monotonic()
    last_error: Optional[BaseException] = None
    last_provider = "none"

    async with _LLM_SEMAPHORE:
        for attempt_name, cfg in configs_to_try:
            remaining = budget - (time.monotonic() - started)
            if remaining <= 1.0:
                logger.error(
                    "LLM Router: chain budget exhausted (%.1fs) before %s; skipping remaining providers",
                    budget,
                    attempt_name,
                )
                break

            api_key = get_api_key(cfg)
            timeout = min(_provider_timeout(cfg.provider), max(1.0, remaining - 0.2))
            extra_retries = 1  # one extra attempt only when classify_llm_error == retry
            attempt = 0
            while attempt <= extra_retries:
                attempt += 1
                remaining = budget - (time.monotonic() - started)
                if remaining <= 0.5:
                    logger.error(
                        "LLM Router: chain budget exhausted (%.1fs) during %s",
                        budget,
                        attempt_name,
                    )
                    break
                timeout = min(timeout, max(1.0, remaining - 0.2))
                try:
                    logger.info(
                        "LLM Router: Trying %s (model=%s, provider=%s, attempt=%s, timeout=%.1fs)",
                        attempt_name,
                        cfg.name,
                        cfg.provider,
                        attempt,
                        timeout,
                    )
                    text = await asyncio.wait_for(
                        _invoke_provider(
                            cfg,
                            api_key,
                            prompt,
                            system,
                            temperature,
                            max_tokens,
                            response_json,
                            timeout=timeout,
                        ),
                        timeout=timeout + 0.75,
                    )
                    if response_json:
                        parsed = _clean_and_parse_json(text)
                        text = json.dumps(parsed)
                    logger.info("LLM Router: Success using %s", attempt_name)
                    _record_llm_success(task_type, cfg.provider, cfg.name)
                    return text
                except Exception as e:
                    last_error = e
                    last_provider = cfg.provider
                    action = classify_llm_error(e)
                    summary = _error_summary(e)
                    _record_llm_failure(task_type, cfg.provider, summary)
                    logger.warning(
                        "LLM Router: %s attempt %s failed (%s): %s",
                        attempt_name,
                        attempt,
                        action,
                        summary,
                    )
                    if action == "retry" and attempt <= extra_retries:
                        backoff = _retry_backoff_seconds()
                        logger.info(
                            "LLM Router: retrying %s in %.2fs then failing over if it fails again",
                            attempt_name,
                            backoff,
                        )
                        if backoff:
                            await asyncio.sleep(backoff)
                        continue
                    break

            logger.error("LLM Router: giving up on %s; moving to next fallback.", attempt_name)

        summary = _error_summary(last_error) if last_error else "no providers attempted"
        _record_llm_failure(task_type, last_provider, summary, exhausted=True)
        err_msg = f"All LLM providers in the chain failed. Last error: {summary}"
        logger.critical(err_msg)
        raise LLMChainExhausted(err_msg)
