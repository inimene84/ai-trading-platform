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
from typing import Literal, Optional

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
#   Automatically selects the best free/cheapest available model per task.
#   auto/smart   → highest quality (default for analysis)
#   auto/coding  → best for code tasks
#   auto/reasoning → best for complex multi-step reasoning
#   auto/fast    → lowest latency
#   auto/cheap   → lowest cost
#   auto/best-free → best completely free model available
#
# Fallback chain: OmniRoute → KieAI → OpenRouter → xAI → OpenAI → Anthropic → Gemini

# OmniRoute config
_OMNIROUTE_BASE_URL = os.getenv("OMNIROUTE_BASE_URL", "https://omni.allikas.online/v1")

# Kie.ai direct model IDs (fallback)
_KIE_MODEL = os.getenv("KIE_MODEL", "gpt-5-6-terra")
_KIE_OPUS_DIRECT_MODEL = os.getenv("KIE_OPUS_MODEL", "gpt-5-6-terra")
_KIE_BASE_URL = os.getenv("KIE_BASE_URL", "https://api.kie.ai")
_LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", os.getenv("PERSONA_LLM_BASE_URL", "http://litellm:4000/v1"))

_DEFAULT_REGISTRY: dict[str, ModelConfig] = {
    # PRIMARY: OmniRoute auto/smart — selects the best available LLM automatically
    # Task-specific presets give the router hints for optimal model selection.
    "persona_analysis": ModelConfig(
        name=os.getenv("PERSONA_LLM_MODEL", "auto/smart"),
        provider="omniroute",
        tier="balanced",
        base_url=_OMNIROUTE_BASE_URL,
        max_tokens=1024,
        temperature=0.3,
        api_key_env="OMNIROUTE_API_KEY",
    ),

    # Deep trading analysis — OmniRoute auto/smart for robust analysis
    "deep_analysis": ModelConfig(
        name=os.getenv("DEEP_ANALYSIS_LLM_MODEL", "auto/smart"),
        provider="omniroute",
        tier="balanced",
        base_url=_OMNIROUTE_BASE_URL,
        max_tokens=1500,
        temperature=0.3,
        api_key_env="OMNIROUTE_API_KEY",
    ),

    # Premium/complex reasoning — OmniRoute auto/smart (highest quality preset)
    "premium_analysis": ModelConfig(
        name=os.getenv("PREMIUM_ANALYSIS_LLM_MODEL", "auto/smart"),
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
        name=os.getenv("ASSISTANT_LLM_MODEL", "auto/smart"),
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
    # KieAI fallback (direct Kie.ai GPT-5.6 Terra / Luna)
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
_KIE_NATIVE_MODELS = {"gpt-5-6-terra", "gpt-5-6-luna", "gpt-5-6-sol"}
_DEAD_CATALOG_PREFIXES = ("kie/", "litellm/")

# Per-provider read timeouts. Kept short so a dead primary cannot consume a
# trading cycle. Override with LLM_PROVIDER_TIMEOUT_SECONDS (global) or
# LLM_<PROVIDER>_TIMEOUT_SECONDS (e.g. LLM_OMNIROUTE_TIMEOUT_SECONDS).
_PROVIDER_TIMEOUT_DEFAULTS = {
    "omniroute": 8.0,
    "kie": 8.0,
    "openrouter": 10.0,
    "openrouter-gemini": 10.0,
    "xai": 10.0,
    "openai": 10.0,
    "anthropic": 10.0,
    "google": 10.0,
    "gemini": 10.0,
    "groq": 8.0,
    "litellm": 10.0,
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
    global_override = os.getenv("LLM_PROVIDER_TIMEOUT_SECONDS")
    if global_override:
        return _env_float("LLM_PROVIDER_TIMEOUT_SECONDS", 8.0)
    key = f"LLM_{provider.upper().replace('-', '_')}_TIMEOUT_SECONDS"
    per = os.getenv(key)
    if per:
        return _env_float(key, _PROVIDER_TIMEOUT_DEFAULTS.get(provider.lower(), 10.0))
    return _PROVIDER_TIMEOUT_DEFAULTS.get(provider.lower(), 10.0)


def _chain_budget_seconds(task_type: str) -> float:
    """Hard cap for the whole fallback chain so the trading loop is not stalled."""
    if task_type == "persona_analysis":
        return _env_float("LLM_PERSONA_CHAIN_BUDGET_SECONDS", 16.0)
    return _env_float("LLM_CHAIN_BUDGET_SECONDS", 24.0)


def _retry_backoff_seconds() -> float:
    return _env_float("LLM_RETRY_BACKOFF_SECONDS", 0.35, allow_zero=True)


def _httpx_timeout(seconds: float) -> httpx.Timeout:
    read = max(1.0, float(seconds))
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
    if task_type == "general":
        return "auto/fast"
    return "auto/smart"


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
            or lower in _KIE_NATIVE_MODELS
        )
        if looks_dead:
            preset = _omniroute_preset_for_task(task_type)
            if name != preset:
                logger.warning(
                    "LLM Router: remapping model %r → OmniRoute %s (dead kie/catalog id)",
                    name,
                    preset,
                )
            return replace(cfg, name=preset)
        return cfg

    if provider == "kie":
        cleaned = name[4:] if lower.startswith("kie/") else name
        cleaned = cleaned.strip()
        if not cleaned or "/" in cleaned:
            logger.warning("LLM Router: skipping Kie fallback with unusable model %r", name)
            return None
        if cleaned != name:
            logger.info("LLM Router: stripping kie/ prefix → %s", cleaned)
            return replace(cfg, name=cleaned)
        return cfg

    return cfg


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
    client_timeout = _httpx_timeout(timeout if timeout is not None else _provider_timeout(prov))

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
        # Kie.ai supports Claude (/claude/v1/messages) and GPT/ChatGPT models (/codex/v1/responses)
        is_claude = "claude" in cfg.name.lower()
        if is_claude:
            url = f"{cfg.base_url.rstrip('/')}/v1/messages" if "/claude" in (cfg.base_url or "") else "https://api.kie.ai/claude/v1/messages"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            messages = [{"role": "user", "content": prompt}]
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
        else:
            # GPT / ChatGPT / Codex models on Kie.ai (e.g. gpt-5-6-terra, gpt-5-6-luna)
            url = f"{cfg.base_url.rstrip('/')}/codex/v1/responses" if cfg.base_url and "api.kie.ai" in cfg.base_url else "https://api.kie.ai/codex/v1/responses"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            input_msgs = []
            if system:
                input_msgs.append({
                    "role": "system",
                    "content": [{"type": "input_text", "text": system}]
                })
            input_msgs.append({
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}]
            })
            payload = {
                "model": cfg.name,
                "stream": False,
                "input": input_msgs,
                "reasoning": {"effort": "low"},
            }
            async with httpx.AsyncClient(timeout=client_timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if not resp.is_success:
                    _raise_http_status(resp, prov)
                data = resp.json()
                text = ""
                if isinstance(data.get("output"), list):
                    for item in data["output"]:
                        for part in item.get("content", []):
                            if isinstance(part, dict) and part.get("type") in ("text", "output_text"):
                                text += part.get("text", "")
                elif isinstance(data.get("choices"), list) and data["choices"]:
                    choice = data["choices"][0]
                    msg = choice.get("message") or {}
                    text = msg.get("content", "") if isinstance(msg, dict) else str(choice.get("text", ""))
                elif "data" in data and isinstance(data["data"], dict):
                    text = data["data"].get("content", "") or data["data"].get("text", "")
                elif "text" in data and isinstance(data["text"], str):
                    text = data["text"]
                elif "response" in data and isinstance(data["response"], str):
                    text = data["response"]
                if not text and isinstance(data, dict):
                    text = json.dumps(data)
                return text

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
        ("Fallback 1 (KieAI)", _DEFAULT_REGISTRY["fallback_kie"]),
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
    return configs_to_try


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
                    return text
                except Exception as e:
                    last_error = e
                    action = classify_llm_error(e)
                    logger.warning(
                        "LLM Router: %s attempt %s failed (%s): %s",
                        attempt_name,
                        attempt,
                        action,
                        _error_summary(e),
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
        err_msg = f"All LLM providers in the chain failed. Last error: {summary}"
        logger.critical(err_msg)
        raise LLMChainExhausted(err_msg)
