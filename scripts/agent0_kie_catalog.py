"""Agent Zero Kie.ai model catalog (docs.kie.ai family endpoints).

Keep lists in sync with ``backend.llm.router.resolve_kie_route`` and
``scripts/kieai_proxy.py``. No secrets in this module.
"""

from __future__ import annotations

KIE_PROXY_BASE = "http://kieai-proxy:11434/v1"
OMNI_BASE = "https://omni.allikas.online/v1"

CLAUDE_MODELS = [
    "claude-fable-5-1",
    "claude-fable-5",
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-haiku-4-5",
]
GPT_MODELS = [
    "gpt-6-astra",
    "gpt-5-6-luna",
    "gpt-5-6-sol",
    "gpt-5-6-terra",
    "gpt-5-2",
]
CODEX_MODELS = [
    "gpt-5.4-codex",
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5.1-codex",
    "gpt-5-codex",
]
GEMINI_MODELS = [
    "gemini-3-8-flash-openai",
    "gemini-3-6-flash-openai",
    "gemini-3-flash",
    "gemini-3-pro",
]
GROK_MODELS = [
    "grok-4-6",
    "grok-4-5",
]
DEEPSEEK_MODELS = [
    "deepseek-chat",
    "deepseek-reasoner",
]
OMNI_MODELS = [
    "auto/chat",
    "auto/cheap",
    "auto/fast",
    "auto/best-coding",
]

ALL_KIE_MODELS = (
    CLAUDE_MODELS
    + GPT_MODELS
    + CODEX_MODELS
    + GEMINI_MODELS
    + GROK_MODELS
    + DEEPSEEK_MODELS
)


def _chat_slot(provider: str, name: str, ctx_length: int = 200000) -> dict:
    return {
        "provider": provider,
        "name": name,
        "ctx_length": ctx_length,
        "ctx_history": 0.7,
        "vision": True,
        "max_tokens": 4096,
        "temperature": 0.7,
    }


def _util_slot(provider: str, name: str) -> dict:
    return {"provider": provider, "name": name}


# Presets upserted into Agent Zero ``plugins/_model_config/presets.yaml``.
AGENT0_KIE_PRESETS: list[dict] = [
    {
        "name": "Kie.ai GPT Astra",
        "chat": _chat_slot("kieai-gpt", "gpt-6-astra", 1000000),
        "utility": _util_slot("kieai-gpt", "gpt-5-6-luna"),
    },
    {
        "name": "Kie.ai Fable 5.1",
        "chat": _chat_slot("kieai-claude", "claude-fable-5-1"),
        "utility": _util_slot("kieai-claude", "claude-haiku-4-5"),
    },
    {
        "name": "Kie.ai Fable 5",
        "chat": _chat_slot("kieai-claude", "claude-fable-5"),
        "utility": _util_slot("kieai-claude", "claude-haiku-4-5"),
    },
    {
        "name": "Kie.ai Claude Sonnet 5",
        "chat": _chat_slot("kieai-claude", "claude-sonnet-5"),
        "utility": _util_slot("kieai-claude", "claude-haiku-4-5"),
    },
    {
        "name": "Kie.ai Claude Opus 5",
        "chat": _chat_slot("kieai-claude", "claude-opus-5"),
        "utility": _util_slot("kieai-claude", "claude-sonnet-5"),
    },
    {
        "name": "Kie.ai GPT 5.6 Luna",
        "chat": _chat_slot("kieai-gpt", "gpt-5-6-luna"),
        "utility": _util_slot("kieai-gpt", "gpt-5-6-luna"),
    },
    {
        "name": "Kie.ai GPT 5.6 Sol",
        "chat": _chat_slot("kieai-gpt", "gpt-5-6-sol"),
        "utility": _util_slot("kieai-gpt", "gpt-5-6-luna"),
    },
    {
        "name": "Kie.ai Gemini 3.8 Flash",
        "chat": _chat_slot("kieai-gemini", "gemini-3-8-flash-openai"),
        "utility": _util_slot("kieai-gemini", "gemini-3-6-flash-openai"),
    },
    {
        "name": "Kie.ai Grok 4.6",
        "chat": _chat_slot("kieai-grok", "grok-4-6"),
        "utility": _util_slot("kieai-gpt", "gpt-5-6-luna"),
    },
    {
        "name": "Kie.ai Sonnet",
        "chat": _chat_slot("kieai-claude", "claude-sonnet-4-6"),
        "utility": _util_slot("kieai-claude", "claude-haiku-4-5"),
    },
    {
        "name": "Kie.ai Haiku",
        "chat": _chat_slot("kieai-claude", "claude-haiku-4-5"),
        "utility": _util_slot("kieai-claude", "claude-haiku-4-5"),
    },
    {
        "name": "Kie.ai Opus",
        "chat": _chat_slot("kieai-claude", "claude-opus-4-6"),
        "utility": _util_slot("kieai-claude", "claude-sonnet-4-6"),
    },
    {
        "name": "Kie.ai Codex 5.4",
        "chat": _chat_slot("kieai-gpt-codex", "gpt-5.4-codex", 128000),
        "utility": _util_slot("kieai-gpt-codex", "gpt-5.4-codex"),
    },
    {
        "name": "Kie.ai Codex 5.1",
        "chat": _chat_slot("kieai-gpt-codex", "gpt-5.1-codex", 128000),
        "utility": _util_slot("kieai-gpt-codex", "gpt-5.4-codex"),
    },
    {
        "name": "Kie.ai GPT 5.2",
        "chat": _chat_slot("kieai-gpt", "gpt-5-2", 128000),
        "utility": _util_slot("kieai-gpt", "gpt-5-2"),
    },
]


def build_providers(kie_key: str, omni_key: str) -> dict[str, dict]:
    """LiteLLM OpenAI-compat providers that talk to the family-aware Kie proxy."""
    kie_kwargs = {
        "a0_api_mode": "chat",
        "api_base": KIE_PROXY_BASE,
        "api_key": kie_key,
    }
    return {
        "kieai-claude": {
            "name": "Kie.ai Claude",
            "litellm_provider": "openai",
            "models_list": {"list": list(CLAUDE_MODELS)},
            "kwargs": dict(kie_kwargs),
        },
        "kieai-gpt": {
            "name": "Kie.ai GPT",
            "litellm_provider": "openai",
            "models_list": {"list": list(GPT_MODELS)},
            "kwargs": dict(kie_kwargs),
        },
        "kieai-gpt-codex": {
            "name": "Kie.ai GPT Codex",
            "litellm_provider": "openai",
            "models_list": {"list": list(CODEX_MODELS)},
            "kwargs": dict(kie_kwargs),
        },
        "kieai-gemini": {
            "name": "Kie.ai Gemini",
            "litellm_provider": "openai",
            "models_list": {"list": list(GEMINI_MODELS)},
            "kwargs": dict(kie_kwargs),
        },
        "kieai-grok": {
            "name": "Kie.ai Grok",
            "litellm_provider": "openai",
            "models_list": {"list": list(GROK_MODELS)},
            "kwargs": dict(kie_kwargs),
        },
        "kieai": {
            "name": "Kie.ai proxy",
            "litellm_provider": "openai",
            "models_list": {"list": list(ALL_KIE_MODELS)},
            "kwargs": dict(kie_kwargs),
        },
        "omniroute": {
            "name": "OmniRoute",
            "litellm_provider": "openai",
            "models_list": {"list": list(OMNI_MODELS)},
            "kwargs": {
                "a0_api_mode": "chat",
                "api_base": OMNI_BASE,
                "api_key": omni_key,
            },
        },
    }
