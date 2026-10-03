import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.llm.router import (
    LLMChainExhausted,
    ModelConfig,
    _OMNIROUTE_TIMEOUT_FLOOR,
    _invoke_provider,
    _provider_timeout,
    build_provider_chain,
    classify_llm_error,
    call_llm_resilient,
    get_llm_router_status,
    resolve_kie_route,
    sanitize_provider_config,
)
from backend.services.persona_adapter import (
    _LLM_UNAVAILABLE_PREFIX,
    _PERSONA_OPINION_CACHE,
    run_all_personas,
    run_persona,
)


def _cfg(provider: str, name: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        provider=provider,
        api_key_env=f"{provider.upper()}_API_KEY",
    )


def _status_error(status: int, body: str = '{"error":{"type":"server_error"}}') -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.kie.ai/codex/v1/responses")
    response = httpx.Response(status, text=body, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}: {body}", request=request, response=response)


def _patch_chain(monkeypatch, *configs: tuple[str, ModelConfig]):
    from backend.llm import router as router_mod

    chain = list(configs)
    monkeypatch.setattr(router_mod, "build_provider_chain", lambda task_type: chain)
    monkeypatch.setattr(router_mod, "get_api_key", lambda cfg: "k" * 32)
    monkeypatch.setenv("LLM_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("LLM_CHAIN_BUDGET_SECONDS", "30")
    monkeypatch.setenv("LLM_PERSONA_CHAIN_BUDGET_SECONDS", "30")


def test_classify_timeout_fails_over():
    assert classify_llm_error(httpx.TimeoutException("timed out")) == "failover"
    assert classify_llm_error(TimeoutError("timed out")) == "failover"
    assert classify_llm_error(asyncio.TimeoutError()) == "failover"


def test_classify_http_500_is_retryable():
    assert classify_llm_error(_status_error(500)) == "retry"
    assert classify_llm_error(_status_error(429)) == "retry"
    assert classify_llm_error(_status_error(503)) == "retry"


def test_classify_http_400_fails_over():
    assert classify_llm_error(_status_error(400, "bad model")) == "failover"
    assert classify_llm_error(_status_error(401)) == "failover"


def test_sanitize_remaps_dead_kie_catalog_on_omniroute(monkeypatch):
    monkeypatch.delenv("OMNIROUTE_DEFAULT_MODEL", raising=False)
    cfg = ModelConfig(name="kie/claude-opus-4", provider="omniroute", api_key_env="OMNIROUTE_API_KEY")
    out = sanitize_provider_config(cfg, "persona_analysis")
    assert out is not None
    assert out.name == "auto/fast"
    assert out.provider == "omniroute"


def test_sanitize_remaps_kie_native_id_on_omniroute_to_auto_fast_for_general(monkeypatch):
    monkeypatch.delenv("OMNIROUTE_DEFAULT_MODEL", raising=False)
    cfg = ModelConfig(name="gpt-5-6-terra", provider="omniroute", api_key_env="OMNIROUTE_API_KEY")
    out = sanitize_provider_config(cfg, "general")
    assert out is not None
    assert out.name == "auto/fast"


def test_sanitize_omniroute_default_model_overrides_remap(monkeypatch):
    monkeypatch.setenv("OMNIROUTE_DEFAULT_MODEL", "auto/cheap")
    cfg = ModelConfig(name="kie/dead", provider="omniroute", api_key_env="OMNIROUTE_API_KEY")
    out = sanitize_provider_config(cfg, "persona_analysis")
    assert out is not None
    assert out.name == "auto/cheap"


def test_sanitize_keeps_omniroute_auto_presets():
    cfg = ModelConfig(name="auto/fast", provider="omniroute", api_key_env="OMNIROUTE_API_KEY")
    out = sanitize_provider_config(cfg, "persona_analysis")
    assert out is not None
    assert out.name == "auto/fast"


def test_sanitize_skips_unusable_kie_catalog_path():
    cfg = ModelConfig(name="kie/openrouter/free", provider="kie", api_key_env="KIE_API_KEY")
    assert sanitize_provider_config(cfg, "persona_analysis") is None


def test_sanitize_strips_kie_prefix_for_native_model():
    cfg = ModelConfig(name="kie/gpt-5-6-terra", provider="kie", api_key_env="KIE_API_KEY")
    out = sanitize_provider_config(cfg, "persona_analysis")
    assert out is not None
    assert out.name == "gpt-5-6-terra"
    assert out.provider == "kie"


@pytest.mark.asyncio
async def test_timeout_fails_over_immediately_without_retrying_primary(monkeypatch):
    omni = _cfg("omniroute", "auto/smart")
    kie = _cfg("kie", "gpt-5-6-terra")
    _patch_chain(monkeypatch, ("Primary (OmniRoute)", omni), ("Fallback 1 (KieAI)", kie))

    calls: list[str] = []
    sleeps: list[float] = []

    async def fake_invoke(cfg, *args, **kwargs):
        calls.append(cfg.provider)
        if cfg.provider == "omniroute":
            raise httpx.TimeoutException("OmniRoute timed out")
        return '{"ok": true}'

    async def fake_sleep(delay):
        sleeps.append(delay)

    with patch("backend.llm.router._invoke_provider", new=fake_invoke), patch(
        "backend.llm.router.asyncio.sleep", new=fake_sleep
    ):
        text = await call_llm_resilient("persona_analysis", prompt="hi")

    assert text == '{"ok": true}'
    assert calls == ["omniroute", "kie"]
    assert sleeps == []


@pytest.mark.asyncio
async def test_http_500_retries_once_then_fails_over(monkeypatch):
    omni = _cfg("omniroute", "auto/smart")
    kie = _cfg("kie", "gpt-5-6-terra")
    openrouter = _cfg("openrouter", "anthropic/claude-sonnet-5")
    _patch_chain(
        monkeypatch,
        ("Primary (OmniRoute)", omni),
        ("Fallback 1 (KieAI)", kie),
        ("Fallback 2 (OpenRouter)", openrouter),
    )
    monkeypatch.setenv("LLM_RETRY_BACKOFF_SECONDS", "0.01")

    calls: list[str] = []

    async def fake_invoke(cfg, *args, **kwargs):
        calls.append(cfg.provider)
        if cfg.provider == "omniroute":
            raise httpx.TimeoutException("OmniRoute timed out")
        if cfg.provider == "kie":
            raise _status_error(500, '{"error":{"type":"server_error","message":"Server exception"}}')
        return "openrouter-ok"

    with patch("backend.llm.router._invoke_provider", new=fake_invoke):
        text = await call_llm_resilient("deep_analysis", prompt="hi")

    assert text == "openrouter-ok"
    assert calls[0] == "omniroute"
    assert calls.count("kie") == 2
    assert calls[-1] == "openrouter"


@pytest.mark.asyncio
async def test_http_500_succeeds_on_short_retry(monkeypatch):
    kie = _cfg("kie", "gpt-5-6-terra")
    _patch_chain(monkeypatch, ("Fallback 1 (KieAI)", kie))
    monkeypatch.setenv("LLM_RETRY_BACKOFF_SECONDS", "0")

    calls = {"n": 0}

    async def fake_invoke(cfg, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _status_error(500)
        return "kie-recovered"

    with patch("backend.llm.router._invoke_provider", new=fake_invoke):
        text = await call_llm_resilient("general", prompt="hi")

    assert text == "kie-recovered"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_all_providers_exhausted_raises_chain_error(monkeypatch):
    omni = _cfg("omniroute", "auto/fast")
    kie = _cfg("kie", "gpt-5-6-terra")
    _patch_chain(monkeypatch, ("Primary (OmniRoute)", omni), ("Fallback 1 (KieAI)", kie))

    async def fake_invoke(cfg, *args, **kwargs):
        if cfg.provider == "omniroute":
            raise httpx.TimeoutException("timed out")
        raise _status_error(500)

    with patch("backend.llm.router._invoke_provider", new=fake_invoke):
        with pytest.raises(LLMChainExhausted, match="All LLM providers in the chain failed"):
            await call_llm_resilient("general", prompt="hi")
    status = get_llm_router_status()
    assert status["degraded"] is True
    assert status["chain_exhausted"] is True
    assert status["last_error"]
    assert status["last_error_provider"] in {"omniroute", "kie"}


@pytest.mark.asyncio
async def test_chain_budget_skips_remaining_providers(monkeypatch):
    from backend.llm import router as router_mod

    omni = _cfg("omniroute", "auto/smart")
    kie = _cfg("kie", "gpt-5-6-terra")
    openrouter = _cfg("openrouter", "anthropic/claude-sonnet-5")
    _patch_chain(
        monkeypatch,
        ("Primary (OmniRoute)", omni),
        ("Fallback 1 (KieAI)", kie),
        ("Fallback 2 (OpenRouter)", openrouter),
    )
    monkeypatch.setenv("LLM_CHAIN_BUDGET_SECONDS", "5")

    class Clock:
        def __init__(self):
            self.t = 0.0

        def monotonic(self):
            return self.t

    clock = Clock()
    monkeypatch.setattr(router_mod.time, "monotonic", clock.monotonic)

    calls: list[str] = []

    async def fake_invoke(cfg, *args, **kwargs):
        calls.append(cfg.provider)
        clock.t += 10.0
        raise httpx.TimeoutException("slow")

    with patch("backend.llm.router._invoke_provider", new=fake_invoke):
        with pytest.raises(LLMChainExhausted):
            await call_llm_resilient("general", prompt="hi")

    assert calls == ["omniroute"]


@pytest.mark.asyncio
async def test_dead_catalog_model_is_not_sent_to_omniroute(monkeypatch):
    from backend.llm import router as router_mod

    monkeypatch.setenv("OMNIROUTE_API_KEY", "k" * 32)
    monkeypatch.delenv("KIE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LLM_CHAIN_BUDGET_SECONDS", "30")

    original = router_mod._DEFAULT_REGISTRY["persona_analysis"]
    monkeypatch.setattr(original, "name", "kie/claude-opus-4")

    seen: list[str] = []

    async def fake_invoke(cfg, *args, **kwargs):
        seen.append(cfg.name)
        return "ok"

    with patch("backend.llm.router._invoke_provider", new=fake_invoke):
        text = await call_llm_resilient("persona_analysis", prompt="hi")

    assert text == "ok"
    assert seen == ["auto/fast"]


@pytest.mark.asyncio
async def test_persona_skips_enrichment_on_llm_chain_failure():
    _PERSONA_OPINION_CACHE.clear()
    bars = [{"close": 100.0, "volume": 1.0}] * 5

    with patch(
        "backend.services.persona_adapter.call_llm_resilient",
        new=AsyncMock(side_effect=LLMChainExhausted("All LLM providers in the chain failed")),
    ):
        opinion = await run_persona("warren_buffett", "BTCUSDC", bars, metrics={})

    assert opinion.signal == "neutral"
    assert opinion.confidence == 0.0
    assert opinion.reasoning == f"{_LLM_UNAVAILABLE_PREFIX} LLMChainExhausted"
    assert "All LLM providers" not in opinion.reasoning
    assert ("BTCUSDC", "warren_buffett") not in _PERSONA_OPINION_CACHE


@pytest.mark.asyncio
async def test_run_all_personas_does_not_raise_when_llms_fail():
    _PERSONA_OPINION_CACHE.clear()
    bars = [{"close": 100.0, "volume": 1.0}] * 5

    with patch(
        "backend.services.persona_adapter.call_llm_resilient",
        new=AsyncMock(side_effect=LLMChainExhausted("All LLM providers in the chain failed")),
    ):
        opinions = await run_all_personas(
            "ETHUSDC",
            bars,
            metrics={},
            selected=["warren_buffett", "cathie_wood"],
        )

    assert len(opinions) == 0


@pytest.mark.asyncio
async def test_connect_error_fails_over_immediately(monkeypatch):
    omni = _cfg("omniroute", "auto/smart")
    kie = _cfg("kie", "gpt-5-6-terra")
    _patch_chain(monkeypatch, ("Primary (OmniRoute)", omni), ("Fallback 1 (KieAI)", kie))

    calls: list[str] = []

    async def fake_invoke(cfg, *args, **kwargs):
        calls.append(cfg.provider)
        if cfg.provider == "omniroute":
            raise httpx.ConnectError("connection refused")
        return "kie-ok"

    with patch("backend.llm.router._invoke_provider", new=fake_invoke):
        text = await call_llm_resilient("persona_analysis", prompt="hi")

    assert text == "kie-ok"
    assert calls == ["omniroute", "kie"]


@pytest.mark.asyncio
async def test_http_400_fails_over_without_retry(monkeypatch):
    kie = _cfg("kie", "gpt-5-6-terra")
    openrouter = _cfg("openrouter", "anthropic/claude-sonnet-5")
    _patch_chain(monkeypatch, ("Fallback 1 (KieAI)", kie), ("Fallback 2 (OpenRouter)", openrouter))

    calls: list[str] = []

    async def fake_invoke(cfg, *args, **kwargs):
        calls.append(cfg.provider)
        if cfg.provider == "kie":
            raise _status_error(400, "bad model")
        return "openrouter-ok"

    with patch("backend.llm.router._invoke_provider", new=fake_invoke):
        text = await call_llm_resilient("general", prompt="hi")

    assert text == "openrouter-ok"
    assert calls.count("kie") == 1
    assert calls[-1] == "openrouter"


@pytest.mark.asyncio
async def test_persona_enrichment_timeout_returns_empty(monkeypatch):
    _PERSONA_OPINION_CACHE.clear()
    monkeypatch.setenv("PERSONA_ENRICHMENT_TIMEOUT_SECONDS", "0.05")
    bars = [{"close": 100.0, "volume": 1.0}] * 5

    async def hang(*_args, **_kwargs):
        await asyncio.sleep(30)

    with patch("backend.services.persona_adapter.call_llm_resilient", new=hang):
        opinions = await run_all_personas(
            "BTCUSDC",
            bars,
            metrics={},
            selected=["warren_buffett"],
        )

    assert opinions == []


def test_build_provider_chain_drops_slashy_kie_catalog(monkeypatch):
    from backend.llm import router as router_mod

    monkeypatch.setenv("OMNIROUTE_API_KEY", "k" * 32)
    monkeypatch.setenv("KIE_API_KEY", "k" * 32)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OMNIROUTE_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("KIE_MODEL", raising=False)
    monkeypatch.setenv("KIE_FALLBACK_MODELS", "")
    monkeypatch.delenv("KIE_OPUS_MODEL", raising=False)
    monkeypatch.setattr(router_mod._DEFAULT_REGISTRY["fallback_kie"], "name", "kie/openrouter/free")

    chain = build_provider_chain("persona_analysis")
    providers = [cfg.provider for _, cfg in chain]
    assert "kie" not in providers
    assert providers[0] == "omniroute"


@pytest.mark.asyncio
async def test_kie_codex_payload_uses_input_text(monkeypatch):
    captured: dict = {}

    class FakeResp:
        is_success = True
        status_code = 200
        text = "{}"
        headers = {}

        def json(self):
            return {"output": [{"content": [{"type": "output_text", "text": "kie-ok"}]}]}

        @property
        def request(self):
            return httpx.Request("POST", "https://api.kie.ai/codex/v1/responses")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["json"] = json
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    cfg = ModelConfig(
        name="gpt-5-6-terra",
        provider="kie",
        base_url="https://api.kie.ai",
        api_key_env="KIE_API_KEY",
    )
    text = await _invoke_provider(cfg, "k" * 32, "hello", "sys", 0.3, 128, False, timeout=5)
    assert text == "kie-ok"
    assert captured["url"].endswith("/codex/v1/responses")
    assert captured["json"]["stream"] is False
    assert captured["json"]["reasoning"] == {"effort": "low"}
    assert captured["json"]["input"][0]["content"][0]["type"] == "input_text"
    assert captured["json"]["input"][1]["content"][0]["type"] == "input_text"


def test_http_error_message_does_not_embed_response_body():
    from backend.llm.router import _raise_http_status

    request = httpx.Request("POST", "https://api.kie.ai/codex/v1/responses")
    response = httpx.Response(
        401,
        text='{"error":"Incorrect API key provided: sk-secret-key"}',
        request=request,
    )
    with pytest.raises(httpx.HTTPStatusError, match="HTTP 401 from kie") as exc_info:
        _raise_http_status(response, "kie")
    assert "sk-secret-key" not in str(exc_info.value)


def test_omniroute_timeout_covers_measured_allikas_latency(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LLM_OMNIROUTE_TIMEOUT_SECONDS", raising=False)
    timeout = _provider_timeout("omniroute")
    assert timeout >= _OMNIROUTE_TIMEOUT_FLOOR
    assert timeout >= 20.0


def test_global_eight_second_timeout_does_not_clip_omniroute(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_TIMEOUT_SECONDS", "8")
    monkeypatch.delenv("LLM_OMNIROUTE_TIMEOUT_SECONDS", raising=False)
    assert _provider_timeout("omniroute") >= _OMNIROUTE_TIMEOUT_FLOOR
    assert _provider_timeout("kie") == 8.0


def test_sanitize_rewrites_kie_endpoint_used_as_omniroute_base():
    cfg = ModelConfig(
        name="auto/fast",
        provider="omniroute",
        base_url="https://api.kie.ai",
        api_key_env="OMNIROUTE_API_KEY",
    )
    out = sanitize_provider_config(cfg, "persona_analysis")
    assert out is not None
    assert out.name == "auto/fast"
    assert out.base_url == "https://omni.allikas.online/v1"


def test_sanitize_appends_v1_to_omniroute_host():
    cfg = ModelConfig(
        name="auto/fast",
        provider="omniroute",
        base_url="https://omni.allikas.online",
        api_key_env="OMNIROUTE_API_KEY",
    )
    out = sanitize_provider_config(cfg, "persona_analysis")
    assert out is not None
    assert out.base_url == "https://omni.allikas.online/v1"


@pytest.mark.parametrize(
    ("model", "kind", "path_suffix", "canonical"),
    [
        ("gpt-5-6-luna", "codex_responses", "/codex/v1/responses", "gpt-5-6-luna"),
        ("gpt-6-astra", "codex_responses", "/codex/v1/responses", "gpt-6-astra"),
        ("gpt-astra", "codex_responses", "/codex/v1/responses", "gpt-6-astra"),
        ("claude-fable-5", "claude", "/claude/v1/messages", "claude-fable-5"),
        ("claude-fable-5.1", "claude", "/claude/v1/messages", "claude-fable-5-1"),
        ("fable-5.1", "claude", "/claude/v1/messages", "claude-fable-5-1"),
        ("claude-sonnet-5", "claude", "/claude/v1/messages", "claude-sonnet-5"),
        ("claude-opus-5", "claude", "/claude/v1/messages", "claude-opus-5"),
        ("gemini-3-6-flash", "openai_chat", "/gemini-3-6-flash-openai/v1/chat/completions", "gemini-3-6-flash-openai"),
        ("gpt-5-6-sol", "codex_responses", "/codex/v1/responses", "gpt-5-6-sol"),
        ("gpt-5-6-terra", "codex_responses", "/codex/v1/responses", "gpt-5-6-terra"),
        ("gpt-5-5", "codex_responses", "/codex/v1/responses", "gpt-5-5"),
        ("gpt-5.1-codex", "gpt_codex", "/api/v1/responses", "gpt-5.1-codex"),
        ("gpt-5-codex", "gpt_codex", "/api/v1/responses", "gpt-5-codex"),
        ("claude-haiku-4-5", "claude", "/claude/v1/messages", "claude-haiku-4-5"),
        ("gemini-3-8-flash-openai", "openai_chat", "/gemini-3-8-flash-openai/v1/chat/completions", "gemini-3-8-flash-openai"),
        ("gemini-3-8-flash", "openai_chat", "/gemini-3-8-flash-openai/v1/chat/completions", "gemini-3-8-flash-openai"),
        ("gemini-3-pro", "openai_chat", "/gemini-3-pro/v1/chat/completions", "gemini-3-pro"),
        ("gemini-3-pro-openai", "openai_chat", "/gemini-3-pro/v1/chat/completions", "gemini-3-pro"),
        ("grok-4-6", "grok_responses", "/grok/v1/responses", "grok-4-6"),
        ("gpt-5-2", "openai_chat", "/gpt-5-2/v1/chat/completions", "gpt-5-2"),
        ("gpt-5.2", "openai_chat", "/gpt-5-2/v1/chat/completions", "gpt-5-2"),
    ],
)
def test_resolve_kie_route_matches_docs(model, kind, path_suffix, canonical):
    route = resolve_kie_route(model)
    assert route.kind == kind
    assert route.path == path_suffix
    assert route.model == canonical


def _install_fake_client(monkeypatch, body: dict, captured: dict):
    class FakeResp:
        is_success = True
        status_code = 200
        text = "{}"
        headers = {}

        def json(self):
            return body

        @property
        def request(self):
            return httpx.Request("POST", captured.get("url") or "https://api.kie.ai/")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)


def _kie_cfg(name: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        provider="kie",
        base_url="https://api.kie.ai",
        api_key_env="KIE_API_KEY",
    )


@pytest.mark.asyncio
async def test_kie_gemini_posts_to_slug_chat_completions(monkeypatch):
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        {"choices": [{"message": {"content": "gemini-ok"}}]},
        captured,
    )
    text = await _invoke_provider(
        _kie_cfg("gemini-3-8-flash-openai"), "k" * 32, "hello", "sys", 0.3, 128, False, timeout=5
    )
    assert text == "gemini-ok"
    assert captured["url"] == "https://api.kie.ai/gemini-3-8-flash-openai/v1/chat/completions"
    assert captured["json"]["stream"] is False
    assert captured["json"]["messages"][0]["role"] == "system"
    assert captured["json"]["messages"][1]["content"] == "hello"


@pytest.mark.asyncio
async def test_kie_claude_sets_stream_false(monkeypatch):
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        {"content": [{"type": "text", "text": "haiku-ok"}], "stop_reason": "end_turn", "usage": {}},
        captured,
    )
    text = await _invoke_provider(
        _kie_cfg("claude-haiku-4-5"), "k" * 32, "hello", "sys", 0.3, 128, False, timeout=5
    )
    assert text == "haiku-ok"
    assert captured["url"] == "https://api.kie.ai/claude/v1/messages"
    assert captured["json"]["stream"] is False
    assert captured["json"]["model"] == "claude-haiku-4-5"


@pytest.mark.asyncio
async def test_kie_gpt_codex_uses_api_v1_responses(monkeypatch):
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        {"output": [{"content": [{"type": "output_text", "text": "codex-ok"}]}]},
        captured,
    )
    text = await _invoke_provider(
        _kie_cfg("gpt-5.1-codex"), "k" * 32, "hello", "sys", 0.3, 128, False, timeout=5
    )
    assert text == "codex-ok"
    assert captured["url"] == "https://api.kie.ai/api/v1/responses"
    assert captured["json"]["stream"] is False
    assert captured["json"]["input"][1]["content"][0]["type"] == "input_text"


@pytest.mark.asyncio
async def test_kie_grok_uses_grok_responses(monkeypatch):
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        {"output": [{"content": [{"type": "output_text", "text": "grok-ok"}]}]},
        captured,
    )
    text = await _invoke_provider(
        _kie_cfg("grok-4-6"), "k" * 32, "hello", "", 0.3, 128, False, timeout=5
    )
    assert text == "grok-ok"
    assert captured["url"] == "https://api.kie.ai/grok/v1/responses"
    assert captured["json"]["stream"] is False


@pytest.mark.asyncio
async def test_kie_gpt52_uses_slug_chat_completions(monkeypatch):
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        {"choices": [{"message": {"content": "gpt52-ok"}}]},
        captured,
    )
    text = await _invoke_provider(
        _kie_cfg("gpt-5-2"), "k" * 32, "hello", "", 0.3, 128, False, timeout=5
    )
    assert text == "gpt52-ok"
    assert captured["url"] == "https://api.kie.ai/gpt-5-2/v1/chat/completions"
    assert captured["json"]["stream"] is False


def test_build_provider_chain_expands_kie_fallback_models(monkeypatch):
    monkeypatch.setenv("OMNIROUTE_API_KEY", "k" * 32)
    monkeypatch.setenv("KIE_API_KEY", "k" * 32)
    monkeypatch.setenv("KIE_MODEL", "gpt-5-6-luna")
    monkeypatch.setenv("KIE_FALLBACK_MODELS", "gemini-3-8-flash-openai,claude-haiku-4-5")
    monkeypatch.delenv("KIE_OPUS_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    chain = build_provider_chain("persona_analysis")
    kie_names = [cfg.name for _, cfg in chain if cfg.provider == "kie"]
    assert kie_names == ["gpt-5-6-luna", "gemini-3-8-flash-openai", "claude-haiku-4-5"]
    assert chain[0][1].provider == "omniroute"
    assert chain[0][1].name == "auto/fast"


def test_build_provider_chain_skips_duplicate_kie_fallback(monkeypatch):
    monkeypatch.setenv("OMNIROUTE_API_KEY", "k" * 32)
    monkeypatch.setenv("KIE_API_KEY", "k" * 32)
    monkeypatch.setenv("KIE_MODEL", "gpt-5-6-luna")
    monkeypatch.setenv("KIE_FALLBACK_MODELS", "gpt-5-6-luna,gemini-3-8-flash-openai")
    monkeypatch.delenv("KIE_OPUS_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    chain = build_provider_chain("general")
    kie_names = [cfg.name for _, cfg in chain if cfg.provider == "kie"]
    assert kie_names == ["gpt-5-6-luna", "gemini-3-8-flash-openai"]


@pytest.mark.asyncio
async def test_kie_gemini_alias_sends_canonical_model_in_body(monkeypatch):
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        {"choices": [{"message": {"content": "alias-ok"}}]},
        captured,
    )
    text = await _invoke_provider(
        _kie_cfg("gemini-3-8-flash"), "k" * 32, "hello", "", 0.3, 128, False, timeout=5
    )
    assert text == "alias-ok"
    assert captured["url"] == "https://api.kie.ai/gemini-3-8-flash-openai/v1/chat/completions"
    assert captured["json"]["model"] == "gemini-3-8-flash-openai"
    assert captured["json"]["temperature"] == 0.3


@pytest.mark.asyncio
async def test_kie_empty_responses_fails_closed(monkeypatch):
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        {"output": [{"type": "reasoning", "summary": []}], "status": "completed"},
        captured,
    )
    with pytest.raises(ValueError, match="no output text"):
        await _invoke_provider(
            _kie_cfg("gpt-5-6-luna"), "k" * 32, "hello", "sys", 0.3, 128, False, timeout=5
        )


def test_sanitize_remaps_kie_family_ids_on_omniroute(monkeypatch):
    monkeypatch.delenv("OMNIROUTE_DEFAULT_MODEL", raising=False)
    for name in ("grok-4-6", "gemini-3-pro", "gpt-5.1-codex", "claude-sonnet-4-6"):
        cfg = ModelConfig(name=name, provider="omniroute", api_key_env="OMNIROUTE_API_KEY")
        out = sanitize_provider_config(cfg, "persona_analysis")
        assert out is not None
        assert out.name == "auto/fast"


def test_sanitize_skips_invalid_kie_slug():
    cfg = ModelConfig(name="gemini-bad slug?", provider="kie", api_key_env="KIE_API_KEY")
    assert sanitize_provider_config(cfg, "persona_analysis") is None


