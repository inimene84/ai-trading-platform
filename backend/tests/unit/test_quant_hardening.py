"""Unit tests for Postmortem Failure Logger, Regime Circuit Breaker, Sizing Override, and Shadow Replay."""
import asyncio
import importlib.util
import types
from pathlib import Path
from unittest.mock import AsyncMock, patch

from backend.services.postmortem import (
    PostmortemPoller,
    _supabase_upsert,
    build_postmortem,
    classify_cause,
    normalize_trade,
    on_trade_closed,
)
from backend.services.regime_circuit_breaker import RegimeCircuitBreaker
from backend.services.sizing_override import SizingOverrideConfig, compute_sizing_override, query_recent_streak_and_drawdown
from backend.services.supabase_admin import service_role_credentials, service_role_key

_SHADOW_REPLAY_PATH = Path(__file__).resolve().parents[3] / "scripts" / "shadow_replay.py"
_sr_spec = importlib.util.spec_from_file_location("shadow_replay_mod", _SHADOW_REPLAY_PATH)
assert _sr_spec and _sr_spec.loader
_sr_mod = importlib.util.module_from_spec(_sr_spec)
_sr_spec.loader.exec_module(_sr_mod)
_is_binance_perp_symbol = _sr_mod._is_binance_perp_symbol
_normalize_original_direction = _sr_mod._normalize_original_direction
build_replay_prompt = _sr_mod.build_replay_prompt
score_direction = _sr_mod.score_direction




# ── Postmortem tests ──────────────────────────────────────────────────────────


def test_postmortem_normalize_orm_trade_and_stop_loss_classification():
    orm_trade = types.SimpleNamespace(
        id=42,
        symbol="SOLUSDT",
        direction="BUY",
        broker="binance_futures",
        mode="paper",
        timestamp="2026-10-18T07:00:00+00:00",
        closed_at="2026-10-18T08:00:00+00:00",
        entry_price=150.0,
        exit_price=147.0,
        stop_loss=147.0,
        take_profit=158.0,
        quantity=1.0,
        pnl=-3.12,
        strategy="combined",
        notes="SL hit on pullback",
        signal_id=10,
        broker_metadata={"commission": 0.12},
    )
    norm = normalize_trade(orm_trade)
    assert norm["trade_id"] == "binance_futures:paper:42"
    assert norm["realized_pnl"] == -3.12
    assert norm["exit_reason"] == "stop_loss"
    assert classify_cause(orm_trade) == "stop_placement"


def test_postmortem_classifications_fee_churn_regime_shift_model_error_winner():
    assert classify_cause({"trade_id": "1", "symbol": "BTCUSDT", "direction": "BUY", "pnl": -0.20, "total_fees": 0.30}) == "fee_churn"
    assert classify_cause({"trade_id": "2", "symbol": "BTCUSDT", "direction": "BUY", "pnl": 12.5, "total_fees": 0.2}) == "winner"
    assert (
        classify_cause(
            {
                "trade_id": "3",
                "symbol": "ETHUSDT",
                "direction": "BUY",
                "pnl": -8.0,
                "regime_at_entry": "TRENDING",
                "regime_at_exit": "RANGING",
            }
        )
        == "regime_shift"
    )
    assert (
        classify_cause(
            {
                "trade_id": "4",
                "symbol": "ETHUSDT",
                "direction": "SELL",
                "pnl": -6.0,
                "exit_reason": "manual",
                "llm_confidence": 0.82,
            }
        )
        == "model_error"
    )


def test_on_trade_closed_builds_and_embeds_without_network():
    async def _fake_llm(*_a, **_kw):
        return "Stop was placed inside 1x ATR noise band; widen stop in volatile regime.", {"model": "mock-fast"}

    with (
        patch("backend.llm.router.call_llm_resilient_meta", _fake_llm),
        patch("backend.services.postmortem._supabase_upsert", AsyncMock(return_value=True)),
        patch("backend.services.postmortem.embed_to_finmem", AsyncMock(return_value=True)),
    ):
        pm = asyncio.run(
            on_trade_closed(
                {
                    "id": 99,
                    "symbol": "BTCUSDT",
                    "direction": "BUY",
                    "broker": "binance_futures",
                    "mode": "paper",
                    "pnl": -15.0,
                    "exit_reason": "stop_loss",
                }
            )
        )
        assert pm is not None
        assert pm["cause"] == "stop_placement"
        assert "widen stop" in (pm["reflection"] or "")
        assert pm["reflection_model"] == "mock-fast"


# ── Regime Circuit Breaker tests ──────────────────────────────────────────────


def test_regime_cb_vol_shock_trips_and_recovers():
    cb = RegimeCircuitBreaker(mode="enforce", cooldown_seconds=10, recovery_cycles=2, atr_jump_ratio=1.5)
    r1 = cb.observe("BTCUSDT", {"regime": "TRENDING", "atr_ratio": 0.010}, now=100.0)
    assert r1.state == "CLOSED"
    assert r1.allow_entry is True

    # Sudden jump into VOLATILE with 2x ATR expansion -> OPEN
    r2 = cb.observe("BTCUSDT", {"regime": "VOLATILE", "atr_ratio": 0.022}, now=101.0)
    assert r2.state == "OPEN"
    assert r2.allow_entry is False
    assert r2.size_factor == 0.0

    # Cooldown expires at t=112 -> HALF_OPEN (50% size)
    r3 = cb.observe("BTCUSDT", {"regime": "VOLATILE", "atr_ratio": 0.020}, now=112.0)
    assert r3.state == "HALF_OPEN"
    assert r3.allow_entry is True
    assert r3.size_factor == 0.50

    # Second stable observation in HALF_OPEN -> CLOSED
    r4 = cb.observe("BTCUSDT", {"regime": "VOLATILE", "atr_ratio": 0.019}, now=115.0)
    assert r4.state == "CLOSED"
    assert r4.size_factor == 1.0


def test_regime_cb_flip_churn_and_market_wide_shift():
    cb = RegimeCircuitBreaker(mode="shadow", flip_threshold=3, window_size=6, universe_min_symbols=4, universe_shift_pct=0.5)
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    for s in syms:
        cb.observe(s, {"regime": "TRENDING", "atr_ratio": 0.01}, now=10.0)
    cb.end_cycle(now=10.0)

    # 3 out of 4 symbols flip simultaneously (75% >= 50%)
    cb.observe("BTCUSDT", {"regime": "RANGING", "atr_ratio": 0.01}, now=20.0)
    cb.observe("ETHUSDT", {"regime": "RANGING", "atr_ratio": 0.01}, now=20.0)
    cb.observe("SOLUSDT", {"regime": "RANGING", "atr_ratio": 0.01}, now=20.0)
    cb.observe("BNBUSDT", {"regime": "TRENDING", "atr_ratio": 0.01}, now=20.0)
    summary = cb.end_cycle(now=20.0)
    assert summary["tripped_now"] is True
    assert summary["global_state"] == "OPEN"

    # In shadow mode, would_block is True but allow_entry remains True
    res = cb.observe("BNBUSDT", {"regime": "TRENDING", "atr_ratio": 0.01}, now=21.0)
    assert res.would_block is True
    assert res.allow_entry is True


# ── Dynamic Sizing Override tests ─────────────────────────────────────────────


def test_sizing_override_monotone_scale_down():
    cfg_enforce = SizingOverrideConfig(mode="enforce", min_multiplier=0.20)
    out = compute_sizing_override(
        1000.0,
        regime="VOLATILE",          # 0.50
        regime_confidence=0.40,     # 0.75
        cb_state="CLOSED",          # 1.00
        drawdown_pct=6.0,           # 0.75
        losing_streak=3,            # 0.75
        bear_strength=0.40,         # 0.80
        config=cfg_enforce,
    )
    assert 0.20 <= out.proposed_multiplier < 1.0
    assert out.effective_multiplier == out.proposed_multiplier
    assert out.adjusted_notional == 1000.0 * out.effective_multiplier

    # Shadow mode keeps effective_multiplier at 1.0 while logging proposed_multiplier
    cfg_shadow = SizingOverrideConfig(mode="shadow", min_multiplier=0.20)
    out_sh = compute_sizing_override(1000.0, regime="VOLATILE", config=cfg_shadow)
    assert out_sh.proposed_multiplier == 0.50
    assert out_sh.effective_multiplier == 1.0
    assert out_sh.adjusted_notional == 1000.0


def test_sizing_override_circuit_breaker_open_zeros_notional():
    cfg_enforce = SizingOverrideConfig(mode="enforce")
    out = compute_sizing_override(500.0, cb_state="OPEN", cb_mode="enforce", config=cfg_enforce)
    assert out.proposed_multiplier == 0.0
    assert out.adjusted_notional == 0.0


def test_shadow_sizing_skips_blocking_db_query(monkeypatch):
    monkeypatch.delenv("SIZING_OVERRIDE_MODE", raising=False)
    with patch("backend.database.connection.SessionLocal") as session_cls:
        streak, dd = query_recent_streak_and_drawdown(account_equity=1000.0)
    assert (streak, dd) == (0, 0.0)
    session_cls.assert_not_called()


def test_default_sizing_override_ignores_open_circuit_breaker(monkeypatch):
    """Default SIZING_OVERRIDE_MODE=shadow + REGIME_CB_MODE=shadow must not resize."""
    monkeypatch.delenv("SIZING_OVERRIDE_MODE", raising=False)
    monkeypatch.delenv("REGIME_CB_MODE", raising=False)
    out = compute_sizing_override(
        500.0,
        regime="VOLATILE",
        regime_confidence=0.10,
        cb_state="OPEN",
        drawdown_pct=15.0,
        losing_streak=8,
        bear_strength=1.0,
    )
    assert out.mode == "shadow"
    assert out.effective_multiplier == 1.0
    assert out.adjusted_notional == 500.0
    assert out.proposed_multiplier < 1.0
    # OPEN -> 0.0 is reserved for REGIME_CB_MODE=enforce
    assert out.components["circuit_breaker"] == 1.0


def test_default_regime_cb_open_does_not_block_or_resize(monkeypatch):
    monkeypatch.delenv("REGIME_CB_MODE", raising=False)
    cb = RegimeCircuitBreaker()
    assert cb.mode == "shadow"
    cb.observe("BTCUSDT", {"regime": "TRENDING", "atr_ratio": 0.010}, now=100.0)
    tripped = cb.observe("BTCUSDT", {"regime": "VOLATILE", "atr_ratio": 0.022}, now=101.0)
    assert tripped.state == "OPEN"
    assert tripped.would_block is True
    assert tripped.allow_entry is True
    assert tripped.size_factor == 1.0


def test_service_role_credentials_never_return_anon(monkeypatch):
    anon_jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFub24iLCJyb2xlIjoiYW5vbiJ9."
        "sig"
    )
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", anon_jwt)
    monkeypatch.setenv("SUPABASE_KEY", anon_jwt)
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", anon_jwt)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    assert service_role_key() == ""
    assert service_role_credentials() == ("", "")


def test_service_role_credentials_accept_service_jwt(monkeypatch):
    svc_jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJyb2xlIjoic2VydmljZV9yb2xlIn0."
        "sig"
    )
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", svc_jwt)
    monkeypatch.setenv("SUPABASE_ANON_KEY", "should-not-win")
    url, key = service_role_credentials()
    assert url == "https://example.supabase.co"
    assert key == svc_jwt


def test_postmortem_upsert_skips_without_service_role(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    with patch("backend.services.postmortem.httpx.AsyncClient") as client_cls:
        ok = asyncio.run(_supabase_upsert("postmortems", {"trade_id": "x"}))
    assert ok is False
    client_cls.assert_not_called()


def test_on_trade_closed_never_raises_without_network():
    async def _boom(*_a, **_kw):
        raise RuntimeError("llm/supabase/qdrant down")

    with (
        patch("backend.services.postmortem.build_postmortem", _boom),
        patch("backend.services.postmortem._supabase_upsert", _boom),
        patch("backend.services.postmortem.embed_to_finmem", _boom),
    ):
        assert asyncio.run(on_trade_closed({"id": 1, "symbol": "BTCUSDT"})) is None


def test_postmortem_poller_start_is_idempotent(monkeypatch):
    monkeypatch.setenv("POSTMORTEM_ENABLED", "true")
    monkeypatch.setenv("POSTMORTEM_POLL_INTERVAL_MIN", "60")

    async def _run():
        poller = PostmortemPoller()
        first = poller.start()
        second = poller.start()
        assert first is not None
        assert first is second
        assert not first.done()
        first.cancel()
        try:
            await first
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())


def test_postmortem_poller_sweep_swallows_db_errors():
    async def _run():
        poller = PostmortemPoller()
        with patch("backend.services.postmortem._fetch_closed_snapshots", side_effect=RuntimeError("no db")):
            out = await poller.sweep_recent_closed(limit=5)
        assert out["processed"] == 0
        assert out["skipped"] == 0

    asyncio.run(_run())


# ── Shadow Replay helper tests ────────────────────────────────────────────────


def test_shadow_replay_schema_helpers():
    assert _is_binance_perp_symbol("BTCUSDT") is True
    assert _is_binance_perp_symbol("SOLUSDC") is True
    assert _is_binance_perp_symbol("EURUSD") is False

    sig = {
        "id": "uuid-1",
        "symbol": "BTCUSDT",
        "timeframe": "M5",
        "decision": "BUY",
        "allowed_side": "BUY",
        "signal_strength": 0.72,
        "probabilities": {"baseline": 0.72, "with_jev": 0.78},
        "raw_answers": {"thesis": "breakout"},
        "created_at": "2026-10-17T12:00:00+00:00",
    }
    assert _normalize_original_direction(sig) == "up"
    prompt = build_replay_prompt(sig)
    assert "BTCUSDT" in prompt
    assert "breakout" in prompt

    assert score_direction("up", 1.25) is True
    assert score_direction("down", 1.25) is False
    assert score_direction("neutral", 1.25) is None
