"""Deterministic PortfolioRiskManager gates (Workstream 3)."""

from __future__ import annotations

import math

from backend.services.portfolio_risk_manager import (
    DRAWDOWN_SOFT_SIZE_MULT,
    FORWARD_GATE_NOTIONAL_USD,
    MAX_CONCURRENT_CRYPTO_POSITIONS,
    MAX_CRYPTO_CLUSTER_GROSS_WEIGHT,
    MAX_SYMBOL_WEIGHT,
    SHADOW_ONLY,
    SIZE_HALVED,
    TARGET_PORTFOLIO_VOL,
    OpenPosition,
    annualized_realized_vol,
    crypto_cluster_state,
    drawdown_state,
    evaluate_entry,
    forward_gate_status,
    funding_blocks_long,
    inverse_vol_notional,
    inverse_vol_weight,
    is_shadow_strategy,
    lookback_bars_for_timeframe,
    peak_equity_30d,
    periods_per_year_for_timeframe,
    prefer_post_only,
    rolling_drawdown_pct,
)


def _flat_closes(n: int = 480, price: float = 100.0) -> list[float]:
    return [price] * n


def _vol_closes(n: int = 481, start: float = 100.0, daily_move: float = 0.01) -> list[float]:
    """Alternating +/− moves so realized vol is non-zero and stable."""
    prices = [start]
    px = start
    sign = 1.0
    for _ in range(n - 1):
        px *= 1.0 + sign * daily_move
        prices.append(px)
        sign *= -1.0
    return prices


def test_forward_gate_status_reports_all_enforced_constants():
    status = forward_gate_status()
    assert status["active"] is True
    assert status["crypto_cluster"]["max_concurrent_positions"] == MAX_CONCURRENT_CRYPTO_POSITIONS
    assert status["crypto_cluster"]["max_gross_weight"] == MAX_CRYPTO_CLUSTER_GROSS_WEIGHT
    assert status["drawdown_breaker"]["soft_pct"] == 0.08
    assert status["drawdown_breaker"]["hard_pct"] == 0.12
    assert status["drawdown_breaker"]["hard_mode"] == SHADOW_ONLY
    assert status["funding_gate"]["long_block_rate"] == 0.0003
    assert status["forward_gate_notional_cap_usd"] == FORWARD_GATE_NOTIONAL_USD
    assert status["inverse_vol"]["lookback_1h_bars"] == 480
    assert status["inverse_vol"]["target_portfolio_vol"] == TARGET_PORTFOLIO_VOL
    assert status["inverse_vol"]["max_symbol_weight"] == MAX_SYMBOL_WEIGHT
    assert "TRADING_MODE" in status["mode_note"]


def test_inverse_vol_high_sigma_sizes_smaller_than_low_sigma():
    high = inverse_vol_weight(1.20)  # 0.12 / 1.20 = 0.10, under the 12.5% cap
    low = inverse_vol_weight(0.20)
    assert high < low
    assert math.isclose(high, TARGET_PORTFOLIO_VOL / 1.20, rel_tol=1e-9)
    assert low == MAX_SYMBOL_WEIGHT  # 0.12 / 0.20 = 0.60, capped at 0.125


def test_inverse_vol_notional_caps_at_12_5_percent_equity():
    equity = 10_000.0
    notional = inverse_vol_notional(equity, sigma=0.10)
    assert notional == equity * MAX_SYMBOL_WEIGHT
    assert notional == 1250.0


def test_annualized_vol_from_480_hourly_bars_is_positive():
    sigma = annualized_realized_vol(_vol_closes())
    assert sigma is not None
    assert sigma > 0
    assert annualized_realized_vol(_flat_closes(10)) is None


def test_crypto_cluster_blocks_fourth_distinct_perp():
    book = [
        OpenPosition("BTCUSDT", 100.0),
        OpenPosition("ETHUSDT", 100.0),
        OpenPosition("SOLUSDT", 100.0),
    ]
    fourth = crypto_cluster_state(book, new_symbol="LINKUSDT", new_notional=50.0, equity=10_000.0)
    assert fourth["count"] == 4
    assert fourth["count_ok"] is False
    assert fourth["ok"] is False

    add_existing = crypto_cluster_state(book, new_symbol="BTCUSDT", new_notional=50.0, equity=10_000.0)
    assert add_existing["count"] == 3
    assert add_existing["count_ok"] is True


def test_crypto_cluster_blocks_45_percent_gross_weight():
    book = [
        OpenPosition("BTCUSDT", 2000.0),
        OpenPosition("ETHUSDT", 2000.0),
    ]
    over = crypto_cluster_state(book, new_symbol="SOLUSDT", new_notional=600.0, equity=10_000.0)
    assert over["gross_weight"] == 0.46
    assert over["weight_ok"] is False

    under = crypto_cluster_state(book, new_symbol="SOLUSDT", new_notional=400.0, equity=10_000.0)
    assert under["gross_weight"] == 0.44
    assert under["ok"] is True


def test_drawdown_soft_halves_size_hard_trips_shadow_only():
    mode, mult, allow = drawdown_state(0.07)
    assert mode == "NORMAL" and mult == 1.0 and allow is True

    mode, mult, allow = drawdown_state(0.08)
    assert mode == SIZE_HALVED
    assert mult == DRAWDOWN_SOFT_SIZE_MULT
    assert allow is True

    mode, mult, allow = drawdown_state(0.12)
    assert mode == SHADOW_ONLY
    assert allow is False
    assert math.isclose(rolling_drawdown_pct(880.0, 1000.0), 0.12)
    assert peak_equity_30d([900.0, 1000.0, 950.0], 880.0) == 1000.0


def test_funding_gate_blocks_long_above_3bps():
    assert funding_blocks_long("BUY", 0.00031) is True
    assert funding_blocks_long("BUY", 0.0003) is False
    assert funding_blocks_long("SELL", 0.01) is False
    assert funding_blocks_long("BUY", None) is False


def test_shadow_gate_recovery_and_meta_labeler():
    assert is_shadow_strategy("Recovery 48") is True
    assert is_shadow_strategy("recovery_192") is True
    assert is_shadow_strategy("meta-labeler") is True
    assert is_shadow_strategy("jesse_ml") is True
    assert is_shadow_strategy("combined") is False
    assert is_shadow_strategy("QuantumAIStrategy") is False


def test_evaluate_entry_inv_vol_and_forward_cap(monkeypatch):
    monkeypatch.delenv("PRM_FORWARD_GATE_NOTIONAL_USD", raising=False)
    decision = evaluate_entry(
        symbol="BTCUSDT",
        side="BUY",
        equity=100_000.0,
        proposed_notional=5_000.0,
        hourly_closes=_vol_closes(),
        open_positions=[],
        peak_equity=100_000.0,
        funding_rate=0.0001,
        strategy="combined",
        timeframe="1h",
    )
    assert decision.allow is True
    assert decision.sized_notional <= FORWARD_GATE_NOTIONAL_USD
    assert decision.sized_notional <= 100_000.0 * MAX_SYMBOL_WEIGHT
    assert decision.mode != SHADOW_ONLY


def test_evaluate_entry_soft_drawdown_halves_notional():
    decision = evaluate_entry(
        symbol="ETHUSDT",
        side="BUY",
        equity=9_200.0,
        proposed_notional=100.0,
        hourly_closes=_vol_closes(),
        open_positions=[],
        rolling_drawdown_pct=0.08,
        funding_rate=0.0,
        strategy="combined",
    )
    assert decision.allow is True
    assert decision.mode == SIZE_HALVED
    assert decision.size_mult == 0.50
    assert decision.sized_notional == 50.0


def test_evaluate_entry_hard_drawdown_is_shadow_only():
    decision = evaluate_entry(
        symbol="ETHUSDT",
        side="BUY",
        equity=8_700.0,
        proposed_notional=100.0,
        rolling_drawdown_pct=0.12,
        funding_rate=0.0,
        strategy="combined",
    )
    assert decision.allow is False
    assert decision.mode == SHADOW_ONLY
    assert "hard kill" in decision.reasons[0]


def test_evaluate_entry_funding_and_cluster_and_shadow():
    funded = evaluate_entry(
        symbol="BTCUSDT",
        side="BUY",
        equity=10_000.0,
        proposed_notional=50.0,
        funding_rate=0.0004,
        strategy="combined",
    )
    assert funded.allow is False
    assert funded.gates["funding"]["blocks_long"] is True

    clustered = evaluate_entry(
        symbol="ADAUSDT",
        side="BUY",
        equity=10_000.0,
        proposed_notional=50.0,
        open_positions=[
            OpenPosition("BTCUSDT", 100.0),
            OpenPosition("ETHUSDT", 100.0),
            OpenPosition("SOLUSDT", 100.0),
        ],
        funding_rate=0.0,
        strategy="combined",
    )
    assert clustered.allow is False
    assert clustered.gates["crypto_cluster"]["count"] == 4

    shadow = evaluate_entry(
        symbol="BTCUSDT",
        side="BUY",
        equity=10_000.0,
        proposed_notional=50.0,
        funding_rate=0.0,
        strategy="Recovery 48",
    )
    assert shadow.allow is False
    assert shadow.mode == SHADOW_ONLY
    assert shadow.gates["shadow_strategy"]["shadow_only"] is True


def test_evaluate_entry_reduce_only_bypasses_gates():
    decision = evaluate_entry(
        symbol="BTCUSDT",
        side="SELL",
        equity=1.0,
        proposed_notional=500.0,
        funding_rate=0.01,
        strategy="Recovery 192",
        rolling_drawdown_pct=0.20,
        reduce_only=True,
    )
    assert decision.allow is True
    assert decision.reasons == ["reduce_only_bypass"]


def test_inv_vol_annualization_scales_with_timeframe():
    assert lookback_bars_for_timeframe("1h") == 480
    assert lookback_bars_for_timeframe("15m") == 1920
    assert periods_per_year_for_timeframe("15m") == 4 * periods_per_year_for_timeframe("1h")


def test_prefer_post_only_for_slow_non_urgent_bars():
    assert prefer_post_only(timeframe="4h") is True
    assert prefer_post_only(timeframe="12h") is True
    assert prefer_post_only(timeframe="1h") is False
    assert prefer_post_only(timeframe="4h", urgent=True) is False
    assert prefer_post_only(timeframe="4h", reduce_only=True) is False
