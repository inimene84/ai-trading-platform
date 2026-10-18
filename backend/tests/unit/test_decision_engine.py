import types

import pytest
from unittest.mock import MagicMock, patch

from backend.services.decision_engine import (
    DecisionEngine,
    compute_sl_tp_levels,
    atr_from_bars,
    pyramid_price_improved,
)
from backend.services.risk_config import RiskConfig
from backend.strategies.base import StrategySignal


def _make_bars(n: int = 200, base: float = 100.0) -> list:
    """Synthetic OHLCV bars with gentle uptrend."""
    bars = []
    for i in range(n):
        c = base + i * 0.5
        bars.append({
            "open": c - 0.2,
            "high": c + 0.5,
            "low": c - 0.5,
            "close": c,
            "volume": 1000.0,
        })
    return bars


@pytest.fixture
def risk_config():
    return RiskConfig(
        min_signal_strength=0.45,
        ai_analysis_threshold=0.30,
        trade_usdt_amount=10.0,
        equity_sizing_enabled=True,
        risk_per_trade_pct=0.01,
        max_trade_notional_equity_mult=2.0,
        min_edge_fee_mult=0.0,  # disable min-edge gate for sizing tests
        sl_atr_mult=1.5,
        tp_atr_mult=3.0,
        enable_jesse_ml=False,
    )


def test_compute_sl_tp_buy_direction(risk_config):
    bars = _make_bars(20, base=100.0)
    sl, tp = compute_sl_tp_levels(bars, "BUY", 100.0, risk_config)
    assert sl < 100.0
    assert tp > 100.0


def test_compute_sl_tp_sell_direction(risk_config):
    bars = _make_bars(20, base=100.0)
    sl, tp = compute_sl_tp_levels(bars, "SELL", 100.0, risk_config)
    assert sl > 100.0
    assert tp < 100.0


def test_compute_sl_tp_zero_signal_sl_falls_back_to_atr(risk_config):
    """A 0.0 strategy stop used to survive min() and then get stripped at
    Binance as 'missing', leaving a naked long. Treat 0 as 'no signal stop'."""
    bars = _make_bars(20, base=100.0)
    sl, tp = compute_sl_tp_levels(bars, "BUY", 100.0, risk_config, signal_sl=0.0, signal_tp=120.0)
    assert sl > 0
    assert sl < 100.0
    assert tp == 120.0


def test_atr_from_bars_handles_short_series():
    bars = _make_bars(15, base=100.0)
    atr = atr_from_bars(bars, 100.0)
    assert atr > 0


def test_short_pyramid_requires_price_decline():
    assert pyramid_price_improved("SELL", 99.4, 100.0, 0.005)
    assert not pyramid_price_improved("SELL", 100.0, 100.0, 0.005)
    assert not pyramid_price_improved("SELL", 101.0, 100.0, 0.005)


def test_create_entry_floors_notional_at_20_usdt(risk_config):
    engine = DecisionEngine(risk_config)
    engine.account_equity = 50.0  # tiny equity → risk sizing would be < $20
    bars = _make_bars(200, base=50000.0)  # expensive symbol
    signal = StrategySignal(
        symbol="ETHUSDT", signal="BUY", confidence=0.8,
        entry_price=bars[-1]["close"], strategy="trend_following",
    )
    decision = engine._create_entry_decision("ETHUSDT", bars, signal, "BUY", is_pyramid=False)
    assert decision is not None
    notional = decision.quantity * decision.entry_price
    assert notional >= 20.0


def test_create_entry_floors_btc_at_100_usdt(risk_config):
    engine = DecisionEngine(risk_config)
    engine.account_equity = 80.0
    bars = _make_bars(200, base=60000.0)
    signal = StrategySignal(
        symbol="BTCUSDT", signal="BUY", confidence=0.8,
        entry_price=bars[-1]["close"], strategy="trend_following",
    )
    decision = engine._create_entry_decision("BTCUSDT", bars, signal, "BUY", is_pyramid=False)
    assert decision is not None
    notional = decision.quantity * decision.entry_price
    assert notional >= 100.0


def test_equity_sizing_scales_with_account(risk_config):
    engine = DecisionEngine(risk_config)
    bars = _make_bars(200, base=2000.0)
    signal = StrategySignal(
        symbol="ETHUSDT", signal="BUY", confidence=0.8,
        entry_price=bars[-1]["close"], strategy="trend_following",
    )

    engine.account_equity = 100.0
    small = engine._create_entry_decision("ETHUSDT", bars, signal, "BUY", is_pyramid=False)

    engine.account_equity = 500.0
    large = engine._create_entry_decision("ETHUSDT", bars, signal, "BUY", is_pyramid=False)

    assert small is not None and large is not None
    assert large.quantity > small.quantity


@pytest.mark.asyncio
async def test_evaluate_symbol_insufficient_bars(risk_config):
    engine = DecisionEngine(risk_config)
    result = await engine.evaluate_symbol("ETHUSDT", _make_bars(10), None, 0, [], False)
    assert result is None
    assert engine.last_evaluation["reason"] == "insufficient bars"


@pytest.mark.asyncio
async def test_evaluate_symbol_below_threshold(risk_config):
    engine = DecisionEngine(risk_config)
    bars = _make_bars(200)

    weak_signal = StrategySignal(
        symbol="ETHUSDT", signal="BUY", confidence=0.20,
        entry_price=bars[-1]["close"], strategy="trend_following",
    )
    engine.strategy.generate_signal = MagicMock(return_value=weak_signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(
        regime="TRENDING", weights=MagicMock(return_value={})
    ))

    result = await engine.evaluate_symbol("ETHUSDT", bars, None, 0, [], False)
    assert result is None
    assert "below threshold" in engine.last_evaluation["reason"]


@pytest.mark.asyncio
async def test_evaluate_symbol_skips_when_cooldown_active(risk_config):
    engine = DecisionEngine(risk_config)
    bars = _make_bars(200)
    result = await engine.evaluate_symbol("ETHUSDT", bars, None, 0, [], cooldown_active=True)
    assert result is None

@pytest.mark.asyncio
async def test_evaluate_symbol_funding_rate_adjustments(risk_config):
    engine = DecisionEngine(risk_config)
    engine.enable_kronos = False
    engine.config.enable_personas = False
    engine.config.use_risk_reviewer_llm = False
    bars = _make_bars(200)

    # SELL signal with positive funding boosts confidence, but the boost is
    # clamped to funding_conf_adj_cap (default 0.05): a routine 0.01% funding
    # must nudge, not flip, a signal (raw adj would be +0.10 here).
    signal = StrategySignal(
        symbol="ETHUSDT", signal="SELL", confidence=0.42,
        entry_price=bars[-1]["close"], strategy="trend_following",
    )
    engine.strategy.generate_signal = MagicMock(return_value=signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(
        regime="TRENDING", weights=MagicMock(return_value={})
    ))

    result = await engine.evaluate_symbol("ETHUSDT", bars, None, 0, [], False, current_funding_rate=0.0001)
    assert result is not None
    assert result.confidence == pytest.approx(0.47)  # 0.42 + capped 0.05


@pytest.mark.asyncio
async def test_funding_adjustment_clamped_for_extreme_rates(risk_config):
    """A huge funding rate must not add more than funding_conf_adj_cap."""
    engine = DecisionEngine(risk_config)
    engine.enable_kronos = False
    engine.config.enable_personas = False
    engine.config.use_risk_reviewer_llm = False
    bars = _make_bars(200)

    signal = StrategySignal(
        symbol="ETHUSDT", signal="SELL", confidence=0.50,
        entry_price=bars[-1]["close"], strategy="trend_following",
    )
    engine.strategy.generate_signal = MagicMock(return_value=signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(
        regime="TRENDING", weights=MagicMock(return_value={})
    ))

    # 0.1% funding -> raw adjustment would be +1.0 (saturating confidence)
    result = await engine.evaluate_symbol("ETHUSDT", bars, None, 0, [], False, current_funding_rate=0.001)
    assert result is not None
    assert result.confidence == pytest.approx(0.55)  # 0.50 + capped 0.05

@pytest.mark.asyncio
async def test_ranging_regime_sizing_and_gate(risk_config):
    engine = DecisionEngine(risk_config)
    engine.enable_kronos = False
    engine.config.enable_personas = False
    engine.config.use_risk_reviewer_llm = False
    engine.config.trade_usdt_amount = 100.0
    bars = _make_bars(200)
    
    signal = StrategySignal(
        symbol="ETHUSDT", signal="BUY", confidence=0.50,
        entry_price=bars[-1]["close"], strategy="mean_reversion",
    )
    engine.strategy.generate_signal = MagicMock(return_value=signal)
    
    # 1. RANGING regime check: should fail because confidence 0.50 < 0.60 required (0.45 + 0.15)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(
        regime="RANGING", weights=MagicMock(return_value={})
    ))
    result_ranging = await engine.evaluate_symbol("ETHUSDT", bars, None, 0, [], False)
    assert result_ranging is None
    
    # 2. TRENDING regime check: should pass because confidence 0.50 >= 0.45
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(
        regime="TRENDING", weights=MagicMock(return_value={})
    ))
    result_trending = await engine.evaluate_symbol("ETHUSDT", bars, None, 0, [], False)
    assert result_trending is not None
    
    # 3. Block new entries in RANGING regime (as per design in commit af175970)
    decision_trending = engine._create_entry_decision("ETHUSDT", bars, signal, "BUY", is_pyramid=False, regime="TRENDING")
    decision_ranging = engine._create_entry_decision("ETHUSDT", bars, signal, "BUY", is_pyramid=False, regime="RANGING")
    
    assert decision_trending is not None
    assert decision_ranging is None


@pytest.mark.asyncio
async def test_funding_gate_records_buy_not_hold(risk_config):
    """HOLD direction made funding blocks invisible to the shadow tracker."""
    engine = DecisionEngine(risk_config)
    engine.enable_kronos = False
    engine.config.enable_personas = False
    engine.config.use_risk_reviewer_llm = False
    engine.config.funding_rate_cap = 0.0001
    bars = _make_bars(200)
    signal = StrategySignal(
        symbol="ETHUSDT", signal="BUY", confidence=0.70,
        entry_price=bars[-1]["close"], strategy="trend_following",
    )
    engine.strategy.generate_signal = MagicMock(return_value=signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(
        regime="TRENDING", weights=MagicMock(return_value={})
    ))
    result = await engine.evaluate_symbol(
        "ETHUSDT", bars, None, 0, [], False, current_funding_rate=0.001
    )
    assert result is None
    assert engine.last_evaluation["direction"] == "BUY"
    assert "funding" in engine.last_evaluation["reason"]


def test_min_edge_reject_records_reason(risk_config):
    engine = DecisionEngine(risk_config)
    engine._passes_min_edge = lambda *a, **k: False
    bars = _make_bars(50)
    signal = StrategySignal(
        symbol="ETHUSDT", signal="BUY", confidence=0.70,
        entry_price=100.0, strategy="trend_following",
    )
    decision = engine._create_entry_decision(
        "ETHUSDT", bars, signal, "BUY", is_pyramid=False, regime="TRENDING"
    )
    assert decision is None
    assert engine.last_evaluation.get("direction") == "BUY"
    assert "min-edge" in engine.last_evaluation.get("reason", "")


@pytest.mark.asyncio
async def test_ranging_flag_off_still_blocks_mean_reversion(risk_config):
    engine = DecisionEngine(risk_config)
    engine.enable_kronos = False
    engine.config.enable_personas = False
    engine.config.use_risk_reviewer_llm = False
    engine.config.allow_ranging_entries = False
    bars = _make_bars(200)
    signal = StrategySignal(
        symbol="ETHUSDT", signal="BUY", confidence=0.80,
        entry_price=bars[-1]["close"], strategy="mean_reversion",
        reasoning="STRONG[RANGING] mean_reversion: BUY(0.80)",
    )
    engine.strategy.generate_signal = MagicMock(return_value=signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(
        regime="RANGING", weights=MagicMock(return_value={})
    ))
    result = await engine.evaluate_symbol("ETHUSDT", bars, None, 0, [], False)
    assert result is None
    assert "RANGING" in engine.last_evaluation["reason"]


@pytest.mark.asyncio
async def test_ranging_flag_on_allows_mean_reversion(risk_config, monkeypatch):
    from backend.services.trading_mode import TradingMode

    monkeypatch.setattr(
        "backend.services.decision_engine.get_trading_mode",
        lambda: TradingMode.PAPER,
    )
    engine = DecisionEngine(risk_config)
    engine.enable_kronos = False
    engine.config.enable_personas = False
    engine.config.use_risk_reviewer_llm = False
    engine.config.allow_ranging_entries = True
    engine.config.allow_ranging_in_live = False
    bars = _make_bars(200)
    signal = StrategySignal(
        symbol="ETHUSDT", signal="BUY", confidence=0.80,
        entry_price=bars[-1]["close"], strategy="mean_reversion",
        reasoning="STRONG[RANGING] mean_reversion: BUY(0.80)",
    )
    engine.strategy.generate_signal = MagicMock(return_value=signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(
        regime="RANGING", weights=MagicMock(return_value={})
    ))
    result = await engine.evaluate_symbol("ETHUSDT", bars, None, 0, [], False)
    assert result is not None
    assert result.action == "BUY"


@pytest.mark.asyncio
async def test_ranging_flag_on_still_applies_confidence_gate(risk_config, monkeypatch):
    from backend.services.trading_mode import TradingMode

    monkeypatch.setattr(
        "backend.services.decision_engine.get_trading_mode",
        lambda: TradingMode.PAPER,
    )
    engine = DecisionEngine(risk_config)
    engine.enable_kronos = False
    engine.config.enable_personas = False
    engine.config.use_risk_reviewer_llm = False
    engine.config.allow_ranging_entries = True
    bars = _make_bars(200)
    # 0.50 < 0.45+0.15 ranging gate
    signal = StrategySignal(
        symbol="ETHUSDT", signal="BUY", confidence=0.50,
        entry_price=bars[-1]["close"], strategy="mean_reversion",
    )
    engine.strategy.generate_signal = MagicMock(return_value=signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(
        regime="RANGING", weights=MagicMock(return_value={})
    ))
    result = await engine.evaluate_symbol("ETHUSDT", bars, None, 0, [], False)
    assert result is None
    assert "below threshold" in engine.last_evaluation["reason"]


@pytest.mark.asyncio
async def test_ranging_flag_on_rejects_non_matching_setup(risk_config, monkeypatch):
    from backend.services.trading_mode import TradingMode

    monkeypatch.setattr(
        "backend.services.decision_engine.get_trading_mode",
        lambda: TradingMode.PAPER,
    )
    engine = DecisionEngine(risk_config)
    engine.enable_kronos = False
    engine.config.enable_personas = False
    engine.config.use_risk_reviewer_llm = False
    engine.config.allow_ranging_entries = True
    bars = _make_bars(200)
    signal = StrategySignal(
        symbol="ETHUSDT", signal="BUY", confidence=0.80,
        entry_price=bars[-1]["close"], strategy="trend_following",
        reasoning="STRONG[RANGING] trend_following: BUY(0.80)",
    )
    engine.strategy.generate_signal = MagicMock(return_value=signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(
        regime="RANGING", weights=MagicMock(return_value={})
    ))
    engine._ranging_skill_match = MagicMock(return_value=False)
    result = await engine.evaluate_symbol("ETHUSDT", bars, None, 0, [], False)
    assert result is None
    assert "RANGING" in engine.last_evaluation["reason"]


def test_ranging_live_stays_blocked_without_live_flag(risk_config, monkeypatch):
    from backend.services.trading_mode import TradingMode

    monkeypatch.setattr(
        "backend.services.decision_engine.get_trading_mode",
        lambda: TradingMode.LIVE,
    )
    engine = DecisionEngine(risk_config)
    engine.config.allow_ranging_entries = True
    engine.config.allow_ranging_in_live = False
    bars = _make_bars(200)
    signal = StrategySignal(
        symbol="ETHUSDT", signal="BUY", confidence=0.80,
        entry_price=bars[-1]["close"], strategy="mean_reversion",
    )
    decision = engine._create_entry_decision(
        "ETHUSDT", bars, signal, "BUY", is_pyramid=False, regime="RANGING",
    )
    assert decision is None


def test_ranging_skill_match_allows_positive_edge_ranging_skill(risk_config, monkeypatch):
    from backend.services.trading_mode import TradingMode

    monkeypatch.setattr(
        "backend.services.decision_engine.get_trading_mode",
        lambda: TradingMode.PAPER,
    )
    engine = DecisionEngine(risk_config)
    engine.config.allow_ranging_entries = True
    monkeypatch.setattr(
        "backend.services.decision_engine.skill_miner.match_skill",
        lambda ctx: {
            "name": "Ranging → Bullish",
            "direction": "bullish",
            "edge_score": 0.4,
            "avg_pnl": 1.2,
        },
    )
    bars = _make_bars(200)
    signal = StrategySignal(
        symbol="ETHUSDT", signal="BUY", confidence=0.80,
        entry_price=bars[-1]["close"], strategy="combined",
        reasoning="CONSENSUS[RANGING] trend:BUY",
    )
    decision = engine._create_entry_decision(
        "ETHUSDT", bars, signal, "BUY", is_pyramid=False, regime="RANGING",
    )
    assert decision is not None


def test_regime_detector_injection_reuses_history(risk_config):
    """A fresh detector per cycle would never accumulate 5-bar smoothing history."""
    from backend.strategies.market_regime import MarketRegimeDetector

    shared = MarketRegimeDetector()
    engine_a = DecisionEngine(risk_config, regime_detector=shared)
    engine_b = DecisionEngine(risk_config, regime_detector=shared)
    assert engine_a.regime_detector is shared
    assert engine_b.regime_detector is shared

    bars = _make_bars(250)
    engine_a.regime_detector.detect(bars)
    engine_b.regime_detector.detect(bars)
    assert len(shared._history) >= 2


def test_new_decision_engine_without_inject_does_not_share_history(risk_config):
    from backend.strategies.market_regime import MarketRegimeDetector

    a = DecisionEngine(risk_config)
    b = DecisionEngine(risk_config)
    assert a.regime_detector is not b.regime_detector
    assert isinstance(a.regime_detector, MarketRegimeDetector)


def test_affordable_notional_caps_tight_stop_on_1x_paper():
    from backend.services.decision_engine import affordable_notional

    # 1% risk / 0.93% stop → $107,549 notional on a $100k 1x book.
    raw = 100_000.0 * 0.01 / (2451.0 - 2428.21) * 2451.0
    assert raw > 107_000
    cap = affordable_notional(
        equity=100_000.0,
        available=100_000.0,
        leverage=1.0,
        max_equity_mult=2.0,
        max_margin_use_pct=0.95,
    )
    assert cap == pytest.approx(95_000.0)
    assert cap < raw


def test_affordable_notional_allows_risk_size_at_10x():
    from backend.services.decision_engine import affordable_notional

    raw = 100_000.0 * 0.01 / (2451.0 - 2428.21) * 2451.0
    cap = affordable_notional(
        equity=100_000.0,
        available=100_000.0,
        leverage=10.0,
        max_equity_mult=2.0,
        max_margin_use_pct=0.95,
    )
    assert cap >= raw


def test_eth_tight_stop_does_not_exceed_paper_cash(risk_config):
    """Dashboard skip: need $107k margin, have $100k at 1x paper leverage."""
    engine = DecisionEngine(risk_config)
    engine.account_equity = 100_000.0
    engine.account_available = 100_000.0
    engine.account_leverage = 1.0
    bars = _make_bars(200, base=2451.0)
    signal = StrategySignal(
        symbol="ETHUSDT", signal="BUY", confidence=0.65,
        entry_price=2451.0, stop_loss=2428.21, take_profit=2496.579,
        strategy="combined",
    )
    decision = engine._create_entry_decision(
        "ETHUSDT", bars, signal, "BUY", is_pyramid=False, regime="TRENDING",
    )
    assert decision is not None
    required = decision.quantity * decision.entry_price / engine.account_leverage
    assert required <= 100_000.0 * 0.95 + 1e-6


def test_paper_leverage_for_binance_defaults_to_ten(monkeypatch):
    from backend.services.trading_mode import paper_leverage_for_broker

    monkeypatch.delenv("BINANCE_LEVERAGE", raising=False)
    assert paper_leverage_for_broker("binance_futures") == 10.0
    monkeypatch.setenv("BINANCE_LEVERAGE", "5")
    assert paper_leverage_for_broker("binance_futures") == 5.0


def test_apply_hard_notional_caps_live_clips_to_trade_usdt_times_leverage():
    from backend.services.decision_engine import apply_hard_notional_caps

    capped, reject = apply_hard_notional_caps(
        480_000.0,
        trade_usdt=35.0,
        max_directional_usdt=1000.0,
        leverage=10.0,
        live=True,
        exchange_min_notional=20.0,
    )
    assert reject is None
    assert capped == pytest.approx(350.0)


def test_apply_hard_notional_caps_raises_cap_to_exchange_min():
    from backend.services.decision_engine import apply_hard_notional_caps

    capped, reject = apply_hard_notional_caps(
        500.0,
        trade_usdt=5.0,
        max_directional_usdt=10.0,
        leverage=1.0,
        live=True,
        exchange_min_notional=20.0,
    )
    assert reject is None
    assert capped == pytest.approx(20.0)


def test_live_entry_cannot_exceed_hard_notional_cap(risk_config, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("PAPER_TRADING", "false")
    monkeypatch.setenv("DRY_RUN_ALL", "false")
    engine = DecisionEngine(risk_config)
    engine.config.trade_usdt_amount = 35.0
    engine.config.max_directional_exposure_usdt = 1000.0
    engine.config.equity_sizing_enabled = True
    engine.account_equity = 104_800.0
    engine.account_available = 104_800.0
    engine.account_leverage = 10.0
    bars = _make_bars(200, base=1.50)
    signal = StrategySignal(
        symbol="OPUSDT", signal="BUY", confidence=0.7,
        entry_price=1.50, stop_loss=1.49, take_profit=1.55,
        strategy="combined",
    )
    decision = engine._create_entry_decision(
        "OPUSDT", bars, signal, "BUY", is_pyramid=False, regime="TRENDING",
    )
    assert decision is not None
    notional = decision.quantity * decision.entry_price
    assert notional <= 350.0 + 1e-6


def test_default_quant_hardening_does_not_resize_entry(risk_config, monkeypatch):
    """Default shadow/off flags must leave live entry size unchanged."""
    monkeypatch.delenv("SIZING_OVERRIDE_MODE", raising=False)
    monkeypatch.delenv("REGIME_CB_MODE", raising=False)
    monkeypatch.delenv("BEAR_VETO_MODE", raising=False)
    engine = DecisionEngine(risk_config)
    engine.account_equity = 500.0
    bars = _make_bars(200, base=2000.0)
    signal = StrategySignal(
        symbol="ETHUSDT", signal="BUY", confidence=0.8,
        entry_price=bars[-1]["close"], strategy="trend_following",
    )

    baseline = engine._create_entry_decision(
        "ETHUSDT", bars, signal, "BUY", is_pyramid=False, regime="TRENDING",
    )

    engine._last_regime_cb = types.SimpleNamespace(state="OPEN", mode="shadow", size_factor=1.0)
    engine._last_regime_result = types.SimpleNamespace(confidence=0.10, regime="VOLATILE")
    stressed = engine._create_entry_decision(
        "ETHUSDT", bars, signal, "BUY", is_pyramid=False, regime="VOLATILE",
    )

    assert baseline is not None and stressed is not None
    assert stressed.quantity == baseline.quantity
    assert stressed.entry_price == baseline.entry_price


@pytest.mark.asyncio
async def test_default_regime_cb_open_does_not_block_evaluate_symbol(risk_config, monkeypatch):
    monkeypatch.delenv("REGIME_CB_MODE", raising=False)
    monkeypatch.delenv("BEAR_VETO_MODE", raising=False)
    monkeypatch.delenv("SIZING_OVERRIDE_MODE", raising=False)
    engine = DecisionEngine(risk_config)
    engine.enable_kronos = False
    engine.config.enable_personas = False
    engine.config.use_risk_reviewer_llm = False
    bars = _make_bars(200)
    signal = StrategySignal(
        symbol="ETHUSDT", signal="SELL", confidence=0.55,
        entry_price=bars[-1]["close"], strategy="trend_following",
    )
    engine.strategy.generate_signal = MagicMock(return_value=signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(
        regime="TRENDING", weights=MagicMock(return_value={}),
        confidence=0.9, adx=30.0, atr_ratio=0.02,
    ))

    from backend.services.regime_circuit_breaker import RegimeCheckResult
    open_shadow = RegimeCheckResult(
        symbol="ETHUSDT",
        state="OPEN",
        mode="shadow",
        allow_entry=True,
        would_block=True,
        size_factor=1.0,
        reason="vol_shock_shift",
        regime="VOLATILE",
    )
    with patch("backend.services.regime_circuit_breaker.regime_circuit_breaker") as mock_cb:
        mock_cb.observe.return_value = open_shadow
        with patch("backend.agents.debate.bear_veto.schedule_shadow_bear_veto", return_value=None):
            result = await engine.evaluate_symbol("ETHUSDT", bars, None, 0, [], False)
    assert result is not None
    assert "regime circuit breaker" not in (engine.last_evaluation.get("reason") or "")

