"""Regression tests for metal contract size, news mapping, and risk caps."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.ctrader_service import CTraderService
from backend.services.signal_candidate_engine import (
    CandidateStatus,
    signal_candidate_engine,
)


def test_units_per_lot_uses_ounces_for_metals_not_fx_units():
    assert CTraderService.units_per_lot("EURUSD") == 100_000
    assert CTraderService.units_per_lot("XAUUSD") == 100
    assert CTraderService.units_per_lot("XAGUSD") == 1_000


def test_silver_mark_to_market_is_not_fx_notional():
    """IC Markets 0.01 lot is 10 oz, not 100k FX units."""
    svc = CTraderService()
    svc._positions = [
        {
            "symbol": "XAGUSD",
            "side": "BUY",
            "quantity": 0.01,
            "entry_price": 67.092,
            "unrealized_pnl": 0.0,
            "position_id": "1",
        }
    ]
    svc._last_spots = {
        "XAGUSD": {"bid": 66.905, "ask": 66.905},
    }
    pos = svc.get_positions()[0]
    # 0.01 * 1000 oz * -0.187 = -1.87
    assert pos["unrealized_pnl"] == pytest.approx(-1.87, abs=0.05)
    assert pos["unrealized_pnl"] > -20


def test_gold_spec_lot_size_is_100_ounces():
    spec = CTraderService().get_symbol_specification("XAUUSD")
    assert spec["lot_size"] == 100


@pytest.mark.asyncio
async def test_news_nzd_event_maps_to_nzdusd_not_gold():
    engine = signal_candidate_engine
    previous = dict(engine.candidates)
    engine.candidates.clear()
    calendar = {
        "events": [{"event": "Official Cash Rate", "currency": "NZD", "impact": "high"}]
    }
    bars = [
        {
            "close": 0.591 + i * 0.0001,
            "high": 0.592 + i * 0.0001,
            "low": 0.590 + i * 0.0001,
            "volume": 100,
        }
        for i in range(30)
    ]
    try:
        with patch(
            "backend.services.signal_candidate_engine.ctrader_service.get_trendbars",
            return_value=bars,
        ), patch(
            "backend.routes.news.get_economic_calendar",
            new=AsyncMock(return_value=calendar),
        ), patch(
            "backend.routes.news.get_news_feed", new=AsyncMock(return_value={})
        ), patch(
            "backend.routes.news.get_market_sentiment", new=AsyncMock(return_value={})
        ), patch.object(
            engine, "_has_open_position", return_value=False
        ), patch.object(
            engine, "_portfolio_risk_breach", return_value=None
        ):
            created = await engine.scan_news_and_events()
        assert created
        assert all(c["symbol"] == "NZDUSD" for c in created)
        assert all(c["symbol"] != "XAUUSD" for c in created)
    finally:
        engine.candidates = previous


@pytest.mark.asyncio
async def test_scan_markets_skips_metals_when_disabled():
    engine = signal_candidate_engine
    previous_cfg = dict(engine.execution_config)
    previous = dict(engine.candidates)
    engine.candidates.clear()
    # Leftover post-loss cooldowns from earlier tests (ctrader_trade_sync sets
    # real 45-min cooldowns on this singleton) would skip symbols before fetch.
    previous_cooldowns = dict(engine._symbol_cooldowns)
    engine._symbol_cooldowns.clear()
    try:
        engine.execution_config["include_metals"] = False
        with patch(
            "backend.services.signal_candidate_engine.ctrader_service.get_trendbars",
            return_value=[],
        ) as fetch, patch(
            "backend.services.sentry_state.is_trading_allowed", return_value=True
        ), patch(
            # Weekend venue closure must not gate this universe-filter test.
            "backend.services.signal_candidate_engine.is_venue_open", return_value=True
        ):
            await engine.scan_markets(universe=["XAGUSD", "EURUSD"], timeframe="M5")
        fetched = [c.args[0] for c in fetch.call_args_list]
        assert "XAGUSD" not in fetched
        assert "EURUSD" in fetched
    finally:
        engine.execution_config = previous_cfg
        engine.candidates = previous
        engine._symbol_cooldowns = previous_cooldowns


def test_same_base_cap_blocks_third_eur_pair():
    engine = signal_candidate_engine
    with patch.object(engine, "_open_ctrader_symbols", return_value={"EURUSD", "EURGBP"}):
        engine.execution_config["max_same_base"] = 2
        assert engine._same_base_slots_available("EURJPY") is False
        assert engine._same_base_slots_available("GBPUSD") is True


def test_prune_drops_expired_candidates():
    engine = signal_candidate_engine
    previous = dict(engine.candidates)
    try:
        engine.candidates = {
            "old": {
                "id": "old",
                "status": CandidateStatus.READY,
                "latest_exec_at": 1,
                "created_at": "2020-01-01",
            },
            "live": {
                "id": "live",
                "status": CandidateStatus.READY,
                "latest_exec_at": 9_999_999_999,
                "created_at": "2026-01-01",
            },
        }
        removed = engine.prune_candidates(now_ts=100)
        assert removed >= 1
        assert "old" not in engine.candidates
        assert "live" in engine.candidates
    finally:
        engine.candidates = previous


def test_sizing_caps_metal_lots():
    engine = signal_candidate_engine
    engine.timing_config["account_equity_override"] = 10_000
    with patch.object(engine, "_account_equity", return_value=10_000.0), patch.dict(
        engine.execution_config, {"max_ctrader_lots": 0.10, "max_metal_lots": 0.01}
    ):
        size = engine._calculate_size("XAGUSD", 67.0, 66.5, "ctrader")
    assert size["lots"] <= 0.01


def test_sizing_uses_live_equity_not_paper_10k():
    engine = signal_candidate_engine
    with patch.object(engine, "_account_equity", return_value=159.0), patch.dict(
        engine.execution_config, {"max_ctrader_lots": 0.10, "max_metal_lots": 0.01}
    ), patch.dict(
        engine.timing_config, {"default_risk_pct": 0.5, "account_equity_override": 10_000}
    ):
        size = engine._calculate_size("EURUSD", 1.1700, 1.1650, "ctrader")
    # 0.5% of $159 is ~$0.80. The $10k override would risk $50 and size 0.10 lots.
    assert size["risk_usd"] == pytest.approx(0.80, abs=0.05)
    assert size["lots"] <= 0.02
    assert size["lots"] < 0.10


def test_account_equity_does_not_use_10k_when_live_creds_have_no_snapshot():
    engine = signal_candidate_engine
    engine.timing_config["account_equity_override"] = 10_000
    with patch(
        "backend.services.signal_candidate_engine.ctrader_service.equity", 0.0
    ), patch(
        "backend.services.signal_candidate_engine.ctrader_service.has_credentials",
        return_value=True,
    ), patch.dict(os.environ, {"CTRADER_FALLBACK_EQUITY": "150"}):
        assert engine._account_equity() == 150.0
