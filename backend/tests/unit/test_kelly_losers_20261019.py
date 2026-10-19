"""Oct 19 approved fixes: Kelly 0 skip, scoped Kelly stats, real equity,
loser blocklists, Binance orphan adoption off."""

import time
import types
from unittest.mock import MagicMock, patch

import pytest

from backend.services import kelly_stats
from backend.services.decision_engine import DecisionEngine, default_stats_modes, kelly_zero_policy
from backend.services.entry_blocklists import (
    binance_orphan_adoption_enabled,
    entry_block_reason,
    is_binance_blocked_symbol,
)
from backend.services.risk_config import RiskConfig
from backend.services.trading_loop_helpers import BrokerPositionSyncService


def _bars(n=60, base=100.0):
    return [
        {"open": base + i * 0.1 - 0.1, "high": base + i * 0.1 + 0.3,
         "low": base + i * 0.1 - 0.3, "close": base + i * 0.1, "volume": 1000.0}
        for i in range(n)
    ]


def _cfg(**kw):
    base = dict(
        min_signal_strength=0.45, ai_analysis_threshold=0.30, trade_usdt_amount=10.0,
        enable_personas=False, use_risk_reviewer_llm=False, min_edge_fee_mult=0.0,
        sl_atr_mult=1.75, tp_atr_mult=5.5, enable_jesse_ml=False,
    )
    base.update(kw)
    return RiskConfig(**base)


def _signal(price, kelly=None, win_prob=None):
    return types.SimpleNamespace(
        entry_price=price, stop_loss=price * 0.98, take_profit=price * 1.06,
        confidence=0.7, kelly_multiplier=kelly, kelly_win_prob=win_prob,
        jesse_ml_gap=None,
    )


def _engine(stats=(100, 0.01, 0.01)):
    eng = DecisionEngine(_cfg())
    eng.account_equity = 1000.0
    eng.account_available = 1000.0
    eng._partition_pnl_stats = MagicMock(return_value=stats)
    return eng


# ── B1: Kelly 0 → skip ────────────────────────────────────────────────────

def test_kelly_zero_policy_defaults_to_skip(monkeypatch):
    monkeypatch.delenv("KELLY_ZERO_POLICY", raising=False)
    assert kelly_zero_policy() == "skip"
    monkeypatch.setenv("KELLY_ZERO_POLICY", "min_size")
    assert kelly_zero_policy() == "min_size"
    monkeypatch.setenv("KELLY_ZERO_POLICY", "garbage")
    assert kelly_zero_policy() == "skip"


def test_kelly_zero_from_ml_skips_entry(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.delenv("KELLY_ZERO_POLICY", raising=False)
    bars = _bars()
    eng = _engine()
    d = eng._create_entry_decision("ETHUSDC", bars, _signal(bars[-1]["close"], kelly=0.0), "BUY", False, "TRENDING")
    assert d is None
    assert "Kelly 0" in eng.last_evaluation.get("reason", "")


def test_kelly_zero_recomputed_from_win_prob_skips(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.delenv("KELLY_ZERO_POLICY", raising=False)
    bars = _bars()
    # b = 1.0, p = 0.40 → full Kelly negative → multiplier 0.
    eng = _engine(stats=(100, 0.01, 0.01))
    d = eng._create_entry_decision("ETHUSDC", bars, _signal(bars[-1]["close"], kelly=0.8, win_prob=0.40), "BUY", False, "TRENDING")
    assert d is None
    assert "Kelly 0" in eng.last_evaluation.get("reason", "")


def test_kelly_zero_min_size_policy_takes_venue_minimum(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("KELLY_ZERO_POLICY", "min_size")
    bars = _bars()
    eng = _engine()
    price = bars[-1]["close"]
    d = eng._create_entry_decision("ETHUSDC", bars, _signal(price, kelly=0.0), "BUY", False, "TRENDING")
    assert d is not None
    assert d.quantity * d.entry_price == pytest.approx(20.0)
    assert d.kelly_multiplier is None


def test_no_ml_kelly_keeps_base_sizing(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "paper")
    bars = _bars()
    eng = _engine()
    d = eng._create_entry_decision("ETHUSDC", bars, _signal(bars[-1]["close"], kelly=None), "BUY", False, "TRENDING")
    assert d is not None
    assert d.quantity > 0


# ── B2: stats scoping and exclusion ────────────────────────────────────────

def _row(symbol="ETHUSDT", broker="binance_futures", mode="paper", pnl=1.0,
         entry=2000.0, qty=0.01, **kw):
    r = dict(symbol=symbol, broker=broker, mode=mode, status="closed", strategy="combined",
             pnl=pnl, entry_price=entry, quantity=qty, notes="", signal_id=None,
             broker_order_id=None, broker_position_id=None, binance_order_id=None)
    r.update(kw)
    return r


def _eth_book(n_win=20, n_loss=20):
    rows = [_row(pnl=+0.4) for _ in range(n_win)]          # +2% on $20 notional
    rows += [_row(pnl=-0.2) for _ in range(n_loss)]        # -1%
    return rows


def test_stats_scope_symbol_and_venue_only():
    rows = _eth_book()
    # cTrader gold with a corrupt huge win must not leak into crypto stats.
    rows += [_row(symbol="XAUUSD", broker="ctrader", mode="live", pnl=5000.0, entry=2400.0, qty=0.01)]
    st = kelly_stats.compute_scoped_stats(rows, symbol="ETHUSDC", venue="binance_futures", modes=("paper",), min_sample=30)
    assert st.scope == "symbol"           # ETHUSDT rows count for ETHUSDC
    assert st.closed_count == 40
    assert st.avg_win / st.avg_loss_abs == pytest.approx(2.0)
    assert st.excluded["other_venue_or_mode"] == 1


def test_stats_exclude_test_void_zero_and_corrupt_price():
    rows = _eth_book()
    junk = [
        _row(pnl=900.0, signal_id="test-123"),
        _row(pnl=900.0, broker_order_id="sim_ct_1788217860045"),
        _row(pnl=900.0, broker="ctrader:paper"),
        _row(pnl=900.0, notes="cTrader order sim_ct_1 placed (simulated)"),
        _row(pnl=900.0, status="void"),
        _row(pnl=900.0, strategy="exchange_reconciliation"),
        _row(pnl=0.0),
        _row(pnl=None),
        _row(pnl=900.0, entry=20.0),      # 99% below the ETH median → corrupt
        _row(pnl=900.0, mode="live"),     # wrong mode for the paper book
    ]
    st = kelly_stats.compute_scoped_stats(rows + junk, symbol="ETHUSDC", venue="binance_futures", modes=("paper",), min_sample=30)
    assert st.closed_count == 40
    assert st.avg_win / st.avg_loss_abs == pytest.approx(2.0)
    ex = st.excluded
    assert ex["test"] == 4
    assert ex["void"] == 1
    assert ex["reconciliation"] == 1
    assert ex["null_or_zero_pnl"] == 2
    assert ex["corrupt_price"] == 1
    assert ex["other_venue_or_mode"] == 1


def test_stats_asset_class_fallback_then_thin_book():
    rows = [_row(symbol="SOLUSDT", entry=150.0, qty=0.1, pnl=+0.3) for _ in range(20)]
    rows += [_row(symbol="SOLUSDT", entry=150.0, qty=0.1, pnl=-0.3) for _ in range(20)]
    rows += _eth_book(3, 3)
    st = kelly_stats.compute_scoped_stats(rows, symbol="ETHUSDC", venue="binance_futures", modes=("paper",), min_sample=30)
    assert st.scope == "asset_class"
    assert st.symbol_count == 6 and st.class_count == 46

    thin = kelly_stats.compute_scoped_stats(_eth_book(3, 3), symbol="ETHUSDC", venue="binance_futures", modes=("paper",), min_sample=30)
    assert thin.scope == "fallback"
    assert thin.closed_count < 30


def test_thin_book_uses_conservative_b(monkeypatch):
    from backend.services.jesse_ml_gates import empirical_payoff_ratio, net_theoretical_payoff_ratio

    monkeypatch.delenv("KELLY_FALLBACK_PAYOFF_CAP", raising=False)
    fb = min(net_theoretical_payoff_ratio(), kelly_stats.kelly_fallback_payoff_cap())
    assert fb <= 1.5
    assert empirical_payoff_ratio(0.0, 0.0, 5, fallback=fb, min_closed=30) == pytest.approx(fb)


def test_engine_stats_use_loop_venue_not_risk_broker(monkeypatch):
    """ACTIVE_BROKER=ctrader must not feed cTrader rows into a Binance symbol."""
    monkeypatch.setenv("ACTIVE_BROKER", "ctrader")
    kelly_stats.clear_cache()
    eng = DecisionEngine(_cfg())
    eng.stats_venue = "binance_futures"
    eng.stats_modes = ("paper",)
    seen = {}

    def fake_load(venue):
        seen["venue"] = venue
        return _eth_book() + [_row(symbol="XAUUSD", broker="ctrader", mode="live", pnl=5000.0)]

    with patch.object(kelly_stats, "load_closed_rows", side_effect=fake_load):
        n, w, l = eng._partition_pnl_stats(symbol="ETHUSDC")
    assert seen["venue"] == "binance_futures"
    assert n == 40 and w / l == pytest.approx(2.0)
    assert eng._last_kelly_stats.scope == "symbol"


def test_default_stats_modes(monkeypatch):
    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "true")
    assert default_stats_modes("binance_futures") == ("paper",)
    assert set(default_stats_modes("ctrader")) == {"live", "demo"}


# ── B3: equity source ──────────────────────────────────────────────────────

def test_sizing_equity_ignores_paper_balance_default():
    from backend.services.trading_loop import TradingLoopService

    default = TradingLoopService._paper_balance_payload(100000.0, 100000.0, 0.0, source="paper_balance_default")
    assert TradingLoopService._sizing_equity(default, 100000.0) == 0.0
    book = TradingLoopService._paper_balance_payload(104805.76, 104805.76, 0.0)
    assert book["equity_source"] == "paper_book"
    assert TradingLoopService._sizing_equity(book, 104805.76) == pytest.approx(104805.76)
    live = {"equity": 512.0, "broker": "binance_futures"}
    assert TradingLoopService._sizing_equity(live, 512.0) == pytest.approx(512.0)


def test_ctrader_stale_equity_snapshot_not_used(monkeypatch):
    from backend.services import signal_candidate_engine as sce

    monkeypatch.setenv("CTRADER_EQUITY_MAX_AGE_SEC", "900")
    monkeypatch.setenv("CTRADER_FALLBACK_EQUITY", "150")
    fake = MagicMock()
    fake.equity = 5000.0
    fake.balance_age_sec.return_value = 4 * 3600.0
    monkeypatch.setattr(sce, "ctrader_service", fake)
    eng = sce.SignalCandidateEngine()
    assert eng._account_equity() == pytest.approx(150.0)
    fake.request_balance_refresh.assert_called_once()

    fake.balance_age_sec.return_value = 30.0
    assert eng._account_equity() == pytest.approx(5000.0)


def test_ctrader_balance_age_tracks_snapshot():
    from backend.services.ctrader_service import CTraderService

    svc = CTraderService.__new__(CTraderService)
    svc._balance_updated_at = 0.0
    assert svc.balance_age_sec() is None
    svc._balance_updated_at = time.monotonic() - 10
    assert 9 <= svc.balance_age_sec() <= 60


def test_binance_paper_candidates_size_from_paper_book(monkeypatch):
    from backend.services import signal_candidate_engine as sce

    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "true")
    with patch("backend.services.unified_trading.UnifiedTrading.get_paper_portfolio",
               return_value={"equity": 104805.76, "cash": 104805.76}):
        eng = sce.SignalCandidateEngine()
        assert eng._resolve_equity("binance_futures") == pytest.approx(104805.76)


# ── C: blocklists ──────────────────────────────────────────────────────────

@pytest.fixture
def default_blocklists(monkeypatch):
    for k in ("CTRADER_DISABLED_STRATEGIES", "CTRADER_SYMBOL_BLOCKLIST", "BINANCE_BASE_BLOCKLIST", "BINANCE_ORPHAN_ADOPTION"):
        monkeypatch.delenv(k, raising=False)


def test_ctrader_momentum_disabled_other_strategies_allowed(default_blocklists):
    assert entry_block_reason("ctrader", "EURUSD", "MOMENTUM_TREND_PULSE")
    assert entry_block_reason("ctrader", "EURUSD", "FADE_OVEREXTENSION") is None
    assert entry_block_reason("ctrader", "EURUSD", "STRADDLE_VOLATILITY_BRACKET") is None
    # Binance momentum is not part of the cTrader cut.
    assert entry_block_reason("binance_futures", "ETHUSDC", "MOMENTUM_TREND_PULSE") is None


def test_ctrader_metals_blocked(default_blocklists):
    for sym in ("XAUUSD", "XAGUSD", "xauusd.", "GOLD", "SILVER"):
        assert entry_block_reason("ctrader", sym, "FADE_OVEREXTENSION"), sym
    assert entry_block_reason("ctrader", "GBPUSD") is None


def test_binance_small_caps_blocked(default_blocklists):
    for sym in ("CELRUSDT", "AGTUSDT", "ICNTUSDT", "ONEUSDT", "AKEUSDT", "BULLAUSDT", "EPICUSDT", "ONEUSDC"):
        assert is_binance_blocked_symbol(sym), sym
        assert entry_block_reason("binance_futures", sym)
    for sym in ("ETHUSDC", "SOLUSDC", "ONDOUSDT", "BTCUSDT"):
        assert not is_binance_blocked_symbol(sym), sym


def test_blocklists_env_override(monkeypatch):
    monkeypatch.setenv("CTRADER_SYMBOL_BLOCKLIST", "none")
    monkeypatch.setenv("CTRADER_DISABLED_STRATEGIES", "none")
    assert entry_block_reason("ctrader", "XAUUSD", "MOMENTUM_TREND_PULSE") is None


@pytest.mark.asyncio
async def test_execute_candidate_refuses_blocked_even_when_forced(default_blocklists):
    from backend.services.signal_candidate_engine import SignalCandidateEngine

    eng = SignalCandidateEngine()
    now = int(time.time())
    eng.candidates["c1"] = {
        "id": "c1", "symbol": "XAGUSD", "broker": "ctrader", "strategy": "FADE_OVEREXTENSION",
        "direction": "BUY", "entry_price": 30.0, "stop_loss": 29.0, "take_profit": 33.0,
        "earliest_exec_at": now - 1, "latest_exec_at": now + 600, "status": "READY",
    }
    eng.candidates["c2"] = dict(eng.candidates["c1"], id="c2", symbol="EURUSD", strategy="MOMENTUM_TREND_PULSE")
    with patch("backend.services.sentry_state.is_trading_allowed", return_value=True):
        r1 = await eng.execute_candidate("c1", force=True)
        r2 = await eng.execute_candidate("c2", force=True)
    assert r1["success"] is False and "BLOCKLIST" in r1["error"]
    assert r2["success"] is False and "MOMENTUM_TREND_PULSE" in r2["error"]
    assert eng.candidates["c1"]["status"] == "CANCELLED"


# ── C: Binance orphan adoption off ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_orphan_adoption_off_by_default(default_blocklists):
    assert binance_orphan_adoption_enabled() is False
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = []
    broker = MagicMock()
    broker.dry_run = False
    broker.get_positions.return_value = [
        {"symbol": "CELRUSDT", "side": "BUY", "quantity": 1000.0, "entry_price": 0.01},
    ]
    updated = await BrokerPositionSyncService.sync_positions(db, broker, {}, {})
    assert updated == 0
    db.add.assert_not_called()
    broker.ensure_protective_orders.assert_not_called()
    broker.cancel_all_orders.assert_not_called()


# ── Review guards (Oct 19 08:52): blocklists are entry-only; sizing; sync ──

def _mem_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.database.models import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_open_ctrader_metal_positions_untouched_by_sync(default_blocklists):
    """Blocked XAUUSD/XAGUSD legs that are already open stay open with SL/TP."""
    from backend.database.models import Trade
    from backend.services.ctrader_trade_sync import reconcile_ctrader_positions

    db = _mem_session()
    for sym, pid, sl, tp, strat in (
        ("XAUUSD", "298065468", 2380.0, 2460.0, "ctrader_live"),
        ("XAGUSD", "298104727", 29.5, 31.5, "MOMENTUM_TREND_PULSE"),
    ):
        db.add(Trade(symbol=sym, direction="BUY", quantity=0.01, entry_price=tp - 40 if sym == "XAUUSD" else 30.0,
                     stop_loss=sl, take_profit=tp, status="open", broker="ctrader", exchange="ctrader",
                     strategy=strat, broker_position_id=pid, mode="demo"))
    db.commit()
    live = [
        {"symbol": "XAUUSD", "side": "BUY", "quantity": 0.01, "entry_price": 2420.0, "position_id": "298065468",
         "stop_loss": 2380.0, "take_profit": 2460.0},
        {"symbol": "XAGUSD", "side": "BUY", "quantity": 0.01, "entry_price": 30.0, "position_id": "298104727",
         "stop_loss": 29.5, "take_profit": 31.5},
    ]
    res = reconcile_ctrader_positions(db, live_positions=live, broker=None)
    assert res.get("closed", 0) == 0
    rows = {t.symbol: t for t in db.query(Trade).all()}
    assert len(rows) == 2
    for sym in ("XAUUSD", "XAGUSD"):
        assert rows[sym].status == "open"
        assert rows[sym].quantity == pytest.approx(0.01)
    assert rows["XAUUSD"].stop_loss == pytest.approx(2380.0) and rows["XAUUSD"].take_profit == pytest.approx(2460.0)
    assert rows["XAGUSD"].stop_loss == pytest.approx(29.5) and rows["XAGUSD"].take_profit == pytest.approx(31.5)


def test_blocklist_not_wired_into_management_or_sync_modules():
    """Entry-only: SL/TP, trailing, partial TP, exits and trade sync never consult it."""
    import inspect
    from backend.services import ctrader_trade_sync, binance_trade_sync, trading_loop_helpers

    for mod in (ctrader_trade_sync, binance_trade_sync):
        assert "entry_block_reason" not in inspect.getsource(mod)
        assert "is_binance_blocked_symbol" not in inspect.getsource(mod)
    helpers = inspect.getsource(trading_loop_helpers)
    assert "entry_block_reason" not in helpers and "is_binance_blocked_symbol" not in helpers
    # Only the orphan-adoption flag is read in the helpers.
    assert helpers.count("binance_orphan_adoption_enabled") == 2  # import + call


@pytest.mark.asyncio
async def test_symbol_gate_keeps_open_blocked_leg_for_management(default_blocklists, monkeypatch):
    from backend.services import trading_loop as tl

    svc = tl.TradingLoopService.__new__(tl.TradingLoopService)
    svc.risk_config = MagicMock(symbol_blacklist=set(), min_24h_quote_volume_usdt=0)
    svc._apply_expectancy_gate = lambda c: c
    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.distinct.return_value.all.return_value = [("CELRUSDT",)]
    monkeypatch.setattr(tl, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(tl.binance_futures_broker, "futures_listing_status", lambda s: True)
    out = await svc._filter_tradeable_symbols(["CELRUSDT", "AGTUSDT", "ETHUSDC"])
    assert "CELRUSDT" in out          # open → still managed (SL/TP, trailing, partial TP)
    assert "AGTUSDT" not in out       # flat → no new entry
    assert "ETHUSDC" in out


def test_kelly_scales_existing_risk_per_trade_formula(monkeypatch):
    """Kelly only multiplies the configured risk_per_trade_pct; it does not replace it."""
    monkeypatch.setenv("TRADING_MODE", "paper")
    bars = _bars()
    price = bars[-1]["close"]
    for pct in (0.005, 0.01):
        eng = DecisionEngine(_cfg(risk_per_trade_pct=pct, max_directional_exposure_usdt=0,
                                  max_trade_notional_equity_mult=1000.0))
        eng.account_equity = 10_000.0
        eng.account_available = 10_000.0
        eng.account_leverage = 100.0
        eng._partition_pnl_stats = MagicMock(return_value=(100, 0.02, 0.01))
        sig = _signal(price, kelly=0.5)
        d = eng._create_entry_decision("ETHUSDC", bars, sig, "BUY", False, "TRENDING")
        assert d is not None
        risk_usd = d.quantity * abs(d.entry_price - d.stop_loss)
        assert risk_usd == pytest.approx(10_000.0 * pct * 0.5, rel=1e-6)


def test_signal_engine_default_risk_pct_unchanged():
    from backend.services.signal_candidate_engine import SignalCandidateEngine

    assert SignalCandidateEngine().timing_config["default_risk_pct"] == 0.5


@pytest.mark.asyncio
async def test_orphan_off_still_rebinds_and_closes_loop_rows(default_blocklists):
    """Adoption off must not change reconciliation of the loop's own rows."""
    own = types.SimpleNamespace(
        symbol="ETHUSDC", direction="BUY", entry_price=2400.0, quantity=0.05, exit_price=None,
        pnl=None, status="filled", closed_at=None, notes="", binance_order_id="777",
        broker_position_id=None, mode="live", broker="binance_futures", broker_order_id="777",
        exchange="binance_futures", strategy="combined",
    )
    gone = types.SimpleNamespace(
        symbol="SOLUSDC", direction="SELL", entry_price=150.0, quantity=1.0, exit_price=None,
        pnl=None, status="filled", closed_at=None, notes="", binance_order_id="888",
        broker_position_id=None, mode="live", broker="binance_futures", broker_order_id="888",
        exchange="binance_futures", strategy="combined",
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [own, gone]
    broker = MagicMock()
    broker.dry_run = False
    broker.get_positions.return_value = [
        {"symbol": "ETHUSDT", "side": "BUY", "quantity": 0.05, "entry_price": 2400.0},
        {"symbol": "AKEUSDT", "side": "BUY", "quantity": 100.0, "entry_price": 0.5},
    ]
    broker.get_exit_price.return_value = 149.0
    await BrokerPositionSyncService.sync_positions(db, broker, {}, {})
    assert own.status == "filled" and own.symbol == "ETHUSDT"     # rebound to the live leg
    assert gone.status == "closed"                                 # own row absent → closed as before
    db.add.assert_not_called()                                     # AKE leg not adopted
    broker.ensure_protective_orders.assert_not_called()


@pytest.mark.asyncio
async def test_paper_book_sync_unaffected_by_orphan_flag(default_blocklists):
    """Paper/dry-run adapters are never reconciled (unchanged)."""
    broker = MagicMock()
    broker.dry_run = True
    assert await BrokerPositionSyncService.sync_positions(MagicMock(), broker, {}, {}) == 0
    broker.get_positions.assert_not_called()
