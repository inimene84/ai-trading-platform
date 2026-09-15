import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone, timedelta

from backend.database.models import Base, Trade, PortfolioSnapshot
from backend.services.risk_config import RiskConfig
from backend.services.risk_guard import enforce_risk_limits, RiskBreach

@pytest.fixture(autouse=True)
def disable_risk_guard_override(monkeypatch):
    monkeypatch.setenv("DISABLE_RISK_GUARD", "false")

@pytest.fixture
def db_session():
    # Setup clean in-memory SQLite database for testing queries
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()

def test_risk_guard_allows_safe_portfolio(db_session):
    cfg = RiskConfig(
        max_position_risk_pct=1.0,
        max_portfolio_drawdown_pct=20.0,
        max_daily_loss_pct=5.0,
        max_open_positions=10,
        sl_cooldown_minutes=30
    )
    
    # Standard snapshot inside limits
    snap = PortfolioSnapshot(
        total_value=10000.0,
        cash=10000.0,
        timestamp=datetime.now(timezone.utc)
    )
    db_session.add(snap)
    db_session.commit()
    
    # Should run cleanly without raising exception
    enforce_risk_limits(db_session, cfg, [], snap)

def test_risk_guard_blocks_excessive_positions(db_session):
    cfg = RiskConfig(
        max_position_risk_pct=1.0,
        max_portfolio_drawdown_pct=20.0,
        max_daily_loss_pct=5.0,
        max_positions=2,
        max_open_positions=2,  # set low limit
        sl_cooldown_minutes=30
    )
    
    trades = [
        Trade(symbol="BTCUSDT", direction="BUY", quantity=0.1, status="open"),
        Trade(symbol="ETHUSDT", direction="BUY", quantity=1.0, status="open"),
        Trade(symbol="SOLUSDT", direction="BUY", quantity=5.0, status="open"),  # 3 open trades
    ]
    
    with pytest.raises(RiskBreach, match="Max open positions exceeded"):
        enforce_risk_limits(db_session, cfg, trades, None)


def test_risk_guard_counts_distinct_symbols_not_dca_layers(db_session):
    """Regression: pyramid-DCA layers of the SAME symbol are ONE position.

    Each DCA entry is its own Trade row. Counting raw rows (len(open_trades))
    over-counts and false-triggers the cap. The guard must count distinct
    symbols, matching the trading loop's func.count(func.distinct(Trade.symbol)).

    Real incident: 6 live positions held 11 DB rows (AVAX had 4 layers); the
    old len()-based check raised 11 > 10 and killed the live loop on a flat book.
    """
    cfg = RiskConfig(
        max_position_risk_pct=1.0,
        max_portfolio_drawdown_pct=20.0,
        max_daily_loss_pct=5.0,
        max_positions=10,
        max_open_positions=10,
        sl_cooldown_minutes=30,
    )

    # 11 rows across only 6 distinct symbols (AVAX = 4 DCA layers) → 6 <= 10, OK.
    trades = [
        Trade(symbol="AVAXUSDT", direction="SELL", quantity=3.0, entry_price=6.78, status="filled"),
        Trade(symbol="AVAXUSDT", direction="SELL", quantity=3.0, entry_price=6.80, status="filled"),
        Trade(symbol="AVAXUSDT", direction="SELL", quantity=3.0, entry_price=6.77, status="filled"),
        Trade(symbol="AVAXUSDT", direction="SELL", quantity=3.0, entry_price=6.73, status="filled"),
        Trade(symbol="LINKUSDT", direction="BUY", quantity=3.18, entry_price=7.88, status="filled"),
        Trade(symbol="LINKUSDT", direction="BUY", quantity=3.16, entry_price=7.91, status="filled"),
        Trade(symbol="ETHUSDT", direction="BUY", quantity=0.015, entry_price=1674.5, status="filled"),
        Trade(symbol="ETHUSDT", direction="BUY", quantity=0.015, entry_price=1682.9, status="filled"),
        Trade(symbol="BNBUSDT", direction="BUY", quantity=0.04, entry_price=606.5, status="filled"),
        Trade(symbol="BTCUSDT", direction="BUY", quantity=0.001, entry_price=63196.9, status="filled"),
        Trade(symbol="SOLUSDT", direction="BUY", quantity=0.4, entry_price=66.22, status="filled"),
    ]

    # Must NOT raise: 6 distinct symbols within the cap of 10.
    enforce_risk_limits(db_session, cfg, trades, None)


def test_paper_drawdown_uses_cash_not_inflated_total(db_session, monkeypatch):
    """Regression: paper equity was cash+margin and halted a flat $100k book.

    Peak $198k (1x LINK/LTC margin added to cash) vs current $133k (10x
    margin added to cash) is a 33% fake drawdown. Cash stayed $100k.
    """
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("PAPER_TRADING", "true")
    cfg = RiskConfig(
        max_portfolio_drawdown_pct=20.0,
        max_daily_loss_pct=99.0,
        max_positions=20,
        max_open_positions=20,
        max_directional_exposure_usdt=0,
        equity_sizing_enabled=False,
    )
    now = datetime.now(timezone.utc)
    db_session.add(
        PortfolioSnapshot(
            total_value=198465.54,
            cash=100000.0,
            positions_value=98474.48,
            timestamp=now - timedelta(hours=3),
        )
    )
    current = PortfolioSnapshot(
        total_value=132960.60,
        cash=100000.0,
        positions_value=428080.51,
        timestamp=now,
    )
    db_session.add(current)
    db_session.commit()
    enforce_risk_limits(db_session, cfg, [], current)


def test_inflated_paper_snapshots_do_not_trip_drawdown_in_live_mode(db_session, monkeypatch):
    """Same $198k/$133k rows must not halt after TRADING_MODE is flipped live."""
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("PAPER_TRADING", "false")
    monkeypatch.setenv("DRY_RUN_ALL", "false")
    cfg = RiskConfig(
        max_portfolio_drawdown_pct=20.0,
        max_daily_loss_pct=99.0,
        max_positions=20,
        max_open_positions=20,
        max_directional_exposure_usdt=0,
        equity_sizing_enabled=False,
    )
    now = datetime.now(timezone.utc)
    db_session.add(
        PortfolioSnapshot(
            total_value=198465.54,
            cash=100000.0,
            positions_value=98474.48,
            timestamp=now - timedelta(hours=3),
        )
    )
    current = PortfolioSnapshot(
        total_value=132960.60,
        cash=100000.0,
        positions_value=428080.51,
        timestamp=now,
    )
    db_session.add(current)
    db_session.commit()
    enforce_risk_limits(db_session, cfg, [], current)


def test_risk_guard_blocks_excessive_drawdown(db_session):
    cfg = RiskConfig(
        max_position_risk_pct=1.0,
        max_portfolio_drawdown_pct=20.0,  # 20% limit
        max_daily_loss_pct=5.0,
        max_open_positions=10,
        sl_cooldown_minutes=30
    )
    
    # Historic peak value is 10000
    snap_peak = PortfolioSnapshot(
        total_value=10000.0,
        cash=10000.0,
        timestamp=datetime.now(timezone.utc) - timedelta(hours=3)
    )
    db_session.add(snap_peak)
    
    # Current value dropped to 7800 (22% drawdown)
    snap_low = PortfolioSnapshot(
        total_value=7800.0,
        cash=7800.0,
        timestamp=datetime.now(timezone.utc)
    )
    db_session.add(snap_low)
    db_session.commit()
    
    with pytest.raises(RiskBreach, match="drawdown exceeded"):
        enforce_risk_limits(db_session, cfg, [], snap_low)

def test_risk_guard_blocks_excessive_daily_loss(db_session):
    cfg = RiskConfig(
        max_position_risk_pct=1.0,
        max_portfolio_drawdown_pct=20.0,
        max_daily_loss_pct=5.0,  # 5% limit
        max_open_positions=10,
        sl_cooldown_minutes=30
    )
    
    # Anchor both snapshots to fixed mid-day UTC timestamps so the test never
    # depends on the wall-clock time CI happens to run at. (A previous version
    # used datetime.now(...).replace(hour=0...) + 10min as the baseline, which
    # broke when CI ran in the first 10 minutes of a UTC day: the "now" snapshot
    # then sorted BEFORE the baseline and became the day's baseline itself,
    # yielding 0% loss and no breach.)
    today = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)

    # First snapshot of today (UTC) value was 10000
    snap_today_start = PortfolioSnapshot(
        total_value=10000.0,
        cash=10000.0,
        timestamp=today
    )
    db_session.add(snap_today_start)
    
    # Drop to 9400 later today (6% daily loss)
    snap_now = PortfolioSnapshot(
        total_value=9400.0,
        cash=9400.0,
        timestamp=today + timedelta(hours=2)
    )
    db_session.add(snap_now)
    db_session.commit()
    
    with pytest.raises(RiskBreach, match="daily loss exceeded"):
        enforce_risk_limits(db_session, cfg, [], snap_now)


def test_risk_guard_blocks_excessive_directional_exposure(db_session):
    # Hard USDT cap applies when equity sizing is off (no dynamic equity multiplier).
    # Default equity_sizing_enabled=True would raise the cap to equity * 4 in paper
    # mode and this $600 book would not trip.
    cfg = RiskConfig(
        max_directional_exposure_usdt=500.0,
        max_positions=10,
        max_open_positions=10,
        equity_sizing_enabled=False,
    )
    trades = [
        Trade(symbol="BTCUSDT", direction="BUY", quantity=0.005, entry_price=60000.0, status="open"),  # $300
        Trade(symbol="ETHUSDT", direction="BUY", quantity=0.1, entry_price=3000.0, status="open"),      # $300
    ]
    with pytest.raises(RiskBreach, match="directional exposure exceeded"):
        enforce_risk_limits(db_session, cfg, trades, None)


def test_risk_guard_equity_multiplier_raises_hard_usdt_cap(db_session):
    """With equity sizing, the directional cap is max(hard USDT, equity * mult)."""
    cfg = RiskConfig(
        max_directional_exposure_usdt=500.0,
        max_direction_notional_equity_mult=4.0,
        max_positions=10,
        max_open_positions=10,
        equity_sizing_enabled=True,
    )
    snap = PortfolioSnapshot(
        total_value=10000.0,
        cash=10000.0,
        timestamp=datetime.now(timezone.utc),
    )
    trades = [
        Trade(symbol="BTCUSDT", direction="BUY", quantity=0.005, entry_price=60000.0, status="open"),  # $300
        Trade(symbol="ETHUSDT", direction="BUY", quantity=0.1, entry_price=3000.0, status="open"),      # $300
    ]
    # $600 < max($500, $10000 * 4) = $40000 — must not trip.
    enforce_risk_limits(db_session, cfg, trades, snap)


def test_risk_guard_equity_multiplier_still_enforces_when_equity_is_small(db_session):
    cfg = RiskConfig(
        max_directional_exposure_usdt=500.0,
        max_direction_notional_equity_mult=4.0,
        max_positions=10,
        max_open_positions=10,
        equity_sizing_enabled=True,
    )
    snap = PortfolioSnapshot(
        total_value=100.0,
        cash=100.0,
        timestamp=datetime.now(timezone.utc),
    )
    trades = [
        Trade(symbol="BTCUSDT", direction="BUY", quantity=0.005, entry_price=60000.0, status="open"),  # $300
        Trade(symbol="ETHUSDT", direction="BUY", quantity=0.1, entry_price=3000.0, status="open"),      # $300
    ]
    # cap = max($500, $100 * 4) = $500; $600 must trip.
    with pytest.raises(RiskBreach, match="directional exposure exceeded"):
        enforce_risk_limits(db_session, cfg, trades, snap)


def test_risk_guard_allows_exposure_within_cap(db_session):
    cfg = RiskConfig(
        max_directional_exposure_usdt=500.0,
        max_positions=10,
        max_open_positions=10,
        equity_sizing_enabled=False,
    )
    trades = [
        Trade(symbol="BTCUSDT", direction="BUY", quantity=0.001, entry_price=60000.0, status="open"),  # $60
    ]
    enforce_risk_limits(db_session, cfg, trades, None)  # no raise


def test_disable_risk_guard_ignored_in_live_mode(db_session, monkeypatch):
    # DISABLE_RISK_GUARD must NOT bypass guards when trading live.
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("DISABLE_RISK_GUARD", "true")
    monkeypatch.setenv("PAPER_TRADING", "false")
    monkeypatch.setenv("DRY_RUN_ALL", "false")
    cfg = RiskConfig(max_positions=1, max_open_positions=1)
    trades = [
        Trade(symbol="BTCUSDT", direction="BUY", quantity=0.1, entry_price=1.0, status="open"),
        Trade(symbol="ETHUSDT", direction="BUY", quantity=0.1, entry_price=1.0, status="open"),
    ]
    with pytest.raises(RiskBreach, match="Max open positions exceeded"):
        enforce_risk_limits(db_session, cfg, trades, None)


def test_disable_risk_guard_honored_in_paper_mode(db_session, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("DISABLE_RISK_GUARD", "true")
    monkeypatch.setenv("DRY_RUN_ALL", "true")  # -> paper mode
    cfg = RiskConfig(max_positions=1, max_open_positions=1)
    trades = [Trade(symbol="BTCUSDT", direction="BUY", quantity=0.1, entry_price=1.0, status="open")] * 3
    # Disabled in paper mode -> returns without raising even though over cap.
    enforce_risk_limits(db_session, cfg, trades, None)


def test_daily_loss_uses_prior_day_baseline_when_no_snapshot_today(db_session):
    # Cold start of a new day: no snapshot row persisted for today yet, but the
    # caller passes a live latest_snapshot reflecting current equity. The check
    # must fall back to yesterday's close as baseline instead of skipping
    # entirely (the old fail-open behaviour). latest_snapshot is intentionally
    # NOT added to the DB, mirroring a live snapshot not yet committed.
    cfg = RiskConfig(max_daily_loss_pct=5.0, max_portfolio_drawdown_pct=99.0,
                     max_positions=10, max_open_positions=10)
    start_today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    # Yesterday's close persisted at 10000 (the fallback baseline)
    db_session.add(PortfolioSnapshot(total_value=10000.0, cash=10000.0,
                                     timestamp=start_today - timedelta(hours=2)))
    db_session.commit()
    # Live equity now 9400 (-6%), passed in but not persisted -> no today row.
    live = PortfolioSnapshot(total_value=9400.0, cash=9400.0,
                             timestamp=datetime.now(timezone.utc))
    with pytest.raises(RiskBreach, match="daily loss exceeded"):
        enforce_risk_limits(db_session, cfg, [], live)


def test_risk_guard_allows_20_positions_across_dual_brokers(db_session, monkeypatch):
    monkeypatch.setenv("ACTIVE_BROKER", "dual")
    cfg = RiskConfig(
        max_positions=20,
        max_open_positions=20,
        max_binance_positions=10,
        max_ctrader_positions=10,
        max_portfolio_drawdown_pct=99.0,
        max_daily_loss_pct=99.0,
        max_directional_exposure_usdt=0,  # disable exposure check
    )

    ctrader_syms = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD", "EURGBP", "EURJPY", "GBPJPY"]
    binance_syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "AVAXUSDT", "LINKUSDT", "NEARUSDT", "LTCUSDT", "DOTUSDT"]

    trades = [
        Trade(symbol=s, direction="BUY", quantity=0.1, entry_price=1.0, broker="ctrader", status="open")
        for s in ctrader_syms
    ] + [
        Trade(symbol=s, direction="BUY", quantity=1.0, entry_price=1.0, broker="binance_futures", status="open")
        for s in binance_syms
    ]

    assert len(trades) == 20
    # 20 distinct symbols should pass cleanly
    enforce_risk_limits(db_session, cfg, trades, None)

    # 21st position should breach risk cap
    trades.append(Trade(symbol="ATOMUSDT", direction="BUY", quantity=1.0, entry_price=1.0, broker="binance_futures", status="open"))
    with pytest.raises(RiskBreach, match="Max open positions exceeded: 21 > 20"):
        enforce_risk_limits(db_session, cfg, trades, None)


def test_risk_guard_split_book_counts_only_live_binance_positions(db_session, monkeypatch):
    monkeypatch.setenv("ACTIVE_BROKER", "ctrader")
    monkeypatch.setenv("CTRADER_ENV", "sandbox")
    monkeypatch.setenv("BINANCE_TESTNET", "false")
    cfg = RiskConfig(
        max_positions=2,
        max_open_positions=2,
        max_portfolio_drawdown_pct=99.0,
        max_daily_loss_pct=99.0,
        max_directional_exposure_usdt=0,
    )
    trades = [
        Trade(symbol="EURUSD", direction="BUY", quantity=0.1, entry_price=1.0, broker="ctrader", status="open"),
        Trade(symbol="GBPUSD", direction="BUY", quantity=0.1, entry_price=1.0, broker="ctrader", status="open"),
        Trade(symbol="XAUUSD", direction="BUY", quantity=0.1, entry_price=1.0, broker="ctrader", status="open"),
        Trade(symbol="BTCUSDC", direction="BUY", quantity=0.1, entry_price=1.0, broker="binance_futures", status="open"),
        Trade(symbol="ETHUSDC", direction="BUY", quantity=0.1, entry_price=1.0, broker="binance_futures", status="open"),
    ]
    enforce_risk_limits(db_session, cfg, trades, None)
    trades.append(
        Trade(symbol="SOLUSDC", direction="BUY", quantity=0.1, entry_price=1.0, broker="binance_futures", status="open")
    )
    with pytest.raises(RiskBreach, match="Max open positions exceeded: 3 > 2"):
        enforce_risk_limits(db_session, cfg, trades, None)


def test_risk_guard_suppresses_drawdown_when_testing_flag_enabled(db_session, monkeypatch):
    monkeypatch.setenv("DISABLE_DRAWDOWN_IN_TESTING", "true")
    cfg = RiskConfig(
        max_position_risk_pct=1.0,
        max_portfolio_drawdown_pct=20.0,
        max_daily_loss_pct=100.0,
        max_open_positions=10,
        sl_cooldown_minutes=30
    )
    snap_peak = PortfolioSnapshot(
        total_value=10000.0,
        cash=10000.0,
        timestamp=datetime.now(timezone.utc) - timedelta(hours=3)
    )
    db_session.add(snap_peak)
    snap_low = PortfolioSnapshot(
        total_value=7500.0,  # 25% drawdown > 20%
        cash=7500.0,
        timestamp=datetime.now(timezone.utc)
    )
    db_session.add(snap_low)
    db_session.commit()

    # Should not raise RiskBreach because DISABLE_DRAWDOWN_IN_TESTING is true
    enforce_risk_limits(db_session, cfg, [], snap_low)


def test_risk_guard_suppresses_drawdown_in_ctrader_sandbox(db_session, monkeypatch):
    monkeypatch.setenv("CTRADER_ENV", "sandbox")
    monkeypatch.setenv("BINANCE_TESTNET", "true")
    monkeypatch.setenv("DISABLE_DRAWDOWN_IN_TESTING", "false")
    cfg = RiskConfig(
        max_position_risk_pct=1.0,
        max_portfolio_drawdown_pct=20.0,
        max_daily_loss_pct=100.0,
        max_open_positions=10,
        sl_cooldown_minutes=30
    )
    snap_peak = PortfolioSnapshot(
        total_value=1000.0,
        cash=1000.0,
        timestamp=datetime.now(timezone.utc) - timedelta(hours=3)
    )
    db_session.add(snap_peak)
    snap_low = PortfolioSnapshot(
        total_value=750.0,  # 25% drawdown > 20%
        cash=750.0,
        timestamp=datetime.now(timezone.utc)
    )
    db_session.add(snap_low)
    db_session.commit()

    # Sandbox suppresses drawdown halt so tests/demos are not locked out
    enforce_risk_limits(db_session, cfg, [], snap_low)


def test_risk_guard_enforces_sandbox_drawdown_when_explicitly_configured(db_session, monkeypatch):
    monkeypatch.setenv("CTRADER_ENV", "sandbox")
    monkeypatch.setenv("BINANCE_TESTNET", "true")
    monkeypatch.setenv("ENFORCE_SANDBOX_DRAWDOWN", "true")
    monkeypatch.setenv("DISABLE_DRAWDOWN_IN_TESTING", "false")
    cfg = RiskConfig(
        max_position_risk_pct=1.0,
        max_portfolio_drawdown_pct=20.0,
        max_daily_loss_pct=100.0,
        max_open_positions=10,
        sl_cooldown_minutes=30
    )
    snap_peak = PortfolioSnapshot(
        total_value=1000.0,
        cash=1000.0,
        timestamp=datetime.now(timezone.utc) - timedelta(hours=3)
    )
    db_session.add(snap_peak)
    snap_low = PortfolioSnapshot(
        total_value=750.0,  # 25% drawdown > 20%
        cash=750.0,
        timestamp=datetime.now(timezone.utc)
    )
    db_session.add(snap_low)
    db_session.commit()

    with pytest.raises(RiskBreach, match="drawdown exceeded"):
        enforce_risk_limits(db_session, cfg, [], snap_low)


def test_filter_snapshots_strict_partitioning_live_vs_paper(monkeypatch):
    from backend.services.risk_guard import _filter_snapshots_for_risk
    from backend.database.models import PortfolioSnapshot

    snap_paper = PortfolioSnapshot(id=1, broker="ctrader", mode="paper", total_value=100000.0, cash=100000.0)
    snap_live = PortfolioSnapshot(id=2, broker="ctrader", mode="live", total_value=500.0, cash=500.0)
    snap_binance = PortfolioSnapshot(id=3, broker="binance_futures", mode="live", total_value=1000.0, cash=1000.0)
    rows = [snap_paper, snap_live, snap_binance]

    # When live and ctrader
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("ACTIVE_BROKER", "ctrader")
    filtered = _filter_snapshots_for_risk(rows)
    assert len(filtered) == 1
    assert filtered[0].id == 2
    assert filtered[0].mode == "live"

    # When live and no live snapshots exist, must return empty list (no fallback to paper)
    filtered_no_live = _filter_snapshots_for_risk([snap_paper])
    assert filtered_no_live == []

    # When paper mode, paper snapshot is returned
    monkeypatch.setenv("TRADING_MODE", "paper")
    filtered_paper = _filter_snapshots_for_risk(rows)
    assert any(s.id == 1 for s in filtered_paper)


def test_filter_snapshots_account_id_isolation(monkeypatch):
    from backend.services.risk_guard import _filter_snapshots_for_risk
    from backend.database.models import PortfolioSnapshot

    snap_acc1 = PortfolioSnapshot(id=1, broker="ctrader", account_id="12345", mode="live")
    snap_acc2 = PortfolioSnapshot(id=2, broker="ctrader", account_id="99999", mode="live")

    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("ACTIVE_BROKER", "ctrader")
    monkeypatch.setenv("CTRADER_ACCOUNT_ID", "12345")

    filtered = _filter_snapshots_for_risk([snap_acc1, snap_acc2])
    assert len(filtered) == 1
    assert filtered[0].account_id == "12345"


def test_filter_snapshots_split_book_ignores_demo_ctrader(monkeypatch):
    from backend.services.risk_guard import _filter_snapshots_for_risk, latest_snapshot_for_risk

    snap_ctrader = PortfolioSnapshot(
        id=1,
        broker="ctrader",
        mode="live",
        total_value=150.0,
        cash=150.0,
        timestamp=datetime.now(timezone.utc),
    )
    snap_binance = PortfolioSnapshot(
        id=2,
        broker="binance_futures",
        mode="live",
        total_value=918.0,
        cash=900.0,
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("ACTIVE_BROKER", "ctrader")
    monkeypatch.setenv("CTRADER_ENV", "sandbox")
    monkeypatch.setenv("BINANCE_TESTNET", "false")

    filtered = _filter_snapshots_for_risk([snap_ctrader, snap_binance])
    assert [s.id for s in filtered] == [2]


def test_latest_snapshot_for_risk_skips_newer_demo_row(db_session, monkeypatch):
    from backend.services.risk_guard import latest_snapshot_for_risk

    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("ACTIVE_BROKER", "ctrader")
    monkeypatch.setenv("CTRADER_ENV", "sandbox")
    monkeypatch.setenv("BINANCE_TESTNET", "false")

    demo = PortfolioSnapshot(
        broker="ctrader",
        mode="live",
        total_value=150.0,
        cash=150.0,
        timestamp=datetime.now(timezone.utc),
    )
    live_bn = PortfolioSnapshot(
        broker="binance_futures",
        mode="live",
        total_value=918.0,
        cash=900.0,
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    db_session.add_all([demo, live_bn])
    db_session.commit()

    latest = latest_snapshot_for_risk(db_session)
    assert latest is not None
    assert latest.broker == "binance_futures"
    assert latest.total_value == 918.0


