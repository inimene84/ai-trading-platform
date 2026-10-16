"""New-bar gate and per-symbol expectancy gate for the trading loop."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from backend.services.trading_loop import TradingLoopService, negative_expectancy_symbols


def _bars(ts: str):
    return [{"date": "2026-07-10T16:00:00", "close": 100.0},
            {"date": ts, "close": 101.0}]


# ── New-bar gate ────────────────────────────────────────────────────────────

def test_new_bar_gate_evaluates_first_sighting():
    loop = TradingLoopService()
    assert loop._should_evaluate_bar("ETHUSDT", _bars("2026-07-10T17:00:00")) is True


def test_new_bar_gate_blocks_same_bar_reevaluation():
    """15-min cycles on 1h bars: cycles 2-4 within the hour must not re-decide."""
    loop = TradingLoopService()
    bars = _bars("2026-07-10T17:00:00")
    assert loop._should_evaluate_bar("ETHUSDT", bars) is True
    assert loop._should_evaluate_bar("ETHUSDT", bars) is False
    assert loop._should_evaluate_bar("ETHUSDT", bars) is False


def test_new_bar_gate_reopens_on_new_bar():
    loop = TradingLoopService()
    assert loop._should_evaluate_bar("ETHUSDT", _bars("2026-07-10T17:00:00")) is True
    assert loop._should_evaluate_bar("ETHUSDT", _bars("2026-07-10T17:00:00")) is False
    assert loop._should_evaluate_bar("ETHUSDT", _bars("2026-07-10T18:00:00")) is True


def test_new_bar_gate_tracks_symbols_independently():
    loop = TradingLoopService()
    bars = _bars("2026-07-10T17:00:00")
    assert loop._should_evaluate_bar("ETHUSDT", bars) is True
    assert loop._should_evaluate_bar("SOLUSDT", bars) is True
    assert loop._should_evaluate_bar("ETHUSDT", bars) is False


def test_new_bar_gate_fails_open_without_timestamps():
    loop = TradingLoopService()
    bars = [{"close": 100.0}]
    assert loop._should_evaluate_bar("ETHUSDT", bars) is True
    assert loop._should_evaluate_bar("ETHUSDT", bars) is True


def test_bar_open_age_seconds_parses_iso():
    loop = TradingLoopService()
    age = loop._bar_open_age_seconds({"date": "2026-07-23T12:00:00+00:00"})
    assert age is not None
    assert age > 0


def test_stale_ws_age_threshold_helpers():
    """Frozen WS rings (last bar many hours old) must be detectable."""
    loop = TradingLoopService()
    fresh = loop._bar_open_age_seconds(
        {"date": datetime.now(timezone.utc).isoformat()}
    )
    assert fresh is not None and fresh < 60
    stale = loop._bar_open_age_seconds({"date": "2026-07-20T00:00:00+00:00"})
    assert stale is not None and stale > 2.5 * 3600



# ── Expectancy gate ─────────────────────────────────────────────────────────

def test_negative_expectancy_symbols_blocks_proven_losers():
    rows = [
        ("AVAXUSDT", 59, -8.08),   # loser, enough sample -> blocked
        ("SOLUSDT", 65, -8.04),    # loser, enough sample -> blocked
        ("ADAUSDT", 78, 12.66),    # winner -> kept
        ("LINKUSDT", 16, -4.92),   # loser but small sample -> kept
        ("ATOMUSDT", 25, 0.0),     # breakeven -> kept
    ]
    blocked = negative_expectancy_symbols(rows, min_trades=20)
    assert blocked == {"AVAXUSDT", "SOLUSDT"}


def test_negative_expectancy_symbols_handles_null_aggregates():
    rows = [("XUSDT", None, None), ("YUSDT", 30, None)]
    assert negative_expectancy_symbols(rows, min_trades=20) == set()


def test_expectancy_gate_never_blocks_open_positions():
    """A symbol with an open position must stay managed even if it bleeds."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from backend.database.models import Base, Trade

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    now = datetime.now(timezone.utc)
    for i in range(20):
        db.add(Trade(
            symbol="AVAXUSDT", direction="SELL", quantity=1, entry_price=10,
            status="closed", pnl=-0.4, strategy="combined", closed_at=now,
        ))
        db.add(Trade(
            symbol="SOLUSDT", direction="SELL", quantity=1, entry_price=10,
            status="closed", pnl=-0.4, strategy="combined", closed_at=now,
        ))
    db.add(Trade(
        symbol="AVAXUSDT", direction="SELL", quantity=1, entry_price=10,
        status="open", strategy="combined",
    ))
    db.commit()

    loop = TradingLoopService()
    loop.risk_config = MagicMock(
        symbol_expectancy_gate_enabled=True,
        symbol_expectancy_lookback_days=30,
        symbol_expectancy_min_trades=20,
        symbol_expectancy_probe_after_hours=72,
        symbol_expectancy_probe_cooldown_hours=24,
    )
    with patch("backend.services.trading_loop.SessionLocal", return_value=db):
        kept = loop._apply_expectancy_gate(["AVAXUSDT", "SOLUSDT", "ADAUSDT"])
    assert kept == ["AVAXUSDT", "ADAUSDT"]


def test_expectancy_gate_disabled_keeps_all():
    loop = TradingLoopService()
    loop.risk_config = MagicMock(symbol_expectancy_gate_enabled=False)
    assert loop._apply_expectancy_gate(["AVAXUSDT", "SOLUSDT"]) == ["AVAXUSDT", "SOLUSDT"]


def test_expectancy_gate_fails_open_on_db_error():
    loop = TradingLoopService()
    loop.risk_config = MagicMock(
        symbol_expectancy_gate_enabled=True,
        symbol_expectancy_lookback_days=30,
        symbol_expectancy_min_trades=20,
    )
    with patch("backend.services.trading_loop.SessionLocal", side_effect=RuntimeError("db down")):
        kept = loop._apply_expectancy_gate(["AVAXUSDT", "SOLUSDT"])
    assert kept == ["AVAXUSDT", "SOLUSDT"]
