from datetime import datetime, timezone

from backend.backtesting.venue_costs import (
    default_cost_config,
    estimate_round_trip,
    swap_multiplier,
    triple_swap_weekday,
    venue_costs_enabled,
)
from backend.backtesting_ctrader.venue_costs import (
    estimate_round_trip as ct_estimate,
    triple_swap_weekday as ct_triple,
)


def test_cost_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("BACKTEST_VENUE_COSTS", raising=False)
    assert venue_costs_enabled() is False


def test_triple_swap_wednesday_fx_friday_crypto():
    assert triple_swap_weekday("forex") == 2
    assert triple_swap_weekday("metal") == 2
    assert triple_swap_weekday("crypto") == 4
    assert ct_triple("crypto") == 4


def test_wednesday_fx_is_triple_friday_crypto_is_triple():
    wed = datetime(2026, 9, 16, 21, 0, tzinfo=timezone.utc)
    fri = datetime(2026, 9, 18, 21, 0, tzinfo=timezone.utc)
    thu = datetime(2026, 9, 17, 21, 0, tzinfo=timezone.utc)
    assert swap_multiplier(wed, "forex") == 3
    assert swap_multiplier(fri, "crypto") == 3
    assert swap_multiplier(thu, "forex") == 1
    assert swap_multiplier(fri, "forex") == 1


def test_binance_round_trip_includes_funding_for_longs():
    costs = estimate_round_trip(
        venue="binance",
        symbol="BTCUSDC",
        notional=10_000.0,
        side="BUY",
        hours_held=8.0,
        maker_entry=True,
        maker_exit=False,
    )
    assert costs["entry_fee"] > 0
    assert costs["exit_fee"] > costs["entry_fee"]  # taker exit vs maker entry
    assert costs["funding"] > 0
    assert costs["swap"] == 0.0
    assert costs["total"] == (
        costs["entry_fee"] + costs["exit_fee"] + costs["spread"] + costs["funding"]
    )


def test_ctrader_round_trip_counts_wednesday_triple_swap():
    wed = datetime(2026, 9, 16, 21, 0, tzinfo=timezone.utc)
    costs = ct_estimate(
        venue="ctrader",
        symbol="EURUSD",
        notional=100_000.0,
        side="BUY",
        lots=1.0,
        pip_value_per_lot=10.0,
        nights=[wed],
    )
    cfg = default_cost_config("ctrader", "EURUSD")
    assert costs["swap"] == cfg.swap_long_pips * 3 * 10.0
    assert costs["funding"] == 0.0
    assert costs["total"] > 0


def test_short_crypto_on_ic_uses_friday_triple():
    fri = datetime(2026, 9, 18, 21, 0, tzinfo=timezone.utc)
    costs = estimate_round_trip(
        venue="ctrader",
        symbol="BTCUSDT",
        notional=50_000.0,
        side="SELL",
        lots=0.1,
        pip_value_per_lot=10.0,
        nights=[fri],
    )
    cfg = default_cost_config("ctrader", "BTCUSDT")
    assert costs["swap"] == cfg.swap_short_pips * 3 * 0.1 * 10.0
