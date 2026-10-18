"""Concurrent-position cap uses the stricter knob and the live book."""

from types import SimpleNamespace

from backend.services.position_caps import (
    binance_book_at_cap,
    effective_binance_position_cap,
    live_binance_position_count,
)


def test_effective_cap_is_the_stricter_knob():
    cfg = SimpleNamespace(max_positions=20, max_open_positions=20, max_binance_positions=2)
    assert effective_binance_position_cap(cfg) == 2
    cfg.max_positions = 2
    cfg.max_binance_positions = 10
    assert effective_binance_position_cap(cfg) == 2


def test_live_count_ignores_flat_and_counts_symbols():
    positions = [
        {"symbol": "ETHUSDC", "quantity": -0.02},
        {"symbol": "SOLUSDC", "positionAmt": 0.5},
        {"symbol": "ADAUSDC", "quantity": 0},
        {"symbol": "ETHUSDC", "quantity": -0.01},
    ]
    assert live_binance_position_count(positions) == 2


def test_book_at_cap_uses_max_of_db_and_live():
    cfg = SimpleNamespace(max_positions=2, max_open_positions=20, max_binance_positions=10)
    at_cap, observed, cap = binance_book_at_cap(
        cfg,
        db_open_count=1,
        live_positions=[
            {"symbol": "ETHUSDC", "quantity": -0.01},
            {"symbol": "SOLUSDC", "quantity": -0.2},
        ],
    )
    assert cap == 2
    assert observed == 2
    assert at_cap is True


def test_book_under_cap_when_both_counts_low():
    cfg = SimpleNamespace(max_positions=2, max_open_positions=2, max_binance_positions=2)
    at_cap, observed, cap = binance_book_at_cap(
        cfg,
        db_open_count=1,
        live_positions=[{"symbol": "ETHUSDC", "quantity": -0.01}],
    )
    assert at_cap is False
    assert observed == 1
    assert cap == 2
