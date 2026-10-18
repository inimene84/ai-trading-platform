from datetime import datetime, timezone

from backend.services.market_hours import (
    entry_cutoff_blocks,
    evaluate_entry_cutoff,
    is_crypto_symbol,
    is_daily_break,
    is_venue_open,
    time_until_venue_close,
)


def test_crypto_always_open_on_weekend():
    saturday = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)  # Saturday
    for symbol in ("BTCUSDT", "ETHUSDC", "BTC-USD", "SOL-USD"):
        assert is_crypto_symbol(symbol) is True
        assert is_venue_open(symbol, saturday) is True


def test_forex_and_metals_closed_saturday():
    saturday = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
    assert is_venue_open("EURUSD", saturday) is False
    assert is_venue_open("XAUUSD", saturday) is False


def test_forex_open_wednesday():
    wednesday = datetime(2026, 9, 16, 14, 0, tzinfo=timezone.utc)
    assert is_venue_open("EURUSD", wednesday) is True
    assert is_venue_open("XAUUSD", wednesday) is True


def test_forex_closed_friday_evening():
    friday_late = datetime(2026, 9, 11, 21, 30, tzinfo=timezone.utc)
    assert is_venue_open("GBPUSD", friday_late) is False


def test_forex_opens_sunday_evening():
    sunday_open = datetime(2026, 9, 13, 21, 0, tzinfo=timezone.utc)
    sunday_closed = datetime(2026, 9, 13, 20, 59, tzinfo=timezone.utc)
    assert is_venue_open("EURUSD", sunday_open) is True
    assert is_venue_open("EURUSD", sunday_closed) is False


def test_daily_break_weekday_not_weekend():
    during = datetime(2026, 9, 16, 21, 15, tzinfo=timezone.utc)  # Wed 21:15
    before = datetime(2026, 9, 16, 20, 50, tzinfo=timezone.utc)
    saturday = datetime(2026, 9, 12, 21, 15, tzinfo=timezone.utc)
    assert is_daily_break(during) is True
    assert is_daily_break(before) is False
    assert is_daily_break(saturday) is False
    # Existing weekend predicate stays unchanged during the weekday break.
    assert is_venue_open("EURUSD", during) is True


def test_time_until_venue_close_crypto_is_none():
    assert time_until_venue_close("BTCUSDC") is None


def test_time_until_friday_close_and_daily_break():
    friday_morning = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)
    remaining = time_until_venue_close("EURUSD", friday_morning)
    assert remaining is not None
    assert 8.9 * 3600 < remaining.total_seconds() <= 9.0 * 3600
    wed_morning = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
    to_break = time_until_venue_close("XAUUSD", wed_morning)
    assert to_break is not None
    assert abs(to_break.total_seconds() - 9 * 3600) < 1


def test_entry_cutoff_default_shadow_does_not_block():
    friday_late = datetime(2026, 9, 11, 20, 45, tzinfo=timezone.utc)
    assert entry_cutoff_blocks("EURUSD", friday_late, minutes=30) is True
    decision = evaluate_entry_cutoff("EURUSD", friday_late, minutes=30, mode="shadow")
    assert decision["would_block"] is True
    assert decision["blocked"] is False
    enforced = evaluate_entry_cutoff("EURUSD", friday_late, minutes=30, mode="enforce")
    assert enforced["blocked"] is True
    off = evaluate_entry_cutoff("EURUSD", friday_late, minutes=30, mode="off")
    assert off["blocked"] is False
    assert evaluate_entry_cutoff("BTCUSDT", friday_late, mode="enforce")["blocked"] is False
