from datetime import datetime, timedelta, timezone

from backend.services.market_hours import (
    closed_ms_in_range,
    entry_cutoff_blocks,
    evaluate_entry_cutoff,
    is_closed_for_lookback,
    is_crypto_symbol,
    is_daily_break,
    is_venue_open,
    market_hours_lookback_ms,
    time_until_venue_close,
)


def test_crypto_always_open_on_weekend():
    saturday = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)  # Saturday
    for symbol in ("BTCUSDT", "ETHUSDC", "BTC-USD", "SOL-USD", "ETHUSD", "BTCUSD"):
        assert is_crypto_symbol(symbol) is True
        assert is_venue_open(symbol, saturday) is True


def test_ic_crypto_respects_daily_break_not_weekend():
    saturday = datetime(2026, 9, 12, 21, 15, tzinfo=timezone.utc)
    wed_break = datetime(2026, 9, 16, 21, 15, tzinfo=timezone.utc)
    wed_open = datetime(2026, 9, 16, 14, 0, tzinfo=timezone.utc)
    assert is_venue_open("ETHUSD", saturday) is True
    assert entry_cutoff_blocks("ETHUSD", saturday, minutes=30) is False
    assert entry_cutoff_blocks("ETHUSD", wed_break, minutes=30) is True
    assert entry_cutoff_blocks("ETHUSD", wed_open, minutes=30) is False
    # Binance perps still never take the IC daily break.
    assert entry_cutoff_blocks("ETHUSDT", wed_break, minutes=30) is False


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


def _ms(hours: float) -> int:
    return int(hours * 3600 * 1000)


def test_weekend_closed_ms_is_forty_eight_hours():
    friday_close = datetime(2026, 10, 16, 21, 0, tzinfo=timezone.utc)
    sunday_open = datetime(2026, 10, 18, 21, 0, tzinfo=timezone.utc)
    assert closed_ms_in_range(friday_close, sunday_open, "EURUSD") == _ms(48)
    assert is_closed_for_lookback("EURUSD", datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)) is True
    assert is_closed_for_lookback("BTCUSDT", datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)) is False


def test_lookback_sunday_after_open_extends_by_weekend_close():
    """40×M5 needs 3.5h of prints; 1h after Sunday 21:00 that window is mostly closed."""
    sunday = datetime(2026, 10, 18, 22, 0, tzinfo=timezone.utc)
    needed = (40 + 2) * 300 * 1000
    ms = market_hours_lookback_ms(needed, now=sunday, symbol="EURUSD")
    assert ms == needed + _ms(48)
    start = sunday - timedelta(milliseconds=ms)
    assert start == datetime(2026, 10, 16, 18, 30, tzinfo=timezone.utc)
    assert closed_ms_in_range(start, sunday, "EURUSD") == _ms(48)


def test_lookback_monday_open_does_not_extend_once_session_covers_count():
    """Monday 01:00 UTC is 4h after Sunday open — a 3.5h M5 window stays inside the session."""
    monday = datetime(2026, 10, 19, 1, 0, tzinfo=timezone.utc)
    needed = (40 + 2) * 300 * 1000
    ms = market_hours_lookback_ms(needed, now=monday, symbol="EURUSD")
    assert ms == needed


def test_lookback_monday_morning_midweek_is_count_bounded():
    monday_late = datetime(2026, 10, 19, 10, 0, tzinfo=timezone.utc)
    needed = (40 + 2) * 300 * 1000
    assert market_hours_lookback_ms(needed, now=monday_late, symbol="EURUSD") == needed


def test_lookback_extends_across_weekday_daily_break():
    monday_break = datetime(2026, 10, 19, 21, 30, tzinfo=timezone.utc)
    needed = (40 + 2) * 300 * 1000
    ms = market_hours_lookback_ms(needed, now=monday_break, symbol="XAUUSD")
    assert ms == needed + _ms(0.5)
    assert is_closed_for_lookback("XAUUSD", monday_break) is True
    assert is_venue_open("XAUUSD", monday_break) is True  # entry gating unchanged


def test_custom_friday_daily_break_does_not_double_count_weekend(monkeypatch):
    monkeypatch.setenv("CTRADER_DAILY_BREAK_START_UTC", "20:00")
    monkeypatch.setenv("CTRADER_DAILY_BREAK_END_UTC", "22:00")
    start = datetime(2026, 10, 16, 19, 0, tzinfo=timezone.utc)
    end = datetime(2026, 10, 16, 23, 0, tzinfo=timezone.utc)
    # Fri 20:00–21:00 daily break + Fri 21:00–23:00 weekend = 3h, not 4h.
    assert closed_ms_in_range(start, end, "EURUSD") == _ms(3)


def test_lookback_crypto_ignores_weekend():
    sunday = datetime(2026, 10, 18, 22, 0, tzinfo=timezone.utc)
    needed = (40 + 2) * 300 * 1000
    assert market_hours_lookback_ms(needed, now=sunday, symbol="BTCUSDT") == needed
