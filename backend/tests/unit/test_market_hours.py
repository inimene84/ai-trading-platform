from datetime import datetime, timezone

from backend.services.market_hours import is_crypto_symbol, is_venue_open


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
