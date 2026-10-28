"""Task 1.2: QuantumAIStrategy may only trade NVDA / MSFT / AAPL."""

from jesse_quant.equity_allowlist import (
    ALLOWED_EQUITY_BASES,
    equity_base,
    is_allowed_equity_symbol,
)


def test_allowed_equity_bases_are_us_tech_only():
    assert ALLOWED_EQUITY_BASES == {"NVDA", "MSFT", "AAPL"}


def test_equity_base_parses_jesse_and_binance_names():
    assert equity_base("NVDA-USD") == "NVDA"
    assert equity_base("AAPL") == "AAPL"
    assert equity_base("MSFT-USDT") == "MSFT"
    assert equity_base("BTC-USDT") == "BTC"
    assert equity_base("BTCUSDT") == "BTC"
    assert equity_base("GBPUSD") == "GBP"


def test_crypto_and_fx_are_rejected():
    assert is_allowed_equity_symbol("NVDA-USD") is True
    assert is_allowed_equity_symbol("MSFT") is True
    assert is_allowed_equity_symbol("AAPL-USDT") is True
    assert is_allowed_equity_symbol("BTC-USDT") is False
    assert is_allowed_equity_symbol("ETHUSDT") is False
    assert is_allowed_equity_symbol("EURUSD") is False
    assert is_allowed_equity_symbol("GBPUSD") is False
    assert is_allowed_equity_symbol("XAUUSD") is False
