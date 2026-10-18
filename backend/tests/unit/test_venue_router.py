"""Asset-class → venue table. Defaults match today's hard-coded routing."""

from backend.services.venue_router import (
    describe_venue_table,
    venue_for_asset_class,
    venue_for_symbol,
)


def test_defaults_route_crypto_to_binance_and_fx_metals_to_ctrader():
    assert venue_for_symbol("BTCUSDC") == "binance_futures"
    assert venue_for_symbol("ETHUSDT") == "binance_futures"
    assert venue_for_symbol("EURUSD") == "ctrader"
    assert venue_for_symbol("XAUUSD") == "ctrader"
    assert venue_for_symbol("XAGUSD") == "ctrader"


def test_aliases_normalize_to_broker_names(monkeypatch):
    monkeypatch.setenv("VENUE_CRYPTO", "binance")
    monkeypatch.setenv("VENUE_FOREX", "ic")
    monkeypatch.setenv("VENUE_METALS", "icmarkets")
    assert venue_for_asset_class("crypto") == "binance_futures"
    assert venue_for_asset_class("forex") == "ctrader"
    assert venue_for_asset_class("metal") == "ctrader"


def test_env_can_reroute_crypto_to_ctrader_without_touching_fx(monkeypatch):
    monkeypatch.setenv("VENUE_CRYPTO", "ctrader")
    assert venue_for_symbol("BTCUSDT") == "ctrader"
    assert venue_for_symbol("GBPUSD") == "ctrader"


def test_unknown_venue_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("VENUE_CRYPTO", "not-a-broker")
    assert venue_for_asset_class("crypto") == "binance_futures"


def test_describe_table_lists_all_asset_classes():
    table = describe_venue_table()
    assert table["crypto"] == "binance_futures"
    assert table["forex"] == "ctrader"
    assert table["metal"] == "ctrader"
    assert set(table) >= {"crypto", "forex", "metal", "oil", "index", "stock"}
