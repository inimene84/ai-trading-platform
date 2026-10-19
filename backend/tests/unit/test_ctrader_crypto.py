"""IC Markets demo crypto add-on: flags default off, mapping, catalog, routing."""

from backend.services.ctrader_crypto import (
    accepted_ctrader_crypto_symbols,
    broker_for_scan_symbol,
    crypto_base_asset,
    ctrader_crypto_enabled,
    ctrader_crypto_orders_allowed,
    ctrader_trade_mode,
    is_ic_crypto_symbol,
    map_to_ic_crypto,
    max_spread_pips_for_symbol,
    reset_ctrader_crypto_catalog_cache,
    validate_ctrader_crypto_catalog,
)
from backend.services.multi_asset_bars import classify_symbol
from backend.services.venue_router import venue_for_symbol


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("CTRADER_CRYPTO_ENABLED", raising=False)
    assert ctrader_crypto_enabled() is False
    assert ctrader_crypto_orders_allowed() is False


def test_orders_allowed_only_when_enabled_and_demo(monkeypatch):
    monkeypatch.setenv("CTRADER_CRYPTO_ENABLED", "true")
    monkeypatch.setenv("CTRADER_ENV", "demo")
    assert ctrader_crypto_orders_allowed() is True
    monkeypatch.setenv("CTRADER_ENV", "live")
    assert ctrader_crypto_orders_allowed() is False
    monkeypatch.setenv("CTRADER_ENV", "demo")
    monkeypatch.setenv("CTRADER_CRYPTO_ENABLED", "false")
    assert ctrader_crypto_orders_allowed() is False


def test_scan_symbol_mapping():
    assert map_to_ic_crypto("ETHUSDT") == "ETHUSD"
    assert map_to_ic_crypto("ETHUSDC") == "ETHUSD"
    assert map_to_ic_crypto("BTC") == "BTCUSD"
    assert map_to_ic_crypto("SOL") == "SOLUSD"
    assert map_to_ic_crypto("XRP") == "XRPUSD"
    assert map_to_ic_crypto("ADA") == "ADAUSD"
    assert map_to_ic_crypto("BNB") == "BNBUSD"
    assert map_to_ic_crypto("LINK") == "LNKUSD"
    assert map_to_ic_crypto("AVAX") == "AVXUSD"
    assert map_to_ic_crypto("DOGE") == "DOGUSD"
    assert map_to_ic_crypto("EURUSD") is None


def test_ic_names_classify_as_crypto_not_forex():
    for sym in ("BTCUSD", "ETHUSD", "LNKUSD", "AVXUSD", "DOGUSD", "ADAUSD"):
        assert classify_symbol(sym) == "crypto"
        assert is_ic_crypto_symbol(sym) is True
    assert classify_symbol("EURUSD") == "forex"
    assert classify_symbol("USDCAD") == "forex"
    assert is_ic_crypto_symbol("ETHUSDT") is False


def test_venue_crypto_stays_binance_when_addon_on(monkeypatch):
    monkeypatch.setenv("VENUE_CRYPTO", "binance")
    monkeypatch.setenv("CTRADER_CRYPTO_ENABLED", "true")
    assert venue_for_symbol("ETHUSDT") == "binance_futures"
    assert venue_for_symbol("BTCUSDC") == "binance_futures"
    assert broker_for_scan_symbol("ETHUSDT") == "binance_futures"
    assert broker_for_scan_symbol("ETHUSD") == "ctrader"


def test_broker_for_scan_ic_name_stays_binance_when_flag_off(monkeypatch):
    monkeypatch.delenv("CTRADER_CRYPTO_ENABLED", raising=False)
    assert broker_for_scan_symbol("ETHUSD") == "binance_futures"


def test_catalog_skips_unknown_names(monkeypatch):
    reset_ctrader_crypto_catalog_cache()
    monkeypatch.setenv("CTRADER_CRYPTO_SYMBOLS", "ETHUSDT,NOTACOIN,SOL")
    accepted = validate_ctrader_crypto_catalog({"ETHUSD", "SOLUSD", "BTCUSD"})
    assert accepted == ["ETHUSD", "SOLUSD"]
    assert "NOTACOIN" not in accepted
    reset_ctrader_crypto_catalog_cache()


def test_accepted_fail_closed_without_live_catalog(monkeypatch):
    """Default IDs are not a catalog — no live ProtoOASymbolsListRes means empty."""
    reset_ctrader_crypto_catalog_cache()
    monkeypatch.setenv("CTRADER_CRYPTO_SYMBOLS", "BTC,ETH,ADA")
    accepted = accepted_ctrader_crypto_symbols()
    assert accepted == []
    reset_ctrader_crypto_catalog_cache()


def test_catalog_ignores_default_ids_not_in_live_book(monkeypatch):
    reset_ctrader_crypto_catalog_cache()
    monkeypatch.setenv("CTRADER_CRYPTO_SYMBOLS", "ETHUSDT,BTC")
    # Live book has FX only; default BTCUSD:36 must not count as accepted.
    accepted = validate_ctrader_crypto_catalog({"EURUSD", "XAUUSD"})
    assert accepted == []
    reset_ctrader_crypto_catalog_cache()


def test_crypto_base_overlap_keys():
    assert crypto_base_asset("ETHUSDT") == "ETH"
    assert crypto_base_asset("ETHUSD") == "ETH"
    assert crypto_base_asset("LINKUSDT") == "LINK"
    assert crypto_base_asset("LNKUSD") == "LINK"
    assert crypto_base_asset("AVXUSD") == "AVAX"
    assert crypto_base_asset("DOGUSD") == "DOGE"
    assert crypto_base_asset("EURUSD") is None


def test_spread_limits_per_asset_class(monkeypatch):
    monkeypatch.delenv("CTRADER_MAX_SPREAD_FOREX", raising=False)
    monkeypatch.delenv("CTRADER_MAX_SPREAD_METALS", raising=False)
    monkeypatch.delenv("CTRADER_MAX_SPREAD_CRYPTO", raising=False)
    assert max_spread_pips_for_symbol("EURUSD") == 2.0
    assert max_spread_pips_for_symbol("XAUUSD") == 40.0
    assert max_spread_pips_for_symbol("ETHUSD") == 80.0
    monkeypatch.setenv("CTRADER_MAX_SPREAD_XAUUSD", "55")
    assert max_spread_pips_for_symbol("XAUUSD") == 55.0


def test_trade_mode_follows_ctrader_env(monkeypatch):
    monkeypatch.setenv("CTRADER_ENV", "demo")
    assert ctrader_trade_mode() == "demo"
    monkeypatch.setenv("CTRADER_ENV", "live")
    assert ctrader_trade_mode() == "live"
    monkeypatch.setenv("CTRADER_ENV", "sandbox")
    assert ctrader_trade_mode() == "demo"
