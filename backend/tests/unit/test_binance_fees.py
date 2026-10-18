from backend.services.binance_fees import (
    DEFAULT_MAKER_RATE,
    DEFAULT_TAKER_RATE,
    classify_fill_role,
    clear_commission_cache,
    fetch_commission_rates,
    get_cached_commission_rates,
    parse_commission_rate_payload,
    store_commission_rates,
    summarize_fills,
)


def setup_function():
    clear_commission_cache()


def test_parse_commission_rate_payload():
    maker, taker = parse_commission_rate_payload({
        "makerCommissionRate": "0.000180",
        "takerCommissionRate": "0.000450",
    })
    assert maker == 0.000180
    assert taker == 0.000450


def test_cache_round_trip():
    store_commission_rates("BTCUSDC", 0.0001, 0.0004)
    cached = get_cached_commission_rates("BTCUSDC")
    assert cached == {"maker_rate": 0.0001, "taker_rate": 0.0004}


def test_fetch_uses_cache_without_client_call():
    store_commission_rates("ETHUSDC", 0.0002, 0.0005)

    class Boom:
        def futures_commission_rate(self, symbol):
            raise AssertionError("should not hit the API")

    rates = fetch_commission_rates(Boom(), "ETHUSDC")
    assert rates["maker_rate"] == 0.0002


def test_fetch_falls_back_to_defaults_on_error():
    class Fail:
        def futures_commission_rate(self, symbol):
            raise RuntimeError("timeout")

    rates = fetch_commission_rates(Fail(), "SOLUSDC")
    assert rates["maker_rate"] == DEFAULT_MAKER_RATE
    assert rates["taker_rate"] == DEFAULT_TAKER_RATE


def test_classify_fill_role():
    assert classify_fill_role({"maker": True}) == "maker"
    assert classify_fill_role({"maker": "false"}) == "taker"
    assert classify_fill_role({}) == "taker"


def test_summarize_fills_mixed_and_expected_commission():
    fills = [
        {"qty": 1.0, "price": 100.0, "commission": 0.02, "maker": True},
        {"qty": 1.0, "price": 100.0, "commission": 0.05, "maker": False},
    ]
    summary = summarize_fills(
        fills, rates={"maker_rate": 0.0002, "taker_rate": 0.0005},
    )
    assert summary["role"] == "mixed"
    assert summary["maker_fills"] == 1
    assert summary["taker_fills"] == 1
    assert summary["commission"] == 0.07
    assert summary["expected_commission"] == 0.07
