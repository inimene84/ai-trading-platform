#!/usr/bin/env python3
"""Tests for the multi-asset Jesse universe and candle row builder."""

import unittest
from datetime import datetime, timezone

import pandas as pd

from asset_universe import ALL_CLASSES, candle_timeframe_candidates, normalize_symbol, resolve_universe
from import_multi_asset import rows_from_ohlcv
from multi_asset_universe import universe_by_class


class TestUniverse(unittest.TestCase):
    def test_all_required_asset_classes_present(self):
        by_class = universe_by_class()
        for name in ALL_CLASSES:
            self.assertTrue(by_class[name], msg=f"{name} universe is empty")
        self.assertIn("BTC-USDT", by_class["crypto"])
        self.assertIn("EURUSD", by_class["forex"])
        self.assertIn("XAUUSD", by_class["metals"])
        self.assertIn("USOIL", by_class["minerals"])
        self.assertIn("AAPL", by_class["stocks"])
        self.assertEqual(len(resolve_universe("all")), sum(len(v) for v in by_class.values()))

    def test_yahoo_and_binance_sources_cover_non_jesse_dumps(self):
        extra = [r for r in resolve_universe("all") if r.source != "jesse_dump"]
        self.assertTrue(any(r.source == "yfinance" for r in extra))
        self.assertTrue(any(r.source == "binance" for r in extra))


class TestSymbolNormalization(unittest.TestCase):
    def test_forex_not_rewritten_as_usdt(self):
        self.assertEqual(normalize_symbol("EURUSD"), "EURUSD")
        self.assertEqual(normalize_symbol("XAUUSD"), "XAUUSD")

    def test_crypto_usdt_pair(self):
        self.assertEqual(normalize_symbol("BTCUSDT"), "BTC-USDT")
        self.assertEqual(normalize_symbol("BTC-USDT"), "BTC-USDT")

    def test_equity_passthrough(self):
        self.assertEqual(normalize_symbol("SPY"), "SPY")
        self.assertEqual(normalize_symbol("AAPL"), "AAPL")


class TestCandleHelpers(unittest.TestCase):
    def test_timeframe_candidates_prefer_1m(self):
        self.assertEqual(candle_timeframe_candidates("1h"), ("1m", "1h"))
        self.assertEqual(candle_timeframe_candidates("1m"), ("1m",))

    def test_rows_from_ohlcv_builds_jesse_tuples(self):
        idx = pd.DatetimeIndex([datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)])
        frame = pd.DataFrame(
            {"Open": [1.1], "High": [1.2], "Low": [1.0], "Close": [1.15], "Volume": [10.0]},
            index=idx,
        )
        rows = rows_from_ohlcv(frame, exchange="Yahoo Finance", symbol="EURUSD", timeframe="1h")
        self.assertEqual(len(rows), 1)
        _id, ts, o, c, h, low, vol, exchange, symbol, tf = rows[0]
        self.assertEqual(ts, int(idx[0].timestamp() * 1000))
        self.assertEqual(o, 1.1)
        self.assertEqual(c, 1.15)
        self.assertEqual(h, 1.2)
        self.assertEqual(low, 1.0)
        self.assertEqual(vol, 10.0)
        self.assertEqual(exchange, "Yahoo Finance")
        self.assertEqual(symbol, "EURUSD")
        self.assertEqual(tf, "1h")
        self.assertTrue(_id)


if __name__ == "__main__":
    unittest.main()
