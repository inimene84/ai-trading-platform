"""Spotware ProtoOATrendbarPeriod enum, cache merge, display-only forming bar.

Source of truth: github.com/spotware/openapi-proto-messages
OpenApiModelMessages.proto (enum ProtoOATrendbarPeriod, message ProtoOATrendbar)
and OpenApiMessages.proto (ProtoOASpotEvent.trendbar, ProtoOASubscribeLiveTrendbarReq).
"""
import time
from types import SimpleNamespace

from backend.services.ctrader_service import CTraderService

SPOTWARE_ENUM = {"M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M10": 6, "M15": 7,
                 "M30": 8, "H1": 9, "H4": 10, "H12": 11, "D1": 12, "W1": 13, "MN1": 14}


def test_period_map_matches_spotware_enum():
    m = CTraderService.PERIOD_MAP
    for name, val in SPOTWARE_ENUM.items():
        assert m[name] == val, name
    assert m["1H"] == 9 and m["15M"] == 7 and m["4H"] == 10 and m["1D"] == 12 and m["5M"] == 5


def test_period_seconds_follow_enum():
    s = CTraderService.PERIOD_SECONDS
    assert [s[i] for i in range(1, 15)] == [60, 120, 180, 240, 300, 600, 900, 1800,
                                            3600, 14400, 43200, 86400, 604800, 2592000]


def test_merge_keeps_longer_history_when_short_reply_arrives():
    svc = CTraderService()
    long = [{"time": 1000 + i * 300, "close": 1.0 + i} for i in range(50)]
    svc._merge_trendbar_cache("EURUSD_5", long)
    svc._merge_trendbar_cache("EURUSD_5", [{"time": long[-1]["time"], "close": 9.9}])
    cached = svc._trendbar_cache["EURUSD_5"]
    assert len(cached) == 50 and cached[-1]["close"] == 9.9


def test_forming_bar_from_fresh_spot_bid():
    svc = CTraderService()
    now = int(time.time())
    bucket = now // 300 * 300
    bars = [{"time": bucket - 300, "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15}]
    svc._last_spots["EURUSD"] = {"bid": 1.16, "ask": 1.1601, "timestamp": now * 1000}
    out = svc.append_forming_bar("EURUSD", "M5", bars)
    assert len(out) == 2 and out[-1]["time"] == bucket and out[-1]["forming"] is True
    assert out[-1]["open"] == 1.15 and out[-1]["close"] == 1.16
    assert out[-1]["forming_source"] == "spot_bid"
    assert bars[-1].get("forming") is None  # input untouched


def test_forming_bar_prefers_spotware_live_trendbar():
    svc = CTraderService()
    now = int(time.time())
    bucket = now // 3600 * 3600
    tb = SimpleNamespace(period=9, low=115000, deltaOpen=200, deltaHigh=900, deltaClose=500,
                         volume=42, utcTimestampInMinutes=bucket // 60)
    svc._store_live_trendbars("EURUSD", [tb])
    bars = [{"time": bucket - 3600, "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15}]
    out = svc.append_forming_bar("EURUSD", "1H", bars)
    assert len(out) == 2
    fb = out[-1]
    assert fb["forming_source"] == "spotware_live_trendbar" and fb["time"] == bucket
    assert fb["low"] == 1.15 and fb["open"] == 1.152 and fb["high"] == 1.159 and fb["close"] == 1.155


def test_forming_bar_skipped_when_spot_stale_or_period_long():
    svc = CTraderService()
    bars = [{"time": 0, "close": 1.0}]
    svc._last_spots["EURUSD"] = {"bid": 1.16, "timestamp": (int(time.time()) - 600) * 1000}
    assert svc.append_forming_bar("EURUSD", "M5", bars) == bars
    svc._last_spots["EURUSD"]["timestamp"] = int(time.time()) * 1000
    assert svc.append_forming_bar("EURUSD", "1W", bars) == bars


def test_forming_bucket_anchored_to_broker_grid():
    # H4 bars at 21:00/01:00/05:00 UTC; epoch flooring would give 00:00.
    sun_21 = 1792357200  # 2026-10-18 21:00 UTC
    now = sun_21 + 4 * 3600 + 13 * 60  # Mon 01:13 UTC
    assert CTraderService.forming_bucket_start(now, 14400, sun_21) == sun_21 + 14400
    # D1 at 21:00 across a weekend gap: Thu 21:00 closed -> Sun 21:00 forming.
    thu_21 = sun_21 - 3 * 86400
    assert CTraderService.forming_bucket_start(now, 86400, thu_21) == sun_21
    assert CTraderService.forming_bucket_start(now, 3600, 0) == now // 3600 * 3600


def test_h4_live_bar_gets_spot_overlay_off_epoch_grid():
    svc = CTraderService()
    now = int(time.time())
    last_closed = (now // 3600 * 3600) - 3 * 3600  # off-epoch H4 grid
    bucket = last_closed + 14400
    if bucket > now:
        bucket -= 14400
        last_closed -= 14400
    tb = SimpleNamespace(period=10, low=115000, deltaOpen=200, deltaHigh=900, deltaClose=500,
                         volume=1, utcTimestampInMinutes=bucket // 60)
    svc._store_live_trendbars("EURUSD", [tb])
    svc._last_spots["EURUSD"] = {"bid": 1.1599, "timestamp": now * 1000}
    out = svc.append_forming_bar("EURUSD", "4H", [{"time": last_closed, "close": 1.15}])
    assert out[-1]["time"] == bucket and out[-1]["close"] == 1.1599


def test_d1_forming_bar_from_bid_on_broker_grid():
    svc = CTraderService()
    now = int(time.time())
    forming = (now + 3 * 3600) // 86400 * 86400 - 3 * 3600  # latest 21:00 UTC <= now
    last_closed = forming - 86400
    svc._last_spots["EURUSD"] = {"bid": 1.2, "timestamp": now * 1000}
    out = svc.append_forming_bar("EURUSD", "1D", [{"time": last_closed, "close": 1.1}])
    assert out[-1]["forming_source"] == "spot_bid"
    assert out[-1]["time"] == last_closed + 86400 and out[-1]["open"] == 1.1
