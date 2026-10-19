"""Proposals 2026-10-19: count-aware trendbar cache, long-period lookback, FX batch."""
import asyncio
import time
from unittest.mock import patch

from backend.services.ctrader_service import CTraderService


def test_cache_shorter_than_request_is_not_fresh_unless_already_asked():
    svc = CTraderService()
    key = "EURUSD_9"
    cached = [{"time": i} for i in range(6)]
    assert svc._trendbar_cache_covers(key, cached, 4) is True
    assert svc._trendbar_cache_covers(key, cached, 200) is False
    svc._trendbar_last_req_count[key] = 400
    assert svc._trendbar_cache_covers(key, cached, 200) is True


def test_h4_and_d1_lookback_cover_120_bars():
    now_ms = int(time.time() * 1000)
    h4 = CTraderService.trendbar_lookback_ms(10, 120, now_ms=now_ms)
    d1 = CTraderService.trendbar_lookback_ms(12, 120, now_ms=now_ms)
    assert h4 >= 120 * 14400 * 1000
    assert d1 >= 120 * 86400 * 1000
    assert h4 <= CTraderService.TRENDBAR_LOOKBACK_CAP_LONG_MS
    assert d1 <= CTraderService.TRENDBAR_LOOKBACK_CAP_LONG_MS


def test_intraday_lookback_still_capped_at_seven_days():
    assert CTraderService.trendbar_lookback_ms(9, 500) == 7 * 86400 * 1000
    assert CTraderService.trendbar_lookback_cap_ms(5) == 7 * 86400 * 1000


def test_fx_batch_symbols_filtered_by_account_catalog(monkeypatch):
    from backend.services import feed_scheduler
    from backend.services.ctrader_service import ctrader_service
    monkeypatch.setattr(ctrader_service, "_symbol_ids", {"EURUSD": 1, "GBPUSD": 2, "USDJPY": 4}, raising=False)
    monkeypatch.setattr(type(ctrader_service), "is_connected", property(lambda self: True))
    assert feed_scheduler._fx_batch_symbols() == ["EURUSD", "GBPUSD", "USDJPY"]
    monkeypatch.setenv("KRONOS_BATCH_FX_ENABLED", "false")
    assert feed_scheduler._fx_batch_symbols() == []
