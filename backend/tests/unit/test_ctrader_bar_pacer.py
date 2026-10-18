"""Unit tests for cTrader trendbar pacing, cooldown backoff, and scan rotation."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from backend.services.ctrader_bar_pacer import (
    DEFAULT_RATE_LIMIT_COOLDOWN_SEC,
    TRENDBAR_CACHE_TTL_SEC,
    TrendbarRequestPacer,
    fair_symbol_order,
    rate_limit_cooldown_sec,
    rotate_symbols,
)
from backend.services.ctrader_service import CTraderService


MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
CROSSES = ["EURGBP", "EURJPY", "GBPJPY", "EURAUD", "GBPAUD", "NZDJPY", "CHFJPY", "AUDJPY", "CADJPY"]
METALS = ["XAUUSD", "XAGUSD"]
FX_BOOK = MAJORS + ["NZDUSD"] + CROSSES + METALS


class _FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def test_pacer_allows_burst_then_waits_to_stay_at_four_per_sec():
    clock = _FakeClock()
    pacer = TrendbarRequestPacer(rate_per_sec=4.0, monotonic=clock.monotonic, sleep=clock.sleep)
    waits = [pacer.acquire() for _ in range(5)]
    assert waits[:4] == [0.0, 0.0, 0.0, 0.0]
    assert abs(waits[4] - 0.25) < 1e-9
    assert clock.sleeps == [0.25]


def test_pacer_refills_after_idle_second():
    clock = _FakeClock()
    pacer = TrendbarRequestPacer(rate_per_sec=4.0, monotonic=clock.monotonic, sleep=clock.sleep)
    for _ in range(4):
        pacer.acquire()
    clock.t += 1.0
    assert pacer.acquire() == 0.0
    assert clock.sleeps == []


def test_rate_limit_cooldown_defaults_and_backoff():
    assert DEFAULT_RATE_LIMIT_COOLDOWN_SEC == 45.0
    assert rate_limit_cooldown_sec(1, base=45) == 45.0
    assert rate_limit_cooldown_sec(2, base=45) == 90.0
    assert rate_limit_cooldown_sec(3, base=45) == 180.0
    assert rate_limit_cooldown_sec(8, base=45) == 300.0  # cap


def test_apply_rate_limit_cooldown_backs_off_until_success(monkeypatch):
    monkeypatch.setenv("CTRADER_RATE_LIMIT_COOLDOWN_SEC", "45")
    svc = CTraderService()
    first = svc.apply_rate_limit_cooldown(now=1_000.0)
    second = svc.apply_rate_limit_cooldown(now=1_045.0)
    assert first == 45.0
    assert second == 90.0
    assert svc._rate_limited_until == 1_045.0 + 90.0
    svc.note_trendbar_success("EURUSD", now=1_100.0)  # still inside cooldown
    assert svc._rate_limit_strikes == 2
    svc.note_trendbar_success("EURUSD", now=1_200.0)  # after cooldown
    third = svc.apply_rate_limit_cooldown(now=1_201.0)
    assert third == 45.0


def test_rotate_symbols_changes_start_each_scan():
    first = rotate_symbols(FX_BOOK, 0)
    second = rotate_symbols(FX_BOOK, 1)
    assert first[0] == "EURUSD"
    assert second[0] == "GBPUSD"
    assert set(first) == set(FX_BOOK)
    assert set(second) == set(FX_BOOK)


def test_fair_order_serves_starved_crosses_before_served_majors():
    last_ok = {sym: 1_000.0 for sym in MAJORS}
    ordered = fair_symbol_order(FX_BOOK + ["BTCUSDT"], last_ok, cursor=0)
    starved = ["NZDUSD"] + CROSSES + METALS
    assert ordered[: len(starved)] == starved
    assert ordered[len(starved) : len(starved) + len(MAJORS)] == MAJORS
    assert ordered[-1] == "BTCUSDT"


def test_fair_order_normalizes_symbol_case():
    last_ok = {"eurusd": 1_000.0}
    ordered = fair_symbol_order(["EURUSD", "EURGBP"], last_ok, cursor=0)
    assert ordered[0] == "EURGBP"
    assert ordered[1] == "EURUSD"


def test_fair_order_two_cold_scans_do_not_start_on_the_same_symbol():
    a = fair_symbol_order(FX_BOOK, {}, cursor=0)
    b = fair_symbol_order(FX_BOOK, {}, cursor=1)
    assert a[0] == "EURUSD"
    assert b[0] == "GBPUSD"


def test_order_symbols_for_scan_advances_cursor_and_unstarves():
    svc = CTraderService()
    first = svc.order_symbols_for_scan(FX_BOOK)
    assert first[0] == "EURUSD"
    svc.note_trendbar_success("EURUSD", now=50.0)
    svc.note_trendbar_success("GBPUSD", now=50.0)
    svc.note_trendbar_success("USDJPY", now=50.0)
    svc.note_trendbar_success("AUDUSD", now=50.0)
    svc.note_trendbar_success("USDCAD", now=50.0)
    svc.note_trendbar_success("USDCHF", now=50.0)
    second = svc.order_symbols_for_scan(FX_BOOK)
    assert second[0] == "NZDUSD"
    assert set(second[-6:]) == set(MAJORS)


def test_ttl_is_reserved_before_pacer_sleep():
    """A second caller must not send while the first is waiting on the token bucket."""
    clock = _FakeClock()
    svc = CTraderService()
    svc._dry_run = False
    svc._connected = True
    svc._authenticated = True
    svc._symbol_ids["EURUSD"] = 1
    svc._trendbar_cache.clear()
    svc._trendbar_last_req.clear()
    sends = {"n": 0}

    def _sleep(seconds: float) -> None:
        assert "EURUSD_5" in svc._trendbar_last_req
        clock.sleeps.append(seconds)
        clock.t += seconds

    svc._bar_pacer = TrendbarRequestPacer(rate_per_sec=4.0, monotonic=clock.monotonic, sleep=_sleep)
    for _ in range(4):
        svc._bar_pacer.acquire()

    class FakeProtocol:
        def _send(self, req, ptype):
            sends["n"] += 1
            waiter = svc._trendbar_events.get("EURUSD_5")
            svc._trendbar_cache["EURUSD_5"] = [{
                "timestamp": 1, "time": 1, "open": 1.0, "high": 1.0,
                "low": 1.0, "close": 1.0, "volume": 1.0,
            }]
            waiter.set()

    svc._protocol = FakeProtocol()
    reactor = MagicMock()
    reactor.callFromThread.side_effect = lambda fn: fn()
    with patch.dict(
        "sys.modules",
        {
            "twisted": MagicMock(),
            "twisted.internet": MagicMock(reactor=reactor),
            "ctrader_open_api": MagicMock(),
            "ctrader_open_api.messages": MagicMock(),
        },
    ):
        svc.get_trendbars("EURUSD", "M5", count=40)
        # TTL already reserved — this must not send again.
        svc.get_trendbars("EURUSD", "M5", count=40)

    assert sends["n"] == 1
    assert clock.sleeps == [0.25]


def test_get_trendbars_sunday_window_covers_weekend_close():
    sunday = datetime(2026, 10, 18, 22, 0, tzinfo=timezone.utc)
    now = sunday.timestamp()
    svc = CTraderService()
    svc._dry_run = False
    svc._connected = True
    svc._authenticated = True
    svc._symbol_ids["EURUSD"] = 1
    svc._trendbar_cache.clear()
    svc._trendbar_last_req.clear()
    captured: dict[str, int] = {}

    class FakeProtocol:
        def _send(self, req, ptype):
            captured["from"] = req.fromTimestamp
            captured["to"] = req.toTimestamp
            waiter = svc._trendbar_events.get("EURUSD_5")
            svc._trendbar_cache["EURUSD_5"] = [{
                "timestamp": 1, "time": 1, "open": 1.0, "high": 1.0,
                "low": 1.0, "close": 1.0, "volume": 1.0,
            }]
            waiter.set()

    svc._protocol = FakeProtocol()
    reactor = MagicMock()
    reactor.callFromThread.side_effect = lambda fn: fn()
    with patch("backend.services.ctrader_service.time.time", return_value=now), patch.dict(
        "sys.modules",
        {
            "twisted": MagicMock(),
            "twisted.internet": MagicMock(reactor=reactor),
            "ctrader_open_api": MagicMock(),
            "ctrader_open_api.messages": MagicMock(),
        },
    ):
        svc.get_trendbars("EURUSD", "M5", count=40)

    span = captured["to"] - captured["from"]
    expected = CTraderService.trendbar_lookback_ms(
        5, 40, now_ms=int(now * 1000), symbol="EURUSD"
    )
    assert span == expected
    assert span >= 48 * 3600 * 1000
    assert span < 7 * 86400 * 1000
    assert TRENDBAR_CACHE_TTL_SEC == 45.0
