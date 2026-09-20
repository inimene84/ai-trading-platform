"""
Unit tests for the Layered Signal Candidate Engine, Timing Windows, and REST Routes.
"""

import pytest
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from starlette.testclient import TestClient
from backend.main import app
from backend.services.signal_candidate_engine import (
    signal_candidate_engine,
    SignalCandidateEngine,
    TimingMode,
    CandidateStatus,
    CANDIDATE_TTL_SECONDS,
)
from backend.services.ctrader_service import CTraderService
from backend.services.multi_asset_bars import classify_symbol, tf_to_binance_interval


@pytest.fixture(autouse=True)
def isolate_ctrader_db_positions(monkeypatch):
    monkeypatch.setattr("backend.services.signal_candidate_engine.open_ctrader_db_symbols", lambda: set())
    monkeypatch.setattr("backend.services.signal_candidate_engine.count_open_ctrader_db_trades", lambda: 0)
    monkeypatch.setattr("backend.services.signal_candidate_engine.open_ctrader_db_positions", lambda: [])
    monkeypatch.setattr("backend.services.signal_candidate_engine.try_open_ctrader_db_positions", lambda: [])


@pytest.fixture
def client():
    return TestClient(app)


def test_feature_computation():
    """Verify feature calculation generates valid ATR, RSI, EMAs, and trends."""
    synthetic_bars = [
        {"close": 1.0800 + (i * 0.0002), "high": 1.0805 + (i * 0.0002), "low": 1.0795 + (i * 0.0002), "volume": 100}
        for i in range(25)
    ]
    features = signal_candidate_engine._compute_features(synthetic_bars)
    assert features["last_close"] > 1.0800
    assert features["atr"] > 0
    assert 0 <= features["rsi"] <= 100
    assert features["trend"] in ["BULLISH", "BEARISH", "NEUTRAL"]
    assert features["volatility_pct"] > 0


def test_feature_atr_clamped_when_bars_are_wrong_scale():
    """JPY-scale mismatch: ATR of ~20 on a 94.5 close must not produce a 142 TP."""
    synthetic_bars = [
        {"close": 94.54, "high": 114.54, "low": 74.54, "volume": 100}
        for _ in range(20)
    ]
    features = signal_candidate_engine._compute_features(synthetic_bars)
    assert features["atr"] <= features["last_close"] * 0.012 + 1e-9
    sl, tp = CTraderService.clamp_protective_prices(
        "NZDJPY",
        features["last_close"],
        features["last_close"] - 1.2 * features["atr"],
        features["last_close"] + 2.4 * features["atr"],
    )
    assert tp is not None and tp < 97.0
    assert sl is not None and sl > 92.0


def test_calculate_size_forex():
    """Verify position sizing uses pip-margin models on cTrader FX pairs."""
    sizing = signal_candidate_engine._calculate_size(
        symbol="EURUSD",
        entry_price=1.0850,
        stop_loss=1.0820,
        broker="ctrader"
    )
    assert "lots" in sizing
    assert sizing["lots"] >= 0.01
    assert sizing["risk_usd"] > 0
    assert sizing["stop_pips"] >= 5.0
    assert sizing["margin_required"] > 0


@pytest.mark.asyncio
async def test_scan_markets_generates_candidates():
    """Verify market scanning populates candidate signals with timing metadata."""
    universe = ["EURUSD", "GBPUSD"]
    candidates = await signal_candidate_engine.scan_markets(universe=universe, timeframe="M5")
    assert isinstance(candidates, list)
    for c in candidates:
        assert "id" in c
        assert c["symbol"] in universe
        assert c["direction"] in ["BUY", "SELL"]
        assert c["timing_mode"] in [
            TimingMode.PRE_EVENT,
            TimingMode.AT_RELEASE,
            TimingMode.POST_REACTION,
            TimingMode.BAR_CLOSE,
        ]
        assert c["earliest_exec_at"] > 0
        assert c["latest_exec_at"] >= c["earliest_exec_at"]
        assert c["sizing"]["lots"] > 0


def test_timing_window_ready_queue():
    """Verify signals inside execution window transition to READY."""
    now_ts = int(time.time())
    test_id = "test-sig-timing-01"
    signal_candidate_engine.candidates[test_id] = {
        "id": test_id,
        "symbol": "EURUSD",
        "broker": "ctrader",
        "strategy": "MOMENTUM_TREND_PULSE",
        "direction": "BUY",
        "entry_price": 1.0850,
        "stop_loss": 1.0820,
        "take_profit": 1.0910,
        "timing_mode": TimingMode.POST_REACTION,
        "status": CandidateStatus.PENDING,
        "earliest_exec_at": now_ts - 10,
        "latest_exec_at": now_ts + 300,
        "sizing": {"lots": 0.1, "quantity": 0.1, "risk_usd": 50.0},
    }

    ready = signal_candidate_engine.get_ready_signals(now_ts)
    matched = [r for r in ready if r["id"] == test_id]
    assert len(matched) == 1
    assert matched[0]["status"] == CandidateStatus.READY


def test_scan_markets_endpoint(client, auth_headers):
    """Test POST /api/signals/scan-markets endpoint."""
    payload = {"universe": ["EURUSD", "BTCUSDT"], "timeframe": "M5"}
    response = client.post("/api/signals/scan-markets", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "candidates" in data


def test_scan_news_endpoint(client, auth_headers):
    """Test POST /api/signals/scan-news endpoint."""
    payload = {"lookahead_minutes": 60}
    response = client.post("/api/signals/scan-news", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "candidates_count" in data


def test_candidates_and_ready_endpoints(client, auth_headers):
    """Test GET /api/signals/candidates and /ready-for-execution."""
    cands_res = client.get("/api/signals/candidates", headers=auth_headers)
    assert cands_res.status_code == 200
    assert "candidates" in cands_res.json()

    ready_res = client.get("/api/signals/ready-for-execution", headers=auth_headers)
    assert ready_res.status_code == 200
    assert "signals" in ready_res.json()


def test_signal_get_endpoints_require_auth(client, monkeypatch):
    """Signal candidates expose live trade intent — GETs must reject anonymous callers."""
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    assert client.get("/api/signals/candidates").status_code == 401
    assert client.get("/api/signals/ready-for-execution").status_code == 401
    assert client.get("/api/signals/timing-config").status_code == 401


def test_timing_config_endpoints(client, auth_headers):
    """Test GET and POST /api/signals/timing-config."""
    get_res = client.get("/api/signals/timing-config", headers=auth_headers)
    assert get_res.status_code == 200
    assert "config" in get_res.json()
    assert "timing_modes" in get_res.json()

    update_payload = {
        "pre_event_window_min": 20,
        "post_reaction_delay_min": 3,
        "max_spread_pips": 2.5,
        "default_risk_pct": 0.75,
        "strategies_enabled": {"momentum": True, "fade": True},
    }
    post_res = client.post("/api/signals/timing-config", json=update_payload, headers=auth_headers)
    assert post_res.status_code == 200
    cfg = post_res.json()["config"]
    assert cfg["pre_event_window_min"] == 20
    assert cfg["post_reaction_delay_min"] == 3
    assert cfg["max_spread_pips"] == 2.5
    assert cfg["default_risk_pct"] == 0.75


def test_execute_candidate_endpoint_dry_run(client, auth_headers):
    """Test POST /api/signals/execute-candidate."""
    # Seed a ready candidate
    now_ts = int(time.time())
    sig_id = "test-exec-sig-01"
    signal_candidate_engine.candidates[sig_id] = {
        "id": sig_id,
        "symbol": "EURUSD",
        "broker": "ctrader",
        "strategy": "MOMENTUM_TREND_PULSE",
        "direction": "BUY",
        "entry_price": 1.0850,
        "stop_loss": 1.0820,
        "take_profit": 1.0910,
        "timing_mode": TimingMode.POST_REACTION,
        "status": CandidateStatus.READY,
        "earliest_exec_at": now_ts - 5,
        "latest_exec_at": now_ts + 600,
        "sizing": {"lots": 0.1, "quantity": 0.1, "risk_usd": 50.0},
    }

    with patch("backend.services.signal_candidate_engine.ctrader_service.has_credentials", return_value=False):
        res = client.post("/api/signals/execute-candidate", json={"candidate_id": sig_id, "force": True}, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["candidate_id"] == sig_id


def test_signals_routes_dual_mounted_for_nginx_rewrite(client, auth_headers):
    """Dashboard hits /api/backend/signals/* which nginx rewrites to /signals/*."""
    get_res = client.get("/signals/timing-config", headers=auth_headers)
    assert get_res.status_code == 200
    assert "config" in get_res.json()

    api_res = client.get("/api/signals/timing-config", headers=auth_headers)
    assert api_res.status_code == 200
    assert api_res.json()["config"]["pre_event_window_min"] == get_res.json()["config"]["pre_event_window_min"]


def test_usdcad_classified_as_forex_not_crypto():
    """USDCAD must not match USDC substring and route to cTrader."""
    assert classify_symbol("USDCAD") == "forex"
    assert classify_symbol("BTCUSDT") == "crypto"


def test_six_letter_usdt_perps_are_crypto_not_forex():
    """OP/ARB/APT are 6-char USDT perps — not EURUSD-style forex."""
    assert classify_symbol("OPUSDT") == "crypto"
    assert classify_symbol("ARBUSDT") == "crypto"
    assert classify_symbol("APTUSDT") == "crypto"
    assert classify_symbol("EURUSD") == "forex"


def test_tf_to_binance_interval_maps_m5():
    """Scanner timeframe M5 must map to Binance 5m, not invalid m5."""
    assert tf_to_binance_interval("M5") == "5m"
    assert tf_to_binance_interval("5m") == "5m"


@pytest.mark.asyncio
async def test_scan_markets_routes_usdcad_to_ctrader():
    """USDCAD should use cTrader trendbars, not Binance klines."""
    with patch.object(
        signal_candidate_engine, "_compute_features", return_value={"last_close": 1.0, "atr": 0.001, "rsi": 50, "trend": "NEUTRAL", "volatility_pct": 0.1}
    ), patch(
        "backend.services.signal_candidate_engine.ctrader_service.get_trendbars",
        return_value=[{"close": 1.36, "high": 1.361, "low": 1.359, "volume": 1}] * 25,
    ) as mock_ctrader, patch(
        "backend.services.signal_candidate_engine.binance_market_data.get_klines",
        new_callable=AsyncMock,
    ) as mock_binance, patch(
        # Weekend venue closure must not gate this routing test.
        "backend.services.signal_candidate_engine.is_venue_open",
        return_value=True,
    ):
        await signal_candidate_engine.scan_markets(universe=["USDCAD"], timeframe="M5")
        # Signal timeframe plus the slower stop-sizing timeframe, both on cTrader.
        assert mock_ctrader.call_count == 2
        requested = [c.args[1] for c in mock_ctrader.call_args_list]
        assert requested == ["M5", signal_candidate_engine.STOP_TIMEFRAME]
        mock_binance.assert_not_called()


@pytest.mark.asyncio
async def test_execute_candidate_accepts_ctrader_sent_status():
    """cTrader live dispatch returns status=sent before fill ack; must count as success."""
    now_ts = int(time.time())
    sig_id = "test-exec-sent-01"
    signal_candidate_engine.candidates[sig_id] = {
        "id": sig_id,
        "symbol": "EURUSD",
        "broker": "ctrader",
        "strategy": "MOMENTUM_TREND_PULSE",
        "direction": "BUY",
        "entry_price": 1.0850,
        "stop_loss": 1.0820,
        "take_profit": 1.0910,
        "timing_mode": TimingMode.BAR_CLOSE,
        "status": CandidateStatus.READY,
        "earliest_exec_at": now_ts - 5,
        "latest_exec_at": now_ts + 600,
        "sizing": {"lots": 0.01, "quantity": 0.01, "risk_usd": 50.0},
    }

    with patch.object(signal_candidate_engine, "_open_ctrader_position_count", return_value=0), patch.dict(
        signal_candidate_engine.execution_config, {"max_portfolio_risk_pct": 0}
    ), patch(
        "backend.services.signal_candidate_engine.ctrader_service.ensure_connected",
        return_value=True,
    ), patch(
        "backend.services.signal_candidate_engine.ctrader_service.place_order",
        return_value={"status": "sent", "symbol": "EURUSD", "direction": "BUY", "quantity": 0.01},
    ), patch(
        "backend.services.signal_candidate_engine.persist_ctrader_execution",
        return_value=99,
    ) as mock_persist:
        res = await signal_candidate_engine.execute_candidate(sig_id, force=True)

    assert res["success"] is True
    assert signal_candidate_engine.candidates[sig_id]["status"] == CandidateStatus.EXECUTED
    mock_persist.assert_called_once()
    assert mock_persist.call_args.kwargs["symbol"] == "EURUSD"
    assert mock_persist.call_args.kwargs["direction"] == "BUY"


@pytest.mark.asyncio
async def test_execute_candidate_rejects_simulated_when_credentials_exist():
    """Paper simulated fills must not masquerade as live cTrader execution."""
    now_ts = int(time.time())
    sig_id = "test-exec-simulated-01"
    signal_candidate_engine.candidates[sig_id] = {
        "id": sig_id,
        "symbol": "EURUSD",
        "broker": "ctrader",
        "strategy": "MOMENTUM_TREND_PULSE",
        "direction": "BUY",
        "entry_price": 1.0850,
        "stop_loss": 1.0820,
        "take_profit": 1.0910,
        "timing_mode": TimingMode.BAR_CLOSE,
        "status": CandidateStatus.READY,
        "earliest_exec_at": now_ts - 5,
        "latest_exec_at": now_ts + 600,
        "sizing": {"lots": 0.01, "quantity": 0.01, "risk_usd": 50.0},
    }

    with patch.object(signal_candidate_engine, "_open_ctrader_position_count", return_value=0), patch.dict(
        signal_candidate_engine.execution_config, {"max_portfolio_risk_pct": 0}
    ), patch(
        "backend.services.signal_candidate_engine.ctrader_service.ensure_connected",
        return_value=False,
    ), patch(
        "backend.services.signal_candidate_engine.ctrader_service.has_credentials",
        return_value=True,
    ), patch(
        "backend.services.signal_candidate_engine.persist_ctrader_execution",
    ) as mock_persist:
        res = await signal_candidate_engine.execute_candidate(sig_id, force=True)

    assert res["success"] is False
    assert "not connected" in res["error"].lower()
    assert signal_candidate_engine.candidates[sig_id]["status"] == CandidateStatus.READY
    mock_persist.assert_not_called()


@pytest.mark.asyncio
async def test_execute_candidate_skips_market_closed_without_blocking_queue():
    now_ts = int(time.time())
    sig_id = "test-exec-closed-01"
    signal_candidate_engine.candidates[sig_id] = {
        "id": sig_id,
        "symbol": "XAUUSD",
        "broker": "ctrader",
        "strategy": "MACRO_EVENT_POST_REACTION",
        "direction": "BUY",
        "entry_price": 2500.0,
        "stop_loss": 2490.0,
        "take_profit": 2520.0,
        "timing_mode": TimingMode.BAR_CLOSE,
        "status": CandidateStatus.READY,
        "earliest_exec_at": now_ts - 5,
        "latest_exec_at": now_ts + 600,
        "sizing": {"lots": 0.01, "quantity": 0.01, "risk_usd": 50.0},
    }

    with patch.object(signal_candidate_engine, "_open_ctrader_position_count", return_value=0), patch.dict(
        signal_candidate_engine.execution_config, {"forex_only": False, "include_metals": True, "max_portfolio_risk_pct": 0}
    ), patch(
        "backend.services.signal_candidate_engine.ctrader_service.ensure_connected",
        return_value=True,
    ), patch(
        "backend.services.signal_candidate_engine.ctrader_service.place_order",
        return_value={"status": "error", "error": "MARKET_CLOSED — Trading is not available: Market is closed."},
    ):
        res = await signal_candidate_engine.execute_candidate(sig_id, force=True)

    assert res["success"] is False
    assert res["skipped"] is True
    assert signal_candidate_engine.candidates[sig_id]["status"] == CandidateStatus.READY


@pytest.mark.asyncio
async def test_execute_candidate_skips_closed_venue_without_placing_order():
    now_ts = int(time.time())
    sig_id = "test-exec-hours-01"
    signal_candidate_engine.candidates[sig_id] = {
        "id": sig_id,
        "symbol": "EURUSD",
        "broker": "ctrader",
        "strategy": "MOMENTUM_TREND_PULSE",
        "direction": "BUY",
        "entry_price": 1.10,
        "stop_loss": 1.09,
        "take_profit": 1.12,
        "timing_mode": TimingMode.BAR_CLOSE,
        "status": CandidateStatus.READY,
        "earliest_exec_at": now_ts - 5,
        "latest_exec_at": now_ts + 600,
        "sizing": {"lots": 0.01, "quantity": 0.01, "risk_usd": 50.0},
    }

    with patch(
        "backend.services.signal_candidate_engine.is_venue_open",
        return_value=False,
    ), patch(
        "backend.services.signal_candidate_engine.ctrader_service.place_order",
    ) as mock_place:
        res = await signal_candidate_engine.execute_candidate(sig_id, force=False)

    assert res["success"] is False
    assert res["skipped"] is True
    assert "VENUE_CLOSED" in res["error"]
    assert signal_candidate_engine.candidates[sig_id]["status"] == CandidateStatus.READY
    mock_place.assert_not_called()


def test_get_ready_signals_forex_only_excludes_crypto():
    now_ts = int(time.time())
    previous = dict(signal_candidate_engine.candidates)
    signal_candidate_engine.candidates.clear()
    try:
        signal_candidate_engine.candidates["fx-ready"] = {
            "id": "fx-ready",
            "symbol": "EURUSD",
            "broker": "ctrader",
            "status": CandidateStatus.READY,
            "confidence": 0.9,
            "earliest_exec_at": now_ts - 5,
            "latest_exec_at": now_ts + 600,
        }
        signal_candidate_engine.candidates["cr-ready"] = {
            "id": "cr-ready",
            "symbol": "BTCUSDT",
            "broker": "binance_futures",
            "status": CandidateStatus.READY,
            "confidence": 0.99,
            "earliest_exec_at": now_ts - 5,
            "latest_exec_at": now_ts + 600,
        }

        with patch.object(
            signal_candidate_engine, "_open_ctrader_position_count", return_value=0
        ), patch.object(
            signal_candidate_engine, "_open_ctrader_symbols", return_value=set()
        ):
            ready = signal_candidate_engine.get_ready_signals(now_ts, forex_only=True, limit=5)

        ids = {c["id"] for c in ready}
        assert "fx-ready" in ids
        assert "cr-ready" not in ids
    finally:
        signal_candidate_engine.candidates = previous


def test_get_ready_signals_forex_only_excludes_metals_by_default():
    now_ts = int(time.time())
    previous = dict(signal_candidate_engine.candidates)
    signal_candidate_engine.candidates.clear()
    try:
        signal_candidate_engine.candidates["fx-eur"] = {
            "id": "fx-eur",
            "symbol": "EURUSD",
            "broker": "ctrader",
            "status": CandidateStatus.READY,
            "confidence": 0.8,
            "earliest_exec_at": now_ts - 5,
            "latest_exec_at": now_ts + 600,
        }
        signal_candidate_engine.candidates["mt-gold"] = {
            "id": "mt-gold",
            "symbol": "XAUUSD",
            "broker": "ctrader",
            "status": CandidateStatus.READY,
            "confidence": 0.99,
            "earliest_exec_at": now_ts - 5,
            "latest_exec_at": now_ts + 600,
        }

        with patch.dict(
            signal_candidate_engine.execution_config,
            {"include_metals": False, "forex_only": True, "max_open_ctrader_positions": 10},
        ), patch.object(
            signal_candidate_engine, "_open_ctrader_position_count", return_value=0
        ), patch.object(
            signal_candidate_engine, "_open_ctrader_symbols", return_value=set()
        ):
            ready = signal_candidate_engine.get_ready_signals(now_ts, forex_only=True, limit=10)

        ids = {c["id"] for c in ready}
        assert "fx-eur" in ids
        assert "mt-gold" not in ids
    finally:
        signal_candidate_engine.candidates = previous


def test_get_ready_signals_empty_when_ctrader_position_cap_reached():
    now_ts = int(time.time())
    signal_candidate_engine.candidates["fx-cap"] = {
        "id": "fx-cap",
        "symbol": "GBPUSD",
        "broker": "ctrader",
        "status": CandidateStatus.READY,
        "confidence": 0.8,
        "earliest_exec_at": now_ts - 5,
        "latest_exec_at": now_ts + 600,
    }

    with patch.dict(signal_candidate_engine.execution_config, {"max_open_ctrader_positions": 1}), patch.object(
        signal_candidate_engine, "_open_ctrader_position_count", return_value=1
    ):
        ready = signal_candidate_engine.get_ready_signals(now_ts, forex_only=True, limit=1)

    assert ready == []


def test_get_ready_signals_allows_multiple_when_slots_remain():
    now_ts = int(time.time())
    for i, sym in enumerate(["EURUSD", "GBPUSD", "USDJPY"]):
        signal_candidate_engine.candidates[f"fx-multi-{i}"] = {
            "id": f"fx-multi-{i}",
            "symbol": sym,
            "broker": "ctrader",
            "status": CandidateStatus.READY,
            "confidence": 0.9 - (i * 0.01),
            "earliest_exec_at": now_ts - 5,
            "latest_exec_at": now_ts + 600,
        }

    with patch.dict(
        signal_candidate_engine.execution_config,
        {"max_open_ctrader_positions": 5, "max_ready_per_poll": 2, "one_position_per_symbol": True},
    ), patch.object(
        signal_candidate_engine, "_open_ctrader_position_count", return_value=1
    ), patch.object(
        signal_candidate_engine, "_open_ctrader_symbols", return_value={"EURUSD"}
    ):
        ready = signal_candidate_engine.get_ready_signals(now_ts, forex_only=True, limit=2)

    symbols = [c["symbol"] for c in ready]
    assert "EURUSD" not in symbols
    assert len(ready) == 2
    assert set(symbols) <= {"GBPUSD", "USDJPY"}


def test_get_ready_signals_fills_up_to_ten_open_slots():
    now_ts = int(time.time())
    previous = dict(signal_candidate_engine.candidates)
    signal_candidate_engine.candidates.clear()
    pairs = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
        "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY", "AUDJPY",
    ]
    try:
        for i, sym in enumerate(pairs):
            signal_candidate_engine.candidates[f"fx-ten-{i}"] = {
                "id": f"fx-ten-{i}",
                "symbol": sym,
                "broker": "ctrader",
                "status": CandidateStatus.READY,
                "confidence": 0.95 - (i * 0.01),
                "earliest_exec_at": now_ts - 5,
                "latest_exec_at": now_ts + 600,
            }

        with patch.dict(
            signal_candidate_engine.execution_config,
            {
                "max_open_ctrader_positions": 10,
                "max_ready_per_poll": 10,
                "one_position_per_symbol": True,
                "max_same_base": 0,
                "max_currency_exposure": 0,  # test targets the slot cap, not currency clusters
            },
        ), patch.object(
            signal_candidate_engine, "_open_ctrader_position_count", return_value=0
        ), patch.object(
            signal_candidate_engine, "_open_ctrader_symbols", return_value=set()
        ):
            ready = signal_candidate_engine.get_ready_signals(now_ts, forex_only=True)

        symbols = [c["symbol"] for c in ready]
        assert len(ready) == 10
        assert len(set(symbols)) == 10
        assert "AUDJPY" not in symbols  # 11th pair is above the cap
    finally:
        signal_candidate_engine.candidates = previous


@pytest.mark.asyncio
async def test_execute_candidate_keeps_ready_on_paper_failure():
    """Failed paper fills must not mark candidate EXECUTED."""
    now_ts = int(time.time())
    sig_id = "test-exec-fail-01"
    signal_candidate_engine.candidates[sig_id] = {
        "id": sig_id,
        "symbol": "BTCUSDT",
        "broker": "binance_futures",
        "strategy": "MOMENTUM_TREND_PULSE",
        "direction": "BUY",
        "entry_price": 0,
        "stop_loss": 90000,
        "take_profit": 110000,
        "timing_mode": TimingMode.BAR_CLOSE,
        "status": CandidateStatus.READY,
        "earliest_exec_at": now_ts - 5,
        "latest_exec_at": now_ts + 600,
        "sizing": {"lots": 0.1, "quantity": 0.01, "risk_usd": 50.0},
    }

    mock_resp = MagicMock(success=False, order_id="paper_000001", message="Paper market entry requires an explicit price or prior market fill")
    with patch.dict(signal_candidate_engine.execution_config, {"forex_only": False, "max_portfolio_risk_pct": 0}), patch.object(
        signal_candidate_engine, "_resolve_mark_price", new_callable=AsyncMock, return_value=0.0
    ), patch.object(
        signal_candidate_engine, "_try_open_binance_positions", return_value=[]
    ), patch(
        "backend.services.signal_candidate_engine.UnifiedTrading"
    ) as mock_ut_cls:
        mock_ut_cls.return_value.list_sessions.return_value = [
            {"id": "binance_futures_paper", "broker": "binance_futures", "mode": "paper"}
        ]
        mock_ut_cls.return_value.place_order.return_value = mock_resp
        res = await signal_candidate_engine.execute_candidate(sig_id, force=True)

    assert res["success"] is False
    assert signal_candidate_engine.candidates[sig_id]["status"] == CandidateStatus.READY
    assert signal_candidate_engine.candidates[sig_id]["execution_result"]["success"] is False
    # Order must be routed to the broker's own session, never the global default
    _, kwargs = mock_ut_cls.return_value.place_order.call_args
    assert kwargs.get("session_id") == "binance_futures_paper"


def test_resolve_equity_uses_live_broker_and_logs_fallback():
    engine = SignalCandidateEngine()
    with patch(
        "backend.services.signal_candidate_engine.ctrader_service.equity",
        40000.0,
    ):
        assert engine._resolve_equity("ctrader") == pytest.approx(40000.0)

    engine._equity_cache.clear()
    with patch(
        "backend.services.signal_candidate_engine.binance_futures_broker.get_balance",
        return_value={"equity": 0.0, "balance": 0.0},
    ):
        assert engine._resolve_equity("binance_futures") == pytest.approx(150.0)


def test_calculate_size_scales_with_live_equity():
    engine = SignalCandidateEngine()
    with patch(
        "backend.services.signal_candidate_engine.ctrader_service.equity",
        10000.0,
    ):
        small = engine._calculate_size("EURUSD", 1.0850, 1.0820, "ctrader")
    with patch(
        "backend.services.signal_candidate_engine.ctrader_service.equity",
        50000.0,
    ):
        large = engine._calculate_size("EURUSD", 1.0850, 1.0820, "ctrader")
    assert large["risk_usd"] == pytest.approx(small["risk_usd"] * 5.0)
    assert large["lots"] > small["lots"]


def test_calculate_size_crypto_uses_candidate_stop_not_hardcoded_pct():
    engine = SignalCandidateEngine()
    with patch(
        "backend.services.signal_candidate_engine.binance_futures_broker.get_balance",
        return_value={"equity": 10000.0},
    ):
        tight = engine._calculate_size("BTCUSDT", 100.0, 99.8, "binance_futures")
        wide = engine._calculate_size("BTCUSDT", 100.0, 96.0, "binance_futures")
    # Same $50 risk: tighter stop → larger qty (sized from the stop, not 0.4%).
    assert tight["quantity"] > wide["quantity"]
    assert tight["risk_usd"] == pytest.approx(50.0)
    assert wide["risk_usd"] == pytest.approx(50.0)


def test_prune_terminal_candidates_drops_old_terminal_keeps_live():
    engine = SignalCandidateEngine()
    now = int(time.time())
    old_iso = datetime.fromtimestamp(now - CANDIDATE_TTL_SECONDS - 60, tz=timezone.utc).isoformat()
    fresh_iso = datetime.fromtimestamp(now - 60, tz=timezone.utc).isoformat()
    engine.candidates = {
        "old-exec": {
            "status": CandidateStatus.EXECUTED,
            "created_at": old_iso,
            "latest_exec_at": now - CANDIDATE_TTL_SECONDS - 10,
        },
        "fresh-exp": {
            "status": CandidateStatus.EXPIRED,
            "created_at": fresh_iso,
            "latest_exec_at": now - 10,
        },
        "live": {
            "status": CandidateStatus.READY,
            "created_at": old_iso,
            "earliest_exec_at": now - 10,
            "latest_exec_at": now + 600,
        },
    }
    removed = engine._prune_terminal_candidates(now)
    assert removed == 1
    assert "old-exec" not in engine.candidates
    assert "fresh-exp" in engine.candidates
    assert "live" in engine.candidates


def test_get_ready_signals_prunes_stale_terminal_store():
    engine = SignalCandidateEngine()
    now = int(time.time())
    old_iso = datetime.fromtimestamp(now - CANDIDATE_TTL_SECONDS - 5, tz=timezone.utc).isoformat()
    engine.candidates["stale"] = {
        "status": CandidateStatus.CANCELLED,
        "created_at": old_iso,
        "latest_exec_at": now - CANDIDATE_TTL_SECONDS,
        "broker": "ctrader",
        "symbol": "EURUSD",
        "confidence": 0.9,
        "earliest_exec_at": now - CANDIDATE_TTL_SECONDS - 100,
    }
    with patch.object(engine, "_open_ctrader_symbols", return_value=set()), \
         patch.object(engine, "_ctrader_execution_slots_remaining", return_value=10):
        engine.get_ready_signals(current_ts=now, enforce_ctrader_position_cap=False)
    assert "stale" not in engine.candidates


def test_momentum_veto_low_adx_chop():
    """Verify MOMENTUM_TREND_PULSE rejects trades when ADX < min_adx."""
    engine = SignalCandidateEngine()
    features = {
        "last_close": 1.0850,
        "atr": 0.0015,
        "rsi": 60,
        "adx": 15.0,  # Below default min 22
        "ema_fast": 1.0855,
        "ema_slow": 1.0845,
        "trend": "BULLISH",
        "h1_trend": "BULLISH",
    }
    sig = engine._evaluate_momentum("EURUSD", features, "ctrader")
    assert sig is None, "Momentum must be vetoed in low-ADX chop"


def test_momentum_veto_opposing_h1_trend():
    """Verify MOMENTUM_TREND_PULSE rejects BUY when H1 is BEARISH and SELL when H1 is BULLISH."""
    engine = SignalCandidateEngine()
    features_buy = {
        "last_close": 1.0850,
        "atr": 0.0015,
        "rsi": 62,
        "adx": 28.0,
        "ema_fast": 1.0855,
        "ema_slow": 1.0845,
        "trend": "BULLISH",
        "h1_trend": "BEARISH",  # Counter-trend!
    }
    assert engine._evaluate_momentum("EURUSD", features_buy, "ctrader") is None

    features_sell = {
        "last_close": 1.0850,
        "atr": 0.0015,
        "rsi": 35,
        "adx": 28.0,
        "ema_fast": 1.0840,
        "ema_slow": 1.0850,
        "trend": "BEARISH",
        "h1_trend": "BULLISH",  # Counter-trend!
    }
    assert engine._evaluate_momentum("EURUSD", features_sell, "ctrader") is None


def test_momentum_accepts_when_adx_and_h1_aligned():
    """Verify MOMENTUM_TREND_PULSE accepts when ADX is strong and H1 is aligned."""
    engine = SignalCandidateEngine()
    features = {
        "last_close": 1.0850,
        "atr": 0.0015,
        "rsi": 62,
        "adx": 28.0,
        "ema_fast": 1.0855,
        "ema_slow": 1.0845,
        "trend": "BULLISH",
        "h1_trend": "BULLISH",
    }
    sig = engine._evaluate_momentum("EURUSD", features, "ctrader")
    assert sig is not None
    assert sig["direction"] == "BUY"
    assert sig["strategy"] == "MOMENTUM_TREND_PULSE"


def test_anti_whipsaw_symbol_cooldown():
    """Verify that cooling down a symbol locks it out from candidate execution."""
    engine = SignalCandidateEngine()
    sym = "USDJPY"
    assert not engine.is_symbol_cooling_down(sym)

    engine.set_symbol_cooldown(sym, duration_sec=1800)
    assert engine.is_symbol_cooling_down(sym)

    # Candidate execution should be skipped for cooling symbol
    now = int(time.time())
    cand_id = "test-cooldown-cand"
    engine.candidates[cand_id] = {
        "id": cand_id,
        "symbol": sym,
        "direction": "BUY",
        "broker": "ctrader",
        "status": CandidateStatus.READY,
        "earliest_exec_at": now - 10,
        "latest_exec_at": now + 60,
        "entry_price": 150.0,
        "stop_loss": 149.0,
        "take_profit": 152.0,
        "sizing": {"lots": 0.01, "quantity": 0.01},
    }
    import asyncio
    with patch("backend.services.signal_candidate_engine.is_venue_open", return_value=True):
        res = asyncio.run(engine.execute_candidate(cand_id, force=False))
    assert res.get("skipped") is True
    assert "cooldown" in res.get("error", "").lower()

    # Force=True bypasses cooldown
    with patch("backend.services.sentry_state.is_trading_allowed", return_value=True), \
         patch.object(engine, "_ctrader_execution_slot_available", return_value=True), \
         patch.object(engine, "_open_ctrader_symbols", return_value=set()), \
         patch.object(engine, "_same_base_slots_available", return_value=True), \
         patch.object(engine, "_currency_exposure_slots_available", return_value=True), \
         patch("backend.services.signal_candidate_engine.live_ctrader_orders_allowed", return_value=False), \
         patch("backend.services.signal_candidate_engine.ctrader_service.get_spread_pips", return_value=0.5), \
         patch("backend.services.signal_candidate_engine.ctrader_service.place_order", return_value={"status": "sent", "order_id": "test-123"}):
        res_forced = asyncio.run(engine.execute_candidate(cand_id, force=True))
        assert res_forced.get("success") is True



@pytest.mark.asyncio
async def test_scan_markets_dedupes_live_twin():
    """A live (symbol, strategy, direction) candidate must not be re-armed by the next scan."""
    engine = SignalCandidateEngine()
    engine.candidates.clear()
    bars = [
        {"close": 1.08, "high": 1.081, "low": 1.079, "volume": 1}
        for _ in range(25)
    ]
    raw = {
        "strategy": "MOMENTUM_TREND_PULSE",
        "direction": "BUY",
        "entry_price": 1.08,
        "stop_loss": 1.075,
        "take_profit": 1.09,
        "timing_mode": TimingMode.POST_REACTION,
        "confidence": 0.8,
        "reason": "test",
    }
    with patch.object(CTraderService, "get_trendbars", return_value=bars), \
         patch.object(engine, "_attach_stop_atr", new_callable=AsyncMock), \
         patch.object(engine, "_evaluate_momentum", side_effect=lambda *a, **k: dict(raw)), \
         patch.object(engine, "_evaluate_fade", return_value=None), \
         patch.object(engine, "_evaluate_straddle", return_value=None), \
         patch.object(engine, "_evaluate_slingshot", return_value=None), \
         patch.object(engine, "_fx_gate_mode", return_value="off"), \
         patch.object(engine, "_has_open_position", return_value=False), \
         patch.object(engine, "_portfolio_risk_breach", return_value=None), \
         patch.object(engine, "_binance_position_cap_breach", return_value=None), \
         patch("backend.services.signal_candidate_engine.is_venue_open", return_value=True):
        first = await engine.scan_markets(universe=["EURUSD"], timeframe="M5")
        second = await engine.scan_markets(universe=["EURUSD"], timeframe="M5")

    assert len(first) == 1
    assert second == []
    live = [
        c for c in engine.candidates.values()
        if c["status"] in (CandidateStatus.PENDING, CandidateStatus.READY)
    ]
    assert len(live) == 1


@pytest.mark.asyncio
async def test_execute_candidate_cancels_duplicate_siblings():
    """Executing one candidate must cancel live twins on the same symbol+direction."""
    engine = SignalCandidateEngine()
    engine.candidates.clear()
    now_ts = int(time.time())

    def _cand(cid):
        return {
            "id": cid,
            "symbol": "BTCUSDT",
            "broker": "binance_futures",
            "strategy": "MOMENTUM_TREND_PULSE",
            "direction": "BUY",
            "entry_price": 100000,
            "stop_loss": 99000,
            "take_profit": 102000,
            "timing_mode": TimingMode.BAR_CLOSE,
            "status": CandidateStatus.READY,
            "earliest_exec_at": now_ts - 5,
            "latest_exec_at": now_ts + 600,
            "sizing": {"lots": 0.01, "quantity": 0.01, "risk_usd": 50.0},
        }

    engine.candidates["dup-a"] = _cand("dup-a")
    engine.candidates["dup-b"] = _cand("dup-b")

    mock_resp = MagicMock(success=True, order_id="ord-1", message="filled")
    with patch.dict(engine.execution_config, {"forex_only": False, "max_portfolio_risk_pct": 0}), patch(
        "backend.services.sentry_state.is_trading_allowed", return_value=True
    ), patch.object(
        engine, "_try_open_binance_positions", return_value=[]
    ), patch.object(
        engine, "_resolve_mark_price", new_callable=AsyncMock, return_value=100000.0
    ), patch(
        "backend.services.signal_candidate_engine.UnifiedTrading"
    ) as mock_ut_cls:
        mock_ut_cls.return_value.list_sessions.return_value = [
            {"id": "binance_futures_paper", "broker": "binance_futures", "mode": "paper"}
        ]
        mock_ut_cls.return_value.place_order.return_value = mock_resp
        res = await engine.execute_candidate("dup-a", force=True)

    assert res["success"] is True
    assert engine.candidates["dup-a"]["status"] == CandidateStatus.EXECUTED
    assert engine.candidates["dup-b"]["status"] == CandidateStatus.CANCELLED
    _, kwargs = mock_ut_cls.return_value.place_order.call_args
    assert kwargs.get("session_id") == "binance_futures_paper"


def _binance_candidate(cid: str, now_ts: int) -> dict:
    return {
        "id": cid,
        "symbol": "BTCUSDT",
        "broker": "binance_futures",
        "strategy": "MOMENTUM_TREND_PULSE",
        "direction": "BUY",
        "entry_price": 100000,
        "stop_loss": 99000,
        "take_profit": 102000,
        "timing_mode": TimingMode.BAR_CLOSE,
        "status": CandidateStatus.READY,
        "earliest_exec_at": now_ts - 5,
        "latest_exec_at": now_ts + 600,
        "sizing": {"lots": 0.01, "quantity": 0.01, "risk_usd": 50.0},
    }


@pytest.mark.asyncio
async def test_execute_candidate_binance_one_position_per_symbol():
    """A binance_futures candidate must be skipped when the symbol already has an open position."""
    engine = SignalCandidateEngine()
    engine.candidates.clear()
    engine.candidates["b-1"] = _binance_candidate("b-1", int(time.time()))

    with patch.dict(engine.execution_config, {"forex_only": False, "max_portfolio_risk_pct": 0}), patch(
        "backend.services.sentry_state.is_trading_allowed", return_value=True
    ), patch.object(
        engine, "_try_open_binance_positions", return_value=[{"symbol": "BTCUSDT"}]
    ):
        res = await engine.execute_candidate("b-1", force=True)

    assert res["success"] is False
    assert res["skipped"] is True
    assert "open Binance position" in res["error"]
    assert engine.candidates["b-1"]["status"] == CandidateStatus.READY


@pytest.mark.asyncio
async def test_execute_candidate_binance_max_positions_cap():
    engine = SignalCandidateEngine()
    engine.candidates.clear()
    engine.candidates["b-1"] = _binance_candidate("b-1", int(time.time()))

    open_positions = [{"symbol": f"SYM{i}USDT"} for i in range(10)]
    with patch.dict(
        engine.execution_config,
        {"forex_only": False, "max_portfolio_risk_pct": 0, "max_binance_positions": 10},
    ), patch(
        "backend.services.sentry_state.is_trading_allowed", return_value=True
    ), patch.object(
        engine, "_try_open_binance_positions", return_value=open_positions
    ):
        res = await engine.execute_candidate("b-1", force=True)

    assert res["success"] is False
    assert res["skipped"] is True
    assert "Max open Binance positions" in res["error"]


@pytest.mark.asyncio
async def test_execute_candidate_portfolio_risk_budget():
    """Committed risk across live candidates plus the new trade may not exceed the budget."""
    engine = SignalCandidateEngine()
    engine.candidates.clear()
    now_ts = int(time.time())
    engine.candidates["b-1"] = _binance_candidate("b-1", now_ts)
    # Another live candidate already commits $50 of the same book.
    committed = _binance_candidate("b-2", now_ts)
    committed["symbol"] = "ETHUSDT"
    engine.candidates["b-2"] = committed

    # equity $10_000, budget 0.5% = $50 -> $50 committed + $50 new = $100 > $50
    with patch.dict(
        engine.execution_config, {"forex_only": False, "max_portfolio_risk_pct": 0.5}
    ), patch(
        "backend.services.sentry_state.is_trading_allowed", return_value=True
    ), patch.object(
        engine, "_try_open_binance_positions", return_value=[]
    ), patch.object(
        engine, "_resolve_equity", return_value=10_000.0
    ):
        res = await engine.execute_candidate("b-1", force=True)

    assert res["success"] is False
    assert res["skipped"] is True
    assert "risk budget" in res["error"].lower()
    assert engine.candidates["b-1"]["status"] == CandidateStatus.READY


@pytest.mark.asyncio
async def test_scans_return_empty_while_halted():
    """Signal generation pauses while the sentry halt flag is set."""
    engine = SignalCandidateEngine()
    with patch(
        "backend.services.sentry_state.is_trading_allowed", return_value=False
    ):
        assert await engine.scan_markets(universe=["EURUSD"]) == []
        assert await engine.scan_news_and_events() == []


def _scan_raw(strategy: str = "MOMENTUM_TREND_PULSE", direction: str = "SELL") -> dict:
    return {
        "strategy": strategy,
        "direction": direction,
        "entry_price": 1.08,
        "stop_loss": 1.085,
        "take_profit": 1.07,
        "timing_mode": TimingMode.POST_REACTION,
        "confidence": 0.8,
        "reason": "test",
    }


def _feature_bars(n: int = 25, falling: bool = True) -> list:
    step = -0.0002 if falling else 0.0002
    return [
        {
            "close": 1.10 + i * step,
            "high": 1.101 + i * step,
            "low": 1.099 + i * step,
            "volume": 1,
        }
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_scan_news_does_not_rearm_after_executed_twin():
    """After a MACRO_EVENT fill, news scan must not arm another same-direction twin."""
    engine = SignalCandidateEngine()
    engine.candidates.clear()
    engine.candidates["filled-macro"] = {
        "id": "filled-macro",
        "symbol": "EURUSD",
        "broker": "ctrader",
        "strategy": "MACRO_EVENT_POST_REACTION",
        "direction": "SELL",
        "status": CandidateStatus.EXECUTED,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sizing": {"risk_usd": 1.0},
    }
    calendar = {"events": [{"event": "NFP", "currency": "USD", "impact": "high"}]}
    with patch(
        "backend.services.sentry_state.is_trading_allowed", return_value=True
    ), patch(
        "backend.services.signal_candidate_engine.ctrader_service.get_trendbars",
        return_value=_feature_bars(),
    ), patch.object(
        engine, "_attach_stop_atr", new_callable=AsyncMock
    ), patch.object(
        engine, "_has_open_position", return_value=False
    ), patch.object(
        engine, "_portfolio_risk_breach", return_value=None
    ), patch(
        "backend.routes.news.get_economic_calendar",
        new=AsyncMock(return_value=calendar),
    ), patch(
        "backend.routes.news.get_news_feed", new=AsyncMock(return_value={})
    ), patch(
        "backend.routes.news.get_market_sentiment", new=AsyncMock(return_value={})
    ):
        created = await engine.scan_news_and_events()

    assert created == []
    live = [
        c
        for c in engine.candidates.values()
        if c["status"] in (CandidateStatus.PENDING, CandidateStatus.READY)
        and c.get("symbol") == "EURUSD"
    ]
    assert live == []


@pytest.mark.asyncio
async def test_scan_markets_skips_when_open_position_exists():
    """Market scan must not arm a candidate that duplicates an open book side."""
    engine = SignalCandidateEngine()
    engine.candidates.clear()
    raw = _scan_raw()
    with patch.object(CTraderService, "get_trendbars", return_value=_feature_bars()), \
         patch.object(engine, "_attach_stop_atr", new_callable=AsyncMock), \
         patch.object(engine, "_evaluate_momentum", side_effect=lambda *a, **k: dict(raw)), \
         patch.object(engine, "_evaluate_fade", return_value=None), \
         patch.object(engine, "_evaluate_straddle", return_value=None), \
         patch.object(engine, "_evaluate_slingshot", return_value=None), \
         patch.object(engine, "_fx_gate_mode", return_value="off"), \
         patch.object(engine, "_has_open_position", return_value=True), \
         patch.object(engine, "_portfolio_risk_breach", return_value=None):
        created = await engine.scan_markets(universe=["EURUSD"], timeframe="M5")

    assert created == []
    assert engine.candidates == {}


@pytest.mark.asyncio
async def test_scan_news_skips_when_open_position_exists():
    engine = SignalCandidateEngine()
    engine.candidates.clear()
    calendar = {"events": [{"event": "NFP", "currency": "USD", "impact": "high"}]}
    with patch(
        "backend.services.sentry_state.is_trading_allowed", return_value=True
    ), patch(
        "backend.services.signal_candidate_engine.ctrader_service.get_trendbars",
        return_value=_feature_bars(),
    ), patch.object(
        engine, "_attach_stop_atr", new_callable=AsyncMock
    ), patch.object(
        engine, "_has_open_position", return_value=True
    ), patch.object(
        engine, "_portfolio_risk_breach", return_value=None
    ), patch(
        "backend.routes.news.get_economic_calendar",
        new=AsyncMock(return_value=calendar),
    ), patch(
        "backend.routes.news.get_news_feed", new=AsyncMock(return_value={})
    ), patch(
        "backend.routes.news.get_market_sentiment", new=AsyncMock(return_value={})
    ):
        created = await engine.scan_news_and_events()

    assert created == []


@pytest.mark.asyncio
async def test_scan_markets_skips_cross_strategy_same_direction_live_twin():
    """MOMENTUM must not sit beside a live MACRO_EVENT on the same symbol+side."""
    engine = SignalCandidateEngine()
    engine.candidates.clear()
    engine.candidates["macro-live"] = {
        "id": "macro-live",
        "symbol": "EURUSD",
        "broker": "ctrader",
        "strategy": "MACRO_EVENT_POST_REACTION",
        "direction": "SELL",
        "status": CandidateStatus.READY,
        "sizing": {"risk_usd": 1.0},
        "latest_exec_at": int(time.time()) + 600,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    raw = _scan_raw(strategy="MOMENTUM_TREND_PULSE", direction="SELL")
    with patch.object(CTraderService, "get_trendbars", return_value=_feature_bars()), \
         patch.object(engine, "_attach_stop_atr", new_callable=AsyncMock), \
         patch.object(engine, "_evaluate_momentum", side_effect=lambda *a, **k: dict(raw)), \
         patch.object(engine, "_evaluate_fade", return_value=None), \
         patch.object(engine, "_evaluate_straddle", return_value=None), \
         patch.object(engine, "_evaluate_slingshot", return_value=None), \
         patch.object(engine, "_fx_gate_mode", return_value="off"), \
         patch.object(engine, "_has_open_position", return_value=False), \
         patch.object(engine, "_portfolio_risk_breach", return_value=None):
        created = await engine.scan_markets(universe=["EURUSD"], timeframe="M5")

    assert created == []
    live = [
        c
        for c in engine.candidates.values()
        if c["status"] in (CandidateStatus.PENDING, CandidateStatus.READY)
    ]
    assert len(live) == 1
    assert live[0]["id"] == "macro-live"


@pytest.mark.asyncio
async def test_scan_markets_respects_portfolio_risk_and_binance_caps():
    """Scan path must apply the same portfolio / Binance caps as execute."""
    engine = SignalCandidateEngine()
    engine.candidates.clear()
    raw = _scan_raw(strategy="MOMENTUM_TREND_PULSE", direction="BUY")
    raw["entry_price"] = 100000
    raw["stop_loss"] = 99000
    raw["take_profit"] = 102000

    with patch.object(engine, "_attach_stop_atr", new_callable=AsyncMock), \
         patch.object(engine, "_evaluate_momentum", side_effect=lambda *a, **k: dict(raw)), \
         patch.object(engine, "_evaluate_fade", return_value=None), \
         patch.object(engine, "_evaluate_straddle", return_value=None), \
         patch.object(engine, "_evaluate_slingshot", return_value=None), \
         patch.object(engine, "_fx_gate_mode", return_value="off"), \
         patch.object(engine, "_has_open_position", return_value=False), \
         patch.object(engine, "_portfolio_risk_breach", return_value="Portfolio risk budget exceeded"), \
         patch(
             "backend.services.signal_candidate_engine.binance_market_data.get_klines",
             new_callable=AsyncMock,
             return_value=_feature_bars(),
         ):
        created = await engine.scan_markets(universe=["BTCUSDT"], timeframe="M5")
    assert created == []

    with patch.object(engine, "_attach_stop_atr", new_callable=AsyncMock), \
         patch.object(engine, "_evaluate_momentum", side_effect=lambda *a, **k: dict(raw)), \
         patch.object(engine, "_evaluate_fade", return_value=None), \
         patch.object(engine, "_evaluate_straddle", return_value=None), \
         patch.object(engine, "_evaluate_slingshot", return_value=None), \
         patch.object(engine, "_fx_gate_mode", return_value="off"), \
         patch.object(engine, "_has_open_position", return_value=False), \
         patch.object(engine, "_portfolio_risk_breach", return_value=None), \
         patch.object(
             engine, "_binance_position_cap_breach",
             return_value="Already have an open Binance position in BTCUSDT.",
         ), \
         patch(
             "backend.services.signal_candidate_engine.binance_market_data.get_klines",
             new_callable=AsyncMock,
             return_value=_feature_bars(),
         ):
        created = await engine.scan_markets(universe=["BTCUSDT"], timeframe="M5")
    assert created == []


@pytest.mark.asyncio
async def test_scan_markets_skips_recent_executed_cross_strategy_twin():
    """After a MACRO fill, MOMENTUM must not re-arm the same symbol+direction."""
    engine = SignalCandidateEngine()
    engine.candidates.clear()
    engine.candidates["filled-macro"] = {
        "id": "filled-macro",
        "symbol": "EURUSD",
        "broker": "ctrader",
        "strategy": "MACRO_EVENT_POST_REACTION",
        "direction": "SELL",
        "status": CandidateStatus.EXECUTED,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sizing": {"risk_usd": 1.0},
    }
    raw = _scan_raw(strategy="MOMENTUM_TREND_PULSE", direction="SELL")
    with patch.object(CTraderService, "get_trendbars", return_value=_feature_bars()), \
         patch.object(engine, "_attach_stop_atr", new_callable=AsyncMock), \
         patch.object(engine, "_evaluate_momentum", side_effect=lambda *a, **k: dict(raw)), \
         patch.object(engine, "_evaluate_fade", return_value=None), \
         patch.object(engine, "_evaluate_straddle", return_value=None), \
         patch.object(engine, "_evaluate_slingshot", return_value=None), \
         patch.object(engine, "_fx_gate_mode", return_value="off"), \
         patch.object(engine, "_has_open_position", return_value=False), \
         patch.object(engine, "_portfolio_risk_breach", return_value=None):
        created = await engine.scan_markets(universe=["EURUSD"], timeframe="M5")

    assert created == []
    live = [
        c for c in engine.candidates.values()
        if c["status"] in (CandidateStatus.PENDING, CandidateStatus.READY)
    ]
    assert live == []


@pytest.mark.asyncio
async def test_execute_refuses_when_binance_book_unreadable():
    engine = SignalCandidateEngine()
    engine.candidates.clear()
    engine.candidates["b-1"] = _binance_candidate("b-1", int(time.time()))

    with patch.dict(engine.execution_config, {"forex_only": False, "max_portfolio_risk_pct": 0}), patch(
        "backend.services.sentry_state.is_trading_allowed", return_value=True
    ), patch.object(
        engine, "_try_open_binance_positions", return_value=None
    ):
        res = await engine.execute_candidate("b-1", force=True)

    assert res["success"] is False
    assert res["skipped"] is True
    assert "Could not read Binance" in res["error"]
    assert engine.candidates["b-1"]["status"] == CandidateStatus.READY


def test_open_position_keys_counts_filled_db_rows_as_open():
    """Dashboard treats status=filled as open; scans must use the same book."""
    engine = SignalCandidateEngine()
    with patch(
        "backend.services.signal_candidate_engine.ctrader_service.status",
        return_value={"connected": True, "open_positions": 0},
    ), patch(
        "backend.services.signal_candidate_engine.ctrader_service.get_positions",
        return_value=[],
    ), patch(
        "backend.services.signal_candidate_engine.try_open_ctrader_db_positions",
        return_value=[{"symbol": "GBPUSD", "direction": "BUY"}],
    ):
        assert engine._open_position_keys("ctrader") == {("GBPUSD", "BUY")}
        assert engine._has_open_position("ctrader", "GBPUSD", "BUY") is True
        assert engine._has_open_position("ctrader", "GBPUSD", "SELL") is False


def test_open_position_keys_unknown_when_ctrader_db_unreadable_and_cache_empty():
    engine = SignalCandidateEngine()
    with patch(
        "backend.services.signal_candidate_engine.ctrader_service.status",
        return_value={"connected": False, "open_positions": 0},
    ), patch(
        "backend.services.signal_candidate_engine.ctrader_service.get_positions",
        return_value=[],
    ), patch(
        "backend.services.signal_candidate_engine.try_open_ctrader_db_positions",
        return_value=None,
    ):
        assert engine._open_position_keys("ctrader") is None
        assert engine._has_open_position("ctrader", "EURUSD", "SELL") is True


def test_has_open_position_fail_closed_when_book_unknown():
    engine = SignalCandidateEngine()
    with patch.object(engine, "_open_position_keys", return_value=None):
        assert engine._has_open_position("ctrader", "EURUSD", "SELL") is True
    with patch.object(engine, "_open_position_keys", return_value={("EURUSD", "SELL")}):
        assert engine._has_open_position("ctrader", "EURUSD", "SELL") is True
        assert engine._has_open_position("ctrader", "EURUSD", "BUY") is False
    with patch.object(engine, "_open_position_keys", return_value=set()):
        assert engine._has_open_position("ctrader", "EURUSD", "SELL") is False


def test_open_position_keys_maps_binance_side():
    engine = SignalCandidateEngine()
    with patch.object(
        engine,
        "_try_open_binance_positions",
        return_value=[{"symbol": "BTCUSDT", "side": "SELL"}],
    ):
        assert engine._open_position_keys("binance_futures") == {("BTCUSDT", "SELL")}


def test_prune_keeps_executed_when_over_cap():
    engine = SignalCandidateEngine()
    previous_cfg = dict(engine.execution_config)
    previous = dict(engine.candidates)
    try:
        engine.execution_config["max_candidates"] = 2
        engine.candidates = {
            "exec-old": {
                "id": "exec-old",
                "status": CandidateStatus.EXECUTED,
                "created_at": "2026-01-01",
                "executed_at": datetime.now(timezone.utc).isoformat(),
                "latest_exec_at": 1,
            },
            "ready-new": {
                "id": "ready-new",
                "status": CandidateStatus.READY,
                "created_at": "2026-09-11",
                "latest_exec_at": 9_999_999_999,
            },
            "ready-newer": {
                "id": "ready-newer",
                "status": CandidateStatus.READY,
                "created_at": "2026-09-12",
                "latest_exec_at": 9_999_999_999,
            },
        }
        engine.prune_candidates(now_ts=100)
        assert "exec-old" in engine.candidates
        assert engine.candidates["exec-old"]["status"] == CandidateStatus.EXECUTED
    finally:
        engine.execution_config = previous_cfg
        engine.candidates = previous


def test_cancel_live_same_direction_leaves_other_rows():
    engine = SignalCandidateEngine()
    engine.candidates.clear()
    engine.candidates["sell"] = {
        "id": "sell",
        "symbol": "EURUSD",
        "direction": "SELL",
        "status": CandidateStatus.READY,
    }
    engine.candidates["buy"] = {
        "id": "buy",
        "symbol": "EURUSD",
        "direction": "BUY",
        "status": CandidateStatus.READY,
    }
    engine.candidates["done"] = {
        "id": "done",
        "symbol": "EURUSD",
        "direction": "SELL",
        "status": CandidateStatus.EXECUTED,
    }
    cancelled = engine.cancel_live_same_direction(
        "EURUSD", "SELL", reason="go-live leftover"
    )
    assert cancelled == ["sell"]
    assert engine.candidates["sell"]["status"] == CandidateStatus.CANCELLED
    assert engine.candidates["buy"]["status"] == CandidateStatus.READY
    assert engine.candidates["done"]["status"] == CandidateStatus.EXECUTED
