"""Phase 2 JEV pipeline: mode gate, auth, additive models, SQLite migration."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from backend.database.models import (
    ApiKey,
    HedgeFundFlow,
    JevMarketScan,
    JevNewsSentimentLog,
    JevPipelineEval,
    JevPipelineTrade,
    NewsArticle,
    PaperOrder,
    PaperPortfolio,
    ShadowOutcome,
    Trade,
)
from backend.jev.config import (
    DEFAULT_JEV_EXECUTION_MODE,
    jev_execution_mode,
    jev_live_execution_allowed,
    jev_may_call_execute,
)
from backend.security import is_sensitive_request, validate_admin_request
from backend.services.jev_pipeline import (
    calibration_snapshot,
    decide_execution,
    ingest_market_scan,
    ingest_news_sentiment,
)


EXISTING_TABLES = {
    "hedge_fund_flows",
    "hedge_fund_flow_runs",
    "hedge_fund_flow_run_cycles",
    "api_keys",
    "trading_signals",
    "trades",
    "paper_portfolios",
    "paper_orders",
    "paper_trades",
    "shadow_outcomes",
    "news_articles",
    "sentiment_scores",
    "calendar_events",
}

NEW_TABLES = {
    "jev_market_scans",
    "jev_news_sentiment_logs",
    "jev_pipeline_evals",
    "jev_pipeline_trades",
}


def _ok_eval(confidence: float = 0.81) -> dict:
    return {
        "status": "ok",
        "action": "BUY",
        "signal": "bullish",
        "confidence": confidence,
        "vetoed": False,
        "conflict": False,
    }


def _req(method: str, path: str) -> Request:
    return Request({"type": "http", "method": method, "path": path, "headers": []})


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    from backend.database.models import Base

    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_execution_mode_defaults_off_and_never_live(monkeypatch):
    monkeypatch.delenv("JEV_EXECUTION_MODE", raising=False)
    monkeypatch.delenv("JEV_LIVE_EXECUTION_CONFIRM", raising=False)
    assert DEFAULT_JEV_EXECUTION_MODE == "off"
    assert jev_execution_mode() == "off"
    assert jev_may_call_execute() is False
    assert jev_live_execution_allowed() is False

    monkeypatch.setenv("JEV_EXECUTION_MODE", "shadow")
    assert jev_execution_mode() == "shadow"
    assert jev_may_call_execute() is False
    assert jev_live_execution_allowed() is False

    monkeypatch.setenv("JEV_EXECUTION_MODE", "paper")
    assert jev_may_call_execute() is True
    assert jev_live_execution_allowed() is False

    monkeypatch.setenv("JEV_EXECUTION_MODE", "live")
    monkeypatch.setenv("JEV_LIVE_EXECUTION_CONFIRM", "OWNER_CONFIRMED")
    assert jev_execution_mode() == "live"
    assert jev_may_call_execute() is False
    assert jev_live_execution_allowed() is False

    monkeypatch.setenv("JEV_EXECUTION_MODE", "not-a-mode")
    assert jev_execution_mode() == "off"


def test_off_and_shadow_never_call_execute(monkeypatch):
    called: list[str] = []

    def boom(**_kwargs):
        called.append("execute")
        return {"success": True, "order_id": "should-not-run"}

    monkeypatch.setenv("JEV_EXECUTION_MODE", "off")
    off = decide_execution(evaluation=_ok_eval(), side="BUY", confidence=0.9, execute_fn=boom)
    assert off["called_execute"] is False
    assert off["executed"] is False
    assert off["would_have_executed"] is False
    assert off["execution_decision"] == "logged_only"
    assert called == []

    monkeypatch.setenv("JEV_EXECUTION_MODE", "shadow")
    shadow = decide_execution(evaluation=_ok_eval(), side="BUY", confidence=0.9, execute_fn=boom)
    assert shadow["called_execute"] is False
    assert shadow["executed"] is False
    assert shadow["would_have_executed"] is True
    assert shadow["execution_decision"] == "shadow_recorded"
    assert called == []

    monkeypatch.setenv("JEV_EXECUTION_MODE", "live")
    live = decide_execution(evaluation=_ok_eval(), side="BUY", confidence=0.9, execute_fn=boom)
    assert live["called_execute"] is False
    assert live["executed"] is False
    assert live["execution_decision"] == "blocked_live"
    assert called == []


def test_confidence_alone_does_not_execute(monkeypatch):
    called: list[str] = []

    def boom(**_kwargs):
        called.append("execute")
        return {"success": True}

    monkeypatch.setenv("JEV_EXECUTION_MODE", "off")
    result = decide_execution(evaluation=_ok_eval(0.99), side="BUY", confidence=0.99, execute_fn=boom)
    assert result["called_execute"] is False
    assert called == []

    monkeypatch.setenv("JEV_EXECUTION_MODE", "paper")
    vetoed = decide_execution(
        evaluation={**_ok_eval(0.99), "vetoed": True},
        side="BUY",
        confidence=0.99,
        execute_fn=boom,
    )
    assert vetoed["called_execute"] is False
    assert vetoed["execution_decision"] == "vetoed"
    assert called == []


def test_paper_calls_execute_only_when_gates_pass(monkeypatch):
    monkeypatch.setenv("JEV_EXECUTION_MODE", "paper")
    monkeypatch.setattr("backend.services.jev_pipeline.is_trading_allowed", lambda: True)
    called: list[dict] = []

    def fake_exec(**kwargs):
        called.append(kwargs)
        return {"success": True, "order_id": "paper-1"}

    result = decide_execution(evaluation=_ok_eval(0.8), side="BUY", confidence=0.8, execute_fn=fake_exec)
    assert result["called_execute"] is True
    assert result["executed"] is True
    assert result["execution_decision"] == "paper_filled"
    assert called and called[0]["side"] == "BUY"

    called.clear()
    monkeypatch.setattr("backend.services.jev_pipeline.is_trading_allowed", lambda: False)
    blocked = decide_execution(evaluation=_ok_eval(0.8), side="BUY", confidence=0.8, execute_fn=fake_exec)
    assert blocked["called_execute"] is False
    assert blocked["execution_decision"] == "blocked_sentry"
    assert called == []


def test_pipeline_routes_require_auth(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setenv("ADMIN_API_KEY", "secret")
    for path in (
        "/jev/ingest/market",
        "/api/jev/ingest/market",
        "/jev/pipeline/evaluate",
        "/api/jev/pipeline/orchestrate",
        "/jev/pipeline/calibration",
        "/data/collect-market",
        "/api/data/collect-news",
    ):
        assert is_sensitive_request(_req("POST", path)) is True
        with pytest.raises(HTTPException) as exc:
            validate_admin_request(_req("POST", path))
        assert exc.value.status_code == 401


def test_pipeline_http_auth_and_status(monkeypatch):
    from starlette.testclient import TestClient

    from backend.main import app

    monkeypatch.setenv("ADMIN_API_KEY", "test-secret-key")
    monkeypatch.setenv("CONFIRM_LIVE_DEPLOY", "true")
    monkeypatch.setenv("JEV_EXECUTION_MODE", "off")

    client = TestClient(app, raise_server_exceptions=False)
    denied = client.post("/api/jev/ingest/market", json={"symbol": "BTCUSDT", "indicators": {}})
    assert denied.status_code == 401
    denied_orch = client.post("/api/jev/pipeline/orchestrate")
    assert denied_orch.status_code == 401
    denied_alias = client.post("/api/data/collect-news", json={"source": "rss", "full_data": {}})
    assert denied_alias.status_code == 401
    status = client.get("/api/jev/pipeline/status", headers={"X-API-Key": "test-secret-key"})
    assert status.status_code == 200
    payload = status.json()
    assert payload["execution_mode"] == "off"
    assert payload["live_execution_allowed"] is False
    assert payload["may_call_execute"] is False
    assert payload["sizing_allowed"] is False


def test_no_parallel_fastapi_stub_vendored():
    root = Path(__file__).resolve().parents[2].parent
    assert not (root / "backend" / "MAIN.py").exists()
    assert not (root / "backend" / "api" / "routes" / "jev_evaluation.py").exists()
    assert not (root / "backend" / "api" / "routes" / "quantumtrade.py").exists()


def test_models_are_additive():
    from backend.database.models import Base

    names = set(Base.metadata.tables)
    missing = EXISTING_TABLES - names
    assert not missing, f"existing tables were removed: {sorted(missing)}"
    missing_new = NEW_TABLES - names
    assert not missing_new, f"new pipeline tables missing: {sorted(missing_new)}"
    assert HedgeFundFlow.__tablename__ == "hedge_fund_flows"
    assert Trade.__tablename__ == "trades"
    assert ApiKey.__tablename__ == "api_keys"
    assert PaperPortfolio.__tablename__ == "paper_portfolios"
    assert PaperOrder.__tablename__ == "paper_orders"
    assert ShadowOutcome.__tablename__ == "shadow_outcomes"
    assert NewsArticle.__tablename__ == "news_articles"
    assert JevMarketScan.__tablename__ == "jev_market_scans"
    assert JevNewsSentimentLog.__tablename__ == "jev_news_sentiment_logs"
    assert JevPipelineEval.__tablename__ == "jev_pipeline_evals"
    assert JevPipelineTrade.__tablename__ == "jev_pipeline_trades"
    trade_cols = {column.name for column in Trade.__table__.columns}
    for required in ("symbol", "direction", "quantity", "entry_price", "mode", "broker"):
        assert required in trade_cols


def test_connection_module_not_gutted():
    from backend.database import connection

    source = Path(connection.__file__).read_text(encoding="utf-8")
    assert "hedge_fund.db" in source
    assert "init_db_schema" in source
    assert "SessionLocal" in source
    assert "get_db" in source
    assert "create_all" in source
    assert hasattr(connection, "init_db_schema")
    assert hasattr(connection, "SessionLocal")
    assert hasattr(connection, "get_db")


def test_ingest_and_calibration_on_sqlite(db_session):
    scan = ingest_market_scan(
        db_session,
        {
            "symbol": "btcusdt",
            "timeframe": "M5",
            "signal_direction": "LONG",
            "confidence": 0.7,
            "indicators": {"rsi": 55},
        },
    )
    assert scan.id is not None
    assert scan.symbol == "BTCUSDT"
    news = ingest_news_sentiment(
        db_session,
        {
            "source": "cryptopanic",
            "headline": "BTC ETF inflows rise",
            "sentiment_score": 0.4,
            "sentiment_label": "bullish",
            "full_data": {"symbol": "BTCUSDT"},
        },
    )
    assert news.id is not None
    report = calibration_snapshot(db_session)
    assert report["sizing_allowed"] is False
    assert report["execution_mode"] in {"off", "shadow", "paper", "live"}
    assert "Do not set JEV_EXECUTION_MODE=live" in report["note"]


def test_migration_upgrade_applies_on_sqlite(tmp_path):
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    from backend.alembic.versions.e4b7c1a9d2f0_add_jev_pipeline_tables import downgrade, upgrade

    engine = create_engine(f"sqlite:///{tmp_path}/pipeline.db")
    with engine.begin() as conn:
        context = MigrationContext.configure(conn)
        with Operations.context(context):
            upgrade()
    tables = set(inspect(engine).get_table_names())
    assert NEW_TABLES <= tables
    with engine.connect() as conn:
        columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(jev_pipeline_evals)").fetchall()}
    assert "execution_mode" in columns
    assert "would_have_executed" in columns
    assert "executed" in columns
    with engine.begin() as conn:
        context = MigrationContext.configure(conn)
        with Operations.context(context):
            downgrade()
    tables_after = set(inspect(engine).get_table_names())
    assert not (NEW_TABLES & tables_after)
