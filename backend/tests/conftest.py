import os
import sys
from pathlib import Path

import pytest

_JESSE_QUANT = Path(__file__).resolve().parents[2] / "jesse_quant"
if str(_JESSE_QUANT) not in sys.path:
    sys.path.insert(0, str(_JESSE_QUANT))

@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def auth_headers(monkeypatch):
    """Admin token so trading/signal POSTs pass the fail-closed auth middleware."""
    key = (
        os.getenv("ADMIN_API_KEY")
        or os.getenv("API_AUTH_TOKEN")
        or os.getenv("BACKEND_API_KEY")
        or "test-admin-key"
    ).strip()
    monkeypatch.setenv("ADMIN_API_KEY", key)
    return {"X-API-Key": key}


@pytest.fixture(autouse=True)
def clean_trading_mode_for_tests(monkeypatch):
    """Default unit tests to live so Binance order-method tests are not blocked.

    Paper-mode tests (e.g. test_binance_paper_mode.py) monkeypatch TRADING_MODE
    to 'paper' in the test body, which overrides this fixture.
    """
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("PAPER_TRADING", "false")
    monkeypatch.setenv("DRY_RUN_ALL", "false")
    monkeypatch.setenv("BINANCE_PAPER_PARALLEL", "false")
    monkeypatch.setenv("CTRADER_ENV", "live")
    monkeypatch.setenv("CTRADER_PAPER_MODE", "false")
    monkeypatch.setenv("CTRADER_LIVE_CONFIRM", "I_UNDERSTAND")
    monkeypatch.setenv("DISABLE_DRAWDOWN_IN_TESTING", "false")
    monkeypatch.setenv("TESTING_MODE", "false")


@pytest.fixture(autouse=True)
def fx_venue_open_for_engine_tests(monkeypatch):
    """Keep IC/cTrader engine scans deterministic on weekend CI.

    ``is_venue_open`` follows Sun 21:00–Fri 21:00 UTC. Sunday runners would
    otherwise skip every FX scan before the assertion under test. Tests that
    need a closed venue patch ``signal_candidate_engine.is_venue_open`` locally.
    The real ``market_hours.is_venue_open`` is left untouched for hour tests.
    """
    monkeypatch.setattr(
        "backend.services.signal_candidate_engine.is_venue_open",
        lambda symbol, now=None: True,
    )


@pytest.fixture
def db_session():
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool
    from sqlalchemy.orm import sessionmaker
    from backend.database.models import Base
    from backend.database.connection import get_db
    from backend.main import app

    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()

    def override_get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield session
    finally:
        session.close()
        app.dependency_overrides.pop(get_db, None)

