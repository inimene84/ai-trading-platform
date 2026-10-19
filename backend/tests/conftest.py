import os
import sys
from pathlib import Path

import pytest

# --- Production DB guard ----------------------------------------------------
# Unit tests ran inside the live backend container with its DATABASE_URL and
# wrote fake cTrader trades (test-123 / sim_ct_*) into /app/data/hedge_fund.db.
# Point the app engine at a throwaway SQLite file before any backend module
# creates it, and refuse any connection to a production database file.
import tempfile as _tempfile

_PROD_DB_URLS = {u for u in (os.environ.get("DATABASE_URL"),) if u}
for _envfile in (Path(__file__).resolve().parents[2] / ".env", Path("/app/.env")):
    try:
        for _line in _envfile.read_text().splitlines():
            if _line.strip().startswith("DATABASE_URL="):
                _v = _line.split("=", 1)[1].strip().strip('"').strip("'")
                if _v:
                    _PROD_DB_URLS.add(_v)
    except OSError:
        pass
_PROD_DB_FILES = {
    "/app/data/hedge_fund.db",
    str(Path(__file__).resolve().parents[1] / "hedge_fund.db"),
    str(Path(__file__).resolve().parents[1] / "data" / "hedge_fund.db"),
}
for _u in list(_PROD_DB_URLS):
    if _u.startswith("sqlite:///"):
        _PROD_DB_FILES.add(os.path.realpath(_u[len("sqlite:///"):]))
_TEST_DB_DIR = _tempfile.mkdtemp(prefix="pytest-db-")
_TEST_DB_URL = f"sqlite:///{_TEST_DB_DIR}/test_hedge_fund.db"
os.environ["DATABASE_URL"] = _TEST_DB_URL
os.environ["PYTEST_DB_GUARD"] = "1"


def _is_prod_target(url_or_path: str) -> bool:
    if not url_or_path:
        return False
    if url_or_path in _PROD_DB_URLS:
        return True
    path = url_or_path
    if path.startswith("sqlite:///"):
        path = path[len("sqlite:///"):]
    if path in ("", ":memory:") or path.startswith("file::memory:"):
        return False
    return os.path.realpath(path) in {os.path.realpath(p) for p in _PROD_DB_FILES}


from sqlalchemy import event as _sa_event  # noqa: E402
from sqlalchemy.engine import Engine as _SAEngine  # noqa: E402


@_sa_event.listens_for(_SAEngine, "do_connect")
def _refuse_prod_db(dialect, conn_rec, cargs, cparams):
    target = str(cargs[0]) if cargs else str(cparams.get("database") or cparams.get("dsn") or "")
    if _is_prod_target(target):
        raise RuntimeError(f"pytest DB guard: refusing connection to production database {target!r}")


from backend.database import connection as _db_connection  # noqa: E402
from backend.database import models as _db_models  # noqa: E402

if _is_prod_target(str(_db_connection.engine.url)) or str(_db_connection.engine.url) != _TEST_DB_URL:
    pytest.exit(
        f"pytest DB guard: app engine points at {_db_connection.engine.url!s}, expected the temp test DB",
        returncode=3,
    )
_db_models.Base.metadata.create_all(_db_connection.engine)


@pytest.fixture(autouse=True)
def _no_ctrader_trade_persist(request, monkeypatch):
    """Execution paths must not write Trade rows from unit tests.

    Tests that exercise persistence itself (test_ctrader_trade_sync) keep the real
    function; everything else gets a recorder.
    """
    if request.module.__name__.endswith("test_ctrader_trade_sync"):
        return
    try:
        import backend.services.signal_candidate_engine as _sce
    except Exception:
        return
    calls = []
    monkeypatch.setattr(_sce, "persist_ctrader_execution", lambda *a, **k: calls.append((a, k)) or None)

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
def isolated_sentry_state(tmp_path, monkeypatch):
    """Keep unit tests off the /app volume and the persistent hedge_fund.db kill switch.

    ``write_state`` defaults to ``/app/data/sentry``, which CI/local agents cannot create.
    ``is_kill_switch_engaged()`` without a session also reads ``control_plane_states`` from
    the on-disk SQLite file and can fail-close every scan if a leftover halt row exists.
    File-backed sentry status still works via the temp dir; explicit ``db=`` checks keep
    the in-memory session used by control-plane tests.
    """
    monkeypatch.setenv("SENTRY_STATE_DIR", str(tmp_path / "sentry"))
    from backend.database.models import ControlPlaneState
    from backend.services import sentry_state as sentry_state_mod

    def _isolated(db=None) -> bool:
        status = sentry_state_mod.get_trading_status()
        if status in (
            sentry_state_mod.TradingStatus.HALTED_MANUAL,
            sentry_state_mod.TradingStatus.HALTED_BY_SENTRY,
        ):
            return True
        if db is None:
            return False
        latest = db.query(ControlPlaneState).order_by(ControlPlaneState.id.desc()).first()
        return bool(latest and latest.kill_switch_engaged)

    monkeypatch.setattr(sentry_state_mod, "is_kill_switch_engaged", _isolated)


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

