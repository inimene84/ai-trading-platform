"""Blueprint upgrades: journal, calibration, conformal veto, dry-run quotes."""

from __future__ import annotations

import pytest

from backend.jev.calibration import calibration_report, fit_isotonic, temperature_scale
from backend.jev.conformal import AdaptiveConformal, reset_conformal
from backend.jev.evidence import gather_evidence
from backend.jev.gateway import MockJevClient, note_credit_failure, reset_circuit, circuit_open
from backend.jev.journal import DecisionJournal, canonical_hash, reset_journal_cache
from backend.jev.meta import meta_take, order_size_fraction, triple_barrier
from backend.jev.quote import live_submit_allowed, plan_quote
from backend.jev.service import clear_evaluation_cache, evaluate_symbol
from backend.jev.state import assert_no_lookahead, bar_time, build_market_state, rows_as_of


def _bars(count: int = 20) -> list[dict]:
    rows = []
    price = 100.0
    start = 1_700_000_000
    for index in range(count):
        price += 0.5
        rows.append({
            "time": start + index * 3600,
            "open": price - 0.2,
            "high": price + 0.4,
            "low": price - 0.4,
            "close": price,
            "volume": 10 + index,
        })
    return rows


def test_state_hash_is_stable_and_journal_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_JOURNAL_PATH", str(tmp_path / "journal.db"))
    reset_journal_cache()
    payload = {"b": 1, "a": {"z": 2, "y": 3}}
    assert canonical_hash(payload) == canonical_hash({"a": {"y": 3, "z": 2}, "b": 1})
    journal = DecisionJournal(tmp_path / "journal.db")
    recorded = journal.record(symbol="BTCUSDT", state=payload, status="ok", raw_answers={"trade_action": "HOLD"}, input_tokens=1000)
    assert recorded["cost_usd"] == pytest.approx(1000 * 0.042 / 1_000_000)
    assert journal.label(recorded["decision_id"], 1, "t+1") is True
    assert journal.label_count() == 1
    assert journal.label(999, 0) is False


def test_calibration_stays_display_only_until_enough_labels():
    sparse = calibration_report([{"label": 1, "answers": {"trade_action": "BUY", "trade_probabilities": {"BUY": 1.0}}}] * 3)
    assert sparse["ready"] is False
    assert sparse["sizing_allowed"] is False
    scaled = temperature_scale({"BUY": 0.5, "HOLD": 0.5}, 1.0)
    assert abs(sum(scaled.values()) - 1.0) < 1e-9
    curve = fit_isotonic([(0.1, 0), (0.2, 0), (0.3, 0), (0.4, 0), (0.8, 1), (0.85, 1), (0.9, 1), (0.95, 1)])
    assert curve is not None
    assert curve[0][1] <= curve[-1][1]


def test_triple_barrier_and_meta_never_size():
    closes = [100, 101, 103, 104]
    assert triple_barrier(closes, 0, upper_frac=0.02, lower_frac=0.02, horizon=3) == 1
    assert triple_barrier([100, 99, 97], 0, 0.02, 0.02, 3) == -1
    assert triple_barrier([100, 100.1, 100.1], 0, 0.02, 0.02, 2) == 0
    blocked = meta_take(labels=10, min_labels=200, margin=0.4, min_margin=0.12, probabilities={"UP": 0.8, "FLAT": 0.1, "DOWN": 0.1}, conformal_veto=False)
    assert blocked["take"] is False
    assert order_size_fraction(probability=0.99, equity=1_000_000) == 0.0


def test_adaptive_conformal_vetoes_a_wide_set_only_after_history():
    model = AdaptiveConformal(min_history=3, step=0.0, alpha=0.1)
    probs = {"UP": 0.34, "FLAT": 0.33, "DOWN": 0.33}
    assert model.veto(probs) is False
    for _ in range(3):
        model.observe(0.7, covered=False)
    model.threshold = 1.0
    assert len(model.prediction_set(probs)) == 3
    assert model.veto(probs) is True


def test_quote_plan_is_dry_run():
    opened = plan_quote(intent="open", bias="long", bid=100, ask=101, tick=0.1, quote_size=1)
    assert opened["submit"] is False
    assert opened["dry_run"] is True
    assert opened["order"]["type"] == "ALO"
    assert opened["order"]["price"] < 101
    closed = plan_quote(intent="close", bias="long", bid=100, ask=101, tick=0.1, position_size=2)
    assert closed["order"]["type"] == "IOC"
    assert closed["order"]["reduce_only"] is True
    assert closed["submit"] is False
    held = plan_quote(intent="hold", bias="long", bid=100, ask=101, tick=0.1)
    assert held["cancel_resting"] is True
    assert live_submit_allowed() is False


def test_lookahead_bars_are_rejected():
    bars = _bars(4)
    as_of = bars[2]["time"]
    kept = rows_as_of(bars, as_of)
    assert len(kept) == 3
    assert_no_lookahead(kept, as_of)
    with pytest.raises(ValueError):
        assert_no_lookahead(bars, as_of)
    state = build_market_state("BTC", _bars(20))
    assert state["market"]["monthly_context"]
    assert state["market"]["as_of"] == _bars(20)[-1]["time"]
    assert bar_time({"date": "2024-01-02T00:00:00Z"}) > 0


@pytest.mark.asyncio
async def test_evidence_category_failure_does_not_invent_hits():
    async def search(query: str):
        if "macro" in query:
            raise RuntimeError("source down")
        if "company" in query:
            return [{"title": "Desk note", "url": "https://example.test/a", "snippet": "rates steady into the close"}]
        return []

    gathered = await gather_evidence("AAPL", search=search)
    assert gathered["categories"]["macro_risks"] == []
    assert gathered["categories"]["company_news"]
    assert gathered["empty"] is False

    async def nothing(_query: str):
        return []

    empty = await gather_evidence("AAPL", search=nothing)
    assert empty["empty"] is True


def test_credit_errors_open_the_circuit(monkeypatch):
    monkeypatch.setenv("JEV_PAUSE_MS", "60000")
    reset_circuit()
    assert circuit_open() is False
    note_credit_failure("provider returned 402 insufficient credits")
    assert circuit_open() is True
    reset_circuit()


@pytest.mark.asyncio
async def test_mock_provider_cannot_influence_the_book(monkeypatch, tmp_path):
    monkeypatch.setenv("JEV_JOURNAL_PATH", str(tmp_path / "journal.db"))
    monkeypatch.setenv("JEV_INFLUENCE_BOOK", "true")
    monkeypatch.setenv("JEV_CALIBRATION_MIN_LABELS", "1")
    monkeypatch.setenv("JEV_CACHE_SECONDS", "0")
    reset_journal_cache()
    reset_conformal()
    clear_evaluation_cache()
    result = await evaluate_symbol("BTC", bars=_bars(20), include_social=False, client=MockJevClient(), fetch_bars=False)
    assert result["status"] == "ok"
    assert result["influence_book"] is False
    assert result["sizing_allowed"] is False
    assert result["order_size_fraction"] == 0.0
    assert result["state_hash"]


def test_signals_and_evidence_routes_stay_sensitive():
    from backend.security import is_sensitive_request
    from starlette.requests import Request

    def req(path: str) -> Request:
        return Request({"type": "http", "method": "GET", "path": path, "headers": []})

    assert is_sensitive_request(req("/signals/jev/crypto")) is True
    assert is_sensitive_request(req("/api/signals/jev/crypto")) is True
    assert is_sensitive_request(req("/jev/journal")) is True
    assert is_sensitive_request(req("/jev/evidence")) is True
