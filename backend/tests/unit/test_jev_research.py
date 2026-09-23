"""Market-research card, trust verdicts, and backtest intervals."""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

from backend.jev.backtest import horizon_flat_bps, multiclass_brier, score_t_plus_one
from backend.jev.evidence import select_passages
from backend.jev.questions import analysis_questions
from backend.jev.research import classify_research, lint_questions, research_questions
from backend.jev.trust import (
    allows_book_influence,
    expected_calibration_error,
    replay_hysteresis,
    sign_jsonl,
    trust_report,
    trust_verdict,
)


def _research_payload() -> dict:
    def choice(winner: str, allowed: tuple[str, ...]) -> dict:
        rest = (1.0 - 0.7) / (len(allowed) - 1)
        return {
            "choice": winner,
            "probabilities": {name: (0.7 if name == winner else rest) for name in allowed},
        }

    answers = {
        "higher_30d": choice("NOT_HIGHER", ("HIGHER", "NOT_HIGHER")),
        "outlook": {"score": 2.0},
        "evidence_quality": {"noul": 0.4},
    }
    for name in ("market_data", "company_news", "industry_news", "analyst_views", "macro_risks"):
        answers[f"bias_{name}"] = choice("NEUTRAL", ("BULLISH", "NEUTRAL", "BEARISH"))
    return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 10}}


def test_shipped_questions_pass_the_offline_lint():
    assert lint_questions(analysis_questions()) == []
    assert lint_questions(research_questions()) == []
    assert lint_questions({})


def test_horizon_bands_and_brier_are_research_scores():
    assert horizon_flat_bps("day") == pytest.approx(30.0)
    assert horizon_flat_bps("30d") == pytest.approx(200.0)
    scored = score_t_plus_one(
        [{"close": 100}, {"close": 102}, {"close": 102.02}],
        ["UP", "DOWN"],
        probability_rows=[({"UP": 1.0, "FLAT": 0.0, "DOWN": 0.0}, "UP")],
    )
    assert scored["wilson95"] is not None
    assert scored["brier"] == pytest.approx(multiclass_brier([({"UP": 1.0, "FLAT": 0.0, "DOWN": 0.0}, "UP")]))
    assert "not a live-trading promise" in scored["reason"]


def test_token_budget_keeps_citations_and_drops_duplicates():
    categories = {
        "company_news": [
            {"title": "A", "url": "https://example.test/a", "snippet": "alpha " * 40, "source_credibility": 0.9},
            {"title": "A again", "url": "https://example.test/b", "snippet": "alpha " * 40, "source_credibility": 0.2},
        ],
        "macro_risks": [
            {"title": "Macro", "url": "https://example.test/c", "snippet": "rates and inflation context", "source_credibility": 0.4},
        ],
    }
    chosen = select_passages(categories, token_budget=30)
    assert chosen
    assert all(item.get("url") for item in chosen)
    assert sum(len(item["snippet"]) // 4 for item in chosen) <= 40


def test_hysteresis_replays_without_a_new_call():
    path = replay_hysteresis([0.2, 0.8, 0.5, 0.1], enter=0.7, exit_below=0.3)
    assert path == ["out", "in", "in", "out"]
    with pytest.raises(ValueError):
        replay_hysteresis([0.5], enter=0.4, exit_below=0.4)


def test_trust_verdict_does_not_unlock_sizing():
    perfect = [(0.99, 1)] * 40
    ece = expected_calibration_error(perfect)
    assert ece is not None and ece <= 0.05
    assert trust_verdict(40, ece, min_labels=30) == "FACE_VALUE"
    assert trust_verdict(10, 0.01, min_labels=30) == "UNVERIFIED"
    assert trust_verdict(40, 0.2, min_labels=30) == "DOWNGRADE"
    assert allows_book_influence("DISCOUNT") is False
    report = trust_report([])
    assert report["verdict"] == "UNVERIFIED"
    assert report["sizing_allowed"] is False
    assert report["allows_book_influence"] is False


def test_unsigned_export_stays_unsigned(tmp_path):
    assert sign_jsonl("row\n", "")["signed"] is False
    key = Ed25519PrivateKey.generate()
    path = tmp_path / "key.pem"
    path.write_bytes(key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))
    signed = sign_jsonl("row\n", str(path))
    assert signed["signed"] is True
    assert signed["algorithm"] == "ed25519"


@pytest.mark.asyncio
async def test_research_card_is_off_and_fail_closed():
    categories = {"company_news": [{"title": "Note", "url": "https://example.test", "snippet": "earnings held steady"}]}
    disabled = await classify_research("AAPL", categories, enabled=False)
    assert disabled["status"] == "no_signal"
    assert disabled["classification"] is None
    assert disabled["influence_book"] is False
    assert disabled["order_size_fraction"] == 0.0

    class Fake:
        async def system_one(self, _state, _questions):
            return _research_payload()

    classified = await classify_research("aapl", categories, client=Fake(), enabled=True)
    assert classified["status"] == "ok"
    assert classified["classification"]["higher_30d"] == "NOT_HIGHER"
    assert classified["influence_book"] is False
    assert classified["sizing_allowed"] is False

    class Broken:
        async def system_one(self, _state, _questions):
            return {"answers": {}}

    failed = await classify_research("AAPL", categories, client=Broken(), enabled=True)
    assert failed["status"] == "no_signal"
    assert failed["classification"] is None
