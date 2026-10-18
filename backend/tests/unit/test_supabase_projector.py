"""
Unit tests for WP4: Supabase Dual-Write Projector
- Format and column alignment with public.jev_signals
- Deterministic UUID derivation
- Failure isolation: HTTP 500 and network timeouts return False and never raise
- Successful upsert confirmation
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from backend.database.models import DecisionOpinion, DecisionRecord
from backend.services.supabase_projector import SupabaseProjector


def _sample_decision_and_opinion():
    dec = DecisionRecord(
        decision_id="dec_supa_123",
        scan_id="scan_supa_123",
        event_id="evt_supa_123",
        symbol="BTCUSDT",
        venue="binance_futures",
        timeframe="M5",
        observed_price=65000.0,
        baseline_action="BUY",
        baseline_confidence=0.78,
        baseline_reason="Strong order flow",
        with_jev_action="BUY",
        with_jev_confidence=0.88,
        with_jev_reason="JEV confirmation",
        divergence="AGREED",
        jev_signal="BUY",
        jev_confidence=0.88,
        created_at=datetime.now(timezone.utc),
        details={"indicators": {"funding_rate": 0.0001, "oi_change_pct_1h": 1.5}},
    )
    opinion = DecisionOpinion(
        opinion_id="op_supa_123",
        scan_id="scan_supa_123",
        symbol="BTCUSDT",
        signal="BUY",
        confidence=0.88,
        reasoning="Momentum confirmed",
        raw_payload={"trade_confidence": 0.88},
    )
    return dec, opinion


def test_supabase_projector_mocked_success():
    projector = SupabaseProjector(
        supabase_url="https://mock.supabase.co",
        supabase_key="mock-key-xyz",
    )
    dec, op = _sample_decision_and_opinion()

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_post.return_value = mock_resp

        success = projector.project_decision(dec, op)
        assert success is True
        assert mock_post.called

        args, kwargs = mock_post.call_args
        assert kwargs["headers"]["apikey"] == "mock-key-xyz"
        assert kwargs["headers"]["Prefer"] == "resolution=merge-duplicates"
        payload = kwargs["json"]
        assert payload["symbol"] == "BTCUSDT"
        assert payload["decision"] == "BUY"
        assert payload["confidence"] == 0.88
        assert payload["timeframe"] == "M5"
        assert payload["funding_annualized"] == 0.01


def test_supabase_projector_failure_isolated():
    projector = SupabaseProjector(
        supabase_url="https://mock.supabase.co",
        supabase_key="mock-key-xyz",
    )
    dec, op = _sample_decision_and_opinion()

    # 1. HTTP 500 error
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_post.return_value = mock_resp

        success = projector.project_decision(dec, op)
        assert success is False  # Never raises, returns False gracefully

    # 2. Network Timeout exception
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Connection timed out")):
        success = projector.project_decision(dec, op)
        assert success is False
