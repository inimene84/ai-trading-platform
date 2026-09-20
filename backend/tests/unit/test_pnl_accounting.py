"""Phase C: lifetime DB PnL is not cash; recon rows are excluded."""

from datetime import datetime, timezone

from backend.services.pnl_accounting import (
    LIFETIME_DB_PNL_NOTE,
    UNION_LABEL,
    attach_portfolio_accounting,
    is_reconciliation_trade,
    realized_windows,
    strategy_trades,
)


def test_reconciliation_rows_are_tagged_and_excluded():
    rows = [
        {"strategy": "combined", "pnl": 10.0, "closed_at": datetime(2026, 9, 20, tzinfo=timezone.utc)},
        {"strategy": "exchange_reconciliation", "pnl": 14000.0, "closed_at": datetime(2026, 9, 19, tzinfo=timezone.utc)},
    ]
    assert is_reconciliation_trade(rows[1]) is True
    assert len(strategy_trades(rows)) == 1
    windows = realized_windows(rows, now=datetime(2026, 9, 20, 12, tzinfo=timezone.utc), equity=800.0)
    assert windows["realized_today"] == 10.0
    assert windows["reconciliation_pnl"] == 14000.0
    assert windows["lifetime_db_pnl"] == 14010.0
    assert windows["lifetime_db_pnl_is_not_cash"] is True
    assert windows["lifetime_db_pnl_note"] == LIFETIME_DB_PNL_NOTE
    # The artifact percent is huge vs cash equity — that is the cockpit bug.
    assert windows["lifetime_db_pnl_pct"] > 1000.0


def test_attach_portfolio_accounting_retags_legacy_total_pnl():
    payload = attach_portfolio_accounting(
        {"equity": 800.0, "balance": 800.0},
        closed_trades=[
            {"strategy": "combined", "pnl": 5.0, "closed_at": datetime(2026, 9, 20, tzinfo=timezone.utc)},
        ],
        unrealized_pnl=12.5,
        now=datetime(2026, 9, 20, 15, tzinfo=timezone.utc),
    )
    assert payload["total_pnl_is_not_cash"] is True
    assert payload["total_pnl_source"] == "lifetime_db_artifact"
    assert payload["open_unrealized_pnl"] == 12.5
    assert payload["realized_today"] == 5.0
    assert payload["union_label"] == UNION_LABEL
