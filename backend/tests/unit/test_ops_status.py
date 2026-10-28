from backend.services.ops_status import trading_ops_snapshot


def test_ops_snapshot_exposes_split_book_and_llm(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("ACTIVE_BROKER", "ctrader")
    monkeypatch.setenv("CTRADER_ENV", "sandbox")
    monkeypatch.setenv("BINANCE_TESTNET", "false")
    snap = trading_ops_snapshot()
    assert snap["equity_books"]["split_book"] is True
    assert snap["equity_books"]["risk_broker"] == "binance_futures"
    assert snap["equity_books"]["union_label"] == "union = cTrader + Binance"
    assert "last_error" in snap["llm_router"]
    assert snap["sentry_auto_resume"]["requires_live_confirm"] is True
    assert snap["sentry_auto_resume"]["enabled"] is False
    assert snap["portfolio_risk"]["funding_gate"]["long_block_rate"] == 0.0003
    assert snap["portfolio_risk"]["crypto_cluster"]["max_concurrent_positions"] == 3
    assert snap["portfolio_risk"]["drawdown_breaker"]["hard_mode"] == "SHADOW_ONLY"
    assert snap["portfolio_risk"]["forward_gate_notional_cap_usd"] == 150.0
