"""add JEV Phase 2 pipeline tables (SQLite-compatible)

Revision ID: e4b7c1a9d2f0
Revises: c8f1e2a3b4d5
Create Date: 2026-09-23 17:20:00.000000

Additive tables for market-scan ingest, news sentiment, JEV eval logs, and
shadow/paper attribution. Uses JSON (not Postgres JSONB) so hedge_fund.db
and optional Postgres both apply.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e4b7c1a9d2f0"
down_revision: Union[str, None] = "c8f1e2a3b4d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    if "jev_market_scans" not in existing:
        op.create_table(
            "jev_market_scans",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("scan_timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("symbol", sa.String(length=20), nullable=False),
            sa.Column("timeframe", sa.String(length=10), nullable=False, server_default="M5"),
            sa.Column("signal_direction", sa.String(length=10), nullable=True),
            sa.Column("confidence", sa.Float(), nullable=True),
            sa.Column("signal_strength", sa.Float(), nullable=True),
            sa.Column("indicators", sa.JSON(), nullable=False),
            sa.Column("price_data", sa.JSON(), nullable=True),
            sa.Column("volume_data", sa.JSON(), nullable=True),
            sa.Column("scan_source", sa.String(length=50), nullable=False, server_default="n8n_market_scanner"),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.UniqueConstraint("symbol", "timeframe", "scan_timestamp", name="uq_jev_market_scan"),
        )
        op.create_index("ix_jev_market_scans_id", "jev_market_scans", ["id"])
        op.create_index("ix_jev_market_scans_scan_timestamp", "jev_market_scans", ["scan_timestamp"])
        op.create_index("ix_jev_market_scans_symbol", "jev_market_scans", ["symbol"])
        op.create_index("idx_jev_market_scan_symbol_time", "jev_market_scans", ["symbol", "scan_timestamp"])

    if "jev_news_sentiment_logs" not in existing:
        op.create_table(
            "jev_news_sentiment_logs",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("log_timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("source", sa.String(length=50), nullable=False),
            sa.Column("headline", sa.Text(), nullable=True),
            sa.Column("url", sa.Text(), nullable=True),
            sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("sentiment_score", sa.Float(), nullable=True),
            sa.Column("sentiment_label", sa.String(length=20), nullable=True),
            sa.Column("impact_rating", sa.String(length=10), nullable=True),
            sa.Column("categories", sa.JSON(), nullable=True),
            sa.Column("entities", sa.JSON(), nullable=True),
            sa.Column("full_data", sa.JSON(), nullable=True),
            sa.Column("market_scan_id", sa.Integer(), sa.ForeignKey("jev_market_scans.id"), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        )
        op.create_index("ix_jev_news_sentiment_logs_id", "jev_news_sentiment_logs", ["id"])
        op.create_index("ix_jev_news_sentiment_logs_log_timestamp", "jev_news_sentiment_logs", ["log_timestamp"])
        op.create_index("ix_jev_news_sentiment_logs_source", "jev_news_sentiment_logs", ["source"])
        op.create_index("ix_jev_news_sentiment_logs_market_scan_id", "jev_news_sentiment_logs", ["market_scan_id"])
        op.create_index("idx_jev_news_source_time", "jev_news_sentiment_logs", ["source", "log_timestamp"])

    if "jev_pipeline_evals" not in existing:
        op.create_table(
            "jev_pipeline_evals",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("log_timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("symbol", sa.String(length=20), nullable=False),
            sa.Column("timeframe", sa.String(length=10), nullable=False, server_default="1H"),
            sa.Column("jev_signal", sa.String(length=20), nullable=True),
            sa.Column("jev_confidence", sa.Float(), nullable=True),
            sa.Column("jev_strength", sa.Float(), nullable=True),
            sa.Column("probabilities", sa.JSON(), nullable=True),
            sa.Column("checks", sa.JSON(), nullable=True),
            sa.Column("market_context", sa.JSON(), nullable=True),
            sa.Column("news_sentiment_context", sa.JSON(), nullable=True),
            sa.Column("combined_context", sa.JSON(), nullable=True),
            sa.Column("combined_evaluation", sa.JSON(), nullable=True),
            sa.Column("outcome_prediction", sa.String(length=20), nullable=True),
            sa.Column("outcome_probability", sa.Float(), nullable=True),
            sa.Column("calibration_status", sa.String(length=30), nullable=True),
            sa.Column("execution_mode", sa.String(length=10), nullable=False, server_default="off"),
            sa.Column("execution_decision", sa.String(length=40), nullable=False, server_default="logged_only"),
            sa.Column("would_have_executed", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("executed", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("execution_id", sa.Integer(), nullable=True),
            sa.Column("quantum_trade_id", sa.String(length=64), nullable=True),
            sa.Column("source_scan_id", sa.Integer(), sa.ForeignKey("jev_market_scans.id"), nullable=True),
            sa.Column("evaluated", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("notes", sa.Text(), nullable=True),
        )
        op.create_index("ix_jev_pipeline_evals_id", "jev_pipeline_evals", ["id"])
        op.create_index("ix_jev_pipeline_evals_log_timestamp", "jev_pipeline_evals", ["log_timestamp"])
        op.create_index("ix_jev_pipeline_evals_symbol", "jev_pipeline_evals", ["symbol"])
        op.create_index("ix_jev_pipeline_evals_source_scan_id", "jev_pipeline_evals", ["source_scan_id"])
        op.create_index("idx_jev_pipeline_eval_symbol_time", "jev_pipeline_evals", ["symbol", "log_timestamp"])
        op.create_index("idx_jev_pipeline_eval_evaluated", "jev_pipeline_evals", ["evaluated"])
        op.create_index("idx_jev_pipeline_eval_executed", "jev_pipeline_evals", ["executed"])

    if "jev_pipeline_trades" not in existing:
        op.create_table(
            "jev_pipeline_trades",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("trade_timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("symbol", sa.String(length=20), nullable=False),
            sa.Column("direction", sa.String(length=10), nullable=False),
            sa.Column("quantity", sa.Float(), nullable=True),
            sa.Column("entry_price", sa.Float(), nullable=True),
            sa.Column("exit_price", sa.Float(), nullable=True),
            sa.Column("stop_loss", sa.Float(), nullable=True),
            sa.Column("take_profit", sa.Float(), nullable=True),
            sa.Column("pnl", sa.Float(), nullable=True),
            sa.Column("pnl_pct", sa.Float(), nullable=True),
            sa.Column("realized_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("source", sa.String(length=50), nullable=False, server_default="jev_pipeline"),
            sa.Column("mode", sa.String(length=20), nullable=False, server_default="shadow"),
            sa.Column("source_scan_id", sa.Integer(), sa.ForeignKey("jev_market_scans.id"), nullable=True),
            sa.Column("jev_eval_id", sa.Integer(), sa.ForeignKey("jev_pipeline_evals.id"), nullable=True),
            sa.Column("ledger_trade_id", sa.Integer(), nullable=True),
            sa.Column("outcome", sa.String(length=20), nullable=False, server_default="open"),
            sa.Column("outcome_reason", sa.Text(), nullable=True),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_jev_pipeline_trades_id", "jev_pipeline_trades", ["id"])
        op.create_index("ix_jev_pipeline_trades_trade_timestamp", "jev_pipeline_trades", ["trade_timestamp"])
        op.create_index("ix_jev_pipeline_trades_symbol", "jev_pipeline_trades", ["symbol"])
        op.create_index("ix_jev_pipeline_trades_mode", "jev_pipeline_trades", ["mode"])
        op.create_index("ix_jev_pipeline_trades_source_scan_id", "jev_pipeline_trades", ["source_scan_id"])
        op.create_index("ix_jev_pipeline_trades_jev_eval_id", "jev_pipeline_trades", ["jev_eval_id"])
        op.create_index("idx_jev_pipeline_trade_symbol_time", "jev_pipeline_trades", ["symbol", "trade_timestamp"])
        op.create_index("idx_jev_pipeline_trade_outcome", "jev_pipeline_trades", ["outcome"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())
    if "jev_pipeline_trades" in existing:
        op.drop_table("jev_pipeline_trades")
    if "jev_pipeline_evals" in existing:
        op.drop_table("jev_pipeline_evals")
    if "jev_news_sentiment_logs" in existing:
        op.drop_table("jev_news_sentiment_logs")
    if "jev_market_scans" in existing:
        op.drop_table("jev_market_scans")
