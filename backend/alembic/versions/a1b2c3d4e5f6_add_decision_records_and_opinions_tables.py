"""add decision records and opinions tables

Revision ID: a1b2c3d4e5f6
Revises: f7a8b9c0d1e2
Create Date: 2026-10-03 07:00:00.000000

Adds:
- decision_records: authoritative record of trading decisions with baseline vs counterfactual (with-JEV) traces
- decision_opinions: structured advisory opinions emitted during observation evaluation
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "f7a8b9c0d1e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    if "decision_records" not in existing:
        op.create_table(
            "decision_records",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("decision_id", sa.String(length=64), unique=True, nullable=False),
            sa.Column("trace_id", sa.String(length=64), nullable=True),
            sa.Column("scan_id", sa.String(length=64), nullable=False),
            sa.Column("event_id", sa.String(length=64), nullable=False),
            sa.Column("symbol", sa.String(length=32), nullable=False),
            sa.Column("venue", sa.String(length=32), nullable=False, server_default="binance_futures"),
            sa.Column("timeframe", sa.String(length=16), nullable=False, server_default="M5"),
            sa.Column("observed_price", sa.Float(), nullable=True),
            sa.Column("baseline_action", sa.String(length=32), nullable=False, server_default="HOLD"),
            sa.Column("baseline_confidence", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("baseline_reason", sa.Text(), nullable=True),
            sa.Column("with_jev_action", sa.String(length=32), nullable=False, server_default="HOLD"),
            sa.Column("with_jev_confidence", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("with_jev_reason", sa.Text(), nullable=True),
            sa.Column("divergence", sa.String(length=32), nullable=False, server_default="AGREED"),
            sa.Column("jev_signal", sa.String(length=32), nullable=True),
            sa.Column("jev_confidence", sa.Float(), nullable=True),
            sa.Column("jev_vetoed", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("jev_conflict", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("jev_reason", sa.Text(), nullable=True),
            sa.Column("execution_mode", sa.String(length=32), nullable=False, server_default="shadow"),
            sa.Column("executed", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("order_intent_id", sa.String(length=64), nullable=True),
            sa.Column("details", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        )
        op.create_index("idx_decision_records_scan", "decision_records", ["scan_id"])
        op.create_index("idx_decision_records_symbol_created", "decision_records", ["symbol", "created_at"])
        op.create_index("idx_decision_records_divergence", "decision_records", ["divergence"])

    if "decision_opinions" not in existing:
        op.create_table(
            "decision_opinions",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("opinion_id", sa.String(length=64), unique=True, nullable=False),
            sa.Column("trace_id", sa.String(length=64), nullable=True),
            sa.Column("scan_id", sa.String(length=64), nullable=False),
            sa.Column("symbol", sa.String(length=32), nullable=False),
            sa.Column("source", sa.String(length=32), nullable=False, server_default="jev_analyst"),
            sa.Column("signal", sa.String(length=32), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("reasoning", sa.Text(), nullable=True),
            sa.Column("vetoed", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("conflict", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("raw_payload", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        )
        op.create_index("idx_decision_opinions_scan_symbol", "decision_opinions", ["scan_id", "symbol"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    if "decision_opinions" in existing:
        op.drop_table("decision_opinions")
    if "decision_records" in existing:
        op.drop_table("decision_records")
