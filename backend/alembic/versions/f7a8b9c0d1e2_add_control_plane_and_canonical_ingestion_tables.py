"""add control plane and canonical ingestion tables

Revision ID: f7a8b9c0d1e2
Revises: e4b7c1a9d2f0
Create Date: 2026-10-03 06:40:00.000000

Adds:
- control_plane_states: persistent kill-switch audit trail and safety states
- ingestion_market_scans: canonical immutable ledger of market scans
- ingestion_outbox: transactional outbox for reliable asynchronous event delivery
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f7a8b9c0d1e2"
down_revision: Union[str, None] = "e4b7c1a9d2f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    if "control_plane_states" not in existing:
        op.create_table(
            "control_plane_states",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("kill_switch_engaged", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("actor", sa.String(length=100), nullable=True),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column("details", sa.JSON(), nullable=True),
            sa.Column("engaged_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("idx_control_plane_engaged", "control_plane_states", ["kill_switch_engaged"])
        op.create_index("idx_control_plane_created", "control_plane_states", ["created_at"])

    if "ingestion_market_scans" not in existing:
        op.create_table(
            "ingestion_market_scans",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("scan_id", sa.String(length=64), nullable=False),
            sa.Column("source", sa.String(length=50), nullable=False),
            sa.Column("scan_timestamp", sa.DateTime(timezone=True), nullable=False),
            sa.Column("received_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column("observation_count", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("payload_sha256", sa.String(length=64), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("trace_id", sa.String(length=64), nullable=True),
            sa.UniqueConstraint("source", "scan_id", name="uq_ingestion_source_scan"),
        )
        op.create_index("idx_ingestion_scans_received", "ingestion_market_scans", ["received_at"])
        op.create_index("idx_ingestion_scans_source_time", "ingestion_market_scans", ["source", "scan_timestamp"])
        op.create_index("idx_ingestion_scans_sha", "ingestion_market_scans", ["payload_sha256"])

    if "ingestion_outbox" not in existing:
        op.create_table(
            "ingestion_outbox",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("event_id", sa.String(length=64), unique=True, nullable=False),
            sa.Column("event_type", sa.String(length=50), nullable=False),
            sa.Column("aggregate_type", sa.String(length=50), nullable=False, server_default="market_scan"),
            sa.Column("aggregate_id", sa.String(length=64), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
            sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="5"),
            sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("locked_by", sa.String(length=64), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("idx_outbox_status_attempt", "ingestion_outbox", ["status", "next_attempt_at"])
        op.create_index("idx_outbox_event_type", "ingestion_outbox", ["event_type"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    if "ingestion_outbox" in existing:
        op.drop_table("ingestion_outbox")
    if "ingestion_market_scans" in existing:
        op.drop_table("ingestion_market_scans")
    if "control_plane_states" in existing:
        op.drop_table("control_plane_states")
