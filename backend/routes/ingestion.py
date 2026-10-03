"""
Canonical Ingestion API (v1)
============================
Strict, idempotent, versioned front door for all market scanner payloads.

Endpoints:
  POST /ingestion/v1/market-scans
  POST /api/ingestion/v1/market-scans (dual mounted)
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.database.models import IngestionMarketScan, IngestionOutbox
from backend.ingestion.schemas import MarketScan, ScanReceipt, canonical_payload_digest
from backend.security import AUTH_ENV_VARS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingestion/v1", tags=["ingestion-v1"])


def authenticate_ingestion_producer(
    x_ingestion_key: str | None = Header(None, alias="X-Ingestion-Key"),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None, alias="Authorization"),
) -> str:
    """
    Authenticate ingestion producers via X-Ingestion-Key or X-API-Key header,
    falling back to standard Bearer token or ADMIN_API_KEY.
    """
    import hmac

    configured_key = os.getenv("INGESTION_N8N_KEY", "").strip()

    candidate = ""
    if x_ingestion_key:
        candidate = x_ingestion_key.strip()
    elif x_api_key:
        candidate = x_api_key.strip()
    elif authorization and authorization.lower().startswith("bearer "):
        candidate = authorization[7:].strip()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing ingestion key. Provide X-Ingestion-Key or Bearer token.",
        )

    # 1. Match configured dedicated ingestion key
    if configured_key and hmac.compare_digest(candidate, configured_key):
        return "n8n_producer"

    # 2. Fall back to standard configured admin API keys
    for var in AUTH_ENV_VARS:
        expected = os.getenv(var, "").strip()
        if expected and hmac.compare_digest(candidate, expected):
            return "admin_producer"

    # If neither matched
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid ingestion key",
    )


@router.post(
    "/market-scans",
    response_model=ScanReceipt,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest canonical market scan envelope",
)
def ingest_market_scan(
    scan: MarketScan,
    response: Response,
    db: Session = Depends(get_db),
    producer: str = Depends(authenticate_ingestion_producer),
) -> ScanReceipt:
    """
    Ingest a batch of market observations into the immutable ledger and outbox.

    Guarantees:
      - Idempotent on (source, scan_id): identical retry returns HTTP 200 with duplicate=True.
      - Conflict detection: retry with altered payload returns HTTP 409 Conflict.
      - Atomic write: scan and outbox event are committed in a single transaction.
      - Freshness verification: stale scans are rejected on first insert but allowed on duplicate retry.
    """
    canonical_data, encoded_bytes, digest = canonical_payload_digest(scan)
    now_utc = datetime.now(timezone.utc)

    # 1. Check for existing scan with same (source, scan_id)
    existing = (
        db.query(IngestionMarketScan)
        .filter(
            IngestionMarketScan.source == scan.source,
            IngestionMarketScan.scan_id == scan.scan_id,
        )
        .first()
    )

    if existing:
        if existing.payload_sha256 != digest:
            logger.warning(
                f"[INGESTION CONFLICT] scan_id={scan.scan_id} from source={scan.source} "
                f"already exists with sha256={existing.payload_sha256}, rejected new sha256={digest}"
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Conflict: scan_id '{scan.scan_id}' already exists with a different payload digest",
            )

        # Identical retry
        outbox_event = (
            db.query(IngestionOutbox)
            .filter(
                IngestionOutbox.aggregate_id == str(existing.id),
                IngestionOutbox.aggregate_type == "market_scan",
            )
            .first()
        )

        response.status_code = status.HTTP_200_OK
        return ScanReceipt(
            ingestion_id=existing.id,
            scan_id=existing.scan_id,
            source=existing.source,
            received_at=existing.received_at,
            observation_count=existing.observation_count,
            duplicate=True,
            payload_sha256=existing.payload_sha256,
            event_id=outbox_event.event_id if outbox_event else None,
            trace_id=existing.trace_id,
        )

    # 2. Freshness check on first insert
    age_seconds = (now_utc - scan.scan_timestamp).total_seconds()
    max_age_seconds = float(os.getenv("MAX_INGESTION_SCAN_AGE_SECONDS", "900"))  # default 15m
    if age_seconds > max_age_seconds:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Scan timestamp is too stale ({age_seconds:.1f}s old, max allowed is {max_age_seconds:.0f}s)",
        )

    # 3. Insert scan row and outbox event in single atomic transaction
    event_id = f"evt_{uuid.uuid4().hex}"
    new_scan = IngestionMarketScan(
        scan_id=scan.scan_id,
        source=scan.source,
        scan_timestamp=scan.scan_timestamp,
        observation_count=len(scan.observations),
        payload_sha256=digest,
        payload=canonical_data,
        trace_id=scan.trace_id,
    )

    try:
        db.add(new_scan)
        db.flush()  # populate new_scan.id

        outbox_item = IngestionOutbox(
            event_id=event_id,
            event_type="market.scan.received",
            aggregate_type="market_scan",
            aggregate_id=str(new_scan.id),
            payload={
                "ingestion_id": new_scan.id,
                "scan_id": scan.scan_id,
                "source": scan.source,
                "scan_timestamp": scan.scan_timestamp.isoformat(),
                "observation_count": len(scan.observations),
                "payload_sha256": digest,
                "trace_id": scan.trace_id,
                "observations": canonical_data.get("observations", []),
            },
            status="pending",
            attempt_count=0,
            next_attempt_at=now_utc,
        )
        db.add(outbox_item)
        db.commit()
    except IntegrityError:
        db.rollback()
        # Concurrent race condition: reload existing record
        existing_concurrent = (
            db.query(IngestionMarketScan)
            .filter(
                IngestionMarketScan.source == scan.source,
                IngestionMarketScan.scan_id == scan.scan_id,
            )
            .first()
        )
        if existing_concurrent and existing_concurrent.payload_sha256 == digest:
            response.status_code = status.HTTP_200_OK
            return ScanReceipt(
                ingestion_id=existing_concurrent.id,
                scan_id=existing_concurrent.scan_id,
                source=existing_concurrent.source,
                received_at=existing_concurrent.received_at,
                observation_count=existing_concurrent.observation_count,
                duplicate=True,
                payload_sha256=existing_concurrent.payload_sha256,
                event_id=None,
                trace_id=existing_concurrent.trace_id,
            )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Concurrent insertion conflict on (source, scan_id)",
        )

    logger.info(
        f"[INGESTION SUCCESS] scan_id={new_scan.scan_id} source={new_scan.source} "
        f"obs={new_scan.observation_count} ingestion_id={new_scan.id} event_id={event_id}"
    )

    response.status_code = status.HTTP_201_CREATED
    return ScanReceipt(
        ingestion_id=new_scan.id,
        scan_id=new_scan.scan_id,
        source=new_scan.source,
        received_at=new_scan.received_at,
        observation_count=new_scan.observation_count,
        duplicate=False,
        payload_sha256=digest,
        event_id=event_id,
        trace_id=new_scan.trace_id,
    )
