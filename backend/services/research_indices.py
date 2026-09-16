"""OpenSearch index template bootstrap for QuantumTrade Pro Research Plane.

Applies templates at backend startup only if RESEARCH_INGEST_ENABLED=true.
Startup MUST succeed even if OpenSearch is offline or down.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict

from backend.services.opensearch_client import opensearch_client

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "opensearch" / "templates"

INDEX_TEMPLATES: Dict[str, str] = {
    "qt-news": "qt-news.json",
    "qt-events": "qt-events.json",
    "qt-decisions": "qt-decisions.json",
    "qt-mt-bars": "qt-mt-bars.json",
}


def is_research_ingest_enabled() -> bool:
    return os.getenv("RESEARCH_INGEST_ENABLED", "false").strip().lower() in ("true", "1", "yes")


async def init_research_indices() -> bool:
    """Initialize OpenSearch indices if research ingest is enabled.
    
    Fail-soft: logs warnings on connectivity issues, never raises.
    """
    if not is_research_ingest_enabled():
        logger.debug("Research ingest is disabled; skipping OpenSearch index initialization")
        return False

    logger.info("RESEARCH_INGEST_ENABLED=true; verifying OpenSearch connectivity and indices...")
    try:
        is_healthy = await opensearch_client.ping(timeout=3.0)
        if not is_healthy:
            logger.warning("OpenSearch cluster unreachable; continuing startup without index bootstrap")
            return False

        for index_name, template_file in INDEX_TEMPLATES.items():
            template_path = TEMPLATES_DIR / template_file
            if not template_path.exists():
                logger.warning(f"Template file missing: {template_path}")
                continue

            try:
                mapping = json.loads(template_path.read_text(encoding="utf-8"))
                success = await opensearch_client.ensure_index(index_name, mapping)
                if success:
                    logger.info(f"OpenSearch index '{index_name}' verified/created successfully")
                else:
                    logger.warning(f"Could not ensure index '{index_name}'")
            except Exception as exc:
                logger.warning(f"Failed to load or apply template {template_file}: {exc}")

        return True
    except Exception as exc:
        logger.warning(f"Failed to bootstrap OpenSearch research indices: {exc}")
        return False
