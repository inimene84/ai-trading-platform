"""
FastAPI Routes for Jesse AI Quant Engine Bridge
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Any, Dict, Optional

from backend.ml.artifacts import default_artifact_dir
from backend.ml.promotion_service import resolve_promotion
from backend.services.jesse_bridge import jesse_bridge
from backend.services.risk_config import get_risk_config

router = APIRouter(tags=["Jesse Quant Engine"])

# Rejected B200 training outputs stay on the VPS. Promotion-status reports
# them only as documentation — never as a live model directory.
GPU_ARTIFACTS_NOTE = (
    "Rejected B200 training outputs stay under /root/gpu-artifacts/ on the VPS; "
    "they are not live models. Point QTP_PROMOTION_ARTIFACT_DIR at a sealed "
    "PROMOTE bundle only after gates pass. Do not copy rejected artifacts into "
    "jesse-trading/storage/models."
)


def _promotion_status_payload() -> Dict[str, Any]:
    """Read-only QTP promotion snapshot. Fail-closed when nothing is promoted."""
    artifact_root = default_artifact_dir()
    state = resolve_promotion(get_risk_config())
    payload: Dict[str, Any] = {
        "artifact_dir": str(artifact_root) if artifact_root is not None else None,
        "gpu_artifacts_note": GPU_ARTIFACTS_NOTE,
        "promoted": False,
    }
    if state is None:
        payload.update({
            "status": "unconfigured",
            "verdict": None,
            "reason": "no promotion artifacts loaded",
            "failed_gate": None,
            "warnings": [],
            "details": {},
        })
        return payload
    payload.update({
        "status": "ok",
        "verdict": state.verdict,
        "reason": state.result.reason,
        "failed_gate": state.result.failed_gate,
        "warnings": state.result.warnings,
        "details": state.result.details,
        "promoted": state.verdict == "PROMOTE",
    })
    return payload


class StrategySyncRequest(BaseModel):
    sl_atr_mult: float = 1.75
    tp_atr_mult: float = 5.5
    trail_activation_atr: float = 1.8
    trail_atr_mult: float = 1.6


class ValidationBacktestRequest(BaseModel):
    strategy: str = "QuantumAIStrategy"
    symbol: str = "BTC-USDT"
    timeframe: str = "1h"
    start_date: str = "2024-01-01"
    finish_date: str = "2024-04-01"
    starting_balance: float = 10000.0


@router.get("/status")
async def get_jesse_bridge_status() -> Dict[str, Any]:
    """Check connectivity and available datasets in the Jesse AI Quant Engine."""
    return await jesse_bridge.get_status()


@router.post("/sync")
def sync_strategy_parameters(req: StrategySyncRequest = StrategySyncRequest()) -> Dict[str, Any]:
    """Sync validated strategy parameters (SL/TP ATR multipliers) to live RiskConfig."""
    return jesse_bridge.sync_strategy_to_risk_config(
        sl_atr_mult=req.sl_atr_mult,
        tp_atr_mult=req.tp_atr_mult,
        trail_activation_atr=req.trail_activation_atr,
        trail_atr_mult=req.trail_atr_mult,
    )


@router.post("/validate")
async def run_validation_simulation(req: ValidationBacktestRequest) -> Dict[str, Any]:
    """Run an on-demand backtest validation via Jesse API."""
    res = await jesse_bridge.run_backtest_simulation(
        strategy=req.strategy,
        symbol=req.symbol,
        timeframe=req.timeframe,
        start_date=req.start_date,
        finish_date=req.finish_date,
        starting_balance=req.starting_balance,
    )
    if res.get("status") == "error":
        raise HTTPException(status_code=500, detail=res.get("error"))
    return res


class MLPredictionRequest(BaseModel):
    symbol: str = "BTC-USDT"
    timeframe: str = "1h"
    model_type: str = "lightgbm"
    threshold: float = 0.45


@router.get("/promotion-status")
def get_promotion_status() -> Dict[str, Any]:
    """Read-only promotion contract verdict. Does not place orders or hit /trading/*."""
    return _promotion_status_payload()


async def _ml_models_payload() -> Dict[str, Any]:
    """Sidecar health plus live promotion snapshot. Always JSON, never 404."""
    models = await jesse_bridge.get_ml_models()
    if not isinstance(models, dict):
        models = {"status": "error", "error": str(models)}
    promo = _promotion_status_payload()
    available = models.get("available_models") or []
    cached = models.get("cached_models") or []
    rejected = models.get("rejected_models") or {}
    return {
        **models,
        "status": models.get("status") or "error",
        "available_models": available,
        "cached_models": cached,
        "rejected_models": rejected,
        "has_promoted_model": bool(promo.get("promoted")),
        "promotion": promo,
    }


@router.get("/models", operation_id="get_jesse_models")
async def get_jesse_models() -> Dict[str, Any]:
    """Ops alias for `/ml-models` — list artifacts and promotion snapshot."""
    return await _ml_models_payload()


@router.get("/ml-models", operation_id="get_jesse_ml_models")
async def get_ml_models() -> Dict[str, Any]:
    """List trained ML artifacts plus the live promotion snapshot.

    Never implies a live model when the promotion contract is unconfigured
    or REJECT.
    """
    return await _ml_models_payload()


@router.get("/ml-predict")
async def get_ml_prediction(
    symbol: str = Query("BTC-USDT"),
    timeframe: str = Query("1h"),
    model_type: str = Query("lightgbm"),
    threshold: float = Query(0.45),
) -> Dict[str, Any]:
    """Query real-time Machine Learning prediction and probabilities via GET."""
    res = await jesse_bridge.get_ml_prediction(
        symbol=symbol,
        timeframe=timeframe,
        model_type=model_type,
        threshold=threshold,
    )
    if res.get("status") == "error":
        raise HTTPException(status_code=500, detail=res.get("error"))
    return res


@router.post("/ml-predict")
async def post_ml_prediction(req: MLPredictionRequest = MLPredictionRequest()) -> Dict[str, Any]:
    """Query real-time Machine Learning prediction and probabilities via POST."""
    res = await jesse_bridge.get_ml_prediction(
        symbol=req.symbol,
        timeframe=req.timeframe,
        model_type=req.model_type,
        threshold=req.threshold,
    )
    if res.get("status") == "error":
        raise HTTPException(status_code=500, detail=res.get("error"))
    return res


# ── FINMEM Cognitive Agent Routes (Stevens Institute / arXiv:2311.13743v2) ───

class FinMemEvalRequest(BaseModel):
    symbol: str = "BTC-USDT"
    timeframe: str = "1h"


class FinMemIngestRequest(BaseModel):
    symbol: str = "BTC-USDT"
    layer: str = "shallow"  # "shallow" | "intermediate" | "deep"
    content: str
    source_type: str = "news"
    base_importance: Optional[float] = None


@router.get("/finmem/status")
async def get_finmem_status(symbol: str = Query("BTC-USDT")) -> Dict[str, Any]:
    """Inspect active character setting, rolling return, and memory state for a symbol."""
    from backend.services.finmem_service import finmem_service
    char = finmem_service.get_character(symbol)
    return {
        "status": "ok",
        "symbol": symbol,
        "risk_mode": char.risk_mode,
        "active_inclination": char.current_inclination,
        "rolling_returns": char.rolling_returns,
        "cumulative_return": sum(char.rolling_returns),
        "collection": finmem_service.memory.collection,
    }


@router.post("/finmem/evaluate")
async def evaluate_finmem(req: FinMemEvalRequest = FinMemEvalRequest()) -> Dict[str, Any]:
    """Trigger FINMEM immediate reflection (layered memory recall + LLM decision)."""
    import os
    if os.getenv("FINMEM_ENABLED", "false").lower() != "true":
        raise HTTPException(
            status_code=503,
            detail="FINMEM evaluation is disabled (FINMEM_ENABLED=false)"
        )

    from backend.services.finmem_service import finmem_service

    # Fetch recent candles from Binance market data
    bars = []
    try:
        from backend.services.binance_market_data import binance_market_data
        clean_sym = req.symbol.replace("-", "").upper()
        bars = await binance_market_data.get_klines(clean_sym, interval=req.timeframe, limit=50)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch market data for {req.symbol} (fail-closed): {e}"
        )

    if not bars:
        raise HTTPException(
            status_code=503,
            detail=f"No market data bars available for {req.symbol} — fail closed (synthetic prices prohibited)"
        )

    dec = await finmem_service.immediate_reflect(req.symbol, bars)
    return {
        "status": "success",
        "symbol": dec.symbol,
        "action": dec.action,
        "confidence": dec.confidence,
        "risk_character": dec.risk_character,
        "reasoning": dec.reasoning,
        "cited_memory_ids": dec.cited_memory_ids,
        "retrieved_memory_count": dec.retrieved_memory_count,
        "momentum_3d": dec.momentum_3d,
        "latency_ms": dec.latency_ms,
    }


@router.post("/finmem/ingest")
async def ingest_finmem_memory(req: FinMemIngestRequest) -> Dict[str, Any]:
    """Ingest external financial insight into Shallow, Intermediate, or Deep memory layer."""
    from backend.services.finmem_service import finmem_service
    mem_id = await finmem_service.memory.store_memory(
        symbol=req.symbol,
        layer=req.layer,
        content=req.content,
        source_type=req.source_type,
        base_importance=req.base_importance,
    )
    return {"status": "stored", "memory_id": mem_id, "layer": req.layer, "symbol": req.symbol}

