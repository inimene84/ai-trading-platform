"""
Jesse-to-QuantumTrade Live Strategy Bridge
Enables communication between the Jesse Quant Research engine and the live
QuantumTrade trading platform:
- Queries Jesse candle storage and health
- Dispatches automated on-demand strategy backtests
- Synchronizes optimized strategy parameters (SL/TP ATR, Trailing thresholds) into live RiskConfig
"""

import asyncio
import os
import re
import uuid
import logging
import httpx
from dotenv import load_dotenv
from typing import Any, Dict, Optional

from backend.services.jesse_ml_gates import (
    STRATEGY_PT_ATR,
    STRATEGY_SL_ATR,
    annotate_ml_prediction,
)
from backend.services.risk_config import refresh_risk_config
from backend.services.trading_mode import live_exchange_orders_allowed

logger = logging.getLogger(__name__)

# Ensure .env is loaded even if this module is imported before backend.main
load_dotenv()

JESSE_API_URL = os.getenv("JESSE_API_URL", "http://jesse-app:9000")
JESSE_LOCAL_URL = os.getenv("JESSE_LOCAL_URL", "http://127.0.0.1:9000")
JESSE_ML_URL = os.getenv("JESSE_ML_URL", "http://jesse-app:9003")
JESSE_ML_LOCAL_URL = os.getenv("JESSE_ML_LOCAL_URL", "http://127.0.0.1:9003")
# Fail-closed: no hardcoded dashboard password. The bridge stays
# unavailable (get_status reports available=False) until JESSE_PASSWORD
# is set in the environment.
JESSE_PASSWORD = os.getenv("JESSE_PASSWORD", "").strip()

# QTP training-tree shadow ingest lives in /root/qtp-training (scripts/ingest_shadow.py).
# QTP_SHADOW_INGEST_ENABLED is documentation-only on this process and stays false.
# Do not copy QTP bundles into jesse-trading/storage/models, do not set
# JESSE_SYNC_TO_LIVE, and do not point the live ML gate at a new artifact.
# QTP_SHADOW_INGEST_ENABLED=false

# Predict errors that mean "this symbol has no live model" — not an ML outage.
# Live fail-closed must not treat these as a veto or the whole universe stalls
# when only a subset of pairs have a promoted LightGBM artifact.
_JESSE_ML_NO_ARTIFACT = "no model artifact found"
_JESSE_ML_PROMOTION_FAILED = "promotion gate failed"


def is_jesse_ml_model_gap(error: Optional[str]) -> bool:
    """True when Jesse ML refused a predict because no deployable model exists.

    Examples from the sidecar:
      - "No model artifact found for AVAX-USDT (1h, lightgbm)"
      - "Model artifact ETH-USDT_1h_lightgbm.joblib refused: promotion gate failed ..."
    """
    text = str(error or "").lower()
    if not text:
        return False
    return _JESSE_ML_NO_ARTIFACT in text or _JESSE_ML_PROMOTION_FAILED in text


class JesseBridgeService:
    def __init__(self):
        self._token: Optional[str] = None
        self._base_url = JESSE_API_URL
        self._ml_url = JESSE_ML_URL

    async def _resolve_url(self) -> str:
        """Dynamically detect if internal Docker hostname or local loopback is reachable."""
        for url in [self._base_url, JESSE_LOCAL_URL]:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    res = await client.get(f"{url}/system/system-info")
                    if res.status_code in (200, 401, 403):
                        self._base_url = url
                        return url
            except Exception:
                continue
        return self._base_url

    async def _resolve_ml_url(self) -> str:
        """Detect reachable ML inference endpoint (internal docker vs host loopback)."""
        for url in [self._ml_url, JESSE_ML_LOCAL_URL]:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    res = await client.get(f"{url}/health")
                    if res.status_code == 200:
                        self._ml_url = url
                        return url
            except Exception:
                continue
        return self._ml_url

    async def get_token(self) -> Optional[str]:
        if self._token:
            return self._token
        if not JESSE_PASSWORD:
            logger.warning(
                "JESSE_PASSWORD is not set; Jesse bridge authentication disabled (fail-closed)."
            )
            return None
        base = await self._resolve_url()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.post(
                    f"{base}/auth/login",
                    json={"password": JESSE_PASSWORD},
                )
                if res.status_code == 200:
                    self._token = res.json().get("auth_token")
                    return self._token
        except Exception as e:
            logger.warning(f"Jesse authentication failed: {e}")
        return None

    async def get_status(self) -> Dict[str, Any]:
        """Check Jesse service connectivity and return candle/system stats."""
        base = await self._resolve_url()
        token = await self.get_token()
        if not token:
            return {
                "available": False,
                "url": base,
                "error": "Could not authenticate to Jesse API",
            }

        headers = {"Authorization": token}
        status_info = {
            "available": True,
            "url": base,
            "datasets": [],
            "strategies": [],
        }

        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                # 1. Existing Candlestick Datasets
                c_res = await client.post(f"{base}/candles/existing", headers=headers, json={})
                if c_res.status_code == 200:
                    status_info["datasets"] = c_res.json().get("data", [])

                # 2. Available Strategies
                s_res = await client.get(f"{base}/strategy/all", headers=headers)
                if s_res.status_code == 200:
                    status_info["strategies"] = s_res.json().get("strategies", [])
        except Exception as e:
            status_info["error"] = str(e)

        return status_info

    async def run_backtest_simulation(
        self,
        strategy: str = "QuantumAIStrategy",
        symbol: str = "BTC-USDT",
        timeframe: str = "1h",
        start_date: str = "2024-01-01",
        finish_date: str = "2024-04-01",
        starting_balance: float = 10000.0,
    ) -> Dict[str, Any]:
        """Dispatch a fast backtest simulation via Jesse API and return performance metrics."""
        base = await self._resolve_url()
        token = await self.get_token()
        if not token:
            return {"status": "error", "error": "Jesse authentication failed"}

        session_id = str(uuid.uuid4())
        exchange = "Binance Perpetual Futures"
        payload = {
            "id": session_id,
            "exchange": exchange,
            "routes": [
                {
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "strategy": strategy,
                }
            ],
            "data_routes": [],
            "config": {
                "warm_up_candles": 210,
                "logging": {
                    "strategy_execution": False,
                    "order_submission": True,
                    "order_cancellation": True,
                    "order_execution": True,
                    "position_opened": True,
                    "position_increased": True,
                    "position_reduced": True,
                    "position_closed": True,
                    "shorter_period_candles": False,
                    "trading_candles": True,
                    "balance_update": True,
                },
                "exchanges": {
                    exchange: {
                        "fee": 0.0006,
                        "type": "futures",
                        "balance": starting_balance,
                        "futures_leverage": 1,
                        "futures_leverage_mode": "cross",
                    }
                },
            },
            "start_date": start_date,
            "finish_date": finish_date,
            "debug_mode": False,
            "export_csv": False,
            "export_json": False,
            "export_chart": False,
            "fast_mode": True,
            "benchmark": False,
            "state": {},
        }

        headers = {"Authorization": token, "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(f"{base}/backtest", headers=headers, json=payload)
                if res.status_code not in (200, 201, 202):
                    return {"status": "error", "error": res.text}

                # Poll for completion
                for _ in range(30):
                    await asyncio.sleep(0.5)
                    sess_res = await client.post(
                        f"{base}/backtest/sessions",
                        headers=headers,
                        json={"limit": 5, "offset": 0},
                    )
                    if sess_res.status_code == 200:
                        sessions = sess_res.json().get("sessions", [])
                        for s in sessions:
                            if s.get("id") == session_id and s.get("status") == "finished":
                                return {
                                    "status": "success",
                                    "session_id": session_id,
                                    "net_profit_pct": s.get("net_profit_percentage"),
                                    "win_rate": s.get("win_rate"),
                                    "total_trades": s.get("total_trades"),
                                    "duration_seconds": s.get("execution_duration"),
                                }

                return {"status": "timeout", "session_id": session_id}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def sync_strategy_to_risk_config(
        self,
        sl_atr_mult: float = STRATEGY_SL_ATR,
        tp_atr_mult: float = STRATEGY_PT_ATR,
        trail_activation_atr: float = 1.8,
        trail_atr_mult: float = 1.6,
    ) -> Dict[str, Any]:
        """Apply validated quant strategy parameters directly to active RiskConfig."""
        if os.getenv("JESSE_SYNC_TO_LIVE", "false").lower() != "true":
            logger.info("JESSE_SYNC_TO_LIVE is disabled — strategy sync is a no-op")
            return {
                "status": "blocked",
                "message": "JESSE_SYNC_TO_LIVE is disabled; sync is a no-op",
                "synced": False,
            }

        env_path = os.getenv("ENV_FILE_PATH", ".env")
        updates = {
            "SL_ATR_MULT": str(sl_atr_mult),
            "TP_ATR_MULT": str(tp_atr_mult),
            "TRAIL_ACTIVATION_ATR": str(trail_activation_atr),
            "TRAIL_ATR_MULT": str(trail_atr_mult),
        }

        # Safely update .env in-place without replacing file inode (works on bind-mounted .env)
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                remaining = set(updates.keys())
                new_lines = []
                for line in lines:
                    matched = False
                    for k, v in updates.items():
                        if re.match(rf"^(export\s+)?{k}=", line.strip()):
                            prefix = "export " if line.strip().startswith("export ") else ""
                            new_lines.append(f"{prefix}{k}={v}\n")
                            matched = True
                            remaining.discard(k)
                            break
                    if not matched:
                        new_lines.append(line)

                for k in sorted(remaining):
                    new_lines.append(f"{k}={updates[k]}\n")

                with open(env_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
            except Exception as e:
                logger.warning(f"Could not persist parameters to {env_path}: {e}")

        for k, v in updates.items():
            os.environ[k] = v

        cfg = refresh_risk_config()
        logger.info(
            f"Synced Jesse parameters to RiskConfig: SL={sl_atr_mult} ATR, TP={tp_atr_mult} ATR, "
            f"TrailActivation={trail_activation_atr} ATR, TrailDist={trail_atr_mult} ATR"
        )
        return {
            "status": "synced",
            "sl_atr_mult": cfg.sl_atr_mult,
            "tp_atr_mult": cfg.tp_atr_mult,
            "trail_activation_atr": cfg.trail_activation_atr,
            "trail_atr_mult": cfg.trail_atr_mult,
        }

    async def get_ml_prediction(
        self,
        symbol: str = "BTC-USDT",
        timeframe: str = "1h",
        model_type: str = "lightgbm",
        threshold: float = 0.45,
    ) -> Dict[str, Any]:
        """Fetch ultra-low-latency real-time ML direction prediction and probabilities."""
        base_ml = await self._resolve_ml_url()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.post(
                    f"{base_ml}/predict",
                    json={
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "model_type": model_type,
                        "threshold": threshold,
                    },
                )
                if res.status_code == 200:
                    payload = res.json()
                    if isinstance(payload, dict):
                        payload = annotate_ml_prediction(payload)
                        if payload.get("status") != "success":
                            err = str(payload.get("error") or "")
                            if is_jesse_ml_model_gap(err):
                                payload["status"] = "no_model"
                    return payload
                # Never rewrite 5xx bodies to no_model — a traceback that
                # happens to mention artifacts is still an outage.
                err = f"ML server returned HTTP {res.status_code}: {res.text}"
                if 400 <= res.status_code < 500 and is_jesse_ml_model_gap(res.text):
                    return {"status": "no_model", "error": err}
                return {"status": "error", "error": err}
        except Exception as e:
            logger.error(f"Failed to query ML prediction endpoint: {e}")
            return {"status": "error", "error": str(e)}

    async def get_meta_prediction(
        self,
        symbol: str,
        primary_signal: str,
        timeframe: str = "1h",
        model_type: str = "lightgbm",
        min_prob: float = 0.45,
    ) -> Dict[str, Any]:
        """Fetch secondary meta-model trade filter and Fractional Kelly sizing recommendation."""
        base_ml = await self._resolve_ml_url()
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.post(
                    f"{base_ml}/meta-predict",
                    json={
                        "symbol": symbol,
                        "primary_signal": primary_signal,
                        "timeframe": timeframe,
                        "model_type": model_type,
                        "min_prob": min_prob,
                    },
                )
                if res.status_code == 200:
                    body = res.json()
                    if body.get("action") == "EXECUTE":
                        annotated = annotate_ml_prediction({
                            "status": "success",
                            "metrics": body.get("metrics") or {},
                            "pt_mult": body.get("pt_mult"),
                            "sl_mult": body.get("sl_mult"),
                            "deflated_sharpe_ratio": body.get("deflated_sharpe_ratio"),
                            "prob_backtest_overfitting": body.get("prob_backtest_overfitting"),
                        })
                        if annotated.get("status") == "error":
                            body["action"] = "VETO"
                            body["reason"] = annotated.get("error") or annotated.get("promotion_reason")
                            body["promotion_ok"] = False
                    return body
                if live_exchange_orders_allowed():
                    return {
                        "action": "VETO",
                        "reason": f"ML server returned HTTP {res.status_code} (fail-closed live)",
                    }
                return {"action": "EXECUTE", "reason": f"ML server returned HTTP {res.status_code} (fail-open paper)"}
        except Exception as e:
            if live_exchange_orders_allowed():
                logger.error(f"Jesse meta-predict failed in LIVE mode ({e}) — fail closed")
                return {"action": "VETO", "reason": f"ML query error (fail-closed live): {e}"}
            logger.debug(f"Jesse meta-predict notice (fail-open paper): {e}")
            return {"action": "EXECUTE", "reason": f"ML query error (fail-open paper): {e}"}


    async def get_ml_models(self) -> Dict[str, Any]:
        """Fetch status and list of trained ML model artifacts."""
        empty: Dict[str, Any] = {
            "available_models": [],
            "cached_models": [],
            "rejected_models": {},
        }
        base_ml = await self._resolve_ml_url()
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{base_ml}/health")
                if res.status_code == 200:
                    payload = res.json()
                    if not isinstance(payload, dict):
                        return {**empty, "status": "error", "error": "ML health returned non-object JSON"}
                    payload.setdefault("available_models", [])
                    payload.setdefault("cached_models", [])
                    payload.setdefault("rejected_models", {})
                    return payload
                return {
                    **empty,
                    "status": "error",
                    "error": f"ML server returned HTTP {res.status_code}",
                }
        except Exception as e:
            return {**empty, "status": "error", "error": str(e)}


jesse_bridge = JesseBridgeService()
