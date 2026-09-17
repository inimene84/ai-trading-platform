"""
Jesse-to-QuantumTrade Live Strategy Bridge
Enables communication between the Jesse Quant Research engine and the live
QuantumTrade trading platform:
- Queries Jesse candle storage and health
- Dispatches automated on-demand strategy backtests
- Synchronizes optimized strategy parameters (SL/TP ATR, Trailing thresholds) into live RiskConfig
"""

import os
import json
import logging
import math
from typing import Any, Dict, Optional

import httpx

from backend.services.jesse_validation import evaluate_validation_gates

logger = logging.getLogger(__name__)

JESSE_API_URL = os.getenv("JESSE_API_URL", "http://jesse-app:9000")
JESSE_LOCAL_URL = os.getenv("JESSE_LOCAL_URL", "http://127.0.0.1:9000")
JESSE_ML_URL = os.getenv("JESSE_ML_URL", "http://jesse-app:9003")
JESSE_ML_LOCAL_URL = os.getenv("JESSE_ML_LOCAL_URL", "http://127.0.0.1:9003")
JESSE_PASSWORD = os.getenv("JESSE_PASSWORD", "QuantumTrading2026!")
# Strategy-aligned triple-barrier defaults (match live RiskConfig SL=1.75 / TP=5.5 ATR)
DEFAULT_SL_ATR_MULT = float(os.getenv("JESSE_DEFAULT_SL_ATR_MULT", "1.75"))
DEFAULT_TP_ATR_MULT = float(os.getenv("JESSE_DEFAULT_TP_ATR_MULT", "5.5"))
DEFAULT_TRAIL_ACTIVATION_ATR = float(os.getenv("JESSE_DEFAULT_TRAIL_ACTIVATION_ATR", "1.8"))
DEFAULT_TRAIL_ATR_MULT = float(os.getenv("JESSE_DEFAULT_TRAIL_ATR_MULT", "1.6"))
DEFAULT_ML_PREDICT_TIMEOUT = 20.0
MAX_ML_PREDICT_TIMEOUT = 30.0


def configured_ml_model_type() -> str:
    """Primary Jesse ML artifact type (`JESSE_ML_MODEL_TYPE`, default lightgbm)."""
    raw = (os.getenv("JESSE_ML_MODEL_TYPE") or "lightgbm").strip().lower()
    return raw or "lightgbm"


def configured_ml_fallback_model_type() -> Optional[str]:
    """Secondary type when the primary sidecar/artifact is missing.

    `JESSE_ML_FALLBACK_MODEL_TYPE` defaults to lstm. Set to none/false/off/0
    to disable fallback (live gate then fail-closes on a missing primary).
    """
    raw = (os.getenv("JESSE_ML_FALLBACK_MODEL_TYPE", "lstm") or "").strip().lower()
    if raw in {"", "none", "false", "off", "0"}:
        return None
    return raw


def ml_predict_timeout_seconds() -> float:
    """HTTP timeout for Jesse `/predict` (`JESSE_ML_PREDICT_TIMEOUT`, 1–30s)."""
    raw = os.getenv("JESSE_ML_PREDICT_TIMEOUT", str(DEFAULT_ML_PREDICT_TIMEOUT))
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_ML_PREDICT_TIMEOUT
    if not math.isfinite(timeout):
        return DEFAULT_ML_PREDICT_TIMEOUT
    return min(MAX_ML_PREDICT_TIMEOUT, max(1.0, timeout))


def metadata_is_usable(metadata: Optional[Dict[str, Any]]) -> bool:
    """True when /model-metadata did not report an error or missing artifact."""
    if not metadata:
        return False
    if metadata.get("status") == "error":
        return False
    err = str(metadata.get("error") or "").lower()
    if "not found" in err or "no model artifact" in err:
        return False
    return True


def is_missing_artifact_error(payload: Optional[Dict[str, Any]]) -> bool:
    """True when Jesse ML reported a missing sidecar or model file."""
    if not payload:
        return False
    err = str(payload.get("error") or "").lower()
    return "no model artifact" in err or "not found" in err


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
        import uuid
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
                    import asyncio
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
        sl_atr_mult: float = DEFAULT_SL_ATR_MULT,
        tp_atr_mult: float = DEFAULT_TP_ATR_MULT,
        trail_activation_atr: float = DEFAULT_TRAIL_ACTIVATION_ATR,
        trail_atr_mult: float = DEFAULT_TRAIL_ATR_MULT,
    ) -> Dict[str, Any]:
        """Apply validated quant strategy parameters directly to active RiskConfig."""
        if os.getenv("JESSE_SYNC_TO_LIVE", "false").lower() != "true":
            logger.info("JESSE_SYNC_TO_LIVE is disabled — strategy sync is a no-op")
            return {
                "status": "blocked",
                "message": "JESSE_SYNC_TO_LIVE is disabled; sync is a no-op",
                "synced": False,
            }

        import re
        from backend.services.risk_config import refresh_risk_config

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
            async with httpx.AsyncClient(timeout=ml_predict_timeout_seconds()) as client:
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
                    return res.json()
                return {"status": "error", "error": f"ML server returned HTTP {res.status_code}: {res.text}"}
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
                    return res.json()
                return {"action": "EXECUTE", "reason": f"ML server returned HTTP {res.status_code} (fail-open)"}
        except Exception as e:
            logger.debug(f"Jesse meta-predict notice (fail-open): {e}")
            return {"action": "EXECUTE", "reason": f"ML query error (fail-open): {e}"}


    async def get_ml_models(self) -> Dict[str, Any]:
        """Fetch status and list of trained ML model artifacts."""
        base_ml = await self._resolve_ml_url()
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{base_ml}/health")
                if res.status_code == 200:
                    return res.json()
                return {"status": "error", "error": f"ML server returned HTTP {res.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @staticmethod
    def normalize_jesse_symbol(symbol: str) -> str:
        """Convert broker symbols (BTCUSDT) to Jesse format (BTC-USDT)."""
        clean = symbol.replace("/", "").replace("-", "").upper()
        if clean.endswith("USDT") and len(clean) > 4:
            return f"{clean[:-4]}-USDT"
        if clean.endswith("USDC") and len(clean) > 4:
            return f"{clean[:-4]}-USDC"
        return symbol

    async def get_model_metadata(
        self,
        symbol: str = "BTC-USDT",
        timeframe: str = "1h",
        model_type: str = "lightgbm",
    ) -> Dict[str, Any]:
        """Fetch trained model MLOps metadata (DSR, PBO, holdout Sharpe) from Jesse ML engine."""
        base_ml = await self._resolve_ml_url()
        jesse_symbol = self.normalize_jesse_symbol(symbol)
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.get(
                    f"{base_ml}/model-metadata",
                    params={
                        "symbol": jesse_symbol,
                        "timeframe": timeframe,
                        "model_type": model_type,
                    },
                )
                if res.status_code == 200:
                    payload = res.json()
                    payload["status"] = payload.get("status", "success")
                    return payload
                return {
                    "status": "error",
                    "error": f"ML metadata returned HTTP {res.status_code}: {res.text}",
                }
        except Exception as e:
            logger.debug(f"Jesse model metadata unavailable: {e}")
            return {"status": "error", "error": str(e)}

    async def resolve_gate_model(
        self,
        symbol: str = "BTC-USDT",
        timeframe: str = "1h",
    ) -> Dict[str, Any]:
        """Pick LightGBM when a production artifact exists, else a promoted LSTM.

        Missing LightGBM is not a reason to disable the live ML gate. ETH/SOL 1h
        LightGBM was quarantined for wrong barrier geometry; promoted LSTMs at
        live 5.5/1.75 ATR remain valid. Fallback never bypasses DSR/PBO checks.
        """
        primary = configured_ml_model_type()
        fallback = configured_ml_fallback_model_type()
        primary_meta = await self.get_model_metadata(symbol, timeframe, primary)
        if metadata_is_usable(primary_meta):
            primary_meta["resolved_model_type"] = primary
            primary_meta["resolved_via"] = "primary"
            return primary_meta

        primary_error = primary_meta.get("error") or (
            f"No model artifact found for {self.normalize_jesse_symbol(symbol)} "
            f"({timeframe}, {primary})"
        )
        if fallback and fallback != primary:
            fallback_meta = await self.get_model_metadata(symbol, timeframe, fallback)
            if metadata_is_usable(fallback_meta):
                logger.info(
                    "Jesse ML gate using %s fallback for %s %s (primary %s unavailable: %s)",
                    fallback,
                    self.normalize_jesse_symbol(symbol),
                    timeframe,
                    primary,
                    primary_error,
                )
                fallback_meta["resolved_model_type"] = fallback
                fallback_meta["resolved_via"] = "fallback"
                fallback_meta["primary_model_type"] = primary
                fallback_meta["primary_error"] = primary_error
                return fallback_meta
            fallback_error = fallback_meta.get("error")
        else:
            fallback_error = None

        return {
            "status": "error",
            "resolved_model_type": primary,
            "resolved_via": "none",
            "primary_model_type": primary,
            "error": primary_error,
            "fallback_model_type": fallback,
            "fallback_error": fallback_error,
        }

    async def get_validation_status(
        self,
        symbol: str = "BTC-USDT",
        timeframe: str = "1h",
        model_type: str = "lightgbm",
    ) -> Dict[str, Any]:
        """Return institutional validation gate status for a Jesse ML model."""
        metadata = await self.get_model_metadata(symbol, timeframe, model_type)
        if metadata.get("status") == "error":
            return {
                "status": "error",
                "symbol": self.normalize_jesse_symbol(symbol),
                "timeframe": timeframe,
                "model_type": model_type,
                "error": metadata.get("error"),
            }

        gates = evaluate_validation_gates(metadata)
        return {
            "status": "success",
            "symbol": metadata.get("symbol", self.normalize_jesse_symbol(symbol)),
            "timeframe": timeframe,
            "model_type": model_type,
            "feature_hash": metadata.get("feature_hash"),
            "labeling_mode": metadata.get("labeling_mode"),
            **gates,
        }


jesse_bridge = JesseBridgeService()
