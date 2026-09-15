import os
import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, JSONResponse
import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from backend.database.connection import SessionLocal
from backend.database.models import TradingSignal, Trade, PortfolioSnapshot
from backend.services.trading_loop import trading_loop
from backend.services.ai_analysis import ai_analysis_service
from backend.services.risk_config import refresh_risk_config
from backend.services.ctrader_service import ctrader_broker
from backend.services.ctrader_trade_sync import (
    overlay_live_mark,
    position_pnl_pct,
    reconcile_ctrader_positions,
    upsert_ctrader_live_trades,
)
from backend.services.binance_trade_sync import upsert_binance_live_trades
from backend.services.ledger import binance_position_key
from backend.services.trading_loop_helpers import (
    is_ctrader_symbol as _is_ctrader_symbol,
    is_ctrader_trade as _is_ctrader_trade,
)
from backend.services.ledger import is_binance_paper_fill
from backend.services.trading_mode import (
    TradingMode,
    get_trading_mode,
    live_binance_orders_allowed,
)
from backend.services.unified_feed import unified_feed
from backend.services.unified_trading import (
    UnifiedTrading, UnifiedOrder, OrderSide, OrderType,
)

import logging
logger = logging.getLogger(__name__)

class StartLoopRequest(BaseModel):
    interval_minutes: int = 15
    symbols: Optional[List[str]] = None
    strategy: str = "combined"

class LoopStatusResponse(BaseModel):
    state: str
    running: bool
    interval_minutes: int
    symbols: List[str]
    strategy: str
    last_cycle: Optional[str]
    next_cycle: Optional[str]
    cycle_count: int
    error: Optional[str]
    cash: Optional[float] = None
    equity: Optional[float] = None
    margin_used: Optional[float] = None
    trading_allowed: Optional[bool] = None
    trading_status: Optional[str] = None

class TradingConfigResponse(BaseModel):
    mode: str
    interval_minutes: int
    symbols: List[str]
    risk_limits: Dict[str, Any]

class ConfigUpdateRequest(BaseModel):
    use_risk_reviewer_llm: Optional[bool] = None
    enable_personas: Optional[bool] = None


class ModifyPositionRequest(BaseModel):
    stop_loss: Optional[float] = Field(default=None, gt=0)
    take_profit: Optional[float] = Field(default=None, gt=0)


class LiveOrderRequest(BaseModel):
    symbol: str = Field(min_length=5, max_length=20, pattern=r"^[A-Za-z0-9=_/-]+$")
    side: Literal["buy", "sell"]
    quantity: float = Field(gt=0, le=1_000_000_000)
    order_type: Literal["market", "limit"] = "market"
    price: float = Field(default=0, ge=0)
    stop_loss: Optional[float] = Field(default=None, gt=0)
    take_profit: Optional[float] = Field(default=None, gt=0)


class AgentTradeRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8_000)
    model: Optional[str] = Field(default=None, max_length=120)
    provider: Literal["xai", "openai", "groq", "ollama"] = "xai"

load_dotenv()

router = APIRouter(prefix="/trading", tags=["trading"])


def _to_yfinance_symbol(symbol: str) -> str:
    """Convert Binance-native BTCUSDT/BTCUSDC → yfinance BTC-USD format."""
    s = symbol.upper().strip()
    if s.endswith('USDT') or s.endswith('USDC'):
        return s[:-4] + '-USD'
    if s.endswith('BUSD'):
        return s[:-4] + '-USD'
    return s  # already yfinance format or unknown


def _coalesce_mark_price(mark: Optional[float], entry: Optional[float]) -> float:
    """Never surface a zero mark — fall back to entry when quotes are missing."""
    for candidate in (mark, entry):
        if candidate is not None:
            val = float(candidate)
            if val > 0:
                return val
    return 0.0


@router.get("/strategies")
async def list_strategies():
    """List available trading strategies with their current parameters."""
    strategies = [
        {
            "name": "breakout",
            "description": "Donchian channel breakout with volume confirmation",
            "params": {"channel_period": 50, "atr_multiplier": 2.5, "volume_factor": 1.8},
        },
        {
            "name": "mean_reversion",
            "description": "Bollinger Bands + RSI mean reversion with ADX filter",
            "params": {"rsi_oversold": 25, "rsi_overbought": 75, "bb_std": 2.5},
        },
        {
            "name": "trend_following",
            "description": "EMA crossover with ADX trend confirmation",
            "params": {"fast_ema": 50, "slow_ema": 100, "adx_threshold": 25},
        },
        {
            "name": "scalping",
            "description": "EMA ribbon + VWAP with momentum confirmation",
            "params": {"min_body_pct": 0.35, "atr_stop_mult": 1.0, "atr_tp_mult": 2.0},
        },
        {
            "name": "combined",
            "description": "Weighted voting across all strategies",
            "params": {
                "vote_threshold": 0.50,
                "weights": {"trend": 0.50, "mean_rev": 0.20, "breakout": 0.30},
            },
        },
    ]
    return {"strategies": strategies}


@router.post("/run-backtest")
async def run_backtest(request: dict):
    """Run a bar-by-bar backtest using CombinedStrategy on Binance or yfinance data."""
    symbol = request.get("symbol", "ETHUSDT")
    strategy = request.get("strategy", "combined")
    days = int(request.get("days", 90))
    balance = float(request.get("balance", 10_000))

    try:
        from backend.backtesting_ctrader.engine import (
            BacktestEngine,
            download_bars_for_symbol,
            backtest_result_to_dict,
        )
        bars = download_bars_for_symbol(symbol, days=days)
        if len(bars) < 50:
            return {"error": f"Insufficient bars ({len(bars)}) for {symbol}", "symbol": symbol}
        engine = BacktestEngine(strategy_name=strategy, initial_balance=balance)
        result = engine.run(symbol=symbol, bars=bars)
        return backtest_result_to_dict(result)
    except Exception as e:
        logger.error(f"Backtest failed for {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "strategy": strategy,
            "pnl": 0,
            "trades": 0,
            "win_rate": 0,
            "sharpe": 0,
        }


async def _get_current_balance() -> dict:
    """Resolve current balance depending on trading mode (paper vs live).

    Live never treats a failed Binance wallet read as $0 equity. Prefer any
    healthy broker book (cTrader and/or Binance) and fall back to last live
    snapshot rather than reporting zeros.
    """
    from backend.services.trading_mode import TradingMode, get_trading_mode
    if get_trading_mode() == TradingMode.PAPER:
        try:
            from backend.services.unified_trading import trading_router
            pf = trading_router.get_paper_portfolio()
            if pf:
                cash = float(pf.get("cash", 100000.0))
                equity = float(pf.get("equity", cash))
                return {
                    "balance": cash,
                    "available": cash,
                    "equity": equity,
                    "margin_used": float(pf.get("margin_used", 0.0)),
                    "broker": "paper_trading",
                }
        except Exception as e:
            logger.warning(f"Could not fetch paper portfolio: {e}")
        return {
            "balance": 100000.0,
            "available": 100000.0,
            "equity": 100000.0,
            "margin_used": 0.0,
            "broker": "paper_trading",
        }

    books: List[Dict[str, Any]] = []
    try:
        if ctrader_broker.has_credentials():
            ct = await asyncio.to_thread(ctrader_broker.get_balance)
            if ct and (float(ct.get("equity") or 0) > 0 or float(ct.get("balance") or 0) > 0):
                books.append(ct)
    except Exception as exc:
        logger.warning("cTrader get_balance failed: %s", exc)

    if live_binance_orders_allowed():
        try:
            from backend.services.binance_futures_service import binance_futures_broker
            bn = await asyncio.to_thread(
                lambda: binance_futures_broker.get_balance(raise_on_error=True)
            )
            if bn and not bn.get("error"):
                books.append(bn)
        except Exception as exc:
            logger.warning("Binance get_balance failed (not treating as $0): %s", exc)

    if books:
        equity = sum(float(b.get("equity") or b.get("balance") or 0) for b in books)
        balance = sum(float(b.get("balance") or 0) for b in books)
        available = sum(float(b.get("available") or 0) for b in books)
        margin_used = sum(float(b.get("margin_used") or 0) for b in books)
        brokers = ",".join(str(b.get("broker") or "?") for b in books)
        return {
            "balance": balance,
            "available": available,
            "equity": equity,
            "margin_used": margin_used,
            "broker": brokers,
            "books": books,
        }
    return {
        "balance": 0.0,
        "available": 0.0,
        "equity": 0.0,
        "margin_used": 0.0,
        "broker": "unverified",
        "error": "no_live_broker_balance",
    }


async def _live_binance_book() -> tuple[list, bool]:
    """(positions, snapshot_ok). snapshot_ok False => do not hide DB crypto rows."""
    if not live_binance_orders_allowed():
        return [], False
    try:
        from backend.services.binance_futures_service import binance_futures_broker
        pos = await asyncio.to_thread(
            lambda: binance_futures_broker.get_positions(raise_on_error=True)
        )
        return list(pos or []), True
    except Exception as exc:
        logger.warning("Binance live-book fetch failed: %s", exc)
        return [], False


@router.get("/portfolio")
async def get_portfolio():
    """Get current portfolio state with live Binance balance (or paper portfolio in paper mode)."""
    db = SessionLocal()
    try:
        if ctrader_broker.has_credentials() and not (
            ctrader_broker.is_connected()
            if callable(getattr(ctrader_broker, "is_connected", None))
            else bool(getattr(ctrader_broker, "is_connected", False))
        ):
            try:
                await asyncio.to_thread(ctrader_broker.ensure_connected)
            except Exception as exc:
                logger.warning("cTrader auto-reconnect in get_portfolio failed: %s", exc)

        live_ctrader: List[Dict[str, Any]] = []
        is_ctrader_connected = (
            ctrader_broker.is_connected()
            if callable(getattr(ctrader_broker, "is_connected", None))
            else bool(getattr(ctrader_broker, "is_connected", False))
        )
        try:
            live_ctrader = list(ctrader_broker.get_positions() or [])
            if is_ctrader_connected or live_ctrader:
                reconcile_ctrader_positions(db, live_ctrader, broker=ctrader_broker)
            elif live_ctrader:
                upsert_ctrader_live_trades(db, live_ctrader)
        except Exception as exc:
            logger.warning("cTrader live-book sync for portfolio failed: %s", exc)

        live_binance, binance_snapshot_ok = await _live_binance_book()
        if binance_snapshot_ok:
            try:
                upsert_binance_live_trades(db, live_binance)
            except Exception as exc:
                logger.warning("Binance live-book upsert for portfolio failed: %s", exc)
        live_binance_syms = {
            str(p.get("symbol") or "").upper()
            for p in live_binance
            if abs(float(p.get("quantity") or 0)) > 0
        }

        open_trades = db.query(Trade).filter(Trade.status.in_(["open", "filled"])).all()
        live_by_pid = {
            str(p.get("position_id")): p for p in live_ctrader if p.get("position_id")
        }
        live_by_symbol = {
            str(p.get("symbol") or "").upper(): p for p in live_ctrader if p.get("symbol")
        }
        non_ctrader_symbols = {
            t.symbol for t in open_trades if not _is_ctrader_trade(t)
        }
        mark_prices = await _fetch_mark_prices_for_symbols(non_ctrader_symbols)

        total_notional = 0.0
        total_unrealized_pnl = 0.0
        positions = []
        paper_positions = []
        live_mode = get_trading_mode() == TradingMode.LIVE
        for t in open_trades:
            is_ctrader = _is_ctrader_trade(t)
            direction = str(t.direction or "BUY").upper()
            sym = str(t.symbol or "").upper()

            if live_mode and not is_ctrader and is_binance_paper_fill(t):
                paper_positions.append({
                    "id": t.id,
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "quantity": t.quantity,
                    "entry_price": t.entry_price,
                    "ledger": "paper",
                })
                continue
            if (
                live_mode
                and not is_ctrader
                and binance_snapshot_ok
                and sym not in live_binance_syms
            ):
                continue

            if is_ctrader:
                pid = str(getattr(t, "broker_position_id", "") or "").strip()
                if live_ctrader:
                    if pid and pid not in live_by_pid:
                        continue
                    if not pid and sym not in live_by_symbol:
                        continue
                mark = ctrader_broker.get_mark_price(sym, direction)
                cur_price = _coalesce_mark_price(mark, t.entry_price) or 0.0
                live = live_by_pid.get(pid) or live_by_symbol.get(sym)
                lots = float((live or {}).get("quantity") or t.quantity or 0)
                volume_cents = (live or {}).get("volume_cents")
                if volume_cents:
                    units = float(volume_cents) / 100.0
                else:
                    units = lots * ctrader_broker.units_per_lot(
                        sym, lot_size_cents=ctrader_broker.lot_size_cents(sym)
                    )
                notional = units * (t.entry_price or 0.0)
                total_notional += notional
                if live and live.get("unrealized_pnl") is not None and (
                    live.get("current_price") or volume_cents
                ):
                    u_pnl = float(live["unrealized_pnl"])
                elif mark and t.entry_price and units:
                    direction_mult = 1 if direction == "BUY" else -1
                    quote_pnl = (mark - float(t.entry_price)) * units * direction_mult
                    rate = ctrader_broker.quote_to_usd_rate(sym, mark)
                    u_pnl = round(quote_pnl * rate, 2) if rate else round(quote_pnl, 2)
                else:
                    u_pnl = 0.0
                total_unrealized_pnl += u_pnl
            else:
                cur_price = mark_prices.get(t.symbol) or t.entry_price or 0.0
                notional = (t.quantity or 0.0) * (t.entry_price or 0.0)
                total_notional += notional

                if direction == "BUY":
                    u_pnl = (cur_price - (t.entry_price or cur_price)) * (t.quantity or 0.0)
                else:
                    u_pnl = ((t.entry_price or cur_price) - cur_price) * (t.quantity or 0.0)
                total_unrealized_pnl += u_pnl

            positions.append({
                "id": t.id,
                "symbol": t.symbol,
                "direction": t.direction,
                "quantity": lots if is_ctrader else t.quantity,
                "entry_price": t.entry_price,
                "current_price": cur_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "unrealized_pnl": round(u_pnl, 2),
                "strategy": t.strategy,
                "broker": "ctrader" if is_ctrader else (
                    getattr(t, "broker", None) or getattr(t, "exchange", None) or "binance_futures"
                ),
                "opened_at": t.timestamp.isoformat() if t.timestamp else None,
            })

        # Prefer live broker/paper balance over stale DB snapshot
        balance = 100000.0
        available = 100000.0
        equity = 100000.0
        positions_value = total_notional
        try:
            bal = await _get_current_balance()
            balance = float(bal.get("balance", 100000.0))
            is_paper = bal.get("broker") == "paper_trading" or os.getenv("TRADING_MODE", "paper") == "paper"

            if is_paper:
                positions_value = total_notional
                available = max(0.0, balance - total_notional)
                equity = balance + total_unrealized_pnl
            else:
                margin_used = float(bal.get("margin_used", 0.0))
                positions_value = margin_used if margin_used > 0 else total_notional
                available = float(bal.get("available", balance - positions_value))
                equity = float(bal.get("equity", balance + total_unrealized_pnl))
                if equity <= 0 and (bal.get("error") or bal.get("broker") == "unverified"):
                    snap = (
                        db.query(PortfolioSnapshot)
                        .filter(PortfolioSnapshot.mode == "live")
                        .filter(PortfolioSnapshot.total_value > 0)
                        .order_by(PortfolioSnapshot.id.desc())
                        .first()
                    )
                    if snap:
                        balance = snap.cash or snap.total_value or 0.0
                        equity = snap.total_value or balance
                        available = max(0.0, balance - (snap.positions_value or 0.0))
                        positions_value = snap.positions_value or total_notional
        except Exception:
            snap = (
                db.query(PortfolioSnapshot)
                .order_by(PortfolioSnapshot.id.desc())
                .first()
            )
            if snap:
                balance = snap.cash or 0.0
                available = max(0.0, balance - total_notional)
                equity = balance + total_unrealized_pnl
                positions_value = total_notional

        # Compute realized PnL from all closed trades
        closed_pnl = db.query(Trade).filter(Trade.status == "closed").with_entities(
            Trade.pnl
        ).all()
        total_pnl = round(sum((r.pnl or 0.0) for r in closed_pnl), 4)
        pnl_pct = round((total_pnl / equity * 100) if equity > 0 else 0.0, 2)

        return {
            "balance": round(balance, 2),
            "available": round(available, 2),
            "equity": round(equity, 2),
            "unrealized_pnl": round(total_unrealized_pnl, 2),
            "positions": positions,
            "paper_positions": paper_positions,
            "total_pnl": total_pnl,
            "total_pnl_pct": pnl_pct,
            "positions_value": round(positions_value, 2),
            "open_positions_count": len(positions),
            "last_updated": datetime.now().isoformat(),
        }
    finally:
        db.close()


@router.get("/performance")
async def get_performance():
    """Rolling performance: win rate, realized PnL, drawdown, trade counts."""
    db = SessionLocal()
    try:
        closed = db.query(Trade).filter(Trade.status == "closed").all()
        wins = sum(1 for t in closed if (t.pnl or 0) > 0)
        losses = sum(1 for t in closed if (t.pnl or 0) < 0)
        decided = wins + losses
        realized = round(sum((t.pnl or 0.0) for t in closed), 4)
        equity = 0.0
        try:
            bal = await _get_current_balance()
            equity = bal.get("equity", bal.get("balance", 0.0))
        except Exception:
            pass
        peak = db.query(PortfolioSnapshot.total_value).order_by(
            PortfolioSnapshot.total_value.desc()
        ).first()
        peak_val = (peak[0] if peak else 0.0) or equity
        drawdown_pct = round(((equity - peak_val) / peak_val * 100.0), 3) if peak_val > 0 else 0.0
        return {
            "win_rate": round((wins / decided * 100.0), 2) if decided else 0.0,
            "wins": wins,
            "losses": losses,
            "total_trades": len(closed),
            "realized_pnl": realized,
            "equity": round(equity, 4),
            "drawdown_pct": drawdown_pct,
        }
    finally:
        db.close()


@router.get("/signals")
async def get_recent_signals():
    """Get recent trading signals."""
    db = SessionLocal()
    try:
        signals = (
            db.query(TradingSignal)
            .order_by(TradingSignal.id.desc())
            .limit(20)
            .all()
        )
        return {
            "signals": [
                {
                    "id": s.id,
                    "timestamp": s.timestamp.isoformat() if s.timestamp else None,
                    "symbol": s.symbol,
                    "strategy": s.strategy,
                    "direction": s.direction,
                    "confidence": s.confidence,
                    "entry_price": s.entry_price,
                    "stop_loss": s.stop_loss,
                    "take_profit": s.take_profit,
                    "status": s.status,
                    "reasoning": s.reasoning,
                    "ai_analysis": json.loads(s.ai_analysis) if s.ai_analysis else None,
                }
                for s in signals
            ]
        }
    finally:
        db.close()


@router.get("/status")
async def get_status():
    """Get system status with real configuration."""
    # Check which LLM providers are configured
    llm_providers = []

    # Cloud providers
    if os.getenv('OMNIROUTE_API_KEY') and os.getenv('OMNIROUTE_API_KEY') != 'omni_live_key_placeholder':
        llm_providers.append({
            'name': 'OmniRoute',
            'model': os.getenv('PERSONA_LLM_MODEL', 'auto/fast'),
            'status': 'configured',
            'type': 'cloud',
            'role': 'Primary (Auto-Select Routing)',
        })
    if os.getenv('XAI_API_KEY') and os.getenv('XAI_API_KEY') != 'your_xai_api_key_here':
        llm_providers.append({'name': 'xAI (Grok)', 'model': os.getenv('XAI_MODEL', 'grok-beta'), 'status': 'configured', 'type': 'cloud'})
    if os.getenv('KIE_API_KEY') and os.getenv('KIE_API_KEY') != 'your_kie_api_key_here':
        llm_providers.append({
            'name': 'Kie.ai',
            'model': os.getenv('KIE_MODEL', 'gpt-5-6-luna'),
            'status': 'configured',
            'type': 'cloud',
            'role': 'fallback / direct via Kie.ai',
        })
    if os.getenv('ANTHROPIC_API_KEY') and os.getenv('ANTHROPIC_API_KEY') != 'your_anthropic_api_key_here':
        llm_providers.append({'name': 'Anthropic', 'model': 'claude', 'status': 'configured', 'type': 'cloud'})
    if os.getenv('OPENAI_API_KEY') and os.getenv('OPENAI_API_KEY') != 'your_openai_api_key_here':
        llm_providers.append({'name': 'OpenAI', 'model': 'gpt-4o', 'status': 'configured', 'type': 'cloud'})
    if os.getenv('GROQ_API_KEY') and os.getenv('GROQ_API_KEY') != 'your_groq_api_key_here':
        llm_providers.append({'name': 'Groq', 'model': 'mixtral', 'status': 'configured', 'type': 'cloud'})
    if os.getenv('GOOGLE_API_KEY') and os.getenv('GOOGLE_API_KEY') != 'your_google_api_key_here':
        llm_providers.append({'name': 'Google', 'model': 'gemini', 'status': 'configured', 'type': 'cloud'})

    # Ollama (local models) — skip when pointed at the LiteLLM proxy, which is
    # an OpenAI-compatible endpoint and has no Ollama /api/tags route.
    ollama_url = os.getenv('OLLAMA_BASE_URL', '')
    if ollama_url and 'litellm' not in ollama_url.lower():
        try:
            resp = httpx.get(f"{ollama_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get('models', [])
                role_map = {
                    os.getenv('OLLAMA_PRIMARY_MODEL', 'phi3.5'): 'Primary (Reasoning)',
                    os.getenv('OLLAMA_SECONDARY_MODEL', 'phi4'): 'Secondary (Fallback)',
                    os.getenv('OLLAMA_LIGHTWEIGHT_MODEL', 'phi3.5'): 'Lightweight (Fast)',
                }
                for m in models:
                    mname = m.get('name', '')
                    role = 'Local'
                    for key, val in role_map.items():
                        if mname.startswith(key):
                            role = val
                            break
                    size_gb = m.get('size', 0) / (1024**3)
                    param_size = m.get('details', {}).get('parameter_size', '')
                    llm_providers.append({
                        'name': f"Ollama: {mname.split(':')[0]}",
                        'model': f"{mname} ({param_size}, {size_gb:.1f}GB)",
                        'status': 'configured',
                        'type': 'local',
                        'role': role,
                    })
        except Exception:
            llm_providers.append({'name': 'Ollama', 'model': 'connection failed', 'status': 'error', 'type': 'local'})

    # Check brokers
    brokers = []
    if os.getenv('BINANCE_API_KEY'):
        brokers.append({'name': 'Binance', 'env': 'testnet' if os.getenv('BINANCE_TESTNET', 'true') == 'true' else 'live', 'status': 'configured'})
    else:
        brokers.append({'name': 'Binance', 'env': 'testnet', 'status': 'not_configured'})
    if os.getenv('CTRADER_ACCESS_TOKEN'):
        from backend.services.ctrader_service import ctrader_broker
        os.getenv('CTRADER_ENV', 'demo') == 'live'
        status_str = 'configured'
        if ctrader_broker.is_connected:
            status_str = 'online'
        elif ctrader_broker.dry_run:
            status_str = 'configured' # default to configured if not explicitly connecting
            
        brokers.append({'name': 'cTrader', 'env': os.getenv('CTRADER_ENV', 'demo'), 'status': status_str})
    else:
        brokers.append({'name': 'cTrader', 'env': 'demo', 'status': 'not_configured'})
    if os.getenv('ALPACA_API_KEY'):
        brokers.append({'name': 'Alpaca', 'env': 'paper' if os.getenv('ALPACA_PAPER', 'true') == 'true' else 'live', 'status': 'configured'})
    else:
        brokers.append({'name': 'Alpaca', 'env': 'paper', 'status': 'not_configured'})
    if os.getenv('OANDA_API_KEY'):
        brokers.append({'name': 'OANDA', 'env': 'practice' if os.getenv('OANDA_PRACTICE', 'true') == 'true' else 'live', 'status': 'configured'})
    else:
        brokers.append({'name': 'OANDA', 'env': 'practice', 'status': 'not_configured'})

    # Check data providers
    data_providers = []
    data_providers.append({'name': 'Binance (Public)', 'status': 'public', 'note': 'No key needed'})
    data_providers.append({'name': 'yfinance (Public)', 'status': 'public', 'note': 'No key needed'})
    for name, env_key in [('CoinGecko', 'COINGECKO_API_KEY'), ('CoinMarketCap', 'COINMARKETCAP_API_KEY'),
                           ('Alpha Vantage', 'ALPHAVANTAGE_API_KEY'), ('Twelve Data', 'TWELVEDATA_API_KEY'),
                           ('Polygon.io', 'POLYGON_API_KEY'), ('FRED', 'FRED_API_KEY'),
                           ('NewsAPI', 'NEWSAPI_KEY'), ('LunarCrush', 'LUNARCRUSH_API_KEY')]:
        data_providers.append({'name': name, 'status': 'configured' if os.getenv(env_key) else 'not_configured'})

    # Risk settings — read from the SAME RiskConfig the trading loop uses so the
    # UI can never disagree with the live engine.
    from backend.services.risk_config import get_risk_config
    _rc = get_risk_config()
    risk_config = {
        'risk_per_trade': _rc.risk_per_trade_pct,
        'max_positions': _rc.max_positions,
        'min_signal_strength': _rc.min_signal_strength,
        'ai_analysis_threshold': _rc.ai_analysis_threshold,
        'min_risk_reward': float(os.getenv('MIN_RISK_REWARD', '1.5')),
        'equity_sizing': _rc.equity_sizing_enabled,
        # Back-compat: the UI "risk-based sizing" indicator. Now reflects the
        # real equity/risk sizing rather than the previously-fake Kelly flag.
        'use_kelly': _rc.equity_sizing_enabled,
        'vix_threshold': float(os.getenv('VIX_THRESHOLD', '25.0')),
        'weights': {
            'technical': float(os.getenv('WEIGHT_TECHNICAL', '0.50')),
            'sentiment': float(os.getenv('WEIGHT_SENTIMENT', '0.25')),
            'macro': float(os.getenv('WEIGHT_MACRO', '0.25'))
        }
    }

    from backend.services.trading_mode import get_trading_mode
    _trading_mode = get_trading_mode()

    # Monitoring flags
    telegram = bool(os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'))
    influxdb = bool(os.getenv('INFLUXDB_TOKEN'))
    n8n = bool(os.getenv('N8N_WEBHOOK_URL'))

    return {
        'backend': 'online',
        'strategies_loaded': 5,
        'dry_run': _trading_mode.value != 'live',
        'mode': _trading_mode.value,
        'active_broker': os.getenv('ACTIVE_BROKER', 'binance_futures'),
        'llm_providers': llm_providers,
        'brokers': brokers,
        'data_providers': data_providers,
        'risk_config': risk_config,
        'telegram': telegram,
        'influxdb': influxdb,
        'n8n': n8n,
        'uptime': 'running',
        'last_cycle': trading_loop.status.get('last_cycle'),
        'trading_loop': trading_loop.status,
    }


@router.get("/config", response_model=TradingConfigResponse)
async def get_config():
    """Get current configuration limits and status."""
    from backend.services.risk_config import get_trading_mode, get_risk_config
    mode = get_trading_mode()
    config = get_risk_config()
    risk_limits = {
        "max_positions": config.max_positions,
        "max_directional_exposure_usdt": config.max_directional_exposure_usdt,
        "trade_usdt_amount": config.trade_usdt_amount,
        "kill_floor_usdt": config.kill_floor_usdt,
        "min_signal_strength": config.min_signal_strength,
        "sl_cooldown_minutes": config.sl_cooldown_minutes,
        "emergency_drawdown_pct": config.emergency_drawdown_pct,
        "use_risk_reviewer_llm": config.use_risk_reviewer_llm,
        "enable_personas": config.enable_personas,
    }
    return TradingConfigResponse(
        mode=mode,
        interval_minutes=trading_loop._interval_minutes,
        symbols=trading_loop._symbols,
        risk_limits=risk_limits,
    )


@router.post("/config/update")
async def update_config(payload: ConfigUpdateRequest):
    """Persist supported toggles to the mounted production .env."""
    updates = {}
    if payload.use_risk_reviewer_llm is not None:
        updates["USE_RISK_REVIEWER_LLM"] = str(payload.use_risk_reviewer_llm).lower()
    if payload.enable_personas is not None:
        updates["ENABLE_PERSONAS"] = str(payload.enable_personas).lower()
    if updates:
        env_path = Path(os.getenv("ENV_FILE_PATH", "/app/.env"))
        existing = env_path.read_text().splitlines() if env_path.exists() else []
        output = []
        remaining = dict(updates)
        for line in existing:
            key = line.split("=", 1)[0] if "=" in line else ""
            if key in remaining:
                output.append(f"{key}={remaining.pop(key)}")
            else:
                output.append(line)
        output.extend(f"{key}={value}" for key, value in remaining.items())
        env_path.write_text("\n".join(output) + "\n")
        os.environ.update(updates)

    config = refresh_risk_config()
    return {
        "status": "success",
        "config": {
            "use_risk_reviewer_llm": config.use_risk_reviewer_llm,
            "enable_personas": config.enable_personas
        }
    }


# ── Market Data Feeds ────────────────────────────────────────────────────────

@router.get("/markets/stocks")
async def get_stocks():
    """Fetch live data for top stocks using yfinance."""
    import yfinance as yf
    symbols = ["SPY", "AAPL", "MSFT", "NVDA", "TSLA", "META"]
    data = []
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                prev_close = hist["Close"].iloc[-2]
                current = hist["Close"].iloc[-1]
                change_pct = ((current - prev_close) / prev_close) * 100
                vol = float(hist["Volume"].iloc[-1])
            elif len(hist) == 1:
                current = hist["Close"].iloc[-1]
                change_pct = 0.0
                vol = float(hist["Volume"].iloc[-1])
            else:
                continue
                
            data.append({
                "symbol": sym,
                "price": float(current),
                "change24h": round(float(change_pct), 2),
                "volume24h": float(vol),
                "up": bool(change_pct >= 0)
            })
        except Exception:
            pass
    return {"data": data}


@router.get("/markets/forex")
async def get_forex_markets():
    """Live forex majors + metals quotes for the dashboard multi-asset panel.

    Response shape matches what MultiAssetPanel.fetchForexPrices parses:
    items keyed by display symbol ("EUR/USD") with price/change24h/up.
    """
    as_of = datetime.now(timezone.utc).isoformat()
    try:
        forex, metals = await asyncio.gather(
            unified_feed.get_quotes(asset_class="forex"),
            unified_feed.get_quotes(asset_class="metal"),
        )
    except Exception as e:
        logger.error("Forex markets error: %s", e)
        return {"data": [], "as_of": as_of, "error": str(e)}

    data = []
    for q in list(forex) + list(metals):
        sym = q["symbol"]
        display = f"{sym[:3]}/{sym[3:]}" if len(sym) == 6 and str(sym).isalpha() else sym
        price = q["price"] or 0.0
        change = q["change_pct"] or 0.0
        data.append({
            "symbol": display,
            "price": float(price),
            "change24h": round(float(change), 4),
            "up": bool(change >= 0),
            "source": q["source"],
            "stale": q["stale"],
        })
    return {"data": data, "as_of": as_of, "quotes": list(forex) + list(metals)}


# Binance public spot endpoints the frontend is allowed to proxy through us.
_BINANCE_PROXY_ALLOWED = {
    "ticker/24hr", "ticker/price", "ticker/bookTicker",
    "klines", "depth", "exchangeInfo", "avgPrice",
}

# Proxy protection state: short-TTL response cache, in-flight coalescing and a
# global backoff window honoured after any 418/429 from Binance. All dashboard
# traffic funnels through this proxy from one VPS IP — without these guards a
# few open dashboards can burn the IP-weight budget and get the *trading*
# engine IP-banned (-1003, 10-60 min cooldown).
_binance_proxy_cache: Dict[str, tuple] = {}
_binance_proxy_inflight: Dict[str, "asyncio.Future"] = {}
_binance_proxy_backoff_until: float = 0.0
_BINANCE_PROXY_TTL = {
    "ticker/24hr": 10.0, "ticker/price": 5.0, "ticker/bookTicker": 5.0,
    "klines": 20.0, "depth": 5.0, "avgPrice": 10.0, "exchangeInfo": 300.0,
}


# Proxy endpoint moved below to avoid shadowing static routes


# ── Trading Loop Control ──────────────────────────────────────────────────────

@router.post("/loop/start", response_model=LoopStatusResponse)
async def start_loop(req: StartLoopRequest = None):
    """Start the automated trading loop."""
    if req is None:
        req = StartLoopRequest()
    await trading_loop.start(
        interval_minutes=req.interval_minutes,
        symbols=req.symbols,
        strategy=req.strategy,
    )
    return LoopStatusResponse(**trading_loop.status)


@router.post("/loop/stop", response_model=LoopStatusResponse)
async def stop_loop():
    """Stop the automated trading loop."""
    await trading_loop.stop()
    return LoopStatusResponse(**trading_loop.status)


@router.get("/loop/status", response_model=LoopStatusResponse)
async def loop_status():
    """Get trading loop status."""
    return LoopStatusResponse(**trading_loop.status)


# ── Positions ─────────────────────────────────────────────────────────────────

async def _fetch_mark_prices_for_symbols(symbols: set[str]) -> dict[str, float]:
    """Public mark prices for dashboard P&L — never constructs a new Binance client."""
    prices: dict[str, float] = {}
    if not symbols:
        return prices
    from backend.services.binance_market_data import binance_market_data
    from backend.services.trading_mode import TradingMode, get_trading_mode

    crypto_syms = {s for s in symbols if str(s).upper().endswith(("USDT", "USDC"))}
    other = symbols - crypto_syms

    if get_trading_mode() == TradingMode.LIVE and os.getenv("ACTIVE_BROKER", "ctrader") == "binance_futures":
        try:
            from backend.services.binance_futures_service import binance_futures_broker
            live = await asyncio.to_thread(binance_futures_broker.get_positions)
            for p in live or []:
                sym = p.get("symbol")
                mark = float(p.get("mark_price") or 0)
                if sym and mark > 0:
                    prices[sym] = mark
        except Exception:
            pass

    missing = crypto_syms - set(prices.keys())
    for sym in missing:
        try:
            tick = await binance_market_data.get_ticker_24h(sym)
            last = float((tick or {}).get("lastPrice") or 0)
            if last > 0:
                prices[sym] = last
        except Exception:
            pass

    leftover = other - set(prices.keys())
    if leftover:
        import yfinance as yf

        def _yf_closes() -> dict[str, float]:
            out: dict[str, float] = {}
            for sym in leftover:
                try:
                    yf_sym = _to_yfinance_symbol(sym)
                    hist = yf.Ticker(yf_sym).history(period="5d")
                    if not hist.empty:
                        out[sym] = float(hist["Close"].iloc[-1])
                except Exception:
                    pass
            return out

        prices.update(await asyncio.to_thread(_yf_closes))
    return prices


@router.get("/positions")
async def get_positions():
    """Get all open positions with current P&L."""
    db = SessionLocal()
    try:
        if ctrader_broker.has_credentials() and not (
            ctrader_broker.is_connected()
            if callable(getattr(ctrader_broker, "is_connected", None))
            else bool(getattr(ctrader_broker, "is_connected", False))
        ):
            try:
                await asyncio.to_thread(ctrader_broker.ensure_connected)
            except Exception as exc:
                logger.warning("cTrader auto-reconnect in get_positions failed: %s", exc)

        live_ctrader: List[Dict[str, Any]] = []
        is_ctrader_connected = (
            ctrader_broker.is_connected()
            if callable(getattr(ctrader_broker, "is_connected", None))
            else bool(getattr(ctrader_broker, "is_connected", False))
        )
        try:
            live_ctrader = list(ctrader_broker.get_positions() or [])
            if is_ctrader_connected or live_ctrader:
                reconcile_ctrader_positions(db, live_ctrader, broker=ctrader_broker)
            elif live_ctrader:
                upsert_ctrader_live_trades(db, live_ctrader)
        except Exception as exc:
            logger.warning("cTrader live-book sync for dashboard failed: %s", exc)

        live_binance, binance_snapshot_ok = await _live_binance_book()
        if binance_snapshot_ok:
            try:
                upsert_binance_live_trades(db, live_binance)
            except Exception as exc:
                logger.warning("Binance live-book upsert for dashboard failed: %s", exc)
        live_binance_syms = {
            str(p.get("symbol") or "").upper()
            for p in live_binance
            if abs(float(p.get("quantity") or 0)) > 0
        }

        open_trades = db.query(Trade).filter(Trade.status.in_(["open", "filled"])).all()
        non_ctrader_symbols = {
            t.symbol for t in open_trades if not _is_ctrader_trade(t)
        }
        mark_prices = await _fetch_mark_prices_for_symbols(non_ctrader_symbols)
        if live_ctrader:
            try:
                ctrader_broker.ensure_spot_quotes(
                    [str(p.get("symbol") or "") for p in live_ctrader if p.get("symbol")]
                )
            except Exception as exc:
                logger.warning("cTrader spot quote refresh failed: %s", exc)
        live_by_pid = {
            str(p.get("position_id")): p for p in live_ctrader if p.get("position_id")
        }
        live_by_symbol = {
            str(p.get("symbol") or "").upper(): p for p in live_ctrader if p.get("symbol")
        }
        live_bn_by_key = {}
        for p in live_binance:
            if abs(float(p.get("quantity") or 0)) <= 0:
                continue
            side = str(p.get("side") or p.get("direction") or "BUY").upper()
            live_bn_by_key[binance_position_key(str(p.get("symbol") or ""), side)] = p
        positions = []
        live_mode = get_trading_mode() == TradingMode.LIVE
        for t in open_trades:
            is_ctrader = _is_ctrader_trade(t)
            direction = str(t.direction or "BUY").upper()
            sym = str(t.symbol or "").upper()
            if live_mode and not is_ctrader and is_binance_paper_fill(t):
                continue
            if (
                live_mode
                and not is_ctrader
                and binance_snapshot_ok
                and sym not in live_binance_syms
            ):
                continue

            if is_ctrader:
                pid = str(getattr(t, "broker_position_id", "") or "").strip()
                # If cTrader is connected or has live positions, skip ghost rows absent from live book
                # A non-empty live book is authoritative. An empty snapshot is
                # ambiguous (rate-limit / reconnect) — keep showing the DB row
                # instead of hiding it as a ghost after we refused a bulk close.
                if live_ctrader:
                    if pid and pid not in live_by_pid:
                        continue
                    if not pid and sym not in live_by_symbol:
                        continue
                mark = ctrader_broker.get_mark_price(sym, direction)
                current_price = _coalesce_mark_price(mark, t.entry_price)
                live = live_by_pid.get(pid) or live_by_symbol.get(sym)
                lots = float((live or {}).get("quantity") or t.quantity or 0)
                volume_cents = (live or {}).get("volume_cents")
                if volume_cents:
                    units = float(volume_cents) / 100.0
                else:
                    units = lots * ctrader_broker.units_per_lot(
                        sym, lot_size_cents=ctrader_broker.lot_size_cents(sym)
                    )
                if live and live.get("unrealized_pnl") is not None and (
                    live.get("current_price") or volume_cents
                ):
                    unrealized_pnl = float(live["unrealized_pnl"])
                elif mark and t.entry_price and units:
                    direction_mult = 1 if direction == "BUY" else -1
                    quote_pnl = (mark - float(t.entry_price)) * units * direction_mult
                    rate = ctrader_broker.quote_to_usd_rate(sym, mark)
                    unrealized_pnl = round(quote_pnl * rate, 2) if rate else round(quote_pnl, 2)
                else:
                    unrealized_pnl = 0.0
            else:
                live_bn = live_bn_by_key.get(binance_position_key(sym, direction))
                bn_qty = float((live_bn or {}).get("quantity") or t.quantity or 0)
                bn_entry = float((live_bn or {}).get("entry_price") or t.entry_price or 0)
                if live_bn:
                    current_price = _coalesce_mark_price(
                        live_bn.get("mark_price") or live_bn.get("current_price") or mark_prices.get(t.symbol),
                        bn_entry,
                    )
                    if live_bn.get("unrealized_pnl") is not None:
                        unrealized_pnl = float(live_bn["unrealized_pnl"])
                    elif direction == "BUY":
                        unrealized_pnl = (current_price - bn_entry) * bn_qty
                    else:
                        unrealized_pnl = (bn_entry - current_price) * bn_qty
                else:
                    current_price = _coalesce_mark_price(
                        mark_prices.get(t.symbol),
                        t.entry_price,
                    )
                    bn_qty = float(t.quantity or 0)
                    bn_entry = float(t.entry_price or 0)
                    if direction == "BUY":
                        unrealized_pnl = (current_price - bn_entry) * bn_qty
                    else:
                        unrealized_pnl = (bn_entry - current_price) * bn_qty

            out_qty = lots if is_ctrader else bn_qty
            out_entry = float(t.entry_price or 0) if is_ctrader else bn_entry
            pnl_pct = position_pnl_pct(
                entry=out_entry,
                mark=current_price,
                direction=direction,
                unrealized_pnl=unrealized_pnl,
                quantity=out_qty,
                is_ctrader=is_ctrader,
            )

            payload = {
                "id": t.id,
                "symbol": t.symbol,
                "direction": t.direction,
                "quantity": out_qty,
                "entry_price": out_entry,
                "current_price": current_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "unrealized_pnl": round(unrealized_pnl, 2),
                "unrealized_pnl_pct": round(pnl_pct, 2),
                "strategy": t.strategy,
                "broker": "ctrader" if is_ctrader else (
                    getattr(t, "broker", None) or getattr(t, "exchange", None)
                ),
                "broker_position_id": getattr(t, "broker_position_id", None),
                "opened_at": t.timestamp.isoformat() if t.timestamp else None,
            }
            positions.append(overlay_live_mark(payload, live_by_pid, live_by_symbol))
        return {"positions": positions, "count": len(positions)}
    finally:
        db.close()


# ── Trade History ─────────────────────────────────────────────────────────────

@router.get("/trades")
async def get_trades(
    symbol: Optional[str] = Query(None),
    strategy: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Get trade history with filtering."""
    db = SessionLocal()
    try:
        if status in (None, "open", "filled"):
            try:
                is_ctrader_connected = (
                    ctrader_broker.is_connected()
                    if callable(getattr(ctrader_broker, "is_connected", None))
                    else bool(getattr(ctrader_broker, "is_connected", False))
                )
                live_ctrader = list(ctrader_broker.get_positions() or [])
                if is_ctrader_connected or live_ctrader:
                    reconcile_ctrader_positions(db, live_ctrader, broker=ctrader_broker)
            except Exception as exc:
                logger.warning("cTrader reconcile in get_trades failed: %s", exc)
            try:
                live_binance, binance_snapshot_ok = await _live_binance_book()
                if binance_snapshot_ok:
                    upsert_binance_live_trades(db, live_binance)
            except Exception as exc:
                logger.warning("Binance live-book upsert in get_trades failed: %s", exc)

        q = db.query(Trade).order_by(Trade.id.desc())
        if symbol:
            q = q.filter(Trade.symbol == symbol)
        if strategy:
            q = q.filter(Trade.strategy == strategy)
        if status:
            q = q.filter(Trade.status == status)

        total = q.count()
        trades = q.offset(offset).limit(limit).all()

        return {
            "trades": [
                {
                    "id": t.id,
                    "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                    "closed_at": t.closed_at.isoformat() if t.closed_at else None,
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "quantity": t.quantity,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "stop_loss": t.stop_loss,
                    "take_profit": t.take_profit,
                    "status": t.status,
                    "pnl": t.pnl,
                    "strategy": t.strategy,
                    "broker": getattr(t, "broker", None) or getattr(t, "exchange", None),
                    "notes": t.notes,
                }
                for t in trades
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    finally:
        db.close()


# ── Signals History ───────────────────────────────────────────────────────────

@router.get("/signals/history")
async def get_signals_history(
    symbol: Optional[str] = Query(None),
    strategy: Optional[str] = Query(None),
    direction: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Get all generated signals with pagination."""
    db = SessionLocal()
    try:
        q = db.query(TradingSignal).order_by(TradingSignal.id.desc())
        if symbol:
            q = q.filter(TradingSignal.symbol == symbol)
        if strategy:
            q = q.filter(TradingSignal.strategy == strategy)
        if direction:
            q = q.filter(TradingSignal.direction == direction)

        total = q.count()
        signals = q.offset(offset).limit(limit).all()

        return {
            "signals": [
                {
                    "id": s.id,
                    "timestamp": s.timestamp.isoformat() if s.timestamp else None,
                    "symbol": s.symbol,
                    "strategy": s.strategy,
                    "direction": s.direction,
                    "confidence": s.confidence,
                    "entry_price": s.entry_price,
                    "stop_loss": s.stop_loss,
                    "take_profit": s.take_profit,
                    "status": s.status,
                    "reasoning": s.reasoning,
                    "ai_analysis": json.loads(s.ai_analysis) if s.ai_analysis else None,
                }
                for s in signals
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    finally:
        db.close()


# ── Portfolio History ─────────────────────────────────────────────────────────

@router.get("/portfolio/history")
async def get_portfolio_history(
    limit: int = Query(100, ge=1, le=1000),
):
    """Get portfolio snapshots for equity curve."""
    db = SessionLocal()
    try:
        snapshots = (
            db.query(PortfolioSnapshot)
            .order_by(PortfolioSnapshot.id.desc())
            .limit(limit)
            .all()
        )
        snapshots.reverse()
        return {
            "snapshots": [
                {
                    "id": s.id,
                    "timestamp": s.timestamp.isoformat() if s.timestamp else None,
                    "total_value": s.total_value,
                    "cash": s.cash,
                    "positions_value": s.positions_value,
                    "total_pnl": s.total_pnl,
                    "open_positions": s.open_positions,
                    "cycle_number": s.cycle_number,
                }
                for s in snapshots
            ],
            "count": len(snapshots),
        }
    finally:
        db.close()


# ── AI Analysis Pipeline ─────────────────────────────────────────────────────

@router.get("/models")
async def get_ai_models():
    """Show which AI models are configured and their roles."""
    return ai_analysis_service.models_info


@router.get("/analysis/{signal_id}")
async def get_signal_analysis(signal_id: int):
    """Get full AI analysis for a specific signal."""
    db = SessionLocal()
    try:
        signal = db.query(TradingSignal).filter(TradingSignal.id == signal_id).first()
        if not signal:
            return {"error": "Signal not found"}
        if signal.ai_analysis:
            return json.loads(signal.ai_analysis)
        return {"error": "No AI analysis available for this signal"}
    finally:
        db.close()


@router.post("/analyze")
async def run_analysis_on_demand(request: dict):
    """Run AI analysis on-demand for a symbol."""
    symbol = request.get("symbol", "BTC-USD")

    if not ai_analysis_service.enabled:
        return {"error": "AI analysis is disabled. Set AI_ANALYSIS_ENABLED=true in .env"}

    # Fetch bars using the trading loop's method
    import yfinance as yf
    try:
        yf_sym = _to_yfinance_symbol(symbol)
        ticker = yf.Ticker(yf_sym)
        df = ticker.history(period="30d", interval="1h")
        if df.empty:
            df = ticker.history(period="60d", interval="1d")
        if df.empty:
            return {"error": f"Could not fetch data for {symbol}"}

        bars = []
        for idx, row in df.iterrows():
            bars.append({
                "timestamp": int(idx.timestamp()) if hasattr(idx, 'timestamp') else 0,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume", 0)),
            })
    except Exception as e:
        return {"error": f"Failed to fetch data for {symbol}: {str(e)}"}

    # Run AI analysis
    try:
        result = await ai_analysis_service.analyze_symbol(symbol, bars)

        # Store as a signal in DB
        db = SessionLocal()
        try:
            db_signal = TradingSignal(
                symbol=symbol,
                strategy="ai_analysis",
                direction=result.get('direction', 'HOLD'),
                confidence=result.get('confidence', 0.0),
                entry_price=result.get('entry_price'),
                stop_loss=result.get('stop_loss'),
                take_profit=result.get('take_profit'),
                status="ai_analyzed",
                reasoning=result.get('reasoning', ''),
                ai_analysis=json.dumps(result),
            )
            db.add(db_signal)
            db.commit()
            result['signal_id'] = db_signal.id
        finally:
            db.close()

        return result
    except Exception as e:
        return {"error": f"AI analysis failed: {str(e)}"}


@router.get("/brokers")
async def get_brokers_status():
    """Get real-time operational status and metrics for all registered brokers."""
    from backend.services.binance_futures_service import binance_futures_broker
    from backend.services.ctrader_service import ctrader_broker
    from backend.services.broker_circuit_breaker import broker_circuit_breaker

    binance_status = binance_futures_broker.status()
    ctrader_status = ctrader_broker.status()

    binance_avail, binance_breaker = broker_circuit_breaker.is_available("binance_futures")
    ctrader_avail, ctrader_breaker = broker_circuit_breaker.is_available("ctrader")

    return {
        "brokers": {
            "binance_futures": {
                "name": "Binance Futures (Crypto)",
                "status": binance_status,
                "circuit_breaker": {
                    "available": binance_avail,
                    "reason": binance_breaker,
                },
            },
            "ctrader": {
                "name": "cTrader Open API (Forex & Multi-Asset)",
                "status": ctrader_status,
                "circuit_breaker": {
                    "available": ctrader_avail,
                    "reason": ctrader_breaker,
                },
            },
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/markets")
async def get_markets():
    """Get multi-asset watchlist and supported instruments across all brokers."""
    markets = [
        # Forex & Commodities (cTrader)
        {"symbol": "EURUSD", "display_name": "EUR/USD", "asset_class": "forex", "broker": "ctrader", "lot_size": 100000, "digits": 5, "base": "EUR", "quote": "USD"},
        {"symbol": "GBPUSD", "display_name": "GBP/USD", "asset_class": "forex", "broker": "ctrader", "lot_size": 100000, "digits": 5, "base": "GBP", "quote": "USD"},
        {"symbol": "USDJPY", "display_name": "USD/JPY", "asset_class": "forex", "broker": "ctrader", "lot_size": 100000, "digits": 3, "base": "USD", "quote": "JPY"},
        {"symbol": "AUDUSD", "display_name": "AUD/USD", "asset_class": "forex", "broker": "ctrader", "lot_size": 100000, "digits": 5, "base": "AUD", "quote": "USD"},
        {"symbol": "USDCAD", "display_name": "USD/CAD", "asset_class": "forex", "broker": "ctrader", "lot_size": 100000, "digits": 5, "base": "USD", "quote": "CAD"},
        {"symbol": "USDCHF", "display_name": "USD/CHF", "asset_class": "forex", "broker": "ctrader", "lot_size": 100000, "digits": 5, "base": "USD", "quote": "CHF"},
        {"symbol": "NZDUSD", "display_name": "NZD/USD", "asset_class": "forex", "broker": "ctrader", "lot_size": 100000, "digits": 5, "base": "NZD", "quote": "USD"},
        {"symbol": "XAUUSD", "display_name": "Gold / USD", "asset_class": "metals", "broker": "ctrader", "lot_size": 100, "digits": 2, "base": "XAU", "quote": "USD"},
        {"symbol": "XAGUSD", "display_name": "Silver / USD", "asset_class": "metals", "broker": "ctrader", "lot_size": 1000, "digits": 3, "base": "XAG", "quote": "USD"},
        # Crypto Perpetuals (Binance Futures)
        {"symbol": "BTCUSDT", "display_name": "Bitcoin Perpetual", "asset_class": "crypto", "broker": "binance_futures", "lot_size": 1, "digits": 2, "base": "BTC", "quote": "USDT"},
        {"symbol": "ETHUSDT", "display_name": "Ethereum Perpetual", "asset_class": "crypto", "broker": "binance_futures", "lot_size": 1, "digits": 2, "base": "ETH", "quote": "USDT"},
        {"symbol": "SOLUSDT", "display_name": "Solana Perpetual", "asset_class": "crypto", "broker": "binance_futures", "lot_size": 1, "digits": 2, "base": "SOL", "quote": "USDT"},
        {"symbol": "BNBUSDT", "display_name": "BNB Perpetual", "asset_class": "crypto", "broker": "binance_futures", "lot_size": 1, "digits": 2, "base": "BNB", "quote": "USDT"},
        {"symbol": "AVAXUSDT", "display_name": "Avalanche Perpetual", "asset_class": "crypto", "broker": "binance_futures", "lot_size": 1, "digits": 2, "base": "AVAX", "quote": "USDT"},
        {"symbol": "ADAUSDT", "display_name": "Cardano Perpetual", "asset_class": "crypto", "broker": "binance_futures", "lot_size": 1, "digits": 4, "base": "ADA", "quote": "USDT"},
        {"symbol": "DOTUSDT", "display_name": "Polkadot Perpetual", "asset_class": "crypto", "broker": "binance_futures", "lot_size": 1, "digits": 3, "base": "DOT", "quote": "USDT"},
        {"symbol": "LINKUSDT", "display_name": "Chainlink Perpetual", "asset_class": "crypto", "broker": "binance_futures", "lot_size": 1, "digits": 3, "base": "LINK", "quote": "USDT"},
    ]
    return {"markets": markets, "total": len(markets)}


class SmartOrderRequest(BaseModel):
    symbol: str
    direction: Literal["BUY", "SELL"]
    quantity: float = Field(gt=0)
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    broker_override: Optional[Literal["binance_futures", "ctrader"]] = None


@router.post("/order/smart")
async def place_smart_order(req: SmartOrderRequest):
    """
    Intelligent multi-broker order router.
    Routes crypto to Binance Futures, forex/metals/CFDs to cTrader Open API.
    Enforces per-broker circuit breakers and logs fills.
    """
    from backend.services.binance_futures_service import binance_futures_broker
    from backend.services.ctrader_service import ctrader_broker
    from backend.services.broker_circuit_breaker import broker_circuit_breaker
    from backend.services.sentry_state import is_trading_allowed
    from backend.services.trading_mode import (
        BINANCE_PAPER_SESSION_ID,
        TradingMode,
        binance_paper_parallel_enabled,
        get_trading_mode,
        paper_starting_balance,
    )

    if not is_trading_allowed():
        raise HTTPException(status_code=403, detail="Trading is halted by sentry.")

    clean_sym = req.symbol.upper().replace("=X", "").replace("-", "").replace("/", "")

    # 1. Determine Target Broker
    if req.broker_override:
        target_broker_name = req.broker_override
    elif clean_sym.endswith("USDT") or clean_sym in ["BTC", "ETH", "SOL", "BNB", "AVAX", "ADA", "DOT", "LINK", "DOGE"]:
        target_broker_name = "binance_futures"
    else:
        target_broker_name = "ctrader"

    # 2. Check Circuit Breaker
    avail, reason = broker_circuit_breaker.is_available(target_broker_name)
    if not avail:
        raise HTTPException(
            status_code=503,
            detail=f"Target broker '{target_broker_name}' is temporarily unavailable: {reason}",
        )

    # 3. Route to Target Broker
    db = SessionLocal()
    try:
        paper_crypto = (
            binance_paper_parallel_enabled()
            and target_broker_name == "binance_futures"
        )
        if get_trading_mode() != TradingMode.LIVE or paper_crypto:
            ut = UnifiedTrading()
            paper_sid = (
                BINANCE_PAPER_SESSION_ID
                if target_broker_name == "binance_futures"
                else f"{target_broker_name}_paper"
            )
            ut.init_session(
                target_broker_name,
                mode="paper",
                paper_balance=paper_starting_balance(),
                session_id=paper_sid,
            )
            px = float(req.price or 0)
            if px <= 0:
                from backend.services.binance_market_data import binance_market_data
                tick = await binance_market_data.get_ticker_24h(clean_sym)
                px = float((tick or {}).get("lastPrice") or 0)
            side = OrderSide.BUY if req.direction == "BUY" else OrderSide.SELL
            paper_resp = ut.place_order(
                UnifiedOrder(
                    symbol=clean_sym,
                    side=side,
                    order_type=OrderType.MARKET if req.order_type == "MARKET" else OrderType.LIMIT,
                    quantity=req.quantity,
                    price=px,
                    stop_loss=float(req.stop_loss or 0),
                    take_profit=float(req.take_profit or 0),
                ),
                session_id=paper_sid,
            )
            result = {
                "status": "filled" if paper_resp.success else "error",
                "order_id": paper_resp.order_id,
                "price": paper_resp.filled_price or px,
                "filled_price": paper_resp.filled_price or px,
                "message": paper_resp.message,
            }
        elif target_broker_name == "binance_futures":
            result = binance_futures_broker.place_order(
                symbol=clean_sym,
                direction=req.direction,
                quantity=req.quantity,
                price=req.price,
                stop_loss=req.stop_loss,
                take_profit=req.take_profit,
            )
            broker_circuit_breaker.record_success("binance_futures")
        else:
            result = ctrader_broker.place_order(
                symbol=clean_sym,
                direction=req.direction,
                volume=req.quantity,
                price=req.price,
                stop_loss=req.stop_loss,
                take_profit=req.take_profit,
            )
            broker_circuit_breaker.record_success("ctrader")

        # 4. Record Trade in DB
        if result.get("status") in ["sent", "simulated", "filled", "ok"]:
            db_trade = Trade(
                symbol=clean_sym,
                direction=req.direction,
                quantity=req.quantity,
                entry_price=float(result.get("price") or req.price or 0.0),
                stop_loss=req.stop_loss,
                take_profit=req.take_profit,
                status="open",
                broker=target_broker_name,
                broker_order_id=str(result.get("order_id") or "") or None,
                broker_position_id=str(result.get("position_id") or "") or None,
                broker_metadata=result,
                notes=f"Smart order routed to {target_broker_name}",
            )
            db.add(db_trade)
            db.commit()
            result["db_trade_id"] = db_trade.id

        return {"success": True, "target_broker": target_broker_name, "execution": result}

    except Exception as e:
        broker_circuit_breaker.record_error(target_broker_name, str(e))
        logger.error(f"[SMART ORDER ERROR] Routing to {target_broker_name} failed: {e}")
        raise HTTPException(status_code=500, detail=f"Order routing error on {target_broker_name}: {str(e)}")
    finally:
        db.close()


@router.get("/ctrader/tokens")
async def get_ctrader_tokens_info():
    """Retrieve cTrader token storage status and expiration info."""
    from backend.services.ctrader_tokens import ctrader_token_store
    tokens = ctrader_token_store.get_tokens()
    if not tokens:
        return {"configured": False, "message": "No cTrader OAuth tokens stored"}
    
    expires_at = tokens.get("updated_at", 0) + tokens.get("expires_in", 0)
    now_ts = datetime.now(timezone.utc).timestamp()
    days_left = max(0.0, round((expires_at - now_ts) / 86400.0, 1))

    return {
        "configured": True,
        "account_id": tokens.get("account_id"),
        "client_id": tokens.get("client_id", "")[:6] + "..." if tokens.get("client_id") else "",
        "days_until_expiration": days_left,
        "is_expired": days_left <= 0,
        "updated_at": tokens.get("updated_at"),
    }


class SaveCTraderTokensRequest(BaseModel):
    account_id: int
    client_id: str
    client_secret: Optional[str] = None
    access_token: str
    refresh_token: str
    expires_in: Optional[int] = 2592000


@router.post("/ctrader/tokens")
async def save_ctrader_tokens(req: SaveCTraderTokensRequest):
    """Save or update persistent OAuth tokens for cTrader Open API."""
    from backend.services.ctrader_tokens import ctrader_token_store
    from backend.services.ctrader_service import ctrader_broker

    data = {
        "account_id": req.account_id,
        "client_id": req.client_id,
        "client_secret": req.client_secret or "",
        "access_token": req.access_token,
        "refresh_token": req.refresh_token,
        "expires_in": req.expires_in or 2592000,
        "updated_at": int(datetime.now(timezone.utc).timestamp()),
    }
    ctrader_token_store.save_tokens(data)
    ctrader_broker._token_store = ctrader_token_store
    return {"success": True, "message": "cTrader tokens stored and updated successfully"}


class CTraderOAuthExchangeRequest(BaseModel):
    code: str = Field(..., min_length=4, max_length=512)
    redirect_uri: Optional[str] = None


@router.get("/ctrader/oauth/url")
async def get_ctrader_oauth_url():
    """Browser authorization URL for requesting new cTrader OAuth tokens."""
    from backend.brokers.auth import get_auth_url
    from backend.services.ctrader_oauth import is_sandbox_app, ctrader_env

    redirect_uri = os.getenv("CTRADER_REDIRECT_URI", "https://localhost/callback")
    return {
        "auth_url": get_auth_url(redirect_uri=redirect_uri),
        "redirect_uri": redirect_uri,
        "ctrader_env": ctrader_env(),
        "sandbox_app": is_sandbox_app(),
        "instructions": (
            "Open auth_url in a browser, approve access, copy the `code` query param "
            "from the redirect URL, then POST /trading/ctrader/oauth/exchange with {\"code\": \"...\"}."
        ),
    }


@router.post("/ctrader/oauth/exchange")
async def exchange_ctrader_oauth_code(req: CTraderOAuthExchangeRequest, request: Request):
    """Exchange Spotware authorization code for fresh access + refresh tokens."""
    from backend.security import validate_admin_request
    from backend.brokers.auth import exchange_code_for_token
    from backend.services.ctrader_tokens import ctrader_token_store
    from backend.services.ctrader_service import ctrader_broker

    validate_admin_request(request)
    redirect_uri = req.redirect_uri or os.getenv("CTRADER_REDIRECT_URI", "https://localhost/callback")

    try:
        tokens = exchange_code_for_token(
            code=req.code.strip(),
            redirect_uri=redirect_uri,
            save_to_env=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"OAuth exchange failed: {exc}") from exc

    account_raw = os.getenv("CTRADER_ACCOUNT_ID", "0").strip()
    data = {
        "account_id": int(account_raw) if account_raw.isdigit() else 0,
        "client_id": os.getenv("CTRADER_CLIENT_ID", ""),
        "client_secret": os.getenv("CTRADER_CLIENT_SECRET", ""),
        "access_token": tokens["access_token"],
        "refresh_token": tokens.get("refresh_token", ""),
        "expires_in": tokens.get("expires_in", 2_628_000),
        "updated_at": int(datetime.now(timezone.utc).timestamp()),
    }
    ctrader_token_store.save_tokens(data)

    env_path = Path(".env")
    if env_path.exists():
        from dotenv import set_key
        set_key(str(env_path), "CTRADER_ACCESS_TOKEN", data["access_token"])
        set_key(str(env_path), "CTRADER_REFRESH_TOKEN", data["refresh_token"])
    os.environ["CTRADER_ACCESS_TOKEN"] = data["access_token"]
    os.environ["CTRADER_REFRESH_TOKEN"] = data["refresh_token"]
    ctrader_broker._token_store = ctrader_token_store

    return {
        "success": True,
        "message": "New cTrader tokens saved. Restart backend or call POST /trading/ctrader/enable to connect.",
        "account_id": data["account_id"],
        "expires_in": data["expires_in"],
    }


class PipMarginCalcRequest(BaseModel):
    symbol: str
    lots: float = Field(gt=0, default=1.0)
    price: Optional[float] = None
    leverage: float = Field(gt=0, default=100.0)
    deposit_asset: str = "USD"


@router.get("/ctrader/trendbars")
async def get_ctrader_trendbars(
    symbol: str = Query("EURUSD", description="Symbol name e.g. EURUSD, GBPUSD, BTCUSD"),
    period: str = Query("M5", description="Period: 1M, 5M, 15M, 30M, 1H, 4H, 1D, 1W"),
    count: int = Query(120, ge=1, le=1000, description="Number of bars"),
    from_ts: Optional[int] = Query(None, description="Start unix timestamp in ms"),
    to_ts: Optional[int] = Query(None, description="End unix timestamp in ms"),
):
    """
    Get historical OHLCV trendbars for cTrader charting and technical analysis.

    Live sessions return broker data only; an empty ``bars`` list means no real
    history was available (timeout or missing symbol). Paper mode may include
    synthetic bars flagged with ``synthetic: true``.
    """
    from backend.services.ctrader_service import ctrader_broker
    bars = await asyncio.to_thread(
        ctrader_broker.get_trendbars,
        symbol=symbol,
        period=period,
        from_ts=from_ts,
        to_ts=to_ts,
        count=count,
    )
    return {
        "symbol": symbol.upper(),
        "period": period.upper(),
        "count": len(bars),
        "bars": bars,
        "synthetic": bool(bars) and all(bar.get("synthetic") for bar in bars),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/ctrader/ticks")
async def get_ctrader_ticks(
    symbol: str = Query("EURUSD", description="Symbol name"),
    type: str = Query("BID", description="Quote type: BID or ASK"),
    hours: int = Query(4, ge=1, le=72, description="Number of historical hours"),
):
    """
    Get historical tick data stream for cTrader symbols.
    Uses ProtoOAGetTickDataReq when connected.
    """
    from backend.services.ctrader_service import ctrader_broker
    ticks = await asyncio.to_thread(
        ctrader_broker.get_tick_data,
        symbol=symbol,
        quote_type=type,
        hours=hours,
    )
    return {
        "symbol": symbol.upper(),
        "type": type.upper(),
        "count": len(ticks),
        "ticks": ticks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/ctrader/symbol-spec")
async def get_ctrader_symbol_spec(
    symbol: str = Query("EURUSD", description="Symbol name"),
):
    """
    Get detailed financial specification for a symbol (digits, pip position, lot step, tick size).
    """
    from backend.services.ctrader_service import ctrader_broker
    spec = ctrader_broker.get_symbol_specification(symbol)
    return spec


@router.post("/ctrader/calc/pip-margin")
async def calculate_pip_margin_endpoint(req: PipMarginCalcRequest):
    """
    Calculate pip value, tick value, required margin, and notional lot volume
    following OpenAPI.Net financial formulas.
    """
    from backend.services.ctrader_service import ctrader_broker
    result = ctrader_broker.calculate_pip_margin(
        symbol=req.symbol,
        lots=req.lots,
        price=req.price,
        leverage=req.leverage,
        deposit_asset=req.deposit_asset,
    )
    return result


class AIAgentTradeRequest(BaseModel):
    prompt: str
    provider: Optional[str] = "xai"
    model: Optional[str] = "grok-beta"


@router.post("/ai/parse-trade")
async def ai_parse_trade(req: AIAgentTradeRequest):
    """
    Parses an AI trading instruction, extracts trading parameters,
    and executes via the multi-broker Smart Order router.
    """
    from backend.services.sentry_state import is_trading_allowed
    if not is_trading_allowed():
        raise HTTPException(status_code=403, detail="Trading is halted by sentry.")
    import re
    prompt = req.prompt.lower()
    direction: Literal["BUY", "SELL"] = "BUY" if any(w in prompt for w in ["buy", "long"]) else "SELL"

    symbol = "BTCUSDT"
    if "eur" in prompt:
        symbol = "EURUSD"
    elif "gbp" in prompt:
        symbol = "GBPUSD"
    elif "gold" in prompt or "xau" in prompt:
        symbol = "XAUUSD"
    elif "eth" in prompt:
        symbol = "ETHUSDT"
    elif "sol" in prompt:
        symbol = "SOLUSDT"

    qty_match = re.search(r'(?:qty|quantity|lots?|size)\s*[:=]?\s*(\d+(\.\d+)?)', prompt)
    quantity = float(qty_match.group(1)) if qty_match else (0.01 if symbol.endswith("USDT") else 0.01)

    smart_req = SmartOrderRequest(
        symbol=symbol,
        direction=direction,
        quantity=quantity,
    )
    res = await place_smart_order(smart_req)
    return {
        "success": True,
        "ai_prompt": req.prompt,
        "extracted_intent": {
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
        },
        "execution": res,
    }


@router.get("/ctrader/status")
async def ctrader_status():
    """Get cTrader broker status."""
    from backend.services.ctrader_service import ctrader_broker
    return ctrader_broker.status()


@router.post("/ctrader/enable")
async def enable_ctrader_live():
    """Enable real cTrader trading (disable DRY_RUN for cTrader)."""
    from backend.services.ctrader_service import ctrader_broker
    import asyncio
    loop = asyncio.get_event_loop()
    connected = await loop.run_in_executor(None, ctrader_broker.connect)
    st = ctrader_broker.status()
    return {
        "message": "cTrader live mode enabled" if connected else "cTrader connection failed",
        "connected": connected,
        "dry_run": st["dry_run"],
        "env": st["env"],
    }


@router.post("/ctrader/disable")
async def disable_ctrader_live():
    """Disable real cTrader trading (enable DRY_RUN)."""
    from backend.services.ctrader_service import ctrader_broker
    ctrader_broker.disconnect()
    return {"message": "cTrader paper mode enabled", "dry_run": True}


@router.get("/ctrader/positions")
async def get_ctrader_live_positions():
    """Live cTrader book (not the SQL Trade table)."""
    from backend.services.ctrader_service import ctrader_broker
    positions = ctrader_broker.get_positions()
    return {
        "connected": ctrader_broker.is_connected,
        "dry_run": ctrader_broker.dry_run,
        "positions": positions,
        "count": len(positions),
    }


@router.post("/ctrader/positions/{position_id}/close")
async def close_ctrader_live_position(
    position_id: str,
    volume: Optional[float] = Query(None, gt=0),
    symbol: Optional[str] = Query(None),
):
    """Close a cTrader position by broker positionId (demo/live book, not DB id)."""
    from backend.services.ctrader_service import ctrader_broker
    if not ctrader_broker.is_connected and not ctrader_broker.dry_run:
        raise HTTPException(status_code=400, detail="cTrader is not connected")
    res = ctrader_broker.close_position(position_id, symbol=symbol, volume=volume)
    if res.get("status") == "error":
        raise HTTPException(status_code=502, detail=res.get("error") or "cTrader close failed")
    return {
        "success": True,
        "result": res,
        "positions": ctrader_broker.get_positions(),
    }


@router.put("/ctrader/positions/{position_id}/amend")
async def amend_ctrader_live_position(position_id: str, body: ModifyPositionRequest):
    """Set absolute SL/TP on a live cTrader position. Send both legs; SL-only clears TP."""
    from backend.services.ctrader_service import ctrader_broker
    if body.stop_loss is None and body.take_profit is None:
        raise HTTPException(status_code=400, detail="Provide stop_loss and/or take_profit")
    if not ctrader_broker.is_connected and not ctrader_broker.dry_run:
        raise HTTPException(status_code=400, detail="cTrader is not connected")
    matched = next(
        (p for p in ctrader_broker.get_positions() if str(p.get("position_id")) == str(position_id)),
        None,
    )
    if not matched:
        raise HTTPException(status_code=404, detail=f"cTrader position {position_id} not found")
    side = str(matched.get("side") or matched.get("direction") or "").upper()
    mark = float(matched.get("current_price") or 0)
    sl = body.stop_loss
    tp = body.take_profit
    if mark > 0 and side in {"BUY", "LONG"}:
        if sl is not None and sl >= mark:
            raise HTTPException(status_code=400, detail="Refusing BUY stop at/above mark")
        if tp is not None and tp <= mark:
            raise HTTPException(status_code=400, detail="Refusing BUY take-profit at/below mark")
    if mark > 0 and side in {"SELL", "SHORT"}:
        if sl is not None and sl <= mark:
            raise HTTPException(status_code=400, detail="Refusing SELL stop at/below mark")
        if tp is not None and tp >= mark:
            raise HTTPException(status_code=400, detail="Refusing SELL take-profit at/above mark")
    res = ctrader_broker.amend_position_sltp(
        position_id, stop_loss=sl, take_profit=tp,
    )
    if res.get("status") == "error":
        raise HTTPException(status_code=502, detail=res.get("error") or "cTrader amend failed")
    return {
        "success": True,
        "result": res,
        "positions": ctrader_broker.get_positions(),
    }


# ── Position Management ────────────────────────────────────────────────────────

@router.post("/positions/{position_id}/close")
async def close_position(position_id: str):
    """Close a position on its originating broker, then persist the actual fill."""
    from backend.services.ctrader_service import ctrader_broker
    from backend.services.binance_futures_service import binance_futures_broker
    from backend.services.trading_mode import (
        get_trading_mode,
        TradingMode,
        live_binance_orders_allowed,
        binance_paper_parallel_enabled,
    )

    db = SessionLocal()
    try:
        trade = None
        str_pos = str(position_id).strip()
        if str_pos.isdigit():
            trade = db.query(Trade).filter(Trade.id == int(str_pos), Trade.status.in_(["open", "filled"])).first()
        if not trade:
            trade = db.query(Trade).filter(
                (Trade.broker_position_id == str_pos)
                | (Trade.broker_order_id == str_pos)
                | (Trade.binance_order_id == str_pos),
                Trade.status.in_(["open", "filled"]),
            ).first()
        if not trade:
            raise HTTPException(status_code=404, detail=f"Open position {position_id} not found")

        target_broker = getattr(trade, "broker", None) or getattr(trade, "exchange", None) or "binance_futures"
        is_binance_paper = (
            target_broker != "ctrader"
            and (
                not live_binance_orders_allowed()
                or binance_paper_parallel_enabled()
                or is_binance_paper_fill(trade)
            )
        )
        paper_mode = (get_trading_mode() != TradingMode.LIVE) or is_binance_paper

        if target_broker == "ctrader":
            res = ctrader_broker.close_position(
                position_id=trade.broker_position_id or trade.broker_order_id or str(trade.id),
                symbol=trade.symbol,
                volume=trade.quantity,
            )
            if res.get("status") == "error":
                raise HTTPException(
                    status_code=502,
                    detail=f"cTrader close failed; DB left open: {res.get('error')}",
                )
            # A live cTrader close is an async protocol send: the fill price
            # arrives later on the execution event. Falling back to entry_price
            # here recorded exit==entry and pnl==0.0 for every forex trade,
            # which hid real losses from the daily-loss and expectancy gates.
            exit_price = res.get("price") or ctrader_broker.get_mark_price(
                trade.symbol, trade.direction
            )
            exit_price = float(exit_price) if exit_price else None
            reported_pnl = res.get("pnl")
            if reported_pnl is not None:
                pnl = float(reported_pnl)
            elif exit_price and trade.entry_price:
                # trade.quantity is lots for cTrader; P&L needs contract units,
                # and lands in the quote currency until converted.
                units = float(trade.quantity or 0) * ctrader_broker.units_per_lot(
                    trade.symbol,
                    lot_size_cents=ctrader_broker.lot_size_cents(trade.symbol),
                )
                direction = 1 if trade.direction == "BUY" else -1
                quote_pnl = (exit_price - trade.entry_price) * units * direction
                rate = ctrader_broker.quote_to_usd_rate(trade.symbol, exit_price)
                pnl = quote_pnl * rate if rate else None
            else:
                pnl = None
        elif paper_mode:
            # Paper/backtest: close DB record without live exchange (loop trades are not
            # always mirrored in the in-memory paper engine, which caused fill errors).
            exit_price = float(trade.entry_price or 0)
            try:
                from backend.services.binance_market_data import binance_market_data
                tick = await binance_market_data.get_ticker_24h(trade.symbol)
                if tick and tick.get("lastPrice"):
                    exit_price = float(tick["lastPrice"])
            except Exception:
                pass
            if not exit_price or exit_price == float(trade.entry_price or 0):
                try:
                    ep = binance_futures_broker.get_exit_price(trade.symbol)
                    if ep and float(ep) > 0:
                        exit_price = float(ep)
                except Exception:
                    pass

            if trade.direction == "BUY":
                pnl = (exit_price - float(trade.entry_price or 0)) * float(trade.quantity or 0)
            else:
                pnl = (float(trade.entry_price or 0) - exit_price) * float(trade.quantity or 0)

        else:
            close_side = OrderSide.SELL if trade.direction == "BUY" else OrderSide.BUY
            try:
                ref_price = binance_futures_broker.get_exit_price(trade.symbol)
            except Exception as e:
                logger.warning(f"Reference price lookup failed for {trade.symbol}: {e}")
                ref_price = None
            order_price = float(ref_price) if ref_price else 0.0
            response = UnifiedTrading().place_order(UnifiedOrder(
                symbol=trade.symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=trade.quantity,
                price=order_price,
                reduce_only=True,
            ))
            if not response.success:
                raise HTTPException(
                    status_code=502,
                    detail=f"Exchange close failed; DB left open: {response.message}",
                )

            exit_price = response.filled_price
            if not exit_price:
                try:
                    exit_price = binance_futures_broker.get_exit_price(trade.symbol)
                except Exception as e:
                    logger.warning(f"Exit price fallback lookup failed for {trade.symbol}: {e}")
                    exit_price = None

            from backend.services.trading_loop_helpers import is_plausible_exit_price
            if is_plausible_exit_price(trade.entry_price, exit_price):
                if response.realized_pnl is not None:
                    pnl = float(response.realized_pnl) - response.commission
                else:
                    if trade.direction == "BUY":
                        pnl = (exit_price - trade.entry_price) * trade.quantity
                    else:
                        pnl = (trade.entry_price - exit_price) * trade.quantity
                    pnl -= response.commission
            else:
                exit_price = None
                pnl = None

        trade.exit_price = exit_price
        trade.pnl = round(pnl, 4) if pnl is not None else None
        trade.status = "closed"
        trade.closed_at = datetime.now(timezone.utc)
        suffix = "paper-mode DB close" if paper_mode and target_broker != "ctrader" else target_broker
        trade.notes = (trade.notes or "") + f" | Closed on {suffix} via manual dashboard"
        db.commit()
        return {
            "success": True,
            "position_id": position_id,
            "broker": target_broker,
            "exit_price": exit_price,
            "pnl": pnl,
            "paper_mode": paper_mode,
            "message": f"Position {position_id} closed on {target_broker}",
        }
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@router.put("/positions/{position_id}/modify")
async def modify_position(position_id: str, body: ModifyPositionRequest):
    """Modify exchange SL/TP first, then persist levels that succeeded."""
    db = SessionLocal()
    try:
        trade = None
        str_pos = str(position_id).strip()
        if str_pos.isdigit():
            trade = db.query(Trade).filter(Trade.id == int(str_pos), Trade.status.in_(["open", "filled"])).first()
        if not trade:
            trade = db.query(Trade).filter(
                (Trade.broker_position_id == str_pos)
                | (Trade.broker_order_id == str_pos)
                | (Trade.binance_order_id == str_pos),
                Trade.status.in_(["open", "filled"]),
            ).first()
        if not trade:
            raise HTTPException(status_code=404, detail=f"Open position {position_id} not found")

        from backend.services.binance_futures_service import binance_futures_broker
        results = {}
        failures = []
        if body.stop_loss is not None:
            result = binance_futures_broker.replace_stop_loss(
                trade.symbol, trade.direction, body.stop_loss,
            )
            results["stop_loss"] = result
            if result.get("status") in {"replaced", "simulated"}:
                trade.stop_loss = body.stop_loss
            else:
                failures.append(f"SL: {result.get('reason') or result.get('message')}")
        if body.take_profit is not None:
            result = binance_futures_broker.replace_take_profit(
                trade.symbol, trade.direction, body.take_profit,
            )
            results["take_profit"] = result
            if result.get("status") in {"replaced", "simulated"}:
                trade.take_profit = body.take_profit
            else:
                failures.append(f"TP: {result.get('reason') or result.get('message')}")

        db.commit()
        if failures:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "One or more exchange protection updates failed",
                    "failures": failures,
                    "results": results,
                },
            )
        return {
            "success": True,
            "position_id": position_id,
            "stop_loss": trade.stop_loss,
            "take_profit": trade.take_profit,
            "exchange_results": results,
        }
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── Binance Futures Endpoints ─────────────────────────────────────────────────

@router.get("/binance/status")
async def binance_status():
    """Get Binance Futures broker status: wallet, positions, open orders."""
    try:
        from backend.services.binance_futures_service import binance_futures_broker as bf
        active_broker = os.getenv("ACTIVE_BROKER", "ctrader")
        balance   = bf.get_balance()
        positions = bf.get_positions()
        orders    = bf.get_open_orders()
        return {
            "broker":       "binance_futures",
            "active":       active_broker == "binance_futures",
            "testnet":      os.getenv("BINANCE_TESTNET", "false") == "true",
            "leverage":     int(os.getenv("BINANCE_LEVERAGE", "10")),
            "margin_type":  bf.margin_type,
            "wallet": {
                "balance":         balance.get("balance", 0),
                "available":       balance.get("available", 0),
                "equity":          balance.get("equity", 0),
                "unrealized_pnl":  balance.get("unrealized_pnl", 0),
                "margin_used":     balance.get("margin_used", 0),
            },
            "positions":     positions,
            "open_orders":   orders,
            "positions_count": len(positions),
            "orders_count":    len(orders),
        }
    except Exception as e:
        return {"broker": "binance_futures", "active": False, "error": str(e)}


@router.get("/binance/wallet")
async def binance_wallet():
    """Get Binance Futures USDT-M wallet balance."""
    try:
        from backend.services.binance_futures_service import binance_futures_broker as bf
        return bf.get_balance()
    except Exception as e:
        return {"error": str(e)}


@router.get("/binance/positions")
async def binance_positions():
    """Get all open Binance Futures positions."""
    try:
        from backend.services.binance_futures_service import binance_futures_broker as bf
        return {"positions": bf.get_positions()}
    except Exception as e:
        return {"error": str(e)}


@router.post("/binance/enable")
async def binance_enable():
    """Switch active broker to Binance Futures."""
    os.environ["ACTIVE_BROKER"] = "binance_futures"
    return {"status": "ok", "active_broker": "binance_futures"}


@router.post("/binance/disable")
async def binance_disable():
    """Switch active broker back to paper/ctrader."""
    os.environ["ACTIVE_BROKER"] = "ctrader"
    return {"status": "ok", "active_broker": "ctrader"}


@router.get("/binance/{endpoint:path}")
async def binance_proxy(endpoint: str, request: Request):
    """Server-side proxy for Binance public spot market data.

    The dashboard queries api.binance.com directly from the browser, which
    fails where Binance is geo-blocked (e.g. US/EEA IPs return HTTP 451) or via
    CORS. The backend reaches Binance reliably, so the frontend falls back to
    this passthrough. Only read-only public market-data endpoints are allowed.
    """
    endpoint = endpoint.strip("/")
    if endpoint not in _BINANCE_PROXY_ALLOWED:
        return JSONResponse({"error": f"endpoint not allowed: {endpoint}"}, status_code=400)

    params = dict(request.query_params)
    key = endpoint + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    now = asyncio.get_event_loop().time()

    # 1) Global backoff: if Binance told us to back off, fail fast locally.
    if now < _binance_proxy_backoff_until:
        retry_in = int(_binance_proxy_backoff_until - now) + 1
        return JSONResponse(
            {"error": "binance rate limited, backing off", "retry_after": retry_in},
            status_code=429,
            headers={"Retry-After": str(retry_in)},
        )

    # 2) Serve from short TTL cache (dashboard polls are highly repetitive).
    cached = _binance_proxy_cache.get(key)
    if cached and now - cached[0] < _BINANCE_PROXY_TTL.get(endpoint, 10.0):
        return JSONResponse(content=cached[1], status_code=200)

    # 3) Coalesce identical concurrent requests into one upstream call.
    fut = _binance_proxy_inflight.get(key)
    if fut is None:
        fut = asyncio.get_event_loop().create_future()
        _binance_proxy_inflight[key] = fut
        try:
            resp = None
            # data-api.binance.vision is Binance's dedicated public market-data
            # host: keeps dashboard traffic off the IP-weight budget of
            # api.binance.com that the live trading engine depends on.
            for host in ("https://data-api.binance.vision", "https://api.binance.com"):
                try:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        resp = await client.get(f"{host}/api/v3/{endpoint}", params=params)
                    if resp.status_code < 500:
                        break
                except Exception:
                    resp = None
                    continue
            if resp is None:
                raise RuntimeError("all binance hosts unreachable")
            if resp.status_code in (418, 429):
                retry_after = int(resp.headers.get("Retry-After", "60"))
                globals()["_binance_proxy_backoff_until"] = (
                    asyncio.get_event_loop().time() + min(retry_after, 3600)
                )
                fut.set_result((resp.status_code, {"error": "binance rate limited", "retry_after": retry_after}))
            else:
                body = resp.json()
                if resp.status_code == 200:
                    _binance_proxy_cache[key] = (asyncio.get_event_loop().time(), body)
                    if len(_binance_proxy_cache) > 512:
                        oldest = min(_binance_proxy_cache, key=lambda k: _binance_proxy_cache[k][0])
                        _binance_proxy_cache.pop(oldest, None)
                fut.set_result((resp.status_code, body))
        except Exception as exc:
            fut.set_result((502, {"error": f"binance proxy failed: {exc}"}))
        finally:
            _binance_proxy_inflight.pop(key, None)

    status, body = await fut
    return JSONResponse(content=body, status_code=status)


@router.get("/crypto-news")
async def get_crypto_news(limit: int = 20):
    """
    Fetch real crypto news via yfinance for major symbols.
    Used by n8n sentiment pipeline. No API key required.
    Only major coins - no meme coins.
    """
    import yfinance as yf
    from datetime import datetime, timezone

    # Use configured trading symbols (crypto only, no forex)
    symbols_env = os.getenv('TRADING_SYMBOLS', 'BTC-USD,ETH-USD,SOL-USD,BNB-USD,XRP-USD')
    symbols = [s.strip() for s in symbols_env.split(',') if '-USD' in s][:6]
    if not symbols:
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD']

    SYMBOL_MAP = {
        'BTC-USD': 'BTC', 'ETH-USD': 'ETH', 'SOL-USD': 'SOL',
        'BNB-USD': 'BNB', 'XRP-USD': 'XRP', 'ADA-USD': 'ADA',
        'AVAX-USD': 'AVAX', 'DOT-USD': 'DOT', 'LINK-USD': 'LINK',
    }

    seen_ids = set()
    articles = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            crypto_sym = SYMBOL_MAP.get(symbol, symbol.replace('-USD', ''))

            for item in news:
                article_id = item.get('id', '')
                if article_id in seen_ids:
                    continue
                seen_ids.add(article_id)

                content = item.get('content', {})
                title = content.get('title', item.get('title', ''))
                summary = content.get('summary', content.get('description', ''))
                pub_date_str = content.get('pubDate', content.get('displayTime', ''))

                # Parse timestamp
                try:
                    if pub_date_str:
                        dt = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                        published_at = int(dt.timestamp())
                    else:
                        published_at = int(datetime.now(timezone.utc).timestamp())
                except Exception:
                    published_at = int(datetime.now(timezone.utc).timestamp())

                if not title:
                    continue

                # Determine which symbols are mentioned
                all_syms = list(SYMBOL_MAP.values())
                text = f"{title} {summary}".upper()
                mentioned = [s for s in all_syms if s in text]
                if crypto_sym not in mentioned:
                    mentioned.insert(0, crypto_sym)

                articles.append({
                    'id': article_id or f'{symbol}_{published_at}',
                    'title': title,
                    'body': summary,
                    'source': 'yfinance',
                    'published_at': published_at,
                    'categories': 'Crypto',
                    'url': content.get('canonicalUrl', {}).get('url', '') if isinstance(content.get('canonicalUrl'), dict) else '',
                    'symbols': mentioned[:3],
                })
        except Exception as e:
            logger.warning(f"crypto-news error for {symbol}: {e}")

    # Sort by publication time, newest first
    articles.sort(key=lambda x: x['published_at'], reverse=True)
    articles = articles[:limit]

    return {'articles': articles, 'count': len(articles), 'symbols': symbols}


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Trading Routes  (Fincept port)
# ════════════════════════════════════════════════════════════════════════════════


@router.post("/session/init")
async def init_session(request: dict):
    """Initialize a trading session (paper or live)."""
    broker = request.get("broker", "binance_futures")
    mode = request.get("mode", "paper")
    balance = float(request.get("paper_balance", 100_000.0))
    leverage = float(request.get("leverage", 1.0))
    ut = UnifiedTrading()
    session = ut.init_session(broker=broker, mode=mode, paper_balance=balance, leverage=leverage)
    return {
        "broker": session.broker,
        "mode": session.mode,
        "paper_portfolio_id": session.paper_portfolio_id,
    }


@router.get("/session/status")
async def session_status():
    """Get current trading session status."""
    ut = UnifiedTrading()
    sess = ut.get_session()
    if not sess:
        return {"active": False}
    return {
        "active": True,
        "broker": sess.broker,
        "mode": sess.mode,
        "paper_portfolio_id": sess.paper_portfolio_id,
        "sessions": ut.list_sessions(),
    }


@router.post("/paper/order")
async def paper_place_order(request: dict):
    """Place a paper/simulated order and persist to Trade table."""
    from backend.services.trading_mode import paper_starting_balance
    ut = UnifiedTrading()
    ut.init_session(
        "binance_futures",
        mode="paper",
        paper_balance=paper_starting_balance(),
        session_id="paper_manual",
    )
    sym = request.get("symbol", "").upper()
    side_str = request.get("side", "buy").lower()
    px = float(request.get("price", 0) or 0)
    if px <= 0:
        try:
            from backend.services.binance_market_data import binance_market_data
            tick = await binance_market_data.get_ticker_24h(sym)
            px = float((tick or {}).get("lastPrice") or 0)
        except Exception:
            pass

    order = UnifiedOrder(
        symbol=sym,
        side=OrderSide(side_str),
        order_type=OrderType(request.get("order_type", "market").lower()),
        quantity=float(request.get("quantity", 0)),
        price=px,
        stop_loss=float(request.get("stop_loss", 0) or 0),
        take_profit=float(request.get("take_profit", 0) or 0),
    )
    resp = ut.place_order(order, session_id="paper_manual")
    if resp.success:
        db = SessionLocal()
        try:
            direction = "BUY" if side_str == "buy" else "SELL"
            trade = Trade(
                symbol=sym,
                direction=direction,
                quantity=float(resp.filled_qty or order.quantity),
                entry_price=float(resp.filled_price or px),
                status="open",
                strategy="paper_manual",
                binance_order_id=resp.order_id,
                stop_loss=order.stop_loss or None,
                take_profit=order.take_profit or None,
                notes="Paper order placed via UI/API",
            )
            db.add(trade)
            db.commit()
        except Exception as e:
            logger.warning(f"Could not persist paper trade to DB: {e}")
        finally:
            db.close()

    return {
        "success": resp.success,
        "order_id": resp.order_id,
        "message": resp.message,
        "mode": resp.mode,
        "filled_price": resp.filled_price,
        "filled_qty": resp.filled_qty,
    }

@router.post("/order")
async def place_live_order(request: LiveOrderRequest):
    """Place a validated live order and persist its exchange fill."""
    from backend.services.sentry_state import is_trading_allowed
    if not is_trading_allowed():
        raise HTTPException(status_code=400, detail="Trading is currently halted by Sentry.")
    from backend.services.risk_config import get_risk_config
    from backend.services.decision_engine import compute_sl_tp_levels
    from backend.services.trading_loop import trading_loop

    symbol = request.symbol.upper()
    side = request.side
    direction = "BUY" if side == "buy" else "SELL"

    raw_sl = request.stop_loss
    raw_tp = request.take_profit
    stop_loss = float(raw_sl) if raw_sl not in (None, "", 0) else None
    take_profit = float(raw_tp) if raw_tp not in (None, "", 0) else None

    # Manual/workflow orders historically sent no SL/TP → naked hedge legs on Binance.
    if stop_loss is None or take_profit is None:
        try:
            bars = await trading_loop._fetch_bars(symbol)
            if bars and len(bars) >= 15:
                entry = request.price or float(bars[-1]["close"])
                if entry:
                    stop_loss, take_profit = compute_sl_tp_levels(
                        bars, direction, entry, get_risk_config(),
                        signal_sl=stop_loss, signal_tp=take_profit,
                    )
        except Exception as exc:
            logger.warning("Could not compute SL/TP for %s: %s", symbol, exc)

    if stop_loss is None or take_profit is None:
        if get_trading_mode() == TradingMode.PAPER:
            logger.warning(
                "Paper order %s %s proceeding without SL/TP (levels unavailable)",
                direction, symbol,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Refusing live order without stop_loss and take_profit "
                    "(levels missing and bar fetch could not compute them)"
                ),
            )

    ut = UnifiedTrading()
    order = UnifiedOrder(
        symbol=symbol,
        side=OrderSide(side),
        order_type=OrderType(request.order_type),
        quantity=request.quantity,
        price=request.price,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
    resp = ut.place_order(order)
    if not resp.success:
        raise HTTPException(status_code=502, detail=f"Exchange order failed: {resp.message}")

    # Live manual orders must enter the same DB lifecycle as loop orders so
    # reconciliation, protection, risk counts, dashboard, and exits all see
    # the exchange leg. Paper mode has its own persistence engine.
    if resp.mode == "live":
        entry_price = float(resp.filled_price or request.price or 0)
        if entry_price <= 0:
            # The exchange order exists but cannot be managed honestly without
            # its fill. Flatten immediately rather than create an orphan leg.
            close_side = OrderSide.SELL if direction == "BUY" else OrderSide.BUY
            ut.place_order(UnifiedOrder(
                symbol=symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=float(resp.filled_qty or request.quantity),
                reduce_only=True,
            ))
            raise HTTPException(
                status_code=502,
                detail="Exchange fill had no price; emergency close attempted",
            )

        db = SessionLocal()
        try:
            trade = Trade(
                symbol=symbol,
                direction=direction,
                quantity=float(resp.filled_qty or request.quantity),
                entry_price=entry_price,
                filled_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status="filled",
                strategy="manual_api",
                notes="Manual live order recorded from exchange fill",
                binance_order_id=str(resp.order_id or ""),
                exchange="binance_futures",
            )
            db.add(trade)
            db.commit()
            db.refresh(trade)
            trade_id = trade.id
        except Exception as db_error:
            db.rollback()
            # Never leave a filled-but-unrecorded manual position unmanaged.
            close_side = OrderSide.SELL if direction == "BUY" else OrderSide.BUY
            close_resp = ut.place_order(UnifiedOrder(
                symbol=symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=float(resp.filled_qty or request.quantity),
                reduce_only=True,
            ))
            logger.critical(
                "Manual order filled but DB persistence failed; emergency close "
                "success=%s: %s", close_resp.success, db_error,
            )
            raise HTTPException(
                status_code=500,
                detail="Order persistence failed; emergency close attempted",
            )
        finally:
            db.close()
    else:
        trade_id = None

    return {
        "success": resp.success,
        "order_id": resp.order_id,
        "message": resp.message,
        "mode": resp.mode,
        "filled_price": resp.filled_price,
        "filled_qty": resp.filled_qty,
        "trade_id": trade_id,
    }



@router.post("/paper/cancel")
async def paper_cancel_order(request: dict):
    """Cancel a paper order."""
    ut = UnifiedTrading()
    resp = ut.cancel_order(request.get("order_id", ""))
    return {
        "success": resp.success,
        "order_id": resp.order_id,
        "message": resp.message,
    }


@router.get("/paper/portfolio")
async def paper_portfolio():
    """Get current paper portfolio state."""
    ut = UnifiedTrading()
    pf = ut.get_paper_portfolio()
    stats = ut.get_paper_stats()
    return {"portfolio": pf, "stats": stats}


@router.get("/paper/positions")
async def paper_positions():
    """Get open paper positions."""
    ut = UnifiedTrading()
    positions = ut.get_paper_positions()
    return {
        "positions": [
            {
                "symbol": p.symbol,
                "side": p.side,
                "quantity": p.quantity,
                "avg_price": p.avg_price,
            }
            for p in positions
        ],
        "count": len(positions),
    }


@router.get("/paper/orders")
async def paper_orders(status: Optional[str] = Query("")):
    """Get paper orders."""
    ut = UnifiedTrading()
    orders = ut.get_paper_orders(status)
    return {"orders": orders, "count": len(orders)}


@router.get("/paper/stats")
async def paper_stats():
    """Get paper trading statistics."""
    ut = UnifiedTrading()
    return ut.get_paper_stats()


# ═══════════════════════════════════════════════════════════════════════════════
# AI Tool Execution  (Fincept LlmService port)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/ai/agent-trade")
async def ai_agent_trade(request: AgentTradeRequest):
    """
    Let the LLM trade autonomously via tool calls.
    Uses the Fincept-style tool execution loop.
    """
    from backend.services.sentry_state import is_trading_allowed
    if not is_trading_allowed():
        raise HTTPException(status_code=400, detail="Trading is currently halted by Sentry.")
    import os
    from backend.services.llm_tool_loop import LlmToolClient, build_trading_tools
    from backend.services.unified_trading import UnifiedTrading
    from backend.services.binance_market_data import binance_market_data

    prompt = request.prompt
    model = request.model or os.getenv("XAI_MODEL", "grok-beta")
    provider = request.provider

    # Pick API key based on provider
    api_key = ""
    base_url = ""
    if provider == "xai":
        api_key = os.getenv("XAI_API_KEY", "")
        base_url = "https://api.x.ai/v1"
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = "https://api.openai.com/v1"
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        base_url = "https://api.groq.com/openai/v1"
    elif provider == "ollama":
        api_key = "ollama"
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"

    if not api_key or api_key == "your_xai_api_key_here":
        return {"error": f"API key not configured for {provider}"}

    ut = UnifiedTrading()
    # The tool is explicitly paper-only. Never let the LLM inherit the
    # process-wide default session, which is live in production.
    paper_session_id = "ai-agent-paper"
    if not ut.get_session(paper_session_id):
        ut.init_session(
            "binance_futures",
            mode="paper",
            paper_balance=100_000.0,
            session_id=paper_session_id,
        )
    tools = build_trading_tools(
        ut, binance_market_data, paper_session_id=paper_session_id,
    )

    client = LlmToolClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        provider=provider,
    )
    client.tools = tools

    system = (
        "You are an autonomous trading agent. You have access to tools that let you "
        "inspect market prices, your portfolio, and place/cancel paper orders. "
        "Analyze carefully, then act decisively. Always report your reasoning."
    )

    result = client.chat(
        user_message=prompt,
        system_prompt=system,
        max_tool_rounds=5,
    )
    return {
        "response": result["content"],
        "tool_calls_used": result.get("tool_calls_used", False),
        "tool_rounds": result.get("tool_rounds", 0),
        "usage": result.get("usage", {}),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DataHub Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/datahub/topics")
async def datahub_topics():
    """List active DataHub topics and subscriber counts."""
    from backend.services.data_hub import DataHub
    return DataHub().stats()


@router.get("/datahub/peek/{topic}")
async def datahub_peek(topic: str):
    """Peek cached value for a DataHub topic."""
    from backend.services.data_hub import DataHub
    val = DataHub().peek(topic)
    return {"topic": topic, "value": val, "cached": val is not None}


# ═══════════════════════════════════════════════════════════════════════════════
# B: SSE Real-time Stream
# ═══════════════════════════════════════════════════════════════════════════════



@router.get("/stream")
async def event_stream(topics: str = ""):
    """
    SSE endpoint for real-time DataHub events.
    Usage: /trading/stream?topics=market:quote:BTCUSDT,paper:fill
    """
    from backend.services.data_hub import DataHub

    requested = [t.strip() for t in topics.split(",") if t.strip()] if topics else []
    queue: asyncio.Queue[dict] = asyncio.Queue()

    def on_event(value: any):
        try:
            queue.put_nowait({"topic": "unknown", "data": value})
        except Exception:
            pass

    # Subscribe to all requested topics
    hub = DataHub()
    for t in requested:
        hub.subscribe(t, lambda v, topic=t: queue.put_nowait({"topic": topic, "data": v}), immediate=False)

    async def generator():
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    # Keep the connection alive instead of ending the stream.
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(generator(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════════════════════════
# ─── Price helper (used by ActiveTradeCard for live P&L) ─────────────────────

@router.get("/price")
async def get_price(symbol: str = "BTCUSDT"):
    """Return current mark/last price for a symbol."""
    sym = symbol.upper().strip()
    if _is_ctrader_symbol(sym):
        mark = ctrader_broker.get_mark_price(sym)
        if not mark or mark <= 0:
            ctrader_broker.ensure_spot_quotes([sym])
            mark = ctrader_broker.get_mark_price(sym)
        if mark and mark > 0:
            return {
                "symbol": sym,
                "price": mark,
                "change24h": 0.0,
                "source": "ctrader",
            }
        return {"symbol": sym, "price": 0.0, "change24h": 0.0, "source": "ctrader"}

    from backend.services.binance_market_data import binance_market_data
    try:
        ticker = await binance_market_data.get_ticker_24h(sym)
        if ticker:
            return {
                "symbol": sym,
                "price": ticker.get("lastPrice", 0.0),
                "change24h": ticker.get("priceChangePercent", 0.0),
                "source": "binance",
            }
    except Exception:
        pass
    return {"symbol": sym, "price": 0.0, "change24h": 0.0, "source": "binance"}


@router.get("/account/summary")
async def get_account_summary():
    """Return live account equity and balance for workflow engine."""
    try:
        pf = await get_portfolio()
        equity = float(pf.get("equity", 100000.0))
        available = float(pf.get("available", equity))
        db = SessionLocal()
        try:
            from sqlalchemy import func
            today = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0,
            )
            daily_pnl = db.query(func.sum(Trade.pnl)).filter(
                Trade.status == "closed",
                Trade.closed_at >= today,
                Trade.pnl.isnot(None),
            ).scalar() or 0.0
        finally:
            db.close()
        return {
            "equity": round(equity, 4),
            "available_balance": round(available, 4),
            "daily_pnl": round(float(daily_pnl), 4),
            "currency": "USDT",
            "source": "binance_futures",
        }
    except Exception:
        raise HTTPException(status_code=503, detail="Account data unavailable")


# ═══════════════════════════════════════════════════════════════════════════════
# C: Multi-Account Routes
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/sessions")
async def list_sessions():
    """List all active trading sessions."""
    ut = UnifiedTrading()
    return {"sessions": ut.list_sessions()}


@router.post("/session/switch")
async def switch_session(request: dict):
    """Switch default session by ID."""
    ut = UnifiedTrading()
    sid = request.get("session_id", "")
    ut.set_default_session(sid)
    return {"message": "Default session switched", "session_id": sid}


# ═══════════════════════════════════════════════════════════════════════════════
# D: Opinion Layer Routes
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/opinion/analyze")
async def analyze_opinion(request: dict):
    """
    Run the full Opinion Layer multi-agent analysis on a symbol.
    Body: {"symbol": "BTCUSDT", "bars": [...], "include_kronos": true, "include_social": true}
    """
    from backend.services.opinion_layer import analyze_symbol as analyze_opinion
    symbol = request.get("symbol", "")
    bars = request.get("bars", [])
    include_kronos = request.get("include_kronos", True)
    include_social = request.get("include_social", True)
    include_alerts = request.get("include_alerts", True)
    include_personas = request.get("include_personas", True)

    if not symbol or not bars:
        return {"error": "symbol and bars required"}

    opinion = await analyze_opinion(
        symbol=symbol,
        bars=bars,
        include_kronos=include_kronos,
        include_social=include_social,
        include_alerts=include_alerts,
        include_personas=include_personas,
    )
    from backend.services.multi_asset_bars import classify_symbol
    asset_class = classify_symbol(symbol)
    return {
        "symbol": opinion.symbol,
        "asset_class": asset_class,
        "direction": opinion.direction,
        "confidence": opinion.confidence,
        "reasoning": opinion.reasoning,
        "agent_opinions": [
            {"agent": op.agent, "signal": op.signal, "confidence": op.confidence, "reasoning": op.reasoning}
            for op in opinion.agent_opinions
        ],
        "kronos": opinion.kronos,
        "social": opinion.social,
        "alerts": opinion.alerts,
        "timestamp": opinion.timestamp,
    }


@router.post("/opinion/weights")
async def update_opinion_weights(request: dict):
    """
    Dynamically adjust agent voting weights in the Opinion Layer.
    Body: {"weights": {"technical_analyst": 0.35, "kronos_foundation": 0.25, ...}}
    """
    from backend.services.opinion_layer import register_agent_weight
    weights = request.get("weights", {})
    for agent, weight in weights.items():
        register_agent_weight(agent, weight)
    return {"message": "Weights updated", "weights": weights}


@router.get("/opinion/weights")
async def get_opinion_weights():
    """Get current agent voting weights."""
    from backend.services import opinion_layer as ol
    return {"weights": ol._AGENT_WEIGHTS}




# ── Trade Memory (Track C): semantic recall of past trades ───────────────────

@router.get("/trade-memory/status")
async def trade_memory_status():
    """Status of the semantic trade-memory Qdrant collection."""
    from backend.services.trade_memory import trade_memory
    return await trade_memory.status()


@router.post("/trade-memory/backfill")
async def trade_memory_backfill(limit: int = Query(1000, ge=1, le=10000)):
    """Vectorise existing closed trades from SQL into Qdrant (idempotent)."""
    from backend.services.trade_memory import trade_memory
    return await trade_memory.backfill_from_sql(limit=limit)


@router.post("/trade-memory/recall")
async def trade_memory_recall(request: Dict[str, Any]):
    """Debug: recall similar past setups for an arbitrary market context.

    Body: {"context": {...feature keys...}, "symbol": "BTCUSDT",
           "same_symbol_only": false, "limit": 8}
    """
    from backend.services.trade_memory import trade_memory
    ctx = request.get("context", {}) or {}
    result = await trade_memory.recall_similar(
        ctx,
        symbol=request.get("symbol"),
        limit=request.get("limit"),
        same_symbol_only=bool(request.get("same_symbol_only", False)),
    )
    return result.to_dict()


# ── Strategy Skills (skill miner): learned, named strategies ─────────────────

@router.get("/skills/status")
async def skills_status():
    """Skill miner status: counts + config."""
    from backend.services.skill_miner import skill_miner
    return skill_miner.status()


@router.get("/skills")
async def skills_list(active_only: bool = Query(True), limit: int = Query(50, ge=1, le=500)):
    """List learned strategy skills (the leaderboard)."""
    from backend.services.skill_miner import skill_miner
    skills = await asyncio.to_thread(skill_miner.list_skills, active_only, limit)
    return {"count": len(skills), "skills": skills}


@router.get("/skills/leaderboard")
async def skills_leaderboard(limit: int = Query(10, ge=1, le=100)):
    """Top skills by edge score — compact leaderboard view."""
    from backend.services.skill_miner import skill_miner
    skills = await asyncio.to_thread(skill_miner.list_skills, True, limit)
    board = [
        {
            "rank": i + 1,
            "name": s["name"],
            "direction": s["direction"],
            "edge_score": s["edge_score"],
            "win_rate": s["win_rate"],
            "avg_pnl": s["avg_pnl"],
            "sample_count": s["sample_count"],
            "symbols": s["symbols"],
        }
        for i, s in enumerate(skills)
    ]
    return {"count": len(board), "leaderboard": board}


@router.post("/skills/mine")
async def skills_mine(limit: Optional[int] = Query(None, ge=1, le=20000)):
    """Trigger a skill-mining pass over closed-trade history (idempotent)."""
    from backend.services.skill_miner import skill_miner
    return await asyncio.to_thread(skill_miner.mine_and_store, limit)


@router.post("/skills/match")
async def skills_match(request: Dict[str, Any]):
    """Debug: match an arbitrary market context to the best learned skill.

    Body: {"context": {...feature keys...}}
    """
    from backend.services.skill_miner import skill_miner
    ctx = request.get("context", {}) or {}
    skill = await asyncio.to_thread(skill_miner.match_skill, ctx)
    return {"matched": skill is not None, "skill": skill}
