"""JEV Phase 2 pipeline: ingest stored context, evaluate, optionally paper-fill.

off/shadow never call the paper engine. live is stubbed and blocked in this PR.
Paper fills go through UnifiedTrading's existing paper session — not an external
https://quantumtrade.local execute URL.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.database.models import (
    JevMarketScan,
    JevNewsSentimentLog,
    JevPipelineEval,
    JevPipelineTrade,
    PortfolioSnapshot,
    Trade,
)
from backend.jev.config import (
    jev_execution_mode,
    jev_may_call_execute,
    jev_paper_quantity,
    jev_trade_threshold,
)
from backend.jev.service import evaluate_symbol
from backend.services.risk_config import get_risk_config
from backend.services.risk_guard import RiskBreach, enforce_risk_limits, latest_snapshot_for_risk
from backend.services.sentry_state import is_trading_allowed
from backend.services.trading_mode import paper_starting_balance
from backend.services.unified_trading import OrderSide, OrderType, UnifiedOrder, UnifiedTrading

logger = logging.getLogger(__name__)

ExecuteFn = Callable[..., dict[str, Any]]

_BULLISH = {"STRONG_BUY", "BUY", "LONG", "BULLISH"}
_BEARISH = {"STRONG_SELL", "SELL", "SHORT", "BEARISH"}
PAPER_SESSION_ID = "jev_pipeline_paper"
PAPER_PORTFOLIO_NAME = "jev_pipeline_paper"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def normalize_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper()


def ingest_market_scan(db: Session, payload: dict[str, Any]) -> JevMarketScan:
    symbol = normalize_symbol(str(payload.get("symbol") or ""))
    if not symbol:
        raise ValueError("symbol is required")
    scan_ts = parse_datetime(payload.get("scan_timestamp") if isinstance(payload.get("scan_timestamp"), str) else None) or utc_now()
    timeframe = str(payload.get("timeframe") or "M5")[:10]
    row = JevMarketScan(
        scan_timestamp=scan_ts,
        symbol=symbol,
        timeframe=timeframe,
        signal_direction=(str(payload.get("signal_direction")).upper()[:10] if payload.get("signal_direction") else None),
        confidence=_optional_float(payload.get("confidence")),
        signal_strength=_optional_float(payload.get("signal_strength")),
        indicators=payload.get("indicators") if isinstance(payload.get("indicators"), dict) else {},
        price_data=payload.get("price_data") if isinstance(payload.get("price_data"), dict) else {},
        volume_data=payload.get("volume_data") if isinstance(payload.get("volume_data"), dict) else {},
        scan_source=str(payload.get("scan_source") or "n8n_market_scanner")[:50],
        status="pending",
    )
    db.add(row)
    try:
        db.commit()
        db.refresh(row)
        return row
    except IntegrityError:
        db.rollback()
        existing = (
            db.query(JevMarketScan)
            .filter(
                JevMarketScan.symbol == symbol,
                JevMarketScan.timeframe == timeframe,
                JevMarketScan.scan_timestamp == scan_ts,
            )
            .first()
        )
        if existing is None:
            raise
        return existing


def ingest_news_sentiment(db: Session, payload: dict[str, Any]) -> JevNewsSentimentLog:
    source = str(payload.get("source") or "").strip()
    if not source:
        raise ValueError("source is required")
    row = JevNewsSentimentLog(
        log_timestamp=utc_now(),
        source=source[:50],
        headline=payload.get("headline"),
        url=payload.get("url"),
        published_at=parse_datetime(payload.get("published_at") if isinstance(payload.get("published_at"), str) else None),
        sentiment_score=_optional_float(payload.get("sentiment_score")),
        sentiment_label=(str(payload.get("sentiment_label"))[:20] if payload.get("sentiment_label") else None),
        impact_rating=(str(payload.get("impact_rating"))[:10] if payload.get("impact_rating") else None),
        categories=payload.get("categories") if isinstance(payload.get("categories"), (dict, list)) else {},
        entities=payload.get("entities") if isinstance(payload.get("entities"), (dict, list)) else {},
        full_data=payload.get("full_data") if isinstance(payload.get("full_data"), dict) else {},
        market_scan_id=payload.get("market_scan_id") if isinstance(payload.get("market_scan_id"), int) else None,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def recent_news_for_symbol(db: Session, symbol: str, days: int = 3, limit: int = 40) -> list[JevNewsSentimentLog]:
    needle = normalize_symbol(symbol)
    cutoff = utc_now() - timedelta(days=max(1, min(days, 14)))
    rows = (
        db.query(JevNewsSentimentLog)
        .filter(JevNewsSentimentLog.log_timestamp >= cutoff)
        .order_by(JevNewsSentimentLog.log_timestamp.desc())
        .limit(200)
        .all()
    )
    matched: list[JevNewsSentimentLog] = []
    for row in rows:
        blob = " ".join(
            str(part or "")
            for part in (row.headline, row.url, row.entities, row.full_data, row.categories)
        ).upper()
        if needle in blob:
            matched.append(row)
        if len(matched) >= limit:
            break
    return matched


def pending_scans(db: Session, limit: int = 1) -> list[JevMarketScan]:
    evaluated_ids = db.query(JevPipelineEval.source_scan_id).filter(JevPipelineEval.source_scan_id.isnot(None))
    return (
        db.query(JevMarketScan)
        .filter(~JevMarketScan.id.in_(evaluated_ids))
        .order_by(JevMarketScan.scan_timestamp.asc())
        .limit(max(1, min(limit, 5)))
        .all()
    )


def bars_from_price_data(price_data: dict[str, Any] | None) -> list[dict]:
    if not isinstance(price_data, dict):
        return []
    raw = price_data.get("ohlcv") or price_data.get("bars") or price_data.get("klines")
    if not isinstance(raw, list):
        return []
    bars: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            bars.append({
                "open": float(item.get("open") or item.get("o") or 0),
                "high": float(item.get("high") or item.get("h") or 0),
                "low": float(item.get("low") or item.get("l") or 0),
                "close": float(item.get("close") or item.get("c") or 0),
                "volume": float(item.get("volume") or item.get("v") or 0),
            })
        except (TypeError, ValueError):
            continue
    return bars


def metrics_from_indicators(indicators: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(indicators, dict):
        return {}
    metrics: dict[str, Any] = {}
    for key in ("funding_rate", "open_interest", "oi_change_pct_1h"):
        if key in indicators:
            metrics[key] = indicators[key]
    return metrics


def map_pipeline_side(evaluation: dict[str, Any], fallback_direction: str | None = None) -> str | None:
    """Side comes from Jev only. Scanner LONG/SHORT must not invent a fill."""
    _ = fallback_direction
    if evaluation.get("conflict") or evaluation.get("vetoed"):
        return None
    action = str(evaluation.get("action") or "").upper()
    signal = str(evaluation.get("signal") or "").upper()
    if action in _BULLISH or signal in _BULLISH:
        if action in _BEARISH or signal in _BEARISH:
            return None
        return "BUY"
    if action in _BEARISH or signal in _BEARISH:
        return "SELL"
    return None


def decide_execution(
    *,
    evaluation: dict[str, Any],
    side: str | None,
    confidence: float,
    execute_fn: ExecuteFn | None = None,
) -> dict[str, Any]:
    """Apply mode + risk + confidence gates. off/shadow never call execute_fn."""
    mode = jev_execution_mode()
    threshold = jev_trade_threshold()
    vetoed = bool(evaluation.get("vetoed"))
    conflict = bool(evaluation.get("conflict"))
    status = str(evaluation.get("status") or "")
    directional = side in {"BUY", "SELL"}
    confident = confidence >= threshold
    eligible = status == "ok" and directional and confident and not vetoed and not conflict
    base = {
        "execution_mode": mode,
        "called_execute": False,
        "would_have_executed": False,
        "executed": False,
        "threshold": threshold,
        "eligible": eligible,
        "side": side,
        "confidence": confidence,
    }
    if not eligible:
        reason = _ineligible_reason(status, directional, confident, vetoed, conflict, threshold)
        return {**base, "execution_decision": reason}

    if mode == "off":
        return {**base, "execution_decision": "logged_only"}
    if mode == "shadow":
        return {**base, "would_have_executed": True, "execution_decision": "shadow_recorded"}
    if mode == "live":
        return {**base, "would_have_executed": True, "execution_decision": "blocked_live"}
    if mode != "paper":
        return {**base, "execution_decision": "logged_only"}

    if not jev_may_call_execute():
        return {**base, "execution_decision": "blocked_mode"}
    if not is_trading_allowed():
        return {**base, "would_have_executed": True, "execution_decision": "blocked_sentry"}

    if execute_fn is None:
        return {**base, "would_have_executed": True, "execution_decision": "paper_skipped_no_executor"}

    result = execute_fn(side=side, evaluation=evaluation)
    success = bool(result.get("success"))
    return {
        **base,
        "called_execute": True,
        "would_have_executed": True,
        "executed": success,
        "execution_decision": "paper_filled" if success else "paper_failed",
        "paper": result,
    }


def check_paper_risk_gates(db: Session) -> None:
    """Fail closed if risk limits are breached. Any unexpected error also blocks."""
    open_trades = db.query(Trade).filter(Trade.status == "open").all()
    snapshot: PortfolioSnapshot | None = latest_snapshot_for_risk(db)
    enforce_risk_limits(db, get_risk_config(), open_trades, snapshot)


def last_price_from_context(
    evaluation: dict[str, Any] | None = None,
    price_data: dict[str, Any] | None = None,
) -> float:
    market = (evaluation or {}).get("market") if isinstance(evaluation, dict) else None
    if isinstance(market, dict):
        close = _optional_float(market.get("last_close") or market.get("close"))
        if close and close > 0:
            return close
    bars = bars_from_price_data(price_data if isinstance(price_data, dict) else None)
    if bars:
        close = _optional_float(bars[-1].get("close"))
        if close and close > 0:
            return close
    if isinstance(price_data, dict):
        for key in ("last", "last_price", "close", "price"):
            close = _optional_float(price_data.get(key))
            if close and close > 0:
                return close
    return 0.0


def execute_paper_order(
    *,
    symbol: str,
    side: str,
    quantity: float | None = None,
    price: float = 0.0,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    db: Session | None = None,
    evaluation: dict[str, Any] | None = None,
    price_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Place a paper/sandbox fill via the platform engine. Never hits a live broker."""
    if db is not None:
        try:
            check_paper_risk_gates(db)
        except RiskBreach as exc:
            logger.warning("JEV paper execute blocked by risk guard: %s", exc)
            return {"success": False, "blocked": True, "reason": f"risk_guard:{exc}", "order_id": ""}
        except Exception as exc:
            logger.warning("JEV paper execute fail-closed on risk check: %s", exc)
            return {"success": False, "blocked": True, "reason": f"risk_guard_error:{exc}", "order_id": ""}

    mark = float(price or 0.0)
    if mark <= 0:
        mark = last_price_from_context(evaluation, price_data)
    if mark <= 0:
        return {"success": False, "blocked": True, "reason": "no_mark_price", "order_id": ""}

    qty = float(quantity if quantity is not None else jev_paper_quantity())
    ut = UnifiedTrading()
    session = ut.init_session(
        "binance_futures",
        mode="paper",
        paper_balance=paper_starting_balance(),
        session_id=PAPER_SESSION_ID,
    )
    dedicated = ut._paper.find_or_create_portfolio(
        name=PAPER_PORTFOLIO_NAME,
        balance=paper_starting_balance(),
        exchange="binance_futures",
    )
    session.paper_portfolio_id = dedicated
    order = UnifiedOrder(
        symbol=normalize_symbol(symbol),
        side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=qty,
        price=mark,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
    resp = ut.place_order(order, session_id=PAPER_SESSION_ID)
    return {
        "success": bool(resp.success),
        "order_id": resp.order_id,
        "filled_price": resp.filled_price,
        "filled_qty": resp.filled_qty,
        "message": resp.message,
        "broker": "paper",
        "portfolio": PAPER_PORTFOLIO_NAME,
    }


async def evaluate_and_log(
    db: Session,
    *,
    symbol: str,
    timeframe: str = "1H",
    source_scan: JevMarketScan | None = None,
    market_context: list[dict[str, Any]] | None = None,
    news_context: list[dict[str, Any]] | None = None,
    fallback_direction: str | None = None,
    fallback_confidence: float | None = None,
    client: Any | None = None,
    fetch_bars: bool = True,
    execute_fn: ExecuteFn | None = None,
) -> dict[str, Any]:
    scan_indicators = source_scan.indicators if source_scan is not None else {}
    scan_price = source_scan.price_data if source_scan is not None else {}
    bars = bars_from_price_data(scan_price if isinstance(scan_price, dict) else {})
    metrics = metrics_from_indicators(scan_indicators if isinstance(scan_indicators, dict) else {})
    evaluation = await evaluate_symbol(
        symbol,
        bars=bars or None,
        metrics=metrics or None,
        include_social=False,
        client=client,
        fetch_bars=bool(fetch_bars and not bars),
    )
    news = news_context if news_context is not None else []
    market = market_context if market_context is not None else []
    if source_scan is not None and not market:
        market = [{"indicators": source_scan.indicators, "price_data": source_scan.price_data}]

    raw_confidence = evaluation.get("confidence")
    if raw_confidence is None:
        confidence = float(fallback_confidence or 0.0)
    else:
        confidence = float(raw_confidence)
    side = map_pipeline_side(evaluation)
    scan_price = source_scan.price_data if source_scan is not None and isinstance(source_scan.price_data, dict) else {}
    decision = decide_execution(
        evaluation=evaluation,
        side=side,
        confidence=confidence,
        execute_fn=None if not jev_may_call_execute() else execute_fn,
    )
    if decision["execution_mode"] == "paper" and decision["eligible"] and execute_fn is None and jev_may_call_execute():
        decision = decide_execution(
            evaluation=evaluation,
            side=side,
            confidence=confidence,
            execute_fn=lambda **kwargs: execute_paper_order(
                symbol=symbol,
                side=str(kwargs.get("side") or side),
                db=db,
                evaluation=evaluation,
                price_data=scan_price,
            ),
        )

    now = utc_now()
    row = JevPipelineEval(
        log_timestamp=now,
        symbol=normalize_symbol(symbol),
        timeframe=(timeframe or "1H")[:10],
        jev_signal=side or str(evaluation.get("action") or evaluation.get("signal") or "HOLD"),
        jev_confidence=confidence,
        jev_strength=_optional_float((evaluation.get("answers") or {}).get("trade_confidence") if isinstance(evaluation.get("answers"), dict) else None),
        probabilities=(evaluation.get("answers") or {}).get("direction_probabilities") if isinstance(evaluation.get("answers"), dict) else {},
        checks={
            "vetoed": bool(evaluation.get("vetoed")),
            "conflict": bool(evaluation.get("conflict")),
            "status": evaluation.get("status"),
            "scanner_direction": fallback_direction,
        },
        market_context={"recent_scans": market},
        news_sentiment_context={"recent_news": news},
        combined_context={"market": market, "news": news},
        combined_evaluation=evaluation,
        outcome_prediction=_prediction_from_eval(evaluation, decision),
        outcome_probability=confidence,
        calibration_status="display_only",
        execution_mode=decision["execution_mode"],
        execution_decision=decision["execution_decision"],
        would_have_executed=bool(decision["would_have_executed"]),
        executed=bool(decision["executed"]),
        source_scan_id=source_scan.id if source_scan is not None else None,
        evaluated=True,
        evaluated_at=now,
        notes=str(evaluation.get("reason") or decision["execution_decision"]),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    paper_meta = decision.get("paper") if isinstance(decision.get("paper"), dict) else {}
    if decision["would_have_executed"] or decision["executed"]:
        trade = JevPipelineTrade(
            symbol=normalize_symbol(symbol),
            direction=side or "HOLD",
            quantity=jev_paper_quantity(),
            entry_price=_optional_float(paper_meta.get("filled_price")),
            source="jev_pipeline",
            mode="paper" if decision["executed"] else "shadow",
            source_scan_id=source_scan.id if source_scan is not None else None,
            jev_eval_id=row.id,
            outcome="open" if decision["executed"] else "shadow",
            outcome_reason=decision["execution_decision"],
            notes=str(paper_meta.get("message") or decision["execution_decision"]),
        )
        db.add(trade)
        db.commit()
        db.refresh(trade)
        row.execution_id = trade.id
        row.quantum_trade_id = str(paper_meta.get("order_id") or "") or None
        db.commit()
        db.refresh(row)

    if source_scan is not None:
        source_scan.status = "evaluated"
        db.commit()

    return {
        "status": "evaluated",
        "eval_id": row.id,
        "symbol": row.symbol,
        "timeframe": row.timeframe,
        "prediction": row.outcome_prediction,
        "confidence": row.jev_confidence,
        "jev_signal": row.jev_signal,
        "executed": row.executed,
        "would_have_executed": row.would_have_executed,
        "execution_mode": row.execution_mode,
        "execution_decision": row.execution_decision,
        "called_execute": bool(decision.get("called_execute")),
        "trade_id": row.quantum_trade_id,
        "pipeline_trade_id": row.execution_id,
        "advisory": True,
        "sizing_allowed": False,
        "context_used": {"market_samples": len(market), "news_samples": len(news)},
        "evaluation": {
            "status": evaluation.get("status"),
            "signal": evaluation.get("signal"),
            "action": evaluation.get("action"),
            "reason": evaluation.get("reason"),
            "vetoed": evaluation.get("vetoed"),
        },
    }


async def orchestrate_scan(
    db: Session,
    *,
    limit: int = 1,
    client: Any | None = None,
    fetch_bars: bool = True,
    execute_fn: ExecuteFn | None = None,
) -> dict[str, Any]:
    scans = pending_scans(db, limit=limit)
    if not scans:
        return {
            "status": "idle",
            "reason": "no pending scans",
            "executed": False,
            "execution_mode": jev_execution_mode(),
            "processed": 0,
            "results": [],
        }
    results: list[dict[str, Any]] = []
    for scan in scans:
        news_rows = recent_news_for_symbol(db, scan.symbol, days=3, limit=20)
        news_ctx = [
            {
                "headline": row.headline,
                "sentiment": row.sentiment_score,
                "impact": row.impact_rating,
                "published_at": row.published_at.isoformat() if row.published_at else None,
            }
            for row in news_rows
        ]
        result = await evaluate_and_log(
            db,
            symbol=scan.symbol,
            timeframe=scan.timeframe,
            source_scan=scan,
            news_context=news_ctx,
            fallback_direction=scan.signal_direction,
            fallback_confidence=scan.confidence,
            client=client,
            fetch_bars=fetch_bars,
            execute_fn=execute_fn,
        )
        result["scan_id"] = scan.id
        results.append(result)
    payload = dict(results[0])
    payload["processed"] = len(results)
    payload["results"] = results
    return payload


def calibration_snapshot(db: Session) -> dict[str, Any]:
    """SQLite-friendly stand-in for the zip's v_jev_calibration view."""
    rows = (
        db.query(
            JevPipelineEval.symbol,
            func.count(JevPipelineEval.id),
            func.avg(JevPipelineEval.jev_confidence),
        )
        .filter(JevPipelineEval.evaluated.is_(True))
        .group_by(JevPipelineEval.symbol)
        .all()
    )
    symbols: list[dict[str, Any]] = []
    for symbol, eval_count, avg_conf in rows:
        trades = (
            db.query(JevPipelineTrade)
            .filter(JevPipelineTrade.symbol == symbol, JevPipelineTrade.outcome != "open")
            .all()
        )
        wins = sum(1 for trade in trades if trade.outcome == "win")
        losses = sum(1 for trade in trades if trade.outcome == "loss")
        scratches = sum(1 for trade in trades if trade.outcome in {"scratch", "shadow"})
        closed = [trade for trade in trades if trade.outcome in {"win", "loss", "scratch"}]
        actual = (sum(1 for trade in closed if trade.outcome in {"win", "scratch"}) / len(closed)) if closed else None
        predicted = float(avg_conf or 0.0)
        if actual is None:
            status = "insufficient_outcomes"
        elif predicted > 0.70 and actual < 0.60:
            status = "overconfident"
        elif predicted < 0.45 and actual > 0.60:
            status = "underconfident"
        else:
            status = "well_calibrated"
        symbols.append({
            "symbol": symbol,
            "eval_count": int(eval_count or 0),
            "wins": wins,
            "losses": losses,
            "scratches": scratches,
            "avg_predicted_confidence": round(predicted, 4),
            "actual_win_rate": None if actual is None else round(actual, 4),
            "calibration_status": status,
        })
    return {
        "symbols": symbols,
        "sizing_allowed": False,
        "execution_mode": jev_execution_mode(),
        "note": (
            "Calibration is display-only. Confidence is not a live sizing input. "
            "Do not set JEV_EXECUTION_MODE=live without owner confirm."
        ),
    }


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ineligible_reason(
    status: str,
    directional: bool,
    confident: bool,
    vetoed: bool,
    conflict: bool,
    threshold: float,
) -> str:
    if status != "ok":
        return "no_signal"
    if vetoed:
        return "vetoed"
    if conflict:
        return "conflict"
    if not directional:
        return "no_direction"
    if not confident:
        return f"below_threshold_{threshold:.2f}"
    return "ineligible"


def _prediction_from_eval(evaluation: dict[str, Any], decision: dict[str, Any]) -> str:
    if evaluation.get("status") != "ok" or evaluation.get("vetoed"):
        return "NO_TRADE"
    if not decision.get("eligible"):
        return "NO_TRADE"
    return "PENDING"
