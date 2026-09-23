from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, JSON, ForeignKey, Float, UniqueConstraint, Index
from sqlalchemy.sql import func
from .connection import Base


class HedgeFundFlow(Base):
    """Table to store React Flow configurations (nodes, edges, viewport)"""
    __tablename__ = "hedge_fund_flows"
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Flow metadata
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # React Flow state
    nodes = Column(JSON, nullable=False)  # Store React Flow nodes as JSON
    edges = Column(JSON, nullable=False)  # Store React Flow edges as JSON
    viewport = Column(JSON, nullable=True)  # Store viewport state (zoom, x, y)
    data = Column(JSON, nullable=True)  # Store node internal states (tickers, models, etc.)
    
    # Additional metadata
    is_template = Column(Boolean, default=False)  # Mark as template for reuse
    tags = Column(JSON, nullable=True)  # Store tags for categorization


class HedgeFundFlowRun(Base):
    """Table to track individual execution runs of a hedge fund flow"""
    __tablename__ = "hedge_fund_flow_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    flow_id = Column(Integer, ForeignKey("hedge_fund_flows.id"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Run execution tracking
    status = Column(String(50), nullable=False, default="IDLE")  # IDLE, IN_PROGRESS, COMPLETE, ERROR
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Run configuration
    trading_mode = Column(String(50), nullable=False, default="one-time")  # one-time, continuous, advisory
    schedule = Column(String(50), nullable=True)  # hourly, daily, weekly (for continuous mode)
    duration = Column(String(50), nullable=True)  # 1day, 1week, 1month (for continuous mode)
    
    # Run data
    request_data = Column(JSON, nullable=True)  # Store the request parameters (tickers, agents, models, etc.)
    initial_portfolio = Column(JSON, nullable=True)  # Store initial portfolio state
    final_portfolio = Column(JSON, nullable=True)  # Store final portfolio state
    results = Column(JSON, nullable=True)  # Store the output/results from the run
    error_message = Column(Text, nullable=True)  # Store error details if run failed
    
    # Metadata
    run_number = Column(Integer, nullable=False, default=1)  # Sequential run number for this flow


class HedgeFundFlowRunCycle(Base):
    """Individual analysis cycles within a trading session"""
    __tablename__ = "hedge_fund_flow_run_cycles"
    
    id = Column(Integer, primary_key=True, index=True)
    flow_run_id = Column(Integer, ForeignKey("hedge_fund_flow_runs.id"), nullable=False, index=True)
    cycle_number = Column(Integer, nullable=False)  # 1, 2, 3, etc. within the run
    
    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Analysis results
    analyst_signals = Column(JSON, nullable=True)  # All agent decisions/signals
    trading_decisions = Column(JSON, nullable=True)  # Portfolio manager decisions
    executed_trades = Column(JSON, nullable=True)  # Actual trades executed (paper trading)
    
    # Portfolio state after this cycle
    portfolio_snapshot = Column(JSON, nullable=True)  # Cash, positions, performance metrics
    
    # Performance metrics for this cycle
    performance_metrics = Column(JSON, nullable=True)  # Returns, sharpe ratio, etc.
    
    # Execution tracking
    status = Column(String(50), nullable=False, default="IN_PROGRESS")  # IN_PROGRESS, COMPLETED, ERROR
    error_message = Column(Text, nullable=True)  # Store error details if cycle failed
    
    # Cost tracking
    llm_calls_count = Column(Integer, nullable=True, default=0)  # Number of LLM calls made
    api_calls_count = Column(Integer, nullable=True, default=0)  # Number of financial API calls made
    estimated_cost = Column(String(20), nullable=True)  # Estimated cost in USD
    
    # Metadata
    trigger_reason = Column(String(100), nullable=True)  # scheduled, manual, market_event, etc.
    market_conditions = Column(JSON, nullable=True)  # Market data snapshot at cycle start


class ApiKey(Base):
    """Table to store API keys for various services"""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # API key details
    provider = Column(String(100), nullable=False, unique=True, index=True)  # e.g., "ANTHROPIC_API_KEY"
    key_value = Column(Text, nullable=False)  # The actual API key (encrypted in production)
    is_active = Column(Boolean, default=True)  # Enable/disable without deletion
    
    # Optional metadata
    description = Column(Text, nullable=True)  # Human-readable description
    last_used = Column(DateTime(timezone=True), nullable=True)  # Track usage



class TradingSignal(Base):
    """Generated trading signals from strategy analysis"""
    __tablename__ = "trading_signals"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    symbol = Column(String(20), nullable=False, index=True)
    strategy = Column(String(50), nullable=False)
    direction = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float, nullable=False, default=0.0)
    entry_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, default="pending")  # pending, executed, expired, rejected
    reasoning = Column(Text, nullable=True)
    ai_analysis = Column(Text, nullable=True)  # JSON string of AI analysis steps


class Trade(Base):
    """Executed trades (paper or live)"""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(String(10), nullable=False)  # BUY, SELL
    quantity = Column(Float, nullable=False, default=0.0)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, default="open")  # open, closed
    pnl = Column(Float, nullable=True, default=0.0)
    strategy = Column(String(50), nullable=True)
    notes = Column(Text, nullable=True)
    signal_id = Column(Integer, nullable=True)
    binance_order_id = Column(String(50), nullable=True, index=True)
    exchange = Column(String(20), nullable=True, default='binance_futures')
    filled_price = Column(Float, nullable=True)
    broker = Column(String(30), nullable=True, default='binance_futures', index=True)
    broker_order_id = Column(String(100), nullable=True, index=True)
    broker_position_id = Column(String(100), nullable=True, index=True)
    broker_account_id = Column(String(50), nullable=True)
    broker_metadata = Column(JSON, nullable=True)
    mode = Column(String(20), nullable=True, default='paper', index=True)


class CTraderToken(Base):
    """Persistent storage for rotating cTrader OAuth tokens."""
    __tablename__ = "ctrader_tokens"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(String(50), unique=True, nullable=False, index=True)
    client_id = Column(String(100), nullable=False)
    client_secret = Column(Text, nullable=True)
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=False)
    expires_in = Column(Integer, default=2592000)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


def _default_snapshot_mode() -> str:
    try:
        from backend.services.trading_mode import get_trading_mode
        m = get_trading_mode()
        return m.value if hasattr(m, "value") else str(m)
    except Exception:
        return "paper"


class PortfolioSnapshot(Base):
    """Portfolio state snapshots for equity curve tracking"""
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    total_value = Column(Float, nullable=False, default=10000.0)
    cash = Column(Float, nullable=False, default=10000.0)
    positions_value = Column(Float, nullable=False, default=0.0)
    total_pnl = Column(Float, nullable=False, default=0.0)
    open_positions = Column(Integer, nullable=False, default=0)
    cycle_number = Column(Integer, nullable=True)
    broker = Column(String(50), nullable=True, default="ctrader", index=True)
    account_id = Column(String(50), nullable=True, index=True)
    mode = Column(String(20), nullable=True, default=_default_snapshot_mode, index=True)

    def __init__(self, **kwargs):
        if "mode" not in kwargs or kwargs["mode"] is None:
            kwargs["mode"] = _default_snapshot_mode()
        super().__init__(**kwargs)


# ═══ Paper Trading Persistence Models ═══ (Fincept Port A)

class PaperPortfolio(Base):
    """Paper trading portfolio accounts"""
    __tablename__ = "paper_portfolios"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    broker = Column(String(50), nullable=False)
    currency = Column(String(10), default="USD")
    leverage = Column(Float, default=1.0)
    margin_mode = Column(String(20), default="cross")
    fee_rate = Column(Float, default=0.001)
    initial_balance = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    margin_used = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class PaperOrder(Base):
    """Paper trading orders"""
    __tablename__ = "paper_orders"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String(50), unique=True, nullable=False, index=True)
    portfolio_id = Column(String(20), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    order_type = Column(String(20), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, default=0.0)
    stop_price = Column(Float, default=0.0)
    filled_qty = Column(Float, default=0.0)
    avg_price = Column(Float, default=0.0)
    status = Column(String(20), default="pending")
    reduce_only = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    filled_at = Column(DateTime(timezone=True), nullable=True)


class PaperTrade(Base):
    """Paper trading fills / executed trades"""
    __tablename__ = "paper_trades"

    id = Column(Integer, primary_key=True, index=True)
    trade_id = Column(String(50), unique=True, nullable=False, index=True)
    order_id = Column(String(50), nullable=False, index=True)
    portfolio_id = Column(String(20), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    pnl = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# ═══ Track C+ : Learned Strategy Skills (skill miner) ═══

class StrategySkill(Base):
    """A learned, named trading 'skill' mined from the agent's own trade history.

    Each row is a recurring market-setup archetype (a cluster of similar trades)
    together with its realised edge. The opinion layer matches the live setup to
    the best-fitting skill and votes with the skill's historical bias/confidence.
    """
    __tablename__ = "strategy_skills"

    id = Column(Integer, primary_key=True, index=True)
    skill_key = Column(String(64), unique=True, nullable=False, index=True)  # stable hash of the centroid
    name = Column(String(120), nullable=False)        # human-readable, e.g. "Trending + bullish momentum (BTC)"
    description = Column(Text, nullable=True)

    # Centroid of the cluster in feature space (the "what the market looks like").
    centroid = Column(JSON, nullable=False)            # list[float], normalised feature vector
    feature_summary = Column(JSON, nullable=True)      # human-readable feature dict at the centroid

    # Realised edge
    direction = Column(String(10), nullable=False, default="neutral")  # bullish | bearish | neutral
    sample_count = Column(Integer, nullable=False, default=0)
    win_rate = Column(Float, nullable=False, default=0.0)
    avg_pnl = Column(Float, nullable=False, default=0.0)
    total_pnl = Column(Float, nullable=False, default=0.0)
    sharpe = Column(Float, nullable=True)              # avg_pnl / std(pnl), consistency proxy
    edge_score = Column(Float, nullable=False, default=0.0)   # composite 0..1 used for ranking/confidence

    symbols = Column(JSON, nullable=True)              # symbols that contributed
    active = Column(Boolean, nullable=False, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_mined_at = Column(DateTime(timezone=True), nullable=True)


# ═══ P1: Shadow PnL outcomes (gate effectiveness measurement) ═══

class ShadowOutcome(Base):
    """Hypothetical outcome of a BLOCKED/vetoed signal (shadow tracker).

    One row per TradingSignal that a gate refused (Kronos veto, RANGING block,
    funding gate, min-edge gate, ...). The tracker walks 1h bars forward from
    the signal and records what WOULD have happened. Aggregated per gate this
    answers: is the gate saving money (blocked losers) or costing money
    (blocked winners)?
    """
    __tablename__ = "shadow_outcomes"

    id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(Integer, unique=True, nullable=False, index=True)  # TradingSignal.id (score-once)
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(String(10), nullable=False)       # BUY | SELL
    gate = Column(String(40), nullable=False, index=True)  # classify_gate() output
    signal_time = Column(DateTime(timezone=True), nullable=True, index=True)
    confidence = Column(Float, nullable=False, default=0.0)

    entry_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    exit_reason = Column(String(20), nullable=False, default="timeout")     # sl | tp | timeout

    pnl_pct = Column(Float, nullable=False, default=0.0)              # signed % captured, direction-aware
    pnl_r = Column(Float, nullable=False, default=0.0)                # PnL in R multiples (risk = entry→SL)
    mfe_r = Column(Float, nullable=False, default=0.0)   # max favorable excursion (R)
    mae_r = Column(Float, nullable=False, default=0.0)   # max adverse excursion (R, negative)
    bars_elapsed = Column(Integer, nullable=False, default=0)

    scored_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)


# ═══ Phase 1: Forex & Macro News Sentiment Models ═══

class NewsArticle(Base):
    """Raw news articles ingested from ForexNewsAPI, Finnhub, RSS, or cryptocurrency.cv."""
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(50), nullable=False, index=True)  # e.g. 'forexnewsapi', 'finnhub', 'cryptocurrency.cv'
    external_id = Column(String(255), nullable=True, index=True)
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    url = Column(Text, nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=False, index=True)
    fetched_at = Column(DateTime(timezone=True), server_default=func.now())
    raw_payload = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('source', 'external_id', name='uq_news_articles_source_external_id'),
    )


class SentimentScore(Base):
    """Sentiment score per currency pair or symbol derived from an article."""
    __tablename__ = "sentiment_scores"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("news_articles.id", ondelete="CASCADE"), nullable=False, index=True)
    pair = Column(String(20), nullable=False, index=True)  # e.g. 'EURUSD', 'GBPJPY', 'XAUUSD', 'BTCUSDT'
    sentiment = Column(String(10), nullable=False)  # 'bullish', 'bearish', 'neutral'
    score = Column(Float, nullable=False)  # -1.0 to +1.0
    confidence = Column(Float, nullable=True, default=0.5)
    model = Column(String(50), nullable=False, default="gemini-2.0-flash")
    reasoning = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (
        Index('idx_sentiment_pair_created', 'pair', 'created_at'),
    )


# ═══ Phase 3: Economic Calendar Event Models ═══

class CalendarEvent(Base):
    """Macro and economic calendar events (NFP, FOMC, CPI, central bank rate decisions)."""
    __tablename__ = "calendar_events"

    id = Column(Integer, primary_key=True, index=True)
    time_utc = Column(DateTime(timezone=True), nullable=False, index=True)
    currency = Column(String(10), nullable=False, index=True)  # USD, EUR, GBP, JPY, AUD, CAD, CHF, NZD
    impact = Column(String(10), nullable=False, index=True)    # HIGH, MEDIUM, LOW, NONE
    title = Column(Text, nullable=False)
    actual = Column(String(50), nullable=True)
    forecast = Column(String(50), nullable=True)
    previous = Column(String(50), nullable=True)
    source = Column(String(50), nullable=False, default="forexfactory")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_calendar_curr_time', 'currency', 'time_utc'),
    )


# ═══ Phase 2: JEV pipeline ingest / eval / shadow-paper attribution ═══
# Additive tables only. Existing HedgeFund/Trade/Paper/ApiKey models are unchanged.
# JSON (not Postgres JSONB) so SQLite hedge_fund.db and optional Postgres both work.


class JevMarketScan(Base):
    """n8n / scanner market snapshot stored for later JEV evaluation."""
    __tablename__ = "jev_market_scans"

    id = Column(Integer, primary_key=True, index=True)
    scan_timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, default="M5")
    signal_direction = Column(String(10), nullable=True)
    confidence = Column(Float, nullable=True)
    signal_strength = Column(Float, nullable=True)
    indicators = Column(JSON, nullable=False, default=dict)
    price_data = Column(JSON, nullable=True)
    volume_data = Column(JSON, nullable=True)
    scan_source = Column(String(50), nullable=False, default="n8n_market_scanner")
    status = Column(String(20), nullable=False, default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "scan_timestamp", name="uq_jev_market_scan"),
        Index("idx_jev_market_scan_symbol_time", "symbol", "scan_timestamp"),
    )


class JevNewsSentimentLog(Base):
    """n8n / news-macro sentiment row stored as JEV context."""
    __tablename__ = "jev_news_sentiment_logs"

    id = Column(Integer, primary_key=True, index=True)
    log_timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    source = Column(String(50), nullable=False, index=True)
    headline = Column(Text, nullable=True)
    url = Column(Text, nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    sentiment_score = Column(Float, nullable=True)
    sentiment_label = Column(String(20), nullable=True)
    impact_rating = Column(String(10), nullable=True)
    categories = Column(JSON, nullable=True)
    entities = Column(JSON, nullable=True)
    full_data = Column(JSON, nullable=True)
    market_scan_id = Column(Integer, ForeignKey("jev_market_scans.id"), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_jev_news_source_time", "source", "log_timestamp"),
    )


class JevPipelineEval(Base):
    """One JEV evaluation against stored market + news context."""
    __tablename__ = "jev_pipeline_evals"

    id = Column(Integer, primary_key=True, index=True)
    log_timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, default="1H")
    jev_signal = Column(String(20), nullable=True)
    jev_confidence = Column(Float, nullable=True)
    jev_strength = Column(Float, nullable=True)
    probabilities = Column(JSON, nullable=True)
    checks = Column(JSON, nullable=True)
    market_context = Column(JSON, nullable=True)
    news_sentiment_context = Column(JSON, nullable=True)
    combined_context = Column(JSON, nullable=True)
    combined_evaluation = Column(JSON, nullable=True)
    outcome_prediction = Column(String(20), nullable=True)
    outcome_probability = Column(Float, nullable=True)
    calibration_status = Column(String(30), nullable=True)
    execution_mode = Column(String(10), nullable=False, default="off")
    execution_decision = Column(String(40), nullable=False, default="logged_only")
    would_have_executed = Column(Boolean, nullable=False, default=False)
    executed = Column(Boolean, nullable=False, default=False)
    execution_id = Column(Integer, nullable=True)
    quantum_trade_id = Column(String(64), nullable=True)
    source_scan_id = Column(Integer, ForeignKey("jev_market_scans.id"), nullable=True, index=True)
    evaluated = Column(Boolean, nullable=False, default=False)
    evaluated_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_jev_pipeline_eval_symbol_time", "symbol", "log_timestamp"),
        Index("idx_jev_pipeline_eval_evaluated", "evaluated"),
        Index("idx_jev_pipeline_eval_executed", "executed"),
    )


class JevPipelineTrade(Base):
    """Shadow intent or paper fill attributed to a pipeline eval. Never a live venue fill."""
    __tablename__ = "jev_pipeline_trades"

    id = Column(Integer, primary_key=True, index=True)
    trade_timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=True)
    entry_price = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    realized_at = Column(DateTime(timezone=True), nullable=True)
    source = Column(String(50), nullable=False, default="jev_pipeline")
    mode = Column(String(20), nullable=False, default="shadow", index=True)
    source_scan_id = Column(Integer, ForeignKey("jev_market_scans.id"), nullable=True, index=True)
    jev_eval_id = Column(Integer, ForeignKey("jev_pipeline_evals.id"), nullable=True, index=True)
    ledger_trade_id = Column(Integer, nullable=True)
    outcome = Column(String(20), nullable=False, default="open")
    outcome_reason = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index("idx_jev_pipeline_trade_symbol_time", "symbol", "trade_timestamp"),
        Index("idx_jev_pipeline_trade_outcome", "outcome"),
    )


