"""Backtesting package: interfaces and shared types for refactoring.

This module defines the public contracts (Protocols/ABCs) for the
backtesting subsystem. Implementations can live elsewhere and be
introduced gradually without changing existing behavior.
"""

from .types import (
    ActionLiteral,
    AgentDecision,
    AgentDecisions,
    AgentOutput,
    AgentSignals,
    PerformanceMetrics,
    PortfolioSnapshot,
    PortfolioValuePoint,
    PositionState,
    PriceDataFrame,
    TickerRealizedGains,
)

from .portfolio import Portfolio
from .trader import TradeExecutor
from .metrics import PerformanceMetricsCalculator
from .controller import AgentController
from .engine import BacktestEngine
from .valuation import calculate_portfolio_value, compute_exposures
from .output import OutputBuilder
from .venue_costs import (
    VenueCostConfig,
    default_cost_config,
    estimate_round_trip,
    venue_costs_enabled,
)

__all__ = [
    # Types
    "ActionLiteral",
    "AgentDecision",
    "AgentDecisions",
    "AgentOutput",
    "AgentSignals",
    "PerformanceMetrics",
    "PortfolioSnapshot",
    "PortfolioValuePoint",
    "PositionState",
    "PriceDataFrame",
    "TickerRealizedGains",
    # Interfaces
    "Portfolio",
    "TradeExecutor",
    "PerformanceMetricsCalculator",
    "AgentController",
    "BacktestEngine",
    "calculate_portfolio_value",
    "compute_exposures",
    "OutputBuilder",
    "VenueCostConfig",
    "default_cost_config",
    "estimate_round_trip",
    "venue_costs_enabled",
]


