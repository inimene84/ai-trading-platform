from __future__ import annotations

from .portfolio import Portfolio
from .types import ActionLiteral, Action
from .venue_costs import estimate_round_trip, venue_costs_enabled


class TradeExecutor:
    """Executes trades against a Portfolio with Backtester-identical semantics."""

    def execute_trade(
        self,
        ticker: str,
        action: ActionLiteral,
        quantity: float,
        current_price: float,
        portfolio: Portfolio,
    ) -> int:
        if quantity is None or quantity <= 0:
            return 0

        # Coerce to enum if strings provided
        try:
            action_enum = Action(action) if not isinstance(action, Action) else action
        except Exception:
            action_enum = Action.HOLD

        filled = 0
        if action_enum == Action.BUY:
            filled = portfolio.apply_long_buy(ticker, int(quantity), float(current_price))
        elif action_enum == Action.SELL:
            filled = portfolio.apply_long_sell(ticker, int(quantity), float(current_price))
        elif action_enum == Action.SHORT:
            filled = portfolio.apply_short_open(ticker, int(quantity), float(current_price))
        elif action_enum == Action.COVER:
            filled = portfolio.apply_short_cover(ticker, int(quantity), float(current_price))
        else:
            return 0
        if filled and venue_costs_enabled():
            costs = estimate_round_trip(
                venue="binance",
                symbol=ticker,
                notional=abs(float(quantity) * float(current_price)),
                side="BUY" if action_enum in {Action.BUY, Action.COVER} else "SELL",
                maker_entry=False,
            )
            cash = portfolio._portfolio.get("cash", 0.0)
            portfolio._portfolio["cash"] = cash - float(costs.get("total") or 0.0)
        return filled

        # hold or unknown action
        return 0


