"""Typed System One questions shared by sentiment and forecast evaluations.

One request evaluates every question. That is the concurrent pattern the
TypeSafe endpoint provides: the model answers the set together.
"""

from __future__ import annotations

TRADE_ACTIONS = (
    "STRONG_BUY",
    "BUY",
    "HOLD",
    "TAKE_PROFIT",
    "SELL",
    "STRONG_SELL",
)

PRICE_DIRECTIONS = ("UP", "FLAT", "DOWN")

SENTIMENT_LEVELS = (
    "Extreme Panic / Capitulation",
    "Cautious / Bearish",
    "Neutral / Mixed",
    "Optimistic / Bullish",
    "Euphoric / Greedy",
)

CATALYST_LEVELS = (
    "No news or pure retail noise",
    "Minor routine update or rumors",
    "Moderate ecosystem milestone",
    "Major market-shifting catalyst",
)


def analysis_questions() -> dict[str, dict]:
    """JSON question payload for POST /v1/systemone."""
    return {
        "trade_action": {
            "type": "choice",
            "instructions": (
                "Using only the supplied past-only `market` state and optional `social_stats`, "
                "what is the best immediate advisory trading stance for `asset`? "
                "This is analysis, not an order. Prefer HOLD when the edge is unclear."
            ),
            "criteria": {
                "STRONG_BUY": "High-conviction long: capitulation, verified breakout, or squeeze setup with confirming state.",
                "BUY": "Favorable long bias with positive upside expectation.",
                "HOLD": "Neutral, range-bound, or no clear asymmetric edge.",
                "TAKE_PROFIT": "Existing longs look extended; not a fresh short entry.",
                "SELL": "Bearish breakdown or deteriorating momentum.",
                "STRONG_SELL": "Crowded exhaustion or severe downside continuation.",
            },
        },
        "sentiment_spectrum": {
            "type": "score",
            "instructions": "Rate the prevailing mood in `social_stats` and `representative_posts`. If social data is empty, score the price state as Neutral / Mixed.",
            "criteria": list(SENTIMENT_LEVELS),
        },
        "is_short_squeeze_risk": {
            "type": "noul",
            "instructions": (
                "Does negative funding clashing with panic, or an extreme oversold reading "
                "against peak fear, indicate elevated short-squeeze or capitulation-bounce risk?"
            ),
            "criteria": {
                "true": "Funding, open interest, or panic diverges in a way that supports a squeeze or bounce.",
                "false": "No such divergence, or perpetuals context is missing.",
            },
        },
        "catalyst_impact": {
            "type": "score",
            "instructions": "Rate the significance of events described in `representative_posts`. Empty posts means no catalyst.",
            "criteria": list(CATALYST_LEVELS),
        },
        "price_direction": {
            "type": "choice",
            "instructions": (
                "Using only completed sessions in `market.sessions` plus RSI, ATR, moving-average distance, "
                "and volume ratio, what is the most likely direction of the next session close versus the last close? "
                "FLAT means a move smaller than about one ATR or no edge. Do not invent future prices."
            ),
            "criteria": {
                "UP": "Next close is likely higher than the last close.",
                "FLAT": "Next close is likely little changed versus the last close.",
                "DOWN": "Next close is likely lower than the last close.",
            },
        },
    }
