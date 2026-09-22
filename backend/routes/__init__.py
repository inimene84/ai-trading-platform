from fastapi import APIRouter

from backend.routes.hedge_fund import router as hedge_fund_router
from backend.routes.health import router as health_router
from backend.routes.storage import router as storage_router
from backend.routes.flows import router as flows_router
from backend.routes.flow_runs import router as flow_runs_router
from backend.routes.ollama import router as ollama_router
from backend.routes.language_models import router as language_models_router
from backend.routes.api_keys import router as api_keys_router
from backend.routes.trading import router as trading_router
from backend.routes.news import router as news_router
from backend.routes.market_data import alerts_router as market_alerts_router
from backend.routes.market_data import router as market_data_router
from backend.routes.feed import router as feed_router
from backend.routes.forecast import router as forecast_router
from backend.routes.historical import router as historical_router
from backend.routes.telemetry import router as telemetry_router
from backend.routes.backtest import router as backtest_router
from backend.routes.sentry import router as sentry_router
from backend.routes.signals import router as signals_router
from backend.routes.assistant import router as assistant_router
from backend.routes.agents import router as agents_router
from backend.routes.sentiment import router as sentiment_router
from backend.routes.calendar import router as calendar_router
from backend.routes.research import router as research_router
from backend.routes.jesse import router as jesse_router
from backend.routes.jev import router as jev_router

# Main API router
api_router = APIRouter()

# Include sub-routers
# Health + sentry are dual-mounted under /api so MCP clients using
# BACKEND_API_PREFIX=/api (same pattern as trading) do not 404.
api_router.include_router(health_router, tags=["health"])
api_router.include_router(health_router, prefix="/api", tags=["health-api"])
api_router.include_router(hedge_fund_router, tags=["hedge-fund"])
api_router.include_router(storage_router, tags=["storage"])
api_router.include_router(flows_router, tags=["flows"])
api_router.include_router(flow_runs_router, tags=["flow-runs"])
api_router.include_router(ollama_router, tags=["ollama"])
api_router.include_router(language_models_router, tags=["language-models"])
api_router.include_router(api_keys_router, tags=["api-keys"])
api_router.include_router(trading_router, tags=["trading"])
api_router.include_router(trading_router, prefix="/api", tags=["trading-api"])
api_router.include_router(news_router, tags=["news"])
api_router.include_router(market_data_router, tags=["market-data"])
api_router.include_router(market_alerts_router, tags=["alerts"])
api_router.include_router(feed_router, tags=["feed"])
api_router.include_router(forecast_router, tags=["forecast"])
api_router.include_router(historical_router)
api_router.include_router(telemetry_router)
api_router.include_router(backtest_router, prefix="/api/backtest", tags=["backtest"])
api_router.include_router(sentry_router, tags=["sentry"])
api_router.include_router(sentry_router, prefix="/api", tags=["sentry-api"])
api_router.include_router(signals_router, tags=["signals"])
api_router.include_router(signals_router, prefix="/api", tags=["signals-api"])
api_router.include_router(assistant_router, tags=["assistant"])
api_router.include_router(assistant_router, prefix="/api", tags=["assistant-api"])
api_router.include_router(assistant_router, prefix="/api/backend", tags=["assistant-backend"])
api_router.include_router(agents_router, tags=["agents"])
api_router.include_router(agents_router, prefix="/api", tags=["agents-api"])
api_router.include_router(agents_router, prefix="/api/backend", tags=["agents-backend"])
api_router.include_router(sentiment_router, tags=["sentiment"])
api_router.include_router(sentiment_router, prefix="/api", tags=["sentiment-api"])
api_router.include_router(calendar_router, tags=["calendar"])
api_router.include_router(calendar_router, prefix="/api", tags=["calendar-api"])
api_router.include_router(research_router, tags=["research"])
api_router.include_router(research_router, prefix="/api", tags=["research-api"])
api_router.include_router(jesse_router, prefix="/jesse", tags=["jesse"])
api_router.include_router(jesse_router, prefix="/api/jesse", tags=["jesse-api"])
api_router.include_router(jev_router, tags=["jev"])
api_router.include_router(jev_router, prefix="/api", tags=["jev-api"])

