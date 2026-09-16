"""Canonical multi-asset training universe for Jesse GPU jobs.

Five user-facing buckets. Symbols are only those QuantumTrade already knows
via unified_feed / multi_asset_bars / TRADING_SYMBOLS. No invented minerals,
gas, or copper feeds — those are absent from the platform maps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

UserAssetClass = Literal["crypto", "forex", "metals", "minerals", "stocks"]
ALL_CLASSES: Tuple[UserAssetClass, ...] = (
    "crypto",
    "forex",
    "metals",
    "minerals",
    "stocks",
)

_CRYPTO_SUFFIXES = ("USDT", "USDC", "BUSD", "PERP")
_FOREX_MAJORS = {
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
    "EURCHF",
    "EURAUD",
}
_METALS = {"XAUUSD", "XAGUSD", "GOLD", "SILVER", "XPTUSD", "XPDUSD"}
_OIL = {"USOIL", "UKOIL", "WTI", "BRENT", "CL=F", "BZ=F"}
_INDEX = {"SPX", "SPY", "QQQ", "DIA", "NDX", "VIX"}


@dataclass(frozen=True)
class UniverseSymbol:
    """One train-able series and how to obtain its OHLCV."""

    symbol: str
    asset_class: UserAssetClass
    source: str
    ticker: str
    native_tf: str = "1h"

    def filename(self) -> str:
        return candle_filename(self.symbol, self.native_tf)


def _crypto(symbol: str, source: str, ticker: str, native_tf: str) -> UniverseSymbol:
    return UniverseSymbol(
        symbol=symbol,
        asset_class="crypto",
        source=source,
        ticker=ticker,
        native_tf=native_tf,
    )


def _yf(symbol: str, asset_class: UserAssetClass, ticker: str) -> UniverseSymbol:
    return UniverseSymbol(
        symbol=symbol,
        asset_class=asset_class,
        source="yfinance",
        ticker=ticker,
        native_tf="1h",
    )


# Jesse Postgres currently holds only BTC/ETH/SOL 1m dumps. Extra crypto is
# public Binance USDT klines (same pairs as live TRADING_SYMBOLS, no keys).
_CRYPTO: Tuple[UniverseSymbol, ...] = (
    _crypto("BTC-USDT", "jesse_dump", "BTCUSDT", "1m"),
    _crypto("ETH-USDT", "jesse_dump", "ETHUSDT", "1m"),
    _crypto("SOL-USDT", "jesse_dump", "SOLUSDT", "1m"),
    _crypto("BNB-USDT", "binance", "BNBUSDT", "1h"),
    _crypto("XRP-USDT", "binance", "XRPUSDT", "1h"),
    _crypto("AVAX-USDT", "binance", "AVAXUSDT", "1h"),
    _crypto("LINK-USDT", "binance", "LINKUSDT", "1h"),
    _crypto("UNI-USDT", "binance", "UNIUSDT", "1h"),
    _crypto("NEAR-USDT", "binance", "NEARUSDT", "1h"),
    _crypto("LTC-USDT", "binance", "LTCUSDT", "1h"),
    _crypto("DOT-USDT", "binance", "DOTUSDT", "1h"),
    _crypto("ATOM-USDT", "binance", "ATOMUSDT", "1h"),
    _crypto("OP-USDT", "binance", "OPUSDT", "1h"),
    _crypto("INJ-USDT", "binance", "INJUSDT", "1h"),
    _crypto("SUI-USDT", "binance", "SUIUSDT", "1h"),
    _crypto("POL-USDT", "binance", "POLUSDT", "1h"),
)

_FOREX: Tuple[UniverseSymbol, ...] = (
    _yf("EURUSD", "forex", "EURUSD=X"),
    _yf("GBPUSD", "forex", "GBPUSD=X"),
    _yf("USDJPY", "forex", "USDJPY=X"),
    _yf("EURJPY", "forex", "EURJPY=X"),
    _yf("AUDUSD", "forex", "AUDUSD=X"),
    _yf("USDCAD", "forex", "USDCAD=X"),
    _yf("USDCHF", "forex", "USDCHF=X"),
    _yf("NZDUSD", "forex", "NZDUSD=X"),
)

_METALS_SERIES: Tuple[UniverseSymbol, ...] = (
    _yf("XAUUSD", "metals", "GC=F"),
    _yf("XAGUSD", "metals", "SI=F"),
    _yf("XPTUSD", "metals", "PL=F"),
    _yf("XPDUSD", "metals", "PA=F"),
)

# Platform commodities = WTI/Brent proxies (USOIL/UKOIL). No copper, NG, or
# industrial-mineral series is wired in unified_feed / multi_asset_bars.
_MINERALS: Tuple[UniverseSymbol, ...] = (
    _yf("USOIL", "minerals", "CL=F"),
    _yf("UKOIL", "minerals", "BZ=F"),
)

_STOCKS: Tuple[UniverseSymbol, ...] = (
    _yf("AAPL", "stocks", "AAPL"),
    _yf("MSFT", "stocks", "MSFT"),
    _yf("NVDA", "stocks", "NVDA"),
)

DEFAULT_UNIVERSE: Dict[UserAssetClass, Tuple[UniverseSymbol, ...]] = {
    "crypto": _CRYPTO,
    "forex": _FOREX,
    "metals": _METALS_SERIES,
    "minerals": _MINERALS,
    "stocks": _STOCKS,
}

# Gaps we refuse to invent. Documented so trainers skip rather than synthesize.
UNAVAILABLE_FEEDS: Tuple[str, ...] = (
    "copper / HG=F — not in unified_feed or multi_asset_bars",
    "natural gas / NG=F — not in the platform commodity map",
    "industrial minerals (iron ore, lithium, rare earths) — no feed",
    "Jesse Postgres forex/metals/stocks/oil candles — only BTC/ETH/SOL 1m exist",
    "Qlib dumps — none on the training node or trading VPS",
    "SPX/NDX/VIX — index feed exists for quotes, excluded from equity training",
)


def normalize_symbol(symbol: str) -> str:
    raw = (symbol or "").strip().upper().replace("/", "-")
    if raw.endswith("USDT") and "-" not in raw:
        return raw[:-4] + "-USDT"
    if raw.endswith("USDC") and "-" not in raw:
        return raw[:-4] + "-USDT"
    return raw


def candle_filename(symbol: str, native_tf: str = "1h") -> str:
    return f"{normalize_symbol(symbol)}_{native_tf}.csv.gz"


def classify_training_symbol(symbol: str) -> Optional[UserAssetClass]:
    """Map a ticker into the five training buckets. None = unknown, skip."""
    sym = (symbol or "").upper().replace("/", "").replace("-", "").strip()
    if not sym:
        return None
    if sym in _METALS or sym.startswith("XAU") or sym.startswith("XAG"):
        return "metals"
    if sym in _OIL or (sym.endswith("=F") and sym.startswith(("CL", "BZ"))):
        return "minerals"
    if sym in _FOREX_MAJORS or (len(sym) == 6 and sym.isalpha()):
        return "forex"
    if any(sym.endswith(sfx) for sfx in _CRYPTO_SUFFIXES) or (
        sym.endswith("USD") and len(sym) > 6
    ):
        return "crypto"
    if sym in _INDEX:
        return None
    return "stocks"


def default_universe() -> Dict[UserAssetClass, Tuple[UniverseSymbol, ...]]:
    return DEFAULT_UNIVERSE


def resolve_universe(
    asset_class: str = "all",
    symbols: Optional[Sequence[str]] = None,
) -> List[UniverseSymbol]:
    """Return the ordered training list for one class, or all five buckets."""
    wanted = (asset_class or "all").strip().lower()
    if wanted in ("commodity", "commodities", "oil"):
        wanted = "minerals"
    if wanted in ("metal",):
        wanted = "metals"
    if wanted in ("equity", "equities", "stock"):
        wanted = "stocks"
    if wanted == "fx":
        wanted = "forex"

    rows: List[UniverseSymbol] = []
    if wanted == "all":
        for cls in ALL_CLASSES:
            rows.extend(DEFAULT_UNIVERSE[cls])
    elif wanted in DEFAULT_UNIVERSE:
        rows.extend(DEFAULT_UNIVERSE[wanted])  # type: ignore[index]
    else:
        raise ValueError(
            f"unknown asset class {asset_class!r}; expected one of {ALL_CLASSES} or 'all'"
        )

    if symbols:
        allow = {normalize_symbol(s) for s in symbols}
        rows = [r for r in rows if normalize_symbol(r.symbol) in allow]
    return rows


def universe_as_dict() -> Dict[str, List[Dict[str, str]]]:
    return {
        cls: [asdict(item) for item in items]
        for cls, items in DEFAULT_UNIVERSE.items()
    }


def lookup_symbol(symbol: str) -> Optional[UniverseSymbol]:
    target = normalize_symbol(symbol)
    for items in DEFAULT_UNIVERSE.values():
        for item in items:
            if normalize_symbol(item.symbol) == target:
                return item
    return None


def iter_downloadable(rows: Iterable[UniverseSymbol]) -> List[UniverseSymbol]:
    """Series that can be fetched without Jesse Postgres (binance / yfinance)."""
    return [r for r in rows if r.source in ("binance", "yfinance")]
