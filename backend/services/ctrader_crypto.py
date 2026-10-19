"""cTrader (IC Markets) demo crypto add-on.

Binance keeps ``VENUE_CRYPTO`` and its own loop. This module is the extra
path: when ``CTRADER_CRYPTO_ENABLED`` is true, mapped crypto names are
evaluated and (demo only) executed on cTrader as well.

Defaults are fail-closed. Deploying the code changes nothing until the
flag is flipped, and live ``CTRADER_ENV`` still refuses crypto orders.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional

from backend.services.equity_scope import ctrader_env
from backend.services.multi_asset_bars import classify_symbol
from backend.services.symbol_aliases import crypto_perp_base
from backend.utils.safe_errors import sanitize_log_text

logger = logging.getLogger(__name__)

# Scan / Binance names → IC Markets cTrader CFD names.
# LINK/AVAX/DOGE use IC's shortened codes (LNK/AVX/DOG).
SCAN_TO_IC_CRYPTO: dict[str, str] = {
    "BTCUSDT": "BTCUSD",
    "BTCUSDC": "BTCUSD",
    "BTCUSD": "BTCUSD",
    "BTC": "BTCUSD",
    "ETHUSDT": "ETHUSD",
    "ETHUSDC": "ETHUSD",
    "ETHUSD": "ETHUSD",
    "ETH": "ETHUSD",
    "SOLUSDT": "SOLUSD",
    "SOLUSDC": "SOLUSD",
    "SOLUSD": "SOLUSD",
    "SOL": "SOLUSD",
    "XRPUSDT": "XRPUSD",
    "XRPUSDC": "XRPUSD",
    "XRPUSD": "XRPUSD",
    "XRP": "XRPUSD",
    "ADAUSDT": "ADAUSD",
    "ADAUSDC": "ADAUSD",
    "ADAUSD": "ADAUSD",
    "ADA": "ADAUSD",
    "BNBUSDT": "BNBUSD",
    "BNBUSDC": "BNBUSD",
    "BNBUSD": "BNBUSD",
    "BNB": "BNBUSD",
    "LINKUSDT": "LNKUSD",
    "LINKUSDC": "LNKUSD",
    "LINKUSD": "LNKUSD",
    "LNKUSD": "LNKUSD",
    "LINK": "LNKUSD",
    "AVAXUSDT": "AVXUSD",
    "AVAXUSDC": "AVXUSD",
    "AVAXUSD": "AVXUSD",
    "AVXUSD": "AVXUSD",
    "AVAX": "AVXUSD",
    "DOGEUSDT": "DOGUSD",
    "DOGEUSDC": "DOGUSD",
    "DOGEUSD": "DOGUSD",
    "DOGUSD": "DOGUSD",
    "DOGE": "DOGUSD",
}

IC_CRYPTO_SYMBOLS: frozenset[str] = frozenset(SCAN_TO_IC_CRYPTO.values())

# IC short codes → canonical base used for cross-venue overlap.
_IC_SHORT_TO_BASE: dict[str, str] = {
    "LNK": "LINK",
    "AVX": "AVAX",
    "DOG": "DOGE",
}
_IC_BASE_TO_SHORT: dict[str, str] = {full: short for short, full in _IC_SHORT_TO_BASE.items()}

# Sensible per-asset-class spread caps (pips of that symbol's pip size).
# Forex stays at the historical 2.0. Gold ~20-40 cents = 20-40 XAU pips.
# Crypto CFDs are wider; BTC pip is $1 so 80 pips ≈ $80.
DEFAULT_SPREAD_FOREX = 2.0
DEFAULT_SPREAD_METALS = 40.0
DEFAULT_SPREAD_CRYPTO = 80.0

_TRUE = {"1", "true", "yes", "on"}

# Last catalog validation outcome (startup / reconnect). None = not validated yet.
_accepted_catalog: Optional[set[str]] = None
_unknown_catalog: set[str] = set()
# canonical IC name → live ProtoOA symbolName that matched (may be an alias).
_resolved_catalog: dict[str, str] = {}


def _truthy(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default) or default).strip().lower() in _TRUE


def _norm_symbol(symbol: str) -> str:
    name = (symbol or "").upper().replace("/", "").replace("-", "").replace("_", "").strip()
    if name.endswith("=X"):
        name = name[:-2]
    return name.rstrip(".")


def _catalog_key(name: str) -> str:
    """Live-catalog compare key: strip slashes, dashes, spaces, trailing dots."""
    return (name or "").upper().replace("/", "").replace("-", "").replace(" ", "").strip().rstrip(".")


def ic_crypto_alias_variants(symbol: str) -> list[str]:
    """Common IC Markets catalog spellings for a mapped crypto CFD.

    IC may list ``XRPUSD.`` (trailing dot), ``AVAXUSD`` instead of ``AVXUSD``,
    or only one of the short/full LINK/DOGE codes. Callers try these against
    the live ProtoOA list and skip names that still do not match.
    """
    name = _catalog_key(symbol)
    if not name:
        return []
    variants: list[str] = []

    def _add(raw: str) -> None:
        key = _catalog_key(raw)
        if not key or key in variants:
            return
        variants.append(key)
        dotted = f"{key}."
        if dotted not in variants:
            variants.append(dotted)

    _add(name)
    mapped = SCAN_TO_IC_CRYPTO.get(name)
    if mapped:
        _add(mapped)
    if name.endswith("USD") and len(name) > 3:
        base = name[:-3]
        if base in _IC_SHORT_TO_BASE:
            _add(f"{_IC_SHORT_TO_BASE[base]}USD")
        if base in _IC_BASE_TO_SHORT:
            _add(f"{_IC_BASE_TO_SHORT[base]}USD")
        mapped_base = SCAN_TO_IC_CRYPTO.get(base)
        if mapped_base:
            _add(mapped_base)
    return variants


def resolve_ic_crypto_catalog_name(symbol: str, known_names: Iterable[str]) -> Optional[str]:
    """Return the live catalog name that matches *symbol*, or None (fail-soft)."""
    known_map: dict[str, str] = {}
    for raw in known_names or []:
        original = str(raw or "").strip()
        if not original:
            continue
        key = _catalog_key(original)
        if not key:
            continue
        # Prefer the dotted-stripped form so it matches normalize_catalog_name.
        known_map.setdefault(key, key)
        known_map.setdefault(original.upper(), key)
    if not known_map:
        return None
    for variant in ic_crypto_alias_variants(symbol):
        resolved = known_map.get(variant) or known_map.get(_catalog_key(variant))
        if resolved:
            return resolved
    return None


def resolved_ic_crypto_name(symbol: str) -> str:
    """Canonical → last live catalog name from validation (identity if unseen)."""
    key = _norm_symbol(symbol)
    return _resolved_catalog.get(key, key)


def ctrader_crypto_enabled() -> bool:
    """Operator flag. Default false — deploy is a no-op."""
    return _truthy("CTRADER_CRYPTO_ENABLED", "false")


def ctrader_crypto_orders_allowed() -> bool:
    """place_order may send crypto only on the IC demo host."""
    return ctrader_crypto_enabled() and ctrader_env() == "demo"


def is_ic_crypto_symbol(symbol: str) -> bool:
    """True for IC Markets crypto CFD names (BTCUSD, LNKUSD, …)."""
    return _norm_symbol(symbol) in IC_CRYPTO_SYMBOLS


def map_to_ic_crypto(symbol: str) -> Optional[str]:
    """Map a scan / Binance / short name to the IC Markets CFD name."""
    raw = (symbol or "").strip()
    if not raw:
        return None
    upper = raw.upper()
    if upper.endswith("=X"):
        upper = upper[:-2]
    key = _norm_symbol(upper)
    if key in SCAN_TO_IC_CRYPTO:
        return SCAN_TO_IC_CRYPTO[key]
    # BTC-USD / ETH-USD already stripped to BTCUSD / ETHUSD above.
    if key.endswith("USD") and key in IC_CRYPTO_SYMBOLS:
        return key
    base = crypto_perp_base(key)
    if base:
        mapped = SCAN_TO_IC_CRYPTO.get(base) or SCAN_TO_IC_CRYPTO.get(f"{base}USDT")
        if mapped:
            return mapped
    return None


def crypto_base_asset(symbol: str) -> Optional[str]:
    """Canonical coin for overlap checks: ETHUSDT / ETHUSD / ETH-USD → ETH."""
    key = _norm_symbol(symbol)
    if not key:
        return None
    ic = map_to_ic_crypto(key)
    if ic:
        raw = ic[:-3] if ic.endswith("USD") else ic
        return _IC_SHORT_TO_BASE.get(raw, raw)
    perp = crypto_perp_base(key)
    if perp:
        return _IC_SHORT_TO_BASE.get(perp, perp)
    if classify_symbol(key) == "crypto" and key.endswith("USD") and len(key) > 3:
        raw = key[:-3]
        return _IC_SHORT_TO_BASE.get(raw, raw)
    return None


def _parse_symbol_list(raw: str) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def requested_ctrader_crypto_symbols() -> list[str]:
    """Operator list (or TRADING_SYMBOLS / built-in map) as IC names, not yet catalog-checked."""
    explicit = _parse_symbol_list(os.getenv("CTRADER_CRYPTO_SYMBOLS", "") or "")
    sources: Iterable[str]
    if explicit:
        sources = explicit
    else:
        trading = _parse_symbol_list(os.getenv("TRADING_SYMBOLS", "") or "")
        sources = trading or list(IC_CRYPTO_SYMBOLS)
    seen: set[str] = set()
    out: list[str] = []
    for raw in sources:
        mapped = map_to_ic_crypto(raw)
        if not mapped or mapped in seen:
            continue
        seen.add(mapped)
        out.append(mapped)
    return out


def validate_ctrader_crypto_catalog(known_names: Optional[Iterable[str]] = None) -> list[str]:
    """Intersect requested IC names with the live (or default) cTrader catalog.

    Unknown names are skipped with a warning. An empty *known_names* means
    the live ProtoOASymbolsListRes has not arrived — accept nothing
    (fail-closed). Default symbol IDs are not a catalog.

    Matching is alias-aware: ``AVXUSD`` can resolve to ``AVAXUSD``, and
    ``XRPUSD`` can resolve to ``XRPUSD.``. The canonical requested name is
    accepted; the live spelling is stored for order lookup.
    """
    global _accepted_catalog, _unknown_catalog, _resolved_catalog
    wanted = requested_ctrader_crypto_symbols()
    known_list = [str(n) for n in (known_names or []) if n]
    accepted: list[str] = []
    unknown: set[str] = set()
    resolved: dict[str, str] = {}
    for name in wanted:
        live = resolve_ic_crypto_catalog_name(name, known_list) if known_list else None
        if live:
            accepted.append(name)
            resolved[name] = live
            if live != name:
                logger.info(
                    "cTrader crypto %s resolved to live catalog name %s",
                    sanitize_log_text(name).replace("\r", "").replace("\n", ""),
                    sanitize_log_text(live).replace("\r", "").replace("\n", ""),
                )
        else:
            unknown.add(name)
            logger.warning(
                "cTrader crypto symbol %s is not in the live catalog — skipping",
                sanitize_log_text(name).replace("\r", "").replace("\n", ""),
            )
    _accepted_catalog = set(accepted)
    _unknown_catalog = unknown
    _resolved_catalog = resolved
    return accepted


def accepted_ctrader_crypto_symbols() -> list[str]:
    """Validated IC crypto names. Re-runs validation when the catalog is unseen."""
    if _accepted_catalog is None:
        try:
            from backend.services.ctrader_service import ctrader_service

            known = set(getattr(ctrader_service, "_live_catalog_names", {}) or {})
        except Exception:
            known = set()
        return validate_ctrader_crypto_catalog(known or None)
    return sorted(_accepted_catalog)


def reset_ctrader_crypto_catalog_cache() -> None:
    """Test helper — forget the last catalog validation."""
    global _accepted_catalog, _unknown_catalog, _resolved_catalog
    _accepted_catalog = None
    _unknown_catalog = set()
    _resolved_catalog = {}


def broker_for_scan_symbol(symbol: str) -> str:
    """Venue used by the candidate scan for this exact symbol.

    IC crypto names go to cTrader when the add-on is on. Everything else
    keeps ``venue_for_symbol`` (so ``VENUE_CRYPTO`` stays binance).
    """
    from backend.services.venue_router import venue_for_symbol

    if ctrader_crypto_enabled() and is_ic_crypto_symbol(symbol):
        return "ctrader"
    return venue_for_symbol(symbol)


def ctrader_trade_mode() -> str:
    """SQL Trade.mode for cTrader rows — follows CTRADER_ENV / account type."""
    env = ctrader_env()
    if env == "live":
        return "live"
    if env in {"demo", "sandbox"}:
        return "demo"
    return "paper"


def max_spread_pips_for_symbol(symbol: str, forex_default: float = DEFAULT_SPREAD_FOREX) -> float:
    """Per-asset-class spread gate. Forex default stays 2.0 unless env overrides."""
    name = _norm_symbol(symbol)
    per_sym = os.getenv(f"CTRADER_MAX_SPREAD_{name}")
    if per_sym:
        try:
            return float(per_sym)
        except (TypeError, ValueError):
            pass
    asset = classify_symbol(name)
    if asset == "metal":
        return _env_float("CTRADER_MAX_SPREAD_METALS", DEFAULT_SPREAD_METALS)
    if asset == "crypto":
        return _env_float("CTRADER_MAX_SPREAD_CRYPTO", DEFAULT_SPREAD_CRYPTO)
    return _env_float("CTRADER_MAX_SPREAD_FOREX", forex_default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default
