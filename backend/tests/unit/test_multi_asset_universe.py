"""Unit tests for the five-bucket GPU training universe and CLI contract."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
JESSE = ROOT / "jesse_quant"
if str(JESSE) not in sys.path:
    sys.path.insert(0, str(JESSE))

from asset_universe import (
    ALL_CLASSES,
    UNAVAILABLE_FEEDS,
    candle_filename,
    classify_training_symbol,
    default_universe,
    lookup_symbol,
    normalize_symbol,
    resolve_universe,
)
from download_ohlcv import build_parser as build_download_parser
from train_gpu import MIN_TRAIN_EVENTS, build_parser as build_train_parser, find_candle_path
from train_ml import TooFewEventsError


def test_five_buckets_present():
    universe = default_universe()
    assert tuple(universe.keys()) == ALL_CLASSES
    assert set(ALL_CLASSES) == {"crypto", "forex", "metals", "minerals", "stocks"}
    for cls in ALL_CLASSES:
        assert len(universe[cls]) >= 2


def test_crypto_extends_beyond_btc_eth_sol():
    names = {item.symbol for item in resolve_universe("crypto")}
    assert {"BTC-USDT", "ETH-USDT", "SOL-USDT"} <= names
    assert "BNB-USDT" in names
    assert "XRP-USDT" in names


def test_minerals_are_platform_oil_only():
    minerals = resolve_universe("minerals")
    tickers = {item.ticker for item in minerals}
    symbols = {item.symbol for item in minerals}
    assert "CL=F" in tickers
    assert "BZ=F" in tickers
    assert "USOIL" in symbols
    assert "HG=F" not in tickers
    assert "NG=F" not in tickers
    assert any("copper" in gap for gap in UNAVAILABLE_FEEDS)


def test_classify_maps_into_user_buckets():
    assert classify_training_symbol("BTC-USDT") == "crypto"
    assert classify_training_symbol("EURUSD") == "forex"
    assert classify_training_symbol("XAUUSD") == "metals"
    assert classify_training_symbol("USOIL") == "minerals"
    assert classify_training_symbol("AAPL") == "stocks"
    assert classify_training_symbol("SPX") is None
    assert classify_training_symbol("") is None


def test_oil_alias_resolves_to_minerals():
    rows = resolve_universe("oil")
    assert rows and all(item.asset_class == "minerals" for item in rows)


def test_normalize_and_filename():
    assert normalize_symbol("btcusdt") == "BTC-USDT"
    assert candle_filename("EURUSD", "1h") == "EURUSD_1h.csv.gz"
    btc = lookup_symbol("BTCUSDT")
    assert btc is not None
    assert btc.native_tf == "1m"
    assert btc.filename() == "BTC-USDT_1m.csv.gz"


def test_train_cli_exposes_asset_class_and_candles_dir():
    parser = build_train_parser()
    ns = parser.parse_args(["--asset-class", "forex", "--candles-dir", "/tmp/candles", "--model", "both"])
    assert ns.asset_class == "forex"
    assert ns.candles_dir == "/tmp/candles"
    assert ns.pt_mult == 5.5
    assert ns.sl_mult == 1.75
    assert MIN_TRAIN_EVENTS == 20


def test_download_cli_lists_classes():
    parser = build_download_parser()
    ns = parser.parse_args(["--asset-class", "metals", "--list"])
    assert ns.asset_class == "metals"
    assert ns.list is True


def test_find_candle_path_prefers_existing(tmp_path):
    path = tmp_path / "EURUSD_1h.csv.gz"
    path.write_bytes(b"x")
    found = find_candle_path("EURUSD", str(tmp_path))
    assert found == str(path)
    assert find_candle_path("NOPE", str(tmp_path)) is None


def test_too_few_events_error_carries_counts():
    err = TooFewEventsError(7, 20)
    assert err.n_events == 7
    assert err.min_events == 20
    assert "7" in str(err)


def test_gpu_train_remote_script_has_universe_commands():
    script = (ROOT / "scripts" / "gpu_train_remote.sh").read_text(encoding="utf-8")
    assert "train-universe)" in script
    assert "download)" in script
    assert "--asset-class" in script


def test_manage_sh_universe_commands():
    script = (ROOT / "jesse_quant" / "manage.sh").read_text(encoding="utf-8")
    assert "gpu-train-universe)" in script
    assert "gpu-download-universe)" in script
    assert "download_ohlcv.py" in script
