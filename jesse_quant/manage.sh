#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker/docker-compose.yml"
API_URL="http://127.0.0.1:9000"

# Source environment
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

get_token() {
    curl -s -X POST "$API_URL/auth/login" \
        -H "Content-Type: application/json" \
        -d "{\"password\": \"${PASSWORD:?JESSE_PASSWORD / PASSWORD must be set}\"}" | jq -r '.auth_token // empty'
}

show_help() {
    echo "========================================================"
    echo "       Jesse AI Institutional Quant Platform v2.0"
    echo "========================================================"
    echo "Usage: ./manage.sh [command]"
    echo ""
    echo "Platform Management:"
    echo "  start               Start Jesse stack (Dashboard on :9000, MCP on :9002)"
    echo "  stop                Stop Jesse stack"
    echo "  restart             Restart Jesse stack"
    echo "  logs                View live logs from Jesse container"
    echo "  status              Check container and service health (Dashboard, MCP, ML)"
    echo ""
    echo "Data & Quant Research:"
    echo "  candles             List existing downloaded candlestick ranges"
    echo "  import-candles      Download historical candles for a specific pair"
    echo "                      Example: ./manage.sh import-candles 'Binance Perpetual Futures' 'BTC-USDT' '2024-01-01'"
    echo "  import-universe     Batch download candles for top assets (SOL, BNB, XRP, LINK, AVAX)"
    echo "                      Example: ./manage.sh import-universe 2024-01-01"
    echo "  optimize            Run Genetic Algorithm optimizer on a strategy (with DSR gate)"
    echo "                      Example: ./manage.sh optimize QuantumAIStrategy BTC-USDT"
    echo "  monte-carlo         Run Monte Carlo simulation on backtest results (true bootstrap)"
    echo "                      Example: ./manage.sh monte-carlo"
    echo "  backtest            Run backtest (CLI)"
    echo "                      Example: ./manage.sh backtest QuantumAIStrategy"
    echo "  run-validation      Run institutional statistical validation suite (DSR, PBO, PurgedKFold)"
    echo ""
    echo "Institutional Machine Learning & MLOps:"
    echo "  train-tb-ml         Train Triple-Barrier calibrated model (PurgedKFold + Uniqueness weights)"
    echo "                      Example: ./manage.sh train-tb-ml BTC-USDT 1h lightgbm 5.5 1.75"
    echo "  gpu-inventory     Probe NVIDIA / CUDA / LightGBM device (GPU training node or host)"
    echo "  gpu-train         Train LightGBM+LSTM on dumped candles with live 5.5/1.75 geometry"
    echo "                      Example: ./manage.sh gpu-train BTC-USDT 1h both storage/candles/BTC-USDT_1m.csv.gz"
    echo "  gpu-download-universe  Fetch public OHLCV for forex/metals/minerals/stocks/extra crypto"
    echo "                      Example: ./manage.sh gpu-download-universe forex"
    echo "  gpu-train-universe     Batch-train one or all asset classes (no every-bar fallback)"
    echo "                      Example: ./manage.sh gpu-train-universe metals both"
    echo "  gpu-auto-retrain-promote  GPU retrain; refuse to promote if DSR/PBO/recall gates fail"
    echo "  train-ml            Train direction ML model (standard forward returns)"
    echo "                      Example: ./manage.sh train-ml BTC-USDT 1h lightgbm"
    echo "  predict-ml          Run ML direction prediction via REST microservice"
    echo "                      Example: ./manage.sh predict-ml BTC-USDT 1h"
    echo "  meta-predict        Run secondary meta-model filter & Fractional Kelly sizing test"
    echo "                      Example: ./manage.sh meta-predict BTC-USDT SELL"
    echo "  verify-parity       Verify feature schema hash parity against active code"
    echo "  ml-status           Check ML inference engine status and model inventory"
    echo "  clear-cache         Clear in-memory model cache in predict microservice"
    echo "  finmem-eval         Run FINMEM immediate reflection (layered memory + LLM reasoning)"
    echo "                      Example: ./manage.sh finmem-eval BTC-USDT"
    echo "  finmem-status       Check active risk character and rolling memory stats"
    echo "                      Example: ./manage.sh finmem-status BTC-USDT"
    echo "  finmem-ingest       Store news or report into shallow/intermediate/deep layer"
    echo "                      Example: ./manage.sh finmem-ingest BTC-USDT shallow 'US ETF inflows surge'"
    echo ""
    echo "Utilities:"
    echo "  strategies          List available trading strategies"
    echo "  shell               Open a bash shell inside the Jesse container"
    echo "  help                Show this help message"
    echo "========================================================"
}

case "$1" in
    start)
        echo "[*] Starting Jesse stack..."
        docker compose -f "$COMPOSE_FILE" up -d
        echo "[✓] Jesse started successfully!"
        echo "    Dashboard:  http://127.0.0.1:9000"
        echo "    MCP Server: http://127.0.0.1:9002/mcp (or http://jesse-app:9002/mcp)"
        echo "    ML Engine:  http://127.0.0.1:9003"
        ;;
    stop)
        echo "[*] Stopping Jesse stack..."
        docker compose -f "$COMPOSE_FILE" down
        echo "[✓] Jesse stopped."
        ;;
    restart)
        echo "[*] Restarting Jesse stack..."
        docker compose -f "$COMPOSE_FILE" restart
        echo "[✓] Restart complete."
        ;;
    logs)
        docker compose -f "$COMPOSE_FILE" logs -f --tail=100 jesse
        ;;
    status)
        echo "=== Jesse Container Status ==="
        docker compose -f "$COMPOSE_FILE" ps
        echo ""
        echo "=== Health Probes ==="
        DASH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -m 3 "$API_URL/" || echo "DOWN")
        MCP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -m 3 "http://127.0.0.1:9002/mcp" || echo "DOWN")
        ML_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -m 3 "http://127.0.0.1:9003/health" || echo "DOWN")
        echo "Dashboard (Port 9000): HTTP $DASH_STATUS"
        echo "MCP Server (Port 9002): HTTP $MCP_STATUS (SSE protocol active)"
        echo "ML Engine  (Port 9003): HTTP $ML_STATUS (Institutional v2.0 active)"
        ;;
    strategies)
        TOKEN=$(get_token)
        if [ -n "$TOKEN" ]; then
            echo "[*] Available Strategies in Jesse:"
            curl -s "$API_URL/strategy/all" -H "Authorization: $TOKEN" | jq '.strategies'
        else
            echo "[!] Could not authenticate to Jesse API."
        fi
        ;;
    candles)
        TOKEN=$(get_token)
        if [ -n "$TOKEN" ]; then
            echo "[*] Existing Candlestick Datasets:"
            curl -s -X POST "$API_URL/candles/existing" -H "Authorization: $TOKEN" | jq '.data'
        else
            echo "[!] Could not authenticate to Jesse API."
        fi
        ;;
    import-candles)
        EXCHANGE="${2:-Binance Perpetual Futures}"
        if [ "$EXCHANGE" = "Binance Futures" ]; then
            EXCHANGE="Binance Perpetual Futures"
        fi
        SYMBOL="${3:-BTC-USDT}"
        START_DATE="${4:-2026-09-01}"
        TOKEN=$(get_token)
        if [ -z "$TOKEN" ]; then
            echo "[!] Authentication failed."
            exit 1
        fi
        IMPORT_ID="import-$(date +%s)"
        echo "[*] Triggering Candle Import: Exchange='$EXCHANGE', Symbol='$SYMBOL', StartDate='$START_DATE'..."
        RESP=$(curl -s -X POST "$API_URL/candles/import" \
            -H "Authorization: $TOKEN" \
            -H "Content-Type: application/json" \
            -d "{\"id\": \"$IMPORT_ID\", \"exchange\": \"$EXCHANGE\", \"symbol\": \"$SYMBOL\", \"start_date\": \"$START_DATE\"}")
        echo "[✓] Response: $RESP"
        ;;
    import-universe)
        START_DATE="${2:-2024-01-01}"
        python3 "$SCRIPT_DIR/import_universe.py" --start "$START_DATE"
        ;;
    gpu-inventory)
        PYTHON_BIN="${JESSE_GPU_PYTHON:-}"
        if [ -z "$PYTHON_BIN" ] && [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
            PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
        fi
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi
        else
            echo "[!] nvidia-smi not on PATH"
        fi
        if [ -n "$PYTHON_BIN" ]; then
            "$PYTHON_BIN" "$SCRIPT_DIR/gpu_device.py"
        elif command -v docker >/dev/null 2>&1; then
            docker exec jesse-app python3 /home/gpu_device.py || echo "[!] gpu_device.py not mounted in jesse-app"
        else
            python3 "$SCRIPT_DIR/gpu_device.py"
        fi
        ;;
    gpu-train)
        SYMBOL="${2:-BTC-USDT}"
        TIMEFRAME="${3:-1h}"
        MODEL="${4:-both}"
        CANDLES="${5:-$SCRIPT_DIR/storage/candles/${SYMBOL}_1m.csv.gz}"
        PYTHON_BIN="${JESSE_GPU_PYTHON:-}"
        if [ -z "$PYTHON_BIN" ] && [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
            PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
        fi
        if [ -z "$PYTHON_BIN" ]; then
            PYTHON_BIN="python3"
        fi
        echo "[*] GPU train $SYMBOL $TIMEFRAME model=$MODEL candles=$CANDLES"
        "$PYTHON_BIN" "$SCRIPT_DIR/train_gpu.py" \
            --symbol "$SYMBOL" \
            --timeframe "$TIMEFRAME" \
            --model "$MODEL" \
            --candles "$CANDLES" \
            --pt-mult 5.5 \
            --sl-mult 1.75
        ;;
    gpu-download-universe)
        CLASS="${2:-all}"
        PYTHON_BIN="${JESSE_GPU_PYTHON:-}"
        if [ -z "$PYTHON_BIN" ] && [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
            PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
        fi
        if [ -z "$PYTHON_BIN" ]; then
            PYTHON_BIN="python3"
        fi
        echo "[*] Download OHLCV universe class=$CLASS"
        "$PYTHON_BIN" "$SCRIPT_DIR/download_ohlcv.py" \
            --asset-class "$CLASS" \
            --out "$SCRIPT_DIR/storage/candles"
        ;;
    gpu-train-universe)
        CLASS="${2:-all}"
        MODEL="${3:-both}"
        PYTHON_BIN="${JESSE_GPU_PYTHON:-}"
        if [ -z "$PYTHON_BIN" ] && [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
            PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
        fi
        if [ -z "$PYTHON_BIN" ]; then
            PYTHON_BIN="python3"
        fi
        echo "[*] GPU universe train class=$CLASS model=$MODEL (5.5/1.75 ATR, fail-closed)"
        "$PYTHON_BIN" "$SCRIPT_DIR/train_gpu.py" \
            --asset-class "$CLASS" \
            --candles-dir "$SCRIPT_DIR/storage/candles" \
            --timeframe 1h \
            --model "$MODEL" \
            --pt-mult 5.5 \
            --sl-mult 1.75
        ;;
    gpu-auto-retrain-promote)
        SYMBOL="${2:-BTC-USDT}"
        TIMEFRAME="${3:-1h}"
        MODEL="${4:-both}"
        CANDLES="${5:-$SCRIPT_DIR/storage/candles/${SYMBOL}_1m.csv.gz}"
        PYTHON_BIN="${JESSE_GPU_PYTHON:-}"
        if [ -z "$PYTHON_BIN" ] && [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
            PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
        fi
        if [ -z "$PYTHON_BIN" ]; then
            PYTHON_BIN="python3"
        fi
        echo "[*] GPU auto-retrain with live 5.5/1.75 ATR geometry and DSR/PBO promotion gates..."
        set +e
        "$PYTHON_BIN" "$SCRIPT_DIR/train_gpu.py" \
            --symbol "$SYMBOL" \
            --timeframe "$TIMEFRAME" \
            --model "$MODEL" \
            --candles "$CANDLES" \
            --pt-mult 5.5 \
            --sl-mult 1.75
        PROMOTE_RC=$?
        set -e
        if [ "$PROMOTE_RC" -ne 0 ]; then
            echo "[!] Promotion gate failed (exit $PROMOTE_RC) — production artifact unchanged"
            exit "$PROMOTE_RC"
        fi
        echo "[*] Reloading inference cache if local ML engine is up..."
        curl -s -X POST "http://127.0.0.1:9003/cache/clear" | jq . || true
        ;;
    train-tb-ml)
        SYMBOL="${2:-BTC-USDT}"
        TIMEFRAME="${3:-1h}"
        MODEL="${4:-lightgbm}"
        PT="${5:-5.5}"
        SL="${6:-1.75}"
        echo "[*] Launching Triple-Barrier Calibrated Quant ML Trainer (live geometry PT=${PT} SL=${SL})..."
        docker exec jesse-app python3 /home/train_ml.py \
            --symbol "$SYMBOL" \
            --timeframe "$TIMEFRAME" \
            --model "$MODEL" \
            --labeling triple_barrier \
            --pt-mult "$PT" \
            --sl-mult "$SL"
        ;;
    auto-retrain-promote)
        SYMBOL="${2:-BTC-USDT}"
        TIMEFRAME="${3:-1h}"
        MODEL="${4:-lightgbm}"
        echo "[*] Auto-retrain with live 5.5/1.75 ATR geometry and DSR/PBO promotion gates..."
        docker exec jesse-app python3 /home/train_ml.py \
            --symbol "$SYMBOL" \
            --timeframe "$TIMEFRAME" \
            --model "$MODEL" \
            --labeling triple_barrier \
            --pt-mult 5.5 \
            --sl-mult 1.75
        PROMOTE_RC=$?
        if [ "$PROMOTE_RC" -ne 0 ]; then
            echo "[!] Promotion gate failed (exit $PROMOTE_RC) — production artifact unchanged"
            exit "$PROMOTE_RC"
        fi
        echo "[*] Reloading inference cache..."
        curl -s -X POST "http://127.0.0.1:9003/cache/clear" | jq .
        ;;
    train-ml)
        SYMBOL="${2:-BTC-USDT}"
        TIMEFRAME="${3:-1h}"
        MODEL="${4:-lightgbm}"
        echo "[*] Launching Jesse ML Training Engine..."
        docker exec jesse-app python3 /home/train_ml.py --symbol "$SYMBOL" --timeframe "$TIMEFRAME" --model "$MODEL"
        ;;
    predict-ml)
        SYMBOL="${2:-BTC-USDT}"
        TIMEFRAME="${3:-1h}"
        MODEL="${4:-lightgbm}"
        curl -s "http://127.0.0.1:9003/predict?symbol=$SYMBOL&timeframe=$TIMEFRAME&model_type=$MODEL" | jq .
        ;;
    meta-predict)
        SYMBOL="${2:-BTC-USDT}"
        SIGNAL="${3:-SELL}"
        TIMEFRAME="${4:-1h}"
        curl -s -X POST "http://127.0.0.1:9003/meta-predict" \
            -H "Content-Type: application/json" \
            -d "{\"symbol\":\"$SYMBOL\",\"primary_signal\":\"$SIGNAL\",\"timeframe\":\"$TIMEFRAME\"}" | jq .
        ;;
    verify-parity)
        echo "[*] Verifying Feature Schema Parity..."
        docker exec jesse-app python3 /home/feature_schema.py
        ;;
    run-validation)
        echo "[*] Running Statistical Validation Test Suite (DSR, PBO, PurgedKFold)..."
        docker exec jesse-app python3 /home/test_validation_metrics.py
        ;;
    ml-status)
        echo "=== Jesse ML Prediction Microservice (:9003) ==="
        curl -s "http://127.0.0.1:9003/health" | jq .
        ;;
    clear-cache)
        echo "[*] Clearing Jesse ML In-Memory Model Cache..."
        curl -s -X POST "http://127.0.0.1:9003/cache/clear" | jq .
        ;;
    finmem-eval)
        SYMBOL="${2:-BTC-USDT}"
        echo "[*] Triggering FINMEM Immediate Reflection for $SYMBOL..."
        curl -s -X POST http://localhost:8001/api/jesse/finmem/evaluate \
            -H "Content-Type: application/json" \
            -H "x-api-key: ${ADMIN_API_KEY:?ADMIN_API_KEY must be set}" \
            -d "{\"symbol\":\"$SYMBOL\",\"timeframe\":\"1h\"}" | jq .
        ;;
    finmem-status)
        SYMBOL="${2:-BTC-USDT}"
        curl -s "http://localhost:8001/api/jesse/finmem/status?symbol=$SYMBOL" | jq .
        ;;
    finmem-ingest)
        SYMBOL="${2:-BTC-USDT}"
        LAYER="${3:-shallow}"
        CONTENT="${4:-}"
        if [ -z "$CONTENT" ]; then
            echo "[!] Usage: ./manage.sh finmem-ingest [SYMBOL] [shallow|intermediate|deep] '[CONTENT]'"
            exit 1
        fi
        curl -s -X POST http://localhost:8001/api/jesse/finmem/ingest \
            -H "Content-Type: application/json" \
            -H "x-api-key: ${ADMIN_API_KEY:?ADMIN_API_KEY must be set}" \
            -d "{\"symbol\":\"$SYMBOL\",\"layer\":\"$LAYER\",\"content\":\"$CONTENT\"}" | jq .
        ;;
    optimize)
        STRATEGY="${2:-QuantumAIStrategy}"
        SYMBOL="${3:-BTC-USDT}"
        python3 "$SCRIPT_DIR/run_optimizer.py" "$STRATEGY" --symbol "$SYMBOL"
        ;;
    monte-carlo)
        SESSION_ID="${2:-}"
        python3 "$SCRIPT_DIR/run_monte_carlo.py" $SESSION_ID
        ;;
    backtest)
        STRATEGY="${2:-QuantumAIStrategy}"
        python3 "$SCRIPT_DIR/run_backtest.py" "$STRATEGY"
        ;;
    shell)
        docker compose -f "$COMPOSE_FILE" exec jesse bash
        ;;
    *)
        show_help
        ;;
esac
