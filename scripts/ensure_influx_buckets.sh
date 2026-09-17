#!/usr/bin/env bash
# Create InfluxDB buckets used by the trading platform (idempotent).
set -euo pipefail

CONTAINER="${INFLUX_CONTAINER:-vps-influxdb}"
ORG="${INFLUXDB_ORG:-hedge-fund}"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "InfluxDB container $CONTAINER is not running — skip bucket setup"
  exit 0
fi

TOKEN="${INFLUXDB_TOKEN:-}"
if [[ -z "$TOKEN" && -f .env ]]; then
  TOKEN=$(grep '^INFLUXDB_TOKEN=' .env | cut -d= -f2- || true)
fi
if [[ -z "$TOKEN" ]]; then
  echo "WARN: INFLUXDB_TOKEN not set — cannot create buckets"
  exit 0
fi

BUCKETS=(
  "trading-system:90d"
  "trading-signals:90d"
  "trading-orders:365d"
  "trading-raw:30d"
  "trading-memory:0"
  "news-sentiment:90d"
)

for spec in "${BUCKETS[@]}"; do
  name="${spec%%:*}"
  retention="${spec##*:}"
  if docker exec "$CONTAINER" influx bucket list --org "$ORG" --token "$TOKEN" --name "$name" >/dev/null 2>&1; then
    # Bucket exists — reconcile retention (create alone would leave a stale
    # shorter retention, e.g. trading-system was 7d before 2026-09).
    echo "  bucket exists: $name — ensuring retention=${retention}"
    bucket_id=$(docker exec "$CONTAINER" influx bucket list \
      --org "$ORG" --token "$TOKEN" --name "$name" --hide-headers | awk '{print $1}' | head -n1)
    if [[ -n "$bucket_id" ]]; then
      docker exec "$CONTAINER" influx bucket update \
        --org "$ORG" --token "$TOKEN" \
        --id "$bucket_id" --retention "${retention}" || true
    fi
  else
    echo "  creating bucket: $name (retention=${retention})"
    docker exec "$CONTAINER" influx bucket create \
      --org "$ORG" --token "$TOKEN" \
      --name "$name" --retention "${retention}" || true
  fi
done
echo "Influx buckets OK"
