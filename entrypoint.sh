#!/bin/bash
set -euo pipefail

# Retain the accepted refresh namespace and the original real accounting root.
# Root owns startup only; both Python owners run under the existing app account.
REFRESH_ROOT="$(realpath -m -- "${UFC_INCREMENTAL_REFRESH_ROOT:-/app/logs/incremental_refresh}")"
LIVE_MODE="${LIVE_TRADING_MODE:-off}"
case "$LIVE_MODE" in
    off) STATE_NAMESPACE=/app/logs/refresh_trading_paper ;;
    real) STATE_NAMESPACE=/app/logs/refresh_trading_live ;;
    *) echo "Combined service requires LIVE_TRADING_MODE=off or real" >&2; exit 1 ;;
esac
STATE_ROOT="$(realpath -m -- "${UFC_REFRESH_TRADING_STATE_DIR:-$STATE_NAMESPACE}")"
case "$REFRESH_ROOT" in
    /app/logs/incremental_refresh|/app/logs/incremental_refresh/*) ;;
    *) echo "Refresh state must be inside its persistent namespace" >&2; exit 1 ;;
esac
case "$STATE_ROOT" in
    "$STATE_NAMESPACE"|"$STATE_NAMESPACE"/*) ;;
    *) echo "Consumer state must be inside its mode-specific persistent namespace" >&2; exit 1 ;;
esac
REFRESH_PORT="${PORT:-8080}"
if [[ ! "$REFRESH_PORT" =~ ^[0-9]{1,5}$ ]] || ((10#$REFRESH_PORT < 1 || 10#$REFRESH_PORT > 65535)); then
    echo "Service port must be an integer from 1 to 65535" >&2
    exit 1
fi
TOKEN_PRESENT="${WEB_DASHBOARD_TOKEN:-}"
if [ -z "${TOKEN_PRESENT//[[:space:]]/}" ]; then
    echo "WEB_DASHBOARD_TOKEN is required for the public combined service" >&2
    exit 1
fi
unset TOKEN_PRESENT
# Check controls before seed/state writes. The Python supervisor checks them again
# under UID 100, including actual model and accounting write permissions.
PYTHONPATH=/app python -c 'from src.combined_service import require_combined_startup; require_combined_startup(host="0.0.0.0")'
if [ "$LIVE_MODE" = off ]; then
    mkdir -p "$REFRESH_ROOT" "$STATE_ROOT/logs" "$STATE_ROOT/data"
else
    mkdir -p "$REFRESH_ROOT" "$STATE_ROOT"
fi
# The inherited installer preserves the seed marker and every valid saved state.
python /app/scripts/install_incremental_seed.py --seed /app/refresh_seed --manifest /app/refresh_seed_manifest.json --target "$REFRESH_ROOT"
chown -hR -P app:app "$REFRESH_ROOT" "$STATE_ROOT"
export PYTHONPATH=/app
export UFC_INCREMENTAL_REFRESH_ROOT="$REFRESH_ROOT"
export UFC_REFRESH_TRADING_STATE_DIR="$STATE_ROOT"
export UFC_INCREMENTAL_REFRESH_PORT="$REFRESH_PORT"
exec su app -s /bin/sh -c 'cd /app && exec python -m src.combined_service --root "$UFC_INCREMENTAL_REFRESH_ROOT" --baseline /app/release_payload --state-dir "$UFC_REFRESH_TRADING_STATE_DIR" --real-logs-dir /app/logs --host 0.0.0.0 --port "$UFC_INCREMENTAL_REFRESH_PORT"'
