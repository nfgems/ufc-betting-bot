#!/bin/bash
set -euo pipefail
# Keep the runtime log/cache directory aligned with Railway's mounted volume when present.
# Safe to run repeatedly: only copy files from legacy locations when the target is missing.

PERSISTENT_LOG_DIR="${UFC_LOGS_DIR:-${RAILWAY_VOLUME_MOUNT_PATH:-/app/data/logs}}"
export UFC_LOGS_DIR="$PERSISTENT_LOG_DIR"

copy_if_missing() {
    src="$1"; dst="$2"
    if [ "$src" = "$dst" ]; then
        return
    fi
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
        if cmp -s "$src" "$dst"; then
            echo "[migrate] $src -> $dst"
        else
            echo "[migrate] FAILED integrity check: $src -> $dst" >&2
            rm -f "$dst"
        fi
    fi
}

copy_log_file() {
    name="$1"
    copy_if_missing "/app/data/logs/$name" "$PERSISTENT_LOG_DIR/$name"
    copy_if_missing "/app/logs/$name" "$PERSISTENT_LOG_DIR/$name"
}

# Ledgers & logs
copy_log_file bet_ledger.json
copy_log_file bet_ledger_single.json
copy_log_file bet_ledger_conviction.json
copy_log_file pnl_history.jsonl
copy_log_file orders.jsonl
copy_log_file positions.jsonl
copy_log_file latest_signals.json
copy_log_file predictions_cache.json
copy_log_file bot.log

# Models - keep the current hosted fallback behavior intact.
copy_if_missing /app/models/xgboost_model.pkl /app/data/models/xgboost_model.pkl
copy_if_missing /app/models/logistic_model.pkl /app/data/models/logistic_model.pkl
copy_if_missing /app/models/xgboost_no_odds_model.pkl /app/data/models/xgboost_no_odds_model.pkl

echo "[migrate] done"

# Ensure the log file exists without truncating prior deployment history.
mkdir -p "$PERSISTENT_LOG_DIR"
chown -R app:app "$PERSISTENT_LOG_DIR"
touch "$PERSISTENT_LOG_DIR/bot.log"
chown app:app "$PERSISTENT_LOG_DIR/bot.log"

# Start the app
exec su app -s /bin/sh -c "python -m src.web.serve"
