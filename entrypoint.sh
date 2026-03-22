#!/bin/bash
set -euo pipefail
# Keep the runtime log/cache directory aligned with Railway's mounted volume when present.
# Safe to run repeatedly: only copy files from legacy locations when the target is missing.

PERSISTENT_DATA_DIR="${UFC_DATA_DIR:-${RAILWAY_VOLUME_MOUNT_PATH:-/app/data}}"
PERSISTENT_LOG_DIR="${UFC_LOGS_DIR:-$PERSISTENT_DATA_DIR/logs}"
PERSISTENT_MODEL_DIR="${UFC_MODELS_DIR:-$PERSISTENT_DATA_DIR/models}"
export UFC_DATA_DIR="$PERSISTENT_DATA_DIR"
export UFC_LOGS_DIR="$PERSISTENT_LOG_DIR"
export UFC_MODELS_DIR="$PERSISTENT_MODEL_DIR"

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

copy_tree_missing() {
    src_root="$1"; dst_root="$2"
    if [ "$src_root" = "$dst_root" ] || [ ! -d "$src_root" ]; then
        return
    fi
    while IFS= read -r -d '' src_file; do
        rel_path="${src_file#$src_root/}"
        dst_file="$dst_root/$rel_path"
        if [ -f "$dst_file" ]; then
            continue
        fi
        mkdir -p "$(dirname "$dst_file")"
        cp "$src_file" "$dst_file"
        if cmp -s "$src_file" "$dst_file"; then
            echo "[migrate] $src_file -> $dst_file"
        else
            echo "[migrate] FAILED integrity check: $src_file -> $dst_file" >&2
            rm -f "$dst_file"
        fi
    done < <(find "$src_root" -type f -print0)
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
copy_if_missing /app/models/xgboost_model.pkl "$PERSISTENT_MODEL_DIR/xgboost_model.pkl"
copy_if_missing /app/models/logistic_model.pkl "$PERSISTENT_MODEL_DIR/logistic_model.pkl"
copy_if_missing /app/models/xgboost_no_odds_model.pkl "$PERSISTENT_MODEL_DIR/xgboost_no_odds_model.pkl"

# Seed the mounted data volume with the repo's baseline artifacts when missing.
copy_tree_missing /app/data/raw "$PERSISTENT_DATA_DIR/raw"
copy_tree_missing /app/data/processed "$PERSISTENT_DATA_DIR/processed"

echo "[migrate] done"

# Ensure the runtime user can update data, logs, and models on the mounted volume.
mkdir -p "$PERSISTENT_DATA_DIR" "$PERSISTENT_DATA_DIR/raw" "$PERSISTENT_DATA_DIR/processed" "$PERSISTENT_DATA_DIR/tmp" "$PERSISTENT_LOG_DIR" "$PERSISTENT_MODEL_DIR"
chown app:app "$PERSISTENT_DATA_DIR"
chown -R app:app "$PERSISTENT_DATA_DIR/raw" "$PERSISTENT_DATA_DIR/processed" "$PERSISTENT_DATA_DIR/tmp" "$PERSISTENT_LOG_DIR" "$PERSISTENT_MODEL_DIR"

# Ensure the log file exists without truncating prior deployment history.
touch "$PERSISTENT_LOG_DIR/bot.log"
chown app:app "$PERSISTENT_LOG_DIR/bot.log"

# Allow the runtime user to persist one-time model metadata repairs.
if [ -f "$PERSISTENT_MODEL_DIR/xgboost_no_odds_model.pkl" ]; then
    if ! su app -s /bin/sh -c "cd /app && PYTHONPATH=/app python scripts/repair_no_odds_training_spec.py \"$PERSISTENT_MODEL_DIR/xgboost_no_odds_model.pkl\""; then
        echo "[migrate] WARNING: failed to normalize no-odds training_spec metadata in $PERSISTENT_MODEL_DIR/xgboost_no_odds_model.pkl" >&2
    fi
fi

APP_ROLE="${APP_ROLE:-web}"

# Reconcile any untracked Polymarket positions into the ledger
if [ "$APP_ROLE" = "web" ] && [ "${LIVE_TRADING_MODE:-}" = "real" ]; then
    echo "[startup] Reconciling Polymarket positions..."
    su app -s /bin/sh -c "cd /app && PYTHONPATH=/app python scripts/reconcile_positions.py" || echo "[startup] WARNING: position reconciliation failed" >&2
fi

case "$APP_ROLE" in
    web)
        exec su app -s /bin/sh -c "cd /app && PYTHONPATH=/app python -m src.web.serve"
        ;;
    ufc-refresh-scheduled)
        exec su app -s /bin/sh -c "cd /app && PYTHONPATH=/app python -m src.bot ufc-refresh-scheduled"
        ;;
    *)
        echo "[startup] Unknown APP_ROLE: $APP_ROLE" >&2
        exit 1
        ;;
esac
