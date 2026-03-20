#!/bin/bash
set -euo pipefail
# One-time migration: move files from old paths to new volume-backed paths.
# Safe to run repeatedly - only moves files that exist at the old location.

migrate() {
    src="$1"; dst="$2"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
        if cmp -s "$src" "$dst"; then
            rm "$src"
            echo "[migrate] $src -> $dst"
        else
            echo "[migrate] FAILED integrity check: $src -> $dst" >&2
            rm -f "$dst"
        fi
    fi
}

# Ledgers & logs
migrate /app/logs/bet_ledger.json /app/data/logs/bet_ledger.json
migrate /app/logs/bet_ledger_single.json /app/data/logs/bet_ledger_single.json
migrate /app/logs/bet_ledger_conviction.json /app/data/logs/bet_ledger_conviction.json
migrate /app/logs/pnl_history.jsonl /app/data/logs/pnl_history.jsonl
migrate /app/logs/orders.jsonl /app/data/logs/orders.jsonl
migrate /app/logs/positions.jsonl /app/data/logs/positions.jsonl
migrate /app/logs/latest_signals.json /app/data/logs/latest_signals.json
migrate /app/logs/predictions_cache.json /app/data/logs/predictions_cache.json
migrate /app/logs/bot.log /app/data/logs/bot.log

# Models
migrate /app/data/models/xgboost_model.pkl /app/models/xgboost_model.pkl
migrate /app/data/models/logistic_model.pkl /app/models/logistic_model.pkl
migrate /app/data/models/xgboost_no_odds_model.pkl /app/models/xgboost_no_odds_model.pkl

echo "[migrate] done"

# Ensure the log file exists without truncating prior deployment history
mkdir -p /app/data/logs
touch /app/data/logs/bot.log

# Start the app
exec python -m src.web.serve
