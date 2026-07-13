#!/bin/bash
set -euo pipefail
# Keep the runtime log/cache directory aligned with Railway's mounted volume when present.
# Safe to run repeatedly: only copy files from legacy locations when the target is missing.

PERSISTENT_DATA_DIR="${UFC_DATA_DIR:-${RAILWAY_VOLUME_MOUNT_PATH:-/app/data}}"
PERSISTENT_LOG_DIR="${UFC_LOGS_DIR:-$PERSISTENT_DATA_DIR/logs}"
LEGACY_DUPLICATE_LOG_DIR=""
if [ -n "${RAILWAY_VOLUME_MOUNT_PATH:-}" ] \
    && [ "$PERSISTENT_DATA_DIR" = "$RAILWAY_VOLUME_MOUNT_PATH" ] \
    && [ "$PERSISTENT_LOG_DIR" = "$PERSISTENT_DATA_DIR/logs" ]; then
    LEGACY_DUPLICATE_LOG_DIR="$PERSISTENT_LOG_DIR"
    PERSISTENT_LOG_DIR="$PERSISTENT_DATA_DIR"
    echo "[startup] normalized duplicate hosted log dir: $LEGACY_DUPLICATE_LOG_DIR -> $PERSISTENT_LOG_DIR"
fi
LEGACY_MODELS_OVERRIDE="${UFC_MODELS_DIR:-}"
LEGACY_BUNDLE_MANIFEST_OVERRIDE="${UFC_PRODUCTION_BUNDLE_MANIFEST:-}"
# Railway hosted runtime always serves models from the image bundle. Do not
# honor legacy env overrides that point back at the persistent volume.
ACTIVE_MODEL_DIR="/app/models"
PRODUCTION_BUNDLE_MANIFEST="$PERSISTENT_DATA_DIR/production_bundle/current/manifest.json"
IMAGE_PRODUCTION_BUNDLE_MANIFEST="/app/models/current_production_model.json"
export UFC_DATA_DIR="$PERSISTENT_DATA_DIR"
export UFC_LOGS_DIR="$PERSISTENT_LOG_DIR"
export UFC_MODELS_DIR="$ACTIVE_MODEL_DIR"
export UFC_PRODUCTION_BUNDLE_MANIFEST="$PRODUCTION_BUNDLE_MANIFEST"

normalize_legacy_log_path_var() {
    var_name="$1"
    if [ -z "$LEGACY_DUPLICATE_LOG_DIR" ]; then
        return
    fi
    raw_value="${!var_name:-}"
    if [ -z "$raw_value" ]; then
        return
    fi
    case "$raw_value" in
        "$LEGACY_DUPLICATE_LOG_DIR"/*)
            fixed_value="$PERSISTENT_LOG_DIR/${raw_value#$LEGACY_DUPLICATE_LOG_DIR/}"
            export "$var_name=$fixed_value"
            echo "[startup] normalized $var_name from duplicate hosted log dir: $raw_value -> $fixed_value"
            ;;
    esac
}

normalize_legacy_log_path_var BTC5M_LEDGER_PATH
normalize_legacy_log_path_var BTC5M_SIGNAL_LOG_PATH
normalize_legacy_log_path_var BTC5M_EXIT_SHADOW_LOG_PATH
normalize_legacy_log_path_var BTC5M_PAPER_LEDGER_DIR
normalize_legacy_log_path_var BTC5M_LIVE_LEDGER_DIR

if [ -n "$LEGACY_MODELS_OVERRIDE" ] && [ "$LEGACY_MODELS_OVERRIDE" != "$ACTIVE_MODEL_DIR" ]; then
    echo "[startup] ignoring legacy UFC_MODELS_DIR override: $LEGACY_MODELS_OVERRIDE" >&2
fi
if [ -n "$LEGACY_BUNDLE_MANIFEST_OVERRIDE" ] && [ "$LEGACY_BUNDLE_MANIFEST_OVERRIDE" != "$PRODUCTION_BUNDLE_MANIFEST" ]; then
    echo "[startup] ignoring legacy UFC_PRODUCTION_BUNDLE_MANIFEST override: $LEGACY_BUNDLE_MANIFEST_OVERRIDE" >&2
fi

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
    if [ -n "$LEGACY_DUPLICATE_LOG_DIR" ]; then
        copy_if_missing "$LEGACY_DUPLICATE_LOG_DIR/$name" "$PERSISTENT_LOG_DIR/$name"
    fi
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
copy_log_file alerts.jsonl

if [ -n "$LEGACY_DUPLICATE_LOG_DIR" ]; then
    copy_tree_missing "$LEGACY_DUPLICATE_LOG_DIR/btc5m_live" "$PERSISTENT_LOG_DIR/btc5m_live"
    copy_tree_missing "$LEGACY_DUPLICATE_LOG_DIR/btc5m_paper" "$PERSISTENT_LOG_DIR/btc5m_paper"
    copy_tree_missing "$LEGACY_DUPLICATE_LOG_DIR/btc5m_opportunity" "$PERSISTENT_LOG_DIR/btc5m_opportunity"
    copy_tree_missing "$LEGACY_DUPLICATE_LOG_DIR/signals" "$PERSISTENT_LOG_DIR/signals"
    copy_tree_missing "$LEGACY_DUPLICATE_LOG_DIR/plots" "$PERSISTENT_LOG_DIR/plots"
fi

# Seed raw inputs when missing. The canonical hosted processed snapshot is
# bootstrapped below via the runtime production-bundle manifest.
copy_tree_missing /app/data/raw "$PERSISTENT_DATA_DIR/raw"
copy_tree_missing /app/data/operator "$PERSISTENT_DATA_DIR/operator"

# Key enrichment files: update the volume copy from the image when the image
# has a larger (more enriched) version.  The copy_tree_missing helper above
# only seeds files that don't exist on the volume, so locally-enriched data
# from newer image builds never reaches the volume.  This block fixes that
# for critical profile files whose staleness directly causes false coverage
# alerts.
update_if_image_larger() {
    src="$1"; dst="$2"
    if [ "$src" = "$dst" ] || [ ! -f "$src" ]; then
        return
    fi
    if [ ! -f "$dst" ]; then
        # Already handled by copy_tree_missing
        return
    fi
    src_size=$(stat -c%s "$src" 2>/dev/null || stat -f%z "$src" 2>/dev/null || echo 0)
    dst_size=$(stat -c%s "$dst" 2>/dev/null || stat -f%z "$dst" 2>/dev/null || echo 0)
    src_rows=$(tail -n +2 "$src" 2>/dev/null | wc -l | tr -d ' ')
    dst_rows=$(tail -n +2 "$dst" 2>/dev/null | wc -l | tr -d ' ')
    if [ "${dst_rows:-0}" -gt "${src_rows:-0}" ]; then
        echo "[seed] kept volume file with more CSV rows than image ($dst_rows > $src_rows): $dst"
        return
    fi
    if [ "${src_rows:-0}" -gt "${dst_rows:-0}" ] || [ "$src_size" -gt "$dst_size" ]; then
        cp "$src" "$dst"
        echo "[seed] updated stale volume file from image (${dst_rows:-0} -> ${src_rows:-0} rows, $dst_size -> $src_size bytes): $dst"
    fi
}

update_official_roster_if_volume_oversized() {
    src="$1"; dst="$2"
    if [ "$src" = "$dst" ] || [ ! -f "$src" ] || [ ! -f "$dst" ]; then
        return
    fi
    src_size=$(stat -c%s "$src" 2>/dev/null || stat -f%z "$src" 2>/dev/null || echo 0)
    dst_size=$(stat -c%s "$dst" 2>/dev/null || stat -f%z "$dst" 2>/dev/null || echo 0)
    src_rows=$(tail -n +2 "$src" 2>/dev/null | wc -l | tr -d ' ')
    dst_rows=$(tail -n +2 "$dst" 2>/dev/null | wc -l | tr -d ' ')
    if [ "${src_rows:-0}" -lt 100 ] || [ "${dst_rows:-0}" -le "${src_rows:-0}" ]; then
        return
    fi

    ratio_allowed=$((src_rows * 135 / 100))
    absolute_allowed=$((src_rows + 250))
    max_allowed="$ratio_allowed"
    if [ "$absolute_allowed" -gt "$max_allowed" ]; then
        max_allowed="$absolute_allowed"
    fi
    if [ "$dst_rows" -gt "$max_allowed" ]; then
        cp "$src" "$dst"
        echo "[seed] replaced suspiciously oversized UFC active roster from image (${dst_rows} -> ${src_rows} rows, $dst_size -> $src_size bytes): $dst"
    fi
}

update_official_roster_if_volume_oversized /app/data/raw/ufc_active_roster_official.csv "$PERSISTENT_DATA_DIR/raw/ufc_active_roster_official.csv"
update_if_image_larger /app/data/raw/ufc_active_roster_official.csv "$PERSISTENT_DATA_DIR/raw/ufc_active_roster_official.csv"
update_if_image_larger /app/data/raw/ufc_fighters_scraped.csv "$PERSISTENT_DATA_DIR/raw/ufc_fighters_scraped.csv"
update_if_image_larger /app/data/raw/ufc-fighter-details.csv "$PERSISTENT_DATA_DIR/raw/ufc-fighter-details.csv"

echo "[migrate] done"

# Ensure the runtime user can update data and logs on the mounted volume.
mkdir -p "$PERSISTENT_DATA_DIR" "$PERSISTENT_DATA_DIR/raw" "$PERSISTENT_DATA_DIR/processed" "$PERSISTENT_DATA_DIR/operator" "$PERSISTENT_DATA_DIR/tmp" "$PERSISTENT_DATA_DIR/production_bundle/current" "$PERSISTENT_LOG_DIR" "$ACTIVE_MODEL_DIR"
chown app:app "$PERSISTENT_DATA_DIR"
chown -R app:app "$PERSISTENT_DATA_DIR/raw" "$PERSISTENT_DATA_DIR/processed" "$PERSISTENT_DATA_DIR/operator" "$PERSISTENT_DATA_DIR/tmp" "$PERSISTENT_DATA_DIR/production_bundle" "$PERSISTENT_LOG_DIR" "$ACTIVE_MODEL_DIR"

# Ensure the log files exist without truncating prior deployment history.
touch "$PERSISTENT_LOG_DIR/bot.log"
chown app:app "$PERSISTENT_LOG_DIR/bot.log"
touch "$PERSISTENT_LOG_DIR/alerts.jsonl"
chown app:app "$PERSISTENT_LOG_DIR/alerts.jsonl"

# Allow the runtime user to persist one-time model metadata repairs.
if [ -f "$ACTIVE_MODEL_DIR/xgboost_no_odds_model.pkl" ]; then
    if ! su app -s /bin/sh -c "cd /app && PYTHONPATH=/app python scripts/repair_no_odds_training_spec.py \"$ACTIVE_MODEL_DIR/xgboost_no_odds_model.pkl\""; then
        echo "[migrate] WARNING: failed to normalize no-odds training_spec metadata in $ACTIVE_MODEL_DIR/xgboost_no_odds_model.pkl" >&2
    fi
fi

if ! su app -s /bin/sh -c "cd /app && PYTHONPATH=/app python scripts/bootstrap_runtime_production_bundle.py --target-manifest \"$PRODUCTION_BUNDLE_MANIFEST\" --source-manifest \"$IMAGE_PRODUCTION_BUNDLE_MANIFEST\" --source-processed-dir \"/app/data/processed\" --target-processed-dir \"$PERSISTENT_DATA_DIR/processed\" --model-path \"$ACTIVE_MODEL_DIR/xgboost_model.pkl\" --no-odds-model-path \"$ACTIVE_MODEL_DIR/xgboost_no_odds_model.pkl\" --logistic-model-path \"$ACTIVE_MODEL_DIR/logistic_model.pkl\""; then
    echo "[startup] ERROR: failed to bootstrap production bundle manifest at $PRODUCTION_BUNDLE_MANIFEST" >&2
    exit 1
fi

APP_ROLE="${APP_ROLE:-web}"

# Reconcile any untracked Polymarket positions into the ledger
if [ "$APP_ROLE" = "web" ] && [ "${LIVE_TRADING_MODE:-}" = "real" ]; then
    echo "[startup] Reconciling Polymarket positions..."
    if ! su app -s /bin/sh -c "cd /app && PYTHONPATH=/app python scripts/reconcile_positions.py"; then
        echo "[startup] ERROR: position reconciliation failed; refusing to start real-money trading" >&2
        exit 1
    fi
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
