#!/bin/bash
set -euo pipefail
# Keep the runtime log/cache directory aligned with Railway's mounted volume when present.
# Safe to run repeatedly: legacy migrations copy missing files, while explicitly
# approved BFO recovery artifacts overlay only their same-name volume copies.

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
# A verified bundle store, when present, is authoritative. Its one atomic
# active pointer selects a complete immutable release; otherwise preserve the
# existing image-bundle startup path for backward-compatible first deployment.
BUNDLE_STORE_ROOT="$PERSISTENT_DATA_DIR/production_bundle/store"
ACTIVE_MODEL_DIR="/app/models"
SOURCE_PRODUCTION_BUNDLE_MANIFEST="/app/models/current_production_model.json"
SOURCE_PROCESSED_DIR="/app/data/processed"
ACTIVE_LOOKUP_DIR="$PERSISTENT_DATA_DIR/processed"
BUNDLE_STORE_ACTIVE=0
if [ -e "$BUNDLE_STORE_ROOT" ]; then
    if [ ! -f "$BUNDLE_STORE_ROOT/.production_bundle_store.json" ] \
        || [ ! -f "$BUNDLE_STORE_ROOT/active_bundle.json" ]; then
        echo "[startup] ERROR: incomplete production bundle store: $BUNDLE_STORE_ROOT" >&2
        exit 1
    fi
    if ! ACTIVE_BUNDLE_RESOLUTION="$(
        cd /app && PYTHONPATH=/app python scripts/install_staged_production_bundle.py \
            resolve \
            --target-root "$BUNDLE_STORE_ROOT"
    )"; then
        echo "[startup] ERROR: active production bundle store validation failed" >&2
        exit 1
    fi
    resolve_bundle_field() {
        printf '%s' "$ACTIVE_BUNDLE_RESOLUTION" \
            | python -c 'import json, sys; value = json.load(sys.stdin).get(sys.argv[1]); print(value) if isinstance(value, str) and value else sys.exit("missing bundle field")' "$1"
    }
    if ! SOURCE_PRODUCTION_BUNDLE_MANIFEST="$(resolve_bundle_field active_manifest_path)" \
        || ! ACTIVE_MODEL_DIR="$(resolve_bundle_field active_models_dir)" \
        || ! SOURCE_PROCESSED_DIR="$(resolve_bundle_field active_processed_dir)" \
        || ! ACTIVE_LOOKUP_DIR="$(resolve_bundle_field active_lookup_dir)"; then
        echo "[startup] ERROR: active bundle resolution omitted required runtime paths" >&2
        exit 1
    fi
    ACTIVE_RELEASE_DIR="$(dirname "$SOURCE_PRODUCTION_BUNDLE_MANIFEST")"
    BUNDLE_STORE_ACTIVE=1
    echo "[startup] selected verified atomic production release: $ACTIVE_RELEASE_DIR"
fi
if [ "$BUNDLE_STORE_ACTIVE" -eq 1 ]; then
    PRODUCTION_BUNDLE_MANIFEST="$ACTIVE_LOOKUP_DIR/runtime_manifest.json"
else
    PRODUCTION_BUNDLE_MANIFEST="$PERSISTENT_DATA_DIR/production_bundle/current/manifest.json"
fi
export UFC_DATA_DIR="$PERSISTENT_DATA_DIR"
export UFC_LOGS_DIR="$PERSISTENT_LOG_DIR"
export UFC_PROCESSED_DIR="$ACTIVE_LOOKUP_DIR"
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

atomic_copy_verified() {
    local src="$1"
    local dst="$2"
    local publish_mode="${3:-replace}"
    local dst_dir
    local dst_name
    local tmp_path
    dst_dir="$(dirname "$dst")"
    dst_name="$(basename "$dst")"
    mkdir -p "$dst_dir"
    tmp_path="$(mktemp "$dst_dir/.${dst_name}.tmp.XXXXXX")"
    if cp -p "$src" "$tmp_path" && cmp -s "$src" "$tmp_path"; then
        if [ "$publish_mode" = "no-replace" ]; then
            if [ -e "$dst" ] || [ -L "$dst" ]; then
                rm -f "$tmp_path"
                if [ -f "$dst" ] && [ ! -L "$dst" ]; then
                    return 2
                fi
                return 1
            fi
            # GNU mv's -n prevents a destination created after the check above
            # from being overwritten. -T requires the destination to be a file
            # path rather than treating a directory as a move target.
            if mv -nT "$tmp_path" "$dst" && [ ! -e "$tmp_path" ]; then
                return 0
            fi
            if [ -f "$dst" ] && [ ! -L "$dst" ]; then
                rm -f "$tmp_path"
                return 2
            fi
        elif [ "$publish_mode" = "replace" ]; then
            if mv -fT "$tmp_path" "$dst"; then
                return 0
            fi
        fi
    fi
    rm -f "$tmp_path"
    return 1
}

copy_if_missing() {
    src="$1"; dst="$2"
    if [ "$src" = "$dst" ]; then
        return
    fi
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        if atomic_copy_verified "$src" "$dst" no-replace; then
            echo "[migrate] $src -> $dst"
        else
            copy_status=$?
            if [ "$copy_status" -eq 2 ]; then
                echo "[migrate] kept destination published concurrently: $dst"
            else
                echo "[migrate] FAILED integrity check: $src -> $dst" >&2
                return 1
            fi
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
        if atomic_copy_verified "$src_file" "$dst_file" no-replace; then
            echo "[migrate] $src_file -> $dst_file"
        else
            copy_status=$?
            if [ "$copy_status" -eq 2 ]; then
                echo "[migrate] kept destination published concurrently: $dst_file"
            else
                echo "[migrate] FAILED integrity check: $src_file -> $dst_file" >&2
                return 1
            fi
        fi
    done < <(find "$src_root" -type f -print0)
}

sync_image_bfo_recovery_files() {
    local src_root="$1"
    local dst_root="$2"
    local src_file
    local dst_file
    local file_name
    if [ "$src_root" = "$dst_root" ] || [ ! -d "$src_root" ]; then
        return
    fi
    mkdir -p "$dst_root"
    for src_file in "$src_root"/historical_odds_bfo_recovered_*.csv; do
        [ -f "$src_file" ] || continue
        file_name="$(basename "$src_file")"
        dst_file="$dst_root/$file_name"
        if [ -f "$dst_file" ] && cmp -s "$src_file" "$dst_file"; then
            continue
        fi
        if atomic_copy_verified "$src_file" "$dst_file" replace; then
            echo "[seed] overlaid approved BFO recovery file from image: $dst_file"
        else
            echo "[seed] FAILED integrity check while overlaying BFO recovery file: $src_file -> $dst_file" >&2
            return 1
        fi
    done
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
copy_log_file bet_ledger_model_tracker.json
# Retired G-trader ledger remains migratable so historical/open positions are
# still visible to reconciliation and settlement code after Gemini removal.
copy_log_file bet_ledger_gemini_tracker.json
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
# Legacy directory name retained for Model Tracker history and historical
# operator-decision recovery. No active Gemini code writes operator decisions.
copy_tree_missing /app/data/operator "$PERSISTENT_DATA_DIR/operator"

# Approved image recovery files replace stale same-name volume copies. Files
# created only on the runtime volume are deliberately retained.
sync_image_bfo_recovery_files \
    /app/data/raw/historical_odds \
    "$PERSISTENT_DATA_DIR/raw/historical_odds"

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
        if atomic_copy_verified "$src" "$dst"; then
            echo "[seed] updated stale volume file from image (${dst_rows:-0} -> ${src_rows:-0} rows, $dst_size -> $src_size bytes): $dst"
        else
            echo "[seed] FAILED integrity check while updating image file: $src -> $dst" >&2
            return 1
        fi
    fi
}

# Existing volume roster state is authoritative.  Row count is not a freshness
# signal: stale retained/inactive rows can make the worse artifact larger.
# copy_tree_missing above still seeds the image roster on a brand-new volume.
if ! PYTHONPATH=/app python scripts/sanitize_active_roster_startup.py \
    --roster "$PERSISTENT_DATA_DIR/raw/ufc_active_roster_official.csv" \
    --fallback-roster /app/data/raw/ufc_active_roster_official.csv \
    --identity-audit "$PERSISTENT_DATA_DIR/raw/ufc_active_roster_identity_audit.csv"; then
    echo "[startup] ERROR: failed to sanitize persisted UFC active roster" >&2
    exit 1
fi
update_if_image_larger /app/data/raw/ufc_fighters_scraped.csv "$PERSISTENT_DATA_DIR/raw/ufc_fighters_scraped.csv"
update_if_image_larger /app/data/raw/ufc-fighter-details.csv "$PERSISTENT_DATA_DIR/raw/ufc-fighter-details.csv"

echo "[migrate] done"

# Ensure the runtime user can update mutable data and logs on the mounted
# volume.  Never chown an installed release: its model/training snapshot must
# remain read-only, while only its generation-specific lookup is mutable.
mkdir -p "$PERSISTENT_DATA_DIR" "$PERSISTENT_DATA_DIR/raw" "$PERSISTENT_DATA_DIR/processed" "$PERSISTENT_DATA_DIR/operator" "$PERSISTENT_DATA_DIR/tmp" "$PERSISTENT_DATA_DIR/production_bundle/current" "$PERSISTENT_LOG_DIR" "$ACTIVE_MODEL_DIR"
chown root:app "$PERSISTENT_DATA_DIR"
chmod 1775 "$PERSISTENT_DATA_DIR"
chown -R app:app "$PERSISTENT_DATA_DIR/raw" "$PERSISTENT_DATA_DIR/processed" "$PERSISTENT_DATA_DIR/operator" "$PERSISTENT_DATA_DIR/tmp"
if [ "$PERSISTENT_LOG_DIR" = "$PERSISTENT_DATA_DIR" ]; then
    # Railway mounts the log volume at the data root. Keep its root-owned,
    # sticky production_bundle entry protected while allowing app-owned log
    # files and mutable data siblings to remain writable.
    find "$PERSISTENT_LOG_DIR" -mindepth 1 -maxdepth 1 \
        ! -name production_bundle -exec chown -R app:app -- {} +
else
    chown -R app:app "$PERSISTENT_LOG_DIR"
fi
chown root:root "$PERSISTENT_DATA_DIR/production_bundle"
chmod 0755 "$PERSISTENT_DATA_DIR/production_bundle"
if [ "$BUNDLE_STORE_ACTIVE" -eq 1 ]; then
    chown -R root:root "$BUNDLE_STORE_ROOT"
    chmod -R a=rX "$ACTIVE_RELEASE_DIR"
    chown -R app:app "$ACTIVE_LOOKUP_DIR"
else
    chown -R app:app "$PERSISTENT_DATA_DIR/production_bundle/current" "$ACTIVE_MODEL_DIR"
fi

# Ensure the log files exist without truncating prior deployment history.
touch "$PERSISTENT_LOG_DIR/bot.log"
chown app:app "$PERSISTENT_LOG_DIR/bot.log"
touch "$PERSISTENT_LOG_DIR/alerts.jsonl"
chown app:app "$PERSISTENT_LOG_DIR/alerts.jsonl"

# Allow the runtime user to persist one-time metadata repairs only for the
# legacy mutable image aliases. Installed releases are immutable and already
# pass their full embedded-contract validation.
if [ "$BUNDLE_STORE_ACTIVE" -eq 0 ] && [ -f "$ACTIVE_MODEL_DIR/xgboost_no_odds_model.pkl" ]; then
    if ! su app -s /bin/sh -c "cd /app && PYTHONPATH=/app python scripts/repair_no_odds_training_spec.py \"$ACTIVE_MODEL_DIR/xgboost_no_odds_model.pkl\""; then
        echo "[migrate] WARNING: failed to normalize no-odds training_spec metadata in $ACTIVE_MODEL_DIR/xgboost_no_odds_model.pkl" >&2
    fi
fi

ACTIVATE_SOURCE_GENERATION_ARG=""
if [ "$BUNDLE_STORE_ACTIVE" -eq 1 ]; then
    ACTIVATE_SOURCE_GENERATION_ARG="--activate-source-generation"
fi
if ! su app -s /bin/sh -c "cd /app && PYTHONPATH=/app python scripts/bootstrap_runtime_production_bundle.py --target-manifest \"$PRODUCTION_BUNDLE_MANIFEST\" --source-manifest \"$SOURCE_PRODUCTION_BUNDLE_MANIFEST\" --source-processed-dir \"$SOURCE_PROCESSED_DIR\" --target-processed-dir \"$ACTIVE_LOOKUP_DIR\" --model-path \"$ACTIVE_MODEL_DIR/xgboost_model.pkl\" --no-odds-model-path \"$ACTIVE_MODEL_DIR/xgboost_no_odds_model.pkl\" --logistic-model-path \"$ACTIVE_MODEL_DIR/logistic_model.pkl\" $ACTIVATE_SOURCE_GENERATION_ARG"; then
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
