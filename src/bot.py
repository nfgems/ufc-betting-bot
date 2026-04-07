"""
Main bot orchestrator — ties together all components.

Usage:
    # Step 1: Train the model (run once, re-run when you have new data)
    python -m src.bot train

    # Step 2: Evaluate model performance
    python -m src.bot evaluate

    # Step 3: Run backtest to validate strategy
    python -m src.bot backtest

    # Step 4: Sensitivity analysis (find best parameters)
    python -m src.bot sensitivity

    # Step 5: Predict upcoming fights
    python -m src.bot predict

    # Step 6: Run live bot (dry run by default)
    python -m src.bot live --dry-run

    # Step 7: Run live bot with real money
    python -m src.bot live --real

    # Scrape latest data from UFCStats
    python -m src.bot scrape

    # Monitor upcoming events continuously (checks every N hours)
    python -m src.bot monitor

    # Track line movement (snapshot odds periodically)
    python -m src.bot track-lines

    # One-time check of all pre-fight signals for upcoming card
    python -m src.bot signals
"""

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from numbers import Integral, Real
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    LOGS_DIR,
    INITIAL_BANKROLL,
    MIN_EDGE_THRESHOLD,
    KELLY_FRACTION,
    BLEND_WEIGHT,
    MAX_BET_FRACTION,
    STOP_LOSS_FRACTION,
    PREDICTION_CACHE_SCHEMA_VERSION,
    PREDICTION_MAX_AGE_HOURS,
    PREDICTION_ODDS_CHANGE_THRESHOLD,
)
from src.live_control import assert_real_trading_allowed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "bot.log"),
    ],
)
logger = logging.getLogger(__name__)

_LIVE_CONTEXT_TABLE_CACHE: dict[str, tuple[float, object]] = {}
_LIVE_LOOKUP_FALLBACK_WINDOW_DAYS = 30
_LIVE_RECENT_MATCHUP_FALLBACK_BLOCK_DAYS = 30
_LIVE_TRADE_START_BUFFER = timedelta(minutes=10)
_LIVE_EVENT_CONTEXT_CACHE_TTL_SECONDS = 3600.0
_LIVE_EVENT_SKIP_LOG_TTL_SECONDS = 6 * 3600.0
_NON_UFC_LIVE_CONTEXT_REASON = (
    "not on any upcoming UFC card and no fight history found - likely a non-UFC MMA "
    "event or fighters not in database"
)
_LAST_GOOD_LIVE_EVENT_CONTEXTS: tuple[float, tuple[str, ...], list[dict]] | None = None
_LIVE_EVENT_SKIP_LOG_CACHE: dict[tuple[str, ...], float] = {}


def _is_truthy_flag(value: object) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"1", "true", "yes", "on", "armed"}


def _default_training_spec():
    """Return the promoted training contract for top-level train/retrain flows."""
    from src.model.production_bundle import ProductionBundleError, load_production_bundle
    from src.model.training_spec import (
        full_live_contract_v6_tuned_spec,
        resolve_named_training_spec,
    )

    fallback_spec = full_live_contract_v6_tuned_spec()

    try:
        bundle = load_production_bundle()
        return resolve_named_training_spec(bundle.model_spec_name)
    except (ProductionBundleError, ValueError) as exc:
        logger.warning(
            "Falling back to %s because the production manifest default training spec "
            "could not be resolved: %s",
            fallback_spec.name,
            exc,
        )
        return fallback_spec


def _resolve_named_training_spec_arg(spec_name: str):
    """Resolve a named training spec passed via CLI."""
    from src.model.training_spec import resolve_named_training_spec

    return resolve_named_training_spec(spec_name)


def _load_training_spec_from_artifact(model_name: str):
    """
    Resolve the reproducible training spec for an existing artifact.

    Retrains should preserve the promoted contract recorded inside the model
    when available. Legacy artifacts without an embedded spec are rejected so
    retrains cannot silently drift onto the current default contract.
    """
    from src.model.train import load_model
    from src.model.training_spec import NamedModelTrainingSpec

    try:
        model_result = load_model(model_name)
    except FileNotFoundError:
        return _default_training_spec()

    spec_payload = model_result.get("training_spec")
    if isinstance(spec_payload, dict):
        try:
            import json as _json
            return NamedModelTrainingSpec.from_json(_json.dumps(spec_payload))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid embedded training spec in {model_name}_model.pkl: {exc}"
            ) from exc

    raise ValueError(
        f"Model artifact {model_name!r} does not embed a reproducible training spec. "
        "Retrain it from a known named spec before using auto-retrain."
    )


def _explicit_model_path(model_name: str) -> Path | None:
    from src.live_control import _explicit_model_path as _lc_explicit
    return _lc_explicit(model_name)


def _training_spec_from_model_result(model_result: dict):
    from src.model.training_spec import NamedModelTrainingSpec

    spec_payload = model_result.get("training_spec")
    if isinstance(spec_payload, dict):
        try:
            # Use from_json to strip legacy fields (add_elo_momentum, etc.)
            import json as _json
            return NamedModelTrainingSpec.from_json(_json.dumps(spec_payload))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid embedded training spec in loaded model artifact: {exc}") from exc
    return _default_training_spec()


def _resolve_no_odds_model_arg(model_name: str) -> str | None:
    explicit_path = _explicit_model_path(model_name)
    if explicit_path is None:
        return "xgboost_no_odds"

    sibling = explicit_path.with_name("xgboost_no_odds_model.pkl")
    if sibling.exists():
        return str(sibling)
    return None


def _resolve_runtime_bundle_summary(
    *,
    model_result: dict,
    no_odds_result: dict | None = None,
) -> dict | None:
    try:
        from src.model.production_bundle import (
            is_hosted_runtime,
            load_production_bundle,
            validate_production_bundle,
        )
    except Exception:
        return None

    if not is_hosted_runtime():
        return None

    bundle = load_production_bundle()
    summary = validate_production_bundle(
        bundle,
        primary_model_result=model_result,
        no_odds_model_result=no_odds_result,
    )
    logger.info(
        "Active production bundle: bundle_id=%s manifest=%s model=%s no_odds=%s spec=%s processed_dir=%s processed_max_event_date=%s built_at=%s git_sha=%s",
        summary["bundle_id"],
        summary["manifest_path"],
        summary["model_path"],
        summary["no_odds_model_path"],
        summary["model_spec_name"],
        summary["processed_dir"],
        summary["processed_snapshot_max_event_date"],
        summary["built_at"],
        summary["git_sha"],
    )
    return summary


def _load_training_dataframe(*, data_path: Path | None, spec):
    """Resolve the raw training dataset for a top-level train/retrain command."""
    from src.data.kaggle_loader import load_kaggle_dataset

    if data_path is not None:
        logger.info("Loading explicit training dataset: %s", data_path)
        return load_kaggle_dataset(data_path)

    dataset_variant = getattr(spec, "dataset_variant", "default")
    if dataset_variant in {None, "", "default"}:
        logger.info("Loading default legacy training dataset.")
        return load_kaggle_dataset()

    from src.data.ufc_refresh import TRAINING_DATASET_VARIANTS, build_training_dataset_variants

    if dataset_variant not in TRAINING_DATASET_VARIANTS:
        known = ", ".join(TRAINING_DATASET_VARIANTS)
        raise ValueError(
            f"Unknown training dataset variant '{dataset_variant}'. "
            f"Known variants: {known}"
        )

    logger.info("Loading training dataset variant: %s", dataset_variant)
    legacy_df = load_kaggle_dataset(RAW_DATA_DIR / "ufc-master.csv")
    all_variants = build_training_dataset_variants(legacy_df=legacy_df)
    return all_variants[dataset_variant].copy()


def _coerce_live_fight_count(value) -> int | None:
    """Coerce a scraped fight-count feature into a non-negative integer."""
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        try:
            parsed = int(float(value))
        except (TypeError, ValueError):
            return None
    return parsed if parsed >= 0 else None


def _serialize_live_context_number(
    value,
    *,
    as_int: bool = False,
    precision: int = 4,
):
    """Convert live numeric payload values into JSON-safe scalars."""
    if value is None:
        return None
    if not isinstance(value, Real):
        return None

    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return None

    return int(numeric) if as_int else round(numeric, precision)


def _resolve_live_fight_counts(
    features: dict,
    fighter_a: str,
    fighter_b: str,
    *,
    fallback_resolver=None,
) -> tuple[int, int]:
    """
    Prefer live-scraped UFC fight counts embedded in the live feature vector.

    The processed dataset can lag behind aliases or recent UFC appearances, so
    using it directly can misclassify active fighters as low-experience.
    """
    if fallback_resolver is None:
        from src.features.build_features import get_fighter_ufc_fight_count

        fallback_resolver = get_fighter_ufc_fight_count

    a_fights = _coerce_live_fight_count(features.get("a_num_fights"))
    b_fights = _coerce_live_fight_count(features.get("b_num_fights"))

    if a_fights is None:
        a_fights = fallback_resolver(fighter_a)
    if b_fights is None:
        b_fights = fallback_resolver(fighter_b)

    return (
        int(a_fights) if a_fights is not None else 0,
        int(b_fights) if b_fights is not None else 0,
    )


def _live_fighter_name_signature(normalized_name: str) -> str:
    tokens = [token for token in str(normalized_name or "").split() if token]
    if len(tokens) < 2:
        return ""
    return f"{tokens[0]} {tokens[-1]}"


def _collapse_live_initials(normalized_name: str) -> str:
    tokens = [token for token in str(normalized_name or "").split() if token]
    if len(tokens) < 3:
        return " ".join(tokens)

    initial_tokens: list[str] = []
    suffix_start = 0
    for idx, token in enumerate(tokens):
        if len(token) == 1 and idx < len(tokens) - 1:
            initial_tokens.append(token)
            continue
        suffix_start = idx
        break
    else:
        return "".join(initial_tokens)

    if len(initial_tokens) < 2:
        return " ".join(tokens)
    return " ".join(["".join(initial_tokens), *tokens[suffix_start:]])


def _load_live_fighter_alias_map() -> dict[str, str]:
    """Load unique roster-backed aliases used to match live fight names."""
    import pandas as pd
    from src.data.name_utils import normalize_cross_source_name
    from src.data.ufc_active_roster import OFFICIAL_ACTIVE_ROSTER_PATH

    official_path = OFFICIAL_ACTIVE_ROSTER_PATH
    alias_columns = {
        "official_name",
        "profile_name",
        "slug_name",
        "alternate_slug_names",
        "ufcstats_name",
    }
    try:
        cache_key = f"live-name-aliases::{official_path.resolve()}"
        mtime = official_path.stat().st_mtime if official_path.exists() else 0.0
    except OSError:
        return {}

    cached = _LIVE_CONTEXT_TABLE_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        aliases = cached[1]
        return aliases if isinstance(aliases, dict) else {}

    if not official_path.exists():
        _LIVE_CONTEXT_TABLE_CACHE[cache_key] = (mtime, {})
        return {}

    exact_candidates: dict[str, set[str]] = {}
    signature_candidates: dict[str, set[str]] = {}
    try:
        official_df = pd.read_csv(
            official_path,
            usecols=lambda column: column in alias_columns,
        )
    except Exception:
        _LIVE_CONTEXT_TABLE_CACHE[cache_key] = (mtime, {})
        return {}

    for _, row in official_df.iterrows():
        canonical = ""
        for column in ("official_name", "profile_name", "ufcstats_name", "slug_name"):
            candidate = _collapse_live_initials(normalize_cross_source_name(row.get(column)))
            if candidate:
                canonical = candidate
                break
        if not canonical:
            continue

        aliases = [
            row.get("official_name"),
            row.get("profile_name"),
            row.get("slug_name"),
            row.get("ufcstats_name"),
        ]
        aliases.extend(str(row.get("alternate_slug_names") or "").split("|"))
        for alias in aliases:
            alias_key = _collapse_live_initials(normalize_cross_source_name(alias))
            if not alias_key:
                continue
            exact_candidates.setdefault(alias_key, set()).add(canonical)
            signature = _live_fighter_name_signature(alias_key)
            if signature:
                signature_candidates.setdefault(signature, set()).add(canonical)

    alias_map: dict[str, str] = {}
    for candidate_map in (exact_candidates, signature_candidates):
        for alias_key, canonical_options in candidate_map.items():
            if len(canonical_options) == 1:
                alias_map.setdefault(alias_key, next(iter(canonical_options)))

    _LIVE_CONTEXT_TABLE_CACHE[cache_key] = (mtime, alias_map)
    return alias_map


def _canonicalize_live_fighter_name(fighter_name: str) -> str:
    from src.data.name_utils import normalize_cross_source_name

    normalized = _collapse_live_initials(normalize_cross_source_name(fighter_name))
    if not normalized:
        return ""
    return _load_live_fighter_alias_map().get(normalized, normalized)


def _live_fight_pair_key(fighter_a: str, fighter_b: str) -> str:
    return "|".join(
        sorted(
            [
                _canonicalize_live_fighter_name(fighter_a),
                _canonicalize_live_fighter_name(fighter_b),
            ]
        )
    )


def _parse_live_context_timestamp(value) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _sanitize_prediction_cache_value(value):
    """Convert nested prediction-cache values into JSON-safe scalars."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    if isinstance(value, dict):
        return {
            str(key): _sanitize_prediction_cache_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_prediction_cache_value(item) for item in value]
    return value


def _prediction_feature_contract_hash(feature_cols: object) -> str:
    payload = json.dumps(list(feature_cols or []), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prediction_cache_artifact_signature(path_value: object) -> dict | None:
    artifact_path = str(path_value or "").strip()
    if not artifact_path:
        return None

    resolved = Path(artifact_path).resolve(strict=False)
    stat_payload: dict[str, object] = {
        "path": str(resolved),
        "size": None,
        "mtime_ns": None,
    }
    try:
        stat_result = resolved.stat()
    except OSError:
        return stat_payload

    stat_payload["size"] = int(stat_result.st_size)
    stat_payload["mtime_ns"] = int(
        getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000))
    )
    return stat_payload


def _prediction_runtime_signature(
    *,
    model_result: dict,
    no_odds_result: dict | None = None,
    runtime_bundle_summary: dict | None = None,
) -> dict:
    primary_spec = model_result.get("training_spec") if isinstance(model_result.get("training_spec"), dict) else {}
    no_odds_spec = (
        no_odds_result.get("training_spec")
        if isinstance(no_odds_result, dict) and isinstance(no_odds_result.get("training_spec"), dict)
        else {}
    )
    signature = {
        "primary_artifact": _prediction_cache_artifact_signature(model_result.get("artifact_path")),
        "primary_spec_name": str(primary_spec.get("name", "") or ""),
        "primary_feature_hash": _prediction_feature_contract_hash(
            primary_spec.get("feature_cols") or model_result.get("feature_cols") or []
        ),
        "primary_feature_count": len(primary_spec.get("feature_cols") or model_result.get("feature_cols") or []),
        "no_odds_artifact": (
            _prediction_cache_artifact_signature(no_odds_result.get("artifact_path"))
            if isinstance(no_odds_result, dict)
            else None
        ),
        "no_odds_spec_name": str(no_odds_spec.get("name", "") or ""),
        "no_odds_feature_hash": (
            _prediction_feature_contract_hash(
                no_odds_spec.get("feature_cols") or no_odds_result.get("feature_cols") or []
            )
            if isinstance(no_odds_result, dict)
            else ""
        ),
        "no_odds_feature_count": (
            len(no_odds_spec.get("feature_cols") or no_odds_result.get("feature_cols") or [])
            if isinstance(no_odds_result, dict)
            else 0
        ),
        "bundle_id": str((runtime_bundle_summary or {}).get("bundle_id", "") or ""),
        "bundle_built_at": str((runtime_bundle_summary or {}).get("built_at", "") or ""),
        "bundle_git_sha": str((runtime_bundle_summary or {}).get("git_sha", "") or ""),
        "bundle_processed_dir": str((runtime_bundle_summary or {}).get("processed_dir", "") or ""),
        "bundle_processed_snapshot_max_event_date": str(
            (runtime_bundle_summary or {}).get("processed_snapshot_max_event_date", "") or ""
        ),
    }
    return _sanitize_prediction_cache_value(signature)


def _prediction_commence_token(value: object) -> str:
    parsed = _parse_live_context_timestamp(value)
    if parsed is not None:
        return parsed.isoformat()
    return str(value or "").strip()


def _prediction_cache_key_for_values(
    fighter_a: object,
    fighter_b: object,
    *,
    event_id: object = None,
    commence_time: object = None,
) -> str:
    pair_key = _live_fight_pair_key(str(fighter_a or ""), str(fighter_b or ""))
    event_token = str(event_id or "").strip()
    if event_token:
        return f"event:{event_token}::{pair_key}"
    return f"time:{_prediction_commence_token(commence_time)}::{pair_key}"


def _prediction_cache_key(fight: dict | object) -> str:
    getter = getattr(fight, "get", None)
    if callable(getter):
        fighter_a = getter("fighter_a", "")
        fighter_b = getter("fighter_b", "")
        event_id = getter("event_id", "")
        commence_time = getter("commence_time", "")
    else:
        fighter_a = getattr(fight, "fighter_a", "")
        fighter_b = getattr(fight, "fighter_b", "")
        event_id = getattr(fight, "event_id", "")
        commence_time = getattr(fight, "commence_time", "")
    return _prediction_cache_key_for_values(
        fighter_a,
        fighter_b,
        event_id=event_id,
        commence_time=commence_time,
    )


def _prediction_event_context_snapshot(fight: dict | object, event_context: dict) -> dict:
    getter = getattr(fight, "get", None)
    event_id = getter("event_id", "") if callable(getter) else getattr(fight, "event_id", "")
    commence_time = getter("commence_time", "") if callable(getter) else getattr(fight, "commence_time", "")
    return _sanitize_prediction_cache_value(
        {
            "event_id": str(event_id or ""),
            "commence_time": _prediction_commence_token(commence_time),
            "weight_class": str(event_context.get("weight_class", "") or ""),
            "num_rounds": event_context.get("num_rounds"),
            "is_title_bout": bool(event_context.get("is_title_bout")),
            "is_empty_arena": event_context.get("is_empty_arena"),
        }
    )


def _prediction_odds_snapshot(fight: dict | object) -> dict:
    getter = getattr(fight, "get", None)
    a_prob = getter("a_fair_prob_avg", None) if callable(getter) else getattr(fight, "a_fair_prob_avg", None)
    b_prob = getter("b_fair_prob_avg", None) if callable(getter) else getattr(fight, "b_fair_prob_avg", None)
    return _sanitize_prediction_cache_value(
        {
            "a_fair_prob_avg": a_prob,
            "b_fair_prob_avg": b_prob,
        }
    )


def _load_existing_prediction_cache() -> dict[str, dict]:
    cache_path = LOGS_DIR / "predictions_cache.json"
    if not cache_path.exists():
        return {}

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load existing prediction cache from %s: %s", cache_path, exc)
        return {}

    if int(payload.get("schema_version") or 0) != PREDICTION_CACHE_SCHEMA_VERSION:
        return {}

    rows = payload.get("predictions")
    if not isinstance(rows, list):
        return {}

    existing: dict[str, dict] = {}
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            continue
        cache_key = str(raw_row.get("cache_key", "") or "").strip()
        if not cache_key:
            continue
        if not raw_row.get("prediction_generated_at"):
            continue
        if not isinstance(raw_row.get("odds_snapshot"), dict):
            continue
        if not isinstance(raw_row.get("event_context_snapshot"), dict):
            continue
        if not isinstance(raw_row.get("runtime_signature"), dict):
            continue
        if not isinstance(raw_row.get("operator_features"), dict):
            continue
        if not isinstance(raw_row.get("operator_provenance"), dict):
            continue
        existing[cache_key] = dict(raw_row)
    return existing


def _prediction_needs_refresh(
    cached: dict,
    current_fight,
    *,
    runtime_signature: dict,
) -> tuple[bool, str]:
    if not isinstance(cached, dict):
        return True, "missing cache row"

    current_cache_key = _prediction_cache_key(current_fight)
    if str(cached.get("cache_key", "") or "") != current_cache_key:
        return True, "fight identity changed"

    fighter_a = str(current_fight.get("fighter_a", "") or "")
    fighter_b = str(current_fight.get("fighter_b", "") or "")
    if str(cached.get("fighter_a", "") or "") != fighter_a or str(cached.get("fighter_b", "") or "") != fighter_b:
        return True, "fighter order changed"

    current_pair_key = _live_fight_pair_key(fighter_a, fighter_b)
    if str(cached.get("pair_key", "") or "") != current_pair_key:
        return True, "pair key changed"

    if cached.get("runtime_signature") != runtime_signature:
        return True, "runtime signature changed"

    if not isinstance(cached.get("event_context_snapshot"), dict):
        return True, "missing event context snapshot"
    if not isinstance(cached.get("operator_features"), dict):
        return True, "missing operator features"
    if not isinstance(cached.get("operator_provenance"), dict):
        return True, "missing operator provenance"

    current_event_id = str(current_fight.get("event_id", "") or "")
    cached_event_id = str(cached.get("event_id", "") or "")
    if cached_event_id != current_event_id:
        return True, "event id changed"

    current_commence = _prediction_commence_token(current_fight.get("commence_time"))
    cached_commence = _prediction_commence_token(cached.get("event_date") or cached.get("commence_time"))
    if cached_commence != current_commence:
        return True, "commence time changed"

    can_trade, start_reason, _ = _live_fight_is_tradeable(current_fight.get("commence_time"))
    if not can_trade:
        return True, start_reason

    generated_at = _parse_live_context_timestamp(cached.get("prediction_generated_at"))
    if generated_at is None:
        return True, "missing generation timestamp"
    age_seconds = (_current_utc() - generated_at).total_seconds()
    if age_seconds >= (PREDICTION_MAX_AGE_HOURS * 3600):
        return True, f"cache older than {PREDICTION_MAX_AGE_HOURS}h"

    odds_snapshot = cached.get("odds_snapshot")
    if not isinstance(odds_snapshot, dict):
        return True, "missing odds snapshot"

    try:
        old_a = float(odds_snapshot.get("a_fair_prob_avg"))
        old_b = float(odds_snapshot.get("b_fair_prob_avg"))
        new_a = float(current_fight.get("a_fair_prob_avg"))
        new_b = float(current_fight.get("b_fair_prob_avg"))
    except (TypeError, ValueError):
        return True, "invalid odds snapshot"

    max_shift = max(abs(new_a - old_a), abs(new_b - old_b))
    if max_shift > PREDICTION_ODDS_CHANGE_THRESHOLD:
        return True, f"odds moved {max_shift:.1%}"

    return False, "cache hit"


def _prediction_cache_sort_key(row: dict) -> tuple[str, str, str]:
    commence = _prediction_commence_token(row.get("event_date") or row.get("commence_time"))
    return (
        commence,
        str(row.get("fighter_a", "") or ""),
        str(row.get("fighter_b", "") or ""),
    )


def _prediction_cache_rows_for_write(
    rows_by_key: dict[str, dict],
    *,
    allowed_keys: set[str] | None = None,
) -> list[dict]:
    rows = [
        dict(row)
        for key, row in rows_by_key.items()
        if allowed_keys is None or key in allowed_keys
    ]
    return sorted(rows, key=_prediction_cache_sort_key)


def _current_utc() -> datetime:
    return datetime.now(timezone.utc)


def _live_fight_is_tradeable(commence_time) -> tuple[bool, str, datetime | None]:
    commence = _parse_live_context_timestamp(commence_time)
    if commence is None:
        return False, "missing or invalid commence_time", None
    cutoff = commence - _LIVE_TRADE_START_BUFFER
    now = _current_utc()
    if now >= cutoff:
        return (
            False,
            f"fight starts at {commence.isoformat()} (safety buffer {_LIVE_TRADE_START_BUFFER})",
            commence,
        )
    return True, "", commence


def _live_card_identity(rows: object) -> tuple[str, ...]:
    if rows is None:
        return ()

    row_dicts: list[dict] = []
    if hasattr(rows, "to_dict"):
        try:
            row_dicts = list(rows.to_dict("records"))
        except Exception:
            row_dicts = []
    elif isinstance(rows, list):
        row_dicts = [row for row in rows if isinstance(row, dict)]
    elif isinstance(rows, tuple):
        row_dicts = [row for row in rows if isinstance(row, dict)]

    event_ids: set[str] = set()
    event_dates: set[str] = set()
    for row in row_dicts:
        event_id = str(row.get("event_id", "") or "").strip()
        if event_id:
            event_ids.add(event_id)
            continue

        commence = _parse_live_context_timestamp(row.get("commence_time"))
        if commence is not None:
            event_dates.add(commence.date().isoformat())
            continue

        raw_event_date = str(row.get("event_date", "") or "").strip()
        if not raw_event_date:
            continue
        for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
            try:
                event_dates.add(datetime.strptime(raw_event_date, fmt).date().isoformat())
                break
            except ValueError:
                continue

    if event_ids:
        return tuple(sorted(f"event:{event_id}" for event_id in event_ids))
    return tuple(sorted(f"date:{event_date}" for event_date in event_dates))


def _cached_live_event_contexts_match(expected_fights: object) -> list[dict]:
    if _LAST_GOOD_LIVE_EVENT_CONTEXTS is None:
        return []

    fetched_at, cached_identity, cached_contexts = _LAST_GOOD_LIVE_EVENT_CONTEXTS
    age_seconds = time.monotonic() - fetched_at
    if age_seconds > _LIVE_EVENT_CONTEXT_CACHE_TTL_SECONDS:
        logger.warning(
            "Discarding cached live UFC event context from %.0fs ago after fetch failure: exceeded %.0fs TTL",
            age_seconds,
            _LIVE_EVENT_CONTEXT_CACHE_TTL_SECONDS,
        )
        return []

    expected_identity = _live_card_identity(expected_fights)
    if expected_identity and cached_identity and expected_identity != cached_identity:
        logger.warning(
            "Discarding cached live UFC event context from %.0fs ago after fetch failure: "
            "cached card %s does not match current card %s",
            age_seconds,
            ",".join(cached_identity),
            ",".join(expected_identity),
        )
        return []

    logger.warning(
        "Using cached live UFC event context from %.0fs ago after fetch failure",
        age_seconds,
    )
    return list(cached_contexts)


def _log_live_fight_skip_once(fight: dict | object, reason: str) -> None:
    fighter_a = str(getattr(fight, "get", lambda _key, _default=None: _default)("fighter_a", "") or "").strip()
    fighter_b = str(getattr(fight, "get", lambda _key, _default=None: _default)("fighter_b", "") or "").strip()
    event_id = str(getattr(fight, "get", lambda _key, _default=None: _default)("event_id", "") or "").strip()
    commence_time = str(getattr(fight, "get", lambda _key, _default=None: _default)("commence_time", "") or "").strip()
    key = (
        event_id or "unknown",
        commence_time or "unknown",
        fighter_a.casefold(),
        fighter_b.casefold(),
        str(reason or "").strip(),
    )

    now = time.monotonic()
    last_logged_at = _LIVE_EVENT_SKIP_LOG_CACHE.get(key)
    if last_logged_at is not None and (now - last_logged_at) < _LIVE_EVENT_SKIP_LOG_TTL_SECONDS:
        return
    _LIVE_EVENT_SKIP_LOG_CACHE[key] = now

    normalized_reason = str(reason or "").strip().casefold()
    is_expected_skip = (
        normalized_reason == _NON_UFC_LIVE_CONTEXT_REASON.casefold()
        or "not on any upcoming ufc card" in normalized_reason
    )
    log_fn = logger.info if is_expected_skip else logger.warning
    log_fn(
        "Skipping %s vs %s: %s (event_id=%s commence_time=%s)",
        fighter_a,
        fighter_b,
        reason,
        event_id,
        commence_time,
    )


def _load_live_event_contexts(expected_fights: object = None) -> list[dict]:
    """Fetch upcoming UFC event metadata used to populate live model context fields."""
    global _LAST_GOOD_LIVE_EVENT_CONTEXTS
    try:
        from src.data.live_monitor import collect_upcoming_fight_contexts
        last_exc = None
        for attempt in range(1, 4):
            try:
                contexts = list(collect_upcoming_fight_contexts())
                if contexts:
                    _LAST_GOOD_LIVE_EVENT_CONTEXTS = (
                        time.monotonic(),
                        _live_card_identity(contexts) or _live_card_identity(expected_fights),
                        contexts,
                    )
                    return contexts
                last_exc = RuntimeError("collector returned no upcoming fight contexts")
                logger.warning(
                    "Live UFC event context fetch returned no rows (attempt %s/3)",
                    attempt,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Could not load live UFC event context (attempt %s/3): %s",
                    attempt,
                    exc,
                )
                if attempt < 3:
                    time.sleep(min(2 ** (attempt - 1), 4))
        cached_contexts = _cached_live_event_contexts_match(expected_fights)
        if cached_contexts:
            return cached_contexts
        if last_exc is not None:
            logger.warning("Could not load live UFC event context: %s", last_exc)
        return []
    except Exception as exc:
        logger.warning("Could not load live UFC event context: %s", exc)
        return _cached_live_event_contexts_match(expected_fights)


def _load_live_event_contexts_for_fights(expected_fights: object = None) -> list[dict]:
    try:
        return _load_live_event_contexts(expected_fights)
    except TypeError as exc:
        if expected_fights is None:
            raise
        try:
            return _load_live_event_contexts()
        except TypeError:
            raise exc


def _normalize_live_weight_class(value) -> str | None:
    text = str(value or "").strip()
    return text or None


def _load_live_history_frame(
    path: Path,
    *,
    usecols: list[str],
    rename_map: dict[str, str] | None = None,
):
    """Load and cache a local UFC history artifact for live context fallback."""
    import pandas as pd
    from src.data.name_utils import normalize_cross_source_name

    if not path.exists():
        return None

    try:
        cache_key = str(path.resolve())
        mtime = path.stat().st_mtime
    except OSError:
        return None

    cached = _LIVE_CONTEXT_TABLE_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    try:
        df = pd.read_csv(path, usecols=usecols)
    except Exception:
        _LIVE_CONTEXT_TABLE_CACHE[cache_key] = (mtime, None)
        return None

    if rename_map:
        df = df.rename(columns=rename_map)

    required_cols = {"fighter_a", "fighter_b", "weight_class"}
    if df.empty or not required_cols.issubset(df.columns):
        _LIVE_CONTEXT_TABLE_CACHE[cache_key] = (mtime, None)
        return None

    prepared = df.copy()
    prepared["fighter_a_norm"] = prepared["fighter_a"].map(normalize_cross_source_name)
    prepared["fighter_b_norm"] = prepared["fighter_b"].map(normalize_cross_source_name)
    if "event_date" in prepared.columns:
        prepared["event_date_sort"] = pd.to_datetime(prepared["event_date"], errors="coerce")

    _LIVE_CONTEXT_TABLE_CACHE[cache_key] = (mtime, prepared)
    return prepared


def _latest_weight_class_from_local_history(norm_name: str) -> str | None:
    """Return the latest known UFC weight class for a fighter from local artifacts."""
    # (path, usecols, rename_map) — rename_map normalises raw-file column names
    # to the canonical {fighter_a, fighter_b, weight_class, event_date} schema
    # that _load_live_history_frame expects.
    history_sources: list[tuple[Path, list[str], dict[str, str]]] = [
        (PROCESSED_DATA_DIR / "fights_cleaned.csv",
         ["fighter_a", "fighter_b", "weight_class", "event_date"], {}),
        (RAW_DATA_DIR / "ufc-master.csv",
         ["RedFighter", "BlueFighter", "WeightClass", "Date"],
         {"RedFighter": "fighter_a", "BlueFighter": "fighter_b",
          "WeightClass": "weight_class", "Date": "event_date"}),
        (RAW_DATA_DIR / "jansen88_ufc_data.csv",
         ["fighter1", "fighter2", "weight_class", "event_date"],
         {"fighter1": "fighter_a", "fighter2": "fighter_b"}),
    ]

    for path, usecols, rename_map in history_sources:
        df = _load_live_history_frame(path, usecols=usecols, rename_map=rename_map)
        if df is None:
            continue

        subset = df.loc[
            (df["fighter_a_norm"] == norm_name) | (df["fighter_b_norm"] == norm_name)
        ].dropna(subset=["weight_class"])
        subset = subset[subset["weight_class"].astype(str).str.strip() != ""]
        if subset.empty:
            continue

        if "event_date_sort" in subset.columns:
            subset = subset.sort_values("event_date_sort", ascending=False, na_position="last")
        elif "event_date" in subset.columns:
            subset = subset.sort_values("event_date", ascending=False)

        weight_class = _normalize_live_weight_class(subset.iloc[0]["weight_class"])
        if weight_class is not None:
            return weight_class

    return None


def _latest_local_head_to_head_event_date(
    fighter_a: str,
    fighter_b: str,
) -> datetime | None:
    """Return the latest recorded local date for this exact fighter pairing."""
    import pandas as pd
    from src.data.name_utils import normalize_cross_source_name

    norm_a = normalize_cross_source_name(fighter_a)
    norm_b = normalize_cross_source_name(fighter_b)
    if not norm_a or not norm_b:
        return None

    history_sources: list[tuple[Path, list[str], dict[str, str]]] = [
        (PROCESSED_DATA_DIR / "fights_cleaned.csv",
         ["fighter_a", "fighter_b", "weight_class", "event_date"], {}),
        (RAW_DATA_DIR / "ufc-master.csv",
         ["RedFighter", "BlueFighter", "WeightClass", "Date"],
         {"RedFighter": "fighter_a", "BlueFighter": "fighter_b",
          "WeightClass": "weight_class", "Date": "event_date"}),
        (RAW_DATA_DIR / "jansen88_ufc_data.csv",
         ["fighter1", "fighter2", "weight_class", "event_date"],
         {"fighter1": "fighter_a", "fighter2": "fighter_b"}),
    ]

    latest_match: pd.Timestamp | None = None
    for path, usecols, rename_map in history_sources:
        df = _load_live_history_frame(path, usecols=usecols, rename_map=rename_map)
        if df is None:
            continue

        subset = df.loc[
            (
                (df["fighter_a_norm"] == norm_a) & (df["fighter_b_norm"] == norm_b)
            ) | (
                (df["fighter_a_norm"] == norm_b) & (df["fighter_b_norm"] == norm_a)
            )
        ]
        if subset.empty:
            continue

        if "event_date_sort" in subset.columns:
            candidate = subset["event_date_sort"].dropna().max()
        else:
            candidate = pd.to_datetime(subset["event_date"], errors="coerce").dropna().max()
        if pd.isna(candidate):
            continue

        candidate_ts = pd.Timestamp(candidate)
        if latest_match is None or candidate_ts > latest_match:
            latest_match = candidate_ts

    if latest_match is None:
        return None
    if latest_match.tzinfo is None:
        latest_match = latest_match.tz_localize(timezone.utc)
    else:
        latest_match = latest_match.tz_convert(timezone.utc)
    return latest_match.to_pydatetime()


def _load_local_ufc_roster_names() -> set[str]:
    """Return normalized fighter names from the local UFC roster artifact."""
    import pandas as pd
    from src.data.name_utils import normalize_cross_source_name
    from src.data.ufc_active_roster import OFFICIAL_ACTIVE_ROSTER_PATH

    scraped_path = RAW_DATA_DIR / "ufc_fighters_scraped.csv"
    official_path = OFFICIAL_ACTIVE_ROSTER_PATH
    if not scraped_path.exists() and not official_path.exists():
        return set()

    try:
        cache_key = f"roster::{scraped_path.resolve()}::{official_path.resolve()}"
        mtime = (
            scraped_path.stat().st_mtime if scraped_path.exists() else 0.0,
            official_path.stat().st_mtime if official_path.exists() else 0.0,
        )
    except OSError:
        return set()

    cached = _LIVE_CONTEXT_TABLE_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        names = cached[1]
        return names if isinstance(names, set) else set()

    names: set[str] = set()
    if scraped_path.exists():
        try:
            df = pd.read_csv(scraped_path, usecols=["name"])
            names.update(
                normalize_cross_source_name(name)
                for name in df["name"].dropna().astype(str)
                if str(name).strip()
            )
        except Exception:
            pass

    if official_path.exists():
        official_cols = ["official_name", "slug_name", "alternate_slug_names", "ufcstats_name"]
        try:
            official_df = pd.read_csv(official_path, usecols=official_cols)
            for column in ("official_name", "slug_name", "ufcstats_name"):
                if column in official_df.columns:
                    names.update(
                        normalize_cross_source_name(value)
                        for value in official_df[column].dropna().astype(str)
                        if str(value).strip()
                    )
            if "alternate_slug_names" in official_df.columns:
                for value in official_df["alternate_slug_names"].dropna().astype(str):
                    for alias in value.split("|"):
                        if alias.strip():
                            names.add(normalize_cross_source_name(alias))
        except Exception:
            pass

    names.discard("")
    _LIVE_CONTEXT_TABLE_CACHE[cache_key] = (mtime, names)
    return names


def _missing_live_event_context_reason(fighter_a: str, fighter_b: str) -> str:
    """Explain why a live MMA bout was skipped after all UFC context fallbacks failed."""
    from src.data.name_utils import normalize_cross_source_name

    latest_head_to_head = _latest_local_head_to_head_event_date(fighter_a, fighter_b)
    if latest_head_to_head is not None:
        return (
            "pair already exists in local UFC history "
            f"({latest_head_to_head.date().isoformat()}) but is not on any upcoming UFC card"
        )

    roster_names = _load_local_ufc_roster_names()
    in_roster_a = normalize_cross_source_name(fighter_a) in roster_names
    in_roster_b = normalize_cross_source_name(fighter_b) in roster_names
    if in_roster_a and in_roster_b:
        return (
            "not on any upcoming UFC card and no local division history found "
            "(both fighters appear in the local UFC roster cache)"
        )
    if in_roster_a or in_roster_b:
        return (
            "not on any upcoming UFC card and no local division history found "
            "(one fighter appears in the local UFC roster cache)"
        )
    return _NON_UFC_LIVE_CONTEXT_REASON


def _infer_weight_class_from_history(fighter_a: str, fighter_b: str) -> str | None:
    """Try to infer weight class from local UFC history artifacts."""
    from src.data.name_utils import normalize_cross_source_name

    norm_a = normalize_cross_source_name(fighter_a)
    norm_b = normalize_cross_source_name(fighter_b)

    wc_a = _latest_weight_class_from_local_history(norm_a)
    wc_b = _latest_weight_class_from_local_history(norm_b)

    if wc_a and wc_b:
        if wc_a.lower() == wc_b.lower():
            return wc_a
        return wc_a
    return wc_a or wc_b


def _upcoming_live_event_dates(live_event_contexts: list[dict]) -> set:
    dates = set()
    for context in live_event_contexts:
        commence = _parse_live_context_timestamp(context.get("commence_time"))
        if commence is not None:
            dates.add(commence.date())
            continue

        raw_event_date = str(context.get("event_date", "") or "").strip()
        if not raw_event_date:
            continue
        for fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                dates.add(datetime.strptime(raw_event_date, fmt).date())
                break
            except ValueError:
                continue
    return dates


def _latest_weight_class_from_fight_records(fights: list[dict]) -> str | None:
    for fight in reversed(list(fights or [])):
        weight_class = _normalize_live_weight_class(fight.get("weight_class"))
        if weight_class is not None:
            return weight_class
    for fight in fights or []:
        weight_class = _normalize_live_weight_class(fight.get("weight_class"))
        if weight_class is not None:
            return weight_class
    return None


def _latest_weight_class_from_processed_or_ufcstats(
    fighter_name: str,
    *,
    reference_date: datetime | None = None,
) -> str | None:
    from src.data.fighter_lookup import (
        _lookup_processed_fighter,
        scrape_fighter_fights,
        search_fighter_url,
    )

    processed_result = _lookup_processed_fighter(
        fighter_name,
        reference_date=reference_date.isoformat() if reference_date is not None else None,
    )
    if processed_result is not None:
        processed_wc = _latest_weight_class_from_fight_records(processed_result.get("fights", []))
        if processed_wc is not None:
            return processed_wc

    fighter_url = search_fighter_url(fighter_name)
    if not fighter_url:
        return None

    try:
        fights = scrape_fighter_fights(fighter_url, fighter_name=fighter_name)
    except Exception as exc:
        logger.debug(
            "Could not scrape UFCStats fight history for %s while inferring live context: %s",
            fighter_name,
            exc,
        )
        return None

    return _latest_weight_class_from_fight_records(fights)


def _infer_weight_class_from_near_term_ufc_lookup(
    fighter_a: str,
    fighter_b: str,
    *,
    requested_commence: datetime | None,
    live_event_contexts: list[dict],
) -> str | None:
    from src.data.name_utils import normalize_cross_source_name

    if requested_commence is None:
        return None

    now = _current_utc()
    if requested_commence < now - timedelta(days=1):
        return None
    if requested_commence > now + timedelta(days=_LIVE_LOOKUP_FALLBACK_WINDOW_DAYS):
        return None

    if requested_commence.date() not in _upcoming_live_event_dates(live_event_contexts):
        return None

    roster_names = _load_local_ufc_roster_names()
    if not roster_names:
        return None
    if (
        normalize_cross_source_name(fighter_a) not in roster_names
        and normalize_cross_source_name(fighter_b) not in roster_names
    ):
        return None

    wc_a = _latest_weight_class_from_processed_or_ufcstats(
        fighter_a,
        reference_date=requested_commence,
    )
    wc_b = _latest_weight_class_from_processed_or_ufcstats(
        fighter_b,
        reference_date=requested_commence,
    )

    if not wc_a or not wc_b:
        return None
    return wc_a if wc_a.lower() == wc_b.lower() else None


def _resolve_live_event_context(
    fight,
    live_event_contexts: list[dict],
    *,
    allow_off_card_history_fallback: bool = True,
) -> dict | None:
    """Match an odds row to scraped UFC event context for live feature building.

    Falls back to historical fighter data if the fight is not found on UFCStats
    (e.g. non-UFC MMA events or far-future UFC cards not yet listed).
    """
    fighter_a = fight.get("fighter_a", "")
    fighter_b = fight.get("fighter_b", "")
    pair_key = _live_fight_pair_key(fighter_a, fighter_b)
    event_id = str(fight.get("event_id", "") or "")
    requested_commence = _parse_live_context_timestamp(fight.get("commence_time"))

    candidates = [
        context
        for context in live_event_contexts
        if _live_fight_pair_key(context.get("fighter_a", ""), context.get("fighter_b", "")) == pair_key
    ]
    if event_id:
        exact_event = [
            context
            for context in candidates
            if str(context.get("event_id", "") or "") == event_id
        ]
        if exact_event:
            candidates = exact_event

    candidates = [
        context
        for context in candidates
        if _normalize_live_weight_class(context.get("weight_class")) is not None
    ]

    if requested_commence is not None and len(candidates) > 1:
        def _distance_seconds(context: dict) -> float:
            context_commence = _parse_live_context_timestamp(context.get("commence_time"))
            if context_commence is None:
                return float("inf")
            return abs((context_commence - requested_commence).total_seconds())

        candidates = sorted(
            candidates,
            key=lambda context: (
                _distance_seconds(context),
                str(context.get("event_id", "") or ""),
            ),
        )

    if candidates:
        best = candidates[0]
        weight_class = _normalize_live_weight_class(best.get("weight_class"))
        if weight_class is not None:
            is_title_bout = bool(best.get("is_title_bout", False))
            try:
                num_rounds = int(best.get("num_rounds"))
            except (TypeError, ValueError):
                num_rounds = 5 if (bool(best.get("is_main_event", False)) or is_title_bout) else 3
            return {
                "weight_class": weight_class,
                "is_title_bout": is_title_bout,
                "is_empty_arena": best.get("is_empty_arena"),
                "num_rounds": num_rounds,
            }

    latest_head_to_head = _latest_local_head_to_head_event_date(fighter_a, fighter_b)
    if latest_head_to_head is not None and requested_commence is not None:
        fallback_block_until = latest_head_to_head + timedelta(
            days=_LIVE_RECENT_MATCHUP_FALLBACK_BLOCK_DAYS
        )
        if requested_commence <= fallback_block_until:
            logger.info(
                "Refusing fallback live context for %s vs %s: local history already has this matchup on %s and no upcoming UFC card row matched requested start %s",
                fighter_a,
                fighter_b,
                latest_head_to_head.date().isoformat(),
                requested_commence.date().isoformat(),
            )
            return None

    if allow_off_card_history_fallback:
        inferred_wc = _infer_weight_class_from_history(fighter_a, fighter_b)
        if inferred_wc:
            logger.info(
                "No UFCStats context for %s vs %s (not on upcoming UFC cards) — "
                "inferred weight class '%s' from fight history",
                fighter_a, fighter_b, inferred_wc,
            )
            return {
                "weight_class": inferred_wc,
                "is_title_bout": False,
                "is_empty_arena": False,
                "num_rounds": 3,
            }

    inferred_lookup_wc = _infer_weight_class_from_near_term_ufc_lookup(
        fighter_a,
        fighter_b,
        requested_commence=requested_commence,
        live_event_contexts=live_event_contexts,
    )
    if inferred_lookup_wc:
        logger.info(
            "No UFCStats card-row context for %s vs %s on %s — "
            "inferred weight class '%s' from processed/UFCStats fighter history",
            fighter_a,
            fighter_b,
            requested_commence.date().isoformat() if requested_commence is not None else "?",
            inferred_lookup_wc,
        )
        return {
            "weight_class": inferred_lookup_wc,
            "is_title_bout": False,
            "is_empty_arena": False,
            "num_rounds": 3,
        }

    return None


def cmd_scrape(args):
    """Scrape latest UFC data from UFCStats.com."""
    from src.data.scraper import scrape_all_fights, scrape_all_fighters

    logger.info("Starting UFC data scrape...")
    if args.fighters_only:
        scrape_all_fighters()
    elif args.fights_only:
        scrape_all_fights()
    else:
        scrape_all_fighters()
        scrape_all_fights()
    logger.info("Scraping complete.")


def cmd_train(args):
    """Load data, build features, and train models."""
    from src.data.kaggle_loader import save_processed
    from src.features.build_features import build_features, save_features
    from src.model.train import train_all_models

    cli_spec_name = getattr(args, "spec", None)
    spec = getattr(args, "training_spec", None)
    if spec is None and cli_spec_name:
        spec = _resolve_named_training_spec_arg(cli_spec_name)
    spec = spec or _default_training_spec()
    logger.info("Using training spec: %s", spec.name)

    output_subdir = getattr(args, "output_subdir", None)
    processed_output_dir = (PROCESSED_DATA_DIR / output_subdir) if output_subdir else PROCESSED_DATA_DIR
    models_output_dir = (MODELS_DIR / output_subdir) if output_subdir else None
    test_set_path = (processed_output_dir / "test_set.csv") if output_subdir else None

    # Step 1: Load data
    logger.info("Loading data...")
    filepath = Path(args.data) if args.data else None
    fights_df = _load_training_dataframe(data_path=filepath, spec=spec)
    save_processed(
        fights_df,
        filename=(Path(output_subdir) / "fights_cleaned.csv") if output_subdir else "fights_cleaned.csv",
    )

    # Step 2: Build features
    logger.info("Building features...")
    features_df = build_features(fights_df)
    save_features(
        features_df,
        filename=(Path(output_subdir) / "features.csv") if output_subdir else "features.csv",
    )

    # Step 3: Train models
    logger.info("Training models...")
    train_kwargs = {}
    if models_output_dir is not None:
        train_kwargs["models_dir"] = models_output_dir
    if test_set_path is not None:
        train_kwargs["test_set_path"] = test_set_path
    results = train_all_models(features_df, spec=spec, **train_kwargs)

    logger.info(
        "Training complete. Models saved to %s",
        results["models_dir"] if results.get("models_dir") is not None else MODELS_DIR,
    )
    logger.info(f"Train size: {len(results['train_df'])}, Test size: {len(results['test_df'])}")


def cmd_evaluate(args):
    """Evaluate trained models on test set."""
    import pandas as pd
    from src.model.evaluate import compare_models, print_feature_importance

    test_path = PROCESSED_DATA_DIR / "test_set.csv"
    if not test_path.exists():
        logger.error("Test set not found. Run 'train' first.")
        return

    test_df = pd.read_csv(test_path, parse_dates=["event_date"])

    models = args.models.split(",") if args.models else ["xgboost", "logistic"]
    compare_models(test_df, model_names=models)
    print_feature_importance(model_name=models[0])


def cmd_backtest(args):
    """Run backtest on historical data. Defaults to walk-forward."""
    import pandas as pd
    from src.model.train import assert_model_matches_test_set, load_model
    from src.strategy.backtest import run_backtest, plot_backtest

    if args.static:
        # Static backtest: single train/test split
        logger.info("Running static backtest (single train/test split)...")
        test_path_arg = getattr(args, "test_set_path", None)
        test_path = Path(test_path_arg).expanduser() if test_path_arg else PROCESSED_DATA_DIR / "test_set.csv"
        if not test_path.exists():
            logger.error("Test set not found. Run 'train' first.")
            return

        model_ref = getattr(args, "model_path", None) or args.model
        model_result = load_model(model_ref)
        if getattr(args, "allow_mismatch", False):
            try:
                assert_model_matches_test_set(model_result, test_set_path=test_path)
            except (FileNotFoundError, ValueError) as exc:
                logger.warning(
                    "Proceeding with static backtest despite model/test-set mismatch because "
                    "--allow-mismatch was set: %s",
                    exc,
                )
        else:
            assert_model_matches_test_set(model_result, test_set_path=test_path)

        test_df = pd.read_csv(test_path, parse_dates=["event_date"])
        no_odds_model_arg = _resolve_no_odds_model_arg(model_ref)
        try:
            agreement_model_result = load_model(no_odds_model_arg) if no_odds_model_arg is not None else None
        except FileNotFoundError:
            agreement_model_result = None

        result = run_backtest(
            test_df,
            model_name=args.model,
            model_result=model_result,
            agreement_model_result=agreement_model_result,
            initial_bankroll=args.bankroll,
            min_edge=args.min_edge,
            kelly_fraction=args.kelly,
            execution_mode=getattr(args, "execution_mode", "legacy"),
        )

        plot_backtest(result)
    else:
        # Walk-forward backtest (default): retrain every N months
        from src.features.build_features import build_features
        from src.data.kaggle_loader import load_kaggle_dataset
        from src.strategy.backtest import run_walkforward_backtest

        logger.info("Running walk-forward backtest (retraining every "
                     f"{args.retrain_months} months)...")
        logger.info("Use --static for single train/test split instead.")

        fights_df = load_kaggle_dataset()
        features_df = build_features(fights_df)

        result = run_walkforward_backtest(
            features_df,
            retrain_months=args.retrain_months,
            initial_train_years=args.initial_years,
            initial_bankroll=args.bankroll,
            min_edge=args.min_edge,
            kelly_fraction=args.kelly,
            execution_mode=getattr(args, "execution_mode", "legacy"),
        )

        plot_backtest(result)


def cmd_backtest_compare(args):
    """Run comparison backtest: full model vs no-odds baseline."""
    import pandas as pd
    from src.data.kaggle_loader import load_kaggle_dataset
    from src.features.build_features import build_features
    from src.strategy.backtest import (
        run_comparison_backtest,
        run_walkforward_strategy_comparison,
        plot_backtest,
    )

    if args.walkforward:
        logger.info(
            "Running walk-forward comparison "
            f"(retraining every {args.retrain_months} months)..."
        )
        features_path = PROCESSED_DATA_DIR / "features.csv"
        if features_path.exists():
            logger.info(f"Loading cached features from {features_path}")
            features_df = pd.read_csv(features_path, parse_dates=["event_date"])
        else:
            logger.info("No cached features found. Building features from Kaggle dataset...")
            fights_df = load_kaggle_dataset()
            features_df = build_features(fights_df)

        comparison = run_walkforward_strategy_comparison(
            features_df,
            retrain_months=args.retrain_months,
            initial_train_years=args.initial_years,
            initial_bankroll=args.bankroll,
            min_edge=args.min_edge,
            kelly_fraction=args.kelly,
        )

        if not comparison["summary"].empty:
            logger.info("\nWalk-forward strategy summary:")
            logger.info(comparison["summary"].to_string(index=False))

        artifacts = comparison.get("artifacts", {})
        if artifacts:
            logger.info("\nArtifacts:")
            for label, path in artifacts.items():
                logger.info(f"  {label}: {path}")
        return

    test_path = PROCESSED_DATA_DIR / "test_set.csv"
    if not test_path.exists():
        logger.error("Test set not found. Run 'train' first.")
        return

    test_df = pd.read_csv(test_path, parse_dates=["event_date"])

    results = run_comparison_backtest(
        test_df,
        initial_bankroll=args.bankroll,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly,
    )

    for name, result in results.items():
        plot_backtest(result)


def cmd_backfill_odds(args):
    """Backfill historical odds from The Odds API for backtesting."""
    import pandas as pd
    from src.data.historical_backfill import run_backfill

    test_path = PROCESSED_DATA_DIR / "test_set.csv"
    if not test_path.exists():
        logger.error("Test set not found. Run 'train' first.")
        return

    test_df = pd.read_csv(test_path, parse_dates=["event_date"])

    logger.info(f"Backfilling historical odds for {len(test_df)} fights...")
    logger.info(f"Unique event dates: {test_df['event_date'].nunique()}")
    logger.info(f"Snapshot offsets: {args.offsets} days before event")

    offsets = [int(x) for x in args.offsets.split(",")]
    result = run_backfill(test_df, offsets=offsets, resume=not args.fresh)

    logger.info(f"Backfill complete: {len(result)} total records")


def cmd_sensitivity(args):
    """Run sensitivity analysis across parameter combinations."""
    import pandas as pd
    from src.strategy.backtest import sensitivity_analysis

    test_path = PROCESSED_DATA_DIR / "test_set.csv"
    if not test_path.exists():
        logger.error("Test set not found. Run 'train' first.")
        return

    test_df = pd.read_csv(test_path, parse_dates=["event_date"])
    sensitivity_analysis(test_df, model_name=args.model)


def cmd_walkforward(args):
    """Run walk-forward backtest with periodic model retraining."""
    import pandas as pd
    from src.features.build_features import build_features
    from src.data.kaggle_loader import load_kaggle_dataset
    from src.strategy.backtest import run_walkforward_backtest, plot_backtest

    logger.info("Loading data and building features for walk-forward backtest...")
    fights_df = load_kaggle_dataset()
    features_df = build_features(fights_df)

    result = run_walkforward_backtest(
        features_df,
        retrain_months=args.retrain_months,
        initial_train_years=args.initial_years,
        initial_bankroll=args.bankroll,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly,
        execution_mode=getattr(args, "execution_mode", "legacy"),
    )

    plot_backtest(result)


def ensure_model_fresh(model_name: str = "xgboost"):
    """Auto-retrain models if they're older than MODEL_RETRAIN_MONTHS."""
    import time
    from src.config import MODELS_DIR, MODEL_RETRAIN_MONTHS

    explicit_model_path = _explicit_model_path(model_name)
    if explicit_model_path is not None:
        logger.info("Skipping auto-retrain freshness check for explicit model artifact: %s", explicit_model_path)
        return

    model_path = MODELS_DIR / f"{model_name}_model.pkl"
    if not model_path.exists():
        logger.info(f"No model found at {model_path}. Training from scratch...")
        try:
            cmd_train(argparse.Namespace(data=None, training_spec=_default_training_spec()))
        except Exception:
            logger.error("Initial training failed for %s; no usable model artifact exists", model_name)
            raise
        return

    model_age_days = (time.time() - model_path.stat().st_mtime) / 86400
    max_age_days = MODEL_RETRAIN_MONTHS * 30

    if model_age_days > max_age_days:
        training_spec = _load_training_spec_from_artifact(model_name)
        logger.info(
            f"Model is {model_age_days:.0f} days old (max: {max_age_days} days). "
            f"Auto-retraining with spec '{training_spec.name}'..."
        )
        try:
            cmd_train(argparse.Namespace(data=None, training_spec=training_spec))
        except Exception as exc:
            logger.warning(
                "Auto-retrain failed for %s; continuing with existing artifact at %s: %s",
                model_name,
                model_path,
                exc,
            )
    else:
        logger.info(
            f"Model is {model_age_days:.0f} days old "
            f"(retrain threshold: {max_age_days} days). Using existing model."
        )


def cmd_predict(args):
    """Predict upcoming UFC fights using blended model-market approach."""
    from src.data.odds_client import OddsClient
    from src.model.predict import predict_fight
    from src.model.train import load_model
    from src.strategy.value import compute_independent_blend_probs, _passes_filters
    from src.data.fighter_lookup import build_fight_features
    from src.data.line_tracker import detect_injury_or_cancellation
    from src.config import MIN_FIGHTER_FIGHTS

    logger.info("Fetching upcoming UFC odds...")
    odds_client = OddsClient()

    try:
        odds = odds_client.get_live_odds()
        odds_df = odds_client.odds_to_dataframe(odds)
        consensus = odds_client.get_consensus_odds(odds_df)
    except Exception as e:
        logger.error(f"Failed to fetch odds: {e}")
        logger.info("Make sure ODDS_API_KEY is set in .env")
        return

    if consensus.empty:
        logger.info("No upcoming UFC fights with odds found.")
        return

    logger.info(f"\nUpcoming UFC fights with predictions:")
    logger.info(f"{'='*80}")

    ensure_model_fresh(args.model)
    model_result = load_model(args.model)
    inference_spec = _training_spec_from_model_result(model_result)
    no_odds_model_arg = _resolve_no_odds_model_arg(args.model)
    try:
        no_odds_result = load_model(no_odds_model_arg) if no_odds_model_arg is not None else None
    except FileNotFoundError:
        no_odds_result = None
    runtime_bundle_summary = _resolve_runtime_bundle_summary(
        model_result=model_result,
        no_odds_result=no_odds_result,
    )
    runtime_processed_data_dir = (
        Path(runtime_bundle_summary["processed_dir"]) if runtime_bundle_summary is not None else None
    )
    live_event_contexts = _load_live_event_contexts_for_fights(consensus)

    for _, fight in consensus.iterrows():
        fighter_a = fight["fighter_a"]
        fighter_b = fight["fighter_b"]
        market_a = fight["a_fair_prob_avg"]
        market_b = fight["b_fair_prob_avg"]
        can_trade, start_reason, _ = _live_fight_is_tradeable(fight.get("commence_time"))
        if not can_trade:
            _log_live_fight_skip_once(fight, start_reason)
            continue
        event_context = _resolve_live_event_context(
            fight,
            live_event_contexts,
            allow_off_card_history_fallback=False,
        )
        if event_context is None:
            _log_live_fight_skip_once(
                fight,
                _missing_live_event_context_reason(fighter_a, fighter_b),
            )
            continue

        # Check for injury/cancellation signals
        injury_tag = ""
        try:
            injury = detect_injury_or_cancellation(
                fighter_a, fighter_b,
                current_odds={"a_prob": market_a, "b_prob": market_b},
            )
            if injury["suspected"]:
                injury_tag = f"  [INJURY ALERT: {injury['reason']}]"
        except Exception as exc:
            logger.warning("Injury/cancellation check failed for %s vs %s: %s", fighter_a, fighter_b, exc)

        # Build full feature vector from live fighter stats + odds
        odds_features = {
            "a_implied_prob": market_a,
            "b_implied_prob": market_b,
            "diff_implied_prob": market_a - market_b,
        }
        features = build_fight_features(
            fighter_a,
            fighter_b,
            odds_features=odds_features,
            weight_class=event_context["weight_class"],
            is_title_bout=event_context["is_title_bout"],
            is_empty_arena=event_context.get("is_empty_arena"),
            num_rounds=event_context["num_rounds"],
            event_id=fight.get("event_id"),
            commence_time=fight.get("commence_time"),
            prefer_live_refresh=True,
            training_spec=inference_spec,
            processed_data_dir=runtime_processed_data_dir,
        )
        logger.info(f"  Built {sum(1 for v in features.values() if v is not None)} features for {fighter_a} vs {fighter_b}")
        a_fights, b_fights = _resolve_live_fight_counts(features, fighter_a, fighter_b)

        exp_warnings = []
        if a_fights < MIN_FIGHTER_FIGHTS:
            exp_warnings.append(f"{fighter_a} ({a_fights} UFC fights)")
        if b_fights < MIN_FIGHTER_FIGHTS:
            exp_warnings.append(f"{fighter_b} ({b_fights} UFC fights)")

        try:
            pred = predict_fight(features, model_result=model_result)
        except Exception as e:
            logger.warning(f"Prediction failed for {fighter_a} vs {fighter_b}: {e}")
            continue

        # No-odds model prediction for agreement
        no_odds_a = no_odds_b = None
        if no_odds_result:
            try:
                no_odds_pred = predict_fight(features, model_result=no_odds_result)
                no_odds_a = no_odds_pred["prob_a"]
                no_odds_b = no_odds_pred["prob_b"]
            except Exception as exc:
                logger.warning("No-odds model prediction failed for %s vs %s: %s", fighter_a, fighter_b, exc)

        # Blend model with market (independent weights for both sides)
        blend_a, blend_b = compute_independent_blend_probs(
            pred["prob_a"], market_a, no_odds_a,
            pred["prob_b"], market_b, no_odds_b,
        )
        edge_a = blend_a - market_a
        edge_b = blend_b - market_b

        # Check if value bet passes all filters (including fighter experience)
        value_a = edge_a >= MIN_EDGE_THRESHOLD and _passes_filters(
            blend_a, market_a, edge_a, fighter_a, no_odds_a,
            a_num_fights=a_fights, b_num_fights=b_fights,
        )
        value_b = edge_b >= MIN_EDGE_THRESHOLD and _passes_filters(
            blend_b, market_b, edge_b, fighter_b, no_odds_b,
            a_num_fights=a_fights, b_num_fights=b_fights,
        )
        value_tag = "  *** VALUE ***" if value_a or value_b else ""
        if exp_warnings:
            value_tag += f"  [LOW EXP: {', '.join(exp_warnings)}]"
        if injury_tag:
            value_tag += injury_tag

        no_odds_str = ""
        if no_odds_a is not None:
            no_odds_str = (
                f"\n  No-odds: {fighter_a} {no_odds_a:.1%} | "
                f"{fighter_b} {no_odds_b:.1%}"
            )

        logger.info(
            f"\n{fighter_a} vs {fighter_b}"
            f"\n  Market:  {fighter_a} {market_a:.1%} | "
            f"{fighter_b} {market_b:.1%} "
            f"({fight['num_bookmakers']:.0f} books)"
            f"\n  Model:   {fighter_a} {pred['prob_a']:.1%} | "
            f"{fighter_b} {pred['prob_b']:.1%}"
            f"{no_odds_str}"
            f"\n  Blended: {fighter_a} {blend_a:.1%} | "
            f"{fighter_b} {blend_b:.1%} "
            f"(w={BLEND_WEIGHT:.0%} model)"
            f"\n  Edge:    {fighter_a} {edge_a:+.1%} | {fighter_b} {edge_b:+.1%}"
            f"{value_tag}"
        )

def cmd_ufc_refresh_scheduled(args):
    """Refresh UFC active-roster raw data and rebuild processed artifacts for scheduled runs."""
    from scripts.run_scheduled_ufc_refresh import run_scheduled_refresh

    summary = run_scheduled_refresh(
        dataset_variant=args.dataset_variant,
        output_subdirs=args.output_subdir,
        limit_fighters=args.limit_fighters,
        audit_json_path=args.audit_json_path,
        audit_csv_path=args.audit_csv_path,
        skip_rebuild=args.skip_rebuild,
        skip_audit=args.skip_audit,
    )
    logger.info("Scheduled UFC refresh complete:\n%s", json.dumps(summary, indent=2))

def cmd_monitor(args):
    """Run continuous monitoring of upcoming UFC events."""
    from src.data.live_monitor import run_monitoring_pass
    from src.data.line_tracker import run_line_tracking_pass
    import time as _time

    interval_hours = args.interval
    logger.info(f"Starting continuous monitor (every {interval_hours} hours)")
    logger.info("Press Ctrl+C to stop")

    while True:
        try:
            # Run monitoring pass
            signals = run_monitoring_pass()

            # Also track lines
            line_summary = run_line_tracking_pass()

            logger.info(
                f"\nNext check in {interval_hours} hours. "
                f"Events tracked: {len(signals['events'])}, "
                f"Sharp moves: {line_summary.get('sharp_moves', 0)}"
            )

            _time.sleep(interval_hours * 3600)

        except KeyboardInterrupt:
            logger.info("Monitor stopped by user.")
            break
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            _time.sleep(300)  # Wait 5 min on error, then retry


def cmd_track_lines(args):
    """Take a snapshot of current odds and analyze line movement."""
    from src.data.line_tracker import run_line_tracking_pass

    summary = run_line_tracking_pass()

    if summary.get("analyses"):
        logger.info(f"\nLine movement for {summary['fights_analyzed']} fights:")
        for fight, analysis in summary["analyses"].items():
            if analysis.get("opening_prob_a") is not None:
                logger.info(
                    f"  {fight}: "
                    f"Opened {analysis['opening_prob_a']:.1%} -> "
                    f"Now {analysis['current_prob_a']:.1%} "
                    f"({analysis['movement']:+.1%} {analysis['direction']})"
                    f"{' *** SHARP ***' if analysis['is_sharp_move'] else ''}"
                    f"{' *** STEAM ***' if analysis['steam_move'] else ''}"
                )


def cmd_signals(args):
    """Check all pre-fight signals for upcoming events."""
    from src.data.live_monitor import run_monitoring_pass
    from src.data.prefight_signals import collect_prefight_signals

    signals = run_monitoring_pass()

    for event in signals.get("events", []):
        logger.info(f"\n{'='*60}")
        logger.info(f"Event: {event['title']} ({event['days_to_event']} days away)")
        logger.info(f"{'='*60}")

        for fight in event.get("fights", []):
            fa = fight["fighter_a"]
            fb = fight["fighter_b"]

            # Check if either fighter has signals
            a_short = any(
                r["new_fighter"].lower() == fa.lower()
                for r in signals.get("short_notice_replacements", [])
            )
            b_short = any(
                r["new_fighter"].lower() == fb.lower()
                for r in signals.get("short_notice_replacements", [])
            )
            a_missed = any(
                m["fighter"].lower() == fa.lower()
                for m in signals.get("missed_weights", [])
            )
            b_missed = any(
                m["fighter"].lower() == fb.lower()
                for m in signals.get("missed_weights", [])
            )

            a_over = next(
                (m["over_by"] for m in signals.get("missed_weights", [])
                 if m["fighter"].lower() == fa.lower()), 0.0
            )
            b_over = next(
                (m["over_by"] for m in signals.get("missed_weights", [])
                 if m["fighter"].lower() == fb.lower()), 0.0
            )

            fight_signals = collect_prefight_signals(
                fighter_a=fa,
                fighter_b=fb,
                event_title=event["title"],
                a_is_short_notice=a_short,
                b_is_short_notice=b_short,
                a_missed_weight=a_missed,
                b_missed_weight=b_missed,
                a_weight_over=a_over,
                b_weight_over=b_over,
            )

            logger.info(f"\n  {fa} vs {fb}:")
            if fight_signals["flags"]:
                for flag in fight_signals["flags"]:
                    logger.info(f"    * {flag}")
            else:
                logger.info(f"    No signals detected")


def cmd_positions(args):
    """Show current Polymarket positions and P&L."""
    from src.polymarket.monitor import PositionMonitor

    monitor = PositionMonitor()
    monitor.print_status()


def cmd_dashboard(args):
    """Run live-updating bet & P&L dashboard."""
    from src.polymarket.tracker import (
        run_live_dashboard,
        auto_settle_from_polymarket,
        BetLedger,
    )
    from src.polymarket.client import ClobClientWrapper
    from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER

    # Auto-settle any resolved markets across all trader ledgers
    settled = 0
    for path in [SINGLE_LEDGER, CONVICTION_LEDGER]:
        if Path(path).exists():
            ledger = BetLedger(path=path)
            settled += auto_settle_from_polymarket(ledger)
    if settled:
        logger.info(f"Auto-settled {settled} bets from resolved markets")

    clob = None
    if not args.offline:
        try:
            clob = ClobClientWrapper()
            logger.info("Connected to Polymarket CLOB for live prices")
        except Exception as e:
            logger.warning(f"Could not connect to CLOB (running offline): {e}")

    run_live_dashboard(
        clob_client=clob,
        refresh_seconds=args.refresh,
        include_dry_runs=not args.real_only,
    )


def cmd_web(args):
    """Launch the web dashboard."""
    from src.web.app import start_server
    from src.polymarket.client import ClobClientWrapper

    clob = None
    if not args.offline:
        try:
            clob = ClobClientWrapper()
            logger.info("Connected to Polymarket CLOB for live prices")
        except Exception as e:
            logger.warning(f"Running offline (no CLOB): {e}")

    start_server(port=args.port, debug=args.debug, clob_client=clob)


def cmd_settle(args):
    """Manually settle a bet or auto-settle from Polymarket."""
    from src.polymarket.tracker import (
        BetLedger,
        auto_settle_from_polymarket,
        load_all_trader_ledgers,
        resolve_merged_bet_reference,
    )
    from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER

    if args.auto:
        settled = 0
        for path in [SINGLE_LEDGER, CONVICTION_LEDGER]:
            if Path(path).exists():
                ledger = BetLedger(path=path)
                settled += auto_settle_from_polymarket(ledger)
        logger.info(f"Auto-settled {settled} bets")
        return

    if args.bet_id and args.result:
        won = args.result.lower() in ("win", "won", "w", "yes")
        target = resolve_merged_bet_reference(args.bet_id, require_open=True)
        if target is None:
            existing = resolve_merged_bet_reference(args.bet_id, require_open=False)
            if existing is not None:
                logger.warning(f"Bet #{args.bet_id} is not open")
            else:
                logger.warning(f"Bet #{args.bet_id} not found in any trader ledger")
        else:
            ledger = BetLedger(path=target["ledger_path"])
            result = ledger.settle_bet(target["original_id"], won)
            if result.ok:
                logger.info(f"Settled bet #{args.bet_id}: {'WON' if won else 'LOST'}")
            elif result.status == "not_open":
                logger.warning(f"Bet #{args.bet_id} is no longer open")
            else:
                logger.warning(f"Bet #{args.bet_id} no longer exists")
    else:
        # Show open bets for manual settlement
        ledger = load_all_trader_ledgers()
        open_bets = ledger.open_bets
        if not open_bets:
            logger.info("No open bets to settle.")
            return

        logger.info(f"\nOpen bets ({len(open_bets)}):")
        for bet in open_bets:
            dry = " [DRY RUN]" if bet.get("dry_run") else ""
            logger.info(
                f"  #{bet['id']}: ${bet['amount']:.2f} on {bet['fighter']} "
                f"vs {bet['opponent']} @ {bet['price']:.4f}{dry}"
            )
        logger.info(
            "\nTo settle: python -m src.bot settle --bet-id <id> --result win/loss"
        )
        logger.info(
            "To auto-settle from Polymarket: python -m src.bot settle --auto"
        )


def cmd_redeem(args):
    """Redeem resolved Polymarket positions that are ready to claim."""
    from src.polymarket.tracker import auto_redeem_positions_from_polymarket

    summary = auto_redeem_positions_from_polymarket(wait=not args.no_wait)
    reason = summary.get("reason")
    if reason == "no_redeemable_positions":
        logger.info("No redeemable Polymarket positions found")
        return
    if reason == "redeemer_not_configured":
        logger.warning(
            "Redeem is not configured. Set POLYMARKET_RELAYER_API_KEY "
            "or the POLYMARKET_BUILDER_* credentials first."
        )
        return
    if reason == "redeem_submission_pending":
        logger.info("Redeem already submitted and is still awaiting relayer confirmation")
        return

    if summary.get("redeemed_conditions"):
        logger.info(
            "Redeemed %s position(s) across %s condition(s)",
            summary.get("redeemed_positions", 0),
            summary.get("redeemed_conditions", 0),
        )
    elif summary.get("submitted_conditions"):
        logger.info(
            "Submitted redeem for %s position(s) across %s condition(s)",
            summary.get("submitted_positions", 0),
            summary.get("submitted_conditions", 0),
        )
    if summary.get("errors"):
        logger.warning("Redeem completed with %s error(s)", len(summary["errors"]))



def cmd_duo_live(args):
    """Run duo traders (S+C) with Single Trader evaluating first, Conviction on remainder."""
    if not getattr(args, "dry_run", True):
        try:
            assert_real_trading_allowed(
                model_name=getattr(args, "model", None),
                host="127.0.0.1",
                startup_source="cli",
            )
        except RuntimeError as exc:
            logger.error(str(exc))
            return {"status": "error", "reason": str(exc)}

    from src.data.odds_client import OddsClient
    from src.model.predict import predict_fight
    from src.model.train import load_model
    from src.polymarket.markets import get_ufc_fight_markets
    from src.polymarket.client import ClobClientWrapper
    from src.strategy.duo_trader import _resolve_total_bankroll, run_duo_traders
    from src.data.line_tracker import get_line_movement_features, detect_injury_or_cancellation
    from src.data.fighter_lookup import build_fight_features
    from src.config import MIN_FIGHTER_FIGHTS, INJURY_BLOCK_BETS
    import pandas as pd

    dry_run = args.dry_run
    mode = "DRY RUN" if dry_run else "LIVE"
    logger.info(f"Starting DUO TRADER bot in {mode} mode...")
    progress_callback = getattr(args, "progress_callback", None)

    def _report_progress(message: str) -> None:
        if not callable(progress_callback):
            return
        try:
            progress_callback(message)
        except Exception as exc:
            logger.debug("Live betting progress callback failed: %s", exc)

    clob = None if dry_run else ClobClientWrapper()

    _report_progress("Cycle active: loading model artifacts")
    ensure_model_fresh(args.model)
    model_result = load_model(args.model)
    inference_spec = _training_spec_from_model_result(model_result)
    no_odds_model_arg = _resolve_no_odds_model_arg(args.model)
    try:
        no_odds_result = load_model(no_odds_model_arg) if no_odds_model_arg is not None else None
    except FileNotFoundError:
        no_odds_result = None
    runtime_bundle_summary = _resolve_runtime_bundle_summary(
        model_result=model_result,
        no_odds_result=no_odds_result,
    )
    runtime_processed_data_dir = (
        Path(runtime_bundle_summary["processed_dir"]) if runtime_bundle_summary is not None else None
    )
    runtime_signature = _prediction_runtime_signature(
        model_result=model_result,
        no_odds_result=no_odds_result,
        runtime_bundle_summary=runtime_bundle_summary,
    )

    # Extract feature cols/medians and global importance for cache enrichment
    _feat_cols = model_result["feature_cols"]
    _col_medians = model_result["col_medians"]
    _global_importance = sorted(
        model_result.get("feature_importance", {}).items(),
        key=lambda x: x[1], reverse=True,
    )[:25]
    cache_write_warning_emitted = False
    shap_state = {
        "initialized": False,
        "explainer": None,
        "base_value": None,
    }

    def _ensure_shap_state() -> tuple[object | None, float | None]:
        if shap_state["initialized"]:
            return shap_state["explainer"], shap_state["base_value"]

        shap_state["initialized"] = True
        try:
            import numpy as np
            import shap

            raw_model = model_result.get("raw_model")
            if raw_model is not None:
                explainer = shap.TreeExplainer(raw_model)
                expected_value = np.atleast_1d(explainer.expected_value)
                shap_state["explainer"] = explainer
                shap_state["base_value"] = float(expected_value[1]) if len(expected_value) > 1 else float(expected_value[0])
        except ImportError:
            logger.info("shap not installed — predictions page will use feature highlights only")
        except Exception as exc:
            logger.warning(f"Failed to create SHAP explainer: {exc}")
        return shap_state["explainer"], shap_state["base_value"]

    def _feature_display_name(col: str) -> str:
        """Convert feature column name to human-readable display name."""
        overrides = {
            "diff_roll_slpm": "Strikes Landed/Min",
            "diff_roll_sapm": "Strikes Absorbed/Min",
            "diff_roll_str_acc": "Striking Accuracy %",
            "diff_roll_str_def": "Strike Defense %",
            "diff_roll_td_avg": "Takedowns/Fight",
            "diff_roll_td_acc": "Takedown Accuracy %",
            "diff_roll_td_def": "Takedown Defense %",
            "diff_roll_sub_avg": "Submissions/Fight",
            "diff_roll_kd": "Knockdowns/Fight",
            "diff_roll_won": "Recent Win Rate",
            "diff_current_win_streak": "Win Streak",
            "diff_lose_streak": "Losing Streak",
            "diff_num_fights": "UFC Experience",
            "diff_total_rounds": "Rounds Fought",
            "diff_age": "Age",
            "diff_height": "Height (cm)",
            "diff_reach": "Reach (cm)",
            "diff_weight": "Weight",
            "diff_ko_rate": "KO Win Rate",
            "diff_sub_rate": "Submission Win Rate",
            "diff_dec_rate": "Decision Win Rate",
            "diff_win_pct": "Overall Win %",
            "diff_implied_prob": "Betting Odds",
            "diff_wc_rank": "Division Ranking",
            "diff_pfp_rank": "P4P Ranking",
            "diff_strike_diff": "Net Striking",
            "diff_striker_edge": "Striking Advantage",
            "diff_grappler_edge": "Grappling Advantage",
            "diff_days_since_last_fight": "Days Since Last Fight",
            "diff_cage_rust": "Ring Rust",
            "diff_title_bouts": "Title Fight Experience",
            "diff_opp_td_avg": "Opponent Takedowns Faced",
            "diff_opp_sub_avg": "Opponent Subs Faced",
            "diff_opp_str_acc": "Opponent Striking Accuracy",
            "diff_ko_odds_prob": "KO Finish Likelihood",
            "diff_dec_odds_prob": "Decision Likelihood",
        }
        if col in overrides:
            return overrides[col]

        # Handle a_/b_ prefixed features (individual fighter stats)
        for prefix, label in [("a_", "A "), ("b_", "B ")]:
            if col.startswith(prefix):
                suffix = col[len(prefix):]
                diff_key = f"diff_{suffix}"
                if diff_key in overrides:
                    return label + overrides[diff_key]
                break

        # Fallback: clean up the raw name
        name = col.replace("diff_", "").replace("a_", "A ").replace("b_", "B ").replace("roll_", "").replace("opp_", "Opp ").replace("_", " ").title()
        return name

    def _persist_prediction_cache(rows, *, announce: bool) -> None:
        nonlocal cache_write_warning_emitted

        try:
            predictions_cache = LOGS_DIR / "predictions_cache.json"
            temp_cache = predictions_cache.with_name(f"{predictions_cache.name}.tmp")
            import json as _json

            payload = {
                "schema_version": PREDICTION_CACHE_SCHEMA_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "predictions": rows,
                "global_feature_importance": [
                    {
                        "feature": feature_name,
                        "display_name": _feature_display_name(feature_name),
                        "importance": round(importance, 4),
                    }
                    for feature_name, importance in _global_importance
                ],
            }
            payload = _sanitize_prediction_cache_value(payload)
            temp_cache.write_text(_json.dumps(payload, default=str), encoding="utf-8")
            temp_cache.replace(predictions_cache)
            cache_write_warning_emitted = False
            if announce:
                logger.info(f"Cached {len(rows)} predictions for dashboard")
        except Exception as e:
            if announce or not cache_write_warning_emitted:
                logger.warning(f"Failed to cache predictions: {e}")
            cache_write_warning_emitted = True

    # 1. Fetch bookmaker consensus odds
    _report_progress("Cycle active: fetching bookmaker odds")
    logger.info("Fetching bookmaker odds from The Odds API...")
    odds_client = OddsClient()
    try:
        raw_odds = odds_client.get_live_odds()
        odds_df = odds_client.odds_to_dataframe(raw_odds)
        consensus = odds_client.get_consensus_odds(odds_df)
    except Exception as e:
        logger.error(f"Failed to fetch odds: {e}")
        return {"status": "error", "reason": f"odds_fetch_failed: {e}"}

    if consensus.empty:
        logger.info("No upcoming UFC fights with bookmaker odds found.")
    else:
        logger.info(f"Got bookmaker consensus for {len(consensus)} fights")
    _report_progress(f"Cycle active: fetched bookmaker consensus for {len(consensus)} fights")

    # 2. Get Polymarket markets
    _report_progress("Cycle active: fetching Polymarket UFC markets")
    logger.info("Fetching Polymarket UFC markets...")
    try:
        markets = get_ufc_fight_markets()
    except Exception as e:
        logger.warning(f"Failed to fetch Polymarket markets: {e}")
        markets = pd.DataFrame()

    if markets.empty:
        logger.info("No active UFC markets found on Polymarket — predictions will still be cached for the dashboard.")
    else:
        logger.info(f"Found {len(markets)} active Polymarket UFC markets")
    _report_progress(f"Cycle active: fetched {len(markets)} Polymarket UFC markets")

    # 3. Generate predictions (same for both traders — they differ only in blend weight)
    existing_cache = _load_existing_prediction_cache()
    current_feed_keys = {
        _prediction_cache_key(fight)
        for _, fight in consensus.iterrows()
    }
    prediction_rows_by_key: dict[str, dict] = {
        cache_key: dict(row)
        for cache_key, row in existing_cache.items()
        if cache_key in current_feed_keys
    }
    retained_prediction_keys: set[str] = set(prediction_rows_by_key)
    validated_prediction_keys: set[str] = set()
    live_event_contexts: list[dict] | None = None

    def _ensure_live_event_contexts() -> list[dict]:
        nonlocal live_event_contexts
        if live_event_contexts is None:
            _report_progress("Cycle active: loading UFC event context for uncached fights")
            live_event_contexts = _load_live_event_contexts_for_fights(consensus)
        return live_event_contexts

    def _persist_current_prediction_cache(*, announce: bool, validated_only: bool = False) -> None:
        allowed_keys = validated_prediction_keys if validated_only else retained_prediction_keys
        rows = _prediction_cache_rows_for_write(
            prediction_rows_by_key,
            allowed_keys=allowed_keys,
        )
        _persist_prediction_cache(rows, announce=announce)

    logger.info("Generating model predictions...")
    _operator_features_by_fight: dict[str, dict] = {}  # for LLM Operator
    _operator_provenance_by_fight: dict[str, dict] = {}
    total_consensus_fights = len(consensus)
    for idx, (_, fight) in enumerate(consensus.iterrows(), start=1):
        fighter_a = fight["fighter_a"]
        fighter_b = fight["fighter_b"]
        fight_cache_key = _prediction_cache_key(fight)
        fight_key = f"{fighter_a}|{fighter_b}"
        _report_progress(
            f"Cycle active: building predictions {idx}/{total_consensus_fights} for {fighter_a} vs {fighter_b}"
        )
        can_trade, start_reason, _ = _live_fight_is_tradeable(fight.get("commence_time"))
        if not can_trade:
            _log_live_fight_skip_once(fight, start_reason)
            prediction_rows_by_key.pop(fight_cache_key, None)
            retained_prediction_keys.discard(fight_cache_key)
            validated_prediction_keys.discard(fight_cache_key)
            _persist_current_prediction_cache(announce=False)
            continue
        try:
            injury = detect_injury_or_cancellation(
                fighter_a, fighter_b,
                current_odds={
                    "a_prob": fight["a_fair_prob_avg"],
                    "b_prob": fight["b_fair_prob_avg"],
                },
            )
            if injury["suspected"]:
                if injury["severity"] == "block" and INJURY_BLOCK_BETS:
                    logger.warning(
                        f"\n  SKIPPING {fighter_a} vs {fighter_b}: {injury['reason']}"
                    )
                    prediction_rows_by_key.pop(fight_cache_key, None)
                    retained_prediction_keys.discard(fight_cache_key)
                    validated_prediction_keys.discard(fight_cache_key)
                    _persist_current_prediction_cache(announce=False)
                    continue
                elif injury["severity"] == "warning":
                    logger.info(
                        f"\n  WARNING for {fighter_a} vs {fighter_b}: {injury['reason']}"
                    )
        except Exception as exc:
            logger.warning("Injury/cancellation check failed for %s vs %s: %s", fighter_a, fighter_b, exc)

        odds_features = {
            "a_implied_prob": fight["a_fair_prob_avg"],
            "b_implied_prob": fight["b_fair_prob_avg"],
            "diff_implied_prob": fight["a_fair_prob_avg"] - fight["b_fair_prob_avg"],
        }

        line_features = {}
        if "line_movement" in getattr(inference_spec, "feature_cols", []):
            try:
                line_features = get_line_movement_features(
                    fighter_a,
                    fighter_b,
                    event_id=fight.get("event_id"),
                    commence_time=fight.get("commence_time"),
                )
                odds_features.update(line_features)
            except Exception as exc:
                logger.warning("Line movement feature extraction failed for %s vs %s: %s", fighter_a, fighter_b, exc)

        cached_row = existing_cache.get(fight_cache_key)
        if cached_row is not None:
            needs_refresh, refresh_reason = _prediction_needs_refresh(
                cached_row,
                fight,
                runtime_signature=runtime_signature,
            )
            if not needs_refresh:
                reused_row = dict(cached_row)
                reused_row["cache_key"] = fight_cache_key
                reused_row["pair_key"] = _live_fight_pair_key(fighter_a, fighter_b)
                reused_row["event_id"] = str(fight.get("event_id", "") or "")
                reused_row["event_date"] = fight.get("commence_time")
                reused_row["a_market_prob"] = fight["a_fair_prob_avg"]
                reused_row["b_market_prob"] = fight["b_fair_prob_avg"]
                reused_row["odds_snapshot"] = _prediction_odds_snapshot(fight)
                reused_row.pop("line_movement", None)
                reused_row.pop("line_is_sharp", None)
                reused_row.pop("line_steam_move", None)
                if line_features:
                    reused_row["line_movement"] = line_features.get("line_movement")
                    reused_row["line_is_sharp"] = line_features.get("line_is_sharp")
                    reused_row["line_steam_move"] = line_features.get("line_steam_move")

                prediction_rows_by_key[fight_cache_key] = _sanitize_prediction_cache_value(reused_row)
                retained_prediction_keys.add(fight_cache_key)
                validated_prediction_keys.add(fight_cache_key)
                _operator_features_by_fight[fight_key] = dict(reused_row.get("operator_features") or {})
                _operator_provenance_by_fight[fight_key] = dict(reused_row.get("operator_provenance") or {})
                logger.info("Reusing cached prediction for %s vs %s", fighter_a, fighter_b)
                _persist_current_prediction_cache(announce=False)
                continue

            logger.info(
                "Refreshing cached prediction for %s vs %s: %s",
                fighter_a,
                fighter_b,
                refresh_reason,
            )

        event_context = _resolve_live_event_context(
            fight,
            _ensure_live_event_contexts(),
            allow_off_card_history_fallback=False,
        )
        if event_context is None:
            _log_live_fight_skip_once(
                fight,
                _missing_live_event_context_reason(fighter_a, fighter_b),
            )
            prediction_rows_by_key.pop(fight_cache_key, None)
            retained_prediction_keys.discard(fight_cache_key)
            validated_prediction_keys.discard(fight_cache_key)
            _persist_current_prediction_cache(announce=False)
            continue

        feature_payload = build_fight_features(
            fighter_a,
            fighter_b,
            odds_features=odds_features,
            weight_class=event_context["weight_class"],
            is_title_bout=event_context["is_title_bout"],
            is_empty_arena=event_context.get("is_empty_arena"),
            num_rounds=event_context["num_rounds"],
            event_id=fight.get("event_id"),
            commence_time=fight.get("commence_time"),
            prefer_live_refresh=True,
            training_spec=inference_spec,
            processed_data_dir=runtime_processed_data_dir,
            include_provenance=True,
        )
        if isinstance(feature_payload, tuple) and len(feature_payload) == 2:
            features, lookup_provenance = feature_payload
        else:
            features = feature_payload
            lookup_provenance = {}
        logger.info(f"  Built {sum(1 for v in features.values() if v is not None)} features for {fighter_a} vs {fighter_b}")
        operator_provenance = {
            **(runtime_bundle_summary or {}),
            **lookup_provenance,
        }
        _operator_features_by_fight[fight_key] = dict(features)
        _operator_provenance_by_fight[fight_key] = dict(operator_provenance)
        a_fights, b_fights = _resolve_live_fight_counts(features, fighter_a, fighter_b)
        low_experience = a_fights < MIN_FIGHTER_FIGHTS or b_fights < MIN_FIGHTER_FIGHTS
        if low_experience:
            low_exp = []
            if a_fights < MIN_FIGHTER_FIGHTS:
                low_exp.append(f"{fighter_a} ({a_fights} fights)")
            if b_fights < MIN_FIGHTER_FIGHTS:
                low_exp.append(f"{fighter_b} ({b_fights} fights)")
            logger.info(
                f"\n  Low experience: {', '.join(low_exp)} - "
                f"prediction generated but trading filters may skip"
            )

        try:
            pred = predict_fight(features, model_result=model_result)
        except Exception as e:
            logger.warning(f"Prediction failed for {fighter_a} vs {fighter_b}: {e}")
            prediction_rows_by_key.pop(fight_cache_key, None)
            retained_prediction_keys.discard(fight_cache_key)
            validated_prediction_keys.discard(fight_cache_key)
            _persist_current_prediction_cache(announce=False)
            continue

        # No-odds model prediction for Trader C conviction checks
        no_odds_a = no_odds_b = None
        if no_odds_result:
            try:
                no_odds_pred = predict_fight(features, model_result=no_odds_result)
                no_odds_a = no_odds_pred["prob_a"]
                no_odds_b = no_odds_pred["prob_b"]
            except Exception as e:
                logger.warning(
                    f"No-odds prediction failed for {fighter_a} vs {fighter_b}: {e} "
                    f"— model agreement filter will block this fight"
                )

        no_odds_str = ""
        if no_odds_a is not None:
            no_odds_str = (
                f"\n    No-odds:    {fighter_a} {no_odds_a:.1%} | "
                f"{fighter_b} {no_odds_b:.1%}"
            )

        logger.info(
            f"\n  {fighter_a} vs {fighter_b}:"
            f"\n    Bookmakers: {fighter_a} {fight['a_fair_prob_avg']:.1%} | "
            f"{fighter_b} {fight['b_fair_prob_avg']:.1%}"
            f"\n    Model:      {fighter_a} {pred['prob_a']:.1%} | "
            f"{fighter_b} {pred['prob_b']:.1%}"
            f"{no_odds_str}"
        )

        # --- Compute SHAP values for this fight ---
        # Reconstruct the same feature vector that predict_fight() uses,
        # including _missing indicator columns.
        fight_shap_values = []
        shap_explainer, shap_base_value = _ensure_shap_state()
        if shap_explainer is not None:
            try:
                import numpy as np

                _base_cols = [c for c in _feat_cols if not c.endswith("_missing")]
                _missing_cols = [c for c in _feat_cols if c.endswith("_missing")]

                base_values = [features.get(col, np.nan) for col in _base_cols]
                X_base = np.array([base_values])

                # Generate missing indicators (1 if NaN, 0 otherwise)
                indicators = [float(np.isnan(X_base[0, i]))
                              for i, col in enumerate(_base_cols)
                              if f"{col}_missing" in _missing_cols]

                # Fill NaNs with training medians
                for i in range(X_base.shape[1]):
                    if np.isnan(X_base[0, i]):
                        X_base[0, i] = _col_medians[i] if i < len(_col_medians) and not np.isnan(_col_medians[i]) else 0.0

                # Combine base + indicators (same as predict_fight)
                if indicators:
                    X = np.column_stack([X_base, np.array([indicators])])
                else:
                    X = X_base

                sv = shap_explainer.shap_values(X)
                # Handle both list (binary) and array output
                if isinstance(sv, list):
                    shap_arr = sv[1][0]  # class 1 = fighter A wins
                else:
                    shap_arr = sv[0]
                # Top 15 by absolute magnitude
                pairs = sorted(
                    zip(_feat_cols, shap_arr.tolist()),
                    key=lambda x: abs(x[1]), reverse=True,
                )[:15]
                fight_shap_values = [
                    {"feature": f, "display_name": _feature_display_name(f), "value": round(v, 4)}
                    for f, v in pairs
                ]
            except Exception as e:
                logger.debug(f"SHAP failed for {fighter_a} vs {fighter_b}: {e}")

        # --- Build feature highlights from top globally-important diff_ features ---
        fight_highlights = []
        for feat_name, importance in _global_importance[:25]:
            if len(fight_highlights) >= 8:
                break
            # Only show differential features (a vs b comparisons)
            if not feat_name.startswith("diff_"):
                continue
            val = features.get(feat_name)
            if val is None:
                continue
            suffix = feat_name[5:]  # strip "diff_"
            a_val = features.get(f"a_{suffix}")
            b_val = features.get(f"b_{suffix}")
            # Skip if both individual values are 0/null (no data)
            if (a_val in (None, 0, 0.0)) and (b_val in (None, 0, 0.0)):
                continue
            favors = None
            if isinstance(val, (int, float)) and val != 0:
                favors = "a" if val > 0 else "b"
            fight_highlights.append({
                "feature": feat_name,
                "display_name": _feature_display_name(feat_name),
                "value": round(float(val), 4) if isinstance(val, (int, float)) else val,
                "a_value": round(float(a_val), 4) if isinstance(a_val, (int, float)) else a_val,
                "b_value": round(float(b_val), 4) if isinstance(b_val, (int, float)) else b_val,
                "favors": favors,
            })

        row_data = {
            "schema_version": PREDICTION_CACHE_SCHEMA_VERSION,
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "prob_a": pred["prob_a"],
            "prob_b": pred["prob_b"],
            "confidence": pred["confidence"],
            "event_id": str(fight.get("event_id", "") or ""),
            "event_date": fight.get("commence_time"),
            "a_market_prob": fight["a_fair_prob_avg"],
            "b_market_prob": fight["b_fair_prob_avg"],
            "no_odds_prob_a": no_odds_a,
            "no_odds_prob_b": no_odds_b,
            "a_num_fights": a_fights,
            "b_num_fights": b_fights,
            "shap_values": fight_shap_values,
            "shap_base_value": shap_base_value,
            "feature_highlights": fight_highlights,
            "low_experience": low_experience,
            "method_stats": {
                k: _serialize_live_context_number(v)
                for k in [
                    "a_ko_rate", "b_ko_rate", "a_sub_rate", "b_sub_rate",
                    "a_dec_rate", "b_dec_rate", "a_roll_slpm", "b_roll_slpm",
                    "a_roll_kd", "b_roll_kd", "a_roll_sub_avg", "b_roll_sub_avg",
                    "a_roll_td_avg", "b_roll_td_avg", "a_total_rounds", "b_total_rounds",
                    "a_roll_str_def", "b_roll_str_def",
                    "a_roll_td_def", "b_roll_td_def",
                    "a_roll_sapm", "b_roll_sapm",
                ]
                for v in [features.get(k)]
            },
            "fighter_context": {
                k: _serialize_live_context_number(
                    v,
                    as_int=k not in ("a_win_pct", "b_win_pct", "a_days_since_last_fight", "b_days_since_last_fight"),
                )
                for k in [
                    "a_wins", "b_wins", "a_losses", "b_losses", "a_draws", "b_draws",
                    "a_win_pct", "b_win_pct",
                    "a_current_win_streak", "b_current_win_streak",
                    "a_lose_streak", "b_lose_streak",
                    "a_days_since_last_fight", "b_days_since_last_fight",
                    "a_cage_rust", "b_cage_rust",
                ]
                for v in [features.get(k)]
            },
            "pair_key": _live_fight_pair_key(fighter_a, fighter_b),
            "cache_key": fight_cache_key,
            "prediction_generated_at": datetime.now(timezone.utc).isoformat(),
            "odds_snapshot": _prediction_odds_snapshot(fight),
            "event_context_snapshot": _prediction_event_context_snapshot(fight, event_context),
            "runtime_signature": runtime_signature,
            "operator_features": _sanitize_prediction_cache_value(features),
            "operator_provenance": _sanitize_prediction_cache_value(operator_provenance),
        }
        # Include line movement metadata for bet filtering
        if line_features:
            row_data["line_movement"] = line_features.get("line_movement")
            row_data["line_is_sharp"] = line_features.get("line_is_sharp")
            row_data["line_steam_move"] = line_features.get("line_steam_move")
        prediction_rows_by_key[fight_cache_key] = _sanitize_prediction_cache_value(row_data)
        retained_prediction_keys.add(fight_cache_key)
        validated_prediction_keys.add(fight_cache_key)
        _persist_current_prediction_cache(announce=False)

    # Finalize the dashboard payload after the full pass completes, even when empty.
    prediction_rows = _prediction_cache_rows_for_write(
        prediction_rows_by_key,
        allowed_keys=validated_prediction_keys,
    )
    _persist_prediction_cache(prediction_rows, announce=True)
    predictions = pd.DataFrame(prediction_rows)

    has_ufc_portfolio = not predictions.empty and not markets.empty
    if not has_ufc_portfolio:
        logger.info("No live UFC opportunities are executable this cycle.")
        _report_progress("Cycle active: no executable UFC opportunities found")
        return {"status": "idle", "reason": "no_executable_opportunities"}

    # Derive event identifier for LLM Operator exposure check
    _operator_event_title = ""
    if not consensus.empty:
        _first_commence = consensus.iloc[0].get("commence_time", "")
        if _first_commence:
            _operator_event_title = str(_first_commence)[:10]  # YYYY-MM-DD date

    # Collect existing open bets for correlated exposure check
    _operator_existing_bets: list[dict] = []
    try:
        from src.polymarket.tracker import BetLedger
        from src.strategy.duo_trader import SINGLE_LEDGER, CONVICTION_LEDGER

        for _ledger_path in [SINGLE_LEDGER, CONVICTION_LEDGER]:
            if _ledger_path.exists():
                _ledger = BetLedger(path=_ledger_path)
                _open = getattr(_ledger, "get_open_bets", None)
                if callable(_open):
                    _operator_existing_bets.extend(_open())
                elif hasattr(_ledger, "bets"):
                    _operator_existing_bets.extend(dict(b) for b in _ledger.bets)
    except Exception as _exc:
        logger.debug("Could not load existing bets for operator exposure check: %s", _exc)

    ufc_results = {"total_orders": 0}
    if has_ufc_portfolio:
        _report_progress("Cycle active: running duo traders and operator checks")
        ufc_results = run_duo_traders(
            predictions=predictions,
            markets=markets,
            clob=clob,
            dry_run=dry_run,
            min_edge=args.min_edge,
            features_by_fight=_operator_features_by_fight,
            provenance_by_fight=_operator_provenance_by_fight,
            event_title=_operator_event_title,
            existing_bets=_operator_existing_bets,
            progress_callback=_report_progress,
        )
    else:
        logger.info("Skipping UFC duo traders this cycle.")

    total_orders = int(ufc_results.get("total_orders", 0))
    logger.info(
        "\nDuo trader run complete. UFC orders: %s",
        total_orders,
    )
    return {"status": "ok", "total_orders": total_orders}


def main():
    parser = argparse.ArgumentParser(
        description="UFC Betting Bot — Predict fights and bet on Polymarket"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape UFC data")
    scrape_parser.add_argument("--fighters-only", action="store_true")
    scrape_parser.add_argument("--fights-only", action="store_true")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train prediction models")
    train_parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    train_parser.add_argument("--spec", type=str, default=None, help="Named training spec to resolve")
    train_parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help="Write processed/test/model artifacts under a subdirectory instead of the canonical promoted paths",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    eval_parser.add_argument("--models", type=str, default="xgboost,logistic")

    # Backtest command (defaults to walk-forward)
    bt_parser = subparsers.add_parser("backtest",
                                       help="Run strategy backtest (walk-forward by default)")
    bt_parser.add_argument("--static", action="store_true",
                           help="Use static single train/test split instead of walk-forward")
    bt_parser.add_argument("--model", type=str, default="xgboost")
    bt_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Explicit model artifact path for static backtests while keeping --model as the logical label.",
    )
    bt_parser.add_argument("--bankroll", type=float, default=INITIAL_BANKROLL)
    bt_parser.add_argument("--min-edge", type=float, default=MIN_EDGE_THRESHOLD)
    bt_parser.add_argument("--kelly", type=float, default=KELLY_FRACTION)
    bt_parser.add_argument(
        "--execution-mode",
        type=str,
        default="legacy",
        choices=["legacy", "realistic"],
        help="Backtest execution assumptions (default: legacy).",
    )
    bt_parser.add_argument(
        "--test-set-path",
        type=str,
        default=None,
        help="Override the static test-set CSV path instead of data/processed/test_set.csv.",
    )
    bt_parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Allow static backtests to proceed even if model/test-set metadata do not match.",
    )
    bt_parser.add_argument("--retrain-months", type=int, default=6,
                           help="Months between model retraining (default: 6)")
    bt_parser.add_argument("--initial-years", type=int, default=5,
                           help="Years of initial training data (default: 5)")

    # Sensitivity command
    sens_parser = subparsers.add_parser("sensitivity", help="Run sensitivity analysis")
    sens_parser.add_argument("--model", type=str, default="xgboost")

    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Predict upcoming fights")
    pred_parser.add_argument("--model", type=str, default="xgboost")

    ufc_refresh_parser = subparsers.add_parser(
        "ufc-refresh-scheduled",
        help="Refresh active-roster UFC raw data, rebuild processed artifacts, and audit profile completeness",
    )
    ufc_refresh_parser.add_argument(
        "--dataset-variant",
        type=str,
        default="pulled_all_plus_legacy_market",
    )
    ufc_refresh_parser.add_argument(
        "--output-subdir",
        action="append",
        default=None,
        help="Processed output subdir(s) to rebuild. Defaults to base plus promoted candidate dirs.",
    )
    ufc_refresh_parser.add_argument("--limit-fighters", type=int, default=None)
    ufc_refresh_parser.add_argument("--skip-rebuild", action="store_true")
    ufc_refresh_parser.add_argument("--skip-audit", action="store_true")
    ufc_refresh_parser.add_argument(
        "--audit-json-path",
        type=Path,
        default=Path("tmp") / "active_roster_profile_completeness_scheduled_latest.json",
    )
    ufc_refresh_parser.add_argument(
        "--audit-csv-path",
        type=Path,
        default=Path("tmp") / "active_roster_profile_completeness_scheduled_latest.csv",
    )

    # Live command
    live_parser = subparsers.add_parser("live", help="Run duo-trader live bot (S+C)")
    live_parser.add_argument("--dry-run", action="store_true", default=True,
                             help="Dry run mode (default: True)")
    live_parser.add_argument("--real", action="store_true",
                             help="Run with real money (requires explicit arming env vars)")
    live_parser.add_argument("--model", type=str, default="xgboost")
    live_parser.add_argument("--min-edge", type=float, default=MIN_EDGE_THRESHOLD)

    # Monitor command
    mon_parser = subparsers.add_parser("monitor", help="Continuous event monitoring")
    mon_parser.add_argument("--interval", type=float, default=6.0,
                            help="Hours between checks (default: 6)")

    # Backtest compare command
    btc_parser = subparsers.add_parser("backtest-compare",
                                        help="Compare full model vs no-odds baseline")
    btc_parser.add_argument("--bankroll", type=float, default=INITIAL_BANKROLL)
    btc_parser.add_argument("--min-edge", type=float, default=MIN_EDGE_THRESHOLD)
    btc_parser.add_argument("--kelly", type=float, default=KELLY_FRACTION)
    btc_parser.add_argument("--walkforward", action="store_true",
                            help="Run walk-forward strategy comparison")
    btc_parser.add_argument("--retrain-months", type=int, default=6,
                            help="Months between model retraining (default: 6)")
    btc_parser.add_argument("--initial-years", type=int, default=5,
                            help="Years of initial training data (default: 5)")

    # Backfill odds command
    bf_parser = subparsers.add_parser("backfill-odds",
                                      help="Backfill historical odds from The Odds API")
    bf_parser.add_argument("--offsets", type=str, default="7,3,1",
                           help="Comma-separated day offsets (default: 7,3,1)")
    bf_parser.add_argument("--fresh", action="store_true",
                           help="Start fresh (ignore existing backfill data)")

    # Walk-forward backtest command
    wf_parser = subparsers.add_parser("walkforward",
                                       help="Walk-forward backtest with periodic retraining")
    wf_parser.add_argument("--retrain-months", type=int, default=6,
                            help="Months between model retraining (default: 6)")
    wf_parser.add_argument("--initial-years", type=int, default=5,
                            help="Years of initial training data (default: 5)")
    wf_parser.add_argument("--bankroll", type=float, default=INITIAL_BANKROLL)
    wf_parser.add_argument("--min-edge", type=float, default=MIN_EDGE_THRESHOLD)
    wf_parser.add_argument("--kelly", type=float, default=KELLY_FRACTION)
    wf_parser.add_argument(
        "--execution-mode",
        type=str,
        default="legacy",
        choices=["legacy", "realistic"],
        help="Backtest execution assumptions (default: legacy).",
    )

    # Positions command
    subparsers.add_parser("positions", help="Show current Polymarket positions and P&L")

    # Web dashboard command
    web_parser = subparsers.add_parser("web",
                                        help="Launch web dashboard (local)")
    web_parser.add_argument("--port", type=int, default=5050,
                             help="Port to run on (default: 5050)")
    web_parser.add_argument("--offline", action="store_true",
                             help="Don't connect to Polymarket for live prices")
    web_parser.add_argument("--debug", action="store_true",
                             help="Run Flask in debug mode")

    # Dashboard command (terminal)
    dash_parser = subparsers.add_parser("dashboard",
                                         help="Terminal-based live dashboard")
    dash_parser.add_argument("--refresh", type=int, default=30,
                              help="Refresh interval in seconds (default: 30)")
    dash_parser.add_argument("--offline", action="store_true",
                              help="Don't fetch live prices from Polymarket")
    dash_parser.add_argument("--real-only", action="store_true",
                              help="Only show real bets (exclude dry runs)")

    # Settle command
    settle_parser = subparsers.add_parser("settle",
                                           help="Settle bets (manual or auto)")
    settle_parser.add_argument("--auto", action="store_true",
                                help="Auto-settle from Polymarket resolved markets")
    settle_parser.add_argument("--bet-id", type=int,
                                help="Bet ID to settle")
    settle_parser.add_argument("--result", type=str,
                                help="Result: win or loss")

    redeem_parser = subparsers.add_parser(
        "redeem",
        help="Redeem resolved Polymarket positions that are ready to claim",
    )
    redeem_parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit redeem transaction(s) without waiting for relayer mining",
    )

    # Track lines command
    subparsers.add_parser("track-lines", help="Snapshot odds and analyze movement")

    # Signals command
    subparsers.add_parser("signals", help="Check pre-fight signals for upcoming events")

    args = parser.parse_args()

    if args.command == "live" and args.real:
        args.dry_run = False

    commands = {
        "scrape": cmd_scrape,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "backtest": cmd_backtest,
        "backtest-compare": cmd_backtest_compare,
        "backfill-odds": cmd_backfill_odds,
        "sensitivity": cmd_sensitivity,
        "walkforward": cmd_walkforward,
        "predict": cmd_predict,
        "ufc-refresh-scheduled": cmd_ufc_refresh_scheduled,
        "live": cmd_duo_live,
        "positions": cmd_positions,
        "web": cmd_web,
        "dashboard": cmd_dashboard,
        "settle": cmd_settle,
        "redeem": cmd_redeem,
        "monitor": cmd_monitor,
        "track-lines": cmd_track_lines,
        "signals": cmd_signals,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
