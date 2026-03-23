"""
Live fighter stats lookup — scrapes UFCStats.com for current fighter data
and builds feature vectors compatible with the trained model.

Used during live predictions to populate the full feature set (rolling stats,
physical attributes, etc.) instead of relying on median imputation.
"""

import logging
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import (
    UFCSTATS_FIGHTER_URL,
    UFCSTATS_FIGHTER_SEARCH_URL,
    ROLLING_WINDOW,
    EWM_HALFLIFE,
    ELO_INITIAL,
    PROCESSED_DATA_DIR,
)
from src.data.name_utils import (
    normalize_cross_source_name,
    normalize_person_name,
    same_person_name,
)
from src.features.stance_utils import encode_stance

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
REQUEST_DELAY = 1.0
LOOKUP_CACHE_TTL_SECONDS = 3600.0
LIVE_PROCESSED_REFRESH_MAX_AGE_DAYS = 7

# Cache to avoid re-scraping during a single session
_fighter_cache: dict[str, dict] = {}
_fighter_cache_cached_at: dict[str, float] = {}
_fighter_url_cache: dict[str, str] = {}
_fighter_url_cache_cached_at: dict[str, float] = {}
_processed_feature_history_cache: dict[str, pd.DataFrame] = {}
_processed_feature_history_mtime: dict[str, float] = {}
_elo_state_cache: dict[str, dict[str, Any]] = {}
_elo_state_cache_mtime: dict[str, float] = {}
_pre_ufc_summary_cache: dict[str, dict] | None = None


def _load_pre_ufc_summary_cache() -> dict[str, dict]:
    """Load and cache per-fighter pre-UFC career summaries from the supplement CSV.

    Returns dict mapping fighter name -> {pre_ufc_total_fights, pre_ufc_wins, ...}.
    Uses module-level cache so the CSV is only parsed once per session.
    """
    global _pre_ufc_summary_cache
    if _pre_ufc_summary_cache is not None:
        return _pre_ufc_summary_cache

    from src.features.build_features import (
        _compute_pre_ufc_summary,
        _load_pre_ufc_supplement,
        _resolve_pre_ufc_supplement_path,
    )

    _pre_ufc_summary_cache = {}
    pre_ufc_path = _resolve_pre_ufc_supplement_path()
    if not pre_ufc_path.exists():
        return _pre_ufc_summary_cache

    supplement = _load_pre_ufc_supplement(pre_ufc_path)
    if supplement.empty:
        return _pre_ufc_summary_cache

    summary_df = _compute_pre_ufc_summary(supplement)
    for _, row in summary_df.iterrows():
        fighter = row["fighter"]
        _pre_ufc_summary_cache[fighter] = {
            col: row[col] for col in summary_df.columns if col != "fighter"
        }
    logger.info(f"Loaded pre-UFC career summaries for {len(_pre_ufc_summary_cache)} fighters")
    return _pre_ufc_summary_cache


# Tracks fighters we already attempted on-demand scraping for this session
# so we don't re-scrape on every prediction call.
_pre_ufc_attempted: set[str] = set()


def _on_demand_pre_ufc_scrape(
    fighter_name: str,
    lookup_data: Optional[dict],
    summary_cache: dict[str, dict],
) -> None:
    """Scrape a fighter's pre-UFC career on-demand and update the cache + CSV.

    Called at live-prediction time when a fighter is missing from the
    pre-UFC supplement.  Tries both Sherdog and Tapology, keeps whichever
    has the most complete data.
    """
    if fighter_name in _pre_ufc_attempted:
        return
    _pre_ufc_attempted.add(fighter_name)

    # Derive first_ufc_date from the fighter's UFC fight history
    first_ufc_date = None
    if lookup_data and lookup_data.get("fights"):
        dates = []
        for f in lookup_data["fights"]:
            d = f.get("event_date")
            if d:
                dates.append(str(d))
        if dates:
            first_ufc_date = min(dates)

    logger.info(
        "On-demand pre-UFC scrape for '%s' (first UFC date: %s)",
        fighter_name, first_ufc_date or "unknown/debut",
    )

    try:
        from src.data.pre_ufc_scraper import (
            append_to_supplement,
            compute_single_fighter_summary,
            scrape_fighter_pre_ufc_fights,
        )

        rows = scrape_fighter_pre_ufc_fights(fighter_name, first_ufc_date)
        if not rows:
            logger.info("  No pre-UFC fights found for '%s'", fighter_name)
            return

        summary = compute_single_fighter_summary(rows)
        if summary:
            summary_cache[fighter_name] = summary
            logger.info(
                "  Cached pre-UFC summary for '%s': %d fights, %.0f%% win rate",
                fighter_name,
                summary.get("pre_ufc_total_fights", 0),
                (summary.get("pre_ufc_win_pct", 0) or 0) * 100,
            )

        # Persist to CSV so future sessions don't re-scrape
        from src.features.build_features import _resolve_pre_ufc_supplement_path
        supplement_path = _resolve_pre_ufc_supplement_path()
        append_to_supplement(rows, supplement_path)
        logger.info("  Persisted %d pre-UFC fight rows to %s", len(rows), supplement_path.name)

    except Exception as e:
        logger.warning("  On-demand pre-UFC scrape failed for '%s': %s", fighter_name, e)


WEIGHT_CLASS_WEIGHT_MAP = {
    "strawweight": 115,
    "women's strawweight": 115,
    "flyweight": 125,
    "women's flyweight": 125,
    "bantamweight": 135,
    "women's bantamweight": 135,
    "featherweight": 145,
    "women's featherweight": 145,
    "lightweight": 155,
    "welterweight": 170,
    "middleweight": 185,
    "light heavyweight": 205,
    "heavyweight": 265,
    "catch weight": None,
}


def _coerce_percentage_feature(value, default: float = 50.0) -> float:
    """Return a percentage-like feature, using the fallback for missing values."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        logger.debug("Coercing non-numeric percentage value %r to default %.1f", value, default)
        return default
    if np.isnan(numeric):
        logger.debug("Coercing NaN percentage to default %.1f", default)
        return default
    return numeric


def _coerce_numeric_feature(value, default: float = 0.0) -> float:
    """Return a numeric feature, using the fallback for missing values."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(numeric) else numeric


def _weight_class_to_weight(value: object) -> Optional[int]:
    """Map a weight-class label to its canonical division weight."""
    text = str(value or "").lower()
    if not text:
        return None
    for label, weight in WEIGHT_CLASS_WEIGHT_MAP.items():
        if label in text:
            return weight
    return None


def _processed_fight_result(
    fighter_name: str,
    opponent_name: str,
    winner_name: object,
) -> str:
    """Return win/loss/neutral for a processed-history row."""
    if same_person_name(winner_name, fighter_name):
        return "win"
    if same_person_name(winner_name, opponent_name):
        return "loss"
    return "neutral"


def _history_fight_result(fight: dict, fighter_name: str) -> str:
    """Return win/loss/draw from a processed or scraped fight-history record."""
    result = str(fight.get("result", "") or "").strip().lower()
    if result in {"win", "loss"}:
        return result
    if result in {"draw", "neutral", "nc", "no contest"}:
        return "draw"

    opponent = fight.get("opponent", "")
    winner = fight.get("winner", "")
    if same_person_name(winner, fighter_name):
        return "win"
    if same_person_name(winner, opponent):
        return "loss"

    if "won" in fight:
        return "win" if int(bool(fight.get("won", 0))) == 1 else "loss"
    return "draw"


def _get_soup(url: str) -> BeautifulSoup:
    """Fetch a URL and return parsed BeautifulSoup."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return BeautifulSoup(resp.text, "lxml")


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _normalized_path_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _features_path(processed_data_dir: Optional[Path] = None) -> Path:
    resolved_dir = Path(processed_data_dir) if processed_data_dir is not None else PROCESSED_DATA_DIR
    return resolved_dir / "features.csv"


def _path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime if path.exists() else 0.0
    except OSError:
        return 0.0


def _cache_is_fresh(
    cache_times: dict[str, float],
    cache_key: str,
    ttl_seconds: float = LOOKUP_CACHE_TTL_SECONDS,
) -> bool:
    cached_at = cache_times.get(cache_key)
    return cached_at is not None and (time.time() - cached_at) <= ttl_seconds


def _coerce_training_spec(training_spec: Any = None):
    if training_spec is None:
        return None

    from src.model.training_spec import NamedModelTrainingSpec, resolve_named_training_spec

    if isinstance(training_spec, NamedModelTrainingSpec):
        return training_spec
    if isinstance(training_spec, dict):
        return NamedModelTrainingSpec(**training_spec)
    if isinstance(training_spec, str):
        return resolve_named_training_spec(training_spec)
    return training_spec


def _resolve_processed_data_dir(
    *,
    training_spec: Any = None,
    processed_data_dir: Optional[Path] = None,
) -> Path:
    if processed_data_dir is not None:
        return Path(processed_data_dir)

    hosted_processed_dir = _hosted_processed_data_dir()
    if hosted_processed_dir is not None:
        return hosted_processed_dir

    spec = _coerce_training_spec(training_spec)
    if spec is not None:
        candidate_dir = PROCESSED_DATA_DIR / "candidates" / spec.name
        if (candidate_dir / "features.csv").exists():
            return candidate_dir

    return PROCESSED_DATA_DIR


def _hosted_processed_data_dir() -> Optional[Path]:
    try:
        from src.model.production_bundle import (
            is_hosted_runtime,
            load_production_bundle_or_none,
        )
    except Exception:
        return None

    if not is_hosted_runtime():
        return None

    bundle = load_production_bundle_or_none()
    if bundle is not None:
        return bundle.processed_dir
    return PROCESSED_DATA_DIR


def _build_lookup_provenance(
    *,
    fighter_a_result: Optional[dict],
    fighter_b_result: Optional[dict],
    processed_data_dir: Path,
    training_spec: Any = None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "fighter_a_source": (
            str(fighter_a_result.get("source")).strip()
            if isinstance(fighter_a_result, dict) and fighter_a_result.get("source") is not None
            else None
        ),
        "fighter_b_source": (
            str(fighter_b_result.get("source")).strip()
            if isinstance(fighter_b_result, dict) and fighter_b_result.get("source") is not None
            else None
        ),
        "processed_dir": str(Path(processed_data_dir)),
    }

    spec = _coerce_training_spec(training_spec)
    if spec is not None and getattr(spec, "name", None):
        provenance["training_spec_name"] = spec.name

    try:
        from src.model.production_bundle import (
            get_processed_snapshot_max_event_date,
            is_hosted_runtime,
            load_production_bundle_or_none,
        )

        provenance["processed_snapshot_max_event_date"] = get_processed_snapshot_max_event_date(
            Path(processed_data_dir)
        )
        if is_hosted_runtime():
            bundle = load_production_bundle_or_none()
            if bundle is not None and bundle.processed_dir.resolve(strict=False) == Path(
                processed_data_dir
            ).resolve(strict=False):
                provenance["bundle_id"] = bundle.bundle_id
                provenance["manifest_path"] = str(bundle.manifest_path)
                provenance["model_spec_name"] = bundle.model_spec_name
    except Exception:
        provenance.setdefault("processed_snapshot_max_event_date", None)

    return provenance


def _requested_feature_columns(training_spec: Any = None) -> Optional[list[str]]:
    spec = _coerce_training_spec(training_spec)
    if spec is None:
        return None
    return list(getattr(spec, "feature_cols", []) or [])


def _wants_feature(requested_feature_set: Optional[set[str]], *columns: str) -> bool:
    return requested_feature_set is None or any(column in requested_feature_set for column in columns)


_PROFILE_FIELD_REQUEST_COLUMNS: dict[str, tuple[str, ...]] = {
    "height": ("a_height", "b_height", "diff_height"),
    "reach": ("a_reach", "b_reach", "diff_reach"),
    "weight": ("a_weight", "b_weight", "diff_weight"),
    "age": ("a_age", "b_age", "diff_age"),
    "stance": ("a_stance_enc", "b_stance_enc", "same_stance"),
}


def _requested_profile_fields(training_spec: Any = None) -> Optional[set[str]]:
    """Return the profile snapshot fields the requested spec explicitly needs."""
    requested_feature_cols = _requested_feature_columns(training_spec)
    if requested_feature_cols is None:
        return None

    requested_feature_set = set(requested_feature_cols)
    return {
        field_name
        for field_name, columns in _PROFILE_FIELD_REQUEST_COLUMNS.items()
        if _wants_feature(requested_feature_set, *columns)
    }


def _is_provenance_strict_spec(training_spec: Any = None) -> bool:
    """
    Return whether live lookup should use the pulled-history strict path.

    `pulled_all` candidates train their history-backed fields directly from
    pulled UFCStats fight rows, so live lookup must mirror that strict
    computation even if the spec selectively re-enables some stable profile
    snapshot fields such as height or age.
    """
    spec = _coerce_training_spec(training_spec)
    if spec is None:
        return False
    variant = getattr(spec, "dataset_variant", None) or ""
    return variant.startswith("pulled_all")


def _fighter_cache_key(
    fighter_name: str,
    as_of_date: Optional[str] = None,
    *,
    reference_date: Any = None,
    processed_data_dir: Optional[Path] = None,
) -> str:
    resolved_dir = Path(processed_data_dir) if processed_data_dir is not None else PROCESSED_DATA_DIR
    reference_token = ""
    if reference_date is not None and str(reference_date).strip():
        reference_token = _reference_date_cache_token(reference_date)
    features_token = f"{_path_mtime(_features_path(resolved_dir)):.6f}"
    return (
        f"{normalize_person_name(fighter_name)}::"
        f"{str(as_of_date or '')}::"
        f"{reference_token}::"
        f"{features_token}::"
        f"{_normalized_path_key(resolved_dir)}"
    )


def _reference_date_cache_token(reference_date: Any = None) -> str:
    if reference_date is None or not str(reference_date).strip():
        return ""
    return _coerce_reference_timestamp(reference_date).isoformat()


def _coerce_reference_timestamp(reference_date: Any = None) -> pd.Timestamp:
    """Return a timezone-naive UTC timestamp for feature-age calculations."""
    if reference_date is None or not str(reference_date).strip():
        return pd.Timestamp.now(tz="UTC").tz_localize(None)

    ts = pd.Timestamp(reference_date)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _load_processed_feature_history(*, processed_data_dir: Optional[Path] = None) -> pd.DataFrame:
    resolved_dir = Path(processed_data_dir) if processed_data_dir is not None else PROCESSED_DATA_DIR
    cache_key = _normalized_path_key(resolved_dir)

    features_path = _features_path(resolved_dir)

    # Invalidate cache if the underlying file has been modified
    cached = _processed_feature_history_cache.get(cache_key)
    if cached is not None:
        current_mtime = _path_mtime(features_path)
        cached_mtime = _processed_feature_history_mtime.get(cache_key, 0.0)
        if current_mtime == cached_mtime:
            return cached
        # File changed — drop stale cache entry
        _processed_feature_history_cache.pop(cache_key, None)
        _processed_feature_history_mtime.pop(cache_key, None)

    def _store(df: pd.DataFrame) -> pd.DataFrame:
        _processed_feature_history_cache[cache_key] = df
        _processed_feature_history_mtime[cache_key] = _path_mtime(features_path)
        return df

    if not features_path.exists():
        return _store(pd.DataFrame())

    try:
        history = pd.read_csv(features_path, parse_dates=["event_date"])
    except Exception as exc:
        logger.warning("Failed to read processed features for fighter lookup from %s: %s", features_path, exc)
        return _store(pd.DataFrame())

    if "event_date" in history.columns:
        history["event_date"] = pd.to_datetime(history["event_date"], errors="coerce")
        history = history.sort_values("event_date")
    return _store(history)


def _empty_elo_state() -> dict[str, Any]:
    return {
        "ratings": {},
        "fight_counts": {},
        "history": {},
        "opponent_history": {},
        "history_dates": {},
        "opponent_history_dates": {},
    }


def _load_elo_state(*, processed_data_dir: Optional[Path] = None) -> dict[str, Any]:
    """
    Load precomputed UFC Elo snapshots from the processed features file.

    This is kept for backward compatibility with older helper APIs/tests.
    If the current features artifact no longer carries Elo columns, the
    function returns an empty/default state instead of failing.
    """
    resolved_dir = Path(processed_data_dir) if processed_data_dir is not None else PROCESSED_DATA_DIR
    cache_key = _normalized_path_key(resolved_dir)
    features_path = _features_path(resolved_dir)
    current_mtime = _path_mtime(features_path)
    cached = _elo_state_cache.get(cache_key)
    cached_mtime = _elo_state_cache_mtime.get(cache_key, 0.0)
    if cached is not None and cached_mtime == current_mtime:
        return cached
    if cached is not None:
        _elo_state_cache.pop(cache_key, None)
        _elo_state_cache_mtime.pop(cache_key, None)

    state = _empty_elo_state()
    if not features_path.exists():
        _elo_state_cache[cache_key] = state
        _elo_state_cache_mtime[cache_key] = current_mtime
        return state

    try:
        df = pd.read_csv(
            features_path,
            usecols=[
                "fighter_a",
                "fighter_b",
                "a_elo",
                "b_elo",
                "a_num_fights",
                "b_num_fights",
                "event_date",
            ],
            parse_dates=["event_date"],
        )
    except (ValueError, KeyError):
        _elo_state_cache[cache_key] = state
        _elo_state_cache_mtime[cache_key] = current_mtime
        return state

    ratings = state["ratings"]
    fight_counts = state["fight_counts"]
    history = state["history"]
    opponent_history = state["opponent_history"]
    history_dates = state["history_dates"]
    opponent_history_dates = state["opponent_history_dates"]

    df = df.sort_values("event_date")
    for _, row in df.iterrows():
        fighter_a = row.get("fighter_a", "")
        fighter_b = row.get("fighter_b", "")
        a_elo = row.get("a_elo", ELO_INITIAL)
        b_elo = row.get("b_elo", ELO_INITIAL)
        event_date = pd.to_datetime(row.get("event_date"), errors="coerce")

        if fighter_a:
            ratings[fighter_a] = a_elo
            fight_counts[fighter_a] = int(row.get("a_num_fights", 0))
            if not pd.isna(a_elo):
                history.setdefault(fighter_a, []).append(float(a_elo))
                if pd.notna(event_date):
                    history_dates.setdefault(fighter_a, []).append((event_date, float(a_elo)))
            if fighter_b and not pd.isna(b_elo):
                opponent_history.setdefault(fighter_a, []).append(float(b_elo))
                if pd.notna(event_date):
                    opponent_history_dates.setdefault(fighter_a, []).append((event_date, float(b_elo)))

        if fighter_b:
            ratings[fighter_b] = b_elo
            fight_counts[fighter_b] = int(row.get("b_num_fights", 0))
            if not pd.isna(b_elo):
                history.setdefault(fighter_b, []).append(float(b_elo))
                if pd.notna(event_date):
                    history_dates.setdefault(fighter_b, []).append((event_date, float(b_elo)))
            if fighter_a and not pd.isna(a_elo):
                opponent_history.setdefault(fighter_b, []).append(float(a_elo))
                if pd.notna(event_date):
                    opponent_history_dates.setdefault(fighter_b, []).append((event_date, float(a_elo)))

    _elo_state_cache[cache_key] = state
    _elo_state_cache_mtime[cache_key] = current_mtime
    return state


def get_fighter_elo(fighter_name: str, *, processed_data_dir: Optional[Path] = None) -> float:
    """Return the latest known Elo for a fighter, or the default baseline."""
    state = _load_elo_state(processed_data_dir=processed_data_dir)
    ratings = state["ratings"]

    if fighter_name in ratings:
        return float(ratings[fighter_name])

    normalized_name = normalize_cross_source_name(fighter_name)
    for name, elo in ratings.items():
        if normalize_cross_source_name(name) == normalized_name:
            return float(elo)

    return float(ELO_INITIAL)


def _resolve_fighter_key(fighter_name: str, lookup: dict) -> Optional[str]:
    """Resolve a fighter name to the canonical key used in a lookup dict."""
    if fighter_name in lookup:
        return fighter_name
    normalized_name = normalize_cross_source_name(fighter_name)
    for key in lookup:
        if normalize_cross_source_name(key) == normalized_name:
            return key
    return None


def _history_values_as_of(
    dated_history: Optional[dict[str, list[tuple[pd.Timestamp, float]]]],
    key: str,
    as_of_date: Optional[str] = None,
) -> list[float]:
    """Return history values filtered strictly before the provided cutoff."""
    if dated_history is None:
        return []

    records = dated_history.get(key, [])
    if as_of_date is None:
        return [value for _, value in records]

    cutoff = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(cutoff):
        return [value for _, value in records]

    return [value for event_date, value in records if pd.notna(event_date) and event_date < cutoff]


def get_fighter_elo_momentum(
    fighter_name: str,
    window: int = 5,
    as_of_date: Optional[str] = None,
    *,
    processed_data_dir: Optional[Path] = None,
) -> float:
    """Return the slope of the fighter's recent Elo history."""
    state = _load_elo_state(processed_data_dir=processed_data_dir)
    history = state["history"]
    history_dates = state["history_dates"]

    lookup = history_dates if as_of_date is not None else history
    key = _resolve_fighter_key(fighter_name, lookup or {})
    if key is None:
        return 0.0

    values_history = history.get(key, [])
    values = (
        _history_values_as_of(history_dates, key, as_of_date)[-window:]
        if as_of_date is not None
        else values_history[-window:]
    )
    if len(values) < 2:
        return 0.0

    x = np.arange(len(values))
    return float(np.polyfit(x, values, 1)[0])


def get_fighter_sos(
    fighter_name: str,
    window: int = 5,
    as_of_date: Optional[str] = None,
    *,
    processed_data_dir: Optional[Path] = None,
) -> float:
    """Return the mean Elo of the fighter's recent opponents."""
    state = _load_elo_state(processed_data_dir=processed_data_dir)
    opponent_history = state["opponent_history"]
    opponent_history_dates = state["opponent_history_dates"]

    lookup = opponent_history_dates if as_of_date is not None else opponent_history
    key = _resolve_fighter_key(fighter_name, lookup or {})
    if key is None:
        return float(ELO_INITIAL)

    opponent_elos = opponent_history.get(key, [])
    recent = (
        _history_values_as_of(opponent_history_dates, key, as_of_date)[-window:]
        if as_of_date is not None
        else opponent_elos[-window:]
    )
    if not recent:
        return float(ELO_INITIAL)
    return float(np.mean(recent))


def _processed_history_latest_event_date(
    *,
    processed_data_dir: Optional[Path] = None,
) -> Optional[pd.Timestamp]:
    history = _load_processed_feature_history(processed_data_dir=processed_data_dir)
    if history.empty or "event_date" not in history.columns:
        return None
    latest = pd.to_datetime(history["event_date"], errors="coerce").max()
    return latest if pd.notna(latest) else None


def _processed_data_is_stale_for_live(
    *,
    reference_date: Any = None,
    processed_data_dir: Optional[Path] = None,
) -> bool:
    if reference_date is None or not str(reference_date).strip():
        return False
    latest_event_date = _processed_history_latest_event_date(
        processed_data_dir=processed_data_dir
    )
    if latest_event_date is None:
        return True
    reference_ts = _coerce_reference_timestamp(reference_date).normalize()
    latest_ts = pd.Timestamp(latest_event_date).normalize()
    freshness_gap_days = max(int((reference_ts - latest_ts).days), 0)
    return freshness_gap_days > LIVE_PROCESSED_REFRESH_MAX_AGE_DAYS


def _extract_prefixed_features(row: pd.Series, prefix: str) -> dict:
    features = {}
    for column, value in row.items():
        if column.startswith(prefix):
            features[column[len(prefix):]] = value
    return features


def _lookup_processed_fighter(
    fighter_name: str,
    *,
    as_of_date: Optional[str] = None,
    reference_date: Any = None,
    processed_data_dir: Optional[Path] = None,
) -> Optional[dict]:
    """Build a fighter snapshot from the local processed feature history."""
    history = _load_processed_feature_history(processed_data_dir=processed_data_dir)
    if history.empty:
        return None

    cutoff = pd.to_datetime(as_of_date, errors="coerce") if as_of_date is not None else None
    latest_features = None
    latest_fighter_name = fighter_name
    latest_event_date = None
    snapshot_exact = False
    prior_fights = []

    for row in history.itertuples(index=False):
        event_date = pd.to_datetime(getattr(row, "event_date", None), errors="coerce")
        if same_person_name(fighter_name, getattr(row, "fighter_a", "")):
            prefix = "a_"
            fighter_value = getattr(row, "fighter_a", fighter_name)
            opponent = getattr(row, "fighter_b", "")
        elif same_person_name(fighter_name, getattr(row, "fighter_b", "")):
            prefix = "b_"
            fighter_value = getattr(row, "fighter_b", fighter_name)
            opponent = getattr(row, "fighter_a", "")
        else:
            continue

        result = _processed_fight_result(
            fighter_value,
            opponent,
            getattr(row, "winner", ""),
        )
        if cutoff is None or pd.isna(event_date) or event_date < cutoff:
            prior_fights.append(
                {
                    "opponent": opponent,
                    "won": 1 if result == "win" else 0,
                    "result": result,
                    "winner": getattr(row, "winner", ""),
                    "weight_class": getattr(row, "weight_class", ""),
                    "event_date": event_date,
                }
            )

        include_snapshot = (
            cutoff is None
            or pd.isna(event_date)
            or event_date <= cutoff
        )
        if not include_snapshot:
            continue

        row_series = pd.Series(row._asdict())
        latest_features = _extract_prefixed_features(row_series, prefix)
        latest_fighter_name = fighter_value
        latest_event_date = event_date
        snapshot_exact = cutoff is not None and pd.notna(event_date) and event_date == cutoff

    if latest_features is None:
        return None

    effective_date = (
        cutoff
        if cutoff is not None
        else _coerce_reference_timestamp(reference_date).normalize()
    )
    should_age_forward = pd.notna(latest_event_date) and (
        cutoff is None or not snapshot_exact
    )
    if should_age_forward:
        days_since_last_fight = max(int((effective_date.normalize() - latest_event_date.normalize()).days), 0)
        latest_features["days_since_last_fight"] = float(days_since_last_fight)
        latest_features["layoff_log"] = float(np.log1p(days_since_last_fight))
        latest_features["cage_rust"] = 1 if days_since_last_fight > 365 else 0
        age = pd.to_numeric(pd.Series([latest_features.get("age")]), errors="coerce").iloc[0]
        if not pd.isna(age):
            latest_features["age"] = float(age + (days_since_last_fight / 365.25))

    wins = pd.to_numeric(pd.Series([latest_features.get("wins")]), errors="coerce").iloc[0]
    losses = pd.to_numeric(pd.Series([latest_features.get("losses")]), errors="coerce").iloc[0]
    draws = pd.to_numeric(pd.Series([latest_features.get("draws")]), errors="coerce").iloc[0]
    record = None
    if not any(pd.isna(value) for value in [wins, losses, draws]):
        record = f"{int(wins)}-{int(losses)}-{int(draws)}"

    return {
        "profile": {
            "name": latest_fighter_name,
            "record": record or "?",
        },
        "fights": prior_fights,
        "features": latest_features,
        "source": "processed",
        "snapshot_event_date": latest_event_date,
        "snapshot_exact": snapshot_exact,
    }


def _call_lookup_fighter(
    fighter_name: str,
    *,
    as_of_date: Optional[str] = None,
    reference_date: Any = None,
    prefer_live_refresh: bool = False,
    training_spec: Any = None,
    processed_data_dir: Optional[Path] = None,
) -> Optional[dict]:
    kwargs = {}
    if as_of_date is not None:
        kwargs["as_of_date"] = as_of_date
    if reference_date is not None:
        kwargs["reference_date"] = reference_date
    if prefer_live_refresh:
        kwargs["prefer_live_refresh"] = True
    if training_spec is not None:
        kwargs["training_spec"] = training_spec
    if processed_data_dir is not None:
        kwargs["processed_data_dir"] = processed_data_dir

    while True:
        try:
            return lookup_fighter(fighter_name, **kwargs)
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword argument" not in message:
                raise
            match = re.search(r"unexpected keyword argument '([^']+)'", message)
            if match is None:
                break
            unsupported = match.group(1)
            if unsupported not in kwargs:
                break
            kwargs.pop(unsupported, None)

    return lookup_fighter(fighter_name)




def _safe_float(value, default=np.nan) -> float:
    """Safely convert a value to float."""
    if value is None or value == "" or value == "--":
        return default
    try:
        return float(str(value).replace("%", "").strip())
    except (ValueError, TypeError):
        return default


def _inches_to_cm(value: float) -> float:
    """Convert an imperial measurement in inches to centimeters."""
    if value is None or np.isnan(value):
        return np.nan
    return round(float(value) * 2.54, 2)


def _parse_height_inches(height_str: str) -> float:
    """Parse height like '5\\'10\"' or '5' 10\"' to inches."""
    if not height_str or height_str == "--":
        return np.nan
    match = re.search(r"(\d+)'?\s*(\d+)", height_str)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        return feet * 12 + inches
    return np.nan


def _parse_reach_inches(reach_str: str) -> float:
    """Parse reach like '74\"' or '74' to inches."""
    if not reach_str or reach_str == "--":
        return np.nan
    match = re.search(r"(\d+)", reach_str)
    if match:
        return float(match.group(1))
    return np.nan


def _parse_weight_lbs(weight_str: str) -> float:
    """Parse weight like '185 lbs.' to float."""
    if not weight_str or weight_str == "--":
        return np.nan
    match = re.search(r"(\d+)", weight_str)
    if match:
        return float(match.group(1))
    return np.nan


def _parse_dob_to_age(dob_str: str, *, reference_date: Any = None) -> float:
    """Parse DOB like 'Sep 22, 1989' to age in years at the reference date."""
    if not dob_str or dob_str == "--":
        return np.nan
    reference_ts = _coerce_reference_timestamp(reference_date).normalize()
    for fmt in ["%b %d, %Y", "%B %d, %Y"]:
        try:
            dob = datetime.strptime(dob_str.strip(), fmt)
            age = (reference_ts - pd.Timestamp(dob).normalize()).days / 365.25
            return round(age, 1)
        except ValueError:
            continue
    return np.nan


def _parse_event_date_text(event_text: str) -> Optional[datetime]:
    """Extract and parse the event date embedded in a UFCStats event cell."""
    if not event_text:
        return None

    cleaned = _clean_text(event_text)
    match = re.search(r"([A-Z][a-z]{2,8}\.?\s+\d{1,2},\s+\d{4})", cleaned)
    if not match:
        return None

    date_text = match.group(1)
    for fmt in ["%b. %d, %Y", "%B %d, %Y", "%b %d, %Y"]:
        try:
            return datetime.strptime(date_text, fmt)
        except ValueError:
            continue
    return None


def _clock_to_seconds(clock_str: str) -> float:
    """Parse an M:SS clock string to seconds."""
    if not clock_str or clock_str == "--":
        return np.nan
    clock_str = str(clock_str).strip()
    match = re.match(r"(\d+):(\d+)", clock_str)
    if not match:
        return np.nan
    return float(int(match.group(1)) * 60 + int(match.group(2)))


def _round_time_to_total_seconds(round_num: Optional[int], finish_time: str) -> float:
    """Convert round number plus in-round clock to total fight seconds."""
    if round_num is None or pd.isna(round_num):
        return np.nan
    seconds_in_round = _clock_to_seconds(finish_time)
    if pd.isna(seconds_in_round):
        return np.nan
    return float(max((int(round_num) - 1) * 300 + seconds_in_round, 0.0))


def _rate_per_minute(value: float | None, fight_minutes: float) -> float:
    if value is None or pd.isna(value) or pd.isna(fight_minutes) or fight_minutes <= 0:
        return np.nan
    return float(value / fight_minutes)


def _rate_per_15(value: float | None, fight_minutes: float) -> float:
    if value is None or pd.isna(value) or pd.isna(fight_minutes) or fight_minutes <= 0:
        return np.nan
    return float(value / fight_minutes * 15.0)


def _pct(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator * 100.0)


def _defense_pct(landed: float | None, attempted: float | None) -> float:
    if landed is None or attempted is None or pd.isna(landed) or pd.isna(attempted) or attempted <= 0:
        return np.nan
    return float((1.0 - (landed / attempted)) * 100.0)


def _parse_ctrl_seconds(ctrl_str: str) -> float:
    """Parse control time like '4:30' to seconds. Returns NaN for missing data."""
    return _clock_to_seconds(ctrl_str)


# ---------------------------------------------------------------------------
# Fighter search on UFCStats.com
# ---------------------------------------------------------------------------

def search_fighter_url(fighter_name: str) -> Optional[str]:
    """
    Search UFCStats.com for a fighter by name. Returns their profile URL.
    Uses the alphabetical fighter listing with last name initial.
    """
    cache_key = normalize_cross_source_name(fighter_name) or normalize_person_name(fighter_name)
    if cache_key in _fighter_url_cache:
        if _cache_is_fresh(_fighter_url_cache_cached_at, cache_key):
            return _fighter_url_cache[cache_key]
        _fighter_url_cache.pop(cache_key, None)
        _fighter_url_cache_cached_at.pop(cache_key, None)

    # Use the cross-source-normalized tokens so suffixes like "Jr." do not
    # send the search to the wrong UFCStats index page.
    search_tokens = [token for token in normalize_cross_source_name(fighter_name).split() if token]
    if not search_tokens:
        search_tokens = [token for token in normalize_person_name(fighter_name).split() if token]
    if not search_tokens:
        return None

    # Try last name initial
    last_name = search_tokens[-1]
    char = last_name[0].lower()

    try:
        url = f"{UFCSTATS_FIGHTER_SEARCH_URL}?char={char}&page=all"
        soup = _get_soup(url)
    except Exception as e:
        logger.warning(f"Failed to search UFCStats for '{fighter_name}': {e}")
        return None

    for row in soup.select("tr.b-statistics__table-row"):
        cols = row.select("td")
        if len(cols) < 2:
            continue

        first_link = cols[0].select_one("a.b-link")
        last_link = cols[1].select_one("a.b-link")
        if not first_link or not last_link:
            continue

        first_name = _clean_text(first_link.text).lower()
        last_name_found = _clean_text(last_link.text).lower()
        full_name = f"{first_name} {last_name_found}"

        fighter_url = first_link.get("href", "").strip()
        if not fighter_url or "fighter-details" not in fighter_url:
            continue

        if same_person_name(fighter_name, full_name):
            _fighter_url_cache[cache_key] = fighter_url
            _fighter_url_cache_cached_at[cache_key] = time.time()
            return fighter_url

    # Fallback: try searching by first name initial (handles Eastern name order on UFCStats)
    first_char = search_tokens[0][0].lower()
    if first_char != char:
        try:
            url = f"{UFCSTATS_FIGHTER_SEARCH_URL}?char={first_char}&page=all"
            soup = _get_soup(url)
        except Exception:
            return None

        for row in soup.select("tr.b-statistics__table-row"):
            cols = row.select("td")
            if len(cols) < 2:
                continue
            first_link = cols[0].select_one("a.b-link")
            last_link = cols[1].select_one("a.b-link")
            if not first_link or not last_link:
                continue
            first_name = _clean_text(first_link.text).lower()
            last_name_found = _clean_text(last_link.text).lower()
            full_name = f"{first_name} {last_name_found}"
            fighter_url = first_link.get("href", "").strip()
            if not fighter_url or "fighter-details" not in fighter_url:
                continue
            if same_person_name(fighter_name, full_name):
                _fighter_url_cache[cache_key] = fighter_url
                _fighter_url_cache_cached_at[cache_key] = time.time()
                return fighter_url

    return None


# ---------------------------------------------------------------------------
# Fighter profile scraping
# ---------------------------------------------------------------------------

def scrape_fighter_profile(fighter_url: str, *, reference_date: Any = None) -> dict:
    """
    Scrape a fighter's profile page for career stats and physical attributes.

    Returns dict with keys: name, height, reach, weight, stance, age, dob,
        slpm, str_acc, sapm, str_def, td_avg, td_acc, td_def, sub_avg, record
    """
    soup = _get_soup(fighter_url)

    name_el = soup.select_one("h2.b-content__title span")
    name = _clean_text(name_el.text) if name_el else ""

    record_el = soup.select_one("span.b-content__title-record")
    record_str = _clean_text(record_el.text.replace("Record:", "")) if record_el else ""

    # Parse win/loss/draw from record
    wins, losses, draws = 0, 0, 0
    rec_match = re.match(r"(\d+)-(\d+)-(\d+)", record_str)
    if rec_match:
        wins = int(rec_match.group(1))
        losses = int(rec_match.group(2))
        draws = int(rec_match.group(3))

    # Physical attributes
    info = {}
    for item in soup.select("li.b-list__box-list-item"):
        text = _clean_text(item.text)
        if "Height:" in text:
            info["height_raw"] = text.replace("Height:", "").strip()
        elif "Weight:" in text:
            info["weight_raw"] = text.replace("Weight:", "").strip()
        elif "Reach:" in text:
            info["reach_raw"] = text.replace("Reach:", "").strip()
        elif "STANCE:" in text:
            info["stance"] = text.replace("STANCE:", "").strip()
        elif "DOB:" in text:
            info["dob"] = text.replace("DOB:", "").strip()

    # Career stats (per 15 min averages)
    stats = {}
    for item in soup.select("li.b-list__box-list-item_type_block"):
        text = _clean_text(item.text)
        if "SLpM:" in text:
            stats["slpm"] = _safe_float(text.replace("SLpM:", ""))
        elif "Str. Acc.:" in text:
            stats["str_acc"] = _safe_float(text.replace("Str. Acc.:", "").replace("%", ""))
        elif "SApM:" in text:
            stats["sapm"] = _safe_float(text.replace("SApM:", ""))
        elif "Str. Def:" in text:
            stats["str_def"] = _safe_float(text.replace("Str. Def:", "").replace("%", ""))
        elif "TD Avg.:" in text:
            stats["td_avg"] = _safe_float(text.replace("TD Avg.:", ""))
        elif "TD Acc.:" in text:
            stats["td_acc"] = _safe_float(text.replace("TD Acc.:", "").replace("%", ""))
        elif "TD Def.:" in text:
            stats["td_def"] = _safe_float(text.replace("TD Def.:", "").replace("%", ""))
        elif "Sub. Avg.:" in text:
            stats["sub_avg"] = _safe_float(text.replace("Sub. Avg.:", ""))

    return {
        "name": name,
        "fighter_url": fighter_url,
        "record": record_str,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "height": _inches_to_cm(_parse_height_inches(info.get("height_raw", ""))),
        "reach": _inches_to_cm(_parse_reach_inches(info.get("reach_raw", ""))),
        "weight": _parse_weight_lbs(info.get("weight_raw", "")),
        "stance": info.get("stance", ""),
        "dob": info.get("dob", ""),
        "age": _parse_dob_to_age(info.get("dob", ""), reference_date=reference_date),
        **stats,
    }


# ---------------------------------------------------------------------------
# Fighter recent fight history
# ---------------------------------------------------------------------------

def _parse_stat_cell(cell_text: str) -> tuple[str, str]:
    """Parse a cell like '50 of 100' into (landed, attempted)."""
    text = _clean_text(cell_text)
    match = re.match(r"(\d+)\s+of\s+(\d+)", text)
    if match:
        return match.group(1), match.group(2)
    return text, text


def _scrape_fight_detail(detail_url: str, fighter_name: str) -> dict:
    """
    Scrape a fight detail page for stats not on the profile table:
    Rev, Ctrl, title bout status, and Significant Strikes breakdown
    (head/body/leg/distance/clinch/ground).

    Args:
        detail_url: URL like https://ufcstats.com/fight-details/{id}
        fighter_name: Name of the fighter we're building features for,
                      used to determine which row is "ours" vs opponent.

    Returns dict with: rev, ctrl_seconds, opp_rev, opp_ctrl_seconds,
        is_title_bout, weight_class, and strike breakdown stats
        (head/body/leg/distance/clinch/ground landed+attempted for both fighters).
        All values are real parsed data or NaN — never estimated or fabricated.
    """
    result = {
        "rev": np.nan,
        "ctrl_seconds": np.nan,
        "opp_rev": np.nan,
        "opp_ctrl_seconds": np.nan,
        "is_title_bout": False,
        "weight_class": "",
        # Strike breakdown — NaN until parsed from the page
        "head_landed": np.nan,
        "head_attempted": np.nan,
        "body_landed": np.nan,
        "body_attempted": np.nan,
        "leg_landed": np.nan,
        "leg_attempted": np.nan,
        "distance_landed": np.nan,
        "distance_attempted": np.nan,
        "clinch_landed": np.nan,
        "clinch_attempted": np.nan,
        "ground_landed": np.nan,
        "ground_attempted": np.nan,
        "opp_head_landed": np.nan,
        "opp_head_attempted": np.nan,
        "opp_body_landed": np.nan,
        "opp_body_attempted": np.nan,
        "opp_leg_landed": np.nan,
        "opp_leg_attempted": np.nan,
        "opp_distance_landed": np.nan,
        "opp_distance_attempted": np.nan,
        "opp_clinch_landed": np.nan,
        "opp_clinch_attempted": np.nan,
        "opp_ground_landed": np.nan,
        "opp_ground_attempted": np.nan,
    }

    try:
        soup = _get_soup(detail_url)
    except Exception as e:
        logger.debug(f"Failed to fetch fight detail {detail_url}: {e}")
        return result

    # Title bout detection — look for belt.png or "Title Bout" text
    fight_title = soup.select_one("i.b-fight-details__fight-title")
    if fight_title:
        belt_img = fight_title.select_one("img[src*='belt.png']")
        title_text = "title bout" in fight_title.get_text().lower()
        result["is_title_bout"] = bool(belt_img or title_text)
        result["weight_class"] = _clean_text(fight_title.get_text())

    # Find the totals table — first table body with fight stats
    # The totals row has both fighters' stats in a single <tr>
    tables = soup.select("table.b-fight-details__table")
    if not tables:
        return result

    # First table is the Totals table
    totals_table = tables[0]
    body = totals_table.select_one("tbody")
    if not body:
        return result

    rows = body.select("tr.b-fight-details__table-row")
    if not rows:
        return result

    totals_row = rows[0]
    cols = totals_row.select("td")
    if len(cols) < 10:
        return result

    # Determine which <p> index (0 or 1) is our fighter
    fighter_ps = cols[0].select("p")
    if len(fighter_ps) < 2:
        return result

    name0 = _clean_text(fighter_ps[0].get_text())
    name1 = _clean_text(fighter_ps[1].get_text())

    if same_person_name(fighter_name, name0):
        our_idx = 0
    elif same_person_name(fighter_name, name1):
        our_idx = 1
    else:
        logger.debug(
            f"Could not match '{fighter_name}' to detail page fighters: "
            f"'{name0}' / '{name1}'"
        )
        return result

    opp_idx = 1 - our_idx

    # Rev is column 8 (two <p> elements)
    rev_ps = cols[8].select("p") if len(cols) > 8 else []
    if len(rev_ps) >= 2:
        result["rev"] = _safe_float(_clean_text(rev_ps[our_idx].text), np.nan)
        result["opp_rev"] = _safe_float(_clean_text(rev_ps[opp_idx].text), np.nan)

    # Ctrl is column 9 (two <p> elements, format "M:SS")
    ctrl_ps = cols[9].select("p") if len(cols) > 9 else []
    if len(ctrl_ps) >= 2:
        result["ctrl_seconds"] = _parse_ctrl_seconds(_clean_text(ctrl_ps[our_idx].text))
        result["opp_ctrl_seconds"] = _parse_ctrl_seconds(_clean_text(ctrl_ps[opp_idx].text))

    # --- Significant Strikes breakdown table (Table 1) ---
    # Columns: Fighter, Sig.Str, Sig.Str.%, Head, Body, Leg, Distance, Clinch, Ground
    # Each cell has two <p> tags (one per fighter), "X of Y" format
    if len(tables) >= 2:
        sig_table = tables[1]
        sig_body = sig_table.select_one("tbody")
        if sig_body:
            sig_rows = sig_body.select("tr.b-fight-details__table-row")
            if sig_rows:
                sig_cols = sig_rows[0].select("td")
                # Column indices: 3=Head, 4=Body, 5=Leg, 6=Distance, 7=Clinch, 8=Ground
                _breakdown_map = [
                    (3, "head"),
                    (4, "body"),
                    (5, "leg"),
                    (6, "distance"),
                    (7, "clinch"),
                    (8, "ground"),
                ]
                for col_idx, stat_name in _breakdown_map:
                    if col_idx < len(sig_cols):
                        ps = sig_cols[col_idx].select("p")
                        if len(ps) >= 2:
                            our_l, our_a = _parse_stat_cell(ps[our_idx].text)
                            opp_l, opp_a = _parse_stat_cell(ps[opp_idx].text)
                            result[f"{stat_name}_landed"] = _safe_float(our_l, np.nan)
                            result[f"{stat_name}_attempted"] = _safe_float(our_a, np.nan)
                            result[f"opp_{stat_name}_landed"] = _safe_float(opp_l, np.nan)
                            result[f"opp_{stat_name}_attempted"] = _safe_float(opp_a, np.nan)

    return result


def scrape_fighter_fights(fighter_url: str, fighter_name: str = "") -> list[dict]:
    """
    Scrape a fighter's fight history from their profile page,
    then enrich each fight with detail page data (Rev, Ctrl, title bout).

    Args:
        fighter_url: UFCStats profile URL
        fighter_name: Fighter's name (for matching on detail pages)

    Returns list of fight dicts with per-fight stats, ordered chronologically.
    """
    soup = _get_soup(fighter_url)

    fights = []
    # The fight history table on the profile page
    fight_rows = soup.select("tr.b-fight-details__table-row")

    for row in fight_rows:
        cols = row.select("td")
        if len(cols) < 10:
            continue

        # Check if this row has fight data
        result_col = cols[0].select_one("a.b-flag") or cols[0].select_one("i")
        if not result_col:
            continue

        result_text = _clean_text(result_col.text).lower()
        result = "win" if "win" in result_text else "loss" if "loss" in result_text else "neutral"
        won = 1 if result == "win" else 0

        # Fight detail URL from the row's data-link attribute or the <a> in col[0]
        detail_url = row.get("data-link", "").strip()
        if not detail_url:
            a_tag = cols[0].select_one("a.b-flag")
            if a_tag:
                detail_url = a_tag.get("href", "").strip()

        # Fighter names (cols[1] has two <p> tags)
        fighter_ps = cols[1].select("p")
        if len(fighter_ps) < 2:
            continue
        opponent = _clean_text(fighter_ps[1].text)

        # Method and round
        method = _clean_text(cols[7].text) if len(cols) > 7 else ""
        round_num_str = _clean_text(cols[8].text) if len(cols) > 8 else ""

        # Parse round — None if unparseable (no guessing)
        round_val = _safe_float(round_num_str)
        round_finished = int(round_val) if not np.isnan(round_val) else None

        # Event metadata
        event_text = _clean_text(cols[6].text) if len(cols) > 6 else ""
        event_date = _parse_event_date_text(event_text)
        finish_time = _clean_text(cols[9].text) if len(cols) > 9 else ""
        total_fight_time_secs = _round_time_to_total_seconds(round_finished, finish_time)
        fight_minutes = (
            total_fight_time_secs / 60.0
            if not pd.isna(total_fight_time_secs) and total_fight_time_secs > 0
            else np.nan
        )

        # Per-fight stats from the table (sig str, td, sub, etc.)
        # The profile fight table has: Result, Fighter, KD, Str, TD, Sub, Weight Class, Method, Round, Time
        kd = _safe_float(_clean_text(cols[2].select("p")[0].text)) if len(cols) > 2 and cols[2].select("p") else 0
        opp_kd = _safe_float(_clean_text(cols[2].select("p")[1].text)) if len(cols) > 2 and len(cols[2].select("p")) > 1 else 0

        # Sig str
        sig_str_text = cols[3].select("p")
        sig_str_landed, sig_str_attempted = 0, 0
        opp_sig_str_landed, opp_sig_str_attempted = 0, 0
        if sig_str_text and len(sig_str_text) >= 2:
            l, a = _parse_stat_cell(sig_str_text[0].text)
            sig_str_landed, sig_str_attempted = _safe_float(l, 0), _safe_float(a, 0)
            l2, a2 = _parse_stat_cell(sig_str_text[1].text)
            opp_sig_str_landed, opp_sig_str_attempted = _safe_float(l2, 0), _safe_float(a2, 0)

        # TD
        td_text = cols[4].select("p") if len(cols) > 4 else []
        td_landed, td_attempted = 0, 0
        opp_td_landed, opp_td_attempted = 0, 0
        if td_text and len(td_text) >= 2:
            l, a = _parse_stat_cell(td_text[0].text)
            td_landed, td_attempted = _safe_float(l, 0), _safe_float(a, 0)
            l2, a2 = _parse_stat_cell(td_text[1].text)
            opp_td_landed, opp_td_attempted = _safe_float(l2, 0), _safe_float(a2, 0)

        # Sub attempts
        sub_text = cols[5].select("p") if len(cols) > 5 else []
        sub_att = _safe_float(_clean_text(sub_text[0].text)) if sub_text else 0
        opp_sub_att = _safe_float(_clean_text(sub_text[1].text)) if len(sub_text) > 1 else 0

        slpm = _rate_per_minute(sig_str_landed, fight_minutes)
        sapm = _rate_per_minute(opp_sig_str_landed, fight_minutes)
        str_acc = _pct(sig_str_landed, sig_str_attempted)
        str_def = _defense_pct(opp_sig_str_landed, opp_sig_str_attempted)
        td_avg = _rate_per_15(td_landed, fight_minutes)
        td_acc = _pct(td_landed, td_attempted)
        td_def = _defense_pct(opp_td_landed, opp_td_attempted)
        sub_avg = _rate_per_15(sub_att, fight_minutes)

        fight = {
            "event_date": event_date,
            "detail_url": detail_url,
            "opponent": opponent,
            "won": won,
            "result": result,
            "winner": fighter_name if result == "win" else opponent if result == "loss" else "",
            "method": method,
            "round_finished": round_finished,
            "kd": kd,
            "sig_str_landed": sig_str_landed,
            "sig_str_attempted": sig_str_attempted,
            "td_landed": td_landed,
            "td_attempted": td_attempted,
            "sub_att": sub_att,
            "slpm": slpm,
            "sapm": sapm,
            "str_acc": str_acc,
            "str_def": str_def,
            "td_avg": td_avg,
            "td_acc": td_acc,
            "td_def": td_def,
            "sub_avg": sub_avg,
            "rev": np.nan,  # Will be filled from detail page
            "ctrl_seconds": np.nan,  # Will be filled from detail page
            "is_title_bout": False,  # Will be filled from detail page
            "weight_class": "",
            "finish_time": finish_time,
            "total_fight_time_secs": total_fight_time_secs,
            # Opponent stats
            "opp_kd": opp_kd,
            "opp_sig_str_landed": opp_sig_str_landed,
            "opp_sig_str_attempted": opp_sig_str_attempted,
            "opp_td_landed": opp_td_landed,
            "opp_td_attempted": opp_td_attempted,
            "opp_sub_att": opp_sub_att,
            "opp_rev": np.nan,
            "opp_ctrl_seconds": np.nan,
        }
        fights.append(fight)

    # Scrape fight detail pages for Rev, Ctrl, and title bout data
    if fighter_name and fights:
        logger.info(f"  Scraping {len(fights)} fight detail pages for {fighter_name}...")
        for fight in fights:
            detail_url = fight.get("detail_url", "")
            if not detail_url or "fight-details" not in detail_url:
                continue
            detail = _scrape_fight_detail(detail_url, fighter_name)
            fight["rev"] = detail["rev"]
            fight["ctrl_seconds"] = detail["ctrl_seconds"]
            fight["opp_rev"] = detail["opp_rev"]
            fight["opp_ctrl_seconds"] = detail["opp_ctrl_seconds"]
            fight["is_title_bout"] = detail["is_title_bout"]
            fight["weight_class"] = detail["weight_class"]
            # Strike breakdown from Significant Strikes table
            for breakdown_stat in [
                "head_landed", "head_attempted",
                "body_landed", "body_attempted",
                "leg_landed", "leg_attempted",
                "distance_landed", "distance_attempted",
                "clinch_landed", "clinch_attempted",
                "ground_landed", "ground_attempted",
                "opp_head_landed", "opp_head_attempted",
                "opp_body_landed", "opp_body_attempted",
                "opp_leg_landed", "opp_leg_attempted",
                "opp_distance_landed", "opp_distance_attempted",
                "opp_clinch_landed", "opp_clinch_attempted",
                "opp_ground_landed", "opp_ground_attempted",
            ]:
                fight[breakdown_stat] = detail[breakdown_stat]

    # Reverse so oldest fight is first (profile page shows newest first)
    fights.reverse()
    return fights


# ---------------------------------------------------------------------------
# Build features for a single fighter
# ---------------------------------------------------------------------------

def _compute_rolling_for_fighter(
    fights: list[dict],
    profile: dict,
    window: int = ROLLING_WINDOW,
    *,
    fighter_name: Optional[str] = None,
    as_of_date: Optional[str] = None,
    reference_date: Any = None,
    strict_mode: bool = False,
    allowed_profile_fields: Optional[set[str]] = None,
    processed_data_dir: Optional[Path] = None,
) -> dict:
    """
    Compute rolling stats and derived features for a fighter from their
    fight history and profile data.

    Returns a dict of feature values (without a_/b_ prefix).
    """
    features = {}
    allow_all_profile_fields = allowed_profile_fields is None
    reference_ts = _coerce_reference_timestamp(reference_date).normalize()

    def _profile_field_enabled(field_name: str) -> bool:
        return allow_all_profile_fields or field_name in allowed_profile_fields

    profile_age = _parse_dob_to_age(profile.get("dob", ""), reference_date=reference_ts)
    if np.isnan(profile_age):
        profile_age = profile.get("age", np.nan)

    # Physical attributes from profile (canonical units: cm for height/reach)
    features["height"] = (
        profile.get("height", np.nan)
        if (not strict_mode or _profile_field_enabled("height"))
        else np.nan
    )
    features["reach"] = (
        profile.get("reach", np.nan)
        if (not strict_mode or _profile_field_enabled("reach"))
        else np.nan
    )
    features["weight"] = (
        profile.get("weight", np.nan)
        if (not strict_mode or _profile_field_enabled("weight"))
        else np.nan
    )
    features["age"] = (
        profile_age
        if (not strict_mode or _profile_field_enabled("age"))
        else np.nan
    )

    # Stance encoding
    encoded_stance = encode_stance(profile.get("stance", ""))
    features["stance_enc"] = (
        float(encoded_stance)
        if (not strict_mode or _profile_field_enabled("stance"))
        and not pd.isna(encoded_stance)
        else np.nan
    )

    # Career record
    canonical_name = fighter_name or profile.get("name", "")
    history_results = [_history_fight_result(fight, canonical_name) for fight in fights]
    if strict_mode:
        features["wins"] = sum(1 for result in history_results if result == "win")
        features["losses"] = sum(1 for result in history_results if result == "loss")
        features["draws"] = sum(1 for result in history_results if result == "draw")
    else:
        features["wins"] = profile.get("wins", np.nan)
        features["losses"] = profile.get("losses", np.nan)
        features["draws"] = profile.get("draws", np.nan)
    total_decisive_fights = features["wins"] + features["losses"]
    features["win_pct"] = np.nan if total_decisive_fights == 0 else features["wins"] / total_decisive_fights

    # Career averages from profile (used as fallback / primary for slpm etc.)
    if strict_mode:
        features["slpm"] = np.nan
        features["sapm"] = np.nan
        features["str_acc"] = np.nan
        features["str_def"] = np.nan
        features["td_avg"] = np.nan
        features["td_acc"] = np.nan
        features["td_def"] = np.nan
        features["sub_avg"] = np.nan
    else:
        features["slpm"] = profile.get("slpm", np.nan)
        features["sapm"] = profile.get("sapm", np.nan)
        features["str_acc"] = profile.get("str_acc", np.nan)
        features["str_def"] = profile.get("str_def", np.nan)
        features["td_avg"] = profile.get("td_avg", np.nan)
        features["td_acc"] = profile.get("td_acc", np.nan)
        features["td_def"] = profile.get("td_def", np.nan)
        features["sub_avg"] = profile.get("sub_avg", np.nan)

    # Number of prior UFC fights completed before this upcoming bout
    features["num_fights"] = len(fights)

    if not fights:
        # No fight history — all rolling stats are NaN (no fabricated values)
        for stat in ["slpm", "sapm", "str_acc", "str_def", "td_avg", "td_acc",
                      "td_def", "sub_avg", "sig_str_landed", "td_landed", "kd",
                      "won"]:
            features[f"roll_{stat}"] = np.nan
        # Extended rolling stats — NaN for debut fighters
        for stat in [
            "head_str_share", "body_str_share", "leg_str_share",
            "distance_str_share", "clinch_str_share", "ground_str_share",
            "control_share", "control_per_td",
            "distance_acc", "clinch_acc", "ground_acc",
        ]:
            features[f"roll_{stat}"] = np.nan
        if strict_mode:
            features["current_win_streak"] = 0.0
            features["lose_streak"] = 0.0
            features["longest_win_streak"] = 0.0
            features["days_since_last_fight"] = np.nan
            features["cage_rust"] = np.nan
            features["layoff_log"] = np.nan
            features["total_rounds"] = 0.0
            features["title_bouts"] = 0.0
            features["strike_diff"] = np.nan
            features["fight_pace"] = np.nan
            features["ctrl_efficiency"] = np.nan
            features["ko_rate"] = np.nan
            features["sub_rate"] = np.nan
            features["dec_rate"] = np.nan
            # v6 new features — NaN for debut fighters (except age-derived)
            age = features.get("age")
            if age is not None and not np.isnan(age):
                features["age_over_35"] = float(age > 35)
                features["age_under_25"] = float(age < 25)
                features["age_squared"] = age ** 2
            else:
                features["age_over_35"] = np.nan
                features["age_under_25"] = np.nan
                features["age_squared"] = np.nan
            features["ko_absorption"] = np.nan
            features["strikes_avoided_pct"] = np.nan
        else:
            features["current_win_streak"] = np.nan
            features["days_since_last_fight"] = np.nan
        return features

    # --- Rolling stats: two categories ---
    # 1. Rate stats (slpm, sapm, str_acc, str_def, td_avg, td_acc, td_def, sub_avg):
    #    Training pipeline uses career-level snapshots from the Kaggle data (which IS
    #    the UFCStats career average at each fight point). The profile page career
    #    average is the best live approximation of the latest snapshot.
    #
    # 2. Per-fight count stats (kd, sig_str_landed, etc.): Training computes EWM
    #    over per-fight values. Live path replicates this from scraped fight history.

    halflife = EWM_HALFLIFE

    if strict_mode:
        rate_stats = ["slpm", "sapm", "str_acc", "str_def", "td_avg", "td_acc", "td_def", "sub_avg"]
        rate_df = pd.DataFrame(
            [{stat: fight.get(stat, np.nan) for stat in rate_stats} for fight in fights]
        )
        for stat in rate_stats:
            if stat in rate_df.columns:
                ewm_val = rate_df[stat].ewm(halflife=halflife, min_periods=1).mean().iloc[-1]
                features[f"roll_{stat}"] = float(ewm_val) if not np.isnan(ewm_val) else np.nan
    else:
        # Non-strict mode uses current profile aggregates as the legacy approximation.
        features["roll_slpm"] = profile.get("slpm", np.nan)
        features["roll_sapm"] = profile.get("sapm", np.nan)
        features["roll_str_acc"] = profile.get("str_acc", np.nan)
        features["roll_str_def"] = profile.get("str_def", np.nan)
        features["roll_td_avg"] = profile.get("td_avg", np.nan)
        features["roll_td_acc"] = profile.get("td_acc", np.nan)
        features["roll_td_def"] = profile.get("td_def", np.nan)
        features["roll_sub_avg"] = profile.get("sub_avg", np.nan)

    # Per-fight count stats: EWM over scraped fight history (shifted by 1)
    per_fight_stats = [
        "kd", "sig_str_landed", "sig_str_attempted",
        "td_landed", "td_attempted", "sub_att", "rev", "ctrl_seconds",
        # Strike breakdown stats (from fight detail page Sig Strikes table)
        "head_landed", "head_attempted",
        "body_landed", "body_attempted",
        "leg_landed", "leg_attempted",
        "distance_landed", "distance_attempted",
        "clinch_landed", "clinch_attempted",
        "ground_landed", "ground_attempted",
    ]
    opp_stats = [
        "opp_kd", "opp_sig_str_landed", "opp_sig_str_attempted",
        "opp_td_landed", "opp_td_attempted", "opp_sub_att",
        "opp_rev", "opp_ctrl_seconds",
        # Opponent strike breakdown
        "opp_head_landed", "opp_head_attempted",
        "opp_body_landed", "opp_body_attempted",
        "opp_leg_landed", "opp_leg_attempted",
        "opp_distance_landed", "opp_distance_attempted",
        "opp_clinch_landed", "opp_clinch_attempted",
        "opp_ground_landed", "opp_ground_attempted",
    ]

    all_count_stats = per_fight_stats + opp_stats + ["won"]
    count_df = pd.DataFrame([
        {stat: f.get(stat, np.nan) for stat in all_count_stats}
        for f in fights
    ])

    # NOTE: No shift(1) here, unlike build_features.py. In the training pipeline,
    # shift(1) prevents leakage (the current fight's result must not predict itself).
    # But here, `fights` only contains PAST fights — the upcoming fight is not in
    # this list. So we use ALL past fights directly.
    for stat in all_count_stats:
        if stat in count_df.columns:
            ewm_val = count_df[stat].ewm(halflife=halflife, min_periods=1).mean().iloc[-1]
            features[f"roll_{stat}"] = float(ewm_val) if not np.isnan(ewm_val) else np.nan

    # Win streak (consecutive wins going back from most recent)
    streak = 0
    for result in reversed(history_results):
        if result == "win":
            streak += 1
        else:
            break
    features["current_win_streak"] = streak

    # Lose streak
    lose_streak = 0
    for result in reversed(history_results):
        if result == "loss":
            lose_streak += 1
        else:
            break
    features["lose_streak"] = lose_streak

    # Longest win streak ever
    longest_streak = 0
    current = 0
    for result in history_results:
        if result == "win":
            current += 1
            longest_streak = max(longest_streak, current)
        else:
            current = 0
    features["longest_win_streak"] = longest_streak

    # Days since last fight
    last_fight = fights[-1]
    if last_fight.get("event_date"):
        last_fight_date = pd.Timestamp(last_fight["event_date"])
        if last_fight_date.tz is not None:
            last_fight_date = last_fight_date.tz_convert("UTC").tz_localize(None)
        days = (reference_ts - last_fight_date.normalize()).days
        features["days_since_last_fight"] = max(days, 0)
    else:
        features["days_since_last_fight"] = np.nan

    # Cage rust and layoff log
    dslf = features["days_since_last_fight"]
    if np.isnan(dslf):
        features["cage_rust"] = np.nan
        features["layoff_log"] = np.nan
    else:
        features["cage_rust"] = 1 if dslf > 365 else 0
        features["layoff_log"] = np.log1p(dslf)

    # Total rounds — sum actual rounds from scraped fight history
    round_vals = [f.get("round_finished") for f in fights if f.get("round_finished") is not None]
    features["total_rounds"] = sum(round_vals) if round_vals else np.nan

    # Title bouts — count from fight detail page scrape
    title_count = sum(1 for f in fights if f.get("is_title_bout", False))
    features["title_bouts"] = title_count

    # Strike differential mirrors the training feature, but fight pace follows
    # experimental_features.add_fight_pace() and defaults missing inputs to 0.
    slpm = features.get("roll_slpm")
    sapm = features.get("roll_sapm")
    if slpm is not None and sapm is not None and not (np.isnan(slpm) or np.isnan(sapm)):
        features["strike_diff"] = slpm - sapm
    else:
        features["strike_diff"] = np.nan
    _slpm = features.get("roll_slpm")
    _sapm = features.get("roll_sapm")
    if _slpm is not None and _sapm is not None and not (np.isnan(_slpm) or np.isnan(_sapm)):
        features["fight_pace"] = _slpm + _sapm
    else:
        features["fight_pace"] = np.nan

    # Cage time efficiency follows experimental_features.add_cage_time_efficiency().
    ctrl_raw = features.get("roll_ctrl_seconds")
    landed_raw = features.get("roll_sig_str_landed")
    if (ctrl_raw is not None and landed_raw is not None
            and not (np.isnan(ctrl_raw) or np.isnan(landed_raw))):
        ctrl = max(ctrl_raw, 1.0)
        features["ctrl_efficiency"] = landed_raw / ctrl
    else:
        features["ctrl_efficiency"] = np.nan

    # --- Age nonlinearity features (v6 new) ---
    age = features.get("age")
    if age is not None and not np.isnan(age):
        features["age_over_35"] = float(age > 35)
        features["age_under_25"] = float(age < 25)
        features["age_squared"] = age ** 2
    else:
        features["age_over_35"] = np.nan
        features["age_under_25"] = np.nan
        features["age_squared"] = np.nan

    # --- KO absorption (v6 new) ---
    # Uses rolling average of knockdowns absorbed (opponent's KDs landed on us)
    opp_kd = features.get("roll_opp_kd")
    features["ko_absorption"] = opp_kd if opp_kd is not None else np.nan

    # --- Strikes avoided percentage (v6 new) ---
    opp_sig_landed = features.get("roll_opp_sig_str_landed")
    opp_sig_attempted = features.get("roll_opp_sig_str_attempted")
    if (opp_sig_landed is not None and opp_sig_attempted is not None
            and not (np.isnan(opp_sig_landed) or np.isnan(opp_sig_attempted))
            and opp_sig_attempted > 0):
        features["strikes_avoided_pct"] = 1.0 - opp_sig_landed / opp_sig_attempted
    else:
        features["strikes_avoided_pct"] = np.nan

    # --- Extended strike features (v6) ---
    # Derived from rolling EWM of per-fight breakdown stats (now scraped from
    # fight detail pages). All values are real parsed data or NaN — no estimates.
    roll_sig = features.get("roll_sig_str_landed")
    _has_sig = roll_sig is not None and not np.isnan(roll_sig) and roll_sig > 0

    # Target shares: fraction of sig strikes to each target
    for target in ["head", "body", "leg"]:
        roll_target = features.get(f"roll_{target}_landed")
        if _has_sig and roll_target is not None and not np.isnan(roll_target):
            features[f"roll_{target}_str_share"] = roll_target / roll_sig
        else:
            features[f"roll_{target}_str_share"] = np.nan

    # Position shares: fraction of sig strikes from each position
    for position in ["distance", "clinch", "ground"]:
        roll_pos = features.get(f"roll_{position}_landed")
        if _has_sig and roll_pos is not None and not np.isnan(roll_pos):
            features[f"roll_{position}_str_share"] = roll_pos / roll_sig
        else:
            features[f"roll_{position}_str_share"] = np.nan

    # Position accuracies: landed / attempted from each position
    for position in ["distance", "clinch", "ground"]:
        roll_l = features.get(f"roll_{position}_landed")
        roll_a = features.get(f"roll_{position}_attempted")
        if (roll_l is not None and roll_a is not None
                and not np.isnan(roll_l) and not np.isnan(roll_a) and roll_a > 0):
            features[f"roll_{position}_acc"] = roll_l / roll_a
        else:
            features[f"roll_{position}_acc"] = np.nan

    # Control share: ctrl_seconds / total fight time (approximated from rolling)
    roll_ctrl = features.get("roll_ctrl_seconds")
    # control_share needs fight-time context; use NaN since we don't have per-fight
    # total time in the rolling average. Training pipeline computes this per-fight
    # then rolls — at inference we compute from the rolled values if available.
    features.setdefault("roll_control_share", np.nan)

    # Control per takedown
    roll_td = features.get("roll_td_landed")
    if (roll_ctrl is not None and roll_td is not None
            and not np.isnan(roll_ctrl) and not np.isnan(roll_td) and roll_td > 0):
        features["roll_control_per_td"] = roll_ctrl / roll_td
    else:
        features["roll_control_per_td"] = np.nan

    # Finish rates — count win methods from fight history
    total_wins = sum(1 for result in history_results if result == "win")
    if total_wins > 0:
        ko_wins = sum(
            1
            for fight, result in zip(fights, history_results)
            if result == "win" and "ko" in str(fight.get("method", "")).lower()
        )
        sub_wins = sum(
            1
            for fight, result in zip(fights, history_results)
            if result == "win" and "sub" in str(fight.get("method", "")).lower()
        )
        dec_wins = sum(
            1
            for fight, result in zip(fights, history_results)
            if result == "win" and "dec" in str(fight.get("method", "")).lower()
        )
        features["ko_rate"] = ko_wins / total_wins
        features["sub_rate"] = sub_wins / total_wins
        features["dec_rate"] = dec_wins / total_wins
    else:
        features["ko_rate"] = np.nan
        features["sub_rate"] = np.nan
        features["dec_rate"] = np.nan

    return features


# ---------------------------------------------------------------------------
# Line movement features (live)
# ---------------------------------------------------------------------------

def get_line_movement_live(
    fighter_a: str,
    fighter_b: str,
    current_a_prob: Optional[float] = None,
    current_b_prob: Optional[float] = None,
    event_id: Optional[str] = None,
    commence_time: Optional[str] = None,
    as_of_date: Optional[str] = None,
) -> dict:
    """
    Get line movement features for live prediction from persisted history only.

    Missing history is a collector/data-coverage issue and resolves to NaN.
    """
    from src.data.line_movement import analysis_to_line_movement_features, line_movement_nan_features

    try:
        from src.data.line_tracker import analyze_line_movement

        analysis = analyze_line_movement(
            fighter_a,
            fighter_b,
            event_id=event_id,
            commence_time=commence_time,
            as_of_date=as_of_date,
        )
        return analysis_to_line_movement_features(analysis)
    except Exception as e:
        logger.debug(f"line_tracker failed for {fighter_a} vs {fighter_b}: {e}")

    logger.warning(f"All line movement sources failed for {fighter_a} vs {fighter_b}")
    return line_movement_nan_features()


def _compute_wc_move_from_history(
    fighter_data: Optional[dict],
    current_weight_class: Optional[str],
) -> float:
    """Compute weight-class movement using the training prior-mode semantics."""
    current_wc_weight = _weight_class_to_weight(current_weight_class)
    if current_wc_weight is None:
        return 0.0
    if not fighter_data:
        return np.nan

    prior_weights = [
        weight
        for weight in (
            _weight_class_to_weight(fight.get("weight_class"))
            for fight in fighter_data.get("fights", [])
        )
        if weight is not None
    ]
    if not prior_weights:
        if fighter_data.get("source") == "processed":
            return 0.0
        return np.nan

    home_weight = Counter(prior_weights).most_common(1)[0][0]
    return float(int(home_weight != current_wc_weight))


def _h2h_summary(
    fighter_a: str,
    fighter_b: str,
    a_data: Optional[dict],
    b_data: Optional[dict],
) -> tuple[int, int]:
    """Compute rematch and H2H record diff using a symmetric source of truth."""
    records: list[tuple[pd.Timestamp, str]] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for owner_name, data in ((fighter_a, a_data), (fighter_b, b_data)):
        if not data:
            continue
        for fight in data.get("fights", []):
            opponent = fight.get("opponent", "")
            if not same_person_name(opponent, fighter_b if same_person_name(owner_name, fighter_a) else fighter_a):
                continue

            event_date = pd.to_datetime(fight.get("event_date"), errors="coerce")
            event_key = event_date.isoformat() if pd.notna(event_date) else ""
            dedupe_key = (
                normalize_person_name(owner_name),
                normalize_person_name(opponent),
                event_key,
            )
            mirror_key = (
                normalize_person_name(opponent),
                normalize_person_name(owner_name),
                event_key,
            )
            if dedupe_key in seen_keys or mirror_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)

            result = str(fight.get("result", "") or "").lower()
            if result == "neutral":
                continue

            winner = fight.get("winner", "")
            if result not in {"win", "loss"}:
                if same_person_name(winner, owner_name):
                    result = "win"
                elif same_person_name(winner, opponent):
                    result = "loss"
                else:
                    continue

            records.append((event_date, "a" if result == "win" and same_person_name(owner_name, fighter_a) else
                            "b" if result == "loss" and same_person_name(owner_name, fighter_a) else
                            "b" if result == "win" and same_person_name(owner_name, fighter_b) else
                            "a"))

    if not records:
        return 0, 0

    a_wins = sum(1 for _event_date, winner in records if winner == "a")
    b_wins = sum(1 for _event_date, winner in records if winner == "b")
    return 1, a_wins - b_wins


# ---------------------------------------------------------------------------
# Main public API: build features for a fight
# ---------------------------------------------------------------------------

def lookup_fighter(
    fighter_name: str,
    as_of_date: Optional[str] = None,
    *,
    reference_date: Any = None,
    prefer_live_refresh: bool = False,
    training_spec: Any = None,
    processed_data_dir: Optional[Path] = None,
) -> Optional[dict]:
    """
    Look up a fighter's complete stats from UFCStats.com, with fallback
    to Sherdog/Tapology for fighters not in the UFC database.

    `reference_date` controls live age/layoff calculations without changing the
    historical fail-closed semantics of `as_of_date`.

    Returns dict with profile info, fight history, and computed rolling stats.
    Caches results for the session to avoid redundant scraping.
    """
    resolved_spec = _coerce_training_spec(training_spec)
    resolved_processed_data_dir = _resolve_processed_data_dir(
        training_spec=resolved_spec,
        processed_data_dir=processed_data_dir,
    )
    strict_mode = _is_provenance_strict_spec(resolved_spec)
    allowed_profile_fields = _requested_profile_fields(resolved_spec)
    cache_key = _fighter_cache_key(
        fighter_name,
        as_of_date,
        reference_date=reference_date,
        processed_data_dir=resolved_processed_data_dir,
    )
    if cache_key in _fighter_cache:
        if _cache_is_fresh(_fighter_cache_cached_at, cache_key):
            return _fighter_cache[cache_key]
        _fighter_cache.pop(cache_key, None)
        _fighter_cache_cached_at.pop(cache_key, None)

    logger.info(f"Looking up fighter stats: {fighter_name}")

    processed_result = _lookup_processed_fighter(
        fighter_name,
        as_of_date=as_of_date,
        reference_date=reference_date,
        processed_data_dir=resolved_processed_data_dir,
    )
    processed_live_stale = (
        prefer_live_refresh
        and as_of_date is None
        and reference_date is not None
        and str(reference_date).strip()
        and _processed_data_is_stale_for_live(
            reference_date=reference_date,
            processed_data_dir=resolved_processed_data_dir,
        )
    )
    if processed_result is not None and not processed_live_stale:
        _fighter_cache[cache_key] = processed_result
        _fighter_cache_cached_at[cache_key] = time.time()
        return processed_result
    if processed_result is not None and processed_live_stale:
        logger.info(
            "Processed live snapshot for %s may be stale relative to %s; attempting live refresh first",
            fighter_name,
            _coerce_reference_timestamp(reference_date).date().isoformat(),
        )

    if as_of_date is not None:
        logger.warning(
            "Processed-history miss for %s as of %s; failing closed instead of scraping current data",
            fighter_name,
            as_of_date,
        )
        return None

    profile = None
    fights = []
    source = "ufcstats"

    # Step 1: Try UFCStats
    fighter_url = search_fighter_url(fighter_name)
    if fighter_url:
        try:
            profile = scrape_fighter_profile(fighter_url, reference_date=reference_date)
        except Exception as e:
            logger.warning(f"Failed to scrape profile for {fighter_name}: {e}")

        if profile:
            try:
                fights = scrape_fighter_fights(
                    fighter_url, fighter_name=profile.get("name", fighter_name)
                )
            except Exception as e:
                logger.warning(f"Failed to scrape fights for {fighter_name}: {e}")
                fights = []

    # Step 2: Fallback to Sherdog/Tapology if UFCStats failed
    if profile is None:
        from src.data.fallback_scrapers import fallback_lookup

        logger.info(f"  UFCStats miss for {fighter_name}, trying fallback sources...")
        fallback_result = fallback_lookup(fighter_name)
        if fallback_result:
            profile, fights = fallback_result
            source = "fallback"

    if profile is None:
        logger.warning(f"Could not find {fighter_name} on any source")
        if processed_result is not None:
            logger.warning(
                "Falling back to processed snapshot for %s because live refresh failed",
                fighter_name,
            )
            _fighter_cache[cache_key] = processed_result
            _fighter_cache_cached_at[cache_key] = time.time()
            return processed_result
        return None

    # Step 3: Compute rolling stats
    rolling = _compute_rolling_for_fighter(
        fights,
        profile,
        fighter_name=fighter_name,
        as_of_date=as_of_date,
        reference_date=reference_date,
        strict_mode=strict_mode,
        allowed_profile_fields=allowed_profile_fields,
        processed_data_dir=resolved_processed_data_dir,
    )

    result = {
        "profile": profile,
        "fights": fights,
        "features": rolling,
        "source": source,
    }

    _fighter_cache[cache_key] = result
    _fighter_cache_cached_at[cache_key] = time.time()
    logger.info(
        f"  {fighter_name} [{source}]: {profile.get('record', '?')} | "
        f"{len(fights)} fights found | "
        f"SLpM: {rolling.get('roll_slpm', '?')}"
    )
    return result


def build_fight_features(
    fighter_a: str,
    fighter_b: str,
    odds_features: Optional[dict] = None,
    weight_class: Optional[str] = None,
    is_title_bout: bool = False,
    is_empty_arena: Optional[float] = None,
    num_rounds: int = 3,
    as_of_date: Optional[str] = None,
    event_id: Optional[str] = None,
    commence_time: Optional[str] = None,
    prefer_live_refresh: bool = False,
    training_spec: Any = None,
    processed_data_dir: Optional[Path] = None,
    include_provenance: bool = False,
) -> dict | tuple[dict, dict[str, Any]]:
    """
    Build a complete feature dict for a fight, compatible with the trained model.

    Looks up both fighters on UFCStats.com, computes rolling stats,
    differentials, and combines with odds features.

    Args:
        fighter_a: Name of fighter A
        fighter_b: Name of fighter B
        odds_features: Dict with a_implied_prob, b_implied_prob, etc.
        weight_class: Weight class string
        is_title_bout: Whether this is a title fight
        is_empty_arena: Explicit live event empty-arena flag when available
        num_rounds: Number of rounds (3 or 5)
        as_of_date: Optional cutoff for date-aware historical lookups
        event_id: Optional live event identifier for snapshot-backed external joins
        commence_time: Optional live event start time for snapshot-backed joins
            and live age/layoff calculations

    Returns:
        Dict of feature_name -> value, ready for predict_fight().
        When `include_provenance=True`, returns `(features, provenance)`.
    """
    features = {}
    resolved_spec = _coerce_training_spec(training_spec)
    requested_feature_cols = _requested_feature_columns(resolved_spec)
    requested_feature_set = set(requested_feature_cols) if requested_feature_cols is not None else None
    resolved_processed_data_dir = _resolve_processed_data_dir(
        training_spec=resolved_spec,
        processed_data_dir=processed_data_dir,
    )

    # Look up both fighters
    lookup_kwargs = {
        "training_spec": resolved_spec,
        "processed_data_dir": resolved_processed_data_dir,
    }
    if commence_time is not None and as_of_date is None:
        lookup_kwargs["reference_date"] = commence_time
    if prefer_live_refresh:
        lookup_kwargs["prefer_live_refresh"] = True
    if as_of_date is None:
        a_data = _call_lookup_fighter(fighter_a, **lookup_kwargs)
        b_data = _call_lookup_fighter(fighter_b, **lookup_kwargs)
    else:
        a_data = _call_lookup_fighter(fighter_a, as_of_date=as_of_date, **lookup_kwargs)
        b_data = _call_lookup_fighter(fighter_b, as_of_date=as_of_date, **lookup_kwargs)

    provenance = (
        _build_lookup_provenance(
            fighter_a_result=a_data,
            fighter_b_result=b_data,
            processed_data_dir=resolved_processed_data_dir,
            training_spec=resolved_spec,
        )
        if include_provenance
        else None
    )

    a_feats = a_data["features"] if a_data else {}
    b_feats = b_data["features"] if b_data else {}

    # Prefix features with a_/b_
    for key, val in a_feats.items():
        features[f"a_{key}"] = val
    for key, val in b_feats.items():
        features[f"b_{key}"] = val

    # NaN is preserved for ko_rate, sub_rate, dec_rate, fight_pace,
    # ctrl_efficiency — XGBoost learned missing-value branch directions
    # during training and must see NaN (not 0.0) at inference.

    # Compute differentials
    diff_stats = [
        "roll_slpm", "roll_sapm", "roll_str_acc", "roll_str_def",
        "roll_td_avg", "roll_td_acc", "roll_td_def", "roll_sub_avg",
        "roll_sig_str_landed", "roll_td_landed", "roll_kd",
        "roll_won", "current_win_streak", "num_fights", "days_since_last_fight",
        "height", "reach", "weight", "age", "strike_diff",
        "ko_rate", "sub_rate", "dec_rate", "win_pct",
        "lose_streak", "longest_win_streak", "total_rounds", "title_bouts", "draws",
        "cage_rust", "layoff_log",
        "fight_pace", "ctrl_efficiency",
        # v6 new features
        "age_over_35", "age_under_25", "age_squared",
        "ko_absorption", "strikes_avoided_pct",
        # Extended strike rolling stats (NaN at live inference until scraper enhanced)
        "roll_head_str_share", "roll_body_str_share", "roll_leg_str_share",
        "roll_distance_str_share", "roll_clinch_str_share", "roll_ground_str_share",
        "roll_control_share", "roll_control_per_td",
        "roll_distance_acc", "roll_clinch_acc", "roll_ground_acc",
    ]

    for stat in diff_stats:
        a_val = a_feats.get(stat)
        b_val = b_feats.get(stat)
        if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
            if not (np.isnan(a_val) or np.isnan(b_val)):
                features[f"diff_{stat}"] = a_val - b_val

    # Opponent strength: A's opponent is B → A's opp_strength = B's win rate
    a_roll_won = features.get("a_roll_won")
    b_roll_won = features.get("b_roll_won")
    if isinstance(b_roll_won, (int, float)) and not np.isnan(b_roll_won):
        features["a_opp_strength"] = b_roll_won
    else:
        features["a_opp_strength"] = np.nan
    if isinstance(a_roll_won, (int, float)) and not np.isnan(a_roll_won):
        features["b_opp_strength"] = a_roll_won
    else:
        features["b_opp_strength"] = np.nan
    a_os = features["a_opp_strength"]
    b_os = features["b_opp_strength"]
    if isinstance(a_os, (int, float)) and isinstance(b_os, (int, float)) and not (np.isnan(a_os) or np.isnan(b_os)):
        features["diff_opp_strength"] = a_os - b_os

    # Pace mismatch: absolute difference in fight pace
    a_pace = features.get("a_fight_pace")
    b_pace = features.get("b_fight_pace")
    if (isinstance(a_pace, (int, float)) and isinstance(b_pace, (int, float))
            and not (np.isnan(a_pace) or np.isnan(b_pace))):
        features["pace_mismatch"] = abs(a_pace - b_pace)
    else:
        features["pace_mismatch"] = np.nan

    # Stance same?
    a_stance = pd.to_numeric(pd.Series([a_feats.get("stance_enc", np.nan)]), errors="coerce").iloc[0]
    b_stance = pd.to_numeric(pd.Series([b_feats.get("stance_enc", np.nan)]), errors="coerce").iloc[0]
    if _wants_feature(requested_feature_set, "same_stance"):
        features["same_stance"] = float(a_stance == b_stance) if pd.notna(a_stance) and pd.notna(b_stance) else np.nan

    # NOTE: finish-rate / pace / efficiency diffs are already computed by the
    # general diff_stats loop above with NaN propagation — no second pass needed.

    # Pre-UFC career summary features (from Sherdog/Tapology supplement)
    if _wants_feature(requested_feature_set, "a_pre_ufc_total_fights", "b_pre_ufc_total_fights"):
        _pre_ufc_summary = _load_pre_ufc_summary_cache()

        # On-demand scrape: if a fighter is missing from the supplement and
        # we're in live-prediction mode (not backtest), scrape Sherdog +
        # Tapology now and persist the result.
        if as_of_date is None:
            for _data_obj, fighter in [(a_data, fighter_a), (b_data, fighter_b)]:
                if fighter in _pre_ufc_summary:
                    continue
                _on_demand_pre_ufc_scrape(fighter, _data_obj, _pre_ufc_summary)

        for prefix, fighter in [("a_", fighter_a), ("b_", fighter_b)]:
            fighter_summary = _pre_ufc_summary.get(fighter, {})
            for col in [
                "pre_ufc_total_fights", "pre_ufc_wins", "pre_ufc_losses",
                "pre_ufc_win_pct", "pre_ufc_ko_rate", "pre_ufc_sub_rate",
                "pre_ufc_dec_rate", "pre_ufc_org_tier_best",
            ]:
                features[f"{prefix}{col}"] = fighter_summary.get(col, np.nan)
        # Differentials
        for col in [
            "pre_ufc_total_fights", "pre_ufc_wins", "pre_ufc_losses",
            "pre_ufc_win_pct", "pre_ufc_ko_rate", "pre_ufc_sub_rate",
            "pre_ufc_dec_rate", "pre_ufc_org_tier_best",
        ]:
            a_v = features.get(f"a_{col}", np.nan)
            b_v = features.get(f"b_{col}", np.nan)
            if isinstance(a_v, (int, float)) and isinstance(b_v, (int, float)):
                if not (np.isnan(a_v) or np.isnan(b_v)):
                    features[f"diff_{col}"] = a_v - b_v

    # Weight class encoding
    if weight_class and _wants_feature(requested_feature_set, "weight_class_enc"):
        wc_order = {
            "strawweight": 0, "women's strawweight": 0,
            "flyweight": 1, "women's flyweight": 1,
            "bantamweight": 2, "women's bantamweight": 2,
            "featherweight": 3, "women's featherweight": 3,
            "lightweight": 4, "welterweight": 5, "middleweight": 6,
            "light heavyweight": 7, "heavyweight": 8, "catch weight": 5,
        }
        wc_lower = weight_class.lower()
        features["weight_class_enc"] = next(
            (v for k, v in wc_order.items() if k in wc_lower), float("nan")
        )

    # Meta features
    if _wants_feature(requested_feature_set, "is_title_bout"):
        features["is_title_bout"] = int(is_title_bout)
    if _wants_feature(requested_feature_set, "num_rounds_feat"):
        features["num_rounds_feat"] = float(num_rounds)
    if _wants_feature(requested_feature_set, "is_empty_arena"):
        features["is_empty_arena"] = float(is_empty_arena) if is_empty_arena is not None else np.nan

    # Style matchup interactions
    if _wants_feature(
        requested_feature_set,
        "a_striker_edge",
        "b_striker_edge",
        "diff_striker_edge",
        "a_grappler_edge",
        "b_grappler_edge",
        "diff_grappler_edge",
    ):
        for prefix_atk, prefix_def in [("a_", "b_"), ("b_", "a_")]:
            ko_rate = _coerce_numeric_feature(features.get(f"{prefix_atk}ko_rate"), default=np.nan)
            str_def_val = _coerce_percentage_feature(features.get(f"{prefix_def}roll_str_def"), default=np.nan)
            sub_rate = _coerce_numeric_feature(features.get(f"{prefix_atk}sub_rate"), default=np.nan)
            td_def_val = _coerce_percentage_feature(features.get(f"{prefix_def}roll_td_def"), default=np.nan)

            features[f"{prefix_atk}striker_edge"] = ko_rate * (1.0 - str_def_val / 100.0)
            features[f"{prefix_atk}grappler_edge"] = sub_rate * (1.0 - td_def_val / 100.0)

        features["diff_striker_edge"] = (
            features.get("a_striker_edge", np.nan) - features.get("b_striker_edge", np.nan)
        )
        features["diff_grappler_edge"] = (
            features.get("a_grappler_edge", np.nan) - features.get("b_grappler_edge", np.nan)
        )

    # Weight class moves — use the same prior-mode semantics as training,
    # and fail closed when live history cannot support the computation.
    if _wants_feature(requested_feature_set, "a_wc_move", "b_wc_move", "diff_wc_move"):
        features["a_wc_move"] = _compute_wc_move_from_history(a_data, weight_class)
        features["b_wc_move"] = _compute_wc_move_from_history(b_data, weight_class)
        if not (np.isnan(features["a_wc_move"]) or np.isnan(features["b_wc_move"])):
            features["diff_wc_move"] = features["a_wc_move"] - features["b_wc_move"]
        else:
            features["diff_wc_move"] = np.nan

    # Rematch detection — symmetric and neutral on draws/no-contests.
    if _wants_feature(requested_feature_set, "is_rematch", "h2h_record_diff"):
        features["is_rematch"], features["h2h_record_diff"] = _h2h_summary(
            fighter_a,
            fighter_b,
            a_data,
            b_data,
        )

    # ------------------------------------------------------------------
    # Phase 2 features: Line Movement, Rankings, Method Odds —
    # closing the training/inference feature gap.
    # ------------------------------------------------------------------

    # --- Group 3: Line Movement (from snapshots / opening odds cache) ---
    # Extract current implied probs from odds_features if available
    if _wants_feature(
        requested_feature_set,
        "line_movement",
        "line_abs_movement",
        "line_is_sharp",
        "line_steam_move",
        "line_direction_toward_a",
        "line_direction_toward_b",
    ):
        current_a_prob = None
        current_b_prob = None
        if odds_features:
            current_a_prob = odds_features.get("a_implied_prob")
            current_b_prob = odds_features.get("b_implied_prob")

        line_feats = get_line_movement_live(
            fighter_a,
            fighter_b,
            current_a_prob=current_a_prob,
            current_b_prob=current_b_prob,
            event_id=event_id,
            commence_time=commence_time,
            as_of_date=as_of_date,
        )
        features.update(line_feats)

    # --- Group 4: Rankings (UFC.com / ESPN / Tapology) ---
    if _wants_feature(
        requested_feature_set,
        "a_wc_rank_feat",
        "b_wc_rank_feat",
        "diff_wc_rank",
        "a_pfp_rank_feat",
        "b_pfp_rank_feat",
        "diff_pfp_rank",
    ):
        try:
            from src.data.rankings_scraper import get_fighter_rankings
            a_ranks = get_fighter_rankings(
                fighter_a,
                weight_class=weight_class,
                as_of_date=as_of_date,
            )
            b_ranks = get_fighter_rankings(
                fighter_b,
                weight_class=weight_class,
                as_of_date=as_of_date,
            )
            features["a_wc_rank_feat"] = a_ranks["wc_rank_feat"]
            features["b_wc_rank_feat"] = b_ranks["wc_rank_feat"]
            features["diff_wc_rank"] = a_ranks["wc_rank_feat"] - b_ranks["wc_rank_feat"]
            features["a_pfp_rank_feat"] = a_ranks["pfp_rank_feat"]
            features["b_pfp_rank_feat"] = b_ranks["pfp_rank_feat"]
            features["diff_pfp_rank"] = a_ranks["pfp_rank_feat"] - b_ranks["pfp_rank_feat"]
        except Exception as e:
            logger.warning(f"Rankings acquisition failed: {e}")
            for col in ["a_wc_rank_feat", "b_wc_rank_feat", "diff_wc_rank",
                         "a_pfp_rank_feat", "b_pfp_rank_feat", "diff_pfp_rank"]:
                features[col] = np.nan

    # --- Group 5: Method Odds (The Odds API / BestFightOdds) ---
    if _wants_feature(
        requested_feature_set,
        "a_ko_odds_prob",
        "a_sub_odds_prob",
        "a_dec_odds_prob",
        "b_ko_odds_prob",
        "b_sub_odds_prob",
        "b_dec_odds_prob",
    ):
        try:
            from src.data.method_odds import get_method_odds
            method_feats = get_method_odds(
                fighter_a,
                fighter_b,
                event_id=event_id,
                commence_time=commence_time,
                as_of_date=as_of_date,
            )
            features.update(method_feats)
        except Exception as e:
            logger.warning(f"Method odds acquisition failed: {e}")
            for col in ["a_ko_odds_prob", "a_sub_odds_prob", "a_dec_odds_prob",
                         "b_ko_odds_prob", "b_sub_odds_prob", "b_dec_odds_prob"]:
                features[col] = np.nan

    # Add odds features last (these override if provided)
    if odds_features:
        if requested_feature_set is None:
            features.update(odds_features)
        else:
            for key, value in odds_features.items():
                if key in requested_feature_set:
                    features[key] = value

    if requested_feature_cols is not None:
        selected_features = {column: features.get(column, np.nan) for column in requested_feature_cols}
        if include_provenance:
            return selected_features, provenance
        return selected_features

    if include_provenance:
        return features, provenance
    return features


def clear_cache():
    """Clear the fighter lookup cache (including fallback scraper caches)."""
    _fighter_cache.clear()
    _fighter_cache_cached_at.clear()
    _fighter_url_cache.clear()
    _fighter_url_cache_cached_at.clear()
    _processed_feature_history_cache.clear()
    _processed_feature_history_mtime.clear()
    _elo_state_cache.clear()
    _elo_state_cache_mtime.clear()
    from src.data.fallback_scrapers import clear_fallback_cache
    clear_fallback_cache()
