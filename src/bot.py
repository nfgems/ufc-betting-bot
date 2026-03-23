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
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DEFAULT_TENNIS_MODEL_NAME,
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
    TENNIS_KELLY_FRACTION,
    TENNIS_MIN_EDGE_THRESHOLD,
    TENNIS_PORTFOLIO_SHARE,
    TENNIS_TRADER_ENABLED,
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
_LIVE_TRADE_START_BUFFER = timedelta(minutes=10)
_LAST_GOOD_LIVE_EVENT_CONTEXTS: tuple[float, list[dict]] | None = None


def _default_training_spec():
    """Return the promoted training contract for top-level train/retrain flows."""
    from src.model.training_spec import full_live_contract_v6_spec

    return full_live_contract_v6_spec()


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


def _live_fight_pair_key(fighter_a: str, fighter_b: str) -> str:
    from src.data.name_utils import normalize_cross_source_name

    return "|".join(sorted([normalize_cross_source_name(fighter_a), normalize_cross_source_name(fighter_b)]))


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


def _load_live_event_contexts() -> list[dict]:
    """Fetch upcoming UFC event metadata used to populate live model context fields."""
    global _LAST_GOOD_LIVE_EVENT_CONTEXTS
    try:
        from src.data.live_monitor import collect_upcoming_fight_contexts
        last_exc = None
        for attempt in range(1, 4):
            try:
                contexts = list(collect_upcoming_fight_contexts())
                if contexts:
                    _LAST_GOOD_LIVE_EVENT_CONTEXTS = (time.monotonic(), contexts)
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
        if _LAST_GOOD_LIVE_EVENT_CONTEXTS is not None:
            age_seconds = time.monotonic() - _LAST_GOOD_LIVE_EVENT_CONTEXTS[0]
            logger.warning(
                "Using cached live UFC event context from %.0fs ago after fetch failure",
                age_seconds,
            )
            return list(_LAST_GOOD_LIVE_EVENT_CONTEXTS[1])
        if last_exc is not None:
            logger.warning("Could not load live UFC event context: %s", last_exc)
        return []
    except Exception as exc:
        logger.warning("Could not load live UFC event context: %s", exc)
        if _LAST_GOOD_LIVE_EVENT_CONTEXTS is not None:
            return list(_LAST_GOOD_LIVE_EVENT_CONTEXTS[1])
        return []


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
    return (
        "not on any upcoming UFC card and no fight history found - likely a non-UFC MMA "
        "event or fighters not in database"
    )


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


def _resolve_live_event_context(fight, live_event_contexts: list[dict]) -> dict | None:
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

    # Fallback: infer weight class from fighter history
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
    from src.strategy.backtest import run_backtest, plot_backtest

    if args.static:
        # Static backtest: single train/test split
        logger.info("Running static backtest (single train/test split)...")
        test_path = PROCESSED_DATA_DIR / "test_set.csv"
        if not test_path.exists():
            logger.error("Test set not found. Run 'train' first.")
            return

        test_df = pd.read_csv(test_path, parse_dates=["event_date"])

        result = run_backtest(
            test_df,
            model_name=args.model,
            initial_bankroll=args.bankroll,
            min_edge=args.min_edge,
            kelly_fraction=args.kelly,
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
    live_event_contexts = _load_live_event_contexts()

    for _, fight in consensus.iterrows():
        fighter_a = fight["fighter_a"]
        fighter_b = fight["fighter_b"]
        market_a = fight["a_fair_prob_avg"]
        market_b = fight["b_fair_prob_avg"]
        can_trade, start_reason, _ = _live_fight_is_tradeable(fight.get("commence_time"))
        if not can_trade:
            logger.warning(
                "Skipping %s vs %s: %s (event_id=%s commence_time=%s)",
                fighter_a,
                fighter_b,
                start_reason,
                fight.get("event_id", ""),
                fight.get("commence_time", ""),
            )
            continue
        event_context = _resolve_live_event_context(fight, live_event_contexts)
        if event_context is None:
            logger.warning(
                "Skipping %s vs %s: %s (event_id=%s commence_time=%s)",
                fighter_a,
                fighter_b,
                _missing_live_event_context_reason(fighter_a, fighter_b),
                fight.get("event_id", ""),
                fight.get("commence_time", ""),
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


def _build_tennis_prediction_frame(model_name: str = DEFAULT_TENNIS_MODEL_NAME):
    """Build live tennis predictions from bookmaker odds and historical ATP/WTA data."""
    import pandas as pd

    from src.config import TENNIS_MIN_MATCHES
    from src.data.tennis_data import load_processed_tennis_data
    from src.data.tennis_odds import fetch_live_tennis_consensus
    from src.features.tennis_features import build_live_tennis_features, filter_minimum_history
    from src.model.tennis_model import load_tennis_model, predict_tennis_batch

    try:
        history_df = load_processed_tennis_data()
    except FileNotFoundError:
        logger.error("Processed tennis data not found. Run 'tennis-train' first.")
        return None

    try:
        consensus = fetch_live_tennis_consensus()
    except Exception as exc:
        logger.error(f"Failed to fetch live tennis odds: {exc}")
        logger.info("Make sure ODDS_API_KEY is set in .env")
        return None

    if consensus.empty:
        logger.info("No live ATP/WTA singles odds found.")
        return None

    try:
        model_result = load_tennis_model(model_name)
    except FileNotFoundError:
        logger.error("Tennis model not found. Run 'tennis-train' first.")
        return None

    live_features = build_live_tennis_features(consensus, history_df)
    gated_features = filter_minimum_history(live_features, min_matches=TENNIS_MIN_MATCHES)
    dropped = len(live_features) - len(gated_features)
    if dropped > 0:
        logger.info(
            "Skipped %s live tennis matches because one or both players had fewer than %s prior matches.",
            dropped,
            TENNIS_MIN_MATCHES,
        )
    if gated_features.empty:
        logger.info("No live tennis matches passed the %s-match minimum-history gate.", TENNIS_MIN_MATCHES)
        return pd.DataFrame()

    predictions = predict_tennis_batch(gated_features, model_result=model_result)
    return predictions


def _save_tennis_live_snapshot(frame, filename: str):
    """Persist the latest tennis live decision frame."""
    import pandas as pd

    if frame is None:
        return None
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)

    output_path = PROCESSED_DATA_DIR / "tennis" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    logger.info("Saved tennis live snapshot to %s", output_path)
    return output_path


def cmd_tennis_discover(args):
    """Discover active tennis sport keys, live bookmaker matches, and Polymarket markets."""
    from src.data.tennis_odds import discover_active_tennis_sports, fetch_live_tennis_consensus
    from src.polymarket.tennis_markets import discover_tennis_markets, match_tennis_markets

    try:
        sports = discover_active_tennis_sports()
        consensus = fetch_live_tennis_consensus(sports=sports)
    except Exception as exc:
        logger.error(f"Failed to discover live tennis sports: {exc}")
        logger.info("Make sure ODDS_API_KEY is set in .env")
        return

    markets = discover_tennis_markets(active_tennis_sports=sports)
    matched = match_tennis_markets(consensus, markets)

    logger.info("Active tennis sport keys: %s", len(sports))
    for sport in sports[:10]:
        logger.info(
            "  %s | %s | %s",
            sport.get("sport_key"),
            sport.get("tour", "").upper(),
            sport.get("tournament_name"),
        )

    logger.info("Live bookmaker tennis matches: %s", len(consensus))
    logger.info("Discovered active tennis Polymarket markets: %s", len(markets))
    logger.info("Current bookmaker/Polymarket tennis matches: %s", len(matched))

    for _, row in matched.head(10).iterrows():
        logger.info(
            "  MATCHED: %s vs %s | %s | %s",
            row["fighter_a"],
            row["fighter_b"],
            row.get("tour", "").upper(),
            row.get("market_event_title", row.get("event_title", "")),
        )


def cmd_tennis_train(args):
    """Download tennis history, build leak-free features, and train a tennis model."""
    from src.config import TENNIS_TRAINING_START_DATE
    from src.data.tennis_data import load_processed_tennis_data, prepare_tennis_data, save_processed_tennis_data
    from src.features.tennis_features import build_tennis_features, save_tennis_features
    from src.model.tennis_model import (
        filter_tennis_training_window,
        train_tennis_model,
        write_tennis_oos_artifacts,
    )

    logger.info("Preparing ATP/WTA singles history...")
    matches_df = prepare_tennis_data(
        start_year=args.start_year,
        end_year=args.end_year,
        force_download=args.force_download,
        refresh_current_year=True,
        fetch_missing_player_profiles=args.refresh_player_profiles,
        fetch_missing_rankings_history=args.refresh_rankings_history,
    )
    if matches_df.empty:
        logger.error("No tennis history loaded.")
        return

    matches_df = filter_tennis_training_window(matches_df)
    if matches_df.empty:
        logger.error("No tennis history remained after applying the strict %s training boundary.", TENNIS_TRAINING_START_DATE)
        return
    save_processed_tennis_data(matches_df)
    matches_df = load_processed_tennis_data()
    logger.info(
        "Tennis training universe: %s rows from %s through %s",
        len(matches_df),
        matches_df["event_date"].min().date(),
        matches_df["event_date"].max().date(),
    )

    logger.info("Building tennis features...")
    features_df = build_tennis_features(matches_df)
    features_path = PROCESSED_DATA_DIR / "tennis" / "features.csv"
    save_tennis_features(features_df, str(features_path))
    logger.info(
        "Tennis training features: %s rows from %s through %s",
        len(features_df),
        features_df["event_date"].min().date(),
        features_df["event_date"].max().date(),
    )

    logger.info("Training tennis model '%s'...", args.model)
    model_result = train_tennis_model(features_df, model_name=args.model)

    evaluation_dir = PROCESSED_DATA_DIR / "tennis"
    artifacts = write_tennis_oos_artifacts(model_result, evaluation_dir)

    logger.info("Tennis training complete. Model saved to models/tennis/")
    for label, path in artifacts.items():
        logger.info("Saved tennis %s artifact to %s", label, path)

    metrics = model_result.get("evaluation_metrics", {})
    summary = model_result.get("evaluation_summary", {})
    if metrics:
        logger.info("Tennis 2022+ OOS log loss: %.4f", metrics.get("log_loss", float("nan")))
        logger.info("Tennis 2022+ OOS Brier score: %.4f", metrics.get("brier_score", float("nan")))
    if summary:
        logger.info(
            "Final tennis model training rows: %s (%s to %s)",
            model_result.get("training_rows"),
            model_result.get("training_date_min"),
            model_result.get("training_date_max"),
        )
        logger.info(
            "Tennis anchored OOS window: %s to %s across %s folds",
            summary.get("oos_start_date"),
            summary.get("oos_end_date_exclusive"),
            len(summary.get("folds", [])),
        )
        logger.info(
            "Tennis 2022+ OOS coverage: %s/%s eligible rows",
            summary.get("oos_prediction_rows"),
            summary.get("eligible_oos_rows"),
        )


def cmd_tennis_player_profiles(args):
    """Fetch official ATP/WTA player profiles and audit their static-field coverage."""
    from src.data.tennis_data import load_tennis_matches, normalize_tennis_matches, save_processed_tennis_data
    from src.data.tennis_player_profiles import (
        collect_tennis_player_profile_targets,
        download_tennis_player_profiles,
        download_secondary_tennis_player_profiles,
        enrich_tennis_matches_with_player_profiles,
        load_tennis_player_profiles,
        write_tennis_player_profile_remaining_targets,
        summarize_tennis_player_profile_enrichment,
        write_tennis_player_profile_enrichment_summary,
        write_tennis_player_profile_targets,
    )

    logger.info("Loading tennis history for player-profile targeting...")
    raw_matches = load_tennis_matches(
        start_year=args.start_year,
        end_year=args.end_year,
        force_download=args.force_download,
    )
    normalized = normalize_tennis_matches(raw_matches)
    if normalized.empty:
        logger.error("No tennis history loaded for player-profile targeting.")
        return

    targets = collect_tennis_player_profile_targets(
        normalized,
        missing_only=not args.all_players,
        official_window_only=not args.all_players,
        official_start_year=2025,
    )
    targets_path = write_tennis_player_profile_targets(targets)
    logger.info("Saved tennis player-profile targets to %s", targets_path)
    logger.info("Official tennis player-profile target count: %s", len(targets))

    if not targets.empty:
        saved_paths = download_tennis_player_profiles(targets, force=args.force_download)
        for path in saved_paths:
            logger.info("Saved official tennis player profiles to %s", path)

    profiles_df = load_tennis_player_profiles()
    enriched = enrich_tennis_matches_with_player_profiles(normalized, profiles_df=profiles_df)

    secondary_targets = collect_tennis_player_profile_targets(
        enriched,
        missing_only=True,
        official_window_only=not args.all_players,
        official_start_year=2025,
    )
    if not secondary_targets.empty:
        saved_paths = download_secondary_tennis_player_profiles(secondary_targets, force=args.force_download)
        for path in saved_paths:
            logger.info("Saved supplemental tennis player profiles to %s", path)
        profiles_df = load_tennis_player_profiles()
        enriched = enrich_tennis_matches_with_player_profiles(normalized, profiles_df=profiles_df)

    remaining_targets = collect_tennis_player_profile_targets(
        enriched,
        missing_only=True,
        official_window_only=not args.all_players,
        official_start_year=2025,
    )
    remaining_targets_path = write_tennis_player_profile_remaining_targets(
        remaining_targets,
        path=None,
        already_filtered=True,
    )
    logger.info("Saved remaining unresolved tennis player-profile targets to %s", remaining_targets_path)
    summary = summarize_tennis_player_profile_enrichment(
        normalized,
        enriched,
        profiles_df=profiles_df,
    )
    summary_path = write_tennis_player_profile_enrichment_summary(summary)
    logger.info("Saved tennis player-profile enrichment summary to %s", summary_path)

    for column, filled in sorted((summary.get("filled_counts") or {}).items()):
        logger.info("Tennis player-profile fills: %s = %s", column, filled)

    if args.refresh_processed:
        save_processed_tennis_data(enriched)
        logger.info("Refreshed processed tennis matches with official player-profile enrichment.")


def cmd_tennis_refresh_daily(args):
    """Refresh tennis history, rolling profiles, and rankings for scheduled daily runs."""
    import pandas as pd

    from src.config import TENNIS_TRAINING_START_DATE
    from src.data.tennis_data import prepare_tennis_data, save_processed_tennis_data
    from src.data.tennis_odds import fetch_live_tennis_consensus
    from src.data.tennis_player_profiles import (
        collect_live_tennis_player_profile_seed_targets,
        download_secondary_tennis_player_profiles,
        enrich_tennis_matches_with_player_profiles,
        load_tennis_player_profiles,
        summarize_tennis_player_profile_enrichment,
        write_tennis_player_profile_enrichment_summary,
        write_tennis_player_profile_remaining_targets,
        collect_tennis_player_profile_targets,
    )

    start_year = args.start_year if args.start_year is not None else int(str(TENNIS_TRAINING_START_DATE)[:4])
    logger.info("Refreshing tennis match history, rankings, and cached player profiles...")
    matches_df = prepare_tennis_data(
        start_year=start_year,
        end_year=args.end_year,
        force_download=args.force_download,
        refresh_current_year=True,
        fetch_missing_player_profiles=True,
        fetch_missing_rankings_history=True,
    )

    if args.skip_live_seeds:
        logger.info("Skipped live-player profile seeding by request.")
        return

    try:
        live_consensus = fetch_live_tennis_consensus()
    except Exception as exc:
        logger.warning("Daily tennis refresh could not fetch live consensus for player seeding: %s", exc)
        return

    if live_consensus.empty:
        logger.info("No live tennis consensus rows available for player-profile seeding.")
        return

    profiles_before = load_tennis_player_profiles()
    live_targets = collect_live_tennis_player_profile_seed_targets(
        live_consensus,
        profiles_df=profiles_before,
        missing_only=True,
    )
    logger.info("Live tennis player-profile seed targets: %s", len(live_targets))
    if live_targets.empty:
        return

    saved_paths = download_secondary_tennis_player_profiles(live_targets, force=args.force_download)
    for path in saved_paths:
        logger.info("Saved supplemental tennis player profiles to %s", path)

    profiles_after = load_tennis_player_profiles()
    before_enrichment = matches_df.copy()
    matches_df = enrich_tennis_matches_with_player_profiles(matches_df, profiles_df=profiles_after)
    save_processed_tennis_data(matches_df)

    remaining_targets = collect_tennis_player_profile_targets(
        matches_df,
        missing_only=True,
        official_window_only=False,
    )
    write_tennis_player_profile_remaining_targets(remaining_targets, already_filtered=True)
    summary = summarize_tennis_player_profile_enrichment(
        before_enrichment,
        matches_df,
        profiles_df=profiles_after,
    )
    write_tennis_player_profile_enrichment_summary(summary)
    logger.info("Daily tennis refresh complete.")


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


def cmd_tennis_rankings_history(args):
    """Fetch official ATP/WTA rankings history and audit rank-field coverage."""
    from src.data.tennis_data import load_tennis_matches, normalize_tennis_matches, save_processed_tennis_data
    from src.data.tennis_rankings_history import (
        enrich_tennis_matches_with_rankings_history,
        summarize_tennis_rankings_enrichment,
        write_tennis_rankings_enrichment_summary,
    )

    logger.info("Loading tennis history for rankings-history targeting...")
    raw_matches = load_tennis_matches(
        start_year=args.start_year,
        end_year=args.end_year,
        force_download=args.force_download,
    )
    normalized = normalize_tennis_matches(raw_matches)
    if normalized.empty:
        logger.error("No tennis history loaded for rankings-history targeting.")
        return

    enriched = enrich_tennis_matches_with_rankings_history(
        normalized,
        fetch_missing=True,
        force_download=args.force_download,
    )
    summary = summarize_tennis_rankings_enrichment(normalized, enriched)
    summary_path = write_tennis_rankings_enrichment_summary(summary)
    logger.info("Saved tennis rankings-history enrichment summary to %s", summary_path)

    for column, filled in sorted((summary.get("filled_counts") or {}).items()):
        logger.info("Tennis rankings-history fills: %s = %s", column, filled)

    if args.refresh_processed:
        save_processed_tennis_data(enriched)
        logger.info("Refreshed processed tennis matches with official rankings-history enrichment.")


def cmd_tennis_bookmaker_audit(args):
    """Audit historical bookmaker evidence for tennis-only joins and timestamps."""
    from src.data.tennis_bookmaker_audit import run_tennis_bookmaker_audit

    result = run_tennis_bookmaker_audit(
        source=args.source,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    logger.info(
        "Tennis bookmaker audit verdict: %s via %s",
        str(result.get("verdict", "unknown")).upper(),
        result.get("source"),
    )
    for label in ["coverage_summary", "join_by_year", "timestamp_checks", "unmatched_examples"]:
        path = result.get(f"{label}_path")
        if path:
            logger.info("Saved tennis bookmaker %s to %s", label.replace("_", " "), path)
    reason = result.get("reason")
    if reason:
        logger.info("Audit reason: %s", reason)


def cmd_tennis_predict(args):
    """Predict live ATP/WTA singles matches using a saved tennis model."""
    import pandas as pd

    from src.config import TENNIS_MIN_EDGE_THRESHOLD
    from src.strategy.tennis_decision import (
        annotate_tennis_reference_edges,
        apply_tennis_automation_controls,
    )
    from src.strategy.tennis_llm_operator import apply_tennis_llm_veto

    predictions = _build_tennis_prediction_frame(model_name=args.model)
    if predictions is None or predictions.empty:
        return
    decisions = annotate_tennis_reference_edges(
        predictions,
        min_edge=TENNIS_MIN_EDGE_THRESHOLD,
    )
    decisions = apply_tennis_automation_controls(decisions)
    decisions = apply_tennis_llm_veto(decisions)
    _save_tennis_live_snapshot(decisions, "live_reference_decisions.csv")
    _save_tennis_live_snapshot(
        decisions[decisions["automation_status"].isin(["auto_skip", "auto_block"])].copy(),
        "live_reference_auto_skipped.csv",
    )
    _save_tennis_live_snapshot(
        decisions[decisions["trade_status"] == "reference_only_eligible"].copy(),
        "live_reference_auto_eligible.csv",
    )
    logger.info("Reference status counts: %s", decisions["decision_status"].value_counts().to_dict())
    logger.info("Automation status counts: %s", decisions["automation_status"].value_counts().to_dict())
    logger.info("LLM veto status counts: %s", decisions["llm_veto_status"].value_counts().to_dict())
    logger.info("Trade status counts: %s", decisions["trade_status"].value_counts().to_dict())

    logger.info("\nLive tennis predictions:")
    logger.info("%s", "=" * 80)

    for _, row in decisions.sort_values(["commence_time", "tour", "fighter_a"]).iterrows():
        logger.info(
            "\n%s vs %s [%s]",
            row["fighter_a"],
            row["fighter_b"],
            str(row.get("tour", "")).upper(),
        )
        logger.info(
            "  Tournament: %s | Sport key: %s",
            row.get("tournament_name", ""),
            row.get("sport_key", ""),
        )
        logger.info(
            "  Bookmakers: %s %.1f%% | %s %.1f%% (%s books)",
            row["fighter_a"],
            row["a_fair_prob_avg"] * 100,
            row["fighter_b"],
            row["b_fair_prob_avg"] * 100,
            int(row["num_bookmakers"]),
        )
        logger.info(
            "  Model:      %s %.1f%% | %s %.1f%%",
            row["fighter_a"],
            row["prob_a"] * 100,
            row["fighter_b"],
            row["prob_b"] * 100,
        )
        logger.info(
            "  Decision:   %s | Model %.1f%% vs Market %.1f%% | Edge %+0.1f%% | Required %+0.1f%%",
            row.get("decision_fighter", ""),
            float(row.get("decision_model_prob", float("nan"))) * 100,
            float(row.get("decision_market_prob", float("nan"))) * 100,
            float(row.get("decision_edge", float("nan"))) * 100,
            float(row.get("required_edge", float("nan"))) * 100,
        )
        logger.info(
            "  Status:     %s | Reasons: %s | Min prior matches: %s",
            row.get("decision_status", ""),
            row.get("decision_reasons", "") or "none",
            int(row["min_player_matches"]) if pd.notna(row.get("min_player_matches")) else "n/a",
        )
        logger.info(
            "  Automation: %s | Execution allowed: %s | Reasons: %s",
            row.get("automation_status", ""),
            bool(row.get("execution_allowed", False)),
            row.get("automation_reasons", "") or "none",
        )
        logger.info(
            "  Cross-check: %s | Detail: %s | LLM veto: %s | Coverage: %s | Contradiction: %s",
            row.get("second_source_status", ""),
            row.get("second_source_detail", "") or "none",
            row.get("llm_veto_status", ""),
            row.get("llm_coverage_quality", "") or "unknown",
            row.get("llm_contradiction_strength", "") or "unknown",
        )
        logger.info(
            "  Trade:      %s | Trade ready: %s | Trade reasons: %s",
            row.get("trade_status", ""),
            bool(row.get("trade_ready", False)),
            row.get("trade_reasons", "") or "none",
        )


def cmd_tennis_live(args):
    """Run the tennis dry-run pipeline without placing orders."""
    from src.config import (
        INITIAL_BANKROLL,
        TENNIS_MIN_EDGE_THRESHOLD,
    )
    from src.polymarket.tennis_markets import discover_tennis_markets, match_tennis_markets
    from src.strategy.tennis_decision import (
        apply_tennis_automation_controls,
        build_tennis_execution_decisions,
    )
    from src.strategy.tennis_llm_operator import apply_tennis_llm_veto

    if not args.dry_run:
        logger.warning("Tennis live trading enabled — proceed with caution.")

    predictions = _build_tennis_prediction_frame(model_name=args.model)
    if predictions is None or predictions.empty:
        return

    markets = discover_tennis_markets()
    if markets.empty:
        logger.info("No active tennis Polymarket markets found.")
        return

    matched = match_tennis_markets(predictions, markets)
    if matched.empty:
        logger.info("No live tennis matches could be matched to Polymarket markets.")
        return

    min_edge = args.min_edge if args.min_edge is not None else TENNIS_MIN_EDGE_THRESHOLD
    decisions = build_tennis_execution_decisions(
        matched,
        min_edge=min_edge,
        bankroll=INITIAL_BANKROLL,
    )
    if decisions.empty:
        logger.info("No tennis dry-run decisions could be constructed.")
        return
    decisions = apply_tennis_automation_controls(decisions)
    decisions = apply_tennis_llm_veto(decisions)
    _save_tennis_live_snapshot(decisions, "live_execution_decisions.csv")
    _save_tennis_live_snapshot(
        decisions[decisions["automation_status"].isin(["auto_skip", "auto_block"])].copy(),
        "live_execution_auto_skipped.csv",
    )
    _save_tennis_live_snapshot(
        decisions[decisions["trade_ready"]].copy(),
        "live_execution_tradeable.csv",
    )

    status_counts = decisions["execution_status"].value_counts().to_dict()
    automation_status_counts = decisions["automation_status"].value_counts().to_dict()
    llm_veto_status_counts = decisions["llm_veto_status"].value_counts().to_dict()
    trade_status_counts = decisions["trade_status"].value_counts().to_dict()
    opportunities = decisions[decisions["trade_ready"]].copy()
    logger.info("Running tennis dry-run on %s matched markets...", len(matched))
    logger.info("Decision status counts: %s", status_counts)
    logger.info("Automation status counts: %s", automation_status_counts)
    logger.info("LLM veto status counts: %s", llm_veto_status_counts)
    logger.info("Trade status counts: %s", trade_status_counts)

    if opportunities.empty:
        logger.info("No tennis dry-run opportunities cleared the %.1f%% execution edge threshold.", min_edge * 100)
        return

    for _, opportunity in opportunities.sort_values("execution_edge", ascending=False).iterrows():
        logger.info(
            "\nDRY RUN: %s vs %s [%s]",
            opportunity["fighter_a"],
            opportunity["fighter_b"],
            str(opportunity.get("tour", "")).upper(),
        )
        logger.info(
            "  Tournament: %s | Sport key: %s",
            opportunity.get("tournament_name", ""),
            opportunity.get("sport_key", ""),
        )
        logger.info(
            "  Hypothetical bet: %s @ %.3f (%.2f USD, %.2f decimal odds)",
            opportunity["decision_fighter"],
            opportunity["execution_price"],
            opportunity["stake_usd"],
            opportunity["execution_decimal_odds"],
        )
        logger.info(
            "  Model %.1f%% | Bookmaker %.1f%% | Polymarket %.1f%%",
            opportunity["decision_model_prob"] * 100,
            opportunity["decision_market_prob"] * 100,
            opportunity["execution_price"] * 100,
        )
        logger.info(
            "  Reference edge %+0.1f%% | Execution edge %+0.1f%% | Required %+0.1f%%",
            opportunity["decision_edge"] * 100,
            opportunity["execution_edge"] * 100,
            opportunity["required_edge"] * 100,
        )
        logger.info("  Reasons: %s", opportunity.get("decision_reasons", "") or "none")
        logger.info("  Polymarket market id: %s", opportunity.get("market_id"))

    logger.info(
        "Tennis dry-run complete. Hypothetical bets: %s | Bankroll basis: %.2f USD",
        len(opportunities),
        INITIAL_BANKROLL,
    )


def _slice_wallet_basis(
    total_equity: float,
    available_cash: float,
    *,
    share: float,
    label: str,
    source: str,
):
    from src.strategy.duo_trader import WalletBankrollBasis

    clamped_share = max(0.0, min(1.0, float(share)))
    return WalletBankrollBasis(
        total_equity=round(total_equity * clamped_share, 2),
        available_cash=round(available_cash * clamped_share, 2),
        source=f"{source}; {label} sleeve {clamped_share:.0%}",
    )


def _build_tennis_trade_candidates(
    *,
    model_name: str = DEFAULT_TENNIS_MODEL_NAME,
    min_edge: float = TENNIS_MIN_EDGE_THRESHOLD,
):
    import pandas as pd

    from src.polymarket.tennis_markets import discover_tennis_markets, match_tennis_markets
    from src.strategy.tennis_decision import (
        apply_tennis_automation_controls,
        build_tennis_execution_decisions,
    )
    from src.strategy.tennis_llm_operator import apply_tennis_llm_veto

    predictions = _build_tennis_prediction_frame(model_name=model_name)
    if predictions is None or predictions.empty:
        return pd.DataFrame()

    markets = discover_tennis_markets()
    if markets.empty:
        logger.info("Skipping tennis trader: no active tennis Polymarket markets found.")
        return pd.DataFrame()

    matched = match_tennis_markets(predictions, markets)
    if matched.empty:
        logger.info("Skipping tennis trader: no live tennis matches matched to Polymarket.")
        return pd.DataFrame()

    decisions = build_tennis_execution_decisions(
        matched,
        min_edge=min_edge,
    )
    if decisions.empty:
        return pd.DataFrame()

    decisions = apply_tennis_automation_controls(decisions)
    decisions = apply_tennis_llm_veto(decisions)
    opportunities = decisions[decisions["trade_ready"]].copy()
    if opportunities.empty:
        logger.info("Skipping tennis trader: no tennis opportunities are trade-ready.")
        return opportunities

    opportunities["bet_on"] = opportunities["decision_fighter"]
    opportunities["bet_side"] = opportunities["decision_side"]
    opportunities["model_prob"] = opportunities["decision_model_prob"].astype(float)
    opportunities["blended_prob"] = opportunities["decision_model_prob"].astype(float)
    opportunities["market_prob"] = opportunities["execution_price"].astype(float)
    opportunities["edge"] = opportunities["execution_edge"].astype(float)
    opportunities["decimal_odds"] = opportunities["execution_decimal_odds"].astype(float)
    if "market_event_date" in opportunities.columns:
        opportunities["event_date"] = opportunities["market_event_date"]
    elif "commence_time" in opportunities.columns:
        opportunities["event_date"] = opportunities["commence_time"]
    else:
        opportunities["event_date"] = ""

    return opportunities.sort_values("edge", ascending=False).reset_index(drop=True)


def _run_tennis_single_trader(
    *,
    trade_candidates,
    bankroll_basis,
    clob,
    dry_run: bool,
    min_edge: float = TENNIS_MIN_EDGE_THRESHOLD,
):
    import pandas as pd

    from src.polymarket.executor import OrderExecutor, assert_live_wallet_exposure_synced
    from src.polymarket.tracker import BetLedger
    from src.strategy.bankroll import BankrollManager
    from src.strategy.duo_trader import TENNIS_LEDGER

    if trade_candidates is None or trade_candidates.empty:
        return {"name": "Tennis Trader", "orders": [], "total_orders": 0}
    if bankroll_basis.total_equity <= 0 or bankroll_basis.available_cash <= 0:
        logger.info(
            "Skipping tennis trader: sleeve has equity $%.2f and cash $%.2f",
            bankroll_basis.total_equity,
            bankroll_basis.available_cash,
        )
        return {"name": "Tennis Trader", "orders": [], "total_orders": 0}

    bankroll = BankrollManager(
        initial_bankroll=bankroll_basis.total_equity,
        total_equity=bankroll_basis.total_equity,
        available_cash=bankroll_basis.available_cash,
        kelly_fraction=TENNIS_KELLY_FRACTION,
        max_bet_fraction=MAX_BET_FRACTION,
        stop_loss_fraction=STOP_LOSS_FRACTION,
        auto_detect_balance=False,
    )
    executor = OrderExecutor(
        bankroll=bankroll,
        clob_client=clob,
        dry_run=dry_run,
        min_edge_threshold=min_edge,
        edge_scaling_base=min_edge,
    )
    executor.ledger = BetLedger(path=TENNIS_LEDGER)

    if not dry_run and clob is not None:
        assert_live_wallet_exposure_synced(
            markets=trade_candidates,
            clob_client=clob,
            import_ledger_path=TENNIS_LEDGER,
        )

    executor.refresh_open_limit_orders(
        matched_predictions=trade_candidates,
        primary_bets=trade_candidates,
        trader_name="Tennis Trader",
    )

    orders = []
    for _, bet in trade_candidates.iterrows():
        if bankroll.is_stopped:
            logger.warning("Tennis trader stop-loss triggered - skipping remaining bets")
            break
        order = executor._place_bet(bet, trade_candidates)
        if order:
            order["trader"] = "T"
            orders.append(order)

    total_wagered = sum(order.get("bet_size_usd", 0.0) for order in orders)
    return {
        "name": "Tennis Trader",
        "allocation": bankroll_basis.total_equity,
        "available_cash_start": bankroll_basis.available_cash,
        "orders": orders,
        "total_wagered": total_wagered,
        "bankroll_remaining": bankroll.available_cash,
        "total_equity": bankroll.total_equity,
        "stats": bankroll.get_stats(),
        "total_orders": len(orders),
    }


def cmd_tennis_lockbox_eval(args):
    """Run a frozen tennis model on a strict holdout lockbox window."""
    from src.data.tennis_data import load_processed_tennis_data
    from src.features.tennis_features import build_tennis_features, save_tennis_features
    from src.model.tennis_model import (
        infer_tennis_feature_contract,
        run_lockbox_evaluation,
        write_tennis_lockbox_artifacts,
    )

    history_df = load_processed_tennis_data()
    features_df = build_tennis_features(history_df)
    save_tennis_features(features_df, str(PROCESSED_DATA_DIR / "tennis" / "features.csv"))

    feature_contract = infer_tennis_feature_contract(args.model)
    evaluation = run_lockbox_evaluation(
        features_df,
        lockbox_start_date=args.lockbox_start,
        min_matches=args.min_matches,
        feature_contract=feature_contract,
        model_name=args.model,
    )
    artifacts = write_tennis_lockbox_artifacts(
        evaluation,
        PROCESSED_DATA_DIR / "tennis",
        model_name=args.model,
        lockbox_start_date=args.lockbox_start,
    )
    summary = evaluation["summary"]
    logger.info(
        "Tennis lockbox evaluation complete for %s from %s: log loss %.4f | Brier %.4f | accuracy %.4f | ECE %.4f",
        args.model,
        args.lockbox_start,
        summary.get("log_loss", float("nan")),
        summary.get("brier_score", float("nan")),
        summary.get("accuracy", float("nan")),
        summary.get("ece_10_bin", float("nan")),
    )
    for label, path in artifacts.items():
        logger.info("Saved tennis lockbox %s artifact to %s", label.replace("_", " "), path)


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
    runtime_label = "PORTFOLIO" if TENNIS_TRADER_ENABLED else "DUO TRADER"
    logger.info(f"Starting {runtime_label} bot in {mode} mode...")

    clob = None if dry_run else ClobClientWrapper()

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
    live_event_contexts = _load_live_event_contexts()

    # Set up SHAP explainer for prediction explanations
    import numpy as np
    shap_explainer = None
    shap_base_value = None
    try:
        import shap
        raw_model = model_result.get("raw_model")
        if raw_model is not None:
            shap_explainer = shap.TreeExplainer(raw_model)
            ev = shap_explainer.expected_value
            ev = np.atleast_1d(ev)
            shap_base_value = float(ev[1]) if len(ev) > 1 else float(ev[0])
    except ImportError:
        logger.info("shap not installed — predictions page will use feature highlights only")
    except Exception as e:
        logger.warning(f"Failed to create SHAP explainer: {e}")

    # Extract feature cols/medians and global importance for cache enrichment
    _feat_cols = model_result["feature_cols"]
    _col_medians = model_result["col_medians"]
    _global_importance = sorted(
        model_result.get("feature_importance", {}).items(),
        key=lambda x: x[1], reverse=True,
    )[:25]
    cache_write_warning_emitted = False

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

    # 2. Get Polymarket markets
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

    # 3. Generate predictions (same for both traders — they differ only in blend weight)
    logger.info("Generating model predictions...")
    prediction_rows = []
    _operator_features_by_fight: dict[str, dict] = {}  # for LLM Operator
    _operator_provenance_by_fight: dict[str, dict] = {}
    for _, fight in consensus.iterrows():
        fighter_a = fight["fighter_a"]
        fighter_b = fight["fighter_b"]
        can_trade, start_reason, _ = _live_fight_is_tradeable(fight.get("commence_time"))
        if not can_trade:
            logger.warning(
                "Skipping %s vs %s: %s (event_id=%s commence_time=%s)",
                fighter_a,
                fighter_b,
                start_reason,
                fight.get("event_id", ""),
                fight.get("commence_time", ""),
            )
            continue
        event_context = _resolve_live_event_context(fight, live_event_contexts)
        if event_context is None:
            logger.warning(
                "Skipping %s vs %s: %s (event_id=%s commence_time=%s)",
                fighter_a,
                fighter_b,
                _missing_live_event_context_reason(fighter_a, fighter_b),
                fight.get("event_id", ""),
                fight.get("commence_time", ""),
            )
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
        _operator_features_by_fight[f"{fighter_a}|{fighter_b}"] = features
        _operator_provenance_by_fight[f"{fighter_a}|{fighter_b}"] = {
            **(runtime_bundle_summary or {}),
            **lookup_provenance,
        }
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
        if shap_explainer is not None:
            try:
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
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "prob_a": pred["prob_a"],
            "prob_b": pred["prob_b"],
            "confidence": pred["confidence"],
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
                k: (round(float(v), 4) if isinstance(v, (int, float)) and not np.isnan(v) else None)
                for k in [
                    "a_ko_rate", "b_ko_rate", "a_sub_rate", "b_sub_rate",
                    "a_dec_rate", "b_dec_rate", "a_roll_slpm", "b_roll_slpm",
                    "a_roll_kd", "b_roll_kd", "a_roll_sub_avg", "b_roll_sub_avg",
                    "a_roll_td_avg", "b_roll_td_avg", "a_total_rounds", "b_total_rounds",
                ]
                for v in [features.get(k)]
            },
        }
        # Include line movement metadata for bet filtering
        if line_features:
            row_data["line_movement"] = line_features.get("line_movement")
            row_data["line_is_sharp"] = line_features.get("line_is_sharp")
            row_data["line_steam_move"] = line_features.get("line_steam_move")
        prediction_rows.append(row_data)
        _persist_prediction_cache(prediction_rows, announce=False)

    # Finalize the dashboard payload after the full pass completes, even when empty.
    _persist_prediction_cache(prediction_rows, announce=True)

    predictions = pd.DataFrame(prediction_rows)

    tennis_candidates = pd.DataFrame()
    if TENNIS_TRADER_ENABLED:
        try:
            logger.info("Building tennis candidates for shared-wallet portfolio...")
            tennis_candidates = _build_tennis_trade_candidates(
                model_name=DEFAULT_TENNIS_MODEL_NAME,
                min_edge=TENNIS_MIN_EDGE_THRESHOLD,
            )
            logger.info("Tennis trade-ready candidates: %s", len(tennis_candidates))
        except Exception as exc:
            logger.warning("Tennis trader build failed; skipping tennis this cycle: %s", exc)
            tennis_candidates = pd.DataFrame()

    has_ufc_portfolio = not predictions.empty and not markets.empty
    has_tennis_portfolio = not tennis_candidates.empty
    if not has_ufc_portfolio and not has_tennis_portfolio:
        logger.info("No live UFC or tennis opportunities are executable this cycle.")
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

    tennis_share = max(0.0, min(1.0, TENNIS_PORTFOLIO_SHARE)) if has_tennis_portfolio else 0.0
    if has_ufc_portfolio and has_tennis_portfolio:
        portfolio_basis = _resolve_total_bankroll(dry_run=dry_run)
        ufc_share = max(0.0, 1.0 - tennis_share)
    elif has_ufc_portfolio:
        portfolio_basis = None
        ufc_share = 1.0
    else:
        portfolio_basis = _resolve_total_bankroll(dry_run=dry_run)
        tennis_share = 1.0
        ufc_share = 0.0

    ufc_results = {"total_orders": 0}
    if has_ufc_portfolio and ufc_share > 0:
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
            bankroll_basis=(
                _slice_wallet_basis(
                    portfolio_basis.total_equity,
                    portfolio_basis.available_cash,
                    share=ufc_share,
                    label="UFC",
                    source=portfolio_basis.source,
                )
                if portfolio_basis is not None
                else None
            ),
        )
    else:
        logger.info("Skipping UFC duo traders this cycle.")

    tennis_results = {"total_orders": 0}
    if has_tennis_portfolio and tennis_share > 0:
        tennis_basis = _slice_wallet_basis(
            portfolio_basis.total_equity,
            portfolio_basis.available_cash,
            share=tennis_share,
            label="Tennis",
            source=portfolio_basis.source,
        )
        tennis_results = _run_tennis_single_trader(
            trade_candidates=tennis_candidates,
            bankroll_basis=tennis_basis,
            clob=clob,
            dry_run=dry_run,
            min_edge=TENNIS_MIN_EDGE_THRESHOLD,
        )
    else:
        logger.info("Skipping tennis trader this cycle.")

    total_orders = int(ufc_results.get("total_orders", 0)) + int(tennis_results.get("total_orders", 0))
    logger.info(
        "\nPortfolio run complete. UFC orders: %s | Tennis orders: %s | Total: %s",
        ufc_results.get("total_orders", 0),
        tennis_results.get("total_orders", 0),
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
    bt_parser.add_argument("--bankroll", type=float, default=INITIAL_BANKROLL)
    bt_parser.add_argument("--min-edge", type=float, default=MIN_EDGE_THRESHOLD)
    bt_parser.add_argument("--kelly", type=float, default=KELLY_FRACTION)
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

    # Tennis discovery command
    subparsers.add_parser("tennis-discover", help="Discover live tennis odds and Polymarket markets")

    # Tennis train command
    tennis_train_parser = subparsers.add_parser("tennis-train", help="Train a tennis probability model")
    tennis_train_parser.add_argument("--model", type=str, default=DEFAULT_TENNIS_MODEL_NAME)
    tennis_train_parser.add_argument("--start-year", type=int, default=None)
    tennis_train_parser.add_argument("--end-year", type=int, default=None)
    tennis_train_parser.add_argument("--force-download", action="store_true")
    tennis_train_parser.add_argument(
        "--refresh-player-profiles",
        action="store_true",
        help="Fetch missing official ATP/WTA player profiles before preparing tennis training data",
    )
    tennis_train_parser.add_argument(
        "--refresh-rankings-history",
        action="store_true",
        help="Fetch missing official ATP/WTA rankings history before preparing tennis training data",
    )

    tennis_profiles_parser = subparsers.add_parser(
        "tennis-player-profiles",
        help="Fetch official ATP/WTA player profiles and audit static-field coverage",
    )
    tennis_profiles_parser.add_argument("--start-year", type=int, default=None)
    tennis_profiles_parser.add_argument("--end-year", type=int, default=None)
    tennis_profiles_parser.add_argument("--force-download", action="store_true")
    tennis_profiles_parser.add_argument(
        "--all-players",
        action="store_true",
        help="Fetch profiles for every player in range instead of only players with missing static fields",
    )
    tennis_profiles_parser.add_argument(
        "--refresh-processed",
        action="store_true",
        help="Overwrite data/processed/tennis/matches.csv with player-profile-enriched rows after the audit",
    )

    tennis_refresh_parser = subparsers.add_parser(
        "tennis-refresh-daily",
        help="Refresh tennis match history, rankings, and rolling player profiles for scheduled daily runs",
    )
    tennis_refresh_parser.add_argument("--start-year", type=int, default=None)
    tennis_refresh_parser.add_argument("--end-year", type=int, default=None)
    tennis_refresh_parser.add_argument("--force-download", action="store_true")
    tennis_refresh_parser.add_argument(
        "--skip-live-seeds",
        action="store_true",
        help="Skip live odds player-name seeding after the history refresh",
    )

    tennis_lockbox_parser = subparsers.add_parser(
        "tennis-lockbox-eval",
        help="Evaluate a frozen tennis model on a strict holdout lockbox window",
    )
    tennis_lockbox_parser.add_argument("--model", type=str, default=DEFAULT_TENNIS_MODEL_NAME)
    tennis_lockbox_parser.add_argument("--lockbox-start", type=str, required=True)
    tennis_lockbox_parser.add_argument("--min-matches", type=int, default=3)

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

    tennis_rankings_parser = subparsers.add_parser(
        "tennis-rankings-history",
        help="Fetch official ATP/WTA rankings history and audit rank-field coverage",
    )
    tennis_rankings_parser.add_argument("--start-year", type=int, default=None)
    tennis_rankings_parser.add_argument("--end-year", type=int, default=None)
    tennis_rankings_parser.add_argument("--force-download", action="store_true")
    tennis_rankings_parser.add_argument(
        "--refresh-processed",
        action="store_true",
        help="Overwrite data/processed/tennis/matches.csv with rankings-history-enriched rows after the audit",
    )

    tennis_audit_parser = subparsers.add_parser(
        "tennis-bookmaker-audit",
        help="Audit historical bookmaker tennis evidence without backtesting",
    )
    tennis_audit_parser.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["auto", "odds_api", "betsapi"],
    )
    tennis_audit_parser.add_argument("--start-date", type=str, default="2022-01-01")
    tennis_audit_parser.add_argument("--end-date", type=str, default=None)

    # Tennis predict command
    tennis_predict_parser = subparsers.add_parser("tennis-predict", help="Predict live ATP/WTA singles matches")
    tennis_predict_parser.add_argument("--model", type=str, default=DEFAULT_TENNIS_MODEL_NAME)

    # Tennis live command
    tennis_live_parser = subparsers.add_parser("tennis-live", help="Run tennis dry-run discovery, prediction, and edge logging")
    tennis_live_parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Required dry-run mode; --no-dry-run is rejected because real-money tennis trading is not implemented",
    )
    tennis_live_parser.add_argument("--model", type=str, default=DEFAULT_TENNIS_MODEL_NAME)
    tennis_live_parser.add_argument("--min-edge", type=float, default=None)

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
        "tennis-discover": cmd_tennis_discover,
        "tennis-train": cmd_tennis_train,
        "tennis-player-profiles": cmd_tennis_player_profiles,
        "tennis-refresh-daily": cmd_tennis_refresh_daily,
        "tennis-lockbox-eval": cmd_tennis_lockbox_eval,
        "ufc-refresh-scheduled": cmd_ufc_refresh_scheduled,
        "tennis-rankings-history": cmd_tennis_rankings_history,
        "tennis-bookmaker-audit": cmd_tennis_bookmaker_audit,
        "tennis-predict": cmd_tennis_predict,
        "tennis-live": cmd_tennis_live,
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
