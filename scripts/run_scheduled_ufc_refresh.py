"""Run the recurring UFC active-roster refresh pipeline in one command.

This is the schedulable path for:
- syncing the official UFC active roster
- refreshing missing UFCStats-backed profile/fight rows for active fighters
- rebuilding processed UFC artifacts
- re-auditing active-roster profile completeness
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_active_roster_profile_completeness import run_audit
from scripts.backfill_active_roster_ufcstats import run_backfill, FIGHTERS_PATH as BACKFILL_FIGHTERS_PATH
from scripts.build_profile_supplement_from_external_profiles import (
    ALL_SOURCES as PROFILE_SUPPLEMENT_ALL_SOURCES,
    run_profile_supplement_refresh,
)
from scripts.rebuild_ufc_processed_artifacts import run_rebuild
from src.model.production_bundle import PRODUCTION_BUNDLE_ENV, is_hosted_runtime
from src.config import DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, PROJECT_ROOT
from src.data.name_utils import normalize_person_name
from src.data.ufc_active_roster import OFFICIAL_ACTIVE_ROSTER_PATH, sync_official_active_roster

logger = logging.getLogger(__name__)

DEFAULT_DATASET_VARIANT = "pulled_all_plus_legacy_market"
DEFAULT_REBUILD_OUTPUT_SUBDIRS = [
    ".",
    "candidates/full_live_contract_v5_fullfit_retrain",
    "candidates/v6_eval",
]
DEFAULT_AUDIT_JSON = DATA_DIR / "tmp" / "active_roster_profile_completeness_scheduled_latest.json"
DEFAULT_AUDIT_CSV = DATA_DIR / "tmp" / "active_roster_profile_completeness_scheduled_latest.csv"
DEFAULT_UNRESOLVED_PROFILE_JSON = DATA_DIR / "tmp" / "active_roster_profile_unresolved_scheduled_latest.json"
DEFAULT_UNRESOLVED_PROFILE_CSV = DATA_DIR / "tmp" / "active_roster_profile_unresolved_scheduled_latest.csv"
PROFILE_SUPPLEMENT_PATH = RAW_DATA_DIR / "ufc_fighters_profile_supplement.csv"
DEFAULT_PROFILE_SUPPLEMENT_REFRESH_SOURCES = ("martialbot", "fightdx", "espn", "tapology", "sherdog", "wikipedia")
NEW_FIGHTER_ALERT_GRACE_DAYS_ENV = "UFC_REFRESH_NEW_FIGHTER_ALERT_GRACE_DAYS"
DEFAULT_NEW_FIGHTER_ALERT_GRACE_DAYS = 7
PROFILE_REPORT_FIELDS = ("age", "weight", "height", "reach", "stance")
PROFILE_REPORT_COLUMNS = (
    "official_name",
    "field",
    "why_missing_code",
    "why_missing",
    "octagon_debut",
    "official_athlete_url",
    "ufcstats_name",
    "ufcstats_url",
    "profile_source_alias",
    "official_field_value",
    "scraped_profile_match",
    "scraped_profile_name",
    "scraped_field_value",
    "supplement_rows",
    "supplement_sources",
    "supplement_field_values",
    "source_availability",
    "model_feature_fields",
    "repair_action",
)
_SOURCE_LIMITED_ALERT_REASON_CODES = frozenset(
    {
        "official_age_blank_and_no_dob_source",
        "no_ufcstats_url_and_supplement_blank",
        "no_ufcstats_url_and_no_supplement_rows",
        "ufcstats_blank_and_supplement_blank",
        "ufcstats_blank_and_no_supplement_rows",
    }
)

# Image-bundled raw data path (used to detect stale volume copies)
_IMAGE_RAW_DIR = Path("/app/data/raw")


def _file_row_count(path: Path) -> int | None:
    """Return the CSV row count (excluding header) or None if the file doesn't exist."""
    if not path.exists():
        return None
    try:
        return int(pd.read_csv(path, usecols=[0]).shape[0])
    except Exception:
        return None


def _file_snapshot(path: Path) -> dict[str, object]:
    """Return diagnostic metadata about a file."""
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "exists": True,
            "size_bytes": int(stat.st_size),
            "row_count": _file_row_count(path) if path.suffix.lower() == ".csv" else None,
            "mtime_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        }
    except OSError:
        return {"path": str(path), "exists": True}


def _log_resolved_data_paths() -> dict[str, str]:
    """Log and return all resolved data paths used by the refresh pipeline."""
    paths = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DATA_DIR": str(DATA_DIR),
        "RAW_DATA_DIR": str(RAW_DATA_DIR),
        "PROCESSED_DATA_DIR": str(PROCESSED_DATA_DIR),
        "OFFICIAL_ACTIVE_ROSTER_PATH": str(OFFICIAL_ACTIVE_ROSTER_PATH),
        "BACKFILL_FIGHTERS_PATH": str(BACKFILL_FIGHTERS_PATH),
        "UFC_DATA_DIR_env": os.environ.get("UFC_DATA_DIR", "(unset)"),
        "RAILWAY_VOLUME_MOUNT_PATH_env": os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "(unset)"),
        "is_hosted_runtime": str(is_hosted_runtime()),
    }
    logger.info(
        "UFC refresh resolved data paths: %s",
        json.dumps(paths, indent=2),
    )
    return paths


def _row_drop_guard_files() -> dict[str, Path]:
    return {
        "ufc_fighters_scraped": RAW_DATA_DIR / "ufc_fighters_scraped.csv",
        "ufc_active_roster_official": OFFICIAL_ACTIVE_ROSTER_PATH,
        "ufc_fight_results": RAW_DATA_DIR / "ufc-fight-results.csv",
        "ufc_fight_stats": RAW_DATA_DIR / "ufc-fight-stats.csv",
        "processed_fights_cleaned": PROCESSED_DATA_DIR / "fights_cleaned.csv",
        "processed_features": PROCESSED_DATA_DIR / "features.csv",
        "processed_test_set": PROCESSED_DATA_DIR / "test_set.csv",
    }


def _row_guard_snapshot() -> dict[str, dict[str, object]]:
    return {name: _file_snapshot(path) for name, path in _row_drop_guard_files().items()}


def _build_row_drop_guard(
    pre_state: dict[str, dict[str, object]],
    post_state: dict[str, dict[str, object]],
) -> dict[str, object]:
    violations: list[dict[str, object]] = []
    for name, pre in pre_state.items():
        post = post_state.get(name) or {}
        pre_rows = pre.get("row_count")
        post_rows = post.get("row_count")
        if pre_rows is None or post_rows is None:
            continue
        try:
            pre_count = int(pre_rows)
            post_count = int(post_rows)
        except Exception:
            continue
        if post_count < pre_count:
            violations.append(
                {
                    "artifact": name,
                    "path": post.get("path") or pre.get("path") or str(_row_drop_guard_files().get(name, "")),
                    "pre_rows": pre_count,
                    "post_rows": post_count,
                    "rows_lost": int(pre_count - post_count),
                }
            )

    return {
        "ok": not violations,
        "checked_artifacts": len(pre_state),
        "violations": violations,
    }


def _seed_stale_scraped_fighters() -> dict[str, object]:
    """If running on hosted and the image has a richer scraped-fighters file, merge it in.

    The entrypoint's copy_tree_missing only seeds files that don't exist on the volume,
    so subsequent deploys with locally-enriched data never update the volume copy.
    This function detects that scenario and merges the image data.
    """
    image_path = _IMAGE_RAW_DIR / "ufc_fighters_scraped.csv"
    runtime_path = BACKFILL_FIGHTERS_PATH

    if not image_path.exists() or image_path == runtime_path or image_path.resolve() == runtime_path.resolve():
        return {"action": "skip", "reason": "no image/runtime divergence"}

    if not runtime_path.exists():
        return {"action": "skip", "reason": "runtime file missing (entrypoint should have seeded it)"}

    try:
        image_df = pd.read_csv(image_path)
        runtime_df = pd.read_csv(runtime_path)
    except Exception as exc:
        logger.warning("Failed to compare image vs runtime scraped fighters: %s", exc)
        return {"action": "error", "reason": str(exc)}

    image_rows = len(image_df)
    runtime_rows = len(runtime_df)

    # Count non-blank stance values as the key enrichment signal
    def _stance_count(df: pd.DataFrame) -> int:
        if "stance" not in df.columns:
            return 0
        return int(df["stance"].fillna("").astype(str).str.strip().ne("").sum())

    image_stance = _stance_count(image_df)
    runtime_stance = _stance_count(runtime_df)

    if image_stance <= runtime_stance and image_rows <= runtime_rows:
        logger.info(
            "Scraped fighters volume copy is current: runtime=%d rows/%d stance, image=%d rows/%d stance",
            runtime_rows, runtime_stance, image_rows, image_stance,
        )
        return {
            "action": "skip",
            "reason": "runtime copy is at least as rich as image",
            "runtime_rows": runtime_rows,
            "runtime_stance": runtime_stance,
            "image_rows": image_rows,
            "image_stance": image_stance,
        }

    # Merge: take runtime as base, fill in any new rows or enriched fields from image
    from src.data.io_utils import write_csv_atomically

    runtime_urls = set(
        runtime_df.get("fighter_url", pd.Series(dtype="object"))
        .fillna("").astype(str).str.strip()
    ) - {""}
    image_new = image_df[
        ~image_df.get("fighter_url", pd.Series(dtype="object"))
        .fillna("").astype(str).str.strip()
        .isin(runtime_urls)
    ]
    merged = pd.concat([runtime_df, image_new], ignore_index=True, sort=False)
    merged = merged.drop_duplicates(subset=["fighter_url"], keep="first")

    # For existing rows, fill blank stance/reach/height from image where available
    image_by_url = {}
    for _, row in image_df.iterrows():
        url = str(row.get("fighter_url") or "").strip()
        if url:
            image_by_url[url] = row

    updated_fields = 0
    for idx, row in merged.iterrows():
        url = str(row.get("fighter_url") or "").strip()
        image_row = image_by_url.get(url)
        if image_row is None:
            continue
        for field in ("stance", "reach", "height", "weight", "dob"):
            current = str(row.get(field) or "").strip()
            if current and current not in ("", "--", "nan", "NaN"):
                continue
            image_val = str(image_row.get(field) or "").strip()
            if image_val and image_val not in ("", "--", "nan", "NaN"):
                merged.at[idx, field] = image_val
                updated_fields += 1

    write_csv_atomically(merged, runtime_path, refuse_empty=True)
    merged_stance = _stance_count(merged)

    logger.info(
        "Seeded stale scraped fighters from image: runtime %d→%d rows, stance %d→%d, updated %d fields, added %d new rows",
        runtime_rows, len(merged), runtime_stance, merged_stance, updated_fields, len(image_new),
    )
    return {
        "action": "merged",
        "runtime_rows_before": runtime_rows,
        "runtime_rows_after": len(merged),
        "runtime_stance_before": runtime_stance,
        "runtime_stance_after": merged_stance,
        "new_rows_from_image": len(image_new),
        "updated_fields": updated_fields,
    }


def _roster_summary(df: pd.DataFrame, *, output_path: Path) -> dict[str, object]:
    attrs = dict(getattr(df, "attrs", {}) or {})
    summary = {
        "rows": int(len(df)),
        "resolved_ufcstats_urls": int(df.get("ufcstats_url", pd.Series(dtype="object")).fillna("").astype(str).str.strip().ne("").sum()),
        "resolved_ufcstats_names": int(df.get("ufcstats_name", pd.Series(dtype="object")).fillna("").astype(str).str.strip().ne("").sum()),
        "with_profile_details": int(df.get("profile_status", pd.Series(dtype="object")).fillna("").astype(str).str.strip().ne("").sum()),
        "output_path": str(output_path),
    }
    sync_source = str(attrs.get("sync_source") or "live").strip() or "live"
    summary["source"] = sync_source
    summary["used_cached_fallback"] = bool(attrs.get("sync_fallback_used"))
    sync_error = str(attrs.get("sync_error") or "").strip()
    if sync_error:
        summary["sync_error"] = sync_error
    cached_snapshot_mtime = str(attrs.get("sync_cached_snapshot_mtime_utc") or "").strip()
    if cached_snapshot_mtime:
        summary["cached_snapshot_mtime_utc"] = cached_snapshot_mtime
    identity_audit_rows = attrs.get("identity_audit_rows")
    if isinstance(identity_audit_rows, list):
        summary["identity_audit_rows"] = int(len(identity_audit_rows))
        action_counts: dict[str, int] = {}
        for row in identity_audit_rows:
            if not isinstance(row, dict):
                continue
            action = str(row.get("action") or "unknown").strip() or "unknown"
            action_counts[action] = action_counts.get(action, 0) + 1
        summary["identity_audit_action_counts"] = action_counts
    return summary


def _write_json(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _summarize_audit_frame(frame: pd.DataFrame) -> dict[str, object]:
    total = int(len(frame))
    if total == 0:
        return {"rows": 0}

    summary: dict[str, object] = {"rows": total}
    for field in (
        "age_present",
        "weight_present",
        "division_present",
        "height_present",
        "reach_present",
        "stance_present",
        "full_physical_bundle_present",
    ):
        if field not in frame.columns:
            continue
        count = int(frame[field].sum())
        summary[field] = {
            "count": count,
            "pct": round((count / total) * 100.0, 2),
        }
    return summary


def _blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() in {"", "--", "nan", "NaN", "N/A"}


def _string_value(value: object) -> str:
    return "" if _blank(value) else str(value).strip()


def _official_url_identity_trusted(row: dict[str, object]) -> bool:
    status = str(row.get("official_url_identity_status") or "").strip().lower()
    if status in {"mismatch", "test_profile"}:
        return False
    explicit = row.get("official_url_identity_valid")
    if _blank(explicit):
        return True
    return str(explicit).strip().lower() not in {"0", "false", "no", "off"}


def _new_fighter_split_column(frame: pd.DataFrame) -> str:
    if "split_alias_aware" in frame.columns:
        return "split_alias_aware"
    return "split_official_name"


def _report_alias_keys(row: dict[str, object]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    alias_fields = ["official_name", "ufcstats_name"]
    if _official_url_identity_trusted(row):
        alias_fields.extend(["profile_name", "slug_name"])
    for field in alias_fields:
        value = _string_value(row.get(field))
        key = normalize_person_name(value)
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    if _official_url_identity_trusted(row):
        for value in str(row.get("alternate_slug_names") or "").split("|"):
            value = _string_value(value)
            key = normalize_person_name(value)
            if not key or key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return keys


def _load_scraped_profile_indexes(path: Path) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]]]:
    if not path.exists():
        return {}, {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}, {}
    rows = frame.to_dict(orient="records")
    by_url: dict[str, dict[str, object]] = {}
    by_name: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        url = _string_value(row.get("fighter_url"))
        if url and url not in by_url:
            by_url[url] = row
        key = normalize_person_name(_string_value(row.get("name")))
        if not key:
            continue
        by_name.setdefault(key, []).append(row)
    return by_url, by_name


def _load_named_row_index(path: Path, *, name_column: str) -> dict[str, list[dict[str, object]]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty or name_column not in frame.columns:
        return {}
    rows = frame.to_dict(orient="records")
    by_name: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        key = normalize_person_name(_string_value(row.get(name_column)))
        if not key:
            continue
        by_name.setdefault(key, []).append(row)
    return by_name


def _match_scraped_profile(
    active_row: dict[str, object],
    *,
    scraped_by_url: dict[str, dict[str, object]],
    scraped_by_name: dict[str, list[dict[str, object]]],
) -> tuple[str, dict[str, object] | None]:
    ufcstats_url = _string_value(active_row.get("ufcstats_url"))
    if ufcstats_url:
        row = scraped_by_url.get(ufcstats_url)
        if row is not None:
            return "ufcstats_url", row
    for alias_key in _report_alias_keys(active_row):
        rows = scraped_by_name.get(alias_key) or []
        if rows:
            return "name_alias", rows[0]
    return "", None


def _matched_supplement_rows(
    active_row: dict[str, object],
    *,
    supplement_by_name: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    matched: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for alias_key in _report_alias_keys(active_row):
        for row in supplement_by_name.get(alias_key, []):
            signature = (
                _string_value(row.get("name")),
                _string_value(row.get("source")),
                _string_value(row.get("fighter_url")),
            )
            if signature in seen:
                continue
            seen.add(signature)
            matched.append(row)
    return matched


def _field_support_column(field: str) -> str:
    return "dob" if field == "age" else field


def _official_report_value(active_row: dict[str, object], field: str) -> object:
    if field == "stance":
        return ""
    return active_row.get(field)


def _first_nonblank_string(value: object) -> str:
    return "" if _blank(value) else str(value).strip()


def _unique_nonblank_values(rows: list[dict[str, object]], field: str) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        text = _first_nonblank_string(row.get(field))
        if not text or text in seen:
            continue
        seen.add(text)
        values.append(text)
    return " | ".join(values)


def _missing_field_reason(
    *,
    field: str,
    active_row: dict[str, object],
    scraped_profile: dict[str, object] | None,
    supplement_rows: list[dict[str, object]],
) -> tuple[str, str]:
    support_column = _field_support_column(field)
    official_value = _official_report_value(active_row, field)
    official_has_value = not _blank(official_value)
    scraped_has_value = scraped_profile is not None and not _blank(scraped_profile.get(support_column))
    supplement_has_rows = bool(supplement_rows)
    supplement_has_value = any(not _blank(row.get(support_column)) for row in supplement_rows)
    has_ufcstats_url = bool(_string_value(active_row.get("ufcstats_url")))

    if field == "age":
        if official_has_value:
            return (
                "official_age_present_but_audit_missed",
                "Official UFC roster age is already present, so the audit missed an available value.",
            )
        if scraped_has_value or supplement_has_value:
            return (
                "official_age_blank_but_dob_available_elsewhere",
                (
                    "Official UFC roster age is blank. A date of birth exists in UFCStats or supplement data, "
                    "but the current audit only counts official-roster age."
                ),
            )
        return (
            "official_age_blank_and_no_dob_source",
            "Official UFC roster age is blank and no date of birth was recovered from UFCStats or supplement sources.",
        )

    if official_has_value:
        return (
            "official_value_present_but_audit_lookup_missed",
            f"An official UFC roster {field} value exists, so the lookup missed an available value.",
        )
    if scraped_has_value or supplement_has_value:
        return (
            "source_value_present_but_audit_lookup_missed",
            f"A {field} value exists in UFCStats or supplement data, so the lookup missed an available value.",
        )
    if not has_ufcstats_url:
        if supplement_has_rows:
            detail = (
                "No UFCStats profile is resolved. Supplement rows exist, but they are blank for this field."
            )
            if field == "stance":
                detail = (
                    "No UFCStats profile is resolved. Supplement rows exist, but they are blank for stance. "
                    "Current automated sources are still source-limited for stance on these new fighters."
                )
            return ("no_ufcstats_url_and_supplement_blank", detail)
        detail = "No UFCStats profile is resolved and no supplement row was recovered for this field."
        if field == "stance":
            detail = (
                "No UFCStats profile is resolved and no supplement row was recovered for stance. "
                "Current automated sources are still source-limited for stance on these new fighters."
            )
        return ("no_ufcstats_url_and_no_supplement_rows", detail)
    if scraped_profile is None:
        if supplement_has_rows:
            return (
                "ufcstats_url_present_but_scraped_profile_missing_and_supplement_blank",
                "A UFCStats URL is resolved, but no scraped UFCStats profile row was available after refresh, and supplement rows are blank for this field.",
            )
        return (
            "ufcstats_url_present_but_scraped_profile_missing",
            "A UFCStats URL is resolved, but no scraped UFCStats profile row was available after refresh.",
        )
    if supplement_has_rows:
        detail = "UFCStats is blank for this field, and supplement rows are also blank."
        if field == "stance":
            detail = (
                "UFCStats is blank for stance, and supplement rows are also blank. "
                "Current automated sources are still source-limited for stance on these new fighters."
            )
        return ("ufcstats_blank_and_supplement_blank", detail)
    detail = "UFCStats is blank for this field, and no supplement row was recovered."
    if field == "stance":
        detail = (
            "UFCStats is blank for stance, and no supplement row was recovered. "
            "Current automated sources are still source-limited for stance on these new fighters."
        )
    return ("ufcstats_blank_and_no_supplement_rows", detail)


_MODEL_FEATURE_FIELDS_BY_PROFILE_FIELD = {
    "age": "a_age | b_age | diff_age",
    "height": "a_height | b_height | diff_height",
    "reach": "a_reach | b_reach | diff_reach",
    "weight": "a_weight | b_weight | diff_weight",
    "stance": "a_stance_enc | b_stance_enc | same_stance",
}


def _source_availability_for_reason(reason_code: str) -> str:
    if reason_code in {
        "official_age_present_but_audit_missed",
        "official_value_present_but_audit_lookup_missed",
        "source_value_present_but_audit_lookup_missed",
        "official_age_blank_but_dob_available_elsewhere",
    }:
        return "available_but_pipeline_missed"
    if reason_code.startswith("no_ufcstats_url"):
        return "no_ufcstats_profile_resolved"
    if reason_code.startswith("ufcstats_url_present_but_scraped_profile_missing"):
        return "ufcstats_profile_scrape_missing"
    if reason_code == "official_age_blank_and_no_dob_source":
        return "source_limited_blank"
    if reason_code.startswith("ufcstats_blank"):
        return "source_limited_blank"
    return "unknown"


def _repair_action_for_reason(reason_code: str) -> str:
    if reason_code in {
        "official_age_present_but_audit_missed",
        "official_value_present_but_audit_lookup_missed",
        "source_value_present_but_audit_lookup_missed",
        "official_age_blank_but_dob_available_elsewhere",
    }:
        return "repair audit/profile merge lookup"
    if reason_code.startswith("no_ufcstats_url"):
        return "resolve UFCStats URL or recover trusted external profile fields"
    if reason_code.startswith("ufcstats_url_present_but_scraped_profile_missing"):
        return "retry UFCStats profile scrape"
    if reason_code.startswith("ufcstats_blank"):
        return "recover trusted external profile field if available"
    return "review manually"


def _build_profile_repair_queue(report_df: pd.DataFrame) -> list[dict[str, object]]:
    if report_df.empty:
        return []

    queue: list[dict[str, object]] = []
    for official_name, group in report_df.groupby("official_name", sort=True):
        missing_fields = sorted(str(field) for field in group["field"].dropna().astype(str).unique())
        source_availability = sorted(
            str(value) for value in group["source_availability"].dropna().astype(str).unique()
        )
        reason_codes = sorted(str(value) for value in group["why_missing_code"].dropna().astype(str).unique())
        model_features = sorted(
            {
                token.strip()
                for value in group["model_feature_fields"].dropna().astype(str)
                for token in value.split("|")
                if token.strip()
            }
        )
        queue.append(
            {
                "official_name": official_name,
                "missing_fields": missing_fields,
                "missing_field_count": int(len(group)),
                "source_availability": source_availability,
                "reason_codes": reason_codes,
                "model_feature_fields": model_features,
                "source_limited_only": all(
                    code in _SOURCE_LIMITED_ALERT_REASON_CODES for code in reason_codes
                ),
                "repair_actions": sorted(
                    str(value) for value in group["repair_action"].dropna().astype(str).unique()
                ),
            }
        )
    queue.sort(key=lambda row: (-int(row["missing_field_count"]), str(row["official_name"])))
    return queue


def _build_unresolved_profile_report(
    *,
    active_roster_path: Path,
    scraped_fighters_path: Path,
    audit_df: pd.DataFrame,
    supplement_path: Path = PROFILE_SUPPLEMENT_PATH,
) -> tuple[dict[str, object], pd.DataFrame]:
    report_df = pd.DataFrame(columns=PROFILE_REPORT_COLUMNS)
    if (
        audit_df.empty
        or "official_name" not in audit_df.columns
        or "split_official_name" not in audit_df.columns
    ):
        return {"rows": 0, "fighters": 0, "fields": {}, "reasons": {}, "reasons_by_field": {}}, report_df

    if not active_roster_path.exists():
        return {"rows": 0, "fighters": 0, "fields": {}, "reasons": {}, "reasons_by_field": {}}, report_df

    active_df = pd.read_csv(active_roster_path)
    if active_df.empty or "official_name" not in active_df.columns:
        return {"rows": 0, "fighters": 0, "fields": {}, "reasons": {}, "reasons_by_field": {}}, report_df

    active_rows = (
        active_df.assign(
            _official_name_key=active_df["official_name"].fillna("").astype(str).map(normalize_person_name)
        )
        .sort_values("official_name")
        .drop_duplicates(subset=["_official_name_key"], keep="first")
        .set_index("_official_name_key")
        .to_dict(orient="index")
    )
    scraped_by_url, scraped_by_name = _load_scraped_profile_indexes(scraped_fighters_path)
    supplement_by_name = _load_named_row_index(supplement_path, name_column="name")

    report_rows: list[dict[str, object]] = []
    split_column = _new_fighter_split_column(audit_df)
    tracked = audit_df[audit_df[split_column].fillna("").eq("newly_added_active_roster")].copy()
    if "coverage_eligible" in tracked.columns:
        tracked = tracked[tracked["coverage_eligible"].fillna(True)].copy()
    for _, audit_row in tracked.iterrows():
        official_name = str(audit_row.get("official_name") or "").strip()
        if not official_name:
            continue
        active_row = active_rows.get(normalize_person_name(official_name))
        if active_row is None:
            continue
        scraped_match, scraped_profile = _match_scraped_profile(
            active_row,
            scraped_by_url=scraped_by_url,
            scraped_by_name=scraped_by_name,
        )
        supplement_rows = _matched_supplement_rows(active_row, supplement_by_name=supplement_by_name)
        supplement_sources = sorted(
            {
                _string_value(row.get("source"))
                for row in supplement_rows
                if _string_value(row.get("source"))
            }
        )

        for field in PROFILE_REPORT_FIELDS:
            audit_column = f"{field}_present"
            if audit_column not in audit_row or bool(audit_row.get(audit_column)):
                continue
            reason_code, reason = _missing_field_reason(
                field=field,
                active_row=active_row,
                scraped_profile=scraped_profile,
                supplement_rows=supplement_rows,
            )
            source_availability = _source_availability_for_reason(reason_code)
            report_rows.append(
                {
                    "official_name": official_name,
                    "field": field,
                    "why_missing_code": reason_code,
                    "why_missing": reason,
                    "octagon_debut": _string_value(active_row.get("octagon_debut")),
                    "official_athlete_url": _string_value(active_row.get("official_athlete_url")),
                    "ufcstats_name": _string_value(active_row.get("ufcstats_name")),
                    "ufcstats_url": _string_value(active_row.get("ufcstats_url")),
                    "profile_source_alias": _string_value(audit_row.get("profile_source_alias")),
                    "official_field_value": _string_value(_official_report_value(active_row, field)),
                    "scraped_profile_match": scraped_match,
                    "scraped_profile_name": _string_value((scraped_profile or {}).get("name")),
                    "scraped_field_value": _string_value((scraped_profile or {}).get(_field_support_column(field))),
                    "supplement_rows": int(len(supplement_rows)),
                    "supplement_sources": " | ".join(supplement_sources),
                    "supplement_field_values": _unique_nonblank_values(supplement_rows, _field_support_column(field)),
                    "source_availability": source_availability,
                    "model_feature_fields": _MODEL_FEATURE_FIELDS_BY_PROFILE_FIELD.get(field, ""),
                    "repair_action": _repair_action_for_reason(reason_code),
                }
            )

    if report_rows:
        report_df = pd.DataFrame(report_rows, columns=PROFILE_REPORT_COLUMNS)
        report_df = report_df.sort_values(["official_name", "field"]).reset_index(drop=True)

    summary = {
        "rows": int(len(report_df)),
        "fighters": int(report_df["official_name"].nunique()) if not report_df.empty else 0,
        "fields": report_df["field"].value_counts().sort_index().to_dict() if not report_df.empty else {},
        "reasons": report_df["why_missing_code"].value_counts().sort_index().to_dict() if not report_df.empty else {},
        "reasons_by_field": (
            {
                field: group["why_missing_code"].value_counts().sort_index().to_dict()
                for field, group in report_df.groupby("field", sort=True)
            }
            if not report_df.empty
            else {}
        ),
        "source_availability": (
            report_df["source_availability"].value_counts().sort_index().to_dict()
            if not report_df.empty
            else {}
        ),
        "repair_queue": _build_profile_repair_queue(report_df),
    }
    return summary, report_df


def _profile_supplement_refresh_enabled() -> bool:
    raw = str(os.getenv("UFC_REFRESH_PROFILE_SUPPLEMENT_ENABLED", "1") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _profile_supplement_refresh_limit() -> int | None:
    raw = str(os.getenv("UFC_REFRESH_PROFILE_SUPPLEMENT_LIMIT", "") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None


def _profile_supplement_refresh_sources() -> list[str]:
    raw = str(os.getenv("UFC_REFRESH_PROFILE_SUPPLEMENT_SOURCES", "") or "").strip()
    if not raw:
        return list(DEFAULT_PROFILE_SUPPLEMENT_REFRESH_SOURCES)
    selected: list[str] = []
    for token in raw.split(","):
        source = token.strip().lower()
        if source in PROFILE_SUPPLEMENT_ALL_SOURCES and source not in selected:
            selected.append(source)
    return selected or list(DEFAULT_PROFILE_SUPPLEMENT_REFRESH_SOURCES)


def _build_profile_gap_candidate_frame(
    *,
    active_roster_path: Path,
    audit_df: pd.DataFrame,
) -> pd.DataFrame:
    if audit_df.empty or "official_name" not in audit_df.columns:
        return pd.DataFrame()

    split_column = _new_fighter_split_column(audit_df)
    target_names = (
        audit_df.loc[
            audit_df[split_column].fillna("").eq("newly_added_active_roster")
            & (audit_df.get("coverage_eligible", pd.Series(True, index=audit_df.index)).fillna(True))
            & (~audit_df["full_physical_bundle_present"].fillna(False)),
            "official_name",
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )
    target_name_set = {name for name in target_names if name}
    if not target_name_set:
        return pd.DataFrame()

    active_df = pd.read_csv(active_roster_path)
    if active_df.empty or "official_name" not in active_df.columns:
        return pd.DataFrame()
    candidate_df = (
        active_df[
            active_df["official_name"].fillna("").astype(str).str.strip().isin(target_name_set)
        ]
        .copy()
        .sort_values("official_name")
        .reset_index(drop=True)
    )
    return candidate_df


def _new_fighter_alert_grace_days() -> int:
    raw = str(os.getenv(NEW_FIGHTER_ALERT_GRACE_DAYS_ENV, DEFAULT_NEW_FIGHTER_ALERT_GRACE_DAYS) or "").strip()
    try:
        value = int(raw)
    except Exception:
        return DEFAULT_NEW_FIGHTER_ALERT_GRACE_DAYS
    return max(value, 0)


def _build_profile_audit_alert_summary(
    *,
    active_roster_path: Path,
    audit_df: pd.DataFrame,
    as_of_utc: datetime | None = None,
    new_fighter_grace_days: int | None = None,
    unresolved_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    grace_days = (
        _new_fighter_alert_grace_days()
        if new_fighter_grace_days is None
        else max(int(new_fighter_grace_days), 0)
    )
    as_of = as_of_utc or datetime.now(timezone.utc)
    base_summary = {
        "as_of_date_utc": as_of.date().isoformat(),
        "new_fighter_grace_days": grace_days,
        "identity_match_method": "alias_aware",
        "newly_added_active_roster": {
            "rows_total": 0,
            "rows_alert_eligible": 0,
            "rows_in_grace": 0,
        },
        "source_limited_missing_fields": _build_new_fighter_source_limited_alert_summary(
            alert_eligible=pd.DataFrame(),
            unresolved_df=unresolved_df,
        ),
        "split_summary_official_name": {
            "newly_added_active_roster": {"rows": 0},
        },
        "split_summary_alias_aware": {
            "newly_added_active_roster": {"rows": 0},
        },
    }
    split_column = _new_fighter_split_column(audit_df)
    if audit_df.empty or "official_name" not in audit_df.columns or split_column not in audit_df.columns:
        return base_summary

    if active_roster_path.exists():
        active_df = pd.read_csv(active_roster_path).copy()
    else:
        active_df = pd.DataFrame(columns=["official_name", "octagon_debut"])

    active_df["_official_name_key"] = active_df.get("official_name", pd.Series(dtype="object")).fillna("").astype(str).map(normalize_person_name)
    if "octagon_debut" in active_df.columns:
        active_df["_octagon_debut_dt"] = pd.to_datetime(active_df["octagon_debut"], errors="coerce", format="mixed")
    else:
        active_df["_octagon_debut_dt"] = pd.NaT

    active_dates = active_df[["_official_name_key", "_octagon_debut_dt"]]
    if not active_dates.empty:
        active_dates = (
            active_dates
            .sort_values("_official_name_key")
            .drop_duplicates(subset=["_official_name_key"], keep="first")
        )

    merged = audit_df.copy()
    merged["_official_name_key"] = merged["official_name"].fillna("").astype(str).map(normalize_person_name)
    merged = merged.merge(active_dates, on="_official_name_key", how="left")

    eligible_mask = merged.get("coverage_eligible", pd.Series(True, index=merged.index)).fillna(True)
    new_fighters = merged[
        merged[split_column].fillna("").eq("newly_added_active_roster")
        & eligible_mask
    ].copy()
    if new_fighters.empty:
        return base_summary

    cutoff = pd.Timestamp(as_of.date()) - pd.Timedelta(days=grace_days)
    alert_eligible = new_fighters[
        new_fighters["_octagon_debut_dt"].isna()
        | (new_fighters["_octagon_debut_dt"] <= cutoff)
    ].copy()
    source_limited_missing_fields = _build_new_fighter_source_limited_alert_summary(
        alert_eligible=alert_eligible,
        unresolved_df=unresolved_df,
    )

    split_summaries: dict[str, dict[str, dict[str, object]]] = {}
    new_fighter_counts_by_method: dict[str, dict[str, int]] = {}
    for candidate_split_column, method_name in (
        ("split_official_name", "official_name"),
        ("split_alias_aware", "alias_aware"),
    ):
        if candidate_split_column not in merged.columns:
            continue
        method_new = merged[
            merged[candidate_split_column].fillna("").eq("newly_added_active_roster")
            & eligible_mask
        ].copy()
        method_alert_eligible = method_new[
            method_new["_octagon_debut_dt"].isna()
            | (method_new["_octagon_debut_dt"] <= cutoff)
        ].copy()
        split_summaries[candidate_split_column] = {
            "newly_added_active_roster": _summarize_audit_frame(method_alert_eligible),
        }
        new_fighter_counts_by_method[method_name] = {
            "rows_total": int(len(method_new)),
            "rows_alert_eligible": int(len(method_alert_eligible)),
            "rows_in_grace": int(len(method_new) - len(method_alert_eligible)),
        }

    return {
        "as_of_date_utc": as_of.date().isoformat(),
        "new_fighter_grace_days": grace_days,
        "identity_match_method": "alias_aware" if split_column == "split_alias_aware" else "official_name",
        "new_fighter_counts_by_method": new_fighter_counts_by_method,
        "newly_added_active_roster": {
            "rows_total": int(len(new_fighters)),
            "rows_alert_eligible": int(len(alert_eligible)),
            "rows_in_grace": int(len(new_fighters) - len(alert_eligible)),
        },
        "source_limited_missing_fields": source_limited_missing_fields,
        "split_summary_official_name": {
            "newly_added_active_roster": (
                split_summaries.get("split_official_name", {})
                .get("newly_added_active_roster", {"rows": 0})
            ),
        },
        "split_summary_alias_aware": {
            "newly_added_active_roster": (
                split_summaries.get("split_alias_aware", {})
                .get("newly_added_active_roster", _summarize_audit_frame(alert_eligible))
            ),
        },
    }


def _build_new_fighter_source_limited_alert_summary(
    *,
    alert_eligible: pd.DataFrame,
    unresolved_df: pd.DataFrame | None,
) -> dict[str, dict[str, object]]:
    field_specs = {
        "reach": {
            "audit_column": "reach_present",
            "required_fields": ("reach",),
        },
        "stance": {
            "audit_column": "stance_present",
            "required_fields": ("stance",),
        },
        "full_physical_bundle": {
            "audit_column": "full_physical_bundle_present",
            "required_fields": ("age", "weight", "height", "reach", "stance"),
        },
    }
    base_summary = {
        field: {
            "rows_missing": 0,
            "rows_source_limited": 0,
            "source_limited_only": False,
        }
        for field in field_specs
    }
    if (
        alert_eligible.empty
        or unresolved_df is None
        or unresolved_df.empty
        or "official_name" not in alert_eligible.columns
        or "official_name" not in unresolved_df.columns
        or "field" not in unresolved_df.columns
        or "why_missing_code" not in unresolved_df.columns
    ):
        return base_summary

    unresolved_reason_codes: dict[tuple[str, str], set[str]] = {}
    eligible_name_keys = {
        normalize_person_name(name)
        for name in alert_eligible["official_name"].fillna("").astype(str)
        if normalize_person_name(name)
    }
    if not eligible_name_keys:
        return base_summary

    for row in unresolved_df.to_dict(orient="records"):
        key = normalize_person_name(row.get("official_name"))
        field = _string_value(row.get("field"))
        reason_code = _string_value(row.get("why_missing_code"))
        if not key or key not in eligible_name_keys or not field or not reason_code:
            continue
        unresolved_reason_codes.setdefault((key, field), set()).add(reason_code)

    if not unresolved_reason_codes:
        return base_summary

    def _row_missing_fields(row: pd.Series, summary_field: str, required_fields: tuple[str, ...]) -> list[str]:
        if summary_field != "full_physical_bundle":
            return list(required_fields)
        missing_fields: list[str] = []
        for field in required_fields:
            audit_column = f"{field}_present"
            present = row.get(audit_column)
            if pd.isna(present) or not bool(present):
                missing_fields.append(field)
        return missing_fields

    for summary_field, spec in field_specs.items():
        missing_rows = alert_eligible[~alert_eligible[spec["audit_column"]].fillna(False)].copy()
        if missing_rows.empty:
            continue
        rows_missing = int(len(missing_rows))
        rows_source_limited = 0
        for _, row in missing_rows.iterrows():
            key = normalize_person_name(row.get("official_name"))
            missing_fields = _row_missing_fields(row, summary_field, spec["required_fields"])
            if not key or not missing_fields:
                continue
            if all(
                (reason_codes := unresolved_reason_codes.get((key, field)))
                and reason_codes.issubset(_SOURCE_LIMITED_ALERT_REASON_CODES)
                for field in missing_fields
            ):
                rows_source_limited += 1

        base_summary[summary_field] = {
            "rows_missing": rows_missing,
            "rows_source_limited": int(rows_source_limited),
            "source_limited_only": bool(rows_missing > 0 and rows_source_limited == rows_missing),
        }

    return base_summary


def _maybe_refresh_profile_supplement(
    *,
    active_roster_path: Path,
    scraped_fighters_path: Path,
    audit_df: pd.DataFrame,
    partial_refresh: bool,
) -> dict[str, object] | None:
    if partial_refresh:
        return {"action": "skip", "reason": "partial refresh"}
    if not _profile_supplement_refresh_enabled():
        return {"action": "skip", "reason": "disabled by UFC_REFRESH_PROFILE_SUPPLEMENT_ENABLED"}

    candidate_df = _build_profile_gap_candidate_frame(
        active_roster_path=active_roster_path,
        audit_df=audit_df,
    )
    if candidate_df.empty:
        return {"action": "skip", "reason": "no new active-roster profile gaps"}

    limit = _profile_supplement_refresh_limit()
    sources = _profile_supplement_refresh_sources()
    with TemporaryDirectory(prefix="ufc_profile_gap_candidates_") as tmp_dir:
        candidate_source_path = Path(tmp_dir) / "active_roster_profile_gap_candidates.csv"
        candidate_df.to_csv(candidate_source_path, index=False)
        summary = run_profile_supplement_refresh(
            scraped_fighters_path=scraped_fighters_path,
            candidate_source_csv=candidate_source_path,
            output_path=PROFILE_SUPPLEMENT_PATH,
            sources=sources,
            limit=limit,
        )
    return {
        "action": "completed",
        "target_rows": int(len(candidate_df)),
        "limit": limit,
        "selected_sources": list(sources),
        **summary,
    }


def run_scheduled_refresh(
    *,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    output_subdirs: list[str] | None = None,
    limit_fighters: int | None = None,
    audit_json_path: Path | None = DEFAULT_AUDIT_JSON,
    audit_csv_path: Path | None = DEFAULT_AUDIT_CSV,
    unresolved_json_path: Path | None = DEFAULT_UNRESOLVED_PROFILE_JSON,
    unresolved_csv_path: Path | None = DEFAULT_UNRESOLVED_PROFILE_CSV,
    skip_rebuild: bool = False,
    skip_audit: bool = False,
) -> dict[str, object]:
    # --- diagnostic: log all resolved paths and pre-refresh file state ---
    resolved_paths = _log_resolved_data_paths()
    scraped_fighters_path = RAW_DATA_DIR / "ufc_fighters_scraped.csv"
    processed_fights_path = PROCESSED_DATA_DIR / "fights_cleaned.csv"

    pre_refresh_state = {
        "active_roster": _file_snapshot(OFFICIAL_ACTIVE_ROSTER_PATH),
        "scraped_fighters": _file_snapshot(scraped_fighters_path),
        "processed_fights": _file_snapshot(processed_fights_path),
    }
    pre_refresh_row_guard_state = _row_guard_snapshot()
    logger.info(
        "UFC refresh pre-state: roster=%s scraped_fighters=%s processed_fights=%s",
        pre_refresh_state["active_roster"],
        pre_refresh_state["scraped_fighters"],
        pre_refresh_state["processed_fights"],
    )

    # --- seed stale scraped fighters from image if hosted ---
    seed_summary: dict[str, object] | None = None
    if is_hosted_runtime():
        seed_summary = _seed_stale_scraped_fighters()

    roster_df = sync_official_active_roster(output_path=OFFICIAL_ACTIVE_ROSTER_PATH)
    roster_summary = _roster_summary(roster_df, output_path=OFFICIAL_ACTIVE_ROSTER_PATH)

    backfill_summary = run_backfill(
        refresh_roster=False,
        limit_fighters=limit_fighters,
        roster_df=roster_df,
    )

    profile_supplement_summary: dict[str, object] | None = None
    if not skip_audit:
        _pre_audit_summary, pre_audit_df = run_audit(
            active_roster_path=OFFICIAL_ACTIVE_ROSTER_PATH,
            processed_fights_path=processed_fights_path,
            scraped_fighters_path=scraped_fighters_path,
        )
        try:
            profile_supplement_summary = _maybe_refresh_profile_supplement(
                active_roster_path=OFFICIAL_ACTIVE_ROSTER_PATH,
                scraped_fighters_path=scraped_fighters_path,
                audit_df=pre_audit_df,
                partial_refresh=bool(limit_fighters is not None),
            )
        except Exception as exc:
            logger.warning("Profile supplement refresh failed: %s", exc, exc_info=True)
            profile_supplement_summary = {"action": "error", "reason": str(exc)}
        if profile_supplement_summary:
            logger.info("Targeted profile supplement refresh result: %s", profile_supplement_summary)

    rebuild_summary: dict[str, object] | None = None
    if not skip_rebuild:
        update_production_manifest = is_hosted_runtime() or bool(
            str(os.getenv(PRODUCTION_BUNDLE_ENV, "") or "").strip()
        )
        rebuild_summary = run_rebuild(
            dataset_variant=dataset_variant,
            output_subdirs=output_subdirs or list(DEFAULT_REBUILD_OUTPUT_SUBDIRS),
            update_production_manifest=update_production_manifest,
        )

    audit_summary: dict[str, object] | None = None
    audit_alert_summary: dict[str, object] | None = None
    unresolved_summary: dict[str, object] | None = None
    unresolved_df = pd.DataFrame(columns=PROFILE_REPORT_COLUMNS)
    if not skip_audit:
        # Verify the audit reads the SAME files the backfill just wrote to
        if str(scraped_fighters_path.resolve()) != str(BACKFILL_FIGHTERS_PATH.resolve()):
            logger.error(
                "PATH MISMATCH: audit scraped_fighters_path=%s != backfill FIGHTERS_PATH=%s",
                scraped_fighters_path, BACKFILL_FIGHTERS_PATH,
            )

        audit_summary, audit_df = run_audit(
            active_roster_path=OFFICIAL_ACTIVE_ROSTER_PATH,
            processed_fights_path=processed_fights_path,
            scraped_fighters_path=scraped_fighters_path,
        )
        if audit_json_path is not None:
            audit_json_path.parent.mkdir(parents=True, exist_ok=True)
            audit_json_path.write_text(json.dumps(audit_summary, indent=2) + "\n", encoding="utf-8")
        if audit_csv_path is not None:
            audit_csv_path.parent.mkdir(parents=True, exist_ok=True)
            audit_df.sort_values(["split_alias_aware", "official_name"]).to_csv(audit_csv_path, index=False)
        unresolved_summary, unresolved_df = _build_unresolved_profile_report(
            active_roster_path=OFFICIAL_ACTIVE_ROSTER_PATH,
            scraped_fighters_path=scraped_fighters_path,
            audit_df=audit_df,
            supplement_path=PROFILE_SUPPLEMENT_PATH,
        )
        audit_alert_summary = _build_profile_audit_alert_summary(
            active_roster_path=OFFICIAL_ACTIVE_ROSTER_PATH,
            audit_df=audit_df,
            unresolved_df=unresolved_df,
        )
        if unresolved_json_path is not None or unresolved_csv_path is not None:
            if unresolved_json_path is not None:
                _write_json(unresolved_json_path, unresolved_summary)
            if unresolved_csv_path is not None:
                unresolved_csv_path.parent.mkdir(parents=True, exist_ok=True)
                unresolved_df.to_csv(unresolved_csv_path, index=False)
            logger.info("Post-refresh unresolved profile report: %s", unresolved_summary)

    # --- diagnostic: post-refresh file state ---
    post_refresh_state = {
        "active_roster": _file_snapshot(OFFICIAL_ACTIVE_ROSTER_PATH),
        "scraped_fighters": _file_snapshot(scraped_fighters_path),
        "processed_fights": _file_snapshot(processed_fights_path),
    }
    post_refresh_row_guard_state = _row_guard_snapshot()
    row_drop_guard = _build_row_drop_guard(pre_refresh_row_guard_state, post_refresh_row_guard_state)
    if row_drop_guard["violations"]:
        logger.warning("UFC refresh row-drop guard violations: %s", row_drop_guard["violations"])
    logger.info(
        "UFC refresh post-state: roster=%s scraped_fighters=%s processed_fights=%s",
        post_refresh_state["active_roster"],
        post_refresh_state["scraped_fighters"],
        post_refresh_state["processed_fights"],
    )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "limit_fighters": int(limit_fighters) if limit_fighters is not None else None,
        "partial_refresh": bool(limit_fighters is not None),
        "resolved_paths": resolved_paths,
        "pre_refresh_file_state": pre_refresh_state,
        "post_refresh_file_state": post_refresh_state,
        "row_drop_guard": row_drop_guard,
        "seed_stale_scraped_fighters": seed_summary,
        "roster_sync": roster_summary,
        "ufcstats_backfill": backfill_summary,
        "profile_supplement_refresh": profile_supplement_summary,
        "rebuild": rebuild_summary,
        "profile_audit": audit_summary,
        "profile_audit_alert_summary": audit_alert_summary,
        "profile_unresolved_report": unresolved_summary,
        "profile_audit_json_path": str(audit_json_path) if audit_json_path is not None else "",
        "profile_audit_csv_path": str(audit_csv_path) if audit_csv_path is not None else "",
        "profile_unresolved_json_path": str(unresolved_json_path) if unresolved_json_path is not None else "",
        "profile_unresolved_csv_path": str(unresolved_csv_path) if unresolved_csv_path is not None else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-variant",
        default=DEFAULT_DATASET_VARIANT,
        help="Which pulled UFC training-row universe to rebuild.",
    )
    parser.add_argument(
        "--output-subdir",
        action="append",
        default=None,
        help="Processed output subdir(s) to rebuild. Defaults to base + promoted candidate dirs.",
    )
    parser.add_argument("--limit-fighters", type=int, default=None)
    parser.add_argument("--skip-rebuild", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--audit-json-path", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--audit-csv-path", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--unresolved-json-path", type=Path, default=DEFAULT_UNRESOLVED_PROFILE_JSON)
    parser.add_argument("--unresolved-csv-path", type=Path, default=DEFAULT_UNRESOLVED_PROFILE_CSV)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_scheduled_refresh(
        dataset_variant=args.dataset_variant,
        output_subdirs=args.output_subdir,
        limit_fighters=args.limit_fighters,
        audit_json_path=args.audit_json_path,
        audit_csv_path=args.audit_csv_path,
        unresolved_json_path=args.unresolved_json_path,
        unresolved_csv_path=args.unresolved_csv_path,
        skip_rebuild=args.skip_rebuild,
        skip_audit=args.skip_audit,
    )
    _write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
