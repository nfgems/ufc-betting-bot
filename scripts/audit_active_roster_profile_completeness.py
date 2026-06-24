"""Audit active-roster profile completeness from official roster + local profile artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR  # noqa: E402
from src.data.name_utils import normalize_person_name  # noqa: E402
from src.data.ufc_refresh import _load_scraped_fighter_lookup, _normalize_name  # noqa: E402


DEFAULT_ACTIVE_ROSTER_PATH = RAW_DATA_DIR / "ufc_active_roster_official.csv"
DEFAULT_PROCESSED_FIGHTS_PATH = PROCESSED_DATA_DIR / "fights_cleaned.csv"
DEFAULT_SCRAPED_FIGHTERS_PATH = RAW_DATA_DIR / "ufc_fighters_scraped.csv"
_TEST_PROFILE_NAME_KEYS = {"test", "test fighter", "test test", "testy test", "testing test"}


def _blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() in {"", "--", "nan", "NaN"}


def _valid_dob(value: object) -> bool:
    if _blank(value):
        return False
    return bool(pd.notna(pd.to_datetime(value, errors="coerce", format="mixed")))


def _coverage_eligible(row: pd.Series | dict[str, object]) -> bool:
    if _is_test_or_staging_profile(row):
        return False
    explicit = row.get("coverage_eligible")
    if not _blank(explicit):
        return str(explicit).strip().lower() not in {"0", "false", "no", "off"}
    return str(row.get("combat_sport") or "").strip().lower() != "power_slap"


def _combat_sport_label(row: pd.Series | dict[str, object]) -> str:
    if _is_test_or_staging_profile(row):
        return "test_profile"
    return str(row.get("combat_sport") or "").strip() or "mma"


def _is_test_or_staging_profile(row: pd.Series | dict[str, object]) -> bool:
    names = [row.get("official_name"), row.get("profile_name"), row.get("slug_name")]
    names.extend(str(row.get("alternate_slug_names") or "").split("|"))
    name_keys = {normalize_person_name(value) for value in names if str(value or "").strip()}
    if name_keys & _TEST_PROFILE_NAME_KEYS:
        return True
    url = str(row.get("official_athlete_url") or "").strip()
    path = urlparse(url).path.strip("/").lower() if url else ""
    slug = path.rsplit("/", 1)[-1] if path else ""
    return slug in {"test", "test-fighter", "test-test", "testy-test", "testing-test"}


def _official_url_identity_trusted(row: pd.Series | dict[str, object]) -> bool:
    status = str(row.get("official_url_identity_status") or "").strip().lower()
    if status in {"mismatch", "test_profile"}:
        return False
    explicit = row.get("official_url_identity_valid")
    if _blank(explicit):
        return True
    return str(explicit).strip().lower() not in {"0", "false", "no", "off"}


def _row_aliases(row: pd.Series | dict[str, object]) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    alias_fields = ["official_name", "ufcstats_name"]
    if _official_url_identity_trusted(row):
        alias_fields.extend(["profile_name", "slug_name"])
    for field in alias_fields:
        value = str(row.get(field) or "").strip()
        key = normalize_person_name(value)
        if not key or key in seen:
            continue
        seen.add(key)
        aliases.append(value)

    if _official_url_identity_trusted(row):
        for value in str(row.get("alternate_slug_names") or "").split("|"):
            value = str(value or "").strip()
            key = normalize_person_name(value)
            if not key or key in seen:
                continue
            seen.add(key)
            aliases.append(value)
    return aliases


def _build_processed_fighter_keys(processed_fights_path: Path) -> set[str]:
    processed_df = pd.read_csv(processed_fights_path, usecols=["fighter_a", "fighter_b"])
    return {
        normalize_person_name(name)
        for name in pd.concat([processed_df["fighter_a"], processed_df["fighter_b"]], ignore_index=True)
        .dropna()
        .astype(str)
        if normalize_person_name(name)
    }


def _build_url_to_name_key(scraped_fighters_path: Path) -> dict[str, str]:
    """Build a fighter_url → lookup name key mapping for URL-based profile resolution."""
    if not scraped_fighters_path.exists():
        return {}
    try:
        df = pd.read_csv(scraped_fighters_path, usecols=["name", "fighter_url"])
    except (ValueError, KeyError):
        return {}
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        url = str(row.get("fighter_url") or "").strip()
        name = str(row.get("name") or "").strip()
        if url and name:
            key = _normalize_name(name)
            if key:
                mapping[url] = key
    return mapping


def _resolve_profile(
    row: pd.Series,
    lookup: dict[str, dict],
    *,
    url_to_name_key: dict[str, str] | None = None,
) -> tuple[str, dict[str, object] | None]:
    # 1. Try URL-based match (most reliable — bypasses name normalization)
    if url_to_name_key:
        ufcstats_url = str(row.get("ufcstats_url") or "").strip()
        if ufcstats_url:
            name_key = url_to_name_key.get(ufcstats_url)
            if name_key and name_key in lookup:
                return ufcstats_url, lookup[name_key]

    # 2. Try name match with _normalize_name (matches lookup key format)
    for alias in _row_aliases(row):
        key = _normalize_name(alias)
        profile = lookup.get(key)
        if profile is not None:
            return alias, profile

    # 3. Fallback: normalize_person_name (looser, catches edge cases)
    for alias in _row_aliases(row):
        key = normalize_person_name(alias)
        profile = lookup.get(key)
        if profile is not None:
            return alias, profile

    return "", None


def _split_label_official_name(row: pd.Series, processed_keys: set[str]) -> str:
    official_key = normalize_person_name(row.get("official_name"))
    if official_key and official_key in processed_keys:
        return "existing_processed_active_roster"
    return "newly_added_active_roster"


def _split_label_alias_aware(row: pd.Series, processed_keys: set[str]) -> str:
    for alias in _row_aliases(row):
        if normalize_person_name(alias) in processed_keys:
            return "existing_processed_active_roster"
    return "newly_added_active_roster"


def _summarize_split(frame: pd.DataFrame) -> dict[str, object]:
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
        count = int(frame[field].sum())
        summary[field] = {
            "count": count,
            "pct": round((count / total) * 100.0, 2),
        }
    return summary


def run_audit(
    *,
    active_roster_path: Path,
    processed_fights_path: Path,
    scraped_fighters_path: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    active_df = pd.read_csv(active_roster_path).copy()
    processed_keys = _build_processed_fighter_keys(processed_fights_path)
    profile_lookup = _load_scraped_fighter_lookup(scraped_fighters_path)
    url_to_name_key = _build_url_to_name_key(scraped_fighters_path)

    # Diagnostic: record the exact source files and their sizes
    import logging as _audit_logging
    _audit_logger = _audit_logging.getLogger(__name__)
    _audit_logger.info(
        "Audit source files: active_roster=%s (%d rows), processed_fights=%s (%d processed keys), "
        "scraped_fighters=%s (%d profile lookup entries, %d URL mappings)",
        active_roster_path, len(active_df),
        processed_fights_path, len(processed_keys),
        scraped_fighters_path, len(profile_lookup), len(url_to_name_key),
    )

    audit_rows: list[dict[str, object]] = []
    for _, row in active_df.iterrows():
        matched_alias, profile = _resolve_profile(row, profile_lookup, url_to_name_key=url_to_name_key)
        coverage_eligible = _coverage_eligible(row)
        age_present = not _blank(row.get("age")) or (
            profile is not None and _valid_dob(profile.get("dob"))
        )
        division_present = not _blank(row.get("division"))
        weight_present = (profile is not None and not _blank(profile.get("weight"))) or not _blank(row.get("weight"))
        height_present = profile is not None and not _blank(profile.get("height"))
        reach_present = profile is not None and not _blank(profile.get("reach"))
        stance_present = profile is not None and not _blank(profile.get("stance"))

        audit_rows.append(
            {
                "official_name": row.get("official_name"),
                "ufcstats_name": row.get("ufcstats_name"),
                "profile_source_alias": matched_alias,
                "combat_sport": _combat_sport_label(row),
                "coverage_eligible": coverage_eligible,
                "split_official_name": _split_label_official_name(row, processed_keys),
                "split_alias_aware": _split_label_alias_aware(row, processed_keys),
                "age_present": age_present,
                "division_present": division_present,
                "weight_present": weight_present,
                "height_present": height_present,
                "reach_present": reach_present,
                "stance_present": stance_present,
                "full_physical_bundle_present": all(
                    [age_present, weight_present, height_present, reach_present, stance_present]
                ),
            }
        )

    audit_df = pd.DataFrame(audit_rows)
    eligible_mask = audit_df["coverage_eligible"].fillna(True) if "coverage_eligible" in audit_df.columns else pd.Series(True, index=audit_df.index)
    eligible_audit_df = audit_df[eligible_mask].copy()
    excluded_audit_df = audit_df[~eligible_mask].copy()
    summary = {
        "active_roster_rows": int(len(active_df)),
        "coverage_eligible_active_roster_rows": int(len(eligible_audit_df)),
        "coverage_excluded_active_roster_rows": int(len(excluded_audit_df)),
        "coverage_excluded_by_sport": (
            excluded_audit_df["combat_sport"].fillna("").astype(str).value_counts().sort_index().to_dict()
            if not excluded_audit_df.empty and "combat_sport" in excluded_audit_df.columns
            else {}
        ),
        "active_roster_unique_normalized_names": int(active_df["official_name"].map(normalize_person_name).nunique()),
        "source_files": {
            "active_roster_path": str(active_roster_path),
            "processed_fights_path": str(processed_fights_path),
            "scraped_fighters_path": str(scraped_fighters_path),
            "profile_lookup_entries": len(profile_lookup),
            "url_to_name_key_entries": len(url_to_name_key),
            "processed_fighter_keys": len(processed_keys),
        },
        "overall_summary": _summarize_split(eligible_audit_df.copy()),
        "processed_active_row_split_official_name": eligible_audit_df["split_official_name"].value_counts().to_dict(),
        "processed_active_row_split_alias_aware": eligible_audit_df["split_alias_aware"].value_counts().to_dict(),
        "split_summary_official_name": {
            split: _summarize_split(group.copy())
            for split, group in eligible_audit_df.groupby("split_official_name", dropna=False)
        },
        "split_summary_alias_aware": {
            split: _summarize_split(group.copy())
            for split, group in eligible_audit_df.groupby("split_alias_aware", dropna=False)
        },
    }
    return summary, audit_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-roster-path", type=Path, default=DEFAULT_ACTIVE_ROSTER_PATH)
    parser.add_argument("--processed-fights-path", type=Path, default=DEFAULT_PROCESSED_FIGHTS_PATH)
    parser.add_argument("--scraped-fighters-path", type=Path, default=DEFAULT_SCRAPED_FIGHTERS_PATH)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary, audit_df = run_audit(
        active_roster_path=args.active_roster_path,
        processed_fights_path=args.processed_fights_path,
        scraped_fighters_path=args.scraped_fighters_path,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        audit_df.sort_values(["split_alias_aware", "official_name"]).to_csv(args.output_csv, index=False)

    if args.json or args.output_json is None:
        print(json.dumps(summary, indent=2))
    else:
        print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
