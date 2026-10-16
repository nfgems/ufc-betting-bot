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
from src.data.name_utils import normalize_person_name, normalize_ufcstats_id  # noqa: E402
from src.data.profile_integrity import source_entity_id  # noqa: E402
from src.data.reviewed_roster_identity import (  # noqa: E402
    reviewed_profile_identity_binding_record,
    reviewed_profile_field_decisions,
    reviewed_profile_official_lookup_key,
    reviewed_ufcstats_identity_rejection,
    reviewed_ufcstats_resolution_blocked,
)
from src.data.reviewed_route_bound_coverage import (  # noqa: E402
    route_bound_coverage_certificate,
)
from src.data.reviewed_roster_recovery import (  # noqa: E402
    reviewed_profile_field_hold,
)
from src.data.ufc_active_roster import (  # noqa: E402
    _explicit_boolean,
    _official_url_identity_positively_trusted,
    _row_has_power_slap_evidence,
)
from src.data.ufc_refresh import (  # noqa: E402
    _coerce_profile_height_cm,
    _coerce_profile_reach_cm,
    _load_scraped_fighter_lookup,
    _normalize_name,
)
from src.features.stance_utils import encode_stance  # noqa: E402


DEFAULT_ACTIVE_ROSTER_PATH = RAW_DATA_DIR / "ufc_active_roster_official.csv"
DEFAULT_PROCESSED_FIGHTS_PATH = PROCESSED_DATA_DIR / "fights_cleaned.csv"
DEFAULT_SCRAPED_FIGHTERS_PATH = RAW_DATA_DIR / "ufc_fighters_scraped.csv"
DEFAULT_SUPPLEMENTAL_PROFILES_PATH = (
    RAW_DATA_DIR / "ufc_fighters_profile_supplement.csv"
)
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


def _valid_stance(value: object) -> bool:
    if _blank(value):
        return False
    return bool(pd.notna(encode_stance(value)))


def _coverage_eligible(row: pd.Series | dict[str, object]) -> bool:
    if _is_test_or_staging_profile(row):
        return False
    if _row_has_power_slap_evidence(row):
        return False
    explicit = row.get("coverage_eligible")
    if not _blank(explicit):
        return _explicit_boolean(explicit) is True
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
    from src.data.ufc_active_roster import _official_url_identity_positively_trusted

    return _official_url_identity_positively_trusted(row)


def _route_bound_official_field_present(
    row: pd.Series | dict[str, object],
    field: str,
    *,
    excluded_fields: set[tuple[str, str, str]] | None = None,
) -> bool:
    """Count a UFC.com physical only when its exact fetched route owns the row."""

    if field not in {"height", "reach"}:
        return False
    parsed_value = (
        _coerce_profile_height_cm(row.get(field))
        if field == "height"
        else _coerce_profile_reach_cm(row.get(field))
    )
    if pd.isna(parsed_value) or float(parsed_value) <= 0:
        return False
    if reviewed_profile_field_hold(row, field=field) is not None:
        return False
    if any(
        str(decision.get("audit_field") or "").strip() == field
        for decision in reviewed_profile_field_decisions(row)
    ):
        # Reviewed cells count only through their exact observation selector and
        # current-profile overlay, never through a changed raw roster value.
        return False
    if str(row.get("official_profile_fetch_outcome") or "").strip().casefold() != "succeeded":
        return False
    audit_only_certificate = route_bound_coverage_certificate(row, field)
    if not (
        _official_url_identity_positively_trusted(row)
        or audit_only_certificate is not None
    ):
        return False
    cached_fields = {
        value.strip()
        for value in str(
            row.get("official_profile_cached_biography_fields") or ""
        ).split("|")
        if value.strip()
    }
    if field in cached_fields:
        return False

    official_url = str(row.get("official_athlete_url") or "").strip().rstrip("/")
    canonical_url = str(row.get("canonical_athlete_url") or "").strip().rstrip("/")
    official_entity = source_entity_id("ufc", official_url)
    canonical_entity = source_entity_id("ufc", canonical_url)
    stable_id = normalize_ufcstats_id(row.get("ufcstats_url")) or ""
    exclusion_key = (official_entity, stable_id, field)
    if excluded_fields and exclusion_key in excluded_fields:
        return False
    return bool(
        official_entity
        and canonical_entity
        and official_entity == canonical_entity
        and official_url.casefold() == canonical_url.casefold()
    )


def _row_aliases(row: pd.Series | dict[str, object]) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()

    def _add_alias(value: object) -> None:
        if _blank(value):
            return
        text = str(value or "").strip()
        key = normalize_person_name(text)
        if not key or key in seen:
            return
        seen.add(key)
        aliases.append(text)

    alias_fields = ["official_name", "ufcstats_name"]
    if _official_url_identity_trusted(row):
        alias_fields.extend(["profile_name", "slug_name"])
    for field in alias_fields:
        _add_alias(row.get(field))

    if _official_url_identity_trusted(row):
        for field in ("canonical_athlete_url", "official_athlete_url"):
            url = str(row.get(field) or "").strip()
            slug = urlparse(url).path.strip("/").rsplit("/", 1)[-1] if url else ""
            if slug:
                _add_alias(slug.replace("-", " "))
        for value in str(row.get("alternate_slug_names") or "").split("|"):
            _add_alias(value)
    return aliases


def _build_processed_fighter_membership(
    processed_fights_path: Path,
) -> tuple[set[str], set[str]]:
    processed_df = pd.read_csv(processed_fights_path)
    required_name_columns = {"fighter_a", "fighter_b"}
    if not required_name_columns.issubset(processed_df.columns):
        raise ValueError(
            "Processed fights are missing fighter_a/fighter_b coverage columns"
        )
    name_keys = {
        normalize_person_name(name)
        for name in pd.concat([processed_df["fighter_a"], processed_df["fighter_b"]], ignore_index=True)
        .dropna()
        .astype(str)
        if normalize_person_name(name)
    }
    stable_ids: set[str] = set()
    for column in ("fighter_a_id", "fighter_b_id"):
        if column not in processed_df.columns:
            continue
        for value in processed_df[column].dropna():
            stable_id = normalize_ufcstats_id(value)
            if stable_id:
                stable_ids.add(stable_id)
    return name_keys, stable_ids


def _row_stable_identity(row: pd.Series | dict[str, object]) -> tuple[str, str]:
    carriers: list[str] = []
    for field in ("stable_ufcstats_id", "fighter_id", "ufcstats_url"):
        value = row.get(field)
        if _blank(value):
            continue
        stable_id = normalize_ufcstats_id(value)
        if stable_id is None:
            return "", "invalid_stable_id_carrier_fail_closed_new"
        carriers.append(stable_id)
    unique = set(carriers)
    if not unique:
        return "", "no_stable_id_fail_closed_new"
    if len(unique) != 1:
        return "", "conflicting_stable_id_carriers_fail_closed_new"
    return next(iter(unique)), "exact_stable_id"


def _split_label_stable_identity(
    row: pd.Series | dict[str, object],
    processed_stable_ids: set[str],
) -> tuple[str, str]:
    stable_id, reason = _row_stable_identity(row)
    if not stable_id:
        return "newly_added_active_roster", reason
    if stable_id in processed_stable_ids:
        return (
            "existing_processed_active_roster",
            "exact_stable_id_in_processed",
        )
    return "newly_added_active_roster", "exact_stable_id_not_in_processed"


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
    """Resolve a profile only through one consistent stable UFCStats identity.

    The legacy name and URL-to-name fallbacks could attach a same-name or
    reviewed-alias profile to the wrong roster row.  Keep the keyword solely
    for compatibility with callers while making it deliberately non-operative.
    """

    del url_to_name_key
    carriers: list[str] = []
    for field in ("stable_ufcstats_id", "fighter_id", "ufcstats_url"):
        value = row.get(field)
        if _blank(value):
            continue
        stable_id = normalize_ufcstats_id(value)
        if stable_id is None:
            return "", None
        carriers.append(stable_id)
    stable_ids = set(carriers)
    if len(stable_ids) > 1:
        return "", None
    reviewed_record = reviewed_profile_identity_binding_record(row)
    reviewed_key = reviewed_profile_official_lookup_key(row)
    if reviewed_record is not None:
        reviewed_stable_id = normalize_ufcstats_id(
            reviewed_record.get("stable_ufcstats_id")
        )
        expected_reviewed_key = (
            f"ufcstats:{reviewed_stable_id}"
            if reviewed_stable_id
            else f"ufc:{reviewed_record['official_source_entity_id']}"
        )
        if not reviewed_key or reviewed_key != expected_reviewed_key:
            return "", None
    if not stable_ids:
        return (reviewed_key, lookup.get(reviewed_key)) if reviewed_key else ("", None)
    if reviewed_ufcstats_identity_rejection(row) is not None:
        return "", None
    stable_id = next(iter(stable_ids))
    key = f"ufcstats:{stable_id}"
    if reviewed_key and reviewed_key != key:
        return "", None
    if reviewed_ufcstats_resolution_blocked(row) and not reviewed_key:
        return "", None
    return (key, lookup.get(key))


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
    supplemental_profiles_path: Path,
    excluded_route_bound_official_fields: set[tuple[str, str, str]] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    active_df = pd.read_csv(active_roster_path).copy()
    processed_keys, processed_stable_ids = _build_processed_fighter_membership(
        processed_fights_path
    )
    profile_lookup = _load_scraped_fighter_lookup(
        scraped_fighters_path,
        supplemental_profiles_path=supplemental_profiles_path,
        official_active_roster_path=active_roster_path,
        include_reviewed_current_profile_overlays=True,
    )
    url_to_name_key = _build_url_to_name_key(scraped_fighters_path)

    # Diagnostic: record the exact source files and their sizes
    import logging as _audit_logging
    _audit_logger = _audit_logging.getLogger(__name__)
    _audit_logger.info(
        "Audit source files: active_roster=%s (%d rows), processed_fights=%s (%d processed keys), "
        "scraped_fighters=%s, supplemental_profiles=%s "
        "(%d profile lookup entries, %d URL mappings)",
        active_roster_path, len(active_df),
        processed_fights_path, len(processed_keys),
        scraped_fighters_path, supplemental_profiles_path,
        len(profile_lookup), len(url_to_name_key),
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
        height_present = (
            profile is not None and not _blank(profile.get("height"))
        ) or _route_bound_official_field_present(
            row,
            "height",
            excluded_fields=excluded_route_bound_official_fields,
        )
        reach_present = (
            profile is not None and not _blank(profile.get("reach"))
        ) or _route_bound_official_field_present(
            row,
            "reach",
            excluded_fields=excluded_route_bound_official_fields,
        )
        stance_present = profile is not None and _valid_stance(profile.get("stance"))
        split_stable_identity, split_stable_identity_reason = (
            _split_label_stable_identity(row, processed_stable_ids)
        )

        audit_rows.append(
            {
                "official_name": row.get("official_name"),
                "official_athlete_url": row.get("official_athlete_url"),
                "ufcstats_name": row.get("ufcstats_name"),
                "ufcstats_url": row.get("ufcstats_url"),
                "stable_ufcstats_id": row.get("stable_ufcstats_id"),
                "profile_source_alias": matched_alias,
                "combat_sport": _combat_sport_label(row),
                "coverage_eligible": coverage_eligible,
                "split_official_name": _split_label_official_name(row, processed_keys),
                "split_alias_aware": _split_label_alias_aware(row, processed_keys),
                "split_stable_identity": split_stable_identity,
                "split_stable_identity_reason": split_stable_identity_reason,
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
            "supplemental_profiles_path": str(supplemental_profiles_path),
            "profile_lookup_entries": len(profile_lookup),
            "url_to_name_key_entries": len(url_to_name_key),
            "processed_fighter_keys": len(processed_keys),
        },
        "overall_summary": _summarize_split(eligible_audit_df.copy()),
        "processed_active_row_split_official_name": eligible_audit_df["split_official_name"].value_counts().to_dict(),
        "processed_active_row_split_alias_aware": eligible_audit_df["split_alias_aware"].value_counts().to_dict(),
        "processed_active_row_split_stable_identity": eligible_audit_df["split_stable_identity"].value_counts().to_dict(),
        "split_summary_official_name": {
            split: _summarize_split(group.copy())
            for split, group in eligible_audit_df.groupby("split_official_name", dropna=False)
        },
        "split_summary_alias_aware": {
            split: _summarize_split(group.copy())
            for split, group in eligible_audit_df.groupby("split_alias_aware", dropna=False)
        },
        "split_summary_stable_identity": {
            split: _summarize_split(group.copy())
            for split, group in eligible_audit_df.groupby("split_stable_identity", dropna=False)
        },
    }
    return summary, audit_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-roster-path", type=Path, default=DEFAULT_ACTIVE_ROSTER_PATH)
    parser.add_argument("--processed-fights-path", type=Path, default=DEFAULT_PROCESSED_FIGHTS_PATH)
    parser.add_argument("--scraped-fighters-path", type=Path, default=DEFAULT_SCRAPED_FIGHTERS_PATH)
    parser.add_argument(
        "--supplemental-profiles-path",
        type=Path,
        default=DEFAULT_SUPPLEMENTAL_PROFILES_PATH,
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary, audit_df = run_audit(
        active_roster_path=args.active_roster_path,
        processed_fights_path=args.processed_fights_path,
        scraped_fighters_path=args.scraped_fighters_path,
        supplemental_profiles_path=args.supplemental_profiles_path,
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
