"""Audit active-roster profile completeness from official roster + local profile artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR  # noqa: E402
from src.data.name_utils import normalize_person_name  # noqa: E402
from src.data.ufc_refresh import _load_scraped_fighter_lookup  # noqa: E402


DEFAULT_ACTIVE_ROSTER_PATH = RAW_DATA_DIR / "ufc_active_roster_official.csv"
DEFAULT_PROCESSED_FIGHTS_PATH = PROCESSED_DATA_DIR / "fights_cleaned.csv"
DEFAULT_SCRAPED_FIGHTERS_PATH = RAW_DATA_DIR / "ufc_fighters_scraped.csv"


def _blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() in {"", "--", "nan", "NaN"}


def _row_aliases(row: pd.Series | dict[str, object]) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    for field in ("official_name", "ufcstats_name", "profile_name", "slug_name"):
        value = str(row.get(field) or "").strip()
        key = normalize_person_name(value)
        if not key or key in seen:
            continue
        seen.add(key)
        aliases.append(value)

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


def _resolve_profile(row: pd.Series, lookup: dict[str, dict]) -> tuple[str, dict[str, object] | None]:
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

    audit_rows: list[dict[str, object]] = []
    for _, row in active_df.iterrows():
        matched_alias, profile = _resolve_profile(row, profile_lookup)
        age_present = not _blank(row.get("age"))
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
    summary = {
        "active_roster_rows": int(len(active_df)),
        "active_roster_unique_normalized_names": int(active_df["official_name"].map(normalize_person_name).nunique()),
        "overall_summary": _summarize_split(audit_df.copy()),
        "processed_active_row_split_official_name": audit_df["split_official_name"].value_counts().to_dict(),
        "processed_active_row_split_alias_aware": audit_df["split_alias_aware"].value_counts().to_dict(),
        "split_summary_official_name": {
            split: _summarize_split(group.copy())
            for split, group in audit_df.groupby("split_official_name", dropna=False)
        },
        "split_summary_alias_aware": {
            split: _summarize_split(group.copy())
            for split, group in audit_df.groupby("split_alias_aware", dropna=False)
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
