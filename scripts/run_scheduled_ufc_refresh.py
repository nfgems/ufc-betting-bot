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
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_active_roster_profile_completeness import run_audit
from scripts.backfill_active_roster_ufcstats import run_backfill
from scripts.rebuild_ufc_processed_artifacts import run_rebuild
from src.model.production_bundle import PRODUCTION_BUNDLE_ENV, is_hosted_runtime
from src.config import DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.ufc_active_roster import OFFICIAL_ACTIVE_ROSTER_PATH, sync_official_active_roster


DEFAULT_DATASET_VARIANT = "pulled_all_plus_legacy_market"
DEFAULT_REBUILD_OUTPUT_SUBDIRS = [
    ".",
    "candidates/full_live_contract_v5_fullfit_retrain",
    "candidates/v6_eval",
]
DEFAULT_AUDIT_JSON = DATA_DIR / "tmp" / "active_roster_profile_completeness_scheduled_latest.json"
DEFAULT_AUDIT_CSV = DATA_DIR / "tmp" / "active_roster_profile_completeness_scheduled_latest.csv"


def _roster_summary(df: pd.DataFrame, *, output_path: Path) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "resolved_ufcstats_urls": int(df.get("ufcstats_url", pd.Series(dtype="object")).fillna("").astype(str).str.strip().ne("").sum()),
        "resolved_ufcstats_names": int(df.get("ufcstats_name", pd.Series(dtype="object")).fillna("").astype(str).str.strip().ne("").sum()),
        "with_profile_details": int(df.get("profile_status", pd.Series(dtype="object")).fillna("").astype(str).str.strip().ne("").sum()),
        "output_path": str(output_path),
    }


def _write_json(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_scheduled_refresh(
    *,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    output_subdirs: list[str] | None = None,
    limit_fighters: int | None = None,
    audit_json_path: Path | None = DEFAULT_AUDIT_JSON,
    audit_csv_path: Path | None = DEFAULT_AUDIT_CSV,
    skip_rebuild: bool = False,
    skip_audit: bool = False,
) -> dict[str, object]:
    roster_df = sync_official_active_roster(output_path=OFFICIAL_ACTIVE_ROSTER_PATH)
    roster_summary = _roster_summary(roster_df, output_path=OFFICIAL_ACTIVE_ROSTER_PATH)

    backfill_summary = run_backfill(
        refresh_roster=False,
        limit_fighters=limit_fighters,
        roster_df=roster_df,
    )

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
    if not skip_audit:
        audit_summary, audit_df = run_audit(
            active_roster_path=OFFICIAL_ACTIVE_ROSTER_PATH,
            processed_fights_path=PROCESSED_DATA_DIR / "fights_cleaned.csv",
            scraped_fighters_path=RAW_DATA_DIR / "ufc_fighters_scraped.csv",
        )
        if audit_json_path is not None:
            audit_json_path.parent.mkdir(parents=True, exist_ok=True)
            audit_json_path.write_text(json.dumps(audit_summary, indent=2) + "\n", encoding="utf-8")
        if audit_csv_path is not None:
            audit_csv_path.parent.mkdir(parents=True, exist_ok=True)
            audit_df.sort_values(["split_alias_aware", "official_name"]).to_csv(audit_csv_path, index=False)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "roster_sync": roster_summary,
        "ufcstats_backfill": backfill_summary,
        "rebuild": rebuild_summary,
        "profile_audit": audit_summary,
        "profile_audit_json_path": str(audit_json_path) if audit_json_path is not None else "",
        "profile_audit_csv_path": str(audit_csv_path) if audit_csv_path is not None else "",
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
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_scheduled_refresh(
        dataset_variant=args.dataset_variant,
        output_subdirs=args.output_subdir,
        limit_fighters=args.limit_fighters,
        audit_json_path=args.audit_json_path,
        audit_csv_path=args.audit_csv_path,
        skip_rebuild=args.skip_rebuild,
        skip_audit=args.skip_audit,
    )
    _write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
