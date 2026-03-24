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

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_active_roster_profile_completeness import run_audit
from scripts.backfill_active_roster_ufcstats import run_backfill, FIGHTERS_PATH as BACKFILL_FIGHTERS_PATH
from scripts.rebuild_ufc_processed_artifacts import run_rebuild
from src.model.production_bundle import PRODUCTION_BUNDLE_ENV, is_hosted_runtime
from src.config import DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, PROJECT_ROOT
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
    # --- diagnostic: log all resolved paths and pre-refresh file state ---
    resolved_paths = _log_resolved_data_paths()
    scraped_fighters_path = RAW_DATA_DIR / "ufc_fighters_scraped.csv"
    processed_fights_path = PROCESSED_DATA_DIR / "fights_cleaned.csv"

    pre_refresh_state = {
        "active_roster": _file_snapshot(OFFICIAL_ACTIVE_ROSTER_PATH),
        "scraped_fighters": _file_snapshot(scraped_fighters_path),
        "processed_fights": _file_snapshot(processed_fights_path),
    }
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

    # --- diagnostic: post-refresh file state ---
    post_refresh_state = {
        "active_roster": _file_snapshot(OFFICIAL_ACTIVE_ROSTER_PATH),
        "scraped_fighters": _file_snapshot(scraped_fighters_path),
        "processed_fights": _file_snapshot(processed_fights_path),
    }
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
        "seed_stale_scraped_fighters": seed_summary,
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
