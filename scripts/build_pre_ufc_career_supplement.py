"""
Build a v2 supplemental pre-UFC fight history from Sherdog + Tapology for UFC fighters.

This script:
1. Loads UFC fight history to identify fighters and their first UFC appearance
2. Scrapes each fighter's full pre-UFC history (Sherdog AND Tapology, takes best)
3. Filters to pre-UFC fights only (before their first UFC appearance)
4. Preserves the event/promotion name as an `organization` field
5. Saves a v2 CSV for later feature-pipeline integration

Per-fight stats (sig str, TD, control, etc.) remain NaN because regional sources
do not publish UFCStats-style box scores.

Usage:
    python scripts/build_pre_ufc_career_supplement.py [--max-fighters N] [--resume]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.name_utils import (  # noqa: E402
    normalize_cross_source_name,
    normalize_person_name,
)
from src.data.io_utils import write_csv_atomically  # noqa: E402
from src.data.pre_ufc_scraper import (  # noqa: E402
    OUTPUT_COLUMNS,
    scrape_fighter_pre_ufc_fights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
CANDIDATES_DIR = REPO_ROOT / "data" / "processed" / "candidates"
OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "pre_ufc_career_supplement_v2.csv"
CHECKPOINT_PATH = REPO_ROOT / "data" / "raw" / "pre_ufc_career_checkpoint_v2.json"
OFFICIAL_ROSTER_PATH = REPO_ROOT / "data" / "raw" / "ufc_active_roster_official.csv"


def _find_best_features_csv() -> Path:
    """Find the most recent v5 candidate features.csv (correct pipeline)."""
    for name in [
        "full_live_contract_v5_fullfit_retrain",
        "full_live_contract_v5_fullfit",
        "full_live_contract_v5_eval_retrain",
        "full_live_contract_v5_eval",
    ]:
        path = CANDIDATES_DIR / name / "fights_cleaned.csv"
        if path.exists():
            return path
    # Fallback to default
    return REPO_ROOT / "data" / "processed" / "fights_cleaned.csv"


def identify_ufc_fighters(fights_path: Path, max_ufc_fights: int | None = None) -> dict[str, dict]:
    """
    Find UFC fighters and their first UFC appearance in the training data.

    Returns {fighter_name: {"ufc_fights": N, "first_ufc_date": date_str, "aliases": [...]}}.
    """
    df = pd.read_csv(fights_path, usecols=["fighter_a", "fighter_b", "event_date"])
    logger.info(f"Loaded {len(df)} fights from {fights_path.name}")
    grouped: dict[str, dict] = {}

    def _record(name: object, event_date: object) -> None:
        display_name = str(name or "").strip()
        if not display_name:
            return
        key = normalize_person_name(display_name)
        if not key:
            return
        row = grouped.setdefault(
            key,
            {
                "fighter_name": display_name,
                "aliases": set(),
                "ufc_fights": 0,
                "first_ufc_date": None,
            },
        )
        row["aliases"].add(display_name)
        row["ufc_fights"] += 1
        if row["fighter_name"].startswith(" ") or len(display_name) < len(row["fighter_name"]):
            row["fighter_name"] = display_name
        event_date_str = str(event_date)
        if row["first_ufc_date"] is None or event_date_str < str(row["first_ufc_date"]):
            row["first_ufc_date"] = event_date_str

    for _, row in df.iterrows():
        _record(row.get("fighter_a"), row.get("event_date"))
        _record(row.get("fighter_b"), row.get("event_date"))

    candidates = {}
    for info in grouped.values():
        if max_ufc_fights is not None and info["ufc_fights"] > max_ufc_fights:
            continue
        candidates[info["fighter_name"]] = {
            "ufc_fights": info["ufc_fights"],
            "first_ufc_date": info["first_ufc_date"],
            "aliases": sorted(info["aliases"]),
        }

    logger.info(
        "Found %d fighters%s",
        len(candidates),
        f" with <= {max_ufc_fights} UFC fights" if max_ufc_fights is not None else "",
    )
    return dict(sorted(candidates.items()))



# scrape_fighter_pre_ufc_fights is now imported from src.data.pre_ufc_scraper


def _dedupe_rows(rows: list[dict]) -> pd.DataFrame:
    """Deduplicate fight rows, preferring entries that preserve organization metadata."""
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(rows)
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    df["_fighter_a_key"] = df["fighter_a"].map(normalize_person_name)
    df["_fighter_b_key"] = df["fighter_b"].map(normalize_person_name)
    df["_org_present"] = df["organization"].fillna("").astype(str).str.len()
    df = df.sort_values(["_org_present"], ascending=False)
    df = df.drop_duplicates(subset=["event_date", "_fighter_a_key", "_fighter_b_key"], keep="first")
    df = df.drop(columns=["_fighter_a_key", "_fighter_b_key", "_org_present"], errors="ignore")
    return df.reindex(columns=OUTPUT_COLUMNS)


def _save_rows(rows: list[dict]) -> None:
    output_df = _dedupe_rows(rows)
    write_csv_atomically(output_df, OUTPUT_PATH)


def _load_existing_rows(*, replace_fighters: set[str] | None = None) -> list[dict]:
    """Load current supplement rows, optionally excluding specific fighters."""
    if not OUTPUT_PATH.exists():
        return []

    existing_df = pd.read_csv(OUTPUT_PATH)
    if replace_fighters:
        existing_df = existing_df[~existing_df["fighter_a"].isin(replace_fighters)]
    return existing_df.to_dict("records")


def load_checkpoint() -> dict:
    """Load scraping checkpoint (fighters already processed)."""
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text())
    return {"processed": {}, "failed": []}


def load_official_roster_lookup() -> dict[str, dict[str, object]]:
    """Load explicit-fighter fallback info from the official roster artifact."""
    if not OFFICIAL_ROSTER_PATH.exists():
        return {}

    roster_df = pd.read_csv(OFFICIAL_ROSTER_PATH)
    if roster_df.empty:
        return {}

    lookup: dict[str, dict[str, object]] = {}
    for _, row in roster_df.iterrows():
        first_ufc_date = row.get("octagon_debut")
        info = {
            "ufc_fights": 0,
            "first_ufc_date": first_ufc_date if pd.notna(first_ufc_date) else None,
            "aliases": [],
        }
        for field in ("official_name", "ufcstats_name"):
            name = str(row.get(field) or "").strip()
            key = normalize_cross_source_name(name)
            if key and key not in lookup:
                lookup[key] = info
    return lookup


def save_checkpoint(checkpoint: dict) -> None:
    """Save scraping checkpoint."""
    CHECKPOINT_PATH.write_text(json.dumps(checkpoint, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description="Build a v2 pre-UFC career supplement from Sherdog")
    parser.add_argument("--max-fighters", type=int, default=None,
                        help="Limit number of fighters to scrape (for testing)")
    parser.add_argument("--max-ufc-fights", type=int, default=None,
                        help="Optional cap: scrape fighters with <= N UFC fights")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just identify candidates, don't scrape")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--retry-zero-rows", action="store_true",
                        help="Re-scrape fighters previously checkpointed with 0 rows")
    parser.add_argument("--fighters", nargs="+", default=None,
                        help="Explicit fighter names to scrape (comma-separated or repeated)")
    args = parser.parse_args()

    fights_path = _find_best_features_csv()
    logger.info(f"Using training data from: {fights_path}")

    candidates = identify_ufc_fighters(fights_path, max_ufc_fights=args.max_ufc_fights)

    if args.dry_run:
        print(f"\nWould scrape {len(candidates)} fighters. Top 20:")
        for name, info in sorted(candidates.items())[:20]:
            print(f"  {name}: {info['ufc_fights']} UFC fights, first: {info['first_ufc_date']}")
        return

    # Load checkpoint
    checkpoint = load_checkpoint() if args.resume else {"processed": {}, "failed": []}
    already_done = set(checkpoint["processed"].keys())
    zero_row_fighters = {
        name for name, count in checkpoint["processed"].items()
        if count == 0
    }

    explicit_fighters: list[str] = []
    if args.fighters:
        for raw_value in args.fighters:
            explicit_fighters.extend(
                value.strip() for value in str(raw_value).split(",") if value.strip()
            )

    # Filter out already-processed
    if explicit_fighters:
        roster_lookup = load_official_roster_lookup()
        to_scrape = {}
        for name in dict.fromkeys(explicit_fighters):
            if name in candidates:
                to_scrape[name] = candidates[name]
                continue
            roster_info = roster_lookup.get(normalize_cross_source_name(name))
            if roster_info is not None:
                to_scrape[name] = roster_info
    elif args.retry_zero_rows:
        to_scrape = {
            name: info for name, info in candidates.items()
            if name in zero_row_fighters
        }
    else:
        to_scrape = {
            name: info for name, info in candidates.items()
            if name not in already_done
        }

    if already_done:
        logger.info(f"Resuming: {len(already_done)} already processed, {len(to_scrape)} remaining")
    if explicit_fighters:
        logger.info(
            "Explicit fighter override: %d requested, %d resolved via candidate/official roster set",
            len(explicit_fighters),
            len(to_scrape),
        )
    if args.retry_zero_rows:
        logger.info("Retrying %d fighters previously saved with 0 rows", len(to_scrape))

    if args.max_fighters:
        to_scrape = dict(list(to_scrape.items())[:args.max_fighters])

    logger.info(f"Scraping {len(to_scrape)} fighters from Sherdog...")

    all_rows: list[dict] = []
    # Load existing rows from previous checkpoint
    if OUTPUT_PATH.exists():
        all_rows = _load_existing_rows(
            replace_fighters=set(to_scrape.keys()) if explicit_fighters else None
        )
        logger.info(f"Loaded {len(all_rows)} existing pre-UFC fight rows")

    scraped_count = 0
    found_count = 0

    for i, (name, info) in enumerate(to_scrape.items()):
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(to_scrape)} fighters scraped")
            save_checkpoint(checkpoint)
            # Save intermediate results
            if all_rows:
                _save_rows(all_rows)

        try:
            rows = scrape_fighter_pre_ufc_fights(name, info["first_ufc_date"])
            scraped_count += 1

            if rows:
                all_rows.extend(rows)
                found_count += 1
                checkpoint["processed"][name] = len(rows)
                logger.info(f"  {name}: {len(rows)} pre-UFC fights found")
            else:
                checkpoint["processed"][name] = 0

        except Exception as e:
            logger.warning(f"  {name}: scrape error: {e}")
            checkpoint["failed"].append(name)

        # _get_soup() already enforces per-request pacing, so no extra
        # per-fighter delay is needed here.
        time.sleep(0)

    # --- Integrity guard: never lose rows we already have ---
    existing_row_count = 0
    existing_fighter_counts: dict[str, int] = {}
    if OUTPUT_PATH.exists():
        existing_df = pd.read_csv(OUTPUT_PATH)
        existing_row_count = len(existing_df)
        if "fighter_a" in existing_df.columns:
            existing_fighter_counts = (
                existing_df.groupby("fighter_a").size().to_dict()
            )

    saved_row_count = 0
    guard_preserved: list[str] = []

    if all_rows:
        df = _dedupe_rows(all_rows)

        # Per-fighter guard: if a fighter had more rows before, keep old rows
        if existing_fighter_counts and "fighter_a" in df.columns:
            new_fighter_counts = df.groupby("fighter_a").size().to_dict()
            for fighter, old_count in existing_fighter_counts.items():
                new_count = new_fighter_counts.get(fighter, 0)
                if new_count < old_count:
                    # Scrape returned fewer rows — restore old data for this fighter
                    old_fighter_rows = existing_df[existing_df["fighter_a"] == fighter]
                    df = df[df["fighter_a"] != fighter]
                    df = pd.concat([df, old_fighter_rows], ignore_index=True)
                    guard_preserved.append(f"{fighter} ({old_count} rows kept, new had {new_count})")

            if guard_preserved:
                df = _dedupe_rows(df.to_dict("records"))

        # Total guard: abort if row count dropped >10%
        if existing_row_count > 0 and len(df) < existing_row_count * 0.9:
            logger.error(
                "INTEGRITY GUARD: Total row count would drop from %d to %d (>10%% loss). "
                "Aborting save to protect existing data.",
                existing_row_count, len(df),
            )
            save_checkpoint(checkpoint)
            sys.exit(1)

        write_csv_atomically(df, OUTPUT_PATH)
        saved_row_count = len(df)
        logger.info(f"\nSaved {len(df)} pre-UFC fight rows to {OUTPUT_PATH}")
    else:
        if existing_row_count > 0:
            logger.info("\nNo new rows scraped; existing data preserved unchanged")
        else:
            logger.info("\nNo pre-UFC fights found")

    save_checkpoint(checkpoint)

    # --- Verification summary ---
    new_fighters_added: list[str] = []
    if all_rows and existing_fighter_counts:
        current_df = pd.read_csv(OUTPUT_PATH) if OUTPUT_PATH.exists() else pd.DataFrame()
        if "fighter_a" in current_df.columns:
            current_fighters = set(current_df["fighter_a"].unique())
            old_fighters = set(existing_fighter_counts.keys())
            new_fighters_added = sorted(current_fighters - old_fighters)

    logger.info("\n" + "=" * 60)
    logger.info("SCRAPE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Fighters scraped this run:    {scraped_count}")
    logger.info(f"  Fighters with pre-UFC fights: {found_count}")
    logger.info(f"  Rows before:                  {existing_row_count}")
    logger.info(f"  Rows after:                   {saved_row_count}")
    logger.info(f"  Net change:                   {saved_row_count - existing_row_count:+d}")
    if new_fighters_added:
        logger.info(f"  New fighters added ({len(new_fighters_added)}):")
        for name in new_fighters_added[:20]:
            logger.info(f"    + {name}")
        if len(new_fighters_added) > 20:
            logger.info(f"    ... and {len(new_fighters_added) - 20} more")
    if guard_preserved:
        logger.info(f"  Guard preserved existing data ({len(guard_preserved)}):")
        for msg in guard_preserved:
            logger.info(f"    ! {msg}")
    logger.info(f"  Failed:                       {len(checkpoint['failed'])}")
    if checkpoint["failed"]:
        for name in checkpoint["failed"][-10:]:
            logger.info(f"    x {name}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
