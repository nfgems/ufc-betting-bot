"""
Build supplemental pro and amateur pre-UFC fight history from Sherdog + Tapology.

This script:
1. Loads UFC fight history to identify fighters and their first UFC appearance
2. Scrapes each fighter's full pre-UFC pro history (Sherdog AND Tapology, takes best)
3. Scrapes each fighter's amateur MMA history into a separate supplement
4. Filters both tracks to pre-UFC dates only (before their first UFC appearance)
5. Preserves the event/promotion name as an `organization` field
6. Saves separate CSVs for later feature-pipeline integration

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
    _dedupe_supplement_rows,
    scrape_fighter_amateur_fights,
    scrape_fighter_pre_ufc_fights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
CANDIDATES_DIR = REPO_ROOT / "data" / "processed" / "candidates"
OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "pre_ufc_career_supplement_v2.csv"
CHECKPOINT_PATH = REPO_ROOT / "data" / "raw" / "pre_ufc_career_checkpoint_v2.json"
AMATEUR_OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "amateur_career_supplement.csv"
AMATEUR_CHECKPOINT_PATH = REPO_ROOT / "data" / "raw" / "amateur_career_checkpoint.json"
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



# scrape_fighter_pre_ufc_fights / scrape_fighter_amateur_fights are imported from src.data.pre_ufc_scraper


def _dedupe_rows(rows: list[dict], *, dedupe_mirrors: bool = False) -> pd.DataFrame:
    """Deduplicate fight rows, preferring entries that preserve organization metadata."""
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(rows)
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    if dedupe_mirrors:
        df = _dedupe_supplement_rows(df)

    df["_fighter_a_key"] = df["fighter_a"].map(normalize_person_name)
    df["_fighter_b_key"] = df["fighter_b"].map(normalize_person_name)
    df["_org_present"] = df["organization"].fillna("").astype(str).str.len()
    df = df.sort_values(["_org_present"], ascending=False)
    df = df.drop_duplicates(subset=["event_date", "_fighter_a_key", "_fighter_b_key"], keep="first")
    df = df.drop(columns=["_fighter_a_key", "_fighter_b_key", "_org_present"], errors="ignore")
    return df.reindex(columns=OUTPUT_COLUMNS)


def _save_rows(
    rows: list[dict],
    *,
    output_path: Path | None = None,
    dedupe_mirrors: bool = False,
) -> None:
    resolved_output_path = output_path or OUTPUT_PATH
    output_df = _dedupe_rows(rows, dedupe_mirrors=dedupe_mirrors)
    write_csv_atomically(output_df, resolved_output_path)


def _load_existing_rows(
    *,
    output_path: Path | None = None,
    replace_fighters: set[str] | None = None,
) -> list[dict]:
    """Load current supplement rows, optionally excluding specific fighters."""
    resolved_output_path = output_path or OUTPUT_PATH
    if not resolved_output_path.exists():
        return []

    existing_df = pd.read_csv(resolved_output_path)
    if replace_fighters:
        existing_df = existing_df[~existing_df["fighter_a"].isin(replace_fighters)]
    return existing_df.to_dict("records")


def load_checkpoint(path: Path | None = None) -> dict:
    """Load scraping checkpoint (fighters already processed)."""
    checkpoint_path = path or CHECKPOINT_PATH
    if checkpoint_path.exists():
        return json.loads(checkpoint_path.read_text())
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


def save_checkpoint(checkpoint: dict, path: Path | None = None) -> None:
    """Save scraping checkpoint."""
    checkpoint_path = path or CHECKPOINT_PATH
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, default=str))


def _finalize_output(
    rows: list[dict],
    *,
    output_path: Path,
    label: str,
    dedupe_mirrors: bool = False,
) -> dict[str, object]:
    """Persist one supplement output with guardrails against accidental data loss."""
    existing_row_count = 0
    existing_fighter_counts: dict[str, int] = {}
    existing_df = pd.DataFrame()
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        existing_row_count = len(existing_df)
        if "fighter_a" in existing_df.columns:
            existing_fighter_counts = existing_df.groupby("fighter_a").size().to_dict()

    saved_row_count = 0
    guard_preserved: list[str] = []
    new_fighters_added: list[str] = []

    if rows:
        df = _dedupe_rows(rows, dedupe_mirrors=dedupe_mirrors)

        if existing_fighter_counts and "fighter_a" in df.columns:
            new_fighter_counts = df.groupby("fighter_a").size().to_dict()
            for fighter, old_count in existing_fighter_counts.items():
                new_count = new_fighter_counts.get(fighter, 0)
                if new_count < old_count:
                    old_fighter_rows = existing_df[existing_df["fighter_a"] == fighter]
                    df = df[df["fighter_a"] != fighter]
                    df = pd.concat([df, old_fighter_rows], ignore_index=True)
                    guard_preserved.append(f"{fighter} ({old_count} rows kept, new had {new_count})")

            if guard_preserved:
                df = _dedupe_rows(df.to_dict("records"), dedupe_mirrors=dedupe_mirrors)

        if existing_row_count > 0 and len(df) < existing_row_count * 0.9:
            logger.error(
                "INTEGRITY GUARD (%s): Total row count would drop from %d to %d (>10%% loss). "
                "Aborting save to protect existing data.",
                label,
                existing_row_count,
                len(df),
            )
            raise RuntimeError(f"{label} supplement integrity guard triggered")

        write_csv_atomically(df, output_path)
        saved_row_count = len(df)
        logger.info("Saved %d %s rows to %s", len(df), label, output_path)

        if existing_fighter_counts:
            current_fighters = set(df["fighter_a"].dropna().unique())
            old_fighters = set(existing_fighter_counts.keys())
            new_fighters_added = sorted(current_fighters - old_fighters)
    elif existing_row_count > 0:
        logger.info("No new %s rows scraped; existing data preserved unchanged", label)
    else:
        logger.info("No %s fights found", label)

    return {
        "existing_row_count": existing_row_count,
        "saved_row_count": saved_row_count,
        "guard_preserved": guard_preserved,
        "new_fighters_added": new_fighters_added,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build pro and amateur pre-UFC career supplements from Sherdog and Tapology"
    )
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

    checkpoint = load_checkpoint(CHECKPOINT_PATH) if args.resume else {"processed": {}, "failed": []}
    amateur_checkpoint = (
        load_checkpoint(AMATEUR_CHECKPOINT_PATH)
        if args.resume
        else {"processed": {}, "failed": []}
    )
    already_done = set(checkpoint["processed"].keys())
    amateur_already_done = set(amateur_checkpoint["processed"].keys())
    zero_row_fighters = {
        name for name, count in checkpoint["processed"].items() if count == 0
    }
    amateur_zero_row_fighters = {
        name for name, count in amateur_checkpoint["processed"].items() if count == 0
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
            if name in zero_row_fighters or name in amateur_zero_row_fighters
        }
    else:
        to_scrape = {
            name: info for name, info in candidates.items()
            if name not in already_done or name not in amateur_already_done
        }

    if args.resume:
        logger.info(
            "Resuming: pro=%d processed, amateur=%d processed, %d remaining",
            len(already_done),
            len(amateur_already_done),
            len(to_scrape),
        )
    if explicit_fighters:
        logger.info(
            "Explicit fighter override: %d requested, %d resolved via candidate/official roster set",
            len(explicit_fighters),
            len(to_scrape),
        )
    if args.retry_zero_rows:
        logger.info("Retrying %d fighters previously saved with 0 rows in either track", len(to_scrape))

    if args.max_fighters:
        to_scrape = dict(list(to_scrape.items())[:args.max_fighters])

    logger.info("Scraping %d fighters for pro and amateur history...", len(to_scrape))

    replace_fighters = set(to_scrape.keys()) if explicit_fighters else None
    pro_rows: list[dict] = []
    amateur_rows: list[dict] = []
    if OUTPUT_PATH.exists():
        pro_rows = _load_existing_rows(
            output_path=OUTPUT_PATH,
            replace_fighters=replace_fighters,
        )
        logger.info("Loaded %d existing pre-UFC fight rows", len(pro_rows))
    if AMATEUR_OUTPUT_PATH.exists():
        amateur_rows = _load_existing_rows(
            output_path=AMATEUR_OUTPUT_PATH,
            replace_fighters=replace_fighters,
        )
        logger.info("Loaded %d existing amateur fight rows", len(amateur_rows))

    scraped_count = 0
    found_count = 0
    amateur_scraped_count = 0
    amateur_found_count = 0

    for i, (name, info) in enumerate(to_scrape.items()):
        if (i + 1) % 50 == 0:
            logger.info("Progress: %d/%d fighters processed", i + 1, len(to_scrape))
            save_checkpoint(checkpoint, CHECKPOINT_PATH)
            save_checkpoint(amateur_checkpoint, AMATEUR_CHECKPOINT_PATH)
            if pro_rows:
                _save_rows(pro_rows, output_path=OUTPUT_PATH)
            if amateur_rows:
                _save_rows(
                    amateur_rows,
                    output_path=AMATEUR_OUTPUT_PATH,
                    dedupe_mirrors=True,
                )

        pro_should_scrape = bool(explicit_fighters) or (
            name in zero_row_fighters if args.retry_zero_rows else name not in already_done
        )
        amateur_should_scrape = bool(explicit_fighters) or (
            name in amateur_zero_row_fighters if args.retry_zero_rows else name not in amateur_already_done
        )

        if pro_should_scrape:
            try:
                rows = scrape_fighter_pre_ufc_fights(name, info["first_ufc_date"])
                scraped_count += 1
                if rows:
                    pro_rows.extend(rows)
                    found_count += 1
                    checkpoint["processed"][name] = len(rows)
                    logger.info("  %s: %d pre-UFC fights found", name, len(rows))
                else:
                    checkpoint["processed"][name] = 0
            except Exception as e:
                logger.warning(f"  {name}: pre-UFC scrape error: {e}")
                checkpoint["failed"].append(name)

        if amateur_should_scrape:
            try:
                rows = scrape_fighter_amateur_fights(name, info["first_ufc_date"])
                amateur_scraped_count += 1
                if rows:
                    amateur_rows.extend(rows)
                    amateur_found_count += 1
                    amateur_checkpoint["processed"][name] = len(rows)
                    logger.info("  %s: %d amateur fights found", name, len(rows))
                else:
                    amateur_checkpoint["processed"][name] = 0
            except Exception as e:
                logger.warning(f"  {name}: amateur scrape error: {e}")
                amateur_checkpoint["failed"].append(name)

        # _get_soup() already enforces per-request pacing, so no extra
        # per-fighter delay is needed here.
        time.sleep(0)

    try:
        pro_summary = _finalize_output(
            pro_rows,
            output_path=OUTPUT_PATH,
            label="pre-UFC",
        )
        amateur_summary = _finalize_output(
            amateur_rows,
            output_path=AMATEUR_OUTPUT_PATH,
            label="amateur",
            dedupe_mirrors=True,
        )
    except RuntimeError:
        save_checkpoint(checkpoint, CHECKPOINT_PATH)
        save_checkpoint(amateur_checkpoint, AMATEUR_CHECKPOINT_PATH)
        sys.exit(1)

    save_checkpoint(checkpoint, CHECKPOINT_PATH)
    save_checkpoint(amateur_checkpoint, AMATEUR_CHECKPOINT_PATH)

    logger.info("\n" + "=" * 60)
    logger.info("SCRAPE SUMMARY")
    logger.info("=" * 60)
    logger.info("  Pro fighters scraped this run:      %d", scraped_count)
    logger.info("  Pro fighters with rows found:       %d", found_count)
    logger.info(
        "  Pro rows before/after/net:          %d / %d / %+d",
        pro_summary["existing_row_count"],
        pro_summary["saved_row_count"],
        int(pro_summary["saved_row_count"]) - int(pro_summary["existing_row_count"]),
    )
    logger.info("  Amateur fighters scraped this run:  %d", amateur_scraped_count)
    logger.info("  Amateur fighters with rows found:   %d", amateur_found_count)
    logger.info(
        "  Amateur rows before/after/net:      %d / %d / %+d",
        amateur_summary["existing_row_count"],
        amateur_summary["saved_row_count"],
        int(amateur_summary["saved_row_count"]) - int(amateur_summary["existing_row_count"]),
    )
    if pro_summary["new_fighters_added"]:
        logger.info("  New pro fighters added (%d):", len(pro_summary["new_fighters_added"]))
        for name in pro_summary["new_fighters_added"][:20]:
            logger.info(f"    + {name}")
        if len(pro_summary["new_fighters_added"]) > 20:
            logger.info("    ... and %d more", len(pro_summary["new_fighters_added"]) - 20)
    if amateur_summary["new_fighters_added"]:
        logger.info("  New amateur fighters added (%d):", len(amateur_summary["new_fighters_added"]))
        for name in amateur_summary["new_fighters_added"][:20]:
            logger.info(f"    + {name}")
        if len(amateur_summary["new_fighters_added"]) > 20:
            logger.info("    ... and %d more", len(amateur_summary["new_fighters_added"]) - 20)
    if pro_summary["guard_preserved"]:
        logger.info("  Pro guard preserved existing data (%d):", len(pro_summary["guard_preserved"]))
        for msg in pro_summary["guard_preserved"]:
            logger.info(f"    ! {msg}")
    if amateur_summary["guard_preserved"]:
        logger.info(
            "  Amateur guard preserved existing data (%d):",
            len(amateur_summary["guard_preserved"]),
        )
        for msg in amateur_summary["guard_preserved"]:
            logger.info(f"    ! {msg}")
    logger.info("  Pro failed:                        %d", len(checkpoint["failed"]))
    if checkpoint["failed"]:
        for name in checkpoint["failed"][-10:]:
            logger.info(f"    x {name}")
    logger.info("  Amateur failed:                    %d", len(amateur_checkpoint["failed"]))
    if amateur_checkpoint["failed"]:
        for name in amateur_checkpoint["failed"][-10:]:
            logger.info(f"    x {name}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
