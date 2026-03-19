"""Backfill moneyline gaps for V4 expansion audits using The Odds API history path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.historical_backfill import BACKFILL_DIR, run_backfill  # noqa: E402


def _load_missing_fights(path: Path) -> pd.DataFrame:
    missing = pd.read_csv(path, parse_dates=["event_date"])
    required = {"event_date", "fighter_a", "fighter_b"}
    missing_cols = required - set(missing.columns)
    if missing_cols:
        raise SystemExit(f"Missing required columns in {path}: {sorted(missing_cols)}")
    fights = (
        missing[list(required)]
        .dropna(subset=["event_date", "fighter_a", "fighter_b"])
        .drop_duplicates()
        .sort_values(["event_date", "fighter_a", "fighter_b"])
        .reset_index(drop=True)
    )
    return fights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-csv",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "coverage_audits" / "v4_144_train" / "moneyline_odds_missing_fights.csv",
        help="Fight-level missing-moneyline inventory CSV from audit_v4_expansion_coverage.py",
    )
    parser.add_argument(
        "--from-date",
        default=None,
        help="Optional inclusive lower bound for fight event_date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--to-date",
        default=None,
        help="Optional inclusive upper bound for fight event_date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--offsets",
        nargs="+",
        type=int,
        default=[7, 3, 1],
        help="Historical query offsets in days before the fight date",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume behavior and refetch even if historical_odds.csv exists",
    )
    args = parser.parse_args()

    fights = _load_missing_fights(args.inventory_csv)
    if args.from_date:
        fights = fights[fights["event_date"] >= pd.Timestamp(args.from_date)]
    if args.to_date:
        fights = fights[fights["event_date"] <= pd.Timestamp(args.to_date)]
    fights = fights.reset_index(drop=True)

    if fights.empty:
        print("No fights matched the requested filters.")
        return 0

    print(
        {
            "inventory_csv": str(args.inventory_csv),
            "fights": int(len(fights)),
            "min_event_date": fights["event_date"].min().date().isoformat(),
            "max_event_date": fights["event_date"].max().date().isoformat(),
            "offsets": list(args.offsets),
            "resume": not args.no_resume,
            "historical_odds_path": str(BACKFILL_DIR / "historical_odds.csv"),
        }
    )

    result = run_backfill(
        fights_df=fights,
        offsets=list(args.offsets),
        resume=not args.no_resume,
    )
    print(
        {
            "backfilled_rows_total": int(len(result)),
            "distinct_fights_with_odds": int(
                result[["event_date", "fighter_a", "fighter_b"]].drop_duplicates().shape[0]
            )
            if not result.empty
            else 0,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
