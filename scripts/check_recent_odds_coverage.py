"""Gate recent-card odds coverage and overall odds non-regression before a refit.

The training coverage gate in src/model/train.py is global (98% of the 2014+
train frame), which leaves enough slack that several fully odds-less recent
cards can pass it. This check closes that hole for automated retrains:

1. Every recent card (event_date within --recent-days) with at least
   --min-card-rows rows must have complete-odds coverage of at least
   --min-card-coverage.
2. The count of complete-odds rows in the train-eligible window must not
   drop below the same count in the committed HEAD snapshot (odds
   non-regression; skipped when HEAD has no copy of the file).
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

FEATURES_REL_PATH = "data/processed/features.csv"
ODDS_COLS = ["a_implied_prob", "b_implied_prob", "diff_implied_prob"]
USECOLS = ["event_date", "target", "a_num_fights", "b_num_fights", *ODDS_COLS]
TRAIN_START_DATE = "2014-01-01"
MIN_FIGHTS_EACH = 2


def _load_frame(buffer) -> pd.DataFrame:
    df = pd.read_csv(buffer, usecols=USECOLS, low_memory=False)
    df["event_date"] = pd.to_datetime(df["event_date"], format="mixed", errors="coerce")
    return df


def _complete_odds_mask(df: pd.DataFrame) -> pd.Series:
    return df[ODDS_COLS].notna().all(axis=1)


def _train_eligible_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["event_date"] >= pd.Timestamp(TRAIN_START_DATE))
        & (df["a_num_fights"] >= MIN_FIGHTS_EACH)
        & (df["b_num_fights"] >= MIN_FIGHTS_EACH)
        & df["target"].notna()
    )


def _head_complete_count() -> int | None:
    result = subprocess.run(
        ["git", "show", f"HEAD:{FEATURES_REL_PATH}"],
        capture_output=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        return None
    head_df = _load_frame(io.BytesIO(result.stdout))
    return int((_complete_odds_mask(head_df) & _train_eligible_mask(head_df)).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=REPO_ROOT / FEATURES_REL_PATH)
    parser.add_argument("--recent-days", type=int, default=60)
    parser.add_argument("--min-card-rows", type=int, default=4)
    parser.add_argument("--min-card-coverage", type=float, default=0.5)
    parser.add_argument(
        "--skip-non-regression",
        action="store_true",
        help="Skip the HEAD comparison (for local snapshots not yet committed).",
    )
    parser.add_argument(
        "--minimum-complete-odds-rows",
        type=int,
        help=(
            "Require this absolute minimum of train-eligible rows with complete "
            "odds. Use this with --skip-non-regression for isolated refit snapshots."
        ),
    )
    args = parser.parse_args()
    if (
        args.minimum_complete_odds_rows is not None
        and args.minimum_complete_odds_rows < 0
    ):
        parser.error("--minimum-complete-odds-rows must be non-negative")

    df = _load_frame(args.features_path)
    complete = _complete_odds_mask(df)

    failures: list[str] = []

    cutoff = df["event_date"].max() - pd.Timedelta(days=args.recent_days)
    recent = df[df["event_date"] >= cutoff]
    for event_date, group in recent.groupby(recent["event_date"].dt.date):
        if len(group) < args.min_card_rows:
            continue
        coverage = float(complete.loc[group.index].mean())
        marker = "ok" if coverage >= args.min_card_coverage else "FAIL"
        print(
            f"card {event_date}: {int(complete.loc[group.index].sum())}/{len(group)} "
            f"complete-odds rows ({coverage:.0%}) [{marker}]"
        )
        if coverage < args.min_card_coverage:
            failures.append(
                f"card {event_date} complete-odds coverage {coverage:.0%} "
                f"below {args.min_card_coverage:.0%}"
            )

    current_count = int((complete & _train_eligible_mask(df)).sum())
    print(f"train-eligible complete-odds rows: {current_count}")
    if (
        args.minimum_complete_odds_rows is not None
        and current_count < args.minimum_complete_odds_rows
    ):
        failures.append(
            "train-eligible complete-odds rows below fixed floor: "
            f"{current_count} < {args.minimum_complete_odds_rows}"
        )
    if not args.skip_non_regression:
        head_count = _head_complete_count()
        if head_count is None:
            print("non-regression: no HEAD copy of features.csv; skipped")
        else:
            print(f"non-regression: HEAD had {head_count} complete-odds rows")
            if current_count < head_count:
                failures.append(
                    f"train-eligible complete-odds rows regressed: {head_count} -> {current_count}"
                )

    if failures:
        for failure in failures:
            print(f"ODDS COVERAGE GATE FAILED: {failure}", file=sys.stderr)
        return 1
    print("Recent odds coverage gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
