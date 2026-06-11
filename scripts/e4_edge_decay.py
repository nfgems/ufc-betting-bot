"""
E4: Edge decay vs time-to-event.

The honest backtest fills bets at the T-7 opening snapshot, but live executes
inside the 48h->1h window. With positive CLV, lines move toward our picks, so
T-7 ROI is an upper bound on live-window economics. This experiment re-prices
the production walk-forward at T-7 / T-3 / T-1 snapshots:

  Arm A (fills-only):     model features stay at opening odds (cached fold
                          predictions are reused across offsets); only the
                          execution price moves. Isolates price decay.
  Arm B (features+fills): the model also SEES the T-N odds (full live-window
                          parity); predictions regenerated per offset.

The T-7 fills-only row must reproduce the promoted-eval anchor
(~247 bets / +8.2% ROI / +2.36% CLV) as a regression check.

Usage:
    python scripts/e4_edge_decay.py --arm A --offsets 7 3 1
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DATA_DIR, TRAIN_CUTOFF_DATE
from src.model.training_spec import resolve_named_training_spec
from src.strategy.backtest import run_walkforward_strategy_comparison

logger = logging.getLogger("e4_edge_decay")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["A", "B"], default="A")
    parser.add_argument("--offsets", type=float, nargs="+", default=[7, 3, 1])
    parser.add_argument("--spec", default="full_live_contract_v6_fullfit")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    spec = resolve_named_training_spec(args.spec)
    features_df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv", parse_dates=["event_date"])

    rows = []
    for offset in args.offsets:
        label = f"T-{offset:g} arm {args.arm}"
        logger.info("=== %s ===", label)
        result = run_walkforward_strategy_comparison(
            features_df.copy(),
            spec=spec,
            bet_start_date=TRAIN_CUTOFF_DATE,
            write_artifacts=False,
            entry_offset_days=offset,
            entry_offset_for_features=(args.arm == "B"),
        )
        summary = result.get("summary")
        if summary is None or (hasattr(summary, "empty") and summary.empty):
            logger.warning("No summary for %s", label)
            continue
        summary = summary.copy()
        summary["entry_offset_days"] = offset
        summary["arm"] = args.arm
        rows.append(summary)
        print(f"\n--- {label} ---")
        cols = [c for c in [
            "strategy", "total_bets", "wins", "win_rate", "roi",
            "total_profit", "avg_clv", "max_drawdown",
        ] if c in summary.columns]
        print(summary[cols].to_string(index=False))

    if rows:
        out = pd.concat(rows, ignore_index=True)
        out_path = args.out or f"logs/e4_edge_decay_arm{args.arm}.csv"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"\nSaved curve to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
