"""
E14: refit cadence + train-window sweep (compute-heavy; run when idle).

Prices the staleness penalty of the frozen production deployment by running
the spec-driven walk-forward at different retrain intervals, and sweeps
train_start_date (never swept anywhere) for the window question.

Usage:
    python scripts/e14_cadence_sweep.py --cadences 1 2 4 6
    python scripts/e14_cadence_sweep.py --train-starts 2014-01-01 2017-01-01 2019-01-01
"""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DATA_DIR, TRAIN_CUTOFF_DATE
from src.model.training_spec import resolve_named_training_spec
from src.strategy.backtest import run_walkforward_strategy_comparison

logger = logging.getLogger("e14_cadence_sweep")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cadences", type=int, nargs="*", default=[1, 2, 4, 6])
    parser.add_argument("--train-starts", nargs="*", default=[])
    parser.add_argument("--spec", default="full_live_contract_v6_tuned")
    parser.add_argument("--out", default="logs/e14_cadence_sweep.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    features_df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv", parse_dates=["event_date"])
    base_spec = resolve_named_training_spec(args.spec)

    arms = [("cadence", months, base_spec) for months in args.cadences]
    for start in args.train_starts:
        arms.append((
            "train_start", start,
            replace(base_spec, name=f"{base_spec.name}_start{start[:4]}", train_start_date=start),
        ))

    rows = []
    for kind, value, spec in arms:
        retrain = value if kind == "cadence" else 6
        logger.info("=== E14 arm: %s=%s ===", kind, value)
        frame = features_df.copy()
        if kind == "train_start":
            frame = frame[frame["event_date"] >= pd.Timestamp(value)]
        result = run_walkforward_strategy_comparison(
            frame,
            spec=spec,
            retrain_months=int(retrain) if kind == "cadence" else 6,
            bet_start_date=TRAIN_CUTOFF_DATE,
            write_artifacts=False,
        )
        predictive = result.get("predictive_metrics", {})
        models = predictive.get("models", {}).get("xgboost", {})
        summary = result.get("summary")
        row = {"arm": f"{kind}={value}"}
        row.update({f"model_{k}": v for k, v in models.items()})
        if summary is not None and hasattr(summary, "empty") and not summary.empty:
            gated = summary[summary["strategy"] == "production_gated"]
            if not gated.empty:
                g = gated.iloc[0]
                for col in ("total_bets", "roi", "avg_clv", "max_drawdown"):
                    if col in gated.columns:
                        row[f"strategy_{col}"] = g[col]
        rows.append(row)
        table = pd.DataFrame(rows)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(table.to_string(index=False))

    print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
