"""
Track B batch (R1): spec-knob experiments vs the v6 tuned baseline.

Runs the spec-driven walk-forward (the PRODUCTION trainer, mirror
augmentation included) for each named spec on the identical feature matrix
and folds, then tabulates model metrics (log-loss/Brier/ECE incl. the
blended probability) and the production-gated strategy economics.

Arms:
  full_live_contract_v6_tuned            baseline (promoted eval spec)
  full_live_contract_v6_refit            E2: full-data booster, holdout sigmoid
  full_live_contract_v6_cal_weighted     E11: weight-aware sigmoid
  full_live_contract_v6_cal_isotonic     E11: isotonic
  full_live_contract_v6_cal_none         E11: no calibration
  full_live_contract_v6_noise_antithetic E12: identity-preserving odds noise

Usage:
    python scripts/track_b_batch.py --out logs/track_b_batch
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

logger = logging.getLogger("track_b_batch")

DEFAULT_SPECS = [
    "full_live_contract_v6_tuned",
    "full_live_contract_v6_refit",
    "full_live_contract_v6_cal_weighted",
    "full_live_contract_v6_cal_isotonic",
    "full_live_contract_v6_cal_none",
    "full_live_contract_v6_noise_antithetic",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", nargs="+", default=DEFAULT_SPECS)
    parser.add_argument("--out", default="logs/track_b_batch")
    parser.add_argument("--retrain-months", type=int, default=6)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv", parse_dates=["event_date"])

    rows = []
    for spec_name in args.specs:
        logger.info("=== Track B arm: %s ===", spec_name)
        spec = resolve_named_training_spec(spec_name)
        result = run_walkforward_strategy_comparison(
            features_df.copy(),
            spec=spec,
            retrain_months=args.retrain_months,
            bet_start_date=TRAIN_CUTOFF_DATE,
            write_artifacts=False,
        )
        predictive = result.get("predictive_metrics", {})
        models = predictive.get("models", {})
        blend = predictive.get("blend", {})
        summary = result.get("summary")

        row = {"spec": spec_name}
        xgb_metrics = models.get("xgboost", {})
        row.update({f"model_{k}": v for k, v in xgb_metrics.items()})
        blend_overall = blend.get("overall", {})
        row.update({f"blend_{k}": v for k, v in blend_overall.items()})
        band = blend.get("band_040_065", {})
        row.update({f"blendband_{k}": v for k, v in band.items()})

        if summary is not None and hasattr(summary, "empty") and not summary.empty:
            gated = summary[summary["strategy"] == "production_gated"]
            if not gated.empty:
                g = gated.iloc[0]
                for col in ("total_bets", "wins", "win_rate", "roi", "total_profit",
                            "avg_clv", "max_drawdown"):
                    if col in gated.columns:
                        row[f"strategy_{col}"] = g[col]
        rows.append(row)

        with (out_dir / f"{spec_name}_predictive.json").open("w", encoding="utf-8") as fh:
            json.dump(predictive, fh, indent=2, default=str)
        if summary is not None and hasattr(summary, "to_csv"):
            summary.to_csv(out_dir / f"{spec_name}_strategy_summary.csv", index=False)
        for strat_name, strat_result in result.get("strategy_results", {}).items():
            bet_log = strat_result.get("bet_log")
            if isinstance(bet_log, pd.DataFrame) and not bet_log.empty:
                bet_log.to_csv(out_dir / f"{spec_name}_{strat_name}_bets.csv", index=False)

        table = pd.DataFrame(rows)
        table.to_csv(out_dir / "track_b_summary.csv", index=False)
        print(f"\n--- progress: {len(rows)}/{len(args.specs)} arms done ---")
        print(table.to_string(index=False))

    print(f"\nSaved Track B summary to {out_dir / 'track_b_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
