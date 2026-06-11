"""
E8: Sweep selection-bias diagnostic — forward-chained config selection.

Strategy configs are currently picked as the ROI-best of a ~2,916-config grid
on the same 2022+ window that produces the reported numbers. This measures
how much of that "best" is selection noise:

  For each fold k >= min_train_folds: pick the best config on folds 1..k-1
  (by cumulative profit / cumulative wagered), then score it on fold k.
  The mean out-of-fold ROI of that forward-chained selector, compared with
  the in-sample best-of-grid ROI, is the selection-bias gap.

Also reports the production-controls row on both bases (the production
anchor is a single pre-registered config, so its numbers are not subject to
selection bias — the gap applies to candidate selection).

Usage:
    python scripts/e8_selection_bias.py --preds-cache logs/fold_predictions_baseline.pkl
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategy.duo_trader_sweep import (
    _evaluate_config,
    _generate_walk_forward_predictions,
    _production_sweep_config,
    build_sweep_configs,
)

logger = logging.getLogger("e8_selection_bias")


def _fold_metrics(fold_predictions, config, bet_start_date="2022-01-01"):
    """Per-fold profit/wagered for one config (bankroll resets per fold)."""
    rows = []
    for fold_num, frame in fold_predictions:
        result = _evaluate_config([(fold_num, frame)], config, bet_start_date=bet_start_date)
        combined = result["combined"]
        rows.append({
            "fold": fold_num,
            "profit": combined["total_profit"],
            "wagered": combined["total_wagered"],
            "bets": combined["total_bets"],
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-cache", default="logs/fold_predictions_baseline.pkl")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--min-train-folds", type=int, default=4)
    parser.add_argument("--out", default="logs/e8_selection_bias.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cache_path = Path(args.preds_cache)
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            fold_predictions = pickle.load(fh)
        logger.info("Loaded cached fold predictions (%d folds)", len(fold_predictions))
    else:
        fold_predictions = _generate_walk_forward_predictions(variant_name=args.variant)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as fh:
            pickle.dump(fold_predictions, fh)

    configs = build_sweep_configs(variant_name=args.variant, include_production_controls=False)
    production = _production_sweep_config(
        variant_name=args.variant, name="production_controls", is_baseline=True,
    )
    logger.info("Evaluating %d grid configs per fold (%d folds)",
                len(configs), len(fold_predictions))

    # Config x fold matrices
    per_config: dict[str, pd.DataFrame] = {}
    for i, config in enumerate(configs):
        per_config[config.name] = _fold_metrics(fold_predictions, config)
        if (i + 1) % 250 == 0:
            logger.info("  %d/%d configs evaluated", i + 1, len(configs))
    production_folds = _fold_metrics(fold_predictions, production)

    fold_ids = sorted({fold for fold, _ in fold_predictions})

    def _cum_roi(frame: pd.DataFrame, folds: list[int]) -> float:
        sub = frame[frame["fold"].isin(folds)]
        wagered = sub["wagered"].sum()
        return sub["profit"].sum() / wagered if wagered > 0 else float("-inf")

    # In-sample best-of-grid (the current selection procedure)
    in_sample_roi = {name: _cum_roi(frame, fold_ids) for name, frame in per_config.items()}
    best_in_sample = max(in_sample_roi, key=in_sample_roi.get)

    # Forward-chained selection
    oof_rows = []
    for k_idx in range(args.min_train_folds, len(fold_ids)):
        train_folds = fold_ids[:k_idx]
        test_fold = fold_ids[k_idx]
        train_roi = {name: _cum_roi(frame, train_folds) for name, frame in per_config.items()}
        chosen = max(train_roi, key=train_roi.get)
        test_frame = per_config[chosen]
        test_row = test_frame[test_frame["fold"] == test_fold]
        oof_rows.append({
            "test_fold": test_fold,
            "chosen_config": chosen,
            "train_roi": train_roi[chosen],
            "oof_profit": test_row["profit"].sum(),
            "oof_wagered": test_row["wagered"].sum(),
            "oof_bets": test_row["bets"].sum(),
        })
    oof = pd.DataFrame(oof_rows)
    oof_roi = oof["oof_profit"].sum() / oof["oof_wagered"].sum() if oof["oof_wagered"].sum() > 0 else np.nan

    # Production anchor on the same OOF folds for scale
    oof_folds = oof["test_fold"].tolist()
    prod_oof = production_folds[production_folds["fold"].isin(oof_folds)]
    prod_oof_roi = prod_oof["profit"].sum() / prod_oof["wagered"].sum() if prod_oof["wagered"].sum() > 0 else np.nan

    print("\n=== E8 selection-bias diagnostic ===")
    print(f"Grid size: {len(configs)} configs, folds: {fold_ids}")
    print(f"In-sample best-of-grid: {best_in_sample}")
    print(f"  in-sample ROI (all folds): {in_sample_roi[best_in_sample]:+.2%}")
    print(f"Forward-chained selector OOF ROI (folds {oof_folds}): {oof_roi:+.2%}")
    print(f"Selection-bias gap (in-sample best - OOF): "
          f"{in_sample_roi[best_in_sample] - oof_roi:+.2%}")
    print(f"Production anchor ROI on the same OOF folds: {prod_oof_roi:+.2%}")
    print("\nPer-fold forward-chained choices:")
    print(oof.to_string(index=False))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(args.out, index=False)
    summary_path = Path(args.out).with_suffix(".summary.txt")
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(
            f"in_sample_best={best_in_sample}\n"
            f"in_sample_roi={in_sample_roi[best_in_sample]:.6f}\n"
            f"oof_roi={oof_roi:.6f}\n"
            f"gap={in_sample_roi[best_in_sample] - oof_roi:.6f}\n"
            f"production_oof_roi={prod_oof_roi:.6f}\n"
        )
    print(f"\nSaved to {args.out} and {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
