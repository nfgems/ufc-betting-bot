"""
E5: Conviction-trader gate sweep.

The Conviction trader is the largest per-bet allocation in the system (flat
5-8% of equity) and production runs it with c_min_edge=0.0 and no odds cap —
it bets the raw unshrunk model probability. This sweep holds the Single
trader at production controls and grids the conviction gates:

    c_min_edge          {0.00, 0.01, 0.02}
    c_max_decimal_odds  {1.8, 2.0, 2.5, 3.0, inf}
    c_bet_fraction      {0.03, 0.05}

against one shared set of walk-forward fold predictions, with the production
controls as the anchor row. Decision metric: c_roi / c_avg_clv / drawdown vs
the anchor, with non-trivial c_bets required for a read.

Usage:
    python scripts/e5_conviction_sweep.py --out logs/e5_conviction_sweep.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    BLEND_WEIGHT,
    CONVICTION_MIN_MODEL_PROB,
    CONVICTION_MIN_NO_ODDS_PROB,
    MIN_EDGE_THRESHOLD,
    MODEL_AGREEMENT_MIN_EDGE,
    TRADER_C_SHARE,
)
from src.strategy.duo_trader_sweep import (
    _evaluate_config,
    _generate_walk_forward_predictions,
    _summary_row_from_result,
    build_sweep_configs,
)

logger = logging.getLogger("e5_conviction_sweep")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="logs/e5_conviction_sweep.csv")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--preds-cache", default="logs/fold_predictions_baseline.pkl",
                        help="Pickle path for fold predictions (reused by other sweep diagnostics)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    import pickle

    cache_path = Path(args.preds_cache)
    if cache_path.exists():
        logger.info("Loading cached fold predictions from %s", cache_path)
        with cache_path.open("rb") as fh:
            fold_predictions = pickle.load(fh)
    else:
        logger.info("Generating shared walk-forward fold predictions (variant=%s)...", args.variant)
        fold_predictions = _generate_walk_forward_predictions(variant_name=args.variant)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as fh:
            pickle.dump(fold_predictions, fh)
        logger.info("Cached fold predictions to %s", cache_path)
    n_rows = sum(len(p) for _, p in fold_predictions)
    logger.info("Fold predictions ready: %d folds, %d fight rows", len(fold_predictions), n_rows)

    configs = build_sweep_configs(
        variant_name=args.variant,
        s_edges=[MIN_EDGE_THRESHOLD],
        s_blends=[BLEND_WEIGHT],
        s_agreements=[MODEL_AGREEMENT_MIN_EDGE],
        c_model_probs=[CONVICTION_MIN_MODEL_PROB],
        c_no_odds_probs=[CONVICTION_MIN_NO_ODDS_PROB],
        c_shares=[TRADER_C_SHARE],
        c_min_edges=[0.00, 0.01, 0.02],
        c_max_decimal_odds_values=[1.8, 2.0, 2.5, 3.0, float("inf")],
        c_bet_fractions=[0.03, 0.05],
        include_production_controls=True,
    )
    logger.info("Evaluating %d configs (incl. production anchor)", len(configs))

    rows = []
    for config in configs:
        result = _evaluate_config(fold_predictions, config)
        rows.append(_summary_row_from_result(config, result))

    summary = pd.DataFrame(rows)
    summary["c_bet_fraction"] = [c.c_bet_fraction for c in configs]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)

    cols = [
        "config", "c_min_edge", "c_max_decimal_odds", "c_bet_fraction",
        "c_bets", "c_roi", "c_avg_clv", "s_bets", "s_roi",
        "roi", "profit", "max_drawdown",
    ]
    present = [c for c in cols if c in summary.columns]
    ranked = summary.sort_values("c_roi", ascending=False)
    print("\n=== E5 conviction sweep (anchor = production controls) ===")
    print(ranked[present].to_string(index=False))
    print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
