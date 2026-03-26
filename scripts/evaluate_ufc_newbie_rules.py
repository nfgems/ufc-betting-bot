"""Evaluate UFC newbie-rule variants on the current production spec."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    INITIAL_BANKROLL,
    KELLY_FRACTION,
    MAX_BET_FRACTION,
    MIN_EDGE_THRESHOLD,
    PROCESSED_DATA_DIR,
)
from src.model.production_bundle import load_production_bundle
from src.model.training_spec import resolve_named_training_spec
from src.strategy.backtest import (
    PRODUCTION_GATED_STRATEGY,
    _simulate_backtest_predictions,
    run_walkforward_strategy_comparison,
)
from src.strategy.lab_stats import compute_sharpe_ratio
from src.strategy.value import (
    DEFAULT_NEWBIE_RULE,
    LEGACY_BASELINE_NEWBIE_RULE,
    TIERED_ORG_AWARE_NEWBIE_RULE,
)

logger = logging.getLogger(__name__)


def _fixed_stake_metrics(bet_log: pd.DataFrame) -> dict[str, float]:
    if bet_log.empty:
        return {
            "equal_stake_profit_units": 0.0,
            "equal_stake_roi": float("nan"),
        }

    won = bet_log["won"].astype(bool)
    unit_profit = won.astype(float) * (bet_log["odds"] - 1.0) + (~won).astype(float) * -1.0
    return {
        "equal_stake_profit_units": float(unit_profit.sum()),
        "equal_stake_roi": float(unit_profit.mean()),
    }


def _newbie_metrics(bet_log: pd.DataFrame) -> dict[str, float | int]:
    if bet_log.empty or "is_newbie_bet" not in bet_log.columns:
        return {
            "newbie_bet_count": 0,
            "newbie_total_wagered": 0.0,
            "newbie_total_profit": 0.0,
            "newbie_roi": float("nan"),
            "newbie_equal_stake_profit_units": 0.0,
            "newbie_equal_stake_roi": float("nan"),
        }

    newbie_log = bet_log[bet_log["is_newbie_bet"].fillna(False)].copy()
    if newbie_log.empty:
        return {
            "newbie_bet_count": 0,
            "newbie_total_wagered": 0.0,
            "newbie_total_profit": 0.0,
            "newbie_roi": float("nan"),
            "newbie_equal_stake_profit_units": 0.0,
            "newbie_equal_stake_roi": float("nan"),
        }

    wagered = float(newbie_log["bet_size"].sum())
    profit = float(newbie_log["profit"].sum())
    roi = profit / wagered if wagered > 0 else float("nan")
    equal_stake = _fixed_stake_metrics(newbie_log)
    return {
        "newbie_bet_count": int(len(newbie_log)),
        "newbie_total_wagered": wagered,
        "newbie_total_profit": profit,
        "newbie_roi": roi,
        "newbie_equal_stake_profit_units": float(equal_stake["equal_stake_profit_units"]),
        "newbie_equal_stake_roi": float(equal_stake["equal_stake_roi"]),
    }


def _summary_row(result: dict) -> dict[str, object]:
    strategy = result["strategy_config"]
    stats = result["stats"]
    bet_log = result["bet_log"]
    equal_stake = _fixed_stake_metrics(bet_log)
    newbie = _newbie_metrics(bet_log)
    return {
        "strategy": strategy.name,
        "newbie_rule": strategy.newbie_rule.name,
        "total_bets": int(stats.get("total_bets", 0) or 0),
        "win_rate": float(stats.get("win_rate", 0.0) or 0.0),
        "roi": float(stats.get("roi", 0.0) or 0.0),
        "total_profit": float(stats.get("total_profit", 0.0) or 0.0),
        "total_wagered": float(stats.get("total_wagered", 0.0) or 0.0),
        "avg_edge": float(stats.get("avg_edge", 0.0) or 0.0),
        "avg_clv": float(stats.get("avg_clv", float("nan"))),
        "sharpe": float(compute_sharpe_ratio(bet_log)),
        "max_drawdown_pct": float(stats.get("max_drawdown_pct", 0.0) or 0.0),
        "final_bankroll": float(stats.get("bankroll", 0.0) or 0.0),
        "equal_stake_profit_units": float(equal_stake["equal_stake_profit_units"]),
        "equal_stake_roi": float(equal_stake["equal_stake_roi"]),
        **newbie,
    }


def _write_outputs(
    output_dir: Path,
    *,
    bundle_id: str,
    spec_name: str,
    min_train_test_fights: int,
    summary_df: pd.DataFrame,
    results: dict[str, dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    for strategy_name, result in results.items():
        bet_log = result["bet_log"]
        if not bet_log.empty:
            bet_log.to_csv(output_dir / f"{strategy_name}_bet_log.csv", index=False)

    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "production_bundle_id": bundle_id,
        "spec_name": spec_name,
        "min_train_test_fights": min_train_test_fights,
        "summary": summary_df.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline, min-2, and tiered newbie rules on walk-forward UFC predictions.",
    )
    parser.add_argument(
        "--features-path",
        default=str(PROCESSED_DATA_DIR / "features.csv"),
        help="Path to processed UFC features.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/newbie_rule_eval",
        help="Directory to write CSV/JSON outputs",
    )
    parser.add_argument(
        "--min-train-test-fights",
        type=int,
        default=0,
        help="Minimum UFC fights required in the walk-forward row universe before strategy filters (default: 0)",
    )
    parser.add_argument("--initial-bankroll", type=float, default=INITIAL_BANKROLL)
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE_THRESHOLD)
    parser.add_argument("--kelly-fraction", type=float, default=KELLY_FRACTION)
    parser.add_argument("--max-bet-fraction", type=float, default=MAX_BET_FRACTION)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    bundle = load_production_bundle()
    spec = resolve_named_training_spec(bundle.model_spec_name)
    features_path = Path(args.features_path)
    logger.info("Loading features from %s", features_path)
    features_df = pd.read_csv(features_path, parse_dates=["event_date"])

    logger.info(
        "Running walk-forward prediction pass for bundle %s (%s) with min_train_test_fights=%s",
        bundle.bundle_id,
        spec.name,
        args.min_train_test_fights,
    )
    comparison = run_walkforward_strategy_comparison(
        features_df,
        initial_bankroll=args.initial_bankroll,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly_fraction,
        max_bet_fraction=args.max_bet_fraction,
        strategies=[PRODUCTION_GATED_STRATEGY],
        write_artifacts=False,
        spec=spec,
        min_train_test_fights=args.min_train_test_fights,
    )
    predictions = comparison["predictions"]
    logger.info("Prepared %s walk-forward prediction rows", len(predictions))

    strategies = (
        replace(
            PRODUCTION_GATED_STRATEGY,
            name="baseline_min_3",
            newbie_rule=LEGACY_BASELINE_NEWBIE_RULE,
        ),
        replace(
            PRODUCTION_GATED_STRATEGY,
            name="threshold_min_2",
            newbie_rule=DEFAULT_NEWBIE_RULE,
        ),
        replace(
            PRODUCTION_GATED_STRATEGY,
            name="tiered_org_aware",
            newbie_rule=TIERED_ORG_AWARE_NEWBIE_RULE,
        ),
    )

    results: dict[str, dict] = {}
    for strategy in strategies:
        logger.info("Simulating strategy %s", strategy.name)
        results[strategy.name] = _simulate_backtest_predictions(
            predictions,
            strategy,
            initial_bankroll=args.initial_bankroll,
            min_edge=args.min_edge,
            kelly_fraction=args.kelly_fraction,
            max_bet_fraction=args.max_bet_fraction,
        )

    summary_df = pd.DataFrame(_summary_row(result) for result in results.values())
    order = {strategy.name: idx for idx, strategy in enumerate(strategies)}
    summary_df["sort_key"] = summary_df["strategy"].map(order)
    summary_df = summary_df.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    _write_outputs(
        output_dir,
        bundle_id=bundle.bundle_id,
        spec_name=spec.name,
        min_train_test_fights=args.min_train_test_fights,
        summary_df=summary_df,
        results=results,
    )

    print("\nNEWBIE RULE EVALUATION")
    print(summary_df.to_string(index=False))
    print(f"\nOutputs written to {output_dir}")


if __name__ == "__main__":
    main()
