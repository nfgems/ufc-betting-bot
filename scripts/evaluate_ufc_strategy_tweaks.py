"""Evaluate UFC decision-layer tweaks on the current production bundle.

Runs one walk-forward prediction pass for the active production spec, then
replays strategy simulations across:
  - alternate single-trader blend weights
  - alternate value-trader max-odds caps

This isolates decision-layer changes from model-training changes.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.model.production_bundle import load_production_bundle
from src.model.training_spec import resolve_named_training_spec
from src.strategy.backtest import (
    PRODUCTION_GATED_STRATEGY,
    _simulate_backtest_predictions,
    run_walkforward_strategy_comparison,
)

logger = logging.getLogger(__name__)


def _strategy_row(label: str, value: float, result: dict) -> dict[str, object]:
    stats = result["stats"]
    return {
        "label": label,
        "value": value,
        "total_bets": int(stats.get("total_bets", 0) or 0),
        "win_rate": float(stats.get("win_rate", 0.0) or 0.0),
        "roi": float(stats.get("roi", 0.0) or 0.0),
        "total_profit": float(stats.get("total_profit", 0.0) or 0.0),
        "avg_edge": float(stats.get("avg_edge", 0.0) or 0.0),
        "avg_clv": float(stats.get("avg_clv", float("nan"))),
        "max_drawdown_pct": float(stats.get("max_drawdown_pct", 0.0) or 0.0),
        "final_bankroll": float(stats.get("bankroll", 0.0) or 0.0),
    }


def _simulate_with_strategy(predictions: pd.DataFrame, *, blend_weight: float, max_decimal_odds: float | None) -> dict:
    strategy = replace(
        PRODUCTION_GATED_STRATEGY,
        name="production_gated_eval",
        blend_weight=blend_weight,
        max_decimal_odds=max_decimal_odds,
    )
    return _simulate_backtest_predictions(predictions, strategy)


def _write_outputs(
    output_dir: Path,
    *,
    production_bundle_id: str,
    spec_name: str,
    blend_df: pd.DataFrame,
    odds_cap_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    blend_df.to_csv(output_dir / "blend_weight_eval.csv", index=False)
    odds_cap_df.to_csv(output_dir / "odds_cap_eval.csv", index=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "production_bundle_id": production_bundle_id,
        "spec_name": spec_name,
        "blend_weights_tested": blend_df.to_dict(orient="records"),
        "odds_caps_tested": odds_cap_df.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate UFC strategy-layer tweaks on the current production bundle.")
    parser.add_argument(
        "--features-path",
        default=str(PROCESSED_DATA_DIR / "features.csv"),
        help="Path to processed UFC features.csv",
    )
    parser.add_argument(
        "--blend-weights",
        default="0.30,0.35,0.40,0.45,0.50",
        help="Comma-separated blend weights to test",
    )
    parser.add_argument(
        "--odds-caps",
        default="3.0,4.0,inf",
        help="Comma-separated decimal-odds caps to test (use 'inf' for no cap)",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/strategy_tweak_eval",
        help="Directory to write CSV/JSON outputs",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    bundle = load_production_bundle()
    spec = resolve_named_training_spec(bundle.model_spec_name)
    features_path = Path(args.features_path)
    logger.info("Loading features from %s", features_path)
    features_df = pd.read_csv(features_path, parse_dates=["event_date"])

    logger.info("Running walk-forward prediction pass for bundle %s (%s)", bundle.bundle_id, spec.name)
    comparison = run_walkforward_strategy_comparison(
        features_df,
        write_artifacts=False,
        spec=spec,
    )
    predictions = comparison["predictions"]
    logger.info("Prepared %s walk-forward prediction rows", len(predictions))

    blend_weights = [float(item.strip()) for item in args.blend_weights.split(",") if item.strip()]
    odds_caps: list[float | None] = []
    for item in args.odds_caps.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"inf", "none", "no_cap"}:
            odds_caps.append(None)
        else:
            odds_caps.append(float(token))

    baseline_blend = blend_weights[0] if blend_weights else PRODUCTION_GATED_STRATEGY.blend_weight
    blend_rows: list[dict[str, object]] = []
    for blend_weight in blend_weights:
        logger.info("Evaluating blend weight %.2f", blend_weight)
        result = _simulate_with_strategy(
            predictions,
            blend_weight=blend_weight,
            max_decimal_odds=PRODUCTION_GATED_STRATEGY.max_decimal_odds,
        )
        blend_rows.append(_strategy_row("blend_weight", blend_weight, result))

    odds_rows: list[dict[str, object]] = []
    for odds_cap in odds_caps:
        label_value = float("inf") if odds_cap is None else odds_cap
        logger.info("Evaluating value-trader odds cap %s", "inf" if odds_cap is None else f"{odds_cap:.2f}")
        result = _simulate_with_strategy(
            predictions,
            blend_weight=baseline_blend,
            max_decimal_odds=odds_cap,
        )
        odds_rows.append(_strategy_row("max_decimal_odds", label_value, result))

    blend_df = pd.DataFrame(blend_rows).sort_values("value").reset_index(drop=True)
    odds_cap_df = pd.DataFrame(odds_rows).sort_values("value").reset_index(drop=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    _write_outputs(
        output_dir,
        production_bundle_id=bundle.bundle_id,
        spec_name=spec.name,
        blend_df=blend_df,
        odds_cap_df=odds_cap_df,
    )

    print("\nBLEND WEIGHT EVALUATION")
    print(blend_df.to_string(index=False))
    print("\nODDS CAP EVALUATION")
    print(odds_cap_df.to_string(index=False))
    print(f"\nOutputs written to {output_dir}")


if __name__ == "__main__":
    main()
