"""
Track C batch (R2): contract-extension experiments on one shared rebuild.

Rebuilds the feature matrix from the cleaned fights table (the rebuild
contains the new E10 diff columns and E13 durability columns), then runs the
spec-driven walk-forward for the control and each candidate ON THE SAME
REBUILT FRAME — the pinned production features.csv is never touched.

Arms:
  full_live_contract_v6_tuned       control (same rebuild, so deltas isolate
                                    the new columns, not rebuild drift)
  full_live_contract_v6_grapdef     E10: opponent-allowed rolling stats
  full_live_contract_v6_durability  E13: loss-method decomposition

Usage:
    python scripts/track_c_batch.py --out logs/track_c_batch
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

logger = logging.getLogger("track_c_batch")

DEFAULT_SPECS = [
    "full_live_contract_v6_tuned",
    "full_live_contract_v6_grapdef",
    "full_live_contract_v6_durability",
    "full_live_contract_v6_plus_rankings",
]


def _enrich_rankings(fights: pd.DataFrame) -> pd.DataFrame:
    """Fill rank gaps point-in-time from the historical archive (E16)."""
    from src.data.ufc_refresh import _fight_key_series, _historical_rankings_overlay

    fights = fights.copy()
    fights["fight_key"] = _fight_key_series(fights)
    overlay = _historical_rankings_overlay(fights)
    if overlay.empty:
        logger.warning("Rankings overlay empty — historical archive missing?")
        return fights.drop(columns=["fight_key"], errors="ignore")
    fights = fights.merge(overlay, on="fight_key", how="left")
    for column in ("a_wc_rank", "b_wc_rank", "a_pfp_rank", "b_pfp_rank"):
        overlay_column = f"{column}__historical_overlay"
        if column in fights.columns:
            fights[column] = fights[column].combine_first(fights[overlay_column])
        else:
            fights[column] = fights[overlay_column]
    fights = fights.drop(
        columns=[c for c in fights.columns if c.endswith("__historical_overlay")]
        + ["fight_key"],
        errors="ignore",
    )
    coverage = pd.to_numeric(fights["a_wc_rank"], errors="coerce").notna().mean()
    logger.info("Rankings enrichment done: a_wc_rank coverage %.1f%%", 100 * coverage)
    return fights


def _load_or_rebuild_features(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        logger.info("Loading rebuilt features from %s", cache_path)
        return pd.read_csv(cache_path, parse_dates=["event_date"])

    from src.features.build_features import build_features

    fights = pd.read_csv(
        PROCESSED_DATA_DIR / "fights_cleaned.csv", parse_dates=["event_date"]
    )
    fights = _enrich_rankings(fights)
    logger.info("Rebuilding features from %d fights (this takes a while)...", len(fights))
    features_df = build_features(fights)

    # The rebuilt frame lacks the consolidated odds overlays applied by
    # ufc_refresh; graft the market columns from the pinned features.csv so
    # odds features match production exactly.
    pinned = pd.read_csv(PROCESSED_DATA_DIR / "features.csv", parse_dates=["event_date"])
    odds_cols = [
        c for c in pinned.columns
        if ("implied_prob" in c or "odds_prob" in c or c in ("a_odds", "b_odds"))
    ]
    key = ["event_date", "fighter_a", "fighter_b"]
    graft = pinned[key + [c for c in odds_cols if c in pinned.columns]]
    features_df = features_df.drop(
        columns=[c for c in odds_cols if c in features_df.columns]
    ).merge(graft, on=key, how="left")

    # Same for target (winner-derived label) when absent.
    if "target" not in features_df.columns and "target" in pinned.columns:
        features_df = features_df.merge(pinned[key + ["target"]], on=key, how="left")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(cache_path, index=False)
    logger.info("Cached rebuilt features to %s (%d rows x %d cols)",
                cache_path, len(features_df), len(features_df.columns))
    return features_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", nargs="+", default=DEFAULT_SPECS)
    parser.add_argument("--out", default="logs/track_c_batch")
    parser.add_argument("--features-cache", default="logs/track_c_features.csv")
    parser.add_argument("--retrain-months", type=int, default=6)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_df = _load_or_rebuild_features(Path(args.features_cache))

    rows = []
    for spec_name in args.specs:
        logger.info("=== Track C arm: %s ===", spec_name)
        spec = resolve_named_training_spec(spec_name)
        missing = [c for c in spec.feature_cols if c not in features_df.columns]
        if missing:
            logger.error("%s: %d contract columns missing from rebuild: %s",
                         spec_name, len(missing), missing[:8])
            continue
        result = run_walkforward_strategy_comparison(
            features_df.copy(),
            spec=spec,
            retrain_months=args.retrain_months,
            bet_start_date=TRAIN_CUTOFF_DATE,
            write_artifacts=False,
        )
        predictive = result.get("predictive_metrics", {})
        models = predictive.get("models", {})
        summary = result.get("summary")

        row = {"spec": spec_name}
        row.update({f"model_{k}": v for k, v in models.get("xgboost", {}).items()})
        blend = predictive.get("blend", {}).get("overall", {})
        row.update({f"blend_{k}": v for k, v in blend.items()})
        if summary is not None and hasattr(summary, "empty") and not summary.empty:
            gated = summary[summary["strategy"] == "production_gated"]
            if not gated.empty:
                g = gated.iloc[0]
                for col in ("total_bets", "win_rate", "roi", "total_profit",
                            "avg_clv", "max_drawdown"):
                    if col in gated.columns:
                        row[f"strategy_{col}"] = g[col]
        rows.append(row)

        with (out_dir / f"{spec_name}_predictive.json").open("w", encoding="utf-8") as fh:
            json.dump(predictive, fh, indent=2, default=str)
        for strat_name, strat_result in result.get("strategy_results", {}).items():
            bet_log = strat_result.get("bet_log")
            if isinstance(bet_log, pd.DataFrame) and not bet_log.empty:
                bet_log.to_csv(out_dir / f"{spec_name}_{strat_name}_bets.csv", index=False)

        table = pd.DataFrame(rows)
        table.to_csv(out_dir / "track_c_summary.csv", index=False)
        print(f"\n--- progress: {len(rows)}/{len(args.specs)} arms ---")
        print(table.to_string(index=False))

    print(f"\nSaved Track C summary to {out_dir / 'track_c_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
