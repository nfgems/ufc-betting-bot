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
import hashlib
import json
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import check_production_refit_contract as contract_gate
from src import config as app_config
from src.config import PROCESSED_DATA_DIR, TRAIN_CUTOFF_DATE
from src.model.training_spec import resolve_named_training_spec
from src.strategy.backtest import (
    BacktestExecutionConfig,
    BacktestStrategyConfig,
    run_walkforward_strategy_comparison,
)
from src.strategy.value import NewbieRuleConfig

logger = logging.getLogger("track_c_batch")

DEFAULT_SPECS = [
    "full_live_contract_v6_tuned",
    "full_live_contract_v6_grapdef",
    "full_live_contract_v6_durability",
    "full_live_contract_v6_plus_rankings",
]

_EVALUATION_INDEX_COLUMNS = [
    "event_date",
    "fighter_a",
    "fighter_b",
    "target",
    "fold",
    "train_end",
    "test_end",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_evaluation_index(predictions: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _EVALUATION_INDEX_COLUMNS if column not in predictions.columns]
    if missing or predictions.empty:
        raise ValueError(
            "walk-forward predictions cannot be fingerprinted; missing columns: "
            + ", ".join(missing)
        )
    result = predictions[_EVALUATION_INDEX_COLUMNS].copy()
    for column in ("event_date", "train_end", "test_end"):
        result[column] = pd.to_datetime(result[column], errors="raise").dt.strftime("%Y-%m-%d")
    result["target"] = pd.to_numeric(result["target"], errors="raise").astype(int)
    result["fold"] = pd.to_numeric(result["fold"], errors="raise").astype(int)
    result["fighter_a"] = result["fighter_a"].astype(str)
    result["fighter_b"] = result["fighter_b"].astype(str)
    if result.duplicated(["event_date", "fighter_a", "fighter_b"]).any():
        raise ValueError("walk-forward predictions contain duplicate fight identities")
    return result.sort_values(_EVALUATION_INDEX_COLUMNS, kind="stable").reset_index(drop=True)


def _evaluation_sample_sha256(predictions: pd.DataFrame) -> str:
    sample = _canonical_evaluation_index(predictions)
    payload = sample.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _legacy_protocol(
    *,
    retrain_months: int,
    model_seed: int | None,
    odds_noise_seed: int | None,
    execution_mode: str,
    entry_offset_days: float | None,
    entry_offset_for_features: bool,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "bet_start_date": TRAIN_CUTOFF_DATE,
        "retrain_months": retrain_months,
        "model_seed": model_seed,
        "odds_noise_seed": odds_noise_seed,
        "execution_mode": execution_mode,
        "entry_offset_days": entry_offset_days,
        "entry_offset_for_features": entry_offset_for_features,
    }


def _scheduled_strategy(policy: dict) -> tuple[BacktestStrategyConfig, BacktestExecutionConfig]:
    evaluation = policy["evaluation"]
    assumptions = evaluation["execution_assumptions"]
    newbie_rule = NewbieRuleConfig(
        name=f"hard_skip_min_{evaluation['minimum_fighter_fights']}",
        mode=evaluation["newbie_mode"],
        min_fighter_fights=int(evaluation["minimum_fighter_fights"]),
    )
    strategy = BacktestStrategyConfig(
        name=evaluation["strategy_name"],
        primary_model="xgboost",
        selection_mode="blended",
        require_agreement=bool(evaluation["require_agreement"]),
        agreement_model=evaluation["agreement_model"],
        use_agreement_for_blend=True,
        blend_weight=float(evaluation["blend_weight"]),
        max_decimal_odds=float(evaluation["max_decimal_odds"]),
        newbie_rule=newbie_rule,
    )
    execution = BacktestExecutionConfig(
        mode=evaluation["execution_mode"],
        min_book_liquidity=float(assumptions["min_book_liquidity"]),
        max_slippage=float(assumptions["max_slippage"]),
        max_bet_vs_book_ratio=float(assumptions["max_bet_vs_book_ratio"]),
        assumed_half_spread=float(assumptions["assumed_half_spread"]),
        synthetic_liquidity_floor=float(assumptions["synthetic_liquidity_floor"]),
        synthetic_liquidity_peak=float(assumptions["synthetic_liquidity_peak"]),
        synthetic_price_step=float(assumptions["synthetic_price_step"]),
        synthetic_depth_notional_shares=tuple(
            float(value) for value in assumptions["synthetic_depth_notional_shares"]
        ),
    )
    return strategy, execution


def _validate_scheduled_runtime_constants(policy: dict) -> None:
    """Fail if implicit strategy globals drift from the reviewed policy."""

    evaluation = policy["evaluation"]
    expected = {
        "MODEL_AGREEMENT_MIN_EDGE": evaluation["model_agreement_min_edge"],
        "MIN_MODEL_PROB": evaluation["min_model_probability"],
        "BLEND_WEIGHT_MIN": evaluation["dynamic_blend_min"],
        "BLEND_WEIGHT_MAX": evaluation["dynamic_blend_max"],
        "BLEND_CONFIDENCE_THRESHOLD": evaluation["dynamic_blend_confidence"],
        "BLEND_AGREEMENT_BOOST": evaluation["dynamic_blend_agreement_boost"],
        "LINE_MOVEMENT_FILTER": evaluation["line_movement_filter"],
        "MIN_FIGHTER_FIGHTS": evaluation["minimum_fighter_fights"],
    }
    mismatches = []
    for name, expected_value in expected.items():
        actual = getattr(app_config, name)
        if actual != expected_value:
            mismatches.append(f"{name}={actual!r}, policy={expected_value!r}")
    if mismatches:
        raise ValueError(
            "scheduled strategy constants drifted from fixed policy: " + "; ".join(mismatches)
        )


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", nargs="+", default=None)
    parser.add_argument("--out", default="logs/track_c_batch")
    parser.add_argument("--features-cache", default="logs/track_c_features.csv")
    parser.add_argument("--retrain-months", type=int, default=None)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--odds-noise-seed", type=int)
    parser.add_argument("--execution-mode", choices=("legacy", "realistic"), default=None)
    parser.add_argument("--entry-offset-days", type=float, default=None)
    parser.add_argument("--entry-offset-for-features", action="store_true")
    parser.add_argument(
        "--scheduled-policy",
        type=Path,
        help=(
            "Run exactly the fixed one-arm scheduled health protocol. In this mode "
            "specs, seeds, cadence, and execution settings come only from the policy."
        ),
    )
    args = parser.parse_args(argv)

    if args.entry_offset_for_features and args.entry_offset_days is None and args.scheduled_policy is None:
        parser.error("--entry-offset-for-features requires --entry-offset-days")
    if (args.model_seed is None) != (args.odds_noise_seed is None):
        parser.error("--model-seed and --odds-noise-seed must be supplied together")

    policy: dict | None = None
    policy_sha256 = ""
    scheduled_strategy: BacktestStrategyConfig | None = None
    scheduled_execution: BacktestExecutionConfig | None = None
    if args.scheduled_policy is not None:
        conflicting = []
        for name, value in (
            ("--specs", args.specs),
            ("--retrain-months", args.retrain_months),
            ("--model-seed", args.model_seed),
            ("--odds-noise-seed", args.odds_noise_seed),
            ("--execution-mode", args.execution_mode),
            ("--entry-offset-days", args.entry_offset_days),
            ("--entry-offset-for-features", args.entry_offset_for_features or None),
        ):
            if value is not None:
                conflicting.append(name)
        if conflicting:
            parser.error(
                "--scheduled-policy cannot be combined with protocol overrides: "
                + ", ".join(conflicting)
            )
        policy = contract_gate.load_policy(args.scheduled_policy)
        registry_errors, evaluation_spec, _fullfit_spec = contract_gate.validate_policy_registry(policy)
        if registry_errors or evaluation_spec is None:
            raise ValueError("fixed scheduled policy registry check failed: " + "; ".join(registry_errors))
        _validate_scheduled_runtime_constants(policy)
        policy_sha256 = contract_gate.file_sha256(args.scheduled_policy)
        evaluation = policy["evaluation"]
        specs = [policy["contract"]["evaluation_spec_name"]]
        retrain_months = int(evaluation["retrain_months"])
        model_seed = int(evaluation["model_seed"])
        odds_noise_seed = int(evaluation["odds_noise_seed"])
        execution_mode = str(evaluation["execution_mode"])
        entry_offset_days = float(evaluation["entry_offset_days"])
        entry_offset_for_features = bool(evaluation["entry_offset_for_features"])
        protocol = contract_gate.scheduled_protocol(policy)
        scheduled_strategy, scheduled_execution = _scheduled_strategy(policy)
    else:
        specs = args.specs or list(DEFAULT_SPECS)
        retrain_months = int(args.retrain_months if args.retrain_months is not None else 6)
        model_seed = args.model_seed
        odds_noise_seed = args.odds_noise_seed
        execution_mode = args.execution_mode or "legacy"
        entry_offset_days = args.entry_offset_days
        entry_offset_for_features = bool(args.entry_offset_for_features)
        protocol = _legacy_protocol(
            retrain_months=retrain_months,
            model_seed=model_seed,
            odds_noise_seed=odds_noise_seed,
            execution_mode=execution_mode,
            entry_offset_days=entry_offset_days,
            entry_offset_for_features=entry_offset_for_features,
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = Path(args.features_cache)
    if policy is not None and not features_path.is_file():
        raise FileNotFoundError(
            f"scheduled feature snapshot must already exist; refusing implicit rebuild: {features_path}"
        )
    features_df = _load_or_rebuild_features(features_path)
    if not features_path.is_file():
        raise FileNotFoundError(f"feature cache was not persisted for fingerprinting: {features_path}")
    features_sha256 = _sha256_file(features_path)
    protocol_sha256 = contract_gate.canonical_json_sha256(protocol)

    rows = []
    for spec_name in specs:
        logger.info("=== Track C arm: %s ===", spec_name)
        registered_spec = resolve_named_training_spec(spec_name)
        spec_payload_sha256 = contract_gate.canonical_json_sha256(asdict(registered_spec))
        spec = registered_spec
        if model_seed is not None:
            spec = replace(
                spec,
                xgb_params={**(spec.xgb_params or {}), "random_state": int(model_seed)},
                odds_noise_seed=int(odds_noise_seed),
            )
        effective_model_seed = int((spec.xgb_params or {}).get("random_state", 42))
        effective_noise_seed = int(
            spec.odds_noise_seed if spec.odds_noise_seed is not None else effective_model_seed
        )
        missing = [column for column in spec.feature_cols if column not in features_df.columns]
        if missing:
            message = (
                f"{spec_name}: {len(missing)} contract columns missing from rebuild: "
                f"{missing[:8]}"
            )
            if policy is not None:
                raise ValueError(message)
            logger.error(message)
            continue

        run_kwargs = {
            "spec": spec,
            "retrain_months": retrain_months,
            "bet_start_date": TRAIN_CUTOFF_DATE,
            "write_artifacts": False,
            "execution_mode": execution_mode,
            "entry_offset_days": entry_offset_days,
            "entry_offset_for_features": entry_offset_for_features,
        }
        if policy is not None:
            evaluation = policy["evaluation"]
            run_kwargs.update(
                {
                    "initial_train_years": int(evaluation["initial_train_years"]),
                    "min_train_test_fights": int(evaluation["min_train_test_fights"]),
                    "min_edge": float(evaluation["min_edge"]),
                    "kelly_fraction": float(evaluation["kelly_fraction"]),
                    "max_bet_fraction": float(evaluation["max_bet_fraction"]),
                    "initial_bankroll": float(evaluation["initial_bankroll"]),
                    "bet_start_date": str(evaluation["bet_start_date"]),
                    "strategies": (scheduled_strategy,),
                    "execution_config": scheduled_execution,
                }
            )
        result = run_walkforward_strategy_comparison(features_df.copy(), **run_kwargs)
        predictive = result.get("predictive_metrics", {})
        models = predictive.get("models", {})
        summary = result.get("summary")
        predictions = result.get("predictions")
        if not isinstance(predictions, pd.DataFrame):
            raise ValueError(f"{spec_name}: walk-forward result has no predictions DataFrame")
        evaluation_index = _canonical_evaluation_index(predictions)
        evaluation_sample_sha256 = _evaluation_sample_sha256(evaluation_index)
        evaluation_index.to_csv(
            out_dir / f"{spec_name}_evaluation_index.csv",
            index=False,
            lineterminator="\n",
        )

        row = {
            "spec": spec_name,
            "spec_payload_sha256": spec_payload_sha256,
            "policy_sha256": policy_sha256,
            "features_sha256": features_sha256,
            "protocol_sha256": protocol_sha256,
            "evaluation_sample_sha256": evaluation_sample_sha256,
            "model_seed": effective_model_seed,
            "odds_noise_seed": effective_noise_seed,
            "execution_mode": execution_mode,
            "entry_offset_days": entry_offset_days,
            "entry_offset_for_features": entry_offset_for_features,
        }
        evaluation_window = predictive.get("evaluation_window", {})
        for key in ("start_date", "end_date", "n_fights", "n_folds"):
            row[f"evaluation_{key}"] = evaluation_window.get(key)
        row.update({f"model_{key}": value for key, value in models.get("xgboost", {}).items()})
        blend = predictive.get("blend", {}).get("overall", {})
        row.update({f"blend_{key}": value for key, value in blend.items()})
        if summary is not None and hasattr(summary, "empty") and not summary.empty:
            gated = summary[summary["strategy"] == "production_gated"]
            if not gated.empty:
                gated_row = gated.iloc[0]
                for column in (
                    "total_bets",
                    "win_rate",
                    "total_wagered",
                    "roi",
                    "total_profit",
                    "avg_clv",
                    "max_drawdown_pct",
                    "execution_mode",
                ):
                    if column in gated.columns:
                        row[f"strategy_{column}"] = gated_row[column]
                if "strategy_max_drawdown_pct" not in row and "max_drawdown" in gated.columns:
                    row["strategy_max_drawdown_pct"] = gated_row["max_drawdown"]
        rows.append(row)

        with (out_dir / f"{spec_name}_predictive.json").open("w", encoding="utf-8") as handle:
            json.dump(predictive, handle, indent=2, default=str)
        production_log_written = False
        for strategy_name, strategy_result in result.get("strategy_results", {}).items():
            bet_log = strategy_result.get("bet_log")
            if isinstance(bet_log, pd.DataFrame):
                bet_log.to_csv(
                    out_dir / f"{spec_name}_{strategy_name}_bets.csv",
                    index=False,
                    lineterminator="\n",
                )
                production_log_written = production_log_written or strategy_name == "production_gated"
        if not production_log_written:
            pd.DataFrame().to_csv(
                out_dir / f"{spec_name}_production_gated_bets.csv",
                index=False,
                lineterminator="\n",
            )

        table = pd.DataFrame(rows)
        table.to_csv(out_dir / "track_c_summary.csv", index=False, lineterminator="\n")
        print(f"\n--- progress: {len(rows)}/{len(specs)} arms ---")
        print(table.to_string(index=False))

    if policy is not None and len(rows) != 1:
        raise ValueError("scheduled policy mode must produce exactly one completed arm")
    print(f"\nSaved Track C summary to {out_dir / 'track_c_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
