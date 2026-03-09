"""
Model Lab — sandbox A/B testing framework for betting model variants.

Runs walk-forward backtests of model/strategy variants against the production
baseline. All output goes to logs/model_lab/ — no production files are modified.

Usage:
    python -m src.strategy.model_lab --variants baseline,blend_b_fix,temporal_sigmoid_cal
    python -m src.strategy.model_lab --all
    python -m src.strategy.model_lab --list
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    INITIAL_BANKROLL,
    MIN_EDGE_THRESHOLD,
    KELLY_FRACTION,
    MAX_BET_FRACTION,
    BLEND_WEIGHT,
    PROCESSED_DATA_DIR,
    LOGS_DIR,
    CONVICTION_MIN_MODEL_PROB,
    CONVICTION_MIN_NO_ODDS_PROB,
    MIN_FIGHTER_FIGHTS,
)
from src.strategy.value import (
    blend_probability,
    dynamic_blend_weight,
    implied_prob_to_decimal_odds,
    _passes_filters,
    calculate_closing_line_value,
)
from src.strategy.bankroll import BankrollManager
from src.strategy.backtest import _merge_historical_odds, _resolve_market_odds
from src.strategy.model_variants import (
    VariantConfig,
    train_variant_model,
    add_elo_momentum,
    add_strength_of_schedule,
    add_rematch_features,
    ALL_VARIANTS,
)
from src.strategy.lab_stats import (
    compare_variants,
    plot_comparison,
    compute_ece,
)
from src.features.build_features import (
    get_feature_columns,
    get_feature_columns_no_odds,
    build_features,
)

logger = logging.getLogger(__name__)

# Output directory — never writes to models/ or data/processed/
LAB_DIR = LOGS_DIR / "model_lab"
LAB_DIR.mkdir(parents=True, exist_ok=True)


def _predict_batch_with_model(
    features_df: pd.DataFrame,
    model_result: dict,
) -> pd.DataFrame:
    """
    Generate predictions using a pre-trained model result dict.

    Like src.model.predict.predict_batch but works with in-memory models
    (no disk I/O) and handles indicator columns from impute_with_indicators.
    """
    model = model_result["model"]
    feature_cols = model_result["feature_cols"]
    col_medians = model_result["col_medians"]
    n_indicator = model_result.get("n_indicator_cols", 0)

    X = features_df[feature_cols].values.copy()

    # Track which columns had NaNs (for indicator columns)
    indicator_cols = []
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        if n_indicator > 0:
            indicator = np.zeros(X.shape[0])
            indicator[mask] = 1.0
            indicator_cols.append(indicator)
        X[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

    if n_indicator > 0 and indicator_cols:
        X = np.column_stack([X] + indicator_cols[:n_indicator])

    proba = model.predict_proba(X)

    result = features_df.copy()
    result["prob_a"] = proba[:, 1]
    result["prob_b"] = proba[:, 0]
    return result


def run_variant_walkforward(
    features_df: pd.DataFrame,
    variant: VariantConfig,
    retrain_months: int = 6,
    initial_train_years: int = 5,
    initial_bankroll: float = INITIAL_BANKROLL,
) -> dict:
    """
    Run a walk-forward backtest for a single variant.

    Mirrors src.strategy.backtest.run_walkforward_backtest but uses
    variant-specific training, calibration, features, and strategy logic.
    """
    features_df = features_df.sort_values("event_date").copy()
    features_df = features_df.dropna(subset=["target"])

    feature_cols = get_feature_columns(features_df)
    no_odds_cols = get_feature_columns_no_odds(features_df)
    no_odds_cols = [c for c in no_odds_cols if c in features_df.columns]

    # Require minimum fighter experience
    if "a_num_fights" in features_df.columns and "b_num_fights" in features_df.columns:
        features_df = features_df[
            (features_df["a_num_fights"] >= 2) & (features_df["b_num_fights"] >= 2)
        ]

    dates = pd.to_datetime(features_df["event_date"])
    min_date = dates.min()
    max_date = dates.max()
    train_end = min_date + pd.DateOffset(years=initial_train_years)

    bankroll = BankrollManager(
        initial_bankroll=initial_bankroll,
        kelly_fraction=variant.kelly_fraction,
        max_bet_fraction=variant.max_bet_fraction,
    )

    all_bet_log = []
    bankroll_history = [initial_bankroll]
    fold_stats = []
    fold_num = 0

    # Collect all predictions for calibration metrics
    all_y_true = []
    all_y_prob = []

    blend_weight = variant.blend_weight

    while train_end < max_date:
        test_end = train_end + pd.DateOffset(months=retrain_months)
        if test_end > max_date:
            test_end = max_date + pd.Timedelta(days=1)

        train_mask = dates < train_end
        test_mask = (dates >= train_end) & (dates < test_end)

        train_df = features_df[train_mask]
        test_df = features_df[test_mask]

        if len(train_df) < 100 or len(test_df) < 5:
            train_end = test_end
            continue

        fold_num += 1
        logger.info(
            f"  [{variant.name}] Fold {fold_num}: "
            f"Train {len(train_df)}, Test {len(test_df)} "
            f"({train_end.date()} to {test_end.date()})"
        )

        # --- Feature selection (if max_features set) ---
        fold_feature_cols = feature_cols
        if variant.max_features:
            # Quick importance estimate: train a small XGBoost for feature ranking
            from xgboost import XGBClassifier
            X_quick = train_df[feature_cols].values.copy()
            y_quick = train_df["target"].values
            for i in range(X_quick.shape[1]):
                mask = np.isnan(X_quick[:, i])
                X_quick[mask, i] = 0.0
            quick_xgb = XGBClassifier(
                n_estimators=50, max_depth=3, random_state=42, use_label_encoder=False,
                eval_metric="logloss",
            )
            quick_xgb.fit(X_quick, y_quick)
            importance = dict(zip(feature_cols, quick_xgb.feature_importances_))
            top_features = sorted(importance, key=importance.get, reverse=True)[:variant.max_features]
            fold_feature_cols = top_features

        fold_no_odds_cols = [c for c in no_odds_cols if c in fold_feature_cols or c in no_odds_cols]
        if variant.max_features:
            from src.features.build_features import ODDS_FEATURE_NAMES
            fold_no_odds_cols = [c for c in fold_feature_cols if c not in ODDS_FEATURE_NAMES]

        # --- Train primary model ---
        model_result = train_variant_model(train_df, fold_feature_cols, variant)

        # --- Train no-odds model (always production config for agreement filter) ---
        no_odds_variant = VariantConfig(name="_no_odds", description="internal")
        no_odds_result = train_variant_model(train_df, fold_no_odds_cols, no_odds_variant)

        # --- Generate predictions ---
        predictions = _predict_batch_with_model(test_df, model_result)
        no_odds_preds = _predict_batch_with_model(test_df, no_odds_result)
        predictions["no_odds_prob_a"] = no_odds_preds["prob_a"]
        predictions["no_odds_prob_b"] = no_odds_preds["prob_b"]

        # Collect for calibration metrics
        valid_mask = predictions["target"].notna()
        all_y_true.extend(predictions.loc[valid_mask, "target"].values.tolist())
        all_y_prob.extend(predictions.loc[valid_mask, "prob_a"].values.tolist())

        # --- Merge historical odds ---
        predictions = _merge_historical_odds(predictions)

        # --- Resolve market odds ---
        try:
            predictions, odds_source = _resolve_market_odds(
                predictions, "a_fair_prob_avg", "b_fair_prob_avg"
            )
        except ValueError:
            train_end = test_end
            continue

        predictions = predictions.sort_values("event_date").reset_index(drop=True)

        fold_bets = 0
        fold_wins = 0

        # --- Betting loop ---
        for _, row in predictions.iterrows():
            if bankroll.is_stopped:
                break

            model_a = row["prob_a"]
            model_b = row["prob_b"]
            market_a = row["a_market_prob"]
            market_b = row["b_market_prob"]

            if pd.isna(market_a) or pd.isna(market_b):
                bankroll_history.append(bankroll.bankroll)
                continue

            actual_winner_is_a = row["target"] == 1
            no_odds_a = row.get("no_odds_prob_a")
            no_odds_b = row.get("no_odds_prob_b")

            # Dynamic blend weights
            dyn_weight_a = dynamic_blend_weight(model_a, market_a, no_odds_a, blend_weight)
            dyn_weight_b = dynamic_blend_weight(model_b, market_b, no_odds_b, blend_weight)

            # --- Blend (with optional bug fix for side B) ---
            if variant.use_independent_blend_b:
                raw_blend_a = blend_probability(model_a, market_a, dyn_weight_a)
                raw_blend_b = blend_probability(model_b, market_b, dyn_weight_b)
                total = raw_blend_a + raw_blend_b
                if total > 0:
                    blend_a = raw_blend_a / total
                    blend_b = raw_blend_b / total
                else:
                    blend_a = 0.5
                    blend_b = 0.5
            else:
                blend_a = blend_probability(model_a, market_a, dyn_weight_a)
                blend_b = 1.0 - blend_a

            edge_a = blend_a - market_a
            edge_b = blend_b - market_b

            # Line movement data
            line_movement = row.get("line_movement")
            line_is_sharp = row.get("line_is_sharp")
            line_steam_move = row.get("line_steam_move")
            if isinstance(line_movement, float) and np.isnan(line_movement):
                line_movement = None

            # Fighter experience
            a_fights = row.get("a_num_fights")
            b_fights = row.get("b_num_fights")
            if isinstance(a_fights, float) and not np.isnan(a_fights):
                a_fights = int(a_fights)
            elif not isinstance(a_fights, int):
                a_fights = None
            if isinstance(b_fights, float) and not np.isnan(b_fights):
                b_fights = int(b_fights)
            elif not isinstance(b_fights, int):
                b_fights = None

            min_edge = variant.min_edge

            if edge_a >= min_edge and edge_a >= edge_b and _passes_filters(
                blend_a, market_a, edge_a, row.get("fighter_a", "A"), no_odds_a,
                line_movement=line_movement, line_is_sharp=line_is_sharp,
                line_steam_move=line_steam_move, bet_side="a",
                a_num_fights=a_fights, b_num_fights=b_fights,
                edge_scaling_base=min_edge,
            ):
                odds = implied_prob_to_decimal_odds(market_a)
                bet_size = bankroll.kelly_bet_size(blend_a, odds)
                if bet_size > 0:
                    bet_idx = len(bankroll.history)
                    bankroll.place_bet(bet_size, row.get("fighter_a", "A"), odds, blend_a, market_a)
                    bankroll.settle_bet(bet_idx, won=actual_winner_is_a)
                    fold_bets += 1
                    if actual_winner_is_a:
                        fold_wins += 1

                    clv = np.nan
                    if pd.notna(row.get("closing_prob_a")):
                        clv = calculate_closing_line_value(market_a, row["closing_prob_a"])

                    all_bet_log.append({
                        "event_date": row.get("event_date"),
                        "fighter_a": row.get("fighter_a", ""),
                        "fighter_b": row.get("fighter_b", ""),
                        "bet_on": row.get("fighter_a", "A"),
                        "bet_side": "a",
                        "bet_size": bet_size,
                        "odds": odds,
                        "blended_prob": blend_a,
                        "market_prob": market_a,
                        "edge": edge_a,
                        "won": actual_winner_is_a,
                        "profit": bankroll.history[-1]["profit"],
                        "bankroll_after": bankroll.bankroll,
                        "clv": clv,
                        "fold": fold_num,
                    })

            elif edge_b >= min_edge and _passes_filters(
                blend_b, market_b, edge_b, row.get("fighter_b", "B"), no_odds_b,
                line_movement=line_movement, line_is_sharp=line_is_sharp,
                line_steam_move=line_steam_move, bet_side="b",
                a_num_fights=a_fights, b_num_fights=b_fights,
                edge_scaling_base=min_edge,
            ):
                odds = implied_prob_to_decimal_odds(market_b)
                bet_size = bankroll.kelly_bet_size(blend_b, odds)
                if bet_size > 0:
                    bet_idx = len(bankroll.history)
                    bankroll.place_bet(bet_size, row.get("fighter_b", "B"), odds, blend_b, market_b)
                    bankroll.settle_bet(bet_idx, won=not actual_winner_is_a)
                    fold_bets += 1
                    if not actual_winner_is_a:
                        fold_wins += 1

                    clv = np.nan
                    if pd.notna(row.get("closing_prob_b")):
                        clv = calculate_closing_line_value(market_b, row["closing_prob_b"])

                    all_bet_log.append({
                        "event_date": row.get("event_date"),
                        "fighter_a": row.get("fighter_a", ""),
                        "fighter_b": row.get("fighter_b", ""),
                        "bet_on": row.get("fighter_b", "B"),
                        "bet_side": "b",
                        "bet_size": bet_size,
                        "odds": odds,
                        "blended_prob": blend_b,
                        "market_prob": market_b,
                        "edge": edge_b,
                        "won": not actual_winner_is_a,
                        "profit": bankroll.history[-1]["profit"],
                        "bankroll_after": bankroll.bankroll,
                        "clv": clv,
                        "fold": fold_num,
                    })

            bankroll_history.append(bankroll.bankroll)

        fold_stats.append({
            "fold": fold_num,
            "train_end": str(train_end.date()),
            "test_size": test_mask.sum(),
            "bets": fold_bets,
            "wins": fold_wins,
            "win_rate": fold_wins / fold_bets if fold_bets > 0 else 0,
            "bankroll": bankroll.bankroll,
        })

        train_end = test_end

    # --- Compute stats ---
    stats = bankroll.get_stats()
    bet_log_df = pd.DataFrame(all_bet_log)

    # CLV stats
    if not bet_log_df.empty and "clv" in bet_log_df.columns:
        valid_clv = bet_log_df["clv"].dropna()
        if len(valid_clv) > 0:
            stats["avg_clv"] = valid_clv.mean()
            stats["median_clv"] = valid_clv.median()
            stats["pct_positive_clv"] = (valid_clv > 0).mean()

    # Calibration metrics
    if all_y_true and all_y_prob:
        from sklearn.metrics import brier_score_loss
        y_true_arr = np.array(all_y_true)
        y_prob_arr = np.array(all_y_prob)
        stats["brier_score"] = brier_score_loss(y_true_arr, y_prob_arr)
        stats["ece"] = compute_ece(y_true_arr, y_prob_arr)

    return {
        "stats": stats,
        "bet_log": bet_log_df,
        "bankroll_history": bankroll_history,
        "fold_stats": pd.DataFrame(fold_stats),
        "variant": variant,
    }


def run_experiment(
    variant_names: list[str],
    features_df: Optional[pd.DataFrame] = None,
    initial_bankroll: float = INITIAL_BANKROLL,
) -> dict:
    """
    Run A/B experiments for the specified variants.

    Always includes baseline as the first variant for comparison.

    Returns dict of {variant_name: backtest_result}.
    """
    # --- Load data ---
    if features_df is None:
        features_path = PROCESSED_DATA_DIR / "features.csv"
        if features_path.exists():
            logger.info(f"Loading features from {features_path}")
            features_df = pd.read_csv(features_path, parse_dates=["event_date"])
        else:
            logger.info("No cached features found. Building from Kaggle dataset...")
            from src.data.kaggle_loader import load_kaggle_dataset
            fights_df = load_kaggle_dataset()
            features_df = build_features(fights_df)

    # Ensure baseline is always first
    if "baseline" not in variant_names:
        variant_names = ["baseline"] + variant_names

    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LAB_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"MODEL LAB — Running {len(variant_names)} variants")
    logger.info(f"Output: {run_dir}")
    logger.info(f"{'='*60}\n")

    for name in variant_names:
        if name not in ALL_VARIANTS:
            logger.warning(f"Unknown variant: {name}. Skipping.")
            continue

        variant = ALL_VARIANTS[name]()
        logger.info(f"\n--- {variant.name}: {variant.description} ---")

        # Build features with variant's custom builder if specified
        variant_features = features_df
        if variant.feature_builder_fn is not None:
            logger.info(f"  Using custom feature builder for {variant.name}")
            from src.data.kaggle_loader import load_kaggle_dataset
            fights_df = load_kaggle_dataset()
            variant_features = variant.feature_builder_fn(fights_df)

        # Add extra features based on variant config
        if variant.name == "elo_momentum":
            variant_features = add_elo_momentum(variant_features)
        if variant.add_strength_of_schedule:
            variant_features = add_strength_of_schedule(variant_features)
        if variant.add_rematch_features:
            variant_features = add_rematch_features(variant_features)

        try:
            result = run_variant_walkforward(
                variant_features,
                variant,
                initial_bankroll=initial_bankroll,
            )
            results[name] = result

            s = result["stats"]
            logger.info(
                f"  Result: ROI {s.get('roi', 0):+.1%}, "
                f"Win rate {s.get('win_rate', 0):.1%}, "
                f"Bets {s.get('total_bets', 0)}, "
                f"Profit ${s.get('total_profit', 0):+.2f}"
            )
            if "brier_score" in s:
                logger.info(f"  Brier: {s['brier_score']:.4f}, ECE: {s.get('ece', 0):.4f}")
            if "avg_clv" in s:
                logger.info(f"  Avg CLV: {s['avg_clv']:+.2%}")

            # Save bet log
            if not result["bet_log"].empty:
                result["bet_log"].to_csv(run_dir / f"{name}_bet_log.csv", index=False)

        except Exception as e:
            logger.error(f"  FAILED: {e}", exc_info=True)

    # --- Comparison ---
    if len(results) >= 2:
        logger.info(f"\n{'='*60}")
        logger.info("VARIANT COMPARISON")
        logger.info(f"{'='*60}")

        comparison = compare_variants(results)
        logger.info(f"\n{comparison.to_string(index=False)}")

        comparison.to_csv(run_dir / "comparison.csv", index=False)
        plot_comparison(results, save_path=str(run_dir / "comparison.png"))

        logger.info(f"\nResults saved to {run_dir}")
    elif len(results) == 1:
        logger.info("Only one variant ran. Need at least 2 for comparison.")

    return results


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LAB_DIR / "model_lab.log"),
        ],
    )

    parser = argparse.ArgumentParser(description="Model Lab — A/B test model variants")
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated list of variant names to test",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available variants",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available variants and exit",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=INITIAL_BANKROLL,
        help=f"Starting bankroll (default: ${INITIAL_BANKROLL:.2f})",
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable variants:")
        print(f"{'Name':<25} Description")
        print("-" * 70)
        for name, factory in ALL_VARIANTS.items():
            v = factory()
            print(f"  {name:<23} {v.description}")
        return

    if args.all:
        variant_names = list(ALL_VARIANTS.keys())
    elif args.variants:
        variant_names = [v.strip() for v in args.variants.split(",")]
    else:
        # Default: run bug fixes + top improvements
        variant_names = [
            "baseline",
            "blend_b_fix",
            "conviction_ev_fix",
            "temporal_sigmoid_cal",
            "missing_indicators",
            "all_bug_fixes",
            "combined_best",
        ]

    run_experiment(variant_names, initial_bankroll=args.bankroll)


if __name__ == "__main__":
    main()
