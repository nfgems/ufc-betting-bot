"""
Backtesting engine — simulates the betting strategy on historical data.

Supports three backtesting modes:
  1. Historical API odds: Uses real opening/closing odds from The Odds API backfill
  2. Kaggle odds: Uses closing odds from the dataset (fallback)
  3. No-odds baseline: Tests model without any odds features

Also computes Closing Line Value (CLV) when both opening and closing odds are available.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.strategy.value import (
    find_value_bets, implied_prob_to_decimal_odds, _passes_filters,
    calculate_closing_line_value, blend_probability, dynamic_blend_weight,
)
from src.strategy.bankroll import BankrollManager
from src.model.predict import predict_batch
from src.model.train import load_model
from src.config import (
    MIN_EDGE_THRESHOLD,
    KELLY_FRACTION,
    MAX_BET_FRACTION,
    INITIAL_BANKROLL,
    BLEND_WEIGHT,
    LOGS_DIR,
    TRAIN_CUTOFF_DATE,
)

logger = logging.getLogger(__name__)


def _resolve_market_odds(
    predictions: pd.DataFrame,
    market_prob_col_a: str,
    market_prob_col_b: str,
) -> tuple[pd.DataFrame, str]:
    """
    Resolve market odds columns for backtesting.

    Priority:
      1. opening_prob_a/b from historical API backfill (realistic bet timing)
      2. a_fair_prob_avg/b_fair_prob_avg from live odds
      3. a_implied_prob/b_implied_prob from Kaggle closing odds (fallback)

    Returns (predictions_df, source_description).
    """
    # Check for historical backfill data (best option)
    if "opening_prob_a" in predictions.columns:
        predictions["a_market_prob"] = predictions["opening_prob_a"]
        predictions["b_market_prob"] = predictions["opening_prob_b"]
        return predictions, "historical_api_opening"

    # Check for requested columns
    if market_prob_col_a in predictions.columns:
        predictions["a_market_prob"] = predictions[market_prob_col_a]
        predictions["b_market_prob"] = predictions[market_prob_col_b]
        return predictions, f"column:{market_prob_col_a}"

    # Fallback: Kaggle closing odds (a_implied_prob)
    if "a_implied_prob" in predictions.columns:
        # Convert American implied prob to fair prob (remove vig)
        a_imp = predictions["a_implied_prob"]
        b_imp = predictions["b_implied_prob"]
        total = a_imp + b_imp
        predictions["a_market_prob"] = a_imp / total
        predictions["b_market_prob"] = b_imp / total
        logger.warning(
            "Using Kaggle CLOSING odds as market line. "
            "This is less realistic than opening odds — the model sees these as features. "
            "Run 'backfill-odds' to get historical opening odds for better backtesting."
        )
        return predictions, "kaggle_closing"

    raise ValueError(
        "No market odds found. Need either historical backfill data, "
        "a_fair_prob_avg columns, or a_implied_prob from Kaggle dataset."
    )


def _merge_historical_odds(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Merge historical backfill odds into predictions if available."""
    from src.data.historical_backfill import load_historical_odds, compute_line_movement_from_backfill

    hist = load_historical_odds()
    if hist.empty:
        return predictions

    logger.info(f"Found {len(hist)} historical odds records for backtest enrichment")

    # Get opening odds (largest offset = earliest snapshot)
    opening = hist.loc[hist.groupby(["event_date", "fighter_a", "fighter_b"])["offset_days"].idxmax()]
    opening = opening.rename(columns={
        "a_fair_prob": "opening_prob_a",
        "b_fair_prob": "opening_prob_b",
        "a_decimal_odds": "opening_odds_a",
        "b_decimal_odds": "opening_odds_b",
    })[["event_date", "fighter_a", "fighter_b", "opening_prob_a", "opening_prob_b",
        "opening_odds_a", "opening_odds_b"]]

    # Get closing odds (smallest offset = latest snapshot)
    closing = hist.loc[hist.groupby(["event_date", "fighter_a", "fighter_b"])["offset_days"].idxmin()]
    closing = closing.rename(columns={
        "a_fair_prob": "closing_prob_a",
        "b_fair_prob": "closing_prob_b",
        "a_decimal_odds": "closing_odds_a",
        "b_decimal_odds": "closing_odds_b",
    })[["event_date", "fighter_a", "fighter_b", "closing_prob_a", "closing_prob_b",
        "closing_odds_a", "closing_odds_b"]]

    # Get line movement features
    movement = compute_line_movement_from_backfill(hist)

    # Normalize event_date for merging
    predictions["event_date_str"] = pd.to_datetime(predictions["event_date"]).dt.strftime("%Y-%m-%d")
    opening["event_date_str"] = opening["event_date"].astype(str)
    closing["event_date_str"] = closing["event_date"].astype(str)

    # Merge opening odds
    merged = predictions.merge(
        opening.drop(columns=["event_date"]),
        on=["event_date_str", "fighter_a", "fighter_b"],
        how="left",
    )

    # Merge closing odds
    merged = merged.merge(
        closing.drop(columns=["event_date"]),
        on=["event_date_str", "fighter_a", "fighter_b"],
        how="left",
    )

    # Merge line movement
    if not movement.empty:
        movement["event_date_str"] = movement["event_date"].astype(str)
        move_cols = ["event_date_str", "fighter_a", "fighter_b",
                     "line_movement", "line_abs_movement", "line_is_sharp",
                     "line_steam_move", "line_direction_toward_a", "line_direction_toward_b"]
        move_cols = [c for c in move_cols if c in movement.columns]
        merged = merged.merge(
            movement[move_cols],
            on=["event_date_str", "fighter_a", "fighter_b"],
            how="left",
        )

    merged = merged.drop(columns=["event_date_str"])

    matched = merged["opening_prob_a"].notna().sum()
    logger.info(f"Matched {matched}/{len(predictions)} fights with historical opening odds")

    return merged


def run_backtest(
    test_df: pd.DataFrame,
    model_name: str = "xgboost",
    model_result: Optional[dict] = None,
    initial_bankroll: float = INITIAL_BANKROLL,
    min_edge: float = MIN_EDGE_THRESHOLD,
    kelly_fraction: float = KELLY_FRACTION,
    max_bet_fraction: float = MAX_BET_FRACTION,
    market_prob_col_a: str = "a_fair_prob_avg",
    market_prob_col_b: str = "b_fair_prob_avg",
    use_historical_odds: bool = True,
    blend_weight: float = BLEND_WEIGHT,
) -> dict:
    """
    Run a full backtest on historical fight data.

    Uses blended model-market probabilities and requires both models to agree.

    Args:
        test_df: DataFrame with features AND market probabilities
        model_name: which model to use for predictions
        model_result: pre-loaded model (if None, loads from disk)
        initial_bankroll: starting bankroll in USD
        min_edge: minimum edge threshold to place a bet
        kelly_fraction: fraction of Kelly criterion to use
        max_bet_fraction: maximum fraction of bankroll per bet
        market_prob_col_a: column name for fighter A's market probability
        market_prob_col_b: column name for fighter B's market probability
        use_historical_odds: if True, try to merge backfilled historical odds
        blend_weight: model weight in market-model blend (default from config)

    Returns dict with backtest results and stats.
    """
    if model_result is None:
        model_result = load_model(model_name)

    # Generate predictions from primary model
    predictions = predict_batch(test_df, model_name=model_name, model_result=model_result)

    # Generate no-odds model predictions for agreement filter
    try:
        no_odds_result = load_model("xgboost_no_odds")
        no_odds_preds = predict_batch(test_df, model_name="xgboost_no_odds", model_result=no_odds_result)
        predictions["no_odds_prob_a"] = no_odds_preds["prob_a"]
        predictions["no_odds_prob_b"] = no_odds_preds["prob_b"]
        logger.info("Loaded no-odds model for agreement filter")
    except FileNotFoundError:
        logger.warning("No-odds model not found. Skipping agreement filter.")

    # Merge historical odds if available
    if use_historical_odds:
        predictions = _merge_historical_odds(predictions)

    # Resolve market odds
    predictions, odds_source = _resolve_market_odds(
        predictions, market_prob_col_a, market_prob_col_b
    )
    logger.info(f"Market odds source: {odds_source}")

    # Process fights chronologically
    predictions = predictions.sort_values("event_date").reset_index(drop=True)

    bankroll = BankrollManager(
        initial_bankroll=initial_bankroll,
        kelly_fraction=kelly_fraction,
        max_bet_fraction=max_bet_fraction,
        auto_detect_balance=False,
    )

    bankroll_history = [initial_bankroll]
    bet_log = []

    for _, row in predictions.iterrows():
        if bankroll.is_stopped:
            logger.warning("Stop-loss triggered. Halting backtest.")
            break

        model_a = row["prob_a"]
        model_b = row["prob_b"]
        market_a = row["a_market_prob"]
        market_b = row["b_market_prob"]

        # Skip fights without market odds
        if pd.isna(market_a) or pd.isna(market_b):
            bankroll_history.append(bankroll.bankroll)
            continue

        actual_winner_is_a = row["target"] == 1

        # No-odds model probs for agreement filter and dynamic blend
        no_odds_a = row.get("no_odds_prob_a")
        no_odds_b = row.get("no_odds_prob_b")

        # Dynamic blend weight: adjusts based on model confidence + agreement
        dyn_weight_a = dynamic_blend_weight(model_a, market_a, no_odds_a, blend_weight)
        dyn_weight_b = dynamic_blend_weight(model_b, market_b, no_odds_b, blend_weight)

        # Blend model with market using dynamic weights
        blend_a = blend_probability(model_a, market_a, dyn_weight_a)
        blend_b = 1.0 - blend_a

        # Edge = blended - market
        edge_a = blend_a - market_a
        edge_b = blend_b - market_b

        # Line movement data for filter
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

        bet_placed = False

        if edge_a >= min_edge and edge_a >= edge_b and _passes_filters(
            blend_a, market_a, edge_a, row.get("fighter_a", "A"), no_odds_a,
            line_movement=line_movement, line_is_sharp=line_is_sharp,
            line_steam_move=line_steam_move, bet_side="a",
            a_num_fights=a_fights, b_num_fights=b_fights,
        ):
            odds = implied_prob_to_decimal_odds(market_a)
            bet_size = bankroll.kelly_bet_size(blend_a, odds)
            if bet_size > 0:
                bet_idx = len(bankroll.history)
                bankroll.place_bet(bet_size, row.get("fighter_a", "A"), odds, blend_a, market_a)
                bankroll.settle_bet(bet_idx, won=actual_winner_is_a)
                bet_placed = True

                clv = np.nan
                if pd.notna(row.get("closing_prob_a")):
                    clv = calculate_closing_line_value(market_a, row["closing_prob_a"])

                bet_log.append({
                    "event_date": row.get("event_date"),
                    "fighter_a": row.get("fighter_a", ""),
                    "fighter_b": row.get("fighter_b", ""),
                    "bet_on": row.get("fighter_a", "A"),
                    "bet_side": "a",
                    "bet_size": bet_size,
                    "odds": odds,
                    "model_prob": model_a,
                    "blended_prob": blend_a,
                    "blend_weight_used": dyn_weight_a,
                    "market_prob": market_a,
                    "edge": edge_a,
                    "won": actual_winner_is_a,
                    "profit": bankroll.history[-1]["profit"],
                    "bankroll_after": bankroll.bankroll,
                    "clv": clv,
                    "closing_prob": row.get("closing_prob_a", np.nan),
                    "odds_source": odds_source,
                })

        elif edge_b >= min_edge and _passes_filters(
            blend_b, market_b, edge_b, row.get("fighter_b", "B"), no_odds_b,
            line_movement=line_movement, line_is_sharp=line_is_sharp,
            line_steam_move=line_steam_move, bet_side="b",
            a_num_fights=a_fights, b_num_fights=b_fights,
        ):
            odds = implied_prob_to_decimal_odds(market_b)
            bet_size = bankroll.kelly_bet_size(blend_b, odds)
            if bet_size > 0:
                bet_idx = len(bankroll.history)
                bankroll.place_bet(bet_size, row.get("fighter_b", "B"), odds, blend_b, market_b)
                bankroll.settle_bet(bet_idx, won=not actual_winner_is_a)
                bet_placed = True

                clv = np.nan
                if pd.notna(row.get("closing_prob_b")):
                    clv = calculate_closing_line_value(market_b, row["closing_prob_b"])

                bet_log.append({
                    "event_date": row.get("event_date"),
                    "fighter_a": row.get("fighter_a", ""),
                    "fighter_b": row.get("fighter_b", ""),
                    "bet_on": row.get("fighter_b", "B"),
                    "bet_side": "b",
                    "bet_size": bet_size,
                    "odds": odds,
                    "model_prob": model_b,
                    "blended_prob": blend_b,
                    "blend_weight_used": dyn_weight_b,
                    "market_prob": market_b,
                    "edge": edge_b,
                    "won": not actual_winner_is_a,
                    "profit": bankroll.history[-1]["profit"],
                    "bankroll_after": bankroll.bankroll,
                    "clv": clv,
                    "closing_prob": row.get("closing_prob_b", np.nan),
                    "odds_source": odds_source,
                })

        bankroll_history.append(bankroll.bankroll)

    # Results
    stats = bankroll.get_stats()
    bet_log_df = pd.DataFrame(bet_log)

    # Compute CLV stats
    clv_stats = {}
    if not bet_log_df.empty and "clv" in bet_log_df.columns:
        valid_clv = bet_log_df["clv"].dropna()
        if len(valid_clv) > 0:
            clv_stats = {
                "avg_clv": valid_clv.mean(),
                "median_clv": valid_clv.median(),
                "pct_positive_clv": (valid_clv > 0).mean(),
                "clv_sample_size": len(valid_clv),
            }

    logger.info(f"\n{'='*60}")
    logger.info("BACKTEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Odds source: {odds_source}")
    logger.info(f"Period: {predictions['event_date'].min()} to {predictions['event_date'].max()}")
    logger.info(f"Total fights analyzed: {len(predictions)}")
    logger.info(f"Bets placed: {stats.get('total_bets', 0)}")
    logger.info(f"Win rate: {stats.get('win_rate', 0):.1%}")
    logger.info(f"Total wagered: ${stats.get('total_wagered', 0):.2f}")
    logger.info(f"Total profit: ${stats.get('total_profit', 0):+.2f}")
    logger.info(f"ROI: {stats.get('roi', 0):+.1%}")
    logger.info(f"Starting bankroll: ${initial_bankroll:.2f}")
    logger.info(f"Ending bankroll: ${bankroll.bankroll:.2f}")
    logger.info(f"Bankroll change: {stats.get('bankroll_change_pct', 0):+.1%}")
    logger.info(f"Avg edge on bets: {stats.get('avg_edge', 0):.1%}")

    if clv_stats:
        logger.info(f"\n--- Closing Line Value (CLV) ---")
        logger.info(f"Avg CLV: {clv_stats['avg_clv']:+.2%}")
        logger.info(f"Median CLV: {clv_stats['median_clv']:+.2%}")
        logger.info(f"Positive CLV rate: {clv_stats['pct_positive_clv']:.1%}")
        logger.info(f"CLV sample size: {clv_stats['clv_sample_size']}")

    logger.info(f"{'='*60}")

    return {
        "stats": {**stats, **clv_stats},
        "bet_log": bet_log_df,
        "bankroll_history": bankroll_history,
        "predictions": predictions,
        "bankroll_manager": bankroll,
        "odds_source": odds_source,
    }


def run_comparison_backtest(
    test_df: pd.DataFrame,
    initial_bankroll: float = INITIAL_BANKROLL,
    min_edge: float = MIN_EDGE_THRESHOLD,
    kelly_fraction: float = KELLY_FRACTION,
) -> dict:
    """
    Run backtests comparing:
      1. Full model (with odds features) — "xgboost"
      2. No-odds model (fighter stats only) — "xgboost_no_odds"

    Both tested against historical opening odds (if available) or Kaggle closing odds.
    This reveals whether the model has independent edge beyond market consensus.
    """
    results = {}

    for model_name, label in [("xgboost", "Full Model (with odds)"),
                               ("xgboost_no_odds", "No-Odds Model (stats only)")]:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"BACKTESTING: {label}")
            logger.info(f"{'='*60}")
            result = run_backtest(
                test_df,
                model_name=model_name,
                initial_bankroll=initial_bankroll,
                min_edge=min_edge,
                kelly_fraction=kelly_fraction,
            )
            results[model_name] = result
        except FileNotFoundError:
            logger.warning(f"Model '{model_name}' not found. Skipping.")
        except Exception as e:
            logger.error(f"Backtest failed for {model_name}: {e}")

    # Comparison summary
    if len(results) >= 2:
        logger.info(f"\n{'='*60}")
        logger.info("MODEL COMPARISON")
        logger.info(f"{'='*60}")
        for name, res in results.items():
            s = res["stats"]
            clv_str = f", CLV: {s.get('avg_clv', 0):+.2%}" if "avg_clv" in s else ""
            logger.info(
                f"  {name}: ROI {s.get('roi', 0):+.1%}, "
                f"Win rate {s.get('win_rate', 0):.1%}, "
                f"Bets {s.get('total_bets', 0)}, "
                f"Profit ${s.get('total_profit', 0):+.2f}"
                f"{clv_str}"
            )

    return results


def plot_backtest(backtest_result: dict, save: bool = True) -> None:
    """Generate and save backtest visualizations."""
    plots_dir = LOGS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    stats = backtest_result["stats"]
    bet_log = backtest_result["bet_log"]
    bankroll_history = backtest_result["bankroll_history"]

    if bet_log.empty:
        logger.warning("No bets to plot.")
        return

    has_clv = "clv" in bet_log.columns and bet_log["clv"].notna().any()
    nrows = 3 if has_clv else 2
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 6 * nrows))

    # 1. Bankroll over time
    ax = axes[0, 0]
    ax.plot(bankroll_history, linewidth=1.5, color="steelblue")
    ax.axhline(y=bankroll_history[0], color="gray", linestyle="--", alpha=0.5, label="Starting bankroll")
    ax.set_xlabel("Bet #")
    ax.set_ylabel("Bankroll ($)")
    odds_src = backtest_result.get("odds_source", "unknown")
    ax.set_title(f"Bankroll Over Time (ROI: {stats.get('roi', 0):+.1%}) [{odds_src}]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Cumulative P&L
    ax = axes[0, 1]
    if "profit" in bet_log.columns:
        cum_pnl = bet_log["profit"].cumsum()
        colors = ["green" if p > 0 else "red" for p in bet_log["profit"]]
        ax.bar(range(len(bet_log)), bet_log["profit"], color=colors, alpha=0.6)
        ax.plot(cum_pnl.values, color="black", linewidth=2, label="Cumulative P&L")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Bet #")
        ax.set_ylabel("Profit ($)")
        ax.set_title("Per-Bet P&L and Cumulative")
        ax.legend()

    # 3. Edge distribution of bets
    ax = axes[1, 0]
    if "edge" in bet_log.columns:
        won = bet_log[bet_log["won"] == True]["edge"]
        lost = bet_log[bet_log["won"] == False]["edge"]
        ax.hist(won, bins=15, alpha=0.6, label=f"Wins ({len(won)})", color="green")
        ax.hist(lost, bins=15, alpha=0.6, label=f"Losses ({len(lost)})", color="red")
        ax.set_xlabel("Edge (model prob - market prob)")
        ax.set_ylabel("Count")
        ax.set_title("Edge Distribution by Outcome")
        ax.legend()

    # 4. Win rate by edge bucket
    ax = axes[1, 1]
    if "edge" in bet_log.columns and "won" in bet_log.columns:
        bet_log_copy = bet_log.copy()
        bet_log_copy["edge_bucket"] = pd.cut(bet_log_copy["edge"], bins=5)
        edge_stats = bet_log_copy.groupby("edge_bucket", observed=True)["won"].agg(["mean", "count"])
        if not edge_stats.empty:
            edge_stats["mean"].plot(kind="bar", ax=ax, color="steelblue")
            ax.set_xlabel("Edge Bucket")
            ax.set_ylabel("Win Rate")
            ax.set_title("Win Rate by Edge Size")
            ax.tick_params(axis="x", rotation=45)
            for i, (_, row) in enumerate(edge_stats.iterrows()):
                ax.text(i, row["mean"] + 0.01, f'n={int(row["count"])}', ha="center", fontsize=8)

    # 5-6. CLV plots (if available)
    if has_clv:
        valid_clv = bet_log.dropna(subset=["clv"])

        ax = axes[2, 0]
        ax.hist(valid_clv["clv"], bins=20, color="steelblue", alpha=0.7, edgecolor="black")
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Break-even CLV")
        avg_clv = valid_clv["clv"].mean()
        ax.axvline(x=avg_clv, color="green", linestyle="-", linewidth=2, label=f"Avg CLV: {avg_clv:+.2%}")
        ax.set_xlabel("Closing Line Value")
        ax.set_ylabel("Count")
        ax.set_title("CLV Distribution")
        ax.legend()

        ax = axes[2, 1]
        won_clv = valid_clv[valid_clv["won"] == True]["clv"]
        lost_clv = valid_clv[valid_clv["won"] == False]["clv"]
        ax.hist(won_clv, bins=15, alpha=0.6, label=f"Wins (avg CLV: {won_clv.mean():+.2%})", color="green")
        ax.hist(lost_clv, bins=15, alpha=0.6, label=f"Losses (avg CLV: {lost_clv.mean():+.2%})", color="red")
        ax.set_xlabel("Closing Line Value")
        ax.set_ylabel("Count")
        ax.set_title("CLV by Outcome")
        ax.legend()

    plt.suptitle("Backtest Results", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = plots_dir / "backtest_results.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved backtest plots to {path}")

        # Save bet log
        log_path = LOGS_DIR / "backtest_bet_log.csv"
        bet_log.to_csv(log_path, index=False)
        logger.info(f"Saved bet log to {log_path}")
    else:
        plt.show()


def sensitivity_analysis(
    test_df: pd.DataFrame,
    model_name: str = "xgboost",
    edge_thresholds: Optional[list] = None,
    kelly_fractions: Optional[list] = None,
) -> pd.DataFrame:
    """
    Run backtests with different parameter combinations to find optimal settings.
    """
    if edge_thresholds is None:
        edge_thresholds = [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
    if kelly_fractions is None:
        kelly_fractions = [0.10, 0.15, 0.20, 0.25, 0.33, 0.50]

    model_result = load_model(model_name)
    results = []

    for edge in edge_thresholds:
        for kelly in kelly_fractions:
            bt = run_backtest(
                test_df,
                model_result=model_result,
                min_edge=edge,
                kelly_fraction=kelly,
            )
            s = bt["stats"]
            results.append({
                "min_edge": edge,
                "kelly_fraction": kelly,
                "total_bets": s.get("total_bets", 0),
                "win_rate": s.get("win_rate", 0),
                "roi": s.get("roi", 0),
                "total_profit": s.get("total_profit", 0),
                "bankroll_change_pct": s.get("bankroll_change_pct", 0),
                "avg_edge": s.get("avg_edge", 0),
                "avg_clv": s.get("avg_clv", np.nan),
            })

    result_df = pd.DataFrame(results)
    path = LOGS_DIR / "sensitivity_analysis.csv"
    result_df.to_csv(path, index=False)
    logger.info(f"Sensitivity analysis saved to {path}")
    logger.info(f"\nBest ROI configs:\n{result_df.nlargest(5, 'roi').to_string(index=False)}")

    return result_df


def run_walkforward_backtest(
    features_df: pd.DataFrame,
    retrain_months: int = 6,
    initial_train_years: int = 5,
    min_edge: float = MIN_EDGE_THRESHOLD,
    kelly_fraction: float = KELLY_FRACTION,
    max_bet_fraction: float = MAX_BET_FRACTION,
    initial_bankroll: float = INITIAL_BANKROLL,
    blend_weight: float = BLEND_WEIGHT,
    bet_start_date: str = TRAIN_CUTOFF_DATE,
) -> dict:
    """
    Walk-forward backtest: retrain the model every N months on expanding window.

    Instead of a single train/test split, this:
      1. Trains on first `initial_train_years` of data
      2. Tests on next `retrain_months` of data
      3. Expands training window to include the tested period
      4. Repeats until all data is consumed

    Only places bets on fights after bet_start_date (default: 2022-01-01).

    Args:
        features_df: Full feature DataFrame (all fights, not pre-split)
        retrain_months: Months of test data per fold
        initial_train_years: Years of initial training data
        min_edge: Minimum edge threshold
        kelly_fraction: Kelly fraction for sizing
        max_bet_fraction: Max fraction per bet
        initial_bankroll: Starting bankroll
        blend_weight: Base blend weight
        bet_start_date: Only bet on fights after this date

    Returns dict with combined results across all folds.
    """
    from src.features.build_features import get_feature_columns, get_feature_columns_no_odds
    from src.model.train import train_xgboost

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

    # Initial training end
    train_end = min_date + pd.DateOffset(years=initial_train_years)

    bankroll = BankrollManager(
        initial_bankroll=initial_bankroll,
        kelly_fraction=kelly_fraction,
        max_bet_fraction=max_bet_fraction,
        auto_detect_balance=False,
    )

    all_bet_log = []
    bankroll_history = [initial_bankroll]
    fold_stats = []
    fold_num = 0
    bet_start = pd.Timestamp(bet_start_date)

    while train_end < max_date:
        test_end = train_end + pd.DateOffset(months=retrain_months)
        if test_end > max_date:
            test_end = max_date + pd.Timedelta(days=1)

        # Skip folds entirely before the betting window
        if pd.Timestamp(test_end) <= bet_start:
            train_end = test_end
            continue

        train_mask = dates < train_end
        test_mask = (dates >= train_end) & (dates < test_end)

        train_df = features_df[train_mask]
        test_df = features_df[test_mask]

        if len(train_df) < 100 or len(test_df) < 5:
            train_end = test_end
            continue

        fold_num += 1
        logger.info(
            f"\n--- Walk-forward fold {fold_num}: "
            f"Train {train_mask.sum()} fights (to {train_end.date()}), "
            f"Test {test_mask.sum()} fights ({train_end.date()} to {test_end.date()}) ---"
        )

        # Train fresh model on expanding window
        model_result = train_xgboost(train_df, feature_cols, calibrate=True)
        no_odds_result = train_xgboost(train_df, no_odds_cols, calibrate=True)

        # Generate predictions
        predictions = predict_batch(test_df, model_result=model_result)
        no_odds_preds = predict_batch(test_df, model_result=no_odds_result)
        predictions["no_odds_prob_a"] = no_odds_preds["prob_a"]
        predictions["no_odds_prob_b"] = no_odds_preds["prob_b"]

        # Merge historical odds
        predictions = _merge_historical_odds(predictions)

        # Resolve market odds
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

        for _, row in predictions.iterrows():
            # Only bet on fights after bet_start_date
            if pd.Timestamp(row.get("event_date")) < bet_start:
                continue

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

            dyn_weight_a = dynamic_blend_weight(model_a, market_a, no_odds_a, blend_weight)
            dyn_weight_b = dynamic_blend_weight(model_b, market_b, no_odds_b, blend_weight)

            blend_a = blend_probability(model_a, market_a, dyn_weight_a)
            blend_b = 1.0 - blend_a

            edge_a = blend_a - market_a
            edge_b = blend_b - market_b

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

            if edge_a >= min_edge and edge_a >= edge_b and _passes_filters(
                blend_a, market_a, edge_a, row.get("fighter_a", "A"), no_odds_a,
                line_movement=line_movement, line_is_sharp=line_is_sharp,
                line_steam_move=line_steam_move, bet_side="a",
                a_num_fights=a_fights, b_num_fights=b_fights,
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
                        "odds_source": odds_source,
                    })

            elif edge_b >= min_edge and _passes_filters(
                blend_b, market_b, edge_b, row.get("fighter_b", "B"), no_odds_b,
                line_movement=line_movement, line_is_sharp=line_is_sharp,
                line_steam_move=line_steam_move, bet_side="b",
                a_num_fights=a_fights, b_num_fights=b_fights,
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
                        "odds_source": odds_source,
                    })

            bankroll_history.append(bankroll.bankroll)

        fold_stats.append({
            "fold": fold_num,
            "train_end": str(train_end.date()),
            "test_end": str(test_end.date()) if not isinstance(test_end, pd.Timestamp) else str(test_end.date()),
            "train_size": train_mask.sum(),
            "test_size": test_mask.sum(),
            "bets": fold_bets,
            "wins": fold_wins,
            "win_rate": fold_wins / fold_bets if fold_bets > 0 else 0,
            "bankroll": bankroll.bankroll,
        })

        train_end = test_end

    # Final results
    stats = bankroll.get_stats()
    bet_log_df = pd.DataFrame(all_bet_log)

    clv_stats = {}
    if not bet_log_df.empty and "clv" in bet_log_df.columns:
        valid_clv = bet_log_df["clv"].dropna()
        if len(valid_clv) > 0:
            clv_stats = {
                "avg_clv": valid_clv.mean(),
                "median_clv": valid_clv.median(),
                "pct_positive_clv": (valid_clv > 0).mean(),
                "clv_sample_size": len(valid_clv),
            }

    fold_df = pd.DataFrame(fold_stats)
    logger.info(f"\n{'='*60}")
    logger.info("WALK-FORWARD BACKTEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Folds: {fold_num}")
    logger.info(f"Retrain interval: {retrain_months} months")
    logger.info(f"Bets placed: {stats.get('total_bets', 0)}")
    logger.info(f"Win rate: {stats.get('win_rate', 0):.1%}")
    logger.info(f"ROI: {stats.get('roi', 0):+.1%}")
    logger.info(f"Total profit: ${stats.get('total_profit', 0):+.2f}")
    logger.info(f"Bankroll: ${bankroll.bankroll:.2f}")
    if clv_stats:
        logger.info(f"Avg CLV: {clv_stats['avg_clv']:+.2%}")
    logger.info(f"\nPer-fold breakdown:")
    logger.info(fold_df.to_string(index=False))
    logger.info(f"{'='*60}")

    return {
        "stats": {**stats, **clv_stats},
        "bet_log": bet_log_df,
        "bankroll_history": bankroll_history,
        "bankroll_manager": bankroll,
        "fold_stats": fold_df,
        "odds_source": "walk_forward",
    }
