"""
Backtesting engine — simulates the betting strategy on historical data.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.strategy.value import find_value_bets, implied_prob_to_decimal_odds, _passes_underdog_filters
from src.strategy.bankroll import BankrollManager
from src.model.predict import predict_batch
from src.model.train import load_model
from src.config import (
    MIN_EDGE_THRESHOLD,
    KELLY_FRACTION,
    MAX_BET_FRACTION,
    INITIAL_BANKROLL,
    LOGS_DIR,
)

logger = logging.getLogger(__name__)


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
) -> dict:
    """
    Run a full backtest on historical fight data.

    Simulates placing bets chronologically on fights where model identifies value.

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

    Returns dict with backtest results and stats.
    """
    if model_result is None:
        model_result = load_model(model_name)

    # Generate predictions
    predictions = predict_batch(test_df, model_name=model_name, model_result=model_result)

    # Ensure market probability columns exist
    if market_prob_col_a not in predictions.columns:
        logger.warning(
            f"Market probability column '{market_prob_col_a}' not found. "
            "Using model probabilities as proxy (this won't show real value)."
        )
        # Simulate market odds: add noise to model probs to create synthetic market
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, len(predictions))
        predictions["a_market_prob"] = np.clip(predictions["prob_a"] + noise, 0.05, 0.95)
        predictions["b_market_prob"] = 1.0 - predictions["a_market_prob"]
        market_prob_col_a = "a_market_prob"
        market_prob_col_b = "b_market_prob"

    # Rename for value detection
    predictions["a_market_prob"] = predictions[market_prob_col_a]
    predictions["b_market_prob"] = predictions[market_prob_col_b]

    # Process fights chronologically
    predictions = predictions.sort_values("event_date").reset_index(drop=True)

    bankroll = BankrollManager(
        initial_bankroll=initial_bankroll,
        kelly_fraction=kelly_fraction,
        max_bet_fraction=max_bet_fraction,
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
        actual_winner_is_a = row["target"] == 1

        # Check for value on both sides
        edge_a = model_a - market_a
        edge_b = model_b - market_b

        bet_placed = False

        if edge_a >= min_edge and edge_a >= edge_b and _passes_underdog_filters(
            model_a, market_a, edge_a, row.get("fighter_a", "A")
        ):
            odds = implied_prob_to_decimal_odds(market_a)
            bet_size = bankroll.kelly_bet_size(model_a, odds)
            if bet_size > 0:
                bet_idx = len(bankroll.history)
                bankroll.place_bet(bet_size, row.get("fighter_a", "A"), odds, model_a, market_a)
                bankroll.settle_bet(bet_idx, won=actual_winner_is_a)
                bet_placed = True
                bet_log.append({
                    "event_date": row.get("event_date"),
                    "fighter_a": row.get("fighter_a", ""),
                    "fighter_b": row.get("fighter_b", ""),
                    "bet_on": row.get("fighter_a", "A"),
                    "bet_side": "a",
                    "bet_size": bet_size,
                    "odds": odds,
                    "model_prob": model_a,
                    "market_prob": market_a,
                    "edge": edge_a,
                    "won": actual_winner_is_a,
                    "profit": bankroll.history[-1]["profit"],
                    "bankroll_after": bankroll.bankroll,
                })

        elif edge_b >= min_edge and _passes_underdog_filters(
            model_b, market_b, edge_b, row.get("fighter_b", "B")
        ):
            odds = implied_prob_to_decimal_odds(market_b)
            bet_size = bankroll.kelly_bet_size(model_b, odds)
            if bet_size > 0:
                bet_idx = len(bankroll.history)
                bankroll.place_bet(bet_size, row.get("fighter_b", "B"), odds, model_b, market_b)
                bankroll.settle_bet(bet_idx, won=not actual_winner_is_a)
                bet_placed = True
                bet_log.append({
                    "event_date": row.get("event_date"),
                    "fighter_a": row.get("fighter_a", ""),
                    "fighter_b": row.get("fighter_b", ""),
                    "bet_on": row.get("fighter_b", "B"),
                    "bet_side": "b",
                    "bet_size": bet_size,
                    "odds": odds,
                    "model_prob": model_b,
                    "market_prob": market_b,
                    "edge": edge_b,
                    "won": not actual_winner_is_a,
                    "profit": bankroll.history[-1]["profit"],
                    "bankroll_after": bankroll.bankroll,
                })

        bankroll_history.append(bankroll.bankroll)

    # Results
    stats = bankroll.get_stats()
    bet_log_df = pd.DataFrame(bet_log)

    logger.info(f"\n{'='*60}")
    logger.info("BACKTEST RESULTS")
    logger.info(f"{'='*60}")
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
    logger.info(f"{'='*60}")

    return {
        "stats": stats,
        "bet_log": bet_log_df,
        "bankroll_history": bankroll_history,
        "predictions": predictions,
        "bankroll_manager": bankroll,
    }


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

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Bankroll over time
    ax = axes[0, 0]
    ax.plot(bankroll_history, linewidth=1.5, color="steelblue")
    ax.axhline(y=bankroll_history[0], color="gray", linestyle="--", alpha=0.5, label="Starting bankroll")
    ax.set_xlabel("Bet #")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title(f"Bankroll Over Time (ROI: {stats.get('roi', 0):+.1%})")
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
            })

    result_df = pd.DataFrame(results)
    path = LOGS_DIR / "sensitivity_analysis.csv"
    result_df.to_csv(path, index=False)
    logger.info(f"Sensitivity analysis saved to {path}")
    logger.info(f"\nBest ROI configs:\n{result_df.nlargest(5, 'roi').to_string(index=False)}")

    return result_df
