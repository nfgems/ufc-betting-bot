"""Compare backtest results across different BLEND_WEIGHT values."""

import sys
import logging
import pandas as pd

logging.basicConfig(level=logging.WARNING)  # Suppress verbose logs

from src.features.build_features import build_features
from src.data.kaggle_loader import load_kaggle_dataset
from src.strategy.backtest import run_walkforward_backtest
from src.config import MIN_EDGE_THRESHOLD, KELLY_FRACTION, MAX_BET_FRACTION, INITIAL_BANKROLL

BLEND_WEIGHTS = [0.20, 0.30, 0.40, 0.50]

def main():
    # Load and build features
    print("Loading data and building features...")
    fights_df = load_kaggle_dataset()
    features_df = build_features(fights_df)
    print(f"Built features for {len(features_df)} fights\n")

    results = {}
    for bw in BLEND_WEIGHTS:
        print(f"--- Running walk-forward backtest with BLEND_WEIGHT = {bw:.2f} ---")
        result = run_walkforward_backtest(
            features_df,
            retrain_months=6,
            initial_train_years=5,
            initial_bankroll=INITIAL_BANKROLL,
            min_edge=MIN_EDGE_THRESHOLD,
            kelly_fraction=KELLY_FRACTION,
            max_bet_fraction=MAX_BET_FRACTION,
            blend_weight=bw,
        )
        stats = result["stats"]
        results[bw] = {
            "blend_weight": bw,
            "total_bets": stats.get("total_bets", 0),
            "win_rate": stats.get("win_rate", 0),
            "roi": stats.get("roi", 0),
            "total_profit": stats.get("total_profit", 0),
            "total_wagered": stats.get("total_wagered", 0),
            "bankroll_change_pct": stats.get("bankroll_change_pct", 0),
            "avg_edge": stats.get("avg_edge", 0),
            "avg_clv": stats.get("avg_clv", float("nan")),
            "final_bankroll": result["bankroll_manager"].bankroll,
        }
        s = results[bw]
        print(f"  Bets: {s['total_bets']} | Win rate: {s['win_rate']:.1%} | "
              f"ROI: {s['roi']:+.1%} | Profit: ${s['total_profit']:+.2f} | "
              f"Final bankroll: ${s['final_bankroll']:.2f}\n")

    # Summary table
    print("\n" + "=" * 80)
    print("BLEND WEIGHT COMPARISON SUMMARY")
    print("=" * 80)
    df = pd.DataFrame(results.values())
    df = df.set_index("blend_weight")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df.to_string())
    print("=" * 80)

    # Best
    best_bw = max(results, key=lambda k: results[k]["roi"])
    print(f"\nBest ROI: BLEND_WEIGHT = {best_bw:.2f} "
          f"(ROI: {results[best_bw]['roi']:+.1%}, "
          f"Profit: ${results[best_bw]['total_profit']:+.2f})")


if __name__ == "__main__":
    main()
