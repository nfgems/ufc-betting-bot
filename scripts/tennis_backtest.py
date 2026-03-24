"""Tennis model backtest against closing odds.

Measures model alpha by comparing model predictions to market-implied probabilities
using walk-forward OOS predictions joined with closing odds from tennis-data.co.uk.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ODDS_RAW_PATH = ROOT / "data" / "processed" / "tennis" / "matches_with_odds.csv"
ODDS_MODELING_SAFE_PATH = ROOT / "data" / "processed" / "tennis" / "matches_with_odds_modeling_safe.csv"

from src.model.tennis_model import train_tennis_model, run_walkforward_evaluation
from src.features.tennis_features import STAGE1_TENNIS_MODEL_COLUMNS


def load_oos_predictions() -> pd.DataFrame:
    """Train model and extract walk-forward OOS predictions."""
    features = pd.read_csv(ROOT / "data" / "processed" / "tennis" / "features.csv", low_memory=False)
    features["event_date"] = pd.to_datetime(features["event_date"])
    result = train_tennis_model(features)

    # Extract predictions from the evaluation
    preds = result["evaluation_predictions"]
    if isinstance(preds, pd.DataFrame):
        return preds
    # It might be stored differently
    return pd.DataFrame(preds)


def load_odds() -> pd.DataFrame:
    """Load the modeling-safe odds export when available."""
    odds_path = ODDS_MODELING_SAFE_PATH if ODDS_MODELING_SAFE_PATH.exists() else ODDS_RAW_PATH
    odds = pd.read_csv(odds_path, low_memory=False)
    odds["event_date"] = pd.to_datetime(odds["event_date"])
    odds.attrs["source_path"] = str(odds_path)
    return odds


def compute_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if pd.isna(decimal_odds) or decimal_odds <= 1.0:
        return np.nan
    return 1.0 / decimal_odds


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Remove vig from implied probabilities to get fair probabilities."""
    total = prob_a + prob_b
    if total <= 0 or pd.isna(total):
        return np.nan, np.nan
    return prob_a / total, prob_b / total


def run_backtest():
    """Run the full backtest pipeline."""
    print("=" * 70)
    print("TENNIS MODEL BACKTEST vs CLOSING ODDS")
    print("=" * 70)

    # Step 1: Get OOS predictions
    print("\n[1] Training model and getting OOS predictions...")
    preds = load_oos_predictions()
    print(f"    Got {len(preds)} OOS predictions")
    print(f"    Columns: {list(preds.columns)}")

    # Step 2: Load odds
    print("\n[2] Loading odds data...")
    odds = load_odds()
    odds_cols = ["event_date", "player_a", "player_b", "b365_a", "b365_b",
                 "ps_a", "ps_b", "max_a", "max_b", "avg_a", "avg_b"]
    odds_slim = odds[odds_cols].copy()
    print(f"    Source: {Path(odds.attrs['source_path']).name}")
    print(f"    {len(odds)} total rows, {odds['b365_a'].notna().sum()} with B365 odds")

    # Step 3: Join predictions with odds
    print("\n[3] Joining predictions with odds...")
    preds["event_date"] = pd.to_datetime(preds["event_date"])

    # Join on event_date + player_a + player_b
    merged = preds.merge(
        odds_slim,
        on=["event_date", "player_a", "player_b"],
        how="inner",
        suffixes=("", "_odds"),
    )
    print(f"    Merged: {len(merged)} rows (from {len(preds)} predictions)")

    # Filter to rows with valid odds
    has_b365 = merged["b365_a"].notna() & merged["b365_b"].notna()
    has_ps = merged["ps_a"].notna() & merged["ps_b"].notna()
    has_avg = merged["avg_a"].notna() & merged["avg_b"].notna()

    print(f"    With B365: {has_b365.sum()}")
    print(f"    With Pinnacle: {has_ps.sum()}")
    print(f"    With Average: {has_avg.sum()}")

    # Step 4: Compute market-implied probabilities (vig-removed)
    print("\n[4] Computing market probabilities...")
    for prefix in ["b365", "ps", "avg", "max"]:
        col_a = f"{prefix}_a"
        col_b = f"{prefix}_b"
        merged[f"{prefix}_imp_a"] = merged[col_a].apply(compute_implied_prob)
        merged[f"{prefix}_imp_b"] = merged[col_b].apply(compute_implied_prob)

        # Remove vig
        fair = merged.apply(
            lambda r: remove_vig(r[f"{prefix}_imp_a"], r[f"{prefix}_imp_b"]),
            axis=1, result_type="expand"
        )
        merged[f"{prefix}_fair_a"] = fair[0]
        merged[f"{prefix}_fair_b"] = fair[1]

    # Step 5: Measure model vs market
    print("\n[5] Computing alpha metrics...")

    # Model prediction column
    pred_col = None
    for candidate in ["prob_a", "pred_prob_a", "p_a", "model_prob_a"]:
        if candidate in merged.columns:
            pred_col = candidate
            break

    if pred_col is None:
        print(f"    Available columns: {list(merged.columns)}")
        raise ValueError("Cannot find model prediction column")

    print(f"    Using model prediction column: {pred_col}")

    target_col = "target"
    model_pred = merged[pred_col].values
    actual = merged[target_col].values

    # Overall model Brier
    model_brier = np.nanmean((model_pred - actual) ** 2)
    print(f"\n    Model Brier Score: {model_brier:.4f}")

    results = {}
    for market, fair_col in [("B365", "b365_fair_a"), ("Pinnacle", "ps_fair_a"),
                              ("Average", "avg_fair_a"), ("MaxOdds", "max_fair_a")]:
        mask = merged[fair_col].notna()
        if mask.sum() == 0:
            continue

        market_pred = merged.loc[mask, fair_col].values
        actual_subset = merged.loc[mask, target_col].values
        model_subset = merged.loc[mask, pred_col].values

        market_brier = np.nanmean((market_pred - actual_subset) ** 2)
        model_brier_sub = np.nanmean((model_subset - actual_subset) ** 2)

        # Brier skill score: positive means model is better than market
        bss = 1 - (model_brier_sub / market_brier)

        # Log loss comparison
        eps = 1e-10
        market_ll = -np.nanmean(
            actual_subset * np.log(np.clip(market_pred, eps, 1 - eps)) +
            (1 - actual_subset) * np.log(np.clip(1 - market_pred, eps, 1 - eps))
        )
        model_ll = -np.nanmean(
            actual_subset * np.log(np.clip(model_subset, eps, 1 - eps)) +
            (1 - actual_subset) * np.log(np.clip(1 - model_subset, eps, 1 - eps))
        )

        # Edge: model_prob - market_prob for matches model predicts correctly
        edge = model_subset - market_pred
        avg_edge = np.nanmean(edge)

        results[market] = {
            "n": int(mask.sum()),
            "market_brier": market_brier,
            "model_brier": model_brier_sub,
            "brier_skill_score": bss,
            "market_logloss": market_ll,
            "model_logloss": model_ll,
            "avg_edge": avg_edge,
        }

        print(f"\n    --- vs {market} ({mask.sum()} matches) ---")
        print(f"    Market Brier: {market_brier:.4f}")
        print(f"    Model Brier:  {model_brier_sub:.4f}")
        print(f"    BSS (model vs market): {bss:+.4f} ({'model better' if bss > 0 else 'market better'})")
        print(f"    Market LogLoss: {market_ll:.4f}")
        print(f"    Model LogLoss:  {model_ll:.4f}")
        print(f"    Avg Edge: {avg_edge:+.4f}")

    # Step 6: Simulated betting P/L
    print("\n" + "=" * 70)
    print("SIMULATED BETTING (Kelly Criterion)")
    print("=" * 70)

    for market, odds_col_a, odds_col_b in [
        ("B365", "b365_a", "b365_b"),
        ("Pinnacle", "ps_a", "ps_b"),
        ("Average", "avg_a", "avg_b"),
    ]:
        mask = merged[odds_col_a].notna() & merged[odds_col_b].notna()
        if mask.sum() == 0:
            continue

        sub = merged[mask].copy()
        model_p = sub[pred_col].values
        odds_a = sub[odds_col_a].values
        odds_b = sub[odds_col_b].values
        actuals = sub[target_col].values

        # Kelly sizing
        kelly_fraction = 0.25  # Quarter Kelly
        min_edge = 0.02  # Minimum 2% edge to bet

        bankroll = 1000.0
        initial_bankroll = bankroll
        n_bets = 0
        n_wins = 0
        total_wagered = 0.0
        max_drawdown = 0.0
        peak = bankroll

        for i in range(len(sub)):
            p_a = model_p[i]
            p_b = 1 - p_a

            # Check edge on player A
            edge_a = p_a - (1.0 / odds_a[i])
            kelly_a = (p_a * odds_a[i] - 1) / (odds_a[i] - 1) if odds_a[i] > 1 else 0

            # Check edge on player B
            edge_b = p_b - (1.0 / odds_b[i])
            kelly_b = (p_b * odds_b[i] - 1) / (odds_b[i] - 1) if odds_b[i] > 1 else 0

            bet_side = None
            if edge_a >= min_edge and kelly_a > 0:
                bet_size = bankroll * kelly_fraction * kelly_a
                bet_side = "a"
                bet_odds = odds_a[i]
                won = actuals[i] == 1
            elif edge_b >= min_edge and kelly_b > 0:
                bet_size = bankroll * kelly_fraction * kelly_b
                bet_side = "b"
                bet_odds = odds_b[i]
                won = actuals[i] == 0
            else:
                continue

            bet_size = min(bet_size, bankroll * 0.10)  # Max 10% of bankroll
            if bet_size < 1:
                continue

            n_bets += 1
            total_wagered += bet_size

            if won:
                bankroll += bet_size * (bet_odds - 1)
                n_wins += 1
            else:
                bankroll -= bet_size

            peak = max(peak, bankroll)
            drawdown = (peak - bankroll) / peak
            max_drawdown = max(max_drawdown, drawdown)

        roi = (bankroll - initial_bankroll) / total_wagered * 100 if total_wagered > 0 else 0
        profit = bankroll - initial_bankroll

        print(f"\n--- {market} ---")
        print(f"  Matches analyzed: {mask.sum()}")
        print(f"  Bets placed: {n_bets} ({100*n_bets/mask.sum():.1f}% of matches)")
        print(f"  Win rate: {100*n_wins/n_bets:.1f}%" if n_bets > 0 else "  No bets")
        print(f"  Total wagered: ${total_wagered:.0f}")
        print(f"  Final bankroll: ${bankroll:.0f} (started at ${initial_bankroll:.0f})")
        print(f"  Profit: ${profit:+.0f}")
        print(f"  ROI: {roi:+.1f}%")
        print(f"  Max drawdown: {100*max_drawdown:.1f}%")

    # Step 7: Edge distribution analysis
    print("\n" + "=" * 70)
    print("EDGE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    avg_mask = merged["avg_fair_a"].notna()
    if avg_mask.sum() > 0:
        sub = merged[avg_mask].copy()
        edge = sub[pred_col] - sub["avg_fair_a"]

        for threshold in [0.0, 0.02, 0.05, 0.10, 0.15]:
            pos_edge = sub[edge >= threshold]
            neg_edge = sub[edge <= -threshold]

            if len(pos_edge) > 0:
                hit_rate = pos_edge[target_col].mean()
                market_hit = pos_edge["avg_fair_a"].mean()
                print(f"  Edge >= {threshold:+.0%}: {len(pos_edge)} matches, "
                      f"actual win rate {hit_rate:.1%}, market expected {market_hit:.1%}")

            if threshold > 0 and len(neg_edge) > 0:
                hit_rate = neg_edge[target_col].mean()
                market_hit = neg_edge["avg_fair_a"].mean()
                print(f"  Edge <= {-threshold:+.0%}: {len(neg_edge)} matches, "
                      f"actual win rate {hit_rate:.1%}, market expected {market_hit:.1%}")

    # Save merged data for further analysis
    output_path = ROOT / "data" / "processed" / "tennis" / "backtest_results.csv"
    merged.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    run_backtest()
