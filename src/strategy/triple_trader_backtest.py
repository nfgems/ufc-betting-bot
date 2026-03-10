"""
Triple-Trader Backtest — simulates the full 3-trader system on historical data.

Unlike model_lab.py (single-trader), this replicates how the live system actually works:
  - Trader A (Conservative): 40% bankroll, blend=0.20, value bets (Kelly)
  - Trader B (Aggressive):   40% bankroll, blend=0.40, value bets (Kelly)
  - Trader C (Conviction):   20% bankroll, flat sizing, 75% XGB + 60% no-odds agreement

Includes full conflict resolution:
  - A vs B opposite sides -> both cancel
  - A vs B same side -> higher edge takes it
  - C opposite from A/B -> C cancelled

Usage:
    python -m src.strategy.triple_trader_backtest
    python -m src.strategy.triple_trader_backtest --bankroll 1000
"""

import argparse
import logging
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import (
    INITIAL_BANKROLL,
    MIN_EDGE_THRESHOLD,
    KELLY_FRACTION,
    MAX_BET_FRACTION,
    PROCESSED_DATA_DIR,
    LOGS_DIR,
    TRAIN_CUTOFF_DATE,
    TRADER_A_BLEND,
    TRADER_B_BLEND,
    TRADER_A_SHARE,
    TRADER_B_SHARE,
    TRADER_C_SHARE,
    CONVICTION_MIN_MODEL_PROB,
    CONVICTION_MIN_NO_ODDS_PROB,
    CONVICTION_BET_FRACTION,
    CONVICTION_CONFIDENCE_BONUS,
    CONVICTION_MAX_BET_FRACTION,
    MIN_FIGHTER_FIGHTS,
)
from src.strategy.value import (
    blend_probability,
    dynamic_blend_weight,
    implied_prob_to_decimal_odds,
    _passes_filters,
)
from src.strategy.bankroll import BankrollManager
from src.strategy.backtest import _merge_historical_odds, _resolve_market_odds
from src.strategy.model_variants import (
    VariantConfig,
    train_variant_model,
    ALL_VARIANTS,
)
from src.features.build_features import (
    get_feature_columns,
    get_feature_columns_no_odds,
    build_features,
)
from src.strategy.model_lab import _predict_batch_with_model

logger = logging.getLogger(__name__)

LAB_DIR = LOGS_DIR / "model_lab"
LAB_DIR.mkdir(parents=True, exist_ok=True)


def _conviction_bet_size_backtest(model_prob: float, bankroll: float) -> float:
    """Conviction bet sizing (same as production)."""
    base = CONVICTION_BET_FRACTION * bankroll
    excess_confidence = max(0.0, model_prob - CONVICTION_MIN_MODEL_PROB)
    bonus_steps = excess_confidence / 0.05
    bonus = bonus_steps * CONVICTION_CONFIDENCE_BONUS * bankroll
    bet = min(base + bonus, CONVICTION_MAX_BET_FRACTION * bankroll)
    return round(bet, 2) if bet >= 1.0 else 0.0


def run_triple_trader_backtest(
    features_df: pd.DataFrame = None,
    initial_bankroll: float = INITIAL_BANKROLL,
    retrain_months: int = 6,
    initial_train_years: int = 5,
    bet_start_date: str = TRAIN_CUTOFF_DATE,
) -> dict:
    """
    Run a walk-forward backtest simulating the full triple-trader system.

    Models train on expanding windows from the start of the dataset, but
    bets are only placed on fights after bet_start_date (default: 2022-01-01).
    This matches the production TRAIN_CUTOFF_DATE.

    Returns per-trader stats + combined portfolio stats.
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

    features_df = features_df.sort_values("event_date").copy()
    features_df = features_df.dropna(subset=["target"])

    feature_cols = get_feature_columns(features_df)
    no_odds_cols = get_feature_columns_no_odds(features_df)
    no_odds_cols = [c for c in no_odds_cols if c in features_df.columns]

    if "a_num_fights" in features_df.columns and "b_num_fights" in features_df.columns:
        features_df = features_df[
            (features_df["a_num_fights"] >= 2) & (features_df["b_num_fights"] >= 2)
        ]

    dates = pd.to_datetime(features_df["event_date"])
    min_date = dates.min()
    max_date = dates.max()
    train_end = min_date + pd.DateOffset(years=initial_train_years)

    # --- Create 3 bankrolls (40/40/20 split) ---
    alloc_a = round(initial_bankroll * TRADER_A_SHARE, 2)
    alloc_b = round(initial_bankroll * TRADER_B_SHARE, 2)
    alloc_c = round(initial_bankroll * TRADER_C_SHARE, 2)

    bank_a = BankrollManager(
        initial_bankroll=alloc_a, kelly_fraction=KELLY_FRACTION,
        max_bet_fraction=MAX_BET_FRACTION, auto_detect_balance=False,
    )
    bank_b = BankrollManager(
        initial_bankroll=alloc_b, kelly_fraction=KELLY_FRACTION,
        max_bet_fraction=MAX_BET_FRACTION, auto_detect_balance=False,
    )
    bank_c = BankrollManager(
        initial_bankroll=alloc_c, kelly_fraction=1.0,
        max_bet_fraction=CONVICTION_MAX_BET_FRACTION, auto_detect_balance=False,
    )

    # Use production baseline variant for training
    variant = ALL_VARIANTS["baseline"]()

    bet_log = []
    bankroll_history = []
    fold_num = 0

    bet_start = pd.Timestamp(bet_start_date)

    logger.info(f"\n{'='*60}")
    logger.info("TRIPLE TRADER BACKTEST")
    logger.info(f"  Total bankroll: ${initial_bankroll:.2f}")
    logger.info(f"  Trader A (Conservative, blend={TRADER_A_BLEND}): ${alloc_a:.2f}")
    logger.info(f"  Trader B (Aggressive, blend={TRADER_B_BLEND}): ${alloc_b:.2f}")
    logger.info(f"  Trader C (Conviction): ${alloc_c:.2f}")
    logger.info(f"  Betting on fights from: {bet_start_date}")
    logger.info(f"{'='*60}\n")

    while train_end < max_date:
        test_end = train_end + pd.DateOffset(months=retrain_months)
        if test_end > max_date:
            test_end = max_date + pd.Timedelta(days=1)

        # Skip folds entirely before the betting window (saves training time)
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
            f"Fold {fold_num}: Train {len(train_df)}, Test {len(test_df)} "
            f"({train_end.date()} to {test_end.date()})"
        )

        # --- Train models ---
        model_result = train_variant_model(train_df, feature_cols, variant)
        no_odds_variant = VariantConfig(name="_no_odds", description="internal")
        no_odds_result = train_variant_model(train_df, no_odds_cols, no_odds_variant)

        # --- Predict ---
        predictions = _predict_batch_with_model(test_df, model_result)
        no_odds_preds = _predict_batch_with_model(test_df, no_odds_result)
        predictions["no_odds_prob_a"] = no_odds_preds["prob_a"]
        predictions["no_odds_prob_b"] = no_odds_preds["prob_b"]

        # --- Merge odds ---
        predictions = _merge_historical_odds(predictions)
        try:
            predictions, _ = _resolve_market_odds(
                predictions, "a_fair_prob_avg", "b_fair_prob_avg"
            )
        except ValueError:
            train_end = test_end
            continue

        predictions = predictions.sort_values("event_date").reset_index(drop=True)

        # --- Per-fight betting loop ---
        for _, row in predictions.iterrows():
            # Only place bets on fights after bet_start_date
            fight_date = pd.Timestamp(row.get("event_date"))
            if fight_date < bet_start:
                continue
            model_a = row["prob_a"]
            model_b = row["prob_b"]
            market_a = row["a_market_prob"]
            market_b = row["b_market_prob"]

            if pd.isna(market_a) or pd.isna(market_b):
                bankroll_history.append({
                    "combined": bank_a.bankroll + bank_b.bankroll + bank_c.bankroll,
                    "trader_a": bank_a.bankroll,
                    "trader_b": bank_b.bankroll,
                    "trader_c": bank_c.bankroll,
                })
                continue

            actual_winner_is_a = row["target"] == 1
            no_odds_a = row.get("no_odds_prob_a")
            no_odds_b = row.get("no_odds_prob_b")
            # DataFrame .get() returns NaN not None for missing values
            if isinstance(no_odds_a, float) and np.isnan(no_odds_a):
                no_odds_a = None
            if isinstance(no_odds_b, float) and np.isnan(no_odds_b):
                no_odds_b = None

            line_movement = row.get("line_movement")
            line_is_sharp = row.get("line_is_sharp")
            line_steam_move = row.get("line_steam_move")
            if isinstance(line_movement, float) and np.isnan(line_movement):
                line_movement = None

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

            min_edge = MIN_EDGE_THRESHOLD

            # ============================================================
            # TRADER A: Conservative (blend=0.20)
            # ============================================================
            bet_a = None
            if not bank_a.is_stopped:
                dyn_w_a = dynamic_blend_weight(model_a, market_a, no_odds_a, TRADER_A_BLEND)
                blend_a_a = blend_probability(model_a, market_a, dyn_w_a)
                blend_a_b = 1.0 - blend_a_a
                edge_a_a = blend_a_a - market_a
                edge_a_b = blend_a_b - market_b

                if edge_a_a >= min_edge and edge_a_a >= edge_a_b and _passes_filters(
                    blend_a_a, market_a, edge_a_a, row.get("fighter_a", "A"), no_odds_a,
                    line_movement=line_movement, line_is_sharp=line_is_sharp,
                    line_steam_move=line_steam_move, bet_side="a",
                    a_num_fights=a_fights, b_num_fights=b_fights,
                ):
                    bet_a = {"side": "a", "edge": edge_a_a, "blend": blend_a_a, "market": market_a}
                elif edge_a_b >= min_edge and _passes_filters(
                    blend_a_b, market_b, edge_a_b, row.get("fighter_b", "B"), no_odds_b,
                    line_movement=line_movement, line_is_sharp=line_is_sharp,
                    line_steam_move=line_steam_move, bet_side="b",
                    a_num_fights=a_fights, b_num_fights=b_fights,
                ):
                    bet_a = {"side": "b", "edge": edge_a_b, "blend": blend_a_b, "market": market_b}

            # ============================================================
            # TRADER B: Aggressive (blend=0.40)
            # ============================================================
            bet_b = None
            if not bank_b.is_stopped:
                dyn_w_b = dynamic_blend_weight(model_a, market_a, no_odds_a, TRADER_B_BLEND)
                blend_b_a = blend_probability(model_a, market_a, dyn_w_b)
                blend_b_b = 1.0 - blend_b_a
                edge_b_a = blend_b_a - market_a
                edge_b_b = blend_b_b - market_b

                if edge_b_a >= min_edge and edge_b_a >= edge_b_b and _passes_filters(
                    blend_b_a, market_a, edge_b_a, row.get("fighter_a", "A"), no_odds_a,
                    line_movement=line_movement, line_is_sharp=line_is_sharp,
                    line_steam_move=line_steam_move, bet_side="a",
                    a_num_fights=a_fights, b_num_fights=b_fights,
                ):
                    bet_b = {"side": "a", "edge": edge_b_a, "blend": blend_b_a, "market": market_a}
                elif edge_b_b >= min_edge and _passes_filters(
                    blend_b_b, market_b, edge_b_b, row.get("fighter_b", "B"), no_odds_b,
                    line_movement=line_movement, line_is_sharp=line_is_sharp,
                    line_steam_move=line_steam_move, bet_side="b",
                    a_num_fights=a_fights, b_num_fights=b_fights,
                ):
                    bet_b = {"side": "b", "edge": edge_b_b, "blend": blend_b_b, "market": market_b}

            # ============================================================
            # TRADER C: Conviction (75% XGB + 60% no-odds)
            # ============================================================
            bet_c = None
            if not bank_c.is_stopped:
                for side, mp, mrkt, nop, fighter in [
                    ("a", model_a, market_a, no_odds_a, row.get("fighter_a", "A")),
                    ("b", model_b, market_b, no_odds_b, row.get("fighter_b", "B")),
                ]:
                    if mp >= CONVICTION_MIN_MODEL_PROB and nop is not None and nop >= CONVICTION_MIN_NO_ODDS_PROB:
                        # Fighter experience check
                        if a_fights is not None and a_fights < MIN_FIGHTER_FIGHTS:
                            continue
                        if b_fights is not None and b_fights < MIN_FIGHTER_FIGHTS:
                            continue
                        bet_c = {"side": side, "model_prob": mp, "market": mrkt, "fighter": fighter}
                        break  # Take the first qualifying side

            # ============================================================
            # CONFLICT RESOLUTION
            # ============================================================
            # A vs B conflicts
            if bet_a is not None and bet_b is not None:
                if bet_a["side"] != bet_b["side"]:
                    # Opposite sides -> cancel both
                    bet_a = None
                    bet_b = None
                else:
                    # Same side -> higher edge takes it
                    if bet_a["edge"] >= bet_b["edge"]:
                        bet_b = None
                    else:
                        bet_a = None

            # C vs A/B conflicts
            if bet_c is not None:
                for vb in [bet_a, bet_b]:
                    if vb is not None and vb["side"] != bet_c["side"]:
                        bet_c = None
                        break

            # ============================================================
            # EXECUTE BETS
            # ============================================================
            # Trader A
            if bet_a is not None:
                odds = implied_prob_to_decimal_odds(bet_a["market"])
                bet_size = bank_a.kelly_bet_size(bet_a["blend"], odds)
                if bet_size > 0:
                    bet_idx = len(bank_a.history)
                    fighter = row.get("fighter_a" if bet_a["side"] == "a" else "fighter_b", "?")
                    bank_a.place_bet(bet_size, fighter, odds, bet_a["blend"], bet_a["market"])
                    won = actual_winner_is_a if bet_a["side"] == "a" else not actual_winner_is_a
                    bank_a.settle_bet(bet_idx, won=won)
                    bet_log.append({
                        "trader": "A", "event_date": row.get("event_date"),
                        "fighter_a": row.get("fighter_a", ""),
                        "fighter_b": row.get("fighter_b", ""),
                        "bet_on": fighter, "side": bet_a["side"],
                        "bet_size": bet_size, "odds": odds,
                        "edge": bet_a["edge"], "won": won,
                        "profit": bank_a.history[-1]["profit"],
                        "bankroll_after": bank_a.bankroll, "fold": fold_num,
                    })

            # Trader B
            if bet_b is not None:
                odds = implied_prob_to_decimal_odds(bet_b["market"])
                bet_size = bank_b.kelly_bet_size(bet_b["blend"], odds)
                if bet_size > 0:
                    bet_idx = len(bank_b.history)
                    fighter = row.get("fighter_a" if bet_b["side"] == "a" else "fighter_b", "?")
                    bank_b.place_bet(bet_size, fighter, odds, bet_b["blend"], bet_b["market"])
                    won = actual_winner_is_a if bet_b["side"] == "a" else not actual_winner_is_a
                    bank_b.settle_bet(bet_idx, won=won)
                    bet_log.append({
                        "trader": "B", "event_date": row.get("event_date"),
                        "fighter_a": row.get("fighter_a", ""),
                        "fighter_b": row.get("fighter_b", ""),
                        "bet_on": fighter, "side": bet_b["side"],
                        "bet_size": bet_size, "odds": odds,
                        "edge": bet_b["edge"], "won": won,
                        "profit": bank_b.history[-1]["profit"],
                        "bankroll_after": bank_b.bankroll, "fold": fold_num,
                    })

            # Trader C
            if bet_c is not None:
                bet_size = _conviction_bet_size_backtest(bet_c["model_prob"], bank_c.bankroll)
                if bet_size > 0:
                    odds = implied_prob_to_decimal_odds(bet_c["market"])
                    bet_idx = len(bank_c.history)
                    bank_c.place_bet(bet_size, bet_c["fighter"], odds, bet_c["model_prob"], bet_c["market"])
                    won = actual_winner_is_a if bet_c["side"] == "a" else not actual_winner_is_a
                    bank_c.settle_bet(bet_idx, won=won)
                    bet_log.append({
                        "trader": "C", "event_date": row.get("event_date"),
                        "fighter_a": row.get("fighter_a", ""),
                        "fighter_b": row.get("fighter_b", ""),
                        "bet_on": bet_c["fighter"], "side": bet_c["side"],
                        "bet_size": bet_size, "odds": odds,
                        "edge": bet_c["model_prob"] - bet_c["market"], "won": won,
                        "profit": bank_c.history[-1]["profit"],
                        "bankroll_after": bank_c.bankroll, "fold": fold_num,
                    })

            bankroll_history.append({
                "combined": bank_a.bankroll + bank_b.bankroll + bank_c.bankroll,
                "trader_a": bank_a.bankroll,
                "trader_b": bank_b.bankroll,
                "trader_c": bank_c.bankroll,
            })

        train_end = test_end

    # --- Results ---
    stats_a = bank_a.get_stats()
    stats_b = bank_b.get_stats()
    stats_c = bank_c.get_stats()

    bet_log_df = pd.DataFrame(bet_log)
    bh_df = pd.DataFrame(bankroll_history)

    combined_profit = (
        (bank_a.bankroll - alloc_a)
        + (bank_b.bankroll - alloc_b)
        + (bank_c.bankroll - alloc_c)
    )
    combined_wagered = (
        stats_a.get("total_wagered", 0)
        + stats_b.get("total_wagered", 0)
        + stats_c.get("total_wagered", 0)
    )
    combined_bets = (
        stats_a.get("total_bets", 0)
        + stats_b.get("total_bets", 0)
        + stats_c.get("total_bets", 0)
    )
    combined_wins = (
        stats_a.get("wins", 0)
        + stats_b.get("wins", 0)
        + stats_c.get("wins", 0)
    )

    final_bankroll = bank_a.bankroll + bank_b.bankroll + bank_c.bankroll

    # Max drawdown from combined bankroll history
    max_dd = 0.0
    if not bh_df.empty:
        peak = bh_df["combined"].expanding().max()
        drawdown = 1.0 - bh_df["combined"] / peak
        max_dd = drawdown.max()

    results = {
        "trader_a": {
            "name": f"Trader A (Conservative, blend={TRADER_A_BLEND})",
            "allocation": alloc_a,
            "final_bankroll": bank_a.bankroll,
            "stats": stats_a,
        },
        "trader_b": {
            "name": f"Trader B (Aggressive, blend={TRADER_B_BLEND})",
            "allocation": alloc_b,
            "final_bankroll": bank_b.bankroll,
            "stats": stats_b,
        },
        "trader_c": {
            "name": "Trader C (Conviction)",
            "allocation": alloc_c,
            "final_bankroll": bank_c.bankroll,
            "stats": stats_c,
        },
        "combined": {
            "initial_bankroll": initial_bankroll,
            "final_bankroll": final_bankroll,
            "total_profit": combined_profit,
            "total_wagered": combined_wagered,
            "total_bets": combined_bets,
            "wins": combined_wins,
            "win_rate": combined_wins / combined_bets if combined_bets > 0 else 0,
            "roi": combined_profit / combined_wagered if combined_wagered > 0 else 0,
            "bankroll_growth": final_bankroll / initial_bankroll if initial_bankroll > 0 else 0,
            "max_drawdown": max_dd,
        },
        "bet_log": bet_log_df,
        "bankroll_history": bh_df,
    }

    return results


def _format_trader_stats(name: str, alloc: float, stats: dict, final: float) -> str:
    """Format a single trader's results."""
    bets = stats.get("total_bets", 0)
    wins = stats.get("wins", 0)
    wr = stats.get("win_rate", 0)
    profit = final - alloc
    roi = stats.get("roi", 0)
    wagered = stats.get("total_wagered", 0)
    avg_edge = stats.get("avg_edge", 0)
    avg_bet = stats.get("avg_bet_size", 0)

    lines = [
        f"  {name}",
        f"    Allocation: ${alloc:.2f} -> ${final:.2f} ({profit:+.2f})",
        f"    Bets: {bets} | Wins: {wins} | Win rate: {wr:.1%}",
        f"    Total wagered: ${wagered:.2f} | ROI: {roi:+.1%}",
        f"    Avg bet: ${avg_bet:.2f} | Avg edge: {avg_edge:.1%}" if bets > 0 else "",
    ]
    return "\n".join(l for l in lines if l)


def print_results(results: dict) -> None:
    """Print formatted backtest results."""
    c = results["combined"]
    print(f"\n{'='*60}")
    print("TRIPLE TRADER BACKTEST RESULTS")
    print(f"{'='*60}")

    for key in ["trader_a", "trader_b", "trader_c"]:
        t = results[key]
        print(_format_trader_stats(
            t["name"], t["allocation"], t["stats"], t["final_bankroll"]
        ))
        print()

    print(f"{'='*60}")
    print("COMBINED PORTFOLIO")
    print(f"{'='*60}")
    print(f"  Initial: ${c['initial_bankroll']:.2f} -> Final: ${c['final_bankroll']:.2f}")
    print(f"  Growth: {c['bankroll_growth']:.2f}x")
    print(f"  Total profit: ${c['total_profit']:+.2f}")
    print(f"  Total bets: {c['total_bets']} | Wins: {c['wins']} | Win rate: {c['win_rate']:.1%}")
    print(f"  Total wagered: ${c['total_wagered']:.2f} | ROI: {c['roi']:+.1%}")
    print(f"  Max drawdown: {c['max_drawdown']:.1%}")

    # Per-trader breakdown
    bet_log = results["bet_log"]
    if not bet_log.empty:
        print(f"\n{'='*60}")
        print("CONFLICT RESOLUTION STATS")
        print(f"{'='*60}")
        a_bets = len(bet_log[bet_log["trader"] == "A"])
        b_bets = len(bet_log[bet_log["trader"] == "B"])
        c_bets = len(bet_log[bet_log["trader"] == "C"])
        print(f"  Trader A placed: {a_bets} bets")
        print(f"  Trader B placed: {b_bets} bets")
        print(f"  Trader C placed: {c_bets} bets")

        # Check for fights where both A and B wanted to bet (would have been conflicts)
        # We can't directly count cancelled conflicts, but we can note unique fight coverage
        unique_fights = bet_log.groupby(["fighter_a", "fighter_b", "event_date"])["trader"].apply(list)
        multi_trader = unique_fights[unique_fights.apply(len) > 1]
        print(f"  Fights with multiple traders: {len(multi_trader)}")

    print(f"{'='*60}\n")


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LAB_DIR / "triple_trader_backtest.log"),
        ],
    )
    # Reduce noise from XGBoost and other libs
    logging.getLogger("src.strategy.bankroll").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Triple Trader Backtest")
    parser.add_argument(
        "--bankroll", type=float, default=INITIAL_BANKROLL,
        help=f"Starting bankroll (default: ${INITIAL_BANKROLL:.2f})",
    )
    parser.add_argument(
        "--start-date", type=str, default=TRAIN_CUTOFF_DATE,
        help=f"Only bet on fights after this date (default: {TRAIN_CUTOFF_DATE})",
    )
    args = parser.parse_args()

    results = run_triple_trader_backtest(
        initial_bankroll=args.bankroll,
        bet_start_date=args.start_date,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LAB_DIR / f"triple_trader_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not results["bet_log"].empty:
        results["bet_log"].to_csv(run_dir / "bet_log.csv", index=False)
    if not results["bankroll_history"].empty:
        results["bankroll_history"].to_csv(run_dir / "bankroll_history.csv", index=False)

    print_results(results)
    logger.info(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
