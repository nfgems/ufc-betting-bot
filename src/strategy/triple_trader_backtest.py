"""
Triple-Trader Backtest — simulates the full 3-trader system on historical data.

Three modes:
  --mode sweep (default):
    Parameter sweep over S+C duo trader configurations.
    Trains models once, then evaluates 36 parameter combos.
    Single Trader (S) + Conviction (C) + optional Medium Conviction (M).

  --mode priority:
    Priority Single Trader bets first on the FULL bankroll (blend=0.30, Kelly).
    If it bets, triple traders skip that fight (double-bet prevention).
    If it passes, triple traders split remaining balance 40/40/20.

  --mode classic:
    Original 3-trader system with fixed 40/40/20 bankroll split.
    - Trader A (Conservative): blend=0.20, value bets (Kelly)
    - Trader B (Aggressive):   blend=0.40, value bets (Kelly)
    - Trader C (Conviction):   flat sizing, 75% XGB + 60% no-odds agreement

Usage:
    python -m src.strategy.triple_trader_backtest
    python -m src.strategy.triple_trader_backtest --mode sweep --bankroll 1000
    python -m src.strategy.triple_trader_backtest --mode priority --bankroll 1000
    python -m src.strategy.triple_trader_backtest --mode classic --bankroll 1000
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime

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
    TRAIN_CUTOFF_DATE,
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

# Legacy backtest constants (removed from config.py, kept here for classic/priority modes)
TRADER_A_BLEND = 0.20
TRADER_B_BLEND = 0.40
TRADER_A_SHARE = 0.40
TRADER_B_SHARE = 0.40

LAB_DIR = LOGS_DIR / "model_lab"
LAB_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Sweep configuration
# ======================================================================

@dataclass
class SweepConfig:
    """Tunable parameters for the S+C duo trader sweep."""
    name: str
    s_min_edge: float = 0.025          # Single trader min edge threshold
    c_min_model_prob: float = 0.75     # Conviction XGB threshold
    c_min_no_odds_prob: float = 0.60   # Conviction no-odds threshold
    c_share: float = 0.20             # Conviction allocation (fraction of remaining after S)
    c_bet_fraction: float = 0.05      # Conviction base bet fraction
    c_max_bet_fraction: float = 0.08  # Conviction max bet fraction
    m_enabled: bool = False           # Medium conviction trader
    m_min_model_prob: float = 0.65    # Medium conviction XGB threshold
    m_min_no_odds_prob: float = 0.50  # Medium conviction no-odds threshold
    m_bet_fraction: float = 0.03     # Medium conviction flat sizing
    m_share: float = 0.30           # Medium conviction allocation


def _conviction_bet_size_param(model_prob: float, bankroll: float, config: SweepConfig) -> float:
    """Conviction bet sizing with configurable thresholds."""
    base = config.c_bet_fraction * bankroll
    excess = max(0.0, model_prob - config.c_min_model_prob)
    bonus_steps = excess / 0.05
    bonus = bonus_steps * CONVICTION_CONFIDENCE_BONUS * bankroll
    bet = min(base + bonus, config.c_max_bet_fraction * bankroll)
    return round(bet, 2) if bet >= 1.0 else 0.0


def _conviction_bet_size_backtest(model_prob: float, bankroll: float) -> float:
    """Conviction bet sizing (same as production)."""
    base = CONVICTION_BET_FRACTION * bankroll
    excess_confidence = max(0.0, model_prob - CONVICTION_MIN_MODEL_PROB)
    bonus_steps = excess_confidence / 0.05
    bonus = bonus_steps * CONVICTION_CONFIDENCE_BONUS * bankroll
    bet = min(base + bonus, CONVICTION_MAX_BET_FRACTION * bankroll)
    return round(bet, 2) if bet >= 1.0 else 0.0


def _kelly_size(blend_prob: float, market_prob: float, allocation: float) -> float:
    """Compute Kelly bet size for a sub-allocation of the shared bankroll."""
    odds = implied_prob_to_decimal_odds(market_prob)
    b = odds - 1.0
    p = blend_prob
    q = 1.0 - p
    if b <= 0 or p <= 0:
        return 0.0
    full_kelly = (b * p - q) / b
    if full_kelly <= 0:
        return 0.0
    frac = min(full_kelly * KELLY_FRACTION, MAX_BET_FRACTION)
    bet = allocation * frac
    return round(bet, 2) if bet >= 1.0 else 0.0


def _generate_walk_forward_predictions(
    features_df: pd.DataFrame = None,
    retrain_months: int = 6,
    initial_train_years: int = 5,
    bet_start_date: str = TRAIN_CUTOFF_DATE,
) -> list[tuple[int, pd.DataFrame]]:
    """
    Train models via walk-forward and generate predictions for all folds.

    Returns a list of (fold_num, predictions_df) tuples. Each predictions_df
    contains model predictions, market odds, no-odds predictions, and metadata
    needed by the betting loop.

    This is the expensive step (~2-3 min). Call once, then evaluate multiple
    parameter configs against the same predictions.
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

    variant = ALL_VARIANTS["baseline"]()
    bet_start = pd.Timestamp(bet_start_date)

    fold_predictions = []
    fold_num = 0

    while train_end < max_date:
        test_end = train_end + pd.DateOffset(months=retrain_months)
        if test_end > max_date:
            test_end = max_date + pd.Timedelta(days=1)

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
        fold_predictions.append((fold_num, predictions))
        train_end = test_end

    logger.info(f"Walk-forward complete: {fold_num} folds, "
                f"{sum(len(p) for _, p in fold_predictions)} total predictions")
    return fold_predictions


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


# ======================================================================
# PRIORITY MODE — Single trader first, triple traders on remaining balance
# ======================================================================

def run_priority_trader_backtest(
    features_df: pd.DataFrame = None,
    initial_bankroll: float = INITIAL_BANKROLL,
    retrain_months: int = 6,
    initial_train_years: int = 5,
    bet_start_date: str = TRAIN_CUTOFF_DATE,
) -> dict:
    """
    Priority Single Trader + Triple Trader backtest.

    The single trader evaluates first with the FULL bankroll (blend=0.30, Kelly).
    If it bets, the triple traders skip that fight (double-bet prevention).
    If it passes, the triple traders split the remaining balance 40/40/20.
    All traders share one bankroll — wins/losses flow to the same pool.
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

    # --- Single shared bankroll ---
    bank = BankrollManager(
        initial_bankroll=initial_bankroll, kelly_fraction=KELLY_FRACTION,
        max_bet_fraction=MAX_BET_FRACTION, auto_detect_balance=False,
    )

    variant = ALL_VARIANTS["baseline"]()

    bet_log = []
    bankroll_history = []
    fold_num = 0
    # Per-trader cumulative P&L for reporting
    pnl_s = 0.0  # single trader
    pnl_a = 0.0
    pnl_b = 0.0
    pnl_c = 0.0

    bet_start = pd.Timestamp(bet_start_date)

    logger.info(f"\n{'='*60}")
    logger.info("PRIORITY TRADER BACKTEST")
    logger.info(f"  Total bankroll: ${initial_bankroll:.2f}")
    logger.info(f"  Single Trader (Priority, blend={BLEND_WEIGHT}): full bankroll")
    logger.info(f"  Trader A (Conservative, blend={TRADER_A_BLEND}): {TRADER_A_SHARE:.0%} of remaining")
    logger.info(f"  Trader B (Aggressive, blend={TRADER_B_BLEND}): {TRADER_B_SHARE:.0%} of remaining")
    logger.info(f"  Trader C (Conviction): {TRADER_C_SHARE:.0%} of remaining")
    logger.info(f"  Betting on fights from: {bet_start_date}")
    logger.info(f"{'='*60}\n")

    while train_end < max_date:
        test_end = train_end + pd.DateOffset(months=retrain_months)
        if test_end > max_date:
            test_end = max_date + pd.Timedelta(days=1)

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
            fight_date = pd.Timestamp(row.get("event_date"))
            if fight_date < bet_start:
                continue

            if bank.is_stopped:
                break

            model_a = row["prob_a"]
            model_b = row["prob_b"]
            market_a = row["a_market_prob"]
            market_b = row["b_market_prob"]

            if pd.isna(market_a) or pd.isna(market_b):
                bankroll_history.append({"combined": bank.bankroll})
                continue

            actual_winner_is_a = row["target"] == 1
            no_odds_a = row.get("no_odds_prob_a")
            no_odds_b = row.get("no_odds_prob_b")
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
            # SINGLE TRADER (Priority): blend=0.30, full bankroll Kelly
            # ============================================================
            single_bet = None
            dyn_w_s = dynamic_blend_weight(model_a, market_a, no_odds_a, BLEND_WEIGHT)
            blend_s_a = blend_probability(model_a, market_a, dyn_w_s)
            blend_s_b = 1.0 - blend_s_a
            edge_s_a = blend_s_a - market_a
            edge_s_b = blend_s_b - market_b

            if edge_s_a >= min_edge and edge_s_a >= edge_s_b and _passes_filters(
                blend_s_a, market_a, edge_s_a, row.get("fighter_a", "A"), no_odds_a,
                line_movement=line_movement, line_is_sharp=line_is_sharp,
                line_steam_move=line_steam_move, bet_side="a",
                a_num_fights=a_fights, b_num_fights=b_fights,
            ):
                odds = implied_prob_to_decimal_odds(market_a)
                bet_size = bank.kelly_bet_size(blend_s_a, odds)
                if bet_size > 0:
                    single_bet = {"side": "a", "blend": blend_s_a, "market": market_a,
                                  "edge": edge_s_a, "size": bet_size}
            elif edge_s_b >= min_edge and _passes_filters(
                blend_s_b, market_b, edge_s_b, row.get("fighter_b", "B"), no_odds_b,
                line_movement=line_movement, line_is_sharp=line_is_sharp,
                line_steam_move=line_steam_move, bet_side="b",
                a_num_fights=a_fights, b_num_fights=b_fights,
            ):
                odds = implied_prob_to_decimal_odds(market_b)
                bet_size = bank.kelly_bet_size(blend_s_b, odds)
                if bet_size > 0:
                    single_bet = {"side": "b", "blend": blend_s_b, "market": market_b,
                                  "edge": edge_s_b, "size": bet_size}

            if single_bet is not None:
                # Place and settle single trader bet
                side = single_bet["side"]
                odds = implied_prob_to_decimal_odds(single_bet["market"])
                bet_idx = len(bank.history)
                fighter = row.get("fighter_a" if side == "a" else "fighter_b", "?")
                bank.place_bet(single_bet["size"], fighter, odds,
                               single_bet["blend"], single_bet["market"])
                won = actual_winner_is_a if side == "a" else not actual_winner_is_a
                bank.settle_bet(bet_idx, won=won)
                profit = bank.history[-1]["profit"]
                pnl_s += profit

                bet_log.append({
                    "trader": "S", "event_date": row.get("event_date"),
                    "fighter_a": row.get("fighter_a", ""),
                    "fighter_b": row.get("fighter_b", ""),
                    "bet_on": fighter, "side": side,
                    "bet_size": single_bet["size"], "odds": odds,
                    "edge": single_bet["edge"], "won": won,
                    "profit": profit,
                    "bankroll_after": bank.bankroll, "fold": fold_num,
                })

                bankroll_history.append({"combined": bank.bankroll})
                continue  # Skip triple traders for this fight

            # ============================================================
            # TRIPLE TRADERS (on remaining balance)
            # ============================================================
            remaining = bank.bankroll
            alloc_a = remaining * TRADER_A_SHARE
            alloc_b = remaining * TRADER_B_SHARE
            alloc_c = remaining * TRADER_C_SHARE

            # --- TRADER A: Conservative (blend=0.20) ---
            bet_a = None
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

            # --- TRADER B: Aggressive (blend=0.40) ---
            bet_b = None
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

            # --- TRADER C: Conviction ---
            bet_c = None
            for side, mp, mrkt, nop, fighter in [
                ("a", model_a, market_a, no_odds_a, row.get("fighter_a", "A")),
                ("b", model_b, market_b, no_odds_b, row.get("fighter_b", "B")),
            ]:
                if mp >= CONVICTION_MIN_MODEL_PROB and nop is not None and nop >= CONVICTION_MIN_NO_ODDS_PROB:
                    if a_fights is not None and a_fights < MIN_FIGHTER_FIGHTS:
                        continue
                    if b_fights is not None and b_fights < MIN_FIGHTER_FIGHTS:
                        continue
                    bet_c = {"side": side, "model_prob": mp, "market": mrkt, "fighter": fighter}
                    break

            # --- CONFLICT RESOLUTION ---
            if bet_a is not None and bet_b is not None:
                if bet_a["side"] != bet_b["side"]:
                    bet_a = None
                    bet_b = None
                else:
                    if bet_a["edge"] >= bet_b["edge"]:
                        bet_b = None
                    else:
                        bet_a = None

            if bet_c is not None:
                for vb in [bet_a, bet_b]:
                    if vb is not None and vb["side"] != bet_c["side"]:
                        bet_c = None
                        break

            # --- EXECUTE TRIPLE TRADER BETS ---
            if bet_a is not None:
                bet_size = _kelly_size(bet_a["blend"], bet_a["market"], alloc_a)
                if bet_size > 0:
                    odds = implied_prob_to_decimal_odds(bet_a["market"])
                    bet_idx = len(bank.history)
                    fighter = row.get("fighter_a" if bet_a["side"] == "a" else "fighter_b", "?")
                    bank.place_bet(bet_size, fighter, odds, bet_a["blend"], bet_a["market"])
                    won = actual_winner_is_a if bet_a["side"] == "a" else not actual_winner_is_a
                    bank.settle_bet(bet_idx, won=won)
                    profit = bank.history[-1]["profit"]
                    pnl_a += profit
                    bet_log.append({
                        "trader": "A", "event_date": row.get("event_date"),
                        "fighter_a": row.get("fighter_a", ""),
                        "fighter_b": row.get("fighter_b", ""),
                        "bet_on": fighter, "side": bet_a["side"],
                        "bet_size": bet_size, "odds": odds,
                        "edge": bet_a["edge"], "won": won,
                        "profit": profit,
                        "bankroll_after": bank.bankroll, "fold": fold_num,
                    })

            if bet_b is not None:
                bet_size = _kelly_size(bet_b["blend"], bet_b["market"], alloc_b)
                if bet_size > 0:
                    odds = implied_prob_to_decimal_odds(bet_b["market"])
                    bet_idx = len(bank.history)
                    fighter = row.get("fighter_a" if bet_b["side"] == "a" else "fighter_b", "?")
                    bank.place_bet(bet_size, fighter, odds, bet_b["blend"], bet_b["market"])
                    won = actual_winner_is_a if bet_b["side"] == "a" else not actual_winner_is_a
                    bank.settle_bet(bet_idx, won=won)
                    profit = bank.history[-1]["profit"]
                    pnl_b += profit
                    bet_log.append({
                        "trader": "B", "event_date": row.get("event_date"),
                        "fighter_a": row.get("fighter_a", ""),
                        "fighter_b": row.get("fighter_b", ""),
                        "bet_on": fighter, "side": bet_b["side"],
                        "bet_size": bet_size, "odds": odds,
                        "edge": bet_b["edge"], "won": won,
                        "profit": profit,
                        "bankroll_after": bank.bankroll, "fold": fold_num,
                    })

            if bet_c is not None:
                bet_size = _conviction_bet_size_backtest(bet_c["model_prob"], alloc_c)
                if bet_size > 0:
                    odds = implied_prob_to_decimal_odds(bet_c["market"])
                    bet_idx = len(bank.history)
                    bank.place_bet(bet_size, bet_c["fighter"], odds,
                                   bet_c["model_prob"], bet_c["market"])
                    won = actual_winner_is_a if bet_c["side"] == "a" else not actual_winner_is_a
                    bank.settle_bet(bet_idx, won=won)
                    profit = bank.history[-1]["profit"]
                    pnl_c += profit
                    bet_log.append({
                        "trader": "C", "event_date": row.get("event_date"),
                        "fighter_a": row.get("fighter_a", ""),
                        "fighter_b": row.get("fighter_b", ""),
                        "bet_on": bet_c["fighter"], "side": bet_c["side"],
                        "bet_size": bet_size, "odds": odds,
                        "edge": bet_c["model_prob"] - bet_c["market"], "won": won,
                        "profit": profit,
                        "bankroll_after": bank.bankroll, "fold": fold_num,
                    })

            bankroll_history.append({"combined": bank.bankroll})

        train_end = test_end

    # --- Results ---
    bet_log_df = pd.DataFrame(bet_log)
    bh_df = pd.DataFrame(bankroll_history)

    final_bankroll = bank.bankroll
    total_profit = final_bankroll - initial_bankroll

    # Per-trader stats from bet log
    def _trader_stats(trader_code: str) -> dict:
        if bet_log_df.empty:
            return {"total_bets": 0, "wins": 0, "win_rate": 0, "total_wagered": 0,
                    "roi": 0, "avg_edge": 0, "avg_bet_size": 0}
        t = bet_log_df[bet_log_df["trader"] == trader_code]
        if t.empty:
            return {"total_bets": 0, "wins": 0, "win_rate": 0, "total_wagered": 0,
                    "roi": 0, "avg_edge": 0, "avg_bet_size": 0}
        bets = len(t)
        wins = int(t["won"].sum())
        wagered = t["bet_size"].sum()
        profit = t["profit"].sum()
        return {
            "total_bets": bets, "wins": wins,
            "win_rate": wins / bets if bets > 0 else 0,
            "total_wagered": wagered,
            "roi": profit / wagered if wagered > 0 else 0,
            "avg_edge": t["edge"].mean() if bets > 0 else 0,
            "avg_bet_size": wagered / bets if bets > 0 else 0,
        }

    stats_s = _trader_stats("S")
    stats_a = _trader_stats("A")
    stats_b = _trader_stats("B")
    stats_c = _trader_stats("C")

    total_bets = stats_s["total_bets"] + stats_a["total_bets"] + stats_b["total_bets"] + stats_c["total_bets"]
    total_wins = stats_s["wins"] + stats_a["wins"] + stats_b["wins"] + stats_c["wins"]
    total_wagered = stats_s["total_wagered"] + stats_a["total_wagered"] + stats_b["total_wagered"] + stats_c["total_wagered"]

    max_dd = 0.0
    if not bh_df.empty:
        peak = bh_df["combined"].expanding().max()
        drawdown = 1.0 - bh_df["combined"] / peak
        max_dd = drawdown.max()

    results = {
        "mode": "priority",
        "trader_s": {
            "name": f"Single Trader (Priority, blend={BLEND_WEIGHT})",
            "allocation": initial_bankroll,
            "final_pnl": pnl_s,
            "stats": stats_s,
        },
        "trader_a": {
            "name": f"Trader A (Conservative, blend={TRADER_A_BLEND})",
            "allocation": 0,  # dynamic, no fixed allocation
            "final_pnl": pnl_a,
            "stats": stats_a,
        },
        "trader_b": {
            "name": f"Trader B (Aggressive, blend={TRADER_B_BLEND})",
            "allocation": 0,
            "final_pnl": pnl_b,
            "stats": stats_b,
        },
        "trader_c": {
            "name": "Trader C (Conviction)",
            "allocation": 0,
            "final_pnl": pnl_c,
            "stats": stats_c,
        },
        "combined": {
            "initial_bankroll": initial_bankroll,
            "final_bankroll": final_bankroll,
            "total_profit": total_profit,
            "total_wagered": total_wagered,
            "total_bets": total_bets,
            "wins": total_wins,
            "win_rate": total_wins / total_bets if total_bets > 0 else 0,
            "roi": total_profit / total_wagered if total_wagered > 0 else 0,
            "bankroll_growth": final_bankroll / initial_bankroll if initial_bankroll > 0 else 0,
            "max_drawdown": max_dd,
        },
        "bet_log": bet_log_df,
        "bankroll_history": bh_df,
    }

    return results


# ======================================================================
# SWEEP MODE — train once, evaluate many S+C configs
# ======================================================================

def _parse_fight_row(row) -> dict:
    """Extract and clean fight data from a prediction row."""
    model_a = row["prob_a"]
    model_b = row["prob_b"]
    market_a = row["a_market_prob"]
    market_b = row["b_market_prob"]

    no_odds_a = row.get("no_odds_prob_a")
    no_odds_b = row.get("no_odds_prob_b")
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

    return {
        "model_a": model_a, "model_b": model_b,
        "market_a": market_a, "market_b": market_b,
        "no_odds_a": no_odds_a, "no_odds_b": no_odds_b,
        "line_movement": line_movement, "line_is_sharp": line_is_sharp,
        "line_steam_move": line_steam_move,
        "a_fights": a_fights, "b_fights": b_fights,
        "actual_winner_is_a": row["target"] == 1,
        "fighter_a": row.get("fighter_a", "A"),
        "fighter_b": row.get("fighter_b", "B"),
        "event_date": row.get("event_date"),
    }


def _evaluate_config(
    fold_predictions: list[tuple[int, pd.DataFrame]],
    config: SweepConfig,
    initial_bankroll: float = INITIAL_BANKROLL,
    bet_start_date: str = TRAIN_CUTOFF_DATE,
) -> dict:
    """
    Evaluate a single S+C(+M) configuration against precomputed predictions.

    This is the fast inner loop — no model training, just the betting logic.
    Returns a results dict with per-trader stats and combined portfolio metrics.
    """
    bank = BankrollManager(
        initial_bankroll=initial_bankroll, kelly_fraction=KELLY_FRACTION,
        max_bet_fraction=MAX_BET_FRACTION, auto_detect_balance=False,
    )

    bet_log = []
    bankroll_history = []
    pnl_s = 0.0
    pnl_c = 0.0
    pnl_m = 0.0

    bet_start = pd.Timestamp(bet_start_date)

    for fold_num, predictions in fold_predictions:
        for _, row in predictions.iterrows():
            fight_date = pd.Timestamp(row.get("event_date"))
            if fight_date < bet_start:
                continue

            if bank.is_stopped:
                break

            f = _parse_fight_row(row)

            if pd.isna(f["market_a"]) or pd.isna(f["market_b"]):
                bankroll_history.append({"combined": bank.bankroll})
                continue

            min_edge = config.s_min_edge

            # ============================================================
            # SINGLE TRADER (S): blend=0.30, full bankroll Kelly
            # ============================================================
            single_bet = None
            dyn_w_s = dynamic_blend_weight(f["model_a"], f["market_a"], f["no_odds_a"], BLEND_WEIGHT)
            blend_s_a = blend_probability(f["model_a"], f["market_a"], dyn_w_s)
            blend_s_b = 1.0 - blend_s_a
            edge_s_a = blend_s_a - f["market_a"]
            edge_s_b = blend_s_b - f["market_b"]

            if edge_s_a >= min_edge and edge_s_a >= edge_s_b and _passes_filters(
                blend_s_a, f["market_a"], edge_s_a, f["fighter_a"], f["no_odds_a"],
                line_movement=f["line_movement"], line_is_sharp=f["line_is_sharp"],
                line_steam_move=f["line_steam_move"], bet_side="a",
                a_num_fights=f["a_fights"], b_num_fights=f["b_fights"],
                edge_scaling_base=config.s_min_edge,
            ):
                odds = implied_prob_to_decimal_odds(f["market_a"])
                bet_size = bank.kelly_bet_size(blend_s_a, odds)
                if bet_size > 0:
                    single_bet = {"side": "a", "blend": blend_s_a, "market": f["market_a"],
                                  "edge": edge_s_a, "size": bet_size}
            elif edge_s_b >= min_edge and _passes_filters(
                blend_s_b, f["market_b"], edge_s_b, f["fighter_b"], f["no_odds_b"],
                line_movement=f["line_movement"], line_is_sharp=f["line_is_sharp"],
                line_steam_move=f["line_steam_move"], bet_side="b",
                a_num_fights=f["a_fights"], b_num_fights=f["b_fights"],
                edge_scaling_base=config.s_min_edge,
            ):
                odds = implied_prob_to_decimal_odds(f["market_b"])
                bet_size = bank.kelly_bet_size(blend_s_b, odds)
                if bet_size > 0:
                    single_bet = {"side": "b", "blend": blend_s_b, "market": f["market_b"],
                                  "edge": edge_s_b, "size": bet_size}

            if single_bet is not None:
                side = single_bet["side"]
                odds = implied_prob_to_decimal_odds(single_bet["market"])
                bet_idx = len(bank.history)
                fighter = f["fighter_a"] if side == "a" else f["fighter_b"]
                bank.place_bet(single_bet["size"], fighter, odds,
                               single_bet["blend"], single_bet["market"])
                won = f["actual_winner_is_a"] if side == "a" else not f["actual_winner_is_a"]
                bank.settle_bet(bet_idx, won=won)
                profit = bank.history[-1]["profit"]
                pnl_s += profit
                bet_log.append({
                    "trader": "S", "event_date": f["event_date"],
                    "fighter_a": f["fighter_a"], "fighter_b": f["fighter_b"],
                    "bet_on": fighter, "side": side,
                    "bet_size": single_bet["size"], "odds": odds,
                    "edge": single_bet["edge"], "won": won,
                    "profit": profit, "bankroll_after": bank.bankroll, "fold": fold_num,
                })
                bankroll_history.append({"combined": bank.bankroll})
                continue  # Skip C/M for this fight

            # ============================================================
            # CONVICTION TRADER (C)
            # ============================================================
            remaining = bank.bankroll
            alloc_c = remaining * config.c_share

            bet_c = None
            for side, mp, mrkt, nop, fighter in [
                ("a", f["model_a"], f["market_a"], f["no_odds_a"], f["fighter_a"]),
                ("b", f["model_b"], f["market_b"], f["no_odds_b"], f["fighter_b"]),
            ]:
                if (mp >= config.c_min_model_prob
                        and nop is not None
                        and nop >= config.c_min_no_odds_prob):
                    if f["a_fights"] is not None and f["a_fights"] < MIN_FIGHTER_FIGHTS:
                        continue
                    if f["b_fights"] is not None and f["b_fights"] < MIN_FIGHTER_FIGHTS:
                        continue
                    bet_c = {"side": side, "model_prob": mp, "market": mrkt, "fighter": fighter}
                    break

            if bet_c is not None:
                bet_size = _conviction_bet_size_param(bet_c["model_prob"], alloc_c, config)
                if bet_size > 0:
                    odds = implied_prob_to_decimal_odds(bet_c["market"])
                    bet_idx = len(bank.history)
                    bank.place_bet(bet_size, bet_c["fighter"], odds,
                                   bet_c["model_prob"], bet_c["market"])
                    won = f["actual_winner_is_a"] if bet_c["side"] == "a" else not f["actual_winner_is_a"]
                    bank.settle_bet(bet_idx, won=won)
                    profit = bank.history[-1]["profit"]
                    pnl_c += profit
                    bet_log.append({
                        "trader": "C", "event_date": f["event_date"],
                        "fighter_a": f["fighter_a"], "fighter_b": f["fighter_b"],
                        "bet_on": bet_c["fighter"], "side": bet_c["side"],
                        "bet_size": bet_size, "odds": odds,
                        "edge": bet_c["model_prob"] - bet_c["market"], "won": won,
                        "profit": profit, "bankroll_after": bank.bankroll, "fold": fold_num,
                    })
                    bankroll_history.append({"combined": bank.bankroll})
                    continue  # Skip M if C bet

            # ============================================================
            # MEDIUM CONVICTION (M) — only if S and C both passed
            # ============================================================
            if config.m_enabled:
                alloc_m = remaining * config.m_share

                for side, mp, mrkt, nop, fighter in [
                    ("a", f["model_a"], f["market_a"], f["no_odds_a"], f["fighter_a"]),
                    ("b", f["model_b"], f["market_b"], f["no_odds_b"], f["fighter_b"]),
                ]:
                    if (mp >= config.m_min_model_prob
                            and nop is not None
                            and nop >= config.m_min_no_odds_prob):
                        if f["a_fights"] is not None and f["a_fights"] < MIN_FIGHTER_FIGHTS:
                            continue
                        if f["b_fights"] is not None and f["b_fights"] < MIN_FIGHTER_FIGHTS:
                            continue
                        # Flat sizing
                        bet_size = round(config.m_bet_fraction * alloc_m, 2)
                        if bet_size >= 1.0:
                            odds = implied_prob_to_decimal_odds(mrkt)
                            bet_idx = len(bank.history)
                            bank.place_bet(bet_size, fighter, odds, mp, mrkt)
                            won = f["actual_winner_is_a"] if side == "a" else not f["actual_winner_is_a"]
                            bank.settle_bet(bet_idx, won=won)
                            profit = bank.history[-1]["profit"]
                            pnl_m += profit
                            bet_log.append({
                                "trader": "M", "event_date": f["event_date"],
                                "fighter_a": f["fighter_a"], "fighter_b": f["fighter_b"],
                                "bet_on": fighter, "side": side,
                                "bet_size": bet_size, "odds": odds,
                                "edge": mp - mrkt, "won": won,
                                "profit": profit, "bankroll_after": bank.bankroll, "fold": fold_num,
                            })
                        break  # Take first qualifying side

            bankroll_history.append({"combined": bank.bankroll})

    # --- Results ---
    bet_log_df = pd.DataFrame(bet_log)
    bh_df = pd.DataFrame(bankroll_history)

    final_bankroll = bank.bankroll
    total_profit = final_bankroll - initial_bankroll

    def _ts(code: str) -> dict:
        if bet_log_df.empty:
            return {"total_bets": 0, "wins": 0, "win_rate": 0, "total_wagered": 0,
                    "roi": 0, "avg_edge": 0, "avg_bet_size": 0}
        t = bet_log_df[bet_log_df["trader"] == code]
        if t.empty:
            return {"total_bets": 0, "wins": 0, "win_rate": 0, "total_wagered": 0,
                    "roi": 0, "avg_edge": 0, "avg_bet_size": 0}
        bets = len(t)
        wins = int(t["won"].sum())
        wagered = t["bet_size"].sum()
        profit = t["profit"].sum()
        return {
            "total_bets": bets, "wins": wins,
            "win_rate": wins / bets if bets > 0 else 0,
            "total_wagered": wagered,
            "roi": profit / wagered if wagered > 0 else 0,
            "avg_edge": t["edge"].mean() if bets > 0 else 0,
            "avg_bet_size": wagered / bets if bets > 0 else 0,
        }

    stats_s = _ts("S")
    stats_c = _ts("C")
    stats_m = _ts("M")

    total_bets = stats_s["total_bets"] + stats_c["total_bets"] + stats_m["total_bets"]
    total_wins = stats_s["wins"] + stats_c["wins"] + stats_m["wins"]
    total_wagered = stats_s["total_wagered"] + stats_c["total_wagered"] + stats_m["total_wagered"]

    max_dd = 0.0
    if not bh_df.empty:
        peak = bh_df["combined"].expanding().max()
        drawdown = 1.0 - bh_df["combined"] / peak
        max_dd = drawdown.max()

    return {
        "config": config,
        "mode": "sweep",
        "trader_s": {"name": f"Single Trader (edge={config.s_min_edge})", "final_pnl": pnl_s, "stats": stats_s},
        "trader_c": {"name": f"Conviction (>={config.c_min_model_prob}/{config.c_min_no_odds_prob}, alloc={config.c_share})",
                     "final_pnl": pnl_c, "stats": stats_c},
        "trader_m": {"name": f"Medium Conv (>={config.m_min_model_prob}/{config.m_min_no_odds_prob})",
                     "final_pnl": pnl_m, "stats": stats_m} if config.m_enabled else None,
        "combined": {
            "initial_bankroll": initial_bankroll,
            "final_bankroll": final_bankroll,
            "total_profit": total_profit,
            "total_wagered": total_wagered,
            "total_bets": total_bets,
            "wins": total_wins,
            "win_rate": total_wins / total_bets if total_bets > 0 else 0,
            "roi": total_profit / total_wagered if total_wagered > 0 else 0,
            "bankroll_growth": final_bankroll / initial_bankroll if initial_bankroll > 0 else 0,
            "max_drawdown": max_dd,
        },
        "bet_log": bet_log_df,
        "bankroll_history": bh_df,
    }


def _build_sweep_configs() -> list[SweepConfig]:
    """Build the grid of parameter combinations to sweep."""
    configs = []
    for s_edge in [0.020, 0.025]:
        for c_model, c_no_odds in [(0.75, 0.60), (0.70, 0.55), (0.65, 0.50)]:
            for c_share in [0.20, 0.50, 1.00]:
                for m_enabled in [False, True]:
                    name_parts = [
                        f"edge={s_edge}",
                        f"conv={c_model}/{c_no_odds}",
                        f"alloc={c_share}",
                    ]
                    if m_enabled:
                        name_parts.append("+medium")
                    name = "_".join(name_parts)
                    configs.append(SweepConfig(
                        name=name,
                        s_min_edge=s_edge,
                        c_min_model_prob=c_model,
                        c_min_no_odds_prob=c_no_odds,
                        c_share=c_share,
                        m_enabled=m_enabled,
                    ))
    return configs


def _print_sweep_results(summary: list[dict]) -> None:
    """Print a formatted comparison table of sweep results."""
    print(f"\n{'='*120}")
    print(f"PARAMETER SWEEP RESULTS ({len(summary)} configs, sorted by profit)")
    print(f"{'='*120}")
    print(f"{'Rank':>4}  {'Config':<50}  {'S':>4} {'C':>4} {'M':>4} {'Total':>5}  "
          f"{'WR':>6}  {'Profit':>10}  {'ROI':>7}  {'MaxDD':>6}")
    print(f"{'----':>4}  {'------':<50}  {'--':>4} {'--':>4} {'--':>4} {'-----':>5}  "
          f"{'--':>6}  {'------':>10}  {'---':>7}  {'-----':>6}")

    for i, row in enumerate(summary, 1):
        print(f"{i:>4}  {row['config']:<50}  "
              f"{row['s_bets']:>4} {row['c_bets']:>4} {row['m_bets']:>4} {row['total_bets']:>5}  "
              f"{row['win_rate']:>5.1%}  "
              f"${row['profit']:>+9.2f}  "
              f"{row['roi']:>+6.1%}  "
              f"{row['max_dd']:>5.1%}")

    print(f"{'='*120}\n")


def run_sweep_backtest(
    initial_bankroll: float = INITIAL_BANKROLL,
    bet_start_date: str = TRAIN_CUTOFF_DATE,
) -> tuple[list[dict], dict[str, dict]]:
    """
    Run parameter sweep: train once, evaluate 36 S+C configs.

    Returns (summary_rows, top_results) where summary_rows is a list of
    dicts for the comparison table and top_results maps config names to
    full result dicts (for saving bet logs).
    """
    logger.info("Starting parameter sweep...")
    logger.info("Phase 1: Generating walk-forward predictions (train once)...")

    fold_predictions = _generate_walk_forward_predictions(
        bet_start_date=bet_start_date,
    )

    configs = _build_sweep_configs()
    logger.info(f"Phase 2: Evaluating {len(configs)} configurations...")

    summary_rows = []
    all_results = {}

    for i, config in enumerate(configs, 1):
        result = _evaluate_config(
            fold_predictions, config,
            initial_bankroll=initial_bankroll,
            bet_start_date=bet_start_date,
        )
        all_results[config.name] = result

        c = result["combined"]
        s = result["trader_s"]["stats"]
        cs = result["trader_c"]["stats"]
        ms = result["trader_m"]["stats"] if result["trader_m"] else {"total_bets": 0}

        summary_rows.append({
            "config": config.name,
            "s_bets": s["total_bets"],
            "c_bets": cs["total_bets"],
            "m_bets": ms["total_bets"],
            "total_bets": c["total_bets"],
            "wins": c["wins"],
            "win_rate": c["win_rate"],
            "profit": c["total_profit"],
            "roi": c["roi"],
            "max_dd": c["max_drawdown"],
            "final_bankroll": c["final_bankroll"],
            "total_wagered": c["total_wagered"],
        })

        if i % 10 == 0:
            logger.info(f"  Evaluated {i}/{len(configs)} configs...")

    # Sort by profit descending
    summary_rows.sort(key=lambda x: x["profit"], reverse=True)

    _print_sweep_results(summary_rows)

    # Return top 5 results for saving
    top_names = [r["config"] for r in summary_rows[:5]]
    top_results = {name: all_results[name] for name in top_names}

    return summary_rows, top_results


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


def _format_priority_trader(name: str, pnl: float, stats: dict) -> str:
    """Format a trader's results for priority mode (no fixed allocation)."""
    bets = stats.get("total_bets", 0)
    wins = stats.get("wins", 0)
    wr = stats.get("win_rate", 0)
    roi = stats.get("roi", 0)
    wagered = stats.get("total_wagered", 0)
    avg_edge = stats.get("avg_edge", 0)
    avg_bet = stats.get("avg_bet_size", 0)

    lines = [
        f"  {name}",
        f"    P&L: ${pnl:+.2f}",
        f"    Bets: {bets} | Wins: {wins} | Win rate: {wr:.1%}",
        f"    Total wagered: ${wagered:.2f} | ROI: {roi:+.1%}",
        f"    Avg bet: ${avg_bet:.2f} | Avg edge: {avg_edge:.1%}" if bets > 0 else "",
    ]
    return "\n".join(l for l in lines if l)


def print_results(results: dict) -> None:
    """Print formatted backtest results."""
    c = results["combined"]
    is_priority = results.get("mode") == "priority"

    print(f"\n{'='*60}")
    print("PRIORITY TRADER BACKTEST RESULTS" if is_priority else "TRIPLE TRADER BACKTEST RESULTS")
    print(f"{'='*60}")

    if is_priority:
        # Priority mode: show single trader first, then A/B/C
        for key in ["trader_s", "trader_a", "trader_b", "trader_c"]:
            t = results[key]
            print(_format_priority_trader(t["name"], t["final_pnl"], t["stats"]))
            print()
    else:
        # Classic mode
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
        print("BET DISTRIBUTION")
        print(f"{'='*60}")
        for code, label in [("S", "Single Trader"), ("A", "Trader A"),
                             ("B", "Trader B"), ("C", "Trader C")]:
            count = len(bet_log[bet_log["trader"] == code])
            if count > 0:
                print(f"  {label}: {count} bets")

        unique_fights = bet_log.groupby(["fighter_a", "fighter_b", "event_date"])["trader"].apply(list)
        multi_trader = unique_fights[unique_fights.apply(len) > 1]
        if len(multi_trader) > 0:
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
        "--mode", type=str, default="sweep", choices=["sweep", "priority", "classic"],
        help="sweep = parameter sweep over S+C configs; "
             "priority = single trader first + triple traders on remainder; "
             "classic = original 3-trader system (default: sweep)",
    )
    parser.add_argument(
        "--bankroll", type=float, default=INITIAL_BANKROLL,
        help=f"Starting bankroll (default: ${INITIAL_BANKROLL:.2f})",
    )
    parser.add_argument(
        "--start-date", type=str, default=TRAIN_CUTOFF_DATE,
        help=f"Only bet on fights after this date (default: {TRAIN_CUTOFF_DATE})",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode == "sweep":
        summary_rows, top_results = run_sweep_backtest(
            initial_bankroll=args.bankroll,
            bet_start_date=args.start_date,
        )

        # Save results
        run_dir = LAB_DIR / f"sweep_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(run_dir / "sweep_summary.csv", index=False)

        for name, result in top_results.items():
            safe_name = name.replace("/", "-").replace("=", "")
            if not result["bet_log"].empty:
                result["bet_log"].to_csv(run_dir / f"bets_{safe_name}.csv", index=False)

        logger.info(f"Sweep results saved to {run_dir}")
        return

    if args.mode == "priority":
        results = run_priority_trader_backtest(
            initial_bankroll=args.bankroll,
            bet_start_date=args.start_date,
        )
    else:
        results = run_triple_trader_backtest(
            initial_bankroll=args.bankroll,
            bet_start_date=args.start_date,
        )

    # Save results
    run_dir = LAB_DIR / f"{args.mode}_trader_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not results["bet_log"].empty:
        results["bet_log"].to_csv(run_dir / "bet_log.csv", index=False)
    if not results["bankroll_history"].empty:
        results["bankroll_history"].to_csv(run_dir / "bankroll_history.csv", index=False)

    print_results(results)
    logger.info(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
