"""S+C duo-trader sweep and BetsAPI challenger comparison helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    BLEND_WEIGHT,
    CONVICTION_CONFIDENCE_BONUS,
    CONVICTION_MIN_MODEL_PROB,
    CONVICTION_MIN_NO_ODDS_PROB,
    INITIAL_BANKROLL,
    KELLY_FRACTION,
    MAX_BET_FRACTION,
    MIN_EDGE_THRESHOLD,
    MIN_FIGHTER_FIGHTS,
    MODEL_AGREEMENT_MIN_EDGE,
    PROCESSED_DATA_DIR,
    TRAIN_CUTOFF_DATE,
    TRADER_C_SHARE,
)
from src.strategy.backtest import (
    BacktestExecutionConfig,
    _estimated_best_ask,
    _resolve_execution_config,
    _simulate_realistic_fill,
)
from src.strategy.bankroll import BankrollManager
from src.strategy.lab_stats import compute_max_drawdown
from src.strategy.model_lab import (
    build_variant_features,
    generate_variant_fold_predictions,
)
from src.strategy.model_variants import ALL_VARIANTS, apply_variant_feature_transforms
from src.strategy.value import (
    _passes_filters,
    blend_probability,
    calculate_closing_line_value,
    compute_independent_blend_probs,
    dynamic_blend_weight,
    implied_prob_to_decimal_odds,
)

logger = logging.getLogger(__name__)


@dataclass
class SweepConfig:
    """Tunable parameters for the S+C duo-trader sweep."""

    name: str
    variant_name: str = "baseline"
    is_baseline: bool = False
    s_min_edge: float = MIN_EDGE_THRESHOLD
    s_blend_weight: float = BLEND_WEIGHT
    s_model_agreement_min_edge: float = MODEL_AGREEMENT_MIN_EDGE
    c_min_model_prob: float = CONVICTION_MIN_MODEL_PROB
    c_min_no_odds_prob: float = CONVICTION_MIN_NO_ODDS_PROB
    c_share: float = TRADER_C_SHARE
    c_bet_fraction: float = 0.05
    c_max_bet_fraction: float = 0.08
    c_min_edge: float = 0.0
    c_max_decimal_odds: float = float("inf")
    m_enabled: bool = False
    m_min_model_prob: float = 0.65
    m_min_no_odds_prob: float = 0.50
    m_bet_fraction: float = 0.03
    m_share: float = 0.30


def _conviction_bet_size_param(model_prob: float, bankroll: float, config: SweepConfig) -> float:
    """Conviction bet sizing with configurable thresholds."""
    base = config.c_bet_fraction * bankroll
    excess = max(0.0, model_prob - config.c_min_model_prob)
    bonus_steps = excess / 0.05
    bonus = bonus_steps * CONVICTION_CONFIDENCE_BONUS * bankroll
    bet = min(base + bonus, config.c_max_bet_fraction * bankroll)
    return round(bet, 2) if bet >= 1.0 else 0.0


def _load_features_for_variant(
    features_df: pd.DataFrame | None,
    variant_name: str,
    dataset_variant: str | None = None,
    feature_family: str | None = None,
) -> pd.DataFrame:
    """Load cached UFC features and optionally augment them for challenger runs.

    Parameters
    ----------
    features_df : pd.DataFrame | None
        Pre-loaded features.  When *None*, features are loaded from the
        processed cache or built from scratch.
    variant_name : str
        Model variant name (e.g. ``"baseline"``, ``"betsapi_challenger"``).
    dataset_variant : str | None
        If provided, build from the named dataset variant (e.g.
        ``"append_only_2026"``) instead of the default features.csv cache.
    feature_family : str | None
        If provided, determines which BetsAPI augmentation (if any) is
        applied.  Families containing ``"betsapi"`` trigger augmentation.
    """
    variant = ALL_VARIANTS[variant_name]()

    if features_df is None:
        if dataset_variant is not None:
            logger.info(
                "Building features for dataset variant '%s'...", dataset_variant
            )
            from src.data.kaggle_loader import load_kaggle_dataset
            from src.config import RAW_DATA_DIR
            from src.data.ufc_refresh import build_training_dataset_variants

            legacy_df = load_kaggle_dataset(RAW_DATA_DIR / "ufc-master.csv")
            all_variants = build_training_dataset_variants(legacy_df=legacy_df)
            if dataset_variant not in all_variants:
                raise ValueError(
                    f"Unknown dataset variant: {dataset_variant!r}. "
                    f"Available: {list(all_variants.keys())}"
                )
            training_df = all_variants[dataset_variant]
            features_df = build_variant_features(
                training_df,
                variant,
                save_artifacts=False,
            )
        else:
            features_path = PROCESSED_DATA_DIR / "features.csv"
            if features_path.exists() and variant.feature_builder_fn is None:
                logger.info(f"Loading features from {features_path}")
                features_df = pd.read_csv(features_path, parse_dates=["event_date"])
                features_df = apply_variant_feature_transforms(features_df, variant)
            else:
                logger.info("No cached features found. Building from Kaggle dataset...")
                from src.data.kaggle_loader import load_kaggle_dataset

                fights_df = load_kaggle_dataset()
                features_df = build_variant_features(
                    fights_df,
                    variant,
                    save_artifacts=False,
                )
    else:
        features_df = features_df.copy()
        if variant.feature_builder_fn is None:
            features_df = apply_variant_feature_transforms(features_df, variant)

    # Determine whether BetsAPI augmentation is needed
    needs_betsapi = False
    if feature_family is not None:
        needs_betsapi = "betsapi" in feature_family
    elif variant_name == "betsapi_challenger":
        needs_betsapi = True

    if needs_betsapi and not any(
        column.startswith("betsapi_") for column in features_df.columns
    ):
        from src.data.betsapi_mma import augment_features_with_betsapi_mma

        logger.info("Augmenting feature matrix with saved BetsAPI MMA challenger features")
        features_df = augment_features_with_betsapi_mma(features_df, save_artifacts=True)

    return features_df


def _generate_walk_forward_predictions(
    features_df: pd.DataFrame | None = None,
    retrain_months: int = 6,
    initial_train_years: int = 5,
    bet_start_date: str = TRAIN_CUTOFF_DATE,
    variant_name: str = "baseline",
    dataset_variant: str | None = None,
    feature_family: str | None = None,
    calibration_method: str | None = None,
) -> list[tuple[int, pd.DataFrame]]:
    """Train models via walk-forward and generate predictions for all folds.

    Parameters
    ----------
    features_df : pd.DataFrame | None
        Pre-loaded features.  When *None*, features are loaded/built
        according to *dataset_variant* and *feature_family*.
    retrain_months : int
        Walk-forward retrain interval.
    initial_train_years : int
        Minimum years of data before the first fold.
    bet_start_date : str
        Date string for the earliest bets.
    variant_name : str
        Model variant name.
    dataset_variant : str | None
        If provided, build from this named dataset variant.
    feature_family : str | None
        If provided, select feature columns for this family.
    calibration_method : str | None
        Override the variant calibration method for this walk-forward run.
    """
    features_df = _load_features_for_variant(
        features_df, variant_name,
        dataset_variant=dataset_variant,
        feature_family=feature_family,
    )

    variant = ALL_VARIANTS[variant_name]()
    if calibration_method is not None:
        variant.calibration_method = calibration_method
    fold_predictions = generate_variant_fold_predictions(
        features_df,
        variant,
        retrain_months=retrain_months,
        initial_train_years=initial_train_years,
        bet_start_date=bet_start_date,
        feature_family=feature_family,
    )

    logger.info(
        "Walk-forward complete: %s folds, %s total predictions",
        len(fold_predictions),
        sum(len(predictions) for _, predictions in fold_predictions),
    )
    return fold_predictions


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
    closing_prob_a = row.get("closing_prob_a")
    closing_prob_b = row.get("closing_prob_b")
    if isinstance(line_movement, float) and np.isnan(line_movement):
        line_movement = None
    if isinstance(closing_prob_a, float) and np.isnan(closing_prob_a):
        closing_prob_a = None
    if isinstance(closing_prob_b, float) and np.isnan(closing_prob_b):
        closing_prob_b = None

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
        "model_a": model_a,
        "model_b": model_b,
        "market_a": market_a,
        "market_b": market_b,
        "no_odds_a": no_odds_a,
        "no_odds_b": no_odds_b,
        "line_movement": line_movement,
        "line_is_sharp": line_is_sharp,
        "line_steam_move": line_steam_move,
        "closing_prob_a": closing_prob_a,
        "closing_prob_b": closing_prob_b,
        "a_fights": a_fights,
        "b_fights": b_fights,
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
    execution_mode: str = "legacy",
    execution_config: "BacktestExecutionConfig | None" = None,
) -> dict:
    """Evaluate a single S+C configuration against precomputed predictions.

    `execution_mode="realistic"` prices fills with the shared backtest
    execution model (half-spread quotes, synthetic order-book walk, liquidity
    and slippage caps) and defers settlement to event boundaries so winnings
    from one fight cannot compound into stakes on the same card. The legacy
    mode is byte-identical to the historical frictionless behavior.
    """
    execution_config = _resolve_execution_config(execution_mode, execution_config=execution_config)
    realistic = execution_config.mode == "realistic"

    bank = BankrollManager(
        initial_bankroll=initial_bankroll,
        kelly_fraction=KELLY_FRACTION,
        max_bet_fraction=MAX_BET_FRACTION,
        auto_detect_balance=False,
    )

    bet_log = []
    bankroll_history = [initial_bankroll]
    pnl_s = 0.0
    pnl_c = 0.0

    bet_start = pd.Timestamp(bet_start_date)

    pending_settlements: list[dict] = []
    current_event_key: str | None = None

    def _settle_pending() -> None:
        nonlocal pnl_s, pnl_c
        for pending in pending_settlements:
            bank.settle_bet(pending["bet_idx"], won=pending["won"])
            profit = bank.history[pending["bet_idx"]]["profit"]
            if pending["trader"] == "S":
                pnl_s += profit
            else:
                pnl_c += profit
            entry = pending["log_entry"]
            entry["profit"] = profit
            entry["bankroll_after"] = bank.bankroll
            bet_log.append(entry)
            bankroll_history.append(bank.bankroll)
        pending_settlements.clear()

    for fold_num, predictions in fold_predictions:
        for _, row in predictions.iterrows():
            fight_date = pd.Timestamp(row.get("event_date"))
            if fight_date < bet_start:
                continue

            if bank.is_stopped:
                break

            if realistic:
                event_key = fight_date.normalize().strftime("%Y-%m-%d")
                if current_event_key is not None and event_key != current_event_key:
                    _settle_pending()
                current_event_key = event_key

            fight = _parse_fight_row(row)

            if pd.isna(fight["market_a"]) or pd.isna(fight["market_b"]):
                continue

            single_bet = None
            blend_s_a, blend_s_b = compute_independent_blend_probs(
                fight["model_a"], fight["market_a"], fight["no_odds_a"],
                fight["model_b"], fight["market_b"], fight["no_odds_b"],
                base_weight=config.s_blend_weight,
            )
            edge_s_a = blend_s_a - fight["market_a"]
            edge_s_b = blend_s_b - fight["market_b"]

            if edge_s_a >= config.s_min_edge and edge_s_a >= edge_s_b and _passes_filters(
                blend_s_a,
                fight["market_a"],
                edge_s_a,
                fight["fighter_a"],
                fight["no_odds_a"],
                line_movement=fight["line_movement"],
                line_is_sharp=fight["line_is_sharp"],
                line_steam_move=fight["line_steam_move"],
                bet_side="a",
                a_num_fights=fight["a_fights"],
                b_num_fights=fight["b_fights"],
                edge_scaling_base=config.s_min_edge,
                model_agreement_min_edge=config.s_model_agreement_min_edge,
            ):
                sizing_prob = (
                    _estimated_best_ask(fight["market_a"], execution_config)
                    if realistic else fight["market_a"]
                )
                odds = implied_prob_to_decimal_odds(sizing_prob)
                bet_size = bank.kelly_bet_size(blend_s_a, odds)
                if bet_size > 0:
                    single_bet = {
                        "side": "a",
                        "blend": blend_s_a,
                        "market": fight["market_a"],
                        "edge": edge_s_a,
                        "size": bet_size,
                    }
            elif edge_s_b >= config.s_min_edge and _passes_filters(
                blend_s_b,
                fight["market_b"],
                edge_s_b,
                fight["fighter_b"],
                fight["no_odds_b"],
                line_movement=fight["line_movement"],
                line_is_sharp=fight["line_is_sharp"],
                line_steam_move=fight["line_steam_move"],
                bet_side="b",
                a_num_fights=fight["a_fights"],
                b_num_fights=fight["b_fights"],
                edge_scaling_base=config.s_min_edge,
                model_agreement_min_edge=config.s_model_agreement_min_edge,
            ):
                sizing_prob = (
                    _estimated_best_ask(fight["market_b"], execution_config)
                    if realistic else fight["market_b"]
                )
                odds = implied_prob_to_decimal_odds(sizing_prob)
                bet_size = bank.kelly_bet_size(blend_s_b, odds)
                if bet_size > 0:
                    single_bet = {
                        "side": "b",
                        "blend": blend_s_b,
                        "market": fight["market_b"],
                        "edge": edge_s_b,
                        "size": bet_size,
                    }

            if single_bet is not None:
                side = single_bet["side"]
                fighter = fight["fighter_a"] if side == "a" else fight["fighter_b"]
                stake = single_bet["size"]
                fill_price = float(single_bet["market"])
                if realistic:
                    fill = _simulate_realistic_fill(
                        row, float(single_bet["market"]), stake, execution_config,
                    )
                    stake = round(float(fill["filled_stake"]), 2)
                    if not fill["ok"] or stake <= 0:
                        continue
                    fill_price = float(fill["fill_price"])
                odds = implied_prob_to_decimal_odds(fill_price)
                bet_idx = len(bank.history)
                placed = bank.place_bet(stake, fighter, odds, single_bet["blend"], fill_price)
                if realistic and not placed:
                    continue
                won = fight["actual_winner_is_a"] if side == "a" else not fight["actual_winner_is_a"]
                clv_basis = fill_price if realistic else single_bet["market"]
                clv = np.nan
                if side == "a" and fight["closing_prob_a"] is not None:
                    clv = calculate_closing_line_value(clv_basis, fight["closing_prob_a"])
                elif side == "b" and fight["closing_prob_b"] is not None:
                    clv = calculate_closing_line_value(clv_basis, fight["closing_prob_b"])
                log_entry = {
                    "trader": "S",
                    "event_date": fight["event_date"],
                    "fighter_a": fight["fighter_a"],
                    "fighter_b": fight["fighter_b"],
                    "bet_on": fighter,
                    "side": side,
                    "bet_size": stake,
                    "odds": odds,
                    "decimal_odds": odds,
                    "fill_price": fill_price,
                    "model_prob": fight["model_a"] if side == "a" else fight["model_b"],
                    "blended_prob": single_bet["blend"],
                    "market_prob": single_bet["market"],
                    "edge": single_bet["edge"],
                    "edge_basis": "blend",
                    "blend_edge": single_bet["edge"],
                    "raw_edge": (
                        fight["model_a"] - fight["market_a"]
                        if side == "a"
                        else fight["model_b"] - fight["market_b"]
                    ),
                    "won": won,
                    "fold": fold_num,
                    "clv": clv,
                }
                if realistic:
                    pending_settlements.append(
                        {"trader": "S", "bet_idx": bet_idx, "won": won, "log_entry": log_entry}
                    )
                else:
                    bank.settle_bet(bet_idx, won=won)
                    profit = bank.history[-1]["profit"]
                    pnl_s += profit
                    log_entry["profit"] = profit
                    log_entry["bankroll_after"] = bank.bankroll
                    bet_log.append(log_entry)
                    bankroll_history.append(bank.bankroll)
                continue

            remaining = bank.bankroll
            alloc_c = remaining * config.c_share

            bet_c = None
            for side, model_prob, market_prob, no_odds_prob, fighter in [
                ("a", fight["model_a"], fight["market_a"], fight["no_odds_a"], fight["fighter_a"]),
                ("b", fight["model_b"], fight["market_b"], fight["no_odds_b"], fight["fighter_b"]),
            ]:
                edge = model_prob - market_prob
                decimal_odds = implied_prob_to_decimal_odds(market_prob)
                if model_prob < config.c_min_model_prob or no_odds_prob is None or no_odds_prob < config.c_min_no_odds_prob:
                    continue
                if edge < config.c_min_edge:
                    continue
                if decimal_odds > config.c_max_decimal_odds:
                    continue
                if fight["a_fights"] is not None and fight["a_fights"] < MIN_FIGHTER_FIGHTS:
                    continue
                if fight["b_fights"] is not None and fight["b_fights"] < MIN_FIGHTER_FIGHTS:
                    continue
                bet_c = {
                    "side": side,
                    "model_prob": model_prob,
                    "market": market_prob,
                    "fighter": fighter,
                    "edge": edge,
                    "decimal_odds": decimal_odds,
                }
                break

            if bet_c is not None:
                bet_size = _conviction_bet_size_param(bet_c["model_prob"], alloc_c, config)
                if bet_size > 0:
                    fill_price_c = float(bet_c["market"])
                    if realistic:
                        fill = _simulate_realistic_fill(
                            row, float(bet_c["market"]), bet_size, execution_config,
                        )
                        bet_size = round(float(fill["filled_stake"]), 2)
                        if not fill["ok"] or bet_size <= 0:
                            continue
                        fill_price_c = float(fill["fill_price"])
                    odds = implied_prob_to_decimal_odds(fill_price_c) if realistic else bet_c["decimal_odds"]
                    bet_idx = len(bank.history)
                    placed = bank.place_bet(bet_size, bet_c["fighter"], odds, bet_c["model_prob"], fill_price_c)
                    if realistic and not placed:
                        continue
                    won = fight["actual_winner_is_a"] if bet_c["side"] == "a" else not fight["actual_winner_is_a"]
                    clv_basis_c = fill_price_c if realistic else bet_c["market"]
                    clv = np.nan
                    if bet_c["side"] == "a" and fight["closing_prob_a"] is not None:
                        clv = calculate_closing_line_value(clv_basis_c, fight["closing_prob_a"])
                    elif bet_c["side"] == "b" and fight["closing_prob_b"] is not None:
                        clv = calculate_closing_line_value(clv_basis_c, fight["closing_prob_b"])
                    log_entry_c = {
                        "trader": "C",
                        "event_date": fight["event_date"],
                        "fighter_a": fight["fighter_a"],
                        "fighter_b": fight["fighter_b"],
                        "bet_on": bet_c["fighter"],
                        "side": bet_c["side"],
                        "bet_size": bet_size,
                        "odds": odds,
                        "decimal_odds": odds,
                        "fill_price": fill_price_c,
                        "model_prob": bet_c["model_prob"],
                        "blended_prob": np.nan,
                        "market_prob": bet_c["market"],
                        "edge": bet_c["edge"],
                        "edge_basis": "raw",
                        "blend_edge": np.nan,
                        "raw_edge": bet_c["edge"],
                        "won": won,
                        "fold": fold_num,
                        "clv": clv,
                    }
                    if realistic:
                        pending_settlements.append(
                            {"trader": "C", "bet_idx": bet_idx, "won": won, "log_entry": log_entry_c}
                        )
                    else:
                        bank.settle_bet(bet_idx, won=won)
                        profit = bank.history[-1]["profit"]
                        pnl_c += profit
                        log_entry_c["profit"] = profit
                        log_entry_c["bankroll_after"] = bank.bankroll
                        bet_log.append(log_entry_c)
                        bankroll_history.append(bank.bankroll)
                    continue

    _settle_pending()

    bet_log_df = pd.DataFrame(bet_log)
    bankroll_df = pd.DataFrame({"combined": bankroll_history})

    def _trader_stats(code: str) -> dict:
        if bet_log_df.empty:
            return {
                "total_bets": 0,
                "wins": 0,
                "win_rate": 0,
                "total_wagered": 0,
                "roi": 0,
                "avg_edge": 0,
                "avg_bet_size": 0,
                "avg_clv": np.nan,
            }
        trader_log = bet_log_df[bet_log_df["trader"] == code]
        if trader_log.empty:
            return {
                "total_bets": 0,
                "wins": 0,
                "win_rate": 0,
                "total_wagered": 0,
                "roi": 0,
                "avg_edge": 0,
                "avg_bet_size": 0,
                "avg_clv": np.nan,
            }
        bets = len(trader_log)
        wins = int(trader_log["won"].sum())
        wagered = trader_log["bet_size"].sum()
        profit = trader_log["profit"].sum()
        return {
            "total_bets": bets,
            "wins": wins,
            "win_rate": wins / bets if bets > 0 else 0,
            "total_wagered": wagered,
            "roi": profit / wagered if wagered > 0 else 0,
            "avg_edge": trader_log["edge"].mean() if bets > 0 else 0,
            "avg_bet_size": wagered / bets if bets > 0 else 0,
            "avg_clv": (
                trader_log["clv"].dropna().mean()
                if "clv" in trader_log.columns and trader_log["clv"].notna().any()
                else np.nan
            ),
        }

    stats_s = _trader_stats("S")
    stats_c = _trader_stats("C")

    final_bankroll = bank.bankroll
    total_profit = final_bankroll - initial_bankroll
    total_bets = stats_s["total_bets"] + stats_c["total_bets"]
    total_wagered = stats_s["total_wagered"] + stats_c["total_wagered"]
    total_wins = stats_s["wins"] + stats_c["wins"]
    combined_clv = np.nan
    if not bet_log_df.empty and "clv" in bet_log_df.columns and bet_log_df["clv"].notna().any():
        combined_clv = bet_log_df["clv"].dropna().mean()

    drawdown_stats = compute_max_drawdown(bankroll_history)
    max_dd = drawdown_stats["max_drawdown_pct"]
    max_dd_duration = drawdown_stats["max_drawdown_duration"]

    return {
        "config": config,
        "model_variant": config.variant_name,
        "mode": "duo_sweep",
        "trader_s": {"name": f"Single Trader (edge={config.s_min_edge})", "final_pnl": pnl_s, "stats": stats_s},
        "trader_c": {
            "name": f"Conviction (>={config.c_min_model_prob}/{config.c_min_no_odds_prob}, alloc={config.c_share})",
            "final_pnl": pnl_c,
            "stats": stats_c,
        },
        "trader_m": None,
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
            "max_drawdown_pct": max_dd,
            "max_drawdown_duration": max_dd_duration,
            "avg_clv": combined_clv,
        },
        "bet_log": bet_log_df,
        "bankroll_history": bankroll_df,
    }


def _production_sweep_config(
    *,
    variant_name: str,
    name: str,
    is_baseline: bool = False,
) -> SweepConfig:
    """Current live production controls as an explicit sweep row."""
    return SweepConfig(
        name=name,
        variant_name=variant_name,
        is_baseline=is_baseline,
        s_min_edge=MIN_EDGE_THRESHOLD,
        s_blend_weight=BLEND_WEIGHT,
        s_model_agreement_min_edge=MODEL_AGREEMENT_MIN_EDGE,
        c_min_model_prob=CONVICTION_MIN_MODEL_PROB,
        c_min_no_odds_prob=CONVICTION_MIN_NO_ODDS_PROB,
        c_share=TRADER_C_SHARE,
        c_min_edge=0.0,
        c_max_decimal_odds=float("inf"),
        m_enabled=False,
    )


def _build_sweep_configs(variant_name: str = "betsapi_challenger") -> list[SweepConfig]:
    """Build the challenger grid of parameter combinations to sweep."""
    return build_sweep_configs(variant_name=variant_name)


def build_sweep_configs(
    *,
    variant_name: str = "betsapi_challenger",
    s_edges: list[float] | None = None,
    s_blends: list[float] | None = None,
    s_agreements: list[float] | None = None,
    c_model_probs: list[float] | None = None,
    c_no_odds_probs: list[float] | None = None,
    c_shares: list[float] | None = None,
    c_min_edges: list[float] | None = None,
    c_max_decimal_odds_values: list[float] | None = None,
    c_bet_fractions: list[float] | None = None,
    include_production_controls: bool = True,
) -> list[SweepConfig]:
    """Build a configurable challenger grid for broad or narrow follow-up sweeps."""
    s_edges = s_edges or [0.02, 0.025, 0.03]
    s_blends = s_blends or [0.25, 0.30, 0.35]
    s_agreements = s_agreements or [0.01, 0.02]
    c_model_probs = c_model_probs or [0.65, 0.70, 0.75]
    c_no_odds_probs = c_no_odds_probs or [0.50, 0.55, 0.60]
    c_shares = c_shares or [0.20, 0.50, 1.00]
    c_min_edges = c_min_edges or [0.00, 0.01]
    c_max_decimal_odds_values = c_max_decimal_odds_values or [2.0, 2.5, 3.0]
    # Sweep the conviction stake only when explicitly requested so existing
    # grids (and their config names) stay byte-identical.
    explicit_c_bet_fractions = c_bet_fractions is not None
    c_bet_fractions = c_bet_fractions or [SweepConfig.c_bet_fraction]

    configs: list[SweepConfig] = []
    if include_production_controls:
        configs.append(
            _production_sweep_config(
                variant_name=variant_name,
                name="challenger_production_controls",
                is_baseline=False,
            )
        )

    for s_edge in s_edges:
        for s_blend in s_blends:
            for s_agreement in s_agreements:
                for c_model in c_model_probs:
                    for c_no_odds in c_no_odds_probs:
                        for c_share in c_shares:
                            for c_min_edge in c_min_edges:
                                for c_max_decimal_odds in c_max_decimal_odds_values:
                                    for c_bet_fraction in c_bet_fractions:
                                        name = (
                                            f"{variant_name}_s={s_edge:.3f}"
                                            f"_blend={s_blend:.2f}"
                                            f"_agree={s_agreement:.2f}"
                                            f"_c={c_model:.2f}/{c_no_odds:.2f}"
                                            f"_share={c_share:.2f}"
                                            f"_cedge={c_min_edge:.2f}"
                                            f"_codds={c_max_decimal_odds:.1f}"
                                        )
                                        if explicit_c_bet_fractions:
                                            name += f"_cbet={c_bet_fraction:.2f}"
                                        configs.append(
                                            SweepConfig(
                                                name=name,
                                                variant_name=variant_name,
                                                s_min_edge=s_edge,
                                                s_blend_weight=s_blend,
                                                s_model_agreement_min_edge=s_agreement,
                                                c_min_model_prob=c_model,
                                                c_min_no_odds_prob=c_no_odds,
                                                c_share=c_share,
                                                c_min_edge=c_min_edge,
                                                c_max_decimal_odds=c_max_decimal_odds,
                                                c_bet_fraction=c_bet_fraction,
                                                m_enabled=False,
                                            )
                                        )
    return configs


def _summary_row_from_result(config: SweepConfig, result: dict) -> dict:
    """Flatten one sweep result row for reporting."""
    combined = result["combined"]
    s_stats = result["trader_s"]["stats"]
    c_stats = result["trader_c"]["stats"]
    m_stats = result["trader_m"]["stats"] if result["trader_m"] else {"total_bets": 0}
    s_edge_profit_gap = max(0.0, s_stats["roi"] - s_stats["avg_edge"])
    return {
        "config": config.name,
        "model_variant": config.variant_name,
        "is_baseline": config.is_baseline,
        "s_min_edge": config.s_min_edge,
        "s_blend_weight": config.s_blend_weight,
        "s_model_agreement_min_edge": config.s_model_agreement_min_edge,
        "c_min_model_prob": config.c_min_model_prob,
        "c_min_no_odds_prob": config.c_min_no_odds_prob,
        "c_share": config.c_share,
        "c_min_edge": config.c_min_edge,
        "c_max_decimal_odds": config.c_max_decimal_odds,
        "s_bets": s_stats["total_bets"],
        "s_wins": s_stats["wins"],
        "s_win_rate": s_stats["win_rate"],
        "s_wagered": s_stats["total_wagered"],
        "s_roi": s_stats["roi"],
        "s_avg_edge": s_stats["avg_edge"],
        "s_edge_profit_gap": s_edge_profit_gap,
        "s_edge_profit_flag": s_edge_profit_gap > 0.05,
        "s_avg_bet_size": s_stats["avg_bet_size"],
        "s_avg_clv": s_stats["avg_clv"],
        "c_bets": c_stats["total_bets"],
        "c_wins": c_stats["wins"],
        "c_win_rate": c_stats["win_rate"],
        "c_wagered": c_stats["total_wagered"],
        "c_roi": c_stats["roi"],
        "c_avg_edge": c_stats["avg_edge"],
        "c_avg_bet_size": c_stats["avg_bet_size"],
        "c_avg_clv": c_stats["avg_clv"],
        "m_bets": m_stats["total_bets"],
        "total_bets": combined["total_bets"],
        "wins": combined["wins"],
        "win_rate": combined["win_rate"],
        "total_profit": combined["total_profit"],
        "profit": combined["total_profit"],
        "roi": combined["roi"],
        "avg_clv": combined.get("avg_clv", np.nan),
        "max_drawdown": combined["max_drawdown"],
        "max_drawdown_pct": combined.get("max_drawdown_pct", combined["max_drawdown"]),
        "max_drawdown_duration": combined.get("max_drawdown_duration", 0),
        "max_dd": combined.get("max_drawdown_pct", combined["max_drawdown"]),
        "final_bankroll": combined["final_bankroll"],
        "total_wagered": combined["total_wagered"],
    }


def _sort_summary_rows(summary_rows: list[dict]) -> None:
    """Sort sweep rows in place by the current decision priorities."""
    summary_rows.sort(
        key=lambda row: (
            bool(row.get("s_edge_profit_flag", False)),
            row.get("s_edge_profit_gap", 0.0),
            -row["roi"],
            -(row["avg_clv"] if pd.notna(row["avg_clv"]) else -999.0),
            row["max_dd"],
            row["total_bets"],
        )
    )


def _print_sweep_results(summary: list[dict]) -> None:
    """Print a formatted comparison table of sweep results."""
    print(f"\n{'='*150}")
    print(f"PARAMETER SWEEP RESULTS ({len(summary)} rows)")
    print(f"{'='*150}")
    print(
        f"{'Rank':>4}  {'Model':<18} {'Baseline':<8} {'Config':<58} {'S':>4} {'C':>4} {'Total':>5}  "
        f"{'WR':>6}  {'Profit':>10}  {'ROI':>7}  {'CLV':>7}  {'MaxDD':>6}"
    )
    print(
        f"{'----':>4}  {'-----':<18} {'--------':<8} {'------':<58} {'--':>4} {'--':>4} {'-----':>5}  "
        f"{'--':>6}  {'------':>10}  {'---':>7}  {'---':>7}  {'-----':>6}"
    )

    for i, row in enumerate(summary, 1):
        rank = "BASE" if row.get("is_baseline") else str(i)
        avg_clv = row.get("avg_clv")
        clv_str = f"{avg_clv:+.1%}" if pd.notna(avg_clv) else "n/a"
        print(
            f"{rank:>4}  {row['model_variant']:<18} {str(row.get('is_baseline', False)):<8} "
            f"{row['config']:<58} {row['s_bets']:>4} {row['c_bets']:>4} {row['total_bets']:>5}  "
            f"{row['win_rate']:>5.1%}  "
            f"${row['profit']:>+9.2f}  "
            f"{row['roi']:>+6.1%}  "
            f"{clv_str:>7}  "
            f"{row['max_dd']:>5.1%}"
        )

    print(f"{'='*150}\n")


def run_sweep_backtest(
    initial_bankroll: float = INITIAL_BANKROLL,
    bet_start_date: str = TRAIN_CUTOFF_DATE,
    variant_name: str = "betsapi_challenger",
) -> tuple[list[dict], dict[str, dict]]:
    """Run the dedicated S+C sweep module against one model variant."""
    logger.info("Starting duo-trader sweep...")
    logger.info("Phase 1: Generating walk-forward predictions (train once)...")

    fold_predictions = _generate_walk_forward_predictions(
        bet_start_date=bet_start_date,
        variant_name=variant_name,
    )

    configs = _build_sweep_configs(variant_name=variant_name)
    logger.info(f"Phase 2: Evaluating {len(configs)} configurations...")

    summary_rows = []
    all_results = {}
    for i, config in enumerate(configs, 1):
        result = _evaluate_config(
            fold_predictions,
            config,
            initial_bankroll=initial_bankroll,
            bet_start_date=bet_start_date,
        )
        all_results[config.name] = result
        summary_rows.append(_summary_row_from_result(config, result))

        if i % 10 == 0:
            logger.info(f"  Evaluated {i}/{len(configs)} configs...")

    _sort_summary_rows(summary_rows)
    _print_sweep_results(summary_rows)

    top_names = [row["config"] for row in summary_rows[:5]]
    top_results = {name: all_results[name] for name in top_names}
    return summary_rows, top_results


def run_betsapi_challenger_sweep_comparison_with_configs(
    configs: list[SweepConfig],
    *,
    initial_bankroll: float = INITIAL_BANKROLL,
    bet_start_date: str = TRAIN_CUTOFF_DATE,
    include_production_baseline: bool = True,
) -> tuple[list[dict], dict[str, dict]]:
    """Compare baseline against an arbitrary challenger config list using one prediction pass per variant."""
    logger.info("Starting baseline vs BetsAPI duo-trader challenger sweep comparison...")

    eval_configs = list(configs)
    if include_production_baseline and not any(config.is_baseline for config in eval_configs):
        eval_configs = [
            _production_sweep_config(
                variant_name="baseline",
                name="production_baseline",
                is_baseline=True,
            )
        ] + eval_configs

    prediction_cache: dict[str, list[tuple[int, pd.DataFrame]]] = {}
    variant_names = {config.variant_name for config in eval_configs}
    for variant_name in sorted(variant_names):
        prediction_cache[variant_name] = _generate_walk_forward_predictions(
            bet_start_date=bet_start_date,
            variant_name=variant_name,
        )

    summary_rows = []
    all_results = {}
    for config in eval_configs:
        result = _evaluate_config(
            prediction_cache[config.variant_name],
            config,
            initial_bankroll=initial_bankroll,
            bet_start_date=bet_start_date,
        )
        all_results[config.name] = result
        summary_rows.append(_summary_row_from_result(config, result))

    baseline_rows = [row for row in summary_rows if row["is_baseline"]]
    challenger_rows = [row for row in summary_rows if not row["is_baseline"]]
    _sort_summary_rows(challenger_rows)
    ordered_rows = baseline_rows + challenger_rows if baseline_rows else challenger_rows

    _print_sweep_results(ordered_rows)

    top_limit = 6 if baseline_rows else 5
    top_names = [row["config"] for row in ordered_rows[:top_limit]]
    top_results = {name: all_results[name] for name in top_names}
    return ordered_rows, top_results


def run_betsapi_challenger_sweep_comparison(
    initial_bankroll: float = INITIAL_BANKROLL,
    bet_start_date: str = TRAIN_CUTOFF_DATE,
) -> tuple[list[dict], dict[str, dict]]:
    """Compare the live production baseline against the BetsAPI S+C challenger sweep."""
    return run_betsapi_challenger_sweep_comparison_with_configs(
        build_sweep_configs(variant_name="betsapi_challenger"),
        initial_bankroll=initial_bankroll,
        bet_start_date=bet_start_date,
        include_production_baseline=True,
    )
