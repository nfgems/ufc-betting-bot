"""
Backtesting engine for static, walk-forward, and strategy-comparison runs.

Supports:
  1. Production-gated backtests with explicit agreement behavior
  2. Clean signal-only comparisons between odds-aware and no-odds models
  3. Walk-forward model-vs-strategy comparison with artifact output
"""

import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.config import (
    BLEND_WEIGHT,
    INITIAL_BANKROLL,
    KELLY_FRACTION,
    LOGS_DIR,
    MAX_BET_FRACTION,
    MIN_EDGE_THRESHOLD,
    TRAIN_CUTOFF_DATE,
)
from src.model.predict import predict_batch
from src.model.train import load_model
from src.strategy.bankroll import BankrollManager
from src.strategy.lab_stats import compute_ece, compute_max_drawdown
from src.strategy.value import (
    DEFAULT_NEWBIE_RULE,
    NewbieRuleConfig,
    _passes_filters,
    blend_probability,
    calculate_closing_line_value,
    compute_independent_blend_probs,
    dynamic_blend_weight,
    implied_prob_to_decimal_odds,
    newbie_penalty,
)

logger = logging.getLogger(__name__)

COMPARISON_DIR = LOGS_DIR / "comparison"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BacktestStrategyConfig:
    """Configuration for a backtested betting strategy."""

    name: str
    primary_model: str
    selection_mode: str = "blended"  # "blended" or "raw"
    require_agreement: bool = False
    agreement_model: Optional[str] = None
    use_agreement_for_blend: bool = False
    blend_weight: float = BLEND_WEIGHT
    max_decimal_odds: Optional[float] = None
    newbie_rule: NewbieRuleConfig = DEFAULT_NEWBIE_RULE

    def required_models(self) -> tuple[str, ...]:
        models = [self.primary_model]
        if self.agreement_model and (self.require_agreement or self.use_agreement_for_blend):
            if self.agreement_model not in models:
                models.append(self.agreement_model)
        return tuple(models)


ODDS_AWARE_SIGNAL_ONLY_STRATEGY = BacktestStrategyConfig(
    name="odds_aware_signal_only",
    primary_model="xgboost",
    selection_mode="raw",
)

NO_ODDS_SIGNAL_ONLY_STRATEGY = BacktestStrategyConfig(
    name="no_odds_signal_only",
    primary_model="xgboost_no_odds",
    selection_mode="raw",
)

PRODUCTION_GATED_STRATEGY = BacktestStrategyConfig(
    name="production_gated",
    primary_model="xgboost",
    selection_mode="blended",
    require_agreement=True,
    agreement_model="xgboost_no_odds",
    use_agreement_for_blend=True,
    blend_weight=BLEND_WEIGHT,
)

COMPARISON_STRATEGIES = (
    ODDS_AWARE_SIGNAL_ONLY_STRATEGY,
    NO_ODDS_SIGNAL_ONLY_STRATEGY,
    PRODUCTION_GATED_STRATEGY,
)


def _resolve_market_odds(
    predictions: pd.DataFrame,
    market_prob_col_a: str,
    market_prob_col_b: str,
    *,
    allow_closing_odds: bool = False,
) -> tuple[pd.DataFrame, str]:
    """
    Resolve market odds columns for backtesting — row-by-row fallback.

    Per-row priority:
      1. opening_prob_a/b from historical API backfill
      2. requested market columns (e.g. a_fair_prob_avg)
      3. Kaggle closing odds with vig removed
    """
    has_opening = "opening_prob_a" in predictions.columns
    has_requested = market_prob_col_a in predictions.columns
    has_closing = "a_implied_prob" in predictions.columns

    if not has_opening and not has_requested and not has_closing:
        raise ValueError(
            "No market odds found. Need historical backfill data, "
            "requested market columns, or Kaggle implied probabilities."
        )

    # Start with NaN, then layer sources from lowest to highest priority.
    predictions["a_market_prob"] = np.nan
    predictions["b_market_prob"] = np.nan

    # Layer 3 (lowest): Kaggle closing odds
    if has_closing:
        if not has_opening and not has_requested and not allow_closing_odds:
            raise ValueError(
                "Only Kaggle CLOSING odds available (look-ahead bias). "
                "Run 'backfill-odds' for historical opening odds, or pass "
                "allow_closing_odds=True to proceed anyway."
            )
        a_imp = predictions["a_implied_prob"]
        b_imp = predictions["b_implied_prob"]
        total = a_imp + b_imp
        mask = total.notna() & (total > 0)
        predictions.loc[mask, "a_market_prob"] = (a_imp / total)[mask]
        predictions.loc[mask, "b_market_prob"] = (b_imp / total)[mask]

    # Layer 2: requested market columns
    if has_requested:
        mask = predictions[market_prob_col_a].notna()
        predictions.loc[mask, "a_market_prob"] = predictions.loc[mask, market_prob_col_a]
        predictions.loc[mask, "b_market_prob"] = predictions.loc[mask, market_prob_col_b]

    # Layer 1 (highest): historical opening odds
    if has_opening:
        mask = predictions["opening_prob_a"].notna()
        predictions.loc[mask, "a_market_prob"] = predictions.loc[mask, "opening_prob_a"]
        predictions.loc[mask, "b_market_prob"] = predictions.loc[mask, "opening_prob_b"]

    # Report coverage
    filled = predictions["a_market_prob"].notna().sum()
    total_rows = len(predictions)
    sources_used = []
    if has_opening:
        sources_used.append("historical_opening")
    if has_requested:
        sources_used.append(f"column:{market_prob_col_a}")
    if has_closing:
        sources_used.append("kaggle_closing")
    source_label = "+".join(sources_used) if sources_used else "none"
    still_missing = total_rows - filled
    if still_missing:
        logger.warning(
            f"Market odds resolved for {filled}/{total_rows} rows "
            f"({still_missing} still missing) using {source_label}"
        )
    else:
        logger.info(f"Market odds resolved for all {total_rows} rows using {source_label}")
    return predictions, source_label


def _merge_historical_odds(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Merge historical opening/closing odds and line-movement metadata."""
    from src.data.historical_backfill import (
        compute_line_movement_from_backfill,
        load_all_historical_odds,
    )

    hist = load_all_historical_odds()
    if hist.empty:
        return predictions

    logger.info(f"Found {len(hist)} historical odds records for backtest enrichment")

    opening = hist.loc[
        hist.groupby(["event_date", "fighter_a", "fighter_b"])["offset_days"].idxmax()
    ]
    opening = opening.rename(columns={
        "a_fair_prob": "opening_prob_a",
        "b_fair_prob": "opening_prob_b",
        "a_decimal_odds": "opening_odds_a",
        "b_decimal_odds": "opening_odds_b",
    })[[
        "event_date",
        "fighter_a",
        "fighter_b",
        "opening_prob_a",
        "opening_prob_b",
        "opening_odds_a",
        "opening_odds_b",
    ]]

    closing = hist.loc[
        hist.groupby(["event_date", "fighter_a", "fighter_b"])["offset_days"].idxmin()
    ]
    closing = closing.rename(columns={
        "a_fair_prob": "closing_prob_a",
        "b_fair_prob": "closing_prob_b",
        "a_decimal_odds": "closing_odds_a",
        "b_decimal_odds": "closing_odds_b",
    })[[
        "event_date",
        "fighter_a",
        "fighter_b",
        "closing_prob_a",
        "closing_prob_b",
        "closing_odds_a",
        "closing_odds_b",
    ]]

    movement = compute_line_movement_from_backfill(hist)

    predictions["event_date_str"] = pd.to_datetime(predictions["event_date"]).dt.strftime("%Y-%m-%d")
    opening["event_date_str"] = opening["event_date"].astype(str)
    closing["event_date_str"] = closing["event_date"].astype(str)

    merged = predictions.merge(
        opening.drop(columns=["event_date"]),
        on=["event_date_str", "fighter_a", "fighter_b"],
        how="left",
    )
    merged = merged.merge(
        closing.drop(columns=["event_date"]),
        on=["event_date_str", "fighter_a", "fighter_b"],
        how="left",
    )

    if not movement.empty:
        movement["event_date_str"] = movement["event_date"].astype(str)
        move_cols = [
            "event_date_str",
            "fighter_a",
            "fighter_b",
            "line_movement",
            "line_abs_movement",
            "line_is_sharp",
            "line_steam_move",
            "line_direction_toward_a",
            "line_direction_toward_b",
        ]
        move_cols = [c for c in move_cols if c in movement.columns]
        move_value_cols = [
            column for column in move_cols
            if column not in {"event_date_str", "fighter_a", "fighter_b"}
        ]
        renamed_move_cols = {
            column: f"{column}__hist"
            for column in move_value_cols
        }
        merged = merged.merge(
            movement[move_cols].rename(columns=renamed_move_cols),
            on=["event_date_str", "fighter_a", "fighter_b"],
            how="left",
        )
        for column, hist_column in renamed_move_cols.items():
            if column in merged.columns:
                merged[column] = merged[column].where(merged[column].notna(), merged[hist_column])
            else:
                merged[column] = merged[hist_column]
        if renamed_move_cols:
            merged = merged.drop(columns=list(renamed_move_cols.values()))

    merged = merged.drop(columns=["event_date_str"])
    matched = merged["opening_prob_a"].notna().sum()
    logger.info(f"Matched {matched}/{len(predictions)} fights with historical opening odds")
    return merged


def _prediction_col(model_name: str, suffix: str) -> str:
    return f"{model_name}_{suffix}"


def _append_model_predictions(
    predictions: pd.DataFrame,
    model_name: str,
    model_result: dict,
) -> pd.DataFrame:
    """Append prefixed probability columns for a model."""
    model_predictions = predict_batch(predictions, model_name=model_name, model_result=model_result)
    predictions[_prediction_col(model_name, "prob_a")] = model_predictions["prob_a"]
    predictions[_prediction_col(model_name, "prob_b")] = model_predictions["prob_b"]
    predictions[_prediction_col(model_name, "confidence")] = model_predictions["confidence"]
    predictions[_prediction_col(model_name, "predicted_winner")] = model_predictions["predicted_winner"]
    return predictions


def _default_strategy_for_model(
    model_name: str,
    blend_weight: float = BLEND_WEIGHT,
) -> BacktestStrategyConfig:
    """Map legacy run_backtest(model_name=...) calls onto explicit strategy configs."""
    if model_name == "xgboost_no_odds":
        return replace(
            NO_ODDS_SIGNAL_ONLY_STRATEGY,
            name=model_name,
            primary_model=model_name,
        )

    if model_name == "xgboost":
        return replace(
            PRODUCTION_GATED_STRATEGY,
            blend_weight=blend_weight,
        )

    return BacktestStrategyConfig(
        name=model_name,
        primary_model=model_name,
        selection_mode="blended",
        require_agreement=True,
        agreement_model="xgboost_no_odds",
        use_agreement_for_blend=True,
        blend_weight=blend_weight,
    )


def _resolve_strategy_config(
    strategy_config: Optional[BacktestStrategyConfig],
    model_name: str,
    blend_weight: float,
) -> BacktestStrategyConfig:
    if strategy_config is None:
        return _default_strategy_for_model(model_name, blend_weight=blend_weight)

    if strategy_config.selection_mode not in {"blended", "raw"}:
        raise ValueError(f"Unsupported selection_mode: {strategy_config.selection_mode}")

    if strategy_config.selection_mode == "raw":
        return replace(strategy_config, blend_weight=1.0, use_agreement_for_blend=False)

    return replace(strategy_config, blend_weight=blend_weight)


def _load_strategy_models(
    strategy_config: BacktestStrategyConfig,
    model_result: Optional[dict] = None,
    agreement_model_result: Optional[dict] = None,
) -> dict[str, dict]:
    """Load only the models required by a strategy."""
    model_results: dict[str, dict] = {}
    for model_name in strategy_config.required_models():
        if model_name == strategy_config.primary_model and model_result is not None:
            model_results[model_name] = model_result
            continue
        if strategy_config.agreement_model == model_name and agreement_model_result is not None:
            model_results[model_name] = agreement_model_result
            continue
        model_results[model_name] = load_model(model_name)
    return model_results


def _prepare_prediction_frame(
    test_df: pd.DataFrame,
    model_results: dict[str, dict],
    market_prob_col_a: str = "a_fair_prob_avg",
    market_prob_col_b: str = "b_fair_prob_avg",
    use_historical_odds: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Prepare a prediction frame with prefixed model outputs and market odds."""
    predictions = test_df.copy()

    # Merge historical odds BEFORE model predictions so opening odds can
    # replace closing odds in the model's implied_prob feature columns.
    # This prevents look-ahead bias: the model sees pre-fight opening odds
    # (matching live conditions) instead of post-fight closing odds.
    if use_historical_odds:
        predictions = _merge_historical_odds(predictions)
        if "opening_prob_a" in predictions.columns:
            mask = predictions["opening_prob_a"].notna()
            n_swapped = mask.sum()
            if n_swapped:
                predictions.loc[mask, "a_implied_prob"] = predictions.loc[mask, "opening_prob_a"]
                predictions.loc[mask, "b_implied_prob"] = predictions.loc[mask, "opening_prob_b"]
                predictions.loc[mask, "diff_implied_prob"] = (
                    predictions.loc[mask, "a_implied_prob"]
                    - predictions.loc[mask, "b_implied_prob"]
                )
                logger.info(
                    f"Replaced closing odds with opening odds in model features "
                    f"for {n_swapped}/{len(predictions)} fights"
                )

    for model_name, model_result in model_results.items():
        predictions = _append_model_predictions(predictions, model_name, model_result)

    predictions, odds_source = _resolve_market_odds(
        predictions,
        market_prob_col_a,
        market_prob_col_b,
    )
    predictions["odds_source"] = odds_source
    predictions = predictions.sort_values("event_date").reset_index(drop=True)
    return predictions, odds_source


def _clean_optional_float(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return float(value)


def _clean_optional_int(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def _model_probs_for_row(
    row: pd.Series,
    model_name: Optional[str],
) -> tuple[Optional[float], Optional[float]]:
    if not model_name:
        return None, None
    prob_a = _clean_optional_float(row.get(_prediction_col(model_name, "prob_a")))
    prob_b = _clean_optional_float(row.get(_prediction_col(model_name, "prob_b")))
    return prob_a, prob_b


def _selection_state_for_row(
    row: pd.Series,
    strategy_config: BacktestStrategyConfig,
) -> dict:
    """Resolve primary, agreement, and selected probabilities for a row."""
    primary_a, primary_b = _model_probs_for_row(row, strategy_config.primary_model)
    if primary_a is None or primary_b is None:
        raise ValueError(f"Missing primary model predictions for {strategy_config.primary_model}")

    agreement_a, agreement_b = _model_probs_for_row(row, strategy_config.agreement_model)
    market_a = row["a_market_prob"]

    if strategy_config.selection_mode == "raw":
        selected_a = primary_a
        selected_b = primary_b
        blend_weight_used = 1.0
    else:
        agreement_for_blend_a = agreement_a if strategy_config.use_agreement_for_blend else None
        agreement_for_blend_b = agreement_b if strategy_config.use_agreement_for_blend else None
        market_b = row["b_market_prob"]
        selected_a, selected_b = compute_independent_blend_probs(
            primary_a, market_a, agreement_for_blend_a,
            primary_b, market_b, agreement_for_blend_b,
            base_weight=strategy_config.blend_weight,
        )
        blend_weight_used = dynamic_blend_weight(
            primary_a,
            market_a,
            agreement_for_blend_a,
            strategy_config.blend_weight,
        )

    return {
        "primary_a": primary_a,
        "primary_b": primary_b,
        "agreement_a": agreement_a,
        "agreement_b": agreement_b,
        "selected_a": selected_a,
        "selected_b": selected_b,
        "blend_weight_used": blend_weight_used,
    }


def _compute_clv_stats(bet_log_df: pd.DataFrame) -> dict:
    if bet_log_df.empty or "clv" not in bet_log_df.columns:
        return {}

    valid_clv = bet_log_df["clv"].dropna()
    if valid_clv.empty:
        return {}

    return {
        "avg_clv": valid_clv.mean(),
        "median_clv": valid_clv.median(),
        "pct_positive_clv": (valid_clv > 0).mean(),
        "clv_sample_size": len(valid_clv),
    }


def _log_backtest_summary(
    title: str,
    strategy_config: BacktestStrategyConfig,
    result: dict,
) -> None:
    stats = result["stats"]
    predictions = result.get("predictions", pd.DataFrame())
    odds_source = result.get("odds_source", "unknown")

    logger.info(f"\n{'=' * 60}")
    logger.info(title)
    logger.info(f"{'=' * 60}")
    logger.info(f"Strategy: {strategy_config.name}")
    logger.info(f"Primary model: {strategy_config.primary_model}")
    logger.info(f"Selection mode: {strategy_config.selection_mode}")
    logger.info(f"Require agreement: {strategy_config.require_agreement}")
    logger.info(f"Odds source: {odds_source}")
    if not predictions.empty and "event_date" in predictions.columns:
        logger.info(
            f"Period: {predictions['event_date'].min()} to {predictions['event_date'].max()}"
        )
        logger.info(f"Total fights analyzed: {len(predictions)}")
    logger.info(f"Bets placed: {stats.get('total_bets', 0)}")
    logger.info(f"Win rate: {stats.get('win_rate', 0):.1%}")
    logger.info(f"Total wagered: ${stats.get('total_wagered', 0):.2f}")
    logger.info(f"Total profit: ${stats.get('total_profit', 0):+.2f}")
    logger.info(f"ROI: {stats.get('roi', 0):+.1%}")
    logger.info(f"Ending bankroll: ${stats.get('bankroll', 0):.2f}")
    logger.info(f"Avg edge on bets: {stats.get('avg_edge', 0):.1%}")
    if "avg_clv" in stats:
        logger.info(f"Avg CLV: {stats['avg_clv']:+.2%}")
    if "max_drawdown_pct" in stats:
        logger.info(f"Max drawdown: {stats['max_drawdown_pct']:.1%}")
    logger.info(f"{'=' * 60}")


def _simulate_backtest_predictions(
    predictions: pd.DataFrame,
    strategy_config: BacktestStrategyConfig,
    initial_bankroll: float = INITIAL_BANKROLL,
    min_edge: float = MIN_EDGE_THRESHOLD,
    kelly_fraction: float = KELLY_FRACTION,
    max_bet_fraction: float = MAX_BET_FRACTION,
    bet_start_date: Optional[str] = None,
) -> dict:
    """Run the betting simulation over a prepared prediction frame."""
    bankroll = BankrollManager(
        initial_bankroll=initial_bankroll,
        kelly_fraction=kelly_fraction,
        max_bet_fraction=max_bet_fraction,
        auto_detect_balance=False,
    )
    bankroll_history = [initial_bankroll]
    bet_log: list[dict] = []
    bet_start = pd.Timestamp(bet_start_date) if bet_start_date else None

    for _, row in predictions.iterrows():
        event_date = pd.Timestamp(row.get("event_date"))
        if bet_start is not None and event_date < bet_start:
            continue

        if bankroll.is_stopped:
            logger.warning(f"Stop-loss triggered for strategy '{strategy_config.name}'.")
            break

        market_a = row["a_market_prob"]
        market_b = row["b_market_prob"]
        if pd.isna(market_a) or pd.isna(market_b):
            continue

        state = _selection_state_for_row(row, strategy_config)
        actual_winner_is_a = row["target"] == 1
        edge_a = state["selected_a"] - market_a
        edge_b = state["selected_b"] - market_b

        line_movement = _clean_optional_float(row.get("line_movement"))
        line_is_sharp = _clean_optional_int(row.get("line_is_sharp"))
        line_steam_move = _clean_optional_int(row.get("line_steam_move"))
        a_fights = _clean_optional_int(row.get("a_num_fights"))
        b_fights = _clean_optional_int(row.get("b_num_fights"))
        a_org_tier = _clean_optional_float(row.get("a_pre_ufc_org_tier_best"))
        b_org_tier = _clean_optional_float(row.get("b_pre_ufc_org_tier_best"))
        newbie_adjustment = newbie_penalty(
            a_fights,
            b_fights,
            org_tier_a=a_org_tier,
            org_tier_b=b_org_tier,
            newbie_rule=strategy_config.newbie_rule,
        )

        placed_bet = False

        if edge_a >= min_edge and edge_a >= edge_b and _passes_filters(
            state["selected_a"],
            market_a,
            edge_a,
            row.get("fighter_a", "A"),
            state["agreement_a"],
            line_movement=line_movement,
            line_is_sharp=line_is_sharp,
            line_steam_move=line_steam_move,
            bet_side="a",
            a_num_fights=a_fights,
            b_num_fights=b_fights,
            a_org_tier=a_org_tier,
            b_org_tier=b_org_tier,
            newbie_rule=strategy_config.newbie_rule,
            require_model_agreement=strategy_config.require_agreement,
            max_decimal_odds=strategy_config.max_decimal_odds,
        ):
            odds = implied_prob_to_decimal_odds(market_a)
            bet_size = bankroll.kelly_bet_size(state["selected_a"], odds)
            bet_size = round(bet_size * newbie_adjustment.size_multiplier, 2) if bet_size > 0 else bet_size
            if bet_size > 0:
                bet_idx = len(bankroll.history)
                bankroll.place_bet(
                    bet_size,
                    row.get("fighter_a", "A"),
                    odds,
                    state["selected_a"],
                    market_a,
                )
                bankroll.settle_bet(bet_idx, won=actual_winner_is_a)
                placed_bet = True
                clv = np.nan
                if pd.notna(row.get("closing_prob_a")):
                    clv = calculate_closing_line_value(market_a, row["closing_prob_a"])

                bet_log.append({
                    "strategy": strategy_config.name,
                    "event_date": event_date,
                    "fighter_a": row.get("fighter_a", ""),
                    "fighter_b": row.get("fighter_b", ""),
                    "bet_on": row.get("fighter_a", "A"),
                    "bet_side": "a",
                    "bet_size": bet_size,
                    "odds": odds,
                    "model_prob": state["primary_a"],
                    "blended_prob": state["selected_a"],
                    "blend_weight_used": state["blend_weight_used"],
                    "agreement_prob": state["agreement_a"],
                    "market_prob": market_a,
                    "edge": edge_a,
                    "won": actual_winner_is_a,
                    "profit": bankroll.history[-1]["profit"],
                    "bankroll_after": bankroll.bankroll,
                    "clv": clv,
                    "closing_prob": row.get("closing_prob_a", np.nan),
                    "fold": row.get("fold", np.nan),
                    "train_end": row.get("train_end"),
                    "test_end": row.get("test_end"),
                    "odds_source": row.get("odds_source", "unknown"),
                    "is_newbie_bet": newbie_adjustment.is_newbie_bet,
                    "size_multiplier": newbie_adjustment.size_multiplier,
                    "newbie_extra_edge_required": newbie_adjustment.extra_edge_required,
                    "newbie_rule": strategy_config.newbie_rule.name,
                    "newbie_reason": newbie_adjustment.reason,
                })

        elif edge_b >= min_edge and _passes_filters(
            state["selected_b"],
            market_b,
            edge_b,
            row.get("fighter_b", "B"),
            state["agreement_b"],
            line_movement=line_movement,
            line_is_sharp=line_is_sharp,
            line_steam_move=line_steam_move,
            bet_side="b",
            a_num_fights=a_fights,
            b_num_fights=b_fights,
            a_org_tier=a_org_tier,
            b_org_tier=b_org_tier,
            newbie_rule=strategy_config.newbie_rule,
            require_model_agreement=strategy_config.require_agreement,
            max_decimal_odds=strategy_config.max_decimal_odds,
        ):
            odds = implied_prob_to_decimal_odds(market_b)
            bet_size = bankroll.kelly_bet_size(state["selected_b"], odds)
            bet_size = round(bet_size * newbie_adjustment.size_multiplier, 2) if bet_size > 0 else bet_size
            if bet_size > 0:
                bet_idx = len(bankroll.history)
                bankroll.place_bet(
                    bet_size,
                    row.get("fighter_b", "B"),
                    odds,
                    state["selected_b"],
                    market_b,
                )
                bankroll.settle_bet(bet_idx, won=not actual_winner_is_a)
                placed_bet = True
                clv = np.nan
                if pd.notna(row.get("closing_prob_b")):
                    clv = calculate_closing_line_value(market_b, row["closing_prob_b"])

                bet_log.append({
                    "strategy": strategy_config.name,
                    "event_date": event_date,
                    "fighter_a": row.get("fighter_a", ""),
                    "fighter_b": row.get("fighter_b", ""),
                    "bet_on": row.get("fighter_b", "B"),
                    "bet_side": "b",
                    "bet_size": bet_size,
                    "odds": odds,
                    "model_prob": state["primary_b"],
                    "blended_prob": state["selected_b"],
                    "blend_weight_used": state["blend_weight_used"],
                    "agreement_prob": state["agreement_b"],
                    "market_prob": market_b,
                    "edge": edge_b,
                    "won": not actual_winner_is_a,
                    "profit": bankroll.history[-1]["profit"],
                    "bankroll_after": bankroll.bankroll,
                    "clv": clv,
                    "closing_prob": row.get("closing_prob_b", np.nan),
                    "fold": row.get("fold", np.nan),
                    "train_end": row.get("train_end"),
                    "test_end": row.get("test_end"),
                    "odds_source": row.get("odds_source", "unknown"),
                    "is_newbie_bet": newbie_adjustment.is_newbie_bet,
                    "size_multiplier": newbie_adjustment.size_multiplier,
                    "newbie_extra_edge_required": newbie_adjustment.extra_edge_required,
                    "newbie_rule": strategy_config.newbie_rule.name,
                    "newbie_reason": newbie_adjustment.reason,
                })

        if placed_bet:
            bankroll_history.append(bankroll.bankroll)

        if placed_bet and bankroll.is_stopped:
            logger.warning(f"Stop-loss triggered for strategy '{strategy_config.name}'.")

    bet_log_df = pd.DataFrame(bet_log)
    stats = bankroll.get_stats()
    stats.update(_compute_clv_stats(bet_log_df))
    dd = compute_max_drawdown(bankroll_history)
    stats.update(dd)
    stats["max_drawdown"] = dd["max_drawdown_pct"]

    return {
        "strategy_config": strategy_config,
        "stats": stats,
        "bet_log": bet_log_df,
        "bankroll_history": bankroll_history,
        "predictions": predictions,
        "bankroll_manager": bankroll,
        "odds_source": predictions["odds_source"].iloc[0] if not predictions.empty else "unknown",
    }


def summarize_strategy_results(
    results: dict[str, dict],
) -> pd.DataFrame:
    """Summarize strategy P&L for comparison output."""
    rows = []
    for name, result in results.items():
        strategy = result["strategy_config"]
        stats = result["stats"]
        rows.append({
            "strategy": name,
            "primary_model": strategy.primary_model,
            "selection_mode": strategy.selection_mode,
            "require_agreement": strategy.require_agreement,
            "agreement_model": strategy.agreement_model or "",
            "blend_weight": strategy.blend_weight if strategy.selection_mode == "blended" else 1.0,
            "max_decimal_odds": strategy.max_decimal_odds,
            "newbie_rule": strategy.newbie_rule.name,
            "total_bets": stats.get("total_bets", 0),
            "wins": stats.get("wins", 0),
            "win_rate": stats.get("win_rate", 0.0),
            "total_wagered": stats.get("total_wagered", 0.0),
            "total_profit": stats.get("total_profit", 0.0),
            "roi": stats.get("roi", 0.0),
            "final_bankroll": stats.get("bankroll", 0.0),
            "bankroll_change_pct": stats.get("bankroll_change_pct", 0.0),
            "avg_edge": stats.get("avg_edge", 0.0),
            "avg_clv": stats.get("avg_clv", np.nan),
            "max_drawdown_pct": stats.get("max_drawdown_pct", 0.0),
            "max_drawdown_duration": stats.get("max_drawdown_duration", 0),
        })
    return pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)


def summarize_strategy_folds(
    combined_predictions: pd.DataFrame,
    results: dict[str, dict],
) -> pd.DataFrame:
    """Summarize strategy performance by walk-forward fold."""
    if combined_predictions.empty:
        return pd.DataFrame()

    base = combined_predictions.groupby("fold", observed=True).agg(
        train_end=("train_end", "first"),
        test_end=("test_end", "first"),
        test_fights=("target", "size"),
    ).reset_index()

    rows = []
    for name, result in results.items():
        bet_log = result["bet_log"]
        if bet_log.empty:
            fold_stats = base.copy()
            fold_stats["strategy"] = name
            fold_stats["bets"] = 0
            fold_stats["wins"] = 0
            fold_stats["win_rate"] = 0.0
            fold_stats["total_wagered"] = 0.0
            fold_stats["total_profit"] = 0.0
            fold_stats["roi"] = 0.0
            fold_stats["avg_clv"] = np.nan
            rows.append(fold_stats)
            continue

        agg = bet_log.groupby("fold", observed=True).agg(
            bets=("won", "size"),
            wins=("won", "sum"),
            total_wagered=("bet_size", "sum"),
            total_profit=("profit", "sum"),
            avg_clv=("clv", "mean"),
        ).reset_index()
        agg["win_rate"] = np.where(agg["bets"] > 0, agg["wins"] / agg["bets"], 0.0)
        agg["roi"] = np.where(agg["total_wagered"] > 0, agg["total_profit"] / agg["total_wagered"], 0.0)

        fold_stats = base.merge(agg, on="fold", how="left").fillna({
            "bets": 0,
            "wins": 0,
            "total_wagered": 0.0,
            "total_profit": 0.0,
            "win_rate": 0.0,
            "roi": 0.0,
        })
        fold_stats["strategy"] = name
        rows.append(fold_stats)

    return pd.concat(rows, ignore_index=True).sort_values(["strategy", "fold"]).reset_index(drop=True)


def _model_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob > 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": float(compute_ece(y_true, y_prob)),
        "n_fights": int(len(y_true)),
    }


def build_predictive_comparison_report(
    combined_predictions: pd.DataFrame,
) -> dict:
    """Compute walk-forward predictive metrics and paired significance tests."""
    from src.model.compare import bootstrap_metric_comparison, mcnemar_test

    if combined_predictions.empty:
        return {
            "evaluation_window": {"n_fights": 0, "n_folds": 0},
            "models": {},
            "comparison": {},
        }

    y_true = combined_predictions["target"].astype(int).to_numpy()
    odds_probs = combined_predictions[_prediction_col("xgboost", "prob_a")].to_numpy()
    no_odds_probs = combined_predictions[_prediction_col("xgboost_no_odds", "prob_a")].to_numpy()

    return {
        "evaluation_window": {
            "start_date": str(pd.to_datetime(combined_predictions["event_date"]).min().date()),
            "end_date": str(pd.to_datetime(combined_predictions["event_date"]).max().date()),
            "n_fights": int(len(combined_predictions)),
            "n_folds": int(combined_predictions["fold"].nunique()) if "fold" in combined_predictions.columns else 1,
        },
        "models": {
            "xgboost": _model_metrics(y_true, odds_probs),
            "xgboost_no_odds": _model_metrics(y_true, no_odds_probs),
        },
        "comparison": {
            "baseline_model": "xgboost_no_odds",
            "challenger_model": "xgboost",
            "mcnemar": mcnemar_test(
                y_true,
                (no_odds_probs > 0.5).astype(int),
                (odds_probs > 0.5).astype(int),
            ),
            "brier_bootstrap": bootstrap_metric_comparison(
                y_true,
                no_odds_probs,
                odds_probs,
                brier_score_loss,
            ),
            "logloss_bootstrap": bootstrap_metric_comparison(
                y_true,
                no_odds_probs,
                odds_probs,
                log_loss,
            ),
        },
    }


def _write_walkforward_comparison_artifacts(
    summary_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    predictive_report: dict,
) -> dict[str, str]:
    """Write walk-forward comparison outputs into logs/comparison."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = COMPARISON_DIR / f"walkforward_strategy_summary_{timestamp}.csv"
    fold_path = COMPARISON_DIR / f"walkforward_fold_results_{timestamp}.csv"
    metrics_path = COMPARISON_DIR / f"walkforward_predictive_metrics_{timestamp}.json"

    latest_summary_path = COMPARISON_DIR / "walkforward_strategy_summary_latest.csv"
    latest_fold_path = COMPARISON_DIR / "walkforward_fold_results_latest.csv"
    latest_metrics_path = COMPARISON_DIR / "walkforward_predictive_metrics_latest.json"

    summary_df.to_csv(summary_path, index=False)
    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(latest_summary_path, index=False)
    fold_df.to_csv(latest_fold_path, index=False)

    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(predictive_report, fh, indent=2)
    with latest_metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(predictive_report, fh, indent=2)

    logger.info(f"Saved walk-forward strategy summary to {summary_path}")
    logger.info(f"Saved walk-forward fold results to {fold_path}")
    logger.info(f"Saved walk-forward predictive report to {metrics_path}")

    return {
        "summary_csv": str(summary_path),
        "fold_csv": str(fold_path),
        "predictive_json": str(metrics_path),
    }


def run_backtest(
    test_df: pd.DataFrame,
    model_name: str = "xgboost",
    model_result: Optional[dict] = None,
    agreement_model_result: Optional[dict] = None,
    strategy_config: Optional[BacktestStrategyConfig] = None,
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
    Run a static backtest on historical fight data.

    Agreement behavior is controlled explicitly by `strategy_config`.
    Legacy callers using only `model_name` are mapped onto a compatible
    strategy config.
    """
    strategy_config = _resolve_strategy_config(strategy_config, model_name, blend_weight)
    model_results = _load_strategy_models(
        strategy_config,
        model_result=model_result,
        agreement_model_result=agreement_model_result,
    )
    predictions, _ = _prepare_prediction_frame(
        test_df,
        model_results=model_results,
        market_prob_col_a=market_prob_col_a,
        market_prob_col_b=market_prob_col_b,
        use_historical_odds=use_historical_odds,
    )
    result = _simulate_backtest_predictions(
        predictions,
        strategy_config,
        initial_bankroll=initial_bankroll,
        min_edge=min_edge,
        kelly_fraction=kelly_fraction,
        max_bet_fraction=max_bet_fraction,
    )
    _log_backtest_summary("BACKTEST RESULTS", strategy_config, result)
    return result


def run_comparison_backtest(
    test_df: pd.DataFrame,
    initial_bankroll: float = INITIAL_BANKROLL,
    min_edge: float = MIN_EDGE_THRESHOLD,
    kelly_fraction: float = KELLY_FRACTION,
    max_bet_fraction: float = MAX_BET_FRACTION,
    strategies: Optional[Sequence[BacktestStrategyConfig]] = None,
) -> dict[str, dict]:
    """Run static strategy comparison on the same prepared predictions."""
    strategies = tuple(strategies or COMPARISON_STRATEGIES)
    model_names = {
        model_name
        for strategy in strategies
        for model_name in strategy.required_models()
    }
    model_results = {model_name: load_model(model_name) for model_name in model_names}
    predictions, odds_source = _prepare_prediction_frame(test_df, model_results=model_results)

    results = {
        strategy.name: _simulate_backtest_predictions(
            predictions,
            strategy,
            initial_bankroll=initial_bankroll,
            min_edge=min_edge,
            kelly_fraction=kelly_fraction,
            max_bet_fraction=max_bet_fraction,
        )
        for strategy in strategies
    }

    logger.info(f"\n{'=' * 60}")
    logger.info("STATIC STRATEGY COMPARISON")
    logger.info(f"{'=' * 60}")
    logger.info(f"Odds source: {odds_source}")
    logger.info(summarize_strategy_results(results).to_string(index=False))
    logger.info(f"{'=' * 60}")

    return results


def run_walkforward_strategy_comparison(
    features_df: pd.DataFrame,
    retrain_months: int = 6,
    initial_train_years: int = 5,
    min_edge: float = MIN_EDGE_THRESHOLD,
    kelly_fraction: float = KELLY_FRACTION,
    max_bet_fraction: float = MAX_BET_FRACTION,
    initial_bankroll: float = INITIAL_BANKROLL,
    bet_start_date: str = TRAIN_CUTOFF_DATE,
    strategies: Optional[Sequence[BacktestStrategyConfig]] = None,
    write_artifacts: bool = True,
    spec: "NamedModelTrainingSpec | None" = None,
    min_train_test_fights: int = 2,
) -> dict:
    """Run a clean walk-forward comparison using the promoted training contract."""
    from src.features.build_features import exclude_market_derived_features
    from src.model.train import train_xgboost
    from src.model.training_spec import (
        full_live_contract_spec,
        materialize_and_validate_spec_features,
    )

    strategies = tuple(strategies or COMPARISON_STRATEGIES)
    spec = spec or full_live_contract_spec()
    violations = spec.validate_feature_contract()
    if violations:
        raise ValueError(f"Training spec has {len(violations)} contract violations")

    features_df = materialize_and_validate_spec_features(features_df, spec)
    features_df = features_df.sort_values("event_date").copy()
    n_before = len(features_df)
    features_df = features_df.dropna(subset=["target"])
    n_dropped = n_before - len(features_df)
    if n_dropped:
        logger.info(f"Dropped {n_dropped} fights with no target (draws/NC/DQ)")

    feature_cols = list(spec.feature_cols)
    no_odds_cols = exclude_market_derived_features(feature_cols)

    if (
        min_train_test_fights > 0
        and "a_num_fights" in features_df.columns
        and "b_num_fights" in features_df.columns
    ):
        features_df = features_df[
            (features_df["a_num_fights"] >= min_train_test_fights)
            & (features_df["b_num_fights"] >= min_train_test_fights)
        ]

    dates = pd.to_datetime(features_df["event_date"])
    min_date = dates.min()
    max_date = dates.max()
    train_end = min_date + pd.DateOffset(years=initial_train_years)
    bet_start = pd.Timestamp(bet_start_date)

    fold_predictions: list[pd.DataFrame] = []
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
            f"\n--- Walk-forward fold {fold_num}: "
            f"Train {len(train_df)} fights (to {train_end.date()}), "
            f"Test {len(test_df)} fights ({train_end.date()} to {test_end.date()}) ---"
        )

        xgb_result = train_xgboost(
            train_df,
            feature_cols,
            calibrate=True,
            impute_strategy=spec.impute_strategy,
            xgb_params=spec.xgb_params,
            calibration_method=spec.calibration_method,
            calibration_cv=spec.calibration_cv,
            odds_noise_std=getattr(spec, "odds_noise_std", 0.04),
            time_decay_half_life_days=getattr(spec, "time_decay_half_life", None),
        )
        no_odds_result = train_xgboost(
            train_df,
            no_odds_cols,
            calibrate=True,
            impute_strategy=spec.impute_strategy,
            xgb_params=spec.xgb_params,
            calibration_method=spec.calibration_method,
            calibration_cv=spec.calibration_cv,
            odds_noise_std=getattr(spec, "odds_noise_std", 0.04),
            time_decay_half_life_days=getattr(spec, "time_decay_half_life", None),
        )

        fold_frame, odds_source = _prepare_prediction_frame(
            test_df,
            model_results={
                "xgboost": xgb_result,
                "xgboost_no_odds": no_odds_result,
            },
            market_prob_col_a="a_fair_prob_avg",
            market_prob_col_b="b_fair_prob_avg",
            use_historical_odds=True,
        )

        fold_frame = fold_frame[pd.to_datetime(fold_frame["event_date"]) >= bet_start].copy()
        if fold_frame.empty:
            train_end = test_end
            continue

        fold_frame["fold"] = fold_num
        fold_frame["train_end"] = str(train_end.date())
        fold_frame["test_end"] = str(test_end.date())
        fold_frame["odds_source"] = odds_source
        fold_predictions.append(fold_frame)

        train_end = test_end

    if not fold_predictions:
        empty = pd.DataFrame()
        return {
            "strategy_results": {},
            "summary": empty,
            "fold_results": empty,
            "predictive_metrics": {
                "evaluation_window": {"n_fights": 0, "n_folds": 0},
                "models": {},
                "comparison": {},
            },
            "predictions": empty,
            "artifacts": {},
        }

    combined_predictions = pd.concat(fold_predictions, ignore_index=True)
    combined_predictions = combined_predictions.sort_values(["event_date", "fold"]).reset_index(drop=True)

    strategy_results = {
        strategy.name: _simulate_backtest_predictions(
            combined_predictions,
            strategy,
            initial_bankroll=initial_bankroll,
            min_edge=min_edge,
            kelly_fraction=kelly_fraction,
            max_bet_fraction=max_bet_fraction,
        )
        for strategy in strategies
    }

    summary_df = summarize_strategy_results(strategy_results)
    fold_df = summarize_strategy_folds(combined_predictions, strategy_results)
    predictive_report = build_predictive_comparison_report(combined_predictions)
    artifacts = (
        _write_walkforward_comparison_artifacts(summary_df, fold_df, predictive_report)
        if write_artifacts
        else {}
    )

    logger.info(f"\n{'=' * 60}")
    logger.info("WALK-FORWARD STRATEGY COMPARISON")
    logger.info(f"{'=' * 60}")
    logger.info(summary_df.to_string(index=False))
    logger.info(f"{'=' * 60}")

    return {
        "strategy_results": strategy_results,
        "summary": summary_df,
        "fold_results": fold_df,
        "predictive_metrics": predictive_report,
        "predictions": combined_predictions,
        "artifacts": artifacts,
        "spec": spec,
    }


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
    spec: "NamedModelTrainingSpec | None" = None,
) -> dict:
    """
    Legacy single-strategy walk-forward backtest.

    This now delegates to the explicit strategy comparison engine and returns
    the production-gated strategy result.
    """
    strategy = replace(PRODUCTION_GATED_STRATEGY, blend_weight=blend_weight)
    comparison = run_walkforward_strategy_comparison(
        features_df,
        retrain_months=retrain_months,
        initial_train_years=initial_train_years,
        min_edge=min_edge,
        kelly_fraction=kelly_fraction,
        max_bet_fraction=max_bet_fraction,
        initial_bankroll=initial_bankroll,
        bet_start_date=bet_start_date,
        strategies=[strategy],
        write_artifacts=False,
        spec=spec,
    )

    result = comparison["strategy_results"][strategy.name]
    fold_stats = comparison["fold_results"]
    if not fold_stats.empty:
        fold_stats = fold_stats[fold_stats["strategy"] == strategy.name].drop(columns=["strategy"])
    result["fold_stats"] = fold_stats.reset_index(drop=True)
    result["odds_source"] = "walk_forward"
    _log_backtest_summary("WALK-FORWARD BACKTEST RESULTS", strategy, result)
    return result


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

    ax = axes[0, 0]
    ax.plot(bankroll_history, linewidth=1.5, color="steelblue")
    ax.axhline(
        y=bankroll_history[0],
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Starting bankroll",
    )
    ax.set_xlabel("Bet #")
    ax.set_ylabel("Bankroll ($)")
    odds_src = backtest_result.get("odds_source", "unknown")
    ax.set_title(f"Bankroll Over Time (ROI: {stats.get('roi', 0):+.1%}) [{odds_src}]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    cum_pnl = bet_log["profit"].cumsum()
    colors = ["green" if p > 0 else "red" for p in bet_log["profit"]]
    ax.bar(range(len(bet_log)), bet_log["profit"], color=colors, alpha=0.6)
    ax.plot(cum_pnl.values, color="black", linewidth=2, label="Cumulative P&L")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Bet #")
    ax.set_ylabel("Profit ($)")
    ax.set_title("Per-Bet P&L and Cumulative")
    ax.legend()

    ax = axes[1, 0]
    won = bet_log[bet_log["won"] == True]["edge"]
    lost = bet_log[bet_log["won"] == False]["edge"]
    ax.hist(won, bins=15, alpha=0.6, label=f"Wins ({len(won)})", color="green")
    ax.hist(lost, bins=15, alpha=0.6, label=f"Losses ({len(lost)})", color="red")
    ax.set_xlabel("Edge (selection prob - market prob)")
    ax.set_ylabel("Count")
    ax.set_title("Edge Distribution by Outcome")
    ax.legend()

    ax = axes[1, 1]
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
            ax.text(i, row["mean"] + 0.01, f"n={int(row['count'])}", ha="center", fontsize=8)

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
    """Run static backtests over parameter combinations."""
    if edge_thresholds is None:
        edge_thresholds = [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
    if kelly_fractions is None:
        kelly_fractions = [0.10, 0.15, 0.20, 0.25, 0.33, 0.50]

    model_result = load_model(model_name)
    results = []

    for edge in edge_thresholds:
        for kelly in kelly_fractions:
            backtest = run_backtest(
                test_df,
                model_name=model_name,
                model_result=model_result,
                min_edge=edge,
                kelly_fraction=kelly,
            )
            stats = backtest["stats"]
            results.append({
                "min_edge": edge,
                "kelly_fraction": kelly,
                "total_bets": stats.get("total_bets", 0),
                "win_rate": stats.get("win_rate", 0),
                "roi": stats.get("roi", 0),
                "total_profit": stats.get("total_profit", 0),
                "bankroll_change_pct": stats.get("bankroll_change_pct", 0),
                "avg_edge": stats.get("avg_edge", 0),
                "avg_clv": stats.get("avg_clv", np.nan),
            })

    result_df = pd.DataFrame(results)
    path = LOGS_DIR / "sensitivity_analysis.csv"
    result_df.to_csv(path, index=False)
    logger.info(f"Sensitivity analysis saved to {path}")
    logger.info(f"\nBest ROI configs:\n{result_df.nlargest(5, 'roi').to_string(index=False)}")
    return result_df
