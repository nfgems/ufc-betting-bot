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
    MAX_BET_VS_BOOK_RATIO,
    MAX_SLIPPAGE,
    MAX_BET_FRACTION,
    MIN_EDGE_THRESHOLD,
    MIN_BOOK_LIQUIDITY,
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
SUPPORTED_EXECUTION_MODES = frozenset({"legacy", "realistic"})


@dataclass(frozen=True)
class BacktestExecutionConfig:
    """Execution assumptions for static and walk-forward backtests."""

    mode: str = "legacy"
    min_book_liquidity: float = MIN_BOOK_LIQUIDITY
    max_slippage: float = MAX_SLIPPAGE
    max_bet_vs_book_ratio: float = MAX_BET_VS_BOOK_RATIO
    assumed_half_spread: float = 0.005
    synthetic_liquidity_floor: float = 35.0
    synthetic_liquidity_peak: float = 180.0
    synthetic_price_step: float = 0.01
    synthetic_depth_notional_shares: tuple[float, ...] = (0.10, 0.15, 0.25, 0.50)

    def assumptions_dict(self) -> dict[str, object]:
        """Return a JSON-safe summary of the execution assumptions."""
        return {
            "mode": self.mode,
            "min_book_liquidity": self.min_book_liquidity,
            "max_slippage": self.max_slippage,
            "max_bet_vs_book_ratio": self.max_bet_vs_book_ratio,
            "assumed_half_spread": self.assumed_half_spread,
            "synthetic_liquidity_floor": self.synthetic_liquidity_floor,
            "synthetic_liquidity_peak": self.synthetic_liquidity_peak,
            "synthetic_price_step": self.synthetic_price_step,
            "synthetic_depth_notional_shares": list(self.synthetic_depth_notional_shares),
        }


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

    # Layer 0 (above opening): explicit time-to-event entry snapshot, present
    # only when the caller requested entry_offset_days pricing.
    has_entry = "entry_prob_a" in predictions.columns
    if has_entry:
        mask = predictions["entry_prob_a"].notna()
        predictions.loc[mask, "a_market_prob"] = predictions.loc[mask, "entry_prob_a"]
        predictions.loc[mask, "b_market_prob"] = predictions.loc[mask, "entry_prob_b"]

    # Report coverage
    filled = predictions["a_market_prob"].notna().sum()
    total_rows = len(predictions)
    sources_used = []
    if has_entry:
        sources_used.append("entry_offset")
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
    entry_offset_days: float | None = None,
) -> pd.DataFrame:
    """Merge historical opening/closing odds and line-movement metadata.

    When `entry_offset_days` is set, an additional `entry_*` snapshot is
    merged: per fight, the snapshot taken closest to (but not after)
    `entry_offset_days` days before the event. This lets backtests price
    fills at a realistic time-to-event instead of the T-7 opening snapshot,
    while `opening_*`/`closing_*` stay intact for features and CLV.
    """
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

    entry = None
    if entry_offset_days is not None:
        eligible = hist[pd.to_numeric(hist["offset_days"], errors="coerce") >= entry_offset_days]
        if not eligible.empty:
            entry = eligible.loc[
                eligible.groupby(["event_date", "fighter_a", "fighter_b"])["offset_days"].idxmin()
            ]
            entry = entry.rename(columns={
                "a_fair_prob": "entry_prob_a",
                "b_fair_prob": "entry_prob_b",
                "a_decimal_odds": "entry_odds_a",
                "b_decimal_odds": "entry_odds_b",
                "offset_days": "entry_offset_actual",
            })[[
                "event_date",
                "fighter_a",
                "fighter_b",
                "entry_prob_a",
                "entry_prob_b",
                "entry_odds_a",
                "entry_odds_b",
                "entry_offset_actual",
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
    if entry is not None:
        entry["event_date_str"] = entry["event_date"].astype(str)
        merged = merged.merge(
            entry.drop(columns=["event_date"]),
            on=["event_date_str", "fighter_a", "fighter_b"],
            how="left",
        )
        matched_entry = merged["entry_prob_a"].notna().sum()
        logger.info(
            f"Matched {matched_entry}/{len(predictions)} fights with T-{entry_offset_days} entry odds"
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
    entry_offset_days: float | None = None,
    entry_offset_for_features: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Prepare a prediction frame with prefixed model outputs and market odds.

    `entry_offset_days` prices fills at the snapshot closest to (but not
    after) that many days before the event instead of the T-7 opening.
    `entry_offset_for_features` additionally feeds the same snapshot into the
    model's implied_prob feature columns (full live-window parity arm).
    """
    predictions = test_df.copy()

    # Merge historical odds BEFORE model predictions so opening odds can
    # replace closing odds in the model's implied_prob feature columns.
    # This prevents look-ahead bias: the model sees pre-fight opening odds
    # (matching live conditions) instead of post-fight closing odds.
    if use_historical_odds:
        predictions = _merge_historical_odds(predictions, entry_offset_days=entry_offset_days)
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
        if entry_offset_for_features and "entry_prob_a" in predictions.columns:
            mask = predictions["entry_prob_a"].notna()
            n_swapped = mask.sum()
            if n_swapped:
                predictions.loc[mask, "a_implied_prob"] = predictions.loc[mask, "entry_prob_a"]
                predictions.loc[mask, "b_implied_prob"] = predictions.loc[mask, "entry_prob_b"]
                predictions.loc[mask, "diff_implied_prob"] = (
                    predictions.loc[mask, "a_implied_prob"]
                    - predictions.loc[mask, "b_implied_prob"]
                )
                logger.info(
                    f"Replaced model-feature odds with T-{entry_offset_days} entry odds "
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


def _resolve_execution_config(
    execution_mode: str,
    execution_config: "BacktestExecutionConfig | None" = None,
) -> BacktestExecutionConfig:
    normalized_mode = str(execution_mode or "").strip().lower() or "legacy"
    if normalized_mode not in SUPPORTED_EXECUTION_MODES:
        supported = ", ".join(sorted(SUPPORTED_EXECUTION_MODES))
        raise ValueError(
            f"Unsupported backtest execution_mode '{execution_mode}'. Supported values: {supported}"
        )

    if execution_config is None:
        return BacktestExecutionConfig(mode=normalized_mode)
    if execution_config.mode != normalized_mode:
        return replace(execution_config, mode=normalized_mode)
    return execution_config


def _event_group_key(row: pd.Series) -> tuple[str, str]:
    event_date = pd.Timestamp(row.get("event_date"))
    event_name = str(row.get("event_name", "") or "").strip()
    fallback_name = str(row.get("location", "") or "").strip()
    label = event_name or fallback_name or "unknown_event"
    return (event_date.normalize().strftime("%Y-%m-%d"), label)


def _normalized_probability(value: float, *, floor: float = 0.01, ceiling: float = 0.99) -> float:
    return float(min(max(value, floor), ceiling))


def _estimated_best_ask(
    market_prob: float,
    execution_config: BacktestExecutionConfig,
) -> float:
    return _normalized_probability(market_prob + execution_config.assumed_half_spread)


def _assumed_book_liquidity(
    row: pd.Series,
    market_prob: float,
    execution_config: BacktestExecutionConfig,
) -> float:
    for column in ("assumed_book_liquidity", "book_liquidity", "liquidity"):
        value = _clean_optional_float(row.get(column))
        if value is not None and value > 0:
            return float(value)

    concentration = 1.0 - (4.0 * market_prob * (1.0 - market_prob))
    concentration = float(min(max(concentration, 0.0), 1.0))
    spread = max(
        execution_config.synthetic_liquidity_peak - execution_config.synthetic_liquidity_floor,
        0.0,
    )
    return float(
        execution_config.synthetic_liquidity_peak - (spread * concentration)
    )


def _synthetic_orderbook_levels(
    best_ask: float,
    total_liquidity: float,
    execution_config: BacktestExecutionConfig,
) -> list[dict[str, float]]:
    levels: list[dict[str, float]] = []
    shares = execution_config.synthetic_depth_notional_shares
    normalized = float(sum(shares))
    if normalized <= 0:
        return levels

    for idx, fraction in enumerate(shares):
        notional = total_liquidity * (float(fraction) / normalized)
        if notional <= 0:
            continue
        level_price = _normalized_probability(
            best_ask + (execution_config.synthetic_price_step * idx),
        )
        levels.append({
            "price": level_price,
            "size": notional / level_price if level_price > 0 else 0.0,
            "notional": notional,
        })
    return levels


def _simulate_realistic_fill(
    row: pd.Series,
    market_prob: float,
    requested_stake: float,
    execution_config: BacktestExecutionConfig,
) -> dict[str, object]:
    best_ask = _estimated_best_ask(market_prob, execution_config)
    available_liquidity = _assumed_book_liquidity(row, market_prob, execution_config)
    result: dict[str, object] = {
        "ok": True,
        "requested_stake": float(requested_stake),
        "filled_stake": float(requested_stake),
        "available_liquidity": float(available_liquidity),
        "best_ask": float(best_ask),
        "fill_price": float(best_ask),
        "slippage": 0.0,
        "partial_fill": False,
        "reason": "",
    }

    if requested_stake <= 0:
        result["ok"] = False
        result["filled_stake"] = 0.0
        result["reason"] = "non_positive_requested_stake"
        return result

    if available_liquidity < execution_config.min_book_liquidity:
        result["ok"] = False
        result["filled_stake"] = 0.0
        result["reason"] = (
            f"insufficient liquidity (${available_liquidity:.0f} < "
            f"${execution_config.min_book_liquidity:.0f} min)"
        )
        return result

    max_fillable = available_liquidity * execution_config.max_bet_vs_book_ratio
    filled_stake = min(requested_stake, max_fillable)
    result["filled_stake"] = float(round(filled_stake, 2))
    result["partial_fill"] = bool(result["filled_stake"] + 1e-9 < requested_stake)

    if result["filled_stake"] <= 0:
        result["ok"] = False
        result["reason"] = "max_bet_vs_book_ratio_reduced_fill_to_zero"
        return result

    levels = _synthetic_orderbook_levels(best_ask, available_liquidity, execution_config)
    filled_cost = 0.0
    filled_shares = 0.0
    for level in levels:
        remaining = float(result["filled_stake"]) - filled_cost
        if remaining <= 1e-9:
            break
        level_price = float(level["price"])
        level_size = float(level["size"])
        take_cost = min(level_price * level_size, remaining)
        if take_cost <= 0:
            continue
        filled_cost += take_cost
        filled_shares += take_cost / level_price

    if filled_shares <= 0:
        result["ok"] = False
        result["filled_stake"] = 0.0
        result["reason"] = "synthetic_orderbook_returned_zero_fill"
        return result

    avg_fill_price = filled_cost / filled_shares
    slippage = (avg_fill_price - best_ask) / best_ask if best_ask > 0 else 0.0
    result["fill_price"] = float(avg_fill_price)
    result["slippage"] = float(slippage)

    if slippage > execution_config.max_slippage:
        result["ok"] = False
        result["filled_stake"] = 0.0
        result["reason"] = (
            f"slippage too high ({slippage:.1%} > {execution_config.max_slippage:.0%}) "
            f"for ${requested_stake:.2f} requested stake"
        )
        return result

    return result


def _execution_metrics_seed(execution_config: BacktestExecutionConfig) -> dict[str, object]:
    return {
        "execution_mode": execution_config.mode,
        "requested_stake_total": 0.0,
        "filled_stake_total": 0.0,
        "fill_rate": 0.0,
        "avg_fill_price": np.nan,
        "avg_slippage": 0.0,
        "max_slippage": 0.0,
        "skipped_for_liquidity_count": 0,
        "skipped_for_cash_count": 0,
        "partial_fills": 0,
        "execution_attempts": 0,
        "peak_locked_capital": 0.0,
        "peak_capital_utilization": 0.0,
        "avg_capital_utilization": 0.0,
    }


def _candidate_for_row(
    row: pd.Series,
    strategy_config: BacktestStrategyConfig,
    *,
    min_edge: float,
) -> dict | None:
    market_a = row["a_market_prob"]
    market_b = row["b_market_prob"]
    if pd.isna(market_a) or pd.isna(market_b):
        return None

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
        return {
            "event_date": pd.Timestamp(row.get("event_date")),
            "event_key": _event_group_key(row),
            "fighter_a": row.get("fighter_a", ""),
            "fighter_b": row.get("fighter_b", ""),
            "bet_on": row.get("fighter_a", "A"),
            "bet_side": "a",
            "model_prob": state["primary_a"],
            "blended_prob": state["selected_a"],
            "blend_weight_used": state["blend_weight_used"],
            "agreement_prob": state["agreement_a"],
            "market_prob": float(market_a),
            "edge": float(edge_a),
            "won": bool(actual_winner_is_a),
            "closing_prob": row.get("closing_prob_a", np.nan),
            "is_newbie_bet": newbie_adjustment.is_newbie_bet,
            "size_multiplier": newbie_adjustment.size_multiplier,
            "newbie_extra_edge_required": newbie_adjustment.extra_edge_required,
            "newbie_rule": strategy_config.newbie_rule.name,
            "newbie_reason": newbie_adjustment.reason,
            "fold": row.get("fold", np.nan),
            "train_end": row.get("train_end"),
            "test_end": row.get("test_end"),
            "odds_source": row.get("odds_source", "unknown"),
        }

    if edge_b >= min_edge and _passes_filters(
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
        return {
            "event_date": pd.Timestamp(row.get("event_date")),
            "event_key": _event_group_key(row),
            "fighter_a": row.get("fighter_a", ""),
            "fighter_b": row.get("fighter_b", ""),
            "bet_on": row.get("fighter_b", "B"),
            "bet_side": "b",
            "model_prob": state["primary_b"],
            "blended_prob": state["selected_b"],
            "blend_weight_used": state["blend_weight_used"],
            "agreement_prob": state["agreement_b"],
            "market_prob": float(market_b),
            "edge": float(edge_b),
            "won": bool(not actual_winner_is_a),
            "closing_prob": row.get("closing_prob_b", np.nan),
            "is_newbie_bet": newbie_adjustment.is_newbie_bet,
            "size_multiplier": newbie_adjustment.size_multiplier,
            "newbie_extra_edge_required": newbie_adjustment.extra_edge_required,
            "newbie_rule": strategy_config.newbie_rule.name,
            "newbie_reason": newbie_adjustment.reason,
            "fold": row.get("fold", np.nan),
            "train_end": row.get("train_end"),
            "test_end": row.get("test_end"),
            "odds_source": row.get("odds_source", "unknown"),
        }

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
    logger.info(f"Execution mode: {stats.get('execution_mode', 'legacy')}")
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
    if stats.get("execution_mode") == "realistic":
        logger.info(
            "Requested/Filled stake: $%.2f / $%.2f (fill rate %.1f%%)",
            stats.get("requested_stake_total", 0.0),
            stats.get("filled_stake_total", 0.0),
            stats.get("fill_rate", 0.0) * 100.0,
        )
        logger.info(
            "Liquidity skips: %s | Partial fills: %s | Avg slippage: %.2f%% | Peak capital utilization: %.1f%%",
            stats.get("skipped_for_liquidity_count", 0),
            stats.get("partial_fills", 0),
            stats.get("avg_slippage", 0.0) * 100.0,
            stats.get("peak_capital_utilization", 0.0) * 100.0,
        )
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
    execution_mode: str = "legacy",
    execution_config: "BacktestExecutionConfig | None" = None,
) -> dict:
    """Run the betting simulation over a prepared prediction frame."""
    execution_config = _resolve_execution_config(
        execution_mode,
        execution_config=execution_config,
    )
    bankroll = BankrollManager(
        initial_bankroll=initial_bankroll,
        kelly_fraction=kelly_fraction,
        max_bet_fraction=max_bet_fraction,
        auto_detect_balance=False,
    )
    bankroll_history = [initial_bankroll]
    bet_log: list[dict] = []
    bet_start = pd.Timestamp(bet_start_date) if bet_start_date else None
    execution_metrics = _execution_metrics_seed(execution_config)
    weighted_fill_price = 0.0
    weighted_slippage = 0.0
    utilization_samples: list[float] = []
    current_event_key: tuple[str, str] | None = None
    open_event_bets: list[dict] = []

    def _record_settlement(open_bet: dict) -> None:
        bet_index = int(open_bet["bet_index"])
        bankroll.settle_bet(bet_index, won=bool(open_bet["won"]))
        settled = bankroll.history[bet_index]
        log_entry = dict(open_bet["log_entry"])
        fill_price = float(log_entry["fill_price"])
        closing_prob = log_entry.get("closing_prob", np.nan)
        clv = np.nan
        if pd.notna(closing_prob):
            clv = calculate_closing_line_value(fill_price, float(closing_prob))
        log_entry["profit"] = settled["profit"]
        log_entry["bankroll_after"] = bankroll.bankroll
        log_entry["total_equity_after"] = bankroll.total_equity
        log_entry["clv"] = clv
        bet_log.append(log_entry)
        bankroll_history.append(bankroll.bankroll)

    def _settle_open_event_bets() -> None:
        nonlocal open_event_bets
        if not open_event_bets:
            return
        for open_bet in open_event_bets:
            _record_settlement(open_bet)
        open_event_bets = []

    for _, row in predictions.iterrows():
        event_date = pd.Timestamp(row.get("event_date"))
        if bet_start is not None and event_date < bet_start:
            continue

        if bankroll.is_stopped:
            logger.warning(f"Stop-loss triggered for strategy '{strategy_config.name}'.")
            break

        event_key = _event_group_key(row)
        if (
            execution_config.mode == "realistic"
            and current_event_key is not None
            and event_key != current_event_key
        ):
            _settle_open_event_bets()

        current_event_key = event_key
        candidate = _candidate_for_row(
            row,
            strategy_config,
            min_edge=min_edge,
        )
        if candidate is None:
            continue

        quoted_price = (
            _estimated_best_ask(candidate["market_prob"], execution_config)
            if execution_config.mode == "realistic"
            else float(candidate["market_prob"])
        )
        quoted_odds = implied_prob_to_decimal_odds(quoted_price)
        requested_stake = bankroll.kelly_bet_size(candidate["blended_prob"], quoted_odds)
        requested_stake = (
            round(requested_stake * float(candidate["size_multiplier"]), 2)
            if requested_stake > 0
            else requested_stake
        )
        if requested_stake <= 0:
            continue

        execution_metrics["execution_attempts"] = int(execution_metrics["execution_attempts"]) + 1
        execution_metrics["requested_stake_total"] = float(execution_metrics["requested_stake_total"]) + requested_stake

        if execution_config.mode == "realistic":
            fill = _simulate_realistic_fill(
                row,
                float(candidate["market_prob"]),
                requested_stake,
                execution_config,
            )
            if not fill["ok"]:
                execution_metrics["skipped_for_liquidity_count"] = (
                    int(execution_metrics["skipped_for_liquidity_count"]) + 1
                )
                continue
        else:
            fill = {
                "ok": True,
                "requested_stake": float(requested_stake),
                "filled_stake": float(requested_stake),
                "available_liquidity": np.nan,
                "best_ask": float(candidate["market_prob"]),
                "fill_price": float(candidate["market_prob"]),
                "slippage": 0.0,
                "partial_fill": False,
                "reason": "",
            }

        filled_stake = round(float(fill["filled_stake"]), 2)
        if filled_stake <= 0:
            execution_metrics["skipped_for_liquidity_count"] = (
                int(execution_metrics["skipped_for_liquidity_count"]) + 1
            )
            continue

        fill_price = float(fill["fill_price"])
        fill_odds = implied_prob_to_decimal_odds(fill_price)
        bet_index = len(bankroll.history)
        placed = bankroll.place_bet(
            filled_stake,
            candidate["bet_on"],
            fill_odds,
            float(candidate["blended_prob"]),
            fill_price,
        )
        if not placed:
            execution_metrics["skipped_for_cash_count"] = (
                int(execution_metrics["skipped_for_cash_count"]) + 1
            )
            continue

        execution_metrics["filled_stake_total"] = float(execution_metrics["filled_stake_total"]) + filled_stake
        if bool(fill["partial_fill"]):
            execution_metrics["partial_fills"] = int(execution_metrics["partial_fills"]) + 1
        execution_metrics["max_slippage"] = max(
            float(execution_metrics["max_slippage"]),
            float(fill["slippage"]),
        )
        weighted_fill_price += filled_stake * fill_price
        weighted_slippage += filled_stake * float(fill["slippage"])

        locked_capital = max(bankroll.total_equity - bankroll.available_cash, 0.0)
        capital_utilization = (
            locked_capital / bankroll.total_equity if bankroll.total_equity > 0 else 0.0
        )
        utilization_samples.append(float(capital_utilization))
        execution_metrics["peak_locked_capital"] = max(
            float(execution_metrics["peak_locked_capital"]),
            locked_capital,
        )
        execution_metrics["peak_capital_utilization"] = max(
            float(execution_metrics["peak_capital_utilization"]),
            capital_utilization,
        )

        log_entry = {
            "strategy": strategy_config.name,
            "event_date": candidate["event_date"],
            "event_name": event_key[1],
            "fighter_a": candidate["fighter_a"],
            "fighter_b": candidate["fighter_b"],
            "bet_on": candidate["bet_on"],
            "bet_side": candidate["bet_side"],
            "requested_stake": requested_stake,
            "bet_size": filled_stake,
            "filled_stake": filled_stake,
            "quoted_price": quoted_price,
            "fill_price": fill_price,
            "quoted_odds": quoted_odds,
            "odds": fill_odds,
            "model_prob": candidate["model_prob"],
            "blended_prob": candidate["blended_prob"],
            "blend_weight_used": candidate["blend_weight_used"],
            "agreement_prob": candidate["agreement_prob"],
            "market_prob": candidate["market_prob"],
            "edge": candidate["edge"],
            "won": candidate["won"],
            "profit": np.nan,
            "bankroll_after": np.nan,
            "available_cash_before_bet": placed.get("available_cash_before", np.nan),
            "available_cash_after_reserve": placed.get("available_cash_after", np.nan),
            "locked_capital_after_bet": locked_capital,
            "capital_utilization_after_bet": capital_utilization,
            "clv": np.nan,
            "closing_prob": candidate["closing_prob"],
            "slippage": float(fill["slippage"]),
            "best_ask": fill["best_ask"],
            "available_liquidity": fill["available_liquidity"],
            "partial_fill": bool(fill["partial_fill"]),
            "execution_mode": execution_config.mode,
            "fold": candidate["fold"],
            "train_end": candidate["train_end"],
            "test_end": candidate["test_end"],
            "odds_source": candidate["odds_source"],
            "is_newbie_bet": candidate["is_newbie_bet"],
            "size_multiplier": candidate["size_multiplier"],
            "newbie_extra_edge_required": candidate["newbie_extra_edge_required"],
            "newbie_rule": candidate["newbie_rule"],
            "newbie_reason": candidate["newbie_reason"],
        }
        open_bet = {
            "bet_index": bet_index,
            "won": candidate["won"],
            "log_entry": log_entry,
        }
        if execution_config.mode == "legacy":
            _record_settlement(open_bet)
        else:
            open_event_bets.append(open_bet)

        if execution_config.mode == "legacy" and bankroll.is_stopped:
            logger.warning(f"Stop-loss triggered for strategy '{strategy_config.name}'.")

    if execution_config.mode == "realistic":
        _settle_open_event_bets()

    bet_log_df = pd.DataFrame(bet_log)
    stats = bankroll.get_stats()
    stats.update(_compute_clv_stats(bet_log_df))
    filled_total = float(execution_metrics["filled_stake_total"])
    requested_total = float(execution_metrics["requested_stake_total"])
    execution_metrics["fill_rate"] = (
        filled_total / requested_total if requested_total > 0 else 0.0
    )
    execution_metrics["avg_fill_price"] = (
        weighted_fill_price / filled_total if filled_total > 0 else np.nan
    )
    execution_metrics["avg_slippage"] = (
        weighted_slippage / filled_total if filled_total > 0 else 0.0
    )
    execution_metrics["avg_capital_utilization"] = (
        float(np.mean(utilization_samples)) if utilization_samples else 0.0
    )
    stats.update(execution_metrics)
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
        "execution_config": execution_config,
        "execution_assumptions": execution_config.assumptions_dict(),
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
            "execution_mode": stats.get("execution_mode", "legacy"),
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

    # E11 phase 0: the BLENDED probability is what crosses the edge threshold
    # and sizes bets, yet it was never calibration-measured anywhere. Score it
    # overall and inside the 0.40-0.65 band where the 2% threshold creates and
    # destroys bets.
    blend_metrics: dict = {}
    if (
        "a_market_prob" in combined_predictions.columns
        and "b_market_prob" in combined_predictions.columns
    ):
        from src.strategy.value import compute_independent_blend_probs

        market_a = pd.to_numeric(combined_predictions["a_market_prob"], errors="coerce").to_numpy()
        market_b = pd.to_numeric(combined_predictions["b_market_prob"], errors="coerce").to_numpy()
        blend_a = np.full(len(combined_predictions), np.nan)
        for i in range(len(combined_predictions)):
            if not (np.isfinite(market_a[i]) and np.isfinite(market_b[i])):
                continue
            try:
                blend_a[i], _ = compute_independent_blend_probs(
                    float(odds_probs[i]), float(market_a[i]), float(no_odds_probs[i]),
                    float(1.0 - odds_probs[i]), float(market_b[i]), float(1.0 - no_odds_probs[i]),
                    base_weight=BLEND_WEIGHT,
                )
            except Exception:
                continue
        valid = np.isfinite(blend_a)
        if valid.sum() >= 25:
            blend_metrics["overall"] = _model_metrics(y_true[valid], blend_a[valid])
            band = valid & (blend_a >= 0.40) & (blend_a <= 0.65)
            if band.sum() >= 25:
                blend_metrics["band_040_065"] = _model_metrics(y_true[band], blend_a[band])
            else:
                blend_metrics["band_040_065"] = {"n_fights": int(band.sum())}
            blend_metrics["blend_weight_base"] = float(BLEND_WEIGHT)

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
        "blend": blend_metrics,
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
    execution_mode: str = "legacy",
    execution_config: "BacktestExecutionConfig | None" = None,
    entry_offset_days: float | None = None,
    entry_offset_for_features: bool = False,
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
        entry_offset_days=entry_offset_days,
        entry_offset_for_features=entry_offset_for_features,
    )
    result = _simulate_backtest_predictions(
        predictions,
        strategy_config,
        initial_bankroll=initial_bankroll,
        min_edge=min_edge,
        kelly_fraction=kelly_fraction,
        max_bet_fraction=max_bet_fraction,
        execution_mode=execution_mode,
        execution_config=execution_config,
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
    execution_mode: str = "legacy",
    execution_config: "BacktestExecutionConfig | None" = None,
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
            execution_mode=execution_mode,
            execution_config=execution_config,
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
    execution_mode: str = "legacy",
    execution_config: "BacktestExecutionConfig | None" = None,
    entry_offset_days: float | None = None,
    entry_offset_for_features: bool = False,
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
            odds_noise_mode=getattr(spec, "odds_noise_mode", "independent"),
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
            odds_noise_mode=getattr(spec, "odds_noise_mode", "independent"),
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
            entry_offset_days=entry_offset_days,
            entry_offset_for_features=entry_offset_for_features,
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
            execution_mode=execution_mode,
            execution_config=execution_config,
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
    execution_mode: str = "legacy",
    execution_config: "BacktestExecutionConfig | None" = None,
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
        execution_mode=execution_mode,
        execution_config=execution_config,
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
