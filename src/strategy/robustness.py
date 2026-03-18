"""
Robustness checks for betting strategy backtests.

Validates that backtest results are not driven by a handful of lucky events,
that edge claims are consistent with realized profits, and that Trader C
behaviour stays within acceptable bounds.

Usage:
    from src.strategy.robustness import (
        year_by_year_analysis,
        leave_one_event_out,
        edge_vs_profit_check,
        volume_sanity_check,
        trader_c_behavior_audit,
        top_event_analysis,
    )
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import INITIAL_BANKROLL, MIN_EDGE_THRESHOLD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_max_drawdown_from_bankroll(bankroll_series: pd.Series) -> float:
    """Peak-to-trough drawdown from a running bankroll series."""
    if bankroll_series.empty:
        return 0.0
    peak = bankroll_series.expanding().max()
    drawdown = (peak - bankroll_series) / peak
    return float(drawdown.max()) if not drawdown.empty else 0.0


def _running_bankroll(bet_log: pd.DataFrame, initial: float = INITIAL_BANKROLL) -> pd.Series:
    """Build a running bankroll series from chronological bet log profits."""
    if bet_log.empty:
        return pd.Series([initial], dtype=float)
    ordered = bet_log.copy()
    if "event_date" in ordered.columns:
        ordered["_event_date_sort"] = pd.to_datetime(
            ordered["event_date"], errors="coerce"
        )
        ordered = ordered.sort_values("_event_date_sort").reset_index(drop=True)
        ordered = ordered.drop(columns="_event_date_sort")
    else:
        ordered = ordered.reset_index(drop=True)
    if "bankroll_after" in ordered.columns and ordered["bankroll_after"].notna().all():
        return ordered["bankroll_after"].astype(float)
    if "profit" not in ordered.columns:
        return pd.Series([initial], dtype=float)
    return initial + ordered["profit"].cumsum()


def _safe_year(row) -> int:
    """Extract year from an event_date value regardless of type."""
    val = row.get("event_date")
    if val is None:
        return 0
    if isinstance(val, str):
        try:
            return int(val[:4])
        except (ValueError, IndexError):
            return 0
    try:
        return int(pd.Timestamp(val).year)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# 1. Year-by-year analysis
# ---------------------------------------------------------------------------

def year_by_year_analysis(bet_log: pd.DataFrame) -> pd.DataFrame:
    """
    Break down performance by calendar year.

    For each year returns: ROI, win_rate, total_bets, avg_edge, avg_clv,
    max_drawdown, and a flag when the year's ROI diverges significantly
    from the overall ROI.

    Parameters
    ----------
    bet_log : pd.DataFrame
        Must contain at minimum: event_date, profit, bet_size, won (bool).
        Optional: edge, clv.

    Returns
    -------
    pd.DataFrame  One row per year plus an ``outlier`` flag column.
    """
    if bet_log.empty:
        return pd.DataFrame()

    df = bet_log.copy()
    df["year"] = df.apply(_safe_year, axis=1)
    df = df[df["year"] > 0]

    if df.empty:
        return pd.DataFrame()

    df["_event_date_sort"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.sort_values("_event_date_sort").reset_index(drop=True)
    df = df.drop(columns="_event_date_sort")
    overall_wagered = df["bet_size"].sum() if "bet_size" in df.columns else 1.0
    overall_roi = df["profit"].sum() / overall_wagered if overall_wagered > 0 else 0.0
    full_bankroll = _running_bankroll(df)
    if (
        "bankroll_after" in df.columns
        and df["bankroll_after"].notna().all()
        and "profit" in df.columns
        and not df["profit"].empty
    ):
        initial_bankroll = float(df["bankroll_after"].iloc[0] - df["profit"].iloc[0])
    else:
        initial_bankroll = float(INITIAL_BANKROLL)
    df["_running_bankroll"] = full_bankroll.reset_index(drop=True)

    rows = []
    for year, grp in df.groupby("year"):
        wagered = grp["bet_size"].sum() if "bet_size" in grp.columns else 0.0
        profit = grp["profit"].sum()
        roi = profit / wagered if wagered > 0 else 0.0
        n_bets = len(grp)

        wins = grp["won"].sum() if "won" in grp.columns else 0
        win_rate = wins / n_bets if n_bets > 0 else 0.0

        avg_edge = grp["edge"].mean() if "edge" in grp.columns else np.nan
        avg_clv = grp["clv"].mean() if "clv" in grp.columns else np.nan

        start_idx = int(grp.index.min())
        year_start_bankroll = (
            initial_bankroll
            if start_idx == 0
            else float(df.loc[start_idx - 1, "_running_bankroll"])
        )
        bankroll = pd.concat(
            [
                pd.Series([year_start_bankroll], dtype=float),
                grp["_running_bankroll"].astype(float).reset_index(drop=True),
            ],
            ignore_index=True,
        )
        max_dd = _compute_max_drawdown_from_bankroll(bankroll)

        rows.append({
            "year": int(year),
            "total_bets": n_bets,
            "wins": int(wins),
            "win_rate": win_rate,
            "wagered": wagered,
            "profit": profit,
            "roi": roi,
            "avg_edge": avg_edge,
            "avg_clv": avg_clv,
            "max_drawdown": max_dd,
        })

    result = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

    # Flag years where ROI diverges > 10pp from overall
    result["roi_divergence"] = result["roi"] - overall_roi
    result["outlier"] = result["roi_divergence"].abs() > 0.10
    result["overall_roi"] = overall_roi

    return result


# ---------------------------------------------------------------------------
# 2. Leave-one-event-out
# ---------------------------------------------------------------------------

def leave_one_event_out(bet_log: pd.DataFrame) -> list[dict]:
    """
    For each unique event date, remove all bets from that event and
    recompute ROI.  Returns a list of "outlier events" whose removal
    shifts ROI by more than 5 percentage points.

    Parameters
    ----------
    bet_log : pd.DataFrame
        Must contain: event_date, profit, bet_size.

    Returns
    -------
    list[dict]  Each dict: event_date, n_bets, event_profit,
                roi_without, roi_full, roi_shift.
    """
    if bet_log.empty or "profit" not in bet_log.columns:
        return []

    df = bet_log.copy()
    total_wagered = df["bet_size"].sum() if "bet_size" in df.columns else 1.0
    total_profit = df["profit"].sum()
    full_roi = total_profit / total_wagered if total_wagered > 0 else 0.0

    outliers: list[dict] = []

    for event_date, grp in df.groupby("event_date"):
        remaining = df[df["event_date"] != event_date]
        rem_wagered = remaining["bet_size"].sum() if "bet_size" in remaining.columns else 0.0
        rem_profit = remaining["profit"].sum()
        rem_roi = rem_profit / rem_wagered if rem_wagered > 0 else 0.0

        roi_shift = rem_roi - full_roi

        if abs(roi_shift) > 0.05:
            outliers.append({
                "event_date": event_date,
                "n_bets": len(grp),
                "event_profit": grp["profit"].sum(),
                "event_wagered": grp["bet_size"].sum() if "bet_size" in grp.columns else 0.0,
                "roi_without": rem_roi,
                "roi_full": full_roi,
                "roi_shift": roi_shift,
            })

    # Sort by absolute magnitude of shift descending
    outliers.sort(key=lambda x: abs(x["roi_shift"]), reverse=True)
    return outliers


# ---------------------------------------------------------------------------
# 3. Edge vs profit check
# ---------------------------------------------------------------------------

def edge_vs_profit_check(bet_log: pd.DataFrame) -> dict:
    """
    Compare the average *claimed* edge against *realized* profit rate.

    - If realized >> claimed: likely luck or data leakage.
    - If realized << claimed: execution drag or wrong calibration.

    Returns
    -------
    dict  Keys: avg_claimed_edge, realized_profit_rate, gap, verdict.
    """
    if bet_log.empty or "profit" not in bet_log.columns:
        return {
            "avg_claimed_edge": 0.0,
            "realized_profit_rate": 0.0,
            "gap": 0.0,
            "edge_basis": "none",
            "verdict": "INSUFFICIENT_DATA",
        }

    scoped_log = bet_log
    edge_col = "edge"
    edge_basis = "edge"

    if "blend_edge" in bet_log.columns:
        if "trader" in bet_log.columns:
            s_blend_log = bet_log[
                (bet_log["trader"] == "S") & bet_log["blend_edge"].notna()
            ]
            if not s_blend_log.empty:
                scoped_log = s_blend_log
                edge_col = "blend_edge"
                edge_basis = "blend_edge_s"
        elif bet_log["blend_edge"].notna().any():
            scoped_log = bet_log[bet_log["blend_edge"].notna()]
            edge_col = "blend_edge"
            edge_basis = "blend_edge"

    avg_edge = scoped_log[edge_col].mean() if edge_col in scoped_log.columns else 0.0
    wagered = scoped_log["bet_size"].sum() if "bet_size" in scoped_log.columns else 1.0
    profit = scoped_log["profit"].sum()
    realized = profit / wagered if wagered > 0 else 0.0

    gap = realized - avg_edge

    # Use a Z-score test when sample is large enough.  Kelly bet-sizing
    # naturally makes high-edge bets larger, so comparing realized profit/wagered
    # to the *simple* average claimed edge creates a systematic upward bias.
    # A Z-score of the per-bet profit rate properly accounts for this variance.
    z_score: float | None = None
    n_bets = len(scoped_log)
    if (
        n_bets >= 30
        and "bet_size" in scoped_log.columns
        and scoped_log["bet_size"].sum() > 0
    ):
        profit_rates = scoped_log["profit"] / scoped_log["bet_size"]
        std_rate = float(profit_rates.std())
        se = std_rate / np.sqrt(n_bets) if std_rate > 0 else None
        if se is not None and se > 0:
            z_score = (realized - avg_edge) / se

    if z_score is not None:
        if z_score > 3.0:
            verdict = "SUSPICIOUSLY_HIGH"
        elif z_score < -3.0:
            verdict = "EXECUTION_DRAG"
        else:
            verdict = "CONSISTENT"
    elif gap > 0.05:
        verdict = "SUSPICIOUSLY_HIGH"
    elif gap < -0.05:
        verdict = "EXECUTION_DRAG"
    else:
        verdict = "CONSISTENT"

    return {
        "avg_claimed_edge": avg_edge,
        "realized_profit_rate": realized,
        "gap": gap,
        "z_score": z_score,
        "edge_basis": edge_basis,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 4. Volume sanity check
# ---------------------------------------------------------------------------

def volume_sanity_check(
    candidate_log: pd.DataFrame,
    baseline_log: pd.DataFrame,
) -> dict:
    """
    Compare candidate bet count and wagered volume against a baseline.

    Flags:
    - ``excess_bets``: candidate places >50 % more bets than baseline.
    - ``trader_c_overexposure``: Trader C accounts for >40 % of total volume.

    Parameters
    ----------
    candidate_log : pd.DataFrame
        Must have: bet_size, and optionally a ``trader`` column ('S' or 'C').
    baseline_log : pd.DataFrame
        Same schema as candidate_log.

    Returns
    -------
    dict  Summary with flags.
    """
    c_bets = len(candidate_log) if not candidate_log.empty else 0
    b_bets = len(baseline_log) if not baseline_log.empty else 1

    c_wagered = candidate_log["bet_size"].sum() if (not candidate_log.empty and "bet_size" in candidate_log.columns) else 0.0
    b_wagered = baseline_log["bet_size"].sum() if (not baseline_log.empty and "bet_size" in baseline_log.columns) else 1.0

    bet_ratio = c_bets / b_bets if b_bets > 0 else float("inf")

    # Trader C share of candidate volume
    trader_c_volume = 0.0
    trader_c_bets = 0
    if not candidate_log.empty and "trader" in candidate_log.columns:
        c_mask = candidate_log["trader"] == "C"
        trader_c_volume = candidate_log.loc[c_mask, "bet_size"].sum() if "bet_size" in candidate_log.columns else 0.0
        trader_c_bets = int(c_mask.sum())

    c_volume_share = trader_c_volume / c_wagered if c_wagered > 0 else 0.0

    flags: list[str] = []
    if bet_ratio > 1.50:
        flags.append(
            f"EXCESS_BETS: candidate places {c_bets} bets vs baseline {b_bets} "
            f"({bet_ratio:.1%} ratio, threshold 150%)"
        )
    if c_volume_share > 0.40:
        flags.append(
            f"TRADER_C_OVEREXPOSURE: C accounts for {c_volume_share:.1%} of "
            f"candidate volume (threshold 40%)"
        )

    return {
        "candidate_bets": c_bets,
        "baseline_bets": b_bets,
        "bet_ratio": bet_ratio,
        "candidate_wagered": c_wagered,
        "baseline_wagered": b_wagered,
        "trader_c_bets": trader_c_bets,
        "trader_c_volume": trader_c_volume,
        "trader_c_volume_share": c_volume_share,
        "flags": flags,
        "passes": len(flags) == 0,
    }


# ---------------------------------------------------------------------------
# 5. Trader C behaviour audit
# ---------------------------------------------------------------------------

def trader_c_behavior_audit(
    bet_log: pd.DataFrame,
) -> dict:
    """
    Audit Trader C's behaviour for warning signs.

    Checks:
    - Bet concentration on underdogs (market_prob < 0.40).
    - Over-betting marginal underdogs (odds > 3.0 with model_prob barely
      above the conviction threshold).
    - Win-rate comparison between S and C.

    Parameters
    ----------
    bet_log : pd.DataFrame
        Must contain: trader ('S'/'C'), market_prob, won, and optionally
        model_prob, odds/decimal_odds.

    Returns
    -------
    dict  Behaviour report with flags.
    """
    if bet_log.empty or "trader" not in bet_log.columns:
        return {"verdict": "NO_TRADER_DATA", "flags": []}

    s_bets = bet_log[bet_log["trader"] == "S"]
    c_bets = bet_log[bet_log["trader"] == "C"]

    if c_bets.empty:
        return {"verdict": "NO_C_BETS", "flags": []}

    flags: list[str] = []

    # --- Underdog concentration ---
    market_prob_col = "market_prob" if "market_prob" in c_bets.columns else None
    if market_prob_col:
        underdog_frac = (c_bets[market_prob_col] < 0.40).mean()
    else:
        underdog_frac = np.nan

    # --- Marginal underdog over-betting ---
    odds_col = "decimal_odds" if "decimal_odds" in c_bets.columns else ("odds" if "odds" in c_bets.columns else None)
    model_prob_col = "model_prob" if "model_prob" in c_bets.columns else ("blended_prob" if "blended_prob" in c_bets.columns else None)

    marginal_count = 0
    if odds_col and model_prob_col:
        marginal_mask = (c_bets[odds_col] > 3.0) & (c_bets[model_prob_col] < 0.70)
        marginal_count = int(marginal_mask.sum())
        if marginal_count > 0:
            flags.append(
                f"MARGINAL_UNDERDOGS: {marginal_count} C bets on odds>3.0 "
                f"with model_prob<0.70"
            )

    # --- Win rate comparison ---
    s_win_rate = s_bets["won"].mean() if (not s_bets.empty and "won" in s_bets.columns) else np.nan
    c_win_rate = c_bets["won"].mean() if "won" in c_bets.columns else np.nan

    if not np.isnan(s_win_rate) and not np.isnan(c_win_rate):
        if c_win_rate < s_win_rate - 0.10:
            flags.append(
                f"LOW_C_WIN_RATE: C win rate {c_win_rate:.1%} is >10pp below "
                f"S win rate {s_win_rate:.1%}"
            )

    # --- Underdog concentration flag ---
    if not np.isnan(underdog_frac) and underdog_frac > 0.50:
        flags.append(
            f"UNDERDOG_HEAVY: {underdog_frac:.0%} of C bets are on underdogs "
            f"(market_prob < 40%)"
        )

    return {
        "s_bets": len(s_bets),
        "c_bets": len(c_bets),
        "s_win_rate": float(s_win_rate) if not np.isnan(s_win_rate) else None,
        "c_win_rate": float(c_win_rate) if not np.isnan(c_win_rate) else None,
        "underdog_fraction": float(underdog_frac) if not np.isnan(underdog_frac) else None,
        "marginal_underdog_bets": marginal_count,
        "flags": flags,
        "passes": len(flags) == 0,
    }


# ---------------------------------------------------------------------------
# 6. Top event analysis
# ---------------------------------------------------------------------------

def top_event_analysis(
    bet_log: pd.DataFrame,
    n_top: int = 5,
) -> pd.DataFrame:
    """
    Identify the top winning and top losing events.

    Parameters
    ----------
    bet_log : pd.DataFrame
        Must contain: event_date, profit.  Optionally: bet_on, fighter_a,
        fighter_b, bet_size.
    n_top : int
        Number of events on each side (winners/losers).

    Returns
    -------
    pd.DataFrame  Up to ``2 * n_top`` rows with event-level summaries.
    """
    if bet_log.empty or "profit" not in bet_log.columns:
        return pd.DataFrame()

    df = bet_log.copy()

    agg: dict[str, tuple] = {
        "profit": ("profit", "sum"),
        "n_bets": ("profit", "count"),
    }
    if "bet_size" in df.columns:
        agg["wagered"] = ("bet_size", "sum")

    event_stats = df.groupby("event_date").agg(**agg).reset_index()

    # Collect fighter names per event
    if "bet_on" in df.columns:
        fighters_by_event = (
            df.groupby("event_date")["bet_on"]
            .apply(lambda x: ", ".join(x.astype(str).unique()))
            .reset_index()
            .rename(columns={"bet_on": "fighters"})
        )
        event_stats = event_stats.merge(fighters_by_event, on="event_date", how="left")

    event_stats = event_stats.sort_values("profit", ascending=False).reset_index(drop=True)

    top_win = event_stats.head(n_top).copy()
    top_win["category"] = "TOP_WIN"
    top_lose = event_stats.tail(n_top).sort_values("profit").copy()
    top_lose["category"] = "TOP_LOSS"

    result = pd.concat([top_win, top_lose], ignore_index=True)
    return result
