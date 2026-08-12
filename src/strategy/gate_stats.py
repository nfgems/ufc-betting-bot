"""
Statistically honest promotion-gate evidence (E7).

The legacy trading bar compares raw ROI/profit point estimates at n~250 bets,
where the ROI 95% CI spans roughly +/-15pp — a coin flip dressed as a verdict.
This module provides:

- paired_event_bootstrap: candidate-minus-baseline P&L difference resampled
  by EVENT (cards are the independent unit — bets within a card share
  bankroll state and correlated outcomes). Fights where one arm didn't bet
  contribute zero for that arm, which is exactly the deployment difference.
- clv_fair_stats: CLV restricted to fights whose historical odds have at
  least two distinct snapshots. Single-snapshot fights have opening ==
  closing by construction, so their CLV is structurally zero and dilutes the
  average toward 0 (82.6% of matched fights in the 2022+ window).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FIGHT_KEY_COLS = ["event_date", "fighter_a", "fighter_b"]


def _event_key(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["event_date"]).dt.normalize().astype(str)


def has_pairing_columns(bet_log: pd.DataFrame | None) -> bool:
    """Whether a bet log carries the columns the paired statistics need."""
    return (
        isinstance(bet_log, pd.DataFrame)
        and not bet_log.empty
        and {"event_date", "profit", "bet_size"}.issubset(bet_log.columns)
    )


def _per_event_pnl(bet_log: pd.DataFrame) -> pd.Series:
    """Total bet P&L per event date (empty log -> empty series)."""
    if (
        bet_log is None or bet_log.empty
        or "profit" not in bet_log.columns or "event_date" not in bet_log.columns
    ):
        return pd.Series(dtype=float)
    frame = bet_log.copy()
    frame["__event"] = _event_key(frame)
    return frame.groupby("__event")["profit"].sum()


def _per_event_stake(bet_log: pd.DataFrame) -> pd.Series:
    if (
        bet_log is None or bet_log.empty
        or "bet_size" not in bet_log.columns or "event_date" not in bet_log.columns
    ):
        return pd.Series(dtype=float)
    frame = bet_log.copy()
    frame["__event"] = _event_key(frame)
    return frame.groupby("__event")["bet_size"].sum()


def paired_event_bootstrap(
    candidate_log: pd.DataFrame,
    baseline_log: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Event-clustered paired bootstrap of the candidate-minus-baseline P&L.

    Events where only one arm bet contribute zero for the silent arm — that
    asymmetry IS part of the strategy difference and must not be dropped.

    Returns observed profit/ROI diffs, the bootstrap CI for the profit diff,
    and p_value = P(candidate total P&L <= baseline) under resampling.
    """
    cand = _per_event_pnl(candidate_log)
    base = _per_event_pnl(baseline_log)
    cand_stake = _per_event_stake(candidate_log)
    base_stake = _per_event_stake(baseline_log)

    events = sorted(set(cand.index) | set(base.index))
    if not events:
        return {
            "n_events": 0,
            "profit_diff": 0.0,
            "roi_diff": 0.0,
            "profit_diff_ci_lo": 0.0,
            "profit_diff_ci_hi": 0.0,
            "p_value": 1.0,
        }

    cand_pnl = np.array([cand.get(e, 0.0) for e in events], dtype=float)
    base_pnl = np.array([base.get(e, 0.0) for e in events], dtype=float)
    cand_stk = np.array([cand_stake.get(e, 0.0) for e in events], dtype=float)
    base_stk = np.array([base_stake.get(e, 0.0) for e in events], dtype=float)

    observed_profit_diff = float(cand_pnl.sum() - base_pnl.sum())
    cand_roi = cand_pnl.sum() / cand_stk.sum() if cand_stk.sum() > 0 else 0.0
    base_roi = base_pnl.sum() / base_stk.sum() if base_stk.sum() > 0 else 0.0

    rng = np.random.RandomState(seed)
    n = len(events)
    diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diffs[i] = cand_pnl[idx].sum() - base_pnl[idx].sum()

    return {
        "n_events": n,
        "profit_diff": observed_profit_diff,
        "roi_diff": float(cand_roi - base_roi),
        "profit_diff_ci_lo": float(np.percentile(diffs, 2.5)),
        "profit_diff_ci_hi": float(np.percentile(diffs, 97.5)),
        "p_value": float(np.mean(diffs <= 0.0)),
    }


def _distinct_snapshot_fights(hist: pd.DataFrame | None = None) -> set[tuple[str, str, str]]:
    """Fight keys whose historical odds contain >=2 distinct snapshots."""
    if hist is None:
        from src.data.historical_backfill import load_all_historical_odds

        hist = load_all_historical_odds()
    if hist is None or hist.empty:
        return set()
    frame = hist.copy()
    if "verified_prefight" in frame.columns:
        frame = frame[frame["verified_prefight"].fillna(False).astype(bool)]
    if frame.empty:
        return set()
    frame["__date"] = pd.to_datetime(frame["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    snapshot_column = "observed_at" if "observed_at" in frame.columns else "offset_days"
    grouped = frame.groupby(["__date", "fighter_a", "fighter_b"])[snapshot_column].nunique()
    distinct = grouped[grouped >= 2]
    return {
        (str(event_date), str(fa), str(fb))
        for (event_date, fa, fb) in distinct.index
    }


def clv_fair_stats(
    bet_log: pd.DataFrame,
    hist: pd.DataFrame | None = None,
    min_distinct: int = 80,
) -> dict:
    """
    Average CLV over bets on fights with >=2 distinct odds snapshots.

    Single-snapshot fights structurally have CLV == 0 (entry IS closing) and
    must be excluded rather than averaged in as zeros.
    """
    required = {"clv", "event_date", "fighter_a", "fighter_b"}
    if bet_log is None or bet_log.empty or not required.issubset(bet_log.columns):
        return {"avg_clv_fair": np.nan, "n_bets_distinct": 0, "n_bets_total": 0, "sufficient": False}

    frame = bet_log.copy()
    frame["__date"] = pd.to_datetime(frame["event_date"]).dt.strftime("%Y-%m-%d")
    distinct = _distinct_snapshot_fights(hist)
    keys = list(zip(frame["__date"], frame["fighter_a"].astype(str), frame["fighter_b"].astype(str)))
    mask = pd.Series([k in distinct for k in keys], index=frame.index)
    fair = frame.loc[mask, "clv"].dropna()
    return {
        "avg_clv_fair": float(fair.mean()) if len(fair) else np.nan,
        "n_bets_distinct": int(len(fair)),
        "n_bets_total": int(len(frame)),
        "sufficient": bool(len(fair) >= min_distinct),
    }
