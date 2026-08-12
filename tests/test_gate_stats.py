"""E7: paired event bootstrap + snapshot-distinct CLV gate evidence."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.gate_stats import clv_fair_stats, paired_event_bootstrap
from src.strategy.promotion_gate import PromotionGate


def _log(rows):
    return pd.DataFrame(rows)


def _bet(date, fa, fb, profit, size=10.0, clv=np.nan):
    return {
        "event_date": pd.Timestamp(date),
        "fighter_a": fa,
        "fighter_b": fb,
        "bet_size": size,
        "profit": profit,
        "clv": clv,
    }


def test_paired_bootstrap_separates_clear_winner():
    dates = pd.date_range("2024-01-06", periods=30, freq="7D")
    cand = _log([_bet(d, f"A{i}", f"B{i}", 8.0) for i, d in enumerate(dates)])
    base = _log([_bet(d, f"A{i}", f"B{i}", 1.0) for i, d in enumerate(dates)])
    boot = paired_event_bootstrap(cand, base, n_bootstrap=500)
    assert boot["profit_diff"] == 30 * 7.0
    assert boot["p_value"] < 0.01
    assert boot["n_events"] == 30


def test_paired_bootstrap_cannot_separate_identical_arms():
    dates = pd.date_range("2024-01-06", periods=20, freq="7D")
    profits = [(-5.0 if i % 2 else 8.0) for i in range(20)]
    cand = _log([_bet(d, f"A{i}", f"B{i}", p) for i, (d, p) in enumerate(zip(dates, profits))])
    base = cand.copy()
    boot = paired_event_bootstrap(cand, base, n_bootstrap=500)
    assert boot["profit_diff"] == 0.0
    assert boot["p_value"] >= 0.5  # diff is exactly zero in every resample


def test_paired_bootstrap_counts_absent_arm_as_zero():
    cand = _log([_bet("2024-01-06", "A", "B", 10.0)])
    base = _log([_bet("2024-02-03", "C", "D", 10.0)])
    boot = paired_event_bootstrap(cand, base, n_bootstrap=200)
    assert boot["n_events"] == 2
    assert boot["profit_diff"] == 0.0


def test_clv_fair_excludes_single_snapshot_fights():
    hist = pd.DataFrame([
        # fight 1: two distinct snapshots
        {"event_date": "2024-01-06", "fighter_a": "A1", "fighter_b": "B1", "offset_days": 7},
        {"event_date": "2024-01-06", "fighter_a": "A1", "fighter_b": "B1", "offset_days": 1},
        # fight 2: single snapshot — CLV structurally zero
        {"event_date": "2024-01-06", "fighter_a": "A2", "fighter_b": "B2", "offset_days": 7},
    ])
    log = _log([
        _bet("2024-01-06", "A1", "B1", 5.0, clv=0.04),
        _bet("2024-01-06", "A2", "B2", 5.0, clv=0.0),
    ])
    stats = clv_fair_stats(log, hist=hist, min_distinct=1)
    assert stats["n_bets_distinct"] == 1
    assert stats["n_bets_total"] == 2
    assert abs(stats["avg_clv_fair"] - 0.04) < 1e-12
    assert stats["sufficient"]


def test_trading_bar_statistical_blocks_noise_point_wins():
    """A tiny point-estimate win with sign-flipping events must not promote."""
    dates = pd.date_range("2024-01-06", periods=24, freq="7D")
    rng = np.random.RandomState(3)
    noise = rng.choice([-10.0, 10.0], size=24)
    cand_log = _log([_bet(d, f"A{i}", f"B{i}", p) for i, (d, p) in enumerate(zip(dates, noise))])
    base_log = _log([_bet(d, f"A{i}", f"B{i}", -p) for i, (d, p) in enumerate(zip(dates, noise))])
    # Give candidate a hair more profit so the point estimate "wins".
    cand_log.loc[0, "profit"] += 1.0

    candidate = {
        "roi": 0.05, "total_profit": float(cand_log["profit"].sum()),
        "max_drawdown_pct": 0.10, "bet_log": cand_log,
    }
    baseline = {
        "roi": 0.049, "total_profit": float(base_log["profit"].sum()),
        "max_drawdown_pct": 0.10, "bet_log": base_log,
    }

    statistical = PromotionGate.check_trading_bar(candidate, baseline)
    assert not statistical["passes"]
    with pytest.raises(TypeError, match="unexpected keyword argument 'statistical'"):
        PromotionGate.check_trading_bar(candidate, baseline, statistical=False)


def test_trading_bar_statistical_passes_consistent_winner():
    dates = pd.date_range("2024-01-06", periods=30, freq="7D")
    cand_rows = []
    base_rows = []
    for i, d in enumerate(dates):
        row = _bet(d, f"A{i}", f"B{i}", 6.0, size=100.0)
        row["edge"] = 0.07
        cand_rows.append(row)
        row_b = _bet(d, f"A{i}", f"B{i}", 1.0, size=100.0)
        row_b["edge"] = 0.02
        base_rows.append(row_b)
    cand_log = _log(cand_rows)
    base_log = _log(base_rows)
    candidate = {
        "roi": 0.06, "total_profit": 180.0, "max_drawdown_pct": 0.08,
        "avg_clv": 0.02, "bet_log": cand_log,
    }
    baseline = {
        "roi": 0.01, "total_profit": 30.0, "max_drawdown_pct": 0.08,
        "avg_clv": 0.01, "bet_log": base_log,
    }
    result = PromotionGate.check_trading_bar(candidate, baseline)
    assert result["passes"], result["reasons"]
