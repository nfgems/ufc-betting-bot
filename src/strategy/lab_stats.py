"""
Statistical comparison utilities for the model lab.

Provides bootstrap confidence intervals, Sharpe ratio, max drawdown,
ECE, and side-by-side variant comparison tables.
"""

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def compute_sharpe_ratio(bet_log: pd.DataFrame, bets_per_year: float = 300.0) -> float:
    """
    Annualized Sharpe ratio from per-bet P&L.

    Assumes ~300 bets/year (40 UFC events * ~12 fights * ~60% bet rate).
    """
    if bet_log.empty or "profit" not in bet_log.columns:
        return 0.0
    profits = bet_log["profit"].values
    if len(profits) < 2:
        return 0.0
    mean_ret = np.mean(profits)
    std_ret = np.std(profits, ddof=1)
    if std_ret == 0:
        return 0.0
    return (mean_ret / std_ret) * np.sqrt(bets_per_year)


def compute_max_drawdown(bankroll_history: list[float]) -> dict:
    """
    Compute max drawdown magnitude and duration (in number of bets).

    Returns dict with:
        - max_drawdown_pct: peak-to-trough percentage loss
        - max_drawdown_duration: number of bets from peak to recovery (or end)
    """
    if not bankroll_history or len(bankroll_history) < 2:
        return {"max_drawdown_pct": 0.0, "max_drawdown_duration": 0}

    arr = np.array(bankroll_history)
    peak = arr[0]
    max_dd = 0.0
    max_dd_duration = 0
    current_dd_start = 0
    in_drawdown = False

    for i in range(1, len(arr)):
        if arr[i] > peak:
            if in_drawdown:
                duration = i - current_dd_start
                max_dd_duration = max(max_dd_duration, duration)
            peak = arr[i]
            in_drawdown = False
        else:
            dd = (peak - arr[i]) / peak
            if dd > max_dd:
                max_dd = dd
            if not in_drawdown:
                current_dd_start = i
                in_drawdown = True

    # If still in drawdown at end
    if in_drawdown:
        duration = len(arr) - current_dd_start
        max_dd_duration = max(max_dd_duration, duration)

    return {
        "max_drawdown_pct": max_dd,
        "max_drawdown_duration": max_dd_duration,
    }


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error.

    Measures how well predicted probabilities match actual outcomes.
    ECE = 0 is perfectly calibrated.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    if total == 0:
        return 0.0

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_pred = y_prob[mask].mean()
        avg_true = y_true[mask].mean()
        ece += (n_bin / total) * abs(avg_pred - avg_true)

    return ece


def compute_roi(bet_log: pd.DataFrame) -> float:
    """Compute ROI from bet log."""
    if bet_log.empty or "profit" not in bet_log.columns or "bet_size" not in bet_log.columns:
        return 0.0
    total_wagered = bet_log["bet_size"].sum()
    if total_wagered == 0:
        return 0.0
    return bet_log["profit"].sum() / total_wagered


def bootstrap_comparison(
    baseline_log: pd.DataFrame,
    variant_log: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap the difference in a metric between baseline and variant.

    Returns dict with:
        - baseline_metric: metric on full baseline sample
        - variant_metric: metric on full variant sample
        - diff_mean: mean bootstrapped difference (variant - baseline)
        - diff_ci_lo: 2.5th percentile of difference
        - diff_ci_hi: 97.5th percentile of difference
        - p_value: fraction of bootstrap samples where baseline >= variant
    """
    rng = np.random.RandomState(seed)

    baseline_val = metric_fn(baseline_log)
    variant_val = metric_fn(variant_log)

    if baseline_log.empty or variant_log.empty:
        return {
            "baseline_metric": baseline_val,
            "variant_metric": variant_val,
            "diff_mean": variant_val - baseline_val,
            "diff_ci_lo": 0.0,
            "diff_ci_hi": 0.0,
            "p_value": 1.0,
        }

    n_base = len(baseline_log)
    n_var = len(variant_log)
    diffs = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        base_sample = baseline_log.iloc[rng.choice(n_base, size=n_base, replace=True)]
        var_sample = variant_log.iloc[rng.choice(n_var, size=n_var, replace=True)]
        diffs[i] = metric_fn(var_sample) - metric_fn(base_sample)

    return {
        "baseline_metric": baseline_val,
        "variant_metric": variant_val,
        "diff_mean": np.mean(diffs),
        "diff_ci_lo": np.percentile(diffs, 2.5),
        "diff_ci_hi": np.percentile(diffs, 97.5),
        "p_value": np.mean(diffs <= 0),  # P(variant not better than baseline)
    }


def compare_variants(results: dict) -> pd.DataFrame:
    """
    Compare all variant results side-by-side.

    Args:
        results: dict of {variant_name: backtest_result_dict}
            Each result must have: stats, bet_log, bankroll_history

    Returns DataFrame with one row per variant.
    """
    rows = []
    baseline_log = None

    for name, res in results.items():
        stats = res.get("stats", {})
        bet_log = res.get("bet_log", pd.DataFrame())
        bankroll_history = res.get("bankroll_history", [])

        dd = compute_max_drawdown(bankroll_history)
        sharpe = compute_sharpe_ratio(bet_log)

        row = {
            "variant": name,
            "n_bets": stats.get("total_bets", len(bet_log)),
            "win_rate": stats.get("win_rate", 0),
            "roi": stats.get("roi", 0),
            "total_profit": stats.get("total_profit", 0),
            "sharpe": sharpe,
            "max_drawdown_pct": dd["max_drawdown_pct"],
            "max_dd_duration": dd["max_drawdown_duration"],
            "avg_edge": stats.get("avg_edge", 0),
            "avg_clv": stats.get("avg_clv", np.nan),
            "brier": stats.get("brier_score", np.nan),
            "ece": stats.get("ece", np.nan),
        }

        # Bootstrap comparison against baseline
        if name == "baseline" or baseline_log is None:
            baseline_log = bet_log
            row["p_value_vs_baseline"] = np.nan
            row["roi_ci_lo"] = np.nan
            row["roi_ci_hi"] = np.nan
        else:
            boot = bootstrap_comparison(baseline_log, bet_log, compute_roi)
            row["p_value_vs_baseline"] = boot["p_value"]
            row["roi_ci_lo"] = boot["diff_ci_lo"]
            row["roi_ci_hi"] = boot["diff_ci_hi"]

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("roi", ascending=False).reset_index(drop=True)
    return df


def plot_comparison(results: dict, save_path: Optional[str] = None) -> None:
    """Generate side-by-side comparison plots for all variants."""
    n_variants = len(results)
    if n_variants == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Bankroll curves
    ax = axes[0, 0]
    for name, res in results.items():
        bh = res.get("bankroll_history", [])
        if bh:
            ax.plot(bh, label=name, linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Bet #")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("Bankroll Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. ROI bar chart
    ax = axes[0, 1]
    names = list(results.keys())
    rois = [results[n].get("stats", {}).get("roi", 0) for n in names]
    colors = ["green" if r > 0 else "red" for r in rois]
    ax.bar(range(len(names)), [r * 100 for r in rois], color=colors, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ROI (%)")
    ax.set_title("ROI by Variant")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # 3. Win rate comparison
    ax = axes[1, 0]
    win_rates = [results[n].get("stats", {}).get("win_rate", 0) for n in names]
    ax.bar(range(len(names)), [w * 100 for w in win_rates], color="steelblue", alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rate by Variant")
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Sharpe ratio comparison
    ax = axes[1, 1]
    sharpes = []
    for n in names:
        bet_log = results[n].get("bet_log", pd.DataFrame())
        sharpes.append(compute_sharpe_ratio(bet_log))
    colors = ["green" if s > 0 else "red" for s in sharpes]
    ax.bar(range(len(names)), sharpes, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio by Variant")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Model Lab: Variant Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved comparison plots to {save_path}")
    else:
        plt.show()
