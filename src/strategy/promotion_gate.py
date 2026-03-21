"""
Promotion Gate --- decides whether a candidate model/strategy should replace
the current production system.

Two independent bars must be cleared:

  **Model Bar**   -- better calibration (Brier or ECE), no major regression,
                     survives fresh-data evaluation.
  **Trading Bar** -- better ROI or profit, acceptable drawdown, CLV not worse,
                     sane behaviour (robustness checks pass).

Verdicts:
  PROMOTE      -- clears both bars
  SHADOW_TEST  -- clears exactly one bar
  REJECT       -- clears neither

Usage:
    python -m src.strategy.promotion_gate \\
        --candidate-metrics candidate_stats.json \\
        --control-metrics  baseline_stats.json \\
        --candidate-sweep candidate_sweep.json \\
        --baseline-sweep  baseline_sweep.json \\
        --candidate-log    candidate_bet_log.csv \\
        --baseline-log     baseline_bet_log.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import INITIAL_BANKROLL, MIN_EDGE_THRESHOLD
from src.strategy.robustness import (
    year_by_year_analysis,
    leave_one_event_out,
    edge_vs_profit_check,
    volume_sanity_check,
    trader_c_behavior_audit,
    top_event_analysis,
)

logger = logging.getLogger(__name__)

# Tolerances ------------------------------------------------------------------
_MODEL_DEGRADATION_THRESHOLD = 0.10   # >10 % worse = material degradation
_ECE_ABS_REGRESSION_THRESHOLD = 0.025 # absolute ECE regression guard (2.5 ppts)
# Year-by-year ECE failures require BOTH relative >10 % AND absolute >2.5 ppts.
# Purely relative thresholds fire spuriously when the baseline ECE is already
# very low (e.g. 0.053): a 10 % relative slack is only 0.005 absolute, which
# is within measurement noise for year-slice calibration.
_DRAWDOWN_MULTIPLIER = 2.0            # reject if drawdown >2x baseline
_CLV_TOLERANCE = 0.01                 # 1 percentage point
_FRESH_DATA_CUTOFF = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")  # rolling 6-month window


def _relative_degradation(candidate_value: float | None, baseline_value: float | None) -> float | None:
    if candidate_value is None or baseline_value is None:
        return None
    if baseline_value == 0:
        return None
    return (candidate_value - baseline_value) / baseline_value


def _looks_like_sweep_summary(payload: dict) -> bool:
    """Return True when a dict already looks like a sweep result row."""
    return any(
        key in payload
        for key in (
            "combined",
            "profit",
            "total_profit",
            "max_dd",
            "max_drawdown",
            "max_drawdown_pct",
            "roi",
            "config",
        )
    )


def _is_baseline_entry(name: str, payload: dict) -> bool:
    """Identify explicit baseline rows inside sweep artifacts."""
    if payload.get("is_baseline") is True:
        return True
    config_name = str(payload.get("config", "")).lower()
    entry_name = name.lower()
    return config_name == "production_baseline" or entry_name == "production_baseline"


def _select_sweep_payload(payload: dict, role: str) -> dict:
    """Pick one concrete sweep row/result from a larger artifact payload."""
    if "combined" in payload or _looks_like_sweep_summary(payload):
        return payload

    if "baseline" in payload or "top5" in payload:
        if role == "baseline":
            baseline = payload.get("baseline")
            if isinstance(baseline, dict):
                return baseline
            raise ValueError("Sweep artifact is missing a baseline section.")
        top5 = payload.get("top5")
        if isinstance(top5, list) and top5:
            return top5[0]
        raise ValueError("Sweep artifact is missing candidate rows in top5.")

    rows = payload.get("rows")
    if isinstance(rows, list):
        if role == "baseline":
            for row in rows:
                if isinstance(row, dict) and row.get("is_baseline"):
                    return row
            raise ValueError("Sweep artifact rows do not contain a baseline entry.")
        for row in rows:
            if isinstance(row, dict) and not row.get("is_baseline"):
                return row
        raise ValueError("Sweep artifact rows do not contain a candidate entry.")

    dict_entries = [
        (name, value)
        for name, value in payload.items()
        if isinstance(value, dict)
    ]
    if dict_entries:
        if role == "baseline":
            for name, entry in dict_entries:
                if _is_baseline_entry(name, entry):
                    return entry
            raise ValueError("Could not identify a baseline entry in sweep artifact.")
        for name, entry in dict_entries:
            if not _is_baseline_entry(name, entry):
                return entry
        raise ValueError("Could not identify a candidate entry in sweep artifact.")

    raise ValueError("Unrecognized sweep artifact shape.")


def _canonicalize_sweep_payload(payload: dict, source_path: str | Path) -> dict:
    """Normalize a sweep row/result to the trading-bar key schema."""
    combined = payload.get("combined") or {}
    if combined and not isinstance(combined, dict):
        raise ValueError(f"Invalid combined sweep payload in {source_path}")

    canonical = dict(payload)
    canonical["roi"] = payload.get("roi", combined.get("roi"))
    canonical["total_profit"] = payload.get(
        "total_profit",
        payload.get("profit", combined.get("total_profit", combined.get("profit"))),
    )
    canonical["profit"] = canonical.get("total_profit")
    canonical["avg_clv"] = payload.get("avg_clv", combined.get("avg_clv"))
    canonical["max_drawdown"] = payload.get(
        "max_drawdown",
        payload.get(
            "max_drawdown_pct",
            payload.get(
                "max_dd",
                combined.get(
                    "max_drawdown",
                    combined.get("max_drawdown_pct", combined.get("max_dd")),
                ),
            ),
        ),
    )
    canonical["max_drawdown_pct"] = canonical.get("max_drawdown")
    canonical["max_dd"] = canonical.get("max_drawdown")
    canonical["total_bets"] = payload.get("total_bets", combined.get("total_bets"))
    canonical["total_wagered"] = payload.get(
        "total_wagered",
        combined.get("total_wagered"),
    )
    canonical["final_bankroll"] = payload.get(
        "final_bankroll",
        combined.get("final_bankroll"),
    )
    canonical["artifact_source"] = str(source_path)

    missing = [
        key for key in ("roi", "total_profit", "max_drawdown")
        if canonical.get(key) is None
    ]
    if missing:
        raise ValueError(
            f"Sweep artifact {source_path} is missing trading keys: {missing}"
        )
    return canonical


def _load_sweep_artifact(path: str | Path, role: str) -> dict:
    """Load and normalize one sweep artifact for trading-bar evaluation."""
    artifact_path = Path(path)
    with artifact_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Sweep artifact {artifact_path} must contain a JSON object.")
    selected = _select_sweep_payload(payload, role=role)
    return _canonicalize_sweep_payload(selected, artifact_path)


# =============================================================================
# Promotion Gate
# =============================================================================

class PromotionGate:
    """
    Two-bar promotion gate for candidate model + strategy evaluation.
    """

    # -----------------------------------------------------------------
    # Model Bar
    # -----------------------------------------------------------------
    @staticmethod
    def check_model_bar(
        candidate_metrics: dict,
        control_metrics: dict,
    ) -> dict:
        """
        Evaluate the Model Bar.

        Passes when:
        1. Better Brier score OR ECE than control (at least one).
        2. No material degradation (>10 %) in the other core metric.
        3. Survives strict fresh-data evaluation (post-2024-12-14).
        4. No major regression in any year-by-year slice.

        Parameters
        ----------
        candidate_metrics, control_metrics : dict
            Expected keys: brier_score, ece.
            Optional: year_by_year (list of dicts), fresh_data_brier,
            fresh_data_ece, win_rate, roi, total_bets.

        Returns
        -------
        dict  {passes: bool, reasons: list[str]}
        """
        reasons: list[str] = []
        passes = True

        # --- 1. Brier / ECE improvement ---
        c_brier = candidate_metrics.get("brier_score")
        b_brier = control_metrics.get("brier_score")
        c_ece = candidate_metrics.get("ece")
        b_ece = control_metrics.get("ece")

        brier_better = False
        ece_better = False

        if c_brier is not None and b_brier is not None:
            if c_brier < b_brier:
                brier_better = True
                reasons.append(
                    f"Brier improved: {c_brier:.4f} vs {b_brier:.4f} "
                    f"(delta {c_brier - b_brier:+.4f})"
                )
            else:
                reasons.append(
                    f"Brier not improved: {c_brier:.4f} vs {b_brier:.4f}"
                )

        if c_ece is not None and b_ece is not None:
            if c_ece < b_ece:
                ece_better = True
                reasons.append(
                    f"ECE improved: {c_ece:.4f} vs {b_ece:.4f} "
                    f"(delta {c_ece - b_ece:+.4f})"
                )
            else:
                reasons.append(
                    f"ECE not improved: {c_ece:.4f} vs {b_ece:.4f}"
                )

        if not brier_better and not ece_better:
            passes = False
            reasons.append("FAIL: Neither Brier nor ECE improved over control.")

        # --- 2. No material degradation in the other ---
        if brier_better and c_ece is not None and b_ece is not None and b_ece > 0:
            degradation = (c_ece - b_ece) / b_ece
            if degradation > _MODEL_DEGRADATION_THRESHOLD:
                passes = False
                reasons.append(
                    f"FAIL: ECE degraded by {degradation:.1%} (threshold "
                    f"{_MODEL_DEGRADATION_THRESHOLD:.0%})"
                )

        if ece_better and c_brier is not None and b_brier is not None and b_brier > 0:
            degradation = (c_brier - b_brier) / b_brier
            if degradation > _MODEL_DEGRADATION_THRESHOLD:
                passes = False
                reasons.append(
                    f"FAIL: Brier degraded by {degradation:.1%} (threshold "
                    f"{_MODEL_DEGRADATION_THRESHOLD:.0%})"
                )

        # --- 3. Fresh-data evaluation ---
        c_fresh_brier = candidate_metrics.get("fresh_data_brier")
        b_fresh_brier = control_metrics.get("fresh_data_brier")
        if c_fresh_brier is not None and b_fresh_brier is not None:
            if c_fresh_brier > b_fresh_brier:
                passes = False
                reasons.append(
                    f"FAIL: Fresh-data Brier worse ({c_fresh_brier:.4f} vs "
                    f"{b_fresh_brier:.4f}, cutoff {_FRESH_DATA_CUTOFF})"
                )
            else:
                reasons.append(
                    f"Fresh-data Brier OK: {c_fresh_brier:.4f} vs {b_fresh_brier:.4f}"
                )

        c_fresh_ece = candidate_metrics.get("fresh_data_ece")
        b_fresh_ece = control_metrics.get("fresh_data_ece")
        if c_fresh_ece is not None and b_fresh_ece is not None:
            if c_fresh_ece > b_fresh_ece:
                reasons.append(
                    f"WARNING: Fresh-data ECE worse ({c_fresh_ece:.4f} vs "
                    f"{b_fresh_ece:.4f})"
                )

        # --- 4. Year-by-year regression ---
        c_yby = candidate_metrics.get("year_by_year", [])
        b_yby = control_metrics.get("year_by_year", [])
        if c_yby and b_yby:
            b_lookup = {y["year"]: y for y in b_yby}
            for cy in c_yby:
                yr = cy.get("year")
                by = b_lookup.get(yr)
                if by is None:
                    continue
                c_roi = cy.get("roi")
                b_roi = by.get("roi")
                if c_roi is not None and b_roi is not None and b_roi - c_roi > 0.15:
                    passes = False
                    reasons.append(
                        f"FAIL: Year {yr} ROI regression "
                        f"({c_roi:+.1%} vs baseline {b_roi:+.1%})"
                    )
                    continue

                c_year_brier = cy.get("brier_score", cy.get("brier"))
                b_year_brier = by.get("brier_score", by.get("brier"))
                brier_degradation = _relative_degradation(c_year_brier, b_year_brier)
                if (
                    brier_degradation is not None
                    and brier_degradation > _MODEL_DEGRADATION_THRESHOLD
                ):
                    passes = False
                    reasons.append(
                        f"FAIL: Year {yr} Brier regression "
                        f"({c_year_brier:.4f} vs baseline {b_year_brier:.4f})"
                    )

                c_year_ece = cy.get("ece")
                b_year_ece = by.get("ece")
                ece_degradation = _relative_degradation(c_year_ece, b_year_ece)
                ece_abs_degradation = (
                    (c_year_ece - b_year_ece)
                    if c_year_ece is not None and b_year_ece is not None
                    else None
                )
                # Require both relative >10 % AND absolute >2.5 ppts to flag.
                # Pure relative checks are too strict when baseline ECE is low.
                if (
                    ece_degradation is not None
                    and ece_abs_degradation is not None
                    and ece_degradation > _MODEL_DEGRADATION_THRESHOLD
                    and ece_abs_degradation > _ECE_ABS_REGRESSION_THRESHOLD
                ):
                    passes = False
                    reasons.append(
                        f"FAIL: Year {yr} ECE regression "
                        f"({c_year_ece:.4f} vs baseline {b_year_ece:.4f}, "
                        f"relative {ece_degradation:.1%}, "
                        f"absolute {ece_abs_degradation:+.4f})"
                    )

        return {"passes": passes, "reasons": reasons}

    # -----------------------------------------------------------------
    # Trading Bar
    # -----------------------------------------------------------------
    @staticmethod
    def check_trading_bar(
        candidate_sweep: dict,
        baseline_sweep: dict,
    ) -> dict:
        """
        Evaluate the Trading Bar.

        Passes when:
        1. Better ROI OR profit than baseline.
        2. Drawdown not >2x baseline drawdown.
        3. CLV not worse (within 1 percentage point).
        4. Behaviour is sane (robustness checks pass).

        Parameters
        ----------
        candidate_sweep, baseline_sweep : dict
            Expected keys: roi, total_profit, max_drawdown_pct (or
            max_drawdown), avg_clv, bet_log (pd.DataFrame).
            Optional: baseline_bet_log for volume_sanity_check.

        Returns
        -------
        dict  {passes: bool, reasons: list[str]}
        """
        reasons: list[str] = []
        passes = True

        # --- 1. ROI / profit improvement ---
        c_roi = candidate_sweep.get("roi", 0.0)
        b_roi = baseline_sweep.get("roi", 0.0)
        c_profit = candidate_sweep.get("total_profit", 0.0)
        b_profit = baseline_sweep.get("total_profit", 0.0)

        roi_better = c_roi > b_roi
        profit_better = c_profit > b_profit

        if roi_better:
            reasons.append(f"ROI improved: {c_roi:+.2%} vs {b_roi:+.2%}")
        else:
            reasons.append(f"ROI not improved: {c_roi:+.2%} vs {b_roi:+.2%}")

        if profit_better:
            reasons.append(f"Profit improved: ${c_profit:+.2f} vs ${b_profit:+.2f}")
        else:
            reasons.append(f"Profit not improved: ${c_profit:+.2f} vs ${b_profit:+.2f}")

        if not roi_better and not profit_better:
            passes = False
            reasons.append("FAIL: Neither ROI nor profit improved.")

        # --- 2. Drawdown ---
        c_dd = candidate_sweep.get("max_drawdown_pct", candidate_sweep.get("max_drawdown", 0.0))
        b_dd = baseline_sweep.get("max_drawdown_pct", baseline_sweep.get("max_drawdown", 0.0))

        if b_dd > 0 and c_dd > _DRAWDOWN_MULTIPLIER * b_dd:
            passes = False
            reasons.append(
                f"FAIL: Drawdown {c_dd:.1%} exceeds {_DRAWDOWN_MULTIPLIER:.0f}x "
                f"baseline {b_dd:.1%}"
            )
        else:
            reasons.append(f"Drawdown OK: {c_dd:.1%} vs baseline {b_dd:.1%}")

        # --- 3. CLV ---
        c_clv = candidate_sweep.get("avg_clv")
        b_clv = baseline_sweep.get("avg_clv")
        if c_clv is not None and b_clv is not None:
            if b_clv - c_clv > _CLV_TOLERANCE:
                passes = False
                reasons.append(
                    f"FAIL: CLV worse by {(b_clv - c_clv):.2%} "
                    f"(tolerance {_CLV_TOLERANCE:.2%})"
                )
            else:
                reasons.append(
                    f"CLV OK: candidate {c_clv:+.2%} vs baseline {b_clv:+.2%}"
                )

        # --- 4. Behaviour / robustness ---
        c_log = candidate_sweep.get("bet_log", pd.DataFrame())
        b_log = baseline_sweep.get("bet_log", pd.DataFrame())

        if not c_log.empty:
            # Edge vs profit
            ep = edge_vs_profit_check(c_log)
            if ep["verdict"] == "SUSPICIOUSLY_HIGH":
                passes = False
                reasons.append(
                    f"FAIL: Realized profit ({ep['realized_profit_rate']:+.2%}) "
                    f"far exceeds claimed edge ({ep['avg_claimed_edge']:+.2%}) -- "
                    f"possible luck or data leakage"
                )
            elif ep["verdict"] == "EXECUTION_DRAG":
                reasons.append(
                    f"WARNING: Realized profit ({ep['realized_profit_rate']:+.2%}) "
                    f"below claimed edge ({ep['avg_claimed_edge']:+.2%}) -- "
                    f"execution drag"
                )

            # Volume sanity
            if not b_log.empty:
                vol = volume_sanity_check(c_log, b_log)
                if not vol["passes"]:
                    passes = False
                    for f in vol["flags"]:
                        reasons.append(f"FAIL: {f}")

            # Trader C audit
            tc = trader_c_behavior_audit(c_log)
            if not tc.get("passes", True):
                passes = False
                for f in tc.get("flags", []):
                    reasons.append(f"FAIL (Trader C): {f}")

        return {"passes": passes, "reasons": reasons}


# =============================================================================
# Verdict
# =============================================================================

def evaluate_for_promotion(
    candidate_model_metrics: dict,
    control_model_metrics: dict,
    candidate_sweep: dict,
    baseline_sweep: dict,
) -> dict:
    """
    Run both bars and issue a verdict.

    Returns
    -------
    dict  Keys: verdict ('PROMOTE'|'SHADOW_TEST'|'REJECT'),
          model_bar, trading_bar, reasoning.
    """
    gate = PromotionGate()

    model_bar = gate.check_model_bar(candidate_model_metrics, control_model_metrics)
    trading_bar = gate.check_trading_bar(candidate_sweep, baseline_sweep)

    model_passes = model_bar["passes"]
    trading_passes = trading_bar["passes"]

    if model_passes and trading_passes:
        verdict = "PROMOTE"
    elif model_passes or trading_passes:
        verdict = "SHADOW_TEST"
    else:
        verdict = "REJECT"

    reasoning = []
    reasoning.append(f"Model Bar: {'PASS' if model_passes else 'FAIL'}")
    reasoning.extend(f"  - {r}" for r in model_bar["reasons"])
    reasoning.append(f"Trading Bar: {'PASS' if trading_passes else 'FAIL'}")
    reasoning.extend(f"  - {r}" for r in trading_bar["reasons"])
    reasoning.append(f"Verdict: {verdict}")

    return {
        "verdict": verdict,
        "model_bar": model_bar,
        "trading_bar": trading_bar,
        "reasoning": reasoning,
    }


# =============================================================================
# Report generation
# =============================================================================

def generate_promotion_report(
    candidate_model_metrics: dict,
    control_model_metrics: dict,
    candidate_sweep: dict,
    baseline_sweep: dict,
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Generate a detailed markdown promotion report.

    Includes:
    - Model bar results (metric-by-metric)
    - Trading bar results (strategy comparison)
    - Robustness check results
    - Year-by-year breakdown
    - Outlier event analysis
    - Final verdict with confidence level

    Returns the report as a string and optionally writes to ``output_path``.
    """
    result = evaluate_for_promotion(
        candidate_model_metrics,
        control_model_metrics,
        candidate_sweep,
        baseline_sweep,
    )

    lines: list[str] = []

    def _h1(text: str):
        lines.append(f"# {text}")
        lines.append("")

    def _h2(text: str):
        lines.append(f"## {text}")
        lines.append("")

    def _h3(text: str):
        lines.append(f"### {text}")
        lines.append("")

    def _bullet(text: str):
        lines.append(f"- {text}")

    def _blank():
        lines.append("")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _h1(f"Promotion Gate Report -- {timestamp}")

    # --- Verdict ---
    _h2("Verdict")
    verdict = result["verdict"]
    _bullet(f"**{verdict}**")
    if verdict == "PROMOTE":
        _bullet("Candidate clears both the Model Bar and the Trading Bar.")
        _bullet("Recommended action: promote to production.")
    elif verdict == "SHADOW_TEST":
        bar_passed = "Model" if result["model_bar"]["passes"] else "Trading"
        bar_failed = "Trading" if bar_passed == "Model" else "Model"
        _bullet(f"Candidate clears the {bar_passed} Bar but fails the {bar_failed} Bar.")
        _bullet("Recommended action: deploy in shadow mode alongside production.")
    else:
        _bullet("Candidate fails both bars.")
        _bullet("Recommended action: reject -- further iteration needed.")
    _blank()

    # --- Model Bar ---
    _h2("Model Bar")
    model_bar = result["model_bar"]
    _bullet(f"Result: **{'PASS' if model_bar['passes'] else 'FAIL'}**")
    _blank()

    _h3("Metric-by-metric comparison")
    metrics_table = [
        ("Brier Score", candidate_model_metrics.get("brier_score"), control_model_metrics.get("brier_score")),
        ("ECE", candidate_model_metrics.get("ece"), control_model_metrics.get("ece")),
        ("Fresh-data Brier", candidate_model_metrics.get("fresh_data_brier"), control_model_metrics.get("fresh_data_brier")),
        ("Fresh-data ECE", candidate_model_metrics.get("fresh_data_ece"), control_model_metrics.get("fresh_data_ece")),
        ("Win Rate", candidate_model_metrics.get("win_rate"), control_model_metrics.get("win_rate")),
    ]

    lines.append("| Metric | Candidate | Control | Delta |")
    lines.append("|--------|-----------|---------|-------|")
    for name, c_val, b_val in metrics_table:
        if c_val is not None and b_val is not None:
            delta = c_val - b_val
            better = delta < 0 if "Brier" in name or "ECE" in name else delta > 0
            marker = " *" if better else ""
            lines.append(
                f"| {name} | {c_val:.4f} | {b_val:.4f} | {delta:+.4f}{marker} |"
            )
        elif c_val is not None:
            lines.append(f"| {name} | {c_val:.4f} | N/A | -- |")
        elif b_val is not None:
            lines.append(f"| {name} | N/A | {b_val:.4f} | -- |")
    _blank()

    for r in model_bar["reasons"]:
        _bullet(r)
    _blank()

    # --- Trading Bar ---
    _h2("Trading Bar")
    trading_bar = result["trading_bar"]
    _bullet(f"Result: **{'PASS' if trading_bar['passes'] else 'FAIL'}**")
    _blank()

    _h3("Strategy comparison")
    strat_table = [
        ("ROI", candidate_sweep.get("roi"), baseline_sweep.get("roi"), True),
        ("Total Profit ($)", candidate_sweep.get("total_profit"), baseline_sweep.get("total_profit"), True),
        ("Max Drawdown", candidate_sweep.get("max_drawdown_pct", candidate_sweep.get("max_drawdown")),
         baseline_sweep.get("max_drawdown_pct", baseline_sweep.get("max_drawdown")), False),
        ("Avg CLV", candidate_sweep.get("avg_clv"), baseline_sweep.get("avg_clv"), True),
        ("Total Bets", candidate_sweep.get("total_bets"), baseline_sweep.get("total_bets"), None),
    ]

    lines.append("| Metric | Candidate | Baseline | Delta |")
    lines.append("|--------|-----------|----------|-------|")
    for name, c_val, b_val, higher_is_better in strat_table:
        if c_val is not None and b_val is not None:
            delta = c_val - b_val
            if higher_is_better is not None:
                better = (delta > 0) if higher_is_better else (delta < 0)
                marker = " *" if better else ""
            else:
                marker = ""
            if isinstance(c_val, float) and abs(c_val) < 10:
                lines.append(
                    f"| {name} | {c_val:+.4f} | {b_val:+.4f} | {delta:+.4f}{marker} |"
                )
            else:
                lines.append(
                    f"| {name} | {c_val} | {b_val} | {delta:+}{marker} |"
                )
    _blank()

    for r in trading_bar["reasons"]:
        _bullet(r)
    _blank()

    # --- Robustness checks ---
    _h2("Robustness Checks")

    c_log = candidate_sweep.get("bet_log", pd.DataFrame())
    b_log = baseline_sweep.get("bet_log", pd.DataFrame())

    # Year-by-year
    _h3("Year-by-year breakdown")
    if not c_log.empty:
        yby = year_by_year_analysis(c_log)
        if not yby.empty:
            for _, row in yby.iterrows():
                flag = " [OUTLIER]" if row.get("outlier", False) else ""
                lines.append(
                    f"- **{int(row['year'])}**: "
                    f"ROI {row['roi']:+.1%}, "
                    f"Win rate {row['win_rate']:.1%}, "
                    f"Bets {row['total_bets']}, "
                    f"Max DD {row['max_drawdown']:.1%}"
                    f"{flag}"
                )
            _blank()
        else:
            _bullet("No year-by-year data available.")
            _blank()
    else:
        _bullet("No candidate bet log available.")
        _blank()

    # Outlier events
    _h3("Outlier event analysis")
    if not c_log.empty:
        outliers = leave_one_event_out(c_log)
        if outliers:
            for o in outliers:
                lines.append(
                    f"- **{o['event_date']}**: "
                    f"Removing shifts ROI by {o['roi_shift']:+.1%} "
                    f"({o['n_bets']} bets, P&L ${o['event_profit']:+.2f})"
                )
            _blank()
        else:
            _bullet("No single event drives ROI by more than 5pp.")
            _blank()
    else:
        _bullet("No candidate bet log available.")
        _blank()

    # Edge vs profit
    _h3("Edge vs profit consistency")
    if not c_log.empty:
        ep = edge_vs_profit_check(c_log)
        _bullet(f"Avg claimed edge: {ep['avg_claimed_edge']:+.2%}")
        _bullet(f"Realized profit rate: {ep['realized_profit_rate']:+.2%}")
        _bullet(f"Gap: {ep['gap']:+.2%}")
        _bullet(f"Verdict: **{ep['verdict']}**")
        _blank()

    # Top events
    _h3("Top winning and losing events")
    if not c_log.empty:
        top = top_event_analysis(c_log)
        if not top.empty:
            for _, row in top.iterrows():
                fighters = row.get("fighters", "")
                lines.append(
                    f"- [{row['category']}] **{row['event_date']}**: "
                    f"P&L ${row['profit']:+.2f}, {row['n_bets']} bets"
                    + (f" ({fighters})" if fighters else "")
                )
            _blank()

    # Volume sanity
    _h3("Volume sanity")
    if not c_log.empty and not b_log.empty:
        vol = volume_sanity_check(c_log, b_log)
        _bullet(f"Candidate bets: {vol['candidate_bets']} vs baseline: {vol['baseline_bets']} (ratio {vol['bet_ratio']:.2f}x)")
        _bullet(f"Trader C volume share: {vol['trader_c_volume_share']:.1%}")
        _bullet(f"Result: **{'PASS' if vol['passes'] else 'FLAG'}**")
        if vol["flags"]:
            for f in vol["flags"]:
                _bullet(f"  {f}")
        _blank()

    # Trader C audit
    _h3("Trader C behaviour audit")
    if not c_log.empty and "trader" in c_log.columns:
        tc = trader_c_behavior_audit(c_log)
        if tc.get("s_win_rate") is not None:
            _bullet(f"S win rate: {tc['s_win_rate']:.1%}")
        if tc.get("c_win_rate") is not None:
            _bullet(f"C win rate: {tc['c_win_rate']:.1%}")
        if tc.get("underdog_fraction") is not None:
            _bullet(f"C underdog fraction: {tc['underdog_fraction']:.0%}")
        _bullet(f"Marginal underdog bets: {tc.get('marginal_underdog_bets', 0)}")
        _bullet(f"Result: **{'PASS' if tc.get('passes', True) else 'FLAG'}**")
        if tc.get("flags"):
            for f in tc["flags"]:
                _bullet(f"  {f}")
        _blank()

    # --- Final reasoning ---
    _h2("Full reasoning log")
    for r in result["reasoning"]:
        lines.append(r)
    _blank()

    report = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        logger.info(f"Promotion report written to {output_path}")

    return report


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line entry point for promotion gate evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(
        description="Promotion Gate -- evaluate a candidate model/strategy for production",
    )
    parser.add_argument(
        "--candidate-metrics",
        type=str,
        required=True,
        help="Path to candidate model metrics JSON file",
    )
    parser.add_argument(
        "--control-metrics",
        type=str,
        required=True,
        help="Path to control (baseline) model metrics JSON file",
    )
    parser.add_argument(
        "--candidate-sweep",
        type=str,
        default=None,
        help="Path to candidate sweep JSON artifact for the trading bar",
    )
    parser.add_argument(
        "--baseline-sweep",
        type=str,
        default=None,
        help="Path to baseline sweep JSON artifact for the trading bar",
    )
    parser.add_argument(
        "--candidate-log",
        type=str,
        default=None,
        help="Path to candidate bet log CSV file",
    )
    parser.add_argument(
        "--baseline-log",
        type=str,
        default=None,
        help="Path to baseline bet log CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the output markdown report",
    )

    args = parser.parse_args()

    if bool(args.candidate_sweep) != bool(args.baseline_sweep):
        parser.error("--candidate-sweep and --baseline-sweep must be provided together")

    # Load model metrics
    with open(args.candidate_metrics, "r", encoding="utf-8") as f:
        candidate_model_metrics = json.load(f)
    with open(args.control_metrics, "r", encoding="utf-8") as f:
        control_model_metrics = json.load(f)

    # Build sweep dicts from actual sweep artifacts when provided, otherwise
    # fall back to any trading metrics embedded in the model-metrics payloads.
    if args.candidate_sweep and args.baseline_sweep:
        candidate_sweep = _load_sweep_artifact(args.candidate_sweep, role="candidate")
        baseline_sweep = _load_sweep_artifact(args.baseline_sweep, role="baseline")
    else:
        candidate_sweep = dict(candidate_model_metrics)
        baseline_sweep = dict(control_model_metrics)
        if _looks_like_sweep_summary(candidate_sweep):
            candidate_sweep = _canonicalize_sweep_payload(
                candidate_sweep,
                args.candidate_metrics,
            )
        if _looks_like_sweep_summary(baseline_sweep):
            baseline_sweep = _canonicalize_sweep_payload(
                baseline_sweep,
                args.control_metrics,
            )

    if args.candidate_log:
        c_log = pd.read_csv(args.candidate_log)
        candidate_sweep["bet_log"] = c_log
    else:
        candidate_sweep.setdefault("bet_log", pd.DataFrame())

    if args.baseline_log:
        b_log = pd.read_csv(args.baseline_log)
        baseline_sweep["bet_log"] = b_log
    else:
        baseline_sweep.setdefault("bet_log", pd.DataFrame())

    # Default output path
    output_path = args.output
    if output_path is None:
        from src.config import LOGS_DIR
        output_path = str(
            LOGS_DIR / f"promotion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )

    # Run gate
    report = generate_promotion_report(
        candidate_model_metrics,
        control_model_metrics,
        candidate_sweep,
        baseline_sweep,
        output_path=output_path,
    )

    # Print verdict summary to stdout
    result = evaluate_for_promotion(
        candidate_model_metrics,
        control_model_metrics,
        candidate_sweep,
        baseline_sweep,
    )
    print(f"\n{'=' * 60}")
    print(f"PROMOTION GATE VERDICT: {result['verdict']}")
    print(f"{'=' * 60}")
    for r in result["reasoning"]:
        print(f"  {r}")
    print(f"\nFull report: {output_path}")


if __name__ == "__main__":
    main()
