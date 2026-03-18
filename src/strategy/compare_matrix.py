"""
Full dataset-variant x feature-family comparison matrix.

Automates the cross-product of dataset arms (from ufc_refresh) and feature
families (production, production_betsapi, expanded BetsAPI, no-odds, etc.)
through walk-forward evaluation, producing per-cell metrics, time-period
slices, and a summary report.

Usage:
    python -m src.strategy.compare_matrix
    python -m src.strategy.compare_matrix --variants append_only_2026,best_of_both_full_history
    python -m src.strategy.compare_matrix --families production,no_odds --bootstrap 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.config import LOGS_DIR, RAW_DATA_DIR, TRAIN_CUTOFF_DATE
from src.data.betsapi_mma import augment_features_with_betsapi_mma
from src.data.kaggle_loader import load_kaggle_dataset
from src.data.ufc_refresh import build_training_dataset_variants
from src.features.build_features import (
    BETSAPI_CHALLENGER_FEATURE_NAMES,
    BETSAPI_HISTORICAL_FEATURE_NAMES,
    build_features,
    get_feature_family_columns,
)
from src.model.compare import (
    bootstrap_metric_comparison,
    mcnemar_test,
    walk_forward_comparison,
)
from src.strategy.lab_stats import compute_ece

logger = logging.getLogger(__name__)

MATRIX_DIR = LOGS_DIR / "compare_matrix"
MATRIX_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature families
# ---------------------------------------------------------------------------

ALL_FEATURE_FAMILIES = (
    "production",
    "production_betsapi",
    "production_betsapi_expanded",
    "ufcstats_refreshed",
    "ufcstats_betsapi_expanded",
    "no_odds",
)
BETSAPI_FEATURE_FAMILIES = (
    "production_betsapi",
    "production_betsapi_expanded",
    "ufcstats_betsapi_expanded",
)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class MatrixConfig:
    """Configuration for a single cell of the comparison matrix."""

    dataset_variant: str
    feature_family: str
    calibration_method: str = "isotonic"
    retrain_months: int = 4
    bootstrap: int = 5000


def build_matrix_configs(
    variants: list[str],
    families: list[str] | None = None,
    calibration_method: str = "isotonic",
    retrain_months: int = 4,
    bootstrap: int = 5000,
) -> list[MatrixConfig]:
    """Generate all (dataset_variant x feature_family) combinations."""
    if families is None:
        families = list(ALL_FEATURE_FAMILIES)
    configs: list[MatrixConfig] = []
    for variant in variants:
        for family in families:
            configs.append(
                MatrixConfig(
                    dataset_variant=variant,
                    feature_family=family,
                    calibration_method=calibration_method,
                    retrain_months=retrain_months,
                    bootstrap=bootstrap,
                )
            )
    return configs


# ---------------------------------------------------------------------------
# Feature column selection helpers
# ---------------------------------------------------------------------------

def _select_feature_columns(
    features_df: pd.DataFrame,
    family: str,
) -> tuple[list[str], list[str]]:
    """Return (baseline_cols, challenger_cols) for a given feature family.

    The walk-forward comparison always needs a baseline and a challenger.
    For most families the baseline is production features and the challenger
    is the family-specific set, so the comparison shows marginal value of
    that family relative to the production baseline.
    """
    baseline_cols = get_feature_family_columns(features_df, "production")
    if family in ALL_FEATURE_FAMILIES:
        challenger_cols = (
            list(baseline_cols)
            if family == "production"
            else get_feature_family_columns(features_df, family)
        )
        return baseline_cols, challenger_cols

    raise ValueError(f"Unknown feature family: {family!r}")


# ---------------------------------------------------------------------------
# Slice helpers
# ---------------------------------------------------------------------------

def _betsapi_feature_names_for_family(feature_family: str | None) -> list[str]:
    """Return the correct BetsAPI feature name list for a given family."""
    if feature_family == "production_betsapi":
        return list(BETSAPI_HISTORICAL_FEATURE_NAMES)
    return list(BETSAPI_CHALLENGER_FEATURE_NAMES)


def _slice_predictions(
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    feature_family: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Split the predictions frame into required result slices."""
    preds = predictions_df.copy()
    preds["event_date"] = pd.to_datetime(preds["event_date"], errors="coerce")

    slices: dict[str, pd.DataFrame] = {}

    # Time-period slices
    slices["2023_2024"] = preds[
        (preds["event_date"] >= "2023-01-01") & (preds["event_date"] < "2025-01-01")
    ]
    slices["2025"] = preds[
        (preds["event_date"] >= "2025-01-01") & (preds["event_date"] < "2026-01-01")
    ]
    slices["2026_ytd"] = preds[preds["event_date"] >= "2026-01-01"]

    # Favorites vs underdogs (based on baseline probability)
    slices["favorites"] = preds[preds["baseline_prob"] >= 0.5]
    slices["underdogs"] = preds[preds["baseline_prob"] < 0.5]

    # BetsAPI coverage slices — join back to features to check coverage
    family_feature_names = _betsapi_feature_names_for_family(feature_family)
    betsapi_cols = [c for c in family_feature_names if c in features_df.columns]
    if betsapi_cols:
        feat = features_df[["event_date", "fighter_a", "fighter_b"] + betsapi_cols].copy()
        feat["event_date"] = pd.to_datetime(feat["event_date"], errors="coerce")
        feat["_has_betsapi"] = feat[betsapi_cols].notna().any(axis=1)
        feat = feat.drop_duplicates(subset=["event_date", "fighter_a", "fighter_b"])
        merged = preds.merge(
            feat[["event_date", "fighter_a", "fighter_b", "_has_betsapi"]],
            on=["event_date", "fighter_a", "fighter_b"],
            how="left",
        )
        merged["_has_betsapi"] = merged["_has_betsapi"].fillna(False)
        slices["high_betsapi_coverage"] = merged[merged["_has_betsapi"]].drop(columns=["_has_betsapi"])
        slices["low_betsapi_coverage"] = merged[~merged["_has_betsapi"]].drop(columns=["_has_betsapi"])
    else:
        slices["high_betsapi_coverage"] = preds.iloc[0:0]
        slices["low_betsapi_coverage"] = preds

    return slices


def _compute_slice_metrics(
    slice_df: pd.DataFrame,
    bootstrap_n: int,
) -> dict[str, Any]:
    """Compute the standard metric set for a prediction slice."""
    if slice_df.empty or len(slice_df) < 2:
        return {"n": 0, "note": "insufficient data"}

    y_true = slice_df["target"].astype(int).to_numpy()
    baseline_probs = slice_df["baseline_prob"].astype(float).to_numpy()
    challenger_probs = slice_df["challenger_prob"].astype(float).to_numpy()

    if len(np.unique(y_true)) < 2:
        return {"n": int(len(y_true)), "note": "single class only"}

    baseline_preds = (baseline_probs > 0.5).astype(int)
    challenger_preds = (challenger_probs > 0.5).astype(int)

    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "baseline_accuracy": float(accuracy_score(y_true, baseline_preds)),
        "challenger_accuracy": float(accuracy_score(y_true, challenger_preds)),
        "baseline_brier": float(brier_score_loss(y_true, baseline_probs)),
        "challenger_brier": float(brier_score_loss(y_true, challenger_probs)),
        "baseline_log_loss": float(log_loss(y_true, baseline_probs)),
        "challenger_log_loss": float(log_loss(y_true, challenger_probs)),
        "baseline_ece": float(compute_ece(y_true, baseline_probs)),
        "challenger_ece": float(compute_ece(y_true, challenger_probs)),
    }

    metrics["mcnemar"] = mcnemar_test(y_true, baseline_preds, challenger_preds)

    metrics["brier_bootstrap"] = bootstrap_metric_comparison(
        y_true, baseline_probs, challenger_probs,
        brier_score_loss, n_bootstrap=bootstrap_n,
    )
    metrics["logloss_bootstrap"] = bootstrap_metric_comparison(
        y_true, baseline_probs, challenger_probs,
        log_loss, n_bootstrap=bootstrap_n,
    )

    return metrics


# ---------------------------------------------------------------------------
# Single matrix cell
# ---------------------------------------------------------------------------

def run_matrix_cell(
    config: MatrixConfig,
    variants: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Run a single (dataset_variant x feature_family) cell.

    Parameters
    ----------
    config : MatrixConfig
        Cell configuration.
    variants : dict[str, pd.DataFrame]
        Pre-built dataset variants from ``build_training_dataset_variants()``.

    Returns
    -------
    dict with keys: config, overall_metrics, slice_metrics, predictions_df,
    features_df, baseline_feature_cols, challenger_feature_cols.
    """
    label = f"{config.dataset_variant} x {config.feature_family}"
    logger.info("=" * 70)
    logger.info("MATRIX CELL: %s", label)
    logger.info("=" * 70)

    training_df = variants[config.dataset_variant]
    logger.info("Building features for %s (%d fights)", config.dataset_variant, len(training_df))
    features_df = build_features(training_df)

    # Augment with BetsAPI columns (needed for betsapi families; harmless for others)
    needs_betsapi = config.feature_family in BETSAPI_FEATURE_FAMILIES
    if needs_betsapi:
        features_df = augment_features_with_betsapi_mma(features_df, save_artifacts=False)

    baseline_cols, challenger_cols = _select_feature_columns(features_df, config.feature_family)

    logger.info("Baseline features: %d, Challenger features: %d", len(baseline_cols), len(challenger_cols))

    wf = walk_forward_comparison(
        features_df,
        baseline_cols,
        challenger_cols,
        retrain_months=config.retrain_months,
        start_date=TRAIN_CUTOFF_DATE,
        calibration_method=config.calibration_method,
    )

    predictions_df = wf["predictions"]
    y_true = wf["y_true"]
    baseline_probs = wf["baseline_probs"]
    challenger_probs = wf["challenger_probs"]

    # Overall metrics
    if len(y_true) == 0:
        logger.warning("No predictions for cell %s", label)
        return {
            "config": config,
            "overall_metrics": {"n": 0, "note": "no predictions"},
            "slice_metrics": {},
            "predictions_df": predictions_df,
            "features_df": features_df,
            "baseline_feature_cols": baseline_cols,
            "challenger_feature_cols": challenger_cols,
            "fold_results": wf["fold_results"],
        }

    overall_metrics = _compute_slice_metrics(predictions_df, config.bootstrap)

    # Result slices
    slices = _slice_predictions(predictions_df, features_df, feature_family=config.feature_family)
    slice_metrics = {
        name: _compute_slice_metrics(sl, config.bootstrap)
        for name, sl in slices.items()
    }

    logger.info(
        "Cell %s done: n=%d, baseline_brier=%.4f, challenger_brier=%.4f",
        label,
        overall_metrics.get("n", 0),
        overall_metrics.get("baseline_brier", float("nan")),
        overall_metrics.get("challenger_brier", float("nan")),
    )

    return {
        "config": config,
        "overall_metrics": overall_metrics,
        "slice_metrics": slice_metrics,
        "predictions_df": predictions_df,
        "features_df": features_df,
        "baseline_feature_cols": baseline_cols,
        "challenger_feature_cols": challenger_cols,
        "fold_results": wf["fold_results"],
    }


# ---------------------------------------------------------------------------
# Full matrix
# ---------------------------------------------------------------------------

def _flatten_cell_to_row(cell_result: dict[str, Any]) -> dict[str, Any]:
    """Flatten a cell result into a single dict row for the summary CSV."""
    cfg = cell_result["config"]
    m = cell_result["overall_metrics"]
    row: dict[str, Any] = {
        "dataset_variant": cfg.dataset_variant,
        "feature_family": cfg.feature_family,
        "calibration_method": cfg.calibration_method,
        "retrain_months": cfg.retrain_months,
        "n": m.get("n", 0),
    }

    for key in (
        "baseline_accuracy", "challenger_accuracy",
        "baseline_brier", "challenger_brier",
        "baseline_log_loss", "challenger_log_loss",
        "baseline_ece", "challenger_ece",
    ):
        row[key] = m.get(key)

    mcn = m.get("mcnemar", {})
    row["mcnemar_p"] = mcn.get("p_value")

    brier_boot = m.get("brier_bootstrap", {})
    row["brier_diff"] = brier_boot.get("diff")
    row["brier_ci_lo"] = brier_boot.get("ci_lo")
    row["brier_ci_hi"] = brier_boot.get("ci_hi")
    row["brier_p"] = brier_boot.get("p_value")

    logloss_boot = m.get("logloss_bootstrap", {})
    row["logloss_diff"] = logloss_boot.get("diff")
    row["logloss_ci_lo"] = logloss_boot.get("ci_lo")
    row["logloss_ci_hi"] = logloss_boot.get("ci_hi")
    row["logloss_p"] = logloss_boot.get("p_value")

    # Add slice summaries as extra columns
    for slice_name, sm in cell_result.get("slice_metrics", {}).items():
        row[f"{slice_name}_n"] = sm.get("n", 0)
        row[f"{slice_name}_baseline_brier"] = sm.get("baseline_brier")
        row[f"{slice_name}_challenger_brier"] = sm.get("challenger_brier")

    return row


def _render_markdown_report(
    run_dir: Path,
    summary_df: pd.DataFrame,
    cell_results: list[dict[str, Any]],
) -> str:
    """Render the full matrix report as markdown."""
    lines: list[str] = []
    lines.append("# Compare Matrix Report")
    lines.append("")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Run directory: `{run_dir}`")
    lines.append("")

    # Overall summary table
    lines.append("## Overall Summary")
    lines.append("")
    display_cols = [
        "dataset_variant", "feature_family", "n",
        "baseline_brier", "challenger_brier", "brier_diff",
        "brier_ci_lo", "brier_ci_hi", "brier_p",
        "baseline_accuracy", "challenger_accuracy", "mcnemar_p",
        "baseline_ece", "challenger_ece",
    ]
    existing = [c for c in display_cols if c in summary_df.columns]
    lines.append("```text")
    lines.append(summary_df[existing].to_string(index=False))
    lines.append("```")
    lines.append("")

    # Per-slice breakdowns
    slice_names = [
        "2023_2024", "2025", "2026_ytd",
        "favorites", "underdogs",
        "high_betsapi_coverage", "low_betsapi_coverage",
    ]
    for slice_name in slice_names:
        lines.append(f"## Slice: {slice_name}")
        lines.append("")
        slice_rows = []
        for cell in cell_results:
            cfg = cell["config"]
            sm = cell.get("slice_metrics", {}).get(slice_name, {})
            if sm.get("n", 0) == 0:
                continue
            slice_rows.append({
                "variant": cfg.dataset_variant,
                "family": cfg.feature_family,
                "n": sm.get("n", 0),
                "baseline_brier": sm.get("baseline_brier"),
                "challenger_brier": sm.get("challenger_brier"),
                "baseline_accuracy": sm.get("baseline_accuracy"),
                "challenger_accuracy": sm.get("challenger_accuracy"),
                "baseline_ece": sm.get("baseline_ece"),
                "challenger_ece": sm.get("challenger_ece"),
            })
        if slice_rows:
            sdf = pd.DataFrame(slice_rows)
            lines.append("```text")
            lines.append(sdf.to_string(index=False))
            lines.append("```")
        else:
            lines.append("*(no data)*")
        lines.append("")

    # Verdicts
    lines.append("## Cell Verdicts")
    lines.append("")
    for cell in cell_results:
        cfg = cell["config"]
        m = cell["overall_metrics"]
        brier_boot = m.get("brier_bootstrap", {})
        label = f"`{cfg.dataset_variant}` x `{cfg.feature_family}`"
        n = m.get("n", 0)
        if n == 0:
            lines.append(f"- {label}: No predictions")
            continue
        diff = brier_boot.get("diff")
        ci_lo = brier_boot.get("ci_lo")
        ci_hi = brier_boot.get("ci_hi")
        if diff is not None and ci_hi is not None and ci_lo is not None:
            if ci_hi < 0:
                verdict = "Challenger SIGNIFICANTLY better (lower Brier)"
            elif ci_lo > 0:
                verdict = "Baseline SIGNIFICANTLY better (lower Brier)"
            elif diff < 0:
                verdict = "Challenger better, not significant"
            else:
                verdict = "Baseline better, not significant"
            lines.append(
                f"- {label} (n={n}): {verdict} "
                f"[diff={diff:+.4f}, 95% CI=({ci_lo:+.4f}, {ci_hi:+.4f})]"
            )
        else:
            lines.append(f"- {label} (n={n}): insufficient bootstrap data")
    lines.append("")
    return "\n".join(lines)


def run_full_matrix(
    variants: list[str] | None = None,
    families: list[str] | None = None,
    retrain_months: int = 4,
    bootstrap: int = 5000,
    calibration_method: str = "isotonic",
) -> dict[str, Any]:
    """Run the full dataset-variant x feature-family comparison matrix.

    Parameters
    ----------
    variants : list[str] | None
        Dataset variants to test. Defaults to a sensible shortlist.
    families : list[str] | None
        Feature families to test. Defaults to all.
    retrain_months : int
        Walk-forward retrain interval.
    bootstrap : int
        Bootstrap iterations for CIs.
    calibration_method : str
        Calibration method for models.

    Returns
    -------
    dict with run_dir, report_path, summary_csv_path, and cell_results.
    """
    if variants is None:
        variants = ["append_only_2026", "best_of_both_full_history"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = MATRIX_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building dataset variants...")
    legacy_df = load_kaggle_dataset(RAW_DATA_DIR / "ufc-master.csv")
    all_variants = build_training_dataset_variants(legacy_df=legacy_df)

    # Validate requested variants exist
    for v in variants:
        if v not in all_variants:
            raise ValueError(
                f"Unknown dataset variant: {v!r}. "
                f"Available: {list(all_variants.keys())}"
            )

    configs = build_matrix_configs(
        variants=variants,
        families=families,
        calibration_method=calibration_method,
        retrain_months=retrain_months,
        bootstrap=bootstrap,
    )

    logger.info(
        "Running %d matrix cells: %d variants x %d families",
        len(configs), len(variants), len(families or ALL_FEATURE_FAMILIES),
    )

    cell_results: list[dict[str, Any]] = []
    for i, config in enumerate(configs, 1):
        logger.info(
            "--- Cell %d/%d: %s x %s ---",
            i, len(configs), config.dataset_variant, config.feature_family,
        )
        try:
            result = run_matrix_cell(config, all_variants)
            cell_results.append(result)
        except Exception:
            logger.exception(
                "Cell %s x %s FAILED",
                config.dataset_variant, config.feature_family,
            )
            cell_results.append({
                "config": config,
                "overall_metrics": {"n": 0, "note": "error"},
                "slice_metrics": {},
                "predictions_df": pd.DataFrame(),
                "features_df": pd.DataFrame(),
                "baseline_feature_cols": [],
                "challenger_feature_cols": [],
                "fold_results": pd.DataFrame(),
            })

    # Save per-cell artifacts
    for cell in cell_results:
        cfg = cell["config"]
        cell_label = f"{cfg.dataset_variant}__{cfg.feature_family}"
        pred_df = cell.get("predictions_df")
        if pred_df is not None and not pred_df.empty:
            pred_df.to_csv(run_dir / f"{cell_label}_predictions.csv", index=False)
        fold_df = cell.get("fold_results")
        if fold_df is not None and not fold_df.empty:
            fold_df.to_csv(run_dir / f"{cell_label}_folds.csv", index=False)
        # Save metrics JSON
        metrics_payload = {
            "dataset_variant": cfg.dataset_variant,
            "feature_family": cfg.feature_family,
            "overall": cell["overall_metrics"],
            "slices": cell.get("slice_metrics", {}),
        }
        (run_dir / f"{cell_label}_metrics.json").write_text(
            json.dumps(metrics_payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    # Summary CSV
    summary_rows = [_flatten_cell_to_row(c) for c in cell_results]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = run_dir / "summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info("Saved summary CSV to %s", summary_csv_path)

    # Markdown report
    report_text = _render_markdown_report(run_dir, summary_df, cell_results)
    report_path = run_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved report to %s", report_path)

    # Metadata
    metadata = {
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "summary_csv_path": str(summary_csv_path),
        "variants": variants,
        "families": families or list(ALL_FEATURE_FAMILIES),
        "retrain_months": retrain_months,
        "bootstrap": bootstrap,
        "calibration_method": calibration_method,
        "cells_total": len(configs),
        "cells_completed": sum(1 for c in cell_results if c["overall_metrics"].get("n", 0) > 0),
        "timestamp": timestamp,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "summary_csv_path": str(summary_csv_path),
        "cell_results": cell_results,
        "summary_df": summary_df,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    parser = argparse.ArgumentParser(
        description="Run the full dataset-variant x feature-family comparison matrix.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated dataset variants (default: append_only_2026,best_of_both_full_history)",
    )
    parser.add_argument(
        "--families",
        type=str,
        default=None,
        help="Comma-separated feature families (default: all)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
        help="Bootstrap iterations for confidence intervals",
    )
    parser.add_argument(
        "--retrain-months",
        type=int,
        default=4,
        help="Walk-forward retrain interval in months",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="isotonic",
        help="Calibration method (default: isotonic)",
    )
    args = parser.parse_args()

    variants = args.variants.split(",") if args.variants else None
    families = args.families.split(",") if args.families else None

    result = run_full_matrix(
        variants=variants,
        families=families,
        retrain_months=args.retrain_months,
        bootstrap=args.bootstrap,
        calibration_method=args.calibration,
    )
    logger.info("Matrix run complete. Report: %s", result["report_path"])


if __name__ == "__main__":
    main()
