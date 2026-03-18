"""Strict new-data UFC source comparison for legacy vs pulled vs hybrid datasets."""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.config import LOGS_DIR, RAW_DATA_DIR, TRAIN_CUTOFF_DATE
from src.data.betsapi_mma import augment_features_with_betsapi_mma
from src.data.kaggle_loader import load_kaggle_dataset
from src.data.ufc_refresh import (
    TRAINING_DATASET_VARIANTS,
    audit_legacy_pulled_overlap,
    build_training_dataset_variants,
)
from src.features.build_features import (
    BETSAPI_CHALLENGER_FEATURE_NAMES,
    build_features,
    get_feature_family_columns,
)
from src.model.compare import bootstrap_metric_comparison, mcnemar_test, walk_forward_comparison
from src.strategy.lab_stats import compute_ece

logger = logging.getLogger(__name__)

SOURCE_COMPARE_DIR = LOGS_DIR / "ufc_source_compare"
SOURCE_COMPARE_DIR.mkdir(parents=True, exist_ok=True)
_BOUT_SPLIT_RE = re.compile(r"\s+vs\.?\s+", flags=re.IGNORECASE)


def _normalize_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _fight_key(fighter_a, fighter_b, event_date) -> str | None:
    event_ts = pd.to_datetime(event_date, errors="coerce", format="mixed")
    left = _normalize_text(fighter_a)
    right = _normalize_text(fighter_b)
    if pd.isna(event_ts) or not left or not right:
        return None
    low_name, high_name = sorted([left, right])
    return f"{event_ts.date().isoformat()}|{low_name}|{high_name}"


def _with_fight_keys(df: pd.DataFrame) -> pd.DataFrame:
    keyed = df.copy()
    keyed["event_date"] = pd.to_datetime(keyed["event_date"], errors="coerce", format="mixed")
    keyed["fight_key"] = keyed.apply(
        lambda row: _fight_key(row.get("fighter_a"), row.get("fighter_b"), row.get("event_date")),
        axis=1,
    )
    return keyed


def _safe_pct(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _table_block(df: pd.DataFrame) -> str:
    if df.empty:
        return "```text\n(no rows)\n```"
    return f"```text\n{df.to_string(index=False)}\n```"


def _compute_sliced_metrics(
    predictions_df: pd.DataFrame,
    *,
    betsapi_coverage_mask: pd.Series | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute lightweight metrics (no bootstrap) for year/favorite/coverage slices.

    Returns a dict of slice_name -> {fights, baseline_brier, challenger_brier,
    baseline_accuracy, challenger_accuracy}.
    """
    slices: dict[str, pd.DataFrame] = {}

    if predictions_df.empty:
        return {}

    dates = pd.to_datetime(predictions_df["event_date"], errors="coerce")

    # --- Year buckets ---
    mask_2023_2024 = dates.dt.year.isin([2023, 2024])
    mask_2025 = dates.dt.year == 2025
    mask_2026 = dates.dt.year == 2026

    if mask_2023_2024.any():
        slices["year_2023_2024"] = predictions_df.loc[mask_2023_2024]
    if mask_2025.any():
        slices["year_2025"] = predictions_df.loc[mask_2025]
    if mask_2026.any():
        slices["year_2026_ytd"] = predictions_df.loc[mask_2026]

    # --- Favorites vs underdogs (baseline_prob as market proxy) ---
    baseline_probs = predictions_df["baseline_prob"].astype(float)
    fav_mask = baseline_probs > 0.5
    dog_mask = baseline_probs <= 0.5
    if fav_mask.any():
        slices["favorites"] = predictions_df.loc[fav_mask]
    if dog_mask.any():
        slices["underdogs"] = predictions_df.loc[dog_mask]

    # --- High vs low BetsAPI coverage ---
    if betsapi_coverage_mask is not None and len(betsapi_coverage_mask) == len(predictions_df):
        assert predictions_df.index.is_monotonic_increasing, (
            "predictions_df index must be monotonic increasing for positional masking"
        )
        aligned_mask = betsapi_coverage_mask.values
        hi = predictions_df.iloc[np.flatnonzero(aligned_mask)]
        lo = predictions_df.iloc[np.flatnonzero(~aligned_mask)]
        if not hi.empty:
            slices["betsapi_high_coverage"] = hi
        if not lo.empty:
            slices["betsapi_low_coverage"] = lo

    result: dict[str, dict[str, Any]] = {}
    for slice_name, sub_df in slices.items():
        y = sub_df["target"].astype(int).to_numpy()
        bp = sub_df["baseline_prob"].astype(float).to_numpy()
        cp = sub_df["challenger_prob"].astype(float).to_numpy()
        result[slice_name] = {
            "fights": int(len(sub_df)),
            "baseline_brier": float(brier_score_loss(y, bp)),
            "challenger_brier": float(brier_score_loss(y, cp)),
            "baseline_accuracy": float(accuracy_score(y, (bp > 0.5).astype(int))),
            "challenger_accuracy": float(accuracy_score(y, (cp > 0.5).astype(int))),
        }
    return result


def _build_variant_summary_table(
    variants: dict[str, pd.DataFrame],
    *,
    legacy_keys: set[str],
    legacy_max_date: pd.Timestamp,
) -> pd.DataFrame:
    """Produce a side-by-side summary of every dataset variant.

    Columns: source, row_count, min_date, max_date, overlap_with_legacy,
    strict_new_count, betsapi_coverage_rate.
    """
    rows = []
    for name, df in variants.items():
        feature_df = build_features(df)
        aug_df = augment_features_with_betsapi_mma(feature_df, save_artifacts=False)

        keyed = _with_fight_keys(df)
        overlap = int(keyed["fight_key"].isin(legacy_keys).sum())
        betsapi_cols = [c for c in BETSAPI_CHALLENGER_FEATURE_NAMES if c in aug_df.columns]
        strict_feature_rows = aug_df.copy()
        strict_feature_rows["event_date"] = pd.to_datetime(
            strict_feature_rows["event_date"],
            errors="coerce",
        )
        strict_feature_rows = strict_feature_rows[
            strict_feature_rows["event_date"] > legacy_max_date
        ].copy()
        strict_feature_rows["fight_key"] = strict_feature_rows.apply(
            lambda row: _fight_key(row.get("fighter_a"), row.get("fighter_b"), row.get("event_date")),
            axis=1,
        )
        strict_feature_rows = strict_feature_rows[
            strict_feature_rows["fight_key"].notna()
        ].drop_duplicates(subset="fight_key")
        strict_new = int(len(strict_feature_rows))
        any_betsapi = int(
            strict_feature_rows[betsapi_cols].notna().any(axis=1).sum()
        ) if betsapi_cols else 0

        rows.append(
            {
                "source": name,
                "row_count": int(len(df)),
                "min_date": str(keyed["event_date"].min().date()) if not keyed.empty else None,
                "max_date": str(keyed["event_date"].max().date()) if not keyed.empty else None,
                "overlap_with_legacy": overlap,
                "strict_new_count": strict_new,
                "betsapi_coverage_rate": _safe_pct(any_betsapi, strict_new),
            }
        )
    return pd.DataFrame(rows)


_MIN_SHORTLIST_FIGHTS = 20
_NON_PROMOTABLE_VARIANTS = {"pulled_all"}


def shortlist_variants(
    result_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return top 1-2 dataset variants that beat the append_only_2026 control arm.

    Rules:
    - Must beat append_only_2026 on challenger Brier score.
    - Must not be worse on challenger log loss.
    - Must have >= _MIN_SHORTLIST_FIGHTS eligible fights.
    - pulled_all is diagnostic only and never shortlisted.

    Returns a list of dicts with keys: source, challenger_brier, challenger_log_loss,
    brier_delta, reason.
    """
    # Find the control arm
    control = None
    for p in result_payloads:
        if p["source"] == "append_only_2026":
            control = p
            break

    if control is None or control["strict_metrics"]["eligible_fights"] == 0:
        return []

    control_brier = control["strict_metrics"]["challenger"].get("brier_score", float("inf"))
    control_logloss = control["strict_metrics"]["challenger"].get("log_loss", float("inf"))

    candidates = []
    for p in result_payloads:
        src = p["source"]
        if src == "append_only_2026" or src in _NON_PROMOTABLE_VARIANTS:
            continue
        m = p["strict_metrics"]
        if m["eligible_fights"] < _MIN_SHORTLIST_FIGHTS:
            continue
        c_brier = m["challenger"].get("brier_score")
        c_logloss = m["challenger"].get("log_loss")
        if c_brier is None or c_logloss is None:
            continue
        if c_brier >= control_brier:
            continue  # must beat control on Brier
        if c_logloss > control_logloss:
            continue  # must not be worse on log loss
        candidates.append(
            {
                "source": src,
                "challenger_brier": c_brier,
                "challenger_log_loss": c_logloss,
                "brier_delta": c_brier - control_brier,
                "reason": f"Brier {c_brier:.4f} vs control {control_brier:.4f} (delta {c_brier - control_brier:+.4f})",
            }
        )

    # Sort by Brier ascending; return top 2
    candidates.sort(key=lambda c: c["challenger_brier"])
    return candidates[:2]


def _compute_metrics(predictions_df: pd.DataFrame, *, bootstrap: int) -> dict[str, Any]:
    if predictions_df.empty:
        return {
            "eligible_fights": 0,
            "baseline": {},
            "challenger": {},
            "mcnemar": {},
            "brier_bootstrap": {},
            "logloss_bootstrap": {},
            "verdict": "No eligible strict-new-data predictions.",
        }

    y_true = predictions_df["target"].astype(int).to_numpy()
    baseline_probs = predictions_df["baseline_prob"].astype(float).to_numpy()
    challenger_probs = predictions_df["challenger_prob"].astype(float).to_numpy()
    baseline_preds = (baseline_probs > 0.5).astype(int)
    challenger_preds = (challenger_probs > 0.5).astype(int)

    metrics = {
        "eligible_fights": int(len(predictions_df)),
        "baseline": {
            "accuracy": float(accuracy_score(y_true, baseline_preds)),
            "brier_score": float(brier_score_loss(y_true, baseline_probs)),
            "log_loss": float(log_loss(y_true, baseline_probs)),
            "ece": float(compute_ece(y_true, baseline_probs)),
        },
        "challenger": {
            "accuracy": float(accuracy_score(y_true, challenger_preds)),
            "brier_score": float(brier_score_loss(y_true, challenger_probs)),
            "log_loss": float(log_loss(y_true, challenger_probs)),
            "ece": float(compute_ece(y_true, challenger_probs)),
        },
    }
    metrics["mcnemar"] = mcnemar_test(y_true, baseline_preds, challenger_preds)
    metrics["brier_bootstrap"] = bootstrap_metric_comparison(
        y_true,
        baseline_probs,
        challenger_probs,
        brier_score_loss,
        n_bootstrap=bootstrap,
    )
    metrics["logloss_bootstrap"] = bootstrap_metric_comparison(
        y_true,
        baseline_probs,
        challenger_probs,
        log_loss,
        n_bootstrap=bootstrap,
    )

    if metrics["brier_bootstrap"]["ci_hi"] < 0:
        verdict = "Challenger is significantly better on strict new-data Brier score."
    elif metrics["challenger"]["brier_score"] < metrics["baseline"]["brier_score"]:
        verdict = "Challenger is better on strict new-data Brier score, but not significant."
    elif metrics["brier_bootstrap"]["ci_lo"] > 0:
        verdict = "Baseline is significantly better on strict new-data Brier score."
    else:
        verdict = "No clear strict new-data compare winner."
    metrics["verdict"] = verdict
    return metrics


def _source_summary(
    source_name: str,
    training_df: pd.DataFrame,
    *,
    legacy_keys: set[str],
    legacy_max_date: pd.Timestamp,
) -> dict[str, Any]:
    keyed = _with_fight_keys(training_df)
    overlap_count = int(keyed["fight_key"].isin(legacy_keys).sum())
    new_vs_legacy_count = int((~keyed["fight_key"].isin(legacy_keys)).sum())
    after_legacy_max_count = int((keyed["event_date"] > legacy_max_date).sum())
    strict_new_count = int(((keyed["event_date"] > legacy_max_date) & keyed["fight_key"].notna()).sum())
    return {
        "source": source_name,
        "dataset_total_fights": int(len(training_df)),
        "dataset_min_date": str(keyed["event_date"].min().date()) if not keyed.empty else None,
        "dataset_max_date": str(keyed["event_date"].max().date()) if not keyed.empty else None,
        "overlap_with_legacy": overlap_count,
        "new_vs_legacy": new_vs_legacy_count,
        "after_legacy_max": after_legacy_max_count,
        "strict_new_eval_candidates": strict_new_count,
    }


def _evaluate_source(
    source_name: str,
    training_df: pd.DataFrame,
    *,
    strict_eval_start: pd.Timestamp,
    bootstrap: int,
    retrain_months: int,
    run_dir: Path,
) -> dict[str, Any]:
    logger.info("Building features for %s (%d fights)", source_name, len(training_df))
    features_df = build_features(training_df)
    augmented_df = augment_features_with_betsapi_mma(features_df, save_artifacts=False)
    baseline_feature_cols = get_feature_family_columns(augmented_df, "production")
    challenger_feature_cols = get_feature_family_columns(
        augmented_df,
        "production_betsapi_expanded",
    )

    wf = walk_forward_comparison(
        augmented_df,
        baseline_feature_cols,
        challenger_feature_cols,
        retrain_months=retrain_months,
        start_date=TRAIN_CUTOFF_DATE,
    )
    predictions = wf["predictions"].copy()
    predictions["fight_key"] = predictions.apply(
        lambda row: _fight_key(row.get("fighter_a"), row.get("fighter_b"), row.get("event_date")),
        axis=1,
    )
    strict_predictions = predictions[pd.to_datetime(predictions["event_date"], errors="coerce") > strict_eval_start].copy()
    strict_predictions = strict_predictions.sort_values(["event_date", "fighter_a", "fighter_b"]).reset_index(drop=True)

    strict_metrics = _compute_metrics(strict_predictions, bootstrap=bootstrap)

    betsapi_cols = [column for column in BETSAPI_CHALLENGER_FEATURE_NAMES if column in augmented_df.columns]

    # --- Sliced metrics ---
    # Build a per-fight BetsAPI coverage mask aligned with strict_predictions
    betsapi_coverage_mask = None
    if not strict_predictions.empty and betsapi_cols:
        # Match predictions back to augmented feature rows by fight_key
        feat_keyed = augmented_df.copy()
        feat_keyed["event_date"] = pd.to_datetime(feat_keyed["event_date"], errors="coerce")
        feat_keyed = feat_keyed[feat_keyed["event_date"] > strict_eval_start].copy()
        feat_keyed["fight_key"] = feat_keyed.apply(
            lambda row: _fight_key(row.get("fighter_a"), row.get("fighter_b"), row.get("event_date")),
            axis=1,
        )
        feat_keyed = feat_keyed.drop_duplicates(subset="fight_key")
        feat_coverage = feat_keyed.set_index("fight_key")[betsapi_cols].notna().any(axis=1)
        # Align to strict_predictions by fight_key
        betsapi_coverage_mask = strict_predictions["fight_key"].map(feat_coverage).fillna(False).astype(bool)
        betsapi_coverage_mask = betsapi_coverage_mask.reset_index(drop=True)

    sliced_metrics = _compute_sliced_metrics(
        strict_predictions,
        betsapi_coverage_mask=betsapi_coverage_mask,
    )

    strict_fold_df = pd.DataFrame()
    if not strict_predictions.empty:
        strict_fold_df = (
            strict_predictions.groupby("fold", dropna=False)
            .agg(
                fights=("target", "size"),
                baseline_brier=("baseline_prob", lambda series: brier_score_loss(strict_predictions.loc[series.index, "target"], series)),
                challenger_brier=("challenger_prob", lambda series: brier_score_loss(strict_predictions.loc[series.index, "target"], series)),
                baseline_accuracy=(
                    "baseline_prob",
                    lambda series: accuracy_score(strict_predictions.loc[series.index, "target"], (series > 0.5).astype(int)),
                ),
                challenger_accuracy=(
                    "challenger_prob",
                    lambda series: accuracy_score(strict_predictions.loc[series.index, "target"], (series > 0.5).astype(int)),
                ),
            )
            .reset_index()
        )
    strict_feature_rows = augmented_df[pd.to_datetime(augmented_df["event_date"], errors="coerce") > strict_eval_start].copy()
    strict_feature_rows["fight_key"] = strict_feature_rows.apply(
        lambda row: _fight_key(row.get("fighter_a"), row.get("fighter_b"), row.get("event_date")),
        axis=1,
    )
    strict_feature_rows = strict_feature_rows.drop_duplicates(subset="fight_key")
    any_betsapi = strict_feature_rows[betsapi_cols].notna().any(axis=1) if betsapi_cols else pd.Series(False, index=strict_feature_rows.index)
    coverage = {
        "strict_feature_rows": int(len(strict_feature_rows)),
        "strict_rows_with_any_betsapi": int(any_betsapi.sum()),
        "strict_rows_with_any_betsapi_pct": _safe_pct(int(any_betsapi.sum()), int(len(strict_feature_rows))),
    }

    strict_predictions.to_csv(run_dir / f"{source_name}_strict_predictions.csv", index=False)
    strict_fold_df.to_csv(run_dir / f"{source_name}_strict_folds.csv", index=False)
    (run_dir / f"{source_name}_strict_metrics.json").write_text(
        json.dumps(strict_metrics, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    # Save sliced metrics
    (run_dir / f"{source_name}_sliced_metrics.json").write_text(
        json.dumps(sliced_metrics, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    return {
        "source": source_name,
        "dataset_total_fights": int(len(training_df)),
        "feature_rows": int(len(features_df)),
        "strict_metrics": strict_metrics,
        "sliced_metrics": sliced_metrics,
        "strict_fold_df": strict_fold_df,
        "coverage": coverage,
        "baseline_feature_count": int(len(baseline_feature_cols)),
        "challenger_feature_count": int(len(challenger_feature_cols)),
    }


def _render_sliced_metrics_block(result_payloads: list[dict[str, Any]]) -> str:
    """Render a markdown section with per-slice tables for each dataset variant."""
    if not result_payloads:
        return ""

    # Collect all slice names across variants
    all_slices: set[str] = set()
    for p in result_payloads:
        all_slices.update(p.get("sliced_metrics", {}).keys())

    if not all_slices:
        return "(no sliced metrics computed)\n"

    blocks = []
    for slice_name in sorted(all_slices):
        rows = []
        for p in result_payloads:
            sm = p.get("sliced_metrics", {}).get(slice_name)
            if sm is None:
                continue
            rows.append(
                {
                    "source": p["source"],
                    "fights": sm["fights"],
                    "baseline_brier": round(sm["baseline_brier"], 4),
                    "challenger_brier": round(sm["challenger_brier"], 4),
                    "baseline_acc": round(sm["baseline_accuracy"], 4),
                    "challenger_acc": round(sm["challenger_accuracy"], 4),
                }
            )
        if rows:
            blocks.append(f"### {slice_name}\n\n{_table_block(pd.DataFrame(rows))}\n")

    return "\n".join(blocks)


def _render_shortlist_block(shortlisted: list[dict[str, Any]]) -> str:
    """Render the automated shortlist recommendation section."""
    if not shortlisted:
        return "No variant met the shortlist criteria (must beat append_only_2026 on Brier, not regress on log loss, and have >20 fights).\n"

    lines = ["Recommended variant(s) for promotion:\n"]
    for entry in shortlisted:
        lines.append(
            f"- **`{entry['source']}`**: {entry['reason']}"
        )
    return "\n".join(lines) + "\n"


def _render_report(
    *,
    run_dir: Path,
    overlap_df: pd.DataFrame,
    field_audit_df: pd.DataFrame,
    results_df: pd.DataFrame,
    result_payloads: list[dict[str, Any]],
    strict_eval_start: pd.Timestamp,
    legacy_max_date: pd.Timestamp,
    audit_summary: dict[str, Any],
    variant_summary_df: pd.DataFrame | None = None,
    shortlisted: list[dict[str, Any]] | None = None,
) -> str:
    report_rows = []
    for payload in result_payloads:
        metrics = payload["strict_metrics"]
        report_rows.append(
            {
                "source": payload["source"],
                "eligible_fights": metrics["eligible_fights"],
                "baseline_brier": metrics["baseline"].get("brier_score"),
                "challenger_brier": metrics["challenger"].get("brier_score"),
                "baseline_ece": metrics["baseline"].get("ece"),
                "challenger_ece": metrics["challenger"].get("ece"),
                "baseline_accuracy": metrics["baseline"].get("accuracy"),
                "challenger_accuracy": metrics["challenger"].get("accuracy"),
                "strict_rows_with_any_betsapi": payload["coverage"]["strict_rows_with_any_betsapi"],
                "strict_rows_with_any_betsapi_pct": payload["coverage"]["strict_rows_with_any_betsapi_pct"],
                "verdict": metrics["verdict"],
            }
        )
    strict_results_df = pd.DataFrame(report_rows)

    variant_summary_section = ""
    if variant_summary_df is not None and not variant_summary_df.empty:
        variant_summary_section = f"""## Dataset Variant Summary

{_table_block(variant_summary_df)}

"""

    sliced_section = _render_sliced_metrics_block(result_payloads)
    shortlist_section = _render_shortlist_block(shortlisted or [])

    return f"""# UFC Source Compare Report

Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Run directory: `{run_dir}`

## Strict Evaluation Rule

This report only grades the pulled-data experiment on fights **after** the legacy dataset cutoff.
- Legacy max date: `{legacy_max_date.date().isoformat()}`
- Strict fresh-data evaluation window: fights after `{strict_eval_start.date().isoformat()}`

## Overlap Audit

{_table_block(overlap_df)}

Interpretation:
- `append_only_2026` is the frozen control arm: legacy history plus only genuinely new fights after the legacy cutoff.
- `hybrid_legacy_first` keeps legacy overlap rows as-is and appends only pulled-only fights.
- `hybrid_pulled_2022plus` swaps the 2022+ window to pulled UFCStats rows outright.
- `best_of_both_field_level` keeps legacy metadata/odds richness while pulling in official UFCStats fight-result and per-fight stat fields.
- `best_of_both_full_history` applies that same field-level merge across the full legacy history so the real pre-2022 train split can use honest UFCStats fight stats.

## Field-Level Audit

Overlap window: `{audit_summary.get("start_date")}` through `{audit_summary.get("end_date")}`
- Overlap fights audited: `{audit_summary.get("overlap_fights")}`
- Legacy rows in window: `{audit_summary.get("legacy_window_fights")}`
- Pulled rows in window: `{audit_summary.get("pulled_window_fights")}`

{_table_block(field_audit_df)}

{variant_summary_section}## Strict New-Data Results

{_table_block(results_df)}

## Sliced Metrics

{sliced_section}

## Verdicts

{"".join(f"- `{payload['source']}`: {payload['strict_metrics']['verdict']}\n" for payload in result_payloads)}

## Shortlist Recommendation

{shortlist_section}
"""


def run_ufc_source_compare(
    *,
    retrain_months: int = 4,
    bootstrap: int = 5000,
    skip_models: bool = False,
) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = SOURCE_COMPARE_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    legacy_df = load_kaggle_dataset(RAW_DATA_DIR / "ufc-master.csv")
    legacy_keyed = _with_fight_keys(legacy_df)
    legacy_keys = set(legacy_keyed["fight_key"].dropna())
    legacy_max_date = pd.to_datetime(legacy_df["event_date"]).max()

    variants = build_training_dataset_variants(legacy_df=legacy_df)
    pulled_df = variants["pulled_all"]
    strict_eval_start = pd.Timestamp(legacy_max_date)
    overlap_audit = audit_legacy_pulled_overlap(legacy_df, pulled_df)
    field_audit_df = overlap_audit["field_summary"]
    fight_audit_df = overlap_audit["fight_level"]

    overlap_df = pd.DataFrame(
        [
            _source_summary("legacy_only", legacy_df, legacy_keys=legacy_keys, legacy_max_date=legacy_max_date),
            *[
                _source_summary(name, training_df, legacy_keys=legacy_keys, legacy_max_date=legacy_max_date)
                for name, training_df in variants.items()
                if name != "legacy_only"
            ],
        ]
    )
    overlap_df.to_csv(run_dir / "source_overlap_summary.csv", index=False)
    field_audit_df.to_csv(run_dir / "overlap_field_audit.csv", index=False)
    fight_audit_df.to_csv(run_dir / "overlap_fight_audit.csv", index=False)
    (run_dir / "merge_rules.json").write_text(
        json.dumps(overlap_audit["merge_rules"], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Build the variant summary table (available even when skip_models=True)
    variant_summary_df = _build_variant_summary_table(
        variants,
        legacy_keys=legacy_keys,
        legacy_max_date=legacy_max_date,
    )
    variant_summary_df.to_csv(run_dir / "variant_summary.csv", index=False)

    results = []
    if not skip_models:
        for source_name in TRAINING_DATASET_VARIANTS:
            if source_name == "legacy_only":
                continue
            training_df = variants[source_name]
            results.append(
                _evaluate_source(
                    source_name,
                    training_df,
                    strict_eval_start=strict_eval_start,
                    bootstrap=bootstrap,
                    retrain_months=retrain_months,
                    run_dir=run_dir,
                )
            )

    results_df = pd.DataFrame(
        [
            {
                "source": payload["source"],
                "eligible_fights": payload["strict_metrics"]["eligible_fights"],
                "baseline_brier": payload["strict_metrics"]["baseline"].get("brier_score"),
                "challenger_brier": payload["strict_metrics"]["challenger"].get("brier_score"),
                "baseline_log_loss": payload["strict_metrics"]["baseline"].get("log_loss"),
                "challenger_log_loss": payload["strict_metrics"]["challenger"].get("log_loss"),
                "baseline_ece": payload["strict_metrics"]["baseline"].get("ece"),
                "challenger_ece": payload["strict_metrics"]["challenger"].get("ece"),
                "baseline_accuracy": payload["strict_metrics"]["baseline"].get("accuracy"),
                "challenger_accuracy": payload["strict_metrics"]["challenger"].get("accuracy"),
                "strict_rows_with_any_betsapi": payload["coverage"]["strict_rows_with_any_betsapi"],
                "strict_rows_with_any_betsapi_pct": payload["coverage"]["strict_rows_with_any_betsapi_pct"],
                "verdict": payload["strict_metrics"]["verdict"],
            }
            for payload in results
        ]
    )
    results_df.to_csv(run_dir / "strict_new_data_results.csv", index=False)

    # Automated shortlisting
    shortlisted = shortlist_variants(results) if results else []
    if shortlisted:
        (run_dir / "shortlist.json").write_text(
            json.dumps(shortlisted, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    report_path = run_dir / "report.md"
    report_path.write_text(
        _render_report(
            run_dir=run_dir,
            overlap_df=overlap_df,
            field_audit_df=field_audit_df,
            results_df=results_df,
            result_payloads=results,
            strict_eval_start=strict_eval_start,
            legacy_max_date=legacy_max_date,
            audit_summary=overlap_audit["summary"],
            variant_summary_df=variant_summary_df,
            shortlisted=shortlisted,
        ),
        encoding="utf-8",
    )

    metadata = {
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "strict_eval_start": strict_eval_start.date().isoformat(),
        "results_path": str(run_dir / "strict_new_data_results.csv"),
        "skip_models": bool(skip_models),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Compare legacy UFC data versus strict new pulled-data sources.")
    parser.add_argument("--retrain-months", type=int, default=4, help="Walk-forward retrain interval")
    parser.add_argument("--bootstrap", type=int, default=5000, help="Bootstrap iterations for strict-window metrics")
    parser.add_argument("--skip-models", action="store_true", help="Only build dataset summaries and overlap audits")
    args = parser.parse_args()

    result = run_ufc_source_compare(
        retrain_months=args.retrain_months,
        bootstrap=args.bootstrap,
        skip_models=args.skip_models,
    )
    logger.info("Saved UFC source-compare report to %s", result["report_path"])


if __name__ == "__main__":
    main()
