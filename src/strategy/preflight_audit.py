"""Generate a pre-run audit packet for the full UFC evaluation matrix.

This script is intentionally conservative. It is meant to be run before any
expensive matrix evaluation so another reviewer can audit the exact inputs and
readiness state first.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import BETSAPI_MMA_PROCESSED_DIR, BETSAPI_MMA_RAW_DIR, LOGS_DIR
from src.data.betsapi_mma import summarize_saved_betsapi_mma_backfill
from src.data.ufc_refresh import TRAINING_DATASET_VARIANTS
from src.strategy.compare_matrix import ALL_FEATURE_FAMILIES
from src.strategy.control_arm import (
    load_frozen_control_metrics,
    load_frozen_control_trading_artifacts,
    load_frozen_sweep_summary,
    validate_frozen_control_arm,
    validate_frozen_control_arm_for_promotion_gate,
    validate_frozen_control_arm_for_selection_gate,
)
from src.strategy.model_variants import ALL_VARIANTS
from src.strategy.run_evaluation import (
    DEFAULT_FREEZE_ID,
    _collect_runtime_code_metadata,
    _default_feature_families,
    _default_variant_names,
    HISTORICAL_MATRIX_EXCLUDED_FAMILIES,
    HISTORICAL_MATRIX_EXCLUDED_VARIANTS,
    validate_betsapi_preflight_readiness,
)


def collect_betsapi_status(
    *,
    raw_root: Path = BETSAPI_MMA_RAW_DIR,
    processed_root: Path = BETSAPI_MMA_PROCESSED_DIR,
) -> dict[str, Any]:
    """Summarize local MMA BetsAPI backfill completeness."""
    return summarize_saved_betsapi_mma_backfill(
        raw_root=raw_root,
        processed_root=processed_root,
    )


def collect_freeze_status(freeze_id: str) -> dict[str, Any]:
    """Summarize frozen control-arm readiness."""
    integrity = validate_frozen_control_arm(freeze_id)
    selection = validate_frozen_control_arm_for_selection_gate(freeze_id)
    promotion = validate_frozen_control_arm_for_promotion_gate(freeze_id)

    metrics = None
    sweep = None
    trading_rows: dict[str, int] = {}
    if integrity["valid"]:
        metrics = load_frozen_control_metrics(freeze_id)
        sweep = load_frozen_sweep_summary(freeze_id)
        artifacts = load_frozen_control_trading_artifacts(freeze_id)
        trading_rows = {
            name: int(len(frame))
            for name, frame in artifacts.items()
        }

    return {
        "freeze_id": freeze_id,
        "integrity": integrity,
        "selection": selection,
        "promotion": promotion,
        "metrics": (
            {
                "accuracy": metrics.get("accuracy"),
                "brier": metrics.get("brier"),
                "log_loss": metrics.get("log_loss"),
                "ece": metrics.get("ece"),
                "fresh_data_brier": metrics.get(
                    "fresh_data_brier",
                    metrics.get("sliced_metrics", {}).get("fresh_window", {}).get("brier"),
                ),
                "year_by_year_keys": sorted(
                    (metrics.get("year_by_year") or metrics.get("sliced_metrics", {}).get("by_year") or {}).keys()
                )
                if isinstance(
                    metrics.get("year_by_year") or metrics.get("sliced_metrics", {}).get("by_year"),
                    dict,
                )
                else None,
            }
            if metrics is not None
            else None
        ),
        "sweep": (
            {
                "config": sweep.get("config") if sweep else None,
                "roi": sweep.get("roi") if sweep else None,
                "total_bets": sweep.get("total_bets") if sweep else None,
                "max_drawdown": sweep.get("max_drawdown") if sweep else None,
                "is_baseline": sweep.get("is_baseline") if sweep else None,
            }
            if sweep is not None
            else None
        ),
        "trading_rows": trading_rows,
    }


def collect_matrix_status(*, has_betsapi_backfill: bool) -> dict[str, Any]:
    """Summarize the matrix shape that would run right now."""
    all_variants = sorted(ALL_VARIANTS)
    all_families = list(ALL_FEATURE_FAMILIES)
    datasets = list(TRAINING_DATASET_VARIANTS)
    default_variants = _default_variant_names(
        has_betsapi_backfill=has_betsapi_backfill,
    )
    default_families = _default_feature_families(
        has_betsapi_backfill=has_betsapi_backfill,
    )

    return {
        "has_betsapi_backfill": has_betsapi_backfill,
        "datasets": datasets,
        "all_variant_count": len(all_variants),
        "all_family_count": len(all_families),
        "default_variant_count": len(default_variants),
        "default_family_count": len(default_families),
        "full_matrix_cells": len(all_variants) * len(all_families) * len(datasets),
        "default_matrix_cells": len(default_variants) * len(default_families) * len(datasets),
        "excluded_variants": sorted(set(all_variants) - set(default_variants)),
        "excluded_families": sorted(set(all_families) - set(default_families)),
        "default_variants": default_variants,
        "default_families": default_families,
    }


def collect_betsapi_runtime_validation(*, has_betsapi_backfill: bool) -> dict[str, Any]:
    """Run a lightweight per-dataset BetsAPI signal check for preflight."""
    if not has_betsapi_backfill:
        return {
            "ready": False,
            "checked": False,
            "reason": "betsapi_backfill_not_ready",
        }

    default_families = _default_feature_families(
        has_betsapi_backfill=has_betsapi_backfill,
    )
    return validate_betsapi_preflight_readiness(
        dataset_names=list(TRAINING_DATASET_VARIANTS),
        feature_families=default_families,
    )


def collect_preflight_status(freeze_id: str) -> dict[str, Any]:
    """Collect the full preflight status payload."""
    betsapi_status = collect_betsapi_status()
    matrix_status = collect_matrix_status(
        has_betsapi_backfill=betsapi_status["ready"],
    )
    betsapi_runtime_validation = collect_betsapi_runtime_validation(
        has_betsapi_backfill=betsapi_status["ready"],
    )
    freeze_status = collect_freeze_status(freeze_id)
    runtime = _collect_runtime_code_metadata()

    ready_for_full_matrix = bool(
        betsapi_status["ready"]
        and betsapi_runtime_validation["ready"]
        and freeze_status["selection"]["ready"]
        and freeze_status["promotion"]["ready"]
        and runtime.get("git_dirty") is False
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ready_for_full_matrix": ready_for_full_matrix,
        "freeze_status": freeze_status,
        "betsapi_status": betsapi_status,
        "betsapi_runtime_validation": betsapi_runtime_validation,
        "matrix_status": matrix_status,
        "runtime": runtime,
    }


def render_preflight_report(status: dict[str, Any]) -> str:
    """Render a markdown report for a human reviewer."""
    freeze_status = status["freeze_status"]
    betsapi_status = status["betsapi_status"]
    betsapi_runtime_validation = status["betsapi_runtime_validation"]
    matrix_status = status["matrix_status"]
    runtime = status["runtime"]

    lines = [
        "# UFC Matrix Preflight Audit",
        "",
        f"Generated at: {status['generated_at']}",
        "",
        "## Overall Readiness",
        f"- Ready for full matrix: {status['ready_for_full_matrix']}",
        f"- Freeze selection-gate ready: {freeze_status['selection']['ready']}",
        f"- Freeze promotion-gate ready: {freeze_status['promotion']['ready']}",
        f"- BetsAPI backfill ready: {betsapi_status['ready']}",
        f"- BetsAPI runtime validation ready: {betsapi_runtime_validation['ready']}",
        "",
        "## Frozen Control Arm",
        f"- Freeze ID: {freeze_status['freeze_id']}",
        f"- Integrity valid: {freeze_status['integrity']['valid']}",
        f"- Selection-gate errors: {freeze_status['selection']['errors'] or ['none']}",
        f"- Promotion-gate errors: {freeze_status['promotion']['errors'] or ['none']}",
        f"- Metrics summary: {freeze_status['metrics']}",
        f"- Frozen sweep summary: {freeze_status['sweep']}",
        f"- Frozen trading row counts: {freeze_status['trading_rows']}",
        "",
        "## BetsAPI MMA Backfill",
        f"- Raw ended-event payload files: {betsapi_status['raw_event_payload_files']}",
        f"- Raw odds-summary payload files: {betsapi_status['raw_summary_payload_files']}",
        f"- Raw odds-summary payload files (all JSON): {betsapi_status.get('raw_summary_payload_files_total')}",
        f"- Non-canonical odds-summary payload files: {betsapi_status.get('raw_summary_payload_files_noncanonical')}",
        f"- Requested UFC event IDs: {betsapi_status['requested_ufc_event_ids']}",
        f"- Hydrated summary IDs: {betsapi_status['hydrated_summary_ids']}",
        f"- Missing summary IDs: {betsapi_status['missing_summary_ids']}",
        f"- Missing summary sample: {betsapi_status['missing_summary_sample']}",
        f"- Usable normalized summary event IDs: {betsapi_status.get('usable_summary_event_ids')}",
        f"- Usable event snapshot IDs: {betsapi_status.get('usable_event_snapshot_ids')}",
        f"- Invalid moneyline rows in processed summary data: {betsapi_status.get('invalid_moneyline_rows')}",
        f"- Invalid moneyline event IDs: {betsapi_status.get('invalid_moneyline_event_ids')}",
        f"- Invalid moneyline bookmakers: {betsapi_status.get('invalid_moneyline_bookmakers')}",
        f"- Processed row counts: {betsapi_status.get('processed_row_counts')}",
        f"- Manifest matches processed counts: {betsapi_status.get('manifest_matches_processed_counts')}",
        f"- Manifest count mismatches: {betsapi_status.get('manifest_count_mismatches')}",
        f"- Manifest is fresh vs processed CSVs: {betsapi_status.get('manifest_is_fresh')}",
        f"- Processed artifacts present: {betsapi_status['processed_artifacts']}",
        f"- Backfill manifest: {betsapi_status['manifest']}",
        f"- Backfill quality errors: {betsapi_status.get('quality_errors') or ['none']}",
        "",
        "## BetsAPI Readiness Validation",
        f"- Checked: {betsapi_runtime_validation.get('checked')}",
        f"- Ready: {betsapi_runtime_validation.get('ready')}",
        f"- Mode: {betsapi_runtime_validation.get('mode')}",
        f"- Families: {betsapi_runtime_validation.get('families')}",
        f"- Datasets: {betsapi_runtime_validation.get('datasets')}",
        f"- Representative profiles: {betsapi_runtime_validation.get('representative_profiles')}",
        f"- Error: {betsapi_runtime_validation.get('error')}",
        "",
        "## Matrix Composition",
        f"- Datasets: {matrix_status['datasets']}",
        f"- All variants: {matrix_status['all_variant_count']}",
        f"- All feature families: {matrix_status['all_family_count']}",
        f"- Full matrix cells when fully ready: {matrix_status['full_matrix_cells']}",
        f"- Default variants active right now: {matrix_status['default_variant_count']}",
        f"- Default feature families active right now: {matrix_status['default_family_count']}",
        f"- Default matrix cells right now: {matrix_status['default_matrix_cells']}",
        f"- Excluded variants: {matrix_status['excluded_variants']}",
        f"- Excluded families: {matrix_status['excluded_families']}",
        "",
        "## Runtime Fingerprint",
        f"- Git SHA: {runtime.get('git_sha')}",
        f"- Git dirty: {runtime.get('git_dirty')}",
        f"- Source fingerprint: {runtime.get('source_fingerprint')}",
        "",
        "## Commands",
        "- Continue/resume BetsAPI backfill:",
        "  `./.venv/Scripts/python.exe -m src.bot betsapi-mma-backfill --start-date 2022-01-01 --odds-summary --ufc-only`",
        "- Smoke run after readiness passes:",
        f"  `./.venv/Scripts/python.exe -m src.strategy.run_evaluation --freeze-id {DEFAULT_FREEZE_ID} --variants baseline,combined_v3,timeseries_sigmoid_cal --datasets append_only_2026 --families production --retrain-months 4 --max-finalists 2 --bootstrap 25 --max-workers 2 --no-narrow`",
        "- Full fresh run after preflight and code audit both pass:",
        f"  `./.venv/Scripts/python.exe -m src.strategy.run_evaluation --freeze-id {DEFAULT_FREEZE_ID} --retrain-months 4 --max-workers 2`",
        "",
        "## Audit Notes",
        "- Do not use `--resume` or `--allow-resume-mismatch` for the new evaluation.",
        "- Do not pass a global `--calibration-method` override unless intentionally forcing it.",
        "- A clean git tree is required, but preflight must also show zero BetsAPI quality errors before a full matrix run.",
        "- The historical evaluation matrix intentionally excludes live snapshot-dependent BetsAPI scope "
        f"({sorted(HISTORICAL_MATRIX_EXCLUDED_VARIANTS)}; {sorted(HISTORICAL_MATRIX_EXCLUDED_FAMILIES)}).",
        "- `bet365_upcoming` and saved prematch snapshots are for live/prematch workflows, not historical matrix readiness.",
        "- The full matrix should not start until BetsAPI backfill is ready and this report says `Ready for full matrix: True`.",
    ]
    return "\n".join(lines) + "\n"


def write_preflight_artifacts(
    *,
    status: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write the markdown and JSON audit artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    markdown_path = output_dir / f"matrix_preflight_{stamp}.md"
    json_path = output_dir / f"matrix_preflight_{stamp}.json"
    markdown_path.write_text(render_preflight_report(status), encoding="utf-8")
    json_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    return markdown_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a pre-run audit packet for the UFC matrix.")
    parser.add_argument("--freeze-id", type=str, default=DEFAULT_FREEZE_ID)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(LOGS_DIR / "preflight"),
        help="Directory where markdown/json audit artifacts will be written.",
    )
    args = parser.parse_args()

    status = collect_preflight_status(args.freeze_id)
    markdown_path, json_path = write_preflight_artifacts(
        status=status,
        output_dir=Path(args.output_dir),
    )
    print(f"ready_for_full_matrix={status['ready_for_full_matrix']}")
    print(f"markdown_report={markdown_path}")
    print(f"json_report={json_path}")


if __name__ == "__main__":
    main()
