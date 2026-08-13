"""Frozen control arm management.

Provides utilities to freeze, validate, and load control arm artifacts
used as the baseline for selection gate and promotion gate evaluation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

FROZEN_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "frozen"
REQUIRED_METRICS = ("accuracy", "brier", "log_loss", "ece")


def _sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_freeze_id(freeze_id: str) -> str:
    """Accept either ``20260313`` or ``control_arm_20260313``."""
    freeze_id = freeze_id.strip()
    if not freeze_id:
        raise ValueError("freeze_id must be a non-empty string")
    return freeze_id if freeze_id.startswith("control_arm_") else f"control_arm_{freeze_id}"


def _control_arm_dir(freeze_id: str) -> Path:
    """Return the directory for a given freeze ID."""
    return FROZEN_DIR / _normalize_freeze_id(freeze_id)


def _collect_checksums(arm_dir: Path) -> dict[str, str]:
    """Return checksums for every frozen artifact except metadata files."""
    checksums: dict[str, str] = {}
    for artifact in sorted(arm_dir.iterdir()):
        if not artifact.is_file():
            continue
        if artifact.name in {"checksums.json", "MANIFEST.md"}:
            continue
        checksums[artifact.name] = _sha256(artifact)
    return checksums


def _write_manifest(arm_dir: Path, freeze_id: str, checksums: dict[str, str]) -> None:
    """Write a human-readable manifest for a frozen control arm."""
    normalized_id = _normalize_freeze_id(freeze_id)
    lines = [
        f"# Control Arm Freeze: {normalized_id}",
        "",
        f"Frozen on: {normalized_id[-8:-4]}-{normalized_id[-4:-2]}-{normalized_id[-2:]}",
        "Source: production baseline model (isotonic calibration, walk-forward 4-month retrain)",
        "",
        "## Artifacts",
    ]
    for name in sorted(checksums):
        lines.append(f"- {name}")
    lines += ["", "## Checksums", "See checksums.json for SHA256 integrity verification.", ""]
    (arm_dir / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


def _coerce_artifact_frame(
    artifact: pd.DataFrame | list[dict[str, Any]] | list[float] | None,
    *,
    default_column: str,
) -> pd.DataFrame | None:
    """Normalize optional frozen trading artifacts to a DataFrame."""
    if artifact is None:
        return None
    if isinstance(artifact, pd.DataFrame):
        return artifact.copy()
    if isinstance(artifact, list):
        if not artifact:
            return pd.DataFrame(columns=[default_column])
        if isinstance(artifact[0], dict):
            return pd.DataFrame(artifact)
        return pd.DataFrame({default_column: list(artifact)})
    raise TypeError(f"Unsupported artifact type: {type(artifact)!r}")


def _load_optional_csv(path: Path) -> pd.DataFrame | None:
    """Load an optional CSV artifact and preserve event dates when present."""
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "event_date" in frame.columns:
        frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    return frame


def _extract_fresh_window_brier(metrics_payload: dict[str, Any]) -> float | None:
    """Return the frozen fresh-window Brier metric when present."""
    top_level = metrics_payload.get("fresh_data_brier")
    if top_level is not None:
        return float(top_level)
    nested = (
        metrics_payload.get("sliced_metrics", {})
        .get("fresh_window", {})
        .get("brier")
    )
    if nested is None:
        return None
    return float(nested)


def _extract_year_by_year_metrics(metrics_payload: dict[str, Any]) -> Any:
    """Return frozen year-by-year metrics when present."""
    raw = metrics_payload.get("year_by_year")
    if raw:
        return raw
    nested = metrics_payload.get("sliced_metrics", {}).get("by_year")
    return nested if nested else None


def _csv_has_rows(path: Path) -> bool:
    """Return True when a CSV artifact exists and contains at least one row."""
    frame = _load_optional_csv(path)
    return bool(frame is not None and not frame.empty)


def _make_json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats so frozen JSON stays standards-compliant."""
    if isinstance(value, dict):
        return {key: _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def freeze_control_arm(
    source_metrics: dict[str, Any],
    freeze_id: str,
    sweep_summary: dict[str, Any] | None = None,
    artifact_sources: dict[str, str | Path] | None = None,
    bet_log: pd.DataFrame | list[dict[str, Any]] | None = None,
    bankroll_history: pd.DataFrame | list[dict[str, Any]] | list[float] | None = None,
) -> Path:
    """Create a new dated frozen control arm.

    Parameters
    ----------
    source_metrics : dict
        Model evaluation metrics (must include accuracy, brier, log_loss, ece).
    freeze_id : str
        Date-based identifier, e.g. "20260313".
    sweep_summary : dict | None
        Optional trading sweep summary to freeze alongside model metrics.
    artifact_sources : dict[str, str | Path] | None
        Optional extra artifacts to copy into the freeze directory. Keys are the
        frozen filenames to create and values are source paths.
    bet_log : pd.DataFrame | list[dict[str, Any]] | None
        Optional baseline bet log to freeze as ``control_bet_log.csv``.
    bankroll_history : pd.DataFrame | list[dict[str, Any]] | list[float] | None
        Optional baseline bankroll history to freeze as
        ``control_bankroll_history.csv``.

    Returns
    -------
    Path
        Directory containing the frozen artifacts.
    """
    missing = [m for m in REQUIRED_METRICS if m not in source_metrics]
    if missing:
        raise ValueError(f"Source metrics missing required keys: {missing}")

    normalized_id = _normalize_freeze_id(freeze_id)
    arm_dir = _control_arm_dir(normalized_id)
    if arm_dir.exists():
        raise FileExistsError(f"Freeze directory already exists: {arm_dir}")
    arm_dir.mkdir(parents=True, exist_ok=False)

    metrics_path = arm_dir / "control_metrics.json"
    metrics_payload = _make_json_safe(dict(source_metrics))
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2, sort_keys=True, allow_nan=False)

    if sweep_summary is not None:
        sweep_path = arm_dir / "control_sweep_summary.json"
        sweep_payload = _make_json_safe(dict(sweep_summary))
        with open(sweep_path, "w") as f:
            json.dump(sweep_payload, f, indent=4, allow_nan=False)

    if artifact_sources:
        for target_name, source_path in artifact_sources.items():
            shutil.copy2(Path(source_path), arm_dir / target_name)

    bet_log_frame = _coerce_artifact_frame(bet_log, default_column="profit")
    if bet_log_frame is not None:
        bet_log_frame.to_csv(arm_dir / "control_bet_log.csv", index=False)

    bankroll_frame = _coerce_artifact_frame(
        bankroll_history,
        default_column="combined",
    )
    if bankroll_frame is not None:
        bankroll_frame.to_csv(arm_dir / "control_bankroll_history.csv", index=False)

    checksums = _collect_checksums(arm_dir)

    with open(arm_dir / "checksums.json", "w") as f:
        json.dump(checksums, f, indent=4)

    _write_manifest(arm_dir, normalized_id, checksums)

    logger.info("Frozen control arm %s at %s", normalized_id, arm_dir)
    return arm_dir


def validate_frozen_control_arm(freeze_id: str) -> dict[str, Any]:
    """Validate a frozen control arm's integrity.

    Checks directory existence, required files, and SHA256 checksums.

    Returns
    -------
    dict with keys: valid (bool), errors (list[str]), freeze_id, path
    """
    normalized_id = _normalize_freeze_id(freeze_id)
    errors: list[str] = []
    arm_dir = _control_arm_dir(normalized_id)

    if not arm_dir.exists():
        return {
            "valid": False,
            "passes": False,
            "errors": [f"Control arm directory not found: {arm_dir}"],
            "freeze_id": normalized_id,
            "freeze_dir": str(arm_dir),
            "path": str(arm_dir),
        }

    for fname in ["control_metrics.json", "checksums.json", "MANIFEST.md"]:
        if not (arm_dir / fname).exists():
            errors.append(f"Missing required file: {fname}")

    if errors:
        return {
            "valid": False,
            "passes": False,
            "errors": errors,
            "freeze_id": normalized_id,
            "freeze_dir": str(arm_dir),
            "path": str(arm_dir),
        }

    # Verify checksums
    with open(arm_dir / "checksums.json") as f:
        expected = json.load(f)

    for fname, expected_hash in expected.items():
        fpath = arm_dir / fname
        if not fpath.exists():
            errors.append(f"Checksummed file missing: {fname}")
            continue
        actual_hash = _sha256(fpath)
        if actual_hash != expected_hash:
            errors.append(
                f"Checksum mismatch for {fname}: "
                f"expected {expected_hash[:12]}..., got {actual_hash[:12]}..."
            )

    # Validate metrics payload
    metrics_path = arm_dir / "control_metrics.json"
    try:
        with open(metrics_path) as f:
            payload = json.load(f)
        missing = [k for k in REQUIRED_METRICS if k not in payload]
        if missing:
            errors.append(f"Metrics missing required keys: {missing}")
    except (json.JSONDecodeError, ValueError) as exc:
        errors.append(f"Invalid control metrics: {exc}")

    passes = len(errors) == 0
    return {
        "valid": passes,
        "passes": passes,
        "errors": errors,
        "freeze_id": normalized_id,
        "freeze_dir": str(arm_dir),
        "path": str(arm_dir),
    }


def load_frozen_control_metrics(freeze_id: str) -> dict[str, Any]:
    """Load metrics from a frozen control arm.

    Validates integrity before loading.

    Raises
    ------
    FileNotFoundError
        If the frozen control arm doesn't exist.
    ValueError
        If integrity validation fails.
    """
    validation = validate_frozen_control_arm(freeze_id)
    if not validation["valid"]:
        if any("not found" in e for e in validation["errors"]):
            raise FileNotFoundError(
                f"Frozen control arm {freeze_id} not found at {validation['path']}"
            )
        raise ValueError(
            f"Frozen control arm {freeze_id} integrity check failed: {validation['errors']}"
        )

    with open(_control_arm_dir(freeze_id) / "control_metrics.json") as f:
        payload = json.load(f)
    loaded = dict(payload)
    for metric in REQUIRED_METRICS:
        loaded[metric] = float(payload[metric])
    return loaded


def load_frozen_sweep_summary(freeze_id: str) -> dict[str, Any] | None:
    """Load sweep summary from a frozen control arm, if present."""
    validation = validate_frozen_control_arm(freeze_id)
    if not validation["valid"]:
        raise ValueError(
            f"Frozen control arm {freeze_id} integrity check failed: {validation['errors']}"
        )

    sweep_path = _control_arm_dir(freeze_id) / "control_sweep_summary.json"
    if not sweep_path.exists():
        return None

    with open(sweep_path) as f:
        return json.load(f)


def load_frozen_control_trading_artifacts(freeze_id: str) -> dict[str, pd.DataFrame]:
    """Load optional baseline trading artifacts from a frozen control arm."""
    validation = validate_frozen_control_arm(freeze_id)
    if not validation["valid"]:
        raise ValueError(
            f"Frozen control arm {freeze_id} integrity check failed: {validation['errors']}"
        )

    arm_dir = _control_arm_dir(freeze_id)
    artifacts: dict[str, pd.DataFrame] = {}
    bet_log = _load_optional_csv(arm_dir / "control_bet_log.csv")
    if bet_log is not None:
        artifacts["bet_log"] = bet_log
    bankroll_history = _load_optional_csv(arm_dir / "control_bankroll_history.csv")
    if bankroll_history is not None:
        artifacts["bankroll_history"] = bankroll_history
    return artifacts


def validate_frozen_control_arm_for_selection_gate(freeze_id: str) -> dict[str, Any]:
    """Validate that a frozen control arm is rich enough for stage-2 gating."""
    validation = validate_frozen_control_arm(freeze_id)
    errors = list(validation["errors"])
    if not validation["valid"]:
        return {
            **validation,
            "ready": False,
            "passes": False,
            "errors": errors,
        }

    with open(_control_arm_dir(freeze_id) / "control_metrics.json") as f:
        metrics_payload = json.load(f)

    if _extract_fresh_window_brier(metrics_payload) is None:
        errors.append(
            "control_metrics.json is missing fresh-window Brier "
            "(expected fresh_data_brier or sliced_metrics.fresh_window.brier)."
        )

    ready = len(errors) == 0
    return {
        **validation,
        "ready": ready,
        "passes": ready,
        "errors": errors,
    }


def main() -> None:
    """CLI entry point: freeze a promoted candidate as a new control arm.

    Usage::

        python -m src.strategy.control_arm \\
          --run-dir data/logs/evaluation/run_20260316_044304 \\
          --candidate-id rematch_features_append_only_2026_production \\
          --freeze-id 20260316
    """
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Freeze a promoted model as a new control arm")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Evaluation run directory (e.g. data/logs/evaluation/run_20260316_044304)",
    )
    parser.add_argument(
        "--candidate-id",
        required=True,
        help=(
            "Promoted candidate ID using single underscores "
            "(e.g. rematch_features_append_only_2026_production)"
        ),
    )
    parser.add_argument(
        "--freeze-id",
        required=True,
        help="Date-based freeze identifier (e.g. 20260316)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = Path(__file__).resolve().parent.parent.parent / run_dir

    # Resolve cell metrics JSON.
    # candidate_id uses single underscores:  rematch_features_append_only_2026_production
    # filename uses double underscores:       rematch_features__append_only_2026__production__isotonic_metrics.json
    cells_dir = run_dir / "cells"
    if not cells_dir.exists():
        logger.error("cells/ directory not found under %s", run_dir)
        sys.exit(1)

    cell_path: Path | None = None
    for f in sorted(cells_dir.glob("*_metrics.json")):
        simplified = (
            f.stem
            .replace("__isotonic_metrics", "")
            .replace("__sigmoid_metrics", "")
            .replace("__", "_")
        )
        if simplified == args.candidate_id:
            cell_path = f
            break

    if cell_path is None:
        candidates = [
            f.stem
            .replace("__isotonic_metrics", "")
            .replace("__sigmoid_metrics", "")
            .replace("__", "_")
            for f in sorted(cells_dir.glob("*_metrics.json"))
        ]
        logger.error("No cell metrics found for candidate %r", args.candidate_id)
        logger.error("Available candidates:\n  %s", "\n  ".join(candidates[:20]))
        sys.exit(1)

    logger.info("Using cell metrics: %s", cell_path.name)
    with open(cell_path) as fh:
        cell_data = json.load(fh)

    m = cell_data["metrics"]
    sliced = cell_data.get("sliced_metrics", {})
    source_metrics: dict[str, Any] = {
        "accuracy": m["accuracy"],
        "brier": m["brier"],
        "log_loss": m["log_loss"],
        "ece": m["ece"],
        "fresh_data_brier": sliced.get("fresh_window", {}).get("brier"),
        "year_by_year": sliced.get("by_year"),
        "bootstrap_ci": m.get("bootstrap_ci"),
        "n_predictions": cell_data.get("n_predictions"),
        "n_folds": cell_data.get("n_folds"),
        "model_variant": cell_data.get("model_variant"),
        "feature_family": cell_data.get("feature_family"),
        "dataset_variant": cell_data.get("dataset_variant"),
        "calibration_method": cell_data.get("calibration_method"),
        "evaluated_at": cell_data.get("evaluated_at"),
    }

    # Load sweep summary (stage3 best config)
    sweep_dir = run_dir / "stage3_sweeps" / args.candidate_id
    sweep_summary: dict[str, Any] | None = None
    best_result_path = sweep_dir / "best_result.json"
    if best_result_path.exists():
        with open(best_result_path) as fh:
            sweep_summary = json.load(fh)
        logger.info("Loaded sweep summary from %s", best_result_path)
    else:
        logger.warning("No sweep summary at %s — proceeding without it", best_result_path)

    # Load bet log and bankroll history
    bet_log_path = sweep_dir / "best_bet_log.csv"
    bet_log: pd.DataFrame | None = None
    if bet_log_path.exists():
        bet_log = pd.read_csv(bet_log_path)
        logger.info("Loaded bet log: %d rows", len(bet_log))
    else:
        logger.warning("No bet log at %s", bet_log_path)

    bh_path = sweep_dir / "best_bankroll_history.csv"
    bankroll_history: pd.DataFrame | None = None
    if bh_path.exists():
        bankroll_history = pd.read_csv(bh_path)
        logger.info("Loaded bankroll history: %d rows", len(bankroll_history))
    else:
        logger.warning("No bankroll history at %s", bh_path)

    arm_dir = freeze_control_arm(
        source_metrics=source_metrics,
        freeze_id=args.freeze_id,
        sweep_summary=sweep_summary,
        bet_log=bet_log,
        bankroll_history=bankroll_history,
    )
    logger.info("Frozen to: %s", arm_dir)

    result = validate_frozen_control_arm(args.freeze_id)
    if result["valid"]:
        logger.info("Integrity check: PASS")
    else:
        logger.error("Integrity check: FAIL — %s", result["errors"])
        sys.exit(1)


if __name__ == "__main__":
    main()


def validate_frozen_control_arm_for_promotion_gate(freeze_id: str) -> dict[str, Any]:
    """Validate that a frozen control arm supports a trustworthy stage-4 verdict."""
    selection_validation = validate_frozen_control_arm_for_selection_gate(freeze_id)
    errors = list(selection_validation["errors"])
    if not selection_validation["valid"]:
        return {
            **selection_validation,
            "ready": False,
            "passes": False,
            "errors": errors,
        }

    arm_dir = _control_arm_dir(freeze_id)
    with open(arm_dir / "control_metrics.json") as f:
        metrics_payload = json.load(f)

    if _extract_year_by_year_metrics(metrics_payload) is None:
        errors.append(
            "control_metrics.json is missing year-by-year metrics "
            "(expected year_by_year or sliced_metrics.by_year)."
        )

    sweep_path = arm_dir / "control_sweep_summary.json"
    if not sweep_path.exists():
        errors.append("Missing control_sweep_summary.json for the trading bar baseline.")
    else:
        try:
            with open(sweep_path) as f:
                sweep_payload = json.load(f)
            from src.strategy.promotion_gate import _canonicalize_sweep_payload

            _canonicalize_sweep_payload(sweep_payload, sweep_path)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(
                "control_sweep_summary.json is not a usable trading baseline: "
                f"{exc}"
            )

    bet_log_path = arm_dir / "control_bet_log.csv"
    if not _csv_has_rows(bet_log_path):
        errors.append("Missing non-empty control_bet_log.csv for baseline trading-volume checks.")

    bankroll_path = arm_dir / "control_bankroll_history.csv"
    if not _csv_has_rows(bankroll_path):
        errors.append("Missing non-empty control_bankroll_history.csv for baseline trading artifacts.")

    ready = len(errors) == 0
    return {
        **selection_validation,
        "ready": ready,
        "passes": ready,
        "errors": errors,
    }
