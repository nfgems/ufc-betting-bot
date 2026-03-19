"""Helpers for evaluating candidate artifacts on a fixed refreshed V4 test set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.model.predict import predict_batch
from src.model.train import load_model


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_KEY = "_metadata"


def _repo_display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_refreshed_v4_test_set(test_set_path: Path | str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a fixed refreshed V4 test set and return metadata for auditability."""
    path = Path(test_set_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Refreshed V4 test set not found: {path}")

    test_df = pd.read_csv(path, parse_dates=["event_date"])
    if "target" not in test_df.columns:
        raise ValueError(f"Refreshed V4 test set is missing required 'target' column: {path}")

    metadata = {
        "test_set_path": _repo_display_path(path),
        "test_set_sha256": _sha256(path),
        "n_test_rows": int(len(test_df)),
    }
    return test_df, metadata


def evaluate_model_artifact(
    *,
    label: str,
    model_path: Path | str,
    test_df: pd.DataFrame,
) -> dict[str, object]:
    """Evaluate one saved model artifact on a supplied refreshed V4 test set."""
    resolved_path = Path(model_path).resolve()
    model_result = load_model(str(resolved_path))
    predictions = predict_batch(test_df, model_result=model_result)

    y_true = predictions["target"].to_numpy()
    y_prob = predictions["prob_a"].to_numpy()
    y_pred = (y_prob > 0.5).astype(int)

    return {
        "model_path": _repo_display_path(resolved_path),
        "feature_count": int(len(model_result.get("feature_cols", []))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def evaluate_models_against_test_set(
    *,
    test_set_path: Path | str,
    labeled_models: dict[str, Path | str] | Iterable[tuple[str, Path | str]],
) -> dict[str, object]:
    """
    Evaluate labeled model artifacts against one explicit refreshed V4 test set.

    The payload is intentionally close to the existing candidate JSONs, but adds
    `_metadata` so future runs always record the exact test-set snapshot used.
    """
    test_df, metadata = load_refreshed_v4_test_set(test_set_path)
    items = labeled_models.items() if isinstance(labeled_models, dict) else labeled_models

    payload: dict[str, object] = {METADATA_KEY: metadata}
    for label, model_path in items:
        payload[label] = evaluate_model_artifact(
            label=label,
            model_path=model_path,
            test_df=test_df,
        )
    return payload


def write_evaluation_payload(payload: dict[str, object], output_path: Path | str) -> Path:
    """Persist refreshed V4 comparison results as stable JSON."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
