import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.model.refreshed_v4_eval import (
    METADATA_KEY,
    evaluate_models_against_test_set,
    load_refreshed_v4_test_set,
    write_evaluation_payload,
)


class DummyProbabilityModel:
    def predict_proba(self, X):
        probs = np.clip(np.nan_to_num(X[:, 0], nan=0.0), 0.0, 1.0)
        return np.column_stack([1.0 - probs, probs])


def _write_model(path: Path) -> None:
    payload = {
        "model": DummyProbabilityModel(),
        "feature_cols": ["score"],
        "training_spec": {"feature_cols": ["score"]},
        "col_medians": np.array([0.5]),
        "impute_strategy": "native_nan",
    }
    joblib.dump(payload, path)


def test_load_refreshed_v4_test_set_requires_target(tmp_path):
    test_set_path = tmp_path / "test_set.csv"
    pd.DataFrame({"event_date": ["2026-01-01"], "score": [0.8]}).to_csv(test_set_path, index=False)

    with pytest.raises(ValueError, match="target"):
        load_refreshed_v4_test_set(test_set_path)


def test_evaluate_models_against_test_set_records_metrics_and_metadata(tmp_path):
    test_df = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
            "score": [0.9, 0.2, 0.8, 0.1],
            "target": [1, 0, 1, 0],
        }
    )
    test_set_path = tmp_path / "test_set.csv"
    test_df.to_csv(test_set_path, index=False)

    model_path = tmp_path / "dummy_model.pkl"
    _write_model(model_path)

    payload = evaluate_models_against_test_set(
        test_set_path=test_set_path,
        labeled_models=[("dummy", model_path)],
    )

    assert payload[METADATA_KEY]["n_test_rows"] == 4
    assert payload[METADATA_KEY]["test_set_sha256"]

    y_true = test_df["target"].to_numpy()
    y_prob = test_df["score"].to_numpy()
    expected_accuracy = accuracy_score(y_true, (y_prob > 0.5).astype(int))
    expected_log_loss = log_loss(y_true, y_prob, labels=[0, 1])
    expected_brier = brier_score_loss(y_true, y_prob)

    entry = payload["dummy"]
    assert entry["feature_count"] == 1
    assert entry["accuracy"] == pytest.approx(expected_accuracy)
    assert entry["log_loss"] == pytest.approx(expected_log_loss)
    assert entry["brier"] == pytest.approx(expected_brier)
    assert entry["model_path"] == model_path.resolve().relative_to(Path.cwd()).as_posix()


def test_write_evaluation_payload_creates_json_file(tmp_path):
    output_path = tmp_path / "artifacts" / "eval.json"
    payload = {
        METADATA_KEY: {
            "test_set_path": "data/processed/candidates/refreshed/test_set.csv",
            "test_set_sha256": "abc123",
            "n_test_rows": 2,
        },
        "dummy": {
            "model_path": "models/candidates/dummy/xgboost_model.pkl",
            "feature_count": 1,
            "accuracy": 0.5,
            "log_loss": 0.6,
            "brier": 0.24,
        },
    }

    written_path = write_evaluation_payload(payload, output_path)

    assert written_path == output_path.resolve()
    assert written_path.exists()
    assert json.loads(written_path.read_text(encoding="utf-8")) == payload
