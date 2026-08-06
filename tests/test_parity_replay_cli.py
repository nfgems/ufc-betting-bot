import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts import parity_replay


def test_established_history_mask_requires_prior_rows_for_both_fighters():
    frame = pd.DataFrame(
        [
            {"event_date": "2025-01-01", "fighter_a": "A", "fighter_b": "B"},
            {"event_date": "2025-01-02", "fighter_a": "A", "fighter_b": "C"},
            {"event_date": "2025-01-03", "fighter_a": "A", "fighter_b": "B"},
            {"event_date": "2025-01-04", "fighter_a": "B", "fighter_b": "A"},
        ]
    )

    mask = parity_replay._established_history_mask(frame)

    assert mask.tolist() == [False, False, True, True]


def _run_replay(
    tmp_path,
    monkeypatch,
    *,
    feature_cols: list[str],
    training_values: dict,
    live_values: dict | None = None,
    mode: str = "exact",
    build_error: Exception | None = None,
) -> tuple[int, dict]:
    processed_dir = tmp_path / "data" / "processed" / "candidates" / "run"
    processed_dir.mkdir(parents=True)
    row = {
        "event_date": "2026-08-01",
        "fighter_a": "Fighter A",
        "fighter_b": "Fighter B",
        "weight_class": "Lightweight",
        "is_title_bout": 0,
        "num_rounds_feat": 3,
        "is_empty_arena": 0,
        **training_values,
    }
    pd.DataFrame([row]).to_csv(processed_dir / "features.csv", index=False)
    pd.DataFrame([{"event_date": "2026-08-01"}]).to_csv(
        processed_dir / "fights_cleaned.csv", index=False
    )

    monkeypatch.setattr(
        parity_replay,
        "resolve_named_training_spec",
        lambda _name: SimpleNamespace(feature_cols=list(feature_cols)),
    )

    def fake_build(*_args, **_kwargs):
        assert _kwargs["processed_data_dir"] == processed_dir.resolve(strict=False)
        if build_error is not None:
            raise build_error
        return dict(live_values if live_values is not None else training_values)

    monkeypatch.setattr(parity_replay, "build_fight_features", fake_build)
    out_path = tmp_path / f"parity_{mode}.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parity_replay.py",
            "--mode",
            mode,
            "--spec",
            "requested_spec",
            "--processed-dir",
            str(processed_dir),
            "--out",
            str(out_path),
        ],
    )

    exit_code = parity_replay.main()
    return exit_code, json.loads(out_path.read_text(encoding="utf-8"))


def test_staged_replay_checks_all_211_requested_features(tmp_path, monkeypatch):
    feature_cols = [f"feature_{index:03d}" for index in range(211)]
    values = {column: float(index) for index, column in enumerate(feature_cols)}

    exit_code, report = _run_replay(
        tmp_path,
        monkeypatch,
        feature_cols=feature_cols,
        training_values=values,
    )

    assert exit_code == 0
    assert report["requested_feature_count"] == 211
    assert report["compared_feature_count"] == 211
    assert report["missing_feature_cols"] == []
    assert report["forbidden_mismatch_count"] == 0
    assert report["passed"] is True
    assert "candidates" in report["processed_dir"]


def test_prefight_time_aging_difference_is_allowlisted_and_reported(
    tmp_path, monkeypatch
):
    exit_code, report = _run_replay(
        tmp_path,
        monkeypatch,
        feature_cols=["stable_feature", "a_age"],
        training_values={"stable_feature": 1.0, "a_age": 30.0},
        live_values={"stable_feature": 1.0, "a_age": 29.997},
        mode="prefight",
    )

    assert exit_code == 0
    assert report["allowlisted_mismatch_count"] == 1
    assert report["forbidden_mismatch_count"] == 0
    assert report["allowlisted_time_aged_features"] == ["a_age"]
    assert report["failures"] == [
        {
            "fight": "Fighter A vs Fighter B",
            "event_date": "2026-08-01",
            "feature": "a_age",
            "verdict": "value_diverge",
            "allowlisted_time_aging": True,
            "live": 29.997,
            "train": 30.0,
        }
    ]


def test_exact_time_aged_difference_is_forbidden(tmp_path, monkeypatch):
    exit_code, report = _run_replay(
        tmp_path,
        monkeypatch,
        feature_cols=["a_age"],
        training_values={"a_age": 30.0},
        live_values={"a_age": 29.997},
        mode="exact",
    )

    assert exit_code == 1
    assert report["allowlisted_time_aged_features"] == []
    assert report["allowlisted_mismatch_count"] == 0
    assert report["forbidden_mismatch_count"] == 1
    assert report["passed"] is False


def test_missing_live_key_is_forbidden_even_when_training_value_is_nan(
    tmp_path, monkeypatch
):
    exit_code, report = _run_replay(
        tmp_path,
        monkeypatch,
        feature_cols=["nan_feature"],
        training_values={"nan_feature": np.nan},
        live_values={},
        mode="prefight",
    )

    assert exit_code == 1
    assert report["forbidden_mismatch_count"] == 1
    assert report["per_feature"][0]["live_feature_missing"] == 1
    assert report["failures"][0]["verdict"] == "live_feature_missing"
    assert report["failures"][0]["allowlisted_time_aging"] is False


def test_missing_requested_training_column_exits_nonzero(tmp_path, monkeypatch):
    exit_code, report = _run_replay(
        tmp_path,
        monkeypatch,
        feature_cols=["present", "missing"],
        training_values={"present": 1.0},
    )

    assert exit_code == 1
    assert report["requested_feature_count"] == 2
    assert report["compared_feature_count"] == 0
    assert report["missing_feature_cols"] == ["missing"]
    assert report["passed"] is False


def test_replay_build_failure_exits_nonzero_and_is_reported(tmp_path, monkeypatch):
    exit_code, report = _run_replay(
        tmp_path,
        monkeypatch,
        feature_cols=["stable_feature"],
        training_values={"stable_feature": 1.0},
        build_error=RuntimeError("synthetic build failure"),
    )

    assert exit_code == 1
    assert report["n_fights"] == 0
    assert report["passed"] is False
    assert report["replay_build_failures"][0]["error"] == (
        "RuntimeError: synthetic build failure"
    )


def test_prefight_non_time_mismatch_exits_nonzero(tmp_path, monkeypatch):
    exit_code, report = _run_replay(
        tmp_path,
        monkeypatch,
        feature_cols=["stable_feature"],
        training_values={"stable_feature": 1.0},
        live_values={"stable_feature": 2.0},
        mode="prefight",
    )

    assert exit_code == 1
    assert report["allowlisted_mismatch_count"] == 0
    assert report["forbidden_mismatch_count"] == 1
    assert report["failures"][0]["allowlisted_time_aging"] is False
