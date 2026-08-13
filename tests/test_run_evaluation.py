"""Tests for the full evaluation orchestrator."""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.strategy.duo_trader_sweep as duo_trader_sweep
import src.strategy.model_lab as model_lab
import src.strategy.model_variants as model_variants
import src.strategy.run_evaluation as run_eval


TEST_RUNTIME_CODE = {
    "git_sha": "deadbeef",
    "git_dirty": False,
    "source_fingerprint": "fingerprint-123",
}


def _write_minimal_betsapi_backfill(
    raw_root,
    processed_root,
    *,
    home_odds: float = 1.8,
    away_odds: float = 2.2,
    missing_summary_ids: int = 0,
) -> None:
    (raw_root / "events_ended").mkdir(parents=True, exist_ok=True)
    (raw_root / "odds_summary").mkdir(parents=True, exist_ok=True)
    (raw_root / "events_ended" / "20220101_page_001.json").write_text(
        json.dumps(
            {
                "success": 1,
                "results": [
                    {
                        "id": "1",
                        "sport_id": 162,
                        "league": {"name": "UFC Fight Night"},
                        "home": {"name": "Fighter A"},
                        "away": {"name": "Fighter B"},
                        "time": 1704067200,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_root / "odds_summary" / "event_1.json").write_text(
        json.dumps(
            {
                "success": 1,
                "results": {
                    "Bet365": {
                        "matching_dir": 1,
                        "odds_update": {"start": "1704063600"},
                        "odds": {
                            "start": {
                                "162_1": {
                                    "id": "line-1",
                                    "home_od": str(home_odds),
                                    "away_od": str(away_odds),
                                    "add_time": "1704060000",
                                }
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    processed_root.mkdir(parents=True, exist_ok=True)
    (processed_root / "ended_events.csv").write_text("event_id\n1\n", encoding="utf-8")
    (processed_root / "odds_summary_rows.csv").write_text(
        "\n".join(
            [
                "event_id,bookmaker,matching_dir,market_phase,market_key,line_id,home_odds,over_odds,draw_odds,away_odds,under_odds,handicap,score_state,selection_added_at_utc,bookmaker_last_update_utc,snapshot_time_utc",
                f"1,Bet365,1,start,162_1,line-1,{home_odds},,,{away_odds},,,,2023-12-31T23:00:00Z,2023-12-31T23:30:00Z,2024-01-01T00:00:00Z",
            ]
        ),
        encoding="utf-8",
    )
    (processed_root / "event_snapshots.csv").write_text("event_id\n1\n", encoding="utf-8")
    (processed_root / "backfill_manifest.json").write_text(
        json.dumps(
            {
                "event_rows": 1,
                "summary_rows": 1,
                "event_snapshots": 1,
                "missing_summary_ids": missing_summary_ids,
            }
        ),
        encoding="utf-8",
    )


def _synthetic_parity_features(n_rows: int = 200) -> pd.DataFrame:
    dates = pd.date_range("2016-01-01", periods=n_rows, freq="14D")
    rows: list[dict[str, object]] = []
    for idx, event_date in enumerate(dates):
        target = int(idx % 2 == 0)
        strength = 1.0 if target else -1.0
        a_implied_prob = 0.58 if target else 0.42
        b_implied_prob = 1.0 - a_implied_prob
        rows.append(
            {
                "event_date": event_date,
                "fighter_a": f"Fighter A {idx}",
                "fighter_b": f"Fighter B {idx}",
                "target": target,
                "diff_skill": strength + (idx % 7) * 0.01,
                "a_roll_slpm": 4.0 + 0.35 * strength + (idx % 5) * 0.02,
                "b_roll_slpm": 4.0 - 0.35 * strength + (idx % 6) * 0.02,
                "a_num_fights": 6 + (idx % 4),
                "b_num_fights": 6 + ((idx + 1) % 4),
                "a_implied_prob": a_implied_prob,
                "b_implied_prob": b_implied_prob,
                "diff_implied_prob": a_implied_prob - b_implied_prob,
            }
        )
    return pd.DataFrame(rows)


def test_build_cell_specs_cross_product():
    cells = run_eval.build_cell_specs(
        variants=["baseline", "combined_v3"],
        datasets=["append_only_2026", "best_of_both_field_level"],
        families=["production", "no_odds"],
        calibration_method="isotonic",
        retrain_months=4,
        bootstrap=100,
    )

    assert len(cells) == 8
    assert cells[0] == run_eval.CellSpec(
        model_variant="baseline",
        dataset_variant="append_only_2026",
        feature_family="production",
        calibration_method="isotonic",
        retrain_months=4,
        bootstrap=100,
    )


def test_build_cell_specs_uses_variant_default_calibration_when_not_overridden():
    cells = run_eval.build_cell_specs(
        variants=["baseline", "combined_best"],
        datasets=["append_only_2026"],
        families=["production"],
        calibration_method=None,
        retrain_months=4,
        bootstrap=100,
    )

    from src.model.training_spec import resolve_named_training_spec
    from src.strategy.model_variants import _promoted_spec_name
    promoted = resolve_named_training_spec(_promoted_spec_name())
    assert [cell.calibration_method for cell in cells] == [promoted.calibration_method, "sigmoid"]


def test_parse_stage_range_supports_single_and_range():
    assert run_eval._parse_stage_range(None) == (1, 5)
    assert run_eval._parse_stage_range("3") == (3, 3)
    assert run_eval._parse_stage_range("2-4") == (2, 4)


def test_checkpoint_round_trip(tmp_path):
    run_eval._save_checkpoint(tmp_path, {"cell_a", "cell_b"})
    payload = run_eval._load_checkpoint(tmp_path)

    assert set(payload["completed_cells"]) == {"cell_a", "cell_b"}
    assert payload["updated_at"] is not None


def test_betsapi_coverage_mask_preserves_prediction_index():
    betsapi_col = run_eval.BETSAPI_CHALLENGER_FEATURE_NAMES[0]
    predictions_df = pd.DataFrame(
        [
            {
                "event_date": "2025-01-01",
                "fighter_a": "A1",
                "fighter_b": "B1",
                "target": 1,
                "prob_a": 0.65,
            },
            {
                "event_date": "2025-01-15",
                "fighter_a": "A2",
                "fighter_b": "B2",
                "target": 0,
                "prob_a": 0.40,
            },
            {
                "event_date": "2025-02-01",
                "fighter_a": "A3",
                "fighter_b": "B3",
                "target": 1,
                "prob_a": 0.70,
            },
        ],
        index=[10, 20, 30],
    )
    features_df = pd.DataFrame(
        [
            {
                "event_date": "2025-01-01",
                "fighter_a": "A1",
                "fighter_b": "B1",
                betsapi_col: 0.55,
            },
            {
                "event_date": "2025-01-15",
                "fighter_a": "A2",
                "fighter_b": "B2",
                betsapi_col: np.nan,
            },
            {
                "event_date": "2025-02-01",
                "fighter_a": "A3",
                "fighter_b": "B3",
                betsapi_col: 0.62,
            },
        ]
    )

    mask = run_eval._betsapi_coverage_mask(predictions_df, features_df)

    assert mask.index.tolist() == [10, 20, 30]
    assert mask.tolist() == [True, False, True]


def test_compute_sliced_metrics_aligns_sparse_prediction_index_for_betsapi_slices():
    betsapi_col = run_eval.BETSAPI_CHALLENGER_FEATURE_NAMES[0]
    predictions_df = pd.DataFrame(
        [
            {
                "event_date": "2025-01-01",
                "fighter_a": "A1",
                "fighter_b": "B1",
                "target": 1,
                "prob_a": 0.65,
            },
            {
                "event_date": "2025-01-15",
                "fighter_a": "A2",
                "fighter_b": "B2",
                "target": 0,
                "prob_a": 0.40,
            },
            {
                "event_date": "2025-02-01",
                "fighter_a": "A3",
                "fighter_b": "B3",
                "target": 1,
                "prob_a": 0.70,
            },
        ],
        index=[10, 20, 30],
    )
    features_df = pd.DataFrame(
        [
            {
                "event_date": "2025-01-01",
                "fighter_a": "A1",
                "fighter_b": "B1",
                betsapi_col: 0.55,
            },
            {
                "event_date": "2025-01-15",
                "fighter_a": "A2",
                "fighter_b": "B2",
                betsapi_col: np.nan,
            },
            {
                "event_date": "2025-02-01",
                "fighter_a": "A3",
                "fighter_b": "B3",
                betsapi_col: 0.62,
            },
        ]
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sliced_metrics = run_eval._compute_sliced_metrics(predictions_df, features_df)

    assert sliced_metrics["by_coverage"]["betsapi_high_coverage"]["n_samples"] == 2
    assert sliced_metrics["by_coverage"]["betsapi_low_coverage"]["n_samples"] == 1
    assert not any(
        "Boolean Series key will be reindexed" in str(item.message)
        for item in caught
    )


def test_resolve_run_dir_creates_fresh_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(run_eval, "EVALUATION_DIR", tmp_path)

    run_dir = run_eval._resolve_run_dir(resume=None)

    assert run_dir.parent == tmp_path
    assert run_dir.exists()
    assert run_dir.is_dir()


def test_resolve_run_dir_rejects_missing_resume_directory(tmp_path):
    with pytest.raises(ValueError, match="Resume run directory does not exist"):
        run_eval._resolve_run_dir(resume=str(tmp_path / "missing"))


def test_build_explicit_resume_overrides_uses_current_default_freeze_id():
    args = SimpleNamespace(
        variants=None,
        datasets=None,
        families=None,
        freeze_id=run_eval.DEFAULT_FREEZE_ID,
        bootstrap=run_eval.DEFAULT_BOOTSTRAP,
        calibration_method=None,
        retrain_months=run_eval.DEFAULT_RETRAIN_MONTHS,
        max_finalists=run_eval.DEFAULT_MAX_FINALISTS,
        no_narrow=False,
    )

    default_overrides = run_eval._build_explicit_resume_overrides(
        args,
        variants=["baseline"],
        datasets=["append_only_2026"],
        families=["production"],
        run_narrow=True,
    )
    assert default_overrides["freeze_id"] is None

    args.freeze_id = "20260313"
    explicit_overrides = run_eval._build_explicit_resume_overrides(
        args,
        variants=["baseline"],
        datasets=["append_only_2026"],
        families=["production"],
        run_narrow=True,
    )
    assert explicit_overrides["freeze_id"] == "20260313"


def test_load_stage1_results_ignores_failed_cells(tmp_path):
    cell_dir = tmp_path / run_eval.CELL_OUTPUT_DIRNAME
    cell_dir.mkdir()
    run_eval._write_json(
        cell_dir / "success_metrics.json",
        {
            "status": "success",
            "cell_key": "good",
            "model_variant": "baseline",
            "dataset_variant": "append_only_2026",
            "feature_family": "production",
            "calibration_method": "isotonic",
            "metrics": {"accuracy": 0.6, "brier": 0.24, "log_loss": 0.68, "ece": 0.05},
            "sliced_metrics": {},
        },
    )
    run_eval._write_json(
        cell_dir / "failed_metrics.json",
        {
            "status": "failed",
            "cell_key": "bad",
            "error": "boom",
        },
    )

    results = run_eval._load_stage1_results(tmp_path)

    assert len(results) == 1
    assert results[0]["cell_key"] == "good"


def test_candidate_from_cell_result_maps_stage1_payload():
    result = {
        "model_variant": "combined_v3",
        "dataset_variant": "append_only_2026",
        "feature_family": "production",
        "calibration_method": "sigmoid",
        "retrain_months": 8,
        "metrics": {"accuracy": 0.61, "brier": 0.23, "log_loss": 0.67, "ece": 0.04},
        "sliced_metrics": {"fresh_window": {"n_samples": 25}},
    }

    candidate = run_eval._candidate_from_cell_result(result)

    assert candidate.name == "combined_v3"
    assert candidate.dataset_variant == "append_only_2026"
    assert candidate.feature_family == "production"
    assert candidate.calibration == "sigmoid"
    assert candidate.retrain_months == 8
    assert candidate.metrics["brier"] == 0.23


def test_control_sweep_placeholder_detection():
    assert run_eval._control_sweep_is_placeholder(
        {
            "roi": 0.0,
            "total_profit": 0.0,
            "max_drawdown": 0.0,
            "avg_clv": 0.0,
            "total_bets": 0,
        }
    )
    assert not run_eval._control_sweep_is_placeholder(
        {
            "roi": 0.02,
            "total_profit": 10.0,
            "max_drawdown": 0.08,
            "avg_clv": 0.01,
            "total_bets": 50,
        }
    )


def test_canonicalize_promotion_model_metrics_from_stage1_shape():
    payload = {
        "metrics": {
            "accuracy": 0.62,
            "brier": 0.23,
            "log_loss": 0.66,
            "ece": 0.04,
        },
        "sliced_metrics": {
            "fresh_window": {"brier": 0.22, "ece": 0.03},
            "by_year": {
                "2025": {"accuracy": 0.61, "brier": 0.24, "ece": 0.05, "log_loss": 0.67, "n_samples": 20},
            },
        },
    }

    metrics = run_eval._canonicalize_promotion_model_metrics(payload)

    assert metrics["brier_score"] == 0.23
    assert metrics["fresh_data_brier"] == 0.22
    assert metrics["year_by_year"][0]["year"] == 2025


def test_canonicalize_promotion_model_metrics_preserves_year_roi_from_list():
    metrics = run_eval._canonicalize_promotion_model_metrics(
        {
            "year_by_year": [
                {
                    "year": "2025",
                    "brier": 0.24,
                    "ece": 0.05,
                    "log_loss": 0.67,
                    "n_samples": 20,
                    "roi": 0.12,
                }
            ]
        }
    )

    assert metrics["year_by_year"][0]["year"] == 2025
    assert metrics["year_by_year"][0]["roi"] == pytest.approx(0.12)


def test_canonicalize_promotion_model_metrics_from_flat_control_arm_shape():
    metrics = run_eval._canonicalize_promotion_model_metrics(
        {
            "accuracy": 0.63,
            "brier": 0.22,
            "log_loss": 0.63,
            "ece": 0.03,
            "fresh_data_brier": 0.24,
            "year_by_year": {
                "2024": {"accuracy": 0.61, "brier": 0.23, "ece": 0.04, "log_loss": 0.64},
                "2025": {"accuracy": 0.64, "brier": 0.21, "ece": 0.03, "log_loss": 0.62},
            },
        }
    )

    assert metrics["brier_score"] == pytest.approx(0.22)
    assert metrics["fresh_data_brier"] == pytest.approx(0.24)
    assert [row["year"] for row in metrics["year_by_year"]] == [2024, 2025]
    assert metrics["year_by_year"][0]["brier_score"] == pytest.approx(0.23)


def test_save_and_load_raw_sweep_result_round_trip(tmp_path):
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    result = {
        "config": {"name": "cfg"},
        "combined": {"roi": 0.1, "total_profit": 12.5},
        "bet_log": pd.DataFrame(
            [
                {
                    "event_date": pd.Timestamp("2025-01-01"),
                    "fighter_a": "A",
                    "fighter_b": "B",
                    "profit": 5.0,
                }
            ]
        ),
        "bankroll_history": pd.DataFrame([{"combined": 105.0}]),
    }

    run_eval._save_raw_sweep_result(candidate_dir, "best", result)
    loaded = run_eval._load_raw_sweep_result(candidate_dir, "best")

    assert loaded["combined"]["roi"] == 0.1
    assert not loaded["bet_log"].empty
    assert not loaded["bankroll_history"].empty


def _fake_stage3_result(candidate_id: str) -> dict:
    best_row = {
        "config": f"{candidate_id}_best",
        "roi": 0.1,
        "profit": 12.5,
        "total_profit": 12.5,
        "avg_clv": 0.02,
        "max_drawdown": 0.08,
        "total_bets": 14,
        "s_bets": 7,
        "s_roi": 0.11,
        "s_avg_clv": 0.02,
        "c_bets": 7,
        "c_roi": 0.09,
        "c_avg_clv": 0.01,
    }
    prod_row = {
        "config": f"{candidate_id}_prod",
        "roi": 0.06,
        "profit": 7.5,
        "total_profit": 7.5,
        "total_bets": 10,
    }
    raw_result = {
        "config": {"name": candidate_id},
        "combined": {"roi": 0.1, "total_profit": 12.5},
        "bet_log": pd.DataFrame(
            [
                {
                    "event_date": pd.Timestamp("2025-01-01"),
                    "fighter_a": "A",
                    "fighter_b": "B",
                    "profit": 5.0,
                }
            ]
        ),
        "bankroll_history": pd.DataFrame([{"combined": 105.0}]),
    }
    return {
        "candidate_id": candidate_id,
        "model_variant": "baseline",
        "production_row": prod_row,
        "production_result": raw_result,
        "broad_summary": [best_row],
        "broad_top_results": {best_row["config"]: raw_result},
        "narrow_summary": [],
        "narrow_top_results": {},
        "best_config": best_row,
        "best_result": raw_result,
        "primary_summary": [best_row],
        "dataset_variant": "append_only_2026",
        "feature_family": "production",
        "calibration_method": "isotonic",
        "retrain_months": 4,
    }


def test_load_or_initialize_metadata_raises_on_material_resume_mismatch(tmp_path):
    saved = run_eval._build_run_metadata(
        variants=["baseline"],
        datasets=["append_only_2026"],
        families=["production"],
        freeze_id="20260313",
        max_workers=1,
        bootstrap=123,
        calibration_method="sigmoid",
        retrain_months=8,
        max_finalists=2,
        run_narrow=True,
        feature_cache_format="pickle",
        runtime_code=TEST_RUNTIME_CODE,
    )
    run_eval._write_json(tmp_path / "metadata.json", saved)

    proposed = run_eval._build_run_metadata(
        variants=["combined_v3"],
        datasets=["append_only_2026"],
        families=["production"],
        freeze_id="20269999",
        max_workers=4,
        bootstrap=5000,
        calibration_method="isotonic",
        retrain_months=4,
        max_finalists=3,
        run_narrow=False,
        feature_cache_format="pickle",
        runtime_code={
            **TEST_RUNTIME_CODE,
            "source_fingerprint": "fingerprint-999",
        },
    )

    with pytest.raises(ValueError, match="Resume metadata mismatch detected"):
        run_eval._load_or_initialize_metadata(
            run_dir=tmp_path,
            proposed_metadata=proposed,
            explicit_overrides={
                "variants": ["combined_v3"],
                "freeze_id": "20269999",
                "calibration_method": "isotonic",
                "run_narrow": False,
            },
        )

def test_load_or_initialize_metadata_allows_material_resume_mismatch_when_requested(tmp_path, caplog):
    saved = run_eval._build_run_metadata(
        variants=["baseline"],
        datasets=["append_only_2026"],
        families=["production"],
        freeze_id="20260313",
        max_workers=1,
        bootstrap=123,
        calibration_method="sigmoid",
        retrain_months=8,
        max_finalists=2,
        run_narrow=True,
        feature_cache_format="pickle",
        runtime_code=TEST_RUNTIME_CODE,
    )
    run_eval._write_json(tmp_path / "metadata.json", saved)

    proposed = run_eval._build_run_metadata(
        variants=["combined_v3"],
        datasets=["append_only_2026"],
        families=["production"],
        freeze_id="20269999",
        max_workers=4,
        bootstrap=5000,
        calibration_method="isotonic",
        retrain_months=4,
        max_finalists=3,
        run_narrow=False,
        feature_cache_format="pickle",
        runtime_code={
            **TEST_RUNTIME_CODE,
            "source_fingerprint": "fingerprint-999",
        },
    )

    with caplog.at_level(logging.WARNING):
        metadata = run_eval._load_or_initialize_metadata(
            run_dir=tmp_path,
            proposed_metadata=proposed,
            explicit_overrides={
                "variants": ["combined_v3"],
                "freeze_id": "20269999",
                "calibration_method": "isotonic",
                "run_narrow": False,
            },
            allow_material_mismatch=True,
        )

    assert metadata["variants"] == ["baseline"]
    assert metadata["freeze_id"] == "20260313"
    assert metadata["calibration_method"] == "sigmoid"
    assert "Resume mismatch allowed" in caplog.text


def test_main_stage2_only_does_not_require_stage3_or_stage4_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(run_eval, "EVALUATION_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "_configure_logging", lambda run_dir: None)
    monkeypatch.setattr(run_eval, "_collect_runtime_code_metadata", lambda: TEST_RUNTIME_CODE)
    monkeypatch.setattr(run_eval, "_feature_cache_format", lambda: "pickle")
    monkeypatch.setattr(run_eval, "_load_stage1_results", lambda run_dir: [])
    monkeypatch.setattr(run_eval, "_run_stage2", lambda **kwargs: [])
    monkeypatch.setattr(
        run_eval,
        "_load_stage3_best_results",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("stage3 load should be skipped")),
    )
    monkeypatch.setattr(
        run_eval,
        "_load_stage4_verdicts",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("stage4 load should be skipped")),
    )
    monkeypatch.setattr(
        run_eval.sys,
        "argv",
        [
            "run_evaluation.py",
            "--stage",
            "2",
            "--variants",
            "baseline",
            "--datasets",
            "append_only_2026",
            "--families",
            "production",
        ],
    )

    run_eval.main()


def test_run_stage3_reuses_completed_candidate_results(monkeypatch, tmp_path):
    finalists = [
        {
            "candidate_id": "candidate_a",
            "model_variant": "baseline",
            "dataset_variant": "append_only_2026",
            "feature_family": "production",
            "calibration_method": "isotonic",
        },
        {
            "candidate_id": "candidate_b",
            "model_variant": "combined_v3",
            "dataset_variant": "append_only_2026",
            "feature_family": "production",
            "calibration_method": "isotonic",
        },
    ]
    metadata = {"source_fingerprint": TEST_RUNTIME_CODE["source_fingerprint"]}
    saved_result = _fake_stage3_result("candidate_a")
    saved_state = run_eval._new_stage3_state(
        run_eval._stage3_context_from_payload(
            finalists[0],
            run_narrow=True,
            metadata=metadata,
        )
    )
    run_eval._persist_stage3_candidate_result(
        tmp_path / run_eval.SWEEP_OUTPUT_DIRNAME / "candidate_a",
        saved_result,
        state=saved_state,
    )

    calls: list[str] = []

    def fake_run_stage3_candidate(**kwargs):
        calls.append(kwargs["payload"]["candidate_id"])
        return _fake_stage3_result(kwargs["payload"]["candidate_id"])

    def fake_compare_finalist_sweeps(sweep_results):
        return pd.DataFrame(
            [{"variant": candidate_id, "roi": 0.1} for candidate_id in sorted(sweep_results)]
        )

    monkeypatch.setattr(run_eval, "_run_stage3_candidate", fake_run_stage3_candidate)
    monkeypatch.setattr(run_eval, "compare_finalist_sweeps", fake_compare_finalist_sweeps)

    results = run_eval._run_stage3(
        run_dir=tmp_path,
        finalists=finalists,
        run_narrow=True,
        metadata=metadata,
    )

    assert sorted(results) == ["candidate_a", "candidate_b"]
    assert calls == ["candidate_b"]
    comparison_df = pd.read_csv(tmp_path / "stage3_comparison.csv")
    assert sorted(comparison_df["variant"].tolist()) == ["candidate_a", "candidate_b"]


def test_run_stage3_candidate_resumes_from_broad_checkpoint(monkeypatch, tmp_path):
    payload = {
        "candidate_id": "candidate_resume",
        "model_variant": "baseline",
        "dataset_variant": "append_only_2026",
        "feature_family": "production",
        "calibration_method": "isotonic",
        "retrain_months": 8,
    }
    metadata = {"source_fingerprint": TEST_RUNTIME_CODE["source_fingerprint"]}
    candidate_dir = tmp_path / run_eval.SWEEP_OUTPUT_DIRNAME / payload["candidate_id"]
    context = run_eval._stage3_context_from_payload(
        payload,
        run_narrow=True,
        metadata=metadata,
    )
    state = run_eval._new_stage3_state(context)

    prod_row = {
        "config": "baseline_production_controls",
        "is_baseline": True,
        "roi": 0.01,
        "profit": 5.0,
        "total_profit": 5.0,
        "total_bets": 10,
        "s_bets": 5,
        "c_bets": 5,
        "avg_clv": 0.0,
        "max_drawdown": 0.06,
        "max_dd": 0.06,
    }
    broad_best_row = {
        "config": "baseline_broad_best",
        "is_baseline": False,
        "roi": 0.08,
        "profit": 12.0,
        "total_profit": 12.0,
        "total_bets": 11,
        "s_bets": 6,
        "c_bets": 5,
        "avg_clv": 0.02,
        "max_drawdown": 0.04,
        "max_dd": 0.04,
        "s_min_edge": 0.02,
        "s_blend_weight": 0.30,
        "s_model_agreement_min_edge": 0.01,
        "c_min_model_prob": 0.65,
        "c_min_no_odds_prob": 0.50,
        "c_share": 0.5,
        "c_min_edge": 0.0,
        "c_max_decimal_odds": 2.0,
    }
    production_result = {
        "config": {"name": "prod"},
        "combined": {"roi": 0.01, "total_profit": 5.0},
        "bet_log": pd.DataFrame([{"event_date": pd.Timestamp("2025-01-01")}]),
        "bankroll_history": pd.DataFrame([{"combined": 505.0}]),
    }
    broad_best_result = {
        "config": {"name": "broad"},
        "combined": {"roi": 0.08, "total_profit": 12.0},
        "bet_log": pd.DataFrame([{"event_date": pd.Timestamp("2025-01-02")}]),
        "bankroll_history": pd.DataFrame([{"combined": 512.0}]),
    }
    run_eval._persist_stage3_production_phase(
        candidate_dir,
        state,
        production_row=prod_row,
        production_result=production_result,
    )
    run_eval._persist_stage3_broad_phase(
        candidate_dir,
        state,
        broad_summary=[broad_best_row, prod_row],
        broad_best_result=broad_best_result,
    )

    calls: list[str] = []
    captured_prediction_kwargs: dict[str, object] = {}

    def fake_generate_walk_forward_predictions(**kwargs):
        captured_prediction_kwargs.update(kwargs)
        calls.append("predictions")
        return [(1, pd.DataFrame())]

    def fail_build_sweep_configs(**kwargs):
        raise AssertionError("Broad sweep should not rerun when broad checkpoint exists")

    def fake_narrow_sweep_around(top_configs, variant_name):
        assert len(top_configs) == 1
        assert top_configs[0].name == "baseline_broad_best"
        return [SimpleNamespace(name="narrow_cfg", variant_name=variant_name)]

    def fake_evaluate_config(fold_predictions, config, initial_bankroll, bet_start_date, **kwargs):
        calls.append(config.name)
        return {
            "config": {"name": config.name},
            "combined": {"roi": 0.12, "total_profit": 20.0},
            "bet_log": pd.DataFrame([{"event_date": pd.Timestamp("2025-01-03")}]),
            "bankroll_history": pd.DataFrame([{"combined": 520.0}]),
        }

    def fake_summary_row_from_result(config, result):
        return {
            "config": config.name,
            "is_baseline": False,
            "roi": result["combined"]["roi"],
            "profit": result["combined"]["total_profit"],
            "total_profit": result["combined"]["total_profit"],
            "total_bets": 9,
            "s_bets": 4,
            "c_bets": 5,
            "avg_clv": 0.01,
            "max_drawdown": 0.05,
            "max_dd": 0.05,
        }

    monkeypatch.setattr(run_eval, "_generate_walk_forward_predictions", fake_generate_walk_forward_predictions)
    monkeypatch.setattr(run_eval, "build_sweep_configs", fail_build_sweep_configs)
    monkeypatch.setattr(run_eval, "narrow_sweep_around", fake_narrow_sweep_around)
    monkeypatch.setattr(run_eval, "_evaluate_config", fake_evaluate_config)
    monkeypatch.setattr(run_eval, "_summary_row_from_result", fake_summary_row_from_result)

    result = run_eval._run_stage3_candidate(
        candidate_dir=candidate_dir,
        payload=payload,
        metadata=metadata,
        run_narrow=True,
    )

    assert calls == ["predictions", "narrow_cfg"]
    assert captured_prediction_kwargs["retrain_months"] == 8
    assert result["best_config"]["config"] == "narrow_cfg"
    assert (candidate_dir / "narrow_summary.csv").exists()
    assert (candidate_dir / "best_result.json").exists()


def test_stage1_and_stage3_prediction_paths_are_strictly_in_parity(monkeypatch, tmp_path):
    features_df = _synthetic_parity_features()
    cell = run_eval.CellSpec(
        model_variant="baseline",
        dataset_variant="append_only_2026",
        feature_family="production",
        calibration_method="sigmoid",
        retrain_months=12,
        bootstrap=1,
    )

    small_xgb_params = {
        "n_estimators": 12,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
    }

    monkeypatch.setattr(model_variants, "_production_xgb_params", lambda: small_xgb_params)
    monkeypatch.setattr(model_lab, "_merge_historical_odds", lambda predictions: predictions)

    def fake_resolve_market_odds(predictions, market_prob_col_a, market_prob_col_b):
        resolved = predictions.copy()
        resolved["a_market_prob"] = resolved["a_implied_prob"]
        resolved["b_market_prob"] = resolved["b_implied_prob"]
        return resolved, "synthetic_kaggle_closing"

    monkeypatch.setattr(model_lab, "_resolve_market_odds", fake_resolve_market_odds)

    captured_stage1: dict[str, pd.DataFrame] = {}

    def fake_compute_model_metrics(predictions_df, bootstrap, seed):
        captured_stage1["predictions"] = predictions_df.copy()
        return {
            "accuracy": 0.0,
            "brier": 0.0,
            "log_loss": 0.0,
            "ece": 0.0,
        }

    monkeypatch.setattr(run_eval, "_compute_model_metrics", fake_compute_model_metrics)
    monkeypatch.setattr(run_eval, "_compute_sliced_metrics", lambda predictions_df, features_df, **kwargs: {})

    features_path = tmp_path / "features.pkl"
    run_eval._save_cached_frame(features_df, features_path, "pickle")

    run_eval._evaluate_cell_worker(
        {
            "cell": asdict(cell),
            "features_path": str(features_path),
        }
    )

    stage3_fold_predictions = duo_trader_sweep._generate_walk_forward_predictions(
        features_df=features_df,
        retrain_months=cell.retrain_months,
        initial_train_years=5,
        bet_start_date=run_eval.TRAIN_CUTOFF_DATE,
        variant_name=cell.model_variant,
        dataset_variant=cell.dataset_variant,
        feature_family=cell.feature_family,
        calibration_method=cell.calibration_method,
    )
    stage3_predictions = pd.concat(
        [predictions for _, predictions in stage3_fold_predictions],
        ignore_index=True,
    )
    stage3_predictions["event_date"] = pd.to_datetime(stage3_predictions["event_date"], errors="coerce")
    stage3_predictions = stage3_predictions[
        stage3_predictions["event_date"] >= pd.Timestamp(run_eval.TRAIN_CUTOFF_DATE)
    ].copy()

    compare_cols = [
        "event_date",
        "fighter_a",
        "fighter_b",
        "prob_a",
        "prob_b",
        "no_odds_prob_a",
        "no_odds_prob_b",
        "fold",
    ]
    stage1_predictions = (
        captured_stage1["predictions"][compare_cols]
        .sort_values(["event_date", "fighter_a", "fighter_b"])
        .reset_index(drop=True)
    )
    stage3_predictions = (
        stage3_predictions[compare_cols]
        .sort_values(["event_date", "fighter_a", "fighter_b"])
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        stage1_predictions[["event_date", "fighter_a", "fighter_b", "fold"]],
        stage3_predictions[["event_date", "fighter_a", "fighter_b", "fold"]],
        check_dtype=False,
    )
    np.testing.assert_allclose(
        stage1_predictions[["prob_a", "prob_b", "no_odds_prob_a", "no_odds_prob_b"]].to_numpy(),
        stage3_predictions[["prob_a", "prob_b", "no_odds_prob_a", "no_odds_prob_b"]].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_generate_variant_fold_predictions_resolves_variant_and_uses_production_no_odds(monkeypatch):
    features_df = _synthetic_parity_features(240)
    variant = model_variants.VariantConfig(name="delta_probe", description="delta")
    resolved_variant = model_variants.VariantConfig(
        name="delta_probe",
        description="resolved delta",
        calibration_method="sigmoid",
        calibration_cv="timeseries_5fold",
    )
    production_no_odds = model_variants.VariantConfig(
        name="_no_odds_production",
        description="internal production no-odds agreement model",
        calibration_method="isotonic",
        calibration_cv="temporal_holdout",
    )
    captured: dict[str, list] = {
        "resolved_feature_variants": [],
        "trained_variants": [],
    }

    monkeypatch.setattr(
        model_lab,
        "_resolve_variant_against_promoted_baseline",
        lambda _variant: resolved_variant,
    )
    monkeypatch.setattr(
        model_lab,
        "resolve_variant_feature_columns",
        lambda _features_df, variant, **_kwargs: (
            captured["resolved_feature_variants"].append(variant) or ["a_implied_prob"],
            [],
        ),
    )
    monkeypatch.setattr(
        model_lab,
        "_select_fold_feature_columns",
        lambda train_df, base_feature_cols, variant: (["a_implied_prob"], ["a_num_fights"]),
    )
    monkeypatch.setattr(
        model_lab,
        "_production_no_odds_variant",
        lambda: production_no_odds,
    )

    def fake_train_variant_model(train_df, feature_cols, variant):
        captured["trained_variants"].append(variant)
        return {"variant": variant, "feature_cols": list(feature_cols)}

    monkeypatch.setattr(model_lab, "train_variant_model", fake_train_variant_model)
    monkeypatch.setattr(
        model_lab,
        "_predict_batch_with_model",
        lambda test_df, _model_result: test_df[
            ["event_date", "fighter_a", "fighter_b", "target", "a_implied_prob", "b_implied_prob"]
        ].assign(prob_a=0.55, prob_b=0.45),
    )
    monkeypatch.setattr(model_lab, "_merge_historical_odds", lambda predictions: predictions)
    monkeypatch.setattr(
        model_lab,
        "_resolve_market_odds",
        lambda predictions, *_args: (
            predictions.assign(
                a_market_prob=predictions["a_implied_prob"],
                b_market_prob=predictions["b_implied_prob"],
            ),
            "synthetic_kaggle_closing",
        ),
    )

    folds = model_lab.generate_variant_fold_predictions(
        features_df,
        variant,
        retrain_months=6,
        initial_train_years=2,
        feature_cols=["a_implied_prob"],
    )

    assert folds
    assert captured["resolved_feature_variants"][0] is resolved_variant
    assert captured["trained_variants"][0] is resolved_variant
    assert captured["trained_variants"][1] is production_no_odds


def test_prepare_feature_caches_uses_side_effect_free_build_variant_features(monkeypatch, tmp_path):
    dataset = pd.DataFrame(
        [
            {"event_date": pd.Timestamp("2025-01-01"), "target": 1},
        ]
    )
    calls: list[bool] = []

    class FakeVariant:
        name = "betsapi_challenger"
        feature_builder_fn = object()
        add_rematch_features = False

    monkeypatch.setattr(run_eval, "_load_dataset_variants", lambda names: {"append_only_2026": dataset})
    monkeypatch.setattr(run_eval, "_selected_profile_variants", lambda names: {"betsapi_profile": "betsapi_challenger"})
    monkeypatch.setitem(run_eval.ALL_VARIANTS, "betsapi_challenger", lambda: FakeVariant())
    monkeypatch.setattr(
        run_eval,
        "build_variant_features",
        lambda fights_df, variant, *, save_artifacts=False: calls.append(save_artifacts) or dataset.copy(),
    )
    monkeypatch.setattr(
        run_eval,
        "augment_features_with_betsapi_mma",
        lambda df, save_artifacts=False: df.copy(),
    )

    run_eval._prepare_feature_caches(
        run_dir=tmp_path,
        dataset_names=["append_only_2026"],
        variant_names=["betsapi_challenger"],
        feature_families=["production"],
        cache_format="pickle",
    )

    assert calls == [False]


def test_prepare_feature_caches_blocks_requested_betsapi_family_when_historical_core_is_all_null(
    monkeypatch,
    tmp_path,
):
    dataset = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2025-01-01"),
                "fighter_a": "A",
                "fighter_b": "B",
                "target": 1,
            },
        ]
    )

    class FakeVariant:
        name = "baseline"
        feature_builder_fn = None
        add_rematch_features = False

    monkeypatch.setattr(run_eval, "_load_dataset_variants", lambda names: {"append_only_2026": dataset})
    monkeypatch.setattr(run_eval, "_selected_profile_variants", lambda names: {"build_features": "baseline"})
    monkeypatch.setitem(run_eval.ALL_VARIANTS, "baseline", lambda: FakeVariant())
    monkeypatch.setattr(
        run_eval,
        "build_variant_features",
        lambda fights_df, variant, *, save_artifacts=False: dataset.copy(),
    )

    def fake_augment(df, save_artifacts=False, include_live_snapshot_features=True):
        augmented = df.copy()
        for column in run_eval.BETSAPI_HISTORICAL_FEATURE_NAMES:
            augmented[column] = np.nan
        augmented["betsapi_snapshot_count"] = 3
        augmented["betsapi_a_implied_prob"] = 0.55
        augmented["betsapi_hist_bookmaker_count"] = 2
        return augmented

    monkeypatch.setattr(run_eval, "augment_features_with_betsapi_mma", fake_augment)

    with pytest.raises(ValueError, match="production_betsapi"):
        run_eval._prepare_feature_caches(
            run_dir=tmp_path,
            dataset_names=["append_only_2026"],
            variant_names=["baseline"],
            feature_families=["production_betsapi"],
            cache_format="pickle",
        )


def test_prepare_feature_caches_allows_production_betsapi_with_real_historical_signal(
    monkeypatch,
    tmp_path,
):
    dataset = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2025-01-01"),
                "fighter_a": "A",
                "fighter_b": "B",
                "target": 1,
            },
        ]
    )

    class FakeVariant:
        name = "baseline"
        feature_builder_fn = None
        add_rematch_features = False

    monkeypatch.setattr(run_eval, "_load_dataset_variants", lambda names: {"append_only_2026": dataset})
    monkeypatch.setattr(run_eval, "_selected_profile_variants", lambda names: {"build_features": "baseline"})
    monkeypatch.setitem(run_eval.ALL_VARIANTS, "baseline", lambda: FakeVariant())
    monkeypatch.setattr(
        run_eval,
        "build_variant_features",
        lambda fights_df, variant, *, save_artifacts=False: dataset.copy(),
    )

    captured = {}

    def fake_augment(df, save_artifacts=False, include_live_snapshot_features=True):
        captured["include_live_snapshot_features"] = include_live_snapshot_features
        augmented = df.copy()
        for column in run_eval.BETSAPI_HISTORICAL_FEATURE_NAMES:
            augmented[column] = np.nan
        augmented["betsapi_hist_opening_a_implied_prob"] = 0.55
        augmented["betsapi_hist_opening_b_implied_prob"] = 0.45
        augmented["betsapi_hist_diff_opening_implied_prob"] = -0.10
        augmented["betsapi_snapshot_count"] = np.nan
        return augmented

    monkeypatch.setattr(run_eval, "augment_features_with_betsapi_mma", fake_augment)

    cache_index = run_eval._prepare_feature_caches(
        run_dir=tmp_path,
        dataset_names=["append_only_2026"],
        variant_names=["baseline"],
        feature_families=["production_betsapi"],
        cache_format="pickle",
    )

    assert (
        "append_only_2026",
        "build_features",
        True,
    ) in cache_index
    assert captured["include_live_snapshot_features"] is False


def test_prepare_feature_caches_allows_expanded_betsapi_family_with_real_signal(
    monkeypatch,
    tmp_path,
):
    dataset = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2025-01-01"),
                "fighter_a": "A",
                "fighter_b": "B",
                "target": 1,
            },
        ]
    )

    class FakeVariant:
        name = "baseline"
        feature_builder_fn = None
        add_rematch_features = False

    monkeypatch.setattr(run_eval, "_load_dataset_variants", lambda names: {"append_only_2026": dataset})
    monkeypatch.setattr(run_eval, "_selected_profile_variants", lambda names: {"build_features": "baseline"})
    monkeypatch.setitem(run_eval.ALL_VARIANTS, "baseline", lambda: FakeVariant())
    monkeypatch.setattr(
        run_eval,
        "build_variant_features",
        lambda fights_df, variant, *, save_artifacts=False: dataset.copy(),
    )

    captured = {}

    def fake_augment(df, save_artifacts=False, include_live_snapshot_features=True):
        captured["include_live_snapshot_features"] = include_live_snapshot_features
        augmented = df.copy()
        for column in run_eval.BETSAPI_HISTORICAL_FEATURE_NAMES:
            augmented[column] = np.nan
        augmented["betsapi_hist_opening_a_implied_prob"] = 0.55
        augmented["betsapi_hist_opening_b_implied_prob"] = 0.45
        augmented["betsapi_hist_diff_opening_implied_prob"] = -0.10
        return augmented

    monkeypatch.setattr(run_eval, "augment_features_with_betsapi_mma", fake_augment)

    cache_index = run_eval._prepare_feature_caches(
        run_dir=tmp_path,
        dataset_names=["append_only_2026"],
        variant_names=["baseline"],
        feature_families=["production_betsapi_expanded"],
        cache_format="pickle",
    )

    assert (
        "append_only_2026",
        "build_features",
        True,
    ) in cache_index
    assert captured["include_live_snapshot_features"] is True


def test_prepare_feature_caches_blocks_requested_betsapi_family_when_implied_probs_are_out_of_range(
    monkeypatch,
    tmp_path,
):
    dataset = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2025-01-01"),
                "fighter_a": "A",
                "fighter_b": "B",
                "target": 1,
            },
        ]
    )

    class FakeVariant:
        name = "baseline"
        feature_builder_fn = None
        add_rematch_features = False

    monkeypatch.setattr(run_eval, "_load_dataset_variants", lambda names: {"append_only_2026": dataset})
    monkeypatch.setattr(run_eval, "_selected_profile_variants", lambda names: {"build_features": "baseline"})
    monkeypatch.setitem(run_eval.ALL_VARIANTS, "baseline", lambda: FakeVariant())
    monkeypatch.setattr(
        run_eval,
        "build_variant_features",
        lambda fights_df, variant, *, save_artifacts=False: dataset.copy(),
    )

    def fake_augment(df, save_artifacts=False, include_live_snapshot_features=True):
        augmented = df.copy()
        for column in run_eval.BETSAPI_HISTORICAL_FEATURE_NAMES:
            augmented[column] = np.nan
        augmented["betsapi_hist_opening_a_implied_prob"] = 1.25
        augmented["betsapi_hist_opening_b_implied_prob"] = 0.45
        return augmented

    monkeypatch.setattr(run_eval, "augment_features_with_betsapi_mma", fake_augment)

    with pytest.raises(ValueError, match="out of range \\[0, 1\\]"):
        run_eval._prepare_feature_caches(
            run_dir=tmp_path,
            dataset_names=["append_only_2026"],
            variant_names=["baseline"],
            feature_families=["production_betsapi"],
            cache_format="pickle",
        )


def test_validate_betsapi_matrix_runtime_readiness_returns_success_without_side_effects(monkeypatch):
    captured = {}

    def fake_prepare(*, run_dir, dataset_names, variant_names, feature_families, cache_format):
        captured["run_dir_exists"] = run_dir.exists()
        captured["dataset_names"] = dataset_names
        captured["variant_names"] = variant_names
        captured["feature_families"] = feature_families
        captured["cache_format"] = cache_format
        return {}

    monkeypatch.setattr(run_eval, "_prepare_feature_caches", fake_prepare)
    monkeypatch.setattr(run_eval, "_feature_cache_format", lambda: "pickle")
    monkeypatch.setattr(
        run_eval,
        "_selected_profile_variants",
        lambda names: {"build_features": "baseline"},
    )

    result = run_eval.validate_betsapi_matrix_runtime_readiness(
        dataset_names=["append_only_2026"],
        variant_names=["baseline"],
        feature_families=["production", "production_betsapi"],
    )

    assert result["ready"] is True
    assert result["checked"] is True
    assert result["families"] == ["production_betsapi"]
    assert captured["dataset_names"] == ["append_only_2026"]
    assert captured["feature_families"] == ["production_betsapi"]
    assert captured["cache_format"] == "pickle"
    assert captured["run_dir_exists"] is True


def test_validate_betsapi_matrix_runtime_readiness_surfaces_runtime_failure(monkeypatch):
    monkeypatch.setattr(
        run_eval,
        "_prepare_feature_caches",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("bad betsapi cache")),
    )
    monkeypatch.setattr(run_eval, "_feature_cache_format", lambda: "pickle")
    monkeypatch.setattr(
        run_eval,
        "_selected_profile_variants",
        lambda names: {"build_features": "baseline"},
    )

    result = run_eval.validate_betsapi_matrix_runtime_readiness(
        dataset_names=["append_only_2026"],
        variant_names=["baseline"],
        feature_families=["production_betsapi"],
    )

    assert result["ready"] is False
    assert result["checked"] is True
    assert result["error"] == "bad betsapi cache"


def test_validate_betsapi_preflight_readiness_uses_lightweight_dataset_join(monkeypatch):
    dataset = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2025-01-01"),
                "fighter_a": "A",
                "fighter_b": "B",
            },
        ]
    )
    captured = {}

    monkeypatch.setattr(
        run_eval,
        "_load_dataset_variants",
        lambda names: {"append_only_2026": dataset},
    )

    def fake_augment(df, *, raw_root, save_artifacts, include_live_snapshot_features=True):
        captured["columns"] = list(df.columns)
        captured["raw_root"] = raw_root
        captured["save_artifacts"] = save_artifacts
        captured["include_live_snapshot_features"] = include_live_snapshot_features
        augmented = df.copy()
        for column in run_eval.BETSAPI_HISTORICAL_FEATURE_NAMES:
            augmented[column] = np.nan
        augmented["betsapi_hist_opening_a_implied_prob"] = 0.55
        augmented["betsapi_hist_opening_b_implied_prob"] = 0.45
        return augmented

    monkeypatch.setattr(run_eval, "augment_features_with_betsapi_mma", fake_augment)

    result = run_eval.validate_betsapi_preflight_readiness(
        dataset_names=["append_only_2026"],
        feature_families=["production", "production_betsapi"],
    )

    assert result["ready"] is True
    assert result["checked"] is True
    assert result["mode"] == "dataset_preflight"
    assert captured["columns"] == [
        "event_date",
        "fighter_a",
        "fighter_b",
        "a_implied_prob",
        "b_implied_prob",
        "diff_implied_prob",
    ]
    assert captured["save_artifacts"] is False
    assert captured["include_live_snapshot_features"] is False


def test_validate_betsapi_preflight_readiness_surfaces_dataset_failure(monkeypatch):
    dataset = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2025-01-01"),
                "fighter_a": "A",
                "fighter_b": "B",
            },
        ]
    )

    monkeypatch.setattr(
        run_eval,
        "_load_dataset_variants",
        lambda names: {"append_only_2026": dataset},
    )
    monkeypatch.setattr(
        run_eval,
        "augment_features_with_betsapi_mma",
        lambda df, **kwargs: (_ for _ in ()).throw(ValueError("dataset join failed")),
    )

    result = run_eval.validate_betsapi_preflight_readiness(
        dataset_names=["append_only_2026"],
        feature_families=["production_betsapi"],
    )

    assert result["ready"] is False
    assert result["error"] == "dataset join failed"


def test_run_stage1_recomputes_when_cell_output_fingerprint_is_stale(monkeypatch, tmp_path):
    cell = run_eval.CellSpec(
        model_variant="baseline",
        dataset_variant="append_only_2026",
        feature_family="production",
        calibration_method="isotonic",
        retrain_months=4,
        bootstrap=1,
    )
    stale_output_path = tmp_path / run_eval.CELL_OUTPUT_DIRNAME / f"{cell.filename_stem}_metrics.json"
    run_eval._write_json(
        stale_output_path,
        {
            "status": "success",
            "cell_key": cell.key,
            "model_variant": cell.model_variant,
            "dataset_variant": cell.dataset_variant,
            "feature_family": cell.feature_family,
            "calibration_method": cell.calibration_method,
            "metrics": {"accuracy": 0.1, "brier": 0.5, "log_loss": 1.0, "ece": 0.5},
            "sliced_metrics": {},
            "source_fingerprint": "fingerprint-old",
        },
    )

    calls: list[str] = []

    monkeypatch.setattr(
        run_eval,
        "_prepare_feature_caches",
        lambda **kwargs: {
            (
                cell.dataset_variant,
                run_eval._variant_profile_key(cell.model_variant),
                False,
            ): str(tmp_path / "unused.pkl")
        },
    )

    def fake_evaluate_cell_worker(task):
        calls.append(task["source_fingerprint"])
        return {
            "status": "success",
            "cell_key": cell.key,
            "model_variant": cell.model_variant,
            "dataset_variant": cell.dataset_variant,
            "feature_family": cell.feature_family,
            "calibration_method": cell.calibration_method,
            "n_folds": 1,
            "n_predictions": 10,
            "metrics": {"accuracy": 0.8, "brier": 0.1, "log_loss": 0.3, "ece": 0.02},
            "sliced_metrics": {},
            "source_fingerprint": task["source_fingerprint"],
        }

    monkeypatch.setattr(run_eval, "_evaluate_cell_worker", fake_evaluate_cell_worker)

    results = run_eval._run_stage1(
        run_dir=tmp_path,
        cells=[cell],
        max_workers=1,
        cache_format="pickle",
        current_source_fingerprint="fingerprint-new",
    )

    assert calls == ["fingerprint-new"]
    assert results[0]["metrics"]["brier"] == pytest.approx(0.1)
    assert results[0]["source_fingerprint"] == "fingerprint-new"


def test_variant_profile_key_regression_coverage():
    assert (
        run_eval._variant_profile_key("baseline")
        == "build_features__rematch_features"
    )
    assert run_eval._variant_profile_key("wc_mode_fix") == "build_features_wc_mode_fix"
    assert run_eval._variant_profile_key("ewm_rolling") == "build_features_ewm"
    assert run_eval._variant_profile_key("rematch_features") == "build_features__rematch_features"
    assert run_eval._variant_profile_key("betsapi_challenger") == "build_features_betsapi_challenger"


def test_default_run_excludes_betsapi_when_backfill_missing():
    variants = run_eval._default_variant_names(has_betsapi_backfill=False)
    families = run_eval._default_feature_families(has_betsapi_backfill=False)

    assert "betsapi_challenger" not in variants
    assert all("betsapi" not in family for family in families)


def test_default_run_excludes_snapshot_dependent_betsapi_scope_even_with_backfill():
    variants = run_eval._default_variant_names(has_betsapi_backfill=True)
    families = run_eval._default_feature_families(has_betsapi_backfill=True)

    assert "betsapi_challenger" not in variants
    assert "production_betsapi" in families
    assert "production_betsapi_expanded" not in families
    assert "ufcstats_betsapi_expanded" not in families


def test_historical_matrix_scope_rejects_snapshot_dependent_variant():
    with pytest.raises(ValueError, match="historical-only"):
        run_eval._ensure_historical_matrix_scope(
            variants=["baseline", "betsapi_challenger"],
            families=["production"],
        )


def test_historical_matrix_scope_rejects_snapshot_dependent_family():
    with pytest.raises(ValueError, match="historical-only"):
        run_eval._ensure_historical_matrix_scope(
            variants=["baseline"],
            families=["production", "production_betsapi_expanded"],
        )


def test_has_saved_betsapi_mma_backfill_requires_completed_processed_artifacts(tmp_path):
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    (raw_root / "events_ended").mkdir(parents=True)
    (raw_root / "odds_summary").mkdir(parents=True)
    (raw_root / "events_ended" / "20220101_page_001.json").write_text("{}", encoding="utf-8")
    (raw_root / "odds_summary" / "event_1.json").write_text("{}", encoding="utf-8")

    assert run_eval._has_saved_betsapi_mma_backfill(
        raw_root=raw_root,
        processed_root=processed_root,
    ) is False

    _write_minimal_betsapi_backfill(raw_root, processed_root)

    assert run_eval._has_saved_betsapi_mma_backfill(
        raw_root=raw_root,
        processed_root=processed_root,
    ) is True

    (raw_root / "events_ended" / "20220101_page_001.json").write_text(
        json.dumps(
            {
                "success": 1,
                "results": [
                    {
                        "id": "1",
                        "sport_id": 162,
                        "league": {"name": "UFC Fight Night"},
                        "home": {"name": "Fighter A"},
                        "away": {"name": "Fighter B"},
                        "time": 1704067200,
                    },
                    {
                        "id": "2",
                        "sport_id": 162,
                        "league": {"name": "UFC Fight Night"},
                        "home": {"name": "Fighter C"},
                        "away": {"name": "Fighter D"},
                        "time": 1704153600,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert run_eval._has_saved_betsapi_mma_backfill(
        raw_root=raw_root,
        processed_root=processed_root,
    ) is False


def test_has_saved_betsapi_mma_backfill_rejects_processed_invalid_moneyline_rows(tmp_path):
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    _write_minimal_betsapi_backfill(
        raw_root,
        processed_root,
        home_odds=0.63,
        away_odds=2.2,
    )

    assert run_eval._has_saved_betsapi_mma_backfill(
        raw_root=raw_root,
        processed_root=processed_root,
    ) is False


def test_has_saved_betsapi_mma_backfill_rejects_stale_manifest_count_mismatch(tmp_path):
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    _write_minimal_betsapi_backfill(raw_root, processed_root)
    with (processed_root / "odds_summary_rows.csv").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n1,Bet365,1,start,162_1,line-2,1.7,,,2.3,,,,2023-12-31T23:05:00Z,2023-12-31T23:35:00Z,2024-01-01T00:05:00Z"
        )

    assert run_eval._has_saved_betsapi_mma_backfill(
        raw_root=raw_root,
        processed_root=processed_root,
    ) is False


def test_has_saved_betsapi_mma_backfill_rejects_manifest_older_than_processed_csvs(tmp_path):
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    _write_minimal_betsapi_backfill(raw_root, processed_root)
    csv_path = processed_root / "odds_summary_rows.csv"
    manifest_path = processed_root / "backfill_manifest.json"
    csv_text = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(csv_text, encoding="utf-8")
    os.utime(manifest_path, (manifest_path.stat().st_atime - 10, manifest_path.stat().st_mtime - 10))

    assert run_eval._has_saved_betsapi_mma_backfill(
        raw_root=raw_root,
        processed_root=processed_root,
    ) is False


def test_explicit_betsapi_request_requires_backfill():
    with pytest.raises(ValueError, match="BetsAPI MMA backfill is not available locally"):
        run_eval._ensure_betsapi_backfill_available(
            variants=["betsapi_challenger"],
            families=["production"],
            has_betsapi_backfill=False,
        )


def test_fingerprint_sources_include_material_training_and_control_files():
    expected = {
        "src/features/build_features.py",
        "src/model/train.py",
        "src/strategy/robustness.py",
        "src/strategy/control_arm.py",
        "src/config.py",
    }
    assert expected.issubset(set(run_eval.FINGERPRINT_SOURCE_FILES))


def test_run_stage4_blocks_when_control_sweep_is_placeholder(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_eval,
        "validate_frozen_control_arm_for_promotion_gate",
        lambda freeze_id: {"ready": True, "errors": []},
    )
    monkeypatch.setattr(
        run_eval,
        "_load_control_arm_payloads",
        lambda freeze_id: (
            {"accuracy": 0.6, "brier": 0.24, "log_loss": 0.68, "ece": 0.05},
            {
                "roi": 0.0,
                "total_profit": 0.0,
                "max_drawdown": 0.0,
                "avg_clv": 0.0,
                "total_bets": 0,
            },
        ),
    )

    finalists = [
        {
            "candidate_id": "baseline_append_only_2026_production",
            "model_variant": "baseline",
            "metrics": {"accuracy": 0.61, "brier": 0.23, "log_loss": 0.67, "ece": 0.04},
            "sliced_metrics": {"fresh_window": {"brier": 0.22, "ece": 0.03}, "by_year": {}},
        }
    ]

    verdicts = run_eval._run_stage4(
        run_dir=tmp_path,
        freeze_id="20260313",
        finalists=finalists,
        sweep_results={},
    )

    assert len(verdicts) == 1
    assert verdicts[0]["blocked"] is True
    assert verdicts[0]["verdict"] == "BLOCKED"

    report = (
        tmp_path
        / run_eval.PROMOTION_OUTPUT_DIRNAME
        / "baseline_append_only_2026_production_promotion.md"
    ).read_text(encoding="utf-8")
    assert "BLOCKED" in report


def test_run_stage4_canonicalizes_nested_stage3_best_result(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_eval,
        "validate_frozen_control_arm_for_promotion_gate",
        lambda freeze_id: {"ready": True, "errors": []},
    )
    monkeypatch.setattr(
        run_eval,
        "_load_control_arm_payloads",
        lambda freeze_id: (
            {"accuracy": 0.6, "brier": 0.24, "log_loss": 0.68, "ece": 0.05},
            {
                "roi": 0.10,
                "total_profit": 100.0,
                "max_drawdown": 0.20,
                "avg_clv": 0.01,
                "total_bets": 50,
            },
        ),
    )

    captured: dict[str, dict] = {}

    def _fake_evaluate(candidate_model_metrics, control_model_metrics, candidate_sweep, baseline_sweep):
        captured["candidate_sweep"] = candidate_sweep
        captured["baseline_sweep"] = baseline_sweep
        return {
            "verdict": "PROMOTE",
            "model_bar": {"passes": True, "reasons": []},
            "trading_bar": {"passes": True, "reasons": []},
            "reasoning": [],
        }

    monkeypatch.setattr(run_eval, "evaluate_for_promotion", _fake_evaluate)
    monkeypatch.setattr(run_eval, "generate_promotion_report", lambda *args, **kwargs: "ok")

    finalists = [
        {
            "candidate_id": "combined_v3_append_only_2026_production",
            "model_variant": "combined_v3",
            "metrics": {"accuracy": 0.61, "brier": 0.23, "log_loss": 0.67, "ece": 0.04},
            "sliced_metrics": {"fresh_window": {"brier": 0.22, "ece": 0.03}, "by_year": {}},
        }
    ]
    sweep_results = {
        "combined_v3_append_only_2026_production": {
            "best_result": {
                "combined": {
                    "roi": 0.22,
                    "total_profit": 150.0,
                    "max_drawdown": 0.08,
                    "avg_clv": 0.02,
                    "total_bets": 25,
                }
            }
        }
    }

    verdicts = run_eval._run_stage4(
        run_dir=tmp_path,
        freeze_id="20260313",
        finalists=finalists,
        sweep_results=sweep_results,
    )

    assert verdicts[0]["blocked"] is False
    assert verdicts[0]["verdict"] == "PROMOTE"
    assert captured["candidate_sweep"]["roi"] == pytest.approx(0.22)
    assert captured["candidate_sweep"]["total_profit"] == pytest.approx(150.0)
    assert captured["candidate_sweep"]["max_drawdown"] == pytest.approx(0.08)
    assert captured["candidate_sweep"]["total_bets"] == 25
    assert captured["baseline_sweep"]["roi"] == pytest.approx(0.10)


def test_run_stage4_preserves_frozen_baseline_trading_artifacts(monkeypatch, tmp_path):
    baseline_bet_log = pd.DataFrame([{"event_date": pd.Timestamp("2025-01-01"), "profit": 5.0}])
    baseline_bankroll = pd.DataFrame([{"combined": 505.0}])
    monkeypatch.setattr(
        run_eval,
        "validate_frozen_control_arm_for_promotion_gate",
        lambda freeze_id: {"ready": True, "errors": []},
    )
    monkeypatch.setattr(
        run_eval,
        "_load_control_arm_payloads",
        lambda freeze_id: (
            {"accuracy": 0.6, "brier": 0.24, "log_loss": 0.68, "ece": 0.05},
            {
                "roi": 0.10,
                "total_profit": 100.0,
                "max_drawdown": 0.20,
                "avg_clv": 0.01,
                "total_bets": 50,
                "bet_log": baseline_bet_log,
                "bankroll_history": baseline_bankroll,
            },
        ),
    )

    captured: dict[str, dict] = {}

    def _fake_evaluate(candidate_model_metrics, control_model_metrics, candidate_sweep, baseline_sweep):
        captured["candidate_sweep"] = candidate_sweep
        captured["baseline_sweep"] = baseline_sweep
        return {
            "verdict": "PROMOTE",
            "model_bar": {"passes": True, "reasons": []},
            "trading_bar": {"passes": True, "reasons": []},
            "reasoning": [],
        }

    monkeypatch.setattr(run_eval, "evaluate_for_promotion", _fake_evaluate)
    monkeypatch.setattr(run_eval, "generate_promotion_report", lambda *args, **kwargs: "ok")

    finalists = [
        {
            "candidate_id": "combined_v3_append_only_2026_production",
            "model_variant": "combined_v3",
            "metrics": {"accuracy": 0.61, "brier": 0.23, "log_loss": 0.67, "ece": 0.04},
            "sliced_metrics": {"fresh_window": {"brier": 0.22, "ece": 0.03}, "by_year": {}},
        }
    ]
    sweep_results = {
        "combined_v3_append_only_2026_production": {
            "best_result": {
                "combined": {
                    "roi": 0.22,
                    "total_profit": 150.0,
                    "max_drawdown": 0.08,
                    "avg_clv": 0.02,
                    "total_bets": 25,
                }
            }
        }
    }

    verdicts = run_eval._run_stage4(
        run_dir=tmp_path,
        freeze_id="20260315",
        finalists=finalists,
        sweep_results=sweep_results,
    )

    assert verdicts[0]["blocked"] is False
    assert captured["baseline_sweep"]["bet_log"].equals(baseline_bet_log)
    assert captured["baseline_sweep"]["bankroll_history"].equals(baseline_bankroll)
    assert not captured["baseline_sweep"]["bet_log"].empty


def test_run_stage5_omits_placeholder_note_when_verdicts_not_blocked(tmp_path):
    pd.DataFrame(
        [
            {
                "cell_key": "baseline|append_only_2026|production|isotonic",
                "model_variant": "baseline",
                "dataset_variant": "append_only_2026",
                "feature_family": "production",
                "calibration_method": "isotonic",
                "accuracy": 0.62,
                "brier": 0.23,
                "log_loss": 0.64,
                "ece": 0.03,
            }
        ]
    ).to_csv(tmp_path / "stage1_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant": "baseline_append_only_2026_production",
                "roi": 0.10,
                "total_profit": 100.0,
            }
        ]
    ).to_csv(tmp_path / "stage3_comparison.csv", index=False)

    metadata = {
        "freeze_id": "20260313",
        "variants": ["baseline"],
        "datasets": ["append_only_2026"],
        "families": ["production"],
        "matrix_cells": 1,
        "feature_cache_format": "parquet",
        "run_narrow": False,
        "git_sha": "deadbeef",
        "git_dirty": False,
    }

    report_path = run_eval._run_stage5(
        run_dir=tmp_path,
        metadata=metadata,
        finalists=[],
        verdicts=[
            {
                "candidate_id": "baseline_append_only_2026_production",
                "model_variant": "baseline",
                "verdict": "SHADOW_TEST",
                "blocked": False,
            }
        ],
    )

    report = report_path.read_text(encoding="utf-8")
    assert "placeholder data" not in report


def test_run_stage5_handles_empty_stage3_comparison_file(tmp_path):
    (tmp_path / "stage3_comparison.csv").write_text("", encoding="utf-8")

    metadata = {
        "freeze_id": "20260315",
        "variants": ["baseline"],
        "datasets": ["append_only_2026"],
        "families": ["production"],
        "matrix_cells": 1,
        "feature_cache_format": "parquet",
        "run_narrow": False,
        "git_sha": "deadbeef",
        "git_dirty": False,
    }

    report_path = run_eval._run_stage5(
        run_dir=tmp_path,
        metadata=metadata,
        finalists=[],
        verdicts=[],
    )

    report = report_path.read_text(encoding="utf-8")
    assert "No finalists passed the selection gate." in report
    assert "```text\n(no rows)\n```" in report


# ---------------------------------------------------------------------------
# T4 — _move_ column exclusion in _validate_requested_betsapi_families_have_signal
# ---------------------------------------------------------------------------

def test_validate_betsapi_signal_ignores_move_columns_for_range_check():
    """Columns with _move_ in the name should be excluded from [0, 1] range checks,
    so an out-of-range _move_ value must NOT cause a rejection."""
    features_df = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2025-01-01"),
                "fighter_a": "A",
                "fighter_b": "B",
                "target": 1,
                # Valid historical implied probs
                "betsapi_hist_opening_a_implied_prob": 0.55,
                "betsapi_hist_opening_b_implied_prob": 0.45,
                "betsapi_hist_diff_opening_implied_prob": 0.10,
                # _move_ column with out-of-range value — should be IGNORED
                "betsapi_hist_m162_2_prob_move_abs": 5.0,
                "betsapi_hist_m162_3_prob_move_abs": 3.0,
            }
        ]
    )

    # Should NOT raise — _move_ columns are excluded from the implied_prob range check
    run_eval._validate_requested_betsapi_families_have_signal(
        features_df,
        requested_families=["production_betsapi"],
        dataset_variant="append_only_2026",
        profile_key="test",
    )


# ---------------------------------------------------------------------------
# Fix #2 tests — fresh-run CLI guards
# ---------------------------------------------------------------------------

def test_fresh_run_rejects_allow_resume_mismatch(monkeypatch, tmp_path):
    """--allow-resume-mismatch on a fresh run (no --resume) should raise ValueError."""
    monkeypatch.setattr(run_eval, "EVALUATION_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "_configure_logging", lambda run_dir: None)
    monkeypatch.setattr(run_eval, "_collect_runtime_code_metadata", lambda: TEST_RUNTIME_CODE)
    monkeypatch.setattr(run_eval, "_feature_cache_format", lambda: "pickle")
    monkeypatch.setattr(
        run_eval.sys,
        "argv",
        [
            "run_evaluation.py",
            "--variants", "baseline",
            "--datasets", "append_only_2026",
            "--families", "production",
            "--allow-resume-mismatch",
        ],
    )

    with pytest.raises(ValueError, match="--allow-resume-mismatch is only valid with --resume"):
        run_eval.main()


def test_fresh_run_warns_on_default_freeze_id(monkeypatch, tmp_path, caplog):
    """Fresh run with default freeze_id should emit a warning."""
    monkeypatch.setattr(run_eval, "EVALUATION_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "_configure_logging", lambda run_dir: None)
    monkeypatch.setattr(run_eval, "_collect_runtime_code_metadata", lambda: TEST_RUNTIME_CODE)
    monkeypatch.setattr(run_eval, "_feature_cache_format", lambda: "pickle")
    monkeypatch.setattr(run_eval, "_has_saved_betsapi_mma_backfill", lambda: False)
    monkeypatch.setattr(
        run_eval.sys,
        "argv",
        [
            "run_evaluation.py",
            "--variants", "baseline",
            "--datasets", "append_only_2026",
            "--families", "production",
            "--stage", "2",
        ],
    )
    monkeypatch.setattr(run_eval, "_load_stage1_results", lambda run_dir: [])
    monkeypatch.setattr(run_eval, "_run_stage2", lambda **kwargs: [])

    with caplog.at_level(logging.WARNING):
        run_eval.main()

    assert any("freeze-id" in record.message.lower() or "freeze_id" in record.message.lower() for record in caplog.records)


def test_fresh_run_warns_on_calibration_override(monkeypatch, tmp_path, caplog):
    """Fresh run with --calibration-method set should emit a warning."""
    monkeypatch.setattr(run_eval, "EVALUATION_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "_configure_logging", lambda run_dir: None)
    monkeypatch.setattr(run_eval, "_collect_runtime_code_metadata", lambda: TEST_RUNTIME_CODE)
    monkeypatch.setattr(run_eval, "_feature_cache_format", lambda: "pickle")
    monkeypatch.setattr(run_eval, "_has_saved_betsapi_mma_backfill", lambda: False)
    monkeypatch.setattr(
        run_eval.sys,
        "argv",
        [
            "run_evaluation.py",
            "--variants", "baseline",
            "--datasets", "append_only_2026",
            "--families", "production",
            "--calibration-method", "sigmoid",
            "--stage", "2",
        ],
    )
    monkeypatch.setattr(run_eval, "_load_stage1_results", lambda run_dir: [])
    monkeypatch.setattr(run_eval, "_run_stage2", lambda **kwargs: [])

    with caplog.at_level(logging.WARNING):
        run_eval.main()

    assert any("calibration" in record.message.lower() for record in caplog.records)
