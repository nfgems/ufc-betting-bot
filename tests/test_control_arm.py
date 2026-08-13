"""Tests for the control arm module."""

import json

import pandas as pd
import pytest

import src.strategy.control_arm as control_arm
from src.strategy.control_arm import (
    freeze_control_arm,
    load_frozen_control_metrics,
    load_frozen_control_trading_artifacts,
    load_frozen_sweep_summary,
    validate_frozen_control_arm,
    validate_frozen_control_arm_for_promotion_gate,
    validate_frozen_control_arm_for_selection_gate,
)

@pytest.fixture
def mock_frozen_root(tmp_path, monkeypatch):
    """Redirect FROZEN_DIR to a temporary directory."""
    monkeypatch.setattr(control_arm, "FROZEN_DIR", tmp_path)
    return tmp_path

def test_freeze_creates_directory_and_files(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.65,
        "brier": 0.21,
        "log_loss": 0.58,
        "ece": 0.04,
        "extra": "ignore me"
    }
    freeze_id = "test_freeze_2026"
    
    # freeze_control_arm returns Path
    result_path = freeze_control_arm(source_metrics, freeze_id)
    
    freeze_dir = mock_frozen_root / "control_arm_test_freeze_2026"
    assert freeze_dir.exists()
    assert result_path == freeze_dir
    assert (freeze_dir / "control_metrics.json").exists()
    assert (freeze_dir / "MANIFEST.md").exists()
    assert (freeze_dir / "checksums.json").exists()

def test_freeze_requires_metrics_keys(mock_frozen_root):
    # Missing 'ece'
    source_metrics = {
        "accuracy": 0.65,
        "brier": 0.21,
        "log_loss": 0.58
    }
    with pytest.raises(ValueError, match="Source metrics missing required keys"):
        freeze_control_arm(source_metrics, "fail_freeze")

def test_validate_passes_for_valid_freeze(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.65,
        "brier": 0.21,
        "log_loss": 0.58,
        "ece": 0.04
    }
    freeze_id = "valid_freeze"
    freeze_control_arm(source_metrics, freeze_id)
    
    validation = validate_frozen_control_arm(freeze_id)
    assert validation["valid"] is True

def test_validate_detects_missing_files(mock_frozen_root):
    freeze_id = "missing_files"
    freeze_dir = mock_frozen_root / f"control_arm_{freeze_id}"
    freeze_dir.mkdir(parents=True)
    # Missing checksums.json and control_metrics.json
    
    validation = validate_frozen_control_arm(freeze_id)
    assert validation["valid"] is False
    assert any("Missing required file" in err for err in validation["errors"])

def test_validate_detects_checksum_mismatch(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.65,
        "brier": 0.21,
        "log_loss": 0.58,
        "ece": 0.04
    }
    freeze_id = "tampered_freeze"
    freeze_control_arm(source_metrics, freeze_id)
    
    # Tamper with the metrics file
    metrics_path = mock_frozen_root / f"control_arm_{freeze_id}" / "control_metrics.json"
    content = json.loads(metrics_path.read_text())
    content["accuracy"] = 0.99 # Change value
    metrics_path.write_text(json.dumps(content))
    
    validation = validate_frozen_control_arm(freeze_id)
    assert validation["valid"] is False
    assert any("Checksum mismatch" in err for err in validation["errors"])

def test_load_metrics_returns_correct_values(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.72,
        "brier": 0.18,
        "log_loss": 0.52,
        "ece": 0.02,
        "fresh_data_brier": 0.19,
        "sliced_metrics": {"fresh_window": {"brier": 0.19, "n_samples": 42}},
    }
    freeze_id = "load_test"
    freeze_control_arm(source_metrics, freeze_id)
    
    loaded = load_frozen_control_metrics(freeze_id)
    assert loaded["accuracy"] == 0.72
    assert loaded["ece"] == 0.02
    assert loaded["fresh_data_brier"] == 0.19
    assert loaded["sliced_metrics"]["fresh_window"]["n_samples"] == 42

def test_load_sweep_summary_returns_correct_values(mock_frozen_root):
    source_metrics = {"accuracy": 0.6, "brier": 0.2, "log_loss": 0.6, "ece": 0.05}
    sweep_summary = {"roi": 0.05, "total_bets": 100}
    freeze_id = "sweep_test"
    freeze_control_arm(source_metrics, freeze_id, sweep_summary=sweep_summary)
    
    loaded = load_frozen_sweep_summary(freeze_id)
    assert loaded == sweep_summary

def test_load_sweep_summary_returns_none_if_missing(mock_frozen_root):
    source_metrics = {"accuracy": 0.6, "brier": 0.2, "log_loss": 0.6, "ece": 0.05}
    freeze_id = "no_sweep_test"
    freeze_control_arm(source_metrics, freeze_id) # No sweep summary
    
    loaded = load_frozen_sweep_summary(freeze_id)
    assert loaded is None


def test_freeze_writes_standard_trading_artifacts(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.72,
        "brier": 0.18,
        "log_loss": 0.52,
        "ece": 0.02,
        "fresh_data_brier": 0.19,
        "year_by_year": [{"year": 2025, "brier": 0.18}],
    }
    freeze_control_arm(
        source_metrics,
        "trading_artifacts",
        sweep_summary={"roi": 0.05, "total_bets": 1},
        bet_log=pd.DataFrame([{"event_date": "2025-01-01", "profit": 5.0}]),
        bankroll_history=[500.0, 505.0],
    )

    artifacts = load_frozen_control_trading_artifacts("trading_artifacts")
    assert len(artifacts["bet_log"]) == 1
    assert artifacts["bankroll_history"]["combined"].tolist() == [500.0, 505.0]


def test_freeze_sanitizes_non_finite_sweep_values(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.72,
        "brier": 0.18,
        "log_loss": 0.52,
        "ece": 0.02,
    }
    freeze_control_arm(
        source_metrics,
        "sanitized_sweep",
        sweep_summary={"roi": 0.05, "c_max_decimal_odds": float("inf")},
    )

    sweep_path = mock_frozen_root / "control_arm_sanitized_sweep" / "control_sweep_summary.json"
    raw_text = sweep_path.read_text(encoding="utf-8")
    payload = json.loads(raw_text)

    assert "Infinity" not in raw_text
    assert payload["c_max_decimal_odds"] is None


def test_selection_gate_readiness_requires_fresh_window_brier(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.72,
        "brier": 0.18,
        "log_loss": 0.52,
        "ece": 0.02,
    }
    freeze_control_arm(source_metrics, "selection_incomplete")

    readiness = validate_frozen_control_arm_for_selection_gate("selection_incomplete")
    assert readiness["ready"] is False
    assert any("fresh-window Brier" in error for error in readiness["errors"])


def test_promotion_gate_readiness_requires_richer_artifacts(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.72,
        "brier": 0.18,
        "log_loss": 0.52,
        "ece": 0.02,
        "fresh_data_brier": 0.19,
    }
    freeze_control_arm(source_metrics, "promotion_incomplete")

    readiness = validate_frozen_control_arm_for_promotion_gate("promotion_incomplete")
    assert readiness["ready"] is False
    assert any("year-by-year metrics" in error for error in readiness["errors"])
    assert any("control_bet_log.csv" in error for error in readiness["errors"])


def test_promotion_gate_readiness_requires_trading_keys_in_sweep_summary(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.72,
        "brier": 0.18,
        "log_loss": 0.52,
        "ece": 0.02,
        "fresh_data_brier": 0.19,
        "year_by_year": [{"year": 2025, "brier": 0.18, "ece": 0.02}],
    }
    freeze_control_arm(
        source_metrics,
        "promotion_bad_sweep",
        sweep_summary={"total_bets": 1},
        bet_log=[{"event_date": "2025-01-01", "profit": 5.0}],
        bankroll_history=[500.0, 505.0],
    )

    readiness = validate_frozen_control_arm_for_promotion_gate("promotion_bad_sweep")
    assert readiness["ready"] is False
    assert any("usable trading baseline" in error for error in readiness["errors"])


def test_promotion_gate_readiness_passes_with_rich_freeze(mock_frozen_root):
    source_metrics = {
        "accuracy": 0.72,
        "brier": 0.18,
        "log_loss": 0.52,
        "ece": 0.02,
        "fresh_data_brier": 0.19,
        "year_by_year": [{"year": 2025, "brier": 0.18, "ece": 0.02}],
    }
    freeze_control_arm(
        source_metrics,
        "promotion_ready",
        sweep_summary={
            "roi": 0.05,
            "total_profit": 5.0,
            "max_drawdown": 0.10,
            "total_bets": 1,
        },
        bet_log=[{"event_date": "2025-01-01", "profit": 5.0}],
        bankroll_history=[500.0, 505.0],
    )

    readiness = validate_frozen_control_arm_for_promotion_gate("promotion_ready")
    assert readiness["ready"] is True

def test_load_metrics_raises_on_missing(mock_frozen_root):
    with pytest.raises(FileNotFoundError):
        load_frozen_control_metrics("non_existent_id")

def test_existing_freeze_20260313_validates():
    """Verify that the actual data/frozen/control_arm_20260313 validates."""
    freeze_dir = control_arm.FROZEN_DIR / "control_arm_20260313"
    if not freeze_dir.exists():
        pytest.skip(f"Generated control-arm freeze not present: {freeze_dir}")
    validation = validate_frozen_control_arm("20260313")
    if validation["valid"]:
        assert True
    else:
        pytest.fail(f"Existing freeze 20260313 failed validation: {validation['errors']}")
