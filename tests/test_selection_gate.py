"""Tests for the candidate selection gate."""

import pytest

from src.strategy.selection_gate import CandidateResult, SelectionGate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONTROL_METRICS = {
    "accuracy": 0.60,
    "brier": 0.24,
    "log_loss": 0.68,
    "ece": 0.05,
    "fresh_data_brier": 0.22,
    "sliced_metrics": {
        "by_year": {
            "2024": {"accuracy": 0.61, "brier": 0.23, "log_loss": 0.67, "ece": 0.04, "n_samples": 120},
            "2025": {"accuracy": 0.60, "brier": 0.25, "log_loss": 0.69, "ece": 0.05, "n_samples": 110},
        }
    },
}


def _make_candidate(
    name="test_candidate",
    accuracy=0.61,
    brier=0.23,
    log_loss=0.67,
    ece=0.045,
    fresh_n=30,
    fresh_brier=0.21,
    fresh_ece=0.04,
    **kwargs,
) -> CandidateResult:
    return CandidateResult(
        name=name,
        dataset_variant=kwargs.get("dataset_variant", "baseline"),
        feature_family=kwargs.get("feature_family", "production"),
        calibration=kwargs.get("calibration", "isotonic"),
        metrics={
            "accuracy": accuracy,
            "brier": brier,
            "log_loss": log_loss,
            "ece": ece,
        },
        sliced_metrics={
            "fresh_window": {"n_samples": fresh_n, "brier": fresh_brier, "ece": fresh_ece},
            "by_year": kwargs.get(
                "by_year",
                {
                    "2024": {"accuracy": 0.61, "brier": 0.23, "log_loss": 0.67, "ece": 0.04, "n_samples": 120},
                    "2025": {"accuracy": 0.60, "brier": 0.24, "log_loss": 0.68, "ece": 0.045, "n_samples": 110},
                },
            ),
            "by_coverage": kwargs.get("by_coverage", {}),
        },
    )


# ---------------------------------------------------------------------------
# SelectionGate.__init__
# ---------------------------------------------------------------------------


def test_gate_rejects_missing_control_keys():
    with pytest.raises(ValueError, match="missing keys"):
        SelectionGate({"accuracy": 0.6, "brier": 0.24})


def test_gate_accepts_valid_control():
    gate = SelectionGate(CONTROL_METRICS)
    assert gate.control == CONTROL_METRICS


# ---------------------------------------------------------------------------
# Gate: calibration improvement
# ---------------------------------------------------------------------------


def test_candidate_improving_brier_passes_calibration_gate():
    gate = SelectionGate(CONTROL_METRICS)
    # Better Brier, same ECE
    cand = _make_candidate(brier=0.22, ece=0.05)
    result = gate.evaluate(cand)
    assert result["gates"]["calibration_improvement"]["passed"] is True


def test_candidate_improving_ece_passes_calibration_gate():
    gate = SelectionGate(CONTROL_METRICS)
    cand = _make_candidate(brier=0.24, ece=0.03)
    result = gate.evaluate(cand)
    assert result["gates"]["calibration_improvement"]["passed"] is True


def test_candidate_worse_on_both_fails_calibration_gate():
    gate = SelectionGate(CONTROL_METRICS)
    cand = _make_candidate(brier=0.26, ece=0.06)
    result = gate.evaluate(cand)
    assert result["gates"]["calibration_improvement"]["passed"] is False


# ---------------------------------------------------------------------------
# Gate: no material degradation
# ---------------------------------------------------------------------------


def test_candidate_with_large_degradation_fails():
    gate = SelectionGate(CONTROL_METRICS)
    # 20% worse log_loss: 0.68 * 1.20 = 0.816
    cand = _make_candidate(log_loss=0.82)
    result = gate.evaluate(cand)
    assert result["gates"]["no_material_degradation"]["passed"] is False


def test_candidate_within_tolerance_passes():
    gate = SelectionGate(CONTROL_METRICS)
    # 3% worse log_loss: within 5% threshold
    cand = _make_candidate(log_loss=0.70)
    result = gate.evaluate(cand)
    assert result["gates"]["no_material_degradation"]["passed"] is True


# ---------------------------------------------------------------------------
# Gate: fresh data credibility
# ---------------------------------------------------------------------------


def test_candidate_insufficient_fresh_data_fails():
    gate = SelectionGate(CONTROL_METRICS)
    cand = _make_candidate(fresh_n=10)
    result = gate.evaluate(cand)
    assert result["gates"]["fresh_data_credibility"]["passed"] is False


def test_candidate_sufficient_fresh_data_passes():
    gate = SelectionGate(CONTROL_METRICS)
    cand = _make_candidate(fresh_n=25)
    result = gate.evaluate(cand)
    assert result["gates"]["fresh_data_credibility"]["passed"] is True


def test_candidate_fresh_brier_worse_than_control_fails():
    gate = SelectionGate(CONTROL_METRICS)
    cand = _make_candidate(fresh_n=25, brier=0.23, fresh_brier=0.26)
    result = gate.evaluate(cand)
    assert result["gates"]["fresh_data_credibility"]["passed"] is False


def test_candidate_slightly_worse_fresh_brier_still_fails():
    gate = SelectionGate(CONTROL_METRICS)
    cand = _make_candidate(
        fresh_n=25,
        brier=0.23,
        fresh_brier=0.221,
        ece=0.03,
        by_year={
            "2024": {"accuracy": 0.61, "brier": 0.23, "log_loss": 0.67, "ece": 0.04, "n_samples": 120},
            "2025": {"accuracy": 0.60, "brier": 0.24, "log_loss": 0.68, "ece": 0.045, "n_samples": 110},
        },
    )
    result = gate.evaluate(cand)
    assert result["gates"]["fresh_data_credibility"]["passed"] is False


# ---------------------------------------------------------------------------
# Gate: year-by-year stability
# ---------------------------------------------------------------------------


def test_candidate_yearly_ece_regression_fails():
    gate = SelectionGate(CONTROL_METRICS)
    cand = _make_candidate(
        brier=0.23,
        ece=0.03,
        by_year={
            "2024": {"accuracy": 0.61, "brier": 0.23, "log_loss": 0.67, "ece": 0.05, "n_samples": 120},
            "2025": {"accuracy": 0.60, "brier": 0.24, "log_loss": 0.68, "ece": 0.045, "n_samples": 110},
        },
    )
    result = gate.evaluate(cand)
    assert result["gates"]["year_by_year_stability"]["passed"] is False


def test_candidate_yearly_stability_passes():
    gate = SelectionGate(CONTROL_METRICS)
    cand = _make_candidate(
        brier=0.23,
        ece=0.03,
        by_year={
            "2024": {"accuracy": 0.61, "brier": 0.22, "log_loss": 0.66, "ece": 0.038, "n_samples": 120},
            "2025": {"accuracy": 0.60, "brier": 0.24, "log_loss": 0.68, "ece": 0.045, "n_samples": 110},
        },
    )
    result = gate.evaluate(cand)
    assert result["gates"]["year_by_year_stability"]["passed"] is True


# ---------------------------------------------------------------------------
# Gate: accuracy-only rejection
# ---------------------------------------------------------------------------


def test_accuracy_only_win_is_rejected():
    gate = SelectionGate(CONTROL_METRICS)
    # Better accuracy, worse calibration
    cand = _make_candidate(accuracy=0.65, brier=0.25, ece=0.06)
    result = gate.evaluate(cand)
    assert result["gates"]["not_accuracy_only"]["passed"] is False


def test_accuracy_plus_calibration_win_accepted():
    gate = SelectionGate(CONTROL_METRICS)
    cand = _make_candidate(accuracy=0.65, brier=0.22, ece=0.04)
    result = gate.evaluate(cand)
    assert result["gates"]["not_accuracy_only"]["passed"] is True


# ---------------------------------------------------------------------------
# Gate: not slice-driven
# ---------------------------------------------------------------------------


def test_improvement_from_tiny_slice_rejected():
    gate = SelectionGate(CONTROL_METRICS)
    # Only improving bucket has 5 samples out of 200 total (2.5%)
    by_coverage = {
        "high_coverage": {"n_samples": 195, "brier": 0.30},  # worse
        "tiny_slice": {"n_samples": 5, "brier": 0.10},  # better but tiny
    }
    cand = _make_candidate(by_coverage=by_coverage)
    result = gate.evaluate(cand)
    assert result["gates"]["not_slice_driven"]["passed"] is False


def test_improvement_from_substantial_slice_accepted():
    gate = SelectionGate(CONTROL_METRICS)
    by_coverage = {
        "high_coverage": {"n_samples": 150, "brier": 0.22},  # better, 75% of data
        "low_coverage": {"n_samples": 50, "brier": 0.30},
    }
    cand = _make_candidate(by_coverage=by_coverage)
    result = gate.evaluate(cand)
    assert result["gates"]["not_slice_driven"]["passed"] is True


# ---------------------------------------------------------------------------
# End-to-end: passed / composite score
# ---------------------------------------------------------------------------


def test_good_candidate_passes_all_gates():
    gate = SelectionGate(CONTROL_METRICS)
    cand = _make_candidate()
    result = gate.evaluate(cand)
    assert result["passed"] is True
    assert result["composite_score"] > 0


def test_failed_candidate_has_passed_false():
    gate = SelectionGate(CONTROL_METRICS)
    # Worse calibration + insufficient fresh data
    cand = _make_candidate(brier=0.30, ece=0.08, fresh_n=5)
    result = gate.evaluate(cand)
    assert result["passed"] is False


# ---------------------------------------------------------------------------
# select_finalists
# ---------------------------------------------------------------------------


def test_select_finalists_returns_max_n():
    gate = SelectionGate(CONTROL_METRICS)
    candidates = [
        _make_candidate(name="a", brier=0.20, ece=0.03),
        _make_candidate(name="b", brier=0.21, ece=0.04),
        _make_candidate(name="c", brier=0.22, ece=0.045),
        _make_candidate(name="d", brier=0.23, ece=0.048),
    ]
    finalists = gate.select_finalists(candidates, max_finalists=2)
    assert len(finalists) <= 2
    # Best composite score first
    assert finalists[0].name == "a"


def test_select_finalists_excludes_failed():
    gate = SelectionGate(CONTROL_METRICS)
    candidates = [
        _make_candidate(name="good", brier=0.20, ece=0.03),
        _make_candidate(name="bad", brier=0.30, ece=0.08, fresh_n=5),
    ]
    finalists = gate.select_finalists(candidates, max_finalists=3)
    assert len(finalists) == 1
    assert finalists[0].name == "good"


def test_select_finalists_empty_when_all_fail():
    gate = SelectionGate(CONTROL_METRICS)
    candidates = [
        _make_candidate(name="bad1", brier=0.30, ece=0.08),
        _make_candidate(name="bad2", brier=0.28, ece=0.07, fresh_n=5),
    ]
    finalists = gate.select_finalists(candidates, max_finalists=3)
    assert finalists == []


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


def test_generate_report_contains_sections():
    gate = SelectionGate(CONTROL_METRICS)
    candidates = [
        _make_candidate(name="good", brier=0.22),
        _make_candidate(name="bad", brier=0.30, ece=0.08),
    ]
    finalists = gate.select_finalists(candidates)
    report = gate.generate_report(candidates, finalists)
    assert "# Selection Gate Report" in report
    assert "Control arm metrics" in report
    assert "Feature Family" in report
    assert "good" in report
    assert "Finalists:" in report


def test_generate_report_marks_finalists_by_full_candidate_identity():
    gate = SelectionGate(CONTROL_METRICS)
    prod = _make_candidate(
        name="duplicate_variant",
        dataset_variant="append_only_2026",
        feature_family="production",
        brier=0.22,
        ece=0.04,
    )
    no_odds = _make_candidate(
        name="duplicate_variant",
        dataset_variant="append_only_2026",
        feature_family="no_odds",
        brier=0.30,
        ece=0.08,
    )

    report = gate.generate_report([prod, no_odds], [prod])
    finalist_rows = [line for line in report.splitlines() if line.startswith("| duplicate_variant |")]

    assert len(finalist_rows) == 2
    assert finalist_rows[0].endswith(" YES |")
    assert not finalist_rows[1].endswith(" YES |")
