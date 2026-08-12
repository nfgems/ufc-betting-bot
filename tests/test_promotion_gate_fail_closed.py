"""Fail-closed regression coverage for mandatory statistical promotion evidence."""

from __future__ import annotations

import inspect
import sys

import pandas as pd
import pytest

from src.strategy import gate_stats
from src.strategy import promotion_gate
from src.strategy import run_evaluation
from src.strategy.promotion_gate import PromotionGate, evaluate_for_promotion


def _usable_log(*, profit: float) -> pd.DataFrame:
    dates = pd.date_range("2026-01-03", periods=20, freq="7D")
    return pd.DataFrame(
        {
            "event_date": dates,
            "profit": [profit] * len(dates),
            "bet_size": [10.0] * len(dates),
        }
    )


def _sweep(bet_log: object, *, roi: float, profit: float) -> dict:
    return {
        "roi": roi,
        "total_profit": profit,
        "max_drawdown": 0.10,
        "avg_clv": 0.0,
        "bet_log": bet_log,
    }


def _identity_metrics(*, candidate: bool) -> dict:
    payload = {
        "brier_score": 0.20,
        "ece": 0.05,
        "evaluation_sample_sha256": "1" * 64,
        "evaluation_input_value_sha256": "2" * 64,
        "prediction_rows_sha256": "3" * 64,
        "prediction_values_sha256": "4" * 64,
        "model_spec_name": "integrity_205_eval",
        "model_spec_payload_sha256": "5" * 64,
        "calibration_method": "isotonic",
    }
    if candidate:
        payload["model_unchanged_declared"] = True
    return payload


@pytest.mark.parametrize(
    ("candidate_log", "baseline_log", "message"),
    [
        (None, _usable_log(profit=0.0), "Candidate bet log is missing"),
        (_usable_log(profit=1.0), pd.DataFrame(), "Baseline bet log is empty"),
        (
            _usable_log(profit=1.0).drop(columns=["bet_size"]),
            _usable_log(profit=0.0),
            "missing required columns",
        ),
        (
            _usable_log(profit=1.0).assign(event_date="not-a-date"),
            _usable_log(profit=0.0),
            "malformed event_date",
        ),
        (
            _usable_log(profit=1.0).assign(profit="not-a-number"),
            _usable_log(profit=0.0),
            "non-finite or non-numeric profit",
        ),
        (
            _usable_log(profit=1.0).assign(bet_size=0.0),
            _usable_log(profit=0.0),
            "non-positive, non-finite",
        ),
    ],
)
def test_statistical_promotion_rejects_missing_or_malformed_logs(
    candidate_log,
    baseline_log,
    message,
):
    result = PromotionGate.check_trading_bar(
        _sweep(candidate_log, roi=0.10, profit=20.0),
        _sweep(baseline_log, roi=0.0, profit=0.0),
    )

    assert result["passes"] is False
    assert any(message in reason for reason in result["reasons"])
    assert not any("Paired event bootstrap:" in reason for reason in result["reasons"])


def test_statistical_thresholds_are_fixed_at_alpha_point_ten_and_twenty_events(
    monkeypatch,
):
    assert promotion_gate.PROMOTION_BOOTSTRAP_ALPHA == pytest.approx(0.10)
    assert promotion_gate.MINIMUM_PROMOTION_EVENT_DATES == 20
    signature = inspect.signature(PromotionGate.check_trading_bar)
    assert "statistical" not in signature.parameters
    assert "bootstrap_alpha" not in signature.parameters
    assert "minimum_bootstrap_events" not in signature.parameters

    monkeypatch.setattr(
        gate_stats,
        "paired_event_bootstrap",
        lambda candidate, baseline: {
            "n_events": 20,
            "profit_diff": 20.0,
            "roi_diff": 0.10,
            "profit_diff_ci_lo": -1.0,
            "profit_diff_ci_hi": 41.0,
            "p_value": 0.11,
        },
    )
    result = PromotionGate.check_trading_bar(
        _sweep(_usable_log(profit=1.0), roi=0.10, profit=20.0),
        _sweep(_usable_log(profit=0.0), roi=0.0, profit=0.0),
    )

    assert result["passes"] is False
    assert any("alpha=0.10" in reason for reason in result["reasons"])


def test_statistically_superior_but_losing_candidate_fails_absolute_floors():
    candidate_log = _usable_log(profit=-1.0)
    baseline_log = _usable_log(profit=-10.0)

    result = PromotionGate.check_trading_bar(
        _sweep(candidate_log, roi=-0.01, profit=-20.0),
        _sweep(baseline_log, roi=-0.10, profit=-200.0),
    )

    assert result["passes"] is False
    assert any(
        "Candidate ROI must be finite and strictly positive" in reason
        for reason in result["reasons"]
    )
    assert any(
        "Candidate total profit must be finite and strictly positive" in reason
        for reason in result["reasons"]
    )
    assert any("Paired event bootstrap:" in reason for reason in result["reasons"])
    assert not any("alpha=0.10" in reason for reason in result["reasons"])


@pytest.mark.parametrize("roi", [0.0, float("nan"), float("inf"), -float("inf")])
def test_candidate_roi_must_be_finite_and_strictly_positive(roi):
    result = PromotionGate.check_trading_bar(
        _sweep(_usable_log(profit=1.0), roi=roi, profit=20.0),
        _sweep(_usable_log(profit=0.0), roi=0.0, profit=0.0),
    )

    assert result["passes"] is False
    assert any(
        "Candidate ROI must be finite and strictly positive" in reason
        for reason in result["reasons"]
    )


@pytest.mark.parametrize("profit", [0.0, float("nan"), float("inf"), -float("inf")])
def test_candidate_profit_must_be_finite_and_strictly_positive(profit):
    result = PromotionGate.check_trading_bar(
        _sweep(_usable_log(profit=1.0), roi=0.10, profit=profit),
        _sweep(_usable_log(profit=0.0), roi=0.0, profit=0.0),
    )

    assert result["passes"] is False
    assert any(
        "Candidate total profit must be finite and strictly positive" in reason
        for reason in result["reasons"]
    )


@pytest.mark.parametrize(
    ("role", "field", "label"),
    [
        ("candidate", "brier_score", "Candidate Brier"),
        ("control", "brier_score", "Control Brier"),
        ("candidate", "ece", "Candidate ECE"),
        ("control", "ece", "Control ECE"),
    ],
)
@pytest.mark.parametrize(
    "value",
    [None, float("nan"), float("inf"), -float("inf")],
)
def test_model_bar_requires_finite_candidate_and_control_core_metrics(
    role,
    field,
    label,
    value,
):
    candidate = {"brier_score": 0.19, "ece": 0.04}
    control = {"brier_score": 0.20, "ece": 0.05}
    target = candidate if role == "candidate" else control
    target[field] = value

    result = PromotionGate.check_model_bar(candidate, control)

    assert result["passes"] is False
    assert any(
        f"{label} must be finite" in reason for reason in result["reasons"]
    )


@pytest.mark.parametrize(
    ("role", "field", "label"),
    [
        ("candidate", "brier_score", "Candidate Brier"),
        ("control", "brier_score", "Control Brier"),
        ("candidate", "ece", "Candidate ECE"),
        ("control", "ece", "Control ECE"),
    ],
)
def test_model_bar_rejects_missing_candidate_or_control_core_metric(
    role,
    field,
    label,
):
    candidate = {"brier_score": 0.19, "ece": 0.04}
    control = {"brier_score": 0.20, "ece": 0.05}
    target = candidate if role == "candidate" else control
    del target[field]

    result = PromotionGate.check_model_bar(candidate, control)

    assert result["passes"] is False
    assert any(
        f"{label} must be finite" in reason for reason in result["reasons"]
    )


@pytest.mark.parametrize(
    ("role", "field", "label"),
    [
        ("candidate", "max_drawdown", "Candidate drawdown"),
        ("baseline", "max_drawdown", "Baseline drawdown"),
        ("candidate", "avg_clv", "Candidate CLV"),
        ("baseline", "avg_clv", "Baseline CLV"),
    ],
)
@pytest.mark.parametrize(
    "value",
    [None, float("nan"), float("inf"), -float("inf")],
)
def test_trading_bar_requires_finite_candidate_and_baseline_risk_metrics(
    role,
    field,
    label,
    value,
):
    candidate = _sweep(_usable_log(profit=1.0), roi=0.10, profit=20.0)
    baseline = _sweep(_usable_log(profit=0.0), roi=0.0, profit=0.0)
    target = candidate if role == "candidate" else baseline
    target[field] = value

    result = PromotionGate.check_trading_bar(candidate, baseline)

    assert result["passes"] is False
    assert any(
        f"{label} must be finite" in reason for reason in result["reasons"]
    )


@pytest.mark.parametrize(
    ("role", "field", "label"),
    [
        ("candidate", "max_drawdown", "Candidate drawdown"),
        ("baseline", "max_drawdown", "Baseline drawdown"),
        ("candidate", "avg_clv", "Candidate CLV"),
        ("baseline", "avg_clv", "Baseline CLV"),
    ],
)
def test_trading_bar_rejects_missing_candidate_or_baseline_risk_metric(
    role,
    field,
    label,
):
    candidate = _sweep(_usable_log(profit=1.0), roi=0.10, profit=20.0)
    baseline = _sweep(_usable_log(profit=0.0), roi=0.0, profit=0.0)
    target = candidate if role == "candidate" else baseline
    del target[field]

    result = PromotionGate.check_trading_bar(candidate, baseline)

    assert result["passes"] is False
    assert any(
        f"{label} must be finite" in reason for reason in result["reasons"]
    )


@pytest.mark.parametrize("role", ["candidate", "baseline"])
@pytest.mark.parametrize(
    "bad_value",
    [None, float("nan"), float("inf"), -float("inf")],
)
def test_raw_clv_remains_mandatory_when_snapshot_clv_is_available(
    monkeypatch,
    role,
    bad_value,
):
    from src.strategy import gate_stats

    candidate = _sweep(_usable_log(profit=1.0), roi=0.10, profit=20.0)
    baseline = _sweep(_usable_log(profit=0.0), roi=0.0, profit=0.0)
    target = candidate if role == "candidate" else baseline
    if bad_value is None:
        target.pop("avg_clv")
    else:
        target["avg_clv"] = bad_value
    monkeypatch.setattr(
        gate_stats,
        "clv_fair_stats",
        lambda _log: {
            "avg_clv_fair": 0.01,
            "n_bets_distinct": 100,
            "n_bets_total": 100,
            "sufficient": True,
        },
    )

    result = PromotionGate.check_trading_bar(candidate, baseline)

    assert result["passes"] is False
    assert any(
        f"{role.capitalize()} CLV must be finite" in reason
        for reason in result["reasons"]
    )


@pytest.mark.parametrize(
    ("candidate_binding", "control_binding", "message"),
    [
        (None, "a" * 64, "binding is incomplete"),
        ("not-a-sha", "not-a-sha", "valid SHA-256"),
        ("a" * 64, "b" * 64, "sample hashes differ"),
    ],
)
def test_promotion_requires_identical_valid_evaluation_binding(
    monkeypatch,
    candidate_binding,
    control_binding,
    message,
):
    monkeypatch.setattr(
        PromotionGate,
        "check_model_bar",
        lambda self, candidate, control: {"passes": True, "reasons": []},
    )
    monkeypatch.setattr(
        PromotionGate,
        "check_trading_bar",
        lambda self, candidate, control: {"passes": True, "reasons": []},
    )

    result = evaluate_for_promotion(
        {
            "evaluation_sample_sha256": candidate_binding,
            "evaluation_input_value_sha256": "c" * 64,
        },
        {
            "evaluation_sample_sha256": control_binding,
            "evaluation_input_value_sha256": "c" * 64,
        },
        {},
        {},
    )

    assert result["verdict"] == "REJECT"
    assert any(message in reason for reason in result["model_bar"]["reasons"])


@pytest.mark.parametrize(
    ("candidate_binding", "control_binding", "message"),
    [
        (None, "a" * 64, "input value binding is incomplete"),
        ("not-a-sha", "not-a-sha", "input value binding must be a valid SHA-256"),
        ("a" * 64, "b" * 64, "input value hashes differ"),
    ],
)
def test_promotion_requires_identical_valid_evaluation_input_value_binding(
    monkeypatch,
    candidate_binding,
    control_binding,
    message,
):
    monkeypatch.setattr(
        PromotionGate,
        "check_model_bar",
        lambda self, candidate, control: {"passes": True, "reasons": []},
    )
    monkeypatch.setattr(
        PromotionGate,
        "check_trading_bar",
        lambda self, candidate, control: {"passes": True, "reasons": []},
    )

    result = evaluate_for_promotion(
        {
            "evaluation_sample_sha256": "c" * 64,
            "evaluation_input_value_sha256": candidate_binding,
        },
        {
            "evaluation_sample_sha256": "c" * 64,
            "evaluation_input_value_sha256": control_binding,
        },
        {},
        {},
    )

    assert result["verdict"] == "REJECT"
    assert any(message in reason for reason in result["model_bar"]["reasons"])


def test_exact_prediction_identity_allows_explicit_unchanged_model_claim():
    result = PromotionGate.check_model_bar(
        _identity_metrics(candidate=True),
        _identity_metrics(candidate=False),
    )

    assert result["passes"] is True, result["reasons"]
    assert result["model_unchanged_no_improvement_claim"] is True
    assert "model_unchanged_no_improvement_claim" in result["reasons"]


@pytest.mark.parametrize(
    ("field", "candidate_value"),
    [
        ("prediction_rows_sha256", "6" * 64),
        ("prediction_values_sha256", "6" * 64),
        ("model_spec_name", "integrity_211_eval"),
        ("model_spec_payload_sha256", "6" * 64),
        ("calibration_method", "sigmoid"),
    ],
)
def test_declared_unchanged_model_claim_fails_on_any_identity_mismatch(
    field,
    candidate_value,
):
    candidate = _identity_metrics(candidate=True)
    candidate[field] = candidate_value

    result = PromotionGate.check_model_bar(
        candidate,
        _identity_metrics(candidate=False),
    )

    assert result["passes"] is False
    assert result["model_unchanged_no_improvement_claim"] is False
    assert any("Unchanged-model claim" in reason for reason in result["reasons"])


@pytest.mark.parametrize(
    "field",
    [
        "prediction_rows_sha256",
        "prediction_values_sha256",
        "model_spec_name",
        "model_spec_payload_sha256",
        "calibration_method",
    ],
)
def test_declared_unchanged_model_claim_fails_on_missing_identity_field(field):
    candidate = _identity_metrics(candidate=True)
    candidate.pop(field)

    result = PromotionGate.check_model_bar(
        candidate,
        _identity_metrics(candidate=False),
    )

    assert result["passes"] is False
    assert result["model_unchanged_no_improvement_claim"] is False
    assert any("Unchanged-model claim" in reason for reason in result["reasons"])


def test_equal_or_rounded_metrics_never_infer_unchanged_model_identity():
    candidate = _identity_metrics(candidate=True)
    candidate.pop("model_unchanged_declared")

    result = PromotionGate.check_model_bar(
        candidate,
        _identity_metrics(candidate=False),
    )

    assert result["passes"] is False
    assert result["model_unchanged_no_improvement_claim"] is False
    assert any("Neither Brier nor ECE improved" in reason for reason in result["reasons"])


def test_challenger_model_still_requires_strict_core_metric_improvement():
    candidate = _identity_metrics(candidate=False)
    candidate["prediction_values_sha256"] = "6" * 64
    candidate["brier_score"] = 0.19

    result = PromotionGate.check_model_bar(
        candidate,
        _identity_metrics(candidate=False),
    )

    assert result["passes"] is True, result["reasons"]
    assert result["model_unchanged_no_improvement_claim"] is False


def test_unchanged_model_claim_preserves_core_no_regression_limit():
    candidate = _identity_metrics(candidate=True)
    candidate["ece"] = 0.056

    result = PromotionGate.check_model_bar(
        candidate,
        _identity_metrics(candidate=False),
    )

    assert result["passes"] is False
    assert result["model_unchanged_no_improvement_claim"] is True
    assert any("under the unchanged-model claim" in reason for reason in result["reasons"])


def test_required_production_bet_log_is_not_synthesized_as_empty(tmp_path):
    (tmp_path / "production_result.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Required production bet log artifact is missing"):
        run_evaluation._load_raw_sweep_result(
            tmp_path,
            "production",
            require_bet_log=True,
        )


def test_standalone_cli_rejects_removed_legacy_gate(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promotion_gate",
            "--candidate-metrics",
            "candidate.json",
            "--control-metrics",
            "control.json",
            "--candidate-log",
            "candidate.csv",
            "--baseline-log",
            "baseline.csv",
            "--legacy-gate",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        promotion_gate.main()

    assert exc_info.value.code == 2
    assert "unrecognized arguments: --legacy-gate" in capsys.readouterr().err
