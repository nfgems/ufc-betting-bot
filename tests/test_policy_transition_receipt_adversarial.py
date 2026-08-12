from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import check_production_refit_contract as gate
from scripts import check_scheduled_refit_quality as quality
from src.model import train as train_module
from test_production_refit_contract import (
    _confirmed_child_fixture,
    _final_track_c_receipt_fixture,
    _write_json,
)


def _builder_inputs(fixture: dict, tmp_path: Path) -> tuple[Path, Path, dict]:
    artifacts_dir = fixture["artifact_paths"]["summary"].parent
    features_path = tmp_path / fixture["binding"]["features_artifact_path"]
    quality_result = {
        "errors": [],
        "metrics": fixture["receipt"]["health_metrics"],
        "derived_limits": fixture["receipt"]["derived_limits"],
    }
    return artifacts_dir, features_path, quality_result


def _use_fixture_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(quality, "REPO_ROOT", tmp_path)


@pytest.mark.parametrize(
    "artifact_name",
    ["summary", "evaluation_index", "data_quality", "predictive", "production_bets"],
)
def test_healthy_pass_receipt_cli_mints_then_real_validator_detects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    fixture = _final_track_c_receipt_fixture(tmp_path, monkeypatch)
    _use_fixture_repo(monkeypatch, tmp_path)
    artifacts_dir, features_path, quality_result = _builder_inputs(fixture, tmp_path)
    monkeypatch.setattr(
        quality,
        "evaluate_quality",
        lambda **_kwargs: dict(quality_result),
    )
    report_path = tmp_path / "evidence/final_track_c/quality_report.json"
    receipt_path = tmp_path / "evidence/final_track_c/cli_pass_receipt.json"

    assert quality.main(
        [
            "--policy",
            str(fixture["policy_path"]),
            "--features",
            str(features_path),
            "--artifacts-dir",
            str(artifacts_dir),
            "--report",
            str(report_path),
            "--pass-receipt",
            str(receipt_path),
        ]
    ) == 0
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["status"] == "PASS"

    fixture["artifact_paths"][artifact_name].write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(gate.ContractInputError, match=f"artifact changed: {artifact_name}"):
        gate.validate_final_track_c_pass_receipt(
            receipt_path,
            expected_policy_path=fixture["policy_path"],
            expected_fullfit_spec=fixture["fullfit_spec"],
            repo_root=tmp_path,
            quality_validator=fixture["quality_validator"],
            confirmation_validator=fixture["confirmation_validator"],
        )


def test_receipt_builder_rejects_track_c_artifacts_outside_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _final_track_c_receipt_fixture(tmp_path, monkeypatch)
    _use_fixture_repo(monkeypatch, tmp_path)
    _artifacts_dir, features_path, quality_result = _builder_inputs(fixture, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "track_c_summary.csv").write_text("proof\n", encoding="utf-8")

    with pytest.raises(quality.QualityInputError, match="must be preserved below evidence"):
        quality.build_final_track_c_pass_receipt(
            policy_path=fixture["policy_path"],
            features_path=features_path,
            artifacts_dir=outside,
            quality_result=quality_result,
        )


def test_receipt_builder_rejects_artifacts_resolving_to_split_parents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _final_track_c_receipt_fixture(tmp_path, monkeypatch)
    _use_fixture_repo(monkeypatch, tmp_path)
    artifacts_dir, features_path, quality_result = _builder_inputs(fixture, tmp_path)
    alternate = tmp_path / "evidence/alternate/production_gated_bets.csv"
    alternate.parent.mkdir(parents=True)
    alternate.write_bytes(fixture["artifact_paths"]["production_bets"].read_bytes())
    real_resolver = quality._durable_repo_path

    def split_resolver(path: Path, *, label: str) -> tuple[Path, str]:
        if label == "final Track-C production_bets":
            return alternate.resolve(), alternate.relative_to(tmp_path).as_posix()
        return real_resolver(path, label=label)

    # This deterministically exercises the resolved-parent check without
    # depending on Windows symlink privileges.
    monkeypatch.setattr(quality, "_durable_repo_path", split_resolver)
    with pytest.raises(quality.QualityInputError, match="must share one durable directory"):
        quality.build_final_track_c_pass_receipt(
            policy_path=fixture["policy_path"],
            features_path=features_path,
            artifacts_dir=artifacts_dir,
            quality_result=quality_result,
        )


def test_receipt_builder_rejects_post_confirmation_input_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _final_track_c_receipt_fixture(tmp_path, monkeypatch)
    _use_fixture_repo(monkeypatch, tmp_path)
    artifacts_dir, features_path, quality_result = _builder_inputs(fixture, tmp_path)
    odds_inventory = tmp_path / fixture["binding"]["odds_source_inventory_path"]
    odds_inventory.write_text('{"tampered": true}\n', encoding="utf-8")

    with pytest.raises(quality.QualityInputError, match="odds-source inventory changed"):
        quality.build_final_track_c_pass_receipt(
            policy_path=fixture["policy_path"],
            features_path=features_path,
            artifacts_dir=artifacts_dir,
            quality_result=quality_result,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("policy", "changed immutable policy field: health_limits"),
        ("contract", "changed immutable contract field: minimum_cutoff_buffer_days"),
        ("evaluation", "changed immutable evaluation field: initial_bankroll"),
        ("baseline", "changed immutable baseline field: features_sha256"),
    ],
)
def test_confirmed_child_transition_allowlist_rejects_immutable_field_changes(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    child, binding, _claim_path = _confirmed_child_fixture(tmp_path)
    if mutation == "policy":
        child["health_limits"]["minimum_strategy_bets"] += 1
    elif mutation == "contract":
        child["contract"]["minimum_cutoff_buffer_days"] += 1
    elif mutation == "evaluation":
        child["evaluation"]["initial_bankroll"] += 1.0
    else:
        child["baseline"]["features_sha256"] = "0" * 64

    errors, _validated = gate.validate_performance_confirmation(
        child,
        repo_root=tmp_path,
        confirmation_validator=lambda _path: binding,
    )
    assert any(message in error for error in errors)


def test_confirmed_child_rejects_parent_byte_drift_and_forged_pass_parent_binding(
    tmp_path: Path,
) -> None:
    child, binding, _claim_path = _confirmed_child_fixture(tmp_path)
    parent_path = tmp_path / child["performance_confirmation"]["parent_policy_path"]
    parent_path.write_text(
        parent_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    errors, validated = gate.validate_performance_confirmation(
        child,
        repo_root=tmp_path,
        confirmation_validator=lambda _path: binding,
    )
    assert validated is None
    assert any("immutable parent policy hash changed" in error for error in errors)

    child, binding, _claim_path = _confirmed_child_fixture(tmp_path / "forged")
    forged_binding = {**binding, "policy_sha256": "0" * 64}
    errors, _validated = gate.validate_performance_confirmation(
        child,
        repo_root=tmp_path / "forged",
        confirmation_validator=lambda _path: forged_binding,
    )
    assert any("PASS result is not bound to the immutable parent policy" in error for error in errors)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("bootstrap_state", "only final_track_c_bound policy state"),
        ("policy_stale", "final Track-C receipt policy hash is stale"),
        ("receipt_binding", "receipt binding mismatch: strategy_config_sha256"),
        ("frozen_metric", "does not reproduce frozen baseline metric: strategy_roi"),
    ],
)
def test_final_receipt_rejects_bootstrap_binding_and_frozen_metric_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    fixture = _final_track_c_receipt_fixture(tmp_path, monkeypatch)
    receipt = json.loads(fixture["receipt_path"].read_text(encoding="utf-8"))
    if mutation == "bootstrap_state":
        policy = json.loads(fixture["policy_path"].read_text(encoding="utf-8"))
        parent_path = tmp_path / policy["performance_confirmation"]["parent_policy_path"]
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        policy["performance_confirmation"]["state"] = "bootstrap_pending_track_c"
        policy["baseline"] = parent["baseline"]
        _write_json(fixture["policy_path"], policy)
        receipt["final_policy_sha256"] = gate.file_sha256(fixture["policy_path"])
    elif mutation == "policy_stale":
        receipt["final_policy_sha256"] = "0" * 64
    elif mutation == "receipt_binding":
        receipt["strategy_config_sha256"] = "0" * 64
    else:
        summary_path = fixture["artifact_paths"]["summary"]
        summary = pd.read_csv(summary_path)
        summary.loc[0, "strategy_roi"] = float(summary.loc[0, "strategy_roi"]) + 0.01
        summary.to_csv(summary_path, index=False)
        receipt["artifacts"]["summary"]["sha256"] = gate.file_sha256(summary_path)
    _write_json(fixture["receipt_path"], receipt)

    with pytest.raises(gate.ContractInputError, match=message):
        gate.validate_final_track_c_pass_receipt(
            fixture["receipt_path"],
            expected_policy_path=fixture["policy_path"],
            expected_fullfit_spec=fixture["fullfit_spec"],
            repo_root=tmp_path,
            quality_validator=fixture["quality_validator"],
            confirmation_validator=fixture["confirmation_validator"],
        )


def test_bootstrap_policy_cannot_authorize_training_with_real_spec_detector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _final_track_c_receipt_fixture(tmp_path, monkeypatch)
    _use_fixture_repo(monkeypatch, tmp_path)
    policy = json.loads(fixture["policy_path"].read_text(encoding="utf-8"))
    parent_path = tmp_path / policy["performance_confirmation"]["parent_policy_path"]
    parent = gate.load_policy(parent_path)
    policy["performance_confirmation"]["state"] = "bootstrap_pending_track_c"
    policy["baseline"] = parent["baseline"]
    _write_json(fixture["policy_path"], policy)
    receipt = json.loads(fixture["receipt_path"].read_text(encoding="utf-8"))
    receipt["final_policy_sha256"] = gate.file_sha256(fixture["policy_path"])
    _write_json(fixture["receipt_path"], receipt)
    monkeypatch.setattr(
        train_module,
        "_SCHEDULED_REFIT_POLICY_PATH",
        fixture["policy_path"],
    )

    with pytest.raises(gate.ContractInputError, match="only final_track_c_bound policy state"):
        train_module.train_all_models(
            pd.DataFrame(),
            spec=fixture["fullfit_spec"],
            training_input_evidence={},
            final_track_c_receipt_path=fixture["receipt_path"],
        )
