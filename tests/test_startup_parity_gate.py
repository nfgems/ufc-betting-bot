from __future__ import annotations

import json
import sys
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import startup_parity_gate
from src.web import serve as web_serve
from src.web import startup_parity


@dataclass(frozen=True)
class _FakeSpec:
    name: str
    feature_cols: list[str]


def _summary() -> dict:
    return {
        "bundle_id": "release-1",
        "rich_release_id": "r-source",
        "source_manifest_sha256": "source",
        "installed_manifest_sha256": "installed",
        "model_spec_name": "integrity_spec",
        "model_sha256": "model",
        "no_odds_model_sha256": "no-odds",
        "logistic_model_sha256": "logistic",
        "processed_fights_sha256": "fights",
        "processed_features_sha256": "features",
        "immutable_training_fights_sha256": "fights",
        "immutable_training_features_sha256": "features",
        "scheduled_refit_policy": {"policy_schema_version": 2},
        "confirmed_strategy": {
            "schema_version": 1,
            "strategy_config_sha256": "c" * 64,
        },
    }


def _strict_summary(tmp_path: Path) -> dict:
    installed_manifest = tmp_path / "manifest.json"
    runtime_manifest = tmp_path / "runtime_manifest.json"
    installed_manifest.write_text('{"manifest": "installed"}\n', encoding="utf-8")
    runtime_manifest.write_text('{"manifest": "runtime"}\n', encoding="utf-8")
    installed_sha = startup_parity._file_sha256(installed_manifest)
    return {
        "bundle_id": "release-1",
        "manifest_path": str(runtime_manifest),
        "installed_manifest_path": str(installed_manifest),
        "installed_manifest_sha256": installed_sha,
        "model_spec_name": "full_live_contract_v6_integrity_205_fullfit",
        "immutable_training_fights_sha256": "a" * 64,
        "immutable_training_features_sha256": "b" * 64,
        "scheduled_refit_policy": {"policy_schema_version": 2},
        "confirmed_strategy": {
            "schema_version": 1,
            "strategy_config_sha256": "c" * 64,
        },
    }


def _strict_receipt(summary: dict, *, deployed_sha: str = "d" * 40) -> dict:
    spec = startup_parity.resolve_named_training_spec(summary["model_spec_name"])
    binding = {
        "deployed_git_sha": deployed_sha,
        "bundle_id": summary["bundle_id"],
        "model_spec_name": summary["model_spec_name"],
        "registered_spec_sha256": startup_parity._canonical_sha256(asdict(spec)),
        "confirmed_strategy_sha256": startup_parity._canonical_sha256(
            summary["confirmed_strategy"]
        ),
        "source_manifest_sha256": startup_parity._file_sha256(
            Path(summary["installed_manifest_path"])
        ),
        "runtime_manifest_sha256": startup_parity._file_sha256(
            Path(summary["manifest_path"])
        ),
        "installed_manifest_sha256": summary["installed_manifest_sha256"],
        "immutable_fights_sha256": summary["immutable_training_fights_sha256"],
        "immutable_features_sha256": summary[
            "immutable_training_features_sha256"
        ],
        "replay_args": {
            "start": "2025-01-01",
            "limit": 250,
            "seed": 7,
            "tol": 1e-6,
            "exact": {"mode": "exact", "established_only": False},
            "prefight": {"mode": "prefight", "established_only": True},
        },
    }
    report = {
        "path": "parity.json",
        "sha256": "c" * 64,
        "n_fights": 10,
        "requested_feature_count": 205,
        "forbidden_mismatch_count": 0,
    }
    return {
        "schema_version": 1,
        "receipt_key": startup_parity._canonical_sha256(binding),
        "passed": True,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "binding": binding,
        "reports": {"exact": dict(report), "prefight": dict(report)},
    }


def _write_strict_receipt(tmp_path: Path, payload: dict) -> Path:
    receipt = tmp_path / "startup_receipt.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    return receipt


def test_require_same_release_allows_newer_mutable_runtime_lookup():
    runtime = _summary()
    runtime["processed_fights_sha256"] = "newer-fights"
    runtime["processed_features_sha256"] = "different"

    startup_parity_gate._require_same_release(_summary(), runtime)


@pytest.mark.parametrize(
    "field",
    [
        "rich_release_id",
        "source_manifest_sha256",
        "installed_manifest_sha256",
        "model_spec_name",
        "model_sha256",
        "no_odds_model_sha256",
        "logistic_model_sha256",
        "immutable_training_fights_sha256",
        "immutable_training_features_sha256",
    ],
)
def test_require_same_release_rejects_immutable_release_mismatch(field):
    runtime = _summary()
    runtime[field] = "different"

    with pytest.raises(RuntimeError, match=field):
        startup_parity_gate._require_same_release(_summary(), runtime)


def test_startup_receipt_checks_immutable_snapshot_not_mutable_lookup(
    tmp_path, monkeypatch
):
    receipt = tmp_path / "startup_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "passed": True,
                "binding": {
                    "deployed_git_sha": "deploy-a",
                    "bundle_id": "release-1",
                    "model_spec_name": "integrity_spec",
                    "immutable_fights_sha256": "fights",
                    "immutable_features_sha256": "features",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("UFC_STARTUP_PARITY_RECEIPT", str(receipt))
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "deploy-a")
    summary = _summary()
    summary["processed_fights_sha256"] = "newer-fights"
    summary["processed_features_sha256"] = "newer-features"

    loaded = web_serve._load_startup_parity_receipt(summary)

    assert loaded["binding"]["immutable_fights_sha256"] == "fights"


def test_startup_receipt_rejects_immutable_snapshot_mismatch(tmp_path, monkeypatch):
    receipt = tmp_path / "startup_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "passed": True,
                "binding": {
                    "deployed_git_sha": "deploy-a",
                    "bundle_id": "release-1",
                    "model_spec_name": "integrity_spec",
                    "immutable_fights_sha256": "fights",
                    "immutable_features_sha256": "features",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("UFC_STARTUP_PARITY_RECEIPT", str(receipt))
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "deploy-a")
    summary = _summary()
    summary["immutable_training_features_sha256"] = "different"

    with pytest.raises(RuntimeError, match="immutable_features_sha256"):
        web_serve._load_startup_parity_receipt(summary)


def test_hosted_real_money_receipt_requires_exact_active_release_binding(
    tmp_path, monkeypatch
):
    summary = _strict_summary(tmp_path)
    deployed_sha = "d" * 40
    receipt = _write_strict_receipt(
        tmp_path,
        _strict_receipt(summary, deployed_sha=deployed_sha),
    )
    monkeypatch.setenv("UFC_STARTUP_PARITY_RECEIPT", str(receipt))
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", deployed_sha)

    loaded = web_serve._load_startup_parity_receipt(
        summary,
        require_exact_binding=True,
    )

    assert loaded["binding"]["source_manifest_sha256"] == summary[
        "installed_manifest_sha256"
    ]
    assert loaded["binding"]["runtime_manifest_sha256"] == (
        startup_parity._file_sha256(Path(summary["manifest_path"]))
    )


@pytest.mark.parametrize("missing_finality", ["scheduled_refit_policy", "confirmed_strategy"])
def test_hosted_real_money_receipt_rejects_nonfinal_bundle_summary(
    tmp_path,
    monkeypatch,
    missing_finality,
):
    summary = _strict_summary(tmp_path)
    payload = _strict_receipt(summary)
    receipt = _write_strict_receipt(tmp_path, payload)
    summary.pop(missing_finality)
    monkeypatch.setenv("UFC_STARTUP_PARITY_RECEIPT", str(receipt))
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "d" * 40)

    with pytest.raises(RuntimeError, match="final-v2|confirmed strategy"):
        web_serve._load_startup_parity_receipt(
            summary,
            require_exact_binding=True,
        )


def test_hosted_real_money_receipt_binds_exact_confirmed_strategy(
    tmp_path,
    monkeypatch,
):
    summary = _strict_summary(tmp_path)
    receipt = _write_strict_receipt(tmp_path, _strict_receipt(summary))
    summary["confirmed_strategy"] = {
        **summary["confirmed_strategy"],
        "strategy_config_sha256": "e" * 64,
    }
    monkeypatch.setenv("UFC_STARTUP_PARITY_RECEIPT", str(receipt))
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "d" * 40)

    with pytest.raises(RuntimeError, match="confirmed_strategy_sha256"):
        web_serve._load_startup_parity_receipt(
            summary,
            require_exact_binding=True,
        )


@pytest.mark.parametrize(
    "field",
    ["deployed_git_sha", "registered_spec_sha256"],
)
def test_hosted_real_money_receipt_rejects_release_identity_drift(
    tmp_path,
    monkeypatch,
    field,
):
    summary = _strict_summary(tmp_path)
    deployed_sha = "d" * 40
    payload = _strict_receipt(summary, deployed_sha=deployed_sha)
    payload["binding"][field] = "e" * (40 if field == "deployed_git_sha" else 64)
    payload["receipt_key"] = startup_parity._canonical_sha256(payload["binding"])
    receipt = _write_strict_receipt(tmp_path, payload)
    monkeypatch.setenv("UFC_STARTUP_PARITY_RECEIPT", str(receipt))
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", deployed_sha)

    with pytest.raises(RuntimeError, match=field):
        web_serve._load_startup_parity_receipt(
            summary,
            require_exact_binding=True,
        )


@pytest.mark.parametrize("missing_field", sorted(startup_parity.REQUIRED_BINDING_FIELDS))
def test_hosted_real_money_receipt_rejects_every_omitted_binding_field(
    tmp_path,
    monkeypatch,
    missing_field,
):
    summary = _strict_summary(tmp_path)
    deployed_sha = "d" * 40
    payload = _strict_receipt(summary, deployed_sha=deployed_sha)
    payload["binding"].pop(missing_field)
    payload["receipt_key"] = startup_parity._canonical_sha256(payload["binding"])
    receipt = _write_strict_receipt(tmp_path, payload)
    monkeypatch.setenv("UFC_STARTUP_PARITY_RECEIPT", str(receipt))
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", deployed_sha)

    with pytest.raises(RuntimeError, match="binding fields are not exact"):
        web_serve._load_startup_parity_receipt(
            summary,
            require_exact_binding=True,
        )


@pytest.mark.parametrize(
    "field",
    [
        "source_manifest_sha256",
        "runtime_manifest_sha256",
        "installed_manifest_sha256",
    ],
)
def test_hosted_real_money_receipt_rejects_manifest_hash_mismatch(
    tmp_path,
    monkeypatch,
    field,
):
    summary = _strict_summary(tmp_path)
    deployed_sha = "d" * 40
    payload = _strict_receipt(summary, deployed_sha=deployed_sha)
    payload["binding"][field] = "f" * 64
    payload["receipt_key"] = startup_parity._canonical_sha256(payload["binding"])
    receipt = _write_strict_receipt(tmp_path, payload)
    monkeypatch.setenv("UFC_STARTUP_PARITY_RECEIPT", str(receipt))
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", deployed_sha)

    with pytest.raises(RuntimeError, match=field):
        web_serve._load_startup_parity_receipt(
            summary,
            require_exact_binding=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("start", "2099-01-01"), ("limit", 1), ("seed", 99), ("tol", 1.0)],
)
def test_hosted_real_money_receipt_rejects_weakened_replay_contract(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    summary = _strict_summary(tmp_path)
    deployed_sha = "d" * 40
    payload = _strict_receipt(summary, deployed_sha=deployed_sha)
    payload["binding"]["replay_args"][field] = value
    payload["receipt_key"] = startup_parity._canonical_sha256(payload["binding"])
    receipt = _write_strict_receipt(tmp_path, payload)
    monkeypatch.setenv("UFC_STARTUP_PARITY_RECEIPT", str(receipt))
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", deployed_sha)

    with pytest.raises(RuntimeError, match="hosted readiness contract"):
        web_serve._load_startup_parity_receipt(
            summary,
            require_exact_binding=True,
        )


def test_hosted_real_money_receipt_is_mandatory_but_optional_modes_remain_compatible(
    tmp_path,
    monkeypatch,
):
    summary = _strict_summary(tmp_path)
    monkeypatch.delenv("UFC_STARTUP_PARITY_RECEIPT", raising=False)

    assert (
        web_serve._load_startup_parity_receipt(
            summary,
            require_exact_binding=False,
        )
        is None
    )
    with pytest.raises(RuntimeError, match="requires UFC_STARTUP_PARITY_RECEIPT"):
        web_serve._load_startup_parity_receipt(
            summary,
            require_exact_binding=True,
        )


@pytest.mark.parametrize("receipt_state", ["missing", "tampered"])
def test_hosted_real_money_serve_aborts_before_startup_without_exact_receipt(
    tmp_path,
    monkeypatch,
    receipt_state,
):
    summary = _strict_summary(tmp_path)
    monkeypatch.setenv("RAILWAY_ENVIRONMENT_NAME", "test")
    monkeypatch.setenv("LIVE_TRADING_MODE", "real")
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "d" * 40)
    monkeypatch.setattr(
        web_serve,
        "_resolve_hosted_bundle_startup_summary",
        lambda: summary,
    )
    if receipt_state == "missing":
        monkeypatch.delenv("UFC_STARTUP_PARITY_RECEIPT", raising=False)
    else:
        payload = _strict_receipt(summary)
        payload["binding"]["bundle_id"] = "tampered-release"
        receipt = _write_strict_receipt(tmp_path, payload)
        monkeypatch.setenv("UFC_STARTUP_PARITY_RECEIPT", str(receipt))

    with pytest.raises(SystemExit) as exc_info:
        web_serve.main()

    assert exc_info.value.code == 1


def test_startup_receipt_is_reused_only_for_exact_binding(tmp_path, monkeypatch):
    source_manifest = tmp_path / "source.json"
    runtime_manifest = tmp_path / "runtime.json"
    source_manifest.write_text('{"source": 1}', encoding="utf-8")
    runtime_manifest.write_text('{"runtime": 1}', encoding="utf-8")
    receipt = tmp_path / "startup_receipt.json"

    def fake_load(path):
        manifest = Path(path)
        return SimpleNamespace(
            manifest_path=manifest,
            bundle_id="release-1",
            model_spec_name="integrity_spec",
        )

    monkeypatch.setattr(startup_parity_gate, "load_production_bundle", fake_load)
    monkeypatch.setattr(
        startup_parity_gate,
        "validate_production_bundle",
        lambda _bundle: _summary(),
    )
    monkeypatch.setattr(
        startup_parity_gate,
        "resolve_named_training_spec",
        lambda _name: _FakeSpec("integrity_spec", ["feature"]),
    )
    calls = []

    def fake_run(command, check):
        assert check is False
        calls.append(command)
        out_path = Path(command[command.index("--out") + 1])
        out_path.write_text(
            json.dumps(
                {
                    "passed": True,
                    "n_fights": 10,
                    "requested_feature_count": 1,
                    "forbidden_mismatch_count": 0,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(startup_parity_gate.subprocess, "run", fake_run)

    def run(deployed_sha: str) -> int:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "startup_parity_gate.py",
                "--source-manifest",
                str(source_manifest),
                "--runtime-manifest",
                str(runtime_manifest),
                "--receipt",
                str(receipt),
                "--deployed-sha",
                deployed_sha,
            ],
        )
        return startup_parity_gate.main()

    assert run("deploy-a") == 0
    assert len(calls) == 2
    assert all(
        Path(command[command.index("--manifest") + 1]) == source_manifest
        for command in calls
    )
    first = json.loads(receipt.read_text(encoding="utf-8"))
    assert first["passed"] is True
    assert first["binding"]["deployed_git_sha"] == "deploy-a"
    assert set(first["binding"]) == startup_parity.REQUIRED_BINDING_FIELDS
    assert first["binding"]["source_manifest_sha256"] == (
        startup_parity._file_sha256(source_manifest)
    )
    assert first["binding"]["runtime_manifest_sha256"] == (
        startup_parity._file_sha256(runtime_manifest)
    )

    assert run("deploy-a") == 0
    assert len(calls) == 2

    assert run("deploy-b") == 0
    assert len(calls) == 4
    changed = json.loads(receipt.read_text(encoding="utf-8"))
    assert changed["receipt_key"] != first["receipt_key"]


def test_startup_gate_propagates_parity_replay_failure(tmp_path, monkeypatch):
    source_manifest = tmp_path / "source.json"
    runtime_manifest = tmp_path / "runtime.json"
    receipt = tmp_path / "startup_receipt.json"
    source_manifest.write_text('{"source": 1}', encoding="utf-8")
    runtime_manifest.write_text('{"runtime": 1}', encoding="utf-8")

    def fake_load(path):
        manifest = Path(path)
        return SimpleNamespace(
            manifest_path=manifest,
            bundle_id="release-1",
            model_spec_name="integrity_spec",
        )

    monkeypatch.setattr(startup_parity_gate, "load_production_bundle", fake_load)
    monkeypatch.setattr(
        startup_parity_gate,
        "validate_production_bundle",
        lambda _bundle: _summary(),
    )
    monkeypatch.setattr(
        startup_parity_gate,
        "resolve_named_training_spec",
        lambda _name: _FakeSpec("integrity_spec", ["feature"]),
    )
    monkeypatch.setattr(
        startup_parity_gate.subprocess,
        "run",
        lambda _command, check: SimpleNamespace(returncode=7),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "startup_parity_gate.py",
            "--source-manifest",
            str(source_manifest),
            "--runtime-manifest",
            str(runtime_manifest),
            "--receipt",
            str(receipt),
            "--deployed-sha",
            "deploy-a",
        ],
    )

    assert startup_parity_gate.main() == 7
    assert not receipt.exists()
