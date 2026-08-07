import argparse
import json
from pathlib import Path

import pytest

from scripts import publish_scheduled_refit_bundle as publisher
from scripts import bfo_lineage


def _write_plan_inputs(tmp_path):
    parent = {
        "bundle_id": "parent-bundle",
        "model_spec_name": "approved-fullfit",
        "model_sha256": "a" * 64,
        "no_odds_model_sha256": "b" * 64,
        "logistic_model_sha256": "c" * 64,
        "immutable_training_fights_sha256": "d" * 64,
        "immutable_training_features_sha256": "e" * 64,
        "processed_fights_sha256": "0" * 64,
        "processed_features_sha256": "9" * 64,
        "rich_release_id": "r-" + "f" * 20,
        "installed_manifest_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "deployed_git_sha": "9" * 40,
        "training_source_git_sha": "a" * 40,
        "immutable_training_snapshot_max_event_date": "2026-08-01",
        "rich_manifest_validated": True,
    }
    readyz = {
        "ok": True,
        "ready": True,
        "requested_live_mode": "real",
        "effective_live_mode": "real",
        "armed_for_real": True,
        "trading_enabled": True,
        "trading_live": True,
        "components": {
            "betting_loop": {"state": "running", "thread_alive": True}
        },
        "production_bundle": parent,
    }
    readyz_path = tmp_path / "parent-readyz.json"
    readyz_path.write_text(json.dumps(readyz), encoding="utf-8")

    binding = {
        "policy_id": "scheduled-refit-v1",
        "sha256": "3" * 64,
        "root_bundle_id": parent["bundle_id"],
        "parent_bundle_id": parent["bundle_id"],
        "parent_model_spec_name": parent["model_spec_name"],
        "parent_model_sha256": parent["model_sha256"],
        "parent_no_odds_model_sha256": parent["no_odds_model_sha256"],
        "parent_logistic_model_sha256": parent["logistic_model_sha256"],
        "parent_processed_fights_sha256": parent[
            "immutable_training_fights_sha256"
        ],
        "parent_processed_features_sha256": parent[
            "immutable_training_features_sha256"
        ],
    }
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    lineage_payload = {
        "schema_version": 1,
        "parent_bundle_id": parent["bundle_id"],
        "parent_source_manifest_sha256": parent["source_manifest_sha256"],
        "previous_lineage_manifest_sha256": None,
        "batches": [],
    }
    lineage_path = bfo_lineage.write_manifest(
        lineage_payload,
        candidate_root / "provenance/bfo_lineage/manifest.json",
    )
    lineage_record = {
        "manifest_staged_path": "provenance/bfo_lineage/manifest.json",
        "manifest_sha256": publisher._sha256_file(lineage_path),
        "manifest_bytes": lineage_path.stat().st_size,
        "batch_count": 0,
        "batches": [],
    }
    manifest = {
        "manifest_version": 3,
        "staging_schema_version": 1,
        "bundle_id": "candidate-bundle",
        "model_spec_name": "approved-fullfit",
        "model_sha256": "4" * 64,
        "no_odds_model_sha256": "5" * 64,
        "logistic_model_sha256": "6" * 64,
        "processed_fights_sha256": "7" * 64,
        "processed_features_sha256": "8" * 64,
        "training_source_git_sha": "b" * 40,
        "snapshot_max_event_date": "2026-08-04",
        "scheduled_refit_policy": binding,
        "raw_input_provenance": {"scheduled_bfo_lineage": lineage_record},
    }
    (candidate_root / "staging_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return candidate_root, readyz_path, readyz


class ExplodingRunner:
    def run(self, _argv):
        raise AssertionError("validation-only mode must not call Railway")


def _volume_args():
    return argparse.Namespace(
        project="project-id",
        environment="environment-id",
        service="service-id",
        volume_mount_path="/app/logs",
    )


def _activation_args():
    return argparse.Namespace(
        project="project-id",
        environment="environment-id",
        service="service-id",
        volume_mount_path="/app/logs",
        target_root="/app/logs/production_bundle/store",
        expected_deployed_git_sha="9" * 40,
        readyz_url="https://example.test/readyz",
        readiness_timeout_seconds=1.0,
        poll_interval_seconds=0.001,
        ssh_identity_file=Path(__file__).resolve(),
    )


def _candidate_readyz(plan, *, thread_alive=True):
    return {
        "ok": True,
        "ready": True,
        "requested_live_mode": "real",
        "effective_live_mode": "real",
        "armed_for_real": True,
        "trading_enabled": True,
        "trading_live": True,
        "components": {
            "betting_loop": {"state": "running", "thread_alive": thread_alive}
        },
        "production_bundle": {
            "rich_manifest_validated": True,
            "bundle_id": plan.candidate_bundle_id,
            "rich_release_id": plan.candidate_release_id,
            "installed_manifest_sha256": "a" * 64,
            "source_manifest_sha256": plan.candidate_source_manifest_sha256,
            "model_spec_name": plan.candidate_model_spec_name,
            "model_sha256": plan.candidate_model_sha256,
            "no_odds_model_sha256": plan.candidate_no_odds_model_sha256,
            "logistic_model_sha256": plan.candidate_logistic_model_sha256,
            "immutable_training_fights_sha256": (
                plan.candidate_processed_fights_sha256
            ),
            "immutable_training_features_sha256": (
                plan.candidate_processed_features_sha256
            ),
            "processed_fights_sha256": plan.candidate_processed_fights_sha256,
            "processed_features_sha256": plan.candidate_processed_features_sha256,
            "training_source_git_sha": plan.candidate_training_source_git_sha,
            "immutable_training_snapshot_max_event_date": (
                plan.candidate_snapshot_max_event_date
            ),
            "deployed_git_sha": "9" * 40,
            "scheduled_bfo_lineage_manifest_sha256": (
                plan.candidate_bfo_lineage_manifest_sha256
            ),
        },
    }


class ScriptedRunner:
    def __init__(self):
        self.commands = []

    def run(self, argv):
        command = list(argv)
        self.commands.append(command)
        if "volume" in command and "list" in command:
            return json.dumps(
                {
                    "volumes": [
                        {
                            "id": "volume-id",
                            "mountPath": "/app/logs",
                            "status": "Ready",
                        }
                    ]
                }
            )
        if "upload" in command:
            return "{}"
        if "promote" in command:
            return '{"status":"promoted"}'
        if "rollback" in command:
            return '{"status":"rolled_back"}'
        raise AssertionError(f"unexpected command: {command}")

def test_validation_only_writes_report_without_railway_mutation(tmp_path):
    candidate_root, readyz_path, _readyz = _write_plan_inputs(tmp_path)
    report = tmp_path / "publish-report.json"

    result = publisher.main(
        [
            "--candidate-root",
            str(candidate_root),
            "--parent-readyz-json",
            str(readyz_path),
            "--readyz-url",
            "https://example.test/readyz",
            "--expected-deployed-git-sha",
            "9" * 40,
            "--project",
            "project-id",
            "--environment",
            "environment-id",
            "--service",
            "service-id",
            "--target-root",
            "/app/logs/production_bundle/store",
            "--volume-mount-path",
            "/app/logs",
            "--report",
            str(report),
        ],
        runner=ExplodingRunner(),
    )

    assert result == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "validated_no_mutation"
    assert payload["mode"] == "validation_only"
    assert payload["plan"]["candidate_bundle_id"] == "candidate-bundle"


def test_parent_binding_mismatch_fails_closed_and_writes_report(tmp_path):
    candidate_root, readyz_path, _readyz = _write_plan_inputs(tmp_path)
    manifest_path = candidate_root / "staging_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scheduled_refit_policy"]["parent_model_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = tmp_path / "publish-report.json"

    result = publisher.main(
        [
            "--candidate-root",
            str(candidate_root),
            "--parent-readyz-json",
            str(readyz_path),
            "--readyz-url",
            "https://example.test/readyz",
            "--expected-deployed-git-sha",
            "9" * 40,
            "--project",
            "p",
            "--environment",
            "e",
            "--service",
            "s",
            "--target-root",
            "/store",
            "--volume-mount-path",
            "/volume",
            "--report",
            str(report),
        ]
    )

    assert result == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert "parent_model_sha256" in payload["error"]


def test_installer_commands_pin_both_sides_of_compare_and_swap(tmp_path):
    candidate_root, readyz_path, _readyz = _write_plan_inputs(tmp_path)
    plan = publisher.build_publish_plan(candidate_root, readyz_path)
    args = argparse.Namespace(target_root="/store")

    promote = publisher._promote_remote_argv(plan, args, "/incoming/candidate")
    rollback = publisher._rollback_remote_argv(
        plan,
        args,
        candidate_installed_manifest_sha256="a" * 64,
    )

    assert promote[promote.index("--expected-active-bundle-id") + 1] == "parent-bundle"
    assert promote[promote.index("--expected-active-release-id") + 1] == "r-" + "f" * 20
    assert promote[promote.index("--expected-active-manifest-sha256") + 1] == "1" * 64
    assert rollback[rollback.index("--expected-active-bundle-id") + 1] == "candidate-bundle"
    assert rollback[rollback.index("--expected-rollback-bundle-id") + 1] == "parent-bundle"
    assert rollback[rollback.index("--expected-active-manifest-sha256") + 1] == "a" * 64
    assert rollback[rollback.index("--expected-rollback-manifest-sha256") + 1] == "1" * 64


def test_candidate_readyz_requires_exact_release_and_live_loop(tmp_path):
    candidate_root, readyz_path, _readyz = _write_plan_inputs(tmp_path)
    plan = publisher.build_publish_plan(candidate_root, readyz_path)
    payload = {
        "ok": True,
        "ready": True,
        "requested_live_mode": "real",
        "effective_live_mode": "real",
        "armed_for_real": True,
        "trading_enabled": True,
        "trading_live": True,
        "components": {"betting_loop": {"state": "running", "thread_alive": True}},
        "production_bundle": {
            "rich_manifest_validated": True,
            "bundle_id": plan.candidate_bundle_id,
            "rich_release_id": plan.candidate_release_id,
            "installed_manifest_sha256": "a" * 64,
            "source_manifest_sha256": plan.candidate_source_manifest_sha256,
            "model_spec_name": plan.candidate_model_spec_name,
            "model_sha256": plan.candidate_model_sha256,
            "no_odds_model_sha256": plan.candidate_no_odds_model_sha256,
            "logistic_model_sha256": plan.candidate_logistic_model_sha256,
            "immutable_training_fights_sha256": plan.candidate_processed_fights_sha256,
            "immutable_training_features_sha256": plan.candidate_processed_features_sha256,
            "processed_fights_sha256": plan.candidate_processed_fights_sha256,
            "processed_features_sha256": plan.candidate_processed_features_sha256,
            "training_source_git_sha": plan.candidate_training_source_git_sha,
            "immutable_training_snapshot_max_event_date": (
                plan.candidate_snapshot_max_event_date
            ),
            "deployed_git_sha": "9" * 40,
            "scheduled_bfo_lineage_manifest_sha256": (
                plan.candidate_bfo_lineage_manifest_sha256
            ),
        },
    }

    publisher._validate_readyz(
        payload,
        plan=plan,
        installed_manifest_sha256="a" * 64,
        expected_deployed_git_sha="9" * 40,
        candidate=True,
    )

    payload["components"]["betting_loop"]["thread_alive"] = False
    try:
        publisher._validate_readyz(
            payload,
            plan=plan,
            installed_manifest_sha256="a" * 64,
            expected_deployed_git_sha="9" * 40,
            candidate=True,
        )
    except publisher.PublishError as exc:
        assert "Betting loop" in str(exc)
    else:
        raise AssertionError("dead betting loop must fail readiness")


def test_candidate_readyz_requires_full_arming_and_training_identity(tmp_path):
    candidate_root, readyz_path, _readyz = _write_plan_inputs(tmp_path)
    plan = publisher.build_publish_plan(candidate_root, readyz_path)
    payload = {
        "ok": True,
        "ready": True,
        "requested_live_mode": "real",
        "effective_live_mode": "real",
        "armed_for_real": True,
        "trading_enabled": True,
        "trading_live": True,
        "components": {"betting_loop": {"state": "running", "thread_alive": True}},
        "production_bundle": {
            "rich_manifest_validated": True,
            "bundle_id": plan.candidate_bundle_id,
            "rich_release_id": plan.candidate_release_id,
            "installed_manifest_sha256": "a" * 64,
            "source_manifest_sha256": plan.candidate_source_manifest_sha256,
            "model_spec_name": plan.candidate_model_spec_name,
            "model_sha256": plan.candidate_model_sha256,
            "no_odds_model_sha256": plan.candidate_no_odds_model_sha256,
            "logistic_model_sha256": plan.candidate_logistic_model_sha256,
            "immutable_training_fights_sha256": plan.candidate_processed_fights_sha256,
            "immutable_training_features_sha256": plan.candidate_processed_features_sha256,
            "processed_fights_sha256": plan.candidate_processed_fights_sha256,
            "processed_features_sha256": plan.candidate_processed_features_sha256,
            "training_source_git_sha": plan.candidate_training_source_git_sha,
            "immutable_training_snapshot_max_event_date": (
                plan.candidate_snapshot_max_event_date
            ),
            "deployed_git_sha": "9" * 40,
        },
    }

    for field, bad_value in (
        ("requested_live_mode", "dry_run"),
        ("armed_for_real", False),
    ):
        changed = dict(payload)
        changed[field] = bad_value
        try:
            publisher._validate_readyz(
                changed,
                plan=plan,
                installed_manifest_sha256="a" * 64,
                expected_deployed_git_sha="9" * 40,
                candidate=True,
            )
        except publisher.PublishError:
            pass
        else:
            raise AssertionError(f"{field} mismatch must fail readiness")

    changed = dict(payload)
    changed["production_bundle"] = dict(payload["production_bundle"])
    changed["production_bundle"]["training_source_git_sha"] = "c" * 40
    try:
        publisher._validate_readyz(
            changed,
            plan=plan,
            installed_manifest_sha256="a" * 64,
            expected_deployed_git_sha="9" * 40,
            candidate=True,
        )
    except publisher.PublishError as exc:
        assert "training_source_git_sha" in str(exc)
    else:
        raise AssertionError("training source mismatch must fail readiness")


def test_status_requires_exact_project_environment_service_and_one_instance():
    payload = {
        "id": "project-id",
        "environments": {
            "edges": [
                {
                    "node": {
                        "id": "environment-id",
                        "serviceInstances": {
                            "edges": [
                                {
                                    "node": {
                                        "serviceId": "service-id",
                                        "activeDeployments": [
                                            {
                                                "id": "deployment-id",
                                                "meta": {"commitHash": "9" * 40},
                                                "instances": [
                                                    {
                                                        "id": "instance-id",
                                                        "status": "RUNNING",
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                }
                            ]
                        },
                    }
                }
            ]
        },
    }

    class StatusRunner:
        def __init__(self, value):
            self.value = value

        def run(self, _argv):
            return json.dumps(self.value)

    assert publisher._status_instance(StatusRunner(payload), _activation_args()) == (
        "deployment-id",
        "instance-id",
        "9" * 40,
    )
    payload["id"] = "wrong-project"
    with pytest.raises(publisher.PublishError, match="exact project"):
        publisher._status_instance(StatusRunner(payload), _activation_args())


def test_readiness_requires_two_consecutive_exact_samples(tmp_path):
    candidate_root, readyz_path, _readyz = _write_plan_inputs(tmp_path)
    plan = publisher.build_publish_plan(candidate_root, readyz_path)
    first = _candidate_readyz(plan)
    bad = _candidate_readyz(plan, thread_alive=False)
    third = _candidate_readyz(plan)
    fourth = _candidate_readyz(plan)
    sequence = iter([first, bad, third, fourth])

    samples = publisher._wait_for_readyz(
        url="https://example.test/readyz",
        plan=plan,
        installed_manifest_sha256="a" * 64,
        expected_deployed_git_sha="9" * 40,
        candidate=True,
        timeout_seconds=1.0,
        interval_seconds=0.001,
        fetch=lambda _url: next(sequence),
    )

    assert samples == [third, fourth]


def test_activate_uploads_unique_bundle_promotes_restarts_and_checks_readyz(
    tmp_path, monkeypatch
):
    candidate_root, readyz_path, parent_readyz = _write_plan_inputs(tmp_path)
    plan = publisher.build_publish_plan(candidate_root, readyz_path)
    parent_state = {
        "active_bundle_id": plan.parent_bundle_id,
        "active_release_id": plan.parent_release_id,
        "active_manifest_sha256": plan.parent_installed_manifest_sha256,
    }
    promoted_state = {
        "active_bundle_id": plan.candidate_bundle_id,
        "active_release_id": plan.candidate_release_id,
        "active_manifest_sha256": "a" * 64,
        "rollback_bundle_id": plan.parent_bundle_id,
        "rollback_release_id": plan.parent_release_id,
        "rollback_manifest_sha256": plan.parent_installed_manifest_sha256,
        "rollback_ready": True,
    }
    states = iter([parent_state, promoted_state])
    restarts = []
    monkeypatch.setattr(
        publisher,
        "_status_instance",
        lambda _runner, _args: ("old-deployment", "instance-id", "9" * 40),
    )
    monkeypatch.setattr(
        publisher, "_installer_resolve", lambda _runner, _args, _instance: next(states)
    )
    monkeypatch.setattr(publisher, "_fetch_readyz", lambda _url: parent_readyz)
    monkeypatch.setattr(publisher, "_restart", lambda _runner, _args: restarts.append(1))
    monkeypatch.setattr(
        publisher,
        "_wait_for_single_instance",
        lambda *_args, **_kwargs: ("new-deployment", "new-instance"),
    )
    expected_samples = [_candidate_readyz(plan), _candidate_readyz(plan)]
    monkeypatch.setattr(
        publisher,
        "_wait_for_readyz",
        lambda **_kwargs: expected_samples,
    )
    runner = ScriptedRunner()

    result = publisher.activate(plan, _activation_args(), runner=runner)

    assert result["status"] == "activated"
    assert result["deployment_id"] == "new-deployment"
    assert restarts == [1]
    upload = next(command for command in runner.commands if "upload" in command)
    remote_upload = upload[upload.index("upload") + 2]
    assert remote_upload.startswith("/production_bundle/incoming/candidate-bundle-")
    promote = next(command for command in runner.commands if "promote" in command)
    assert promote[promote.index("--expected-bundle-id") + 1] == "candidate-bundle"
    ssh_commands = [
        command for command in runner.commands if command[:2] == ["railway", "ssh"]
    ]
    assert ssh_commands
    assert all(
        command[command.index("--identity-file") + 1] == str(Path(__file__).resolve())
        for command in ssh_commands
    )
    assert not any(
        command[:2] == ["railway", "whoami"] for command in runner.commands
    )
    assert result["readiness_samples"] == expected_samples


def test_activation_failure_uses_only_installer_rollback(tmp_path, monkeypatch):
    candidate_root, readyz_path, parent_readyz = _write_plan_inputs(tmp_path)
    plan = publisher.build_publish_plan(candidate_root, readyz_path)
    parent_state = {
        "active_bundle_id": plan.parent_bundle_id,
        "active_release_id": plan.parent_release_id,
        "active_manifest_sha256": plan.parent_installed_manifest_sha256,
    }
    candidate_state = {
        "active_bundle_id": plan.candidate_bundle_id,
        "active_release_id": plan.candidate_release_id,
        "active_manifest_sha256": "a" * 64,
        "rollback_bundle_id": plan.parent_bundle_id,
        "rollback_release_id": plan.parent_release_id,
        "rollback_manifest_sha256": plan.parent_installed_manifest_sha256,
        "rollback_ready": True,
    }
    states = iter([parent_state, candidate_state, candidate_state])
    monkeypatch.setattr(
        publisher,
        "_status_instance",
        lambda _runner, _args: ("deployment", "instance-id", "9" * 40),
    )
    monkeypatch.setattr(
        publisher, "_installer_resolve", lambda _runner, _args, _instance: next(states)
    )
    monkeypatch.setattr(publisher, "_fetch_readyz", lambda _url: parent_readyz)
    restart_calls = []

    def restart(_runner, _args):
        restart_calls.append(1)
        if len(restart_calls) == 1:
            raise publisher.PublishError("candidate restart failed")

    monkeypatch.setattr(publisher, "_restart", restart)
    runner = ScriptedRunner()

    with pytest.raises(publisher.PublishError, match="installer recovery status") as exc_info:
        publisher.activate(plan, _activation_args(), runner=runner)

    recovery = exc_info.value.details["recovery"]
    assert recovery["status"] == "parent_pointer_restored"
    assert recovery["restart"] == "requested"
    assert sum("rollback" in command for command in runner.commands) == 1
    assert not any(
        any(operation in command for operation in ("download", "delete", "rename"))
        for command in runner.commands
    )
    ssh_commands = [
        command for command in runner.commands if command[:2] == ["railway", "ssh"]
    ]
    assert ssh_commands
    assert all(
        command[command.index("--identity-file") + 1] == str(Path(__file__).resolve())
        for command in ssh_commands
    )


def test_activation_requires_real_nonempty_ssh_identity_before_railway(tmp_path):
    candidate_root, readyz_path, _parent_readyz = _write_plan_inputs(tmp_path)
    plan = publisher.build_publish_plan(candidate_root, readyz_path)
    args = _activation_args()
    runner = ExplodingRunner()

    args.ssh_identity_file = None
    with pytest.raises(publisher.PublishError, match="required with --activate"):
        publisher.activate(plan, args, runner=runner)

    args.ssh_identity_file = tmp_path / "missing-key"
    with pytest.raises(publisher.PublishError, match="must be a real file"):
        publisher.activate(plan, args, runner=runner)

    empty_key = tmp_path / "empty-key"
    empty_key.touch()
    args.ssh_identity_file = empty_key
    with pytest.raises(publisher.PublishError, match="must not be empty"):
        publisher.activate(plan, args, runner=runner)


def test_volume_upload_path_must_be_inside_exact_mount():
    args = _volume_args()
    assert (
        publisher._volume_remote_path(
            args, "/app/logs/production_bundle/store/active_bundle.json"
        )
        == "/production_bundle/store/active_bundle.json"
    )
    try:
        publisher._volume_remote_path(args, "/app/models/active_bundle.json")
    except publisher.PublishError as exc:
        assert "outside Railway volume mount" in str(exc)
    else:
        raise AssertionError("an out-of-volume pointer path must be rejected")
    try:
        publisher._volume_remote_path(args, "/app/logs/../models/candidate")
    except publisher.PublishError as exc:
        assert "cannot contain '..'" in str(exc)
    else:
        raise AssertionError("a traversing volume path must be rejected")

    argv = publisher._volume_upload_argv(
        args,
        volume_id="volume-id",
        local_path="candidate",
        remote_path="/production_bundle/incoming/candidate",
    )
    assert argv[argv.index("--volume") + 1] == "volume-id"
    assert argv[argv.index("upload") + 1 :] == [
        "candidate",
        "/production_bundle/incoming/candidate",
        "--json",
    ]
