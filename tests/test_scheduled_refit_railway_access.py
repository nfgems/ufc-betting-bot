import argparse
import json
from pathlib import Path

from scripts import check_scheduled_refit_railway_access as access_check


class VolumeRunner:
    def __init__(self):
        self.commands = []

    def run(self, argv):
        command = list(argv)
        self.commands.append(command)
        assert command[:2] == ["railway", "volume"]
        return json.dumps(
            {
                "volumes": [
                    {"id": "volume-id", "mountPath": "/app/logs", "status": "Ready"}
                ]
            }
        )


def _args(tmp_path):
    identity = tmp_path / "identity"
    identity.write_text("private-key-placeholder", encoding="utf-8")
    return argparse.Namespace(
        project="project-id",
        environment="environment-id",
        service="service-id",
        target_root="/app/logs/production_bundle/store",
        volume_mount_path="/app/logs",
        readyz_url="https://example.test/readyz",
        expected_deployed_git_sha="9" * 40,
        ssh_identity_file=identity,
    )


def _installer_state():
    return {
        "active_bundle_id": "bundle-id",
        "active_release_id": "release-id",
        "active_manifest_sha256": "a" * 64,
        "rollback_bundle_id": "rollback-bundle",
        "rollback_release_id": "rollback-release",
        "rollback_manifest_sha256": "b" * 64,
        "rollback_ready": True,
    }


def _readyz():
    return {
        "ready": True,
        "ok": True,
        "production_bundle": {
            "bundle_id": "bundle-id",
            "rich_release_id": "release-id",
            "installed_manifest_sha256": "a" * 64,
            "deployed_git_sha": "9" * 40,
        },
    }


def test_access_check_verifies_exact_read_only_state(tmp_path, monkeypatch):
    args = _args(tmp_path)
    runner = VolumeRunner()
    installer_calls = []
    monkeypatch.setattr(
        access_check,
        "_status_instance",
        lambda _runner, _args: ("deployment-id", "instance-id", "9" * 40),
    )
    monkeypatch.setattr(
        access_check,
        "_installer_resolve",
        lambda _runner, _args, instance: installer_calls.append(instance)
        or _installer_state(),
    )

    result = access_check.check_access(
        args,
        runner=runner,
        fetch_readyz=lambda _url: _readyz(),
    )

    assert result["status"] == "read_only_access_verified"
    assert result["deployment_id"] == "deployment-id"
    assert result["volume_id"] == "volume-id"
    assert installer_calls == ["instance-id"]
    assert args.ssh_identity_file == Path(args.ssh_identity_file).resolve()
    assert runner.commands == [
        [
            "railway",
            "volume",
            "--project",
            "project-id",
            "--environment",
            "environment-id",
            "--service",
            "service-id",
            "list",
            "--json",
        ]
    ]


def test_access_check_fails_before_railway_for_missing_identity(tmp_path):
    args = _args(tmp_path)
    args.ssh_identity_file = tmp_path / "missing"

    class ExplodingRunner:
        def run(self, _argv):
            raise AssertionError("Railway must not run before identity validation")

    try:
        access_check.check_access(args, runner=ExplodingRunner())
    except access_check.PublishError as exc:
        assert "must be a real file" in str(exc)
    else:
        raise AssertionError("a missing identity must fail closed")


def test_access_checker_contains_no_mutating_railway_operations():
    source = Path(access_check.__file__).read_text(encoding="utf-8")
    for forbidden in (
        '"upload"',
        '"restart"',
        '"promote"',
        '"rollback"',
        '"delete"',
        '"rename"',
    ):
        assert forbidden not in source
