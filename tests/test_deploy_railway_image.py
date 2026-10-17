import copy
import json
import subprocess
from types import SimpleNamespace

import pytest

from tools import deploy_railway_image as release

COMMIT = "c" * 40
OLD = release.IMAGE_PREFIX + "a" * 64
NEW = release.IMAGE_PREFIX + "b" * 64


def raw_status():
    return {
        "id": release.PROJECT,
        "environments": {"edges": [{"node": {
            "id": release.ENVIRONMENT, "unmergedChangesCount": 0,
            "volumeInstances": {"edges": [{"node": {
                "serviceId": release.SERVICE, "mountPath": "/app/logs",
                "volume": {"id": release.VOLUME}}}]},
            "serviceInstances": {"edges": [{"node": {
                "serviceId": release.SERVICE, "environmentId": release.ENVIRONMENT,
                "source": {"image": OLD, "repo": None},
                "startCommand": "/bin/bash /app/entrypoint.sh", "numReplicas": None,
                "cronSchedule": None,
                "latestDeployment": {"id": "previous", "status": "SUCCESS", "meta": {
                    "image": OLD, "registryCredentials": {"password": "PRIVATE-SECRET"}}},
                "activeDeployments": [{"id": "previous", "status": "SUCCESS"}],
            }}]},
        }}]},
    }


def raw_configuration():
    return {"services": {release.SERVICE: {
        "source": {"image": OLD, "autoUpdates": False},
        "volumeMounts": {release.VOLUME: {"mountPath": "/app/logs"}},
        "variables": {"CONTROL": {"value": "PRIVATE-SECRET"}},
        "deploy": {"healthcheckPath": "/readyz", "registryCredentials": {
            "username": "owner", "password": "PRIVATE-REGISTRY-SECRET"}},
        "build": {"builder": "DOCKERFILE"},
    }}, "volumes": {release.VOLUME: {"name": "ufc-bot-volume"}}}


def mock_read(command):
    return raw_configuration() if "config" in command else raw_status()


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    monkeypatch.setattr(release, "command_json", mock_read)
    state = release.snapshot()
    monkeypatch.setattr(release, "check_main", lambda commit: None)
    monkeypatch.setattr(release, "snapshot", lambda **kwargs: copy.deepcopy(state))
    path = tmp_path / "release.json"
    release.prepare(COMMIT, path)
    return path, state


def successor(before, status="SUCCESS"):
    after = copy.deepcopy(before)
    after["image"] = NEW
    after["deployment"] = {"id": "candidate", "status": status, "image": NEW}
    after["active_deployments"] = [{"id": "candidate", "status": status}]
    return after


def test_snapshot_never_exports_registry_metadata(monkeypatch):
    monkeypatch.setattr(release, "command_json", mock_read)
    state = release.snapshot()
    release.require_stable(state)
    serialized = json.dumps(state)
    assert "PRIVATE-SECRET" not in serialized
    assert "registryCredentials" not in serialized
    assert state["settings"]["volumes"] == [{"mount_path": "/app/logs", "volume_id": release.VOLUME,
                                               "deleted_at": None, "pending_deletion": False}]


@pytest.mark.parametrize("damage", ["wrong_project", "repo_source", "staged", "missing_volume", "swapped_volume"])
def test_snapshot_refuses_wrong_target_or_configuration(monkeypatch, damage):
    raw = raw_status()
    environment = raw["environments"]["edges"][0]["node"]
    service = environment["serviceInstances"]["edges"][0]["node"]
    if damage == "wrong_project":
        raw["id"] = "other"
    elif damage == "repo_source":
        service["source"]["repo"] = release.REPOSITORY
    elif damage == "staged":
        environment["unmergedChangesCount"] = 1
    elif damage == "missing_volume":
        environment["volumeInstances"]["edges"] = []
    else:
        environment["volumeInstances"]["edges"][0]["node"]["volume"]["id"] = "other"
    monkeypatch.setattr(release, "command_json", lambda command: raw)
    with pytest.raises(release.ReleaseError):
        release.snapshot()


def test_stale_commit_stops_before_any_railway_call(monkeypatch, tmp_path):
    calls = []
    def run(command):
        calls.append(command)
        return {"object": {"sha": "d" * 40}}
    monkeypatch.setattr(release, "command_json", run)
    with pytest.raises(release.ReleaseError, match="no longer GitHub main"):
        release.prepare(COMMIT, tmp_path / "release.json")
    assert len(calls) == 1 and calls[0][0] == "gh"


def test_inflight_deployment_blocks_prepare(monkeypatch, tmp_path):
    monkeypatch.setattr(release, "command_json", mock_read)
    state = release.snapshot()
    state["active_deployments"].append({"id": "competing", "status": "DEPLOYING"})
    monkeypatch.setattr(release, "snapshot", lambda **kwargs: state)
    monkeypatch.setattr(release, "check_main", lambda commit: None)
    with pytest.raises(release.ReleaseError, match="competing"):
        release.prepare(COMMIT, tmp_path / "release.json")


def test_changed_production_blocks_mutation(monkeypatch, prepared):
    path, before = prepared
    monkeypatch.setattr(release, "snapshot", lambda **kwargs: successor(before))
    monkeypatch.setattr(release, "command_json", lambda command: pytest.fail("mutation attempted"))
    with pytest.raises(release.ReleaseError, match="changed since preparation"):
        release.deploy(COMMIT, NEW, path)
    assert json.loads(path.read_text())["status"] == "prepared"


def test_exactly_one_mutation_then_digest_and_health_verified(monkeypatch, prepared):
    path, before = prepared
    after = successor(before)
    snapshots = iter([before, after, after])
    monkeypatch.setattr(release, "snapshot", lambda **kwargs: next(snapshots))
    health_calls, commands = [], []
    monkeypatch.setattr(release, "healthy", lambda: health_calls.append(True) or True)
    def run(command):
        commands.append(command)
        # Intent is durable before mutation, so a lost response cannot replay it.
        assert json.loads(path.read_text())["status"] == "source_update_requested"
        return {"id": release.SERVICE, "image": NEW, "password": "PRIVATE-SECRET"}
    monkeypatch.setattr(release, "command_json", run)
    result = release.deploy(COMMIT, NEW, path)
    assert result["status"] == "verified"
    assert result["before"]["image"] == OLD
    assert result["after"]["deployment"]["id"] == "candidate"
    assert commands == [["railway", "service", "source", "connect", *release.TARGET,
                         "--image", NEW, "--json"]]
    assert len(health_calls) == 1
    assert "PRIVATE-SECRET" not in path.read_text()
    with pytest.raises(release.ReleaseError, match="already consumed"):
        release.deploy(COMMIT, NEW, path)


def test_lost_mutation_response_consumes_receipt_without_retry(monkeypatch, prepared):
    path, before = prepared
    commands = []
    def run(command):
        commands.append(command)
        raise release.ReleaseError("Release CLI command could not complete")
    monkeypatch.setattr(release, "command_json", run)
    with pytest.raises(release.ReleaseError):
        release.deploy(COMMIT, NEW, path)
    assert len(commands) == 1
    assert json.loads(path.read_text())["status"] == "verification_failed"
    with pytest.raises(release.ReleaseError, match="already consumed"):
        release.deploy(COMMIT, NEW, path)
    assert len(commands) == 1


@pytest.mark.parametrize("problem", ["failed", "competing", "changed_mount"])
def test_failed_or_competing_rollout_never_redeploys_or_restores_state(monkeypatch, prepared, problem):
    path, before = prepared
    after = successor(before)
    if problem == "failed":
        after["deployment"]["status"] = "FAILED"
    elif problem == "competing":
        after["image"] = release.IMAGE_PREFIX + "d" * 64
    else:
        after["settings"]["volumes"] = [{"mount_path": "/other"}]
    states = iter([before, after])
    monkeypatch.setattr(release, "snapshot", lambda **kwargs: next(states))
    commands = []
    monkeypatch.setattr(release, "command_json", lambda command: commands.append(command) or {
        "id": release.SERVICE, "image": NEW})
    with pytest.raises(release.ReleaseError):
        release.deploy(COMMIT, NEW, path)
    assert len(commands) == 1
    result = json.loads(path.read_text())
    assert result["status"] == "verification_failed"
    assert result["before"]["image"] == OLD


@pytest.mark.parametrize("failure", ["exit", "timeout", "invalid_json"])
def test_cli_failures_do_not_echo_credentials(monkeypatch, failure):
    def run(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], 90, output="PRIVATE-SECRET")
        return SimpleNamespace(returncode=1 if failure == "exit" else 0,
                               stdout="PRIVATE-SECRET", stderr="PRIVATE-SECRET")
    monkeypatch.setattr(release.subprocess, "run", run)
    with pytest.raises(release.ReleaseError) as error:
        release.command_json(["railway", "status", "--json"])
    assert "PRIVATE-SECRET" not in str(error.value)


@pytest.mark.parametrize("image", ["ghcr.io/nfgems/ufc-betting-bot:latest",
                                   "ghcr.io/other/image@sha256:" + "b" * 64])
def test_mutable_or_foreign_image_is_rejected(monkeypatch, prepared, image):
    path, _ = prepared
    monkeypatch.setattr(release, "command_json", lambda command: pytest.fail("mutation attempted"))
    with pytest.raises(release.ReleaseError, match="digest-pinned"):
        release.deploy(COMMIT, image, path)


def test_configuration_fingerprint_ignores_only_image(monkeypatch):
    configuration = raw_configuration()
    monkeypatch.setattr(release, "command_json", lambda command: copy.deepcopy(configuration))
    before = release.configuration_fingerprint(OLD)
    configuration["services"][release.SERVICE]["source"]["image"] = NEW
    assert release.configuration_fingerprint(NEW) == before
    configuration["services"][release.SERVICE]["variables"]["CONTROL"]["value"] = "changed"
    assert release.configuration_fingerprint(NEW) != before
    assert "PRIVATE-SECRET" not in before


def test_configuration_source_race_is_rejected(monkeypatch):
    monkeypatch.setattr(release, "command_json", lambda command: raw_configuration())
    with pytest.raises(release.ReleaseError, match="disagree"):
        release.configuration_fingerprint(NEW)


def test_only_expected_transition_mismatch_is_retryable(monkeypatch):
    monkeypatch.setattr(release, "command_json", lambda command: raw_configuration())
    baseline = release.configuration_fingerprint(OLD)
    with pytest.raises(release.PropagationPending) as pending:
        release.configuration_fingerprint(NEW, transition_images=(OLD, NEW))
    assert pending.value.configuration_sha256 == baseline
    with pytest.raises(release.ReleaseError, match="disagree"):
        release.configuration_fingerprint(release.IMAGE_PREFIX + "d" * 64,
                                          transition_images=(OLD, NEW))


def fake_clock(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(release.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(release.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    return clock


def test_prior_and_split_reads_converge_without_repeating_mutation(monkeypatch, prepared):
    path, before = prepared
    after = successor(before)
    clock = fake_clock(monkeypatch)
    reads = iter([before, before,
                  release.PropagationPending(before["settings"]["configuration_sha256"]),
                  after, after])
    def read(**kwargs):
        value = next(reads)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(release, "snapshot", read)
    monkeypatch.setattr(release, "healthy", lambda: True)
    commands = []
    monkeypatch.setattr(release, "command_json", lambda command: commands.append(command) or {
        "id": release.SERVICE, "image": NEW})
    assert release.deploy(COMMIT, NEW, path)["status"] == "verified"
    assert clock[0] == 20
    assert len(commands) == 1


@pytest.mark.parametrize("split", [False, True])
def test_propagation_grace_expires_at_thirty_seconds(monkeypatch, prepared, split):
    path, before = prepared
    clock = fake_clock(monkeypatch)
    calls = [0]
    def read(**kwargs):
        calls[0] += 1
        if calls[0] > 1 and split:
            raise release.PropagationPending(before["settings"]["configuration_sha256"])
        return before
    monkeypatch.setattr(release, "snapshot", read)
    commands = []
    monkeypatch.setattr(release, "command_json", lambda command: commands.append(command) or {
        "id": release.SERVICE, "image": NEW})
    with pytest.raises(release.ReleaseError, match="after acknowledgement"):
        release.deploy(COMMIT, NEW, path, interval=11)
    assert clock[0] == 30
    assert len(commands) == 1


def test_config_change_during_split_reads_fails_without_grace(monkeypatch, prepared):
    path, before = prepared
    clock = fake_clock(monkeypatch)
    calls = [0]
    def read(**kwargs):
        calls[0] += 1
        if calls[0] > 1:
            raise release.PropagationPending("different-configuration")
        return before
    monkeypatch.setattr(release, "snapshot", read)
    commands = []
    monkeypatch.setattr(release, "command_json", lambda command: commands.append(command) or {
        "id": release.SERVICE, "image": NEW})
    with pytest.raises(release.ReleaseError, match="configuration changed"):
        release.deploy(COMMIT, NEW, path)
    assert clock[0] == 0
    assert len(commands) == 1
