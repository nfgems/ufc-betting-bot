"""Release one tested immutable image without changing service settings or data.

The prepare receipt must precede the private image build. Deploy refuses a stale
Git commit, concurrent Railway change, staged configuration, or active rollout.
The mutation is one environment-scoped sparse patch containing source.image only.
CLI responses can contain registry credentials: they are parsed in memory and
never logged, attached to errors, or copied into the public release receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

PROJECT = "0f1fd486-bc96-4fb7-8c23-3f56ff8dbb2b"
ENVIRONMENT = "7b128011-3c33-4ada-8602-a8ab1863ad3e"
SERVICE = "702f574c-389b-48ea-911f-a1054b3e4405"
VOLUME = "a9c56c45-be3f-4ff6-b0e8-208204bb2712"
REPOSITORY = "nfgems/ufc-betting-bot"
IMAGE_PREFIX = "ghcr.io/nfgems/ufc-betting-bot@sha256:"
URL = "https://ufc-bot-production.up.railway.app"
ENVIRONMENT_TARGET = ["--project", PROJECT, "--environment", ENVIRONMENT]
TERMINAL_FAILURES = {"FAILED", "CRASHED", "REMOVED", "SKIPPED"}


class ReleaseError(RuntimeError):
    """An intentionally credential-free release failure."""


class CommandError(ReleaseError):
    """Only static operation/category names and an exit code may be reported."""

    def __init__(self, operation, category, exit_code=None):
        self.details = {"operation": operation, "category": category, "exit_code": exit_code}
        super().__init__(f"{operation}: {category}; raw CLI output withheld")


class PropagationPending(ReleaseError):
    """Two reads disagree only about the acknowledged old/new image change."""

    def __init__(self, configuration_sha256):
        super().__init__("Railway source update has not propagated across reads")
        self.configuration_sha256 = configuration_sha256


def require(condition, message):
    if not condition:
        raise ReleaseError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def valid_image(value):
    return isinstance(value, str) and re.fullmatch(
        re.escape(IMAGE_PREFIX) + r"[0-9a-f]{64}", value
    ) is not None


def command_operation(command):
    for prefix, operation in (
        (["railway", "environment", "edit"], "railway_environment_image_patch"),
        (["railway", "environment", "config"], "railway_environment_read"),
        (["railway", "status"], "railway_status_read"),
        (["gh", "api"], "github_api_read"),
    ):
        if command[:len(prefix)] == prefix:
            return operation
    return "external_cli"


def failure_category(output):
    # Never return a vendor message or extracted value. Output may contain secrets.
    text = output.lower()
    if "project token" in text and any(value in text for value in ("not supported", "unsupported", "cannot")):
        return "project_token_scope_unsupported"
    if any(value in text for value in ("unauthorized", "not authorized", "invalid token", "token not found")):
        return "authentication_rejected"
    if "rate limit" in text or "too many requests" in text:
        return "rate_limited"
    if any(value in text for value in ("unexpected argument", "unrecognized subcommand", "invalid value")):
        return "invalid_arguments"
    if any(value in text for value in ("error sending request", "connection refused", "connection reset", "timed out")):
        return "network_failure"
    if any(value in text for value in ("project not found", "service not found", "environment not found")):
        return "target_not_found"
    return "command_rejected"


def command_json(command, *, input_json=None):
    operation = command_operation(command)
    executable = shutil.which(command[0]) or command[0]
    try:
        result = subprocess.run([executable, *command[1:]], capture_output=True,
                                text=True, timeout=90,
                                input=json.dumps(input_json) if input_json is not None else None)
    except subprocess.TimeoutExpired:
        raise CommandError(operation, "timeout") from None
    except OSError:
        raise CommandError(operation, "process_start_failed") from None
    if result.returncode != 0:
        raise CommandError(operation, failure_category(result.stdout + result.stderr), result.returncode)
    try:
        return json.loads(result.stdout)
    except (TypeError, ValueError):
        raise CommandError(operation, "invalid_json") from None


def check_main(commit):
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit),
            "Expected a full Git commit SHA")
    ref = command_json(["gh", "api", f"repos/{REPOSITORY}/git/ref/heads/main"])
    require(ref.get("object", {}).get("sha") == commit,
            "Release commit is no longer GitHub main; refusing stale deployment")


def one(nodes, predicate, message):
    found = [node for node in nodes if predicate(node)]
    require(len(found) == 1, message)
    return found[0]


def nodes(connection):
    return [edge["node"] for edge in connection.get("edges", [])]


def configuration_fingerprint(expected_image, *, transition_images=()):
    configuration = command_json(["railway", "environment", "config", "--environment",
                                  ENVIRONMENT, "--json"])
    services = configuration.get("services") or {}
    require(SERVICE in services, "Expected service is missing from environment configuration")
    selected = services[SERVICE]
    source = selected.get("source") or {}
    configured_image = source.get("image")
    require(source.get("repo") is None and valid_image(configured_image),
            "Unexpected source in Railway environment configuration")
    require(selected.get("volumeMounts", {}).get(VOLUME, {}).get("mountPath") == "/app/logs",
            "Persistent volume configuration changed")
    # Hash the whole configured environment, not individual secret values.
    # Image is the sole expected configuration change. CLI configuration excludes
    # runtime timestamps, usage and generated variables; all other fields remain.
    del source["image"]
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()
    if configured_image != expected_image:
        if expected_image in transition_images and configured_image in transition_images:
            raise PropagationPending(fingerprint)
        raise ReleaseError("Railway status and configuration disagree about production source")
    return fingerprint


def snapshot(*, transition_images=()):
    raw = command_json(["railway", "status", "--project", PROJECT,
                        "--environment", ENVIRONMENT, "--json"])
    require(raw.get("id") == PROJECT, "Unexpected Railway project")
    environment = one(nodes(raw.get("environments", {})),
                      lambda item: item.get("id") == ENVIRONMENT,
                      "Expected production environment was not found uniquely")
    require(not environment.get("deletedAt"), "Production environment is deleted")
    require(environment.get("unmergedChangesCount") in (None, 0),
            "Railway has staged configuration changes")
    service = one(nodes(environment.get("serviceInstances", {})),
                  lambda item: item.get("serviceId") == SERVICE,
                  "Expected production service was not found uniquely")
    require(service.get("environmentId") == ENVIRONMENT, "Unexpected service environment")
    source = service.get("source") or {}
    require(source.get("repo") is None and valid_image(source.get("image")),
            "Production source must remain the expected private immutable image")
    deployment = service.get("latestDeployment") or {}
    meta = deployment.get("meta") or {}
    require(isinstance(deployment.get("id"), str), "Production deployment is missing")
    require(valid_image(meta.get("image")), "Deployment image metadata is missing")
    volumes = sorted([
        {"mount_path": item.get("mountPath"), "volume_id": (item.get("volume") or {}).get("id"),
         "deleted_at": item.get("deletedAt"), "pending_deletion": bool(item.get("isPendingDeletion"))}
        for item in nodes(environment.get("volumeInstances", {}))
        if item.get("serviceId") == SERVICE
    ], key=lambda item: str(item["mount_path"]))
    require(volumes == [{"mount_path": "/app/logs", "volume_id": VOLUME,
                         "deleted_at": None, "pending_deletion": False}],
            "Expected persistent production volume mount is missing or changed")
    active = sorted([
        {"id": item.get("id"), "status": item.get("status")}
        for item in service.get("activeDeployments", [])
    ], key=lambda item: str(item["id"]))
    # Only this explicit allowlist can leave the process. Never serialize raw.
    return {
        "project": PROJECT, "environment": ENVIRONMENT, "service": SERVICE,
        "image": source["image"],
        "deployment": {"id": deployment["id"], "status": deployment.get("status"),
                       "image": meta["image"]},
        "active_deployments": active,
        "settings": {"start_command": service.get("startCommand"),
                     "replicas": service.get("numReplicas"),
                     "cron_schedule": service.get("cronSchedule"), "volumes": volumes,
                     "configuration_sha256": configuration_fingerprint(
                         source["image"], transition_images=transition_images)},
    }


def require_stable(state):
    deployment = state["deployment"]
    require(deployment["status"] == "SUCCESS" and deployment["image"] == state["image"],
            "Production has an incomplete or unsuccessful deployment")
    require(state["active_deployments"] == [{"id": deployment["id"], "status": "SUCCESS"}],
            "Production has competing or incomplete active deployments")


def save(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize(receipt):
    destination = os.environ.get("GITHUB_STEP_SUMMARY")
    if not destination:
        return
    lines = ["### Railway production release", "",
             f"- Status: `{receipt['status']}`",
             f"- Git commit: `{receipt['commit']}`"]
    if receipt.get("candidate_image"):
        lines.append(f"- Candidate image: `{receipt['candidate_image']}`")
    if receipt.get("after"):
        deployment = receipt["after"]["deployment"]["id"]
        lines.append(f"- Verified Railway deployment: `{deployment}`")
        lines.append(f"- Health and readiness: [production]({URL})")
    lines.append(f"- Previous image for code rollback: `{receipt['before']['image']}`")
    lines.append("- Rollback must preserve the existing volume and newer trading/accounting records.")
    with open(destination, "a", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")


def prepare(commit, path):
    require(not path.exists(), "Release receipt already exists; do not overwrite it")
    check_main(commit)
    state = snapshot()
    require_stable(state)
    receipt = {"schema_version": 1, "status": "prepared", "commit": commit,
               "prepared_at": now(), "before": state}
    save(path, receipt)
    return receipt


def healthy():
    for route in ("/healthz", "/readyz"):
        try:
            with urlopen(URL + route, timeout=20) as response:
                if response.status != 200:
                    return False
        except (OSError, URLError):
            return False
    return True


def deploy(commit, image, path, *, timeout=900, interval=10):
    require(valid_image(image), "Candidate must be a digest-pinned private production image")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    require(receipt.get("schema_version") == 1 and receipt.get("status") == "prepared"
            and receipt.get("commit") == commit, "Invalid or already consumed prepare receipt")
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(receipt["prepared_at"])).total_seconds()
    require(0 <= age < 3600, "Prepare receipt expired; inspect production before retrying")
    check_main(commit)
    before = snapshot()
    require_stable(before)
    require(before == receipt["before"], "Production changed since preparation; refusing rollout")
    require(before["image"] != image, "Candidate image is already configured")
    receipt.update(status="source_update_requested", candidate_image=image, requested_at=now())
    # Consume before sending: timeout/retry must not trigger a duplicate mutation.
    save(path, receipt)
    try:
        # CLI v5.23.3 environment edit reads JSON stdin before flags and issues
        # one environmentPatchCommit(environmentId, patch). It does not commit
        # the environment's existing staged patch. Missing fields are omitted.
        response = command_json(["railway", "environment", "edit",
                                 *ENVIRONMENT_TARGET, "--json"],
                                input_json={"services": {SERVICE: {"source": {"image": image}}}})
        require(response.get("committed") is True and response.get("environmentId") == ENVIRONMENT,
                "Unexpected environment-patch acknowledgement; inspect Railway before retrying")
        deadline = time.monotonic() + timeout
        propagation_deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                current = snapshot(transition_images=(before["image"], image))
            except PropagationPending as pending:
                require(pending.configuration_sha256 == before["settings"]["configuration_sha256"],
                        "Production configuration changed during source propagation")
                require(time.monotonic() < propagation_deadline,
                        "Railway source/configuration did not converge after acknowledgement")
                time.sleep(min(interval, max(0, propagation_deadline - time.monotonic())))
                continue
            if current == before:
                require(time.monotonic() < propagation_deadline,
                        "Railway source update did not propagate after acknowledgement")
                time.sleep(min(interval, max(0, propagation_deadline - time.monotonic())))
                continue
            require(current["image"] == image, "Railway source changed during rollout")
            require(current["settings"] == before["settings"],
                    "Production startup or persistent storage configuration changed")
            latest = current["deployment"]
            if latest["id"] != before["deployment"]["id"]:
                require(latest["image"] == image, "A competing deployment appeared during rollout")
                require(latest["status"] not in TERMINAL_FAILURES,
                        "Candidate deployment failed; preserve volume and inspect code rollback")
                if (latest["status"] == "SUCCESS"
                        and current["active_deployments"] == [{"id": latest["id"], "status": "SUCCESS"}]
                        and healthy()):
                    final = snapshot()
                    require(final == current, "Railway changed during health verification")
                    receipt.update(status="verified", verified_at=now(), after=final,
                                   health_paths=["/healthz", "/readyz"])
                    save(path, receipt)
                    summarize(receipt)
                    return receipt
            time.sleep(interval)
        raise ReleaseError("Timed out waiting for candidate deployment and readiness; inspect Railway")
    except Exception as error:
        receipt.update(status="verification_failed", failed_at=now())
        if isinstance(error, CommandError):
            receipt["failure"] = error.details
        save(path, receipt)
        summarize(receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "deploy"))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--image")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.operation == "prepare":
            result = prepare(args.commit, args.report)
        else:
            result = deploy(args.commit, args.image, args.report)
        print(json.dumps(result))
    except Exception as error:
        # Even unexpected parse/transport exceptions must not expose raw metadata.
        print(str(error) if isinstance(error, ReleaseError)
              else "Release operation failed; inspect the sanitized receipt", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
