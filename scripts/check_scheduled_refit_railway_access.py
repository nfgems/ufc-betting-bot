"""Prove scheduled-refit Railway access without mutating production."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.publish_scheduled_refit_bundle import (  # noqa: E402
    PublishError,
    RailwayRunner,
    _fetch_readyz,
    _installer_resolve,
    _json_from_output,
    _railway_target_flags,
    _status_instance,
    _write_report,
)


def _require_identity_file(value: Path) -> Path:
    path = value.expanduser()
    if path.is_symlink() or not path.is_file():
        raise PublishError("--ssh-identity-file must be a real file")
    if path.stat().st_size == 0:
        raise PublishError("--ssh-identity-file must not be empty")
    return path.resolve(strict=True)


def _require_sha(value: object, *, label: str, length: int) -> str:
    normalized = str(value or "").strip().lower()
    if not re.fullmatch(rf"[0-9a-f]{{{length}}}", normalized):
        raise PublishError(f"{label} must be {length} lowercase hex characters")
    return normalized


def _validate_installer_state(payload: dict[str, Any]) -> dict[str, Any]:
    required_text = (
        "active_bundle_id",
        "active_release_id",
        "rollback_bundle_id",
        "rollback_release_id",
    )
    missing = [key for key in required_text if not str(payload.get(key) or "").strip()]
    if missing:
        raise PublishError(f"Installer state is missing required identities: {missing}")
    if payload.get("rollback_ready") is not True:
        raise PublishError("Installer state does not have an exact rollback ready")
    return {
        "active_bundle_id": str(payload["active_bundle_id"]),
        "active_release_id": str(payload["active_release_id"]),
        "active_manifest_sha256": _require_sha(
            payload.get("active_manifest_sha256"),
            label="active manifest SHA-256",
            length=64,
        ),
        "rollback_bundle_id": str(payload["rollback_bundle_id"]),
        "rollback_release_id": str(payload["rollback_release_id"]),
        "rollback_manifest_sha256": _require_sha(
            payload.get("rollback_manifest_sha256"),
            label="rollback manifest SHA-256",
            length=64,
        ),
        "rollback_ready": True,
    }


def check_access(
    args: argparse.Namespace,
    *,
    runner: RailwayRunner,
    fetch_readyz: Callable[[str], dict[str, Any]] = _fetch_readyz,
) -> dict[str, Any]:
    expected_git_sha = _require_sha(
        args.expected_deployed_git_sha,
        label="expected deployed Git SHA",
        length=40,
    )
    args.ssh_identity_file = _require_identity_file(args.ssh_identity_file)

    deployment_id, instance_id, deployed_git_sha = _status_instance(runner, args)
    if deployed_git_sha != expected_git_sha:
        raise PublishError(
            f"Railway deployed Git SHA mismatch: {deployed_git_sha} != {expected_git_sha}"
        )
    installer_state = _validate_installer_state(
        _installer_resolve(runner, args, instance_id)
    )

    volume_output = runner.run(
        ["railway", "volume", *_railway_target_flags(args), "list", "--json"]
    )
    volume_payload = _json_from_output(volume_output, label="railway volume list")
    volumes = volume_payload.get("volumes") if isinstance(volume_payload, dict) else None
    matching = [
        volume
        for volume in volumes or []
        if volume.get("mountPath") == args.volume_mount_path
        and volume.get("status") == "Ready"
    ]
    if len(matching) != 1 or not str(matching[0].get("id") or "").strip():
        raise PublishError("Expected exactly one ready Railway volume at the mount path")

    readyz = fetch_readyz(args.readyz_url)
    bundle = readyz.get("production_bundle")
    if readyz.get("ready") is not True or readyz.get("ok") is not True:
        raise PublishError("Runtime is not ready")
    if not isinstance(bundle, dict) or bundle.get("deployed_git_sha") != expected_git_sha:
        raise PublishError("Runtime deployed Git SHA does not match the access-check SHA")
    if (
        bundle.get("bundle_id") != installer_state["active_bundle_id"]
        or bundle.get("rich_release_id") != installer_state["active_release_id"]
        or bundle.get("installed_manifest_sha256")
        != installer_state["active_manifest_sha256"]
    ):
        raise PublishError("Runtime identity does not match the installer active state")

    return {
        "schema_version": 1,
        "status": "read_only_access_verified",
        "project_id": args.project,
        "environment_id": args.environment,
        "service_id": args.service,
        "deployment_id": deployment_id,
        "instance_id": instance_id,
        "deployed_git_sha": deployed_git_sha,
        "volume_id": str(matching[0]["id"]),
        "volume_mount_path": args.volume_mount_path,
        "ready": True,
        "ok": True,
        "installer": installer_state,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--volume-mount-path", required=True)
    parser.add_argument("--readyz-url", required=True)
    parser.add_argument("--expected-deployed-git-sha", required=True)
    parser.add_argument("--ssh-identity-file", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: RailwayRunner | None = None,
    fetch_readyz: Callable[[str], dict[str, Any]] = _fetch_readyz,
) -> int:
    args = _parser().parse_args(argv)
    report: dict[str, Any] = {"schema_version": 1, "status": "error"}
    try:
        report = check_access(
            args,
            runner=runner or RailwayRunner(),
            fetch_readyz=fetch_readyz,
        )
        _write_report(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (PublishError, OSError, ValueError) as exc:
        report["error"] = str(exc)
        _write_report(args.report, report)
        print(f"RAILWAY ACCESS CHECK FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
