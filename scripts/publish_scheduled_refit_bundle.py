"""Atomically publish an approved scheduled-refit bundle to Railway.

The default mode is validation-only and performs no Railway mutation.  The
explicit ``--activate`` path uploads to a unique volume directory, promotes by
compare-and-swap through ``install_staged_production_bundle.py``, restarts the
existing deployment, and requires two exact readiness samples.  If readiness
fails after promotion, it attempts the installer's exact pointer rollback and
restarts again.  Uploaded and installed evidence is deliberately retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
BUNDLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
SCHEDULED_POLICY_KEYS = {
    "policy_id",
    "sha256",
    "root_bundle_id",
    "parent_bundle_id",
    "parent_model_spec_name",
    "parent_model_sha256",
    "parent_no_odds_model_sha256",
    "parent_logistic_model_sha256",
    "parent_processed_fights_sha256",
    "parent_processed_features_sha256",
}


class PublishError(RuntimeError):
    """Raised when publication cannot continue without weakening a gate."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.details = details


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PublishError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublishError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PublishError(f"{label} must be a JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    target = path.expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _require_sha256(value: object, *, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(normalized):
        raise PublishError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _require_git_sha(value: object, *, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", normalized):
        raise PublishError(f"{label} must be a lowercase 40-character Git SHA")
    return normalized


def _require_iso_date(value: object, *, label: str) -> str:
    normalized = str(value or "").strip()
    try:
        parsed = date.fromisoformat(normalized)
    except ValueError as exc:
        raise PublishError(f"{label} must be an ISO date in YYYY-MM-DD form") from exc
    if parsed.isoformat() != normalized:
        raise PublishError(f"{label} must be an ISO date in YYYY-MM-DD form")
    return normalized


@dataclass(frozen=True)
class PublishPlan:
    candidate_root: str
    candidate_bundle_id: str
    candidate_source_manifest_sha256: str
    candidate_release_id: str
    candidate_model_spec_name: str
    candidate_model_sha256: str
    candidate_no_odds_model_sha256: str
    candidate_logistic_model_sha256: str
    candidate_processed_fights_sha256: str
    candidate_processed_features_sha256: str
    candidate_training_source_git_sha: str
    candidate_snapshot_max_event_date: str
    candidate_bfo_lineage_manifest_sha256: str
    parent_bundle_id: str
    parent_release_id: str
    parent_installed_manifest_sha256: str
    parent_source_manifest_sha256: str
    parent_model_spec_name: str
    parent_model_sha256: str
    parent_no_odds_model_sha256: str
    parent_logistic_model_sha256: str
    parent_processed_fights_sha256: str
    parent_processed_features_sha256: str
    parent_runtime_fights_sha256: str
    parent_runtime_features_sha256: str
    parent_training_source_git_sha: str
    parent_snapshot_max_event_date: str
    parent_bfo_lineage_manifest_sha256: str | None
    policy_id: str
    policy_sha256: str


def _assert_safe_candidate_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise PublishError(f"Candidate root must be a real directory: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PublishError(f"Candidate tree contains a link: {path}")
        if path.is_dir() and not any(path.iterdir()):
            raise PublishError(f"Candidate tree contains an undeclared empty directory: {path}")


def build_publish_plan(candidate_root: Path, parent_readyz_path: Path) -> PublishPlan:
    root = candidate_root.expanduser().resolve(strict=True)
    _assert_safe_candidate_tree(root)
    manifest_candidates = [
        path
        for path in (root / "staging_manifest.json", root / "manifest.json")
        if path.is_file()
    ]
    if len(manifest_candidates) != 1:
        raise PublishError(
            "Candidate root must contain exactly one direct child "
            "staging_manifest.json or manifest.json"
        )
    manifest_path = manifest_candidates[0]
    if manifest_path.resolve(strict=True).parent != root:
        raise PublishError("Candidate manifest must be a real direct child file")
    manifest = _load_json_object(manifest_path, label="candidate manifest")
    if manifest.get("manifest_version") != 3 or manifest.get("staging_schema_version") != 1:
        raise PublishError("Candidate must use rich manifest v3 / staging schema v1")

    bundle_id = str(manifest.get("bundle_id") or "").strip()
    if not BUNDLE_ID_RE.fullmatch(bundle_id):
        raise PublishError(f"Candidate bundle_id is unsafe: {bundle_id!r}")
    source_manifest_sha = _sha256_file(manifest_path)
    release_id = f"r-{source_manifest_sha[:20]}"

    binding = manifest.get("scheduled_refit_policy")
    if not isinstance(binding, dict) or set(binding) != SCHEDULED_POLICY_KEYS:
        raise PublishError(
            "Candidate scheduled_refit_policy must contain exactly the approved binding keys"
        )
    policy_id = str(binding.get("policy_id") or "").strip()
    if not policy_id:
        raise PublishError("Candidate scheduled-refit policy_id is empty")
    policy_sha = _require_sha256(binding.get("sha256"), label="policy sha256")

    readyz = _load_json_object(parent_readyz_path, label="parent readyz")
    if readyz.get("ready") is not True or readyz.get("ok") is not True:
        raise PublishError("Frozen parent readyz was not healthy")
    if (
        readyz.get("requested_live_mode") != "real"
        or readyz.get("effective_live_mode") != "real"
        or readyz.get("armed_for_real") is not True
        or readyz.get("trading_enabled") is not True
        or readyz.get("trading_live") is not True
    ):
        raise PublishError("Frozen parent readyz was not armed in requested/effective real mode")
    betting = readyz.get("components", {}).get("betting_loop", {})
    if betting.get("state") != "running" or betting.get("thread_alive") is not True:
        raise PublishError("Frozen parent readyz did not have a running live betting loop")
    parent = readyz.get("production_bundle")
    if not isinstance(parent, dict) or parent.get("rich_manifest_validated") is not True:
        raise PublishError("Frozen parent readyz lacks a validated rich production bundle")

    parent_values = {
        "parent_bundle_id": str(parent.get("bundle_id") or ""),
        "parent_model_spec_name": str(parent.get("model_spec_name") or ""),
        "parent_model_sha256": _require_sha256(
            parent.get("model_sha256"), label="parent primary model"
        ),
        "parent_no_odds_model_sha256": _require_sha256(
            parent.get("no_odds_model_sha256"), label="parent no-odds model"
        ),
        "parent_logistic_model_sha256": _require_sha256(
            parent.get("logistic_model_sha256"), label="parent logistic model"
        ),
        "parent_processed_fights_sha256": _require_sha256(
            parent.get("immutable_training_fights_sha256"),
            label="parent immutable fights",
        ),
        "parent_processed_features_sha256": _require_sha256(
            parent.get("immutable_training_features_sha256"),
            label="parent immutable features",
        ),
    }
    for key, expected in parent_values.items():
        if binding.get(key) != expected:
            raise PublishError(
                f"Candidate scheduled-refit parent binding mismatch for {key}: "
                f"{binding.get(key)!r} != {expected!r}"
            )

    parent_release_id = str(parent.get("rich_release_id") or "").strip()
    if not re.fullmatch(r"r-[0-9a-f]{20}", parent_release_id):
        raise PublishError("Parent readyz lacks an immutable rich release id")
    parent_installed_manifest_sha = _require_sha256(
        parent.get("installed_manifest_sha256"), label="parent installed manifest"
    )
    parent_source_manifest_sha = _require_sha256(
        parent.get("source_manifest_sha256"), label="parent source manifest"
    )
    parent_runtime_fights_sha = _require_sha256(
        parent.get("processed_fights_sha256"), label="parent runtime fights"
    )
    parent_runtime_features_sha = _require_sha256(
        parent.get("processed_features_sha256"), label="parent runtime features"
    )
    parent_training_source_git_sha = _require_git_sha(
        parent.get("training_source_git_sha"), label="parent training source Git SHA"
    )
    parent_snapshot_max_event_date = _require_iso_date(
        parent.get("immutable_training_snapshot_max_event_date"),
        label="parent immutable snapshot maximum event date",
    )
    candidate_snapshot_max_event_date = _require_iso_date(
        manifest.get("snapshot_max_event_date"),
        label="candidate snapshot maximum event date",
    )
    if date.fromisoformat(candidate_snapshot_max_event_date) < date.fromisoformat(
        parent_snapshot_max_event_date
    ):
        raise PublishError("Candidate snapshot predates the active parent snapshot")

    parent_lineage_value = parent.get("scheduled_bfo_lineage_manifest_sha256")
    if parent_lineage_value is None:
        if parent.get("bundle_id") != binding.get("root_bundle_id"):
            raise PublishError("Non-root parent lacks its scheduled BFO lineage identity")
        parent_lineage_sha: str | None = None
    else:
        parent_lineage_sha = _require_sha256(
            parent_lineage_value,
            label="parent scheduled BFO lineage manifest",
        )
    raw_provenance = manifest.get("raw_input_provenance")
    lineage = (
        raw_provenance.get("scheduled_bfo_lineage")
        if isinstance(raw_provenance, dict)
        else None
    )
    expected_lineage_keys = {
        "manifest_staged_path",
        "manifest_sha256",
        "manifest_bytes",
        "batch_count",
        "batches",
    }
    if not isinstance(lineage, dict) or set(lineage) != expected_lineage_keys:
        raise PublishError("Candidate lacks an exact scheduled BFO lineage identity")
    lineage_relative = "provenance/bfo_lineage/manifest.json"
    if lineage.get("manifest_staged_path") != lineage_relative:
        raise PublishError("Candidate scheduled BFO lineage manifest path is not exact")
    lineage_path = (root / lineage_relative).resolve(strict=True)
    try:
        lineage_path.relative_to(root)
    except ValueError as exc:
        raise PublishError("Candidate scheduled BFO lineage manifest escapes its root") from exc
    candidate_lineage_sha = _require_sha256(
        lineage.get("manifest_sha256"),
        label="candidate scheduled BFO lineage manifest",
    )
    if (
        _sha256_file(lineage_path) != candidate_lineage_sha
        or lineage_path.stat().st_size != lineage.get("manifest_bytes")
    ):
        raise PublishError("Candidate scheduled BFO lineage manifest identity changed")
    # Keep Railway access helpers importable on a fresh runner without loading
    # the model/data dependency stack needed only for candidate validation.
    from scripts import bfo_lineage

    try:
        lineage_payload = bfo_lineage.validate_package(
            lineage_path,
            expected_manifest_sha256=candidate_lineage_sha,
        )
    except (bfo_lineage.BfoLineageError, OSError) as exc:
        raise PublishError(f"Candidate scheduled BFO lineage package is invalid: {exc}") from exc
    if (
        lineage.get("batches") != lineage_payload.get("batches")
        or lineage.get("batch_count") != len(lineage_payload["batches"])
        or lineage_payload.get("parent_bundle_id") != parent.get("bundle_id")
        or lineage_payload.get("parent_source_manifest_sha256")
        != parent_source_manifest_sha
        or lineage_payload.get("previous_lineage_manifest_sha256")
        != parent_lineage_sha
    ):
        raise PublishError("Candidate scheduled BFO lineage does not chain to its parent")

    return PublishPlan(
        candidate_root=str(root),
        candidate_bundle_id=bundle_id,
        candidate_source_manifest_sha256=source_manifest_sha,
        candidate_release_id=release_id,
        candidate_model_spec_name=str(manifest.get("model_spec_name") or ""),
        candidate_model_sha256=_require_sha256(
            manifest.get("model_sha256"), label="candidate primary model"
        ),
        candidate_no_odds_model_sha256=_require_sha256(
            manifest.get("no_odds_model_sha256"), label="candidate no-odds model"
        ),
        candidate_logistic_model_sha256=_require_sha256(
            manifest.get("logistic_model_sha256"), label="candidate logistic model"
        ),
        candidate_processed_fights_sha256=_require_sha256(
            manifest.get("processed_fights_sha256"), label="candidate fights"
        ),
        candidate_processed_features_sha256=_require_sha256(
            manifest.get("processed_features_sha256"), label="candidate features"
        ),
        candidate_training_source_git_sha=_require_git_sha(
            manifest.get("training_source_git_sha"),
            label="candidate training source Git SHA",
        ),
        candidate_snapshot_max_event_date=candidate_snapshot_max_event_date,
        candidate_bfo_lineage_manifest_sha256=candidate_lineage_sha,
        parent_bundle_id=parent_values["parent_bundle_id"],
        parent_release_id=parent_release_id,
        parent_installed_manifest_sha256=parent_installed_manifest_sha,
        parent_source_manifest_sha256=parent_source_manifest_sha,
        parent_model_spec_name=parent_values["parent_model_spec_name"],
        parent_model_sha256=parent_values["parent_model_sha256"],
        parent_no_odds_model_sha256=parent_values["parent_no_odds_model_sha256"],
        parent_logistic_model_sha256=parent_values["parent_logistic_model_sha256"],
        parent_processed_fights_sha256=parent_values[
            "parent_processed_fights_sha256"
        ],
        parent_processed_features_sha256=parent_values[
            "parent_processed_features_sha256"
        ],
        parent_runtime_fights_sha256=parent_runtime_fights_sha,
        parent_runtime_features_sha256=parent_runtime_features_sha,
        parent_training_source_git_sha=parent_training_source_git_sha,
        parent_snapshot_max_event_date=parent_snapshot_max_event_date,
        parent_bfo_lineage_manifest_sha256=parent_lineage_sha,
        policy_id=policy_id,
        policy_sha256=policy_sha,
    )


class RailwayRunner:
    def run(self, argv: Sequence[str]) -> str:
        result = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise PublishError(f"Command failed ({argv[0]} {argv[1]}): {detail}")
        return result.stdout


def _json_from_output(output: str, *, label: str) -> Any:
    decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_keys)
    for index, char in enumerate(output):
        if char not in "[{":
            continue
        try:
            payload, _end = decoder.raw_decode(output[index:])
        except (json.JSONDecodeError, PublishError):
            continue
        return payload
    raise PublishError(f"{label} did not emit JSON")


def _railway_target_flags(args: argparse.Namespace) -> list[str]:
    return [
        "--project",
        args.project,
        "--environment",
        args.environment,
        "--service",
        args.service,
    ]


def _volume_remote_path(args: argparse.Namespace, container_path: str) -> str:
    """Convert an absolute container path to the same path inside the volume."""

    mount = PurePosixPath(str(args.volume_mount_path))
    target = PurePosixPath(str(container_path))
    if not mount.is_absolute() or not target.is_absolute():
        raise PublishError("Railway volume and target paths must be absolute")
    if ".." in mount.parts or ".." in target.parts:
        raise PublishError("Railway volume and target paths cannot contain '..'")
    try:
        relative = target.relative_to(mount)
    except ValueError as exc:
        raise PublishError(
            f"Target path is outside Railway volume mount {mount}: {target}"
        ) from exc
    return "/" if not relative.parts else f"/{relative.as_posix()}"


def _volume_upload_argv(
    args: argparse.Namespace,
    *,
    volume_id: str,
    local_path: str,
    remote_path: str,
) -> list[str]:
    return [
        "railway",
        "volume",
        *_railway_target_flags(args),
        "files",
        "--volume",
        volume_id,
        "upload",
        local_path,
        remote_path,
        "--json",
    ]


def _status_instance(
    runner: RailwayRunner,
    args: argparse.Namespace,
) -> tuple[str, str, str]:
    output = runner.run(
        [
            "railway",
            "status",
            "--project",
            args.project,
            "--environment",
            args.environment,
            "--json",
        ]
    )
    payload = _json_from_output(output, label="railway status")
    if str(payload.get("id") or "") != args.project:
        raise PublishError("Railway status did not contain the exact project")
    environments = payload.get("environments", {}).get("edges", [])
    selected_environment = [
        edge.get("node", {})
        for edge in environments
        if str(edge.get("node", {}).get("id")) == args.environment
    ]
    if len(selected_environment) != 1:
        raise PublishError("Railway status did not contain the exact environment")
    service_edges = (
        selected_environment[0].get("serviceInstances", {}).get("edges", [])
    )
    selected_services = [
        edge.get("node", {})
        for edge in service_edges
        if str(edge.get("node", {}).get("serviceId")) == args.service
    ]
    if len(selected_services) != 1:
        raise PublishError("Railway status did not contain the exact service")
    deployments = selected_services[0].get("activeDeployments")
    if not isinstance(deployments, list) or len(deployments) != 1:
        raise PublishError("Railway must have exactly one active deployment")
    deployment = deployments[0]
    instances = deployment.get("instances")
    running = [
        instance
        for instance in instances or []
        if instance.get("status") == "RUNNING"
    ]
    if len(instances or []) != 1 or len(running) != 1:
        raise PublishError("Railway must have exactly one RUNNING deployment instance")
    commit_sha = str(deployment.get("meta", {}).get("commitHash") or "").lower()
    return str(deployment.get("id") or ""), str(running[0].get("id") or ""), commit_sha


def _ssh_command(
    args: argparse.Namespace,
    instance_id: str,
    remote_argv: Sequence[str],
) -> list[str]:
    return [
        "railway",
        "ssh",
        *_railway_target_flags(args),
        "--identity-file",
        str(args.ssh_identity_file),
        "--deployment-instance",
        instance_id,
        *remote_argv,
    ]


def _installer_resolve(
    runner: RailwayRunner,
    args: argparse.Namespace,
    instance_id: str,
) -> dict[str, Any]:
    output = runner.run(
        _ssh_command(
            args,
            instance_id,
            [
                "python",
                "/app/scripts/install_staged_production_bundle.py",
                "resolve",
                "--target-root",
                args.target_root,
            ],
        )
    )
    payload = _json_from_output(output, label="installer resolve")
    if not isinstance(payload, dict):
        raise PublishError("Installer resolve output must be an object")
    return payload


def _assert_parent_resolve(resolve: dict[str, Any], plan: PublishPlan) -> None:
    expected = {
        "active_bundle_id": plan.parent_bundle_id,
        "active_release_id": plan.parent_release_id,
        "active_manifest_sha256": plan.parent_installed_manifest_sha256,
    }
    mismatches = [key for key, value in expected.items() if resolve.get(key) != value]
    if mismatches:
        raise PublishError(f"Installer parent CAS precondition failed: {mismatches}")


def _fetch_readyz(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "Cache-Control": "no-cache"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise PublishError(f"Readiness request failed: {exc}") from exc
    try:
        payload = json.loads(body, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublishError(f"Readiness response was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PublishError("Readiness response must be a JSON object")
    return payload


def _validate_readyz(
    payload: dict[str, Any],
    *,
    plan: PublishPlan,
    installed_manifest_sha256: str,
    expected_deployed_git_sha: str,
    candidate: bool,
) -> None:
    if payload.get("ready") is not True or payload.get("ok") is not True:
        raise PublishError("Runtime is not ready")
    if (
        payload.get("requested_live_mode") != "real"
        or payload.get("effective_live_mode") != "real"
        or payload.get("armed_for_real") is not True
        or payload.get("trading_enabled") is not True
        or payload.get("trading_live") is not True
    ):
        raise PublishError("Runtime is not armed and trading live in requested/effective real mode")
    betting = payload.get("components", {}).get("betting_loop", {})
    if betting.get("state") != "running" or betting.get("thread_alive") is not True:
        raise PublishError("Betting loop is not running with a live thread")
    bundle = payload.get("production_bundle")
    if not isinstance(bundle, dict) or bundle.get("rich_manifest_validated") is not True:
        raise PublishError("Runtime does not report a validated rich production bundle")

    if candidate:
        expected = {
            "bundle_id": plan.candidate_bundle_id,
            "rich_release_id": plan.candidate_release_id,
            "installed_manifest_sha256": installed_manifest_sha256,
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
            "deployed_git_sha": expected_deployed_git_sha,
            "scheduled_bfo_lineage_manifest_sha256": (
                plan.candidate_bfo_lineage_manifest_sha256
            ),
        }
    else:
        expected = {
            "bundle_id": plan.parent_bundle_id,
            "rich_release_id": plan.parent_release_id,
            "installed_manifest_sha256": plan.parent_installed_manifest_sha256,
            "source_manifest_sha256": plan.parent_source_manifest_sha256,
            "model_spec_name": plan.parent_model_spec_name,
            "model_sha256": plan.parent_model_sha256,
            "no_odds_model_sha256": plan.parent_no_odds_model_sha256,
            "logistic_model_sha256": plan.parent_logistic_model_sha256,
            "immutable_training_fights_sha256": plan.parent_processed_fights_sha256,
            "immutable_training_features_sha256": plan.parent_processed_features_sha256,
            "processed_fights_sha256": plan.parent_runtime_fights_sha256,
            "processed_features_sha256": plan.parent_runtime_features_sha256,
            "training_source_git_sha": plan.parent_training_source_git_sha,
            "immutable_training_snapshot_max_event_date": (
                plan.parent_snapshot_max_event_date
            ),
            "deployed_git_sha": expected_deployed_git_sha,
            "scheduled_bfo_lineage_manifest_sha256": (
                plan.parent_bfo_lineage_manifest_sha256
            ),
        }
    mismatches = [key for key, value in expected.items() if bundle.get(key) != value]
    if mismatches:
        raise PublishError(f"Runtime bundle identity mismatch: {mismatches}")


def _wait_for_readyz(
    *,
    url: str,
    plan: PublishPlan,
    installed_manifest_sha256: str,
    expected_deployed_git_sha: str,
    candidate: bool,
    timeout_seconds: float,
    interval_seconds: float,
    fetch: Callable[[str], dict[str, Any]] = _fetch_readyz,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    consecutive: list[dict[str, Any]] = []
    last_error = "no readiness response"
    while time.monotonic() < deadline:
        try:
            payload = fetch(url)
            _validate_readyz(
                payload,
                plan=plan,
                installed_manifest_sha256=installed_manifest_sha256,
                expected_deployed_git_sha=expected_deployed_git_sha,
                candidate=candidate,
            )
            consecutive.append(payload)
            if len(consecutive) == 2:
                return consecutive
        except PublishError as exc:
            last_error = str(exc)
            consecutive.clear()
        time.sleep(interval_seconds)
    raise PublishError(f"Timed out waiting for two exact readiness samples: {last_error}")


def _restart(runner: RailwayRunner, args: argparse.Namespace) -> None:
    runner.run(
        [
            "railway",
            "restart",
            *_railway_target_flags(args),
            "--yes",
            "--json",
        ]
    )


def _wait_for_single_instance(
    runner: RailwayRunner,
    args: argparse.Namespace,
    *,
    expected_git_sha: str,
    timeout_seconds: float,
    interval_seconds: float,
) -> tuple[str, str]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "no status"
    while time.monotonic() < deadline:
        try:
            deployment_id, instance_id, commit_sha = _status_instance(runner, args)
            if commit_sha != expected_git_sha:
                raise PublishError(
                    f"Railway deployed Git SHA mismatch: {commit_sha} != {expected_git_sha}"
                )
            return deployment_id, instance_id
        except PublishError as exc:
            last_error = str(exc)
        time.sleep(interval_seconds)
    raise PublishError(f"Timed out waiting for one Railway instance: {last_error}")


def _promote_remote_argv(
    plan: PublishPlan,
    args: argparse.Namespace,
    remote_source_root: str,
) -> list[str]:
    return [
        "python",
        "/app/scripts/install_staged_production_bundle.py",
        "promote",
        "--source-root",
        remote_source_root,
        "--target-root",
        args.target_root,
        "--expected-bundle-id",
        plan.candidate_bundle_id,
        "--expected-source-manifest-sha256",
        plan.candidate_source_manifest_sha256,
        "--expected-active-bundle-id",
        plan.parent_bundle_id,
        "--expected-active-release-id",
        plan.parent_release_id,
        "--expected-active-manifest-sha256",
        plan.parent_installed_manifest_sha256,
    ]


def _rollback_remote_argv(
    plan: PublishPlan,
    args: argparse.Namespace,
    *,
    candidate_installed_manifest_sha256: str,
) -> list[str]:
    return [
        "python",
        "/app/scripts/install_staged_production_bundle.py",
        "rollback",
        "--target-root",
        args.target_root,
        "--expected-active-bundle-id",
        plan.candidate_bundle_id,
        "--expected-rollback-bundle-id",
        plan.parent_bundle_id,
        "--expected-active-release-id",
        plan.candidate_release_id,
        "--expected-rollback-release-id",
        plan.parent_release_id,
        "--expected-active-manifest-sha256",
        candidate_installed_manifest_sha256,
        "--expected-rollback-manifest-sha256",
        plan.parent_installed_manifest_sha256,
    ]


def activate(
    plan: PublishPlan,
    args: argparse.Namespace,
    *,
    runner: RailwayRunner,
) -> dict[str, Any]:
    identity_file = getattr(args, "ssh_identity_file", None)
    if identity_file is None:
        raise PublishError("--ssh-identity-file is required with --activate")
    identity_path = Path(identity_file).expanduser()
    if identity_path.is_symlink() or not identity_path.is_file():
        raise PublishError("--ssh-identity-file must be a real file")
    if identity_path.stat().st_size == 0:
        raise PublishError("--ssh-identity-file must not be empty")
    args.ssh_identity_file = identity_path.resolve(strict=True)

    deployment_id, instance_id, commit_sha = _status_instance(runner, args)
    expected_git_sha = args.expected_deployed_git_sha.lower()
    if commit_sha != expected_git_sha:
        raise PublishError(
            f"Railway deployed Git SHA mismatch: {commit_sha} != {expected_git_sha}"
        )
    current = _installer_resolve(runner, args, instance_id)
    _assert_parent_resolve(current, plan)
    live_parent = _fetch_readyz(args.readyz_url)
    _validate_readyz(
        live_parent,
        plan=plan,
        installed_manifest_sha256=plan.parent_installed_manifest_sha256,
        expected_deployed_git_sha=expected_git_sha,
        candidate=False,
    )

    volume_output = runner.run(
        ["railway", "volume", *_railway_target_flags(args), "list", "--json"]
    )
    volume_payload = _json_from_output(volume_output, label="railway volume list")
    volumes = volume_payload.get("volumes") if isinstance(volume_payload, dict) else None
    matching_volumes = [
        volume
        for volume in volumes or []
        if volume.get("mountPath") == args.volume_mount_path
        and volume.get("status") == "Ready"
    ]
    if len(matching_volumes) != 1:
        raise PublishError("Expected exactly one ready Railway volume at the mount path")
    volume_id = str(matching_volumes[0].get("id") or "").strip()
    if not volume_id:
        raise PublishError("The unique ready Railway volume has no immutable id")

    target_root = PurePosixPath(args.target_root)
    if not target_root.is_absolute():
        raise PublishError("--target-root must be an absolute container path")
    upload_token = (
        f"{plan.candidate_bundle_id}-{plan.candidate_source_manifest_sha256[:20]}-"
        f"{uuid.uuid4().hex[:16]}"
    )
    remote_source_root = (target_root.parent / "incoming" / upload_token).as_posix()
    remote_volume_path = _volume_remote_path(args, remote_source_root)
    runner.run(
        _volume_upload_argv(
            args,
            volume_id=volume_id,
            local_path=plan.candidate_root,
            remote_path=remote_volume_path,
        )
    )

    promotion: Any = None
    promoted: dict[str, Any] | None = None
    try:
        promote_output = runner.run(
            _ssh_command(
                args,
                instance_id,
                _promote_remote_argv(plan, args, remote_source_root),
            )
        )
        promotion = _json_from_output(promote_output, label="installer promote")
        promoted = _installer_resolve(runner, args, instance_id)
        expected_promoted = {
            "active_bundle_id": plan.candidate_bundle_id,
            "active_release_id": plan.candidate_release_id,
            "rollback_bundle_id": plan.parent_bundle_id,
            "rollback_release_id": plan.parent_release_id,
            "rollback_manifest_sha256": plan.parent_installed_manifest_sha256,
            "rollback_ready": True,
        }
        mismatches = [
            key
            for key, value in expected_promoted.items()
            if promoted.get(key) != value
        ]
        if mismatches:
            raise PublishError(f"Post-promotion resolve mismatch: {mismatches}")
        installed_manifest_sha = _require_sha256(
            promoted.get("active_manifest_sha256"),
            label="installed candidate manifest",
        )
        _restart(runner, args)
        new_deployment_id, _new_instance_id = _wait_for_single_instance(
            runner,
            args,
            expected_git_sha=expected_git_sha,
            timeout_seconds=args.readiness_timeout_seconds,
            interval_seconds=args.poll_interval_seconds,
        )
        samples = _wait_for_readyz(
            url=args.readyz_url,
            plan=plan,
            installed_manifest_sha256=installed_manifest_sha,
            expected_deployed_git_sha=expected_git_sha,
            candidate=True,
            timeout_seconds=args.readiness_timeout_seconds,
            interval_seconds=args.poll_interval_seconds,
        )
        return {
            "status": "activated",
            "previous_deployment_id": deployment_id,
            "deployment_id": new_deployment_id,
            "volume_id": volume_id,
            "remote_source_root": remote_source_root,
            "promotion": promotion,
            "resolve": promoted,
            "readiness_samples": samples,
        }
    except Exception as activation_error:
        recovery: dict[str, Any] = {
            "activation_error": str(activation_error),
            "status": "not_attempted",
        }
        recovery_error: str | None = None
        try:
            _recovery_deployment, rollback_instance, recovery_git_sha = _status_instance(
                runner, args
            )
            if recovery_git_sha != expected_git_sha:
                raise PublishError(
                    "Railway recovery instance Git SHA mismatch: "
                    f"{recovery_git_sha} != {expected_git_sha}"
                )
            recovery_state = _installer_resolve(runner, args, rollback_instance)
            if (
                recovery_state.get("active_bundle_id") == plan.candidate_bundle_id
                and recovery_state.get("active_release_id") == plan.candidate_release_id
            ):
                candidate_manifest_sha = _require_sha256(
                    recovery_state.get("active_manifest_sha256"),
                    label="installed candidate manifest during recovery",
                )
                rollback_output = runner.run(
                    _ssh_command(
                        args,
                        rollback_instance,
                        _rollback_remote_argv(
                            plan,
                            args,
                            candidate_installed_manifest_sha256=candidate_manifest_sha,
                        ),
                    )
                )
                recovery["installer"] = _json_from_output(
                    rollback_output, label="installer rollback"
                )
                recovery["status"] = "parent_pointer_restored"
                try:
                    _restart(runner, args)
                    recovery["restart"] = "requested"
                except Exception as restart_error:
                    recovery["restart"] = "failed"
                    recovery["restart_error"] = str(restart_error)
            else:
                _assert_parent_resolve(recovery_state, plan)
                recovery["status"] = "parent_already_active"
        except Exception as exc:  # retain both primary and rollback failures
            recovery_error = str(exc)
            recovery["status"] = "recovery_failed"
            recovery["error"] = recovery_error

        detail = f"Activation failed: {activation_error}"
        if recovery_error:
            detail += f"; installer rollback also failed: {recovery_error}"
        else:
            detail += f"; installer recovery status={recovery['status']}"
        raise PublishError(
            detail,
            details={
                "volume_id": volume_id,
                "remote_source_root": remote_source_root,
                "promotion": promotion,
                "resolve": promoted,
                "recovery": recovery,
            },
        ) from activation_error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--parent-readyz-json", type=Path, required=True)
    parser.add_argument("--readyz-url", required=True)
    parser.add_argument("--expected-deployed-git-sha", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--volume-mount-path", required=True)
    parser.add_argument("--ssh-identity-file", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--readiness-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=10.0)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: RailwayRunner | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    report: dict[str, Any] = {
        "schema_version": 1,
        "mode": "activate" if args.activate else "validation_only",
        "status": "error",
    }
    try:
        expected_git_sha = str(args.expected_deployed_git_sha).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{40}", expected_git_sha):
            raise PublishError("--expected-deployed-git-sha must be a full Git SHA")
        if args.readiness_timeout_seconds <= 0 or args.poll_interval_seconds <= 0:
            raise PublishError("Readiness timeout and poll interval must be positive")
        plan = build_publish_plan(args.candidate_root, args.parent_readyz_json)
        report["plan"] = asdict(plan)
        if not args.activate:
            report["status"] = "validated_no_mutation"
            _write_report(args.report, report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        result = activate(plan, args, runner=runner or RailwayRunner())
        report.update(result)
        _write_report(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (PublishError, OSError, ValueError) as exc:
        report["error"] = str(exc)
        if isinstance(exc, PublishError) and exc.details is not None:
            report["failure_details"] = exc.details
        _write_report(args.report, report)
        print(f"SCHEDULED REFIT PUBLISH FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
