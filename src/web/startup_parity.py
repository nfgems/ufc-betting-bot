"""Validation helpers for the hosted startup-parity readiness receipt."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.live_control import LIVE_MODE_ENV, LIVE_MODE_REAL, normalize_live_mode
from src.model.production_bundle import is_hosted_runtime
from src.model.training_spec import resolve_named_training_spec


REQUIRED_BINDING_FIELDS = frozenset(
    {
        "deployed_git_sha",
        "bundle_id",
        "model_spec_name",
        "registered_spec_sha256",
        "confirmed_strategy_sha256",
        "source_manifest_sha256",
        "runtime_manifest_sha256",
        "installed_manifest_sha256",
        "immutable_fights_sha256",
        "immutable_features_sha256",
        "replay_args",
    }
)
_RECEIPT_FIELDS = frozenset(
    {"schema_version", "receipt_key", "passed", "checked_at", "binding", "reports"}
)
_REPORT_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "n_fights",
        "requested_feature_count",
        "forbidden_mismatch_count",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_REPLAY_ARGS = {
    "start": "2025-01-01",
    "limit": 250,
    "seed": 7,
    "tol": 1e-6,
    "exact": {"mode": "exact", "established_only": False},
    "prefight": {"mode": "prefight", "established_only": True},
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hosted_real_money_parity_required(runtime_status: dict | None = None) -> bool:
    """Return whether this process is a hosted real-money web runtime."""
    if not is_hosted_runtime():
        return False
    if normalize_live_mode(os.getenv(LIVE_MODE_ENV)) == LIVE_MODE_REAL:
        return True
    status = runtime_status if isinstance(runtime_status, dict) else {}
    return bool(status.get("trading_live")) or any(
        normalize_live_mode(status.get(field)) == LIVE_MODE_REAL
        for field in (
            "requested_live_mode",
            "requested_live_mode_raw",
            "effective_live_mode",
        )
    )


def _required_string(payload: dict, field: str, *, label: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{label} is missing required field {field}")
    return value.strip()


def _required_sha256(payload: dict, field: str, *, label: str) -> str:
    value = _required_string(payload, field, label=label)
    if _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{label}.{field} is not a lowercase SHA-256")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{label} must be a positive integer")
    return value


def _validate_replay_args(value: object) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("Startup parity receipt binding.replay_args must be an object")
    required = {"start", "limit", "seed", "tol", "exact", "prefight"}
    if set(value) != required:
        raise RuntimeError(
            "Startup parity receipt binding.replay_args fields are not exact"
        )
    _required_string(value, "start", label="Startup parity replay arguments")
    _positive_int(value.get("limit"), label="Startup parity replay limit")
    if isinstance(value.get("seed"), bool) or not isinstance(value.get("seed"), int):
        raise RuntimeError("Startup parity replay seed must be an integer")
    tolerance = value.get("tol")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        raise RuntimeError("Startup parity replay tolerance must be finite and positive")
    if value.get("exact") != {"mode": "exact", "established_only": False}:
        raise RuntimeError("Startup parity exact replay binding is invalid")
    if value.get("prefight") != {"mode": "prefight", "established_only": True}:
        raise RuntimeError("Startup parity prefight replay binding is invalid")
    if value != EXPECTED_REPLAY_ARGS:
        raise RuntimeError(
            "Startup parity replay arguments do not match the hosted readiness contract"
        )


def validate_exact_receipt_payload(payload: object) -> dict:
    """Validate the complete receipt shape and its self-binding receipt key."""
    if not isinstance(payload, dict) or set(payload) != _RECEIPT_FIELDS:
        raise RuntimeError("Startup parity receipt fields are not exact")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise RuntimeError("Startup parity receipt schema_version must be 1")
    if payload.get("passed") is not True:
        raise RuntimeError("Startup parity receipt is not passing")
    checked_at = _required_string(payload, "checked_at", label="Startup parity receipt")
    try:
        datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("Startup parity receipt checked_at is invalid") from exc

    binding = payload.get("binding")
    if not isinstance(binding, dict) or set(binding) != REQUIRED_BINDING_FIELDS:
        raise RuntimeError("Startup parity receipt binding fields are not exact")
    for field in ("deployed_git_sha", "bundle_id", "model_spec_name"):
        _required_string(binding, field, label="Startup parity receipt binding")
    for field in (
        "registered_spec_sha256",
        "confirmed_strategy_sha256",
        "source_manifest_sha256",
        "runtime_manifest_sha256",
        "installed_manifest_sha256",
        "immutable_fights_sha256",
        "immutable_features_sha256",
    ):
        _required_sha256(binding, field, label="Startup parity receipt binding")
    _validate_replay_args(binding.get("replay_args"))

    receipt_key = _required_sha256(payload, "receipt_key", label="Startup parity receipt")
    if receipt_key != _canonical_sha256(binding):
        raise RuntimeError("Startup parity receipt_key does not bind the exact receipt")

    reports = payload.get("reports")
    if not isinstance(reports, dict) or set(reports) != {"exact", "prefight"}:
        raise RuntimeError("Startup parity receipt reports are not exact")
    for mode in ("exact", "prefight"):
        report = reports.get(mode)
        if not isinstance(report, dict) or set(report) != _REPORT_FIELDS:
            raise RuntimeError(f"Startup parity {mode} report fields are not exact")
        _required_string(report, "path", label=f"Startup parity {mode} report")
        _required_sha256(report, "sha256", label=f"Startup parity {mode} report")
        _positive_int(
            report.get("n_fights"),
            label=f"Startup parity {mode} report n_fights",
        )
        _positive_int(
            report.get("requested_feature_count"),
            label=f"Startup parity {mode} report requested_feature_count",
        )
        if report.get("forbidden_mismatch_count") != 0:
            raise RuntimeError(
                f"Startup parity {mode} report has forbidden feature mismatches"
            )
    return payload


def exact_receipt_payload_is_valid(payload: object) -> bool:
    try:
        validate_exact_receipt_payload(payload)
    except RuntimeError:
        return False
    return True


def startup_parity_validation_record(payload: object) -> dict:
    """Create the in-memory marker proving strict validation ran at startup."""
    receipt = validate_exact_receipt_payload(payload)
    binding = receipt["binding"]
    return {
        "validated_at_startup": True,
        "receipt_key": receipt["receipt_key"],
        "bundle_id": binding["bundle_id"],
        "source_manifest_sha256": binding["source_manifest_sha256"],
        "runtime_manifest_sha256": binding["runtime_manifest_sha256"],
        "installed_manifest_sha256": binding["installed_manifest_sha256"],
    }


def startup_parity_validation_record_is_valid(
    payload: object,
    record: object,
) -> bool:
    if not isinstance(record, dict):
        return False
    try:
        expected = startup_parity_validation_record(payload)
    except RuntimeError:
        return False
    return record == expected


def _strict_expected_binding(bundle_summary: dict) -> dict[str, Any]:
    if not isinstance(bundle_summary, dict):
        raise RuntimeError("Hosted real-money startup has no production bundle summary")

    deployed_sha = str(os.getenv("RAILWAY_GIT_COMMIT_SHA", "") or "").strip()
    if not deployed_sha:
        raise RuntimeError("Hosted real-money startup has no Railway deployed Git SHA")

    installed_manifest_path = Path(
        _required_string(
            bundle_summary,
            "installed_manifest_path",
            label="Production bundle summary",
        )
    )
    runtime_manifest_path = Path(
        _required_string(
            bundle_summary,
            "manifest_path",
            label="Production bundle summary",
        )
    )
    try:
        source_manifest_sha256 = _file_sha256(installed_manifest_path)
        runtime_manifest_sha256 = _file_sha256(runtime_manifest_path)
    except OSError as exc:
        raise RuntimeError(f"Startup parity manifest is unreadable: {exc}") from exc

    installed_manifest_sha256 = _required_sha256(
        bundle_summary,
        "installed_manifest_sha256",
        label="Production bundle summary",
    )
    if installed_manifest_sha256 != source_manifest_sha256:
        raise RuntimeError(
            "Production bundle installed manifest hash does not match its active source manifest"
        )

    model_spec_name = _required_string(
        bundle_summary,
        "model_spec_name",
        label="Production bundle summary",
    )
    scheduled_policy = bundle_summary.get("scheduled_refit_policy")
    if (
        not isinstance(scheduled_policy, dict)
        or scheduled_policy.get("policy_schema_version") != 2
    ):
        raise RuntimeError(
            "Hosted real-money startup requires a final-v2 scheduled refit policy"
        )
    confirmed_strategy = bundle_summary.get("confirmed_strategy")
    if not isinstance(confirmed_strategy, dict):
        raise RuntimeError(
            "Hosted real-money startup requires a validated confirmed strategy"
        )
    registered_spec_sha256 = _canonical_sha256(
        asdict(resolve_named_training_spec(model_spec_name))
    )
    return {
        "deployed_git_sha": deployed_sha,
        "bundle_id": _required_string(
            bundle_summary,
            "bundle_id",
            label="Production bundle summary",
        ),
        "model_spec_name": model_spec_name,
        "registered_spec_sha256": registered_spec_sha256,
        "confirmed_strategy_sha256": _canonical_sha256(confirmed_strategy),
        "source_manifest_sha256": source_manifest_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "installed_manifest_sha256": installed_manifest_sha256,
        "immutable_fights_sha256": _required_sha256(
            bundle_summary,
            "immutable_training_fights_sha256",
            label="Production bundle summary",
        ),
        "immutable_features_sha256": _required_sha256(
            bundle_summary,
            "immutable_training_features_sha256",
            label="Production bundle summary",
        ),
    }


def _validate_exact_binding_against_bundle(binding: dict, bundle_summary: dict) -> None:
    expected = _strict_expected_binding(bundle_summary)
    mismatches = [field for field, value in expected.items() if binding.get(field) != value]
    if mismatches:
        raise RuntimeError(
            "Startup parity receipt does not match the active release: "
            + ", ".join(mismatches)
        )


def load_startup_parity_receipt(
    bundle_summary: dict | None,
    *,
    require_exact_binding: bool,
) -> dict | None:
    """Load a receipt, enforcing the strict hosted-real contract when required."""
    raw_path = str(os.getenv("UFC_STARTUP_PARITY_RECEIPT", "") or "").strip()
    if not raw_path:
        if require_exact_binding:
            raise RuntimeError(
                "Hosted real-money readiness requires UFC_STARTUP_PARITY_RECEIPT"
            )
        return None

    path = Path(raw_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Startup parity receipt is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("passed") is not True:
        raise RuntimeError(f"Startup parity receipt is not passing: {path}")
    binding = payload.get("binding")
    if not isinstance(binding, dict):
        raise RuntimeError(f"Startup parity receipt has no binding: {path}")

    if require_exact_binding:
        validate_exact_receipt_payload(payload)
        _validate_exact_binding_against_bundle(binding, bundle_summary or {})
        return payload

    # Preserve the pre-existing tolerant behavior outside hosted real-money mode.
    deployed_sha = str(os.getenv("RAILWAY_GIT_COMMIT_SHA", "") or "").strip()
    if deployed_sha and binding.get("deployed_git_sha") != deployed_sha:
        raise RuntimeError("Startup parity receipt was produced by a different deploy")
    if bundle_summary is not None:
        for receipt_field, bundle_field in (
            ("bundle_id", "bundle_id"),
            ("model_spec_name", "model_spec_name"),
            ("immutable_fights_sha256", "immutable_training_fights_sha256"),
            ("immutable_features_sha256", "immutable_training_features_sha256"),
        ):
            expected = bundle_summary.get(bundle_field)
            actual = binding.get(receipt_field)
            if expected and actual and expected != actual:
                raise RuntimeError(
                    f"Startup parity receipt {receipt_field} does not match active bundle"
                )
    return payload
