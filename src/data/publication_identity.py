"""Pure publication identity helpers shared by publisher and saved monthly intake.

Definitions moved without changing their semantics. No source or model work.
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime
from pathlib import Path

def _file_content_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def _accepted_data_generation_identity(processed_dir: Path) -> dict[str, object]:
    """Content-address one accepted processed data generation.

    A production bundle id identifies the model contract and may intentionally
    survive a data-only refresh. Monthly candidates need the identity of the
    exact accepted data instead, so derive it only from the two processed
    artifacts that the publication pointer commits.
    """
    artifacts = {
        "fights_cleaned.csv": _file_content_identity(
            processed_dir / "fights_cleaned.csv"
        ),
        "features.csv": _file_content_identity(processed_dir / "features.csv"),
    }
    return accepted_data_generation_from_artifacts(artifacts)


def accepted_data_generation_from_artifacts(artifacts: dict) -> dict[str, object]:
    """Apply the publisher's identity to captured or independently pinned bytes."""
    identity_payload = {
        "schema_version": "ufc_accepted_data_generation_v1",
        "artifacts": artifacts,
    }
    digest = hashlib.sha256(
        json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        **identity_payload,
        "accepted_generation_id": f"ufc-data-v1-{digest}",
    }


def _accepted_raw_generation_identity(
    refresh_input_inventory: dict[str, object],
) -> dict[str, str]:
    """Content-address the complete mutable/read-only raw rebuild contract."""
    if (
        not isinstance(refresh_input_inventory, dict)
        or refresh_input_inventory.get("schema_version")
        != "ufc_refresh_input_inventory_v1"
        or not isinstance(refresh_input_inventory.get("artifacts"), dict)
        or not refresh_input_inventory["artifacts"]
    ):
        raise ValueError("refresh_input_inventory is not a complete v1 inventory")
    digest = hashlib.sha256(
        json.dumps(
            refresh_input_inventory,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {"accepted_raw_generation_id": f"ufc-raw-v1-{digest}"}


def _validated_calendar_claim(
    value: dict[str, object] | None,
) -> dict[str, object] | None:
    """Validate the small durable weekly-claim identity before any staging."""
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("calendar_claim must be a schema-version 1 object")
    normalized: dict[str, object] = {"schema_version": 1}
    for field in ("claim_id", "iso_week", "slot", "claimed_at_utc", "due_at_utc"):
        text = str(value.get(field) or "").strip()
        if not text:
            raise ValueError(f"calendar_claim.{field} is required")
        normalized[field] = text
    if normalized["slot"] not in {"tuesday", "wednesday"}:
        raise ValueError("calendar_claim.slot must be tuesday or wednesday")
    for field in ("claimed_at_utc", "due_at_utc"):
        try:
            parsed = datetime.fromisoformat(
                str(normalized[field]).replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(f"calendar_claim.{field} is invalid") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"calendar_claim.{field} must be timezone-aware")
    return normalized


APPROVED_V7_BASELINE_LINEAGE = {
    "schema_version": 1,
    "dataset_id": "modeling_dataset_2022_v7",
    "dataset_sha256": "03ef274c70c23facdde335582095b40dea8558451555db1d4df639816c50c604",
    "dataset_evidence_sha256": "19be8201f068a0af4f349517de98856ee7e8cdb3ca06f326a711bb7eb29e54bb",
    "through_event_date": "2026-08-22",
    "seed_roster_sha256": "d83792d089241b3ba31cee4a37360f9793d54207dd2d416d2496f0ec0a003e2f",
}
