"""Bootstrap the hosted runtime manifest and canonical processed snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.io_utils import copy_file_atomically
from src.model.production_bundle import (
    ProductionBundleError,
    get_model_artifact_fingerprints,
    get_processed_snapshot_fingerprints,
    get_processed_snapshot_max_event_date,
    reconcile_production_bundle_manifest,
    runtime_manifest_needs_source_bootstrap,
)


_PROCESSED_HASH_FIELDS = (
    "processed_fights_sha256",
    "processed_features_sha256",
)
_PROCESSED_SIZE_FIELDS = (
    "processed_fights_bytes",
    "processed_features_bytes",
)
_MODEL_HASH_FIELDS = (
    "model_sha256",
    "no_odds_model_sha256",
    "logistic_model_sha256",
)


def _processed_snapshot_exists(processed_dir: Path) -> bool:
    return all((processed_dir / filename).is_file() for filename in ("fights_cleaned.csv", "features.csv"))


def _copy_processed_snapshot(*, source_processed_dir: Path, target_processed_dir: Path) -> None:
    for filename in ("fights_cleaned.csv", "features.csv"):
        copy_file_atomically(
            source_processed_dir / filename,
            target_processed_dir / filename,
        )

    source_fingerprints = get_processed_snapshot_fingerprints(source_processed_dir)
    target_fingerprints = get_processed_snapshot_fingerprints(target_processed_dir)
    if target_fingerprints != source_fingerprints:
        raise ProductionBundleError(
            "Processed snapshot copy failed verification: runtime files do not match "
            "the approved image snapshot."
        )


def _load_manifest_payload(path: Path, *, label: str) -> dict[str, object]:
    if not path.is_file():
        raise ProductionBundleError(f"{label} manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProductionBundleError(f"{label} manifest is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProductionBundleError(f"{label} manifest must contain a JSON object: {path}")
    return payload


def _manifest_text(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    text = str(value or "").strip()
    return text or None


def _parse_manifest_timestamp(payload: dict[str, object], *, label: str) -> datetime:
    raw = _manifest_text(payload, "built_at")
    if raw is None:
        raise ProductionBundleError(f"{label} manifest is missing built_at.")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProductionBundleError(
            f"{label} manifest has an invalid built_at timestamp: {raw!r}."
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validate_declared_fingerprints(
    payload: dict[str, object],
    actual: dict[str, int | str],
    *,
    fields: tuple[str, ...],
    label: str,
) -> None:
    for field in fields:
        if field not in payload or payload[field] in (None, ""):
            continue
        expected = payload[field]
        if isinstance(actual.get(field), int):
            try:
                expected = int(expected)
            except (TypeError, ValueError) as exc:
                raise ProductionBundleError(
                    f"{label} manifest field {field} is not an integer."
                ) from exc
        else:
            expected = str(expected).strip().lower()
        if expected != actual.get(field):
            raise ProductionBundleError(
                f"{label} manifest field {field} does not match its artifact: "
                f"manifest={expected!r}, actual={actual.get(field)!r}."
            )


def _validate_source_manifest(
    *,
    source_manifest: Path,
    source_processed_dir: Path,
    model_path: Path,
    no_odds_model_path: Path,
    logistic_model_path: Path | None,
) -> tuple[dict[str, object], dict[str, int | str]]:
    """Validate every fingerprint the approved image manifest declares."""
    payload = _load_manifest_payload(source_manifest, label="Source production bundle")
    _parse_manifest_timestamp(payload, label="Source production bundle")
    processed_fingerprints = get_processed_snapshot_fingerprints(source_processed_dir)
    _validate_declared_fingerprints(
        payload,
        processed_fingerprints,
        fields=_PROCESSED_HASH_FIELDS + _PROCESSED_SIZE_FIELDS,
        label="Source production bundle",
    )

    source_date = get_processed_snapshot_max_event_date(source_processed_dir)
    manifest_date = _manifest_text(payload, "snapshot_max_event_date")
    if source_date is None or manifest_date != source_date:
        raise ProductionBundleError(
            "Source production bundle snapshot date does not match its processed files: "
            f"manifest={manifest_date!r}, actual={source_date!r}."
        )

    model_fingerprints = get_model_artifact_fingerprints(
        model_path,
        no_odds_model_path,
        logistic_model_path if logistic_model_path and logistic_model_path.exists() else None,
    )
    _validate_declared_fingerprints(
        payload,
        model_fingerprints,
        fields=_MODEL_HASH_FIELDS,
        label="Source production bundle",
    )
    return payload, processed_fingerprints


def _require_source_processed_hashes(payload: dict[str, object]) -> None:
    missing = [field for field in _PROCESSED_HASH_FIELDS if not _manifest_text(payload, field)]
    if missing:
        raise ProductionBundleError(
            "Cannot promote an unpinned source processed snapshot; source manifest is "
            f"missing: {', '.join(missing)}."
        )


def _model_identity(
    payload: dict[str, object],
    *,
    logistic_required: bool,
    label: str,
) -> tuple[str, ...]:
    required_fields = list(_MODEL_HASH_FIELDS[:2])
    if logistic_required:
        required_fields.append("logistic_model_sha256")
    values: list[str] = []
    for field in required_fields:
        value = _manifest_text(payload, field)
        if value is None:
            raise ProductionBundleError(
                "Cannot resolve an equal-date processed snapshot conflict because "
                f"the {label} manifest is missing {field}."
            )
        values.append(value.lower())
    return tuple(values)


def _source_wins_equal_date_conflict(
    *,
    source_payload: dict[str, object],
    target_payload: dict[str, object],
    logistic_required: bool,
) -> bool:
    """Resolve equal-date/different-content snapshots without guessing.

    A newer approved build or different fully-pinned model generation promotes
    the image snapshot.  The same (or an older) generation preserves legitimate
    runtime enrichment.  Missing identity metadata is intentionally fatal.
    """
    source_built_at = _parse_manifest_timestamp(source_payload, label="Source")
    target_built_at = _parse_manifest_timestamp(target_payload, label="Runtime")
    source_core_identity = _model_identity(
        source_payload,
        logistic_required=False,
        label="source",
    )
    target_core_identity = _model_identity(
        target_payload,
        logistic_required=False,
        label="runtime",
    )
    if source_built_at > target_built_at or source_core_identity != target_core_identity:
        # Either proof is sufficient to roll forward a legacy runtime manifest
        # that may not yet pin the logistic artifact.
        return True
    if not logistic_required:
        return False

    source_logistic_identity = _model_identity(
        source_payload,
        logistic_required=True,
        label="source",
    )[-1]
    target_logistic_identity = _model_identity(
        target_payload,
        logistic_required=True,
        label="runtime",
    )[-1]
    return source_logistic_identity != target_logistic_identity


def _source_generation_differs(
    *,
    source_payload: dict[str, object],
    target_payload: dict[str, object],
) -> bool:
    """Return whether an explicitly selected release differs from runtime.

    The active release pointer is authoritative during a promotion or rollback.
    Comparing the bundle id plus the two always-required model hashes lets that
    one deliberate generation switch restore its matching processed snapshot,
    even when the mutable runtime lookup data has a later event date.  A legacy
    runtime manifest may omit the logistic hash, so absence alone is not a
    generation change; two present, unequal logistic hashes are.
    """
    if not target_payload:
        return True

    source_bundle_id = _manifest_text(source_payload, "bundle_id")
    target_bundle_id = _manifest_text(target_payload, "bundle_id")
    if source_bundle_id is None or target_bundle_id is None:
        return True
    if source_bundle_id != target_bundle_id:
        return True

    try:
        source_core = _model_identity(
            source_payload,
            logistic_required=False,
            label="source",
        )
        target_core = _model_identity(
            target_payload,
            logistic_required=False,
            label="runtime",
        )
    except ProductionBundleError:
        return True
    if source_core != target_core:
        return True

    source_logistic = _manifest_text(source_payload, "logistic_model_sha256")
    target_logistic = _manifest_text(target_payload, "logistic_model_sha256")
    return bool(
        source_logistic
        and target_logistic
        and source_logistic.lower() != target_logistic.lower()
    )


def bootstrap_runtime_production_bundle(
    *,
    target_manifest: Path,
    source_manifest: Path,
    source_processed_dir: Path,
    target_processed_dir: Path,
    model_path: Path,
    no_odds_model_path: Path,
    logistic_model_path: Path | None = None,
    activate_source_generation: bool = False,
) -> dict[str, object]:
    source_snapshot_exists = _processed_snapshot_exists(source_processed_dir)
    target_snapshot_exists = _processed_snapshot_exists(target_processed_dir)
    source_payload: dict[str, object] = {}
    source_fingerprints: dict[str, int | str] = {}
    if source_snapshot_exists:
        source_payload, source_fingerprints = _validate_source_manifest(
            source_manifest=source_manifest,
            source_processed_dir=source_processed_dir,
            model_path=model_path,
            no_odds_model_path=no_odds_model_path,
            logistic_model_path=logistic_model_path,
        )

    needs_source_bootstrap = runtime_manifest_needs_source_bootstrap(
        target_manifest_path=target_manifest,
        source_manifest_path=source_manifest,
    )
    target_payload: dict[str, object] = {}
    if target_manifest.is_file():
        try:
            target_payload = _load_manifest_payload(
                target_manifest,
                label="Runtime production bundle",
            )
        except ProductionBundleError:
            # A corrupt runtime manifest is recoverable when dates prove that
            # the approved image is newer. Equal-date conflicts still fail
            # closed below because generation metadata is then unavailable.
            target_payload = {}

    bootstrap_action = "reused_existing_runtime_bundle"
    promote_source = False

    generation_activation = bool(
        activate_source_generation
        and source_snapshot_exists
        and _source_generation_differs(
            source_payload=source_payload,
            target_payload=target_payload,
        )
    )

    if generation_activation:
        _require_source_processed_hashes(source_payload)
        promote_source = True
    elif source_snapshot_exists and not target_snapshot_exists:
        _require_source_processed_hashes(source_payload)
        promote_source = True
    elif source_snapshot_exists and target_snapshot_exists:
        source_date = get_processed_snapshot_max_event_date(source_processed_dir)
        target_date = get_processed_snapshot_max_event_date(target_processed_dir)
        if source_date is None or target_date is None:
            raise ProductionBundleError(
                "Cannot compare production snapshots because one has no usable event_date."
            )

        if source_date > target_date:
            _require_source_processed_hashes(source_payload)
            promote_source = True
        elif source_date == target_date:
            target_fingerprints = get_processed_snapshot_fingerprints(target_processed_dir)
            hashes_match = all(
                source_fingerprints[field] == target_fingerprints[field]
                for field in _PROCESSED_HASH_FIELDS
            )
            if not hashes_match:
                _require_source_processed_hashes(source_payload)
                promote_source = _source_wins_equal_date_conflict(
                    source_payload=source_payload,
                    target_payload=target_payload,
                    logistic_required=bool(
                        logistic_model_path is not None and logistic_model_path.exists()
                    ),
                )

        if not promote_source and needs_source_bootstrap:
            bootstrap_action = "adopted_existing_runtime_snapshot"
    elif target_snapshot_exists:
        if needs_source_bootstrap:
            bootstrap_action = "adopted_existing_runtime_snapshot"
    else:
        missing_files = [
            str(source_processed_dir / filename)
            for filename in ("fights_cleaned.csv", "features.csv")
        ]
        raise FileNotFoundError(
            "Cannot bootstrap production bundle: canonical processed snapshot is missing "
            f"from image and runtime volume. Missing files: {', '.join(missing_files)}"
        )

    if promote_source:
        _copy_processed_snapshot(
            source_processed_dir=source_processed_dir,
            target_processed_dir=target_processed_dir,
        )
        bootstrap_action = "promoted_source_bundle"

    summary = reconcile_production_bundle_manifest(
        target_manifest_path=target_manifest,
        source_manifest_path=source_manifest,
        model_path=model_path,
        no_odds_model_path=no_odds_model_path,
        logistic_model_path=logistic_model_path,
        processed_dir=target_processed_dir,
        authoritative_source_manifest=generation_activation,
    )
    summary["bootstrap_action"] = bootstrap_action
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-processed-dir", type=Path, required=True)
    parser.add_argument("--target-processed-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--no-odds-model-path", type=Path, required=True)
    parser.add_argument("--logistic-model-path", type=Path, default=None)
    parser.add_argument(
        "--activate-source-generation",
        action="store_true",
        help=(
            "Treat the explicitly selected source release as authoritative for one "
            "model-generation switch, including its matching processed snapshot."
        ),
    )
    args = parser.parse_args()

    summary = bootstrap_runtime_production_bundle(
        target_manifest=args.target_manifest,
        source_manifest=args.source_manifest,
        source_processed_dir=args.source_processed_dir,
        target_processed_dir=args.target_processed_dir,
        model_path=args.model_path,
        no_odds_model_path=args.no_odds_model_path,
        logistic_model_path=args.logistic_model_path,
        activate_source_generation=args.activate_source_generation,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
