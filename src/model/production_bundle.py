from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from src.data.io_utils import write_json_atomically

logger = logging.getLogger(__name__)

HOSTED_PROJECT_ROOT = Path("/app")
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "models" / "current_production_model.json"
PRODUCTION_BUNDLE_ENV = "UFC_PRODUCTION_BUNDLE_MANIFEST"


class ProductionBundleError(RuntimeError):
    """Raised when the active production bundle is missing or inconsistent."""


@dataclass(frozen=True)
class ProductionBundle:
    manifest_path: Path
    bundle_id: str
    model_spec_name: str
    model_path: Path
    no_odds_model_path: Path
    processed_dir: Path
    snapshot_max_event_date: str
    built_at: str
    git_sha: str
    no_odds_model_spec_name: str | None = None
    logistic_model_path: Path | None = None
    processed_fights_sha256: str | None = None
    processed_features_sha256: str | None = None
    processed_fights_bytes: int | None = None
    processed_features_bytes: int | None = None
    manifest_updated_at: str | None = None
    manifest_version: int = 1


def is_hosted_runtime(*, project_root: Path | None = None) -> bool:
    root = project_root or PROJECT_ROOT
    if root == HOSTED_PROJECT_ROOT:
        return True
    for env_name in (
        "RAILWAY_VOLUME_MOUNT_PATH",
        "RAILWAY_ENVIRONMENT_NAME",
        "RAILWAY_SERVICE_NAME",
    ):
        if str(os.getenv(env_name, "") or "").strip():
            return True
    return False


def _resolved_path(path: Path) -> Path:
    return path.resolve(strict=False)


def _path_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


def _required_string(payload: dict[str, Any], key: str) -> str:
    raw = payload.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ProductionBundleError(f"Production bundle manifest is missing required field '{key}'.")
    return raw.strip()


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    raw = payload.get(key)
    text = str(raw or "").strip()
    return text or None


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    raw = payload.get(key)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _token_values() -> dict[str, str]:
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DATA_DIR": str(DATA_DIR),
        "UFC_DATA_DIR": str(DATA_DIR),
        "MODELS_DIR": str(MODELS_DIR),
        "UFC_MODELS_DIR": str(MODELS_DIR),
        "PROCESSED_DATA_DIR": str(PROCESSED_DATA_DIR),
    }


def _expand_tokens(raw: str) -> str:
    expanded = raw
    for key, value in _token_values().items():
        expanded = expanded.replace(f"${{{key}}}", value)
    return expanded


def _resolve_bundle_path(raw: str) -> Path:
    expanded = _expand_tokens(raw)
    candidate = Path(expanded)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _resolve_manifest_path(manifest_path: str | Path | None = None) -> Path:
    if manifest_path is not None:
        candidate = Path(manifest_path)
    else:
        raw_env = str(os.getenv(PRODUCTION_BUNDLE_ENV, "") or "").strip()
        candidate = Path(raw_env) if raw_env else DEFAULT_MANIFEST_PATH
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _load_manifest_payload(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProductionBundleError(f"Production bundle manifest is invalid JSON: {path}: {exc}") from exc


def _runtime_timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_bundle_id(*, model_spec_name: str, snapshot_max_event_date: str) -> str:
    date_token = snapshot_max_event_date.replace("-", "")
    return f"ufc-production-{date_token}-{model_spec_name}"


def _determine_git_sha(base_payload: dict[str, Any]) -> str:
    for env_name in ("RAILWAY_GIT_COMMIT_SHA", "GIT_SHA"):
        value = str(os.getenv(env_name, "") or "").strip()
        if value:
            return value
    existing = _optional_string(base_payload, "git_sha")
    if existing:
        return existing
    return "unknown"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_size_bytes(path: Path) -> int:
    return int(path.stat().st_size)


def load_production_bundle(manifest_path: str | Path | None = None) -> ProductionBundle:
    path = _resolve_manifest_path(manifest_path)
    if not path.exists():
        raise ProductionBundleError(f"Production bundle manifest not found: {path}")
    payload = _load_manifest_payload(path)

    return ProductionBundle(
        manifest_path=path,
        bundle_id=_required_string(payload, "bundle_id"),
        model_spec_name=_required_string(payload, "model_spec_name"),
        model_path=_resolve_bundle_path(_required_string(payload, "model_path")),
        no_odds_model_path=_resolve_bundle_path(_required_string(payload, "no_odds_model_path")),
        processed_dir=_resolve_bundle_path(_required_string(payload, "processed_dir")),
        snapshot_max_event_date=_required_string(payload, "snapshot_max_event_date"),
        built_at=_required_string(payload, "built_at"),
        git_sha=_required_string(payload, "git_sha"),
        no_odds_model_spec_name=str(payload.get("no_odds_model_spec_name") or "").strip() or None,
        logistic_model_path=(
            _resolve_bundle_path(str(payload["logistic_model_path"]).strip())
            if str(payload.get("logistic_model_path") or "").strip()
            else None
        ),
        processed_fights_sha256=_optional_string(payload, "processed_fights_sha256"),
        processed_features_sha256=_optional_string(payload, "processed_features_sha256"),
        processed_fights_bytes=_optional_int(payload, "processed_fights_bytes"),
        processed_features_bytes=_optional_int(payload, "processed_features_bytes"),
        manifest_updated_at=_optional_string(payload, "manifest_updated_at"),
        manifest_version=int(payload.get("manifest_version") or 1),
    )


def load_production_bundle_or_none(
    manifest_path: str | Path | None = None,
) -> ProductionBundle | None:
    try:
        return load_production_bundle(manifest_path)
    except ProductionBundleError:
        return None


@lru_cache(maxsize=32)
def _cached_snapshot_max_event_date(
    processed_dir_str: str,
    fights_mtime_ns: int,
    features_mtime_ns: int,
) -> str | None:
    import pandas as pd

    processed_dir = Path(processed_dir_str)
    max_dates: list[str] = []

    for filename in ("fights_cleaned.csv", "features.csv"):
        path = processed_dir / filename
        if not path.exists():
            continue
        try:
            event_dates = pd.read_csv(path, usecols=["event_date"])["event_date"]
        except ValueError:
            continue
        parsed = pd.to_datetime(event_dates, errors="coerce").dropna()
        if parsed.empty:
            continue
        max_dates.append(parsed.max().date().isoformat())

    if not max_dates:
        return None
    return max(max_dates)


def get_processed_snapshot_max_event_date(processed_dir: Path) -> str | None:
    resolved_dir = _resolved_path(Path(processed_dir))
    fights_path = resolved_dir / "fights_cleaned.csv"
    features_path = resolved_dir / "features.csv"
    return _cached_snapshot_max_event_date(
        str(resolved_dir),
        _path_mtime_ns(fights_path),
        _path_mtime_ns(features_path),
    )


def get_processed_snapshot_fingerprints(processed_dir: Path) -> dict[str, int | str]:
    resolved_dir = _resolved_path(Path(processed_dir))
    fights_path = resolved_dir / "fights_cleaned.csv"
    features_path = resolved_dir / "features.csv"
    _require_existing_file(fights_path, label="processed fights snapshot")
    _require_existing_file(features_path, label="processed features snapshot")
    return {
        "processed_fights_sha256": _file_sha256(fights_path),
        "processed_features_sha256": _file_sha256(features_path),
        "processed_fights_bytes": _file_size_bytes(fights_path),
        "processed_features_bytes": _file_size_bytes(features_path),
    }


def _embedded_training_spec_name(path: Path, model_result: dict[str, Any] | None = None) -> str:
    if model_result is None:
        import joblib

        model_result = joblib.load(path)
    spec = model_result.get("training_spec")
    if not isinstance(spec, dict):
        raise ProductionBundleError(f"Model artifact {path} is missing an embedded training_spec.")
    spec_name = str(spec.get("name") or "").strip()
    if not spec_name:
        raise ProductionBundleError(f"Model artifact {path} is missing training_spec.name.")
    return spec_name


def _artifact_path_from_model_result(model_result: dict[str, Any] | None) -> Path | None:
    if not isinstance(model_result, dict):
        return None
    raw = str(model_result.get("artifact_path") or "").strip()
    return Path(raw) if raw else None


def _require_existing_file(path: Path, *, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise ProductionBundleError(f"Production bundle {label} is missing: {path}")


def _require_matching_path(*, label: str, expected: Path, actual: Path) -> None:
    if _resolved_path(expected) != _resolved_path(actual):
        raise ProductionBundleError(
            f"Production bundle {label} mismatch: expected {expected}, got {actual}."
        )


def reconcile_production_bundle_manifest(
    *,
    target_manifest_path: str | Path | None = None,
    source_manifest_path: str | Path | None = None,
    model_path: str | Path | None = None,
    no_odds_model_path: str | Path | None = None,
    logistic_model_path: str | Path | None = None,
    processed_dir: str | Path | None = None,
) -> dict[str, Any]:
    target_path = _resolve_manifest_path(target_manifest_path)

    base_payload: dict[str, Any] = {}
    source_path: Path | None = None
    if source_manifest_path is not None:
        source_path = _resolve_manifest_path(source_manifest_path)
        if source_path.exists():
            base_payload = _load_manifest_payload(source_path)
    elif target_path.exists():
        base_payload = _load_manifest_payload(target_path)
    elif DEFAULT_MANIFEST_PATH.exists() and _resolved_path(target_path) != _resolved_path(DEFAULT_MANIFEST_PATH):
        source_path = DEFAULT_MANIFEST_PATH
        base_payload = _load_manifest_payload(DEFAULT_MANIFEST_PATH)

    resolved_model_path = _resolved_path(Path(model_path) if model_path is not None else MODELS_DIR / "xgboost_model.pkl")
    resolved_no_odds_model_path = _resolved_path(
        Path(no_odds_model_path) if no_odds_model_path is not None else MODELS_DIR / "xgboost_no_odds_model.pkl"
    )
    resolved_processed_dir = _resolved_path(Path(processed_dir) if processed_dir is not None else PROCESSED_DATA_DIR)

    resolved_logistic_model_path: Path | None
    if logistic_model_path is not None:
        candidate = _resolved_path(Path(logistic_model_path))
        resolved_logistic_model_path = candidate if candidate.exists() else None
    else:
        candidate = MODELS_DIR / "logistic_model.pkl"
        resolved_logistic_model_path = _resolved_path(candidate) if candidate.exists() else None

    model_spec_name = _embedded_training_spec_name(resolved_model_path)
    no_odds_spec_name = _embedded_training_spec_name(resolved_no_odds_model_path)
    snapshot_max_event_date = get_processed_snapshot_max_event_date(resolved_processed_dir)
    if snapshot_max_event_date is None:
        raise ProductionBundleError(
            f"Production bundle processed snapshot in {resolved_processed_dir} has no usable event_date column."
        )
    processed_fingerprints = get_processed_snapshot_fingerprints(resolved_processed_dir)
    manifest_updated_at = _runtime_timestamp_now()

    payload = dict(base_payload)
    payload.update(
        {
            "manifest_version": int(payload.get("manifest_version") or 1),
            "bundle_id": _optional_string(base_payload, "bundle_id")
            or _default_bundle_id(
                model_spec_name=model_spec_name,
                snapshot_max_event_date=snapshot_max_event_date,
            ),
            "model_spec_name": model_spec_name,
            "no_odds_model_spec_name": no_odds_spec_name,
            "model_path": str(resolved_model_path),
            "no_odds_model_path": str(resolved_no_odds_model_path),
            "processed_dir": str(resolved_processed_dir),
            "snapshot_max_event_date": snapshot_max_event_date,
            "built_at": _optional_string(base_payload, "built_at") or manifest_updated_at,
            "git_sha": _determine_git_sha(base_payload),
            "manifest_updated_at": manifest_updated_at,
            **processed_fingerprints,
        }
    )
    if resolved_logistic_model_path is not None:
        payload["logistic_model_path"] = str(resolved_logistic_model_path)
    else:
        payload.pop("logistic_model_path", None)

    write_json_atomically(payload, target_path)
    summary = validate_production_bundle(load_production_bundle(target_path))
    summary["source_manifest_path"] = str(source_path) if source_path is not None else None
    return summary


def runtime_manifest_needs_source_bootstrap(
    *,
    target_manifest_path: str | Path | None = None,
    source_manifest_path: str | Path | None = None,
) -> bool:
    if source_manifest_path is None:
        raise ProductionBundleError("A source manifest is required to determine runtime bundle bootstrap policy.")
    source_bundle = load_production_bundle(source_manifest_path)
    target_bundle = load_production_bundle_or_none(target_manifest_path)
    if target_bundle is None:
        return True
    if (
        target_bundle.bundle_id != source_bundle.bundle_id
        or target_bundle.model_spec_name != source_bundle.model_spec_name
    ):
        return True
    try:
        validate_production_bundle(target_bundle)
    except ProductionBundleError:
        return True
    return False


def validate_production_bundle(
    bundle: ProductionBundle,
    *,
    primary_model_result: dict[str, Any] | None = None,
    no_odds_model_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _require_existing_file(bundle.manifest_path, label="manifest")
    _require_existing_file(bundle.model_path, label="primary model")
    _require_existing_file(bundle.no_odds_model_path, label="no-odds model")
    if bundle.logistic_model_path is not None:
        _require_existing_file(bundle.logistic_model_path, label="logistic model")

    fights_path = bundle.processed_dir / "fights_cleaned.csv"
    features_path = bundle.processed_dir / "features.csv"
    _require_existing_file(fights_path, label="processed fights snapshot")
    _require_existing_file(features_path, label="processed features snapshot")

    actual_snapshot_max_event_date = get_processed_snapshot_max_event_date(bundle.processed_dir)
    if actual_snapshot_max_event_date is None:
        raise ProductionBundleError(
            f"Production bundle processed snapshot in {bundle.processed_dir} has no usable event_date column."
        )
    if actual_snapshot_max_event_date < bundle.snapshot_max_event_date:
        raise ProductionBundleError(
            "Production bundle processed snapshot is older than expected: "
            f"{actual_snapshot_max_event_date} < {bundle.snapshot_max_event_date}."
        )
    actual_processed_fingerprints = get_processed_snapshot_fingerprints(bundle.processed_dir)
    if (
        bundle.processed_fights_sha256 is not None
        and actual_processed_fingerprints["processed_fights_sha256"] != bundle.processed_fights_sha256
    ):
        raise ProductionBundleError(
            "Production bundle processed fights snapshot hash mismatch: "
            f"manifest expects {bundle.processed_fights_sha256}, "
            f"artifact is {actual_processed_fingerprints['processed_fights_sha256']}."
        )
    if (
        bundle.processed_features_sha256 is not None
        and actual_processed_fingerprints["processed_features_sha256"] != bundle.processed_features_sha256
    ):
        raise ProductionBundleError(
            "Production bundle processed features snapshot hash mismatch: "
            f"manifest expects {bundle.processed_features_sha256}, "
            f"artifact is {actual_processed_fingerprints['processed_features_sha256']}."
        )
    if (
        bundle.processed_fights_bytes is not None
        and actual_processed_fingerprints["processed_fights_bytes"] != bundle.processed_fights_bytes
    ):
        raise ProductionBundleError(
            "Production bundle processed fights snapshot size mismatch: "
            f"manifest expects {bundle.processed_fights_bytes}, "
            f"artifact is {actual_processed_fingerprints['processed_fights_bytes']}."
        )
    if (
        bundle.processed_features_bytes is not None
        and actual_processed_fingerprints["processed_features_bytes"] != bundle.processed_features_bytes
    ):
        raise ProductionBundleError(
            "Production bundle processed features snapshot size mismatch: "
            f"manifest expects {bundle.processed_features_bytes}, "
            f"artifact is {actual_processed_fingerprints['processed_features_bytes']}."
        )

    _require_matching_path(
        label="primary alias path",
        expected=bundle.model_path,
        actual=MODELS_DIR / "xgboost_model.pkl",
    )
    _require_matching_path(
        label="no-odds alias path",
        expected=bundle.no_odds_model_path,
        actual=MODELS_DIR / "xgboost_no_odds_model.pkl",
    )
    if bundle.logistic_model_path is not None:
        _require_matching_path(
            label="logistic alias path",
            expected=bundle.logistic_model_path,
            actual=MODELS_DIR / "logistic_model.pkl",
        )

    loaded_primary_path = _artifact_path_from_model_result(primary_model_result)
    if loaded_primary_path is not None:
        _require_matching_path(
            label="loaded primary model",
            expected=bundle.model_path,
            actual=loaded_primary_path,
        )
    loaded_no_odds_path = _artifact_path_from_model_result(no_odds_model_result)
    if loaded_no_odds_path is not None:
        _require_matching_path(
            label="loaded no-odds model",
            expected=bundle.no_odds_model_path,
            actual=loaded_no_odds_path,
        )

    embedded_model_spec_name = _embedded_training_spec_name(
        bundle.model_path,
        model_result=primary_model_result,
    )
    if embedded_model_spec_name != bundle.model_spec_name:
        raise ProductionBundleError(
            "Production bundle model spec mismatch: "
            f"manifest expects {bundle.model_spec_name}, artifact embeds {embedded_model_spec_name}."
        )

    embedded_no_odds_model_spec_name = _embedded_training_spec_name(
        bundle.no_odds_model_path,
        model_result=no_odds_model_result,
    )
    expected_no_odds_spec = bundle.no_odds_model_spec_name
    if expected_no_odds_spec and embedded_no_odds_model_spec_name != expected_no_odds_spec:
        raise ProductionBundleError(
            "Production bundle no-odds model spec mismatch: "
            f"manifest expects {expected_no_odds_spec}, artifact embeds {embedded_no_odds_model_spec_name}."
        )

    return {
        "bundle_id": bundle.bundle_id,
        "manifest_path": str(bundle.manifest_path),
        "manifest_version": bundle.manifest_version,
        "model_spec_name": bundle.model_spec_name,
        "embedded_model_spec_name": embedded_model_spec_name,
        "model_path": str(bundle.model_path),
        "no_odds_model_spec_name": expected_no_odds_spec or embedded_no_odds_model_spec_name,
        "embedded_no_odds_model_spec_name": embedded_no_odds_model_spec_name,
        "no_odds_model_path": str(bundle.no_odds_model_path),
        "logistic_model_path": str(bundle.logistic_model_path) if bundle.logistic_model_path else None,
        "processed_dir": str(bundle.processed_dir),
        "manifest_snapshot_max_event_date": bundle.snapshot_max_event_date,
        "processed_snapshot_max_event_date": actual_snapshot_max_event_date,
        "processed_fights_sha256": actual_processed_fingerprints["processed_fights_sha256"],
        "processed_features_sha256": actual_processed_fingerprints["processed_features_sha256"],
        "processed_fights_bytes": actual_processed_fingerprints["processed_fights_bytes"],
        "processed_features_bytes": actual_processed_fingerprints["processed_features_bytes"],
        "built_at": bundle.built_at,
        "manifest_updated_at": bundle.manifest_updated_at,
        "git_sha": bundle.git_sha,
    }
