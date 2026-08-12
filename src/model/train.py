"""
Model training — trains XGBoost and Logistic Regression models
for UFC fight prediction with probability calibration.
"""

import logging
import inspect
import json
import hashlib
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    TRAIN_CUTOFF_DATE, MODELS_DIR, PROCESSED_DATA_DIR,
    TIME_DECAY_ENABLED, TIME_DECAY_HALF_LIFE_DAYS,
    ODDS_NOISE_STD,
)
from src.features.build_features import (
    ODDS_FEATURE_NAMES,
    exclude_market_derived_features,
    get_feature_columns,
    get_feature_columns_no_odds,
)


_SCHEDULED_REFIT_POLICY_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "scheduled_refit_policy_v2.json"
)
_METHOD_CONTRACT_DECISION_PATH = (
    Path(__file__).resolve().parents[2]
    / "evidence"
    / "method_odds"
    / "integrity_v2_contract_decision.json"
)
from src.model.orientation import has_ab_feature_pair, swap_directional_frame

logger = logging.getLogger(__name__)
SUPPORTED_CALIBRATION_CV = frozenset({
    "random_5fold", "timeseries_5fold", "temporal_holdout", "temporal_holdout_refit",
    "temporal_holdout_weighted",
})
SUPPORTED_ODDS_NOISE_MODES = frozenset({"independent", "antithetic"})


def _matches_method_selected_fullfit_semantics(
    spec: "NamedModelTrainingSpec | None",
) -> bool:
    if spec is None:
        return False
    from src.model.training_spec import (
        named_training_spec_factories,
        resolve_named_training_spec,
    )

    base_specs = []
    try:
        decision = json.loads(_METHOD_CONTRACT_DECISION_PATH.read_text(encoding="utf-8"))
        base_specs.append(
            resolve_named_training_spec(str(decision["selected_fullfit_contract"]))
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError):
        pass
    # Discover the registered integrity full-fit family even when the method
    # decision is readable. This prevents an unselected base contract (or a
    # registered descendant) from being treated as unrelated research merely
    # because the decision selected a sibling feature contract.
    for name in named_training_spec_factories():
        if "_integrity_" in name and name.endswith("_fullfit"):
            try:
                registered = resolve_named_training_spec(name)
            except ValueError:
                continue
            if all(registered.name != existing.name for existing in base_specs):
                base_specs.append(registered)
    if not base_specs:
        return False
    invariant_fields = (
        "dataset_variant",
        "train_start_date",
        "train_end_date",
        "train_cutoff_date",
        "odds_noise_std",
        "odds_noise_seed",
        "odds_noise_mode",
        "add_rematch_features",
        "add_line_movement",
        "impute_strategy",
        "impute_with_indicators",
    )
    return any(
        list(spec.feature_cols) == list(base.feature_cols)
        and all(
            getattr(spec, field) == getattr(base, field)
            for field in invariant_fields
        )
        and (spec.xgb_params or {}).get("random_state")
        == (base.xgb_params or {}).get("random_state")
        for base in base_specs
    )


def _is_current_policy_selected_fullfit_spec(
    spec: "NamedModelTrainingSpec | None",
) -> bool:
    """Identify the selected full-fit contract and fail closed on policy damage."""

    if spec is None:
        return False
    method_fullfit = _matches_method_selected_fullfit_semantics(spec)
    try:
        from scripts import check_production_refit_contract as contract_gate

        policy = contract_gate.load_policy(_SCHEDULED_REFIT_POLICY_PATH)
        registry_errors, _evaluation, fullfit = contract_gate.validate_policy_registry(
            policy
        )
    except (OSError, ValueError) as exc:
        if method_fullfit:
            raise ValueError(
                "Policy-selected integrity full-fit cannot validate the scheduled policy"
            ) from exc
        return False
    if registry_errors or fullfit is None:
        if method_fullfit:
            raise ValueError(
                "Policy-selected integrity full-fit cannot use an invalid scheduled policy: "
                + "; ".join(registry_errors)
            )
        return False
    contract = policy["contract"]
    if contract.get("fullfit_spec_name") != spec.name:
        if method_fullfit:
            raise ValueError(
                f"Integrity full-fit spec {spec.name!r} is not selected by policy"
            )
        return False
    from dataclasses import asdict

    payload = asdict(spec)
    payload["trained_at"] = ""
    payload["git_hash"] = ""
    if contract.get("fullfit_spec_payload_sha256") != _canonical_json_sha256(payload):
        raise ValueError(
            "Policy-selected full-fit spec payload differs from the strict registry"
        )
    return True


class HoldoutCalibratedRefitModel:
    """
    Serve a full-data booster through a holdout-fitted calibrator (E2).

    ``temporal_holdout`` trains the served booster on only the first ~80% of
    rows; the newest fights (the heaviest time-decay mass) never reach it.
    This wrapper keeps the calibration honest — the sigmoid is fitted on the
    inner booster's out-of-sample holdout predictions exactly as before — but
    applies that frozen mapping to a booster refit on 100% of the data.

    The calibrator must NOT be refit against the full booster's in-sample
    predictions (that would calibrate on training outputs).
    """

    def __init__(self, base_estimator, holdout_calibrated):
        self.base_estimator = base_estimator
        self.holdout_calibrated = holdout_calibrated
        self.classes_ = getattr(base_estimator, "classes_", np.array([0, 1]))

    def _calibrator(self):
        calibrated = self.holdout_calibrated.calibrated_classifiers_[0]
        return calibrated.calibrators[0]

    def predict_proba(self, X):
        raw = np.asarray(self.base_estimator.predict_proba(X), dtype=float)[:, 1]
        calibrated = np.asarray(self._calibrator().predict(raw), dtype=float)
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
TEST_SET_METADATA_SUFFIX = ".metadata.json"


def _feature_contract_hash(feature_cols: list[str]) -> str:
    """Return a stable hash for an ordered feature contract."""
    payload = json.dumps(list(feature_cols), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_set_metadata_path(test_set_path: Path | str) -> Path:
    """Return the JSON metadata sidecar path for a test set CSV."""
    path = Path(test_set_path)
    return path.with_name(f"{path.name}{TEST_SET_METADATA_SUFFIX}")


def build_test_set_metadata(
    *,
    spec: "NamedModelTrainingSpec | None",
    feature_cols: list[str],
    test_df: pd.DataFrame,
    generated_at: str | None = None,
    training_input_evidence: dict | None = None,
) -> dict:
    """Build static test-set compatibility metadata."""
    spec_name = ""
    dataset_variant = "default"
    train_start_date = ""
    train_end_date = ""
    train_cutoff_date = TRAIN_CUTOFF_DATE
    training_spec_payload = None

    if spec is not None:
        from dataclasses import asdict

        training_spec_payload = asdict(spec)
        spec_name = str(training_spec_payload.get("name", "") or "")
        dataset_variant = str(training_spec_payload.get("dataset_variant", "default") or "default")
        train_start_date = str(training_spec_payload.get("train_start_date", "") or "")
        train_end_date = str(training_spec_payload.get("train_end_date", "") or "")
        train_cutoff_date = str(training_spec_payload.get("train_cutoff_date", TRAIN_CUTOFF_DATE) or TRAIN_CUTOFF_DATE)

    metadata = {
        "spec_name": spec_name,
        "feature_count": len(feature_cols),
        "feature_hash": _feature_contract_hash(feature_cols),
        "dataset_variant": dataset_variant,
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "train_cutoff_date": train_cutoff_date,
        "generated_at": generated_at or datetime.now().isoformat(),
        "row_count": int(len(test_df)),
        "training_spec": training_spec_payload,
    }
    if training_input_evidence is not None:
        metadata["training_input_evidence"] = deepcopy(training_input_evidence)
    return metadata


def write_test_set_metadata(
    *,
    test_set_path: Path | str,
    spec: "NamedModelTrainingSpec | None",
    feature_cols: list[str],
    test_df: pd.DataFrame,
    training_input_evidence: dict | None = None,
) -> dict:
    """Write a JSON sidecar describing the static test-set contract."""
    test_set_path = Path(test_set_path)
    metadata_path = test_set_metadata_path(test_set_path)
    metadata = build_test_set_metadata(
        spec=spec,
        feature_cols=list(feature_cols),
        test_df=test_df,
        training_input_evidence=training_input_evidence,
    )
    metadata["test_set_path"] = str(test_set_path.resolve(strict=False))
    metadata["test_set_sha256"] = _sha256_file(test_set_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved test-set metadata to %s", metadata_path)
    return metadata


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_training_input_evidence(
    evidence: dict | None,
    *,
    features_df: pd.DataFrame,
    required_spec: "NamedModelTrainingSpec | None" = None,
) -> dict | None:
    """Validate and copy a policy-bound trainer-input receipt."""
    if evidence is None:
        return None
    if not isinstance(evidence, dict):
        raise ValueError("training_input_evidence must be a mapping")
    payload = deepcopy(evidence)
    receipt_sha256 = str(payload.pop("receipt_sha256", "") or "").lower()
    if receipt_sha256 != _canonical_json_sha256(payload):
        raise ValueError("training_input_evidence receipt SHA-256 is invalid")
    if payload.get("schema_version") != 1:
        raise ValueError("training_input_evidence schema_version must be 1")
    if payload.get("preparation") != "verified_t_minus_entry_model_odds":
        raise ValueError("training_input_evidence preparation is unsupported")
    if required_spec is not None:
        from dataclasses import asdict

        if payload.get("fullfit_spec_name") != required_spec.name:
            raise ValueError(
                "training_input_evidence full-fit spec does not match trainer spec"
            )
        spec_payload = asdict(required_spec)
        spec_payload["trained_at"] = ""
        spec_payload["git_hash"] = ""
        spec_sha256 = _canonical_json_sha256(spec_payload)
        if payload.get("fullfit_spec_payload_sha256") != spec_sha256:
            raise ValueError(
                "training_input_evidence full-fit spec SHA-256 does not match trainer spec"
            )

        policy_path = Path(str(payload.get("policy_path") or ""))
        if not policy_path.is_file():
            raise ValueError("training_input_evidence policy file is missing")
        if str(payload.get("policy_sha256") or "").lower() != _sha256_file(
            policy_path
        ):
            raise ValueError("training_input_evidence policy SHA-256 is stale")
        try:
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("training_input_evidence policy cannot be inspected") from exc
        contract = policy.get("contract") if isinstance(policy, dict) else None
        evaluation = policy.get("evaluation") if isinstance(policy, dict) else None
        try:
            policy_entry_offset = float(
                evaluation.get("entry_offset_days", -1.0)
                if isinstance(evaluation, dict)
                else -1.0
            )
            receipt_entry_offset = float(payload.get("entry_offset_days", -2.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "training_input_evidence entry offset is malformed"
            ) from exc
        if (
            not isinstance(policy, dict)
            or policy.get("schema_version") != 2
            or not isinstance(contract, dict)
            or not isinstance(evaluation, dict)
            or contract.get("fullfit_spec_name") != required_spec.name
            or contract.get("fullfit_spec_payload_sha256") != spec_sha256
            or evaluation.get("entry_offset_for_features") is not True
            or evaluation.get("require_entry_odds") is not True
            or policy_entry_offset != receipt_entry_offset
            or evaluation.get("quality_allowed_prefight_sources")
            != payload.get("allowed_prefight_sources")
        ):
            raise ValueError(
                "training_input_evidence does not match the bound integrity policy"
            )
    if int(payload.get("row_count", -1)) != len(features_df):
        raise ValueError("training_input_evidence row_count does not match trainer input")

    odds_columns = payload.get("prepared_odds_columns")
    provenance_columns = payload.get("provenance_columns")
    if not isinstance(odds_columns, list) or not isinstance(provenance_columns, list):
        raise ValueError("training_input_evidence column contracts are malformed")
    missing_columns = [
        column
        for column in [*odds_columns, *provenance_columns]
        if column not in features_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"training_input_evidence columns are missing from trainer input: {missing_columns}"
        )
    present = features_df[odds_columns].notna().all(axis=1)
    partial = features_df[odds_columns].notna().any(axis=1) & ~present
    if partial.any():
        raise ValueError("trainer input contains partial policy-bound odds rows")
    if int(payload.get("rows_with_verified_t_minus_entry", -1)) != int(present.sum()):
        raise ValueError("training_input_evidence verified T-1 row count is stale")
    if int(payload.get("rows_missing_t_minus_entry", -1)) != int((~present).sum()):
        raise ValueError("training_input_evidence missing T-1 row count is stale")
    verified = features_df["model_odds_verified_prefight"].map(
        lambda value: value is True or (isinstance(value, np.bool_) and bool(value))
    )
    allowed_sources = payload.get("allowed_prefight_sources")
    if not isinstance(allowed_sources, list) or not allowed_sources:
        raise ValueError("training_input_evidence allowed sources are malformed")
    try:
        minimum_hours = float(payload["entry_offset_days"]) * 24.0
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("training_input_evidence entry offset is malformed") from exc
    hours = pd.to_numeric(features_df["model_odds_hours_to_start"], errors="coerce")
    invalid_present = present & (
        ~verified
        | ~features_df["model_odds_source_kind"].isin(allowed_sources)
        | ~hours.ge(minimum_hours)
    )
    if invalid_present.any() or verified[~present].any():
        raise ValueError("trainer input odds do not have verified T-1 provenance")
    if present.any() and not np.allclose(
        features_df.loc[present, "diff_implied_prob"].to_numpy(dtype=float),
        (
            features_df.loc[present, "a_implied_prob"]
            - features_df.loc[present, "b_implied_prob"]
        ).to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("trainer input diff_implied_prob is inconsistent")

    features_record = payload.get("features_csv")
    if not isinstance(features_record, dict):
        raise ValueError("training_input_evidence features_csv record is missing")
    features_path = Path(str(features_record.get("path") or ""))
    expected_sha256 = str(features_record.get("sha256") or "").lower()
    if not features_path.is_file():
        raise ValueError(f"training_input_evidence features CSV is missing: {features_path}")
    if expected_sha256 != _sha256_file(features_path):
        raise ValueError("training_input_evidence features CSV SHA-256 is stale")
    if int(features_record.get("bytes", -1)) != int(features_path.stat().st_size):
        raise ValueError("training_input_evidence features CSV byte count is stale")
    try:
        saved_header = list(pd.read_csv(features_path, nrows=0).columns)
        saved_rows = sum(
            len(chunk)
            for chunk in pd.read_csv(features_path, chunksize=10_000, low_memory=False)
        )
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError("training_input_evidence features CSV cannot be inspected") from exc
    if saved_header != list(features_df.columns) or saved_rows != len(features_df):
        raise ValueError("training_input_evidence features CSV shape differs from trainer input")
    in_memory_sha256 = hashlib.sha256(
        features_df.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    if in_memory_sha256 != expected_sha256:
        raise ValueError(
            "training_input_evidence features CSV values/order differ from trainer input"
        )

    source_fights_record = payload.get("source_fights_csv")
    if not isinstance(source_fights_record, dict):
        raise ValueError(
            "training_input_evidence source_fights_csv record is missing"
        )
    source_fights_path = Path(
        str(source_fights_record.get("path") or "")
    ).resolve(strict=False)
    source_fights_sha256 = str(
        source_fights_record.get("sha256") or ""
    ).lower()
    if not source_fights_path.is_file():
        raise ValueError(
            f"training_input_evidence source fights CSV is missing: {source_fights_path}"
        )
    if source_fights_sha256 != _sha256_file(source_fights_path):
        raise ValueError(
            "training_input_evidence source fights CSV SHA-256 is stale"
        )
    if int(source_fights_record.get("bytes", -1)) != int(
        source_fights_path.stat().st_size
    ):
        raise ValueError(
            "training_input_evidence source fights CSV byte count is stale"
        )

    return {**payload, "receipt_sha256": receipt_sha256}


def load_test_set_metadata(test_set_path: Path | str) -> dict:
    """Load the JSON metadata sidecar for a static test set."""
    metadata_path = test_set_metadata_path(test_set_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Static test-set metadata not found: {metadata_path}. "
            "Retrain or regenerate the test set so the metadata sidecar is written."
        )
    with metadata_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def compare_model_to_test_set_metadata(
    model_result: dict,
    test_set_metadata: dict,
) -> list[dict[str, object]]:
    """Return field mismatches between a model artifact and a static test set."""
    spec = model_result.get("training_spec")
    if not isinstance(spec, dict):
        raise ValueError("Model artifact is missing an embedded training_spec.")

    model_feature_cols = spec.get("feature_cols")
    if not isinstance(model_feature_cols, list):
        raise ValueError("Model artifact training_spec.feature_cols is invalid.")

    expected = {
        "spec_name": str(spec.get("name", "") or ""),
        "feature_count": len(model_feature_cols),
        "feature_hash": _feature_contract_hash(model_feature_cols),
        "dataset_variant": str(spec.get("dataset_variant", "default") or "default"),
        "train_start_date": str(spec.get("train_start_date", "") or ""),
        "train_end_date": str(spec.get("train_end_date", "") or ""),
        "train_cutoff_date": str(spec.get("train_cutoff_date", TRAIN_CUTOFF_DATE) or TRAIN_CUTOFF_DATE),
    }

    mismatches: list[dict[str, object]] = []
    for field, model_value in expected.items():
        test_value = test_set_metadata.get(field)
        if test_value != model_value:
            mismatches.append({
                "field": field,
                "model": model_value,
                "test_set": test_value,
            })
    return mismatches


def format_model_test_set_mismatches(
    mismatches: list[dict[str, object]],
    *,
    model_path: str | None = None,
    test_set_path: Path | str | None = None,
) -> str:
    """Format model/test-set metadata mismatches for a human-readable error."""
    model_label = str(model_path or "model artifact")
    test_label = str(Path(test_set_path) if test_set_path is not None else "test set")
    lines = [
        f"Static backtest artifact mismatch between {model_label} and {test_label}:",
    ]
    for mismatch in mismatches:
        lines.append(
            f"  - {mismatch['field']}: model={mismatch['model']!r}, "
            f"test_set={mismatch['test_set']!r}"
        )
    return "\n".join(lines)


def assert_model_matches_test_set(
    model_result: dict,
    *,
    test_set_path: Path | str,
) -> dict:
    """Fail closed when a static test set does not match the model contract."""
    metadata = load_test_set_metadata(test_set_path)
    mismatches = compare_model_to_test_set_metadata(model_result, metadata)
    if mismatches:
        raise ValueError(
            format_model_test_set_mismatches(
                mismatches,
                model_path=model_result.get("artifact_path"),
                test_set_path=test_set_path,
            )
        )
    return metadata


def _build_no_odds_training_spec_payload(spec, no_odds_cols: list[str]) -> dict:
    """Return the embedded spec payload for the no-odds artifact."""
    from dataclasses import asdict, replace

    description = str(getattr(spec, "description", "") or "").strip()
    no_odds_spec = replace(
        spec,
        name=f"{spec.name}_no_odds",
        description=f"{description} (no-odds variant)" if description else "No-odds variant",
        feature_cols=list(no_odds_cols),
    )
    return asdict(no_odds_spec)


def _repair_legacy_no_odds_training_spec_payload(
    path: Path,
    result: dict,
    artifact_cols: list[str],
    spec: dict,
) -> bool:
    """
    Repair the known legacy no-odds metadata bug in promoted artifacts.

    Older no-odds models embedded the full odds-aware spec even though the
    trained artifact removed market-derived columns. Accept only that exact
    mismatch so the strict feature-contract validator still rejects any other
    contract drift.
    """
    if path.name != "xgboost_no_odds_model.pkl":
        return False

    spec_cols = spec.get("feature_cols")
    if not isinstance(spec_cols, list):
        return False

    expected_no_odds_cols = exclude_market_derived_features(spec_cols)
    if artifact_cols != expected_no_odds_cols:
        return False

    repaired_spec = dict(spec)
    repaired_spec["feature_cols"] = list(artifact_cols)
    name = str(repaired_spec.get("name", "") or "").strip()
    repaired_spec["name"] = f"{name}_no_odds" if name and not name.endswith("_no_odds") else (name or "xgboost_no_odds")

    description = str(repaired_spec.get("description", "") or "").strip()
    if description and "no-odds" not in description.lower():
        repaired_spec["description"] = f"{description} (no-odds variant)"
    elif not description:
        repaired_spec["description"] = "No-odds variant"

    result["training_spec"] = repaired_spec
    logger.warning(
        "Repairing legacy no-odds training_spec feature contract for %s.",
        path,
    )
    return True


def _persist_repaired_training_spec_payload(path: Path, result: dict) -> None:
    """Best-effort persistence for in-place metadata repairs."""
    try:
        joblib.dump(result, path)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(
            "Persisting repaired training_spec metadata failed for %s: %s",
            path,
            exc,
        )


def _call_train_logistic(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    odds_noise_std: float,
    odds_noise_seed: int | None = None,
) -> dict:
    """Call train_logistic while remaining compatible with older test doubles."""
    signature = inspect.signature(train_logistic)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs: dict[str, object] = {}
    if accepts_kwargs or "odds_noise_std" in signature.parameters:
        kwargs["odds_noise_std"] = odds_noise_std
    if accepts_kwargs or "odds_noise_seed" in signature.parameters:
        kwargs["odds_noise_seed"] = odds_noise_seed
    if kwargs:
        return train_logistic(train_df, feature_cols, **kwargs)
    return train_logistic(train_df, feature_cols)


def _resolve_repo_git_hash() -> str:
    """Return the current repo HEAD hash, or an empty string if unavailable."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def _dead_train_feature_columns(train_df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Return declared feature columns that are entirely missing on trainable rows."""
    dead_columns: list[str] = []
    for column in feature_cols:
        if column not in train_df.columns:
            continue
        if train_df[column].isna().all():
            dead_columns.append(column)
    return dead_columns


def prepare_train_test(
    features_df: pd.DataFrame,
    cutoff_date: Optional[str] = None,
    min_fights: int = 2,
    feature_cols: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Split features into train/test sets by date.
    Filters out fights where either fighter has fewer than min_fights prior bouts.

    If *start_date* is provided, training rows before that date are dropped.
    If *end_date* is provided, training rows on/after that date are dropped.
    The test set is unaffected.

    Returns (train_df, test_df, feature_columns).
    """
    cutoff = pd.Timestamp(cutoff_date or TRAIN_CUTOFF_DATE)
    feature_cols = feature_cols or get_feature_columns(features_df)

    # Filter to fights where we have enough data
    df = features_df.copy()
    if "a_num_fights" in df.columns and "b_num_fights" in df.columns:
        df = df[
            (df["a_num_fights"] >= min_fights) & (df["b_num_fights"] >= min_fights)
        ]

    # Drop rows with missing target
    df = df.dropna(subset=["target"])

    # Drop rows where all features are NaN
    df = df.dropna(subset=feature_cols, how="all")

    if "event_date" not in df.columns:
        raise ValueError("features_df must include event_date for temporal train/test split")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    if df["event_date"].isna().any():
        raise ValueError("features_df contains unparseable event_date values after filtering")
    df["_input_order"] = np.arange(len(df))
    df = df.sort_values(["event_date", "_input_order"], kind="mergesort").drop(columns="_input_order")
    if not df["event_date"].is_monotonic_increasing:
        raise ValueError("features_df event_date ordering is not monotonic after temporal sort")

    train = df[df["event_date"] < cutoff].copy()
    test = df[df["event_date"] >= cutoff].copy()

    # Apply optional lower-bound filter to training data only
    if start_date:
        start = pd.Timestamp(start_date)
        pre_filter_count = len(train)
        train = train[train["event_date"] >= start].copy()
        logger.info(
            f"train_start_date={start.date()}: dropped {pre_filter_count - len(train)} "
            f"pre-{start.date()} training rows"
        )
    if end_date:
        end = pd.Timestamp(end_date)
        pre_filter_count = len(train)
        train = train[train["event_date"] < end].copy()
        logger.info(
            f"train_end_date={end.date()}: dropped {pre_filter_count - len(train)} "
            f"training rows on/after {end.date()}"
        )

    logger.info(
        f"Train: {len(train)} fights (before {cutoff.date()}), "
        f"Test: {len(test)} fights (after {cutoff.date()})"
    )
    logger.info(f"Using {len(feature_cols)} features")

    return train, test, feature_cols


def _mirror_augment_training_rows(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, bool]:
    """Duplicate observed rows with A/B swapped and the binary target flipped."""
    if not has_ab_feature_pair(feature_cols):
        return train_df.copy(), False
    if "target" not in train_df.columns:
        raise ValueError("Training data must include target before mirror augmentation")

    base = train_df.copy()
    target = pd.to_numeric(base["target"], errors="coerce")
    valid_targets = target.notna() & target.isin([0, 1])
    if not valid_targets.all():
        raise ValueError("Mirror augmentation requires observed binary targets only")

    base["_mirror_group_id"] = np.arange(len(base), dtype=int)
    base["_mirror_augmented"] = 0

    schema = list(dict.fromkeys(list(base.columns) + list(feature_cols)))
    mirrored = swap_directional_frame(base, columns=schema)
    mirrored["target"] = 1.0 - target.astype(float).to_numpy()
    mirrored["_mirror_group_id"] = base["_mirror_group_id"].to_numpy()
    mirrored["_mirror_augmented"] = 1

    augmented = pd.concat([base, mirrored], ignore_index=True, sort=False)
    sort_cols = []
    if "event_date" in augmented.columns:
        augmented["event_date"] = pd.to_datetime(augmented["event_date"], errors="coerce")
        sort_cols.append("event_date")
    sort_cols.extend(["_mirror_group_id", "_mirror_augmented"])
    augmented = augmented.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    logger.info(
        "Applied A/B mirror augmentation: %d observed rows -> %d training rows",
        len(base),
        len(augmented),
    )
    return augmented, True


def _temporal_holdout_indices(train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return row indices for the inner-train/calibration temporal split."""
    if "_mirror_group_id" in train_df.columns:
        groups = train_df["_mirror_group_id"].to_numpy()
        unique_groups = pd.unique(groups)
        group_split_idx = int(len(unique_groups) * 0.8)
        group_split_idx = max(1, min(group_split_idx, len(unique_groups) - 1))
        train_groups = set(unique_groups[:group_split_idx])
        cal_groups = set(unique_groups[group_split_idx:])
        inner = np.array([i for i, group in enumerate(groups) if group in train_groups], dtype=int)
        cal = np.array([i for i, group in enumerate(groups) if group in cal_groups], dtype=int)
        return inner, cal

    split_idx = int(len(train_df) * 0.8)
    split_idx = max(1, min(split_idx, len(train_df) - 1))
    return np.arange(split_idx), np.arange(split_idx, len(train_df))


def _timeseries_cv_for_training(train_df: pd.DataFrame, n_splits: int = 5):
    """Build time-series CV splits without separating mirrored row pairs."""
    if "_mirror_group_id" not in train_df.columns:
        return TimeSeriesSplit(n_splits=n_splits)

    groups = train_df["_mirror_group_id"].to_numpy()
    unique_groups = pd.unique(groups)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_group_idx, test_group_idx in splitter.split(unique_groups):
        train_groups = set(unique_groups[train_group_idx])
        test_groups = set(unique_groups[test_group_idx])
        train_idx = np.array([i for i, group in enumerate(groups) if group in train_groups], dtype=int)
        test_idx = np.array([i for i, group in enumerate(groups) if group in test_groups], dtype=int)
        splits.append((train_idx, test_idx))
    return splits


def _group_cv_for_training(train_df: pd.DataFrame, n_splits: int = 5):
    """Build random-style CV splits without mirrored pair leakage."""
    if "_mirror_group_id" not in train_df.columns:
        return n_splits

    groups = train_df["_mirror_group_id"].to_numpy()
    unique_groups = pd.unique(groups)
    effective_splits = min(n_splits, len(unique_groups))
    if effective_splits < 2:
        raise ValueError("Group calibration requires at least two observed fights")
    return list(GroupKFold(n_splits=effective_splits).split(np.zeros(len(train_df)), train_df["target"], groups))


def _compute_sample_weights(
    train_df: pd.DataFrame,
    *,
    half_life_days: float | None = None,
) -> Optional[np.ndarray]:
    """Compute time-decay sample weights based on fight recency."""
    if not TIME_DECAY_ENABLED:
        return None
    if "event_date" not in train_df.columns:
        return None

    effective_half_life = half_life_days if half_life_days is not None else TIME_DECAY_HALF_LIFE_DAYS

    dates = pd.to_datetime(train_df["event_date"])
    max_date = dates.max()
    days_ago = (max_date - dates).dt.days.values.astype(float)

    # Exponential decay: weight = 2^(-days_ago / half_life)
    weights = np.power(2.0, -days_ago / effective_half_life)

    # Normalize so mean weight = 1.0 (preserves effective sample size interpretation)
    weights = weights / weights.mean()

    logger.info(
        f"Time-decay weights: half-life={effective_half_life}d, "
        f"min={weights.min():.3f}, max={weights.max():.3f}, "
        f"effective N={weights.sum():.0f}/{len(weights)}"
    )
    return weights


def _add_odds_noise(
    X: np.ndarray,
    feature_cols: list[str],
    noise_std: float = ODDS_NOISE_STD,
    rng: Optional[np.random.RandomState] = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Add Gaussian noise to odds-derived features to mitigate closing odds leakage.

    Training data contains closing odds (final lines right before the fight), but
    at prediction time we only have current/opening odds. Closing odds are more
    accurate, so the model overfits to their precision. Adding noise simulates
    the gap between current odds and closing odds.

    Implied probabilities are clipped to [0.02, 0.98] after noise.
    """
    if noise_std <= 0:
        return X

    if rng is None:
        rng = np.random.RandomState(42 if seed is None else int(seed))

    odds_indices = [i for i, col in enumerate(feature_cols) if col in ODDS_FEATURE_NAMES]

    if not odds_indices:
        return X

    X = X.copy()
    for idx in odds_indices:
        col_name = feature_cols[idx]
        noise = rng.normal(0, noise_std, size=X.shape[0])
        X[:, idx] = X[:, idx] + noise

        # Clip implied probabilities to valid range
        if "implied_prob" in col_name and "diff" not in col_name:
            X[:, idx] = np.clip(X[:, idx], 0.02, 0.98)

    noised_cols = [feature_cols[i] for i in odds_indices]
    logger.info(f"Added odds noise (std={noise_std}) to {len(odds_indices)} features: {noised_cols}")

    return X


def _add_antithetic_odds_noise(
    X: np.ndarray,
    feature_cols: list[str],
    train_df: pd.DataFrame,
    noise_std: float = ODDS_NOISE_STD,
    seed: int | None = None,
) -> np.ndarray:
    """
    Mirror-consistent odds noise that preserves the live no-vig identities.

    The legacy independent draws break two structures every live row
    satisfies exactly: a+b=1 and diff=a-b (the model trains off the manifold
    it predicts on), and mirrored twins receive unrelated noise, breaking the
    A/B symmetry mirror augmentation established.

    Here each mirror group draws ONE market shock eps: the original row gets
    a+eps (b and diff recomputed from the identities), the mirrored row gets
    the sign-flipped shock. Rows missing either moneyline probability keep
    legacy independent noise (there is no identity to preserve). Method-odds
    columns are vig-included with no live identity — they keep independent
    draws.
    """
    if noise_std <= 0:
        return X

    rng = np.random.RandomState(42 if seed is None else int(seed))
    col_index = {col: i for i, col in enumerate(feature_cols)}
    a_idx = col_index.get("a_implied_prob")
    b_idx = col_index.get("b_implied_prob")
    diff_idx = col_index.get("diff_implied_prob")

    X = X.copy()
    n_rows = X.shape[0]

    if a_idx is not None and b_idx is not None:
        if "_mirror_group_id" in train_df.columns:
            group_ids = train_df["_mirror_group_id"].to_numpy()
            mirrored = train_df.get("_mirror_augmented")
            mirrored = (
                mirrored.to_numpy().astype(float)
                if mirrored is not None
                else np.zeros(n_rows)
            )
        else:
            group_ids = np.arange(n_rows)
            mirrored = np.zeros(n_rows)

        unique_groups, group_pos = np.unique(group_ids, return_inverse=True)
        eps_per_group = rng.normal(0, noise_std, size=len(unique_groups))
        eps = eps_per_group[group_pos] * np.where(mirrored > 0, -1.0, 1.0)

        a_vals = X[:, a_idx]
        b_vals = X[:, b_idx]
        pair_mask = ~np.isnan(a_vals) & ~np.isnan(b_vals)

        a_noised = np.clip(a_vals[pair_mask] + eps[pair_mask], 0.02, 0.98)
        X[pair_mask, a_idx] = a_noised
        X[pair_mask, b_idx] = 1.0 - a_noised
        if diff_idx is not None:
            X[pair_mask, diff_idx] = 2.0 * a_noised - 1.0

        # Rows with a one-sided moneyline keep the legacy independent noise.
        loose_mask = ~pair_mask
        for idx in (a_idx, b_idx):
            vals = X[loose_mask, idx]
            vals = vals + rng.normal(0, noise_std, size=loose_mask.sum())
            X[loose_mask, idx] = np.clip(vals, 0.02, 0.98)

    # Method-odds (and any other odds features outside the moneyline trio):
    # independent draws, as before.
    moneyline = {"a_implied_prob", "b_implied_prob", "diff_implied_prob"}
    other_odds = [
        i for i, col in enumerate(feature_cols)
        if col in ODDS_FEATURE_NAMES and col not in moneyline
    ]
    for idx in other_odds:
        col_name = feature_cols[idx]
        X[:, idx] = X[:, idx] + rng.normal(0, noise_std, size=n_rows)
        if "implied_prob" in col_name and "diff" not in col_name:
            X[:, idx] = np.clip(X[:, idx], 0.02, 0.98)

    logger.info(
        "Added antithetic odds noise (std=%s) — moneyline identities preserved",
        noise_std,
    )
    return X


def _validate_calibration_cv(calibration_cv: str) -> str:
    normalized = str(calibration_cv or "").strip()
    if normalized not in SUPPORTED_CALIBRATION_CV:
        supported = ", ".join(sorted(SUPPORTED_CALIBRATION_CV))
        raise ValueError(f"Unsupported calibration_cv '{calibration_cv}'. Supported values: {supported}")
    return normalized


def train_xgboost(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    calibrate: bool = True,
    impute_strategy: str = "native_nan",
    xgb_params: dict | None = None,
    calibration_method: str = "isotonic",
    calibration_cv: str = "timeseries_5fold",
    odds_noise_std: float = ODDS_NOISE_STD,
    odds_noise_seed: int | None = None,
    time_decay_half_life_days: float | None = None,
    odds_noise_mode: str = "independent",
) -> dict:
    """
    Train an XGBoost classifier with optional probability calibration.

    Args:
        impute_strategy: "native_nan" preserves NaN for XGBoost's native
            missing-value handling. "median" uses legacy median imputation.
        xgb_params: Override XGBoost hyperparameters (None = production defaults).
        calibration_method: "isotonic", "sigmoid", or "none".
        calibration_cv: "timeseries_5fold", "random_5fold", "temporal_holdout",
            "temporal_holdout_refit" (full-data booster behind the holdout
            calibrator), or "temporal_holdout_weighted" (calibrator fitted
            with time-decay weights).
        odds_noise_mode: "independent" (legacy per-column draws) or
            "antithetic" (mirror-consistent, no-vig identities preserved).

    Returns dict with:
        - model: trained model (or calibrated wrapper)
        - feature_cols: list of feature columns used
        - feature_importance: dict of feature name -> importance
        - col_medians: median array (for LogisticRegression or legacy compat)
        - impute_strategy: which strategy was used
    """
    calibration_cv = _validate_calibration_cv(calibration_cv)
    observed_training_rows = len(train_df)
    train_df, mirror_augmentation_applied = _mirror_augment_training_rows(train_df, feature_cols)
    X_train = train_df[feature_cols].values.copy()
    y_train = train_df["target"].values

    # XGBoost hyperparameters
    default_params = {
        "n_estimators": 135,
        "max_depth": 7,
        "learning_rate": 0.0124,
        "subsample": 0.659,
        "colsample_bytree": 0.706,
        "min_child_weight": 6,
        "gamma": 0.444,
        "reg_alpha": 0.00443,
        "reg_lambda": 0.00772,
        "scale_pos_weight": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    params = xgb_params if xgb_params is not None else default_params
    effective_noise_seed = (
        int(odds_noise_seed)
        if odds_noise_seed is not None
        else int(params.get("random_state", 42))
    )

    # Compute medians regardless (needed for col_medians metadata)
    col_medians = np.nanmedian(X_train, axis=0)

    if impute_strategy == "median":
        # Legacy path: median imputation + missing indicators
        indicator_cols = []
        indicator_names = []
        for i in range(X_train.shape[1]):
            mask = np.isnan(X_train[:, i])
            if mask.any():
                indicator_cols.append(mask.astype(float))
                indicator_names.append(f"{feature_cols[i]}_missing")
            X_train[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

        if indicator_cols:
            X_train = np.column_stack([X_train] + indicator_cols)
            feature_cols = list(feature_cols) + indicator_names
            logger.info(f"Added {len(indicator_cols)} missing value indicator features")
    else:
        # native_nan: XGBoost handles NaN natively — no imputation
        n_nan = np.isnan(X_train).sum()
        logger.info(f"Native NaN mode: {n_nan} NaN values preserved for XGBoost")

    # Add noise to odds features to mitigate closing odds leakage
    if str(odds_noise_mode or "independent") not in SUPPORTED_ODDS_NOISE_MODES:
        supported = ", ".join(sorted(SUPPORTED_ODDS_NOISE_MODES))
        raise ValueError(
            f"Unsupported odds_noise_mode '{odds_noise_mode}'. Supported values: {supported}"
        )
    if odds_noise_mode == "antithetic":
        X_train = _add_antithetic_odds_noise(
            X_train,
            feature_cols,
            train_df,
            noise_std=odds_noise_std,
            seed=effective_noise_seed,
        )
    else:
        X_train = _add_odds_noise(
            X_train,
            feature_cols,
            noise_std=odds_noise_std,
            seed=effective_noise_seed,
        )

    # Compute time-decay sample weights
    sample_weights = _compute_sample_weights(
        train_df,
        half_life_days=time_decay_half_life_days,
    )

    xgb = XGBClassifier(**params)

    model = xgb
    if calibrate and calibration_method != "none":
        from sklearn.base import clone as _sklearn_clone

        if calibration_cv in (
            "temporal_holdout", "temporal_holdout_refit", "temporal_holdout_weighted",
        ):
            inner_idx, cal_idx = _temporal_holdout_indices(train_df)
            if len(inner_idx) == 0 or len(cal_idx) == 0:
                raise ValueError("temporal_holdout calibration requires at least 2 training rows")

            X_inner = X_train[inner_idx]
            y_inner = y_train[inner_idx]
            X_cal = X_train[cal_idx]
            y_cal = y_train[cal_idx]
            w_inner = sample_weights[inner_idx] if sample_weights is not None else None

            xgb_inner = XGBClassifier(**params)
            xgb_inner.fit(X_inner, y_inner, sample_weight=w_inner)
            train_indices = np.array([], dtype=int)
            test_indices = np.arange(len(X_cal))
            model = CalibratedClassifierCV(
                FrozenEstimator(xgb_inner),
                cv=[(train_indices, test_indices)],
                method=calibration_method,
            )
            if calibration_cv == "temporal_holdout_weighted" and sample_weights is not None:
                # The default path silently drops time-decay weights here, so
                # 2024 fights weigh the same as 2026 in the probability
                # mapping despite the recency-weighted booster.
                model.fit(X_cal, y_cal, sample_weight=sample_weights[cal_idx])
            else:
                model.fit(X_cal, y_cal)
        elif calibration_cv == "timeseries_5fold":
            cv = _timeseries_cv_for_training(train_df, n_splits=5)
            # Pass an *unfitted* clone so CalibratedClassifierCV fits fresh
            # base estimators inside each CV fold — avoids data leakage.
            model = CalibratedClassifierCV(
                _sklearn_clone(xgb), cv=cv, method=calibration_method,
            )
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            cv = _group_cv_for_training(train_df, n_splits=5)
            model = CalibratedClassifierCV(
                _sklearn_clone(xgb), cv=cv, method=calibration_method,
            )
            model.fit(X_train, y_train, sample_weight=sample_weights)
        # Fit the raw xgb on the full training set for feature importances
        xgb.fit(X_train, y_train, sample_weight=sample_weights)
        if calibration_cv == "temporal_holdout_refit":
            # Serve the full-data booster through the holdout-fitted
            # calibrator. The holdout booster only ever sees the first ~80%
            # of rows; the refit booster recovers the newest fights (~80% of
            # the time-decay weight mass) at zero extra training cost.
            model = HoldoutCalibratedRefitModel(xgb, model)
    else:
        xgb.fit(X_train, y_train, sample_weight=sample_weights)

    # Feature importance from the raw XGBoost model
    importance = dict(zip(feature_cols, xgb.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    logger.info("Top 10 features:")
    for feat, imp in list(importance.items())[:10]:
        logger.info(f"  {feat}: {imp:.4f}")

    # Extend col_medians for indicator columns if median imputation was used
    if impute_strategy == "median":
        n_indicators = len([c for c in feature_cols if c.endswith("_missing")])
        if n_indicators > 0:
            col_medians = np.concatenate([col_medians, np.zeros(n_indicators)])

    return {
        "model": model,
        "raw_model": xgb,
        "feature_cols": feature_cols,
        "feature_importance": importance,
        "col_medians": col_medians,
        "impute_strategy": impute_strategy,
        "mirror_augmentation": mirror_augmentation_applied,
        "observed_training_rows": observed_training_rows,
        "effective_training_rows": len(train_df),
    }


def train_logistic(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    odds_noise_std: float = ODDS_NOISE_STD,
    odds_noise_seed: int | None = None,
) -> dict:
    """
    Train a Logistic Regression baseline with StandardScaler.
    LR naturally produces calibrated probabilities.
    """
    observed_training_rows = len(train_df)
    train_df, mirror_augmentation_applied = _mirror_augment_training_rows(train_df, feature_cols)
    X_train = train_df[feature_cols].values.copy()
    y_train = train_df["target"].values

    col_medians = np.nanmedian(X_train, axis=0)
    for i in range(X_train.shape[1]):
        mask = np.isnan(X_train[:, i])
        X_train[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

    # Add noise to odds features to mitigate closing odds leakage
    X_train = _add_odds_noise(
        X_train,
        feature_cols,
        noise_std=odds_noise_std,
        seed=(42 if odds_noise_seed is None else int(odds_noise_seed)),
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)

    # Feature importance via coefficients
    coefs = pipeline.named_steps["lr"].coef_[0]
    importance = dict(zip(feature_cols, np.abs(coefs)))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return {
        "model": pipeline,
        "feature_cols": feature_cols,
        "feature_importance": importance,
        "col_medians": col_medians,
        "mirror_augmentation": mirror_augmentation_applied,
        "observed_training_rows": observed_training_rows,
        "effective_training_rows": len(train_df),
    }


def train_all_models(
    features_df: pd.DataFrame,
    spec: "NamedModelTrainingSpec | None" = None,
    *,
    models_dir: Path | None = None,
    test_set_path: Path | None = None,
    training_input_evidence: dict | None = None,
    final_track_c_receipt_path: Path | None = None,
) -> dict:
    """
    Train XGBoost, Logistic Regression, and no-odds baseline models.
    Saves models to disk. Returns dict of model results.

    If a NamedModelTrainingSpec is provided, it overrides the default
    feature selection, imputation strategy, and hyperparameters.
    """
    required_evidence_spec = spec if _is_current_policy_selected_fullfit_spec(spec) else None
    if required_evidence_spec is not None and training_input_evidence is None:
        raise ValueError(
            f"{required_evidence_spec.name} requires policy-bound training_input_evidence"
        )
    final_track_c_binding = None
    if required_evidence_spec is not None:
        if final_track_c_receipt_path is None:
            raise ValueError(
                f"{required_evidence_spec.name} requires a final-policy Track-C PASS receipt"
            )
        from scripts.check_production_refit_contract import (
            validate_final_track_c_pass_receipt,
        )

        final_track_c_binding = validate_final_track_c_pass_receipt(
            Path(final_track_c_receipt_path),
            expected_policy_path=_SCHEDULED_REFIT_POLICY_PATH,
            expected_fullfit_spec=required_evidence_spec,
        )
    training_input_evidence = _validate_training_input_evidence(
        training_input_evidence,
        features_df=features_df,
        required_spec=required_evidence_spec,
    )
    if final_track_c_binding is not None:
        if (
            training_input_evidence.get("policy_sha256")
            != final_track_c_binding["policy_sha256"]
        ):
            raise ValueError(
                "training_input_evidence policy differs from final Track-C PASS"
            )
        source_fights_record = training_input_evidence.get("source_fights_csv")
        if (
            not isinstance(source_fights_record, dict)
            or Path(str(source_fights_record.get("path") or "")).resolve(
                strict=False
            )
            != final_track_c_binding["dataset_fights_path"]
            or source_fights_record.get("sha256")
            != final_track_c_binding["dataset_fights_sha256"]
        ):
            raise ValueError(
                "training_input_evidence source fights differ from final Track-C PASS"
            )
        bound_training_evidence = {
            key: value
            for key, value in training_input_evidence.items()
            if key != "receipt_sha256"
        }
        bound_training_evidence.update({
            "final_track_c_pass_receipt_path": str(
                final_track_c_binding["receipt_path"]
            ),
            "final_track_c_pass_receipt_sha256": final_track_c_binding[
                "receipt_sha256"
            ],
            "performance_confirmation_result_sha256": final_track_c_binding[
                "confirmation_result_sha256"
            ],
            "confirmed_strategy_config_sha256": final_track_c_binding[
                "strategy_config_sha256"
            ],
            "confirmation_evaluation_input_value_sha256": final_track_c_binding[
                "confirmation_evaluation_input_value_sha256"
            ],
            "feature_contract_count": final_track_c_binding[
                "feature_contract_count"
            ],
            "feature_contract_sha256": final_track_c_binding[
                "feature_contract_sha256"
            ],
            "confirmation_features_value_sha256": final_track_c_binding[
                "features_value_sha256"
            ],
            "confirmation_source_features_path": str(
                final_track_c_binding["source_features_path"]
            ),
            "confirmation_source_features_sha256": final_track_c_binding[
                "source_features_sha256"
            ],
            "confirmation_odds_source_inventory_path": str(
                final_track_c_binding["odds_source_inventory_path"]
            ),
            "confirmation_odds_source_inventory_sha256": final_track_c_binding[
                "odds_source_inventory_sha256"
            ],
            "confirmation_source_fingerprint": final_track_c_binding[
                "source_fingerprint"
            ],
            "confirmation_source_inventory_path": str(
                final_track_c_binding["source_inventory_path"]
            ),
            "confirmation_source_inventory_sha256": final_track_c_binding[
                "source_inventory_sha256"
            ],
            "confirmation_source_inventory_artifact_sha256": final_track_c_binding[
                "source_inventory_artifact_sha256"
            ],
            "confirmation_environment_path": str(
                final_track_c_binding["environment_path"]
            ),
            "confirmation_environment_artifact_sha256": final_track_c_binding[
                "environment_artifact_sha256"
            ],
            "confirmation_environment_payload_sha256": final_track_c_binding[
                "environment_payload_sha256"
            ],
            "confirmation_evaluation_protocol_sha256": final_track_c_binding[
                "evaluation_protocol_sha256"
            ],
        })
        training_input_evidence = {
            **bound_training_evidence,
            "receipt_sha256": _canonical_json_sha256(bound_training_evidence),
        }

    if spec is not None:
        # Validate the spec's feature contract
        violations = spec.validate_feature_contract()
        if violations:
            for v in violations:
                logger.error(v)
            raise ValueError(f"Training spec has {len(violations)} contract violations")

        from src.model.training_spec import (
            compute_feature_family_coverage,
            materialize_and_validate_spec_features,
        )

        if not spec.trained_at:
            spec.trained_at = datetime.now().isoformat()
        if not spec.git_hash:
            spec.git_hash = _resolve_repo_git_hash()
        features_df = materialize_and_validate_spec_features(features_df, spec)

        # Use the spec's ordered feature list after materializing every requested transform.
        feature_cols = list(spec.feature_cols)

        cutoff = spec.train_cutoff_date
        train_df, test_df, _ = prepare_train_test(
            features_df,
            cutoff_date=cutoff,
            feature_cols=feature_cols,
            start_date=spec.train_start_date or None,
            end_date=getattr(spec, "train_end_date", "") or None,
        )
        dead_train_columns = _dead_train_feature_columns(train_df, feature_cols)
        if dead_train_columns:
            raise ValueError(
                f"Training spec '{spec.name}' has dead train-time contract columns "
                f"for dataset variant '{spec.dataset_variant}': {dead_train_columns}. "
                "Refusing to train until the dataset or spec is fixed."
            )
        required_family_coverage = dict(getattr(spec, "required_feature_family_coverage_pct", {}) or {})
        if required_family_coverage:
            coverage_summary = compute_feature_family_coverage(
                train_df,
                feature_cols=feature_cols,
                family_names=list(required_family_coverage.keys()),
            )
            coverage_failures = []
            for family_name, min_pct in required_family_coverage.items():
                family_summary = coverage_summary.get(family_name)
                if family_summary is None:
                    coverage_failures.append(
                        f"{family_name}: no active feature columns found in the training frame"
                    )
                    continue
                actual_pct = float(family_summary["coverage_pct"])
                if actual_pct + 1e-9 < float(min_pct):
                    coverage_failures.append(
                        f"{family_name}: {actual_pct:.2f}% complete coverage "
                        f"({family_summary['rows_complete']}/{family_summary['rows_total']}) "
                        f"< required {float(min_pct):.2f}%"
                    )
            if coverage_failures:
                joined = "; ".join(coverage_failures)
                raise ValueError(
                    f"Training spec '{spec.name}' failed required external-family coverage gates: {joined}"
                )
        impute_strategy = spec.impute_strategy
        xgb_params = spec.xgb_params
        calibration_method = spec.calibration_method
        calibration_cv = spec.calibration_cv
        time_decay_half_life_days = spec.time_decay_half_life
        odds_noise_std = spec.odds_noise_std
        xgb_random_state = xgb_params.get("random_state", 42) if isinstance(xgb_params, dict) else 42
        odds_noise_seed = (
            int(spec.odds_noise_seed)
            if spec.odds_noise_seed is not None
            else int(xgb_random_state)
        )
        if spec.odds_noise_seed is None:
            spec.odds_noise_seed = odds_noise_seed
        logger.info(f"Training with spec '{spec.name}': {len(feature_cols)} features, "
                     f"impute={impute_strategy}, cal={calibration_method}")
    else:
        train_df, test_df, feature_cols = prepare_train_test(features_df)
        impute_strategy = "median"  # Legacy default
        xgb_params = None
        calibration_method = "isotonic"
        calibration_cv = "timeseries_5fold"
        time_decay_half_life_days = None
        odds_noise_std = ODDS_NOISE_STD
        odds_noise_seed = 42

        # SHAP feature selection — reduce to top 40 features to avoid overfitting
        try:
            from src.model.feature_selection import select_top_features
            feature_cols = select_top_features(
                train_df, feature_cols, n_keep=40, method="shap"
            )
            logger.info(f"SHAP selected {len(feature_cols)} features")
        except Exception as e:
            logger.warning(f"SHAP feature selection failed, using all features: {e}")

    # Train models
    logger.info("Training XGBoost...")
    xgb_result = train_xgboost(
        train_df, feature_cols,
        impute_strategy=impute_strategy,
        xgb_params=xgb_params,
        calibration_method=calibration_method,
        calibration_cv=calibration_cv,
        odds_noise_std=odds_noise_std,
        odds_noise_seed=odds_noise_seed,
        time_decay_half_life_days=time_decay_half_life_days,
    )

    logger.info("Training Logistic Regression...")
    # LR always uses median imputation (StandardScaler cannot handle NaN)
    lr_result = _call_train_logistic(
        train_df,
        feature_cols,
        odds_noise_std=odds_noise_std,
        odds_noise_seed=odds_noise_seed,
    )

    # Train no-odds baseline (fighter stats only — measures independent edge)
    no_odds_cols = get_feature_columns_no_odds(features_df)
    no_odds_cols = [c for c in no_odds_cols if c in train_df.columns]
    if spec is not None:
        # Constrain no-odds cols to the promoted contract minus market-derived features.
        no_odds_cols = exclude_market_derived_features(feature_cols)
    logger.info(f"Training XGBoost (no-odds baseline, {len(no_odds_cols)} features)...")
    xgb_no_odds_result = train_xgboost(
        train_df, no_odds_cols,
        impute_strategy=impute_strategy,
        xgb_params=xgb_params,
        calibration_method=calibration_method,
        calibration_cv=calibration_cv,
        odds_noise_std=odds_noise_std,
        odds_noise_seed=odds_noise_seed,
        time_decay_half_life_days=time_decay_half_life_days,
    )

    # Embed spec in all model dicts for portability (no sidecar dependency)
    if spec is not None:
        from dataclasses import asdict

        spec_dict = asdict(spec)
        xgb_result["training_spec"] = spec_dict
        lr_result["training_spec"] = spec_dict
        xgb_no_odds_result["training_spec"] = _build_no_odds_training_spec_payload(
            spec,
            no_odds_cols,
        )
    if training_input_evidence is not None:
        for result in (xgb_result, lr_result, xgb_no_odds_result):
            result["training_input_evidence"] = deepcopy(training_input_evidence)

    # Save models
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    test_set_path = Path(test_set_path) if test_set_path is not None else PROCESSED_DATA_DIR / "test_set.csv"
    models_dir.mkdir(parents=True, exist_ok=True)
    test_set_path.parent.mkdir(parents=True, exist_ok=True)

    xgb_path = models_dir / "xgboost_model.pkl"
    lr_path = models_dir / "logistic_model.pkl"
    no_odds_path = models_dir / "xgboost_no_odds_model.pkl"
    joblib.dump(xgb_result, xgb_path)
    joblib.dump(lr_result, lr_path)
    joblib.dump(xgb_no_odds_result, no_odds_path)
    logger.info(f"Saved XGBoost to {xgb_path}")
    logger.info(f"Saved Logistic Regression to {lr_path}")
    logger.info(f"Saved XGBoost (no-odds) to {no_odds_path}")

    # Save training spec alongside model if provided
    if spec is not None:
        spec_path = models_dir / f"{spec.name}_spec.json"
        spec.save(spec_path)
    else:
        spec_path = None

    # Save test set for evaluation
    test_df.to_csv(test_set_path, index=False)
    test_set_metadata = write_test_set_metadata(
        test_set_path=test_set_path,
        spec=spec,
        feature_cols=feature_cols,
        test_df=test_df,
        training_input_evidence=training_input_evidence,
    )

    return {
        "xgboost": xgb_result,
        "logistic": lr_result,
        "xgboost_no_odds": xgb_no_odds_result,
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "no_odds_feature_cols": no_odds_cols,
        "spec": spec,
        "models_dir": models_dir,
        "test_set_path": test_set_path,
        "test_set_metadata": test_set_metadata,
        "spec_path": spec_path,
        "training_input_evidence": training_input_evidence,
    }


def load_model(model_name: str = "xgboost") -> dict:
    """Load a saved model from disk by model alias or explicit artifact path.

    Rejects artifacts that do not embed the exact training feature contract.
    """
    model_ref = Path(model_name)
    if model_ref.suffix == ".pkl" or model_ref.is_absolute() or any(sep in model_name for sep in ("/", "\\")):
        path = model_ref
    else:
        path = MODELS_DIR / f"{model_name}_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    result = joblib.load(path)

    artifact_cols = result.get("feature_cols")
    spec = result.get("training_spec")
    if not isinstance(spec, dict):
        raise ValueError(
            f"Model artifact {path} is missing an embedded training_spec. "
            "Retrain the model with the current contract before using it."
        )
    if not isinstance(artifact_cols, list):
        raise ValueError(f"Model artifact {path} is missing feature_cols.")

    spec_cols = spec.get("feature_cols")
    if not isinstance(spec_cols, list):
        raise ValueError(f"Model artifact {path} has an invalid embedded training_spec feature contract.")
    if artifact_cols != spec_cols:
        if _repair_legacy_no_odds_training_spec_payload(path, result, artifact_cols, spec):
            _persist_repaired_training_spec_payload(path, result)
            result["artifact_path"] = str(path.resolve(strict=False))
            return result
        raise ValueError(
            f"Model artifact {path} failed feature-contract validation: "
            "feature_cols do not exactly match embedded training_spec.feature_cols."
        )
    result["artifact_path"] = str(path.resolve(strict=False))
    return result
