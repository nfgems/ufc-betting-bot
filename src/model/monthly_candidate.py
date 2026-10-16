"""Durable, candidate-only monthly model training orchestration.

This module deliberately has no promotion or production-manifest dependency.  A
successful run creates an immutable candidate receipt for owner review; it never
installs the candidate or writes to the live model directory.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import pandas as pd

from src.data.io_utils import write_bytes_atomically


MONTHLY_CANDIDATE_SCHEMA_VERSION = 1
CANDIDATE_SPEC_NAME = "full_live_contract_v6_durability_fullfit"
EXPECTED_PRIMARY_FEATURE_COUNT = 211
EXPECTED_LOGISTIC_FEATURE_COUNT = 211
EXPECTED_NO_ODDS_FEATURE_COUNT = 202
MODELING_START_DATE = "2022-01-01"
UNSUPPORTED_CONTRACT_START_DATE = "2027-01-01"

_REQUIRED_SOURCE_COLUMNS = {
    "event_date",
    "fighter_a",
    "fighter_b",
    "winner",
    "method",
    "target",
}
_SAFE_ATTEMPT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class MonthlyCandidateError(RuntimeError):
    """The monthly candidate lifecycle cannot safely continue."""


class CandidateTrainer(Protocol):
    def __call__(
        self,
        *,
        features_df: pd.DataFrame,
        spec: Any,
        models_dir: Path,
        test_set_path: Path,
    ) -> object: ...


class CandidateValidator(Protocol):
    def __call__(
        self,
        *,
        candidate_dir: Path,
        contract: "ResolvedMonthlyCandidateContract",
    ) -> Mapping[str, Any]: ...


class CandidateEvaluator(Protocol):
    def __call__(
        self,
        *,
        candidate_dir: Path,
        input_receipt: Mapping[str, Any],
        contract: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ResolvedMonthlyCandidateContract:
    """Resolved immutable facts plus the fresh mutable training-spec object."""

    spec: Any
    registered_spec_payload: dict[str, Any]
    primary_features: tuple[str, ...]
    no_odds_features: tuple[str, ...]
    primary_feature_sha256: str
    no_odds_feature_sha256: str
    monthly_policy: dict[str, Any] | None = None

    def receipt_payload(self) -> dict[str, Any]:
        return {
            "name": self.spec.name if self.monthly_policy else CANDIDATE_SPEC_NAME,
            "primary_feature_count": len(self.primary_features),
            "logistic_feature_count": len(self.primary_features),
            "no_odds_feature_count": len(self.no_odds_features),
            "primary_feature_sha256": self.primary_feature_sha256,
            "logistic_feature_sha256": self.primary_feature_sha256,
            "no_odds_feature_sha256": self.no_odds_feature_sha256,
            "modeling_start_date": MODELING_START_DATE,
            "unsupported_contract_start_date": (self.monthly_policy["train_cutoff_date"]
                if self.monthly_policy else UNSUPPORTED_CONTRACT_START_DATE),
            **({"monthly_policy": deepcopy(self.monthly_policy)} if self.monthly_policy else {}),
        }


def _canonical_json_bytes(payload: object, *, pretty: bool = True) -> bytes:
    if pretty:
        rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    else:
        rendered = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    return (rendered + ("\n" if pretty else "")).encode("utf-8")


def _write_json_atomically(payload: object, path: Path) -> Path:
    return write_bytes_atomically(_canonical_json_bytes(payload), path)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MonthlyCandidateError(f"Missing {label}: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MonthlyCandidateError(f"Invalid {label} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MonthlyCandidateError(f"{label} must be a JSON object: {path}")
    return payload


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MonthlyCandidateError(f"Could not hash required file {path}: {exc}") from exc
    return digest.hexdigest()


def _feature_hash(features: tuple[str, ...] | list[str]) -> str:
    content = json.dumps(list(features), separators=(",", ":"), ensure_ascii=True)
    return _sha256_bytes(content.encode("utf-8"))


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise MonthlyCandidateError("Candidate timestamps must be timezone-aware.")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _clock_utc(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    _utc_iso(value)
    return value.astimezone(timezone.utc)


def _parse_month(value: str, *, label: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{4})-(\d{2})", str(value or "").strip())
    if match is None:
        raise MonthlyCandidateError(f"{label} must use YYYY-MM format.")
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        raise MonthlyCandidateError(f"{label} has an invalid month: {value}")
    return year, month


def _resolved(path: Path | str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _assert_isolated_paths(
    *, candidate_root: Path, state_path: Path, live_models_dir: Path
) -> None:
    if _paths_overlap(candidate_root, live_models_dir):
        raise MonthlyCandidateError(
            "candidate_root and live_models_dir must be completely disjoint."
        )
    if _paths_overlap(state_path, live_models_dir):
        raise MonthlyCandidateError(
            "state_path and live_models_dir must be completely disjoint."
        )


def _directory_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "files": []}
    if not path.is_dir():
        raise MonthlyCandidateError(f"Live model path is not a directory: {path}")
    files = []
    for item in sorted(path.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if item.is_file():
            files.append(
                {
                    "path": item.relative_to(path).as_posix(),
                    "size_bytes": item.stat().st_size,
                    "sha256": _sha256_file(item),
                }
            )
    return {"exists": True, "files": files}


def _assert_live_unchanged(
    live_models_dir: Path, before: Mapping[str, Any]
) -> dict[str, Any]:
    after = _directory_fingerprint(live_models_dir)
    if after != before:
        raise MonthlyCandidateError(
            "The live model directory changed during a candidate-only run; "
            "completion is forbidden and operator investigation is required."
        )
    return after


@contextmanager
def _candidate_writer_lock(candidate_root: Path):
    candidate_root.mkdir(parents=True, exist_ok=True)
    lock_path = candidate_root / ".monthly-candidate.lock"
    handle = lock_path.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise MonthlyCandidateError(
                    "monthly_candidate_already_running"
                ) from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise MonthlyCandidateError(
                    "monthly_candidate_already_running"
                ) from exc
        acquired = True
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def resolve_monthly_candidate_contract(policy: Any = None) -> ResolvedMonthlyCandidateContract:
    """Resolve and strictly pin the approved v7 211/211/202 contract."""

    from src.features.build_features import exclude_market_derived_features
    from src.model.training_spec import resolve_named_training_spec

    if policy is not None:
        from src.model.monthly_policy import load_monthly_policy
        # Derive from retained original bytes, not mutable caller spec/binding objects.
        selected = load_monthly_policy(policy.pin, body=policy.body, context=policy.context)
        if selected.spec is None or policy.spec is None:
            raise MonthlyCandidateError("Candidate requires a policy bound to the original saved completion.")
        if selected.binding != policy.binding or asdict(selected.spec) != asdict(policy.spec):
            raise MonthlyCandidateError("Saved monthly policy object conflicts with original bytes.")
        spec = selected.spec
        primary = tuple(spec.feature_cols)
        no_odds = tuple(exclude_market_derived_features(list(primary)))
        if (not primary or len(set(primary)) != len(primary)
            or not no_odds or len(set(no_odds)) != len(no_odds)
            or not set(no_odds).issubset(primary)):
            raise MonthlyCandidateError("Resolved supported monthly feature contract is invalid.")
        return ResolvedMonthlyCandidateContract(
            spec=spec, registered_spec_payload=deepcopy(asdict(spec)), primary_features=primary,
            no_odds_features=no_odds, primary_feature_sha256=_feature_hash(primary),
            no_odds_feature_sha256=_feature_hash(no_odds), monthly_policy=deepcopy(selected.binding))

    spec = resolve_named_training_spec(CANDIDATE_SPEC_NAME)
    primary = tuple(spec.feature_cols)
    no_odds = tuple(exclude_market_derived_features(list(primary)))
    if spec.name != CANDIDATE_SPEC_NAME:
        raise MonthlyCandidateError(
            f"Resolved unexpected candidate contract {spec.name!r}."
        )
    if len(primary) != EXPECTED_PRIMARY_FEATURE_COUNT or len(set(primary)) != len(primary):
        raise MonthlyCandidateError(
            "Approved primary/logistic contract is not exactly 211 unique features."
        )
    if len(no_odds) != EXPECTED_NO_ODDS_FEATURE_COUNT or len(set(no_odds)) != len(no_odds):
        raise MonthlyCandidateError(
            "Approved no-odds contract is not exactly 202 unique features."
        )
    if not set(no_odds).issubset(primary):
        raise MonthlyCandidateError("No-odds contract is not a primary-contract subset.")
    if str(spec.train_cutoff_date) != UNSUPPORTED_CONTRACT_START_DATE:
        raise MonthlyCandidateError(
            "Approved contract cutoff changed; a future monthly contract is required."
        )
    registered_payload = deepcopy(asdict(spec))
    return ResolvedMonthlyCandidateContract(
        spec=spec,
        registered_spec_payload=registered_payload,
        primary_features=primary,
        no_odds_features=no_odds,
        primary_feature_sha256=_feature_hash(primary),
        no_odds_feature_sha256=_feature_hash(no_odds),
    )


def _normalized_name(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.casefold()


def _date_bounds(dates: pd.Series) -> tuple[str, str]:
    non_null = dates.dropna()
    if non_null.empty:
        return "", ""
    return non_null.min().date().isoformat(), non_null.max().date().isoformat()


def _incoming_result_policy_exclusions(
    frame: pd.DataFrame,
    *,
    numeric_target: pd.Series,
) -> pd.Series:
    """Validate future-only result markers and identify non-binary new results."""

    exclusions = pd.Series(False, index=frame.index)
    if "result_policy_version" not in frame.columns:
        return exclusions

    from src.data.fight_results import (
        RESULT_CLASS_DQ,
        RESULT_CLASS_DRAW,
        RESULT_CLASS_NO_CONTEST,
        RESULT_CLASS_NORMAL,
        RESULT_POLICY_VERSION,
        parse_policy_boolean,
    )

    marker_text = frame["result_policy_version"].fillna("").astype(str).str.strip()
    marked = marker_text.ne("") & marker_text.str.casefold().ne("nan")
    if not marked.any():
        return exclusions
    missing_policy_columns = sorted(
        {"result_class", "binary_target_eligible"} - set(frame.columns)
    )
    if missing_policy_columns:
        raise MonthlyCandidateError(
            "Policy-marked incoming rows are missing result columns: "
            f"{missing_policy_columns}"
        )

    for index in frame.index[marked]:
        marker = marker_text.loc[index]
        raw_result_class = frame.at[index, "result_class"]
        result_class = (
            ""
            if pd.isna(raw_result_class)
            else str(raw_result_class).strip().casefold()
        )
        try:
            eligible = parse_policy_boolean(frame.at[index, "binary_target_eligible"])
        except (TypeError, ValueError) as exc:
            raise MonthlyCandidateError(
                f"Policy-marked incoming row {index!r} has invalid eligibility: {exc}"
            ) from exc
        raw_winner = frame.at[index, "winner"]
        winner = "" if pd.isna(raw_winner) else str(raw_winner).strip()
        if winner.casefold() in {"nan", "none", "<na>"}:
            winner = ""
        target_missing = pd.isna(numeric_target.loc[index])
        if marker != RESULT_POLICY_VERSION:
            raise MonthlyCandidateError(
                f"Policy-marked incoming row {index!r} has unsupported marker {marker!r}."
            )
        if result_class == RESULT_CLASS_NORMAL:
            if eligible is not True or target_missing:
                raise MonthlyCandidateError(
                    f"Policy-marked normal result row {index!r} lacks a binary target."
                )
            continue
        if result_class == RESULT_CLASS_DQ:
            if eligible is not False or not target_missing or not winner:
                raise MonthlyCandidateError(
                    f"Policy-marked DQ row {index!r} violates future-only semantics."
                )
            exclusions.loc[index] = True
            continue
        if result_class in {RESULT_CLASS_DRAW, RESULT_CLASS_NO_CONTEST}:
            if eligible is not False or not target_missing or winner:
                raise MonthlyCandidateError(
                    f"Policy-marked {result_class} row {index!r} violates "
                    "future-only semantics."
                )
            exclusions.loc[index] = True
            continue
        raise MonthlyCandidateError(
            f"Policy-marked incoming row {index!r} has unknown or contradictory "
            f"result class {result_class!r}."
        )
    return exclusions


def build_monthly_modeling_view(
    source: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str],
    policy: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Derive the pinned 2022+ binary-winner universe from accepted features."""

    missing = sorted((_REQUIRED_SOURCE_COLUMNS | set(feature_columns)) - set(source.columns))
    if missing:
        raise MonthlyCandidateError(
            f"Accepted feature source is missing required columns: {missing}"
        )
    if "_accepted_source_row_number" in source.columns:
        raise MonthlyCandidateError(
            "Accepted feature source uses reserved column _accepted_source_row_number."
        )

    frame = source.copy()
    event_dates = pd.to_datetime(frame["event_date"], errors="coerce", utc=True)
    source_min, source_max = _date_bounds(event_dates)
    if policy is not None:
        selected_contract = resolve_monthly_candidate_contract(policy)
        if tuple(feature_columns) != selected_contract.primary_features:
            raise MonthlyCandidateError("Monthly modeling feature order conflicts with accepted policy.")
        if (event_dates.ne(event_dates.dt.normalize()) & event_dates.notna()).any():
            raise MonthlyCandidateError("Monthly policy requires date-only UTC event dates.")
        completed = pd.Timestamp(selected_contract.monthly_policy["input_completion_cutoff_utc"])
        if event_dates.gt(completed).fillna(False).any():
            raise MonthlyCandidateError("Monthly modeling rows exceed original completion cutoff.")
    unsupported_start = pd.Timestamp(
        policy.binding["train_cutoff_date"] if policy else UNSUPPORTED_CONTRACT_START_DATE, tz="UTC")
    if event_dates.ge(unsupported_start).fillna(False).any():
        raise MonthlyCandidateError(
            "Accepted feature source contains rows outside the approved monthly training contract."
        )

    remaining = pd.Series(True, index=frame.index)
    exclusions: dict[str, int] = {}

    invalid_date = event_dates.isna()
    if invalid_date.any():
        raise MonthlyCandidateError(
            "Accepted feature source contains invalid or missing event dates."
        )
    exclusions["invalid_event_date"] = 0

    pre_start = event_dates.lt(pd.Timestamp(MODELING_START_DATE, tz="UTC")).fillna(False)
    exclusions["before_2022"] = int((remaining & pre_start).sum())
    remaining &= ~pre_start

    numeric_target = pd.to_numeric(frame["target"], errors="coerce")
    incoming_unusual = _incoming_result_policy_exclusions(
        frame,
        numeric_target=numeric_target,
    )
    exclusions["incoming_unusual_result"] = int((remaining & incoming_unusual).sum())
    remaining &= ~incoming_unusual

    non_binary = ~numeric_target.isin((0, 1))
    exclusions["non_binary_target"] = int((remaining & non_binary).sum())
    remaining &= ~non_binary

    fighter_a = _normalized_name(frame["fighter_a"])
    fighter_b = _normalized_name(frame["fighter_b"])
    winner = _normalized_name(frame["winner"])
    expected_winner = fighter_b.copy()
    expected_winner.loc[numeric_target.eq(1)] = fighter_a.loc[numeric_target.eq(1)]
    invalid_identity = (
        fighter_a.eq("")
        | fighter_b.eq("")
        | fighter_a.eq(fighter_b)
        | winner.eq("")
        | winner.ne(expected_winner)
    )
    invalid_selected_identity = remaining & invalid_identity
    if invalid_selected_identity.any():
        examples = [str(index) for index in frame.index[invalid_selected_identity][:5]]
        raise MonthlyCandidateError(
            "Accepted binary-result rows have invalid fighter/winner identity at "
            f"source rows: {examples}"
        )
    exclusions["winner_or_identity_mismatch"] = 0

    source_row_numbers = pd.Series(range(2, len(frame) + 2), index=frame.index)
    selected = frame.loc[remaining].copy()
    if selected.empty:
        raise MonthlyCandidateError("The accepted feature source produced no eligible rows.")
    selected.insert(
        0,
        "_accepted_source_row_number",
        source_row_numbers.loc[remaining].to_numpy(),
    )
    selected["event_date"] = event_dates.loc[remaining].dt.date.astype(str)
    selected["target"] = numeric_target.loc[remaining].astype(int)

    normalized_a = _normalized_name(selected["fighter_a"])
    normalized_b = _normalized_name(selected["fighter_b"])
    identity_keys = []
    for date_value, first, second in zip(
        selected["event_date"], normalized_a, normalized_b
    ):
        ordered = sorted((first, second))
        identity_keys.append(f"{date_value}|{ordered[0]}|{ordered[1]}")
    duplicate_mask = pd.Series(identity_keys, index=selected.index).duplicated(keep=False)
    if duplicate_mask.any():
        examples = sorted(set(pd.Series(identity_keys, index=selected.index)[duplicate_mask]))[:5]
        raise MonthlyCandidateError(
            "Accepted feature source contains duplicate fight identities in the "
            f"monthly universe: {examples}"
        )

    selected_dates = pd.to_datetime(selected["event_date"], errors="raise", utc=True)
    derived_min, derived_max = _date_bounds(selected_dates)
    summary = {
        "source_row_count": int(len(frame)),
        "source_min_event_date": source_min,
        "source_max_event_date": source_max,
        "derived_row_count": int(len(selected)),
        "derived_min_event_date": derived_min,
        "derived_max_event_date": derived_max,
        "excluded_row_counts": exclusions,
        "modeling_start_date": MODELING_START_DATE,
        "unsupported_contract_start_date": (policy.binding["train_cutoff_date"]
            if policy else UNSUPPORTED_CONTRACT_START_DATE),
    }
    return selected.reset_index(drop=True), summary


def _read_source_frame(content: bytes) -> pd.DataFrame:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MonthlyCandidateError("Accepted feature source is not UTF-8 CSV.") from exc
    try:
        header = next(csv.reader(io.StringIO(text)))
    except (StopIteration, csv.Error) as exc:
        raise MonthlyCandidateError("Accepted feature source has no valid CSV header.") from exc
    duplicates = sorted({column for column in header if header.count(column) > 1})
    if duplicates:
        raise MonthlyCandidateError(
            f"Accepted feature source has duplicate CSV headers: {duplicates}"
        )
    try:
        return pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as exc:  # pandas exposes several parser exception classes
        raise MonthlyCandidateError(f"Could not parse accepted feature source: {exc}") from exc


def _dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def _verify_input_receipt(
    input_dir: Path,
    *,
    contract: ResolvedMonthlyCandidateContract,
    accepted_generation_id: str | None = None,
    weekly_update_completed_at: datetime | None = None,
    monthly_input: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = _load_json_object(input_dir / "input_receipt.json", label="input receipt")
    if type(receipt.get("schema_version")) is not int or receipt["schema_version"] != MONTHLY_CANDIDATE_SCHEMA_VERSION:
        raise MonthlyCandidateError("Pinned monthly input receipt has an unknown schema.")
    if accepted_generation_id is not None and receipt.get("accepted_generation_id") != accepted_generation_id:
        raise MonthlyCandidateError("Pinned monthly generation differs from supplied selection.")
    if weekly_update_completed_at is not None and receipt.get("weekly_update_completed_at") != _utc_iso(weekly_update_completed_at):
        raise MonthlyCandidateError("Pinned monthly cutoff differs from supplied selection.")
    if monthly_input is not None and receipt.get("monthly_input") != monthly_input:
        raise MonthlyCandidateError("Pinned monthly outer selection differs from supplied claim.")
    if receipt.get("contract") != contract.receipt_payload():
        raise MonthlyCandidateError("Pinned monthly input contract no longer matches source.")
    for name, prefix in (
        ("accepted_features.csv", "source"),
        ("modeling_dataset.csv", "derived"),
    ):
        path = input_dir / name
        if not path.is_file():
            raise MonthlyCandidateError(f"Pinned monthly input artifact is missing: {path}")
        if _sha256_file(path) != receipt.get(f"{prefix}_sha256"):
            raise MonthlyCandidateError(f"Pinned monthly input artifact changed: {path}")
        if path.stat().st_size != receipt.get(f"{prefix}_size_bytes"):
            raise MonthlyCandidateError(f"Pinned monthly input artifact size changed: {path}")
    if monthly_input is not None:
        from src.model.monthly_input import load_monthly_input
        selected = load_monthly_input(dict(monthly_input), context=dict(monthly_input["context"]))
        if ((input_dir / "accepted_features.csv").read_bytes() != selected.selection.artifact_bytes
            or (input_dir / "modeling_dataset.csv").read_bytes() != _dataframe_csv_bytes(selected.modeling_dataframe)):
            raise MonthlyCandidateError("Candidate-local input bytes conflict with the original saved monthly selection.")
    return receipt


def _prepare_or_load_input(
    *,
    month_dir: Path,
    accepted_features_path: Path,
    accepted_generation_id: str,
    weekly_update_completed_at: datetime,
    contract: ResolvedMonthlyCandidateContract,
    created_at: str,
    loaded_monthly_input: Any = None,
) -> dict[str, Any]:
    input_dir = month_dir / "input"
    if input_dir.exists():
        if not input_dir.is_dir():
            raise MonthlyCandidateError(f"Pinned input path is not a directory: {input_dir}")
        return _verify_input_receipt(input_dir, contract=contract,
            accepted_generation_id=accepted_generation_id,
            weekly_update_completed_at=weekly_update_completed_at,
            monthly_input=loaded_monthly_input.transport if loaded_monthly_input else None)

    if loaded_monthly_input is not None:
        from src.model.training_preflight import resolved_spec_identity
        if loaded_monthly_input.transport["training_spec"] != {
            "name": contract.spec.name, "sha256": resolved_spec_identity(contract.spec)
        }:
            raise MonthlyCandidateError("Saved named recipe requires a separately approved candidate implementation.")
        source_content = loaded_monthly_input.selection.artifact_bytes
        modeling_view = loaded_monthly_input.modeling_dataframe.copy(deep=True)
        view_summary = dict(loaded_monthly_input.receipt["report"]["view"])
    else:
        try:
            source_content = accepted_features_path.read_bytes()
        except OSError as exc:
            raise MonthlyCandidateError(
                f"Could not read accepted feature source {accepted_features_path}: {exc}"
            ) from exc
        source = _read_source_frame(source_content)
        modeling_view, view_summary = build_monthly_modeling_view(
            source, feature_columns=contract.primary_features
        )
    derived_content = _dataframe_csv_bytes(modeling_view)
    receipt = {
        "schema_version": MONTHLY_CANDIDATE_SCHEMA_VERSION,
        "created_at": created_at,
        "accepted_generation_id": accepted_generation_id,
        "weekly_update_completed_at": _utc_iso(weekly_update_completed_at),
        "accepted_feature_source_path": str(accepted_features_path),
        "source_sha256": _sha256_bytes(source_content),
        "source_size_bytes": len(source_content),
        "derived_sha256": _sha256_bytes(derived_content),
        "derived_size_bytes": len(derived_content),
        "contract": contract.receipt_payload(),
        **view_summary,
        **({"monthly_input": loaded_monthly_input.transport,
            "accepted_update_report": deepcopy(loaded_monthly_input.publication["update_report"])}
           if loaded_monthly_input else {}),
    }

    month_dir.mkdir(parents=True, exist_ok=True)
    staging = month_dir / f".input-staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        write_bytes_atomically(source_content, staging / "accepted_features.csv")
        write_bytes_atomically(derived_content, staging / "modeling_dataset.csv")
        _write_json_atomically(receipt, staging / "input_receipt.json")
        staging.replace(input_dir)
    except Exception:
        # Preserve every failed or interrupted staging byte and consumed ID.
        raise
    return _verify_input_receipt(input_dir, contract=contract)


def _new_state(promotion_month: str, *, contract: ResolvedMonthlyCandidateContract | None = None) -> dict[str, Any]:
    if contract is not None and contract.monthly_policy is not None:
        return {"schema_version": 2, "kind": "monthly_policy_candidate_state_v1",
                "adoptions": {}, "months": {}}
    return {
        "schema_version": MONTHLY_CANDIDATE_SCHEMA_VERSION,
        "contract_name": CANDIDATE_SPEC_NAME,
        "v7_promotion_month": promotion_month,
        "months": {},
    }


def _load_state(state_path: Path, *, promotion_month: str,
                contract: ResolvedMonthlyCandidateContract | None = None) -> dict[str, Any]:
    if not state_path.exists():
        return _new_state(promotion_month, contract=contract)
    state = _load_json_object(state_path, label="monthly candidate state")
    if contract is not None and contract.monthly_policy is not None:
        if (type(state.get("schema_version")) is not int or state["schema_version"] != 2
            or state.get("kind") != "monthly_policy_candidate_state_v1"
            or not isinstance(state.get("months"), dict) or not isinstance(state.get("adoptions"), dict)):
            raise MonthlyCandidateError("Successor monthly policy state conflicts; legacy state is not migrated.")
        return state
    if type(state.get("schema_version")) is not int or state["schema_version"] != MONTHLY_CANDIDATE_SCHEMA_VERSION:
        raise MonthlyCandidateError("Monthly candidate state has an unknown schema.")
    if state.get("contract_name") != CANDIDATE_SPEC_NAME:
        raise MonthlyCandidateError("Monthly candidate state contract changed unexpectedly.")
    if state.get("v7_promotion_month") != promotion_month:
        raise MonthlyCandidateError(
            "Configured v7 promotion month differs from durable candidate state."
        )
    if not isinstance(state.get("months"), dict):
        raise MonthlyCandidateError("Monthly candidate state months must be an object.")
    return state


def _validate_policy_state(state: Mapping[str, Any], *, month: str,
                           contract: ResolvedMonthlyCandidateContract) -> dict[str, Any] | None:
    if contract.monthly_policy is None:
        return None
    binding = contract.monthly_policy
    adoption = {key: binding[key] for key in
                ("adoption_id", "model_id", "promotion_month_utc", "named_training_spec", "adapter")}
    previous_adoption = state["adoptions"].get(binding["adoption_id"])
    if previous_adoption is not None and previous_adoption != adoption:
        raise MonthlyCandidateError("Durable monthly adoption identity conflicts.")
    previous_month = state["months"].get(month)
    if previous_month is not None and (not isinstance(previous_month, dict)
                                     or previous_month.get("monthly_policy") != binding):
        raise MonthlyCandidateError("Durable monthly policy selection conflicts.")
    return adoption


def _json_mapping(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MonthlyCandidateError(f"{label} must return a mapping.")
    try:
        encoded = json.dumps(dict(value), sort_keys=True, ensure_ascii=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise MonthlyCandidateError(f"{label} result must be JSON serializable: {exc}") from exc
    if not isinstance(decoded, dict):
        raise MonthlyCandidateError(f"{label} must return a JSON object.")
    if decoded.get("passed") is not True:
        raise MonthlyCandidateError(f"{label} did not return explicit passed=true.")
    from src.model.monthly_evidence import MonthlyEvidenceRefusal, validate_correctness_mapping
    try:
        validate_correctness_mapping(decoded)
    except MonthlyEvidenceRefusal as exc:
        raise MonthlyCandidateError(f"{label}: {exc}") from exc
    return decoded


def _effective_registered_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(dict(payload))
    normalized.pop("git_hash", None)
    normalized.pop("trained_at", None)
    if normalized.get("odds_noise_seed") is None:
        params = normalized.get("xgb_params")
        random_state = params.get("random_state", 42) if isinstance(params, dict) else 42
        normalized["odds_noise_seed"] = int(random_state)
    return normalized


def _required_artifact_paths(candidate_dir: Path, contract: ResolvedMonthlyCandidateContract | None = None) -> dict[str, Path]:
    models_dir = candidate_dir / "models"
    processed_dir = candidate_dir / "processed"
    return {
        "primary_model": models_dir / "xgboost_model.pkl",
        "logistic_model": models_dir / "logistic_model.pkl",
        "no_odds_model": models_dir / "xgboost_no_odds_model.pkl",
        "saved_training_spec": models_dir / f"{contract.spec.name if contract else CANDIDATE_SPEC_NAME}_spec.json",
        "modeling_dataset": processed_dir / "modeling_dataset.csv",
        "test_set": processed_dir / "test_set.csv",
        "test_set_metadata": processed_dir / "test_set.csv.metadata.json",
    }


def _require_candidate_artifacts(candidate_dir: Path, contract: ResolvedMonthlyCandidateContract | None = None) -> None:
    paths = _required_artifact_paths(candidate_dir, contract)
    missing = sorted(label for label, path in paths.items() if not path.is_file())
    if missing:
        raise MonthlyCandidateError(f"Candidate artifacts are missing: {missing}")


def _validate_model_payload(
    result: Mapping[str, Any],
    *,
    label: str,
    feature_count: int,
    require_impute_strategy: bool,
) -> None:
    model = result.get("model")
    if model is None or not callable(getattr(model, "predict_proba", None)):
        raise MonthlyCandidateError(
            f"Monthly {label} artifact has no usable probability model."
        )
    medians = result.get("col_medians")
    try:
        median_count = len(medians)
    except TypeError as exc:
        raise MonthlyCandidateError(
            f"Monthly {label} artifact has no valid imputation vector."
        ) from exc
    if median_count != feature_count:
        raise MonthlyCandidateError(
            f"Monthly {label} imputation vector is not {feature_count} values."
        )
    if require_impute_strategy and result.get("impute_strategy") not in {
        "native_nan",
        "median",
    }:
        raise MonthlyCandidateError(
            f"Monthly {label} artifact has no valid imputation strategy."
        )


def validate_monthly_candidate_artifacts(
    *,
    candidate_dir: Path,
    contract: ResolvedMonthlyCandidateContract,
) -> dict[str, Any]:
    """Load candidate artifacts through production's read-only spec loader."""

    from src.model.production_bundle import (
        _embedded_training_spec_payload,
        _expected_no_odds_spec_payload,
    )

    candidate_dir = _resolved(candidate_dir)
    paths = _required_artifact_paths(candidate_dir, contract)
    _require_candidate_artifacts(candidate_dir, contract)

    primary_spec, primary_result = _embedded_training_spec_payload(
        paths["primary_model"], label="monthly primary model"
    )
    logistic_spec, logistic_result = _embedded_training_spec_payload(
        paths["logistic_model"], label="monthly logistic model"
    )
    no_odds_spec, no_odds_result = _embedded_training_spec_payload(
        paths["no_odds_model"], label="monthly no-odds model"
    )
    _validate_model_payload(
        primary_result,
        label="primary model",
        feature_count=len(contract.primary_features),
        require_impute_strategy=True,
    )
    _validate_model_payload(
        logistic_result,
        label="logistic model",
        feature_count=len(contract.primary_features),
        require_impute_strategy=False,
    )
    _validate_model_payload(
        no_odds_result,
        label="no-odds model",
        feature_count=len(contract.no_odds_features),
        require_impute_strategy=True,
    )
    if logistic_spec != primary_spec:
        raise MonthlyCandidateError(
            "Monthly logistic model does not exactly match the primary training spec."
        )
    if _expected_no_odds_spec_payload(primary_spec) != no_odds_spec:
        raise MonthlyCandidateError(
            "Monthly no-odds model differs beyond the approved no-odds transform."
        )
    if primary_spec.get("name") != contract.spec.name:
        raise MonthlyCandidateError("Monthly primary model embeds the wrong contract name.")
    if tuple(primary_result.get("feature_cols", ())) != contract.primary_features:
        raise MonthlyCandidateError("Monthly primary model differs from the exact selected primary contract.")
    if tuple(logistic_result.get("feature_cols", ())) != contract.primary_features:
        raise MonthlyCandidateError("Monthly logistic model differs from the exact selected primary contract.")
    if tuple(no_odds_result.get("feature_cols", ())) != contract.no_odds_features:
        raise MonthlyCandidateError("Monthly no-odds model differs from the exact selected no-odds contract.")
    if set(primary_spec) != set(contract.registered_spec_payload):
        raise MonthlyCandidateError("Monthly primary training spec is incomplete or extended.")
    if _effective_registered_payload(primary_spec) != _effective_registered_payload(
        contract.registered_spec_payload
    ):
        raise MonthlyCandidateError(
            "Monthly primary training spec does not match the registered approved contract."
        )

    saved_spec = _load_json_object(paths["saved_training_spec"], label="saved training spec")
    if saved_spec != primary_spec:
        raise MonthlyCandidateError(
            "Saved monthly training spec does not exactly match the embedded primary spec."
        )
    metadata = _load_json_object(paths["test_set_metadata"], label="test-set metadata")
    if metadata.get("training_spec") != primary_spec:
        raise MonthlyCandidateError(
            "Monthly test-set metadata does not match the embedded primary spec."
        )
    if metadata.get("spec_name") != contract.spec.name:
        raise MonthlyCandidateError("Monthly test-set metadata names the wrong contract.")
    if metadata.get("feature_count") != len(contract.primary_features):
        raise MonthlyCandidateError("Monthly test-set metadata does not match the selected feature count.")
    if metadata.get("feature_hash") != contract.primary_feature_sha256:
        raise MonthlyCandidateError("Monthly test-set feature hash is invalid.")
    if metadata.get("test_set_sha256") != _sha256_file(paths["test_set"]):
        raise MonthlyCandidateError("Monthly test-set content hash is invalid.")

    artifact_inventory = _inventory(candidate_dir, excluded={"candidate_receipt.json"})
    return {
        "passed": True,
        "loader": "src.model.production_bundle._embedded_training_spec_payload",
        "contract": contract.receipt_payload(),
        "artifact_inventory": artifact_inventory,
    }


def _candidate_evaluation_model_hashes(candidate_dir: Path, *, contract: Mapping[str, Any]) -> dict[str, str]:
    name = contract.get("name")
    if not isinstance(name, str) or _SAFE_ATTEMPT_ID.fullmatch(name) is None:
        raise MonthlyCandidateError("Evaluator artifact spec name is unsafe.")
    models = candidate_dir / "models"
    paths = {"primary": models / "xgboost_model.pkl", "logistic": models / "logistic_model.pkl",
             "no_odds": models / "xgboost_no_odds_model.pkl", "spec": models / f"{name}_spec.json"}
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise MonthlyCandidateError("Evaluator candidate artifact identities are missing or linked.")
    return {key: _sha256_file(path) for key, path in paths.items()}


def _evaluator_request(candidate_dir: Path, *, input_receipt: Mapping[str, Any],
                       contract: Mapping[str, Any], producer_inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from src.model.monthly_evidence import build_evaluator_request
    return build_evaluator_request(candidate_dir=str(candidate_dir.resolve(strict=False)),
        input_receipt=dict(input_receipt), contract=dict(contract),
        model_hashes=_candidate_evaluation_model_hashes(candidate_dir, contract=contract),
        accepted_update_report=input_receipt.get("accepted_update_report"), producer_inputs=producer_inputs)


def validate_monthly_evaluator_result(result: Any, *, request: Mapping[str, Any]) -> dict[str, Any]:
    """Direct candidate and runtime use the same pure semantic decision."""
    from src.model.monthly_evidence import MonthlyEvidenceRefusal, validate_evaluator_report
    try:
        return validate_evaluator_report(result, request=request)
    except MonthlyEvidenceRefusal as exc:
        raise MonthlyCandidateError(str(exc)) from exc


def _validate_completed_evaluation(receipt: Mapping[str, Any], *, candidate_dir: Path,
                                   input_receipt: Mapping[str, Any],
                                   contract: ResolvedMonthlyCandidateContract) -> None:
    request = _evaluator_request(candidate_dir, input_receipt=input_receipt,
                                 contract=contract.receipt_payload(), producer_inputs=receipt.get("producer_inputs"))
    if receipt.get("evaluation_request") != request:
        raise MonthlyCandidateError("Completed candidate evaluator request conflict.")
    for field in ("attempt_validation", "final_validation"):
        report = _json_mapping(receipt.get(field), label=f"completed {field}")
        if (report.get("contract") != contract.receipt_payload()
            or report.get("loader") != "src.model.production_bundle._embedded_training_spec_payload"
            or not isinstance(report.get("artifact_inventory"), list)):
            raise MonthlyCandidateError("Completed candidate validation contract is incomplete.")
    validate_monthly_evaluator_result(receipt.get("evaluation"), request=request)
    if receipt.get("producer_inputs") is not None:
        from src.model.monthly_evaluator_inputs import load_evaluator_inputs
        load_evaluator_inputs(receipt["producer_inputs"], request=request)
        from src.model.monthly_evaluator import verify_saved_report
        if verify_saved_report(receipt["producer_output_root"], request=request) != receipt["evaluation"]:
            raise MonthlyCandidateError("Completed evaluator report differs from the original candidate receipt.")


def _validated_artifact_report(value: Any, *, contract: ResolvedMonthlyCandidateContract,
                               candidate_dir: Path) -> dict[str, Any]:
    report = _json_mapping(value, label="candidate validator")
    if (set(report) != {"passed", "loader", "contract", "artifact_inventory"}
        or report["loader"] != "src.model.production_bundle._embedded_training_spec_payload"
        or report["contract"] != contract.receipt_payload()
        or report["artifact_inventory"] != _inventory(candidate_dir, excluded={"candidate_receipt.json"})):
        raise MonthlyCandidateError("Candidate validation report is incomplete or conflicts with artifacts.")
    return report


def _default_trainer(
    *,
    features_df: pd.DataFrame,
    spec: Any,
    models_dir: Path,
    test_set_path: Path,
) -> object:
    from src.model.train import train_all_models

    return train_all_models(
        features_df,
        spec=spec,
        models_dir=models_dir,
        test_set_path=test_set_path,
    )


def _inventory(root: Path, *, excluded: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = excluded or set()
    inventory = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        inventory.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return inventory


def _verify_receipt_inventory(final_dir: Path, receipt: Mapping[str, Any]) -> None:
    expected = receipt.get("artifact_inventory")
    if not isinstance(expected, list):
        raise MonthlyCandidateError("Candidate receipt has no artifact inventory.")
    actual = _inventory(final_dir, excluded={"candidate_receipt.json"})
    if actual != expected:
        raise MonthlyCandidateError("Completed monthly candidate artifacts changed.")


def _completion_receipt(month_dir: Path, *, repair_marker: bool = True) -> dict[str, Any] | None:
    marker_path = month_dir / "completed.json"
    receipt_path = month_dir / "final" / "candidate_receipt.json"
    if not marker_path.exists() and not receipt_path.exists():
        return None
    if not receipt_path.is_file():
        raise MonthlyCandidateError("Monthly completion marker has no candidate receipt.")
    receipt = _load_json_object(receipt_path, label="candidate completion receipt")
    if type(receipt.get("schema_version")) is not int or receipt["schema_version"] != MONTHLY_CANDIDATE_SCHEMA_VERSION:
        raise MonthlyCandidateError("Candidate completion receipt schema is invalid.")
    if receipt.get("status") != "complete_awaiting_owner_review":
        raise MonthlyCandidateError("Candidate completion receipt has an invalid status.")
    _verify_receipt_inventory(month_dir / "final", receipt)
    receipt_hash = _sha256_file(receipt_path)
    if marker_path.exists():
        marker = _load_json_object(marker_path, label="candidate completion marker")
        if type(marker.get("schema_version")) is not int:
            raise MonthlyCandidateError("Candidate completion marker schema is invalid.")
        if marker != {
            "schema_version": MONTHLY_CANDIDATE_SCHEMA_VERSION,
            "candidate_receipt": "final/candidate_receipt.json",
            "candidate_receipt_sha256": receipt_hash,
        }:
            raise MonthlyCandidateError("Candidate completion marker is invalid.")
    elif repair_marker:
        _write_json_atomically(
            {
                "schema_version": MONTHLY_CANDIDATE_SCHEMA_VERSION,
                "candidate_receipt": "final/candidate_receipt.json",
                "candidate_receipt_sha256": receipt_hash,
            },
            marker_path,
        )
    return receipt


def _quarantine_incomplete_final(month_dir: Path) -> None:
    final_dir = month_dir / "final"
    if not final_dir.exists():
        return
    failed_root = month_dir / "failed"
    failed_root.mkdir(parents=True, exist_ok=True)
    destination = failed_root / f"incomplete-final-{uuid.uuid4().hex}"
    final_dir.replace(destination)


def _rewrite_test_metadata_path(candidate_dir: Path) -> None:
    metadata_path = candidate_dir / "processed" / "test_set.csv.metadata.json"
    if not metadata_path.is_file():
        return
    metadata = _load_json_object(metadata_path, label="test-set metadata")
    metadata["test_set_path"] = str(
        (candidate_dir / "processed" / "test_set.csv").resolve(strict=False)
    )
    _write_json_atomically(metadata, metadata_path)


def _failure_payload(
    *, month: str, attempt_id: str, stage: str, error: Exception, output_dir: Path
) -> dict[str, Any]:
    return {
        "status": "failed_retryable",
        "passed": False,
        "ran": True,
        "month": month,
        "attempt_id": attempt_id,
        "failed_stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "output_dir": str(output_dir),
        "retryable": True,
        "auto_promotion_performed": False,
        "live_model_replaced": False,
    }


def run_monthly_candidate(
    *,
    weekly_update_succeeded: bool,
    weekly_update_completed_at: datetime,
    accepted_generation_id: str,
    accepted_features_path: Path | str,
    candidate_root: Path | str,
    live_models_dir: Path | str,
    v7_promotion_month: str,
    evaluator: CandidateEvaluator,
    state_path: Path | str | None = None,
    monthly_input: Mapping[str, Any] | None = None,
    monthly_context: Mapping[str, Any] | None = None,
    producer_inputs: Mapping[str, Any] | None = None,
    producer_output_root: Path | str | None = None,
    trainer: CandidateTrainer | None = None,
    validator: CandidateValidator | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    attempt_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
) -> dict[str, Any]:
    """Run, retry, or durably no-op one isolated candidate for a UTC month.

    ``weekly_update_succeeded`` is the publication result of the accepted weekly
    update.  The caller must pass the exact accepted generation and feature CSV.
    The first successful call pins a byte-for-byte copy for all retries.
    """

    update_at_utc = weekly_update_completed_at.astimezone(timezone.utc) if (
        weekly_update_completed_at.tzinfo is not None
        and weekly_update_completed_at.utcoffset() is not None
    ) else None
    if update_at_utc is None:
        raise MonthlyCandidateError("weekly_update_completed_at must be timezone-aware.")
    candidate_month = update_at_utc.strftime("%Y-%m")
    promotion_tuple = _parse_month(v7_promotion_month, label="v7_promotion_month")
    candidate_tuple = _parse_month(candidate_month, label="candidate month")
    if not weekly_update_succeeded:
        return {
            "status": "no_op",
            "passed": True,
            "ran": False,
            "month": candidate_month,
            "reason": "weekly_update_not_successful",
            "auto_promotion_performed": False,
        }
    if candidate_tuple <= promotion_tuple and monthly_input is None:
        return {
            "status": "no_op",
            "passed": True,
            "ran": False,
            "month": candidate_month,
            "reason": "not_after_v7_promotion_month",
            "auto_promotion_performed": False,
        }
    if not str(accepted_generation_id or "").strip():
        raise MonthlyCandidateError("accepted_generation_id is required.")

    root = _resolved(candidate_root)
    live_dir = _resolved(live_models_dir)
    durable_state_path = _resolved(state_path) if state_path else root / "state.json"
    source_path = _resolved(accepted_features_path)
    _assert_isolated_paths(
        candidate_root=root,
        state_path=durable_state_path,
        live_models_dir=live_dir,
    )
    loaded_monthly_input = None
    monthly_protected_paths = []
    if monthly_input is not None:
        from src.model.monthly_input import load_monthly_input
        loaded_monthly_input = load_monthly_input(dict(monthly_input), context=dict(monthly_context or {}))
        selected_context = loaded_monthly_input.transport["context"]
        if (selected_context["utc_month"] != candidate_month
            or selected_context["accepted_update_id"] != accepted_generation_id
            or weekly_update_completed_at.astimezone(timezone.utc) != datetime.fromisoformat(
                selected_context["accepted_at_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
            or selected_context["promotion_month_utc"] != v7_promotion_month):
            raise MonthlyCandidateError("Candidate arguments conflict with original monthly selection.")
        from src.model import prepared_input as prepared
        original_source = prepared._path(loaded_monthly_input.selection.document["artifact"]["path"])
        saved_source = prepared._path(loaded_monthly_input.selection.saved_root / "input.csv")
        if prepared._path(accepted_features_path) not in {original_source, saved_source}:
            raise MonthlyCandidateError("Candidate feature path differs from original saved selection.")
        source_path = original_source
        descriptor = prepared._json(prepared._attached(loaded_monthly_input.receipt["acceptance"]))
        originals = [*prepared._paths(loaded_monthly_input.receipt),
                     *prepared._paths(loaded_monthly_input.selection.references)]
        protected = [prepared._path(monthly_input["root"]), prepared._path(descriptor["raw_root"]),
                     *originals, *(path.parent for path in originals)]
        monthly_protected_paths = list(protected)
        for output in (prepared._path(candidate_root), prepared._path(state_path) if state_path else root / "state.json"):
            if any(prepared._overlaps(output, original) for original in protected):
                raise MonthlyCandidateError("Candidate output overlaps saved monthly input or original evidence.")
    selected_policy = getattr(loaded_monthly_input, "policy", None)
    contract = (resolve_monthly_candidate_contract(selected_policy) if selected_policy is not None
                else resolve_monthly_candidate_contract())
    if loaded_monthly_input is not None:
        from src.model.training_preflight import resolved_spec_identity
        if loaded_monthly_input.transport["training_spec"] != {
            "name": contract.spec.name, "sha256": resolved_spec_identity(contract.spec)}:
            raise MonthlyCandidateError("Saved named recipe requires a separately approved candidate implementation.")
    # Reject policy/state incompatibility before acquiring a writer lock. Re-read
    # under the lock below to retain the ordinary concurrent-writer protection.
    prior_state = _load_state(durable_state_path, promotion_month=v7_promotion_month, contract=contract)
    _validate_policy_state(prior_state, month=candidate_month, contract=contract)
    prior_month = prior_state.get("months", {}).get(candidate_month, {})
    original_completion = _completion_receipt(root / candidate_month, repair_marker=False)
    if original_completion is not None and original_completion.get("producer_inputs") is not None:
        prior_month = original_completion
    prior_producer = prior_month.get("producer_inputs")
    if prior_producer is not None:
        producer_inputs = prior_producer
        producer_output_root = prior_month.get("producer_output_base")
    if getattr(evaluator, "requires_producer_inputs", False) and producer_inputs is None:
        raise MonthlyCandidateError("Concrete producer requires independently accepted evaluator inputs before training.")
    if producer_inputs is None and producer_output_root is not None:
        raise MonthlyCandidateError("Producer output requires explicitly accepted evaluator inputs.")
    if producer_inputs is not None:
        from src.model.monthly_evaluator_inputs import load_evaluator_inputs, validate_selection, safe_path, overlaps
        selected_evaluator = load_evaluator_inputs(producer_inputs)
        if monthly_input is None or producer_output_root is None:
            raise MonthlyCandidateError("Producer needs saved monthly input and explicit isolated report output.")
        validate_selection(selected_evaluator["template"], monthly_input)
        from src.model.monthly_evaluator import preflight_external_inputs
        preflight_external_inputs(selected_evaluator, candidate_spec=asdict(contract.spec),
                                  policy_binding=contract.monthly_policy)
        report_root = safe_path(producer_output_root)
        protected = [root, live_dir, durable_state_path, safe_path(monthly_input["root"]),
                     safe_path(producer_inputs["root"]), Path(__file__).resolve().parents[2],
                     *monthly_protected_paths]
        from src.model.monthly_evaluator_inputs import _original_path
        protected.extend(_original_path(pin["path"]).parent for pin in selected_evaluator["original_pins"].values())
        if any(overlaps(report_root, path) for path in protected):
            raise MonthlyCandidateError("Producer report output overlaps protected source or evidence.")
    prior_input = root / candidate_month / "input"
    if prior_input.exists():
        _verify_input_receipt(prior_input, contract=contract,
            accepted_generation_id=accepted_generation_id,
            weekly_update_completed_at=weekly_update_completed_at,
            monthly_input=dict(monthly_input) if monthly_input is not None else None)
    trainer_fn = trainer or _default_trainer
    validator_fn = validator or validate_monthly_candidate_artifacts

    with _candidate_writer_lock(root):
        now = _clock_utc(clock)
        now_text = _utc_iso(now)
        state = _load_state(durable_state_path, promotion_month=v7_promotion_month, contract=contract)
        adoption = _validate_policy_state(state, month=candidate_month, contract=contract)
        if producer_inputs is not None:
            # Recheck the saved originals under the existing writer lock. Current
            # configuration cannot replace this month's previously selected plan.
            selected_evaluator = load_evaluator_inputs(producer_inputs)
            validate_selection(selected_evaluator["template"], monthly_input)
            preflight_external_inputs(selected_evaluator, candidate_spec=asdict(contract.spec),
                                      policy_binding=contract.monthly_policy)
            prior = state.get("months", {}).get(candidate_month, {}).get("producer_inputs")
            if prior is not None and prior != producer_inputs:
                raise MonthlyCandidateError("Saved producer selection changed before candidate lock.")
        if adoption is not None:
            state["adoptions"].setdefault(adoption["adoption_id"], adoption)
        month_dir = root / candidate_month

        recovered_receipt = _completion_receipt(month_dir, repair_marker=False)
        if recovered_receipt is not None:
            saved = _verify_input_receipt(month_dir / "input", contract=contract,
                accepted_generation_id=accepted_generation_id,
                weekly_update_completed_at=weekly_update_completed_at,
                monthly_input=dict(monthly_input) if monthly_input is not None else None)
            if (recovered_receipt.get("accepted_generation_id") != accepted_generation_id
                or recovered_receipt.get("input_receipt_sha256") != _sha256_file(month_dir / "input" / "input_receipt.json")
                or (monthly_input is not None and recovered_receipt.get("monthly_input") != monthly_input)):
                raise MonthlyCandidateError("Completed candidate differs from original selected input.")
            if recovered_receipt.get("month") != candidate_month:
                raise MonthlyCandidateError("Candidate receipt month is invalid.")
            if recovered_receipt.get("contract") != contract.receipt_payload():
                raise MonthlyCandidateError("Candidate receipt contract is invalid.")
            if (recovered_receipt.get("owner_review_required") is not True
                or recovered_receipt.get("auto_promotion_performed") is not False
                or recovered_receipt.get("live_model_replaced") is not False):
                raise MonthlyCandidateError("Candidate completion receipt has unsafe promotion metadata.")
            if contract.monthly_policy is not None or "evaluation" in recovered_receipt:
                _validate_completed_evaluation(recovered_receipt, candidate_dir=month_dir / "final",
                                               input_receipt=saved, contract=contract)
            # Only a fully matching input may repair its interrupted marker.
            _completion_receipt(month_dir, repair_marker=True)
            state["months"][candidate_month] = {
                "status": "completed",
                **({"producer_inputs": deepcopy(recovered_receipt["producer_inputs"])}
                   if recovered_receipt.get("producer_inputs") is not None else {}),
                **({"producer_output_base": recovered_receipt["producer_output_base"]}
                   if recovered_receipt.get("producer_inputs") is not None else {}),
                **({"monthly_policy": deepcopy(contract.monthly_policy)} if contract.monthly_policy else {}),
                "accepted_generation_id": recovered_receipt.get(
                    "accepted_generation_id"
                ),
                "candidate_receipt": "final/candidate_receipt.json",
                "candidate_receipt_sha256": _sha256_file(
                    month_dir / "final" / "candidate_receipt.json"
                ),
                "completed_at": recovered_receipt.get("completed_at"),
            }
            _write_json_atomically(state, durable_state_path)
            return {
                "status": "no_op",
                "passed": True,
                "ran": False,
                "month": candidate_month,
                "reason": "already_completed",
                "candidate_dir": str(month_dir / "final"),
                "auto_promotion_performed": False,
            }

        existing_month_state = state["months"].get(candidate_month)
        if isinstance(existing_month_state, dict) and existing_month_state.get(
            "status"
        ) == "completed":
            raise MonthlyCandidateError(
                "Durable state says the month completed, but its completion receipt "
                "is missing or invalid."
            )

        _quarantine_incomplete_final(month_dir)
        input_receipt = _prepare_or_load_input(
            month_dir=month_dir,
            accepted_features_path=source_path,
            accepted_generation_id=str(accepted_generation_id).strip(),
            weekly_update_completed_at=weekly_update_completed_at,
            contract=contract,
            created_at=now_text,
            loaded_monthly_input=loaded_monthly_input,
        )

        month_state = state["months"].setdefault(
            candidate_month,
            {
                "status": "pending",
                **({"producer_inputs": deepcopy(dict(producer_inputs))} if producer_inputs is not None else {}),
                **({"producer_output_base": str(report_root)} if producer_inputs is not None else {}),
                **({"monthly_policy": deepcopy(contract.monthly_policy)} if contract.monthly_policy else {}),
                "accepted_generation_id": input_receipt["accepted_generation_id"],
                "source_sha256": input_receipt["source_sha256"],
                "derived_sha256": input_receipt["derived_sha256"],
                "input_receipt": f"{candidate_month}/input/input_receipt.json",
                "attempts": [],
            },
        )
        if month_state.get("accepted_generation_id") != input_receipt.get(
            "accepted_generation_id"
        ) or month_state.get("source_sha256") != input_receipt.get("source_sha256"):
            raise MonthlyCandidateError("Durable monthly input identity changed.")
        if not isinstance(month_state.get("attempts"), list):
            raise MonthlyCandidateError("Monthly candidate attempts state is invalid.")
        if producer_inputs is not None:
            prior = month_state.setdefault("producer_inputs", deepcopy(dict(producer_inputs)))
            if prior != producer_inputs:
                raise MonthlyCandidateError("Monthly producer input identity changed.")
            if month_state.setdefault("producer_output_base", str(report_root)) != str(report_root):
                raise MonthlyCandidateError("Monthly producer output identity changed.")

        attempt_id = str(attempt_id_factory())
        if _SAFE_ATTEMPT_ID.fullmatch(attempt_id) is None:
            raise MonthlyCandidateError("attempt_id_factory returned an unsafe identifier.")
        attempt_dir = month_dir / "attempts" / attempt_id
        if attempt_dir.exists():
            raise MonthlyCandidateError(f"Monthly attempt already exists: {attempt_id}")
        (attempt_dir / "models").mkdir(parents=True, exist_ok=False)
        (attempt_dir / "processed").mkdir(parents=True, exist_ok=False)
        write_bytes_atomically(
            (month_dir / "input" / "modeling_dataset.csv").read_bytes(),
            attempt_dir / "processed" / "modeling_dataset.csv",
        )
        attempt_state = {
            "attempt_id": attempt_id,
            "status": "running",
            "started_at": now_text,
            "output_dir": str(attempt_dir),
        }
        month_state["attempts"].append(attempt_state)
        month_state["status"] = "running"
        month_state["last_attempt_id"] = attempt_id
        _write_json_atomically(state, durable_state_path)
        _write_json_atomically(attempt_state, attempt_dir / "attempt.json")

        live_before = _directory_fingerprint(live_dir)
        stage = "load_modeling_dataset"
        current_output_dir = attempt_dir
        completion_marker_written = False
        try:
            if loaded_monthly_input is not None:
                features_df = loaded_monthly_input.modeling_dataframe.copy(deep=True)
            else:
                features_df = pd.read_csv(
                    attempt_dir / "processed" / "modeling_dataset.csv", low_memory=False
                )
            stage = "training"
            trainer_fn(
                features_df=features_df,
                spec=contract.spec,
                models_dir=attempt_dir / "models",
                test_set_path=attempt_dir / "processed" / "test_set.csv",
            )
            stage = "attempt_validation"
            if _sha256_file(
                attempt_dir / "processed" / "modeling_dataset.csv"
            ) != input_receipt.get("derived_sha256"):
                raise MonthlyCandidateError(
                    "Trainer changed the pinned monthly modeling dataset."
                )
            _require_candidate_artifacts(attempt_dir, contract)
            first_validation = _validated_artifact_report(
                validator_fn(candidate_dir=attempt_dir, contract=contract),
                contract=contract, candidate_dir=attempt_dir,
            )

            stage = "finalize_directory"
            final_dir = month_dir / "final"
            if final_dir.exists():
                raise MonthlyCandidateError("Final candidate directory already exists.")
            attempt_dir.replace(final_dir)
            current_output_dir = final_dir
            _rewrite_test_metadata_path(final_dir)
            stage = "final_validation"
            _require_candidate_artifacts(final_dir, contract)
            final_validation = _validated_artifact_report(
                validator_fn(candidate_dir=final_dir, contract=contract),
                contract=contract, candidate_dir=final_dir,
            )
            stage = "evaluation"
            evaluation_request = _evaluator_request(final_dir, input_receipt=input_receipt,
                                                    contract=contract.receipt_payload(), producer_inputs=producer_inputs)
            # Save the request before dispatch, including interrupted/failed calls.
            _write_json_atomically(evaluation_request, final_dir / "evaluation_request.json")
            producer_kwargs = {}
            if producer_inputs is not None:
                from src.model.monthly_evaluator_inputs import load_evaluator_inputs
                plan = load_evaluator_inputs(producer_inputs, request=evaluation_request)
                _write_json_atomically(plan, final_dir / "evaluation_plan.json")
                producer_kwargs = {"producer_inputs": deepcopy(dict(producer_inputs)),
                    "producer_output_root": str(report_root / candidate_month / attempt_id)}
            evaluation = evaluator(candidate_dir=final_dir, input_receipt=deepcopy(input_receipt),
                                   contract=contract.receipt_payload(), **producer_kwargs)
            _write_json_atomically(evaluation, final_dir / "evaluation_original.json")
            if _evaluator_request(final_dir, input_receipt=input_receipt,
                                  contract=contract.receipt_payload(), producer_inputs=producer_inputs) != evaluation_request:
                raise MonthlyCandidateError("Evaluator changed the selected candidate artifact identities.")
            evaluation = validate_monthly_evaluator_result(evaluation, request=evaluation_request)
            if producer_inputs is not None:
                load_evaluator_inputs(producer_inputs, request=evaluation_request)
                from src.model.monthly_evaluator import verify_saved_report
                saved_report = verify_saved_report(producer_kwargs["producer_output_root"], request=evaluation_request)
                if saved_report != evaluation:
                    raise MonthlyCandidateError("Producer returned a different report than its retained original.")
            stage = "live_model_immutability"
            live_after = _assert_live_unchanged(live_dir, live_before)

            stage = "completion_receipt"
            completed_at = _utc_iso(_clock_utc(clock))
            attempt_state.update(
                {
                    "status": "completed",
                    "completed_at": completed_at,
                    "output_dir": str(final_dir),
                }
            )
            _write_json_atomically(attempt_state, final_dir / "attempt.json")
            inventory = _inventory(final_dir, excluded={"candidate_receipt.json"})
            receipt = {
                "schema_version": MONTHLY_CANDIDATE_SCHEMA_VERSION,
                "status": "complete_awaiting_owner_review",
                "month": candidate_month,
                "attempt_id": attempt_id,
                "completed_at": completed_at,
                "accepted_generation_id": input_receipt["accepted_generation_id"],
                **({"monthly_input": dict(monthly_input)} if monthly_input is not None else {}),
                "input_receipt_sha256": _sha256_file(
                    month_dir / "input" / "input_receipt.json"
                ),
                "source_sha256": input_receipt["source_sha256"],
                "derived_sha256": input_receipt["derived_sha256"],
                "source_date_bounds": {
                    "min": input_receipt["source_min_event_date"],
                    "max": input_receipt["source_max_event_date"],
                },
                "derived_date_bounds": {
                    "min": input_receipt["derived_min_event_date"],
                    "max": input_receipt["derived_max_event_date"],
                },
                "contract": contract.receipt_payload(),
                "attempt_validation": first_validation,
                "final_validation": final_validation,
                "evaluation": evaluation,
                "evaluation_request": evaluation_request,
                **producer_kwargs,
                **({"producer_output_base": str(report_root)} if producer_inputs is not None else {}),
                "artifact_inventory": inventory,
                "live_models_before": live_before,
                "live_models_after": live_after,
                "auto_promotion_performed": False,
                "live_model_replaced": False,
                "owner_review_required": True,
            }
            receipt_path = final_dir / "candidate_receipt.json"
            _write_json_atomically(receipt, receipt_path)
            receipt_hash = _sha256_file(receipt_path)
            _write_json_atomically(
                {
                    "schema_version": MONTHLY_CANDIDATE_SCHEMA_VERSION,
                    "candidate_receipt": "final/candidate_receipt.json",
                    "candidate_receipt_sha256": receipt_hash,
                },
                month_dir / "completed.json",
            )
            completion_marker_written = True
            month_state.update(
                {
                    "status": "completed",
                    "completed_at": completed_at,
                    "candidate_receipt": f"{candidate_month}/final/candidate_receipt.json",
                    "candidate_receipt_sha256": receipt_hash,
                }
            )
            _write_json_atomically(state, durable_state_path)
            return {
                "status": "complete_awaiting_owner_review",
                "passed": True,
                "ran": True,
                "month": candidate_month,
                "attempt_id": attempt_id,
                "candidate_dir": str(final_dir),
                "candidate_receipt": str(receipt_path),
                "accepted_generation_id": input_receipt["accepted_generation_id"],
                "auto_promotion_performed": False,
                "live_model_replaced": False,
                "owner_review_required": True,
            }
        except Exception as exc:
            try:
                _assert_live_unchanged(live_dir, live_before)
            except MonthlyCandidateError:
                raise
            if completion_marker_written:
                # The marker is the durable completion point.  Leave the final
                # directory intact so the next call can reconstruct state.
                raise
            if current_output_dir.name == "final" and current_output_dir.exists():
                failed_root = month_dir / "failed"
                failed_root.mkdir(parents=True, exist_ok=True)
                failed_dir = failed_root / attempt_id
                if failed_dir.exists():
                    failed_dir = failed_root / f"{attempt_id}-{uuid.uuid4().hex}"
                current_output_dir.replace(failed_dir)
                current_output_dir = failed_dir
            failure = _failure_payload(
                month=candidate_month,
                attempt_id=attempt_id,
                stage=stage,
                error=exc,
                output_dir=current_output_dir,
            )
            attempt_state.update(failure)
            month_state["status"] = "failed_retryable"
            month_state["last_failure"] = failure
            if current_output_dir.exists():
                _write_json_atomically(failure, current_output_dir / "attempt.json")
            _write_json_atomically(state, durable_state_path)
            return failure


__all__ = [
    "CANDIDATE_SPEC_NAME",
    "EXPECTED_LOGISTIC_FEATURE_COUNT",
    "EXPECTED_NO_ODDS_FEATURE_COUNT",
    "EXPECTED_PRIMARY_FEATURE_COUNT",
    "MODELING_START_DATE",
    "MonthlyCandidateError",
    "ResolvedMonthlyCandidateContract",
    "UNSUPPORTED_CONTRACT_START_DATE",
    "build_monthly_modeling_view",
    "resolve_monthly_candidate_contract",
    "run_monthly_candidate",
    "validate_monthly_candidate_artifacts",
]
