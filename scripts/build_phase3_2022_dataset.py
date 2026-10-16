"""Build deterministic Phase 3 dataset evidence without training a model.

The command accepts only the completed Session 3 identity-correct snapshot. It
builds prior-only features from the full dated chronology, materializes the
registered 2022-present durability contract, and only then selects completed,
labeled model-example rows from 2022 onward.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.io_utils import write_bytes_atomically  # noqa: E402
from src.features.build_features import build_features  # noqa: E402
from src.model.training_spec import (  # noqa: E402
    NamedModelTrainingSpec,
    materialize_and_validate_spec_features,
    resolve_named_training_spec,
)


SPEC_NAME = "full_live_contract_v6_durability_2022"
TARGET_START_DATE = "2022-01-01"
SESSION3_ROOT = REPO_ROOT / ".codex_stage" / "phase3_2022_present_20260822_v1"
SESSION3_INPUT_PATH = SESSION3_ROOT / "fights_cleaned.csv"
SESSION3_INPUT_SHA256 = (
    "29a0d591917c9f31099d250229ef44dcee91ce67f5baa8c7ccb8ad70bdee7c74"
)
DATASET_FILENAME = "modeling_dataset.csv"
EVIDENCE_FILENAME = "dataset_evidence.json"
_ALLOWED_OUTPUT_NAMES = frozenset({DATASET_FILENAME, EVIDENCE_FILENAME})
_SOURCE_ROW_COLUMN = "__phase3_source_row"

_REQUIRED_INPUT_COLUMNS = (
    "event_date",
    "fighter_a",
    "fighter_b",
    "fighter_a_id",
    "fighter_b_id",
    "winner",
    "target",
)
_MODEL_METADATA_COLUMNS = (
    "source_row_number",
    "event_date",
    "event_name",
    "weight_class",
    "fighter_a",
    "fighter_b",
    "fighter_a_id",
    "fighter_b_id",
    "winner",
    "method",
    "target",
)
_UNSAFE_OUTPUT_ROOTS = (
    REPO_ROOT / "data" / "processed",
    REPO_ROOT / "data" / "frozen",
    REPO_ROOT / "models",
    REPO_ROOT / ".codex_stage" / "phase2_identity_migration_20260822_v1_bundle",
)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _validate_paths(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    try:
        expected_input = SESSION3_INPUT_PATH.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"Required Session 3 input is missing: {SESSION3_INPUT_PATH}") from exc

    try:
        resolved_input = Path(input_path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"Input file does not exist: {input_path}") from exc
    if not resolved_input.is_file():
        raise ValueError(f"Input must be a file: {resolved_input}")
    if resolved_input != expected_input:
        raise ValueError(
            "Input must be the exact Session 3 identity-correct snapshot: "
            f"{expected_input}"
        )

    actual_input_hash = _sha256_bytes(resolved_input.read_bytes())
    if actual_input_hash != SESSION3_INPUT_SHA256:
        raise ValueError(
            "Session 3 input SHA-256 mismatch: "
            f"expected {SESSION3_INPUT_SHA256}, observed {actual_input_hash}"
        )

    resolved_root = SESSION3_ROOT.resolve(strict=True)
    resolved_output = Path(output_dir).resolve(strict=False)
    if resolved_output == resolved_root or not _is_within(resolved_output, resolved_root):
        raise ValueError(
            "Output must be a child directory beneath the isolated Phase 3 root; "
            "canonical and frozen paths are forbidden"
        )

    protected_inputs = (resolved_root / "inputs").resolve(strict=False)
    if _is_within(resolved_output, protected_inputs):
        raise ValueError("Output cannot overwrite or nest beneath Session 3 inputs")

    for unsafe_root in _UNSAFE_OUTPUT_ROOTS:
        resolved_unsafe = unsafe_root.resolve(strict=False)
        if _is_within(resolved_output, resolved_unsafe):
            raise ValueError(f"Output cannot be written beneath canonical/frozen root: {resolved_unsafe}")

    if resolved_output.exists() and not resolved_output.is_dir():
        raise ValueError(f"Output must be a directory: {resolved_output}")
    if resolved_output.exists():
        unexpected = sorted(
            child.name
            for child in resolved_output.iterdir()
            if child.name not in _ALLOWED_OUTPUT_NAMES
        )
        if unexpected:
            raise ValueError(
                "Output directory contains files not owned by this command: "
                + ", ".join(unexpected)
            )

    return resolved_input, resolved_output


def _row_selection(
    frame: pd.DataFrame,
    *,
    start_date: str = TARGET_START_DATE,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Select completed/labeled targets and return a fixed-order exclusion ledger."""
    required = {"event_date", "fighter_a", "fighter_b", "winner", "target"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot select modeling rows; missing columns: {missing}")

    selected = frame.copy()
    dates = pd.to_datetime(selected["event_date"], errors="coerce", format="mixed")
    target = pd.to_numeric(selected["target"], errors="coerce")
    fighter_a = selected["fighter_a"].astype("string").fillna("").str.strip()
    fighter_b = selected["fighter_b"].astype("string").fillna("").str.strip()
    winner = selected["winner"].astype("string").fillna("").str.strip()

    start = pd.Timestamp(start_date)
    invalid_date = dates.isna()
    pre_start = dates.notna() & dates.lt(start)
    in_window = dates.notna() & dates.ge(start)
    invalid_target = in_window & ~(target.notna() & target.isin([0, 1]))
    eligible_target = in_window & ~invalid_target
    winner_is_a = fighter_a.ne("") & winner.eq(fighter_a)
    winner_is_b = fighter_b.ne("") & winner.eq(fighter_b)
    invalid_winner = eligible_target & ~(winner_is_a | winner_is_b)
    eligible_winner = eligible_target & ~invalid_winner
    target_winner_mismatch = eligible_winner & ~(
        (winner_is_a & target.eq(1)) | (winner_is_b & target.eq(0))
    )
    include = eligible_winner & ~target_winner_mismatch

    reasons = [
        {
            "code": "invalid_event_date",
            "count": int(invalid_date.sum()),
            "meaning": "event_date cannot be placed in the legitimate chronology",
        },
        {
            "code": "pre_2022_history_only",
            "count": int(pre_start.sum()),
            "meaning": (
                "fight is retained while building priors but is not a model example"
            ),
        },
        {
            "code": "missing_or_nonbinary_target",
            "count": int(invalid_target.sum()),
            "meaning": "2022-present fight has no observed binary target",
        },
        {
            "code": "winner_not_exact_participant",
            "count": int(invalid_winner.sum()),
            "meaning": "winner is missing or does not exactly match fighter A or B",
        },
        {
            "code": "target_winner_mismatch",
            "count": int(target_winner_mismatch.sum()),
            "meaning": "binary target conflicts with the exact observed winner",
        },
    ]
    excluded = sum(int(reason["count"]) for reason in reasons)
    if excluded + int(include.sum()) != len(selected):
        raise AssertionError("Modeling-row selection reasons do not partition the source rows")

    selected = selected.loc[include].copy()
    selected["event_date"] = dates.loc[include]
    selected["target"] = target.loc[include].astype(int)
    return selected, reasons


def _present(series: pd.Series) -> pd.Series:
    text = series.astype("string").fillna("").str.strip()
    return text.ne("")


def _identity_coverage(frame: pd.DataFrame) -> dict[str, int]:
    a_present = _present(frame["fighter_a_id"])
    b_present = _present(frame["fighter_b_id"])
    return {
        "rows": int(len(frame)),
        "rows_with_both_ids": int((a_present & b_present).sum()),
        "rows_with_either_id": int((a_present | b_present).sum()),
        "rows_missing_either_id": int((~(a_present & b_present)).sum()),
        "missing_participant_id_slots": int((~a_present).sum() + (~b_present).sum()),
    }


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(
        buffer,
        index=False,
        lineterminator="\n",
        na_rep="",
        float_format="%.17g",
    )
    return buffer.getvalue().encode("utf-8")


def _validate_spec() -> NamedModelTrainingSpec:
    spec = resolve_named_training_spec(SPEC_NAME)
    if spec.train_start_date != TARGET_START_DATE:
        raise ValueError(
            f"{SPEC_NAME} must start at {TARGET_START_DATE}, observed {spec.train_start_date!r}"
        )
    violations = spec.validate_feature_contract()
    if violations:
        raise ValueError(f"{SPEC_NAME} has feature-contract violations: {violations}")
    return spec


def build_phase3_dataset(*, input_path: Path, output_dir: Path) -> dict[str, object]:
    resolved_input, resolved_output = _validate_paths(input_path, output_dir)
    input_bytes = resolved_input.read_bytes()
    input_hash = _sha256_bytes(input_bytes)
    if input_hash != SESSION3_INPUT_SHA256:
        raise ValueError("Session 3 input changed between path validation and snapshot read")

    source = pd.read_csv(io.BytesIO(input_bytes), low_memory=False)
    missing_input_columns = sorted(set(_REQUIRED_INPUT_COLUMNS) - set(source.columns))
    if missing_input_columns:
        raise ValueError(f"Session 3 input is missing required columns: {missing_input_columns}")
    if _SOURCE_ROW_COLUMN in source.columns:
        raise ValueError(f"Session 3 input unexpectedly contains reserved column {_SOURCE_ROW_COLUMN}")
    source[_SOURCE_ROW_COLUMN] = range(len(source))

    spec = _validate_spec()
    source_targets, exclusion_reasons = _row_selection(source)
    valid_dates = pd.to_datetime(source["event_date"], errors="coerce", format="mixed")
    chronology = source.loc[valid_dates.notna()].copy()
    chronology["event_date"] = valid_dates.loc[valid_dates.notna()]

    built = build_features(chronology)
    if _SOURCE_ROW_COLUMN not in built.columns:
        raise ValueError("Feature builder did not preserve the Phase 3 source-row identity")
    if len(built) != len(chronology) or built[_SOURCE_ROW_COLUMN].duplicated().any():
        raise ValueError("Feature builder changed the one-row-per-fight chronology contract")
    materialized = materialize_and_validate_spec_features(built, spec)
    modeled, materialized_reasons = _row_selection(materialized)

    expected_source_rows = sorted(source_targets[_SOURCE_ROW_COLUMN].astype(int).tolist())
    observed_source_rows = sorted(modeled[_SOURCE_ROW_COLUMN].astype(int).tolist())
    if observed_source_rows != expected_source_rows:
        raise ValueError(
            "Feature materialization changed the completed/labeled 2022-present row universe"
        )
    if materialized_reasons[0]["count"] != 0:
        raise ValueError("Dated chronology produced invalid event dates during feature construction")

    modeled = modeled.sort_values(
        ["event_date", _SOURCE_ROW_COLUMN],
        kind="stable",
    ).reset_index(drop=True)
    modeled["source_row_number"] = modeled[_SOURCE_ROW_COLUMN].astype(int) + 1
    modeled["event_date"] = modeled["event_date"].dt.strftime("%Y-%m-%d")

    metadata_columns = [column for column in _MODEL_METADATA_COLUMNS if column in modeled]
    output_columns = metadata_columns + [
        column for column in spec.feature_cols if column not in metadata_columns
    ]
    modeling_dataset = modeled.loc[:, output_columns].copy()
    if modeling_dataset.empty:
        raise ValueError("Refusing to write an empty 2022-present modeling dataset")

    dataset_bytes = _csv_bytes(modeling_dataset)
    dataset_hash = _sha256_bytes(dataset_bytes)
    dataset_path = resolved_output / DATASET_FILENAME
    evidence_path = resolved_output / EVIDENCE_FILENAME

    missingness = []
    for feature in spec.feature_cols:
        missing_rows = int(modeling_dataset[feature].isna().sum())
        missingness.append(
            {
                "feature": feature,
                "missing_rows": missing_rows,
                "observed_rows": int(len(modeling_dataset) - missing_rows),
                "missing_pct": round(
                    missing_rows / len(modeling_dataset) * 100.0,
                    6,
                ),
            }
        )

    cutoff = pd.Timestamp(spec.train_cutoff_date)
    modeled_dates = pd.to_datetime(modeling_dataset["event_date"], errors="raise")
    report: dict[str, object] = {
        "schema_version": 1,
        "dataset_id": "phase3-2022-present-20260822-v1",
        "spec": {
            "name": spec.name,
            "description": spec.description,
            "dataset_variant": spec.dataset_variant,
            "train_start_date": spec.train_start_date,
            "inherited_train_cutoff_date": spec.train_cutoff_date,
            "feature_count": len(spec.feature_cols),
            "feature_contract_sha256": _sha256_bytes(
                _canonical_json_bytes(spec.feature_cols)
            ),
            "spec_sha256": _sha256_bytes(_canonical_json_bytes(asdict(spec))),
        },
        "selection": {
            "model_example_rule": (
                "event_date >= 2022-01-01; target is observed binary; winner "
                "exactly matches the target-consistent fighter"
            ),
            "history_rule": (
                "build prior-only features from every legitimately dated source fight "
                "before filtering model examples"
            ),
            "legacy_cutoff_handling": (
                "train_start_date equals the inherited train_cutoff_date, so the legacy "
                "pre-cutoff train side is intentionally empty; this evidence command "
                "selects the 2022-present modeling universe directly and does not train"
            ),
            "legacy_pre_cutoff_training_side_rows": int(modeled_dates.lt(cutoff).sum()),
            "modeling_universe_rows": int(len(modeling_dataset)),
        },
        "counts": {
            "all_ufc_source_fights": int(len(source)),
            "legitimately_dated_history_rows": int(len(chronology)),
            "fights_2022_present": int(
                pd.to_datetime(source["event_date"], errors="coerce", format="mixed")
                .ge(pd.Timestamp(TARGET_START_DATE))
                .sum()
            ),
            "completed_labeled_2022_present_model_rows": int(len(modeling_dataset)),
            "excluded_or_unresolved_rows": int(
                sum(int(reason["count"]) for reason in exclusion_reasons)
            ),
        },
        "ordered_exclusion_reasons": exclusion_reasons,
        "identity_coverage": {
            "source": _identity_coverage(source),
            "modeling_universe": _identity_coverage(source_targets),
            "unresolved_ambiguous_identity_policy": "fatal_before_output",
        },
        "missingness": {
            "rows": int(len(modeling_dataset)),
            "feature_count": len(spec.feature_cols),
            "fully_missing_features": int(
                sum(item["missing_rows"] == len(modeling_dataset) for item in missingness)
            ),
            "features": missingness,
        },
        "artifacts": {
            "input": {
                "filename": SESSION3_INPUT_PATH.name,
                "bytes": len(input_bytes),
                "rows": int(len(source)),
                "sha256": input_hash,
            },
            "modeling_dataset": {
                "filename": DATASET_FILENAME,
                "bytes": len(dataset_bytes),
                "rows": int(len(modeling_dataset)),
                "columns": len(modeling_dataset.columns),
                "sha256": dataset_hash,
            },
        },
    }
    report["evidence_core_sha256"] = _sha256_bytes(_canonical_json_bytes(report))
    evidence_bytes = _pretty_json_bytes(report)

    if resolved_input.read_bytes() != input_bytes:
        raise RuntimeError("Session 3 input changed while the dataset evidence was built")

    write_bytes_atomically(dataset_bytes, dataset_path)
    write_bytes_atomically(evidence_bytes, evidence_path)
    if dataset_path.read_bytes() != dataset_bytes or evidence_path.read_bytes() != evidence_bytes:
        raise RuntimeError("Phase 3 output verification failed after atomic write")

    return {
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "evidence_path": str(evidence_path),
        "evidence_sha256": _sha256_bytes(evidence_bytes),
        "rows": int(len(modeling_dataset)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Exact Session 3 fights CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New child directory beneath the isolated Phase 3 root",
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        result = build_phase3_dataset(input_path=args.input, output_dir=args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
