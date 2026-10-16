"""Explicit read-only model and retained-feature evaluation connections.

The producer owns independent acceptance, original-byte retention and before/
after identity checks. These adapters reject incompatible contracts before the
genuine load/score operation. No active model alias, configured data directory,
repairing loader, source request, sampling, tuning or artifact write is used.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.model.feature_parity import classify, expand_time_aged, numeric_value


MODEL_ROLES = ("primary", "logistic", "no_odds")
IDENTITY_FIELDS = ("row_id", "event_date", "fighter_a_id", "fighter_b_id")
MAX_ROWS = 100000


class EvaluatorAdapterRefusal(ValueError):
    """The exact accepted inputs cannot support the requested operation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluatorAdapterRefusal(message)


def _day(value: Any) -> date:
    _require(isinstance(value, str), "Event dates must be original date-only strings.")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise EvaluatorAdapterRefusal("Invalid event date.") from exc
    _require(parsed.isoformat() == value, "Event date must use YYYY-MM-DD.")
    return parsed


def _window(window: dict) -> tuple[pd.Timestamp, pd.Timestamp, bool]:
    _require(isinstance(window, dict) and set(window) == {"start_inclusive", "end", "end_inclusive"}
             and type(window["end_inclusive"]) is bool, "Explicit evaluation window required.")
    start = pd.Timestamp(_day(window["start_inclusive"]), tz="UTC")
    try:
        end = pd.Timestamp(window["end"])
    except (TypeError, ValueError) as exc:
        raise EvaluatorAdapterRefusal("Invalid evaluation end.") from exc
    if end.tz is None:
        end = pd.Timestamp(_day(window["end"]), tz="UTC")
    else:
        _require(end.utcoffset().total_seconds() == 0, "Evaluation end must be UTC.")
        end = end.tz_convert("UTC")
    _require(not pd.isna(end) and start <= end, "Conflicting evaluation window.")
    return start, end, window["end_inclusive"]


def _identity(row: dict) -> dict:
    identity = {key: row.get(key) for key in IDENTITY_FIELDS}
    _require(all(isinstance(value, str) and bool(value.strip()) for value in identity.values()),
             "Missing row identity or fighter-side orientation.")
    _require(identity["fighter_a_id"] != identity["fighter_b_id"], "A fighter cannot occupy both sides.")
    _day(identity["event_date"])
    return identity


def preflight_rows(*, frame: pd.DataFrame, rows: list[dict], feature_cols: list[str],
                   window: dict, allow_missing: bool = False) -> pd.DataFrame:
    """Verify the independently fixed population; never supply absent features."""
    _require(isinstance(frame, pd.DataFrame) and frame.columns.is_unique,
             "Table columns must be unique.")
    _require(isinstance(rows, list) and 0 < len(rows) <= MAX_ROWS, "Fixed rows absent or oversized.")
    _require(isinstance(feature_cols, list) and 0 < len(feature_cols) <= 512
             and all(isinstance(column, str) and bool(column) for column in feature_cols)
             and len(set(feature_cols)) == len(feature_cols), "Exact feature order is missing or duplicated.")
    required = set(IDENTITY_FIELDS) | set(feature_cols)
    _require(required.issubset(frame.columns), "Required table columns are absent; NaN filling is forbidden.")
    _require(len(frame) <= MAX_ROWS, "Table row bound exceeded.")
    start, end, inclusive = _window(window)
    identities = [_identity(row) for row in rows]
    ids = [row["row_id"] for row in identities]
    _require(len(set(ids)) == len(ids), "Selected row IDs repeat.")
    fight_keys = [(row["event_date"], *sorted((row["fighter_a_id"], row["fighter_b_id"]))) for row in identities]
    _require(len(set(fight_keys)) == len(fight_keys), "The selected population repeats an oriented or mirrored fight.")
    actual = [_identity(row) for row in frame.to_dict("records")]
    actual_ids = [row["row_id"] for row in actual]
    _require(len(set(actual_ids)) == len(actual_ids), "Table row IDs repeat.")
    _require(set(actual_ids).issubset(ids), "Table has unselected rows.")
    _require(allow_missing or set(actual_ids) == set(ids), "Required rows are missing.")
    selected = {row["row_id"]: row for row in identities}
    _require(all(row == selected[row["row_id"]] for row in actual), "Row identity/date/orientation changed.")
    for row in identities:
        instant = pd.Timestamp(_day(row["event_date"]), tz="UTC")
        _require(instant >= start and (instant <= end if inclusive else instant < end),
                 "Selected row is outside the independently fixed window.")
    for column in feature_cols:
        for value in frame[column]:
            try:
                numeric_value(value)
            except (TypeError, ValueError) as exc:
                raise EvaluatorAdapterRefusal(f"Invalid numeric feature {column}.") from exc
    indexed = frame.set_index("row_id", drop=False)
    return indexed.loc[[row_id for row_id in ids if row_id in indexed.index]].copy()


def validate_primary_spec(primary_spec: dict, training_window: dict) -> dict[str, dict]:
    """Validate the supported recipe, exact feature order and approved no-odds transform."""
    from src.model.training_spec import resolve_named_training_spec
    from src.model.production_bundle import _expected_no_odds_spec_payload

    _require(isinstance(primary_spec, dict)
             and primary_spec.get("name") == "full_live_contract_v6_durability_2022",
             "Unsupported explicit evaluator recipe.")
    registered = asdict(resolve_named_training_spec(primary_spec["name"]))
    _require(set(primary_spec) == set(registered), "Incomplete or extended resolved training spec.")
    _require(isinstance(training_window, dict)
             and set(training_window) == {"start_inclusive", "end_exclusive"},
             "Exact original training window required.")
    start, end = _day(training_window["start_inclusive"]), _day(training_window["end_exclusive"])
    _require(start == date(2022, 1, 1) and start < end, "Unsupported training history window.")
    expected = deepcopy(registered)
    expected.update(train_start_date=start.isoformat(), train_end_date="", train_cutoff_date=end.isoformat())
    # Fit metadata is supplied by the accepted sidecar and must later match the
    # embedded sidecar exactly. It does not alter the named modeling recipe.
    for key in ("git_hash", "trained_at"):
        _require(isinstance(primary_spec[key], str), "Invalid original fit metadata.")
        expected[key] = primary_spec[key]
    _require(primary_spec["xgb_params"] == registered["xgb_params"], "Unsupported model hyperparameters.")
    if expected["odds_noise_seed"] is None:
        expected["odds_noise_seed"] = int(expected["xgb_params"].get("random_state", 42))
    effective = deepcopy(primary_spec)
    if effective["odds_noise_seed"] is None:
        effective["odds_noise_seed"] = int(effective["xgb_params"].get("random_state", 42))
    _require(effective == expected, "Resolved spec differs from the accepted supported recipe/window.")
    return {"primary": deepcopy(primary_spec), "logistic": deepcopy(primary_spec),
            "no_odds": _expected_no_odds_spec_payload(primary_spec)}


@dataclass(frozen=True)
class LoadedBundle:
    results: dict[str, Any]
    measurement: dict[str, Any]


def validate_training_accounting(result: dict, *, training_row_count: int, feature_cols: list[str]) -> dict:
    """Match artifact counters to the actual selected pre-augmentation population."""
    from src.model.orientation import has_ab_feature_pair

    _require(type(training_row_count) is int and 0 < training_row_count <= MAX_ROWS,
             "Actual observed training row count required.")
    mirrored = has_ab_feature_pair(feature_cols)
    expected = {"observed_training_rows": training_row_count,
                "effective_training_rows": training_row_count * (2 if mirrored else 1),
                "mirror_augmentation": mirrored}
    _require(isinstance(result, dict) and all(type(result.get(key)) is type(value) and result.get(key) == value
                                             for key, value in expected.items()),
             "Artifact observed/mirrored training row accounting conflicts with original population.")
    return expected


def load_bundle(*, model_paths: dict, primary_spec: dict, training_window: dict,
                role: str, training_row_count: int) -> LoadedBundle:
    """Route explicit immutable artifacts to production's nonrepairing loader."""
    from src.model.production_bundle import _embedded_training_spec_payload
    from src.model.monthly_candidate import _validate_model_payload

    _require(role in {"candidate", "historical", "reference"}, "Unsupported model bundle role.")
    _require(isinstance(model_paths, dict) and set(model_paths) == set(MODEL_ROLES),
             "Exactly three explicit model paths are required.")
    specs = validate_primary_spec(primary_spec, training_window)
    _require(type(training_row_count) is int and 0 < training_row_count <= MAX_ROWS,
             "Actual observed training row count required before load.")
    paths = {}
    for model_role in MODEL_ROLES:
        path = Path(model_paths[model_role])
        _require(path.is_absolute() and ".." not in path.parts and path.is_file(),
                 "Explicit existing model path required; aliases are forbidden.")
        _require(not any(parent.is_symlink() for parent in (path, *path.parents)), "Linked model paths refuse.")
        for component in (path, *path.parents):
            _require(not getattr(component.lstat(), "st_file_attributes", 0) & 0x400,
                     "Model reparse points are unsupported.")
        paths[model_role] = path
    _require(len(set(path.resolve() for path in paths.values())) == 3, "Model roles cannot share an artifact path.")
    results, measurements = {}, []
    for model_role in MODEL_ROLES:
        spec, result = _embedded_training_spec_payload(paths[model_role], label=f"evaluator {role} {model_role}")
        _require(spec == specs[model_role], "Loaded model does not match the exact accepted sidecar/role.")
        _require(result.get("feature_cols") == specs[model_role]["feature_cols"],
                 "Loaded feature order conflicts with the resolved contract.")
        _validate_model_payload(result, label=model_role,
                                feature_count=len(spec["feature_cols"]),
                                require_impute_strategy=model_role != "logistic")
        training_rows = validate_training_accounting(result, training_row_count=training_row_count,
                                                     feature_cols=spec["feature_cols"])
        results[model_role] = result
        measurements.append({"model_role": model_role, "spec": deepcopy(spec),
                             "feature_columns": list(spec["feature_cols"]),
                             "training_window": deepcopy(training_window), "training_rows": training_rows})
    return LoadedBundle(results, {"passed": True, "role": role, "models": measurements, "errors": []})


def score_bundle(*, loaded: LoadedBundle, frame: pd.DataFrame, rows: list[dict], window: dict) -> dict:
    """Use the established scorer with explicit already-loaded model results."""
    from src.model.refreshed_v4_eval import evaluate_loaded_model

    _require(isinstance(loaded, LoadedBundle) and set(loaded.results) == set(MODEL_ROLES),
             "An explicitly validated loaded bundle is required.")
    _require("target" in frame.columns and frame["target"].isin([0, 1]).all()
             and not pd.api.types.is_bool_dtype(frame["target"].dtype), "Binary observed targets required.")
    _require(all(row.get("exclusion") is None for row in rows), "Scoring rows cannot be excluded.")
    metrics, probabilities, feature_columns = {}, {}, {}
    ordered = None
    for model_role in MODEL_ROLES:
        result = loaded.results[model_role]
        columns = result["feature_cols"]
        ordered = preflight_rows(frame=frame, rows=rows, feature_cols=columns, window=window)
        measured = evaluate_loaded_model(test_df=ordered, model_result=result)
        predictions = measured["predictions"]
        _require(isinstance(predictions, pd.DataFrame) and len(predictions) == len(ordered)
                 and predictions[list(IDENTITY_FIELDS)].to_dict("records") == ordered[list(IDENTITY_FIELDS)].to_dict("records")
                 and predictions["target"].tolist() == ordered["target"].tolist(),
                 "Prediction rows were lost, replaced or reoriented.")
        values = [numeric_value(value) for value in predictions["prob_a"]]
        _require(all(value is not None and 0 <= value <= 1 for value in values), "Invalid probability output.")
        metrics[model_role] = deepcopy(measured["metrics"])
        probabilities[model_role] = values
        feature_columns[model_role] = list(columns)
    measured_rows = []
    for index, row in enumerate(ordered.to_dict("records")):
        measured_rows.append({**_identity(row), "target": int(row["target"]),
                              "predictions": {model_role: probabilities[model_role][index] for model_role in MODEL_ROLES}})
    return {"passed": True, "role": loaded.measurement["role"], "rows": measured_rows, "metrics": metrics,
            "feature_columns": feature_columns, "errors": []}


def validate_aging_allowances(allowances: list[dict], *, rows: list[dict], feature_cols: list[str], mode: str) -> dict:
    """Validate independently accepted original exceptions, never derive approval."""
    _require(isinstance(allowances, list) and len(allowances) <= len(rows) * len(feature_cols),
             "Bounded original aging allowance list required.")
    _require(mode == "prefight" or not allowances, "Exact comparison cannot carry aging allowances.")
    selected = {row["row_id"]: _identity(row) for row in rows}
    aged = expand_time_aged(feature_cols)
    result = {}
    for allowance in allowances:
        _require(isinstance(allowance, dict) and set(allowance) == {
            "row_id", "feature", "as_of_date", "live", "train", "delta", "reason"},
            "Original aging allowance fields missing or unknown.")
        row_id, feature = allowance["row_id"], allowance["feature"]
        _require(isinstance(row_id, str) and isinstance(feature, str)
                 and row_id in selected and feature in aged, "Aging allowance selects unknown row/feature.")
        _require((row_id, feature) not in result, "Original aging allowance repeats.")
        expected_date = _day(selected[row_id]["event_date"]) - timedelta(days=1)
        _require(allowance["as_of_date"] == expected_date.isoformat(), "Aging allowance reference date conflicts.")
        _require(all(type(allowance[key]) in (int, float) and math.isfinite(allowance[key])
                     for key in ("live", "train", "delta")), "Aging allowances require finite original numeric values.")
        _require(abs(allowance["live"] - allowance["train"]) > 1e-6
                 and math.isclose(allowance["delta"], allowance["live"] - allowance["train"],
                                  rel_tol=1e-12, abs_tol=1e-12), "Aging allowance delta conflicts.")
        _require(isinstance(allowance["reason"], str) and bool(allowance["reason"].strip())
                 and len(allowance["reason"]) <= 1024, "Original aging allowance explanation missing.")
        result[(row_id, feature)] = deepcopy(allowance)
    return result


def compare_parity(*, oracle_frame: pd.DataFrame, replay_frame: pd.DataFrame,
                   history_frame: pd.DataFrame, rows: list[dict], feature_cols: list[str],
                   window: dict, mode: str, tolerance: float = 1e-6,
                   aging_allowances: list[dict] | None = None) -> dict:
    """Compare accepted original feature outputs with full selected-row accounting.

    This invokes the same numeric/NaN and prefight-aging rules used by
    scripts/parity_replay.py. It never invokes that command's sampling, configured
    source readers or feature-builder fallback. Original replay outputs must have
    an explicit as_of_date matching the selected mode for every compared row.
    """
    _require(mode in {"exact", "prefight"}, "Unsupported parity mode.")
    _require(type(tolerance) in {int, float} and tolerance == 1e-6,
             "Unsupported parity tolerance; the accepted comparison rule is fixed.")
    oracle = preflight_rows(frame=oracle_frame, rows=rows, feature_cols=feature_cols, window=window)
    replay = preflight_rows(frame=replay_frame, rows=rows, feature_cols=feature_cols,
                            window=window, allow_missing=True)
    _require("as_of_date" in replay.columns, "Original replay reference dates are required.")
    _require(isinstance(history_frame, pd.DataFrame) and history_frame.columns.is_unique
             and set(IDENTITY_FIELDS).issubset(history_frame.columns) and len(history_frame) <= MAX_ROWS,
             "Explicit bounded oriented history is required for prefight exclusions.")
    history = [_identity(row) for row in history_frame.to_dict("records")]
    _require(len({row["row_id"] for row in history}) == len(history), "History row IDs repeat.")
    earliest: dict[str, date] = {}
    for row in history:
        day = _day(row["event_date"])
        for side in ("fighter_a_id", "fighter_b_id"):
            earliest[row[side]] = min(day, earliest.get(row[side], day))
    allowances = validate_aging_allowances([] if aging_allowances is None else aging_allowances,
        rows=rows, feature_cols=feature_cols, mode=mode)
    used_allowances = set()
    counts = {"compared_rows": 0, "excluded_rows": 0, "missing_rows": 0,
              "failed_rows": 0, "mismatch_count": 0, "allowlisted_mismatch_count": 0}
    records = []
    for selector in rows:
        identity = _identity(selector)
        day = _day(identity["event_date"])
        category = selector.get("coverage_category")
        _require(isinstance(category, str) and bool(category.strip()), "Coverage category missing.")
        record = {**identity, "coverage_category": category, "disposition": "compared",
                  "reason": None, "differences": []}
        exclusion = selector.get("exclusion")
        established = all(earliest.get(identity[side], day) < day for side in ("fighter_a_id", "fighter_b_id"))
        if exclusion is not None and mode == "prefight":
            _require(exclusion == "no_prior_processed_history" and not established,
                     "Unproved or unsupported prefight exclusion.")
            record.update(disposition="excluded", reason=exclusion)
            counts["excluded_rows"] += 1
        elif identity["row_id"] not in replay.index:
            record.update(disposition="missing", reason="original_replay_row_missing")
            counts["missing_rows"] += 1
        else:
            replay_row = replay.loc[identity["row_id"]]
            expected_as_of = day if mode == "exact" else day - timedelta(days=1)
            if replay_row["as_of_date"] != expected_as_of.isoformat():
                record.update(disposition="failed", reason="replay_reference_date_conflict")
                counts["failed_rows"] += 1
            elif mode == "prefight" and not established:
                record.update(disposition="failed", reason="unaccounted_no_prior_processed_history")
                counts["failed_rows"] += 1
            else:
                counts["compared_rows"] += 1
                for column in feature_cols:
                    observed = numeric_value(replay_row[column])
                    expected = numeric_value(oracle.loc[identity["row_id"], column])
                    verdict = classify(observed, expected, tolerance)
                    if verdict == "match":
                        continue
                    allowance = allowances.get((identity["row_id"], column))
                    allowed = bool(mode == "prefight" and verdict == "value_diverge" and allowance is not None
                                   and observed == allowance["live"] and expected == allowance["train"])
                    if allowed:
                        used_allowances.add((identity["row_id"], column))
                    counts["allowlisted_mismatch_count" if allowed else "mismatch_count"] += 1
                    record["differences"].append({"feature": column, "verdict": verdict,
                                                  "allowlisted_time_aging": allowed,
                                                  "live": observed, "train": expected})
        records.append(record)
    _require(used_allowances == set(allowances), "Original aging allowance does not match a measured difference.")
    passed = counts["compared_rows"] > 0 and not any(counts[key] for key in
        ("missing_rows", "failed_rows", "mismatch_count"))
    return {"passed": passed, "mode": mode, "required_rows": len(rows), **counts,
            "rows": records, "sampling_performed": False, "truncated": False,
            "measurement_scope": "retained_original_feature_outputs",
            "fresh_feature_build_performed": False,
            "errors": [] if passed else ["parity_or_coverage_failed"]}
