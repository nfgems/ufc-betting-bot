"""Bounded explicit monthly report production; no configured paths or model defaults.

This module orchestrates real read-only adapters when separately authorized to run.
Its injectable operations are test seams, not an alternative production evaluator.
Accepted original parity tables are compared here; collecting fresh replay tables
through source-capable feature builders is a separate, explicitly unsupported step.
"""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

from src.model.monthly_evidence import (
    REQUIRED_CHECKS, REPORT_KIND, canonical_bytes, evidence_sha256,
    parse_evaluator_json, validate_evaluator_report, validate_evaluator_request,
)

COMMAND_KIND = "monthly_evaluator_producer_command_v1"
ROW_KIND = "monthly_evaluator_rows_v1"
MAX_COMMAND_BYTES = 4 * 1024 * 1024
IDENTITY_FIELDS = ("row_id", "event_date", "fighter_a_id", "fighter_b_id")
MODEL_ROLES = ("primary", "logistic", "no_odds")


class MonthlyEvaluatorRefusal(ValueError):
    """Explicit accepted inputs or measured correctness could not be established."""


def _require(condition, message):
    if not condition:
        raise MonthlyEvaluatorRefusal(message)


def _safe_path(value):
    _require(isinstance(value, (str, Path)) and str(value), "explicit evaluator path missing")
    path = Path(value)
    _require(path.is_absolute() and ".." not in path.parts, "absolute evaluator path required without traversal")
    for component in (path, *path.parents):
        _require(not component.is_symlink(), "evaluator links are unsupported")
        if component.exists():
            attrs = getattr(component.lstat(), "st_file_attributes", 0)
            _require(not attrs & 0x400, "evaluator reparse points are unsupported")
    return path.resolve(strict=False)


def _read_pin(pin, limit):
    _require(isinstance(pin, dict) and set(pin) == {"path", "bytes", "sha256"}, "invalid evaluator original pin")
    _require(type(pin["bytes"]) is int and 0 < pin["bytes"] <= limit,
             "evaluator original exceeds its accepted finite bound")
    _require(isinstance(pin["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]),
             "invalid evaluator original hash")
    path = _safe_path(pin["path"])
    _require(path.is_file() and path.stat().st_size == pin["bytes"], "evaluator original size changed")
    with path.open("rb") as handle:
        body = handle.read(limit + 1)
    _require(len(body) == pin["bytes"] and hashlib.sha256(body).hexdigest() == pin["sha256"],
             "evaluator original identity changed")
    _safe_path(path)
    return body


def _utc(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, AttributeError, ValueError) as exc:
        raise MonthlyEvaluatorRefusal("invalid evaluator row/window date") from exc
    if result.tzinfo is None:
        _require(isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value),
                 "event dates must be date-only UTC midnight")
        result = result.replace(tzinfo=timezone.utc)
    _require(result.utcoffset().total_seconds() == 0, "evaluator dates must use UTC")
    return result


def _identity(row):
    _require(isinstance(row, dict), "evaluator row identity must be an object")
    _require(all(isinstance(row.get(key), str) and row[key] and row[key].strip() == row[key]
                 for key in IDENTITY_FIELDS), "complete stable oriented evaluator row identity required")
    _require(row["fighter_a_id"] != row["fighter_b_id"], "same fighter on both row sides")
    _require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["event_date"]), "row date must be date-only")
    _utc(row["event_date"])
    return {key: row[key] for key in IDENTITY_FIELDS}


def parse_row_selection(body, *, max_rows, role):
    """Validate every independently selected identity, including explicit exclusions."""
    value = parse_evaluator_json(body)
    _require(set(value) == {"schema", "kind", "rows"} and type(value["schema"]) is int
             and value["schema"] == 1 and value["kind"] == ROW_KIND, "unsupported evaluator row selection")
    rows = value["rows"]
    _require(isinstance(rows, list) and 0 < len(rows) <= max_rows, "evaluator row selection exceeds bound or is empty")
    identifiers, orientations = set(), set()
    for row in rows:
        _require(isinstance(row, dict) and set(row) == set(IDENTITY_FIELDS) | {"coverage_category", "exclusion"},
                 "evaluator selected row fields missing or unknown")
        identity = _identity(row)
        _require(identity["row_id"] not in identifiers, "duplicate evaluator row ID")
        oriented = (identity["event_date"], identity["fighter_a_id"], identity["fighter_b_id"])
        unordered = (identity["event_date"], *sorted((identity["fighter_a_id"], identity["fighter_b_id"])))
        _require(unordered not in orientations, "duplicate fight or swapped orientation in selected rows")
        identifiers.add(identity["row_id"])
        orientations.add(unordered)
        _require(isinstance(row["coverage_category"], str) and bool(row["coverage_category"].strip()),
                 "explicit row coverage category required")
        _require(row["exclusion"] is None or (role == "parity" and row["exclusion"] == "no_prior_processed_history"),
                 "unsupported row exclusion")
    return deepcopy(rows)


def parse_table(body, *, max_rows):
    """Parse supplied numbers and NaNs without adding feature columns or defaults."""
    import pandas as pd
    try:
        reader = csv.reader(io.StringIO(body.decode("utf-8-sig"), newline=""), strict=True)
        header = next(reader)
        _require(header and len(header) == len(set(header)) and all(key and key.strip() == key for key in header),
                 "empty, duplicate or ambiguous evaluator table columns")
        count = 0
        for row in reader:
            count += 1
            _require(len(row) == len(header), "evaluator CSV row width differs")
            _require(count <= max_rows, "evaluator table row bound exceeded")
        _require(count > 0, "empty evaluator table")
        identity_cols = {key for key in header if key.endswith("_id")} | ({"event_date", "fighter_a", "fighter_b"} & set(header))
        frame = pd.read_csv(io.BytesIO(body), encoding="utf-8-sig", low_memory=False,
            converters={key: str for key in identity_cols}, float_precision="round_trip")
        _require(frame.columns.tolist() == header and len(frame) == count, "evaluator CSV parser changed row/column identity")
        return frame
    except (UnicodeError, csv.Error, StopIteration, pd.errors.ParserError) as exc:
        raise MonthlyEvaluatorRefusal("malformed evaluator table") from exc


def select_table(frame, rows, *, feature_cols, window, targets):
    """Require exact complete table accounting and independently fixed chronology."""
    import pandas as pd
    _require(len(feature_cols) > 0 and len(feature_cols) == len(set(feature_cols)), "invalid ordered evaluator feature contract")
    required = set(IDENTITY_FIELDS) | set(feature_cols) | ({"target"} if targets else set())
    _require(required.issubset(frame.columns), "missing evaluator table identity/feature/target column")
    _require(len(frame) == len(rows), "fixed evaluator rows cannot be replaced, sampled or omitted")
    start, end = _utc(window["start_inclusive"]), _utc(window["end"])
    _require(type(window["end_inclusive"]) is bool and (start <= end if window["end_inclusive"] else start < end),
             "invalid evaluator comparison window")
    actual = {}
    for record in frame.to_dict(orient="records"):
        identity = _identity(record)
        _require(identity["row_id"] not in actual, "duplicate evaluator table row ID")
        actual[identity["row_id"]] = identity
    _require(set(actual) == {row["row_id"] for row in rows}, "evaluator selected/table row identities differ")
    for row in rows:
        _require(actual[row["row_id"]] == _identity(row), "evaluator row orientation/date differs from selection")
        event = _utc(row["event_date"])
        _require(start <= event and (event <= end if window["end_inclusive"] else event < end),
                 "future, leaking or out-of-window evaluator row")
    for name in feature_cols:
        numeric = pd.to_numeric(frame[name], errors="coerce")
        _require(not ((numeric.isna()) & frame[name].notna()).any(), "nonnumeric supplied evaluator feature")
        _require(all(pd.isna(item) or math.isfinite(float(item)) for item in numeric), "infinite evaluator feature")
    if targets:
        _require(frame["target"].notna().all() and frame["target"].isin([0, 1]).all(), "invalid evaluator binary targets")
    # Reorder rows only to the explicitly selected order; retain feature values/NaNs.
    return frame.set_index("row_id", drop=False).loc[[row["row_id"] for row in rows]].reset_index(drop=True)


def _exclusive_bytes(root, name, body, limit):
    _require(len(body) <= limit, "measured evaluator report exceeds finite bound")
    _require(Path(name).name == name, "unsafe evaluator output name")
    path = _safe_path(root / name)
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": str(path), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def _save_json(root, name, value, limit):
    return _exclusive_bytes(root, name, canonical_bytes(value), limit)


def _output_root(value, *, plan, request):
    root = _safe_path(value)
    protected = [_safe_path(plan["saved_root"]), _safe_path(request["candidate_dir"]),
                 Path(__file__).resolve().parents[2],
                 Path(request["input_receipt"]["monthly_input"]["root"])]
    # Accepted originals can live outside saved_root; output may never contain them.
    def visit(value):
        if isinstance(value, dict):
            if set(value) == {"path", "bytes", "sha256"}:
                # Original locations are provenance and overlap boundaries, not
                # an instruction to follow whatever occupies their former paths.
                path = Path(value["path"])
                _require(path.is_absolute() and ".." not in path.parts, "unsafe original evaluator location")
                protected.extend((path, path.parent))
            else:
                for item in value.values():
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(plan)
    for selected in protected:
        _require(root != selected and root not in selected.parents and selected not in root.parents,
                 "evaluator output overlaps source, candidate or accepted evidence")
    return root


def _record_original(root, stage, value, limit):
    """Keep malformed operation bytes before attempting semantic normalization."""
    if isinstance(value, bytes):
        body = value
    elif isinstance(value, str):
        body = value.encode("utf-8")
    else:
        # Original malformed JSON (including nonfinite values) must survive even
        # though strict semantic parsing subsequently refuses it.
        try:
            body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                              allow_nan=True).encode("utf-8")
        except (TypeError, ValueError):
            body = repr(value).encode("utf-8")
    if len(body) > limit:
        _exclusive_bytes(root, f"{stage}.partial.bin", body[:limit], limit)
        _save_json(root, f"{stage}.truncation.json", {"original_bytes": len(body),
            "original_sha256": hashlib.sha256(body).hexdigest(), "retained_bytes": limit, "truncated": True}, limit)
        raise MonthlyEvaluatorRefusal("measured evaluator report exceeds finite bound; original prefix retained")
    pin = _exclusive_bytes(root, f"{stage}.original.json", body, limit)
    parsed = parse_evaluator_json(body)
    return parsed, pin


def _successful(value, label):
    from src.model.monthly_evidence import validate_correctness_mapping
    try:
        validate_correctness_mapping(value)
    except ValueError as exc:
        raise MonthlyEvaluatorRefusal(f"{label} measured correctness failed") from exc


def validate_score_measurement(value, *, rows, features, frame):
    """A scalar success flag cannot erase a missing prediction or changed orientation."""
    _require(isinstance(value, dict) and set(value) == {"passed", "role", "feature_columns", "rows", "metrics", "errors"},
             "score measurement fields missing or unknown")
    _successful({key: item for key, item in value.items() if key != "metrics"}, "scoring")
    _require(value["role"] in {"candidate", "historical", "reference"} and value["feature_columns"] == features,
             "score model role/features differ")
    measured = value["rows"]
    _require(isinstance(measured, list) and len(measured) == len(rows), "missing scored required row")
    expected_targets = dict(zip(frame["row_id"], frame["target"]))
    for actual, expected in zip(measured, rows):
        _require(isinstance(actual, dict) and set(actual) == set(IDENTITY_FIELDS) | {"target", "predictions"},
                 "score row fields missing or unknown")
        _require(_identity(actual) == _identity(expected), "score row order/orientation differs")
        _require(type(actual["target"]) is int and actual["target"] in (0, 1)
                 and actual["target"] == expected_targets[actual["row_id"]], "score target changed")
        probs = actual["predictions"]
        _require(isinstance(probs, dict) and set(probs) == set(MODEL_ROLES), "score artifact role missing")
        _require(all(type(prob) in (int, float) and math.isfinite(prob) and 0 <= prob <= 1 for prob in probs.values()),
                 "invalid model probability")
    _require(isinstance(value["metrics"], dict) and set(value["metrics"]) == set(MODEL_ROLES)
             and all(isinstance(metrics, dict) and metrics for metrics in value["metrics"].values()),
             "measured metrics missing; defaults are not evidence")
    import pandas as pd
    from src.model.refreshed_v4_eval import prediction_metrics
    for role in MODEL_ROLES:
        calculated = prediction_metrics(pd.DataFrame({"target": [row["target"] for row in measured],
            "prob_a": [row["predictions"][role] for row in measured]}))
        metrics = value["metrics"][role]
        _require(set(metrics) == set(calculated) and all(type(metrics[key]) in (int, float)
                 and math.isfinite(metrics[key]) and math.isclose(metrics[key], result, rel_tol=1e-12, abs_tol=1e-12)
                 for key, result in calculated.items()), "score metrics differ from measured row probabilities")
    canonical_bytes(value)
    return deepcopy(value)


def validate_parity_measurement(value, *, rows, mode, aging_allowances=None):
    fields = {"passed", "mode", "required_rows", "compared_rows", "excluded_rows", "missing_rows", "failed_rows",
              "mismatch_count", "allowlisted_mismatch_count", "rows", "sampling_performed", "truncated", "errors",
              "measurement_scope", "fresh_feature_build_performed"}
    _require(isinstance(value, dict) and set(value) == fields, "parity measurement fields missing or unknown")
    _require(value["mode"] == mode and value["sampling_performed"] is False and value["truncated"] is False,
             "parity mode/sampling/truncation differs")
    _require(value["measurement_scope"] == "retained_original_feature_outputs"
             and value["fresh_feature_build_performed"] is False, "unsupported parity measurement scope")
    _require(isinstance(value["rows"], list) and len(value["rows"]) == len(rows), "parity omitted required rows")
    counts = {name: 0 for name in ("compared", "excluded", "missing", "failed")}
    forbidden = allowed = 0
    accepted_aging = {(item["row_id"], item["feature"]): item for item in (aging_allowances or [])}
    from src.model.feature_parity import expand_time_aged
    for actual, expected in zip(value["rows"], rows):
        _require(isinstance(actual, dict) and set(actual) == set(IDENTITY_FIELDS) | {
            "disposition", "coverage_category", "reason", "differences"}, "invalid parity row record")
        _require(_identity(actual) == _identity(expected) and actual["coverage_category"] == expected["coverage_category"],
                 "parity row identity/orientation/coverage category differs")
        disposition = actual["disposition"]
        _require(disposition in counts, "unknown parity disposition")
        counts[disposition] += 1
        _require(isinstance(actual["differences"], list), "parity differences missing")
        if disposition == "excluded":
            _require(mode == "prefight" and expected["exclusion"] == "no_prior_processed_history"
                     and actual["reason"] == expected["exclusion"] and not actual["differences"],
                     "unapproved parity exclusion")
        elif disposition == "compared":
            _require(actual["reason"] is None, "compared parity row has contradictory reason")
        for difference in actual["differences"]:
            _require(isinstance(difference, dict) and set(difference) == {
                "feature", "verdict", "live", "train", "allowlisted_time_aging"}, "invalid parity difference record")
            aging = difference["allowlisted_time_aging"]
            _require(type(aging) is bool, "invalid parity aging flag")
            if aging:
                _require(mode == "prefight" and difference["verdict"] == "value_diverge"
                         and difference["feature"] in expand_time_aged([difference["feature"]])
                         and all(type(difference[key]) in (int, float) and math.isfinite(difference[key])
                                 for key in ("live", "train")), "invalid prefight aging allowance")
                allowance = accepted_aging.pop((actual["row_id"], difference["feature"]), None)
                _require(allowance is not None and allowance["live"] == difference["live"]
                         and allowance["train"] == difference["train"], "parity aging difference lacks original acceptance")
                allowed += 1
            else:
                forbidden += 1
    expected_counts = {"required_rows": len(rows), **{f"{name}_rows": count for name, count in counts.items()},
                       "mismatch_count": forbidden, "allowlisted_mismatch_count": allowed}
    _require(all(type(value[key]) is int and value[key] == count for key, count in expected_counts.items()),
             "parity aggregate omitted or altered measured rows/differences")
    _require(counts["compared"] > 0 and counts["missing"] == counts["failed"] == forbidden == 0,
             "parity has missing/failed rows or forbidden mismatches")
    _require(not accepted_aging, "original parity aging allowance was unused")
    _successful(value, "parity")
    return deepcopy(value)


def _bundle_hashes(bundle):
    return {**{role: bundle["models"][role]["sha256"] for role in MODEL_ROLES}, "spec": bundle["spec"]["sha256"]}


def _training_rows(frame, *, features, window):
    _require({"event_date", "fighter_a_id", "fighter_b_id", "target"}.issubset(frame.columns),
             "training provenance requires observed rows with stable fighter identities")
    frame = frame.copy()
    if "row_id" not in frame.columns:
        frame["row_id"] = [evidence_sha256({key: record[key] for key in IDENTITY_FIELDS if key != "row_id"})
                           for record in frame.to_dict(orient="records")]
    rows = [{**_identity(record), "coverage_category": "original_training_row", "exclusion": None}
            for record in frame.to_dict(orient="records")]
    rows = parse_row_selection(canonical_bytes({"schema": 1, "kind": ROW_KIND, "rows": rows}),
                               max_rows=len(rows), role="training")
    select_table(frame, rows, feature_cols=features,
                 window={"start_inclusive": window["start_inclusive"], "end": window["end_exclusive"],
                         "end_inclusive": False}, targets=False)
    _require(frame["target"].dropna().isin([0, 1]).all(), "invalid original training target")
    from src.model.training_preflight import prepare_train_test
    trained, _, columns = prepare_train_test(frame, cutoff_date=window["end_exclusive"], min_fights=2,
        feature_cols=features, start_date=window["start_inclusive"], end_date=None)
    _require(columns == features and len(trained) > 0, "original training table has no eligible observed rows")
    selected = {item["row_id"]: item for item in rows}
    actual_rows = [selected[value] for value in trained["row_id"]]
    used = set(trained["row_id"])
    excluded = []
    import pandas as pd
    for record in frame.to_dict(orient="records"):
        if record["row_id"] in used:
            continue
        if "a_num_fights" in frame.columns and "b_num_fights" in frame.columns and not (
                record["a_num_fights"] >= 2 and record["b_num_fights"] >= 2):
            reason = "insufficient_prior_bouts"
        elif pd.isna(record["target"]):
            reason = "unknown_target"
        elif all(pd.isna(record[name]) for name in features):
            reason = "all_features_missing"
        else:
            raise MonthlyEvaluatorRefusal("unaccounted ordinary training exclusion")
        excluded.append({**_identity(record), "reason": reason})
    return actual_rows, {"required_input_rows": len(rows), "observed_training_rows": len(actual_rows),
                         "excluded_input_rows": excluded, "input_rows": rows}


def preflight_external_inputs(plan, *, candidate_spec, policy_binding):
    """Check accepted external inputs before a future candidate exists or is fitted."""
    from src.model.monthly_evaluator_adapters import validate_primary_spec
    limit, max_rows = plan["bounds"]["max_file_bytes"], plan["bounds"]["max_rows"]
    policy = policy_binding
    candidate_window = {"start_inclusive": policy["modeling_start_date"], "end_exclusive": policy["train_cutoff_date"]}
    candidate_variants = validate_primary_spec(candidate_spec, candidate_window)
    from src.model.training_preflight import resolved_spec_identity
    from src.model.training_spec import NamedModelTrainingSpec, resolve_named_training_spec
    _require(policy["training_spec"] == {"name": candidate_spec["name"],
        "sha256": resolved_spec_identity(NamedModelTrainingSpec(**candidate_spec))}
        and policy["named_training_spec"] == {"name": candidate_spec["name"],
            "sha256": resolved_spec_identity(resolve_named_training_spec(candidate_spec["name"]))},
        "candidate resolved recipe identity differs from original policy acceptance")
    specs, features = {"candidate": candidate_spec}, {"candidate": {key: value["feature_cols"] for key, value in candidate_variants.items()}}
    training_windows, training_records = {"candidate": candidate_window}, {}
    for role in ("historical", "reference"):
        bundle = plan["bundles"][role]
        spec = parse_evaluator_json(_read_pin(bundle["spec"], limit))
        evidence = parse_evaluator_json(_read_pin(bundle["training_evidence"], limit))
        _require(set(evidence) == {"schema", "kind", "model_hashes", "spec_sha256", "training_rows_sha256", "window", "row_ids"}
                 and type(evidence["schema"]) is int and evidence["schema"] == 1
                 and evidence["kind"] == "accepted_training_evidence_v1", "unsupported original training evidence")
        _require(evidence["model_hashes"] == {key: bundle["models"][key]["sha256"] for key in MODEL_ROLES}
                 and evidence["spec_sha256"] == bundle["spec"]["sha256"]
                 and evidence["training_rows_sha256"] == bundle["training_rows"]["sha256"],
                 "original training evidence is unbound or swapped")
        window = evidence["window"]
        if role == "historical":
            _require(window == {"start_inclusive": plan["windows"]["train_start_inclusive"],
                                "end_exclusive": plan["windows"]["train_end_exclusive"]},
                     "historical replica must train only before 2025")
        variants = validate_primary_spec(spec, window)
        role_features = {key: value["feature_cols"] for key, value in variants.items()}
        frame = parse_table(_read_pin(bundle["training_rows"], limit), max_rows=max_rows)
        rows, training_accounting = _training_rows(frame, features=role_features["primary"], window=window)
        _require(evidence["row_ids"] == [row["row_id"] for row in rows], "training evidence dropped/reordered actual rows")
        specs[role], features[role], training_windows[role] = spec, role_features, window
        training_records[role] = {"window": window, "rows": rows, "original_evidence": evidence,
                                  "training_rows": bundle["training_rows"], "accounting": training_accounting}
    _require(features["candidate"] == features["historical"] == features["reference"],
             "comparison artifacts have incompatible feature contracts")
    tables, rows_by_role = {}, {}
    for role, selection in plan["row_sets"].items():
        rows = parse_row_selection(_read_pin(selection["rows"], limit), max_rows=max_rows, role=role)
        frame = parse_table(_read_pin(selection["table"], limit), max_rows=max_rows)
        tables[role] = select_table(frame, rows, feature_cols=features["candidate"]["primary"],
                                   window=_window(plan, role), targets=role != "parity")
        rows_by_role[role] = rows
    _require(set(tables) == {"validation", "evaluation", "comparison", "parity"}, "fixed evaluator population role missing")
    historical_keys = {(row["event_date"], *sorted((row["fighter_a_id"], row["fighter_b_id"])))
                       for row in training_records["historical"]["rows"]}
    for role in ("validation", "evaluation"):
        _require(not historical_keys & {(row["event_date"], *sorted((row["fighter_a_id"], row["fighter_b_id"])))
                                       for row in rows_by_role[role]}, "holdout fight leaked into historical training")
    parity_inputs = plan["parity_inputs"]
    parity_evidence = parse_evaluator_json(_read_pin(parity_inputs["evidence"], limit))
    expected_parity = {"schema": 1, "kind": "accepted_parity_replay_evidence_v1",
        "policy_sha256": policy["policy"]["sha256"], "feature_columns": features["candidate"]["primary"],
        "table_hashes": {**{key: parity_inputs[key]["sha256"] for key in ("history", "exact", "prefight")},
                         "oracle": plan["row_sets"]["parity"]["table"]["sha256"]}}
    _require(set(parity_evidence) == set(expected_parity) | {"replay_records", "aging_allowances"}
             and all(parity_evidence[key] == value for key, value in expected_parity.items()),
        "original parity evidence does not bind selected output tables/policy/features")
    parity = {key: parse_table(_read_pin(parity_inputs[key], limit), max_rows=max_rows)
              for key in ("history", "exact", "prefight")}
    _validate_replay_records(parity_evidence["replay_records"], rows=rows_by_role["parity"], tables=parity)
    _require(isinstance(parity_evidence["aging_allowances"], dict)
             and set(parity_evidence["aging_allowances"]) == {"exact", "prefight"}, "explicit original parity aging policy missing")
    from src.model.monthly_evaluator_adapters import compare_parity
    measured_parity = {}
    for mode in ("exact", "prefight"):
        record = compare_parity(oracle_frame=tables["parity"], replay_frame=parity[mode], history_frame=parity["history"],
            rows=rows_by_role["parity"], feature_cols=features["candidate"]["primary"], window=_window(plan, "parity"), mode=mode,
            aging_allowances=parity_evidence["aging_allowances"][mode])
        measured_parity[mode] = validate_parity_measurement(record, rows=rows_by_role["parity"], mode=mode,
            aging_allowances=parity_evidence["aging_allowances"][mode])
    # Adapter preflight checks the original as-of dates and explicit exclusions
    # before comparison; missing replay rows are preserved as measured failures.
    return {"specs": specs, "features": features, "training_windows": training_windows,
            "training_records": training_records, "tables": tables, "rows": rows_by_role,
            "parity": parity, "parity_evidence": parity_evidence, "parity_measurements": measured_parity}


def _prepare_operations(plan, request):
    _require(plan["request_sha256"] == evidence_sha256(request), "realized evaluator plan belongs to another request")
    _require(set(plan["bundles"]) == {"candidate", "historical", "reference"}, "model bundle role missing or unknown")
    _recheck_inputs(plan)
    bundle = plan["bundles"]["candidate"]
    expected = {"kind": "monthly_candidate_request_training_v1", "request_sha256": evidence_sha256(request),
                "input_receipt": request["input_receipt"]}
    _require(bundle["training_evidence"] == expected, "candidate original training request changed")
    _require(_bundle_hashes(bundle) == request["model_hashes"], "candidate artifact identities differ from request")
    _require(len({request["model_hashes"][key] for key in MODEL_ROLES}) == 3, "candidate estimator roles share bytes")
    _require(all(plan["bundles"]["historical"]["models"][key]["sha256"] != request["model_hashes"][key]
                 for key in MODEL_ROLES), "monthly refit cannot stand in for historical replica")
    spec = parse_evaluator_json(_read_pin(bundle["spec"], plan["bounds"]["max_file_bytes"]))
    prepared = preflight_external_inputs(plan, candidate_spec=spec, policy_binding=request["bindings"]["monthly_policy"])
    from src.model.monthly_policy import feature_order_sha256
    features = prepared["features"]["candidate"]
    _require(feature_order_sha256(features["primary"]) == request["contract"]["primary_feature_sha256"]
             and feature_order_sha256(features["no_odds"]) == request["contract"]["no_odds_feature_sha256"],
             "accepted evaluator feature contract differs from original request")
    frame = parse_table(_read_pin(bundle["training_rows"], plan["bounds"]["max_file_bytes"]), max_rows=plan["bounds"]["max_rows"])
    rows, accounting = _training_rows(frame, features=features["primary"], window=prepared["training_windows"]["candidate"])
    prepared["training_records"]["candidate"] = {"window": prepared["training_windows"]["candidate"], "rows": rows,
        "original_evidence": expected, "training_rows": bundle["training_rows"], "accounting": accounting}
    return prepared


def _validate_replay_records(records, *, rows, tables):
    _require(isinstance(records, dict) and set(records) == {"exact", "prefight"}, "original replay outcomes missing")
    for mode in ("exact", "prefight"):
        actual = records[mode]
        _require(isinstance(actual, list) and len(actual) == len(rows), "original replay omitted required outcome rows")
        _require(set(IDENTITY_FIELDS) | {"as_of_date"} <= set(tables[mode].columns), "original replay identity/asof columns missing")
        table_rows = {}
        for record in tables[mode].to_dict(orient="records"):
            identity = _identity(record)
            _require(identity["row_id"] not in table_rows, "original replay table repeats row")
            table_rows[identity["row_id"]] = record
        built_ids = set()
        for record, selected in zip(actual, rows):
            _require(isinstance(record, dict) and set(record) == set(IDENTITY_FIELDS) | {"as_of_date", "outcome", "error"},
                     "original replay outcome fields missing or unknown")
            _require(_identity(record) == _identity(selected), "original replay outcome orientation differs")
            as_of = (_utc(selected["event_date"]) - timedelta(days=mode == "prefight")).date().isoformat()
            _require(record["as_of_date"] == as_of, "original replay outcome cutoff differs")
            _require(record["outcome"] in {"built", "excluded", "failed"}, "unknown original replay outcome")
            _require(record["outcome"] != "failed", "original replay contains a failed feature build")
            _require(record["error"] is None, "original replay has a swallowed build error")
            if record["outcome"] == "excluded":
                _require(mode == "prefight" and selected["exclusion"] == "no_prior_processed_history"
                         and selected["row_id"] not in table_rows, "original replay exclusion conflicts")
            else:
                measured = table_rows.get(selected["row_id"])
                _require(measured is not None and _identity(measured) == _identity(selected)
                         and measured["as_of_date"] == as_of, "built replay outcome lacks exact original feature row")
                built_ids.add(selected["row_id"])
        _require(set(table_rows) == built_ids, "unaccounted original replay rows")


def _assemble_report(request, plan, prepared, scores, parity, originals):
    windows = plan["windows"]
    report = {"schema_version": 2, "kind": REPORT_KIND, "passed": True,
        "request_sha256": evidence_sha256(request), "bindings": deepcopy(request["bindings"]),
        "model_hashes": deepcopy(request["model_hashes"]), "checks": {name: True for name in REQUIRED_CHECKS},
        "evidence": {}, "comparison_with_current_live": {}, "owner_review_required": True,
        "auto_promotion_performed": False, "live_model_replaced": False}
    def section(name, specific, stages, account):
        report["evidence"][name] = _section(name, specific, request=request,
            originals={key: originals[key] for key in stages}, account=account)
    def training_provenance(role):
        # Full original records remain in selected-plan/request and saved input
        # evidence. Their arbitrary metadata keys are not correctness decisions.
        record = {key: deepcopy(value) for key, value in prepared["training_records"][role].items()
                  if key != "original_evidence"}
        record["original_evidence_reference"] = ({"request_sha256": evidence_sha256(request)} if role == "candidate"
            else deepcopy(plan["bundles"][role]["training_evidence"]))
        return record
    section("production_load", {"loaded_model_hashes": request["model_hashes"]}, ["load-candidate"],
            {"required_model_roles": list(MODEL_ROLES), "loaded_model_roles": list(MODEL_ROLES)})
    section("fixed_evaluation", {"windows": windows, "honest_historical_replicas": True,
        "historical_replica_model_hashes": _bundle_hashes(plan["bundles"]["historical"]),
        "replica_training_window": {"train_start_inclusive": windows["train_start_inclusive"],
                                     "train_end_exclusive": windows["train_end_exclusive"]},
        "evaluation_coverage": {"start_inclusive": windows["evaluation_start_inclusive"],
                                "end_inclusive": windows["common_supported_end_utc"]},
        "report_guided_selection_performed": False},
        ["load-historical", "score-historical-validation", "score-historical-evaluation"],
        {"training_provenance": training_provenance("historical"),
         "validation_rows": prepared["rows"]["validation"], "evaluation_rows": prepared["rows"]["evaluation"],
         "validation_report_only": True, "sampling_performed": False, "truncated": False})
    for mode in ("exact", "prefight"):
        section(f"{mode}_parity", {"compared_rows": parity[mode]["compared_rows"], "mismatch_count": parity[mode]["mismatch_count"]},
                [f"parity-{mode}"], {key: value for key, value in parity[mode].items() if key not in {"passed", "errors"}})
    coverage = {role: {"required_rows": len(rows), "compared_rows": len(rows), "excluded_rows": 0,
                       "missing_rows": 0, "failed_rows": 0, "rows": rows}
                for role, rows in prepared["rows"].items() if role != "parity"}
    coverage["parity"] = {mode: {key: record[key] for key in ("required_rows", "compared_rows", "excluded_rows",
                            "missing_rows", "failed_rows", "rows")} for mode, record in parity.items()}
    section("coverage_report", {"evaluated_rows": len(prepared["rows"]["evaluation"]), "missing_required_rows": 0},
            list(originals), {"populations": coverage, "sampling_performed": False, "truncated": False})
    comparison = {"candidate": scores["score-candidate-comparison"]["metrics"],
                  "reference": scores["score-reference-comparison"]["metrics"]}
    report["comparison_with_current_live"] = comparison
    section("current_live_comparison", {"reference_model_hashes": _bundle_hashes(plan["bundles"]["reference"]),
        "comparison_scope": "descriptive_only", "metrics": comparison},
        ["load-reference", "score-candidate-comparison", "score-reference-comparison"],
        {"rows": prepared["rows"]["comparison"], "reference_training_provenance": training_provenance("reference"),
         "candidate_training_provenance": training_provenance("candidate"),
         "retrospective_comparison": True, "selection_performed": False})
    return report


def _window(plan, role):
    windows = plan["windows"]
    if role == "validation":
        return {"start_inclusive": windows["validation_start_inclusive"],
                "end": windows["validation_end_exclusive"], "end_inclusive": False}
    return {"start_inclusive": windows["evaluation_start_inclusive"],
            "end": windows["common_supported_end_utc"], "end_inclusive": True}


def _all_pins(value):
    if isinstance(value, dict):
        if set(value) == {"path", "bytes", "sha256"}:
            yield value
        else:
            for item in value.values():
                yield from _all_pins(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_pins(item)


def _recheck_inputs(plan):
    # The saved plan preserves original paths as provenance. Only normalized
    # operation pins are reopened; disappearance of former external paths is OK.
    sections = [plan["row_sets"], plan["parity_inputs"]]
    for role, bundle in plan["bundles"].items():
        sections.append({key: value for key, value in bundle.items()
                         if role != "candidate" or key != "training_evidence"})
    for section in sections:
        for pin in _all_pins(section):
            _read_pin(pin, plan["bounds"]["max_file_bytes"])


def _section(name, specific, *, request, originals, account):
    body = {"check": name, "request_sha256": evidence_sha256(request),
            "model_hashes": deepcopy(request["model_hashes"]), "passed": True, "errors": [],
            "checks": {"measured_originals": {"passed": True, "originals": deepcopy(originals),
                        "accounting": deepcopy(account)}}, **specific}
    encoded = canonical_bytes(body)
    return {"identity": f"{name}:{evidence_sha256(request)}", "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(), "body": body}


def _inventory(root, *, limit, total_limit):
    result = []
    members = []
    for path in root.iterdir():
        _require(len(members) < 33, "evaluator output entry bound exceeded")
        members.append(path)
    for path in sorted(members):
        _safe_path(path)
        _require(path.is_file(), "unexpected evaluator output entry")
        if path.name == "complete.json":
            continue
        _require(path.stat().st_size <= limit, "evaluator retained file bound exceeded")
        body = path.read_bytes()
        result.append({"name": path.name, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
        _require(sum(item["bytes"] for item in result) <= total_limit, "evaluator retained output bound exceeded")
    return result


def load_completed_report(output_root, *, request, plan):
    """Duplicate completion validates saved originals and never redispatches models."""
    root = _output_root(output_root, plan=plan, request=request)
    completion_path = root / "complete.json"
    _require(completion_path.is_file(), "previous evaluator attempt is incomplete; preserve it and choose a fresh output root")
    _safe_path(completion_path)
    _require(completion_path.stat().st_size <= plan["bounds"]["max_report_bytes"], "oversized evaluator completion")
    completion = parse_evaluator_json(completion_path.read_bytes())
    _require(set(completion) == {"schema", "kind", "request_sha256", "inventory"}
             and type(completion["schema"]) is int and completion["schema"] == 1
             and completion["kind"] == "complete_monthly_evaluator_output_v1"
             and completion["request_sha256"] == evidence_sha256(request), "evaluator completion binding conflict")
    _require(completion["inventory"] == _inventory(root, limit=plan["bounds"]["max_report_bytes"],
             total_limit=plan["bounds"]["max_total_bytes"]), "saved evaluator output inventory changed")
    _require(parse_evaluator_json((root / "request.json").read_bytes()) == request,
             "saved evaluator original request changed")
    _recheck_inputs(plan)
    report = parse_evaluator_json((root / "report.json").read_bytes())
    _require(parse_evaluator_json((root / "selected-plan.json").read_bytes()) == plan, "saved evaluator plan differs")
    prepared = _prepare_operations(plan, request)
    originals, scores, parity = {}, {}, {}
    def original(stage):
        path = root / f"{stage}.original.json"
        _require(path.is_file(), "required measured original missing")
        body = path.read_bytes()
        originals[stage] = {"path": str(path), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        return parse_evaluator_json(body)
    for role in ("candidate", "historical", "reference"):
        _validate_load(original(f"load-{role}"), role=role, spec=prepared["specs"][role],
            features=prepared["features"][role], training_window=prepared["training_windows"][role],
            training_row_count=len(prepared["training_records"][role]["rows"]))
    for role, row_role in (("historical", "validation"), ("historical", "evaluation"),
                           ("candidate", "comparison"), ("reference", "comparison")):
        stage = f"score-{role}-{row_role}"
        record = original(stage)
        _require(record.get("role") == role, "saved score result substituted model role")
        scores[stage] = validate_score_measurement(record, rows=prepared["rows"][row_role],
            features=prepared["features"][role], frame=prepared["tables"][row_role])
    for mode in ("exact", "prefight"):
        parity[mode] = validate_parity_measurement(original(f"parity-{mode}"), rows=prepared["rows"]["parity"], mode=mode,
            aging_allowances=prepared["parity_evidence"]["aging_allowances"][mode])
        _require(parity[mode] == prepared["parity_measurements"][mode], "saved parity differs from selected original tables")
    expected = _assemble_report(request, plan, prepared, scores, parity, originals)
    _require(report == expected, "saved report differs from original measured results")
    return validate_evaluator_report(report, request=request)


def verify_saved_report(output_root, *, request):
    """Ordinary candidate/runtime recovery uses saved accepted inputs and outputs."""
    from src.model.monthly_evaluator_inputs import load_evaluator_inputs
    selected = validate_evaluator_request(request)
    _require("producer_inputs" in selected, "completed producer report lacks accepted inputs")
    plan = load_evaluator_inputs(selected["producer_inputs"], request=selected)
    return load_completed_report(output_root, request=selected, plan=plan)


def _validate_load(value, *, role, spec, features, training_window, training_row_count):
    _require(isinstance(value, dict) and set(value) == {"passed", "role", "models", "errors"},
             "load measurement fields missing or unknown")
    _successful(value, "production load")
    _require(value["role"] == role and isinstance(value["models"], list) and len(value["models"]) == 3,
             "load measurement role/model count differs")
    from src.model.production_bundle import _expected_no_odds_spec_payload
    expected_specs = {"primary": spec, "logistic": spec, "no_odds": _expected_no_odds_spec_payload(spec)}
    seen = set()
    for item in value["models"]:
        _require(isinstance(item, dict) and set(item) == {"model_role", "spec", "feature_columns", "training_window", "training_rows"},
                 "incomplete load model measurement")
        model_role = item["model_role"]
        _require(model_role in MODEL_ROLES and model_role not in seen, "swapped or duplicate loaded artifact role")
        seen.add(model_role)
        _require(item["spec"] == expected_specs[model_role] and item["feature_columns"] == features[model_role]
                 and item["training_window"] == training_window, "loaded spec/features/training window conflicts")
        from src.model.monthly_evaluator_adapters import validate_training_accounting
        validate_training_accounting(item["training_rows"], training_row_count=training_row_count, feature_cols=features[model_role])
    return value


def produce_evaluator_report(*, request, output_root, operations=None):
    """Produce and retain six strict measured evidence sections from accepted inputs.

    Default adapters perform genuine model operations and therefore require the
    caller's separate execution authority. Offline tests supply explicitly labeled
    synthetic result records; those results cannot prove real model correctness.
    """
    from src.model.monthly_evaluator_inputs import load_evaluator_inputs
    selected = validate_evaluator_request(request)
    _require("producer_inputs" in selected, "accepted producer inputs required before evaluator dispatch")
    plan = load_evaluator_inputs(selected["producer_inputs"], request=selected)
    root = _output_root(output_root, plan=plan, request=selected)
    if root.exists():
        return load_completed_report(root, request=selected, plan=plan)
    # Parse and check every selected input before any output writer or model boundary.
    prepared = _prepare_operations(plan, selected)
    if operations is None:
        from src.model import monthly_evaluator_adapters as operations
    _recheck_inputs(plan)
    root.mkdir(parents=True, exist_ok=False)
    limit = plan["bounds"]["max_report_bytes"]
    originals = {}
    stage = "request"
    try:
        _save_json(root, "request.json", selected, limit)
        _save_json(root, "selected-plan.json", plan, limit)
        loaded = {}
        load_records = {}
        for role in ("candidate", "historical", "reference"):
            stage = f"load-{role}"
            bundle = plan["bundles"][role]
            _recheck_inputs(plan)
            result = operations.load_bundle(model_paths={key: Path(bundle["models"][key]["path"]) for key in MODEL_ROLES},
                primary_spec=prepared["specs"][role], training_window=prepared["training_windows"][role], role=role,
                training_row_count=len(prepared["training_records"][role]["rows"]))
            record, pin = _record_original(root, stage, result.measurement, limit)
            originals[stage] = pin
            _validate_load(record, role=role, spec=prepared["specs"][role], features=prepared["features"][role],
                           training_window=prepared["training_windows"][role],
                           training_row_count=len(prepared["training_records"][role]["rows"]))
            _recheck_inputs(plan)
            loaded[role], load_records[role] = result, record
        scores = {}
        for role, row_role in (("historical", "validation"), ("historical", "evaluation"),
                               ("candidate", "comparison"), ("reference", "comparison")):
            stage = f"score-{role}-{row_role}"
            _recheck_inputs(plan)
            value = operations.score_bundle(loaded=loaded[role], frame=prepared["tables"][row_role],
                rows=prepared["rows"][row_role], window=_window(plan, row_role))
            record, pin = _record_original(root, stage, value, limit)
            originals[stage] = pin
            _require(record.get("role") == role, "score result substituted model role")
            scores[stage] = validate_score_measurement(record, rows=prepared["rows"][row_role],
                features=prepared["features"][role], frame=prepared["tables"][row_role])
            _recheck_inputs(plan)
        parity = {}
        for mode in ("exact", "prefight"):
            stage = f"parity-{mode}"
            _recheck_inputs(plan)
            value = operations.compare_parity(oracle_frame=prepared["tables"]["parity"],
                replay_frame=prepared["parity"][mode], history_frame=prepared["parity"]["history"],
                rows=prepared["rows"]["parity"], feature_cols=prepared["features"]["candidate"]["primary"],
                window=_window(plan, "parity"), mode=mode,
                aging_allowances=prepared["parity_evidence"]["aging_allowances"][mode])
            record, pin = _record_original(root, stage, value, limit)
            originals[stage] = pin
            parity[mode] = validate_parity_measurement(record, rows=prepared["rows"]["parity"], mode=mode,
                aging_allowances=prepared["parity_evidence"]["aging_allowances"][mode])
            _require(parity[mode] == prepared["parity_measurements"][mode], "measured parity omitted or altered original table differences")
            _recheck_inputs(plan)
        stage = "report"
        report = _assemble_report(selected, plan, prepared, scores, parity, originals)
        _save_json(root, "report.json", report, limit)
        validate_evaluator_report(report, request=selected)
        _recheck_inputs(plan)
        inventory = _inventory(root, limit=limit, total_limit=plan["bounds"]["max_total_bytes"])
        _require(sum(item["bytes"] for item in inventory) <= plan["bounds"]["max_total_bytes"],
                 "evaluator retained output bound exceeded")
        _save_json(root, "complete.json", {"schema": 1, "kind": "complete_monthly_evaluator_output_v1",
            "request_sha256": evidence_sha256(selected), "inventory": inventory}, limit)
        return report
    except BaseException as exc:
        # Interruptions consume this output root. Available originals are never
        # replaced or removed, even when final failure-record writing also fails.
        failure = {"schema": 1, "kind": "failed_monthly_evaluator_output_v1", "request_sha256": evidence_sha256(selected),
                   "stage": stage, "exception_type": type(exc).__name__, "message": str(exc)[:8192],
                   "retained_originals": originals}
        try:
            _save_json(root, "failure.json", failure, limit)
        except (OSError, ValueError):
            pass
        raise


def execute_command(command):
    _require(isinstance(command, dict) and set(command) == {"schema", "kind", "request", "output_root"}
             and type(command["schema"]) is int and command["schema"] == 1 and command["kind"] == COMMAND_KIND,
             "unsupported evaluator producer command")
    return produce_evaluator_report(request=command["request"], output_root=command["output_root"])


def evaluate_monthly_candidate(*, candidate_dir, input_receipt, contract, producer_inputs, producer_output_root):
    """Ordinary direct candidate adapter with explicit saved inputs/output only."""
    from src.model.monthly_candidate import _evaluator_request
    request = _evaluator_request(Path(candidate_dir), input_receipt=input_receipt, contract=contract,
                                 producer_inputs=producer_inputs)
    return produce_evaluator_report(request=request, output_root=producer_output_root)


evaluate_monthly_candidate.requires_producer_inputs = True


def main(argv=None, *, stdin=None, stdout=None, stderr=None):
    """Thin bounded JSON-stdin adapter; no active CLI or subprocess dispatch."""
    import argparse
    parser = argparse.ArgumentParser(description="Produce explicitly accepted monthly evaluator evidence")
    parser.parse_args(argv)
    stdin, stdout, stderr = stdin or sys.stdin, stdout or sys.stdout, stderr or sys.stderr
    try:
        wire = stdin.read(MAX_COMMAND_BYTES + 1)
        _require(len(wire.encode("utf-8") if isinstance(wire, str) else wire) <= MAX_COMMAND_BYTES,
                 "evaluator stdin command exceeds bound")
        command = parse_evaluator_json(wire)
        report = execute_command(command)
        stdout.write(canonical_bytes(report).decode("utf-8") + "\n")
        return 0
    except Exception as exc:
        stderr.write(f"Monthly evaluator refused: {type(exc).__name__}: {str(exc)[:8192]}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
