"""Independently selected evaluator originals, retained before model dispatch.

Accepting a template never requires an unknowable future candidate hash. The
request binds that future candidate to the already selected external originals.
No path discovery, active model alias, cleanup or model operation belongs here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Mapping

from src.model.monthly_evidence import canonical_bytes, evidence_sha256, parse_evaluator_json

TEMPLATE_KIND = "accepted_monthly_evaluator_inputs_v1"
TRANSPORT_KIND = "saved_monthly_evaluator_inputs_v1"
MAX_TEMPLATE_BYTES = 128 * 1024
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_REPORT_BYTES = 16 * 1024 * 1024
MODEL_ROLES = ("primary", "logistic", "no_odds")


class EvaluatorInputRefusal(ValueError):
    """Missing or contradictory independently selected evaluator inputs."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise EvaluatorInputRefusal(message)


def safe_path(value: Any) -> Path:
    require(isinstance(value, (str, Path)) and bool(str(value)), "explicit evaluator path required")
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, "evaluator path must be absolute without traversal")
    for part in (path, *path.parents):
        require(not part.is_symlink() and not (hasattr(part, "is_junction") and part.is_junction()),
                "evaluator links and junctions are unsupported")
        if part.exists():
            require(not (getattr(part.lstat(), "st_file_attributes", 0) & 0x400),
                    "evaluator reparse points are unsupported")
    return path.resolve(strict=False)


def overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _hash(value: Any) -> None:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "independent lowercase evaluator SHA-256 required")


def _original_path(value: Any) -> Path:
    # Original paths are identities during reload, not current filesystem lookup
    # locations. Only a capture/read may inspect their present ancestry.
    require(isinstance(value, str) and bool(value), "explicit original evaluator path required")
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, "original path must be absolute without traversal")
    return path


def validate_pin(value: Any, *, limit: int = MAX_FILE_BYTES) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"},
            "exact evaluator pin fields required")
    _original_path(value["path"])
    require(type(value["bytes"]) is int and 0 < value["bytes"] <= limit,
            "evaluator original byte count outside bound")
    _hash(value["sha256"])
    return deepcopy(value)


def read_pin(pin: Mapping[str, Any], *, limit: int = MAX_FILE_BYTES) -> bytes:
    selected = validate_pin(dict(pin), limit=limit)
    path = safe_path(selected["path"])
    require(path.is_file() and path.stat().st_size == selected["bytes"], "evaluator original size changed or missing")
    with path.open("rb") as stream:
        body = stream.read(selected["bytes"] + 1)
    require(len(body) == selected["bytes"] and hashlib.sha256(body).hexdigest() == selected["sha256"],
            "evaluator original pin mismatch")
    safe_path(path)
    return body


def _pin_body(path: Path, body: bytes) -> dict[str, Any]:
    return {"path": str(path), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def _exclusive(path: Path, body: bytes) -> None:
    safe_path(path)
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def _small_saved_read(path: Path, limit: int = MAX_TEMPLATE_BYTES) -> bytes:
    selected = safe_path(path)
    require(selected.is_file() and 0 < selected.stat().st_size <= limit,
            "saved evaluator inventory missing or oversized")
    with selected.open("rb") as stream:
        body = stream.read(limit + 1)
    require(0 < len(body) <= limit, "oversized saved evaluator inventory")
    return body


def _identity(value: Any) -> None:
    require(isinstance(value, dict) and set(value) == {"name", "sha256"}
            and value["name"] == "full_live_contract_v6_durability_2022", "unsupported evaluator recipe")
    _hash(value["sha256"])


def _external_pins(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pins = {}
    for role, bundle in document["bundles"].items():
        require(isinstance(bundle, dict) and set(bundle) == {"models", "spec", "training_rows", "training_evidence"},
                "evaluator bundle fields missing or unknown")
        require(isinstance(bundle["models"], dict) and set(bundle["models"]) == set(MODEL_ROLES),
                "evaluator model roles missing or unknown")
        for model, pin in bundle["models"].items():
            pins[f"bundles.{role}.models.{model}"] = pin
        for field in ("spec", "training_rows", "training_evidence"):
            pins[f"bundles.{role}.{field}"] = bundle[field]
    for role, rows in document["row_sets"].items():
        require(isinstance(rows, dict) and set(rows) == {"table", "rows"}, "evaluator row-set fields conflict")
        for field, pin in rows.items():
            pins[f"row_sets.{role}.{field}"] = pin
    for field, pin in document["parity_inputs"].items():
        pins[f"parity_inputs.{field}"] = pin
    require(len(pins) <= 32, "too many evaluator originals")
    return pins


def validate_template(document: Any) -> dict[str, Any]:
    require(isinstance(document, dict) and set(document) == {
        "schema", "kind", "template_id", "selection", "windows", "bundles", "row_sets", "parity_inputs", "bounds"
    }, "evaluator template fields missing or unknown")
    require(type(document["schema"]) is int and document["schema"] == 1 and document["kind"] == TEMPLATE_KIND,
            "unsupported evaluator input template")
    require(isinstance(document["template_id"], str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", document["template_id"]),
            "invalid evaluator template identity")
    selection = document["selection"]
    require(isinstance(selection, dict) and set(selection) == {
        "monthly_input_sha256", "policy_sha256", "training_spec"}, "missing original evaluator selection")
    _hash(selection["monthly_input_sha256"])
    _hash(selection["policy_sha256"])
    _identity(selection["training_spec"])
    # Windows are protocol facts; exact original policy equality is checked when
    # the known saved monthly input is connected, before any writer/model work.
    from src.model.monthly_evidence import _instant
    fixed = {"timezone": "UTC", "train_start_inclusive": "2022-01-01", "train_end_exclusive": "2025-01-01",
             "validation_start_inclusive": "2025-01-01", "validation_end_exclusive": "2026-01-01",
             "validation_report_only": True, "evaluation_start_inclusive": "2026-01-01",
             "evaluation_end_inclusive": True}
    windows = document["windows"]
    require(isinstance(windows, dict) and set(windows) == set(fixed) | {"common_supported_end_utc"}
            and all(windows[k] == v and type(windows[k]) is type(v) for k, v in fixed.items()),
            "unsupported fixed evaluator windows")
    require(_instant(windows["common_supported_end_utc"]).year == 2026, "unsupported common evaluation end")
    require(isinstance(document["bundles"], dict) and set(document["bundles"]) == {"historical", "reference"},
            "explicit historical and reference roles required")
    require(isinstance(document["row_sets"], dict) and set(document["row_sets"]) == {
        "validation", "evaluation", "comparison", "parity"}, "fixed evaluator row sets required")
    require(isinstance(document["parity_inputs"], dict) and set(document["parity_inputs"]) == {
        "history", "exact", "prefight", "evidence"}, "complete original parity inputs required")
    limits = {"max_rows": 100000, "max_file_bytes": MAX_FILE_BYTES,
              "max_total_bytes": MAX_TOTAL_BYTES, "max_report_bytes": MAX_REPORT_BYTES}
    bounds = document["bounds"]
    require(isinstance(bounds, dict) and set(bounds) == set(limits)
            and all(type(bounds[k]) is int and 0 < bounds[k] <= v for k, v in limits.items()),
            "explicit finite evaluator bounds required")
    pins = _external_pins(document)
    for pin in pins.values():
        validate_pin(pin, limit=bounds["max_file_bytes"])
    require(sum(pin["bytes"] for pin in pins.values()) <= bounds["max_total_bytes"], "evaluator input total too large")
    # Shared immutable tables are legitimate. A role cannot change the identity
    # of the same original path, and estimator role swapping is never implicit.
    by_path = {}
    for pin in pins.values():
        key = os.path.normcase(str(_original_path(pin["path"])))
        require(key not in by_path or by_path[key] == pin, "conflicting pins for one evaluator original")
        by_path[key] = pin
    model_pins = [document["bundles"][r]["models"][m] for r in ("historical", "reference") for m in MODEL_ROLES]
    require(len({pin["sha256"] for pin in model_pins[:3]}) == 3
            and len({pin["sha256"] for pin in model_pins[3:]}) == 3, "duplicate estimator roles")
    canonical_bytes(document)
    return deepcopy(document)


def load_evaluator_template(pin: Mapping[str, Any], *, body: bytes | None = None) -> dict[str, Any]:
    selected = validate_pin(dict(pin), limit=MAX_TEMPLATE_BYTES)
    original = read_pin(selected, limit=MAX_TEMPLATE_BYTES) if body is None else body
    require(isinstance(original, bytes) and len(original) == selected["bytes"]
            and hashlib.sha256(original).hexdigest() == selected["sha256"], "original evaluator template changed")
    return validate_template(parse_evaluator_json(original))


def validate_selection(document: Mapping[str, Any], monthly_input: Mapping[str, Any]) -> None:
    policy = monthly_input.get("policy_binding")
    require(isinstance(policy, dict), "producer requires accepted successor monthly policy")
    require(document["selection"] == {"monthly_input_sha256": evidence_sha256(monthly_input),
            "policy_sha256": policy["policy"]["sha256"], "training_spec": policy["training_spec"]}
            and document["windows"] == policy["evaluation_windows"], "evaluator selection conflicts with saved A/policy")


def validate_transport(transport: Any) -> dict[str, Any]:
    require(isinstance(transport, dict) and set(transport) == {
        "schema", "kind", "root", "template_pin", "inventory_sha256"}, "invalid saved evaluator input transport")
    require(type(transport["schema"]) is int and transport["schema"] == 1 and transport["kind"] == TRANSPORT_KIND,
            "unsupported saved evaluator inputs")
    safe_path(transport["root"])
    validate_pin(transport["template_pin"], limit=MAX_TEMPLATE_BYTES)
    _hash(transport["inventory_sha256"])
    return deepcopy(transport)


def save_evaluator_inputs(*, template_pin: Mapping[str, Any], output_root: str | Path,
                          monthly_input: Mapping[str, Any], protected_paths) -> dict[str, Any]:
    """Save the selected original template and every directly needed external byte.

    A complete root reloads its original selection without reopening former paths.
    A partial root is retained and refuses reuse, preserving the consumed attempt.
    The caller supplies an already verified monthly transport and all directly
    protected original/reference/raw paths from that verification. This function
    does not replace the monthly publication/acceptance validator.
    """
    root = safe_path(output_root)
    if root.exists():
        complete_path = root / "complete.json"
        require(complete_path.is_file(), "partial evaluator input root is consumed; retain it and use a fresh root")
        # Caller should normally retain transport. This same-root continuation
        # reads only saved originals; changed current settings cannot select B.
        body = _small_saved_read(complete_path)
        complete = parse_evaluator_json(body)
        require(complete.get("template_pin") == dict(template_pin),
                "saved evaluator root belongs to a different independently accepted template")
        transport = {"schema": 1, "kind": TRANSPORT_KIND, "root": str(root),
                     "template_pin": complete.get("template_pin"), "inventory_sha256": hashlib.sha256(body).hexdigest()}
        loaded = load_evaluator_inputs(transport)
        validate_selection(loaded["template"], monthly_input)
        return transport
    body = read_pin(template_pin, limit=MAX_TEMPLATE_BYTES)
    document = load_evaluator_template(template_pin, body=body)
    validate_selection(document, monthly_input)
    pins = _external_pins(document)
    protected = [safe_path(p) for p in protected_paths]
    protected += [safe_path(monthly_input["root"]), safe_path(template_pin["path"]).parent]
    protected += [safe_path(pin["path"]).parent for pin in pins.values()]
    protected += [Path(__file__).resolve().parents[2]]
    require(all(not overlaps(root, p) for p in protected),
            "evaluator saved output overlaps source or accepted evidence")
    # Verify every original before creating any claimed save. Retained intent is
    # written before individual copies; failed writes never replace a prior root.
    originals = {label: read_pin(pin, limit=document["bounds"]["max_file_bytes"]) for label, pin in pins.items()}
    root.mkdir(parents=True, exist_ok=False)
    intent = {"schema": 1, "kind": "monthly_evaluator_input_intent_v1", "template_pin": dict(template_pin),
              "selection": document["selection"], "original_pins": pins}
    _exclusive(root / "intent.json", canonical_bytes(intent))
    _exclusive(root / "template.json", body)
    inventory = {}
    for relative in ("intent.json", "template.json"):
        content = (root / relative).read_bytes()
        inventory[relative] = {k: v for k, v in _pin_body(root / relative, content).items() if k != "path"}
    members = {}
    (root / "originals").mkdir()
    for index, (label, content) in enumerate(originals.items()):
        relative = f"originals/{index:02d}.bin"
        _exclusive(root / relative, content)
        inventory[relative] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        members[label] = relative
    complete = {"schema": 1, "kind": "complete_monthly_evaluator_inputs_v1", "template_pin": dict(template_pin),
                "inventory": inventory, "members": members}
    complete_body = canonical_bytes(complete)
    _exclusive(root / "complete.json", complete_body)
    transport = {"schema": 1, "kind": TRANSPORT_KIND, "root": str(root), "template_pin": dict(template_pin),
                 "inventory_sha256": hashlib.sha256(complete_body).hexdigest()}
    load_evaluator_inputs(transport)
    return transport


def load_evaluator_inputs(transport: Mapping[str, Any], *, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
    selected = validate_transport(dict(transport))
    root = safe_path(selected["root"])
    complete_path = root / "complete.json"
    body = _small_saved_read(complete_path)
    require(hashlib.sha256(body).hexdigest() == selected["inventory_sha256"], "saved evaluator inventory changed")
    complete = parse_evaluator_json(body)
    require(set(complete) == {"schema", "kind", "template_pin", "inventory", "members"}
            and type(complete["schema"]) is int and complete["schema"] == 1
            and complete["kind"] == "complete_monthly_evaluator_inputs_v1"
            and complete["template_pin"] == selected["template_pin"], "saved evaluator inventory conflict")
    inventory, members = complete["inventory"], complete["members"]
    require(isinstance(inventory, dict) and isinstance(members, dict) and len(members) <= 32,
            "invalid bounded evaluator inventory")
    require(set(inventory) == {"intent.json", "template.json"} | set(members.values())
            and len(set(members.values())) == len(members), "saved evaluator inventory incomplete")
    for relative, pin in inventory.items():
        require(isinstance(relative, str) and (relative in {"intent.json", "template.json"}
                or re.fullmatch(r"originals/[0-9]{2}\.bin", relative)), "unsafe saved evaluator member")
        require(isinstance(pin, dict) and set(pin) == {"bytes", "sha256"}, "invalid saved member pin")
        read_pin({"path": str(root / relative), **pin})
    actual = set()
    for count, path in enumerate(root.rglob("*"), 1):
        require(count <= 40, "saved evaluator root exceeds finite entry bound")
        safe_path(path)
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        else:
            require(path.is_dir() and path.relative_to(root).as_posix() == "originals",
                    "unexpected saved evaluator directory or special file")
    require(actual == set(inventory) | {"complete.json"}, "unexpected or missing saved evaluator originals")
    template = load_evaluator_template(selected["template_pin"], body=(root / "template.json").read_bytes())
    pins = _external_pins(template)
    require(set(members) == set(pins), "saved evaluator direct references differ")
    intent = parse_evaluator_json((root / "intent.json").read_bytes())
    require(intent == {"schema": 1, "kind": "monthly_evaluator_input_intent_v1", "template_pin": selected["template_pin"],
                       "selection": template["selection"], "original_pins": pins}, "original evaluator intent changed")
    normalized = deepcopy(template)
    paths = {}
    for label, original in pins.items():
        relative = members[label]
        require(inventory[relative] == {"bytes": original["bytes"], "sha256": original["sha256"]},
                "saved evaluator original binding conflict")
        cursor = normalized
        parts = label.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = {**original, "path": str(root / relative)}
        paths[label] = str(root / relative)
    plan = {"schema": 1, "kind": "realized_monthly_evaluator_plan_v1", "request_sha256": None,
            "template_pin": selected["template_pin"], "template": template, "saved_root": str(root),
            "paths": paths, "original_pins": pins, "bundles": normalized["bundles"],
            "row_sets": normalized["row_sets"], "parity_inputs": normalized["parity_inputs"],
            "bounds": template["bounds"], "windows": template["windows"]}
    if request is not None:
        from src.model.monthly_evidence import validate_evaluator_request
        checked = validate_evaluator_request(dict(request))
        require(checked.get("producer_inputs") == selected, "request differs from original evaluator input transport")
        validate_selection(template, checked["input_receipt"]["monthly_input"])
        plan["request_sha256"] = evidence_sha256(checked)
        plan["bundles"]["candidate"] = candidate_bundle(checked)
    return plan


def candidate_bundle(request: Mapping[str, Any]) -> dict[str, Any]:
    """Realize candidate locations only from the exact original model request."""
    root = safe_path(request["candidate_dir"])
    spec_name = request["contract"]["name"]
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", spec_name), "unsafe candidate spec name")
    paths = {"primary": root / "models/xgboost_model.pkl", "logistic": root / "models/logistic_model.pkl",
             "no_odds": root / "models/xgboost_no_odds_model.pkl", "spec": root / "models" / f"{spec_name}_spec.json"}
    pins = {}
    for role, path in paths.items():
        safe_path(path)
        require(path.is_file(), "candidate artifact absent")
        pin = {"path": str(path), "bytes": path.stat().st_size, "sha256": request["model_hashes"][role]}
        read_pin(pin)
        pins[role] = pin
    rows = root / "processed/modeling_dataset.csv"
    row_pin = {"path": str(rows), "bytes": request["input_receipt"]["derived_size_bytes"],
               "sha256": request["input_receipt"]["derived_sha256"]}
    read_pin(row_pin)
    return {"models": {role: pins[role] for role in MODEL_ROLES}, "spec": pins["spec"],
            "training_rows": row_pin, "training_evidence": {"kind": "monthly_candidate_request_training_v1",
                "request_sha256": evidence_sha256(request), "input_receipt": deepcopy(request["input_receipt"])}}
