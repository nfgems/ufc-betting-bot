"""Offline trusted adoption of completed bridge publications and original recurrence.

The externally pinned configuration fixes the initial accepted authority and all
discovery namespaces. Root-owned adoption receipts join independently recomputed
original source cycles to their published seeds. They never reset accounting,
extend a consumed plan, enable scheduling, or provision a hosted successor.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from src.data.roster_hosted_store import _require, _safe, _time, file_pin, verify_pin


def _load(pin):
    return json.loads(Path(verify_pin(pin)["path"]).read_bytes())


def provision_adoption_configuration(store, destination, *, accepted_pointer_pin,
        expected_pointer_sha256, expected_release_sha256, checkpoint_pin,
        expected_checkpoint_sha256, controller_history_root, integration_root,
        accepted_raw_root, accepted_manifest_path, now):
    """Seal externally supplied genesis anchors in a fresh native root directory."""
    from src.data.roster_hosted_custody import _root, _trusted
    from src.data.roster_hosted_inputs import _accepted, _checkpoint
    from src.data import roster_refresh_integration as bridge

    _root()
    root = _safe(destination)
    _require(not root.exists() and root.parent.is_dir(), "Fresh adoption configuration directory required")
    state = store.snapshot()
    pointer, _ = _accepted(store, accepted_pointer_pin, expected_pointer_sha256, expected_release_sha256)
    _require(pointer["store_manifest_sha256"] == state["head"]["manifest_sha256"],
             "Adoption genesis belongs to another accounting store")
    _checkpoint(store, state, accepted_pointer_pin, checkpoint_pin, expected_checkpoint_sha256, _time(now))
    root.mkdir(mode=0o700)
    _trusted(root, directory=True)
    value = {"schema": 1, "kind": "offline_roster_adoption_configuration_v1", "root": root.as_posix(),
        "store_root": store.root.as_posix(), "accepted_pointer": accepted_pointer_pin,
        "expected_pointer_sha256": expected_pointer_sha256,
        "expected_release_sha256": expected_release_sha256, "checkpoint": checkpoint_pin,
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "controller_history_root": _safe(controller_history_root).as_posix(),
        "integration_root": _safe(integration_root).as_posix(),
        "accepted_raw_root": _safe(accepted_raw_root).as_posix(),
        "accepted_manifest_path": _safe(accepted_manifest_path).as_posix(),
        "hosted_activation_authorized": False}
    (root / "adoptions").mkdir(mode=0o700)
    return bridge._save(root / "configuration.json", value)


def _configuration(store, pin, expected_sha256):
    from src.data.roster_hosted_custody import _trusted

    _require(pin["sha256"] == expected_sha256, "External adoption configuration hash mismatch")
    _trusted(pin["path"], directory=False)
    config = _load(pin)
    _require(config.get("kind") == "offline_roster_adoption_configuration_v1"
             and config.get("schema") == 1 and config["store_root"] == store.root.as_posix()
             and Path(pin["path"]) == Path(config["root"]) / "configuration.json",
             "Exact externally anchored adoption configuration required")
    _trusted(config["root"], directory=True)
    _trusted(Path(config["root"]) / "adoptions", directory=True)
    return config


def _completion(config, binding_pin):
    from src.data import roster_refresh_integration as bridge

    binding = _load(binding_pin)
    _require(Path(binding["output_root"]).parent == Path(config["controller_history_root"])
             and Path(binding_pin["path"]) == Path(binding["output_root"]) / "binding.json",
             "Adoption must bind an original sibling controller")
    cycle_id = bridge.cycle_identity(binding_pin)
    root = Path(config["integration_root"]) / "cycles" / cycle_id
    completion_pin = file_pin(root / bridge.COMPLETION_FILE)
    completion = _load(completion_pin)
    _require(completion.get("kind") == bridge.COMPLETION_KIND and completion.get("cycle_id") == cycle_id
             and completion.get("outcome") in bridge.OUTCOMES,
             "Original durable bridge completion required before later recurrence")
    return completion_pin, completion


def _validate_transition(store, config, record, prior_pointer, *, now):
    """Recompute the source, publication, seed and store edge; receipt flags are insufficient."""
    from src.data import roster_refresh_integration as bridge
    from src.data.roster_hosted_candidate import tree_inventory
    from src.data.roster_hosted_inputs import _accepted, _checkpoint
    from src.data.roster_hosted_recurrence import validated_cycle_status
    from src.data.roster_hosted_needs import derive_need_evidence

    binding = _load(record["binding"])
    completion_pin, completion = _completion(config, record["binding"])
    _require(completion_pin == record["completion"] and completion["outcome"] == bridge.OUTCOME_PUBLISHED,
             "Only the exact original published completion can advance accepted authority")
    _require(binding["accepted_pointer"] == prior_pointer,
             "Adoption predecessor differs from original controller authority")
    original = validated_cycle_status(store, record["binding"], now=now)
    _require(original["status"] == "succeeded", "Failed original collection cannot advance accepted authority")
    stages = completion["stages"]
    cycle_root = Path(config["integration_root"]) / "cycles" / completion["cycle_id"]
    for key, filename in (("retained_inputs", bridge.RETAINED_INPUTS_FILE), ("publication", bridge.PUBLICATION_FILE),
                          ("accepted_baseline", bridge.BASELINE_FILE), ("processing", bridge.PROCESSING_FILE),
                          ("quality", bridge.QUALITY_FILE)):
        _require(Path(stages[key]["path"]) == cycle_root / filename,
                 "Accepted adoption requires the original cycle stage files")
    retained = _load(stages["retained_inputs"])
    publication = _load(stages["publication"])
    baseline = _load(stages["accepted_baseline"])
    processing = _load(stages["processing"])
    quality = _load(stages["quality"])
    cycle_id = completion["cycle_id"]
    _require(retained["binding"] == record["binding"] and retained["input_pins"] == original["input_pins"]
             and retained["accepted_pointer"] == prior_pointer
             and all(value["cycle_id"] == cycle_id for value in (retained, publication, baseline, processing, quality)),
             "Adoption artifacts must bind this exact original source cycle")
    _require(quality["passed"] is True and quality["processing"] == stages["processing"]
             and processing["manifest"] == stages["retained_inputs"]
             and publication["quality"] == stages["quality"]
             and publication["processing"] == stages["processing"]
             and baseline["publication"] == stages["publication"],
             "Published processing and quality linkage changed")
    _require(publication["publication_performed"] is True and publication["outcome"] == bridge.OUTCOME_PUBLISHED
             and completion["accepted_generation_id"] == publication["accepted_generation_id"]
             == baseline["accepted_generation_id"]
             and completion["accepted_at_utc"] == publication["accepted_at_utc"]
             and _time(publication["accepted_at_utc"]) <= _time(completion["completed_at_utc"]) <= _time(now),
             "Published generation or original publication time differs")
    snapshot = _load(record["publication_manifest"])
    _require(record["publication_manifest"]["sha256"] == publication["accepted_manifest"]["sha256"]
             and record["publication_manifest"]["bytes"] == publication["accepted_manifest"]["bytes"]
             and Path(publication["accepted_manifest"]["path"]) == Path(config["accepted_manifest_path"]),
             "Trusted manifest snapshot differs from the original published pointer")
    accepted = bridge._accepted_publication(snapshot)
    _require(accepted.get("published_by") == bridge.PUBLISHED_BY and accepted.get("bridge_cycle_id") == cycle_id
             and accepted["accepted_generation_id"] == publication["accepted_generation_id"]
             and accepted["accepted_at_utc"] == publication["accepted_at_utc"]
             and accepted["processed_dir"] == str(Path(processing["staged_processed_root"]).resolve()),
             "Manifest does not commit this original cycle")
    identity = bridge._scheduled()._accepted_data_generation_identity(Path(processing["staged_processed_root"]))
    _require(identity["accepted_generation_id"] == accepted["accepted_generation_id"],
             "Retained processed generation no longer proves the publication")
    expected_root = Path(config["integration_root"]) / "accepted" / f"{baseline['accepted_generation_id']}-{cycle_id}"
    _require(Path(baseline["root"]) == expected_root and Path(baseline["seed_root"]) == expected_root / "seed",
             "Accepted baseline escaped its configured cycle namespace")
    seed = Path(baseline["seed_root"])
    _require(tree_inventory(seed) == _load(baseline["seed_inventory"]), "Adopted seed inventory changed")
    bridge._seed_matches_accepted_raw(seed, Path(processing["staging"]["candidate_raw_root"]))
    pointer, inputs = _accepted(store, baseline["accepted_pointer"], baseline["expected_pointer_sha256"],
                               config["expected_release_sha256"])
    _require(pointer["input_manifest"] == baseline["accepted_input_manifest"] and inputs == baseline["input_pins"]
             and inputs["membership"] == original["input_pins"]["membership"]
             and inputs["release"] == original["input_pins"]["release"]
             and _load(inputs["owners"]) == {"schema": 1, "seed_root": str(seed), "seed_inventory": baseline["seed_inventory"]},
             "Accepted BASE_ROLES do not follow the retained source and published seed")
    catalog = Path(processing["staging"]["candidate_raw_root"]) / bridge.ROSTERWATCH_SUBDIR / bridge.IDENTITY_CAPTURES_FILE
    expected_catalog = json.loads(catalog.read_bytes()) if catalog.is_file() else _load(original["input_pins"]["catalog"])
    _require(_load(inputs["catalog"]) == expected_catalog, "Accepted catalog diverges from published retained captures")
    derive_need_evidence(inputs, now=_time(publication["accepted_at_utc"]))
    state = store.snapshot()
    checkpoint = _checkpoint(store, state, baseline["accepted_pointer"], baseline["checkpoint"],
                             baseline["expected_checkpoint_sha256"], _time(now))
    _require(checkpoint["store_head"] == baseline["store_head"] == retained["store_head"]
             and pointer["store_manifest_sha256"] == state["head"]["manifest_sha256"],
             "Adoption must retain the original publication accounting head")
    return {"slot_id": original["slot_id"], "final_inputs": original["input_pins"], "accepted_inputs": inputs,
            "accepted_pointer": baseline["accepted_pointer"], "checkpoint": baseline["checkpoint"],
            "completion": completion_pin, "binding": record["binding"]}


def resolve_adoption_context(store, configuration_pin, *, expected_configuration_sha256, now):
    """Read the protected adoption chain, anchored outside bridge-produced artifacts."""
    from src.data.roster_hosted_custody import _trusted
    from src.data.roster_hosted_inputs import _accepted, _checkpoint

    config = _configuration(store, configuration_pin, expected_configuration_sha256)
    state = store.snapshot()
    pointer, genesis = _accepted(store, config["accepted_pointer"], config["expected_pointer_sha256"],
                                config["expected_release_sha256"])
    _require(pointer["store_manifest_sha256"] == state["head"]["manifest_sha256"], "Adoption accounting manifest changed")
    _checkpoint(store, state, config["accepted_pointer"], config["checkpoint"],
                config["expected_checkpoint_sha256"], _time(now))
    current_pointer, checkpoint, previous = config["accepted_pointer"], config["checkpoint"], None
    transitions, receipts = {}, []
    files = sorted((Path(config["root"]) / "adoptions").glob("*.json"))
    _require(len(files) <= 10000, "Adoption history ceiling")
    for path in files:
        _trusted(path, directory=False)
        pin = file_pin(path)
        record = _load(pin)
        _require(record.get("kind") == "offline_roster_baseline_adoption_v1"
                 and record["configuration"] == configuration_pin and record["previous_adoption"] == previous,
                 "Adoption receipt chain is missing, reordered, or belongs to another configuration")
        edge = _validate_transition(store, config, record, current_pointer, now=now)
        _require(edge["slot_id"] not in transitions and path.name == edge["slot_id"] + ".json",
                 "Duplicate or renamed accepted transition")
        transitions[edge["slot_id"]] = edge
        current_pointer, checkpoint, previous = edge["accepted_pointer"], edge["checkpoint"], pin
        receipts.append(pin)
    _require(store.snapshot() == state, "Accounting changed while validating adoption history")
    return {"configuration": config, "genesis_inputs": genesis, "transitions": transitions,
            "accepted_pointer": current_pointer, "checkpoint": checkpoint, "receipts": receipts,
            "expected_release_sha256": config["expected_release_sha256"]}


def adopt_completed_publication(store, binding_pin, *, configuration_pin, expected_configuration_sha256, now):
    """Independently validate and adopt one published completion; exact retries are idempotent."""
    from src.data.roster_hosted_custody import _root
    from src.data.roster_hosted_controller import _discover_roots
    from src.data.roster_hosted_inputs import _recover_cycles
    from src.data import roster_refresh_integration as bridge

    _root()
    config = _configuration(store, configuration_pin, expected_configuration_sha256)
    with bridge.cycle_lock(Path(config["root"]) / "adoption-lock"):
        context = resolve_adoption_context(store, configuration_pin,
            expected_configuration_sha256=expected_configuration_sha256, now=now)
        completion_pin, completion = _completion(config, binding_pin)
        _require(completion["outcome"] == bridge.OUTCOME_PUBLISHED, "Only changed publication has an accepted-baseline transition")
        binding = _load(binding_pin)
        slot = _load(binding["root_plan"])["slot_id"]
        existing = context["transitions"].get(slot)
        if existing:
            _require(existing["binding"] == binding_pin and existing["completion"] == completion_pin,
                     "Existing accepted transition belongs to different evidence")
            return {"existing": True, "adoption": next(p for p in context["receipts"] if Path(p["path"]).stem == slot)}
        state = store.snapshot()
        _, lineage, excluded, _ = _recover_cycles(store, state, context["genesis_inputs"], _time(now),
            _discover_roots(store, Path(config["controller_history_root"])), adoption_context=context)
        _require(lineage and not excluded and lineage[-1]["root_plan"] == binding["root_plan"],
                 "Only the latest complete original lineage can advance accepted authority")
        publication = _load(completion["stages"]["publication"])
        raw_pointer = Path(config["accepted_manifest_path"])
        _require(file_pin(raw_pointer) == publication["accepted_manifest"],
                 "Current accepted manifest differs from the completed publication")
        baseline = _load(completion["stages"]["accepted_baseline"])
        bridge._seed_matches_accepted_raw(Path(baseline["seed_root"]), Path(config["accepted_raw_root"]))
        snapshot_path = Path(config["root"]) / (slot + "-publication-manifest.json")
        raw = raw_pointer.read_bytes()
        if snapshot_path.exists():
            _require(snapshot_path.read_bytes() == raw, "Interrupted adoption manifest snapshot changed")
            snapshot_pin = file_pin(snapshot_path)
        else:
            snapshot_pin = bridge._write_new(snapshot_path, raw)
        record = {"schema": 1, "kind": "offline_roster_baseline_adoption_v1", "configuration": configuration_pin,
            "previous_adoption": context["receipts"][-1] if context["receipts"] else None,
            "binding": binding_pin, "completion": completion_pin, "publication_manifest": snapshot_pin}
        _validate_transition(store, config, record, context["accepted_pointer"], now=now)
        _require(store.snapshot() == state and file_pin(raw_pointer) == publication["accepted_manifest"],
                 "Accepted state or accounting changed during adoption")
        result = bridge._save(Path(config["root"]) / "adoptions" / (slot + ".json"), record)
        return {"existing": False, "adoption": result}


def prepare_adopted_successor(store, previous_binding_pin, output_directory, *, configuration_pin,
        expected_configuration_sha256, policy_pin, expected_policy_sha256,
        synthetic_responses_pin=None, persist_checkpoint=None, clock=lambda: datetime.now(timezone.utc)):
    """Select a later original calendar slot through trusted adoption and the real controller."""
    from src.data.roster_hosted_controller import prepare_controller
    from src.data.roster_hosted_recurrence import validate_successor_cycle
    from src.data.roster_hosted_plan import _policy
    from src.data.roster_hosted_custody import _root
    from src.data import roster_refresh_integration as bridge

    _root()
    _policy(policy_pin, expected_policy_sha256)
    now = _time(clock())
    context = resolve_adoption_context(store, configuration_pin,
        expected_configuration_sha256=expected_configuration_sha256, now=now)
    config = context["configuration"]
    _require(_safe(output_directory).parent == Path(config["controller_history_root"]),
             "Successor must remain in the configured original controller namespace")
    _, completion = _completion(config, previous_binding_pin)
    if completion["outcome"] == bridge.OUTCOME_PUBLISHED:
        slot = _load(_load(previous_binding_pin)["root_plan"])["slot_id"]
        _require(slot in context["transitions"], "Published predecessor requires trusted adoption before recurrence")
    decision = validate_successor_cycle(store, previous_binding_pin, policy_pin, now=now,
        adoption_configuration_pin=configuration_pin,
        expected_adoption_configuration_sha256=expected_configuration_sha256)
    recovery_pin = None
    if decision["recovery_evidence"] is not None:
        path = Path(config["root"]) / (decision["slot_id"] + "-recovery-evidence.json")
        if path.exists():
            _require(_load(file_pin(path)) == decision["recovery_evidence"], "Original recovery evidence changed")
            recovery_pin = file_pin(path)
        else:
            recovery_pin = bridge._save(path, decision["recovery_evidence"])
    pointer, checkpoint = context["accepted_pointer"], context["checkpoint"]
    return prepare_controller(store, pointer, output_directory,
        expected_pointer_sha256=pointer["sha256"], expected_release_sha256=config["expected_release_sha256"],
        checkpoint_pin=checkpoint, expected_checkpoint_sha256=checkpoint["sha256"],
        policy_pin=policy_pin, expected_policy_sha256=expected_policy_sha256,
        synthetic_responses_pin=synthetic_responses_pin, persist_checkpoint=persist_checkpoint,
        recovery_evidence_pin=recovery_pin, clock=clock,
        adoption_configuration_pin=configuration_pin,
        expected_adoption_configuration_sha256=expected_configuration_sha256)


def run_and_adopt_cycle(store, binding_pin, *, configuration_pin, expected_configuration_sha256,
                        monthly_input_request_provider=None, **kwargs):
    """Drive the isolated bridge and then advance trusted configuration on publication."""
    from src.data import roster_refresh_integration as bridge
    from src.data.roster_hosted_controller import _discover_roots
    from src.data.roster_hosted_inputs import _recover_cycles

    from src.data.roster_hosted_custody import _root

    _root()
    config = _configuration(store, configuration_pin, expected_configuration_sha256)
    for key in ("integration_root", "accepted_raw_root", "accepted_manifest_path"):
        _require(_safe(kwargs[key]) == Path(config[key]), "Bridge target differs from trusted adoption configuration")
    binding = _load(binding_pin)
    _require(binding.get("mode") == "disabled_controller_v1" and binding.get("store_root") == store.root.as_posix()
             and Path(binding["output_root"]).parent == Path(config["controller_history_root"])
             and Path(binding_pin["path"]) == Path(binding["output_root"]) / "binding.json",
             "Bridge binding differs from the trusted original controller namespace")
    now = _time(kwargs["clock"]()) if "clock" in kwargs else datetime.now(timezone.utc)
    context = resolve_adoption_context(store, configuration_pin,
        expected_configuration_sha256=expected_configuration_sha256, now=now)
    prior_adoption = next((edge for edge in context["transitions"].values() if edge["binding"] == binding_pin), None)
    if prior_adoption is None:
        _require(binding["accepted_pointer"] == context["accepted_pointer"],
                 "Bridge binding does not start from the current trusted accepted authority")
        state = store.snapshot()
        _, lineage, excluded, _ = _recover_cycles(store, state, context["genesis_inputs"], now,
            _discover_roots(store, Path(config["controller_history_root"])), adoption_context=context)
        _require(lineage and not excluded and lineage[-1]["root_plan"] == binding["root_plan"],
                 "Bridge must follow the latest complete original lineage")
        _require(store.snapshot() == state, "Accounting changed during bridge adoption preflight")
    # A historical retry is permitted only when the trusted chain has already
    # recomputed its exact original completion and adoption. run_cycle returns
    # that durable completion; it cannot republish it into the newer destination.
    if kwargs.get("schedule") is not None and monthly_input_request_provider is None:
        from src.model.monthly_provisioning import configure_monthly_calendar_from_env
        monthly_input_request_provider = configure_monthly_calendar_from_env(kwargs.get("environ"))["provider"]
    result = bridge.run_cycle(store, binding_pin,
                             monthly_input_request_provider=monthly_input_request_provider, **kwargs)
    if result["outcome"] == bridge.OUTCOME_PUBLISHED:
        now = _time(kwargs["clock"]()) if "clock" in kwargs else datetime.now(timezone.utc)
        result["trusted_adoption"] = adopt_completed_publication(store, binding_pin,
            configuration_pin=configuration_pin, expected_configuration_sha256=expected_configuration_sha256, now=now)
    return result
