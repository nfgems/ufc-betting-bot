from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import install_staged_production_bundle as installer


REAL_REQUIRE_CANDIDATE_PREDECESSOR = installer._require_candidate_predecessor


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _make_stage(parent: Path, bundle_id: str) -> tuple[Path, str]:
    root = parent / bundle_id
    root.mkdir(parents=True)
    files = {
        "models/xgboost_model.pkl": b"primary-" + bundle_id.encode(),
        "models/xgboost_no_odds_model.pkl": b"no-odds-" + bundle_id.encode(),
        "models/logistic_model.pkl": b"logistic-" + bundle_id.encode(),
        f"models/{bundle_id}_spec.json": b'{"name":"fixture"}\n',
        "processed/fights_cleaned.csv": b"event_date,target\n2026-08-01,1\n",
        "processed/features.csv": b"event_date,target,f\n2026-08-01,1,0.5\n",
        "processed/test_set.csv": b"event_date,target,f\n",
        "processed/test_set.csv.metadata.json": b'{"rows":0}\n',
        "provenance/pretraining_model_input_inventory.json": b'{"kind":"pre"}\n',
        "provenance/assembly_model_input_inventory.json": b'{"kind":"assembly"}\n',
        "provenance/independent_audit_snapshot/fights_cleaned.csv": b"audit fights\n",
        "provenance/independent_audit_snapshot/features.csv": b"audit features\n",
        "provenance/data/raw/historical_odds/revalidation.jsonl": b'{"accepted":true}\n',
        "rollback/previous_readyz.json": b'{"ready":true}\n',
        "evidence/selection.json": b'{"selected":true}\n',
    }
    for relative, content in files.items():
        _write(root / relative, content)

    payload = {
        "manifest_version": 3,
        "staging_schema_version": 1,
        "bundle_id": bundle_id,
        "model_spec_name": bundle_id,
        "no_odds_model_spec_name": f"{bundle_id}_no_odds",
        "model_path": str((root / "models/xgboost_model.pkl").resolve()),
        "no_odds_model_path": str((root / "models/xgboost_no_odds_model.pkl").resolve()),
        "logistic_model_path": str((root / "models/logistic_model.pkl").resolve()),
        "processed_dir": str((root / "processed").resolve()),
        "model_sha256": _sha256(root / "models/xgboost_model.pkl"),
        "no_odds_model_sha256": _sha256(root / "models/xgboost_no_odds_model.pkl"),
        "logistic_model_sha256": _sha256(root / "models/logistic_model.pkl"),
        "processed_fights_sha256": _sha256(root / "processed/fights_cleaned.csv"),
        "processed_features_sha256": _sha256(root / "processed/features.csv"),
        "processed_fights_bytes": (root / "processed/fights_cleaned.csv").stat().st_size,
        "processed_features_bytes": (root / "processed/features.csv").stat().st_size,
        "model_artifacts": {
            "primary": {"staged_path": "models/xgboost_model.pkl"},
            "no_odds": {"staged_path": "models/xgboost_no_odds_model.pkl"},
            "logistic": {"staged_path": "models/logistic_model.pkl"},
        },
        "saved_fullfit_spec": {
            "staged_path": f"models/{bundle_id}_spec.json",
        },
        "immutable_training_snapshot": {
            "immutable": True,
            "fights": {"staged_path": "processed/fights_cleaned.csv"},
            "features": {"staged_path": "processed/features.csv"},
            "test_set": {
                "staged_path": "processed/test_set.csv",
                "metadata_staged_path": "processed/test_set.csv.metadata.json",
            },
        },
        "source_identity": {
            "pretraining_inventory_artifact": {
                "staged_path": "provenance/pretraining_model_input_inventory.json",
            },
            "assembly_inventory_artifact": {
                "staged_path": "provenance/assembly_model_input_inventory.json",
            },
        },
        "raw_input_provenance": {
            "bfo_ledger": {
                "staged_path": "provenance/data/raw/historical_odds/revalidation.jsonl",
            },
        },
        "selection_evidence": {
            "files": [{"staged_path": "evidence/selection.json"}],
        },
        "training_invocation": {
            "independent_audit_snapshot": {
                "fights": {
                    "staged_path": "provenance/independent_audit_snapshot/fights_cleaned.csv",
                },
                "features": {
                    "staged_path": "provenance/independent_audit_snapshot/features.csv",
                },
            },
        },
        "previous_rollback_identity": {
            "readyz_evidence": {"staged_path": "rollback/previous_readyz.json"},
        },
    }
    manifest = root / "staging_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root, _sha256(manifest)


def _make_legacy_capture_stub(parent: Path) -> installer.ValidatedLegacyCapture:
    root = parent / "legacy-capture"
    model_spec_name = "legacy-spec"
    files = {
        "staging_manifest.json": b"{}\n",
        "runtime/manifest.json": b"{}\n",
        "models/xgboost_model.pkl": b"legacy-primary",
        "models/xgboost_no_odds_model.pkl": b"legacy-no-odds",
        "models/logistic_model.pkl": b"legacy-logistic",
        f"models/{model_spec_name}_spec.json": b"{}\n",
        "processed/fights_cleaned.csv": b"event_date\n2026-08-01\n",
        "processed/features.csv": b"event_date,f\n2026-08-01,1\n",
    }
    for relative, content in files.items():
        _write(root / relative, content)
    capture_payload = {
        "files": {
            "saved_spec": {
                "sha256": _sha256(root / f"models/{model_spec_name}_spec.json")
            },
            "runtime_manifest": {"sha256": _sha256(root / "runtime/manifest.json")},
        }
    }
    capture = root / "rollback_capture_manifest.json"
    capture.write_text(json.dumps(capture_payload), encoding="utf-8")
    payload = {
        "manifest_version": 2,
        "bundle_id": "legacy-live",
        "model_spec_name": model_spec_name,
        "no_odds_model_spec_name": f"{model_spec_name}_no_odds",
        "snapshot_max_event_date": "2026-08-01",
        "model_sha256": _sha256(root / "models/xgboost_model.pkl"),
        "no_odds_model_sha256": _sha256(root / "models/xgboost_no_odds_model.pkl"),
        "logistic_model_sha256": _sha256(root / "models/logistic_model.pkl"),
        "processed_fights_sha256": _sha256(root / "processed/fights_cleaned.csv"),
        "processed_features_sha256": _sha256(root / "processed/features.csv"),
        "processed_fights_bytes": (root / "processed/fights_cleaned.csv").stat().st_size,
        "processed_features_bytes": (root / "processed/features.csv").stat().st_size,
    }
    source_manifest = root / "staging_manifest.json"
    source_manifest.write_text(json.dumps(payload), encoding="utf-8")
    return installer.ValidatedLegacyCapture(
        source_manifest=source_manifest.resolve(),
        runtime_manifest=(root / "runtime/manifest.json").resolve(),
        capture_manifest=capture.resolve(),
        primary_model=(root / "models/xgboost_model.pkl").resolve(),
        no_odds_model=(root / "models/xgboost_no_odds_model.pkl").resolve(),
        logistic_model=(root / "models/logistic_model.pkl").resolve(),
        saved_spec=(root / f"models/{model_spec_name}_spec.json").resolve(),
        processed_fights=(root / "processed/fights_cleaned.csv").resolve(),
        processed_features=(root / "processed/features.csv").resolve(),
        payload=payload,
        source_manifest_sha256=_sha256(source_manifest),
    )


@pytest.fixture(autouse=True)
def _stub_expensive_semantic_validation(monkeypatch, tmp_path: Path):
    # The assembler/production validators have their own focused suites.  These
    # tests exercise copying, confinement, exact-tree receipts, and pointer
    # transactions with small fake artifacts.
    monkeypatch.setattr(installer, "_semantic_validate_manifest", lambda *_args, **_kwargs: None)

    def predecessor_stub(*_args, active_lookup_path: Path, **_kwargs):
        return installer._lookup_identity(active_lookup_path)

    monkeypatch.setattr(installer, "_require_candidate_predecessor", predecessor_stub)
    yield
    # Published releases/checkpoints are intentionally non-writable on POSIX.
    # Restore only test-owned temporary stores so pytest can remove them.
    for marker in tmp_path.rglob(installer.STORE_MARKER_NAME):
        installer._make_owned_tree_deletable(marker.parent)


def _initialize(tmp_path: Path, bundle_id: str = "bundle-a") -> tuple[Path, Path, dict]:
    source, manifest_sha = _make_stage(tmp_path / "sources", bundle_id)
    target = tmp_path / "runtime-store"
    result = installer.initialize_store(
        source_root=source,
        target_root=target,
        expected_bundle_id=bundle_id,
        expected_source_manifest_sha256=manifest_sha,
        allow_no_rollback_test_only=True,
    )
    return source, target, result


def _state(target: Path) -> dict:
    return json.loads((target / installer.STATE_NAME).read_text(encoding="utf-8"))


def _write_matching_runtime_manifest(target: Path, active_record: dict) -> Path:
    release_manifest = json.loads(
        (target / active_record["manifest_path"]).read_text(encoding="utf-8")
    )
    lookup = target / active_record["lookup_path"]
    identity = installer._lookup_identity(lookup)
    payload = {
        "bundle_id": release_manifest["bundle_id"],
        "model_sha256": release_manifest["model_sha256"],
        "no_odds_model_sha256": release_manifest["no_odds_model_sha256"],
        "logistic_model_sha256": release_manifest["logistic_model_sha256"],
        "processed_fights_sha256": identity["fights_sha256"],
        "processed_features_sha256": identity["features_sha256"],
        "processed_fights_bytes": identity["fights_bytes"],
        "processed_features_bytes": identity["features_bytes"],
    }
    path = lookup / installer._RUNTIME_MANIFEST_NAME
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _promote_fixture(target: Path, source: Path, manifest_sha: str) -> dict:
    active = _state(target)["active"]
    return installer.promote_bundle(
        source_root=source,
        target_root=target,
        expected_bundle_id="bundle-b",
        expected_source_manifest_sha256=manifest_sha,
        expected_active_bundle_id=active["bundle_id"],
        expected_active_release_id=active["release_id"],
        expected_active_manifest_sha256=active["manifest_sha256"],
    )


def test_initialize_copies_one_self_contained_immutable_release(tmp_path: Path):
    source, target, result = _initialize(tmp_path)

    assert result["active_bundle_id"] == "bundle-a"
    assert result["rollback_ready"] is False
    assert result["rollback_bundle_id"] is None
    state = json.loads((target / installer.STATE_NAME).read_text(encoding="utf-8"))
    release = target / state["active"]["release_path"]
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))

    assert Path(manifest["model_path"]) == release / "models/xgboost_model.pkl"
    assert Path(manifest["no_odds_model_path"]) == release / "models/xgboost_no_odds_model.pkl"
    assert Path(manifest["logistic_model_path"]) == release / "models/logistic_model.pkl"
    assert Path(manifest["processed_dir"]) == release / "processed"
    assert (release / "processed/test_set.csv.metadata.json").is_file()
    assert (release / "provenance/pretraining_model_input_inventory.json").is_file()
    assert (release / "rollback/previous_readyz.json").is_file()
    assert (release / "evidence/selection.json").is_file()
    assert _sha256(release / "provenance/source_staging_manifest.json") == _sha256(
        source / "staging_manifest.json"
    )
    receipt = json.loads((release / installer.RECEIPT_NAME).read_text(encoding="utf-8"))
    inventoried = {row["path"] for row in receipt["tree_inventory"]["files"]}
    assert "manifest.json" in inventoried
    assert "models/xgboost_model.pkl" in inventoried
    assert "processed/features.csv" in inventoried
    assert "provenance/source_staging_manifest.json" in inventoried
    assert installer.LOCK_NAME not in {path.name for path in target.iterdir()}


def test_promote_atomically_records_complete_predecessor_for_rollback(tmp_path: Path):
    _, target, initial = _initialize(tmp_path)
    source_b, sha_b = _make_stage(tmp_path / "sources", "bundle-b")

    promoted = _promote_fixture(target, source_b, sha_b)

    assert promoted["active_bundle_id"] == "bundle-b"
    assert promoted["rollback_bundle_id"] == "bundle-a"
    assert promoted["rollback_ready"] is True
    assert Path(initial["active_manifest_path"]).is_file()
    assert Path(promoted["active_manifest_path"]).is_file()
    assert len([path for path in (target / "releases").iterdir() if path.is_dir()]) == 2


def test_rollback_is_one_state_swap_and_supports_roll_forward(tmp_path: Path):
    _, target, _ = _initialize(tmp_path)
    source_b, sha_b = _make_stage(tmp_path / "sources", "bundle-b")
    _promote_fixture(target, source_b, sha_b)

    state = _state(target)
    rolled_back = installer.rollback_bundle(
        target_root=target,
        expected_active_bundle_id="bundle-b",
        expected_rollback_bundle_id="bundle-a",
        expected_active_release_id=state["active"]["release_id"],
        expected_rollback_release_id=state["rollback"]["release_id"],
        expected_active_manifest_sha256=state["active"]["manifest_sha256"],
        expected_rollback_manifest_sha256=state["rollback"]["manifest_sha256"],
    )
    assert rolled_back["active_bundle_id"] == "bundle-a"
    assert rolled_back["rollback_bundle_id"] == "bundle-b"

    state = _state(target)
    rolled_forward = installer.rollback_bundle(
        target_root=target,
        expected_active_bundle_id="bundle-a",
        expected_rollback_bundle_id="bundle-b",
        expected_active_release_id=state["active"]["release_id"],
        expected_rollback_release_id=state["rollback"]["release_id"],
        expected_active_manifest_sha256=state["active"]["manifest_sha256"],
        expected_rollback_manifest_sha256=state["rollback"]["manifest_sha256"],
    )
    assert rolled_forward["active_bundle_id"] == "bundle-b"
    assert rolled_forward["rollback_bundle_id"] == "bundle-a"


def test_source_failure_happens_before_target_creation(tmp_path: Path):
    source, manifest_sha = _make_stage(tmp_path / "sources", "bundle-a")
    target = tmp_path / "must-not-exist"

    with pytest.raises(installer.BundleInstallError, match="Source manifest hash mismatch"):
        installer.initialize_store(
            source_root=source,
            target_root=target,
            expected_bundle_id="bundle-a",
            expected_source_manifest_sha256="0" * 64,
            allow_no_rollback_test_only=True,
        )

    assert not target.exists()
    assert manifest_sha != "0" * 64


def test_unmarked_nonempty_target_is_never_adopted_or_modified(tmp_path: Path):
    source, manifest_sha = _make_stage(tmp_path / "sources", "bundle-a")
    target = tmp_path / "unrelated"
    target.mkdir()
    sentinel = target / "user-data.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    with pytest.raises(installer.BundleInstallError, match="unexpected top-level"):
        installer.initialize_store(
            source_root=source,
            target_root=target,
            expected_bundle_id="bundle-a",
            expected_source_manifest_sha256=manifest_sha,
            allow_no_rollback_test_only=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep me"
    assert set(target.iterdir()) == {sentinel}


def test_promotion_pointer_failure_leaves_old_active_unit(tmp_path: Path, monkeypatch):
    _, target, initial = _initialize(tmp_path)
    source_b, sha_b = _make_stage(tmp_path / "sources", "bundle-b")
    original_atomic_write = installer._atomic_write_json

    def fail_state(path: Path, payload):
        if path.name == installer.STATE_NAME:
            raise OSError("injected state replace failure")
        return original_atomic_write(path, payload)

    monkeypatch.setattr(installer, "_atomic_write_json", fail_state)
    active = _state(target)["active"]
    with pytest.raises(OSError, match="injected state replace failure"):
        installer.promote_bundle(
            source_root=source_b,
            target_root=target,
            expected_bundle_id="bundle-b",
            expected_source_manifest_sha256=sha_b,
            expected_active_bundle_id="bundle-a",
            expected_active_release_id=active["release_id"],
            expected_active_manifest_sha256=active["manifest_sha256"],
        )

    monkeypatch.setattr(installer, "_atomic_write_json", original_atomic_write)
    unchanged = installer.resolve_store(target_root=target)
    assert unchanged["state_sha256"] == initial["state_sha256"]
    assert unchanged["active_bundle_id"] == "bundle-a"
    assert unchanged["rollback_ready"] is False
    # The fully validated B release may remain inert, but no active pointer can
    # observe it and A remains complete.
    assert len([path for path in (target / "releases").iterdir() if path.is_dir()]) == 2


def test_tampered_active_release_blocks_resolve_and_next_promotion(tmp_path: Path):
    _, target, _ = _initialize(tmp_path)
    state_before = (target / installer.STATE_NAME).read_bytes()
    state = json.loads(state_before)
    active_model = target / state["active"]["release_path"] / "models/xgboost_model.pkl"
    active_model.chmod(0o644)
    active_model.write_bytes(b"tampered")
    source_b, sha_b = _make_stage(tmp_path / "sources", "bundle-b")

    with pytest.raises(installer.BundleInstallError, match="exact install inventory"):
        installer.resolve_store(target_root=target)
    with pytest.raises(installer.BundleInstallError, match="exact install inventory"):
        active = state["active"]
        installer.promote_bundle(
            source_root=source_b,
            target_root=target,
            expected_bundle_id="bundle-b",
            expected_source_manifest_sha256=sha_b,
            expected_active_bundle_id="bundle-a",
            expected_active_release_id=active["release_id"],
            expected_active_manifest_sha256=active["manifest_sha256"],
        )
    assert (target / installer.STATE_NAME).read_bytes() == state_before


def test_rollback_requires_both_explicit_expected_bundle_ids(tmp_path: Path):
    _, target, _ = _initialize(tmp_path)
    source_b, sha_b = _make_stage(tmp_path / "sources", "bundle-b")
    _promote_fixture(target, source_b, sha_b)
    state_before = (target / installer.STATE_NAME).read_bytes()
    state = json.loads(state_before)

    with pytest.raises(installer.BundleInstallError, match="explicit expectation"):
        installer.rollback_bundle(
            target_root=target,
            expected_active_bundle_id="wrong-active",
            expected_rollback_bundle_id="bundle-a",
            expected_active_release_id=state["active"]["release_id"],
            expected_rollback_release_id=state["rollback"]["release_id"],
            expected_active_manifest_sha256=state["active"]["manifest_sha256"],
            expected_rollback_manifest_sha256=state["rollback"]["manifest_sha256"],
        )
    with pytest.raises(installer.BundleInstallError, match="explicit expectation"):
        installer.rollback_bundle(
            target_root=target,
            expected_active_bundle_id="bundle-b",
            expected_rollback_bundle_id="wrong-rollback",
            expected_active_release_id=state["active"]["release_id"],
            expected_rollback_release_id=state["rollback"]["release_id"],
            expected_active_manifest_sha256=state["active"]["manifest_sha256"],
            expected_rollback_manifest_sha256=state["rollback"]["manifest_sha256"],
        )
    assert (target / installer.STATE_NAME).read_bytes() == state_before


def test_existing_foreign_lock_is_not_deleted(tmp_path: Path):
    _, target, _ = _initialize(tmp_path)
    source_b, sha_b = _make_stage(tmp_path / "sources", "bundle-b")
    lock = target / installer.LOCK_NAME
    lock.write_text("another process", encoding="utf-8")
    active = _state(target)["active"]

    with pytest.raises(installer.BundleInstallError, match="lock exists"):
        installer.promote_bundle(
            source_root=source_b,
            target_root=target,
            expected_bundle_id="bundle-b",
            expected_source_manifest_sha256=sha_b,
            expected_active_bundle_id="bundle-a",
            expected_active_release_id=active["release_id"],
            expected_active_manifest_sha256=active["manifest_sha256"],
        )

    assert lock.read_text(encoding="utf-8") == "another process"
    assert installer._load_json_object(target / installer.STATE_NAME, label="state")["active"][
        "bundle_id"
    ] == "bundle-a"


def test_source_and_target_roots_must_be_disjoint(tmp_path: Path):
    source, manifest_sha = _make_stage(tmp_path / "sources", "bundle-a")

    with pytest.raises(installer.BundleInstallError, match="inside the source bundle"):
        installer.initialize_store(
            source_root=source,
            target_root=source / "runtime-store",
            expected_bundle_id="bundle-a",
            expected_source_manifest_sha256=manifest_sha,
            allow_no_rollback_test_only=True,
        )


def test_rich_candidate_cannot_initialize_active_store_without_rollback(tmp_path: Path):
    source, manifest_sha = _make_stage(tmp_path / "sources", "bundle-a")
    target = tmp_path / "must-not-activate"

    with pytest.raises(installer.BundleInstallError, match="cannot initialize"):
        installer.initialize_store(
            source_root=source,
            target_root=target,
            expected_bundle_id="bundle-a",
            expected_source_manifest_sha256=manifest_sha,
        )

    assert not target.exists()


def test_legacy_initialize_creates_exact_resolvable_predecessor(
    tmp_path: Path,
    monkeypatch,
):
    legacy = _make_legacy_capture_stub(tmp_path / "sources")
    monkeypatch.setattr(installer, "_validate_legacy_capture", lambda **_kwargs: legacy)
    target = tmp_path / "store"

    result = installer.initialize_legacy_store(
        source_manifest=legacy.source_manifest,
        runtime_manifest=legacy.runtime_manifest,
        capture_manifest=legacy.capture_manifest,
        primary_model=legacy.primary_model,
        no_odds_model=legacy.no_odds_model,
        logistic_model=legacy.logistic_model,
        saved_spec=legacy.saved_spec,
        processed_fights=legacy.processed_fights,
        processed_features=legacy.processed_features,
        target_root=target,
        expected_bundle_id=legacy.bundle_id,
        expected_model_spec_name="legacy-spec",
        expected_snapshot_max_event_date="2026-08-01",
        expected_source_manifest_sha256="1" * 64,
        expected_runtime_manifest_sha256="2" * 64,
        expected_capture_manifest_sha256="3" * 64,
        expected_saved_spec_sha256="4" * 64,
        expected_primary_model_sha256="5" * 64,
        expected_no_odds_model_sha256="6" * 64,
        expected_logistic_model_sha256="7" * 64,
        expected_processed_fights_sha256="8" * 64,
        expected_processed_features_sha256="9" * 64,
    )

    assert result["active_bundle_id"] == "legacy-live"
    assert result["rollback_ready"] is False
    assert Path(result["active_lookup_dir"]).is_dir()
    state = _state(target)
    release = target / state["active"]["release_path"]
    receipt = json.loads((release / installer.RECEIPT_NAME).read_text(encoding="utf-8"))
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    assert receipt["release_kind"] == "legacy_runtime_capture"
    assert manifest["legacy_runtime_capture"]["processed_role"] == (
        "mutable_runtime_lookup_snapshot"
    )
    assert manifest["legacy_runtime_capture"]["claims_immutable_training_snapshot"] is False


def test_generation_lookups_are_isolated_and_follow_rollback(tmp_path: Path):
    _, target, old = _initialize(tmp_path)
    old_lookup = Path(old["active_lookup_dir"])
    old_release_features = Path(old["active_processed_dir"]) / "features.csv"
    immutable_before = old_release_features.read_bytes()
    (old_lookup / "features.csv").write_bytes(b"runtime-enriched-old\n")
    _write_matching_runtime_manifest(target, _state(target)["active"])
    source_b, sha_b = _make_stage(tmp_path / "sources", "bundle-b")

    promoted = _promote_fixture(target, source_b, sha_b)
    new_lookup = Path(promoted["active_lookup_dir"])
    assert new_lookup != old_lookup
    assert new_lookup.parent == old_lookup.parent
    assert (old_lookup / "features.csv").read_bytes() == b"runtime-enriched-old\n"
    assert old_release_features.read_bytes() == immutable_before
    assert (new_lookup / "features.csv").read_bytes() == (
        Path(promoted["active_processed_dir"]) / "features.csv"
    ).read_bytes()

    state = _state(target)
    rollback_snapshot = target / state["rollback"]["lookup_path"]
    assert rollback_snapshot != old_lookup
    assert (rollback_snapshot / "features.csv").read_bytes() == b"runtime-enriched-old\n"
    # A prior process may continue writing its old active path during cutover;
    # the rollback checkpoint is disconnected and must remain unchanged.
    (old_lookup / "features.csv").write_bytes(b"late-writer-after-promotion\n")
    rolled_back = installer.rollback_bundle(
        target_root=target,
        expected_active_bundle_id=state["active"]["bundle_id"],
        expected_rollback_bundle_id=state["rollback"]["bundle_id"],
        expected_active_release_id=state["active"]["release_id"],
        expected_rollback_release_id=state["rollback"]["release_id"],
        expected_active_manifest_sha256=state["active"]["manifest_sha256"],
        expected_rollback_manifest_sha256=state["rollback"]["manifest_sha256"],
    )
    rolled_back_lookup = Path(rolled_back["active_lookup_dir"])
    assert rolled_back_lookup not in {old_lookup, rollback_snapshot}
    assert (rolled_back_lookup / "features.csv").read_bytes() == b"runtime-enriched-old\n"
    assert (old_lookup / "features.csv").read_bytes() == b"late-writer-after-promotion\n"


def test_rollback_rejects_transitional_csv_pair_not_bound_by_runtime_manifest(
    tmp_path: Path,
):
    _, target, _ = _initialize(tmp_path)
    source_b, sha_b = _make_stage(tmp_path / "sources", "bundle-b")
    _promote_fixture(target, source_b, sha_b)
    state_bytes = (target / installer.STATE_NAME).read_bytes()
    state = json.loads(state_bytes)
    active_lookup = target / state["active"]["lookup_path"]
    _write_matching_runtime_manifest(target, state["active"])
    # Simulate the interval after one atomic CSV replacement but before the
    # runtime manifest is reconciled to the new two-file generation.
    (active_lookup / "features.csv").write_bytes(b"new-features-only\n")

    with pytest.raises(installer.BundleInstallError, match="coherent outgoing"):
        installer.rollback_bundle(
            target_root=target,
            expected_active_bundle_id=state["active"]["bundle_id"],
            expected_rollback_bundle_id=state["rollback"]["bundle_id"],
            expected_active_release_id=state["active"]["release_id"],
            expected_rollback_release_id=state["rollback"]["release_id"],
            expected_active_manifest_sha256=state["active"]["manifest_sha256"],
            expected_rollback_manifest_sha256=state["rollback"]["manifest_sha256"],
        )

    assert (target / installer.STATE_NAME).read_bytes() == state_bytes


def test_corrupt_rollback_snapshot_fails_closed_before_state_swap(tmp_path: Path):
    _, target, _ = _initialize(tmp_path)
    source_b, sha_b = _make_stage(tmp_path / "sources", "bundle-b")
    _promote_fixture(target, source_b, sha_b)
    state_bytes = (target / installer.STATE_NAME).read_bytes()
    state = json.loads(state_bytes)
    rollback_features = target / state["rollback"]["lookup_path"] / "features.csv"
    rollback_features.chmod(0o640)
    rollback_features.write_bytes(b"corrupt\n")

    with pytest.raises(installer.BundleInstallError, match="identity mismatch"):
        installer.rollback_bundle(
            target_root=target,
            expected_active_bundle_id=state["active"]["bundle_id"],
            expected_rollback_bundle_id=state["rollback"]["bundle_id"],
            expected_active_release_id=state["active"]["release_id"],
            expected_rollback_release_id=state["rollback"]["release_id"],
            expected_active_manifest_sha256=state["active"]["manifest_sha256"],
            expected_rollback_manifest_sha256=state["rollback"]["manifest_sha256"],
        )

    assert (target / installer.STATE_NAME).read_bytes() == state_bytes


def test_windows_reparse_attribute_fallback_rejects_link_like_path():
    class Python311WindowsPathStub:
        def is_symlink(self):
            return False

        def lstat(self):
            return SimpleNamespace(
                st_file_attributes=getattr(
                    installer.stat,
                    "FILE_ATTRIBUTE_REPARSE_POINT",
                    0x400,
                )
            )

    assert installer._is_link_like(Python311WindowsPathStub()) is True


def test_broken_symlink_ancestor_is_rejected(tmp_path: Path):
    broken = tmp_path / "broken-parent"
    try:
        broken.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlink creation is unavailable on this host: {exc}")

    with pytest.raises(installer.BundleInstallError, match="traverses a symlink"):
        installer._safe_target_root(broken / "store")


@pytest.mark.parametrize("extra_kind", ["file", "directory"])
def test_rich_source_rejects_every_unmanifested_entry_before_target_mutation(
    tmp_path: Path,
    extra_kind: str,
):
    source, manifest_sha = _make_stage(tmp_path / "sources", "bundle-a")
    if extra_kind == "file":
        (source / ".env").write_text("SECRET=value", encoding="utf-8")
    else:
        (source / "unexpected-empty-dir").mkdir()
    target = tmp_path / "must-not-exist"

    with pytest.raises(installer.BundleInstallError, match="exactly its manifest-declared"):
        installer.initialize_store(
            source_root=source,
            target_root=target,
            expected_bundle_id="bundle-a",
            expected_source_manifest_sha256=manifest_sha,
            allow_no_rollback_test_only=True,
        )

    assert not target.exists()


def test_relocated_source_is_validated_through_private_rebased_copy(
    tmp_path: Path,
    monkeypatch,
):
    original, manifest_sha = _make_stage(tmp_path / "original", "bundle-a")
    relocated = tmp_path / "uploaded/bundle"
    shutil.copytree(original, relocated)
    shutil.rmtree(original)
    observed = []

    def semantic_spy(manifest_path: Path, root: Path, *, rich: bool):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed.append((manifest_path, root, rich))
        assert Path(payload["model_path"]) == root / "models/xgboost_model.pkl"
        assert Path(payload["processed_dir"]) == root / "processed"
        assert Path(payload["rich_release_root"]) == root

    monkeypatch.setattr(installer, "_semantic_validate_manifest", semantic_spy)
    validated = installer._validate_source(
        relocated,
        expected_bundle_id="bundle-a",
        expected_manifest_sha256=manifest_sha,
    )

    assert validated.root == relocated.resolve()
    assert len(observed) == 1
    assert observed[0][0].parent != relocated.resolve()


def test_candidate_predecessor_requires_exact_models_and_runtime_lookup(tmp_path: Path):
    active_manifest = tmp_path / "active/manifest.json"
    active_lookup = tmp_path / "lookup"
    _write(active_lookup / "fights_cleaned.csv", b"live fights")
    _write(active_lookup / "features.csv", b"live features")
    active = {
        "bundle_id": "old-live",
        "model_sha256": "1" * 64,
        "no_odds_model_sha256": "2" * 64,
        "logistic_model_sha256": "3" * 64,
    }
    _write(active_manifest, json.dumps(active).encode("utf-8"))
    fights_sha = _sha256(active_lookup / "fights_cleaned.csv")
    features_sha = _sha256(active_lookup / "features.csv")
    candidate = {
        "previous_rollback_identity": {
            "source_manifest": {"payload": {"bundle_id": "old-live"}},
            "local_model_artifacts": {
                "primary": {"sha256": "1" * 64},
                "no_odds": {"sha256": "2" * 64},
                "logistic": {"sha256": "3" * 64},
            },
            "runtime_lookup_hashes": {
                "processed_fights_sha256": fights_sha,
                "processed_features_sha256": features_sha,
            },
            "readyz_evidence": {
                "payload": {
                    "production_bundle": {
                        "bundle_id": "old-live",
                        "processed_fights_sha256": fights_sha,
                        "processed_features_sha256": features_sha,
                        "processed_fights_bytes": len(b"live fights"),
                        "processed_features_bytes": len(b"live features"),
                    }
                }
            },
        }
    }

    REAL_REQUIRE_CANDIDATE_PREDECESSOR(
        candidate,
        active_manifest_path=active_manifest,
        active_lookup_path=active_lookup,
    )
    candidate["previous_rollback_identity"]["local_model_artifacts"]["logistic"][
        "sha256"
    ] = "4" * 64
    with pytest.raises(installer.BundleInstallError, match="logistic model"):
        REAL_REQUIRE_CANDIDATE_PREDECESSOR(
            candidate,
            active_manifest_path=active_manifest,
            active_lookup_path=active_lookup,
        )


def test_exact_local_legacy_capture_passes_read_only_validation_when_available():
    root = Path(".codex_stage/previous_production_runtime_20260806_v1").resolve()
    if not root.is_dir():
        pytest.skip("Local read-only predecessor capture is not present")

    capture = installer._validate_legacy_capture(
        source_manifest=root / "staging_manifest.json",
        runtime_manifest=root / "runtime/manifest.json",
        capture_manifest=root / "rollback_capture_manifest.json",
        primary_model=root / "models/xgboost_model.pkl",
        no_odds_model=root / "models/xgboost_no_odds_model.pkl",
        logistic_model=root / "models/logistic_model.pkl",
        saved_spec=root / "models/full_live_contract_v6_durability_fullfit_spec.json",
        processed_fights=root / "processed/fights_cleaned.csv",
        processed_features=root / "processed/features.csv",
        expected_bundle_id="ufc-production-20260801-full_live_contract_v6_durability_fullfit",
        expected_model_spec_name="full_live_contract_v6_durability_fullfit",
        expected_snapshot_max_event_date="2026-08-01",
        expected_source_manifest_sha256="1435968903b6bd9943f539d8541e0b4404892a79aa73ce408d634572920f48e1",
        expected_runtime_manifest_sha256="e9fc112c6f01ae88a2b9459c84084b2038e5f1361c71ce74bc5ce1c65327305b",
        expected_capture_manifest_sha256="45a67035a7bf17fdf853471a46e46e4b9ad273e9374a980be9213386c7225c90",
        expected_saved_spec_sha256="9253fa7ddaf30e87422bbdc995ff41aa88c4bfc89b2e8e1257db149f1f3b0067",
        expected_primary_model_sha256="5c196935937d2c0847e16183a2e063d223bff48dd659197c7a5cc9b8e7de3530",
        expected_no_odds_model_sha256="a3a4cfb0df525cd8cedaec5f6ca10558cca5a1798129e695a2bd28b3914ae7eb",
        expected_logistic_model_sha256="d36583c59a1b965e099547fbdabc05bb5e64c7c12db49965a265278f71fe5ba2",
        expected_processed_fights_sha256="554a45930060ec7504f5f0313f6a2dddb512ccaa7440c7eff9c229de55c7741e",
        expected_processed_features_sha256="42381a2ca16458cea0f3891da899df351d14a56554ffb1fd0f15c08d5a9cc07a",
    )
    assert capture.bundle_id == "ufc-production-20260801-full_live_contract_v6_durability_fullfit"
