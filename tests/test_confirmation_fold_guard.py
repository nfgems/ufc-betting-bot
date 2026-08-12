from __future__ import annotations

from dataclasses import asdict
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import threading

import numpy as np
import pandas as pd
import pytest

from src.model.training_spec import resolve_named_training_spec
from src.strategy import model_lab
from src.strategy import model_variants
from src.strategy import run_evaluation as run_eval
from src.strategy import confirmation_ledger


def _scheduled_features() -> pd.DataFrame:
    dates = pd.date_range("2014-01-01", "2026-08-01", freq="7D")
    return pd.DataFrame(
        {
            "event_date": dates,
            "fighter_a": [f"A-{idx}" for idx in range(len(dates))],
            "fighter_b": [f"B-{idx}" for idx in range(len(dates))],
            "target": [idx % 2 for idx in range(len(dates))],
            "a_num_fights": 3,
            "b_num_fights": 3,
            "x": [idx / 100.0 for idx in range(len(dates))],
            "a_implied_prob": 0.55,
            "b_implied_prob": 0.45,
        }
    )


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _pin_canonical_origin(monkeypatch: pytest.MonkeyPatch, remote: Path) -> None:
    """Treat the test's bare remote as the canonical ledger identity."""
    monkeypatch.setattr(
        confirmation_ledger,
        "CANONICAL_ORIGIN_IDENTITY",
        confirmation_ledger._normalized_origin_identity(str(remote)),
    )


def test_selection_generator_never_fits_or_scores_final_two_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _scheduled_features()
    variant = model_variants.VariantConfig(name="guard_probe", description="guard")
    scored_dates: list[pd.Timestamp] = []
    fit_count = 0
    preflight_calls = 0
    durable_preflight_calls = 0

    def durable_preflight(manifest, **_kwargs):
        nonlocal durable_preflight_calls
        assert fit_count == 0
        assert len(manifest) >= 3
        durable_preflight_calls += 1
        return "0" * 64

    monkeypatch.setattr(
        run_eval,
        "_preflight_selection_fold_manifest",
        durable_preflight,
    )

    monkeypatch.setattr(
        model_lab,
        "_prepare_evaluation_odds",
        lambda frame, **_kwargs: frame.copy(),
    )
    monkeypatch.setattr(
        model_lab,
        "resolve_variant_feature_columns",
        lambda *_args, **_kwargs: (["x"], ["x"]),
    )
    monkeypatch.setattr(
        model_lab,
        "_select_fold_feature_columns",
        lambda *_args, **_kwargs: (["x"], ["x"]),
    )

    def fake_train(*_args, **_kwargs):
        nonlocal fit_count
        fit_count += 1
        return {"model": object(), "feature_cols": ["x"]}

    def fake_predict(frame, _model_result):
        scored_dates.extend(pd.to_datetime(frame["event_date"]).tolist())
        return frame.copy().assign(prob_a=0.6, prob_b=0.4)

    def fake_market(predictions, *_args, **_kwargs):
        return predictions.assign(a_market_prob=0.55, b_market_prob=0.45), "test"

    monkeypatch.setattr(model_lab, "train_variant_model", fake_train)
    monkeypatch.setattr(model_lab, "_predict_batch_with_model", fake_predict)
    monkeypatch.setattr(model_lab, "_resolve_market_odds", fake_market)
    monkeypatch.setattr(
        model_lab,
        "_production_no_odds_variant",
        lambda: model_variants.VariantConfig(name="no_odds", description="no odds"),
    )

    def preflight(manifest):
        nonlocal preflight_calls
        assert fit_count == 0
        assert len(manifest) >= 3
        preflight_calls += 1

    selected, manifest = model_lab.generate_variant_fold_predictions(
        features,
        variant,
        retrain_months=12,
        initial_train_years=5,
        bet_start_date="2022-01-01",
        evaluation_partition="selection",
        confirmation_fold_count=2,
        return_fold_manifest=True,
        include_mirror_diagnostics=True,
        pre_fit_manifest_callback=preflight,
    )

    selection_manifest, confirmation_manifest = model_lab.partition_walk_forward_folds(
        manifest,
        confirmation_fold_count=2,
    )
    confirmation_start = pd.to_datetime(confirmation_manifest[0][1]["event_date"]).min()
    assert [fold for fold, _frame in selected] == [
        fold for fold, _frame in selection_manifest
    ]
    assert fit_count == 2 * len(selection_manifest)
    assert preflight_calls == 1
    assert durable_preflight_calls == 1
    assert scored_dates
    assert max(scored_dates) < confirmation_start
    assert not any(
        date >= confirmation_start for date in scored_dates
    )
    assert all(
        {
            "mirror_swapped_prob_a",
            "mirror_swapped_no_odds_prob_a",
        }.issubset(frame.columns)
        for _fold, frame in selected
    )


def test_direct_confirmation_generator_rejects_self_signed_forged_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    fake_module = repo / "src" / "strategy" / "model_lab.py"
    fake_module.parent.mkdir(parents=True)
    monkeypatch.setattr(model_lab, "__file__", str(fake_module))
    key_material = {
        "key_contract": "confirmation_sample_consumption_v1",
        "confirmation_evaluation_fight_identity_sha256": "1" * 64,
    }
    global_key = run_eval._canonical_json_sha256(key_material)
    claim_dir = repo / "evidence" / "confirmation_claims" / global_key
    claim_dir.mkdir(parents=True)
    claim_path = claim_dir / run_eval.CONFIRMATION_CLAIM_FILENAME
    claim = {
        "schema_version": 1,
        "protocol": run_eval._CONFIRMATION_PROTOCOL,
        "status": "claimed",
        "global_claim_key": global_key,
        "global_claim_key_material": key_material,
        "claim_path": claim_path.relative_to(repo).as_posix(),
        "global_result_path": (
            claim_dir / run_eval.CONFIRMATION_RESULT_FILENAME
        ).relative_to(repo).as_posix(),
        "lock_path": (
            claim_dir / run_eval.CONFIRMATION_LOCK_FILENAME
        ).relative_to(repo).as_posix(),
        "lock_sha256": "2" * 64,
    }
    run_eval._write_json(claim_path, claim)

    with pytest.raises(ValueError, match="valid package lock"):
        model_lab.generate_variant_fold_predictions(
            _scheduled_features(),
            model_variants.VariantConfig(name="forged", description="forged"),
            evaluation_partition="confirmation",
            confirmation_fold_count=2,
            confirmation_claim_path=claim_path,
            confirmation_claim_sha256=run_eval._file_sha256(claim_path),
        )


def _manifest_folds() -> list[tuple[int, pd.DataFrame]]:
    folds: list[tuple[int, pd.DataFrame]] = []
    for fold, date in enumerate(("2024-01-01", "2025-01-01", "2026-01-01"), 1):
        frame = pd.DataFrame(
            {
                "event_date": pd.to_datetime([date]),
                "fighter_a": [f"A{fold}"],
                "fighter_b": [f"B{fold}"],
                "target": [fold % 2],
                "fold": [fold],
                "train_end": pd.to_datetime([date]),
                "test_end": pd.to_datetime([date]) + pd.Timedelta(days=30),
            }
        )
        folds.append((fold, frame))
    return folds


def test_selection_evidence_rejects_scored_rows_different_from_manifest() -> None:
    manifest = _manifest_folds()
    scored_selection = [(1, manifest[0][1].iloc[0:0].copy())]
    with pytest.raises(ValueError, match="empty bound partition"):
        run_eval._fold_partition_evidence(
            manifest,
            scored_selection,
            reserved_confirmation_folds=2,
        )


def test_loading_legacy_all_fold_stage1_evidence_fails_before_ranking(
    tmp_path: Path,
) -> None:
    cells = tmp_path / run_eval.CELL_OUTPUT_DIRNAME
    cells.mkdir()
    (cells / "legacy_metrics.json").write_text(
        json.dumps(
            {
                "status": "success",
                "cell_key": "legacy",
                "evaluation_sample_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="selection-only evidence"):
        run_eval._load_stage1_results(
            tmp_path,
            reserved_confirmation_folds=2,
        )


def test_legacy_stage3_hydration_cannot_launder_all_fold_results(
    tmp_path: Path,
) -> None:
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be rebound"):
        run_eval._hydrate_legacy_stage3_state(
            tmp_path,
            {"reserved_confirmation_folds": 2},
        )


def _strategy_config() -> dict[str, object]:
    return {
        "name": "locked",
        "s_min_edge": 0.02,
        "s_blend_weight": 0.30,
        "s_model_agreement_min_edge": 0.01,
        "c_min_model_prob": 0.65,
        "c_min_no_odds_prob": 0.50,
        "c_share": 0.50,
        "c_bet_fraction": 0.05,
        "c_max_bet_fraction": 0.08,
        "c_min_edge": 0.0,
        "c_max_decimal_odds": 2.5,
        "m_enabled": False,
        "m_min_model_prob": 0.65,
        "m_min_no_odds_prob": 0.50,
        "m_bet_fraction": 0.03,
        "m_share": 0.0,
    }


def test_locked_strategy_ignores_dormant_near_miss_share() -> None:
    strategy = _strategy_config()
    strategy.update({"c_share": 1.0, "m_share": 0.30, "m_enabled": False})

    assert (
        run_eval._validate_locked_strategy_config(strategy, label="candidate")[
            "c_share"
        ]
        == 1.0
    )

    strategy["m_enabled"] = True
    with pytest.raises(ValueError, match="disable the near-miss trader"):
        run_eval._validate_locked_strategy_config(strategy, label="candidate")


def _selection_evidence() -> dict[str, object]:
    spec = resolve_named_training_spec("full_live_contract_v6_integrity_205")
    columns = list(spec.feature_cols)
    evidence = {
        "evaluation_partition": "selection",
        "reserved_confirmation_folds": 2,
        "selection_fold_ids": [1, 2, 3],
        "confirmation_fold_ids": [4, 5],
        "evaluation_sample_sha256": "1" * 64,
        "n_predictions": 100,
        "n_folds": 3,
        "full_evaluation_sample_sha256": "2" * 64,
        "full_evaluation_n_fights": 140,
        "full_evaluation_n_folds": 5,
        "confirmation_evaluation_sample_sha256": "3" * 64,
        "confirmation_evaluation_n_fights": 40,
        "confirmation_evaluation_n_folds": 2,
        "evaluation_input_value_sha256": "4" * 64,
        "full_evaluation_input_value_sha256": "5" * 64,
        "confirmation_evaluation_input_value_sha256": "6" * 64,
        "feature_contract_columns": columns,
        "feature_contract_count": len(columns),
        "feature_contract_sha256": run_eval._canonical_json_sha256(columns),
        "dataset_fights_sha256": "7" * 64,
        "source_dataset_fights_sha256": "7" * 64,
        "features_artifact_sha256": "8" * 64,
        "features_value_sha256": "9" * 64,
        "source_features_sha256": "8" * 64,
        "odds_source_inventory_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "scheduled_protocol_sha256": "c" * 64,
        "evaluation_protocol_sha256": "d" * 64,
        "source_fingerprint": "e" * 64,
        "source_inventory_sha256": "e" * 64,
        "source_inventory_artifact_sha256": "f" * 64,
        "evaluation_end_date": "2026-08-01",
        "fresh_window_cutoff": "2026-02-01",
    }
    input_provenance = dict(evidence)
    input_provenance["input_provenance_payload_sha256"] = "0" * 64
    evidence["input_provenance"] = input_provenance
    evidence["input_provenance_payload_sha256"] = "0" * 64
    return evidence


def _write_lock_inputs(tmp_path: Path) -> tuple[Path, dict, dict, dict, dict]:
    spec = resolve_named_training_spec("full_live_contract_v6_integrity_205")
    spec_hash = run_eval._canonical_json_sha256(asdict(spec))
    package = {
        "model_variant": "baseline",
        "dataset_variant": "pulled_all_plus_legacy_market",
        "feature_family": "production",
        "calibration_method": spec.calibration_method,
        "retrain_months": 6,
        "model_spec_name": spec.name,
        "model_spec_payload_sha256": spec_hash,
        "strategy_config": _strategy_config(),
    }
    experiment = {
        "schema_version": 1,
        "protocol": run_eval._CONFIRMATION_PROTOCOL,
        "reserved_confirmation_folds": 2,
        "packages": [
            {
                "package_id": "locked-package",
                "package_contract_sha256": run_eval._canonical_json_sha256(package),
            }
        ],
    }
    experiment_path = tmp_path / "experiments.json"
    experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
    source_fingerprint = "a" * 64
    package_manifest = {
        "schema_version": 1,
        "protocol": run_eval._CONFIRMATION_PROTOCOL,
        "package_id": "locked-package",
        "generic_stage3_search_used": False,
        "prior_all_fold_baseline_visibility": True,
        "baseline_visibility_limitation": "The all-fold baseline aggregate was observed.",
        "source_fingerprint": source_fingerprint,
        "predeclared_experiment_manifest_path": experiment_path.name,
        "predeclared_experiment_manifest_sha256": run_eval._file_sha256(
            experiment_path
        ),
        "candidate_package": package,
        "selection_evidence": _selection_evidence(),
    }
    package_path = tmp_path / "package.json"
    package_path.write_text(json.dumps(package_manifest), encoding="utf-8")

    control_dir = tmp_path / "control"
    control_dir.mkdir()
    (control_dir / "checksums.json").write_text("{}", encoding="utf-8")
    receipt = {
        **_selection_evidence(),
        "evaluation_index_sha256": "4" * 64,
    }
    (control_dir / "fixed_control_bootstrap_receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    control_metrics = {
        **_selection_evidence(),
        "evaluation_index_sha256": "4" * 64,
        "model_variant": "baseline",
        "dataset_variant": "pulled_all_plus_legacy_market",
        "feature_family": "production",
        "calibration_method": spec.calibration_method,
        "retrain_months": 6,
        "model_spec_name": spec.name,
        "model_spec_payload_sha256": spec_hash,
    }
    control_sweep = {"config": _strategy_config()}
    metadata = {
        "source_fingerprint": source_fingerprint,
        "git_sha": "deadbeef",
        "execution_mode": "realistic",
        "entry_offset_days": 1.0,
        "entry_offset_for_features": True,
        "require_entry_odds": True,
        "allow_closing_odds": False,
        "reserved_confirmation_folds": 2,
        "bootstrap": 5000,
    }
    return package_path, metadata, control_metrics, control_sweep, {
        "path": str(control_dir)
    }


def test_confirmation_lock_is_immutable_and_claim_is_single_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    run_one = repo_root / "run_one"
    run_two = repo_root / "run_two"
    run_one.mkdir(parents=True)
    run_two.mkdir(parents=True)
    monkeypatch.setattr(run_eval, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(
        run_eval,
        "require_remotely_anchored_git_artifacts",
        lambda *_args, **_kwargs: None,
    )
    selection_identities = [
        {
            "event_date": "2025-12-01",
            "fighter_name_low": "selection-alpha",
            "fighter_name_high": "selection-beta",
        }
    ]
    run_eval._record_global_selection_exposure(selection_identities)
    identities = [
        {
            "event_date": "2026-01-01",
            "fighter_name_low": "alpha",
            "fighter_name_high": "beta",
        }
    ]
    selection = {
        "evaluation_sample_sha256": "1" * 64,
        "evaluation_fight_identity_sha256": run_eval._canonical_json_sha256(
            selection_identities
        ),
        "selection_fight_identities": selection_identities,
        "full_evaluation_sample_sha256": "2" * 64,
        "confirmation_evaluation_sample_sha256": "3" * 64,
        "confirmation_evaluation_fight_identity_sha256": (
            run_eval._canonical_json_sha256(identities)
        ),
        "confirmation_fight_identities": identities,
        "confirmation_evaluation_input_value_sha256": "4" * 64,
        "confirmation_evaluation_n_fights": 1,
        "confirmation_fold_ids": [9, 10],
    }
    lock = {
        "protocol": run_eval._CONFIRMATION_PROTOCOL,
        "package_id": "locked-package",
        "source_fingerprint": "5" * 64,
        "selection_evidence": selection,
        "frozen_control": {"bootstrap_receipt_sha256": "6" * 64},
    }
    lock_path = run_one / run_eval.CONFIRMATION_LOCK_FILENAME
    run_eval._write_json(lock_path, lock)
    lock_hash = run_eval._file_sha256(lock_path)
    claim, claim_hash, _claim_path = run_eval._claim_confirmation_once(
        run_dir=run_one,
        lock_payload=lock,
        lock_sha256=lock_hash,
    )
    assert claim["status"] == "claimed"
    assert len(claim_hash) == 64
    second_lock = {
        **lock,
        "package_id": "different-package",
        "source_fingerprint": "7" * 64,
        "selection_evidence": {
            **selection,
            "confirmation_evaluation_input_value_sha256": "8" * 64,
        },
        "frozen_control": {"bootstrap_receipt_sha256": "9" * 64},
    }
    second_lock_path = run_two / run_eval.CONFIRMATION_LOCK_FILENAME
    run_eval._write_json(second_lock_path, second_lock)
    with pytest.raises(ValueError, match="overlaps|consumed"):
        run_eval._claim_confirmation_once(
            run_dir=run_two,
            lock_payload=second_lock,
            lock_sha256=run_eval._file_sha256(second_lock_path),
        )

    run_three = repo_root / "run_three"
    run_three.mkdir()
    partly_overlapping_identities = [
        identities[0],
        {
            "event_date": "2026-02-01",
            "fighter_name_low": "delta",
            "fighter_name_high": "gamma",
        },
    ]
    partial_lock = {
        **second_lock,
        "selection_evidence": {
            **selection,
            "confirmation_evaluation_fight_identity_sha256": (
                run_eval._canonical_json_sha256(partly_overlapping_identities)
            ),
            "confirmation_fight_identities": partly_overlapping_identities,
            "confirmation_evaluation_n_fights": 2,
        },
    }
    partial_lock_path = run_three / run_eval.CONFIRMATION_LOCK_FILENAME
    run_eval._write_json(partial_lock_path, partial_lock)
    with pytest.raises(ValueError, match="overlaps|consumed"):
        run_eval._claim_confirmation_once(
            run_dir=run_three,
            lock_payload=partial_lock,
            lock_sha256=run_eval._file_sha256(partial_lock_path),
        )


def test_orphan_confirmation_marker_blocks_via_atomic_marker_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    monkeypatch.setattr(run_eval, "_repo_root", lambda: repo)
    identity = {
        "event_date": "2026-01-01",
        "fighter_name_low": "alpha",
        "fighter_name_high": "beta",
    }
    identity_sha256 = run_eval._canonical_json_sha256(identity)
    marker = (
        repo
        / run_eval.GLOBAL_CONFIRMATION_CLAIMS_DIR
        / "_fight_reservations"
        / f"{identity_sha256}.json"
    )
    run_eval._write_json(
        marker,
        {
            "schema_version": 1,
            "protocol": run_eval._CONFIRMATION_PROTOCOL,
            "global_claim_key": "a" * 64,
            "fight_identity_sha256": identity_sha256,
            "fight_identity": identity,
        },
    )

    with pytest.raises(ValueError, match="fight identity was already consumed"):
        run_eval._reserve_global_confirmation_fights(
            global_claim_key="b" * 64,
            identities=[identity],
        )


@pytest.mark.parametrize("partially_overlapping", [False, True])
def test_concurrent_confirmation_reservations_cannot_both_succeed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partially_overlapping: bool,
) -> None:
    repo = tmp_path / "repo"
    monkeypatch.setattr(run_eval, "_repo_root", lambda: repo)
    shared = {
        "event_date": "2026-01-01",
        "fighter_name_low": "alpha",
        "fighter_name_high": "beta",
    }
    left = [shared]
    right = [shared]
    if partially_overlapping:
        left.append(
            {
                "event_date": "2026-02-01",
                "fighter_name_low": "charlie",
                "fighter_name_high": "delta",
            }
        )
        right.append(
            {
                "event_date": "2026-03-01",
                "fighter_name_low": "echo",
                "fighter_name_high": "foxtrot",
            }
        )
    barrier = threading.Barrier(2)

    def reserve(key: str, identities: list[dict[str, str]]) -> bool:
        barrier.wait()
        try:
            run_eval._reserve_global_confirmation_fights(
                global_claim_key=key,
                identities=identities,
            )
        except ValueError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(
            executor.map(
                lambda args: reserve(*args),
                [("1" * 64, left), ("2" * 64, right)],
            )
        )
    assert outcomes.count(True) <= 1
    assert outcomes.count(False) >= 1


def test_confirmation_rejects_selection_overlap_and_backward_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    monkeypatch.setattr(run_eval, "_repo_root", lambda: repo)
    monkeypatch.setattr(
        run_eval,
        "require_remotely_anchored_git_artifacts",
        lambda *_args, **_kwargs: None,
    )
    exposed = {
        "event_date": "2026-02-01",
        "fighter_name_low": "alpha",
        "fighter_name_high": "beta",
    }
    run_eval._record_global_selection_exposure([exposed])

    with pytest.raises(ValueError, match="previously exposed selection"):
        run_eval._reserve_global_confirmation_fights(
            global_claim_key="1" * 64,
            identities=[exposed],
        )
    with pytest.raises(ValueError, match="strictly forward"):
        run_eval._reserve_global_confirmation_fights(
            global_claim_key="2" * 64,
            identities=[
                {
                    "event_date": "2026-01-15",
                    "fighter_name_low": "charlie",
                    "fighter_name_high": "delta",
                }
            ],
        )

    run_eval._reserve_global_confirmation_fights(
        global_claim_key="3" * 64,
        identities=[
            {
                "event_date": "2026-03-01",
                "fighter_name_low": "echo",
                "fighter_name_high": "foxtrot",
            }
        ],
    )


def test_confirmation_claim_pauses_until_remote_anchor_then_resumes_exact_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    run_dir = repo / "run"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(run_eval, "_repo_root", lambda: repo)
    claim_anchored = False

    def require_anchor(_repo, paths, *, label):
        if any(Path(path).name == run_eval.CONFIRMATION_CLAIM_FILENAME for path in paths):
            if not claim_anchored:
                raise ValueError(f"{label} must be committed and pushed")

    monkeypatch.setattr(
        run_eval,
        "require_remotely_anchored_git_artifacts",
        require_anchor,
    )
    selection_identities = [
        {
            "event_date": "2026-01-01",
            "fighter_name_low": "selection-a",
            "fighter_name_high": "selection-b",
        }
    ]
    run_eval._record_global_selection_exposure(selection_identities)
    confirmation_identities = [
        {
            "event_date": "2026-02-01",
            "fighter_name_low": "confirm-a",
            "fighter_name_high": "confirm-b",
        }
    ]
    lock = {
        "protocol": run_eval._CONFIRMATION_PROTOCOL,
        "package_id": "winner",
        "source_fingerprint": "1" * 64,
        "selection_evidence": {
            "evaluation_sample_sha256": "2" * 64,
            "evaluation_fight_identity_sha256": run_eval._canonical_json_sha256(
                selection_identities
            ),
            "selection_fight_identities": selection_identities,
            "full_evaluation_sample_sha256": "3" * 64,
            "confirmation_evaluation_sample_sha256": "4" * 64,
            "confirmation_evaluation_fight_identity_sha256": (
                run_eval._canonical_json_sha256(confirmation_identities)
            ),
            "confirmation_fight_identities": confirmation_identities,
            "confirmation_evaluation_input_value_sha256": "5" * 64,
            "confirmation_evaluation_n_fights": 1,
            "confirmation_fold_ids": [9, 10],
        },
        "frozen_control": {"bootstrap_receipt_sha256": "6" * 64},
    }
    lock_path = run_dir / run_eval.CONFIRMATION_LOCK_FILENAME
    run_eval._write_json(lock_path, lock)
    lock_sha256 = run_eval._file_sha256(lock_path)

    with pytest.raises(ValueError, match="committed and pushed"):
        run_eval._claim_confirmation_once(
            run_dir=run_dir,
            lock_payload=lock,
            lock_sha256=lock_sha256,
        )
    reference_path = run_dir / run_eval.CONFIRMATION_CLAIM_REFERENCE_FILENAME
    first_reference = json.loads(reference_path.read_text(encoding="utf-8"))

    claim_anchored = True
    claim, claim_sha256, claim_path = run_eval._claim_confirmation_once(
        run_dir=run_dir,
        lock_payload=lock,
        lock_sha256=lock_sha256,
    )
    assert claim["status"] == "claimed"
    assert claim_sha256 == first_reference["claim_sha256"]
    assert claim_path == repo / first_reference["claim_path"]

    fake_model_lab_path = repo / "src" / "strategy" / "model_lab.py"
    fake_model_lab_path.parent.mkdir(parents=True)
    monkeypatch.setattr(model_lab, "__file__", str(fake_model_lab_path))

    def reject_unanchored(*_args, **_kwargs):
        raise ValueError("remote anchor required")

    monkeypatch.setattr(
        model_lab,
        "require_remotely_anchored_git_artifacts",
        reject_unanchored,
    )
    with pytest.raises(ValueError, match="remote anchor required"):
        model_lab.generate_variant_fold_predictions(
            _scheduled_features(),
            model_variants.VariantConfig(name="guard", description="guard"),
            evaluation_partition="confirmation",
            confirmation_fold_count=2,
            confirmation_claim_path=claim_path,
            confirmation_claim_sha256=claim_sha256,
        )


def _integrity_spec_frame(spec) -> pd.DataFrame:
    base = _scheduled_features()
    extra = {
        column: np.linspace(0.0, 1.0, len(base)) + float(index)
        for index, column in enumerate(spec.feature_cols)
        if column not in base.columns
    }
    return pd.concat([base, pd.DataFrame(extra, index=base.index)], axis=1)


def _locked_confirmation_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    """Create a real durable claim chain for the locked integrity-205 package.

    The claim/lock/exposure/marker store is written by the real writers into a
    temp repository; only remote anchoring (tested with real git elsewhere) and
    model fitting are stubbed.
    """
    repo = tmp_path / "repo"
    run_dir = repo / "run"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(run_eval, "_repo_root", lambda: repo)
    fake_model_lab_path = repo / "src" / "strategy" / "model_lab.py"
    fake_model_lab_path.parent.mkdir(parents=True)
    monkeypatch.setattr(model_lab, "__file__", str(fake_model_lab_path))
    monkeypatch.setattr(
        run_eval,
        "require_remotely_anchored_git_artifacts",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        model_lab,
        "require_remotely_anchored_git_artifacts",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        model_lab,
        "_prepare_evaluation_odds",
        lambda frame, **_kwargs: frame.copy(),
    )
    monkeypatch.setattr(
        run_eval,
        "_preflight_selection_fold_manifest",
        lambda *_args, **_kwargs: "0" * 64,
    )

    spec_name = "full_live_contract_v6_integrity_205"
    spec = resolve_named_training_spec(spec_name)
    feature_cols = list(spec.feature_cols)
    features = _integrity_spec_frame(spec)
    variant = model_variants.variant_from_named_training_spec(
        spec_name,
        variant_name="baseline",
    )

    fit_calls: list[int] = []

    def fake_train(*_args, **_kwargs):
        fit_calls.append(1)
        return {"model": object(), "feature_cols": feature_cols}

    monkeypatch.setattr(model_lab, "train_variant_model", fake_train)
    monkeypatch.setattr(
        model_lab,
        "_predict_batch_with_model",
        lambda frame, _model_result: frame.copy().assign(prob_a=0.6, prob_b=0.4),
    )
    monkeypatch.setattr(
        model_lab,
        "_resolve_market_odds",
        lambda predictions, *_args, **_kwargs: (
            predictions.assign(a_market_prob=0.55, b_market_prob=0.45),
            "test",
        ),
    )
    monkeypatch.setattr(
        model_lab,
        "_production_no_odds_variant",
        lambda: model_variants.VariantConfig(name="no_odds", description="no odds"),
    )
    monkeypatch.setattr(
        model_lab,
        "_select_fold_feature_columns",
        lambda *_args, **_kwargs: (feature_cols, feature_cols),
    )

    _selected, manifest = model_lab.generate_variant_fold_predictions(
        features,
        variant,
        retrain_months=12,
        initial_train_years=5,
        bet_start_date="2022-01-01",
        feature_cols=feature_cols,
        evaluation_partition="selection",
        confirmation_fold_count=2,
        return_fold_manifest=True,
    )
    fit_calls.clear()
    selection_manifest, confirmation_manifest = (
        model_lab.partition_walk_forward_folds(
            manifest,
            confirmation_fold_count=2,
        )
    )
    selection_index = run_eval._post_cutoff_predictions(selection_manifest)
    confirmation_index = run_eval._post_cutoff_predictions(confirmation_manifest)
    selection_identities = run_eval._canonical_evaluation_fight_identities(
        selection_index
    )
    run_eval._record_global_selection_exposure(selection_identities)

    lock = {
        "protocol": run_eval._CONFIRMATION_PROTOCOL,
        "package_id": "locked-winner",
        "source_fingerprint": "1" * 64,
        "candidate_package": {
            "model_variant": "baseline",
            "dataset_variant": "pulled_all_plus_legacy_market",
            "feature_family": "production",
            "calibration_method": spec.calibration_method,
            "retrain_months": 12,
            "model_spec_name": spec_name,
            "model_spec_payload_sha256": run_eval._canonical_json_sha256(
                asdict(spec)
            ),
        },
        "selection_evidence": {
            "evaluation_sample_sha256": run_eval._evaluation_sample_sha256(
                selection_index
            ),
            "evaluation_fight_identity_sha256": run_eval._canonical_json_sha256(
                selection_identities
            ),
            "selection_fight_identities": selection_identities,
            "full_evaluation_sample_sha256": run_eval._evaluation_sample_sha256(
                run_eval._post_cutoff_predictions(manifest)
            ),
            "confirmation_evaluation_sample_sha256": (
                run_eval._evaluation_sample_sha256(confirmation_index)
            ),
            "confirmation_evaluation_fight_identity_sha256": (
                run_eval._evaluation_fight_identity_sha256(confirmation_index)
            ),
            "confirmation_fight_identities": (
                run_eval._canonical_evaluation_fight_identities(
                    confirmation_index
                )
            ),
            "confirmation_evaluation_input_value_sha256": (
                run_eval._manifest_input_value_sha256(
                    confirmation_index,
                    feature_contract_columns=feature_cols,
                )
            ),
            "confirmation_evaluation_n_fights": int(len(confirmation_index)),
            "confirmation_fold_ids": [
                fold for fold, _frame in confirmation_manifest
            ],
            "feature_contract_columns": feature_cols,
        },
        "frozen_control": {"bootstrap_receipt_sha256": "6" * 64},
    }
    lock_path = run_dir / run_eval.CONFIRMATION_LOCK_FILENAME
    run_eval._write_json(lock_path, lock)
    _claim, claim_sha256, claim_path = run_eval._claim_confirmation_once(
        run_dir=run_dir,
        lock_payload=lock,
        lock_sha256=run_eval._file_sha256(lock_path),
    )
    return {
        "features": features,
        "variant": variant,
        "feature_cols": feature_cols,
        "claim_path": claim_path,
        "claim_sha256": claim_sha256,
        "fit_calls": fit_calls,
        "confirmation_manifest": confirmation_manifest,
    }


def _confirmation_generate_kwargs(fixture: dict[str, object]) -> dict[str, object]:
    return {
        "retrain_months": 12,
        "initial_train_years": 5,
        "bet_start_date": "2022-01-01",
        "feature_cols": fixture["feature_cols"],
        "evaluation_partition": "confirmation",
        "confirmation_fold_count": 2,
        "confirmation_claim_path": fixture["claim_path"],
        "confirmation_claim_sha256": fixture["claim_sha256"],
    }


def test_confirmation_fit_rejects_non_locked_spec_before_any_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DIR-AUD-P2-015: an orphan claim cannot fit a substituted variant/spec."""
    fixture = _locked_confirmation_fixture(tmp_path, monkeypatch)
    rogue = model_variants.VariantConfig(name="rogue", description="rogue")
    with pytest.raises(ValueError, match="locked candidate/control package"):
        model_lab.generate_variant_fold_predictions(
            fixture["features"],
            rogue,
            **_confirmation_generate_kwargs(fixture),
        )
    assert not fixture["fit_calls"]


def test_confirmation_fit_rejects_substituted_features_frame_before_any_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DIR-AUD-P2-015: an orphan claim cannot fit a different features frame."""
    fixture = _locked_confirmation_fixture(tmp_path, monkeypatch)
    tampered = fixture["features"].copy()
    target_fighter = fixture["confirmation_manifest"][0][1]["fighter_a"].iloc[0]
    row_mask = tampered["fighter_a"] == target_fighter
    assert row_mask.any()
    tampered.loc[row_mask, fixture["feature_cols"][0]] = (
        tampered.loc[row_mask, fixture["feature_cols"][0]] + 1.0
    )
    with pytest.raises(ValueError, match="differs from the locked package inputs"):
        model_lab.generate_variant_fold_predictions(
            tampered,
            fixture["variant"],
            **_confirmation_generate_kwargs(fixture),
        )
    assert not fixture["fit_calls"]


def test_confirmation_fit_accepts_exact_locked_package_and_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _locked_confirmation_fixture(tmp_path, monkeypatch)
    confirmation_folds = model_lab.generate_variant_fold_predictions(
        fixture["features"],
        fixture["variant"],
        **_confirmation_generate_kwargs(fixture),
    )
    assert fixture["fit_calls"]
    assert [fold for fold, _frame in confirmation_folds] == [
        fold for fold, _frame in fixture["confirmation_manifest"]
    ]


def test_git_anchor_requires_committed_pushed_exact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = tmp_path / "remote.git"
    repo = tmp_path / "repo"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "init", "-b", "main", str(repo))
    _git(repo, "config", "user.name", "Integrity Test")
    _git(repo, "config", "user.email", "integrity@example.invalid")
    _git(repo, "remote", "add", "origin", str(remote))
    _pin_canonical_origin(monkeypatch, remote)
    baseline = repo / "baseline.txt"
    baseline.write_text("baseline\n", encoding="utf-8")
    (repo / ".gitattributes").write_text(
        "evidence/confirmation_claims/** -text\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".gitattributes", "baseline.txt")
    _git(repo, "commit", "-m", "baseline")
    _git(repo, "push", "-u", "origin", "main")
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")

    artifact = repo / "evidence" / "confirmation_claims" / "anchor.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"status":"claimed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="untracked"):
        confirmation_ledger.require_remotely_anchored_git_artifacts(
            repo,
            [artifact],
            label="claim",
        )

    _git(repo, "add", artifact.relative_to(repo).as_posix())
    with pytest.raises(ValueError, match="bytes differ from HEAD"):
        confirmation_ledger.require_remotely_anchored_git_artifacts(
            repo,
            [artifact],
            label="claim",
        )

    _git(repo, "commit", "-m", "claim")
    with pytest.raises(ValueError, match="origin default-branch tip"):
        confirmation_ledger.require_remotely_anchored_git_artifacts(
            repo,
            [artifact],
            label="claim",
        )

    _git(repo, "push")
    anchored_tip = confirmation_ledger.require_remotely_anchored_git_artifacts(
        repo,
        [artifact],
        label="claim",
    )
    assert anchored_tip == _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "unrelated")
    (repo / "unrelated.txt").write_text("unrelated branch\n", encoding="utf-8")
    _git(repo, "add", "unrelated.txt")
    _git(repo, "commit", "-m", "unrelated branch tip")
    _git(repo, "push", "-u", "origin", "unrelated")
    _git(repo, "checkout", "main")
    _git(repo, "checkout", "--detach", "HEAD")
    confirmation_ledger.require_remotely_anchored_git_artifacts(
        repo,
        [artifact],
        label="claim",
    )
    _git(repo, "checkout", "main")
    artifact.write_text('{"status":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bytes differ from HEAD"):
        confirmation_ledger.require_remotely_anchored_git_artifacts(
            repo,
            [artifact],
            label="claim",
        )

    # DIR-NRA-P3-013: a locally substituted origin — even a fully controlled
    # mirror of the same history — must not satisfy the remote anchor.
    impostor = tmp_path / "impostor.git"
    _git(tmp_path, "init", "--bare", str(impostor))
    _git(repo, "remote", "set-url", "origin", str(impostor))
    with pytest.raises(ValueError, match="not the canonical ledger remote"):
        confirmation_ledger.require_remotely_anchored_git_artifacts(
            repo,
            [artifact],
            label="claim",
        )


def test_recorded_anchor_must_exist_in_local_history(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _git(tmp_path, "init", "-b", "main", str(repo))
    _git(repo, "config", "user.name", "Integrity Test")
    _git(repo, "config", "user.email", "integrity@example.invalid")
    (repo / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "baseline.txt")
    _git(repo, "commit", "-m", "baseline")
    head = _git(repo, "rev-parse", "HEAD")

    assert (
        confirmation_ledger.require_recorded_anchor_in_history(
            repo,
            head,
            label="confirmation result",
        )
        == head
    )
    with pytest.raises(ValueError, match="records no valid anchored origin tip"):
        confirmation_ledger.require_recorded_anchor_in_history(
            repo,
            None,
            label="confirmation result",
        )
    with pytest.raises(ValueError, match="ancestor|verification failed"):
        confirmation_ledger.require_recorded_anchor_in_history(
            repo,
            "f" * 40,
            label="confirmation result",
        )


def test_git_anchor_rejects_feature_branch_only_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = tmp_path / "remote.git"
    repo = tmp_path / "repo"
    default_clone = tmp_path / "default-clone"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "init", "-b", "main", str(repo))
    _git(repo, "config", "user.name", "Integrity Test")
    _git(repo, "config", "user.email", "integrity@example.invalid")
    _git(repo, "remote", "add", "origin", str(remote))
    _pin_canonical_origin(monkeypatch, remote)
    (repo / ".gitattributes").write_text(
        "evidence/confirmation_claims/** -text\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".gitattributes")
    _git(repo, "commit", "-m", "baseline")
    _git(repo, "push", "-u", "origin", "main")
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")

    _git(repo, "checkout", "-b", "claim-feature")
    artifact = repo / "evidence" / "confirmation_claims" / "feature-only.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"status":"claimed"}\n', encoding="utf-8")
    _git(repo, "add", artifact.relative_to(repo).as_posix())
    _git(repo, "commit", "-m", "feature-only claim")
    _git(repo, "push", "-u", "origin", "claim-feature")

    with pytest.raises(ValueError, match="origin default-branch tip"):
        confirmation_ledger.require_remotely_anchored_git_artifacts(
            repo,
            [artifact],
            label="claim",
        )
    _git(tmp_path, "clone", "--branch", "main", str(remote), str(default_clone))
    assert not (default_clone / artifact.relative_to(repo)).exists()


def test_fresh_clone_refuses_remotely_anchored_confirmation_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = tmp_path / "remote.git"
    source = tmp_path / "source"
    clone = tmp_path / "clone"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "init", "-b", "main", str(source))
    _git(source, "config", "user.name", "Integrity Test")
    _git(source, "config", "user.email", "integrity@example.invalid")
    _git(source, "remote", "add", "origin", str(remote))
    _pin_canonical_origin(monkeypatch, remote)
    monkeypatch.setattr(run_eval, "_repo_root", lambda: source)

    selection_identities = [
        {
            "event_date": "2026-01-01",
            "fighter_name_low": "selection-a",
            "fighter_name_high": "selection-b",
        }
    ]
    exposure_material = run_eval._selection_exposure_key_material(selection_identities)
    exposure_key = run_eval._canonical_json_sha256(exposure_material)
    exposure_path = run_eval._selection_exposure_path(selection_identities)
    run_eval._write_json(
        exposure_path,
        {
            "schema_version": 1,
            "protocol": run_eval._CONFIRMATION_PROTOCOL,
            "status": "selection_exposed",
            "selection_exposure_key": exposure_key,
            "selection_exposure_key_material": exposure_material,
            "selection_evaluation_fight_identity_sha256": (
                run_eval._canonical_json_sha256(selection_identities)
            ),
            "selection_fight_identities": selection_identities,
            "selection_min_event_date": "2026-01-01",
            "selection_max_event_date": "2026-01-01",
            "recorded_at": "2026-08-09T00:00:00+00:00",
        },
    )
    confirmation_identities = [
        {
            "event_date": "2026-02-01",
            "fighter_name_low": "confirm-a",
            "fighter_name_high": "confirm-b",
        }
    ]
    key_material = {
        "key_contract": "confirmation_sample_consumption_v1",
        "confirmation_evaluation_fight_identity_sha256": (
            run_eval._canonical_json_sha256(confirmation_identities)
        ),
    }
    global_key = run_eval._canonical_json_sha256(key_material)
    prior_claim_path = (
        source
        / run_eval.GLOBAL_CONFIRMATION_CLAIMS_DIR
        / global_key
        / run_eval.CONFIRMATION_CLAIM_FILENAME
    )
    run_eval._write_json(
        prior_claim_path,
        {
            "schema_version": 1,
            "protocol": run_eval._CONFIRMATION_PROTOCOL,
            "status": "claimed",
            "global_claim_key": global_key,
            "confirmation_fight_identities": confirmation_identities,
        },
    )
    (source / ".gitattributes").write_text(
        "evidence/confirmation_claims/** -text\n",
        encoding="utf-8",
    )
    _git(source, "add", ".gitattributes", "evidence/confirmation_claims")
    _git(source, "commit", "-m", "anchor confirmation consumption")
    _git(source, "push", "-u", "origin", "main")
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(tmp_path, "clone", "--branch", "main", str(remote), str(clone))
    assert run_eval._file_sha256(prior_claim_path) == run_eval._file_sha256(
        clone / prior_claim_path.relative_to(source)
    )
    assert run_eval._file_sha256(exposure_path) == run_eval._file_sha256(
        clone / exposure_path.relative_to(source)
    )
    monkeypatch.setattr(run_eval, "_repo_root", lambda: clone)

    run_dir = clone / "run"
    run_dir.mkdir()
    lock = {
        "protocol": run_eval._CONFIRMATION_PROTOCOL,
        "package_id": "retry",
        "source_fingerprint": "1" * 64,
        "selection_evidence": {
            "evaluation_sample_sha256": "2" * 64,
            "evaluation_fight_identity_sha256": run_eval._canonical_json_sha256(
                selection_identities
            ),
            "selection_fight_identities": selection_identities,
            "full_evaluation_sample_sha256": "3" * 64,
            "confirmation_evaluation_sample_sha256": "4" * 64,
            "confirmation_evaluation_fight_identity_sha256": (
                run_eval._canonical_json_sha256(confirmation_identities)
            ),
            "confirmation_fight_identities": confirmation_identities,
            "confirmation_evaluation_input_value_sha256": "5" * 64,
            "confirmation_evaluation_n_fights": 1,
            "confirmation_fold_ids": [9, 10],
        },
        "frozen_control": {"bootstrap_receipt_sha256": "6" * 64},
    }
    lock_path = run_dir / run_eval.CONFIRMATION_LOCK_FILENAME
    run_eval._write_json(lock_path, lock)
    with pytest.raises(ValueError, match="previously exposed selection or claim"):
        run_eval._claim_confirmation_once(
            run_dir=run_dir,
            lock_payload=lock,
            lock_sha256=run_eval._file_sha256(lock_path),
        )


def test_reserved_stage4_refuses_generic_selection_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="confirmation-package-manifest"):
        run_eval._run_stage4(
            run_dir=tmp_path,
            freeze_id="control",
            finalists=[],
            sweep_results={},
            metadata={"reserved_confirmation_folds": 2},
        )


def test_confirmation_fight_identity_survives_orientation_ids_and_reviewed_aliases() -> None:
    without_ids = pd.DataFrame(
        {
            "event_date": ["2026-08-01"],
            "fighter_a": ["Asu Almabaev"],
            "fighter_b": ["Opponent Name"],
        }
    )
    swapped_with_ids = pd.DataFrame(
        {
            "event_date": ["2026-08-01"],
            "fighter_a": ["Opponent Name"],
            "fighter_b": ["Asu Almabayev"],
            "fighter_a_id": ["opponent-id"],
            "fighter_b_id": ["asu-id"],
        }
    )

    assert run_eval._canonical_evaluation_fight_identities(without_ids) == (
        run_eval._canonical_evaluation_fight_identities(swapped_with_ids)
    )
    assert run_eval._evaluation_fight_identity_sha256(without_ids) == (
        run_eval._evaluation_fight_identity_sha256(swapped_with_ids)
    )


def _diagnostic_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    predictions = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "fighter_a": ["A1", "A2"],
            "fighter_b": ["B1", "B2"],
            "target": [1, 0],
            "fold": [1, 2],
            "train_end": pd.to_datetime(["2024-12-01", "2025-01-01"]),
            "test_end": pd.to_datetime(["2025-01-31", "2025-02-28"]),
            "prob_a": [0.70, 0.40],
            "prob_b": [0.30, 0.60],
            "no_odds_prob_a": [0.65, 0.45],
            "no_odds_prob_b": [0.35, 0.55],
            "mirror_swapped_prob_a": [0.30, 0.60],
            "mirror_swapped_no_odds_prob_a": [0.35, 0.55],
            "a_market_prob": [0.60, 0.45],
            "b_market_prob": [0.40, 0.55],
            "odds_source": ["line_history", "odds_api"],
        }
    )
    selection_index = predictions[
        [
            "event_date",
            "fighter_a",
            "fighter_b",
            "target",
            "fold",
            "train_end",
            "test_end",
        ]
    ].copy()
    selection_index["feature_one"] = [1.0, 2.0]
    selection_index["feature_two"] = [3.0, np.nan]
    bet_log = pd.DataFrame(
        {
            "event_date": predictions["event_date"],
            "fighter_a": predictions["fighter_a"],
            "fighter_b": predictions["fighter_b"],
            "fold": [1, 2],
            "trader": ["S", "C"],
            "side": ["a", "a"],
            "bet_on": ["A1", "A2"],
            "bet_size": [20.0, 10.0],
            "profit": [10.0, -10.0],
            "won": [True, False],
            "edge": [0.10, 0.05],
            "odds_source": ["line_history", "odds_api"],
        }
    )
    return predictions, selection_index, {"bet_log": bet_log}


def test_bounded_diagnostics_bind_strategy_pnl_slices_and_real_swapped_inference() -> None:
    predictions, selection_index, sweep = _diagnostic_frames()
    diagnostics = run_eval._bounded_selection_diagnostics(
        predictions=predictions,
        selection_index=selection_index,
        feature_contract_columns=["feature_one", "feature_two"],
        sliced_metrics={"by_year": {"2025": {"n_samples": 2}}},
        sweep=sweep,
    )

    mirror = diagnostics["mirror_symmetry"]
    assert mirror["check"] == "actual_ab_swapped_feature_inference"
    assert mirror["passed"] is True
    assert len(mirror["inference_receipt_sha256"]) == 64
    strategy = diagnostics["strategy_performance"]
    assert strategy["overall"] == {
        "n_bets": 2,
        "wins": 1,
        "total_wagered": 30.0,
        "total_profit": 0.0,
        "roi": 0.0,
    }
    assert strategy["by_fold"]["1"]["roi"] == pytest.approx(0.5)
    assert strategy["by_favorite_underdog"]["underdog"]["total_profit"] == -10.0
    assert set(strategy["by_odds_source"]) == {"line_history", "odds_api"}
    assert set(strategy["by_missingness_bucket"]) == {"none", "over_25pct"}

    predictions.loc[0, "mirror_swapped_prob_a"] = 0.31
    with pytest.raises(ValueError, match="swapped inference parity"):
        run_eval._bounded_selection_diagnostics(
            predictions=predictions,
            selection_index=selection_index,
            feature_contract_columns=["feature_one", "feature_two"],
            sliced_metrics={"by_year": {}},
            sweep=sweep,
        )


def test_actual_swapped_inference_tolerance_is_deterministically_reproducible() -> None:
    from src.model.orientation import swap_directional_frame

    class DirectionalModel:
        def predict_proba(self, values):
            raw = values[:, 0] - values[:, 1]
            probability = 1.0 / (1.0 + np.exp(-raw))
            return np.column_stack([1.0 - probability, probability])

    frame = pd.DataFrame(
        {
            "a_signal": [0.2, 1.1, -0.4],
            "b_signal": [0.8, -0.3, 0.7],
        }
    )
    model_result = {
        "model": DirectionalModel(),
        "feature_cols": ["a_signal", "b_signal"],
        "col_medians": np.array([0.0, 0.0]),
        "impute_strategy": "native_nan",
    }
    original = model_lab._predict_batch_with_model(frame, model_result)
    swapped = model_lab._predict_batch_with_model(
        swap_directional_frame(
            frame,
            columns=model_result["feature_cols"],
        ),
        model_result,
    )
    error = np.abs(original["prob_a"] + swapped["prob_a"] - 1.0)
    assert float(error.max()) <= 1e-12


def test_locked_package_rejects_fullfit_descendant_with_wrong_policy_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation_spec = resolve_named_training_spec("full_live_contract_v6_integrity_205")
    fullfit_spec = resolve_named_training_spec(
        "full_live_contract_v6_integrity_205_fullfit"
    )
    bad_fullfit = replace(
        fullfit_spec,
        name="test_bad_cutoff_fullfit",
        train_cutoff_date="2028-01-01",
    )
    real_resolver = run_eval.resolve_named_training_spec

    def resolve(name: str):
        if name == bad_fullfit.name:
            return bad_fullfit
        return real_resolver(name)

    monkeypatch.setattr(run_eval, "resolve_named_training_spec", resolve)
    package = {
        "model_variant": "baseline",
        "dataset_variant": evaluation_spec.dataset_variant,
        "feature_family": "production",
        "calibration_method": evaluation_spec.calibration_method,
        "retrain_months": 4,
        "model_spec_name": evaluation_spec.name,
        "model_spec_payload_sha256": run_eval._canonical_json_sha256(
            asdict(evaluation_spec)
        ),
        "fullfit_model_spec_name": bad_fullfit.name,
        "fullfit_model_spec_payload_sha256": run_eval._canonical_json_sha256(
            asdict(bad_fullfit)
        ),
        "strategy_config": _strategy_config(),
        "runtime_invariants": {"near_miss_enabled": False},
    }
    with pytest.raises(ValueError, match="immutable policy cutoff"):
        run_eval._validate_locked_model_package(
            package,
            label="test package",
            expected_spec_name=evaluation_spec.name,
            expected_spec_sha256=run_eval._canonical_json_sha256(
                asdict(evaluation_spec)
            ),
            expected_feature_contract=list(evaluation_spec.feature_cols),
            expected_retrain_months=4,
            expected_fullfit_cutoff_date="2027-01-01",
        )


def test_runtime_inventory_binds_required_scripts_and_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    for relative in run_eval.EVALUATION_SCRIPT_SOURCE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    (repo / "requirements.txt").write_text("numpy==2.4.3\n", encoding="utf-8")
    monkeypatch.setattr(run_eval, "_repo_root", lambda: repo)

    first = run_eval._collect_runtime_code_metadata()
    source_paths = {
        entry["path"] for entry in first["source_inventory"]["sources"]
    }
    assert "src/module.py" in source_paths
    assert set(run_eval.EVALUATION_SCRIPT_SOURCE_FILES).issubset(source_paths)
    assert first["source_inventory"]["schema_version"] == 2
    assert first["environment"]["schema_version"] == 1
    assert set(first["environment"]["packages"]) == {
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
    }

    tampered = repo / run_eval.EVALUATION_SCRIPT_SOURCE_FILES[0]
    tampered.write_text("# changed\n", encoding="utf-8")
    second = run_eval._collect_runtime_code_metadata()
    assert second["source_fingerprint"] != first["source_fingerprint"]


def _write_durable_confirmation_result_fixture(
    repo: Path,
    *,
    verdict: dict,
) -> Path:
    identities = [
        {
            "event_date": "2026-07-01",
            "fighter_name_low": "alpha",
            "fighter_name_high": "beta",
        }
    ]
    identity_sha = run_eval._canonical_json_sha256(identities)
    key_material = {
        "key_contract": "confirmation_sample_consumption_v1",
        "confirmation_evaluation_fight_identity_sha256": identity_sha,
    }
    global_key = run_eval._canonical_json_sha256(key_material)
    claim_dir = repo / run_eval.GLOBAL_CONFIRMATION_CLAIMS_DIR / global_key
    artifact_dir = claim_dir / "artifacts" / "confirmation"
    input_dir = claim_dir / "artifacts" / "inputs"
    artifact_dir.mkdir(parents=True)
    input_dir.mkdir(parents=True)

    fights_path = input_dir / "fights_cleaned.csv"
    features_path = input_dir / "features.csv"
    source_inventory_path = input_dir / run_eval.SOURCE_INVENTORY_FILENAME
    environment_path = input_dir / run_eval.ENVIRONMENT_INVENTORY_FILENAME
    odds_inventory_path = input_dir / "odds_source_inventory.json"
    fights_path.write_text("event_date,fighter_a,fighter_b\n", encoding="utf-8")
    features_path.write_text("event_date,fighter_a,fighter_b,feature\n", encoding="utf-8")
    source_inventory = {"schema_version": 2}
    environment = {"schema_version": 1}
    odds_inventory = {
        "schema_version": 1,
        "evaluation_protocol": {"reserved_confirmation_folds": 2},
        "entries": [],
    }
    run_eval._write_json(source_inventory_path, source_inventory)
    run_eval._write_json(environment_path, environment)
    run_eval._write_json(odds_inventory_path, odds_inventory)

    valid = {character: character * 64 for character in "123456789abcdef"}
    selection_identities = [
        {
            "event_date": "2026-05-01",
            "fighter_name_low": "selection-alpha",
            "fighter_name_high": "selection-beta",
        },
        {
            "event_date": "2026-06-01",
            "fighter_name_low": "selection-delta",
            "fighter_name_high": "selection-gamma",
        },
    ]
    selection = {
        "evaluation_partition": "selection",
        "reserved_confirmation_folds": 2,
        "selection_fold_ids": [1, 2],
        "confirmation_fold_ids": [3, 4],
        "evaluation_sample_sha256": valid["1"],
        "evaluation_fight_identity_sha256": run_eval._canonical_json_sha256(
            selection_identities
        ),
        "selection_fight_identities": selection_identities,
        "n_predictions": 2,
        "n_folds": 2,
        "full_evaluation_sample_sha256": valid["3"],
        "full_evaluation_fight_identity_sha256": valid["4"],
        "full_evaluation_n_fights": 3,
        "full_evaluation_n_folds": 4,
        "confirmation_evaluation_sample_sha256": valid["5"],
        "confirmation_evaluation_fight_identity_sha256": identity_sha,
        "confirmation_fight_identities": identities,
        "confirmation_evaluation_n_fights": 1,
        "confirmation_evaluation_n_folds": 2,
        "evaluation_input_value_sha256": valid["6"],
        "full_evaluation_input_value_sha256": valid["7"],
        "confirmation_evaluation_input_value_sha256": valid["8"],
        "feature_contract_columns": ["feature"],
        "feature_contract_count": 1,
        "feature_contract_sha256": run_eval._canonical_json_sha256(["feature"]),
        "dataset_fights_sha256": run_eval._file_sha256(fights_path),
        "source_dataset_fights_sha256": run_eval._file_sha256(fights_path),
        "features_artifact_sha256": run_eval._file_sha256(features_path),
        "features_value_sha256": valid["9"],
        "source_features_sha256": run_eval._file_sha256(features_path),
        "odds_source_inventory_sha256": run_eval._file_sha256(
            odds_inventory_path
        ),
        "policy_sha256": valid["b"],
        "scheduled_protocol_sha256": valid["c"],
        "evaluation_protocol_sha256": valid["d"],
        "source_fingerprint": run_eval._canonical_json_sha256(source_inventory),
        "source_inventory_sha256": run_eval._canonical_json_sha256(
            source_inventory
        ),
        "source_inventory_artifact_sha256": run_eval._file_sha256(
            source_inventory_path
        ),
        "environment_artifact_sha256": run_eval._file_sha256(environment_path),
        "environment_payload_sha256": run_eval._canonical_json_sha256(
            {"schema_version": 1}
        ),
        "evaluation_end_date": "2026-07-01",
        "fresh_window_cutoff": "2026-01-01",
    }
    exposure_key_material = run_eval._selection_exposure_key_material(
        selection_identities
    )
    exposure_key = run_eval._canonical_json_sha256(exposure_key_material)
    exposure_path = run_eval._selection_exposure_path(selection_identities)
    run_eval._write_json(
        exposure_path,
        {
            "schema_version": 1,
            "protocol": run_eval._CONFIRMATION_PROTOCOL,
            "status": "selection_exposed",
            "selection_exposure_key": exposure_key,
            "selection_exposure_key_material": exposure_key_material,
            "selection_evaluation_fight_identity_sha256": selection[
                "evaluation_fight_identity_sha256"
            ],
            "selection_fight_identities": selection_identities,
            "selection_min_event_date": "2026-05-01",
            "selection_max_event_date": "2026-06-01",
            "recorded_at": "2026-08-09T00:00:00+00:00",
        },
    )
    strategy = {"name": "locked"}
    package = {
        "model_spec_name": "registered_eval",
        "model_spec_payload_sha256": valid["f"],
        "fullfit_model_spec_name": "registered_fullfit",
        "fullfit_model_spec_payload_sha256": valid["1"],
        "calibration_method": "isotonic",
        "strategy_config": strategy,
        "runtime_invariants": {"near_miss_enabled": False},
    }
    package_sha = run_eval._canonical_json_sha256(package)
    evaluation_protocol = {"reserved_confirmation_folds": 2}
    lock = {
        "schema_version": 1,
        "protocol": run_eval._CONFIRMATION_PROTOCOL,
        "package_id": "winner",
        "candidate_package": package,
        "candidate_package_contract_sha256": package_sha,
        "selection_result_sha256": valid["2"],
        "selection_evidence": selection,
        "candidate_odds_source_inventory": odds_inventory,
        "frozen_control": {
            "bootstrap_receipt_sha256": valid["3"],
            "package": dict(package),
        },
        "evaluation_protocol": evaluation_protocol,
        "evaluation_protocol_sha256": run_eval._canonical_json_sha256(
            evaluation_protocol
        ),
        "policy_sha256": selection["policy_sha256"],
        "scheduled_protocol_sha256": selection["scheduled_protocol_sha256"],
        "source_fingerprint": selection["source_fingerprint"],
        "baseline_visibility_limitation": "all-fold baseline was previously visible",
    }
    lock_path = claim_dir / run_eval.CONFIRMATION_LOCK_FILENAME
    run_eval._write_json(lock_path, lock)
    lock_sha = run_eval._file_sha256(lock_path)
    claim_path = claim_dir / run_eval.CONFIRMATION_CLAIM_FILENAME
    global_result_path = claim_dir / run_eval.CONFIRMATION_RESULT_FILENAME
    claim = {
        "schema_version": 1,
        "protocol": run_eval._CONFIRMATION_PROTOCOL,
        "status": "claimed",
        "global_claim_key": global_key,
        "global_claim_key_material": key_material,
        "claim_path": run_eval._repo_relative_artifact_path(claim_path),
        "global_result_path": run_eval._repo_relative_artifact_path(
            global_result_path
        ),
        "lock_path": run_eval._repo_relative_artifact_path(lock_path),
        "lock_sha256": lock_sha,
        "package_id": "winner",
        "source_fingerprint": selection["source_fingerprint"],
        "frozen_control_receipt_sha256": valid["3"],
        "full_evaluation_sample_sha256": selection[
            "full_evaluation_sample_sha256"
        ],
        "selection_evaluation_sample_sha256": selection[
            "evaluation_sample_sha256"
        ],
        "confirmation_evaluation_sample_sha256": selection[
            "confirmation_evaluation_sample_sha256"
        ],
        "confirmation_evaluation_fight_identity_sha256": identity_sha,
        "confirmation_fight_identities": identities,
        "selection_exposure_key": exposure_key,
        "selection_exposure_path": run_eval._repo_relative_artifact_path(
            exposure_path
        ),
        "selection_exposure_sha256": run_eval._file_sha256(exposure_path),
        "confirmation_evaluation_input_value_sha256": selection[
            "confirmation_evaluation_input_value_sha256"
        ],
        "confirmation_evaluation_n_fights": 1,
        "confirmation_fold_ids": [3, 4],
        "claimed_at": "2026-08-09T00:00:00+00:00",
    }
    run_eval._write_json(claim_path, claim)
    claim_sha = run_eval._file_sha256(claim_path)
    reservation_dir = repo / run_eval.GLOBAL_CONFIRMATION_CLAIMS_DIR / "_fight_reservations"
    reservation_dir.mkdir(parents=True)
    identity_record = identities[0]
    identity_record_sha = run_eval._canonical_json_sha256(identity_record)
    run_eval._write_json(
        reservation_dir / f"{identity_record_sha}.json",
        {
            "schema_version": 1,
            "protocol": run_eval._CONFIRMATION_PROTOCOL,
            "global_claim_key": global_key,
            "fight_identity_sha256": identity_record_sha,
            "fight_identity": identity_record,
        },
    )

    prediction_rows_sha = valid["4"]
    prediction_values_sha = valid["5"]
    model_payload = {
        "metrics": {
            "accuracy": 0.6,
            "brier": 0.20,
            "ece": 0.05,
            "log_loss": 0.60,
            "evaluation_input_value_sha256": selection[
                "confirmation_evaluation_input_value_sha256"
            ],
            "prediction_rows_sha256": prediction_rows_sha,
            "prediction_values_sha256": prediction_values_sha,
            "model_spec_name": package["model_spec_name"],
            "model_spec_payload_sha256": package[
                "model_spec_payload_sha256"
            ],
            "calibration_method": package["calibration_method"],
        },
        "sliced_metrics": {"by_year": {}},
        "evaluation_partition": "confirmation",
        "evaluation_sample_sha256": selection[
            "confirmation_evaluation_sample_sha256"
        ],
        "confirmation_fold_ids": [3, 4],
        "n_predictions": 1,
        "n_folds": 2,
        "evaluation_input_value_sha256": selection[
            "confirmation_evaluation_input_value_sha256"
        ],
        "prediction_rows_sha256": prediction_rows_sha,
        "prediction_values_sha256": prediction_values_sha,
        "model_spec_name": package["model_spec_name"],
        "model_spec_payload_sha256": package["model_spec_payload_sha256"],
        "calibration_method": package["calibration_method"],
    }
    candidate_model = {**model_payload, "model_unchanged_declared": True}
    candidate_model["metrics"] = {
        **model_payload["metrics"],
        "model_unchanged_declared": True,
    }
    control_model = dict(model_payload)
    control_model["metrics"] = dict(model_payload["metrics"])
    run_eval._write_json(artifact_dir / "candidate_model_metrics.json", candidate_model)
    run_eval._write_json(artifact_dir / "control_model_metrics.json", control_model)

    sweep_binding = {
        "evaluation_partition": "confirmation",
        "evaluation_sample_sha256": selection[
            "confirmation_evaluation_sample_sha256"
        ],
        "confirmation_fold_ids": [3, 4],
        "confirmation_lock_sha256": lock_sha,
        "confirmation_claim_sha256": claim_sha,
        "promotion_eligible": True,
        "combined": {
            "roi": 0.10,
            "total_profit": 10.0,
            "max_drawdown": 0.05,
            "avg_clv": 0.01,
            "total_bets": 1,
            "total_wagered": 100.0,
            "final_bankroll": 110.0,
        },
    }
    for prefix in ("candidate", "control"):
        run_eval._write_json(artifact_dir / f"{prefix}_result.json", sweep_binding)
        pd.DataFrame(
            {
                "event_date": ["2026-07-01"],
                "profit": [10.0],
                "bet_size": [100.0],
            }
        ).to_csv(artifact_dir / f"{prefix}_bet_log.csv", index=False)
        pd.DataFrame({"combined": [100.0, 110.0]}).to_csv(
            artifact_dir / f"{prefix}_bankroll_history.csv", index=False
        )
    (artifact_dir / "promotion_report.md").write_text("report\n", encoding="utf-8")

    confirmation_names = {
        "candidate_model_metrics.json",
        "control_model_metrics.json",
        "candidate_result.json",
        "candidate_bet_log.csv",
        "candidate_bankroll_history.csv",
        "control_result.json",
        "control_bet_log.csv",
        "control_bankroll_history.csv",
        "promotion_report.md",
    }
    artifact_bindings = {
        name: {
            "path": run_eval._repo_relative_artifact_path(artifact_dir / name),
            "sha256": run_eval._file_sha256(artifact_dir / name),
        }
        for name in confirmation_names
    }
    inventory_entries = []
    for logical_name, path in (
        ("confirmation_claim", claim_path),
        ("confirmation_lock", lock_path),
        *[(f"confirmation/{name}", artifact_dir / name) for name in sorted(confirmation_names)],
        ("inputs/fights_cleaned.csv", fights_path),
        ("inputs/features.csv", features_path),
        ("inputs/odds_source_inventory.json", odds_inventory_path),
        ("inputs/evaluation_source_inventory.json", source_inventory_path),
        ("inputs/evaluation_environment.json", environment_path),
    ):
        inventory_entries.append(
            {
                "logical_name": logical_name,
                "path": run_eval._repo_relative_artifact_path(path),
                "sha256": run_eval._file_sha256(path),
            }
        )
    inventory_path = claim_dir / "confirmation_evidence_inventory.json"
    run_eval._write_json(
        inventory_path,
        {
            "schema_version": 1,
            "protocol": run_eval._CONFIRMATION_PROTOCOL,
            "global_claim_key": global_key,
            "entries": inventory_entries,
        },
    )
    completed_at = "2026-08-09T00:01:00+00:00"
    result = {
        "schema_version": 1,
        "protocol": run_eval._CONFIRMATION_PROTOCOL,
        "status": "complete",
        "confirmation_passed": verdict["verdict"] == "PROMOTE",
        "confirmation_outcome": (
            "pass" if verdict["verdict"] == "PROMOTE" else "fail"
        ),
        "package_id": "winner",
        "lock_path": claim["lock_path"],
        "lock_sha256": lock_sha,
        "global_claim_key": global_key,
        "claim_path": claim["claim_path"],
        "claim_sha256": claim_sha,
        "claim": claim,
        "anchored_origin_tip_sha256": "a" * 40,
        "evaluation_partition": "confirmation",
        "evaluation_sample_sha256": selection[
            "confirmation_evaluation_sample_sha256"
        ],
        "confirmation_evaluation_fight_identity_sha256": identity_sha,
        "confirmation_fold_ids": [3, 4],
        "candidate_control_index_identical": True,
        "selection_evidence": selection,
        "candidate_package_contract_sha256": package_sha,
        "candidate_package": package,
        "candidate_model_spec_name": package["model_spec_name"],
        "candidate_model_spec_payload_sha256": package[
            "model_spec_payload_sha256"
        ],
        "candidate_fullfit_model_spec_name": package["fullfit_model_spec_name"],
        "candidate_fullfit_model_spec_payload_sha256": package[
            "fullfit_model_spec_payload_sha256"
        ],
        "candidate_strategy_config_sha256": run_eval._canonical_json_sha256(strategy),
        "candidate_calibration_method": package["calibration_method"],
        "candidate_runtime_invariants": package["runtime_invariants"],
        "candidate_model_unchanged_declared": True,
        "candidate_prediction_rows_sha256": prediction_rows_sha,
        "candidate_prediction_values_sha256": prediction_values_sha,
        "control_model_spec_name": package["model_spec_name"],
        "control_model_spec_payload_sha256": package["model_spec_payload_sha256"],
        "control_calibration_method": package["calibration_method"],
        "control_prediction_rows_sha256": prediction_rows_sha,
        "control_prediction_values_sha256": prediction_values_sha,
        "candidate_selection_result_sha256": lock["selection_result_sha256"],
        "feature_contract_count": 1,
        "feature_contract_sha256": selection["feature_contract_sha256"],
        "dataset_fights_path": run_eval._repo_relative_artifact_path(fights_path),
        "dataset_fights_sha256": selection["dataset_fights_sha256"],
        "source_dataset_fights_path": run_eval._repo_relative_artifact_path(
            fights_path
        ),
        "source_dataset_fights_sha256": selection[
            "source_dataset_fights_sha256"
        ],
        "features_artifact_path": run_eval._repo_relative_artifact_path(features_path),
        "features_artifact_sha256": selection["features_artifact_sha256"],
        "features_value_sha256": selection["features_value_sha256"],
        "source_features_path": run_eval._repo_relative_artifact_path(features_path),
        "source_features_sha256": selection["source_features_sha256"],
        "confirmation_evaluation_input_value_sha256": selection[
            "confirmation_evaluation_input_value_sha256"
        ],
        "odds_source_inventory_path": run_eval._repo_relative_artifact_path(
            odds_inventory_path
        ),
        "odds_source_inventory_sha256": selection[
            "odds_source_inventory_sha256"
        ],
        "policy_sha256": lock["policy_sha256"],
        "scheduled_protocol_sha256": lock["scheduled_protocol_sha256"],
        "evaluation_protocol": evaluation_protocol,
        "evaluation_protocol_sha256": lock["evaluation_protocol_sha256"],
        "frozen_control_receipt_sha256": valid["3"],
        "source_fingerprint": selection["source_fingerprint"],
        "source_inventory_path": run_eval._repo_relative_artifact_path(
            source_inventory_path
        ),
        "source_inventory_sha256": selection["source_inventory_sha256"],
        "source_inventory_artifact_sha256": selection[
            "source_inventory_artifact_sha256"
        ],
        "environment_path": run_eval._repo_relative_artifact_path(
            environment_path
        ),
        "environment_artifact_sha256": selection[
            "environment_artifact_sha256"
        ],
        "environment_payload_sha256": selection["environment_payload_sha256"],
        "prior_all_fold_baseline_visibility": True,
        "baseline_visibility_limitation": lock["baseline_visibility_limitation"],
        "verdict": verdict,
        "evidence_inventory_path": run_eval._repo_relative_artifact_path(
            inventory_path
        ),
        "evidence_inventory_sha256": run_eval._file_sha256(inventory_path),
        "artifacts": artifact_bindings,
        "completed_at": completed_at,
    }
    run_eval._write_json(global_result_path, result)
    completion_path = claim_dir / run_eval.CONFIRMATION_COMPLETION_FILENAME
    run_eval._write_json(
        completion_path,
        {
            "schema_version": 1,
            "protocol": run_eval._CONFIRMATION_PROTOCOL,
            "global_claim_key": global_key,
            "claim_sha256": claim_sha,
            "lock_sha256": lock_sha,
            "result_path": run_eval._repo_relative_artifact_path(global_result_path),
            "result_sha256": run_eval._file_sha256(global_result_path),
            "verdict_sha256": run_eval._canonical_json_sha256(verdict),
            "completed_at": completed_at,
        },
    )
    return global_result_path


def test_confirmation_validator_rejects_unanchored_fabricated_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(run_eval, "_repo_root", lambda: repo)
    monkeypatch.setattr(run_eval, "GLOBAL_CONFIRMATION_CLAIMS_DIR", Path("c"))
    result_path = _write_durable_confirmation_result_fixture(
        repo,
        verdict={
            "verdict": "PROMOTE",
            "model_bar": {"passes": True, "reasons": []},
            "trading_bar": {"passes": True, "reasons": []},
            "reasoning": ["Verdict: PROMOTE"],
        },
    )

    with pytest.raises(ValueError, match="git checkout|committed and pushed"):
        run_eval.validate_confirmation_result_for_promotion(result_path)


def test_confirmation_result_validator_recomputes_pass_and_rejects_forged_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(run_eval, "_repo_root", lambda: repo)
    monkeypatch.setattr(run_eval, "GLOBAL_CONFIRMATION_CLAIMS_DIR", Path("c"))
    monkeypatch.setattr(
        run_eval,
        "require_remotely_anchored_git_artifacts",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        run_eval,
        "require_recorded_anchor_in_history",
        lambda *_args, **_kwargs: None,
    )
    promote = {
        "verdict": "PROMOTE",
        "model_bar": {"passes": True, "reasons": []},
        "trading_bar": {"passes": True, "reasons": []},
        "reasoning": ["Verdict: PROMOTE"],
    }
    result_path = _write_durable_confirmation_result_fixture(
        repo,
        verdict=promote,
    )
    durable_result = json.loads(result_path.read_text(encoding="utf-8"))
    source_inventory = json.loads(
        (repo / durable_result["source_inventory_path"]).read_text(encoding="utf-8")
    )
    environment = json.loads(
        (repo / durable_result["environment_path"]).read_text(encoding="utf-8")
    )
    odds_inventory = json.loads(
        (repo / durable_result["odds_source_inventory_path"]).read_text(
            encoding="utf-8"
        )
    )
    monkeypatch.setattr(
        run_eval,
        "_collect_runtime_code_metadata",
        lambda: {
            "source_fingerprint": durable_result["source_fingerprint"],
            "source_inventory_sha256": durable_result[
                "source_inventory_sha256"
            ],
            "source_inventory": source_inventory,
            "environment_payload_sha256": durable_result[
                "environment_payload_sha256"
            ],
            "environment": environment,
        },
    )
    monkeypatch.setattr(
        run_eval,
        "_historical_odds_inventory_payload",
        lambda _protocol: odds_inventory,
    )
    monkeypatch.setattr(run_eval, "evaluate_for_promotion", lambda *_args: promote)
    binding = run_eval.validate_confirmation_result_for_promotion(result_path)
    assert binding["package_id"] == "winner"
    assert binding["dataset_fights_path"].endswith("fights_cleaned.csv")

    monkeypatch.setattr(
        run_eval,
        "_historical_odds_inventory_payload",
        lambda _protocol: {**odds_inventory, "entries": [{"changed": True}]},
    )
    with pytest.raises(ValueError, match="odds inventory differs"):
        run_eval.validate_confirmation_result_for_promotion(result_path)
    monkeypatch.setattr(
        run_eval,
        "_historical_odds_inventory_payload",
        lambda _protocol: odds_inventory,
    )

    forged_recompute = {
        "verdict": "REJECT",
        "model_bar": {"passes": False, "reasons": ["failed"]},
        "trading_bar": {"passes": False, "reasons": ["failed"]},
        "reasoning": ["Verdict: REJECT"],
    }
    monkeypatch.setattr(
        run_eval,
        "evaluate_for_promotion",
        lambda *_args: forged_recompute,
    )
    with pytest.raises(ValueError, match="does not reproduce durable gate evidence"):
        run_eval.validate_confirmation_result_for_promotion(result_path)


def test_bounded_executor_accepts_reduced_control_metrics_and_rejects_changed_identity_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_spec = resolve_named_training_spec("full_live_contract_v6_integrity_205")
    fullfit_spec = resolve_named_training_spec(
        "full_live_contract_v6_integrity_205_fullfit"
    )
    eval_sha = run_eval._canonical_json_sha256(asdict(eval_spec))
    fullfit_sha = run_eval._canonical_json_sha256(asdict(fullfit_spec))
    protocol = {
        "bet_start_date": "2022-01-01",
        "execution_mode": "realistic",
        "initial_bankroll": 100.0,
    }
    strategy = {
        "name": "bounded_controls",
        "s_min_edge": 0.03,
        "s_blend_weight": 0.30,
        "s_model_agreement_min_edge": 0.01,
        "c_min_model_prob": 0.70,
        "c_min_no_odds_prob": 0.55,
        "c_share": 0.50,
        "c_bet_fraction": 0.05,
        "c_max_bet_fraction": 0.08,
        "c_min_edge": 0.01,
        "c_max_decimal_odds": 3.0,
        "m_enabled": False,
        "m_min_model_prob": 0.65,
        "m_min_no_odds_prob": 0.50,
        "m_bet_fraction": 0.03,
        "m_share": 0.30,
    }

    def package(edge: float) -> dict:
        return {
            "model_variant": "baseline",
            "dataset_variant": eval_spec.dataset_variant,
            "feature_family": "production",
            "calibration_method": eval_spec.calibration_method,
            "retrain_months": 6,
            "model_spec_name": eval_spec.name,
            "model_spec_payload_sha256": eval_sha,
            "fullfit_model_spec_name": fullfit_spec.name,
            "fullfit_model_spec_payload_sha256": fullfit_sha,
            "strategy_config": {**strategy, "s_min_edge": edge},
            "runtime_invariants": {"near_miss_enabled": False},
        }

    declared = [("lower_roi", package(0.04)), ("winner", package(0.05))]
    manifest = {
        "schema_version": 1,
        "protocol": run_eval._CONFIRMATION_PROTOCOL,
        "created_at": "2026-08-09T00:00:00+00:00",
        "reserved_confirmation_folds": 2,
        "ranking_policy": run_eval.BOUNDED_SELECTION_RANKING_POLICY,
        "source_fingerprint": "1" * 64,
        "policy_sha256": "2" * 64,
        "scheduled_protocol_sha256": "3" * 64,
        "evaluation_protocol_sha256": run_eval._canonical_json_sha256(protocol),
        "packages": [
            {
                "package_id": package_id,
                "package": value,
                "package_contract_sha256": run_eval._canonical_json_sha256(value),
                "hypothesis": f"Predeclared hypothesis for {package_id}",
                "rationale": "One justified strategy-only threshold change.",
                "changed_fields": ["strategy_config.s_min_edge"],
            }
            for package_id, value in declared
        ],
    }
    manifest_path = tmp_path / "experiment.json"
    run_eval._write_json(manifest_path, manifest)
    policy_path = tmp_path / "policy.json"
    run_eval._write_json(
        policy_path,
        {
            "contract": {
                "evaluation_spec_name": eval_spec.name,
                "evaluation_spec_payload_sha256": eval_sha,
                "fullfit_spec_name": fullfit_spec.name,
                "fullfit_spec_payload_sha256": fullfit_sha,
                "exclusive_train_cutoff_date": fullfit_spec.train_cutoff_date,
            },
            "evaluation": {
                "bet_start_date": "2022-01-01",
                "execution_mode": "realistic",
                "entry_offset_days": 1.0,
                "entry_offset_for_features": True,
                "require_entry_odds": True,
                "retrain_months": 6,
                "initial_train_years": 5,
            },
        },
    )
    fights_path = tmp_path / "fights.csv"
    features_path = tmp_path / "features.csv"
    fights_path.write_text("event_date,fighter_a,fighter_b,target\n", encoding="utf-8")
    features_path.write_text("event_date,fighter_a,fighter_b,target\n", encoding="utf-8")
    runtime = {
        "source_fingerprint": "1" * 64,
        "source_inventory_sha256": "1" * 64,
        "source_inventory": {"schema_version": 2},
        "environment": {"schema_version": 1},
        "environment_payload_sha256": "4" * 64,
    }
    monkeypatch.setattr(run_eval, "_collect_runtime_code_metadata", lambda: runtime)
    monkeypatch.setattr(
        run_eval,
        "_load_policy_provenance",
        lambda: {
            "policy_path": str(policy_path),
            "policy_sha256": "2" * 64,
            "scheduled_protocol_sha256": "3" * 64,
            "corrected_fights_path": str(fights_path),
            "corrected_features_path": str(features_path),
        },
    )
    monkeypatch.setattr(run_eval, "_evaluation_protocol_payload", lambda **_kwargs: protocol)
    monkeypatch.setattr(
        run_eval,
        "validate_frozen_control_arm_for_selection_gate",
        lambda _freeze_id: {"ready": True, "errors": []},
    )
    control_metrics = {
        "model_variant": "baseline",
        "dataset_variant": eval_spec.dataset_variant,
        "feature_family": "production",
        "calibration_method": eval_spec.calibration_method,
        "retrain_months": 6,
        "model_spec_name": eval_spec.name,
        "model_spec_payload_sha256": eval_sha,
    }
    monkeypatch.setattr(
        run_eval,
        "_load_control_arm_payloads",
        lambda _freeze_id: (control_metrics, {"config": strategy}),
    )
    monkeypatch.setattr(
        run_eval,
        "_validate_locked_model_package",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        run_eval,
        "_package_changed_fields",
        lambda *_args: ["strategy_config.s_min_edge"],
    )
    monkeypatch.setattr(run_eval, "_load_cached_frame", lambda _path: pd.DataFrame())
    monkeypatch.setattr(
        model_lab,
        "_materialize_variant_contract_features",
        lambda frame, _variant: frame,
    )
    monkeypatch.setattr(
        run_eval,
        "_validate_registered_challenger_spec",
        lambda **_kwargs: eval_spec,
    )
    monkeypatch.setattr(
        run_eval,
        "variant_from_named_training_spec",
        lambda *_args, **_kwargs: model_variants.VariantConfig(
            name="baseline", description="bounded test"
        ),
    )
    monkeypatch.setattr(
        run_eval,
        "resolve_variant_feature_columns",
        lambda *_args, **_kwargs: (list(eval_spec.feature_cols), []),
    )

    def fold(fold_id: int, date: str) -> tuple[int, pd.DataFrame]:
        return fold_id, pd.DataFrame(
            {
                "event_date": pd.to_datetime([date]),
                "fighter_a": [f"A{fold_id}"],
                "fighter_b": [f"B{fold_id}"],
                "target": [1],
                "fold": [fold_id],
                "train_end": pd.to_datetime([date]),
                "test_end": pd.to_datetime([date]),
                "prob_a": [0.60],
                "prob_b": [0.40],
                "no_odds_prob_a": [0.55],
                "no_odds_prob_b": [0.45],
                "mirror_swapped_prob_a": [0.40],
                "mirror_swapped_no_odds_prob_a": [0.45],
            }
        )

    selection_folds = [fold(1, "2024-01-01")]
    fold_manifest = selection_folds + [fold(2, "2025-01-01"), fold(3, "2026-01-01")]
    selection_index = run_eval._post_cutoff_predictions(selection_folds)
    full_index = run_eval._post_cutoff_predictions(fold_manifest)
    confirmation_index = run_eval._post_cutoff_predictions(fold_manifest[-2:])
    selection_identities = run_eval._canonical_evaluation_fight_identities(
        selection_index
    )
    confirmation_identities = run_eval._canonical_evaluation_fight_identities(
        confirmation_index
    )
    feature_contract_columns = list(eval_spec.feature_cols)
    generation_calls = []

    def generate(*_args, **kwargs):
        generation_calls.append(kwargs)
        kwargs["pre_fit_manifest_callback"](fold_manifest)
        return selection_folds, fold_manifest

    monkeypatch.setattr(run_eval, "generate_variant_fold_predictions", generate)
    partition = {
        "evaluation_partition": "selection",
        "reserved_confirmation_folds": 2,
        "selection_fold_ids": [1],
        "confirmation_fold_ids": [2, 3],
        "evaluation_sample_sha256": run_eval._evaluation_sample_sha256(selection_index),
        "evaluation_fight_identity_sha256": run_eval._canonical_json_sha256(
            selection_identities
        ),
        "selection_fight_identities": selection_identities,
        "full_evaluation_sample_sha256": run_eval._evaluation_sample_sha256(full_index),
        "full_evaluation_fight_identity_sha256": (
            run_eval._evaluation_fight_identity_sha256(full_index)
        ),
        "confirmation_evaluation_sample_sha256": (
            run_eval._evaluation_sample_sha256(confirmation_index)
        ),
        "confirmation_evaluation_fight_identity_sha256": (
            run_eval._canonical_json_sha256(confirmation_identities)
        ),
        "confirmation_fight_identities": confirmation_identities,
        "evaluation_input_value_sha256": "8" * 64,
        "full_evaluation_input_value_sha256": "9" * 64,
        "confirmation_evaluation_input_value_sha256": "a" * 64,
        "n_predictions": 1,
        "n_folds": 1,
        "full_evaluation_n_fights": 3,
        "full_evaluation_n_folds": 3,
        "confirmation_evaluation_n_fights": 2,
        "confirmation_evaluation_n_folds": 2,
        "feature_contract_columns": feature_contract_columns,
        "feature_contract_count": len(feature_contract_columns),
        "feature_contract_sha256": run_eval._canonical_json_sha256(
            feature_contract_columns
        ),
        "evaluation_end_date": "2026-01-01",
        "fresh_window_cutoff": "2023-07-01",
    }
    input_provenance = {
        **partition,
        "dataset_fights_sha256": "b" * 64,
        "source_dataset_fights_sha256": "c" * 64,
        "features_artifact_sha256": "d" * 64,
        "features_value_sha256": "e" * 64,
        "source_features_sha256": "f" * 64,
        "odds_source_inventory_sha256": "0" * 64,
        "policy_sha256": "2" * 64,
        "scheduled_protocol_sha256": "3" * 64,
        "evaluation_protocol_sha256": run_eval._canonical_json_sha256(protocol),
        "source_fingerprint": "1" * 64,
        "source_inventory_sha256": "1" * 64,
        "source_inventory_artifact_sha256": "2" * 64,
        "environment_artifact_sha256": "3" * 64,
        "environment_payload_sha256": "4" * 64,
        "input_provenance_payload_sha256": "5" * 64,
    }
    raw_identity_fields = {
        "selection_fight_identities",
        "confirmation_fight_identities",
    }
    control_metrics.update(
        {
            field: value
            for field, value in run_eval._selection_binding_from_payload(
                input_provenance
            ).items()
            if field not in raw_identity_fields
        }
    )
    control_metrics.update(
        {
            "prediction_rows_sha256": "6" * 64,
            "prediction_values_sha256": "7" * 64,
        }
    )
    monkeypatch.setattr(
        run_eval, "_fold_partition_evidence", lambda *_args, **_kwargs: dict(partition)
    )
    monkeypatch.setattr(
        run_eval,
        "_build_input_provenance_payload",
        lambda **_kwargs: dict(input_provenance),
    )
    monkeypatch.setattr(
        run_eval,
        "_compute_model_metrics",
        lambda *_args, **_kwargs: {
            "accuracy": 0.60,
            "brier": 0.20,
            "ece": 0.04,
            "log_loss": 0.60,
        },
    )
    monkeypatch.setattr(
        run_eval,
        "_compute_sliced_metrics",
        lambda *_args, **_kwargs: {"by_year": {"2024": {"brier": 0.20}}},
    )

    class Gate:
        def __init__(self, _control):
            pass

        def evaluate(self, _candidate):
            return {"passed": True, "composite_score": 0.10}

    monkeypatch.setattr(run_eval, "SelectionGate", Gate)

    def evaluate(_folds, config, **_kwargs):
        roi = 0.20 if abs(config.s_min_edge - 0.05) < 1e-12 else 0.10
        return {
            "combined": {"roi": roi, "max_drawdown": 0.05},
            "bet_log": pd.DataFrame(
                [{"event_date": "2024-01-01", "bet_size": 10.0, "profit": roi}]
            ),
            "bankroll_history": pd.DataFrame({"combined": [100.0, 100.0 + roi]}),
        }

    monkeypatch.setattr(run_eval, "_evaluate_config", evaluate)
    monkeypatch.setattr(
        run_eval,
        "_bounded_selection_diagnostics",
        lambda **_kwargs: {
            "schema_version": 1,
            "protocol": run_eval._CONFIRMATION_PROTOCOL,
            "evaluation_partition": "selection",
            "evaluation_sample_sha256": partition["evaluation_sample_sha256"],
            "mirror_symmetry": {
                "check": "actual_ab_swapped_feature_inference",
                "passed": True,
                "inference_receipt_sha256": "c" * 64,
                "max_abs_model_probability_complement_error": 0.0,
                "max_abs_no_odds_probability_complement_error": 0.0,
            },
            "strategy_performance": {
                "overall": {"n_bets": 1},
                "by_fold": {"1": {"n_bets": 1}},
                "by_event_month": {"2024-01": {"n_bets": 1}},
                "by_trader": {"S": {"n_bets": 1}},
                "by_side": {"a": {"n_bets": 1}},
                "by_favorite_underdog": {"favorite": {"n_bets": 1}},
                "by_edge_bucket": {"5_10pct": {"n_bets": 1}},
                "by_odds_source": {"test": {"n_bets": 1}},
                "by_missingness_bucket": {"none": {"n_bets": 1}},
                "joined_bet_rows_sha256": "d" * 64,
            },
        },
    )
    run_dir = tmp_path / "run"
    winner_path, winner = run_eval._run_bounded_selection_experiment(
        run_dir=run_dir,
        experiment_manifest_path=manifest_path,
        freeze_id="integrity_v2_test",
    )

    assert len(generation_calls) == 1
    assert generation_calls[0]["evaluation_partition"] == "selection"
    assert generation_calls[0]["confirmation_fold_count"] == 2
    assert winner["package_id"] == "winner"
    assert winner["ranking"] == {
        "selected": True,
        "rank": 1,
        "n_packages": 2,
        "n_eligible": 2,
    }
    assert winner["selection_gate"]["route"] == "registered_challenger_strict_improvement"
    assert (
        winner["candidate_metrics"]["model"]["prediction_rows_sha256"]
        != control_metrics["prediction_rows_sha256"]
    )
    assert (
        winner["candidate_metrics"]["model"]["prediction_values_sha256"]
        != control_metrics["prediction_values_sha256"]
    )
    assert "prediction_rows_sha256" not in winner["selection_evidence"]
    assert "prediction_values_sha256" not in winner["selection_evidence"]
    assert selection_identities
    assert confirmation_identities
    assert len(selection_identities) == partition["n_predictions"]
    assert len(confirmation_identities) == partition[
        "confirmation_evaluation_n_fights"
    ]
    assert run_eval._canonical_json_sha256(selection_identities) == partition[
        "evaluation_fight_identity_sha256"
    ]
    assert run_eval._canonical_json_sha256(confirmation_identities) == partition[
        "confirmation_evaluation_fight_identity_sha256"
    ]
    assert raw_identity_fields.isdisjoint(control_metrics)
    assert winner["selection_evidence"]["selection_fight_identities"] == (
        selection_identities
    )
    assert winner["selection_evidence"]["confirmation_fight_identities"] == (
        confirmation_identities
    )
    assert winner_path.name == "winner_selection_result.json"
    summary = json.loads(
        (run_dir / run_eval.BOUNDED_SELECTION_SUMMARY_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert summary["winner_package_id"] == "winner"
    assert sorted(row["rank"] for row in summary["results"]) == [1, 2]
    run_eval._validate_bounded_selection_summary(
        summary_path=run_dir / run_eval.BOUNDED_SELECTION_SUMMARY_FILENAME,
        summary=summary,
        experiment_path=manifest_path,
        experiment_manifest=manifest,
        winner_path=winner_path,
        winner_result=winner,
    )

    changed_identity_sha256 = "8" * 64
    assert changed_identity_sha256 != partition[
        "confirmation_evaluation_fight_identity_sha256"
    ]
    control_metrics["confirmation_evaluation_fight_identity_sha256"] = (
        changed_identity_sha256
    )
    with pytest.raises(
        ValueError,
        match=r"bounded package lower_roi/control immutable selection input differs",
    ):
        run_eval._run_bounded_selection_experiment(
            run_dir=tmp_path / "changed_identity_hash_run",
            experiment_manifest_path=manifest_path,
            freeze_id="integrity_v2_test",
        )
