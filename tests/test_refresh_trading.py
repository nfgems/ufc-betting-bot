"""Offline integration proof for the final-pair trading inference consumer."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.data.refresh_predictions import MODEL_CONTRACT
from src.data.refresh_service import RefreshService
from src.data.refresh_trading import PublishedTradingSnapshot, RefreshTradingError
from src.features.build_features import filter_feature_columns_for_no_odds
from src.model.training_spec import full_live_contract_v6_durability_2022_spec


NOW = datetime(2026, 10, 15, 18, tzinfo=timezone.utc)
IDS = ("1111111111111111", "2222222222222222")


def pin(path):
    body = Path(path).read_bytes()
    return {"path": str(path), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


class OrderedClassifier:
    """A deterministic model that asserts native missingness and exact order."""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, *_args, **_kwargs):
        raise AssertionError("Trading inference must not fit models")

    def predict_proba(self, matrix):
        assert matrix.shape[1] == len(self.columns)
        assert np.isnan(matrix[:, self.columns.index("a_weight")]).all()
        a_count = matrix[:, self.columns.index("a_num_fights")]
        b_count = matrix[:, self.columns.index("b_num_fights")]
        assert set(a_count.tolist()).issubset({4.0, 7.0})
        assert set(b_count.tolist()).issubset({4.0, 7.0})
        signal = matrix[:, self.columns.index("diff_roll_slpm")]
        if "diff_implied_prob" in self.columns:
            signal = signal + np.nan_to_num(matrix[:, self.columns.index("diff_implied_prob")])
            for name in ("a_ko_odds_prob", "a_sub_odds_prob", "a_dec_odds_prob",
                         "b_ko_odds_prob", "b_sub_odds_prob", "b_dec_odds_prob"):
                assert np.isnan(matrix[:, self.columns.index(name)]).all()
        probabilities = 0.5 + 0.2 * np.tanh(signal)
        return np.column_stack((1 - probabilities, probabilities))


class Producer:
    """Use the real service transactions with a bounded offline producer."""
    def __init__(self, pointer):
        self.pointer = pointer
        self.calls = []
        self.reads = 0

    def load_current_snapshot(self, **_kwargs):
        self.reads += 1
        return deepcopy(self.pointer)

    def inspect_active_refresh(self, _root):
        return None

    def run_incremental_refresh(self, **kwargs):
        self.calls.append(kwargs)
        processed = Path(kwargs["state_dir"]) / "generations" / kwargs["job_id"] / "processed"
        processed.mkdir(parents=True)
        artifacts = {}
        for name in ("fights_cleaned.csv", "features.csv"):
            path = processed / name
            path.write_bytes(b"accepted prefix\nobserved successor\n")
            artifacts[name] = {**pin(path), "rows": 2}
        self.pointer = {**self.pointer, "generation_id": kwargs["job_id"],
                        "processed_dir": str(processed), "artifacts": artifacts, "total_rows": 2}
        return {**self.pointer, "job_id": kwargs["job_id"], "status": "completed",
                "new_rows": 1, "finished_at": kwargs["decision_at"].isoformat()}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from src.data import refresh_service
    original_clock = refresh_service._now
    monkeypatch.setattr(refresh_service, "_now", lambda value=None: original_clock(NOW if value is None else value))
    baseline = tmp_path / "release"
    baseline.mkdir()
    selection = baseline / "selected-source-manifest.json"
    selection.write_text("{}\n", encoding="utf-8")
    spec = json.loads(full_live_contract_v6_durability_2022_spec().to_json())
    columns = spec["feature_cols"]
    assert len(columns) == 211
    models = {}
    for role in ("primary", "no_odds"):
        cols = columns if role == "primary" else filter_feature_columns_for_no_odds(columns)
        artifact = {
            "model": OrderedClassifier(cols), "feature_cols": cols,
            "training_spec": {**spec, "name": MODEL_CONTRACT + ("_no_odds" if role == "no_odds" else ""),
                              "feature_cols": cols},
            "impute_strategy": "native_nan", "col_medians": np.full(len(cols), 123456.0),
        }
        path = baseline / (role + ".pkl")
        joblib.dump(artifact, path)
        models[role] = pin(path)
    comparator = baseline / "logistic.pkl"
    comparator.write_bytes(b"preserved comparator")
    models["logistic"] = pin(comparator)
    manifest_path = baseline / "selected-runtime-manifest.json"
    manifest = {
        "release_id": "accepted-v9", "schema_version": 1,
        "models": {role: {**record, "path": Path(record["path"]).name} for role, record in models.items()},
        "selection": {**pin(selection), "path": selection.name},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    processed = baseline / "processed"
    processed.mkdir()
    artifacts = {}
    for name in ("features.csv", "fights_cleaned.csv"):
        path = processed / name
        path.write_bytes(b"accepted prefix\n")
        artifacts[name] = {**pin(path), "rows": 1}
    producer = Producer({"generation_id": "baseline", "processed_dir": str(processed),
                         "artifacts": artifacts, "models": models, "total_rows": 1})
    predictions = []

    def predict(**kwargs):
        predictions.append(kwargs)
        pointer = kwargs["accepted_pointer"]
        features = {column: None for column in columns}
        features.update(a_num_fights=7.0, b_num_fights=4.0, diff_num_fights=3.0,
                        diff_roll_slpm=0.25 if pointer["generation_id"] == "baseline" else 0.75,
                        a_roll_slpm=3.0, b_roll_slpm=2.75, h2h_record_diff=1.0,
                        a_implied_prob=0.9, b_implied_prob=0.1, diff_implied_prob=0.8,
                        a_ko_odds_prob=0.8, b_sub_odds_prob=0.1)
        provenance = {"processed_dir": pointer["processed_dir"]}
        for side, identity in zip(("a", "b"), IDS):
            provenance.update({f"fighter_{side}_source": "selected_saved_inputs",
                               f"fighter_{side}_history_scope": "selected_processed",
                               f"fighter_{side}_fight_history_status": "selected_history",
                               f"fighter_{side}_id": identity})
        row = {"fighter_a": "Alpha", "fighter_b": "Beta", "fighter_a_id": IDS[0], "fighter_b_id": IDS[1],
               "event_id": "aaaaaaaaaaaaaaaa", "event_date": "2026-10-17", "weight_class": "Lightweight",
               "is_title_bout": False, "num_rounds": 3, "is_empty_arena": None,
               "prob_a": .6, "prob_b": .4, "no_odds_prob_a": .55, "no_odds_prob_b": .45,
               "model_features": features, "feature_provenance": provenance}
        return {"status": "success", "model_contract": MODEL_CONTRACT,
                "accepted_generation": pointer["generation_id"], "decision_at": kwargs["decision_at"].isoformat(),
                "model_pins": pointer["models"], "data_pins": pointer["artifacts"],
                "selected_runtime_manifest": pin(manifest_path), "original_selection": pin(selection),
                "prediction_count": 1, "predictions": [row]}

    def make():
        return RefreshService(root=tmp_path / "workflow", baseline=baseline, producer=producer, predictor=predict)

    service = make()
    service.initialize_predictions(now=NOW - timedelta(days=1))
    return {"service": service, "make": make, "producer": producer,
            "predictor": predict, "predictions": predictions, "baseline": baseline}


def fight(**values):
    return {"fighter_a": "Alpha", "fighter_b": "Beta", "event_id": "bookmaker-event-123",
            "commence_time": "2026-10-18T00:00:00Z",
            "a_fair_prob_avg": .55, "b_fair_prob_avg": .45, **values}


def score(adapter, **values):
    return adapter.score([fight(**values)], observed_at=NOW, decision_at=NOW)[0]


def test_final_consumer_uses_observed_prices_exact_preprocessing_and_native_nan(setup):
    # These prohibited boundaries must never be reached by this consumer.
    setup["producer"].load_current_snapshot = lambda **_: pytest.fail("consumer read mutable producer pointer")
    adapter = PublishedTradingSnapshot(setup["service"]).read()
    result = score(adapter)
    assert result["accepted_generation"] == "baseline"
    assert result["event_id"] == "bookmaker-event-123"
    assert result["official_event_id"] == "aaaaaaaaaaaaaaaa"
    assert result["official_event_date"] == "2026-10-17"
    assert result["model_features"]["a_implied_prob"] == .55
    assert result["model_features"]["b_implied_prob"] == .45
    assert result["model_features"]["diff_implied_prob"] == pytest.approx(.1)
    assert result["model_features"]["a_weight"] is None
    assert result["model_features"]["a_ko_odds_prob"] is None
    assert result["model_features"]["b_sub_odds_prob"] is None
    assert result["prob_a"] == pytest.approx(.5 + .2 * np.tanh(.35))
    assert result["no_odds_prob_a"] == pytest.approx(.5 + .2 * np.tanh(.25))
    assert result["feature_provenance"]["live_odds_observed_at"] == NOW.isoformat()
    assert result["feature_provenance"]["saved_feature_decision_at"] != NOW.isoformat()
    assert not setup["producer"].calls
    from src.data.runtime_selected_inputs import verified_selected_provenance
    assert verified_selected_provenance(result["feature_provenance"], "a")


def test_current_quote_changes_primary_but_not_companion_or_saved_report(setup):
    service = setup["service"]
    original = service.published_path.read_bytes()
    report_path = Path(service.published_pair()[0]["prediction_report"]["path"])
    original_report = report_path.read_bytes()
    adapter = PublishedTradingSnapshot(service).read()
    first = score(adapter)
    second = score(adapter, a_fair_prob_avg=.7, b_fair_prob_avg=.3)
    assert first["prob_a"] != second["prob_a"]
    assert first["no_odds_prob_a"] == second["no_odds_prob_a"]
    assert service.published_path.read_bytes() == original
    assert report_path.read_bytes() == original_report


def test_reversed_current_outcomes_swap_full_features_ids_and_provenance(setup):
    adapter = PublishedTradingSnapshot(setup["service"]).read()
    forward = score(adapter)
    reverse = score(adapter, fighter_a="Beta", fighter_b="Alpha", a_fair_prob_avg=.45, b_fair_prob_avg=.55)
    assert reverse["prob_a"] == pytest.approx(forward["prob_b"])
    assert reverse["no_odds_prob_a"] == pytest.approx(forward["no_odds_prob_b"])
    assert reverse["fighter_a_id"] == IDS[1]
    assert reverse["fighter_b_id"] == IDS[0]
    assert reverse["a_num_fights"] == 4
    assert reverse["model_features"]["a_roll_slpm"] == 2.75
    assert reverse["model_features"]["b_roll_slpm"] == 3.0
    assert reverse["model_features"]["h2h_record_diff"] == -1.0
    assert reverse["model_features"]["diff_num_fights"] == -3.0
    assert reverse["feature_provenance"]["fighter_a_id"] == IDS[1]


def test_missing_current_quote_keeps_native_nan_and_no_odds_companion(setup):
    result = score(PublishedTradingSnapshot(setup["service"]).read(), a_fair_prob_avg=None, b_fair_prob_avg=None)
    assert result["model_features"]["a_implied_prob"] is None
    assert result["model_features"]["diff_implied_prob"] is None
    assert result["a_market_prob"] is None
    assert result["market_inputs_available"] is False
    assert result["prob_a"] == pytest.approx(result["no_odds_prob_a"])


@pytest.mark.parametrize("values", [
    {"fighter_a": "Alphaa"},
    {"commence_time": "2026-10-25T00:00:00Z"},
    {"official_event_date": "2026-10-17", "commence_time": "2026-10-25T00:00:00Z"},
    {"fighter_a_id": "3333333333333333"},
    {"official_event_id": "bbbbbbbbbbbbbbbb"},
    {"a_fair_prob_avg": 0.6, "b_fair_prob_avg": .6},
    {"a_fair_prob_avg": float("nan")},
    {"a_fair_prob_avg": True},
    {"a_fair_prob_avg": 1.0},
])
def test_unbound_match_or_invalid_current_price_is_refused(setup, values):
    with pytest.raises(RefreshTradingError):
        score(PublishedTradingSnapshot(setup["service"]).read(), **values)


@pytest.mark.parametrize("observed,age", [
    (None, None),
    (NOW + timedelta(seconds=1), None),
    (NOW - timedelta(hours=1), 300),
])
def test_observation_timestamp_must_be_real_and_fresh(setup, observed, age):
    adapter = PublishedTradingSnapshot(setup["service"]).read()
    with pytest.raises(RefreshTradingError):
        adapter.score([fight()], observed_at=observed, decision_at=NOW, max_observation_age_seconds=age)


def test_completed_refresh_is_next_cycle_only_for_already_read_consumer(setup):
    service = setup["service"]
    original = PublishedTradingSnapshot(service).read()
    service.refresh_now(job_id="new-completed", now=NOW)
    successor = PublishedTradingSnapshot(service).read()
    assert score(original)["accepted_generation"] == "baseline"
    assert score(successor)["accepted_generation"] == "new-completed"
    assert score(original)["prob_a"] != score(successor)["prob_a"]
    assert len(setup["producer"].calls) == 1


def test_pending_candidate_and_failed_prediction_keep_last_good_pair_on_restart(setup):
    service = setup["service"]
    original_publication = service.published_path.read_bytes()

    def fail(**_kwargs):
        raise RuntimeError("prediction interrupted")

    service._predictor_override = fail
    with pytest.raises(RuntimeError, match="prediction interrupted"):
        service.refresh_now(job_id="pending-data", now=NOW)
    assert setup["producer"].pointer["generation_id"] == "pending-data"
    assert service._state()["pending"]["phase"] == "data_completed"
    assert score(PublishedTradingSnapshot(service).read())["accepted_generation"] == "baseline"
    assert service.published_path.read_bytes() == original_publication
    recovered = setup["make"]()
    assert score(PublishedTradingSnapshot(recovered).read())["accepted_generation"] == "baseline"
    recovered.refresh_now(job_id="pending-data", now=NOW)
    assert score(PublishedTradingSnapshot(recovered).read())["accepted_generation"] == "pending-data"
    assert len(setup["producer"].calls) == 1
    assert setup["predictions"][-1]["decision_at"] == NOW


def test_interruption_after_atomic_publication_reuses_final_pair_without_collection(setup):
    service = setup["service"]
    publish = service._publish_pair

    def interrupt(*args, **kwargs):
        publish(*args, **kwargs)
        raise SystemExit("interrupted after publication")

    service._publish_pair = interrupt
    with pytest.raises(SystemExit, match="interrupted after publication"):
        service.refresh_now(job_id="saved-final", now=NOW)
    published_bytes = service.published_path.read_bytes()
    result = score(PublishedTradingSnapshot(service).read())
    assert result["accepted_generation"] == "saved-final"
    recovered = setup["make"]()
    recovered.refresh_now(job_id="saved-final", now=NOW + timedelta(hours=1))
    assert score(PublishedTradingSnapshot(recovered).read()) == result
    assert recovered.published_path.read_bytes() == published_bytes
    assert len(setup["producer"].calls) == 1
    assert len(setup["predictions"]) == 2
    assert recovered._state()["pending"] is None


@pytest.mark.parametrize("role", ["data", "model"])
def test_drifted_final_data_or_model_is_refused_before_inference(setup, role):
    service = setup["service"]
    pointer = service.published_pair()[1]
    record = pointer["artifacts"]["features.csv"] if role == "data" else pointer["models"]["primary"]
    path = Path(record["path"])
    path.write_bytes(b"drift")
    with pytest.raises(RuntimeError, match="exact saved pin"):
        PublishedTradingSnapshot(service).read()


def test_wrong_saved_feature_contract_is_refused(setup):
    service = setup["service"]
    publication, pointer, report = service.published_pair()
    report["predictions"][0]["model_features"].pop("a_weight")
    service._publish_pair("bad-feature-report", pointer, report)
    with pytest.raises(RefreshTradingError, match="exact accepted contract"):
        PublishedTradingSnapshot(service).read()


def test_namesake_matchup_with_two_official_id_pairs_is_ambiguous(setup):
    service = setup["service"]
    _, pointer, report = service.published_pair()
    namesake = deepcopy(report["predictions"][0])
    namesake["fighter_a_id"] = "3333333333333333"
    report["predictions"].append(namesake)
    report["prediction_count"] = 2
    service._publish_pair("ambiguous-card", pointer, report)
    with pytest.raises(RefreshTradingError, match="ambiguous name/date"):
        PublishedTradingSnapshot(service).read()
