"""Offline checks for pinned prediction, durable recovery, and source identity."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from src.data.incremental_refresh import ResponseCache
from src.data.refresh_predictions import (
    MODEL_CONTRACT, UPCOMING_URL, RefreshPredictionError, parse_event_card,
    parse_fighter_profile, parse_upcoming_events, run_refresh_predictions,
)

IDS = ("1111111111111111", "2222222222222222")
EVENT = "http://ufcstats.com/event-details/aaaaaaaaaaaaaaaa"
DECISION = datetime(2026, 10, 10, 15, 0, tzinfo=timezone.utc)


class NoFitClassifier:
    def fit(self, *_args, **_kwargs):
        raise AssertionError("Prediction must never fit a model")

    def predict_proba(self, matrix):
        # Large saved medians must not replace genuine missing native inputs.
        assert np.isnan(matrix).any()
        return np.tile([0.3, 0.7], (len(matrix), 1))


def pin(path):
    body = path.read_bytes()
    return {"path": str(path), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="")
    return pin(path)


def profile(name):
    return f'''<h2 class="b-content__title"><span>{name}</span></h2>
    <span class="b-content__title-record">Record: 10-2-0</span>
    <ul><li class="b-list__box-list-item">Height: 6' 0"</li>
    <li class="b-list__box-list-item">Reach: 72"</li>
    <li class="b-list__box-list-item">STANCE: Orthodox</li>
    <li class="b-list__box-list-item">DOB: Jan 01, 1990</li></ul>'''


def card(status="Next", title="UFC Test Event", event_date="October 17, 2026"):
    return f'''<h2 class="b-content__title"><span class="b-content__title-highlight">{title}</span></h2>
    <li class="b-list__box-list-item"><i class="b-list__box-item-title">Date:</i>{event_date}</li>
    <li class="b-list__box-list-item">Location: Paris, France</li>
    <table><tr class="b-fight-details__table-row"><th>Header</th></tr>
    <tr class="b-fight-details__table-row"><td>{status}</td><td>
    <a class="b-link" href="http://ufcstats.com/fighter-details/{IDS[0]}">Alpha</a>
    <a class="b-link" href="http://ufcstats.com/fighter-details/{IDS[1]}">Beta</a>
    </td><td>Lightweight</td></tr></table>'''


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path / "release"
    processed = root / "processed"
    pair = "event_date,fighter_a,fighter_b,fighter_a_id,fighter_b_id,winner\n"
    pair += f"2026-09-12,Alpha,Beta,{IDS[0]},{IDS[1]},Alpha\n"
    artifacts = {}
    for name in ("features.csv", "fights_cleaned.csv"):
        artifacts[name] = {**write(processed / name, pair), "rows": 1}
    base_spec_path = Path(__file__).resolve().parents[1] / "release_payload/models/full_live_contract_v6_durability_2022_spec.json"
    spec = json.loads(base_spec_path.read_bytes())
    odds = {"a_implied_prob", "b_implied_prob", "diff_implied_prob", "a_ko_odds_prob",
            "a_sub_odds_prob", "a_dec_odds_prob", "b_ko_odds_prob", "b_sub_odds_prob", "b_dec_odds_prob"}
    models = {}
    for role, name in (("primary", "xgboost"), ("no_odds", "xgboost_no_odds")):
        cols = [column for column in spec["feature_cols"] if role == "primary" or column not in odds]
        embedded = {**spec, "name": MODEL_CONTRACT + ("_no_odds" if role == "no_odds" else ""),
                    "feature_cols": cols}
        artifact = {"model": NoFitClassifier(), "feature_cols": cols, "training_spec": embedded,
                    "impute_strategy": "native_nan", "col_medians": np.full(len(cols), 999.0)}
        path = root / "models" / (name + "_model.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)
        models[role] = pin(path)
    models["logistic"] = write(root / "models/logistic_model.pkl", "unchanged comparator")
    profiles = "name,fighter_url,height,reach,weight,stance,dob,record\n"
    profiles += f'Alpha,http://ufcstats.com/fighter-details/{IDS[0]},180,182,155,Orthodox,Jan 01 1990,10-2-0\n'
    profiles += f'Beta,http://ufcstats.com/fighter-details/{IDS[1]},180,182,155,Orthodox,Jan 01 1990,10-2-0\n'
    files = {"profiles": write(root / "inputs/profiles.csv", profiles),
             "identity_aliases": write(root / "inputs/aliases.json", "[]"),
             "profile_decisions": write(root / "inputs/decisions.csv", "stable_ufcstats_id,field,decision_status,model_safe_value\n"),
             "prior_profile_decisions": write(root / "inputs/prior.csv", "stable_ufcstats_id,field,decision_status,model_safe_value\n")}
    selection = write(root / "selected-source-manifest.json", json.dumps({"files": files}))
    manifest = root / "selected-runtime-manifest.json"
    document = {"schema_version": 1, "release_id": "accepted-v9",
                "models": {role: {**value, "path": str(Path(value["path"]).relative_to(root))}
                           for role, value in models.items()},
                "selection": {**selection, "path": "selected-source-manifest.json"}}
    write(manifest, json.dumps(document))
    snapshot = {"schema": 1, "generation_id": "updated-observed-1", "processed_dir": str(processed),
                "artifacts": artifacts, "models": models}
    responses = {UPCOMING_URL: f'''<table><tr class="b-statistics__table-row"><td>
                 <a class="b-link" href="{EVENT}">UFC Test Event</a>
                 <span class="b-statistics__date">October 17, 2026</span></td></tr></table>''',
                 EVENT: card(),
                 f"http://ufcstats.com/fighter-details/{IDS[0]}": profile("Alpha"),
                 f"http://ufcstats.com/fighter-details/{IDS[1]}": profile("Beta")}
    calls = []

    def fetch(url):
        calls.append(url)
        return responses[url]

    def build(fighter_a, fighter_b, **kwargs):
        from src.data.selected_source_inputs import active_sources
        sources = active_sources()
        assert sources is not None
        assert sources.processed_data_dir == processed
        assert sources.files["profile_decisions"]["sha256"] == files["profile_decisions"]["sha256"]
        assert kwargs["event_date"] == "2026-10-17"
        assert "commence_time" not in kwargs  # Do not fabricate a broadcast time.
        values = {column: np.nan for column in kwargs["training_spec"].feature_cols}
        values.update(a_num_fights=5.0, b_num_fights=4.0)
        return values, {"fighter_a_history_scope": "selected_processed",
                        "fighter_b_history_scope": "selected_processed"}

    monkeypatch.setattr("src.data.fighter_lookup.build_fight_features", build)
    return {"snapshot": snapshot, "manifest": manifest, "state": tmp_path / "prediction-job",
            "responses": responses, "calls": calls, "fetch": fetch,
            "processed": processed, "build": build}


def run(inputs, **kwargs):
    cache = ResponseCache(inputs["state"] / "cache", inputs["fetch"])
    return run_refresh_predictions(accepted_pointer=inputs["snapshot"],
                                   selected_manifest=inputs["manifest"], state_dir=inputs["state"],
                                   decision_at=DECISION, response_cache=cache, **kwargs)


def test_prediction_parent_links_are_synced_before_original_intent(tmp_path, monkeypatch):
    from src.data import refresh_predictions as module
    parents = []
    synced = []
    closed = []
    fake_os = SimpleNamespace(name="posix", O_RDONLY=0, O_DIRECTORY=0,
        open=lambda path, flags: parents.append(Path(path)) or len(parents),
        fsync=lambda descriptor: synced.append(descriptor),
        close=lambda descriptor: closed.append(descriptor))
    monkeypatch.setattr(module, "os", fake_os)
    module._make_directory(tmp_path / "original-job/inputs")
    assert parents == [tmp_path, tmp_path / "original-job"]
    assert synced == closed == [1, 2]
    assert (tmp_path / "original-job/inputs").is_dir()


def test_failed_parent_sync_cannot_save_original_intent(tmp_path, monkeypatch):
    from src.data import refresh_predictions as module
    def fail_sync(_descriptor):
        raise OSError("parent persistence failed")
    fake_os = SimpleNamespace(name="posix", O_RDONLY=0, O_DIRECTORY=0,
        open=lambda path, flags: 1, fsync=fail_sync, close=lambda descriptor: None)
    monkeypatch.setattr(module, "os", fake_os)
    intent = tmp_path / "original-job/prediction-intent.json"
    with pytest.raises(OSError, match="parent persistence"):
        module._save_json(intent, {"original_job": "unchanged"})
    assert not intent.exists()
    assert intent.parent.is_dir()  # Failed partials remain inspectable.


def test_prediction_reuses_original_models_missing_values_and_cached_completion(inputs):
    before = {path: path.read_bytes() for path in inputs["processed"].iterdir()}
    before.update({Path(pin_["path"]): Path(pin_["path"]).read_bytes()
                   for pin_ in inputs["snapshot"]["models"].values()})
    report = run(inputs)
    assert report["prediction_count"] == 1
    assert report["model_training_performed"] is False
    assert report["betting_performed"] is False
    assert report["predictions"][0]["prob_a"] == pytest.approx(0.5)
    assert report["predictions"][0]["model_features"]["a_weight"] is None
    assert report["accepted_generation"] == "updated-observed-1"
    assert len(inputs["calls"]) == 4 and len(set(inputs["calls"])) == 4
    again = run(inputs)
    assert again == report
    assert len(inputs["calls"]) == 4
    assert all(path.read_bytes() == body for path, body in before.items())
    assert all(not Path(item["path"]).is_absolute() for item in report["saved_input_pins"])


def test_interrupted_inference_recovers_original_instant_without_refetch(inputs, monkeypatch):
    def interrupt(*args, **kwargs):
        raise RuntimeError("simulated interruption after durable acquisition")
    monkeypatch.setattr("src.data.fighter_lookup.build_fight_features", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(inputs)
    assert not (inputs["state"] / "predictions.json").exists()
    assert len(inputs["calls"]) == 4
    monkeypatch.setattr("src.data.fighter_lookup.build_fight_features", inputs["build"])
    report = run_refresh_predictions(accepted_pointer=inputs["snapshot"],
              selected_manifest=inputs["manifest"], state_dir=inputs["state"],
              decision_at=datetime(2026, 10, 12, tzinfo=timezone.utc),
              response_cache=ResponseCache(inputs["state"] / "cache", inputs["fetch"]))
    assert report["decision_at"] == DECISION.isoformat()
    assert len(inputs["calls"]) == 4


def test_pin_drift_refuses_before_any_source_request(inputs):
    Path(inputs["snapshot"]["models"]["primary"]["path"]).write_bytes(b"changed model")
    with pytest.raises(ValueError, match="Accepted model changed"):
        run(inputs)
    assert inputs["calls"] == []


def test_invalid_probability_cannot_replace_complete_report(inputs, monkeypatch):
    from src.model import predict
    monkeypatch.setattr(predict, "predict_fight", lambda *args, **kwargs:
                        {"prob_a": float("nan"), "prob_b": float("nan")})
    with pytest.raises(RefreshPredictionError, match="invalid win probabilities"):
        run(inputs)
    assert not (inputs["state"] / "predictions.json").exists()


def test_completed_history_is_not_predicted_again(inputs):
    inputs["responses"][UPCOMING_URL] = inputs["responses"][UPCOMING_URL].replace("October 17, 2026", "September 12, 2026")
    report = run(inputs)
    assert report["prediction_count"] == 0
    assert inputs["calls"] == [UPCOMING_URL]


def test_changed_saved_source_envelope_is_refused_without_refetch(inputs):
    report = run(inputs)
    envelope = next(item for item in report["saved_input_pins"] if item["path"].startswith("cache/"))
    (inputs["state"] / envelope["path"]).write_bytes(b"changed response")
    with pytest.raises(RefreshPredictionError, match="pin mismatch"):
        run(inputs)
    assert len(inputs["calls"]) == 4


def test_source_parsers_require_exact_routes_and_real_metadata():
    with pytest.raises(RefreshPredictionError, match="Untrusted"):
        parse_upcoming_events('''<tr class="b-statistics__table-row"><td><a class="b-link"
          href="https://evil.example/event-details/aaaaaaaaaaaaaaaa">Event</a>
          <span class="b-statistics__date">October 17, 2026</span></td></tr>''')
    parsed = parse_event_card(card(), {"title": "UFC Test Event", "event_date": "2026-10-17", "url": EVENT})
    assert parsed[0]["fighter_a_id"] == IDS[0]
    assert parsed[0]["weight_class"] == "Lightweight"
    assert parsed[0]["num_rounds"] == 5
    fighter = parse_fighter_profile(profile("Alpha"), f"http://ufcstats.com/fighter-details/{IDS[0]}")
    assert fighter["height"] == '6\' 0"'
    assert fighter["td_acc"] == ""  # Unobserved data stays missing.
    assert parse_event_card(card(status="win"),
                           {"title": "UFC Test Event", "event_date": "2026-10-17", "url": EVENT}) == []


@pytest.mark.parametrize("html", [card(title="Wrong Card"), card(event_date="October 24, 2026"),
    card().replace('class="b-content__title-highlight"', 'class="missing-title"'),
    card().replace("Date:", "Missing:")])
def test_event_card_body_must_match_original_selected_event(html):
    with pytest.raises(RefreshPredictionError, match="Card body"):
        parse_event_card(html, {"title": "UFC Test Event", "event_date": "2026-10-17", "url": EVENT})


def test_completed_main_event_does_not_promote_remaining_bout_rounds():
    remaining = card()
    start = remaining.index('<tr class="b-fight-details__table-row"><td>Next')
    end = remaining.index("</tr>", start) + len("</tr>")
    remaining = remaining[start:end].replace(IDS[0], "3333333333333333").replace(IDS[1], "4444444444444444")
    html = card(status="nc nc").replace("</table>", remaining + "</table>")
    fights = parse_event_card(html, {"title": "UFC Test Event", "event_date": "2026-10-17", "url": EVENT})
    assert len(fights) == 1
    assert fights[0]["is_main_event"] is False
    assert fights[0]["num_rounds"] == 3


def test_default_refresh_acquires_only_nearest_announced_card(inputs):
    next_event = "http://ufcstats.com/event-details/bbbbbbbbbbbbbbbb"
    inputs["responses"][UPCOMING_URL] += f'''<table><tr class="b-statistics__table-row"><td>
        <a class="b-link" href="{next_event}">Later Announced Event</a>
        <span class="b-statistics__date">October 24, 2026</span></td></tr></table>'''
    report = run(inputs)
    assert report["prediction_count"] == 1
    assert report["coverage"]["eligible_events"] == 2
    assert report["coverage"]["selected_events"] == 1
    assert report["coverage"]["bounded"] is True
    assert next_event not in inputs["calls"]
    assert len(inputs["calls"]) == 4


def test_wrong_profile_body_cannot_create_an_identity_alias(inputs):
    inputs["responses"][f"http://ufcstats.com/fighter-details/{IDS[0]}"] = profile("Unrelated Person")
    with pytest.raises(RefreshPredictionError, match="profile body name"):
        run(inputs)
    assert not (inputs["state"] / "predictions.json").exists()
    assert not (inputs["state"] / "inputs/observed_identity_aliases.json").exists()
    assert len(inputs["calls"]) == 4
    with pytest.raises(RefreshPredictionError, match="profile body name"):
        run(inputs)
    assert len(inputs["calls"]) == 4  # Original successful requests are never repeated.


def test_interruption_after_first_bout_reuses_original_feature_and_prediction_work(inputs, monkeypatch):
    from src.data import refresh_predictions as module
    from src.model.predict import predict_fight as original_predict
    next_row = card()
    start = next_row.index('<tr class="b-fight-details__table-row"><td>Next')
    end = next_row.index("</tr>", start) + len("</tr>")
    next_row = next_row[start:end].replace(IDS[0], "3333333333333333").replace(IDS[1], "4444444444444444")
    next_row = next_row.replace("Alpha", "Gamma").replace("Beta", "Delta")
    inputs["responses"][EVENT] = card().replace("</table>", next_row + "</table>")
    inputs["responses"]["http://ufcstats.com/fighter-details/3333333333333333"] = profile("Gamma")
    inputs["responses"]["http://ufcstats.com/fighter-details/4444444444444444"] = profile("Delta")
    feature_calls, prediction_calls, interrupted = [], [], []
    original_build, original_save = inputs["build"], module._save_json
    def counted_build(*args, **kwargs):
        feature_calls.append((kwargs["fighter_a"], kwargs["fighter_b"]))
        return original_build(*args, **kwargs)
    def counted_predict(features, *, model_result):
        prediction_calls.append(model_result["training_spec"]["name"])
        return original_predict(features, model_result=model_result)
    def save_then_interrupt(path, document):
        original_save(path, document)
        if path.parent.name == "bouts" and not interrupted:
            interrupted.append(True)
            raise RuntimeError("interrupted after atomic first bout")
    monkeypatch.setattr("src.data.fighter_lookup.build_fight_features", counted_build)
    monkeypatch.setattr("src.model.predict.predict_fight", counted_predict)
    monkeypatch.setattr(module, "_save_json", save_then_interrupt)
    with pytest.raises(RuntimeError, match="atomic first bout"):
        run(inputs)
    assert feature_calls == [("Alpha", "Beta")]
    assert len(prediction_calls) == 2 and len(inputs["calls"]) == 6
    assert not (inputs["state"] / "predictions.json").exists()
    report = run(inputs)
    assert report["prediction_count"] == 2
    assert feature_calls == [("Alpha", "Beta"), ("Gamma", "Delta")]
    assert len(prediction_calls) == 4 and len(inputs["calls"]) == 6
    assert len(report["completed_bouts"]) == 2
    assert report["decision_at"] == DECISION.isoformat()


@pytest.mark.parametrize("field", ["decision_at", "binding", "card", "row"])
def test_completed_bout_binding_or_row_drift_cannot_resume(inputs, monkeypatch, field):
    from src.data import refresh_predictions as module
    original_save = module._save_json
    def interrupt_after_bout(path, document):
        original_save(path, document)
        if path.parent.name == "bouts":
            raise RuntimeError("original result saved")
    monkeypatch.setattr(module, "_save_json", interrupt_after_bout)
    with pytest.raises(RuntimeError, match="original result"):
        run(inputs)
    path = next((inputs["state"] / "bouts").iterdir())
    record = json.loads(path.read_bytes())
    record[field] = "different original evidence"
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr("src.data.fighter_lookup.build_fight_features",
                        lambda *args, **kwargs: pytest.fail("invalid checkpoint cannot recalculate"))
    with pytest.raises(RefreshPredictionError, match="Saved bout"):
        run(inputs)
    assert len(inputs["calls"]) == 4
    assert not (inputs["state"] / "predictions.json").exists()


@pytest.mark.parametrize("root_name", ["method_observations", "rankings_observations", "line_history"])
def test_new_live_observation_cannot_reuse_original_bout(inputs, monkeypatch, root_name):
    from src.data import refresh_predictions as module
    original_save = module._save_json
    def interrupt_after_bout(path, document):
        original_save(path, document)
        if path.parent.name == "bouts":
            raise RuntimeError("original result saved")
    monkeypatch.setattr(module, "_save_json", interrupt_after_bout)
    with pytest.raises(RuntimeError, match="original result"):
        run(inputs)
    plan = json.loads((inputs["state"] / "calculation-intent.json").read_bytes())
    assert set(plan["live_observation_inventory"]) == {"method_observations", "rankings_observations", "line_history"}
    assert all(item["entries"] == [] for item in plan["live_observation_inventory"].values())
    added = inputs["state"] / "observations" / root_name / "new-observation.json"
    added.parent.mkdir(parents=True)
    added.write_text("{}")
    monkeypatch.setattr("src.data.fighter_lookup.build_fight_features",
                        lambda *args, **kwargs: pytest.fail("changed context cannot recalculate"))
    with pytest.raises(RefreshPredictionError, match="observation roots must remain empty"):
        run(inputs)
    assert len(inputs["calls"]) == 4
    assert not (inputs["state"] / "predictions.json").exists()


def test_observation_appearing_during_feature_calculation_cannot_save_bout(inputs, monkeypatch):
    original_build = inputs["build"]
    def changed_context(*args, **kwargs):
        result = original_build(*args, **kwargs)
        added = inputs["state"] / "observations/line_history/odds_changed.csv"
        added.parent.mkdir(parents=True)
        added.write_text("changed current context\n")
        return result
    monkeypatch.setattr("src.data.fighter_lookup.build_fight_features", changed_context)
    monkeypatch.setattr("src.model.predict.predict_fight",
                        lambda *args, **kwargs: pytest.fail("changed context cannot predict"))
    with pytest.raises(RefreshPredictionError, match="observation roots must remain empty"):
        run(inputs)
    assert not (inputs["state"] / "bouts").exists()
    assert not (inputs["state"] / "predictions.json").exists()
    assert len(inputs["calls"]) == 4


def test_relocated_job_reuses_probabilities_without_source_requests(inputs, tmp_path):
    first = run(inputs)
    moved = tmp_path / "moved-job"
    shutil.copytree(inputs["state"], moved)
    cache = ResponseCache(moved / "cache", lambda url: pytest.fail("relocation must not refetch"))
    report = run_refresh_predictions(accepted_pointer=inputs["snapshot"],
              selected_manifest=inputs["manifest"], state_dir=moved,
              decision_at=datetime(2026, 11, 1, tzinfo=timezone.utc), response_cache=cache)
    assert report["decision_at"] == first["decision_at"]
    assert report["predictions"] == first["predictions"]


def test_release_and_job_relocation_rehydrates_selected_manifest(inputs, tmp_path):
    root = inputs["manifest"].parent
    selection = root / "selected-source-manifest.json"
    source_document = json.loads(selection.read_bytes())
    for item in source_document["files"].values():
        item["path"] = Path(item["path"]).relative_to(root).as_posix()
    selection.write_text(json.dumps(source_document), encoding="utf-8")
    runtime = json.loads(inputs["manifest"].read_bytes())
    runtime["selection"] = {**pin(selection), "path": selection.name}
    inputs["manifest"].write_text(json.dumps(runtime), encoding="utf-8")
    first = run(inputs)
    moved = tmp_path / "relocated-service"
    moved_release = moved / "release"
    moved_job = moved / "prediction-job"
    shutil.copytree(root, moved_release)
    shutil.copytree(inputs["state"], moved_job)
    snapshot = json.loads(json.dumps(inputs["snapshot"]).replace(
        str(root).replace("\\", "\\\\"), str(moved_release).replace("\\", "\\\\")))
    cache = ResponseCache(moved_job / "cache", lambda url: pytest.fail("relocation must not refetch"))
    report = run_refresh_predictions(accepted_pointer=snapshot,
              selected_manifest=moved_release / inputs["manifest"].name, state_dir=moved_job,
              decision_at=datetime(2026, 11, 1, tzinfo=timezone.utc), response_cache=cache)
    assert report["predictions"] == first["predictions"]
    assert report["decision_at"] == first["decision_at"]
    assert report["binding"] == first["binding"]
    assert report["original_selection"]["path"] == str(moved_release / selection.name)
    assert report["original_selection"]["sha256"] == first["original_selection"]["sha256"]


def test_fresh_profiles_cannot_replace_permanent_owner_dates(tmp_path):
    from dataclasses import replace
    from src.data.refresh_predictions import _merged_profiles
    from src.data.selected_source_inputs import SelectedSourceInputs
    from src.data.ufc_refresh import _load_scraped_fighter_lookup
    root = Path(__file__).resolve().parents[1] / "release_payload"
    sources = SelectedSourceInputs.from_manifest(root / "selected-source-manifest.json",
                                                processed_data_dir=root / "processed")
    observations = [
        {"name": "Javad Mahjoub", "fighter_url": "http://ufcstats.com/fighter-details/e9f9add1481fc11c",
         "dob": "2000-01-01", "height": "77 in", "reach": "77 in", "stance": "Orthodox"},
        {"name": "Ethyn Ewing", "fighter_url": "http://ufcstats.com/fighter-details/d776b5814f2400b4",
         "dob": "1996-02-19", "height": "77 in", "reach": "77 in", "stance": "Orthodox"},
    ]
    fresh = _merged_profiles(sources, observations, tmp_path / "current_profiles.csv")
    selected = replace(sources, files={**sources.files, "profiles": fresh})
    with selected.activate():
        bank = _load_scraped_fighter_lookup(selected.path("profiles"),
            profile_integrity_decisions_path=selected.path("profile_decisions"),
            prior_profile_integrity_decisions_path=selected.path("prior_profile_decisions"))
    assert bank["ufcstats:e9f9add1481fc11c"]["dob"].date().isoformat() == "1992-05-26"
    assert bank["ufcstats:d776b5814f2400b4"]["dob"].date().isoformat() == "1998-02-16"
    assert np.isnan(bank["ufcstats:e9f9add1481fc11c"]["weight"])


def test_original_completion_is_not_replaced_by_current_pointer(inputs):
    pointer = inputs["state"] / "data/accepted.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({**inputs["snapshot"], "generation_id": "later-unrelated-job"}))
    inputs["snapshot"]["accepted_pointer"] = str(pointer)
    report = run(inputs)
    assert report["accepted_generation"] == "updated-observed-1"


def test_explicit_portable_pointer_uses_producer_decoder(inputs):
    root = inputs["manifest"].parent
    original = inputs["snapshot"]
    encoded = json.dumps(original).replace(str(root).replace("\\", "\\\\"), "${BASELINE_DIR}")
    pointer = inputs["state"] / "data/accepted.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(encoded)
    cache = ResponseCache(inputs["state"] / "cache", inputs["fetch"])
    report = run_refresh_predictions(accepted_pointer=pointer,
              selected_manifest=inputs["manifest"], state_dir=inputs["state"],
              decision_at=DECISION, response_cache=cache)
    assert report["accepted_generation"] == "updated-observed-1"
