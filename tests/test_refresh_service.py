from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from src.data.refresh_service import RefreshService, create_app, generation_id
from src.data.ufc_update_schedule import UFCUpdateSchedule


TUESDAY = datetime(2026, 10, 13, 15, tzinfo=timezone.utc)
WEDNESDAY = datetime(2026, 10, 14, 15, tzinfo=timezone.utc)


def pin(path):
    body = path.read_bytes()
    return {"path": str(path), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


class Producer:
    def __init__(self):
        self.pointer = {"generation_id": "baseline", "processed_dir": "baseline/processed", "total_rows": 9453}
        self.calls = []
        self.recoveries = []
        self.active = None
        self.result = None
        self.interrupt = False

    def load_current_snapshot(self, **kwargs):
        return dict(self.pointer)

    def inspect_active_refresh(self, state_dir):
        return self.active

    def run_incremental_refresh(self, **kwargs):
        self.calls.append(kwargs)
        processed = Path(kwargs["state_dir"]) / "generations" / kwargs["job_id"] / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for name in ("fights_cleaned.csv", "features.csv"):
            path = processed / name
            path.write_text("saved fixture " + kwargs["job_id"])
            artifacts[name] = {**pin(path), "rows": 9454}
        self.pointer = {"generation_id": kwargs["job_id"], "processed_dir": str(processed), "total_rows": 9454,
                        "models": self.pointer["models"], "artifacts": artifacts}
        self.result = {**self.pointer, "status": "completed", "new_rows": 1, "job_id": kwargs["job_id"], "finished_at": kwargs["decision_at"].isoformat()}
        if self.interrupt:
            self.active = {"job_id": kwargs["job_id"], "phase": "STAGED"}
            self.interrupt = False
            raise SystemExit("interrupted after producer commit")
        return self.result

    def recover_incremental_refresh(self, **kwargs):
        self.recoveries.append(kwargs)
        self.active = None
        return self.result


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from src.data import refresh_service
    actual_now = refresh_service._now
    synthetic_wallclock = datetime(2027, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(refresh_service, "_now", lambda value=None: actual_now(synthetic_wallclock if value is None else value))
    baseline = tmp_path / "release"
    baseline.mkdir()
    manifest = baseline / "selected-source-manifest.json"
    manifest.write_text('{}\n', encoding="utf-8")
    producer = Producer()
    models = {}
    for role in ("primary", "no_odds", "logistic"):
        path = baseline / f"{role}.pkl"
        path.write_bytes(role.encode())
        models[role] = pin(path)
    processed = baseline / "processed"
    processed.mkdir()
    artifacts = {}
    for name in ("fights_cleaned.csv", "features.csv"):
        path = processed / name
        path.write_bytes(b"accepted inherited fixture\n")
        artifacts[name] = {**pin(path), "rows": 9453}
    producer.pointer.update(processed_dir=str(processed), models=models, artifacts=artifacts)
    (baseline / "selected-runtime-manifest.json").write_text(json.dumps({"models": models}) + "\n", encoding="utf-8")
    predictions = []

    def predict(**kwargs):
        predictions.append(kwargs)
        return {"status": "success", "accepted_generation": generation_id(producer.pointer),
                "model_pins": producer.pointer["models"], "data_pins": producer.pointer["artifacts"],
                "prediction_count": 1, "predictions": [{"fighter_a": "A", "fighter_b": "B", "prob_a": .6,
                                                          "prob_b": .4, "no_odds_prob_a": .55, "no_odds_prob_b": .45}]}

    def make():
        return RefreshService(root=tmp_path / "state", baseline=baseline, producer=producer, predictor=predict)

    return make, producer, predictions


def test_calendar_waits_and_suppresses_wednesday_after_success(setup):
    make, producer, predictions = setup
    service = make()
    before = datetime(2026, 10, 13, 14, 59, tzinfo=timezone.utc)
    assert service.tick(now=before)["schedule"]["reason"] == "waiting_for_tuesday_slot"
    assert not producer.calls
    assert not service.state_path.exists()
    status = service.tick(now=TUESDAY)
    assert status["last_completed_job"]["new_rows"] == 1
    assert producer.calls[0]["decision_at"] == TUESDAY
    assert predictions[0]["decision_at"] == TUESDAY
    assert make().tick(now=WEDNESDAY)["schedule"]["reason"] == "weekly_update_already_succeeded"
    assert len(producer.calls) == 1
    assert not service._schedule().load_state()["monthly_candidates"]


def test_genuinely_missed_tuesday_allows_wednesday(setup):
    make, producer, _ = setup
    make().tick(now=WEDNESDAY)
    assert producer.calls[0]["job_id"] == "2026-W42-wednesday"
    state = make()._schedule().load_state()
    assert state["weekly"]["2026-W42"]["slots"]["tuesday"]["failure"] == "tuesday_slot_missed"


def test_new_york_eleven_uses_winter_utc_offset(setup):
    make, producer, _ = setup
    before = datetime(2026, 12, 1, 15, 59, tzinfo=timezone.utc)
    due = datetime(2026, 12, 1, 16, tzinfo=timezone.utc)
    assert make().tick(now=before)["schedule"]["reason"] == "waiting_for_tuesday_slot"
    make().tick(now=due)
    assert producer.calls[0]["decision_at"] == due


def test_original_intent_recovers_before_late_fresh_plan(setup, monkeypatch):
    make, producer, _ = setup
    original = UFCUpdateSchedule.claim_data_update

    def interrupt(*args, **kwargs):
        raise SystemExit("before atomic claim")

    monkeypatch.setattr(UFCUpdateSchedule, "claim_data_update", interrupt)
    with pytest.raises(SystemExit):
        make().tick(now=TUESDAY)
    state = make()._state()
    assert state["pending"]["phase"] == "intent"
    assert not producer.calls
    monkeypatch.setattr(UFCUpdateSchedule, "claim_data_update", original)
    make().tick(now=WEDNESDAY)
    assert len(producer.calls) == 1
    assert producer.calls[0]["job_id"] == "2026-W42-tuesday"
    assert producer.calls[0]["decision_at"] == TUESDAY


def test_saved_producer_completion_recovers_without_second_acquisition(setup):
    make, producer, predictions = setup
    producer.interrupt = True
    with pytest.raises(SystemExit):
        make().tick(now=TUESDAY)
    assert make()._state()["pending"]["phase"] == "claimed"
    make().tick(now=WEDNESDAY)
    assert len(producer.calls) == 1
    assert len(producer.recoveries) == 1
    assert "decision_at" not in producer.recoveries[0]
    assert len(predictions) == 1
    assert make()._state()["pending"] is None


def test_calendar_completion_interruption_is_idempotent(setup, monkeypatch):
    make, producer, predictions = setup
    original = UFCUpdateSchedule.complete_data_update

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise SystemExit("after calendar completion")

    monkeypatch.setattr(UFCUpdateSchedule, "complete_data_update", interrupt)
    with pytest.raises(SystemExit):
        make().tick(now=TUESDAY)
    accepted = make()._schedule().load_state()["weekly"]["2026-W42"]["accepted_update"]
    monkeypatch.setattr(UFCUpdateSchedule, "complete_data_update", original)
    make().tick(now=WEDNESDAY)
    assert len(producer.calls) == 1
    assert len(predictions) == 1
    assert make()._schedule().load_state()["weekly"]["2026-W42"]["accepted_update"] == accepted


def test_uncertain_source_holds_and_never_creates_wednesday_claim(setup):
    make, producer, _ = setup
    producer.interrupt = True
    with pytest.raises(SystemExit):
        make().tick(now=TUESDAY)
    producer.result = {"job_id": "2026-W42-tuesday", "status": "held", "resume_allowed": False}
    status = make().tick(now=WEDNESDAY)
    assert status["state"] == "held"
    assert len(producer.calls) == 1
    assert "wednesday" not in make()._schedule().load_state()["weekly"]["2026-W42"]["slots"]


def test_completed_failure_preserves_data_and_allows_wednesday(setup):
    make, producer, _ = setup

    def fail(**kwargs):
        producer.calls.append(kwargs)
        return {"status": "failed", "failure_complete": True, "job_id": kwargs["job_id"], "error": "validated_failure", "finished_at": kwargs["decision_at"].isoformat()}

    original = producer.run_incremental_refresh
    producer.run_incremental_refresh = fail
    make().tick(now=TUESDAY)
    assert producer.pointer["generation_id"] == "baseline"
    producer.run_incremental_refresh = original
    make().tick(now=WEDNESDAY)
    assert [call["job_id"] for call in producer.calls] == ["2026-W42-tuesday", "2026-W42-wednesday"]


def test_http_keeps_last_good_publication_when_internal_pointer_advances(setup, monkeypatch):
    monkeypatch.delenv("WEB_DASHBOARD_TOKEN", raising=False)
    make, producer, _ = setup
    service = make()
    service.initialize_predictions(now=TUESDAY)
    client = create_app(service).test_client()
    assert client.get("/readyz").status_code == 200
    assert client.get("/api/predictions").json["accepted_generation"] == "baseline"
    assert client.get("/predictions").status_code == 200
    producer.pointer["generation_id"] = "later"
    assert client.get("/readyz").status_code == 200
    assert client.get("/api/predictions").json["accepted_generation"] == "baseline"
    status = client.get("/api/refresh-status").json
    assert status["pending_data_generation"] == "later"
    assert status["served_generation"] == "baseline"
    assert not producer.calls


def test_prediction_interruption_keeps_completed_data_and_original_instant(setup):
    make, producer, predictions = setup
    service = make()
    predictor = service._predictor()

    def interrupted(**kwargs):
        raise RuntimeError("prediction interrupted")

    service._predictor_override = interrupted
    with pytest.raises(RuntimeError, match="prediction interrupted"):
        service.tick(now=TUESDAY)
    assert service._state()["pending"]["phase"] == "data_completed"
    assert service._schedule().load_state()["weekly"]["2026-W42"]["slots"]["tuesday"]["status"] == "claimed"
    assert "${SERVICE_DIR}" in service.state_path.read_text()
    service._predictor_override = predictor
    service.tick(now=WEDNESDAY)
    assert len(producer.calls) == 1
    assert predictions[0]["decision_at"] == TUESDAY


def test_prediction_failure_retains_served_pair_until_atomic_replacement(setup, monkeypatch):
    monkeypatch.delenv("WEB_DASHBOARD_TOKEN", raising=False)
    make, producer, predictions = setup
    service = make()
    service.initialize_predictions(now=TUESDAY)
    original_publication = service.published_path.read_bytes()
    predictor = service._predictor()

    def failed_prediction(**kwargs):
        raise RuntimeError("replacement prediction failed")

    service._predictor_override = failed_prediction
    with pytest.raises(RuntimeError, match="replacement prediction failed"):
        service.tick(now=TUESDAY)
    assert producer.pointer["generation_id"] == "2026-W42-tuesday"
    assert service.published_path.read_bytes() == original_publication
    attempt = service._schedule().load_state()["weekly"]["2026-W42"]["slots"]["tuesday"]
    assert attempt["status"] == "claimed"
    client = create_app(service).test_client()
    assert client.get("/readyz").status_code == 200
    assert client.get("/api/predictions").json["accepted_generation"] == "baseline"
    service._predictor_override = predictor
    service.tick(now=WEDNESDAY)
    assert len(producer.calls) == 1
    assert len(predictions) == 2
    assert service.published_path.read_bytes() != original_publication
    assert client.get("/api/predictions").json["accepted_generation"] == "2026-W42-tuesday"
    assert client.get("/readyz").json["total_rows"] == 9454


def test_interruption_after_publication_reuses_exact_pair_without_more_work(setup):
    make, producer, predictions = setup
    service = make()
    publish = service._publish_pair

    def interrupted(*args, **kwargs):
        publish(*args, **kwargs)
        raise SystemExit("interrupted after atomic publication")

    service._publish_pair = interrupted
    with pytest.raises(SystemExit):
        service.tick(now=TUESDAY)
    original_publication = service.published_path.read_bytes()
    original_report = Path(service.published_pair()[0]["prediction_report"]["path"]).read_bytes()
    assert service._state()["pending"]["phase"] == "data_completed"
    assert service._schedule().load_state()["weekly"]["2026-W42"]["slots"]["tuesday"]["status"] == "claimed"
    recovered = make()
    recovered.tick(now=WEDNESDAY)
    assert len(producer.calls) == 1
    assert len(predictions) == 1
    assert recovered._state()["pending"] is None
    assert recovered.published_path.read_bytes() == original_publication
    assert Path(recovered.published_pair()[0]["prediction_report"]["path"]).read_bytes() == original_report
    assert recovered._schedule().load_state()["weekly"]["2026-W42"]["slots"]["tuesday"]["status"] == "succeeded"


def test_published_hash_drift_fails_readiness(setup, monkeypatch):
    monkeypatch.delenv("WEB_DASHBOARD_TOKEN", raising=False)
    service = setup[0]()
    service.initialize_predictions(now=TUESDAY)
    _publication, pointer, _report = service.published_pair()
    Path(pointer["artifacts"]["features.csv"]["path"]).write_bytes(b"drift")
    assert create_app(service).test_client().get("/readyz").status_code == 503


def test_relocated_payload_reuses_same_manifest_binding(setup, tmp_path):
    make, producer, predictions = setup
    service = make()
    service.initialize_predictions(now=TUESDAY)
    moved = tmp_path / "moved_release"
    moved.mkdir()
    for name in ("selected-source-manifest.json", "selected-runtime-manifest.json"):
        (moved / name).write_bytes((service.baseline / name).read_bytes())
    relocated = RefreshService(root=service.root, baseline=moved, producer=producer,
                               predictor=service._predictor())
    assert relocated._state()["binding"] == service.binding


def test_http_preserves_dashboard_token_with_secure_browser_session(setup, monkeypatch):
    monkeypatch.setenv("WEB_DASHBOARD_TOKEN", "test-dashboard-access")
    service = setup[0]()
    service.initialize_predictions(now=TUESDAY)
    client = create_app(service).test_client()
    assert client.get("/api/predictions").status_code == 401
    assert client.get("/predictions").status_code == 302
    assert client.get("/readyz").status_code == 200
    assert client.post("/login", data={"token": "wrong"}, base_url="https://localhost").status_code == 401
    response = client.post("/login", data={"token": "test-dashboard-access"}, base_url="https://localhost")
    assert response.status_code == 302
    assert "Secure" in response.headers["Set-Cookie"]
    assert "HttpOnly" in response.headers["Set-Cookie"]
    assert client.get("/api/predictions", base_url="https://localhost").status_code == 200


def test_manual_job_has_original_intent_and_does_not_fabricate_calendar_slot(setup):
    make, producer, _ = setup
    service = make()
    service.refresh_now(job_id="manual-initial", now=TUESDAY)
    make().refresh_now(job_id="manual-initial", now=WEDNESDAY)
    assert len(producer.calls) == 1
    assert service._schedule().load_state()["weekly"] == {}
    assert make()._state()["completed_jobs"]["manual-initial"]["decision"]["action"] == "manual_refresh"


@pytest.mark.parametrize("job_id", ["initial", "Initial", "2026-W42-tuesday", "2026-w42-WEDNESDAY", "NUL", "COM1"])
def test_manual_namespace_cannot_replace_calendar_startup_or_native_roles(setup, job_id):
    make, producer, _ = setup
    service = make()
    with pytest.raises(ValueError, match="reserved"):
        service.refresh_now(job_id=job_id, now=TUESDAY)
    assert not service.state_path.exists()
    assert not producer.calls


def test_naive_clock_and_changed_manifest_fail_before_source_work(setup):
    make, producer, _ = setup
    with pytest.raises(ValueError, match="aware"):
        make().tick(now=datetime(2026, 10, 13, 11))
    service = make()
    service.initialize_predictions(now=TUESDAY)
    service.selected_manifest.write_text('{"changed":true}\n')
    with pytest.raises(RuntimeError, match="binding"):
        make().tick(now=TUESDAY)
    assert not producer.calls
