import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

import src.config as config_module
from src.web import app as web_app


def _make_repo_local_tmp_dir() -> Path:
    path = Path.cwd() / "data" / f"web-upcoming-events-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


@pytest.fixture(autouse=True)
def _reset_dashboard_host(monkeypatch):
    monkeypatch.setattr(web_app, "_server_host", "127.0.0.1")


def test_api_upcoming_events_uses_prediction_cache_to_lift_snapshot_fight_count(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        raw_dir = temp_root / "raw"
        logs_dir = temp_root / "logs"
        snapshot_dir = raw_dir / "snapshots"
        snapshot_dir.mkdir(parents=True)
        logs_dir.mkdir()

        future = datetime.now(timezone.utc) + timedelta(days=7)
        snapshot_payload = {
            "event": "UFC Fight Night: Example vs. Example",
            "event_date": future.strftime("%B %d, %Y"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fights": [{"fighter_a": "Ricky Simon", "fighter_b": "Adrian Yanez"}],
        }
        (snapshot_dir / "example.json").write_text(json.dumps(snapshot_payload), encoding="utf-8")

        predictions_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "predictions": [
                {
                    "fighter_a": "Ricky Simon",
                    "fighter_b": "Adrian Yanez",
                    "event_date": future.isoformat().replace("+00:00", "Z"),
                },
                {
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "event_date": (future + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
                },
            ],
        }
        (logs_dir / "predictions_cache.json").write_text(json.dumps(predictions_payload), encoding="utf-8")

        monkeypatch.setattr(config_module, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr(web_app, "LOGS_DIR", logs_dir)

        client = web_app.app.test_client()
        response = client.get("/api/upcoming-events")

        assert response.status_code == 200
        data = response.get_json()
        assert len(data) == 1
        assert data[0]["event"] == snapshot_payload["event"]
        assert data[0]["fight_count"] == 2
        assert data[0]["source"] == "snapshot+predictions"
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_api_upcoming_events_falls_back_to_prediction_cache_when_snapshots_missing(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        raw_dir = temp_root / "raw"
        logs_dir = temp_root / "logs"
        raw_dir.mkdir()
        logs_dir.mkdir()

        future = datetime.now(timezone.utc) + timedelta(days=10)
        predictions_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "predictions": [
                {
                    "fighter_a": "Gamma",
                    "fighter_b": "Delta",
                    "event_date": future.isoformat().replace("+00:00", "Z"),
                },
                {
                    "fighter_a": "Epsilon",
                    "fighter_b": "Zeta",
                    "event_date": future.isoformat().replace("+00:00", "Z"),
                },
            ],
        }
        (logs_dir / "predictions_cache.json").write_text(json.dumps(predictions_payload), encoding="utf-8")

        monkeypatch.setattr(config_module, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr(web_app, "LOGS_DIR", logs_dir)

        client = web_app.app.test_client()
        response = client.get("/api/upcoming-events")

        assert response.status_code == 200
        data = response.get_json()
        assert len(data) == 1
        assert data[0]["event"] == "Live UFC odds card"
        assert data[0]["fight_count"] == 2
        assert data[0]["source"] == "predictions_cache"
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
