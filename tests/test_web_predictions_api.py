import json
from datetime import datetime, timedelta, timezone

import pytest

from src import betting_window
from src.web import app as web_app


@pytest.fixture(autouse=True)
def _reset_dashboard_host(monkeypatch):
    monkeypatch.setattr(web_app, "_server_host", "127.0.0.1")


def test_public_read_predictions_do_not_require_token(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "_server_host", "0.0.0.0")
    monkeypatch.delenv("WEB_DASHBOARD_TOKEN", raising=False)
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/predictions-detail")

    assert response.status_code == 200


def test_api_predictions_detail_returns_enriched_prediction_fields(tmp_path, monkeypatch):
    payload = {
        "timestamp": "2026-03-09T20:57:34.931375",
        "global_feature_importance": [{"feature": "diff_skill", "importance": 0.12}],
        "predictions": [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "prob_a": 0.64,
                "prob_b": 0.36,
                "confidence": 0.64,
                "a_market_prob": 0.52,
                "b_market_prob": 0.48,
                "no_odds_prob_a": 0.59,
                "no_odds_prob_b": 0.41,
                "low_experience": True,
                "feature_highlights": [],
                "shap_values": [],
            }
        ],
    }
    (tmp_path / "predictions_cache.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/predictions-detail")

    assert response.status_code == 200
    data = response.get_json()
    assert data["timestamp"] == payload["timestamp"]
    assert data["data_timestamp"] == payload["timestamp"]
    assert data["prediction_count"] == 1
    assert data["cache_status"] == "stale"
    assert data["cache_available"] is True
    assert data["is_stale"] is True
    assert data["freshness_age_minutes"] is not None
    assert data["timestamp_parse_failed"] is False
    assert data["global_feature_importance"] == payload["global_feature_importance"]

    pred = data["predictions"][0]
    assert pred["predicted_winner"] == "Alpha"
    assert pred["predicted_side"] == "a"
    assert pred["market_pick"] == "Alpha"
    assert pred["market_gap"] == 0.12
    assert pred["market_disagreement"] is True
    assert pred["market_disagreement_note"] == (
        "The model is 12.0 percentage points higher on Alpha than the market is."
    )
    assert pred["no_odds_pick"] == "Alpha"
    assert pred["experience_flag"] == "low_sample"
    assert pred["confidence_tier"] in {"lean", "strong_lean"}
    assert pred["blended_prob_a"] > 0.0
    assert pred["edge_a"] > 0.0
    assert pred["best_edge"] == pred["edge_a"]
    assert pred["value_status"] == "potential_value"
    assert pred["value_has_positive_edge"] is True
    assert pred["value_execution_status"] == "stale"
    assert pred["pick_value_status"] == "potential_value"
    assert pred["pick_has_positive_edge"] is True
    assert pred["pick_execution_status"] == "stale"
    assert pred["pick_is_bettable"] is False
    assert pred["prediction_is_stale"] is True
    assert pred["prediction_cache_status"] == "stale"


def test_api_predictions_returns_same_enriched_contract_without_global_importance(tmp_path, monkeypatch):
    payload = {
        "timestamp": "2026-03-10T00:00:00",
        "predictions": [
            {
                "fighter_a": "Gamma",
                "fighter_b": "Delta",
                "prob_a": 0.48,
                "prob_b": 0.52,
                "confidence": 0.52,
                "a_market_prob": 0.40,
                "b_market_prob": 0.60,
                "feature_highlights": [],
                "shap_values": [],
            }
        ],
    }
    (tmp_path / "predictions_cache.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/predictions")

    assert response.status_code == 200
    data = response.get_json()
    assert "global_feature_importance" not in data
    assert data["data_timestamp"] == payload["timestamp"]
    assert data["prediction_count"] == 1
    assert data["cache_status"] == "stale"
    assert data["cache_available"] is True

    pred = data["predictions"][0]
    assert pred["predicted_winner"] == "Delta"
    assert pred["predicted_side"] == "b"
    assert pred["predicted_prob"] == 0.52
    assert pred["predicted_market_prob"] == 0.6
    assert pred["experience_flag"] == "normal"
    assert pred["pick_value_status"] == "pass"
    assert pred["pick_has_positive_edge"] is False
    assert pred["pick_execution_status"] == "pass"
    assert pred["value_status"] == "pass"
    assert pred["value_has_positive_edge"] is False
    assert pred["value_execution_status"] == "pass"


def test_api_predictions_detail_dedupes_cross_source_fighter_aliases(tmp_path, monkeypatch):
    event_date = "2026-05-30T22:00:00+00:00"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predictions": [
            {
                "fighter_a": "Luis Dias de Assis",
                "fighter_b": "Yi Sak Lee",
                "prob_a": 0.614,
                "prob_b": 0.386,
                "confidence": 0.614,
                "a_market_prob": 0.607,
                "b_market_prob": 0.393,
                "a_num_fights": 0,
                "b_num_fights": 0,
                "low_experience": True,
                "event_date": event_date,
                "feature_highlights": [],
                "shap_values": [],
            },
            {
                "fighter_a": "Luis Felipe Dias",
                "fighter_b": "Yi Sak Lee",
                "prob_a": 0.574,
                "prob_b": 0.426,
                "confidence": 0.574,
                "a_market_prob": 0.61,
                "b_market_prob": 0.39,
                "a_num_fights": 3,
                "b_num_fights": 2,
                "low_experience": False,
                "event_date": event_date,
                "feature_highlights": [],
                "shap_values": [],
            },
        ],
    }
    (tmp_path / "predictions_cache.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/predictions-detail")

    assert response.status_code == 200
    data = response.get_json()
    assert data["prediction_count"] == 1
    pred = data["predictions"][0]
    assert pred["fighter_a"] == "Luis Felipe Dias"
    assert pred["fighter_b"] == "Yi Sak Lee"
    assert pred["predicted_prob"] == 0.574
    assert pred["experience_flag"] == "normal"


def test_api_predictions_detail_separates_pick_from_best_priced_side(tmp_path, monkeypatch):
    payload = {
        "timestamp": "2026-03-09T20:57:34.931375",
        "predictions": [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "prob_a": 0.65,
                "prob_b": 0.35,
                "confidence": 0.65,
                "a_market_prob": 0.80,
                "b_market_prob": 0.20,
                "feature_highlights": [],
                "shap_values": [],
            }
        ],
    }
    (tmp_path / "predictions_cache.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/predictions-detail")

    assert response.status_code == 200
    pred = response.get_json()["predictions"][0]
    assert pred["predicted_winner"] == "Alpha"
    assert pred["predicted_side"] == "a"
    assert pred["predicted_edge"] < 0
    assert pred["market_gap"] == 0.15
    assert pred["market_disagreement"] is True
    assert pred["market_disagreement_note"] == (
        "The market is 15.0 percentage points higher on Alpha than the model is."
    )
    assert pred["pick_value_status"] == "pass"
    assert pred["pick_has_positive_edge"] is False
    assert pred["pick_execution_status"] == "pass"
    assert pred["pick_is_bettable"] is False
    assert pred["value_side"] == "b"
    assert pred["value_fighter"] == "Beta"
    assert pred["best_edge"] > 0
    assert pred["value_status"] == "potential_value"
    assert pred["value_has_positive_edge"] is True
    assert pred["value_execution_status"] == "stale"


def test_api_predictions_detail_distinguishes_positive_edge_from_execution_pipeline(tmp_path, monkeypatch):
    payload = {
        "timestamp": datetime.now().isoformat(),
        "predictions": [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "prob_a": 0.80,
                "prob_b": 0.20,
                "confidence": 0.80,
                "a_market_prob": 0.60,
                "b_market_prob": 0.40,
                "feature_highlights": [],
                "shap_values": [],
            }
        ],
    }
    (tmp_path / "predictions_cache.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/predictions-detail")

    assert response.status_code == 200
    pred = response.get_json()["predictions"][0]
    assert pred["pick_value_status"] == "potential_value"
    assert pred["pick_has_positive_edge"] is True
    assert pred["pick_execution_status"] == "pass"
    assert pred["pick_is_bettable"] is False


def test_api_predictions_detail_allows_bettable_status_for_current_cache(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    payload = {
        "timestamp": now.isoformat(),
        "predictions": [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "prob_a": 0.64,
                "prob_b": 0.36,
                "confidence": 0.64,
                "a_market_prob": 0.52,
                "b_market_prob": 0.48,
                "no_odds_prob_a": 0.59,
                "no_odds_prob_b": 0.41,
                "a_num_fights": 10,
                "b_num_fights": 8,
                "event_date": (now + timedelta(hours=24)).isoformat(),
                "feature_highlights": [],
                "shap_values": [],
            }
        ],
    }
    (tmp_path / "predictions_cache.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/predictions-detail")

    assert response.status_code == 200
    data = response.get_json()
    assert data["cache_status"] == "current"
    assert data["is_stale"] is False

    pred = data["predictions"][0]
    assert pred["pick_execution_status"] == "bettable_now"
    assert pred["pick_is_bettable"] is True
    assert pred["value_execution_status"] == "bettable_now"
    assert pred["prediction_is_stale"] is False
    assert pred["prediction_cache_status"] == "current"


def test_api_predictions_detail_marks_positive_edge_as_waiting_when_event_is_more_than_2_days_out(
    tmp_path,
    monkeypatch,
):
    now = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(betting_window, "_current_utc", lambda: now)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "predictions": [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "prob_a": 0.64,
                "prob_b": 0.36,
                "confidence": 0.64,
                "a_market_prob": 0.52,
                "b_market_prob": 0.48,
                "no_odds_prob_a": 0.59,
                "no_odds_prob_b": 0.41,
                "a_num_fights": 10,
                "b_num_fights": 8,
                "event_date": (now + timedelta(days=4)).isoformat(),
                "feature_highlights": [],
                "shap_values": [],
            }
        ],
    }
    (tmp_path / "predictions_cache.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/predictions-detail")

    assert response.status_code == 200
    pred = response.get_json()["predictions"][0]
    assert pred["pick_has_positive_edge"] is True
    assert pred["pick_execution_status"] == "pass"
    assert pred["pick_is_bettable"] is False
    assert pred["pick_filter_reason"] == "Bet window not open"
    assert "48h before fight starts" in pred["pick_filter_detail"]


def test_api_predictions_detail_marks_missing_cache_as_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/predictions-detail")

    assert response.status_code == 200
    data = response.get_json()
    assert data["prediction_count"] == 0
    assert data["data_timestamp"] is None
    assert data["cache_status"] == "missing"
    assert data["cache_available"] is False
    assert data["is_stale"] is True
    assert data["timestamp_parse_failed"] is False
    assert data["global_feature_importance"] == []


def test_api_predictions_marks_invalid_cache_as_error(tmp_path, monkeypatch):
    (tmp_path / "predictions_cache.json").write_text("{not valid json", encoding="utf-8")

    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/predictions")

    assert response.status_code == 200
    data = response.get_json()
    assert data["prediction_count"] == 0
    assert data["data_timestamp"] is None
    assert data["cache_status"] == "error"
    assert data["cache_available"] is False
    assert data["is_stale"] is True
    assert data["timestamp_parse_failed"] is False


def test_api_predictions_detail_marks_unparseable_timestamp_as_stale_but_unavailable(tmp_path, monkeypatch):
    payload = {
        "timestamp": "not-a-timestamp",
        "predictions": [],
    }
    (tmp_path / "predictions_cache.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    client = web_app.app.test_client()

    response = client.get("/api/predictions-detail")

    assert response.status_code == 200
    data = response.get_json()
    assert data["cache_status"] == "stale"
    assert data["cache_available"] is True
    assert data["is_stale"] is True
    assert data["freshness_age_minutes"] is None
    assert data["timestamp_parse_failed"] is True
