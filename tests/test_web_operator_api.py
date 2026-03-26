import json

from src.strategy import llm_operator
from src.web import app as web_app


def test_api_operator_decisions_returns_ufc_entries_with_no_store(monkeypatch):
    monkeypatch.setattr(
        llm_operator,
        "load_decision_log",
        lambda: [
            {
                "verdict": "PASS",
                "timestamp": "2026-03-23T21:55:14.235844+00:00",
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "bet_on": "Fighter A",
            }
        ],
    )

    client = web_app.app.test_client()

    response = client.get("/api/operator-decisions")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
    payload = response.get_json()
    assert payload["count"] == 1
    assert payload["decisions"][0]["sport"] == "ufc"


def test_api_operator_decisions_can_limit_results(monkeypatch):
    monkeypatch.setattr(
        llm_operator,
        "load_decision_log",
        lambda: [
            {
                "verdict": "PASS",
                "timestamp": "2026-03-24T02:41:57.517679+00:00",
                "fighter_a": "New Fighter",
                "fighter_b": "New Opponent",
            },
            {
                "verdict": "BLOCK",
                "timestamp": "2026-03-23T02:41:57.517679+00:00",
                "fighter_a": "Old Fighter",
                "fighter_b": "Old Opponent",
            },
        ],
    )

    client = web_app.app.test_client()

    response = client.get("/api/operator-decisions?limit=1")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 1
    assert payload["total_count"] == 2
    assert [decision["fighter_a"] for decision in payload["decisions"]] == ["New Fighter"]
