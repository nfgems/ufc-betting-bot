import json

from src.strategy import llm_operator, tennis_llm_operator
from src.web import app as web_app


def test_api_operator_decisions_returns_tennis_entries_with_no_store(tmp_path, monkeypatch):
    tennis_log = tmp_path / "tennis_veto_log.jsonl"
    tennis_log.write_text(
        json.dumps(
            {
                "verdict": "NO_VETO",
                "timestamp": "2026-03-24T02:40:57.517679+00:00",
                "fighter_a": "Tomas Martin Etcheverry",
                "fighter_b": "Tommy Paul",
                "decision_fighter": "Tommy Paul",
            }
        ) + "\n",
        encoding="utf-8",
    )

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
    monkeypatch.setattr(tennis_llm_operator, "TENNIS_LLM_VETO_LOG_PATH", tennis_log)

    client = web_app.app.test_client()

    response = client.get("/api/operator-decisions")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
    payload = response.get_json()
    assert payload["count"] == 2
    assert {decision["sport"] for decision in payload["decisions"]} == {"ufc", "tennis"}
    assert payload["decisions"][0]["sport"] == "tennis"


def test_api_operator_decisions_can_filter_tennis_only(tmp_path, monkeypatch):
    tennis_log = tmp_path / "tennis_veto_log.jsonl"
    tennis_log.write_text(
        json.dumps(
            {
                "verdict": "AUTO_SKIP",
                "timestamp": "2026-03-24T02:41:57.517679+00:00",
                "fighter_a": "Karolina Muchova",
                "fighter_b": "Alexandra Eala",
                "decision_fighter": "Alexandra Eala",
            }
        ) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(llm_operator, "load_decision_log", lambda: [])
    monkeypatch.setattr(tennis_llm_operator, "TENNIS_LLM_VETO_LOG_PATH", tennis_log)

    client = web_app.app.test_client()

    response = client.get("/api/operator-decisions?sport=tennis")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 1
    assert [decision["sport"] for decision in payload["decisions"]] == ["tennis"]
