import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.strategy.execution_audit import (
    ExecutionAuditCollector,
    explain_conviction_row,
    explain_single_value_row,
    persist_cycle_payload,
)
from src.web import app as web_app


def _future_event() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()


def test_single_value_explanation_names_no_odds_gate():
    row = {
        "fighter_a": "Alpha",
        "fighter_b": "Beta",
        "prob_a": 0.80,
        "prob_b": 0.20,
        "a_market_prob": 0.60,
        "b_market_prob": 0.40,
        "no_odds_prob_a": 0.49,
        "no_odds_prob_b": 0.51,
        "a_num_fights": 8,
        "b_num_fights": 6,
        "event_date": _future_event(),
        "market_id": "market-1",
        "token_id_yes": "token-a",
        "token_id_no": "token-b",
    }

    decision = explain_single_value_row(pd.Series(row), min_edge=0.02, edge_scaling_base=0.02)

    assert decision["status"] == "skipped"
    assert decision["gate"] == "value_no_odds_edge"
    assert "no-odds edge" in decision["explanation"]
    assert decision["numbers"]["market_id"] == "market-1"
    assert decision["numbers"]["token_id"] == "token-a"


def test_conviction_explanation_names_experience_gate():
    row = {
        "fighter_a": "Alpha",
        "fighter_b": "Beta",
        "prob_a": 0.72,
        "prob_b": 0.28,
        "a_market_prob": 0.60,
        "b_market_prob": 0.40,
        "no_odds_prob_a": 0.67,
        "no_odds_prob_b": 0.33,
        "a_num_fights": 0,
        "b_num_fights": 6,
        "event_date": _future_event(),
    }

    decision = explain_conviction_row(pd.Series(row))

    assert decision["status"] == "skipped"
    assert decision["gate"] == "conviction_experience"
    assert "0 UFC fights" in decision["explanation"]
    assert decision["numbers"]["min_fight_threshold"] == 2


def test_execution_audit_collector_persists_cycle(tmp_path):
    collector = ExecutionAuditCollector(dry_run=True, event_title="UFC Test")
    collector.record_path(
        "S",
        {
            "fighter_a": "Alpha",
            "fighter_b": "Beta",
            "event_date": _future_event(),
        },
        status="skipped",
        gate="value_min_edge",
        explanation="Skipped by Single Trader because value edge was -1.0%, needs +2.0%.",
        numbers={"edge": -0.01, "required_edge": 0.02},
    )
    payload = collector.to_payload()

    persist_cycle_payload(
        payload,
        log_path=tmp_path / "execution_decision_audit.jsonl",
        latest_path=tmp_path / "execution_decision_audit_latest.json",
    )

    latest = json.loads((tmp_path / "execution_decision_audit_latest.json").read_text())
    assert latest["cycle_id"] == payload["cycle_id"]
    assert latest["fights"][0]["paths"]["S"]["gate"] == "value_min_edge"
    assert (tmp_path / "execution_decision_audit.jsonl").read_text().strip()


def test_api_execution_breakdown_reads_latest(tmp_path, monkeypatch):
    cycle = {
        "schema_version": 1,
        "cycle_id": "cycle-test",
        "started_at": "2026-06-19T00:00:00+00:00",
        "completed_at": "2026-06-19T00:01:00+00:00",
        "dry_run": False,
        "fight_count": 1,
        "path_counts": {"S": {"skipped": 1}},
        "fights": [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "event_date": "2026-06-20T00:00:00+00:00",
                "paths": {
                    "S": {
                        "label": "Single Trader / value pipeline",
                        "status": "skipped",
                        "gate": "value_no_odds_edge",
                        "explanation": "Skipped by Single Trader because no-odds edge was -11.0%, needs +1.0%.",
                        "numbers": {"edge": 0.02},
                        "order": {},
                        "stages": [],
                    }
                },
            }
        ],
    }
    (tmp_path / "execution_decision_audit_latest.json").write_text(json.dumps(cycle), encoding="utf-8")
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)

    response = web_app.app.test_client().get("/api/execution-breakdown")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["latest_available"] is True
    assert payload["cycle"]["cycle_id"] == "cycle-test"
    assert payload["cycle"]["fights"][0]["paths"]["S"]["gate"] == "value_no_odds_edge"
