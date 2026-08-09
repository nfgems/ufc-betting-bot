import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

from src.strategy.execution_audit import (
    ExecutionAuditCollector,
    explain_conviction_row,
    explain_single_value_row,
    load_execution_audit_cycles,
    persist_cycle_payload,
)
from src.web import app as web_app


def _stub_execution_enrichment(
    monkeypatch,
    *,
    ledger_bets=None,
    tracker_records=None,
):
    import src.strategy.tracker_decisions as tracker_decisions

    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: SimpleNamespace(bets=list(ledger_bets or [])),
    )
    monkeypatch.setattr(
        tracker_decisions,
        "load_tracker_decision_log",
        lambda: list(tracker_records or []),
    )


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


def test_execution_audit_history_is_reverse_paginated(tmp_path):
    log_path = tmp_path / "execution_decision_audit.jsonl"
    for number in range(1, 5):
        persist_cycle_payload(
            {
                "cycle_id": f"cycle-{number}",
                "completed_at": f"2026-06-{number:02d}T00:00:00+00:00",
                "fights": [{"large": "payload"}],
            },
            log_path=log_path,
            latest_path=tmp_path / "latest.json",
        )

    page = load_execution_audit_cycles(limit=2, offset=1, log_path=log_path)

    assert [cycle["cycle_id"] for cycle in page] == ["cycle-3", "cycle-2"]


def test_execution_audit_finalizes_unresolved_candidate_paths():
    collector = ExecutionAuditCollector(dry_run=False)
    row = {
        "fighter_a": "Alpha",
        "fighter_b": "Beta",
        "event_date": _future_event(),
    }
    collector.record_path(
        "S",
        row,
        status="candidate",
        gate="value_candidate",
        explanation="Candidate awaiting execution.",
        final=False,
    )

    payload = collector.to_payload()
    path = payload["fights"][0]["paths"]["S"]

    assert path["status"] == "incomplete"
    assert path["gate"] == "audit_incomplete"
    assert set(payload["fights"][0]["paths"]) == {"S", "C", "M"}


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
    _stub_execution_enrichment(monkeypatch)

    response = web_app.app.test_client().get("/api/execution-breakdown")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["latest_available"] is True
    assert payload["cycle"]["cycle_id"] == "cycle-test"
    assert payload["cycle"]["fights"][0]["paths"]["S"]["gate"] == "value_no_odds_edge"


def test_api_execution_history_returns_compact_paginated_index(tmp_path, monkeypatch):
    for number in range(1, 4):
        cycle = {
            "schema_version": 1,
            "cycle_id": f"cycle-{number}",
            "started_at": f"2026-06-{number:02d}T00:00:00+00:00",
            "completed_at": f"2026-06-{number:02d}T00:01:00+00:00",
            "dry_run": False,
            "event_title": "UFC Test",
            "fight_count": 1,
            "path_counts": {"S": {"skipped": 1}},
            "fights": [{"fighter_a": "Alpha", "fighter_b": f"Beta {number}", "paths": {}}],
        }
        persist_cycle_payload(
            cycle,
            log_path=tmp_path / "execution_decision_audit.jsonl",
            latest_path=tmp_path / "execution_decision_audit_latest.json",
        )
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    _stub_execution_enrichment(monkeypatch)

    response = web_app.app.test_client().get(
        "/api/execution-breakdown?history=1&limit=1"
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["cycle"]["cycle_id"] == "cycle-3"
    assert payload["cycle"]["fights"]
    assert payload["cycles"] == [
        {
            "schema_version": 1,
            "cycle_id": "cycle-3",
            "started_at": "2026-06-03T00:00:00+00:00",
            "completed_at": "2026-06-03T00:01:00+00:00",
            "dry_run": False,
            "event_title": "UFC Test",
            "fight_count": 1,
            "path_counts": {"S": {"skipped": 1}},
        }
    ]
    assert payload["has_more"] is True
    assert payload["next_offset"] == 1


def test_api_execution_breakdown_normalizes_duplicate_skip_to_already_bet(tmp_path, monkeypatch):
    cycle = {
        "schema_version": 1,
        "cycle_id": "cycle-duplicate",
        "started_at": "2026-06-19T00:00:00+00:00",
        "completed_at": "2026-06-19T00:01:00+00:00",
        "dry_run": False,
        "fight_count": 1,
        "path_counts": {"C": {"skipped": 1}},
        "fights": [
            {
                "fighter_a": "Andre Fili",
                "fighter_b": "Vinicius Oliveira",
                "event_date": "2026-06-20T21:00:00+00:00",
                "paths": {
                    "C": {
                        "label": "Conviction Trader",
                        "status": "skipped",
                        "gate": "duplicate_open_position",
                        "explanation": "Skipped because already have open bet on market 2556936, ledger #76.",
                        "numbers": {
                            "market_id": "2556936",
                            "existing_ledger_id": 76,
                        },
                        "order": {},
                        "stages": [],
                    }
                },
            }
        ],
    }
    (tmp_path / "execution_decision_audit_latest.json").write_text(json.dumps(cycle), encoding="utf-8")
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    _stub_execution_enrichment(
        monkeypatch,
        ledger_bets=[
            {
                "id": 467,
                "_original_id": 76,
                "_ledger_path": "logs/conviction_ledger.json",
                "fighter": "Vinicius Oliveira",
                "opponent": "Andre Fili",
                "market_id": "2556936",
                "token_id": "token-vinicius",
                "amount": 55.9,
                "price": 0.63,
                "shares": 88.73,
                "order_id": "order-123",
                "order_type": "marketable_limit",
                "placement_state": "matched",
                "status": "open",
                "model_prob": 0.74,
                "market_prob": 0.725,
                "edge": 0.015,
                "placed_at": "2026-06-19T01:00:00+00:00",
                "reason": "Conviction signal on Vinicius Oliveira: model 74%, no-odds 61%, market 72.5%, positive EV confirmed.",
            }
        ],
    )

    response = web_app.app.test_client().get("/api/execution-breakdown")

    assert response.status_code == 200
    payload = response.get_json()
    path = payload["cycle"]["fights"][0]["paths"]["C"]
    assert path["status"] == "already_bet"
    assert path["gate"] == "duplicate_open_position"
    assert "Already bet by Conviction Trader: $55.90 at 0.6300" in path["explanation"]
    assert "Current cycle did not place another order because duplicate_open_position found the existing position." in path["explanation"]
    assert "Original reason: Conviction signal on Vinicius Oliveira" in path["explanation"]
    assert path["original_reason"].startswith("Conviction signal on Vinicius Oliveira")
    assert path["order"]["ledger_id"] == 76
    assert path["order"]["display_ledger_id"] == 467
    assert path["order"]["order_id"] == "order-123"
    assert path["order"]["amount"] == 55.9
    assert path["order"]["price"] == 0.63
    assert payload["cycle"]["path_counts"]["C"] == {"already_bet": 1}


def test_api_execution_breakdown_enriches_tracker_already_bet(tmp_path, monkeypatch):
    cycle = {
        "schema_version": 1,
        "cycle_id": "cycle-tracker-duplicate",
        "started_at": "2026-06-19T00:00:00+00:00",
        "completed_at": "2026-06-19T00:01:00+00:00",
        "dry_run": False,
        "fight_count": 1,
        "path_counts": {"M": {"skipped": 1}},
        "fights": [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "event_date": "2026-06-20T21:00:00+00:00",
                "paths": {
                    "M": {
                        "label": "Model Tracker",
                        "status": "skipped",
                        "gate": "duplicate_open_position",
                        "explanation": "Skipped because already have open bet on market 123, ledger #14.",
                        "numbers": {
                            "market_id": "123",
                            "existing_ledger_id": 14,
                        },
                        "order": {},
                        "stages": [],
                    },
                    "S": {
                        "label": "Single Trader / value pipeline",
                        "status": "skipped",
                        "gate": "value_min_edge",
                        "explanation": "Skipped by Single Trader because value edge was -2.0%, needs +2.0%.",
                        "numbers": {"edge": -0.02},
                        "order": {},
                        "stages": [],
                    },
                },
            }
        ],
    }
    (tmp_path / "execution_decision_audit_latest.json").write_text(json.dumps(cycle), encoding="utf-8")
    monkeypatch.setattr(web_app, "LOGS_DIR", tmp_path)
    _stub_execution_enrichment(
        monkeypatch,
        ledger_bets=[
            {
                "id": 22,
                "_original_id": 14,
                "_ledger_path": "logs/model_tracker_ledger.json",
                "fighter": "Beta",
                "opponent": "Alpha",
                "event_date": "2026-06-20T21:00:00+00:00",
                "market_id": "123",
                "token_id": "token-beta",
                "amount": 1.25,
                "price": 0.42,
                "order_id": "model-order",
                "order_type": "limit_bid",
                "placement_state": "resting",
                "status": "open",
                "model_prob": 0.45,
                "market_prob": 0.42,
                "edge": 0.03,
                "reason": "Model tracker tiny calibration bet.",
            }
        ],
        tracker_records=[
            {
                "timestamp": "2026-06-19T00:50:00+00:00",
                "type": "decision",
                "decision_id": "model-1",
                "trader": "M",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "event_date": "2026-06-20T21:00:00+00:00",
                "pick": "Beta",
                "status": "eligible",
                "rationale": "Model tracker saw enough edge for a tiny bet.",
            },
                {
                    "timestamp": "2026-06-19T00:51:00+00:00",
                    "type": "outcome",
                    "decision_id": "model-1",
                    "trader": "M",
                    "bet_placed": True,
                "order_status": "resting",
            },
                {
                    "timestamp": "2026-06-19T00:52:00+00:00",
                    "type": "outcome",
                    "decision_id": "model-1",
                    "trader": "M",
                    "bet_placed": False,
                "order_status": "skipped",
                "error": "skipped_by_executor",
            },
        ],
    )

    response = web_app.app.test_client().get("/api/execution-breakdown")

    assert response.status_code == 200
    payload = response.get_json()
    paths = payload["cycle"]["fights"][0]["paths"]
    model_path = paths["M"]
    single_path = paths["S"]
    assert model_path["status"] == "already_bet"
    assert model_path["original_reason"] == "Model tracker tiny calibration bet."
    assert "Already bet by Model Tracker: $1.25 at 0.4200" in model_path["explanation"]
    assert "Tracker rationale: Model tracker saw enough edge for a tiny bet." in model_path["explanation"]
    assert model_path["tracker_decision"]["outcome"]["order_status"] == "resting"
    assert model_path["tracker_decision"]["outcome"]["retry_after_placement"] is True
    assert (
        model_path["tracker_decision"]["outcome"]["latest_attempt_status"]
        == "skipped"
    )
    assert (
        model_path["tracker_decision"]["outcome"]["latest_attempt_disposition"]
        == "already_placed"
    )
    assert (
        model_path["tracker_decision"]["outcome"]["latest_attempt"]["order_status"]
        == "skipped"
    )
    assert model_path["order"]["amount"] == 1.25
    assert single_path["status"] == "skipped"
    assert payload["cycle"]["path_counts"]["M"] == {"already_bet": 1}
    assert payload["cycle"]["path_counts"]["S"] == {"skipped": 1}
