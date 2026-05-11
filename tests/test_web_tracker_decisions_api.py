from types import SimpleNamespace

import pytest

from src.web import app as web_app


@pytest.fixture(autouse=True)
def _reset_dashboard_host(monkeypatch):
    monkeypatch.setattr(web_app, "_server_host", "127.0.0.1")


def test_api_tracker_decisions_uses_card_day_keys_and_ignores_tracker_only_rows(monkeypatch):
    event_date = "2099-04-12T01:45:00+00:00"
    market_event_date = "2099-04-11 17:00:00+00"
    tracker_only_event_date = "2099-05-17T00:00:00+00:00"
    tracker_only_market_event_date = "2099-05-17 01:00:00+00"

    monkeypatch.setattr(
        web_app,
        "_load_prediction_payload",
        lambda include_global_feature_importance=False: {
            "predictions": [
                {
                    "fighter_a": "Dominick Reyes",
                    "fighter_b": "Johnny Walker",
                    "event_date": event_date,
                    "weight_class": "Light Heavyweight",
                }
            ]
        },
    )
    monkeypatch.setattr("src.strategy.llm_operator.load_decision_log", lambda: [])
    monkeypatch.setattr(
        "src.strategy.llm_operator.load_tracker_decision_log",
        lambda: [
            {
                "type": "decision",
                "timestamp": "2026-04-06T21:10:24.545782+00:00",
                "trader": "M",
                "decision_id": "M_1",
                "fighter_a": "Dominick Reyes",
                "fighter_b": "Johnny Walker",
                "event_date": event_date,
                "market_event_date": market_event_date,
                "event_title": "2099-05-17",
                "status": "outside_window",
                "summary": "Outside tracker window",
                "rationale": "Model Tracker skipped this fight because it is outside the tracker window.",
            },
            {
                "type": "decision",
                "timestamp": "2026-04-06T21:10:24.861192+00:00",
                "trader": "G",
                "decision_id": "G_1",
                "fighter_a": "Dominick Reyes",
                "fighter_b": "Johnny Walker",
                "event_date": event_date,
                "market_event_date": market_event_date,
                "event_title": "2099-05-17",
                "status": "outside_window",
                "summary": "Outside tracker window",
                "rationale": "Gemini Tracker skipped this fight because it is outside the tracker window.",
            },
            {
                "type": "decision",
                "timestamp": "2026-04-06T21:10:24.561629+00:00",
                "trader": "M",
                "decision_id": "M_2",
                "fighter_a": "Ronda Rousey",
                "fighter_b": "Gina Carano",
                "event_date": tracker_only_event_date,
                "market_event_date": tracker_only_market_event_date,
                "event_title": "2099-05-17",
                "status": "outside_window",
                "summary": "Outside tracker window",
                "rationale": "Model Tracker skipped this fight because it is outside the tracker window.",
            },
        ],
    )
    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: SimpleNamespace(bets=[]),
    )

    client = web_app.app.test_client()
    response = client.get("/api/tracker-decisions")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 1

    fight = payload["fights"][0]
    assert fight["fighter_a"] == "Dominick Reyes"
    assert fight["fighter_b"] == "Johnny Walker"
    assert fight["event_group_date"] == "2099-04-11"
    assert fight["M"]["status"] == "outside_window"
    assert fight["G"]["status"] == "outside_window"


def test_api_tracker_decisions_merges_ledger_bets_onto_tracker_card_day(monkeypatch):
    event_date = "2099-04-12T04:00:00+00:00"
    market_event_date = "2099-04-11 17:00:00+00"

    monkeypatch.setattr(
        web_app,
        "_load_prediction_payload",
        lambda include_global_feature_importance=False: {
            "predictions": [
                {
                    "fighter_a": "Jiri Prochazka",
                    "fighter_b": "Carlos Ulberg",
                    "event_date": event_date,
                    "market_event_date": market_event_date,
                    "weight_class": "Light Heavyweight",
                }
            ]
        },
    )
    monkeypatch.setattr("src.strategy.llm_operator.load_decision_log", lambda: [])
    monkeypatch.setattr(
        "src.strategy.llm_operator.load_tracker_decision_log",
        lambda: [
            {
                "type": "decision",
                "timestamp": "2026-04-08T21:39:08.389937+00:00",
                "trader": "M",
                "decision_id": "M_1",
                "fighter_a": "Jiri Prochazka",
                "fighter_b": "Carlos Ulberg",
                "event_date": event_date,
                "market_event_date": market_event_date,
                "status": "outside_window",
                "summary": "Outside tracker window",
                "rationale": "Model Tracker skipped this fight because it is outside the tracker window.",
            },
            {
                "type": "decision",
                "timestamp": "2026-04-08T21:39:09.106666+00:00",
                "trader": "G",
                "decision_id": "G_1",
                "fighter_a": "Jiri Prochazka",
                "fighter_b": "Carlos Ulberg",
                "event_date": event_date,
                "market_event_date": market_event_date,
                "status": "outside_window",
                "summary": "Outside tracker window",
                "rationale": "Gemini Tracker skipped this fight because it is outside the tracker window.",
            },
        ],
    )
    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: SimpleNamespace(
            bets=[
                {
                    "fighter": "Carlos Ulberg",
                    "opponent": "Jiri Prochazka",
                    "side": "a",
                    "edge": 0.026,
                    "reason": "Model edge",
                    "event_date": event_date,
                    "placed_at": "2026-04-08T21:39:07+00:00",
                    "_ledger_path": "bet_ledger_single.json",
                }
            ]
        ),
    )

    client = web_app.app.test_client()
    response = client.get("/api/tracker-decisions")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 1

    fight = payload["fights"][0]
    assert fight["fighter_a"] == "Jiri Prochazka"
    assert fight["fighter_b"] == "Carlos Ulberg"
    assert fight["event_group_date"] == "2099-04-11"
    assert fight["S"]["status"] == "bet"
    assert fight["S"]["text"] == "Carlos Ulberg"
    assert fight["M"]["status"] == "outside_window"
    assert fight["G"]["status"] == "outside_window"


def test_api_tracker_decisions_keeps_tracker_pick_after_later_started_log(monkeypatch):
    event_date = "2099-04-12T04:00:00+00:00"
    market_event_date = "2099-04-11 17:00:00+00"

    monkeypatch.setattr(
        web_app,
        "_load_prediction_payload",
        lambda include_global_feature_importance=False: {
            "predictions": [
                {
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "event_date": event_date,
                    "market_event_date": market_event_date,
                    "prob_a": 0.62,
                    "prob_b": 0.38,
                    "a_market_prob": 0.55,
                    "b_market_prob": 0.45,
                }
            ]
        },
    )
    monkeypatch.setattr("src.strategy.llm_operator.load_decision_log", lambda: [])
    monkeypatch.setattr(
        "src.strategy.llm_operator.load_tracker_decision_log",
        lambda: [
            {
                "type": "decision",
                "timestamp": "2026-05-09T23:01:00+00:00",
                "trader": "M",
                "decision_id": "M_same",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "event_date": event_date,
                "market_event_date": market_event_date,
                "status": "event_started",
                "summary": "Event already started",
                "rationale": "Model Tracker skipped this fight because the market event time is no longer in the future.",
            },
            {
                "type": "decision",
                "timestamp": "2026-05-09T22:01:00+00:00",
                "trader": "M",
                "decision_id": "M_same",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "event_date": event_date,
                "market_event_date": market_event_date,
                "status": "eligible",
                "summary": "Pick: Alpha",
                "pick": "Alpha",
                "edge": 0.07,
                "rationale": "Pure model pick: Alpha.",
            },
        ],
    )
    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: SimpleNamespace(bets=[]),
    )

    client = web_app.app.test_client()
    response = client.get("/api/tracker-decisions")

    assert response.status_code == 200
    fight = response.get_json()["fights"][0]
    assert fight["M"]["status"] == "eligible"
    assert fight["M"]["text"] == "Alpha"
    assert fight["M"]["rationale"] == "Pure model pick: Alpha."


def test_api_tracker_decisions_uses_tracker_ledger_bet_over_started_log(monkeypatch):
    event_date = "2099-04-12T04:00:00+00:00"
    market_event_date = "2099-04-11 17:00:00+00"

    monkeypatch.setattr(
        web_app,
        "_load_prediction_payload",
        lambda include_global_feature_importance=False: {
            "predictions": [
                {
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "event_date": event_date,
                    "market_event_date": market_event_date,
                    "prob_a": 0.62,
                    "prob_b": 0.38,
                    "a_market_prob": 0.55,
                    "b_market_prob": 0.45,
                }
            ]
        },
    )
    monkeypatch.setattr("src.strategy.llm_operator.load_decision_log", lambda: [])
    monkeypatch.setattr(
        "src.strategy.llm_operator.load_tracker_decision_log",
        lambda: [
            {
                "type": "decision",
                "timestamp": "2026-05-09T23:01:00+00:00",
                "trader": "G",
                "decision_id": "G_same",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "event_date": event_date,
                "market_event_date": market_event_date,
                "status": "event_started",
                "summary": "Event already started",
                "rationale": "Gemini Tracker skipped this fight because the market event time is no longer in the future.",
            },
        ],
    )
    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: SimpleNamespace(
            bets=[
                {
                    "fighter": "Beta",
                    "opponent": "Alpha",
                    "side": "b",
                    "edge": 0.04,
                    "reason": "Gemini pick",
                    "event_date": event_date,
                    "market_event_date": market_event_date,
                    "placed_at": "2026-05-09T22:01:00+00:00",
                    "_ledger_path": "bet_ledger_gemini_tracker.json",
                    "order_type": "marketable_limit",
                    "placement_state": "submitted",
                }
            ]
        ),
    )

    client = web_app.app.test_client()
    response = client.get("/api/tracker-decisions")

    assert response.status_code == 200
    fight = response.get_json()["fights"][0]
    assert fight["G"]["status"] == "bet"
    assert fight["G"]["text"] == "Beta"
    assert fight["G"]["rationale"] == "Gemini pick"


def test_api_tracker_decisions_marks_unmatched_markets(monkeypatch):
    event_date = "2099-04-12T04:00:00+00:00"

    monkeypatch.setattr(
        web_app,
        "_load_prediction_payload",
        lambda include_global_feature_importance=False: {
            "predictions": [
                {
                    "fighter_a": "Colby Thicknesse",
                    "fighter_b": "Vince Morales",
                    "event_date": event_date,
                    "weight_class": "Bantamweight",
                    "prob_a": 0.55,
                    "prob_b": 0.45,
                }
            ]
        },
    )
    monkeypatch.setattr("src.strategy.llm_operator.load_decision_log", lambda: [])
    monkeypatch.setattr("src.strategy.llm_operator.load_tracker_decision_log", lambda: [])
    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: SimpleNamespace(bets=[]),
    )

    client = web_app.app.test_client()
    response = client.get("/api/tracker-decisions")

    assert response.status_code == 200
    fight = response.get_json()["fights"][0]
    assert fight["S"]["status"] == "no_market"
    assert fight["C"]["status"] == "no_market"
    assert fight["M"]["status"] == "no_market"
    assert fight["G"]["status"] == "no_market"
