from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.web import app as web_app


@pytest.fixture(autouse=True)
def _reset_dashboard_host(monkeypatch):
    monkeypatch.setattr(web_app, "_server_host", "127.0.0.1")
    web_app._endpoint_cache.clear()


def test_fight_relevance_uses_timestamp_before_card_day():
    assert web_app._fight_is_relevant(
        {
            "card_date": "2026-06-14",
            "event_date": "2026-06-14T23:00:00+00:00",
        },
        datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc),
    )


def test_fight_relevance_ignores_conflicting_timestamp_when_card_day_is_known():
    assert web_app._fight_is_relevant(
        {
            "card_date": "2026-06-14",
            "event_date": "2026-06-13T23:00:00+00:00",
        },
        datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc),
    )


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
    monkeypatch.setattr(
        "src.strategy.tracker_decisions.load_tracker_decision_log",
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
    monkeypatch.setattr(
        "src.strategy.tracker_decisions.load_tracker_decision_log",
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
    monkeypatch.setattr(
        "src.strategy.tracker_decisions.load_tracker_decision_log",
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
    monkeypatch.setattr(
        "src.strategy.tracker_decisions.load_tracker_decision_log",
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
                    "reason": "Model tracker pick",
                    "event_date": event_date,
                    "market_event_date": market_event_date,
                    "placed_at": "2026-05-09T22:01:00+00:00",
                    "_ledger_path": "bet_ledger_model_tracker.json",
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
    assert fight["M"]["status"] == "bet"
    assert fight["M"]["text"] == "Beta"
    assert fight["M"]["rationale"] == "Model tracker pick"


def test_api_tracker_decisions_keeps_placed_outcome_after_later_executor_retry(monkeypatch):
    event_date = "2099-04-12T04:00:00+00:00"
    market_event_date = "2099-04-11 17:00:00+00"
    tracker_records = []
    for trader in ("M",):
        decision_id = f"{trader}_same"
        tracker_records.extend(
            [
                {
                    "type": "decision",
                    "timestamp": "2026-05-09T22:00:00+00:00",
                    "trader": trader,
                    "decision_id": decision_id,
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "event_date": event_date,
                    "market_event_date": market_event_date,
                    "status": "eligible",
                    "summary": "Pick: Beta",
                    "pick": "Beta",
                    "rationale": f"{trader} picked Beta.",
                },
                {
                    "type": "outcome",
                    "timestamp": "2026-05-09T22:01:00+00:00",
                    "trader": trader,
                    "decision_id": decision_id,
                    "bet_placed": True,
                    "order_status": "placed",
                    "order_type": "marketable_limit",
                    "order_id": f"{trader.lower()}-order",
                },
                {
                    "type": "outcome",
                    "timestamp": "2026-05-09T22:30:00+00:00",
                    "trader": trader,
                    "decision_id": decision_id,
                    "bet_placed": False,
                    "order_status": "skipped",
                    "error": "skipped_by_executor",
                },
            ]
        )

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
                    "prob_a": 0.38,
                    "prob_b": 0.62,
                    "a_market_prob": 0.45,
                    "b_market_prob": 0.55,
                }
            ]
        },
    )
    monkeypatch.setattr(
        "src.strategy.tracker_decisions.load_tracker_decision_log",
        lambda: tracker_records,
    )
    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: SimpleNamespace(bets=[]),
    )

    response = web_app.app.test_client().get("/api/tracker-decisions")

    assert response.status_code == 200
    fight = response.get_json()["fights"][0]
    for trader in ("M",):
        cell = fight[trader]
        assert cell["status"] == "bet"
        assert cell["bet_placed"] is True
        assert cell["order_status"] == "placed"
        assert cell["order_type"] == "marketable_limit"
        assert cell["retry_after_placement"] is True
        assert cell["latest_attempt_status"] == "skipped"
        assert cell["latest_attempt_disposition"] == "already_placed"
        assert cell["latest_attempt"]["order_status"] == "skipped"
        assert cell["latest_attempt"]["error"] == "skipped_by_executor"


def test_tracker_outcome_index_does_not_promote_old_dry_run_over_live_failure():
    outcomes = web_app._build_tracker_outcome_index(
        [
            {
                "type": "outcome",
                "timestamp": "2026-05-09T22:01:00+00:00",
                "decision_id": "M_same",
                "bet_placed": True,
                "order_status": "dry_run",
                "dry_run": True,
            },
            {
                "type": "outcome",
                "timestamp": "2026-05-09T22:30:00+00:00",
                "decision_id": "M_same",
                "bet_placed": False,
                "order_status": "failed",
                "dry_run": False,
                "error": "live submission failed",
            },
        ]
    )

    assert outcomes["M_same"]["order_status"] == "failed"
    assert outcomes["M_same"]["bet_placed"] is False


def test_trader_bet_index_prefers_real_placement_over_later_dry_run():
    event_date = "2026-05-09T22:00:00+00:00"
    index = web_app._build_trader_bet_index(
        [
            {
                "trader": "M",
                "fighter": "Alpha",
                "opponent": "Beta",
                "event_date": event_date,
                "placed_at": "2026-05-09T22:01:00+00:00",
                "amount": 2.0,
                "dry_run": False,
                "status": "placed",
            },
            {
                "trader": "M",
                "fighter": "Alpha",
                "opponent": "Beta",
                "event_date": event_date,
                "placed_at": "2026-05-09T22:30:00+00:00",
                "amount": 99.0,
                "dry_run": True,
                "status": "dry_run",
            },
        ]
    )

    bet = next(iter(index.values()))
    assert bet["status"] == "placed"
    assert bet["amount"] == 2.0


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
    monkeypatch.setattr(
        "src.strategy.tracker_decisions.load_tracker_decision_log",
        lambda: [],
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
    assert fight["S"]["status"] == "no_market"
    assert fight["C"]["status"] == "no_market"
    assert fight["M"]["status"] == "no_market"


def test_api_tracker_decisions_groups_sunday_card_by_official_card_date(monkeypatch):
    official_card_date = "June 14, 2026"
    wrong_source_date = "2026-06-13"
    market_previous_day = "2026-06-13 21:00:00+00"

    monkeypatch.setattr(
        web_app,
        "_load_prediction_payload",
        lambda include_global_feature_importance=False: {
            "predictions": [
                {
                    "fighter_a": "Alex Pereira",
                    "fighter_b": "Carlos Ulberg",
                    "event_date": wrong_source_date,
                    "card_date": official_card_date,
                    "weight_class": "Light Heavyweight",
                },
                {
                    "fighter_a": "Diego Lopes",
                    "fighter_b": "Steve Garcia Jr.",
                    "event_date": "2026-06-14T22:10:00+00:00",
                    "market_event_date": market_previous_day,
                    "card_date": "2026-06-14",
                    "weight_class": "Featherweight",
                },
            ]
        },
    )
    monkeypatch.setattr(
        "src.strategy.tracker_decisions.load_tracker_decision_log",
        lambda: [
            {
                "type": "decision",
                "timestamp": "2026-06-07T18:00:00+00:00",
                "trader": "M",
                "decision_id": "M_whitehouse_1",
                "fighter_a": "Alex Pereira",
                "fighter_b": "Carlos Ulberg",
                "event_date": wrong_source_date,
                "status": "no_market",
                "summary": "No market matched",
                "rationale": "Model Tracker did not make its flat tracker bet because no active Polymarket market was matched for this fight.",
            },
            {
                "type": "decision",
                "timestamp": "2026-06-07T18:01:00+00:00",
                "trader": "M",
                "decision_id": "M_whitehouse_2",
                "fighter_a": "Diego Lopes",
                "fighter_b": "Steve Garcia Jr.",
                "event_date": "2026-06-14T22:10:00+00:00",
                "market_event_date": market_previous_day,
                "status": "outside_window",
                "summary": "Bet window not open",
                "rationale": "Model Tracker skipped this fight because Bet window opens later.",
            },
        ],
    )
    monkeypatch.setattr(
        web_app,
        "load_all_trader_ledgers",
        lambda: SimpleNamespace(bets=[]),
    )

    client = web_app.app.test_client()
    response = client.get("/api/tracker-decisions?history=1")

    assert response.status_code == 200
    fights = response.get_json()["fights"]
    assert len(fights) == 2
    assert {fight["event_group_date"] for fight in fights} == {"2026-06-14"}
    assert {fight["card_date"] for fight in fights} == {"2026-06-14"}
    pereira = next(fight for fight in fights if fight["fighter_a"] == "Alex Pereira")
    assert pereira["M"]["status"] == "no_market"
    lopes = next(fight for fight in fights if fight["fighter_a"] == "Diego Lopes")
    assert lopes["M"]["status"] == "outside_window"
