import pytest

from src.web import app as web_app


@pytest.fixture(autouse=True)
def _reset_dashboard_host(monkeypatch):
    monkeypatch.setattr(web_app, "_server_host", "127.0.0.1")


def _operator_decision(**overrides):
    base = {
        "verdict": "BLOCK",
        "confidence": 1.0,
        "model_prob": 0.58,
        "operator_prob": 0.58,
        "market_prob": 0.52,
        "edge": 0.06,
        "rationale": "Operator blocked: stale matchup read.",
        "fighter_a": "Alpha",
        "fighter_b": "Beta",
        "bet_on": "Alpha",
        "decision_context": "S",
        "event_title": "UFC Test",
        "event_date": "2099-04-12T04:00:00+00:00",
        "timestamp": "2026-05-20T10:00:00+00:00",
        "risk_flags": ["stale_matchup"],
        "research_summary": {
            "fighter_assessment": "Alpha sharper striker.",
            "grounded_research": {
                "fight_status": "upcoming",
                "memo_text": "Competitive flyweight matchup.",
                "recent_form": "Alpha 3-0, Beta 1-2.",
                "level_of_competition": "Alpha stepped up.",
                "style_matchup": "Striker vs grappler.",
                "paths_to_victory": "Alpha by decision.",
                "model_pick_concerns": "Beta's wrestling.",
                "verified_records": {"fighter_a": "10-2", "fighter_b": "8-4", "source": "FightMatrix"},
                "key_flags": ["short notice"],
                "sources": ["https://example.com/a", "https://example.com/b"],
                "model_used": "gemini-3.1-pro-preview",
                "cached": False,
            },
        },
    }
    base.update(overrides)
    return base


def _tracker_record(**overrides):
    base = {
        "type": "decision",
        "trader": "G",
        "decision_id": "G_1",
        "status": "eligible",
        "summary": "Pick: Beta",
        "pick": "Beta",
        "confidence": 0.6,
        "market_prob": 0.45,
        "fighter_a": "Alpha",
        "fighter_b": "Beta",
        "event_title": "UFC Test",
        "event_date": "2099-04-12T04:00:00+00:00",
        "market_event_date": "2099-04-11 17:00:00+00",
        "weight_class": "Flyweight",
        "timestamp": "2026-05-21T10:00:00+00:00",
        "rationale": "Gemini likes Beta's grappling.",
        "fighter_assessment": "Beta grinds it out.",
        "sources": ["https://example.com/c"],
        "grounded_research": {
            "memo_text": "Beta's wrestling is the X factor.",
            "recent_form": "Beta on a 2-fight skid but tough outs.",
            "sources": ["https://example.com/c"],
            "model_used": "gemini-3.1-pro-preview",
        },
    }
    base.update(overrides)
    return base


def test_gemini_reasoning_merges_operator_and_tracker(monkeypatch):
    monkeypatch.setattr(
        "src.strategy.llm_operator.load_decision_log",
        lambda: [_operator_decision()],
    )
    monkeypatch.setattr(
        "src.strategy.llm_operator.load_tracker_decision_log",
        lambda: [
            _tracker_record(),
            # Model tracker — must be excluded (not Gemini)
            _tracker_record(trader="M", decision_id="M_1", timestamp="2026-05-22T10:00:00+00:00"),
            # Outcome record — must be excluded
            _tracker_record(type="outcome", decision_id="G_out", timestamp="2026-05-23T10:00:00+00:00"),
        ],
    )

    client = web_app.app.test_client()
    resp = client.get("/api/gemini-reasoning")
    assert resp.status_code == 200
    payload = resp.get_json()

    assert payload["count"] == 2  # 1 operator + 1 Gemini tracker decision
    assert payload["total_count"] == 2
    assert payload["research_count"] == 2

    # Newest first: tracker (05-21) before operator (05-20)
    assert [e["source"] for e in payload["entries"]] == ["tracker", "operator"]

    tracker_entry = payload["entries"][0]
    assert tracker_entry["pick"] == "Beta"
    assert tracker_entry["source_label"] == "Gemini Tracker (G)"
    assert tracker_entry["grounded_research"]["memo_text"].startswith("Beta's wrestling")
    assert tracker_entry["has_research"] is True

    operator_entry = payload["entries"][1]
    assert operator_entry["verdict"] == "BLOCK"
    assert operator_entry["bet_on"] == "Alpha"
    assert operator_entry["sources"] == ["https://example.com/a", "https://example.com/b"]
    assert operator_entry["verified_records"]["source"] == "FightMatrix"
    assert operator_entry["has_research"] is True


def test_gemini_reasoning_source_filter(monkeypatch):
    monkeypatch.setattr(
        "src.strategy.llm_operator.load_decision_log",
        lambda: [_operator_decision()],
    )
    monkeypatch.setattr(
        "src.strategy.llm_operator.load_tracker_decision_log",
        lambda: [_tracker_record()],
    )

    client = web_app.app.test_client()

    op_only = client.get("/api/gemini-reasoning?source=operator").get_json()
    assert op_only["count"] == 1
    assert all(e["source"] == "operator" for e in op_only["entries"])

    tracker_only = client.get("/api/gemini-reasoning?source=tracker").get_json()
    assert tracker_only["count"] == 1
    assert all(e["source"] == "tracker" for e in tracker_only["entries"])


def test_gemini_reasoning_limit_and_has_research_flag(monkeypatch):
    thin_operator = _operator_decision(
        verdict="PASS",
        rationale="Operator skipped: fight already has a recorded bet/order",
        risk_flags=["existing_bet"],
        research_summary={},
        timestamp="2026-05-19T10:00:00+00:00",
    )
    monkeypatch.setattr(
        "src.strategy.llm_operator.load_decision_log",
        lambda: [_operator_decision(), thin_operator],
    )
    monkeypatch.setattr(
        "src.strategy.llm_operator.load_tracker_decision_log",
        lambda: [],
    )

    client = web_app.app.test_client()
    payload = client.get("/api/gemini-reasoning?limit=1").get_json()

    assert payload["total_count"] == 2
    assert payload["count"] == 1  # limit applied
    assert payload["research_count"] == 1  # only the rich one has research

    # The thin operator decision carries no grounded research
    full = client.get("/api/gemini-reasoning").get_json()
    by_ts = {e["timestamp"]: e for e in full["entries"]}
    assert by_ts["2026-05-19T10:00:00+00:00"]["has_research"] is False
