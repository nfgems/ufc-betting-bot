import json
from datetime import datetime, timezone

import pytest

from src.prediction_history import (
    _prediction_event_hints,
    PREDICTION_HISTORY_SCHEMA_VERSION,
    archive_prediction_payload,
    initialize_prediction_history,
    load_prediction_history,
    prediction_archive_key,
    recover_prediction_rows_from_logs,
    recover_prediction_rows_from_model_tracker,
    recover_prediction_rows_from_operator_decisions,
    resolve_predicted_winner,
)


def _archive_row(
    *,
    fighter_a="Alpha Fighter",
    fighter_b="Beta Fighter",
    card_date="2026-08-01",
    generated_at="2026-08-01T12:00:00+00:00",
    prob_a=0.64,
    prob_b=0.36,
    **extra,
):
    row = {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "card_date": card_date,
        "event_date": f"{card_date}T23:00:00+00:00",
        "prediction_generated_at": generated_at,
        "prob_a": prob_a,
        "prob_b": prob_b,
        "confidence": max(prob_a, prob_b),
    }
    row.update(extra)
    return row


def test_archive_current_payload_preserves_display_fields_and_strips_live_internals(tmp_path):
    history_path = tmp_path / "predictions_history.json"
    archived_at = datetime(2026, 8, 1, 13, 0, tzinfo=timezone.utc)
    row = _archive_row(
        feature_highlights=[{"feature": "diff_roll_slpm", "value": 1.2}],
        method_stats={"a_ko_rate": 0.4, "b_ko_rate": 0.2},
        fighter_context={"a_wins": 12, "b_wins": 8},
        a_market_prob=0.55,
        b_market_prob=0.45,
        cache_key="event:live-only",
        event_context_snapshot={"weight_class": "Lightweight"},
        method_odds_fingerprint="method-v1",
        odds_snapshot={"a_fair_prob_avg": 0.55},
        operator_features={"diff_roll_slpm": 1.2},
        operator_provenance={"bundle_id": "bundle-1"},
        pair_key="alpha|beta",
        prediction_input_line_features={"line_movement": 0.01},
        prediction_input_odds_snapshot={"a_fair_prob_avg": 0.55},
        runtime_signature={"bundle_id": "bundle-1"},
        archive_untrusted_marker="must-not-survive",
    )
    payload = {
        "schema_version": 4,
        "timestamp": "2026-08-01T12:30:00+00:00",
        "predictions": [row],
    }

    result = archive_prediction_payload(payload, history_path, now=archived_at)
    loaded = load_prediction_history(history_path)

    assert result == {"added": 1, "updated": 0, "skipped": 0, "total": 1}
    assert loaded["schema_version"] == PREDICTION_HISTORY_SCHEMA_VERSION
    assert loaded["archive_status"] == "current"
    assert loaded["prediction_count"] == 1

    archived = loaded["predictions"][0]
    assert archived["predicted_winner"] == "Alpha Fighter"
    assert archived["detail_level"] == "full"
    assert archived["source"] == "prediction_cache"
    assert archived["source_schema_version"] == 4
    assert archived["source_cache_timestamp"] == payload["timestamp"]
    assert archived["first_archived_at"] == archived_at.isoformat()
    assert archived["last_archived_at"] == archived_at.isoformat()
    assert archived["feature_highlights"] == row["feature_highlights"]
    assert archived["method_stats"] == row["method_stats"]
    assert archived["fighter_context"] == row["fighter_context"]

    stripped = {
        "cache_key",
        "event_context_snapshot",
        "method_odds_fingerprint",
        "odds_snapshot",
        "operator_features",
        "operator_provenance",
        "pair_key",
        "prediction_input_line_features",
        "prediction_input_odds_snapshot",
        "runtime_signature",
        "archive_untrusted_marker",
    }
    assert stripped.isdisjoint(archived)


def test_sparse_legacy_winner_is_recovered_without_fabricated_probabilities(tmp_path):
    history_path = tmp_path / "predictions_history.json"
    payload = {
        "timestamp": "2025-11-15T18:00:00+00:00",
        "picks": [
            {
                "fighter1": "Legacy Red",
                "fighter2": "Legacy Blue",
                "fight": "Legacy Red vs Legacy Blue",
                "event_date": "November 15, 2025",
                "pick_side": "fighter_2",
            }
        ],
    }

    result = archive_prediction_payload(
        payload,
        history_path,
        source="recovered_legacy_snapshot",
        now=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    archived = load_prediction_history(history_path)["predictions"][0]

    assert result["added"] == 1
    assert archived["fighter_a"] == "Legacy Red"
    assert archived["fighter_b"] == "Legacy Blue"
    assert archived["predicted_winner"] == "Legacy Blue"
    assert archived["detail_level"] == "pick_only"
    assert archived["recovered"] is True
    assert archived["source_schema_version"] is None
    assert "prob_a" not in archived
    assert "prob_b" not in archived
    assert "confidence" not in archived

    assert resolve_predicted_winner(
        {
            "red_fighter": "Percent Red",
            "blue_fighter": "Percent Blue",
            "red_probability": 42,
            "blue_probability": 58,
        }
    ) == "Percent Blue"
    assert prediction_archive_key(
        {
            "fighter_a": "Ian Garry",
            "fighter_b": "Opponent Fighter",
            "card_date": "2026-08-01",
        }
    ) == prediction_archive_key(
        {
            "fighter_a": "Opponent Fighter",
            "fighter_b": "Ian Machado Garry",
            "card_date": "2026-08-01",
        }
    )


def test_archive_upserts_same_bout_prefers_latest_pick_and_keeps_rematches(tmp_path):
    history_path = tmp_path / "predictions_history.json"
    first_archived_at = datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc)
    original = _archive_row(
        generated_at="2026-08-01T09:00:00+00:00",
        feature_highlights=[{"feature": "original", "value": 1.0}],
    )
    archive_prediction_payload(
        {"schema_version": 4, "predictions": [original]},
        history_path,
        now=first_archived_at,
    )

    sparse_newer = {
        "fighter_a": "Beta Fighter",
        "fighter_b": "Alpha Fighter",
        "card_date": "2026-08-01",
        "prediction_generated_at": "2026-08-01T11:00:00+00:00",
        "predicted_winner": "Beta Fighter",
    }
    archive_prediction_payload(
        {"predictions": [sparse_newer]},
        history_path,
        source="recovered_bot_log",
        now=datetime(2026, 8, 1, 11, 30, tzinfo=timezone.utc),
    )

    after_sparse = load_prediction_history(history_path)["predictions"]
    assert len(after_sparse) == 1
    assert after_sparse[0]["predicted_winner"] == "Beta Fighter"
    assert after_sparse[0]["detail_level"] == "pick_only"
    assert after_sparse[0]["first_archived_at"] == first_archived_at.isoformat()
    assert after_sparse[0]["last_archived_at"] == "2026-08-01T11:30:00+00:00"

    refreshed = _archive_row(
        fighter_a="Beta Fighter",
        fighter_b="Alpha Fighter",
        generated_at="2026-08-01T12:00:00+00:00",
        prob_a=0.72,
        prob_b=0.28,
        feature_highlights=[{"feature": "refreshed", "value": 2.0}],
    )
    update_result = archive_prediction_payload(
        {"schema_version": 4, "predictions": [refreshed]},
        history_path,
        source="live_cache_completed",
        now=datetime(2026, 8, 1, 12, 30, tzinfo=timezone.utc),
    )

    rematch = _archive_row(
        card_date="2027-01-10",
        generated_at="2027-01-10T10:00:00+00:00",
        prob_a=0.41,
        prob_b=0.59,
        feature_highlights=[{"feature": "rematch", "value": -1.0}],
    )
    rematch_result = archive_prediction_payload(
        {"schema_version": 4, "predictions": [rematch]},
        history_path,
        now=datetime(2027, 1, 10, 11, 0, tzinfo=timezone.utc),
    )

    rows = load_prediction_history(history_path)["predictions"]
    assert update_result["updated"] == 1
    assert rematch_result["added"] == 1
    assert len(rows) == 2
    assert {row["card_date"] for row in rows} == {"2026-08-01", "2027-01-10"}

    first_bout = next(row for row in rows if row["card_date"] == "2026-08-01")
    assert first_bout["predicted_winner"] == "Beta Fighter"
    assert first_bout["feature_highlights"] == refreshed["feature_highlights"]
    assert first_bout["first_archived_at"] == first_archived_at.isoformat()
    assert first_bout["last_archived_at"] == "2026-08-01T12:30:00+00:00"
    assert prediction_archive_key(original) == prediction_archive_key(refreshed)
    assert prediction_archive_key(original) != prediction_archive_key(rematch)


def test_malformed_archive_is_reported_and_never_clobbered(tmp_path):
    history_path = tmp_path / "predictions_history.json"
    malformed = b'{"schema_version":1,"predictions":['
    history_path.write_bytes(malformed)

    loaded = load_prediction_history(history_path)
    assert loaded["archive_status"] == "error"
    assert loaded["prediction_count"] == 0
    assert loaded["error"]

    with pytest.raises(json.JSONDecodeError):
        archive_prediction_payload(
            {"schema_version": 4, "predictions": [_archive_row()]},
            history_path,
        )

    assert history_path.read_bytes() == malformed
    assert not (tmp_path / "predictions_history.json.tmp").exists()


def test_recover_prediction_rows_from_realistic_rotating_bot_logs(tmp_path):
    first_log = tmp_path / "bot.log.1"
    current_log = tmp_path / "bot.log"
    first_log.write_text(
        "\n".join(
            [
                "2026-01-01 10:00:00,123 [INFO] src.bot: Starting prediction pass",
                "  Alpha Fighter vs Beta Fighter:",
                "    Bookmakers: Alpha Fighter 55.0% | Beta Fighter 45.0%",
                "    Model:      Alpha Fighter 63.0% | Beta Fighter 37.0%",
                "    No-odds:    Alpha Fighter 60.0% | Beta Fighter 40.0%",
                "2026-01-08 10:00:00,456 [INFO] src.bot: Starting prediction pass",
                "  Alpha Fighter vs Beta Fighter:",
                "    Bookmakers: Alpha Fighter 54.0% | Beta Fighter 46.0%",
                "    Model:      Alpha Fighter 61.0% | Beta Fighter 39.0%",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    current_log.write_text(
        "\n".join(
            [
                "2026-02-10 12:30:00,789 [INFO] src.bot: Starting prediction pass",
                "  Beta Fighter vs Alpha Fighter:",
                "    Bookmakers: Beta Fighter 58.0% | Alpha Fighter 42.0%",
                "    Model:      Beta Fighter 66.0% | Alpha Fighter 34.0%",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = recover_prediction_rows_from_logs([first_log, current_log])

    assert len(rows) == 2
    assert rows[0]["prediction_generated_at"] == "2026-02-10 12:30:00,789"
    assert rows[0]["predicted_winner"] == "Beta Fighter"
    assert rows[0]["prob_a"] == pytest.approx(0.66)
    assert rows[0]["prob_b"] == pytest.approx(0.34)

    january = rows[1]
    assert january["prediction_generated_at"] == "2026-01-08 10:00:00,456"
    assert january["recovered_group_date"] == "2026-01-01"
    assert january["predicted_winner"] == "Alpha Fighter"
    assert january["a_market_prob"] == pytest.approx(0.54)
    assert january["b_market_prob"] == pytest.approx(0.46)


def test_recover_prediction_rows_from_model_tracker_uses_exact_model_side(tmp_path):
    ledger_path = tmp_path / "bet_ledger_model_tracker.json"
    ledger_path.write_text(
        json.dumps(
            {
                "bets": [
                    {
                        "fighter": "Beta Fighter",
                        "opponent": "Alpha Fighter",
                        "side": "b",
                        "model_prob": 0.68,
                        "market_prob": 0.57,
                        "probability_source": "model",
                        "event_date": "2026-08-01T20:00:00+00:00",
                        "market_event_date": "2026-08-01T20:00:00+00:00",
                        "card_date": "2026-08-01",
                        "event_title": "UFC Test Card",
                        "placed_at": "2026-08-01T18:30:00+00:00",
                    },
                    {
                        "fighter": "Gemini Pick",
                        "opponent": "Other Fighter",
                        "side": "a",
                        "model_prob": 0.50,
                        "probability_source": "market_neutral",
                        "event_date": "2026-08-01T21:00:00+00:00",
                    },
                    {
                        "fighter": "Incomplete Pick",
                        "opponent": "Incomplete Opponent",
                        "model_prob": 0.70,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = recover_prediction_rows_from_model_tracker(ledger_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["fighter_a"] == "Alpha Fighter"
    assert row["fighter_b"] == "Beta Fighter"
    assert row["predicted_side"] == "b"
    assert row["predicted_winner"] == "Beta Fighter"
    assert row["prob_a"] == pytest.approx(0.32)
    assert row["prob_b"] == pytest.approx(0.68)
    assert row["a_market_prob"] == pytest.approx(0.43)
    assert row["b_market_prob"] == pytest.approx(0.57)
    assert row["prediction_generated_at"] == "2026-08-01T18:30:00+00:00"
    assert row["recovery_provenance"] == "model_tracker_ledger"


def test_operator_recovery_chooses_opponent_when_bet_side_model_prob_is_below_half(
    tmp_path,
):
    decisions_path = tmp_path / "decision_log.jsonl"
    decisions = [
        {
            "fighter_a": "Value Underdog",
            "fighter_b": "Model Favorite",
            "bet_on": "Value Underdog",
            "bet_side": "a",
            "model_prob": 0.42,
            "market_prob": 0.30,
            "event_date": "2026-08-01T20:00:00+00:00",
            "card_date": "2026-08-01",
            "event_title": "UFC Test Card",
            "timestamp": "2026-08-01T18:00:00+00:00",
            "rationale": "Positive value despite not being the model favorite.",
            "research_summary": {"matchup_analysis": "Recovered context"},
        },
        {
            "fighter_a": "Late Alpha",
            "fighter_b": "Late Beta",
            "bet_on": "Late Alpha",
            "model_prob": 0.80,
            "event_date": "2026-08-01T20:00:00+00:00",
            "timestamp": "2026-08-01T20:05:00+00:00",
        },
    ]
    decisions_path.write_text(
        "\n".join([json.dumps(decisions[0]), "{malformed", json.dumps(decisions[1])])
        + "\n",
        encoding="utf-8",
    )

    rows = recover_prediction_rows_from_operator_decisions([decisions_path])

    assert len(rows) == 1
    row = rows[0]
    assert row["fighter_a"] == "Value Underdog"
    assert row["fighter_b"] == "Model Favorite"
    assert row["predicted_side"] == "b"
    assert row["predicted_winner"] == "Model Favorite"
    assert row["prob_a"] == pytest.approx(0.42)
    assert row["prob_b"] == pytest.approx(0.58)
    assert row["confidence"] == pytest.approx(0.58)
    assert row["a_market_prob"] == pytest.approx(0.30)
    assert row["b_market_prob"] == pytest.approx(0.70)
    assert row["recovery_provenance"] == "operator_decision_log"
    assert row["recovered_rationale"].startswith("Positive value")
    assert row["recovered_research_summary"] == decisions[0]["research_summary"]


def test_event_hints_select_the_final_pre_start_bot_log_prediction(tmp_path):
    log_path = tmp_path / "bot.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-08-01 10:00:00,000 [INFO] src.bot: Starting prediction pass",
                "  Alpha Fighter vs Beta Fighter:",
                "    Model:      Alpha Fighter 55.0% | Beta Fighter 45.0%",
                "2026-08-01 19:59:00,000 [INFO] src.bot: Starting prediction pass",
                "  Alpha Fighter vs Beta Fighter:",
                "    Model:      Alpha Fighter 38.0% | Beta Fighter 62.0%",
                "2026-08-01 20:05:00,000 [INFO] src.bot: Starting prediction pass",
                "  Alpha Fighter vs Beta Fighter:",
                "    Model:      Alpha Fighter 70.0% | Beta Fighter 30.0%",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    event_hints = {
        ("alphafighter", "betafighter"): [
            {
                "event_date": "2026-08-01T20:00:00+00:00",
                "card_date": "2026-08-01",
                "event_title": "UFC Final Snapshot Test",
            }
        ]
    }

    rows = recover_prediction_rows_from_logs(
        [log_path],
        event_hints=event_hints,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["event_date"] == "2026-08-01T20:00:00+00:00"
    assert row["card_date"] == "2026-08-01"
    assert row["event_title"] == "UFC Final Snapshot Test"
    assert row["prediction_generated_at"] == "2026-08-01 19:59:00,000"
    assert row["predicted_winner"] == "Beta Fighter"
    assert row["prob_a"] == pytest.approx(0.38)
    assert row["prob_b"] == pytest.approx(0.62)


def test_event_hints_use_latest_observed_corrected_start_time():
    hints = _prediction_event_hints(
        [
            {
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "event_date": "2026-08-01T17:00:00+00:00",
                "prediction_generated_at": "2026-07-30T12:00:00+00:00",
            },
            {
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "event_date": "2026-08-01T22:50:00+00:00",
                "prediction_generated_at": "2026-08-01T18:00:00+00:00",
            },
        ]
    )

    pair_hints = hints[("alphafighter", "betafighter")]
    assert len(pair_hints) == 1
    assert pair_hints[0]["event_date"] == "2026-08-01T22:50:00+00:00"


def test_log_recovery_skips_mapped_bout_with_only_post_start_output(tmp_path):
    log_path = tmp_path / "bot.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-08-01 20:05:00,000 [INFO] src.bot: Starting prediction pass",
                "  Alpha Fighter vs Beta Fighter:",
                "    Model:      Alpha Fighter 70.0% | Beta Fighter 30.0%",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = recover_prediction_rows_from_logs(
        [log_path],
        event_hints={
            ("alphafighter", "betafighter"): [
                {"event_date": "2026-08-01T20:00:00+00:00"}
            ]
        },
    )

    assert rows == []


def test_initialize_uses_canonical_data_paths_and_merges_final_log_with_research(
    tmp_path,
):
    data_dir = tmp_path / "data"
    logs_dir = data_dir / "logs"
    raw_data_dir = data_dir / "raw"
    operator_dir = data_dir / "operator"
    snapshots_dir = raw_data_dir / "snapshots"
    for directory in (logs_dir, operator_dir, snapshots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    decision = {
        "fighter_a": "Alpha Fighter",
        "fighter_b": "Beta Fighter",
        "bet_on": "Alpha Fighter",
        "bet_side": "a",
        "model_prob": 0.55,
        "event_date": "2026-08-01T20:00:00+00:00",
        "timestamp": "2026-08-01T10:00:00+00:00",
        "research_summary": {"matchup_analysis": "Recovered research"},
    }
    undated_later_decision = {
        "fighter_a": "Alpha Fighter",
        "fighter_b": "Beta Fighter",
        "bet_on": "Beta Fighter",
        "bet_side": "b",
        "model_prob": 0.65,
        "timestamp": "2026-08-01T19:50:00+00:00",
    }
    (operator_dir / "decision_log.jsonl").write_text(
        json.dumps(decision) + "\n" + json.dumps(undated_later_decision) + "\n",
        encoding="utf-8",
    )
    (snapshots_dir / "ufc_test.json").write_text(
        json.dumps(
            {
                "event": "UFC Test Card",
                "event_date": "August 1, 2026",
                "fights": [
                    {"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter"}
                ],
            }
        ),
        encoding="utf-8",
    )
    (logs_dir / "bot.log").write_text(
        "\n".join(
            [
                "2026-08-01 19:59:00,000 [INFO] src.bot: Starting prediction pass",
                "  Alpha Fighter vs Beta Fighter:",
                "    Model:      Alpha Fighter 38.0% | Beta Fighter 62.0%",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = initialize_prediction_history(
        logs_dir,
        data_dir=data_dir,
        raw_data_dir=raw_data_dir,
    )
    history = load_prediction_history(logs_dir / "predictions_history.json")

    assert summary["recovered_from_operator_decisions"] == 1
    assert summary["recovered_from_logs"] == 1
    assert history["prediction_count"] == 1
    row = history["predictions"][0]
    assert row["predicted_winner"] == "Beta Fighter"
    assert row["prediction_generated_at"] == "2026-08-01 19:59:00,000"
    assert row["card_date"] == "2026-08-01"
    assert row["event_title"] == "UFC Test Card"
    assert row["recovered_research_summary"] == decision["research_summary"]
