import json
from datetime import datetime, timezone

import pytest

from src.prediction_history import (
    _apply_event_hints_to_recovered_rows,
    _apply_snapshot_card_hints,
    _load_completed_event_card_hints,
    _merge_card_hint_maps,
    _load_snapshot_card_hints,
    _prediction_event_hints,
    PREDICTION_HISTORY_SCHEMA_VERSION,
    archive_prediction_payload,
    initialize_prediction_history,
    load_prediction_history,
    prediction_archive_key,
    reconcile_prediction_history_cards,
    recover_prediction_rows_from_bet_ledgers,
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


def test_snapshot_hints_map_removed_log_bout_to_official_card_and_one_row(tmp_path):
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()
    for name, root_date, observed_at in (
        ("early.json", "August 16, 2026", "2026-07-01T00:00:00+00:00"),
        ("latest.json", "August 15, 2026", "2026-08-01T00:00:00+00:00"),
    ):
        (snapshots_dir / name).write_text(
            json.dumps(
                {
                    "event": "UFC Test Card",
                    "event_date": root_date,
                    "event_url": (
                        "https://www.ufc.com/event/"
                        "ufc-test-card-august-15-2026"
                    ),
                    "timestamp": observed_at,
                    "fights": [
                        {
                            "fighter_a": "Removed Fighter",
                            "fighter_b": "Original Opponent",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    log_path = tmp_path / "bot.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-06-30 12:00:00,000 [INFO] src.bot: Prediction pass",
                "  Removed Fighter vs Original Opponent:",
                "    Model:      Removed Fighter 55.0% | Original Opponent 45.0%",
                "2026-08-14 12:00:00,000 [INFO] src.bot: Prediction pass",
                "  Removed Fighter vs Original Opponent:",
                "    Model:      Removed Fighter 44.0% | Original Opponent 56.0%",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    card_hints = _load_snapshot_card_hints(snapshots_dir)
    rows = recover_prediction_rows_from_logs(
        [log_path],
        card_hints=card_hints,
    )

    assert len(rows) == 1
    assert rows[0]["card_date"] == "2026-08-15"
    assert rows[0]["event_title"] == "UFC Test Card"
    assert rows[0]["prediction_generated_at"] == "2026-08-14 12:00:00,000"
    assert rows[0]["predicted_winner"] == "Original Opponent"


def test_log_card_context_dates_bout_and_reorients_auxiliary_probabilities(tmp_path):
    log_path = tmp_path / "bot.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-07-08 01:28:13,153 [INFO] src.bot: Built features",
                "  Alpha Fighter vs Beta Fighter:",
                "    Bookmakers: Beta Fighter 40.0% | Alpha Fighter 60.0%",
                "    Model:      Alpha Fighter 55.0% | Beta Fighter 45.0%",
                "    No-odds:    Beta Fighter 35.0% | Alpha Fighter 65.0%",
                "2026-07-10 00:08:27,363 [INFO] src.bot: No official card-row context for Alpha Fighter vs Beta Fighter on 2026-07-18 — inferred weight class",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = recover_prediction_rows_from_logs([log_path])

    assert len(rows) == 1
    assert rows[0]["card_date"] == "2026-07-18"
    assert rows[0]["a_market_prob"] == pytest.approx(0.60)
    assert rows[0]["b_market_prob"] == pytest.approx(0.40)
    assert rows[0]["no_odds_prob_a"] == pytest.approx(0.65)
    assert rows[0]["no_odds_prob_b"] == pytest.approx(0.35)


def test_mixed_cp1252_log_name_matches_utf8_card_hint(tmp_path):
    log_path = tmp_path / "bot.log"
    lines = [
        "2026-07-08 01:28:13,153 [INFO] src.bot: Built features",
        "  M\u00e1rcio Barbosa vs Beta Fighter:",
        "    Model:      M\u00e1rcio Barbosa 55.0% | Beta Fighter 45.0%",
    ]
    log_path.write_bytes(("\n".join(lines) + "\n").encode("cp1252"))
    card_hints = {
        ("betafighter", "marciobarbosa"): [
            {"card_date": "2026-07-18", "event_title": "UFC Encoding Test"}
        ]
    }

    rows = recover_prediction_rows_from_logs([log_path], card_hints=card_hints)

    assert len(rows) == 1
    assert rows[0]["fighter_a"] == "M\u00e1rcio Barbosa"
    assert rows[0]["card_date"] == "2026-07-18"


def test_log_uses_only_agreeing_nearby_fighter_reference_dates(tmp_path):
    log_path = tmp_path / "bot.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-07-08 01:00:00,000 [INFO] src.data.fighter_lookup: Processed live snapshot for Alpha Fighter may be stale relative to 2026-07-18; attempting live refresh first",
                "2026-07-08 01:00:01,000 [INFO] src.data.fighter_lookup: Processed live snapshot for Beta Fighter may be stale relative to 2026-07-18; attempting live refresh first",
                "2026-07-08 01:00:02,000 [INFO] src.bot: Built features",
                "  Alpha Fighter vs Beta Fighter:",
                "    Model:      Alpha Fighter 60.0% | Beta Fighter 40.0%",
                "2026-07-08 02:00:00,000 [INFO] src.data.fighter_lookup: Processed live snapshot for Gamma Fighter may be stale relative to 2026-07-18; attempting live refresh first",
                "2026-07-08 02:00:01,000 [INFO] src.data.fighter_lookup: Processed live snapshot for Delta Fighter may be stale relative to 2026-07-25; attempting live refresh first",
                "2026-07-08 02:00:02,000 [INFO] src.bot: Built features",
                "  Gamma Fighter vs Delta Fighter:",
                "    Model:      Gamma Fighter 60.0% | Delta Fighter 40.0%",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = recover_prediction_rows_from_logs([log_path])

    alpha = next(row for row in rows if row["fighter_a"] == "Alpha Fighter")
    gamma = next(row for row in rows if row["fighter_a"] == "Gamma Fighter")
    assert alpha["card_date"] == "2026-07-18"
    assert "card_date" not in gamma


def test_reconcile_moves_unknown_row_and_collapses_scheduled_duplicate(tmp_path):
    history_path = tmp_path / "predictions_history.json"
    unknown = {
        "fighter_a": "Ian Garry",
        "fighter_b": "Test Opponent",
        "predicted_winner": "Ian Garry",
        "prob_a": 0.61,
        "prob_b": 0.39,
        "prediction_generated_at": "2026-06-30T12:00:00+00:00",
        "recovered_group_date": "2026-06-30",
    }
    scheduled = {
        "fighter_a": "Test Opponent",
        "fighter_b": "Ian Machado Garry",
        "predicted_winner": "Test Opponent",
        "prob_a": 0.62,
        "prob_b": 0.38,
        "card_date": "2026-08-15",
        "prediction_generated_at": "2026-08-14T12:00:00+00:00",
    }
    archive_prediction_payload(
        {"predictions": [unknown]},
        history_path,
        source="recovered_bot_log",
    )
    archive_prediction_payload(
        {"predictions": [scheduled]},
        history_path,
        source="live_cache_completed",
    )
    hints = {
        ("ianmachadogarry", "testopponent"): [
            {
                "card_date": "2026-08-15",
                "event_title": "UFC Test Card",
            }
        ]
    }

    result = reconcile_prediction_history_cards(history_path, hints)
    rows = load_prediction_history(history_path)["predictions"]

    assert result["rekeyed"] == 1
    assert result["scheduled"] == 1
    assert result["deduplicated"] == 1
    assert result["total"] == 1
    assert len(rows) == 1
    assert rows[0]["history_key"].startswith("2026-08-15::")
    assert rows[0]["card_date"] == "2026-08-15"
    assert rows[0]["event_title"] == "UFC Test Card"
    assert rows[0]["predicted_winner"] == "Test Opponent"

    repeated = reconcile_prediction_history_cards(history_path, hints)
    assert repeated["rekeyed"] == 0
    assert repeated["deduplicated"] == 0
    assert repeated["total"] == 1


def test_reconcile_discards_post_start_unknown_duplicate(tmp_path):
    history_path = tmp_path / "predictions_history.json"
    scheduled = {
        "fighter_a": "Alpha Fighter",
        "fighter_b": "Beta Fighter",
        "predicted_winner": "Alpha Fighter",
        "prob_a": 0.60,
        "prob_b": 0.40,
        "card_date": "2026-08-01",
        "event_date": "2026-08-01T20:00:00+00:00",
        "prediction_generated_at": "2026-08-01T19:00:00+00:00",
    }
    post_start = {
        "fighter_a": "Beta Fighter",
        "fighter_b": "Alpha Fighter",
        "predicted_winner": "Beta Fighter",
        "prob_a": 0.70,
        "prob_b": 0.30,
        "prediction_generated_at": "2026-08-01T21:00:00+00:00",
    }
    archive_prediction_payload({"predictions": [scheduled]}, history_path)
    archive_prediction_payload(
        {"predictions": [post_start]},
        history_path,
        source="recovered_bot_log",
    )

    result = reconcile_prediction_history_cards(
        history_path,
        {
            ("alphafighter", "betafighter"): [
                {"card_date": "2026-08-01", "event_title": "UFC Test Card"}
            ]
        },
    )
    rows = load_prediction_history(history_path)["predictions"]

    assert result["discarded_post_start"] == 1
    assert result["total"] == 1
    assert rows[0]["predicted_winner"] == "Alpha Fighter"


def test_completed_results_join_supplies_card_hints_with_fighter_aliases(tmp_path):
    (tmp_path / "ufc-event-dates.csv").write_text(
        "event_name,event_date,location\n"
        "ufc fight night: adesanya vs. pyfer,2026-03-28,Seattle\n",
        encoding="utf-8",
    )
    (tmp_path / "ufc-fight-results.csv").write_text(
        "EVENT,BOUT,OUTCOME\n"
        "UFC Fight Night: Adesanya vs. Pyfer ,"
        "Israel Adesanya vs. Joe Pyfer,L/W\n",
        encoding="utf-8",
    )

    hints = _load_completed_event_card_hints(tmp_path)

    pair = ("israeladesanya", "josephpyfer")
    assert hints[pair] == [
        {
            "card_date": "2026-03-28",
            "event_title": "UFC Fight Night: Adesanya vs. Pyfer",
            "authoritative": True,
            "observed_at": "2026-03-28",
        }
    ]


def test_completed_results_join_handles_accents_and_preserves_rematches(tmp_path):
    (tmp_path / "ufc-event-dates.csv").write_text(
        "event_name,event_date\n"
        "The Ultimate Fighter: Team Joanna vs. Team Claudia Finale,2025-01-01\n"
        "UFC Test Rematch,2026-08-01\n",
        encoding="utf-8",
    )
    (tmp_path / "ufc-fight-results.csv").write_text(
        "EVENT,BOUT\n"
        "The Ultimate Fighter: Team Joanna vs. Team Cláudia Finale ,"
        "Alpha Fighter vs. Beta Fighter\n"
        "UFC Test Rematch,"
        "Beta Fighter vs. Alpha Fighter\n",
        encoding="utf-8",
    )

    hints = _load_completed_event_card_hints(tmp_path)

    assert [hint["card_date"] for hint in hints[("alphafighter", "betafighter")]] == [
        "2025-01-01",
        "2026-08-01",
    ]


def test_card_hint_does_not_map_future_rematch_to_only_old_completed_bout():
    row = {
        "fighter_a": "Alexandre Pantoja",
        "fighter_b": "Joshua Van",
        "prediction_generated_at": "2026-08-09T12:00:00+00:00",
    }

    _apply_snapshot_card_hints(
        [row],
        {
            ("alexandrepantoja", "joshuavan"): [
                {
                    "card_date": "2025-12-06",
                    "event_title": "UFC 323",
                    "authoritative": True,
                }
            ]
        },
    )

    assert "card_date" not in row


def test_completed_hint_outranks_conflicting_stale_snapshot_date():
    row = {
        "fighter_a": "Alpha Fighter",
        "fighter_b": "Beta Fighter",
        "card_date": "2026-08-02",
        "prediction_generated_at": "2026-07-31T12:00:00+00:00",
    }
    hints = _merge_card_hint_maps(
        {
            ("alphafighter", "betafighter"): [
                {
                    "card_date": "2026-08-01",
                    "event_title": "Official completed card",
                    "authoritative": True,
                }
            ]
        },
        {
            ("alphafighter", "betafighter"): [
                {"card_date": "2026-08-02", "event_title": "Stale snapshot"}
            ]
        },
    )

    _apply_snapshot_card_hints([row], hints)

    assert row["card_date"] == "2026-08-01"
    assert row["event_title"] == "Official completed card"


def test_event_hint_uses_source_cache_timestamp_to_choose_rematch():
    row = {
        "fighter_a": "Alpha Fighter",
        "fighter_b": "Beta Fighter",
        "source_cache_timestamp": "2026-07-01T12:00:00+00:00",
    }
    _apply_event_hints_to_recovered_rows(
        [row],
        {
            ("alphafighter", "betafighter"): [
                {"event_date": "2025-01-01T20:00:00+00:00"},
                {
                    "event_date": "2026-08-01T20:00:00+00:00",
                    "card_date": "2026-08-01",
                },
            ]
        },
    )

    assert row["event_date"] == "2026-08-01T20:00:00+00:00"
    assert row["card_date"] == "2026-08-01"


def test_reconcile_discards_prediction_generated_after_date_only_card(tmp_path):
    history_path = tmp_path / "predictions_history.json"
    archive_prediction_payload(
        {
            "timestamp": "2026-08-02T12:00:00+00:00",
            "predictions": [
                {
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "predicted_winner": "Alpha Fighter",
                }
            ]
        },
        history_path,
        source="recovered_bot_log",
    )

    result = reconcile_prediction_history_cards(
        history_path,
        {
            ("alphafighter", "betafighter"): [
                {"card_date": "2026-08-01", "authoritative": True}
            ]
        },
    )

    assert result["discarded_post_start"] == 1
    assert load_prediction_history(history_path)["predictions"] == []


def test_reconcile_authoritative_card_hint_corrects_existing_utc_next_day_key(
    tmp_path,
):
    history_path = tmp_path / "predictions_history.json"
    archive_prediction_payload(
        {
            "predictions": [
                {
                    "fighter_a": "Jiri Prochazka",
                    "fighter_b": "Carlos Ulberg",
                    "predicted_winner": "Carlos Ulberg",
                    "prob_a": 0.40,
                    "prob_b": 0.60,
                    "card_date": "2026-04-12",
                    "event_date": "2026-04-12T04:00:00+00:00",
                    "prediction_generated_at": "2026-04-10T18:00:00+00:00",
                }
            ]
        },
        history_path,
        source="recovered_operator_decision",
    )

    result = reconcile_prediction_history_cards(
        history_path,
        {
            ("carlosulberg", "jiriprochazka"): [
                {
                    "card_date": "2026-04-11",
                    "event_title": "UFC 327: Prochazka vs. Ulberg",
                }
            ]
        },
    )
    row = load_prediction_history(history_path)["predictions"][0]

    assert result["rekeyed"] == 1
    assert row["card_date"] == "2026-04-11"
    assert row["event_title"] == "UFC 327: Prochazka vs. Ulberg"
    assert row["history_key"].startswith("2026-04-11::")


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


def test_historical_bet_ledger_recovers_dated_model_pick_not_value_side(tmp_path):
    ledger_path = tmp_path / "bet_ledger_single.json"
    ledger_path.write_text(
        json.dumps(
            {
                "bets": [
                    {
                        "fighter": "Value Underdog",
                        "opponent": "Model Favorite",
                        "side": "a",
                        "model_prob": 0.42,
                        "market_prob": 0.30,
                        "placed_at": "2026-03-21T19:47:53+00:00",
                        "event_date": "2026-05-17T00:00:00+00:00",
                    },
                    {
                        "fighter": "Placeholder Fighter",
                        "opponent": "Placeholder Opponent",
                        "side": "a",
                        "model_prob": 0.0,
                        "event_date": "2026-05-17T00:00:00+00:00",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = recover_prediction_rows_from_bet_ledgers([ledger_path])

    assert len(rows) == 1
    row = rows[0]
    assert row["fighter_a"] == "Value Underdog"
    assert row["fighter_b"] == "Model Favorite"
    assert row["predicted_side"] == "b"
    assert row["predicted_winner"] == "Model Favorite"
    assert row["prob_a"] == pytest.approx(0.42)
    assert row["prob_b"] == pytest.approx(0.58)
    assert row["recovery_provenance"].endswith("bet_ledger_single.json")


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


def test_operator_recovery_treats_date_only_event_as_end_of_card_day(tmp_path):
    decisions_path = tmp_path / "decision_log.jsonl"
    decisions_path.write_text(
        json.dumps(
            {
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "bet_on": "Alpha Fighter",
                "bet_side": "a",
                "model_prob": 0.60,
                "event_date": "2026-08-01",
                "timestamp": "2026-08-01T18:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = recover_prediction_rows_from_operator_decisions([decisions_path])

    assert len(rows) == 1
    assert rows[0]["predicted_winner"] == "Alpha Fighter"


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
                "event_url": "https://www.ufc.com/event/ufc-test-august-01-2026",
                "fights": [
                    {"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter"},
                    {
                        "fighter_a": "Removed Fighter",
                        "fighter_b": "Original Opponent",
                    },
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
                "2026-07-15 12:00:00,000 [INFO] src.bot: Starting prediction pass",
                "  Removed Fighter vs Original Opponent:",
                "    Model:      Removed Fighter 57.0% | Original Opponent 43.0%",
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
    assert summary["recovered_from_logs"] == 2
    assert history["prediction_count"] == 2
    row = next(
        row for row in history["predictions"] if row["fighter_a"] == "Alpha Fighter"
    )
    assert row["predicted_winner"] == "Beta Fighter"
    assert row["prediction_generated_at"] == "2026-08-01 19:59:00,000"
    assert row["card_date"] == "2026-08-01"
    assert row["event_title"] == "UFC Test Card"
    assert row["recovered_research_summary"] == decision["research_summary"]
    removed = next(
        row for row in history["predictions"] if row["fighter_a"] == "Removed Fighter"
    )
    assert removed["card_date"] == "2026-08-01"
    assert removed["event_title"] == "UFC Test Card"


def test_initialize_maps_undated_operator_pick_from_completed_event_csv(tmp_path):
    logs_dir = tmp_path / "logs"
    operator_dir = logs_dir / "operator"
    raw_dir = tmp_path / "raw"
    operator_dir.mkdir(parents=True)
    raw_dir.mkdir()
    (operator_dir / "decision_log.jsonl").write_text(
        json.dumps(
            {
                "fighter_a": "Israel Adesanya",
                "fighter_b": "Joseph Pyfer",
                "bet_on": "Israel Adesanya",
                "bet_side": "a",
                "model_prob": 0.62,
                "timestamp": "2026-03-27T20:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "ufc-event-dates.csv").write_text(
        "event_name,event_date,location\n"
        "ufc fight night: adesanya vs. pyfer,2026-03-28,Seattle\n",
        encoding="utf-8",
    )
    (raw_dir / "ufc-fight-results.csv").write_text(
        "EVENT,BOUT,OUTCOME\n"
        "UFC Fight Night: Adesanya vs. Pyfer,"
        "Israel Adesanya vs. Joe Pyfer,L/W\n",
        encoding="utf-8",
    )

    summary = initialize_prediction_history(logs_dir, raw_data_dir=raw_dir)
    history = load_prediction_history(logs_dir / "predictions_history.json")

    assert summary["completed_event_card_hints"] == 1
    assert history["prediction_count"] == 1
    row = history["predictions"][0]
    assert row["card_date"] == "2026-03-28"
    assert row["event_title"] == "UFC Fight Night: Adesanya vs. Pyfer"
    assert not row["history_key"].startswith("unknown:")


def test_initialized_card_hints_canonicalize_later_stale_live_payload(tmp_path):
    logs_dir = tmp_path / "logs"
    snapshots_dir = tmp_path / "raw" / "snapshots"
    logs_dir.mkdir()
    snapshots_dir.mkdir(parents=True)
    history_path = logs_dir / "predictions_history.json"

    (snapshots_dir / "ufc-test-card.json").write_text(
        json.dumps(
            {
                "event": "UFC Test Card",
                "event_date": "August 16, 2026",
                "event_url": "https://www.ufc.com/event/ufc-test-august-15-2026",
                "timestamp": "2026-08-14T10:00:00+00:00",
                "fights": [
                    {
                        "fighter_a": "Alpha Fighter",
                        "fighter_b": "Beta Fighter",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (logs_dir / "predictions_cache.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-08-14T12:00:00+00:00",
                "predictions": [
                    {
                        "fighter_a": "Alpha Fighter",
                        "fighter_b": "Beta Fighter",
                        "predicted_winner": "Alpha Fighter",
                        "prob_a": 0.55,
                        "prob_b": 0.45,
                        "card_date": "2026-08-16",
                        "prediction_generated_at": "2026-08-14T12:00:00+00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    initialize_prediction_history(logs_dir, raw_data_dir=tmp_path / "raw")
    archive_prediction_payload(
        {
            "timestamp": "2026-08-14T18:00:00+00:00",
            "predictions": [
                {
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "predicted_winner": "Beta Fighter",
                    "prob_a": 0.40,
                    "prob_b": 0.60,
                    "card_date": "2026-08-16",
                    "prediction_generated_at": "2026-08-14T18:00:00+00:00",
                }
            ],
        },
        history_path,
        source="prediction_cache",
    )

    rows = load_prediction_history(history_path)["predictions"]

    assert len(rows) == 1
    assert rows[0]["card_date"] == "2026-08-15"
    assert rows[0]["history_key"].startswith("2026-08-15::")
    assert rows[0]["predicted_winner"] == "Beta Fighter"
    assert all(not row["history_key"].startswith("2026-08-16::") for row in rows)


def test_initialize_uses_dated_legacy_ledger_to_schedule_early_operator_pick(
    tmp_path,
):
    logs_dir = tmp_path / "logs"
    operator_dir = logs_dir / "operator"
    operator_dir.mkdir(parents=True)
    (logs_dir / "bet_ledger_single.json").write_text(
        json.dumps(
            {
                "bets": [
                    {
                        "fighter": "Alpha Fighter",
                        "opponent": "Beta Fighter",
                        "side": "a",
                        "model_prob": 0.61,
                        "placed_at": "2026-03-21T19:47:53+00:00",
                        "event_date": "2026-05-17T00:00:00+00:00",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    decision = {
        "fighter_a": "Alpha Fighter",
        "fighter_b": "Beta Fighter",
        "bet_on": "Alpha Fighter",
        "bet_side": "a",
        "model_prob": 0.64,
        "timestamp": "2026-03-28T06:00:00+00:00",
        "research_summary": {"matchup_analysis": "Later recovered research"},
    }
    cache_dated_decision = {
        "fighter_a": "Gamma Fighter",
        "fighter_b": "Delta Fighter",
        "bet_on": "Delta Fighter",
        "bet_side": "b",
        "model_prob": 0.58,
        "timestamp": "2026-03-27T06:00:00+00:00",
    }
    (operator_dir / "decision_log.jsonl").write_text(
        json.dumps(decision) + "\n" + json.dumps(cache_dated_decision) + "\n",
        encoding="utf-8",
    )
    (operator_dir / "gemini_pick_cache.json").write_text(
        json.dumps(
            {
                "2026-03-28|delta fighter|gamma fighter": {
                    "event_date": "2026-03-28T20:00:00+00:00",
                    "event_title": "UFC Seattle Test",
                    "cached_at": 1774600000,
                    "response": {"pick": "Ignored Gemini Pick"},
                }
            }
        ),
        encoding="utf-8",
    )

    summary = initialize_prediction_history(logs_dir)
    rows = load_prediction_history(logs_dir / "predictions_history.json")[
        "predictions"
    ]

    assert summary["recovered_from_historical_ledgers"] == 1
    assert summary["recovered_from_operator_decisions"] == 2
    assert summary["operator_cache_card_hints"] == 1
    assert len(rows) == 2
    alpha = next(row for row in rows if row["fighter_a"] == "Alpha Fighter")
    assert alpha["card_date"] == "2026-05-16"
    assert alpha["prediction_generated_at"] == decision["timestamp"]
    assert alpha["recovered_research_summary"] == decision["research_summary"]
    gamma = next(row for row in rows if row["fighter_a"] == "Gamma Fighter")
    assert gamma["card_date"] == "2026-03-28"
    assert gamma["event_title"] == "UFC Seattle Test"
    assert gamma["predicted_winner"] == "Delta Fighter"
