import pytest

from src.web import app as web_app
from src.web import serve as web_serve


def test_ufc_refresh_env_parsing(monkeypatch):
    monkeypatch.setenv("UFC_REFRESH_ENABLED", "yes")
    monkeypatch.setenv("UFC_REFRESH_INTERVAL_HOURS", "0")
    monkeypatch.setenv("UFC_REFRESH_INITIAL_DELAY_MINUTES", "bad")
    monkeypatch.setenv("UFC_REFRESH_LIMIT_FIGHTERS", "7")

    assert web_serve._ufc_refresh_enabled() is True
    assert web_serve._ufc_refresh_interval_hours() == pytest.approx(1.0)
    assert web_serve._ufc_refresh_initial_delay_seconds() == pytest.approx(1800.0)
    assert web_serve._ufc_refresh_limit_fighters() == 7


def test_ufc_refresh_interval_defaults_to_daily(monkeypatch):
    monkeypatch.delenv("UFC_REFRESH_INTERVAL_HOURS", raising=False)

    assert web_serve._ufc_refresh_interval_hours() == pytest.approx(24.0)


def test_ufc_refresh_interval_caps_weekly_cadence(monkeypatch, caplog):
    monkeypatch.setenv("UFC_REFRESH_INTERVAL_HOURS", "168")

    with caplog.at_level("WARNING"):
        interval_hours = web_serve._ufc_refresh_interval_hours()

    assert interval_hours == pytest.approx(24.0)
    assert "freshness-safe max" in caplog.text


def test_ufc_refresh_operational_alerts_ignore_expected_identity_audit_actions():
    alerts = web_serve._ufc_refresh_operational_alerts(
        {
            "roster_sync": {
                "identity_audit_rows": 78,
                "identity_audit_action_counts": {
                    "excluded_test_profile": 1,
                    "excluded_inactive_profile_status": 21,
                    "quarantined_untrusted_slug_alias": 30,
                    "quarantined_unknown_profile_status": 26,
                },
            },
            "row_drop_guard": {
                "violations": [
                    {
                        "artifact": "ufc_active_roster_official",
                        "pre_rows": 935,
                        "post_rows": 934,
                        "rows_lost": 1,
                    }
                ]
            },
        }
    )

    assert alerts == []


def test_ufc_refresh_operational_notes_report_inactive_profile_exclusions():
    notes = web_serve._ufc_refresh_operational_notes(
        {
            "roster_sync": {
                "identity_audit_rows": 2178,
                "identity_audit_action_counts": {
                    "excluded_inactive_profile_status": 2178,
                },
            }
        }
    )

    assert notes == [
        "excluded 2178 inactive/non-fighting UFC profile row(s) from active roster sync"
    ]


def test_ufc_refresh_operational_notes_report_unknown_status_quarantines():
    notes = web_serve._ufc_refresh_operational_notes(
        {
            "roster_sync": {
                "identity_audit_rows": 26,
                "identity_audit_action_counts": {
                    "quarantined_unknown_profile_status": 26,
                },
            }
        }
    )

    assert notes == [
        "quarantined 26 UFC profile row(s) with unverified current status "
        "from active-roster coverage"
    ]


def test_ufc_refresh_operational_alerts_keep_true_identity_quarantine():
    alerts = web_serve._ufc_refresh_operational_alerts(
        {
            "roster_sync": {
                "identity_audit_rows": 2,
                "identity_audit_action_counts": {
                    "quarantined_untrusted_url_identity": 1,
                    "quarantined_untrusted_slug_alias": 1,
                },
            }
        }
    )

    assert alerts == [
        "official UFC roster identity audit flagged 2 row(s): "
        "1 URL identity quarantined, 1 slug aliases suppressed"
    ]


def test_ufc_refresh_operational_notes_report_retained_missing_live_roster_rows():
    notes = web_serve._ufc_refresh_operational_notes(
        {
            "roster_sync": {
                "rows": 900,
                "retained_missing_live_rows": 2,
                "retained_missing_live_fighters": [
                    {
                        "official_name": "Omitted Fighter",
                        "official_athlete_url": "https://www.ufc.com/athlete/omitted-fighter",
                    },
                    {
                        "official_name": "Second Fighter",
                        "official_athlete_url": "https://www.ufc.com/athlete/second-fighter",
                    },
                ],
            }
        }
    )

    assert notes == [
        "official UFC roster live sync omitted 2 previously tracked row(s); "
        "retained cached rows: Omitted Fighter, Second Fighter"
    ]


def test_ufc_refresh_operational_notes_report_discarded_suspicious_cached_rows():
    notes = web_serve._ufc_refresh_operational_notes(
        {
            "roster_sync": {
                "rows": 958,
                "discarded_suspicious_cached_rows": 2177,
            }
        }
    )

    assert notes == [
        "discarded 2177 stale cached UFC roster row(s) after live active-roster sync"
    ]


def test_ufc_refresh_operational_alerts_ignore_intentional_suspicious_cached_roster_drop():
    alerts = web_serve._ufc_refresh_operational_alerts(
        {
            "roster_sync": {
                "rows": 958,
                "discarded_suspicious_cached_rows": 2177,
            },
            "row_drop_guard": {
                "violations": [
                    {
                        "artifact": "ufc_active_roster_official",
                        "pre_rows": 3135,
                        "post_rows": 958,
                        "rows_lost": 2177,
                    }
                ]
            },
        }
    )

    assert alerts == []


def test_ufc_refresh_operational_alerts_ignore_normal_retained_missing_live_roster_rows():
    alerts = web_serve._ufc_refresh_operational_alerts(
        {
            "roster_sync": {
                "rows": 900,
                "retained_missing_live_rows": 2,
                "retained_missing_live_fighters": [
                    {
                        "official_name": "Omitted Fighter",
                        "official_athlete_url": "https://www.ufc.com/athlete/omitted-fighter",
                    },
                    {
                        "official_name": "Second Fighter",
                        "official_athlete_url": "https://www.ufc.com/athlete/second-fighter",
                    },
                ],
            }
        }
    )

    assert alerts == []


def test_ufc_refresh_operational_alerts_report_incomplete_live_roster_scan():
    alerts = web_serve._ufc_refresh_operational_alerts(
        {
            "roster_sync": {
                "source": "live",
                "sync_complete": False,
                "sync_completeness_reason": "card_parse_coverage_incomplete",
            }
        }
    )

    assert alerts == [
        "official UFC roster live scan was incomplete: card_parse_coverage_incomplete"
    ]


def test_ufc_refresh_operational_alerts_escalate_large_retained_missing_live_roster_rows(monkeypatch):
    monkeypatch.setenv("UFC_REFRESH_MAX_RETAINED_MISSING_LIVE_ROWS", "1")
    alerts = web_serve._ufc_refresh_operational_alerts(
        {
            "roster_sync": {
                "rows": 900,
                "retained_missing_live_rows": 2,
                "retained_missing_live_fighters": [
                    {
                        "official_name": "Omitted Fighter",
                        "official_athlete_url": "https://www.ufc.com/athlete/omitted-fighter",
                    },
                    {
                        "official_name": "Second Fighter",
                        "official_athlete_url": "https://www.ufc.com/athlete/second-fighter",
                    },
                ],
            }
        }
    )

    assert alerts == [
        "official UFC roster live sync omitted 2 previously tracked row(s); "
        "retained cached rows: Omitted Fighter, Second Fighter "
        "(exceeds row threshold 1)"
    ]


def test_run_background_ufc_refresh_loop_reports_success(monkeypatch, caplog):
    updates: list[tuple[str, str, str, dict]] = []

    def fake_update_runtime_component(component, state, message="", **metadata):
        updates.append((component, state, message, metadata))

    monkeypatch.setattr(web_app, "update_runtime_component", fake_update_runtime_component)
    monkeypatch.setattr(
        web_serve,
        "_run_ufc_refresh_cycle",
        lambda **_kwargs: {
            "roster_sync": {"rows": 11},
            "ufcstats_backfill": {"new_result_rows": 1, "new_stat_rows": 2},
            "rebuild": {"outputs": [{"fight_rows": 123}]},
        },
    )

    def fake_sleep(_seconds):
        raise RuntimeError("stop refresh loop")

    monkeypatch.setattr(web_serve.time, "sleep", fake_sleep)

    with caplog.at_level("INFO"):
        with pytest.raises(RuntimeError, match="stop refresh loop"):
            web_serve.run_background_ufc_refresh_loop(
                interval_hours=24.0,
                initial_delay_seconds=0.0,
                limit_fighters=5,
            )

    assert updates[0][0] == "ufc_refresh_loop"
    assert updates[0][1] == "running"
    assert "active" in updates[0][2].lower()

    final_component, final_state, final_message, final_metadata = updates[-1]
    assert final_component == "ufc_refresh_loop"
    assert final_state == "running"
    assert "completed" in final_message.lower()
    assert final_metadata["fight_rows"] == 123
    completed_record = next(
        record
        for record in caplog.records
        if record.getMessage().startswith("Scheduled UFC refresh completed:")
    )
    assert completed_record.alert_recovered_incident_keys == [
        "scheduled-ufc-refresh:degraded"
    ]


def test_run_background_ufc_refresh_loop_reports_retained_roster_rows_as_notes(monkeypatch):
    updates: list[tuple[str, str, str, dict]] = []

    def fake_update_runtime_component(component, state, message="", **metadata):
        updates.append((component, state, message, metadata))

    monkeypatch.setattr(web_app, "update_runtime_component", fake_update_runtime_component)
    monkeypatch.setattr(
        web_serve,
        "_run_ufc_refresh_cycle",
        lambda **_kwargs: {
            "roster_sync": {
                "rows": 900,
                "retained_missing_live_rows": 27,
                "retained_missing_live_fighters": [
                    {
                        "official_name": "Former Fighter",
                        "official_athlete_url": "https://www.ufc.com/athlete/former-fighter",
                    }
                ],
            },
            "ufcstats_backfill": {"new_result_rows": 0, "new_stat_rows": 0},
            "rebuild": {"outputs": [{"fight_rows": 123}]},
        },
    )

    def fake_sleep(_seconds):
        raise RuntimeError("stop refresh loop")

    monkeypatch.setattr(web_serve.time, "sleep", fake_sleep)

    with pytest.raises(RuntimeError, match="stop refresh loop"):
        web_serve.run_background_ufc_refresh_loop(
            interval_hours=24.0,
            initial_delay_seconds=0.0,
            limit_fighters=None,
        )

    final_component, final_state, _final_message, final_metadata = updates[-1]
    assert final_component == "ufc_refresh_loop"
    assert final_state == "running"
    assert final_metadata["coverage_alerts"] == []
    assert "official UFC roster live sync omitted 27" in final_metadata["coverage_notes"][0]


def test_run_background_ufc_refresh_loop_reports_failure_immediately(
    monkeypatch,
    caplog,
):
    updates: list[tuple[str, str, str, dict]] = []

    def fake_update_runtime_component(component, state, message="", **metadata):
        updates.append((component, state, message, metadata))

    monkeypatch.setattr(web_app, "update_runtime_component", fake_update_runtime_component)

    def fail_refresh(**_kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(web_serve, "_run_ufc_refresh_cycle", fail_refresh)

    def fake_sleep(_seconds):
        raise RuntimeError("stop refresh loop")

    monkeypatch.setattr(web_serve.time, "sleep", fake_sleep)

    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="stop refresh loop"):
            web_serve.run_background_ufc_refresh_loop(
                interval_hours=24.0,
                initial_delay_seconds=0.0,
                limit_fighters=None,
            )

    final_component, final_state, final_message, final_metadata = updates[-1]
    assert final_component == "ufc_refresh_loop"
    assert final_state == "degraded"
    assert "failed" in final_message.lower()
    assert "refresh failure" in final_metadata["coverage_alerts"][0]
    failed_record = next(
        record
        for record in caplog.records
        if record.getMessage().startswith("Scheduled UFC refresh failed:")
    )
    assert failed_record.alert_incident_key == "scheduled-ufc-refresh:degraded"


def test_run_background_ufc_refresh_loop_reports_initial_delay_metadata(monkeypatch):
    updates: list[tuple[str, str, str, dict]] = []

    def fake_update_runtime_component(component, state, message="", **metadata):
        updates.append((component, state, message, metadata))

    monkeypatch.setattr(web_app, "update_runtime_component", fake_update_runtime_component)

    def fake_sleep(_seconds):
        raise RuntimeError("stop refresh loop")

    monkeypatch.setattr(web_serve.time, "sleep", fake_sleep)

    with pytest.raises(RuntimeError, match="stop refresh loop"):
        web_serve.run_background_ufc_refresh_loop(
            interval_hours=24.0,
            initial_delay_seconds=1800.0,
            limit_fighters=None,
        )

    component, state, message, metadata = updates[0]
    assert component == "ufc_refresh_loop"
    assert state == "starting"
    assert "waiting for first run" in message
    assert metadata["next_planned_refresh_at"]
    assert metadata["last_successful_refresh_at"] is None
    assert metadata["last_error"] is None


def test_run_background_ufc_refresh_loop_reports_configured_coverage_drop(monkeypatch):
    updates: list[tuple[str, str, str, dict]] = []

    def fake_update_runtime_component(component, state, message="", **metadata):
        updates.append((component, state, message, metadata))

    monkeypatch.setattr(web_app, "update_runtime_component", fake_update_runtime_component)
    monkeypatch.setenv("UFC_REFRESH_MIN_NEW_FIGHTER_STANCE_PCT", "70")
    monkeypatch.setattr(
        web_serve,
        "_run_ufc_refresh_cycle",
        lambda **_kwargs: {
            "roster_sync": {"rows": 11},
            "ufcstats_backfill": {"new_result_rows": 0, "new_stat_rows": 0},
            "rebuild": {"outputs": [{"fight_rows": 123}]},
            "profile_audit": {
                "active_roster_rows": 11,
                "overall_summary": {
                    "reach_present": {"pct": 95.0},
                    "stance_present": {"pct": 89.0},
                    "full_physical_bundle_present": {"pct": 88.0},
                },
                "split_summary_official_name": {
                    "newly_added_active_roster": {
                        "reach_present": {"pct": 86.0},
                        "stance_present": {"pct": 69.0},
                        "full_physical_bundle_present": {"pct": 65.0},
                    }
                },
            },
        },
    )

    def fake_sleep(_seconds):
        raise RuntimeError("stop refresh loop")

    monkeypatch.setattr(web_serve.time, "sleep", fake_sleep)

    with pytest.raises(RuntimeError, match="stop refresh loop"):
        web_serve.run_background_ufc_refresh_loop(
            interval_hours=24.0,
            initial_delay_seconds=0.0,
            limit_fighters=None,
        )

    final_component, final_state, _final_message, final_metadata = updates[-1]
    assert final_component == "ufc_refresh_loop"
    assert final_state == "degraded"
    assert "stance coverage" in final_metadata["coverage_alerts"][0]


def test_run_background_ufc_refresh_loop_marks_cached_roster_fallback_degraded(monkeypatch):
    updates: list[tuple[str, str, str, dict]] = []

    def fake_update_runtime_component(component, state, message="", **metadata):
        updates.append((component, state, message, metadata))

    monkeypatch.setattr(web_app, "update_runtime_component", fake_update_runtime_component)
    monkeypatch.setattr(
        web_serve,
        "_run_ufc_refresh_cycle",
        lambda **_kwargs: {
            "roster_sync": {
                "rows": 11,
                "source": "cached",
                "used_cached_fallback": True,
                "sync_error": "HTTPSConnectionPool(host='www.ufc.com', port=443): Read timed out. (read timeout=30)",
                "cached_snapshot_mtime_utc": "2026-03-28T20:00:00+00:00",
            },
            "ufcstats_backfill": {"new_result_rows": 0, "new_stat_rows": 0},
            "rebuild": {"outputs": [{"fight_rows": 123}]},
        },
    )

    def fake_sleep(_seconds):
        raise RuntimeError("stop refresh loop")

    monkeypatch.setattr(web_serve.time, "sleep", fake_sleep)

    with pytest.raises(RuntimeError, match="stop refresh loop"):
        web_serve.run_background_ufc_refresh_loop(
            interval_hours=24.0,
            initial_delay_seconds=0.0,
            limit_fighters=None,
        )

    final_component, final_state, final_message, final_metadata = updates[-1]
    assert final_component == "ufc_refresh_loop"
    assert final_state == "degraded"
    assert "completed" in final_message.lower()
    assert "cached roster snapshot" in final_metadata["coverage_alerts"][0]
    assert "Read timed out" in final_metadata["coverage_alerts"][0]


def test_run_background_ufc_refresh_loop_prefers_alert_summary_for_new_fighter_floor(monkeypatch):
    updates: list[tuple[str, str, str, dict]] = []

    def fake_update_runtime_component(component, state, message="", **metadata):
        updates.append((component, state, message, metadata))

    monkeypatch.setattr(web_app, "update_runtime_component", fake_update_runtime_component)
    monkeypatch.setenv("UFC_REFRESH_MIN_NEW_FIGHTER_REACH_PCT", "84")
    monkeypatch.setattr(
        web_serve,
        "_run_ufc_refresh_cycle",
        lambda **_kwargs: {
            "roster_sync": {"rows": 11},
            "ufcstats_backfill": {"new_result_rows": 0, "new_stat_rows": 0},
            "rebuild": {"outputs": [{"fight_rows": 123}]},
            "profile_audit": {
                "active_roster_rows": 11,
                "overall_summary": {
                    "reach_present": {"pct": 95.0},
                    "stance_present": {"pct": 89.0},
                    "full_physical_bundle_present": {"pct": 88.0},
                },
                "split_summary_official_name": {
                    "newly_added_active_roster": {
                        "reach_present": {"pct": 68.0},
                        "stance_present": {"pct": 20.0},
                        "full_physical_bundle_present": {"pct": 20.0},
                    }
                },
            },
            "profile_audit_alert_summary": {
                "new_fighter_grace_days": 7,
                "newly_added_active_roster": {
                    "rows_total": 77,
                    "rows_alert_eligible": 11,
                    "rows_in_grace": 66,
                },
                "split_summary_official_name": {
                    "newly_added_active_roster": {
                        "reach_present": {"pct": 90.91},
                        "stance_present": {"pct": 54.55},
                        "full_physical_bundle_present": {"pct": 54.55},
                    }
                },
            },
        },
    )

    def fake_sleep(_seconds):
        raise RuntimeError("stop refresh loop")

    monkeypatch.setattr(web_serve.time, "sleep", fake_sleep)

    with pytest.raises(RuntimeError, match="stop refresh loop"):
        web_serve.run_background_ufc_refresh_loop(
            interval_hours=24.0,
            initial_delay_seconds=0.0,
            limit_fighters=None,
        )

    final_component, final_state, _final_message, final_metadata = updates[-1]
    assert final_component == "ufc_refresh_loop"
    assert final_state == "running"
    assert final_metadata["coverage_alerts"] == []
    assert final_metadata["coverage_snapshot"]["new_fighter_reach_pct"] == pytest.approx(90.91)
    assert final_metadata["coverage_snapshot"]["new_fighter_rows_in_grace"] == 66


def test_run_background_ufc_refresh_loop_skips_source_limited_new_fighter_alerts(monkeypatch):
    updates: list[tuple[str, str, str, dict]] = []

    def fake_update_runtime_component(component, state, message="", **metadata):
        updates.append((component, state, message, metadata))

    monkeypatch.setattr(web_app, "update_runtime_component", fake_update_runtime_component)
    monkeypatch.setenv("UFC_REFRESH_MIN_NEW_FIGHTER_STANCE_PCT", "65")
    monkeypatch.setenv("UFC_REFRESH_MIN_NEW_FIGHTER_FULL_PHYSICAL_PCT", "60")
    monkeypatch.setattr(
        web_serve,
        "_run_ufc_refresh_cycle",
        lambda **_kwargs: {
            "roster_sync": {"rows": 11},
            "ufcstats_backfill": {"new_result_rows": 0, "new_stat_rows": 0},
            "rebuild": {"outputs": [{"fight_rows": 123}]},
            "profile_audit": {
                "active_roster_rows": 11,
                "overall_summary": {
                    "reach_present": {"pct": 95.0},
                    "stance_present": {"pct": 89.0},
                    "full_physical_bundle_present": {"pct": 88.0},
                },
                "split_summary_official_name": {
                    "newly_added_active_roster": {
                        "reach_present": {"pct": 68.0},
                        "stance_present": {"pct": 20.0},
                        "full_physical_bundle_present": {"pct": 20.0},
                    }
                },
            },
            "profile_audit_alert_summary": {
                "new_fighter_grace_days": 7,
                "newly_added_active_roster": {
                    "rows_total": 77,
                    "rows_alert_eligible": 12,
                    "rows_in_grace": 65,
                },
                "source_limited_missing_fields": {
                    "reach": {
                        "rows_missing": 0,
                        "rows_source_limited": 0,
                        "source_limited_only": False,
                    },
                    "stance": {
                        "rows_missing": 5,
                        "rows_source_limited": 5,
                        "source_limited_only": True,
                    },
                    "full_physical_bundle": {
                        "rows_missing": 5,
                        "rows_source_limited": 5,
                        "source_limited_only": True,
                    },
                },
                "split_summary_official_name": {
                    "newly_added_active_roster": {
                        "reach_present": {"pct": 91.67},
                        "stance_present": {"pct": 58.33},
                        "full_physical_bundle_present": {"pct": 58.33},
                    }
                },
            },
        },
    )

    def fake_sleep(_seconds):
        raise RuntimeError("stop refresh loop")

    monkeypatch.setattr(web_serve.time, "sleep", fake_sleep)

    with pytest.raises(RuntimeError, match="stop refresh loop"):
        web_serve.run_background_ufc_refresh_loop(
            interval_hours=24.0,
            initial_delay_seconds=0.0,
            limit_fighters=None,
        )

    final_component, final_state, _final_message, final_metadata = updates[-1]
    assert final_component == "ufc_refresh_loop"
    assert final_state == "running"
    assert final_metadata["coverage_alerts"] == []
    assert "source-limited" in final_metadata["coverage_notes"][0]
    assert final_metadata["coverage_snapshot"]["new_fighter_stance_pct"] == pytest.approx(58.33)
    assert final_metadata["coverage_snapshot"]["new_fighter_stance_source_limited_only"] is True


def test_run_background_ufc_refresh_loop_skips_coverage_alerts_for_partial_refresh(monkeypatch):
    updates: list[tuple[str, str, str, dict]] = []

    def fake_update_runtime_component(component, state, message="", **metadata):
        updates.append((component, state, message, metadata))

    monkeypatch.setattr(web_app, "update_runtime_component", fake_update_runtime_component)
    monkeypatch.setenv("UFC_REFRESH_MIN_NEW_FIGHTER_STANCE_PCT", "70")
    monkeypatch.setattr(
        web_serve,
        "_run_ufc_refresh_cycle",
        lambda **_kwargs: {
            "limit_fighters": 5,
            "partial_refresh": True,
            "roster_sync": {"rows": 11},
            "ufcstats_backfill": {"new_result_rows": 0, "new_stat_rows": 0},
            "rebuild": {"outputs": [{"fight_rows": 123}]},
            "profile_audit": {
                "active_roster_rows": 11,
                "split_summary_official_name": {
                    "newly_added_active_roster": {
                        "reach_present": {"pct": 59.74},
                        "stance_present": {"pct": 3.90},
                        "full_physical_bundle_present": {"pct": 3.90},
                    }
                },
            },
        },
    )

    def fake_sleep(_seconds):
        raise RuntimeError("stop refresh loop")

    monkeypatch.setattr(web_serve.time, "sleep", fake_sleep)

    with pytest.raises(RuntimeError, match="stop refresh loop"):
        web_serve.run_background_ufc_refresh_loop(
            interval_hours=24.0,
            initial_delay_seconds=0.0,
            limit_fighters=5,
        )

    final_component, final_state, _final_message, final_metadata = updates[-1]
    assert final_component == "ufc_refresh_loop"
    assert final_state == "running"
    assert final_metadata["coverage_alerts"] == []
    assert "partial" in final_metadata["coverage_skip_reason"]


def test_run_background_ufc_refresh_loop_refreshes_runtime_bundle_status(monkeypatch):
    updates: list[tuple[str, str, str, dict]] = []
    statuses: list[dict] = []

    monkeypatch.setattr(web_app, "update_runtime_component", lambda component, state, message="", **metadata: updates.append((component, state, message, metadata)))
    monkeypatch.setattr(web_app, "get_runtime_status", lambda: {"service": "ufc-betting-bot", "components": {}})
    monkeypatch.setattr(web_app, "set_runtime_status", lambda status: statuses.append(status))
    monkeypatch.setattr(
        web_serve,
        "_run_ufc_refresh_cycle",
        lambda **_kwargs: {
            "roster_sync": {"rows": 11},
            "ufcstats_backfill": {"new_result_rows": 1, "new_stat_rows": 2},
            "rebuild": {
                "outputs": [{"fight_rows": 123}],
                "production_bundle": {"bundle_id": "bundle-1", "model_spec_name": "prod_spec"},
            },
        },
    )

    def fake_sleep(_seconds):
        raise RuntimeError("stop refresh loop")

    monkeypatch.setattr(web_serve.time, "sleep", fake_sleep)

    with pytest.raises(RuntimeError, match="stop refresh loop"):
        web_serve.run_background_ufc_refresh_loop(
            interval_hours=24.0,
            initial_delay_seconds=0.0,
            limit_fighters=None,
        )

    assert statuses[-1]["production_bundle"]["bundle_id"] == "bundle-1"


def test_ufc_refresh_cycle_marker_tracks_running_state():
    web_serve._mark_ufc_refresh_cycle_finished()
    assert web_serve._ufc_refresh_cycle_in_progress() is False

    web_serve._mark_ufc_refresh_cycle_started()
    assert web_serve._ufc_refresh_cycle_in_progress() is True

    web_serve._mark_ufc_refresh_cycle_finished()
    assert web_serve._ufc_refresh_cycle_in_progress() is False
