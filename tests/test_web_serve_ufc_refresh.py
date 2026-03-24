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


def test_run_background_ufc_refresh_loop_reports_success(monkeypatch):
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


def test_run_background_ufc_refresh_loop_reports_failure_immediately(monkeypatch):
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
