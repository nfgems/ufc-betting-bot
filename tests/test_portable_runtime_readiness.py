import pytest

from src.web import app as web_app
from src.web import serve as web_serve


def _status(refresh_component):
    return {
        "ready": True,
        "errors": [],
        "warnings": [],
        "trading_enabled": False,
        "effective_live_mode": "off",
        "components": {"ufc_refresh_loop": refresh_component},
    }


def test_completed_failed_continuity_is_not_ready(monkeypatch):
    monkeypatch.setattr(web_app, "_runtime_threads", {})
    monkeypatch.setattr(
        web_app,
        "get_runtime_status",
        lambda: _status(
            {
                "state": "degraded",
                "last_cycle_completed_at": "2026-08-20T12:00:00+00:00",
                "continuity_green": False,
                "published": False,
            }
        ),
    )
    result = web_app._runtime_status_with_liveness()
    assert result["ready"] is False
    assert "ufc_refresh_continuity_failed" in result["errors"]


def test_delayed_first_attempt_remains_exempt(monkeypatch):
    monkeypatch.setattr(web_app, "_runtime_threads", {})
    monkeypatch.setattr(
        web_app,
        "get_runtime_status",
        lambda: _status(
            {
                "state": "starting",
                "continuity_green": False,
                "published": False,
            }
        ),
    )
    result = web_app._runtime_status_with_liveness()
    assert result["ready"] is True
    assert "ufc_refresh_continuity_failed" not in result["errors"]


def test_completed_green_but_unpublished_refresh_is_not_ready(monkeypatch):
    monkeypatch.setattr(web_app, "_runtime_threads", {})
    monkeypatch.setattr(
        web_app,
        "get_runtime_status",
        lambda: _status(
            {
                "state": "degraded",
                "last_cycle_completed_at": "2026-08-20T12:00:00+00:00",
                "continuity_green": True,
                "published": False,
            }
        ),
    )

    result = web_app._runtime_status_with_liveness()

    assert result["ready"] is False
    assert "ufc_refresh_publication_failed" in result["errors"]


def test_hosted_refresh_cycle_opts_into_changed_profile_refresh(monkeypatch):
    import scripts.run_scheduled_ufc_refresh as scheduled

    calls = []
    monkeypatch.setattr(
        scheduled,
        "run_scheduled_refresh",
        lambda **kwargs: calls.append(kwargs) or {"continuity_green": True, "published": True},
    )
    assert web_serve._run_ufc_refresh_cycle(limit_fighters=7)["published"] is True
    assert calls == [{"limit_fighters": 7, "refresh_existing_profiles": True}]


def test_collection_failures_are_operator_visible():
    alerts = web_serve._ufc_refresh_operational_alerts(
        {
            "outcome_reasons": ["ufcstats_partial_fight_observations"],
            "ufcstats_backfill": {
                "scraped_profile_scrape_failures": 2,
                "fighter_fight_list_scrape_failures": 1,
                "fight_detail_status_counts": {"partial": 1, "failed": 1},
            },
            "profile_supplement_refresh": {
                "action": "completed",
                "source_error_count": 1,
                "source_errors": {"sherdog": 1},
            },
        }
    )
    assert any("continuity gate failed" in alert for alert in alerts)
    assert any("profile collection failed" in alert for alert in alerts)
    assert any("partial observations" in alert for alert in alerts)
    assert any("source error" in alert for alert in alerts)


@pytest.mark.parametrize(
    ("summary", "message_fragment"),
    [
        (
            {
                "continuity_green": False,
                "published": False,
                "outcome_reasons": ["partial observation"],
                "roster_sync": {"rows": 1, "source": "live", "sync_complete": True},
                "ufcstats_backfill": {},
                "rebuild": {"outputs": [{"fight_rows": 1}]},
            },
            "failed continuity",
        ),
        (
            {
                "continuity_green": True,
                "published": False,
                "outcome_reasons": [],
                "roster_sync": {"rows": 1, "source": "live", "sync_complete": True},
                "ufcstats_backfill": {},
                "rebuild": {"outputs": [{"fight_rows": 1}]},
            },
            "without publication",
        ),
    ],
)
def test_nonpublished_attempt_does_not_advance_last_success(
    monkeypatch,
    summary,
    message_fragment,
):
    updates = []
    prior_success = "2026-08-19T12:00:00+00:00"
    monkeypatch.setattr(
        web_app,
        "get_runtime_status",
        lambda: {
            "components": {"ufc_refresh_loop": {"last_successful_refresh_at": prior_success}}
        },
    )
    monkeypatch.setattr(web_app, "set_runtime_status", lambda _status: None)
    monkeypatch.setattr(
        web_app,
        "update_runtime_component",
        lambda component, state, message="", **metadata: updates.append(
            (component, state, message, metadata)
        ),
    )
    monkeypatch.setattr(web_serve, "_run_ufc_refresh_cycle", lambda **_kwargs: summary)
    monkeypatch.setattr(
        web_serve,
        "_wait_for_next_ufc_refresh",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("stop refresh loop")),
    )

    with pytest.raises(RuntimeError, match="stop refresh loop"):
        web_serve.run_background_ufc_refresh_loop(
            interval_hours=24.0,
            initial_delay_seconds=0.0,
        )

    _component, state, message, metadata = updates[-1]
    assert state == "degraded"
    assert message_fragment in message
    assert metadata["last_successful_refresh_at"] == prior_success
