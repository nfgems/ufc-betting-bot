import hashlib
import json
import logging
import os
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from src import bot
from src.data.name_utils import normalize_cross_source_name


def test_runtime_bundle_freshness_uses_official_card_date_for_utc_rollover():
    summary = {"processed_snapshot_max_event_date": "2026-05-30"}
    fights = [
        {
            "event_id": "evt-june-6",
            "commence_time": "2026-06-07T00:45:00Z",
            "fighter_a": "Belal Muhammad",
            "fighter_b": "Gabriel Bonfim",
        }
    ]
    live_contexts = [
        {
            "event_id": "evt-june-6",
            "event_date": "June 6, 2026",
            "fighter_a": "Belal Muhammad",
            "fighter_b": "Gabriel Bonfim",
        }
    ]

    reference_date = bot._runtime_bundle_live_reference_date(fights, live_contexts)

    assert reference_date == date(2026, 6, 6)
    assert bot._runtime_bundle_freshness_messages(
        summary,
        reference_date=reference_date,
    ) == []


def test_runtime_bundle_freshness_does_not_treat_active_rollover_card_as_missing():
    summary = {"processed_snapshot_max_event_date": "2026-06-27"}
    fights = [
        {
            "event_id": "evt-ufc-329",
            "commence_time": "2026-07-12T02:50:00Z",
            "fighter_a": "Benoit Saint-Denis",
            "fighter_b": "Paddy Pimblett",
        }
    ]
    recovered_completed_context = [
        {
            "event_date": "July 11, 2026",
            "fighter_a": "Benoit Saint Denis",
            "fighter_b": "Paddy Pimblett",
        }
    ]

    reference_date = bot._runtime_bundle_live_reference_date(fights, recovered_completed_context)
    completed_before_active = {
        event_date for event_date in {date(2026, 7, 11)} if event_date < reference_date
    }

    assert reference_date == date(2026, 7, 11)
    assert completed_before_active == set()
    assert bot._runtime_bundle_freshness_messages(
        summary,
        reference_date=reference_date,
        completed_event_dates=completed_before_active,
    ) == []


def test_runtime_bundle_freshness_still_blocks_distinct_card_before_rollover_card():
    summary = {"processed_snapshot_max_event_date": "2026-06-27"}

    messages = bot._runtime_bundle_freshness_messages(
        summary,
        reference_date=date(2026, 7, 11),
        completed_event_dates={date(2026, 7, 4)},
    )

    assert messages == [
        "processed snapshot max event date=2026-06-27 is missing completed UFC "
        "event date(s) 2026-07-04 before active UFC card date=2026-07-11"
    ]


def test_runtime_bundle_freshness_blocks_when_active_card_is_more_than_one_week_ahead():
    summary = {"processed_snapshot_max_event_date": "2026-05-30"}
    fights = [
        {
            "event_id": "evt-june-14",
            "commence_time": "2026-06-14T23:00:00Z",
            "fighter_a": "Alpha Fighter",
            "fighter_b": "Beta Fighter",
        }
    ]
    live_contexts = [
        {
            "event_id": "evt-june-14",
            "event_date": "June 14, 2026",
            "fighter_a": "Alpha Fighter",
            "fighter_b": "Beta Fighter",
        }
    ]

    reference_date = bot._runtime_bundle_live_reference_date(fights, live_contexts)
    messages = bot._runtime_bundle_freshness_messages(
        summary,
        reference_date=reference_date,
    )

    assert reference_date == date(2026, 6, 14)
    assert messages == [
        "processed snapshot max event date=2026-05-30 is 15 days old "
        "relative to active UFC card date=2026-06-14 (max 7 days)"
    ]


def test_runtime_bundle_freshness_blocks_missing_completed_intervening_card():
    summary = {"processed_snapshot_max_event_date": "2026-05-30"}

    messages = bot._runtime_bundle_freshness_messages(
        summary,
        reference_date=date(2026, 6, 14),
        completed_event_dates={date(2026, 6, 6)},
    )

    assert messages == [
        "processed snapshot max event date=2026-05-30 is missing completed UFC "
        "event date(s) 2026-06-06 before active UFC card date=2026-06-14"
    ]


def test_runtime_bundle_freshness_allows_previous_completed_card_even_after_eight_days():
    summary = {"processed_snapshot_max_event_date": "2026-06-06"}

    messages = bot._runtime_bundle_freshness_messages(
        summary,
        reference_date=date(2026, 6, 14),
        completed_event_dates={date(2026, 6, 6)},
    )

    assert messages == []


def test_runtime_bundle_freshness_allows_one_day_completed_card_source_offset():
    summary = {"processed_snapshot_max_event_date": "2026-06-06"}

    messages = bot._runtime_bundle_freshness_messages(
        summary,
        reference_date=date(2026, 6, 15),
        completed_event_dates={date(2026, 6, 7)},
    )

    assert messages == []


def test_runtime_bundle_freshness_still_blocks_completed_card_after_source_offset():
    summary = {"processed_snapshot_max_event_date": "2026-06-06"}

    messages = bot._runtime_bundle_freshness_messages(
        summary,
        reference_date=date(2026, 6, 15),
        completed_event_dates={date(2026, 6, 8)},
    )

    assert messages == [
        "processed snapshot max event date=2026-06-06 is missing completed UFC "
        "event date(s) 2026-06-08 before active UFC card date=2026-06-15"
    ]


@pytest.mark.parametrize(
    ("snapshot_event_date", "reference_date", "completed_event_date", "expected"),
    [
        ("2026-08-01", date(2026, 8, 15), date(2026, 8, 8), (date(2026, 8, 8),)),
        ("2026-08-08", date(2026, 8, 15), date(2026, 8, 8), ()),
        ("2026-08-01", date(2026, 8, 16), date(2026, 8, 9), (date(2026, 8, 9),)),
        # UFC.com/Odds API UTC rollover dates remain covered by the prior local
        # UFCStats card date; the wakeup uses the same rule as the guard.
        ("2026-08-08", date(2026, 8, 16), date(2026, 8, 9), ()),
    ],
)
def test_runtime_bundle_missing_completed_dates_matches_august_card_and_utc_rollover(
    snapshot_event_date,
    reference_date,
    completed_event_date,
    expected,
):
    assert bot._runtime_bundle_missing_completed_event_dates(
        {"processed_snapshot_max_event_date": snapshot_event_date},
        reference_date=reference_date,
        completed_event_dates={completed_event_date},
    ) == expected


def test_runtime_bundle_freshness_requests_refresh_before_preserving_strict_guard():
    callback_calls = []

    def request_refresh(**kwargs):
        callback_calls.append(kwargs)
        raise ValueError("wakeup transport failed")

    with pytest.raises(RuntimeError, match="missing completed UFC event date"):
        bot._enforce_runtime_bundle_freshness(
            {"processed_snapshot_max_event_date": "2026-08-01"},
            strict=True,
            reference_date=date(2026, 8, 15),
            completed_event_dates={date(2026, 8, 8)},
            missing_completed_event_callback=request_refresh,
        )

    assert callback_calls == [
        {
            "missing_event_dates": ("2026-08-08",),
            "reference_date": "2026-08-15",
        }
    ]


def test_runtime_bundle_freshness_warning_requests_refresh_outside_bet_window(caplog):
    callback_calls = []

    with caplog.at_level(logging.WARNING):
        bot._enforce_runtime_bundle_freshness(
            {"processed_snapshot_max_event_date": "2026-08-01"},
            strict=False,
            reference_date=date(2026, 8, 15),
            completed_event_dates={date(2026, 8, 8)},
            missing_completed_event_callback=lambda **kwargs: callback_calls.append(kwargs),
        )

    assert callback_calls == [
        {
            "missing_event_dates": ("2026-08-08",),
            "reference_date": "2026-08-15",
        }
    ]
    assert "Runtime bundle freshness guard warning" in caplog.text


def test_runtime_bundle_freshness_allows_long_gap_when_no_completed_card_was_missed():
    summary = {"processed_snapshot_max_event_date": "2026-06-01"}

    messages = bot._runtime_bundle_freshness_messages(
        summary,
        reference_date=date(2026, 6, 15),
        completed_event_dates={date(2026, 6, 1)},
    )

    assert messages == []


def test_runtime_completed_ufc_event_dates_before_uses_completed_ufc_com_events(monkeypatch):
    from src.data import live_monitor

    monkeypatch.setattr(
        live_monitor,
        "_scrape_ufc_com_events",
        lambda *, include_completed=False: [
            {"date": "June 6, 2026", "status": "completed"},
            {"date": "June 14, 2026", "status": "upcoming"},
            {"date": "May 30, 2026", "status": "completed"},
        ],
    )

    dates = bot._runtime_completed_ufc_event_dates_before(date(2026, 6, 14))

    assert dates == {date(2026, 5, 30), date(2026, 6, 6)}


def test_runtime_bundle_freshness_age_fallback_is_advisory_when_completed_dates_unavailable(caplog):
    # Sat Jun 6 card -> Sun Jun 14 card is an unavoidable 8-day gap on fresh data,
    # so a strict block here would halt live trading on a transient UFC.com outage.
    summary = {"processed_snapshot_max_event_date": "2026-06-06"}

    with caplog.at_level(logging.WARNING):
        bot._enforce_runtime_bundle_freshness(
            summary,
            strict=True,
            reference_date=date(2026, 6, 14),
            completed_event_dates=None,
        )

    assert "advisory only" in caplog.text
    assert "8 days old" in caplog.text


def test_runtime_bundle_freshness_still_blocks_missing_completed_card_when_strict():
    summary = {"processed_snapshot_max_event_date": "2026-05-30"}

    with pytest.raises(RuntimeError, match="missing completed UFC event date"):
        bot._enforce_runtime_bundle_freshness(
            summary,
            strict=True,
            reference_date=date(2026, 6, 14),
            completed_event_dates={date(2026, 6, 6)},
        )


def test_runtime_bundle_freshness_age_check_stays_strict_without_reference_date():
    summary = {"processed_snapshot_max_event_date": "2020-01-01"}

    with pytest.raises(RuntimeError, match="days old"):
        bot._enforce_runtime_bundle_freshness(summary, strict=True)


def test_completed_event_dates_cache_bridges_fetch_failure(monkeypatch):
    from src.data import live_monitor

    calls = {"count": 0}

    def _fake_scrape(*, include_completed=False):
        calls["count"] += 1
        if calls["count"] == 1:
            return [
                {"date": "June 6, 2026", "status": "completed"},
                {"date": "June 14, 2026", "status": "upcoming"},
            ]
        return None

    monkeypatch.setattr(live_monitor, "_scrape_ufc_com_events", _fake_scrape)
    monkeypatch.setattr(bot, "_LAST_GOOD_COMPLETED_UFC_EVENT_DATES", None)

    assert bot._runtime_completed_ufc_event_dates_before(date(2026, 6, 14)) == {date(2026, 6, 6)}
    # Second call: the scrape fails, but the cached copy keeps the event-aware check alive.
    assert bot._runtime_completed_ufc_event_dates_before(date(2026, 6, 14)) == {date(2026, 6, 6)}
    assert calls["count"] == 2


def test_completed_event_dates_cache_expires(monkeypatch):
    import time as time_module

    from src.data import live_monitor

    monkeypatch.setattr(
        live_monitor, "_scrape_ufc_com_events", lambda *, include_completed=False: None
    )
    monkeypatch.setattr(
        bot,
        "_LAST_GOOD_COMPLETED_UFC_EVENT_DATES",
        (
            time_module.monotonic() - bot._COMPLETED_EVENT_DATES_CACHE_TTL_SECONDS - 1.0,
            {date(2026, 6, 6)},
        ),
    )

    assert bot._runtime_completed_ufc_event_dates_before(date(2026, 6, 14)) is None


def test_completed_event_dates_zero_completed_is_treated_as_markup_change(monkeypatch, caplog):
    from src.data import live_monitor

    monkeypatch.setattr(
        live_monitor,
        "_scrape_ufc_com_events",
        lambda *, include_completed=False: [{"date": "June 14, 2026", "status": "upcoming"}],
    )
    monkeypatch.setattr(bot, "_LAST_GOOD_COMPLETED_UFC_EVENT_DATES", None)

    with caplog.at_level(logging.WARNING):
        assert bot._runtime_completed_ufc_event_dates_before(date(2026, 6, 14)) is None

    # An empty completed set must not be cached or returned: it would let the
    # event-aware check pass with nothing to verify.
    assert bot._LAST_GOOD_COMPLETED_UFC_EVENT_DATES is None
    assert "none marked completed" in caplog.text


def test_live_cycle_missing_context_degradation_flags_total_context_loss():
    reason = bot._live_cycle_missing_context_degradation(
        tradeable_fight_count=3,
        missing_context_fights=[
            {
                "fighter_a": f"Fighter A{index}",
                "fighter_b": f"Fighter B{index}",
                "commence_time": "2026-06-14T23:00:00Z",
            }
            for index in range(3)
        ],
        live_event_contexts=[],
    )

    assert reason is not None
    assert "halted" in reason


def test_live_cycle_missing_context_degradation_ignores_normal_skips(monkeypatch):
    monkeypatch.setattr(
        bot,
        "_load_local_ufc_roster_names",
        lambda: {normalize_cross_source_name("Some Roster Fighter")},
    )
    non_ufc_fights = [
        {
            "fighter_a": "Non Ufc Alpha",
            "fighter_b": "Non Ufc Beta",
            "commence_time": "2026-06-13T23:00:00Z",
        }
    ]
    other_card_contexts = [{"event_id": "evt-other", "commence_time": "2026-06-21T23:00:00Z"}]

    # Non-roster fights with contexts loaded are normal non-UFC skips.
    assert (
        bot._live_cycle_missing_context_degradation(
            tradeable_fight_count=1,
            missing_context_fights=non_ufc_fights,
            live_event_contexts=other_card_contexts,
        )
        is None
    )
    # Not all tradeable fights were skipped.
    assert (
        bot._live_cycle_missing_context_degradation(
            tradeable_fight_count=3,
            missing_context_fights=non_ufc_fights,
            live_event_contexts=[],
        )
        is None
    )
    # No tradeable fights at all.
    assert (
        bot._live_cycle_missing_context_degradation(
            tradeable_fight_count=0,
            missing_context_fights=[],
            live_event_contexts=[],
        )
        is None
    )


def test_live_cycle_missing_context_degradation_flags_partial_context_loss(monkeypatch):
    monkeypatch.setattr(
        bot,
        "_load_local_ufc_roster_names",
        lambda: {
            normalize_cross_source_name("Alpha Fighter"),
            normalize_cross_source_name("Beta Fighter"),
        },
    )

    reason = bot._live_cycle_missing_context_degradation(
        tradeable_fight_count=1,
        missing_context_fights=[
            {
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "commence_time": "2026-06-14T23:00:00Z",
            }
        ],
        # Other cards loaded, but the active card's date is absent.
        live_event_contexts=[{"event_id": "evt-other", "commence_time": "2026-06-21T23:00:00Z"}],
    )

    assert reason is not None
    assert "Alpha Fighter" in reason


def test_live_cycle_missing_context_degradation_tolerates_loaded_card_with_name_mismatch(monkeypatch):
    monkeypatch.setattr(
        bot,
        "_load_local_ufc_roster_names",
        lambda: {
            normalize_cross_source_name("Alpha Fighter"),
            normalize_cross_source_name("Beta Fighter"),
        },
    )

    # The card date did load (UTC rollover tolerance): the skip is a name-mismatch
    # problem, not a context outage.
    assert (
        bot._live_cycle_missing_context_degradation(
            tradeable_fight_count=1,
            missing_context_fights=[
                {
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "commence_time": "2026-06-15T00:45:00Z",
                }
            ],
            live_event_contexts=[
                {"event_id": "evt-active", "commence_time": "2026-06-14T23:00:00Z"}
            ],
        )
        is None
    )


def test_runtime_bundle_live_freshness_scope_is_warning_until_bet_window_opens():
    fights = [
        {
            "event_id": "evt-june-14",
            "commence_time": "2026-06-14T23:00:00Z",
            "fighter_a": "Alpha Fighter",
            "fighter_b": "Beta Fighter",
        }
    ]
    live_contexts = [
        {
            "event_id": "evt-june-14",
            "event_date": "June 14, 2026",
            "fighter_a": "Alpha Fighter",
            "fighter_b": "Beta Fighter",
        }
    ]

    early_reference_date, early_strict = bot._runtime_bundle_live_freshness_scope(
        fights,
        live_contexts,
        now=datetime(2026, 6, 7, 1, 33, tzinfo=timezone.utc),
    )
    in_window_reference_date, in_window_strict = bot._runtime_bundle_live_freshness_scope(
        fights,
        live_contexts,
        now=datetime(2026, 6, 13, 1, 33, tzinfo=timezone.utc),
    )

    assert early_reference_date == date(2026, 6, 14)
    assert early_strict is False
    assert in_window_reference_date == date(2026, 6, 14)
    assert in_window_strict is True


def test_prediction_event_context_snapshot_carries_official_card_date():
    fight = {
        "event_id": "evt-white-house",
        "commence_time": "2026-06-14T23:00:00Z",
        "fighter_a": "Alpha Fighter",
        "fighter_b": "Beta Fighter",
    }
    event_context = {
        "event_date": "June 14, 2026",
        "weight_class": "Light Heavyweight",
        "num_rounds": 3,
        "is_title_bout": False,
        "is_empty_arena": False,
    }

    snapshot = bot._prediction_event_context_snapshot(fight, event_context)

    assert snapshot["event_date"] == "June 14, 2026"
    assert snapshot["card_date"] == "2026-06-14"


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("2026-08-16T00:30:00+00:00", "2026-08-15"),
        ("2026-08-15T20:30:00-04:00", "2026-08-15"),
        ("2026-08-15", "2026-08-15"),
        ("August 15, 2026", "2026-08-15"),
    ],
)
def test_canonical_card_date_uses_event_timezone_without_shifting_advertised_dates(
    monkeypatch,
    raw_value,
    expected,
):
    monkeypatch.setenv("DASHBOARD_EVENT_TIMEZONE", "America/New_York")

    assert bot._canonical_card_date(raw_value) == expected


def test_prediction_event_context_snapshot_prefers_card_date_and_preserves_raw_event(
    monkeypatch,
):
    monkeypatch.setenv("DASHBOARD_EVENT_TIMEZONE", "America/New_York")
    raw_event_date = "2026-08-16T00:30:00+00:00"

    snapshot = bot._prediction_event_context_snapshot(
        {
            "event_id": "evt-august-15",
            "commence_time": raw_event_date,
            "fighter_a": "Alpha Fighter",
            "fighter_b": "Beta Fighter",
        },
        {
            "event_date": raw_event_date,
            "card_date": "August 15, 2026",
            "weight_class": "Lightweight",
            "num_rounds": 3,
            "is_title_bout": False,
            "is_empty_arena": False,
        },
    )

    assert snapshot["event_date"] == raw_event_date
    assert snapshot["card_date"] == "2026-08-15"


def test_resolve_live_event_context_prefers_existing_card_date(monkeypatch):
    monkeypatch.setenv("DASHBOARD_EVENT_TIMEZONE", "America/New_York")
    raw_event_date = "2026-08-16T00:30:00+00:00"

    event_context = bot._resolve_live_event_context(
        {
            "event_id": "evt-august-15",
            "commence_time": raw_event_date,
            "fighter_a": "Alpha Fighter",
            "fighter_b": "Beta Fighter",
        },
        [
            {
                "event_id": "evt-august-15",
                "event_date": raw_event_date,
                "card_date": "August 15, 2026",
                "commence_time": raw_event_date,
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "weight_class": "Lightweight",
                "num_rounds": 3,
                "is_title_bout": False,
                "is_empty_arena": False,
            }
        ],
        allow_off_card_history_fallback=False,
    )

    assert event_context is not None
    assert event_context["event_date"] == raw_event_date
    assert event_context["card_date"] == "2026-08-15"


def test_runtime_commence_date_remains_utc_for_operational_checks(monkeypatch):
    monkeypatch.setenv("DASHBOARD_EVENT_TIMEZONE", "America/New_York")

    assert bot._runtime_commence_date("2026-08-16T00:30:00+00:00") == date(
        2026,
        8,
        16,
    )
    assert bot._parse_runtime_event_date("2026-08-16T00:30:00+00:00") == date(
        2026,
        8,
        16,
    )


def test_resolve_live_event_context_returns_official_card_date():
    fight = {
        "event_id": "evt-white-house",
        "commence_time": "2026-06-14T23:00:00Z",
        "fighter_a": "Alpha Fighter",
        "fighter_b": "Beta Fighter",
    }
    event_context = bot._resolve_live_event_context(
        fight,
        [
            {
                "event_id": "evt-white-house",
                "event_date": "June 14, 2026",
                "commence_time": "2026-06-14T23:00:00Z",
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "weight_class": "Light Heavyweight",
                "num_rounds": 3,
                "is_title_bout": False,
                "is_empty_arena": False,
            }
        ],
    )

    assert event_context["event_date"] == "June 14, 2026"
    assert event_context["card_date"] == "2026-06-14"


def test_resolve_live_event_context_matches_almabaev_market_spelling():
    fight = {
        "event_id": "evt-baku",
        "commence_time": "2026-06-27T17:15:00Z",
        "fighter_a": "Asu Almabaev",
        "fighter_b": "Charles Johnson",
    }

    event_context = bot._resolve_live_event_context(
        fight,
        [
            {
                "event_id": "evt-baku",
                "event_date": "June 27, 2026",
                "commence_time": "2026-06-27T17:15:00Z",
                "fighter_a": "Asu Almabayev",
                "fighter_b": "Charles Johnson",
                "weight_class": "Flyweight",
                "num_rounds": 3,
                "is_title_bout": False,
                "is_empty_arena": False,
            }
        ],
        allow_off_card_history_fallback=False,
    )

    assert event_context is not None
    assert event_context["weight_class"] == "Flyweight"
    assert event_context["card_date"] == "2026-06-27"


def test_resolve_live_event_context_matches_ludovit_klein_apostrophe_artifact():
    fight = {
        "event_id": "odds-ufc-belgrade",
        "commence_time": "2026-08-01T16:50:00Z",
        "fighter_a": "L'udovit Klein",
        "fighter_b": "Tofiq Musayev",
    }

    event_context = bot._resolve_live_event_context(
        fight,
        [
            {
                "event_id": "ufc-belgrade",
                "event_date": "August 1, 2026",
                "commence_time": "2026-08-01T16:50:00Z",
                "fighter_a": "Ludovít Klein",
                "fighter_b": "Tofiq Musayev",
                "weight_class": "Lightweight",
                "num_rounds": 3,
                "is_title_bout": False,
                "is_empty_arena": False,
            }
        ],
        allow_off_card_history_fallback=False,
    )

    assert event_context is not None
    assert event_context["weight_class"] == "Lightweight"
    assert event_context["card_date"] == "2026-08-01"


@pytest.mark.parametrize(
    ("odds_names", "card_names", "weight_class"),
    [
        (("Damien Anderson", "Ezra Elliott"), ("Ezra Elliot", "Damien Anderson"), "Featherweight"),
        (("Jean-Paul Lebosnoyani", "Seok Hyun Ko"), ("Jean-Paul Lebosnoyani", "Seokhyeon Ko"), "Welterweight"),
    ],
)
def test_resolve_live_event_context_matches_real_ufc_oklahoma_card_variants(
    odds_names,
    card_names,
    weight_class,
):
    event_context = bot._resolve_live_event_context(
        {
            "event_id": "odds-event",
            "commence_time": "2026-07-18T22:25:00Z",
            "fighter_a": odds_names[0],
            "fighter_b": odds_names[1],
        },
        [{
            "event_date": "July 18, 2026",
            "fighter_a": card_names[0],
            "fighter_b": card_names[1],
            "weight_class": weight_class,
            "num_rounds": 3,
            "is_title_bout": False,
        }],
        allow_off_card_history_fallback=False,
    )

    assert event_context is not None
    assert event_context["weight_class"] == weight_class


@pytest.mark.parametrize(
    ("odds_names", "card_names", "weight_class"),
    [
        (("Billy Quarantillo", "Carlos Diego Ferreira"), ("Diego Ferreira", "Billy Quarantillo"), "Lightweight Bout"),
        (("Darren Elkins", "Yadier DelValle"), ("Darren Elkins", "Yadier del Valle"), "Featherweight Bout"),
        (("Jose Montanha", "Louie Sutherland"), ("Louie Sutherland", "Jose Montanha da Silva"), "Heavyweight Bout"),
        (("Billy Goff", "Ty Miller"), ("Ty Miller", "Billy Ray Goff"), "Welterweight Bout"),
    ],
)
def test_resolve_live_event_context_matches_august_2026_card_variants(
    odds_names,
    card_names,
    weight_class,
):
    event_context = bot._resolve_live_event_context(
        {
            "event_id": "odds-august-8",
            # UFC.com reports the local August 8 card date while the odds feed
            # rolls the Las Vegas card into August 9 UTC.
            "commence_time": "2026-08-09T00:00:00Z",
            "fighter_a": odds_names[0],
            "fighter_b": odds_names[1],
        },
        [{
            "event_date": "August 8, 2026",
            "fighter_a": card_names[0],
            "fighter_b": card_names[1],
            "weight_class": weight_class,
            "num_rounds": 3,
            "is_title_bout": False,
        }],
        allow_off_card_history_fallback=False,
    )

    assert event_context is not None
    assert event_context["weight_class"] == weight_class
    assert event_context["card_date"] == "2026-08-08"


def test_resolve_live_event_context_matches_gigi_canuto_official_card_name():
    event_context = bot._resolve_live_event_context(
        {
            "event_id": "825e28002cc8ba14f67a913a93833346",
            "commence_time": "2026-08-08T21:10:00Z",
            "fighter_a": "Carol Foro",
            "fighter_b": "Giovanna Canuto",
        },
        [{
            "event_date": "August 8, 2026",
            "fighter_a": "Gigi Canuto",
            "fighter_b": "Carol Foro",
            "weight_class": "Women's Strawweight Bout",
            "num_rounds": 3,
            "is_title_bout": False,
        }],
        allow_off_card_history_fallback=False,
    )

    assert event_context is not None
    assert event_context["weight_class"] == "Women's Strawweight Bout"
    assert event_context["card_date"] == "2026-08-08"


def test_resolve_live_event_context_rejects_duplicate_pair_on_different_card_date():
    event_context = bot._resolve_live_event_context(
        {
            "event_id": "odds-august-15",
            "commence_time": "2026-08-15T23:30:00Z",
            "fighter_a": "Jose Montanha",
            "fighter_b": "Louie Sutherland",
        },
        [{
            "event_id": "odds-august-8",
            "event_date": "August 8, 2026",
            "fighter_a": "Louie Sutherland",
            "fighter_b": "Jose Montanha da Silva",
            "weight_class": "Heavyweight Bout",
            "num_rounds": 3,
            "is_title_bout": False,
        }],
        allow_off_card_history_fallback=False,
    )

    assert event_context is None


@pytest.mark.parametrize(
    ("fighter_a", "fighter_b"),
    [
        ("Trey Waters", "Trukon Carson"),
        ("Bruno Cappelozza", "Valentin Moldavsky"),
        ("Bryan Battle", "Dalton Rosta"),
    ],
)
def test_plausible_live_card_identity_mismatch_rejects_unrelated_same_date_mma(
    fighter_a,
    fighter_b,
):
    fight = {
        "commence_time": "2026-08-08T02:00:00Z",
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
    }
    contexts = [{
        "event_date": "August 8, 2026",
        "fighter_a": "Billy Quarantillo",
        "fighter_b": "Diego Ferreira",
    }]

    assert not bot._plausible_live_card_identity_mismatch(fight, contexts)


def test_plausible_live_card_identity_mismatch_accepts_canonical_fighter_overlap():
    fight = {
        # UTC rollover relative to the official local card date is intentional.
        "commence_time": "2026-08-09T00:30:00Z",
        "fighter_a": "L'udovit Klein",
        "fighter_b": "Replacement Opponent",
    }
    contexts = [{
        "event_date": "August 8, 2026",
        "fighter_a": "Ludovít Klein",
        "fighter_b": "Tofiq Musayev",
    }]

    assert bot._plausible_live_card_identity_mismatch(fight, contexts)


def test_plausible_live_card_identity_mismatch_requires_nearby_card_date():
    fight = {
        "commence_time": "2026-08-16T00:30:00Z",
        "fighter_a": "L'udovit Klein",
        "fighter_b": "Replacement Opponent",
    }
    contexts = [{
        "event_date": "August 8, 2026",
        "fighter_a": "Ludovít Klein",
        "fighter_b": "Tofiq Musayev",
    }]

    assert not bot._plausible_live_card_identity_mismatch(fight, contexts)


def test_plausible_live_card_identity_mismatch_accepts_reversed_two_name_fuzzy_pair():
    fight = {
        "commence_time": "2026-08-08T22:30:00Z",
        "fighter_a": "Tofiq Musaev",
        "fighter_b": "Ludovit Kline",
    }
    contexts = [{
        "event_date": "August 8, 2026",
        "fighter_a": "Ludovit Klein",
        "fighter_b": "Tofiq Musayev",
    }]

    assert bot._plausible_live_card_identity_mismatch(fight, contexts)


def test_plausible_live_card_identity_mismatch_requires_both_names_for_fuzzy_only_match():
    fight = {
        "commence_time": "2026-08-08T22:30:00Z",
        "fighter_a": "Ludovit Kline",
        "fighter_b": "Unrelated Opponent",
    }
    contexts = [{
        "event_date": "August 8, 2026",
        "fighter_a": "Ludovit Klein",
        "fighter_b": "Tofiq Musayev",
    }]

    assert not bot._plausible_live_card_identity_mismatch(fight, contexts)


def _make_repo_local_tmp_dir() -> Path:
    path = Path.cwd() / "data" / f"bot-live-context-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _base_live_features() -> dict:
    return {
        "a_num_fights": 5,
        "b_num_fights": 6,
        "a_ko_rate": 0.12,
        "b_ko_rate": 0.18,
        "a_sub_rate": 0.04,
        "b_sub_rate": 0.03,
        "a_dec_rate": 0.54,
        "b_dec_rate": 0.49,
        "a_roll_slpm": 3.8,
        "b_roll_slpm": 4.1,
        "a_roll_kd": 0.4,
        "b_roll_kd": 0.5,
        "a_roll_sub_avg": 0.2,
        "b_roll_sub_avg": 0.1,
        "a_roll_td_avg": 1.5,
        "b_roll_td_avg": 1.2,
        "a_total_rounds": 18.0,
        "b_total_rounds": 20.0,
        "a_roll_str_def": 0.57,
        "b_roll_str_def": 0.54,
        "a_roll_td_def": 0.69,
        "b_roll_td_def": 0.64,
        "a_roll_sapm": 2.9,
        "b_roll_sapm": 3.2,
        "a_wins": 8.0,
        "b_wins": 10.0,
        "a_losses": 2.0,
        "b_losses": 3.0,
        "a_draws": 0.0,
        "b_draws": 0.0,
        "a_win_pct": 0.8,
        "b_win_pct": 0.7692,
        "a_current_win_streak": 2.0,
        "b_current_win_streak": 1.0,
        "a_lose_streak": 0.0,
        "b_lose_streak": 0.0,
        "a_days_since_last_fight": 180.0,
        "b_days_since_last_fight": 210.5,
        "a_cage_rust": 0.0,
        "b_cage_rust": 0.0,
    }


def _fake_model_result(artifact_path: Path, *, feature_cols: list[str] | None = None) -> dict:
    cols = list(feature_cols or [])
    return {
        "feature_cols": cols,
        "col_medians": np.zeros(len(cols)),
        "feature_importance": {},
        "raw_model": None,
        "artifact_path": str(artifact_path),
        "training_spec": {
            "name": "unit-test-spec",
            "feature_cols": cols,
        },
    }


def _cached_prediction_row(
    fight: dict,
    *,
    runtime_signature: dict,
    generated_at: str,
    prob_a: float = 0.61,
    prob_b: float = 0.39,
    confidence: float = 0.61,
    a_market_prob: float | None = None,
    b_market_prob: float | None = None,
    line_movement: float | None = None,
    trade_blocked: bool = False,
    data_quality_retry_after: str | None = None,
) -> dict:
    a_market = fight["a_fair_prob_avg"] if a_market_prob is None else a_market_prob
    b_market = fight["b_fair_prob_avg"] if b_market_prob is None else b_market_prob
    features = _base_live_features()
    row = {
        "schema_version": bot.PREDICTION_CACHE_SCHEMA_VERSION,
        "fighter_a": fight["fighter_a"],
        "fighter_b": fight["fighter_b"],
        "prob_a": prob_a,
        "prob_b": prob_b,
        "confidence": confidence,
        "event_id": fight.get("event_id", ""),
        "event_date": fight["commence_time"],
        "a_market_prob": a_market,
        "b_market_prob": b_market,
        "no_odds_prob_a": None,
        "no_odds_prob_b": None,
        "a_num_fights": 5,
        "b_num_fights": 6,
        "shap_values": [],
        "shap_base_value": None,
        "feature_highlights": [],
        "low_experience": False,
        "trade_blocked": trade_blocked,
        "trade_block_reason": "",
        "data_quality_retry_after": data_quality_retry_after,
        "method_stats": {
            "a_ko_rate": 0.12,
            "b_ko_rate": 0.18,
        },
        "fighter_context": {
            "a_wins": 8,
            "b_wins": 10,
        },
        "pair_key": bot._live_fight_pair_key(fight["fighter_a"], fight["fighter_b"]),
        "cache_key": bot._prediction_cache_key(fight),
        "prediction_generated_at": generated_at,
        "odds_snapshot": {
            "a_fair_prob_avg": a_market,
            "b_fair_prob_avg": b_market,
        },
        "prediction_input_odds_snapshot": {
            "a_fair_prob_avg": a_market,
            "b_fair_prob_avg": b_market,
        },
        "prediction_input_line_features": {},
        "event_context_snapshot": {
            "event_id": fight.get("event_id", ""),
            "commence_time": bot._prediction_commence_token(fight["commence_time"]),
            "weight_class": "Bantamweight",
            "num_rounds": 3,
            "is_title_bout": False,
            "is_empty_arena": False,
        },
        "runtime_signature": runtime_signature,
        "method_odds_fingerprint": "method-odds:not-requested",
        "model_features": features,
        "feature_provenance": {
            "bundle_id": "unit-bundle",
            "model_spec_name": "unit-test-spec",
        },
    }
    if line_movement is not None:
        row["prediction_input_line_features"] = {"line_movement": line_movement}
        row["line_movement"] = line_movement
        row["line_is_sharp"] = 0
        row["line_steam_move"] = 0
    return row


def test_resolve_live_fight_counts_prefers_live_feature_counts():
    counts = bot._resolve_live_fight_counts(
        {
            "a_num_fights": 17,
            "b_num_fights": "9",
        },
        "Israel Adesanya",
        "Joseph Pyfer",
        fallback_resolver=lambda _: 0,
    )

    assert counts == (17, 9)


def test_resolve_live_fight_counts_falls_back_when_live_counts_missing():
    fallback_counts = {
        "Israel Adesanya": 18,
        "Joseph Pyfer": 0,
    }
    counts = bot._resolve_live_fight_counts(
        {
            "a_num_fights": None,
            "b_num_fights": float("nan"),
        },
        "Israel Adesanya",
        "Joseph Pyfer",
        fallback_resolver=lambda name: fallback_counts[name],
    )

    assert counts == (18, 0)


def test_canonicalize_live_fighter_name_uses_project_alias_for_ian_garry(monkeypatch):
    monkeypatch.setattr(bot, "_load_live_fighter_alias_map", lambda: {})

    assert bot._canonicalize_live_fighter_name("Ian Garry") == "ian machado garry"


def test_live_alias_map_does_not_infer_missing_middle_name(tmp_path, monkeypatch):
    roster_path = tmp_path / "roster.csv"
    pd.DataFrame(
        [
            {
                "official_name": "John Michael Smith",
                "official_url_identity_status": "matched",
                "official_url_identity_valid": True,
            }
        ]
    ).to_csv(roster_path, index=False)
    monkeypatch.setattr(
        "src.data.ufc_active_roster.OFFICIAL_ACTIVE_ROSTER_PATH",
        roster_path,
    )
    bot._LIVE_CONTEXT_TABLE_CACHE.clear()

    assert bot._canonicalize_live_fighter_name("John Smith") == "john smith"


def test_live_alias_map_accepts_explicit_missing_middle_name_alias(tmp_path, monkeypatch):
    roster_path = tmp_path / "roster.csv"
    pd.DataFrame(
        [
            {
                "official_name": "John Michael Smith",
                "alternate_slug_names": "John Smith",
                "official_url_identity_status": "matched",
                "official_url_identity_valid": True,
            }
        ]
    ).to_csv(roster_path, index=False)
    monkeypatch.setattr(
        "src.data.ufc_active_roster.OFFICIAL_ACTIVE_ROSTER_PATH",
        roster_path,
    )
    bot._LIVE_CONTEXT_TABLE_CACHE.clear()

    assert (
        bot._canonicalize_live_fighter_name("John Smith")
        == "john michael smith"
    )


def test_live_prediction_quality_blocks_generic_fallback_history():
    assessment = bot._live_prediction_quality_assessment(
        _base_live_features(),
        {"fighter_a_source": "fallback", "fighter_b_source": "ufcstats"},
        fighter_a="Ian Garry",
        fighter_b="Islam Makhachev",
        a_fights=11,
        b_fights=16,
    )

    assert assessment["blocked"] is True
    assert "lower-fidelity external fallback" in assessment["reasons"][0]


def test_live_prediction_quality_blocks_unavailable_ufcstats_history():
    assessment = bot._live_prediction_quality_assessment(
        _base_live_features(),
        {
            "fighter_a_source": "ufcstats",
            "fighter_a_fight_history_status": "unavailable",
            "fighter_b_source": "ufcstats",
            "fighter_b_fight_history_status": "complete",
        },
        fighter_a="Partial Profile",
        fighter_b="Complete Profile",
        a_fights=0,
        b_fights=6,
    )

    assert assessment["blocked"] is True
    assert "fight history was unavailable" in assessment["reasons"][0]
    assert assessment["fighters"]["a"]["fight_history_status"] == "unavailable"


def test_live_prediction_quality_allows_native_missing_values_for_verified_newcomers():
    assessment = bot._live_prediction_quality_assessment(
        {"a_num_fights": 0, "b_num_fights": 1},
        {
            "fighter_a_source": "ufcstats",
            "fighter_a_fight_history_status": "complete",
            "fighter_b_source": "ufcstats",
            "fighter_b_fight_history_status": "complete",
        },
        fighter_a="Alpha",
        fighter_b="Beta",
        a_fights=0,
        b_fights=1,
    )

    assert assessment["blocked"] is False


def test_live_prediction_quality_rejects_test_only_provenance():
    assessment = bot._live_prediction_quality_assessment(
        _base_live_features(),
        {"fighter_a_source": "test", "fighter_b_source": "ufcstats"},
        fighter_a="Alpha",
        fighter_b="Beta",
        a_fights=5,
        b_fights=6,
    )

    assert assessment["blocked"] is True
    assert "unsupported fighter-data provenance 'test'" in assessment["reasons"][0]


def test_live_prediction_quality_blocks_missing_lookup_provenance():
    assessment = bot._live_prediction_quality_assessment(
        _base_live_features(),
        {},
        fighter_a="Unknown Fighter",
        fighter_b="Known Fighter",
        a_fights=8,
        b_fights=8,
    )

    assert assessment["blocked"] is True
    assert "no verified fighter-data provenance" in assessment["reasons"][0]


def test_prediction_cache_refreshes_when_method_odds_fingerprint_changes(monkeypatch):
    fight = {
        "event_id": "evt-1",
        "commence_time": "2026-03-28T20:00:00Z",
        "fighter_a": "Alpha Fighter",
        "fighter_b": "Beta Fighter",
        "a_fair_prob_avg": 0.55,
        "b_fair_prob_avg": 0.45,
    }
    runtime_signature = {"model": "unit"}
    cached = _cached_prediction_row(
        fight,
        runtime_signature=runtime_signature,
        generated_at="2026-03-28T18:00:00+00:00",
    )
    cached["method_odds_fingerprint"] = "method-odds-v1:old"
    monkeypatch.setattr(
        bot,
        "_current_utc",
        lambda: datetime(2026, 3, 28, 18, 30, tzinfo=timezone.utc),
    )

    unchanged = bot._prediction_needs_refresh(
        cached,
        fight,
        runtime_signature=runtime_signature,
        method_odds_fingerprint="method-odds-v1:old",
        current_event_context_snapshot=cached["event_context_snapshot"],
        current_line_feature_snapshot=cached["prediction_input_line_features"],
    )
    changed = bot._prediction_needs_refresh(
        cached,
        fight,
        runtime_signature=runtime_signature,
        method_odds_fingerprint="method-odds-v1:new",
        current_event_context_snapshot=cached["event_context_snapshot"],
        current_line_feature_snapshot=cached["prediction_input_line_features"],
    )

    assert unchanged == (False, "cache hit")
    assert changed == (True, "method odds changed")


def test_prediction_cache_measures_cumulative_odds_move_from_generation(monkeypatch):
    fight = {
        "event_id": "evt-1",
        "commence_time": "2026-03-28T20:00:00Z",
        "fighter_a": "Alpha Fighter",
        "fighter_b": "Beta Fighter",
        "a_fair_prob_avg": 0.50,
        "b_fair_prob_avg": 0.50,
    }
    runtime_signature = {"model": "unit"}
    cached = _cached_prediction_row(
        fight,
        runtime_signature=runtime_signature,
        generated_at="2026-03-28T18:00:00+00:00",
    )
    monkeypatch.setattr(
        bot,
        "_current_utc",
        lambda: datetime(2026, 3, 28, 18, 30, tzinfo=timezone.utc),
    )

    first_poll = dict(fight, a_fair_prob_avg=0.52, b_fair_prob_avg=0.48)
    assert bot._prediction_needs_refresh(
        cached,
        first_poll,
        runtime_signature=runtime_signature,
        method_odds_fingerprint="method-odds:not-requested",
        current_event_context_snapshot=cached["event_context_snapshot"],
        current_line_feature_snapshot=cached["prediction_input_line_features"],
    ) == (False, "cache hit")

    # A cache hit updates the display/current-market snapshot, but the model's
    # immutable input snapshot must remain at 50/50.
    cached["odds_snapshot"] = bot._prediction_odds_snapshot(first_poll)
    second_poll = dict(fight, a_fair_prob_avg=0.54, b_fair_prob_avg=0.46)
    refresh, reason = bot._prediction_needs_refresh(
        cached,
        second_poll,
        runtime_signature=runtime_signature,
        method_odds_fingerprint="method-odds:not-requested",
        current_event_context_snapshot=cached["event_context_snapshot"],
        current_line_feature_snapshot=cached["prediction_input_line_features"],
    )

    assert refresh is True
    assert reason == "odds moved 4.0%"
    assert cached["prediction_input_odds_snapshot"]["a_fair_prob_avg"] == 0.50


def test_prediction_cache_with_nonnumeric_schema_version_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "LOGS_DIR", tmp_path)
    (tmp_path / "predictions_cache.json").write_text(
        json.dumps({"schema_version": "corrupt", "predictions": []}),
        encoding="utf-8",
    )

    assert bot._load_existing_prediction_cache() == {}


@pytest.mark.parametrize("invalid_value", [None, "false", 0, 1])
def test_prediction_cache_rejects_missing_or_nonboolean_trade_gate(
    tmp_path,
    monkeypatch,
    invalid_value,
):
    monkeypatch.setattr(bot, "LOGS_DIR", tmp_path)
    fight = {
        "event_id": "evt-1",
        "commence_time": "2026-03-28T20:00:00Z",
        "fighter_a": "Alpha Fighter",
        "fighter_b": "Beta Fighter",
        "a_fair_prob_avg": 0.55,
        "b_fair_prob_avg": 0.45,
    }
    row = _cached_prediction_row(
        fight,
        runtime_signature={"model": "unit"},
        generated_at="2026-03-28T18:00:00+00:00",
    )
    if invalid_value is None:
        row.pop("trade_blocked")
    else:
        row["trade_blocked"] = invalid_value
    (tmp_path / "predictions_cache.json").write_text(
        json.dumps(
            {
                "schema_version": bot.PREDICTION_CACHE_SCHEMA_VERSION,
                "predictions": [row],
            }
        ),
        encoding="utf-8",
    )

    assert bot._load_existing_prediction_cache() == {}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weight_class", "Featherweight"),
        ("num_rounds", 5),
        ("is_title_bout", True),
        ("is_empty_arena", True),
    ],
)
def test_prediction_cache_refreshes_when_model_event_context_changes(
    monkeypatch,
    field,
    value,
):
    fight = {
        "event_id": "evt-1",
        "commence_time": "2026-03-28T20:00:00Z",
        "fighter_a": "Alpha Fighter",
        "fighter_b": "Beta Fighter",
        "a_fair_prob_avg": 0.55,
        "b_fair_prob_avg": 0.45,
    }
    runtime_signature = {"model": "unit"}
    cached = _cached_prediction_row(
        fight,
        runtime_signature=runtime_signature,
        generated_at="2026-03-28T18:00:00+00:00",
    )
    current_context = dict(cached["event_context_snapshot"])
    current_context[field] = value
    monkeypatch.setattr(
        bot,
        "_current_utc",
        lambda: datetime(2026, 3, 28, 18, 30, tzinfo=timezone.utc),
    )

    assert bot._prediction_needs_refresh(
        cached,
        fight,
        runtime_signature=runtime_signature,
        method_odds_fingerprint="method-odds:not-requested",
        current_event_context_snapshot=current_context,
        current_line_feature_snapshot=cached["prediction_input_line_features"],
    ) == (True, f"event context changed: {field}")


def test_prediction_runtime_signature_versions_live_quality_gate(monkeypatch):
    model_result = {
        "feature_cols": ["a_roll_slpm"],
        "training_spec": {"name": "unit", "feature_cols": ["a_roll_slpm"]},
    }
    first = bot._prediction_runtime_signature(model_result=model_result)

    monkeypatch.setattr(
        bot,
        "LIVE_DATA_QUALITY_MAX_MISSING_CRITICAL",
        bot.LIVE_DATA_QUALITY_MAX_MISSING_CRITICAL + 1,
    )
    changed = bot._prediction_runtime_signature(model_result=model_result)

    assert first["live_data_quality_gate_version"] == bot._LIVE_DATA_QUALITY_GATE_VERSION
    assert first["live_data_quality_block_fallback"] is bool(
        bot.LIVE_DATA_QUALITY_BLOCK_FALLBACK
    )
    assert first["live_data_quality_max_missing_critical"] + 1 == changed[
        "live_data_quality_max_missing_critical"
    ]
    assert first != changed

    monkeypatch.setattr(
        bot,
        "_LIVE_PREDICTION_INFERENCE_CONTRACT_VERSION",
        bot._LIVE_PREDICTION_INFERENCE_CONTRACT_VERSION + 1,
    )
    contract_changed = bot._prediction_runtime_signature(model_result=model_result)
    assert contract_changed["inference_contract_version"] == 2
    assert contract_changed != first


def test_prediction_artifact_signature_is_stable_across_mtime_only_deploy_restore(
    tmp_path,
):
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"promoted-model-bytes")

    first = bot._prediction_cache_artifact_signature(artifact)
    original_stat = artifact.stat()
    os.utime(
        artifact,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 5_000_000_000),
    )
    restored = bot._prediction_cache_artifact_signature(artifact)

    assert first == restored
    assert first["sha256"] == hashlib.sha256(b"promoted-model-bytes").hexdigest()

    artifact.write_bytes(b"different-model-bytes")
    replaced = bot._prediction_cache_artifact_signature(artifact)

    assert replaced != restored
    assert replaced["sha256"] == hashlib.sha256(b"different-model-bytes").hexdigest()


def test_prediction_runtime_signature_accepts_hosted_mtime_to_hash_migration():
    stable = {
        "primary_spec_name": "unit",
        "bundle_id": "promoted-bundle",
        "bundle_built_at": "2026-07-28T10:00:00Z",
    }
    cached = {
        **stable,
        "primary_artifact": {
            "path": "/app/models/xgboost_model.pkl",
            "size": 1234,
            "mtime_ns": 100,
        },
        "no_odds_artifact": None,
    }
    current = {
        **stable,
        "primary_artifact": {
            "path": "/app/models/xgboost_model.pkl",
            "size": 1234,
            "sha256": "abc123",
        },
        "no_odds_artifact": None,
    }

    assert bot._prediction_runtime_signatures_match(cached, current)
    assert not bot._prediction_runtime_signatures_match(
        cached,
        {**current, "bundle_id": "different-bundle"},
    )
    assert not bot._prediction_runtime_signatures_match(
        current,
        {
            **current,
            "primary_artifact": {
                **current["primary_artifact"],
                "sha256": "changed",
            },
        },
    )


def test_prediction_cache_rate_limits_previous_data_quality_block(monkeypatch):
    fight = {
        "event_id": "evt-1",
        "commence_time": "2026-03-28T20:00:00Z",
        "fighter_a": "Alpha Fighter",
        "fighter_b": "Beta Fighter",
        "a_fair_prob_avg": 0.55,
        "b_fair_prob_avg": 0.45,
    }
    runtime_signature = {"model": "unit"}
    cached = _cached_prediction_row(
        fight,
        runtime_signature=runtime_signature,
        generated_at="2026-03-28T18:00:00+00:00",
        trade_blocked=True,
        data_quality_retry_after="2026-03-28T18:45:00+00:00",
    )
    monkeypatch.setattr(
        bot,
        "_current_utc",
        lambda: datetime(2026, 3, 28, 18, 30, tzinfo=timezone.utc),
    )

    assert bot._prediction_needs_refresh(
        cached,
        fight,
        runtime_signature=runtime_signature,
        method_odds_fingerprint="method-odds:not-requested",
        current_event_context_snapshot=cached["event_context_snapshot"],
        current_line_feature_snapshot=cached["prediction_input_line_features"],
    ) == (False, "blocked data-quality retry cooldown")

    monkeypatch.setattr(
        bot,
        "_current_utc",
        lambda: datetime(2026, 3, 28, 18, 46, tzinfo=timezone.utc),
    )
    assert bot._prediction_needs_refresh(
        cached,
        fight,
        runtime_signature=runtime_signature,
        method_odds_fingerprint="method-odds:not-requested",
        current_event_context_snapshot=cached["event_context_snapshot"],
        current_line_feature_snapshot=cached["prediction_input_line_features"],
    ) == (True, bot._DATA_QUALITY_RETRY_REASON)


def test_infer_weight_class_from_history_falls_back_to_raw_history_when_processed_schema_invalid(
    monkeypatch,
):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        processed_dir.mkdir()
        raw_dir.mkdir()

        (processed_dir / "fights_cleaned.csv").write_text("foo,bar\n1,2\n", encoding="utf-8")
        (raw_dir / "ufc-master.csv").write_text(
            "\n".join(
                [
                    "RedFighter,BlueFighter,WeightClass,Date",
                    "Rob Font,Adrian Yanez,Bantamweight,2023-04-08",
                    "Song Yadong,Ricky Simon,Bantamweight,2023-04-29",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)

        assert bot._infer_weight_class_from_history("Ricky Simon", "Adrian Yanez") == "Bantamweight"
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_uses_raw_history_when_processed_history_missing(
    monkeypatch,
):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        processed_dir.mkdir()
        raw_dir.mkdir()

        (raw_dir / "ufc-master.csv").write_text(
            "\n".join(
                [
                    "RedFighter,BlueFighter,WeightClass,Date",
                    "Rob Font,Adrian Yanez,Bantamweight,2023-04-08",
                    "Song Yadong,Ricky Simon,Bantamweight,2023-04-29",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (raw_dir / "ufc_fighters_scraped.csv").write_text(
            "name\nRicky Simon\nAdrian Yanez\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)

        event_context = bot._resolve_live_event_context(
            {
                "event_id": "evt-1",
                "commence_time": "2026-03-28T20:00:00+00:00",
                "fighter_a": "Ricky Simon",
                "fighter_b": "Adrian Yanez",
            },
            [],
        )

        assert event_context is not None
        assert event_context["weight_class"] == "Bantamweight"
        assert event_context["num_rounds"] == 3
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_falls_back_to_near_term_ufc_lookup_when_history_missing(
    monkeypatch,
):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        processed_dir.mkdir()
        raw_dir.mkdir()

        (raw_dir / "ufc_fighters_scraped.csv").write_text(
            "name\nRicky Simon\nAdrian Yanez\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr(
            bot,
            "_current_utc",
            lambda: datetime(2026, 3, 20, tzinfo=timezone.utc),
        )
        monkeypatch.setattr(
            "src.data.fighter_lookup._lookup_processed_fighter",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "src.data.fighter_lookup.search_fighter_url",
            lambda fighter_name: f"http://example.test/{fighter_name.replace(' ', '-').lower()}",
        )
        monkeypatch.setattr(
            "src.data.fighter_lookup.scrape_fighter_fights",
            lambda *args, **kwargs: [
                {"event_date": "2025-09-01", "weight_class": "Bantamweight"},
            ],
        )

        event_context = bot._resolve_live_event_context(
            {
                "event_id": "evt-1",
                "commence_time": "2026-03-28T20:00:00+00:00",
                "fighter_a": "Ricky Simon",
                "fighter_b": "Adrian Yanez",
            },
            [
                {
                    "event_id": "evt-official",
                    "commence_time": "2026-03-28T20:00:00+00:00",
                    "event_date": "March 28, 2026",
                    "fighter_a": "Israel Adesanya",
                    "fighter_b": "Joe Pyfer",
                    "weight_class": "Middleweight",
                }
            ],
        )

        assert event_context is not None
        assert event_context["weight_class"] == "Bantamweight"
        assert event_context["num_rounds"] == 3
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_near_term_lookup_requires_both_fighters_to_resolve(
    monkeypatch,
):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        processed_dir.mkdir()
        raw_dir.mkdir()

        (raw_dir / "ufc_fighters_scraped.csv").write_text(
            "name\nMarcin Tybura\nTyrell Fortune\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr(
            bot,
            "_current_utc",
            lambda: datetime(2026, 3, 20, tzinfo=timezone.utc),
        )
        monkeypatch.setattr(
            "src.data.fighter_lookup._lookup_processed_fighter",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "src.data.fighter_lookup.search_fighter_url",
            lambda fighter_name: "http://example.test/marcin-tybura" if fighter_name == "Marcin Tybura" else None,
        )
        monkeypatch.setattr(
            "src.data.fighter_lookup.scrape_fighter_fights",
            lambda *args, **kwargs: [
                {"event_date": "2025-09-01", "weight_class": "Heavyweight"},
            ],
        )
        monkeypatch.setattr(bot, "_official_roster_weight_class", lambda *_args: None)

        event_context = bot._resolve_live_event_context(
            {
                "event_id": "evt-1",
                "commence_time": "2026-03-28T20:00:00+00:00",
                "fighter_a": "Marcin Tybura",
                "fighter_b": "Tyrell Fortune",
            },
            [
                {
                    "event_id": "evt-official",
                    "commence_time": "2026-03-28T20:00:00+00:00",
                    "event_date": "March 28, 2026",
                    "fighter_a": "Israel Adesanya",
                    "fighter_b": "Joe Pyfer",
                    "weight_class": "Middleweight",
                }
            ],
        )

        assert event_context is None
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_uses_official_roster_for_late_debutant_addition(
    monkeypatch,
):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        processed_dir.mkdir()
        raw_dir.mkdir()
        official_path = raw_dir / "ufc_active_roster_official.csv"
        official_path.write_text(
            "official_name,profile_division,profile_status,coverage_eligible\n"
            "Ezra Elliott,Featherweight Division,Active,True\n"
            "Damien Anderson,Featherweight Division,Active,True\n",
            encoding="utf-8",
        )
        (raw_dir / "ufc_fighters_scraped.csv").write_text(
            "name\nEzra Elliott\nDamien Anderson\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr("src.data.ufc_active_roster.OFFICIAL_ACTIVE_ROSTER_PATH", official_path)
        monkeypatch.setattr(bot, "_current_utc", lambda: datetime(2026, 7, 18, tzinfo=timezone.utc))
        monkeypatch.setattr("src.data.fighter_lookup._lookup_processed_fighter", lambda *args, **kwargs: None)
        monkeypatch.setattr("src.data.fighter_lookup.search_fighter_url", lambda *args, **kwargs: None)
        bot._LIVE_CONTEXT_TABLE_CACHE.clear()

        event_context = bot._resolve_live_event_context(
            {
                "event_id": "late-addition",
                "commence_time": "2026-07-18T22:25:00+00:00",
                "fighter_a": "Damien Anderson",
                "fighter_b": "Ezra Elliott",
            },
            [{
                "event_id": "confirmed-card",
                "commence_time": "2026-07-18T23:15:00+00:00",
                "event_date": "July 18, 2026",
                "fighter_a": "Other Fighter",
                "fighter_b": "Another Fighter",
                "weight_class": "Bantamweight",
            }],
            allow_off_card_history_fallback=False,
        )

        assert event_context is not None
        assert event_context["weight_class"] == "Featherweight"
        assert event_context["num_rounds"] == 3
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_official_roster_late_addition_fallback_requires_confirmed_ufc_date(monkeypatch):
    monkeypatch.setattr(bot, "_current_utc", lambda: datetime(2026, 7, 18, tzinfo=timezone.utc))
    monkeypatch.setattr(bot, "_official_roster_weight_class", lambda *_args: "Featherweight")
    monkeypatch.setattr("src.data.fighter_lookup._lookup_processed_fighter", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.data.fighter_lookup.search_fighter_url", lambda *args, **kwargs: None)

    event_context = bot._resolve_live_event_context(
        {
            "event_id": "unconfirmed-event",
            "commence_time": "2026-07-19T22:25:00+00:00",
            "fighter_a": "Alpha Fighter",
            "fighter_b": "Beta Fighter",
        },
        [{
            "event_id": "confirmed-card",
            "commence_time": "2026-07-18T23:15:00+00:00",
            "event_date": "July 18, 2026",
            "fighter_a": "Other Fighter",
            "fighter_b": "Another Fighter",
            "weight_class": "Bantamweight",
        }],
        allow_off_card_history_fallback=False,
    )

    assert event_context is None


def test_load_local_ufc_roster_names_unions_official_roster_artifact(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        raw_dir = temp_root / "raw"
        raw_dir.mkdir()
        official_path = raw_dir / "ufc_active_roster_official.csv"

        (raw_dir / "ufc_fighters_scraped.csv").write_text(
            "name\nRicky Simon\n",
            encoding="utf-8",
        )
        official_path.write_text(
            "\n".join(
                [
                    "official_name,slug_name,alternate_slug_names,ufcstats_name",
                    "Nariman Abbassov,nariman abbassov,nariman abbasov,Nariman Abbasov",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr("src.data.ufc_active_roster.OFFICIAL_ACTIVE_ROSTER_PATH", official_path)
        bot._LIVE_CONTEXT_TABLE_CACHE.clear()

        roster_names = bot._load_local_ufc_roster_names()

        assert normalize_cross_source_name("Ricky Simon") in roster_names
        assert "nariman abbassov" in roster_names
        assert "nariman abbasov" in roster_names
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_matches_official_slug_aliases(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        raw_dir = temp_root / "raw"
        raw_dir.mkdir()
        official_path = raw_dir / "ufc_active_roster_official.csv"
        official_path.write_text(
            "\n".join(
                [
                    "official_name,profile_name,slug_name,alternate_slug_names,ufcstats_name",
                    "Patricio Pitbull,Patricio Pitbull,patricio pitbull freire,patricio pitbull freire,",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr("src.data.ufc_active_roster.OFFICIAL_ACTIVE_ROSTER_PATH", official_path)
        bot._LIVE_CONTEXT_TABLE_CACHE.clear()

        event_context = bot._resolve_live_event_context(
            {
                "event_id": "evt-1",
                "commence_time": "2026-04-12T00:40:00+00:00",
                "fighter_a": "Aaron Pico",
                "fighter_b": "Patricio Pitbull",
            },
            [
                {
                    "event_id": "evt-1",
                    "commence_time": "2026-04-12T00:40:00+00:00",
                    "event_date": "April 11, 2026",
                    "fighter_a": "Patricio Freire",
                    "fighter_b": "Aaron Pico",
                    "weight_class": "Featherweight",
                    "is_main_event": False,
                    "is_title_bout": False,
                    "num_rounds": 3,
                }
            ],
            allow_off_card_history_fallback=False,
        )

        assert event_context is not None
        assert event_context["weight_class"] == "Featherweight"
        assert event_context["num_rounds"] == 3
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_matches_initialed_name_variants(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        raw_dir = temp_root / "raw"
        raw_dir.mkdir()
        official_path = raw_dir / "missing_roster.csv"

        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr("src.data.ufc_active_roster.OFFICIAL_ACTIVE_ROSTER_PATH", official_path)
        bot._LIVE_CONTEXT_TABLE_CACHE.clear()

        event_context = bot._resolve_live_event_context(
            {
                "event_id": "evt-2",
                "commence_time": "2026-04-18T22:00:00+00:00",
                "fighter_a": "J.J. Aldrich",
                "fighter_b": "Jamey-Lyn Horth",
            },
            [
                {
                    "event_id": "evt-2",
                    "commence_time": "2026-04-18T22:00:00+00:00",
                    "event_date": "April 18, 2026",
                    "fighter_a": "JJ Aldrich",
                    "fighter_b": "Jamey-Lyn Horth",
                    "weight_class": "Women's Flyweight",
                    "is_main_event": False,
                    "is_title_bout": False,
                    "num_rounds": 3,
                }
            ],
            allow_off_card_history_fallback=False,
        )

        assert event_context is not None
        assert event_context["weight_class"] == "Women's Flyweight"
        assert event_context["num_rounds"] == 3
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_skips_near_term_lookup_for_far_future_dates(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        processed_dir.mkdir()
        raw_dir.mkdir()

        (raw_dir / "ufc_fighters_scraped.csv").write_text(
            "name\nRicky Simon\nAdrian Yanez\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr(
            bot,
            "_current_utc",
            lambda: datetime(2026, 3, 20, tzinfo=timezone.utc),
        )
        monkeypatch.setattr(
            "src.data.fighter_lookup.search_fighter_url",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected UFCStats lookup")),
        )

        event_context = bot._resolve_live_event_context(
            {
                "event_id": "evt-1",
                "commence_time": "2026-05-01T20:00:00+00:00",
                "fighter_a": "Ricky Simon",
                "fighter_b": "Adrian Yanez",
            },
            [
                {
                    "event_id": "evt-official",
                    "commence_time": "2026-05-01T20:00:00+00:00",
                    "event_date": "May 1, 2026",
                    "fighter_a": "Israel Adesanya",
                    "fighter_b": "Joe Pyfer",
                    "weight_class": "Middleweight",
                }
            ],
        )

        assert event_context is None
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_blocks_recent_prior_matchup_without_official_context(
    monkeypatch,
    caplog,
):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        processed_dir.mkdir()
        raw_dir.mkdir()

        (raw_dir / "ufc-master.csv").write_text(
            "\n".join(
                [
                    "RedFighter,BlueFighter,WeightClass,Date",
                    "Charles Johnson,Bruno Silva,Flyweight,2026-03-14",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)

        with caplog.at_level(logging.INFO):
            event_context = bot._resolve_live_event_context(
                {
                    "event_id": "evt-1",
                    "commence_time": "2026-03-28T20:00:00+00:00",
                    "fighter_a": "Charles Johnson",
                    "fighter_b": "Bruno Silva",
                },
                [],
            )

        assert event_context is None
        assert any(
            record.levelno == logging.INFO and "Refusing fallback live context" in record.message
            for record in caplog.records
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_allows_far_future_rematch_history_fallback(
    monkeypatch,
):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        processed_dir.mkdir()
        raw_dir.mkdir()

        (raw_dir / "ufc-master.csv").write_text(
            "\n".join(
                [
                    "RedFighter,BlueFighter,WeightClass,Date",
                    "Charles Johnson,Bruno Silva,Flyweight,2026-03-14",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)

        event_context = bot._resolve_live_event_context(
            {
                "event_id": "evt-1",
                "commence_time": "2026-05-20T20:00:00+00:00",
                "fighter_a": "Charles Johnson",
                "fighter_b": "Bruno Silva",
            },
            [],
        )

        assert event_context is not None
        assert event_context["weight_class"] == "Flyweight"
        assert event_context["num_rounds"] == 3
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_resolve_live_event_context_blocks_off_card_history_fallback_in_strict_mode(
    monkeypatch,
):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        processed_dir.mkdir()
        raw_dir.mkdir()

        (raw_dir / "ufc-master.csv").write_text(
            "\n".join(
                [
                    "RedFighter,BlueFighter,WeightClass,Date",
                    "Charles Johnson,Bruno Silva,Flyweight,2026-03-14",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)

        event_context = bot._resolve_live_event_context(
            {
                "event_id": "evt-1",
                "commence_time": "2026-05-20T20:00:00+00:00",
                "fighter_a": "Charles Johnson",
                "fighter_b": "Bruno Silva",
            },
            [],
            allow_off_card_history_fallback=False,
        )

        assert event_context is None
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_cmd_duo_live_caches_predictions_with_resolved_live_context(
    monkeypatch,
):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        logs_dir = temp_root / "logs"
        processed_dir.mkdir()
        raw_dir.mkdir()
        logs_dir.mkdir()

        (raw_dir / "ufc-master.csv").write_text(
            "\n".join(
                [
                    "RedFighter,BlueFighter,WeightClass,Date",
                    "Rob Font,Adrian Yanez,Bantamweight,2023-04-08",
                    "Song Yadong,Ricky Simon,Bantamweight,2023-04-29",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        class FakeOddsClient:
            def get_live_odds(self):
                return []

            def odds_to_dataframe(self, _odds):
                return pd.DataFrame()

            def get_consensus_odds(self, _odds_df):
                return pd.DataFrame(
                    [
                        {
                            "event_id": "evt-1",
                            "commence_time": "2026-03-28T20:00:00Z",
                            "fighter_a": "Ricky Simon",
                            "fighter_b": "Adrian Yanez",
                            "a_fair_prob_avg": 0.54,
                            "b_fair_prob_avg": 0.46,
                            "num_bookmakers": 8,
                        }
                    ]
                )

        captured = {}

        def fake_build_fight_features(*args, **kwargs):
            captured["fighter_a"] = args[0]
            captured["fighter_b"] = args[1]
            captured["weight_class"] = kwargs["weight_class"]
            return {
                "a_num_fights": 5,
                "b_num_fights": 6,
                "a_ko_rate": 0.0,
                "b_ko_rate": 0.0,
                "a_sub_rate": 0.0,
                "b_sub_rate": 0.0,
                "a_dec_rate": 0.0,
                "b_dec_rate": 0.0,
                "a_roll_slpm": 0.0,
                "b_roll_slpm": 0.0,
                "a_roll_kd": 0.0,
                "b_roll_kd": 0.0,
                "a_roll_sub_avg": 0.0,
                "b_roll_sub_avg": 0.0,
                "a_roll_td_avg": 0.0,
                "b_roll_td_avg": 0.0,
                "a_total_rounds": 0.0,
                "b_total_rounds": 0.0,
            }

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
        monkeypatch.setattr(bot, "_current_utc", lambda: datetime(2026, 3, 28, 18, 30, tzinfo=timezone.utc))  # >1h before 20:00 commence (LIVE_TRADE_START_BUFFER)
        monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_load_live_event_contexts", lambda: [])
        monkeypatch.setattr(
            bot,
            "_resolve_live_event_context",
            lambda *_args, **_kwargs: {
                "weight_class": "Bantamweight",
                "is_title_bout": False,
                "is_empty_arena": False,
                "num_rounds": 3,
            },
        )
        monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
        monkeypatch.setattr(
            "src.model.train.load_model",
            lambda _name: {
                "feature_cols": [],
                "col_medians": np.array([]),
                "feature_importance": {},
                "raw_model": None,
            },
        )
        monkeypatch.setattr(
            "src.model.predict.predict_fight",
            lambda *_args, **_kwargs: {"prob_a": 0.61, "prob_b": 0.39, "confidence": 0.61},
        )
        monkeypatch.setattr("src.data.fighter_lookup.build_fight_features", fake_build_fight_features)
        monkeypatch.setattr("src.data.line_tracker.get_line_movement_features", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(
            "src.data.line_tracker.detect_injury_or_cancellation",
            lambda *_args, **_kwargs: {"suspected": False},
        )
        monkeypatch.setattr(
            "src.polymarket.markets.get_ufc_fight_markets",
            lambda: pd.DataFrame([{"slug": "ricky-simon-vs-adrian-yanez"}]),
        )
        monkeypatch.setattr(
            "src.strategy.duo_trader.run_duo_traders",
            lambda *_args, **_kwargs: {"total_orders": 0},
        )

        bot.cmd_duo_live(type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})())

        cache_path = logs_dir / "predictions_cache.json"
        assert cache_path.exists()
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        assert captured["fighter_a"] == "Ricky Simon"
        assert captured["fighter_b"] == "Adrian Yanez"
        assert captured["weight_class"] == "Bantamweight"
        assert len(payload["predictions"]) == 1
        assert payload["predictions"][0]["fighter_a"] == "Ricky Simon"
        assert payload["predictions"][0]["fighter_b"] == "Adrian Yanez"
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_cmd_duo_live_caches_but_never_executes_fallback_quality_block(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        logs_dir = temp_root / "logs"
        processed_dir.mkdir()
        raw_dir.mkdir()
        logs_dir.mkdir()

        class FakeOddsClient:
            def get_live_odds(self):
                return []

            def odds_to_dataframe(self, _odds):
                return pd.DataFrame()

            def get_consensus_odds(self, _odds_df):
                return pd.DataFrame(
                    [
                        {
                            "event_id": "evt-quality",
                            "commence_time": "2026-03-28T22:00:00Z",
                            "fighter_a": "Ian Garry",
                            "fighter_b": "Opponent Fighter",
                            "a_fair_prob_avg": 0.54,
                            "b_fair_prob_avg": 0.46,
                            "num_bookmakers": 8,
                        }
                    ]
                )

        trader_calls: list[int] = []
        maintenance_calls: list[dict] = []
        force_refresh_values: list[bool] = []
        fingerprint_pairs: list[tuple[str, str]] = []
        method_fingerprint = ["method-odds:v1"]
        now_value = [datetime(2026, 3, 28, 18, 30, tzinfo=timezone.utc)]

        def fake_build_fight_features(*_args, **_kwargs):
            force_refresh_values.append(bool(_kwargs.get("force_fighter_refresh")))
            if len(force_refresh_values) == 2:
                # The unrelated method-odds rebuild starts before the retry
                # deadline but feature construction finishes after it.
                now_value[0] = datetime(2026, 3, 28, 19, 31, tzinfo=timezone.utc)
            return _base_live_features(), {
                "fighter_a_source": "fallback",
                "fighter_b_source": "ufcstats",
            }

        def fake_method_fingerprint(
            _fight,
            *,
            inference_spec,
            fighter_a,
            fighter_b,
        ):
            fingerprint_pairs.append((fighter_a, fighter_b))
            return method_fingerprint[0]

        def fail_if_trader_runs(*_args, **_kwargs):
            trader_calls.append(1)
            pytest.fail("quality-blocked predictions must not reach run_duo_traders")

        def fake_cancel_duo_open_limit_orders(**kwargs):
            maintenance_calls.append(dict(kwargs))
            return {
                "status": "dry_run",
                "cancelled": 0,
                "kept": 0,
                "reconciled": 0,
                "ledgers": {},
            }

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
        monkeypatch.setattr(
            bot,
            "_current_utc",
            lambda: now_value[0],
        )
        monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_load_live_event_contexts", lambda: [])
        monkeypatch.setattr(
            bot,
            "_resolve_live_event_context",
            lambda *_args, **_kwargs: {
                "weight_class": "Welterweight",
                "is_title_bout": False,
                "is_empty_arena": False,
                "num_rounds": 3,
            },
        )
        monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
        monkeypatch.setattr(
            "src.model.train.load_model",
            lambda _name: {
                "feature_cols": [],
                "col_medians": np.array([]),
                "feature_importance": {},
                "raw_model": None,
            },
        )
        monkeypatch.setattr(
            "src.model.predict.predict_fight",
            lambda *_args, **_kwargs: {
                "prob_a": 0.61,
                "prob_b": 0.39,
                "confidence": 0.61,
            },
        )
        monkeypatch.setattr(
            "src.data.fighter_lookup.build_fight_features",
            fake_build_fight_features,
        )
        monkeypatch.setattr(
            bot,
            "_prediction_method_odds_fingerprint",
            fake_method_fingerprint,
        )
        monkeypatch.setattr(
            "src.data.line_tracker.get_line_movement_features",
            lambda *_args, **_kwargs: {},
        )
        monkeypatch.setattr(
            "src.data.line_tracker.detect_injury_or_cancellation",
            lambda *_args, **_kwargs: {"suspected": False},
        )
        monkeypatch.setattr(
            "src.polymarket.markets.get_ufc_fight_markets",
            lambda: pd.DataFrame(),
        )
        monkeypatch.setattr(
            "src.strategy.duo_trader.run_duo_traders",
            fail_if_trader_runs,
        )
        monkeypatch.setattr(
            "src.strategy.duo_trader.cancel_duo_open_limit_orders",
            fake_cancel_duo_open_limit_orders,
        )

        args = type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})()
        result = bot.cmd_duo_live(args)
        method_fingerprint[0] = "method-odds:v2"
        now_value[0] = datetime(2026, 3, 28, 19, 29, tzinfo=timezone.utc)
        retry_result = bot.cmd_duo_live(args)
        interim_payload = json.loads(
            (logs_dir / "predictions_cache.json").read_text(encoding="utf-8")
        )
        assert (
            interim_payload["predictions"][0]["data_quality_retry_after"]
            == "2026-03-28T19:30:00+00:00"
        )
        due_retry_result = bot.cmd_duo_live(args)
        original_retry_due = bot._blocked_data_quality_retry_due
        retry_boundary_calls = [0]

        def retry_deadline_crosses_between_checks(cached, *, now=None):
            retry_boundary_calls[0] += 1
            if retry_boundary_calls[0] == 2:
                now_value[0] = datetime(2026, 3, 28, 20, 32, tzinfo=timezone.utc)
            return original_retry_due(cached, now=now)

        now_value[0] = datetime(2026, 3, 28, 20, 30, tzinfo=timezone.utc)
        monkeypatch.setattr(
            bot,
            "_blocked_data_quality_retry_due",
            retry_deadline_crosses_between_checks,
        )
        boundary_retry_result = bot.cmd_duo_live(args)

        assert result["status"] == "degraded"
        assert result["total_orders"] == 0
        assert retry_result["status"] == "degraded"
        assert retry_result["total_orders"] == 0
        assert due_retry_result["status"] == "degraded"
        assert due_retry_result["total_orders"] == 0
        assert boundary_retry_result["status"] == "degraded"
        assert boundary_retry_result["total_orders"] == 0
        assert trader_calls == []
        assert force_refresh_values == [False, False, True, True]
        assert [call["reason"] for call in maintenance_calls] == [
            "live_data_quality_blocked",
            "live_data_quality_blocked",
            "live_data_quality_blocked",
            "live_data_quality_blocked",
        ]
        assert fingerprint_pairs == [
            ("Ian Garry", "Opponent Fighter"),
            ("Ian Garry", "Opponent Fighter"),
            ("Ian Garry", "Opponent Fighter"),
            ("Ian Garry", "Opponent Fighter"),
        ]
        payload = json.loads((logs_dir / "predictions_cache.json").read_text(encoding="utf-8"))
        assert len(payload["predictions"]) == 1
        prediction = payload["predictions"][0]
        assert prediction["trade_blocked"] is True
        assert prediction["data_quality_retry_after"] == "2026-03-28T21:32:00+00:00"
        assert "lower-fidelity external fallback" in prediction["trade_block_reason"]
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_cmd_duo_live_serializes_nan_fighter_context_without_crashing(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        logs_dir = temp_root / "logs"
        processed_dir.mkdir()
        raw_dir.mkdir()
        logs_dir.mkdir()

        (raw_dir / "ufc-master.csv").write_text(
            "\n".join(
                [
                    "RedFighter,BlueFighter,WeightClass,Date",
                    "Rob Font,Adrian Yanez,Bantamweight,2023-04-08",
                    "Song Yadong,Ricky Simon,Bantamweight,2023-04-29",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        class FakeOddsClient:
            def get_live_odds(self):
                return []

            def odds_to_dataframe(self, _odds):
                return pd.DataFrame()

            def get_consensus_odds(self, _odds_df):
                return pd.DataFrame(
                    [
                        {
                            "event_id": "evt-1",
                            "commence_time": "2026-03-28T20:00:00Z",
                            "fighter_a": "Ricky Simon",
                            "fighter_b": "Adrian Yanez",
                            "a_fair_prob_avg": 0.54,
                            "b_fair_prob_avg": 0.46,
                            "num_bookmakers": 8,
                        }
                    ]
                )

        def fake_build_fight_features(*args, **kwargs):
            return {
                "a_num_fights": 5,
                "b_num_fights": 6,
                "a_ko_rate": float("nan"),
                "b_ko_rate": 0.18,
                "a_sub_rate": 0.0,
                "b_sub_rate": 0.0,
                "a_dec_rate": 0.0,
                "b_dec_rate": 0.0,
                "a_roll_slpm": 0.0,
                "b_roll_slpm": 0.0,
                "a_roll_kd": 0.0,
                "b_roll_kd": 0.0,
                "a_roll_sub_avg": 0.0,
                "b_roll_sub_avg": 0.0,
                "a_roll_td_avg": 0.0,
                "b_roll_td_avg": 0.0,
                "a_total_rounds": 0.0,
                "b_total_rounds": 0.0,
                "a_roll_str_def": 0.0,
                "b_roll_str_def": 0.0,
                "a_roll_td_def": 0.0,
                "b_roll_td_def": 0.0,
                "a_roll_sapm": 0.0,
                "b_roll_sapm": 0.0,
                "a_wins": float("nan"),
                "b_wins": 10.0,
                "a_losses": 2.0,
                "b_losses": 3.0,
                "a_draws": 0.0,
                "b_draws": 1.0,
                "a_win_pct": float("nan"),
                "b_win_pct": 0.7692,
                "a_current_win_streak": float("nan"),
                "b_current_win_streak": 2.0,
                "a_lose_streak": 1.0,
                "b_lose_streak": 0.0,
                "a_days_since_last_fight": float("nan"),
                "b_days_since_last_fight": 210.5,
                "a_cage_rust": float("nan"),
                "b_cage_rust": 0.0,
            }

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
        monkeypatch.setattr(bot, "_current_utc", lambda: datetime(2026, 3, 28, 18, 30, tzinfo=timezone.utc))  # >1h before 20:00 commence (LIVE_TRADE_START_BUFFER)
        monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_load_live_event_contexts", lambda: [])
        monkeypatch.setattr(
            bot,
            "_resolve_live_event_context",
            lambda *_args, **_kwargs: {
                "weight_class": "Bantamweight",
                "is_title_bout": False,
                "is_empty_arena": False,
                "num_rounds": 3,
            },
        )
        monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
        monkeypatch.setattr(
            "src.model.train.load_model",
            lambda _name: {
                "feature_cols": [],
                "col_medians": np.array([]),
                "feature_importance": {},
                "raw_model": None,
            },
        )
        monkeypatch.setattr(
            "src.model.predict.predict_fight",
            lambda *_args, **_kwargs: {"prob_a": 0.61, "prob_b": 0.39, "confidence": 0.61},
        )
        monkeypatch.setattr("src.data.fighter_lookup.build_fight_features", fake_build_fight_features)
        monkeypatch.setattr("src.data.line_tracker.get_line_movement_features", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(
            "src.data.line_tracker.detect_injury_or_cancellation",
            lambda *_args, **_kwargs: {"suspected": False},
        )
        monkeypatch.setattr(
            "src.polymarket.markets.get_ufc_fight_markets",
            lambda: pd.DataFrame([{"slug": "ricky-simon-vs-adrian-yanez"}]),
        )
        monkeypatch.setattr(
            "src.strategy.duo_trader.run_duo_traders",
            lambda *_args, **_kwargs: {"total_orders": 0},
        )

        bot.cmd_duo_live(type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})())

        payload = json.loads((logs_dir / "predictions_cache.json").read_text(encoding="utf-8"))
        prediction = payload["predictions"][0]

        assert prediction["fighter_context"]["a_wins"] is None
        assert prediction["fighter_context"]["a_win_pct"] is None
        assert prediction["fighter_context"]["a_current_win_streak"] is None
        assert prediction["fighter_context"]["a_days_since_last_fight"] is None
        assert prediction["fighter_context"]["a_cage_rust"] is None
        assert prediction["fighter_context"]["b_wins"] == 10
        assert prediction["fighter_context"]["b_win_pct"] == 0.7692
        assert prediction["fighter_context"]["b_days_since_last_fight"] == 210.5
        assert prediction["method_stats"]["a_ko_rate"] is None
        assert prediction["method_stats"]["b_ko_rate"] == 0.18
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_cmd_duo_live_writes_empty_cache_when_all_fights_are_skipped(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        processed_dir = temp_root / "processed"
        raw_dir = temp_root / "raw"
        logs_dir = temp_root / "logs"
        processed_dir.mkdir()
        raw_dir.mkdir()
        logs_dir.mkdir()

        class FakeOddsClient:
            def get_live_odds(self):
                return []

            def odds_to_dataframe(self, _odds):
                return pd.DataFrame()

            def get_consensus_odds(self, _odds_df):
                return pd.DataFrame(
                    [
                        {
                            "event_id": "evt-1",
                            "commence_time": "2026-03-28T20:00:00Z",
                            "fighter_a": "Ricky Simon",
                            "fighter_b": "Adrian Yanez",
                            "a_fair_prob_avg": 0.54,
                            "b_fair_prob_avg": 0.46,
                            "num_bookmakers": 8,
                        }
                    ]
                )

        monkeypatch.setattr(bot, "PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr(bot, "RAW_DATA_DIR", raw_dir)
        monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
        monkeypatch.setattr(bot, "_current_utc", lambda: datetime(2026, 3, 28, 19, 55, tzinfo=timezone.utc))
        monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_load_live_event_contexts", lambda: [])
        monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
        monkeypatch.setattr(
            "src.model.train.load_model",
            lambda _name: {
                "feature_cols": [],
                "col_medians": np.array([]),
                "feature_importance": {},
                "raw_model": None,
            },
        )
        monkeypatch.setattr(
            "src.polymarket.markets.get_ufc_fight_markets",
            lambda: pd.DataFrame([{"slug": "ricky-simon-vs-adrian-yanez"}]),
        )
        monkeypatch.setattr(
            "src.strategy.duo_trader.run_duo_traders",
            lambda *_args, **_kwargs: {"total_orders": 0},
        )

        bot.cmd_duo_live(type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})())

        cache_path = logs_dir / "predictions_cache.json"
        assert cache_path.exists()
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        assert payload["predictions"] == []
        assert isinstance(payload["timestamp"], str) and payload["timestamp"]
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_real_cmd_duo_live_retires_legacy_g_before_no_market_idle(monkeypatch):
    from src.strategy import duo_trader

    temp_root = _make_repo_local_tmp_dir()
    try:
        logs_dir = temp_root / "logs"
        model_dir = temp_root / "models"
        logs_dir.mkdir()
        model_dir.mkdir()
        artifact_path = model_dir / "xgboost_model.pkl"
        artifact_path.write_text("primary", encoding="utf-8")
        model_result = _fake_model_result(artifact_path)
        fake_clob = object()
        calls = []
        completed = {
            "status": "ok",
            "cancelled": 1,
            "kept": 0,
            "reconciled": 0,
            "maintenance_incomplete": False,
        }

        class FakeOddsClient:
            def get_live_odds(self):
                return []

            def odds_to_dataframe(self, _odds):
                return pd.DataFrame()

            def get_consensus_odds(self, _odds_df):
                return pd.DataFrame()

        monkeypatch.setattr(bot, "assert_real_trading_allowed", lambda **_kwargs: None)
        monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
        monkeypatch.setattr(bot, "initialize_prediction_history", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "archive_prediction_payload", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_resolve_no_odds_model_arg", lambda *_args, **_kwargs: None)
        monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
        monkeypatch.setattr("src.model.train.load_model", lambda _name: model_result)
        monkeypatch.setattr(
            "src.polymarket.markets.get_ufc_fight_markets",
            lambda **_kwargs: pd.DataFrame(),
        )
        monkeypatch.setattr(
            duo_trader,
            "ensure_legacy_g_orders_retired",
            lambda **kwargs: calls.append(kwargs) or dict(completed),
        )
        monkeypatch.setattr(
            duo_trader,
            "run_duo_traders",
            lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("runner must not execute without markets or predictions")
            ),
        )

        result = bot.cmd_duo_live(
            type(
                "Args",
                (),
                {
                    "model": "xgboost",
                    "dry_run": False,
                    "min_edge": 0.02,
                    "clob_client": fake_clob,
                },
            )()
        )

        assert result == {
            "status": "idle",
            "reason": "no_executable_opportunities",
            "legacy_g_order_retirement": completed,
        }
        assert calls == [{"clob": fake_clob, "dry_run": False}]
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_real_cmd_duo_live_continues_after_degraded_legacy_g_cleanup(
    monkeypatch,
):
    from src.strategy import duo_trader

    summary = {
        "status": "degraded",
        "cancelled": 0,
        "kept": 1,
        "reconciled": 0,
        "maintenance_incomplete": True,
        "errors": ["open CLOB order state unavailable"],
    }
    model_loads = []

    def stop_after_model_load(*_args, **_kwargs):
        model_loads.append(True)
        raise RuntimeError("stop after model load")

    monkeypatch.setattr(bot, "assert_real_trading_allowed", lambda **_kwargs: None)
    monkeypatch.setattr(
        duo_trader,
        "ensure_legacy_g_orders_retired",
        lambda **_kwargs: dict(summary),
    )
    monkeypatch.setattr(
        bot,
        "ensure_model_fresh",
        stop_after_model_load,
    )

    with pytest.raises(RuntimeError, match="stop after model load"):
        bot.cmd_duo_live(
            type(
                "Args",
                (),
                {
                    "model": "xgboost",
                    "dry_run": False,
                    "min_edge": 0.02,
                    "clob_client": object(),
                },
            )()
        )

    assert model_loads == [True]


def test_cmd_duo_live_skips_off_card_fight_before_line_and_injury_checks(
    monkeypatch,
    caplog,
):
    temp_root = _make_repo_local_tmp_dir()
    try:
        logs_dir = temp_root / "logs"
        model_dir = temp_root / "models"
        logs_dir.mkdir()
        model_dir.mkdir()

        artifact_path = model_dir / "xgboost_model.pkl"
        artifact_path.write_text("primary", encoding="utf-8")
        model_result = _fake_model_result(artifact_path, feature_cols=["line_movement"])
        # Non-roster names: a genuinely off-card/non-UFC pair. Roster-matched UFC
        # fighters whose card never loaded would instead trip the partial-context-loss
        # degraded signal.
        fight = {
            "event_id": "stale-odds-event",
            "commence_time": "2026-06-07T00:45:00Z",
            "fighter_a": "Stale Offcard Alpha",
            "fighter_b": "Stale Offcard Beta",
            "a_fair_prob_avg": 0.56,
            "b_fair_prob_avg": 0.44,
            "num_bookmakers": 1,
        }

        class FakeOddsClient:
            def get_live_odds(self):
                return []

            def odds_to_dataframe(self, _odds):
                return pd.DataFrame()

            def get_consensus_odds(self, _odds_df):
                return pd.DataFrame([fight])

        line_calls: list[tuple] = []
        injury_calls: list[tuple] = []

        monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
        monkeypatch.setattr(bot, "_current_utc", lambda: datetime(2026, 6, 1, 21, 0, tzinfo=timezone.utc))
        monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_resolve_no_odds_model_arg", lambda *_args, **_kwargs: None)
        # Contexts load fine, including a UFC card on the same local date; this
        # particular fight is unrelated other-promotion feed noise. An empty context
        # list would instead signal a degraded cycle (upstream sources down).
        monkeypatch.setattr(
            bot,
            "_load_live_event_contexts_for_fights",
            lambda *_args, **_kwargs: [
                {
                    "event_id": "official-ufc-card",
                    "event_date": "June 6, 2026",
                    "fighter_a": "Other Alpha",
                    "fighter_b": "Other Beta",
                }
            ],
        )
        monkeypatch.setattr(bot, "_resolve_live_event_context", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            bot,
            "_missing_live_event_context_reason",
            lambda *_args, **_kwargs: "not on any upcoming UFC card",
        )
        monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
        monkeypatch.setattr("src.model.train.load_model", lambda _name: model_result)
        monkeypatch.setattr(
            "src.data.line_tracker.get_line_movement_features",
            lambda *args, **kwargs: line_calls.append((args, kwargs)) or {},
        )
        monkeypatch.setattr(
            "src.data.line_tracker.detect_injury_or_cancellation",
            lambda *args, **kwargs: injury_calls.append((args, kwargs)) or {"suspected": False},
        )
        monkeypatch.setattr("src.polymarket.markets.get_ufc_fight_markets", lambda: pd.DataFrame())
        monkeypatch.setattr("src.strategy.duo_trader.run_duo_traders", lambda **_kwargs: {"total_orders": 0})
        bot._LIVE_EVENT_SKIP_LOG_CACHE.clear()

        with caplog.at_level(logging.INFO):
            result = bot.cmd_duo_live(type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})())

        assert result == {"status": "idle", "reason": "no_executable_opportunities"}
        assert line_calls == []
        assert injury_calls == []
        skip_records = [
            record
            for record in caplog.records
            if "Skipping Stale Offcard Alpha vs Stale Offcard Beta" in record.message
        ]
        assert len(skip_records) == 1
        assert skip_records[0].levelno == logging.INFO
        payload = json.loads((logs_dir / "predictions_cache.json").read_text(encoding="utf-8"))
        assert payload["predictions"] == []
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_cmd_duo_live_reports_degraded_when_event_context_is_unavailable(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        logs_dir = temp_root / "logs"
        model_dir = temp_root / "models"
        logs_dir.mkdir()
        model_dir.mkdir()

        artifact_path = model_dir / "xgboost_model.pkl"
        artifact_path.write_text("primary", encoding="utf-8")
        model_result = _fake_model_result(artifact_path, feature_cols=["line_movement"])
        fight = {
            "event_id": "active-card-event",
            "commence_time": "2026-06-07T00:45:00Z",
            "fighter_a": "Bryce Mitchell",
            "fighter_b": "Santiago Ponzinibbio",
            "a_fair_prob_avg": 0.56,
            "b_fair_prob_avg": 0.44,
            "num_bookmakers": 1,
        }

        class FakeOddsClient:
            def get_live_odds(self):
                return []

            def odds_to_dataframe(self, _odds):
                return pd.DataFrame()

            def get_consensus_odds(self, _odds_df):
                return pd.DataFrame([fight])

        monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
        monkeypatch.setattr(bot, "_current_utc", lambda: datetime(2026, 6, 1, 21, 0, tzinfo=timezone.utc))
        monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_resolve_no_odds_model_arg", lambda *_args, **_kwargs: None)
        # Total context loss: collector and cache both came back empty.
        monkeypatch.setattr(bot, "_load_live_event_contexts_for_fights", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(bot, "_resolve_live_event_context", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_runtime_completed_ufc_event_dates_before", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            bot,
            "_missing_live_event_context_reason",
            lambda *_args, **_kwargs: "not on any upcoming UFC card",
        )
        monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
        monkeypatch.setattr("src.model.train.load_model", lambda _name: model_result)
        monkeypatch.setattr("src.polymarket.markets.get_ufc_fight_markets", lambda: pd.DataFrame())
        monkeypatch.setattr("src.strategy.duo_trader.run_duo_traders", lambda **_kwargs: {"total_orders": 0})

        result = bot.cmd_duo_live(type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})())

        assert result["status"] == "degraded"
        assert "halted" in result["reason"]
        payload = json.loads((logs_dir / "predictions_cache.json").read_text(encoding="utf-8"))
        assert payload["predictions"] == []
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_cmd_duo_live_rebuilds_when_enabled_line_feature_changes(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        logs_dir = temp_root / "logs"
        model_dir = temp_root / "models"
        logs_dir.mkdir()
        model_dir.mkdir()

        artifact_path = model_dir / "xgboost_model.pkl"
        artifact_path.write_text("primary", encoding="utf-8")
        model_result = _fake_model_result(artifact_path, feature_cols=["line_movement"])
        runtime_signature = bot._prediction_runtime_signature(
            model_result=model_result,
            no_odds_result=None,
            runtime_bundle_summary=None,
        )
        fight = {
            "event_id": "evt-1",
            "commence_time": "2026-03-28T20:00:00Z",
            "fighter_a": "Ricky Simon",
            "fighter_b": "Adrian Yanez",
            "a_fair_prob_avg": 0.55,
            "b_fair_prob_avg": 0.45,
            "num_bookmakers": 8,
        }
        cached_row = _cached_prediction_row(
            fight,
            runtime_signature=runtime_signature,
            generated_at="2026-03-28T18:00:00+00:00",
            a_market_prob=0.55,
            b_market_prob=0.45,
            line_movement=0.0,
        )
        (logs_dir / "predictions_cache.json").write_text(
            json.dumps(
                {
                    "schema_version": bot.PREDICTION_CACHE_SCHEMA_VERSION,
                    "timestamp": "2026-03-28T18:00:00+00:00",
                    "predictions": [cached_row],
                    "global_feature_importance": [],
                }
            ),
            encoding="utf-8",
        )

        class FakeOddsClient:
            def get_live_odds(self):
                return []

            def odds_to_dataframe(self, _odds):
                return pd.DataFrame()

            def get_consensus_odds(self, _odds_df):
                return pd.DataFrame(
                    [
                        {
                            **fight,
                            "a_fair_prob_avg": 0.56,
                            "b_fair_prob_avg": 0.44,
                        }
                    ]
                )

        build_calls: list[int] = []
        predict_calls: list[int] = []
        captured: dict = {}

        def fake_build_fight_features(*_args, **_kwargs):
            build_calls.append(1)
            features = _base_live_features()
            features.update({"a_num_fights": 1, "b_num_fights": 1})
            return features, {
                "fighter_a_source": "ufcstats",
                "fighter_a_fight_history_status": "complete",
                "fighter_b_source": "ufcstats",
                "fighter_b_fight_history_status": "complete",
            }

        def fake_predict_fight(*_args, **_kwargs):
            predict_calls.append(1)
            return {"prob_a": 0.63, "prob_b": 0.37, "confidence": 0.63}

        def fake_run_duo_traders(**kwargs):
            captured["predictions"] = kwargs["predictions"].copy()
            return {"total_orders": 0}

        monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
        monkeypatch.setattr(bot, "_current_utc", lambda: datetime(2026, 3, 28, 18, 30, tzinfo=timezone.utc))  # >1h before 20:00 commence (LIVE_TRADE_START_BUFFER)
        monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_resolve_no_odds_model_arg", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            bot,
            "_load_live_event_contexts_for_fights",
            lambda *_args, **_kwargs: [{}],
        )
        monkeypatch.setattr(
            bot,
            "_resolve_live_event_context",
            lambda *_args, **_kwargs: {
                "weight_class": "Bantamweight",
                "is_title_bout": False,
                "is_empty_arena": False,
                "num_rounds": 3,
            },
        )
        monkeypatch.setattr(
            "src.data.odds_client.OddsClient",
            FakeOddsClient,
        )
        monkeypatch.setattr("src.model.train.load_model", lambda _name: model_result)
        monkeypatch.setattr("src.model.predict.predict_fight", fake_predict_fight)
        monkeypatch.setattr("src.data.fighter_lookup.build_fight_features", fake_build_fight_features)
        monkeypatch.setattr(
            "src.data.line_tracker.get_line_movement_features",
            lambda *_args, **_kwargs: {"line_movement": 0.01, "line_is_sharp": 0, "line_steam_move": 0},
        )
        monkeypatch.setattr(
            "src.data.line_tracker.detect_injury_or_cancellation",
            lambda *_args, **_kwargs: {"suspected": False},
        )
        monkeypatch.setattr(
            "src.polymarket.markets.get_ufc_fight_markets",
            lambda: pd.DataFrame([{"slug": "ricky-simon-vs-adrian-yanez"}]),
        )
        monkeypatch.setattr("src.strategy.duo_trader.run_duo_traders", fake_run_duo_traders)

        result = bot.cmd_duo_live(type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})())

        assert result == {"status": "ok", "total_orders": 0}
        assert build_calls == [1]
        assert predict_calls == [1]
        assert float(captured["predictions"].iloc[0]["a_market_prob"]) == 0.56
        assert float(captured["predictions"].iloc[0]["line_movement"]) == 0.01

        payload = json.loads((logs_dir / "predictions_cache.json").read_text(encoding="utf-8"))
        prediction = payload["predictions"][0]
        assert prediction["prediction_generated_at"] != "2026-03-28T18:00:00+00:00"
        assert prediction["a_market_prob"] == 0.56
        assert prediction["odds_snapshot"]["a_fair_prob_avg"] == 0.56
        assert prediction["prediction_input_odds_snapshot"]["a_fair_prob_avg"] == 0.56
        assert prediction["prediction_input_line_features"]["line_movement"] == 0.01
        assert prediction["line_movement"] == 0.01
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_cmd_duo_live_keeps_full_cache_visible_while_refreshing_single_fight(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        logs_dir = temp_root / "logs"
        model_dir = temp_root / "models"
        logs_dir.mkdir()
        model_dir.mkdir()

        artifact_path = model_dir / "xgboost_model.pkl"
        artifact_path.write_text("primary", encoding="utf-8")
        model_result = _fake_model_result(artifact_path, feature_cols=["line_movement"])
        runtime_signature = bot._prediction_runtime_signature(
            model_result=model_result,
            no_odds_result=None,
            runtime_bundle_summary=None,
        )
        fight_one = {
            "event_id": "evt-1",
            "commence_time": "2026-03-28T20:00:00Z",
            "fighter_a": "Ricky Simon",
            "fighter_b": "Adrian Yanez",
            "a_fair_prob_avg": 0.55,
            "b_fair_prob_avg": 0.45,
            "num_bookmakers": 8,
        }
        fight_two = {
            "event_id": "evt-2",
            "commence_time": "2026-03-28T21:00:00Z",
            "fighter_a": "Rob Font",
            "fighter_b": "Marlon Vera",
            "a_fair_prob_avg": 0.40,
            "b_fair_prob_avg": 0.60,
            "num_bookmakers": 8,
        }
        cached_predictions = [
            _cached_prediction_row(
                fight_one,
                runtime_signature=runtime_signature,
                generated_at="2026-03-28T18:00:00+00:00",
                a_market_prob=0.55,
                b_market_prob=0.45,
                line_movement=0.02,
            ),
            _cached_prediction_row(
                fight_two,
                runtime_signature=runtime_signature,
                generated_at="2026-03-28T18:00:00+00:00",
                a_market_prob=0.40,
                b_market_prob=0.60,
                line_movement=0.02,
            ),
        ]
        cached_predictions[0]["model_features"]["cached_model_marker"] = 17
        cached_predictions[0]["feature_provenance"]["cached_source_marker"] = (
            "persisted-cache"
        )
        (logs_dir / "predictions_cache.json").write_text(
            json.dumps(
                {
                    "schema_version": bot.PREDICTION_CACHE_SCHEMA_VERSION,
                    "timestamp": "2026-03-28T18:00:00+00:00",
                    "predictions": cached_predictions,
                    "global_feature_importance": [],
                }
            ),
            encoding="utf-8",
        )

        class FakeOddsClient:
            def get_live_odds(self):
                return []

            def odds_to_dataframe(self, _odds):
                return pd.DataFrame()

            def get_consensus_odds(self, _odds_df):
                return pd.DataFrame(
                    [
                        {
                            **fight_one,
                            "a_fair_prob_avg": 0.56,
                            "b_fair_prob_avg": 0.44,
                        },
                        {
                            **fight_two,
                            "a_fair_prob_avg": 0.47,
                            "b_fair_prob_avg": 0.53,
                        },
                    ]
                )

        build_calls: list[str] = []
        predict_calls: list[str] = []
        context_loads: list[int] = []
        cache_write_lengths: list[int] = []
        cache_write_metadata: list[tuple[bool, object]] = []

        original_write_text = Path.write_text

        def recording_write_text(path_obj, data, *args, **kwargs):
            if path_obj.name == "predictions_cache.json.tmp":
                written_payload = json.loads(data)
                cache_write_lengths.append(len(written_payload["predictions"]))
                cache_write_metadata.append(
                    (
                        bool(written_payload.get("refresh_in_progress")),
                        written_payload.get("timestamp"),
                    )
                )
            return original_write_text(path_obj, data, *args, **kwargs)

        def fake_build_fight_features(fighter_a, fighter_b, **_kwargs):
            build_calls.append(f"{fighter_a}|{fighter_b}")
            features = _base_live_features()
            features.update({"a_num_fights": 1, "b_num_fights": 1})
            return features, {
                "fighter_a_source": "ufcstats",
                "fighter_a_fight_history_status": "complete",
                "fighter_b_source": "ufcstats",
                "fighter_b_fight_history_status": "complete",
            }

        def fake_predict_fight(features, **_kwargs):
            predict_calls.append("predict")
            return {"prob_a": 0.58, "prob_b": 0.42, "confidence": 0.58}

        def fake_run_duo_traders(**_kwargs):
            return {"total_orders": 0}

        monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
        monkeypatch.setattr(bot, "_current_utc", lambda: datetime(2026, 3, 28, 18, 30, tzinfo=timezone.utc))  # >1h before 20:00 commence (LIVE_TRADE_START_BUFFER)
        monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_resolve_no_odds_model_arg", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(Path, "write_text", recording_write_text)
        monkeypatch.setattr("src.data.odds_client.OddsClient", FakeOddsClient)
        monkeypatch.setattr("src.model.train.load_model", lambda _name: model_result)
        monkeypatch.setattr("src.model.predict.predict_fight", fake_predict_fight)
        monkeypatch.setattr("src.data.fighter_lookup.build_fight_features", fake_build_fight_features)
        monkeypatch.setattr(
            bot,
            "_load_live_event_contexts_for_fights",
            lambda *_args, **_kwargs: context_loads.append(1) or [{}],
        )
        monkeypatch.setattr(
            bot,
            "_resolve_live_event_context",
            lambda *_args, **_kwargs: {
                "weight_class": "Bantamweight",
                "is_title_bout": False,
                "is_empty_arena": False,
                "num_rounds": 3,
            },
        )
        monkeypatch.setattr(
            "src.data.line_tracker.get_line_movement_features",
            lambda *_args, **_kwargs: {"line_movement": 0.02, "line_is_sharp": 0, "line_steam_move": 0},
        )
        monkeypatch.setattr(
            "src.data.line_tracker.detect_injury_or_cancellation",
            lambda *_args, **_kwargs: {"suspected": False},
        )
        monkeypatch.setattr(
            "src.polymarket.markets.get_ufc_fight_markets",
            lambda: pd.DataFrame([{"slug": "ufc-card"}]),
        )
        monkeypatch.setattr(
            "src.strategy.duo_trader.run_duo_traders",
            fake_run_duo_traders,
        )

        bot.cmd_duo_live(type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})())

        assert build_calls == ["Rob Font|Marlon Vera"]
        assert predict_calls == ["predict"]
        assert context_loads == [1]
        assert cache_write_lengths
        assert all(length == 2 for length in cache_write_lengths)
        assert all(
            timestamp == "2026-03-28T18:00:00+00:00"
            for in_progress, timestamp in cache_write_metadata[:-1]
            if in_progress
        )
        assert cache_write_metadata[-1][0] is False
        assert cache_write_metadata[-1][1] != "2026-03-28T18:00:00+00:00"

        payload = json.loads((logs_dir / "predictions_cache.json").read_text(encoding="utf-8"))
        assert len(payload["predictions"]) == 2
        assert payload["refresh_in_progress"] is False
        cached = next(row for row in payload["predictions"] if row["fighter_a"] == "Ricky Simon")
        assert cached["model_features"]["cached_model_marker"] == 17
        assert cached["feature_provenance"]["cached_source_marker"] == "persisted-cache"
        refreshed = next(row for row in payload["predictions"] if row["fighter_a"] == "Rob Font")
        assert refreshed["prediction_generated_at"] != "2026-03-28T18:00:00+00:00"
        assert refreshed["a_market_prob"] == 0.47
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_cmd_duo_live_prunes_removed_cached_predictions(monkeypatch):
    temp_root = _make_repo_local_tmp_dir()
    try:
        logs_dir = temp_root / "logs"
        model_dir = temp_root / "models"
        logs_dir.mkdir()
        model_dir.mkdir()

        artifact_path = model_dir / "xgboost_model.pkl"
        artifact_path.write_text("primary", encoding="utf-8")
        model_result = _fake_model_result(artifact_path, feature_cols=["line_movement"])
        runtime_signature = bot._prediction_runtime_signature(
            model_result=model_result,
            no_odds_result=None,
            runtime_bundle_summary=None,
        )
        current_fight = {
            "event_id": "evt-1",
            "commence_time": "2026-03-28T20:00:00Z",
            "fighter_a": "Ricky Simon",
            "fighter_b": "Adrian Yanez",
            "a_fair_prob_avg": 0.55,
            "b_fair_prob_avg": 0.45,
            "num_bookmakers": 8,
        }
        removed_fight = {
            "event_id": "evt-2",
            "commence_time": "2026-03-28T21:00:00Z",
            "fighter_a": "Rob Font",
            "fighter_b": "Marlon Vera",
            "a_fair_prob_avg": 0.40,
            "b_fair_prob_avg": 0.60,
            "num_bookmakers": 8,
        }
        (logs_dir / "predictions_cache.json").write_text(
            json.dumps(
                {
                    "schema_version": bot.PREDICTION_CACHE_SCHEMA_VERSION,
                    "timestamp": "2026-03-28T18:00:00+00:00",
                    "predictions": [
                        _cached_prediction_row(
                            current_fight,
                            runtime_signature=runtime_signature,
                            generated_at="2026-03-28T18:00:00+00:00",
                            line_movement=0.01,
                        ),
                        _cached_prediction_row(
                            removed_fight,
                            runtime_signature=runtime_signature,
                            generated_at="2026-03-28T18:00:00+00:00",
                        ),
                    ],
                    "global_feature_importance": [],
                }
            ),
            encoding="utf-8",
        )

        class FakeOddsClient:
            def get_live_odds(self):
                return []

            def odds_to_dataframe(self, _odds):
                return pd.DataFrame()

            def get_consensus_odds(self, _odds_df):
                return pd.DataFrame([current_fight])

        monkeypatch.setattr(bot, "LOGS_DIR", logs_dir)
        monkeypatch.setattr(bot, "_current_utc", lambda: datetime(2026, 3, 28, 18, 30, tzinfo=timezone.utc))  # >1h before 20:00 commence (LIVE_TRADE_START_BUFFER)
        monkeypatch.setattr(bot, "ensure_model_fresh", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(bot, "_resolve_no_odds_model_arg", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            bot,
            "_load_live_event_contexts_for_fights",
            lambda *_args, **_kwargs: [{}],
        )
        monkeypatch.setattr(
            bot,
            "_resolve_live_event_context",
            lambda *_args, **_kwargs: {
                "weight_class": "Bantamweight",
                "is_title_bout": False,
                "is_empty_arena": False,
                "num_rounds": 3,
            },
        )
        monkeypatch.setattr(
            "src.data.odds_client.OddsClient",
            FakeOddsClient,
        )
        monkeypatch.setattr("src.model.train.load_model", lambda _name: model_result)
        monkeypatch.setattr(
            "src.data.line_tracker.get_line_movement_features",
            lambda *_args, **_kwargs: {"line_movement": 0.01, "line_is_sharp": 0, "line_steam_move": 0},
        )
        monkeypatch.setattr(
            "src.data.line_tracker.detect_injury_or_cancellation",
            lambda *_args, **_kwargs: {"suspected": False},
        )
        monkeypatch.setattr(
            "src.polymarket.markets.get_ufc_fight_markets",
            lambda: pd.DataFrame([{"slug": "ufc-card"}]),
        )
        monkeypatch.setattr(
            "src.strategy.duo_trader.run_duo_traders",
            lambda **_kwargs: {"total_orders": 0},
        )

        bot.cmd_duo_live(type("Args", (), {"model": "xgboost", "dry_run": True, "min_edge": 0.02})())

        payload = json.loads((logs_dir / "predictions_cache.json").read_text(encoding="utf-8"))
        assert len(payload["predictions"]) == 1
        assert payload["predictions"][0]["fighter_a"] == "Ricky Simon"
        assert payload["predictions"][0]["fighter_b"] == "Adrian Yanez"

        history_payload = json.loads(
            (logs_dir / "predictions_history.json").read_text(encoding="utf-8")
        )
        archived_matchups = {
            (row["fighter_a"], row["fighter_b"])
            for row in history_payload["predictions"]
        }
        assert ("Ricky Simon", "Adrian Yanez") in archived_matchups
        assert ("Rob Font", "Marlon Vera") in archived_matchups
        removed_archive = next(
            row
            for row in history_payload["predictions"]
            if row["fighter_a"] == "Rob Font"
        )
        assert removed_archive["predicted_winner"] == "Rob Font"
        assert "runtime_signature" not in removed_archive
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_load_live_event_contexts_reuses_matching_cached_card_within_ttl(monkeypatch):
    from src.data import live_monitor

    cached_contexts = [
        {
            "event_id": "evt-1",
            "commence_time": "2026-03-28T20:00:00+00:00",
            "event_date": "March 28, 2026",
        }
    ]

    def _fail_collect(expected_fights=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(live_monitor, "collect_upcoming_fight_contexts", _fail_collect)
    monkeypatch.setattr(bot, "_LAST_GOOD_LIVE_EVENT_CONTEXTS", (100.0, ("event:evt-1",), cached_contexts))
    monkeypatch.setattr(bot.time, "monotonic", lambda: 110.0)

    result = bot._load_live_event_contexts(
        [
            {
                "event_id": "evt-1",
                "commence_time": "2026-03-28T20:00:00+00:00",
            }
        ]
    )

    assert result == cached_contexts


def test_load_live_event_contexts_reuses_cached_card_subset_within_ttl(monkeypatch):
    from src.data import live_monitor

    cached_contexts = [
        {
            "event_id": "evt-1",
            "commence_time": "2026-03-28T20:00:00+00:00",
            "event_date": "March 28, 2026",
            "fighter_a": "Alpha Fighter",
            "fighter_b": "Beta Fighter",
            "weight_class": "Lightweight",
        }
    ]
    collect_calls = {"count": 0}

    def _empty_collect(expected_fights=None):
        collect_calls["count"] += 1
        return []

    monkeypatch.setattr(live_monitor, "collect_upcoming_fight_contexts", _empty_collect)
    monkeypatch.setattr(bot, "_LAST_GOOD_LIVE_EVENT_CONTEXTS", (100.0, ("event:evt-1",), cached_contexts))
    monkeypatch.setattr(bot.time, "monotonic", lambda: 110.0)

    result = bot._load_live_event_contexts(
        [
            {
                "event_id": "evt-1",
                "commence_time": "2026-03-28T20:00:00+00:00",
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
            },
            {
                "event_id": "evt-2",
                "commence_time": "2026-03-28T21:00:00+00:00",
                "fighter_a": "Gamma Fighter",
                "fighter_b": "Delta Fighter",
            },
        ]
    )

    assert collect_calls["count"] == 1
    assert result == cached_contexts


def test_load_live_event_contexts_rejects_mismatched_cached_card(monkeypatch):
    from src.data import live_monitor

    def _fail_collect(expected_fights=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(live_monitor, "collect_upcoming_fight_contexts", _fail_collect)
    monkeypatch.setattr(
        bot,
        "_LAST_GOOD_LIVE_EVENT_CONTEXTS",
        (
            100.0,
            ("event:evt-1",),
            [{"event_id": "evt-1", "commence_time": "2026-03-28T20:00:00+00:00"}],
        ),
    )
    monkeypatch.setattr(bot.time, "monotonic", lambda: 110.0)

    result = bot._load_live_event_contexts(
        [
            {
                "event_id": "evt-2",
                "commence_time": "2026-04-04T20:00:00+00:00",
            }
        ]
    )

    assert result == []


def test_log_live_fight_skip_once_dedupes_non_ufc_noise(monkeypatch, caplog):
    bot._LIVE_EVENT_SKIP_LOG_CACHE.clear()
    monotonic_values = iter([100.0, 101.0])
    monkeypatch.setattr(bot.time, "monotonic", lambda: next(monotonic_values))

    fight = {
        "fighter_a": "Masayuki Kikuiri",
        "fighter_b": "Ernesto Rodriguez",
        "event_id": "evt-1",
        "commence_time": "2026-03-27T23:00:00+00:00",
    }

    with caplog.at_level(logging.INFO):
        bot._log_live_fight_skip_once(fight, bot._NON_UFC_LIVE_CONTEXT_REASON)
        bot._log_live_fight_skip_once(fight, bot._NON_UFC_LIVE_CONTEXT_REASON)

    skip_records = [
        record for record in caplog.records
        if "Skipping Masayuki Kikuiri vs Ernesto Rodriguez" in record.message
    ]
    assert len(skip_records) == 1
    assert skip_records[0].levelno == logging.INFO


def test_log_live_fight_skip_once_treats_missing_upcoming_card_skip_as_info(monkeypatch, caplog):
    bot._LIVE_EVENT_SKIP_LOG_CACHE.clear()
    monkeypatch.setattr(bot.time, "monotonic", lambda: 100.0)

    fight = {
        "fighter_a": "Merab Dvalishvili",
        "fighter_b": "Petr Yan",
        "event_id": "evt-2",
        "commence_time": "2026-06-28T02:00:00+00:00",
    }
    reason = (
        "pair already exists in local UFC history (2025-12-06) "
        "but is not on any upcoming UFC card"
    )

    with caplog.at_level(logging.INFO):
        bot._log_live_fight_skip_once(fight, reason)

    skip_records = [
        record for record in caplog.records
        if "Skipping Merab Dvalishvili vs Petr Yan" in record.message
    ]
    assert len(skip_records) == 1
    assert skip_records[0].levelno == logging.INFO


def test_log_live_fight_skip_once_escalates_loaded_card_identity_mismatch(
    monkeypatch,
    caplog,
):
    bot._LIVE_EVENT_SKIP_LOG_CACHE.clear()
    monotonic_values = iter([455.0, 456.0])
    monkeypatch.setattr(bot.time, "monotonic", lambda: next(monotonic_values))
    fight = {
        "event_id": "odds-current-card",
        "commence_time": "2026-08-01T16:50:00Z",
        "fighter_a": "L'udovit Kline",
        "fighter_b": "Tofiq Musayev",
    }
    contexts = [{
        "event_id": "ufc-current-card",
        "event_date": "August 1, 2026",
        "fighter_a": "Ludovít Klein",
        "fighter_b": "Tofiq Musayev",
    }]

    with caplog.at_level(logging.INFO):
        bot._log_live_fight_skip_once(
            fight,
            "not on any upcoming UFC card",
        )
        bot._log_live_fight_skip_once(
            fight,
            "not on any upcoming UFC card",
            force_warning=bot._plausible_live_card_identity_mismatch(fight, contexts),
        )

    skip_records = [
        record
        for record in caplog.records
        if "Skipping L'udovit Kline vs Tofiq Musayev" in record.message
    ]
    assert len(skip_records) == 2
    assert [record.levelno for record in skip_records] == [
        logging.INFO,
        logging.WARNING,
    ]


def test_log_live_fight_skip_once_treats_safety_buffer_skip_as_info(monkeypatch, caplog):
    bot._LIVE_EVENT_SKIP_LOG_CACHE.clear()
    monkeypatch.setattr(bot.time, "monotonic", lambda: 100.0)

    fight = {
        "fighter_a": "Ming Shi",
        "fighter_b": "Puja Tomar",
        "event_id": "evt-3",
        "commence_time": "2026-05-29T13:40:00+00:00",
    }

    with caplog.at_level(logging.INFO):
        bot._log_live_fight_skip_once(
            fight,
            "fight starts at 2026-05-29T13:40:00+00:00 (safety buffer 1:00:00)",
        )

    skip_records = [
        record for record in caplog.records
        if "Skipping Ming Shi vs Puja Tomar" in record.message
    ]
    assert len(skip_records) == 1
    assert skip_records[0].levelno == logging.INFO


@pytest.mark.parametrize(
    "coverage_value",
    ["0.0", "0.00", "0e0", "not-a-boolean"],
)
def test_official_roster_weight_class_rejects_nontrue_coverage(
    tmp_path,
    monkeypatch,
    coverage_value,
):
    from src.data import ufc_active_roster

    roster_path = tmp_path / "ufc_active_roster_official.csv"
    pd.DataFrame(
        [
            {
                "official_name": "Blocked Fighter",
                "profile_division": "Featherweight Division",
                "profile_status": "Active",
                "combat_sport": "",
                "official_url_identity_status": "valid",
                "official_url_identity_valid": True,
                "coverage_eligible": coverage_value,
            }
        ]
    ).to_csv(roster_path, index=False)
    monkeypatch.setattr(
        ufc_active_roster,
        "OFFICIAL_ACTIVE_ROSTER_PATH",
        roster_path,
    )

    assert bot._official_roster_weight_class("Blocked Fighter") is None


@pytest.mark.parametrize("false_value", [0.0, "0.0", "0.00", "0e0"])
def test_live_alias_map_rejects_numeric_false_identity_flags(false_value):
    assert not bot._official_url_identity_trusted(
        {
            "official_url_identity_status": "valid",
            "official_url_identity_valid": false_value,
        }
    )
