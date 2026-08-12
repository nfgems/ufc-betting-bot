from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import bot
from src.data import event_context, fighter_lookup
from src.features import build_features as build_features_module


@pytest.mark.parametrize("missing_flag", [None, np.nan, np.inf, "unknown"])
def test_live_title_bout_total_fails_closed_on_unknown_history(
    monkeypatch,
    missing_flag,
):
    monkeypatch.setattr(fighter_lookup, "_seeded_scrape_rolling", lambda *_args: None)
    fights = [
        {
            "event_date": "2025-01-01",
            "opponent": "One",
            "result": "win",
            "round_finished": 1,
            "is_title_bout": True,
        },
        {
            "event_date": "2025-02-01",
            "opponent": "Two",
            "result": "loss",
            "round_finished": 2,
            "is_title_bout": missing_flag,
        },
    ]

    features = fighter_lookup._compute_rolling_for_fighter(
        fights,
        {"name": "Alpha", "dob": None},
        fighter_name="Alpha",
        reference_date="2025-03-01",
        strict_mode=True,
    )

    assert np.isnan(features["title_bouts"])


def test_title_bout_totals_count_observed_flags_and_propagate_unknown(monkeypatch):
    monkeypatch.setattr(fighter_lookup, "_seeded_scrape_rolling", lambda *_args: None)
    live = fighter_lookup._compute_rolling_for_fighter(
        [
            {
                "event_date": "2025-01-01",
                "opponent": "One",
                "result": "win",
                "round_finished": 1,
                "is_title_bout": False,
            },
            {
                "event_date": "2025-02-01",
                "opponent": "Two",
                "result": "win",
                "round_finished": 1,
                "is_title_bout": True,
            },
        ],
        {"name": "Alpha", "dob": None},
        fighter_name="Alpha",
        reference_date="2025-03-01",
        strict_mode=True,
    )
    assert live["title_bouts"] == pytest.approx(1.0)

    career = build_features_module._career_history_columns(
        pd.DataFrame(
            [
                {"title_bout": True, "result_label": "win"},
                {"title_bout": np.nan, "result_label": "loss"},
                {"title_bout": False, "result_label": "win"},
            ]
        )
    )
    assert career[7][0] == pytest.approx(0.0)
    assert career[7][1] == pytest.approx(1.0)
    assert np.isnan(career[7][2])


@pytest.mark.parametrize(
    ("title", "arena", "rounds"),
    [
        (np.nan, np.nan, np.nan),
        (np.inf, -np.inf, np.inf),
        ("unknown", "unknown", 4.5),
    ],
)
def test_live_context_rejects_malformed_nullable_values(title, arena, rounds):
    resolved = bot._resolve_live_event_context(
        {
            "event_id": "event",
            "fighter_a": "Alpha",
            "fighter_b": "Beta",
        },
        [
            {
                "event_id": "event",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "weight_class": "Lightweight",
                "is_title_bout": title,
                "is_empty_arena": arena,
                "num_rounds": rounds,
            }
        ],
        allow_off_card_history_fallback=False,
    )

    assert resolved is not None
    assert resolved["is_title_bout"] is None
    assert resolved["is_empty_arena"] is None
    assert resolved["num_rounds"] is None


def test_live_context_preserves_observed_false_zero_and_valid_rounds():
    resolved = bot._resolve_live_event_context(
        {"event_id": "event", "fighter_a": "Alpha", "fighter_b": "Beta"},
        [
            {
                "event_id": "event",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "weight_class": "Lightweight",
                "is_title_bout": "false",
                "is_empty_arena": 0.0,
                "num_rounds": "5.0",
            }
        ],
        allow_off_card_history_fallback=False,
    )

    assert resolved["is_title_bout"] is False
    assert resolved["is_empty_arena"] is False
    assert resolved["num_rounds"] == 5


def test_fighter_builder_rejects_malformed_context_but_keeps_literal_zero(
    monkeypatch,
):
    monkeypatch.setattr(
        fighter_lookup,
        "_call_lookup_fighter",
        lambda name, **_kwargs: {
            "profile": {"name": name},
            "features": {},
            "fights": [],
        },
    )
    spec = SimpleNamespace(
        name="nullable_context_test",
        feature_cols=["is_title_bout", "is_empty_arena", "num_rounds_feat"],
    )

    malformed = fighter_lookup.build_fight_features(
        "Alpha",
        "Beta",
        is_title_bout=np.nan,
        is_empty_arena="unknown",
        num_rounds=3.5,
        training_spec=spec,
    )
    observed = fighter_lookup.build_fight_features(
        "Alpha",
        "Beta",
        is_title_bout=False,
        is_empty_arena=0.0,
        num_rounds=3,
        training_spec=spec,
    )

    assert all(
        np.isnan(malformed[column])
        for column in ("is_title_bout", "is_empty_arena", "num_rounds_feat")
    )
    assert observed["is_title_bout"] == pytest.approx(0.0)
    assert observed["is_empty_arena"] == pytest.approx(0.0)
    assert observed["num_rounds_feat"] == pytest.approx(3.0)


def test_incomplete_non_apex_context_stays_unknown():
    assert np.isnan(event_context.infer_empty_arena("UFC Fight Night: A vs B", None))
    assert np.isnan(
        event_context.infer_empty_arena(None, "Las Vegas, Nevada, USA")
    )
    assert event_context.infer_empty_arena(
        "UFC 300",
        "Kaseya Center, Miami, Florida, USA",
    ) == pytest.approx(0.0)


@pytest.mark.parametrize("field", ["fighter_a_id", "fighter_b_id"])
def test_prediction_cache_refreshes_when_scoped_fighter_id_changes(field):
    fight = {
        "event_id": "event",
        "commence_time": "2026-08-15T22:00:00Z",
        "fighter_a": "Alpha",
        "fighter_b": "Beta",
    }
    cached_context = {
        "weight_class": "Lightweight",
        "num_rounds": 3,
        "is_title_bout": False,
        "is_empty_arena": False,
        "fighter_a_id": "aaaaaaaaaaaaaaaa",
        "fighter_b_id": "bbbbbbbbbbbbbbbb",
    }
    current_context = dict(cached_context)
    current_context[field] = "cccccccccccccccc"
    cached = {
        "cache_key": bot._prediction_cache_key(fight),
        "fighter_a": "Alpha",
        "fighter_b": "Beta",
        "pair_key": bot._live_fight_pair_key("Alpha", "Beta"),
        "runtime_signature": {},
        "method_odds_fingerprint": "method-odds:not-requested",
        "event_context_snapshot": cached_context,
    }

    assert bot._prediction_needs_refresh(
        cached,
        fight,
        runtime_signature={},
        method_odds_fingerprint="method-odds:not-requested",
        current_event_context_snapshot=current_context,
        current_line_feature_snapshot={},
    ) == (True, f"event context changed: {field}")
