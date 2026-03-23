from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import scripts.build_pre_ufc_career_supplement as build_pre_ufc
from src.data import pre_ufc_scraper
from src.model.feature_provenance import classify_feature_family, describe_live_lineage
from src.model.predict import _resolve_impute_strategy


def test_resolve_impute_strategy_uses_embedded_training_spec():
    model_result = {
        "training_spec": {"impute_strategy": "native_nan"},
    }
    assert _resolve_impute_strategy(model_result) == "native_nan"


def test_resolve_impute_strategy_rejects_missing_artifact_metadata():
    with pytest.raises(ValueError, match="missing a valid impute_strategy"):
        _resolve_impute_strategy({})


def test_resolve_impute_strategy_uses_legacy_logistic_pipeline_fallback():
    model_result = {
        "model": Pipeline([]),
    }
    assert _resolve_impute_strategy(model_result) == "median"


def test_load_existing_rows_preserves_non_targeted_fighters(tmp_path, monkeypatch):
    output_path = tmp_path / "pre_ufc_career_supplement_v2.csv"
    pd.DataFrame(
        [
            {"fighter_a": "Alpha", "fighter_b": "Opp A", "event_date": "2020-01-01"},
            {"fighter_a": "Beta", "fighter_b": "Opp B", "event_date": "2021-01-01"},
        ]
    ).to_csv(output_path, index=False)
    monkeypatch.setattr(build_pre_ufc, "OUTPUT_PATH", output_path)

    rows = build_pre_ufc._load_existing_rows(replace_fighters={"Alpha"})

    assert len(rows) == 1
    assert rows[0]["fighter_a"] == "Beta"


def test_tapology_pre_ufc_scrape_skips_profile_name_mismatch(monkeypatch):
    monkeypatch.setattr(pre_ufc_scraper, "search_sherdog", lambda _name: None)
    monkeypatch.setattr(pre_ufc_scraper, "search_tapology", lambda _name: "https://example.com/tapology/fighter")
    monkeypatch.setattr(
        pre_ufc_scraper,
        "search_tapology_candidates",
        lambda _name, limit=5: ["https://example.com/tapology/fighter"],
    )
    monkeypatch.setattr(
        pre_ufc_scraper,
        "scrape_tapology_profile",
        lambda _url: {"name": "Wrong Fighter"},
    )

    called = {"fights": 0}

    def _fake_scrape_tapology_fights(_url: str, _fighter_name: str) -> list[dict]:
        called["fights"] += 1
        return []

    monkeypatch.setattr(pre_ufc_scraper, "scrape_tapology_fights", _fake_scrape_tapology_fights)

    rows = pre_ufc_scraper.scrape_fighter_pre_ufc_fights("Right Fighter", "2024-01-01")

    assert rows == []
    assert called["fights"] == 0


@pytest.mark.parametrize(
    ("feature_name", "expected_family"),
    [
        ("a_opp_strength", "experimental"),
        ("a_ko_absorption", "experimental"),
        ("a_strikes_avoided_pct", "experimental"),
        ("a_age_over_35", "experimental"),
        ("a_pre_ufc_total_fights", "pre_ufc_history"),
        ("diff_pre_ufc_win_pct", "pre_ufc_history"),
        ("pace_mismatch", "experimental"),
    ],
)
def test_v6_features_have_explicit_provenance_families(feature_name, expected_family):
    assert classify_feature_family(feature_name) == expected_family


def test_pre_ufc_live_lineage_uses_supplement_artifact_description():
    assert (
        describe_live_lineage("a_pre_ufc_total_fights")
        == "local processed history plus supplemental pre-UFC fight-history artifact"
    )
