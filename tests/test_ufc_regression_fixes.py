from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import scripts.build_pre_ufc_career_supplement as build_pre_ufc
import src.data.fallback_scrapers as fallback_scrapers
import src.features.build_features as build_features_module
from src.data import pre_ufc_scraper
from src.model import training_spec
from src.model.feature_provenance import (
    classify_feature_family,
    describe_live_lineage,
    describe_training_lineage,
)
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


def test_scraper_identity_accepts_middle_name_variants():
    assert pre_ufc_scraper._same_fighter_identity("Keifer Roberts", "Keifer Michael Roberts")


def test_scrape_tapology_fights_accepts_amateur_division_alias(monkeypatch):
    html = """
    <div data-bout-id="1" data-sport="mma" data-division="amateur" data-status="loss">
      <div class="result"><div>L</div><div>SUB</div></div>
      <a href="/fightcenter/fighters/123-opponent" title="Opponent Fighter Page">Luke Nelson</a>
      <a href="/fightcenter/bouts/456-test-bout" title="Bout Page">Rear Naked Choke · 2:19 · R1</a>
      <a href="/fightcenter/events/789-test-event" title="Event Page">Caged Madness 41</a>
      <a href="/fightcenter/events/789-test-event">2016 Feb 27</a>
    </div>
    """

    monkeypatch.setattr(
        fallback_scrapers,
        "_get_tapology_soup",
        lambda _url: fallback_scrapers.BeautifulSoup(html, "lxml"),
    )

    fights = fallback_scrapers.scrape_tapology_fights(
        "https://example.com/fighter",
        "Keifer Roberts",
        division="am",
    )

    assert len(fights) == 1
    assert fights[0]["opponent"] == "Luke Nelson"
    assert fights[0]["result"] == "loss"


@pytest.mark.parametrize(
    ("feature_name", "expected_family"),
    [
        ("a_opp_strength", "experimental"),
        ("a_ko_absorption", "experimental"),
        ("a_strikes_avoided_pct", "experimental"),
        ("a_age_over_35", "experimental"),
        ("a_pre_ufc_total_fights", "pre_ufc_history"),
        ("diff_pre_ufc_win_pct", "pre_ufc_history"),
        ("a_amateur_total_fights", "amateur_history"),
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


def test_amateur_training_and_live_lineage_use_supplement_artifact_descriptions():
    assert (
        describe_training_lineage("a_amateur_total_fights")
        == "aggregated from supplemental amateur fight-history rows"
    )
    assert (
        describe_live_lineage("a_amateur_total_fights")
        == "local processed history plus supplemental amateur fight-history artifact"
    )


def test_compute_amateur_summary_dedupes_mirrored_rows():
    raw = pd.DataFrame(
        [
            {
                "event_date": "2021-01-01",
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "winner": "Alpha Fighter",
                "method": "KO/TKO",
                "finish_round": 1,
                "title_bout": 0,
                "organization": "State Amateur League",
                "weight_class": "Lightweight",
            },
            {
                "event_date": "2021-01-01",
                "fighter_a": "Beta Fighter",
                "fighter_b": "Alpha Fighter",
                "winner": "Alpha Fighter",
                "method": "KO/TKO",
                "finish_round": 1,
                "title_bout": 0,
                "organization": "State Amateur League",
                "weight_class": "Lightweight",
            },
            {
                "event_date": "2021-03-01",
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Gamma Fighter",
                "winner": "Gamma Fighter",
                "method": "Decision - Unanimous",
                "finish_round": 3,
                "title_bout": 0,
                "organization": "State Amateur League",
                "weight_class": "Lightweight",
            },
        ]
    )

    summary = build_features_module._compute_amateur_summary(raw).set_index("fighter")

    assert set(summary.columns) == {
        "amateur_total_fights",
        "amateur_wins",
        "amateur_losses",
        "amateur_win_pct",
        "amateur_ko_rate",
        "amateur_sub_rate",
        "amateur_dec_rate",
    }
    assert summary.loc["Alpha Fighter", "amateur_total_fights"] == pytest.approx(2.0)
    assert summary.loc["Alpha Fighter", "amateur_wins"] == pytest.approx(1.0)
    assert summary.loc["Alpha Fighter", "amateur_losses"] == pytest.approx(1.0)
    assert summary.loc["Beta Fighter", "amateur_total_fights"] == pytest.approx(1.0)


def test_build_features_adds_nan_amateur_columns_when_fighter_has_no_amateur_data(
    tmp_path,
    monkeypatch,
):
    amateur_path = tmp_path / "amateur_career_supplement.csv"
    pd.DataFrame(
        [
            {
                "event_date": "2020-01-01",
                "fighter_a": "Someone Else",
                "fighter_b": "Opponent",
                "winner": "Someone Else",
                "method": "Decision - Unanimous",
                "finish_round": 3,
                "title_bout": 0,
                "organization": "Regional Amateur",
                "weight_class": "Flyweight",
            }
        ]
    ).to_csv(amateur_path, index=False)

    monkeypatch.setattr(
        build_features_module,
        "_resolve_pre_ufc_supplement_path",
        lambda: tmp_path / "missing_pre_ufc.csv",
    )
    monkeypatch.setattr(
        build_features_module,
        "_resolve_amateur_supplement_path",
        lambda: amateur_path,
    )

    fights_df = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2024-01-01"),
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "winner": "Alpha Fighter",
                "method": "KO/TKO",
                "finish_round": 1,
                "weight_class": "Lightweight",
            }
        ]
    )

    features = build_features_module.build_features(fights_df)
    row = features.iloc[0]

    assert "a_amateur_total_fights" in features.columns
    assert pd.isna(row["a_amateur_total_fights"])
    assert pd.isna(row["b_amateur_total_fights"])
    assert pd.isna(row["diff_amateur_total_fights"])


def test_build_features_keeps_ufc_counters_separate_from_amateur_history(tmp_path, monkeypatch):
    amateur_path = tmp_path / "amateur_career_supplement.csv"
    pd.DataFrame(
        [
            {
                "event_date": "2023-01-01",
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Ammy One",
                "winner": "Alpha Fighter",
                "method": "Submission",
                "finish_round": 1,
                "title_bout": 0,
                "organization": "Regional Amateur",
                "weight_class": "Lightweight",
            },
            {
                "event_date": "2023-06-01",
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Ammy Two",
                "winner": "Alpha Fighter",
                "method": "Submission",
                "finish_round": 1,
                "title_bout": 0,
                "organization": "Regional Amateur",
                "weight_class": "Lightweight",
            },
        ]
    ).to_csv(amateur_path, index=False)

    monkeypatch.setattr(
        build_features_module,
        "_resolve_pre_ufc_supplement_path",
        lambda: tmp_path / "missing_pre_ufc.csv",
    )
    monkeypatch.setattr(
        build_features_module,
        "_resolve_amateur_supplement_path",
        lambda: amateur_path,
    )

    fights_df = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2024-01-01"),
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "winner": "Alpha Fighter",
                "method": "KO/TKO",
                "finish_round": 1,
                "weight_class": "Lightweight",
            },
            {
                "event_date": pd.Timestamp("2024-06-01"),
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Gamma Fighter",
                "winner": "Gamma Fighter",
                "method": "Decision - Unanimous",
                "finish_round": 3,
                "weight_class": "Lightweight",
            },
        ]
    )

    features = build_features_module.build_features(fights_df)
    row = features.loc[features["fighter_b"] == "Gamma Fighter"].iloc[0]

    assert row["a_num_fights"] == pytest.approx(1.0)
    assert row["a_current_win_streak"] == pytest.approx(1.0)
    assert row["a_amateur_total_fights"] == pytest.approx(2.0)


def test_full_live_contract_v7_contains_223_features():
    spec = training_spec.full_live_contract_v7_spec()

    assert spec.name == "full_live_contract_v7"
    assert len(training_spec.AMATEUR_FEATURE_COLS) == 21
    assert len(spec.feature_cols) == 223
    assert all(column in spec.feature_cols for column in training_spec.AMATEUR_FEATURE_COLS)
