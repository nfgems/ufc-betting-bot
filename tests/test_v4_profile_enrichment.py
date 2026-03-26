import platform
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bs4 import BeautifulSoup

import scripts.backfill_active_roster_ufcstats as roster_backfill
import scripts.build_profile_supplement_from_external_profiles as external_profiles
from src.data import fallback_scrapers, fighter_lookup, scraper, ufc_active_roster, ufc_refresh
from src.features import build_features as build_features_module
from src.model import training_spec


def test_scrape_all_fighters_prefers_inventory_urls(tmp_path, monkeypatch):
    inventory_path = tmp_path / "ufc-fighter-details.csv"
    output_path = tmp_path / "ufc_fighters_scraped.csv"
    pd.DataFrame(
        [
            {"FIRST": "Alpha", "LAST": "One", "NICKNAME": "", "URL": "http://example.test/fighter/1"},
            {"FIRST": "Beta", "LAST": "Two", "NICKNAME": "", "URL": "http://example.test/fighter/2"},
            {"FIRST": "Alpha", "LAST": "One", "NICKNAME": "", "URL": "http://example.test/fighter/1"},
        ]
    ).to_csv(inventory_path, index=False)

    seen_urls: list[str] = []

    def fake_scrape_fighter(url: str) -> dict:
        seen_urls.append(url)
        return {"name": f"fighter-{len(seen_urls)}", "fighter_url": url}

    monkeypatch.setattr(scraper, "scrape_fighter", fake_scrape_fighter)
    monkeypatch.setattr(
        scraper,
        "scrape_all_fighter_urls",
        lambda: pytest.fail("inventory-backed scrape should not crawl the live fighter directory"),
    )

    scraped = scraper.scrape_all_fighters(
        output_path=output_path,
        fighter_details_path=inventory_path,
    )

    assert seen_urls == [
        "http://example.test/fighter/1",
        "http://example.test/fighter/2",
    ]
    assert scraped["fighter_url"].tolist() == seen_urls
    saved = pd.read_csv(output_path)
    assert saved["fighter_url"].tolist() == seen_urls


def test_scrape_all_fighters_resumes_from_existing_output(tmp_path, monkeypatch):
    inventory_path = tmp_path / "ufc-fighter-details.csv"
    output_path = tmp_path / "ufc_fighters_scraped.csv"
    pd.DataFrame(
        [
            {"FIRST": "Alpha", "LAST": "One", "NICKNAME": "", "URL": "http://example.test/fighter/1"},
            {"FIRST": "Beta", "LAST": "Two", "NICKNAME": "", "URL": "http://example.test/fighter/2"},
        ]
    ).to_csv(inventory_path, index=False)
    pd.DataFrame(
        [
            {"name": "Alpha One", "fighter_url": "http://example.test/fighter/1"},
        ]
    ).to_csv(output_path, index=False)

    seen_urls: list[str] = []

    def fake_scrape_fighter(url: str) -> dict:
        seen_urls.append(url)
        return {"name": "Beta Two", "fighter_url": url}

    monkeypatch.setattr(scraper, "scrape_fighter", fake_scrape_fighter)

    scraped = scraper.scrape_all_fighters(
        output_path=output_path,
        fighter_details_path=inventory_path,
    )

    assert seen_urls == ["http://example.test/fighter/2"]
    assert set(scraped["fighter_url"].tolist()) == {
        "http://example.test/fighter/1",
        "http://example.test/fighter/2",
    }


def test_scrape_all_fights_resumes_from_existing_output(tmp_path, monkeypatch):
    output_path = tmp_path / "ufc_fights_scraped.csv"
    pd.DataFrame(
        [
            {
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "fight_url": "http://example.test/fight/1",
                "event_title": "UFC Test",
                "event_date": "2024-01-01",
            }
        ]
    ).to_csv(output_path, index=False)

    monkeypatch.setattr(scraper, "scrape_all_event_urls", lambda: ["http://example.test/event/1"])
    monkeypatch.setattr(
        scraper,
        "scrape_event",
        lambda event_url: {
            "title": "UFC Test",
            "date": "2024-01-01",
            "fight_urls": [
                "http://example.test/fight/1",
                "http://example.test/fight/2",
            ],
        },
    )

    seen_urls: list[str] = []

    def fake_scrape_fight(url: str) -> dict:
        seen_urls.append(url)
        return {
            "fighter_a": "Gamma",
            "fighter_b": "Delta",
            "winner": "Gamma",
            "result_type": "win",
            "method": "Decision - Unanimous",
            "round": "3",
            "time": "5:00",
            "weight_class": "Lightweight Bout",
            "fight_url": url,
        }

    monkeypatch.setattr(scraper, "scrape_fight", fake_scrape_fight)

    scraped = scraper.scrape_all_fights(output_path=output_path)

    assert seen_urls == ["http://example.test/fight/2"]
    assert set(scraped["fight_url"].tolist()) == {
        "http://example.test/fight/1",
        "http://example.test/fight/2",
    }


def test_scrape_fight_uses_summary_tables_and_parses_sig_breakdown(monkeypatch):
    def paired_cell(a_val: str, b_val: str) -> str:
        return (
            '<td class="b-fight-details__table-col">'
            f"<p>{a_val}</p><p>{b_val}</p>"
            "</td>"
        )

    totals_summary = "".join(
        [
            paired_cell("Alpha Fighter", "Beta Fighter"),
            paired_cell("1", "0"),
            paired_cell("42 of 102", "34 of 84"),
            paired_cell("41%", "40%"),
            paired_cell("67 of 131", "53 of 105"),
            paired_cell("6 of 10", "0 of 0"),
            paired_cell("60%", "---"),
            paired_cell("0", "0"),
            paired_cell("0", "0"),
            paired_cell("2:30", "1:15"),
        ]
    )
    totals_wrong = "".join(
        [
            paired_cell("Alpha Fighter", "Beta Fighter"),
            paired_cell("0", "0"),
            paired_cell("16 of 38", "17 of 28"),
            paired_cell("42%", "60%"),
            paired_cell("20 of 42", "19 of 30"),
            paired_cell("1 of 2", "0 of 0"),
            paired_cell("50%", "---"),
            paired_cell("0", "0"),
            paired_cell("0", "0"),
            paired_cell("0:57", "0:38"),
        ]
    )
    sig_summary = "".join(
        [
            paired_cell("Alpha Fighter", "Beta Fighter"),
            paired_cell("42 of 102", "34 of 84"),
            paired_cell("41%", "40%"),
            paired_cell("25 of 73", "11 of 49"),
            paired_cell("10 of 16", "13 of 26"),
            paired_cell("7 of 13", "10 of 11"),
            paired_cell("40 of 99", "28 of 75"),
            paired_cell("2 of 2", "1 of 3"),
            paired_cell("0 of 1", "5 of 6"),
        ]
    )
    sig_wrong = "".join(
        [
            paired_cell("Alpha Fighter", "Beta Fighter"),
            paired_cell("99 of 99", "99 of 99"),
            paired_cell("99%", "99%"),
            paired_cell("99 of 99", "99 of 99"),
            paired_cell("99 of 99", "99 of 99"),
            paired_cell("99 of 99", "99 of 99"),
            paired_cell("99 of 99", "99 of 99"),
            paired_cell("99 of 99", "99 of 99"),
            paired_cell("99 of 99", "99 of 99"),
        ]
    )

    html = f"""<html><body>
    <h3 class="b-fight-details__person-name"><a>Alpha Fighter</a></h3>
    <h3 class="b-fight-details__person-name"><a>Beta Fighter</a></h3>
    <div class="b-fight-details__person"><i class="b-fight-details__person-status">W</i></div>
    <div class="b-fight-details__person"><i class="b-fight-details__person-status">L</i></div>
    <i class="b-fight-details__text-item_first">Method: Decision - Unanimous</i>
    <i class="b-fight-details__text-item">Round: 3</i>
    <i class="b-fight-details__text-item">Time: 5:00</i>
    <i class="b-fight-details__fight-title">Light Heavyweight Bout</i>
    <table><tbody><tr class="b-fight-details__table-row">{totals_summary}</tr></tbody></table>
    <table class="b-fight-details__table js-fight-table"><tbody><tr class="b-fight-details__table-row">{totals_wrong}</tr></tbody></table>
    <table><tbody><tr class="b-fight-details__table-row">{sig_summary}</tr></tbody></table>
    <table class="b-fight-details__table js-fight-table"><tbody><tr class="b-fight-details__table-row">{sig_wrong}</tr></tbody></table>
    </body></html>"""

    monkeypatch.setattr(scraper, "_get_soup", lambda url: BeautifulSoup(html, "html.parser"))

    result = scraper.scrape_fight("http://example.test/fight/1")

    assert result["a_sig_str_landed"] == "42"
    assert result["a_total_str_landed"] == "67"
    assert result["a_ctrl"] == "2:30"
    assert result["a_head_landed"] == "25"
    assert result["a_body_landed"] == "10"
    assert result["a_leg_landed"] == "7"
    assert result["a_distance_landed"] == "40"
    assert result["a_clinch_landed"] == "2"
    assert result["a_ground_landed"] == "0"
    assert result["b_head_landed"] == "11"
    assert result["b_ground_attempted"] == "6"
    assert result["a_sig_str_landed"] != "16"


def test_build_training_rows_from_pulled_data_uses_scraped_fighter_profiles(tmp_path):
    results_path = tmp_path / "ufc-fight-results.csv"
    stats_path = tmp_path / "ufc-fight-stats.csv"
    profiles_path = tmp_path / "ufc_fighters_scraped.csv"

    pd.DataFrame(
        [
            {
                "EVENT": "UFC Fight Night: Alpha vs. Beta",
                "BOUT": "Alpha Fighter vs. Beta Fighter",
                "OUTCOME": "W/L",
                "METHOD": "Decision - Unanimous",
                "ROUND": 3,
                "TIME": "5:00",
                "TIME FORMAT": "3 Rnd (5-5-5)",
                "WEIGHTCLASS": "Lightweight Bout",
                "URL": "http://example.test/fight/1",
            }
        ]
    ).to_csv(results_path, index=False)

    pd.DataFrame(
        [
            {
                "EVENT": "UFC Test Night",
                "BOUT": "Alpha Fighter vs. Beta Fighter",
                "FIGHTER": "Alpha Fighter",
                "KD": 1,
                "SIG.STR.": "30 of 60",
                "TOTAL STR.": "35 of 70",
                "TD": "2 of 4",
                "SUB.ATT": 1,
                "REV.": 0,
                "CTRL": "3:00",
            },
            {
                "EVENT": "UFC Test Night",
                "BOUT": "Alpha Fighter vs. Beta Fighter",
                "FIGHTER": "Beta Fighter",
                "KD": 0,
                "SIG.STR.": "20 of 50",
                "TOTAL STR.": "24 of 56",
                "TD": "1 of 3",
                "SUB.ATT": 0,
                "REV.": 0,
                "CTRL": "1:30",
            },
        ]
    ).to_csv(stats_path, index=False)

    pd.DataFrame(
        [
            {
                "name": "Alpha Fighter",
                "height": "6' 0\"",
                "reach": "74\"",
                "weight": "155 lbs.",
                "stance": "Orthodox",
                "dob": "1990-01-01",
            },
            {
                "name": "Beta Fighter",
                "height": "5' 10\"",
                "reach": "72\"",
                "weight": "155 lbs.",
                "stance": "Southpaw",
                "dob": "1992-01-01",
            },
        ]
    ).to_csv(profiles_path, index=False)

    pulled_rows = ufc_refresh.build_training_rows_from_pulled_data(
        fight_results_path=results_path,
        fight_stats_path=stats_path,
        scraped_fighters_path=profiles_path,
        event_date_lookup={"ufc fight night: alpha vs. beta": pd.Timestamp("2024-01-01")},
        event_metadata_lookup={
            "ufc fight night: alpha vs. beta": {
                "event_date": pd.Timestamp("2024-01-01"),
                "location": "Las Vegas, Nevada, USA",
            }
        },
    )

    assert len(pulled_rows) == 1
    row = pulled_rows.iloc[0]
    assert row["a_height"] == pytest.approx(182.88)
    assert row["b_height"] == pytest.approx(177.8)
    assert row["a_reach"] == pytest.approx(187.96)
    assert row["b_reach"] == pytest.approx(182.88)
    assert row["a_weight"] == pytest.approx(155.0)
    assert row["b_weight"] == pytest.approx(155.0)
    assert row["a_stance"] == "Orthodox"
    assert row["b_stance"] == "Southpaw"
    assert row["a_age"] == pytest.approx((pd.Timestamp("2024-01-01") - pd.Timestamp("1990-01-01")).days / 365.25)
    assert row["b_age"] == pytest.approx((pd.Timestamp("2024-01-01") - pd.Timestamp("1992-01-01")).days / 365.25)
    assert row["empty_arena"] == pytest.approx(1.0)

    features = build_features_module.build_features(pulled_rows)
    materialized = training_spec.materialize_spec_transforms(features, training_spec.full_live_contract_v4_spec())
    assert materialized.loc[0, "a_weight"] == pytest.approx(155.0)
    assert materialized.loc[0, "b_weight"] == pytest.approx(155.0)
    assert materialized.loc[0, "diff_weight"] == pytest.approx(0.0)
    assert materialized.loc[0, "a_stance_enc"] == pytest.approx(0.0)
    assert materialized.loc[0, "b_stance_enc"] == pytest.approx(1.0)
    assert materialized.loc[0, "same_stance"] == pytest.approx(0.0)
    assert materialized.loc[0, "diff_height"] == pytest.approx(5.08)
    assert materialized.loc[0, "diff_reach"] == pytest.approx(5.08)
    assert materialized.loc[0, "diff_age"] == pytest.approx(row["a_age"] - row["b_age"])
    assert materialized.loc[0, "is_empty_arena"] == pytest.approx(1.0)


def test_build_training_rows_from_pulled_data_backfills_static_lengths_from_legacy_when_scraped_blank(tmp_path):
    results_path = tmp_path / "ufc-fight-results.csv"
    stats_path = tmp_path / "ufc-fight-stats.csv"
    profiles_path = tmp_path / "ufc_fighters_scraped.csv"

    pd.DataFrame(
        [
            {
                "EVENT": "UFC Fight Night: Alpha vs. Beta",
                "BOUT": "Alpha Fighter vs. Beta Fighter",
                "OUTCOME": "W/L",
                "METHOD": "Decision - Unanimous",
                "ROUND": 3,
                "TIME": "5:00",
                "TIME FORMAT": "3 Rnd (5-5-5)",
                "WEIGHTCLASS": "Lightweight Bout",
                "URL": "http://example.test/fight/1",
            }
        ]
    ).to_csv(results_path, index=False)

    pd.DataFrame(
        [
            {
                "EVENT": "UFC Fight Night: Alpha vs. Beta",
                "BOUT": "Alpha Fighter vs. Beta Fighter",
                "FIGHTER": "Alpha Fighter",
                "KD": 1,
                "SIG.STR.": "30 of 60",
                "TOTAL STR.": "35 of 70",
                "TD": "2 of 4",
                "SUB.ATT": 1,
                "REV.": 0,
                "CTRL": "3:00",
            },
            {
                "EVENT": "UFC Fight Night: Alpha vs. Beta",
                "BOUT": "Alpha Fighter vs. Beta Fighter",
                "FIGHTER": "Beta Fighter",
                "KD": 0,
                "SIG.STR.": "20 of 50",
                "TOTAL STR.": "24 of 56",
                "TD": "1 of 3",
                "SUB.ATT": 0,
                "REV.": 0,
                "CTRL": "1:30",
            },
        ]
    ).to_csv(stats_path, index=False)

    pd.DataFrame(
        [
            {
                "name": "Alpha Fighter",
                "height": "--",
                "reach": "--",
                "weight": "155 lbs.",
                "stance": "Orthodox",
                "dob": "1990-01-01",
            },
            {
                "name": "Beta Fighter",
                "height": "5' 10\"",
                "reach": "72\"",
                "weight": "155 lbs.",
                "stance": "Southpaw",
                "dob": "1992-01-01",
            },
        ]
    ).to_csv(profiles_path, index=False)

    legacy_df = pd.DataFrame(
        [
            {
                "event_date": "2023-01-01",
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "a_height": 180.34,
                "a_reach": 185.42,
                "b_height": 190.0,
                "b_reach": 190.5,
            }
        ]
    )

    pulled_rows = ufc_refresh.build_training_rows_from_pulled_data(
        fight_results_path=results_path,
        fight_stats_path=stats_path,
        scraped_fighters_path=profiles_path,
        legacy_df=legacy_df,
        event_date_lookup={"ufc fight night: alpha vs. beta": pd.Timestamp("2024-01-01")},
        event_metadata_lookup={
            "ufc fight night: alpha vs. beta": {
                "event_date": pd.Timestamp("2024-01-01"),
                "location": "Las Vegas, Nevada, USA",
            }
        },
    )

    row = pulled_rows.iloc[0]
    assert row["a_height"] == pytest.approx(180.34)
    assert row["a_reach"] == pytest.approx(185.42)
    assert row["b_height"] == pytest.approx(177.8)
    assert row["b_reach"] == pytest.approx(182.88)


def test_build_training_rows_from_pulled_data_skips_ambiguous_legacy_reach_backfill(tmp_path):
    results_path = tmp_path / "ufc-fight-results.csv"
    stats_path = tmp_path / "ufc-fight-stats.csv"
    profiles_path = tmp_path / "ufc_fighters_scraped.csv"

    pd.DataFrame(
        [
            {
                "EVENT": "UFC Fight Night: Alpha vs. Beta",
                "BOUT": "Alpha Fighter vs. Beta Fighter",
                "OUTCOME": "W/L",
                "METHOD": "Decision - Unanimous",
                "ROUND": 3,
                "TIME": "5:00",
                "TIME FORMAT": "3 Rnd (5-5-5)",
                "WEIGHTCLASS": "Lightweight Bout",
                "URL": "http://example.test/fight/1",
            }
        ]
    ).to_csv(results_path, index=False)

    pd.DataFrame(
        [
            {
                "EVENT": "UFC Fight Night: Alpha vs. Beta",
                "BOUT": "Alpha Fighter vs. Beta Fighter",
                "FIGHTER": "Alpha Fighter",
                "KD": 1,
                "SIG.STR.": "30 of 60",
                "TOTAL STR.": "35 of 70",
                "TD": "2 of 4",
                "SUB.ATT": 1,
                "REV.": 0,
                "CTRL": "3:00",
            },
            {
                "EVENT": "UFC Fight Night: Alpha vs. Beta",
                "BOUT": "Alpha Fighter vs. Beta Fighter",
                "FIGHTER": "Beta Fighter",
                "KD": 0,
                "SIG.STR.": "20 of 50",
                "TOTAL STR.": "24 of 56",
                "TD": "1 of 3",
                "SUB.ATT": 0,
                "REV.": 0,
                "CTRL": "1:30",
            },
        ]
    ).to_csv(stats_path, index=False)

    pd.DataFrame(
        [
            {
                "name": "Alpha Fighter",
                "height": "--",
                "reach": "--",
                "weight": "155 lbs.",
                "stance": "Orthodox",
                "dob": "1990-01-01",
            },
            {
                "name": "Beta Fighter",
                "height": "5' 10\"",
                "reach": "72\"",
                "weight": "155 lbs.",
                "stance": "Southpaw",
                "dob": "1992-01-01",
            },
        ]
    ).to_csv(profiles_path, index=False)

    legacy_df = pd.DataFrame(
        [
            {
                "event_date": "2023-01-01",
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Gamma Fighter",
                "a_height": 180.34,
                "a_reach": 185.42,
            },
            {
                "event_date": "2023-02-01",
                "fighter_a": "Delta Fighter",
                "fighter_b": "Alpha Fighter",
                "b_height": 180.34,
                "b_reach": 190.50,
            },
        ]
    )

    pulled_rows = ufc_refresh.build_training_rows_from_pulled_data(
        fight_results_path=results_path,
        fight_stats_path=stats_path,
        scraped_fighters_path=profiles_path,
        legacy_df=legacy_df,
        event_date_lookup={"ufc fight night: alpha vs. beta": pd.Timestamp("2024-01-01")},
        event_metadata_lookup={
            "ufc fight night: alpha vs. beta": {
                "event_date": pd.Timestamp("2024-01-01"),
                "location": "Las Vegas, Nevada, USA",
            }
        },
    )

    row = pulled_rows.iloc[0]
    assert row["a_height"] == pytest.approx(180.34)
    assert pd.isna(row["a_reach"])


def test_build_training_rows_from_pulled_data_uses_supplemental_profile_artifact_for_blank_fields(tmp_path):
    results_path = tmp_path / "ufc-fight-results.csv"
    stats_path = tmp_path / "ufc-fight-stats.csv"
    profiles_path = tmp_path / "ufc_fighters_scraped.csv"
    supplement_path = tmp_path / "ufc_fighters_profile_supplement.csv"

    pd.DataFrame(
        [
            {
                "EVENT": "UFC Fight Night: Alpha vs. Beta",
                "BOUT": "Alpha Fighter vs. Beta Fighter",
                "OUTCOME": "W/L",
                "METHOD": "Decision - Unanimous",
                "ROUND": 3,
                "TIME": "5:00",
                "TIME FORMAT": "3 Rnd (5-5-5)",
                "WEIGHTCLASS": "Lightweight Bout",
                "URL": "http://example.test/fight/1",
            }
        ]
    ).to_csv(results_path, index=False)

    pd.DataFrame(
        [
            {
                "EVENT": "UFC Fight Night: Alpha vs. Beta",
                "BOUT": "Alpha Fighter vs. Beta Fighter",
                "FIGHTER": "Alpha Fighter",
                "KD": 1,
                "SIG.STR.": "30 of 60",
                "TOTAL STR.": "35 of 70",
                "TD": "2 of 4",
                "SUB.ATT": 1,
                "REV.": 0,
                "CTRL": "3:00",
            },
            {
                "EVENT": "UFC Fight Night: Alpha vs. Beta",
                "BOUT": "Alpha Fighter vs. Beta Fighter",
                "FIGHTER": "Beta Fighter",
                "KD": 0,
                "SIG.STR.": "20 of 50",
                "TOTAL STR.": "24 of 56",
                "TD": "1 of 3",
                "SUB.ATT": 0,
                "REV.": 0,
                "CTRL": "1:30",
            },
        ]
    ).to_csv(stats_path, index=False)

    pd.DataFrame(
        [
            {
                "name": "Alpha Fighter",
                "height": "--",
                "reach": "--",
                "weight": "--",
                "stance": "Orthodox",
                "dob": "--",
            },
            {
                "name": "Beta Fighter",
                "height": "5' 10\"",
                "reach": "72\"",
                "weight": "155 lbs.",
                "stance": "Southpaw",
                "dob": "1992-01-01",
            },
        ]
    ).to_csv(profiles_path, index=False)

    pd.DataFrame(
        [
            {
                "name": "Alpha Fighter",
                "source": "sherdog",
                "source_name": "Alpha Fighter",
                "search_name": "Alpha Fighter",
                "fighter_url": "https://example.test/fighter/alpha",
                "height": "5'11\" / 180.34 cm",
                "reach": "",
                "weight": "220 lbs / 99.79 kg",
                "stance": "",
                "dob": "Jul 16, 1968",
            }
        ]
    ).to_csv(supplement_path, index=False)

    pulled_rows = ufc_refresh.build_training_rows_from_pulled_data(
        fight_results_path=results_path,
        fight_stats_path=stats_path,
        scraped_fighters_path=profiles_path,
        supplemental_profiles_path=supplement_path,
        event_date_lookup={"ufc fight night: alpha vs. beta": pd.Timestamp("2024-01-01")},
        event_metadata_lookup={
            "ufc fight night: alpha vs. beta": {
                "event_date": pd.Timestamp("2024-01-01"),
                "location": "Las Vegas, Nevada, USA",
            }
        },
    )

    row = pulled_rows.iloc[0]
    assert row["a_height"] == pytest.approx(180.34)
    assert row["a_weight"] == pytest.approx(220.0)
    assert row["a_age"] == pytest.approx((pd.Timestamp("2024-01-01") - pd.Timestamp("1968-07-16")).days / 365.25)


def test_build_training_rows_from_pulled_data_backfills_missing_event_metadata_from_fight_pages(
    tmp_path,
    monkeypatch,
    caplog,
):
    results_path = tmp_path / "ufc-fight-results.csv"
    stats_path = tmp_path / "ufc-fight-stats.csv"
    profiles_path = tmp_path / "ufc_fighters_scraped.csv"
    event_cache_path = tmp_path / "ufc-event-dates.csv"

    pd.DataFrame(
        [
            {
                "EVENT": "DWCS 5.4",
                "BOUT": "Alpha Fighter vs. Beta Fighter",
                "OUTCOME": "W/L",
                "METHOD": "Decision - Unanimous",
                "ROUND": 3,
                "TIME": "5:00",
                "TIME FORMAT": "3 Rnd (5-5-5)",
                "WEIGHTCLASS": "Lightweight Bout",
                "URL": "http://example.test/fight-details/fight-1",
            }
        ]
    ).to_csv(results_path, index=False)

    pd.DataFrame(
        [
            {
                "EVENT": "DWCS 5.4",
                "BOUT": "Alpha Fighter vs. Beta Fighter",
                "FIGHTER": "Alpha Fighter",
                "KD": 1,
                "SIG.STR.": "30 of 60",
                "TOTAL STR.": "35 of 70",
                "TD": "2 of 4",
                "SUB.ATT": 1,
                "REV.": 0,
                "CTRL": "3:00",
            },
            {
                "EVENT": "DWCS 5.4",
                "BOUT": "Alpha Fighter vs. Beta Fighter",
                "FIGHTER": "Beta Fighter",
                "KD": 0,
                "SIG.STR.": "20 of 50",
                "TOTAL STR.": "24 of 56",
                "TD": "1 of 3",
                "SUB.ATT": 0,
                "REV.": 0,
                "CTRL": "1:30",
            },
        ]
    ).to_csv(stats_path, index=False)

    pd.DataFrame(
        [
            {
                "name": "Alpha Fighter",
                "height": "6' 0\"",
                "reach": "74\"",
                "weight": "155 lbs.",
                "stance": "Orthodox",
                "dob": "1990-01-01",
            },
            {
                "name": "Beta Fighter",
                "height": "5' 10\"",
                "reach": "72\"",
                "weight": "155 lbs.",
                "stance": "Southpaw",
                "dob": "1992-01-01",
            },
        ]
    ).to_csv(profiles_path, index=False)

    class _FakeResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            return None

    def fake_get(url, headers=None, timeout=30):
        if url == f"{ufc_refresh.UFCSTATS_BASE_URL}?page=1":
            return _FakeResponse("<html><body><table></table></body></html>")
        if url == "http://example.test/fight-details/fight-1":
            return _FakeResponse(
                """
                <html><body>
                  <a class="b-link" href="http://example.test/event-details/dwcs54">DWCS 5.4</a>
                </body></html>
                """
            )
        if url == "http://example.test/event-details/dwcs54":
            return _FakeResponse(
                """
                <html><body>
                  <h2 class="b-content__title"><span>DWCS 5.4</span></h2>
                  <ul>
                    <li class="b-list__box-list-item">Date: September 21, 2021</li>
                    <li class="b-list__box-list-item">Location: Las Vegas, Nevada, USA</li>
                  </ul>
                </body></html>
                """
            )
        raise AssertionError(f"unexpected URL requested: {url}")

    monkeypatch.setattr(ufc_refresh.requests, "get", fake_get)
    monkeypatch.setattr(ufc_refresh.time, "sleep", lambda _seconds: None)

    with caplog.at_level("WARNING", logger="src.data.ufc_refresh"):
        pulled_rows = ufc_refresh.build_training_rows_from_pulled_data(
            fight_results_path=results_path,
            fight_stats_path=stats_path,
            scraped_fighters_path=profiles_path,
            event_dates_cache_path=event_cache_path,
        )

    assert len(pulled_rows) == 1
    row = pulled_rows.iloc[0]
    assert row["event_name"] == "DWCS 5.4"
    assert row["event_date"] == pd.Timestamp("2021-09-21")
    assert row["location"] == "Las Vegas, Nevada, USA"

    saved_cache = pd.read_csv(event_cache_path)
    assert saved_cache.to_dict(orient="records") == [
        {
            "event_name": "dwcs 5.4",
            "event_date": "2021-09-21",
            "location": "Las Vegas, Nevada, USA",
        }
    ]
    assert "Missing UFC event metadata mappings" not in caplog.text


def test_external_profile_candidates_accept_active_roster_alias_rows(tmp_path):
    scraped_path = tmp_path / "ufc_fighters_scraped.csv"
    roster_path = tmp_path / "ufc_active_roster_official.csv"

    pd.DataFrame(
        [
            {
                "name": "Alpha Fighter",
                "height": "--",
                "reach": '76"',
                "weight": "--",
                "stance": "--",
                "dob": "--",
            },
            {
                "name": "Complete Fighter",
                "height": '5\'11"',
                "reach": '72"',
                "weight": "170 lbs.",
                "stance": "Orthodox",
                "dob": "1991-01-01",
            },
        ]
    ).to_csv(scraped_path, index=False)

    pd.DataFrame(
        [
            {
                "official_name": "Alpha Bravo",
                "ufcstats_name": "Alpha Fighter",
                "profile_name": "Alpha Bravo",
                "slug_name": "alpha bravo",
                "alternate_slug_names": "alpha b|a bravo",
                "weight": 185,
            },
            {
                "official_name": "Complete Fighter",
                "ufcstats_name": "Complete Fighter",
                "profile_name": "Complete Fighter",
                "slug_name": "complete fighter",
                "alternate_slug_names": "",
                "weight": 170,
            },
        ]
    ).to_csv(roster_path, index=False)

    candidate_universe, candidates = external_profiles._load_candidates(
        scraped_path,
        candidate_source_csv=roster_path,
    )

    universe_names = set(candidate_universe["name"])
    assert universe_names == {"Alpha Fighter", "Complete Fighter"}

    alpha_row = candidate_universe[candidate_universe["name"] == "Alpha Fighter"].iloc[0]
    assert alpha_row["weight"] == pytest.approx(185.0)
    assert set(alpha_row["search_names"].split("|")) >= {"Alpha Fighter", "Alpha Bravo", "alpha b"}

    assert candidates["name"].tolist() == ["Alpha Fighter"]


def test_external_profile_candidates_keep_blank_ufcstats_active_roster_rows(tmp_path):
    scraped_path = tmp_path / "ufc_fighters_scraped.csv"
    roster_path = tmp_path / "ufc_active_roster_official.csv"

    pd.DataFrame(
        [
            {
                "name": "Covered Fighter",
                "height": '5\'11"',
                "reach": '72"',
                "weight": "170 lbs.",
                "stance": "Orthodox",
                "dob": "1991-01-01",
            },
        ]
    ).to_csv(scraped_path, index=False)

    pd.DataFrame(
        [
            {
                "official_name": "Dallas Marron",
                "ufcstats_name": float("nan"),
                "profile_name": "Dallas Marron",
                "slug_name": "dallas marron",
                "alternate_slug_names": "dallas merron",
                "weight": "",
            },
            {
                "official_name": "Dominik Melendez",
                "ufcstats_name": float("nan"),
                "profile_name": "Dominik Melendez",
                "slug_name": "dominik melendez",
                "alternate_slug_names": "",
                "weight": "",
            },
            {
                "official_name": "Covered Fighter",
                "ufcstats_name": "Covered Fighter",
                "profile_name": "Covered Fighter",
                "slug_name": "covered fighter",
                "alternate_slug_names": "",
                "weight": 170,
            },
        ]
    ).to_csv(roster_path, index=False)

    candidate_universe, candidates = external_profiles._load_candidates(
        scraped_path,
        candidate_source_csv=roster_path,
    )

    universe_names = set(candidate_universe["name"])
    assert "nan" not in universe_names
    assert {"Dallas Marron", "Dominik Melendez", "Covered Fighter"} <= universe_names

    assert {"Dallas Marron", "Dominik Melendez"} <= set(candidates["name"])


def test_load_scraped_fighter_lookup_backfills_missing_weight_from_official_active_roster(tmp_path, monkeypatch):
    profiles_path = tmp_path / "ufc_fighters_scraped.csv"
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    missing_supplement_path = tmp_path / "missing_supplement.csv"

    pd.DataFrame(
        [
            {
                "name": "Alpha Fighter",
                "height": '6\'0"',
                "reach": '74"',
                "weight": "--",
                "stance": "Orthodox",
                "dob": "1990-01-01",
            }
        ]
    ).to_csv(profiles_path, index=False)

    pd.DataFrame(
        [
            {
                "official_name": "Alpha Bravo",
                "ufcstats_name": "Alpha Fighter",
                "profile_name": "Alpha Bravo",
                "slug_name": "alpha bravo",
                "alternate_slug_names": "a bravo",
                "weight": 185,
            },
            {
                "official_name": "Beta Official",
                "ufcstats_name": "Beta Fighter",
                "profile_name": "Beta Official",
                "slug_name": "beta official",
                "alternate_slug_names": "beta b",
                "weight": 170,
            },
        ]
    ).to_csv(roster_path, index=False)

    monkeypatch.setattr(ufc_refresh, "OFFICIAL_ACTIVE_ROSTER_PATH", roster_path)

    lookup = ufc_refresh._load_scraped_fighter_lookup(
        profiles_path,
        supplemental_profiles_path=missing_supplement_path,
    )

    assert lookup[ufc_refresh._normalize_name("Alpha Fighter")]["weight"] == pytest.approx(185.0)
    assert lookup[ufc_refresh._normalize_name("Alpha Bravo")]["weight"] == pytest.approx(185.0)
    assert lookup[ufc_refresh._normalize_name("Beta Fighter")]["weight"] == pytest.approx(170.0)
    assert lookup[ufc_refresh._normalize_name("Beta Fighter")]["dob"] is None


def test_scrape_official_athlete_profile_parses_height_and_reach(monkeypatch):
    html = """
    <html><body>
      <h1 class="hero-profile__name">Isaac Thomson</h1>
      <p class="hero-profile__division-title">Bantamweight Division</p>
      <p class="hero-profile__division-body">0-0-0 (W-L-D)</p>
      <p class="hero-profile__tag">Active</p>
      <div class="c-bio__field"><div class="c-bio__label">Age</div><div class="c-bio__text">22</div></div>
      <div class="c-bio__field"><div class="c-bio__label">Height</div><div class="c-bio__text">70.00</div></div>
      <div class="c-bio__field"><div class="c-bio__label">Reach</div><div class="c-bio__text">69.50</div></div>
      <div class="c-bio__field"><div class="c-bio__label">Weight</div><div class="c-bio__text">135.00</div></div>
    </body></html>
    """

    monkeypatch.setattr(
        ufc_active_roster,
        "_get_soup",
        lambda _url, session=None: BeautifulSoup(html, "lxml"),
    )

    profile = ufc_active_roster.scrape_official_athlete_profile("https://www.ufc.com/athlete/isaac-thomson")

    assert profile["height"] == "70.00 in"
    assert profile["reach"] == "69.50 in"
    assert profile["weight"] == "135.00"


def test_load_scraped_fighter_lookup_backfills_missing_height_reach_and_weight_from_official_active_roster(tmp_path, monkeypatch):
    profiles_path = tmp_path / "ufc_fighters_scraped.csv"
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    missing_supplement_path = tmp_path / "missing_supplement.csv"

    pd.DataFrame(
        [
            {
                "name": "Alpha Fighter",
                "height": "--",
                "reach": "--",
                "weight": "--",
                "stance": "Orthodox",
                "dob": "1990-01-01",
            }
        ]
    ).to_csv(profiles_path, index=False)

    pd.DataFrame(
        [
            {
                "official_name": "Alpha Bravo",
                "ufcstats_name": "Alpha Fighter",
                "profile_name": "Alpha Bravo",
                "slug_name": "alpha bravo",
                "alternate_slug_names": "a bravo",
                "height": "70 in",
                "reach": "74.5 in",
                "weight": 185,
            },
        ]
    ).to_csv(roster_path, index=False)

    monkeypatch.setattr(ufc_refresh, "OFFICIAL_ACTIVE_ROSTER_PATH", roster_path)

    lookup = ufc_refresh._load_scraped_fighter_lookup(
        profiles_path,
        supplemental_profiles_path=missing_supplement_path,
    )

    assert lookup[ufc_refresh._normalize_name("Alpha Fighter")]["height"] == pytest.approx(177.8)
    assert lookup[ufc_refresh._normalize_name("Alpha Fighter")]["reach"] == pytest.approx(189.23)
    assert lookup[ufc_refresh._normalize_name("Alpha Fighter")]["weight"] == pytest.approx(185.0)
    assert lookup[ufc_refresh._normalize_name("Alpha Bravo")]["height"] == pytest.approx(177.8)
    assert lookup[ufc_refresh._normalize_name("Alpha Bravo")]["reach"] == pytest.approx(189.23)
    assert lookup[ufc_refresh._normalize_name("Alpha Bravo")]["weight"] == pytest.approx(185.0)


def test_external_profile_builder_rejects_mismatched_source_profile(monkeypatch):
    row = pd.Series(
        {
            "name": "Abdul Azeem Badakhshi",
            "search_names": "Abdul Azeem Badakhshi|Abdul A. Badakhshi",
            "height": "",
            "reach": "",
            "weight": "",
            "stance": "",
            "dob": "",
        }
    )
    current_state = {}

    monkeypatch.setattr(
        external_profiles,
        "search_martialbot",
        lambda _name: "https://example.test/martialbot/abdul-razak-alhassan",
    )
    monkeypatch.setattr(
        external_profiles,
        "scrape_martialbot_profile",
        lambda _url: {
            "name": "Abdul Razak Alhassan",
            "height_raw": "178 cm",
            "reach_raw": "185 cm",
            "stance": "Orthodox",
            "dob": "Aug 11, 1985",
        },
    )

    assert external_profiles._build_martialbot_row(row, current_state) is None


def test_wikipedia_fallback_rejects_non_fighter_disambiguation_title(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_wiki_api(_session, **params):
        calls.append(params)
        if params.get("titles"):
            return {
                "query": {
                    "pages": [
                        {
                            "missing": True,
                            "title": params["titles"],
                        }
                    ]
                }
            }
        return {
            "query": {
                "search": [
                    {"title": "Sean McInerney (stunt performer)"},
                    {"title": "Sean McInerney (mixed martial artist)"},
                ]
            }
        }

    monkeypatch.setattr(external_profiles, "_wiki_api", fake_wiki_api)

    title = external_profiles._wikipedia_find_title(object(), "Sean Mcinerney")

    assert title == "Sean McInerney (mixed martial artist)"


def test_external_profile_builder_adds_wikipedia_stance(monkeypatch):
    row = pd.Series(
        {
            "name": "Jane Doe",
            "search_names": "Jane Doe",
            "height": "",
            "reach": "",
            "weight": "",
            "stance": "",
            "dob": "",
        }
    )
    current_state = {}

    monkeypatch.setattr(
        external_profiles,
        "_wikipedia_find_title",
        lambda _session, _name: "Jane Doe (mixed martial artist)",
    )
    monkeypatch.setattr(
        external_profiles,
        "_wiki_api",
        lambda _session, **_params: {
            "query": {
                "pages": [
                    {
                        "revisions": [
                            {
                                "content": """
{{Infobox martial artist
| stance = [[Southpaw stance|Southpaw]]
| reach = {{convert|72|in|cm}}
| birth_date = {{birth date and age|1995|04|26}}
}}
"""
                            }
                        ]
                    }
                ]
            }
        },
    )

    row_out = external_profiles._build_wikipedia_row(object(), row, current_state)

    assert row_out is not None
    assert row_out["source"] == "wikipedia"
    assert row_out["stance"] == "Southpaw"
    assert row_out["reach"] == "72 in"
    assert row_out["dob"] == "1995-04-26"


def test_external_profile_builder_adds_sherdog_height_and_dob(monkeypatch):
    row = pd.Series(
        {
            "name": "Dallas Marron",
            "search_names": "Dallas Marron|dallas merron",
            "height": "",
            "reach": "",
            "weight": "",
            "stance": "",
            "dob": "",
        }
    )
    current_state = {}

    monkeypatch.setattr(
        external_profiles,
        "search_sherdog",
        lambda _name: "https://www.sherdog.com/fighter/Dallas-Marron-123456",
    )
    monkeypatch.setattr(
        external_profiles,
        "scrape_sherdog_page",
        lambda _url, _fighter_name: (
            {
                "name": "Dallas Marron",
                "height": 185.0,
                "height_raw": '6\'1" / 185 cm',
                "weight": float("nan"),
                "weight_raw": "",
                "dob": "Apr 26, 1995",
            },
            [],
        ),
    )

    row_out = external_profiles._build_sherdog_row(row, current_state)

    assert row_out is not None
    assert row_out["source"] == "sherdog"
    assert row_out["height"] == '6\'1" / 185 cm'
    assert row_out["dob"] == "Apr 26, 1995"


def test_external_profile_builder_ignores_zero_valued_sherdog_placeholders(monkeypatch):
    row = pd.Series(
        {
            "name": "Jonathan Correa",
            "search_names": "Jonathan Correa",
            "height": "",
            "reach": "",
            "weight": "170 lbs.",
            "stance": "",
            "dob": "",
        }
    )
    current_state = {}

    monkeypatch.setattr(
        external_profiles,
        "search_sherdog",
        lambda _name: "https://www.sherdog.com/fighter/Jonathan-Correa-57482",
    )
    monkeypatch.setattr(
        external_profiles,
        "scrape_sherdog_page",
        lambda _url, _fighter_name: (
            {
                "name": "Jonathan Correa",
                "height": 0.0,
                "height_raw": '0\'0" / 0 cm',
                "weight": 0.0,
                "weight_raw": "0 lbs / 0 kg",
                "dob": "",
            },
            [],
        ),
    )

    assert external_profiles._build_sherdog_row(row, current_state) is None


def test_append_missing_profiles_refreshes_incomplete_active_roster_profile(tmp_path, monkeypatch):
    fighters_path = tmp_path / "ufc_fighters_scraped.csv"
    pd.DataFrame(
        [
            {
                "name": "Isaac Thomson",
                "record": "9-2-0",
                "fighter_url": "http://ufcstats.com/fighter-details/isaac",
                "height": '5\' 10"',
                "weight": "135 lbs.",
                "reach": "--",
                "stance": "--",
                "dob": "May 03, 2002",
                "slpm": "1.0",
                "str_acc": "50",
                "sapm": "1.0",
                "str_def": "50",
                "td_avg": "0.0",
                "td_acc": "0",
                "td_def": "0",
                "sub_avg": "0.0",
            }
        ]
    ).to_csv(fighters_path, index=False)

    roster_df = pd.DataFrame(
        [
            {
                "official_name": "Isaac Thomson",
                "ufcstats_url": "http://ufcstats.com/fighter-details/isaac",
            }
        ]
    )

    monkeypatch.setattr(roster_backfill, "FIGHTERS_PATH", fighters_path)
    monkeypatch.setattr(
        roster_backfill,
        "_profile_row_from_url",
        lambda _url: {
            "name": "Isaac Thomson",
            "record": "9-2-0",
            "fighter_url": "http://ufcstats.com/fighter-details/isaac",
            "height": '5\' 10"',
            "weight": "135 lbs.",
            "reach": '69"',
            "stance": "Switch",
            "dob": "May 03, 2002",
            "slpm": "1.0",
            "str_acc": "50",
            "sapm": "1.0",
            "str_def": "50",
            "td_avg": "0.0",
            "td_acc": "0",
            "td_def": "0",
            "sub_avg": "0.0",
        },
    )

    added, updated = roster_backfill._append_missing_profiles(roster_df)
    refreshed = pd.read_csv(fighters_path)

    assert added == 0
    assert updated == 1
    assert refreshed.loc[0, "reach"] == '69"'
    assert refreshed.loc[0, "stance"] == "Switch"


def test_scrape_sherdog_page_captures_dob_from_age_row(monkeypatch):
    html = """
    <html><body>
      <h1>Jess Liaudin</h1>
      <div class="winloses"><span>Wins</span><span>20</span></div>
      <div class="winloses"><span>Losses</span><span>11</span></div>
      <div class="winloses"><span>Draws</span><span>0</span></div>
      <table>
        <tr><td>HEIGHT</td><td>5'9" / 175.26 cm</td></tr>
        <tr><td>WEIGHT</td><td>170 lbs / 77.11 kg</td></tr>
        <tr><td>AGE</td><td>52 / Dec 21, 1973</td></tr>
      </table>
      <table class="fighter">
        <tr><th>Result</th></tr>
      </table>
    </body></html>
    """

    monkeypatch.setattr(
        fallback_scrapers,
        "_get_soup",
        lambda _url: BeautifulSoup(html, "lxml"),
    )

    profile, fights = fallback_scrapers.scrape_sherdog_page(
        "https://example.test/fighter/jess-liaudin",
        "Jess Liaudin",
    )

    assert profile["name"] == "Jess Liaudin"
    assert profile["height"] == pytest.approx(175.26)
    assert profile["weight"] == pytest.approx(170.0)
    assert profile["age"] == pytest.approx(52.0)
    assert profile["dob"] == "Dec 21, 1973"
    assert fights == []


def test_search_tapology_uses_tapology_search_results(monkeypatch):
    html = """
    <html><body>
      <a href="/fightcenter/fighters/steve-nelmark-the-sandman">Steve "The Sandman" Nelmark</a>
      <a href="/fightcenter/fighters/steve-lopez">Steve Lopez</a>
    </body></html>
    """

    monkeypatch.setattr(
        fallback_scrapers,
        "_get_tapology_soup",
        lambda _url, params=None, **_kwargs: BeautifulSoup(html, "lxml"),
    )
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.search_tapology("Steve Nelmark")

    assert result == "https://www.tapology.com/fightcenter/fighters/steve-nelmark-the-sandman"


def test_build_tapology_scraper_sets_modern_user_agent(monkeypatch):
    captured_browser = {}

    class _FakeScraper:
        def __init__(self):
            self.headers = {}

    def fake_create_scraper(*, browser):
        captured_browser["browser"] = browser
        return _FakeScraper()

    monkeypatch.setattr(
        fallback_scrapers,
        "cloudscraper",
        types.SimpleNamespace(create_scraper=fake_create_scraper),
    )
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    scraper = fallback_scrapers._build_tapology_scraper()

    assert captured_browser["browser"] == {
        "browser": "chrome",
        "platform": "linux",
        "mobile": False,
    }
    assert scraper.headers["User-Agent"] == (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
    assert scraper.headers["Accept"] == fallback_scrapers.HEADERS["Accept"]
    assert scraper.headers["Accept-Language"] == fallback_scrapers.HEADERS["Accept-Language"]


def test_search_tapology_falls_back_to_site_search(monkeypatch):
    class _FakeResponse:
        text = """
        <html><body>
          <a href="https://www.tapology.com/fightcenter/fighters/steve-nelmark-the-sandman">
            Steve Nelmark | MMA Fighter Page | Tapology
          </a>
        </body></html>
        """
        status_code = 200

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        fallback_scrapers,
        "_get_tapology_soup",
        lambda _url, params=None, **_kwargs: BeautifulSoup("<html><body></body></html>", "lxml"),
    )
    monkeypatch.setattr(fallback_scrapers.requests, "get", lambda *args, **kwargs: _FakeResponse())
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.search_tapology("Steve Nelmark")

    assert result == "https://www.tapology.com/fightcenter/fighters/steve-nelmark-the-sandman"


def test_search_tapology_disables_native_search_after_403(monkeypatch):
    class _FakeResponse:
        text = """
        <html><body>
          <a href="https://www.tapology.com/fightcenter/fighters/steve-nelmark-the-sandman">
            Steve Nelmark | MMA Fighter Page | Tapology
          </a>
        </body></html>
        """
        status_code = 200

        def raise_for_status(self):
            return None

    native_calls = []

    def fake_get_tapology_soup(_url, params=None, max_retries=None, retry_statuses=None):
        native_calls.append(
            {
                "term": (params or {}).get("term"),
                "max_retries": max_retries,
                "retry_statuses": retry_statuses,
            }
        )
        raise fallback_scrapers.TapologyRequestError(_url, status_code=403)

    monkeypatch.setattr(fallback_scrapers, "_get_tapology_soup", fake_get_tapology_soup)
    monkeypatch.setattr(fallback_scrapers.requests, "get", lambda *args, **kwargs: _FakeResponse())
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()

    first_result = fallback_scrapers.search_tapology_candidates("Steve Nelmark", limit=1)
    second_result = fallback_scrapers.search_tapology_candidates("Another Fighter", limit=1)

    assert first_result == ["https://www.tapology.com/fightcenter/fighters/steve-nelmark-the-sandman"]
    assert second_result == []
    assert native_calls == [
        {
            "term": "Steve Nelmark",
            "max_retries": 1,
            "retry_statuses": {429, 503},
        }
    ]


def test_search_martialbot_uses_json_search_results(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "id": "steve-nelmark-13e62d766b709aa6",
                        "name": "Steve Nelmark",
                        "display_name": "Steve Nelmark",
                    },
                    {
                        "id": "steve-lopez-deadbeef",
                        "name": "Steve Lopez",
                        "display_name": "Steve Lopez",
                    },
                ]
            }

    monkeypatch.setattr(fallback_scrapers.requests, "get", lambda *args, **kwargs: _FakeResponse())
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.search_martialbot("Steve Nelmark")

    assert result == "https://www.martialbot.com/mma/fighters/steve-nelmark-13e62d766b709aa6"


def test_search_martialbot_rejects_weak_false_positive_result(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "id": "cody-garbrandt-0e6f9ad3c89cbe9220d18547707f43aa",
                        "name": "Cody Garbrandt",
                        "display_name": "Cody Garbrandt",
                    }
                ]
            }

    monkeypatch.setattr(fallback_scrapers.requests, "get", lambda *args, **kwargs: _FakeResponse())
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.search_martialbot("Cody Belisle")

    assert result is None


def test_search_fightdx_uses_slugged_profile_page(monkeypatch):
    class _FakeResponse:
        status_code = 200
        text = """
        <html><head><title>Steve Nelmark | MMA Fighter Stats &amp; Record</title></head><body>
          <h1>Steve Nelmark</h1>
        </body></html>
        """

    monkeypatch.setattr(fallback_scrapers.requests, "get", lambda *args, **kwargs: _FakeResponse())
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.search_fightdx("Steve Nelmark")

    assert result == "https://fightdx.com/person/steve-nelmark"


def test_search_fightdx_rejects_weak_slug_false_positive(monkeypatch):
    class _FakeResponse:
        def __init__(self, status_code=200, text=""):
            self.status_code = status_code
            self.text = text

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError("http error")
            return None

    def fake_get(url, *args, **kwargs):
        if url == "https://fightdx.com/person/cody-belisle":
            return _FakeResponse(
                text="""
                <html><head><title>Cody Bell | MMA Fighter Stats &amp; Record</title></head><body>
                  <h1>Cody Bell</h1>
                </body></html>
                """
            )
        if url == "https://fightdx.com/sitemap.xml":
            return _FakeResponse(status_code=404)
        return _FakeResponse(status_code=404)

    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.search_fightdx("Cody Belisle")

    assert result is None


def test_search_fightdx_falls_back_to_sitemap(monkeypatch):
    class _FakeResponse:
        def __init__(self, status_code=200, text=""):
            self.status_code = status_code
            self.text = text

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError("http error")
            return None

    def fake_get(url, *args, **kwargs):
        if url == "https://fightdx.com/sitemap.xml":
            return _FakeResponse(
                text="""
                <sitemapindex>
                  <sitemap><loc>https://fightdx.com/sitemap-complete_people.xml</loc></sitemap>
                </sitemapindex>
                """
            )
        if url == "https://fightdx.com/sitemap-complete_people.xml":
            return _FakeResponse(
                text="""
                <urlset>
                  <url><loc>https://fightdx.com/person/steve-nelmark</loc></url>
                  <url><loc>https://fightdx.com/person/steve-lopez</loc></url>
                </urlset>
                """
            )
        if url == "https://fightdx.com/person/steve-nelmark-the-wrong-slug":
            return _FakeResponse(status_code=404)
        if url == "https://fightdx.com/person/steve-nelmark":
            return _FakeResponse(
                text="""
                <html><head><title>Steve Nelmark | MMA Fighter Stats &amp; Record</title></head><body>
                  <h1>Steve Nelmark</h1>
                </body></html>
                """
            )
        return _FakeResponse(status_code=404)

    monkeypatch.setattr(
        fallback_scrapers,
        "_slugify_person_name",
        lambda _name: "steve-nelmark-the-wrong-slug",
    )
    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.search_fightdx("Steve Nelmark")

    assert result == "https://fightdx.com/person/steve-nelmark"


def test_scrape_fightdx_profile_parses_reach_tile(monkeypatch):
    html = """
    <html><head><title>Marcus Bossett | MMA Fighter Stats &amp; Record</title></head><body>
      <h1>Marcus Bossett</h1>
      <div class="info-stat-tile">
        <span class="info-stat-label">Height</span>
        <span class="info-stat-value">6'1" (1.85m)</span>
      </div>
      <div class="info-stat-tile">
        <span class="info-stat-label">Weight</span>
        <span class="info-stat-value">220lbs (100kg)</span>
      </div>
      <div class="info-stat-tile">
        <span class="info-stat-label">Reach</span>
        <span class="info-stat-value">6'1" (1.85m)</span>
      </div>
      <div class="info-stat-tile">
        <span class="info-stat-label">Style</span>
        <span class="info-stat-value">Orthodox</span>
      </div>
      <div class="info-stat-tile">
        <span class="info-stat-label">Age</span>
        <span class="info-stat-value">-</span>
      </div>
    </body></html>
    """

    monkeypatch.setattr(
        fallback_scrapers,
        "_get_soup",
        lambda _url: BeautifulSoup(html, "lxml"),
    )

    profile = fallback_scrapers.scrape_fightdx_profile(
        "https://fightdx.com/person/marcus-bossett"
    )

    assert profile["name"] == "Marcus Bossett"
    assert profile["height_raw"] == "6'1\" (1.85m)"
    assert profile["reach_raw"] == "6'1\" (1.85m)"
    assert profile["weight_raw"] == "220lbs (100kg)"
    assert profile["stance"] == "Orthodox"
    assert profile["dob"] == ""
    assert profile["reach"] == pytest.approx(185.0, abs=0.1)  # 1.85m → 185 cm


def test_scrape_martialbot_profile_parses_reach_and_exact_dob(monkeypatch):
    html = """
    <html><body>
      <h1>Patrick Smith</h1>
      <dl>
        <dt>Born</dt><dd>Aug 28, 1963</dd>
        <dt>Record</dt><dd>20-17 (19)</dd>
        <dt>Height</dt><dd>188 cm</dd>
        <dt>Reach</dt><dd>188 cm</dd>
        <dt>Stance</dt><dd>Orthodox</dd>
      </dl>
    </body></html>
    """

    monkeypatch.setattr(
        fallback_scrapers,
        "_get_soup",
        lambda _url: BeautifulSoup(html, "lxml"),
    )

    profile = fallback_scrapers.scrape_martialbot_profile(
        "https://www.martialbot.com/mma/fighters/patrick-smith-0d62c17098b5f99e75621d360b1cfc1e"
    )

    assert profile["name"] == "Patrick Smith"
    assert profile["record"] == "20-17 (19)"
    assert profile["wins"] == 20
    assert profile["losses"] == 17
    assert profile["draws"] == 0
    assert profile["height_raw"] == "188 cm"
    assert profile["reach_raw"] == "188 cm"
    assert profile["stance"] == "Orthodox"
    assert profile["dob"] == "Aug 28, 1963"
    assert profile["reach"] == pytest.approx(188.0, abs=0.1)


def test_scrape_tapology_profile_parses_reach_and_exact_dob(monkeypatch):
    html = """
    <html><head><title>Din Thomas ("Dinyero") | MMA Fighter Page | Tapology</title></head><body>
      <div class="div flex flex-col rounded h-20 p-1 bg-blue-50 w-1/4 items-center justify-center">
        <div>Age</div>
        <div><span data-controller="age-calc">1976-09-28</span></div>
        <div>1976 • Sep 28</div>
      </div>
      <div class="div flex flex-col rounded h-20 p-1 bg-blue-50 w-1/4 items-center justify-center">
        <div>Height</div>
        <div>5'9"</div>
        <div>175cm</div>
      </div>
      <div class="div flex flex-col rounded h-20 p-1 bg-blue-50 w-1/4 items-center justify-center">
        <div>Reach</div>
        <div>75.0"</div>
        <div>191cm</div>
      </div>
      <div class="div flex flex-col rounded h-20 p-1 bg-blue-50 w-1/4 items-center justify-center">
        <div>Weight</div>
        <div>155.0</div>
        <div>Lightweight</div>
      </div>
      <div class="div flex flex-col px-1 md:px-0 text-[13px] md:text-xs text-tap_3 mt-1 md:mt-3">
        Name: Din Thomas Nickname: Dinyero Pro MMA Record: 26-9-0 (Win-Loss-Draw)
      </div>
    </body></html>
    """

    monkeypatch.setattr(
        fallback_scrapers,
        "_get_tapology_soup",
        lambda _url, params=None: BeautifulSoup(html, "lxml"),
    )

    profile = fallback_scrapers.scrape_tapology_profile(
        "https://www.tapology.com/fightcenter/fighters/din-thomas-dinyero"
    )

    assert profile["name"] == "Din Thomas"
    assert profile["record"] == "26-9-0"
    assert profile["wins"] == 26
    assert profile["losses"] == 9
    assert profile["draws"] == 0
    assert profile["height_raw"] == "5'9\" (175cm)"
    assert profile["reach_raw"] == "75.0\" (191cm)"
    assert profile["weight_raw"] == "155.0 lbs"
    assert profile["dob"] == "1976-09-28"
    assert profile["reach"] == pytest.approx(191.0, abs=0.1)


def test_full_live_contract_v4_spec_reincludes_only_recoverable_profile_fields():
    v3_spec = training_spec.full_live_contract_v3_spec()
    v4_spec = training_spec.full_live_contract_v4_spec()

    assert v4_spec.name == "full_live_contract_v4"
    assert v4_spec.dataset_variant == "pulled_all"
    assert len(v4_spec.feature_cols) == len(v3_spec.feature_cols) + 16

    for included in [
        "a_height",
        "b_height",
        "diff_height",
        "a_reach",
        "b_reach",
        "diff_reach",
        "a_weight",
        "b_weight",
        "diff_weight",
        "a_age",
        "b_age",
        "diff_age",
        "a_stance_enc",
        "b_stance_enc",
        "same_stance",
        "is_empty_arena",
    ]:
        assert included in v4_spec.feature_cols

    for excluded in [
        "a_implied_prob",
        "a_wc_rank_feat",
        "a_ko_odds_prob",
    ]:
        assert excluded not in v4_spec.feature_cols

    resolved = training_spec.resolve_named_training_spec("full_live_contract_v4")
    assert resolved.name == v4_spec.name


def test_full_live_contract_v4_144_spec_adds_market_rankings_and_method_odds_on_top_of_v4():
    v4_spec = training_spec.full_live_contract_v4_spec()
    expanded_spec = training_spec.full_live_contract_v4_144_spec()

    assert expanded_spec.name == "full_live_contract_v4_144"
    assert expanded_spec.dataset_variant == "pulled_all_plus_legacy_market"
    assert len(expanded_spec.feature_cols) == len(v4_spec.feature_cols) + 15

    for included in [
        "a_implied_prob",
        "b_implied_prob",
        "diff_implied_prob",
        "a_wc_rank_feat",
        "b_wc_rank_feat",
        "diff_wc_rank",
        "a_pfp_rank_feat",
        "b_pfp_rank_feat",
        "diff_pfp_rank",
        "a_ko_odds_prob",
        "a_sub_odds_prob",
        "a_dec_odds_prob",
        "b_ko_odds_prob",
        "b_sub_odds_prob",
        "b_dec_odds_prob",
    ]:
        assert included in expanded_spec.feature_cols

    resolved = training_spec.resolve_named_training_spec("full_live_contract_v4_144")
    assert resolved.name == expanded_spec.name


def test_full_live_contract_v4_138_spec_adds_moneyline_and_method_odds_but_not_rankings():
    v4_spec = training_spec.full_live_contract_v4_spec()
    expanded_spec = training_spec.full_live_contract_v4_138_spec()

    assert expanded_spec.name == "full_live_contract_v4_138"
    assert expanded_spec.dataset_variant == "pulled_all_plus_legacy_market"
    assert len(expanded_spec.feature_cols) == len(v4_spec.feature_cols) + 9

    for included in [
        "a_implied_prob",
        "b_implied_prob",
        "diff_implied_prob",
        "a_ko_odds_prob",
        "a_sub_odds_prob",
        "a_dec_odds_prob",
        "b_ko_odds_prob",
        "b_sub_odds_prob",
        "b_dec_odds_prob",
    ]:
        assert included in expanded_spec.feature_cols

    for excluded in [
        "a_wc_rank_feat",
        "b_wc_rank_feat",
        "diff_wc_rank",
        "a_pfp_rank_feat",
        "b_pfp_rank_feat",
        "diff_pfp_rank",
    ]:
        assert excluded not in expanded_spec.feature_cols

    resolved = training_spec.resolve_named_training_spec("full_live_contract_v4_138")
    assert resolved.name == expanded_spec.name


def test_historical_moneyline_overlay_converts_decimal_prices_to_american(monkeypatch):
    historical_df = pd.DataFrame(
        [
            {
                "event_date": "2024-01-01",
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "query_date": "2023-12-31",
                "offset_days": 1,
                "a_decimal_odds": 1.80,
                "b_decimal_odds": 2.20,
            }
        ]
    )

    monkeypatch.setattr(
        "src.data.historical_backfill.load_all_historical_odds",
        lambda: historical_df,
    )

    overlay = ufc_refresh._historical_moneyline_overlay(
        pd.DataFrame(
            [
                {
                    "fight_key": "2024-01-01|alpha fighter|beta fighter",
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                }
            ]
        )
    )

    assert overlay.loc[0, "a_odds__historical_overlay"] == pytest.approx(-125.0)
    assert overlay.loc[0, "b_odds__historical_overlay"] == pytest.approx(120.0)


def test_historical_rankings_overlay_uses_latest_snapshot_on_or_before_fight_date(tmp_path, monkeypatch):
    rankings_dir = tmp_path / "kaggle_rankings"
    rankings_dir.mkdir()
    pd.DataFrame(
        [
            {"date": "2023-12-25", "weightclass": "Lightweight", "fighter": "Alpha Fighter", "rank": 4},
            {"date": "2023-12-25", "weightclass": "Lightweight", "fighter": "Beta Fighter", "rank": 11},
            {"date": "2023-12-25", "weightclass": "POUND-FOR-POUND", "fighter": "Alpha Fighter", "rank": 9},
            {"date": "2023-12-25", "weightclass": "POUND-FOR-POUND", "fighter": "Beta Fighter", "rank": 14},
        ]
    ).to_csv(rankings_dir / "rankings_history_extended.csv", index=False)

    monkeypatch.setattr(ufc_refresh, "RAW_DATA_DIR", tmp_path)

    overlay = ufc_refresh._historical_rankings_overlay(
        pd.DataFrame(
            [
                {
                    "fight_key": "2024-01-01|alpha fighter|beta fighter",
                    "event_date": "2024-01-01",
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "weight_class": "Lightweight",
                }
            ]
        )
    )

    assert overlay.loc[0, "a_wc_rank__historical_overlay"] == pytest.approx(4.0)
    assert overlay.loc[0, "b_wc_rank__historical_overlay"] == pytest.approx(11.0)
    assert overlay.loc[0, "a_pfp_rank__historical_overlay"] == pytest.approx(9.0)
    assert overlay.loc[0, "b_pfp_rank__historical_overlay"] == pytest.approx(14.0)


def test_historical_method_odds_overlay_orients_saved_probabilities(tmp_path, monkeypatch):
    method_dir = tmp_path / "method_odds"
    method_dir.mkdir()
    pd.DataFrame(
        [
            {
                "event_date": "2024-01-01",
                "fighter_a": "Beta Fighter",
                "fighter_b": "Alpha Fighter",
                "a_ko_odds_prob": 0.18,
                "a_sub_odds_prob": 0.09,
                "a_dec_odds_prob": 0.22,
                "b_ko_odds_prob": 0.31,
                "b_sub_odds_prob": 0.14,
                "b_dec_odds_prob": 0.26,
            }
        ]
    ).to_csv(method_dir / "historical_method_odds_all.csv", index=False)

    monkeypatch.setattr(ufc_refresh, "RAW_DATA_DIR", tmp_path)

    overlay = ufc_refresh._historical_method_odds_overlay(
        pd.DataFrame(
            [
                {
                    "fight_key": "2024-01-01|alpha fighter|beta fighter",
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                }
            ]
        )
    )

    assert overlay.loc[0, "a_ko_odds_prob__historical_overlay"] == pytest.approx(0.31)
    assert overlay.loc[0, "a_sub_odds_prob__historical_overlay"] == pytest.approx(0.14)
    assert overlay.loc[0, "a_dec_odds_prob__historical_overlay"] == pytest.approx(0.26)
    assert overlay.loc[0, "b_ko_odds_prob__historical_overlay"] == pytest.approx(0.18)
    assert overlay.loc[0, "b_sub_odds_prob__historical_overlay"] == pytest.approx(0.09)
    assert overlay.loc[0, "b_dec_odds_prob__historical_overlay"] == pytest.approx(0.22)


def test_build_features_preserves_precomputed_market_probabilities_when_raw_odds_missing():
    fights = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2024-01-01"),
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "winner": "Alpha Fighter",
                "weight_class": "Lightweight",
                "title_bout": 0.0,
                "num_rounds": 3.0,
                "a_ko_odds_prob": 0.31,
                "a_sub_odds_prob": 0.14,
                "a_dec_odds_prob": 0.26,
                "b_ko_odds_prob": 0.18,
                "b_sub_odds_prob": 0.09,
                "b_dec_odds_prob": 0.22,
                "a_implied_prob": 0.58,
                "b_implied_prob": 0.42,
            }
        ]
    )

    built = build_features_module.build_features(fights)

    assert built.loc[0, "a_ko_odds_prob"] == pytest.approx(0.31)
    assert built.loc[0, "a_sub_odds_prob"] == pytest.approx(0.14)
    assert built.loc[0, "a_dec_odds_prob"] == pytest.approx(0.26)
    assert built.loc[0, "b_ko_odds_prob"] == pytest.approx(0.18)
    assert built.loc[0, "b_sub_odds_prob"] == pytest.approx(0.09)
    assert built.loc[0, "b_dec_odds_prob"] == pytest.approx(0.22)
    assert built.loc[0, "a_implied_prob"] == pytest.approx(0.58)
    assert built.loc[0, "b_implied_prob"] == pytest.approx(0.42)
    assert built.loc[0, "diff_implied_prob"] == pytest.approx(0.16)


def test_full_live_contract_v4_live_lookup_keeps_strict_history_and_only_reenables_requested_profiles(monkeypatch):
    profile_html = """
    <html><body>
      <h2 class="b-content__title"><span>Beta Fighter</span></h2>
      <span class="b-content__title-record">Record: 5-0-0</span>
      <li class="b-list__box-list-item">Height: 5' 9"</li>
      <li class="b-list__box-list-item">Weight: 145 lbs.</li>
      <li class="b-list__box-list-item">Reach: 72"</li>
      <li class="b-list__box-list-item">STANCE: Orthodox</li>
      <li class="b-list__box-list-item">DOB: Apr 27, 1995</li>
      <li class="b-list__box-list-item_type_block">SLpM: 3.2</li>
      <li class="b-list__box-list-item_type_block">Str. Acc.: 65%</li>
      <li class="b-list__box-list-item_type_block">SApM: 1.1</li>
      <li class="b-list__box-list-item_type_block">Str. Def: 62%</li>
      <li class="b-list__box-list-item_type_block">TD Avg.: 2.5</li>
      <li class="b-list__box-list-item_type_block">TD Acc.: 55%</li>
      <li class="b-list__box-list-item_type_block">TD Def.: 80%</li>
      <li class="b-list__box-list-item_type_block">Sub. Avg.: 0.4</li>
      <tr class="b-fight-details__table-row" data-link="http://example.test/fight-details/1">
        <td><a class="b-flag">win</a></td>
        <td><p>Beta Fighter</p><p>Cam Teague</p></td>
        <td><p>1</p><p>0</p></td>
        <td><p>15 of 30</p><p>9 of 18</p></td>
        <td><p>1 of 2</p><p>0 of 1</p></td>
        <td><p>1</p><p>0</p></td>
        <td>DWCS 9.5 Sep. 09, 2025</td>
        <td>Decision - Unanimous</td>
        <td>3</td>
        <td>5:00</td>
      </tr>
    </body></html>
    """
    detail_html = """
    <html><body>
      <i class="b-fight-details__fight-title">Featherweight Bout</i>
      <table class="b-fight-details__table">
        <tbody>
          <tr class="b-fight-details__table-row">
            <td><p>Beta Fighter</p><p>Cam Teague</p></td>
            <td><p>1</p><p>0</p></td>
            <td><p>15 of 30</p><p>9 of 18</p></td>
            <td><p>50%</p><p>50%</p></td>
            <td><p>15 of 30</p><p>9 of 18</p></td>
            <td><p>1 of 2</p><p>0 of 1</p></td>
            <td><p>50%</p><p>0%</p></td>
            <td><p>1</p><p>0</p></td>
            <td><p>0</p><p>0</p></td>
            <td><p>2:30</p><p>0:45</p></td>
          </tr>
        </tbody>
      </table>
    </body></html>
    """

    soups = {
        "http://example.test/fighter": BeautifulSoup(profile_html, "lxml"),
        "http://example.test/fight-details/1": BeautifulSoup(detail_html, "lxml"),
    }

    monkeypatch.setattr(fighter_lookup, "search_fighter_url", lambda *_args, **_kwargs: "http://example.test/fighter")
    monkeypatch.setattr(fighter_lookup, "_get_soup", lambda url: soups[url])

    fighter_lookup.clear_cache()
    reference_date = "2026-04-01"
    expected_days_since_last_fight = (pd.Timestamp(reference_date) - pd.Timestamp("2025-09-09")).days

    result = fighter_lookup.lookup_fighter(
        "Beta Fighter",
        training_spec=training_spec.full_live_contract_v4_spec(),
        reference_date=reference_date,
    )

    assert result is not None
    assert result["features"]["roll_slpm"] == pytest.approx(1.0)
    assert result["features"]["roll_td_avg"] == pytest.approx(1.0)
    assert result["features"]["height"] == pytest.approx(175.26)
    assert result["features"]["reach"] == pytest.approx(182.88)
    assert result["features"]["weight"] == pytest.approx(145.0)
    assert result["features"]["age"] == pytest.approx(
        fighter_lookup._parse_dob_to_age("Apr 27, 1995", reference_date=reference_date)
    )
    assert result["features"]["stance_enc"] == pytest.approx(0.0)
    assert result["features"]["days_since_last_fight"] == pytest.approx(expected_days_since_last_fight)

    fighter_lookup.clear_cache()


def test_build_features_preserves_unknown_stance_as_nan(monkeypatch):
    monkeypatch.setattr(build_features_module, "_resolve_pre_ufc_supplement_path", lambda: Path("/nonexistent"))
    fights_df = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2024-01-01"),
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "winner": "Alpha Fighter",
                "weight_class": "Lightweight",
                "a_stance": None,
                "b_stance": "Orthodox",
            }
        ]
    )

    features = build_features_module.build_features(fights_df)
    row = features.iloc[0]

    assert pd.isna(row["a_stance_enc"])
    assert row["b_stance_enc"] == pytest.approx(0.0)
    assert pd.isna(row["same_stance"])


def test_build_features_encodes_additional_ufcstats_stance_labels(monkeypatch):
    monkeypatch.setattr(build_features_module, "_resolve_pre_ufc_supplement_path", lambda: Path("/nonexistent"))
    fights_df = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2024-01-01"),
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "winner": "Alpha Fighter",
                "weight_class": "Lightweight",
                "a_stance": "Open Stance",
                "b_stance": "Sideways",
            }
        ]
    )

    features = build_features_module.build_features(fights_df)
    row = features.iloc[0]

    assert row["a_stance_enc"] == pytest.approx(3.0)
    assert row["b_stance_enc"] == pytest.approx(4.0)
    assert row["same_stance"] == pytest.approx(0.0)


def test_full_live_contract_v4_live_lookup_encodes_open_stance(monkeypatch):
    profile_html = """
    <html><body>
      <h2 class="b-content__title"><span>Beta Fighter</span></h2>
      <span class="b-content__title-record">Record: 5-0-0</span>
      <li class="b-list__box-list-item">Height: 5' 9"</li>
      <li class="b-list__box-list-item">Weight: 145 lbs.</li>
      <li class="b-list__box-list-item">Reach: 72"</li>
      <li class="b-list__box-list-item">STANCE: Open Stance</li>
      <li class="b-list__box-list-item">DOB: Apr 27, 1995</li>
    </body></html>
    """

    monkeypatch.setattr(fighter_lookup, "search_fighter_url", lambda *_args, **_kwargs: "http://example.test/fighter")
    monkeypatch.setattr(
        fighter_lookup,
        "_get_soup",
        lambda _url: BeautifulSoup(profile_html, "lxml"),
    )

    fighter_lookup.clear_cache()

    result = fighter_lookup.lookup_fighter(
        "Beta Fighter",
        training_spec=training_spec.full_live_contract_v4_spec(),
    )

    assert result is not None
    assert result["features"]["stance_enc"] == pytest.approx(3.0)

    fighter_lookup.clear_cache()


def test_scrape_fight_detail_sig_strikes_uses_summary_table(monkeypatch):
    """Sig strikes should be parsed from the second summary table, not the
    per-round table that has the js-fight-table class."""
    # Build a realistic 4-table HTML structure:
    #   table 0: Totals summary
    #   table 1: Totals per-round (js-fight-table) — wrong data if selected
    #   table 2: Sig Strikes summary — correct data
    #   table 3: Sig Strikes per-round (js-fight-table)
    def _make_row(fighter_a, fighter_b, cols_data):
        """Build a single <tr> with paired <p> cells."""
        cells = []
        for a_val, b_val in cols_data:
            cells.append(
                f"<td><p>{a_val}</p><p>{b_val}</p></td>"
            )
        return f'<tr class="b-fight-details__table-row">{"".join(cells)}</tr>'

    # Totals summary (table 0) — 10 columns minimum
    totals_cols = [
        ("Alpha Fighter", "Beta Fighter"),  # col 0: names
        *[("0", "0")] * 7,                  # cols 1-7: placeholder stats
        ("3", "1"),                          # col 8: rev
        ("2:30", "1:15"),                    # col 9: ctrl
    ]
    totals_row = _make_row("Alpha Fighter", "Beta Fighter", totals_cols)
    totals_table = f'<table><tbody>{totals_row}</tbody></table>'

    # Per-round totals (table 1) — has js-fight-table, should be skipped
    wrong_cols = [
        ("Alpha Fighter", "Beta Fighter"),
        *[("0", "0")] * 7,
        ("99 of 99", "99 of 99"),  # garbage data that would be wrong if parsed
        *[("99 of 99", "99 of 99")] * 5,
    ]
    wrong_row = _make_row("Alpha Fighter", "Beta Fighter", wrong_cols)
    wrong_table = f'<table class="b-fight-details__table js-fight-table"><tbody>{wrong_row}</tbody></table>'

    # Sig Strikes summary (table 2) — correct data
    sig_cols = [
        ("Alpha Fighter", "Beta Fighter"),  # col 0: names
        ("50 of 100", "30 of 80"),           # col 1: Sig.Str
        ("50%", "37%"),                      # col 2: Sig.Str.%
        ("20 of 40", "10 of 30"),            # col 3: Head
        ("15 of 25", "8 of 20"),             # col 4: Body
        ("10 of 20", "7 of 15"),             # col 5: Leg
        ("30 of 60", "18 of 50"),            # col 6: Distance
        ("12 of 20", "6 of 15"),             # col 7: Clinch
        ("8 of 20", "6 of 15"),              # col 8: Ground
    ]
    sig_row = _make_row("Alpha Fighter", "Beta Fighter", sig_cols)
    sig_table = f'<table><tbody>{sig_row}</tbody></table>'

    # Per-round sig strikes (table 3) — also js-fight-table, should be skipped
    wrong_sig_table = f'<table class="b-fight-details__table js-fight-table"><tbody>{wrong_row}</tbody></table>'

    html = f"""<html><body>
    <i class="b-fight-details__fight-title">Lightweight Bout</i>
    <section class="b-fight-details__section js-fight-section">
    {totals_table}
    {wrong_table}
    </section>
    <section class="b-fight-details__section js-fight-section">
      <p class="b-fight-details__collapse-link_tot">Significant Strikes</p>
    </section>
    {sig_table}
    {wrong_sig_table}
    </body></html>"""

    monkeypatch.setattr(
        fighter_lookup,
        "_get_soup",
        lambda url: BeautifulSoup(html, "html.parser"),
    )

    result = fighter_lookup._scrape_fight_detail(
        "http://example.test/fight-details/abc123",
        "Alpha Fighter",
    )

    # Verify sig strikes came from the correct summary table
    assert result["head_landed"] == 20.0
    assert result["head_attempted"] == 40.0
    assert result["body_landed"] == 15.0
    assert result["leg_landed"] == 10.0
    assert result["distance_landed"] == 30.0
    assert result["clinch_landed"] == 12.0
    assert result["ground_landed"] == 8.0

    # Opponent values
    assert result["opp_head_landed"] == 10.0
    assert result["opp_body_landed"] == 8.0

    # Make sure we didn't get garbage from the per-round tables
    assert result["head_landed"] != 99.0
