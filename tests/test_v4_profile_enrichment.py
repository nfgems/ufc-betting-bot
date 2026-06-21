import logging
import os
import platform
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
from bs4 import BeautifulSoup

import scripts.backfill_active_roster_ufcstats as roster_backfill
import scripts.audit_active_roster_profile_completeness as roster_profile_audit
import scripts.build_profile_supplement_from_external_profiles as external_profiles
import scripts.build_profile_supplement_from_martialbot as martialbot_supplement
from src.data import fallback_scrapers, fighter_lookup, scraper, ufc_active_roster, ufc_refresh, ufcstats_http
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


def test_build_training_rows_from_pulled_data_uses_scraped_fighter_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(
        build_features_module,
        "_resolve_pre_ufc_supplement_path",
        lambda: tmp_path / "missing_pre_ufc.csv",
    )

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

    def fake_request_ufcstats(url, **_kwargs):
        return fake_get(url)

    monkeypatch.setattr(ufc_refresh.requests, "get", fake_get)
    monkeypatch.setattr(ufc_refresh, "request_ufcstats", fake_request_ufcstats)
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


def test_external_profile_candidates_skip_power_slap_rows(tmp_path):
    scraped_path = tmp_path / "ufc_fighters_scraped.csv"
    roster_path = tmp_path / "ufc_active_roster_official.csv"

    pd.DataFrame(
        [
            {
                "name": "Jae Hyun Park",
                "height": "",
                "reach": "",
                "weight": "",
                "stance": "",
                "dob": "",
            },
        ]
    ).to_csv(scraped_path, index=False)

    pd.DataFrame(
        [
            {
                "official_name": "Jonathan Correa",
                "profile_name": "Jonathan Correa",
                "slug_name": "jonathan correa",
                "alternate_slug_names": "",
                "combat_sport": "power_slap",
                "coverage_eligible": False,
            },
            {
                "official_name": "Jae Hyun Park",
                "profile_name": "Jae Hyun Park",
                "slug_name": "jae hyun park",
                "alternate_slug_names": "",
                "combat_sport": "mma",
                "coverage_eligible": True,
            },
        ]
    ).to_csv(roster_path, index=False)

    candidate_universe, candidates = external_profiles._load_candidates(
        scraped_path,
        candidate_source_csv=roster_path,
    )

    assert "Jonathan Correa" not in set(candidate_universe["name"])
    assert set(candidates["name"]) == {"Jae Hyun Park"}


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
      <link rel="canonical" href="https://www.ufc.com/athlete/isaac-thomson">
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
    assert profile["canonical_athlete_url"] == "https://www.ufc.com/athlete/isaac-thomson"


def test_sync_official_active_roster_reuses_cached_snapshot_when_live_sync_times_out(tmp_path, monkeypatch):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    expected_rows = [
        {
            "official_name": "Cached Fighter",
            "official_athlete_url": "https://www.ufc.com/athlete/cached-fighter",
            "ufcstats_url": "http://ufcstats.test/cached-fighter",
            "profile_status": "Active",
        }
    ]
    pd.DataFrame(expected_rows).to_csv(roster_path, index=False)

    def fail_scrape(**_kwargs):
        raise requests.exceptions.ReadTimeout(
            "HTTPSConnectionPool(host='www.ufc.com', port=443): Read timed out. (read timeout=30)"
        )

    monkeypatch.setattr(ufc_active_roster, "scrape_official_active_roster", fail_scrape)

    synced = ufc_active_roster.sync_official_active_roster(output_path=roster_path)

    assert synced.to_dict(orient="records") == expected_rows
    assert synced.attrs["sync_source"] == "cached"
    assert synced.attrs["sync_fallback_used"] is True
    assert "Read timed out" in synced.attrs["sync_error"]
    assert synced.attrs["sync_cached_snapshot_mtime_utc"]


def test_sync_official_active_roster_reuses_cached_snapshot_on_suspicious_live_growth(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    cached_rows = [
        {
            "official_name": f"Cached Fighter {index}",
            "official_athlete_url": f"https://www.ufc.com/athlete/cached-fighter-{index}",
            "ufcstats_url": f"http://ufcstats.test/cached-fighter-{index}",
            "profile_status": "Active",
        }
        for index in range(300)
    ]
    pd.DataFrame(cached_rows).to_csv(roster_path, index=False)

    live_df = pd.DataFrame(
        [
            {
                "official_name": f"Live Fighter {index}",
                "official_athlete_url": f"https://www.ufc.com/athlete/live-fighter-{index}",
                "ufcstats_url": f"http://ufcstats.test/live-fighter-{index}",
                "profile_status": "Active",
            }
            for index in range(700)
        ]
    )
    live_df.attrs["identity_audit_rows"] = []
    monkeypatch.setattr(ufc_active_roster, "scrape_official_active_roster", lambda **_kwargs: live_df)

    synced = ufc_active_roster.sync_official_active_roster(output_path=roster_path)

    assert synced["official_name"].tolist() == [row["official_name"] for row in cached_rows]
    assert synced.attrs["sync_source"] == "cached"
    assert synced.attrs["sync_fallback_used"] is True
    assert "exceeds the growth guard" in synced.attrs["sync_error"]


def test_sync_official_active_roster_discards_suspicious_oversized_cached_roster(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    cached_rows = [
        {
            "official_name": f"Cached Fighter {index}",
            "official_athlete_url": f"https://www.ufc.com/athlete/cached-fighter-{index}",
            "ufcstats_url": f"http://ufcstats.test/cached-fighter-{index}",
            "profile_status": "Active",
        }
        for index in range(700)
    ]
    pd.DataFrame(cached_rows).to_csv(roster_path, index=False)

    live_rows = [
        {
            "official_name": f"Live Fighter {index}",
            "official_athlete_url": f"https://www.ufc.com/athlete/live-fighter-{index}",
            "ufcstats_url": f"http://ufcstats.test/live-fighter-{index}",
            "profile_status": "Active",
        }
        for index in range(300)
    ]
    live_df = pd.DataFrame(live_rows)
    live_df.attrs["identity_audit_rows"] = []
    monkeypatch.setattr(ufc_active_roster, "scrape_official_active_roster", lambda **_kwargs: live_df)

    synced = ufc_active_roster.sync_official_active_roster(output_path=roster_path)
    saved = pd.read_csv(roster_path)

    assert synced["official_name"].tolist() == [row["official_name"] for row in live_rows]
    assert saved["official_name"].tolist() == [row["official_name"] for row in live_rows]
    assert synced.attrs["sync_source"] == "live"
    assert synced.attrs["retained_missing_live_rows"] == []
    assert synced.attrs["discarded_suspicious_cached_rows"] == 400


def test_sync_official_active_roster_retains_cached_rows_missing_from_live_sync(tmp_path, monkeypatch):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    pd.DataFrame(
        [
            {
                "official_name": "Still Active",
                "official_athlete_url": "https://www.ufc.com/athlete/still-active",
                "ufcstats_url": "http://ufcstats.test/still-active",
                "profile_status": "Active",
            },
            {
                "official_name": "Omitted Fighter",
                "official_athlete_url": "https://www.ufc.com/athlete/omitted-fighter",
                "ufcstats_url": "http://ufcstats.test/omitted-fighter",
                "profile_status": "Active",
            },
        ]
    ).to_csv(roster_path, index=False)

    live_df = pd.DataFrame(
        [
            {
                "official_name": "Still Active",
                "official_athlete_url": "https://www.ufc.com/athlete/still-active",
                "ufcstats_url": "http://ufcstats.test/still-active",
                "profile_status": "Active",
            }
        ]
    )
    live_df.attrs["identity_audit_rows"] = []
    monkeypatch.setattr(ufc_active_roster, "scrape_official_active_roster", lambda **_kwargs: live_df)

    synced = ufc_active_roster.sync_official_active_roster(output_path=roster_path)
    saved = pd.read_csv(roster_path)

    assert len(synced) == 2
    assert set(saved["official_name"]) == {"Still Active", "Omitted Fighter"}
    retained = saved.loc[saved["official_name"].eq("Omitted Fighter")].iloc[0]
    assert str(retained["active_roster_live_present"]).lower() == "false"
    assert str(retained["active_roster_retained_from_previous"]).lower() == "true"
    assert retained["active_roster_missing_from_live_reason"] == "absent_from_latest_ufc_active_roster_sync"
    assert str(retained["coverage_eligible"]).lower() in {"false", "0", "0.0"}
    assert synced.attrs["retained_missing_live_rows"] == [
        {
            "official_name": "Omitted Fighter",
            "official_athlete_url": "https://www.ufc.com/athlete/omitted-fighter",
            "ufcstats_url": "http://ufcstats.test/omitted-fighter",
        }
    ]


def test_sync_official_active_roster_matches_cached_row_when_live_url_alias_changes(tmp_path, monkeypatch):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    pd.DataFrame(
        [
            {
                "official_name": "Anthony Hernandez",
                "profile_name": "Anthony Hernandez",
                "official_athlete_url": "https://www.ufc.com/athlete/anthony-hernandez",
                "ufcstats_url": "http://ufcstats.test/anthony-hernandez",
                "profile_status": "Active",
            }
        ]
    ).to_csv(roster_path, index=False)

    live_df = pd.DataFrame(
        [
            {
                "official_name": "Anthony Hernandez",
                "profile_name": "Anthony Hernandez",
                "slug_name": "",
                "official_athlete_url": "https://www.ufc.com/athlete/ansoni-herunantesu",
                "ufcstats_url": "http://ufcstats.test/anthony-hernandez",
                "profile_status": "Active",
                "official_url_identity_status": "slug_mismatch_profile_valid",
            }
        ]
    )
    live_df.attrs["identity_audit_rows"] = []
    monkeypatch.setattr(ufc_active_roster, "scrape_official_active_roster", lambda **_kwargs: live_df)

    synced = ufc_active_roster.sync_official_active_roster(output_path=roster_path)
    saved = pd.read_csv(roster_path)

    assert len(synced) == 1
    assert synced.attrs["retained_missing_live_rows"] == []
    assert saved.loc[0, "official_name"] == "Anthony Hernandez"
    assert saved.loc[0, "official_athlete_url"] == "https://www.ufc.com/athlete/anthony-hernandez"
    assert saved.loc[0, "slug_name"] == "anthony hernandez"


def test_sync_official_active_roster_does_not_retain_alias_duplicate_when_live_count_drops(tmp_path, monkeypatch):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    pd.DataFrame(
        [
            {
                "official_name": "Anthony Hernandez",
                "profile_name": "Anthony Hernandez",
                "official_athlete_url": "https://www.ufc.com/athlete/anthony-hernandez",
                "ufcstats_url": "http://ufcstats.test/anthony-hernandez",
                "profile_status": "Active",
            },
            {
                "official_name": "Actually Omitted",
                "profile_name": "Actually Omitted",
                "official_athlete_url": "https://www.ufc.com/athlete/actually-omitted",
                "ufcstats_url": "http://ufcstats.test/actually-omitted",
                "profile_status": "Active",
            },
        ]
    ).to_csv(roster_path, index=False)

    live_df = pd.DataFrame(
        [
            {
                "official_name": "Anthony Hernandez",
                "profile_name": "Anthony Hernandez",
                "slug_name": "",
                "official_athlete_url": "https://www.ufc.com/athlete/ansoni-herunantesu",
                "ufcstats_url": "http://ufcstats.test/anthony-hernandez",
                "profile_status": "Active",
                "official_url_identity_status": "slug_mismatch_profile_valid",
            }
        ]
    )
    live_df.attrs["identity_audit_rows"] = []
    monkeypatch.setattr(ufc_active_roster, "scrape_official_active_roster", lambda **_kwargs: live_df)

    synced = ufc_active_roster.sync_official_active_roster(output_path=roster_path)
    saved = pd.read_csv(roster_path)

    assert len(synced) == 2
    assert saved["official_name"].tolist() == ["Actually Omitted", "Anthony Hernandez"]
    assert saved.loc[saved["official_name"].eq("Anthony Hernandez"), "official_athlete_url"].iloc[0] == (
        "https://www.ufc.com/athlete/anthony-hernandez"
    )
    assert synced.attrs["retained_missing_live_rows"] == [
        {
            "official_name": "Actually Omitted",
            "official_athlete_url": "https://www.ufc.com/athlete/actually-omitted",
            "ufcstats_url": "http://ufcstats.test/actually-omitted",
        }
    ]


def test_sync_official_active_roster_does_not_merge_same_name_with_conflicting_ufcstats_url(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    pd.DataFrame(
        [
            {
                "official_name": "Shared Name",
                "profile_name": "Shared Name",
                "official_athlete_url": "https://www.ufc.com/athlete/shared-name-old",
                "ufcstats_url": "http://ufcstats.test/shared-name-old",
                "profile_status": "Active",
            },
            {
                "official_name": "Anchor Fighter",
                "profile_name": "Anchor Fighter",
                "official_athlete_url": "https://www.ufc.com/athlete/anchor-fighter",
                "ufcstats_url": "http://ufcstats.test/anchor-fighter",
                "profile_status": "Active",
            },
        ]
    ).to_csv(roster_path, index=False)

    live_df = pd.DataFrame(
        [
            {
                "official_name": "Shared Name",
                "profile_name": "Shared Name",
                "official_athlete_url": "https://www.ufc.com/athlete/shared-name-new",
                "ufcstats_url": "http://ufcstats.test/shared-name-new",
                "profile_status": "Active",
            }
        ]
    )
    live_df.attrs["identity_audit_rows"] = []
    monkeypatch.setattr(ufc_active_roster, "scrape_official_active_roster", lambda **_kwargs: live_df)

    synced = ufc_active_roster.sync_official_active_roster(output_path=roster_path)

    assert len(synced) == 3
    assert synced.attrs["retained_missing_live_rows"] == [
        {
            "official_name": "Shared Name",
            "official_athlete_url": "https://www.ufc.com/athlete/shared-name-old",
            "ufcstats_url": "http://ufcstats.test/shared-name-old",
        },
        {
            "official_name": "Anchor Fighter",
            "official_athlete_url": "https://www.ufc.com/athlete/anchor-fighter",
            "ufcstats_url": "http://ufcstats.test/anchor-fighter",
        },
    ]


def test_sync_official_active_roster_does_not_retain_intentionally_excluded_test_rows(tmp_path, monkeypatch):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    pd.DataFrame(
        [
            {
                "official_name": "Real Fighter",
                "official_athlete_url": "https://www.ufc.com/athlete/real-fighter",
                "ufcstats_url": "http://ufcstats.test/real-fighter",
            },
            {
                "official_name": "Testy Test",
                "official_athlete_url": "https://www.ufc.com/athlete/testy-test",
                "ufcstats_url": "",
            },
        ]
    ).to_csv(roster_path, index=False)

    live_df = pd.DataFrame(
        [
            {
                "official_name": "Real Fighter",
                "official_athlete_url": "https://www.ufc.com/athlete/real-fighter",
                "ufcstats_url": "http://ufcstats.test/real-fighter",
            }
        ]
    )
    live_df.attrs["identity_audit_rows"] = [
        {
            "official_name": "Testy Test",
            "official_athlete_url": "https://www.ufc.com/athlete/testy-test",
            "profile_name": "",
            "slug_name": "testy test",
            "identity_status": "test_profile",
            "identity_reason": "test_or_staging_profile",
            "action": "excluded_test_profile",
        }
    ]
    monkeypatch.setattr(ufc_active_roster, "scrape_official_active_roster", lambda **_kwargs: live_df)

    synced = ufc_active_roster.sync_official_active_roster(output_path=roster_path)

    assert synced["official_name"].tolist() == ["Real Fighter"]
    assert synced.attrs["retained_missing_live_rows"] == []


def test_official_roster_identity_mismatch_does_not_use_slug_alias_for_ufcstats_match():
    row = {
        "official_name": "Right Fighter",
        "profile_name": "Wrong Fighter",
        "slug_name": "wrong-fighter",
        "alternate_slug_names": "Wrong F",
    }
    row.update(ufc_active_roster._validate_official_url_identity(row))

    candidates = {
        "right fighter": [
            {
                "name": "Right Fighter",
                "fighter_url": "http://ufcstats.test/right",
                "source": "fixture",
            }
        ],
        "wrong fighter": [
            {
                "name": "Wrong Fighter",
                "fighter_url": "http://ufcstats.test/wrong",
                "source": "fixture",
            }
        ],
    }

    resolved = ufc_active_roster._resolve_local_ufcstats_profile(row, candidates=candidates)

    assert row["official_url_identity_status"] == "mismatch"
    assert resolved["ufcstats_url"] == "http://ufcstats.test/right"


def test_official_roster_accepts_curated_slug_identity_alias():
    row = {
        "official_name": "King Green",
        "profile_name": "King Green",
        "slug_name": "bobby green",
        "alternate_slug_names": "",
    }
    row.update(ufc_active_roster._validate_official_url_identity(row))

    assert row["official_url_identity_valid"] is True
    assert row["official_url_identity_status"] == "valid"
    assert ufc_active_roster._official_url_identity_trusted(row) is True


def test_official_roster_keeps_profile_fields_when_only_slug_alias_mismatches(monkeypatch):
    roster_html = """
    <html><body>
      <div class="c-listing-athlete-flipcard">
        <a href="/athlete/patricio-freire">Profile</a>
        <span class="c-listing-athlete__name">Patricio Pitbull</span>
        <span class="c-listing-athlete__title"><span class="field__item">Featherweight</span></span>
      </div>
    </body></html>
    """
    profile_html = """
    <html><body>
      <h1 class="hero-profile__name">Patricio Pitbull</h1>
      <p class="hero-profile__division-title">Featherweight Division</p>
      <div class="c-bio__field"><div class="c-bio__label">Height</div><div class="c-bio__text">65.00</div></div>
      <div class="c-bio__field"><div class="c-bio__label">Reach</div><div class="c-bio__text">65.50</div></div>
      <div class="c-bio__field"><div class="c-bio__label">Weight</div><div class="c-bio__text">145.00</div></div>
    </body></html>
    """

    def fake_get_soup(url, session=None):
        if "patricio-freire" in url:
            return BeautifulSoup(profile_html, "lxml")
        return BeautifulSoup(roster_html, "lxml")

    monkeypatch.setattr(ufc_active_roster, "_get_soup", fake_get_soup)
    monkeypatch.setattr(
        ufc_active_roster,
        "_classify_combat_sport",
        lambda row, session=None: {
            "combat_sport": "mma",
            "combat_sport_reason": "",
            "combat_sport_profile_url": "",
        },
    )

    df = ufc_active_roster.scrape_official_active_roster(
        fetch_profile_details=True,
        max_pages=1,
        resolve_ufcstats=False,
    )
    row = df.iloc[0]

    assert row["official_url_identity_status"] == "slug_mismatch_profile_valid"
    assert row["profile_name"] == "Patricio Pitbull"
    assert row["height"] == "65.00 in"
    assert row["slug_name"] == ""
    assert df.attrs["identity_audit_rows"][0]["action"] == "quarantined_untrusted_slug_alias"


def test_official_roster_uses_profile_canonical_url_for_localized_card_alias(monkeypatch):
    roster_html = """
    <html><body>
      <div class="c-listing-athlete-flipcard">
        <a href="/athlete/ansoni-herunantesu">Profile</a>
        <span class="c-listing-athlete__name">Anthony Hernandez</span>
        <span class="c-listing-athlete__title"><span class="field__item">Middleweight</span></span>
      </div>
    </body></html>
    """
    profile_html = """
    <html><head>
      <link rel="canonical" href="https://www.ufc.com/athlete/anthony-hernandez">
    </head><body>
      <h1 class="hero-profile__name">Anthony Hernandez</h1>
      <p class="hero-profile__division-title">Middleweight Division</p>
      <p class="hero-profile__tag">Active</p>
    </body></html>
    """

    def fake_get_soup(url, session=None):
        if "ansoni-herunantesu" in url:
            return BeautifulSoup(profile_html, "lxml")
        return BeautifulSoup(roster_html, "lxml")

    monkeypatch.setattr(ufc_active_roster, "_get_soup", fake_get_soup)
    monkeypatch.setattr(
        ufc_active_roster,
        "_classify_combat_sport",
        lambda row, session=None: {
            "combat_sport": "mma",
            "combat_sport_reason": "",
            "combat_sport_profile_url": "",
        },
    )

    df = ufc_active_roster.scrape_official_active_roster(
        fetch_profile_details=True,
        max_pages=1,
        resolve_ufcstats=False,
    )
    row = df.iloc[0]

    assert row["official_athlete_url"] == "https://www.ufc.com/athlete/anthony-hernandez"
    assert row["slug_name"] == "anthony hernandez"
    assert row["official_url_identity_status"] == "valid"
    assert df.attrs["identity_audit_rows"] == []


def test_official_roster_dedupes_cards_after_canonical_url_rewrite(monkeypatch):
    roster_html = """
    <html><body>
      <div class="c-listing-athlete-flipcard">
        <a href="/athlete/ansoni-herunantesu">Profile</a>
        <span class="c-listing-athlete__name">Anthony Hernandez</span>
      </div>
      <div class="c-listing-athlete-flipcard">
        <a href="/athlete/anthony-hernandez">Profile</a>
        <span class="c-listing-athlete__name">Anthony Hernandez</span>
      </div>
    </body></html>
    """
    profile_html = """
    <html><head>
      <link rel="canonical" href="https://www.ufc.com/athlete/anthony-hernandez">
    </head><body>
      <h1 class="hero-profile__name">Anthony Hernandez</h1>
      <p class="hero-profile__tag">Active</p>
    </body></html>
    """

    def fake_get_soup(url, session=None):
        if "/athlete/" in url:
            return BeautifulSoup(profile_html, "lxml")
        return BeautifulSoup(roster_html, "lxml")

    monkeypatch.setattr(ufc_active_roster, "_get_soup", fake_get_soup)
    monkeypatch.setattr(
        ufc_active_roster,
        "_classify_combat_sport",
        lambda row, session=None: {
            "combat_sport": "mma",
            "combat_sport_reason": "",
            "combat_sport_profile_url": "",
        },
    )

    df = ufc_active_roster.scrape_official_active_roster(
        fetch_profile_details=True,
        max_pages=1,
        resolve_ufcstats=False,
    )

    assert df["official_athlete_url"].tolist() == ["https://www.ufc.com/athlete/anthony-hernandez"]


def test_scrape_official_active_roster_excludes_test_profiles_from_output(monkeypatch):
    roster_html = """
    <html><body>
      <div class="c-listing-athlete-flipcard">
        <a href="/athlete/testy-test">Profile</a>
        <span class="c-listing-athlete__name">Testy Test</span>
        <span class="c-listing-athlete__title"><span class="field__item">Heavyweight</span></span>
      </div>
    </body></html>
    """

    def fake_get_soup(_url, session=None):
        return BeautifulSoup(roster_html, "lxml")

    monkeypatch.setattr(ufc_active_roster, "_get_soup", fake_get_soup)

    df = ufc_active_roster.scrape_official_active_roster(
        fetch_profile_details=False,
        max_pages=1,
        resolve_ufcstats=False,
    )

    assert df.empty
    assert df.attrs["identity_audit_rows"] == [
        {
            "official_name": "Testy Test",
            "official_athlete_url": "https://www.ufc.com/athlete/testy-test",
            "profile_name": "",
            "slug_name": "testy test",
            "identity_status": "test_profile",
            "identity_reason": "test_or_staging_profile",
            "action": "excluded_test_profile",
        }
    ]


def test_scrape_official_active_roster_excludes_inactive_profile_statuses(monkeypatch):
    roster_html = """
    <html><body>
      <div class="c-listing-athlete-flipcard">
        <a href="/athlete/current-fighter">Profile</a>
        <span class="c-listing-athlete__name">Current Fighter</span>
        <span class="c-listing-athlete__title"><span class="field__item">Welterweight</span></span>
      </div>
      <div class="c-listing-athlete-flipcard">
        <a href="/athlete/retired-fighter">Profile</a>
        <span class="c-listing-athlete__name">Retired Fighter</span>
        <span class="c-listing-athlete__title"><span class="field__item">Heavyweight</span></span>
      </div>
      <div class="c-listing-athlete-flipcard">
        <a href="/athlete/not-fighting">Profile</a>
        <span class="c-listing-athlete__name">Not Fighting</span>
        <span class="c-listing-athlete__title"><span class="field__item">Lightweight</span></span>
      </div>
    </body></html>
    """
    active_profile_html = """
    <html><body>
      <h1 class="hero-profile__name">Current Fighter</h1>
      <p class="hero-profile__tag">Welterweight Division</p>
      <p class="hero-profile__tag">Active</p>
    </body></html>
    """
    retired_profile_html = """
    <html><body>
      <h1 class="hero-profile__name">Retired Fighter</h1>
      <p class="hero-profile__tag">Heavyweight Division</p>
      <p class="hero-profile__tag">Retired</p>
    </body></html>
    """
    not_fighting_profile_html = """
    <html><body>
      <h1 class="hero-profile__name">Not Fighting</h1>
      <p class="hero-profile__tag">Lightweight Division</p>
      <p class="hero-profile__tag">Not Fighting</p>
    </body></html>
    """

    def fake_get_soup(url, session=None):
        if "current-fighter" in url:
            return BeautifulSoup(active_profile_html, "lxml")
        if "retired-fighter" in url:
            return BeautifulSoup(retired_profile_html, "lxml")
        if "not-fighting" in url:
            return BeautifulSoup(not_fighting_profile_html, "lxml")
        return BeautifulSoup(roster_html, "lxml")

    monkeypatch.setattr(ufc_active_roster, "_get_soup", fake_get_soup)
    monkeypatch.setattr(
        ufc_active_roster,
        "_classify_combat_sport",
        lambda row, session=None: {
            "combat_sport": "mma",
            "combat_sport_reason": "",
            "combat_sport_profile_url": "",
        },
    )

    df = ufc_active_roster.scrape_official_active_roster(
        fetch_profile_details=True,
        max_pages=1,
        resolve_ufcstats=False,
    )

    assert df["official_name"].tolist() == ["Current Fighter"]
    assert df.attrs["identity_audit_rows"] == [
        {
            "official_name": "Retired Fighter",
            "official_athlete_url": "https://www.ufc.com/athlete/retired-fighter",
            "profile_name": "Retired Fighter",
            "slug_name": "retired fighter",
            "identity_status": "valid",
            "identity_reason": "",
            "action": "excluded_inactive_profile_status",
        },
        {
            "official_name": "Not Fighting",
            "official_athlete_url": "https://www.ufc.com/athlete/not-fighting",
            "profile_name": "Not Fighting",
            "slug_name": "not fighting",
            "identity_status": "valid",
            "identity_reason": "",
            "action": "excluded_inactive_profile_status",
        },
    ]


def test_sync_official_active_roster_writes_identity_audit_report(tmp_path, monkeypatch):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    identity_audit_path = tmp_path / "ufc_active_roster_identity_audit.csv"
    df = pd.DataFrame(
        [
            {
                "official_name": "Right Fighter",
                "official_athlete_url": "https://www.ufc.com/athlete/wrong-fighter",
                "official_url_identity_valid": False,
                "official_url_identity_status": "mismatch",
                "official_url_identity_reason": "slug_name_mismatch:wrong fighter",
            }
        ]
    )
    df.attrs["identity_audit_rows"] = [
        {
            "official_name": "Right Fighter",
            "official_athlete_url": "https://www.ufc.com/athlete/wrong-fighter",
            "profile_name": "Wrong Fighter",
            "slug_name": "wrong fighter",
            "identity_status": "mismatch",
            "identity_reason": "slug_name_mismatch:wrong fighter",
            "action": "quarantined_untrusted_url_identity",
        }
    ]

    monkeypatch.setattr(ufc_active_roster, "scrape_official_active_roster", lambda **_kwargs: df)

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=identity_audit_path,
    )
    audit_df = pd.read_csv(identity_audit_path)

    assert len(synced) == 1
    assert audit_df.loc[0, "action"] == "quarantined_untrusted_url_identity"


def test_classify_combat_sport_flags_power_slap_rows(monkeypatch):
    monkeypatch.setattr(
        ufc_active_roster,
        "_search_powerslap_profile",
        lambda fighter_name, session=None: (
            {
                "profile_name": "Jonathan Correa",
                "profile_url": "https://www.powerslap.com/striker/jonathan-correa/",
            }
            if fighter_name == "Jonathan Correa"
            else {}
        ),
    )

    classified = ufc_active_roster._classify_combat_sport(
        {
            "official_name": "Jonathan Correa",
            "profile_name": "Jonathan Correa",
            "slug_name": "jonathan correa",
            "alternate_slug_names": "",
            "ufcstats_url": "",
        }
    )

    assert classified == {
        "combat_sport": "power_slap",
        "combat_sport_reason": "powerslap_profile_match",
        "combat_sport_profile_url": "https://www.powerslap.com/striker/jonathan-correa/",
    }


def test_run_audit_excludes_power_slap_rows_from_coverage_summary(tmp_path):
    active_roster_path = tmp_path / "ufc_active_roster_official.csv"
    processed_fights_path = tmp_path / "fights_cleaned.csv"
    scraped_fighters_path = tmp_path / "ufc_fighters_scraped.csv"

    pd.DataFrame(
        [
            {
                "official_name": "Jonathan Correa",
                "profile_name": "Jonathan Correa",
                "slug_name": "jonathan correa",
                "alternate_slug_names": "",
                "combat_sport": "power_slap",
                "coverage_eligible": False,
                "age": "",
                "division": "Heavyweight",
                "weight": "",
                "ufcstats_url": "",
            },
            {
                "official_name": "Jae Hyun Park",
                "profile_name": "Jae Hyun Park",
                "slug_name": "jae hyun park",
                "alternate_slug_names": "",
                "combat_sport": "mma",
                "coverage_eligible": True,
                "age": 21,
                "division": "Lightweight",
                "weight": 155,
                "ufcstats_url": "",
            },
        ]
    ).to_csv(active_roster_path, index=False)
    pd.DataFrame(columns=["fighter_a", "fighter_b"]).to_csv(processed_fights_path, index=False)
    pd.DataFrame(
        [
            {
                "name": "Jae Hyun Park",
                "fighter_url": "",
                "height": '5\'10"',
                "reach": '72"',
                "weight": "155 lbs",
                "stance": "Orthodox",
                "dob": "2001-12-10",
            }
        ]
    ).to_csv(scraped_fighters_path, index=False)

    summary, audit_df = roster_profile_audit.run_audit(
        active_roster_path=active_roster_path,
        processed_fights_path=processed_fights_path,
        scraped_fighters_path=scraped_fighters_path,
    )

    assert summary["active_roster_rows"] == 2
    assert summary["coverage_eligible_active_roster_rows"] == 1
    assert summary["coverage_excluded_active_roster_rows"] == 1
    assert summary["coverage_excluded_by_sport"] == {"power_slap": 1}
    assert summary["overall_summary"]["rows"] == 1
    assert not bool(audit_df.loc[audit_df["official_name"] == "Jonathan Correa", "coverage_eligible"].iloc[0])


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


def test_martialbot_url_map_helper_uses_shared_profile_scraper(monkeypatch):
    captured: dict[str, str] = {}

    def fake_scrape_martialbot_profile(fighter_url: str) -> dict[str, object]:
        captured["fighter_url"] = fighter_url
        return {
            "name": "Tank Abbott",
            "height_raw": "183 cm",
            "reach_raw": "183 cm",
            "stance": "orthodox",
            "dob": "1965-04-26",
        }

    monkeypatch.setattr(
        martialbot_supplement,
        "scrape_martialbot_profile",
        fake_scrape_martialbot_profile,
    )

    row = pd.Series(
        {
            "name": "David Abbott",
            "fighter_url": "https://www.martialbot.com/mma/fighters/tank-abbott-abc123",
        }
    )

    supplement = martialbot_supplement._build_row(row, current_state={})

    assert captured["fighter_url"] == "https://www.martialbot.com/mma/fighters/tank-abbott-abc123"
    assert supplement == {
        "name": "David Abbott",
        "source": "martialbot",
        "source_name": "Tank Abbott",
        "search_name": "David Abbott",
        "fighter_url": "https://www.martialbot.com/mma/fighters/tank-abbott-abc123",
        "height": "183 cm",
        "reach": "183 cm",
        "weight": "",
        "stance": "orthodox",
        "dob": "1965-04-26",
    }


def test_martialbot_url_map_helper_updates_existing_blank_fields(tmp_path, monkeypatch):
    scraped_path = tmp_path / "ufc_fighters_scraped.csv"
    url_map_path = tmp_path / "martialbot_fighter_urls.csv"
    output_path = tmp_path / "ufc_fighters_profile_supplement.csv"

    pd.DataFrame(
        [
            {
                "name": "Dave Beneteau",
                "height": '6\' 2"',
                "reach": "--",
                "stance": "Orthodox",
                "dob": "--",
            }
        ]
    ).to_csv(scraped_path, index=False)
    pd.DataFrame(
        [
            {
                "name": "Dave Beneteau",
                "fighter_url": "https://www.martialbot.com/mma/fighters/dave-beneteau-abc123",
            }
        ]
    ).to_csv(url_map_path, index=False)
    pd.DataFrame(
        [
            {
                "name": "Dave Beneteau",
                "source": "martialbot",
                "source_name": "Dave Beneteau",
                "search_name": "Dave Beneteau",
                "fighter_url": "https://www.martialbot.com/mma/fighters/dave-beneteau-abc123",
                "height": "",
                "reach": "188 cm",
                "weight": "",
                "stance": "",
                "dob": "",
            }
        ]
    ).to_csv(output_path, index=False)

    monkeypatch.setattr(
        martialbot_supplement,
        "scrape_martialbot_profile",
        lambda _url: {
            "name": "Dave Beneteau",
            "height_raw": "188 cm",
            "reach_raw": "188 cm",
            "stance": "orthodox",
            "dob": "1967-06-22",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_profile_supplement_from_martialbot.py",
            "--scraped-fighters-path",
            str(scraped_path),
            "--url-map-csv",
            str(url_map_path),
            "--output",
            str(output_path),
            "--json",
        ],
    )

    martialbot_supplement.main()

    saved = pd.read_csv(output_path)
    assert len(saved) == 1
    assert saved.loc[0, "reach"] == "188 cm"
    assert saved.loc[0, "dob"] == "1967-06-22"


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


def test_wiki_api_retries_429_with_retry_after(monkeypatch):
    sleeps: list[float] = []

    class _FakeResponse:
        def __init__(self, status_code: int, payload: dict | None = None, headers: dict | None = None):
            self.status_code = status_code
            self._payload = payload or {}
            self.headers = headers or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} error", response=self)

        def json(self):
            return self._payload

    class _FakeSession:
        def __init__(self):
            self.calls: list[dict[str, object]] = []
            self.responses = [
                _FakeResponse(429, headers={"Retry-After": "0.5"}),
                _FakeResponse(200, payload={"ok": True}),
            ]

        def get(self, _url, **kwargs):
            self.calls.append(kwargs)
            return self.responses.pop(0)

    monkeypatch.setattr(external_profiles.time, "sleep", lambda seconds: sleeps.append(seconds))

    session = _FakeSession()
    result = external_profiles._wiki_api(session, action="query", titles="Jane Doe")

    assert result == {"ok": True}
    assert sleeps == [0.5]
    assert len(session.calls) == 2


def test_wiki_api_uses_exponential_backoff_for_429_without_retry_after(monkeypatch):
    sleeps: list[float] = []

    class _FakeResponse:
        status_code = 429
        headers: dict[str, str] = {}

        def raise_for_status(self):
            raise requests.HTTPError("429 error", response=self)

        def json(self):
            return {}

    class _SuccessResponse:
        status_code = 200
        headers: dict[str, str] = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    class _FakeSession:
        def __init__(self):
            self.responses = [_FakeResponse(), _FakeResponse(), _SuccessResponse()]

        def get(self, *_args, **_kwargs):
            return self.responses.pop(0)

    monkeypatch.setattr(external_profiles.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = external_profiles._wiki_api(_FakeSession(), action="query", titles="Jane Doe")

    assert result == {"ok": True}
    assert sleeps == [10.0, 20.0]


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


def test_external_profile_builder_adds_espn_stance_and_reach(monkeypatch):
    row = pd.Series(
        {
            "name": "Jae Hyun Park",
            "search_names": "Jae Hyun Park",
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
        "search_espn",
        lambda _name: "https://www.espn.com/mma/fighter/_/id/5138589/jae-hyun-park",
    )
    monkeypatch.setattr(
        external_profiles,
        "scrape_espn_profile",
        lambda _url: {
            "name": "Jae Hyun Park",
            "height_raw": '5\' 10"',
            "reach_raw": '72"',
            "weight_raw": "155 lbs",
            "stance": "Orthodox",
            "dob": "2001-12-10",
        },
    )

    row_out = external_profiles._build_espn_row(row, current_state)

    assert row_out is not None
    assert row_out["source"] == "espn"
    assert row_out["reach"] == '72"'
    assert row_out["stance"] == "Orthodox"
    assert row_out["dob"] == "2001-12-10"


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


def test_profile_supplement_refresh_skips_later_sources_after_profile_complete(tmp_path, monkeypatch):
    scraped_path = tmp_path / "ufc_fighters_scraped.csv"
    output_path = tmp_path / "ufc_fighters_profile_supplement.csv"
    pd.DataFrame(
        [
            {
                "name": "Complete After MartialBot",
                "height": "",
                "reach": "",
                "weight": "170 lbs",
                "stance": "",
                "dob": "",
            }
        ]
    ).to_csv(scraped_path, index=False)

    monkeypatch.setattr(
        external_profiles,
        "search_martialbot",
        lambda _name: "https://www.martialbot.com/mma/fighters/complete-after-martialbot",
    )
    monkeypatch.setattr(
        external_profiles,
        "scrape_martialbot_profile",
        lambda _url: {
            "name": "Complete After MartialBot",
            "height_raw": "180 cm",
            "reach_raw": "184 cm",
            "stance": "orthodox",
            "dob": "1995-04-26",
        },
    )

    def fail_fightdx(_name):
        raise AssertionError("FightDX should be skipped after all recoverable gaps are filled")

    monkeypatch.setattr(external_profiles, "search_fightdx", fail_fightdx)

    summary = external_profiles.run_profile_supplement_refresh(
        scraped_fighters_path=scraped_path,
        output_path=output_path,
        sources=["martialbot", "fightdx"],
    )

    assert summary["recovered_rows"] == 1
    assert summary["recovered_by_source"]["martialbot"] == 1
    assert summary["recovered_by_source"]["fightdx"] == 0


def test_build_tapology_row_accepts_source_specific_search_alias(monkeypatch):
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
    searched_names = []

    def fake_search_tapology(name):
        searched_names.append(name)
        if name == "Abdul Azim Badakhshi":
            return "https://www.tapology.com/fightcenter/fighters/49423-abdul-azeem-badakhshi"
        return None

    monkeypatch.setattr(external_profiles, "search_tapology", fake_search_tapology)
    monkeypatch.setattr(
        external_profiles,
        "scrape_tapology_profile",
        lambda _url: {
            "name": "Abdul Azim Badakhshi",
            "height_raw": "5'9\" (174cm)",
            "reach_raw": "",
            "weight_raw": "144.2 lbs",
            "dob": "",
        },
    )

    result = external_profiles._build_tapology_row(row, {})

    assert result is not None
    assert result["name"] == "Abdul Azeem Badakhshi"
    assert result["source_name"] == "Abdul Azim Badakhshi"
    assert result["search_name"] == "Abdul Azim Badakhshi"
    assert result["height"] == "5'9\" (174cm)"
    assert result["weight"] == "144.2 lbs"
    assert searched_names[0] == "Abdul Azim Badakhshi"


def test_profile_supplement_refresh_skips_sources_that_cannot_fill_remaining_gap(tmp_path, monkeypatch):
    scraped_path = tmp_path / "ufc_fighters_scraped.csv"
    output_path = tmp_path / "ufc_fighters_profile_supplement.csv"
    pd.DataFrame(
        [
            {
                "name": "Stance Only Gap",
                "height": "180 cm",
                "reach": "184 cm",
                "weight": "170 lbs",
                "stance": "",
                "dob": "1995-04-26",
            }
        ]
    ).to_csv(scraped_path, index=False)

    monkeypatch.setattr(
        external_profiles,
        "search_tapology",
        lambda _name: pytest.fail("Tapology cannot recover stance and should be skipped"),
    )
    monkeypatch.setattr(
        external_profiles,
        "search_sherdog",
        lambda _name: pytest.fail("Sherdog cannot recover stance and should be skipped"),
    )

    summary = external_profiles.run_profile_supplement_refresh(
        scraped_fighters_path=scraped_path,
        output_path=output_path,
        sources=["tapology", "sherdog"],
    )

    assert summary["attempted_rows"] == 1
    assert summary["recovered_rows"] == 0


def test_append_missing_profiles_refreshes_incomplete_active_roster_profile(tmp_path, monkeypatch):
    fighters_path = tmp_path / "ufc_fighters_scraped.csv"
    failures_path = tmp_path / "ufcstats_profile_scrape_failures.csv"
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
    monkeypatch.setattr(roster_backfill, "PROFILE_SCRAPE_FAILURES_PATH", failures_path)
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


def test_backfill_treats_ufcstats_loading_page_as_scrape_failure(monkeypatch):
    html = """
    <!doctype html>
    <html>
      <head>
        <title>Loading...</title>
        <meta name="robots" content="noindex">
      </head>
      <body>Please wait.</body>
    </html>
    """

    monkeypatch.setattr(
        roster_backfill,
        "_get_soup",
        lambda _url, *, session: BeautifulSoup(html, "lxml"),
    )

    with pytest.raises(ValueError, match="did not expose profile markup"):
        roster_backfill._extract_completed_fight_urls(
            "http://ufcstats.com/fighter-details/blocked",
            session=requests.Session(),
        )


def test_ufcstats_request_solves_browser_check_challenge():
    challenge_html = """
    <!doctype html><html><head>
      <title>Loading...</title><meta name="robots" content="noindex">
    </head><body>
      <p>Checking your browser...</p>
      <noscript>This site requires JavaScript.</noscript>
      <script>
        var nonce="abc123",
            target=new Array(1+1).join('0');
        var xhr=new XMLHttpRequest();
        xhr.open('POST',"/__c",true);
      </script>
    </body></html>
    """
    profile_html = """
    <html><head><title>Stats | UFC</title></head>
      <body><h2 class="b-content__title"><span>Recovered Fighter</span></h2></body>
    </html>
    """

    class FakeResponse:
        def __init__(self, text: str, *, url: str = "http://ufcstats.test/fighter", status_code: int = 200):
            self.text = text
            self.url = url
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(response=self)

    class FakeSession:
        def __init__(self):
            self.get_calls = 0
            self.post_calls: list[dict] = []

        def get(self, url, *, headers, timeout):
            self.get_calls += 1
            if self.get_calls == 1:
                return FakeResponse(challenge_html, url=url)
            return FakeResponse(profile_html, url=url)

        def post(self, url, *, headers, data, timeout):
            self.post_calls.append({"url": url, "headers": headers, "data": data})
            return FakeResponse("", url=url, status_code=204)

    session = FakeSession()

    response = ufcstats_http.request_ufcstats(
        "http://ufcstats.test/fighter",
        session=session,
    )

    assert response.text == profile_html
    assert session.get_calls == 2
    assert session.post_calls[0]["url"] == "http://ufcstats.test/__c"
    assert session.post_calls[0]["data"]["nonce"] == "abc123"
    assert int(session.post_calls[0]["data"]["n"]) >= 0


def test_ufcstats_request_rewrites_https_ufcstats_urls_to_http():
    class FakeResponse:
        def __init__(self, text: str, *, url: str, status_code: int = 200):
            self.text = text
            self.url = url
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(response=self)

    class FakeSession:
        def __init__(self):
            self.get_urls: list[str] = []

        def get(self, url, *, headers, timeout):
            self.get_urls.append(url)
            return FakeResponse("<html><body>ok</body></html>", url=url)

    session = FakeSession()

    response = ufcstats_http.request_ufcstats(
        "https://ufcstats.com/fighter-details/d3df1add9d9a7efb",
        session=session,
    )

    assert response.text == "<html><body>ok</body></html>"
    assert session.get_urls == ["http://ufcstats.com/fighter-details/d3df1add9d9a7efb"]


def test_ufcstats_request_retries_transient_timeout(monkeypatch):
    sleeps: list[float] = []

    class FakeResponse:
        text = "<html><body>ok</body></html>"
        url = "http://ufcstats.test/fighter"
        status_code = 200

        def raise_for_status(self) -> None:
            return None

    class FakeSession:
        def __init__(self):
            self.get_calls = 0

        def get(self, url, *, headers, timeout):
            self.get_calls += 1
            if self.get_calls == 1:
                raise requests.Timeout("read timed out")
            return FakeResponse()

    monkeypatch.setattr(ufcstats_http.time, "sleep", lambda seconds: sleeps.append(seconds))
    session = FakeSession()

    response = ufcstats_http.request_ufcstats("http://ufcstats.test/fighter", session=session)

    assert response.text == "<html><body>ok</body></html>"
    assert session.get_calls == 2
    assert sleeps == [0.5]


def test_fighter_lookup_get_soup_uses_shared_ufcstats_request(monkeypatch):
    class FakeResponse:
        text = "<html><body><h1>ok</h1></body></html>"

    calls = []

    def fake_request_ufcstats(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr(fighter_lookup, "request_ufcstats", fake_request_ufcstats)
    monkeypatch.setattr(fighter_lookup, "REQUEST_DELAY", 0)

    soup = fighter_lookup._get_soup("https://ufcstats.com/fighter-details/d3df1add9d9a7efb")

    assert soup.select_one("h1").text == "ok"
    assert calls[0]["url"] == "https://ufcstats.com/fighter-details/d3df1add9d9a7efb"
    assert calls[0]["session"] is fighter_lookup._UFCSTATS_SESSION
    assert calls[0]["headers"]["Accept"]


def test_scrape_fighter_profile_canonicalizes_https_profile_url(monkeypatch):
    calls = []
    html = """
    <html><body>
      <h2 class="b-content__title"><span>Derrick Lewis</span></h2>
      <span class="b-content__title-record">Record: 29-13-0 (1 NC)</span>
    </body></html>
    """

    def fake_get_soup(url):
        calls.append(url)
        return BeautifulSoup(html, "lxml")

    monkeypatch.setattr(fighter_lookup, "_get_soup", fake_get_soup)

    profile = fighter_lookup.scrape_fighter_profile(
        "https://ufcstats.com/fighter-details/d3df1add9d9a7efb"
    )

    assert calls == ["http://ufcstats.com/fighter-details/d3df1add9d9a7efb"]
    assert profile["fighter_url"] == "http://ufcstats.com/fighter-details/d3df1add9d9a7efb"


def test_scrape_fighter_fights_canonicalizes_https_detail_urls(monkeypatch):
    html = """
    <html><body>
      <table>
        <tr class="b-fight-details__table-row" data-link="https://ufcstats.com/fight-details/example">
          <td><a class="b-flag">win</a></td>
          <td><p>Derrick Lewis</p><p>Test Opponent</p></td>
          <td><p>1</p><p>0</p></td>
          <td><p>10</p><p>5</p></td>
          <td><p>0</p><p>0</p></td>
          <td><p>0</p><p>0</p></td>
          <td>UFC Test Jan. 01, 2024</td>
          <td>KO/TKO</td>
          <td>1</td>
          <td>4:00</td>
        </tr>
      </table>
    </body></html>
    """

    monkeypatch.setattr(fighter_lookup, "_get_soup", lambda _url: BeautifulSoup(html, "lxml"))

    fights = fighter_lookup.scrape_fighter_fights(
        "https://ufcstats.com/fighter-details/d3df1add9d9a7efb"
    )

    assert fights[0]["detail_url"] == "http://ufcstats.com/fight-details/example"


def test_profile_backfill_reports_failure_reasons_and_writes_audit_csv(tmp_path, monkeypatch):
    fighters_path = tmp_path / "ufc_fighters_scraped.csv"
    failures_path = tmp_path / "ufcstats_profile_scrape_failures.csv"
    pd.DataFrame(
        [
            {
                "name": "Blocked Fighter",
                "fighter_url": "http://ufcstats.com/fighter-details/blocked",
                "height": "--",
                "weight": "155 lbs.",
                "reach": "",
                "stance": "",
                "dob": "",
            }
        ]
    ).to_csv(fighters_path, index=False)

    roster_df = pd.DataFrame(
        [
            {
                "official_name": "Blocked Fighter",
                "ufcstats_url": "http://ufcstats.com/fighter-details/blocked",
            }
        ]
    )

    monkeypatch.setattr(roster_backfill, "FIGHTERS_PATH", fighters_path)
    monkeypatch.setattr(roster_backfill, "PROFILE_SCRAPE_FAILURES_PATH", failures_path)

    def fake_profile(_url: str):
        raise roster_backfill.UFCStatsChallengeError("blocked")

    monkeypatch.setattr(roster_backfill, "_profile_row_from_url", fake_profile)

    summary = roster_backfill._append_missing_profiles_with_summary(roster_df)

    assert summary["scraped_profile_scrape_failures"] == 1
    assert summary["scraped_profile_scrape_failure_reasons"] == {
        "ufcstats_challenge_page": 1,
    }
    assert summary["scraped_profile_scrape_failure_details"][0]["fighter"] == "Blocked Fighter"
    assert summary["scraped_profile_scrape_failure_details"][0]["missing_fields"] == "height,reach,stance,dob"
    saved_failures = pd.read_csv(failures_path)
    assert saved_failures.loc[0, "failure_reason"] == "ufcstats_challenge_page"
    assert bool(saved_failures.loc[0, "cached_profile_present"]) is True


def test_profile_backfill_updates_columns_inferred_numeric_from_blank_csv(tmp_path, monkeypatch):
    fighters_path = tmp_path / "ufc_fighters_scraped.csv"
    failures_path = tmp_path / "ufcstats_profile_scrape_failures.csv"
    fighter_url = "http://ufcstats.com/fighter-details/polyana"
    pd.DataFrame(
        [
            {
                "name": "Polyana Viana",
                "fighter_url": fighter_url,
                "height": "--",
                "weight": "115 lbs.",
                "reach": "",
                "stance": "",
                "dob": "",
            }
        ]
    ).to_csv(fighters_path, index=False)

    monkeypatch.setattr(roster_backfill, "FIGHTERS_PATH", fighters_path)
    monkeypatch.setattr(roster_backfill, "PROFILE_SCRAPE_FAILURES_PATH", failures_path)
    monkeypatch.setattr(
        roster_backfill,
        "_profile_row_from_url",
        lambda _url: {
            "name": "Polyana Viana",
            "record": "13-9-0",
            "fighter_url": fighter_url,
            "height": '5\' 5"',
            "weight": "115 lbs.",
            "reach": '67"',
            "stance": "Orthodox",
            "dob": "Jun 14, 1992",
        },
    )

    summary = roster_backfill._append_missing_profiles_with_summary(
        pd.DataFrame(
            [
                {
                    "official_name": "Polyana Viana",
                    "ufcstats_url": fighter_url,
                }
            ]
        )
    )

    assert summary["scraped_profiles_updated"] == 1
    refreshed = pd.read_csv(fighters_path)
    assert refreshed.loc[0, "reach"] == '67"'
    assert refreshed.loc[0, "stance"] == "Orthodox"
    assert refreshed.loc[0, "dob"] == "Jun 14, 1992"


def test_backfill_valid_empty_ufcstats_profile_counts_as_no_completed_bouts(monkeypatch):
    html = """
    <html>
      <body>
        <h2 class="b-content__title">
          <span class="b-content__title-highlight">New Fighter</span>
        </h2>
        <table><tbody></tbody></table>
      </body>
    </html>
    """

    monkeypatch.setattr(
        roster_backfill,
        "_get_soup",
        lambda _url, *, session: BeautifulSoup(html, "lxml"),
    )

    fight_urls = roster_backfill._extract_completed_fight_urls(
        "http://ufcstats.com/fighter-details/new-fighter",
        session=requests.Session(),
    )

    assert fight_urls == []


def test_run_backfill_separates_fight_list_failures_from_no_bout_profiles(tmp_path, monkeypatch):
    results_path = tmp_path / "ufc-fight-results.csv"
    stats_path = tmp_path / "ufc-fight-stats.csv"
    fighters_path = tmp_path / "ufc_fighters_scraped.csv"
    pd.DataFrame(
        [
            {
                "EVENT": "Existing Event",
                "BOUT": "Veteran vs. Opponent",
                "OUTCOME": "W/L",
                "WEIGHTCLASS": "Lightweight",
                "METHOD": "Decision",
                "ROUND": "3",
                "TIME": "5:00",
                "TIME FORMAT": "3 Rnd (5-5-5)",
                "REFEREE": "",
                "DETAILS": "",
                "URL": "http://ufcstats.com/fight-details/existing",
            }
        ],
        columns=roster_backfill.RESULTS_COLUMNS,
    ).to_csv(results_path, index=False)
    pd.DataFrame(
        [
            {
                "EVENT": "Existing Event",
                "BOUT": "Veteran vs. Opponent",
                "ROUND": "1",
                "FIGHTER": "Veteran",
                "KD": "0",
                "SIG.STR.": "1 of 2",
                "SIG.STR. %": "50%",
                "TOTAL STR.": "1 of 2",
                "TD": "0 of 0",
                "TD %": "0%",
                "SUB.ATT": "0",
                "REV.": "0",
                "CTRL": "0:00",
                "HEAD": "1 of 2",
                "BODY": "0 of 0",
                "LEG": "0 of 0",
                "DISTANCE": "1 of 2",
                "CLINCH": "0 of 0",
                "GROUND": "0 of 0",
            }
        ],
        columns=roster_backfill.STATS_COLUMNS,
    ).to_csv(stats_path, index=False)
    pd.DataFrame(
        [
            {
                "name": "Veteran",
                "fighter_url": "http://ufcstats.com/fighter-details/veteran",
                "height": "70",
                "weight": "155",
                "reach": "72",
                "stance": "Orthodox",
                "dob": "Jan 01, 1990",
                "slpm": "1.0",
                "sapm": "1.0",
                "td_avg": "0.0",
                "sub_avg": "0.0",
                "str_acc": "50",
                "str_def": "50",
                "td_acc": "0",
                "td_def": "0",
            }
        ]
    ).to_csv(fighters_path, index=False)

    monkeypatch.setattr(roster_backfill, "RESULTS_PATH", results_path)
    monkeypatch.setattr(roster_backfill, "STATS_PATH", stats_path)
    monkeypatch.setattr(roster_backfill, "FIGHTERS_PATH", fighters_path)
    monkeypatch.setattr(
        roster_backfill,
        "_append_missing_profiles_with_summary",
        lambda _roster_df: {
            "scraped_profiles_added": 0,
            "scraped_profiles_updated": 0,
            "scraped_profiles_needed_refresh": 0,
            "scraped_profile_scrape_failures": 0,
            "failed_profile_urls": [],
        },
    )

    def fake_extract(fighter_url, *, session):
        if fighter_url.endswith("/blocked"):
            raise ValueError("blocked by loading page")
        if fighter_url.endswith("/new"):
            return []
        return ["http://ufcstats.com/fight-details/existing"]

    monkeypatch.setattr(roster_backfill, "_extract_completed_fight_urls", fake_extract)

    roster_df = pd.DataFrame(
        [
            {
                "official_name": "Blocked Fighter",
                "ufcstats_url": "http://ufcstats.com/fighter-details/blocked",
                "coverage_eligible": True,
            },
            {
                "official_name": "New Fighter",
                "ufcstats_url": "http://ufcstats.com/fighter-details/new",
                "coverage_eligible": True,
            },
            {
                "official_name": "Veteran",
                "ufcstats_url": "http://ufcstats.com/fighter-details/veteran",
                "coverage_eligible": True,
            },
        ]
    )

    summary = roster_backfill.run_backfill(roster_df=roster_df)

    assert summary["fighters_checked"] == 2
    assert summary["fighters_with_completed_ufcstats_bouts"] == 1
    assert summary["fighters_without_completed_ufcstats_bouts"] == 1
    assert summary["fighter_fight_list_scrape_failures"] == 1
    assert summary["fighter_fight_list_failure_details"][0]["fighter"] == "Blocked Fighter"
    assert summary["fighter_fight_list_failure_details_truncated"] == 0
    assert summary["missing_fight_urls_found"] == 0
    assert summary["fighters_with_nonzero_ufcstats_rate_stats"] == 1


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


def test_search_sherdog_uses_saint_abbreviation_variants(monkeypatch):
    calls = []

    def fake_get_soup(url):
        calls.append(url)
        if "SearchTxt=Benoit%20St%20Denis" in url:
            return BeautifulSoup(
                """
                <html><body>
                  <table class="fightfinder_result">
                    <tr><td><a href="/fighter/Benoit-St-Denis-317103">Benoit St. Denis</a></td></tr>
                  </table>
                </body></html>
                """,
                "lxml",
            )
        return BeautifulSoup("<html><body></body></html>", "lxml")

    monkeypatch.setattr(fallback_scrapers, "_get_soup", fake_get_soup)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.search_sherdog("Benoit Saint-Denis")

    assert result == "https://www.sherdog.com/fighter/Benoit-St-Denis-317103"
    assert calls[:2] == [
        "https://www.sherdog.com/stats/fightfinder?SearchTxt=Benoit%20Saint-Denis",
        "https://www.sherdog.com/stats/fightfinder?SearchTxt=Benoit%20St%20Denis",
    ]


def test_build_tapology_scraper_sets_modern_user_agent(monkeypatch):
    captured_browser = {}

    class _FakeScraper:
        def __init__(self):
            self.headers = {}
            self.proxies = {}

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


def test_get_tapology_soup_retries_cloudflare_403_with_alternate_browser_profile(monkeypatch):
    captured_browsers = []
    calls = []

    class _FakeResponse:
        def __init__(self, status_code, text, headers=None):
            self.status_code = status_code
            self.text = text
            self.headers = headers or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} error")

    class _FakeScraper:
        def __init__(self, response):
            self.headers = {}
            self.proxies = {}
            self._response = response

        def get(self, url, **kwargs):
            calls.append(url)
            return self._response

    def fake_create_scraper(*, browser):
        captured_browsers.append(browser)
        if len(captured_browsers) == 1:
            response = _FakeResponse(
                403,
                "<html><title>Just a moment...</title></html>",
                {"server": "cloudflare"},
            )
        else:
            response = _FakeResponse(200, "<html><body>Tapology OK</body></html>")
        return _FakeScraper(response)

    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_PROXY_URL", "")
    monkeypatch.setattr(
        fallback_scrapers,
        "cloudscraper",
        types.SimpleNamespace(create_scraper=fake_create_scraper),
    )
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    fallback_scrapers.clear_fallback_cache()

    soup = fallback_scrapers._get_tapology_soup(
        "https://www.tapology.com/fightcenter/fighters/example",
        max_retries=1,
    )

    assert soup.get_text(strip=True) == "Tapology OK"
    assert captured_browsers == [
        {"browser": "chrome", "platform": "windows", "mobile": False},
        {"browser": "chrome", "platform": "linux", "mobile": False},
    ]
    assert calls == [
        "https://www.tapology.com/fightcenter/fighters/example",
        "https://www.tapology.com/fightcenter/fighters/example",
    ]
    assert fallback_scrapers._tapology_blocked is None


def test_get_tapology_soup_rejects_200_cloudflare_challenge(monkeypatch):
    calls = []

    class _FakeResponse:
        text = (
            "<html><head><title>Just a moment...</title></head>"
            "<body>Performing security verification to check you are not a bot</body></html>"
        )
        status_code = 200
        headers = {"server": "cloudflare"}

    class _FakeScraper:
        def __init__(self):
            self.headers = {}
            self.proxies = {}

        def get(self, url, **kwargs):
            calls.append(url)
            return _FakeResponse()

    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_PROXY_URL", "")
    monkeypatch.setattr(
        fallback_scrapers,
        "cloudscraper",
        types.SimpleNamespace(create_scraper=lambda **_kwargs: _FakeScraper()),
    )
    fallback_scrapers.clear_fallback_cache()

    with pytest.raises(fallback_scrapers.TapologyRequestError) as excinfo:
        fallback_scrapers._get_tapology_soup(
            "https://www.tapology.com/fightcenter/fighters/example",
            max_retries=1,
        )

    assert excinfo.value.detail == "Cloudflare challenge"
    assert fallback_scrapers._tapology_blocked is True
    assert calls == [
        "https://www.tapology.com/fightcenter/fighters/example",
        "https://www.tapology.com/fightcenter/fighters/example",
    ]


def test_search_tapology_does_not_probe_fightcenter_before_search(monkeypatch):
    class _FakeResponse:
        text = """
        <html><body>
          <a href="/fightcenter/fighters/steve-nelmark-the-sandman">Steve "The Sandman" Nelmark</a>
        </body></html>
        """
        status_code = 200

    class _FakeScraper:
        def __init__(self):
            self.headers = {}
            self.proxies = {}

        def get(self, url, **kwargs):
            calls.append((url, kwargs.get("params")))
            return _FakeResponse()

    calls = []
    monkeypatch.setattr(
        fallback_scrapers,
        "cloudscraper",
        types.SimpleNamespace(create_scraper=lambda **_kwargs: _FakeScraper()),
    )
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.search_tapology("Steve Nelmark")

    assert result == "https://www.tapology.com/fightcenter/fighters/steve-nelmark-the-sandman"
    assert calls == [(fallback_scrapers.TAPOLOGY_SEARCH_URL, {"term": "Steve Nelmark"})]


def test_get_tapology_soup_uses_configured_proxy(monkeypatch):
    proxy_url = "http://user:pass@proxy.example:8080"
    captured: dict[str, object] = {}

    class _FakeResponse:
        text = "<html><body>ok</body></html>"
        status_code = 200

    class _FakeScraper:
        def __init__(self):
            self.headers = {}
            self.proxies = {}
            captured["session_proxies"] = self.proxies

        def get(self, url, **kwargs):
            captured["url"] = url
            captured["request_proxies"] = kwargs.get("proxies")
            return _FakeResponse()

    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_PROXY_URL", proxy_url)
    monkeypatch.setattr(
        fallback_scrapers,
        "cloudscraper",
        types.SimpleNamespace(create_scraper=lambda **_kwargs: _FakeScraper()),
    )
    fallback_scrapers.clear_fallback_cache()

    soup = fallback_scrapers._get_tapology_soup(fallback_scrapers.TAPOLOGY_SEARCH_URL)

    expected = {"http": proxy_url, "https": proxy_url}
    assert soup.get_text(strip=True) == "ok"
    assert captured["session_proxies"] == expected
    assert captured["request_proxies"] == expected


def test_get_tapology_soup_fails_fast_on_cloudflare_403_without_proxy(monkeypatch, caplog):
    class _FakeResponse:
        text = "<html><body>Forbidden</body></html>"
        status_code = 403
        headers = {"server": "cloudflare"}

    class _FakeScraper:
        def __init__(self):
            self.headers = {}
            self.proxies = {}

        def get(self, url, **kwargs):
            calls.append(url)
            return _FakeResponse()

    calls = []
    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_PROXY_URL", "")
    monkeypatch.setattr(
        fallback_scrapers,
        "cloudscraper",
        types.SimpleNamespace(create_scraper=lambda **_kwargs: _FakeScraper()),
    )
    fallback_scrapers.clear_fallback_cache()
    caplog.set_level(logging.ERROR)

    with pytest.raises(fallback_scrapers.TapologyRequestError) as excinfo:
        fallback_scrapers._get_tapology_soup(
            "https://www.tapology.com/fightcenter/fighters/example"
        )

    assert excinfo.value.status_code == 403
    assert excinfo.value.detail == "Cloudflare challenge"
    assert fallback_scrapers._tapology_blocked is True

    with pytest.raises(fallback_scrapers.TapologyRequestError) as second_excinfo:
        fallback_scrapers._get_tapology_soup(
            "https://www.tapology.com/fightcenter/fighters/another-example"
        )

    assert second_excinfo.value.status_code == 403
    assert second_excinfo.value.detail == "Tapology blocked from this environment"
    assert calls == [
        "https://www.tapology.com/fightcenter/fighters/example",
        "https://www.tapology.com/fightcenter/fighters/example",
    ]
    assert any(
        record.levelname == "ERROR"
        and "External data source unavailable: Tapology - profile pages blocked by Cloudflare" in record.getMessage()
        for record in caplog.records
    )


def test_get_tapology_soup_uses_browser_fallback_after_cloudflare(monkeypatch, caplog):
    class _FakeResponse:
        text = "<html><head><title>Just a moment...</title></head><body>Cloudflare</body></html>"
        status_code = 403
        headers = {"server": "cloudflare"}

    class _FakeScraper:
        def __init__(self):
            self.headers = {}
            self.proxies = {}

        def get(self, url, **kwargs):
            calls.append((url, kwargs.get("params")))
            return _FakeResponse()

    browser_calls = []
    calls = []
    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_PROXY_URL", "")
    monkeypatch.setattr(
        fallback_scrapers,
        "cloudscraper",
        types.SimpleNamespace(create_scraper=lambda **_kwargs: _FakeScraper()),
    )
    monkeypatch.setattr(fallback_scrapers, "_tapology_browser_fallback_available", lambda: True)

    def fake_browser_soup(url, params=None):
        browser_calls.append((url, params))
        return BeautifulSoup("<html><body>Tapology Browser OK</body></html>", "lxml")

    monkeypatch.setattr(fallback_scrapers, "_get_tapology_soup_with_browser", fake_browser_soup)
    fallback_scrapers.clear_fallback_cache()
    caplog.set_level(logging.WARNING, logger="src.data.fallback_scrapers")

    soup = fallback_scrapers._get_tapology_soup(
        fallback_scrapers.TAPOLOGY_SEARCH_URL,
        params={"term": "Benoit Saint-Denis"},
        max_retries=1,
    )

    assert soup.get_text(strip=True) == "Tapology Browser OK"
    assert browser_calls == [
        (fallback_scrapers.TAPOLOGY_SEARCH_URL, {"term": "Benoit Saint-Denis"})
    ]
    assert fallback_scrapers._tapology_blocked is None
    assert calls == [
        (fallback_scrapers.TAPOLOGY_SEARCH_URL, {"term": "Benoit Saint-Denis"}),
        (fallback_scrapers.TAPOLOGY_SEARCH_URL, {"term": "Benoit Saint-Denis"}),
    ]
    assert not any(
        "retrying through hosted browser fallback" in record.getMessage()
        for record in caplog.records
    )


def test_get_tapology_soup_uses_browser_fallback_after_cloudflare_with_proxy(monkeypatch):
    class _FakeResponse:
        text = "<html><head><title>Just a moment...</title></head><body>Cloudflare</body></html>"
        status_code = 403
        headers = {"server": "cloudflare"}

    class _FakeScraper:
        def __init__(self):
            self.headers = {}
            self.proxies = {}

        def get(self, url, **kwargs):
            calls.append((url, kwargs.get("params"), kwargs.get("proxies")))
            return _FakeResponse()

    browser_calls = []
    calls = []
    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_PROXY_URL", "http://proxy.example:3128")
    monkeypatch.setattr(
        fallback_scrapers,
        "cloudscraper",
        types.SimpleNamespace(create_scraper=lambda **_kwargs: _FakeScraper()),
    )
    monkeypatch.setattr(fallback_scrapers, "_tapology_browser_fallback_available", lambda: True)

    def fake_browser_soup(url, params=None):
        browser_calls.append((url, params))
        return BeautifulSoup("<html><body>Tapology Browser OK</body></html>", "lxml")

    monkeypatch.setattr(fallback_scrapers, "_get_tapology_soup_with_browser", fake_browser_soup)
    fallback_scrapers.clear_fallback_cache()

    soup = fallback_scrapers._get_tapology_soup(
        fallback_scrapers.TAPOLOGY_SEARCH_URL,
        params={"term": "Benoit Saint-Denis"},
        max_retries=1,
    )

    assert soup.get_text(strip=True) == "Tapology Browser OK"
    assert browser_calls == [
        (fallback_scrapers.TAPOLOGY_SEARCH_URL, {"term": "Benoit Saint-Denis"})
    ]
    assert calls == [
        (
            fallback_scrapers.TAPOLOGY_SEARCH_URL,
            {"term": "Benoit Saint-Denis"},
            {"http": "http://proxy.example:3128", "https": "http://proxy.example:3128"},
        ),
        (
            fallback_scrapers.TAPOLOGY_SEARCH_URL,
            {"term": "Benoit Saint-Denis"},
            {"http": "http://proxy.example:3128", "https": "http://proxy.example:3128"},
        ),
    ]


def test_tapology_browser_fallback_recovers_partial_page_after_navigation_timeout(monkeypatch):
    from selenium.common.exceptions import TimeoutException
    from selenium import webdriver

    class _FakeDriver:
        def __init__(self):
            self.stopped = False
            self.quit_called = False

        def set_page_load_timeout(self, seconds):
            page_timeouts.append(seconds)

        def execute_cdp_cmd(self, command, params):
            cdp_calls.append((command, params))

        def get(self, url):
            visited_urls.append(url)
            raise TimeoutException("renderer timed out")

        def execute_script(self, script):
            script_calls.append(script)
            self.stopped = True

        @property
        def page_source(self):
            return "<html><body>Pro MMA Record: 10-1-0</body></html>"

        def quit(self):
            self.quit_called = True
            quit_calls.append(True)

    captured_options = []
    drivers = []
    page_timeouts = []
    cdp_calls = []
    visited_urls = []
    script_calls = []
    quit_calls = []

    def fake_chrome(service, options):
        captured_options.append(options)
        driver = _FakeDriver()
        drivers.append(driver)
        return driver

    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_BROWSER_FALLBACK_ENABLED", True)
    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_BROWSER_PAGE_TIMEOUT_SECONDS", 2)
    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_BROWSER_REQUEST_DELAY_SECONDS", 0)
    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_PROXY_URL", "http://proxy.example:3128")
    monkeypatch.setattr(
        fallback_scrapers,
        "_tapology_browser_dependency_paths",
        lambda: ("chromium", "", "xvfb"),
    )
    monkeypatch.setattr(fallback_scrapers, "_tapology_virtual_display", lambda: nullcontext())
    monkeypatch.setattr(webdriver, "Chrome", fake_chrome)
    fallback_scrapers.clear_fallback_cache()

    first_html = fallback_scrapers._get_tapology_html_with_browser(
        "https://www.tapology.com/fightcenter/fighters/example-one"
    )
    second_html = fallback_scrapers._get_tapology_html_with_browser(
        "https://www.tapology.com/fightcenter/fighters/example-two"
    )

    assert "Pro MMA Record" in first_html
    assert "Pro MMA Record" in second_html
    assert visited_urls == [
        "https://www.tapology.com/fightcenter/fighters/example-one",
        "https://www.tapology.com/fightcenter/fighters/example-two",
    ]
    assert script_calls == ["window.stop();", "window.stop();"]
    assert page_timeouts == [2, 2]
    assert len(quit_calls) == 2
    assert all(driver.quit_called for driver in drivers)
    assert all(options.page_load_strategy == "eager" for options in captured_options)
    assert all(
        "--proxy-server=http://proxy.example:3128" in options.arguments
        for options in captured_options
    )
    profile_args = [
        arg
        for options in captured_options
        for arg in options.arguments
        if arg.startswith("--user-data-dir=")
    ]
    assert len(profile_args) == 2
    assert profile_args[0] == profile_args[1]
    assert any(command == "Page.addScriptToEvaluateOnNewDocument" for command, _params in cdp_calls)


def test_tapology_browser_ready_requires_fighter_profile_content():
    url = "https://www.tapology.com/fightcenter/fighters/example"

    assert not fallback_scrapers._tapology_browser_page_ready(
        url,
        "<html><title>Tapology</title><body>Tapology loading shell</body></html>",
    )
    assert fallback_scrapers._tapology_browser_page_ready(
        url,
        "<html><body>Pro MMA Record: 10-1-0</body></html>",
    )
    assert fallback_scrapers._tapology_browser_page_ready(
        url,
        '<html><body><div data-bout-id="123"></div></body></html>',
    )


def test_tapology_browser_environment_uses_writable_temp_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", "/nonexistent")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

    with fallback_scrapers._tapology_browser_environment(str(tmp_path)):
        assert os.environ["HOME"] == str(tmp_path / "home")
        assert os.environ["XDG_CONFIG_HOME"] == str(tmp_path / "config")
        assert os.environ["XDG_CACHE_HOME"] == str(tmp_path / "cache")
        assert os.environ["XDG_RUNTIME_DIR"] == str(tmp_path / "runtime")
        assert (tmp_path / "home").is_dir()
        assert (tmp_path / "runtime").is_dir()
        if os.name != "nt":
            assert oct((tmp_path / "runtime").stat().st_mode & 0o777) == "0o700"

    assert os.environ["HOME"] == "/nonexistent"
    assert "XDG_CONFIG_HOME" not in os.environ
    assert "XDG_CACHE_HOME" not in os.environ
    assert "XDG_RUNTIME_DIR" not in os.environ


def test_get_tapology_soup_uses_browser_fallback_when_requests_path_cached_blocked(monkeypatch):
    browser_calls = []
    monkeypatch.setattr(fallback_scrapers, "_tapology_browser_fallback_available", lambda: True)

    def fake_browser_soup(url, params=None):
        browser_calls.append((url, params))
        return BeautifulSoup("<html><body>Recovered From Browser</body></html>", "lxml")

    monkeypatch.setattr(fallback_scrapers, "_get_tapology_soup_with_browser", fake_browser_soup)
    fallback_scrapers.clear_fallback_cache()
    fallback_scrapers._tapology_blocked = True

    soup = fallback_scrapers._get_tapology_soup("https://www.tapology.com/fightcenter/fighters/example")

    assert soup.get_text(strip=True) == "Recovered From Browser"
    assert browser_calls == [("https://www.tapology.com/fightcenter/fighters/example", None)]


def test_search_tapology_uses_browser_fallback_when_requests_path_cached_blocked(monkeypatch):
    browser_calls = []
    html = """
    <html><body>
      <a href="/fightcenter/fighters/steve-nelmark-the-sandman">Steve "The Sandman" Nelmark</a>
    </body></html>
    """
    monkeypatch.setattr(fallback_scrapers, "_tapology_browser_fallback_available", lambda: True)

    def fake_browser_soup(url, params=None):
        browser_calls.append((url, params))
        return BeautifulSoup(html, "lxml")

    monkeypatch.setattr(fallback_scrapers, "_get_tapology_soup_with_browser", fake_browser_soup)
    fallback_scrapers.clear_fallback_cache()
    fallback_scrapers._tapology_blocked = True

    result = fallback_scrapers.search_tapology_candidates("Steve Nelmark", limit=1)

    assert result == ["https://www.tapology.com/fightcenter/fighters/steve-nelmark-the-sandman"]
    assert browser_calls == [(fallback_scrapers.TAPOLOGY_SEARCH_URL, {"term": "Steve Nelmark"})]


def test_tapology_browser_cloudflare_block_stops_later_browser_retries(monkeypatch, caplog):
    class _FakeResponse:
        text = "<html><head><title>Just a moment...</title></head><body>Cloudflare</body></html>"
        status_code = 403
        headers = {"server": "cloudflare"}

    class _FakeScraper:
        def __init__(self):
            self.headers = {}
            self.proxies = {}

        def get(self, url, **kwargs):
            request_calls.append((url, kwargs.get("params")))
            return _FakeResponse()

    browser_calls = []
    request_calls = []
    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_PROXY_URL", "")
    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_READER_FALLBACK_ENABLED", False)
    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_BROWSER_FALLBACK_ENABLED", True)
    monkeypatch.setattr(
        fallback_scrapers,
        "_tapology_browser_dependency_paths",
        lambda: ("chromium", "", "xvfb"),
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "cloudscraper",
        types.SimpleNamespace(create_scraper=lambda **_kwargs: _FakeScraper()),
    )

    def fake_browser_soup(url, params=None):
        browser_calls.append((url, params))
        raise fallback_scrapers.TapologyRequestError(
            fallback_scrapers._tapology_fetch_url(url, params),
            status_code=403,
            detail="Cloudflare challenge from browser fallback",
        )

    monkeypatch.setattr(fallback_scrapers, "_get_tapology_soup_with_browser", fake_browser_soup)
    fallback_scrapers.clear_fallback_cache()
    caplog.set_level(logging.WARNING, logger="src.data.fallback_scrapers")

    with pytest.raises(fallback_scrapers.TapologyRequestError) as excinfo:
        fallback_scrapers._get_tapology_soup(
            fallback_scrapers.TAPOLOGY_SEARCH_URL,
            params={"term": "Abdulrakhman Yakhyaev"},
            max_retries=1,
        )

    assert excinfo.value.detail == "Cloudflare challenge"
    assert fallback_scrapers._tapology_blocked is True
    assert fallback_scrapers._tapology_browser_cloudflare_blocked is True
    assert browser_calls == [
        (fallback_scrapers.TAPOLOGY_SEARCH_URL, {"term": "Abdulrakhman Yakhyaev"})
    ]

    result = fallback_scrapers.search_tapology_candidates("Nursultan Ruziboev", limit=1)

    assert result == []
    assert browser_calls == [
        (fallback_scrapers.TAPOLOGY_SEARCH_URL, {"term": "Abdulrakhman Yakhyaev"})
    ]
    assert request_calls == [
        (fallback_scrapers.TAPOLOGY_SEARCH_URL, {"term": "Abdulrakhman Yakhyaev"}),
        (fallback_scrapers.TAPOLOGY_SEARCH_URL, {"term": "Abdulrakhman Yakhyaev"}),
    ]
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)
    assert any(
        "External data source unavailable: Tapology - browser fallback blocked by Cloudflare"
        in record.getMessage()
        for record in caplog.records
    )


def test_clear_fallback_cache_preserves_tapology_environment_blocks(monkeypatch):
    monkeypatch.setattr(
        fallback_scrapers,
        "_tapology_browser_dependency_paths",
        lambda: ("chromium", "", "xvfb"),
    )
    fallback_scrapers.clear_fallback_cache()
    fallback_scrapers._tapology_blocked = True
    fallback_scrapers._tapology_search_blocked = True
    fallback_scrapers._tapology_browser_cloudflare_blocked = True
    fallback_scrapers._tapology_browser_unavailable = True
    fallback_scrapers._site_search_disabled = True
    fallback_scrapers._tapology_url_cache["blocked"] = "https://www.tapology.com/fightcenter/fighters/blocked"
    fallback_scrapers._tapology_browser_html_cache["https://www.tapology.com/search?term=blocked"] = "html"
    fallback_scrapers._external_source_alert_keys.add(("Tapology", "browser fallback blocked by Cloudflare"))

    fallback_scrapers.clear_fallback_cache(preserve_environment_blocks=True)

    assert fallback_scrapers._tapology_url_cache == {}
    assert fallback_scrapers._tapology_browser_html_cache == {}
    assert fallback_scrapers._tapology_blocked is True
    assert fallback_scrapers._tapology_search_blocked is True
    assert fallback_scrapers._tapology_browser_cloudflare_blocked is True
    assert fallback_scrapers._tapology_browser_unavailable is True
    assert fallback_scrapers._site_search_disabled is True
    assert ("Tapology", "browser fallback blocked by Cloudflare") in fallback_scrapers._external_source_alert_keys

    fallback_scrapers.clear_fallback_cache()
    assert fallback_scrapers._tapology_blocked is None
    assert fallback_scrapers._tapology_search_blocked is False
    assert fallback_scrapers._tapology_browser_cloudflare_blocked is False
    assert fallback_scrapers._tapology_browser_unavailable is False
    assert fallback_scrapers._site_search_disabled is False
    assert fallback_scrapers._external_source_alert_keys == set()


def test_search_tapology_candidates_uses_duckduckgo_when_tapology_origin_blocked(monkeypatch):
    class _FakeResponse:
        status_code = 200
        text = """
        <html><body>
          <a class="result__a"
             href="/l/?kh=-1&uddg=https%3A%2F%2Fwww.tapology.com%2Ffightcenter%2Ffighters%2F49423-abdul-azeem-badakhshi">
            Abdul Azim Badakhshi ("The Afghan Lion") | MMA Fighter Page | Tapology
          </a>
        </body></html>
        """

        def raise_for_status(self):
            return None

    native_calls = []

    def fake_get_tapology_soup(*args, **kwargs):
        native_calls.append((args, kwargs))
        raise AssertionError("Tapology-origin search should be skipped after cached environment block")

    duck_calls = []

    def fake_get(url, **kwargs):
        duck_calls.append((url, kwargs.get("params")))
        return _FakeResponse()

    monkeypatch.setattr(fallback_scrapers, "BRAVE_SEARCH_API_KEY", "")
    monkeypatch.setattr(fallback_scrapers, "BRAVE_SEARCH_HTML_FALLBACK_ENABLED", False)
    monkeypatch.setattr(fallback_scrapers, "DUCKDUCKGO_SEARCH_HTML_FALLBACK_ENABLED", True)
    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_READER_FALLBACK_ENABLED", True)
    monkeypatch.setattr(fallback_scrapers, "_get_tapology_soup", fake_get_tapology_soup)
    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()
    fallback_scrapers._tapology_blocked = True
    fallback_scrapers._tapology_browser_cloudflare_blocked = True

    result = fallback_scrapers.search_tapology_candidates("Abdul Azeem Badakhshi", limit=1)

    assert result == ["https://www.tapology.com/fightcenter/fighters/49423-abdul-azeem-badakhshi"]
    assert native_calls == []
    assert duck_calls == [
        (
            fallback_scrapers.DUCKDUCKGO_SEARCH_HTML_URL,
            {"q": "Abdul Azeem Badakhshi tapology.com/fightcenter/fighters"},
        )
    ]


def test_scrape_tapology_profile_uses_reader_after_cloudflare_block(monkeypatch):
    markdown = """
    Title: Abdul Azim Badakhshi ("The Afghan Lion") | MMA Fighter Page | Tapology
    URL Source: https://www.tapology.com/fightcenter/fighters/49423-abdul-azeem-badakhshi
    Markdown Content:
    #### Fighter Details
    Age ??
    Height 5'9" 174cm
    Reach ??
    Weight 144.2 Featherweight
    **Name:**Abdul Azim Badakhshi
    **Nickname:**The Afghan Lion
    **Pro MMA Record:**14-5-0 (Win-Loss-Draw)
    **Current MMA Streak:**2 Losses
    **Age & Date of Birth:**N/A
    **Height:**5'9" (174cm)**| Reach:** N/A
    **Weight Class:**Featherweight**| Last Weigh-In:** 144.2 lbs
    **Affiliation:**MMA Matrix Gym
    """

    class _FakeResponse:
        status_code = 200
        text = markdown

        def raise_for_status(self):
            return None

    reader_calls = []

    def fake_get(url, **kwargs):
        reader_calls.append((url, kwargs))
        return _FakeResponse()

    monkeypatch.setattr(
        fallback_scrapers,
        "_get_tapology_soup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            fallback_scrapers.TapologyRequestError(
                "https://www.tapology.com/fightcenter/fighters/49423-abdul-azeem-badakhshi",
                status_code=403,
                detail="Tapology blocked from this environment",
            )
        ),
    )
    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_READER_FALLBACK_ENABLED", True)
    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    fallback_scrapers.clear_fallback_cache()

    profile = fallback_scrapers.scrape_tapology_profile(
        "https://www.tapology.com/fightcenter/fighters/49423-abdul-azeem-badakhshi"
    )

    assert profile["name"] == "Abdul Azim Badakhshi"
    assert profile["record"] == "14-5-0"
    assert profile["wins"] == 14
    assert profile["losses"] == 5
    assert profile["draws"] == 0
    assert profile["height_raw"] == "5'9\" (174cm)"
    assert profile["height"] == pytest.approx(174.0)
    assert profile["reach_raw"] == ""
    assert np.isnan(profile["reach"])
    assert profile["weight_raw"] == "144.2 lbs"
    assert profile["weight"] == pytest.approx(144.0)
    assert profile["dob"] == ""
    assert reader_calls == [
        (
            "https://r.jina.ai/https://www.tapology.com/fightcenter/fighters/49423-abdul-azeem-badakhshi",
            {"timeout": fallback_scrapers.TAPOLOGY_READER_TIMEOUT_SECONDS},
        )
    ]


def test_scrape_tapology_profile_retries_transient_non_profile_reader_response(monkeypatch):
    bad_markdown = """
    Title: https://s.amazon-adsystem.com/dcm
    URL Source: https://www.tapology.com/fightcenter/fighters/243673-abdulrakhman-yakhyaev
    Markdown Content:
    # https://s.amazon-adsystem.com/dcm
    A 1x1 image, likely be a tracker probe
    """
    good_markdown = """
    Title: Abdul Rakhman Yakhyaev ("The Hunter") | MMA Fighter Page | Tapology
    URL Source: https://www.tapology.com/fightcenter/fighters/243673-abdulrakhman-yakhyaev
    Markdown Content:
    #### Fighter Details
    **Name:**Abdul Rakhman Yakhyaev
    **Nickname:**The Hunter
    **Pro MMA Record:**7-0-0 (Win-Loss-Draw)
    **Current MMA Streak:**7 Wins
    **Age & Date of Birth:**N/A
    **Height:**6'2" (188cm)**| Reach:** N/A
    **Weight Class:**Heavyweight**| Last Weigh-In:** 260.0 lbs
    **Affiliation:**Yakhyaev Team
    """

    class _FakeResponse:
        status_code = 200

        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    responses = [_FakeResponse(bad_markdown), _FakeResponse(good_markdown)]
    reader_calls = []

    def fake_get(url, **kwargs):
        reader_calls.append(url)
        return responses.pop(0)

    monkeypatch.setattr(
        fallback_scrapers,
        "_get_tapology_soup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            fallback_scrapers.TapologyRequestError(
                "https://www.tapology.com/fightcenter/fighters/243673-abdulrakhman-yakhyaev",
                status_code=403,
                detail="Tapology blocked from this environment",
            )
        ),
    )
    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_READER_FALLBACK_ENABLED", True)
    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    monkeypatch.setattr(fallback_scrapers.time, "sleep", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()

    profile = fallback_scrapers.scrape_tapology_profile(
        "https://www.tapology.com/fightcenter/fighters/243673-abdulrakhman-yakhyaev"
    )

    assert profile["name"] == "Abdul Rakhman Yakhyaev"
    assert profile["record"] == "7-0-0"
    assert profile["height"] == pytest.approx(188.0)
    assert profile["weight"] == pytest.approx(260.0)
    assert len(reader_calls) == 2
    assert fallback_scrapers._tapology_reader_unavailable is False


def test_fallback_lookup_keeps_tapology_reader_profile_when_fight_history_fails(monkeypatch):
    monkeypatch.setattr(fallback_scrapers, "search_sherdog", lambda _name: None)
    monkeypatch.setattr(
        fallback_scrapers,
        "search_tapology",
        lambda _name: "https://www.tapology.com/fightcenter/fighters/49423-abdul-azeem-badakhshi",
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "scrape_tapology_profile",
        lambda _url: {
            "name": "Abdul Azim Badakhshi",
            "fighter_url": _url,
            "record": "14-5-0",
            "wins": 14,
            "losses": 5,
            "draws": 0,
            "height_raw": "5'9\" (174cm)",
            "height": 174.0,
            "reach_raw": "",
            "reach": np.nan,
            "weight_raw": "144.2 lbs",
            "weight": 144.0,
            "stance": "",
            "dob": "",
            "age": np.nan,
        },
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "scrape_tapology_fights",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            fallback_scrapers.TapologyRequestError(
                "https://www.tapology.com/fightcenter/fighters/49423-abdul-azeem-badakhshi",
                status_code=403,
            )
        ),
    )
    monkeypatch.setattr(fallback_scrapers, "search_espn", lambda _name: None)
    monkeypatch.setattr(fallback_scrapers, "search_martialbot", lambda _name: None)
    monkeypatch.setattr(fallback_scrapers, "search_fightdx", lambda _name: None)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.fallback_lookup("Abdul Azeem Badakhshi")

    assert result is not None
    profile, fights = result
    assert fights == []
    assert profile["record"] == "14-5-0"
    assert profile["height"] == pytest.approx(174.0)
    assert profile["weight"] == pytest.approx(144.0)


def test_fallback_lookup_recovers_tapology_profile_via_search_index_and_reader(monkeypatch):
    duckduckgo_html = """
    <html><body>
      <a class="result__a"
         href="/l/?kh=-1&uddg=https%3A%2F%2Fwww.tapology.com%2Ffightcenter%2Ffighters%2F49423-abdul-azeem-badakhshi">
        Abdul Azim Badakhshi ("The Afghan Lion") | MMA Fighter Page | Tapology
      </a>
    </body></html>
    """
    reader_markdown = """
    Title: Abdul Azim Badakhshi ("The Afghan Lion") | MMA Fighter Page | Tapology
    URL Source: https://www.tapology.com/fightcenter/fighters/49423-abdul-azeem-badakhshi
    Markdown Content:
    #### Fighter Details
    **Name:**Abdul Azim Badakhshi
    **Nickname:**The Afghan Lion
    **Pro MMA Record:**14-5-0 (Win-Loss-Draw)
    **Current MMA Streak:**2 Losses
    **Age & Date of Birth:**N/A
    **Height:**5'9" (174cm)**| Reach:** N/A
    **Weight Class:**Featherweight**| Last Weigh-In:** 144.2 lbs
    **Affiliation:**MMA Matrix Gym
    """

    class _FakeResponse:
        def __init__(self, text):
            self.status_code = 200
            self.text = text

        def raise_for_status(self):
            return None

    request_urls = []

    def fake_get(url, **kwargs):
        request_urls.append((url, kwargs.get("params")))
        if url == fallback_scrapers.DUCKDUCKGO_SEARCH_HTML_URL:
            return _FakeResponse(duckduckgo_html)
        if str(url).startswith(fallback_scrapers.TAPOLOGY_READER_BASE_URL):
            return _FakeResponse(reader_markdown)
        raise AssertionError(f"unexpected request: {url}")

    native_calls = []

    def fake_get_tapology_soup(url, **kwargs):
        native_calls.append((url, kwargs))
        raise fallback_scrapers.TapologyRequestError(
            url,
            status_code=403,
            detail="Tapology blocked from this environment",
        )

    monkeypatch.setattr(fallback_scrapers, "BRAVE_SEARCH_API_KEY", "")
    monkeypatch.setattr(fallback_scrapers, "BRAVE_SEARCH_HTML_FALLBACK_ENABLED", False)
    monkeypatch.setattr(fallback_scrapers, "DUCKDUCKGO_SEARCH_HTML_FALLBACK_ENABLED", True)
    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_READER_FALLBACK_ENABLED", True)
    monkeypatch.setattr(fallback_scrapers, "_get_tapology_soup", fake_get_tapology_soup)
    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    monkeypatch.setattr(fallback_scrapers, "search_sherdog", lambda _name: None)
    monkeypatch.setattr(fallback_scrapers, "search_espn", lambda _name: None)
    monkeypatch.setattr(fallback_scrapers, "search_martialbot", lambda _name: None)
    monkeypatch.setattr(fallback_scrapers, "search_fightdx", lambda _name: None)
    fallback_scrapers.clear_fallback_cache()
    fallback_scrapers._tapology_blocked = True
    fallback_scrapers._tapology_browser_cloudflare_blocked = True

    result = fallback_scrapers.fallback_lookup("Abdul Azeem Badakhshi")

    assert result is not None
    profile, fights = result
    assert fights == []
    assert profile["name"] == "Abdul Azim Badakhshi"
    assert profile["record"] == "14-5-0"
    assert profile["height"] == pytest.approx(174.0)
    assert profile["weight"] == pytest.approx(144.0)
    assert request_urls == [
        (
            fallback_scrapers.DUCKDUCKGO_SEARCH_HTML_URL,
            {"q": "Abdul Azeem Badakhshi tapology.com/fightcenter/fighters"},
        ),
        (
            "https://r.jina.ai/https://www.tapology.com/fightcenter/fighters/49423-abdul-azeem-badakhshi",
            None,
        ),
    ]
    assert native_calls == [
        (
            "https://www.tapology.com/fightcenter/fighters/49423-abdul-azeem-badakhshi",
            {},
        ),
        (
            "https://www.tapology.com/fightcenter/fighters/49423-abdul-azeem-badakhshi",
            {},
        ),
    ]


def test_profile_supplement_refresh_preserves_tapology_block_across_cache_clear(tmp_path, monkeypatch):
    scraped_path = tmp_path / "ufc_fighters_scraped.csv"
    output_path = tmp_path / "ufc_fighters_profile_supplement.csv"
    pd.DataFrame(
        [
            {
                "name": "Blocked Tapology Fighter",
                "height": "",
                "reach": "",
                "weight": "",
                "stance": "Orthodox",
                "dob": "",
            }
        ]
    ).to_csv(scraped_path, index=False)
    tapology_calls = []

    def fail_tapology_fetch(*args, **kwargs):
        tapology_calls.append((args, kwargs))
        raise AssertionError("Tapology should stay skipped after a cached environment block")

    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_READER_FALLBACK_ENABLED", False)
    monkeypatch.setattr(fallback_scrapers, "_get_tapology_soup", fail_tapology_fetch)
    monkeypatch.setattr(external_profiles, "scrape_tapology_profile", fail_tapology_fetch)
    fallback_scrapers.clear_fallback_cache()
    fallback_scrapers._tapology_blocked = True
    fallback_scrapers._tapology_browser_cloudflare_blocked = True

    summary = external_profiles.run_profile_supplement_refresh(
        scraped_fighters_path=scraped_path,
        output_path=output_path,
        sources=["tapology"],
    )

    assert summary["attempted_rows"] == 1
    assert summary["recovered_rows"] == 0
    assert tapology_calls == []
    assert fallback_scrapers._tapology_blocked is True
    assert fallback_scrapers._tapology_browser_cloudflare_blocked is True


def test_external_source_request_failure_logs_error_once(monkeypatch, caplog):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(args[0])
        raise requests.Timeout("read timed out")

    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    fallback_scrapers.clear_fallback_cache()
    caplog.set_level(logging.ERROR)

    for _ in range(2):
        with pytest.raises(requests.Timeout):
            fallback_scrapers._get_soup("https://www.sherdog.com/fighter/example", max_retries=1)

    error_records = [
        record
        for record in caplog.records
        if record.levelname == "ERROR"
        and "External data source unavailable: Sherdog - request timed out" in record.getMessage()
    ]
    assert len(error_records) == 1
    assert calls == [
        "https://www.sherdog.com/fighter/example",
        "https://www.sherdog.com/fighter/example",
    ]


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
    monkeypatch.setattr(fallback_scrapers, "BRAVE_SEARCH_API_KEY", "")
    monkeypatch.setattr(fallback_scrapers, "BRAVE_SEARCH_HTML_FALLBACK_ENABLED", True)
    monkeypatch.setattr(fallback_scrapers.requests, "get", lambda *args, **kwargs: _FakeResponse())
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.search_tapology("Steve Nelmark")

    assert result == "https://www.tapology.com/fightcenter/fighters/steve-nelmark-the-sandman"


def test_site_search_uses_brave_api_when_configured(monkeypatch):
    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "web": {
                    "results": [
                        {
                            "url": "https://www.tapology.com/fightcenter/fighters/steve-nelmark-the-sandman",
                            "title": "Steve Nelmark | MMA Fighter Page | Tapology",
                            "description": "Steve Nelmark MMA profile",
                        }
                    ]
                }
            }

    calls = []

    def fake_get(url, *args, **kwargs):
        calls.append((url, kwargs))
        return _FakeResponse()

    monkeypatch.setattr(fallback_scrapers, "BRAVE_SEARCH_API_KEY", "test-api-key")
    monkeypatch.setattr(fallback_scrapers, "BRAVE_SEARCH_HTML_FALLBACK_ENABLED", False)
    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers._search_site_candidates(
        "Steve Nelmark",
        site_query="tapology.com/fightcenter/fighters",
        required_path_fragment="/fightcenter/fighters/",
    )

    assert result[0][0] == "https://www.tapology.com/fightcenter/fighters/steve-nelmark-the-sandman"
    assert result[0][1] >= 8
    assert calls[0][0] == fallback_scrapers.BRAVE_SEARCH_API_URL
    assert calls[0][1]["headers"]["X-Subscription-Token"] == "test-api-key"


def test_site_search_skips_brave_html_without_api_key(monkeypatch):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Brave HTML search should be disabled by default")

    monkeypatch.setattr(fallback_scrapers, "BRAVE_SEARCH_API_KEY", "")
    monkeypatch.setattr(fallback_scrapers, "BRAVE_SEARCH_HTML_FALLBACK_ENABLED", False)
    monkeypatch.setattr(fallback_scrapers, "DUCKDUCKGO_SEARCH_HTML_FALLBACK_ENABLED", False)
    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers._search_site_candidates(
        "Dakota Weigher",
        site_query="tapology.com/fightcenter/fighters",
        required_path_fragment="/fightcenter/fighters/",
    )

    assert result == []
    assert calls == []
    assert fallback_scrapers._site_search_disabled is True


def test_site_search_rate_limit_warns_once_without_error(monkeypatch, caplog):
    class _FakeResponse:
        text = ""
        status_code = 429

    calls = []

    def fake_get(url, *args, **kwargs):
        calls.append((url, kwargs))
        return _FakeResponse()

    monkeypatch.setattr(fallback_scrapers, "BRAVE_SEARCH_API_KEY", "")
    monkeypatch.setattr(fallback_scrapers, "BRAVE_SEARCH_HTML_FALLBACK_ENABLED", True)
    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    fallback_scrapers.clear_fallback_cache()
    caplog.set_level(logging.DEBUG, logger="src.data.fallback_scrapers")

    result = fallback_scrapers._search_site_candidates(
        "Dakota Weigher",
        site_query="tapology.com/fightcenter/fighters",
        required_path_fragment="/fightcenter/fighters/",
    )

    assert result == []
    assert fallback_scrapers._site_search_disabled is True
    assert len(calls) == 1
    assert any(
        record.levelno == logging.WARNING
        and "External data source unavailable: Brave site search - HTML search blocked or rate limited"
        in record.getMessage()
        for record in caplog.records
    )
    assert not any(
        record.levelno >= logging.ERROR and "Brave site search" in record.getMessage()
        for record in caplog.records
    )


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
    monkeypatch.setattr(fallback_scrapers, "BRAVE_SEARCH_API_KEY", "")
    monkeypatch.setattr(fallback_scrapers, "BRAVE_SEARCH_HTML_FALLBACK_ENABLED", True)
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


def test_search_tapology_cloudflare_403_marks_runtime_blocked_without_site_search(monkeypatch, caplog):
    native_calls = []
    site_calls = []

    def fake_get_tapology_soup(_url, params=None, max_retries=None, retry_statuses=None):
        native_calls.append(
            {
                "url": _url,
                "term": (params or {}).get("term"),
                "max_retries": max_retries,
                "retry_statuses": retry_statuses,
            }
        )
        raise fallback_scrapers.TapologyRequestError(
            _url,
            status_code=403,
            detail="Cloudflare challenge",
        )

    def fake_get(url, *args, **kwargs):
        site_calls.append(url)
        raise AssertionError("site search should be skipped when Tapology is Cloudflare-blocked")

    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_PROXY_URL", "")
    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_READER_FALLBACK_ENABLED", False)
    monkeypatch.setattr(fallback_scrapers, "_get_tapology_soup", fake_get_tapology_soup)
    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    fallback_scrapers.clear_fallback_cache()
    caplog.set_level(logging.ERROR)

    result = fallback_scrapers.search_tapology_candidates("Steve Nelmark", limit=1)

    assert result == []
    assert fallback_scrapers._tapology_blocked is True
    assert native_calls == [
        {
            "url": fallback_scrapers.TAPOLOGY_SEARCH_URL,
            "term": "Steve Nelmark",
            "max_retries": 1,
            "retry_statuses": {429, 503},
        }
    ]
    assert site_calls == []
    assert any(
        record.levelname == "ERROR"
        and "External data source unavailable: Tapology - native search blocked by Cloudflare" in record.getMessage()
        for record in caplog.records
    )


def test_fallback_lookup_merges_static_profile_fields_across_sources(monkeypatch):
    monkeypatch.setattr(fallback_scrapers, "search_sherdog", lambda _name: None)
    tapology_calls = []

    def fake_search_tapology(_name):
        tapology_calls.append(_name)
        return None

    monkeypatch.setattr(fallback_scrapers, "search_tapology", fake_search_tapology)
    monkeypatch.setattr(
        fallback_scrapers,
        "search_martialbot",
        lambda _name: "https://www.martialbot.com/mma/fighters/example",
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "scrape_martialbot_profile",
        lambda _url: {
            "name": "Benoit Saint-Denis",
            "fighter_url": _url,
            "record": "17-3-0",
            "wins": 17,
            "losses": 3,
            "draws": 0,
            "height_raw": "180 cm",
            "height": 180.0,
            "reach_raw": "185 cm",
            "reach": 185.0,
            "weight_raw": "",
            "weight": np.nan,
            "stance": "orthodox",
            "dob": "1995-12-18",
            "age": 30.5,
        },
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "search_espn",
        lambda _name: "https://www.espn.com/mma/fighter/_/id/1/example",
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "scrape_espn_profile",
        lambda _url: {
            "name": "Benoit Saint-Denis",
            "fighter_url": _url,
            "height_raw": "5' 11\"",
            "height": 180.34,
            "reach_raw": "73\"",
            "reach": 185.42,
            "weight_raw": "156 lbs",
            "weight": 156.0,
            "stance": "Southpaw",
            "dob": "1995-12-18",
        },
    )
    fightdx_calls = []
    monkeypatch.setattr(
        fallback_scrapers,
        "search_fightdx",
        lambda _name: "https://fightdx.com/person/example",
    )

    def fake_scrape_fightdx_profile(_url):
        fightdx_calls.append(_url)
        return {
            "name": "Benoit Saint-Denis",
            "fighter_url": _url,
            "height_raw": "5'11\" (1.8m)",
            "height": 180.34,
            "reach_raw": "6'1\" (1.85m)",
            "reach": 185.0,
            "weight_raw": "154lbs (70kg)",
            "weight": 154.0,
            "stance": "Kickboxing, Judo, Bjj",
            "dob": "",
            "age": 30.0,
        }

    monkeypatch.setattr(fallback_scrapers, "scrape_fightdx_profile", fake_scrape_fightdx_profile)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.fallback_lookup("Benoit Saint-Denis")

    assert result is not None
    profile, fights = result
    assert fights == []
    assert profile["record"] == "17-3-0"
    assert profile["height"] == pytest.approx(180.34)
    assert profile["reach"] == pytest.approx(185.42)
    assert profile["weight_raw"] == "156 lbs"
    assert profile["weight"] == pytest.approx(156.0)
    assert profile["stance"] == "Southpaw"
    assert profile["dob"] == "1995-12-18"
    assert fightdx_calls == []


def test_fallback_lookup_uses_static_sources_when_tapology_browser_is_blocked(monkeypatch):
    monkeypatch.setattr(fallback_scrapers, "search_sherdog", lambda _name: None)
    monkeypatch.setattr(fallback_scrapers, "TAPOLOGY_READER_FALLBACK_ENABLED", False)
    monkeypatch.setattr(
        fallback_scrapers,
        "_get_tapology_soup",
        lambda *_args, **_kwargs: pytest.fail("blocked Tapology should not be queried"),
    )
    monkeypatch.setattr(fallback_scrapers, "search_espn", lambda _name: None)
    monkeypatch.setattr(
        fallback_scrapers,
        "search_martialbot",
        lambda _name: "https://www.martialbot.com/mma/fighters/nursulton-ruziboev",
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "scrape_martialbot_profile",
        lambda _url: {
            "name": "Nursultan Ruziboev",
            "fighter_url": _url,
            "record": "36-9-2",
            "wins": 36,
            "losses": 9,
            "draws": 2,
            "height_raw": "188 cm",
            "height": 188.0,
            "reach_raw": "76 in",
            "reach": 193.04,
            "weight_raw": "185 lbs",
            "weight": 185.0,
            "stance": "orthodox",
            "dob": "1993-11-19",
            "age": 32.6,
        },
    )
    monkeypatch.setattr(fallback_scrapers, "search_fightdx", lambda _name: None)
    fallback_scrapers.clear_fallback_cache()
    fallback_scrapers._tapology_blocked = True
    fallback_scrapers._tapology_browser_cloudflare_blocked = True

    result = fallback_scrapers.fallback_lookup("Nursultan Ruziboev")

    assert result is not None
    profile, fights = result
    assert fights == []
    assert profile["name"] == "Nursultan Ruziboev"
    assert profile["record"] == "36-9-2"
    assert profile["reach"] == pytest.approx(193.04)
    assert profile["weight"] == pytest.approx(185.0)


def test_fallback_lookup_enriches_sherdog_profile_without_dropping_fights(monkeypatch):
    sherdog_fights = [{"opponent": "Marc Domont", "result": "win"}]
    monkeypatch.setattr(
        fallback_scrapers,
        "search_sherdog",
        lambda _name: "https://www.sherdog.com/fighter/Benoit-St-Denis-317103",
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "scrape_sherdog_page",
        lambda _url, _name: (
            {
                "name": "Benoit St. Denis",
                "fighter_url": _url,
                "record": "17-3-0",
                "wins": 17,
                "losses": 3,
                "draws": 0,
                "height_raw": "5'11\" / 180.34 cm",
                "height": 180.34,
                "reach_raw": "",
                "reach": np.nan,
                "weight_raw": "155 lbs / 70.31 kg",
                "weight": 155.0,
                "stance": "",
                "dob": "Dec 18, 1995",
                "age": 30.0,
            },
            sherdog_fights,
        ),
    )
    tapology_calls = []

    def fake_search_tapology(_name):
        tapology_calls.append(_name)
        return None

    monkeypatch.setattr(fallback_scrapers, "search_tapology", fake_search_tapology)
    monkeypatch.setattr(
        fallback_scrapers,
        "search_espn",
        lambda _name: "https://www.espn.com/mma/fighter/_/id/4895362/benoit-saint-denis",
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "scrape_espn_profile",
        lambda _url: {
            "name": "Benoit Saint-Denis",
            "fighter_url": _url,
            "height_raw": "5' 11\"",
            "height": 180.34,
            "reach_raw": "73\"",
            "reach": 185.42,
            "weight_raw": "156 lbs",
            "weight": 156.0,
            "stance": "Southpaw",
            "dob": "1995-12-18",
            "age": 30.5,
        },
    )
    monkeypatch.setattr(fallback_scrapers, "search_martialbot", lambda _name: None)
    monkeypatch.setattr(fallback_scrapers, "search_fightdx", lambda _name: None)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.fallback_lookup("Benoit Saint-Denis")

    assert result is not None
    profile, fights = result
    assert fights == sherdog_fights
    assert profile["record"] == "17-3-0"
    assert profile["height"] == pytest.approx(180.34)
    assert profile["weight"] == pytest.approx(155.0)
    assert profile["reach"] == pytest.approx(185.42)
    assert profile["stance"] == "Southpaw"
    assert tapology_calls == []


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


def test_search_martialbot_queries_fighters_search_endpoint(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{"id": "jon-jones-1a37d02c", "name": "Jon Jones"}]}

    def fake_get(url, *args, **kwargs):
        captured["url"] = url
        captured["params"] = kwargs.get("params")
        return _FakeResponse()

    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.search_martialbot("Jon Jones")

    # The MartialBot search migrated to a JSON API that requires both the
    # fighter name and an explicit sport; the legacy "/mma/search?term=" path
    # 404s for every fighter.
    assert captured["url"] == "https://www.martialbot.com/api/fighters-search"
    assert captured["params"] == {"name": "Jon Jones", "sport": "mma"}
    assert result == "https://www.martialbot.com/mma/fighters/jon-jones-1a37d02c"


def test_decode_turbo_stream_resolves_refs_and_sentinels():
    # Object keys and values are index references into the flat array; "_<idx>"
    # keys name a field via another array entry, and negative references are
    # null/undefined sentinels.
    rows = [{"_1": 2, "_3": 4, "_5": -1}, "a", "x", "b", [6], "c", "y"]
    assert fallback_scrapers._decode_turbo_stream(rows) == {"a": "x", "b": ["y"], "c": None}
    # Defensive: non-list / empty payloads decode to None rather than raising.
    assert fallback_scrapers._decode_turbo_stream([]) is None
    assert fallback_scrapers._decode_turbo_stream("not-a-list") is None


def test_search_espn_uses_player_search_results(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "count": 2,
                "items": [
                    {
                        "displayName": "Park Jae-Hyun",
                        "sport": "soccer",
                        "links": [
                            {"rel": ["overview"], "href": "https://www.espn.com/soccer/player/_/id/337330/park-jae-hyun"},
                        ],
                    },
                    {
                        "displayName": "Jae Hyun Park",
                        "sport": "mma",
                        "links": [
                            {"rel": ["overview", "desktop", "athlete"], "href": "https://www.espn.com/mma/fighter/_/id/5138589/jae-hyun-park"},
                        ],
                    },
                ],
            }

    monkeypatch.setattr(fallback_scrapers.requests, "get", lambda *args, **kwargs: _FakeResponse())
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.search_espn("Jae Hyun Park")

    assert result == "https://www.espn.com/mma/fighter/_/id/5138589/jae-hyun-park"


def test_scrape_espn_profile_parses_structured_profile(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "displayName": "Jae Hyun Park",
                "dateOfBirth": "2001-12-10T08:00Z",
                "displayHeight": '5\' 10"',
                "displayWeight": "155 lbs",
                "displayReach": '72"',
                "height": 70.0,
                "weight": 155.0,
                "reach": 72.0,
                "stance": {"id": "75", "text": "Orthodox"},
            }

    monkeypatch.setattr(fallback_scrapers.requests, "get", lambda *args, **kwargs: _FakeResponse())
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)

    profile = fallback_scrapers.scrape_espn_profile(
        "https://www.espn.com/mma/fighter/_/id/5138589/jae-hyun-park"
    )

    assert profile["name"] == "Jae Hyun Park"
    assert profile["height_raw"] == '5\' 10"'
    assert profile["reach_raw"] == '72"'
    assert profile["weight_raw"] == "155 lbs"
    assert profile["stance"] == "Orthodox"
    assert profile["dob"] == "2001-12-10"
    assert profile["reach"] == pytest.approx(182.88, abs=0.1)


def test_scrape_espn_profile_tolerates_missing_optional_fields(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "displayName": "Kennedy Freeman",
                "displayHeight": None,
                "displayWeight": None,
                "displayReach": None,
                "height": None,
                "weight": None,
                "reach": None,
                "stance": None,
                "dateOfBirth": None,
            }

    monkeypatch.setattr(fallback_scrapers.requests, "get", lambda *args, **kwargs: _FakeResponse())
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)

    profile = fallback_scrapers.scrape_espn_profile(
        "https://www.espn.com/mma/fighter/_/id/999999/kennedy-freeman"
    )

    assert profile["name"] == "Kennedy Freeman"
    assert profile["height_raw"] == ""
    assert profile["reach_raw"] == ""
    assert profile["weight_raw"] == ""
    assert profile["stance"] == ""
    assert profile["dob"] == ""


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


def test_search_fightdx_uses_search_page_for_alternate_slug(monkeypatch):
    class _FakeResponse:
        def __init__(self, status_code=200, text=""):
            self.status_code = status_code
            self.text = text

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError("http error")
            return None

    calls = []

    def fake_get(url, *args, **kwargs):
        calls.append((url, kwargs.get("params")))
        if url == "https://fightdx.com/person/ian-garry":
            return _FakeResponse(status_code=404)
        if url == "https://fightdx.com/search/":
            assert kwargs.get("params") == {"query": "Ian Garry"}
            return _FakeResponse(
                text="""
                <html><body>
                  <a href="/person/ian-machado-garry">Ian Machado Garry</a>
                  <a href="/person/belal-muhammad">Belal Muhammad</a>
                </body></html>
                """
            )
        if url == "https://fightdx.com/person/ian-machado-garry":
            return _FakeResponse(
                text="""
                <html><head><title>Ian Machado Garry | MMA Fighter Stats &amp; Record</title></head><body>
                  <h1>Ian Machado Garry</h1>
                </body></html>
                """
            )
        raise AssertionError(f"unexpected FightDX URL: {url}")

    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.search_fightdx("Ian Garry")

    assert result == "https://fightdx.com/person/ian-machado-garry"
    assert not any(url == "https://fightdx.com/sitemap.xml" for url, _params in calls)


def test_search_fightdx_empty_search_skips_sitemap_crawl(monkeypatch):
    class _FakeResponse:
        def __init__(self, status_code=200, text=""):
            self.status_code = status_code
            self.text = text

    def fake_get(url, *args, **kwargs):
        if url == "https://fightdx.com/person/dakota-weigher":
            return _FakeResponse(status_code=404)
        if url == "https://fightdx.com/search/":
            assert kwargs.get("params") == {"query": "Dakota Weigher"}
            return _FakeResponse(text="<html><body>No results</body></html>")
        raise AssertionError(f"unexpected FightDX URL: {url}")

    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.search_fightdx("Dakota Weigher")

    assert result is None


def test_search_fightdx_rejects_last_name_only_candidate(monkeypatch):
    class _FakeResponse:
        def __init__(self, status_code=200, text=""):
            self.status_code = status_code
            self.text = text

    fetched_candidate = False

    def fake_get(url, *args, **kwargs):
        nonlocal fetched_candidate
        if url == "https://fightdx.com/person/ian-garry":
            return _FakeResponse(status_code=404)
        if url == "https://fightdx.com/search/":
            return _FakeResponse(
                text="""
                <html><body>
                  <a href="/person/garry">Garry</a>
                </body></html>
                """
            )
        if url == "https://fightdx.com/person/garry":
            fetched_candidate = True
            return _FakeResponse(
                text="""
                <html><head><title>Garry | MMA Fighter Stats &amp; Record</title></head><body>
                  <h1>Garry</h1>
                </body></html>
                """
            )
        raise AssertionError(f"unexpected FightDX URL: {url}")

    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.search_fightdx("Ian Garry")

    assert result is None
    assert fetched_candidate is False


def test_search_fightdx_sitemap_rejects_first_name_only_candidates(monkeypatch):
    class _FakeResponse:
        def __init__(self, status_code=200, text=""):
            self.status_code = status_code
            self.text = text

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError("http error")
            return None

    fetched_candidate = False

    def fake_get(url, *args, **kwargs):
        nonlocal fetched_candidate
        if url == "https://fightdx.com/person/dakota-weigher":
            return _FakeResponse(status_code=404)
        if url == "https://fightdx.com/search/":
            return _FakeResponse(status_code=503)
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
                  <url><loc>https://fightdx.com/person/dakota-bush</loc></url>
                  <url><loc>https://fightdx.com/person/dakota-cochrane</loc></url>
                </urlset>
                """
            )
        if url.startswith("https://fightdx.com/person/dakota-"):
            fetched_candidate = True
            return _FakeResponse(text="<html><body><h1>Dakota Bush</h1></body></html>")
        raise AssertionError(f"unexpected FightDX URL: {url}")

    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.search_fightdx("Dakota Weigher")

    assert result is None
    assert fetched_candidate is False


def test_search_fightdx_timeout_alerts_are_warning_level(monkeypatch, caplog):
    calls = []

    def fake_get(*_args, **_kwargs):
        calls.append((_args[0], _kwargs.get("params")))
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()
    caplog.set_level(logging.WARNING, logger="src.data.fallback_scrapers")

    result = fallback_scrapers.search_fightdx("Dakota Weigher")

    assert result is None
    assert calls == [("https://fightdx.com/person/dakota-weigher", None)]
    assert fallback_scrapers.search_fightdx("Another Fighter") is None
    assert calls == [("https://fightdx.com/person/dakota-weigher", None)]
    assert any(
        record.levelno == logging.WARNING
        and "External data source unavailable: FightDX - profile lookup failed" in record.getMessage()
        for record in caplog.records
    )
    assert not any(
        record.levelno >= logging.ERROR
        and "External data source unavailable: FightDX" in record.getMessage()
        for record in caplog.records
    )


def test_search_fightdx_search_timeout_skips_sitemap(monkeypatch):
    class _FakeResponse:
        status_code = 404
        text = ""

    calls = []

    def fake_get(url, *args, **kwargs):
        calls.append((url, kwargs.get("params")))
        if url == "https://fightdx.com/person/dakota-weigher":
            return _FakeResponse()
        if url == "https://fightdx.com/search/":
            raise requests.exceptions.Timeout("timed out")
        raise AssertionError(f"unexpected FightDX URL after search timeout: {url}")

    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.search_fightdx("Dakota Weigher")

    assert result is None
    assert calls == [
        ("https://fightdx.com/person/dakota-weigher", None),
        ("https://fightdx.com/search/", {"query": "Dakota Weigher"}),
    ]


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
        "_get_fightdx_soup",
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


def test_scrape_fightdx_profile_uses_fightdx_timeout(monkeypatch):
    fallback_scrapers.clear_fallback_cache()
    captured = {}

    class _FakeResponse:
        text = """
        <html><head><title>Marcus Bossett | MMA Fighter Stats &amp; Record</title></head><body>
          <h1>Marcus Bossett</h1>
        </body></html>
        """

        def raise_for_status(self):
            return None

    def fake_get(url, *args, **kwargs):
        captured["url"] = url
        captured["timeout"] = kwargs.get("timeout")
        return _FakeResponse()

    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)

    profile = fallback_scrapers.scrape_fightdx_profile(
        "https://fightdx.com/person/marcus-bossett"
    )

    assert profile["name"] == "Marcus Bossett"
    assert captured == {
        "url": "https://fightdx.com/person/marcus-bossett",
        "timeout": fallback_scrapers.FIGHTDX_REQUEST_TIMEOUT_SECONDS,
    }


def test_scrape_martialbot_profile_parses_reach_and_exact_dob(monkeypatch):
    # MartialBot fighter pages are client-rendered; the bio is served by the
    # React Router single-fetch ".data" route as a turbo-stream payload: a flat
    # array whose objects/arrays reference children by index ("_<idx>" keys),
    # with negative references acting as null/undefined sentinels. This fixture
    # encodes a Patrick Smith bio in that exact wire format.
    data_rows = [
        {"_1": 2}, "routes/mma.fighters.$id", {"_3": 4}, "data", {"_5": 6}, "data",
        {"_7": 8, "_37": 38}, "fighter",
        {"_9": 10, "_11": 12, "_13": 14, "_15": 16, "_17": 18, "_19": 20,
         "_21": 22, "_23": 24, "_26": 27, "_36": -1},
        "id", "patrick-smith-0d62c17098b5f99e75621d360b1cfc1e",
        "name", "Patrick Smith", "stance", "orthodox", "birthdate", "1963-08-28",
        "height_cm", "188 cm", "reach_cm", "188 cm",
        "latest_weight_class", "heavyweight",
        "nationalities", [25], "American",
        "record", {"_28": 29, "_30": 31, "_32": 33, "_34": 35},
        "wins", 20, "losses", 17, "draws", 0, "no_contests", 0,
        "deathDate", "fights", [],
    ]

    captured: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return data_rows

    def fake_get(url, *args, **kwargs):
        captured["url"] = url
        return _FakeResponse()

    monkeypatch.setattr(fallback_scrapers.requests, "get", fake_get)
    monkeypatch.setattr(fallback_scrapers, "_sleep_after_request", lambda _seconds: None)

    profile = fallback_scrapers.scrape_martialbot_profile(
        "https://www.martialbot.com/mma/fighters/patrick-smith-0d62c17098b5f99e75621d360b1cfc1e"
    )

    # The structured bio is fetched from the ".data" single-fetch route.
    assert str(captured["url"]).endswith(".data")
    assert profile["name"] == "Patrick Smith"
    assert profile["record"] == "20-17-0"
    assert profile["wins"] == 20
    assert profile["losses"] == 17
    assert profile["draws"] == 0
    assert profile["height_raw"] == "188 cm"
    assert profile["height"] == pytest.approx(188.0, abs=0.1)
    assert profile["reach_raw"] == "188 cm"
    assert profile["reach"] == pytest.approx(188.0, abs=0.1)
    assert profile["stance"] == "orthodox"
    assert profile["dob"] == "1963-08-28"
    assert profile["weight_raw"] == ""


def test_scrape_martialbot_profile_canonicalizes_switcher_stance(monkeypatch):
    # MartialBot labels switch-stance fighters "switcher", which the feature
    # encoder does not recognize; the scraper must map it to "switch".
    from src.features.stance_utils import encode_stance

    monkeypatch.setattr(
        fallback_scrapers,
        "_fetch_martialbot_fighter",
        lambda _url: {
            "name": "Anderson Silva",
            "stance": "switcher",
            "height_cm": "188 cm",
            "reach_cm": "197 cm",
            "birthdate": "1975-04-14",
            "record": {"wins": 34, "losses": 11, "draws": 0},
        },
    )

    profile = fallback_scrapers.scrape_martialbot_profile(
        "https://www.martialbot.com/mma/fighters/anderson-silva-abc123"
    )

    assert profile["stance"] == "switch"
    assert encode_stance(profile["stance"]) == 2.0


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


def test_build_features_preserves_precomputed_market_probabilities_when_raw_odds_missing(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        build_features_module,
        "_resolve_pre_ufc_supplement_path",
        lambda: tmp_path / "missing_pre_ufc.csv",
    )

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
