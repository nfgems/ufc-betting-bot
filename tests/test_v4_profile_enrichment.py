import pandas as pd
import pytest
from bs4 import BeautifulSoup

from src.data import fallback_scrapers, fighter_lookup, scraper, ufc_refresh
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
        lambda _url, params=None: BeautifulSoup(html, "lxml"),
    )
    fallback_scrapers.clear_fallback_cache()

    result = fallback_scrapers.search_tapology("Steve Nelmark")

    assert result == "https://www.tapology.com/fightcenter/fighters/steve-nelmark-the-sandman"


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
    assert pd.isna(profile["reach"])


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
    assert pd.isna(profile["reach"])


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
    assert pd.isna(profile["reach"])


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
        "src.data.historical_backfill.load_historical_odds",
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
    ).to_csv(rankings_dir / "rankings_history.csv", index=False)

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
    ).to_csv(method_dir / "historical_method_odds.csv", index=False)

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
    monkeypatch.setattr(fighter_lookup, "get_fighter_elo", lambda *_args, **_kwargs: 1500.0)
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


def test_build_features_preserves_unknown_stance_as_nan():
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


def test_build_features_encodes_additional_ufcstats_stance_labels():
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
    monkeypatch.setattr(fighter_lookup, "get_fighter_elo", lambda *_args, **_kwargs: 1500.0)
    fighter_lookup.clear_cache()

    result = fighter_lookup.lookup_fighter(
        "Beta Fighter",
        training_spec=training_spec.full_live_contract_v4_spec(),
    )

    assert result is not None
    assert result["features"]["stance_enc"] == pytest.approx(3.0)

    fighter_lookup.clear_cache()
