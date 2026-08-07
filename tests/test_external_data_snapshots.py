import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
from bs4 import BeautifulSoup

from scripts import scrape_bfo_moneyline
from src.data import line_tracker, live_monitor, method_odds, rankings_scraper


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _fresh_snapshot_time(*, hours_ago: int = 0) -> str:
    return (datetime.now().replace(microsecond=0) - timedelta(hours=hours_ago)).isoformat()


def _snapshot_filename(prefix: str, snapshot_time: str) -> str:
    timestamp = datetime.fromisoformat(snapshot_time).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.json"


def _write_rankings_snapshot(snapshot_dir: Path, snapshot_time: str, payload: dict) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / _snapshot_filename("rankings", snapshot_time)
    path.write_text(json.dumps(payload, indent=2))
    return path


def _write_method_snapshot(snapshot_dir: Path, snapshot_time: str, payload: dict) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / _snapshot_filename("method_odds", snapshot_time)
    path.write_text(json.dumps(payload, indent=2))
    return path


def _patch_line_history_dir(monkeypatch, tmp_path: Path) -> Path:
    history_dir = tmp_path / "line_history"
    history_dir.mkdir()
    monkeypatch.setattr(line_tracker, "LINE_HISTORY_DIR", history_dir)
    monkeypatch.setattr(line_tracker, "OPENING_LINES_PATH", history_dir / "opening_lines.json")
    monkeypatch.setattr(line_tracker, "_alert_log_state", {})
    monkeypatch.setattr(line_tracker, "_alert_log_state_path_loaded", None)
    return history_dir


def test_parse_ufc_rankings_fixture_extracts_normalized_rankings():
    html = (FIXTURES_DIR / "rankings" / "ufc_rankings_sample.html").read_text()
    parsed = rankings_scraper._parse_ufc_rankings_html(html)

    assert parsed is not None
    assert parsed["source"] == "ufc.com"
    assert parsed["wc"]["lightweight"]["islam makhachev"] == 1
    assert parsed["wc"]["women's strawweight"]["zhang weili"] == 1
    assert parsed["pfp"]["alex pereira"] == 2


def test_parse_tapology_rankings_overview_extracts_division_cards():
    html = """
    <html><body>
      <a href="/rankings/ufc/ultimate-fighting-championship-mens-lightweight-155-pounds">Lightweight</a>
      <a href="/fightcenter/fighters/129278-ilia-topuria">Ilia "El Matador" Topuria</a>
      <a href="/fightcenter/fighters/40148-islam-makhachev">Islam Makhachev</a>
      <a href="/rankings/ufc/ultimate-fighting-championship-womens-strawweight-115-pounds">Women's Strawweight</a>
      <a href="/fightcenter/fighters/102073-mackenzie-dern">Mackenzie Dern</a>
      <a href="/fightcenter/fighters/186595-weili-zhang">Weili Zhang</a>
    </body></html>
    """

    parsed = rankings_scraper._parse_tapology_rankings_html(html)

    assert parsed is not None
    assert parsed["source"] == "tapology.com"
    assert parsed["wc"]["lightweight"]["ilia topuria"] == 1
    assert parsed["wc"]["lightweight"]["islam makhachev"] == 2
    assert parsed["wc"]["women's strawweight"]["mackenzie dern"] == 1
    assert parsed["wc"]["women's strawweight"]["weili zhang"] == 2


def test_scrape_tapology_rankings_tries_known_urls_with_browser_aware_fetch(monkeypatch):
    from src.data import fallback_scrapers

    calls = []
    html = """
    <html><body>
      <a href="/rankings/ufc/ultimate-fighting-championship-mens-lightweight-155-pounds">Lightweight</a>
      <a href="/fightcenter/fighters/129278-ilia-topuria">Ilia "El Matador" Topuria</a>
    </body></html>
    """

    def fake_get_tapology_soup(url):
        calls.append(url)
        if len(calls) == 1:
            return BeautifulSoup("<html><body>No ranking links here</body></html>", "lxml")
        return BeautifulSoup(html, "lxml")

    monkeypatch.setattr(fallback_scrapers, "_get_tapology_soup", fake_get_tapology_soup)
    monkeypatch.setattr(
        rankings_scraper,
        "_fetch_html",
        lambda _url: pytest.fail("plain Tapology rankings fetch should not be needed"),
    )

    parsed = rankings_scraper._scrape_tapology_rankings()

    assert parsed is not None
    assert parsed["wc"]["lightweight"]["ilia topuria"] == 1
    assert calls == [
        "https://www.tapology.com/rankings/ufc",
        "https://www.tapology.com/rankings/current-ufc-rankings",
    ]


def test_scrape_tapology_rankings_logs_environment_block_at_info(monkeypatch, caplog):
    from src.data import fallback_scrapers

    def fake_get_tapology_soup(url):
        raise fallback_scrapers.TapologyRequestError(
            url,
            status_code=403,
            detail="Tapology blocked from this environment",
        )

    monkeypatch.setattr(fallback_scrapers, "_get_tapology_soup", fake_get_tapology_soup)

    with caplog.at_level(logging.INFO):
        parsed = rankings_scraper._scrape_tapology_rankings()

    assert parsed is None
    assert "Tapology rankings browser-aware fetch failed" in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_get_rankings_uses_latest_successful_snapshot_not_failed_latest(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "rankings"
    monkeypatch.setattr(rankings_scraper, "RANKINGS_SNAPSHOT_DIR", snapshot_dir)
    fresh_time = _fresh_snapshot_time(hours_ago=1)
    failed_time = _fresh_snapshot_time()

    _write_rankings_snapshot(
        snapshot_dir,
        fresh_time,
        {
            "schema_version": 1,
            "snapshot_time": fresh_time,
            "status": "success",
            "source": "ufc.com",
            "acquisition_failed": False,
            "wc": {"lightweight": {"alpha fighter": 3}},
            "pfp": {"alpha fighter": 8},
            "sources": [],
        },
    )
    _write_rankings_snapshot(
        snapshot_dir,
        failed_time,
        {
            "schema_version": 1,
            "snapshot_time": failed_time,
            "status": "failed",
            "source": "none",
            "acquisition_failed": True,
            "wc": {},
            "pfp": {},
            "sources": [],
        },
    )

    rankings = rankings_scraper.get_rankings()
    fighter_ranks = rankings_scraper.get_fighter_rankings("Alpha Fighter", weight_class="Lightweight")

    assert rankings["source"] == "ufc.com"
    assert fighter_ranks["wc_rank_feat"] == 3
    assert fighter_ranks["pfp_rank_feat"] == 8


def test_get_rankings_without_successful_snapshot_returns_acquisition_failed(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "rankings"
    monkeypatch.setattr(rankings_scraper, "RANKINGS_SNAPSHOT_DIR", snapshot_dir)
    fresh_time = _fresh_snapshot_time()

    _write_rankings_snapshot(
        snapshot_dir,
        fresh_time,
        {
            "schema_version": 1,
            "snapshot_time": fresh_time,
            "status": "failed",
            "source": "none",
            "acquisition_failed": True,
            "wc": {},
            "pfp": {},
            "sources": [],
        },
    )

    rankings = rankings_scraper.get_rankings()
    fighter_ranks = rankings_scraper.get_fighter_rankings("Alpha Fighter", weight_class="Lightweight")

    assert rankings["acquisition_failed"] is True
    assert np.isnan(fighter_ranks["wc_rank_feat"])
    assert np.isnan(fighter_ranks["pfp_rank_feat"])


def test_get_rankings_stale_snapshot_returns_degraded_data(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "rankings"
    monkeypatch.setattr(rankings_scraper, "RANKINGS_SNAPSHOT_DIR", snapshot_dir)

    stale_time = (datetime.now() - rankings_scraper.RANKINGS_SNAPSHOT_MAX_AGE - timedelta(hours=1)).replace(microsecond=0).isoformat()
    _write_rankings_snapshot(
        snapshot_dir,
        stale_time,
        {
            "schema_version": 1,
            "snapshot_time": stale_time,
            "status": "success",
            "source": "ufc.com",
            "acquisition_failed": False,
            "wc": {"lightweight": {"alpha fighter": 3}},
            "pfp": {"alpha fighter": 8},
            "sources": [],
        },
    )

    rankings = rankings_scraper.get_rankings()
    fighter_ranks = rankings_scraper.get_fighter_rankings("Alpha Fighter", weight_class="Lightweight")

    assert rankings["status"] == "stale"
    assert rankings["acquisition_failed"] is True
    assert np.isnan(fighter_ranks["wc_rank_feat"])
    assert np.isnan(fighter_ranks["pfp_rank_feat"])


def test_rankings_lookup_uses_snapshot_at_or_before_as_of_date(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "rankings"
    monkeypatch.setattr(rankings_scraper, "RANKINGS_SNAPSHOT_DIR", snapshot_dir)

    _write_rankings_snapshot(
        snapshot_dir,
        "2024-01-10T10:00:00",
        {
            "schema_version": 1,
            "snapshot_time": "2024-01-10T10:00:00",
            "status": "success",
            "source": "ufc.com",
            "acquisition_failed": False,
            "wc": {"lightweight": {"alpha fighter": 7}},
            "pfp": {"alpha fighter": 15},
            "sources": [],
        },
    )
    _write_rankings_snapshot(
        snapshot_dir,
        "2024-03-10T10:00:00",
        {
            "schema_version": 1,
            "snapshot_time": "2024-03-10T10:00:00",
            "status": "success",
            "source": "ufc.com",
            "acquisition_failed": False,
            "wc": {"lightweight": {"alpha fighter": 2}},
            "pfp": {"alpha fighter": 5},
            "sources": [],
        },
    )

    fighter_ranks = rankings_scraper.get_fighter_rankings(
        "Alpha Fighter",
        weight_class="Lightweight",
        as_of_date="2024-02-01",
    )

    assert fighter_ranks["wc_rank_feat"] == 7
    assert fighter_ranks["pfp_rank_feat"] == 15


def test_rankings_require_full_name_match_not_last_name_only(monkeypatch):
    monkeypatch.setattr(rankings_scraper, "get_rankings", lambda **_kwargs: {
        "wc": {"middleweight": {"anderson silva": 4}},
        "pfp": {"anderson silva": 10},
        "source": "ufc.com",
        "acquisition_failed": False,
    })

    result = rankings_scraper.get_fighter_rankings("Bruno Silva", weight_class="Middleweight")

    assert result["wc_rank_feat"] == rankings_scraper.UNRANKED_DEFAULT
    assert result["pfp_rank_feat"] == rankings_scraper.UNRANKED_DEFAULT


def test_womens_rankings_lookup_supports_legacy_collapsed_snapshot(monkeypatch):
    monkeypatch.setattr(rankings_scraper, "get_rankings", lambda **_kwargs: {
        "wc": {"strawweight": {"zhang weili": 1}},
        "pfp": {},
        "source": "ufc.com",
        "acquisition_failed": False,
    })

    result = rankings_scraper.get_fighter_rankings("Zhang Weili", weight_class="Women's Strawweight")

    assert result["wc_rank_feat"] == 1
    assert result["pfp_rank_feat"] == rankings_scraper.UNRANKED_DEFAULT


def test_parse_bfo_method_odds_fixture():
    html = (FIXTURES_DIR / "method_odds" / "bfo_method_odds_sample.html").read_text()
    soup = BeautifulSoup(html, "lxml")

    result = method_odds._parse_bfo_method_odds(soup, "Bruno Silva", "Anderson Silva")

    assert result is not None
    assert result["a_ko_odds_prob"] == pytest.approx(method_odds._american_to_implied_prob(200))
    assert result["b_dec_odds_prob"] == pytest.approx(method_odds._american_to_implied_prob(150))


def test_parse_bfo_method_odds_ambiguous_fixture_returns_none():
    html = (FIXTURES_DIR / "method_odds" / "bfo_method_odds_ambiguous.html").read_text()
    soup = BeautifulSoup(html, "lxml")

    result = method_odds._parse_bfo_method_odds(soup, "Bruno Silva", "Anderson Silva")

    assert result is None


def test_parse_bfo_method_odds_scopes_stale_replacement_matchup():
    html = """
    <html><body>
      <table class="odds-table odds-table-responsive-header">
        <tr><th>Sportsbook</th></tr>
        <tr id="mu-44409"><th><a href="/fighters/gianni-vazquez"><span class="t-b-fcc">Gianni Vazquez</span></a></th></tr>
        <tr><th><a href="/fighters/miles-johns"><span class="t-b-fcc">Miles Johns</span></a></th></tr>
        <tr class="pr"><th>Over 2.5 rounds</th></tr>
        <tr class="pr"><th>Under 2.5 rounds</th></tr>
        <tr id="mu-44334"><th><a href="/fighters/jessie-rosas"><span class="t-b-fcc">Jessie Rosas</span></a></th></tr>
        <tr><th><a href="/fighters/miles-johns"><span class="t-b-fcc">Miles Johns</span></a></th></tr>
        <tr class="pr"><th>Rosas wins by TKO/KO</th></tr>
        <tr class="pr"><th>Johns wins by TKO/KO</th></tr>
        <tr class="pr"><th>Rosas wins by submission</th></tr>
        <tr class="pr"><th>Johns wins by submission</th></tr>
        <tr class="pr"><th>Rosas wins by decision</th></tr>
        <tr class="pr"><th>Johns wins by decision</th></tr>
      </table>
      <table class="odds-table">
        <tr><th>Sportsbook</th></tr>
        <tr><th><span class="t-b-fcc">Gianni Vazquez</span></th><td>+140</td></tr>
        <tr><th><span class="t-b-fcc">Miles Johns</span></th><td>-160</td></tr>
        <tr class="pr"><th>Over 2.5 rounds</th><td>-180</td></tr>
        <tr class="pr"><th>Under 2.5 rounds</th><td>+150</td></tr>
        <tr><th><span class="t-b-fcc">Jessie Rosas</span></th><td>+130</td></tr>
        <tr><th><span class="t-b-fcc">Miles Johns</span></th><td>-175</td></tr>
        <tr class="pr"><th>Rosas wins by TKO/KO</th><td>+200</td></tr>
        <tr class="pr"><th>Johns wins by TKO/KO</th><td>+150</td></tr>
        <tr class="pr"><th>Rosas wins by submission</th><td>+400</td></tr>
        <tr class="pr"><th>Johns wins by submission</th><td>+500</td></tr>
        <tr class="pr"><th>Rosas wins by decision</th><td>+300</td></tr>
        <tr class="pr"><th>Johns wins by decision</th><td>+250</td></tr>
      </table>
      <table class="odds-table">
        <tr><th>Sportsbook</th></tr>
        <tr><th><span class="t-b-fcc">Gamma Fighter</span></th><td>-110</td></tr>
        <tr><th><span class="t-b-fcc">Delta Fighter</span></th><td>-110</td></tr>
        <tr class="pr"><th>Johns wins by TKO/KO</th><td>+111</td></tr>
        <tr class="pr"><th>Johns wins by decision</th><td>+222</td></tr>
        <tr><th>Filler 1</th><td>n/a</td></tr>
        <tr><th>Filler 2</th><td>n/a</td></tr>
        <tr><th>Filler 3</th><td>n/a</td></tr>
        <tr><th>Filler 4</th><td>n/a</td></tr>
        <tr><th>Filler 5</th><td>n/a</td></tr>
        <tr><th>Filler 6</th><td>n/a</td></tr>
        <tr><th>Filler 7</th><td>n/a</td></tr>
        <tr><th>Filler 8</th><td>n/a</td></tr>
      </table>
    </body></html>
    """
    soup = BeautifulSoup(html, "lxml")

    current_matchup = method_odds._parse_bfo_method_odds(
        soup,
        "Miles Johns",
        "Gianni Vazquez",
    )
    stale_matchup = method_odds._parse_bfo_method_odds(
        soup,
        "Jessie Rosas",
        "Miles Johns",
    )

    assert current_matchup is None
    assert stale_matchup is not None
    assert stale_matchup["a_ko_odds_prob"] == pytest.approx(
        method_odds._american_to_implied_prob(200)
    )
    assert stale_matchup["b_dec_odds_prob"] == pytest.approx(
        method_odds._american_to_implied_prob(250)
    )


def test_parse_bfo_method_odds_accepts_unique_last_name_shorthand():
    html = """
    <html><body>
      <h1>Sean Strickland vs Dricus Du Plessis</h1>
      <table>
        <tr><td>Plessis wins by TKO/KO</td><td>+230</td></tr>
        <tr><td>Plessis wins by submission</td><td>+500</td></tr>
        <tr><td>Plessis wins by decision</td><td>+750</td></tr>
        <tr><td>Strickland wins by TKO/KO</td><td>+250</td></tr>
        <tr><td>Strickland wins by submission</td><td>+1400</td></tr>
        <tr><td>Strickland wins by decision</td><td>+340</td></tr>
      </table>
    </body></html>
    """
    soup = BeautifulSoup(html, "lxml")

    result = method_odds._parse_bfo_method_odds(
        soup,
        "Sean Strickland",
        "Dricus Du Plessis",
    )

    assert result["a_ko_odds_prob"] == pytest.approx(method_odds._american_to_implied_prob(250))
    assert result["a_sub_odds_prob"] == pytest.approx(method_odds._american_to_implied_prob(1400))
    assert result["a_dec_odds_prob"] == pytest.approx(method_odds._american_to_implied_prob(340))
    assert result["b_ko_odds_prob"] == pytest.approx(method_odds._american_to_implied_prob(230))
    assert result["b_sub_odds_prob"] == pytest.approx(method_odds._american_to_implied_prob(500))
    assert result["b_dec_odds_prob"] == pytest.approx(method_odds._american_to_implied_prob(750))


def test_parse_bfo_method_odds_accepts_curated_surname_spelling_alias():
    html = """
    <html><body>
      <h1>Bogdan Grad vs Dennis Buzukia</h1>
      <table>
        <tr><th>Grad wins by TKO/KO</th><td>+375</td></tr>
        <tr><th>Grad wins by submission</th><td>+600</td></tr>
        <tr><th>Grad wins by decision</th><td>+145</td></tr>
        <tr><th>Buzukia wins by TKO/KO</th><td>+450</td></tr>
        <tr><th>Buzukia wins by submission</th><td>+1650</td></tr>
        <tr><th>Buzukia wins by decision</th><td>+425</td></tr>
      </table>
    </body></html>
    """
    soup = BeautifulSoup(html, "lxml")

    result = method_odds._parse_bfo_method_odds(
        soup,
        "Dennis Buzukja",
        "Bogdan Grad",
    )

    assert result is not None
    assert result["a_ko_odds_prob"] == pytest.approx(method_odds._american_to_implied_prob(450))
    assert result["a_sub_odds_prob"] == pytest.approx(method_odds._american_to_implied_prob(1650))
    assert result["a_dec_odds_prob"] == pytest.approx(method_odds._american_to_implied_prob(425))
    assert result["b_ko_odds_prob"] == pytest.approx(method_odds._american_to_implied_prob(375))
    assert result["b_sub_odds_prob"] == pytest.approx(method_odds._american_to_implied_prob(600))
    assert result["b_dec_odds_prob"] == pytest.approx(method_odds._american_to_implied_prob(145))


def test_method_odds_reads_snapshot_with_event_context(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "method_odds"
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", snapshot_dir)
    method_odds._method_odds_cache.clear()
    fresh_time = _fresh_snapshot_time()

    _write_method_snapshot(
        snapshot_dir,
        fresh_time,
        {
            "schema_version": 1,
            "snapshot_time": fresh_time,
            "status": "success",
            "record_count": 2,
            "sources": [],
            "records": [
                {
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "fighter_a_norm": "alpha fighter",
                    "fighter_b_norm": "beta fighter",
                    "event_id": "evt-1",
                    "commence_time": "2024-02-01T18:00:00Z",
                    "event_title": "Event 1",
                    "source": "odds_api:method_of_victory",
                    "captured_at": "2024-01-01T10:00:00",
                    "method_odds": {
                        "a_ko_odds_prob": 0.40,
                        "a_sub_odds_prob": None,
                        "a_dec_odds_prob": None,
                        "b_ko_odds_prob": None,
                        "b_sub_odds_prob": None,
                        "b_dec_odds_prob": 0.30,
                    },
                },
                {
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "fighter_a_norm": "alpha fighter",
                    "fighter_b_norm": "beta fighter",
                    "event_id": "evt-2",
                    "commence_time": "2024-03-01T18:00:00Z",
                    "event_title": "Event 2",
                    "source": "odds_api:method_of_victory",
                    "captured_at": "2024-01-01T10:00:00",
                    "method_odds": {
                        "a_ko_odds_prob": 0.55,
                        "a_sub_odds_prob": None,
                        "a_dec_odds_prob": None,
                        "b_ko_odds_prob": None,
                        "b_sub_odds_prob": None,
                        "b_dec_odds_prob": 0.22,
                    },
                },
            ],
        },
    )

    result = method_odds.get_method_odds(
        "Alpha Fighter",
        "Beta Fighter",
        event_id="evt-2",
        commence_time="2024-03-01T18:00:00Z",
    )

    assert result["a_ko_odds_prob"] == pytest.approx(0.55)
    assert result["b_dec_odds_prob"] == pytest.approx(0.22)


def test_method_odds_ambiguous_snapshot_match_without_event_context_returns_nan(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "method_odds"
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", snapshot_dir)
    method_odds._method_odds_cache.clear()
    fresh_time = _fresh_snapshot_time()

    _write_method_snapshot(
        snapshot_dir,
        fresh_time,
        {
            "schema_version": 1,
            "snapshot_time": fresh_time,
            "status": "success",
            "record_count": 2,
            "sources": [],
            "records": [
                {
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "fighter_a_norm": "alpha fighter",
                    "fighter_b_norm": "beta fighter",
                    "event_id": "evt-1",
                    "commence_time": "2024-02-01T18:00:00Z",
                    "event_title": "Event 1",
                    "source": "odds_api:method_of_victory",
                    "captured_at": "2024-01-01T10:00:00",
                    "method_odds": {"a_ko_odds_prob": 0.40, "a_sub_odds_prob": None, "a_dec_odds_prob": None, "b_ko_odds_prob": None, "b_sub_odds_prob": None, "b_dec_odds_prob": 0.30},
                },
                {
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "fighter_a_norm": "alpha fighter",
                    "fighter_b_norm": "beta fighter",
                    "event_id": "evt-2",
                    "commence_time": "2024-03-01T18:00:00Z",
                    "event_title": "Event 2",
                    "source": "odds_api:method_of_victory",
                    "captured_at": "2024-01-01T10:00:00",
                    "method_odds": {"a_ko_odds_prob": 0.55, "a_sub_odds_prob": None, "a_dec_odds_prob": None, "b_ko_odds_prob": None, "b_sub_odds_prob": None, "b_dec_odds_prob": 0.22},
                },
            ],
        },
    )

    result = method_odds.get_method_odds("Alpha Fighter", "Beta Fighter")

    assert all(np.isnan(result[column]) for column in result)


def test_method_odds_short_lived_cache_survives_snapshot_removal(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "method_odds"
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", snapshot_dir)
    method_odds._method_odds_cache.clear()
    fresh_time = _fresh_snapshot_time()

    snapshot_path = _write_method_snapshot(
        snapshot_dir,
        fresh_time,
        {
            "schema_version": 1,
            "snapshot_time": fresh_time,
            "status": "success",
            "record_count": 1,
            "sources": [],
            "records": [
                {
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "fighter_a_norm": "alpha fighter",
                    "fighter_b_norm": "beta fighter",
                    "event_id": "evt-1",
                    "commence_time": "2024-02-01T18:00:00Z",
                    "event_title": "Event 1",
                    "source": "odds_api:method_of_victory",
                    "captured_at": "2024-01-01T10:00:00",
                    "method_odds": {"a_ko_odds_prob": 0.40, "a_sub_odds_prob": None, "a_dec_odds_prob": None, "b_ko_odds_prob": None, "b_sub_odds_prob": None, "b_dec_odds_prob": 0.30},
                }
            ],
        },
    )

    first = method_odds.get_method_odds("Alpha Fighter", "Beta Fighter", event_id="evt-1")
    snapshot_path.unlink()
    second = method_odds.get_method_odds("Alpha Fighter", "Beta Fighter", event_id="evt-1")

    assert first == second
    assert second["a_ko_odds_prob"] == pytest.approx(0.40)


def test_method_odds_stale_snapshot_returns_nan(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "method_odds"
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", snapshot_dir)
    method_odds._method_odds_cache.clear()

    stale_time = (datetime.now() - method_odds.METHOD_ODDS_SNAPSHOT_MAX_AGE - timedelta(hours=1)).replace(microsecond=0).isoformat()
    _write_method_snapshot(
        snapshot_dir,
        stale_time,
        {
            "schema_version": 1,
            "snapshot_time": stale_time,
            "status": "success",
            "record_count": 1,
            "sources": [],
            "records": [
                {
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "fighter_a_norm": "alpha fighter",
                    "fighter_b_norm": "beta fighter",
                    "event_id": "evt-1",
                    "commence_time": "2024-02-01T18:00:00Z",
                    "event_title": "Event 1",
                    "source": "odds_api:method_of_victory",
                    "captured_at": stale_time,
                    "method_odds": {"a_ko_odds_prob": 0.40, "a_sub_odds_prob": None, "a_dec_odds_prob": None, "b_ko_odds_prob": None, "b_sub_odds_prob": None, "b_dec_odds_prob": 0.30},
                }
            ],
        },
    )

    result = method_odds.get_method_odds("Alpha Fighter", "Beta Fighter", event_id="evt-1")

    assert all(np.isnan(result[column]) for column in result)


def test_method_odds_uses_snapshot_at_or_before_as_of_date(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "method_odds"
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", snapshot_dir)
    monkeypatch.setattr(
        method_odds,
        "METHOD_ODDS_SNAPSHOT_MAX_AGE",
        timedelta(hours=48),
    )
    method_odds._method_odds_cache.clear()

    _write_method_snapshot(
        snapshot_dir,
        "2024-01-14T12:00:00",
        {
            "schema_version": 1,
            "snapshot_time": "2024-01-14T12:00:00",
            "status": "success",
            "record_count": 1,
            "sources": [],
            "records": [
                {
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "fighter_a_norm": "alpha fighter",
                    "fighter_b_norm": "beta fighter",
                    "event_id": "evt-1",
                    "commence_time": "2024-02-01T18:00:00Z",
                    "event_title": "Event 1",
                    "source": "odds_api:method_of_victory",
                    "captured_at": "2024-01-14T12:00:00",
                    "method_odds": {
                        "a_ko_odds_prob": 0.40,
                        "a_sub_odds_prob": None,
                        "a_dec_odds_prob": None,
                        "b_ko_odds_prob": None,
                        "b_sub_odds_prob": None,
                        "b_dec_odds_prob": 0.30,
                    },
                }
            ],
        },
    )
    _write_method_snapshot(
        snapshot_dir,
        "2024-01-16T12:00:00",
        {
            "schema_version": 1,
            "snapshot_time": "2024-01-16T12:00:00",
            "status": "success",
            "record_count": 1,
            "sources": [],
            "records": [
                {
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "fighter_a_norm": "alpha fighter",
                    "fighter_b_norm": "beta fighter",
                    "event_id": "evt-1",
                    "commence_time": "2024-02-01T18:00:00Z",
                    "event_title": "Event 1",
                    "source": "odds_api:method_of_victory",
                    "captured_at": "2024-01-16T12:00:00",
                    "method_odds": {
                        "a_ko_odds_prob": 0.55,
                        "a_sub_odds_prob": None,
                        "a_dec_odds_prob": None,
                        "b_ko_odds_prob": None,
                        "b_sub_odds_prob": None,
                        "b_dec_odds_prob": 0.22,
                    },
                }
            ],
        },
    )

    result = method_odds.get_method_odds(
        "Alpha Fighter",
        "Beta Fighter",
        event_id="evt-1",
        commence_time="2024-02-01T18:00:00Z",
        as_of_date="2024-01-15T12:00:00Z",
    )

    assert result["a_ko_odds_prob"] == pytest.approx(0.40)
    assert result["b_dec_odds_prob"] == pytest.approx(0.30)


def test_method_odds_rejects_snapshot_older_than_as_of_lookback(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "method_odds"
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", snapshot_dir)
    monkeypatch.setattr(
        method_odds,
        "METHOD_ODDS_SNAPSHOT_MAX_AGE",
        timedelta(hours=48),
    )
    method_odds._method_odds_cache.clear()

    _write_method_snapshot(
        snapshot_dir,
        "2024-01-12T12:00:00",
        {
            "schema_version": 1,
            "snapshot_time": "2024-01-12T12:00:00",
            "status": "success",
            "record_count": 1,
            "sources": [],
            "records": [
                {
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "fighter_a_norm": "alpha fighter",
                    "fighter_b_norm": "beta fighter",
                    "event_id": "evt-1",
                    "commence_time": "2024-02-01T18:00:00Z",
                    "event_title": "Event 1",
                    "source": "odds_api:method_of_victory",
                    "captured_at": "2024-01-12T12:00:00",
                    "method_odds": {
                        "a_ko_odds_prob": 0.40,
                        "a_sub_odds_prob": None,
                        "a_dec_odds_prob": None,
                        "b_ko_odds_prob": None,
                        "b_sub_odds_prob": None,
                        "b_dec_odds_prob": 0.30,
                    },
                }
            ],
        },
    )

    result = method_odds.get_method_odds(
        "Alpha Fighter",
        "Beta Fighter",
        event_id="evt-1",
        commence_time="2024-02-01T18:00:00Z",
        as_of_date="2024-01-15T12:00:00Z",
    )

    assert all(np.isnan(result[column]) for column in result)


def test_method_odds_eventless_record_is_reachable_with_or_without_event_id(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "method_odds"
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", snapshot_dir)
    method_odds._method_odds_cache.clear()
    fresh_time = _fresh_snapshot_time()

    _write_method_snapshot(
        snapshot_dir,
        fresh_time,
        {
            "schema_version": 1,
            "snapshot_time": fresh_time,
            "status": "success",
            "record_count": 1,
            "sources": [],
            "records": [
                {
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "fighter_a_norm": "alpha fighter",
                    "fighter_b_norm": "beta fighter",
                    "event_id": "",
                    "commence_time": "2024-02-01T18:00:00Z",
                    "event_title": "BFO-only Event",
                    "source": "bestfightodds",
                    "captured_at": "2024-01-01T10:00:00",
                    "method_odds": {
                        "a_ko_odds_prob": 0.41,
                        "a_sub_odds_prob": None,
                        "a_dec_odds_prob": None,
                        "b_ko_odds_prob": None,
                        "b_sub_odds_prob": None,
                        "b_dec_odds_prob": 0.29,
                    },
                }
            ],
        },
    )

    without_event_id = method_odds.get_method_odds(
        "Alpha Fighter",
        "Beta Fighter",
        commence_time="2024-02-01T18:00:00Z",
    )
    method_odds._method_odds_cache.clear()
    with_event_id = method_odds.get_method_odds(
        "Alpha Fighter",
        "Beta Fighter",
        event_id="evt-live",
        commence_time="2024-02-01T18:00:00Z",
    )

    assert without_event_id["a_ko_odds_prob"] == pytest.approx(0.41)
    assert without_event_id["b_dec_odds_prob"] == pytest.approx(0.29)
    assert with_event_id["a_ko_odds_prob"] == pytest.approx(0.41)
    assert with_event_id["b_dec_odds_prob"] == pytest.approx(0.29)


def test_save_odds_snapshot_records_opening_line_and_first_snapshot_features_are_nan(tmp_path, monkeypatch):
    _patch_line_history_dir(monkeypatch, tmp_path)

    df = pd.DataFrame(
        [
            {
                "event_id": "evt-1",
                "commence_time": "2024-02-01T18:00:00Z",
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "bookmaker": "Book A",
                "a_odds": 1.80,
                "b_odds": 2.10,
                "a_implied_prob": 0.56,
                "b_implied_prob": 0.44,
                "a_fair_prob": 0.55,
                "b_fair_prob": 0.45,
            }
        ]
    )

    line_tracker.save_odds_snapshot(df, snapshot_time="2024-01-01T10:00:00")
    features = line_tracker.get_line_movement_features("Alpha Fighter", "Beta Fighter", event_id="evt-1")
    opening_lines = line_tracker.load_opening_lines()

    assert "event::evt-1" in opening_lines
    assert opening_lines["event::evt-1"]["opening_prob_a"] == pytest.approx(0.55)
    assert np.isnan(features["line_movement"])
    assert np.isnan(features["line_abs_movement"])
    assert np.isnan(features["line_steam_move"])


def test_injury_detector_treats_extreme_line_move_as_advisory_warning(tmp_path, monkeypatch):
    _patch_line_history_dir(monkeypatch, tmp_path)
    alert = line_tracker.detect_injury_or_cancellation(
        "Bryce Mitchell",
        "Said Nurmagomedov",
        current_odds={"a_prob": 0.43, "b_prob": 0.57},
        analysis={
            "opening_prob_a": 0.62,
            "current_prob_a": 0.43,
            "movement": -0.19,
            "direction": "toward_b",
            "steam_move": True,
        },
    )

    assert alert["suspected"] is True
    assert alert["severity"] == "warning"
    assert "not blocked" in alert["reason"]
    assert "Betting is blocked" not in alert["reason"]


def test_injury_detector_treats_near_zero_price_as_advisory_warning(tmp_path, monkeypatch):
    _patch_line_history_dir(monkeypatch, tmp_path)
    alert = line_tracker.detect_injury_or_cancellation(
        "Belal Muhammad",
        "Gabriel Bonfim",
        current_odds={"a_prob": 0.04, "b_prob": 0.96},
        analysis={"movement": 0.0, "steam_move": False},
        commence_time="2026-06-07T02:00:00Z",
        now="2026-06-06T22:00:00Z",
    )

    assert alert["suspected"] is True
    assert alert["severity"] == "warning"
    assert "not blocked" in alert["reason"]
    assert "Betting is blocked" not in alert["reason"]


def test_line_move_warning_stays_deduped_after_transient_clear_and_restart(
    tmp_path,
    monkeypatch,
    caplog,
):
    history_dir = tmp_path / "line_history"
    monkeypatch.setattr(line_tracker, "LINE_HISTORY_DIR", history_dir)
    monkeypatch.setattr(line_tracker, "_alert_log_state", {})
    monkeypatch.setattr(line_tracker, "_alert_log_state_path_loaded", None)
    caplog.set_level(logging.DEBUG, logger=line_tracker.logger.name)

    alert_kwargs = {
        "event_id": "event-123",
        "commence_time": "2026-07-19T00:15:00Z",
    }
    moved = {
        "opening_prob_a": 0.59,
        "current_prob_a": 0.40,
        "movement": -0.19,
        "direction": "toward_b",
        "steam_move": False,
    }
    stable = {
        "opening_prob_a": 0.59,
        "current_prob_a": 0.59,
        "movement": 0.0,
        "direction": "stable",
        "steam_move": False,
    }

    line_tracker.detect_injury_or_cancellation(
        "Alberto Montes", "Tommy McMillen", analysis=moved, **alert_kwargs
    )
    line_tracker.detect_injury_or_cancellation(
        "Alberto Montes", "Tommy McMillen", analysis=stable, **alert_kwargs
    )
    line_tracker.detect_injury_or_cancellation(
        "Alberto Montes", "Tommy McMillen", analysis=moved, **alert_kwargs
    )

    # Simulate a fresh process; the next call must reload durable state.
    line_tracker._alert_log_state = {}
    line_tracker._alert_log_state_path_loaded = None
    line_tracker.detect_injury_or_cancellation(
        "Alberto Montes", "Tommy McMillen", analysis=moved, **alert_kwargs
    )

    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and "LINE MOVE ALERT" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert (history_dir / line_tracker.ALERT_LOG_STATE_FILENAME).exists()


def test_line_move_warning_realerts_for_larger_move_or_different_event(
    tmp_path,
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(line_tracker, "LINE_HISTORY_DIR", tmp_path / "line_history")
    monkeypatch.setattr(line_tracker, "_alert_log_state", {})
    monkeypatch.setattr(line_tracker, "_alert_log_state_path_loaded", None)
    caplog.set_level(logging.DEBUG, logger=line_tracker.logger.name)

    def _detect(movement, event_id):
        line_tracker.detect_injury_or_cancellation(
            "Alpha Fighter",
            "Beta Fighter",
            analysis={
                "movement": -movement,
                "direction": "toward_b",
                "steam_move": False,
            },
            event_id=event_id,
            commence_time="2026-07-19T00:15:00Z",
        )

    _detect(0.19, "event-1")
    _detect(0.22, "event-1")
    _detect(0.25, "event-1")
    _detect(0.19, "event-2")

    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and "LINE MOVE ALERT" in record.getMessage()
    ]
    assert len(warnings) == 3


def test_load_line_history_reorients_reversed_rows_and_uses_event_id(tmp_path, monkeypatch):
    _patch_line_history_dir(monkeypatch, tmp_path)

    line_tracker.save_odds_snapshot(
        pd.DataFrame(
            [
                {
                    "event_id": "evt-1",
                    "commence_time": "2024-02-01T18:00:00Z",
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "bookmaker": "Book A",
                    "a_odds": 1.80,
                    "b_odds": 2.10,
                    "a_implied_prob": 0.56,
                    "b_implied_prob": 0.44,
                    "a_fair_prob": 0.55,
                    "b_fair_prob": 0.45,
                }
            ]
        ),
        snapshot_time="2024-01-01T10:00:00",
    )
    line_tracker.save_odds_snapshot(
        pd.DataFrame(
            [
                {
                    "event_id": "evt-1",
                    "commence_time": "2024-02-01T18:00:00Z",
                    "fighter_a": "Beta Fighter",
                    "fighter_b": "Alpha Fighter",
                    "bookmaker": "Book A",
                    "a_odds": 1.60,
                    "b_odds": 2.40,
                    "a_implied_prob": 0.62,
                    "b_implied_prob": 0.38,
                    "a_fair_prob": 0.60,
                    "b_fair_prob": 0.40,
                }
            ]
        ),
        snapshot_time="2024-01-01T11:00:00",
    )

    history = line_tracker.load_line_history("Alpha Fighter", "Beta Fighter", event_id="evt-1")

    assert list(history["a_fair_prob"]) == pytest.approx([0.55, 0.40])
    assert list(history["b_fair_prob"]) == pytest.approx([0.45, 0.60])


def test_load_line_history_without_event_id_prefers_latest_fight_key(tmp_path, monkeypatch):
    _patch_line_history_dir(monkeypatch, tmp_path)

    line_tracker.save_odds_snapshot(
        pd.DataFrame(
            [
                {
                    "event_id": "evt-1",
                    "commence_time": "2024-02-01T18:00:00Z",
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "bookmaker": "Book A",
                    "a_odds": 1.80,
                    "b_odds": 2.10,
                    "a_implied_prob": 0.56,
                    "b_implied_prob": 0.44,
                    "a_fair_prob": 0.55,
                    "b_fair_prob": 0.45,
                }
            ]
        ),
        snapshot_time="2024-01-01T10:00:00",
    )
    line_tracker.save_odds_snapshot(
        pd.DataFrame(
            [
                {
                    "event_id": "evt-2",
                    "commence_time": "2024-04-01T18:00:00Z",
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "bookmaker": "Book A",
                    "a_odds": 1.50,
                    "b_odds": 2.60,
                    "a_implied_prob": 0.66,
                    "b_implied_prob": 0.34,
                    "a_fair_prob": 0.64,
                    "b_fair_prob": 0.36,
                }
            ]
        ),
        snapshot_time="2024-03-25T10:00:00",
    )

    history = line_tracker.load_line_history("Alpha Fighter", "Beta Fighter")

    assert history["event_id"].nunique() == 1
    assert history["event_id"].iloc[0] == "evt-2"
    assert history["a_fair_prob"].iloc[0] == pytest.approx(0.64)


def test_line_movement_as_of_date_filters_future_snapshots(tmp_path, monkeypatch):
    _patch_line_history_dir(monkeypatch, tmp_path)

    first_snapshot = pd.DataFrame(
        [
            {
                "event_id": "evt-1",
                "commence_time": "2024-02-01T18:00:00Z",
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "bookmaker": "Book A",
                "a_odds": 1.80,
                "b_odds": 2.10,
                "a_implied_prob": 0.56,
                "b_implied_prob": 0.44,
                "a_fair_prob": 0.55,
                "b_fair_prob": 0.45,
            }
        ]
    )
    second_snapshot = pd.DataFrame(
        [
            {
                "event_id": "evt-1",
                "commence_time": "2024-02-01T18:00:00Z",
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "bookmaker": "Book A",
                "a_odds": 1.60,
                "b_odds": 2.40,
                "a_implied_prob": 0.62,
                "b_implied_prob": 0.38,
                "a_fair_prob": 0.60,
                "b_fair_prob": 0.40,
            }
        ]
    )

    line_tracker.save_odds_snapshot(first_snapshot, snapshot_time="2024-01-01T10:00:00")
    line_tracker.save_odds_snapshot(second_snapshot, snapshot_time="2024-01-02T10:00:00")

    before_second = line_tracker.get_line_movement_features(
        "Alpha Fighter",
        "Beta Fighter",
        event_id="evt-1",
        as_of_date="2024-01-01",
    )
    after_second = line_tracker.get_line_movement_features(
        "Alpha Fighter",
        "Beta Fighter",
        event_id="evt-1",
        as_of_date="2024-01-02",
    )

    assert np.isnan(before_second["line_movement"])
    assert after_second["line_movement"] == pytest.approx(0.05)
    assert after_second["line_abs_movement"] == pytest.approx(0.05)


def test_run_monitoring_pass_collects_snapshot_metadata(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    monkeypatch.setattr(live_monitor, "SNAPSHOTS_DIR", snapshot_dir)
    monkeypatch.setattr(live_monitor, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(live_monitor, "scrape_upcoming_events", lambda: [])
    monkeypatch.setattr(
        live_monitor,
        "collect_rankings_snapshot",
        lambda: {
            "status": "success",
            "source": "ufc.com",
            "snapshot_time": "2024-01-01T10:00:00",
            "snapshot_path": "rankings.json",
        },
        raising=False,
    )
    monkeypatch.setattr(
        live_monitor,
        "collect_method_odds_snapshot",
        lambda tracked_fights=None: {
            "status": "failed",
            "record_count": 0,
            "snapshot_time": "2024-01-01T10:00:00",
            "snapshot_path": "method_odds.json",
            "latest_usable_snapshot": {
                "snapshot_time": "2024-01-01T09:00:00",
                "snapshot_path": "method_odds_previous.json",
                "record_count": 1,
                "is_stale": False,
            },
        },
        raising=False,
    )

    # Patch the imported functions inside run_monitoring_pass via sys.modules lookup path.
    import src.data.rankings_scraper as rankings_module
    import src.data.method_odds as method_module

    monkeypatch.setattr(rankings_module, "collect_rankings_snapshot", live_monitor.collect_rankings_snapshot)
    monkeypatch.setattr(method_module, "collect_method_odds_snapshot", live_monitor.collect_method_odds_snapshot)

    signals = live_monitor.run_monitoring_pass()

    assert signals["rankings_snapshot"]["status"] == "success"
    assert signals["rankings_snapshot"]["source"] == "ufc.com"
    assert signals["method_odds_snapshot"]["status"] == "failed"
    assert signals["method_odds_snapshot"]["record_count"] == 0
    assert signals["method_odds_snapshot"]["latest_usable_snapshot"]["record_count"] == 1


@pytest.mark.parametrize(
    ("tracked_fight_count", "expects_recovery"),
    [(0, False), (1, True)],
)
def test_run_monitoring_pass_only_recovers_method_odds_for_tracked_fights(
    tmp_path,
    monkeypatch,
    caplog,
    tracked_fight_count,
    expects_recovery,
):
    monkeypatch.setattr(live_monitor, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(live_monitor, "collect_upcoming_event_cards", lambda: [])

    import src.data.rankings_scraper as rankings_module
    import src.data.method_odds as method_module

    monkeypatch.setattr(
        rankings_module,
        "collect_rankings_snapshot",
        lambda: {
            "status": "success",
            "source": "ufc.com",
            "snapshot_time": "2024-01-01T10:00:00",
            "snapshot_path": "rankings.json",
        },
    )
    monkeypatch.setattr(
        method_module,
        "collect_method_odds_snapshot",
        lambda tracked_fights=None: {
            "status": "success",
            "record_count": tracked_fight_count,
            "tracked_fight_count": tracked_fight_count,
            "covered_fight_count": tracked_fight_count,
            "snapshot_time": "2024-01-01T10:00:00",
            "snapshot_path": "method_odds.json",
        },
    )

    caplog.set_level(logging.INFO, logger=live_monitor.logger.name)
    live_monitor.run_monitoring_pass()

    snapshot_logs = [
        record
        for record in caplog.records
        if record.getMessage().startswith("Method-odds snapshot: success")
    ]
    assert len(snapshot_logs) == 1
    recovered_keys = getattr(
        snapshot_logs[0],
        "alert_recovered_incident_keys",
        [],
    )
    assert bool(recovered_keys) is expects_recovery
    if expects_recovery:
        assert recovered_keys == [live_monitor._METHOD_ODDS_ALERT_INCIDENT_KEY]


def test_run_line_tracking_pass_reports_progress(monkeypatch):
    odds_df = pd.DataFrame(
        [
            {
                "fight_key": "fight-1",
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "event_id": "evt-1",
                "commence_time": "2024-01-01T10:00:00Z",
            },
            {
                "fight_key": "fight-2",
                "fighter_a": "Gamma Fighter",
                "fighter_b": "Delta Fighter",
                "event_id": "evt-2",
                "commence_time": "2024-01-02T10:00:00Z",
            },
        ]
    )
    progress_messages = []

    monkeypatch.setattr(line_tracker, "snapshot_odds", lambda: odds_df)
    monkeypatch.setattr(line_tracker, "snapshot_polymarket_prices", lambda: pd.DataFrame())
    monkeypatch.setattr(
        line_tracker,
        "line_history_health",
        lambda _df: {
            "tracked_fights": 2,
            "with_opening_line": 2,
            "with_two_snapshots": 2,
        },
    )
    monkeypatch.setattr(
        line_tracker,
        "analyze_line_movement",
        lambda *_args, **_kwargs: {"is_sharp_move": False, "steam_move": False},
    )

    summary = line_tracker.run_line_tracking_pass(progress_callback=progress_messages.append)

    assert summary["fights_analyzed"] == 2
    assert progress_messages == [
        "Line tracking: snapshotting bookmaker odds and Polymarket prices",
        "Line tracking: analyzing 1/2 (Alpha Fighter vs Beta Fighter)",
        "Line tracking: analyzing 2/2 (Gamma Fighter vs Delta Fighter)",
    ]


def test_attach_event_identity_uses_live_odds_context(monkeypatch):
    class _FakeOddsClient:
        def get_live_odds(self):
            return {"fake": True}

        def odds_to_dataframe(self, _odds):
            return pd.DataFrame(
                [
                    {
                        "event_id": "evt-1",
                        "commence_time": "2024-02-01T18:00:00Z",
                        "fighter_a": "Alpha Fighter",
                        "fighter_b": "Beta Fighter",
                        "bookmaker": "Book",
                        "a_odds": 1.8,
                        "b_odds": 2.1,
                        "a_implied_prob": 0.56,
                        "b_implied_prob": 0.44,
                        "a_fair_prob": 0.55,
                        "b_fair_prob": 0.45,
                    }
                ]
            )

        def get_consensus_odds(self, odds_df):
            return odds_df[["event_id", "commence_time", "fighter_a", "fighter_b"]].drop_duplicates().copy()

    import src.data.odds_client as odds_module

    monkeypatch.setattr(odds_module, "OddsClient", _FakeOddsClient)

    tracked = live_monitor._attach_event_identity(
        [
            {
                "event_title": "UFC Test",
                "event_date": "2024-02-01",
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "weight_class": "Lightweight",
            }
        ]
    )

    assert tracked[0]["event_id"] == "evt-1"
    assert tracked[0]["commence_time"] == "2024-02-01T18:00:00Z"


def test_detect_short_notice_requires_one_for_one_replacement(caplog):
    previous_card = [
        {"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter"},
        {"fighter_a": "Stable Red", "fighter_b": "Stable Blue"},
    ]
    current_card = [
        {"fighter_a": "Alpha Fighter", "fighter_b": "Gamma Fighter"},
        {"fighter_a": "Stable Red", "fighter_b": "Stable Blue"},
    ]

    with caplog.at_level(logging.WARNING, logger=live_monitor.logger.name):
        replacements = live_monitor.detect_short_notice(
            current_card=current_card,
            previous_card=previous_card,
            days_to_event=2,
        )

    assert replacements == [
        {
            "new_fighter": "Gamma Fighter",
            "replaced_fighter": "Beta Fighter",
            "days_until_event_at_detection": 2,
            "is_short_notice": True,
        }
    ]
    assert (
        "SHORT NOTICE: Gamma Fighter replacing Beta Fighter; "
        "late-detected with 2 days until event"
    ) in caplog.text


def test_detect_short_notice_ignores_new_fighters_without_removed_opponent_match(caplog):
    previous_card = [
        {"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter"},
    ]
    current_card = [
        {"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter"},
        {"fighter_a": "Cameron Smotherman", "fighter_b": "Kai Asakura"},
    ]

    with caplog.at_level(logging.WARNING, logger=live_monitor.logger.name):
        replacements = live_monitor.detect_short_notice(
            current_card=current_card,
            previous_card=previous_card,
            days_to_event=0,
        )

    assert replacements == []
    assert "SHORT NOTICE" not in caplog.text


def _patch_monitoring_pass_dependencies(monkeypatch, event_card):
    monkeypatch.setattr(
        live_monitor,
        "collect_upcoming_event_cards",
        lambda: [event_card],
    )
    monkeypatch.setattr(live_monitor, "_attach_event_identity", lambda fights: fights)
    monkeypatch.setattr(
        rankings_scraper,
        "collect_rankings_snapshot",
        lambda: {
            "status": "success",
            "source": "ufc.com",
            "snapshot_time": "2026-07-29T10:00:00",
            "snapshot_path": "rankings.json",
        },
    )
    monkeypatch.setattr(
        method_odds,
        "collect_method_odds_snapshot",
        lambda tracked_fights=None: {
            "status": "success",
            "record_count": len(tracked_fights or []),
            "tracked_fight_count": len(tracked_fights or []),
            "snapshot_time": "2026-07-29T10:00:00",
            "snapshot_path": "method_odds.json",
        },
    )


def test_monitoring_pass_carries_short_notice_replacement_without_rewarning(
    tmp_path,
    monkeypatch,
    caplog,
):
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    monkeypatch.setattr(live_monitor, "SNAPSHOTS_DIR", snapshot_dir)

    event_title = "UFC Persistence Test"
    event_date = (
        pd.Timestamp.now().normalize() + pd.Timedelta(days=4)
    ).strftime("%Y-%m-%d")
    previous_card = [
        {"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter"},
    ]
    current_card = [
        {"fighter_a": "Alpha Fighter", "fighter_b": "Gamma Fighter"},
    ]
    live_monitor.save_card_snapshot(
        event_title,
        previous_card,
        event_date=event_date,
    )
    _patch_monitoring_pass_dependencies(
        monkeypatch,
        {
            "event": {
                "title": event_title,
                "date": event_date,
                "url": "https://example.test/event",
            },
            "fights": current_card,
        },
    )

    with caplog.at_level(logging.WARNING, logger=live_monitor.logger.name):
        first_pass = live_monitor.run_monitoring_pass()
        second_pass = live_monitor.run_monitoring_pass()

    first_replacements = first_pass["short_notice_replacements"]
    assert len(first_replacements) == 1
    assert second_pass["short_notice_replacements"] == first_replacements
    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("SHORT NOTICE:")
    ]
    assert len(warning_messages) == 1

    latest = live_monitor._latest_card_snapshot_payload(event_title)
    assert latest is not None
    assert latest[1]["short_notice_replacements"] == first_replacements
    assert len(list(snapshot_dir.glob("UFC_Persistence_Test_*.json"))) == 2


def test_monitoring_pass_recovers_short_notice_state_from_legacy_history(
    tmp_path,
    monkeypatch,
    caplog,
):
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    monkeypatch.setattr(live_monitor, "SNAPSHOTS_DIR", snapshot_dir)

    event_title = "UFC Legacy Persistence Test"
    event_date = (
        pd.Timestamp.now().normalize() + pd.Timedelta(days=4)
    ).strftime("%Y-%m-%d")
    previous_card = [
        {"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter"},
    ]
    current_card = [
        {"fighter_a": "Alpha Fighter", "fighter_b": "Gamma Fighter"},
    ]
    captured_times = [
        datetime.now() - timedelta(hours=2),
        datetime.now() - timedelta(hours=1),
    ]
    for captured_at, card in zip(captured_times, (previous_card, current_card)):
        path = snapshot_dir / (
            "UFC_Legacy_Persistence_Test_"
            f"{captured_at.strftime('%Y%m%d_%H%M%S')}.json"
        )
        path.write_text(
            json.dumps(
                {
                    "event": event_title,
                    "event_date": event_date,
                    "timestamp": captured_at.isoformat(),
                    "fights": card,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    _patch_monitoring_pass_dependencies(
        monkeypatch,
        {
            "event": {
                "title": event_title,
                "date": event_date,
                "url": "https://example.test/event",
            },
            "fights": current_card,
        },
    )

    with caplog.at_level(logging.WARNING, logger=live_monitor.logger.name):
        signals = live_monitor.run_monitoring_pass()

    replacements = signals["short_notice_replacements"]
    assert len(replacements) == 1
    assert replacements[0]["new_fighter"] == "Gamma Fighter"
    assert replacements[0]["replaced_fighter"] == "Beta Fighter"
    assert not any(
        record.getMessage().startswith("SHORT NOTICE:")
        for record in caplog.records
    )

    latest = live_monitor._latest_card_snapshot_payload(event_title)
    assert latest is not None
    assert latest[1]["short_notice_replacements"] == replacements


def test_snapshot_identity_separates_same_title_on_different_dates(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(live_monitor, "SNAPSHOTS_DIR", tmp_path)
    title = "UFC Reused Title"
    first_card = [{"fighter_a": "First Alpha", "fighter_b": "First Beta"}]
    second_card = [{"fighter_a": "Second Alpha", "fighter_b": "Second Beta"}]

    live_monitor.save_card_snapshot(title, first_card, event_date="2026-08-01")
    live_monitor.save_card_snapshot(title, second_card, event_date="2026-08-08")

    assert live_monitor.load_latest_snapshot(
        title,
        event_date="2026-08-01",
    ) == first_card
    assert live_monitor.load_latest_snapshot(
        title,
        event_date="2026-08-08",
    ) == second_card


def test_stable_event_url_preserves_state_and_history_order_across_title_rename(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(live_monitor, "SNAPSHOTS_DIR", tmp_path)
    event_url = "https://www.ufc.com/event/ufc-fight-night-august-01-2026"
    event_date = "2026-08-01"
    old_event = {
        "title": "Zulu Original Title",
        "date": event_date,
        "url": event_url,
    }
    new_event = {
        "title": "Alpha Renamed Title",
        "date": event_date,
        "url": "https://ufc.com/event/ufc-fight-night-august-01-2026/",
    }
    current_card = [
        {"fighter_a": "Stable Fighter", "fighter_b": "Replacement Fighter"},
    ]
    replacement = {
        "new_fighter": "Replacement Fighter",
        "replaced_fighter": "Original Fighter",
        "days_until_event_at_detection": 4,
        "is_short_notice": True,
    }
    live_monitor.save_card_snapshot(
        old_event["title"],
        current_card,
        event_date=event_date,
        event_url=event_url,
        short_notice_replacements=[replacement],
    )

    carried = live_monitor._update_short_notice_event_state(
        new_event,
        current_card,
        3,
    )

    assert len(carried) == 1
    assert carried[0]["event_key"] == live_monitor.event_identity_key(old_event)
    assert carried[0]["event_title"] == new_event["title"]
    history = live_monitor._card_snapshot_history(
        new_event["title"],
        event_date=event_date,
        event_url=event_url,
    )
    assert [payload["event"] for _, payload in history] == [
        old_event["title"],
        new_event["title"],
    ]


def test_active_replacement_refreshes_accented_display_and_legacy_countdown():
    event = {
        "title": "UFC Accent Test",
        "date": "2026-08-01",
        "url": "https://www.ufc.com/event/ufc-accent-test",
    }
    current_card = [
        {"fighter_a": "Stable Fighter", "fighter_b": "Borislav Nikolić"},
    ]
    legacy_state = [
        {
            "new_fighter": "Borislav Nikolic",
            "replaced_fighter": "Josias Musasa",
            "days_notice": 3,
            "is_short_notice": True,
        },
    ]

    merged = live_monitor._merge_active_short_notice_replacements(
        current_card,
        legacy_state,
        event_identity=event,
    )

    assert merged[0]["new_fighter"] == "Borislav Nikolić"
    assert merged[0]["new_fighter_key"] == "borislav nikolic"
    assert merged[0]["days_until_event_at_detection"] == 3
    assert "days_notice" not in merged[0]
    assert merged[0]["event_key"] == live_monitor.event_identity_key(event)


def test_malformed_latest_state_recovers_from_trustworthy_history(
    tmp_path,
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(live_monitor, "SNAPSHOTS_DIR", tmp_path)
    event = {
        "title": "UFC Malformed State Test",
        "date": "2026-08-01",
        "url": "https://www.ufc.com/event/ufc-malformed-state-test",
    }
    previous_card = [
        {"fighter_a": "Stable Fighter", "fighter_b": "Original Fighter"},
    ]
    current_card = [
        {"fighter_a": "Stable Fighter", "fighter_b": "Replacement Fighter"},
    ]
    live_monitor.save_card_snapshot(
        event["title"],
        previous_card,
        event_date=event["date"],
        event_url=event["url"],
        short_notice_replacements=[],
    )
    detected = live_monitor.detect_short_notice(
        current_card,
        previous_card,
        3,
        emit_warning=False,
        event_identity=event,
    )
    live_monitor.save_card_snapshot(
        event["title"],
        current_card,
        event_date=event["date"],
        event_url=event["url"],
        short_notice_replacements=detected,
    )
    latest = live_monitor._latest_card_snapshot_payload(
        event["title"],
        event_date=event["date"],
        event_url=event["url"],
    )
    assert latest is not None
    malformed_payload = dict(latest[1])
    malformed_payload["timestamp"] = datetime.now().isoformat()
    malformed_payload["short_notice_replacements"] = {"corrupt": True}
    malformed_path = tmp_path / "malformed_latest.json"
    malformed_path.write_text(
        json.dumps(malformed_payload, indent=2),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger=live_monitor.logger.name):
        recovered = live_monitor._update_short_notice_event_state(
            event,
            current_card,
            2,
        )

    assert len(recovered) == 1
    assert recovered[0]["new_fighter"] == "Replacement Fighter"
    assert "Quarantining malformed short-notice state" in caplog.text
    repaired_latest = live_monitor._latest_card_snapshot_payload(
        event["title"],
        event_date=event["date"],
        event_url=event["url"],
    )
    assert repaired_latest is not None
    assert repaired_latest[1]["short_notice_replacements"] == recovered


def test_malformed_latest_state_with_unusable_chronology_is_not_overwritten(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(live_monitor, "SNAPSHOTS_DIR", tmp_path)
    event = {
        "title": "UFC Unusable Chronology Test",
        "date": "2026-08-01",
        "url": "https://www.ufc.com/event/ufc-unusable-chronology-test",
    }
    previous_card = [
        {"fighter_a": "Stable Fighter", "fighter_b": "Original Fighter"},
    ]
    current_card = [
        {"fighter_a": "Stable Fighter", "fighter_b": "Replacement Fighter"},
    ]
    live_monitor.save_card_snapshot(
        event["title"],
        previous_card,
        event_date=event["date"],
        event_url=event["url"],
        short_notice_replacements=[],
    )
    malformed_payload = {
        "event": event["title"],
        "event_date": "not-a-date",
        "event_url": event["url"],
        "event_key": live_monitor.event_identity_key(event),
        "timestamp": "not-a-timestamp",
        "fights": current_card,
        "short_notice_replacements": {"corrupt": True},
    }
    malformed_path = tmp_path / "unusable_chronology_latest.json"
    malformed_path.write_text(
        json.dumps(malformed_payload, indent=2),
        encoding="utf-8",
    )
    paths_before = set(tmp_path.glob("*.json"))

    recovered = live_monitor._update_short_notice_event_state(
        event,
        current_card,
        2,
    )

    assert recovered == []
    assert set(tmp_path.glob("*.json")) == paths_before
    latest = live_monitor._latest_card_snapshot_payload(
        event["title"],
        event_date=event["date"],
        event_url=event["url"],
    )
    assert latest is not None
    assert latest[0] == malformed_path
    assert latest[1]["short_notice_replacements"] == {"corrupt": True}


def test_malformed_latest_state_with_non_list_fights_is_not_overwritten(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(live_monitor, "SNAPSHOTS_DIR", tmp_path)
    event = {
        "title": "UFC Non-List Card Test",
        "date": "2026-08-01",
        "url": "https://www.ufc.com/event/ufc-non-list-card-test",
    }
    previous_card = [
        {"fighter_a": "Stable Fighter", "fighter_b": "Original Fighter"},
    ]
    current_card = [
        {"fighter_a": "Stable Fighter", "fighter_b": "Replacement Fighter"},
    ]
    live_monitor.save_card_snapshot(
        event["title"],
        previous_card,
        event_date=event["date"],
        event_url=event["url"],
        short_notice_replacements=[],
    )
    malformed_payload = {
        "event": event["title"],
        "event_date": event["date"],
        "event_url": event["url"],
        "event_key": live_monitor.event_identity_key(event),
        "timestamp": datetime.now().isoformat(),
        "fights": {"corrupt": True},
        "short_notice_replacements": {"corrupt": True},
    }
    malformed_path = tmp_path / "non_list_card_latest.json"
    malformed_path.write_text(
        json.dumps(malformed_payload, indent=2),
        encoding="utf-8",
    )
    paths_before = set(tmp_path.glob("*.json"))

    recovered = live_monitor._update_short_notice_event_state(
        event,
        current_card,
        2,
    )

    assert recovered == []
    assert set(tmp_path.glob("*.json")) == paths_before
    latest = live_monitor._latest_card_snapshot_payload(
        event["title"],
        event_date=event["date"],
        event_url=event["url"],
    )
    assert latest is not None
    assert latest[0] == malformed_path
    assert latest[1]["fights"] == {"corrupt": True}


def test_short_notice_event_day_and_calendar_countdowns_are_nonnegative(caplog):
    previous_card = [
        {"fighter_a": "Stable Fighter", "fighter_b": "Original Fighter"},
    ]
    current_card = [
        {"fighter_a": "Stable Fighter", "fighter_b": "Replacement Fighter"},
    ]
    with caplog.at_level(logging.WARNING, logger=live_monitor.logger.name):
        replacements = live_monitor.detect_short_notice(
            current_card,
            previous_card,
            -1,
        )

    assert replacements[0]["days_until_event_at_detection"] == 0
    assert "late-detected with 0 days until event" in caplog.text
    today = datetime.now()
    event_date = (today + timedelta(days=4)).date().isoformat()
    assert live_monitor._snapshot_days_to_event(
        {
            "event_date": event_date,
            "timestamp": today.replace(hour=23, minute=59).isoformat(),
        }
    ) == 4


def test_short_notice_replacement_removal_and_chaining_drops_stale_fighter():
    event = {
        "title": "UFC Chain Test",
        "date": "2026-08-01",
        "url": "https://www.ufc.com/event/ufc-chain-test",
    }
    first_card = [
        {"fighter_a": "Stable Fighter", "fighter_b": "First Replacement"},
    ]
    first_state = live_monitor._merge_active_short_notice_replacements(
        first_card,
        [
            {
                "new_fighter": "First Replacement",
                "replaced_fighter": "Original Fighter",
                "days_until_event_at_detection": 5,
                "is_short_notice": True,
            },
        ],
        event_identity=event,
    )
    assert live_monitor._merge_active_short_notice_replacements(
        [{"fighter_a": "Other Alpha", "fighter_b": "Other Beta"}],
        first_state,
        event_identity=event,
    ) == []

    second_card = [
        {"fighter_a": "Stable Fighter", "fighter_b": "Second Replacement"},
    ]
    detected = live_monitor.detect_short_notice(
        second_card,
        first_card,
        2,
        emit_warning=False,
        event_identity=event,
    )
    chained = live_monitor._merge_active_short_notice_replacements(
        second_card,
        first_state,
        detected,
        event_identity=event,
    )

    assert len(chained) == 1
    assert chained[0]["new_fighter"] == "Second Replacement"
    assert chained[0]["replaced_fighter"] == "First Replacement"


def test_monitoring_pass_uses_calendar_days_for_event_countdown(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(live_monitor, "SNAPSHOTS_DIR", tmp_path)
    event_date = (
        pd.Timestamp.now().normalize() + pd.Timedelta(days=4)
    ).strftime("%Y-%m-%d")
    event_card = {
        "event": {
            "title": "UFC Calendar Test",
            "date": event_date,
            "url": "https://www.ufc.com/event/ufc-calendar-test",
        },
        "fights": [
            {"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter"},
        ],
    }
    _patch_monitoring_pass_dependencies(monkeypatch, event_card)

    signals = live_monitor.run_monitoring_pass()

    assert signals["events"][0]["days_to_event"] == 4


def test_scrape_event_card_extracts_weight_class_from_nonblank_row_text(monkeypatch):
    class _FakeResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            return None

    html = """
    <table>
      <tr class="b-fight-details__table-row">
        <td class="b-fight-details__table-col_style_align-center"></td>
        <td>
          <a class="b-link">Alpha Fighter</a>
          <a class="b-link">Beta Fighter</a>
        </td>
        <td class="b-fight-details__table-col_style_align-center">
          <p class="b-fight-details__table-text">Women's Flyweight Title Bout</p>
        </td>
        <td><img src="/images/belt.png" /></td>
      </tr>
    </table>
    """

    monkeypatch.setattr(
        live_monitor.requests,
        "get",
        lambda *_args, **_kwargs: _FakeResponse(html),
    )

    fights = live_monitor.scrape_event_card("https://example.test/event")

    assert len(fights) == 1
    assert fights[0]["fighter_a"] == "Alpha Fighter"
    assert fights[0]["fighter_b"] == "Beta Fighter"
    assert fights[0]["weight_class"] == "Women's Flyweight"
    assert fights[0]["is_title_bout"] is True
    assert fights[0]["num_rounds"] == 5


def test_scrape_event_card_extracts_event_location(monkeypatch):
    class _FakeResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            return None

    html = """
    <ul>
      <li class="b-list__box-list-item">Location: Las Vegas, Nevada, USA</li>
    </ul>
    <table>
      <tr class="b-fight-details__table-row">
        <td><a class="b-link">Alpha Fighter</a><a class="b-link">Beta Fighter</a></td>
        <td><p class="b-fight-details__table-text">Lightweight Bout</p></td>
      </tr>
    </table>
    """

    monkeypatch.setattr(
        live_monitor.requests,
        "get",
        lambda *_args, **_kwargs: _FakeResponse(html),
    )

    fights = live_monitor.scrape_event_card("https://example.test/event")

    assert len(fights) == 1
    assert fights[0]["location"] == "Las Vegas, Nevada, USA"


def test_scrape_upcoming_events_uses_ufc_com_schedule(monkeypatch):
    class _FakeResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            return None

    ufc_com_events = """
    <article class="c-card-event--result">
      <a href="/event/ufc-fight-night-may-30-2026">View Event Details</a>
      <div>Alpha vs Beta</div>
      <div>Sat, May 30 / 7:00 AM EDT / Main Card</div>
      <div>Galaxy Arena</div>
      <div>Macao</div>
      <div>How to Watch</div>
    </article>
    <article class="c-card-event--result">
      <a href="/event/ufc-fight-night-may-16-2026">View Event Details</a>
      <div>Old vs Replay</div>
      <div>Sat, May 16 / 8:00 PM EDT / Main Card</div>
      <div>Watch Replay</div>
    </article>
    """

    def _fake_get(url, *_args, **_kwargs):
        if url == live_monitor.UFC_COM_EVENTS_URL:
            return _FakeResponse(ufc_com_events)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(live_monitor.requests, "get", _fake_get)

    events = live_monitor.scrape_upcoming_events()

    assert events == [
        {
            "title": "UFC Fight Night: Alpha vs Beta",
            "url": "https://www.ufc.com/event/ufc-fight-night-may-30-2026",
            "date": "May 30, 2026",
            "location": "Galaxy Arena, Macao",
            "source": "ufc.com",
        }
    ]


def test_scrape_ufc_com_events_can_include_completed_cards(monkeypatch):
    class _FakeResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            return None

    ufc_com_events = """
    <article class="c-card-event--result">
      <a href="/event/ufc-fight-night-june-14-2026">View Event Details</a>
      <div>Future vs Card</div>
      <div>Sun, Jun 14 / 7:00 PM EDT / Main Card</div>
      <div>Washington</div>
      <div>DC</div>
    </article>
    <article class="c-card-event--result">
      <a href="/event/ufc-fight-night-june-06-2026">View Event Details</a>
      <div>Completed vs Card</div>
      <div>Sat, Jun 6 / 8:00 PM EDT / Main Card</div>
      <div>Watch Replay</div>
    </article>
    """

    def _fake_get(url, *_args, **_kwargs):
        if url == live_monitor.UFC_COM_EVENTS_URL:
            return _FakeResponse(ufc_com_events)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(live_monitor.requests, "get", _fake_get)

    events = live_monitor.scrape_ufc_com_events(include_completed=True)

    assert events == [
        {
            "title": "UFC Fight Night: Future vs Card",
            "url": "https://www.ufc.com/event/ufc-fight-night-june-14-2026",
            "date": "June 14, 2026",
            "location": "Washington, DC",
            "source": "ufc.com",
            "status": "upcoming",
        },
        {
            "title": "UFC Fight Night: Completed vs Card",
            "url": "https://www.ufc.com/event/ufc-fight-night-june-06-2026",
            "date": "June 6, 2026",
            "location": "",
            "source": "ufc.com",
            "status": "completed",
        },
    ]


def test_scrape_upcoming_events_retries_ufc_com_timeout(monkeypatch):
    class _FakeResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            return None

    ufc_com_events = """
    <article class="c-card-event--result">
      <a href="/event/ufc-fight-night-jun-13-2026">View Event Details</a>
      <div>Alpha vs Beta</div>
      <div>Sat, Jun 13 / 8:00 PM EDT / Main Card</div>
      <div>Arena</div>
      <div>City</div>
    </article>
    """
    attempts = {"count": 0}

    def _fake_get(url, *_args, **kwargs):
        assert kwargs["timeout"] == live_monitor.UPSTREAM_FETCH_TIMEOUT_SECONDS
        if url != live_monitor.UFC_COM_EVENTS_URL:
            raise AssertionError(f"unexpected URL: {url}")
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("read timed out")
        return _FakeResponse(ufc_com_events)

    monkeypatch.setattr(live_monitor.requests, "get", _fake_get)
    monkeypatch.setattr(live_monitor.time, "sleep", lambda *_args, **_kwargs: None)

    events = live_monitor.scrape_upcoming_events()

    assert attempts["count"] == 2
    assert events[0]["title"] == "UFC Fight Night: Alpha vs Beta"
    assert events[0]["date"] == "June 13, 2026"


def test_scrape_upcoming_events_falls_back_to_ufcstats_when_ufc_com_has_no_rows(monkeypatch):
    from src.data import ufcstats_http

    class _FakeResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            return None

    ufcstats_events = """
    <table>
      <tr class="b-statistics__table-row">
        <td>
          <a class="b-link" href="http://ufcstats.com/event-details/test-event">UFC Fight Night: Alpha vs. Beta</a>
          <span class="b-statistics__date">May 30, 2026</span>
        </td>
      </tr>
    </table>
    """

    def _fake_get(url, *_args, **_kwargs):
        if url == live_monitor.UFC_COM_EVENTS_URL:
            return _FakeResponse("<html><body>No upcoming cards here</body></html>")
        raise AssertionError(f"unexpected URL: {url}")

    def _fake_request_ufcstats(url, **_kwargs):
        if url == live_monitor.UFCSTATS_UPCOMING_URL:
            return _FakeResponse(ufcstats_events)
        raise AssertionError(f"unexpected UFCStats URL: {url}")

    monkeypatch.setattr(live_monitor.requests, "get", _fake_get)
    monkeypatch.setattr(ufcstats_http, "request_ufcstats", _fake_request_ufcstats)

    events = live_monitor.scrape_upcoming_events()

    assert events == [
        {
            "title": "UFC Fight Night: Alpha vs. Beta",
            "url": "http://ufcstats.com/event-details/test-event",
            "date": "May 30, 2026",
        }
    ]


def test_scrape_event_card_parses_ufc_com_card(monkeypatch):
    class _FakeResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            return None

    html = """
    <div class="field--name-venue">Galaxy Arena, <span>Macao</span></div>
    <div class="c-listing-fight">
      <div class="c-listing-fight__class c-listing-fight__class--desktop">
        <div class="c-listing-fight__class-text">Bantamweight Bout</div>
      </div>
      <div class="c-listing-fight__names-row">
        <div class="c-listing-fight__corner-name c-listing-fight__corner-name--red">
          <a><span>Alpha</span> <span>Fighter</span></a>
        </div>
        <div class="c-listing-fight__corner-name c-listing-fight__corner-name--blue">
          <a>Beta Fighter</a>
        </div>
      </div>
    </div>
    <div class="c-listing-fight">
      <div class="c-listing-fight__class c-listing-fight__class--desktop">
        <div class="c-listing-fight__class-text">Lightweight Bout</div>
      </div>
      <div class="c-listing-fight__names-row">
        <div class="c-listing-fight__corner-name c-listing-fight__corner-name--red">Gamma Fighter</div>
        <div class="c-listing-fight__corner-name c-listing-fight__corner-name--blue">Delta Fighter</div>
      </div>
    </div>
    """

    monkeypatch.setattr(
        live_monitor.requests,
        "get",
        lambda *_args, **_kwargs: _FakeResponse(html),
    )

    fights = live_monitor.scrape_event_card("https://www.ufc.com/event/ufc-fight-night-may-30-2026")

    assert len(fights) == 2
    assert fights[0] == {
        "fighter_a": "Alpha Fighter",
        "fighter_b": "Beta Fighter",
        "weight_class": "Bantamweight Bout",
        "is_main_event": True,
        "is_title_bout": False,
        "num_rounds": 5,
        "location": "Galaxy Arena, Macao",
    }
    assert fights[1]["num_rounds"] == 3


def test_collect_upcoming_fight_contexts_marks_las_vegas_fight_night_as_empty(monkeypatch):
    monkeypatch.setattr(
        live_monitor,
        "scrape_upcoming_events",
        lambda: [
            {
                "title": "UFC Fight Night: Alpha vs. Beta",
                "url": "https://example.test/event",
                "date": "2026-04-01",
            }
        ],
    )
    monkeypatch.setattr(
        live_monitor,
        "scrape_event_card",
        lambda *_args, **_kwargs: [
            {
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "weight_class": "Lightweight",
                "is_main_event": False,
                "is_title_bout": False,
                "num_rounds": 3,
                "location": "Las Vegas, Nevada, USA",
            }
        ],
    )
    monkeypatch.setattr(live_monitor, "_attach_event_identity", lambda tracked: tracked)

    contexts = live_monitor.collect_upcoming_fight_contexts()

    assert len(contexts) == 1
    assert contexts[0]["is_empty_arena"] == pytest.approx(1.0)


def test_collect_upcoming_fight_contexts_recovers_active_card_after_utc_rollover(monkeypatch):
    upcoming_url = "https://example.test/upcoming"
    active_url = "https://example.test/ufc-329"
    monkeypatch.setattr(
        live_monitor,
        "scrape_upcoming_events",
        lambda: [{"title": "Next Card", "url": upcoming_url, "date": "July 18, 2026"}],
    )
    monkeypatch.setattr(
        live_monitor,
        "_scrape_ufc_com_events",
        lambda *, include_completed=False: [
            {
                "title": "UFC 329: McGregor vs Holloway 2",
                "url": active_url,
                "date": "July 11, 2026",
                "status": "completed",
            }
        ],
    )

    def fake_card(url):
        if url == active_url:
            return [
                {
                    "fighter_a": "Benoit Saint Denis",
                    "fighter_b": "Paddy Pimblett",
                    "weight_class": "Lightweight",
                    "location": "Las Vegas, Nevada, USA",
                }
            ]
        return [
            {
                "fighter_a": "Future Fighter",
                "fighter_b": "Future Opponent",
                "weight_class": "Welterweight",
                "location": "Oklahoma City, Oklahoma, USA",
            }
        ]

    monkeypatch.setattr(live_monitor, "scrape_event_card", fake_card)
    monkeypatch.setattr(live_monitor, "_attach_event_identity", lambda tracked: tracked)

    contexts = live_monitor.collect_upcoming_fight_contexts(
        expected_fights=[
            {
                "fighter_a": "Benoit Saint-Denis",
                "fighter_b": "Paddy Pimblett",
                "commence_time": "2026-07-12T02:50:00Z",
            }
        ]
    )

    recovered = [context for context in contexts if context["fighter_b"] == "Paddy Pimblett"]
    assert len(recovered) == 1
    assert recovered[0]["event_date"] == "July 11, 2026"


def test_external_modules_share_accent_safe_name_normalization():
    assert rankings_scraper._normalize_name("José Aldo") == "jose aldo"
    assert method_odds._normalize_name("José Aldo") == "jose aldo"
    assert line_tracker._normalize_fighter_name("José Aldo") == "jose aldo"


@pytest.mark.parametrize(
    ("fighter_name", "expected_query", "fighter_slug", "bfo_name"),
    [
        ("Duško Todorović", "dusko todorovic", "Dusko-Todorovic-9397", "Dusko Todorovic"),
        ("Aleksandar Rakić", "aleksandar rakic", "Aleksandar-Rakic-7356", "Aleksandar Rakic"),
        ("Uroš Medić", "uros medic", "Uros-Medic-10229", "Uros Medic"),
    ],
)
def test_bfo_fighter_search_uses_ascii_query_for_accented_names(
    monkeypatch,
    fighter_name,
    expected_query,
    fighter_slug,
    bfo_name,
):
    queries = []

    class _FakeResponse:
        text = f'<a href="/fighters/{fighter_slug}">{bfo_name}</a>'

    def fake_bfo_get(_url, **kwargs):
        query = kwargs.get("params", {}).get("query")
        queries.append(query)
        assert query.isascii()
        return _FakeResponse()

    monkeypatch.setattr(method_odds, "_bfo_get", fake_bfo_get)

    url = method_odds._bfo_find_fighter_url(fighter_name)

    assert queries == [expected_query]
    assert url == f"https://www.bestfightodds.com/fighters/{fighter_slug}"


def test_bfo_snapshot_fallback_parses_one_event_page_for_card(monkeypatch):
    event_url = "https://www.bestfightodds.com/events/ufc-vegas-119-4225"
    latest_html = f"""
    <html><body>
      <a href="/events/ufc-vegas-119-4225">UFC Vegas 119</a>
    </body></html>
    """
    event_html = """
    <html><body>
      <table>
        <tr><td>Andre Fili</td><td>Vinicius Oliveira</td></tr>
        <tr><th>Fili wins by TKO/KO</th><td>+250</td></tr>
        <tr><th>Fili wins by submission</th><td>+1200</td></tr>
        <tr><th>Fili wins by decision</th><td>+340</td></tr>
        <tr><th>Oliveira wins by TKO/KO</th><td>+230</td></tr>
        <tr><th>Oliveira wins by submission</th><td>+500</td></tr>
        <tr><th>Oliveira wins by decision</th><td>+750</td></tr>
        <tr><td>Ion Cutelaba</td><td>Navajo Stirling</td></tr>
        <tr><th>Cutelaba wins by TKO/KO</th><td>+180</td></tr>
        <tr><th>Cutelaba wins by submission</th><td>+900</td></tr>
        <tr><th>Cutelaba wins by decision</th><td>+400</td></tr>
        <tr><th>Stirling wins by TKO/KO</th><td>+300</td></tr>
        <tr><th>Stirling wins by submission</th><td>+1600</td></tr>
        <tr><th>Stirling wins by decision</th><td>+360</td></tr>
      </table>
    </body></html>
    """
    calls = []

    class _FakeResponse:
        def __init__(self, text):
            self.text = text

    def fake_bfo_get(url, **kwargs):
        calls.append(url)
        if url == method_odds.BFO_LATEST_URL:
            return _FakeResponse(latest_html)
        if url == event_url:
            return _FakeResponse(event_html)
        pytest.fail(f"unexpected BFO URL: {url}")

    monkeypatch.setattr(method_odds, "_bfo_get", fake_bfo_get)
    monkeypatch.setattr(method_odds.time, "sleep", lambda _seconds: None)

    tracked_fights = [
        {
            "event_title": "UFC Vegas 119",
            "event_date": "2026-06-20",
            "fighter_a": "Andre Fili",
            "fighter_b": "Vinicius Oliveira",
            "event_id": "fight-1",
        },
        {
            "event_title": "UFC Vegas 119",
            "event_date": "2026-06-20",
            "fighter_a": "Ion Cutelaba",
            "fighter_b": "Navajo Stirling",
            "event_id": "fight-2",
        },
    ]

    records, source_run = method_odds._collect_bfo_records_for_missing(tracked_fights, [])

    assert calls == [method_odds.BFO_LATEST_URL, event_url]
    assert source_run["status"] == "success"
    assert source_run["attempted"] == 2
    assert source_run["attempted_events"] == 1
    assert len(records) == 2
    assert records[0]["fighter_a"] == "Andre Fili"
    assert records[1]["fighter_a"] == "Ion Cutelaba"
    assert records[0]["method_odds"]["a_ko_odds_prob"] == pytest.approx(
        method_odds._american_to_implied_prob(250)
    )
    assert records[1]["method_odds"]["b_dec_odds_prob"] == pytest.approx(
        method_odds._american_to_implied_prob(360)
    )


def test_bfo_snapshot_fallback_stops_after_failure_budget(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs.get("params", {}).get("query")))
        raise requests.Timeout("BFO hung")

    monkeypatch.setattr(method_odds.requests, "get", fake_get)
    monkeypatch.setattr(method_odds.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(method_odds, "_BFO_MAX_RETRIES", 0)
    monkeypatch.setattr(method_odds, "_BFO_FAILURE_BUDGET_PER_SNAPSHOT", 1)

    tracked_fights = [
        {"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter"},
        {"fighter_a": "Gamma Fighter", "fighter_b": "Delta Fighter"},
        {"fighter_a": "Epsilon Fighter", "fighter_b": "Zeta Fighter"},
    ]

    records, source_run = method_odds._collect_bfo_records_for_missing(tracked_fights, [])

    assert records == []
    assert calls == [(method_odds.BFO_LATEST_URL, None)]
    assert source_run["status"] == "failed"
    assert source_run["attempted"] == 1
    assert source_run["attempted_events"] == 1
    assert "bestfightodds unavailable after 1 consecutive request failures" in source_run["error"]


def test_bfo_get_retries_timeout_then_logs_recovery(monkeypatch, caplog):
    calls = []
    sleep_calls = []

    class _FakeResponse:
        status_code = 200
        headers = {}
        text = "ok"

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        calls.append((url, kwargs.get("timeout")))
        if len(calls) < 3:
            raise requests.Timeout("BFO hung")
        return _FakeResponse()

    monkeypatch.setattr(method_odds.requests, "get", fake_get)
    monkeypatch.setattr(method_odds.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(method_odds, "_BFO_MAX_RETRIES", 2)
    monkeypatch.setattr(method_odds, "_BFO_RETRY_BACKOFF", 2)
    monkeypatch.setattr(method_odds, "_BFO_REQUEST_TIMEOUT", 7)
    caplog.set_level(logging.INFO, logger="src.data.method_odds")

    response = method_odds._bfo_get(method_odds.BFO_LATEST_URL)

    assert response is not None
    assert response.text == "ok"
    assert calls == [
        (method_odds.BFO_LATEST_URL, 7),
        (method_odds.BFO_LATEST_URL, 7),
        (method_odds.BFO_LATEST_URL, 7),
    ]
    assert sleep_calls == [2, 4]
    assert "BFO request recovered on attempt 3/3" in caplog.text


def test_bfo_get_honors_retry_after_on_transient_status(monkeypatch):
    calls = []
    sleep_calls = []

    class _FakeResponse:
        def __init__(self, status_code, headers=None):
            self.status_code = status_code
            self.headers = headers or {}
            self.text = "ok"

        def raise_for_status(self):
            if self.status_code < 400:
                return None
            exc = requests.HTTPError(f"status {self.status_code}")
            exc.response = self
            raise exc

    def fake_get(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            return _FakeResponse(429, {"Retry-After": "9"})
        return _FakeResponse(200)

    monkeypatch.setattr(method_odds.requests, "get", fake_get)
    monkeypatch.setattr(method_odds.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(method_odds, "_BFO_MAX_RETRIES", 1)
    monkeypatch.setattr(method_odds, "_BFO_RETRY_BACKOFF", 2)

    response = method_odds._bfo_get(method_odds.BFO_LATEST_URL)

    assert response is not None
    assert len(calls) == 2
    assert sleep_calls == [9.0]


def test_method_odds_snapshot_retries_bfo_failure_even_with_api_records(monkeypatch):
    event_url = "https://www.bestfightodds.com/events/ufc-test-4000"
    latest_html = f"""
    <html><body>
      <a href="/events/ufc-test-4000">UFC Test</a>
    </body></html>
    """
    event_html = """
    <html><body>
      <table>
        <tr><td>Andre Fili</td><td>Vinicius Oliveira</td></tr>
        <tr><th>Fili wins by TKO/KO</th><td>+250</td></tr>
        <tr><th>Fili wins by submission</th><td>+1200</td></tr>
        <tr><th>Fili wins by decision</th><td>+340</td></tr>
        <tr><th>Oliveira wins by TKO/KO</th><td>+230</td></tr>
        <tr><th>Oliveira wins by submission</th><td>+500</td></tr>
        <tr><th>Oliveira wins by decision</th><td>+750</td></tr>
      </table>
    </body></html>
    """
    api_record = method_odds._snapshot_record(
        fighter_a="Gamma Fighter",
        fighter_b="Delta Fighter",
        method_odds={
            "a_ko_odds_prob": 0.25,
            "a_sub_odds_prob": np.nan,
            "a_dec_odds_prob": np.nan,
            "b_ko_odds_prob": np.nan,
            "b_sub_odds_prob": np.nan,
            "b_dec_odds_prob": 0.35,
        },
        source="odds_api",
        captured_at="2026-06-27T04:23:00",
        event_id="api-fight",
    )
    calls = []
    sleep_calls = []

    class _FakeResponse:
        status_code = 200

        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            raise requests.Timeout("BFO hung")
        if url == method_odds.BFO_LATEST_URL:
            return _FakeResponse(latest_html)
        if url == event_url:
            return _FakeResponse(event_html)
        pytest.fail(f"unexpected BFO URL: {url}")

    monkeypatch.setattr(method_odds, "_collect_api_snapshot_records", lambda: ([api_record], []))
    monkeypatch.setattr(method_odds.requests, "get", fake_get)
    monkeypatch.setattr(method_odds.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(method_odds, "_BFO_MAX_RETRIES", 0)
    monkeypatch.setattr(method_odds, "_BFO_FAILURE_BUDGET_PER_SNAPSHOT", 2)
    monkeypatch.setattr(method_odds, "METHOD_ODDS_COLLECTION_MAX_ATTEMPTS", 2)

    tracked_fights = [
        {
            "event_title": "UFC Test",
            "event_date": "2026-06-27",
            "fighter_a": "Andre Fili",
            "fighter_b": "Vinicius Oliveira",
            "event_id": "bfo-fight",
        }
    ]

    records, source_runs = method_odds._collect_method_odds_snapshot_records(tracked_fights=tracked_fights)

    assert calls == [method_odds.BFO_LATEST_URL, method_odds.BFO_LATEST_URL, event_url]
    assert sleep_calls[0] == method_odds._BFO_RETRY_BACKOFF
    assert len(records) == 2
    assert records[0]["source"] == "odds_api"
    assert records[1]["source"] == "bestfightodds"
    assert source_runs[-1]["status"] == "success"
    assert source_runs[-1]["collection_attempts"] == 2


def test_collect_method_odds_snapshot_retries_zero_record_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(method_odds, "METHOD_ODDS_COLLECTION_MAX_ATTEMPTS", 2)
    sleep_calls = []
    monkeypatch.setattr(method_odds.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    record = method_odds._snapshot_record(
        fighter_a="Alpha Fighter",
        fighter_b="Beta Fighter",
        method_odds={
            "a_ko_odds_prob": 0.25,
            "a_sub_odds_prob": np.nan,
            "a_dec_odds_prob": np.nan,
            "b_ko_odds_prob": np.nan,
            "b_sub_odds_prob": np.nan,
            "b_dec_odds_prob": 0.35,
        },
        source="bestfightodds",
        captured_at="2026-06-27T04:23:00",
        event_id="evt-1",
    )
    calls = []

    def fake_collect(*, tracked_fights=None):
        calls.append(tracked_fights)
        if len(calls) == 1:
            return [], [
                {
                    "source": "bestfightodds",
                    "status": "failed",
                    "record_count": 0,
                    "error": "bestfightodds unavailable after retries",
                }
            ]
        return [record], [
            {
                "source": "bestfightodds",
                "status": "success",
                "record_count": 1,
                "error": "",
            }
        ]

    monkeypatch.setattr(method_odds, "_collect_method_odds_snapshot_records", fake_collect)

    tracked_fights = [{"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter"}]
    snapshot = method_odds.collect_method_odds_snapshot(tracked_fights=tracked_fights)

    assert len(calls) == 2
    assert calls == [tracked_fights, tracked_fights]
    assert sleep_calls == [method_odds._BFO_RETRY_BACKOFF]
    assert snapshot["status"] == "success"
    assert snapshot["record_count"] == 1
    assert snapshot["collection_attempts"] == 2
    saved_paths = list(tmp_path.glob("method_odds_*.json"))
    assert len(saved_paths) == 1
    saved_snapshot = json.loads(saved_paths[0].read_text())
    assert saved_snapshot["status"] == "success"
    assert saved_snapshot["record_count"] == 1


def test_collect_method_odds_snapshot_does_not_retry_unpublished_props(tmp_path, monkeypatch):
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(method_odds, "METHOD_ODDS_COLLECTION_MAX_ATTEMPTS", 3)
    sleep_calls = []
    collect_calls = []
    monkeypatch.setattr(method_odds.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    def fake_collect(*, tracked_fights=None):
        collect_calls.append(tracked_fights)
        return [], [
            {
                "source": "bestfightodds",
                "status": "unavailable",
                "record_count": 0,
                "attempted": 10,
                "error": "no confident pages parsed",
            }
        ]

    monkeypatch.setattr(method_odds, "_collect_method_odds_snapshot_records", fake_collect)

    snapshot = method_odds.collect_method_odds_snapshot(
        tracked_fights=[{"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter"}]
    )

    assert len(collect_calls) == 1
    assert sleep_calls == []
    assert snapshot["status"] == "unavailable"
    assert snapshot["collection_attempts"] == 1


def test_collect_method_odds_snapshot_ignores_unrelated_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", tmp_path)
    unrelated_record = method_odds._snapshot_record(
        fighter_a="Old Fighter",
        fighter_b="Old Opponent",
        method_odds={
            "a_ko_odds_prob": 0.25,
            "a_sub_odds_prob": np.nan,
            "a_dec_odds_prob": np.nan,
            "b_ko_odds_prob": np.nan,
            "b_sub_odds_prob": np.nan,
            "b_dec_odds_prob": 0.35,
        },
        source="bestfightodds",
        captured_at=_fresh_snapshot_time(hours_ago=1),
        event_id="old-event",
    )
    _write_method_snapshot(
        tmp_path,
        _fresh_snapshot_time(hours_ago=1),
        {
            "schema_version": method_odds.SNAPSHOT_SCHEMA_VERSION,
            "status": "success",
            "record_count": 1,
            "records": [unrelated_record],
            "sources": [],
        },
    )
    monkeypatch.setattr(
        method_odds,
        "_collect_method_odds_snapshot_records",
        lambda *, tracked_fights=None: (
            [],
            [{"source": "bestfightodds", "status": "unavailable", "error": "no confident pages parsed"}],
        ),
    )

    snapshot = method_odds.collect_method_odds_snapshot(
        tracked_fights=[
            {
                "fighter_a": "Current Fighter",
                "fighter_b": "Current Opponent",
                "event_id": "current-event",
            }
        ]
    )

    assert snapshot["status"] == "unavailable"
    assert "latest_usable_snapshot" not in snapshot


def test_odds_api_method_source_is_disabled_without_request(monkeypatch):
    monkeypatch.setattr(
        method_odds.requests,
        "get",
        lambda *_args, **_kwargs: pytest.fail("unsupported Odds API markets must not be queried"),
    )

    records, source_runs = method_odds._collect_api_snapshot_records()

    assert records == []
    assert source_runs == [
        {
            "source": "odds_api",
            "status": "unavailable",
            "captured_at": source_runs[0]["captured_at"],
            "error": "MMA method-of-victory markets unsupported; source disabled",
            "record_count": 0,
        }
    ]


def test_method_odds_unpublished_props_warn_only_when_expected_soon():
    quiet_level, quiet_message = live_monitor._method_odds_snapshot_log_message(
        {
            "status": "unavailable",
            "record_count": 0,
            "availability_expected": False,
            "expected_fight_count": 0,
        }
    )
    warning_level, warning_message = live_monitor._method_odds_snapshot_log_message(
        {
            "status": "unavailable",
            "record_count": 0,
            "availability_expected": True,
            "expected_fight_count": 2,
        }
    )

    assert quiet_level == logging.INFO
    assert "unavailable" in quiet_message
    assert warning_level == logging.WARNING
    assert "props expected for 2 fight(s) within 48h" in warning_message


def test_method_odds_partial_expected_coverage_emits_one_collection_warning():
    level, message = live_monitor._method_odds_snapshot_log_message(
        {
            "status": "partial",
            "record_count": 12,
            "coverage_status": "partial",
            "tracked_fight_count": 13,
            "covered_fight_count": 12,
            "availability_expected": True,
            "expected_fight_count": 13,
            "expected_covered_fight_count": 12,
            "expected_missing_fight_count": 1,
        }
    )

    assert level == logging.WARNING
    assert "coverage 12/13" in message
    assert "1 expected fight(s) missing" in message


def test_collect_method_odds_snapshot_reports_latest_usable_snapshot_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(method_odds, "METHOD_ODDS_COLLECTION_MAX_ATTEMPTS", 1)

    record = method_odds._snapshot_record(
        fighter_a="Alpha Fighter",
        fighter_b="Beta Fighter",
        method_odds={
            "a_ko_odds_prob": 0.25,
            "a_sub_odds_prob": np.nan,
            "a_dec_odds_prob": np.nan,
            "b_ko_odds_prob": np.nan,
            "b_sub_odds_prob": np.nan,
            "b_dec_odds_prob": 0.35,
        },
        source="bestfightodds",
        captured_at="2026-06-27T04:23:00",
        event_id="evt-1",
    )
    usable_time = _fresh_snapshot_time(hours_ago=1)
    usable_path = _write_method_snapshot(
        tmp_path,
        usable_time,
        {
            "schema_version": method_odds.SNAPSHOT_SCHEMA_VERSION,
            "snapshot_time": usable_time,
            "status": "success",
            "record_count": 1,
            "records": [record],
            "sources": [{"source": "bestfightodds", "status": "success", "record_count": 1}],
        },
    )

    monkeypatch.setattr(
        method_odds,
        "_collect_method_odds_snapshot_records",
        lambda *, tracked_fights=None: (
            [],
            [{"source": "bestfightodds", "status": "failed", "record_count": 0}],
        ),
    )

    snapshot = method_odds.collect_method_odds_snapshot(
        tracked_fights=[{"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter"}],
    )

    assert snapshot["status"] == "failed"
    latest_usable = snapshot["latest_usable_snapshot"]
    assert latest_usable["snapshot_time"] == usable_time
    assert latest_usable["snapshot_path"] == str(usable_path)
    assert latest_usable["record_count"] == 1
    assert latest_usable["is_stale"] is False
    assert latest_usable["covered_fight_count"] == 1

    saved_failed = json.loads(Path(snapshot["snapshot_path"]).read_text())
    assert saved_failed["latest_usable_snapshot"]["snapshot_time"] == usable_time


def test_method_odds_timestamps_are_timezone_aware_utc():
    timestamp = datetime.fromisoformat(method_odds._now_iso())

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timedelta(0)


def test_method_odds_live_lookup_uses_latest_matching_fight_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", tmp_path)
    method_odds._method_odds_cache.clear()
    matching_time = _fresh_snapshot_time(hours_ago=2)
    unrelated_time = _fresh_snapshot_time(hours_ago=1)

    matching_record = method_odds._snapshot_record(
        fighter_a="Alpha Fighter",
        fighter_b="Beta Fighter",
        method_odds={
            "a_ko_odds_prob": 0.40,
            "a_sub_odds_prob": np.nan,
            "a_dec_odds_prob": np.nan,
            "b_ko_odds_prob": np.nan,
            "b_sub_odds_prob": np.nan,
            "b_dec_odds_prob": 0.30,
        },
        source="bestfightodds",
        captured_at=matching_time,
        event_id="evt-current",
    )
    unrelated_record = method_odds._snapshot_record(
        fighter_a="Old Fighter",
        fighter_b="Old Opponent",
        method_odds={
            "a_ko_odds_prob": 0.91,
            "a_sub_odds_prob": np.nan,
            "a_dec_odds_prob": np.nan,
            "b_ko_odds_prob": np.nan,
            "b_sub_odds_prob": np.nan,
            "b_dec_odds_prob": 0.82,
        },
        source="bestfightodds",
        captured_at=unrelated_time,
        event_id="evt-old",
    )
    _write_method_snapshot(
        tmp_path,
        matching_time,
        {
            "schema_version": method_odds.SNAPSHOT_SCHEMA_VERSION,
            "snapshot_time": matching_time,
            "status": "success",
            "record_count": 1,
            "records": [matching_record],
            "sources": [],
        },
    )
    _write_method_snapshot(
        tmp_path,
        unrelated_time,
        {
            "schema_version": method_odds.SNAPSHOT_SCHEMA_VERSION,
            "snapshot_time": unrelated_time,
            "status": "success",
            "record_count": 1,
            "records": [unrelated_record],
            "sources": [],
        },
    )

    result = method_odds.get_method_odds(
        "Alpha Fighter",
        "Beta Fighter",
        event_id="evt-current",
    )

    assert result["a_ko_odds_prob"] == pytest.approx(0.40)
    assert result["b_dec_odds_prob"] == pytest.approx(0.30)


def test_method_odds_zero_record_attempt_drives_diagnostics_without_per_fight_warning(
    tmp_path,
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", tmp_path)
    method_odds._method_odds_cache.clear()
    stale_time = (
        datetime.now(timezone.utc) - method_odds.METHOD_ODDS_SNAPSHOT_MAX_AGE - timedelta(hours=1)
    ).replace(microsecond=0).isoformat()
    attempt_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    stale_record = method_odds._snapshot_record(
        fighter_a="Alpha Fighter",
        fighter_b="Beta Fighter",
        method_odds={
            "a_ko_odds_prob": 0.40,
            "a_sub_odds_prob": np.nan,
            "a_dec_odds_prob": np.nan,
            "b_ko_odds_prob": np.nan,
            "b_sub_odds_prob": np.nan,
            "b_dec_odds_prob": 0.30,
        },
        source="bestfightodds",
        captured_at=stale_time,
        event_id="evt-1",
    )
    _write_method_snapshot(
        tmp_path,
        stale_time,
        {
            "schema_version": method_odds.SNAPSHOT_SCHEMA_VERSION,
            "snapshot_time": stale_time,
            "status": "success",
            "record_count": 1,
            "records": [stale_record],
            "sources": [],
        },
    )
    _write_method_snapshot(
        tmp_path,
        attempt_time,
        {
            "schema_version": method_odds.SNAPSHOT_SCHEMA_VERSION,
            "snapshot_time": attempt_time,
            "status": "unavailable",
            "record_count": 0,
            "records": [],
            "sources": [{"source": "bestfightodds", "status": "unavailable"}],
        },
    )
    tracked_fights = [
        {"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter", "event_id": "evt-1"}
    ]
    caplog.set_level(logging.WARNING, logger="src.data.method_odds")

    result = method_odds.get_method_odds("Alpha Fighter", "Beta Fighter", event_id="evt-1")
    diagnostics = method_odds.get_method_odds_snapshot_diagnostics(tracked_fights=tracked_fights)

    assert all(np.isnan(result[column]) for column in result)
    assert not any("snapshot is stale" in record.getMessage() for record in caplog.records)
    assert diagnostics["status"] == "unavailable"
    assert diagnostics["latest_attempt"]["record_count"] == 0
    assert diagnostics["latest_matching_snapshot"]["is_stale"] is True
    assert diagnostics["latest_matching_snapshot"]["covered_fight_count"] == 1


def test_method_odds_snapshot_without_parseable_time_is_unusable(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "method_odds"
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", snapshot_dir)
    method_odds._method_odds_cache.clear()
    filename_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    _write_method_snapshot(
        snapshot_dir,
        filename_time,
        {
            "schema_version": 1,
            "snapshot_time": "not-a-time",
            "status": "success",
            "record_count": 1,
            "sources": [],
            "records": [
                {
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "fighter_a_norm": "alpha fighter",
                    "fighter_b_norm": "beta fighter",
                    "event_id": "evt-1",
                    "commence_time": "2026-08-01T18:00:00Z",
                    "captured_at": filename_time,
                    "method_odds": {
                        "a_ko_odds_prob": 0.40,
                        "a_sub_odds_prob": None,
                        "a_dec_odds_prob": None,
                        "b_ko_odds_prob": None,
                        "b_sub_odds_prob": None,
                        "b_dec_odds_prob": 0.30,
                    },
                }
            ],
        },
    )

    result = method_odds.get_method_odds(
        "Alpha Fighter", "Beta Fighter", event_id="evt-1"
    )

    assert all(np.isnan(result[column]) for column in result)


def test_collect_method_odds_snapshot_reports_partial_expected_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(method_odds, "METHOD_ODDS_COLLECTION_MAX_ATTEMPTS", 1)
    commence_time = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    record = method_odds._snapshot_record(
        fighter_a="Alpha Fighter",
        fighter_b="Beta Fighter",
        method_odds={
            "a_ko_odds_prob": 0.40,
            "a_sub_odds_prob": np.nan,
            "a_dec_odds_prob": np.nan,
            "b_ko_odds_prob": np.nan,
            "b_sub_odds_prob": np.nan,
            "b_dec_odds_prob": 0.30,
        },
        source="bestfightodds",
        captured_at=method_odds._now_iso(),
        event_id="evt-1",
        commence_time=commence_time,
        event_title="UFC Test",
    )
    monkeypatch.setattr(
        method_odds,
        "_collect_method_odds_snapshot_records",
        lambda *, tracked_fights=None: (
            [record],
            [{"source": "bestfightodds", "status": "success", "record_count": 1}],
        ),
    )
    tracked_fights = [
        {
            "fighter_a": "Alpha Fighter",
            "fighter_b": "Beta Fighter",
            "event_id": "evt-1",
            "event_title": "UFC Test",
            "commence_time": commence_time,
        },
        {
            "fighter_a": "Gamma Fighter",
            "fighter_b": "Delta Fighter",
            "event_id": "evt-1",
            "event_title": "UFC Test",
            "commence_time": commence_time,
        },
    ]

    snapshot = method_odds.collect_method_odds_snapshot(tracked_fights=tracked_fights)

    assert snapshot["status"] == "partial"
    assert snapshot["coverage_status"] == "partial"
    assert snapshot["tracked_fight_count"] == 2
    assert snapshot["covered_fight_count"] == 1
    assert snapshot["expected_coverage_status"] == "partial"
    assert snapshot["expected_fight_count"] == 2
    assert snapshot["expected_covered_fight_count"] == 1
    assert snapshot["expected_missing_fight_count"] == 1
    assert snapshot["expected_event_count"] == 1
    assert snapshot["expected_events"][0]["status"] == "partial"
    assert snapshot["missing_expected_fights"][0]["fighter_a"] == "Gamma Fighter"


def test_method_odds_fingerprint_changes_only_when_fight_inputs_change(tmp_path, monkeypatch):
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", tmp_path)
    method_odds._method_odds_cache.clear()
    base_time = datetime.now(timezone.utc).replace(microsecond=0)

    missing_fingerprint = method_odds.get_method_odds_fingerprint(
        "Alpha Fighter", "Beta Fighter", event_id="evt-1"
    )

    def write_snapshot(snapshot_time: datetime, ko_probability: float, captured_at: str) -> None:
        record = method_odds._snapshot_record(
            fighter_a="Alpha Fighter",
            fighter_b="Beta Fighter",
            method_odds={
                "a_ko_odds_prob": ko_probability,
                "a_sub_odds_prob": np.nan,
                "a_dec_odds_prob": np.nan,
                "b_ko_odds_prob": np.nan,
                "b_sub_odds_prob": np.nan,
                "b_dec_odds_prob": 0.30,
            },
            source="bestfightodds",
            captured_at=captured_at,
            event_id="evt-1",
        )
        timestamp = snapshot_time.isoformat()
        _write_method_snapshot(
            tmp_path,
            timestamp,
            {
                "schema_version": method_odds.SNAPSHOT_SCHEMA_VERSION,
                "snapshot_time": timestamp,
                "status": "success",
                "record_count": 1,
                "records": [record],
                "sources": [],
            },
        )

    write_snapshot(base_time - timedelta(minutes=3), 0.40, "first capture")
    first_available = method_odds.get_method_odds_fingerprint(
        "Alpha Fighter", "Beta Fighter", event_id="evt-1"
    )
    write_snapshot(base_time - timedelta(minutes=2), 0.40, "second capture")
    unchanged = method_odds.get_method_odds_fingerprint(
        "Alpha Fighter", "Beta Fighter", event_id="evt-1"
    )
    write_snapshot(base_time - timedelta(minutes=1), 0.55, "third capture")
    changed = method_odds.get_method_odds_fingerprint(
        "Alpha Fighter", "Beta Fighter", event_id="evt-1"
    )
    refreshed_result = method_odds.get_method_odds(
        "Alpha Fighter", "Beta Fighter", event_id="evt-1"
    )

    assert first_available != missing_fingerprint
    assert unchanged == first_available
    assert changed != unchanged
    assert refreshed_result["a_ko_odds_prob"] == pytest.approx(0.55)


def test_method_odds_alias_and_canonical_name_share_fingerprint_and_result_cache(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", tmp_path)
    method_odds._method_odds_cache.clear()
    snapshot_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    record = method_odds._snapshot_record(
        fighter_a="Ian Machado Garry",
        fighter_b="Opponent Fighter",
        method_odds={
            "a_ko_odds_prob": 0.40,
            "a_sub_odds_prob": np.nan,
            "a_dec_odds_prob": 0.35,
            "b_ko_odds_prob": 0.30,
            "b_sub_odds_prob": np.nan,
            "b_dec_odds_prob": 0.45,
        },
        source="bestfightodds",
        captured_at=snapshot_time,
        event_id="evt-ian",
    )
    legacy_alias_record = dict(record)
    legacy_alias_record["fighter_a"] = "Ian Garry"
    legacy_alias_record["fighter_a_norm"] = "ian garry"
    _write_method_snapshot(
        tmp_path,
        snapshot_time,
        {
            "schema_version": method_odds.SNAPSHOT_SCHEMA_VERSION,
            "snapshot_time": snapshot_time,
            "status": "success",
            "record_count": 2,
            "records": [record, legacy_alias_record],
            "sources": [],
        },
    )

    canonical_fingerprint = method_odds.get_method_odds_fingerprint(
        "Ian Machado Garry", "Opponent Fighter", event_id="evt-ian"
    )
    alias_fingerprint = method_odds.get_method_odds_fingerprint(
        "Ian Garry", "Opponent Fighter", event_id="evt-ian"
    )
    monkeypatch.setattr(
        method_odds,
        "_resolve_method_odds_result",
        lambda *_args, **_kwargs: pytest.fail("alias lookup should reuse the canonical cache entry"),
    )

    cached = method_odds.get_method_odds(
        "Ian Garry", "Opponent Fighter", event_id="evt-ian"
    )

    assert canonical_fingerprint == alias_fingerprint
    assert cached["a_ko_odds_prob"] == pytest.approx(0.40)


def test_method_odds_publish_cannot_leave_a_prepublication_result_cached(
    tmp_path,
    monkeypatch,
):
    method_odds._method_odds_cache.clear()
    resolver_entered = threading.Event()
    release_resolver = threading.Event()
    publisher_saved = threading.Event()
    errors: list[BaseException] = []
    old_result = {
        "a_ko_odds_prob": 0.40,
        "a_sub_odds_prob": np.nan,
        "a_dec_odds_prob": np.nan,
        "b_ko_odds_prob": np.nan,
        "b_sub_odds_prob": np.nan,
        "b_dec_odds_prob": 0.30,
    }

    def blocking_resolve(*_args, **_kwargs):
        resolver_entered.set()
        if not release_resolver.wait(timeout=2):
            raise RuntimeError("test did not release method-odds resolver")
        return dict(old_result), True

    def fake_save(_snapshot):
        publisher_saved.set()
        return tmp_path / "published.json"

    monkeypatch.setattr(method_odds, "_resolve_method_odds_result", blocking_resolve)
    monkeypatch.setattr(
        method_odds,
        "_collect_method_odds_snapshot_records",
        lambda **_kwargs: ([{"fighter_a": "New", "fighter_b": "Record"}], []),
    )
    monkeypatch.setattr(method_odds, "_save_snapshot", fake_save)

    def run_fingerprint():
        try:
            method_odds.get_method_odds_fingerprint("Alpha", "Beta")
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def run_publish():
        try:
            method_odds.collect_method_odds_snapshot()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    fingerprint_thread = threading.Thread(target=run_fingerprint)
    publish_thread = threading.Thread(target=run_publish)
    fingerprint_thread.start()
    assert resolver_entered.wait(timeout=1)
    publish_thread.start()

    # Publication must wait for the resolve+cache transaction instead of
    # clearing first and allowing the old result to be written afterward.
    assert not publisher_saved.wait(timeout=0.05)
    release_resolver.set()
    fingerprint_thread.join(timeout=2)
    publish_thread.join(timeout=2)

    assert not fingerprint_thread.is_alive()
    assert not publish_thread.is_alive()
    assert errors == []
    assert publisher_saved.is_set()
    assert method_odds._method_odds_cache == {}


def test_discover_bfo_event_url_rejects_ufc_number_substring_collisions(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(scrape_bfo_moneyline, "EVENTS_CACHE_PATH", tmp_path / "bfo_event_urls.json")
    monkeypatch.setattr(
        scrape_bfo_moneyline,
        "search_bfo_events",
        lambda query: [
            ("https://www.bestfightodds.com/events/ufc-200-tate-vs-nunes-1102", "UFC 200: Tate vs Nunes"),
            ("https://www.bestfightodds.com/events/ufc-2-no-way-out-2", "UFC 2: No Way Out"),
        ],
    )

    def fake_parse(url: str) -> list[dict]:
        if "ufc-200" in url:
            return [{"fighter_a": "Amanda Nunes", "fighter_b": "Miesha Tate"}]
        return [
            {"fighter_a": "Royce Gracie", "fighter_b": "Remco Pardoel"},
            {"fighter_a": "Patrick Smith", "fighter_b": "Johnny Rhodes"},
        ]

    monkeypatch.setattr(scrape_bfo_moneyline, "parse_bfo_event_page", fake_parse)

    url = scrape_bfo_moneyline.discover_bfo_event_url(
        "UFC 2: No Way Out",
        "1994-03-11",
        [{"fighter_a": "Royce Gracie", "fighter_b": "Remco Pardoel"}],
    )

    assert url == "https://www.bestfightodds.com/events/ufc-2-no-way-out-2"


def test_discover_bfo_event_url_requires_actual_fight_overlap(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(scrape_bfo_moneyline, "EVENTS_CACHE_PATH", tmp_path / "bfo_event_urls.json")
    monkeypatch.setattr(
        scrape_bfo_moneyline,
        "search_bfo_events",
        lambda query: [
            ("https://www.bestfightodds.com/events/lfa-100-altamirano-vs-smith-2068", "LFA 100: Altamirano vs Smith"),
            ("https://www.bestfightodds.com/events/ufc-100-lesnar-vs-mir-2-1600", "UFC 100: Lesnar vs Mir 2"),
        ],
    )

    def fake_parse(url: str) -> list[dict]:
        if "lfa-100" in url:
            return [{"fighter_a": "Nate Smith", "fighter_b": "Victor Altamirano"}]
        return [
            {"fighter_a": "Brock Lesnar", "fighter_b": "Frank Mir"},
            {"fighter_a": "Dan Henderson", "fighter_b": "Michael Bisping"},
        ]

    monkeypatch.setattr(scrape_bfo_moneyline, "parse_bfo_event_page", fake_parse)

    url = scrape_bfo_moneyline.discover_bfo_event_url(
        "UFC 100",
        "2009-07-11",
        [{"fighter_a": "Brock Lesnar", "fighter_b": "Frank Mir"}],
    )

    assert url == "https://www.bestfightodds.com/events/ufc-100-lesnar-vs-mir-2-1600"
