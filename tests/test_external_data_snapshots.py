import json
import logging
from datetime import datetime, timedelta
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
    method_odds._method_odds_cache.clear()

    _write_method_snapshot(
        snapshot_dir,
        "2024-01-10T10:00:00",
        {
            "schema_version": 1,
            "snapshot_time": "2024-01-10T10:00:00",
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
                    "captured_at": "2024-01-10T10:00:00",
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
        "2024-01-20T10:00:00",
        {
            "schema_version": 1,
            "snapshot_time": "2024-01-20T10:00:00",
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
                    "captured_at": "2024-01-20T10:00:00",
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
        as_of_date="2024-01-15",
    )

    assert result["a_ko_odds_prob"] == pytest.approx(0.40)
    assert result["b_dec_odds_prob"] == pytest.approx(0.30)


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


def test_injury_detector_treats_extreme_line_move_as_advisory_warning():
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


def test_injury_detector_treats_near_zero_price_as_advisory_warning():
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
            "days_notice": 2,
            "is_short_notice": True,
        }
    ]
    assert "SHORT NOTICE: Gamma Fighter replacing Beta Fighter with 2 days notice" in caplog.text


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
