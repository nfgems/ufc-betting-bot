import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest
from bs4 import BeautifulSoup

from src import config
from src.data import fallback_scrapers, method_odds


def _disable_tapology_origin(monkeypatch) -> None:
    monkeypatch.setattr(fallback_scrapers, "_tapology_prefer_reader", lambda: True)
    monkeypatch.setattr(
        fallback_scrapers,
        "_tapology_runtime_fetch_allowed",
        lambda: False,
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "_tapology_browser_fallback_available",
        lambda: False,
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "_tapology_reader_available",
        lambda: True,
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "_sleep_after_request",
        lambda _seconds: None,
    )


def test_tapology_explicit_zero_result_search_is_healthy(monkeypatch, caplog):
    fallback_scrapers.clear_fallback_cache()
    _disable_tapology_origin(monkeypatch)
    monkeypatch.setattr(
        fallback_scrapers,
        "_get_tapology_search_markdown_with_reader",
        lambda _url: (
            "Title: Search Fighters, Bouts & Events | Tapology\n"
            "Markdown Content:\n"
            "### Search Results (0)"
        ),
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "_search_site_candidates",
        lambda *_args, **_kwargs: pytest.fail(
            "an explicit healthy zero result must not fall through to uncertain discovery"
        ),
    )

    diagnostics = {}
    with caplog.at_level("WARNING"):
        candidates = fallback_scrapers.search_tapology_candidates(
            "Missing Fighter",
            diagnostics=diagnostics,
        )

    assert candidates == []
    assert diagnostics["healthy"] is True
    assert diagnostics["candidate_count"] == 0
    assert diagnostics["attempts"]
    assert {
        attempt["result"] for attempt in diagnostics["attempts"]
    } == {"no_results"}
    assert fallback_scrapers._tapology_reader_unavailable is False
    assert "External data source unavailable: Tapology" not in caplog.text


def test_tapology_nonzero_result_without_parseable_links_is_unhealthy(
    monkeypatch,
    caplog,
):
    fallback_scrapers.clear_fallback_cache()
    _disable_tapology_origin(monkeypatch)
    fetches = []

    def fake_reader(search_url):
        fetches.append(search_url)
        return (
            "Title: Search Fighters, Bouts & Events | Tapology\n"
            "Markdown Content:\n"
            "### Search Results (1)\n"
            "| Fighters (1) | Weight Class | Record |\n"
            "| --- | --- | --- |"
        )

    monkeypatch.setattr(
        fallback_scrapers,
        "_get_tapology_search_markdown_with_reader",
        fake_reader,
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "_search_site_candidates",
        lambda *_args, **_kwargs: [],
    )

    diagnostics = {}
    with caplog.at_level("WARNING"):
        assert fallback_scrapers.search_tapology_candidates(
            "Parser Drift Fighter",
            diagnostics=diagnostics,
        ) == []

    assert diagnostics["healthy"] is False
    assert any(
        attempt["result"] == "unrecognized_markup"
        for attempt in diagnostics["attempts"]
    )
    assert "Tapology - reader search returned unrecognized markup" in caplog.text

    first_fetch_count = len(fetches)
    assert fallback_scrapers.search_tapology("Parser Drift Fighter") is None
    assert len(fetches) > first_fetch_count
    assert "Parser Drift Fighter" not in fallback_scrapers._tapology_url_cache


def test_tapology_origin_unrecognized_markup_is_unhealthy(monkeypatch):
    fallback_scrapers.clear_fallback_cache()
    monkeypatch.setattr(fallback_scrapers, "_tapology_prefer_reader", lambda: False)
    monkeypatch.setattr(
        fallback_scrapers,
        "_tapology_runtime_fetch_allowed",
        lambda: True,
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "_tapology_browser_fallback_available",
        lambda: False,
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "_tapology_reader_available",
        lambda: False,
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "_get_tapology_soup",
        lambda *_args, **_kwargs: BeautifulSoup(
            "<html><body><h1>Scheduled maintenance</h1></body></html>",
            "lxml",
        ),
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "_search_site_candidates",
        lambda *_args, **_kwargs: [],
    )

    diagnostics = {}
    assert fallback_scrapers.search_tapology_candidates(
        "Parser Drift Fighter",
        diagnostics=diagnostics,
    ) == []

    assert diagnostics["healthy"] is False
    assert any(
        attempt["channel"] == "origin"
        and attempt["result"] == "unrecognized_markup"
        for attempt in diagnostics["attempts"]
    )


def test_tapology_origin_explicit_zero_requires_search_page_identity(monkeypatch):
    fallback_scrapers.clear_fallback_cache()
    monkeypatch.setattr(fallback_scrapers, "_tapology_prefer_reader", lambda: False)
    monkeypatch.setattr(
        fallback_scrapers,
        "_tapology_runtime_fetch_allowed",
        lambda: True,
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "_tapology_browser_fallback_available",
        lambda: False,
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "_tapology_reader_available",
        lambda: False,
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "_name_query_variants",
        lambda fighter_name: [fighter_name],
    )
    pages = iter(
        (
            BeautifulSoup(
                """
                <html>
                  <head><title>Search Fighters, Bouts & Events | Tapology</title></head>
                  <body><h3>Search Results (0)</h3></body>
                </html>
                """,
                "lxml",
            ),
            BeautifulSoup(
                """
                <html>
                  <head><title>Tapology Maintenance</title></head>
                  <body>
                    <h1>Tapology maintenance</h1>
                    <p>No results while maintenance is underway.</p>
                  </body>
                </html>
                """,
                "lxml",
            ),
        )
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "_get_tapology_soup",
        lambda *_args, **_kwargs: next(pages),
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "_search_site_candidates",
        lambda *_args, **_kwargs: [],
    )

    healthy_diagnostics = {}
    assert fallback_scrapers.search_tapology_candidates(
        "Healthy Missing Fighter",
        diagnostics=healthy_diagnostics,
    ) == []
    assert healthy_diagnostics["healthy"] is True
    assert any(
        attempt["channel"] == "origin" and attempt["result"] == "no_results"
        for attempt in healthy_diagnostics["attempts"]
    )

    unhealthy_diagnostics = {}
    assert fallback_scrapers.search_tapology_candidates(
        "Uncertain Missing Fighter",
        diagnostics=unhealthy_diagnostics,
    ) == []
    assert unhealthy_diagnostics["healthy"] is False
    assert any(
        attempt["channel"] == "origin"
        and attempt["result"] == "unrecognized_markup"
        for attempt in unhealthy_diagnostics["attempts"]
    )


class _FightDXResponse:
    status_code = 200

    def __init__(self, text: str):
        self.text = text


def test_fightdx_explicit_zero_result_search_is_healthy(monkeypatch):
    fallback_scrapers.clear_fallback_cache()
    monkeypatch.setattr(
        fallback_scrapers,
        "_request_fightdx",
        lambda *_args, **_kwargs: _FightDXResponse(
            """
            <html>
              <head><title>Search | FightDX</title></head>
              <body><main>No fighters found</main></body>
            </html>
            """
        ),
    )

    healthy, candidates = fallback_scrapers._search_fightdx_site_candidates(
        "Missing Fighter"
    )

    assert healthy is True
    assert candidates == []
    assert fallback_scrapers._fightdx_unavailable_active is False
    assert not any(
        source == "FightDX" and "search" in issue
        for source, issue in fallback_scrapers._external_source_alert_keys
    )


def test_fightdx_unrecognized_search_markup_is_unhealthy(monkeypatch, caplog):
    fallback_scrapers.clear_fallback_cache()
    monkeypatch.setattr(
        fallback_scrapers,
        "_request_fightdx",
        lambda *_args, **_kwargs: _FightDXResponse(
            "<html><body><form>Sign in to continue</form></body></html>"
        ),
    )

    with caplog.at_level("WARNING"):
        healthy, candidates = fallback_scrapers._search_fightdx_site_candidates(
            "Missing Fighter"
        )

    assert healthy is False
    assert candidates == []
    assert "FightDX - search returned unrecognized markup" in caplog.text


@pytest.mark.parametrize(
    "markup",
    [
        """
        <html>
          <head><title>FightDX Maintenance</title></head>
          <body><h1>FightDX maintenance</h1><p>No results are available.</p></body>
        </html>
        """,
        """
        <html>
          <head><title>Search | FightDX</title></head>
          <body><main>10 results</main></body>
        </html>
        """,
    ],
)
def test_fightdx_uncertain_empty_markup_is_unhealthy_and_not_cached(
    monkeypatch,
    markup,
):
    fallback_scrapers.clear_fallback_cache()
    calls = []

    def fake_request(url, **_kwargs):
        calls.append(url)
        if "/person/" in url:
            response = _FightDXResponse("")
            response.status_code = 404
            return response
        return _FightDXResponse(markup)

    monkeypatch.setattr(
        fallback_scrapers,
        "_request_fightdx",
        fake_request,
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "_search_fightdx_sitemap_candidates",
        lambda *_args, **_kwargs: [],
    )

    assert fallback_scrapers.search_fightdx("Uncertain Fighter") is None
    first_call_count = len(calls)
    assert "Uncertain Fighter" not in fallback_scrapers._fightdx_url_cache

    assert fallback_scrapers.search_fightdx("Uncertain Fighter") is None
    assert len(calls) > first_call_count
    assert "Uncertain Fighter" not in fallback_scrapers._fightdx_url_cache


def test_fightdx_semantically_invalid_200_does_not_clear_outage(monkeypatch):
    fallback_scrapers.clear_fallback_cache()
    incident = (
        "FightDX",
        fallback_scrapers._FIGHTDX_TRANSIENT_ALERT_ISSUE,
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "_fightdx_unavailable_active",
        True,
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "_fightdx_unavailable_until",
        0.0,
    )
    monkeypatch.setattr(
        fallback_scrapers,
        "_external_source_alert_keys",
        {incident},
    )
    monkeypatch.setattr(
        fallback_scrapers.requests,
        "get",
        lambda *_args, **_kwargs: _FightDXResponse(
            """
            <html>
              <head><title>FightDX Maintenance</title></head>
              <body><p>No results during maintenance.</p></body>
            </html>
            """
        ),
    )

    healthy, candidates = fallback_scrapers._search_fightdx_site_candidates(
        "Missing Fighter"
    )

    assert healthy is False
    assert candidates == []
    assert fallback_scrapers._fightdx_unavailable_active is True
    assert incident in fallback_scrapers._external_source_alert_keys


@pytest.mark.parametrize(
    "configured_value",
    ["0", "-0.5", "banana", "nan", "inf", "-inf"],
)
def test_invalid_method_odds_max_age_rejects_config_import(configured_value):
    env = os.environ.copy()
    env["METHOD_ODDS_SNAPSHOT_MAX_AGE_HOURS"] = configured_value

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src import config; print(config.METHOD_ODDS_SNAPSHOT_MAX_AGE_HOURS)",
        ],
        cwd=config.PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode != 0
    output = f"{completed.stdout}\n{completed.stderr}"
    assert (
        "METHOD_ODDS_SNAPSHOT_MAX_AGE_HOURS must be a finite number "
        "greater than 0"
    ) in output


def test_positive_method_odds_max_age_imports_and_preserves_value():
    env = os.environ.copy()
    env["METHOD_ODDS_SNAPSHOT_MAX_AGE_HOURS"] = "2.5"

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src import config; print(config.METHOD_ODDS_SNAPSHOT_MAX_AGE_HOURS)",
        ],
        cwd=config.PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "2.5"


def test_positive_method_odds_max_age_keeps_existing_strict_cutoff():
    snapshot = {"snapshot_time": "2026-07-23T10:00:00Z"}
    cutoff = timedelta(hours=2.5)
    exact_cutoff = datetime(2026, 7, 23, 12, 30, tzinfo=timezone.utc)

    assert method_odds._snapshot_is_stale_at(
        snapshot,
        reference_time=exact_cutoff,
        max_age=cutoff,
    ) is False
    assert method_odds._snapshot_is_stale_at(
        snapshot,
        reference_time=exact_cutoff + timedelta(microseconds=1),
        max_age=cutoff,
    ) is True
