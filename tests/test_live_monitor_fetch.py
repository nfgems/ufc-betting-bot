"""Fetch-layer tests for live_monitor: HTML dedup cache, retry backoff, the
UFCStats challenge-solver routing, and the fetch-failed vs zero-rows logging split."""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.data import live_monitor, ufcstats_http


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        return None


UFC_COM_EVENTS_HTML = """
<article class="c-card-event--result">
  <a href="/event/ufc-fight-night-june-14-2026">View Event Details</a>
  <div>Alpha vs Beta</div>
  <div>Sun, Jun 14 / 7:00 PM EDT / Main Card</div>
  <div>Arena</div>
  <div>City</div>
</article>
"""

UFCSTATS_UPCOMING_HTML = """
<table>
  <tr class="b-statistics__table-row">
    <td>
      <a class="b-link" href="http://ufcstats.com/event-details/test-event">UFC Fight Night: Alpha vs. Beta</a>
      <span class="b-statistics__date">June 14, 2026</span>
    </td>
  </tr>
</table>
"""

UFCSTATS_CARD_HTML = """
<table>
  <tr class="b-fight-details__table-row">
    <td>
      <a class="b-link" href="#">Alpha Fighter</a>
      <a class="b-link" href="#">Beta Fighter</a>
    </td>
    <td>Lightweight Bout</td>
  </tr>
</table>
"""


def test_fetch_upstream_html_reuses_recent_successful_fetch(monkeypatch):
    calls = {"count": 0}

    def _fake_get(url, *_args, **_kwargs):
        calls["count"] += 1
        return _FakeResponse(UFC_COM_EVENTS_HTML)

    monkeypatch.setattr(live_monitor.requests, "get", _fake_get)

    first = live_monitor._fetch_upstream_html(
        live_monitor.UFC_COM_EVENTS_URL, label="UFC.com events"
    )
    second = live_monitor._fetch_upstream_html(
        live_monitor.UFC_COM_EVENTS_URL, label="UFC.com events"
    )

    assert first == second == UFC_COM_EVENTS_HTML
    assert calls["count"] == 1


def test_shared_upstream_cache_helpers_use_one_cross_fetch_lock():
    started = threading.Event()
    completed = threading.Event()

    def store_from_other_fetch_path():
        started.set()
        live_monitor._store_upstream_html("https://example.test/card", "<html></html>")
        completed.set()

    with live_monitor._UPSTREAM_CACHE_LOCK:
        worker = threading.Thread(target=store_from_other_fetch_path)
        worker.start()
        assert started.wait(timeout=1)
        assert not completed.wait(timeout=0.05)

    worker.join(timeout=1)
    assert not worker.is_alive()
    assert completed.is_set()
    assert live_monitor._cached_upstream_html("https://example.test/card") == "<html></html>"


def test_fetch_upstream_first_success_signals_restart_safe_recovery_once(
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(
        live_monitor.requests,
        "get",
        lambda *_args, **_kwargs: _FakeResponse(UFC_COM_EVENTS_HTML),
    )

    with caplog.at_level(logging.INFO):
        assert live_monitor._fetch_upstream_html(
            live_monitor.UFC_COM_EVENTS_URL,
            label="UFC.com events",
        )
        # Force a second real request without changing the process health state.
        live_monitor._UPSTREAM_HTML_CACHE.clear()
        assert live_monitor._fetch_upstream_html(
            live_monitor.UFC_COM_EVENTS_URL,
            label="UFC.com events",
        )

    recovery_records = [
        record
        for record in caplog.records
        if getattr(record, "alert_recovered_incident_keys", None)
    ]
    assert len(recovery_records) == 1
    assert recovery_records[0].alert_recovered_incident_keys == [
        f"upstream-fetch:{live_monitor.UFC_COM_EVENTS_URL}"
    ]


def test_fetch_upstream_html_backs_off_and_negative_caches_failures(monkeypatch):
    calls = {"count": 0}
    delays: list[float] = []

    def _fake_get(url, *_args, **_kwargs):
        calls["count"] += 1
        raise TimeoutError("read timed out")

    monkeypatch.setattr(live_monitor.requests, "get", _fake_get)
    monkeypatch.setattr(live_monitor.time, "sleep", lambda seconds: delays.append(seconds))

    assert (
        live_monitor._fetch_upstream_html(
            live_monitor.UFC_COM_EVENTS_URL, label="UFC.com events"
        )
        is None
    )
    assert calls["count"] == live_monitor.UPSTREAM_FETCH_ATTEMPTS == 3
    assert delays == [1.0, 2.0]

    # Within the negative-cache window the retry ladder is not paid again, so
    # multiple callers in one outage cycle (contexts + freshness guard) share
    # one failure instead of stacking timeouts.
    assert (
        live_monitor._fetch_upstream_html(
            live_monitor.UFC_COM_EVENTS_URL, label="UFC.com events"
        )
        is None
    )
    assert calls["count"] == 3

    # Once the window expires the fetch is retried for real (failures are
    # never served as content).
    live_monitor._UPSTREAM_FETCH_FAILURE_CACHE[live_monitor.UFC_COM_EVENTS_URL] = (
        time.monotonic() - live_monitor.UPSTREAM_FETCH_FAILURE_TTL_SECONDS - 1.0
    )
    assert (
        live_monitor._fetch_upstream_html(
            live_monitor.UFC_COM_EVENTS_URL, label="UFC.com events"
        )
        is None
    )
    assert calls["count"] == 6


def test_fetch_upstream_200_challenge_is_not_cached_or_marked_recovered(
    monkeypatch,
    caplog,
):
    url = live_monitor.UFC_COM_EVENTS_URL
    live_monitor._UPSTREAM_FETCH_ALERT_ACTIVE_URLS.add(url)
    live_monitor._UPSTREAM_FETCH_RECOVERY_PROBED_URLS.add(url)
    monkeypatch.setattr(
        live_monitor.requests,
        "get",
        lambda *_args, **_kwargs: _FakeResponse(
            "<html><title>Just a moment...</title><body>__cf_chl_tk__</body></html>"
        ),
    )

    with caplog.at_level(logging.INFO):
        result = live_monitor._fetch_upstream_html(
            url,
            label="UFC.com events",
            attempts=1,
        )

    assert result is None
    assert url not in live_monitor._UPSTREAM_HTML_CACHE
    assert url in live_monitor._UPSTREAM_FETCH_ALERT_ACTIVE_URLS
    assert url in live_monitor._UPSTREAM_FETCH_FAILURE_CACHE
    assert not any(
        getattr(record, "alert_recovered_incident_keys", None)
        for record in caplog.records
    )
    assert "browser challenge/interstitial" in caplog.text


def test_fetch_upstream_200_parser_drift_is_not_cached_or_marked_recovered(
    monkeypatch,
    caplog,
):
    url = "https://www.ufc.com/event/parser-drift"
    monkeypatch.setattr(
        live_monitor.requests,
        "get",
        lambda *_args, **_kwargs: _FakeResponse(
            "<html><body><h1>Scheduled maintenance</h1></body></html>"
        ),
    )

    with caplog.at_level(logging.INFO):
        result = live_monitor._fetch_upstream_html(
            url,
            label="UFC.com event card",
            attempts=1,
            required_selector=".c-listing-fight",
            manage_incident=False,
        )

    assert result is None
    assert url not in live_monitor._UPSTREAM_HTML_CACHE
    assert url not in live_monitor._UPSTREAM_FETCH_ALERT_ACTIVE_URLS
    assert not any(
        getattr(record, "alert_recovered_incident_keys", None)
        for record in caplog.records
    )
    warning = next(record for record in caplog.records if record.levelno == logging.WARNING)
    assert getattr(warning, "alert_incident_key", None) is None
    assert "missing expected markup" in caplog.text


def test_ufc_card_inner_name_markup_drift_is_not_cached_as_healthy(
    monkeypatch,
    caplog,
):
    url = "https://www.ufc.com/event/inner-parser-drift"
    monkeypatch.setattr(
        live_monitor.requests,
        "get",
        lambda *_args, **_kwargs: _FakeResponse(
            """
            <html><body>
              <div class="c-listing-fight">
                <span class="new-red-name">Alpha</span>
                <span class="new-blue-name">Beta</span>
              </div>
            </body></html>
            """
        ),
    )

    with caplog.at_level(logging.WARNING):
        fights = live_monitor.scrape_event_card(url)

    assert fights == []
    assert url not in live_monitor._UPSTREAM_HTML_CACHE
    assert url in live_monitor._UPSTREAM_FETCH_FAILURE_CACHE
    assert "parsed zero complete fighter rows" in caplog.text


@pytest.mark.parametrize(
    "empty_event_body",
    [
        """
        <body class="path-node page-node-type-event">
          <article class="node node--type-event">
            <section class="c-hero">
              <div class="c-hero__headline">TBD vs TBD</div>
            </section>
          </article>
        </body>
        """,
        "<body><main>Fight card coming soon</main></body>",
    ],
)
def test_positively_identified_empty_ufc_event_is_healthy_and_not_failure_cached(
    monkeypatch,
    caplog,
    empty_event_body,
):
    url = "https://www.ufc.com/event/announced-empty-card"
    html = f"""
    <html>
      <head><link rel="canonical" href="{url}"></head>
      {empty_event_body}
    </html>
    """
    monkeypatch.setattr(
        live_monitor.requests,
        "get",
        lambda *_args, **_kwargs: _FakeResponse(html),
    )

    with caplog.at_level(logging.INFO):
        card = live_monitor.scrape_event_card(url)

    assert card == []
    assert card.source_healthy is True
    assert live_monitor._cached_upstream_html(url) == html
    assert url not in live_monitor._UPSTREAM_FETCH_FAILURE_CACHE
    assert "announced with zero bouts" in caplog.text


@pytest.mark.parametrize(
    "invalid_html",
    [
        "",
        "<html><title>Just a moment...</title><body>__cf_chl_tk__</body></html>",
        "<html><body><h1>Scheduled maintenance</h1></body></html>",
    ],
)
def test_blank_challenge_or_unrecognized_ufc_event_page_remains_unhealthy(
    monkeypatch,
    invalid_html,
):
    url = "https://www.ufc.com/event/not-a-validated-card"
    monkeypatch.setattr(
        live_monitor.requests,
        "get",
        lambda *_args, **_kwargs: _FakeResponse(invalid_html),
    )
    monkeypatch.setattr(live_monitor.time, "sleep", lambda _seconds: None)

    card = live_monitor.scrape_event_card(url)

    assert card == []
    assert card.source_healthy is False
    assert url not in live_monitor._UPSTREAM_HTML_CACHE
    assert url in live_monitor._UPSTREAM_FETCH_FAILURE_CACHE


def test_valid_empty_event_scan_uses_healthy_reuse_ttl(monkeypatch):
    clock = [100.0]
    calls = {"events": 0, "cards": 0}

    def fake_events():
        calls["events"] += 1
        return [
            {
                "title": "UFC Announced Event",
                "url": "https://www.ufc.com/event/announced-empty-card",
            }
        ]

    def fake_card(_url):
        calls["cards"] += 1
        return live_monitor._EventCardResult(source_healthy=True)

    monkeypatch.setattr(live_monitor, "scrape_upcoming_events", fake_events)
    monkeypatch.setattr(live_monitor, "scrape_event_card", fake_card)
    monkeypatch.setattr(live_monitor.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(live_monitor, "UPSTREAM_FETCH_FAILURE_TTL_SECONDS", 60.0)
    monkeypatch.setattr(live_monitor, "LIVE_EVENT_CONTEXT_REUSE_TTL_SECONDS", 540.0)
    live_monitor.clear_upcoming_event_cards_cache()

    assert live_monitor.collect_upcoming_event_cards() == []
    assert live_monitor._UPCOMING_EVENT_CARDS_CACHE[3] is True

    clock[0] += 61.0
    assert live_monitor.collect_upcoming_event_cards() == []
    assert calls == {"events": 1, "cards": 1}
    live_monitor.clear_upcoming_event_cards_cache()


def test_upcoming_and_completed_scrapes_share_one_events_fetch(monkeypatch):
    calls = {"count": 0}

    def _fake_get(url, *_args, **_kwargs):
        calls["count"] += 1
        return _FakeResponse(UFC_COM_EVENTS_HTML)

    monkeypatch.setattr(live_monitor.requests, "get", _fake_get)

    upcoming = live_monitor.scrape_upcoming_events()
    completed_view = live_monitor.scrape_ufc_com_events(include_completed=True)

    assert upcoming and completed_view
    assert calls["count"] == 1


def test_scrape_ufcstats_upcoming_events_uses_challenge_solver(monkeypatch):
    seen = {}

    def _fake_request_ufcstats(url, *, session=None, **_kwargs):
        seen["url"] = url
        seen["session"] = session
        return _FakeResponse(UFCSTATS_UPCOMING_HTML)

    monkeypatch.setattr(ufcstats_http, "request_ufcstats", _fake_request_ufcstats)

    events = live_monitor._scrape_ufcstats_upcoming_events()

    assert seen["url"] == live_monitor.UFCSTATS_UPCOMING_URL
    assert seen["session"] is not None
    assert events == [
        {
            "title": "UFC Fight Night: Alpha vs. Beta",
            "url": "http://ufcstats.com/event-details/test-event",
            "date": "June 14, 2026",
        }
    ]


def test_fetch_ufcstats_html_reuses_valid_cached_markup(monkeypatch):
    url = "http://ufcstats.com/statistics/events/upcoming"
    live_monitor._store_upstream_html(url, UFCSTATS_UPCOMING_HTML)

    monkeypatch.setattr(
        ufcstats_http,
        "request_ufcstats",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("valid cached UFCStats HTML should avoid a network fetch")
        ),
    )

    result = live_monitor._fetch_ufcstats_html(
        url,
        label="UFCStats upcoming events",
        required_selector="tr.b-statistics__table-row",
    )

    assert result == UFCSTATS_UPCOMING_HTML
    assert live_monitor._cached_upstream_html(url) == UFCSTATS_UPCOMING_HTML


@pytest.mark.parametrize(
    ("invalid_html", "expected_error"),
    [
        ("", "empty response body"),
        (
            "<html><body>Checking your browser; JavaScript is required</body></html>",
            "browser challenge/block page",
        ),
        (
            "<html><title>Access Denied</title><body>Request blocked</body></html>",
            "browser challenge/block page",
        ),
        (
            "<html><body><h1>UFCStats maintenance</h1></body></html>",
            "missing expected selector",
        ),
    ],
)
def test_fetch_ufcstats_html_rejects_invalid_fresh_markup(
    monkeypatch,
    caplog,
    invalid_html,
    expected_error,
):
    url = "http://ufcstats.com/statistics/events/upcoming"
    monkeypatch.setattr(
        ufcstats_http,
        "request_ufcstats",
        lambda *_args, **_kwargs: _FakeResponse(invalid_html),
    )

    with caplog.at_level(logging.WARNING):
        result = live_monitor._fetch_ufcstats_html(
            url,
            label="UFCStats upcoming events",
            required_selector="tr.b-statistics__table-row",
        )

    assert result is None
    assert url not in live_monitor._UPSTREAM_HTML_CACHE
    assert url in live_monitor._UPSTREAM_FETCH_FAILURE_CACHE
    assert expected_error in caplog.text


@pytest.mark.parametrize(
    "invalid_cached_html",
    [
        "",
        "<html><body>Checking your browser; JavaScript is required</body></html>",
        "<html><title>Access Denied</title><body>Request blocked</body></html>",
        "<html><body><h1>UFCStats maintenance</h1></body></html>",
    ],
)
def test_fetch_ufcstats_html_discards_invalid_cache_then_fetches_valid_markup(
    monkeypatch,
    invalid_cached_html,
):
    url = "http://ufcstats.com/statistics/events/upcoming"
    calls = []
    live_monitor._store_upstream_html(url, invalid_cached_html)

    def _fake_request_ufcstats(request_url, **_kwargs):
        calls.append(request_url)
        return _FakeResponse(UFCSTATS_UPCOMING_HTML)

    monkeypatch.setattr(
        ufcstats_http,
        "request_ufcstats",
        _fake_request_ufcstats,
    )

    result = live_monitor._fetch_ufcstats_html(
        url,
        label="UFCStats upcoming events",
        required_selector="tr.b-statistics__table-row",
    )

    assert result == UFCSTATS_UPCOMING_HTML
    assert calls == [url]
    assert live_monitor._cached_upstream_html(url) == UFCSTATS_UPCOMING_HTML


def test_ufcstats_call_sites_supply_parser_required_selectors(monkeypatch):
    calls = []

    def _fake_fetch(url, *, label, required_selector=None):
        calls.append((url, label, required_selector))
        return None

    monkeypatch.setattr(live_monitor, "_fetch_ufcstats_html", _fake_fetch)

    assert live_monitor._scrape_ufcstats_upcoming_events() == []
    card_url = "http://ufcstats.com/event-details/test-event"
    assert live_monitor.scrape_event_card(card_url) == []
    assert calls == [
        (
            live_monitor.UFCSTATS_UPCOMING_URL,
            "UFCStats upcoming events",
            "tr.b-statistics__table-row",
        ),
        (
            card_url,
            "UFCStats event card",
            "tr.b-fight-details__table-row",
        ),
    ]


def test_scrape_ufcstats_upcoming_events_fails_soft_on_unsolved_challenge(monkeypatch, caplog):
    def _fake_request_ufcstats(url, **_kwargs):
        raise ufcstats_http.UFCStatsChallengeError("still gated")

    monkeypatch.setattr(ufcstats_http, "request_ufcstats", _fake_request_ufcstats)

    with caplog.at_level(logging.WARNING):
        events = live_monitor._scrape_ufcstats_upcoming_events()

    assert events == []
    assert "UFCStats upcoming events fetch failed" in caplog.text


def test_scrape_event_card_routes_ufcstats_urls_through_solver(monkeypatch):
    seen = {}

    def _fake_request_ufcstats(url, **_kwargs):
        seen["url"] = url
        return _FakeResponse(UFCSTATS_CARD_HTML)

    monkeypatch.setattr(ufcstats_http, "request_ufcstats", _fake_request_ufcstats)

    fights = live_monitor.scrape_event_card("http://ufcstats.com/event-details/test-event")

    assert seen["url"] == "http://ufcstats.com/event-details/test-event"
    assert len(fights) == 1
    assert fights[0]["fighter_a"] == "Alpha Fighter"
    assert fights[0]["fighter_b"] == "Beta Fighter"


def test_scrape_upcoming_events_logs_fetch_failure_distinctly(monkeypatch, caplog):
    monkeypatch.setattr(live_monitor, "_scrape_ufc_com_upcoming_events", lambda: None)
    monkeypatch.setattr(live_monitor, "_scrape_ufcstats_upcoming_events", lambda: [])

    with caplog.at_level(logging.WARNING):
        assert live_monitor.scrape_upcoming_events() == []

    assert "UFC.com upcoming events fetch failed; no events found from UFCStats fallback" in caplog.text


def test_scrape_upcoming_events_logs_zero_rows_distinctly(monkeypatch, caplog):
    monkeypatch.setattr(live_monitor, "_scrape_ufc_com_upcoming_events", lambda: [])
    monkeypatch.setattr(live_monitor, "_scrape_ufcstats_upcoming_events", lambda: [])

    with caplog.at_level(logging.WARNING):
        assert live_monitor.scrape_upcoming_events() == []

    assert (
        "UFC.com upcoming events fetched but parsed zero event rows; no events found "
        "from UFCStats fallback"
    ) in caplog.text


def test_scrape_upcoming_events_logs_recovered_ufcstats_fallback_at_info(monkeypatch, caplog):
    monkeypatch.setattr(live_monitor, "_scrape_ufc_com_upcoming_events", lambda: None)
    monkeypatch.setattr(
        live_monitor,
        "_scrape_ufcstats_upcoming_events",
        lambda: [{"title": "Fallback Event", "url": "http://ufcstats.com/event-details/test"}],
    )

    with caplog.at_level(logging.INFO):
        events = live_monitor.scrape_upcoming_events()

    assert events == [{"title": "Fallback Event", "url": "http://ufcstats.com/event-details/test"}]
    assert "UFC.com upcoming events fetch failed; using UFCStats fallback" in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_upcoming_event_card_producer_is_shared_across_consumers(monkeypatch):
    calls = {"events": 0, "cards": 0}

    def fake_events():
        calls["events"] += 1
        return [
            {
                "title": "UFC Test",
                "url": "https://www.ufc.com/event/test",
                "date": "2026-07-25",
            }
        ]

    def fake_card(_url):
        calls["cards"] += 1
        return [
            {
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
                "weight_class": "Lightweight",
                "num_rounds": 3,
            }
        ]

    monkeypatch.setattr(live_monitor, "scrape_upcoming_events", fake_events)
    monkeypatch.setattr(live_monitor, "scrape_event_card", fake_card)
    monkeypatch.setattr(live_monitor, "LIVE_EVENT_CONTEXT_REUSE_TTL_SECONDS", 900)
    live_monitor.clear_upcoming_event_cards_cache()

    cards = live_monitor.collect_upcoming_event_cards()
    contexts = live_monitor.collect_upcoming_fight_contexts()

    assert len(cards) == 1
    assert len(contexts) == 1
    assert calls == {"events": 1, "cards": 1}
    live_monitor.clear_upcoming_event_cards_cache()


def test_upcoming_event_card_producer_is_single_flight_for_overlapping_threads(monkeypatch):
    calls = {"events": 0, "cards": 0}
    producer_started = threading.Event()
    release_producer = threading.Event()

    def fake_events():
        calls["events"] += 1
        producer_started.set()
        assert release_producer.wait(timeout=2.0)
        return [{"title": "UFC Test", "url": "https://www.ufc.com/event/test"}]

    def fake_card(_url):
        calls["cards"] += 1
        return [{"fighter_a": "Alpha", "fighter_b": "Beta"}]

    monkeypatch.setattr(live_monitor, "scrape_upcoming_events", fake_events)
    monkeypatch.setattr(live_monitor, "scrape_event_card", fake_card)
    live_monitor.clear_upcoming_event_cards_cache()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(live_monitor.collect_upcoming_event_cards)
        assert producer_started.wait(timeout=2.0)
        second = pool.submit(live_monitor.collect_upcoming_event_cards)
        release_producer.set()
        assert first.result(timeout=2.0) == second.result(timeout=2.0)

    assert calls == {"events": 1, "cards": 1}
    live_monitor.clear_upcoming_event_cards_cache()


def test_upcoming_event_card_producer_shares_empty_failure_briefly(monkeypatch):
    calls = {"events": 0}

    def fake_events():
        calls["events"] += 1
        return []

    monkeypatch.setattr(live_monitor, "scrape_upcoming_events", fake_events)
    live_monitor.clear_upcoming_event_cards_cache()

    assert live_monitor.collect_upcoming_event_cards() == []
    assert live_monitor.collect_upcoming_fight_contexts() == []
    assert calls["events"] == 1
    live_monitor.clear_upcoming_event_cards_cache()


def test_upcoming_event_card_producer_uses_short_ttl_for_partial_scan(monkeypatch):
    calls = {"events": 0, "cards": 0}
    clock = [100.0]

    def fake_events():
        calls["events"] += 1
        return [
            {"title": "UFC A", "url": "https://www.ufc.com/event/a"},
            {"title": "UFC B", "url": "https://www.ufc.com/event/b"},
        ]

    def fake_card(url):
        calls["cards"] += 1
        if calls["events"] == 1 and url.endswith("/b"):
            return []
        return [{"fighter_a": "Alpha", "fighter_b": "Beta"}]

    monkeypatch.setattr(live_monitor, "scrape_upcoming_events", fake_events)
    monkeypatch.setattr(live_monitor, "scrape_event_card", fake_card)
    monkeypatch.setattr(live_monitor.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(live_monitor.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(live_monitor, "UPSTREAM_FETCH_FAILURE_TTL_SECONDS", 60.0)
    monkeypatch.setattr(live_monitor, "LIVE_EVENT_CONTEXT_REUSE_TTL_SECONDS", 540.0)
    live_monitor.clear_upcoming_event_cards_cache()

    assert len(live_monitor.collect_upcoming_event_cards()) == 1
    clock[0] += 30.0
    assert len(live_monitor.collect_upcoming_event_cards()) == 1
    assert calls == {"events": 1, "cards": 2}

    clock[0] += 31.0
    assert len(live_monitor.collect_upcoming_event_cards()) == 2
    assert calls == {"events": 2, "cards": 4}
    live_monitor.clear_upcoming_event_cards_cache()
