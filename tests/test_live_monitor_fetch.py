"""Fetch-layer tests for live_monitor: HTML dedup cache, retry backoff, the
UFCStats challenge-solver routing, and the fetch-failed vs zero-rows logging split."""

import logging
import time

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
