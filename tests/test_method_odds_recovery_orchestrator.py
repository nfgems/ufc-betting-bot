import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from bs4 import BeautifulSoup

from scripts import recover_post_cutoff_method_odds as recovery
from src.data import method_odds
from src.model.training_spec import V4_METHOD_ODDS_FEATURE_COLS


def _scoped_html(*, alpha_ko="+200"):
    return f"""
    <html><body><table class="odds-table">
      <tr id="mu-100"><td class="t-b-fcc">Alpha Fighter</td></tr>
      <tr><td class="t-b-fcc">Beta Fighter</td></tr>
      <tr><th>Alpha Fighter wins by TKO/KO</th><td>{alpha_ko}</td></tr>
      <tr><th>Alpha Fighter wins by submission</th><td>+400</td></tr>
      <tr><th>Alpha Fighter wins by decision</th><td>+300</td></tr>
      <tr><th>Beta Fighter wins by TKO/KO</th><td>+250</td></tr>
      <tr><th>Beta Fighter wins by submission</th><td>+500</td></tr>
      <tr><th>Beta Fighter wins by decision</th><td>+350</td></tr>
    </table></body></html>
    """


def _markerless_html():
    return """
    <html><body><h1>Alpha Fighter vs Beta Fighter</h1><table>
      <tr><th>Alpha Fighter wins by TKO/KO</th><td>+200</td></tr>
      <tr><th>Beta Fighter wins by decision</th><td>+350</td></tr>
    </table></body></html>
    """


def _commence_provenance(fighter_a, fighter_b, event_id="evt-1"):
    return {
        "commence_time_source_kind": "odds_api_historical_events",
        "commence_time_source_locator": (
            "https://api.the-odds-api.com/v4/historical/sports/"
            "mma_mixed_martial_arts/events?date=2026-07-31T12:00:00Z"
        ),
        "commence_time_source_observed_at": "2026-07-31T12:00:00Z",
        "commence_time_source_sha256": "a" * 64,
        "commence_time_source_event_id": event_id,
        "commence_time_source_matchup": f"{fighter_a} vs {fighter_b}",
        "commence_time_semantics": "scheduled_bout_time",
    }


def _target_frame(*, commence_time="2026-08-01T22:00:00Z"):
    provenance = (
        _commence_provenance("Alpha Fighter", "Beta Fighter")
        if commence_time
        else {}
    )
    return recovery.prepare_targets(
        pd.DataFrame(
            [
                {
                    "event_date": "2026-08-01",
                    "event_id": "evt-1",
                    "event_title": "UFC Test",
                    "fighter_a": "Alpha Fighter",
                    "fighter_b": "Beta Fighter",
                    "commence_time": commence_time,
                    "source_url": "https://www.bestfightodds.com/events/ufc-test",
                    "a_num_fights": 8,
                    "b_num_fights": 9,
                    **provenance,
                }
            ]
        )
    )


def _evidence(acquisition, captured_at, html=None, url=None):
    source_url = url or "https://www.bestfightodds.com/events/ufc-test"
    return {
        "acquisition": acquisition,
        "source_url": source_url,
        "original_source_url": "https://www.bestfightodds.com/events/ufc-test",
        "captured_at": captured_at,
        "html": html or _scoped_html(),
    }


def _result(evidence=None, *, urls=None, error=""):
    return {
        "evidence": evidence or [],
        "urls": urls or ["https://www.bestfightodds.com/events/ufc-test"],
        "error": error,
    }


def _parsed_values(html=None):
    parsed = method_odds._parse_bfo_method_odds(
        BeautifulSoup(html or _scoped_html(), "lxml"),
        "Alpha Fighter",
        "Beta Fighter",
    )
    assert parsed is not None
    return parsed


def test_direct_success_cache_preserves_first_capture_and_retries_failures(monkeypatch):
    url = "https://www.bestfightodds.com/events/ufc-test"
    cache = recovery.RecoveryRunCache()
    calls = []

    class Response:
        text = _scoped_html()

    responses = [None, Response()]

    def fetch(requested_url):
        calls.append(requested_url)
        return responses.pop(0)

    capture_calls = []
    monkeypatch.setattr(method_odds, "_bfo_get", fetch)
    monkeypatch.setattr(
        recovery,
        "_now_iso",
        lambda: capture_calls.append(True) or "2026-07-30T12:00:00+00:00",
    )

    failed = recovery.acquire_direct_bfo({}, [url], cache=cache)
    first = recovery.acquire_direct_bfo({}, [url], cache=cache)
    cached = recovery.acquire_direct_bfo({}, [url], cache=cache)

    assert failed["evidence"] == []
    assert calls == [url, url]
    assert len(capture_calls) == 1
    assert cached["evidence"][0] == first["evidence"][0]
    assert cached["evidence"][0]["captured_at"] == "2026-07-30T12:00:00+00:00"
    assert cached["evidence"][0]["html"] == _scoped_html()


def test_playwright_success_cache_avoids_second_navigation(monkeypatch):
    url = "https://www.bestfightodds.com/events/ufc-test"
    cache = recovery.RecoveryRunCache()

    class Page:
        def __init__(self):
            self.goto_calls = []
            self.content_calls = 0

        def goto(self, requested_url, **_kwargs):
            self.goto_calls.append(requested_url)

        def wait_for_selector(self, *_args, **_kwargs):
            return None

        def content(self):
            self.content_calls += 1
            return _scoped_html()

    page = Page()
    acquirer = recovery.PlaywrightBfoAcquirer(cache=cache)
    acquirer._page = page
    captures = iter(
        ["2026-07-30T12:00:00+00:00", "2026-07-30T13:00:00+00:00"]
    )
    monkeypatch.setattr(recovery, "_now_iso", lambda: next(captures))

    first = acquirer({}, [url])
    cached = acquirer({}, [url])

    assert page.goto_calls == [url]
    assert page.content_calls == 1
    assert cached["evidence"][0] == first["evidence"][0]
    assert cached["evidence"][0]["captured_at"] == "2026-07-30T12:00:00+00:00"


def test_wayback_success_cache_reuses_cdx_and_html_but_reparses_target(monkeypatch):
    url = "https://www.bestfightodds.com/events/ufc-test"
    timestamp = "20260731215959"
    calls = []

    class Response:
        def __init__(self, *, payload=None, text=""):
            self._payload = payload
            self.text = text

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fetch(requested_url, **_kwargs):
        calls.append(requested_url)
        if requested_url.endswith("/cdx/search/cdx"):
            return Response(payload=[["timestamp", "statuscode"], [timestamp, "200"]])
        return Response(text=_scoped_html())

    parse_calls = []
    original_parse = method_odds._parse_bfo_method_odds

    def tracked_parse(*args, **kwargs):
        parse_calls.append((args[1], args[2]))
        return original_parse(*args, **kwargs)

    monkeypatch.setattr(recovery.requests, "get", fetch)
    monkeypatch.setattr(method_odds, "_parse_bfo_method_odds", tracked_parse)
    cache = recovery.RecoveryRunCache()
    target = _target_frame().iloc[0].to_dict()

    first = recovery.acquire_wayback_bfo(target, [url], cache=cache)
    cached = recovery.acquire_wayback_bfo(target, [url], cache=cache)

    archive_url = f"https://web.archive.org/web/{timestamp}id_/{url}"
    assert calls == ["https://web.archive.org/cdx/search/cdx", archive_url]
    assert len(parse_calls) == 2
    assert cached["evidence"][0] == first["evidence"][0]


def test_wayback_empty_cdx_result_is_not_cached(monkeypatch):
    url = "https://www.bestfightodds.com/events/ufc-test"
    calls = []

    class Response:
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return [["timestamp", "statuscode"]]

    def fetch(requested_url, **_kwargs):
        calls.append(requested_url)
        return Response()

    monkeypatch.setattr(recovery.requests, "get", fetch)
    cache = recovery.RecoveryRunCache()
    target = _target_frame().iloc[0].to_dict()

    recovery.acquire_wayback_bfo(target, [url], cache=cache)
    recovery.acquire_wayback_bfo(target, [url], cache=cache)

    assert calls == [
        "https://web.archive.org/cdx/search/cdx",
        "https://web.archive.org/cdx/search/cdx",
    ]


def test_event_url_reuse_requires_an_exact_target_parse():
    url = "https://www.bestfightodds.com/events/shared-event"

    def campaign(first_html):
        targets = recovery.prepare_targets(
            pd.DataFrame(
                [
                    {
                        "event_date": "2026-08-01", "event_id": "shared",
                        "event_title": "UFC Shared", "fighter_a": "Alpha Fighter",
                        "fighter_b": "Beta Fighter", "commence_time": "2026-08-01T22:00:00Z",
                        "source_url": url,
                        **_commence_provenance(
                            "Alpha Fighter", "Beta Fighter", "shared"
                        ),
                    },
                    {
                        "event_date": "2026-08-01", "event_id": "shared",
                        "event_title": "UFC Shared", "fighter_a": "Gamma Fighter",
                        "fighter_b": "Delta Fighter", "commence_time": "2026-08-01T22:00:00Z",
                        "source_url": "",
                        **_commence_provenance(
                            "Gamma Fighter", "Delta Fighter", "shared"
                        ),
                    },
                ]
            )
        )
        seen = []

        def acquire(target, known_urls):
            seen.append((target["fighter_a"], list(known_urls)))
            if target["fighter_a"] == "Alpha Fighter":
                evidence = _evidence(
                    "direct_bfo", "2026-07-30T12:00:00Z", first_html, url
                )
                evidence["original_source_url"] = url
                return _result([evidence])
            return _result(urls=[])

        recovery.run_recovery(
            targets,
            acquirers=(("direct_bfo", acquire),),
            run_cache=recovery.RecoveryRunCache(),
        )
        return seen

    exact = campaign(_scoped_html())
    rejected = campaign(_markerless_html())

    assert exact[1] == ("Gamma Fighter", [url])
    assert rejected[1] == ("Gamma Fighter", [])


def test_load_targets_uses_only_unique_verified_line_history_start(tmp_path):
    targets_path = tmp_path / "targets.csv"
    pd.DataFrame(
        [
            {
                "event_date": "2026-08-01",
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
            },
            {
                "event_date": "2026-08-01",
                "fighter_a": "Gamma Fighter",
                "fighter_b": "Delta Fighter",
            },
        ]
    ).to_csv(targets_path, index=False)

    line_dir = tmp_path / "line_history"
    line_dir.mkdir()
    common = {
        "bookmaker": "Example",
        "a_odds": 1.9,
        "b_odds": 2.0,
        "a_fair_prob": 0.51,
        "b_fair_prob": 0.49,
        "snapshot_time": "2026-07-30T00:00:00Z",
    }
    pd.DataFrame(
        [
            {
                **common,
                "event_id": "evt-1",
                "commence_time": "2026-08-02T02:00:00Z",
                "fighter_a": "Alpha Fighter",
                "fighter_b": "Beta Fighter",
            },
            {
                **common,
                "event_id": "evt-2a",
                "commence_time": "2026-08-02T01:00:00Z",
                "fighter_a": "Gamma Fighter",
                "fighter_b": "Delta Fighter",
            },
            {
                **common,
                "event_id": "evt-2b",
                "commence_time": "2026-08-02T03:00:00Z",
                "fighter_a": "Gamma Fighter",
                "fighter_b": "Delta Fighter",
            },
        ]
    ).to_csv(line_dir / "odds_20260730_000000.csv", index=False)

    loaded = recovery.load_targets(
        targets_path,
        event_url_map=None,
        line_history_dirs=[line_dir],
    )

    alpha = loaded.loc[loaded["fighter_a"] == "Alpha Fighter"].iloc[0]
    gamma = loaded.loc[loaded["fighter_a"] == "Gamma Fighter"].iloc[0]
    assert alpha["commence_time"] == "2026-08-02T02:00:00+00:00"
    assert alpha["commence_time_source_kind"] == "line_history"
    assert recovery._SHA256_RE.fullmatch(alpha["commence_time_source_sha256"])
    assert alpha["commence_time_source_matchup"] == (
        "Alpha Fighter vs Beta Fighter"
    )
    assert alpha["commence_time_semantics"] == "scheduled_bout_time"
    assert not recovery._text(gamma["commence_time"])


def test_load_targets_preserves_existing_event_name_for_run_local_url_reuse(tmp_path):
    targets_path = tmp_path / "targets.csv"
    pd.DataFrame(
        [{
            "event_date": "2026-08-01",
            "event_name": "UFC Shared Event",
            "fighter_a": "Alpha Fighter",
            "fighter_b": "Beta Fighter",
            "commence_time": "2026-08-01T22:00:00Z",
        }]
    ).to_csv(targets_path, index=False)

    loaded = recovery.load_targets(targets_path)

    assert loaded.loc[0, "event_title"] == "UFC Shared Event"


def test_historical_events_fill_reviewed_exact_commence_and_prefer_literal_pair():
    targets = recovery.prepare_targets(
        pd.DataFrame(
            [
                {
                    "event_date": "2026-03-14",
                    "fighter_a": "Bia Mesquita",
                    "fighter_b": "Montse Rendon",
                    "commence_time": "",
                },
                {
                    "event_date": "2026-05-30",
                    "fighter_a": "YiSak Lee",
                    "fighter_b": "Luis Felipe Dias",
                    "commence_time": "",
                },
            ]
        )
    )
    payloads = {
        "2026-03-13T12:00:00Z": {
            "timestamp": "2026-03-13T11:59:40Z",
            "data": [
                {
                    "id": "bia",
                    "home_team": "Beatriz Mesquita",
                    "away_team": "Montserrat Rendon",
                    "commence_time": "2026-03-14T20:00:00Z",
                }
            ],
        },
        "2026-05-29T12:00:00Z": {
            "timestamp": "2026-05-29T11:59:40Z",
            "data": [
                {
                    "id": "legal-alias",
                    "home_team": "Yi Sak Lee",
                    "away_team": "Luis Dias de Assis",
                    "commence_time": "2026-05-30T10:35:00Z",
                },
                {
                    "id": "literal-target",
                    "home_team": "Yi Sak Lee",
                    "away_team": "Luis Felipe Dias",
                    "commence_time": "2026-05-30T10:45:00Z",
                },
            ],
        },
    }

    class Response:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload
            self.content = json.dumps(payload, sort_keys=True).encode("utf-8")

        def json(self):
            return self._payload

    calls = []

    def request_get(_url, *, params, timeout):
        assert timeout == 30
        assert params["apiKey"] == "secret"
        calls.append(params["date"])
        return Response(payloads[params["date"]])

    enriched = recovery.enrich_commence_times_from_odds_api_history(
        targets,
        api_key="secret",
        request_get=request_get,
    )

    assert calls == ["2026-03-13T12:00:00Z", "2026-05-29T12:00:00Z"]
    assert enriched.loc[0, "commence_time"] == "2026-03-14T20:00:00+00:00"
    assert enriched.loc[0, "commence_time_source_event_id"] == "bia"
    assert enriched.loc[1, "commence_time"] == "2026-05-30T10:45:00+00:00"
    assert enriched.loc[1, "commence_time_source_event_id"] == "literal-target"
    assert set(enriched["commence_time_source_kind"]) == {
        "odds_api_historical_events"
    }
    assert enriched["commence_time_source_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert not enriched["commence_time_source_locator"].str.contains("secret").any()


@pytest.mark.parametrize(
    ("fighter_a", "fighter_b", "source_matchup"),
    [
        ("Diego Lopes", "Steve Garcia", "Diego Lopes vs Steve Garcia Jr."),
        ("Asu Almabayev", "Charles Johnson", "Asu Almabaev vs Charles Johnson"),
        ("Luke Riley", "Kai Kamaka III", "Kai Kamaka vs Luke Riley"),
        ("Ryan Gandra", "Zach Reese", "Ryan Gandra vs Zachary Reese"),
        ("Levi Rodrigues Jr.", "Felipe Franco", "Felipe Franco vs Levi Rodrigues"),
        (
            "Jean-Paul Lebosnoyani",
            "Seok Hyun Ko",
            "Jean-Paul Lebosnoyani vs Seokhyeon Ko",
        ),
    ],
)
def test_line_history_commence_provenance_accepts_reviewed_exact_name_variants(
    fighter_a,
    fighter_b,
    source_matchup,
):
    target = {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "commence_time": "2026-07-18T22:50:00+00:00",
        "commence_time_source_kind": "line_history",
        "commence_time_source_locator": "data/raw/line_history/reviewed.csv",
        "commence_time_source_observed_at": "2026-07-18T20:00:00+00:00",
        "commence_time_source_sha256": "a" * 64,
        "commence_time_source_matchup": source_matchup,
        "commence_time_semantics": "scheduled_bout_time",
    }

    assert recovery._commence_provenance_complete(target) is True


def test_jsonld_page_fills_only_exact_sports_event_commence():
    targets = recovery.prepare_targets(
        pd.DataFrame(
            [
                {
                    "event_date": "2026-05-28",
                    "fighter_a": "Xie Bin",
                    "fighter_b": "Yudi Cahyadi",
                    "commence_time": "",
                }
            ]
        )
    )
    payload = {
        "@context": "https://schema.org",
        "@type": "SportsEvent",
        "name": "Xie Bin vs Yudi Cahyadi",
        "startDate": "2026-05-28T10:00:00.000Z",
        "competitor": [
            {"@type": "Person", "name": "Xie Bin"},
            {"@type": "Person", "name": "Yudi Cahyadi"},
        ],
    }
    html = (
        '<html><script type="application/ld+json">'
        + json.dumps(payload)
        + "</script></html>"
    ).encode("utf-8")

    class Response:
        status_code = 200
        content = html

    enriched = recovery.enrich_commence_times_from_jsonld_pages(
        targets,
        ["https://example.test/event"],
        request_get=lambda *_args, **_kwargs: Response(),
    )

    assert enriched.loc[0, "commence_time"] == "2026-05-28T10:00:00+00:00"
    assert enriched.loc[0, "commence_time_source_kind"] == "jsonld_sports_event"
    assert enriched.loc[0, "commence_time_source_locator"] == "https://example.test/event"
    assert len(enriched.loc[0, "commence_time_source_sha256"]) == 64


def test_run_recovery_uses_only_unambiguous_v1_commence_metadata():
    targets = _target_frame(commence_time="")
    pair = recovery._pair_key("Alpha Fighter", "Beta Fighter")

    def entry(commence_time):
        return {
            "record": {
                "fighter_a": "Beta Fighter",
                "fighter_b": "Alpha Fighter",
                "commence_time": commence_time,
                "captured_at": "2026-07-30T00:00:00Z",
                # Deliberately untrusted; no page evidence means it cannot recover.
                "method_odds": {"a_ko_odds_prob": 0.99},
            },
            "snapshot_time": "2026-07-30T00:00:00Z",
            "snapshot_path": "schema-v1.json",
            "snapshot_sha256": "b" * 64,
        }

    seen_starts = []

    def no_evidence(target, _urls):
        seen_starts.append(target["commence_time"])
        return _result(error="offline")

    terminal, records = recovery.run_recovery(
        targets,
        v1_index={pair: [entry("2026-08-02T02:00:00Z")]},
        acquirers=(("direct_bfo", no_evidence),),
    )

    assert seen_starts == ["2026-08-02T02:00:00+00:00"]
    assert terminal.iloc[0]["terminal_reason"] == "no_exact_prefight_or_revalidated_evidence"
    assert records == []

    ambiguous, _ = recovery.run_recovery(
        targets,
        v1_index={
            pair: [
                entry("2026-08-02T02:00:00Z"),
                entry("2026-08-02T03:00:00Z"),
            ]
        },
        acquirers=(("direct_bfo", no_evidence),),
    )
    assert ambiguous.iloc[0]["terminal_reason"] == "missing_or_invalid_commence_time"
    assert ambiguous.iloc[0]["target_recovery_complete"] == False
    attempts = json.loads(ambiguous.iloc[0]["attempted_sources"])
    assert [attempt["pass"] for attempt in attempts] == ["revalidate", "direct"]
    assert {attempt["status"] for attempt in attempts} == {"precondition_failed"}


def test_recovery_exhausts_paths_until_exact_prefight_wayback_success():
    calls = []

    def direct(_target, _urls):
        calls.append("direct")
        return _result(
            [_evidence("direct_bfo", "2026-08-02T00:00:00Z", _markerless_html())]
        )

    def playwright(_target, _urls):
        calls.append("playwright")
        return _result(error="browser page had no exact block")

    def wayback(_target, _urls):
        calls.append("wayback")
        return _result(
            [
                _evidence(
                    "wayback_bfo",
                    "2026-07-31T21:59:59Z",
                    url=(
                        "https://web.archive.org/web/20260731215959id_/"
                        "https://www.bestfightodds.com/events/ufc-test"
                    ),
                )
            ]
        )

    terminal, records = recovery.run_recovery(
        _target_frame(),
        acquirers=(
            ("direct_bfo", direct),
            ("playwright_bfo", playwright),
            ("wayback_bfo", wayback),
        ),
    )

    assert calls == ["direct", "playwright", "wayback"]
    assert len(terminal) == 1
    assert terminal.loc[0, "status"] == "success"
    assert terminal.loc[0, "success_source"] == "wayback_bestfightodds_recovery"
    assert terminal.loc[0, "method_odds_complete"] == True
    assert len(records) == 1
    assert method_odds._record_has_v2_provenance(records[0])


def test_recovery_records_every_bounded_pass_after_early_success():
    calls = []

    def acquirer(name, result):
        def acquire(_target, _urls):
            calls.append(name)
            return result

        return acquire

    terminal, records = recovery.run_recovery(
        _target_frame(),
        acquirers=(
            (
                "direct_bfo",
                acquirer(
                    "direct",
                    _result([_evidence("direct_bfo", "2026-07-31T20:00:00Z")]),
                ),
            ),
            ("playwright_bfo", acquirer("playwright", _result())),
            ("wayback_bfo", acquirer("wayback", _result())),
        ),
    )

    assert calls == ["direct", "playwright", "wayback"]
    attempts = json.loads(terminal.loc[0, "attempted_sources"])
    assert recovery._attempt_pass_sequence(attempts) == recovery.RECOVERY_PASS_ORDER
    assert terminal.loc[0, "target_recovery_complete"] == True
    assert terminal.loc[0, "status"] == "success"
    assert len(records) == 1


def test_postfight_page_can_only_revalidate_exact_prefight_v1_values(tmp_path):
    snapshot_time = "2026-07-30T20:00:00Z"
    values = _parsed_values()
    v1_record = method_odds._snapshot_record(
        fighter_a="Alpha Fighter",
        fighter_b="Beta Fighter",
        method_odds=values,
        source="bestfightodds",
        captured_at=snapshot_time,
        event_id="evt-1",
        commence_time="2026-08-01T22:00:00Z",
        source_url="https://www.bestfightodds.com/events/ufc-test",
    )
    snapshot_path = tmp_path / "method_odds_20260730_200000.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_time": snapshot_time,
                "records": [v1_record],
            }
        ),
        encoding="utf-8",
    )
    v1_index = recovery.load_schema_v1_candidates([tmp_path])

    terminal, records = recovery.run_recovery(
        _target_frame(),
        v1_index=v1_index,
        acquirers=(
            (
                "direct_bfo",
                lambda _target, _urls: _result(
                    [_evidence("direct_bfo", "2026-08-02T12:00:00Z")]
                ),
            ),
        ),
    )

    assert terminal.loc[0, "status"] == "success"
    assert terminal.loc[0, "revalidated_schema_v1"] == True
    assert terminal.loc[0, "observation_captured_at"] == "2026-07-30T20:00:00+00:00"
    assert terminal.loc[0, "validation_captured_at"] == "2026-08-02T12:00:00+00:00"
    assert terminal.loc[0, "schema_v1_snapshot_path"] == str(snapshot_path)
    assert records[0]["revalidated_from_schema_version"] == 1
    assert method_odds._record_has_v2_provenance(records[0])


def test_postfight_page_does_not_revalidate_mismatched_v1_values(tmp_path):
    values = _parsed_values()
    values["a_ko_odds_prob"] = 0.99
    record = method_odds._snapshot_record(
        fighter_a="Alpha Fighter",
        fighter_b="Beta Fighter",
        method_odds=values,
        source="bestfightodds",
        captured_at="2026-07-30T20:00:00Z",
        event_id="evt-1",
        commence_time="2026-08-01T22:00:00Z",
    )
    (tmp_path / "method_odds_20260730_200000.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_time": "2026-07-30T20:00:00Z",
                "records": [record],
            }
        ),
        encoding="utf-8",
    )

    terminal, records = recovery.run_recovery(
        _target_frame(),
        v1_index=recovery.load_schema_v1_candidates([tmp_path]),
        acquirers=(
            (
                "direct_bfo",
                lambda _target, _urls: _result(
                    [_evidence("direct_bfo", "2026-08-02T12:00:00Z")]
                ),
            ),
            ("playwright_bfo", lambda _target, _urls: _result()),
            ("wayback_bfo", lambda _target, _urls: _result()),
        ),
    )

    assert terminal.loc[0, "status"] == "failed"
    assert terminal.loc[0, "terminal_reason"] == "no_exact_prefight_or_revalidated_evidence"
    assert records == []
    attempts = json.loads(terminal.loc[0, "attempted_sources"])
    direct_attempts = [attempt for attempt in attempts if attempt["pass"] == "direct"]
    assert direct_attempts[0]["detail"] == "post_fight_page_without_exact_prefight_v1_value_match"


def test_prefight_lead_boundary_is_exactly_24_hours():
    too_late, _ = recovery.run_recovery(
        _target_frame(),
        acquirers=(
            (
                "direct_bfo",
                lambda _target, _urls: _result(
                    [_evidence("direct_bfo", "2026-07-31T22:00:01Z")]
                ),
            ),
        ),
    )
    boundary, records = recovery.run_recovery(
        _target_frame(),
        acquirers=(
            (
                "direct_bfo",
                lambda _target, _urls: _result(
                    [_evidence("direct_bfo", "2026-07-31T22:00:00Z")]
                ),
            ),
        ),
    )

    assert too_late.loc[0, "status"] == "failed"
    assert boundary.loc[0, "status"] == "success"
    assert len(records) == 1


def test_terminal_report_has_one_failure_row_per_target_even_on_exceptions():
    targets = recovery.prepare_targets(
        pd.DataFrame(
            [
                {
                    "event_date": "2026-07-25",
                    "fighter_a": "One Alpha",
                    "fighter_b": "One Beta",
                    "commence_time": "2026-07-25T20:00:00Z",
                    **_commence_provenance("One Alpha", "One Beta", "one"),
                },
                {
                    "event_date": "2026-08-01",
                    "fighter_a": "Two Alpha",
                    "fighter_b": "Two Beta",
                    "commence_time": "2026-08-01T20:00:00Z",
                    **_commence_provenance("Two Alpha", "Two Beta", "two"),
                },
            ]
        )
    )

    def broken(_target, _urls):
        raise RuntimeError("offline")

    terminal, records = recovery.run_recovery(
        targets,
        acquirers=(("direct_bfo", broken),),
    )

    assert len(terminal) == len(targets) == terminal["target_key"].nunique()
    assert terminal["status"].tolist() == ["failed", "failed"]
    assert records == []


def test_recovery_terminal_does_not_persist_credentials_from_urls_or_errors():
    assert recovery._url_values(
        "https://user:password@example.test/event;"
        "https://example.test/event?token=secret"
    ) == []

    def broken(_target, _urls):
        raise RuntimeError("https://example.test/event?apiKey=secret")

    terminal, _ = recovery.run_recovery(
        _target_frame(),
        acquirers=(("direct_bfo", broken),),
    )

    attempts = terminal.loc[0, "attempted_sources"]
    assert "secret" not in attempts
    assert "apiKey" not in attempts
    assert "RuntimeError" in attempts


def _decision_frame(*, complete_rows, event_date="2026-08-01"):
    attempts = json.dumps(
        [
            {
                "pass": pass_name,
                "source": recovery._PASS_SOURCES[pass_name],
                "status": "no_evidence",
                "detail": "bounded pass completed without exact evidence",
            }
            for pass_name in recovery.RECOVERY_PASS_ORDER
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    rows = []
    for index in range(50):
        complete = index < complete_rows
        row = {
            "target_key": f"target-{index:02d}",
            "event_date": event_date,
            "commence_time": f"{event_date}T22:00:00Z",
            "established_fighters": True,
            "status": "success" if complete else "failed",
            "terminal_reason": (
                "validated_exact_prefight_method_odds"
                if complete
                else "no_exact_prefight_or_revalidated_evidence"
            ),
            "recovery_preconditions_met": True,
            "target_recovery_complete": True,
            "attempted_sources": attempts,
            "method_odds_complete": complete,
        }
        row.update(
            {
                column: 0.25 + (index / 10000) if complete else np.nan
                for column in recovery.METHOD_FEATURE_COLUMNS
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.attrs["passes_completed"] = list(recovery.RECOVERY_PASS_ORDER)
    return frame


def test_contract_decision_is_deterministic_211_at_fixed_gate_boundary():
    frame = _decision_frame(complete_rows=35)

    first = recovery.decide_feature_contract(frame, reference_date="2026-08-01")
    second = recovery.decide_feature_contract(frame, reference_date="2026-08-01")

    assert first == second
    assert first["gate_passed"] is True
    assert first["complete_method_coverage"] == 0.70
    assert first["selected_evaluation_contract"] == "full_live_contract_v6_integrity_211"
    assert first["selected_fullfit_contract"] == "full_live_contract_v6_integrity_211_fullfit"
    assert first["selected_feature_count"] == 211
    assert first["selected_spec_name"] == first["selected_evaluation_contract"]
    assert first["passes_completed"] == [
        "revalidate",
        "direct",
        "playwright",
        "wayback",
    ]
    assert first["targets_total"] == first["targets_terminal"] == 50
    assert first["all_targets_terminal"] is True
    assert first["all_targets_recovery_complete"] is True
    assert first["recovery_complete"] is True
    assert first["targets_recovery_complete"] == 50
    assert first["precondition_failures"] == 0
    assert first["selected_removed_features"] == []
    assert first["odds_noise"] == {"mode": "antithetic", "std": 0.06, "seed": 42}
    assert first["model_metrics_consulted"] is False


def test_contract_decision_selects_exact_205_fallback_for_coverage_or_lag():
    under_coverage = recovery.decide_feature_contract(
        _decision_frame(complete_rows=34), reference_date="2026-08-01"
    )
    stale = recovery.decide_feature_contract(
        _decision_frame(complete_rows=35, event_date="2026-06-22"),
        reference_date="2026-08-01",
    )

    for decision in (under_coverage, stale):
        assert decision["gate_passed"] is False
        assert decision["selected_evaluation_contract"] == "full_live_contract_v6_integrity_205"
        assert decision["selected_fullfit_contract"] == "full_live_contract_v6_integrity_205_fullfit"
        assert decision["selected_feature_count"] == 205
        assert decision["selected_removed_features"] == sorted(V4_METHOD_ODDS_FEATURE_COLS)
        assert len(decision["selected_removed_features"]) == 6
    assert under_coverage["criteria"]["minimum_complete_method_coverage"] is False
    assert stale["method_data_lag_days"] == 40
    assert stale["criteria"]["maximum_method_data_lag"] is False


def test_contract_decision_is_not_recovery_complete_before_all_four_passes():
    frame = _decision_frame(complete_rows=35)
    frame["attempted_sources"] = frame["attempted_sources"].map(
        lambda value: json.dumps(json.loads(value)[:-1])
    )

    decision = recovery.decide_feature_contract(frame, reference_date="2026-08-01")

    assert decision["all_targets_terminal"] is True
    assert decision["passes_completed"] == []
    assert decision["attempt_ledger_failures"] == 50
    assert decision["recovery_complete"] is False
    assert decision["gate_passed"] is False
    assert decision["selected_feature_count"] == 205


@pytest.mark.parametrize(
    "ledger",
    [
        "[]",
        "not-json",
        json.dumps(
            [
                {
                    "pass": "direct",
                    "source": "direct_bfo",
                    "status": "no_evidence",
                    "detail": "done",
                },
                {
                    "pass": "revalidate",
                    "source": "schema_v1",
                    "status": "no_evidence",
                    "detail": "done",
                },
                {
                    "pass": "playwright",
                    "source": "playwright_bfo",
                    "status": "no_evidence",
                    "detail": "done",
                },
                {
                    "pass": "wayback",
                    "source": "wayback_bfo",
                    "status": "no_evidence",
                    "detail": "done",
                },
            ]
        ),
        json.dumps(
            [
                {
                    "pass": pass_name,
                    "source": recovery._PASS_SOURCES[pass_name],
                    "status": "invented_status",
                    "detail": "done",
                }
                for pass_name in recovery.RECOVERY_PASS_ORDER
            ]
        ),
    ],
)
def test_contract_decision_rejects_unusable_attempt_ledgers(ledger):
    frame = _decision_frame(complete_rows=35)
    frame.loc[0, "attempted_sources"] = ledger

    decision = recovery.decide_feature_contract(frame, reference_date="2026-08-01")

    assert decision["attempt_ledger_failures"] == 1
    assert decision["targets_recovery_complete"] == 49
    assert decision["recovery_complete"] is False
    assert decision["gate_passed"] is False


def test_contract_decision_is_not_complete_when_one_target_failed_preconditions():
    frame = _decision_frame(complete_rows=35)
    frame.loc[0, "recovery_preconditions_met"] = False
    frame.loc[0, "target_recovery_complete"] = False
    frame.loc[0, "terminal_reason"] = "missing_or_invalid_commence_time"
    frame.loc[0, "attempted_sources"] = json.dumps(
        [
            {
                "pass": pass_name,
                "source": pass_name,
                "status": "precondition_failed",
                "detail": "missing_or_invalid_commence_time",
            }
            for pass_name in recovery.RECOVERY_PASS_ORDER
        ]
    )

    decision = recovery.decide_feature_contract(frame, reference_date="2026-08-01")

    assert decision["all_targets_terminal"] is True
    assert decision["all_targets_recovery_complete"] is False
    assert decision["targets_recovery_complete"] == 49
    assert decision["precondition_failures"] == 1
    assert decision["recovery_complete"] is False
    assert decision["gate_passed"] is False
    assert decision["selected_feature_count"] == 205


def test_publish_outputs_writes_terminal_snapshot_and_pre_metrics_decision(tmp_path):
    terminal = _decision_frame(complete_rows=35)
    terminal["fighter_a"] = "Alpha Fighter"
    terminal["fighter_b"] = "Beta Fighter"
    terminal["commence_time"] = "2026-08-01T22:00:00Z"
    decision = recovery.decide_feature_contract(terminal, reference_date="2026-08-01")

    paths = recovery.publish_outputs(tmp_path, terminal, [], decision)

    assert set(paths) == {"terminal", "snapshot", "decision"}
    assert len(pd.read_csv(paths["terminal"])) == 50
    snapshot = json.loads(paths["snapshot"].read_text(encoding="utf-8"))
    written_decision = json.loads(paths["decision"].read_text(encoding="utf-8"))
    assert snapshot["schema_version"] == 2
    assert snapshot["target_count"] == 50
    assert written_decision["selected_evaluation_contract"] == recovery.CONTRACT_211
    assert written_decision["terminal_report_sha256"] == recovery._sha256_bytes(
        paths["terminal"].read_bytes()
    )
    supplied_sha = written_decision.pop("decision_sha256")
    assert supplied_sha == recovery._sha256_bytes(
        json.dumps(
            written_decision, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def test_cli_requires_explicit_network_campaign_acknowledgement(tmp_path, capsys):
    result = recovery.main(
        [
            "--targets",
            str(tmp_path / "targets.csv"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert result == 2
    assert "--run-network-campaign" in capsys.readouterr().err
