from pathlib import Path

import pandas as pd
import pytest
import requests

import src.data.betsapi_mma as betsapi_mma


class _FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)
        return None

    def json(self):
        return self._payload


def test_betsapi_mma_client_uses_expected_versions_and_params(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def fake_get(url, params=None, timeout=30):
        calls.append((url, dict(params or {})))
        return _FakeResponse({"success": 1, "results": []})

    monkeypatch.setattr(betsapi_mma.requests, "get", fake_get)

    client = betsapi_mma.BetsApiMMAClient(token="test-token")
    client.get_mma_upcoming_events(page=2, day="20260313")
    client.get_mma_ended_events(page=3)
    client.get_bet365_mma_upcoming(page=4, day="20260314")
    client.get_bet365_mma_prematch("fi-123")
    client.get_event_odds_summary("evt-9")

    assert calls[0][0].endswith("/v3/events/upcoming")
    assert calls[0][1]["sport_id"] == betsapi_mma.BETSAPI_MMA_SPORT_ID
    assert calls[0][1]["page"] == 2
    assert calls[0][1]["day"] == "20260313"
    assert calls[0][1]["token"] == "test-token"

    assert calls[1][0].endswith("/v3/events/ended")
    assert calls[1][1]["page"] == 3

    assert calls[2][0].endswith("/v1/bet365/upcoming")
    assert calls[2][1]["page"] == 4
    assert calls[2][1]["day"] == "20260314"

    assert calls[3][0].endswith("/v4/bet365/prematch")
    assert calls[3][1]["FI"] == "fi-123"

    assert calls[4][0].endswith("/v2/event/odds/summary")
    assert calls[4][1]["event_id"] == "evt-9"
    assert calls[4][1]["source"] == "bet365"


def test_betsapi_mma_client_retries_retryable_http_status(monkeypatch):
    calls: list[int] = []
    sleeps: list[float] = []

    def fake_get(url, params=None, timeout=30):
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            return _FakeResponse({"success": 0}, status_code=502)
        return _FakeResponse({"success": 1, "results": []})

    monkeypatch.setattr(betsapi_mma.requests, "get", fake_get)
    monkeypatch.setattr(betsapi_mma.time, "sleep", lambda seconds: sleeps.append(seconds))

    client = betsapi_mma.BetsApiMMAClient(token="test-token")
    payload = client.get_mma_ended_events(page=1, day="20220318")

    assert payload["success"] == 1
    assert len(calls) == 2
    assert sleeps == [1.0]


def test_betsapi_mma_client_retries_request_exception(monkeypatch):
    calls: list[int] = []
    sleeps: list[float] = []

    def fake_get(url, params=None, timeout=30):
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            raise requests.ConnectionError("temporary outage")
        return _FakeResponse({"success": 1, "results": []})

    monkeypatch.setattr(betsapi_mma.requests, "get", fake_get)
    monkeypatch.setattr(betsapi_mma.time, "sleep", lambda seconds: sleeps.append(seconds))

    client = betsapi_mma.BetsApiMMAClient(token="test-token")
    payload = client.get_event_odds_summary("evt-9")

    assert payload["success"] == 1
    assert len(calls) == 2
    assert sleeps == [1.0]


def test_betsapi_mma_client_error_does_not_expose_tokenized_url(monkeypatch):
    def fake_get(url, params=None, timeout=30):
        response = _FakeResponse({"success": 0}, status_code=403)
        error = requests.HTTPError(
            f"403 for {url}?token={params['token']}", response=response
        )
        response.raise_for_status = lambda: (_ for _ in ()).throw(error)
        return response

    monkeypatch.setattr(betsapi_mma.requests, "get", fake_get)
    client = betsapi_mma.BetsApiMMAClient(token="do-not-leak")

    with pytest.raises(RuntimeError) as exc_info:
        client.get_mma_ended_events()

    assert "do-not-leak" not in str(exc_info.value)
    assert "token=" not in str(exc_info.value)
    assert "HTTP 403" in str(exc_info.value)


def test_normalize_query_date_returns_naive_normalized_day():
    aware = pd.Timestamp("2026-03-15 13:45:00", tz="UTC")

    normalized = betsapi_mma._normalize_query_date(aware)

    assert normalized == pd.Timestamp("2026-03-15")
    assert normalized.tzinfo is None


def test_snapshot_mma_payloads_uses_bet365_upcoming_id_as_prematch_fi(monkeypatch, tmp_path: Path):
    written: list[Path] = []

    def fake_write_raw_payload(directory, payload, stem, metadata=None):
        path = tmp_path / f"{stem}.json"
        written.append(path)
        return path

    class FakeClient:
        def __init__(self):
            self.prematch_fis: list[str] = []

        def get_mma_upcoming_events(self, page=1, day=None):
            return {"success": 1, "results": []}

        def get_bet365_mma_upcoming(self, page=1, day=None):
            return {
                "success": 1,
                "results": [
                    {
                        "id": "fi-123",
                        "sport_id": betsapi_mma.BETSAPI_MMA_SPORT_ID,
                        "league": {"name": "UFC Fight Night"},
                        "home": {"name": "Fighter A"},
                        "away": {"name": "Fighter B"},
                        "time": 1773360000,
                    }
                ],
            }

        def get_bet365_mma_prematch(self, fi):
            self.prematch_fis.append(fi)
            return {"success": 1, "results": {"FI": fi, "event_id": "evt-1", "main": {"sp": {}}}}

    fake_client = FakeClient()
    monkeypatch.setattr(betsapi_mma, "_write_raw_payload", fake_write_raw_payload)

    results = betsapi_mma.snapshot_mma_payloads(client=fake_client, max_prematch=1, ufc_only=True)

    assert fake_client.prematch_fis == ["fi-123"]
    assert "prematch_fi-123" in results
    assert results["prematch_fi-123"].payload["results"]["FI"] == "fi-123"
    assert any(path.name == "bet365_prematch_fi_fi-123.json" for path in written)


def test_snapshot_mma_payloads_hydrates_odds_summary_for_list_wrapped_prematch_results(
    monkeypatch,
    tmp_path: Path,
):
    written: list[Path] = []

    def fake_write_raw_payload(directory, payload, stem, metadata=None):
        path = tmp_path / f"{stem}.json"
        written.append(path)
        return path

    class FakeClient:
        def __init__(self):
            self.prematch_fis: list[str] = []
            self.summary_event_ids: list[str] = []

        def get_mma_upcoming_events(self, page=1, day=None):
            return {"success": 1, "results": []}

        def get_bet365_mma_upcoming(self, page=1, day=None):
            return {
                "success": 1,
                "results": [
                    {
                        "id": "fi-123",
                        "sport_id": betsapi_mma.BETSAPI_MMA_SPORT_ID,
                        "league": {"name": "UFC Fight Night"},
                        "home": {"name": "Fighter A"},
                        "away": {"name": "Fighter B"},
                        "time": 1773360000,
                    }
                ],
            }

        def get_bet365_mma_prematch(self, fi):
            self.prematch_fis.append(fi)
            return {
                "success": 1,
                "results": [
                    {
                        "FI": fi,
                        "event_id": "evt-1",
                        "main": {"sp": {}},
                    }
                ],
            }

        def get_event_odds_summary(self, event_id, source="bet365"):
            self.summary_event_ids.append(event_id)
            return {"success": 1, "results": [{"event_id": event_id, "source": source}]}

    fake_client = FakeClient()
    monkeypatch.setattr(betsapi_mma, "_write_raw_payload", fake_write_raw_payload)

    results = betsapi_mma.snapshot_mma_payloads(
        client=fake_client,
        max_prematch=1,
        ufc_only=True,
        hydrate_odds_summary=True,
    )

    assert fake_client.prematch_fis == ["fi-123"]
    assert fake_client.summary_event_ids == ["evt-1"]
    assert "odds_summary_evt-1" in results
    assert results["prematch_fi-123"].payload["results"][0]["FI"] == "fi-123"
    assert any(path.name == "odds_summary_event_evt-1.json" for path in written)


def test_backfill_betsapi_mma_history_saves_processed_artifacts(tmp_path: Path):
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"

    class FakeClient:
        def __init__(self):
            self.summary_event_ids: list[str] = []

        def get_mma_ended_events(self, page=1, day=None):
            assert page == 1
            return {
                "success": 1,
                "pager": {"page": 1, "total_page": 1},
                "results": [
                    {
                        "id": "evt-1",
                        "sport_id": betsapi_mma.BETSAPI_MMA_SPORT_ID,
                        "league": {"name": "UFC Fight Night"},
                        "home": {"name": "Fighter A"},
                        "away": {"name": "Fighter B"},
                        "time": 1773360000,
                    }
                ],
            }

        def get_event_odds_summary(self, event_id, source="bet365"):
            self.summary_event_ids.append(str(event_id))
            return {
                "success": 1,
                "results": {
                    "Bet365": {
                        "matching_dir": 1,
                        "odds_update": {"start": "1773352800"},
                        "odds": {
                            "start": {
                                "1_1": {
                                    "id": "line-1",
                                    "home_od": "1.80",
                                    "away_od": "2.20",
                                    "add_time": "1773351000",
                                }
                            }
                        },
                    }
                },
            }

    result = betsapi_mma.backfill_betsapi_mma_history(
        start_date="2026-03-13",
        end_date="2026-03-13",
        client=FakeClient(),
        hydrate_odds_summary=True,
        raw_root=raw_root,
        processed_root=processed_root,
    )

    assert result["event_rows"] == 1
    assert result["summary_rows"] == 1
    assert result["event_snapshots"] == 1
    assert Path(result["events_path"]).exists()
    assert Path(result["summary_rows_path"]).exists()
    assert Path(result["event_snapshots_path"]).exists()
    assert Path(result["manifest_path"]).exists()

    snapshots = pd.read_csv(result["event_snapshots_path"])
    assert snapshots.loc[0, "opening_bookmakers"] == 1


def test_backfill_betsapi_mma_history_ufc_only_limits_summary_hydration(tmp_path: Path):
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"

    class FakeClient:
        def __init__(self):
            self.summary_event_ids: list[str] = []

        def get_mma_ended_events(self, page=1, day=None):
            return {
                "success": 1,
                "pager": {"page": 1, "total_page": 1},
                "results": [
                    {
                        "id": "evt-ufc",
                        "sport_id": betsapi_mma.BETSAPI_MMA_SPORT_ID,
                        "league": {"name": "UFC Fight Night"},
                        "home": {"name": "Fighter A"},
                        "away": {"name": "Fighter B"},
                        "time": 1773360000,
                    },
                    {
                        "id": "evt-bellator",
                        "sport_id": betsapi_mma.BETSAPI_MMA_SPORT_ID,
                        "league": {"name": "Bellator"},
                        "home": {"name": "Fighter C"},
                        "away": {"name": "Fighter D"},
                        "time": 1773363600,
                    },
                ],
            }

        def get_event_odds_summary(self, event_id, source="bet365"):
            self.summary_event_ids.append(str(event_id))
            return {
                "success": 1,
                "results": {
                    "Bet365": {
                        "matching_dir": 1,
                        "odds_update": {"start": "1773352800"},
                        "odds": {
                            "start": {
                                "1_1": {
                                    "id": f"line-{event_id}",
                                    "home_od": "1.80",
                                    "away_od": "2.20",
                                    "add_time": "1773351000",
                                }
                            }
                        },
                    }
                },
            }

    fake_client = FakeClient()
    result = betsapi_mma.backfill_betsapi_mma_history(
        start_date="2026-03-13",
        end_date="2026-03-13",
        client=fake_client,
        hydrate_odds_summary=True,
        ufc_only=True,
        raw_root=raw_root,
        processed_root=processed_root,
    )

    assert result["event_rows"] == 2
    assert result["summary_rows"] == 1
    assert fake_client.summary_event_ids == ["evt-ufc"]
