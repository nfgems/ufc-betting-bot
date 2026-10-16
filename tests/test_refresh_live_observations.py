from datetime import datetime, timezone
from types import SimpleNamespace

from src.data.incremental_refresh import INDEX_URL
from src.data.refresh_predictions import UPCOMING_URL, parse_event_card
from src.data.refresh_live_observations import observe_publication
from test_refresh_predictions import card, EVENT, IDS


def fixture(tmp_path):
    event = {"title": "UFC Test Event", "event_date": "2026-10-17", "url": EVENT}
    rows = parse_event_card(card(), event)
    csv = tmp_path / "fights.csv"
    csv.write_text("event_date\n2026-10-10\n", encoding="utf-8")
    consumer = SimpleNamespace(rows=rows, snapshot={"fights_path": str(csv)},
        models={"primary": {}}, service=SimpleNamespace(baseline=tmp_path))
    index = '<table><tr class="b-statistics__table-row"><td><a class="b-link" href="{}">{}</a><span class="b-statistics__date">{}</span></td></tr></table>'
    responses = {UPCOMING_URL: index.format(EVENT, event["title"], "October 17, 2026"),
                 EVENT: card(), INDEX_URL: index.format(EVENT, event["title"], "October 10, 2026")}
    return consumer, responses, index


def test_current_official_card_and_request_reuse(tmp_path):
    consumer, responses, _ = fixture(tmp_path)
    calls = []
    def fetch(url):
        calls.append(url)
        return responses[url]
    now = datetime(2026, 10, 11, 15, tzinfo=timezone.utc)
    result = observe_publication(consumer, tmp_path / "observations", now=now, fetcher=fetch)
    assert result["official_card_matches_publication"] is True
    assert result["trade_block_reasons"] == []
    assert result["processed_snapshot_max_event_date"] == "2026-10-10"
    observe_publication(consumer, tmp_path / "observations", now=now, fetcher=fetch)
    assert len(calls) == 3


def test_changed_identity_and_completed_event_block(tmp_path):
    consumer, responses, index = fixture(tmp_path)
    responses[EVENT] = card().replace(IDS[1], "3333333333333333")
    responses[INDEX_URL] = index.format(EVENT, "UFC Completed", "October 12, 2026")
    result = observe_publication(consumer, tmp_path / "observations",
        now=datetime(2026, 10, 13, 15, tzinfo=timezone.utc), fetcher=responses.__getitem__)
    assert result["trade_block_reasons"] == ["current_official_card_changed", "final_publication_freshness_blocked"]


def test_model_age_retains_original_guard(tmp_path):
    consumer, responses, _ = fixture(tmp_path)
    (tmp_path / "staging_manifest.json").write_text('{"built_at":"2020-01-01T00:00:00Z"}')
    result = observe_publication(consumer, tmp_path / "observations",
        now=datetime(2026, 10, 11, 15, tzinfo=timezone.utc), fetcher=responses.__getitem__)
    assert result["trade_block_reasons"] == ["final_publication_freshness_blocked"]
