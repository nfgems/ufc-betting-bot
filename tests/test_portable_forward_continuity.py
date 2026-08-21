"""Portable regressions for the narrow Sunday forward-continuity slice."""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_sunday_pre_ufc_collection as sunday
from src.data import live_monitor
from src.data.name_utils import normalize_ufcstats_id
from src.data.practical_forward_collection import (
    build_pre_ufc_candidate_rows,
    build_upcoming_targets,
    diff_upcoming_targets,
    persist_upcoming_discovery,
)


NOW = "2026-08-23T06:00:00Z"
CARD = {
    "event": {
        "title": "UFC Portable",
        "date": "2026-08-29",
        "url": "http://ufcstats.com/event-details/portable",
    },
    "fights": [
        {
            "fighter_a": "Alpha Fighter",
            "fighter_a_id": "aaaaaaaaaaaaaaaa",
            "fighter_b": "Beta Fighter",
            "fighter_b_id": "bbbbbbbbbbbbbbbb",
            "weight_class": "Lightweight",
        }
    ],
}


def test_ufcstats_id_adapter_is_deliberately_minimal():
    assert normalize_ufcstats_id("AAAAAAAAAAAAAAAA") == "aaaaaaaaaaaaaaaa"
    assert (
        normalize_ufcstats_id(
            "http://ufcstats.com/fighter-details/BBBBBBBBBBBBBBBB"
        )
        == "bbbbbbbbbbbbbbbb"
    )
    assert normalize_ufcstats_id("not-a-stable-id") is None
    assert normalize_ufcstats_id("0aaaaaaaaaaaaaaaa") is None
    assert normalize_ufcstats_id("prefixaaaaaaaaaaaaaaaa") is None
    assert (
        normalize_ufcstats_id(
            "https://example.test/fighter-details/aaaaaaaaaaaaaaaa"
        )
        is None
    )


def test_live_snapshot_exposes_raw_body_and_original_retrieval_time(monkeypatch):
    event_url = "https://www.ufc.com/event/portable"
    retrieved = "2026-08-23T05:59:00Z"
    monkeypatch.setattr(
        live_monitor,
        "scrape_upcoming_events",
        lambda: [{"title": "UFC Portable", "date": "2026-08-29", "url": event_url}],
    )
    monkeypatch.setattr(
        live_monitor,
        "scrape_event_card",
        lambda _url: live_monitor._EventCardResult(
            [{"fighter_a": "Alpha Fighter", "fighter_b": "Beta Fighter"}],
            source_healthy=True,
        ),
    )
    live_monitor.clear_upcoming_event_cards_cache()
    live_monitor._UPSTREAM_HTML_CACHE.clear()
    live_monitor._UPSTREAM_HTML_CACHE[event_url] = (time.monotonic(), retrieved, "<card />")

    snapshot = live_monitor.collect_upcoming_event_cards_snapshot(force_refresh=True)

    assert snapshot["scan_complete"] is True
    assert snapshot["source_responses"] == [
        {"url": event_url, "retrieval_time_utc": retrieved, "body": "<card />"}
    ]
    assert snapshot["collected_at_utc"].endswith("Z")


def test_incomplete_discovery_persists_attempt_without_erasing_complete_baseline(tmp_path):
    complete = persist_upcoming_discovery(
        event_cards=[CARD],
        scan_complete=True,
        storage_root=tmp_path,
        observed_at=NOW,
        source_responses=[
            {
                "url": "https://www.ufc.com/event/portable",
                "retrieval_time_utc": NOW,
                "body": "<html>complete card</html>",
            }
        ],
    )
    baseline = (tmp_path / "latest-complete-upcoming-targets.csv").read_bytes()

    incomplete = persist_upcoming_discovery(
        event_cards=[],
        scan_complete=False,
        storage_root=tmp_path,
        observed_at="2026-08-23T07:00:00Z",
        source_responses=[],
    )

    assert complete["publication_complete"] is True
    snapshot_path = Path(complete["snapshot_path"])
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == complete[
        "snapshot_sha256"
    ]
    assert incomplete["publication_complete"] is False
    assert "upcoming_scan_incomplete" in incomplete["warning_codes"]
    assert "incomplete_scan_removals_suppressed" in incomplete["warning_codes"]
    assert not any(
        row["change_type"] == "fight_cancelled_or_removed"
        for row in incomplete["changes"]
    )
    assert (tmp_path / "latest-complete-upcoming-targets.csv").read_bytes() == baseline
    assert len(list((tmp_path / "raw" / "upcoming").glob("*.html"))) == 1


def test_same_stable_id_opponents_never_enter_candidate_queue():
    collision = {
        "event": CARD["event"],
        "fights": [
            {
                "fighter_a": "Alpha Fighter",
                "fighter_a_id": "aaaaaaaaaaaaaaaa",
                "fighter_b": "Beta Fighter",
                "fighter_b_id": "aaaaaaaaaaaaaaaa",
            }
        ],
    }
    targets, issues = build_upcoming_targets([collision], observed_at=NOW)
    rows, candidate_issues = build_pre_ufc_candidate_rows(
        targets,
        observed_at=NOW,
    )

    assert targets[0]["fighter_a_id"] == ""
    assert targets[0]["fighter_b_id"] == ""
    assert "same_stable_id_opponents" in {row["code"] for row in issues}
    assert rows == []
    assert candidate_issues == []


def test_each_start_time_move_has_an_exact_distinct_lifecycle_fact():
    baseline, _ = build_upcoming_targets([CARD], observed_at=NOW)
    first = [{**baseline[0], "scheduled_start_utc": "2026-08-29T22:00:00Z"}]
    second = [{**baseline[0], "scheduled_start_utc": "2026-08-29T23:00:00Z"}]

    first_changes, _ = diff_upcoming_targets(
        baseline,
        first,
        scan_complete=True,
        observed_at="2026-08-23T07:00:00Z",
    )
    second_changes, _ = diff_upcoming_targets(
        first,
        second,
        scan_complete=True,
        observed_at="2026-08-23T08:00:00Z",
    )

    assert first_changes[0]["scheduled_start_utc_before"] == ""
    assert first_changes[0]["scheduled_start_utc_after"] == "2026-08-29T22:00:00Z"
    assert second_changes[0]["scheduled_start_utc_before"] == "2026-08-29T22:00:00Z"
    assert second_changes[0]["scheduled_start_utc_after"] == "2026-08-29T23:00:00Z"
    assert first_changes[0]["change_id"] != second_changes[0]["change_id"]


def test_default_resolver_uses_only_reviewed_roster_url_binding(tmp_path):
    roster = tmp_path / "roster.csv"
    pd.DataFrame(
        [
            {
                "official_athlete_url": "https://ufc.com/athlete/alpha-fighter",
                "ufcstats_url": (
                    "http://ufcstats.com/fighter-details/aaaaaaaaaaaaaaaa"
                ),
            }
        ]
    ).to_csv(roster, index=False)
    resolver = sunday.build_stable_id_resolver(roster)

    assert (
        resolver("Alpha Fighter", "https://www.ufc.com/athlete/alpha-fighter/")
        == "aaaaaaaaaaaaaaaa"
    )
    assert resolver("Alpha Fighter", "https://www.ufc.com/athlete/namesake") is None


def test_default_card_sources_carry_only_direct_identity_evidence(
    tmp_path,
    monkeypatch,
):
    ufc_com_html = """
    <div class="c-listing-fight">
      <div class="c-listing-fight__names-row">
        <div class="c-listing-fight__corner-name--red">
          <a href="/athlete/alpha-fighter">Alpha Fighter</a>
        </div>
        <div class="c-listing-fight__corner-name--blue">
          <a href="https://www.ufc.com/athlete/beta-fighter">Beta Fighter</a>
        </div>
      </div>
      <div class="c-listing-fight__class-text">Lightweight Bout</div>
    </div>
    """
    monkeypatch.setattr(
        live_monitor,
        "_fetch_upstream_html",
        lambda *_args, **_kwargs: ufc_com_html,
    )
    card = live_monitor._scrape_ufc_com_event_card(
        "https://www.ufc.com/event/portable"
    )
    assert card[0]["fighter_a_athlete_url"] == (
        "https://www.ufc.com/athlete/alpha-fighter"
    )
    assert card[0]["fighter_b_athlete_url"] == (
        "https://www.ufc.com/athlete/beta-fighter"
    )

    roster = tmp_path / "roster.csv"
    pd.DataFrame(
        [
            {
                "official_athlete_url": "https://www.ufc.com/athlete/alpha-fighter",
                "ufcstats_url": "http://ufcstats.com/fighter-details/aaaaaaaaaaaaaaaa",
            },
            {
                "official_athlete_url": "https://www.ufc.com/athlete/beta-fighter",
                "ufcstats_url": "http://ufcstats.com/fighter-details/bbbbbbbbbbbbbbbb",
            },
        ]
    ).to_csv(roster, index=False)
    targets, issues = build_upcoming_targets(
        [{"event": CARD["event"], "fights": list(card)}],
        observed_at=NOW,
        fighter_id_resolver=sunday.build_stable_id_resolver(roster),
    )
    assert issues == []
    assert targets[0]["identity_status"] == "resolved"

    ufcstats_html = """
    <table><tr class="b-fight-details__table-row"><td>
      <a class="b-link" href="http://ufcstats.com/fighter-details/cccccccccccccccc">Gamma Fighter</a>
      <a class="b-link" href="http://ufcstats.com/fighter-details/dddddddddddddddd">Delta Fighter</a>
    </td><td>Welterweight Bout</td></tr></table>
    """
    monkeypatch.setattr(
        live_monitor,
        "_fetch_ufcstats_html",
        lambda *_args, **_kwargs: ufcstats_html,
    )
    fallback_card = live_monitor.scrape_event_card(
        "http://ufcstats.com/event-details/portable"
    )
    assert fallback_card[0]["fighter_a_id"] == "cccccccccccccccc"
    assert fallback_card[0]["fighter_b_id"] == "dddddddddddddddd"
    contexts = live_monitor._event_fight_contexts(CARD["event"], fallback_card)
    assert contexts[0]["fighter_a_id"] == "cccccccccccccccc"
    assert contexts[0]["fighter_b_id"] == "dddddddddddddddd"


def test_sunday_cycle_is_daily_only_locked_and_keeps_retryable_state_visible(tmp_path):
    root = tmp_path / "forward"
    invoked: list[Path] = []

    def pre_ufc(queue_path: Path, summary_path: Path):
        invoked.append(queue_path)
        payload = {"status": "incomplete", "warning_codes": ["retry_pending"], "exit_code": 2}
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    result = sunday.run_collection_cycle(
        mode="daily",
        execution_contract="sunday_pre_ufc",
        storage_root=root,
        now=NOW,
        completed_fights_path=tmp_path / "absent-fights.csv",
        roster_path=tmp_path / "absent-roster.csv",
        discovery_provider=lambda **_kwargs: {
            "event_cards": [CARD],
            "scan_complete": True,
            "collected_at_utc": NOW,
            "source_responses": [
                {"url": CARD["event"]["url"], "retrieval_time_utc": NOW, "body": "raw"}
            ],
        },
        context_enricher=lambda rows: [
            {**row, "commence_time": "2026-08-29T23:00:00Z"} for row in rows
        ],
        fighter_id_resolver=lambda _name, _url: None,
        pre_ufc_runner=pre_ufc,
    )

    assert result["status"] == "incomplete"
    assert result["mode"] == "daily"
    assert result["execution_contract"] == "sunday_pre_ufc"
    assert result["execution_contract_selected"] is True
    assert "hosted_schedule_activated" not in result
    assert "hosted_execution_authorized" not in result
    assert result["method_or_post_event_collection_performed"] is False
    assert invoked and invoked[0].exists()
    assert not (root / ".coordinator.lock").exists()
    assert (root / "latest-run-summary.json").exists()
    assert (root / "run-ledger.jsonl").exists()
    assert (root / "stage-attempt-ledger.jsonl").exists()
    with pytest.raises(ValueError, match="unsupported_sunday_collection_mode"):
        sunday.run_collection_cycle(
            mode="weekly",
            execution_contract="sunday_pre_ufc",
            storage_root=tmp_path / "rejected",
        )


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [("healthy", 0), ("incomplete", 2), ("failed", 1)],
)
def test_cli_maps_visible_health_to_exact_exit_codes(
    tmp_path, monkeypatch, status: str, exit_code: int
):
    summary_path = tmp_path / f"{status}.json"
    monkeypatch.setattr(
        sunday,
        "run_collection_cycle",
        lambda **_kwargs: {"status": status, "health_state": status},
    )
    assert (
        sunday.main(
            [
                "--mode",
                "daily",
                "--execution-contract",
                "sunday_pre_ufc",
                "--storage-root",
                str(tmp_path / "root"),
                "--summary-json",
                str(summary_path),
            ]
        )
        == exit_code
    )
    assert json.loads(summary_path.read_text(encoding="utf-8"))["status"] == status


def test_sunday_workflow_persists_state_before_enforcing_exit():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "pre-ufc-scrape.yml").read_text(
        encoding="utf-8"
    )
    active = "\n".join(
        line for line in workflow.splitlines() if line.strip() and not line.lstrip().startswith("#")
    )
    assert active.count("cron:") == 1
    assert "cron: '0 6 * * 0'" in active
    assert "scripts/run_sunday_pre_ufc_collection.py" in active
    assert "--mode daily" in active
    assert "--execution-contract sunday_pre_ufc" in active
    assert "--mode weekly" not in active
    assert "--mode method" not in active
    assert "--mode post-event" not in active
    assert "scripts/build_pre_ufc_career_supplement.py" not in active
    assert workflow.index("id: forward_collection") < workflow.index(
        "name: Commit durable discovery, retry, and supplement state if changed"
    ) < workflow.index("name: Enforce current-card and pre-UFC collection status")
    assert "data/raw/practical_forward_collection_v1/.coordinator.lock" in (
        root / ".gitignore"
    ).read_text(encoding="utf-8")
    runner = (root / "scripts" / "run_sunday_pre_ufc_collection.py").read_text(
        encoding="utf-8"
    )
    module = (root / "src" / "data" / "practical_forward_collection.py").read_text(
        encoding="utf-8"
    )
    forbidden = ("practical_method_odds", "phase4", "historical_runner")
    assert not any(token in (runner + module).casefold() for token in forbidden)


def test_invalid_now_still_persists_a_truthful_failure_summary(tmp_path):
    summary_path = tmp_path / "failed.json"

    assert (
        sunday.main(
            [
                "--mode",
                "daily",
                "--execution-contract",
                "sunday_pre_ufc",
                "--now",
                "not-a-timestamp",
                "--storage-root",
                str(tmp_path / "root"),
                "--summary-json",
                str(summary_path),
            ]
        )
        == 1
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["error_type"] == "ValueError"
    assert summary["execution_contract_selected"] is True
