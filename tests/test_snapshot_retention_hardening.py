import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.data import live_monitor, method_odds, rankings_scraper
from src.storage_retention import prune_json_snapshot_history
from src.strategy import llm_operator


NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _timestamp(days_ago: int, *, hour: int = 12) -> str:
    value = NOW - timedelta(days=days_ago)
    return value.replace(hour=hour).isoformat()


def test_json_snapshot_retention_keeps_recent_and_thins_older_days(tmp_path):
    recent_a = _write_json(
        tmp_path / "snap_recent_a.json",
        {"snapshot_time": _timestamp(5, hour=9)},
    )
    recent_b = _write_json(
        tmp_path / "snap_recent_b.json",
        {"snapshot_time": _timestamp(5, hour=18)},
    )
    daily_old = _write_json(
        tmp_path / "snap_daily_old.json",
        {"snapshot_time": _timestamp(40, hour=9)},
    )
    daily_new = _write_json(
        tmp_path / "snap_daily_new.json",
        {"snapshot_time": _timestamp(40, hour=18)},
    )
    expired = _write_json(
        tmp_path / "snap_expired.json",
        {"snapshot_time": _timestamp(401)},
    )
    malformed = tmp_path / "snap_malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    unrelated = _write_json(tmp_path / "canonical_history.json", {"history": True})

    removed = prune_json_snapshot_history(
        tmp_path,
        "snap_*.json",
        retention_days=400,
        full_resolution_days=30,
        daily_keep=1,
        now=NOW,
    )

    assert set(removed) == {daily_old, expired}
    assert recent_a.exists()
    assert recent_b.exists()
    assert daily_new.exists()
    assert malformed.exists()
    assert unrelated.exists()


def test_rankings_retention_preserves_newest_and_last_successful(tmp_path, monkeypatch):
    monkeypatch.setattr(rankings_scraper, "RANKINGS_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(rankings_scraper, "RANKINGS_SNAPSHOT_RETENTION_DAYS", 10)
    monkeypatch.setattr(rankings_scraper, "RANKINGS_SNAPSHOT_FULL_RESOLUTION_DAYS", 2)
    monkeypatch.setattr(rankings_scraper, "RANKINGS_SNAPSHOT_DAILY_KEEP", 1)

    superseded = _write_json(
        tmp_path / "rankings_20260608_120000.json",
        {
            "snapshot_time": _timestamp(40),
            "status": "success",
            "source": "ufc.com",
            "acquisition_failed": False,
        },
    )
    usable = _write_json(
        tmp_path / "rankings_20260618_120000.json",
        {
            "snapshot_time": _timestamp(30),
            "status": "success",
            "source": "ufc.com",
            "acquisition_failed": False,
        },
    )
    newest = _write_json(
        tmp_path / "rankings_20260628_120000.json",
        {
            "snapshot_time": _timestamp(20),
            "status": "failed",
            "source": "none",
            "acquisition_failed": True,
        },
    )
    canonical = _write_json(
        tmp_path / "rankings_history.json",
        {"snapshot_time": _timestamp(500), "history": []},
    )

    removed = rankings_scraper.prune_rankings_snapshots(now=NOW, force=True)

    assert removed == 1
    assert not superseded.exists()
    assert usable.exists()
    assert newest.exists()
    assert canonical.exists()


def test_method_odds_retention_preserves_newest_usable_and_canonical_data(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_RETENTION_DAYS", 10)
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_FULL_RESOLUTION_DAYS", 2)
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DAILY_KEEP", 1)

    superseded = _write_json(
        tmp_path / "method_odds_20260608_120000.json",
        {"snapshot_time": _timestamp(40), "records": [{"id": "old"}]},
    )
    usable = _write_json(
        tmp_path / "method_odds_20260618_120000.json",
        {"snapshot_time": _timestamp(30), "records": [{"id": "latest"}]},
    )
    newest = _write_json(
        tmp_path / "method_odds_20260628_120000.json",
        {"snapshot_time": _timestamp(20), "status": "failed", "records": []},
    )
    canonical = tmp_path / "historical_method_odds_all.csv"
    canonical.write_text("fighter_a,fighter_b\nAlpha,Beta\n", encoding="utf-8")
    checkpoint = _write_json(
        tmp_path / "method_odds_checkpoint.json",
        {"snapshot_time": _timestamp(500), "records": [{"id": "canonical"}]},
    )

    removed = method_odds.prune_method_odds_snapshots(now=NOW, force=True)

    assert removed == 1
    assert not superseded.exists()
    assert usable.exists()
    assert newest.exists()
    assert canonical.exists()
    assert checkpoint.exists()


def _operator_decision(number: int) -> llm_operator.OperatorDecision:
    return llm_operator.OperatorDecision(
        verdict="PASS",
        confidence=0.8,
        model_prob=0.6,
        operator_prob=None,
        rationale=f"decision-{number}-" + "x" * 180,
        research_summary={},
        risk_flags=[],
        timestamp=f"2026-07-18T12:{number:02d}:00+00:00",
        fighter_a=f"Fighter {number}",
        fighter_b="Opponent",
    )


def test_operator_decision_log_is_bounded_and_tail_reads_are_chronological(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "decision_log.jsonl"
    monkeypatch.setattr(llm_operator, "DECISION_LOG_PATH", path)
    monkeypatch.setattr(llm_operator, "OPERATOR_DECISION_LOG_MAX_BYTES", 2_500)

    for number in range(30):
        llm_operator._log_decision(_operator_decision(number))

    retained = llm_operator.load_decision_log(limit=None)
    latest_two = llm_operator.load_decision_log(limit=2)
    assert path.stat().st_size <= 2_500
    assert 0 < len(retained) < 30
    assert retained[-1]["fighter_a"] == "Fighter 29"
    assert [row["fighter_a"] for row in latest_two] == ["Fighter 28", "Fighter 29"]


def _write_card_snapshot(
    root: Path,
    name: str,
    *,
    event: str,
    event_date: str,
    captured: datetime,
    fights: list[dict] | None = None,
) -> Path:
    return _write_json(
        root / name,
        {
            "event": event,
            "event_date": event_date,
            "timestamp": captured.isoformat(),
            "fights": fights or [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
        },
    )


def test_card_write_deduplicates_and_prunes_expired_events(tmp_path, monkeypatch):
    monkeypatch.setattr(live_monitor, "SNAPSHOTS_DIR", tmp_path)
    monkeypatch.setattr(live_monitor, "_last_card_snapshot_prune_monotonic", 0.0)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_PAST_EVENT_RETENTION_DAYS", 30)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_UNKNOWN_DATE_RETENTION_DAYS", 180)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_MAX_PER_EVENT", 64)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_MAX_FILES", 1000)

    expired = _write_card_snapshot(
        tmp_path,
        "Expired_Event_20200101_000000.json",
        event="Expired Event",
        event_date="2020-01-01",
        captured=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    card = [{"fighter_a": "Current Alpha", "fighter_b": "Current Beta"}]

    first = live_monitor.save_card_snapshot("Current Event", card, event_date="2099-01-01")
    second = live_monitor.save_card_snapshot("Current Event", card, event_date="2099-01-01")

    assert first == second
    assert first.exists()
    assert len(list(tmp_path.glob("Current_Event_*.json"))) == 1
    assert not expired.exists()

    changed = live_monitor.save_card_snapshot(
        "Current Event",
        [*card, {"fighter_a": "Gamma", "fighter_b": "Delta"}],
        event_date="2099-01-01",
    )
    assert changed != first
    assert len(list(tmp_path.glob("Current_Event_*.json"))) == 2


def test_card_pruning_applies_age_and_caps_but_keeps_active_fallbacks(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_PAST_EVENT_RETENTION_DAYS", 30)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_UNKNOWN_DATE_RETENTION_DAYS", 180)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_MAX_PER_EVENT", 4)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_MAX_FILES", 3)

    unknown_old = _write_card_snapshot(
        tmp_path,
        "Unknown_20200101_000000.json",
        event="Unknown",
        event_date="",
        captured=NOW - timedelta(days=181),
    )
    event_paths = []
    for number in range(7):
        event_paths.append(
            _write_card_snapshot(
                tmp_path,
                f"Active_{number:02d}.json",
                event="Active Event",
                event_date="2099-01-01",
                captured=NOW - timedelta(hours=number),
                fights=[{"fighter_a": f"Alpha {number}", "fighter_b": "Beta"}],
            )
        )
    malformed = tmp_path / "manual_recovery.json"
    malformed.write_text("not-json", encoding="utf-8")

    removed = live_monitor.prune_card_snapshots(
        snapshot_dir=tmp_path,
        now=NOW.timestamp(),
        force=True,
    )

    survivors = list(tmp_path.glob("Active_*.json"))
    assert removed == 5
    assert not unknown_old.exists()
    assert len(survivors) == 3
    assert event_paths[0].exists()
    assert malformed.exists()


def test_card_global_cap_never_deletes_sole_active_event_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_MAX_PER_EVENT", 64)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_MAX_FILES", 1)
    for number in range(2):
        _write_card_snapshot(
            tmp_path,
            f"Event_{number}.json",
            event=f"Active Event {number}",
            event_date="2099-01-01",
            captured=NOW - timedelta(hours=number),
        )

    live_monitor.prune_card_snapshots(
        snapshot_dir=tmp_path,
        now=NOW.timestamp(),
        force=True,
    )

    assert len(list(tmp_path.glob("Event_*.json"))) == 2
