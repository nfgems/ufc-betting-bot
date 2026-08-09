import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.data import live_monitor, method_odds, rankings_scraper
from src.storage_retention import prune_json_snapshot_history


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


def test_json_snapshot_retention_prefers_matching_daily_survivor(tmp_path):
    usable = _write_json(
        tmp_path / "snap_mixed_usable.json",
        {"snapshot_time": _timestamp(40, hour=9), "status": "success"},
    )
    later_failed = _write_json(
        tmp_path / "snap_mixed_failed.json",
        {"snapshot_time": _timestamp(40, hour=18), "status": "failed"},
    )
    earlier_failed = _write_json(
        tmp_path / "snap_failed_only_early.json",
        {"snapshot_time": _timestamp(50, hour=9), "status": "failed"},
    )
    latest_failed = _write_json(
        tmp_path / "snap_failed_only_latest.json",
        {"snapshot_time": _timestamp(50, hour=18), "status": "failed"},
    )
    recent_usable = _write_json(
        tmp_path / "snap_recent_usable.json",
        {"snapshot_time": _timestamp(5), "status": "success"},
    )

    removed = prune_json_snapshot_history(
        tmp_path,
        "snap_*.json",
        retention_days=400,
        full_resolution_days=30,
        daily_keep=1,
        protect_latest_matching=lambda payload: payload.get("status") == "success",
        prefer_daily_matching=lambda payload: payload.get("status") == "success",
        now=NOW,
    )

    assert set(removed) == {later_failed, earlier_failed}
    assert usable.exists()
    assert latest_failed.exists()
    assert recent_usable.exists()


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


def test_rankings_retention_prefers_successful_snapshot_within_each_day(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(rankings_scraper, "RANKINGS_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(rankings_scraper, "RANKINGS_SNAPSHOT_RETENTION_DAYS", 100)
    monkeypatch.setattr(rankings_scraper, "RANKINGS_SNAPSHOT_FULL_RESOLUTION_DAYS", 30)
    monkeypatch.setattr(rankings_scraper, "RANKINGS_SNAPSHOT_DAILY_KEEP", 1)

    usable = _write_json(
        tmp_path / "rankings_20260608_090000.json",
        {
            "snapshot_time": _timestamp(40, hour=9),
            "status": "success",
            "source": "ufc.com",
            "acquisition_failed": False,
        },
    )
    later_failed = _write_json(
        tmp_path / "rankings_20260608_180000.json",
        {
            "snapshot_time": _timestamp(40, hour=18),
            "status": "failed",
            "source": "none",
            "acquisition_failed": True,
        },
    )
    recent_usable = _write_json(
        tmp_path / "rankings_20260713_120000.json",
        {
            "snapshot_time": _timestamp(5),
            "status": "success",
            "source": "ufc.com",
            "acquisition_failed": False,
        },
    )

    removed = rankings_scraper.prune_rankings_snapshots(now=NOW, force=True)

    assert removed == 1
    assert usable.exists()
    assert not later_failed.exists()
    assert recent_usable.exists()


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


def test_method_odds_retention_prefers_usable_snapshot_within_each_day(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_RETENTION_DAYS", 100)
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_FULL_RESOLUTION_DAYS", 14)
    monkeypatch.setattr(method_odds, "METHOD_ODDS_SNAPSHOT_DAILY_KEEP", 1)

    usable = _write_json(
        tmp_path / "method_odds_20260608_090000.json",
        {
            "snapshot_time": _timestamp(40, hour=9),
            "status": "success",
            "records": [{"id": "usable"}],
        },
    )
    later_failed = _write_json(
        tmp_path / "method_odds_20260608_180000.json",
        {
            "snapshot_time": _timestamp(40, hour=18),
            "status": "failed",
            "records": [],
        },
    )
    recent_usable = _write_json(
        tmp_path / "method_odds_20260713_120000.json",
        {
            "snapshot_time": _timestamp(5),
            "status": "success",
            "records": [{"id": "recent"}],
        },
    )

    removed = method_odds.prune_method_odds_snapshots(now=NOW, force=True)

    assert removed == 1
    assert usable.exists()
    assert not later_failed.exists()
    assert recent_usable.exists()


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


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "not-a-date",
        ["2026-01-01"],
        {"year": 2026, "month": 1, "day": 1},
        1_720_000_000,
        1.5,
        True,
    ],
)
def test_card_snapshot_epoch_rejects_unsupported_or_invalid_values(value):
    assert live_monitor._card_snapshot_epoch(value) is None


def test_card_snapshot_epoch_parses_strings_and_guards_post_parse_failures(monkeypatch):
    value = "2026-01-01T00:00:00Z"
    assert live_monitor._card_snapshot_epoch(value) == datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    ).timestamp()

    def fail_isna(_value):
        raise RuntimeError("post-parse failure")

    monkeypatch.setattr(live_monitor.pd, "isna", fail_isna)
    assert live_monitor._card_snapshot_epoch(value) is None


def test_card_snapshot_epoch_does_not_swallow_memory_error(monkeypatch):
    def fail_parse(*_args, **_kwargs):
        raise MemoryError("out of memory")

    monkeypatch.setattr(live_monitor.pd, "to_datetime", fail_parse)
    with pytest.raises(MemoryError, match="out of memory"):
        live_monitor._card_snapshot_epoch("2026-01-01")


@pytest.mark.parametrize(
    "invalid_timestamp",
    [
        ["2026-01-01"],
        {"value": "2026-01-01"},
    ],
)
def test_card_invalid_timestamp_uses_mtime_and_remains_cap_eligible(
    tmp_path,
    monkeypatch,
    invalid_timestamp,
):
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_PAST_EVENT_RETENTION_DAYS", 30)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_UNKNOWN_DATE_RETENTION_DAYS", 180)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_MAX_PER_EVENT", 1)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_MAX_FILES", 1000)

    invalid = _write_json(
        tmp_path / "Active_invalid.json",
        {
            "event": "Active Event",
            "event_date": "2099-01-01",
            "timestamp": invalid_timestamp,
            "fights": [],
        },
    )
    newest_mtime = (NOW - timedelta(hours=1)).timestamp()
    os.utime(invalid, (newest_mtime, newest_mtime))
    valid_older = _write_card_snapshot(
        tmp_path,
        "Active_valid_older.json",
        event="Active Event",
        event_date="2099-01-01",
        captured=NOW - timedelta(hours=2),
    )

    removed = live_monitor.prune_card_snapshots(
        snapshot_dir=tmp_path,
        now=NOW.timestamp(),
        force=True,
    )

    assert removed == 1
    assert invalid.exists()
    assert not valid_older.exists()


@pytest.mark.parametrize(
    "invalid_event_date",
    [
        ["2099-01-01"],
        {"value": "2099-01-01"},
    ],
)
def test_card_invalid_event_date_uses_unknown_date_retention(
    tmp_path,
    monkeypatch,
    invalid_event_date,
):
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_PAST_EVENT_RETENTION_DAYS", 30)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_UNKNOWN_DATE_RETENTION_DAYS", 180)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_MAX_PER_EVENT", 64)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_MAX_FILES", 1000)

    invalid = _write_json(
        tmp_path / "Unknown_invalid.json",
        {
            "event": "Unknown Date Event",
            "event_date": invalid_event_date,
            "timestamp": (NOW - timedelta(days=181)).isoformat(),
            "fights": [],
        },
    )

    removed = live_monitor.prune_card_snapshots(
        snapshot_dir=tmp_path,
        now=NOW.timestamp(),
        force=True,
    )

    assert removed == 1
    assert not invalid.exists()


def test_card_scan_exception_preserves_file_and_does_not_abort_peers(
    tmp_path,
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_PAST_EVENT_RETENTION_DAYS", 30)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_UNKNOWN_DATE_RETENTION_DAYS", 180)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_MAX_PER_EVENT", 64)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_MAX_FILES", 1000)
    expired = _write_card_snapshot(
        tmp_path,
        "Expired_healthy.json",
        event="Expired Healthy Event",
        event_date="2020-01-01",
        captured=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    poisoned = _write_json(
        tmp_path / "Poisoned_runtime_error.json",
        {
            "event": "Unique Poison Event",
            "event_date": "raise-runtime-error",
            "timestamp": NOW.isoformat(),
            "fights": [],
        },
    )
    original_epoch = live_monitor._card_snapshot_epoch

    def selectively_fail(value):
        if value == "raise-runtime-error":
            raise RuntimeError("injected per-file failure")
        return original_epoch(value)

    monkeypatch.setattr(live_monitor, "_card_snapshot_epoch", selectively_fail)
    caplog.set_level(logging.WARNING, logger=live_monitor.__name__)

    removed = live_monitor.prune_card_snapshots(
        snapshot_dir=tmp_path,
        now=NOW.timestamp(),
        force=True,
    )

    assert removed == 1
    assert not expired.exists()
    assert poisoned.exists()
    assert poisoned.name in caplog.text
    assert "injected per-file failure" in caplog.text
    assert "skipped=1" in caplog.text


def test_card_retention_logs_skipped_and_fallback_counts_without_removal(
    tmp_path,
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_PAST_EVENT_RETENTION_DAYS", 30)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_UNKNOWN_DATE_RETENTION_DAYS", 180)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_MAX_PER_EVENT", 64)
    monkeypatch.setattr(live_monitor, "CARD_SNAPSHOT_MAX_FILES", 1000)
    malformed = tmp_path / "Malformed.json"
    malformed.write_text("not-json", encoding="utf-8")
    missing_event = _write_json(
        tmp_path / "Missing_event.json",
        {
            "event_date": "2099-01-01",
            "timestamp": NOW.isoformat(),
            "fights": [],
        },
    )
    fallback = _write_json(
        tmp_path / "Fallback_metadata.json",
        {
            "event": "Active Fallback Event",
            "event_date": ["2099-01-01"],
            "timestamp": [NOW.isoformat()],
            "fights": [],
        },
    )
    caplog.set_level(logging.WARNING, logger=live_monitor.__name__)

    removed = live_monitor.prune_card_snapshots(
        snapshot_dir=tmp_path,
        now=NOW.timestamp(),
        force=True,
    )

    assert removed == 0
    assert malformed.exists()
    assert missing_event.exists()
    assert fallback.exists()
    assert malformed.name in caplog.text
    assert missing_event.name in caplog.text
    assert fallback.name in caplog.text
    assert "removed=0, skipped=2, metadata_fallbacks=1" in caplog.text


@pytest.mark.parametrize("error_type", [MemoryError, SystemExit])
def test_card_retention_propagates_resource_and_process_control_failures(
    tmp_path,
    monkeypatch,
    error_type,
):
    snapshot = _write_json(
        tmp_path / "Fatal_failure.json",
        {
            "event": "Fatal Failure Event",
            "event_date": "raise-fatal-error",
            "timestamp": NOW.isoformat(),
            "fights": [],
        },
    )
    original_epoch = live_monitor._card_snapshot_epoch

    def selectively_fail(value):
        if value == "raise-fatal-error":
            raise error_type("fatal retention failure")
        return original_epoch(value)

    monkeypatch.setattr(live_monitor, "_card_snapshot_epoch", selectively_fail)

    with pytest.raises(error_type, match="fatal retention failure"):
        live_monitor.prune_card_snapshots(
            snapshot_dir=tmp_path,
            now=NOW.timestamp(),
            force=True,
        )

    assert snapshot.exists()


@pytest.mark.parametrize("unclassifiable_event", [None, ""])
def test_latest_card_snapshot_skips_unclassifiable_event_and_uses_exact_match(
    tmp_path,
    unclassifiable_event,
):
    exact = _write_json(
        tmp_path / "Target_Event_20260719_120000.json",
        {
            "event": "Target Event",
            "event_date": "2099-01-01",
            "timestamp": NOW.isoformat(),
            "fights": [{"fighter_a": "Exact", "fighter_b": "Match"}],
        },
    )
    unclassifiable_payload = {
        "event_date": "2099-01-01",
        "timestamp": NOW.isoformat(),
        "fights": [{"fighter_a": "Wrong", "fighter_b": "Snapshot"}],
    }
    if unclassifiable_event is not None:
        unclassifiable_payload["event"] = unclassifiable_event
    unclassifiable = _write_json(
        tmp_path / "Target_Event_20260719_130000.json",
        unclassifiable_payload,
    )

    latest = live_monitor._latest_card_snapshot_payload(
        "Target Event",
        snapshot_dir=tmp_path,
    )

    assert unclassifiable.exists()
    assert latest is not None
    assert latest[0] == exact
    assert latest[1]["event"] == "Target Event"


def test_save_card_snapshot_does_not_swallow_retention_memory_error(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(live_monitor, "SNAPSHOTS_DIR", tmp_path)

    def fail_prune(*_args, **_kwargs):
        raise MemoryError("retention out of memory")

    monkeypatch.setattr(live_monitor, "prune_card_snapshots", fail_prune)

    with pytest.raises(MemoryError, match="retention out of memory"):
        live_monitor.save_card_snapshot(
            "Memory Event",
            [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
            event_date="2099-01-01",
        )

    assert len(list(tmp_path.glob("Memory_Event_*.json"))) == 1


def test_deduplicated_card_save_does_not_swallow_retention_memory_error(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(live_monitor, "SNAPSHOTS_DIR", tmp_path)
    card = [{"fighter_a": "Alpha", "fighter_b": "Beta"}]
    existing = _write_json(
        tmp_path / "Memory_Event_20260719_120000.json",
        {
            "event": "Memory Event",
            "event_date": "2099-01-01",
            "timestamp": NOW.isoformat(),
            "fights": card,
        },
    )

    def fail_prune(*_args, **_kwargs):
        raise MemoryError("retention out of memory")

    monkeypatch.setattr(live_monitor, "prune_card_snapshots", fail_prune)

    with pytest.raises(MemoryError, match="retention out of memory"):
        live_monitor.save_card_snapshot(
            "Memory Event",
            card,
            event_date="2099-01-01",
        )

    assert list(tmp_path.glob("Memory_Event_*.json")) == [existing]
