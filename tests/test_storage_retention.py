import gzip
import json
import os
import time

from src.data import line_history_archive
from src.data import line_tracker
from src.storage_retention import compact_file_tail
from src.strategy import tracker_decisions
from src.strategy.execution_audit import load_execution_audit_cycles, persist_cycle_payload


def test_compact_file_tail_preserves_newest_complete_lines(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text("".join(f'{{"id":{number},"value":"xxxxxxxx"}}\n' for number in range(20)))

    reclaimed = compact_file_tail(path, 160)

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert reclaimed > 0
    assert path.stat().st_size <= 160
    assert records[-1]["id"] == 19
    assert records[0]["id"] > 0


def test_execution_audit_history_is_bounded(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    latest_path = tmp_path / "latest.json"
    for number in range(20):
        persist_cycle_payload(
            {
                "cycle_id": f"cycle-{number}",
                "completed_at": f"2026-07-18T00:{number:02d}:00+00:00",
                "fights": [{"payload": "x" * 200}],
            },
            log_path=log_path,
            latest_path=latest_path,
            max_bytes=1_000,
        )

    cycles = load_execution_audit_cycles(limit=100, log_path=log_path)
    assert log_path.stat().st_size <= 1_000
    assert cycles[0]["cycle_id"] == "cycle-19"
    assert json.loads(latest_path.read_text())["cycle_id"] == "cycle-19"


def test_tracker_decision_history_is_bounded(tmp_path, monkeypatch):
    path = tmp_path / "tracker.jsonl"
    monkeypatch.setattr(tracker_decisions, "TRACKER_DECISION_LOG_PATH", path)
    monkeypatch.setattr(tracker_decisions, "TRACKER_DECISION_LOG_MAX_BYTES", 1_000)

    for number in range(50):
        tracker_decisions.log_tracker_decision({"id": number, "payload": "x" * 80})

    records = tracker_decisions.load_tracker_decision_log(limit=None)
    assert path.stat().st_size <= 1_000
    assert records[-1]["id"] == 49
    assert records[0]["id"] > 0


def test_tracker_decision_history_limit_returns_latest_records_in_order(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "tracker.jsonl"
    path.write_text(
        "".join(json.dumps({"id": number}) + "\n" for number in range(5)),
        encoding="utf-8",
    )
    monkeypatch.setattr(tracker_decisions, "TRACKER_DECISION_LOG_PATH", path)

    records = tracker_decisions.load_tracker_decision_log(limit=2)

    assert [record["id"] for record in records] == [3, 4]


def test_prune_line_history_removes_only_expired_snapshots(tmp_path, monkeypatch):
    now = time.time()
    old_odds = tmp_path / "odds_old.csv"
    old_poly = tmp_path / "polymarket_old.csv"
    recent_odds = tmp_path / "odds_recent.csv"
    opening_lines = tmp_path / "opening_lines.json"
    for path in (old_odds, old_poly, recent_odds, opening_lines):
        path.write_text("data")
    old_time = now - 61 * 24 * 60 * 60
    os.utime(old_odds, (old_time, old_time))
    os.utime(old_poly, (old_time, old_time))

    monkeypatch.setattr(line_tracker, "LINE_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(line_tracker, "LINE_HISTORY_RETENTION_DAYS", 60)
    monkeypatch.setattr(line_history_archive, "LINE_HISTORY_ARCHIVE_BUCKET", "")

    removed = line_tracker.prune_line_history(now=now, force=True)

    assert removed == 2
    assert not old_odds.exists()
    assert not old_poly.exists()
    assert recent_odds.exists()
    assert opening_lines.exists()


def test_prune_line_history_archives_before_deleting(tmp_path, monkeypatch):
    now = time.time()
    expired = tmp_path / "odds_expired.csv"
    expired.write_text("fighter_a,fighter_b\nAlpha,Beta\n")
    old_time = now - 181 * 24 * 60 * 60
    os.utime(expired, (old_time, old_time))
    archived = []

    monkeypatch.setattr(line_tracker, "LINE_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(line_tracker, "LINE_HISTORY_RETENTION_DAYS", 180)
    monkeypatch.setattr(line_history_archive, "LINE_HISTORY_ARCHIVE_BUCKET", "archive")
    monkeypatch.setattr(
        line_history_archive,
        "archive_line_history_snapshot",
        lambda path: archived.append(path),
    )

    removed = line_tracker.prune_line_history(now=now, force=True)

    assert removed == 1
    assert archived == [expired]
    assert not expired.exists()


def test_prune_line_history_preserves_file_when_archive_fails(tmp_path, monkeypatch):
    now = time.time()
    expired = tmp_path / "polymarket_expired.csv"
    expired.write_text("market,price\nexample,0.5\n")
    old_time = now - 181 * 24 * 60 * 60
    os.utime(expired, (old_time, old_time))

    monkeypatch.setattr(line_tracker, "LINE_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(line_tracker, "LINE_HISTORY_RETENTION_DAYS", 180)
    monkeypatch.setattr(line_history_archive, "LINE_HISTORY_ARCHIVE_BUCKET", "archive")

    def fail_archive(_path):
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(
        line_history_archive,
        "archive_line_history_snapshot",
        fail_archive,
    )

    removed = line_tracker.prune_line_history(now=now, force=True)

    assert removed == 0
    assert expired.exists()


def test_line_history_archive_uploads_deterministic_gzip(tmp_path, monkeypatch):
    source = tmp_path / "odds_20260101T120000.csv"
    source.write_text("fighter_a,fighter_b\nAlpha,Beta\n")
    timestamp = 1767268800
    os.utime(source, (timestamp, timestamp))
    uploads = []

    class FakeClient:
        def put_object(self, **kwargs):
            uploads.append(kwargs)

    monkeypatch.setattr(line_history_archive, "LINE_HISTORY_ARCHIVE_BUCKET", "archive")
    monkeypatch.setattr(line_history_archive, "LINE_HISTORY_ARCHIVE_PREFIX", "ufc/history")

    key = line_history_archive.archive_line_history_snapshot(
        source,
        client=FakeClient(),
    )

    assert key == "ufc/history/odds/2026/01/odds_20260101T120000.csv.gz"
    assert uploads[0]["Bucket"] == "archive"
    assert uploads[0]["Key"] == key
    assert uploads[0]["ContentEncoding"] == "gzip"
    assert gzip.decompress(uploads[0]["Body"]) == source.read_bytes()
