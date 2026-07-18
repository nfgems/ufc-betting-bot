import json
import os
import time

from src.data import line_tracker
from src.storage_retention import compact_file_tail
from src.strategy import llm_operator
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
    monkeypatch.setattr(llm_operator, "TRACKER_DECISION_LOG_PATH", path)
    monkeypatch.setattr(llm_operator, "TRACKER_DECISION_LOG_MAX_BYTES", 1_000)

    for number in range(50):
        llm_operator.log_tracker_decision({"id": number, "payload": "x" * 80})

    records = llm_operator.load_tracker_decision_log(limit=None)
    assert path.stat().st_size <= 1_000
    assert records[-1]["id"] == 49
    assert records[0]["id"] > 0


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

    removed = line_tracker.prune_line_history(now=now, force=True)

    assert removed == 2
    assert not old_odds.exists()
    assert not old_poly.exists()
    assert recent_odds.exists()
    assert opening_lines.exists()
