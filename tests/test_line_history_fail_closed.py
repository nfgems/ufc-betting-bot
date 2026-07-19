import logging
import os
import time

import pytest

from src import config
from src.data import line_history_archive
from src.data import line_tracker


def _clear_archive_policy_environment(monkeypatch):
    for name in (
        "LINE_HISTORY_ARCHIVE_REQUIRED",
        "RAILWAY_PROJECT_ID",
        "RAILWAY_ENVIRONMENT_NAME",
        "RAILWAY_ENVIRONMENT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_archive_not_required_by_default_outside_railway_production(monkeypatch):
    _clear_archive_policy_environment(monkeypatch)

    assert config._line_history_archive_required() is False


def test_archive_can_be_required_outside_railway_production(monkeypatch):
    _clear_archive_policy_environment(monkeypatch)
    monkeypatch.setenv("LINE_HISTORY_ARCHIVE_REQUIRED", "1")

    assert config._line_history_archive_required() is True


def test_railway_production_forces_archive_requirement(monkeypatch):
    _clear_archive_policy_environment(monkeypatch)
    monkeypatch.setenv("RAILWAY_PROJECT_ID", "project-id")
    monkeypatch.setenv("RAILWAY_ENVIRONMENT_NAME", "production")
    monkeypatch.setenv("LINE_HISTORY_ARCHIVE_REQUIRED", "0")

    assert config._line_history_archive_required() is True


@pytest.mark.parametrize(
    ("archive_required", "bucket", "expected_archives", "expected_removed"),
    (
        (False, "", 0, 1),
        (True, "", 0, 0),
        (False, "archive", 1, 1),
        (True, "archive", 1, 1),
    ),
)
def test_prune_line_history_archive_policy_matrix(
    tmp_path,
    monkeypatch,
    archive_required,
    bucket,
    expected_archives,
    expected_removed,
):
    now = time.time()
    expired = tmp_path / "odds_expired.csv"
    recent = tmp_path / "polymarket_recent.csv"
    expired.write_text("fighter_a,fighter_b\nAlpha,Beta\n")
    recent.write_text("market,price\nexample,0.5\n")
    old_time = now - 181 * 24 * 60 * 60
    os.utime(expired, (old_time, old_time))
    archived = []

    monkeypatch.setattr(line_tracker, "LINE_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(line_tracker, "LINE_HISTORY_RETENTION_DAYS", 180)
    monkeypatch.setattr(
        line_tracker,
        "LINE_HISTORY_ARCHIVE_REQUIRED",
        archive_required,
    )
    monkeypatch.setattr(line_history_archive, "LINE_HISTORY_ARCHIVE_BUCKET", bucket)
    monkeypatch.setattr(
        line_history_archive,
        "archive_line_history_snapshot",
        lambda path: archived.append(path),
    )

    removed = line_tracker.prune_line_history(now=now, force=True)

    assert removed == expected_removed
    assert len(archived) == expected_archives
    assert expired.exists() is (expected_removed == 0)
    assert recent.exists()


def test_required_archive_upload_failure_preserves_expired_snapshot(
    tmp_path,
    monkeypatch,
    caplog,
):
    now = time.time()
    expired = tmp_path / "polymarket_expired.csv"
    expired.write_text("market,price\nexample,0.5\n")
    old_time = now - 181 * 24 * 60 * 60
    os.utime(expired, (old_time, old_time))

    monkeypatch.setattr(line_tracker, "LINE_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(line_tracker, "LINE_HISTORY_RETENTION_DAYS", 180)
    monkeypatch.setattr(line_tracker, "LINE_HISTORY_ARCHIVE_REQUIRED", True)
    monkeypatch.setattr(
        line_history_archive,
        "LINE_HISTORY_ARCHIVE_BUCKET",
        "archive",
    )

    def fail_archive(_path):
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(
        line_history_archive,
        "archive_line_history_snapshot",
        fail_archive,
    )

    with caplog.at_level(logging.WARNING):
        removed = line_tracker.prune_line_history(now=now, force=True)

    assert removed == 0
    assert expired.exists()
    assert "archive upload failed" in caplog.text


def test_required_archive_missing_bucket_logs_and_preserves_all_expired_snapshots(
    tmp_path,
    monkeypatch,
    caplog,
):
    now = time.time()
    expired_paths = (
        tmp_path / "odds_expired.csv",
        tmp_path / "polymarket_expired.csv",
    )
    old_time = now - 181 * 24 * 60 * 60
    for path in expired_paths:
        path.write_text("data\n")
        os.utime(path, (old_time, old_time))

    archive_calls = []
    monkeypatch.setattr(line_tracker, "LINE_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(line_tracker, "LINE_HISTORY_RETENTION_DAYS", 180)
    monkeypatch.setattr(line_tracker, "LINE_HISTORY_ARCHIVE_REQUIRED", True)
    monkeypatch.setattr(line_history_archive, "LINE_HISTORY_ARCHIVE_BUCKET", "")
    monkeypatch.setattr(
        line_history_archive,
        "archive_line_history_snapshot",
        lambda path: archive_calls.append(path),
    )

    with caplog.at_level(logging.ERROR):
        removed = line_tracker.prune_line_history(now=now, force=True)

    assert removed == 0
    assert archive_calls == []
    assert all(path.exists() for path in expired_paths)
    assert "durable archiving is required" in caplog.text
    assert "LINE_HISTORY_ARCHIVE_BUCKET is not configured" in caplog.text
