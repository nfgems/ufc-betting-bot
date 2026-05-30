import json
import logging
import threading
import time

from src.web import alert_store
from src.web.alert_store import (
    ALERT_STORE_FILENAME,
    DurableAlertHandler,
    install_alert_handler,
    load_recent_alerts,
    maybe_prune_alert_store,
    prune_alert_store,
)


def _read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_handler_persists_warning_and_error_but_not_info(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    handler = DurableAlertHandler(path)
    logger = logging.getLogger("test.alert_store.levels")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        logger.info("just info")
        logger.debug("just debug")
        logger.warning("a warning happened")
        logger.error("an error happened")
    finally:
        logger.removeHandler(handler)

    records = _read_lines(path)
    assert [r["level"] for r in records] == ["WARNING", "ERROR"]
    assert records[0]["source"] == "test.alert_store.levels"
    assert records[0]["message"] == "a warning happened"
    assert isinstance(records[0]["ts"], (int, float))
    # Display timestamp matches bot.log's local-time format.
    assert len(records[0]["timestamp"]) == len("2026-05-29 02:14:03")


def test_handler_includes_exception_traceback(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    handler = DurableAlertHandler(path)
    logger = logging.getLogger("test.alert_store.exc")
    logger.addHandler(handler)
    try:
        try:
            raise ValueError("boom")
        except ValueError:
            logger.error("operation failed", exc_info=True)
    finally:
        logger.removeHandler(handler)

    records = _read_lines(path)
    assert "operation failed" in records[0]["message"]
    assert "ValueError: boom" in records[0]["message"]
    assert "Traceback" in records[0]["message"]


def test_load_recent_alerts_filters_by_age(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    now = 1_000_000.0
    fresh = {"ts": now - 3600, "timestamp": "x", "level": "ERROR", "source": "s", "message": "fresh"}
    stale = {"ts": now - 80 * 3600, "timestamp": "y", "level": "ERROR", "source": "s", "message": "stale"}
    path.write_text(json.dumps(stale) + "\n" + json.dumps(fresh) + "\n", encoding="utf-8")

    kept = load_recent_alerts(path, max_age_hours=72, now=now)
    assert [r["message"] for r in kept] == ["fresh"]


def test_load_recent_alerts_skips_malformed_lines(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    now = 1_000_000.0
    good = {"ts": now - 60, "timestamp": "x", "level": "WARNING", "source": "s", "message": "ok"}
    path.write_text(
        "not json at all\n"
        "\n"
        + json.dumps(good) + "\n"
        + "{ broken json\n",
        encoding="utf-8",
    )

    kept = load_recent_alerts(path, max_age_hours=72, now=now)
    assert [r["message"] for r in kept] == ["ok"]


def test_load_recent_alerts_falls_back_to_timestamp_string(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    stamp = "2026-05-29 02:14:03"
    epoch = time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))
    # No "ts" field — reader must parse the display timestamp instead.
    obj = {"timestamp": stamp, "level": "ERROR", "source": "s", "message": "legacy"}
    path.write_text(json.dumps(obj) + "\n", encoding="utf-8")

    kept = load_recent_alerts(path, max_age_hours=72, now=epoch + 3600)
    assert [r["message"] for r in kept] == ["legacy"]
    stale = load_recent_alerts(path, max_age_hours=72, now=epoch + 80 * 3600)
    assert stale == []


def test_prune_alert_store_drops_out_of_window_records(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    now = 1_000_000.0
    fresh = {"ts": now - 3600, "timestamp": "x", "level": "ERROR", "source": "s", "message": "fresh"}
    stale = {"ts": now - 80 * 3600, "timestamp": "y", "level": "ERROR", "source": "s", "message": "stale"}
    path.write_text(json.dumps(stale) + "\n" + json.dumps(fresh) + "\n", encoding="utf-8")

    kept = prune_alert_store(path, max_age_hours=72, now=now)
    assert kept == 1
    remaining = _read_lines(path)
    assert [r["message"] for r in remaining] == ["fresh"]


def test_prune_alert_store_preserves_file_when_read_fails(tmp_path, monkeypatch):
    path = tmp_path / ALERT_STORE_FILENAME
    original = json.dumps(
        {"ts": 1_000_000.0, "timestamp": "x", "level": "ERROR", "source": "s", "message": "keep"}
    ) + "\n"
    path.write_text(original, encoding="utf-8")

    def _fail_read(_path):
        raise OSError("read failed")

    monkeypatch.setattr(alert_store, "_iter_alert_records", _fail_read)

    kept = prune_alert_store(path, max_age_hours=72, now=1_000_000.0)

    assert kept == 0
    assert path.read_text(encoding="utf-8") == original


def test_handler_waits_for_store_lock_before_appending(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    handler = DurableAlertHandler(path)
    logger = logging.getLogger("test.alert_store.locking")
    logger.setLevel(logging.DEBUG)
    original_propagate = logger.propagate
    logger.propagate = False
    logger.addHandler(handler)
    started = threading.Event()

    def _log_warning():
        started.set()
        logger.warning("blocked warning")

    try:
        with alert_store._locked_alert_store(path):
            thread = threading.Thread(target=_log_warning)
            thread.start()
            assert started.wait(timeout=1.0)
            time.sleep(0.1)
            assert not path.exists()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    finally:
        logger.removeHandler(handler)
        logger.propagate = original_propagate

    records = _read_lines(path)
    assert [r["message"] for r in records] == ["blocked warning"]


def test_maybe_prune_is_throttled(tmp_path, monkeypatch):
    path = tmp_path / ALERT_STORE_FILENAME
    now = 1_000_000.0
    stale = {"ts": now - 80 * 3600, "timestamp": "y", "level": "ERROR", "source": "s", "message": "stale"}
    path.write_text(json.dumps(stale) + "\n", encoding="utf-8")

    # Reset throttle so this test controls the first-call behaviour.
    monkeypatch.setattr(alert_store, "_last_prune_by_path", {})

    maybe_prune_alert_store(path, max_age_hours=72, min_interval_seconds=3600, now=now)
    assert _read_lines(path) == []  # first call pruned the stale record

    # A second stale record written immediately after should survive because the
    # throttle blocks a second prune within the interval.
    path.write_text(json.dumps(stale) + "\n", encoding="utf-8")
    maybe_prune_alert_store(path, max_age_hours=72, min_interval_seconds=3600, now=now)
    assert len(_read_lines(path)) == 1


def test_install_alert_handler_is_idempotent(tmp_path):
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        h1 = install_alert_handler(tmp_path)
        h2 = install_alert_handler(tmp_path)
        assert h1 is h2
        added = [h for h in root.handlers if isinstance(h, DurableAlertHandler) and h.path == tmp_path / ALERT_STORE_FILENAME]
        assert len(added) == 1
    finally:
        for h in list(root.handlers):
            if h not in before:
                root.removeHandler(h)
