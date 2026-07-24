import json
import logging
import threading
import time

from src.web import alert_store
from src.web.alert_store import (
    ALERT_STORE_FILENAME,
    DurableAlertHandler,
    alert_fingerprint,
    install_alert_handler,
    load_alert_incidents,
    load_recent_alerts,
    mark_alert_recovered,
    maybe_prune_alert_store,
    prune_alert_store,
    record_alert,
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


def test_load_alert_incidents_coalesces_normalized_legacy_records(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    now = 1_000_000.0
    records = [
        {"ts": now - 120, "timestamp": "one", "level": "WARNING",
         "source": "src.data.scraper", "message": "Reader   temporarily unavailable"},
        {"ts": now - 60, "timestamp": "two", "level": "WARNING",
         "source": "SRC.DATA.SCRAPER", "message": " reader temporarily UNAVAILABLE "},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    incidents = load_alert_incidents(path, max_age_hours=72, now=now)

    assert len(incidents) == 1
    assert incidents[0]["status"] == "active"
    assert incidents[0]["occurrence_count"] == 2
    assert incidents[0]["active_occurrence_count"] == 2
    assert incidents[0]["first_seen"] == "one"
    assert incidents[0]["last_seen"] == "two"


def test_explicit_recovery_closes_incident_and_preserves_history(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    first = record_alert(
        path,
        level="WARNING",
        source="src.data.fallback_scrapers",
        message="Reader failed with status 403",
        incident_key="tapology-reader",
        created=100.0,
    )
    record_alert(
        path,
        level="ERROR",
        source="src.data.fallback_scrapers",
        message="Reader failed again with changing detail",
        incident_key="tapology-reader",
        created=110.0,
    )
    mark_alert_recovered(
        path,
        incident_key="tapology-reader",
        source="src.data.fallback_scrapers",
        recovery_message="Reader probe succeeded",
        created=120.0,
    )

    incidents = load_alert_incidents(path, max_age_hours=72, now=130.0)

    assert len(incidents) == 1
    incident = incidents[0]
    assert incident["fingerprint"] == first["fingerprint"]
    assert incident["status"] == "recovered"
    assert incident["level"] == "ERROR"
    assert incident["occurrence_count"] == 2
    assert incident["active_occurrence_count"] == 0
    assert incident["recovery_count"] == 1
    assert incident["recovered_ts"] == 120.0
    assert incident["recovery_message"] == "Reader probe succeeded"


def test_handler_records_info_level_explicit_recovery(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    handler = DurableAlertHandler(path)
    alert = logging.LogRecord(
        "src.data.method_odds",
        logging.WARNING,
        __file__,
        1,
        "Expected method odds are missing",
        (),
        None,
    )
    alert.alert_incident_key = "method-odds:expected-coverage"
    recovery = logging.LogRecord(
        "src.data.method_odds",
        logging.INFO,
        __file__,
        2,
        "Method odds coverage recovered",
        (),
        None,
    )
    recovery.alert_recovered_incident_keys = ["method-odds:expected-coverage"]

    handler.emit(alert)
    handler.emit(recovery)

    incidents = load_alert_incidents(path, max_age_hours=1)
    assert handler.level == logging.INFO
    assert len(incidents) == 1
    assert incidents[0]["status"] == "recovered"
    assert incidents[0]["recovery_message"] == "Method odds coverage recovered"


def test_handler_does_not_append_recovery_without_active_incident(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    handler = DurableAlertHandler(path)
    recovery = logging.LogRecord(
        "src.data.method_odds",
        logging.INFO,
        __file__,
        1,
        "Method odds coverage healthy",
        (),
        None,
    )
    recovery.alert_recovered_incident_keys = ["method-odds:expected-coverage"]

    handler.emit(recovery)
    handler.emit(recovery)

    assert not path.exists()


def test_handler_recovers_pre_restart_incident_once(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    record_alert(
        path,
        level="WARNING",
        source="src.data.fallback_scrapers",
        message="FightDX timed out",
        incident_key="external-source:fightdx:transient-request-failure",
    )

    # A new handler represents a restarted process with no in-memory outage
    # flag. Repeated healthy probes must close once and then become no-ops.
    handler = DurableAlertHandler(path)
    recovery = logging.LogRecord(
        "src.data.fallback_scrapers",
        logging.INFO,
        __file__,
        1,
        "FightDX request recovered",
        (),
        None,
    )
    recovery.alert_recovered_incident_keys = [
        "external-source:fightdx:transient-request-failure"
    ]
    handler.emit(recovery)
    handler.emit(recovery)

    lines = _read_lines(path)
    assert [line["event"] for line in lines] == ["alert", "recovery"]
    incident = load_alert_incidents(path, max_age_hours=1)[0]
    assert incident["status"] == "recovered"
    assert incident["recovery_count"] == 1


def test_handler_recovers_legacy_unkeyed_source_incidents(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    now = time.time()
    legacy_records = [
        {
            "ts": now - 30,
            "timestamp": "legacy",
            "level": "WARNING",
            "source": "src.data.method_odds",
            "message": "Newest method-odds snapshot is stale: 2026-07-18T23:27:13",
        },
        {
            "ts": now - 20,
            "timestamp": "legacy",
            "level": "WARNING",
            "source": "src.data.fallback_scrapers",
            "message": (
                "External data source unavailable: Tapology - reader temporarily "
                "unavailable: status 403: reader search unavailable"
            ),
        },
        {
            "ts": now - 10,
            "timestamp": "legacy",
            "level": "WARNING",
            "source": "scripts.build_profile_supplement_from_external_profiles",
            "message": (
                "FightDX profile lookup failed for 'Felix Klinkhammer': "
                "Read timed out. (read timeout=8.0)"
            ),
        },
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in legacy_records) + "\n",
        encoding="utf-8",
    )
    handler = DurableAlertHandler(path)

    for incident_key in (
        "method-odds:expected-coverage",
        "external-source:tapology:reader-circuit-opened",
        "external-source:fightdx:transient-request-failure",
    ):
        recovery = logging.LogRecord(
            "health.probe",
            logging.INFO,
            __file__,
            1,
            "Hosted source probe succeeded",
            (),
            None,
        )
        recovery.alert_recovered_incident_keys = [incident_key]
        handler.emit(recovery)

    incidents = load_alert_incidents(path, max_age_hours=72)

    assert len(incidents) == 3
    assert {incident["status"] for incident in incidents} == {"recovered"}


def test_schema_v2_unkeyed_legacy_pattern_remains_unmanaged(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    record = record_alert(
        path,
        level="WARNING",
        source="src.data.fallback_scrapers",
        message=(
            "Tapology reader search failed: reader search request failed: "
            "Read timed out"
        ),
        created=100.0,
    )

    current = load_alert_incidents(path, max_age_hours=1, now=200.0)
    expired = load_alert_incidents(path, max_age_hours=1, now=4_000.0)

    assert record["schema_version"] == 2
    assert "fingerprint" in record
    assert "incident_key" not in record
    assert len(current) == 1
    assert current[0]["status"] == "active"
    assert current[0]["lifecycle_managed"] is False
    assert expired == []


def test_alert_after_recovery_reactivates_same_incident(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    record_alert(
        path,
        level="ERROR",
        source="src.source",
        message="first failure",
        incident_key="source-health",
        created=100.0,
    )
    mark_alert_recovered(
        path,
        incident_key="source-health",
        recovery_message="healthy",
        created=110.0,
    )
    record_alert(
        path,
        level="WARNING",
        source="src.source",
        message="failed again",
        incident_key="source-health",
        created=120.0,
    )

    incident = load_alert_incidents(path, max_age_hours=72, now=130.0)[0]

    assert incident["status"] == "active"
    assert incident["level"] == "WARNING"
    assert incident["occurrence_count"] == 2
    assert incident["active_occurrence_count"] == 1
    assert incident["episode_count"] == 2
    assert incident["recovery_count"] == 1
    assert incident["recovered_ts"] == 110.0


def test_recovery_without_retained_observation_does_not_create_incident(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    mark_alert_recovered(
        path,
        fingerprint=alert_fingerprint(source="src.source", message="missing"),
        recovery_message="healthy",
        created=100.0,
    )

    assert load_alert_incidents(path, max_age_hours=72, now=110.0) == []


def test_prune_alert_store_drops_out_of_window_unmanaged_records(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    now = 1_000_000.0
    fresh = {"ts": now - 3600, "timestamp": "x", "level": "ERROR", "source": "s", "message": "fresh"}
    stale = {"ts": now - 80 * 3600, "timestamp": "y", "level": "ERROR", "source": "s", "message": "stale"}
    path.write_text(json.dumps(stale) + "\n" + json.dumps(fresh) + "\n", encoding="utf-8")

    kept = prune_alert_store(path, max_age_hours=72, now=now)
    assert kept == 1
    remaining = _read_lines(path)
    assert [r["message"] for r in remaining] == ["fresh"]


def test_recent_recovery_keeps_old_alert_anchor_and_then_ages_out(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    now = 1_000_000.0
    record_alert(
        path,
        level="ERROR",
        source="src.source",
        message="old active failure",
        incident_key="source-health",
        created=now - 80 * 3600,
    )
    mark_alert_recovered(
        path,
        incident_key="source-health",
        recovery_message="healthy again",
        created=now - 3600,
    )

    incidents = load_alert_incidents(path, max_age_hours=72, now=now)
    assert len(incidents) == 1
    assert incidents[0]["status"] == "recovered"

    assert prune_alert_store(path, max_age_hours=72, now=now) == 2
    assert [entry["event"] for entry in _read_lines(path)] == ["alert", "recovery"]

    later = now + 80 * 3600
    assert prune_alert_store(path, max_age_hours=72, now=later) == 0
    assert _read_lines(path) == []


def test_unrecovered_incident_remains_active_after_history_window(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    now = 1_000_000.0
    record_alert(
        path,
        level="WARNING",
        source="src.source",
        message="still broken",
        incident_key="source-health",
        created=now - 80 * 3600,
    )

    incidents = load_alert_incidents(path, max_age_hours=72, now=now)

    assert len(incidents) == 1
    assert incidents[0]["status"] == "active"


def test_unmanaged_observation_expires_after_history_window(tmp_path):
    path = tmp_path / ALERT_STORE_FILENAME
    now = 1_000_000.0
    path.write_text(
        json.dumps(
            {
                "ts": now - 80 * 3600,
                "timestamp": "old",
                "level": "WARNING",
                "source": "src.data.live_monitor",
                "message": "SHORT NOTICE: replacement announced",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_alert_incidents(path, max_age_hours=72, now=now) == []


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
    assert _read_lines(path) == []

    # A second unmanaged stale record written immediately after should survive
    # because the throttle blocks a second prune within the interval.
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
