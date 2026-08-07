import logging
import threading
import time

import pytest

from src.web import app as web_app


PROFILE_CACHE_KEY = "dashboard-polymarket-profile"


def _wait_for_profile_refresh(timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with web_app._cache_lock:
            refreshing = PROFILE_CACHE_KEY in web_app._endpoint_inflight
        if not refreshing:
            return
        time.sleep(0.005)
    pytest.fail("profile cache refresh did not finish")


@pytest.fixture(autouse=True)
def _reset_profile_cache_state():
    with web_app._cache_lock:
        web_app._endpoint_cache.clear()
        web_app._endpoint_inflight.clear()
        web_app._profile_snapshot_warning_state.clear()
        web_app._profile_snapshot_failure_state.clear()
        web_app._background_cache_refreshes = 0
    yield
    _wait_for_profile_refresh()
    with web_app._cache_lock:
        web_app._endpoint_cache.clear()
        web_app._endpoint_inflight.clear()
        web_app._profile_snapshot_warning_state.clear()
        web_app._profile_snapshot_failure_state.clear()
        web_app._background_cache_refreshes = 0


def test_cold_profile_snapshot_refresh_is_non_blocking(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = []
    expected = {"total_pnl": 12.5, "username": "cache-test"}

    def compute():
        calls.append(1)
        started.set()
        release.wait(2.0)
        return expected

    monkeypatch.setattr(web_app, "_compute_polymarket_profile_snapshot", compute)

    began_at = time.perf_counter()
    snapshot, source = web_app._load_polymarket_profile_snapshot()
    elapsed = time.perf_counter() - began_at

    try:
        assert elapsed < 0.5
        assert snapshot == {}
        assert source == "unavailable"
        assert started.wait(0.5)
        assert calls == [1]
    finally:
        release.set()

    _wait_for_profile_refresh()
    snapshot, source = web_app._load_polymarket_profile_snapshot()

    assert snapshot == expected
    assert source == "live"
    assert calls == [1]


def test_profile_scheduler_skips_refresh_if_cache_became_fresh_after_load_inspection(
    monkeypatch,
):
    stale = {"total_pnl": 1.0, "username": "stale-profile"}
    fresh = {"total_pnl": 2.0, "username": "fresh-profile"}
    with web_app._cache_lock:
        web_app._endpoint_cache[PROFILE_CACHE_KEY] = {"data": stale, "ts": 0.0}

    compute_calls = 0

    def compute():
        nonlocal compute_calls
        compute_calls += 1
        return {"total_pnl": 3.0}

    original_scheduler = web_app._refresh_cache_key_in_background

    def cache_fresh_then_schedule(key, compute_fn, **kwargs):
        with web_app._cache_lock:
            web_app._endpoint_cache[key] = {"data": fresh, "ts": time.time()}
        return original_scheduler(key, compute_fn, **kwargs)

    monkeypatch.setattr(web_app, "_compute_polymarket_profile_snapshot", compute)
    monkeypatch.setattr(
        web_app,
        "_refresh_cache_key_in_background",
        cache_fresh_then_schedule,
    )

    assert web_app._load_polymarket_profile_snapshot() == (stale, "stale")
    assert web_app._load_polymarket_profile_snapshot() == (fresh, "live")
    assert compute_calls == 0
    assert PROFILE_CACHE_KEY not in web_app._endpoint_inflight


def test_profile_scheduler_skips_refresh_if_cooldown_opens_after_load_inspection(
    monkeypatch,
):
    stale = {"total_pnl": 4.0, "username": "stale-profile"}
    with web_app._cache_lock:
        web_app._endpoint_cache[PROFILE_CACHE_KEY] = {"data": stale, "ts": 0.0}

    compute_calls = 0

    def compute():
        nonlocal compute_calls
        compute_calls += 1
        return {"total_pnl": 5.0}

    original_scheduler = web_app._refresh_cache_key_in_background

    def open_cooldown_then_schedule(key, compute_fn, **kwargs):
        with web_app._cache_lock:
            web_app._profile_snapshot_failure_state[key] = {
                "retry_after": time.monotonic() + 60.0,
                "error": "another refresh failed",
            }
        return original_scheduler(key, compute_fn, **kwargs)

    monkeypatch.setattr(web_app, "_compute_polymarket_profile_snapshot", compute)
    monkeypatch.setattr(
        web_app,
        "_refresh_cache_key_in_background",
        open_cooldown_then_schedule,
    )

    assert web_app._load_polymarket_profile_snapshot() == (stale, "stale")
    assert compute_calls == 0
    assert PROFILE_CACHE_KEY not in web_app._endpoint_inflight


def test_concurrent_cold_profile_callers_share_failure_cooldown(monkeypatch):
    caller_count = 8
    barrier = threading.Barrier(caller_count)
    compute_started = threading.Event()
    release_compute = threading.Event()
    compute_lock = threading.Lock()
    compute_calls = 0
    results = []

    def compute():
        nonlocal compute_calls
        with compute_lock:
            compute_calls += 1
        compute_started.set()
        release_compute.wait(2.0)
        raise TimeoutError("profile request timed out")

    def load_snapshot():
        barrier.wait()
        results.append(web_app._load_polymarket_profile_snapshot())

    monkeypatch.setattr(web_app, "_compute_polymarket_profile_snapshot", compute)
    threads = [threading.Thread(target=load_snapshot) for _ in range(caller_count)]
    for thread in threads:
        thread.start()

    try:
        assert compute_started.wait(0.5)
        for thread in threads:
            thread.join(0.5)
        assert all(not thread.is_alive() for thread in threads)
        assert results == [({}, "unavailable")] * caller_count
        assert compute_calls == 1
    finally:
        release_compute.set()

    _wait_for_profile_refresh()
    snapshot, source = web_app._load_polymarket_profile_snapshot()

    assert snapshot == {}
    assert source == "unavailable"
    assert compute_calls == 1
    with web_app._cache_lock:
        failure = dict(web_app._profile_snapshot_failure_state[PROFILE_CACHE_KEY])
    assert failure["retry_after"] > time.monotonic()


def test_failed_profile_refresh_preserves_stale_data_and_debounces_warning(
    monkeypatch,
    caplog,
):
    stale = {"total_pnl": 9.25, "username": "stale-profile"}
    with web_app._cache_lock:
        web_app._endpoint_cache[PROFILE_CACHE_KEY] = {"data": stale, "ts": 0.0}

    compute_calls = 0

    def compute():
        nonlocal compute_calls
        compute_calls += 1
        raise TimeoutError("same upstream timeout")

    monkeypatch.setattr(web_app, "_compute_polymarket_profile_snapshot", compute)
    caplog.set_level(logging.DEBUG, logger=web_app.logger.name)

    assert web_app._load_polymarket_profile_snapshot() == (stale, "stale")
    _wait_for_profile_refresh()
    assert web_app._load_polymarket_profile_snapshot() == (stale, "stale")
    assert compute_calls == 1

    with web_app._cache_lock:
        web_app._profile_snapshot_failure_state[PROFILE_CACHE_KEY]["retry_after"] = (
            time.monotonic() - 1.0
        )

    assert web_app._load_polymarket_profile_snapshot() == (stale, "stale")
    _wait_for_profile_refresh()

    warning_prefix = f"Background cache refresh failed for {PROFILE_CACHE_KEY}"
    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and warning_prefix in record.getMessage()
    ]
    suppressed_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG and "suppressed duplicate" in record.getMessage()
    ]

    assert compute_calls == 2
    assert web_app._endpoint_cache[PROFILE_CACHE_KEY]["data"] == stale
    assert len(warning_messages) == 1
    assert len(suppressed_messages) == 1


def test_background_refresh_cleanup_does_not_remove_replacement_inflight_entry():
    cache_key = "replacement-inflight-test"
    compute_started = threading.Event()
    release_compute = threading.Event()

    def compute():
        compute_started.set()
        release_compute.wait(2.0)
        return {"value": "fresh"}

    assert web_app._refresh_cache_key_in_background(cache_key, compute)
    assert compute_started.wait(0.5)

    with web_app._cache_lock:
        original_pending = web_app._endpoint_inflight[cache_key]
        replacement_pending = {"event": threading.Event()}
        web_app._endpoint_inflight[cache_key] = replacement_pending

    release_compute.set()
    assert original_pending["event"].wait(0.5)

    with web_app._cache_lock:
        assert web_app._endpoint_inflight[cache_key] is replacement_pending
        web_app._endpoint_inflight.pop(cache_key)
    assert web_app._background_cache_refreshes == 0
