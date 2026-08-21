import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.backfill_active_roster_ufcstats as backfill
import scripts.run_scheduled_ufc_refresh as scheduled
from src.data import io_utils


def test_csv_pair_commit_is_verified_and_manifest_is_last(tmp_path):
    results = tmp_path / "results.csv"
    stats = tmp_path / "stats.csv"
    manifest = io_utils.csv_pair_manifest_path(results, stats)

    io_utils.write_csvs_atomically(
        (
            (pd.DataFrame([{"fight": "one"}]), results),
            (pd.DataFrame([{"fight": "one", "round": 1}]), stats),
        ),
        manifest_path=manifest,
    )

    result_frame, stat_frame = io_utils.read_csv_pair_verified(results, stats)
    assert result_frame.to_dict("records") == [{"fight": "one"}]
    assert stat_frame.to_dict("records") == [{"fight": "one", "round": 1}]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["state"] == "committed"
    assert [entry["rows"] for entry in payload["artifacts"]] == [1, 1]


def test_csv_pair_second_replace_failure_restores_both_files(tmp_path, monkeypatch):
    results = tmp_path / "results.csv"
    stats = tmp_path / "stats.csv"
    pd.DataFrame([{"generation": "old"}]).to_csv(results, index=False)
    pd.DataFrame([{"generation": "old", "round": 1}]).to_csv(stats, index=False)
    old_results, old_stats = results.read_bytes(), stats.read_bytes()
    original_replace = io_utils._replace_with_retry

    def fail_second_csv(source: Path, target: Path):
        if target == stats and source.name.endswith(".pair.tmp"):
            raise OSError("injected second target failure")
        original_replace(source, target)

    monkeypatch.setattr(io_utils, "_replace_with_retry", fail_second_csv)
    with pytest.raises(OSError, match="second target"):
        io_utils.write_csvs_atomically(
            (
                (pd.DataFrame([{"generation": "new"}]), results),
                (pd.DataFrame([{"generation": "new", "round": 1}]), stats),
            ),
            manifest_path=io_utils.csv_pair_manifest_path(results, stats),
        )

    assert results.read_bytes() == old_results
    assert stats.read_bytes() == old_stats
    io_utils.read_csv_pair_verified(results, stats)


def test_new_pair_preparation_failure_removes_initializing_marker_and_can_retry(
    tmp_path,
    monkeypatch,
):
    results = tmp_path / "results.csv"
    stats = tmp_path / "stats.csv"
    manifest = io_utils.csv_pair_manifest_path(results, stats)
    original_copy = io_utils.shutil.copyfile

    def fail_manifest_backup(source: Path, target: Path):
        if Path(source) == manifest and Path(target).name.endswith(".rollback.bak"):
            raise OSError("injected manifest backup failure")
        return original_copy(source, target)

    monkeypatch.setattr(io_utils.shutil, "copyfile", fail_manifest_backup)
    with pytest.raises(OSError, match="manifest backup"):
        io_utils.write_csvs_atomically(
            (
                (pd.DataFrame([{"fight": "one"}]), results),
                (pd.DataFrame([{"fight": "one", "round": 1}]), stats),
            ),
            manifest_path=manifest,
        )

    assert not results.exists()
    assert not stats.exists()
    assert not manifest.exists()

    monkeypatch.setattr(io_utils.shutil, "copyfile", original_copy)
    io_utils.write_csvs_atomically(
        (
            (pd.DataFrame([{"fight": "one"}]), results),
            (pd.DataFrame([{"fight": "one", "round": 1}]), stats),
        ),
        manifest_path=manifest,
    )
    io_utils.read_csv_pair_verified(results, stats)


def test_verified_pair_read_rejects_interleaved_manifest_change(tmp_path, monkeypatch):
    results = tmp_path / "results.csv"
    stats = tmp_path / "stats.csv"
    manifest = io_utils.csv_pair_manifest_path(results, stats)
    io_utils.write_csvs_atomically(
        (
            (pd.DataFrame([{"fight": "one"}]), results),
            (pd.DataFrame([{"fight": "one", "round": 1}]), stats),
        ),
        manifest_path=manifest,
    )
    original_read = io_utils._read_csv_bytes
    calls = 0

    def interleave(path: Path) -> bytes:
        nonlocal calls
        calls += 1
        content = original_read(path)
        if calls == 2:
            manifest.write_bytes(manifest.read_bytes() + b"\n")
        return content

    monkeypatch.setattr(io_utils, "_read_csv_bytes", interleave)
    with pytest.raises(io_utils.CSVPairIntegrityError, match="changed during verified read"):
        io_utils.read_csv_pair_verified(results, stats, manifest_path=manifest)


def test_changed_profile_refresh_preserves_blanks_and_observes_zero():
    existing = {"record": "10-1", "reach": "70", "slpm": "3.2", "stance": "Orthodox"}
    refreshed = {"record": "11-1", "reach": "--", "slpm": 0, "stance": None}

    merged, changed = backfill._merge_profile_rows(
        existing,
        refreshed,
        refresh_existing_values=True,
    )

    assert changed is True
    assert merged == {"record": "11-1", "reach": "70", "slpm": 0, "stance": "Orthodox"}


def _complete_fight_observation():
    result = {column: "observed" for column in backfill._REQUIRED_RESULT_FIELDS}
    result.update(
        {
            "EVENT": "UFC Test",
            "BOUT": "Alpha vs. Beta",
            "OUTCOME": "W/L",
            "ROUND": "1",
            "URL": "http://ufcstats.test/fight",
        }
    )
    rows = []
    for fighter in ("Alpha", "Beta"):
        row = {column: "0" for column in backfill._REQUIRED_STAT_FIELDS}
        row.update(
            {"EVENT": "UFC Test", "BOUT": "Alpha vs. Beta", "ROUND": "1", "FIGHTER": fighter}
        )
        rows.append(row)
    return result, rows


def test_partial_fight_slice_is_rejected_before_pair_publication():
    result, rows = _complete_fight_observation()
    complete, reasons = backfill._fight_observation_completeness(result, rows[:1])
    assert complete is False
    assert "incomplete_round_1_fighters" in reasons
    assert backfill._fight_observation_completeness(result, rows) == (True, [])


def _install_refresh_fakes(tmp_path, monkeypatch, *, backfill_summary, supplement_summary):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    roster_path = raw_dir / "roster.csv"
    active_manifest = tmp_path / "active.json"
    raw_dir.mkdir()
    processed_dir.mkdir()
    active_manifest.write_text('{"bundle_id":"old"}\n', encoding="utf-8")
    events: list[str] = []

    monkeypatch.setattr(scheduled, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(scheduled, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(scheduled, "OFFICIAL_ACTIVE_ROSTER_PATH", roster_path)
    monkeypatch.setattr(scheduled, "is_hosted_runtime", lambda: False)
    monkeypatch.setenv(scheduled.PRODUCTION_BUNDLE_ENV, str(active_manifest))
    monkeypatch.setattr(scheduled, "_load_fresh_cached_roster_for_hosted_refresh", lambda _path: None)

    def sync(*, output_path):
        frame = pd.DataFrame([{"official_name": "Alpha", "ufcstats_url": "http://u/a"}])
        frame.attrs.update(sync_source="live", sync_complete=True)
        frame.to_csv(output_path, index=False)
        return frame

    def rebuild(*, output_subdirs, **_kwargs):
        stage = Path(output_subdirs[0])
        stage.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"event_date": "2026-08-01"}]).to_csv(stage / "fights_cleaned.csv", index=False)
        pd.DataFrame([{"event_date": "2026-08-01", "value": 1}]).to_csv(
            stage / "features.csv", index=False
        )
        events.append("rebuild")
        return {"outputs": [{"fight_rows": 1, "feature_rows": 1}], "production_bundle": {"bad": True}}

    audit_count = 0

    def audit(**_kwargs):
        nonlocal audit_count
        audit_count += 1
        events.append(f"audit-{audit_count}")
        return {"active_roster_rows": 1}, pd.DataFrame(
            [{"official_name": "Alpha", "split_alias_aware": "existing"}]
        )

    def reconcile(*, target_manifest_path, processed_dir):
        events.append("publish")
        Path(target_manifest_path).write_text(
            json.dumps({"bundle_id": "new", "processed_dir": str(processed_dir)}),
            encoding="utf-8",
        )
        return {"bundle_id": "new"}

    monkeypatch.setattr(scheduled, "sync_official_active_roster", sync)
    monkeypatch.setattr(scheduled, "run_backfill", lambda **_kwargs: dict(backfill_summary))
    monkeypatch.setattr(scheduled, "run_rebuild", rebuild)
    monkeypatch.setattr(scheduled, "run_audit", audit)
    monkeypatch.setattr(scheduled, "_maybe_refresh_profile_supplement", lambda **_kwargs: dict(supplement_summary))
    monkeypatch.setattr(scheduled, "_build_unresolved_profile_report", lambda **_kwargs: ({"rows": 0}, pd.DataFrame()))
    monkeypatch.setattr(scheduled, "_build_profile_audit_alert_summary", lambda **_kwargs: {"rows": 0})
    monkeypatch.setattr(scheduled, "reconcile_production_bundle_manifest", reconcile)
    return active_manifest, processed_dir, events


def test_green_refresh_publishes_only_after_staged_audits(tmp_path, monkeypatch):
    active, processed, events = _install_refresh_fakes(
        tmp_path,
        monkeypatch,
        backfill_summary={
            "missing_fight_urls_found": 0,
            "new_result_rows": 0,
            "new_stat_rows": 0,
            "fight_detail_status_counts": {"complete": 0, "partial": 0, "failed": 0},
        },
        supplement_summary={"action": "skip"},
    )
    summary = scheduled.run_scheduled_refresh(
        audit_json_path=None,
        audit_csv_path=None,
        unresolved_json_path=None,
        unresolved_csv_path=None,
        update_production_manifest=True,
    )

    assert summary["continuity_green"] is True
    assert summary["published"] is True
    assert events[-1] == "publish"
    assert "production_bundle" not in summary["rebuild"] or summary["rebuild"]["production_bundle"]["bundle_id"] == "new"
    assert json.loads(active.read_text(encoding="utf-8"))["bundle_id"] == "new"
    assert not (processed / "fights_cleaned.csv").exists()


def test_partial_or_source_error_refresh_never_publishes(tmp_path, monkeypatch):
    active, _processed, events = _install_refresh_fakes(
        tmp_path,
        monkeypatch,
        backfill_summary={
            "missing_fight_urls_found": 1,
            "new_result_rows": 0,
            "new_stat_rows": 0,
            "fight_detail_status_counts": {"complete": 0, "partial": 1, "failed": 0},
            "partial_fight_urls": ["fight-a"],
        },
        supplement_summary={"action": "completed", "source_error_count": 1},
    )
    original = active.read_bytes()
    summary = scheduled.run_scheduled_refresh(
        audit_json_path=None,
        audit_csv_path=None,
        unresolved_json_path=None,
        unresolved_csv_path=None,
        update_production_manifest=True,
    )

    assert summary["continuity_green"] is False
    assert summary["published"] is False
    assert "ufcstats_partial_fight_observations" in summary["outcome_reasons"]
    assert "profile_supplement_source_errors" in summary["outcome_reasons"]
    assert "publish" not in events
    assert active.read_bytes() == original


def test_custom_isolated_output_is_preserved_alongside_unaddressed_stage(
    tmp_path,
    monkeypatch,
):
    _active, processed, _events = _install_refresh_fakes(
        tmp_path,
        monkeypatch,
        backfill_summary={
            "missing_fight_urls_found": 0,
            "new_result_rows": 0,
            "new_stat_rows": 0,
            "fight_detail_status_counts": {"complete": 0, "partial": 0, "failed": 0},
        },
        supplement_summary={"action": "skip"},
    )
    original_rebuild = scheduled.run_rebuild
    calls: list[list[str]] = []

    def capture_rebuild(**kwargs):
        calls.append(list(kwargs["output_subdirs"]))
        return original_rebuild(**kwargs)

    monkeypatch.setattr(scheduled, "run_rebuild", capture_rebuild)
    summary = scheduled.run_scheduled_refresh(
        output_subdirs=["candidates/isolated-pre-recovery"],
        audit_json_path=None,
        audit_csv_path=None,
        unresolved_json_path=None,
        unresolved_csv_path=None,
        update_production_manifest=False,
    )

    assert len(calls) == 1
    assert Path(calls[0][0]).parent == processed / scheduled.REFRESH_GENERATIONS_SUBDIR
    assert calls[0][1] == "candidates/isolated-pre-recovery"
    assert summary["materialized_output_subdirs"] == calls[0]


def test_publishable_hosted_cycle_requires_live_roster_even_when_cache_is_fresh(
    tmp_path,
    monkeypatch,
):
    _active, _processed, _events = _install_refresh_fakes(
        tmp_path,
        monkeypatch,
        backfill_summary={
            "missing_fight_urls_found": 0,
            "new_result_rows": 0,
            "new_stat_rows": 0,
            "fight_detail_status_counts": {"complete": 0, "partial": 0, "failed": 0},
        },
        supplement_summary={"action": "skip"},
    )
    monkeypatch.setattr(scheduled, "is_hosted_runtime", lambda: True)
    monkeypatch.setattr(
        scheduled,
        "_load_fresh_cached_roster_for_hosted_refresh",
        lambda _path: pytest.fail("publishable continuity cannot use an unproven cache"),
    )

    summary = scheduled.run_scheduled_refresh(
        audit_json_path=None,
        audit_csv_path=None,
        unresolved_json_path=None,
        unresolved_csv_path=None,
        update_production_manifest=True,
    )

    assert summary["roster_sync"]["source"] == "live"
    assert summary["roster_sync"]["sync_complete"] is True
    assert summary["continuity_green"] is True
    assert summary["published"] is True
