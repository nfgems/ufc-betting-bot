import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

import scripts.run_scheduled_ufc_refresh as scheduled_refresh
from scripts.backfill_active_roster_ufcstats import FIGHTERS_PATH as BACKFILL_FIGHTERS_PATH
from src.config import RAW_DATA_DIR
from src.data.io_utils import write_csv_atomically


@pytest.mark.parametrize(("continuity_green", "expected"), [(True, 0), (False, 2)])
def test_src_bot_scheduled_dispatch_propagates_continuity_exit(
    monkeypatch,
    continuity_green,
    expected,
):
    from src import bot as bot_cli

    monkeypatch.setattr(
        scheduled_refresh,
        "run_scheduled_refresh",
        lambda **_kwargs: {"continuity_green": continuity_green},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["bot", "ufc-refresh-scheduled", "--skip-rebuild", "--skip-audit"],
    )

    assert bot_cli.main() == expected


@pytest.mark.parametrize("false_value", [0.0, "0.0", "0.00", "0e0"])
def test_scheduled_alias_reports_reject_numeric_false_identity_flags(
    false_value,
):
    assert not scheduled_refresh._official_url_identity_trusted(
        {
            "official_url_identity_status": "valid",
            "official_url_identity_valid": false_value,
        }
    )


def test_write_csv_atomically_refuses_empty_overwrite(tmp_path):
    target = tmp_path / "artifact.csv"
    pd.DataFrame([{"name": "Alpha"}]).to_csv(target, index=False)

    with pytest.raises(ValueError):
        write_csv_atomically(pd.DataFrame(), target, refuse_empty=True)

    preserved = pd.read_csv(target)
    assert preserved.to_dict(orient="records") == [{"name": "Alpha"}]


def test_run_scheduled_refresh_chains_pipeline_and_writes_audit_outputs(tmp_path, monkeypatch):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    audit_json_path = tmp_path / "scheduled_audit.json"
    audit_csv_path = tmp_path / "scheduled_audit.csv"
    calls: dict[str, object] = {}

    monkeypatch.setattr(scheduled_refresh, "OFFICIAL_ACTIVE_ROSTER_PATH", roster_path)
    monkeypatch.setattr(scheduled_refresh, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(scheduled_refresh, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setenv("UFC_PRODUCTION_BUNDLE_MANIFEST", str(tmp_path / "runtime_bundle.json"))

    def fake_sync_official_active_roster(*, output_path):
        calls["sync_output_path"] = output_path
        df = pd.DataFrame(
            [
                {
                    "official_name": "Alpha Fighter",
                    "ufcstats_url": "http://ufcstats.test/alpha",
                    "ufcstats_name": "Alpha Fighter",
                    "profile_status": "ok",
                }
            ]
        )
        df.attrs.update(sync_source="live", sync_complete=True)
        df.to_csv(output_path, index=False)
        return df

    def fake_run_backfill(*, refresh_roster, limit_fighters, roster_df):
        calls["backfill"] = {
            "refresh_roster": refresh_roster,
            "limit_fighters": limit_fighters,
            "rows": len(roster_df),
        }
        return {
            "roster_rows_with_ufcstats_url": 1,
            "fighters_checked": 1,
            "fighters_with_missing_fights": 1,
            "missing_fight_urls_found": 1,
            "new_result_rows": 1,
            "new_stat_rows": 2,
            "scraped_profiles_added": 0,
            "scraped_profiles_updated": 1,
            "failed_fight_urls": [],
        }

    def fake_run_rebuild(*, dataset_variant, output_subdirs, update_production_manifest):
        staged_dir = Path(output_subdirs[0])
        staged_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"event_date": "2026-08-01"}]).to_csv(
            staged_dir / "fights_cleaned.csv", index=False
        )
        pd.DataFrame([{"event_date": "2026-08-01", "feature": 1}]).to_csv(
            staged_dir / "features.csv", index=False
        )
        calls["rebuild"] = {
            "dataset_variant": dataset_variant,
            "output_subdirs": output_subdirs,
            "update_production_manifest": update_production_manifest,
        }
        return {
            "dataset_variant": dataset_variant,
            "outputs": [{"fight_rows": 10, "feature_rows": 10, "feature_cols": 5}],
            "production_bundle": {"bundle_id": "bundle-1"},
        }

    def fake_run_audit(*, active_roster_path, processed_fights_path, scraped_fighters_path):
        calls["audit"] = {
            "active_roster_path": active_roster_path,
            "processed_fights_path": processed_fights_path,
            "scraped_fighters_path": scraped_fighters_path,
        }
        return (
            {
                "active_roster_rows": 1,
                "split_summary_official_name": {
                    "newly_added_active_roster": {
                        "rows": 1,
                        "reach_present": {"count": 1, "pct": 100.0},
                        "stance_present": {"count": 0, "pct": 0.0},
                    }
                },
            },
            pd.DataFrame(
                [
                    {
                        "official_name": "Alpha Fighter",
                        "split_alias_aware": "newly_added_active_roster",
                        "reach_present": True,
                        "stance_present": False,
                    }
                ]
            ),
        )

    monkeypatch.setattr(scheduled_refresh, "sync_official_active_roster", fake_sync_official_active_roster)
    monkeypatch.setattr(scheduled_refresh, "run_backfill", fake_run_backfill)
    monkeypatch.setattr(scheduled_refresh, "run_rebuild", fake_run_rebuild)
    monkeypatch.setattr(scheduled_refresh, "run_audit", fake_run_audit)
    monkeypatch.setattr(
        scheduled_refresh,
        "run_profile_supplement_refresh",
        lambda **_kwargs: {
            "candidate_rows": 1,
            "attempted_rows": 1,
            "recovered_rows": 0,
            "selected_sources": ["martialbot"],
            "recovered_by_source": {
                "martialbot": 0,
                "fightdx": 0,
                "espn": 0,
                "tapology": 0,
                "sherdog": 0,
                "wikipedia": 0,
            },
            "output_path": str(raw_dir / "ufc_fighters_profile_supplement.csv"),
        },
    )

    summary = scheduled_refresh.run_scheduled_refresh(
        dataset_variant="pulled_all_plus_legacy_market",
        output_subdirs=None,
        limit_fighters=25,
        audit_json_path=audit_json_path,
        audit_csv_path=audit_csv_path,
        unresolved_json_path=None,
        unresolved_csv_path=None,
    )

    assert calls["sync_output_path"] == roster_path
    assert calls["backfill"] == {
        "refresh_roster": False,
        "limit_fighters": 25,
        "rows": 1,
    }
    assert calls["rebuild"]["dataset_variant"] == "pulled_all_plus_legacy_market"
    assert calls["rebuild"]["update_production_manifest"] is False
    staged_dir = Path(calls["rebuild"]["output_subdirs"][0])
    assert staged_dir.parent == processed_dir / scheduled_refresh.REFRESH_GENERATIONS_SUBDIR
    assert staged_dir.name.startswith("refresh-")
    assert calls["audit"]["active_roster_path"] == roster_path
    assert calls["audit"]["processed_fights_path"] == staged_dir / "fights_cleaned.csv"
    assert calls["audit"]["scraped_fighters_path"] == raw_dir / "ufc_fighters_scraped.csv"

    assert summary["roster_sync"]["rows"] == 1
    assert summary["ufcstats_backfill"]["new_result_rows"] == 1
    assert summary["rebuild"]["outputs"][0]["fight_rows"] == 10
    assert "production_bundle" not in summary["rebuild"]
    assert summary["continuity_green"] is False
    assert summary["published"] is False
    assert "partial_refresh_limit" in summary["outcome_reasons"]
    assert summary["profile_audit"]["active_roster_rows"] == 1

    saved_audit = json.loads(audit_json_path.read_text(encoding="utf-8"))
    assert saved_audit["active_roster_rows"] == 1
    saved_audit_csv = pd.read_csv(audit_csv_path)
    assert saved_audit_csv.loc[0, "official_name"] == "Alpha Fighter"

    # Verify resolved_paths and file state diagnostics are present
    assert "resolved_paths" in summary
    assert summary["resolved_paths"]["RAW_DATA_DIR"] == str(raw_dir)
    assert summary["resolved_paths"]["PROCESSED_DATA_DIR"] == str(processed_dir)
    assert "pre_refresh_file_state" in summary
    assert "post_refresh_file_state" in summary


def test_backfill_and_audit_use_same_scraped_fighters_path():
    """The audit and backfill must resolve to the exact same scraped fighters path."""
    audit_path = RAW_DATA_DIR / "ufc_fighters_scraped.csv"
    assert audit_path.resolve() == BACKFILL_FIGHTERS_PATH.resolve(), (
        f"Path mismatch: audit uses {audit_path}, backfill uses {BACKFILL_FIGHTERS_PATH}"
    )


def test_roster_summary_counts_identity_audit_actions(tmp_path):
    roster_df = pd.DataFrame(
        [
            {
                "official_name": "Real Fighter",
                "ufcstats_url": "http://ufcstats.test/real",
                "ufcstats_name": "Real Fighter",
                "profile_status": "Active",
            }
        ]
    )
    roster_df.attrs["identity_audit_rows"] = [
        {"action": "excluded_test_profile"},
        {"action": "quarantined_untrusted_slug_alias"},
        {"action": "quarantined_untrusted_slug_alias"},
    ]

    summary = scheduled_refresh._roster_summary(
        roster_df,
        output_path=tmp_path / "ufc_active_roster_official.csv",
    )

    assert summary["identity_audit_rows"] == 3
    assert summary["identity_audit_action_counts"] == {
        "excluded_test_profile": 1,
        "quarantined_untrusted_slug_alias": 2,
    }


def test_roster_summary_reports_retained_missing_live_rows(tmp_path):
    roster_df = pd.DataFrame(
        [
            {
                "official_name": "Real Fighter",
                "ufcstats_url": "http://ufcstats.test/real",
                "ufcstats_name": "Real Fighter",
                "profile_status": "Active",
            }
        ]
    )
    roster_df.attrs["retained_missing_live_rows"] = [
        {
            "official_name": "Omitted Fighter",
            "official_athlete_url": "https://www.ufc.com/athlete/omitted-fighter",
            "ufcstats_url": "http://ufcstats.test/omitted-fighter",
        }
    ]

    summary = scheduled_refresh._roster_summary(
        roster_df,
        output_path=tmp_path / "ufc_active_roster_official.csv",
    )

    assert summary["retained_missing_live_rows"] == 1
    assert summary["retained_missing_live_fighters"] == [
        {
            "official_name": "Omitted Fighter",
            "official_athlete_url": "https://www.ufc.com/athlete/omitted-fighter",
            "ufcstats_url": "http://ufcstats.test/omitted-fighter",
        }
    ]


def test_roster_summary_reports_discarded_suspicious_cached_rows(tmp_path):
    roster_df = pd.DataFrame(
        [
            {
                "official_name": "Real Fighter",
                "ufcstats_url": "http://ufcstats.test/real",
                "ufcstats_name": "Real Fighter",
                "profile_status": "Active",
            }
        ]
    )
    roster_df.attrs["discarded_suspicious_cached_rows"] = 2177

    summary = scheduled_refresh._roster_summary(
        roster_df,
        output_path=tmp_path / "ufc_active_roster_official.csv",
    )

    assert summary["discarded_suspicious_cached_rows"] == 2177


def test_roster_summary_reports_intentionally_removed_cached_rows(tmp_path):
    roster_df = pd.DataFrame(
        [
            {
                "official_name": "Real Fighter",
                "ufcstats_url": "http://ufcstats.test/real",
                "ufcstats_name": "Real Fighter",
                "profile_status": "Active",
            }
        ]
    )
    roster_df.attrs["intentionally_removed_cached_rows"] = [
        {
            "official_name": "Retired Fighter",
            "official_athlete_url": (
                "https://www.ufc.com/athlete/retired-fighter"
            ),
            "ufcstats_url": "http://ufcstats.test/retired-fighter",
            "reason": "excluded_inactive_profile_status",
        }
    ]

    summary = scheduled_refresh._roster_summary(
        roster_df,
        output_path=tmp_path / "ufc_active_roster_official.csv",
    )

    assert summary["intentionally_removed_cached_rows"] == 1
    assert summary["intentionally_removed_cached_fighters"] == [
        {
            "official_name": "Retired Fighter",
            "official_athlete_url": (
                "https://www.ufc.com/athlete/retired-fighter"
            ),
            "ufcstats_url": "http://ufcstats.test/retired-fighter",
            "reason": "excluded_inactive_profile_status",
        }
    ]


def test_run_scheduled_refresh_rebuilds_before_and_after_recovered_profile_gaps(tmp_path, monkeypatch):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    calls: dict[str, object] = {"order": []}

    # This test exercises the local default-source path, independent of any
    # runtime source override inherited by the pytest process.
    monkeypatch.delenv("UFC_REFRESH_PROFILE_SUPPLEMENT_SOURCES", raising=False)
    monkeypatch.setattr(scheduled_refresh, "OFFICIAL_ACTIVE_ROSTER_PATH", roster_path)
    monkeypatch.setattr(scheduled_refresh, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(scheduled_refresh, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(scheduled_refresh, "PROFILE_SUPPLEMENT_PATH", raw_dir / "ufc_fighters_profile_supplement.csv")

    def fake_sync_official_active_roster(*, output_path):
        calls["order"].append("sync")
        df = pd.DataFrame(
            [
                {"official_name": "Gap Fighter", "ufcstats_url": "http://ufcstats.test/gap", "profile_status": "ok"},
                {"official_name": "Covered Fighter", "ufcstats_url": "http://ufcstats.test/covered", "profile_status": "ok"},
            ]
        )
        df.to_csv(output_path, index=False)
        return df

    def fake_run_backfill(*, refresh_roster, limit_fighters, roster_df):
        calls["order"].append("backfill")
        return {
            "refresh_roster": refresh_roster,
            "limit_fighters": limit_fighters,
            "rows": len(roster_df),
        }

    audit_calls = {"count": 0}

    def fake_run_audit(*, active_roster_path, processed_fights_path, scraped_fighters_path):
        audit_calls["count"] += 1
        calls["order"].append(f"audit_{audit_calls['count']}")
        if audit_calls["count"] == 1:
            return (
                {"active_roster_rows": 2},
                pd.DataFrame(
                    [
                        {
                            "official_name": "Gap Fighter",
                            "split_official_name": "newly_added_active_roster",
                            "full_physical_bundle_present": False,
                        },
                        {
                            "official_name": "Covered Fighter",
                            "split_official_name": "existing_processed_active_roster",
                            "full_physical_bundle_present": True,
                        },
                    ]
                ),
            )
        return (
            {"active_roster_rows": 2, "overall_summary": {"rows": 2}},
            pd.DataFrame(
                [
                    {
                        "official_name": "Gap Fighter",
                        "split_official_name": "newly_added_active_roster",
                        "full_physical_bundle_present": True,
                    }
                ]
            ),
        )

    def fake_run_profile_supplement_refresh(
        *,
        scraped_fighters_path,
        candidate_source_csv,
        output_path,
        sources,
        limit,
        candidate_rotation_index,
    ):
        calls["order"].append("supplement")
        calls["supplement_candidate_rows"] = pd.read_csv(candidate_source_csv).to_dict(orient="records")
        calls["supplement_sources"] = list(sources)
        calls["supplement_limit"] = limit
        calls["supplement_rotation_index"] = candidate_rotation_index
        return {
            "candidate_rows": 1,
            "attempted_rows": 1,
            "recovered_rows": 1,
            "selected_sources": list(sources),
            "recovered_by_source": {
                "martialbot": 1,
                "fightdx": 0,
                "espn": 0,
                "tapology": 0,
                "sherdog": 0,
                "wikipedia": 0,
            },
            "output_path": str(output_path),
        }

    def fake_run_rebuild(*, dataset_variant, output_subdirs, update_production_manifest):
        calls["order"].append("rebuild")
        return {
            "dataset_variant": dataset_variant,
            "outputs": [{"fight_rows": 10, "feature_rows": 10, "feature_cols": 5}],
        }

    monkeypatch.setattr(scheduled_refresh, "sync_official_active_roster", fake_sync_official_active_roster)
    monkeypatch.setattr(scheduled_refresh, "run_backfill", fake_run_backfill)
    monkeypatch.setattr(scheduled_refresh, "run_audit", fake_run_audit)
    monkeypatch.setattr(scheduled_refresh, "run_profile_supplement_refresh", fake_run_profile_supplement_refresh)
    monkeypatch.setattr(scheduled_refresh, "run_rebuild", fake_run_rebuild)

    summary = scheduled_refresh.run_scheduled_refresh(
        dataset_variant="pulled_all_plus_legacy_market",
        output_subdirs=None,
        limit_fighters=None,
        audit_json_path=None,
        audit_csv_path=None,
        unresolved_json_path=None,
        unresolved_csv_path=None,
    )

    assert calls["order"] == [
        "sync",
        "backfill",
        "rebuild",
        "audit_1",
        "supplement",
        "rebuild",
        "audit_2",
    ]
    assert calls["supplement_candidate_rows"] == [
        {"official_name": "Gap Fighter", "ufcstats_url": "http://ufcstats.test/gap", "profile_status": "ok"}
    ]
    assert calls["supplement_sources"] == list(scheduled_refresh.DEFAULT_PROFILE_SUPPLEMENT_REFRESH_SOURCES)
    assert calls["supplement_limit"] is None
    assert calls["supplement_rotation_index"] is None
    assert summary["profile_supplement_refresh"]["action"] == "completed"
    assert summary["profile_supplement_refresh"]["recovered_rows"] == 1


def test_run_scheduled_refresh_skips_profile_supplement_for_partial_refresh(tmp_path, monkeypatch):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    monkeypatch.setattr(scheduled_refresh, "OFFICIAL_ACTIVE_ROSTER_PATH", roster_path)
    monkeypatch.setattr(scheduled_refresh, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(scheduled_refresh, "PROCESSED_DATA_DIR", processed_dir)

    monkeypatch.setattr(
        scheduled_refresh,
        "sync_official_active_roster",
        lambda *, output_path: pd.DataFrame([{"official_name": "Gap Fighter", "ufcstats_url": "http://ufcstats.test/gap"}]),
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "run_backfill",
        lambda **_kwargs: {"fighters_checked": 1},
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "run_rebuild",
        lambda **_kwargs: {"outputs": [{"fight_rows": 1, "feature_rows": 1, "feature_cols": 1}]},
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "run_audit",
        lambda **_kwargs: (
            {"active_roster_rows": 1},
            pd.DataFrame(
                [
                    {
                        "official_name": "Gap Fighter",
                        "split_official_name": "newly_added_active_roster",
                        "full_physical_bundle_present": False,
                    }
                ]
            ),
        ),
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "run_profile_supplement_refresh",
        lambda **_kwargs: pytest.fail("partial refresh should not trigger profile supplement refresh"),
    )

    summary = scheduled_refresh.run_scheduled_refresh(
        dataset_variant="pulled_all_plus_legacy_market",
        output_subdirs=None,
        limit_fighters=5,
        audit_json_path=None,
        audit_csv_path=None,
        unresolved_json_path=None,
        unresolved_csv_path=None,
    )

    assert summary["profile_supplement_refresh"] == {
        "action": "skip",
        "reason": "partial refresh",
    }


def test_run_scheduled_refresh_reports_cached_roster_fallback(tmp_path, monkeypatch):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    monkeypatch.setattr(scheduled_refresh, "OFFICIAL_ACTIVE_ROSTER_PATH", roster_path)
    monkeypatch.setattr(scheduled_refresh, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(scheduled_refresh, "PROCESSED_DATA_DIR", processed_dir)

    def fake_sync_official_active_roster(*, output_path):
        df = pd.DataFrame(
            [
                {
                    "official_name": "Cached Fighter",
                    "ufcstats_url": "http://ufcstats.test/cached-fighter",
                    "ufcstats_name": "Cached Fighter",
                    "profile_status": "Active",
                }
            ]
        )
        df.attrs["sync_source"] = "cached"
        df.attrs["sync_fallback_used"] = True
        df.attrs["sync_error"] = (
            "HTTPSConnectionPool(host='www.ufc.com', port=443): Read timed out. (read timeout=30)"
        )
        df.attrs["sync_cached_snapshot_mtime_utc"] = "2026-03-28T20:00:00+00:00"
        return df

    monkeypatch.setattr(scheduled_refresh, "sync_official_active_roster", fake_sync_official_active_roster)
    monkeypatch.setattr(
        scheduled_refresh,
        "run_backfill",
        lambda **kwargs: {"fighters_checked": len(kwargs["roster_df"])},
    )

    summary = scheduled_refresh.run_scheduled_refresh(
        dataset_variant="pulled_all_plus_legacy_market",
        output_subdirs=None,
        limit_fighters=None,
        audit_json_path=None,
        audit_csv_path=None,
        unresolved_json_path=None,
        unresolved_csv_path=None,
        skip_rebuild=True,
        skip_audit=True,
    )

    assert summary["roster_sync"]["rows"] == 1
    assert summary["roster_sync"]["source"] == "cached"
    assert summary["roster_sync"]["used_cached_fallback"] is True
    assert summary["roster_sync"]["cached_snapshot_mtime_utc"] == "2026-03-28T20:00:00+00:00"
    assert "Read timed out" in summary["roster_sync"]["sync_error"]
    assert summary["ufcstats_backfill"]["fighters_checked"] == 1


def test_hosted_scheduled_refresh_uses_fresh_cached_roster_before_live_sync(tmp_path, monkeypatch):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    pd.DataFrame(
        [
            {
                "official_name": "Cached Fighter",
                "ufcstats_url": "http://ufcstats.test/cached-fighter",
                "ufcstats_name": "Cached Fighter",
                "profile_status": "Active",
            }
        ]
    ).to_csv(roster_path, index=False)

    monkeypatch.setattr(scheduled_refresh, "OFFICIAL_ACTIVE_ROSTER_PATH", roster_path)
    monkeypatch.setattr(scheduled_refresh, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(scheduled_refresh, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(scheduled_refresh, "is_hosted_runtime", lambda: True)
    monkeypatch.setattr(
        scheduled_refresh,
        "sync_official_active_roster",
        lambda **_kwargs: pytest.fail("fresh hosted roster cache should be used before live sync"),
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "run_backfill",
        lambda **kwargs: {
            "fighters_checked": len(kwargs["roster_df"]),
            "roster_names": kwargs["roster_df"]["official_name"].tolist(),
        },
    )

    summary = scheduled_refresh.run_scheduled_refresh(
        dataset_variant="pulled_all_plus_legacy_market",
        output_subdirs=None,
        limit_fighters=None,
        audit_json_path=None,
        audit_csv_path=None,
        unresolved_json_path=None,
        unresolved_csv_path=None,
        skip_rebuild=True,
        skip_audit=True,
    )

    assert summary["roster_sync"]["source"] == "cached_runtime_preexisting"
    assert summary["roster_sync"]["rows"] == 1
    assert summary["ufcstats_backfill"]["roster_names"] == ["Cached Fighter"]


def test_hosted_profile_supplement_refresh_sources_exclude_tapology_by_default(monkeypatch):
    monkeypatch.delenv("UFC_REFRESH_PROFILE_SUPPLEMENT_SOURCES", raising=False)
    monkeypatch.setattr(scheduled_refresh, "is_hosted_runtime", lambda: True)

    sources = scheduled_refresh._profile_supplement_refresh_sources()

    assert "tapology" not in sources
    assert set(sources) == {
        "espn",
        "fightdx",
        "martialbot",
        "sherdog",
        "wikipedia",
    }


def test_profile_supplement_rotation_index_advances_with_hosted_cadence(monkeypatch):
    start = datetime(2026, 7, 23, tzinfo=timezone.utc)
    monkeypatch.setenv("UFC_REFRESH_INTERVAL_HOURS", "24")

    first = scheduled_refresh._profile_supplement_refresh_rotation_index(start)

    assert scheduled_refresh._profile_supplement_refresh_rotation_index(start) == first
    assert (
        scheduled_refresh._profile_supplement_refresh_rotation_index(
            start + timedelta(hours=23)
        )
        == first
    )
    assert (
        scheduled_refresh._profile_supplement_refresh_rotation_index(
            start + timedelta(hours=24)
        )
        == first + 1
    )

    monkeypatch.setenv("UFC_REFRESH_INTERVAL_HOURS", "6")
    six_hour_first = scheduled_refresh._profile_supplement_refresh_rotation_index(
        start
    )
    assert (
        scheduled_refresh._profile_supplement_refresh_rotation_index(
            start + timedelta(hours=6)
        )
        == six_hour_first + 1
    )


def test_hosted_profile_supplement_passes_deterministic_rotation_index(
    tmp_path,
    monkeypatch,
):
    captured = {}
    candidate_df = pd.DataFrame([{"official_name": "Gap Fighter"}])
    monkeypatch.setattr(scheduled_refresh, "is_hosted_runtime", lambda: True)
    monkeypatch.setattr(
        scheduled_refresh,
        "_profile_supplement_refresh_rotation_index",
        lambda: 731,
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "_build_profile_gap_candidate_frame",
        lambda **_kwargs: candidate_df,
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "run_profile_supplement_refresh",
        lambda **kwargs: captured.update(kwargs)
        or {
            "candidate_rows": 1,
            "attempted_rows": 1,
            "recovered_rows": 0,
        },
    )

    summary = scheduled_refresh._maybe_refresh_profile_supplement(
        active_roster_path=tmp_path / "roster.csv",
        scraped_fighters_path=tmp_path / "fighters.csv",
        audit_df=pd.DataFrame(),
        partial_refresh=False,
    )

    assert summary["action"] == "completed"
    assert captured["candidate_rotation_index"] == 731
    assert captured["limit"] == scheduled_refresh._profile_supplement_refresh_limit()


def test_run_scheduled_refresh_writes_post_refresh_unresolved_profile_report(tmp_path, monkeypatch):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    audit_json_path = tmp_path / "scheduled_audit.json"
    audit_csv_path = tmp_path / "scheduled_audit.csv"
    unresolved_json_path = tmp_path / "scheduled_unresolved.json"
    unresolved_csv_path = tmp_path / "scheduled_unresolved.csv"
    raw_dir.mkdir()
    processed_dir.mkdir()

    monkeypatch.setattr(scheduled_refresh, "OFFICIAL_ACTIVE_ROSTER_PATH", roster_path)
    monkeypatch.setattr(scheduled_refresh, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(scheduled_refresh, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(scheduled_refresh, "PROFILE_SUPPLEMENT_PATH", raw_dir / "ufc_fighters_profile_supplement.csv")

    pd.DataFrame(
        [
            {
                "name": "Blank Stats Fighter",
                "fighter_url": "http://ufcstats.test/blank",
                "height": "--",
                "weight": "155 lbs.",
                "reach": "--",
                "stance": "",
                "dob": "",
            }
        ]
    ).to_csv(raw_dir / "ufc_fighters_scraped.csv", index=False)
    pd.DataFrame(
        [
            {
                "name": "Blank Stats Fighter",
                "source": "tapology",
                "source_name": "Blank Stats Fighter",
                "search_name": "Blank Stats Fighter",
                "fighter_url": "https://www.tapology.com/fightcenter/fighters/blank-stats-fighter",
                "height": "",
                "reach": "",
                "weight": "",
                "stance": "",
                "dob": "",
            }
        ]
    ).to_csv(raw_dir / "ufc_fighters_profile_supplement.csv", index=False)

    def fake_sync_official_active_roster(*, output_path):
        df = pd.DataFrame(
            [
                {
                    "official_name": "Missing Url Fighter",
                    "official_athlete_url": "https://www.ufc.com/athlete/missing-url-fighter",
                    "octagon_debut": "Mar. 23, 2026",
                    "age": "",
                    "height": "",
                    "reach": "",
                    "weight": 125,
                    "ufcstats_name": "",
                    "ufcstats_url": "",
                    "alternate_slug_names": "",
                },
                {
                    "official_name": "Blank Stats Fighter",
                    "official_athlete_url": "https://www.ufc.com/athlete/blank-stats-fighter",
                    "octagon_debut": "Mar. 23, 2026",
                    "age": 29,
                    "height": "",
                    "reach": "",
                    "weight": 155,
                    "ufcstats_name": "Blank Stats Fighter",
                    "ufcstats_url": "http://ufcstats.test/blank",
                    "alternate_slug_names": "",
                },
            ]
        )
        df.to_csv(output_path, index=False)
        return df

    def fake_run_audit(*, active_roster_path, processed_fights_path, scraped_fighters_path):
        return (
            {"active_roster_rows": 2},
            pd.DataFrame(
                [
                    {
                        "official_name": "Missing Url Fighter",
                        "profile_source_alias": "",
                        "split_official_name": "newly_added_active_roster",
                        "split_alias_aware": "newly_added_active_roster",
                        "age_present": False,
                        "weight_present": True,
                        "height_present": False,
                        "reach_present": False,
                        "stance_present": False,
                        "full_physical_bundle_present": False,
                    },
                    {
                        "official_name": "Blank Stats Fighter",
                        "profile_source_alias": "Blank Stats Fighter",
                        "split_official_name": "newly_added_active_roster",
                        "split_alias_aware": "newly_added_active_roster",
                        "age_present": True,
                        "weight_present": True,
                        "height_present": False,
                        "reach_present": False,
                        "stance_present": False,
                        "full_physical_bundle_present": False,
                    },
                ]
            ),
        )

    monkeypatch.setattr(scheduled_refresh, "sync_official_active_roster", fake_sync_official_active_roster)
    monkeypatch.setattr(scheduled_refresh, "run_backfill", lambda **_kwargs: {"fighters_checked": 2})
    monkeypatch.setattr(
        scheduled_refresh,
        "run_profile_supplement_refresh",
        lambda **_kwargs: {
            "candidate_rows": 2,
            "attempted_rows": 2,
            "recovered_rows": 0,
            "selected_sources": list(scheduled_refresh.DEFAULT_PROFILE_SUPPLEMENT_REFRESH_SOURCES),
            "recovered_by_source": {
                "martialbot": 0,
                "fightdx": 0,
                "espn": 0,
                "tapology": 0,
                "sherdog": 0,
                "wikipedia": 0,
            },
            "output_path": str(raw_dir / "ufc_fighters_profile_supplement.csv"),
        },
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "run_rebuild",
        lambda **_kwargs: {"outputs": [{"fight_rows": 2, "feature_rows": 2, "feature_cols": 1}]},
    )
    monkeypatch.setattr(scheduled_refresh, "run_audit", fake_run_audit)

    summary = scheduled_refresh.run_scheduled_refresh(
        dataset_variant="pulled_all_plus_legacy_market",
        output_subdirs=None,
        limit_fighters=None,
        audit_json_path=audit_json_path,
        audit_csv_path=audit_csv_path,
        unresolved_json_path=unresolved_json_path,
        unresolved_csv_path=unresolved_csv_path,
    )

    unresolved_summary = json.loads(unresolved_json_path.read_text(encoding="utf-8"))
    unresolved_csv = pd.read_csv(unresolved_csv_path)

    assert summary["profile_unresolved_report"]["rows"] == 7
    assert unresolved_summary["fields"] == {"age": 1, "height": 2, "reach": 2, "stance": 2}
    assert unresolved_summary["source_availability"] == {
        "no_ufcstats_profile_resolved": 3,
        "source_limited_blank": 4,
    }
    assert unresolved_summary["repair_queue"][0]["official_name"] == "Missing Url Fighter"
    assert "a_stance_enc" in unresolved_summary["repair_queue"][0]["model_feature_fields"]
    assert unresolved_summary["reasons_by_field"]["age"] == {
        "official_age_blank_and_no_dob_source": 1,
    }
    assert unresolved_summary["reasons_by_field"]["stance"] == {
        "no_ufcstats_url_and_no_supplement_rows": 1,
        "ufcstats_blank_and_supplement_blank": 1,
    }

    missing_url_stance = unresolved_csv[
        (unresolved_csv["official_name"] == "Missing Url Fighter") & (unresolved_csv["field"] == "stance")
    ].iloc[0]
    assert missing_url_stance["why_missing_code"] == "no_ufcstats_url_and_no_supplement_rows"

    blank_stats_reach = unresolved_csv[
        (unresolved_csv["official_name"] == "Blank Stats Fighter") & (unresolved_csv["field"] == "reach")
    ].iloc[0]
    assert blank_stats_reach["why_missing_code"] == "ufcstats_blank_and_supplement_blank"
    assert blank_stats_reach["supplement_sources"] == "tapology"


def test_seed_stale_scraped_fighters_skips_when_no_divergence(tmp_path, monkeypatch):
    """When image and runtime paths are the same, seeding should be skipped."""
    monkeypatch.setattr(scheduled_refresh, "_IMAGE_RAW_DIR", tmp_path / "nonexistent")
    result = scheduled_refresh._seed_stale_scraped_fighters()
    assert result["action"] == "skip"


def test_seed_stale_scraped_fighters_merges_richer_image(tmp_path, monkeypatch):
    """When the image copy has more stance data, it should be merged into the runtime copy."""
    image_raw = tmp_path / "image_raw"
    image_raw.mkdir()
    runtime_raw = tmp_path / "runtime_raw"
    runtime_raw.mkdir()

    # Image has 3 fighters with stance data
    image_df = pd.DataFrame([
        {"name": "A", "fighter_url": "http://test/a", "stance": "Orthodox", "reach": "72", "height": "5' 10\"", "weight": "155", "dob": "1990-01-01"},
        {"name": "B", "fighter_url": "http://test/b", "stance": "Southpaw", "reach": "70", "height": "5' 8\"", "weight": "145", "dob": ""},
        {"name": "C", "fighter_url": "http://test/c", "stance": "Switch", "reach": "74", "height": "6' 0\"", "weight": "170", "dob": ""},
    ])
    image_df.to_csv(image_raw / "ufc_fighters_scraped.csv", index=False)

    # Runtime has only 1 fighter, no stance
    runtime_df = pd.DataFrame([
        {"name": "A", "fighter_url": "http://test/a", "stance": "", "reach": "72", "height": "5' 10\"", "weight": "155", "dob": "1990-01-01"},
    ])
    runtime_df.to_csv(runtime_raw / "ufc_fighters_scraped.csv", index=False)

    monkeypatch.setattr(scheduled_refresh, "_IMAGE_RAW_DIR", image_raw)
    monkeypatch.setattr(scheduled_refresh, "BACKFILL_FIGHTERS_PATH", runtime_raw / "ufc_fighters_scraped.csv")

    result = scheduled_refresh._seed_stale_scraped_fighters()
    assert result["action"] == "merged"
    assert result["new_rows_from_image"] == 2  # B and C are new
    assert result["updated_fields"] >= 1  # A's stance gets filled

    merged = pd.read_csv(runtime_raw / "ufc_fighters_scraped.csv")
    assert len(merged) == 3
    # A should now have stance filled from image
    a_row = merged[merged["name"] == "A"].iloc[0]
    assert a_row["stance"] == "Orthodox"


def test_seed_stale_scraped_fighters_does_not_count_style_label_as_stance(tmp_path, monkeypatch):
    image_raw = tmp_path / "image_raw"
    image_raw.mkdir()
    runtime_raw = tmp_path / "runtime_raw"
    runtime_raw.mkdir()

    pd.DataFrame(
        [
            {
                "name": "A",
                "fighter_url": "http://test/a",
                "stance": "Striker",
                "reach": "72",
            },
        ]
    ).to_csv(image_raw / "ufc_fighters_scraped.csv", index=False)
    pd.DataFrame(
        [
            {
                "name": "A",
                "fighter_url": "http://test/a",
                "stance": "",
                "reach": "72",
            },
        ]
    ).to_csv(runtime_raw / "ufc_fighters_scraped.csv", index=False)

    monkeypatch.setattr(scheduled_refresh, "_IMAGE_RAW_DIR", image_raw)
    monkeypatch.setattr(scheduled_refresh, "BACKFILL_FIGHTERS_PATH", runtime_raw / "ufc_fighters_scraped.csv")

    result = scheduled_refresh._seed_stale_scraped_fighters()

    assert result["action"] == "skip"
    assert result["runtime_stance"] == 0
    assert result["image_stance"] == 0
    runtime = pd.read_csv(runtime_raw / "ufc_fighters_scraped.csv")
    assert runtime.loc[0, "stance"] != "Striker"


def test_seed_stale_scraped_fighters_skips_when_runtime_is_richer(tmp_path, monkeypatch):
    """When the runtime copy is already richer, seeding should be skipped."""
    image_raw = tmp_path / "image_raw"
    image_raw.mkdir()
    runtime_raw = tmp_path / "runtime_raw"
    runtime_raw.mkdir()

    small_df = pd.DataFrame([
        {"name": "A", "fighter_url": "http://test/a", "stance": "Orthodox", "reach": "72"},
    ])
    small_df.to_csv(image_raw / "ufc_fighters_scraped.csv", index=False)

    bigger_df = pd.DataFrame([
        {"name": "A", "fighter_url": "http://test/a", "stance": "Orthodox", "reach": "72"},
        {"name": "B", "fighter_url": "http://test/b", "stance": "Southpaw", "reach": "70"},
    ])
    bigger_df.to_csv(runtime_raw / "ufc_fighters_scraped.csv", index=False)

    monkeypatch.setattr(scheduled_refresh, "_IMAGE_RAW_DIR", image_raw)
    monkeypatch.setattr(scheduled_refresh, "BACKFILL_FIGHTERS_PATH", runtime_raw / "ufc_fighters_scraped.csv")

    result = scheduled_refresh._seed_stale_scraped_fighters()
    assert result["action"] == "skip"
    assert result["reason"] == "runtime copy is at least as rich as image"


def test_seed_stale_profile_supplement_merges_image_rows_and_fields(tmp_path, monkeypatch):
    image_raw = tmp_path / "image_raw"
    image_raw.mkdir()
    runtime_raw = tmp_path / "runtime_raw"
    runtime_raw.mkdir()

    pd.DataFrame(
        [
            {
                "name": "Existing Fighter",
                "source": "tapology",
                "source_name": "Existing Fighter",
                "fighter_url": "https://www.tapology.com/fightcenter/fighters/existing",
                "height": "180.0",
                "reach": "185.0",
                "weight": "155.0",
                "stance": "",
                "dob": "1992-01-01",
            },
            {
                "name": "New Fighter",
                "source": "tapology",
                "source_name": "New Fighter",
                "fighter_url": "https://www.tapology.com/fightcenter/fighters/new",
                "height": "175.0",
                "reach": "178.0",
                "weight": "145.0",
                "stance": "",
                "dob": "1995-02-02",
            },
        ]
    ).to_csv(image_raw / "ufc_fighters_profile_supplement.csv", index=False)
    pd.DataFrame(
        [
            {
                "name": "Existing Fighter",
                "source": "tapology",
                "source_name": "Existing Fighter",
                "fighter_url": "https://www.tapology.com/fightcenter/fighters/existing",
                "height": "",
                "reach": "",
                "weight": "155.0",
                "stance": "",
                "dob": "",
            },
            {
                "name": "Runtime Only",
                "source": "espn",
                "source_name": "Runtime Only",
                "fighter_url": "https://www.espn.com/mma/fighter/_/id/runtime",
                "height": "182.0",
                "reach": "",
                "weight": "170.0",
                "stance": "Orthodox",
                "dob": "",
            },
        ]
    ).to_csv(runtime_raw / "ufc_fighters_profile_supplement.csv", index=False)

    monkeypatch.setattr(scheduled_refresh, "_IMAGE_RAW_DIR", image_raw)
    monkeypatch.setattr(
        scheduled_refresh,
        "PROFILE_SUPPLEMENT_PATH",
        runtime_raw / "ufc_fighters_profile_supplement.csv",
    )

    result = scheduled_refresh._seed_stale_profile_supplement()

    assert result["action"] == "merged"
    assert result["new_rows_from_image"] == 1
    assert result["updated_fields"] == 3
    merged = pd.read_csv(runtime_raw / "ufc_fighters_profile_supplement.csv")
    existing = merged[(merged["name"] == "Existing Fighter") & (merged["source"] == "tapology")].iloc[0]
    assert str(existing["height"]) == "180.0"
    assert str(existing["reach"]) == "185.0"
    assert str(existing["dob"]) == "1992-01-01"
    assert "New Fighter" in set(merged["name"])
    assert "Runtime Only" in set(merged["name"])


def test_seed_stale_profile_supplement_skips_when_runtime_contains_image(tmp_path, monkeypatch):
    image_raw = tmp_path / "image_raw"
    image_raw.mkdir()
    runtime_raw = tmp_path / "runtime_raw"
    runtime_raw.mkdir()

    supplement = pd.DataFrame(
        [
            {
                "name": "Existing Fighter",
                "source": "tapology",
                "source_name": "Existing Fighter",
                "fighter_url": "https://www.tapology.com/fightcenter/fighters/existing",
                "height": "180.0",
                "reach": "185.0",
                "weight": "155.0",
                "stance": "",
                "dob": "1992-01-01",
            },
        ]
    )
    supplement.to_csv(image_raw / "ufc_fighters_profile_supplement.csv", index=False)
    supplement.to_csv(runtime_raw / "ufc_fighters_profile_supplement.csv", index=False)

    monkeypatch.setattr(scheduled_refresh, "_IMAGE_RAW_DIR", image_raw)
    monkeypatch.setattr(
        scheduled_refresh,
        "PROFILE_SUPPLEMENT_PATH",
        runtime_raw / "ufc_fighters_profile_supplement.csv",
    )

    result = scheduled_refresh._seed_stale_profile_supplement()

    assert result["action"] == "skip"
    assert result["reason"] == "runtime copy already contains image supplement rows"


def test_seed_stale_profile_supplement_cleans_invalid_runtime_dob(tmp_path, monkeypatch):
    image_raw = tmp_path / "image_raw"
    image_raw.mkdir()
    runtime_raw = tmp_path / "runtime_raw"
    runtime_raw.mkdir()

    image = pd.DataFrame(
        [
            {
                "name": "Vineesh Subrahmanyan",
                "source": "tapology",
                "source_name": "Vineesh Subrahmanyan",
                "fighter_url": "https://www.tapology.com/fightcenter/fighters/485663-vineesh-subrahmanyan-vini",
                "height": "",
                "reach": "",
                "weight": "",
                "stance": "",
                "dob": "",
            },
        ]
    )
    runtime = image.copy()
    runtime.loc[0, "dob"] = "3335 Round 2"
    image.to_csv(image_raw / "ufc_fighters_profile_supplement.csv", index=False)
    runtime.to_csv(runtime_raw / "ufc_fighters_profile_supplement.csv", index=False)

    monkeypatch.setattr(scheduled_refresh, "_IMAGE_RAW_DIR", image_raw)
    monkeypatch.setattr(
        scheduled_refresh,
        "PROFILE_SUPPLEMENT_PATH",
        runtime_raw / "ufc_fighters_profile_supplement.csv",
    )

    result = scheduled_refresh._seed_stale_profile_supplement()

    assert result["action"] == "merged"
    assert result["sanitized_fields"] == 1
    merged = pd.read_csv(runtime_raw / "ufc_fighters_profile_supplement.csv")
    assert pd.isna(merged.loc[0, "dob"]) or merged.loc[0, "dob"] == ""


def test_build_profile_audit_alert_summary_excludes_recent_new_fighters(tmp_path):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    pd.DataFrame(
        [
            {"official_name": "Fresh Reach Gap", "octagon_debut": "Mar. 23, 2026"},
            {"official_name": "Matured Fighter", "octagon_debut": "Sep. 6, 2025"},
            {"official_name": "Existing Fighter", "octagon_debut": "Jan. 1, 2020"},
        ]
    ).to_csv(roster_path, index=False)

    audit_df = pd.DataFrame(
        [
            {
                "official_name": "Fresh Reach Gap",
                "split_official_name": "newly_added_active_roster",
                "age_present": True,
                "weight_present": True,
                "division_present": True,
                "height_present": False,
                "reach_present": False,
                "stance_present": False,
                "full_physical_bundle_present": False,
            },
            {
                "official_name": "Matured Fighter",
                "split_official_name": "newly_added_active_roster",
                "age_present": True,
                "weight_present": True,
                "division_present": True,
                "height_present": True,
                "reach_present": True,
                "stance_present": True,
                "full_physical_bundle_present": True,
            },
            {
                "official_name": "Existing Fighter",
                "split_official_name": "existing_processed_active_roster",
                "age_present": True,
                "weight_present": True,
                "division_present": True,
                "height_present": True,
                "reach_present": True,
                "stance_present": True,
                "full_physical_bundle_present": True,
            },
        ]
    )

    summary = scheduled_refresh._build_profile_audit_alert_summary(
        active_roster_path=roster_path,
        audit_df=audit_df,
        as_of_utc=datetime(2026, 3, 27, tzinfo=timezone.utc),
        new_fighter_grace_days=7,
    )

    assert summary["new_fighter_grace_days"] == 7
    assert summary["newly_added_active_roster"] == {
        "rows_total": 2,
        "rows_alert_eligible": 1,
        "rows_in_grace": 1,
    }
    assert summary["split_summary_official_name"]["newly_added_active_roster"]["rows"] == 1
    assert summary["split_summary_official_name"]["newly_added_active_roster"]["reach_present"] == {
        "count": 1,
        "pct": 100.0,
    }


def test_build_profile_audit_alert_summary_uses_alias_aware_new_fighters_by_default(tmp_path):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    pd.DataFrame(
        [
            {"official_name": "Alias Existing", "octagon_debut": "Jan. 1, 2025"},
            {"official_name": "True New", "octagon_debut": "Jan. 1, 2025"},
        ]
    ).to_csv(roster_path, index=False)

    audit_df = pd.DataFrame(
        [
            {
                "official_name": "Alias Existing",
                "split_official_name": "newly_added_active_roster",
                "split_alias_aware": "existing_processed_active_roster",
                "age_present": True,
                "weight_present": True,
                "division_present": True,
                "height_present": True,
                "reach_present": True,
                "stance_present": True,
                "full_physical_bundle_present": True,
            },
            {
                "official_name": "True New",
                "split_official_name": "newly_added_active_roster",
                "split_alias_aware": "newly_added_active_roster",
                "age_present": True,
                "weight_present": True,
                "division_present": True,
                "height_present": True,
                "reach_present": False,
                "stance_present": False,
                "full_physical_bundle_present": False,
            },
        ]
    )

    summary = scheduled_refresh._build_profile_audit_alert_summary(
        active_roster_path=roster_path,
        audit_df=audit_df,
        as_of_utc=datetime(2026, 3, 27, tzinfo=timezone.utc),
        new_fighter_grace_days=7,
    )

    assert summary["identity_match_method"] == "alias_aware"
    assert summary["newly_added_active_roster"]["rows_total"] == 1
    assert summary["new_fighter_counts_by_method"]["official_name"]["rows_total"] == 2
    assert summary["new_fighter_counts_by_method"]["alias_aware"]["rows_total"] == 1
    assert summary["split_summary_alias_aware"]["newly_added_active_roster"]["reach_present"] == {
        "count": 0,
        "pct": 0.0,
    }


def test_build_profile_audit_alert_summary_marks_source_limited_only_missing_fields(tmp_path):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    pd.DataFrame(
        [
            {"official_name": "Source Limited Fighter", "octagon_debut": "Jan. 1, 2025"},
        ]
    ).to_csv(roster_path, index=False)

    audit_df = pd.DataFrame(
        [
            {
                "official_name": "Source Limited Fighter",
                "split_official_name": "newly_added_active_roster",
                "age_present": True,
                "weight_present": True,
                "division_present": True,
                "height_present": True,
                "reach_present": True,
                "stance_present": False,
                "full_physical_bundle_present": False,
            },
        ]
    )
    unresolved_df = pd.DataFrame(
        [
            {
                "official_name": "Source Limited Fighter",
                "field": "stance",
                "why_missing_code": "no_ufcstats_url_and_no_supplement_rows",
            },
        ]
    )

    summary = scheduled_refresh._build_profile_audit_alert_summary(
        active_roster_path=roster_path,
        audit_df=audit_df,
        unresolved_df=unresolved_df,
        as_of_utc=datetime(2026, 3, 27, tzinfo=timezone.utc),
        new_fighter_grace_days=7,
    )

    assert summary["source_limited_missing_fields"]["reach"] == {
        "rows_missing": 0,
        "rows_source_limited": 0,
        "source_limited_only": False,
    }
    assert summary["source_limited_missing_fields"]["stance"] == {
        "rows_missing": 1,
        "rows_source_limited": 1,
        "source_limited_only": True,
    }
    assert summary["source_limited_missing_fields"]["full_physical_bundle"] == {
        "rows_missing": 1,
        "rows_source_limited": 1,
        "source_limited_only": True,
    }


def test_row_drop_guard_reports_key_artifact_regressions():
    pre = {
        "ufc_fighters_scraped": {"path": "ufc_fighters_scraped.csv", "row_count": 4466},
        "ufc_fight_stats": {"path": "ufc-fight-stats.csv", "row_count": 100},
    }
    post = {
        "ufc_fighters_scraped": {"path": "ufc_fighters_scraped.csv", "row_count": 4455},
        "ufc_fight_stats": {"path": "ufc-fight-stats.csv", "row_count": 100},
    }

    guard = scheduled_refresh._build_row_drop_guard(pre, post)

    assert guard["ok"] is False
    assert guard["violations"] == [
        {
            "artifact": "ufc_fighters_scraped",
            "path": "ufc_fighters_scraped.csv",
            "pre_rows": 4466,
            "post_rows": 4455,
            "rows_lost": 11,
        }
    ]
    assert guard["explained_drops"] == []


def _row_guard_fighter_identity(index: int | str) -> dict[str, object]:
    return {
        "official_name": f"Fighter {index}",
        "official_athlete_url": (
            f"https://www.ufc.com/athlete/fighter-{index}"
        ),
        "ufcstats_url": f"http://ufcstats.test/fighter-{index}",
        "official_url_identity_valid": "True",
        "official_url_identity_status": "valid",
    }


def test_identity_matching_preserves_url_constrained_duplicate_name():
    weak_name_only = {
        "official_name": "Shared Name",
        "official_athlete_url": "",
        "ufcstats_url": "",
    }
    url_constrained = {
        "official_name": "Shared Name",
        "official_athlete_url": (
            "https://www.ufc.com/athlete/shared-name-current"
        ),
        "ufcstats_url": "http://ufcstats.test/shared-name-current",
    }

    unmatched = scheduled_refresh._unmatched_identity_rows(
        [weak_name_only, url_constrained],
        [dict(url_constrained)],
    )

    assert unmatched == [weak_name_only]


def test_explanation_matching_maximizes_duplicate_name_mixed_url_matches():
    weak_name_only = {
        "official_name": "Shared Name",
        "official_athlete_url": "",
        "ufcstats_url": "",
    }
    first_url = {
        "official_name": "Shared Name",
        "official_athlete_url": (
            "https://www.ufc.com/athlete/shared-name-first"
        ),
        "ufcstats_url": "http://ufcstats.test/shared-name-first",
    }
    second_url = {
        "official_name": "Shared Name",
        "official_athlete_url": (
            "https://www.ufc.com/athlete/shared-name-second"
        ),
        "ufcstats_url": "http://ufcstats.test/shared-name-second",
    }

    reason_counts, unexpected = (
        scheduled_refresh._partition_explained_identity_rows(
            [second_url, first_url],
            {
                "lifecycle_cleanup": [
                    weak_name_only,
                    dict(second_url),
                ],
            },
        )
    )

    assert reason_counts == {"lifecycle_cleanup": 2}
    assert unexpected == []


def test_row_drop_guard_accepts_exactly_explained_active_roster_churn():
    pre_identities = [
        _row_guard_fighter_identity(index)
        for index in range(1004)
    ]
    expected_removals = pre_identities[:33]
    pre = {
        "ufc_active_roster_official": {
            "path": "ufc_active_roster_official.csv",
            "row_count": 1004,
            "identity_rows": pre_identities,
        },
    }
    post = {
        "ufc_active_roster_official": {
            "path": "ufc_active_roster_official.csv",
            "row_count": 971,
            "identity_rows": pre_identities[33:],
        },
    }

    guard = scheduled_refresh._build_row_drop_guard(
        pre,
        post,
        explained_identity_reasons={
            "ufc_active_roster_official": {
                "intentionally_removed_cached_rows": expected_removals,
            },
        },
    )

    assert guard["ok"] is True
    assert guard["violations"] == []
    assert guard["explained_drops"] == [
        {
            "artifact": "ufc_active_roster_official",
            "path": "ufc_active_roster_official.csv",
            "pre_rows": 1004,
            "post_rows": 971,
            "rows_lost": 33,
            "identity_rows_lost": 33,
            "explained_rows_lost": 33,
            "explained_identity_rows_lost": 33,
            "explanations": {
                "intentionally_removed_cached_rows": 33,
            },
        }
    ]


def test_row_drop_guard_accepts_explained_removals_hidden_by_additions():
    pre_identities = [
        _row_guard_fighter_identity(index)
        for index in range(10)
    ]
    expected_removals = pre_identities[:2]
    additions = [
        _row_guard_fighter_identity("new-a"),
        _row_guard_fighter_identity("new-b"),
    ]
    pre = {
        "ufc_active_roster_official": {
            "path": "ufc_active_roster_official.csv",
            "row_count": 10,
            "identity_rows": pre_identities,
        },
    }
    post = {
        "ufc_active_roster_official": {
            "path": "ufc_active_roster_official.csv",
            "row_count": 10,
            "identity_rows": [*pre_identities[2:], *additions],
        },
    }

    guard = scheduled_refresh._build_row_drop_guard(
        pre,
        post,
        explained_identity_reasons={
            "ufc_active_roster_official": {
                "intentionally_removed_cached_rows": expected_removals,
            },
        },
    )

    assert guard["ok"] is True
    assert guard["violations"] == []
    assert guard["explained_drops"] == [
        {
            "artifact": "ufc_active_roster_official",
            "path": "ufc_active_roster_official.csv",
            "pre_rows": 10,
            "post_rows": 10,
            "rows_lost": 0,
            "identity_rows_lost": 2,
            "explained_identity_rows_lost": 2,
            "explanations": {
                "intentionally_removed_cached_rows": 2,
            },
        }
    ]


def test_row_drop_guard_reports_net_zero_mixed_churn():
    pre_identities = [
        _row_guard_fighter_identity(index)
        for index in range(10)
    ]
    expected_removals = pre_identities[:3]
    unexpected_removals = pre_identities[3:5]
    additions = [
        _row_guard_fighter_identity(f"new-{index}")
        for index in range(5)
    ]
    pre = {
        "ufc_active_roster_official": {
            "path": "ufc_active_roster_official.csv",
            "row_count": 10,
            "identity_rows": pre_identities,
        },
    }
    post = {
        "ufc_active_roster_official": {
            "path": "ufc_active_roster_official.csv",
            "row_count": 10,
            "identity_rows": [*pre_identities[5:], *additions],
        },
    }

    guard = scheduled_refresh._build_row_drop_guard(
        pre,
        post,
        explained_identity_reasons={
            "ufc_active_roster_official": {
                "intentionally_removed_cached_rows": expected_removals,
            },
        },
    )

    assert guard["ok"] is False
    assert guard["explained_drops"] == []
    violation = guard["violations"][0]
    assert violation["rows_lost"] == 0
    assert violation["identity_rows_lost"] == 5
    assert violation["explained_identity_rows_lost"] == 3
    assert violation["unexpected_rows_lost"] == 2
    assert violation["unexpected_identity_rows_lost"] == 2
    assert violation["unexpected_identities"] == [
        {
            "official_name": row["official_name"],
            "official_athlete_url": row["official_athlete_url"],
            "ufcstats_url": row["ufcstats_url"],
        }
        for row in unexpected_removals
    ]


def test_row_drop_guard_reports_mixed_churn_unexplained_identities():
    pre_identities = [
        _row_guard_fighter_identity(index)
        for index in range(1004)
    ]
    expected_removals = pre_identities[:33]
    unexpected_removals = pre_identities[33:35]
    additions = [
        _row_guard_fighter_identity("new-a"),
        _row_guard_fighter_identity("new-b"),
    ]
    pre = {
        "ufc_active_roster_official": {
            "path": "ufc_active_roster_official.csv",
            "row_count": 1004,
            "identity_rows": pre_identities,
        },
    }
    post = {
        "ufc_active_roster_official": {
            "path": "ufc_active_roster_official.csv",
            "row_count": 971,
            "identity_rows": [*pre_identities[35:], *additions],
        },
    }

    guard = scheduled_refresh._build_row_drop_guard(
        pre,
        post,
        explained_identity_reasons={
            "ufc_active_roster_official": {
                "intentionally_removed_cached_rows": expected_removals,
            },
        },
    )

    assert guard["ok"] is False
    assert guard["explained_drops"] == []
    assert guard["violations"] == [
        {
            "artifact": "ufc_active_roster_official",
            "path": "ufc_active_roster_official.csv",
            "pre_rows": 1004,
            "post_rows": 971,
            "rows_lost": 33,
            "identity_rows_lost": 35,
            "explained_rows_lost": 33,
            "explained_identity_rows_lost": 33,
            "explanations": {
                "intentionally_removed_cached_rows": 33,
            },
            "unexpected_rows_lost": 2,
            "unexpected_identity_rows_lost": 2,
            "unexpected_identities": [
                {
                    "official_name": unexpected_removals[0]["official_name"],
                    "official_athlete_url": unexpected_removals[0][
                        "official_athlete_url"
                    ],
                    "ufcstats_url": unexpected_removals[0]["ufcstats_url"],
                },
                {
                    "official_name": unexpected_removals[1]["official_name"],
                    "official_athlete_url": unexpected_removals[1][
                        "official_athlete_url"
                    ],
                    "ufcstats_url": unexpected_removals[1]["ufcstats_url"],
                },
            ],
        }
    ]


def test_build_profile_audit_alert_summary_treats_age_without_any_dob_source_as_source_limited(tmp_path):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    pd.DataFrame(
        [
            {"official_name": "Age Gap Fighter", "octagon_debut": "Jan. 1, 2025"},
        ]
    ).to_csv(roster_path, index=False)

    audit_df = pd.DataFrame(
        [
            {
                "official_name": "Age Gap Fighter",
                "split_official_name": "newly_added_active_roster",
                "age_present": False,
                "weight_present": True,
                "division_present": True,
                "height_present": True,
                "reach_present": True,
                "stance_present": True,
                "full_physical_bundle_present": False,
            },
        ]
    )
    unresolved_df = pd.DataFrame(
        [
            {
                "official_name": "Age Gap Fighter",
                "field": "age",
                "why_missing_code": "official_age_blank_and_no_dob_source",
            },
        ]
    )

    summary = scheduled_refresh._build_profile_audit_alert_summary(
        active_roster_path=roster_path,
        audit_df=audit_df,
        unresolved_df=unresolved_df,
        as_of_utc=datetime(2026, 3, 27, tzinfo=timezone.utc),
        new_fighter_grace_days=7,
    )

    assert summary["source_limited_missing_fields"]["full_physical_bundle"] == {
        "rows_missing": 1,
        "rows_source_limited": 1,
        "source_limited_only": True,
    }
