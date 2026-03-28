import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import scripts.run_scheduled_ufc_refresh as scheduled_refresh
from scripts.backfill_active_roster_ufcstats import FIGHTERS_PATH as BACKFILL_FIGHTERS_PATH
from src.config import RAW_DATA_DIR
from src.data.io_utils import write_csv_atomically


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
    assert calls["rebuild"] == {
        "dataset_variant": "pulled_all_plus_legacy_market",
        "output_subdirs": scheduled_refresh.DEFAULT_REBUILD_OUTPUT_SUBDIRS,
        "update_production_manifest": True,
    }
    assert calls["audit"]["active_roster_path"] == roster_path
    assert calls["audit"]["processed_fights_path"] == processed_dir / "fights_cleaned.csv"
    assert calls["audit"]["scraped_fighters_path"] == raw_dir / "ufc_fighters_scraped.csv"

    assert summary["roster_sync"]["rows"] == 1
    assert summary["ufcstats_backfill"]["new_result_rows"] == 1
    assert summary["rebuild"]["outputs"][0]["fight_rows"] == 10
    assert summary["rebuild"]["production_bundle"]["bundle_id"] == "bundle-1"
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


def test_run_scheduled_refresh_targets_new_active_roster_profile_gaps_before_rebuild(tmp_path, monkeypatch):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    calls: dict[str, object] = {"order": []}

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

    def fake_run_profile_supplement_refresh(*, scraped_fighters_path, candidate_source_csv, output_path, sources, limit):
        calls["order"].append("supplement")
        calls["supplement_candidate_rows"] = pd.read_csv(candidate_source_csv).to_dict(orient="records")
        calls["supplement_sources"] = list(sources)
        calls["supplement_limit"] = limit
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

    assert calls["order"] == ["sync", "backfill", "audit_1", "supplement", "rebuild", "audit_2"]
    assert calls["supplement_candidate_rows"] == [
        {"official_name": "Gap Fighter", "ufcstats_url": "http://ufcstats.test/gap", "profile_status": "ok"}
    ]
    assert calls["supplement_sources"] == list(scheduled_refresh.DEFAULT_PROFILE_SUPPLEMENT_REFRESH_SOURCES)
    assert calls["supplement_limit"] is None
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
