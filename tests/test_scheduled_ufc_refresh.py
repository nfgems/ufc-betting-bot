import json

import pandas as pd
import pytest

import scripts.run_scheduled_ufc_refresh as scheduled_refresh
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
        return pd.DataFrame(
            [
                {
                    "official_name": "Alpha Fighter",
                    "ufcstats_url": "http://ufcstats.test/alpha",
                    "ufcstats_name": "Alpha Fighter",
                    "profile_status": "ok",
                }
            ]
        )

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

    summary = scheduled_refresh.run_scheduled_refresh(
        dataset_variant="pulled_all_plus_legacy_market",
        output_subdirs=None,
        limit_fighters=25,
        audit_json_path=audit_json_path,
        audit_csv_path=audit_csv_path,
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
