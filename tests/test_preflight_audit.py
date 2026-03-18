from __future__ import annotations

import src.strategy.preflight_audit as preflight_audit


def test_collect_matrix_status_reports_expected_cell_counts(monkeypatch):
    monkeypatch.setattr(preflight_audit, "ALL_VARIANTS", {"baseline": object(), "betsapi_challenger": object()})
    monkeypatch.setattr(preflight_audit, "ALL_FEATURE_FAMILIES", ("production", "production_betsapi"))
    monkeypatch.setattr(preflight_audit, "TRAINING_DATASET_VARIANTS", ("append_only_2026", "legacy_only"))
    monkeypatch.setattr(
        preflight_audit,
        "_default_variant_names",
        lambda **kwargs: ["baseline"],
    )
    monkeypatch.setattr(
        preflight_audit,
        "_default_feature_families",
        lambda **kwargs: ["production"],
    )

    status = preflight_audit.collect_matrix_status(has_betsapi_backfill=False)

    assert status["full_matrix_cells"] == 8
    assert status["default_matrix_cells"] == 2
    assert status["excluded_variants"] == ["betsapi_challenger"]
    assert status["excluded_families"] == ["production_betsapi"]


def test_render_preflight_report_includes_readiness_and_commands():
    report = preflight_audit.render_preflight_report(
        {
            "generated_at": "2026-03-15T14:30:00",
            "ready_for_full_matrix": False,
            "freeze_status": {
                "freeze_id": "20260315",
                "integrity": {"valid": True},
                "selection": {"ready": True, "errors": []},
                "promotion": {"ready": True, "errors": []},
                "metrics": {"brier": 0.22},
                "sweep": {"roi": 0.10},
                "trading_rows": {"bet_log": 386, "bankroll_history": 1386},
            },
            "betsapi_status": {
                "ready": False,
                "raw_event_payload_files": 1554,
                "raw_summary_payload_files": 1527,
                "raw_summary_payload_files_total": 1540,
                "raw_summary_payload_files_noncanonical": 13,
                "requested_ufc_event_ids": 2694,
                "hydrated_summary_ids": 1527,
                "missing_summary_ids": 1167,
                "missing_summary_sample": ["5079105"],
                "usable_summary_event_ids": 1200,
                "usable_event_snapshot_ids": 1180,
                "invalid_moneyline_rows": 9,
                "invalid_moneyline_event_ids": 4,
                "invalid_moneyline_bookmakers": {"VirginBet": 7, "Polymarket": 2},
                "processed_artifacts": {"backfill_manifest.json": False},
                "manifest": None,
                "quality_errors": ["processed odds_summary_rows.csv still contains 9 non-decimal moneyline rows"],
            },
            "betsapi_runtime_validation": {
                "ready": False,
                "checked": True,
                "families": ["production_betsapi"],
                "datasets": ["append_only_2026"],
                "representative_profiles": {"build_features": "baseline"},
                "error": "production_betsapi: all added BetsAPI columns are null",
            },
            "matrix_status": {
                "datasets": ["append_only_2026"],
                "all_variant_count": 26,
                "all_family_count": 6,
                "full_matrix_cells": 936,
                "default_variant_count": 25,
                "default_family_count": 3,
                "default_matrix_cells": 450,
                "excluded_variants": ["betsapi_challenger"],
                "excluded_families": ["production_betsapi"],
            },
            "runtime": {
                "git_sha": "deadbeef",
                "git_dirty": True,
                "source_fingerprint": "abc123",
            },
        }
    )

    assert "Ready for full matrix: False" in report
    assert "Full matrix cells when fully ready: 936" in report
    assert "Invalid moneyline rows in processed summary data: 9" in report
    assert "betsapi-mma-backfill" in report
    assert "run_evaluation --freeze-id 20260315" in report
    assert "historical evaluation matrix intentionally excludes live snapshot-dependent BetsAPI scope" in report
    assert "preflight and code audit both pass" in report


def test_collect_preflight_status_requires_clean_git(monkeypatch):
    monkeypatch.setattr(
        preflight_audit,
        "collect_betsapi_status",
        lambda: {"ready": True},
    )
    monkeypatch.setattr(
        preflight_audit,
        "collect_matrix_status",
        lambda has_betsapi_backfill: {"has_betsapi_backfill": has_betsapi_backfill},
    )
    monkeypatch.setattr(
        preflight_audit,
        "collect_betsapi_runtime_validation",
        lambda has_betsapi_backfill: {"ready": True, "checked": True},
    )
    monkeypatch.setattr(
        preflight_audit,
        "collect_freeze_status",
        lambda freeze_id: {
            "selection": {"ready": True},
            "promotion": {"ready": True},
        },
    )
    monkeypatch.setattr(
        preflight_audit,
        "_collect_runtime_code_metadata",
        lambda: {"git_sha": "deadbeef", "git_dirty": True, "source_fingerprint": "abc123"},
    )

    status = preflight_audit.collect_preflight_status("20260315")

    assert status["ready_for_full_matrix"] is False


def test_collect_betsapi_runtime_validation_uses_default_historical_families(monkeypatch):
    monkeypatch.setattr(
        preflight_audit,
        "_default_feature_families",
        lambda **kwargs: ["production", "production_betsapi"],
    )
    monkeypatch.setattr(
        preflight_audit,
        "validate_betsapi_preflight_readiness",
        lambda **kwargs: kwargs,
    )

    result = preflight_audit.collect_betsapi_runtime_validation(
        has_betsapi_backfill=True,
    )

    assert result["dataset_names"] == list(preflight_audit.TRAINING_DATASET_VARIANTS)
    assert result["feature_families"] == ["production", "production_betsapi"]


def test_collect_preflight_status_requires_runtime_betsapi_validation(monkeypatch):
    monkeypatch.setattr(
        preflight_audit,
        "collect_betsapi_status",
        lambda: {"ready": True},
    )
    monkeypatch.setattr(
        preflight_audit,
        "collect_matrix_status",
        lambda has_betsapi_backfill: {"has_betsapi_backfill": has_betsapi_backfill},
    )
    monkeypatch.setattr(
        preflight_audit,
        "collect_betsapi_runtime_validation",
        lambda has_betsapi_backfill: {"ready": False, "checked": True, "error": "bad signal"},
    )
    monkeypatch.setattr(
        preflight_audit,
        "collect_freeze_status",
        lambda freeze_id: {
            "selection": {"ready": True},
            "promotion": {"ready": True},
        },
    )
    monkeypatch.setattr(
        preflight_audit,
        "_collect_runtime_code_metadata",
        lambda: {"git_sha": "deadbeef", "git_dirty": False, "source_fingerprint": "abc123"},
    )

    status = preflight_audit.collect_preflight_status("20260315")

    assert status["ready_for_full_matrix"] is False
    assert status["betsapi_runtime_validation"]["error"] == "bad signal"
