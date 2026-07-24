from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

import scripts.refresh_tapology_profile_supplement as hosted_refresh
from scripts.build_profile_supplement_from_external_profiles import (
    _candidate_source_row_is_coverage_eligible,
)


def _healthy_discovery(candidate_url: str):
    def fake_search(_name, limit=3, *, diagnostics=None):
        if diagnostics is not None:
            diagnostics.update(
                {
                    "healthy": True,
                    "candidate_count": 1,
                    "attempts": [
                        {"channel": "reader", "query": _name, "result": "scored"}
                    ],
                }
            )
        return [candidate_url][:limit]

    return fake_search


def _args(tmp_path: Path, **overrides) -> Namespace:
    values = {
        "probe_name": "Feng Pengchao",
        "probe_url": "https://www.tapology.com/fightcenter/fighters/example",
        "probe_only": False,
        "sync_active_roster": False,
        "full_active_roster_sync": False,
        "active_roster_path": tmp_path / "roster.csv",
        "scraped_fighters_path": tmp_path / "fighters.csv",
        "output": tmp_path / "supplement.csv",
        "limit": 20,
        "candidate_offset": None,
        "require_source_health": True,
        "json": False,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    "false_value",
    [False, 0, 0.0, "0", "0.0", "0.00", "0e0", "-0.0"],
)
def test_candidate_coverage_gate_preserves_explicit_numeric_false(false_value):
    row = {
        "official_name": "Blocked Candidate",
        "official_athlete_url": (
            "https://www.ufc.com/athlete/blocked-candidate"
        ),
        "profile_status": "Active",
        "combat_sport": "mma",
        "official_url_identity_status": "valid",
        "official_url_identity_valid": True,
        "coverage_eligible": false_value,
    }

    assert not _candidate_source_row_is_coverage_eligible(row)


@pytest.mark.parametrize(
    "row",
    [
        {
            "profile_status": "Active",
            "combat_sport": "",
            "coverage_eligible": "not-a-boolean",
        },
        {
            "profile_status": "Active",
            "combat_sport": "power_slap",
            "coverage_eligible": True,
        },
        {
            "profile_status": "Active",
            "combat_sport": "mma",
            "coverage_eligible": True,
            "official_url_identity_status": "mismatch",
        },
        {
            "profile_status": "Released",
            "combat_sport": "mma",
            "coverage_eligible": True,
        },
    ],
)
def test_candidate_coverage_gate_blocks_ambiguous_or_negative_roster_rows(row):
    candidate = {
        "official_name": "Blocked Candidate",
        "official_athlete_url": (
            "https://www.ufc.com/athlete/blocked-candidate"
        ),
        "official_url_identity_status": "valid",
        "official_url_identity_valid": True,
        **row,
    }

    assert not _candidate_source_row_is_coverage_eligible(candidate)


@pytest.mark.parametrize("missing_value", ["", float("nan"), "not-a-boolean"])
def test_candidate_coverage_gate_repairs_only_positive_current_mma(
    missing_value,
):
    candidate = {
        "official_name": "Verified Candidate",
        "official_athlete_url": (
            "https://www.ufc.com/athlete/verified-candidate"
        ),
        "profile_status": "Active",
        "combat_sport": "mma",
        "official_url_identity_status": "valid",
        "official_url_identity_valid": True,
        "coverage_eligible": missing_value,
    }

    assert _candidate_source_row_is_coverage_eligible(candidate)


def test_nonroster_candidate_invalid_coverage_is_not_treated_as_true():
    assert not _candidate_source_row_is_coverage_eligible(
        {
            "name": "Ambiguous Candidate",
            "coverage_eligible": "not-a-boolean",
        }
    )


def test_probe_failure_classifies_hosted_cloudflare_block(monkeypatch):
    profile_url = "https://www.tapology.com/fightcenter/fighters/example"
    monkeypatch.setattr(hosted_refresh, "clear_fallback_cache", lambda **_kwargs: None)
    monkeypatch.setattr(
        hosted_refresh,
        "search_tapology_candidates",
        _healthy_discovery(profile_url),
    )
    monkeypatch.setattr(
        hosted_refresh,
        "scrape_tapology_profile",
        lambda _url: (_ for _ in ()).throw(
        hosted_refresh.TapologyRequestError(
                _url,
                status_code=403,
                detail="Cloudflare challenge",
            )
        ),
    )

    summary = hosted_refresh.probe_tapology_access(
        fighter_name="Feng Pengchao",
        fighter_url=profile_url,
    )

    assert summary["ok"] is False
    assert summary["failure_kind"] == "hosted_egress_blocked"


def test_probe_rejects_profile_with_fields_but_no_identity(monkeypatch):
    profile_url = "https://www.tapology.com/fightcenter/fighters/example"
    monkeypatch.setattr(hosted_refresh, "clear_fallback_cache", lambda **_kwargs: None)
    monkeypatch.setattr(
        hosted_refresh,
        "search_tapology_candidates",
        _healthy_discovery(profile_url),
    )
    monkeypatch.setattr(
        hosted_refresh,
        "scrape_tapology_profile",
        lambda _url: {"name": "", "height": "5' 11\"", "reach": "74 in"},
    )

    summary = hosted_refresh.probe_tapology_access(
        fighter_name="Feng Pengchao",
        fighter_url=profile_url,
    )

    assert summary["ok"] is False
    assert summary["failure_kind"] == "profile_fetch_or_parse_failed"
    assert "without a fighter identity" in summary["errors"][0]


def test_probe_rejects_working_known_profile_when_discovery_is_unhealthy(monkeypatch):
    scrape_calls = []
    monkeypatch.setattr(hosted_refresh, "clear_fallback_cache", lambda **_kwargs: None)

    def failed_discovery(_name, limit=3, *, diagnostics=None):
        diagnostics.update(
            {
                "healthy": False,
                "candidate_count": 0,
                "attempts": [
                    {
                        "channel": "reader",
                        "query": _name,
                        "result": "failed:reader unavailable",
                    }
                ],
            }
        )
        return []

    monkeypatch.setattr(hosted_refresh, "search_tapology_candidates", failed_discovery)
    monkeypatch.setattr(
        hosted_refresh,
        "scrape_tapology_profile",
        lambda url: scrape_calls.append(url) or {
            "name": "Feng Pengchao",
            "height": 180.0,
        },
    )

    summary = hosted_refresh.probe_tapology_access(
        fighter_name="Feng Pengchao",
        fighter_url="https://www.tapology.com/fightcenter/fighters/example",
    )

    assert summary["ok"] is False
    assert summary["failure_kind"] == "discovery_failed"
    assert scrape_calls == []


def test_hosted_refresh_rotates_backlog_from_github_run_number(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "4")
    monkeypatch.setattr(
        hosted_refresh,
        "probe_tapology_access",
        lambda **_kwargs: {"ok": True, "fighter_name": "Feng Pengchao"},
    )

    def fake_refresh(**kwargs):
        captured.update(kwargs)
        return {
            "candidate_universe_rows": 40,
            "recoverable_candidate_rows": 40,
            "attempted_rows": 20,
            "recovered_rows": 1,
            "source_error_count": 0,
        }

    monkeypatch.setattr(hosted_refresh, "run_profile_supplement_refresh", fake_refresh)

    summary = hosted_refresh.run_hosted_refresh(_args(tmp_path))

    assert captured["candidate_offset"] == 0
    assert captured["candidate_rotation_index"] == 3
    assert summary["progress_state"] == "recovered"
    assert summary["source_health_ok"] is True


def test_hosted_refresh_marks_zero_progress_unhealthy(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hosted_refresh,
        "probe_tapology_access",
        lambda **_kwargs: {"ok": True, "fighter_name": "Feng Pengchao"},
    )
    monkeypatch.setattr(
        hosted_refresh,
        "run_profile_supplement_refresh",
        lambda **_kwargs: {
            "candidate_universe_rows": 8,
            "recoverable_candidate_rows": 8,
            "attempted_rows": 8,
            "recovered_rows": 0,
            "source_error_count": 0,
        },
    )

    summary = hosted_refresh.run_hosted_refresh(_args(tmp_path))

    assert summary["action"] == "refreshed"
    assert summary["progress_state"] == "no_fields_available"
    assert summary["source_health_ok"] is True


def test_hosted_refresh_marks_candidate_source_error_unhealthy(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hosted_refresh,
        "probe_tapology_access",
        lambda **_kwargs: {"ok": True, "fighter_name": "Feng Pengchao"},
    )
    monkeypatch.setattr(
        hosted_refresh,
        "run_profile_supplement_refresh",
        lambda **_kwargs: {
            "candidate_universe_rows": 8,
            "recoverable_candidate_rows": 8,
            "attempted_rows": 1,
            "recovered_rows": 0,
            "source_error_count": 1,
        },
    )

    summary = hosted_refresh.run_hosted_refresh(_args(tmp_path))

    assert summary["progress_state"] == "source_error"
    assert summary["source_health_ok"] is False


def test_hosted_refresh_stops_before_refresh_when_probe_is_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hosted_refresh,
        "probe_tapology_access",
        lambda **_kwargs: {
            "ok": False,
            "failure_kind": "hosted_egress_blocked",
            "errors": ["status 403: Cloudflare challenge"],
        },
    )
    monkeypatch.setattr(
        hosted_refresh,
        "run_profile_supplement_refresh",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("refresh must not run")),
    )

    summary = hosted_refresh.run_hosted_refresh(_args(tmp_path))

    assert summary["action"] == "access_probe_failed"
    assert summary["progress_state"] == "source_error"
    assert summary["source_health_ok"] is False


def test_hosted_refresh_fails_when_candidate_universe_is_empty(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        hosted_refresh,
        "probe_tapology_access",
        lambda **_kwargs: {"ok": True, "fighter_name": "Feng Pengchao"},
    )
    monkeypatch.setattr(
        hosted_refresh,
        "run_profile_supplement_refresh",
        lambda **_kwargs: {
            "candidate_universe_rows": 0,
            "gap_candidate_rows": 0,
            "recoverable_candidate_rows": 0,
            "candidate_rows": 0,
            "attempted_rows": 0,
            "recovered_rows": 0,
            "source_error_count": 0,
        },
    )

    summary = hosted_refresh.run_hosted_refresh(_args(tmp_path))

    assert summary["action"] == "configuration_error"
    assert summary["progress_state"] == "configuration_error"
    assert summary["source_health_ok"] is False
    assert "zero eligible fighter rows" in summary["error"]


def test_hosted_refresh_keeps_complete_nonempty_universe_healthy(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        hosted_refresh,
        "probe_tapology_access",
        lambda **_kwargs: {"ok": True, "fighter_name": "Feng Pengchao"},
    )
    monkeypatch.setattr(
        hosted_refresh,
        "run_profile_supplement_refresh",
        lambda **_kwargs: {
            "candidate_universe_rows": 12,
            "gap_candidate_rows": 0,
            "recoverable_candidate_rows": 0,
            "candidate_rows": 0,
            "attempted_rows": 0,
            "recovered_rows": 0,
            "source_error_count": 0,
        },
    )

    summary = hosted_refresh.run_hosted_refresh(_args(tmp_path))

    assert summary["action"] == "refreshed"
    assert summary["progress_state"] == "no_recoverable_candidates"
    assert summary["source_health_ok"] is True


def test_main_exits_nonzero_for_empty_candidate_universe(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(hosted_refresh, "parse_args", lambda: args)
    monkeypatch.setattr(
        hosted_refresh,
        "probe_tapology_access",
        lambda **_kwargs: {"ok": True, "fighter_name": "Feng Pengchao"},
    )
    monkeypatch.setattr(
        hosted_refresh,
        "run_profile_supplement_refresh",
        lambda **_kwargs: {
            "candidate_universe_rows": 0,
            "gap_candidate_rows": 0,
            "recoverable_candidate_rows": 0,
            "candidate_rows": 0,
            "attempted_rows": 0,
            "recovered_rows": 0,
            "source_error_count": 0,
        },
    )

    assert hosted_refresh.main() == 4


def test_workflow_diagnostics_artifact_name_is_retry_safe():
    workflow = (
        hosted_refresh.REPO_ROOT
        / ".github"
        / "workflows"
        / "tapology-profile-refresh.yml"
    ).read_text(encoding="utf-8")

    assert (
        "name: tapology-refresh-diagnostics-${{ github.run_id }}-"
        "${{ github.run_attempt }}"
    ) in workflow


def test_hosted_refresh_rejects_nonpositive_limit_before_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hosted_refresh,
        "probe_tapology_access",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("probe must not run")),
    )

    summary = hosted_refresh.run_hosted_refresh(_args(tmp_path, limit=0))

    assert summary["action"] == "configuration_error"
    assert summary["source_health_ok"] is False
    assert "positive integer" in summary["error"]


def test_hosted_refresh_prioritizes_actual_current_card_names(tmp_path, monkeypatch):
    roster_path = tmp_path / "roster.csv"
    roster_path.write_text(
        "name,official_name,profile_refresh_priority\n"
        "Legacy Ian,Ian Garry,0\n"
        "Backlog Fighter,Backlog Fighter,10\n",
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(
        hosted_refresh,
        "probe_tapology_access",
        lambda **_kwargs: {"ok": True, "fighter_name": "Feng Pengchao"},
    )
    monkeypatch.setattr(
        hosted_refresh,
        "_load_upcoming_fighter_names",
        lambda: (["Ian Garry"], 1),
    )

    def fake_refresh(**kwargs):
        captured["candidates"] = hosted_refresh.pd.read_csv(kwargs["candidate_source_csv"])
        return {
            "candidate_universe_rows": 2,
            "recoverable_candidate_rows": 2,
            "attempted_rows": 2,
            "recovered_rows": 1,
            "source_error_count": 0,
        }

    monkeypatch.setattr(hosted_refresh, "run_profile_supplement_refresh", fake_refresh)

    summary = hosted_refresh.run_hosted_refresh(
        _args(tmp_path, active_roster_path=roster_path)
    )

    priorities = captured["candidates"].set_index("name")["profile_refresh_priority"]
    assert priorities["Legacy Ian"] >= 100
    assert priorities["Backlog Fighter"] == 10
    assert summary["candidate_priority"]["matched_candidate_rows"] == 1


def test_default_hosted_roster_sync_fetches_statuses_and_keeps_candidates(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "roster.csv"
    captured = {}
    pd.DataFrame(
        [
            {
                "name": "Eligible Fighter",
                "official_name": "Eligible Fighter",
                "official_athlete_url": "https://www.ufc.com/athlete/eligible-fighter",
                "profile_status": "Active",
                "coverage_eligible": True,
            }
        ]
    ).to_csv(roster_path, index=False)
    monkeypatch.setattr(
        hosted_refresh,
        "probe_tapology_access",
        lambda **_kwargs: {"ok": True, "fighter_name": "Feng Pengchao"},
    )
    monkeypatch.setattr(
        hosted_refresh,
        "_load_upcoming_fighter_names",
        lambda: ([], 0),
    )

    def fake_sync(**kwargs):
        assert kwargs["fetch_profile_details"] is False
        assert kwargs["resolve_ufcstats"] is False
        assert kwargs["classify_combat_sport"] is False
        assert kwargs["identity_audit_path"] is None
        roster = pd.DataFrame(
            [
                {
                    "name": "Eligible Fighter",
                    "official_name": "Eligible Fighter",
                    "official_athlete_url": "https://www.ufc.com/athlete/eligible-fighter",
                    "profile_status": "status_unknown",
                    "coverage_eligible": False,
                }
            ]
        )
        roster.to_csv(kwargs["output_path"], index=False)
        return roster

    def fake_refresh(**kwargs):
        candidates = pd.read_csv(kwargs["candidate_source_csv"])
        captured["names"] = candidates["official_name"].tolist()
        return {
            "candidate_universe_rows": 1,
            "recoverable_candidate_rows": 1,
            "attempted_rows": 1,
            "recovered_rows": 1,
            "source_error_count": 0,
        }

    monkeypatch.setattr(hosted_refresh, "sync_official_active_roster", fake_sync)
    monkeypatch.setattr(hosted_refresh, "run_profile_supplement_refresh", fake_refresh)

    summary = hosted_refresh.run_hosted_refresh(
        _args(
            tmp_path,
            sync_active_roster=True,
            full_active_roster_sync=False,
            active_roster_path=roster_path,
        )
    )

    assert captured["names"] == ["Eligible Fighter"]
    assert summary["active_roster"]["action"] == "synced_fast_verified_cache_join"
    assert summary["active_roster"]["verified_candidate_rows"] == 1


def test_booked_blank_newcomer_becomes_temporary_verified_candidate(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "roster.csv"
    pd.DataFrame(
        [
            {
                "official_name": "Ian Machado Garry",
                "official_athlete_url": "https://www.ufc.com/athlete/ian-machado-garry",
                "profile_status": "status_unknown",
                "coverage_eligible": False,
            }
        ]
    ).to_csv(roster_path, index=False)
    captured = {}
    monkeypatch.setattr(
        hosted_refresh,
        "probe_tapology_access",
        lambda **_kwargs: {"ok": True, "fighter_name": "Feng Pengchao"},
    )
    monkeypatch.setattr(
        hosted_refresh,
        "_load_upcoming_fighter_names",
        lambda: (["Ian Garry"], 1),
    )

    def fake_sync(**kwargs):
        roster = pd.DataFrame(
            [
                {
                    "official_name": "Ian Machado Garry",
                    "official_athlete_url": (
                        "https://www.ufc.com/athlete/ian-machado-garry"
                    ),
                    "profile_status": "",
                    "coverage_eligible": False,
                }
            ]
        )
        roster.to_csv(kwargs["output_path"], index=False)
        return roster

    def fake_refresh(**kwargs):
        captured["candidates"] = pd.read_csv(kwargs["candidate_source_csv"])
        return {
            "candidate_universe_rows": 1,
            "recoverable_candidate_rows": 1,
            "attempted_rows": 1,
            "recovered_rows": 1,
            "source_error_count": 0,
        }

    monkeypatch.setattr(hosted_refresh, "sync_official_active_roster", fake_sync)
    monkeypatch.setattr(hosted_refresh, "run_profile_supplement_refresh", fake_refresh)

    summary = hosted_refresh.run_hosted_refresh(
        _args(
            tmp_path,
            sync_active_roster=True,
            active_roster_path=roster_path,
        )
    )

    candidate = captured["candidates"].iloc[0]
    assert hosted_refresh._coverage_enabled(
        candidate["active_roster_current_verified"]
    )
    assert hosted_refresh._coverage_enabled(candidate["coverage_eligible"])
    assert candidate["active_roster_status_reason"] == (
        "verified_from_current_ufc_card"
    )
    assert _candidate_source_row_is_coverage_eligible(candidate.to_dict())
    assert summary["active_roster"]["current_card_verified_rows"] == 1
    assert summary["active_roster"]["verified_candidate_rows"] == 1

    canonical = pd.read_csv(roster_path)
    assert "active_roster_current_verified" not in canonical.columns
    assert not hosted_refresh._coverage_enabled(canonical.loc[0, "coverage_eligible"])


def test_unbooked_blank_roster_row_stays_blocked(tmp_path, monkeypatch):
    source_path = tmp_path / "roster.csv"
    output_path = tmp_path / "candidates.csv"
    pd.DataFrame(
        [
            {
                "official_name": "Unbooked Newcomer",
                "official_athlete_url": (
                    "https://www.ufc.com/athlete/unbooked-newcomer"
                ),
                "profile_status": "status_unknown",
                "coverage_eligible": False,
            }
        ]
    ).to_csv(source_path, index=False)
    monkeypatch.setattr(
        hosted_refresh,
        "_load_upcoming_fighter_names",
        lambda: (["Different Fighter"], 1),
    )

    prioritized_path, summary = hosted_refresh._prioritize_current_card_candidates(
        source_path,
        output_path=output_path,
    )

    candidate = pd.read_csv(prioritized_path).iloc[0]
    assert not hosted_refresh._coverage_enabled(
        candidate["active_roster_current_verified"]
    )
    assert not hosted_refresh._coverage_enabled(candidate["coverage_eligible"])
    assert not _candidate_source_row_is_coverage_eligible(candidate.to_dict())
    assert summary["current_card_verified_rows"] == 0


@pytest.mark.parametrize(
    ("profile_status", "combat_sport", "combat_sport_reason"),
    [
        ("Inactive", "", ""),
        ("Cut", "", ""),
        ("Retired", "", ""),
        ("Released", "", ""),
        ("status_unknown", "power_slap", ""),
        ("status_unknown", "", "powerslap_profile_match"),
    ],
)
def test_current_card_never_overrides_inactive_or_power_slap_evidence(
    tmp_path,
    monkeypatch,
    profile_status,
    combat_sport,
    combat_sport_reason,
):
    source_path = tmp_path / "roster.csv"
    output_path = tmp_path / "candidates.csv"
    pd.DataFrame(
        [
            {
                "official_name": "Blocked Fighter",
                "official_athlete_url": (
                    "https://www.ufc.com/athlete/blocked-fighter"
                ),
                "profile_status": profile_status,
                "combat_sport": combat_sport,
                "combat_sport_reason": combat_sport_reason,
                "coverage_eligible": True,
            }
        ]
    ).to_csv(source_path, index=False)
    monkeypatch.setattr(
        hosted_refresh,
        "_load_upcoming_fighter_names",
        lambda: (["Blocked Fighter"], 1),
    )

    prioritized_path, summary = hosted_refresh._prioritize_current_card_candidates(
        source_path,
        output_path=output_path,
    )

    candidate = pd.read_csv(prioritized_path).iloc[0]
    assert not hosted_refresh._coverage_enabled(
        candidate["active_roster_current_verified"]
    )
    assert not hosted_refresh._coverage_enabled(candidate["coverage_eligible"])
    assert not _candidate_source_row_is_coverage_eligible(candidate.to_dict())
    assert summary["current_card_verified_rows"] == 0


@pytest.mark.parametrize(
    ("identity_status", "identity_valid"),
    [
        ("mismatch", True),
        ("test_profile", True),
        ("", False),
        ("", 0),
        ("", 0.0),
    ],
)
def test_current_card_never_overrides_explicit_untrusted_live_identity(
    tmp_path,
    monkeypatch,
    identity_status,
    identity_valid,
):
    source_path = tmp_path / "roster.csv"
    output_path = tmp_path / "candidates.csv"
    pd.DataFrame(
        [
            {
                "official_name": "Untrusted Fighter",
                "official_athlete_url": (
                    "https://www.ufc.com/athlete/untrusted-fighter"
                ),
                "profile_status": "status_unknown",
                "official_url_identity_status": identity_status,
                "official_url_identity_valid": identity_valid,
                "coverage_eligible": True,
            }
        ]
    ).to_csv(source_path, index=False)
    monkeypatch.setattr(
        hosted_refresh,
        "_load_upcoming_fighter_names",
        lambda: (["Untrusted Fighter"], 1),
    )

    prioritized_path, summary = hosted_refresh._prioritize_current_card_candidates(
        source_path,
        output_path=output_path,
    )

    candidate = pd.read_csv(prioritized_path).iloc[0]
    assert not hosted_refresh._coverage_enabled(
        candidate["active_roster_current_verified"]
    )
    assert not hosted_refresh._coverage_enabled(candidate["coverage_eligible"])
    assert summary["current_card_verified_rows"] == 0


def test_ambiguous_current_card_identity_stays_blocked(tmp_path, monkeypatch):
    source_path = tmp_path / "roster.csv"
    output_path = tmp_path / "candidates.csv"
    pd.DataFrame(
        [
            {
                "official_name": "Shared Fighter",
                "official_athlete_url": "https://www.ufc.com/athlete/shared-one",
                "profile_status": "status_unknown",
                "coverage_eligible": False,
            },
            {
                "official_name": "Shared Fighter",
                "official_athlete_url": "https://www.ufc.com/athlete/shared-two",
                "profile_status": "status_unknown",
                "coverage_eligible": False,
            },
        ]
    ).to_csv(source_path, index=False)
    monkeypatch.setattr(
        hosted_refresh,
        "_load_upcoming_fighter_names",
        lambda: (["Shared Fighter"], 1),
    )

    prioritized_path, summary = hosted_refresh._prioritize_current_card_candidates(
        source_path,
        output_path=output_path,
    )

    candidates = pd.read_csv(prioritized_path)
    assert not candidates["active_roster_current_verified"].map(
        hosted_refresh._coverage_enabled
    ).any()
    assert not candidates["coverage_eligible"].map(
        hosted_refresh._coverage_enabled
    ).any()
    assert summary["ambiguous_current_card_identities"] == 1
    assert summary["current_card_verified_rows"] == 0


@pytest.mark.parametrize(
    ("live_status", "live_combat_sport"),
    [
        ("Cut", ""),
        ("", "power_slap"),
    ],
)
def test_bounded_sync_does_not_hydrate_over_live_inactive_evidence(
    tmp_path,
    monkeypatch,
    live_status,
    live_combat_sport,
):
    roster_path = tmp_path / "roster.csv"
    pd.DataFrame(
        [
            {
                "official_name": "Blocked Fighter",
                "official_athlete_url": (
                    "https://www.ufc.com/athlete/blocked-fighter"
                ),
                "profile_status": "Active",
                "combat_sport": "mma",
                "coverage_eligible": True,
            }
        ]
    ).to_csv(roster_path, index=False)
    monkeypatch.setattr(
        hosted_refresh,
        "probe_tapology_access",
        lambda **_kwargs: {"ok": True, "fighter_name": "Feng Pengchao"},
    )
    monkeypatch.setattr(
        hosted_refresh,
        "_load_upcoming_fighter_names",
        lambda: (["Blocked Fighter"], 1),
    )

    def fake_sync(**kwargs):
        roster = pd.DataFrame(
            [
                {
                    "official_name": "Blocked Fighter",
                    "official_athlete_url": (
                        "https://www.ufc.com/athlete/blocked-fighter"
                    ),
                    "profile_status": live_status,
                    "combat_sport": live_combat_sport,
                    "coverage_eligible": True,
                }
            ]
        )
        roster.to_csv(kwargs["output_path"], index=False)
        return roster

    monkeypatch.setattr(hosted_refresh, "sync_official_active_roster", fake_sync)
    monkeypatch.setattr(
        hosted_refresh,
        "run_profile_supplement_refresh",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("blocked live evidence must not enter refresh")
        ),
    )

    summary = hosted_refresh.run_hosted_refresh(
        _args(
            tmp_path,
            sync_active_roster=True,
            active_roster_path=roster_path,
        )
    )

    assert summary["action"] == "candidate_roster_unavailable"
    assert summary["active_roster"]["matched_cached_status_rows"] == 0
    assert summary["active_roster"]["verified_candidate_rows"] == 0
    assert summary["candidate_priority"]["current_card_verified_rows"] == 0


def test_bounded_fast_sync_preserves_cached_power_slap_block_before_card_override(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "roster.csv"
    temporary_path = tmp_path / "bounded.csv"
    prioritized_path = tmp_path / "prioritized.csv"
    pd.DataFrame(
        [
            {
                "official_name": "Blocked Fighter",
                "official_athlete_url": (
                    "https://www.ufc.com/athlete/blocked-fighter"
                ),
                "profile_status": "status_unknown",
                "combat_sport": "power_slap",
                "combat_sport_reason": "powerslap_profile_match",
                "coverage_eligible": False,
            }
        ]
    ).to_csv(roster_path, index=False)

    def fake_fast_sync(**kwargs):
        assert kwargs["fetch_profile_details"] is False
        assert kwargs["resolve_ufcstats"] is False
        assert kwargs["classify_combat_sport"] is False
        live = pd.DataFrame(
            [
                {
                    "official_name": "Blocked Fighter",
                    "official_athlete_url": (
                        "https://www.ufc.com/athlete/blocked-fighter"
                    ),
                    "profile_status": "status_unknown",
                    "combat_sport": "",
                    "coverage_eligible": False,
                }
            ]
        )
        live.to_csv(kwargs["output_path"], index=False)
        return live

    monkeypatch.setattr(
        hosted_refresh,
        "sync_official_active_roster",
        fake_fast_sync,
    )
    bounded_path, summary = hosted_refresh._build_bounded_active_roster_candidates(
        active_roster_path=roster_path,
        temporary_output_path=temporary_path,
    )
    bounded = pd.read_csv(bounded_path).iloc[0]

    assert bounded["combat_sport"] == "power_slap"
    assert not hosted_refresh._coverage_enabled(bounded["coverage_eligible"])
    assert summary["matched_cached_status_rows"] == 0
    assert summary["verified_candidate_rows"] == 0

    monkeypatch.setattr(
        hosted_refresh,
        "_load_upcoming_fighter_names",
        lambda: (["Blocked Fighter"], 1),
    )
    candidate_path, priority_summary = (
        hosted_refresh._prioritize_current_card_candidates(
            bounded_path,
            output_path=prioritized_path,
        )
    )
    candidate = pd.read_csv(candidate_path).iloc[0]
    assert not hosted_refresh._coverage_enabled(
        candidate["active_roster_current_verified"]
    )
    assert not hosted_refresh._coverage_enabled(candidate["coverage_eligible"])
    assert priority_summary["current_card_verified_rows"] == 0


def test_bounded_fast_sync_preserves_cached_identity_mismatch_before_card_override(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "roster.csv"
    temporary_path = tmp_path / "bounded.csv"
    prioritized_path = tmp_path / "prioritized.csv"
    pd.DataFrame(
        [
            {
                "official_name": "Untrusted Fighter",
                "official_athlete_url": (
                    "https://www.ufc.com/athlete/untrusted-fighter"
                ),
                "profile_status": "Active",
                "combat_sport": "mma",
                "official_url_identity_status": "mismatch",
                "official_url_identity_valid": 0.0,
                "official_url_identity_reason": "profile_name_mismatch",
                "coverage_eligible": True,
            }
        ]
    ).to_csv(roster_path, index=False)

    def fake_fast_sync(**kwargs):
        live = pd.DataFrame(
            [
                {
                    "official_name": "Untrusted Fighter",
                    "official_athlete_url": (
                        "https://www.ufc.com/athlete/untrusted-fighter"
                    ),
                    "profile_status": "status_unknown",
                    "combat_sport": "",
                    "official_url_identity_status": "valid",
                    "official_url_identity_valid": True,
                    "coverage_eligible": False,
                }
            ]
        )
        live.to_csv(kwargs["output_path"], index=False)
        return live

    monkeypatch.setattr(
        hosted_refresh,
        "sync_official_active_roster",
        fake_fast_sync,
    )
    bounded_path, summary = hosted_refresh._build_bounded_active_roster_candidates(
        active_roster_path=roster_path,
        temporary_output_path=temporary_path,
    )
    bounded = pd.read_csv(bounded_path).iloc[0]

    assert bounded["official_url_identity_status"] == "mismatch"
    assert (
        hosted_refresh._explicit_boolean(
            bounded["official_url_identity_valid"]
        )
        is False
    )
    assert not hosted_refresh._coverage_enabled(bounded["coverage_eligible"])
    assert summary["matched_cached_status_rows"] == 0
    assert summary["verified_candidate_rows"] == 0

    monkeypatch.setattr(
        hosted_refresh,
        "_load_upcoming_fighter_names",
        lambda: (["Untrusted Fighter"], 1),
    )
    candidate_path, priority_summary = (
        hosted_refresh._prioritize_current_card_candidates(
            bounded_path,
            output_path=prioritized_path,
        )
    )
    candidate = pd.read_csv(candidate_path).iloc[0]
    assert not hosted_refresh._coverage_enabled(
        candidate["active_roster_current_verified"]
    )
    assert not hosted_refresh._coverage_enabled(candidate["coverage_eligible"])
    assert priority_summary["current_card_verified_rows"] == 0


def test_bounded_fast_sync_preserves_reason_only_cached_hard_block(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "roster.csv"
    temporary_path = tmp_path / "bounded.csv"
    prioritized_path = tmp_path / "prioritized.csv"
    pd.DataFrame(
        [
            {
                "official_name": "Reason Blocked Fighter",
                "official_athlete_url": (
                    "https://www.ufc.com/athlete/reason-blocked-fighter"
                ),
                "profile_status": "status_unknown",
                "combat_sport": "",
                "coverage_eligible": False,
                "active_roster_status_reason": "retired_from_cached_roster",
            }
        ]
    ).to_csv(roster_path, index=False)

    def fake_fast_sync(**kwargs):
        live = pd.DataFrame(
            [
                {
                    "official_name": "Reason Blocked Fighter",
                    "official_athlete_url": (
                        "https://www.ufc.com/athlete/reason-blocked-fighter"
                    ),
                    "profile_status": "status_unknown",
                    "combat_sport": "",
                    "coverage_eligible": False,
                }
            ]
        )
        live.to_csv(kwargs["output_path"], index=False)
        return live

    monkeypatch.setattr(
        hosted_refresh,
        "sync_official_active_roster",
        fake_fast_sync,
    )
    bounded_path, summary = hosted_refresh._build_bounded_active_roster_candidates(
        active_roster_path=roster_path,
        temporary_output_path=temporary_path,
    )
    bounded = pd.read_csv(bounded_path).iloc[0]

    assert bounded["active_roster_status_reason"] == (
        "retired_from_cached_roster"
    )
    assert hosted_refresh._coverage_enabled(
        bounded[
            hosted_refresh._HOSTED_CURRENT_VERIFICATION_BLOCK_COLUMN
        ]
    )
    assert not hosted_refresh._coverage_enabled(bounded["coverage_eligible"])
    assert summary["verified_candidate_rows"] == 0

    monkeypatch.setattr(
        hosted_refresh,
        "_load_upcoming_fighter_names",
        lambda: (["Reason Blocked Fighter"], 1),
    )
    candidate_path, priority_summary = (
        hosted_refresh._prioritize_current_card_candidates(
            bounded_path,
            output_path=prioritized_path,
        )
    )
    candidate = pd.read_csv(candidate_path).iloc[0]

    assert not hosted_refresh._coverage_enabled(
        candidate["active_roster_current_verified"]
    )
    assert not hosted_refresh._coverage_enabled(candidate["coverage_eligible"])
    assert priority_summary["current_card_verified_rows"] == 0


def test_hosted_roster_sync_fails_loudly_without_verified_cached_candidates(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        hosted_refresh,
        "probe_tapology_access",
        lambda **_kwargs: {"ok": True, "fighter_name": "Feng Pengchao"},
    )
    monkeypatch.setattr(
        hosted_refresh,
        "_load_upcoming_fighter_names",
        lambda: ([], 0),
    )

    def fake_sync(**kwargs):
        roster = pd.DataFrame(
            [
                {
                    "official_name": "Unknown Fighter",
                    "official_athlete_url": "https://www.ufc.com/athlete/unknown-fighter",
                    "profile_status": "status_unknown",
                    "coverage_eligible": False,
                }
            ]
        )
        roster.to_csv(kwargs["output_path"], index=False)
        return roster

    monkeypatch.setattr(hosted_refresh, "sync_official_active_roster", fake_sync)
    monkeypatch.setattr(
        hosted_refresh,
        "run_profile_supplement_refresh",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("refresh must not run without verified candidates")
        ),
    )

    summary = hosted_refresh.run_hosted_refresh(
        _args(tmp_path, sync_active_roster=True)
    )

    assert summary["action"] == "candidate_roster_unavailable"
    assert summary["source_health_ok"] is False
    assert summary["progress_state"] == "configuration_error"
