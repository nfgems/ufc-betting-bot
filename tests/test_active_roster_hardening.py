from __future__ import annotations

import os
import sys

import pandas as pd
import pytest
import requests
from bs4 import BeautifulSoup

from scripts.sanitize_active_roster_startup import sanitize_active_roster_startup
from scripts import sync_active_ufc_roster as sync_active_roster_script
from src.data import ufc_active_roster


def _row(name: str, *, status: str = "Active") -> dict[str, object]:
    slug = name.casefold().replace(" ", "-")
    return {
        "official_name": name,
        "official_athlete_url": f"https://www.ufc.com/athlete/{slug}",
        "ufcstats_url": f"http://ufcstats.test/{slug}",
        "profile_status": status,
        "official_url_identity_valid": True,
        "official_url_identity_status": "valid",
        "combat_sport": "mma",
        "coverage_eligible": True,
    }


def _set_live_scrape(monkeypatch, rows, *, audit_rows=None):
    live_df = pd.DataFrame(rows)
    live_df.attrs["identity_audit_rows"] = list(audit_rows or [])
    live_df.attrs["sync_complete"] = True
    live_df.attrs["sync_completeness_reason"] = "test_validated_complete"
    live_df.attrs["pages_scraped"] = 1
    monkeypatch.setattr(
        ufc_active_roster,
        "scrape_official_active_roster",
        lambda **_kwargs: live_df.copy(),
    )


def test_identity_diff_runs_when_live_and_cached_counts_are_equal(tmp_path, monkeypatch):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    pd.DataFrame([_row("Anchor"), _row("Missing Fighter")]).to_csv(roster_path, index=False)
    _set_live_scrape(monkeypatch, [_row("Anchor"), _row("New Fighter")])

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )

    assert set(synced["official_name"]) == {"Anchor", "Missing Fighter", "New Fighter"}
    assert [row["official_name"] for row in synced.attrs["retained_missing_live_rows"]] == [
        "Missing Fighter"
    ]


def test_inactive_identity_audit_prevents_cached_row_resurrection(tmp_path, monkeypatch):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    pd.DataFrame([_row("Anchor"), _row("Retired Fighter")]).to_csv(roster_path, index=False)
    audit = {
        "official_name": "Retired Fighter",
        "official_athlete_url": "https://www.ufc.com/athlete/retired-fighter",
        "profile_name": "Retired Fighter",
        "slug_name": "retired fighter",
        "identity_status": "valid",
        "identity_reason": "",
        "action": "excluded_inactive_profile_status",
    }
    _set_live_scrape(monkeypatch, [_row("Anchor"), _row("New Fighter")], audit_rows=[audit])

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )

    assert set(synced["official_name"]) == {"Anchor", "New Fighter"}
    assert synced.attrs["retained_missing_live_rows"] == []


def test_inactive_identity_audit_does_not_delete_same_name_with_conflicting_url(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    old_alias = _row("Retired Fighter") | {
        "official_athlete_url": "https://www.ufc.com/athlete/retired-fighter-old"
    }
    pd.DataFrame([_row("Anchor"), old_alias]).to_csv(roster_path, index=False)
    audit = {
        "official_name": "Retired Fighter",
        "official_athlete_url": "https://www.ufc.com/athlete/retired-fighter-new",
        "profile_name": "Retired Fighter",
        "slug_name": "retired fighter",
        "identity_status": "valid",
        "identity_reason": "",
        "action": "excluded_inactive_profile_status",
    }
    _set_live_scrape(monkeypatch, [_row("Anchor")], audit_rows=[audit])

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )

    assert set(synced["official_name"]) == {"Anchor", "Retired Fighter"}
    assert [
        row["official_name"] for row in synced.attrs["retained_missing_live_rows"]
    ] == ["Retired Fighter"]


def test_exact_official_url_identity_match_wins_over_changed_ufcstats_url():
    cached = _row("Same Fighter") | {
        "ufcstats_url": "http://ufcstats.test/same-fighter-old"
    }
    live = _row("Same Fighter") | {
        "ufcstats_url": "http://ufcstats.test/same-fighter-new"
    }
    conflicting = live | {
        "official_athlete_url": "https://www.ufc.com/athlete/different-fighter"
    }

    assert ufc_active_roster._roster_rows_same_identity(cached, live) is True
    assert ufc_active_roster._roster_rows_same_identity(cached, conflicting) is False


def test_unknown_profile_status_is_quarantined_from_coverage(tmp_path, monkeypatch):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    _set_live_scrape(monkeypatch, [_row("Unknown Fighter", status="")])

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )

    assert synced.loc[0, "profile_status"] == "status_unknown"
    assert bool(synced.loc[0, "coverage_eligible"]) is False
    assert synced.loc[0, "active_roster_status_reason"] == "profile_status_missing_or_unrecognized"


def test_sitewide_successful_profile_status_loss_reuses_untouched_cached_roster(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    cached_rows = [_row(f"Fighter {index}") for index in range(100)]
    pd.DataFrame(cached_rows).to_csv(roster_path, index=False)
    original_bytes = roster_path.read_bytes()
    _set_live_scrape(
        monkeypatch,
        [
            row
            | {
                "profile_status": "",
                "official_profile_fetch_outcome": "succeeded",
            }
            for row in cached_rows
        ],
    )

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )

    assert synced.attrs["sync_source"] == "cached"
    assert synced.attrs["sync_fallback_used"] is True
    assert "profile-status markup drift" in synced.attrs["sync_error"]
    assert set(synced["profile_status"]) == {"Active"}
    assert synced["coverage_eligible"].map(bool).all()
    assert roster_path.read_bytes() == original_bytes


def test_sitewide_profile_status_loss_raises_without_fallback_or_overwrite(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    cached_rows = [_row(f"Fighter {index}") for index in range(100)]
    pd.DataFrame(cached_rows).to_csv(roster_path, index=False)
    original_bytes = roster_path.read_bytes()
    _set_live_scrape(
        monkeypatch,
        [
            row
            | {
                "profile_status": "",
                "official_profile_fetch_outcome": "succeeded",
            }
            for row in cached_rows
        ],
    )

    with pytest.raises(RuntimeError, match="profile-status markup drift"):
        ufc_active_roster.sync_official_active_roster(
            output_path=roster_path,
            identity_audit_path=None,
            allow_cached_fallback=False,
        )

    assert roster_path.read_bytes() == original_bytes


def test_cacheless_sitewide_profile_status_loss_is_not_published(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    live_rows = [
        _row(f"Fighter {index}")
        | {
            "profile_status": "",
            "official_profile_fetch_outcome": "succeeded",
        }
        for index in range(100)
    ]
    _set_live_scrape(monkeypatch, live_rows)

    with pytest.raises(RuntimeError, match="cacheless sync"):
        ufc_active_roster.sync_official_active_roster(
            output_path=roster_path,
            identity_audit_path=None,
        )

    assert not roster_path.exists()


def test_profile_status_loss_below_drift_ratio_keeps_per_row_quarantine(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    cached_rows = [_row(f"Fighter {index}") for index in range(100)]
    pd.DataFrame(cached_rows).to_csv(roster_path, index=False)
    live_rows = []
    for index, row in enumerate(cached_rows):
        live_rows.append(
            row
            | {
                "profile_status": "" if index < 79 else "Active",
                "official_profile_fetch_outcome": "succeeded",
            }
        )
    _set_live_scrape(monkeypatch, live_rows)

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )

    assert synced.attrs["sync_source"] == "live"
    assert int(synced["profile_status"].eq("status_unknown").sum()) == 79
    assert int(synced["coverage_eligible"].map(bool).sum()) == 21


def test_missing_coverage_column_is_initialized_per_row_without_disabling_active_fighters():
    rows = pd.DataFrame(
        [
            _row("Active Fighter") | {"combat_sport": "mma"},
            _row("Retired Fighter", status="Retired") | {"combat_sport": "mma"},
            _row("Power Slap Fighter") | {"combat_sport": "power_slap"},
            _row("Unknown Fighter", status="") | {"combat_sport": "mma"},
        ]
    ).drop(columns=["coverage_eligible"])

    normalized = ufc_active_roster._apply_profile_status_eligibility(rows)
    by_name = normalized.set_index("official_name")

    assert bool(by_name.at["Active Fighter", "coverage_eligible"]) is True
    assert bool(by_name.at["Retired Fighter", "coverage_eligible"]) is False
    assert bool(by_name.at["Power Slap Fighter", "coverage_eligible"]) is False
    assert bool(by_name.at["Unknown Fighter", "coverage_eligible"]) is False
    assert by_name.at["Unknown Fighter", "profile_status"] == "status_unknown"


def test_mixed_coverage_cells_repair_only_positively_verified_current_mma_rows():
    rows = pd.DataFrame(
        [
            _row("Explicit True"),
            _row("Explicit False") | {"coverage_eligible": False},
            _row("Blank Active MMA") | {"coverage_eligible": ""},
            _row("NaN Active MMA") | {"coverage_eligible": float("nan")},
            _row("Invalid Active MMA") | {"coverage_eligible": "not-a-boolean"},
            _row("Ambiguous Sport") | {
                "coverage_eligible": "",
                "combat_sport": "",
            },
            _row("Inactive Fighter", status="Inactive"),
            _row("Cut Fighter", status="Cut"),
            _row("Released Fighter", status="Released"),
            _row("Unknown Fighter", status=""),
            _row("Untrusted Fighter") | {
                "official_url_identity_valid": False,
                "official_url_identity_status": "mismatch",
            },
            _row("Power Slap Fighter") | {
                "combat_sport": "power_slap",
                "combat_sport_reason": "powerslap_profile_match",
            },
        ]
    )

    normalized = ufc_active_roster._apply_profile_status_eligibility(rows)
    eligibility = normalized.set_index("official_name")["coverage_eligible"].map(
        bool
    )

    assert bool(eligibility["Explicit True"]) is True
    assert bool(eligibility["Explicit False"]) is False
    assert bool(eligibility["Blank Active MMA"]) is True
    assert bool(eligibility["NaN Active MMA"]) is True
    assert bool(eligibility["Invalid Active MMA"]) is True
    assert bool(eligibility["Ambiguous Sport"]) is False
    assert bool(eligibility["Inactive Fighter"]) is False
    assert bool(eligibility["Cut Fighter"]) is False
    assert bool(eligibility["Released Fighter"]) is False
    assert bool(eligibility["Unknown Fighter"]) is False
    assert bool(eligibility["Untrusted Fighter"]) is False
    assert bool(eligibility["Power Slap Fighter"]) is False


@pytest.mark.parametrize(
    "identity_status",
    ["test_profile", "test profile", "test-profile", "mismatch"],
)
def test_identity_status_variants_remain_explicit_hard_blocks(identity_status):
    row = _row("Untrusted Fighter") | {
        "coverage_eligible": True,
        "official_url_identity_status": identity_status,
        "official_url_identity_valid": "",
    }

    normalized = ufc_active_roster._apply_profile_status_eligibility(
        pd.DataFrame([row])
    ).iloc[0]

    assert bool(normalized["coverage_eligible"]) is False
    assert normalized["active_roster_status_reason"] == (
        "untrusted_official_profile_identity"
    )


@pytest.mark.parametrize(
    "identity_status",
    [
        "slug_mismatch_profile_valid",
        "slug mismatch profile valid",
        "slug-mismatch-profile-valid",
    ],
)
def test_trusted_identity_status_variants_allow_positive_mma_repair(
    identity_status,
):
    row = _row("Trusted Fighter") | {
        "coverage_eligible": "",
        "official_url_identity_status": identity_status,
        "official_url_identity_valid": "",
    }

    normalized = ufc_active_roster._apply_profile_status_eligibility(
        pd.DataFrame([row])
    ).iloc[0]

    assert bool(normalized["coverage_eligible"]) is True


def test_csv_numeric_coverage_values_preserve_false_true_and_repair_blank(
    tmp_path,
):
    roster_path = tmp_path / "mixed_numeric_coverage.csv"
    pd.DataFrame(
        [
            _row("Numeric False")
            | {
                "coverage_eligible": 0.0,
                "official_url_identity_valid": 1.0,
            },
            _row("Blank Repair")
            | {
                "coverage_eligible": "",
                "official_url_identity_valid": 1.0,
            },
            _row("Numeric True")
            | {
                "coverage_eligible": 1.0,
                "official_url_identity_valid": 1.0,
            },
            _row("Numeric Identity False")
            | {
                "coverage_eligible": "",
                "official_url_identity_valid": 0.0,
                "official_url_identity_status": "valid",
            },
        ]
    ).to_csv(roster_path, index=False)
    round_tripped = pd.read_csv(roster_path)
    assert pd.api.types.is_float_dtype(round_tripped["coverage_eligible"])
    assert pd.api.types.is_float_dtype(
        round_tripped["official_url_identity_valid"]
    )

    normalized = ufc_active_roster._apply_profile_status_eligibility(
        round_tripped
    ).set_index("official_name")

    assert bool(normalized.at["Numeric False", "coverage_eligible"]) is False
    assert bool(normalized.at["Blank Repair", "coverage_eligible"]) is True
    assert bool(normalized.at["Numeric True", "coverage_eligible"]) is True
    assert (
        bool(normalized.at["Numeric Identity False", "coverage_eligible"])
        is False
    )
    untrusted_inactive = round_tripped.loc[
        round_tripped["official_name"].eq("Numeric Identity False")
    ].iloc[0].copy()
    untrusted_inactive["profile_status"] = "Inactive"
    assert (
        ufc_active_roster._verified_inactive_profile_status(untrusted_inactive)
        is False
    )


def test_csv_all_blank_status_and_reason_columns_mutate_fail_closed(tmp_path):
    roster_path = tmp_path / "blank_status_and_reason.csv"
    pd.DataFrame(
        [
            _row("Unknown Fighter", status="")
            | {
                "coverage_eligible": 1,
                "active_roster_status_reason": "",
            }
        ]
    ).to_csv(roster_path, index=False)
    round_tripped = pd.read_csv(roster_path)
    assert pd.api.types.is_float_dtype(round_tripped["profile_status"])
    assert pd.api.types.is_float_dtype(
        round_tripped["active_roster_status_reason"]
    )

    normalized = ufc_active_roster._apply_profile_status_eligibility(
        round_tripped
    ).iloc[0]

    assert normalized["profile_status"] == "status_unknown"
    assert bool(normalized["coverage_eligible"]) is False
    assert normalized["active_roster_status_reason"] == (
        "profile_status_missing_or_unrecognized"
    )


def test_csv_textual_decimal_booleans_preserve_false_and_identity_blocks(
    tmp_path,
):
    roster_path = tmp_path / "mixed_object_booleans.csv"
    pd.DataFrame(
        [
            _row("Text Decimal False")
            | {
                "coverage_eligible": "0.0",
                "official_url_identity_valid": "True",
            },
            _row("Text Decimal True")
            | {
                "coverage_eligible": "1.0",
                "official_url_identity_valid": "True",
            },
            _row("Text Decimal Identity False")
            | {
                "coverage_eligible": "True",
                "official_url_identity_valid": "0.0",
                "official_url_identity_status": "",
            },
            _row("Object Blank Repair")
            | {
                "coverage_eligible": "",
                "official_url_identity_valid": "True",
            },
        ]
    ).to_csv(roster_path, index=False)
    round_tripped = pd.read_csv(roster_path)
    assert pd.api.types.is_string_dtype(round_tripped["coverage_eligible"])
    assert pd.api.types.is_string_dtype(
        round_tripped["official_url_identity_valid"]
    )

    normalized = ufc_active_roster._apply_profile_status_eligibility(
        round_tripped
    ).set_index("official_name")

    assert bool(normalized.at["Text Decimal False", "coverage_eligible"]) is False
    assert bool(normalized.at["Text Decimal True", "coverage_eligible"]) is True
    assert (
        bool(
            normalized.at[
                "Text Decimal Identity False",
                "coverage_eligible",
            ]
        )
        is False
    )
    assert bool(normalized.at["Object Blank Repair", "coverage_eligible"]) is True


@pytest.mark.parametrize("false_text", ["0.00", "0e0", "-0.0"])
def test_csv_equivalent_numeric_false_text_never_repairs_or_trusts(
    tmp_path,
    false_text,
):
    roster_path = tmp_path / "equivalent_numeric_false.csv"
    pd.DataFrame(
        [
            _row("Explicit False")
            | {
                "coverage_eligible": false_text,
                "official_url_identity_valid": "1e0",
            },
            _row("Identity False")
            | {
                "coverage_eligible": "true",
                "official_url_identity_valid": false_text,
                "official_url_identity_status": "valid",
            },
            _row("Keep Object Dtype")
            | {
                "coverage_eligible": "not-a-boolean",
                "official_url_identity_valid": "true",
            },
        ]
    ).to_csv(roster_path, index=False)
    round_tripped = pd.read_csv(roster_path)
    assert pd.api.types.is_string_dtype(round_tripped["coverage_eligible"])
    assert pd.api.types.is_string_dtype(
        round_tripped["official_url_identity_valid"]
    )

    normalized = ufc_active_roster._apply_profile_status_eligibility(
        round_tripped
    ).set_index("official_name")

    assert bool(normalized.at["Explicit False", "coverage_eligible"]) is False
    assert bool(normalized.at["Identity False", "coverage_eligible"]) is False


def test_eligibility_normalization_is_position_safe_with_duplicate_index():
    rows = pd.DataFrame(
        [
            _row("Inactive Fighter", status="Inactive"),
            _row("Active Fighter")
            | {
                "coverage_eligible": "",
                "combat_sport": "mma",
            },
        ],
        index=[0, 0],
    )

    normalized = ufc_active_roster._apply_profile_status_eligibility(rows)

    assert normalized.index.tolist() == [0, 0]
    assert normalized["coverage_eligible"].tolist() == [False, True]


def test_profile_request_failure_carries_forward_one_cached_verified_current_status(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    pd.DataFrame([_row("Current Fighter")]).to_csv(roster_path, index=False)
    failed_live_row = _row("Current Fighter", status="") | {
        "coverage_eligible": False,
        ufc_active_roster._PROFILE_FETCH_OUTCOME_COLUMN: (
            ufc_active_roster._PROFILE_FETCH_REQUEST_FAILED
        ),
    }
    _set_live_scrape(monkeypatch, [failed_live_row])

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )

    current = synced.iloc[0]
    assert current["profile_status"] == "Active"
    assert bool(current["coverage_eligible"]) is True
    assert current["active_roster_status_reason"] == (
        "cached_verified_current_after_profile_request_failure"
    )
    assert synced.attrs["restored_profile_request_failure_statuses"] == 1


def test_profile_request_failure_restores_only_blank_stable_cached_biography(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    cached_row = _row("Current Fighter") | {
        "birthplace": "Dublin, Ireland",
        "height": "74 in",
        "reach": "76 in",
        "weight": "170 lbs",
        "octagon_debut": "Jul. 11, 2020",
        "age": "31",
        "profile_record": "15-1-0",
        "profile_division": "Welterweight Division",
    }
    pd.DataFrame([cached_row]).to_csv(roster_path, index=False)
    failed_live_row = _row("Current Fighter", status="") | {
        "coverage_eligible": False,
        "birthplace": "",
        "height": "75 in",
        "reach": "",
        "weight": "",
        "octagon_debut": "",
        "age": "",
        "profile_record": "",
        "profile_division": "",
        ufc_active_roster._PROFILE_FETCH_OUTCOME_COLUMN: (
            ufc_active_roster._PROFILE_FETCH_REQUEST_FAILED
        ),
    }
    _set_live_scrape(monkeypatch, [failed_live_row])

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )

    current = synced.iloc[0]
    assert current["birthplace"] == "Dublin, Ireland"
    assert current["height"] == "75 in"
    assert current["reach"] == "76 in"
    assert current["weight"] == "170 lbs"
    assert current["octagon_debut"] == "Jul. 11, 2020"
    assert current["age"] == ""
    assert current["profile_record"] == ""
    assert current["profile_division"] == ""
    assert current[ufc_active_roster._CACHED_PROFILE_BIOGRAPHY_FIELDS_COLUMN] == (
        "birthplace|reach|weight|octagon_debut"
    )
    assert current[ufc_active_roster._CACHED_PROFILE_BIOGRAPHY_REASON_COLUMN] == (
        ufc_active_roster._CACHED_PROFILE_BIOGRAPHY_REASON
    )


def test_http_200_challenge_profile_restores_cached_verified_current_status(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    pd.DataFrame([_row("Current Fighter")]).to_csv(roster_path, index=False)
    listing_page = BeautifulSoup(
        """
        <html><head><title>Athletes - All | UFC</title></head><body>
          <form class="views-exposed-form"></form>
          <div class="c-listing-athlete-flipcard">
            <a href="/athlete/current-fighter">Profile</a>
            <span class="c-listing-athlete__name">Current Fighter</span>
          </div>
        </body></html>
        """,
        "lxml",
    )
    challenge_page = BeautifulSoup(
        """
        <html><head><title>Just a moment...</title></head>
        <body>Checking your browser before accessing UFC.com</body></html>
        """,
        "lxml",
    )
    terminal_page = BeautifulSoup(
        """
        <html><head><title>Athletes - All | UFC</title></head><body>
          <form class="views-exposed-form"></form>
          <div class="view-empty">No Result Found</div>
        </body></html>
        """,
        "lxml",
    )

    def fake_get_soup(url, session=None):
        if "/athlete/current-fighter" in url:
            return challenge_page
        if "page=1" in url:
            return terminal_page
        return listing_page

    monkeypatch.setattr(ufc_active_roster, "_get_soup", fake_get_soup)
    monkeypatch.setattr(
        ufc_active_roster,
        "_classify_combat_sport",
        lambda _row, session=None: {
            "combat_sport": "mma",
            "combat_sport_reason": "",
            "combat_sport_profile_url": "",
        },
    )

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        resolve_ufcstats=False,
        identity_audit_path=None,
    )

    current = synced.iloc[0]
    assert current[ufc_active_roster._PROFILE_FETCH_OUTCOME_COLUMN] == (
        ufc_active_roster._PROFILE_FETCH_INVALID_RESPONSE
    )
    assert current["profile_status"] == "Active"
    assert bool(current["coverage_eligible"]) is True
    assert synced.attrs["restored_profile_request_failure_statuses"] == 1


def test_valid_profile_with_blank_status_does_not_restore_cached_active(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    pd.DataFrame(
        [
            _row("Current Fighter")
            | {
                "birthplace": "Cached Birthplace",
                "height": "73 in",
                "reach": "75 in",
                "weight": "170 lbs",
                "octagon_debut": "Jan. 1, 2020",
            }
        ]
    ).to_csv(roster_path, index=False)
    listing_page = BeautifulSoup(
        """
        <html><head><title>Athletes - All | UFC</title></head><body>
          <form class="views-exposed-form"></form>
          <div class="c-listing-athlete-flipcard">
            <a href="/athlete/current-fighter">Profile</a>
            <span class="c-listing-athlete__name">Current Fighter</span>
          </div>
        </body></html>
        """,
        "lxml",
    )
    valid_profile_without_status = BeautifulSoup(
        """
        <html><body>
          <h1 class="hero-profile__name">Current Fighter</h1>
          <p class="hero-profile__division-title">Welterweight Division</p>
        </body></html>
        """,
        "lxml",
    )
    terminal_page = BeautifulSoup(
        """
        <html><head><title>Athletes - All | UFC</title></head><body>
          <form class="views-exposed-form"></form>
          <div class="view-empty">No Result Found</div>
        </body></html>
        """,
        "lxml",
    )

    def fake_get_soup(url, session=None):
        if "/athlete/current-fighter" in url:
            return valid_profile_without_status
        if "page=1" in url:
            return terminal_page
        return listing_page

    monkeypatch.setattr(ufc_active_roster, "_get_soup", fake_get_soup)
    monkeypatch.setattr(
        ufc_active_roster,
        "_classify_combat_sport",
        lambda _row, session=None: {
            "combat_sport": "mma",
            "combat_sport_reason": "",
            "combat_sport_profile_url": "",
        },
    )

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        resolve_ufcstats=False,
        identity_audit_path=None,
    )

    current = synced.iloc[0]
    assert current[ufc_active_roster._PROFILE_FETCH_OUTCOME_COLUMN] == (
        ufc_active_roster._PROFILE_FETCH_SUCCEEDED
    )
    assert current["profile_status"] == "status_unknown"
    assert bool(current["coverage_eligible"]) is False
    for field in ufc_active_roster._CACHED_STABLE_PROFILE_BIOGRAPHY_FIELDS:
        assert str(current.get(field) or "") == ""
    assert (
        ufc_active_roster._CACHED_PROFILE_BIOGRAPHY_REASON_COLUMN
        not in synced.columns
    )
    assert synced.attrs["restored_profile_request_failure_statuses"] == 0


@pytest.mark.parametrize("status_code", [404, 410])
def test_terminal_profile_http_error_never_restores_cached_status_or_biography(
    tmp_path,
    monkeypatch,
    status_code,
):
    roster_path = tmp_path / f"ufc_active_roster_http_{status_code}.csv"
    pd.DataFrame(
        [
            _row("Current Fighter")
            | {
                "birthplace": "Cached Birthplace",
                "height": "73 in",
                "reach": "75 in",
                "weight": "170 lbs",
                "octagon_debut": "Jan. 1, 2020",
            }
        ]
    ).to_csv(roster_path, index=False)
    listing_page = BeautifulSoup(
        """
        <html><head><title>Athletes - All | UFC</title></head><body>
          <form class="views-exposed-form"></form>
          <div class="c-listing-athlete-flipcard">
            <a href="/athlete/current-fighter">Profile</a>
            <span class="c-listing-athlete__name">Current Fighter</span>
          </div>
        </body></html>
        """,
        "lxml",
    )
    terminal_page = BeautifulSoup(
        """
        <html><head><title>Athletes - All | UFC</title></head><body>
          <form class="views-exposed-form"></form>
          <div class="view-empty">No Result Found</div>
        </body></html>
        """,
        "lxml",
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "_get_soup",
        lambda url, session=None: (
            terminal_page if "page=1" in url else listing_page
        ),
    )

    def fail_profile(*_args, **_kwargs):
        response = requests.Response()
        response.status_code = status_code
        raise requests.HTTPError(
            f"profile returned HTTP {status_code}",
            response=response,
        )

    monkeypatch.setattr(
        ufc_active_roster,
        "scrape_official_athlete_profile",
        fail_profile,
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "_classify_combat_sport",
        lambda _row, session=None: {
            "combat_sport": "mma",
            "combat_sport_reason": "",
            "combat_sport_profile_url": "",
        },
    )

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        resolve_ufcstats=False,
        identity_audit_path=None,
    )

    current = synced.iloc[0]
    assert current[ufc_active_roster._PROFILE_FETCH_OUTCOME_COLUMN] == (
        ufc_active_roster._PROFILE_FETCH_HTTP_ERROR
    )
    assert current["profile_status"] == "status_unknown"
    assert bool(current["coverage_eligible"]) is False
    for field in ufc_active_roster._CACHED_STABLE_PROFILE_BIOGRAPHY_FIELDS:
        assert ufc_active_roster._optional_text(current.get(field)) == ""
    assert (
        ufc_active_roster._CACHED_PROFILE_BIOGRAPHY_REASON_COLUMN
        not in synced.columns
    )
    assert synced.attrs["restored_profile_request_failure_statuses"] == 0


@pytest.mark.parametrize(
    "outcome",
    [
        ufc_active_roster._PROFILE_FETCH_SUCCEEDED,
        ufc_active_roster._PROFILE_FETCH_HTTP_ERROR,
        ufc_active_roster._PROFILE_FETCH_REQUEST_ERROR,
        ufc_active_roster._PROFILE_FETCH_PARSE_FAILED,
        ufc_active_roster._PROFILE_FETCH_NOT_REQUESTED,
    ],
)
def test_non_request_profile_blank_does_not_inherit_cached_active_status(
    tmp_path,
    monkeypatch,
    outcome,
):
    roster_path = tmp_path / f"ufc_active_roster_{outcome}.csv"
    pd.DataFrame([_row("Current Fighter")]).to_csv(roster_path, index=False)
    blank_live_row = _row("Current Fighter", status="") | {
        "coverage_eligible": False,
        ufc_active_roster._PROFILE_FETCH_OUTCOME_COLUMN: outcome,
    }
    _set_live_scrape(monkeypatch, [blank_live_row])

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )

    assert synced.loc[0, "profile_status"] == "status_unknown"
    assert bool(synced.loc[0, "coverage_eligible"]) is False
    assert synced.attrs["restored_profile_request_failure_statuses"] == 0


def test_profile_request_failure_does_not_inherit_ambiguous_cached_status(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    first = _row("Current Fighter") | {"birthplace": "Cached Birthplace"}
    second = _row("Current Fighter") | {
        "ufcstats_url": "http://ufcstats.test/current-fighter-alternate",
        "birthplace": "Other Cached Birthplace",
    }
    pd.DataFrame([first, second]).to_csv(roster_path, index=False)
    failed_live_row = _row("Current Fighter", status="") | {
        "coverage_eligible": False,
        ufc_active_roster._PROFILE_FETCH_OUTCOME_COLUMN: (
            ufc_active_roster._PROFILE_FETCH_REQUEST_FAILED
        ),
    }
    _set_live_scrape(monkeypatch, [failed_live_row])

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )

    assert synced.loc[0, "profile_status"] == "status_unknown"
    assert bool(synced.loc[0, "coverage_eligible"]) is False
    assert ufc_active_roster._optional_text(synced.loc[0].get("birthplace")) == ""
    assert (
        ufc_active_roster._CACHED_PROFILE_BIOGRAPHY_REASON_COLUMN
        not in synced.columns
    )
    assert synced.attrs["restored_profile_request_failure_statuses"] == 0


@pytest.mark.parametrize(
    ("case", "cached_overrides"),
    [
        (
            "untrusted",
            {
                "official_url_identity_valid": False,
                "official_url_identity_status": "mismatch",
            },
        ),
        ("inactive", {"profile_status": "Inactive"}),
        (
            "power_slap",
            {
                "combat_sport": "power_slap",
                "combat_sport_reason": "powerslap_profile_match",
            },
        ),
    ],
)
def test_profile_request_failure_restores_nothing_from_blocked_cached_identity(
    tmp_path,
    monkeypatch,
    case,
    cached_overrides,
):
    roster_path = tmp_path / f"ufc_active_roster_{case}.csv"
    cached_row = _row("Current Fighter") | {
        "birthplace": "Cached Birthplace",
        "height": "73 in",
        **cached_overrides,
    }
    pd.DataFrame([cached_row]).to_csv(roster_path, index=False)
    failed_live_row = _row("Current Fighter", status="") | {
        "coverage_eligible": False,
        "birthplace": "",
        "height": "",
        ufc_active_roster._PROFILE_FETCH_OUTCOME_COLUMN: (
            ufc_active_roster._PROFILE_FETCH_REQUEST_FAILED
        ),
    }
    _set_live_scrape(monkeypatch, [failed_live_row])

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )

    current = synced.iloc[0]
    assert current["profile_status"] == "status_unknown"
    assert bool(current["coverage_eligible"]) is False
    assert current["birthplace"] == ""
    assert current["height"] == ""
    assert (
        ufc_active_roster._CACHED_PROFILE_BIOGRAPHY_REASON_COLUMN
        not in synced.columns
    )
    assert synced.attrs["restored_profile_request_failure_statuses"] == 0


def test_retained_row_preserves_first_missing_time_and_expires_on_third_miss(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    pd.DataFrame([_row("Anchor"), _row("Missing Fighter")]).to_csv(roster_path, index=False)
    _set_live_scrape(monkeypatch, [_row("Anchor")])

    first = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )
    first_retained = first.loc[first["official_name"].eq("Missing Fighter")].iloc[0]
    first_missing_at = first_retained["active_roster_first_missing_at_utc"]
    assert int(first_retained["active_roster_consecutive_missing_syncs"]) == 1

    second = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )
    second_retained = second.loc[second["official_name"].eq("Missing Fighter")].iloc[0]
    assert second_retained["active_roster_first_missing_at_utc"] == first_missing_at
    assert second_retained["active_roster_retained_at_utc"] == first_missing_at
    assert int(second_retained["active_roster_consecutive_missing_syncs"]) == 2

    third = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )
    assert third["official_name"].tolist() == ["Anchor"]
    assert third.attrs["retained_missing_live_rows"] == []
    assert third.attrs["expired_missing_live_rows"][0]["official_name"] == "Missing Fighter"
    assert third.attrs["expired_missing_live_rows"][0]["first_missing_at_utc"] == first_missing_at
    assert third.attrs["expired_missing_live_rows"][0]["consecutive_missing_syncs"] == 3


def test_cached_fallback_reconstructs_retained_diagnostics(
    tmp_path,
    monkeypatch,
    caplog,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    cached = _row("Missing Fighter") | {
        "active_roster_live_present": False,
        "active_roster_retained_from_previous": True,
        "active_roster_retained_at_utc": "2026-07-20T00:00:00+00:00",
        "active_roster_first_missing_at_utc": "2026-07-20T00:00:00+00:00",
        "active_roster_consecutive_missing_syncs": 2,
        "active_roster_missing_from_live_reason": "absent_from_latest_ufc_active_roster_sync",
    }
    pd.DataFrame([cached]).to_csv(roster_path, index=False)

    def fail_scrape(**_kwargs):
        raise requests.ReadTimeout("timed out")

    monkeypatch.setattr(ufc_active_roster, "scrape_official_active_roster", fail_scrape)
    with caplog.at_level("WARNING"):
        synced = ufc_active_roster.sync_official_active_roster(
            output_path=roster_path,
            identity_audit_path=None,
        )

    assert synced.attrs["sync_fallback_used"] is True
    assert synced.attrs["retained_missing_live_rows"] == [
        {
            "official_name": "Missing Fighter",
            "official_athlete_url": "https://www.ufc.com/athlete/missing-fighter",
            "ufcstats_url": "http://ufcstats.test/missing-fighter",
            "first_missing_at_utc": "2026-07-20T00:00:00+00:00",
            "consecutive_missing_syncs": 2,
        }
    ]
    alert_record = next(
        record
        for record in caplog.records
        if record.getMessage().startswith("Official UFC roster sync failed;")
    )
    assert alert_record.alert_incident_key == "ufc-active-roster:live-sync-failed"


def test_startup_sanitizer_uses_status_and_audit_not_row_count(tmp_path):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    audit_path = tmp_path / "ufc_active_roster_identity_audit.csv"
    generation = "accepted-generation"
    roster_df = pd.DataFrame(
        [
            _row("Current Fighter"),
            _row("Verified Retired", status="Retired"),
            _row("Audit Retired"),
            _row("Unknown Fighter", status=""),
        ]
    )
    roster_df[ufc_active_roster.ACTIVE_ROSTER_SYNC_GENERATION_COLUMN] = generation
    roster_df.to_csv(roster_path, index=False)
    pd.DataFrame(
        [
            {
                "official_name": "Audit Retired",
                "official_athlete_url": "https://www.ufc.com/athlete/audit-retired",
                "profile_name": "Audit Retired",
                "slug_name": "audit retired",
                "identity_status": "valid",
                "identity_reason": "",
                "action": "excluded_inactive_profile_status",
                ufc_active_roster.ACTIVE_ROSTER_SYNC_GENERATION_COLUMN: generation,
            }
        ]
    ).to_csv(audit_path, index=False)

    summary = sanitize_active_roster_startup(
        roster_path=roster_path,
        identity_audit_path=audit_path,
    )
    sanitized = pd.read_csv(roster_path)

    assert summary["rows_before"] == 4
    assert summary["rows_after"] == 2
    assert summary["removed_by_identity_audit"] == 1
    assert summary["removed_verified_inactive_status"] == 1
    assert summary["status_unknown_rows"] == 1
    assert set(sanitized["official_name"]) == {"Current Fighter", "Unknown Fighter"}
    unknown = sanitized.loc[sanitized["official_name"].eq("Unknown Fighter")].iloc[0]
    assert unknown["profile_status"] == "status_unknown"
    assert str(unknown["coverage_eligible"]).casefold() == "false"


@pytest.mark.parametrize(
    "persisted_kind",
    ["missing", "zero_byte", "header_only", "unreadable"],
)
def test_startup_sanitizer_recovers_invalid_persisted_roster_from_valid_fallback(
    tmp_path,
    persisted_kind,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    fallback_path = tmp_path / "image_ufc_active_roster_official.csv"
    if persisted_kind == "missing":
        pass
    elif persisted_kind == "zero_byte":
        roster_path.write_bytes(b"")
    elif persisted_kind == "header_only":
        roster_path.write_text(
            "official_name,official_athlete_url,profile_status\n",
            encoding="utf-8",
        )
    else:
        roster_path.write_bytes(b"\x80\x81\x82")
    pd.DataFrame([_row("Image Current Fighter")]).to_csv(
        fallback_path,
        index=False,
    )

    summary = sanitize_active_roster_startup(
        roster_path=roster_path,
        fallback_roster_path=fallback_path,
    )
    recovered = pd.read_csv(roster_path)

    assert summary["action"] == "recovered_from_fallback"
    assert summary["fallback_recovery_used"] is True
    assert summary["fallback_roster_path"] == str(fallback_path)
    assert summary["persisted_roster_load_error"]
    assert recovered["official_name"].tolist() == ["Image Current Fighter"]
    assert str(recovered.loc[0, "coverage_eligible"]).casefold() == "true"


def test_startup_sanitizer_fails_when_persisted_and_fallback_rosters_are_missing(
    tmp_path,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    fallback_path = tmp_path / "image_ufc_active_roster_official.csv"

    with pytest.raises(RuntimeError, match="image fallback cannot be used"):
        sanitize_active_roster_startup(
            roster_path=roster_path,
            fallback_roster_path=fallback_path,
        )

    assert not roster_path.exists()


def test_startup_sanitizer_never_replaces_valid_persisted_roster_with_fallback(
    tmp_path,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    fallback_path = tmp_path / "image_ufc_active_roster_official.csv"
    pd.DataFrame([_row("Volume Current Fighter")]).to_csv(
        roster_path,
        index=False,
    )
    pd.DataFrame([_row("Image Current Fighter")]).to_csv(
        fallback_path,
        index=False,
    )
    original_bytes = roster_path.read_bytes()

    summary = sanitize_active_roster_startup(
        roster_path=roster_path,
        fallback_roster_path=fallback_path,
    )

    assert summary["action"] == "unchanged"
    assert summary["fallback_recovery_used"] is False
    assert roster_path.read_bytes() == original_bytes
    assert pd.read_csv(roster_path)["official_name"].tolist() == [
        "Volume Current Fighter"
    ]


@pytest.mark.parametrize(
    "fallback_kind",
    ["missing", "zero_byte", "header_only", "missing_identity_schema", "unreadable"],
)
def test_startup_sanitizer_refuses_invalid_fallback_and_preserves_bad_volume(
    tmp_path,
    fallback_kind,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    fallback_path = tmp_path / "image_ufc_active_roster_official.csv"
    roster_path.write_text(
        "official_name,official_athlete_url,profile_status\n",
        encoding="utf-8",
    )
    if fallback_kind == "zero_byte":
        fallback_path.write_bytes(b"")
    elif fallback_kind == "header_only":
        fallback_path.write_text(
            "official_name,official_athlete_url,profile_status\n",
            encoding="utf-8",
        )
    elif fallback_kind == "missing_identity_schema":
        fallback_path.write_text(
            "official_name,profile_status\nImage Fighter,Active\n",
            encoding="utf-8",
        )
    elif fallback_kind == "unreadable":
        fallback_path.write_bytes(b"\x80\x81\x82")
    original_bytes = roster_path.read_bytes()

    with pytest.raises(RuntimeError, match="image fallback cannot be used"):
        sanitize_active_roster_startup(
            roster_path=roster_path,
            fallback_roster_path=fallback_path,
        )

    assert roster_path.read_bytes() == original_bytes


def test_rejected_roster_generation_does_not_publish_its_identity_audit(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    audit_path = tmp_path / "ufc_active_roster_identity_audit.csv"
    pd.DataFrame([_row(f"Cached Fighter {index}") for index in range(100)]).to_csv(
        roster_path,
        index=False,
    )
    prior_audit = "official_name,action\nPrior Fighter,accepted_prior_audit\n"
    audit_path.write_text(prior_audit, encoding="utf-8")
    rejected_audit = {
        "official_name": "Rejected Retired Fighter",
        "official_athlete_url": "https://www.ufc.com/athlete/rejected-retired-fighter",
        "profile_name": "Rejected Retired Fighter",
        "slug_name": "rejected retired fighter",
        "identity_status": "valid",
        "identity_reason": "",
        "action": "excluded_inactive_profile_status",
    }
    _set_live_scrape(
        monkeypatch,
        [_row(f"Live Fighter {index}") for index in range(400)],
        audit_rows=[rejected_audit],
    )

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=audit_path,
    )

    assert synced.attrs["sync_fallback_used"] is True
    assert "growth guard" in synced.attrs["sync_error"]
    assert audit_path.read_text(encoding="utf-8") == prior_audit


def test_exact_five_percent_and_fifty_row_shrink_is_treated_as_incomplete(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    cached_rows = [_row(f"Fighter {index}") for index in range(1000)]
    pd.DataFrame(cached_rows).to_csv(roster_path, index=False)
    _set_live_scrape(monkeypatch, cached_rows[:950])
    monkeypatch.setattr(
        ufc_active_roster,
        "_merge_cached_roster_rows_missing_from_live",
        lambda live_df, *_args, **_kwargs: (live_df, [], []),
    )

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )

    assert synced.attrs["sync_complete"] is False
    assert synced.attrs["sync_completeness_reason"] == (
        "suspicious_live_shrink:950_of_1000"
    )


def test_shrink_guard_excludes_already_retained_rows_from_cached_baseline(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    live_rows = [
        _row(f"Live Fighter {index}")
        | {
            "active_roster_live_present": True,
            "active_roster_retained_from_previous": False,
            "active_roster_missing_from_live_reason": "",
        }
        for index in range(19)
    ]
    retained = _row("Previously Missing") | {
        "coverage_eligible": False,
        "active_roster_live_present": False,
        "active_roster_retained_from_previous": True,
        "active_roster_first_missing_at_utc": "2026-07-20T00:00:00+00:00",
        "active_roster_retained_at_utc": "2026-07-20T00:00:00+00:00",
        "active_roster_consecutive_missing_syncs": 1,
        "active_roster_missing_from_live_reason": (
            "absent_from_latest_ufc_active_roster_sync"
        ),
    }
    pd.DataFrame([*live_rows, retained]).to_csv(roster_path, index=False)
    _set_live_scrape(monkeypatch, live_rows)
    monkeypatch.setattr(
        ufc_active_roster,
        "_ROSTER_SHRINK_INCOMPLETE_MIN_CACHED_ROWS",
        1,
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "_ROSTER_SHRINK_INCOMPLETE_ABSOLUTE_ROWS",
        1,
    )

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )

    assert synced.attrs["sync_complete"] is True
    missing = synced.loc[synced["official_name"].eq("Previously Missing")].iloc[0]
    assert int(missing["active_roster_consecutive_missing_syncs"]) == 2


def test_persistent_truncated_scan_cannot_ratchet_down_shrink_baseline(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    cached_rows = [_row(f"Fighter {index}") for index in range(100)]
    pd.DataFrame(cached_rows).to_csv(roster_path, index=False)
    _set_live_scrape(monkeypatch, cached_rows[:50])

    for _ in range(4):
        synced = ufc_active_roster.sync_official_active_roster(
            output_path=roster_path,
            identity_audit_path=None,
        )

        assert synced.attrs["sync_complete"] is False
        assert synced.attrs["sync_completeness_reason"] == (
            "suspicious_live_shrink:50_of_100"
        )
        assert len(synced) == 100
        retained = synced.loc[
            synced["official_name"].isin(
                {f"Fighter {index}" for index in range(50, 100)}
            )
        ]
        assert len(retained) == 50
        assert retained["coverage_eligible"].map(bool).all()
        assert set(
            retained["active_roster_consecutive_missing_syncs"].astype(int)
        ) == {0}
        assert set(retained["active_roster_missing_from_live_reason"]) == {
            "absent_from_incomplete_ufc_active_roster_sync"
        }


def test_equal_size_new_identity_churn_cannot_hide_cached_identity_shrink(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    cached_rows = [_row(f"Fighter {index}") for index in range(100)]
    new_rows = [_row(f"New Fighter {index}") for index in range(50)]
    pd.DataFrame(cached_rows).to_csv(roster_path, index=False)
    _set_live_scrape(monkeypatch, [*cached_rows[:50], *new_rows])
    omitted_names = {f"Fighter {index}" for index in range(50, 100)}

    for _ in range(4):
        synced = ufc_active_roster.sync_official_active_roster(
            output_path=roster_path,
            identity_audit_path=None,
        )

        assert synced.attrs["sync_complete"] is False
        assert synced.attrs["sync_completeness_reason"].startswith(
            "suspicious_live_shrink:"
        )
        assert len(synced) == 150
        retained = synced.loc[synced["official_name"].isin(omitted_names)]
        assert len(retained) == 50
        assert retained["coverage_eligible"].map(bool).all()
        assert set(
            retained["active_roster_consecutive_missing_syncs"].astype(int)
        ) == {0}


def test_current_verified_inactive_audit_can_lower_shrink_baseline(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    cached_rows = [_row(f"Fighter {index}") for index in range(100)]
    pd.DataFrame(cached_rows).to_csv(roster_path, index=False)
    audit_rows = [
        row | {"action": "excluded_inactive_profile_status"}
        for row in cached_rows[50:]
    ]
    _set_live_scrape(monkeypatch, cached_rows[:50], audit_rows=audit_rows)

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )

    assert synced.attrs["sync_complete"] is True
    assert set(synced["official_name"]) == {
        f"Fighter {index}" for index in range(50)
    }
    assert synced.attrs["retained_missing_live_rows"] == []


def test_conflicting_inactive_audit_urls_cannot_lower_shrink_baseline(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    cached_rows = [_row(f"Fighter {index}") for index in range(100)]
    pd.DataFrame(cached_rows).to_csv(roster_path, index=False)
    audit_rows = [
        row
        | {
            "official_athlete_url": f"{row['official_athlete_url']}-different",
            "ufcstats_url": f"{row['ufcstats_url']}-different",
            "action": "excluded_inactive_profile_status",
        }
        for row in cached_rows[50:]
    ]
    _set_live_scrape(monkeypatch, cached_rows[:50], audit_rows=audit_rows)

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )

    assert synced.attrs["sync_complete"] is False
    assert synced.attrs["sync_completeness_reason"] == (
        "suspicious_live_shrink:50_of_100"
    )
    assert len(synced) == 100


def test_shrink_baseline_excludes_legacy_complete_miss_but_not_incomplete_miss():
    normal = _row("Normal Fighter")
    legacy_complete = _row("Legacy Complete Miss") | {
        "active_roster_retained_from_previous": True,
        "active_roster_live_present": False,
        "active_roster_missing_from_live_reason": (
            "absent_from_latest_ufc_active_roster_sync"
        ),
    }
    incomplete = _row("Incomplete Miss") | {
        "active_roster_retained_from_previous": True,
        "active_roster_live_present": False,
        "active_roster_consecutive_missing_syncs": 0,
        "active_roster_missing_from_live_reason": (
            "absent_from_incomplete_ufc_active_roster_sync"
        ),
    }

    baseline = ufc_active_roster._cached_roster_shrink_baseline(
        pd.DataFrame([normal, legacy_complete, incomplete])
    )

    assert set(baseline["official_name"]) == {
        "Normal Fighter",
        "Incomplete Miss",
    }


@pytest.mark.parametrize(
    "cached_text",
    [
        "",
        "official_name,official_athlete_url\n",
        "garbage\nnot-a-roster\n",
    ],
)
def test_existing_invalid_cached_roster_is_not_overwritten(
    tmp_path,
    monkeypatch,
    cached_text,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    roster_path.write_text(cached_text, encoding="utf-8")
    original_bytes = roster_path.read_bytes()
    _set_live_scrape(monkeypatch, [_row("Live Fighter")])

    with pytest.raises(RuntimeError, match="could not be safely loaded"):
        ufc_active_roster.sync_official_active_roster(
            output_path=roster_path,
            identity_audit_path=None,
        )

    assert roster_path.read_bytes() == original_bytes


def test_incomplete_listing_does_not_advance_or_expire_missing_counter(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    missing = _row("Missing Fighter") | {
        "active_roster_live_present": False,
        "active_roster_retained_from_previous": True,
        "active_roster_retained_at_utc": "2026-07-20T00:00:00+00:00",
        "active_roster_first_missing_at_utc": "2026-07-20T00:00:00+00:00",
        "active_roster_consecutive_missing_syncs": 2,
        "active_roster_missing_from_live_reason": "absent_from_latest_ufc_active_roster_sync",
    }
    pd.DataFrame([_row("Anchor"), missing]).to_csv(roster_path, index=False)
    live_df = pd.DataFrame([_row("Anchor")])
    live_df.attrs["identity_audit_rows"] = []
    live_df.attrs["sync_complete"] = False
    live_df.attrs["sync_completeness_reason"] = "unexpected_empty_page"
    live_df.attrs["pages_scraped"] = 1
    monkeypatch.setattr(
        ufc_active_roster,
        "scrape_official_active_roster",
        lambda **_kwargs: live_df.copy(),
    )

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=None,
    )

    retained = synced.loc[synced["official_name"].eq("Missing Fighter")].iloc[0]
    assert int(retained["active_roster_consecutive_missing_syncs"]) == 2
    assert retained["active_roster_first_missing_at_utc"] == "2026-07-20T00:00:00+00:00"
    assert retained["active_roster_missing_from_live_reason"] == (
        "absent_from_incomplete_ufc_active_roster_sync"
    )
    assert bool(retained["coverage_eligible"]) is True
    assert retained["profile_status"] == "Active"
    assert synced.attrs["expired_missing_live_rows"] == []


def test_unverified_inactive_status_is_quarantined_not_deleted(tmp_path):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    unverified = _row("Legacy Retired", status="Retired")
    unverified.pop("official_url_identity_valid")
    unverified.pop("official_url_identity_status")
    pd.DataFrame([_row("Current Fighter"), unverified]).to_csv(roster_path, index=False)

    summary = sanitize_active_roster_startup(roster_path=roster_path)
    sanitized = pd.read_csv(roster_path)

    assert set(sanitized["official_name"]) == {"Current Fighter", "Legacy Retired"}
    retired = sanitized.loc[sanitized["official_name"].eq("Legacy Retired")].iloc[0]
    assert str(retired["coverage_eligible"]).casefold() == "false"
    assert retired["active_roster_status_reason"] == "unverified_inactive_profile_status"
    assert summary["removed_verified_inactive_status"] == 0
    assert summary["quarantined_unverified_inactive_status"] == 1


def test_accepted_audit_does_not_delete_same_name_with_conflicting_url(tmp_path):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    audit_path = tmp_path / "ufc_active_roster_identity_audit.csv"
    generation = "accepted-generation"
    roster = pd.DataFrame(
        [
            _row("Current Fighter"),
            _row("Alias Fighter")
            | {"official_athlete_url": "https://www.ufc.com/athlete/alias-fighter-old"},
        ]
    )
    roster[ufc_active_roster.ACTIVE_ROSTER_SYNC_GENERATION_COLUMN] = generation
    roster.to_csv(roster_path, index=False)
    pd.DataFrame(
        [
            {
                "official_name": "Alias Fighter",
                "official_athlete_url": "https://www.ufc.com/athlete/alias-fighter-new",
                "profile_name": "Alias Fighter",
                "slug_name": "alias fighter",
                "identity_status": "valid",
                "identity_reason": "",
                "action": "excluded_inactive_profile_status",
                ufc_active_roster.ACTIVE_ROSTER_SYNC_GENERATION_COLUMN: generation,
            }
        ]
    ).to_csv(audit_path, index=False)

    summary = sanitize_active_roster_startup(
        roster_path=roster_path,
        identity_audit_path=audit_path,
    )
    sanitized = pd.read_csv(roster_path)

    assert set(sanitized["official_name"]) == {"Current Fighter", "Alias Fighter"}
    assert summary["removed_by_identity_audit"] == 0


def test_accepted_audit_name_alias_cannot_override_conflicting_url(tmp_path):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    audit_path = tmp_path / "ufc_active_roster_identity_audit.csv"
    generation = "accepted-generation"
    roster = pd.DataFrame(
        [
            _row("Current Fighter"),
            _row("Former Listing Name")
            | {
                "official_athlete_url": "https://www.ufc.com/athlete/former-listing-name",
                "profile_name": "Former Listing Name",
                "slug_name": "former listing name",
            },
        ]
    )
    roster[ufc_active_roster.ACTIVE_ROSTER_SYNC_GENERATION_COLUMN] = generation
    roster.to_csv(roster_path, index=False)
    pd.DataFrame(
        [
            {
                "official_name": "Current Listing Name",
                "official_athlete_url": "https://www.ufc.com/athlete/current-listing-name",
                "profile_name": "Current Listing Name",
                "slug_name": "current listing name",
                "alternate_slug_names": "Former Listing Name",
                "identity_status": "valid",
                "identity_reason": "",
                "action": "excluded_inactive_profile_status",
                ufc_active_roster.ACTIVE_ROSTER_SYNC_GENERATION_COLUMN: generation,
            }
        ]
    ).to_csv(audit_path, index=False)

    summary = sanitize_active_roster_startup(
        roster_path=roster_path,
        identity_audit_path=audit_path,
    )

    assert set(pd.read_csv(roster_path)["official_name"]) == {
        "Current Fighter",
        "Former Listing Name",
    }
    assert summary["removed_by_identity_audit"] == 0


def test_startup_ignores_unreadable_generation_matched_audit(tmp_path, caplog):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    audit_path = tmp_path / "ufc_active_roster_identity_audit.csv"
    generation = "accepted-generation"
    roster = pd.DataFrame([_row("Current Fighter"), _row("Still Current")])
    roster[ufc_active_roster.ACTIVE_ROSTER_SYNC_GENERATION_COLUMN] = generation
    roster.to_csv(roster_path, index=False)
    audit_path.write_text("", encoding="utf-8")

    summary = sanitize_active_roster_startup(
        roster_path=roster_path,
        identity_audit_path=audit_path,
    )

    assert set(pd.read_csv(roster_path)["official_name"]) == {
        "Current Fighter",
        "Still Current",
    }
    assert summary["removed_by_identity_audit"] == 0
    assert "Ignoring unreadable UFC active-roster identity audit" in caplog.text


def test_startup_ignores_audit_from_different_roster_generation(tmp_path):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    audit_path = tmp_path / "ufc_active_roster_identity_audit.csv"
    roster = pd.DataFrame([_row("Current Fighter"), _row("Still Current")])
    roster[ufc_active_roster.ACTIVE_ROSTER_SYNC_GENERATION_COLUMN] = "new-generation"
    roster.to_csv(roster_path, index=False)
    pd.DataFrame(
        [
            {
                "official_name": "Still Current",
                "official_athlete_url": "https://www.ufc.com/athlete/still-current",
                "action": "excluded_inactive_profile_status",
                ufc_active_roster.ACTIVE_ROSTER_SYNC_GENERATION_COLUMN: "rejected-generation",
            }
        ]
    ).to_csv(audit_path, index=False)

    summary = sanitize_active_roster_startup(
        roster_path=roster_path,
        identity_audit_path=audit_path,
    )

    assert set(pd.read_csv(roster_path)["official_name"]) == {
        "Current Fighter",
        "Still Current",
    }
    assert summary["removed_by_identity_audit"] == 0


def test_startup_ignores_audit_when_roster_generation_is_partially_stamped(tmp_path):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    audit_path = tmp_path / "ufc_active_roster_identity_audit.csv"
    roster = pd.DataFrame([_row("Current Fighter"), _row("Still Current")])
    roster[ufc_active_roster.ACTIVE_ROSTER_SYNC_GENERATION_COLUMN] = [
        "accepted-generation",
        "",
    ]
    roster.to_csv(roster_path, index=False)
    pd.DataFrame(
        [
            {
                "official_name": "Still Current",
                "official_athlete_url": "https://www.ufc.com/athlete/still-current",
                "action": "excluded_inactive_profile_status",
                ufc_active_roster.ACTIVE_ROSTER_SYNC_GENERATION_COLUMN: "accepted-generation",
            }
        ]
    ).to_csv(audit_path, index=False)

    summary = sanitize_active_roster_startup(
        roster_path=roster_path,
        identity_audit_path=audit_path,
    )

    assert set(pd.read_csv(roster_path)["official_name"]) == {
        "Current Fighter",
        "Still Current",
    }
    assert summary["accepted_sync_generation_id"] == ""
    assert summary["removed_by_identity_audit"] == 0


def test_scraper_marks_expected_next_page_with_no_cards_incomplete(monkeypatch):
    first_page = BeautifulSoup(
        """
        <html><body>
          <div class="c-listing-athlete-flipcard">
            <a href="/athlete/current-fighter">Profile</a>
            <span class="c-listing-athlete__name">Current Fighter</span>
          </div>
          <a rel="next" href="?page=1">Load More</a>
        </body></html>
        """,
        "lxml",
    )
    empty_second_page = BeautifulSoup("<html><body>challenge</body></html>", "lxml")
    monkeypatch.setattr(
        ufc_active_roster,
        "_get_soup",
        lambda url, session=None: empty_second_page if "page=1" in url else first_page,
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "_classify_combat_sport",
        lambda row, session=None: {
            "combat_sport": "mma",
            "combat_sport_reason": "test",
            "combat_sport_profile_url": "",
        },
    )

    scraped = ufc_active_roster.scrape_official_active_roster(
        fetch_profile_details=False,
        resolve_ufcstats=False,
    )

    assert scraped.attrs["sync_complete"] is False
    assert scraped.attrs["sync_completeness_reason"] == "unexpected_empty_page"
    assert scraped.attrs["pages_scraped"] == 1


def test_fast_listing_scrape_skips_per_row_combat_sport_requests(monkeypatch):
    first_page = BeautifulSoup(
        """
        <html><head><title>Athletes - All | UFC</title></head><body>
          <form class="views-exposed-form"></form>
          <div class="c-listing-athlete-flipcard">
            <a href="/athlete/current-fighter">Profile</a>
            <span class="c-listing-athlete__name">Current Fighter</span>
          </div>
        </body></html>
        """,
        "lxml",
    )
    terminal_page = BeautifulSoup(
        """
        <html><head><title>Athletes - All | UFC</title></head><body>
          <form class="views-exposed-form"></form>
          <div class="view-empty">Search No Result Found For Please try another term.</div>
        </body></html>
        """,
        "lxml",
    )
    pages = iter((first_page, terminal_page))
    monkeypatch.setattr(
        ufc_active_roster,
        "_get_soup",
        lambda _url, session=None: next(pages),
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "_classify_combat_sport",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fast listing scan must not call Power Slap classification")
        ),
    )

    scraped = ufc_active_roster.scrape_official_active_roster(
        fetch_profile_details=False,
        resolve_ufcstats=False,
        classify_combat_sport=False,
    )

    assert len(scraped) == 1
    assert scraped.iloc[0]["profile_status"] == "status_unknown"
    assert not bool(scraped.iloc[0]["coverage_eligible"])
    assert scraped.attrs["sync_complete"] is True


def test_scraper_requires_explicit_empty_terminal_when_next_link_is_missing(
    monkeypatch,
):
    first_page = BeautifulSoup(
        """
        <html><head><title>Athletes - All | UFC</title></head><body>
          <form class="views-exposed-form"></form>
          <div class="c-listing-athlete-flipcard">
            <a href="/athlete/current-fighter">Profile</a>
            <span class="c-listing-athlete__name">Current Fighter</span>
          </div>
          <div class="new-pager-markup">Continue</div>
        </body></html>
        """,
        "lxml",
    )
    challenge_page = BeautifulSoup("<html><body>challenge</body></html>", "lxml")
    monkeypatch.setattr(
        ufc_active_roster,
        "_get_soup",
        lambda url, session=None: challenge_page if "page=1" in url else first_page,
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "_classify_combat_sport",
        lambda row, session=None: {
            "combat_sport": "mma",
            "combat_sport_reason": "test",
            "combat_sport_profile_url": "",
        },
    )

    scraped = ufc_active_roster.scrape_official_active_roster(
        fetch_profile_details=False,
        resolve_ufcstats=False,
    )

    assert scraped.attrs["sync_complete"] is False
    assert scraped.attrs["sync_completeness_reason"] == "unexpected_empty_page"
    assert scraped.attrs["pages_scraped"] == 1


def test_scraper_accepts_real_explicit_empty_terminal_page(monkeypatch):
    first_page = BeautifulSoup(
        """
        <html><head><title>Athletes - All | UFC</title></head><body>
          <form class="views-exposed-form"></form>
          <div class="c-listing-athlete-flipcard">
            <a href="/athlete/current-fighter">Profile</a>
            <span class="c-listing-athlete__name">Current Fighter</span>
          </div>
        </body></html>
        """,
        "lxml",
    )
    terminal_page = BeautifulSoup(
        """
        <html><head><title>Athletes - All | UFC</title></head><body>
          <form class="views-exposed-form"></form>
          <div class="view-empty">Search No Result Found For Please try another term.</div>
        </body></html>
        """,
        "lxml",
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "_get_soup",
        lambda url, session=None: terminal_page if "page=1" in url else first_page,
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "_classify_combat_sport",
        lambda row, session=None: {
            "combat_sport": "mma",
            "combat_sport_reason": "test",
            "combat_sport_profile_url": "",
        },
    )

    scraped = ufc_active_roster.scrape_official_active_roster(
        fetch_profile_details=False,
        resolve_ufcstats=False,
    )

    assert scraped.attrs["sync_complete"] is True
    assert scraped.attrs["sync_completeness_reason"] == "explicit_empty_terminal_page"
    assert scraped.attrs["pages_scraped"] == 1


def test_scraper_marks_partial_card_parse_as_incomplete(monkeypatch):
    partial_page = BeautifulSoup(
        """
        <html><head><title>Athletes - All | UFC</title></head><body>
          <form class="views-exposed-form"></form>
          <div class="c-listing-athlete-flipcard">
            <a href="/athlete/current-fighter">Profile</a>
            <span class="c-listing-athlete__name">Current Fighter</span>
          </div>
          <div class="c-listing-athlete-flipcard">
            <a href="/athlete/markup-drifted-fighter">Profile</a>
          </div>
        </body></html>
        """,
        "lxml",
    )
    terminal_page = BeautifulSoup(
        """
        <html><head><title>Athletes - All | UFC</title></head><body>
          <form class="views-exposed-form"></form>
          <div class="view-empty">No Result Found</div>
        </body></html>
        """,
        "lxml",
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "_get_soup",
        lambda url, session=None: terminal_page if "page=1" in url else partial_page,
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "_classify_combat_sport",
        lambda row, session=None: {
            "combat_sport": "mma",
            "combat_sport_reason": "test",
            "combat_sport_profile_url": "",
        },
    )

    scraped = ufc_active_roster.scrape_official_active_roster(
        fetch_profile_details=False,
        resolve_ufcstats=False,
    )

    assert scraped["official_name"].tolist() == ["Current Fighter"]
    assert scraped.attrs["sync_complete"] is False
    assert scraped.attrs["sync_completeness_reason"] == "card_parse_coverage_incomplete"
    assert scraped.attrs["selected_card_count"] == 2
    assert scraped.attrs["parsed_card_count"] == 1


@pytest.mark.parametrize(
    ("profile_error", "expected_outcome"),
    [
        (
            requests.ConnectionError("profile request failed"),
            ufc_active_roster._PROFILE_FETCH_REQUEST_FAILED,
        ),
        (
            requests.Timeout("profile request timed out"),
            ufc_active_roster._PROFILE_FETCH_REQUEST_FAILED,
        ),
        (
            requests.HTTPError("profile returned HTTP 404"),
            ufc_active_roster._PROFILE_FETCH_HTTP_ERROR,
        ),
        (
            requests.RequestException("other profile request failure"),
            ufc_active_roster._PROFILE_FETCH_REQUEST_ERROR,
        ),
        (
            ValueError("profile parser failed"),
            ufc_active_roster._PROFILE_FETCH_PARSE_FAILED,
        ),
        (
            ufc_active_roster.OfficialUFCProfileResponseUnavailableError(
                "not a profile page"
            ),
            ufc_active_roster._PROFILE_FETCH_INVALID_RESPONSE,
        ),
    ],
)
def test_scraper_tags_profile_request_and_parse_failures_separately(
    monkeypatch,
    profile_error,
    expected_outcome,
):
    first_page = BeautifulSoup(
        """
        <html><head><title>Athletes - All | UFC</title></head><body>
          <form class="views-exposed-form"></form>
          <div class="c-listing-athlete-flipcard">
            <a href="/athlete/current-fighter">Profile</a>
            <span class="c-listing-athlete__name">Current Fighter</span>
          </div>
        </body></html>
        """,
        "lxml",
    )
    terminal_page = BeautifulSoup(
        """
        <html><head><title>Athletes - All | UFC</title></head><body>
          <form class="views-exposed-form"></form>
          <div class="view-empty">No Result Found</div>
        </body></html>
        """,
        "lxml",
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "_get_soup",
        lambda url, session=None: terminal_page if "page=1" in url else first_page,
    )

    def fail_profile(*_args, **_kwargs):
        raise profile_error

    monkeypatch.setattr(
        ufc_active_roster,
        "scrape_official_athlete_profile",
        fail_profile,
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "_classify_combat_sport",
        lambda row, session=None: {
            "combat_sport": "mma",
            "combat_sport_reason": "test",
            "combat_sport_profile_url": "",
        },
    )

    scraped = ufc_active_roster.scrape_official_active_roster(
        fetch_profile_details=True,
        resolve_ufcstats=False,
    )

    assert scraped.loc[0, ufc_active_roster._PROFILE_FETCH_OUTCOME_COLUMN] == (
        expected_outcome
    )
    assert scraped.loc[0, "profile_status"] == "status_unknown"
    assert bool(scraped.loc[0, "coverage_eligible"]) is False


def test_scraper_continues_to_later_pages_after_partial_card_parse(monkeypatch):
    first_page = BeautifulSoup(
        """
        <html><head><title>Athletes - All | UFC</title></head><body>
          <form class="views-exposed-form"></form>
          <div class="c-listing-athlete-flipcard">
            <a href="/athlete/first-fighter">Profile</a>
            <span class="c-listing-athlete__name">First Fighter</span>
          </div>
          <div class="c-listing-athlete-flipcard">
            <a href="/athlete/malformed-fighter">Profile</a>
          </div>
        </body></html>
        """,
        "lxml",
    )
    second_page = BeautifulSoup(
        """
        <html><head><title>Athletes - All | UFC</title></head><body>
          <form class="views-exposed-form"></form>
          <div class="c-listing-athlete-flipcard">
            <a href="/athlete/later-fighter">Profile</a>
            <span class="c-listing-athlete__name">Later Fighter</span>
          </div>
        </body></html>
        """,
        "lxml",
    )
    terminal_page = BeautifulSoup(
        """
        <html><head><title>Athletes - All | UFC</title></head><body>
          <form class="views-exposed-form"></form>
          <div class="view-empty">No Result Found</div>
        </body></html>
        """,
        "lxml",
    )

    def fake_get_soup(url, session=None):
        if "page=2" in url:
            return terminal_page
        if "page=1" in url:
            return second_page
        return first_page

    monkeypatch.setattr(ufc_active_roster, "_get_soup", fake_get_soup)
    monkeypatch.setattr(
        ufc_active_roster,
        "_classify_combat_sport",
        lambda row, session=None: {
            "combat_sport": "mma",
            "combat_sport_reason": "test",
            "combat_sport_profile_url": "",
        },
    )

    scraped = ufc_active_roster.scrape_official_active_roster(
        fetch_profile_details=False,
        resolve_ufcstats=False,
    )

    assert scraped["official_name"].tolist() == ["First Fighter", "Later Fighter"]
    assert scraped.attrs["sync_complete"] is False
    assert scraped.attrs["sync_completeness_reason"] == "card_parse_coverage_incomplete"
    assert scraped.attrs["pages_scraped"] == 2
    assert scraped.attrs["selected_card_count"] == 3
    assert scraped.attrs["parsed_card_count"] == 2


def test_cached_roster_is_reused_when_retention_merge_raises(tmp_path, monkeypatch):
    roster_path = tmp_path / "ufc_active_roster_official.csv"
    audit_path = tmp_path / "ufc_active_roster_identity_audit.csv"
    original = pd.DataFrame([_row("Anchor"), _row("Missing Fighter")])
    original.to_csv(roster_path, index=False)
    audit_path.write_text("prior accepted audit", encoding="utf-8")
    _set_live_scrape(monkeypatch, [_row("Anchor"), _row("New Fighter")])
    monkeypatch.setattr(
        ufc_active_roster,
        "_merge_cached_roster_rows_missing_from_live",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("merge broke")),
    )

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=roster_path,
        identity_audit_path=audit_path,
    )

    assert synced.attrs["sync_source"] == "cached"
    assert synced.attrs["sync_fallback_used"] is True
    assert "Failed to preserve cached" in synced.attrs["sync_error"]
    assert set(pd.read_csv(roster_path)["official_name"]) == {
        "Anchor",
        "Missing Fighter",
    }
    assert audit_path.read_text(encoding="utf-8") == "prior accepted audit"


def test_skipped_profile_details_refuse_canonical_output_without_modifying_bytes(
    tmp_path,
    monkeypatch,
):
    canonical_path = tmp_path / "ufc_active_roster_official.csv"
    original_bytes = b"preexisting canonical roster bytes\r\n\x00"
    canonical_path.write_bytes(original_bytes)
    monkeypatch.setattr(
        ufc_active_roster,
        "OFFICIAL_ACTIVE_ROSTER_PATH",
        canonical_path,
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "scrape_official_active_roster",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("canonical refusal must happen before scraping")
        ),
    )
    aliased_canonical_path = canonical_path.parent / "unused" / ".." / canonical_path.name

    with pytest.raises(ValueError, match="noncanonical temporary --output path"):
        ufc_active_roster.sync_official_active_roster(
            output_path=aliased_canonical_path,
            fetch_profile_details=False,
            identity_audit_path=None,
        )

    assert canonical_path.read_bytes() == original_bytes


def test_skipped_combat_classification_refuses_canonical_output(
    tmp_path,
    monkeypatch,
):
    canonical_path = tmp_path / "ufc_active_roster_official.csv"
    original_bytes = b"preexisting classified canonical roster bytes"
    canonical_path.write_bytes(original_bytes)
    monkeypatch.setattr(
        ufc_active_roster,
        "OFFICIAL_ACTIVE_ROSTER_PATH",
        canonical_path,
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "scrape_official_active_roster",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("classification refusal must happen before scraping")
        ),
    )

    with pytest.raises(ValueError, match="without combat-sport classification"):
        ufc_active_roster.sync_official_active_roster(
            classify_combat_sport=False,
            identity_audit_path=None,
        )

    assert canonical_path.read_bytes() == original_bytes


def test_skipped_profile_details_can_write_noncanonical_temporary_output(
    tmp_path,
    monkeypatch,
):
    canonical_path = tmp_path / "ufc_active_roster_official.csv"
    temporary_path = tmp_path / "tapology_candidates.csv"
    monkeypatch.setattr(
        ufc_active_roster,
        "OFFICIAL_ACTIVE_ROSTER_PATH",
        canonical_path,
    )
    _set_live_scrape(monkeypatch, [_row("Listing Candidate", status="")])

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=temporary_path,
        fetch_profile_details=False,
        identity_audit_path=None,
    )

    assert temporary_path.exists()
    assert synced.loc[0, "profile_status"] == "status_unknown"
    assert bool(synced.loc[0, "coverage_eligible"]) is False
    assert pd.read_csv(temporary_path).loc[0, "official_name"] == "Listing Candidate"
    assert not canonical_path.exists()


def test_listing_only_cli_does_not_replace_canonical_identity_audit(
    tmp_path,
    monkeypatch,
):
    temporary_path = tmp_path / "tapology_candidates.csv"
    observed_kwargs: dict[str, object] = {}

    def fake_sync(**kwargs):
        observed_kwargs.update(kwargs)
        return pd.DataFrame([_row("Listing Candidate", status="")])

    monkeypatch.setattr(
        sync_active_roster_script,
        "sync_official_active_roster",
        fake_sync,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sync_active_ufc_roster.py",
            "--skip-profile-details",
            "--output",
            str(temporary_path),
        ],
    )

    assert sync_active_roster_script.main() == 0
    assert observed_kwargs["output_path"] == temporary_path
    assert observed_kwargs["fetch_profile_details"] is False
    assert observed_kwargs["identity_audit_path"] is None


def test_full_profile_custom_cli_leaves_audit_selection_to_core(
    tmp_path,
    monkeypatch,
):
    temporary_path = tmp_path / "full_profile_diagnostic.csv"
    observed_kwargs: dict[str, object] = {}

    def fake_sync(**kwargs):
        observed_kwargs.update(kwargs)
        return pd.DataFrame([_row("Verified Diagnostic Fighter")])

    monkeypatch.setattr(
        sync_active_roster_script,
        "sync_official_active_roster",
        fake_sync,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sync_active_ufc_roster.py",
            "--output",
            str(temporary_path),
        ],
    )

    assert sync_active_roster_script.main() == 0
    assert observed_kwargs["output_path"] == temporary_path
    assert observed_kwargs["fetch_profile_details"] is True
    assert "identity_audit_path" not in observed_kwargs


def test_full_profile_noncanonical_output_does_not_replace_canonical_audit(
    tmp_path,
    monkeypatch,
):
    canonical_path = tmp_path / "ufc_active_roster_official.csv"
    canonical_audit_path = tmp_path / "ufc_active_roster_identity_audit.csv"
    temporary_path = tmp_path / "full_profile_diagnostic.csv"
    original_audit_bytes = b"canonical audit must remain untouched\r\n\x00"
    canonical_audit_path.write_bytes(original_audit_bytes)
    monkeypatch.setattr(
        ufc_active_roster,
        "OFFICIAL_ACTIVE_ROSTER_PATH",
        canonical_path,
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "OFFICIAL_ACTIVE_ROSTER_IDENTITY_AUDIT_PATH",
        canonical_audit_path,
    )
    _set_live_scrape(monkeypatch, [_row("Verified Diagnostic Fighter")])

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=temporary_path,
        fetch_profile_details=True,
    )

    assert synced.loc[0, "official_name"] == "Verified Diagnostic Fighter"
    assert temporary_path.exists()
    assert canonical_audit_path.read_bytes() == original_audit_bytes
    assert not canonical_path.exists()


def test_explicit_canonical_audit_rejects_noncanonical_roster_generation(
    tmp_path,
    monkeypatch,
):
    canonical_path = tmp_path / "ufc_active_roster_official.csv"
    canonical_audit_path = tmp_path / "ufc_active_roster_identity_audit.csv"
    temporary_path = tmp_path / "full_profile_diagnostic.csv"
    original_audit_bytes = b"accepted canonical audit bytes"
    canonical_audit_path.write_bytes(original_audit_bytes)
    monkeypatch.setattr(
        ufc_active_roster,
        "OFFICIAL_ACTIVE_ROSTER_PATH",
        canonical_path,
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "OFFICIAL_ACTIVE_ROSTER_IDENTITY_AUDIT_PATH",
        canonical_audit_path,
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "scrape_official_active_roster",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("cross-artifact refusal must happen before scraping")
        ),
    )

    with pytest.raises(ValueError, match="noncanonical UFC roster generation"):
        ufc_active_roster.sync_official_active_roster(
            output_path=temporary_path,
            identity_audit_path=canonical_audit_path,
        )

    assert canonical_audit_path.read_bytes() == original_audit_bytes
    assert not temporary_path.exists()


def test_listing_only_hardlink_alias_of_canonical_is_refused(
    tmp_path,
    monkeypatch,
):
    canonical_path = tmp_path / "ufc_active_roster_official.csv"
    hardlink_alias = tmp_path / "diagnostic-hardlink.csv"
    original_bytes = b"preexisting canonical roster bytes"
    canonical_path.write_bytes(original_bytes)
    os.link(canonical_path, hardlink_alias)
    monkeypatch.setattr(
        ufc_active_roster,
        "OFFICIAL_ACTIVE_ROSTER_PATH",
        canonical_path,
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "scrape_official_active_roster",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("hardlink refusal must happen before scraping")
        ),
    )

    with pytest.raises(ValueError, match="noncanonical temporary --output path"):
        ufc_active_roster.sync_official_active_roster(
            output_path=hardlink_alias,
            fetch_profile_details=False,
            identity_audit_path=None,
        )

    assert canonical_path.read_bytes() == original_bytes
    assert hardlink_alias.read_bytes() == original_bytes


def test_full_profile_hardlink_alias_does_not_publish_canonical_audit(
    tmp_path,
    monkeypatch,
):
    canonical_path = tmp_path / "ufc_active_roster_official.csv"
    hardlink_alias = tmp_path / "full-profile-hardlink.csv"
    canonical_audit_path = tmp_path / "ufc_active_roster_identity_audit.csv"
    pd.DataFrame([_row("Cached Fighter")]).to_csv(canonical_path, index=False)
    original_canonical_bytes = canonical_path.read_bytes()
    os.link(canonical_path, hardlink_alias)
    original_audit_bytes = b"canonical audit remains generation-bound"
    canonical_audit_path.write_bytes(original_audit_bytes)
    monkeypatch.setattr(
        ufc_active_roster,
        "OFFICIAL_ACTIVE_ROSTER_PATH",
        canonical_path,
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "OFFICIAL_ACTIVE_ROSTER_IDENTITY_AUDIT_PATH",
        canonical_audit_path,
    )
    _set_live_scrape(monkeypatch, [_row("Cached Fighter")])

    synced = ufc_active_roster.sync_official_active_roster(
        output_path=hardlink_alias,
        fetch_profile_details=True,
    )

    assert synced.loc[0, "official_name"] == "Cached Fighter"
    assert canonical_path.read_bytes() == original_canonical_bytes
    assert hardlink_alias.read_bytes() != original_canonical_bytes
    assert canonical_audit_path.read_bytes() == original_audit_bytes


def test_identity_audit_path_cannot_overwrite_canonical_roster_bytes(
    tmp_path,
    monkeypatch,
):
    canonical_path = tmp_path / "ufc_active_roster_official.csv"
    temporary_path = tmp_path / "listing_candidates.csv"
    original_bytes = b"verified canonical roster bytes\r\n\x00"
    canonical_path.write_bytes(original_bytes)
    monkeypatch.setattr(
        ufc_active_roster,
        "OFFICIAL_ACTIVE_ROSTER_PATH",
        canonical_path,
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "scrape_official_active_roster",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("audit-path refusal must happen before scraping")
        ),
    )

    with pytest.raises(ValueError, match="identity-audit output path"):
        ufc_active_roster.sync_official_active_roster(
            output_path=temporary_path,
            fetch_profile_details=False,
            identity_audit_path=canonical_path,
        )

    assert canonical_path.read_bytes() == original_bytes
    assert not temporary_path.exists()


def test_roster_and_identity_audit_outputs_must_be_distinct(
    tmp_path,
    monkeypatch,
):
    shared_path = tmp_path / "shared.csv"
    original_bytes = b"preexisting shared output"
    shared_path.write_bytes(original_bytes)
    monkeypatch.setattr(
        ufc_active_roster,
        "scrape_official_active_roster",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("path-collision refusal must happen before scraping")
        ),
    )

    with pytest.raises(ValueError, match="same output path"):
        ufc_active_roster.sync_official_active_roster(
            output_path=shared_path,
            identity_audit_path=shared_path,
        )

    assert shared_path.read_bytes() == original_bytes


def test_full_profile_sync_can_write_canonical_output(tmp_path, monkeypatch):
    canonical_path = tmp_path / "ufc_active_roster_official.csv"
    canonical_audit_path = tmp_path / "ufc_active_roster_identity_audit.csv"
    monkeypatch.setattr(
        ufc_active_roster,
        "OFFICIAL_ACTIVE_ROSTER_PATH",
        canonical_path,
    )
    monkeypatch.setattr(
        ufc_active_roster,
        "OFFICIAL_ACTIVE_ROSTER_IDENTITY_AUDIT_PATH",
        canonical_audit_path,
    )
    _set_live_scrape(monkeypatch, [_row("Verified Active Fighter")])

    synced = ufc_active_roster.sync_official_active_roster(
        fetch_profile_details=True,
    )

    assert synced.loc[0, "profile_status"] == "Active"
    assert bool(synced.loc[0, "coverage_eligible"]) is True
    assert pd.read_csv(canonical_path).loc[0, "official_name"] == "Verified Active Fighter"
    audit = pd.read_csv(canonical_audit_path)
    assert audit.empty
    assert (
        audit.columns.tolist()
        == list(ufc_active_roster.IDENTITY_AUDIT_COLUMNS)
    )
