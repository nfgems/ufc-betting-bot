from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import fighter_lookup as fighter_lookup_module
from src.data import fallback_scrapers
from src.data.name_utils import (
    REVIEWED_FIGHTER_IDENTITIES,
    canonicalize_reviewed_fighter_name,
    normalize_cross_source_name,
    same_person_name,
)
from src.data.ufc_refresh import _canonicalize_reviewed_training_identities
from src.features import build_features as build_features_module


EXPECTED_PRO_TOTALS = {
    "Gabriel Santos": (10, 10, 0),
    "Kai Kamaka III": (9, 7, 2),
    "Mizuki": (18, 13, 5),
}


def test_reviewed_identity_map_pins_profiles_dobs_and_only_exact_aliases():
    expected = {
        "f2f140ce7532e327": ("Gabriel Santos", "1996-11-28", "179211", "159865"),
        "eee0ef3e2b14816b": ("Kai Kamaka III", "1995-01-05", "117585", "74838"),
        "43a59ce3bb40449e": ("Mizuki", "1994-08-19", "71390", "25717"),
    }
    assert set(REVIEWED_FIGHTER_IDENTITIES) == set(expected)
    for ufcstats_id, values in expected.items():
        identity = REVIEWED_FIGHTER_IDENTITIES[ufcstats_id]
        assert (
            identity["canonical_name"],
            identity["dob"],
            identity["sherdog_profile_id"],
            identity["tapology_profile_id"],
        ) == values
        assert str(identity["ufcstats_url"]).endswith(ufcstats_id)
        assert str(identity["sherdog_url"]).startswith("https://www.sherdog.com/")
        assert str(identity["tapology_url"]).startswith("https://www.tapology.com/")

    assert canonicalize_reviewed_fighter_name("Kai Kamaka") == "Kai Kamaka III"
    assert canonicalize_reviewed_fighter_name("Kai Kamaka III") == "Kai Kamaka III"
    assert canonicalize_reviewed_fighter_name("Kai Kamaka Jr.") == "Kai Kamaka Jr."
    assert canonicalize_reviewed_fighter_name("Mizuki Endo") == "Mizuki Endo"
    assert canonicalize_reviewed_fighter_name("Gabriel Santos Jr.") == "Gabriel Santos Jr."
    assert canonicalize_reviewed_fighter_name("Lance Gibson") == "Lance Gibson"
    assert canonicalize_reviewed_fighter_name("Lance Gibson Jr.") == "Lance Gibson Jr."
    assert same_person_name("Kai Kamaka", "Kai Kamaka III")
    assert same_person_name("Mizuki Inoue", "Mizuki")
    assert same_person_name("kai kamaka", "Kai Kamaka III")
    assert same_person_name("GABRIEL SANTOS", "Gabriel Santos")
    assert same_person_name("mizuki inoue", "Mizuki")
    assert not same_person_name("Kai Kamaka Jr.", "Kai Kamaka III")
    assert not same_person_name("Gabriel Santos Jr.", "Gabriel Santos")
    assert not same_person_name("Mizuki Endo", "Mizuki")


def test_cross_source_keys_preserve_reviewed_suffix_negative_controls():
    assert normalize_cross_source_name("Kai Kamaka") == "kai kamaka iii"
    assert normalize_cross_source_name("Kai Kamaka III") == "kai kamaka iii"
    assert normalize_cross_source_name("kai kamaka") == "kai kamaka iii"
    assert normalize_cross_source_name("Mizuki Inoue") == "mizuki"
    assert normalize_cross_source_name("Kai Kamaka Jr.") == "kai kamaka jr"
    assert normalize_cross_source_name("Gabriel Santos Jr.") == "gabriel santos jr"

    # Preserve the legacy suffix behavior for names outside the reviewed set.
    assert normalize_cross_source_name("Steve Garcia Jr.") == (
        normalize_cross_source_name("Steve Garcia")
    )


def test_live_alias_map_does_not_merge_reviewed_suffix_negative_controls(
    monkeypatch,
):
    from src import bot

    alias_map = {
        normalize_cross_source_name("Kai Kamaka III"): "kai kamaka iii",
        normalize_cross_source_name("Gabriel Santos"): "gabriel santos",
        normalize_cross_source_name("Mizuki"): "mizuki",
    }
    monkeypatch.setattr(bot, "_load_live_fighter_alias_map", lambda: alias_map)

    assert bot._canonicalize_live_fighter_name("Kai Kamaka") == "kai kamaka iii"
    assert bot._canonicalize_live_fighter_name("Mizuki Inoue") == "mizuki"
    assert bot._canonicalize_live_fighter_name("Kai Kamaka Jr.") == "kai kamaka jr"
    assert bot._canonicalize_live_fighter_name("Gabriel Santos Jr.") == (
        "gabriel santos jr"
    )


def test_roster_local_profile_does_not_merge_reviewed_suffix_negative_controls():
    from src.data import ufc_active_roster

    kai_candidate = {
        "name": "Kai Kamaka III",
        "fighter_url": "http://ufcstats.com/fighter-details/eee0ef3e2b14816b",
        "source": "fixture",
    }
    gabriel_candidate = {
        "name": "Gabriel Santos",
        "fighter_url": "http://ufcstats.com/fighter-details/f2f140ce7532e327",
        "source": "fixture",
    }
    candidates = {
        normalize_cross_source_name(kai_candidate["name"]): [kai_candidate],
        normalize_cross_source_name(gabriel_candidate["name"]): [gabriel_candidate],
    }

    approved = ufc_active_roster._resolve_local_ufcstats_profile(
        {"official_name": "Kai Kamaka"},
        candidates=candidates,
    )
    kai_negative = ufc_active_roster._resolve_local_ufcstats_profile(
        {"official_name": "Kai Kamaka Jr."},
        candidates=candidates,
    )
    gabriel_negative = ufc_active_roster._resolve_local_ufcstats_profile(
        {"official_name": "Gabriel Santos Jr."},
        candidates=candidates,
    )

    assert approved["ufcstats_url"] == kai_candidate["fighter_url"]
    assert kai_negative["ufcstats_url"] == ""
    assert gabriel_negative["ufcstats_url"] == ""


def test_reviewed_history_rows_have_direct_evidence_and_exact_totals():
    raw = pd.read_csv(
        build_features_module._resolve_reviewed_supplement_history_path()
    )
    assert len(raw) == 40
    assert raw["ufcstats_id"].notna().all()
    assert raw["source_profile_id"].notna().all()
    assert raw["source_profile_url"].str.startswith("https://www.sherdog.com/").all()
    assert raw["subject_dob"].eq(raw["source_profile_dob"]).all()
    assert raw["dob_match"].eq(True).all()
    assert raw["fighter_b"].astype(str).str.strip().ne("").all()
    assert raw["event_date"].notna().all()
    assert raw["subject_result"].isin({"win", "loss"}).all()
    assert raw["source"].eq("sherdog").all()

    professional = build_features_module._load_reviewed_supplement_history(
        "professional"
    )
    summary = build_features_module._compute_pre_ufc_summary(professional).set_index(
        "fighter"
    )
    observed = {
        fighter: (
            int(summary.loc[fighter, "pre_ufc_total_fights"]),
            int(summary.loc[fighter, "pre_ufc_wins"]),
            int(summary.loc[fighter, "pre_ufc_losses"]),
        )
        for fighter in EXPECTED_PRO_TOTALS
    }
    assert observed == EXPECTED_PRO_TOTALS

    amateur = build_features_module._load_reviewed_supplement_history("amateur")
    amateur_summary = build_features_module._compute_amateur_summary(amateur).set_index(
        "fighter"
    )
    assert list(amateur_summary.index) == ["Kai Kamaka III"]
    assert (
        int(amateur_summary.loc["Kai Kamaka III", "amateur_total_fights"]),
        int(amateur_summary.loc["Kai Kamaka III", "amateur_wins"]),
        int(amateur_summary.loc["Kai Kamaka III", "amateur_losses"]),
    ) == (3, 2, 1)
    assert "Gabriel Santos" not in amateur_summary.index


def test_reviewed_history_exact_rows_are_pinned_by_canonical_digest(
    tmp_path: Path,
    monkeypatch,
):
    source_path = build_features_module._resolve_reviewed_supplement_history_path()
    assert (
        build_features_module._reviewed_history_canonical_sha256(source_path)
        == build_features_module._REVIEWED_SUPPLEMENT_HISTORY_CANONICAL_SHA256
    )

    source = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    source.loc[0, "fighter_b"] = "Invented Opponent"
    source.loc[0, "event_date"] = "2025-01-01"
    source.loc[0, "method"] = "Invented Method"
    candidate = tmp_path / "reviewed_fighter_history.csv"
    source.to_csv(candidate, index=False)
    monkeypatch.setattr(
        build_features_module,
        "_resolve_reviewed_supplement_history_path",
        lambda: candidate,
    )

    assert build_features_module._load_reviewed_supplement_history(
        "professional"
    ).empty
    assert build_features_module._load_reviewed_supplement_history("amateur").empty


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("subject_dob", "1973-03-05"),
        ("source_profile_url", "https://www.sherdog.com/fighter/Kai-Kamaka-504"),
        ("dob_match", False),
    ],
)
def test_reviewed_history_rejects_an_entire_identity_on_bad_evidence(
    tmp_path: Path,
    monkeypatch,
    field: str,
    bad_value: object,
):
    source = pd.read_csv(
        build_features_module._resolve_reviewed_supplement_history_path()
    )
    target_id = "eee0ef3e2b14816b"
    source.loc[
        (source["ufcstats_id"] == target_id)
        & (source["history_type"] == "professional"),
        field,
    ] = bad_value
    candidate = tmp_path / "reviewed_fighter_history.csv"
    source.to_csv(candidate, index=False)
    monkeypatch.setattr(
        build_features_module,
        "_resolve_reviewed_supplement_history_path",
        lambda: candidate,
    )
    monkeypatch.setattr(
        build_features_module,
        "_reviewed_history_canonical_sha256",
        lambda _path: build_features_module._REVIEWED_SUPPLEMENT_HISTORY_CANONICAL_SHA256,
    )

    reviewed = build_features_module._load_reviewed_supplement_history(
        "professional"
    )

    assert "Kai Kamaka III" not in set(reviewed.get("fighter", []))
    assert {"Gabriel Santos", "Mizuki"}.issubset(set(reviewed["fighter"]))


def test_legacy_collisions_stay_quarantined_while_reviewed_histories_load():
    professional = build_features_module._load_pre_ufc_supplement(
        build_features_module._resolve_pre_ufc_supplement_path(),
        include_reviewed=True,
    )
    for fighter, expected in EXPECTED_PRO_TOTALS.items():
        rows = professional[professional["fighter"] == fighter]
        assert (len(rows), int(rows["won"].sum()), int((rows["result_label"] == "loss").sum())) == expected

    assert "Kai Kamaka" not in set(professional["fighter"])
    assert professional.loc[
        professional["fighter"] == "Kai Kamaka III", "event_date"
    ].min() == pd.Timestamp("2012-10-20")
    assert professional.loc[
        professional["fighter"] == "Mizuki", "event_date"
    ].min() == pd.Timestamp("2010-10-10")
    assert pd.Timestamp("2007-03-22") not in set(
        professional.loc[
            professional["fighter"] == "Gabriel Santos", "event_date"
        ]
    )

    amateur = build_features_module._load_amateur_supplement(
        build_features_module._resolve_amateur_supplement_path(),
        include_reviewed=True,
    )
    amateur_summary = build_features_module._compute_amateur_summary(amateur).set_index(
        "fighter"
    )
    assert "Gabriel Santos" not in amateur_summary.index
    assert "Kai Kamaka" not in amateur_summary.index
    assert int(amateur_summary.loc["Kai Kamaka III", "amateur_total_fights"]) == 3


def test_training_identity_repair_maps_fighter_columns_and_winner_only_exactly():
    frame = pd.DataFrame(
        [
            {
                "fighter_a": "Kai Kamaka",
                "fighter_b": "Opponent One",
                "winner": "Kai Kamaka",
            },
            {
                "fighter_a": "Kai Kamaka Jr.",
                "fighter_b": "Mizuki Endo",
                "winner": "Kai Kamaka Jr.",
            },
        ]
    )

    repaired = _canonicalize_reviewed_training_identities(frame)

    assert repaired.loc[0].to_dict() == {
        "fighter_a": "Kai Kamaka III",
        "fighter_b": "Opponent One",
        "winner": "Kai Kamaka III",
    }
    assert repaired.loc[1].to_dict() == frame.loc[1].to_dict()


def test_live_processed_lookup_honors_reviewed_aliases_and_suffix_controls(
    tmp_path: Path,
):
    pd.DataFrame(
        [
            {
                "event_date": "2026-01-01",
                "fighter_a": "Kai Kamaka III",
                "fighter_b": "Opponent One",
                "winner": "Kai Kamaka III",
                "a_num_fights": 5,
                "b_num_fights": 1,
            },
            {
                "event_date": "2026-01-02",
                "fighter_a": "Gabriel Santos",
                "fighter_b": "Mizuki",
                "winner": "Mizuki",
                "a_num_fights": 3,
                "b_num_fights": 3,
            },
        ]
    ).to_csv(tmp_path / "features.csv", index=False)
    fighter_lookup_module._processed_feature_history_cache.clear()
    fighter_lookup_module._processed_feature_history_mtime.clear()

    kai = fighter_lookup_module._lookup_processed_fighter(
        "Kai Kamaka",
        as_of_date="2026-01-01",
        processed_data_dir=tmp_path,
    )
    mizuki = fighter_lookup_module._lookup_processed_fighter(
        "Mizuki Inoue",
        as_of_date="2026-01-02",
        processed_data_dir=tmp_path,
    )

    assert kai is not None and kai["profile"]["name"] == "Kai Kamaka III"
    assert mizuki is not None and mizuki["profile"]["name"] == "Mizuki"
    assert fighter_lookup_module._lookup_processed_fighter(
        "Kai Kamaka Jr.",
        as_of_date="2026-01-02",
        processed_data_dir=tmp_path,
    ) is None
    assert fighter_lookup_module._lookup_processed_fighter(
        "Gabriel Santos Jr.",
        as_of_date="2026-01-02",
        processed_data_dir=tmp_path,
    ) is None


def test_ufcstats_search_does_not_cache_or_cross_match_reviewed_suffix_controls(
    monkeypatch,
):
    fighter_lookup_module.clear_cache()
    requested_urls = []

    def fake_get_soup(url):
        requested_urls.append(url)
        return fighter_lookup_module.BeautifulSoup(
            """
            <table>
              <tr class="b-statistics__table-row">
                <td><a class="b-link" href="http://ufcstats.com/fighter-details/eee0ef3e2b14816b">Kai</a></td>
                <td><a class="b-link">Kamaka III</a></td>
              </tr>
            </table>
            """,
            "lxml",
        )

    monkeypatch.setattr(fighter_lookup_module, "_get_soup", fake_get_soup)

    assert fighter_lookup_module.search_fighter_url("kai kamaka") == (
        "http://ufcstats.com/fighter-details/eee0ef3e2b14816b"
    )
    canonical_requests = len(requested_urls)
    assert fighter_lookup_module.search_fighter_url("Kai Kamaka Jr.") is None
    assert len(requested_urls) > canonical_requests


@pytest.mark.parametrize(
    ("query", "candidate", "profile_url", "expected"),
    [
        (
            "Kai Kamaka",
            "Kai Kamaka III",
            "https://www.sherdog.com/fighter/Kai-Kamaka-III-117585",
            True,
        ),
        (
            "Mizuki Inoue",
            "Mizuki",
            "https://www.sherdog.com/fighter/Mizuki-Inoue-71390",
            True,
        ),
        (
            "Kai Kamaka Jr.",
            "Kai Kamaka III",
            "https://www.sherdog.com/fighter/Kai-Kamaka-III-117585",
            False,
        ),
        (
            "Gabriel Santos Jr.",
            "Gabriel Santos",
            "https://www.sherdog.com/fighter/Gabriel-Santos-179211",
            False,
        ),
    ],
)
def test_fallback_search_refuses_fuzzy_matches_for_reviewed_identities(
    query: str,
    candidate: str,
    profile_url: str,
    expected: bool,
):
    assert fallback_scrapers._best_name_score(
        query,
        candidate,
        profile_url,
    ) == (100 if expected else 0)
    assert fallback_scrapers._candidate_has_required_name_tokens(
        query,
        candidate,
        profile_url,
    ) is expected


def test_reviewed_supplements_load_without_legacy_files(
    tmp_path: Path,
    monkeypatch,
):
    missing_pre_ufc = tmp_path / "missing_pre_ufc.csv"
    missing_amateur = tmp_path / "missing_amateur.csv"
    professional = build_features_module._load_pre_ufc_supplement(
        missing_pre_ufc,
        include_reviewed=True,
    )
    amateur = build_features_module._load_amateur_supplement(
        missing_amateur,
        include_reviewed=True,
    )
    assert len(professional[professional["fighter"] == "Kai Kamaka III"]) == 9
    assert len(amateur[amateur["fighter"] == "Kai Kamaka III"]) == 3

    monkeypatch.setattr(
        build_features_module,
        "_resolve_pre_ufc_supplement_path",
        lambda: missing_pre_ufc,
    )
    monkeypatch.setattr(
        build_features_module,
        "_resolve_amateur_supplement_path",
        lambda: missing_amateur,
    )
    fight = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2020-08-15"),
                "fighter_a": "Kai Kamaka III",
                "fighter_b": "Tony Kelley",
                "winner": "Kai Kamaka III",
                "method": "Decision",
                "finish_round": 3,
                "num_rounds": 3,
                "title_bout": 0,
            }
        ]
    )
    features = build_features_module.build_features(fight)
    assert features.iloc[0]["a_pre_ufc_total_fights"] == 9
    assert features.iloc[0]["a_amateur_total_fights"] == 3


def test_reviewed_fight_count_aliases_do_not_merge_suffix_negative_controls(
    tmp_path: Path,
    monkeypatch,
):
    pd.DataFrame(
        [
            {"fighter_a": "Kai Kamaka III", "fighter_b": "Mizuki", "a_num_fights": 0, "b_num_fights": 0},
            {"fighter_a": "Gabriel Santos", "fighter_b": "Kai Kamaka III", "a_num_fights": 0, "b_num_fights": 1},
        ]
    ).to_csv(tmp_path / "features.csv", index=False)
    monkeypatch.setattr(build_features_module, "PROCESSED_DATA_DIR", tmp_path)

    assert build_features_module.get_fighter_ufc_fight_count("Kai Kamaka") == 2
    assert build_features_module.get_fighter_ufc_fight_count("Mizuki Inoue") == 1
    assert build_features_module.get_fighter_ufc_fight_count("Kai Kamaka Jr.") == 0
    assert build_features_module.get_fighter_ufc_fight_count("Gabriel Santos Jr.") == 0


def test_kai_six_ufc_rows_form_one_history_with_four_and_five_prior_fights():
    fights = pd.DataFrame(
        [
            ("2020-08-15", "Kai Kamaka", "Tony Kelley", "Kai Kamaka"),
            ("2020-11-28", "Jonathan Pearce", "Kai Kamaka", "Jonathan Pearce"),
            ("2021-05-01", "Kai Kamaka", "TJ Brown", "TJ Brown"),
            ("2021-07-31", "Danny Chavez", "Kai Kamaka", ""),
            ("2026-04-04", "Kai Kamaka III", "Dakota Hope", "Kai Kamaka III"),
            ("2026-07-11", "Luke Riley", "Kai Kamaka III", "Luke Riley"),
        ],
        columns=["event_date", "fighter_a", "fighter_b", "winner"],
    )
    fights["event_date"] = pd.to_datetime(fights["event_date"])
    fights["method"] = "Decision"
    fights["finish_round"] = 3
    fights["num_rounds"] = 3
    fights["title_bout"] = 0
    repaired = _canonicalize_reviewed_training_identities(fights)

    features = build_features_module.build_features(repaired)
    first_2026 = features[features["event_date"] == pd.Timestamp("2026-04-04")].iloc[0]
    second_2026 = features[features["event_date"] == pd.Timestamp("2026-07-11")].iloc[0]

    assert set(repaired.loc[
        repaired["fighter_a"].str.contains("Kai Kamaka", na=False), "fighter_a"
    ]).union(
        repaired.loc[
            repaired["fighter_b"].str.contains("Kai Kamaka", na=False), "fighter_b"
        ]
    ) == {"Kai Kamaka III"}
    assert first_2026["a_num_fights"] == 4
    assert second_2026["b_num_fights"] == 5
    assert not np.isnan(first_2026["a_amateur_total_fights"])


def test_gabriel_unverified_amateur_history_remains_nan():
    fights = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2023-03-18"),
                "fighter_a": "Gabriel Santos",
                "fighter_b": "Lerone Murphy",
                "winner": "Lerone Murphy",
                "method": "Decision",
                "finish_round": 3,
                "num_rounds": 3,
                "title_bout": 0,
            }
        ]
    )

    features = build_features_module.build_features(fights)

    assert pd.isna(features.iloc[0]["a_amateur_total_fights"])
