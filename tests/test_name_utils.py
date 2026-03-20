from src.data import fighter_lookup
from src.data.name_utils import name_appears_in_text, same_person_name


def test_same_person_name_matches_aliases_and_suffixes():
    assert same_person_name("Joseph Pyfer", "Joe Pyfer")
    assert same_person_name("Steve Garcia Jr.", "Steve Garcia")
    assert not same_person_name("Joseph Pyfer", "Jack Hermansson")


def test_name_appears_in_text_matches_cross_source_aliases():
    assert name_appears_in_text(
        "Joseph Pyfer",
        "UFC Fight Night: Hermansson vs. Joe Pyfer",
    )
    assert name_appears_in_text(
        "Steve Garcia Jr.",
        "Featherweight bout: Steve Garcia vs. Calvin Kattar",
    )


def test_resolve_fighter_key_matches_cross_source_aliases():
    assert fighter_lookup._resolve_fighter_key("Joseph Pyfer", {"Joe Pyfer": 1}) == "Joe Pyfer"


def test_get_fighter_elo_matches_cross_source_aliases(monkeypatch):
    monkeypatch.setattr(
        fighter_lookup,
        "_load_elo_state",
        lambda processed_data_dir=None: {"ratings": {"Joe Pyfer": 1542.0}},
    )

    assert fighter_lookup.get_fighter_elo("Joseph Pyfer") == 1542.0
