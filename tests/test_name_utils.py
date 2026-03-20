from src.data import fighter_lookup
from src.data.name_utils import name_appears_in_text, same_person_name
from bs4 import BeautifulSoup


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


def test_search_fighter_url_uses_suffix_stripped_last_name_initial(monkeypatch):
    fighter_lookup.clear_cache()
    requested_urls = []

    def fake_get_soup(url):
        requested_urls.append(url)
        return BeautifulSoup(
            """
            <table>
              <tr class="b-statistics__table-row">
                <td><a class="b-link" href="http://ufcstats.com/fighter-details/steve-garcia">Steve</a></td>
                <td><a class="b-link">Garcia</a></td>
              </tr>
            </table>
            """,
            "lxml",
        )

    monkeypatch.setattr(fighter_lookup, "_get_soup", fake_get_soup)

    assert (
        fighter_lookup.search_fighter_url("Steve Garcia Jr.")
        == "http://ufcstats.com/fighter-details/steve-garcia"
    )
    assert requested_urls[0].endswith("char=g&page=all")


def test_search_fighter_url_cache_uses_cross_source_alias_key(monkeypatch):
    fighter_lookup.clear_cache()
    requested_urls = []

    def fake_get_soup(url):
        requested_urls.append(url)
        return BeautifulSoup(
            """
            <table>
              <tr class="b-statistics__table-row">
                <td><a class="b-link" href="http://ufcstats.com/fighter-details/joe-pyfer">Joe</a></td>
                <td><a class="b-link">Pyfer</a></td>
              </tr>
            </table>
            """,
            "lxml",
        )

    monkeypatch.setattr(fighter_lookup, "_get_soup", fake_get_soup)

    assert fighter_lookup.search_fighter_url("Joe Pyfer") == "http://ufcstats.com/fighter-details/joe-pyfer"
    assert fighter_lookup.search_fighter_url("Joseph Pyfer") == "http://ufcstats.com/fighter-details/joe-pyfer"
    assert len(requested_urls) == 1
