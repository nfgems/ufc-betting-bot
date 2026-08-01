from src.data import fighter_lookup
from src.data.name_utils import (
    canonical_fighter_display_name,
    name_appears_in_text,
    normalize_cross_source_name,
    normalize_person_name,
    same_person_name,
)
from bs4 import BeautifulSoup


def test_same_person_name_matches_aliases_and_suffixes():
    assert same_person_name("Joseph Pyfer", "Joe Pyfer")
    assert same_person_name("Steve Garcia Jr.", "Steve Garcia")
    assert not same_person_name("Joseph Pyfer", "Jack Hermansson")


def test_same_person_name_matches_ian_garry_canonical_name():
    assert same_person_name("Ian Garry", "Ian Machado Garry")


def test_name_appears_in_text_matches_cross_source_aliases():
    assert name_appears_in_text(
        "Joseph Pyfer",
        "UFC Fight Night: Hermansson vs. Joe Pyfer",
    )
    assert name_appears_in_text(
        "Steve Garcia Jr.",
        "Featherweight bout: Steve Garcia vs. Calvin Kattar",
    )


def test_normalize_person_name_transliterates_non_decomposing_letters():
    assert normalize_person_name("Jan Błachowicz") == "jan blachowicz"
    assert same_person_name("Jan Błachowicz", "Jan Blachowicz")


def test_luis_felipe_dias_cross_source_alias():
    assert normalize_cross_source_name("Luis Dias de Assis") == "luis felipe dias"
    assert normalize_cross_source_name("Luis Felipe Dias de Assis") == "luis felipe dias"
    assert same_person_name("Luis Dias de Assis", "Luis Felipe Dias")
    assert canonical_fighter_display_name("Luis Dias de Assis") == "Luis Felipe Dias"


def test_bobby_green_cross_source_alias():
    assert normalize_cross_source_name("Bobby Green") == "king green"
    assert same_person_name("Bobby Green", "King Green")
    assert canonical_fighter_display_name("Bobby Green") == "King Green"


def test_nursultan_ruziboev_cross_source_alias():
    assert normalize_cross_source_name("Nursultan Ruziboev") == "nursulton ruziboev"
    assert same_person_name("Nursultan Ruziboev", "Nursulton Ruziboev")
    assert canonical_fighter_display_name("Nursultan Ruziboev") == "Nursulton Ruziboev"


def test_patricio_freire_cross_source_alias():
    assert normalize_cross_source_name("Patricio Freire") == "patricio pitbull"
    assert (
        normalize_cross_source_name("Patricio Pitbull Freire")
        == "patricio pitbull"
    )
    assert same_person_name("Patricio Freire", "Patricio Pitbull")


def test_zachary_reese_cross_source_alias():
    assert normalize_cross_source_name("Zach Reese") == "zachary reese"
    assert normalize_cross_source_name("Zack Reese") == "zachary reese"
    assert same_person_name("Zachary Reese", "Zach Reese")


def test_asu_almabaev_cross_source_alias():
    assert normalize_cross_source_name("Asu Almabaev") == "asu almabayev"
    assert normalize_cross_source_name("Assu Almabaev") == "asu almabayev"
    assert same_person_name("Asu Almabaev", "Asu Almabayev")
    assert canonical_fighter_display_name("Asu Almabaev") == "Asu Almabayev"


def test_current_ufc_card_cross_source_aliases():
    assert normalize_cross_source_name("Ezra Elliot") == "ezra elliott"
    assert normalize_cross_source_name("Ezra Elliott") == "ezra elliott"
    assert normalize_cross_source_name("Seokhyeon Ko") == "seok hyun ko"
    assert normalize_cross_source_name("Seok Hyun Ko") == "seok hyun ko"


def test_dennis_buzukia_cross_source_token_alias():
    assert normalize_person_name("Dennis Buzukia") == "dennis buzukia"
    assert normalize_cross_source_name("Dennis Buzukia") == "dennis buzukja"
    assert same_person_name("Dennis Buzukia", "Dennis Buzukja")
    assert name_appears_in_text(
        "Dennis Buzukja",
        "UFC Belgrade: Bogdan Grad vs Dennis Buzukia",
    )
    assert canonical_fighter_display_name("Dennis Buzukia") == "Dennis Buzukja"


def test_ludovit_klein_apostrophe_artifact_matches_official_name():
    assert normalize_person_name("L'udovit Klein") == "l udovit klein"
    assert normalize_cross_source_name("L'udovit Klein") == "ludovit klein"
    assert same_person_name("L'udovit Klein", "Ludovít Klein")
    assert name_appears_in_text(
        "L'udovit Klein",
        "UFC Belgrade: Ludovít Klein vs Tofiq Musayev",
    )
    assert canonical_fighter_display_name("L'udovit Klein") == "Ludovit Klein"


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


def test_search_fighter_url_uses_curated_fighter_alias(monkeypatch):
    fighter_lookup.clear_cache()
    requested_urls = []

    def fake_get_soup(url):
        requested_urls.append(url)
        return BeautifulSoup(
            """
            <table>
              <tr class="b-statistics__table-row">
                <td><a class="b-link" href="http://ufcstats.com/fighter-details/king-green">King</a></td>
                <td><a class="b-link">Green</a></td>
              </tr>
            </table>
            """,
            "lxml",
        )

    monkeypatch.setattr(fighter_lookup, "_get_soup", fake_get_soup)

    assert fighter_lookup.search_fighter_url("Bobby Green") == "http://ufcstats.com/fighter-details/king-green"
    assert requested_urls[0].endswith("char=g&page=all")


def test_search_fighter_url_uses_nursultan_ruziboev_alias(monkeypatch):
    fighter_lookup.clear_cache()
    requested_urls = []

    def fake_get_soup(url):
        requested_urls.append(url)
        return BeautifulSoup(
            """
            <table>
              <tr class="b-statistics__table-row">
                <td><a class="b-link" href="http://ufcstats.com/fighter-details/nursulton">Nursulton</a></td>
                <td><a class="b-link">Ruziboev</a></td>
              </tr>
            </table>
            """,
            "lxml",
        )

    monkeypatch.setattr(fighter_lookup, "_get_soup", fake_get_soup)

    assert (
        fighter_lookup.search_fighter_url("Nursultan Ruziboev")
        == "http://ufcstats.com/fighter-details/nursulton"
    )
    assert requested_urls[0].endswith("char=r&page=all")


def test_search_fighter_url_tries_compound_surname_initial(monkeypatch):
    fighter_lookup.clear_cache()
    requested_urls = []

    def fake_get_soup(url):
        requested_urls.append(url)
        if "char=m" not in url:
            return BeautifulSoup("<table></table>", "lxml")
        return BeautifulSoup(
            """
            <table>
              <tr class="b-statistics__table-row">
                <td>
                  <a class="b-link"
                     href="http://ufcstats.com/fighter-details/442c9011034ae1fd">
                    Ian
                  </a>
                </td>
                <td><a class="b-link">Machado Garry</a></td>
              </tr>
            </table>
            """,
            "lxml",
        )

    monkeypatch.setattr(fighter_lookup, "_get_soup", fake_get_soup)

    assert (
        fighter_lookup.search_fighter_url("Ian Garry")
        == "http://ufcstats.com/fighter-details/442c9011034ae1fd"
    )
    assert [
        url.split("char=", 1)[1].split("&", 1)[0]
        for url in requested_urls
    ] == ["g", "i", "m"]


def test_search_fighter_url_keeps_raw_surname_initial_after_curated_alias(
    monkeypatch,
):
    fighter_lookup.clear_cache()
    requested_urls = []

    def fake_get_soup(url):
        requested_urls.append(url)
        if "char=f" not in url:
            return BeautifulSoup("<table></table>", "lxml")
        return BeautifulSoup(
            """
            <table>
              <tr class="b-statistics__table-row">
                <td>
                  <a class="b-link"
                     href="http://ufcstats.com/fighter-details/patricio-freire">
                    Patricio
                  </a>
                </td>
                <td><a class="b-link">Freire</a></td>
              </tr>
            </table>
            """,
            "lxml",
        )

    monkeypatch.setattr(fighter_lookup, "_get_soup", fake_get_soup)

    assert (
        fighter_lookup.search_fighter_url("Patricio Freire")
        == "http://ufcstats.com/fighter-details/patricio-freire"
    )
    assert [
        url.split("char=", 1)[1].split("&", 1)[0]
        for url in requested_urls
    ] == ["p", "f"]


def test_search_fighter_url_maps_patricio_ring_name_to_ufcstats_surname(
    monkeypatch,
):
    fighter_lookup.clear_cache()
    requested_urls = []

    def fake_get_soup(url):
        requested_urls.append(url)
        if "char=f" not in url:
            return BeautifulSoup("<table></table>", "lxml")
        return BeautifulSoup(
            """
            <table>
              <tr class="b-statistics__table-row">
                <td>
                  <a class="b-link"
                     href="http://ufcstats.com/fighter-details/patricio-freire">
                    Patricio
                  </a>
                </td>
                <td><a class="b-link">Freire</a></td>
              </tr>
            </table>
            """,
            "lxml",
        )

    monkeypatch.setattr(fighter_lookup, "_get_soup", fake_get_soup)

    assert (
        fighter_lookup.search_fighter_url("Patricio Pitbull")
        == "http://ufcstats.com/fighter-details/patricio-freire"
    )
    assert [
        url.split("char=", 1)[1].split("&", 1)[0]
        for url in requested_urls
    ] == ["p", "f"]


def test_search_fighter_url_canonicalizes_https_ufcstats_links(monkeypatch):
    fighter_lookup.clear_cache()

    def fake_get_soup(url):
        return BeautifulSoup(
            """
            <table>
              <tr class="b-statistics__table-row">
                <td><a class="b-link" href="https://ufcstats.com/fighter-details/derrick-lewis">Derrick</a></td>
                <td><a class="b-link">Lewis</a></td>
              </tr>
            </table>
            """,
            "lxml",
        )

    monkeypatch.setattr(fighter_lookup, "_get_soup", fake_get_soup)

    assert (
        fighter_lookup.search_fighter_url("Derrick Lewis")
        == "http://ufcstats.com/fighter-details/derrick-lewis"
    )
