import pandas as pd
import requests

from src.data import tennis_player_profiles as profiles


class FakeResponse:
    def __init__(self, payload=None, *, text: str = "", status_code: int = 200):
        self._payload = payload
        self.text = text
        self.status_code = status_code
        self.headers = {}

    def json(self):
        if self._payload is None:
            raise ValueError("No JSON payload")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def test_fetch_atp_player_profile_parses_official_json(monkeypatch):
    payload = {
        "FirstName": "Carlos",
        "LastName": "Alcaraz",
        "NatlId": "ESP",
        "Nationality": "Spain",
        "BirthDate": "2003-05-05T00:00:00",
        "BirthCity": "El Palmar, Murcia, Spain",
        "PlayHand": {"Id": "R", "Description": "Right-Handed"},
        "BackHand": {"Id": "2", "Description": "Two-Handed"},
        "HeightCm": 183,
        "WeightKg": 74,
        "ProYear": 2018,
        "Coach": "Samuel Lopez",
        "Active": {"Id": "A", "Description": "Active"},
        "SglRank": 1,
        "DblRank": 519,
        "ScRelativeUrlPlayerProfile": "/en/players/carlos-alcaraz/a0e2/overview",
    }

    monkeypatch.setattr(
        profiles,
        "_get_with_retries",
        lambda session, url, timeout=60, max_attempts=5: FakeResponse(payload),
    )

    row = profiles.fetch_atp_player_profile("a0e2")

    assert row["tour"] == "atp"
    assert row["player_id"] == "A0E2"
    assert row["full_name"] == "Carlos Alcaraz"
    assert row["birth_date"] == "2003-05-05"
    assert row["hand_code"] == "R"
    assert row["backhand_description"] == "Two-Handed"
    assert row["height_cm"] == 183
    assert row["current_singles_rank"] == 1
    assert row["fetch_status"] == "ok"
    assert row["profile_url"] == "https://www.atptour.com/en/players/carlos-alcaraz/a0e2/overview"


def test_fetch_atp_player_profile_allows_initial_surname_expected_name(monkeypatch):
    payload = {
        "FirstName": "Alex",
        "LastName": "de Minaur",
        "BirthDate": "1999-02-17T00:00:00",
        "PlayHand": {"Id": "R", "Description": "Right-Handed"},
    }

    monkeypatch.setattr(
        profiles,
        "_get_with_retries",
        lambda session, url, timeout=60, max_attempts=5: FakeResponse(payload),
    )

    row = profiles.fetch_atp_player_profile("dh58", expected_name="A. de Minaur")

    assert row["player_id"] == "DH58"
    assert row["full_name"] == "Alex de Minaur"
    assert row["birth_date"] == "1999-02-17"
    assert row["hand_code"] == "R"
    assert row["fetch_status"] == "ok"


def test_fetch_atp_player_profile_allows_initial_suffix_surname_expected_name(monkeypatch):
    payload = {
        "FirstName": "Juan Manuel",
        "LastName": "Cerundolo",
        "BirthDate": "2001-11-15T00:00:00",
        "PlayHand": {"Id": "L", "Description": "Left-Handed"},
    }

    monkeypatch.setattr(
        profiles,
        "_get_with_retries",
        lambda session, url, timeout=60, max_attempts=5: FakeResponse(payload),
    )

    row = profiles.fetch_atp_player_profile("c0c8", expected_name="J. Cerundolo")

    assert row["player_id"] == "C0C8"
    assert row["full_name"] == "Juan Manuel Cerundolo"
    assert row["birth_date"] == "2001-11-15"
    assert row["hand_code"] == "L"
    assert row["fetch_status"] == "ok"


def test_fetch_atp_player_profile_allows_junior_suffix_expected_name(monkeypatch):
    payload = {
        "FirstName": "Martin",
        "LastName": "Damm",
        "BirthDate": "2003-09-30T00:00:00",
        "PlayHand": {"Id": "L", "Description": "Left-Handed"},
    }

    monkeypatch.setattr(
        profiles,
        "_get_with_retries",
        lambda session, url, timeout=60, max_attempts=5: FakeResponse(payload),
    )

    row = profiles.fetch_atp_player_profile("d0dt", expected_name="M. Damm Jr")

    assert row["player_id"] == "D0DT"
    assert row["full_name"] == "Martin Damm"
    assert row["birth_date"] == "2003-09-30"
    assert row["hand_code"] == "L"
    assert row["fetch_status"] == "ok"


def test_fetch_atp_player_profile_handles_empty_official_payload(monkeypatch):
    class EmptyJsonResponse(FakeResponse):
        def json(self):
            return None

    monkeypatch.setattr(
        profiles,
        "_get_with_retries",
        lambda session, url, timeout=60, max_attempts=5: EmptyJsonResponse(),
    )

    row = profiles.fetch_atp_player_profile("zzzz")

    assert row["player_id"] == "ZZZZ"
    assert row["fetch_status"] == "not_found"
    assert "empty payload" in row["error"].lower()


def test_fetch_wta_player_profile_combines_api_and_profile_page(monkeypatch):
    api_payload = {
        "id": 332285,
        "firstName": "Iva",
        "lastName": "Jovic",
        "fullName": "Iva Jovic",
        "countryCode": "USA",
        "dateOfBirth": "2007-12-06",
    }
    html = """
    <html>
      <head>
        <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "Person",
          "name": "Iva Jovic",
          "nationality": {"@type": "Country", "name": "United States"},
          "birthDate": "2007-12-06",
          "birthPlace": {"@type": "Place", "address": {"addressLocality": "Torrance", "addressCountry": "CA"}},
          "url": "https://www.wtatennis.com/players/332285/iva-jovic/",
          "additionalProperty": [
            {"@type": "PropertyValue", "name": "WTA Singles Rank", "value": "17"},
            {"@type": "PropertyValue", "name": "WTA Doubles Rank", "value": "132"},
            {"@type": "PropertyValue", "name": "Plays", "value": "Right-Handed"}
          ]
        }
        </script>
      </head>
      <body>
        <section class="profile-bio">
          <ul>
            <li class="profile-bio__summary-item">Coached by Thomas Gutteridge</li>
          </ul>
          <div class="profile-bio__info-block">
            <h2 class="profile-bio__info-title">Plays</h2>
            <span class="profile-bio__info-content">Right-Handed</span>
          </div>
          <div class="profile-bio__info-block">
            <h2 class="profile-bio__info-title">Height</h2>
            <span class="profile-bio__info-content">5' 8" (1.73m)</span>
          </div>
          <div class="profile-bio__info-block">
            <h2 class="profile-bio__info-title">Birthplace</h2>
            <span class="profile-bio__info-content">Torrance, CA, USA</span>
          </div>
        </section>
      </body>
    </html>
    """

    def fake_get_with_retries(session, url, timeout=60, max_attempts=5):
        if url.endswith("/tennis/players/332285"):
            return FakeResponse(api_payload)
        if url.endswith("/tennis/players/332285/detailed"):
            return FakeResponse({"bio": {}})
        if url.endswith("/players/332285/iva-jovic/"):
            return FakeResponse(text=html)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(profiles, "_get_with_retries", fake_get_with_retries)

    row = profiles.fetch_wta_player_profile(332285)

    assert row["tour"] == "wta"
    assert row["player_id"] == "332285"
    assert row["full_name"] == "Iva Jovic"
    assert row["birth_date"] == "2007-12-06"
    assert row["birth_place"] == "Torrance, CA, USA"
    assert row["country_name"] == "United States"
    assert row["hand_code"] == "R"
    assert row["height_cm"] == 173
    assert row["coach"] == "Thomas Gutteridge"
    assert row["current_singles_rank"] == 17
    assert row["fetch_status"] == "ok"


def test_fetch_wta_player_profile_uses_detailed_bio_fallbacks(monkeypatch):
    api_payload = {
        "id": 321454,
        "firstName": "Greet",
        "lastName": "Minnen",
        "fullName": "Greet Minnen",
        "countryCode": "BEL",
        "dateOfBirth": "1997-08-14",
    }
    detailed_payload = {
        "bio": {
            "countryname": "Belgium",
            "dateofbirth": "1997-08-14T00:00:00+00:00",
            "birthcity": "Turnhout, Belgium",
            "height": "5' 9\"",
            "playhand": "Right-Handed",
            "sglrank": 156,
            "dblrank": 981,
        }
    }
    html = """
    <html>
      <head>
        <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "Person",
          "name": "Greet Minnen",
          "nationality": {"@type": "Country", "name": "Belgium"},
          "birthDate": "1997-08-14",
          "birthPlace": {"@type": "Place", "address": {"addressLocality": "Turnhout", "addressCountry": "Belgium"}},
          "url": "https://www.wtatennis.com/players/321454/greet-minnen/"
        }
        </script>
      </head>
      <body>
        <section class="profile-bio">
          <div class="profile-bio__info-block">
            <h2 class="profile-bio__info-title">Plays</h2>
            <span class="profile-bio__info-content">Right-Handed</span>
          </div>
          <div class="profile-bio__info-block">
            <h2 class="profile-bio__info-title">Height</h2>
            <span class="profile-bio__info-content">-</span>
          </div>
        </section>
      </body>
    </html>
    """

    def fake_get_with_retries(session, url, timeout=60, max_attempts=5, **kwargs):
        if url.endswith("/tennis/players/321454"):
            return FakeResponse(api_payload)
        if url.endswith("/tennis/players/321454/detailed"):
            return FakeResponse(detailed_payload)
        if url.endswith("/players/321454/greet-minnen/"):
            return FakeResponse(text=html)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(profiles, "_get_with_retries", fake_get_with_retries)

    row = profiles.fetch_wta_player_profile(321454)

    assert row["birth_place"] == "Turnhout, Belgium"
    assert row["country_name"] == "Belgium"
    assert row["hand_code"] == "R"
    assert row["height_cm"] == 175
    assert row["current_singles_rank"] == 156
    assert row["current_doubles_rank"] == 981


def test_fetch_wta_player_profile_rejects_name_mismatch(monkeypatch):
    api_payload = {
        "id": 332285,
        "firstName": "Wrong",
        "lastName": "Player",
        "fullName": "Wrong Player",
        "countryCode": "USA",
        "dateOfBirth": "2007-12-06",
    }

    monkeypatch.setattr(
        profiles,
        "_get_with_retries",
        lambda session, url, timeout=60, max_attempts=5: FakeResponse(api_payload),
    )

    row = profiles.fetch_wta_player_profile(332285, expected_name="Iva Jovic")

    assert row["fetch_status"] == "name_mismatch"
    assert row["full_name"] == "Wrong Player"


def test_parse_wta_height_cm_accepts_feet_inches_and_compact_values():
    assert profiles._parse_wta_height_cm("5' 9\"") == 175
    assert profiles._parse_wta_height_cm("1.70m") == 170
    assert profiles._parse_wta_height_cm("175.5") == 176


def test_tennisexplorer_fetch_accepts_birthdate_verified_name_variant(monkeypatch):
    payload = {"links": [{"url": "minnen", "name": "Minnen, Greetje (BEL)"}]}
    page_text = """
    Minnen Greetje - profile
    Height / Weight: 175 cm / 68 kg
    Age: 28 (14. 8. 1997)
    Plays: right
    """

    class FakeSession:
        headers = {}

        def get(self, url, **kwargs):
            if url == profiles.TENNIS_EXPLORER_SEARCH_URL:
                return FakeResponse(payload)
            raise AssertionError(f"Unexpected direct session URL: {url}")

    def fake_request_session(session=None):
        return session or FakeSession()

    def fake_get_with_retries(session, url, timeout=60, max_attempts=5):
        if url.endswith("/player/minnen/"):
            return FakeResponse(text=page_text)
        raise AssertionError(f"Unexpected profile URL: {url}")

    monkeypatch.setattr(profiles, "_request_session", fake_request_session)
    monkeypatch.setattr(profiles, "_get_with_retries", fake_get_with_retries)

    row = profiles.fetch_tennisexplorer_player_profile(
        "wta",
        "321454",
        "Greet Minnen",
        known_birth_date="1997-08-14",
    )

    assert row["fetch_status"] == "ok"
    assert row["full_name"] == "Greetje Minnen"
    assert row["birth_date"] == "1997-08-14"
    assert row["height_cm"] == 175
    assert row["hand_code"] == "R"
    assert "accepted_via_birth_date_match" in row["source_notes"]


def test_tennisexplorer_fetch_does_not_take_opponent_height_from_match_detail(monkeypatch):
    payload = {"links": [{"url": "costoulas", "name": "Costoulas, Sofia (BEL)"}]}
    profile_html = """
    <html>
      <body>
        <div>Costoulas Sofia - profile</div>
        <a href="/match-detail/?id=3075894">Recent match</a>
      </body>
    </html>
    """
    detail_html = """
    <table class="result gDetail noMgB">
      <thead>
        <tr>
          <th class="plName" colspan="2"><a href="/player/costoulas">Costoulas Sofia</a></th>
          <td class="gScore">2 : 1</td>
          <th class="plName" colspan="2"><a href="/player/aksu">Aksu Ayla</a></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="tr">2. 4. 2005</td>
          <th>Birthdate</th>
          <td class="tl">15. 7. 1996</td>
        </tr>
        <tr>
          <td class="tr">-</td>
          <th>Height</th>
          <td class="tl">170 cm</td>
        </tr>
        <tr>
          <td class="tr">right</td>
          <th>Plays</th>
          <td class="tl">right</td>
        </tr>
      </tbody>
    </table>
    """

    class FakeSession:
        headers = {}

        def get(self, url, **kwargs):
            if url == profiles.TENNIS_EXPLORER_SEARCH_URL:
                return FakeResponse(payload)
            raise AssertionError(f"Unexpected direct session URL: {url}")

    def fake_request_session(session=None):
        return session or FakeSession()

    def fake_get_with_retries(session, url, timeout=60, max_attempts=5):
        if url.endswith("/player/costoulas/"):
            return FakeResponse(text=profile_html)
        if "match-detail/?id=3075894" in url:
            return FakeResponse(text=detail_html)
        raise AssertionError(f"Unexpected profile URL: {url}")

    monkeypatch.setattr(profiles, "_request_session", fake_request_session)
    monkeypatch.setattr(profiles, "_get_with_retries", fake_get_with_retries)

    row = profiles.fetch_tennisexplorer_player_profile(
        "wta",
        "330451",
        "Sofia Costoulas",
        known_birth_date="2005-04-02",
    )

    assert row["fetch_status"] == "ok"
    assert row["birth_date"] == "2005-04-02"
    assert row["height_cm"] is None
    assert row["hand_code"] == "R"
    assert "match_detail_pages_checked=1" in row["source_notes"]


def test_tennisexplorer_fetch_reads_match_detail_height_when_target_on_left(monkeypatch):
    payload = {"links": [{"url": "salkova", "name": "Salkova, Dominika (CZE)"}]}
    profile_html = """
    <html>
      <body>
        <div>Dominika Salkova - profile</div>
        <a href="/match-detail/?id=4001">Recent match</a>
      </body>
    </html>
    """
    detail_html = """
    <table class="result gDetail noMgB">
      <thead>
        <tr>
          <th class="plName" colspan="2"><a href="/player/salkova">Salkova Dominika</a></th>
          <td class="gScore">2 : 0</td>
          <th class="plName" colspan="2"><a href="/player/other-player">Other Player</a></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="tr">28. 6. 2004</td>
          <th>Birthdate</th>
          <td class="tl">1. 1. 1999</td>
        </tr>
        <tr>
          <td class="tr">174 cm</td>
          <th>Height</th>
          <td class="tl">-</td>
        </tr>
        <tr>
          <td class="tr">right</td>
          <th>Plays</th>
          <td class="tl">left</td>
        </tr>
      </tbody>
    </table>
    """

    class FakeSession:
        headers = {}

        def get(self, url, **kwargs):
            if url == profiles.TENNIS_EXPLORER_SEARCH_URL:
                return FakeResponse(payload)
            raise AssertionError(f"Unexpected direct session URL: {url}")

    def fake_request_session(session=None):
        return session or FakeSession()

    def fake_get_with_retries(session, url, timeout=60, max_attempts=5):
        if url.endswith("/player/salkova/"):
            return FakeResponse(text=profile_html)
        if "match-detail/?id=4001" in url:
            return FakeResponse(text=detail_html)
        raise AssertionError(f"Unexpected profile URL: {url}")

    monkeypatch.setattr(profiles, "_request_session", fake_request_session)
    monkeypatch.setattr(profiles, "_get_with_retries", fake_get_with_retries)

    row = profiles.fetch_tennisexplorer_player_profile(
        "wta",
        "330149",
        "Dominika Salkova",
        known_birth_date="2004-06-28",
    )

    assert row["fetch_status"] == "ok"
    assert row["birth_date"] == "2004-06-28"
    assert row["height_cm"] == 174
    assert row["hand_code"] == "R"
    assert row["source_page_url"].endswith("match-detail/?id=4001")


def test_tennisexplorer_fetch_reads_match_detail_height_when_target_on_right(monkeypatch):
    payload = {
        "links": [
            {
                "url": "rakotomanga-rajaonah-3bf4e",
                "name": "Rakotomanga Rajaonah, Tiantsoa Sarah (FRA)",
            }
        ]
    }
    profile_html = """
    <html>
      <body>
        <div>Rakotomanga Rajaonah Tiantsoa Sarah - profile</div>
        <a href="/match-detail/?id=4002">Recent match</a>
      </body>
    </html>
    """
    detail_html = """
    <table class="result gDetail noMgB">
      <thead>
        <tr>
          <th class="plName" colspan="2"><a href="/player/sabalenka">Sabalenka Aryna</a></th>
          <td class="gScore">2 : 1</td>
          <th class="plName" colspan="2"><a href="/player/rakotomanga-rajaonah-3bf4e">Rakotomanga Rajaonah Tiantsoa Sarah</a></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="tr">5. 5. 1998</td>
          <th>Birthdate</th>
          <td class="tl">15. 12. 2005</td>
        </tr>
        <tr>
          <td class="tr">182 cm</td>
          <th>Height</th>
          <td class="tl">182 cm</td>
        </tr>
        <tr>
          <td class="tr">right</td>
          <th>Plays</th>
          <td class="tl">left</td>
        </tr>
      </tbody>
    </table>
    """

    class FakeSession:
        headers = {}

        def get(self, url, **kwargs):
            if url == profiles.TENNIS_EXPLORER_SEARCH_URL:
                return FakeResponse(payload)
            raise AssertionError(f"Unexpected direct session URL: {url}")

    def fake_request_session(session=None):
        return session or FakeSession()

    def fake_get_with_retries(session, url, timeout=60, max_attempts=5):
        if url.endswith("/player/rakotomanga-rajaonah-3bf4e/"):
            return FakeResponse(text=profile_html)
        if "match-detail/?id=4002" in url:
            return FakeResponse(text=detail_html)
        raise AssertionError(f"Unexpected profile URL: {url}")

    monkeypatch.setattr(profiles, "_request_session", fake_request_session)
    monkeypatch.setattr(profiles, "_get_with_retries", fake_get_with_retries)

    row = profiles.fetch_tennisexplorer_player_profile(
        "wta",
        "330970",
        "Tiantsoa Sarah Rakotomanga Rajaonah",
        known_birth_date="2005-12-15",
    )

    assert row["fetch_status"] == "ok"
    assert row["birth_date"] == "2005-12-15"
    assert row["height_cm"] == 182
    assert row["hand_code"] == "L"
    assert row["source_page_url"].endswith("match-detail/?id=4002")


def test_tennisexplorer_fetch_rejects_match_detail_birthdate_mismatch(monkeypatch):
    payload = {"links": [{"url": "todoni", "name": "Todoni, Anca Alexia (ROU)"}]}
    profile_html = """
    <html>
      <body>
        <div>Anca Alexia Todoni - profile</div>
        <a href="/match-detail/?id=4003">Recent match</a>
      </body>
    </html>
    """
    detail_html = """
    <table class="result gDetail noMgB">
      <thead>
        <tr>
          <th class="plName" colspan="2"><a href="/player/todoni">Todoni Anca Alexia</a></th>
          <td class="gScore">2 : 0</td>
          <th class="plName" colspan="2"><a href="/player/other-player">Other Player</a></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="tr">1. 1. 2003</td>
          <th>Birthdate</th>
          <td class="tl">2. 2. 1999</td>
        </tr>
        <tr>
          <td class="tr">182 cm</td>
          <th>Height</th>
          <td class="tl">-</td>
        </tr>
      </tbody>
    </table>
    """

    class FakeSession:
        headers = {}

        def get(self, url, **kwargs):
            if url == profiles.TENNIS_EXPLORER_SEARCH_URL:
                return FakeResponse(payload)
            raise AssertionError(f"Unexpected direct session URL: {url}")

    def fake_request_session(session=None):
        return session or FakeSession()

    def fake_get_with_retries(session, url, timeout=60, max_attempts=5):
        if url.endswith("/player/todoni/"):
            return FakeResponse(text=profile_html)
        if "match-detail/?id=4003" in url:
            return FakeResponse(text=detail_html)
        raise AssertionError(f"Unexpected profile URL: {url}")

    monkeypatch.setattr(profiles, "_request_session", fake_request_session)
    monkeypatch.setattr(profiles, "_get_with_retries", fake_get_with_retries)

    row = profiles.fetch_tennisexplorer_player_profile(
        "wta",
        "330442",
        "Anca Alexia Todoni",
        known_birth_date="2004-10-10",
    )

    assert row["fetch_status"] == "not_found"
    assert row["height_cm"] is None
    assert row["source_page_url"].endswith("/player/todoni/")


def test_wikipedia_height_parser_handles_metric_infobox_values():
    soup = profiles.BeautifulSoup(
        """
        <table class="infobox">
          <tr><th>Height</th><td>1.72 m (5 ft 8 in)</td></tr>
          <tr><th>Plays</th><td>Right-handed</td></tr>
        </table>
        """,
        "lxml",
    )

    assert profiles._wikipedia_infobox_height_cm(soup) == 172


def test_collect_tennis_player_profile_targets_only_includes_missing_static_fields():
    matches_df = pd.DataFrame(
        [
            {
                "tour": "atp",
                "event_date": "2025-01-01",
                "player_a_id": "A1",
                "player_a": "Player A",
                "player_a_age": pd.NA,
                "player_a_hand": pd.NA,
                "player_a_height_cm": pd.NA,
                "player_b_id": "B1",
                "player_b": "Player B",
                "player_b_age": 25.0,
                "player_b_hand": "R",
                "player_b_height_cm": 188.0,
            },
            {
                "tour": "atp",
                "event_date": "2025-01-02",
                "player_a_id": "A1",
                "player_a": "Player A Long",
                "player_a_age": pd.NA,
                "player_a_hand": pd.NA,
                "player_a_height_cm": 190.0,
                "player_b_id": "C1",
                "player_b": "Player C",
                "player_b_age": 24.0,
                "player_b_hand": "",
                "player_b_height_cm": 185.0,
            },
        ]
    )

    targets = profiles.collect_tennis_player_profile_targets(matches_df, missing_only=True)

    assert targets["player_id"].tolist() == ["A1", "C1"]
    assert targets.loc[0, "player_name"] == "Player A Long"
    assert targets.loc[0, "affected_rows"] == 2
    assert targets.loc[0, "latest_event_date"] == "2025-01-02"
    assert targets.loc[0, "missing_fields"] == "age,hand,height_cm"
    assert targets.loc[1, "missing_fields"] == "hand"


def test_collect_tennis_player_profile_targets_can_limit_to_official_window():
    matches_df = pd.DataFrame(
        [
            {
                "tour": "wta",
                "event_date": "2024-12-31",
                "player_a_id": "111",
                "player_a": "Older Player",
                "player_a_age": pd.NA,
                "player_a_hand": pd.NA,
                "player_a_height_cm": pd.NA,
                "player_b_id": "222",
                "player_b": "Other Older Player",
                "player_b_age": 25.0,
                "player_b_hand": "R",
                "player_b_height_cm": 175.0,
            },
            {
                "tour": "wta",
                "event_date": "2025-01-01",
                "player_a_id": "333",
                "player_a": "Official Window Player",
                "player_a_age": pd.NA,
                "player_a_hand": pd.NA,
                "player_a_height_cm": pd.NA,
                "player_b_id": "444",
                "player_b": "Other Player",
                "player_b_age": 25.0,
                "player_b_hand": "R",
                "player_b_height_cm": 175.0,
            },
        ]
    )

    targets = profiles.collect_tennis_player_profile_targets(
        matches_df,
        missing_only=True,
        official_window_only=True,
        official_start_year=2025,
    )

    assert targets["player_id"].tolist() == ["333"]


def test_collect_live_tennis_player_profile_seed_targets_detects_new_live_player():
    live_df = pd.DataFrame(
        [
            {
                "tour": "wta",
                "commence_time": "2026-03-22T17:00:00Z",
                "fighter_a": "New Phenom",
                "fighter_b": "Known Player",
            }
        ]
    )
    profiles_df = pd.DataFrame(
        [
            {
                "tour": "wta",
                "player_id": "123",
                "full_name": "Known Player",
                "birth_date": "2000-01-01",
                "hand_code": "R",
                "height_cm": 180,
                "fetch_status": "ok",
                "source_kind": "manual_verified_wta_official_alias",
            }
        ]
    )

    targets = profiles.collect_live_tennis_player_profile_seed_targets(live_df, profiles_df=profiles_df)

    assert targets["player_name"].tolist() == ["New Phenom"]
    assert targets.loc[0, "player_id"] == ""
    assert targets.loc[0, "latest_event_date"] == "2026-03-22"
    assert targets.loc[0, "missing_fields"] == "age,hand,height_cm"


def test_should_retry_not_found_profile_after_30_days():
    existing_row = {
        "fetch_status": "not_found",
        "fetched_at_utc": "2026-01-01T00:00:00Z",
        "observed_rows": 1,
    }
    target = {"affected_rows": 1, "latest_event_date": "2026-03-01"}

    should_retry = profiles._should_retry_cached_profile_attempt(
        existing_row,
        target=target,
        now_utc=pd.Timestamp("2026-02-05T00:00:00Z"),
    )

    assert should_retry is True


def test_should_retry_not_found_profile_after_three_new_observations():
    existing_row = {
        "fetch_status": "not_found",
        "fetched_at_utc": "2026-03-01T00:00:00Z",
        "observed_rows": 2,
    }
    target = {"affected_rows": 5, "latest_event_date": "2026-03-22"}

    should_retry = profiles._should_retry_cached_profile_attempt(
        existing_row,
        target=target,
        now_utc=pd.Timestamp("2026-03-10T00:00:00Z"),
    )

    assert should_retry is True


def test_should_not_retry_recent_not_found_profile_without_new_observations():
    existing_row = {
        "fetch_status": "not_found",
        "fetched_at_utc": "2026-03-01T00:00:00Z",
        "observed_rows": 4,
    }
    target = {"affected_rows": 5, "latest_event_date": "2026-03-05"}

    should_retry = profiles._should_retry_cached_profile_attempt(
        existing_row,
        target=target,
        now_utc=pd.Timestamp("2026-03-10T00:00:00Z"),
    )

    assert should_retry is False


def test_remaining_tennis_player_profile_targets_excludes_resolved_profiles():
    targets = pd.DataFrame(
        [
            {"tour": "atp", "player_id": "A1", "player_name": "Resolved Player"},
            {"tour": "wta", "player_id": "222", "player_name": "Unresolved Player"},
        ]
    )
    profiles_df = pd.DataFrame(
        [
            {
                "tour": "atp",
                "player_id": "A1",
                "fetch_status": "ok",
                "fetched_at_utc": "2026-03-21T17:00:00Z",
            }
        ]
    )

    remaining = profiles.remaining_tennis_player_profile_targets(targets, profiles_df=profiles_df)

    assert remaining["player_id"].tolist() == ["222"]


def test_enrich_tennis_matches_with_player_profiles_fills_only_missing_fields():
    matches_df = pd.DataFrame(
        [
            {
                "tour": "wta",
                "event_date": "2026-02-16",
                "player_a_id": 332285,
                "player_b_id": 445566,
                "player_a_age": float("nan"),
                "player_b_age": 27.4,
                "player_a_hand": pd.NA,
                "player_b_hand": "L",
                "player_a_height_cm": float("nan"),
                "player_b_height_cm": 181.0,
                "player_a_rank": pd.NA,
                "player_b_rank": pd.NA,
            }
        ]
    )
    profiles_df = pd.DataFrame(
        [
            {
                "tour": "wta",
                "player_id": "332285",
                "birth_date": "2007-12-06",
                "birth_place": "Torrance, CA, USA",
                "country_code": "USA",
                "country_name": "United States",
                "hand_code": "R",
                "hand_description": "Right-Handed",
                "backhand_code": "",
                "backhand_description": "",
                "height_cm": 173,
                "weight_kg": pd.NA,
                "fetch_status": "ok",
                "fetched_at_utc": "2026-03-21T17:00:00Z",
                "current_singles_rank": 17,
            },
            {
                "tour": "wta",
                "player_id": "445566",
                "birth_date": "1998-01-01",
                "birth_place": "Paris, France",
                "country_code": "FRA",
                "country_name": "France",
                "hand_code": "R",
                "hand_description": "Right-Handed",
                "backhand_code": "",
                "backhand_description": "",
                "height_cm": 179,
                "weight_kg": pd.NA,
                "fetch_status": "ok",
                "fetched_at_utc": "2026-03-21T17:00:00Z",
                "current_singles_rank": 25,
            },
        ]
    )

    enriched = profiles.enrich_tennis_matches_with_player_profiles(matches_df, profiles_df=profiles_df)

    assert enriched.loc[0, "player_a_hand"] == "R"
    assert enriched.loc[0, "player_a_height_cm"] == 173
    assert float(enriched.loc[0, "player_a_age"]) > 18.0
    assert enriched.loc[0, "player_b_hand"] == "L"
    assert enriched.loc[0, "player_b_height_cm"] == 181.0
    assert pd.isna(enriched.loc[0, "player_a_rank"])
    assert pd.isna(enriched.loc[0, "player_b_rank"])
    assert "current_singles_rank" not in enriched.columns


def test_load_tennis_player_profiles_includes_supplemental_rows(monkeypatch):
    supplement_df = pd.DataFrame(
        [
            {
                "tour": "wta",
                "player_id": "904523",
                "full_name": "Alyssa Ahn",
                "birth_date": "2006-12-27",
                "hand_code": "R",
                "fetch_status": "ok",
                "fetched_at_utc": "2026-03-21T17:00:00Z",
                "source_kind": "manual_verified_wta_search_snippet",
            }
        ]
    )

    monkeypatch.setattr(profiles, "_load_profile_cache", lambda tour: pd.DataFrame(columns=profiles.PLAYER_PROFILE_COLUMNS))
    monkeypatch.setattr(profiles, "_load_profile_supplement", lambda: supplement_df.copy())

    combined = profiles.load_tennis_player_profiles()

    assert len(combined) == 1
    assert combined.loc[0, "player_id"] == "904523"
    assert combined.loc[0, "birth_date"] == "2006-12-27"
    assert combined.loc[0, "hand_code"] == "R"


def test_enrich_tennis_matches_with_player_profiles_aliases_same_name_across_ids():
    matches_df = pd.DataFrame(
        [
            {
                "tour": "wta",
                "event_date": "2026-02-16",
                "player_a": "Claire Liu",
                "player_a_id": "179300",
                "player_a_age": pd.NA,
                "player_a_hand": pd.NA,
                "player_a_height_cm": pd.NA,
                "player_b": "Other Player",
                "player_b_id": "445566",
                "player_b_age": 25.0,
                "player_b_hand": "L",
                "player_b_height_cm": 181.0,
            }
        ]
    )
    profiles_df = pd.DataFrame(
        [
            {
                "tour": "wta",
                "player_id": "323966",
                "full_name": "Claire Liu",
                "birth_date": "2000-05-25",
                "hand_code": "R",
                "hand_description": "Right-Handed",
                "height_cm": 170,
                "fetch_status": "ok",
                "fetched_at_utc": "2026-03-21T17:00:00Z",
            }
        ]
    )

    enriched = profiles.enrich_tennis_matches_with_player_profiles(matches_df, profiles_df=profiles_df)

    assert float(enriched.loc[0, "player_a_age"]) > 25.0
    assert enriched.loc[0, "player_a_hand"] == "R"
    assert enriched.loc[0, "player_a_height_cm"] == 170


def test_enrich_tennis_matches_with_player_profiles_aliases_initial_name_and_historical_static_backfill():
    matches_df = pd.DataFrame(
        [
            {
                "tour": "atp",
                "event_date": "2026-01-18",
                "player_a": "R. Jodar",
                "player_a_id": "SR:COMPETITOR:972327",
                "player_a_age": pd.NA,
                "player_a_hand": pd.NA,
                "player_a_height_cm": pd.NA,
                "player_b": "Other Player",
                "player_b_id": "OTHER1",
                "player_b_age": 26.0,
                "player_b_hand": "R",
                "player_b_height_cm": 188.0,
            },
            {
                "tour": "wta",
                "event_date": "2024-03-01",
                "player_a": "Katherine Sebov",
                "player_a_id": "OLD1",
                "player_a_age": 25.0,
                "player_a_hand": "R",
                "player_a_height_cm": 173.0,
                "player_b": "Someone Else",
                "player_b_id": "OLD2",
                "player_b_age": 24.0,
                "player_b_hand": "L",
                "player_b_height_cm": 175.0,
            },
            {
                "tour": "wta",
                "event_date": "2026-02-01",
                "player_a": "Katherine Sebov",
                "player_a_id": "NEW1",
                "player_a_age": pd.NA,
                "player_a_hand": pd.NA,
                "player_a_height_cm": pd.NA,
                "player_b": "Another Player",
                "player_b_id": "NEW2",
                "player_b_age": 21.0,
                "player_b_hand": "R",
                "player_b_height_cm": 176.0,
            },
        ]
    )
    profiles_df = pd.DataFrame(
        [
            {
                "tour": "atp",
                "player_id": "A0J1",
                "full_name": "Rafael Jodar",
                "birth_date": "2006-09-17",
                "hand_code": "R",
                "hand_description": "Right-Handed",
                "height_cm": 191,
                "fetch_status": "ok",
                "fetched_at_utc": "2026-03-21T17:00:00Z",
            }
        ]
    )

    enriched = profiles.enrich_tennis_matches_with_player_profiles(matches_df, profiles_df=profiles_df)

    assert float(enriched.loc[0, "player_a_age"]) > 19.0
    assert enriched.loc[0, "player_a_hand"] == "R"
    assert enriched.loc[0, "player_a_height_cm"] == 191
    assert enriched.loc[2, "player_a_hand"] == "R"
    assert enriched.loc[2, "player_a_height_cm"] == 173.0


def test_enrich_tennis_matches_with_player_profiles_prefers_official_values_over_conflicting_secondary():
    matches_df = pd.DataFrame(
        [
            {
                "tour": "wta",
                "event_date": "2025-01-05",
                "player_a": "Aoi Ito",
                "player_a_id": "329009",
                "player_a_age": pd.NA,
                "player_a_hand": pd.NA,
                "player_a_height_cm": pd.NA,
                "player_a_birth_date": pd.NA,
                "player_b": "Other Player",
                "player_b_id": "OTHER1",
                "player_b_age": 24.0,
                "player_b_hand": "R",
                "player_b_height_cm": 176.0,
                "player_b_birth_date": "2001-01-01",
            }
        ]
    )
    profiles_df = pd.DataFrame(
        [
            {
                "tour": "wta",
                "player_id": "329009",
                "full_name": "Aoi Ito",
                "birth_date": "2004-05-21",
                "hand_code": "R",
                "hand_description": "Right-Handed",
                "source_kind": "wta_official_api_plus_profile_page",
                "fetch_status": "ok",
                "fetched_at_utc": "2026-03-21T17:00:00Z",
            },
            {
                "tour": "wta",
                "player_id": "329009",
                "full_name": "Aoi Ito",
                "birth_date": "2004-05-01",
                "hand_code": "L",
                "hand_description": "Left-Handed",
                "source_kind": "tennisexplorer_profile",
                "fetch_status": "ok",
                "fetched_at_utc": "2026-03-21T18:00:00Z",
            },
        ]
    )

    enriched = profiles.enrich_tennis_matches_with_player_profiles(matches_df, profiles_df=profiles_df)

    assert enriched.loc[0, "player_a_birth_date"] == "2004-05-21"
    assert enriched.loc[0, "player_a_hand"] == "R"
    assert float(enriched.loc[0, "player_a_age"]) > 20.0


def test_build_profile_id_lookup_keeps_conflicting_same_priority_blank():
    profiles_df = pd.DataFrame(
        [
            {
                "tour": "wta",
                "player_id": "123",
                "full_name": "Conflict Player",
                "hand_code": "L",
                "source_kind": "manual_verified_public_consensus",
            },
            {
                "tour": "wta",
                "player_id": "123",
                "full_name": "Conflict Player",
                "hand_code": "R",
                "source_kind": "wikipedia_page_plus_wikidata",
            },
        ]
    )

    lookup = profiles._build_profile_id_lookup(profiles_df)

    assert pd.isna(lookup.loc[0, "hand_code"]) or lookup.loc[0, "hand_code"] in ["", None]


def test_build_profile_id_lookup_prefers_manual_official_event_entry_over_secondary_sources():
    profiles_df = pd.DataFrame(
        [
            {
                "tour": "wta",
                "player_id": "326929",
                "full_name": "Mananchaya Sawangkaew",
                "birth_date": "2002-07-10",
                "hand_code": "R",
                "height_cm": 162,
                "source_kind": "manual_verified_official_event_entry",
            },
            {
                "tour": "wta",
                "player_id": "326929",
                "full_name": "Mananchaya Sawangkaew",
                "birth_date": "2002-07-01",
                "hand_code": "L",
                "height_cm": 160,
                "source_kind": "wikipedia_page_plus_wikidata",
            },
        ]
    )

    lookup = profiles._build_profile_id_lookup(profiles_df)

    assert lookup.loc[0, "birth_date"] == "2002-07-10"
    assert lookup.loc[0, "hand_code"] == "R"
    assert lookup.loc[0, "height_cm"] == 162


def test_build_profile_id_lookup_prefers_manual_tennisboard_height_over_secondary_sources():
    profiles_df = pd.DataFrame(
        [
            {
                "tour": "wta",
                "player_id": "323119",
                "full_name": "Miriam Bulgaru",
                "height_cm": 169,
                "source_kind": "manual_verified_tennisboard_player_page",
            },
            {
                "tour": "wta",
                "player_id": "323119",
                "full_name": "Miriam Bulgaru",
                "height_cm": 168,
                "source_kind": "wikipedia_page_plus_wikidata",
            },
        ]
    )

    lookup = profiles._build_profile_id_lookup(profiles_df)

    assert lookup.loc[0, "height_cm"] == 169
