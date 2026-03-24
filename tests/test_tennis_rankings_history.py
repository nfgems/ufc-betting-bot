import pandas as pd

from src.data import tennis_rankings_history as rankings


class FakeResponse:
    def __init__(self, payload=None, *, text: str = "", status_code: int = 200, url: str = ""):
        self._payload = payload
        self.text = text
        self.status_code = status_code
        self.headers = {}
        self.url = url or "https://example.test"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code} error")


def test_fetch_atp_rankings_snapshot_parses_official_html(monkeypatch):
    html = """
    <html>
      <body>
        <table class="mega-table">
          <tbody>
            <tr class="lower-row">
              <td class="rank">1</td>
              <td class="player">
                <a href="/en/players/carlos-alcaraz/a0e2/overview"><span class="lastName">C. Alcaraz</span></a>
              </td>
              <td class="points"><a>13,550</a></td>
            </tr>
            <tr class="lower-row">
              <td class="rank">2</td>
              <td class="player">
                <a href="/en/players/jannik-sinner/s0ag/overview"><span class="lastName">J. Sinner</span></a>
              </td>
              <td class="points"><a>11,400</a></td>
            </tr>
          </tbody>
        </table>
      </body>
    </html>
    """

    def fake_get_with_retries(session, url, timeout=60, max_attempts=5, **kwargs):
        return FakeResponse(text=html, url="https://www.atptour.com/en/rankings/singles?dateWeek=2026-03-16")

    monkeypatch.setattr(rankings, "_get_with_retries", fake_get_with_retries)

    frame = rankings.fetch_atp_rankings_snapshot("2026-03-16")

    assert frame["player_id"].tolist() == ["A0E2", "S0AG"]
    assert frame["display_name"].tolist() == ["C. Alcaraz", "J. Sinner"]
    assert frame["full_name"].tolist() == ["Carlos Alcaraz", "Jannik Sinner"]
    assert frame["rank"].tolist() == [1, 2]
    assert frame["rank_points"].tolist() == [13550, 11400]


def test_fetch_atp_rankings_snapshot_falls_back_to_chunked_ranges(monkeypatch):
    error_html = """
    <html>
      <body>
        <div class="atp_rankings-all">
            <!-- Error while rendering the view [Singles Rankings] -->
        </div>
      </body>
    </html>
    """
    chunk_html = """
    <html>
      <body>
        <table class="mega-table">
          <tbody>
            <tr class="lower-row">
              <td class="rank">45</td>
              <td class="player">
                <a href="/en/players/nuno-borges/bt72/overview"><span class="lastName">N. Borges</span></a>
              </td>
              <td class="points"><a>1,120T</a></td>
            </tr>
            <tr class="lower-row">
              <td class="rank">45</td>
              <td class="player">
                <a href="/en/players/nuno-borges/bt72/overview"><span class="lastName">Nuno Borges</span></a>
              </td>
              <td class="points"><a>1,120T</a></td>
            </tr>
            <tr class="lower-row">
              <td class="rank">65</td>
              <td class="player">
                <a href="/en/players/damir-dzumhur/d923/overview"><span class="lastName">D. Dzumhur</span></a>
              </td>
              <td class="points"><a>837</a></td>
            </tr>
          </tbody>
        </table>
      </body>
    </html>
    """
    seen_ranges = []

    def fake_get_with_retries(session, url, timeout=60, max_attempts=5, **kwargs):
        params = kwargs.get("params") or {}
        rank_range = params.get("rankRange")
        seen_ranges.append(rank_range)
        if rank_range == "0-5000":
            return FakeResponse(text=error_html, url="https://www.atptour.com/en/rankings/singles?dateWeek=2026-01-05&rankRange=0-5000")
        return FakeResponse(text=chunk_html, url=f"https://www.atptour.com/en/rankings/singles?dateWeek=2026-01-05&rankRange={rank_range}")

    monkeypatch.setattr(rankings, "_get_with_retries", fake_get_with_retries)
    monkeypatch.setattr(rankings.time, "sleep", lambda *_args, **_kwargs: None)

    frame = rankings.fetch_atp_rankings_snapshot("2026-01-05")

    assert seen_ranges[0] == "0-5000"
    assert frame["player_id"].tolist() == ["BT72", "D923"]
    assert frame["rank"].tolist() == [45, 65]
    assert frame["rank_points"].tolist() == [1120, 837]


def test_resolve_wta_snapshot_date_uses_ranked_at_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(rankings, "TENNIS_RAW_DIR", tmp_path)

    def fake_get_with_retries(session, url, timeout=60, max_attempts=5, **kwargs):
        return FakeResponse(
            payload=[
                {
                    "player": {"id": 320760, "fullName": "Aryna Sabalenka"},
                    "ranking": 1,
                    "points": 9606,
                    "rankedAt": "2025-03-17T00:00:00Z",
                }
            ]
        )

    monkeypatch.setattr(rankings, "_get_with_retries", fake_get_with_retries)

    resolved = rankings.resolve_wta_snapshot_date("2025-03-18")

    assert resolved == "2025-03-17"


def test_enrich_tennis_matches_with_rankings_history_fills_by_id_and_name(monkeypatch):
    matches = pd.DataFrame(
        [
            {
                "tour": "atp",
                "event_date": pd.Timestamp("2025-03-18"),
                "player_a_id": "A0E2",
                "player_b_id": "S0AG",
                "player_a": "C. Alcaraz",
                "player_b": "J. Sinner",
                "player_a_rank": pd.NA,
                "player_b_rank": 2.0,
                "player_a_rank_points": pd.NA,
                "player_b_rank_points": 11400.0,
            },
            {
                "tour": "wta",
                "event_date": pd.Timestamp("2025-03-18"),
                "player_a_id": "OLD123",
                "player_b_id": "326408",
                "player_a": "Polina Kudermetova",
                "player_b": "Iga Swiatek",
                "player_a_rank": pd.NA,
                "player_b_rank": 2.0,
                "player_a_rank_points": pd.NA,
                "player_b_rank_points": 7375.0,
            },
        ]
    )

    monkeypatch.setattr(rankings, "_atp_rankings_available_dates", lambda session=None: ["2025-03-17"])
    monkeypatch.setattr(rankings, "resolve_wta_snapshot_date", lambda request_date, force=False, session=None: "2025-03-17")

    def fake_load_rankings_snapshots(tour, snapshot_dates, fetch_missing=False, force=False, session=None):
        if tour == "atp":
            return pd.DataFrame(
                [
                    {
                        "tour": "atp",
                        "snapshot_date": "2025-03-17",
                        "player_id": "A0E2",
                        "display_name": "C. Alcaraz",
                        "full_name": "Carlos Alcaraz",
                        "rank": 1,
                        "rank_points": 13550,
                    },
                    {
                        "tour": "atp",
                        "snapshot_date": "2025-03-17",
                        "player_id": "S0AG",
                        "display_name": "J. Sinner",
                        "full_name": "Jannik Sinner",
                        "rank": 2,
                        "rank_points": 11400,
                    },
                ]
            )
        return pd.DataFrame(
            [
                {
                    "tour": "wta",
                    "snapshot_date": "2025-03-17",
                    "player_id": "328897",
                    "display_name": "Polina Kudermetova",
                    "full_name": "Polina Kudermetova",
                    "rank": 58,
                    "rank_points": 1078,
                },
                {
                    "tour": "wta",
                    "snapshot_date": "2025-03-17",
                    "player_id": "326408",
                    "display_name": "Iga Swiatek",
                    "full_name": "Iga Swiatek",
                    "rank": 2,
                    "rank_points": 7375,
                },
            ]
        )

    monkeypatch.setattr(rankings, "load_rankings_snapshots", fake_load_rankings_snapshots)

    enriched = rankings.enrich_tennis_matches_with_rankings_history(matches, fetch_missing=True)

    assert float(enriched.loc[0, "player_a_rank"]) == 1.0
    assert float(enriched.loc[0, "player_a_rank_points"]) == 13550.0
    assert float(enriched.loc[1, "player_a_rank"]) == 58.0
    assert float(enriched.loc[1, "player_a_rank_points"]) == 1078.0


def test_enrich_tennis_matches_with_rankings_history_falls_back_to_cached_snapshot_dates(monkeypatch, tmp_path):
    monkeypatch.setattr(rankings, "TENNIS_RAW_DIR", tmp_path)
    cached_dir = tmp_path / "atp" / "rankings"
    cached_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "tour": "atp",
                "snapshot_date": "2025-03-17",
                "player_id": "M0NI",
                "display_name": "Jakub Mensik",
                "full_name": "Jakub Mensik",
                "rank": 54,
                "rank_points": 1042,
            }
        ]
    ).to_csv(cached_dir / "atp_singles_rankings_2025-03-17.csv", index=False)

    matches = pd.DataFrame(
        [
            {
                "tour": "atp",
                "event_date": pd.Timestamp("2025-03-18"),
                "player_a_id": "",
                "player_b_id": "M0NI",
                "player_a": "Frances Tiafoe",
                "player_b": "Jakub Mensik",
                "player_a_rank": 17.0,
                "player_b_rank": pd.NA,
                "player_a_rank_points": 2485.0,
                "player_b_rank_points": pd.NA,
            }
        ]
    )

    def fail_available_dates(session=None):
        raise RuntimeError("official ATP rankings unavailable")

    monkeypatch.setattr(rankings, "_atp_rankings_available_dates", fail_available_dates)

    enriched = rankings.enrich_tennis_matches_with_rankings_history(matches, fetch_missing=True)

    assert float(enriched.loc[0, "player_b_rank"]) == 54.0
    assert float(enriched.loc[0, "player_b_rank_points"]) == 1042.0


def test_enrich_live_tennis_matchups_with_current_rankings_overwrites_stale_values_and_restores_unmatched(monkeypatch):
    live_matchups = pd.DataFrame(
        [
            {
                "tour": "atp",
                "commence_time": "2026-03-23T18:00:00Z",
                "fighter_a": "Frances Tiafoe",
                "fighter_b": "Jakub Mensik",
                "player_a_rank": 17.0,
                "player_b_rank": 48.0,
                "player_a_rank_points": 2585.0,
                "player_b_rank_points": 1136.0,
            },
            {
                "tour": "atp",
                "commence_time": "2026-03-23T19:40:00Z",
                "fighter_a": "Quentin Halys",
                "fighter_b": "Kamil Majchrzak",
                "player_a_rank": 84.0,
                "player_b_rank": 163.0,
                "player_a_rank_points": 693.0,
                "player_b_rank_points": 364.0,
            },
        ]
    )

    monkeypatch.setattr(rankings, "_atp_rankings_available_dates", lambda session=None: ["2026-03-23"])

    def fake_load_rankings_snapshots(tour, snapshot_dates, fetch_missing=False, force=False, session=None):
        assert tour == "atp"
        assert snapshot_dates == ["2026-03-23"]
        return pd.DataFrame(
            [
                {
                    "tour": "atp",
                    "snapshot_date": "2026-03-23",
                    "player_id": "TD51",
                    "display_name": "Frances Tiafoe",
                    "full_name": "Frances Tiafoe",
                    "rank": 17,
                    "rank_points": 2585,
                },
                {
                    "tour": "atp",
                    "snapshot_date": "2026-03-23",
                    "player_id": "M0NI",
                    "display_name": "Jakub Mensik",
                    "full_name": "Jakub Mensik",
                    "rank": 12,
                    "rank_points": 4290,
                },
            ]
        )

    monkeypatch.setattr(rankings, "load_rankings_snapshots", fake_load_rankings_snapshots)

    enriched = rankings.enrich_live_tennis_matchups_with_current_rankings(live_matchups, fetch_missing=True)

    assert float(enriched.loc[0, "player_a_rank"]) == 17.0
    assert float(enriched.loc[0, "player_b_rank"]) == 12.0
    assert float(enriched.loc[0, "player_b_rank_points"]) == 4290.0
    assert float(enriched.loc[1, "player_a_rank"]) == 84.0
    assert float(enriched.loc[1, "player_b_rank"]) == 163.0
    assert float(enriched.loc[1, "player_a_rank_points"]) == 693.0
    assert float(enriched.loc[1, "player_b_rank_points"]) == 364.0
