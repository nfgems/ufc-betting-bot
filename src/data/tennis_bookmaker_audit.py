"""Read-only evidence audit for historical tennis bookmaker data sources."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import BETSAPI_BASE_URL, BETSAPI_TOKEN, ODDS_API_KEY, PROCESSED_DATA_DIR
from src.data.odds_client import OddsClient
from src.data.tennis_data import derive_tournament_seed, load_processed_tennis_data

logger = logging.getLogger(__name__)

BOOKMAKER_AUDIT_DIR = PROCESSED_DATA_DIR / "tennis" / "bookmaker_audit"
BOOKMAKER_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_AUDIT_START_DATE = "2022-01-01"
AUDIT_SOURCE_ORDER = ["odds_api", "betsapi"]
UNMATCHED_EXAMPLE_LIMIT = 50
TIMESTAMP_CHECK_LIMIT = 200
THE_ODDS_API_HISTORICAL_URL = "https://the-odds-api.com/historical-odds-data/"
THE_ODDS_API_TENNIS_COVERAGE_URL = "https://the-odds-api.com/sports/tennis-odds.html"
BETSAPI_EVENTS_URL = "https://b365api.com/events/"
BETSAPI_EVENT_ODDS_URL = "https://b365api.com/event-odds/"
BETSAPI_TENNIS_SPORT_ID = 13
ODDS_API_SCAN_DAYS_PER_TOURNAMENT = 1
ODDS_API_MAX_INSTANCES_PER_YEAR_TOUR = 2
ODDS_API_QUERY_LEAD_DAYS = 1
ODDS_API_QUERY_HOUR_UTC = 18
MATCH_PROGRESS_LOG_INTERVAL = 2000
GRAND_SLAM_TOURNAMENT_SEEDS = {
    "australian open",
    "french open",
    "wimbledon",
    "us open",
}


class AuditHardFailure(RuntimeError):
    """Raised when a source cannot be audited at all and fallback should run."""

    def __init__(self, source: str, reason: str):
        super().__init__(reason)
        self.source = source
        self.reason = reason


@dataclass(frozen=True)
class AuditArtifacts:
    """Concrete output paths for one audit source."""

    coverage_summary: Path
    join_by_year: Path
    timestamp_checks: Path
    unmatched_examples: Path


def normalize_tennis_text(value: object) -> str:
    """Normalize tennis join text into a lowercase ASCII token string."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_tennis_player_name(value: object) -> str:
    """Normalize a tennis player name for cross-source joins."""
    return normalize_tennis_text(value)


def tennis_player_names_match(left: object, right: object) -> bool:
    """Match tennis player names with an exact-or-initial surname fallback."""
    left_norm = normalize_tennis_player_name(left)
    right_norm = normalize_tennis_player_name(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True

    left_tokens = left_norm.split()
    right_tokens = right_norm.split()
    if not left_tokens or not right_tokens or left_tokens[-1] != right_tokens[-1]:
        return False

    left_first = left_tokens[0]
    right_first = right_tokens[0]
    return (
        left_first == right_first
        or left_first.startswith(right_first)
        or right_first.startswith(left_first)
        or left_first[:1] == right_first[:1]
    )


def infer_tour_from_name(value: object) -> str:
    """Infer ATP/WTA from tournament or league copy when available."""
    text = normalize_tennis_text(value)
    if any(token in text for token in ["wta", "women", "womens", "ladies"]):
        return "wta"
    if any(token in text for token in ["atp", "men", "mens"]):
        return "atp"
    return ""


def prepare_tennis_ground_truth(
    matches_df: pd.DataFrame,
    start_date: str = DEFAULT_AUDIT_START_DATE,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Filter and enrich the processed tennis universe for source audits."""
    frame = matches_df.copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    frame = frame.dropna(subset=["event_date", "tour", "player_a", "player_b"]).copy()
    if end_date is None:
        end_date = frame["event_date"].max().strftime("%Y-%m-%d")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    frame = frame[(frame["event_date"] >= start_ts) & (frame["event_date"] <= end_ts)].copy()
    frame = frame.sort_values(["event_date", "tour", "tourney_name", "match_num"]).reset_index(drop=True)
    frame["tour"] = frame["tour"].astype(str).str.lower()
    frame["audit_row_id"] = frame.index.astype(int)
    frame["event_year"] = frame["event_date"].dt.year.astype(int)
    frame["tournament_seed"] = frame["tournament_seed"].fillna(frame["tourney_name"].map(derive_tournament_seed))
    frame["tournament_seed_norm"] = frame["tournament_seed"].map(normalize_tennis_text)
    frame["tourney_name_norm"] = frame["tourney_name"].map(normalize_tennis_text)
    frame["player_a_norm"] = frame["player_a"].map(normalize_tennis_player_name)
    frame["player_b_norm"] = frame["player_b"].map(normalize_tennis_player_name)
    return frame


def prepare_bookmaker_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a flat bookmaker event frame into audit join columns."""
    columns = [
        "source",
        "source_event_id",
        "tour",
        "tournament_name",
        "tournament_seed",
        "player_a",
        "player_b",
        "scheduled_match_start",
        "snapshot_timestamp",
        "bookmaker_last_update",
        "bookmaker_count",
        "source_context",
    ]
    if events_df.empty:
        return pd.DataFrame(columns=columns + [
            "tournament_seed_norm",
            "tournament_name_norm",
            "player_a_norm",
            "player_b_norm",
        ])

    frame = events_df.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame["tour"] = frame["tour"].fillna("").astype(str).str.lower()
    missing_tour = frame["tour"] == ""
    if missing_tour.any():
        frame.loc[missing_tour, "tour"] = frame.loc[missing_tour, "tournament_name"].map(infer_tour_from_name)
    frame["tournament_seed"] = frame["tournament_seed"].fillna(frame["tournament_name"].map(derive_tournament_seed))
    frame["scheduled_match_start"] = pd.to_datetime(frame["scheduled_match_start"], errors="coerce", utc=True)
    frame["snapshot_timestamp"] = pd.to_datetime(frame["snapshot_timestamp"], errors="coerce", utc=True)
    frame["bookmaker_last_update"] = pd.to_datetime(frame["bookmaker_last_update"], errors="coerce", utc=True)
    frame["tournament_seed_norm"] = frame["tournament_seed"].map(normalize_tennis_text)
    frame["tournament_name_norm"] = frame["tournament_name"].map(normalize_tennis_text)
    frame["player_a_norm"] = frame["player_a"].map(normalize_tennis_player_name)
    frame["player_b_norm"] = frame["player_b"].map(normalize_tennis_player_name)
    return frame[columns + [
        "tournament_seed_norm",
        "tournament_name_norm",
        "player_a_norm",
        "player_b_norm",
    ]].copy()


def tournament_window_days(tournament_seed: object) -> int:
    """Use a broader window for Slams because matches span multiple weeks."""
    seed = normalize_tennis_text(tournament_seed)
    return 21 if seed in GRAND_SLAM_TOURNAMENT_SEEDS else 14


def tournaments_match(
    ground_seed: object,
    ground_name: object,
    event_seed: object,
    event_name: object,
) -> bool:
    """Match tournaments by derived seed first, then normalized title fallback."""
    ground_seed_norm = normalize_tennis_text(ground_seed)
    event_seed_norm = normalize_tennis_text(event_seed)
    if ground_seed_norm and event_seed_norm and ground_seed_norm == event_seed_norm:
        return True

    ground_name_norm = normalize_tennis_text(ground_name)
    event_name_norm = normalize_tennis_text(event_name)
    if not ground_name_norm or not event_name_norm:
        return False
    return ground_name_norm == event_name_norm


def _event_in_tournament_window(
    ground_event_date: pd.Timestamp,
    event_start: pd.Timestamp,
    tournament_seed: object,
) -> bool:
    if pd.isna(event_start):
        return False
    start = pd.Timestamp(ground_event_date).normalize().tz_localize("UTC")
    end = start + pd.Timedelta(days=tournament_window_days(tournament_seed))
    return start <= pd.Timestamp(event_start) <= end


def _candidate_event_frame(
    *,
    row_tour: str,
    row_tournament_seed_norm: str,
    row_tourney_name_norm: str,
    event_groups_by_seed: dict[tuple[str, str], pd.DataFrame],
    event_groups_by_name: dict[tuple[str, str], pd.DataFrame],
    empty_template: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve candidate tournament rows from keyed groups without scanning all events."""
    candidate_groups: list[pd.DataFrame] = []

    if row_tournament_seed_norm:
        for key in ((row_tour, row_tournament_seed_norm), ("", row_tournament_seed_norm)):
            group = event_groups_by_seed.get(key)
            if group is not None and not group.empty:
                candidate_groups.append(group)

    if row_tourney_name_norm:
        for key in ((row_tour, row_tourney_name_norm), ("", row_tourney_name_norm)):
            group = event_groups_by_name.get(key)
            if group is not None and not group.empty:
                candidate_groups.append(group)

    if not candidate_groups:
        return empty_template.copy()

    combined = pd.concat(candidate_groups, axis=0)
    return combined.loc[~combined.index.duplicated(keep="first")].copy()


def match_ground_truth_to_bookmaker_events(
    ground_truth_df: pd.DataFrame,
    bookmaker_events_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join tennis ground truth rows to bookmaker events using tennis-only keys."""
    ground_truth = (
        prepare_tennis_ground_truth(ground_truth_df)
        if "audit_row_id" not in ground_truth_df.columns
        else ground_truth_df.copy()
    )
    events = prepare_bookmaker_events(bookmaker_events_df)
    if events.empty:
        return pd.DataFrame(), ground_truth.copy()

    joined_rows: list[dict[str, object]] = []
    matched_ids: set[int] = set()
    event_groups_by_seed = {
        key: group
        for key, group in events.groupby(["tour", "tournament_seed_norm"], dropna=False)
    }
    event_groups_by_name = {
        key: group
        for key, group in events.groupby(["tour", "tournament_name_norm"], dropna=False)
    }
    empty_candidates = events.iloc[0:0].copy()
    total_rows = int(len(ground_truth))
    logger.info(
        "Starting tennis bookmaker join for %s ground-truth rows against %s bookmaker events",
        total_rows,
        len(events),
    )

    for row_number, row in enumerate(ground_truth.itertuples(index=False), start=1):
        candidates = _candidate_event_frame(
            row_tour=row.tour,
            row_tournament_seed_norm=row.tournament_seed_norm,
            row_tourney_name_norm=row.tourney_name_norm,
            event_groups_by_seed=event_groups_by_seed,
            event_groups_by_name=event_groups_by_name,
            empty_template=empty_candidates,
        )
        if not candidates.empty:
            player_candidates = candidates[
                candidates.apply(
                    lambda candidate: (
                        tennis_player_names_match(row.player_a, candidate["player_a"])
                        and tennis_player_names_match(row.player_b, candidate["player_b"])
                    )
                    or (
                        tennis_player_names_match(row.player_a, candidate["player_b"])
                        and tennis_player_names_match(row.player_b, candidate["player_a"])
                    ),
                    axis=1,
                )
            ].copy()

            if not player_candidates.empty:
                player_candidates = player_candidates[
                    player_candidates["scheduled_match_start"].map(
                        lambda value: _event_in_tournament_window(row.event_date, value, row.tournament_seed)
                    )
                ].copy()

                if not player_candidates.empty:
                    best_match = (
                        player_candidates.sort_values(
                            ["scheduled_match_start", "snapshot_timestamp", "source_event_id"],
                            na_position="last",
                        )
                        .iloc[0]
                        .to_dict()
                    )
                    joined_rows.append(
                        {
                            "audit_row_id": int(row.audit_row_id),
                            "event_year": int(row.event_year),
                            "tour": row.tour,
                            "event_date": pd.Timestamp(row.event_date),
                            "tourney_name": row.tourney_name,
                            "tournament_seed": row.tournament_seed,
                            "player_a": row.player_a,
                            "player_b": row.player_b,
                            "source": best_match["source"],
                            "source_event_id": best_match["source_event_id"],
                            "source_tournament_name": best_match["tournament_name"],
                            "source_tournament_seed": best_match["tournament_seed"],
                            "scheduled_match_start": best_match["scheduled_match_start"],
                            "snapshot_timestamp": best_match["snapshot_timestamp"],
                            "bookmaker_last_update": best_match["bookmaker_last_update"],
                            "bookmaker_count": best_match["bookmaker_count"],
                            "source_context": best_match["source_context"],
                            "candidate_count": int(len(player_candidates)),
                        }
                    )
                    matched_ids.add(int(row.audit_row_id))

        if row_number % MATCH_PROGRESS_LOG_INTERVAL == 0 or row_number == total_rows:
            logger.info(
                "Tennis bookmaker join progress: %s/%s rows processed, %s matched",
                row_number,
                total_rows,
                len(joined_rows),
            )

    joined = pd.DataFrame(joined_rows)
    unmatched = ground_truth[~ground_truth["audit_row_id"].isin(matched_ids)].copy()
    logger.info(
        "Completed tennis bookmaker join with %s matched rows and %s unmatched rows",
        len(joined),
        len(unmatched),
    )
    return joined, unmatched


def build_join_by_year(
    ground_truth_df: pd.DataFrame,
    joined_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate audit join coverage by year and tour."""
    base = ground_truth_df.groupby(["event_year", "tour"]).agg(
        ground_truth_rows=("audit_row_id", "size"),
        ground_truth_tournaments=("tournament_seed_norm", "nunique"),
    )

    if joined_df.empty:
        result = base.reset_index()
        result["matched_rows"] = 0
        result["matched_tournaments"] = 0
        result["unmatched_rows"] = result["ground_truth_rows"]
        result["matched_rate"] = 0.0
        return result.sort_values(["event_year", "tour"]).reset_index(drop=True)

    matched = joined_df.groupby(["event_year", "tour"]).agg(
        matched_rows=("audit_row_id", "size"),
        matched_tournaments=("tournament_seed", lambda series: pd.Series(series).map(normalize_tennis_text).nunique()),
    )
    result = base.join(matched, how="left").fillna(0).reset_index()
    result["matched_rows"] = result["matched_rows"].astype(int)
    result["matched_tournaments"] = result["matched_tournaments"].astype(int)
    result["unmatched_rows"] = result["ground_truth_rows"] - result["matched_rows"]
    result["matched_rate"] = (
        result["matched_rows"] / result["ground_truth_rows"].where(result["ground_truth_rows"] > 0, 1)
    ).round(6)
    return result.sort_values(["event_year", "tour"]).reset_index(drop=True)


def build_timestamp_checks(joined_df: pd.DataFrame, limit: int = TIMESTAMP_CHECK_LIMIT) -> pd.DataFrame:
    """Create bounded timestamp evidence rows for the join sample."""
    columns = [
        "event_year",
        "tour",
        "player_a",
        "player_b",
        "source_event_id",
        "scheduled_match_start",
        "snapshot_timestamp",
        "bookmaker_last_update",
        "snapshot_before_start",
        "bookmaker_update_before_start",
        "snapshot_gap_minutes",
    ]
    if joined_df.empty:
        return pd.DataFrame(columns=columns)

    frame = joined_df.copy()
    frame["scheduled_match_start"] = pd.to_datetime(frame["scheduled_match_start"], errors="coerce", utc=True)
    frame["snapshot_timestamp"] = pd.to_datetime(frame["snapshot_timestamp"], errors="coerce", utc=True)
    frame["bookmaker_last_update"] = pd.to_datetime(frame["bookmaker_last_update"], errors="coerce", utc=True)
    frame["snapshot_before_start"] = (
        frame["snapshot_timestamp"].notna()
        & frame["scheduled_match_start"].notna()
        & (frame["snapshot_timestamp"] < frame["scheduled_match_start"])
    )
    frame["bookmaker_update_before_start"] = (
        frame["bookmaker_last_update"].notna()
        & frame["scheduled_match_start"].notna()
        & (frame["bookmaker_last_update"] < frame["scheduled_match_start"])
    )
    frame["snapshot_gap_minutes"] = (
        (frame["scheduled_match_start"] - frame["snapshot_timestamp"]).dt.total_seconds() / 60.0
    ).round(2)
    return frame[columns].sort_values(
        ["snapshot_before_start", "event_year", "tour", "scheduled_match_start"],
        ascending=[True, True, True, True],
        na_position="last",
    ).head(limit).reset_index(drop=True)


def build_unmatched_examples(unmatched_df: pd.DataFrame, limit: int = UNMATCHED_EXAMPLE_LIMIT) -> pd.DataFrame:
    """Keep a bounded sample of unmatched rows for manual inspection."""
    columns = ["event_year", "tour", "event_date", "tourney_name", "tournament_seed", "player_a", "player_b"]
    if unmatched_df.empty:
        return pd.DataFrame(columns=columns)
    return unmatched_df[columns].sort_values(
        ["event_year", "tour", "event_date", "tourney_name", "player_a", "player_b"]
    ).head(limit).reset_index(drop=True)


def _the_odds_api_coverage(session: Optional[requests.Session] = None) -> tuple[pd.DataFrame, dict[str, object]]:
    """Scrape the official The Odds API tennis coverage page into a small table."""
    session = session or requests.Session()
    response = session.get(THE_ODDS_API_TENNIS_COVERAGE_URL, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text("\n")
    rows: list[dict[str, object]] = []
    for table_row in soup.find_all("tr"):
        cells = [" ".join(cell.get_text(" ", strip=True).split()) for cell in table_row.find_all("td")]
        if len(cells) < 3:
            continue
        tournament_name, sport_key, historical_start = cells[:3]
        tour_label = infer_tour_from_name(tournament_name)
        if not sport_key.startswith(("tennis_atp_", "tennis_wta_")):
            continue
        rows.append(
            {
                "tour": tour_label or ("wta" if sport_key.startswith("tennis_wta_") else "atp"),
                "tournament_name": tournament_name,
                "tournament_seed": derive_tournament_seed(tournament_name),
                "sport_key": sport_key,
                "historical_start": pd.Timestamp(historical_start, tz="UTC"),
            }
        )

    scope_note = ""
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if "Grand Slams" in line and "ATP 1000" in line and "WTA 1000" in line:
            scope_note = line
            break

    coverage = pd.DataFrame(rows)
    meta = {
        "coverage_rows": int(len(coverage)),
        "coverage_scope_note": scope_note,
        "coverage_url": THE_ODDS_API_TENNIS_COVERAGE_URL,
        "historical_url": THE_ODDS_API_HISTORICAL_URL,
        "scan_days_per_tournament": ODDS_API_SCAN_DAYS_PER_TOURNAMENT,
        "max_instances_per_year_tour": ODDS_API_MAX_INSTANCES_PER_YEAR_TOUR,
        "query_lead_days": ODDS_API_QUERY_LEAD_DAYS,
        "query_hour_utc": ODDS_API_QUERY_HOUR_UTC,
    }
    return coverage, meta


def _odds_api_tournament_instances(
    ground_truth_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> pd.DataFrame:
    instances = (
        ground_truth_df[["tour", "event_date", "tournament_seed", "tournament_seed_norm"]]
        .drop_duplicates()
        .merge(
            coverage_df[["tour", "tournament_seed", "sport_key", "historical_start"]],
            how="inner",
            on=["tour", "tournament_seed"],
        )
    )
    instances["event_year"] = pd.to_datetime(instances["event_date"]).dt.year.astype(int)
    return (
        instances.sort_values(["event_date", "tour", "tournament_seed"])
        .groupby(["event_year", "tour"], dropna=False)
        .head(ODDS_API_MAX_INSTANCES_PER_YEAR_TOUR)
        .reset_index(drop=True)
    )


def _flatten_odds_api_response(
    source_response: dict,
    *,
    tour: str,
    tournament_name: str,
    tournament_seed: str,
    sport_key: str,
) -> pd.DataFrame:
    events = source_response.get("data", []) if isinstance(source_response, dict) else []
    snapshot_timestamp = pd.to_datetime(source_response.get("timestamp"), errors="coerce", utc=True)
    rows: list[dict[str, object]] = []
    for event in events:
        updates = [
            bookmaker.get("last_update")
            for bookmaker in event.get("bookmakers", [])
            if bookmaker.get("last_update")
        ]
        last_update = pd.to_datetime(max(updates), errors="coerce", utc=True) if updates else pd.NaT
        rows.append(
            {
                "source": "odds_api",
                "source_event_id": event.get("id", ""),
                "tour": tour,
                "tournament_name": tournament_name or event.get("sport_title") or sport_key,
                "tournament_seed": tournament_seed,
                "player_a": event.get("home_team", ""),
                "player_b": event.get("away_team", ""),
                "scheduled_match_start": event.get("commence_time"),
                "snapshot_timestamp": snapshot_timestamp,
                "bookmaker_last_update": last_update,
                "bookmaker_count": len(event.get("bookmakers", [])),
                "source_context": sport_key,
            }
        )
    return pd.DataFrame(rows)


def collect_odds_api_events(
    ground_truth_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> pd.DataFrame:
    """Collect tennis historical events from The Odds API across covered tournaments."""
    if not ODDS_API_KEY:
        raise AuditHardFailure("odds_api", "ODDS_API_KEY missing")
    if coverage_df.empty:
        return pd.DataFrame()

    client = OddsClient()
    rows: list[pd.DataFrame] = []
    instances = _odds_api_tournament_instances(ground_truth_df, coverage_df)
    logger.info(
        "Odds API audit coverage resolved to %s tournament instances; scanning up to %s historical snapshots",
        len(instances),
        len(instances) * ODDS_API_SCAN_DAYS_PER_TOURNAMENT,
    )
    for row in instances.itertuples(index=False):
        window_days = min(tournament_window_days(row.tournament_seed), ODDS_API_SCAN_DAYS_PER_TOURNAMENT)
        for offset in range(window_days):
            query_timestamp = (
                pd.Timestamp(row.event_date).normalize().tz_localize("UTC")
                - pd.Timedelta(days=ODDS_API_QUERY_LEAD_DAYS)
                + pd.Timedelta(days=offset, hours=ODDS_API_QUERY_HOUR_UTC)
            )
            if query_timestamp < row.historical_start:
                continue

            logger.info(
                "Odds API audit query: %s %s @ %s",
                row.tour.upper(),
                row.tournament_seed,
                query_timestamp.isoformat(),
            )
            response = client.get_historical_sport_odds(
                date=query_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                sport_key=row.sport_key,
                regions="uk,eu,us",
                markets="h2h",
                odds_format="decimal",
            )
            flat = _flatten_odds_api_response(
                response,
                tour=row.tour,
                tournament_name=row.tournament_seed,
                tournament_seed=row.tournament_seed,
                sport_key=row.sport_key,
            )
            if not flat.empty:
                rows.append(flat)

    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["source_event_id", "player_a", "player_b", "scheduled_match_start", "snapshot_timestamp"]
    ).reset_index(drop=True)
    logger.info("Odds API audit collection produced %s flattened bookmaker rows", len(combined))
    return combined


class BetsApiClient:
    """Minimal BetsAPI/B365API client for the tennis evidence audit."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or BETSAPI_TOKEN
        if not self.token:
            raise AuditHardFailure("betsapi", "BETSAPI_TOKEN missing")
        self.base_url = BETSAPI_BASE_URL.rstrip("/")

    def _get(self, endpoint: str, params: Optional[dict[str, object]] = None) -> dict:
        payload = dict(params or {})
        payload["token"] = self.token
        response = requests.get(f"{self.base_url}/{endpoint}", params=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if int(data.get("success", 0)) != 1:
            raise AuditHardFailure("betsapi", f"BetsAPI request failed for {endpoint}")
        return data

    def get_ended_events(self, day: pd.Timestamp, page: int = 1) -> dict:
        return self._get(
            "events/ended",
            params={
                "sport_id": BETSAPI_TENNIS_SPORT_ID,
                "day": day.strftime("%Y%m%d"),
                "page": page,
            },
        )

    def get_event_odds_summary(self, event_id: object) -> dict:
        return self._get(
            "event/odds/summary",
            params={
                "event_id": event_id,
                "source": "bet365",
            },
        )


def _betsapi_event_rows(payload: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event in payload.get("results", []):
        league = event.get("league") or {}
        home = event.get("home") or {}
        away = event.get("away") or {}
        league_name = league.get("name", "")
        rows.append(
            {
                "source": "betsapi",
                "source_event_id": event.get("id", ""),
                "tour": infer_tour_from_name(league_name),
                "tournament_name": league_name,
                "tournament_seed": derive_tournament_seed(league_name),
                "player_a": home.get("name", ""),
                "player_b": away.get("name", ""),
                "scheduled_match_start": pd.to_datetime(
                    pd.to_numeric(event.get("time"), errors="coerce"),
                    unit="s",
                    errors="coerce",
                    utc=True,
                ),
                "snapshot_timestamp": pd.NaT,
                "bookmaker_last_update": pd.NaT,
                "bookmaker_count": 1,
                "source_context": league_name,
            }
        )
    return pd.DataFrame(rows)


def _extract_betsapi_summary_timestamps(summary_payload: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    results = summary_payload.get("results") or {}
    bet365 = results.get("Bet365") or {}
    odds_update = bet365.get("odds_update") or {}
    last_update_values = [value for value in odds_update.values() if str(value).strip()]

    start_times: list[pd.Timestamp] = []
    odds = bet365.get("odds") or {}
    for market in (odds.get("start") or {}).values():
        add_time = market.get("add_time")
        parsed = pd.to_datetime(pd.to_numeric(add_time, errors="coerce"), unit="s", errors="coerce", utc=True)
        if not pd.isna(parsed):
            start_times.append(parsed)

    snapshot_timestamp = min(start_times) if start_times else pd.NaT
    bookmaker_last_update = (
        pd.to_datetime(max(pd.to_numeric(last_update_values, errors="coerce")), unit="s", errors="coerce", utc=True)
        if last_update_values
        else pd.NaT
    )
    return snapshot_timestamp, bookmaker_last_update


def collect_betsapi_events(
    ground_truth_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Collect completed tennis events day by day from BetsAPI."""
    client = BetsApiClient()
    all_rows: list[pd.DataFrame] = []
    for day in pd.date_range(start_date, end_date, freq="D"):
        page = 1
        while True:
            logger.info("BetsAPI audit query: ended tennis events on %s page %s", day.strftime("%Y-%m-%d"), page)
            payload = client.get_ended_events(day, page=page)
            frame = _betsapi_event_rows(payload)
            if not frame.empty:
                all_rows.append(frame)

            pager = payload.get("pager") or {}
            total_pages = int(pager.get("total_page", page) or page)
            if page >= total_pages:
                break
            page += 1

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=["source_event_id"]).reset_index(drop=True)
    joined, _ = match_ground_truth_to_bookmaker_events(ground_truth_df, combined)
    if joined.empty:
        return combined

    summary_updates: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for event_id in joined["source_event_id"].astype(str).unique():
        payload = client.get_event_odds_summary(event_id)
        summary_updates[event_id] = _extract_betsapi_summary_timestamps(payload)

    combined["source_event_id"] = combined["source_event_id"].astype(str)
    combined["snapshot_timestamp"] = combined["source_event_id"].map(lambda event_id: summary_updates.get(event_id, (pd.NaT, pd.NaT))[0])
    combined["bookmaker_last_update"] = combined["source_event_id"].map(lambda event_id: summary_updates.get(event_id, (pd.NaT, pd.NaT))[1])
    return combined


def _timestamp_metrics(joined_df: pd.DataFrame) -> dict[str, object]:
    checks = build_timestamp_checks(joined_df, limit=max(len(joined_df), 1))
    snapshot_supported = bool(not checks.empty and checks["snapshot_timestamp"].notna().any())
    pre_match_passes = int(checks["snapshot_before_start"].sum()) if "snapshot_before_start" in checks.columns else 0
    total_checks = int(len(checks[checks["snapshot_timestamp"].notna() & checks["scheduled_match_start"].notna()]))
    return {
        "timestamp_supported": snapshot_supported,
        "timestamp_check_rows": total_checks,
        "timestamp_pre_match_passes": pre_match_passes,
        "timestamp_all_pass": bool(total_checks > 0 and pre_match_passes == total_checks),
    }


def _required_year_tour_evidence(join_by_year: pd.DataFrame) -> dict[str, bool]:
    evidence: dict[str, bool] = {}
    for year in range(2022, 2027):
        year_frame = join_by_year[join_by_year["event_year"] == year]
        evidence[f"{year}_atp"] = bool((year_frame["tour"] == "atp").any() and int(year_frame.loc[year_frame["tour"] == "atp", "matched_rows"].iloc[0]) > 0)
        evidence[f"{year}_wta"] = bool((year_frame["tour"] == "wta").any() and int(year_frame.loc[year_frame["tour"] == "wta", "matched_rows"].iloc[0]) > 0)
    return evidence


def _narrow_subset_flag(
    source: str,
    ground_truth_df: pd.DataFrame,
    coverage_meta: dict[str, object],
    coverage_df: Optional[pd.DataFrame] = None,
) -> bool:
    if source != "odds_api":
        return False

    ground_truth_tournaments = int(ground_truth_df["tournament_seed_norm"].nunique())
    covered_tournaments = int(coverage_df["tournament_seed"].map(normalize_tennis_text).nunique()) if coverage_df is not None and not coverage_df.empty else 0
    coverage_ratio = (covered_tournaments / ground_truth_tournaments) if ground_truth_tournaments else 0.0
    scope_note = str(coverage_meta.get("coverage_scope_note") or "")
    return coverage_ratio < 0.5 or (
        "Grand Slams" in scope_note and "ATP 1000" in scope_note and "WTA 1000" in scope_note
    )


def evaluate_audit_summary(
    *,
    source: str,
    ground_truth_df: pd.DataFrame,
    joined_df: pd.DataFrame,
    join_by_year: pd.DataFrame,
    hard_fail: bool,
    reason: str,
    coverage_meta: dict[str, object],
    coverage_df: Optional[pd.DataFrame] = None,
) -> dict[str, object]:
    """Convert audit evidence into one go/no-go summary document."""
    required_evidence = _required_year_tour_evidence(join_by_year)
    timestamp_metrics = _timestamp_metrics(joined_df)
    stable_join_supported = bool(not joined_df.empty and int((joined_df["candidate_count"] > 1).sum()) == 0)
    narrow_subset = _narrow_subset_flag(source, ground_truth_df, coverage_meta, coverage_df=coverage_df)
    all_years_covered = all(required_evidence.values())

    verdict = "go"
    reasons: list[str] = []
    if hard_fail:
        verdict = "no_go"
        reasons.append(reason)
    if not all_years_covered:
        verdict = "no_go"
        reasons.append("Missing matched ATP/WTA evidence in at least one required year from 2022 through 2026")
    if not timestamp_metrics["timestamp_all_pass"]:
        verdict = "no_go"
        reasons.append("Could not prove snapshot_timestamp < scheduled_match_start for all timestamp-checked joins")
    if not stable_join_supported:
        verdict = "no_go"
        reasons.append("Stable joins are not established from the available tournament and player keys")
    if narrow_subset:
        verdict = "no_go"
        reasons.append("Coverage appears limited to a narrow tournament subset")

    return {
        "source": source,
        "verdict": verdict,
        "hard_fail": bool(hard_fail),
        "reason": reason,
        "reasons": reasons,
        "ground_truth_rows": int(len(ground_truth_df)),
        "matched_rows": int(len(joined_df)),
        "ground_truth_date_min": ground_truth_df["event_date"].min().strftime("%Y-%m-%d") if not ground_truth_df.empty else None,
        "ground_truth_date_max": ground_truth_df["event_date"].max().strftime("%Y-%m-%d") if not ground_truth_df.empty else None,
        "required_year_tour_evidence": required_evidence,
        "stable_join_supported": stable_join_supported,
        "narrow_tournament_subset": narrow_subset,
        "timestamp_metrics": timestamp_metrics,
        "ground_truth_date_note": "matches.csv event_date is the tournament start date, so joins use tournament-window scheduled starts",
        **coverage_meta,
    }


def _artifacts_for_source(source: str) -> AuditArtifacts:
    return AuditArtifacts(
        coverage_summary=BOOKMAKER_AUDIT_DIR / f"{source}_coverage_summary.json",
        join_by_year=BOOKMAKER_AUDIT_DIR / f"{source}_join_by_year.csv",
        timestamp_checks=BOOKMAKER_AUDIT_DIR / f"{source}_timestamp_checks.csv",
        unmatched_examples=BOOKMAKER_AUDIT_DIR / f"{source}_unmatched_examples.csv",
    )


def write_audit_artifacts(
    *,
    source: str,
    summary: dict[str, object],
    join_by_year: pd.DataFrame,
    timestamp_checks: pd.DataFrame,
    unmatched_examples: pd.DataFrame,
) -> AuditArtifacts:
    """Persist the compact audit evidence files for one source."""
    artifacts = _artifacts_for_source(source)
    artifacts.coverage_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    join_by_year.to_csv(artifacts.join_by_year, index=False)
    timestamp_checks.to_csv(artifacts.timestamp_checks, index=False)
    unmatched_examples.to_csv(artifacts.unmatched_examples, index=False)
    return artifacts


def audit_tennis_bookmaker_source(
    source: str,
    ground_truth_df: pd.DataFrame,
) -> dict[str, object]:
    """Run a single source audit and write the source-scoped evidence files."""
    source = str(source).lower()
    if source not in {"odds_api", "betsapi"}:
        raise ValueError(f"Unsupported tennis bookmaker audit source: {source}")

    hard_fail = False
    reason = ""
    coverage_df = pd.DataFrame()
    coverage_meta: dict[str, object] = {}
    bookmaker_events = pd.DataFrame()
    joined_df = pd.DataFrame()
    unmatched_df = ground_truth_df.copy()

    try:
        if source == "odds_api":
            coverage_df, coverage_meta = _the_odds_api_coverage()
            bookmaker_events = collect_odds_api_events(ground_truth_df, coverage_df)
        else:
            coverage_meta = {
                "coverage_url": BETSAPI_EVENTS_URL,
                "event_odds_url": BETSAPI_EVENT_ODDS_URL,
                "coverage_scope_note": "Events API supports historical queries and event odds summary exposes start/add_time when available.",
            }
            bookmaker_events = collect_betsapi_events(
                ground_truth_df,
                start_date=ground_truth_df["event_date"].min().strftime("%Y-%m-%d"),
                end_date=ground_truth_df["event_date"].max().strftime("%Y-%m-%d"),
            )

        logger.info(
            "%s tennis bookmaker audit collected %s bookmaker rows; starting join evaluation",
            source,
            len(bookmaker_events),
        )
        joined_df, unmatched_df = match_ground_truth_to_bookmaker_events(ground_truth_df, bookmaker_events)
    except AuditHardFailure as exc:
        hard_fail = True
        reason = exc.reason
        logger.warning("%s tennis bookmaker audit hard-failed: %s", source, exc.reason)
    except requests.RequestException as exc:
        hard_fail = True
        reason = f"{type(exc).__name__}: {exc}"
        logger.warning("%s tennis bookmaker audit request failed: %s", source, exc)

    join_by_year = build_join_by_year(ground_truth_df, joined_df)
    timestamp_checks = build_timestamp_checks(joined_df)
    unmatched_examples = build_unmatched_examples(unmatched_df)
    summary = evaluate_audit_summary(
        source=source,
        ground_truth_df=ground_truth_df,
        joined_df=joined_df,
        join_by_year=join_by_year,
        hard_fail=hard_fail,
        reason=reason,
        coverage_meta=coverage_meta,
        coverage_df=coverage_df if not coverage_df.empty else None,
    )
    artifacts = write_audit_artifacts(
        source=source,
        summary=summary,
        join_by_year=join_by_year,
        timestamp_checks=timestamp_checks,
        unmatched_examples=unmatched_examples,
    )
    return {
        **summary,
        "coverage_summary_path": artifacts.coverage_summary,
        "join_by_year_path": artifacts.join_by_year,
        "timestamp_checks_path": artifacts.timestamp_checks,
        "unmatched_examples_path": artifacts.unmatched_examples,
    }


def run_tennis_bookmaker_audit(
    source: str = "auto",
    start_date: str = DEFAULT_AUDIT_START_DATE,
    end_date: Optional[str] = None,
) -> dict[str, object]:
    """Run the tennis bookmaker evidence audit with odds_api -> betsapi priority."""
    matches_df = load_processed_tennis_data()
    ground_truth_df = prepare_tennis_ground_truth(matches_df, start_date=start_date, end_date=end_date)
    if ground_truth_df.empty:
        raise ValueError("No processed tennis matches fall inside the requested audit window")

    requested_source = str(source or "auto").lower()
    if requested_source == "auto":
        sources = AUDIT_SOURCE_ORDER
    elif requested_source in {"odds_api", "betsapi"}:
        sources = [requested_source]
    else:
        raise ValueError(f"Unsupported tennis bookmaker audit source: {source}")

    attempted_results: list[dict[str, object]] = []
    final_result: Optional[dict[str, object]] = None
    for current_source in sources:
        result = audit_tennis_bookmaker_source(current_source, ground_truth_df)
        attempted_results.append(result)
        final_result = result
        if requested_source != "auto":
            break
        if current_source == "odds_api" and not result.get("hard_fail", False):
            break
        if current_source == "odds_api" and result.get("hard_fail", False):
            logger.info("Odds API audit hard-failed; falling back to BetsAPI")
            continue
        break

    if final_result is None:
        raise RuntimeError("No tennis bookmaker audit result was produced")

    return {
        **final_result,
        "attempted_sources": [item["source"] for item in attempted_results],
        "requested_source": requested_source,
    }
