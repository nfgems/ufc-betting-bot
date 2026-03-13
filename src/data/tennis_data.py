"""Tennis data ingestion and normalization for ATP and WTA singles."""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

logger = logging.getLogger(__name__)

TENNIS_RAW_DIR = RAW_DATA_DIR / "tennis"
ATP_RAW_DIR = TENNIS_RAW_DIR / "atp"
WTA_RAW_DIR = TENNIS_RAW_DIR / "wta"
TENNIS_PROCESSED_DIR = PROCESSED_DATA_DIR / "tennis"

for directory in [ATP_RAW_DIR, WTA_RAW_DIR, TENNIS_PROCESSED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

SACKMANN_GITHUB_API = "https://api.github.com/repos/JeffSackmann"
SACKMANN_RAW_BASE = "https://raw.githubusercontent.com/JeffSackmann"
ATP_RESULTS_ARCHIVE_URL = "https://www.atptour.com/en/scores/results-archive?year={year}"
ATP_OVERVIEW_URL = "https://www.atptour.com/en/-/tournaments/profile/{event_id}/overview"
ATP_DRAWS_URL = "https://www.atptour.com/en/scores/archive/{slug}/{event_id}/{year}/draws"
WTA_TOURNAMENTS_URL = "https://www.wtatennis.com/tournaments?year={year}"
WTA_TOURNAMENTS_API_URL = "https://api.wtatennis.com/tennis/tournaments?page={page}&pageSize={page_size}"
WTA_TOURNAMENT_URL = "https://api.wtatennis.com/tennis/tournaments/{tournament_id}/{year}"
WTA_TOURNAMENT_MATCHES_URL = "https://api.wtatennis.com/tennis/tournaments/{tournament_id}/{year}/matches"
WTA_TOURNAMENTS_PAGE_SIZE = 100
WTA_MANIFEST_SCHEMA_VERSION = 1
WTA_PROGRESS_SCHEMA_VERSION = 1
TRAINING_EXCLUSION_PATTERN = re.compile(
    r"\b(?:RET|W/O|WO|DEF|ABN|ABD|walkover|default|abandoned|cancelled)\b",
    re.IGNORECASE,
)
STANDARD_MATCH_FILE_RE = re.compile(r"^(atp|wta)_matches_(\d{4})\.csv$")
OFFICIAL_MATCH_FILE_RE = re.compile(r"^(atp|wta)_matches_(\d{4})_official\.csv$")
USER_AGENT = "Mozilla/5.0"
REQUEST_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
OFFICIAL_REQUEST_PAUSE_SECONDS = 0.25
TOURNAMENT_STOPWORDS = {
    "atp",
    "wta",
    "tennis",
    "singles",
    "single",
    "mens",
    "men",
    "womens",
    "women",
    "championships",
    "championship",
    "tour",
}
TOURNAMENT_SEED_ALIASES = {
    "roland garros": "french open",
    "united states open": "us open",
    "u s open": "us open",
}
GENERIC_TOURNAMENT_TOKENS = {
    "open",
    "masters",
    "master",
    "championship",
    "championships",
    "classic",
    "cup",
    "final",
    "finals",
    "trophy",
}
RAW_TENNIS_COLUMNS = [
    "tour",
    "tourney_id",
    "tourney_name",
    "tourney_level",
    "surface",
    "draw_size",
    "tourney_date",
    "match_num",
    "round",
    "best_of",
    "minutes",
    "score",
    "winner_name",
    "loser_name",
    "winner_id",
    "loser_id",
    "winner_rank",
    "loser_rank",
    "winner_rank_points",
    "loser_rank_points",
    "winner_age",
    "loser_age",
    "winner_hand",
    "loser_hand",
    "winner_ht",
    "loser_ht",
    "winner_seed",
    "loser_seed",
    "winner_entry",
    "loser_entry",
]
ROUND_SEQUENCE = {
    "RR": 0,
    "R128": 1,
    "R64": 2,
    "R32": 3,
    "R16": 4,
    "QF": 5,
    "SF": 6,
    "BR": 7,
    "ER": 8,
    "F": 9,
}


def normalize_text(value: object) -> str:
    """Normalize free text for matching."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_player_name(name: object) -> str:
    """Normalize a tennis player name for matching across feeds."""
    return normalize_text(name)


def normalize_name_tokens(name: object) -> tuple[str, ...]:
    """Return normalized name tokens."""
    normalized = normalize_player_name(name)
    return tuple(token for token in normalized.split() if token)


def derive_tournament_seed(value: object) -> str:
    """Extract a tournament-oriented search seed from a title or sport key."""
    if value is None or pd.isna(value):
        return ""

    raw_text = unicodedata.normalize("NFKD", str(value))
    raw_text = raw_text.encode("ascii", "ignore").decode("ascii").lower()
    if ":" in raw_text:
        raw_text = raw_text.split(":", 1)[0]

    normalized = re.sub(r"\b(?:qf|sf)\b", " ", raw_text)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    tokens = [
        token
        for token in normalized.split()
        if token
        and token not in TOURNAMENT_STOPWORDS
        and not (token.isdigit() and len(token) >= 3)
    ]
    seed = " ".join(tokens)
    return TOURNAMENT_SEED_ALIASES.get(seed, seed)


def tournament_seed_tokens(seed: object) -> tuple[str, ...]:
    """Return normalized tournament-seed tokens."""
    return tuple(token for token in derive_tournament_seed(seed).split() if token)


def collect_tournament_seed_candidates(*values: object) -> list[str]:
    """Return unique normalized tournament-seed candidates in input order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = derive_tournament_seed(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def tournament_seeds_compatible(left_seed: object, right_seed: object) -> bool:
    """Return True when two tournament seeds refer to the same event.

    Compatibility is intentionally conservative. Distinct tournaments that only
    share a location token, such as "Paris Masters" and "Paris Open", must not
    match. Numeric suffixes are also preserved so "Adelaide 1" and
    "Adelaide 2" remain distinct.
    """

    left_norm = derive_tournament_seed(left_seed)
    right_norm = derive_tournament_seed(right_seed)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True

    left_tokens = set(tournament_seed_tokens(left_norm))
    right_tokens = set(tournament_seed_tokens(right_norm))
    if not left_tokens or not right_tokens:
        return False

    left_numbers = {token for token in left_tokens if token.isdigit()}
    right_numbers = {token for token in right_tokens if token.isdigit()}
    if left_numbers != right_numbers:
        return False

    smaller_tokens, larger_tokens = (left_tokens, right_tokens)
    if len(left_tokens) > len(right_tokens):
        smaller_tokens, larger_tokens = right_tokens, left_tokens

    if len(smaller_tokens) < 2:
        return False

    return smaller_tokens.issubset(larger_tokens)


def _tour_repo_name(tour: str) -> str:
    tour = str(tour).lower()
    if tour not in {"atp", "wta"}:
        raise ValueError(f"Unsupported tennis tour: {tour}")
    return f"tennis_{tour}"


def _raw_dir_for_tour(tour: str) -> Path:
    return ATP_RAW_DIR if str(tour).lower() == "atp" else WTA_RAW_DIR


def _match_filename(tour: str, year: int) -> str:
    return f"{tour.lower()}_matches_{int(year)}.csv"


def _match_url(tour: str, year: int) -> str:
    repo_name = _tour_repo_name(tour)
    filename = _match_filename(tour, year)
    return f"{SACKMANN_RAW_BASE}/{repo_name}/master/{filename}"


def _official_match_filename(tour: str, year: int) -> str:
    return f"{tour.lower()}_matches_{int(year)}_official.csv"


def _wta_manifest_filename(year: int) -> str:
    return f"wta_tournaments_{int(year)}_official_manifest.json"


def _wta_progress_state_filename(year: int) -> str:
    return f"wta_matches_{int(year)}_official_progress.json"


def _wta_progress_rows_filename(year: int) -> str:
    return f"wta_matches_{int(year)}_official_progress.csv"


def _wta_manifest_progress_filename(year: int) -> str:
    return f"wta_tournaments_{int(year)}_official_manifest_progress.json"


def _current_tennis_year() -> int:
    return date.today().year


def _request_session(session: Optional[requests.Session] = None) -> requests.Session:
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)
    return session


def _get_with_retries(session: requests.Session, url: str, timeout: int = 60, max_attempts: int = 5) -> requests.Response:
    response: Optional[requests.Response] = None
    for attempt in range(max_attempts):
        response = session.get(url, timeout=timeout)
        if response.status_code not in REQUEST_RETRY_STATUS_CODES:
            return response

        if attempt >= max_attempts - 1:
            return response

        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            wait_seconds = max(float(retry_after), 1.0)
        elif response.status_code == 429:
            wait_seconds = float(10 * (attempt + 1))
        else:
            wait_seconds = float(min(2**attempt, 8))
        logger.warning(
            "Retrying %s after HTTP %s in %.1f seconds (attempt %s/%s)",
            url,
            response.status_code,
            wait_seconds,
            attempt + 1,
            max_attempts,
        )
        time.sleep(wait_seconds)

    if response is None:
        raise RuntimeError(f"Failed to request {url}")
    return response


def _load_json_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _with_raw_tennis_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=RAW_TENNIS_COLUMNS)
    result = frame.copy()
    for column in RAW_TENNIS_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result


def _parse_seed_or_entry(raw_value: object) -> tuple[object, object]:
    if raw_value is None or pd.isna(raw_value):
        return pd.NA, pd.NA
    text = str(raw_value).strip().strip("()")
    if not text:
        return pd.NA, pd.NA
    if text.isdigit():
        return int(text), pd.NA
    return pd.NA, text.upper()


def _parse_player_id_from_href(href: object) -> object:
    if href is None or pd.isna(href):
        return pd.NA
    parts = [part for part in str(href).split("/") if part]
    if len(parts) >= 2:
        return parts[-2]
    return pd.NA


def _parse_atp_archive_start_date(date_text: object, fallback_year: int) -> int:
    text = _clean_text(date_text)
    if not text:
        return int(pd.Timestamp(year=int(fallback_year), month=1, day=1).strftime("%Y%m%d"))

    parts = text.split(" - ", 1)
    start_text = parts[0]
    if len(parts) == 2 and "," not in start_text:
        end_tokens = parts[1].replace(",", "").split()
        start_tokens = start_text.replace(",", "").split()
        if len(start_tokens) == 1 and len(end_tokens) >= 2:
            start_text = f"{start_tokens[0]} {end_tokens[-2]} {end_tokens[-1]}"
        elif len(start_tokens) >= 2 and end_tokens:
            start_text = f"{' '.join(start_tokens[:2])} {end_tokens[-1]}"

    parsed = pd.to_datetime(start_text, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.Timestamp(year=int(fallback_year), month=1, day=1)
    return int(parsed.strftime("%Y%m%d"))


def _parse_date_to_yyyymmdd(value: object, fallback_year: int) -> int:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.Timestamp(year=int(fallback_year), month=1, day=1)
    return int(parsed.strftime("%Y%m%d"))


def _today_yyyymmdd() -> int:
    return int(pd.Timestamp.today().strftime("%Y%m%d"))


def _wta_competitor_matches(label: object, first_name: object, last_name: object) -> bool:
    label_norm = normalize_text(re.sub(r"\[[^\]]+\]", " ", str(label or "")))
    full_name_norm = normalize_text(f"{_clean_text(first_name)} {_clean_text(last_name)}")
    if not label_norm or not full_name_norm:
        return False

    label_tokens = label_norm.split()
    full_tokens = full_name_norm.split()
    if not label_tokens or not full_tokens:
        return False
    if label_tokens[-1] != full_tokens[-1]:
        return False

    label_first = label_tokens[0].rstrip(".")
    full_first = full_tokens[0]
    if len(label_first) == 1:
        return full_first.startswith(label_first)
    return label_first == full_first


def _infer_wta_winner_side(match: dict[str, object]) -> Optional[str]:
    result_string = str(match.get("ResultString") or "")
    if " d " in result_string:
        winner_label = result_string.split(" d ", 1)[0]
        if _wta_competitor_matches(winner_label, match.get("PlayerNameFirstA"), match.get("PlayerNameLastA")):
            return "A"
        if _wta_competitor_matches(winner_label, match.get("PlayerNameFirstB"), match.get("PlayerNameLastB")):
            return "B"

    winner_flag = str(match.get("Winner") or "").strip()
    if winner_flag.isdigit():
        return "A" if int(winner_flag) % 2 == 0 else "B"
    return None


def _round_sort_key(round_code: object) -> int:
    if round_code is None or pd.isna(round_code):
        return 99
    return ROUND_SEQUENCE.get(str(round_code).strip().upper(), 99)


def _map_wta_round_id(
    round_id: object,
    draw_size: object,
    tournament_level: object = None,
    tournament_name: object = None,
) -> str:
    if round_id is None or pd.isna(round_id):
        return ""
    round_code = str(round_id).strip().upper()
    if round_code == "F":
        return "F"
    if round_code == "S":
        return "SF"
    if round_code == "Q":
        return "QF"
    if round_code in {"RR", "BR", "ER"}:
        return round_code
    if round_code.isdigit():
        level_norm = normalize_text(tournament_level)
        name_norm = normalize_text(tournament_name)
        if level_norm == "finals" or "finals" in name_norm:
            return "RR"
        try:
            bracket_size = 2
            draw_size_int = max(int(draw_size), 2)
            while bracket_size < draw_size_int:
                bracket_size *= 2
            early_rounds: list[str] = []
            current_size = bracket_size
            while current_size > 8:
                early_rounds.append(f"R{current_size}")
                current_size //= 2
            index = int(round_code) - 1
            if 0 <= index < len(early_rounds):
                return early_rounds[index]
        except (TypeError, ValueError):
            return round_code
    return round_code


def _infer_wta_draw_size(tournament_ref: dict[str, object], matches: list[dict[str, object]]) -> int:
    stored_draw_size = tournament_ref.get("draw_size")
    try:
        stored_draw_size_int = int(stored_draw_size)
        if stored_draw_size_int > 0:
            return stored_draw_size_int
    except (TypeError, ValueError):
        pass

    numeric_round_ids = []
    for match in matches:
        round_id = str(match.get("RoundID") or "").strip()
        if round_id.isdigit():
            numeric_round_ids.append(int(round_id))

    if not numeric_round_ids:
        return 0

    first_round_id = min(numeric_round_ids)
    first_round_match_count = sum(1 for round_id in numeric_round_ids if round_id == first_round_id)
    if first_round_match_count > 0:
        return first_round_match_count * 2

    return int(2 ** (len(set(numeric_round_ids)) + 3))


def _map_atp_round_label(label: object) -> str:
    normalized = normalize_text(label)
    mapping = {
        "round robin": "RR",
        "bronze match": "BR",
        "elimination round": "ER",
        "round of 128": "R128",
        "round of 64": "R64",
        "round of 32": "R32",
        "round of 16": "R16",
        "quarterfinals": "QF",
        "quarter finals": "QF",
        "semi finals": "SF",
        "semifinals": "SF",
        "semi final": "SF",
        "semifinal": "SF",
        "final": "F",
        "finals": "F",
    }
    if label is None or pd.isna(label):
        return ""
    return mapping.get(normalized, str(label).strip())


def _build_match_row(
    *,
    tour: str,
    tourney_id: str,
    tourney_name: str,
    tourney_level: object,
    surface: object,
    draw_size: object,
    tourney_date: int,
    round_code: str,
    best_of: int,
    score: object,
    winner_name: str,
    loser_name: str,
    winner_id: object,
    loser_id: object,
    winner_seed: object = pd.NA,
    loser_seed: object = pd.NA,
    winner_entry: object = pd.NA,
    loser_entry: object = pd.NA,
) -> dict[str, object]:
    return {
        "tour": tour,
        "tourney_id": tourney_id,
        "tourney_name": tourney_name,
        "tourney_level": tourney_level,
        "surface": surface,
        "draw_size": draw_size,
        "tourney_date": tourney_date,
        "match_num": pd.NA,
        "round": round_code,
        "best_of": best_of,
        "minutes": pd.NA,
        "score": _clean_text(score),
        "winner_name": winner_name,
        "loser_name": loser_name,
        "winner_id": winner_id,
        "loser_id": loser_id,
        "winner_rank": pd.NA,
        "loser_rank": pd.NA,
        "winner_rank_points": pd.NA,
        "loser_rank_points": pd.NA,
        "winner_age": pd.NA,
        "loser_age": pd.NA,
        "winner_hand": pd.NA,
        "loser_hand": pd.NA,
        "winner_ht": pd.NA,
        "loser_ht": pd.NA,
        "winner_seed": winner_seed,
        "loser_seed": loser_seed,
        "winner_entry": winner_entry,
        "loser_entry": loser_entry,
    }


def _finalize_tournament_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            _round_sort_key(row.get("round")),
            row.get("_sort_time", ""),
            str(row.get("_sort_id", "")),
            normalize_player_name(row.get("winner_name")),
            normalize_player_name(row.get("loser_name")),
        ),
    )
    for match_num, row in enumerate(sorted_rows, start=1):
        row["match_num"] = match_num
        row.pop("_sort_time", None)
        row.pop("_sort_id", None)
    return sorted_rows


def list_available_sackmann_years(tour: str, session: Optional[requests.Session] = None) -> list[int]:
    """List yearly Jeff Sackmann match files available for a tour."""
    session = _request_session(session)
    repo_name = _tour_repo_name(tour)
    response = _get_with_retries(session, f"{SACKMANN_GITHUB_API}/{repo_name}/contents", timeout=30)
    response.raise_for_status()

    years = []
    for item in response.json():
        match = STANDARD_MATCH_FILE_RE.match(item.get("name", ""))
        if match and match.group(1) == tour.lower():
            years.append(int(match.group(2)))
    years.sort()
    return years


def download_sackmann_matches(
    tour: str,
    years: Iterable[int],
    force: bool = False,
    session: Optional[requests.Session] = None,
) -> list[Path]:
    """Download yearly Jeff Sackmann match files for ATP or WTA singles."""
    session = _request_session(session)
    saved_paths: list[Path] = []

    for year in years:
        destination = _raw_dir_for_tour(tour) / _match_filename(tour, year)
        if destination.exists() and not force:
            saved_paths.append(destination)
            continue

        url = _match_url(tour, year)
        response = _get_with_retries(session, url, timeout=60)
        if response.status_code == 404:
            logger.warning("Jeff Sackmann file missing for %s %s: %s", tour, year, url)
            continue
        response.raise_for_status()
        destination.write_bytes(response.content)
        saved_paths.append(destination)
        logger.info("Saved %s %s raw matches to %s", tour.upper(), year, destination)

    return saved_paths


def load_sackmann_matches(
    tour: str,
    years: Optional[Iterable[int]] = None,
    force_download: bool = False,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Load yearly Jeff Sackmann singles match files for one tour."""
    session = _request_session(session)
    available_years = list_available_sackmann_years(tour, session=session)
    if not available_years:
        raise ValueError(f"No Jeff Sackmann files found for {tour}")

    selected_years = set(int(year) for year in years) if years is not None else set(available_years)
    files = download_sackmann_matches(
        tour=tour,
        years=sorted(year for year in available_years if year in selected_years),
        force=force_download,
        session=session,
    )

    frames: list[pd.DataFrame] = []
    for csv_path in files:
        frame = pd.read_csv(csv_path)
        frame["tour"] = tour.lower()
        frame["source_file"] = csv_path.name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %s %s matches from %s yearly files", len(combined), tour.upper(), len(frames))
    return combined


def _discover_atp_tournaments(year: int, session: requests.Session) -> list[dict[str, object]]:
    response = _get_with_retries(session, ATP_RESULTS_ARCHIVE_URL.format(year=int(year)), timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")

    tournaments: list[dict[str, object]] = []
    seen: set[tuple[str, str, int]] = set()
    for item in soup.select("ul.events li"):
        results_link = item.select_one('a.results[href*="/en/scores/archive/"]')
        if results_link is None:
            continue

        href = str(results_link.get("href") or "")
        match = re.search(r"/en/scores/archive/([a-z0-9-]+)/(\d+)/(\d{4})/results$", href)
        if not match:
            continue

        slug, event_id, event_year = match.groups()
        if int(event_year) != int(year):
            continue

        key = (slug, event_id, int(event_year))
        if key in seen:
            continue
        seen.add(key)

        profile = item.select_one("a.tournament__profile")
        name_node = profile.select_one(".name") if profile else None
        date_node = profile.select_one(".Date") if profile else None
        tournament_name = _clean_text(name_node.get_text(" ", strip=True) if name_node else slug.replace("-", " ").title())
        tournament_date = _parse_atp_archive_start_date(
            date_node.get_text(" ", strip=True) if date_node else "",
            int(event_year),
        )

        tournaments.append(
            {
                "slug": slug,
                "event_id": event_id,
                "year": int(event_year),
                "tourney_name": tournament_name,
                "tourney_date": tournament_date,
            }
        )

    return tournaments


def _parse_atp_draw_player(stats_item: BeautifulSoup) -> dict[str, object]:
    name_anchor = stats_item.select_one(".player-info .name a")
    player_name = _clean_text(name_anchor.get_text(" ", strip=True) if name_anchor else "")
    name_spans = stats_item.select(".player-info .name span")
    extra_label = _clean_text(" ".join(span.get_text(" ", strip=True) for span in name_spans))
    seed, entry = _parse_seed_or_entry(extra_label)

    set_scores: list[str] = []
    for score_item in stats_item.select(".scores .score-item"):
        spans = [_clean_text(span.get_text(" ", strip=True)) for span in score_item.select("span")]
        games = spans[0] if spans else ""
        if games:
            set_scores.append(games)

    return {
        "name": player_name,
        "player_id": _parse_player_id_from_href(name_anchor.get("href") if name_anchor else None),
        "seed": seed,
        "entry": entry,
        "is_winner": stats_item.select_one(".player-info .winner") is not None,
        "set_scores": set_scores,
    }


def _format_atp_score(left_player: dict[str, object], right_player: dict[str, object], winner_index: int) -> str:
    left_sets = list(left_player.get("set_scores") or [])
    right_sets = list(right_player.get("set_scores") or [])
    set_count = max(len(left_sets), len(right_sets))
    score_sets: list[str] = []
    for index in range(set_count):
        left_games = left_sets[index] if index < len(left_sets) else ""
        right_games = right_sets[index] if index < len(right_sets) else ""
        if left_games and right_games:
            score_sets.append(f"{left_games}-{right_games}")
    score = " ".join(score_sets)
    if score:
        return score
    return ""


def _fetch_atp_official_year(year: int, session: requests.Session) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    tournaments = _discover_atp_tournaments(year, session)

    for tournament in tournaments:
        event_id = str(tournament["event_id"])
        overview_response = _get_with_retries(session, ATP_OVERVIEW_URL.format(event_id=event_id), timeout=60)
        overview_response.raise_for_status()
        overview = overview_response.json()

        draw_response = _get_with_retries(
            session,
            ATP_DRAWS_URL.format(slug=tournament["slug"], event_id=event_id, year=int(year)),
            timeout=60,
        )
        if draw_response.status_code == 404:
            logger.warning("ATP draw page missing for %s %s", tournament["slug"], year)
            continue
        draw_response.raise_for_status()
        draw_soup = BeautifulSoup(draw_response.text, "lxml")

        tournament_rows: list[dict[str, object]] = []
        for draw_section in draw_soup.select("div.atp-draw-container div.draw"):
            header = draw_section.select_one(".draw-header")
            round_code = _map_atp_round_label(header.get_text(" ", strip=True) if header else "")
            if not round_code:
                continue

            for draw_item in draw_section.select(".draw-content > .draw-item"):
                players = draw_item.select(".draw-stats > .stats-item")
                if len(players) != 2:
                    continue

                left_player = _parse_atp_draw_player(players[0])
                right_player = _parse_atp_draw_player(players[1])
                if not left_player["name"] or not right_player["name"]:
                    continue

                winner_index = 0 if left_player["is_winner"] else 1 if right_player["is_winner"] else -1
                if winner_index not in {0, 1}:
                    continue

                winner_player = left_player if winner_index == 0 else right_player
                loser_player = right_player if winner_index == 0 else left_player
                score = _format_atp_score(left_player, right_player, winner_index)
                best_of = 5 if str(overview.get("EventType") or "").upper() == "GS" else 3

                row = _build_match_row(
                    tour="atp",
                    tourney_id=f"{year}-{event_id}",
                    tourney_name=str(tournament["tourney_name"]),
                    tourney_level=overview.get("EventType"),
                    surface=overview.get("Surface") or "Unknown",
                    draw_size=overview.get("SinglesDrawSize"),
                    tourney_date=int(tournament["tourney_date"]),
                    round_code=round_code,
                    best_of=best_of,
                    score=score,
                    winner_name=str(winner_player["name"]),
                    loser_name=str(loser_player["name"]),
                    winner_id=winner_player["player_id"],
                    loser_id=loser_player["player_id"],
                    winner_seed=winner_player["seed"],
                    loser_seed=loser_player["seed"],
                    winner_entry=winner_player["entry"],
                    loser_entry=loser_player["entry"],
                )
                row["_sort_id"] = f"{round_code}:{winner_player['player_id']}:{loser_player['player_id']}"
                tournament_rows.append(row)

        rows.extend(_finalize_tournament_rows(tournament_rows))

    return _with_raw_tennis_columns(pd.DataFrame(rows))


def _wta_manifest_path(year: int) -> Path:
    return WTA_RAW_DIR / _wta_manifest_filename(year)


def _wta_progress_state_path(year: int) -> Path:
    return WTA_RAW_DIR / _wta_progress_state_filename(year)


def _wta_progress_rows_path(year: int) -> Path:
    return WTA_RAW_DIR / _wta_progress_rows_filename(year)


def _wta_manifest_progress_path(year: int) -> Path:
    return WTA_RAW_DIR / _wta_manifest_progress_filename(year)


def _load_cached_wta_manifest(year: int) -> list[dict[str, object]]:
    payload = _load_json_file(_wta_manifest_path(year))
    if not payload:
        return []
    if int(payload.get("schema_version") or 0) != WTA_MANIFEST_SCHEMA_VERSION:
        return []
    if int(payload.get("year") or 0) != int(year):
        return []

    tournaments = payload.get("tournaments", [])
    if not isinstance(tournaments, list):
        return []
    return [item for item in tournaments if isinstance(item, dict)]


def _save_wta_manifest(year: int, tournaments: list[dict[str, object]]) -> None:
    _write_json_file(
        _wta_manifest_path(year),
        {
            "schema_version": WTA_MANIFEST_SCHEMA_VERSION,
            "year": int(year),
            "tournaments": tournaments,
        },
    )


def _load_wta_manifest_progress(year: int) -> tuple[Optional[int], Optional[int], list[dict[str, object]]]:
    payload = _load_json_file(_wta_manifest_progress_path(year))
    if not payload:
        return None, None, []
    if int(payload.get("schema_version") or 0) != WTA_MANIFEST_SCHEMA_VERSION:
        return None, None, []
    if int(payload.get("year") or 0) != int(year):
        return None, None, []

    last_page = payload.get("last_page")
    next_page = payload.get("next_page")
    tournaments = payload.get("tournaments", [])
    if not isinstance(tournaments, list):
        tournaments = []
    return (
        int(last_page) if last_page is not None else None,
        int(next_page) if next_page is not None else None,
        [item for item in tournaments if isinstance(item, dict)],
    )


def _save_wta_manifest_progress(
    year: int,
    last_page: int,
    next_page: int,
    tournaments: list[dict[str, object]],
) -> None:
    _write_json_file(
        _wta_manifest_progress_path(year),
        {
            "schema_version": WTA_MANIFEST_SCHEMA_VERSION,
            "year": int(year),
            "last_page": int(last_page),
            "next_page": int(next_page),
            "tournaments": tournaments,
        },
    )


def _clear_wta_manifest_progress(year: int) -> None:
    path = _wta_manifest_progress_path(year)
    if path.exists():
        path.unlink()


def _load_wta_progress(year: int) -> tuple[set[str], list[dict[str, object]]]:
    state_payload = _load_json_file(_wta_progress_state_path(year))
    rows_path = _wta_progress_rows_path(year)
    if not state_payload or not rows_path.exists():
        return set(), []
    if int(state_payload.get("schema_version") or 0) != WTA_PROGRESS_SCHEMA_VERSION:
        return set(), []
    if int(state_payload.get("year") or 0) != int(year):
        return set(), []

    completed_ids = {
        str(tournament_id)
        for tournament_id in state_payload.get("completed_tournament_ids", [])
        if tournament_id is not None
    }
    progress_rows = pd.read_csv(rows_path).to_dict("records")
    return completed_ids, progress_rows


def _save_wta_progress(
    year: int,
    completed_tournament_ids: set[str],
    rows: list[dict[str, object]],
    tournaments: list[dict[str, object]],
    last_error: Optional[str] = None,
) -> None:
    rows_path = _wta_progress_rows_path(year)
    _with_raw_tennis_columns(pd.DataFrame(rows)).to_csv(rows_path, index=False)
    _write_json_file(
        _wta_progress_state_path(year),
        {
            "schema_version": WTA_PROGRESS_SCHEMA_VERSION,
            "year": int(year),
            "completed_tournament_ids": sorted(completed_tournament_ids),
            "total_tournaments": len(tournaments),
            "saved_rows": len(rows),
            "last_error": last_error,
        },
    )


def _clear_wta_progress(year: int) -> None:
    for path in [_wta_progress_rows_path(year), _wta_progress_state_path(year)]:
        if path.exists():
            path.unlink()


def _discover_wta_tournaments(year: int, session: requests.Session) -> list[dict[str, object]]:
    cached_manifest = _load_cached_wta_manifest(year)
    if cached_manifest:
        return cached_manifest

    last_page, next_page, tournaments = _load_wta_manifest_progress(year)
    seen: set[tuple[str, int]] = {
        (str(item.get("tournament_id")), int(item.get("year") or year))
        for item in tournaments
        if item.get("tournament_id") is not None
    }

    if last_page is None or next_page is None:
        index_response = _get_with_retries(
            session,
            WTA_TOURNAMENTS_API_URL.format(page=0, page_size=WTA_TOURNAMENTS_PAGE_SIZE),
            timeout=60,
        )
        index_response.raise_for_status()
        index_payload = index_response.json()
        total_entries = int(index_payload.get("pageInfo", {}).get("numEntries") or 0)
        if total_entries <= 0:
            return []
        last_page = max((total_entries - 1) // WTA_TOURNAMENTS_PAGE_SIZE, 0)
        next_page = last_page
        _save_wta_manifest_progress(year, last_page, next_page, tournaments)

    page = int(next_page)
    try:
        while page >= 0:
            response = _get_with_retries(
                session,
                WTA_TOURNAMENTS_API_URL.format(page=page, page_size=WTA_TOURNAMENTS_PAGE_SIZE),
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload.get("content", [])
            page_years = {int(item.get("year")) for item in content if item.get("year") is not None}
            stop_early = bool(page_years) and max(page_years) < int(year)

            for item in content:
                event_year = item.get("year")
                tournament_group = item.get("tournamentGroup", {})
                tournament_id = tournament_group.get("id")
                if event_year != int(year) or tournament_id is None:
                    continue
                key = (str(tournament_id), int(event_year))
                if key in seen:
                    continue
                seen.add(key)
                tournaments.append(
                    {
                        "tournament_id": str(tournament_id),
                        "slug": normalize_text(tournament_group.get("name")).replace(" ", "-"),
                        "year": int(event_year),
                        "tourney_name": _clean_text(
                            tournament_group.get("name")
                            or str(item.get("title") or "").split(" - ", 1)[0]
                            or ""
                        ).title(),
                        "tourney_level": item.get("level"),
                        "surface": item.get("surface") or "Unknown",
                        "draw_size": item.get("singlesDrawSize"),
                        "tourney_date": _parse_date_to_yyyymmdd(item.get("startDate"), int(event_year)),
                        "tourney_end_date": _parse_date_to_yyyymmdd(item.get("endDate"), int(event_year)),
                    }
                )

            page -= 1
            _save_wta_manifest_progress(year, int(last_page), page, tournaments)
            if stop_early:
                break
    except (requests.RequestException, ValueError):
        _save_wta_manifest_progress(year, int(last_page), page, tournaments)
        raise

    _save_wta_manifest(year, tournaments)
    _clear_wta_manifest_progress(year)
    return tournaments


def _fetch_wta_official_year(year: int, session: requests.Session) -> pd.DataFrame:
    tournaments = _discover_wta_tournaments(year, session)
    completed_tournament_ids, rows = _load_wta_progress(year)
    valid_tournament_ids = {str(tournament["tournament_id"]) for tournament in tournaments}
    completed_tournament_ids = {
        tournament_id for tournament_id in completed_tournament_ids if tournament_id in valid_tournament_ids
    }

    for tournament_ref in tournaments:
        tournament_id = str(tournament_ref["tournament_id"])
        if tournament_id in completed_tournament_ids:
            continue

        try:
            matches_response = _get_with_retries(
                session,
                WTA_TOURNAMENT_MATCHES_URL.format(tournament_id=tournament_id, year=int(year)),
                timeout=60,
            )
            if matches_response.status_code == 404:
                completed_tournament_ids.add(tournament_id)
                _save_wta_progress(year, completed_tournament_ids, rows, tournaments)
                continue
            matches_response.raise_for_status()
            payload = matches_response.json()
        except (requests.RequestException, ValueError) as exc:
            _save_wta_progress(year, completed_tournament_ids, rows, tournaments, last_error=str(exc))
            raise

        main_singles_matches = [
            match
            for match in payload.get("matches", [])
            if match.get("DrawMatchType") == "S"
            and match.get("DrawLevelType") == "M"
            and match.get("MatchState") == "F"
        ]
        effective_draw_size = _infer_wta_draw_size(tournament_ref, main_singles_matches)
        tournament_rows: list[dict[str, object]] = []
        for match in main_singles_matches:
            player_a = _clean_text(f"{match.get('PlayerNameFirstA', '')} {match.get('PlayerNameLastA', '')}")
            player_b = _clean_text(f"{match.get('PlayerNameFirstB', '')} {match.get('PlayerNameLastB', '')}")
            if not player_a or not player_b:
                continue

            winner_side = _infer_wta_winner_side(match)
            if winner_side == "A":
                winner_name, loser_name = player_a, player_b
                winner_id, loser_id = match.get("PlayerIDA"), match.get("PlayerIDB")
                winner_seed_text, loser_seed_text = match.get("SeedA"), match.get("SeedB")
                winner_entry_text, loser_entry_text = match.get("EntryTypeA"), match.get("EntryTypeB")
            elif winner_side == "B":
                winner_name, loser_name = player_b, player_a
                winner_id, loser_id = match.get("PlayerIDB"), match.get("PlayerIDA")
                winner_seed_text, loser_seed_text = match.get("SeedB"), match.get("SeedA")
                winner_entry_text, loser_entry_text = match.get("EntryTypeB"), match.get("EntryTypeA")
            else:
                continue

            winner_seed, winner_entry = _parse_seed_or_entry(winner_seed_text)
            loser_seed, loser_entry = _parse_seed_or_entry(loser_seed_text)
            round_code = _map_wta_round_id(
                match.get("RoundID"),
                effective_draw_size,
                tournament_level=tournament_ref.get("tourney_level"),
                tournament_name=tournament_ref.get("tourney_name"),
            )
            row = _build_match_row(
                tour="wta",
                tourney_id=f"{year}-{tournament_id}",
                tourney_name=str(tournament_ref.get("tourney_name") or tournament_ref["slug"].replace("-", " ").title()),
                tourney_level=tournament_ref.get("tourney_level"),
                surface=tournament_ref.get("surface") or "Unknown",
                draw_size=effective_draw_size or tournament_ref.get("draw_size"),
                tourney_date=int(tournament_ref.get("tourney_date") or _parse_date_to_yyyymmdd(None, int(year))),
                round_code=round_code,
                best_of=3,
                score=str(match.get("ScoreString") or "").replace(",", " "),
                winner_name=winner_name,
                loser_name=loser_name,
                winner_id=winner_id,
                loser_id=loser_id,
                winner_seed=winner_seed,
                loser_seed=loser_seed,
                winner_entry=winner_entry,
                loser_entry=loser_entry,
            )
            row["_sort_time"] = str(match.get("MatchTimeStamp") or "")
            row["_sort_id"] = str(match.get("MatchID") or "")
            tournament_rows.append(row)

        completed_tournament = bool(tournament_rows)
        if not completed_tournament:
            completed_tournament = int(tournament_ref.get("tourney_end_date") or 0) < _today_yyyymmdd()
        if completed_tournament:
            rows.extend(_finalize_tournament_rows(tournament_rows))
            completed_tournament_ids.add(tournament_id)
            _save_wta_progress(year, completed_tournament_ids, rows, tournaments)
        time.sleep(OFFICIAL_REQUEST_PAUSE_SECONDS)

    return _with_raw_tennis_columns(pd.DataFrame(rows))


def download_official_matches(
    tour: str,
    years: Iterable[int],
    force: bool = False,
    session: Optional[requests.Session] = None,
) -> list[Path]:
    """Download post-Sackmann ATP/WTA singles match files for one tour."""
    session = _request_session(session)
    fetcher = _fetch_atp_official_year if str(tour).lower() == "atp" else _fetch_wta_official_year
    saved_paths: list[Path] = []

    for year in years:
        destination = _raw_dir_for_tour(tour) / _official_match_filename(tour, year)
        if destination.exists() and not force:
            saved_paths.append(destination)
            continue

        try:
            frame = fetcher(int(year), session)
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Failed to fetch official %s %s matches: %s", str(tour).upper(), int(year), exc)
            continue
        if frame.empty:
            logger.warning("No official %s matches found for %s", str(tour).upper(), int(year))
            continue

        _with_raw_tennis_columns(frame).to_csv(destination, index=False)
        if str(tour).lower() == "wta":
            _clear_wta_progress(int(year))
        saved_paths.append(destination)
        logger.info("Saved official %s %s raw matches to %s", str(tour).upper(), int(year), destination)

    return saved_paths


def load_official_matches(
    tour: str,
    years: Iterable[int],
    force_download: bool = False,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Load cached post-Sackmann ATP/WTA singles matches for one tour."""
    files = download_official_matches(
        tour=tour,
        years=years,
        force=force_download,
        session=session,
    )

    frames: list[pd.DataFrame] = []
    for csv_path in files:
        frame = pd.read_csv(csv_path)
        frame["tour"] = str(tour).lower()
        frame["source_file"] = csv_path.name
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=RAW_TENNIS_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %s official %s matches from %s yearly files", len(combined), str(tour).upper(), len(frames))
    return combined


def load_tennis_matches(
    tours: Iterable[str] = ("atp", "wta"),
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load ATP and WTA singles match history into one raw DataFrame."""
    session = _request_session()
    frames: list[pd.DataFrame] = []
    effective_end_year = min(int(end_year), _current_tennis_year()) if end_year is not None else _current_tennis_year()

    for tour in tours:
        available_years = list_available_sackmann_years(tour, session=session)
        latest_sackmann_year = max(available_years) if available_years else None

        chosen_sackmann_years = [
            year
            for year in available_years
            if (start_year is None or year >= start_year)
            and year <= effective_end_year
        ]
        if chosen_sackmann_years:
            frames.append(
                load_sackmann_matches(
                    tour=tour,
                    years=chosen_sackmann_years,
                    force_download=force_download,
                    session=session,
                )
            )

        official_start_year = max((latest_sackmann_year or 0) + 1, int(start_year) if start_year is not None else 0)
        if official_start_year <= effective_end_year:
            official_years = list(range(official_start_year, effective_end_year + 1))
            official_frame = load_official_matches(
                tour=tour,
                years=official_years,
                force_download=force_download,
                session=session,
            )
            if not official_frame.empty:
                frames.append(official_frame)

        if not chosen_sackmann_years and (latest_sackmann_year is None or official_start_year > effective_end_year):
            logger.warning("No %s years matched requested range", tour.upper())

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _score_is_trainable(score: object) -> bool:
    if score is None or pd.isna(score):
        return False
    score_text = str(score).strip()
    if not score_text:
        return False
    return TRAINING_EXCLUSION_PATTERN.search(score_text) is None


def _player_order_key(name: object, player_id: object) -> tuple[str, str]:
    player_id_text = "" if player_id is None or pd.isna(player_id) else str(player_id)
    return normalize_player_name(name), player_id_text


def _orient_players(row: pd.Series) -> dict[str, object]:
    winner_first = _player_order_key(row.get("winner_name"), row.get("winner_id")) <= _player_order_key(
        row.get("loser_name"),
        row.get("loser_id"),
    )

    def _pick(winner_col: str, loser_col: str) -> tuple[object, object]:
        if winner_first:
            return row.get(winner_col), row.get(loser_col)
        return row.get(loser_col), row.get(winner_col)

    player_a, player_b = _pick("winner_name", "loser_name")
    player_a_id, player_b_id = _pick("winner_id", "loser_id")
    player_a_rank, player_b_rank = _pick("winner_rank", "loser_rank")
    player_a_rank_points, player_b_rank_points = _pick("winner_rank_points", "loser_rank_points")
    player_a_age, player_b_age = _pick("winner_age", "loser_age")
    player_a_hand, player_b_hand = _pick("winner_hand", "loser_hand")
    player_a_height_cm, player_b_height_cm = _pick("winner_ht", "loser_ht")
    player_a_seed, player_b_seed = _pick("winner_seed", "loser_seed")
    player_a_entry, player_b_entry = _pick("winner_entry", "loser_entry")

    return {
        "player_a": player_a,
        "player_b": player_b,
        "player_a_id": player_a_id,
        "player_b_id": player_b_id,
        "player_a_rank": player_a_rank,
        "player_b_rank": player_b_rank,
        "player_a_rank_points": player_a_rank_points,
        "player_b_rank_points": player_b_rank_points,
        "player_a_age": player_a_age,
        "player_b_age": player_b_age,
        "player_a_hand": player_a_hand,
        "player_b_hand": player_b_hand,
        "player_a_height_cm": player_a_height_cm,
        "player_b_height_cm": player_b_height_cm,
        "player_a_seed": player_a_seed,
        "player_b_seed": player_b_seed,
        "player_a_entry": player_a_entry,
        "player_b_entry": player_b_entry,
        "target": int(player_a == row.get("winner_name")),
    }


def normalize_tennis_matches(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Jeff Sackmann ATP/WTA singles files into one training schema."""
    if raw_df.empty:
        return raw_df.copy()

    frame = _with_raw_tennis_columns(raw_df)
    frame["event_date"] = pd.to_datetime(frame["tourney_date"], format="%Y%m%d", errors="coerce")
    frame = frame.dropna(subset=["event_date", "winner_name", "loser_name"]).copy()
    frame = frame[frame["score"].map(_score_is_trainable)].copy()
    frame = frame.reset_index(drop=True)
    frame["tour"] = frame["tour"].astype(str).str.lower()
    frame["surface"] = frame["surface"].fillna("Unknown")
    frame["best_of"] = pd.to_numeric(frame["best_of"], errors="coerce")
    frame["winner_rank"] = pd.to_numeric(frame["winner_rank"], errors="coerce")
    frame["loser_rank"] = pd.to_numeric(frame["loser_rank"], errors="coerce")
    frame["winner_rank_points"] = pd.to_numeric(frame["winner_rank_points"], errors="coerce")
    frame["loser_rank_points"] = pd.to_numeric(frame["loser_rank_points"], errors="coerce")
    frame["winner_age"] = pd.to_numeric(frame["winner_age"], errors="coerce")
    frame["loser_age"] = pd.to_numeric(frame["loser_age"], errors="coerce")
    frame["winner_ht"] = pd.to_numeric(frame["winner_ht"], errors="coerce")
    frame["loser_ht"] = pd.to_numeric(frame["loser_ht"], errors="coerce")
    frame["match_num"] = pd.to_numeric(frame["match_num"], errors="coerce")
    frame["minutes"] = pd.to_numeric(frame["minutes"], errors="coerce")
    frame["tournament_seed"] = frame["tourney_name"].map(derive_tournament_seed)

    oriented_rows = frame.apply(_orient_players, axis=1, result_type="expand")
    normalized = pd.concat([frame, oriented_rows], axis=1)

    normalized = normalized[
        [
            "tour",
            "event_date",
            "tourney_id",
            "tourney_name",
            "tournament_seed",
            "tourney_level",
            "surface",
            "draw_size",
            "match_num",
            "round",
            "best_of",
            "minutes",
            "score",
            "winner_name",
            "loser_name",
            "winner_rank",
            "loser_rank",
            "winner_rank_points",
            "loser_rank_points",
            "player_a",
            "player_b",
            "target",
            "player_a_id",
            "player_b_id",
            "player_a_rank",
            "player_b_rank",
            "player_a_rank_points",
            "player_b_rank_points",
            "player_a_age",
            "player_b_age",
            "player_a_hand",
            "player_b_hand",
            "player_a_height_cm",
            "player_b_height_cm",
            "player_a_seed",
            "player_b_seed",
            "player_a_entry",
            "player_b_entry",
        ]
    ].rename(columns={"winner_name": "winner"})

    normalized = normalized.sort_values(["event_date", "tour", "tourney_name", "match_num"]).reset_index(drop=True)
    logger.info(
        "Normalized %s tennis matches after exclusions",
        len(normalized),
    )
    return normalized


def save_processed_tennis_data(df: pd.DataFrame, filename: str = "matches.csv") -> Path:
    """Save normalized tennis data to the processed tennis directory."""
    path = TENNIS_PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    logger.info("Saved processed tennis data to %s", path)
    return path


def load_processed_tennis_data(filename: str = "matches.csv") -> pd.DataFrame:
    """Load a processed tennis dataset from disk."""
    path = TENNIS_PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Processed tennis data not found: {path}")
    return pd.read_csv(path, parse_dates=["event_date"])


def prepare_tennis_data(
    tours: Iterable[str] = ("atp", "wta"),
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Download, normalize, and save ATP/WTA singles history."""
    raw_matches = load_tennis_matches(
        tours=tours,
        start_year=start_year,
        end_year=end_year,
        force_download=force_download,
    )
    normalized = normalize_tennis_matches(raw_matches)
    save_processed_tennis_data(normalized)
    return normalized
