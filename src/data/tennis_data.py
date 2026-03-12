"""Tennis data ingestion and normalization for ATP and WTA singles."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

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
TRAINING_EXCLUSION_PATTERN = re.compile(
    r"\b(?:RET|W/O|WO|DEF|ABN|ABD|walkover|default|abandoned|cancelled)\b",
    re.IGNORECASE,
)
STANDARD_MATCH_FILE_RE = re.compile(r"^(atp|wta)_matches_(\d{4})\.csv$")
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


def list_available_sackmann_years(tour: str, session: Optional[requests.Session] = None) -> list[int]:
    """List yearly Jeff Sackmann match files available for a tour."""
    session = session or requests.Session()
    repo_name = _tour_repo_name(tour)
    response = session.get(f"{SACKMANN_GITHUB_API}/{repo_name}/contents", timeout=30)
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
    session = session or requests.Session()
    saved_paths: list[Path] = []

    for year in years:
        destination = _raw_dir_for_tour(tour) / _match_filename(tour, year)
        if destination.exists() and not force:
            saved_paths.append(destination)
            continue

        url = _match_url(tour, year)
        response = session.get(url, timeout=60)
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
    session = session or requests.Session()
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


def load_tennis_matches(
    tours: Iterable[str] = ("atp", "wta"),
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load ATP and WTA singles match history into one raw DataFrame."""
    session = requests.Session()
    frames: list[pd.DataFrame] = []

    for tour in tours:
        available_years = list_available_sackmann_years(tour, session=session)
        chosen_years = [
            year
            for year in available_years
            if (start_year is None or year >= start_year)
            and (end_year is None or year <= end_year)
        ]
        if not chosen_years:
            logger.warning("No %s years matched requested range", tour.upper())
            continue
        frames.append(
            load_sackmann_matches(
                tour=tour,
                years=chosen_years,
                force_download=force_download,
                session=session,
            )
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _score_is_trainable(score: object) -> bool:
    score_text = str(score or "").strip()
    if not score_text:
        return False
    return TRAINING_EXCLUSION_PATTERN.search(score_text) is None


def _player_order_key(name: object, player_id: object) -> tuple[str, str]:
    return normalize_player_name(name), str(player_id or "")


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

    frame = raw_df.copy()
    frame["event_date"] = pd.to_datetime(frame["tourney_date"], format="%Y%m%d", errors="coerce")
    frame = frame.dropna(subset=["event_date", "winner_name", "loser_name"]).copy()
    frame = frame[frame["score"].map(_score_is_trainable)].copy()
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
    normalized = pd.concat([frame.reset_index(drop=True), oriented_rows], axis=1)

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
