"""Leak-free tennis feature engineering for ATP and WTA singles."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    ELO_INITIAL,
    TENNIS_ELO_K_FACTOR,
    TENNIS_ELO_SURFACE_K,
    TENNIS_MIN_MATCHES,
)
from src.data.tennis_data import (
    collect_tournament_seed_candidates,
    derive_tournament_seed,
    normalize_player_name,
    tournament_seeds_compatible,
)

logger = logging.getLogger(__name__)

RECENT_FORM_WINDOW = 5
RANK_TREND_WINDOW = 3
STAGE1_TENNIS_MODEL_COLUMNS = [
    "diff_elo",
    "diff_surface_elo",
    "best_of_5",
]
STAGE2_TENNIS_MODEL_COLUMNS = [
    "diff_elo",
    "diff_surface_elo",
    "best_of_5",
    "log_rank_diff",
    "log_rank_points_diff",
    "diff_age",
]
TENNIS_ENGINEERED_FEATURE_COLUMNS = [
    "a_elo",
    "b_elo",
    "diff_elo",
    "a_surface_elo",
    "b_surface_elo",
    "diff_surface_elo",
    "a_recent_form",
    "b_recent_form",
    "diff_recent_form",
    "a_surface_win_rate",
    "b_surface_win_rate",
    "diff_surface_win_rate",
    "a_h2h_wins",
    "b_h2h_wins",
    "diff_h2h_wins",
    "a_days_since_last_match",
    "b_days_since_last_match",
    "diff_days_since_last_match",
    "a_matches_last_7d",
    "b_matches_last_7d",
    "diff_matches_last_7d",
    "a_matches_last_30d",
    "b_matches_last_30d",
    "diff_matches_last_30d",
    "a_rank",
    "b_rank",
    "diff_rank",
    "a_rank_points",
    "b_rank_points",
    "diff_rank_points",
    "a_rank_trend",
    "b_rank_trend",
    "diff_rank_trend",
    "log_rank_diff",
    "log_rank_points_diff",
    "best_of",
    "best_of_5",
    "a_age",
    "b_age",
    "diff_age",
    "a_is_left_handed",
    "b_is_left_handed",
    "diff_is_left_handed",
    "same_hand",
    "surface_is_hard",
    "surface_is_clay",
    "surface_is_grass",
    "surface_is_carpet",
    "a_num_matches",
    "b_num_matches",
    "diff_num_matches",
]
ATP_GRAND_SLAM_SEEDS = {
    derive_tournament_seed(name)
    for name in [
        "Australian Open",
        "French Open",
        "Roland Garros",
        "Wimbledon",
        "US Open",
        "United States Open",
    ]
}


def normalize_surface(surface: object) -> str:
    """Normalize a tennis surface label."""
    surface_norm = str(surface or "").strip().lower()
    if surface_norm in {"hard", "clay", "grass", "carpet"}:
        return surface_norm
    return "unknown"


def _coerce_event_timestamp(value: object) -> pd.Timestamp:
    """Convert history and live event timestamps into a common naive date basis."""
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT
    if getattr(timestamp, "tzinfo", None) is not None:
        timestamp = timestamp.tz_localize(None)
    return pd.Timestamp(timestamp).normalize()


def _player_state_key(tour: object, player_name: object) -> tuple[str, str]:
    return str(tour or "").strip().lower(), normalize_player_name(player_name)


class SurfaceAwareElo:
    """Overall and per-surface Elo tracker."""

    def __init__(
        self,
        initial_rating: float = ELO_INITIAL,
        k_factor: float = TENNIS_ELO_K_FACTOR,
        surface_k_factor: float = TENNIS_ELO_SURFACE_K,
    ):
        self.initial_rating = float(initial_rating)
        self.k_factor = float(k_factor)
        self.surface_k_factor = float(surface_k_factor)
        self.overall: dict[tuple[str, str], float] = {}
        self.surface: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)

    def get_overall(self, player_key: tuple[str, str]) -> float:
        return self.overall.get(player_key, self.initial_rating)

    def get_surface(self, player_key: tuple[str, str], surface: str) -> float:
        return self.surface[surface].get(player_key, self.initial_rating)

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(
        self,
        player_a: tuple[str, str],
        player_b: tuple[str, str],
        winner_key: tuple[str, str],
        surface: str,
    ) -> None:
        rating_a = self.get_overall(player_a)
        rating_b = self.get_overall(player_b)
        expected_a = self.expected_score(rating_a, rating_b)
        actual_a = 1.0 if winner_key == player_a else 0.0

        self.overall[player_a] = rating_a + self.k_factor * (actual_a - expected_a)
        self.overall[player_b] = rating_b + self.k_factor * ((1.0 - actual_a) - (1.0 - expected_a))

        surface_rating_a = self.get_surface(player_a, surface)
        surface_rating_b = self.get_surface(player_b, surface)
        surface_expected_a = self.expected_score(surface_rating_a, surface_rating_b)

        self.surface[surface][player_a] = surface_rating_a + self.surface_k_factor * (actual_a - surface_expected_a)
        self.surface[surface][player_b] = surface_rating_b + self.surface_k_factor * (
            (1.0 - actual_a) - (1.0 - surface_expected_a)
        )


class TennisFeatureBuilder:
    """Stateful no-leakage tennis feature builder."""

    def __init__(self):
        self.elo = SurfaceAwareElo()
        self.match_counts: dict[tuple[str, str], int] = defaultdict(int)
        self.recent_results: dict[tuple[str, str], deque[int]] = defaultdict(
            lambda: deque(maxlen=RECENT_FORM_WINDOW)
        )
        self.surface_results: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0])
        )
        self.h2h: dict[tuple[tuple[str, str], tuple[str, str]], dict[tuple[str, str], int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.last_event_date: dict[tuple[str, str], pd.Timestamp] = {}
        self.recent_dates: dict[tuple[str, str], deque[pd.Timestamp]] = defaultdict(deque)
        self.rank_history: dict[tuple[str, str], deque[float]] = defaultdict(
            lambda: deque(maxlen=RANK_TREND_WINDOW)
        )
        self.last_rank: dict[tuple[str, str], float] = {}
        self.last_rank_points: dict[tuple[str, str], float] = {}
        self.last_age: dict[tuple[str, str], float] = {}
        self.last_hand: dict[tuple[str, str], str] = {}

    def ingest_history(self, matches_df: pd.DataFrame) -> None:
        """Advance the builder state through historical matches."""
        if matches_df.empty:
            return
        sorted_matches = matches_df.sort_values(["event_date", "tour", "tourney_name", "match_num"])
        for row in sorted_matches.to_dict("records"):
            self.update_from_row(row)

    def _clean_recent_dates(self, player_key: tuple[str, str], event_date: pd.Timestamp) -> deque[pd.Timestamp]:
        queue = self.recent_dates[player_key]
        while queue and (event_date - queue[0]).days > 30:
            queue.popleft()
        return queue

    def _player_snapshot(
        self,
        tour: object,
        player_name: str,
        event_date: pd.Timestamp,
        surface: str,
        current_rank: Optional[float] = None,
        current_rank_points: Optional[float] = None,
        current_age: Optional[float] = None,
        current_hand: Optional[str] = None,
    ) -> dict[str, object]:
        player_key = _player_state_key(tour, player_name)
        recent_dates = self._clean_recent_dates(player_key, event_date)
        last_date = self.last_event_date.get(player_key)
        history_ranks = list(self.rank_history[player_key])

        rank_value = current_rank
        if pd.isna(rank_value):
            rank_value = self.last_rank.get(player_key, np.nan)

        rank_points_value = current_rank_points
        if pd.isna(rank_points_value):
            rank_points_value = self.last_rank_points.get(player_key, np.nan)

        age_value = current_age
        if pd.isna(age_value):
            age_value = self.last_age.get(player_key, np.nan)

        hand_value = current_hand if isinstance(current_hand, str) and current_hand else self.last_hand.get(player_key, "")
        surface_wins, surface_matches = self.surface_results[player_key][surface]

        snapshot = {
            "elo": self.elo.get_overall(player_key),
            "surface_elo": self.elo.get_surface(player_key, surface),
            "recent_form": float(mean(self.recent_results[player_key])) if self.recent_results[player_key] else np.nan,
            "surface_win_rate": (surface_wins / surface_matches) if surface_matches else np.nan,
            "num_matches": self.match_counts.get(player_key, 0),
            "days_since_last_match": float((event_date - last_date).days) if last_date is not None else np.nan,
            "matches_last_7d": float(sum((event_date - prior_date).days <= 7 for prior_date in recent_dates)),
            "matches_last_30d": float(len(recent_dates)),
            "rank": float(rank_value) if not pd.isna(rank_value) else np.nan,
            "rank_points": float(rank_points_value) if not pd.isna(rank_points_value) else np.nan,
            "rank_trend": float(mean(history_ranks) - rank_value)
            if history_ranks and not pd.isna(rank_value)
            else np.nan,
            "age": float(age_value) if not pd.isna(age_value) else np.nan,
            "hand": hand_value,
            "is_left_handed": float(str(hand_value).upper() == "L"),
        }
        return snapshot

    def snapshot_match(self, row: dict[str, object]) -> dict[str, object]:
        """Build a feature row without mutating state."""
        event_date = _coerce_event_timestamp(row.get("event_date") or row.get("commence_time"))
        surface = normalize_surface(row.get("surface"))
        tour = str(row.get("tour") or "").strip().lower()
        player_a = str(row.get("player_a") or row.get("fighter_a") or "")
        player_b = str(row.get("player_b") or row.get("fighter_b") or "")
        player_a_rank = row.get("player_a_rank", row.get("a_rank"))
        player_b_rank = row.get("player_b_rank", row.get("b_rank"))
        player_a_rank_points = row.get("player_a_rank_points", row.get("a_rank_points"))
        player_b_rank_points = row.get("player_b_rank_points", row.get("b_rank_points"))
        player_a_age = row.get("player_a_age", row.get("a_age"))
        player_b_age = row.get("player_b_age", row.get("b_age"))
        player_a_hand = row.get("player_a_hand", row.get("a_hand"))
        player_b_hand = row.get("player_b_hand", row.get("b_hand"))
        best_of = pd.to_numeric(pd.Series([row.get("best_of", 3)]), errors="coerce").iloc[0]

        a_snapshot = self._player_snapshot(
            tour=tour,
            player_name=player_a,
            event_date=event_date,
            surface=surface,
            current_rank=player_a_rank,
            current_rank_points=player_a_rank_points,
            current_age=player_a_age,
            current_hand=player_a_hand,
        )
        b_snapshot = self._player_snapshot(
            tour=tour,
            player_name=player_b,
            event_date=event_date,
            surface=surface,
            current_rank=player_b_rank,
            current_rank_points=player_b_rank_points,
            current_age=player_b_age,
            current_hand=player_b_hand,
        )

        a_key = _player_state_key(tour, player_a)
        b_key = _player_state_key(tour, player_b)
        h2h_key = tuple(sorted([a_key, b_key]))
        h2h_stats = self.h2h[h2h_key]
        a_h2h = h2h_stats.get(a_key, 0)
        b_h2h = h2h_stats.get(b_key, 0)

        features = {
            "fighter_a": player_a,
            "fighter_b": player_b,
            "surface": surface,
            "best_of": float(best_of) if not pd.isna(best_of) else np.nan,
            "best_of_5": float(best_of == 5) if not pd.isna(best_of) else 0.0,
            "surface_is_hard": float(surface == "hard"),
            "surface_is_clay": float(surface == "clay"),
            "surface_is_grass": float(surface == "grass"),
            "surface_is_carpet": float(surface == "carpet"),
            "a_h2h_wins": float(a_h2h),
            "b_h2h_wins": float(b_h2h),
            "diff_h2h_wins": float(a_h2h - b_h2h),
            "same_hand": float(a_snapshot["hand"] == b_snapshot["hand"] and bool(a_snapshot["hand"])),
        }

        for prefix, snapshot in [("a", a_snapshot), ("b", b_snapshot)]:
            features[f"{prefix}_elo"] = snapshot["elo"]
            features[f"{prefix}_surface_elo"] = snapshot["surface_elo"]
            features[f"{prefix}_recent_form"] = snapshot["recent_form"]
            features[f"{prefix}_surface_win_rate"] = snapshot["surface_win_rate"]
            features[f"{prefix}_num_matches"] = snapshot["num_matches"]
            features[f"{prefix}_days_since_last_match"] = snapshot["days_since_last_match"]
            features[f"{prefix}_matches_last_7d"] = snapshot["matches_last_7d"]
            features[f"{prefix}_matches_last_30d"] = snapshot["matches_last_30d"]
            features[f"{prefix}_rank"] = snapshot["rank"]
            features[f"{prefix}_rank_points"] = snapshot["rank_points"]
            features[f"{prefix}_rank_trend"] = snapshot["rank_trend"]
            features[f"{prefix}_age"] = snapshot["age"]
            features[f"{prefix}_is_left_handed"] = snapshot["is_left_handed"]

        diff_pairs = {
            "diff_elo": ("a_elo", "b_elo"),
            "diff_surface_elo": ("a_surface_elo", "b_surface_elo"),
            "diff_recent_form": ("a_recent_form", "b_recent_form"),
            "diff_surface_win_rate": ("a_surface_win_rate", "b_surface_win_rate"),
            "diff_num_matches": ("a_num_matches", "b_num_matches"),
            "diff_days_since_last_match": ("a_days_since_last_match", "b_days_since_last_match"),
            "diff_matches_last_7d": ("a_matches_last_7d", "b_matches_last_7d"),
            "diff_matches_last_30d": ("a_matches_last_30d", "b_matches_last_30d"),
            "diff_rank": ("a_rank", "b_rank"),
            "diff_rank_points": ("a_rank_points", "b_rank_points"),
            "diff_rank_trend": ("a_rank_trend", "b_rank_trend"),
            "diff_age": ("a_age", "b_age"),
            "diff_is_left_handed": ("a_is_left_handed", "b_is_left_handed"),
        }
        for feature_name, (a_col, b_col) in diff_pairs.items():
            features[feature_name] = features[a_col] - features[b_col]

        a_rank = features["a_rank"]
        b_rank = features["b_rank"]
        a_rank_points = features["a_rank_points"]
        b_rank_points = features["b_rank_points"]
        features["log_rank_diff"] = np.log1p(b_rank) - np.log1p(a_rank) if pd.notna(a_rank) and pd.notna(b_rank) else np.nan
        features["log_rank_points_diff"] = (
            np.log1p(a_rank_points) - np.log1p(b_rank_points)
            if pd.notna(a_rank_points) and pd.notna(b_rank_points)
            else np.nan
        )

        return features

    def update_from_row(self, row: dict[str, object]) -> None:
        """Update state after a completed match."""
        surface = normalize_surface(row.get("surface"))
        tour = str(row.get("tour") or "").strip().lower()
        player_a = str(row.get("player_a") or row.get("fighter_a") or "")
        player_b = str(row.get("player_b") or row.get("fighter_b") or "")
        a_key = _player_state_key(tour, player_a)
        b_key = _player_state_key(tour, player_b)
        winner_key = _player_state_key(tour, row.get("winner"))
        event_date = _coerce_event_timestamp(row.get("event_date") or row.get("commence_time"))

        if not a_key[1] or not b_key[1] or not winner_key[1]:
            return

        self.elo.update(a_key, b_key, winner_key=winner_key, surface=surface)
        self.match_counts[a_key] += 1
        self.match_counts[b_key] += 1
        self.recent_results[a_key].append(1 if winner_key == a_key else 0)
        self.recent_results[b_key].append(1 if winner_key == b_key else 0)

        self.surface_results[a_key][surface][1] += 1
        self.surface_results[b_key][surface][1] += 1
        if winner_key == a_key:
            self.surface_results[a_key][surface][0] += 1
        if winner_key == b_key:
            self.surface_results[b_key][surface][0] += 1

        self.recent_dates[a_key].append(event_date)
        self.recent_dates[b_key].append(event_date)
        self.last_event_date[a_key] = event_date
        self.last_event_date[b_key] = event_date

        pair_key = tuple(sorted([a_key, b_key]))
        self.h2h[pair_key][winner_key] += 1

        for player_key, rank, rank_points, age, hand in [
            (
                a_key,
                row.get("player_a_rank", row.get("a_rank")),
                row.get("player_a_rank_points", row.get("a_rank_points")),
                row.get("player_a_age", row.get("a_age")),
                row.get("player_a_hand", row.get("a_hand")),
            ),
            (
                b_key,
                row.get("player_b_rank", row.get("b_rank")),
                row.get("player_b_rank_points", row.get("b_rank_points")),
                row.get("player_b_age", row.get("b_age")),
                row.get("player_b_hand", row.get("b_hand")),
            ),
        ]:
            if not pd.isna(rank):
                self.rank_history[player_key].append(float(rank))
                self.last_rank[player_key] = float(rank)
            if not pd.isna(rank_points):
                self.last_rank_points[player_key] = float(rank_points)
            if not pd.isna(age):
                self.last_age[player_key] = float(age)
            if isinstance(hand, str) and hand:
                self.last_hand[player_key] = hand


def build_tennis_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Build leak-free tennis features from normalized match history."""
    if matches_df.empty:
        return matches_df.copy()

    sorted_matches = matches_df.sort_values(["event_date", "tour", "tourney_name", "match_num"]).reset_index(drop=True)
    builder = TennisFeatureBuilder()
    feature_rows = []

    for row in sorted_matches.to_dict("records"):
        feature_rows.append(builder.snapshot_match(row))
        builder.update_from_row(row)

    feature_frame = pd.DataFrame(feature_rows)
    feature_frame = feature_frame.drop(columns=[column for column in feature_frame.columns if column in sorted_matches.columns])
    features_df = pd.concat([sorted_matches, feature_frame], axis=1)
    logger.info("Built %s tennis feature rows with %s columns", len(features_df), len(features_df.columns))
    return features_df


def save_tennis_features(features_df: pd.DataFrame, path: Optional[str] = None) -> str:
    """Save tennis features under the processed tennis directory."""
    output_path = Path(path) if path else Path("data/processed/tennis/features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    logger.info("Saved tennis features to %s", output_path)
    return str(output_path)


def get_tennis_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return all engineered tennis feature columns available in a frame.

    This is broader than the saved Stage 1 model contract, which is exposed via
    `get_stage1_tennis_feature_columns()`.
    """
    return [column for column in TENNIS_ENGINEERED_FEATURE_COLUMNS if column in features_df.columns]


def get_stage1_tennis_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return Stage 1 saved-model features present in a frame."""
    return [column for column in STAGE1_TENNIS_MODEL_COLUMNS if column in features_df.columns]


def get_stage2_tennis_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return Stage 2 saved-model features present in a frame."""
    return [column for column in STAGE2_TENNIS_MODEL_COLUMNS if column in features_df.columns]


def require_stage1_tennis_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return the full Stage 1 feature contract or raise if any column is missing."""
    missing = [column for column in STAGE1_TENNIS_MODEL_COLUMNS if column not in features_df.columns]
    if missing:
        raise ValueError(
            "Missing required Stage 1 tennis feature columns: "
            + ", ".join(missing)
        )
    return list(STAGE1_TENNIS_MODEL_COLUMNS)


def require_stage2_tennis_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return the full Stage 2 feature contract or raise if any column is missing."""
    missing = [column for column in STAGE2_TENNIS_MODEL_COLUMNS if column not in features_df.columns]
    if missing:
        raise ValueError(
            "Missing required Stage 2 tennis feature columns: "
            + ", ".join(missing)
        )
    return list(STAGE2_TENNIS_MODEL_COLUMNS)


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return not pd.isna(value)


def _matchup_seed_candidates(row: dict[str, object]) -> list[str]:
    return collect_tournament_seed_candidates(
        row.get("tournament_seed"),
        row.get("tournament_name"),
        row.get("sport_key"),
        row.get("event_title"),
    )


def _is_atp_grand_slam(row: dict[str, object]) -> bool:
    if str(row.get("tour", "")).lower() != "atp":
        return False
    if str(row.get("tourney_level", "")).upper() == "G":
        return True

    for seed in _matchup_seed_candidates(row):
        if seed in ATP_GRAND_SLAM_SEEDS:
            return True
        seed_tokens = set(seed.split())
        for slam_seed in ATP_GRAND_SLAM_SEEDS:
            slam_tokens = set(slam_seed.split())
            if slam_tokens and slam_tokens.issubset(seed_tokens):
                return True
    return False


def _default_live_best_of(row: dict[str, object]) -> int:
    return 5 if _is_atp_grand_slam(row) else 3


def _merge_context_value(row: dict[str, object], key: str, value: object) -> None:
    if not _has_value(row.get(key)) and _has_value(value):
        row[key] = value


def build_tournament_context(history_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize historical tournament context for live ATP/WTA matches."""
    if history_df.empty:
        return pd.DataFrame(columns=["tour", "tournament_seed", "surface", "best_of", "tourney_level"])

    enriched = history_df.copy()
    if "tour" not in enriched.columns:
        enriched["tour"] = ""
    if "tournament_seed" not in enriched.columns:
        enriched["tournament_seed"] = enriched["tourney_name"].map(derive_tournament_seed)

    def _mode(series: pd.Series):
        non_null = series.dropna()
        if non_null.empty:
            return np.nan
        return non_null.mode().iloc[0]

    context = (
        enriched.groupby(["tour", "tournament_seed"], dropna=False)
        .agg(
            surface=("surface", _mode),
            best_of=("best_of", _mode),
            tourney_level=("tourney_level", _mode),
            tournament_name=("tourney_name", _mode),
        )
        .reset_index()
    )
    return context


def enrich_matchups_with_context(matchups_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """Fill live matchup surface and best-of using historical tournament context."""
    if matchups_df.empty:
        return matchups_df.copy()

    context = build_tournament_context(history_df)
    lookup = context.set_index(["tour", "tournament_seed"]).to_dict("index") if not context.empty else {}
    enriched_rows = []
    for row in matchups_df.to_dict("records"):
        enriched = dict(row)
        tour = str(enriched.get("tour", "")).lower()
        seed_candidates = _matchup_seed_candidates(enriched)

        matched_context = None
        for seed in seed_candidates:
            key = (tour, seed)
            if key in lookup:
                matched_context = lookup[key]
                break

            for (context_tour, context_seed), context_row in lookup.items():
                if tour and context_tour and context_tour != tour:
                    continue
                if tournament_seeds_compatible(seed, context_seed):
                    matched_context = context_row
                    break
            if matched_context is not None:
                break

        if matched_context is not None:
            _merge_context_value(enriched, "surface", matched_context.get("surface"))
            _merge_context_value(enriched, "best_of", matched_context.get("best_of"))
            _merge_context_value(enriched, "tourney_level", matched_context.get("tourney_level"))
            _merge_context_value(enriched, "tournament_name", matched_context.get("tournament_name"))

        if _is_atp_grand_slam(enriched):
            _merge_context_value(enriched, "tourney_level", "G")

        if not _has_value(enriched.get("best_of")):
            enriched["best_of"] = _default_live_best_of(enriched)

        if not _has_value(enriched.get("surface")):
            enriched["surface"] = "unknown"

        enriched_rows.append(enriched)

    return pd.DataFrame(enriched_rows)


def build_live_tennis_features(matchups_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """Build live pre-match tennis features from historical matches only."""
    if matchups_df.empty:
        return matchups_df.copy()

    builder = TennisFeatureBuilder()
    builder.ingest_history(history_df)
    enriched_matchups = enrich_matchups_with_context(matchups_df, history_df)
    feature_rows = [builder.snapshot_match(row) for row in enriched_matchups.to_dict("records")]
    feature_frame = pd.DataFrame(feature_rows)
    feature_frame = feature_frame.drop(columns=[column for column in feature_frame.columns if column in enriched_matchups.columns])
    output = pd.concat([enriched_matchups.reset_index(drop=True), feature_frame], axis=1)
    return output


def filter_minimum_history(features_df: pd.DataFrame, min_matches: int = TENNIS_MIN_MATCHES) -> pd.DataFrame:
    """Filter out rows without enough prior match history."""
    if features_df.empty:
        return features_df.copy()
    if min_matches <= 0:
        return features_df.copy()

    required_columns = ["a_num_matches", "b_num_matches"]
    missing = [column for column in required_columns if column not in features_df.columns]
    if missing:
        raise ValueError(
            "Missing tennis history columns required for minimum-history filtering: "
            + ", ".join(missing)
        )
    return features_df[
        (features_df["a_num_matches"] >= min_matches)
        & (features_df["b_num_matches"] >= min_matches)
    ].copy()
