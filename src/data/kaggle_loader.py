"""
Kaggle dataset loader — loads and cleans UFC fight data from Kaggle CSVs.

Supports multiple common Kaggle UFC dataset formats:
- "Ultimate UFC Dataset" (mdabbert)
- "UFC Complete Dataset 1996-2024" (maksbasher)
- "UFC DATASETS 1994-2025" (neelagiriaditya)

Place downloaded CSV files in data/raw/ before running.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data.fight_results import (
    RESULT_CLASS_NORMAL,
    has_result_policy_marker,
    validate_marked_result,
    winner_for_side,
)
from src.data.io_utils import write_csv_atomically

logger = logging.getLogger(__name__)
_INCH_TO_CM = 2.54


def _parse_height_cm(height_str: str) -> Optional[float]:
    """Convert imperial height strings to centimeters."""
    if pd.isna(height_str) or str(height_str).strip() in ("", "--"):
        return None
    s = str(height_str).strip().lower()
    import re
    metric_match = re.search(r"(\d+(?:\.\d+)?)\s*cm\b", s)
    if metric_match:
        return float(metric_match.group(1))
    match = re.match(r"""(\d+)\s*(?:'|ft|feet)\s*(\d+)?""", s)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2)) if match.group(2) else 0
        return round((feet * 12 + inches) * _INCH_TO_CM, 2)
    cleaned = (
        s.replace('"', "")
        .replace("inches", "")
        .replace("inch", "")
        .replace("in", "")
        .strip()
    )
    try:
        return round(float(cleaned) * _INCH_TO_CM, 2)
    except ValueError:
        return None


def _parse_reach_cm(reach_str: str) -> Optional[float]:
    """Convert imperial reach strings to centimeters."""
    if pd.isna(reach_str) or str(reach_str).strip() in ("", "--", "-"):
        return None
    s = str(reach_str).strip().lower()
    import re
    metric_match = re.search(r"(\d+(?:\.\d+)?)\s*cm\b", s)
    if metric_match:
        return float(metric_match.group(1))
    meters_match = re.search(r"(\d+(?:\.\d+)?)\s*m\b", s)
    if meters_match:
        return round(float(meters_match.group(1)) * 100.0, 2)
    feet_match = re.match(r"""(\d+)\s*(?:'|ft|feet)\s*(\d+)?""", s)
    if feet_match:
        feet = int(feet_match.group(1))
        inches = int(feet_match.group(2)) if feet_match.group(2) else 0
        return round((feet * 12 + inches) * _INCH_TO_CM, 2)
    cleaned = (
        s.replace('"', "")
        .replace("'", "")
        .replace("inches", "")
        .replace("inch", "")
        .replace("in", "")
        .strip()
    )
    try:
        return round(float(cleaned) * _INCH_TO_CM, 2)
    except ValueError:
        return None


def _parse_pct(pct_str) -> Optional[float]:
    """Convert percentage string like '45%' or '0.45' to float 0-100."""
    if pd.isna(pct_str) or str(pct_str).strip() in ("", "--"):
        return None
    s = str(pct_str).strip().replace("%", "")
    try:
        val = float(s)
        if val <= 1.0:
            return val * 100
        return val
    except ValueError:
        return None


def _safe_float(val) -> Optional[float]:
    """Safely convert to float."""
    if pd.isna(val) or str(val).strip() in ("", "--"):
        return None
    try:
        return float(str(val).strip())
    except ValueError:
        return None


def _parse_nullable_binary(value) -> Optional[float]:
    """Convert mixed boolean-ish inputs to 0/1 while preserving unknowns."""
    if pd.isna(value) or str(value).strip() in ("", "--"):
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0

    text = str(value).strip().lower()
    truthy = {"1", "true", "t", "yes", "y"}
    falsy = {"0", "false", "f", "no", "n"}
    if text in truthy:
        return 1.0
    if text in falsy:
        return 0.0

    numeric = _safe_float(value)
    if numeric is None:
        return None
    if numeric in (0.0, 1.0):
        return float(numeric)
    return None


def _normalize_length_value(
    value,
    parser,
    *,
    source_is_metric: bool,
) -> Optional[float]:
    """Normalize mixed numeric/string length inputs to centimeters."""
    if pd.isna(value) or str(value).strip() in ("", "--"):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if source_is_metric else round(numeric * _INCH_TO_CM, 2)
    if source_is_metric:
        numeric = _safe_float(value)
        if numeric is not None:
            return numeric
    return parser(value)


def load_kaggle_dataset(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load a Kaggle UFC dataset and normalize columns to a standard schema.

    Returns DataFrame with columns:
        event_date, event_name, weight_class, winner,
        fighter_a, fighter_b, method,
        a_slpm, a_sapm, a_str_acc, a_str_def, a_td_avg, a_td_acc, a_td_def, a_sub_avg,
        b_slpm, b_sapm, b_str_acc, b_str_def, b_td_avg, b_td_acc, b_td_def, b_sub_avg,
        a_height, a_reach, a_weight, a_age, a_stance, a_wins, a_losses,
        b_height, b_reach, b_weight, b_age, b_stance, b_wins, b_losses,
        a_sig_str_landed, a_sig_str_attempted, a_td_landed, a_td_attempted,
        b_sig_str_landed, b_sig_str_attempted, b_td_landed, b_td_attempted,
        a_kd, b_kd, a_sub_att, b_sub_att, a_rev, b_rev,
        a_ctrl_seconds, b_ctrl_seconds
    """
    if filepath is None:
        # Auto-detect: prefer ufc-master.csv, then any CSV in raw data dir
        preferred = RAW_DATA_DIR / "ufc-master.csv"
        if preferred.exists():
            filepath = preferred
        else:
            csvs = list(RAW_DATA_DIR.glob("*.csv"))
            if not csvs:
                raise FileNotFoundError(
                    f"No CSV files found in {RAW_DATA_DIR}. "
                    "Download a UFC dataset from Kaggle and place it there."
                )
            filepath = csvs[0]
        logger.info(f"Auto-detected dataset: {filepath.name}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows from {filepath.name}")
    logger.info(f"Columns: {list(df.columns)}")

    # Detect dataset format and normalize
    cols_lower = {c.lower().strip(): c for c in df.columns}
    normalized = _normalize_columns(df, cols_lower)

    # Parse dates
    if "event_date" in normalized.columns:
        normalized["event_date"] = pd.to_datetime(
            normalized["event_date"], errors="coerce", format="mixed"
        )

    # Sort by date
    normalized = normalized.sort_values("event_date").reset_index(drop=True)

    # Parse numeric fields
    float_cols = [
        "a_slpm", "a_sapm", "a_str_acc", "a_str_def", "a_td_avg", "a_td_acc",
        "a_td_def", "a_sub_avg", "b_slpm", "b_sapm", "b_str_acc", "b_str_def",
        "b_td_avg", "b_td_acc", "b_td_def", "b_sub_avg",
        "a_weight", "a_age",
        "b_weight", "b_age",
        "a_wins", "a_losses", "b_wins", "b_losses",
        "a_sig_str_landed", "a_sig_str_attempted", "a_td_landed", "a_td_attempted",
        "b_sig_str_landed", "b_sig_str_attempted", "b_td_landed", "b_td_attempted",
        "a_kd", "b_kd", "a_sub_att", "b_sub_att", "a_rev", "b_rev",
        "a_ctrl_seconds", "b_ctrl_seconds",
        # New: finish method counts, streaks, rounds, rankings, odds
        "a_wins_ko", "b_wins_ko", "a_wins_sub", "b_wins_sub",
        "a_wins_dec", "b_wins_dec", "a_wins_dec_maj", "b_wins_dec_maj",
        "a_wins_dec_split", "b_wins_dec_split", "a_wins_tko_doc", "b_wins_tko_doc",
        "a_draws", "b_draws",
        "a_win_streak", "b_win_streak", "a_lose_streak", "b_lose_streak",
        "a_longest_win_streak", "b_longest_win_streak",
        "a_total_rounds", "b_total_rounds",
        "a_title_bouts", "b_title_bouts",
        "a_odds", "b_odds", "a_expected_value", "b_expected_value",
        "a_wc_rank", "b_wc_rank", "a_pfp_rank", "b_pfp_rank",
        "a_dec_odds", "b_dec_odds", "a_sub_odds", "b_sub_odds",
        "a_ko_odds", "b_ko_odds",
        "title_bout", "num_rounds", "empty_arena",
    ]
    binary_cols = {"title_bout", "empty_arena"}
    for col in float_cols:
        if col in normalized.columns:
            if col in binary_cols:
                normalized[col] = normalized[col].apply(_parse_nullable_binary)
            elif "pct" not in col and "acc" not in col and "def" not in col:
                normalized[col] = normalized[col].apply(_safe_float)
            else:
                normalized[col] = normalized[col].apply(_parse_pct)

    length_source_map = {
        "a_height": {"r_height_cms", "redheightcms"},
        "b_height": {"b_height_cms", "blueheightcms"},
        "a_reach": {"r_reach_cms", "redreachcms"},
        "b_reach": {"b_reach_cms", "bluereachcms"},
    }
    for column, parser in (
        ("a_height", _parse_height_cm),
        ("b_height", _parse_height_cm),
        ("a_reach", _parse_reach_cm),
        ("b_reach", _parse_reach_cm),
    ):
        if column not in normalized.columns:
            continue
        source_is_metric = any(source in cols_lower for source in length_source_map[column])
        normalized[column] = normalized[column].apply(
            lambda x, parser=parser, source_is_metric=source_is_metric: _normalize_length_value(
                x,
                parser,
                source_is_metric=source_is_metric,
            )
        )

    # Legacy, unmarked rows intentionally keep their historical semantics.  The
    # incoming-result policy applies only to rows explicitly carrying its marker.
    is_a_win = normalized["winner"] == normalized["fighter_a"]
    is_b_win = normalized["winner"] == normalized["fighter_b"]
    normalized["target"] = np.where(is_a_win, 1.0, np.where(is_b_win, 0.0, np.nan))
    policy_markers = normalized.get(
        "result_policy_version",
        pd.Series(index=normalized.index, dtype="object"),
    )
    for index in normalized.index[policy_markers.map(has_result_policy_marker)]:
        row = normalized.loc[index]
        classification = validate_marked_result(
            row.get("official_outcome"),
            row.get("method"),
            policy_version=row.get("result_policy_version"),
            persisted_result_class=row.get("result_class"),
            persisted_binary_target_eligible=row.get(
                "binary_target_eligible"
            ),
        )
        classified_winner = winner_for_side(
            classification.winner_side,
            fighter_a=row.get("fighter_a"),
            fighter_b=row.get("fighter_b"),
        )
        observed_winner = row.get("winner")
        observed_winner_present = not pd.isna(observed_winner) and bool(
            str(observed_winner).strip()
        )
        if observed_winner_present and observed_winner != classified_winner:
            raise ValueError(
                "marked result winner contradicts official outcome at row "
                f"{index}"
            )
        normalized.at[index, "winner"] = classified_winner
        normalized.at[index, "target"] = (
            1.0
            if classified_winner == row.get("fighter_a")
            else 0.0
        ) if classification.result_class == RESULT_CLASS_NORMAL else np.nan

    n_non_binary = int(normalized["target"].isna().sum())
    if n_non_binary:
        logger.info(
            "Excluded %d non-binary result rows from target (set to NaN)",
            n_non_binary,
        )

    logger.info(f"Normalized dataset: {len(normalized)} fights, {len(normalized.columns)} columns")
    return normalized


def _normalize_columns(df: pd.DataFrame, cols_lower: dict) -> pd.DataFrame:
    """Map various Kaggle column naming conventions to our standard schema."""
    out = pd.DataFrame()

    # Common column name mappings (lowercase key -> our column name)
    mappings = {
        # Event info
        "date": "event_date",
        "event_date": "event_date",
        "event": "event_name",
        "event_name": "event_name",
        "location": "location",
        "weight_class": "weight_class",
        "weightclass": "weight_class",

        # Fight outcome
        "winner": "winner",
        "w/l": "winner",
        "method": "method",
        "win_by": "method",
        "finish": "method",
        "official_outcome": "official_outcome",
        "result_policy_version": "result_policy_version",
        "result_class": "result_class",
        "binary_target_eligible": "binary_target_eligible",
        "result_quarantine_reason": "result_quarantine_reason",

        # Fighter names (many formats)
        "fighter_a": "fighter_a",
        "fighter_b": "fighter_b",
        "fighter_a_id": "fighter_a_id",
        "fighter_b_id": "fighter_b_id",
        "r_fighter": "fighter_a",
        "b_fighter": "fighter_b",
        "fighter_1": "fighter_a",
        "fighter_2": "fighter_b",
        "red_fighter": "fighter_a",
        "blue_fighter": "fighter_b",
        "fighter1": "fighter_a",
        "fighter2": "fighter_b",
        # CamelCase format (shortlikeafox/mdabbert dataset)
        "redfighter": "fighter_a",
        "bluefighter": "fighter_b",

        # Red/Fighter A career stats (snake_case)
        "r_avg_sig_str_landed": "a_slpm",
        "r_avg_sig_str_pct": "a_str_acc",
        "r_avg_sig_str_absorbed": "a_sapm",
        "r_avg_str_def": "a_str_def",
        "r_avg_td_landed": "a_td_avg",
        "r_avg_td_pct": "a_td_acc",
        "r_avg_td_def": "a_td_def",
        "r_avg_sub_att": "a_sub_avg",

        "b_avg_sig_str_landed": "b_slpm",
        "b_avg_sig_str_pct": "b_str_acc",
        "b_avg_sig_str_absorbed": "b_sapm",
        "b_avg_str_def": "b_str_def",
        "b_avg_td_landed": "b_td_avg",
        "b_avg_td_pct": "b_td_acc",
        "b_avg_td_def": "b_td_def",
        "b_avg_sub_att": "b_sub_avg",

        # CamelCase career stats (shortlikeafox/mdabbert dataset)
        "redavgsigstrlanded": "a_slpm",
        "redavgsigstrpct": "a_str_acc",
        "blueavgsigstrlanded": "b_slpm",
        "blueavgsigstrpct": "b_str_acc",
        "redavgsubatt": "a_sub_avg",
        "blueavgsubatt": "b_sub_avg",
        "redavgtdlanded": "a_td_avg",
        "blueavgtdlanded": "b_td_avg",
        "redavgtdpct": "a_td_acc",
        "blueavgtdpct": "b_td_acc",

        # Physical attributes (snake_case)
        "r_height_cms": "a_height",
        "b_height_cms": "b_height",
        "r_reach_cms": "a_reach",
        "b_reach_cms": "b_reach",
        "r_weight_lbs": "a_weight",
        "b_weight_lbs": "b_weight",
        "r_age": "a_age",
        "b_age": "b_age",
        "r_stance": "a_stance",
        "b_stance": "b_stance",
        "r_wins": "a_wins",
        "r_losses": "a_losses",
        "b_wins": "b_wins",
        "b_losses": "b_losses",

        # CamelCase physical attributes (shortlikeafox/mdabbert dataset)
        "redheightcms": "a_height",
        "blueheightcms": "b_height",
        "redreachcms": "a_reach",
        "bluereachcms": "b_reach",
        "redweightlbs": "a_weight",
        "blueweightlbs": "b_weight",
        "redage": "a_age",
        "blueage": "b_age",
        "redstance": "a_stance",
        "bluestance": "b_stance",
        "redwins": "a_wins",
        "redlosses": "a_losses",
        "bluewins": "b_wins",
        "bluelosses": "b_losses",

        # CamelCase extra stats
        "redcurrentwinstreak": "a_win_streak",
        "bluecurrentwinstreak": "b_win_streak",
        "redcurrentlosestreak": "a_lose_streak",
        "bluecurrentlosestreak": "b_lose_streak",
        "redtotalroundsfought": "a_total_rounds",
        "bluetotalroundsfought": "b_total_rounds",
        "redtotaltitlebouts": "a_title_bouts",
        "bluetotaltitlebouts": "b_title_bouts",
        "redwinsbyko": "a_wins_ko",
        "bluewinsbyko": "b_wins_ko",
        "redwinsbysubmission": "a_wins_sub",
        "bluewinsbysubmission": "b_wins_sub",
        "redwinsbydecisionunanimous": "a_wins_dec",
        "bluewinsbydecisionunanimous": "b_wins_dec",
        "redodds": "a_odds",
        "blueodds": "b_odds",
        "redexpectedvalue": "a_expected_value",
        "blueexpectedvalue": "b_expected_value",
        "titlebout": "title_bout",
        "numberofrounds": "num_rounds",
        "finishround": "finish_round",
        "finishroundtime": "finish_time",
        "totalfighttimesecs": "total_fight_time_secs",
        "gender": "gender",
        "country": "country",
        "emptyarena": "empty_arena",

        # Draw counts
        "reddraws": "a_draws",
        "bluedraws": "b_draws",

        # Longest win streak
        "redlongestwinstreak": "a_longest_win_streak",
        "bluelongestwinstreak": "b_longest_win_streak",

        # Additional win method breakdowns
        "redwinsbydecisionmajority": "a_wins_dec_maj",
        "bluewinsbydecisionmajority": "b_wins_dec_maj",
        "redwinsbydecisionsplit": "a_wins_dec_split",
        "bluewinsbydecisionsplit": "b_wins_dec_split",
        "redwinsbytkodoctorstoppage": "a_wins_tko_doc",
        "bluewinsbytkodoctorstoppage": "b_wins_tko_doc",

        # UFC Rankings at fight time
        "rmatchwcrank": "a_wc_rank",
        "bmatchwcrank": "b_wc_rank",
        "rpfprank": "a_pfp_rank",
        "bpfprank": "b_pfp_rank",
        "betterrank": "better_rank",

        # Method-specific betting odds
        "reddecodds": "a_dec_odds",
        "bluedecodds": "b_dec_odds",
        "rsubodds": "a_sub_odds",
        "bsubodds": "b_sub_odds",
        "rkoodds": "a_ko_odds",
        "bkoodds": "b_ko_odds",

        # Per-fight stats
        "r_sig_str.": "a_sig_str_landed",
        "b_sig_str.": "b_sig_str_landed",
        "r_td": "a_td_landed",
        "b_td": "b_td_landed",
        "r_kd": "a_kd",
        "b_kd": "b_kd",
        "r_sub_att": "a_sub_att",
        "b_sub_att": "b_sub_att",
        "r_rev": "a_rev",
        "b_rev": "b_rev",
        "r_ctrl": "a_ctrl_seconds",
        "b_ctrl": "b_ctrl_seconds",
    }

    for col_lower, orig_col in cols_lower.items():
        target_col = mappings.get(col_lower)
        if target_col:
            out[target_col] = df[orig_col]

    # If we didn't find mapped columns, try direct copy for any remaining standard names
    for col in df.columns:
        cl = col.lower().strip()
        if cl in [c.lower() for c in out.columns]:
            continue
        # Check if column starts with r_ or b_ (common red/blue format)
        if cl.startswith("r_") and cl not in mappings:
            target = "a_" + cl[2:]
            if target not in out.columns:
                out[target] = df[col]
        elif cl.startswith("b_") and cl not in mappings:
            target = "b_" + cl[2:]
            if target not in out.columns:
                out[target] = df[col]

    # Handle winner column for Red/Blue format datasets
    if "winner" in out.columns:
        winner_vals = out["winner"].astype(str).str.strip().str.lower()
        if winner_vals.isin(["red", "blue", "draw", "no contest"]).any():
            # Convert Red/Blue to actual fighter names
            mask_red = winner_vals == "red"
            mask_blue = winner_vals == "blue"
            out.loc[mask_red, "winner"] = out.loc[mask_red, "fighter_a"]
            out.loc[mask_blue, "winner"] = out.loc[mask_blue, "fighter_b"]

    return out


def save_processed(df: pd.DataFrame, filename: str | Path = "fights_cleaned.csv") -> Path:
    """Save processed DataFrame to the processed data directory."""
    path = Path(filename)
    if not path.is_absolute():
        path = PROCESSED_DATA_DIR / path
    write_csv_atomically(df, path, refuse_empty=True)
    logger.info(f"Saved processed data to {path}")
    return path
