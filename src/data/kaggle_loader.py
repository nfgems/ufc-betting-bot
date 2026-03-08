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

logger = logging.getLogger(__name__)


def _parse_height_inches(height_str: str) -> Optional[float]:
    """Convert height string like '5\\'10\"' or '70' to inches."""
    if pd.isna(height_str) or str(height_str).strip() in ("", "--"):
        return None
    s = str(height_str).strip()
    # Format: 5'10" or 5' 10"
    import re
    match = re.match(r"""(\d+)['"]\s*(\d+)?""", s)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2)) if match.group(2) else 0
        return feet * 12 + inches
    # Already numeric
    try:
        return float(s)
    except ValueError:
        return None


def _parse_reach_inches(reach_str: str) -> Optional[float]:
    """Convert reach string like '72\"' or '72.0' to inches."""
    if pd.isna(reach_str) or str(reach_str).strip() in ("", "--"):
        return None
    s = str(reach_str).replace('"', "").replace("'", "").strip()
    try:
        return float(s)
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
        "a_height", "a_reach", "a_weight", "a_age",
        "b_height", "b_reach", "b_weight", "b_age",
        "a_wins", "a_losses", "b_wins", "b_losses",
        "a_sig_str_landed", "a_sig_str_attempted", "a_td_landed", "a_td_attempted",
        "b_sig_str_landed", "b_sig_str_attempted", "b_td_landed", "b_td_attempted",
        "a_kd", "b_kd", "a_sub_att", "b_sub_att", "a_rev", "b_rev",
        "a_ctrl_seconds", "b_ctrl_seconds",
    ]
    for col in float_cols:
        if col in normalized.columns:
            if "pct" not in col and "acc" not in col and "def" not in col:
                normalized[col] = normalized[col].apply(_safe_float)
            else:
                normalized[col] = normalized[col].apply(_parse_pct)

    # Height/reach parsing
    for prefix in ["a_", "b_"]:
        if f"{prefix}height" in normalized.columns:
            normalized[f"{prefix}height"] = normalized[f"{prefix}height"].apply(
                lambda x: _parse_height_inches(x) if not isinstance(x, (int, float)) else x
            )
        if f"{prefix}reach" in normalized.columns:
            normalized[f"{prefix}reach"] = normalized[f"{prefix}reach"].apply(
                lambda x: _parse_reach_inches(x) if not isinstance(x, (int, float)) else x
            )

    # Create binary target: 1 if fighter_a wins, 0 if fighter_b wins
    normalized["target"] = (normalized["winner"] == normalized["fighter_a"]).astype(int)

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

        # Fighter names (many formats)
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
        "titlebout": "title_bout",
        "numberofrounds": "num_rounds",
        "finishround": "finish_round",
        "finishroundtime": "finish_time",
        "totalfighttimesecs": "total_fight_time_secs",

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


def save_processed(df: pd.DataFrame, filename: str = "fights_cleaned.csv") -> Path:
    """Save processed DataFrame to the processed data directory."""
    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=False)
    logger.info(f"Saved processed data to {path}")
    return path
