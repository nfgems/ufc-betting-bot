"""
Live fighter stats lookup — scrapes UFCStats.com for current fighter data
and builds feature vectors compatible with the trained model.

Used during live predictions to populate the full feature set (rolling stats,
physical attributes, Elo, etc.) instead of relying on median imputation.
"""

import logging
import re
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import (
    UFCSTATS_FIGHTER_URL,
    ROLLING_WINDOW,
    ELO_INITIAL,
    ELO_K_FACTOR,
    PROCESSED_DATA_DIR,
)

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
REQUEST_DELAY = 1.0

# Cache to avoid re-scraping during a single session
_fighter_cache: dict[str, dict] = {}
_fighter_url_cache: dict[str, str] = {}

# Elo state loaded from historical data
_elo_ratings: Optional[dict[str, float]] = None
_elo_fight_counts: Optional[dict[str, int]] = None


def _get_soup(url: str) -> BeautifulSoup:
    """Fetch a URL and return parsed BeautifulSoup."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return BeautifulSoup(resp.text, "lxml")


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _safe_float(value, default=np.nan) -> float:
    """Safely convert a value to float."""
    if value is None or value == "" or value == "--":
        return default
    try:
        return float(str(value).replace("%", "").strip())
    except (ValueError, TypeError):
        return default


def _parse_height_inches(height_str: str) -> float:
    """Parse height like '5\\'10\"' or '5' 10\"' to inches."""
    if not height_str or height_str == "--":
        return np.nan
    match = re.search(r"(\d+)'?\s*(\d+)", height_str)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        return feet * 12 + inches
    return np.nan


def _parse_reach_inches(reach_str: str) -> float:
    """Parse reach like '74\"' or '74' to inches."""
    if not reach_str or reach_str == "--":
        return np.nan
    match = re.search(r"(\d+)", reach_str)
    if match:
        return float(match.group(1))
    return np.nan


def _parse_weight_lbs(weight_str: str) -> float:
    """Parse weight like '185 lbs.' to float."""
    if not weight_str or weight_str == "--":
        return np.nan
    match = re.search(r"(\d+)", weight_str)
    if match:
        return float(match.group(1))
    return np.nan


def _parse_dob_to_age(dob_str: str) -> float:
    """Parse DOB like 'Sep 22, 1989' to current age in years."""
    if not dob_str or dob_str == "--":
        return np.nan
    for fmt in ["%b %d, %Y", "%B %d, %Y"]:
        try:
            dob = datetime.strptime(dob_str.strip(), fmt)
            age = (datetime.now() - dob).days / 365.25
            return round(age, 1)
        except ValueError:
            continue
    return np.nan


def _parse_ctrl_seconds(ctrl_str: str) -> float:
    """Parse control time like '4:30' to seconds. Returns NaN for missing data."""
    if not ctrl_str or ctrl_str == "--":
        return np.nan
    ctrl_str = ctrl_str.strip()
    if ctrl_str == "0:00":
        return 0.0
    match = re.match(r"(\d+):(\d+)", ctrl_str)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    return np.nan


# ---------------------------------------------------------------------------
# Fighter search on UFCStats.com
# ---------------------------------------------------------------------------

def search_fighter_url(fighter_name: str) -> Optional[str]:
    """
    Search UFCStats.com for a fighter by name. Returns their profile URL.
    Uses the alphabetical fighter listing with last name initial.
    """
    if fighter_name in _fighter_url_cache:
        return _fighter_url_cache[fighter_name]

    # Try last name initial
    parts = fighter_name.strip().split()
    if not parts:
        return None

    last_name = parts[-1]
    char = last_name[0].lower()

    try:
        url = f"http://ufcstats.com/statistics/fighters?char={char}&page=all"
        soup = _get_soup(url)
    except Exception as e:
        logger.warning(f"Failed to search UFCStats for '{fighter_name}': {e}")
        return None

    name_lower = fighter_name.lower().strip()
    best_url = None
    best_score = 0

    for row in soup.select("tr.b-statistics__table-row"):
        cols = row.select("td")
        if len(cols) < 2:
            continue

        first_link = cols[0].select_one("a.b-link")
        last_link = cols[1].select_one("a.b-link")
        if not first_link or not last_link:
            continue

        first_name = _clean_text(first_link.text).lower()
        last_name_found = _clean_text(last_link.text).lower()
        full_name = f"{first_name} {last_name_found}"

        fighter_url = first_link.get("href", "").strip()
        if not fighter_url or "fighter-details" not in fighter_url:
            continue

        # Exact match (also check reversed for Eastern name order, e.g. "Zhang Weili" vs "Weili Zhang")
        reversed_name = f"{last_name_found} {first_name}"
        if full_name == name_lower or reversed_name == name_lower:
            _fighter_url_cache[fighter_name] = fighter_url
            return fighter_url

        # Score partial matches
        score = 0
        if last_name_found == parts[-1].lower():
            score += 5
        if first_name == parts[0].lower():
            score += 5
        elif first_name and parts[0].lower() and first_name[0] == parts[0].lower()[0]:
            score += 2
        if name_lower in full_name or full_name in name_lower:
            score += 3

        if score > best_score:
            best_score = score
            best_url = fighter_url

    if best_url and best_score >= 5:
        _fighter_url_cache[fighter_name] = best_url
        return best_url

    # Fallback: try searching by first name initial (handles Eastern name order on UFCStats)
    first_char = parts[0][0].lower()
    if first_char != char:
        try:
            url = f"http://ufcstats.com/statistics/fighters?char={first_char}&page=all"
            soup = _get_soup(url)
        except Exception:
            return None

        for row in soup.select("tr.b-statistics__table-row"):
            cols = row.select("td")
            if len(cols) < 2:
                continue
            first_link = cols[0].select_one("a.b-link")
            last_link = cols[1].select_one("a.b-link")
            if not first_link or not last_link:
                continue
            first_name = _clean_text(first_link.text).lower()
            last_name_found = _clean_text(last_link.text).lower()
            full_name = f"{first_name} {last_name_found}"
            reversed_name = f"{last_name_found} {first_name}"
            fighter_url = first_link.get("href", "").strip()
            if not fighter_url or "fighter-details" not in fighter_url:
                continue
            if full_name == name_lower or reversed_name == name_lower:
                _fighter_url_cache[fighter_name] = fighter_url
                return fighter_url

    return None


# ---------------------------------------------------------------------------
# Fighter profile scraping
# ---------------------------------------------------------------------------

def scrape_fighter_profile(fighter_url: str) -> dict:
    """
    Scrape a fighter's profile page for career stats and physical attributes.

    Returns dict with keys: name, height, reach, weight, stance, age, dob,
        slpm, str_acc, sapm, str_def, td_avg, td_acc, td_def, sub_avg, record
    """
    soup = _get_soup(fighter_url)

    name_el = soup.select_one("h2.b-content__title span")
    name = _clean_text(name_el.text) if name_el else ""

    record_el = soup.select_one("span.b-content__title-record")
    record_str = _clean_text(record_el.text.replace("Record:", "")) if record_el else ""

    # Parse win/loss/draw from record
    wins, losses, draws = 0, 0, 0
    rec_match = re.match(r"(\d+)-(\d+)-(\d+)", record_str)
    if rec_match:
        wins = int(rec_match.group(1))
        losses = int(rec_match.group(2))
        draws = int(rec_match.group(3))

    # Physical attributes
    info = {}
    for item in soup.select("li.b-list__box-list-item"):
        text = _clean_text(item.text)
        if "Height:" in text:
            info["height_raw"] = text.replace("Height:", "").strip()
        elif "Weight:" in text:
            info["weight_raw"] = text.replace("Weight:", "").strip()
        elif "Reach:" in text:
            info["reach_raw"] = text.replace("Reach:", "").strip()
        elif "STANCE:" in text:
            info["stance"] = text.replace("STANCE:", "").strip()
        elif "DOB:" in text:
            info["dob"] = text.replace("DOB:", "").strip()

    # Career stats (per 15 min averages)
    stats = {}
    for item in soup.select("li.b-list__box-list-item_type_block"):
        text = _clean_text(item.text)
        if "SLpM:" in text:
            stats["slpm"] = _safe_float(text.replace("SLpM:", ""))
        elif "Str. Acc.:" in text:
            stats["str_acc"] = _safe_float(text.replace("Str. Acc.:", "").replace("%", ""))
        elif "SApM:" in text:
            stats["sapm"] = _safe_float(text.replace("SApM:", ""))
        elif "Str. Def:" in text:
            stats["str_def"] = _safe_float(text.replace("Str. Def:", "").replace("%", ""))
        elif "TD Avg.:" in text:
            stats["td_avg"] = _safe_float(text.replace("TD Avg.:", ""))
        elif "TD Acc.:" in text:
            stats["td_acc"] = _safe_float(text.replace("TD Acc.:", "").replace("%", ""))
        elif "TD Def.:" in text:
            stats["td_def"] = _safe_float(text.replace("TD Def.:", "").replace("%", ""))
        elif "Sub. Avg.:" in text:
            stats["sub_avg"] = _safe_float(text.replace("Sub. Avg.:", ""))

    return {
        "name": name,
        "fighter_url": fighter_url,
        "record": record_str,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "height": _parse_height_inches(info.get("height_raw", "")),
        "reach": _parse_reach_inches(info.get("reach_raw", "")),
        "weight": _parse_weight_lbs(info.get("weight_raw", "")),
        "stance": info.get("stance", ""),
        "age": _parse_dob_to_age(info.get("dob", "")),
        **stats,
    }


# ---------------------------------------------------------------------------
# Fighter recent fight history
# ---------------------------------------------------------------------------

def _parse_stat_cell(cell_text: str) -> tuple[str, str]:
    """Parse a cell like '50 of 100' into (landed, attempted)."""
    text = _clean_text(cell_text)
    match = re.match(r"(\d+)\s+of\s+(\d+)", text)
    if match:
        return match.group(1), match.group(2)
    return text, text


def _scrape_fight_detail(detail_url: str, fighter_name: str) -> dict:
    """
    Scrape a fight detail page for stats not on the profile table:
    Rev, Ctrl, and title bout status.

    Args:
        detail_url: URL like http://ufcstats.com/fight-details/{id}
        fighter_name: Name of the fighter we're building features for,
                      used to determine which row is "ours" vs opponent.

    Returns dict with: rev, ctrl_seconds, opp_rev, opp_ctrl_seconds, is_title_bout
    """
    result = {
        "rev": np.nan,
        "ctrl_seconds": np.nan,
        "opp_rev": np.nan,
        "opp_ctrl_seconds": np.nan,
        "is_title_bout": False,
    }

    try:
        soup = _get_soup(detail_url)
    except Exception as e:
        logger.debug(f"Failed to fetch fight detail {detail_url}: {e}")
        return result

    # Title bout detection — look for belt.png or "Title Bout" text
    fight_title = soup.select_one("i.b-fight-details__fight-title")
    if fight_title:
        belt_img = fight_title.select_one("img[src*='belt.png']")
        title_text = "title bout" in fight_title.get_text().lower()
        result["is_title_bout"] = bool(belt_img or title_text)

    # Find the totals table — first table body with fight stats
    # The totals row has both fighters' stats in a single <tr>
    tables = soup.select("table.b-fight-details__table")
    if not tables:
        return result

    # First table is the Totals table
    totals_table = tables[0]
    body = totals_table.select_one("tbody")
    if not body:
        return result

    rows = body.select("tr.b-fight-details__table-row")
    if not rows:
        return result

    totals_row = rows[0]
    cols = totals_row.select("td")
    if len(cols) < 10:
        return result

    # Determine which <p> index (0 or 1) is our fighter
    fighter_ps = cols[0].select("p")
    if len(fighter_ps) < 2:
        return result

    name0 = _clean_text(fighter_ps[0].get_text()).lower()
    name1 = _clean_text(fighter_ps[1].get_text()).lower()
    fighter_lower = fighter_name.lower().strip()

    # Match by checking if fighter name is contained in the cell text
    if fighter_lower in name0 or name0 in fighter_lower:
        our_idx = 0
    elif fighter_lower in name1 or name1 in fighter_lower:
        our_idx = 1
    else:
        # Fallback: try matching last names
        fighter_last = fighter_lower.split()[-1] if fighter_lower.split() else ""
        if fighter_last and fighter_last in name0:
            our_idx = 0
        elif fighter_last and fighter_last in name1:
            our_idx = 1
        else:
            logger.debug(
                f"Could not match '{fighter_name}' to detail page fighters: "
                f"'{name0}' / '{name1}'"
            )
            return result

    opp_idx = 1 - our_idx

    # Rev is column 8 (two <p> elements)
    rev_ps = cols[8].select("p") if len(cols) > 8 else []
    if len(rev_ps) >= 2:
        result["rev"] = _safe_float(_clean_text(rev_ps[our_idx].text), np.nan)
        result["opp_rev"] = _safe_float(_clean_text(rev_ps[opp_idx].text), np.nan)

    # Ctrl is column 9 (two <p> elements, format "M:SS")
    ctrl_ps = cols[9].select("p") if len(cols) > 9 else []
    if len(ctrl_ps) >= 2:
        result["ctrl_seconds"] = _parse_ctrl_seconds(_clean_text(ctrl_ps[our_idx].text))
        result["opp_ctrl_seconds"] = _parse_ctrl_seconds(_clean_text(ctrl_ps[opp_idx].text))

    return result


def scrape_fighter_fights(fighter_url: str, fighter_name: str = "") -> list[dict]:
    """
    Scrape a fighter's fight history from their profile page,
    then enrich each fight with detail page data (Rev, Ctrl, title bout).

    Args:
        fighter_url: UFCStats profile URL
        fighter_name: Fighter's name (for matching on detail pages)

    Returns list of fight dicts with per-fight stats, ordered chronologically.
    """
    soup = _get_soup(fighter_url)

    fights = []
    # The fight history table on the profile page
    fight_rows = soup.select("tr.b-fight-details__table-row")

    for row in fight_rows:
        cols = row.select("td")
        if len(cols) < 10:
            continue

        # Check if this row has fight data
        result_col = cols[0].select_one("a.b-flag") or cols[0].select_one("i")
        if not result_col:
            continue

        result_text = _clean_text(result_col.text).lower()
        won = 1 if "win" in result_text else 0

        # Fight detail URL from the row's data-link attribute or the <a> in col[0]
        detail_url = row.get("data-link", "").strip()
        if not detail_url:
            a_tag = cols[0].select_one("a.b-flag")
            if a_tag:
                detail_url = a_tag.get("href", "").strip()

        # Fighter names (cols[1] has two <p> tags)
        fighter_ps = cols[1].select("p")
        if len(fighter_ps) < 2:
            continue
        opponent = _clean_text(fighter_ps[1].text)

        # Method and round
        method = _clean_text(cols[7].text) if len(cols) > 7 else ""
        round_num_str = _clean_text(cols[8].text) if len(cols) > 8 else ""

        # Parse round — None if unparseable (no guessing)
        round_val = _safe_float(round_num_str)
        round_finished = int(round_val) if not np.isnan(round_val) else None

        # Date
        date_text = _clean_text(cols[6].text) if len(cols) > 6 else ""
        event_date = None
        for fmt in ["%b. %d, %Y", "%B %d, %Y", "%b %d, %Y"]:
            try:
                event_date = datetime.strptime(date_text, fmt)
                break
            except ValueError:
                continue

        # Per-fight stats from the table (sig str, td, sub, etc.)
        # The profile fight table has: Result, Fighter, KD, Str, TD, Sub, Weight Class, Method, Round, Time
        kd = _safe_float(_clean_text(cols[2].select("p")[0].text)) if len(cols) > 2 and cols[2].select("p") else 0
        opp_kd = _safe_float(_clean_text(cols[2].select("p")[1].text)) if len(cols) > 2 and len(cols[2].select("p")) > 1 else 0

        # Sig str
        sig_str_text = cols[3].select("p")
        sig_str_landed, sig_str_attempted = 0, 0
        opp_sig_str_landed, opp_sig_str_attempted = 0, 0
        if sig_str_text and len(sig_str_text) >= 2:
            l, a = _parse_stat_cell(sig_str_text[0].text)
            sig_str_landed, sig_str_attempted = _safe_float(l, 0), _safe_float(a, 0)
            l2, a2 = _parse_stat_cell(sig_str_text[1].text)
            opp_sig_str_landed, opp_sig_str_attempted = _safe_float(l2, 0), _safe_float(a2, 0)

        # TD
        td_text = cols[4].select("p") if len(cols) > 4 else []
        td_landed, td_attempted = 0, 0
        opp_td_landed, opp_td_attempted = 0, 0
        if td_text and len(td_text) >= 2:
            l, a = _parse_stat_cell(td_text[0].text)
            td_landed, td_attempted = _safe_float(l, 0), _safe_float(a, 0)
            l2, a2 = _parse_stat_cell(td_text[1].text)
            opp_td_landed, opp_td_attempted = _safe_float(l2, 0), _safe_float(a2, 0)

        # Sub attempts
        sub_text = cols[5].select("p") if len(cols) > 5 else []
        sub_att = _safe_float(_clean_text(sub_text[0].text)) if sub_text else 0
        opp_sub_att = _safe_float(_clean_text(sub_text[1].text)) if len(sub_text) > 1 else 0

        fight = {
            "event_date": event_date,
            "detail_url": detail_url,
            "opponent": opponent,
            "won": won,
            "method": method,
            "round_finished": round_finished,
            "kd": kd,
            "sig_str_landed": sig_str_landed,
            "sig_str_attempted": sig_str_attempted,
            "td_landed": td_landed,
            "td_attempted": td_attempted,
            "sub_att": sub_att,
            "rev": np.nan,  # Will be filled from detail page
            "ctrl_seconds": np.nan,  # Will be filled from detail page
            "is_title_bout": False,  # Will be filled from detail page
            # Opponent stats
            "opp_kd": opp_kd,
            "opp_sig_str_landed": opp_sig_str_landed,
            "opp_sig_str_attempted": opp_sig_str_attempted,
            "opp_td_landed": opp_td_landed,
            "opp_td_attempted": opp_td_attempted,
            "opp_sub_att": opp_sub_att,
            "opp_rev": np.nan,
            "opp_ctrl_seconds": np.nan,
        }
        fights.append(fight)

    # Scrape fight detail pages for Rev, Ctrl, and title bout data
    if fighter_name and fights:
        logger.info(f"  Scraping {len(fights)} fight detail pages for {fighter_name}...")
        for fight in fights:
            detail_url = fight.get("detail_url", "")
            if not detail_url or "fight-details" not in detail_url:
                continue
            detail = _scrape_fight_detail(detail_url, fighter_name)
            fight["rev"] = detail["rev"]
            fight["ctrl_seconds"] = detail["ctrl_seconds"]
            fight["opp_rev"] = detail["opp_rev"]
            fight["opp_ctrl_seconds"] = detail["opp_ctrl_seconds"]
            fight["is_title_bout"] = detail["is_title_bout"]

    # Reverse so oldest fight is first (profile page shows newest first)
    fights.reverse()
    return fights


# ---------------------------------------------------------------------------
# Build features for a single fighter
# ---------------------------------------------------------------------------

def _compute_rolling_for_fighter(
    fights: list[dict], profile: dict, window: int = ROLLING_WINDOW
) -> dict:
    """
    Compute rolling stats and derived features for a fighter from their
    fight history and profile data.

    Returns a dict of feature values (without a_/b_ prefix).
    """
    features = {}

    # Physical attributes from profile
    features["height"] = profile.get("height", np.nan)
    features["reach"] = profile.get("reach", np.nan)
    features["weight"] = profile.get("weight", np.nan)
    features["age"] = profile.get("age", np.nan)

    # Stance encoding
    stance_map = {"Orthodox": 0, "Southpaw": 1, "Switch": 2}
    features["stance_enc"] = stance_map.get(profile.get("stance", ""), -1)

    # Career record
    features["wins"] = profile.get("wins", 0)
    features["losses"] = profile.get("losses", 0)
    features["draws"] = profile.get("draws", 0)
    total_fights = features["wins"] + features["losses"] + features["draws"]
    features["win_pct"] = features["wins"] / max(total_fights, 1)

    # Career averages from profile (used as fallback / primary for slpm etc.)
    features["slpm"] = profile.get("slpm", np.nan)
    features["sapm"] = profile.get("sapm", np.nan)
    features["str_acc"] = profile.get("str_acc", np.nan)
    features["str_def"] = profile.get("str_def", np.nan)
    features["td_avg"] = profile.get("td_avg", np.nan)
    features["td_acc"] = profile.get("td_acc", np.nan)
    features["td_def"] = profile.get("td_def", np.nan)
    features["sub_avg"] = profile.get("sub_avg", np.nan)

    # Number of UFC fights
    features["num_fights"] = len(fights)

    if not fights:
        # No fight history — use career averages as rolling stats
        for stat in ["slpm", "sapm", "str_acc", "str_def", "td_avg", "td_acc",
                      "td_def", "sub_avg"]:
            features[f"roll_{stat}"] = profile.get(stat, np.nan)
        features["roll_sig_str_landed"] = np.nan
        features["roll_td_landed"] = np.nan
        features["roll_kd"] = np.nan
        features["roll_won"] = np.nan
        features["current_win_streak"] = np.nan
        features["days_since_last_fight"] = np.nan
        return features

    # Use last N fights for rolling averages
    recent = fights[-window:] if len(fights) >= window else fights

    # Rolling per-fight stats (use np.nanmean to skip missing values)
    for stat in ["kd", "sig_str_landed", "sig_str_attempted",
                 "td_landed", "td_attempted", "sub_att", "rev", "ctrl_seconds"]:
        vals = [f.get(stat, np.nan) for f in recent]
        features[f"roll_{stat}"] = float(np.nanmean(vals)) if any(not np.isnan(v) for v in vals) else np.nan

    # Rolling opponent stats
    for stat in ["opp_kd", "opp_sig_str_landed", "opp_sig_str_attempted",
                 "opp_td_landed", "opp_td_attempted", "opp_sub_att",
                 "opp_rev", "opp_ctrl_seconds"]:
        vals = [f.get(stat, np.nan) for f in recent]
        features[f"roll_{stat}"] = float(np.nanmean(vals)) if any(not np.isnan(v) for v in vals) else np.nan

    # Career averages from profile are better than per-fight for rate stats
    features["roll_slpm"] = profile.get("slpm", np.nan)
    features["roll_sapm"] = profile.get("sapm", np.nan)
    features["roll_str_acc"] = profile.get("str_acc", np.nan)
    features["roll_str_def"] = profile.get("str_def", np.nan)
    features["roll_td_avg"] = profile.get("td_avg", np.nan)
    features["roll_td_acc"] = profile.get("td_acc", np.nan)
    features["roll_td_def"] = profile.get("td_def", np.nan)
    features["roll_sub_avg"] = profile.get("sub_avg", np.nan)

    # Rolling win rate
    wins_recent = [f.get("won", 0) for f in recent]
    features["roll_won"] = np.mean(wins_recent) if wins_recent else np.nan

    # Win streak (consecutive wins going back from most recent)
    streak = 0
    for f in reversed(fights):
        if f.get("won", 0) == 1:
            streak += 1
        else:
            break
    features["current_win_streak"] = streak

    # Lose streak
    lose_streak = 0
    for f in reversed(fights):
        if f.get("won", 0) == 0:
            lose_streak += 1
        else:
            break
    features["lose_streak"] = lose_streak

    # Longest win streak ever
    longest_streak = 0
    current = 0
    for f in fights:
        if f.get("won", 0) == 1:
            current += 1
            longest_streak = max(longest_streak, current)
        else:
            current = 0
    features["longest_win_streak"] = longest_streak

    # Days since last fight
    last_fight = fights[-1]
    if last_fight.get("event_date"):
        days = (datetime.now() - last_fight["event_date"]).days
        features["days_since_last_fight"] = max(days, 0)
    else:
        features["days_since_last_fight"] = np.nan

    # Cage rust and layoff log
    dslf = features["days_since_last_fight"]
    if np.isnan(dslf):
        features["cage_rust"] = np.nan
        features["layoff_log"] = np.nan
    else:
        features["cage_rust"] = 1 if dslf > 365 else 0
        features["layoff_log"] = np.log1p(dslf)

    # Total rounds — sum actual rounds from scraped fight history
    round_vals = [f.get("round_finished") for f in fights if f.get("round_finished") is not None]
    features["total_rounds"] = sum(round_vals) if round_vals else np.nan

    # Title bouts — count from fight detail page scrape
    title_count = sum(1 for f in fights if f.get("is_title_bout", False))
    features["title_bouts"] = title_count

    # Strike differential — NaN if either stat is missing
    slpm = features.get("roll_slpm")
    sapm = features.get("roll_sapm")
    if slpm is not None and sapm is not None and not (np.isnan(slpm) or np.isnan(sapm)):
        features["strike_diff"] = slpm - sapm
        features["fight_pace"] = slpm + sapm
    else:
        features["strike_diff"] = np.nan
        features["fight_pace"] = np.nan

    # Cage time efficiency (sig strikes / control time — falls back to NaN if no ctrl data)
    ctrl = features.get("roll_ctrl_seconds", 0) or 0
    if ctrl > 1:
        features["ctrl_efficiency"] = (features.get("roll_sig_str_landed", 0) or 0) / ctrl
    else:
        features["ctrl_efficiency"] = np.nan

    # Finish rates — count win methods from fight history
    total_wins = sum(1 for f in fights if f.get("won", 0) == 1)
    if total_wins > 0:
        ko_wins = sum(1 for f in fights if f.get("won") == 1 and "ko" in f.get("method", "").lower())
        sub_wins = sum(1 for f in fights if f.get("won") == 1 and "sub" in f.get("method", "").lower())
        dec_wins = sum(1 for f in fights if f.get("won") == 1 and "dec" in f.get("method", "").lower())
        features["ko_rate"] = ko_wins / total_wins
        features["sub_rate"] = sub_wins / total_wins
        features["dec_rate"] = dec_wins / total_wins
    else:
        features["ko_rate"] = 0.0
        features["sub_rate"] = 0.0
        features["dec_rate"] = 0.0

    # Quality-adjusted win rate (win_pct weighted by opponent strength)
    # Uses opponent Elo ratings to give more credit for beating strong opponents
    win_pct = features.get("win_pct")
    _load_elo_ratings()  # Ensure Elo ratings are loaded
    if win_pct is not None and not np.isnan(win_pct) and fights and _elo_ratings:
        opp_elos = []
        for f in fights[-5:]:  # Last 5 opponents
            opp_name = f.get("opponent", "")
            if opp_name:
                opp_elo = _elo_ratings.get(opp_name, ELO_INITIAL)
                opp_elos.append(opp_elo)
        if opp_elos:
            avg_opp_elo = np.mean(opp_elos)
            features["adj_win_pct"] = win_pct * (avg_opp_elo / ELO_INITIAL)
        else:
            features["adj_win_pct"] = win_pct
    else:
        features["adj_win_pct"] = np.nan if (win_pct is None or np.isnan(win_pct)) else win_pct

    return features


# ---------------------------------------------------------------------------
# Elo lookup from historical data
# ---------------------------------------------------------------------------

def _load_elo_ratings():
    """Load pre-computed Elo ratings from the processed features file."""
    global _elo_ratings, _elo_fight_counts

    if _elo_ratings is not None:
        return

    features_path = PROCESSED_DATA_DIR / "features.csv"
    if not features_path.exists():
        _elo_ratings = {}
        _elo_fight_counts = {}
        return

    try:
        df = pd.read_csv(
            features_path,
            usecols=["fighter_a", "fighter_b", "a_elo", "b_elo",
                      "a_num_fights", "b_num_fights", "event_date"],
            parse_dates=["event_date"],
        )
    except (ValueError, KeyError):
        _elo_ratings = {}
        _elo_fight_counts = {}
        return

    # Get latest Elo for each fighter
    _elo_ratings = {}
    _elo_fight_counts = {}

    df = df.sort_values("event_date")
    for _, row in df.iterrows():
        fa = row.get("fighter_a", "")
        fb = row.get("fighter_b", "")
        if fa:
            _elo_ratings[fa] = row.get("a_elo", ELO_INITIAL)
            _elo_fight_counts[fa] = int(row.get("a_num_fights", 0))
        if fb:
            _elo_ratings[fb] = row.get("b_elo", ELO_INITIAL)
            _elo_fight_counts[fb] = int(row.get("b_num_fights", 0))

    logger.info(f"Loaded Elo ratings for {len(_elo_ratings)} fighters")


def get_fighter_elo(fighter_name: str) -> float:
    """Get the most recent Elo rating for a fighter."""
    _load_elo_ratings()

    # Try exact match first
    if fighter_name in _elo_ratings:
        return _elo_ratings[fighter_name]

    # Try case-insensitive
    name_lower = fighter_name.lower()
    for name, elo in _elo_ratings.items():
        if name.lower() == name_lower:
            return elo

    return ELO_INITIAL


# ---------------------------------------------------------------------------
# Main public API: build features for a fight
# ---------------------------------------------------------------------------

def lookup_fighter(fighter_name: str) -> Optional[dict]:
    """
    Look up a fighter's complete stats from UFCStats.com.

    Returns dict with profile info, fight history, and computed rolling stats.
    Caches results for the session to avoid redundant scraping.
    """
    if fighter_name in _fighter_cache:
        return _fighter_cache[fighter_name]

    logger.info(f"Looking up fighter stats: {fighter_name}")

    # Step 1: Find fighter URL
    fighter_url = search_fighter_url(fighter_name)
    if not fighter_url:
        logger.warning(f"Could not find {fighter_name} on UFCStats.com")
        return None

    # Step 2: Scrape profile
    try:
        profile = scrape_fighter_profile(fighter_url)
    except Exception as e:
        logger.warning(f"Failed to scrape profile for {fighter_name}: {e}")
        return None

    # Step 3: Scrape fight history
    try:
        fights = scrape_fighter_fights(fighter_url, fighter_name=profile.get("name", fighter_name))
    except Exception as e:
        logger.warning(f"Failed to scrape fights for {fighter_name}: {e}")
        fights = []

    # Step 4: Compute rolling stats
    rolling = _compute_rolling_for_fighter(fights, profile)

    # Step 5: Add Elo
    rolling["elo"] = get_fighter_elo(fighter_name)

    result = {
        "profile": profile,
        "fights": fights,
        "features": rolling,
    }

    _fighter_cache[fighter_name] = result
    logger.info(
        f"  {fighter_name}: {profile.get('record', '?')} | "
        f"Elo: {rolling['elo']:.0f} | "
        f"{len(fights)} fights found | "
        f"SLpM: {rolling.get('roll_slpm', '?')}"
    )
    return result


def build_fight_features(
    fighter_a: str,
    fighter_b: str,
    odds_features: Optional[dict] = None,
    weight_class: Optional[str] = None,
    is_title_bout: bool = False,
    num_rounds: int = 3,
) -> dict:
    """
    Build a complete feature dict for a fight, compatible with the trained model.

    Looks up both fighters on UFCStats.com, computes rolling stats, Elo,
    differentials, and combines with odds features.

    Args:
        fighter_a: Name of fighter A
        fighter_b: Name of fighter B
        odds_features: Dict with a_implied_prob, b_implied_prob, etc.
        weight_class: Weight class string
        is_title_bout: Whether this is a title fight
        num_rounds: Number of rounds (3 or 5)

    Returns:
        Dict of feature_name -> value, ready for predict_fight()
    """
    features = {}

    # Look up both fighters
    a_data = lookup_fighter(fighter_a)
    b_data = lookup_fighter(fighter_b)

    a_feats = a_data["features"] if a_data else {}
    b_feats = b_data["features"] if b_data else {}

    # Prefix features with a_/b_
    for key, val in a_feats.items():
        features[f"a_{key}"] = val
    for key, val in b_feats.items():
        features[f"b_{key}"] = val

    # Compute differentials
    diff_stats = [
        "roll_slpm", "roll_sapm", "roll_str_acc", "roll_str_def",
        "roll_td_avg", "roll_td_acc", "roll_td_def", "roll_sub_avg",
        "roll_sig_str_landed", "roll_td_landed", "roll_kd",
        "roll_won", "elo", "current_win_streak", "num_fights", "days_since_last_fight",
        "height", "reach", "weight", "age", "strike_diff",
        "ko_rate", "sub_rate", "dec_rate", "win_pct",
        "lose_streak", "longest_win_streak", "total_rounds", "title_bouts", "draws",
        "cage_rust", "layoff_log",
        "fight_pace", "ctrl_efficiency", "adj_win_pct",
    ]

    for stat in diff_stats:
        a_val = a_feats.get(stat)
        b_val = b_feats.get(stat)
        if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
            if not (np.isnan(a_val) or np.isnan(b_val)):
                features[f"diff_{stat}"] = a_val - b_val

    # Stance same?
    a_stance = a_feats.get("stance_enc", -1)
    b_stance = b_feats.get("stance_enc", -1)
    features["same_stance"] = int(a_stance == b_stance) if a_stance >= 0 and b_stance >= 0 else 0

    # Finish rate differentials
    for stat in ["ko_rate", "sub_rate", "dec_rate"]:
        a_val = a_feats.get(stat, 0)
        b_val = b_feats.get(stat, 0)
        features[f"diff_{stat}"] = (a_val or 0) - (b_val or 0)

    # Weight class encoding
    if weight_class:
        wc_order = {
            "strawweight": 0, "women's strawweight": 0,
            "flyweight": 1, "women's flyweight": 1,
            "bantamweight": 2, "women's bantamweight": 2,
            "featherweight": 3, "women's featherweight": 3,
            "lightweight": 4, "welterweight": 5, "middleweight": 6,
            "light heavyweight": 7, "heavyweight": 8, "catch weight": 5,
        }
        wc_lower = weight_class.lower()
        features["weight_class_enc"] = next(
            (v for k, v in wc_order.items() if k in wc_lower), 5
        )

    # Meta features
    features["is_title_bout"] = int(is_title_bout)
    features["num_rounds_feat"] = float(num_rounds)
    features["is_empty_arena"] = 0  # No more COVID empty arenas

    # Style matchup interactions
    for prefix_atk, prefix_def in [("a_", "b_"), ("b_", "a_")]:
        ko_rate = features.get(f"{prefix_atk}ko_rate", 0) or 0
        str_def_val = features.get(f"{prefix_def}roll_str_def", 50) or 50
        sub_rate = features.get(f"{prefix_atk}sub_rate", 0) or 0
        td_def_val = features.get(f"{prefix_def}roll_td_def", 50) or 50

        features[f"{prefix_atk}striker_edge"] = ko_rate * (1.0 - str_def_val / 100.0)
        features[f"{prefix_atk}grappler_edge"] = sub_rate * (1.0 - td_def_val / 100.0)

    features["diff_striker_edge"] = (
        features.get("a_striker_edge", 0) - features.get("b_striker_edge", 0)
    )
    features["diff_grappler_edge"] = (
        features.get("a_grappler_edge", 0) - features.get("b_grappler_edge", 0)
    )

    # Weight class moves (default: no move)
    features["a_wc_move"] = 0
    features["b_wc_move"] = 0
    features["diff_wc_move"] = 0

    # Add odds features last (these override if provided)
    if odds_features:
        features.update(odds_features)

    return features


def clear_cache():
    """Clear the fighter lookup cache."""
    _fighter_cache.clear()
    _fighter_url_cache.clear()
