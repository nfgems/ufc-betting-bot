"""
Incremental backfill — scrape UFC events after a given date from UFCStats.com
and produce rows matching the Kaggle ufc-master.csv schema.

This fills the gap between the last Kaggle snapshot (Dec 14, 2024) and
the current UFCStats event listing (Mar 14, 2026).
"""

import logging
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
REQUEST_DELAY = 1.5


def _get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return BeautifulSoup(resp.text, "lxml")


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


# ---------------------------------------------------------------------------
# Step 1: Get event URLs after cutoff date
# ---------------------------------------------------------------------------

def get_events_after(cutoff_date: str) -> list[dict]:
    """Get all completed UFC events after the given date."""
    cutoff = pd.Timestamp(cutoff_date)
    url = "http://ufcstats.com/statistics/events/completed?page=all"
    soup = _get_soup(url)
    rows = soup.select("tr.b-statistics__table-row")

    events = []
    for row in rows:
        link = row.select_one("a.b-link")
        date_el = row.select_one("span.b-statistics__date")
        if link and date_el:
            title = link.text.strip()
            date_str = date_el.text.strip()
            href = link["href"].strip()
            try:
                event_date = pd.Timestamp(date_str)
                if event_date > cutoff:
                    events.append({
                        "title": title,
                        "date": date_str,
                        "date_parsed": event_date,
                        "url": href,
                    })
            except Exception:
                continue

    events.sort(key=lambda x: x["date_parsed"])
    logger.info(f"Found {len(events)} events after {cutoff_date}")
    return events


# ---------------------------------------------------------------------------
# Step 2: Scrape fight details from event pages
# ---------------------------------------------------------------------------

def scrape_event_fights(event_url: str) -> list[str]:
    """Get fight URLs from an event page."""
    soup = _get_soup(event_url)
    fight_urls = []
    for row in soup.select("tr.b-fight-details__table-row"):
        link = row.get("data-link")
        if link:
            fight_urls.append(link.strip())
    return fight_urls


def _parse_stat_cell(cell_text: str) -> tuple[str, str]:
    text = _clean_text(cell_text)
    match = re.match(r"(\d+)\s+of\s+(\d+)", text)
    if match:
        return match.group(1), match.group(2)
    return text, text


def _parse_ctrl_seconds(ctrl_str: str) -> float:
    """Parse control time string like '3:45' to seconds."""
    text = _clean_text(ctrl_str)
    if text in ("--", "", "0:00"):
        return 0.0
    match = re.match(r"(\d+):(\d+)", text)
    if match:
        return float(int(match.group(1)) * 60 + int(match.group(2)))
    return 0.0


def scrape_fight_detail(fight_url: str) -> dict | None:
    """Scrape detailed stats for a single fight."""
    soup = _get_soup(fight_url)

    # Fighter names
    fighters = soup.select("h3.b-fight-details__person-name a")
    if len(fighters) < 2:
        return None
    fighter_a = _clean_text(fighters[0].text)
    fighter_b = _clean_text(fighters[1].text)

    # Winner
    results = soup.select("div.b-fight-details__person")
    winner = None
    winner_side = None
    if results:
        for i, res in enumerate(results[:2]):
            status = res.select_one("i.b-fight-details__person-status")
            if status and "W" in _clean_text(status.text):
                winner = fighter_a if i == 0 else fighter_b
                winner_side = "Red" if i == 0 else "Blue"

    # Method, round, time
    method_el = soup.select_one("i.b-fight-details__text-item_first")
    method = _clean_text(method_el.text.replace("Method:", "")) if method_el else ""

    details_items = soup.select("i.b-fight-details__text-item")
    round_num = ""
    fight_time = ""
    for item in details_items:
        text = _clean_text(item.text)
        if "Round:" in text:
            round_num = text.replace("Round:", "").strip()
        elif "Time:" in text:
            fight_time = text.replace("Time:", "").strip()

    # Weight class
    weight_class = ""
    wc_el = soup.select_one("i.b-fight-details__fight-title")
    if wc_el:
        weight_class = _clean_text(wc_el.text)

    # Determine title bout and num rounds from weight class text
    is_title = "Title" in weight_class or "title" in weight_class
    # Clean weight class name
    wc_clean = weight_class
    for removal in ["UFC", "Interim", "Title", "Bout", "bout"]:
        wc_clean = wc_clean.replace(removal, "")
    wc_clean = _clean_text(wc_clean).strip("' ")

    fight_data = {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "winner": winner,
        "winner_side": winner_side,
        "method": method,
        "round": round_num,
        "time": fight_time,
        "weight_class": wc_clean,
        "weight_class_raw": weight_class,
        "is_title_bout": is_title,
        "fight_url": fight_url,
    }

    # Parse totals table
    tables = soup.select("table.b-fight-details__table")
    if tables:
        totals_table = tables[0]
        rows = totals_table.select("tr.b-fight-details__table-row")
        for row in rows:
            cols = row.select("td.b-fight-details__table-col")
            if len(cols) >= 10:
                p_tags = [col.select("p") for col in cols]
                if all(len(p) >= 2 for p in p_tags[1:]):
                    for idx, prefix in [(0, "a_"), (1, "b_")]:
                        kd = _clean_text(p_tags[1][idx].text) if len(p_tags[1]) > idx else "0"
                        sig_l, sig_a = _parse_stat_cell(p_tags[2][idx].text) if len(p_tags[2]) > idx else ("0", "0")
                        sig_pct = _clean_text(p_tags[3][idx].text).replace("%", "") if len(p_tags[3]) > idx else "0"
                        total_l, total_a = _parse_stat_cell(p_tags[4][idx].text) if len(p_tags[4]) > idx else ("0", "0")
                        td_l, td_a = _parse_stat_cell(p_tags[5][idx].text) if len(p_tags[5]) > idx else ("0", "0")
                        td_pct = _clean_text(p_tags[6][idx].text).replace("%", "") if len(p_tags[6]) > idx else "0"
                        sub_att = _clean_text(p_tags[7][idx].text) if len(p_tags[7]) > idx else "0"
                        rev = _clean_text(p_tags[8][idx].text) if len(p_tags[8]) > idx else "0"
                        ctrl = _clean_text(p_tags[9][idx].text) if len(p_tags[9]) > idx else "0:00"

                        fight_data.update({
                            f"{prefix}kd": _safe_int(kd),
                            f"{prefix}sig_str_landed": _safe_int(sig_l),
                            f"{prefix}sig_str_attempted": _safe_int(sig_a),
                            f"{prefix}td_landed": _safe_int(td_l),
                            f"{prefix}td_attempted": _safe_int(td_a),
                            f"{prefix}sub_att": _safe_int(sub_att),
                            f"{prefix}rev": _safe_int(rev),
                            f"{prefix}ctrl_seconds": _parse_ctrl_seconds(ctrl),
                        })
                    break

    return fight_data


def _safe_int(val) -> int:
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Step 3: Scrape fighter profiles for physical attributes
# ---------------------------------------------------------------------------

def scrape_fighter_profile(fighter_url: str) -> dict | None:
    """Scrape a single fighter's profile for physical attributes."""
    soup = _get_soup(fighter_url)

    name_el = soup.select_one("h2.b-content__title span")
    if not name_el:
        return None
    name = _clean_text(name_el.text)

    record_el = soup.select_one("span.b-content__title-record")
    record = _clean_text(record_el.text.replace("Record:", "")) if record_el else ""

    info = {"name": name, "record": record}
    for item in soup.select("li.b-list__box-list-item"):
        text = _clean_text(item.text)
        if "Height:" in text:
            info["height"] = text.replace("Height:", "").strip()
        elif "Weight:" in text:
            info["weight"] = text.replace("Weight:", "").strip()
        elif "Reach:" in text:
            info["reach"] = text.replace("Reach:", "").strip()
        elif "STANCE:" in text:
            info["stance"] = text.replace("STANCE:", "").strip()
        elif "DOB:" in text:
            info["dob"] = text.replace("DOB:", "").strip()

    # Career stats
    for item in soup.select("li.b-list__box-list-item_type_block"):
        text = _clean_text(item.text)
        if "SLpM:" in text:
            info["slpm"] = _safe_float(text.replace("SLpM:", "").strip())
        elif "Str. Acc.:" in text:
            info["str_acc"] = _safe_float(text.replace("Str. Acc.:", "").strip().replace("%", ""))
        elif "SApM:" in text:
            info["sapm"] = _safe_float(text.replace("SApM:", "").strip())
        elif "Str. Def:" in text:
            info["str_def"] = _safe_float(text.replace("Str. Def:", "").strip().replace("%", ""))
        elif "TD Avg.:" in text:
            info["td_avg"] = _safe_float(text.replace("TD Avg.:", "").strip())
        elif "TD Acc.:" in text:
            info["td_acc"] = _safe_float(text.replace("TD Acc.:", "").strip().replace("%", ""))
        elif "TD Def.:" in text:
            info["td_def"] = _safe_float(text.replace("TD Def.:", "").strip().replace("%", ""))
        elif "Sub. Avg.:" in text:
            info["sub_avg"] = _safe_float(text.replace("Sub. Avg.:", "").strip())

    return info


def _safe_float(val) -> float | None:
    if pd.isna(val) or str(val).strip() in ("", "--"):
        return None
    try:
        return float(str(val).strip())
    except ValueError:
        return None


def _parse_height_inches(h_str: str) -> float | None:
    """Parse height like 5' 11\" to inches, then to cm."""
    if not h_str or h_str.strip() in ("", "--"):
        return None
    match = re.match(r"(\d+)'?\s*(\d+)?\"?", h_str.strip())
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2)) if match.group(2) else 0
        return round((feet * 12 + inches) * 2.54, 2)
    return None


def _parse_reach_inches(r_str: str) -> float | None:
    """Parse reach like 72\" to cm."""
    if not r_str or r_str.strip() in ("", "--"):
        return None
    cleaned = r_str.strip().replace('"', '').replace("'", "")
    try:
        return round(float(cleaned) * 2.54, 2)
    except ValueError:
        return None


def _parse_weight_lbs(w_str: str) -> float | None:
    """Parse weight like '155 lbs.' to float."""
    if not w_str or w_str.strip() in ("", "--"):
        return None
    cleaned = w_str.strip().lower().replace("lbs.", "").replace("lbs", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_dob_to_age(dob_str: str, event_date: pd.Timestamp) -> float | None:
    """Parse DOB and compute age at event date."""
    if not dob_str or dob_str.strip() in ("", "--"):
        return None
    try:
        dob = pd.Timestamp(dob_str)
        age = (event_date - dob).days / 365.25
        return round(age, 0)
    except Exception:
        return None


def get_fighter_urls_from_event(event_url: str) -> dict[str, str]:
    """Get fighter name -> profile URL mapping from an event page."""
    soup = _get_soup(event_url)
    fighter_urls = {}
    # Fight detail pages have fighter links
    for row in soup.select("tr.b-fight-details__table-row"):
        for link in row.select("a.b-link.b-link_style_black"):
            href = link.get("href", "")
            if "fighter-details" in href:
                name = _clean_text(link.text)
                fighter_urls[name] = href.strip()
    return fighter_urls


# ---------------------------------------------------------------------------
# Step 4: Compute career stats at fight time from fight sequence
# ---------------------------------------------------------------------------

def compute_career_stats_at_fight_time(
    existing_fights: pd.DataFrame,
    new_fights: list[dict],
) -> list[dict]:
    """
    Compute career record stats at the time of each new fight.

    Uses the full fight history (existing + new, processed chronologically)
    to compute wins, losses, streaks, finish method counts, etc.
    """
    # Build a chronological sequence of all fights
    existing_records = []
    for _, row in existing_fights.iterrows():
        existing_records.append({
            "event_date": pd.Timestamp(row.get("event_date")),
            "fighter_a": str(row.get("fighter_a", row.get("RedFighter", ""))).strip(),
            "fighter_b": str(row.get("fighter_b", row.get("BlueFighter", ""))).strip(),
            "winner": str(row.get("winner", row.get("Winner", ""))).strip(),
            "method": str(row.get("method", row.get("Finish", ""))).strip(),
        })

    # Track career stats per fighter
    career: dict[str, dict] = defaultdict(lambda: {
        "wins": 0, "losses": 0, "draws": 0,
        "wins_ko": 0, "wins_sub": 0, "wins_dec": 0,
        "wins_dec_maj": 0, "wins_dec_split": 0, "wins_tko_doc": 0,
        "current_win_streak": 0, "current_lose_streak": 0,
        "longest_win_streak": 0,
        "total_rounds": 0, "title_bouts": 0,
    })

    # Process all existing fights to build career state
    for rec in sorted(existing_records, key=lambda x: x["event_date"]):
        fa = rec["fighter_a"]
        fb = rec["fighter_b"]
        winner = rec["winner"]
        method = str(rec.get("method", "")).lower()

        for fighter, opponent in [(fa, fb), (fb, fa)]:
            if not fighter:
                continue
            if winner == fighter:
                career[fighter]["wins"] += 1
                career[fighter]["current_win_streak"] += 1
                career[fighter]["current_lose_streak"] = 0
                if career[fighter]["current_win_streak"] > career[fighter]["longest_win_streak"]:
                    career[fighter]["longest_win_streak"] = career[fighter]["current_win_streak"]
                # Classify win method
                if "ko" in method or "tko" in method:
                    career[fighter]["wins_ko"] += 1
                elif "sub" in method:
                    career[fighter]["wins_sub"] += 1
                elif "dec" in method:
                    career[fighter]["wins_dec"] += 1
                    if "split" in method:
                        career[fighter]["wins_dec_split"] += 1
                    elif "majority" in method:
                        career[fighter]["wins_dec_maj"] += 1
            elif winner == opponent:
                career[fighter]["losses"] += 1
                career[fighter]["current_lose_streak"] += 1
                career[fighter]["current_win_streak"] = 0
            else:
                # Draw or NC
                career[fighter]["draws"] += 1

    logger.info(f"Processed {len(existing_records)} existing fights, tracking {len(career)} fighters")

    # Now process new fights and record career stats AT fight time
    enriched = []
    for fight in sorted(new_fights, key=lambda x: x.get("event_date", pd.Timestamp.min)):
        fa = fight["fighter_a"]
        fb = fight["fighter_b"]

        # Snapshot career stats BEFORE this fight
        fight["a_wins"] = career[fa]["wins"]
        fight["a_losses"] = career[fa]["losses"]
        fight["a_draws"] = career[fa]["draws"]
        fight["a_wins_ko"] = career[fa]["wins_ko"]
        fight["a_wins_sub"] = career[fa]["wins_sub"]
        fight["a_wins_dec"] = career[fa]["wins_dec"]
        fight["a_wins_dec_maj"] = career[fa]["wins_dec_maj"]
        fight["a_wins_dec_split"] = career[fa]["wins_dec_split"]
        fight["a_wins_tko_doc"] = career[fa]["wins_tko_doc"]
        fight["a_win_streak"] = career[fa]["current_win_streak"]
        fight["a_lose_streak"] = career[fa]["current_lose_streak"]
        fight["a_longest_win_streak"] = career[fa]["longest_win_streak"]
        fight["a_total_rounds"] = career[fa]["total_rounds"]
        fight["a_title_bouts"] = career[fa]["title_bouts"]

        fight["b_wins"] = career[fb]["wins"]
        fight["b_losses"] = career[fb]["losses"]
        fight["b_draws"] = career[fb]["draws"]
        fight["b_wins_ko"] = career[fb]["wins_ko"]
        fight["b_wins_sub"] = career[fb]["wins_sub"]
        fight["b_wins_dec"] = career[fb]["wins_dec"]
        fight["b_wins_dec_maj"] = career[fb]["wins_dec_maj"]
        fight["b_wins_dec_split"] = career[fb]["wins_dec_split"]
        fight["b_wins_tko_doc"] = career[fb]["wins_tko_doc"]
        fight["b_win_streak"] = career[fb]["current_win_streak"]
        fight["b_lose_streak"] = career[fb]["current_lose_streak"]
        fight["b_longest_win_streak"] = career[fb]["longest_win_streak"]
        fight["b_total_rounds"] = career[fb]["total_rounds"]
        fight["b_title_bouts"] = career[fb]["title_bouts"]

        enriched.append(fight)

        # Update career stats AFTER this fight
        winner = fight.get("winner")
        method = str(fight.get("method", "")).lower()
        finish_round = fight.get("round", "3")
        try:
            rounds_fought = int(finish_round)
        except (ValueError, TypeError):
            rounds_fought = 3

        for fighter, opponent in [(fa, fb), (fb, fa)]:
            career[fighter]["total_rounds"] += rounds_fought
            if fight.get("is_title_bout"):
                career[fighter]["title_bouts"] += 1

            if winner == fighter:
                career[fighter]["wins"] += 1
                career[fighter]["current_win_streak"] += 1
                career[fighter]["current_lose_streak"] = 0
                if career[fighter]["current_win_streak"] > career[fighter]["longest_win_streak"]:
                    career[fighter]["longest_win_streak"] = career[fighter]["current_win_streak"]
                if "ko" in method or "tko" in method:
                    career[fighter]["wins_ko"] += 1
                elif "sub" in method:
                    career[fighter]["wins_sub"] += 1
                elif "dec" in method:
                    career[fighter]["wins_dec"] += 1
                    if "split" in method:
                        career[fighter]["wins_dec_split"] += 1
                    elif "majority" in method:
                        career[fighter]["wins_dec_maj"] += 1
            elif winner == opponent:
                career[fighter]["losses"] += 1
                career[fighter]["current_lose_streak"] += 1
                career[fighter]["current_win_streak"] = 0
            else:
                career[fighter]["draws"] += 1

    return enriched


# ---------------------------------------------------------------------------
# Step 5: Build Kaggle-schema rows
# ---------------------------------------------------------------------------

def build_kaggle_rows(
    enriched_fights: list[dict],
    fighter_profiles: dict[str, dict],
) -> pd.DataFrame:
    """Convert enriched fight data to Kaggle CSV schema."""
    rows = []
    for fight in enriched_fights:
        fa = fight["fighter_a"]
        fb = fight["fighter_b"]
        prof_a = fighter_profiles.get(fa, {})
        prof_b = fighter_profiles.get(fb, {})
        event_date = fight.get("event_date", pd.NaT)

        row = {
            "RedFighter": fa,
            "BlueFighter": fb,
            "Date": event_date.strftime("%Y-%m-%d") if pd.notna(event_date) else "",
            "Winner": fight.get("winner_side", ""),
            "WeightClass": fight.get("weight_class", ""),
            "TitleBout": fight.get("is_title_bout", False),
            "NumberOfRounds": 5 if fight.get("is_title_bout") else 3,
            "Gender": "MALE",  # Default; women's detected by weight class
            "Finish": fight.get("method", ""),
            "FinishRound": fight.get("round", ""),
            "FinishRoundTime": fight.get("time", ""),
            "EmptyArena": False,

            # Physical attributes from profiles
            "RedHeightCms": _parse_height_inches(prof_a.get("height", "")),
            "RedReachCms": _parse_reach_inches(prof_a.get("reach", "")),
            "RedWeightLbs": _parse_weight_lbs(prof_a.get("weight", "")),
            "RedAge": _parse_dob_to_age(prof_a.get("dob", ""), event_date) if pd.notna(event_date) else None,
            "RedStance": prof_a.get("stance", ""),

            "BlueHeightCms": _parse_height_inches(prof_b.get("height", "")),
            "BlueReachCms": _parse_reach_inches(prof_b.get("reach", "")),
            "BlueWeightLbs": _parse_weight_lbs(prof_b.get("weight", "")),
            "BlueAge": _parse_dob_to_age(prof_b.get("dob", ""), event_date) if pd.notna(event_date) else None,
            "BlueStance": prof_b.get("stance", ""),

            # Career stats from profiles
            "RedAvgSigStrLanded": prof_a.get("slpm"),
            "RedAvgSigStrPct": prof_a.get("str_acc"),
            "RedAvgSubAtt": prof_a.get("sub_avg"),
            "RedAvgTDLanded": prof_a.get("td_avg"),
            "RedAvgTDPct": prof_a.get("td_acc"),

            "BlueAvgSigStrLanded": prof_b.get("slpm"),
            "BlueAvgSigStrPct": prof_b.get("str_acc"),
            "BlueAvgSubAtt": prof_b.get("sub_avg"),
            "BlueAvgTDLanded": prof_b.get("td_avg"),
            "BlueAvgTDPct": prof_b.get("td_acc"),

            # Career record at fight time (computed)
            "RedWins": fight.get("a_wins", 0),
            "RedLosses": fight.get("a_losses", 0),
            "RedDraws": fight.get("a_draws", 0),
            "RedWinsByKO": fight.get("a_wins_ko", 0),
            "RedWinsBySubmission": fight.get("a_wins_sub", 0),
            "RedWinsByDecisionUnanimous": fight.get("a_wins_dec", 0),
            "RedWinsByDecisionMajority": fight.get("a_wins_dec_maj", 0),
            "RedWinsByDecisionSplit": fight.get("a_wins_dec_split", 0),
            "RedWinsByTKODoctorStoppage": fight.get("a_wins_tko_doc", 0),
            "RedCurrentWinStreak": fight.get("a_win_streak", 0),
            "RedCurrentLoseStreak": fight.get("a_lose_streak", 0),
            "RedLongestWinStreak": fight.get("a_longest_win_streak", 0),
            "RedTotalRoundsFought": fight.get("a_total_rounds", 0),
            "RedTotalTitleBouts": fight.get("a_title_bouts", 0),

            "BlueWins": fight.get("b_wins", 0),
            "BlueLosses": fight.get("b_losses", 0),
            "BlueDraws": fight.get("b_draws", 0),
            "BlueWinsByKO": fight.get("b_wins_ko", 0),
            "BlueWinsBySubmission": fight.get("b_wins_sub", 0),
            "BlueWinsByDecisionUnanimous": fight.get("b_wins_dec", 0),
            "BlueWinsByDecisionMajority": fight.get("b_wins_dec_maj", 0),
            "BlueWinsByDecisionSplit": fight.get("b_wins_dec_split", 0),
            "BlueWinsByTKODoctorStoppage": fight.get("b_wins_tko_doc", 0),
            "BlueCurrentWinStreak": fight.get("b_win_streak", 0),
            "BlueCurrentLoseStreak": fight.get("b_lose_streak", 0),
            "BlueLongestWinStreak": fight.get("b_longest_win_streak", 0),
            "BlueTotalRoundsFought": fight.get("b_total_rounds", 0),
            "BlueTotalTitleBouts": fight.get("b_title_bouts", 0),

            # Odds — not available from scraping, will be NaN
            "RedOdds": np.nan,
            "BlueOdds": np.nan,
            "RedExpectedValue": np.nan,
            "BlueExpectedValue": np.nan,

            # Rankings — not available from scraping
            "RMatchWCRank": np.nan,
            "BMatchWCRank": np.nan,
            "RPFPRank": np.nan,
            "BPFPRank": np.nan,

            # Method odds — not available
            "RedDecOdds": np.nan,
            "BlueDecOdds": np.nan,
            "RSubOdds": np.nan,
            "BSubOdds": np.nan,
            "RKOOdds": np.nan,
            "BKOOdds": np.nan,
        }

        # Detect gender from weight class
        wc = str(fight.get("weight_class", "")).lower()
        if "women" in wc or "w." in wc:
            row["Gender"] = "FEMALE"

        # Detect num_rounds from title bout or main event position
        if fight.get("is_title_bout"):
            row["NumberOfRounds"] = 5

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_incremental_backfill(cutoff_date: str = "2024-12-14"):
    """Run the full incremental backfill pipeline."""
    logger.info(f"Starting incremental backfill from {cutoff_date}")

    # Step 1: Get events after cutoff
    events = get_events_after(cutoff_date)
    if not events:
        logger.info("No new events to backfill")
        return

    # Step 2: Scrape fights from each event
    all_fights = []
    all_fighter_urls = {}
    progress_path = RAW_DATA_DIR / "backfill_progress.csv"

    for i, event in enumerate(events):
        logger.info(f"[{i+1}/{len(events)}] Scraping {event['title']} ({event['date']})")

        # Get fight URLs
        fight_urls = scrape_event_fights(event["url"])
        logger.info(f"  Found {len(fight_urls)} fights")

        # Also collect fighter profile URLs from event page
        fighter_urls = get_fighter_urls_from_event(event["url"])
        all_fighter_urls.update(fighter_urls)

        for j, fight_url in enumerate(fight_urls):
            try:
                fight = scrape_fight_detail(fight_url)
                if fight:
                    fight["event_title"] = event["title"]
                    fight["event_date"] = event["date_parsed"]
                    all_fights.append(fight)
                    logger.info(f"    [{j+1}/{len(fight_urls)}] {fight['fighter_a']} vs {fight['fighter_b']}: {fight.get('winner', 'N/A')}")
            except Exception as e:
                logger.warning(f"    Failed to scrape fight {fight_url}: {e}")

        # Save progress every 5 events
        if (i + 1) % 5 == 0:
            pd.DataFrame(all_fights).to_csv(progress_path, index=False)
            logger.info(f"  Progress saved: {len(all_fights)} fights scraped")

    logger.info(f"Scraped {len(all_fights)} total fights from {len(events)} events")

    # Step 3: Scrape fighter profiles
    # Collect unique fighter names from new fights
    new_fighters = set()
    for fight in all_fights:
        new_fighters.add(fight["fighter_a"])
        new_fighters.add(fight["fighter_b"])

    logger.info(f"Scraping profiles for {len(new_fighters)} fighters...")
    fighter_profiles = {}
    fighters_scraped = 0

    for name in sorted(new_fighters):
        if name in all_fighter_urls:
            try:
                profile = scrape_fighter_profile(all_fighter_urls[name])
                if profile:
                    fighter_profiles[name] = profile
                    fighters_scraped += 1
                    if fighters_scraped % 20 == 0:
                        logger.info(f"  Scraped {fighters_scraped}/{len(new_fighters)} fighter profiles")
            except Exception as e:
                logger.warning(f"  Failed to scrape profile for {name}: {e}")
        else:
            # Try to find fighter URL from the alphabetical listing
            logger.debug(f"  No URL found for {name}, skipping profile")

    logger.info(f"Scraped {len(fighter_profiles)} fighter profiles")

    # Step 4: Compute career stats at fight time
    logger.info("Computing career stats at fight time...")
    existing_path = RAW_DATA_DIR / "ufc-master.csv"
    existing_df = pd.read_csv(existing_path)
    # Map Kaggle columns to standard names for career computation
    existing_mapped = existing_df.rename(columns={
        "RedFighter": "fighter_a",
        "BlueFighter": "fighter_b",
        "Winner": "winner",
        "Date": "event_date",
        "Finish": "method",
    })
    # Handle Red/Blue winner format
    winner_vals = existing_mapped["winner"].astype(str).str.strip().str.lower()
    mask_red = winner_vals == "red"
    mask_blue = winner_vals == "blue"
    existing_mapped.loc[mask_red, "winner"] = existing_mapped.loc[mask_red, "fighter_a"]
    existing_mapped.loc[mask_blue, "winner"] = existing_mapped.loc[mask_blue, "fighter_b"]
    existing_mapped["event_date"] = pd.to_datetime(existing_mapped["event_date"], errors="coerce")

    enriched = compute_career_stats_at_fight_time(existing_mapped, all_fights)

    # Step 5: Build Kaggle-schema rows
    logger.info("Building Kaggle-schema rows...")
    new_rows_df = build_kaggle_rows(enriched, fighter_profiles)

    # Step 6: Append to master CSV
    logger.info(f"Appending {len(new_rows_df)} new rows to ufc-master.csv")
    combined = pd.concat([existing_df, new_rows_df], ignore_index=True)
    # Sort by date
    combined["_date_sort"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.sort_values("_date_sort").drop(columns=["_date_sort"])
    combined.to_csv(existing_path, index=False)

    logger.info(f"Updated ufc-master.csv: {len(existing_df)} → {len(combined)} rows")
    logger.info(f"  New fights added: {len(new_rows_df)}")
    logger.info(f"  Date range: {new_rows_df['Date'].min()} → {new_rows_df['Date'].max()}")

    # Clean up progress file
    if progress_path.exists():
        progress_path.unlink()

    return new_rows_df


if __name__ == "__main__":
    run_incremental_backfill()
