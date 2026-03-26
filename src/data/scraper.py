"""
UFCStats.com scraper — collects fighter stats and fight results.
"""

import re
import time
import logging
from typing import Optional
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup

from src.config import (
    UFCSTATS_BASE_URL,
    UFCSTATS_EVENT_URL,
    UFCSTATS_FIGHT_URL,
    UFCSTATS_FIGHTER_SEARCH_URL,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
REQUEST_DELAY = 1.5  # seconds between requests to be polite
FIGHTER_DETAILS_PATH = RAW_DATA_DIR / "ufc-fighter-details.csv"


def _get_soup(url: str) -> BeautifulSoup:
    """Fetch a URL and return parsed BeautifulSoup."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return BeautifulSoup(resp.text, "lxml")


def _clean_text(text: str) -> str:
    """Strip whitespace and normalize."""
    return re.sub(r"\s+", " ", text.strip())


def _load_existing_rows(output_path: Path, key_column: str) -> tuple[list[dict], set[str]]:
    """Load an existing scrape artifact so interrupted runs can resume safely."""
    if not output_path.exists():
        return [], set()

    try:
        existing_df = pd.read_csv(output_path)
    except Exception as exc:
        logger.warning("Failed to load existing scrape artifact %s: %s", output_path, exc)
        return [], set()

    if existing_df.empty or key_column not in existing_df.columns:
        return [], set()

    keyed = existing_df.copy()
    keyed[key_column] = keyed[key_column].astype(str).str.strip()
    keyed = keyed[keyed[key_column] != ""]
    if keyed.empty:
        return [], set()

    keyed = keyed.drop_duplicates(subset=[key_column], keep="last")
    return keyed.to_dict("records"), set(keyed[key_column].tolist())


# ---------------------------------------------------------------------------
# Event scraping
# ---------------------------------------------------------------------------

def scrape_all_event_urls() -> list[str]:
    """Get URLs for all completed UFC events."""
    event_urls = []
    page = 1
    while True:
        url = f"{UFCSTATS_BASE_URL}?action=getEventsResults&page={page}"
        soup = _get_soup(url)
        rows = soup.select("tr.b-statistics__table-row")
        if not rows:
            break
        for row in rows:
            link = row.select_one("a.b-link")
            if link and link.get("href"):
                event_urls.append(link["href"].strip())
        page += 1
    logger.info(f"Found {len(event_urls)} events")
    return event_urls


def scrape_event(event_url: str) -> dict:
    """Scrape a single event page for metadata and fight URLs."""
    soup = _get_soup(event_url)
    title_el = soup.select_one("h2.b-content__title span")
    date_el = soup.select_one("li.b-list__box-list-item")

    title = _clean_text(title_el.text) if title_el else "Unknown"
    date_str = ""
    if date_el:
        date_str = _clean_text(date_el.text.replace("Date:", "").strip())

    fight_urls = []
    for row in soup.select("tr.b-fight-details__table-row"):
        link = row.get("data-link")
        if link:
            fight_urls.append(link.strip())

    return {"title": title, "date": date_str, "fight_urls": fight_urls}


# ---------------------------------------------------------------------------
# Fight scraping
# ---------------------------------------------------------------------------

def _parse_stat_cell(cell_text: str) -> tuple[str, str]:
    """Parse a cell like '50 of 100' into (landed, attempted) or return (value, value)."""
    text = _clean_text(cell_text)
    match = re.match(r"(\d+)\s+of\s+(\d+)", text)
    if match:
        return match.group(1), match.group(2)
    if text:
        logger.debug("Stat cell %r has no 'X of Y' format — using value as both landed and attempted", text)
    return text, text


def scrape_fight(fight_url: str) -> Optional[dict]:
    """Scrape detailed stats for a single fight."""
    soup = _get_soup(fight_url)

    # Fighter names
    fighters = soup.select("h3.b-fight-details__person-name a")
    if len(fighters) < 2:
        return None
    fighter_a = _clean_text(fighters[0].text)
    fighter_b = _clean_text(fighters[1].text)

    # Winner / result type
    results = soup.select("div.b-fight-details__person")
    winner = None
    result_type = None  # "win", "draw", "nc", or None
    if results:
        for i, res in enumerate(results[:2]):
            status = res.select_one("i.b-fight-details__person-status")
            if status:
                status_text = _clean_text(status.text).upper()
                if "W" in status_text:
                    winner = fighter_a if i == 0 else fighter_b
                    result_type = "win"
                elif "D" in status_text:
                    result_type = "draw"
                elif "NC" in status_text:
                    result_type = "nc"
    if result_type in ("draw", "nc"):
        logger.info(f"Non-standard result ({result_type}): {fighter_a} vs {fighter_b}")

    # Fight details (method, round, time, weight class)
    method_el = soup.select_one("i.b-fight-details__text-item_first")
    method = _clean_text(method_el.text.replace("Method:", "")) if method_el else ""

    details_items = soup.select("i.b-fight-details__text-item")
    round_num = ""
    fight_time = ""
    weight_class = ""

    for item in details_items:
        text = _clean_text(item.text)
        if "Round:" in text:
            round_num = text.replace("Round:", "").strip()
        elif "Time:" in text:
            fight_time = text.replace("Time:", "").strip()

    wc_el = soup.select_one("i.b-fight-details__fight-title")
    if wc_el:
        weight_class = _clean_text(wc_el.text)

    # Totals table (first table = totals)
    tables = soup.select("table.b-fight-details__table")
    fight_data = {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "winner": winner,
        "result_type": result_type,
        "method": method,
        "round": round_num,
        "time": fight_time,
        "weight_class": weight_class,
        "fight_url": fight_url,
    }

    for prefix in ("a_", "b_"):
        fight_data.update(
            {
                f"{prefix}head_landed": "",
                f"{prefix}head_attempted": "",
                f"{prefix}body_landed": "",
                f"{prefix}body_attempted": "",
                f"{prefix}leg_landed": "",
                f"{prefix}leg_attempted": "",
                f"{prefix}distance_landed": "",
                f"{prefix}distance_attempted": "",
                f"{prefix}clinch_landed": "",
                f"{prefix}clinch_attempted": "",
                f"{prefix}ground_landed": "",
                f"{prefix}ground_attempted": "",
            }
        )

    # Live UFCStats pages now render the summary tables without a table class.
    # Filter out the per-round tables (js-fight-table) and use the two summary
    # tables for totals + significant-strike breakdowns.
    summary_tables = [table for table in soup.select("table") if "js-fight-table" not in (table.get("class") or [])]

    if summary_tables:
        totals_table = summary_tables[0]
        rows = totals_table.select("tbody tr.b-fight-details__table-row") or totals_table.select("tbody tr")
        for row in rows:
            cols = row.select("td")
            if len(cols) >= 10:
                p_tags = [col.select("p") for col in cols]
                if len(p_tags) >= 10 and all(len(p) >= 2 for p in p_tags[:10]):
                    for prefix, idx in [("a_", 0), ("b_", 1)]:
                        kd = _clean_text(p_tags[1][idx].text)
                        sig_str_l, sig_str_a = _parse_stat_cell(p_tags[2][idx].text)
                        sig_str_pct = _clean_text(p_tags[3][idx].text).replace("%", "")
                        total_str_l, total_str_a = _parse_stat_cell(p_tags[4][idx].text)
                        td_l, td_a = _parse_stat_cell(p_tags[5][idx].text)
                        td_pct = _clean_text(p_tags[6][idx].text).replace("%", "")
                        sub_att = _clean_text(p_tags[7][idx].text)
                        rev = _clean_text(p_tags[8][idx].text)
                        ctrl = _clean_text(p_tags[9][idx].text)

                        fight_data.update({
                            f"{prefix}kd": kd,
                            f"{prefix}sig_str_landed": sig_str_l,
                            f"{prefix}sig_str_attempted": sig_str_a,
                            f"{prefix}sig_str_pct": sig_str_pct,
                            f"{prefix}total_str_landed": total_str_l,
                            f"{prefix}total_str_attempted": total_str_a,
                            f"{prefix}td_landed": td_l,
                            f"{prefix}td_attempted": td_a,
                            f"{prefix}td_pct": td_pct,
                            f"{prefix}sub_att": sub_att,
                            f"{prefix}rev": rev,
                            f"{prefix}ctrl": ctrl,
                        })
                break  # Only need the totals summary row

        if len(summary_tables) >= 2:
            sig_table = summary_tables[1]
            sig_rows = sig_table.select("tbody tr.b-fight-details__table-row") or sig_table.select("tbody tr")
            if sig_rows:
                sig_cols = sig_rows[0].select("td")
                breakdown_map = [
                    (3, "head"),
                    (4, "body"),
                    (5, "leg"),
                    (6, "distance"),
                    (7, "clinch"),
                    (8, "ground"),
                ]
                for prefix, idx in [("a_", 0), ("b_", 1)]:
                    for col_idx, stat_name in breakdown_map:
                        if col_idx >= len(sig_cols):
                            continue
                        ps = sig_cols[col_idx].select("p")
                        if len(ps) < 2:
                            continue
                        landed, attempted = _parse_stat_cell(ps[idx].text)
                        fight_data[f"{prefix}{stat_name}_landed"] = landed
                        fight_data[f"{prefix}{stat_name}_attempted"] = attempted

    return fight_data


# ---------------------------------------------------------------------------
# Fighter details scraping
# ---------------------------------------------------------------------------

def scrape_all_fighter_urls() -> list[str]:
    """Get URLs for all fighters from the alphabetical listing."""
    fighter_urls = []
    for char in "abcdefghijklmnopqrstuvwxyz":
        url = f"{UFCSTATS_FIGHTER_SEARCH_URL}?char={char}&page=all"
        soup = _get_soup(url)
        for row in soup.select("tr.b-statistics__table-row"):
            link = row.select_one("a.b-link")
            if link and link.get("href") and "fighter-details" in link["href"]:
                fighter_urls.append(link["href"].strip())
    logger.info(f"Found {len(fighter_urls)} fighters")
    return list(set(fighter_urls))


def _load_fighter_inventory_urls(details_path: Optional[Path] = None) -> list[str]:
    """Load fighter profile URLs from the repo's canonical fighter inventory."""
    details_path = Path(details_path) if details_path is not None else FIGHTER_DETAILS_PATH
    if not details_path.exists():
        return []

    details_df = pd.read_csv(details_path)
    if details_df.empty or "URL" not in details_df.columns:
        return []

    fighter_urls: list[str] = []
    seen: set[str] = set()
    for value in details_df["URL"]:
        if value is None or pd.isna(value):
            continue
        url = str(value).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        fighter_urls.append(url)
    return fighter_urls


def scrape_fighter(fighter_url: str) -> Optional[dict]:
    """Scrape a single fighter's profile page."""
    soup = _get_soup(fighter_url)

    name_el = soup.select_one("h2.b-content__title span")
    if not name_el:
        return None
    name = _clean_text(name_el.text)

    record_el = soup.select_one("span.b-content__title-record")
    record = _clean_text(record_el.text.replace("Record:", "")) if record_el else ""

    info = {}
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
    stats = {}
    for item in soup.select("li.b-list__box-list-item_type_block"):
        text = _clean_text(item.text)
        if "SLpM:" in text:
            stats["slpm"] = text.replace("SLpM:", "").strip()
        elif "Str. Acc.:" in text:
            stats["str_acc"] = text.replace("Str. Acc.:", "").strip().replace("%", "")
        elif "SApM:" in text:
            stats["sapm"] = text.replace("SApM:", "").strip()
        elif "Str. Def:" in text:
            stats["str_def"] = text.replace("Str. Def:", "").strip().replace("%", "")
        elif "TD Avg.:" in text:
            stats["td_avg"] = text.replace("TD Avg.:", "").strip()
        elif "TD Acc.:" in text:
            stats["td_acc"] = text.replace("TD Acc.:", "").strip().replace("%", "")
        elif "TD Def.:" in text:
            stats["td_def"] = text.replace("TD Def.:", "").strip().replace("%", "")
        elif "Sub. Avg.:" in text:
            stats["sub_avg"] = text.replace("Sub. Avg.:", "").strip()

    return {
        "name": name,
        "record": record,
        "fighter_url": fighter_url,
        **info,
        **stats,
    }


def _scrape_fighter_url_batch(fighter_urls: list[str], *, output_path: Path) -> pd.DataFrame:
    """Scrape a fixed list of fighter profile URLs into a CSV artifact."""
    existing_rows, existing_urls = _load_existing_rows(output_path, "fighter_url")
    all_fighters = list(existing_rows)
    remaining_urls = [url for url in fighter_urls if url not in existing_urls]

    if existing_urls:
        logger.info(
            "Resuming fighter scrape from %d/%d complete using %s",
            len(existing_urls),
            len(fighter_urls),
            output_path,
        )
    if not remaining_urls:
        logger.info("Fighter scrape already complete: %d/%d fighters present", len(existing_urls), len(fighter_urls))
        return pd.DataFrame(all_fighters)

    completed = len(existing_urls)
    for i, url in enumerate(remaining_urls, start=completed + 1):
        logger.info(f"Scraping fighter {i}/{len(fighter_urls)}")
        try:
            fighter = scrape_fighter(url)
            if fighter:
                all_fighters.append(fighter)
        except Exception as e:
            logger.warning(f"Failed to scrape fighter {url}: {e}")

        if i % 100 == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(all_fighters).drop_duplicates(subset=["fighter_url"], keep="last").to_csv(output_path, index=False)

    df = pd.DataFrame(all_fighters).drop_duplicates(subset=["fighter_url"], keep="last")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Scraped {len(df)} fighters. Saved to {output_path}")
    return df


def scrape_fighters_from_inventory(
    output_path: Optional[Path] = None,
    fighter_details_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Scrape fighter profiles using the repo's committed fighter URL inventory."""
    if output_path is None:
        output_path = RAW_DATA_DIR / "ufc_fighters_scraped.csv"

    details_path = Path(fighter_details_path) if fighter_details_path is not None else FIGHTER_DETAILS_PATH
    fighter_urls = _load_fighter_inventory_urls(details_path)
    if not fighter_urls:
        raise FileNotFoundError(f"fighter URL inventory not found or empty: {details_path}")

    logger.info("Scraping %d fighters from inventory %s", len(fighter_urls), details_path)
    return _scrape_fighter_url_batch(fighter_urls, output_path=output_path)


# ---------------------------------------------------------------------------
# Full scrape orchestration
# ---------------------------------------------------------------------------

def scrape_all_fights(output_path: Optional[Path] = None) -> pd.DataFrame:
    """Scrape all fights from all events. Returns DataFrame and saves CSV."""
    if output_path is None:
        output_path = RAW_DATA_DIR / "ufc_fights_scraped.csv"

    event_urls = scrape_all_event_urls()
    existing_rows, existing_fight_urls = _load_existing_rows(output_path, "fight_url")
    all_fights = list(existing_rows)

    if existing_fight_urls:
        logger.info(
            "Resuming fight scrape with %d existing fights from %s",
            len(existing_fight_urls),
            output_path,
        )

    for i, event_url in enumerate(event_urls):
        logger.info(f"Scraping event {i+1}/{len(event_urls)}: {event_url}")
        try:
            event = scrape_event(event_url)
            for fight_url in event["fight_urls"]:
                fight_url = str(fight_url).strip()
                if not fight_url or fight_url in existing_fight_urls:
                    continue
                try:
                    fight = scrape_fight(fight_url)
                    if fight:
                        fight["event_title"] = event["title"]
                        fight["event_date"] = event["date"]
                        all_fights.append(fight)
                        existing_fight_urls.add(fight_url)
                except Exception as e:
                    logger.warning(f"Failed to scrape fight {fight_url}: {e}")
        except Exception as e:
            logger.warning(f"Failed to scrape event {event_url}: {e}")

        # Save progress every 50 events
        if (i + 1) % 50 == 0:
            pd.DataFrame(all_fights).drop_duplicates(subset=["fight_url"], keep="last").to_csv(output_path, index=False)
            logger.info(f"Progress saved: {len(all_fights)} fights")

    df = pd.DataFrame(all_fights).drop_duplicates(subset=["fight_url"], keep="last")
    df.to_csv(output_path, index=False)
    logger.info(f"Scraped {len(df)} fights total. Saved to {output_path}")
    return df


def scrape_all_fighters(
    output_path: Optional[Path] = None,
    fighter_details_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Scrape all fighter profiles, preferring the repo inventory when available."""
    if output_path is None:
        output_path = RAW_DATA_DIR / "ufc_fighters_scraped.csv"

    details_path = Path(fighter_details_path) if fighter_details_path is not None else FIGHTER_DETAILS_PATH
    fighter_urls = _load_fighter_inventory_urls(details_path)
    if fighter_urls:
        logger.info("Using fighter URL inventory at %s", details_path)
        return _scrape_fighter_url_batch(fighter_urls, output_path=output_path)

    logger.info(
        "Fighter URL inventory unavailable at %s; falling back to live UFCStats directory crawl",
        details_path,
    )
    fighter_urls = scrape_all_fighter_urls()
    return _scrape_fighter_url_batch(fighter_urls, output_path=output_path)
