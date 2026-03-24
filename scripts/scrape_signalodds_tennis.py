"""Recover tennis odds from SignalOdds event pages.

SignalOdds event URLs are often predictable from player names and date. The
raw HTML exposes event metadata, while the rendered page exposes the moneyline
odds in body text.
"""

from __future__ import annotations

import re
import sys
import time
import unicodedata
from datetime import timedelta
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fix_tennis_odds_matching import names_match_flexible, normalize_text

DIAGNOSTICS_PATH = ROOT / "data" / "processed" / "tennis" / "unmatched_diagnostics.csv"
OUTPUT_PATH = ROOT / "data" / "raw" / "tennis" / "signalodds_odds.csv"

REQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    )
}
SEARCH_DELAY_SECONDS = 1.0


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1600,1200")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=options)


def load_rows(filters: list[str]) -> pd.DataFrame:
    df = pd.read_csv(DIAGNOSTICS_PATH, low_memory=False)
    df = df[df["classification"] == "source_absent"].copy()
    if filters:
        lowered = [f.lower() for f in filters]
        mask = df["bucket"].astype(str).str.lower().apply(
            lambda bucket: any(token in bucket for token in lowered)
        )
        df = df[mask].copy()
    df = df.drop_duplicates(subset=["event_date", "winner", "loser_name", "bucket"]).reset_index(drop=True)
    return df


def search_yahoo(session: requests.Session, query: str) -> list[dict]:
    url = "https://search.yahoo.com/search?p=" + requests.utils.quote(query)
    response = session.get(url, headers=REQ_HEADERS, timeout=45)
    response.raise_for_status()
    time.sleep(SEARCH_DELAY_SECONDS)

    soup = BeautifulSoup(response.text, "lxml")
    results = []
    for block in soup.select("div.algo"):
        link = block.select_one(".compTitle a[href]")
        if not link:
            continue

        raw_url = link.get("href", "")
        yahoo_match = re.search(r"/RU=([^/]+)/RK=", raw_url)
        resolved_url = unquote(yahoo_match.group(1)) if yahoo_match else raw_url
        snippet = block.select_one(".compText p")
        results.append({
            "title": block.select_one("h3").get_text(" ", strip=True) if block.select_one("h3") else "",
            "url": resolved_url,
            "snippet": snippet.get_text(" ", strip=True) if snippet else "",
        })
    return results


def direct_candidate_urls(row: pd.Series) -> list[str]:
    winner = slugify(row["winner"])
    loser = slugify(row["loser_name"])
    base_date = pd.to_datetime(row["event_date"])
    urls = []
    for day_offset in range(-3, 4):
        date_token = (base_date + timedelta(days=day_offset)).strftime("%Y%m%d")
        urls.append(f"https://www.signalodds.com/events/{winner}-vs-{loser}-{date_token}")
        urls.append(f"https://www.signalodds.com/events/{loser}-vs-{winner}-{date_token}")
    return urls


def search_candidate_urls(session: requests.Session, row: pd.Series) -> list[str]:
    year = pd.to_datetime(row["event_date"]).year
    queries = [
        f'site:signalodds.com/events "{row["winner"]}" "{row["loser_name"]}" "{row["tourney_name"]}" {year} tennis odds',
        f'site:signalodds.com/events "{row["winner"]}" "{row["loser_name"]}" "{row["bucket"]}" {year}',
        f'{row["winner"]} {row["loser_name"]} signalodds {row["tourney_name"]} {year}',
    ]
    urls = []
    seen = set()
    for query in queries:
        try:
            results = search_yahoo(session, query)
        except Exception:
            continue
        for result in results:
            url = result.get("url", "")
            if "signalodds.com/events/" not in url:
                continue
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)
    return urls


def validate_signalodds_html(html: str, row: pd.Series) -> bool:
    winner = normalize_text(row["winner"])
    loser = normalize_text(row["loser_name"])
    tourney = normalize_text(row["tourney_name"])
    html_norm = normalize_text(html)
    return (
        winner in html_norm
        and loser in html_norm
        and ("miami" in html_norm or tourney in html_norm)
    )


def parse_moneyline_text(body_text: str, row: pd.Series) -> tuple[float | None, float | None]:
    lines = [line.strip() for line in body_text.splitlines() if line.strip()]
    try:
        start_idx = lines.index("Moneyline")
    except ValueError:
        return None, None

    stop_markers = {
        "Handicap",
        "Over/Under",
        "Show 3 more markets",
        "Show 5 more markets",
        "Frequently Asked Questions",
    }
    section = []
    for line in lines[start_idx + 1:]:
        if line in stop_markers:
            break
        section.append(line)

    winner = row["winner"]
    loser = row["loser_name"]
    winner_odds = None
    loser_odds = None

    for idx, line in enumerate(section):
        if winner_odds is None and names_match_flexible(line, winner):
            for next_line in section[idx + 1: idx + 8]:
                if re.fullmatch(r"\d+(?:\.\d+)?", next_line):
                    winner_odds = float(next_line)
                    break
        if loser_odds is None and names_match_flexible(line, loser):
            for next_line in section[idx + 1: idx + 8]:
                if re.fullmatch(r"\d+(?:\.\d+)?", next_line):
                    loser_odds = float(next_line)
                    break

    return winner_odds, loser_odds


def merge_rows(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    new_df = pd.DataFrame(rows)
    old_df = pd.read_csv(OUTPUT_PATH, low_memory=False) if OUTPUT_PATH.exists() else pd.DataFrame()
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "Winner", "Loser", "source_url"], keep="last")
    combined.to_csv(OUTPUT_PATH, index=False)
    return new_df, combined


def main():
    filters = [arg.strip() for arg in sys.argv[1:] if arg.strip()]
    rows = load_rows(filters)
    if rows.empty:
        print("No unresolved rows selected.")
        return

    session = requests.Session()
    driver = get_driver()
    recovered = []

    try:
        for _, row in rows.iterrows():
            matched_url = None
            candidate_urls = direct_candidate_urls(row)
            for url in candidate_urls:
                try:
                    html = session.get(url, headers=REQ_HEADERS, timeout=30).text
                except Exception:
                    continue
                if validate_signalodds_html(html, row):
                    matched_url = url
                    break

            if not matched_url:
                for url in search_candidate_urls(session, row):
                    try:
                        html = session.get(url, headers=REQ_HEADERS, timeout=30).text
                    except Exception:
                        continue
                    if validate_signalodds_html(html, row):
                        matched_url = url
                        break

            if not matched_url:
                continue

            try:
                driver.get(matched_url)
                time.sleep(10)
                body_text = driver.find_element(By.TAG_NAME, "body").text
            except Exception:
                continue

            if "Betting Odds" not in body_text or "Moneyline" not in body_text:
                continue

            odds_w, odds_l = parse_moneyline_text(body_text, row)
            if odds_w is None or odds_l is None:
                continue

            recovered.append({
                "Date": row["event_date"],
                "Winner": row["winner"],
                "Loser": row["loser_name"],
                "odds_w": odds_w,
                "odds_l": odds_l,
                "tournament": row["tourney_name"],
                "tour": row["tour"],
                "event_type": "signalodds",
                "source": "signalodds",
                "source_url": matched_url,
                "bucket": row["bucket"],
            })
            print(f"Recovered {row['winner']} vs {row['loser_name']} from {matched_url}")
    finally:
        driver.quit()

    if not recovered:
        print("No rows recovered.")
        return

    new_df, combined_df = merge_rows(recovered)
    print(f"Recovered rows this run: {len(new_df)}")
    print(f"Combined SignalOdds rows: {len(combined_df)}")
    print(new_df["bucket"].value_counts().to_string())


if __name__ == "__main__":
    main()
