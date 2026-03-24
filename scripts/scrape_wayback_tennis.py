"""Targeted Wayback recovery for unresolved tennis odds buckets.

This script only attempts buckets that are still classified as source_absent in
data/processed/tennis/unmatched_diagnostics.csv. It keeps archive provenance
separate from live scrapes and stops after the first snapshot that yields
numeric odds for a bucket.
"""

from __future__ import annotations

import re
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIAGNOSTICS_PATH = ROOT / "data" / "processed" / "tennis" / "unmatched_diagnostics.csv"
OUTPUT_PATH = ROOT / "data" / "raw" / "tennis" / "wayback_tennis_odds.csv"

REQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    )
}
CDX_DELAY_SECONDS = 1.0
SNAPSHOT_DELAY_SECONDS = 3.5
MAX_SNAPSHOTS = 5

from scripts.scrape_oddsportal_manual_pages import extract_rows_from_body
from scripts.scrape_oddsportal_wta125 import TARGETS as WTA125_TARGETS


def bucket_base_name(bucket):
    return re.sub(r"\s+\d{4}$", "", str(bucket or "")).strip()


def build_target_specs():
    specs = {}
    for target in WTA125_TARGETS:
        specs[target["tournament"]] = [{
            "url": target["url"],
            "tournament": target["tournament"],
            "tour": "wta",
            "event_type": "125",
            "source": "wayback_oddsportal",
            "parser": "oddsportal",
        }]

    specs.update({
        "Miami": [
            {
                "url": "https://www.oddsportal.com/tennis/usa/wta-miami/results/",
                "tournament": "Miami",
                "tour": "wta",
                "event_type": "1000",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
            {
                "url": "https://www.betexplorer.com/tennis/wta-singles/miami/results/?season=2026",
                "tournament": "wta-miami-2026",
                "tour": "wta",
                "event_type": "1000",
                "source": "wayback_betexplorer",
                "parser": "betexplorer",
            },
        ],
        "Next Gen ATP Finals": [
            {
                "url": "https://www.oddsportal.com/tennis/world/atp-next-gen-finals-jeddah/results/",
                "tournament": "Next Gen ATP Finals",
                "tour": "atp",
                "event_type": "nextgen",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
            {
                "url": "https://www.betexplorer.com/tennis/atp-singles/next-gen-finals-jeddah/results/?season=2025",
                "tournament": "atp-nextgen-jeddah-2025",
                "tour": "atp",
                "event_type": "nextgen",
                "source": "wayback_betexplorer",
                "parser": "betexplorer",
            },
            {
                "url": "https://www.oddsportal.com/tennis/world/atp-next-gen-finals-{year}/results/",
                "tournament": "Next Gen ATP Finals",
                "tour": "atp",
                "event_type": "nextgen",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
            {
                "url": "https://www.betexplorer.com/tennis/atp-singles/next-gen-finals/results/?season={year}",
                "tournament": "atp-nextgen-{year}",
                "tour": "atp",
                "event_type": "nextgen",
                "source": "wayback_betexplorer",
                "parser": "betexplorer",
            },
        ],
        "Atp Cup": [
            {
                "url": "https://www.oddsportal.com/tennis/australia/atp-atp-cup/results/",
                "tournament": "ATP Cup",
                "tour": "atp",
                "event_type": "atp_cup",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
            {
                "url": "https://www.betexplorer.com/tennis/atp-singles/atp-cup/results/?season={year}",
                "tournament": "atp-cup-{year}",
                "tour": "atp",
                "event_type": "atp_cup",
                "source": "wayback_betexplorer",
                "parser": "betexplorer",
            },
        ],
        "Davis Cup": [
            {
                "url": "https://www.oddsportal.com/tennis/world/atp-davis-cup-{year}/results/",
                "tournament": "Davis Cup",
                "tour": "atp",
                "event_type": "davis_cup",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
            {
                "url": "https://www.oddsportal.com/tennis/world/atp-davis-cup-world-group-{year}/results/",
                "tournament": "Davis Cup",
                "tour": "atp",
                "event_type": "davis_cup",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
            {
                "url": "https://www.oddsportal.com/tennis/world/atp-davis-cup-world-group-ii-{year}/results/",
                "tournament": "Davis Cup",
                "tour": "atp",
                "event_type": "davis_cup",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
            {
                "url": "https://www.oddsportal.com/tennis/world/atp-davis-cup-group-i-{year}/results/",
                "tournament": "Davis Cup",
                "tour": "atp",
                "event_type": "davis_cup",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
            {
                "url": "https://www.oddsportal.com/tennis/world/atp-davis-cup-group-ii-{year}/results/",
                "tournament": "Davis Cup",
                "tour": "atp",
                "event_type": "davis_cup",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
            {
                "url": "https://www.betexplorer.com/tennis/atp-singles/davis-cup-world-group/results/?season={year}",
                "tournament": "davis-cup-world-group-{year}",
                "tour": "atp",
                "event_type": "davis_cup",
                "source": "wayback_betexplorer",
                "parser": "betexplorer",
            },
            {
                "url": "https://www.betexplorer.com/tennis/atp-singles/davis-cup-world-group-ii/results/?season={year}",
                "tournament": "davis-cup-world-group-ii-{year}",
                "tour": "atp",
                "event_type": "davis_cup",
                "source": "wayback_betexplorer",
                "parser": "betexplorer",
            },
            {
                "url": "https://www.betexplorer.com/tennis/atp-singles/davis-cup-group-i/results/?season={year}",
                "tournament": "davis-cup-group-i-{year}",
                "tour": "atp",
                "event_type": "davis_cup",
                "source": "wayback_betexplorer",
                "parser": "betexplorer",
            },
            {
                "url": "https://www.betexplorer.com/tennis/atp-singles/davis-cup-group-ii/results/?season={year}",
                "tournament": "davis-cup-group-ii-{year}",
                "tour": "atp",
                "event_type": "davis_cup",
                "source": "wayback_betexplorer",
                "parser": "betexplorer",
            },
        ],
        "Metz": [
            {
                "url": "https://www.oddsportal.com/tennis/france/atp-metz-{year}/results/",
                "tournament": "Metz",
                "tour": "atp",
                "event_type": "main_tour",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
            {
                "url": "https://www.oddsportal.com/tennis/france/atp-metz/results/",
                "tournament": "Metz",
                "tour": "atp",
                "event_type": "main_tour",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
            {
                "url": "https://www.betexplorer.com/tennis/atp-singles/metz/results/?season={year}",
                "tournament": "metz-{year}",
                "tour": "atp",
                "event_type": "main_tour",
                "source": "wayback_betexplorer",
                "parser": "betexplorer",
            },
        ],
        "Nitto ATP Finals": [
            {
                "url": "https://www.oddsportal.com/tennis/italy/atp-nitto-atp-finals-{year}/results/",
                "tournament": "ATP Finals",
                "tour": "atp",
                "event_type": "main_tour",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
            {
                "url": "https://www.oddsportal.com/tennis/italy/atp-tour-finals-{year}/results/",
                "tournament": "ATP Finals",
                "tour": "atp",
                "event_type": "main_tour",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
            {
                "url": "https://www.betexplorer.com/tennis/atp-singles/nitto-atp-finals/results/?season={year}",
                "tournament": "atp-finals-{year}",
                "tour": "atp",
                "event_type": "main_tour",
                "source": "wayback_betexplorer",
                "parser": "betexplorer",
            },
        ],
        "Bad Homburg": [
            {
                "url": "https://www.oddsportal.com/tennis/germany/wta-bad-homburg/results/",
                "tournament": "Bad Homburg",
                "tour": "wta",
                "event_type": "main_tour",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
            {
                "url": "https://www.betexplorer.com/tennis/wta-singles/bad-homburg/results/?season={year}",
                "tournament": "bad-homburg-{year}",
                "tour": "wta",
                "event_type": "main_tour",
                "source": "wayback_betexplorer",
                "parser": "betexplorer",
            },
        ],
        "Saint Malo 125": [
            {
                "url": "https://www.oddsportal.com/tennis/france/challenger-women-singles-saint-malo/results/",
                "tournament": "Saint Malo 125",
                "tour": "wta",
                "event_type": "125",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
        ],
        "Ljubljana 125": [
            {
                "url": "https://www.oddsportal.com/tennis/slovenia/challenger-women-singles-ljubljana/results/",
                "tournament": "Ljubljana 125",
                "tour": "wta",
                "event_type": "125",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
        ],
        "Quito 125": [
            {
                "url": "https://www.oddsportal.com/tennis/ecuador/challenger-women-singles-quito/results/",
                "tournament": "Quito 125",
                "tour": "wta",
                "event_type": "125",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
        ],
        "Hamburg": [
            {
                "url": "https://www.oddsportal.com/tennis/germany/atp-hamburg/results/",
                "tournament": "Hamburg",
                "tour": "atp",
                "event_type": "main_tour",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
            {
                "url": "https://www.betexplorer.com/tennis/atp-singles/hamburg/results/?season={year}",
                "tournament": "hamburg-{year}",
                "tour": "atp",
                "event_type": "main_tour",
                "source": "wayback_betexplorer",
                "parser": "betexplorer",
            },
        ],
        "Lyon": [
            {
                "url": "https://www.oddsportal.com/tennis/france/atp-lyon/results/",
                "tournament": "Lyon",
                "tour": "atp",
                "event_type": "main_tour",
                "source": "wayback_oddsportal",
                "parser": "oddsportal",
            },
            {
                "url": "https://www.betexplorer.com/tennis/atp-singles/lyon/results/?season={year}",
                "tournament": "lyon-{year}",
                "tour": "atp",
                "event_type": "main_tour",
                "source": "wayback_betexplorer",
                "parser": "betexplorer",
            },
        ],
    })
    return specs


def fetch_cdx_snapshots(session, url, start_date, end_date):
    params = {
        "url": url,
        "output": "json",
        "fl": "timestamp,original,statuscode,digest",
        "filter": "statuscode:200",
        "collapse": "digest",
        "limit": "50",
        "from": (start_date - timedelta(days=10)).strftime("%Y%m%d000000"),
        "to": (end_date + timedelta(days=45)).strftime("%Y%m%d235959"),
    }
    response = session.get(
        "https://web.archive.org/cdx/search/cdx",
        params=params,
        headers=REQ_HEADERS,
        timeout=60,
    )
    time.sleep(CDX_DELAY_SECONDS)
    response.raise_for_status()

    payload = response.json()
    if len(payload) <= 1:
        return []

    header, *rows = payload
    frame = pd.DataFrame(rows, columns=header)
    frame["captured_at"] = pd.to_datetime(frame["timestamp"], format="%Y%m%d%H%M%S", errors="coerce")
    frame = frame.dropna(subset=["captured_at"]).copy()
    if frame.empty:
        return []

    midpoint = start_date + (end_date - start_date) / 2
    frame["distance_seconds"] = (frame["captured_at"] - midpoint).abs().dt.total_seconds()
    frame = frame.sort_values(["distance_seconds", "captured_at"]).head(MAX_SNAPSHOTS)
    return frame.to_dict("records")


def parse_float(text):
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def parse_betexplorer_html(html, tournament, tour, event_type):
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if table is None:
        return []

    rows = []
    current_round = ""
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue

        first_text = cells[0].get_text(" ", strip=True)
        colspan = cells[0].get("colspan")
        if colspan and first_text:
            current_round = first_text
            continue

        if len(cells) < 4:
            continue

        players_text = cells[0].get_text(" ", strip=True)
        match = re.match(r"^(.+?)\s*-\s*(.+)$", players_text)
        if not match:
            continue
        player1 = match.group(1).strip()
        player2 = match.group(2).strip()
        if len(player1) < 3 or len(player2) < 3:
            continue

        score_text = cells[1].get_text(" ", strip=True)
        score_match = re.match(r"(\d+):(\d+)", score_text)
        if not score_match:
            continue
        sets1 = int(score_match.group(1))
        sets2 = int(score_match.group(2))
        if sets1 == sets2:
            continue

        odds_cells = [
            cell for cell in cells
            if "table-main__odds" in " ".join(cell.get("class", []))
        ]
        odds_p1 = parse_float(odds_cells[0].get_text(" ", strip=True)) if len(odds_cells) >= 1 else None
        odds_p2 = parse_float(odds_cells[1].get_text(" ", strip=True)) if len(odds_cells) >= 2 else None
        if odds_p1 is None or odds_p2 is None:
            continue

        date_text = cells[-1].get_text(" ", strip=True)
        date_match = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", date_text)
        if not date_match:
            continue
        row_date = f"{date_match.group(3)}-{date_match.group(2)}-{date_match.group(1)}"

        if sets1 > sets2:
            winner, loser, odds_w, odds_l = player1, player2, odds_p1, odds_p2
        else:
            winner, loser, odds_w, odds_l = player2, player1, odds_p2, odds_p1

        rows.append({
            "Date": row_date,
            "Winner": winner,
            "Loser": loser,
            "odds_w": odds_w,
            "odds_l": odds_l,
            "round": current_round,
            "tournament": tournament,
            "tour": tour,
            "event_type": event_type,
        })

    return rows


def parse_snapshot_rows(html, spec):
    if spec["parser"] == "betexplorer":
        return parse_betexplorer_html(
            html,
            spec["tournament"],
            spec["tour"],
            spec["event_type"],
        )

    body = BeautifulSoup(html, "lxml").get_text("\n")
    return extract_rows_from_body(
        body,
        spec["tournament"],
        spec["tour"],
        spec["event_type"],
    )


def filter_rows_to_window(rows, start_date, end_date):
    if not rows:
        return []

    frame = pd.DataFrame(rows)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).copy()
    frame = frame[
        (frame["Date"] >= start_date - timedelta(days=3)) &
        (frame["Date"] <= end_date + timedelta(days=3))
    ].copy()
    if frame.empty:
        return []
    frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")
    return frame.to_dict("records")


def merge_output(rows):
    if not rows:
        return 0

    new_df = pd.DataFrame(rows)
    if OUTPUT_PATH.exists():
        old_df = pd.read_csv(OUTPUT_PATH, low_memory=False)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(
        subset=["Date", "Winner", "Loser", "tournament", "source"],
        keep="last",
    ).sort_values(["Date", "tournament", "Winner", "Loser"]).reset_index(drop=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    return len(new_df)


def main():
    if not DIAGNOSTICS_PATH.exists():
        print(f"Diagnostics file not found: {DIAGNOSTICS_PATH}")
        return

    diagnostics_df = pd.read_csv(DIAGNOSTICS_PATH)
    unresolved = diagnostics_df[diagnostics_df["classification"] == "source_absent"].copy()
    if unresolved.empty:
        print("No source_absent buckets remain.")
        return

    unresolved["event_date"] = pd.to_datetime(unresolved["event_date"], errors="coerce")
    target_specs = build_target_specs()
    recovered_rows = []

    session = requests.Session()
    session.headers.update(REQ_HEADERS)

    for bucket, group in unresolved.groupby("bucket", sort=True):
        base_name = bucket_base_name(bucket)
        specs = target_specs.get(base_name, [])
        if not specs and base_name.startswith("Davis Cup"):
            specs = target_specs.get("Davis Cup", [])
        if not specs:
            continue

        start_date = group["event_date"].min()
        end_date = group["event_date"].max()
        if pd.isna(start_date) or pd.isna(end_date):
            continue

        print("=" * 72)
        print(f"{bucket} ({len(group)} unresolved rows)")
        print("=" * 72)

        bucket_recovered = False
        for spec in specs:
            spec_url = spec["url"].format(year=start_date.year)
            try:
                snapshots = fetch_cdx_snapshots(session, spec_url, start_date, end_date)
            except Exception as exc:
                print(f"  CDX error for {spec_url}: {exc}")
                continue

            print(f"  {spec['source']} {spec_url} -> {len(snapshots)} snapshots")
            if not snapshots:
                continue

            for snapshot in snapshots:
                archive_url = f"https://web.archive.org/web/{snapshot['timestamp']}if_/{spec_url}"
                try:
                    response = session.get(archive_url, timeout=60)
                    time.sleep(SNAPSHOT_DELAY_SECONDS)
                    response.raise_for_status()
                except Exception as exc:
                    print(f"    snapshot {snapshot['timestamp']} failed: {exc}")
                    continue

                rows = parse_snapshot_rows(response.text, spec)
                rows = filter_rows_to_window(rows, start_date, end_date)
                if not rows:
                    print(f"    snapshot {snapshot['timestamp']} -> no in-window numeric rows")
                    continue

                for row in rows:
                    row["source"] = spec["source"]
                    row["source_url"] = spec_url
                    row["archive_url"] = archive_url
                    row["captured_at"] = snapshot["captured_at"].isoformat()

                recovered_rows.extend(rows)
                print(f"    snapshot {snapshot['timestamp']} -> recovered {len(rows)} rows")
                bucket_recovered = True
                break

            if bucket_recovered:
                break

        if not bucket_recovered:
            print("  archive-empty")

    appended = merge_output(recovered_rows)
    print()
    print(f"Recovered rows this run: {len(recovered_rows)}")
    print(f"Appended rows this run: {appended}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
