"""Scrape BetExplorer.com for specific missing tournament odds using Selenium.

Targets found:
- WTA Guadalajara 2 (Feb 2022, WTA 250): /tennis/wta-singles/guadalajara-2/results/
- WTA Guadalajara 2022 (Oct 2022, WTA 1000): /tennis/wta-singles/guadalajara-2022/results/
- ATP Metz 2025: /tennis/atp-singles/metz/results/
- ATP Nitto Finals Turin 2025:
    Group stage: /tennis/atp-singles/atp-finals-turin/results/?stage=fwudgPd3
    Playoffs: /tennis/atp-singles/atp-finals-turin/results/?stage=GhT4i3RF
- Olympics 2024: NOT available on BetExplorer (all URLs redirect to homepage)
"""

import re
import sys
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://www.betexplorer.com"


def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(30)
    return driver


def parse_results_page(driver, url, tournament_name, tour, event_type):
    """Parse a BetExplorer results page for match rows with odds."""
    rows = []
    try:
        print(f"    Fetching {url} ...", flush=True)
        driver.get(url)
        time.sleep(3)

        # Check for redirect to homepage
        if driver.current_url in ("https://www.betexplorer.com/", "https://www.betexplorer.com"):
            print(f"    -> Redirected to homepage, skipping")
            return rows

        tables = driver.find_elements(By.CSS_SELECTOR, "table")
        if not tables:
            print(f"    -> No table found")
            return rows

        table = tables[0]
        tr_elements = table.find_elements(By.TAG_NAME, "tr")
        current_round = ""

        for tr in tr_elements:
            tds = tr.find_elements(By.TAG_NAME, "td")
            ths = tr.find_elements(By.TAG_NAME, "th")
            cells = ths + tds

            if not cells:
                continue

            first_text = cells[0].text.strip()

            # Round header row
            colspan = cells[0].get_attribute("colspan")
            if colspan and first_text:
                current_round = first_text
                continue

            if len(cells) < 4:
                continue

            players_text = cells[0].text.strip()
            if not players_text or len(players_text) < 5:
                continue

            # Split player names
            m = re.match(r'^(.+?\.)\s*-\s*(.+)$', players_text)
            if not m:
                m = re.match(r'^(.+?)\s+-\s+(.+)$', players_text)
            if not m:
                continue

            player1 = m.group(1).strip()
            player2 = m.group(2).strip()

            if not player1 or not player2 or len(player1) < 3 or len(player2) < 3:
                continue

            # Score
            score_text = cells[1].text.strip() if len(cells) > 1 else ""
            score_match = re.match(r"(\d+):(\d+)", score_text)
            if not score_match:
                continue
            sets1, sets2 = int(score_match.group(1)), int(score_match.group(2))
            winner = player1 if sets1 > sets2 else player2
            loser = player2 if sets1 > sets2 else player1

            # Get odds from rendered cells
            odds_cells = [
                c for c in cells
                if "table-main__odds" in (c.get_attribute("class") or "")
            ]
            odds_p1 = None
            odds_p2 = None

            if len(odds_cells) >= 2:
                try:
                    t1 = odds_cells[0].text.strip()
                    if t1:
                        odds_p1 = float(t1)
                except (ValueError, IndexError):
                    pass
                try:
                    t2 = odds_cells[1].text.strip()
                    if t2:
                        odds_p2 = float(t2)
                except (ValueError, IndexError):
                    pass

            # Also try data-odd attribute
            if odds_p1 is None or odds_p2 is None:
                odd_spans = []
                for c in cells:
                    for s in c.find_elements(By.CSS_SELECTOR, "[data-odd]"):
                        val = s.get_attribute("data-odd")
                        if val:
                            odd_spans.append(val)
                if len(odd_spans) >= 2:
                    try:
                        if odds_p1 is None:
                            odds_p1 = float(odd_spans[0])
                        if odds_p2 is None:
                            odds_p2 = float(odd_spans[1])
                    except ValueError:
                        pass

            # Date (last cell)
            date_text = cells[-1].text.strip()
            date_match = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", date_text)
            date_str = (
                f"{date_match.group(3)}-{date_match.group(2)}-{date_match.group(1)}"
                if date_match
                else ""
            )

            if not date_str:
                continue

            is_p1_winner = winner == player1
            odds_w = odds_p1 if is_p1_winner else odds_p2
            odds_l = odds_p2 if is_p1_winner else odds_p1

            rows.append({
                "Date": date_str,
                "Winner": winner,
                "Loser": loser,
                "score": score_text,
                "round": current_round,
                "odds_w": odds_w,
                "odds_l": odds_l,
                "tournament": tournament_name,
                "tour": tour,
                "event_type": event_type,
                "source": "betexplorer",
            })

    except Exception as e:
        print(f"    Error: {e}")

    return rows


def wants_section(filters, *labels):
    if not filters:
        return True

    haystack = " ".join(str(label).lower() for label in labels)
    return any(filter_text in haystack for filter_text in filters)


def main():
    filters = [arg.strip().lower() for arg in sys.argv[1:] if arg.strip()]
    driver = get_driver()
    all_data = []

    try:
        # ===== 1. WTA Guadalajara 2 (Feb 2022, WTA 250) =====
        if wants_section(filters, "guadalajara", "guadalajara 2"):
            print("=" * 60)
            print("1. WTA Guadalajara 2 (Feb 2022)")
            print("=" * 60)
            rows = parse_results_page(
                driver,
                f"{BASE}/tennis/wta-singles/guadalajara-2/results/",
                "wta-guadalajara-2-2022", "wta", "250",
            )
            n_odds = sum(1 for r in rows if r["odds_w"] is not None)
            print(f"    -> {len(rows)} matches ({n_odds} with odds)")
            all_data.extend(rows)

            rows_q = parse_results_page(
                driver,
                f"{BASE}/tennis/wta-singles/guadalajara-2/results/?stage=UczH3VkR",
                "wta-guadalajara-2-2022-qual", "wta", "250_qual",
            )
            n_odds = sum(1 for r in rows_q if r["odds_w"] is not None)
            print(f"    -> {len(rows_q)} qual matches ({n_odds} with odds)")
            all_data.extend(rows_q)

        # ===== 2. WTA Guadalajara 2022 (Oct 2022, WTA 1000) =====
        if wants_section(filters, "guadalajara", "guadalajara 2022"):
            print("\n" + "=" * 60)
            print("2. WTA Guadalajara 2022 (Oct 2022, WTA 1000)")
            print("=" * 60)
            rows = parse_results_page(
                driver,
                f"{BASE}/tennis/wta-singles/guadalajara-2022/results/",
                "wta-guadalajara-1000-2022", "wta", "1000",
            )
            n_odds = sum(1 for r in rows if r["odds_w"] is not None)
            print(f"    -> {len(rows)} matches ({n_odds} with odds)")
            all_data.extend(rows)

            rows_q = parse_results_page(
                driver,
                f"{BASE}/tennis/wta-singles/guadalajara-2022/results/?stage=foiBO6hQ",
                "wta-guadalajara-1000-2022-qual", "wta", "1000_qual",
            )
            n_odds = sum(1 for r in rows_q if r["odds_w"] is not None)
            print(f"    -> {len(rows_q)} qual matches ({n_odds} with odds)")
            all_data.extend(rows_q)

        # ===== 3. ATP Metz 2025 =====
        if wants_section(filters, "metz"):
            print("\n" + "=" * 60)
            print("3. ATP Metz 2025")
            print("=" * 60)
            rows = parse_results_page(
                driver,
                f"{BASE}/tennis/atp-singles/metz/results/",
                "atp-metz-2025", "atp", "250",
            )
            n_odds = sum(1 for r in rows if r["odds_w"] is not None)
            print(f"    -> {len(rows)} matches ({n_odds} with odds)")
            all_data.extend(rows)

        # ===== 4. ATP Nitto Finals Turin 2025 =====
        if wants_section(filters, "finals", "nitto", "turin"):
            print("\n" + "=" * 60)
            print("4. ATP Nitto Finals Turin 2025")
            print("=" * 60)

            print("  Group stage:")
            rows_group = parse_results_page(
                driver,
                f"{BASE}/tennis/atp-singles/atp-finals-turin/results/?stage=fwudgPd3",
                "atp-nitto-finals-2025", "atp", "finals",
            )
            n_odds = sum(1 for r in rows_group if r["odds_w"] is not None)
            print(f"    -> {len(rows_group)} matches ({n_odds} with odds)")
            all_data.extend(rows_group)

            print("  Playoffs:")
            rows_playoffs = parse_results_page(
                driver,
                f"{BASE}/tennis/atp-singles/atp-finals-turin/results/?stage=GhT4i3RF",
                "atp-nitto-finals-2025", "atp", "finals",
            )
            n_odds = sum(1 for r in rows_playoffs if r["odds_w"] is not None)
            print(f"    -> {len(rows_playoffs)} matches ({n_odds} with odds)")
            all_data.extend(rows_playoffs)

        if wants_section(filters, "nextgen", "miami"):
            print("\n" + "=" * 60)
            print("5. Next Gen Finals 2025 + Miami 2026")
            print("=" * 60)

            for label, url, tournament, tour, event_type in [
                (
                    "Next Gen Finals Jeddah 2025",
                    f"{BASE}/tennis/atp-singles/next-gen-finals-jeddah/results/?season=2025",
                    "atp-nextgen-jeddah-2025",
                    "atp",
                    "nextgen",
                ),
                (
                    "ATP Miami 2026",
                    f"{BASE}/tennis/atp-singles/miami/results/?season=2026",
                    "atp-miami-2026",
                    "atp",
                    "1000",
                ),
                (
                    "WTA Miami 2026",
                    f"{BASE}/tennis/wta-singles/miami/results/?season=2026",
                    "wta-miami-2026",
                    "wta",
                    "1000",
                ),
            ]:
                print(f"  {label}")
                rows = parse_results_page(driver, url, tournament, tour, event_type)
                n_odds = sum(1 for r in rows if r["odds_w"] is not None)
                print(f"    -> {len(rows)} matches ({n_odds} with odds)")
                all_data.extend(rows)

        # ===== 5. Olympics 2024 (attempt) =====
        if wants_section(filters, "olympics"):
            print("\n" + "=" * 60)
            print("6. Olympics Tennis 2024 (attempting)")
            print("=" * 60)

            olympics_found = False
            for label, path in [
                ("ATP Olympic Games", "/tennis/world/atp-olympic-games/results/"),
                ("WTA Olympic Games", "/tennis/world/wta-olympic-games/results/"),
                ("Olympic Games Men", "/tennis/france/olympic-games-men-singles/results/"),
                ("Olympic Games Women", "/tennis/france/olympic-games-women-singles/results/"),
                ("Olympic Games Paris Men", "/tennis/france/paris-olympic-games-men/results/"),
                ("Olympic Games Paris Women", "/tennis/france/paris-olympic-games-women/results/"),
                ("Olympics Men", "/tennis/exhibition-men/olympic-games/results/"),
                ("Olympics Women", "/tennis/exhibition-women/olympic-games/results/"),
                ("Olympics Paris 2024 Men", "/tennis/france/olympic-games-2024-men/results/"),
                ("Olympics Paris 2024 Women", "/tennis/france/olympic-games-2024-women/results/"),
            ]:
                url = f"{BASE}{path}"
                driver.get(url)
                time.sleep(2)
                if driver.current_url not in ("https://www.betexplorer.com/", "https://www.betexplorer.com"):
                    tables = driver.find_elements(By.CSS_SELECTOR, "table")
                    if tables:
                        trs = tables[0].find_elements(By.TAG_NAME, "tr")
                        has_data = False
                        for tr in trs[:3]:
                            cells = tr.find_elements(By.TAG_NAME, "td")
                            if cells and len(cells) >= 4 and cells[0].text.strip():
                                has_data = True
                                break
                        if has_data:
                            print(f"    FOUND: {label} at {path}")
                            olympics_found = True
                            tour = "atp" if "men" in label.lower() or "atp" in label.lower() else "wta"
                            rows = parse_results_page(driver, url, f"olympics-{tour}-2024", tour, "olympics")
                            all_data.extend(rows)
                            continue
                print(f"    Not found: {label}")

            if not olympics_found:
                print("    Olympics 2024 tennis data is NOT available on BetExplorer")

    finally:
        driver.quit()

    # ===== Save results =====
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if not all_data:
        print("No data scraped!")
        return

    df = pd.DataFrame(all_data)
    print(f"Total matches scraped: {len(df)}")
    print(f"\nBy tournament:")
    for name, group in df.groupby("tournament"):
        both = (group["odds_w"].notna() & group["odds_l"].notna()).sum()
        print(f"  {name}: {len(group)} matches ({both} with both odds)")
    print(f"\nDate range: {df['Date'].min()} to {df['Date'].max()}")

    both_odds = (df["odds_w"].notna() & df["odds_l"].notna()).sum()
    any_odds = (df["odds_w"].notna() | df["odds_l"].notna()).sum()
    print(f"\nTotal with both odds: {both_odds} / {len(df)}")
    print(f"Total with any odds: {any_odds} / {len(df)}")

    output_path = ROOT / "data" / "raw" / "tennis" / "betexplorer_extra_odds.csv"
    if output_path.exists():
        existing = pd.read_csv(output_path, low_memory=False)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["Date", "Winner", "Loser", "tournament"], keep="last")
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
