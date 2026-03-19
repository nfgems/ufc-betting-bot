"""
Explore BFO fighter profile pages to check if they contain method-of-victory
odds history (not just moneyline).

Fighter profile pages like /fighters/Ilia-Topuria-8322 show fight history.
This script loads those pages with Playwright and reports the page structure,
looking specifically for method props (KO/TKO, submission, decision odds).

Target fights with missing method odds:
  - Ilia Topuria vs Bryce Mitchell, Dec 2022
  - Dustin Poirier vs Michael Chandler, Nov 2022
  - Alexander Volkanovski vs Yair Rodriguez, Jul 2023
  - Alexa Grasso vs Shevchenko, Sep 2023
  - Israel Adesanya vs Pereira, Apr 2023
  - Zhang Weili vs Esparza, Nov 2022
"""
import os
import re
import sys
import time
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE_DELAY = 3.0

# Known profile URLs (or search queries for unknown ones)
FIGHTERS = [
    {
        "name": "Ilia Topuria",
        "url": "/fighters/Ilia-Topuria-8322",
        "target_fight": "vs Bryce Mitchell, Dec 2022",
    },
    {
        "name": "Dustin Poirier",
        "url": "/fighters/Dustin-Poirier-2034",
        "target_fight": "vs Michael Chandler, Nov 2022",
    },
    {
        "name": "Alexander Volkanovski",
        "url": "/fighters/Alex-Volkanovski-6723",
        "target_fight": "vs Yair Rodriguez, Jul 2023",
    },
    {
        "name": "Alexa Grasso",
        "url": None,  # need to search
        "target_fight": "vs Shevchenko, Sep 2023",
    },
    {
        "name": "Israel Adesanya",
        "url": None,  # need to search
        "target_fight": "vs Pereira, Apr 2023",
    },
    {
        "name": "Zhang Weili",
        "url": None,  # need to search
        "target_fight": "vs Esparza, Nov 2022",
    },
]

BASE_URL = "https://www.bestfightodds.com"


def search_fighter_url(page, name):
    """Search BFO for a fighter profile URL."""
    query = name.replace(" ", "+")
    search_url = f"{BASE_URL}/search?query={query}"
    print(f"  Searching: {search_url}")
    try:
        page.goto(search_url, wait_until="domcontentloaded", timeout=20000)
        time.sleep(2)

        # Look for fighter profile links
        links = page.query_selector_all("a[href*='/fighters/']")
        for link in links:
            href = link.get_attribute("href") or ""
            text = link.inner_text().strip()
            if href and "/fighters/" in href and len(text) > 2:
                print(f"    Found fighter link: {text} -> {href}")
                # Return first reasonable match
                name_parts = name.lower().split()
                if any(p in href.lower() for p in name_parts):
                    return href if href.startswith("/") else "/" + href

        # Also check the page content for any fighter links
        all_links = page.query_selector_all("a")
        for link in all_links:
            href = link.get_attribute("href") or ""
            if "/fighters/" in href:
                text = link.inner_text().strip()
                print(f"    Other fighter link: {text} -> {href}")

    except Exception as e:
        print(f"  Search error: {e}")
    return None


def explore_fighter_profile(page, fighter_info):
    """Load a fighter profile page and report its structure."""
    name = fighter_info["name"]
    url_path = fighter_info["url"]
    target = fighter_info["target_fight"]

    print(f"\n{'='*70}")
    print(f"Fighter: {name}")
    print(f"Target fight: {target}")
    print(f"{'='*70}")

    # If no URL, search for it
    if url_path is None:
        url_path = search_fighter_url(page, name)
        if url_path is None:
            print(f"  FAILED: Could not find profile URL for {name}")
            return
        time.sleep(PAGE_DELAY)

    full_url = f"{BASE_URL}{url_path}" if not url_path.startswith("http") else url_path
    print(f"  Loading: {full_url}")

    try:
        page.goto(full_url, wait_until="networkidle", timeout=30000)
        time.sleep(3)  # Wait for JS to render odds
    except Exception as e:
        print(f"  Page load error: {e}")
        return

    # --- Report page title ---
    title = page.title()
    print(f"  Page title: {title}")

    # --- Check for tables ---
    tables = page.query_selector_all("table")
    print(f"  Number of <table> elements: {len(tables)}")

    # --- Look at table classes ---
    for idx, table in enumerate(tables):
        table_class = table.get_attribute("class") or "(no class)"
        table_id = table.get_attribute("id") or "(no id)"
        rows = table.query_selector_all("tr")
        print(f"  Table {idx}: class='{table_class}', id='{table_id}', rows={len(rows)}")

        # Show first few rows to understand structure
        for ri, row in enumerate(rows[:5]):
            cells = row.query_selector_all("td, th")
            cell_texts = []
            for c in cells[:8]:  # First 8 cells
                ct = c.inner_text().strip().replace("\n", " | ")
                if len(ct) > 60:
                    ct = ct[:60] + "..."
                cell_texts.append(ct)
            print(f"    Row {ri}: {cell_texts}")

        if len(rows) > 5:
            print(f"    ... ({len(rows) - 5} more rows)")

    # --- Look for odds-table specifically ---
    odds_tables = page.query_selector_all("table.odds-table")
    print(f"\n  'odds-table' tables: {len(odds_tables)}")

    for idx, table in enumerate(odds_tables):
        rows = table.query_selector_all("tr")
        print(f"\n  odds-table {idx}: {len(rows)} rows")

        for ri, row in enumerate(rows):
            cells = row.query_selector_all("td, th")
            row_class = row.get_attribute("class") or ""

            cell_texts = []
            for c in cells[:10]:
                ct = c.inner_text().strip().replace("\n", " | ")
                if len(ct) > 50:
                    ct = ct[:50] + "..."
                cell_texts.append(ct)

            # Highlight rows that might contain method odds
            marker = ""
            full_text = " ".join(cell_texts).lower()
            if any(kw in full_text for kw in ["ko", "tko", "sub", "dec", "decision", "method"]):
                marker = " <<<< METHOD RELATED"
            if any(kw in full_text for kw in ["mitchell", "chandler", "rodriguez", "shevchenko",
                                                "pereira", "esparza", "grasso", "topuria",
                                                "poirier", "volkanovski", "adesanya", "weili"]):
                marker += " <<<< TARGET FIGHTER"

            print(f"    Row {ri} [{row_class}]: {cell_texts}{marker}")

    # --- Look for any links that lead to event or fight-specific pages ---
    fight_links = page.query_selector_all("a[href*='/events/']")
    print(f"\n  Event links on page: {len(fight_links)}")
    for link in fight_links[:15]:
        href = link.get_attribute("href") or ""
        text = link.inner_text().strip()
        if text:
            print(f"    {text} -> {href}")

    # --- Check for expandable sections, buttons, tabs ---
    buttons = page.query_selector_all("button, .toggle, .expand, [data-toggle]")
    print(f"\n  Interactive elements (buttons/toggles): {len(buttons)}")
    for b in buttons[:5]:
        print(f"    {b.get_attribute('class')} | {b.inner_text().strip()[:50]}")

    # --- Look for any text mentioning method/prop ---
    body_text = page.inner_text("body")
    method_keywords = ["method", "ko/tko", "by ko", "by tko", "by sub", "by dec",
                       "inside the distance", "itd", "prop"]
    print(f"\n  Method-related keywords in page text:")
    for kw in method_keywords:
        count = body_text.lower().count(kw)
        if count > 0:
            print(f"    '{kw}': {count} occurrences")

    if not any(body_text.lower().count(kw) for kw in method_keywords):
        print(f"    (none found)")

    # --- Sample of the raw text around fight history area ---
    print(f"\n  Page text length: {len(body_text)} chars")

    # Find opponent name in the text to locate the target fight
    opponent_name = target.split("vs ")[-1].split(",")[0].strip()
    opp_lower = opponent_name.lower()
    pos = body_text.lower().find(opp_lower)
    if pos >= 0:
        snippet_start = max(0, pos - 200)
        snippet_end = min(len(body_text), pos + 300)
        snippet = body_text[snippet_start:snippet_end].replace("\n", " | ")
        print(f"\n  Text around '{opponent_name}' mention:")
        print(f"    ...{snippet}...")
    else:
        print(f"\n  '{opponent_name}' NOT found in page text")


def main():
    print("BFO Fighter Profile Explorer")
    print("Checking if fighter profiles contain method-of-victory odds")
    print(f"Examining {len(FIGHTERS)} fighters\n")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        for fighter in FIGHTERS:
            explore_fighter_profile(page, fighter)
            time.sleep(PAGE_DELAY)

        browser.close()

    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
