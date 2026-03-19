"""
Search Wayback Machine for archived BFO event pages (2022-2023).
Uses the CDX API to find snapshots, then checks for method-of-victory odds.
"""

import sys
import time
import requests
import re

sys.stdout.reconfigure(encoding="utf-8")

CDX_API = "https://web.archive.org/cdx/search/cdx"
WAYBACK_BASE = "https://web.archive.org/web"
DELAY = 4  # seconds between requests

# Target events with multiple URL search patterns
EVENTS = [
    {
        "name": "UFC 290 - Volkanovski vs Rodriguez (Jul 2023)",
        "patterns": [
            "https://www.bestfightodds.com/events/ufc-290-*",
            "https://www.bestfightodds.com/events/*volkanovski-vs-rodriguez*",
            "bestfightodds.com/events/ufc-290*",
            "https://www.bestfightodds.com/events/ufc-290-volkanovski-vs-rodriguez",
        ],
    },
    {
        "name": "UFC 287 - Pereira vs Adesanya 2 (Apr 2023)",
        "patterns": [
            "https://www.bestfightodds.com/events/ufc-287-*",
            "https://www.bestfightodds.com/events/*pereira-vs-adesanya*",
            "bestfightodds.com/events/ufc-287*",
            "https://www.bestfightodds.com/events/ufc-287-pereira-vs-adesanya-2",
        ],
    },
    {
        "name": "UFC 282 - Blachowicz vs Ankalaev (Dec 2022)",
        "patterns": [
            "https://www.bestfightodds.com/events/ufc-282-*",
            "https://www.bestfightodds.com/events/*blachowicz-vs-ankalaev*",
            "bestfightodds.com/events/ufc-282*",
        ],
    },
    {
        "name": "UFC 281 - Adesanya vs Pereira (Nov 2022)",
        "patterns": [
            "https://www.bestfightodds.com/events/ufc-281-*",
            "https://www.bestfightodds.com/events/*adesanya-vs-pereira*",
            "bestfightodds.com/events/ufc-281*",
        ],
    },
    {
        "name": "UFC FN: Grasso vs Shevchenko 2 (Sep 2023)",
        "patterns": [
            "https://www.bestfightodds.com/events/*grasso-vs-shevchenko*",
            "bestfightodds.com/events/*grasso-vs-shevchenko*",
            "https://www.bestfightodds.com/events/ufc-fight-night-*grasso*",
            "https://www.bestfightodds.com/events/ufc-on-espn-*grasso*",
            "https://www.bestfightodds.com/events/noche-de-ufc*",
        ],
    },
    {
        "name": "UFC FN: Sandhagen vs Font (Aug 2023)",
        "patterns": [
            "https://www.bestfightodds.com/events/*sandhagen-vs-font*",
            "bestfightodds.com/events/*sandhagen-vs-font*",
            "https://www.bestfightodds.com/events/ufc-fight-night-*sandhagen*",
            "https://www.bestfightodds.com/events/ufc-on-espn-*sandhagen*",
        ],
    },
    {
        "name": "UFC FN: Hermansson vs Strickland (Feb 2022)",
        "patterns": [
            "https://www.bestfightodds.com/events/*hermansson-vs-strickland*",
            "bestfightodds.com/events/*hermansson-vs-strickland*",
            "https://www.bestfightodds.com/events/ufc-fight-night-*hermansson*",
            "https://www.bestfightodds.com/events/ufc-on-espn-*hermansson*",
            "https://www.bestfightodds.com/events/ufc-vegas-47*",
        ],
    },
    {
        "name": "UFC FN: Marreta vs Anders (Sep 2018)",
        "patterns": [
            "https://www.bestfightodds.com/events/*santos-vs-anders*",
            "https://www.bestfightodds.com/events/*marreta*",
            "bestfightodds.com/events/*santos-vs-anders*",
            "https://www.bestfightodds.com/events/ufc-fight-night-*anders*",
            "https://www.bestfightodds.com/events/ufc-fight-night-137*",
        ],
    },
]

# Also do broad prefix searches for all BFO event pages in the date ranges
BROAD_SEARCHES = [
    {
        "name": "All BFO UFC 28x events",
        "url": "https://www.bestfightodds.com/events/ufc-28",
        "matchType": "prefix",
    },
    {
        "name": "All BFO UFC 29x events",
        "url": "https://www.bestfightodds.com/events/ufc-29",
        "matchType": "prefix",
    },
]


def cdx_search(url_pattern, match_type="prefix", from_ts=None, to_ts=None):
    """Query the Wayback CDX API for archived snapshots."""
    # Clean up the pattern for the CDX API
    # CDX supports: exact, prefix, host, domain matchType
    # For wildcard patterns, we use prefix with the part before the wildcard

    clean_url = url_pattern
    use_match_type = match_type

    if "*" in url_pattern:
        # If wildcard is at the end, use prefix match with the part before *
        if url_pattern.endswith("*"):
            clean_url = url_pattern.rstrip("*")
            use_match_type = "prefix"
        # If wildcard is in the middle, we need to use prefix up to the wildcard
        # and then filter results
        else:
            parts = url_pattern.split("*")
            clean_url = parts[0]
            use_match_type = "prefix"

    params = {
        "url": clean_url,
        "matchType": use_match_type,
        "output": "json",
        "fl": "timestamp,original,statuscode,mimetype,length",
        "collapse": "urlkey",  # deduplicate by URL
        "limit": 50,
    }
    if from_ts:
        params["from"] = from_ts
    if to_ts:
        params["to"] = to_ts

    try:
        resp = requests.get(CDX_API, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if len(data) > 1:  # first row is header
            return data[1:]  # skip header row
        return []
    except Exception as e:
        print(f"  CDX error for {clean_url}: {e}")
        return []


def check_method_odds(wayback_url):
    """Fetch a Wayback snapshot and check for method-of-victory odds."""
    try:
        resp = requests.get(wayback_url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (research bot; method odds archival search)"
        })
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"

        text = resp.text

        # Check for method odds indicators
        method_patterns = [
            r"wins\s+by\s+TKO/KO",
            r"wins\s+by\s+submission",
            r"wins\s+by\s+decision",
            r"wins\s+by\s+KO/TKO",
            r"by\s+TKO/KO",
            r"by\s+submission",
            r"by\s+decision",
            r"Method of Victory",
            r"method-of-victory",
            r"Fight Goes the Distance",
            r"Goes the Distance",
            r"fight-goes-the-distance",
        ]

        found_methods = []
        for pat in method_patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            if matches:
                found_methods.extend(matches)

        # Check for actual odds values near method text
        # BFO uses formats like +250, -150, etc.
        odds_near_method = False
        for pat in [r"wins\s+by\s+(?:TKO/KO|submission|decision)", r"Goes the Distance"]:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                # Check surrounding context (500 chars) for odds values
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 500)
                context = text[start:end]
                odds = re.findall(r"[+-]\d{3,4}", context)
                if odds:
                    odds_near_method = True
                    found_methods.append(f"ODDS FOUND: {odds[:6]}")

        # Also check for fight-specific prop bet sections
        prop_indicators = re.findall(r"(?:prop|props|specials)", text, re.IGNORECASE)

        if found_methods:
            return True, f"Method odds indicators: {', '.join(set(found_methods[:10]))}"
        elif prop_indicators:
            return False, "Has prop section but no method odds found"
        else:
            return False, "No method odds content found"
    except Exception as e:
        return None, f"Fetch error: {e}"


def main():
    print("=" * 80)
    print("WAYBACK MACHINE SEARCH FOR ARCHIVED BFO EVENT PAGES")
    print("=" * 80)

    all_results = {}

    # Search for each target event
    for event in EVENTS:
        print(f"\n{'='*70}")
        print(f"EVENT: {event['name']}")
        print(f"{'='*70}")

        event_snapshots = []
        seen_urls = set()

        for pattern in event["patterns"]:
            print(f"\n  Pattern: {pattern}")
            time.sleep(DELAY)

            results = cdx_search(pattern)

            if results:
                for row in results:
                    timestamp, original_url, status, mime, length = row
                    if original_url not in seen_urls:
                        seen_urls.add(original_url)
                        event_snapshots.append((timestamp, original_url, status))
                        print(f"    FOUND: [{timestamp}] {original_url} (HTTP {status})")
            else:
                print(f"    No results")

        if event_snapshots:
            print(f"\n  Total unique URLs found: {len(event_snapshots)}")
            all_results[event["name"]] = event_snapshots
        else:
            print(f"\n  NO SNAPSHOTS FOUND for this event")

    # Broad prefix searches
    print(f"\n{'='*70}")
    print("BROAD PREFIX SEARCHES")
    print(f"{'='*70}")

    for search in BROAD_SEARCHES:
        print(f"\n  Search: {search['name']} ({search['url']})")
        time.sleep(DELAY)

        results = cdx_search(search["url"], match_type=search["matchType"])
        if results:
            for row in results:
                timestamp, original_url, status, mime, length = row
                print(f"    [{timestamp}] {original_url} (HTTP {status})")
        else:
            print(f"    No results")

    # Also search for BFO method odds pages specifically
    print(f"\n{'='*70}")
    print("SEARCHING FOR BFO METHOD ODDS URL PATTERNS")
    print(f"{'='*70}")

    method_odds_patterns = [
        "https://www.bestfightodds.com/events/ufc-290",
        "https://www.bestfightodds.com/events/ufc-287",
        "https://www.bestfightodds.com/events/ufc-282",
        "https://www.bestfightodds.com/events/ufc-281",
    ]

    for pattern in method_odds_patterns:
        print(f"\n  Searching prefix: {pattern}")
        time.sleep(DELAY)

        # Search without collapse to see all timestamps
        params = {
            "url": pattern,
            "matchType": "prefix",
            "output": "json",
            "fl": "timestamp,original,statuscode,mimetype,length",
            "limit": 100,
        }
        try:
            resp = requests.get(CDX_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if len(data) > 1:
                print(f"    Found {len(data)-1} snapshots:")
                # Group by URL
                urls = {}
                for row in data[1:]:
                    ts, url, status, mime, length = row
                    if url not in urls:
                        urls[url] = []
                    urls[url].append(ts)

                for url, timestamps in urls.items():
                    print(f"      {url}")
                    print(f"        Timestamps: {', '.join(timestamps[:5])}" +
                          (f" ... +{len(timestamps)-5} more" if len(timestamps) > 5 else ""))
            else:
                print(f"    No results")
        except Exception as e:
            print(f"    Error: {e}")

    # Now fetch and check the most promising snapshots for method odds
    print(f"\n{'='*70}")
    print("CHECKING SNAPSHOTS FOR METHOD-OF-VICTORY ODDS")
    print(f"{'='*70}")

    snapshots_to_check = []
    for event_name, snapshots in all_results.items():
        for timestamp, url, status in snapshots:
            if status in ("200", "-"):
                snapshots_to_check.append((event_name, timestamp, url))

    if not snapshots_to_check:
        print("\n  No HTTP 200 snapshots to check.")
    else:
        print(f"\n  Checking {len(snapshots_to_check)} snapshots...")

        for event_name, timestamp, url in snapshots_to_check[:20]:  # limit to 20
            wayback_url = f"{WAYBACK_BASE}/{timestamp}/{url}"
            print(f"\n  Checking: {event_name}")
            print(f"    URL: {wayback_url}")
            time.sleep(DELAY)

            has_methods, detail = check_method_odds(wayback_url)
            if has_methods:
                print(f"    *** HAS METHOD ODDS: {detail}")
            elif has_methods is None:
                print(f"    Error: {detail}")
            else:
                print(f"    No method odds: {detail}")

    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if all_results:
        for event_name, snapshots in all_results.items():
            print(f"\n  {event_name}:")
            for ts, url, status in snapshots:
                print(f"    [{ts}] {url} (HTTP {status})")
    else:
        print("\n  No archived snapshots found for any target events.")

    print("\nDone.")


if __name__ == "__main__":
    main()
