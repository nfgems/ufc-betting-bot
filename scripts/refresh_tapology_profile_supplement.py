"""Hosted Tapology profile supplement refresh and access probe.

This is intended for cloud schedulers such as GitHub Actions. It verifies that
Tapology is reachable from the hosted runtime before attempting a refresh, then
updates the repository's supplemental physical-profile artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_profile_supplement_from_external_profiles import run_profile_supplement_refresh
from src.config import RAW_DATA_DIR
from src.data.fallback_scrapers import (
    TapologyRequestError,
    clear_fallback_cache,
    scrape_tapology_profile,
    search_tapology_candidates,
)
from src.data.name_utils import same_person_name
from src.data.ufc_active_roster import OFFICIAL_ACTIVE_ROSTER_PATH, sync_official_active_roster


logger = logging.getLogger(__name__)

DEFAULT_SCRAPED_FIGHTERS_PATH = RAW_DATA_DIR / "ufc_fighters_scraped.csv"
DEFAULT_PROFILE_SUPPLEMENT_PATH = RAW_DATA_DIR / "ufc_fighters_profile_supplement.csv"
DEFAULT_PROBE_NAME = "Andre Fili"
PROFILE_FIELDS = ("height", "reach", "weight", "dob")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _present(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text and text.lower() not in {"nan", "none", "n/a", "??", "-", "--"})


def probe_tapology_access(*, fighter_name: str = DEFAULT_PROBE_NAME, fighter_url: str = "") -> dict[str, object]:
    clear_fallback_cache(preserve_environment_blocks=False)
    errors: list[str] = []
    candidates = [fighter_url] if fighter_url else search_tapology_candidates(fighter_name, limit=3)
    if not candidates:
        return {
            "ok": False,
            "fighter_name": fighter_name,
            "candidate_urls": [],
            "errors": ["no Tapology profile candidates found"],
        }

    for candidate_url in candidates:
        try:
            profile = scrape_tapology_profile(candidate_url)
        except TapologyRequestError as exc:
            errors.append(f"{candidate_url}: {exc}")
            continue
        except Exception as exc:
            errors.append(f"{candidate_url}: {type(exc).__name__}: {exc}")
            continue

        parsed_name = str(profile.get("name") or "").strip()
        fields = {field: profile.get(field) for field in PROFILE_FIELDS if _present(profile.get(field))}
        if parsed_name and not same_person_name(fighter_name, parsed_name):
            errors.append(f"{candidate_url}: parsed profile name {parsed_name!r} did not match {fighter_name!r}")
            continue
        if not fields:
            errors.append(f"{candidate_url}: profile parsed but no physical fields were present")
            continue
        return {
            "ok": True,
            "fighter_name": fighter_name,
            "candidate_url": candidate_url,
            "parsed_name": parsed_name,
            "fields": fields,
        }

    return {
        "ok": False,
        "fighter_name": fighter_name,
        "candidate_urls": candidates,
        "errors": errors,
    }


def run_hosted_refresh(args: argparse.Namespace) -> dict[str, object]:
    probe = probe_tapology_access(fighter_name=args.probe_name, fighter_url=args.probe_url)
    if not probe["ok"]:
        raise SystemExit(f"Tapology hosted access probe failed: {json.dumps(probe, default=_json_default)}")

    if args.probe_only:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "action": "probe_only",
            "tapology_probe": probe,
        }

    roster_summary: dict[str, object] = {"action": "skip", "reason": "disabled"}
    if args.sync_active_roster:
        roster_df = sync_official_active_roster(output_path=args.active_roster_path)
        roster_summary = {
            "action": "synced",
            "rows": int(len(roster_df)),
            "output_path": str(args.active_roster_path),
        }

    refresh_summary = run_profile_supplement_refresh(
        scraped_fighters_path=args.scraped_fighters_path,
        candidate_source_csv=args.active_roster_path,
        output_path=args.output,
        sources=["tapology"],
        limit=args.limit,
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "action": "refreshed",
        "tapology_probe": probe,
        "active_roster": roster_summary,
        "profile_supplement_refresh": refresh_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scraped-fighters-path", type=Path, default=DEFAULT_SCRAPED_FIGHTERS_PATH)
    parser.add_argument("--active-roster-path", type=Path, default=OFFICIAL_ACTIVE_ROSTER_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROFILE_SUPPLEMENT_PATH)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--sync-active-roster", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-name", default=DEFAULT_PROBE_NAME)
    parser.add_argument("--probe-url", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args()
    summary = run_hosted_refresh(args)
    if args.json:
        print(json.dumps(summary, indent=2, default=_json_default))
    else:
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
