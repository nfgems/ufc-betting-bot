"""Probe Railway Tapology runtime behavior.

This script is intentionally read-only. It verifies that Railway production
does not attempt Tapology reader/origin/site-search fetches by default and that
fallback lookup can still recover static profile data from non-Tapology sources.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_scheduled_ufc_refresh as scheduled_refresh
from src.data import fallback_scrapers


def _json_default(value: Any) -> str:
    return str(value)


def main() -> int:
    tapology_calls: list[str] = []

    def fail_tapology_path(*_args: object, **_kwargs: object) -> Any:
        tapology_calls.append("tapology_network_path")
        raise AssertionError("Railway runtime attempted a Tapology network path")

    fallback_scrapers.clear_fallback_cache()
    fallback_scrapers._search_tapology_candidates_with_reader = fail_tapology_path  # type: ignore[assignment]
    fallback_scrapers._get_tapology_soup = fail_tapology_path  # type: ignore[assignment]
    fallback_scrapers._search_site_candidates = fail_tapology_path  # type: ignore[assignment]
    candidates = fallback_scrapers.search_tapology_candidates("Ian Garry", limit=1)

    fallback_scrapers.clear_fallback_cache()
    fallback_scrapers.search_sherdog = lambda _name: None  # type: ignore[assignment]
    fallback_scrapers.search_tapology = fail_tapology_path  # type: ignore[assignment]
    fallback_scrapers.scrape_tapology_profile = fail_tapology_path  # type: ignore[assignment]
    fallback_scrapers.scrape_tapology_fights = fail_tapology_path  # type: ignore[assignment]
    fallback_scrapers.search_espn = lambda _name: "https://www.espn.com/mma/fighter/_/id/1/ian-garry"  # type: ignore[assignment]
    fallback_scrapers.scrape_espn_profile = lambda _url: {  # type: ignore[assignment]
        "name": "Ian Garry",
        "fighter_url": _url,
        "record": "",
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "height_raw": "6' 3\"",
        "height": 190.5,
        "reach_raw": "74\"",
        "reach": 187.96,
        "weight_raw": "170 lbs",
        "weight": 170.0,
        "stance": "Orthodox",
        "dob": "1997-11-17",
    }
    fallback_scrapers.search_martialbot = lambda _name: None  # type: ignore[assignment]
    fallback_scrapers.search_fightdx = lambda _name: None  # type: ignore[assignment]
    fallback_result = fallback_scrapers.fallback_lookup("Ian Garry")
    fallback_profile = fallback_result[0] if fallback_result else {}

    hosted_runtime = scheduled_refresh.is_hosted_runtime()
    sources = scheduled_refresh._profile_supplement_refresh_sources()
    result = {
        "railway_environment": bool(
            os.getenv("RAILWAY_PROJECT_ID")
            or os.getenv("RAILWAY_SERVICE_ID")
            or os.getenv("RAILWAY_ENVIRONMENT")
        ),
        "hosted_runtime": hosted_runtime,
        "tapology_runtime_fetch_allowed": fallback_scrapers._tapology_runtime_fetch_allowed(),
        "tapology_profile_fetch_available": fallback_scrapers._tapology_profile_fetch_available(),
        "tapology_candidates": candidates,
        "tapology_network_path_calls": tapology_calls,
        "fallback_lookup_profile_name": fallback_profile.get("name", ""),
        "fallback_lookup_reach": fallback_profile.get("reach"),
        "hosted_profile_supplement_sources": sources,
    }
    result["ok"] = (
        result["railway_environment"] is True
        and result["tapology_runtime_fetch_allowed"] is False
        and result["tapology_profile_fetch_available"] is False
        and result["tapology_candidates"] == []
        and result["tapology_network_path_calls"] == []
        and result["fallback_lookup_profile_name"] == "Ian Garry"
        and result["hosted_runtime"] is True
        and "tapology" not in sources
    )
    print(json.dumps(result, indent=2, default=_json_default))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
