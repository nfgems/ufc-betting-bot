"""Read-only Railway smoke probe for Tapology and FightDX profile recovery.

This command is meant to be run *inside* the deployed Railway service.  It
uses the real hosted network paths and the deployed parser code; it deliberately
fails outside Railway so a local-machine result cannot be mistaken for
production evidence.

No repository or volume files are written. The request wrapper records public
URLs, injects one process-local fault into each source, and otherwise delegates
to the real ``requests.get`` implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import fallback_scrapers  # noqa: E402
from src.data.name_utils import normalize_cross_source_name  # noqa: E402

TAPOLOGY_PROBE_NAME = "Ian Garry"
TAPOLOGY_PROFILE_PATH = "/fightcenter/fighters/171377-ian-garry"
FIGHTDX_PROBE_NAME = "Felix Klinkhammer"
FIGHTDX_PROFILE_PATH = "/person/felix-klinkhammer"


def _running_on_railway() -> bool:
    # These values are supplied to deployed replicas by Railway. Requiring the
    # deployment/replica markers as well as project identity makes a shell with
    # one copied Railway variable insufficient production evidence.
    return all(
        str(os.getenv(name) or "").strip()
        for name in (
            "RAILWAY_PROJECT_ID",
            "RAILWAY_SERVICE_ID",
            "RAILWAY_ENVIRONMENT_ID",
            "RAILWAY_DEPLOYMENT_ID",
            "RAILWAY_REPLICA_ID",
        )
    )


def _runtime_identity() -> dict[str, str]:
    return {
        name: str(os.getenv(name) or "").strip()
        for name in (
            "RAILWAY_PROJECT_ID",
            "RAILWAY_PROJECT_NAME",
            "RAILWAY_SERVICE_ID",
            "RAILWAY_SERVICE_NAME",
            "RAILWAY_ENVIRONMENT_ID",
            "RAILWAY_ENVIRONMENT_NAME",
            "RAILWAY_DEPLOYMENT_ID",
            "RAILWAY_REPLICA_ID",
            "RAILWAY_REPLICA_REGION",
            "RAILWAY_GIT_COMMIT_SHA",
        )
    }


def _prepared_url(url: object, params: object = None) -> str:
    try:
        return str(requests.Request("GET", str(url), params=params).prepare().url)
    except Exception:
        return str(url)


@contextmanager
def _trace_real_requests(
    *,
    inject_one_timeout_for_path: str = "",
    inject_one_status_for_path: str = "",
    injected_status_code: int = 403,
) -> Iterator[dict[str, Any]]:
    """Record requests made by the scraper while preserving real I/O."""
    trace: dict[str, Any] = {
        "attempt_urls": [],
        "real_request_urls": [],
        "injected_timeout_count": 0,
        "injected_status_count": 0,
    }
    real_get = requests.get

    def traced_get(url: object, *args: object, **kwargs: object):
        prepared_url = _prepared_url(url, kwargs.get("params"))
        trace["attempt_urls"].append(prepared_url)
        if (
            inject_one_timeout_for_path
            and inject_one_timeout_for_path in prepared_url
            and trace["injected_timeout_count"] == 0
        ):
            trace["injected_timeout_count"] = 1
            raise requests.exceptions.Timeout(
                "synthetic first-attempt timeout from read-only Railway smoke probe"
            )
        if (
            inject_one_status_for_path
            and inject_one_status_for_path in prepared_url
            and trace["injected_status_count"] == 0
        ):
            trace["injected_status_count"] = 1
            response = requests.Response()
            response.status_code = int(injected_status_code)
            response.url = prepared_url
            response.headers["server"] = "cloudflare"
            response._content = b"synthetic blocked response from Railway smoke probe"
            return response
        trace["real_request_urls"].append(prepared_url)
        return real_get(url, *args, **kwargs)

    with patch.object(fallback_scrapers.requests, "get", side_effect=traced_get):
        yield trace


def _same_identity(left: object, right: object) -> bool:
    return bool(left and right) and normalize_cross_source_name(left) == normalize_cross_source_name(right)


def _matching_request_count(calls: list[str], *, host: str, path: str) -> int:
    return sum(1 for url in calls if host in url and path in url)


def _published_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    text = str(value).strip()
    return bool(text) and text.casefold() not in {
        "nan",
        "none",
        "null",
        "n/a",
        "na",
        "?",
        "??",
        "-",
        "--",
    }


def _implementation_summary() -> dict[str, Any]:
    source_path = Path(fallback_scrapers.__file__).resolve()
    probe_path = Path(__file__).resolve()
    return {
        "fallback_scrapers_path": str(source_path),
        "fallback_scrapers_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "probe_path": str(probe_path),
        "probe_sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        "tapology_reader_block_cooldown_seconds": getattr(
            fallback_scrapers, "TAPOLOGY_READER_BLOCK_COOLDOWN_SECONDS", None
        ),
        "fightdx_request_max_attempts": getattr(
            fallback_scrapers, "FIGHTDX_REQUEST_MAX_ATTEMPTS", None
        ),
        "fightdx_failure_cooldown_seconds": getattr(
            fallback_scrapers, "FIGHTDX_FAILURE_COOLDOWN_SECONDS", None
        ),
        "fightdx_retry_status_codes": sorted(
            getattr(fallback_scrapers, "FIGHTDX_RETRY_STATUS_CODES", set())
        ),
    }


def _runtime_binding_report(
    *,
    identity: dict[str, str],
    implementation: dict[str, Any],
    expected_project_id: str = "",
    expected_service_id: str = "",
    expected_environment_id: str = "",
    expected_source_sha256: str = "",
    expected_probe_sha256: str = "",
    expected_deployment_id: str = "",
    expected_git_commit_sha: str = "",
) -> dict[str, Any]:
    """Bind a green probe to an explicitly selected deployment target/code file."""
    required_expectations = {
        "RAILWAY_PROJECT_ID": str(expected_project_id or "").strip(),
        "RAILWAY_SERVICE_ID": str(expected_service_id or "").strip(),
        "RAILWAY_ENVIRONMENT_ID": str(expected_environment_id or "").strip(),
        "fallback_scrapers_sha256": str(expected_source_sha256 or "").strip().casefold(),
        "probe_sha256": str(expected_probe_sha256 or "").strip().casefold(),
    }
    optional_expectations = {
        "RAILWAY_DEPLOYMENT_ID": str(expected_deployment_id or "").strip(),
        "RAILWAY_GIT_COMMIT_SHA": str(expected_git_commit_sha or "").strip(),
    }
    missing_expectations = [
        key for key, expected in required_expectations.items() if not expected
    ]
    mismatches: list[dict[str, str]] = []
    actual_values = {
        **identity,
        "fallback_scrapers_sha256": str(
            implementation.get("fallback_scrapers_sha256") or ""
        ).casefold(),
        "probe_sha256": str(implementation.get("probe_sha256") or "").casefold(),
    }
    for key, expected in {**required_expectations, **optional_expectations}.items():
        if not expected:
            continue
        actual = str(actual_values.get(key) or "").strip()
        comparison_actual = actual.casefold() if key.endswith("sha256") else actual
        comparison_expected = expected.casefold() if key.endswith("sha256") else expected
        if comparison_actual != comparison_expected:
            mismatches.append({"field": key, "expected": expected, "actual": actual})
    return {
        "ok": not missing_expectations and not mismatches,
        "required_expectations_present": not missing_expectations,
        "missing_expectations": missing_expectations,
        "mismatches": mismatches,
        "expected": {**required_expectations, **optional_expectations},
        "actual": actual_values,
    }


def _tapology_probe() -> dict[str, Any]:
    fallback_scrapers.clear_fallback_cache(preserve_environment_blocks=False)
    started = time.monotonic()
    search_url = fallback_scrapers._tapology_fetch_url(
        fallback_scrapers.TAPOLOGY_SEARCH_URL,
        {"term": TAPOLOGY_PROBE_NAME},
    )
    circuit_error = ""
    suppressed_error = ""
    circuit_opened = False
    suppression_preserved_attempt_count = False
    discovery_diagnostics: dict[str, object] = {}
    with _trace_real_requests(
        inject_one_status_for_path="r.jina.ai/https://www.tapology.com/search",
        injected_status_code=403,
    ) as trace:
        try:
            fallback_scrapers._get_tapology_search_markdown_with_reader(search_url)
        except fallback_scrapers.TapologyRequestError as exc:
            circuit_error = str(exc)
        circuit_opened = bool(
            circuit_error
            and fallback_scrapers._tapology_reader_cooldown_remaining_seconds() > 0
            and fallback_scrapers._tapology_reader_unavailable
        )

        attempts_after_block = len(trace["attempt_urls"])
        try:
            fallback_scrapers._get_tapology_search_markdown_with_reader(search_url)
        except fallback_scrapers.TapologyRequestError as exc:
            suppressed_error = str(exc)
        suppression_preserved_attempt_count = bool(
            suppressed_error
            and "circuit open" in suppressed_error.casefold()
            and len(trace["attempt_urls"]) == attempts_after_block
        )

        # Expire only this process-local deadline so the next call is the real
        # recovery probe. The deployed service and its files are untouched.
        fallback_scrapers._tapology_reader_unavailable_until = time.monotonic() - 1.0
        candidates = fallback_scrapers.search_tapology_candidates(
            TAPOLOGY_PROBE_NAME,
            limit=5,
            diagnostics=discovery_diagnostics,
        )
        fighter_url = next(
            (url for url in candidates if TAPOLOGY_PROFILE_PATH in url),
            "",
        )
        profile = (
            fallback_scrapers.scrape_tapology_profile(fighter_url)
            if fighter_url
            else {}
        )
        fights = (
            fallback_scrapers.scrape_tapology_fights(
                fighter_url,
                str(profile.get("name") or TAPOLOGY_PROBE_NAME),
            )
            if fighter_url and profile
            else []
        )

    profile_request_count = _matching_request_count(
        trace["real_request_urls"],
        host="r.jina.ai",
        path=TAPOLOGY_PROFILE_PATH,
    )
    required_physical_fields = ("height", "reach", "weight", "dob")
    published_physical_fields = {
        field: _published_value(profile.get(field)) for field in required_physical_fields
    }
    circuit_recovered = bool(
        discovery_diagnostics.get("healthy")
        and fighter_url
        and not fallback_scrapers._tapology_reader_unavailable
        and fallback_scrapers._tapology_reader_cooldown_remaining_seconds() <= 0
    )
    circuit_check_ok = bool(
        trace["injected_status_count"] == 1
        and circuit_opened
        and suppression_preserved_attempt_count
        and circuit_recovered
    )
    ok = bool(
        fighter_url
        and TAPOLOGY_PROFILE_PATH in fighter_url
        and _same_identity(profile.get("name"), TAPOLOGY_PROBE_NAME)
        and str(profile.get("record") or "").strip()
        and fights
        and all(published_physical_fields.values())
        and profile_request_count == 1
        and circuit_check_ok
    )
    return {
        "ok": ok,
        "reader_preferred": fallback_scrapers._tapology_prefer_reader(),
        "fighter_url": fighter_url,
        "parsed_name": profile.get("name"),
        "parsed_record": profile.get("record"),
        "parsed_height_cm": profile.get("height"),
        "parsed_reach_cm": profile.get("reach"),
        "parsed_weight_lbs": profile.get("weight"),
        "parsed_dob": profile.get("dob"),
        "parsed_fight_count": len(fights),
        "published_physical_fields": published_physical_fields,
        "discovery_diagnostics": discovery_diagnostics,
        "circuit_check_ok": circuit_check_ok,
        "circuit_opened_after_synthetic_403": circuit_opened,
        "circuit_suppressed_request_while_open": suppression_preserved_attempt_count,
        "circuit_recovered_through_real_request": circuit_recovered,
        "circuit_open_error": circuit_error,
        "circuit_suppressed_error": suppressed_error,
        "injected_status_count": trace["injected_status_count"],
        "reader_profile_request_count": profile_request_count,
        "request_attempt_urls": trace["attempt_urls"],
        "real_request_urls": trace["real_request_urls"],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def _fightdx_probe(*, exercise_retry: bool = False) -> dict[str, Any]:
    fallback_scrapers.clear_fallback_cache(preserve_environment_blocks=False)
    started = time.monotonic()
    with _trace_real_requests(
        inject_one_timeout_for_path=(FIGHTDX_PROFILE_PATH if exercise_retry else "")
    ) as trace:
        fighter_url = fallback_scrapers.search_fightdx(FIGHTDX_PROBE_NAME)
        profile = (
            fallback_scrapers.scrape_fightdx_profile(fighter_url)
            if fighter_url
            else {}
        )

    profile_request_count = _matching_request_count(
        trace["real_request_urls"],
        host="fightdx.com",
        path=FIGHTDX_PROFILE_PATH,
    )
    has_published_physical_field = any(
        _published_value(profile.get(field))
        for field in ("height", "reach", "weight")
    )
    profile_attempt_count = _matching_request_count(
        trace["attempt_urls"],
        host="fightdx.com",
        path=FIGHTDX_PROFILE_PATH,
    )
    retry_check_ok = (
        trace["injected_timeout_count"] == 1 and profile_attempt_count == 2
        if exercise_retry
        else trace["injected_timeout_count"] == 0 and profile_attempt_count == 1
    )
    ok = bool(
        fighter_url
        and FIGHTDX_PROFILE_PATH in fighter_url
        and _same_identity(profile.get("name"), FIGHTDX_PROBE_NAME)
        and has_published_physical_field
        # Search verifies the direct profile and hands that exact HTML to the
        # parser. One GET proves the duplicate-fetch bug is absent.
        and profile_request_count == 1
        and retry_check_ok
    )
    return {
        "ok": ok,
        "fighter_url": fighter_url,
        "parsed_name": profile.get("name"),
        "parsed_record": profile.get("record"),
        "parsed_height_cm": profile.get("height"),
        "parsed_reach_cm": profile.get("reach"),
        "parsed_weight_lbs": profile.get("weight"),
        "parsed_dob": profile.get("dob"),
        "profile_request_count": profile_request_count,
        "profile_attempt_count": profile_attempt_count,
        "synthetic_retry_exercised": exercise_retry,
        "injected_timeout_count": trace["injected_timeout_count"],
        "request_attempt_urls": trace["attempt_urls"],
        "real_request_urls": trace["real_request_urls"],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exercise-fightdx-retry",
        action="store_true",
        help=(
            "Inject one timeout in this ephemeral probe process, then require the "
            "deployed retry path to recover through a real hosted FightDX request."
        ),
    )
    parser.add_argument("--expected-project-id", default="")
    parser.add_argument("--expected-service-id", default="")
    parser.add_argument("--expected-environment-id", default="")
    parser.add_argument(
        "--expected-source-sha256",
        default="",
        help="Expected SHA-256 of the deployed src/data/fallback_scrapers.py file.",
    )
    parser.add_argument(
        "--expected-probe-sha256",
        default="",
        help="Expected SHA-256 of this deployed probe script.",
    )
    parser.add_argument("--expected-deployment-id", default="")
    parser.add_argument("--expected-git-commit-sha", default="")
    args = parser.parse_args(argv)
    identity = _runtime_identity()
    implementation = _implementation_summary()
    result: dict[str, Any] = {
        "railway_environment": _running_on_railway(),
        "read_only": True,
        "runtime_identity": identity,
        "implementation": implementation,
    }
    if not result["railway_environment"]:
        result.update(
            {
                "ok": False,
                "error": (
                    "This probe must run inside Railway; local execution is not "
                    "accepted as production evidence."
                ),
            }
        )
        print(json.dumps(result, indent=2, default=str))
        return 2

    result["runtime_binding"] = _runtime_binding_report(
        identity=identity,
        implementation=implementation,
        expected_project_id=args.expected_project_id,
        expected_service_id=args.expected_service_id,
        expected_environment_id=args.expected_environment_id,
        expected_source_sha256=args.expected_source_sha256,
        expected_probe_sha256=args.expected_probe_sha256,
        expected_deployment_id=args.expected_deployment_id,
        expected_git_commit_sha=args.expected_git_commit_sha,
    )
    if not result["runtime_binding"]["ok"]:
        result.update(
            {
                "ok": False,
                "error": (
                    "Railway runtime identity or deployed source did not match the "
                    "explicit production expectations; network probes were not run."
                ),
            }
        )
        print(json.dumps(result, indent=2, default=str))
        return 3
    if not args.exercise_fightdx_retry:
        result.update(
            {
                "ok": False,
                "error": (
                    "--exercise-fightdx-retry is required so a green production "
                    "probe proves the deployed retry path, not only the happy path."
                ),
            }
        )
        print(json.dumps(result, indent=2, default=str))
        return 3

    probes = (
        ("tapology", _tapology_probe),
        (
            "fightdx",
            lambda: _fightdx_probe(exercise_retry=args.exercise_fightdx_retry),
        ),
    )
    for source, probe in probes:
        try:
            result[source] = probe()
        except Exception as exc:
            result[source] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    result["ok"] = bool(
        result.get("runtime_binding", {}).get("ok")
        and result.get("tapology", {}).get("ok")
        and result.get("fightdx", {}).get("ok")
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
