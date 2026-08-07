"""Wait until production proves that one exact refit release is live and trading."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Mapping
from datetime import date
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_READYZ_URL = "https://ufc-bot-production.up.railway.app/readyz"


def fetch_readyz(url: str, timeout_seconds: float) -> tuple[int, Any]:
    """Fetch and decode a readyz response, including JSON error responses."""
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Cache-Control": "no-cache",
            "User-Agent": "ufc-betting-bot-deployment-gate/2.0",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.getcode())
            body = response.read()
    except HTTPError as exc:
        status = int(exc.code)
        body = exc.read()

    if not body:
        raise ValueError(f"ready endpoint returned an empty HTTP {status} response")
    try:
        return status, json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"ready endpoint returned invalid JSON with HTTP {status}: {exc}"
        ) from exc


def _runtime_error_summary(payload: Mapping[str, Any]) -> str:
    errors = payload.get("errors")
    if not isinstance(errors, list):
        return ""
    messages: list[str] = []
    for error in errors[:3]:
        if isinstance(error, str) and error.strip():
            messages.append(error.strip())
        elif isinstance(error, Mapping):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                messages.append(message.strip())
    return " | ".join(messages)


def _expected_text(value: str, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    return text


def _expected_sha256(value: str, *, field: str) -> str:
    text = _expected_text(value, field=field).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA-256")
    return text


def _bundle_text(bundle: Mapping[str, Any], field: str) -> str | None:
    value = bundle.get(field)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _require_text_match(
    bundle: Mapping[str, Any],
    *,
    field: str,
    expected: str,
    label: str | None = None,
    case_insensitive: bool = False,
) -> tuple[bool, str]:
    actual = _bundle_text(bundle, field)
    display = label or f"production_bundle.{field}"
    if actual is None:
        return False, f"matching ready response is missing {display}"
    matches = actual.lower() == expected.lower() if case_insensitive else actual == expected
    if not matches:
        return False, f"{display} is {actual}; expected {expected}"
    return True, ""


def evaluate_readyz(
    http_status: int,
    payload: Any,
    *,
    expected_bundle_id: str,
    expected_release_id: str,
    expected_installed_manifest_sha256: str,
    expected_source_manifest_sha256: str,
    expected_deployed_git_sha: str,
    expected_training_source_git_sha: str,
    expected_spec: str,
    expected_model_sha256: str,
    expected_no_odds_model_sha256: str,
    expected_logistic_model_sha256: str,
    expected_training_fights_sha256: str,
    expected_training_features_sha256: str,
    expected_training_snapshot_max_event_date: str | None = None,
    expected_runtime_lookup_fights_sha256: str | None = None,
    expected_runtime_lookup_features_sha256: str | None = None,
) -> tuple[bool, str]:
    """Return whether readyz proves that the exact release is live and armed.

    Immutable training identities and mutable runtime lookup identities are
    deliberately checked through different fields.  The latter can never
    satisfy a missing or mismatched immutable identity.
    """
    expected_values = {
        "bundle_id": _expected_text(expected_bundle_id, field="expected_bundle_id"),
        "rich_release_id": _expected_text(
            expected_release_id, field="expected_release_id"
        ),
        "deployed_git_sha": _expected_text(
            expected_deployed_git_sha, field="expected_deployed_git_sha"
        ),
        "training_source_git_sha": _expected_text(
            expected_training_source_git_sha,
            field="expected_training_source_git_sha",
        ),
        "model_spec_name": _expected_text(expected_spec, field="expected_spec"),
    }
    expected_hashes = {
        "installed_manifest_sha256": _expected_sha256(
            expected_installed_manifest_sha256,
            field="expected_installed_manifest_sha256",
        ),
        "source_manifest_sha256": _expected_sha256(
            expected_source_manifest_sha256,
            field="expected_source_manifest_sha256",
        ),
        "model_sha256": _expected_sha256(
            expected_model_sha256, field="expected_model_sha256"
        ),
        "no_odds_model_sha256": _expected_sha256(
            expected_no_odds_model_sha256,
            field="expected_no_odds_model_sha256",
        ),
        "logistic_model_sha256": _expected_sha256(
            expected_logistic_model_sha256,
            field="expected_logistic_model_sha256",
        ),
        "immutable_training_fights_sha256": _expected_sha256(
            expected_training_fights_sha256,
            field="expected_training_fights_sha256",
        ),
        "immutable_training_features_sha256": _expected_sha256(
            expected_training_features_sha256,
            field="expected_training_features_sha256",
        ),
    }
    optional_runtime_hashes: dict[str, str] = {}
    if expected_runtime_lookup_fights_sha256 is not None:
        optional_runtime_hashes["processed_fights_sha256"] = _expected_sha256(
            expected_runtime_lookup_fights_sha256,
            field="expected_runtime_lookup_fights_sha256",
        )
    if expected_runtime_lookup_features_sha256 is not None:
        optional_runtime_hashes["processed_features_sha256"] = _expected_sha256(
            expected_runtime_lookup_features_sha256,
            field="expected_runtime_lookup_features_sha256",
        )
    if expected_training_snapshot_max_event_date is not None:
        expected_training_snapshot_max_event_date = _expected_text(
            expected_training_snapshot_max_event_date,
            field="expected_training_snapshot_max_event_date",
        )

    if not isinstance(payload, Mapping):
        return False, f"ready endpoint returned JSON {type(payload).__name__}, not an object"
    if http_status != 200 or payload.get("ready") is not True:
        detail = _runtime_error_summary(payload)
        suffix = f"; runtime errors: {detail}" if detail else ""
        return False, f"service is not ready (HTTP {http_status}){suffix}"

    required_runtime_state = {
        "requested_live_mode": "real",
        "effective_live_mode": "real",
    }
    for field, expected in required_runtime_state.items():
        actual = payload.get(field)
        if not isinstance(actual, str) or actual.strip().lower() != expected:
            return False, f"{field} is {actual!r}; expected {expected!r}"
    for field in ("armed_for_real", "trading_enabled", "trading_live"):
        if payload.get(field) is not True:
            return False, f"{field} is not true"

    components = payload.get("components")
    if not isinstance(components, Mapping):
        return False, "components metadata is missing"
    betting_loop = components.get("betting_loop")
    if not isinstance(betting_loop, Mapping):
        return False, "components.betting_loop metadata is missing"
    state = betting_loop.get("state")
    if not isinstance(state, str) or state.strip().lower() != "running":
        return False, f"components.betting_loop.state is {state!r}; expected 'running'"
    if betting_loop.get("thread_alive") is not True:
        return False, "components.betting_loop.thread_alive is not true"

    bundle = payload.get("production_bundle")
    if not isinstance(bundle, Mapping):
        return False, "service is ready but production_bundle metadata is missing"

    labels = {
        "rich_release_id": "production_bundle.rich_release_id",
        "immutable_training_fights_sha256": (
            "production_bundle.immutable_training_fights_sha256"
        ),
        "immutable_training_features_sha256": (
            "production_bundle.immutable_training_features_sha256"
        ),
    }
    for field, expected in expected_values.items():
        matched, reason = _require_text_match(
            bundle,
            field=field,
            expected=expected,
            label=labels.get(field),
        )
        if not matched:
            return False, reason
    for field, expected in {**expected_hashes, **optional_runtime_hashes}.items():
        matched, reason = _require_text_match(
            bundle,
            field=field,
            expected=expected,
            label=labels.get(field),
            case_insensitive=True,
        )
        if not matched:
            return False, reason

    if expected_training_snapshot_max_event_date is not None:
        matched, reason = _require_text_match(
            bundle,
            field="immutable_training_snapshot_max_event_date",
            expected=expected_training_snapshot_max_event_date,
            label="production_bundle.immutable_training_snapshot_max_event_date",
        )
        if not matched:
            return False, reason

    runtime_note = ""
    if optional_runtime_hashes:
        runtime_note = " and the expected runtime lookup"
    return True, (
        f"production is ready on release {expected_values['rich_release_id']} "
        f"(bundle {expected_values['bundle_id']}) with immutable training identity"
        f"{runtime_note}, real-money arming, and a live betting loop"
    )


def wait_for_production_deployment(
    *,
    url: str,
    expected_bundle_id: str,
    expected_release_id: str,
    expected_installed_manifest_sha256: str,
    expected_source_manifest_sha256: str,
    expected_deployed_git_sha: str,
    expected_training_source_git_sha: str,
    expected_spec: str,
    expected_model_sha256: str,
    expected_no_odds_model_sha256: str,
    expected_logistic_model_sha256: str,
    expected_training_fights_sha256: str,
    expected_training_features_sha256: str,
    timeout_seconds: float,
    poll_seconds: float,
    request_timeout_seconds: float,
    expected_training_snapshot_max_event_date: str | None = None,
    expected_runtime_lookup_fights_sha256: str | None = None,
    expected_runtime_lookup_features_sha256: str | None = None,
    successes_required: int = 2,
    fetch: Callable[[str, float], tuple[int, Any]] = fetch_readyz,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    emit: Callable[[str], None] = print,
) -> Any:
    """Poll readyz until consecutive responses prove the exact live release."""
    expectations = {
        "expected_bundle_id": expected_bundle_id,
        "expected_release_id": expected_release_id,
        "expected_installed_manifest_sha256": expected_installed_manifest_sha256,
        "expected_source_manifest_sha256": expected_source_manifest_sha256,
        "expected_deployed_git_sha": expected_deployed_git_sha,
        "expected_training_source_git_sha": expected_training_source_git_sha,
        "expected_spec": expected_spec,
        "expected_model_sha256": expected_model_sha256,
        "expected_no_odds_model_sha256": expected_no_odds_model_sha256,
        "expected_logistic_model_sha256": expected_logistic_model_sha256,
        "expected_training_fights_sha256": expected_training_fights_sha256,
        "expected_training_features_sha256": expected_training_features_sha256,
        "expected_training_snapshot_max_event_date": (
            expected_training_snapshot_max_event_date
        ),
        "expected_runtime_lookup_fights_sha256": (
            expected_runtime_lookup_fights_sha256
        ),
        "expected_runtime_lookup_features_sha256": (
            expected_runtime_lookup_features_sha256
        ),
    }
    # Validate all expected values before the first network request.
    _expected_text(expected_bundle_id, field="expected_bundle_id")
    _expected_text(expected_release_id, field="expected_release_id")
    _expected_text(expected_deployed_git_sha, field="expected_deployed_git_sha")
    _expected_text(
        expected_training_source_git_sha,
        field="expected_training_source_git_sha",
    )
    _expected_text(expected_spec, field="expected_spec")
    for field, value in (
        ("expected_installed_manifest_sha256", expected_installed_manifest_sha256),
        ("expected_source_manifest_sha256", expected_source_manifest_sha256),
        ("expected_model_sha256", expected_model_sha256),
        ("expected_no_odds_model_sha256", expected_no_odds_model_sha256),
        ("expected_logistic_model_sha256", expected_logistic_model_sha256),
        ("expected_training_fights_sha256", expected_training_fights_sha256),
        ("expected_training_features_sha256", expected_training_features_sha256),
    ):
        _expected_sha256(value, field=field)
    for field, value in (
        (
            "expected_runtime_lookup_fights_sha256",
            expected_runtime_lookup_fights_sha256,
        ),
        (
            "expected_runtime_lookup_features_sha256",
            expected_runtime_lookup_features_sha256,
        ),
    ):
        if value is not None:
            _expected_sha256(value, field=field)
    if expected_training_snapshot_max_event_date is not None:
        snapshot_date = _expected_text(
            expected_training_snapshot_max_event_date,
            field="expected_training_snapshot_max_event_date",
        )
        try:
            date.fromisoformat(snapshot_date)
        except ValueError as exc:
            raise ValueError(
                "expected_training_snapshot_max_event_date must be an ISO date "
                "(YYYY-MM-DD)"
            ) from exc
    intervals = (timeout_seconds, poll_seconds, request_timeout_seconds)
    if any(not math.isfinite(value) or value <= 0 for value in intervals):
        raise ValueError(
            "timeout, poll interval, and request timeout must be finite and positive"
        )
    if successes_required < 2:
        raise ValueError("successes_required must be at least 2")

    deadline = monotonic() + timeout_seconds
    attempt = 0
    consecutive_successes = 0
    last_reason = "no ready response received"
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"timed out after {timeout_seconds:g}s; last gate result: {last_reason}"
            )
        attempt += 1
        try:
            http_status, payload = fetch(url, min(request_timeout_seconds, remaining))
            passed, last_reason = evaluate_readyz(
                http_status,
                payload,
                **expectations,
            )
        except (HTTPException, OSError, ValueError) as exc:
            passed = False
            payload = None
            last_reason = f"ready endpoint request failed: {type(exc).__name__}: {exc}"

        if passed:
            consecutive_successes += 1
            if consecutive_successes < successes_required:
                last_reason = (
                    f"{last_reason}; confirmation "
                    f"{consecutive_successes}/{successes_required}"
                )
        else:
            consecutive_successes = 0
        emit(f"Deployment gate attempt {attempt}: {last_reason}")
        if passed and consecutive_successes >= successes_required:
            return payload

        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"timed out after {timeout_seconds:g}s; last gate result: {last_reason}"
            )
        sleep(min(poll_seconds, remaining))


def _positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return seconds


def _positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_READYZ_URL)
    parser.add_argument("--expected-bundle-id", required=True)
    parser.add_argument("--expected-release-id", required=True)
    parser.add_argument("--expected-installed-manifest-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-deployed-git-sha", required=True)
    parser.add_argument("--expected-training-source-git-sha", required=True)
    parser.add_argument("--expected-spec", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-no-odds-model-sha256", required=True)
    parser.add_argument("--expected-logistic-model-sha256", required=True)
    parser.add_argument("--expected-training-fights-sha256", required=True)
    parser.add_argument("--expected-training-features-sha256", required=True)
    parser.add_argument("--expected-training-snapshot-max-event-date")
    parser.add_argument("--expected-runtime-lookup-fights-sha256")
    parser.add_argument("--expected-runtime-lookup-features-sha256")
    parser.add_argument("--timeout-seconds", type=_positive_seconds, default=1200.0)
    parser.add_argument("--poll-seconds", type=_positive_seconds, default=15.0)
    parser.add_argument("--request-timeout-seconds", type=_positive_seconds, default=20.0)
    parser.add_argument(
        "--successes-required",
        type=_positive_integer,
        default=2,
        help="consecutive matching ready responses required before success (default: 2)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        wait_for_production_deployment(
            url=args.url,
            expected_bundle_id=args.expected_bundle_id,
            expected_release_id=args.expected_release_id,
            expected_installed_manifest_sha256=args.expected_installed_manifest_sha256,
            expected_source_manifest_sha256=args.expected_source_manifest_sha256,
            expected_deployed_git_sha=args.expected_deployed_git_sha,
            expected_training_source_git_sha=args.expected_training_source_git_sha,
            expected_spec=args.expected_spec,
            expected_model_sha256=args.expected_model_sha256,
            expected_no_odds_model_sha256=args.expected_no_odds_model_sha256,
            expected_logistic_model_sha256=args.expected_logistic_model_sha256,
            expected_training_fights_sha256=args.expected_training_fights_sha256,
            expected_training_features_sha256=args.expected_training_features_sha256,
            expected_training_snapshot_max_event_date=(
                args.expected_training_snapshot_max_event_date
            ),
            expected_runtime_lookup_fights_sha256=(
                args.expected_runtime_lookup_fights_sha256
            ),
            expected_runtime_lookup_features_sha256=(
                args.expected_runtime_lookup_features_sha256
            ),
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
            successes_required=args.successes_required,
        )
    except (TimeoutError, ValueError) as exc:
        print(f"Production deployment verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
