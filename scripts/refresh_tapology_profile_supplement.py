"""Hosted Tapology profile supplement refresh and access probe.

This is intended for cloud schedulers such as GitHub Actions. It verifies that
Tapology is reachable from the hosted runtime before attempting a refresh, then
updates the repository's supplemental physical-profile artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlparse

import pandas as pd


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
from src.data.name_utils import normalize_cross_source_name, same_person_name
from src.data.ufc_active_roster import (
    ACTIVE_ROSTER_SYNC_GENERATION_COLUMN,
    OFFICIAL_ACTIVE_ROSTER_PATH,
    _apply_profile_status_eligibility,
    _explicit_boolean,
    _roster_identity_keys,
    _roster_rows_same_identity,
    sync_official_active_roster,
)


logger = logging.getLogger(__name__)

DEFAULT_SCRAPED_FIGHTERS_PATH = RAW_DATA_DIR / "ufc_fighters_scraped.csv"
DEFAULT_PROFILE_SUPPLEMENT_PATH = RAW_DATA_DIR / "ufc_fighters_profile_supplement.csv"
DEFAULT_PROBE_NAME = "Feng Pengchao"
DEFAULT_PROBE_URL = (
    "https://www.tapology.com/fightcenter/fighters/230675-feng-peng-zhao-winged-tiger"
)
PROFILE_FIELDS = ("height", "reach", "weight", "dob")
_HOSTED_ORIGINAL_PROFILE_STATUS_COLUMN = "_hosted_original_profile_status"
_HOSTED_CURRENT_VERIFICATION_BLOCK_COLUMN = (
    "_hosted_current_verification_blocked"
)
_EXPLICIT_INACTIVE_STATUS_KEYS = frozenset(
    {
        "not fighting",
        "inactive",
        "retired",
        "cut",
        "cut from ufc",
        "released",
        "released from ufc",
    }
)


def _coverage_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def _optional_cell_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "nat"} else text


def _normalized_status_key(value: object) -> str:
    return " ".join(
        normalize_cross_source_name(_optional_cell_text(value))
        .replace("_", " ")
        .split()
    )


def _row_has_explicit_inactive_evidence(row: pd.Series | dict[str, object]) -> bool:
    for column in ("profile_status", _HOSTED_ORIGINAL_PROFILE_STATUS_COLUMN):
        status_key = _normalized_status_key(row.get(column))
        if status_key in _EXPLICIT_INACTIVE_STATUS_KEYS:
            return True

    reason_tokens = set(
        _normalized_status_key(row.get("active_roster_status_reason")).split()
    )
    return bool(reason_tokens & {"inactive", "retired", "cut", "released"})


def _row_has_power_slap_evidence(row: pd.Series | dict[str, object]) -> bool:
    evidence = " ".join(
        _optional_cell_text(row.get(column))
        for column in (
            "combat_sport",
            "combat_sport_reason",
            "combat_sport_profile_url",
            "active_roster_status_reason",
        )
    ).casefold()
    compact = "".join(character for character in evidence if character.isalnum())
    return "powerslap" in compact


def _row_has_explicit_untrusted_identity_evidence(
    row: pd.Series | dict[str, object],
) -> bool:
    identity_status = _normalized_status_key(
        row.get("official_url_identity_status")
    )
    status_reason = _normalized_status_key(
        row.get("active_roster_status_reason")
    )
    return (
        identity_status in {"mismatch", "test profile"}
        or _explicit_boolean(row.get("official_url_identity_valid")) is False
        or status_reason == "untrusted official profile identity"
    )


def _row_has_hard_current_verification_block(
    row: pd.Series | dict[str, object],
) -> bool:
    if _optional_cell_text(row.get("active_roster_status_reason")) == "rosterwatch_authoritative_current":
        return not _coverage_enabled(row.get("coverage_eligible"))
    return (
        _coverage_enabled(row.get(_HOSTED_CURRENT_VERIFICATION_BLOCK_COLUMN))
        or _row_has_explicit_inactive_evidence(row)
        or _row_has_power_slap_evidence(row)
        or _row_has_explicit_untrusted_identity_evidence(row)
    )


def _profile_status_is_blank_or_unknown(
    row: pd.Series | dict[str, object],
) -> bool:
    status_key = _normalized_status_key(row.get("profile_status"))
    return status_key in {"", "status unknown"}


def _row_is_unverified_hollow_zero_record(
    row: pd.Series | dict[str, object],
) -> bool:
    """Identify archive-only rows with no evidence of an MMA fighter profile."""
    return (
        _normalized_status_key(row.get("profile_status")) == "active"
        and _normalized_status_key(row.get("record"))
        in {"0 0 0", "0 0 0 w l d"}
        and _normalized_status_key(row.get("combat_sport")) == "mma"
        and not _optional_cell_text(row.get("combat_sport_reason"))
        and all(
            not _optional_cell_text(row.get(column))
            for column in (
                "profile_record",
                "birthplace",
                "height",
                "reach",
                "weight",
                "ufcstats_name",
                "ufcstats_url",
            )
        )
    )


def _verified_candidate_row_count(candidate_df: pd.DataFrame) -> int:
    verified_rows = 0
    for _, row in candidate_df.iterrows():
        if (
            _row_has_hard_current_verification_block(row)
            or not _coverage_enabled(row.get("coverage_eligible"))
        ):
            continue
        if _optional_cell_text(row.get("active_roster_status_reason")) == "rosterwatch_authoritative_current":
            verified_rows += 1
            continue
        status_key = _normalized_status_key(row.get("profile_status"))
        if status_key in {"active", "current"} or _coverage_enabled(
            row.get("active_roster_current_verified")
        ):
            verified_rows += 1
    return verified_rows


def _load_candidate_status_cache(path: Path) -> tuple[pd.DataFrame, str]:
    """Load validated cached identities for positive hydration and hard blocks."""
    if not path.exists():
        return pd.DataFrame(), "missing"
    try:
        cached = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Could not read cached active-roster statuses from %s: %s", path, exc)
        return pd.DataFrame(), "unreadable"
    if cached.empty:
        return pd.DataFrame(), "empty"

    generation_state = "legacy_unstamped"
    if ACTIVE_ROSTER_SYNC_GENERATION_COLUMN in cached.columns:
        generations = [
            str(value).strip()
            for value in cached[ACTIVE_ROSTER_SYNC_GENERATION_COLUMN].fillna("")
            if str(value).strip() and str(value).strip().casefold() != "nan"
        ]
        if generations:
            if len(generations) != len(cached) or len(set(generations)) != 1:
                logger.warning(
                    "Refusing mixed/partially stamped cached active-roster statuses from %s",
                    path,
                )
                return pd.DataFrame(), "invalid_generation"
            generation_state = "generation_valid"

    original_hard_blocks = cached.apply(
        _row_has_hard_current_verification_block,
        axis=1,
    )
    original_reasons = (
        cached["active_roster_status_reason"].copy()
        if "active_roster_status_reason" in cached.columns
        else pd.Series("", index=cached.index, dtype="object")
    )
    normalized = _apply_profile_status_eligibility(cached)
    normalized[_HOSTED_CURRENT_VERIFICATION_BLOCK_COLUMN] = (
        original_hard_blocks.astype(bool)
    )
    for index in normalized.index[original_hard_blocks]:
        original_reason = _optional_cell_text(original_reasons.at[index])
        if original_reason:
            normalized.at[index, "active_roster_status_reason"] = (
                original_reason
            )
    return normalized, generation_state


def _build_bounded_active_roster_candidates(
    *,
    active_roster_path: Path,
    temporary_output_path: Path,
    accepted_membership: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object]]:
    """Nominate only an explicitly pinned accepted Roster Watch generation."""
    from src.data.provider_membership import load_accepted_rosterwatch_membership
    from src.data.io_utils import write_csv_atomically

    if accepted_membership is None:
        raise ValueError("An explicit accepted Roster Watch membership pin is required")
    if Path(accepted_membership["path"]).resolve() != Path(active_roster_path).resolve():
        raise ValueError("Accepted membership path differs from the nominated roster")
    frame = load_accepted_rosterwatch_membership(
        active_roster_path, expected_sha256=accepted_membership["sha256"],
        expected_generation=accepted_membership["generation"],
    )
    write_csv_atomically(frame, temporary_output_path)
    return temporary_output_path, {
        "action": "accepted_rosterwatch_membership",
        "rows": len(frame),
        "verified_candidate_rows": int(frame["coverage_eligible"].sum()),
        "pending_identity_rows": int((~frame["coverage_eligible"]).sum()),
        "accepted_membership_sha256": accepted_membership["sha256"],
        "accepted_membership_generation": accepted_membership["generation"],
        "output_path": str(temporary_output_path),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _present(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text and text.lower() not in {"nan", "none", "n/a", "??", "-", "--"})


def _probe_failure_kind(errors: list[str]) -> str:
    detail = " ".join(errors).casefold()
    if (
        "cloudflare" in detail
        or "status 403" in detail
        or "status 429" in detail
        or "status 451" in detail
        or "rate limit" in detail
        or "too many requests" in detail
        or "blocked" in detail
        or "runtime_block" in detail
    ):
        return "hosted_egress_blocked"
    if "timed out" in detail or "connection" in detail:
        return "network_unavailable"
    if "discovery" in detail or "no tapology profile candidates" in detail:
        return "discovery_failed"
    return "profile_fetch_or_parse_failed"


def _profile_refresh_failure_kind(refresh_summary: dict[str, object]) -> str:
    errors: list[str] = []
    details = refresh_summary.get("source_error_details") or []
    if isinstance(details, list):
        for detail in details:
            if isinstance(detail, dict):
                error = str(detail.get("error") or "").strip()
            else:
                error = str(detail or "").strip()
            if error:
                errors.append(error)
    return _probe_failure_kind(errors)


def probe_tapology_access(
    *,
    fighter_name: str = DEFAULT_PROBE_NAME,
    fighter_url: str = DEFAULT_PROBE_URL,
) -> dict[str, object]:
    clear_fallback_cache(preserve_environment_blocks=False)
    discovery_errors: list[str] = []
    discovery: dict[str, object] = {}
    try:
        candidates = search_tapology_candidates(
            fighter_name,
            limit=3,
            diagnostics=discovery,
        )
    except Exception as exc:
        candidates = []
        discovery_errors.append(
            f"Tapology discovery raised {type(exc).__name__}: {exc}"
        )
    if not bool(discovery.get("healthy")):
        discovery_errors.append(
            "Tapology discovery was unavailable: "
            f"{discovery.get('attempts') or 'no healthy search channel'}"
        )
    if not candidates:
        discovery_errors.append("no Tapology profile candidates found")

    if fighter_url:
        expected = urlparse(fighter_url)
        expected_identity = (expected.netloc.casefold(), expected.path.rstrip("/"))
        discovered_identities = {
            (urlparse(candidate).netloc.casefold(), urlparse(candidate).path.rstrip("/"))
            for candidate in candidates
        }
        if expected_identity not in discovered_identities:
            discovery_errors.append(
                f"Tapology discovery did not return the known profile URL {fighter_url}"
            )

    # Discovery and profile fetching are independent health dimensions. Always
    # test the supplied canary URL directly, even when search is unavailable,
    # so a search-endpoint block cannot hide a healthy profile/parser path.
    profile_probe_urls = list(
        dict.fromkeys([url for url in (fighter_url, *candidates) if url])
    )
    profile_errors: list[str] = []
    parsed_profile: dict[str, object] | None = None
    for candidate_url in profile_probe_urls:
        try:
            profile = scrape_tapology_profile(candidate_url)
        except TapologyRequestError as exc:
            profile_errors.append(f"{candidate_url}: {exc}")
            continue
        except Exception as exc:
            profile_errors.append(
                f"{candidate_url}: {type(exc).__name__}: {exc}"
            )
            continue

        parsed_name = str(profile.get("name") or "").strip()
        fields = {
            field: profile.get(field)
            for field in PROFILE_FIELDS
            if _present(profile.get(field))
        }
        if not parsed_name:
            profile_errors.append(
                f"{candidate_url}: profile parsed without a fighter identity"
            )
            continue
        if not same_person_name(fighter_name, parsed_name):
            profile_errors.append(
                f"{candidate_url}: parsed profile name {parsed_name!r} "
                f"did not match {fighter_name!r}"
            )
            continue
        if not fields:
            profile_errors.append(
                f"{candidate_url}: profile parsed but no physical fields were present"
            )
            continue
        parsed_profile = {
            "candidate_url": candidate_url,
            "parsed_name": parsed_name,
            "fields": fields,
        }
        break

    if not profile_probe_urls:
        profile_errors.append("no Tapology profile URL was available to probe")

    discovery_ok = not discovery_errors
    profile_ok = parsed_profile is not None
    result: dict[str, object] = {
        "ok": discovery_ok and profile_ok,
        "fighter_name": fighter_name,
        "candidate_urls": candidates,
        "profile_probe_urls": profile_probe_urls,
        "discovery": discovery,
        "discovery_ok": discovery_ok,
        "profile_ok": profile_ok,
    }
    if parsed_profile is not None:
        result.update(parsed_profile)
    if not result["ok"]:
        errors = [*discovery_errors, *profile_errors]
        result["failure_kind"] = _probe_failure_kind(errors)
        result["errors"] = errors
    return result


def _hosted_candidate_rotation_index() -> int:
    try:
        run_number = max(int(os.getenv("GITHUB_RUN_NUMBER", "1")), 1)
    except ValueError:
        run_number = 1
    return run_number - 1


_PRIORITY_NAME_COLUMNS = (
    "name",
    "official_name",
    "ufcstats_name",
    "profile_name",
    "slug_name",
)


def _load_upcoming_fighter_names() -> tuple[list[str], int]:
    """Fetch the actual UFC cards used by production and return their fighters."""
    from src.data.live_monitor import collect_upcoming_fight_contexts

    contexts = collect_upcoming_fight_contexts()
    names: list[str] = []
    for context in contexts:
        for field in ("fighter_a", "fighter_b"):
            value = str(context.get(field) or "").strip()
            if value and value not in names:
                names.append(value)
    return names, len(contexts)


def _unambiguous_current_card_candidate_indices(
    candidate_df: pd.DataFrame,
    upcoming_names: list[str],
) -> tuple[set[object], dict[str, int]]:
    """Return one-to-one roster matches for current-card fighter identities.

    Matching delegates to the active-roster identity contract so only official
    names, UFCStats names, and trusted aliases participate. A card identity
    matching multiple roster rows, or a roster row matching multiple distinct
    card identities, is deliberately left unresolved.
    """
    upcoming_by_key: dict[str, str] = {}
    for name in upcoming_names:
        key = normalize_cross_source_name(name)
        if key:
            upcoming_by_key.setdefault(key, str(name).strip())

    matches_by_upcoming: dict[str, list[object]] = {}
    for key, upcoming_name in upcoming_by_key.items():
        upcoming_row = {"official_name": upcoming_name}
        matches_by_upcoming[key] = [
            index
            for index, candidate_row in candidate_df.iterrows()
            if _roster_rows_same_identity(candidate_row, upcoming_row)
        ]

    uniquely_matched_by_row: dict[object, list[str]] = {}
    for key, row_indices in matches_by_upcoming.items():
        if len(row_indices) == 1:
            uniquely_matched_by_row.setdefault(row_indices[0], []).append(key)

    matched_indices = {
        index
        for index, keys in uniquely_matched_by_row.items()
        if len(keys) == 1
    }
    return matched_indices, {
        "unambiguous_current_card_matches": int(len(matched_indices)),
        "ambiguous_current_card_identities": int(
            sum(len(indices) > 1 for indices in matches_by_upcoming.values())
        ),
        "ambiguous_candidate_rows": int(
            sum(len(keys) > 1 for keys in uniquely_matched_by_row.values())
        ),
    }


def _prioritize_current_card_candidates(
    candidate_source_csv: Path,
    *,
    output_path: Path,
) -> tuple[Path, dict[str, object]]:
    """Join live UFC card identities into the candidate source without mutating it."""
    if not candidate_source_csv.exists():
        return candidate_source_csv, {
            "action": "skip",
            "reason": "candidate source missing",
            "source_path": str(candidate_source_csv),
        }
    try:
        upcoming_names, fight_count = _load_upcoming_fighter_names()
    except Exception as exc:
        logger.warning("Could not load current UFC cards for profile priority: %s", exc)
        return candidate_source_csv, {
            "action": "fallback_to_rotation",
            "reason": str(exc),
            "source_path": str(candidate_source_csv),
        }
    upcoming_keys = {
        normalize_cross_source_name(name) for name in upcoming_names if str(name).strip()
    }
    if not upcoming_keys:
        return candidate_source_csv, {
            "action": "fallback_to_rotation",
            "reason": "no current UFC fight contexts",
            "fight_context_rows": int(fight_count),
        }

    candidate_df = pd.read_csv(candidate_source_csv)
    if candidate_df.empty:
        return candidate_source_csv, {
            "action": "fallback_to_rotation",
            "reason": "candidate source empty",
            "fight_context_rows": int(fight_count),
        }

    current_verified_indices, identity_summary = (
        _unambiguous_current_card_candidate_indices(
            candidate_df,
            upcoming_names,
        )
    )
    # This verifier is deliberately ephemeral: discard any incoming value and
    # reproduce it solely from this run's live card identities in the temporary
    # candidate artifact.
    candidate_df["active_roster_current_verified"] = False
    current_verified_rows = 0
    hollow_quarantined_names: list[str] = []
    for index, row in candidate_df.iterrows():
        if _optional_cell_text(row.get("active_roster_status_reason")) == "rosterwatch_authoritative_current":
            # Cards may adjust priority, never grant or revoke membership/identity.
            continue
        if _row_has_hard_current_verification_block(row):
            candidate_df.at[index, "coverage_eligible"] = False
            continue
        if (
            _row_is_unverified_hollow_zero_record(row)
            and index not in current_verified_indices
        ):
            candidate_df.at[index, "coverage_eligible"] = False
            candidate_df.at[index, "active_roster_status_reason"] = (
                "unverified_hollow_zero_record_not_on_current_card"
            )
            fighter_name = _optional_cell_text(row.get("official_name"))
            if not fighter_name:
                fighter_name = _optional_cell_text(row.get("name"))
            if fighter_name:
                hollow_quarantined_names.append(fighter_name)
            continue
        if (
            index not in current_verified_indices
            or not _profile_status_is_blank_or_unknown(row)
        ):
            continue
        candidate_df.at[index, "active_roster_current_verified"] = True
        candidate_df.at[index, "coverage_eligible"] = True
        candidate_df.at[index, "active_roster_status_reason"] = (
            "verified_from_current_ufc_card"
        )
        current_verified_rows += 1

    def _row_is_upcoming(row: pd.Series) -> bool:
        names: list[str] = []
        for column in _PRIORITY_NAME_COLUMNS:
            if column in row.index:
                names.append(str(row.get(column) or ""))
        for column in ("alternate_slug_names", "search_names"):
            if column in row.index:
                names.extend(str(row.get(column) or "").split("|"))
        return any(normalize_cross_source_name(name) in upcoming_keys for name in names if name.strip())

    upcoming_mask = candidate_df.apply(_row_is_upcoming, axis=1)
    existing_priority = pd.Series(0, index=candidate_df.index, dtype="int64")
    if "profile_refresh_priority" in candidate_df.columns:
        existing_priority = pd.to_numeric(
            candidate_df["profile_refresh_priority"], errors="coerce"
        ).fillna(0).clip(lower=0).astype(int)
    candidate_df["profile_refresh_priority"] = existing_priority.where(
        ~upcoming_mask,
        existing_priority.clip(lower=100),
    )
    candidate_df = candidate_df.drop(
        columns=[_HOSTED_ORIGINAL_PROFILE_STATUS_COLUMN],
        errors="ignore",
    )
    verified_candidate_rows = _verified_candidate_row_count(candidate_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(output_path, index=False)
    return output_path, {
        "action": "prioritized_current_ufc_cards",
        "fight_context_rows": int(fight_count),
        "upcoming_fighter_names": int(len(upcoming_keys)),
        "matched_candidate_rows": int(upcoming_mask.sum()),
        "current_card_verified_rows": int(current_verified_rows),
        "hollow_zero_record_quarantined_rows": int(
            len(hollow_quarantined_names)
        ),
        "hollow_zero_record_quarantined_names": hollow_quarantined_names,
        "verified_candidate_rows": int(verified_candidate_rows),
        **identity_summary,
        "output_path": str(output_path),
    }


def run_hosted_refresh(args: argparse.Namespace) -> dict[str, object]:
    if getattr(args, "retained_provider_observations", None) is not None:
        # Offline replay is an explicit processing input, not a failed-fetch
        # fallback. Validate membership before considering any source operation.
        accepted = getattr(args, "accepted_membership", None)
        if accepted is None:
            raise ValueError("Retained provider processing requires accepted membership")
        with TemporaryDirectory(prefix="retained_tapology_candidates_") as tmp_dir:
            _, roster_summary = _build_bounded_active_roster_candidates(
                active_roster_path=Path(accepted["path"]),
                temporary_output_path=Path(tmp_dir) / "candidates.csv",
                accepted_membership=accepted,
            )
            summary = run_profile_supplement_refresh(
                scraped_fighters_path=args.scraped_fighters_path,
                output_path=args.output, sources=["tapology"], limit=args.limit,
                accepted_membership=accepted,
                retained_provider_observations=args.retained_provider_observations,
                available_before=args.available_before,
                observation_history_root=args.observation_history_root,
                profile_integrity_decisions_path=args.profile_integrity_decisions_path,
            )
        refused = bool(summary["processing_error_count"] or summary["identity_rejection_count"])
        return {"action": "processing_refused" if refused else "refreshed_retained",
                "active_roster": roster_summary, "profile_supplement_refresh": summary,
                "source_requests": 0, "progress_state": "processing_refused" if refused else "complete"}
    if not getattr(args, "probe_only", False):
        # Collection uses only accepted members and their independently reviewed
        # bound response routes. A separate access probe cannot authorize it.
        accepted = getattr(args, "accepted_membership", None)
        required = (accepted, getattr(args, "available_before", None),
                    getattr(args, "observation_history_root", None),
                    getattr(args, "profile_integrity_decisions_path", None))
        if args.limit is not None and int(args.limit) <= 0:
            return {"action": "configuration_error", "progress_state": "configuration_error",
                    "source_requests": 0, "source_health_ok": False, "error": "--limit must be a positive integer"}
        if any(value is None for value in required):
            return {"action": "processing_refused", "progress_state": "processing_refused",
                    "source_requests": 0, "source_health_ok": False,
                    "error": "Ordinary provider collection requires original accepted membership, reviewed decisions, cutoff, evidence destination, and a positive limit"}
        with TemporaryDirectory(prefix="ordinary_tapology_candidates_") as tmp_dir:
            nominated, roster_summary = _build_bounded_active_roster_candidates(
                active_roster_path=Path(accepted["path"]), accepted_membership=accepted,
                temporary_output_path=Path(tmp_dir) / "accepted-candidates.csv",
            )
            # Optional supplied card names are priority hints only. No card
            # acquisition is hidden inside ordinary provider collection.
            names = getattr(args, "current_card_names", ()) or ()
            keys = {normalize_cross_source_name(name) for name in names}
            nominated_rows = pd.read_csv(nominated, dtype=str, keep_default_na=False)
            priorities = {str(row["stable_ufcstats_id"]): 100 for row in nominated_rows.to_dict("records")
                          if str(row.get("stable_ufcstats_id", "")) and any(normalize_cross_source_name(row.get(column, "")) in keys
                                 for column in _PRIORITY_NAME_COLUMNS)}
            summary = run_profile_supplement_refresh(
                scraped_fighters_path=args.scraped_fighters_path,
                output_path=args.output, sources=["tapology"], limit=args.limit,
                accepted_membership=accepted, candidate_priorities=priorities,
                available_before=args.available_before,
                observation_history_root=args.observation_history_root,
                profile_integrity_decisions_path=args.profile_integrity_decisions_path,
                provider_transport=getattr(args, "provider_transport", None),
                provider_response_format=getattr(args, "provider_response_format", "html"),
                candidate_offset=max(int(getattr(args, "candidate_offset", 0) or 0), 0),
                candidate_rotation_index=(None if getattr(args, "candidate_offset", None) is not None
                                          else _hosted_candidate_rotation_index()),
            )
        refused = bool(summary.get("processing_error_count") or summary.get("identity_rejection_count"))
        source_failed = bool(summary.get("source_error_count"))
        empty = int(summary.get("candidate_universe_rows") or 0) <= 0
        recoverable = int(summary.get("recoverable_candidate_rows") or 0)
        zero_attempts = recoverable > 0 and int(summary.get("attempted_rows") or 0) <= 0
        configuration_error = empty or zero_attempts
        progress = ("processing_refused" if refused else "configuration_error" if configuration_error
                    else "source_error" if source_failed else "recovered" if summary.get("recovered_rows")
                    else "no_fields_available" if recoverable else "no_recoverable_candidates")
        result = {"action": "processing_refused" if refused else "configuration_error" if configuration_error else "refreshed",
                "active_roster": roster_summary, "profile_supplement_refresh": summary,
                "candidate_priority": {"accepted_priority_ids": sorted(priorities)},
                "source_health_ok": not (refused or source_failed or configuration_error), "progress_state": progress}
        if source_failed:
            result["failure_kind"] = _profile_refresh_failure_kind(summary)
        if configuration_error:
            result["error"] = ("Tapology candidate source produced zero eligible fighter rows; refusing to report a healthy no-op"
                               if empty else "Tapology candidate source had recoverable rows but selected zero attempts; refusing to report a healthy no-op")
        return result
    if args.limit is not None and int(args.limit) <= 0:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "action": "configuration_error",
            "progress_state": "configuration_error",
            "source_health_ok": False,
            "error": "--limit must be a positive integer",
        }
    probe = probe_tapology_access(fighter_name=args.probe_name, fighter_url=args.probe_url)
    if not probe["ok"]:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "action": "access_probe_failed",
            "tapology_probe": probe,
            "progress_state": "source_error",
            "source_health_ok": False,
        }

    if args.probe_only:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "action": "probe_only",
            "tapology_probe": probe,
        }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scraped-fighters-path", type=Path, default=DEFAULT_SCRAPED_FIGHTERS_PATH)
    parser.add_argument("--active-roster-path", type=Path, default=OFFICIAL_ACTIVE_ROSTER_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROFILE_SUPPLEMENT_PATH)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--candidate-offset",
        type=int,
        default=None,
        help=(
            "Rotate the recoverable non-upcoming backlog before applying --limit. "
            "When omitted, hosted runs derive a rotation index from the GitHub run "
            "number and multiply it by the actual number of backlog slots."
        ),
    )
    parser.add_argument(
        "--require-source-health",
        action="store_true",
        help=(
            "Exit non-zero when candidate refreshes hit source access/parser errors. "
            "A healthy source with no additional fields remains a diagnostic, not a failure."
        ),
    )
    parser.add_argument("--sync-active-roster", action="store_true")
    parser.add_argument(
        "--full-active-roster-sync",
        action="store_true",
        help=(
            "Deprecated compatibility flag retained for callers; hosted candidate "
            "nomination requires a pinned accepted Roster Watch generation."
        ),
    )
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-name", default=DEFAULT_PROBE_NAME)
    parser.add_argument("--probe-url", default=DEFAULT_PROBE_URL)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--retained-provider-observations", type=Path)
    parser.add_argument("--accepted-membership", type=Path)
    parser.add_argument("--available-before")
    parser.add_argument("--observation-history-root", type=Path)
    parser.add_argument("--profile-integrity-decisions-path", type=Path)
    args = parser.parse_args()
    for key in ("retained_provider_observations", "accepted_membership"):
        path = getattr(args, key)
        if path is not None:
            setattr(args, key, json.loads(path.read_text(encoding="utf-8")))
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args()
    summary = run_hosted_refresh(args)
    if args.json:
        print(json.dumps(summary, indent=2, default=_json_default))
    else:
        print(summary)
    if summary.get("progress_state") == "processing_refused":
        return 4
    if summary.get("action") == "access_probe_failed":
        logger.error(
            "Tapology hosted access probe failed (%s)",
            summary.get("tapology_probe", {}).get("failure_kind", "unknown"),
        )
        return 2
    if summary.get("progress_state") == "configuration_error":
        logger.error("Tapology refresh configuration error: %s", summary.get("error"))
        return 4
    if not summary.get("source_health_ok", True):
        logger.error(
            "Tapology refresh lost source access or encountered parser errors; "
            "the hosted workflow must not report the source as healthy"
        )
        return 3
    if summary.get("progress_state") == "no_fields_available":
        logger.warning(
            "Tapology access and parsing were healthy, but no candidate exposed a missing field"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
