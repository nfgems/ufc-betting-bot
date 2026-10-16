"""Read-only identity withholding and review visibility; never binds a fighter."""
from __future__ import annotations

import csv
import json
import logging
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)


def _text(value: object) -> str:
    text = str(value if value is not None else "").strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "na", "nat"} else text


def _false(value: object) -> bool:
    return value is False or _text(value).casefold() in {"false", "0"}


def _name_key(value: object) -> str:
    # This comparison only withholds an already-pending identity; it cannot bind.
    text = unicodedata.normalize("NFKD", _text(value).casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^\w]+", " ", text).strip()


def _resolved_owner_ids(review: dict) -> set[str]:
    values = review.get("resolved_owner_ids", [])
    if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Identity review resolved_owner_ids must be a list of nonblank strings")
    resolved = {value.strip() for value in values}
    if any(row.get("identity_complete") is False and _text(row.get("internal_owner_id")) in resolved
           for row in review.get("rows", []) if isinstance(row, dict)):
        raise ValueError("Identity review contradicts a pending owner's identity status")
    return resolved


def load_identity_review(path: Path | str | None = None) -> dict | None:
    """Read the exact feed without mutation; absent is None, malformed raises."""
    if path is None:
        from src.config import ROSTERWATCH_IDENTITY_REVIEW_PATH
        path = ROSTERWATCH_IDENTITY_REVIEW_PATH
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError("Identity review must contain a rows list")
    for row in payload["rows"]:
        if not isinstance(row, dict) or not _text(row.get("fighter")):
            raise ValueError("Identity review contains an invalid fighter row")
        if type(row.get("identity_complete")) is not bool:
            raise ValueError("Identity review identity_complete must be boolean")
    _resolved_owner_ids(payload)
    if payload.get("membership_current") is False or payload.get("status") in {"stale", "unavailable"}:
        raise ValueError("Identity review is stale or unavailable")
    return payload


def pending_fighters_for_prediction(
    row, *, review_path: Path | str | None = None, ledger_path: Path | str | None = None,
) -> list[str]:
    """Withhold explicit pending identities, including when predictions are cached.

    The membership ledger is authoritative when present. Review evidence is a
    fallback, never a way to overturn a resolved ledger carrier. An unavailable
    configured file fails closed for this fight. Legacy installations with
    neither file retain their existing certified-identity checks.
    """
    names = {side: _text(row.get(f"fighter_{side}")) for side in ("a", "b")}
    pending = set()
    for side, name in names.items():
        if any(_false(row.get(key)) for key in (
            "identity_complete", "identity_confirmed",
            f"{side}_identity_complete", f"fighter_{side}_identity_complete",
            f"{side}_identity_confirmed", f"fighter_{side}_identity_confirmed",
        )) or any(_text(row.get(key)).casefold() in {
            "pending_external_carrier", "pending", "unconfirmed", "identity_unconfirmed",
        } for key in ("carrier_resolution_status", f"{side}_carrier_resolution_status", f"fighter_{side}_carrier_resolution_status")):
            pending.add(name)
    if review_path is None:
        from src.config import ROSTERWATCH_IDENTITY_REVIEW_PATH
        review_path = ROSTERWATCH_IDENTITY_REVIEW_PATH
    review_path = Path(review_path)
    ledger_path = Path(ledger_path) if ledger_path is not None else review_path.with_name("rosterwatch_membership_ledger.csv")
    try:
        if ledger_path.exists():
            with ledger_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if not {"rw_fighter_raw", "carrier_resolution_status"}.issubset(reader.fieldnames or ()):
                    raise ValueError("Membership ledger lacks identity columns")
                records = list(reader)
            if not records or any(not _text(record.get("rw_fighter_raw")) or _text(record.get("carrier_resolution_status")) not in {"resolved", "pending_external_carrier"} for record in records):
                raise ValueError("Membership ledger identity rows are incomplete")
            unresolved = [record for record in records if _text(record.get("carrier_resolution_status")) != "resolved"]
            pending_names = {_name_key(record.get("rw_fighter_raw")) for record in unresolved}
        else:
            review = load_identity_review(review_path)
            pending_names = {_name_key(record.get("fighter")) for record in (review or {}).get("rows", []) if record["identity_complete"] is False}
        pending.update(name for name in names.values() if _name_key(name) in pending_names)
    except (OSError, ValueError, csv.Error):
        # Failed identity-state reads cannot make a cached prediction executable.
        pending.update(names.values())
    return sorted(name for name in pending if name)


def prediction_identities_confirmed(row, *, review_path: Path | str | None = None) -> bool:
    """Positive evidence for recovery; absence of a pending entry is insufficient."""
    if pending_fighters_for_prediction(row, review_path=review_path):
        return False
    if row.get("identity_complete") is True or row.get("identity_confirmed") is True:
        return True
    if review_path is None:
        from src.config import ROSTERWATCH_IDENTITY_REVIEW_PATH
        review_path = ROSTERWATCH_IDENTITY_REVIEW_PATH
    ledger_path = Path(review_path).with_name("rosterwatch_membership_ledger.csv")
    try:
        if ledger_path.exists():
            with ledger_path.open(newline="", encoding="utf-8") as stream:
                records = list(csv.DictReader(stream))
            resolved_names = {_name_key(record.get("rw_fighter_raw")) for record in records if record.get("carrier_resolution_status") == "resolved"}
        else:
            review = load_identity_review(review_path)
            resolved_names = {_name_key(record.get("fighter")) for record in (review or {}).get("rows", []) if record["identity_complete"] is True}
        return all(_name_key(row.get(f"fighter_{side}")) in resolved_names for side in ("a", "b"))
    except (OSError, ValueError, csv.Error):
        return False


def _visibility_issues(review: dict) -> dict[str, str]:
    from src.config import PENDING_IDENTITY_MAX_AGE_DAYS, PENDING_IDENTITY_MAX_COUNT
    if not isinstance(review, dict) or not isinstance(review.get("rows"), list) or any(not isinstance(row, dict) or type(row.get("identity_complete")) is not bool or not _text(row.get("fighter")) for row in review["rows"]):
        raise ValueError("Malformed identity review cannot update or recover incidents")
    _resolved_owner_ids(review)
    if review.get("membership_current") is False or review.get("status") in {"stale", "unavailable"}:
        raise ValueError("A stale identity review cannot update or recover incidents")
    thresholds = review.get("thresholds") or {}
    age_limit = int(thresholds.get("pending_age_days", PENDING_IDENTITY_MAX_AGE_DAYS))
    count_limit = int(thresholds.get("pending_count", PENDING_IDENTITY_MAX_COUNT))
    issues = {}
    pending = [row for row in review.get("rows", []) if row.get("identity_complete") is False]
    for row in pending:
        fighter = _text(row.get("fighter"))
        owner = _text(row.get("internal_owner_id")) or f"{_name_key(fighter)}:{_text(row.get('date_of_birth'))}"
        clears = _text(row.get("what_clears")) or "An evidence-backed identity link clears this issue."
        if row.get("booked") is True:
            issues[f"rosterwatch-identity:booked:{owner}"] = (
                f"UFC identity review: {fighter} vs {_text(row.get('next_opp')) or 'unconfirmed opponent'} "
                f"on {_text(row.get('next_date')) or 'unconfirmed date'}: booked fighter identity unconfirmed; bets withheld. {clears}"
            )
        try:
            days = int(row.get("days_pending") or 0)
        except (ValueError, TypeError):
            days = 0
        if days > age_limit:
            issues[f"rosterwatch-identity:aged:{owner}"] = (
                f"UFC identity review: {fighter} pending for {days} days "
                f"(threshold {age_limit}); identity unconfirmed. {clears}"
            )
    if len(pending) > count_limit:
        issues["rosterwatch-identity:pending-count"] = (
            f"UFC identity review: {len(pending)} pending fighters exceeds threshold {count_limit}: "
            + ", ".join(_text(row.get("fighter")) for row in pending)
            + ". Evidence-backed identity links reducing the pending count clear this issue."
        )
    return issues


def emit_identity_review_alerts(review: dict, previous_review: dict | None = None) -> None:
    """Mirror only ERROR conditions and explicitly recover prior cleared ones."""
    if not isinstance(review, dict) or not isinstance(review.get("rows"), list):
        raise ValueError("A complete identity review is required before updating incidents")
    current = _visibility_issues(review)
    valid_previous = previous_review is not None
    try:
        previous = _visibility_issues(previous_review) if valid_previous else {}
    except (ValueError, TypeError, AttributeError):
        logger.warning("Previous identity review is invalid; current issues remain visible and no prior incidents are recovered")
        previous = {}
        valid_previous = False
    for key, message in current.items():
        logger.error(message, extra={"alert_incident_key": key})
    linked_owners = {
        _text(row.get("internal_owner_id")) or f"{_name_key(row.get('fighter'))}:{_text(row.get('date_of_birth'))}"
        for row in review["rows"] if row["identity_complete"] is True
    } | _resolved_owner_ids(review)
    recovered = set(previous) - set(current)
    if valid_previous:
        # A booking can disappear while its unresolved incident remains active.
        # Positive links clear that older issue even if the immediately previous
        # feed no longer lists it. The durable handler ignores inactive keys.
        recovered.update(f"rosterwatch-identity:{condition}:{owner}"
                         for owner in linked_owners for condition in ("booked", "aged"))
    for key in sorted(recovered):
        if key != "rosterwatch-identity:pending-count" and key.split(":", 2)[-1] not in linked_owners:
            continue
        logger.info("UFC identity review condition cleared by a confirmed identity link or resolved pending-count threshold: %s", key,
                    extra={"alert_recovered_incident_keys": [key]})
