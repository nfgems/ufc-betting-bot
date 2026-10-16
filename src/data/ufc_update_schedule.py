"""Durable calendar state for weekly UFC updates and monthly candidates.

This module deliberately does not scrape data, train models, or publish any
artifacts.  It is a small, fail-closed state machine that an integration layer
can use to claim and complete the owner-approved calendar slots:

* Tuesday at 11:00 America/New_York for the weekly data update.
* Wednesday at 11:00 in the same ISO week only when Tuesday failed (or the
  Tuesday slot was missed while the service was unavailable).
* One monthly candidate sourced from the first accepted weekly update in an
  eligible UTC month.  A failed Tuesday candidate may be retried candidate-only
  on Wednesday; a Wednesday candidate has no further retry.

Claims are written before work begins.  A process restart therefore cannot
silently claim the same slot twice.  An interrupted claim remains visible in
the state file and must be explicitly completed after the caller reconciles the
external job outcome; this module never guesses that an in-flight write failed.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.config import (
    UFC_UPDATE_CLAIM_STALE_AFTER_SECONDS,
    UFC_UPDATE_SCHEDULE_LOCAL_HOUR,
    UFC_UPDATE_SCHEDULE_LOCAL_MINUTE,
    UFC_UPDATE_SCHEDULE_STATE_PATH,
    UFC_UPDATE_SCHEDULE_TIMEZONE,
    UFC_UPDATE_TUESDAY_ISO_WEEKDAY,
    UFC_UPDATE_WEDNESDAY_ISO_WEEKDAY,
    UFC_V7_PROMOTION_MONTH_UTC,
)
from src.data.io_utils import write_json_atomically

STATE_SCHEMA_VERSION = 1

ACTION_NONE = "none"
ACTION_DATA_UPDATE = "data_update"
ACTION_MONTHLY_CANDIDATE = "monthly_candidate"

SLOT_TUESDAY = "tuesday"
SLOT_WEDNESDAY = "wednesday"
_SLOTS = (SLOT_TUESDAY, SLOT_WEDNESDAY)

STATUS_CLAIMED = "claimed"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
_ATTEMPT_STATUSES = frozenset({STATUS_CLAIMED, STATUS_SUCCEEDED, STATUS_FAILED})

_ISO_WEEK_RE = re.compile(r"^(?P<year>[0-9]{4})-W(?P<week>[0-9]{2})$")
_UTC_MONTH_RE = re.compile(r"^(?P<year>[0-9]{4})-(?P<month>[0-9]{2})$")


class UFCUpdateScheduleError(RuntimeError):
    """Base class for durable schedule failures."""


class UFCUpdateScheduleStateError(UFCUpdateScheduleError):
    """Raised when persisted state is unreadable or violates its contract."""


class UFCUpdateScheduleTransitionError(UFCUpdateScheduleError):
    """Raised when a caller requests an invalid or conflicting transition."""


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _utc_iso(value: datetime) -> str:
    return _aware_utc(value, field="datetime").isoformat()


def _parse_utc_iso(value: object, *, field: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise UFCUpdateScheduleStateError(f"{field} is missing")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise UFCUpdateScheduleStateError(
            f"{field} is not an ISO datetime: {text!r}"
        ) from exc
    try:
        return _aware_utc(parsed, field=field)
    except ValueError as exc:
        raise UFCUpdateScheduleStateError(str(exc)) from exc


def iso_week_identity(
    value: datetime, *, timezone_name: str = UFC_UPDATE_SCHEDULE_TIMEZONE
) -> str:
    """Return the local ISO-week identity used by the weekly schedule."""
    aware = _aware_utc(value, field="value")
    try:
        local = aware.astimezone(ZoneInfo(timezone_name))
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown update schedule timezone {timezone_name!r}") from exc
    iso = local.date().isocalendar()
    return f"{iso.year:04d}-W{iso.week:02d}"


def utc_month_identity(value: datetime) -> str:
    """Return ``YYYY-MM`` for an accepted update timestamp in UTC."""
    return _aware_utc(value, field="value").strftime("%Y-%m")


def _parse_iso_week(value: str) -> tuple[int, int]:
    match = _ISO_WEEK_RE.fullmatch(str(value or "").strip())
    if match is None:
        raise ValueError(f"Invalid ISO-week identity {value!r}; expected YYYY-Www")
    year = int(match.group("year"))
    week = int(match.group("week"))
    try:
        date.fromisocalendar(year, week, 1)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO-week identity {value!r}") from exc
    return year, week


def _parse_utc_month(value: str | None, *, allow_blank: bool) -> tuple[int, int] | None:
    text = str(value or "").strip()
    if not text and allow_blank:
        return None
    match = _UTC_MONTH_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"Invalid UTC month {value!r}; expected YYYY-MM")
    year = int(match.group("year"))
    month = int(match.group("month"))
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid UTC month {value!r}; expected YYYY-MM")
    return year, month


def _new_state() -> dict[str, object]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "updated_at_utc": None,
        "weekly": {},
        "monthly_candidates": {},
    }


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    """Hold a tiny cross-process lock around state reads and transitions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        acquired = True
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class UFCUpdateSchedule:
    """Persist and enforce weekly update and monthly candidate attempt slots."""

    def __init__(
        self,
        *,
        state_path: Path | str = UFC_UPDATE_SCHEDULE_STATE_PATH,
        promotion_month_utc: str | None = UFC_V7_PROMOTION_MONTH_UTC,
        monthly_policy=None,
        timezone_name: str = UFC_UPDATE_SCHEDULE_TIMEZONE,
        claim_stale_after_seconds: int = UFC_UPDATE_CLAIM_STALE_AFTER_SECONDS,
        clock: Callable[[], datetime] | None = None,
        claim_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.state_path = Path(state_path)
        self.lock_path = self.state_path.with_name(f"{self.state_path.name}.lock")
        self.timezone_name = str(timezone_name or "").strip()
        try:
            self.local_timezone = ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(
                f"Unknown update schedule timezone {self.timezone_name!r}"
            ) from exc
        if monthly_policy is not None:
            from src.model.monthly_policy import load_monthly_policy
            monthly_policy = load_monthly_policy(monthly_policy.pin, body=monthly_policy.body)
        self.monthly_policy = monthly_policy
        if monthly_policy is not None:
            # Explicit successor authority is never inferred from the V7 setting.
            promotion_month_utc = monthly_policy.promotion_month_utc
        self.promotion_month_utc = str(promotion_month_utc or "").strip()
        self._promotion_month_value = _parse_utc_month(
            self.promotion_month_utc,
            allow_blank=True,
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._claim_id_factory = claim_id_factory or (lambda: uuid4().hex)
        self.claim_stale_after_seconds = max(int(claim_stale_after_seconds), 1)

    def _now(self, value: datetime | None = None) -> datetime:
        return _aware_utc(value if value is not None else self._clock(), field="now")

    def _load_unlocked(self) -> dict[str, object]:
        if not self.state_path.exists():
            return _new_state()
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise UFCUpdateScheduleStateError(
                f"Could not load UFC update schedule state at {self.state_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise UFCUpdateScheduleStateError(
                "UFC update schedule state must be a JSON object"
            )
        if payload.get("schema_version") != STATE_SCHEMA_VERSION:
            raise UFCUpdateScheduleStateError(
                "Unsupported UFC update schedule state schema_version "
                f"{payload.get('schema_version')!r}"
            )
        for field in ("weekly", "monthly_candidates"):
            if not isinstance(payload.get(field), dict):
                raise UFCUpdateScheduleStateError(
                    f"UFC update schedule state field {field!r} must be an object"
                )
        return payload

    def _save_unlocked(self, state: dict[str, object], *, now_utc: datetime) -> None:
        state["updated_at_utc"] = _utc_iso(now_utc)
        write_json_atomically(state, self.state_path)

    @contextmanager
    def _locked_state(self) -> Iterator[dict[str, object]]:
        with _exclusive_file_lock(self.lock_path):
            yield self._load_unlocked()

    def load_state(self) -> dict[str, object]:
        """Return a copy of durable state without changing it."""
        with self._locked_state() as state:
            return deepcopy(state)

    def reconcile_stale_claims(
        self,
        *,
        now: datetime | None = None,
    ) -> dict[str, object]:
        """Fail stale interrupted claims so the approved retry window can proceed.

        Claims remain in progress for a generous bounded lease.  Once that lease
        expires, a still-claimed slot cannot belong to a healthy job and is
        durably recorded as failed.  This preserves fail-closed behavior while
        preventing a hard process restart from stranding Wednesday recovery.
        """

        now_utc = self._now(now)
        recovered: list[dict[str, object]] = []
        with self._locked_state() as state:
            for collection_name in ("weekly", "monthly_candidates"):
                records = state.get(collection_name)
                if not isinstance(records, dict):
                    raise UFCUpdateScheduleStateError(
                        f"UFC update schedule state field {collection_name!r} must be an object"
                    )
                for record_key, raw_entry in records.items():
                    if not isinstance(raw_entry, dict):
                        raise UFCUpdateScheduleStateError(
                            f"{collection_name}.{record_key} must be an object"
                        )
                    slots = self._slots(
                        raw_entry,
                        context=f"{collection_name}.{record_key}",
                    )
                    for slot in _SLOTS:
                        attempt = slots.get(slot)
                        if not isinstance(attempt, dict):
                            continue
                        if str(attempt.get("status") or "") != STATUS_CLAIMED:
                            continue
                        if collection_name == "weekly" and attempt.get("completion_evidence_required") is True:
                            # A publication-capable caller may have committed before
                            # its final calendar write. Time alone proves no failure.
                            continue
                        claimed_at = _parse_utc_iso(
                            attempt.get("claimed_at_utc"),
                            field=(
                                f"{collection_name}.{record_key}.slots.{slot}."
                                "claimed_at_utc"
                            ),
                        )
                        age_seconds = (now_utc - claimed_at).total_seconds()
                        if age_seconds < self.claim_stale_after_seconds:
                            continue
                        attempt["status"] = STATUS_FAILED
                        attempt["completed_at_utc"] = _utc_iso(now_utc)
                        attempt["error"] = "interrupted_claim_lease_expired"
                        attempt["recovered_after_restart"] = True
                        attempt["claim_age_seconds"] = float(age_seconds)
                        if collection_name == "monthly_candidates":
                            raw_entry["status"] = (
                                "retry_pending" if slot == SLOT_TUESDAY else "failed"
                            )
                        recovered.append(
                            {
                                "collection": collection_name,
                                "record": str(record_key),
                                "slot": slot,
                                "claim_id": attempt.get("claim_id"),
                                "claim_age_seconds": float(age_seconds),
                            }
                        )
            if recovered:
                self._save_unlocked(state, now_utc=now_utc)
        return {
            "changed": bool(recovered),
            "recovered_count": len(recovered),
            "recovered_claims": recovered,
            "claim_stale_after_seconds": self.claim_stale_after_seconds,
        }

    def _week_key(self, now_utc: datetime) -> str:
        return iso_week_identity(now_utc, timezone_name=self.timezone_name)

    def _slot_datetime(self, week_key: str, slot: str) -> datetime:
        year, week = _parse_iso_week(week_key)
        if slot == SLOT_TUESDAY:
            weekday = UFC_UPDATE_TUESDAY_ISO_WEEKDAY
        elif slot == SLOT_WEDNESDAY:
            weekday = UFC_UPDATE_WEDNESDAY_ISO_WEEKDAY
        else:
            raise ValueError(f"Unknown schedule slot {slot!r}")
        local_date = date.fromisocalendar(year, week, weekday)
        return datetime.combine(
            local_date,
            time(
                UFC_UPDATE_SCHEDULE_LOCAL_HOUR,
                UFC_UPDATE_SCHEDULE_LOCAL_MINUTE,
            ),
            tzinfo=self.local_timezone,
        )

    def _next_week_key(self, week_key: str) -> str:
        year, week = _parse_iso_week(week_key)
        next_week = date.fromisocalendar(year, week, 1) + timedelta(days=7)
        iso = next_week.isocalendar()
        return f"{iso.year:04d}-W{iso.week:02d}"

    def _decision(
        self,
        *,
        action: str,
        reason: str,
        now_utc: datetime,
        week_key: str,
        slot: str | None,
        due_at_local: datetime | None,
        next_due_at_local: datetime | None,
        next_due_is_conditional: bool = False,
        utc_month: str | None = None,
        accepted_update_id: str | None = None,
        accepted_at_utc: datetime | None = None,
        candidate_only: bool = False,
        monthly_input: dict[str, object] | None = None,
        monthly_context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "action": action,
            "is_due": action != ACTION_NONE,
            "reason": reason,
            "now_utc": _utc_iso(now_utc),
            "iso_week": week_key,
            "slot": slot,
            "due_at_local": due_at_local.isoformat() if due_at_local else None,
            "due_at_utc": _utc_iso(due_at_local) if due_at_local else None,
            "next_due_at_local": (
                next_due_at_local.isoformat() if next_due_at_local else None
            ),
            "next_due_at_utc": (
                _utc_iso(next_due_at_local) if next_due_at_local else None
            ),
            "next_due_is_conditional": bool(next_due_is_conditional),
            "utc_month": utc_month,
            "accepted_update_id": accepted_update_id,
            "accepted_at_utc": (
                _utc_iso(accepted_at_utc) if accepted_at_utc is not None else None
            ),
            "candidate_only": bool(candidate_only),
            "monthly_input": deepcopy(monthly_input),
            "monthly_context": deepcopy(monthly_context),
        }

    @staticmethod
    def _slots(entry: dict[str, object], *, context: str) -> dict[str, object]:
        slots = entry.get("slots", {})
        if not isinstance(slots, dict):
            raise UFCUpdateScheduleStateError(f"{context}.slots must be an object")
        return slots

    @staticmethod
    def _attempt_status(
        slots: dict[str, object],
        slot: str,
        *,
        context: str,
    ) -> str | None:
        attempt = slots.get(slot)
        if attempt is None:
            return None
        if not isinstance(attempt, dict):
            raise UFCUpdateScheduleStateError(
                f"{context}.slots.{slot} must be an object"
            )
        status = str(attempt.get("status") or "").strip()
        if status not in _ATTEMPT_STATUSES:
            raise UFCUpdateScheduleStateError(
                f"{context}.slots.{slot}.status is invalid: {status!r}"
            )
        return status

    def _weekly_entry(
        self,
        state: dict[str, object],
        week_key: str,
    ) -> dict[str, object]:
        weekly = state["weekly"]
        assert isinstance(weekly, dict)
        entry = weekly.get(week_key, {})
        if not isinstance(entry, dict):
            raise UFCUpdateScheduleStateError(f"weekly.{week_key} must be an object")
        return entry

    def _plan_data_update(
        self,
        state: dict[str, object],
        *,
        now_utc: datetime,
    ) -> dict[str, object]:
        local_now = now_utc.astimezone(self.local_timezone)
        week_key = self._week_key(now_utc)
        for original_week, original_entry in state["weekly"].items():
            for original_slot, original_attempt in self._slots(original_entry, context=f"weekly.{original_week}").items():
                if (original_attempt.get("status") == STATUS_CLAIMED
                    and original_attempt.get("completion_evidence_required") is True):
                    return self._decision(
                        action=ACTION_NONE, reason="original_completion_evidence_required",
                        now_utc=now_utc, week_key=original_week, slot=original_slot,
                        due_at_local=None, next_due_at_local=None)
        next_week_tuesday = self._slot_datetime(
            self._next_week_key(week_key), SLOT_TUESDAY
        )
        tuesday_due = self._slot_datetime(week_key, SLOT_TUESDAY)
        wednesday_due = self._slot_datetime(week_key, SLOT_WEDNESDAY)
        entry = self._weekly_entry(state, week_key)
        slots = self._slots(entry, context=f"weekly.{week_key}")
        tuesday_status = self._attempt_status(
            slots, SLOT_TUESDAY, context=f"weekly.{week_key}"
        )
        wednesday_status = self._attempt_status(
            slots, SLOT_WEDNESDAY, context=f"weekly.{week_key}"
        )
        accepted_update = entry.get("accepted_update")
        if accepted_update is not None and not isinstance(accepted_update, dict):
            raise UFCUpdateScheduleStateError(
                f"weekly.{week_key}.accepted_update must be an object"
            )
        if (
            tuesday_status == STATUS_SUCCEEDED or wednesday_status == STATUS_SUCCEEDED
        ) and not accepted_update:
            raise UFCUpdateScheduleStateError(
                f"weekly.{week_key} succeeded without accepted_update metadata"
            )

        if accepted_update:
            return self._decision(
                action=ACTION_NONE,
                reason="weekly_update_already_succeeded",
                now_utc=now_utc,
                week_key=week_key,
                slot=None,
                due_at_local=None,
                next_due_at_local=next_week_tuesday,
            )

        weekday = local_now.isoweekday()
        if weekday < UFC_UPDATE_TUESDAY_ISO_WEEKDAY or (
            weekday == UFC_UPDATE_TUESDAY_ISO_WEEKDAY and local_now < tuesday_due
        ):
            return self._decision(
                action=ACTION_NONE,
                reason="waiting_for_tuesday_slot",
                now_utc=now_utc,
                week_key=week_key,
                slot=None,
                due_at_local=None,
                next_due_at_local=tuesday_due,
            )

        if weekday == UFC_UPDATE_TUESDAY_ISO_WEEKDAY:
            if tuesday_status is None:
                return self._decision(
                    action=ACTION_DATA_UPDATE,
                    reason="tuesday_update_due",
                    now_utc=now_utc,
                    week_key=week_key,
                    slot=SLOT_TUESDAY,
                    due_at_local=tuesday_due,
                    next_due_at_local=tuesday_due,
                )
            if tuesday_status == STATUS_CLAIMED:
                return self._decision(
                    action=ACTION_NONE,
                    reason="tuesday_update_in_progress",
                    now_utc=now_utc,
                    week_key=week_key,
                    slot=SLOT_TUESDAY,
                    due_at_local=tuesday_due,
                    next_due_at_local=wednesday_due,
                    next_due_is_conditional=True,
                )
            if tuesday_status == STATUS_FAILED:
                return self._decision(
                    action=ACTION_NONE,
                    reason="waiting_for_wednesday_data_retry",
                    now_utc=now_utc,
                    week_key=week_key,
                    slot=None,
                    due_at_local=None,
                    next_due_at_local=wednesday_due,
                    next_due_is_conditional=True,
                )

        if weekday == UFC_UPDATE_WEDNESDAY_ISO_WEEKDAY:
            if local_now < wednesday_due:
                if tuesday_status == STATUS_CLAIMED:
                    reason = "tuesday_update_in_progress"
                elif tuesday_status == STATUS_FAILED:
                    reason = "waiting_for_wednesday_data_retry"
                else:
                    reason = "tuesday_slot_missed_waiting_for_wednesday"
                return self._decision(
                    action=ACTION_NONE,
                    reason=reason,
                    now_utc=now_utc,
                    week_key=week_key,
                    slot=None,
                    due_at_local=None,
                    next_due_at_local=wednesday_due,
                    next_due_is_conditional=True,
                )
            if tuesday_status == STATUS_CLAIMED:
                return self._decision(
                    action=ACTION_NONE,
                    reason="tuesday_update_still_in_progress",
                    now_utc=now_utc,
                    week_key=week_key,
                    slot=SLOT_TUESDAY,
                    due_at_local=tuesday_due,
                    next_due_at_local=wednesday_due,
                    next_due_is_conditional=True,
                )
            if wednesday_status is None and tuesday_status in {None, STATUS_FAILED}:
                return self._decision(
                    action=ACTION_DATA_UPDATE,
                    reason=(
                        "wednesday_retry_due_after_tuesday_failure"
                        if tuesday_status == STATUS_FAILED
                        else "wednesday_retry_due_after_missed_tuesday_slot"
                    ),
                    now_utc=now_utc,
                    week_key=week_key,
                    slot=SLOT_WEDNESDAY,
                    due_at_local=wednesday_due,
                    next_due_at_local=wednesday_due,
                )
            if wednesday_status == STATUS_CLAIMED:
                return self._decision(
                    action=ACTION_NONE,
                    reason="wednesday_data_retry_in_progress",
                    now_utc=now_utc,
                    week_key=week_key,
                    slot=SLOT_WEDNESDAY,
                    due_at_local=wednesday_due,
                    next_due_at_local=next_week_tuesday,
                )

        return self._decision(
            action=ACTION_NONE,
            reason="weekly_data_window_complete",
            now_utc=now_utc,
            week_key=week_key,
            slot=None,
            due_at_local=None,
            next_due_at_local=next_week_tuesday,
        )

    def plan_data_update(self, *, now: datetime | None = None) -> dict[str, object]:
        """Return the current weekly decision without claiming work."""
        now_utc = self._now(now)
        with self._locked_state() as state:
            return self._plan_data_update(state, now_utc=now_utc)

    def claim_data_update(self, *, now: datetime | None = None) -> dict[str, object]:
        """Atomically claim a due Tuesday update or conditional Wednesday retry."""
        now_utc = self._now(now)
        with self._locked_state() as state:
            decision = self._plan_data_update(state, now_utc=now_utc)
            if decision["action"] != ACTION_DATA_UPDATE:
                return {**decision, "claimed": False, "claim_id": None}

            week_key = str(decision["iso_week"])
            slot = str(decision["slot"])
            weekly = state["weekly"]
            assert isinstance(weekly, dict)
            entry = weekly.setdefault(
                week_key,
                {"iso_week": week_key, "slots": {}, "accepted_update": None},
            )
            if not isinstance(entry, dict):
                raise UFCUpdateScheduleStateError(
                    f"weekly.{week_key} must be an object"
                )
            slots = entry.setdefault("slots", {})
            if not isinstance(slots, dict):
                raise UFCUpdateScheduleStateError(
                    f"weekly.{week_key}.slots must be an object"
                )
            if slot == SLOT_WEDNESDAY and SLOT_TUESDAY not in slots:
                missed_tuesday_due = self._slot_datetime(
                    week_key,
                    SLOT_TUESDAY,
                )
                slots[SLOT_TUESDAY] = {
                    "status": STATUS_FAILED,
                    "claim_id": None,
                    "due_at_utc": _utc_iso(missed_tuesday_due),
                    "completed_at_utc": _utc_iso(now_utc),
                    "failure": "tuesday_slot_missed",
                    "synthetic": True,
                }
            claim_id = str(self._claim_id_factory() or "").strip()
            if not claim_id:
                raise UFCUpdateScheduleTransitionError(
                    "claim_id_factory returned an empty claim ID"
                )
            slots[slot] = {
                "status": STATUS_CLAIMED,
                "claim_id": claim_id,
                "due_at_utc": decision["due_at_utc"],
                "claimed_at_utc": _utc_iso(now_utc),
            }
            if self.monthly_policy is not None:
                slots[slot]["monthly_policy"] = {
                    "binding": deepcopy(self.monthly_policy.binding),
                    "promotion_month_utc": self.monthly_policy.promotion_month_utc,
                }
            self._save_unlocked(state, now_utc=now_utc)
            result = {**decision, "claimed": True, "claim_id": claim_id}
            if "monthly_policy" in slots[slot]:
                result["monthly_policy"] = deepcopy(slots[slot]["monthly_policy"])
            return result

    @staticmethod
    def _locate_claim(
        records: dict[str, object],
        claim_id: str,
        *,
        context: str,
    ) -> tuple[str, str, dict[str, object], dict[str, object]]:
        matches: list[tuple[str, str, dict[str, object], dict[str, object]]] = []
        for record_key, raw_entry in records.items():
            if not isinstance(raw_entry, dict):
                raise UFCUpdateScheduleStateError(
                    f"{context}.{record_key} must be an object"
                )
            slots = raw_entry.get("slots", {})
            if not isinstance(slots, dict):
                raise UFCUpdateScheduleStateError(
                    f"{context}.{record_key}.slots must be an object"
                )
            for slot in _SLOTS:
                raw_attempt = slots.get(slot)
                if not isinstance(raw_attempt, dict):
                    continue
                if str(raw_attempt.get("claim_id") or "") == claim_id:
                    matches.append((str(record_key), slot, raw_entry, raw_attempt))
        if not matches:
            raise UFCUpdateScheduleTransitionError(
                f"Unknown {context} claim ID {claim_id!r}"
            )
        if len(matches) != 1:
            raise UFCUpdateScheduleStateError(
                f"Duplicate {context} claim ID {claim_id!r}"
            )
        return matches[0]

    def protect_data_completion(self, claim_id: str, *, now: datetime | None = None):
        """Fence unknown publication outcomes before a publication-capable job."""
        with self._locked_state() as state:
            _week, _slot, _entry, attempt = self._locate_claim(
                state["weekly"], claim_id, context="weekly")
            if attempt.get("status") != STATUS_CLAIMED:
                raise UFCUpdateScheduleTransitionError("Data completion fence requires an active claim")
            if attempt.get("completion_evidence_required") is not True:
                attempt["completion_evidence_required"] = True
                self._save_unlocked(state, now_utc=self._now(now))

    def record_monthly_preparation_refusal(self, claim_id: str, error: str, *, now=None):
        """Retain an upstream pin-provisioning refusal after data completion."""
        with self._locked_state() as state:
            _week, _slot, entry, attempt = self._locate_claim(state["weekly"], claim_id, context="weekly")
            if attempt.get("status") != STATUS_SUCCEEDED:
                raise UFCUpdateScheduleTransitionError("Monthly refusal requires completed data")
            month = str(entry["accepted_update"]["accepted_at_utc"])[:7]
            intent = state.get("monthly_input_intents", {}).get(month)
            if not isinstance(intent, dict) or intent["context"].get("claim_id") != claim_id:
                return
            refusal = {"stage": "accepted_input_provisioning", "error": str(error)}
            if refusal not in intent["attempts"]:
                intent["attempts"].append(refusal)
                self._save_unlocked(state, now_utc=self._now(now))

    def _register_monthly_source_update(
        self,
        state: dict[str, object],
        *,
        accepted_update_id: str,
        accepted_at_utc: datetime,
        week_key: str,
        data_slot: str,
        monthly_input: dict[str, object],
    ) -> dict[str, object]:
        accepted_month_key = utc_month_identity(accepted_at_utc)
        accepted_month_value = _parse_utc_month(
            accepted_month_key,
            allow_blank=False,
        )
        promotion_month_value = _parse_utc_month(
            monthly_input["context"]["promotion_month_utc"], allow_blank=True)
        if promotion_month_value is None:
            return {
                "registered": False,
                "changed": False,
                "reason": "v7_promotion_month_not_configured",
                "utc_month": accepted_month_key,
            }
        assert accepted_month_value is not None
        if accepted_month_value <= promotion_month_value:
            return {
                "registered": False,
                "changed": False,
                "reason": "accepted_update_not_after_v7_promotion_month",
                "utc_month": accepted_month_key,
            }

        monthly = state["monthly_candidates"]
        assert isinstance(monthly, dict)
        existing = monthly.get(accepted_month_key)
        if existing is not None:
            if not isinstance(existing, dict):
                raise UFCUpdateScheduleStateError(
                    f"monthly_candidates.{accepted_month_key} must be an object"
                )
            source = existing.get("source_update")
            if not isinstance(source, dict):
                raise UFCUpdateScheduleStateError(
                    f"monthly_candidates.{accepted_month_key}.source_update must be an object"
                )
            return {
                "registered": True,
                "changed": False,
                "reason": "first_accepted_update_already_registered",
                "utc_month": accepted_month_key,
                "accepted_update_id": source.get("accepted_update_id"),
            }

        monthly[accepted_month_key] = {
            "utc_month": accepted_month_key,
            "status": "pending",
            "source_update": {
                "accepted_update_id": accepted_update_id,
                "accepted_at_utc": _utc_iso(accepted_at_utc),
                "iso_week": week_key,
                "data_slot": data_slot,
                "monthly_input": deepcopy(monthly_input),
            },
            "slots": {},
        }
        return {
            "registered": True,
            "changed": True,
            "reason": "first_eligible_accepted_update_registered",
            "utc_month": accepted_month_key,
            "accepted_update_id": accepted_update_id,
        }

    def _monthly_context(self, *, accepted_id, accepted_at, week, slot, claim_id, state=None):
        promotion_month = self.promotion_month_utc
        if state is not None:
            attempt = (state.get("weekly", {}).get(week, {}).get("slots", {}).get(slot) or {})
            if attempt.get("claim_id") == claim_id and attempt.get("monthly_policy") is not None:
                promotion_month = attempt["monthly_policy"]["promotion_month_utc"]
            original = state.get("monthly_input_intents", {}).get(utc_month_identity(accepted_at), {}).get("context")
            if original is not None and original.get("claim_id") == claim_id:
                promotion_month = original["promotion_month_utc"]
        return {
            "utc_month": utc_month_identity(accepted_at),
            "accepted_update_id": accepted_id,
            "accepted_at_utc": _utc_iso(accepted_at),
            "iso_week": week, "data_slot": slot, "claim_id": claim_id,
            "schedule_timezone": self.timezone_name,
            "promotion_month_utc": promotion_month,
        }

    @staticmethod
    def _monthly_request(request):
        value = json.loads(json.dumps(request, allow_nan=False))
        if not isinstance(value, dict) or set(value) != {"acceptance", "output_root"}:
            raise ValueError("monthly request requires acceptance and output_root")
        if not isinstance(value["acceptance"], dict) or set(value["acceptance"]) != {"path", "bytes", "sha256"}:
            raise ValueError("monthly request requires an independent acceptance pin")
        if not isinstance(value["output_root"], str) or not value["output_root"]:
            raise ValueError("monthly request requires an explicit output root")
        return value

    def _reserve_monthly_input(self, state, context, request=None):
        """Reserve the first completion's identity in the same write as data success."""
        month = context["utc_month"]
        value = _parse_utc_month(month, allow_blank=False)
        promotion_month_value = _parse_utc_month(context["promotion_month_utc"], allow_blank=True)
        if promotion_month_value is None:
            return {"registered": False, "changed": False,
                    "reason": "v7_promotion_month_not_configured", "utc_month": month}
        if value <= promotion_month_value:
            return {"registered": False, "changed": False,
                    "reason": "accepted_update_not_after_v7_promotion_month", "utc_month": month}
        if month in state["monthly_candidates"]:
            return None
        intents = state.setdefault("monthly_input_intents", {})
        if not isinstance(intents, dict):
            raise UFCUpdateScheduleStateError("monthly_input_intents must be an object")
        if month not in intents:
            intents[month] = {"schema": 1, "context": deepcopy(context),
                             "status": "waiting_for_accepted_input", "attempts": []}
            attempt = state.get("weekly", {}).get(context["iso_week"], {}).get("slots", {}).get(context["data_slot"], {})
            policy = attempt.get("monthly_policy") if attempt.get("claim_id") == context["claim_id"] else None
            if policy is not None:
                intents[month]["policy_binding"] = deepcopy(policy["binding"])
        intent = intents[month]
        if intent.get("context") == context and "request" not in intent and request is not None:
            try:
                intent["request"] = self._monthly_request(request)
            except (TypeError, ValueError) as exc:
                intent["attempts"].append({"stage": "accepted_input_provisioning",
                                           "error": f"invalid request: {exc}"})
        return None

    def complete_data_update(
        self, claim_id: str, *, succeeded: bool,
        accepted_update_id: str | None = None, accepted_at: datetime | None = None,
        monthly_input_request: dict[str, object] | None = None,
        monthly_retry: dict[str, str] | None = None,
        completion_evidence_pin: dict[str, object] | None = None,
        expected_monthly_context: dict[str, str] | None = None,
        error: str | None = None, now: datetime | None = None,
    ) -> dict[str, object]:
        """Persist data success before independently preparing a monthly selection.

        An interruption or monthly refusal never makes completed acquisition due
        again. Duplicate data completions reconcile the original reserved input.
        """
        claim_id = str(claim_id or "").strip()
        accepted_id = str(accepted_update_id or "").strip()
        if not claim_id:
            raise ValueError("claim_id must not be empty")
        if succeeded and not accepted_id:
            raise ValueError("accepted_update_id is required for a successful update")
        now_utc = self._now(now)
        accepted_at_utc = self._now(accepted_at) if accepted_at is not None else now_utc
        with self._locked_state() as state:
            week, slot, entry, attempt = self._locate_claim(state["weekly"], claim_id,
                                                          context="weekly")
            if expected_monthly_context is not None and expected_monthly_context != self._monthly_context(
                accepted_id=accepted_id, accepted_at=accepted_at_utc, week=week, slot=slot, claim_id=claim_id, state=state
            ):
                raise UFCUpdateScheduleTransitionError("Original completion schedule context conflicts")
            status = str(attempt.get("status") or "")
            expected = STATUS_SUCCEEDED if succeeded else STATUS_FAILED
            changed = status == STATUS_CLAIMED
            if not changed:
                if status != expected:
                    raise UFCUpdateScheduleTransitionError(
                        f"Data claim {claim_id!r} is already {status}, not {expected}")
                if completion_evidence_pin is not None and attempt.get("completion_evidence_pin") not in (None, completion_evidence_pin):
                    raise UFCUpdateScheduleTransitionError("Completed attempt evidence conflicts")
                if succeeded:
                    accepted = entry.get("accepted_update")
                    if not isinstance(accepted, dict) or accepted.get("accepted_update_id") != accepted_id:
                        raise UFCUpdateScheduleTransitionError("Completed data generation conflicts")
                    original_at = _parse_utc_iso(accepted.get("accepted_at_utc"), field="accepted_at_utc")
                    if accepted_at is not None and original_at != accepted_at_utc:
                        raise UFCUpdateScheduleTransitionError("Completed data cutoff conflicts")
                    if completion_evidence_pin is not None and accepted.get("completion_evidence_pin") not in (None, completion_evidence_pin):
                        raise UFCUpdateScheduleTransitionError("Completed data evidence conflicts")
                    accepted_at_utc = original_at
            else:
                attempt["status"] = expected
                attempt["completed_at_utc"] = _utc_iso(now_utc)
                if completion_evidence_pin is not None:
                    attempt["completion_evidence_pin"] = deepcopy(completion_evidence_pin)
                if succeeded:
                    entry["accepted_update"] = {"accepted_update_id": accepted_id,
                        "accepted_at_utc": _utc_iso(accepted_at_utc), "slot": slot}
                    if completion_evidence_pin is not None:
                        entry["accepted_update"]["completion_evidence_pin"] = deepcopy(completion_evidence_pin)
                else:
                    attempt["error"] = str(error or "update_failed")
            context = self._monthly_context(accepted_id=accepted_id, accepted_at=accepted_at_utc,
                                           week=week, slot=slot, claim_id=claim_id, state=state)
            before = deepcopy(state)
            if succeeded:
                self._reserve_monthly_input(state, context, monthly_input_request)
            if changed or state != before:
                self._save_unlocked(state, now_utc=now_utc)
            next_data = self._plan_data_update(state, now_utc=now_utc)
        registration = None
        if succeeded:
            registration = self.register_accepted_update(
                accepted_update_id=accepted_id, accepted_at=accepted_at_utc,
                iso_week=week, data_slot=slot, claim_id=claim_id,
                monthly_input_request=monthly_input_request, monthly_retry=monthly_retry, now=now_utc)
        return {"changed": changed,
                "reason": "data_result_recorded" if changed else "data_result_already_recorded",
                "iso_week": week, "slot": slot, "succeeded": succeeded,
                "monthly_registration": registration, "next": next_data}

    def register_accepted_update(
        self, *, accepted_update_id: str, accepted_at: datetime, iso_week: str,
        data_slot: str, claim_id: str = "",
        monthly_input_request: dict[str, object] | None = None,
        monthly_retry: dict[str, str] | None = None,
        now: datetime | None = None,
    ) -> dict[str, object]:
        """Save and verify the original monthly input before committing registration.

        Preparation is outside the schedule lock. Its separate lock serializes
        duplicate readers without holding up source scheduling. Partial outputs
        refuse; an explicit fresh-root retry must name the original failed root.
        """
        from src.model.monthly_input import prepare_monthly_input, load_monthly_input

        accepted_id = str(accepted_update_id or "").strip()
        if not accepted_id:
            raise ValueError("accepted_update_id must not be empty")
        _parse_iso_week(iso_week)
        if data_slot not in _SLOTS:
            raise ValueError(f"data_slot must be one of {_SLOTS!r}")
        accepted_at_utc = _aware_utc(accepted_at, field="accepted_at")
        now_utc = self._now(now)
        context = self._monthly_context(accepted_id=accepted_id, accepted_at=accepted_at_utc,
                                       week=iso_week, slot=data_slot, claim_id=claim_id)
        month = context["utc_month"]
        with _exclusive_file_lock(self.state_path.with_name(self.state_path.name + ".monthly-input.lock")):
            with self._locked_state() as state:
                context = self._monthly_context(accepted_id=accepted_id, accepted_at=accepted_at_utc,
                    week=iso_week, slot=data_slot, claim_id=claim_id, state=state)
                before = deepcopy(state)
                ineligible = self._reserve_monthly_input(state, context)
                if ineligible is not None:
                    return ineligible
                existing = state["monthly_candidates"].get(month)
                if existing is not None:
                    source = existing.get("source_update", {})
                    transport = source.get("monthly_input")
                    if transport is None:
                        return {"registered": False, "changed": False, "utc_month": month,
                                "reason": "historical_monthly_input_requires_original_evidence"}
                    original_context = transport.get("context")
                    if context == original_context and monthly_input_request is not None:
                        intent = state.get("monthly_input_intents", {}).get(month, {})
                        if monthly_input_request != intent.get("request"):
                            raise UFCUpdateScheduleTransitionError("Completed monthly request conflicts")
                    # A later completion cannot change an already selected month.
                    duplicate = deepcopy(transport)
                    request = None
                else:
                    intent = state["monthly_input_intents"][month]
                    original_context = intent.get("context")
                    if original_context != context:
                        return {"registered": False, "changed": False, "utc_month": month,
                                "reason": "first_qualifying_input_pending",
                                "accepted_update_id": (original_context or {}).get("accepted_update_id")}
                    request = deepcopy(intent.get("request"))
                    if monthly_input_request is not None:
                        try:
                            supplied = self._monthly_request(monthly_input_request)
                        except (TypeError, ValueError) as exc:
                            refusal = {"stage": "accepted_input_provisioning", "error": f"invalid request: {exc}"}
                            if refusal not in intent["attempts"]:
                                intent["attempts"].append(refusal)
                            intent["status"] = "refused"
                            self._save_unlocked(state, now_utc=now_utc)
                            return {"registered": False, "changed": True, "utc_month": month,
                                    "reason": "monthly_input_refused", "error": refusal["error"],
                                    "data_update_succeeded": True, "reacquisition_authorized": False}
                        if request is not None and supplied != request:
                            if (not isinstance(monthly_retry, dict)
                                or monthly_retry.get("previous_output_root") != request.get("output_root")
                                or not str(monthly_retry.get("reason") or "").strip()
                                or supplied.get("acceptance") != request.get("acceptance")
                                or supplied.get("output_root") == request.get("output_root")
                                or intent.get("status") != "refused"):
                                raise UFCUpdateScheduleTransitionError("Original monthly request conflicts")
                            intent["attempts"].append({"request": request,
                                                      "retry": deepcopy(monthly_retry)})
                        request = supplied
                        intent["request"] = request
                    if request is None:
                        if state != before:
                            self._save_unlocked(state, now_utc=now_utc)
                        return {"registered": False, "changed": state != before, "utc_month": month,
                                "reason": "monthly_accepted_input_required"}
                    intent["status"] = "preparing"
                    policy_binding = deepcopy(intent.get("policy_binding"))
                    if state != before:
                        self._save_unlocked(state, now_utc=now_utc)
                    duplicate = None
            if duplicate is not None:
                load_monthly_input(duplicate, context=original_context)
                return {"registered": True, "changed": False, "utc_month": month,
                        "reason": "first_accepted_update_already_registered",
                        "accepted_update_id": original_context["accepted_update_id"],
                        "monthly_input": duplicate}
            try:
                if policy_binding is not None:
                    from src.model.monthly_policy import load_monthly_policy
                    from src.model import prepared_input
                    root = prepared_input._path(request["output_root"])
                    policy_body = None
                    if root.exists():
                        # Final saved originals remain authoritative after the
                        # former descriptor/policy locations disappear.
                        receipt = prepared_input._json(prepared_input._path(root / "monthly-receipt.json").read_bytes())
                        if receipt.get("acceptance", {}).get("pin") != request["acceptance"]:
                            raise UFCUpdateScheduleTransitionError("Saved original acceptance pin conflicts")
                        descriptor = prepared_input._json(prepared_input._attached(receipt["acceptance"]))
                        policy_body = prepared_input._attached(receipt["policy"])
                    else:
                        descriptor = prepared_input._json(prepared_input._capture(request["acceptance"]))
                    if descriptor.get("schema") != 2 or load_monthly_policy(descriptor["policy"], body=policy_body).binding != policy_binding:
                        raise UFCUpdateScheduleTransitionError("Original monthly claim policy conflicts")
                    resolved_policy_binding = load_monthly_policy(descriptor["policy"], body=policy_body, context=context).binding
                    with self._locked_state() as state:
                        intent = state["monthly_input_intents"][month]
                        if intent.get("context") != context or intent.get("request") != request:
                            raise UFCUpdateScheduleTransitionError("Monthly intent changed before policy realization")
                        if intent.get("resolved_policy_binding") not in (None, resolved_policy_binding):
                            raise UFCUpdateScheduleTransitionError("Original realized monthly policy conflicts")
                        if intent.get("resolved_policy_binding") is None:
                            intent["resolved_policy_binding"] = deepcopy(resolved_policy_binding)
                            self._save_unlocked(state, now_utc=now_utc)
                transport = prepare_monthly_input(request, context=context)
                load_monthly_input(transport, context=context)
                if policy_binding is not None and transport.get("policy_binding") != resolved_policy_binding:
                    raise UFCUpdateScheduleTransitionError("Saved monthly policy differs from original claim")
            except (ValueError, OSError, RuntimeError, KeyError, TypeError, AttributeError) as exc:
                with self._locked_state() as state:
                    intent = state["monthly_input_intents"][month]
                    refusal = {"request": request, "error": f"{type(exc).__name__}: {exc}"}
                    if not intent["attempts"] or intent["attempts"][-1] != refusal:
                        intent["attempts"].append(refusal)
                    intent["status"] = "refused"
                    self._save_unlocked(state, now_utc=now_utc)
                return {"registered": False, "changed": True, "utc_month": month,
                        "reason": "monthly_input_refused", "error": refusal["error"],
                        "data_update_succeeded": True, "reacquisition_authorized": False}
            with self._locked_state() as state:
                intent = state["monthly_input_intents"][month]
                if intent.get("context") != context or intent.get("request") != request:
                    raise UFCUpdateScheduleTransitionError("Monthly intent changed during preparation")
                result = self._register_monthly_source_update(
                    state, accepted_update_id=accepted_id, accepted_at_utc=accepted_at_utc,
                    week_key=iso_week, data_slot=data_slot, monthly_input=transport)
                intent["status"] = "registered"
                intent["monthly_input"] = deepcopy(transport)
                self._save_unlocked(state, now_utc=now_utc)
                return {**result, "monthly_input": transport}

    def reconcile_monthly_inputs(self, *, now: datetime | None = None, monthly_input_request_provider=None):
        """Retry only saved original descriptors; never consult active publication."""
        state = self.load_state()
        results = []
        for intent in state.get("monthly_input_intents", {}).values():
            if intent.get("status") == "registered":
                continue
            context = intent["context"]
            request = None
            if not intent.get("request"):
                if monthly_input_request_provider is None:
                    continue
                try:
                    request = monthly_input_request_provider.request_for_claim(
                        context["claim_id"], context=context, policy_binding=intent.get("policy_binding"))
                except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
                    self.record_monthly_preparation_refusal(context["claim_id"], f"{type(exc).__name__}: {exc}", now=now)
                    results.append({"changed": False, "reason": "accepted_input_provisioning_refused", "error": str(exc)})
                    continue
                if request is None:
                    continue
            results.append(self.register_accepted_update(
                accepted_update_id=context["accepted_update_id"],
                accepted_at=_parse_utc_iso(context["accepted_at_utc"], field="accepted_at_utc"),
                iso_week=context["iso_week"], data_slot=context["data_slot"],
                claim_id=context["claim_id"], monthly_input_request=request, now=now))
        return {"changed": any(item["changed"] for item in results), "results": results}

    def _monthly_entry_for_week(
        self,
        state: dict[str, object],
        *,
        week_key: str,
    ) -> tuple[str, dict[str, object]] | None:
        monthly = state["monthly_candidates"]
        assert isinstance(monthly, dict)
        matches: list[tuple[str, dict[str, object], datetime]] = []
        for month_key, raw_entry in monthly.items():
            if not isinstance(raw_entry, dict):
                raise UFCUpdateScheduleStateError(
                    f"monthly_candidates.{month_key} must be an object"
                )
            source = raw_entry.get("source_update")
            if not isinstance(source, dict):
                raise UFCUpdateScheduleStateError(
                    f"monthly_candidates.{month_key}.source_update must be an object"
                )
            if str(source.get("iso_week") or "") != week_key:
                continue
            accepted_at = _parse_utc_iso(
                source.get("accepted_at_utc"),
                field=f"monthly_candidates.{month_key}.source_update.accepted_at_utc",
            )
            matches.append((str(month_key), raw_entry, accepted_at))
        if not matches:
            return None
        matches.sort(key=lambda item: (item[2], item[0]))
        if len(matches) > 1:
            raise UFCUpdateScheduleStateError(
                f"Multiple monthly candidates are anchored to ISO week {week_key}"
            )
        month_key, entry, _accepted_at = matches[0]
        return month_key, entry

    def _plan_monthly_candidate(
        self,
        state: dict[str, object],
        *,
        now_utc: datetime,
    ) -> dict[str, object]:
        week_key = self._week_key(now_utc)
        local_now = now_utc.astimezone(self.local_timezone)
        wednesday_due = self._slot_datetime(week_key, SLOT_WEDNESDAY)
        next_data = self._plan_data_update(state, now_utc=now_utc)
        fallback_next = (
            datetime.fromisoformat(str(next_data["next_due_at_local"]))
            if next_data.get("next_due_at_local")
            else None
        )
        matched = self._monthly_entry_for_week(state, week_key=week_key)
        if matched is None:
            return self._decision(
                action=ACTION_NONE,
                reason=(
                    "v7_promotion_month_not_configured"
                    if self._promotion_month_value is None
                    else "no_eligible_accepted_update_for_iso_week"
                ),
                now_utc=now_utc,
                week_key=week_key,
                slot=None,
                due_at_local=None,
                next_due_at_local=fallback_next,
                next_due_is_conditional=True,
            )

        month_key, entry = matched
        source = entry["source_update"]
        assert isinstance(source, dict)
        source_update_id = str(source.get("accepted_update_id") or "").strip()
        source_slot = str(source.get("data_slot") or "").strip()
        if not source_update_id:
            raise UFCUpdateScheduleStateError(
                f"monthly_candidates.{month_key}.source_update.accepted_update_id is missing"
            )
        if source_slot not in _SLOTS:
            raise UFCUpdateScheduleStateError(
                f"monthly_candidates.{month_key}.source_update.data_slot is invalid"
            )
        accepted_at_utc = _parse_utc_iso(
            source.get("accepted_at_utc"),
            field=f"monthly_candidates.{month_key}.source_update.accepted_at_utc",
        )
        accepted_at_local = accepted_at_utc.astimezone(self.local_timezone)
        status = str(entry.get("status") or "").strip()
        if status not in {
            "pending",
            "in_progress",
            "retry_pending",
            "succeeded",
            "failed",
        }:
            raise UFCUpdateScheduleStateError(
                f"monthly_candidates.{month_key}.status is invalid: {status!r}"
            )
        slots = self._slots(
            entry,
            context=f"monthly_candidates.{month_key}",
        )
        tuesday_status = self._attempt_status(
            slots,
            SLOT_TUESDAY,
            context=f"monthly_candidates.{month_key}",
        )
        wednesday_status = self._attempt_status(
            slots,
            SLOT_WEDNESDAY,
            context=f"monthly_candidates.{month_key}",
        )

        common = {
            "now_utc": now_utc,
            "week_key": week_key,
            "utc_month": month_key,
            "accepted_update_id": source_update_id,
            "accepted_at_utc": accepted_at_utc,
            "candidate_only": True,
            "monthly_input": deepcopy(source.get("monthly_input")),
            "monthly_context": deepcopy((source.get("monthly_input") or {}).get("context")),
        }
        if status == "succeeded":
            return self._decision(
                action=ACTION_NONE,
                reason="monthly_candidate_already_succeeded",
                slot=None,
                due_at_local=None,
                next_due_at_local=fallback_next,
                **common,
            )
        if status == "failed":
            return self._decision(
                action=ACTION_NONE,
                reason="monthly_candidate_retry_exhausted",
                slot=None,
                due_at_local=None,
                next_due_at_local=fallback_next,
                **common,
            )

        weekday = local_now.isoweekday()
        if source_slot == SLOT_TUESDAY:
            if tuesday_status is None:
                if (
                    weekday == UFC_UPDATE_TUESDAY_ISO_WEEKDAY
                    and local_now >= accepted_at_local
                ):
                    return self._decision(
                        action=ACTION_MONTHLY_CANDIDATE,
                        reason="monthly_candidate_due_after_tuesday_update",
                        slot=SLOT_TUESDAY,
                        due_at_local=accepted_at_local,
                        next_due_at_local=accepted_at_local,
                        **common,
                    )
                if (
                    weekday == UFC_UPDATE_WEDNESDAY_ISO_WEEKDAY
                    and local_now >= wednesday_due
                    and wednesday_status is None
                ):
                    return self._decision(
                        action=ACTION_MONTHLY_CANDIDATE,
                        reason="wednesday_candidate_only_retry_due_after_missed_tuesday_candidate",
                        slot=SLOT_WEDNESDAY,
                        due_at_local=wednesday_due,
                        next_due_at_local=wednesday_due,
                        **common,
                    )
                return self._decision(
                    action=ACTION_NONE,
                    reason=(
                        "waiting_for_wednesday_candidate_only_retry"
                        if weekday <= UFC_UPDATE_WEDNESDAY_ISO_WEEKDAY
                        else "monthly_candidate_window_complete"
                    ),
                    slot=None,
                    due_at_local=None,
                    next_due_at_local=(
                        wednesday_due
                        if weekday <= UFC_UPDATE_WEDNESDAY_ISO_WEEKDAY
                        else fallback_next
                    ),
                    next_due_is_conditional=(
                        weekday <= UFC_UPDATE_WEDNESDAY_ISO_WEEKDAY
                    ),
                    **common,
                )
            if tuesday_status == STATUS_CLAIMED:
                return self._decision(
                    action=ACTION_NONE,
                    reason="tuesday_monthly_candidate_in_progress",
                    slot=SLOT_TUESDAY,
                    due_at_local=accepted_at_local,
                    next_due_at_local=wednesday_due,
                    next_due_is_conditional=True,
                    **common,
                )
            if tuesday_status == STATUS_FAILED:
                if (
                    weekday == UFC_UPDATE_WEDNESDAY_ISO_WEEKDAY
                    and local_now >= wednesday_due
                    and wednesday_status is None
                ):
                    return self._decision(
                        action=ACTION_MONTHLY_CANDIDATE,
                        reason="wednesday_candidate_only_retry_due",
                        slot=SLOT_WEDNESDAY,
                        due_at_local=wednesday_due,
                        next_due_at_local=wednesday_due,
                        **common,
                    )
                if weekday < UFC_UPDATE_WEDNESDAY_ISO_WEEKDAY or (
                    weekday == UFC_UPDATE_WEDNESDAY_ISO_WEEKDAY
                    and local_now < wednesday_due
                ):
                    return self._decision(
                        action=ACTION_NONE,
                        reason="waiting_for_wednesday_candidate_only_retry",
                        slot=None,
                        due_at_local=None,
                        next_due_at_local=wednesday_due,
                        next_due_is_conditional=True,
                        **common,
                    )
            if wednesday_status == STATUS_CLAIMED:
                return self._decision(
                    action=ACTION_NONE,
                    reason="wednesday_monthly_candidate_in_progress",
                    slot=SLOT_WEDNESDAY,
                    due_at_local=wednesday_due,
                    next_due_at_local=fallback_next,
                    **common,
                )

        if source_slot == SLOT_WEDNESDAY:
            if (
                wednesday_status is None
                and weekday == UFC_UPDATE_WEDNESDAY_ISO_WEEKDAY
                and local_now >= accepted_at_local
            ):
                return self._decision(
                    action=ACTION_MONTHLY_CANDIDATE,
                    reason="monthly_candidate_due_after_wednesday_update",
                    slot=SLOT_WEDNESDAY,
                    due_at_local=accepted_at_local,
                    next_due_at_local=accepted_at_local,
                    **common,
                )
            if wednesday_status == STATUS_CLAIMED:
                return self._decision(
                    action=ACTION_NONE,
                    reason="wednesday_monthly_candidate_in_progress",
                    slot=SLOT_WEDNESDAY,
                    due_at_local=accepted_at_local,
                    next_due_at_local=fallback_next,
                    **common,
                )

        return self._decision(
            action=ACTION_NONE,
            reason="monthly_candidate_window_complete",
            slot=None,
            due_at_local=None,
            next_due_at_local=fallback_next,
            **common,
        )

    def plan_monthly_candidate(
        self,
        *,
        now: datetime | None = None,
    ) -> dict[str, object]:
        """Return the candidate-only decision without claiming training work."""
        now_utc = self._now(now)
        with self._locked_state() as state:
            return self._plan_monthly_candidate(state, now_utc=now_utc)

    def claim_monthly_candidate(
        self,
        *,
        now: datetime | None = None,
    ) -> dict[str, object]:
        """Atomically claim a due candidate attempt; never starts training."""
        now_utc = self._now(now)
        with self._locked_state() as state:
            decision = self._plan_monthly_candidate(state, now_utc=now_utc)
            if decision["action"] != ACTION_MONTHLY_CANDIDATE:
                return {**decision, "claimed": False, "claim_id": None}
            month_key = str(decision["utc_month"])
            slot = str(decision["slot"])
            monthly = state["monthly_candidates"]
            assert isinstance(monthly, dict)
            entry = monthly.get(month_key)
            if not isinstance(entry, dict):
                raise UFCUpdateScheduleStateError(
                    f"monthly_candidates.{month_key} must be an object"
                )
            slots = entry.setdefault("slots", {})
            if not isinstance(slots, dict):
                raise UFCUpdateScheduleStateError(
                    f"monthly_candidates.{month_key}.slots must be an object"
                )
            if slot == SLOT_WEDNESDAY and SLOT_TUESDAY not in slots:
                missed_tuesday_due = self._slot_datetime(
                    str(decision["iso_week"]),
                    SLOT_TUESDAY,
                )
                slots[SLOT_TUESDAY] = {
                    "status": STATUS_FAILED,
                    "claim_id": None,
                    "due_at_utc": _utc_iso(missed_tuesday_due),
                    "completed_at_utc": _utc_iso(now_utc),
                    "error": "tuesday_candidate_slot_interrupted_or_missed",
                    "synthetic": True,
                }
            claim_id = str(self._claim_id_factory() or "").strip()
            if not claim_id:
                raise UFCUpdateScheduleTransitionError(
                    "claim_id_factory returned an empty claim ID"
                )
            slots[slot] = {
                "status": STATUS_CLAIMED,
                "claim_id": claim_id,
                "due_at_utc": decision["due_at_utc"],
                "claimed_at_utc": _utc_iso(now_utc),
            }
            entry["status"] = "in_progress"
            self._save_unlocked(state, now_utc=now_utc)
            return {**decision, "claimed": True, "claim_id": claim_id}

    def complete_monthly_candidate(
        self,
        claim_id: str,
        *,
        succeeded: bool,
        candidate_id: str | None = None,
        error: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, object]:
        """Record candidate evaluation status without publishing a live model."""
        claim_id = str(claim_id or "").strip()
        if not claim_id:
            raise ValueError("claim_id must not be empty")
        candidate_id_value = str(candidate_id or "").strip()
        if succeeded and not candidate_id_value:
            raise ValueError("candidate_id is required for a successful candidate")
        now_utc = self._now(now)
        with self._locked_state() as state:
            monthly = state["monthly_candidates"]
            assert isinstance(monthly, dict)
            month_key, slot, entry, attempt = self._locate_claim(
                monthly,
                claim_id,
                context="monthly_candidates",
            )
            status = str(attempt.get("status") or "")
            expected_status = STATUS_SUCCEEDED if succeeded else STATUS_FAILED
            if status in {STATUS_SUCCEEDED, STATUS_FAILED}:
                if status != expected_status:
                    raise UFCUpdateScheduleTransitionError(
                        f"Candidate claim {claim_id!r} is already {status}, not {expected_status}"
                    )
                if (
                    succeeded
                    and str(attempt.get("candidate_id") or "") != candidate_id_value
                ):
                    raise UFCUpdateScheduleTransitionError(
                        f"Candidate claim {claim_id!r} already completed for a different candidate"
                    )
                return {
                    "changed": False,
                    "reason": "candidate_result_already_recorded",
                    "utc_month": month_key,
                    "slot": slot,
                    "succeeded": succeeded,
                    "next": self._plan_monthly_candidate(state, now_utc=now_utc),
                }
            if status != STATUS_CLAIMED:
                raise UFCUpdateScheduleTransitionError(
                    f"Candidate claim {claim_id!r} is not in progress"
                )

            attempt["status"] = expected_status
            attempt["completed_at_utc"] = _utc_iso(now_utc)
            if succeeded:
                attempt["candidate_id"] = candidate_id_value
                entry["status"] = "succeeded"
                entry["successful_candidate_id"] = candidate_id_value
                entry["succeeded_at_utc"] = _utc_iso(now_utc)
            else:
                attempt["error"] = str(error or "candidate_failed")
                entry["status"] = "retry_pending" if slot == SLOT_TUESDAY else "failed"

            self._save_unlocked(state, now_utc=now_utc)
            return {
                "changed": True,
                "reason": "candidate_result_recorded",
                "utc_month": month_key,
                "slot": slot,
                "succeeded": succeeded,
                "next": self._plan_monthly_candidate(state, now_utc=now_utc),
            }

    def status(self, *, now: datetime | None = None) -> dict[str, object]:
        """Return durable state plus both integration decisions in one read."""
        now_utc = self._now(now)
        with self._locked_state() as state:
            return {
                "state_path": str(self.state_path),
                "state": deepcopy(state),
                "data_update": self._plan_data_update(state, now_utc=now_utc),
                "monthly_candidate": self._plan_monthly_candidate(
                    state,
                    now_utc=now_utc,
                ),
            }


__all__ = [
    "ACTION_DATA_UPDATE",
    "ACTION_MONTHLY_CANDIDATE",
    "ACTION_NONE",
    "SLOT_TUESDAY",
    "SLOT_WEDNESDAY",
    "STATE_SCHEMA_VERSION",
    "UFCUpdateSchedule",
    "UFCUpdateScheduleError",
    "UFCUpdateScheduleStateError",
    "UFCUpdateScheduleTransitionError",
    "iso_week_identity",
    "utc_month_identity",
]
