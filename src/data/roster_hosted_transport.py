"""Accounted transport primitive for a future trusted roster source broker.

No default wire, network configuration, scheduling or application import exists.
The supervisor still owns owner authorization, process/socket/native confinement,
independent review semantics and the complete protected-state inventory. This
callable can occupy acquire_reviewed's existing explicit ``wire`` seam only after
the future supervisor has selected and reviewed that integration. Both journals
then describe the same sends: the source engine's local ledger is diagnostic and
MUST NOT also be reimported as new charges into this persistent accounting domain.
"""
from datetime import datetime, timezone
from copy import deepcopy
import json
import re
import threading
from types import MappingProxyType

from src.data.roster_hosted_store import StoreError, assess_source_freshness, verify_pin
from src.data.ufcstats_identity_access import POST_URL, _document_url


class DispatchError(StoreError):
    """No unreviewed or repeated source dispatch is permitted."""


def _require(value, reason):
    if not value:
        raise DispatchError(reason)


def _utc():
    return datetime.now(timezone.utc)


def _timestamp(value):
    try:
        value = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    except ValueError as exc:
        raise DispatchError("Invalid authority timestamp") from exc
    _require(isinstance(value, datetime) and value.tzinfo is not None, "Aware authority timestamp required")
    return value.astimezone(timezone.utc)


class AccountedDispatcher:
    """One finite sealed request sequence, durably charged before callback entry.

    The sealed plan must add a ``hosted_dispatch`` object with ``sequence`` rows
    (request_id, method, url), ``observed_at`` and ``expires_at``. Its ordinary
    ``requests`` and ``challenge`` fields retain the acquisition CLI schema.
    A new instance refuses any prior charge from this sequence: a crash never
    turns an unknown request into replay permission. A failure stops this object.
    ``clock`` is actual UTC in a real caller; injection is only for isolated tests.
    """

    def __init__(self, store, plan_pin, review_pin, *, wire, protected_verifier, clock=_utc):
        _require(callable(wire) and callable(protected_verifier) and callable(clock),
                 "Explicit trusted wire, protected verifier and clock required")
        verify_pin(plan_pin)
        verify_pin(review_pin)
        with open(plan_pin["path"], encoding="utf-8-sig") as stream:
            self.plan = json.load(stream)
        self.plan_id = self.plan["plan_id"]
        _require(store.claimed_authority(self.plan_id) ==
                 {"plan_id": self.plan_id, "plan_pin": plan_pin, "review_pin": review_pin},
                 "Dispatcher does not bind the exact durable plan claim")
        with open(review_pin["path"], encoding="utf-8-sig") as stream:
            review = json.load(stream)
        _require(review.get("status") == "passed" and review.get("blockers") == []
                 and review.get("plan") == plan_pin, "Review does not bind passed exact plan")
        control = self.plan.get("hosted_dispatch", {})
        self.sequence = control.get("sequence")
        state = store.snapshot()
        _require(not state["uncertain_requests"], "Uncertain prior source request; reviewed recovery required")
        _require(isinstance(self.sequence, list) and 0 < len(self.sequence) <= state["policy"]["plan_ceiling"],
                 "Finite bounded dispatch sequence required")
        ids, posts = set(), 0
        requested = {row["url"] for row in self.plan["requests"] if row.get("method") == "GET"}
        for row in self.sequence:
            _require(isinstance(row, dict) and set(row) == {"request_id", "method", "url"},
                     "Invalid sealed request row")
            _require(isinstance(row["request_id"], str) and
                     re.fullmatch(r"[A-Za-z0-9_-]{1,100}", row["request_id"])
                     and row["request_id"] not in ids, "Repeated or invalid request identity")
            ids.add(row["request_id"])
            if row["method"] == "POST":
                posts += 1
                _require(row["url"] == POST_URL, "Unexpected challenge endpoint")
            else:
                _require(row["method"] == "GET" and _document_url(row["url"])
                         and row["url"] in requested, "GET is outside sealed document requests")
        challenge = self.plan["challenge"]
        _require(posts <= 1 and (not posts or challenge.get("enabled") is True)
                 and challenge.get("max_rounds") == 1 and challenge.get("max_difficulty") == 2
                 and challenge.get("max_iterations") == 100000, "Challenge bounds changed")
        for index, row in enumerate(self.sequence):
            if row["method"] == "POST":
                _require(0 < index < len(self.sequence) - 1
                         and self.sequence[index - 1]["method"] == "GET"
                         and self.sequence[index + 1]["method"] == "GET"
                         and self.sequence[index - 1]["url"] == self.sequence[index + 1]["url"],
                         "Challenge must be bounded by its initial GET and same-document reload")
        _require(len(self.sequence) <= self.plan["send_ceiling"], "Sequence exceeds sealed send ceiling")
        _require(not ids.intersection(set(state["pending_requests"]) | set(state["results"])),
                 "Sequence already charged; never replay or resume it")
        self.store, self.plan_pin, self.review_pin = store, dict(plan_pin), dict(review_pin)
        self.wire, self.verifier, self.clock = wire, protected_verifier, clock
        self.observed_at, self.expires_at = control["observed_at"], _timestamp(control["expires_at"])
        self.max_age = state["policy"]["max_source_age_seconds"]
        self.sequence = tuple(MappingProxyType(dict(row)) for row in self.sequence)
        self._index, self._stopped, self._lock = 0, False, threading.Lock()
        self._authority()

    def _authority(self):
        verify_pin(self.plan_pin)
        verify_pin(self.review_pin)
        with open(self.plan_pin["path"], encoding="utf-8-sig") as stream:
            _require(json.load(stream) == self.plan, "In-memory sealed authority changed")
        now = _timestamp(self.clock())
        _require(now <= self.expires_at, "Sealed dispatch authority expired")
        freshness = assess_source_freshness(self.observed_at, now=now, max_age_seconds=self.max_age)
        _require(freshness["eligible"], "Source freshness gate: " + freshness["reason"])
        self.verifier()
        return now

    def __call__(self, method, url, headers, form, plan):
        _require(self._lock.acquire(blocking=False), "Concurrent dispatch forbidden")
        try:
            _require(not self._stopped and self._index < len(self.sequence), "Dispatch sequence closed")
            self._stopped = True  # Every exceptional exit consumes this object.
            now = self._authority()
            _require(plan == self.plan, "Acquisition plan changed")
            row = self.sequence[self._index]
            _require((method, url) == (row["method"], row["url"]), "Request differs from exact next intent")
            _require(isinstance(headers, dict) and all(isinstance(k, str) and isinstance(v, str)
                     and len(k) <= 100 and len(v) <= 8192 and "\r" not in k + v and "\n" not in k + v
                     for k, v in headers.items()), "Invalid request headers")
            if method == "GET":
                _require(form is None, "GET body forbidden")
            else:
                _require(isinstance(form, bytes) and re.fullmatch(rb"nonce=[a-f0-9]{16}&n=[0-9]{1,5}", form),
                         "Unbounded challenge form")
            self.store.charge_request(self.plan_id, row["request_id"], "ufcstats.com", method, url, now=now)
            try:
                result = self.wire(method, url, dict(headers), form, deepcopy(self.plan))
            except Exception as exc:
                result = {"body": b"", "headers": [], "status": None,
                          "complete": False, "error": type(exc).__name__}
            _require(isinstance(result, dict) and isinstance(result.get("body"), bytes)
                     and len(result["body"]) <= 8388608, "Malformed or over-limit wire response")
            complete = result.get("complete") is True and not result.get("error")
            # An exception or incomplete response cannot prove whether the
            # request reached the server. Keep its bytes but retain uncertainty
            # globally; neither a new object nor a new plan clears this gate.
            outcome = "complete" if complete else "unknown"
            self.store.record_result(row["request_id"], outcome, result["body"], now=_timestamp(self.clock()))
            self._index += 1
            self._stopped = not complete or self._index == len(self.sequence)
            return result
        finally:
            self._lock.release()
