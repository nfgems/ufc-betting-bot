"""Finite parent-side collection with durable evidence and no default network.

The only supported execution mode is synthetic local preparation. A future
live transport and Linux socket boundary require separate implementation review
and authorization. Neither this module nor its mailbox activates source access.
The worker supplies no URL, method, headers or challenge solution: all sends
are derived by this trusted parent from a sealed finite plan and saved replies.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
import time

from src.data.roster_hosted_store import StoreError, assess_source_freshness, verify_pin
from src.data.ufcstats_identity_access import (
    POST_URL, _cookie, _document_supported, _document_url, _solve, _template,
    header_safety, is_challenge,
)

MEMBERSHIP_URL = "https://www.roster.watch/"


class BrokerError(StoreError):
    pass


def require(ok, reason):
    if not ok:
        raise BrokerError(reason)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def stamp(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(isinstance(value, datetime) and value.tzinfo is not None, "Aware UTC timestamp required")
    return value.astimezone(timezone.utc)


class SourceBroker:
    """One claimed source plan; fake transport is explicitly required for now.

    Plan kind roster_source_collection_v1: publication_requested=false,
    execution_mode=synthetic, membership bool, documents unique UFCStats URLs,
    challenge {enabled, template_pin when enabled}, issued_at/expires_at,
    observed_at, user_agent, and max_requests (documents + membership + two
    optional challenge sends). Pin freshness concerns planning input only;
    returned evidence keeps the actual per-response time.
    """

    def __init__(self, store, plan_pin, review_pin, *, synthetic_wire,
                 protected_verifier, clock=lambda: datetime.now(timezone.utc),
                 monotonic=time.monotonic, sleeper=time.sleep):
        require(callable(synthetic_wire) and callable(protected_verifier),
                "Explicit synthetic wire and protected verifier required")
        verify_pin(plan_pin)
        verify_pin(review_pin)
        self.plan = json.loads(Path(plan_pin["path"]).read_bytes())
        review = json.loads(Path(review_pin["path"]).read_bytes())
        plan = self.plan
        require(review.get("status") == "passed" and review.get("blockers") == []
                and review.get("plan") == plan_pin, "Exact passed plan review required")
        require(plan.get("kind") == "roster_source_collection_v1"
                and plan.get("execution_mode") == "synthetic"
                and plan.get("publication_requested") is False, "Only unpublished synthetic preparation supported")
        require(re.fullmatch(r"[A-Za-z0-9_-]{1,70}", str(plan.get("plan_id", ""))), "Invalid plan id")
        require(plan.get("store_root") == store.root.as_posix(), "Source plan is bound to another accounting store")
        require(store.claimed_authority(plan["plan_id"]) == {
            "plan_id": plan["plan_id"], "plan_pin": plan_pin, "review_pin": review_pin},
            "Collection plan must bind durable claim")
        require(type(plan.get("membership")) is bool and isinstance(plan.get("documents"), list),
                "Explicit membership/document scope required")
        docs = plan["documents"]
        require(all(isinstance(u, str) and _document_url(u) for u in docs)
                and len(set(docs)) == len(docs), "Unsupported or duplicate document")
        challenge = plan.get("challenge", {})
        require(type(challenge.get("enabled")) is bool, "Explicit challenge scope required")
        require(not challenge["enabled"] or docs, "Challenge requires a document")
        self.template = None
        if challenge["enabled"]:
            verify_pin(challenge["template_pin"])
            self.template = Path(challenge["template_pin"]["path"]).read_bytes()
            _template(self.template)
        ceiling = len(docs) + int(plan["membership"]) + 2 * int(challenge["enabled"])
        state = store.snapshot()
        require(type(plan.get("max_requests")) is int and plan["max_requests"] == ceiling
                and 0 < ceiling <= state["policy"]["plan_ceiling"], "Exact finite maximum required")
        require(state["total_charged"] + ceiling <= state["policy"]["total_ceiling"]
                and state["rosterwatch_charged"] + int(plan["membership"]) <= state["policy"]["rosterwatch_ceiling"],
                "Insufficient cumulative budget for optional branches")
        require(not state["uncertain_requests"], "Uncertain earlier send blocks collection")
        prefix = plan["plan_id"] + "-"
        require(not any(k.startswith(prefix) for k in state["results"]), "Collection already started; never resume")
        ua = plan.get("user_agent")
        require(isinstance(ua, str) and 1 <= len(ua) <= 256 and ua.isascii()
                and all(32 <= ord(c) < 127 for c in ua) and "mozilla/" not in ua.lower(), "Identifying user agent required")
        require(0 < (stamp(plan["expires_at"]) - stamp(plan["issued_at"])).total_seconds() <= 86400,
                "Plan lifetime exceeds one day")
        self.store, self.plan_pin, self.review_pin = store, deepcopy(plan_pin), deepcopy(review_pin)
        self.wire, self.verifier, self.clock = synthetic_wire, protected_verifier, clock
        self.monotonic, self.sleeper = monotonic, sleeper
        self.max_age = state["policy"]["max_source_age_seconds"]
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.started = False
        self.last_start = None
        self.records, self.captures = [], []
        self._authority()

    def _authority(self):
        require(not self.cancelled.is_set(), "Parent cancelled collection")
        verify_pin(self.plan_pin)
        verify_pin(self.review_pin)
        require(json.loads(Path(self.plan_pin["path"]).read_bytes()) == self.plan, "Plan memory drift")
        if self.template is not None:
            verify_pin(self.plan["challenge"]["template_pin"])
        now = stamp(self.clock())
        require(stamp(self.plan["issued_at"]) <= now < stamp(self.plan["expires_at"]), "Expired or future collection plan")
        require(assess_source_freshness(self.plan["observed_at"], now=now,
                max_age_seconds=self.max_age)["eligible"], "Stale planning observation")
        self.verifier()
        return now

    def cancel(self):
        self.cancelled.set()

    def _send(self, method, url, *, form=None, cookie=None):
        require(len(self.records) < self.plan["max_requests"], "Collection ceiling reached")
        if self.last_start is not None:
            wait = 2.2 - (self.monotonic() - self.last_start)
            if wait > 0:
                self.sleeper(wait)
        now = self._authority()
        request_id = self.plan["plan_id"] + "-" + str(len(self.records) + 1)
        headers = {"User-Agent": self.plan["user_agent"], "Accept": "text/html",
                   "Accept-Encoding": "identity", "Connection": "close"}
        if form is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        if cookie:
            headers["Cookie"] = cookie
        source = "roster.watch" if url == MEMBERSHIP_URL else "ufcstats.com"
        # A failed fsync never reaches the callback. A crash after this point
        # leaves a charged pending intent, so successors cannot silently retry.
        self.store.charge_request(self.plan["plan_id"], request_id, source, method, url, now=now)
        self.last_start = self.monotonic()
        try:
            result = self.wire(method, url, dict(headers), form, deepcopy(self.plan))
        except Exception as exc:
            result = {"status": None, "headers": [], "body": b"", "complete": False,
                      "error": type(exc).__name__}
        require(isinstance(result, dict) and isinstance(result.get("body"), bytes), "Malformed wire response; charged intent remains pending")
        body = result["body"]
        raw_response = result.get("raw_response")
        require(raw_response is None or isinstance(raw_response, bytes) and len(raw_response) <= 8388609,
                "Wire violated raw HTTP response bound; intent remains pending")
        # Preserve a bounded oversized response as unknown, including the
        # overflow witness. A larger malformed callback leaves a pending charge.
        require(len(body) <= 8388609, "Wire violated bounded response contract; intent remains pending")
        response_headers = result.get("headers", [])
        require(isinstance(response_headers, (list, tuple)) and len(response_headers) <= 100
                and all(isinstance(r, (list, tuple)) and len(r) == 2
                and all(isinstance(v, str) and len(v) <= 8192 and "\r" not in v and "\n" not in v for v in r)
                for r in response_headers), "Malformed response metadata; intent remains pending")
        complete = result.get("complete") is True and not result.get("error")
        error = result.get("error")
        try:
            require(len(body) <= 8388608, "Response byte ceiling")
            header_safety(response_headers, len(body))
            require(type(result.get("status")) is int and 100 <= result["status"] <= 599, "Invalid response status")
        except (StoreError, RuntimeError) as exc:
            complete, error = False, str(exc)
        finished = stamp(self.clock())
        envelope = {"schema_version": 1, "plan_id": self.plan['plan_id'],
            "plan": deepcopy(self.plan_pin), "review": deepcopy(self.review_pin),
            "request": {"request_id": request_id,
            "method": method, "url": url, "headers": list(map(list, headers.items())),
            "body_sha256": sha(form) if form is not None else None,
            "body_base64": base64.b64encode(form).decode() if form is not None else None,
            "started_at": now.isoformat()},
            "response": {"status": result.get("status"), "headers": list(map(list, response_headers)),
                "body_sha256": sha(body), "completed_at": finished.isoformat(),
                "complete": bool(complete), "error": error,
                "wire_metadata": deepcopy(result.get("wire_metadata", {}))}}
        receipt = self.store.record_result(request_id, "complete" if complete else "unknown",
                                           body, envelope=envelope, raw_response=raw_response, now=finished)
        self.records.append({"request_id": request_id, "envelope": envelope, "receipt": receipt})
        require(complete, "Incomplete delivery; source run stopped with charge retained")
        require(finished < stamp(self.plan["expires_at"]), "Response finished after authority expiry")
        return {**result, "headers": response_headers, "envelope": envelope, "receipt": receipt}

    def collect(self):
        require(self.lock.acquire(blocking=False), "Concurrent collection forbidden")
        try:
            require(not self.started, "Collection is single use")
            self.started = True
            self._authority()
            membership, rounds = None, 0
            if self.plan["membership"]:
                result = self._send("GET", MEMBERSHIP_URL)
                require(result["status"] == 200 and not is_challenge(result["body"]), "Membership status/challenge rejected; no redirect or retry")
                media = [v.split(";", 1)[0].strip().lower() for k, v in result["headers"] if k.lower() == "content-type"]
                require(media == ["text/html"], "Membership media rejected")
                result["body"].decode("utf-8", errors="strict")
                membership = {"raw": result["receipt"]["blob"], "envelope": result["envelope"]}
            cookie = None
            for url in self.plan["documents"]:
                result = self._send("GET", url, cookie=cookie)
                self._capture(url, result)
                require(result["status"] == 200, "Document status rejected; no redirect or retry")
                if is_challenge(result["body"]):
                    require(self.template is not None and rounds == 0, "Challenge disabled or recurring")
                    form = _solve(result["body"], self.template)
                    rounds += 1
                    posted = self._send("POST", POST_URL, form=form)
                    require(200 <= posted["status"] < 300, "Challenge POST rejected")
                    cookie = _cookie(posted["headers"])
                    result = self._send("GET", url, cookie=cookie)
                    self._capture(url, result)
                require(result["status"] == 200 and _document_supported(url, result["body"], result["headers"]), "Unsupported document or repeated challenge")
                result["body"].decode("utf-8", errors="strict")
            self._authority()
            return {"status": "passed", "execution_mode": "synthetic", "source_requests": 0,
                    "synthetic_charges": len(self.records), "publication_requested": False,
                    "membership": membership, "catalog": {"schema_version": 1, "captures": self.captures},
                    "records": deepcopy(self.records), "challenge_rounds": rounds,
                    "unused_optional_requests": self.plan["max_requests"] - len(self.records)}
        finally:
            self.lock.release()

    def _capture(self, url, result):
        body = result["body"]
        try:
            body.decode("utf-8", errors="strict")
            decoded = True
        except UnicodeDecodeError:
            decoded = False
        self.captures.append({"url": url, "status_code": result["status"],
            "body_base64": base64.b64encode(body).decode(), "decoded_sha256": sha(body),
            "decoded_size": len(body), "captured_at": result["envelope"]["response"]["completed_at"],
            "capture_complete": True, "decode_complete": decoded,
            "ufcstats_challenge_detected": is_challenge(body),
            "semantic_evidence_eligible": decoded and result["status"] == 200
                and _document_supported(url, body, result["headers"]), "error": result.get("error")})
