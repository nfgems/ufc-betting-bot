"""Bounded read-only lookup of externally accepted original monthly requests.

Configuration supplies original pins. Neither this registry nor a current
publication can approve a new input, policy, or output location.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path

from src.model import prepared_input as prepared

INDEX_KIND = "accepted_monthly_request_index_v1"
MAX_INDEX_BYTES = 1024 * 1024
MAX_REQUESTS = 128
MAX_DESCRIPTOR_BYTES = 1024 * 1024


class MonthlyProvisioningRefusal(ValueError):
    """Independent original request authority is absent or contradictory."""


def _require(condition, message):
    if not condition:
        raise MonthlyProvisioningRefusal(message)


def _capture(pin, maximum):
    prepared._pin(pin)
    _require(pin["bytes"] <= maximum, "monthly provisioning record exceeds byte limit")
    path = prepared._path(pin["path"])
    _require(path.stat().st_size == pin["bytes"], "monthly provisioning record length changed")
    with path.open("rb") as stream:
        body = stream.read(maximum + 1)
    _require(len(body) <= maximum, "monthly provisioning record exceeds byte limit")
    return prepared._check_bytes(body, pin)


def _configured_pin(environ, prefix):
    values = [str(environ.get(prefix + suffix, "") or "").strip()
              for suffix in ("_PATH", "_BYTES", "_SHA256")]
    if not any(values):
        return None
    _require(all(values), f"{prefix} requires independent path, byte count and SHA-256")
    _require(values[1].isascii() and values[1].isdigit(), "invalid configured monthly byte count")
    return prepared._pin({"path": values[0], "bytes": int(values[1]), "sha256": values[2]})


class AcceptedMonthlyRequestProvider:
    """One immutable externally pinned finite request index; no directory scan."""

    def __init__(self, index_pin):
        self.index_pin = deepcopy(prepared._pin(index_pin))
        document = prepared._json(_capture(self.index_pin, MAX_INDEX_BYTES))
        _require(isinstance(document, dict) and set(document) == {"schema", "kind", "requests"}
                 and type(document["schema"]) is int and document["schema"] == 1
                 and document["kind"] == INDEX_KIND, "invalid accepted monthly request index")
        rows = document["requests"]
        _require(isinstance(rows, list) and len(rows) <= MAX_REQUESTS, "monthly request index limit")
        from src.model.monthly_input import _context
        self._requests = {}
        for row in rows:
            _require(isinstance(row, dict) and set(row) == {"context", "request", "policy"},
                     "invalid original monthly request entry")
            context = _context(row["context"])
            request = row["request"]
            _require(isinstance(request, dict) and set(request) == {"acceptance", "output_root"},
                     "invalid provisioned monthly request")
            prepared._pin(request["acceptance"])
            prepared._pin(row["policy"])
            _require(isinstance(request["output_root"], str) and Path(request["output_root"]).is_absolute(),
                     "provisioned output root must be explicit and absolute")
            prepared._path(request["output_root"])
            _require(context["claim_id"] and context["claim_id"] not in self._requests,
                     "duplicate or missing original monthly claim identity")
            self._requests[context["claim_id"]] = deepcopy(row)

    def request_for_claim(self, claim_id, *, context=None, completion_pin=None, policy_binding=None):
        """Validate only the matching original descriptor and its semantic completion."""
        row = self._requests.get(claim_id)
        if row is None:
            return None
        if context is not None:
            _require(row["context"] == context, "provisioned original monthly context conflicts")
        request = row["request"]
        descriptor = prepared._json(_capture(request["acceptance"], MAX_DESCRIPTOR_BYTES))
        _require(isinstance(descriptor, dict) and descriptor.get("schema") == 2
                 and descriptor.get("policy") == row["policy"],
                 "provisioned descriptor and independently pinned policy conflict")
        from src.model.monthly_policy import load_monthly_policy
        policy = load_monthly_policy(row["policy"])
        if policy_binding is not None:
            _require(policy.binding == policy_binding, "provisioned policy differs from original claim")
        _require(policy.promotion_month_utc == row["context"]["promotion_month_utc"],
                 "provisioned policy adoption month conflicts")
        if completion_pin is not None:
            _require(descriptor.get("completion") == completion_pin, "provisioned original completion pin conflicts")
        from src.model.monthly_input import inspect_monthly_completion
        _require(inspect_monthly_completion(request) == row["context"],
                 "provisioned original completion context conflicts")
        return deepcopy(request)

    def __call__(self, claim, summary):
        """Ordinary web completion callback, invoked after durable data success."""
        row = self._requests.get(claim.get("claim_id"))
        if row is None:
            return None
        context = row["context"]
        receipt = (summary or {}).get("data_publication_receipt") or {}
        _require(claim.get("iso_week") == context["iso_week"] and claim.get("slot") == context["data_slot"],
                 "provisioned original claim calendar conflicts")
        _require(receipt.get("accepted_generation_id") == context["accepted_update_id"],
                 "provisioned original generation conflicts")
        from src.model.monthly_input import _time
        _require(_time(receipt.get("accepted_at_utc")) == _time(context["accepted_at_utc"]),
                 "provisioned original completion cutoff conflicts")
        return self.request_for_claim(claim["claim_id"],
            policy_binding=(claim.get("monthly_policy") or {}).get("binding"))

    def reconcile(self, schedule, *, now=None):
        """Recover original protected claims, then saved/pending monthly intents."""
        state = schedule.load_state()
        pending = [attempt for entry in state["weekly"].values()
                   for attempt in entry.get("slots", {}).values()
                   if attempt.get("status") == "claimed" and attempt.get("completion_evidence_required") is True]
        _require(len(pending) <= MAX_REQUESTS, "original completion recovery limit")
        results = []
        for attempt in pending:
            try:
                request = self.request_for_claim(attempt["claim_id"],
                    policy_binding=(attempt.get("monthly_policy") or {}).get("binding"))
                if request is None:
                    continue
                context = self._requests[attempt["claim_id"]]["context"]
                result = schedule.complete_data_update(context["claim_id"], succeeded=True,
                    accepted_update_id=context["accepted_update_id"],
                    accepted_at=datetime.fromisoformat(context["accepted_at_utc"].replace("Z", "+00:00")),
                    monthly_input_request=request, expected_monthly_context=context, now=now)
                results.append(result)
            except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
                results.append({"changed": False, "claim_id": attempt["claim_id"],
                                "error": f"{type(exc).__name__}: {exc}",
                                "reason": "original_completion_authority_refused"})
        monthly = schedule.reconcile_monthly_inputs(now=now, monthly_input_request_provider=self)
        results.extend(monthly["results"])
        return {"changed": any(item.get("changed") or (item.get("monthly_registration") or {}).get("changed")
                               for item in results), "results": results,
                "reason": "accepted_original_requests_reconciled"}


def build_monthly_provisioning_from_env(environ=None):
    """Resolve optional explicit pins; absent configuration creates no paths."""
    source = os.environ if environ is None else environ
    policy_pin = _configured_pin(source, "UFC_MONTHLY_POLICY")
    index_pin = _configured_pin(source, "UFC_MONTHLY_REQUEST_INDEX")
    _require(index_pin is None or policy_pin is not None,
             "accepted request provisioning requires an explicit monthly policy pin")
    from src.model.monthly_policy import load_monthly_policy
    policy = load_monthly_policy(policy_pin) if policy_pin is not None else None
    provider = AcceptedMonthlyRequestProvider(index_pin) if index_pin is not None else None
    return {"policy": policy, "provider": provider}


def load_configured_monthly_policy(environ=None):
    source = os.environ if environ is None else environ
    pin = _configured_pin(source, "UFC_MONTHLY_POLICY")
    from src.model.monthly_policy import load_monthly_policy
    return load_monthly_policy(pin) if pin is not None else None


class UnavailableMonthlyRequestProvider:
    """A configuration refusal disables monthly provisioning, never data work."""

    def __init__(self, error):
        self.error = str(error)

    def request_for_claim(self, *args, **kwargs):
        raise MonthlyProvisioningRefusal(self.error)

    def __call__(self, *args, **kwargs):
        raise MonthlyProvisioningRefusal(self.error)

    def reconcile(self, schedule, *, now=None):
        saved = schedule.reconcile_monthly_inputs(now=now)
        return {**saved, "reason": "monthly_provisioning_configuration_refused", "error": self.error}


def configure_monthly_calendar_from_env(environ=None):
    """Keep the ordinary data calendar available when monthly configuration refuses."""
    source = os.environ if environ is None else environ
    configured = any(str(source.get("UFC_MONTHLY_" + part + suffix, "") or "").strip()
                     for part in ("POLICY", "REQUEST_INDEX") for suffix in ("_PATH", "_BYTES", "_SHA256"))
    policy = None
    try:
        policy_pin = _configured_pin(source, "UFC_MONTHLY_POLICY")
        from src.model.monthly_policy import load_monthly_policy
        policy = load_monthly_policy(policy_pin) if policy_pin is not None else None
        result = build_monthly_provisioning_from_env(source)
        return {**result, "configured": configured, "error": ""}
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        return {"policy": policy, "provider": UnavailableMonthlyRequestProvider(error),
                "configured": configured, "error": error}
