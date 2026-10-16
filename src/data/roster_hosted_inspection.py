"""Read-only accounting diagnosis through already provisioned custody.

This reports retained accounting, including charged uncertainty. It never
admits recovery: semantic inputs, original authority lifetimes, calendar and
clock continuity still require their existing recovery validation. The caller
must keep the real parent guard and independent custody ownership boundary.
"""
from __future__ import annotations

import json

from src.data.roster_hosted_inputs import _checkpoint
from src.data.roster_hosted_store import _read, _require, _time, verify_pin


def inspect_retained_state(store, *, custodian, clock=_time):
    """Inspect the exact independent current checkpoint without writing state.

    A retained checkpoint may be an ancestor of newer independently committed
    accounting. That lag is reported separately from uncertainty. Neither an
    empty uncertainty list nor this observation authorizes work or replay.
    Uncommitted intent, missing evidence and independent-head drift raise the
    existing error; they are never converted to an apparently coherent report.
    The default uses real UTC; local controller API tests may supply their same
    continuous fixture clock. The connected CLI has no clock override.
    """
    _require(store.mutation_guard is custodian
             and getattr(custodian, "strict_checkpoint_ack", False),
             "Inspection requires connected independent custody")
    current = custodian.current()
    state = store.snapshot()
    checkpoint = current["checkpoint"]
    # The independent owner validates its retained chain; snapshot validates
    # every accounting event, claim, charge and retained receipt. Recheck the
    # exact checkpoint ancestry, without treating the oldest binding as latest.
    retained = json.loads(_read(verify_pin(checkpoint)["path"]))
    _checkpoint(store, state, retained["accepted_pointer"], checkpoint,
                checkpoint["sha256"], _time(clock()))
    _require(current["current_store_head"] == state["head"]
             and current["retained_store_head"] == retained["store_head"]
             and current["unretained_store_tail"] is (retained["store_head"] != state["head"]),
             "Independent inspection accounting heads disagree")
    confirmed = custodian.check(state["head"])
    _require(confirmed["store_head"] == state["head"],
             "Independent current accounting confirmation changed")
    _require(store.snapshot() == state and custodian.current() == current,
             "Saved accounting changed during read-only inspection")
    uncertain = sorted(state["uncertain_requests"])
    return {
        "schema": 1,
        "status": ("paused_uncertain_delivery_execution_disabled" if uncertain
                   else "saved_accounting_inspected_execution_disabled"),
        "checkpoint": checkpoint,
        "retained_store_head": current["retained_store_head"],
        "current_store_head": state["head"],
        "retention_head": current["retention_head"],
        "custody_ack": current["custody_ack"],
        "unretained_store_tail": current["unretained_store_tail"],
        "total_charged": state["total_charged"],
        "rosterwatch_charged": state["rosterwatch_charged"],
        "uncertain_requests": uncertain,
        "pending_requests": sorted(state["pending_requests"]),
        "consumed_plans": sorted(state["consumed_plans"]),
        "accounting_inspected": True,
        "read_only": True,
        "semantic_recovery_validated": False,
        "clock_continuity_validated": False,
        "recovery_authorized": False,
        "replay_authorized": False,
        "automatic_retry_authorized": False,
        "source_requests": 0,
        "activation_authorized": False,
        "publication_authorized": False,
        "schedule_enabled": False,
        "scope": "Current independently retained accounting only; no recovery admission",
    }
