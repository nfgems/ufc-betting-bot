"""Local startup routing through independent inspection and existing recovery.

The native launcher supplies custody and an explicitly provisioned original
controller binding plus its external hash. It must establish the existing
network/process and ownership boundary before calling this module. This does
not provision authority, start helpers, select history or admit interrupted
requests. Original recovery validation remains authoritative.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.data.roster_hosted_inspection import inspect_retained_state
from src.data.roster_hosted_store import _require, verify_pin


def startup_retained_controller(store, binding_pin, *, expected_binding_sha256,
                                custodian, synthetic_helper=None,
                                clock=lambda: datetime.now(timezone.utc)):
    """Pause uncertainty, otherwise ask existing recovery to validate originals.

    Independent accounting inspection is not semantic recovery permission.
    Recovery obtains the exact latest checkpoint again from custody and retains
    all original lifetime, calendar, ancestry, budget and consumed-stage checks.
    A completed intermediate receipt can still require a later collection;
    only a proved final required collection permits a no-broker demonstration.
    """
    verify_pin(binding_pin)
    _require(binding_pin["sha256"] == expected_binding_sha256,
             "External startup controller binding hash mismatch")
    inspection = inspect_retained_state(store, custodian=custodian, clock=clock)
    if inspection["uncertain_requests"]:
        return dict(inspection, startup_route="paused_uncertain_originals_retained")
    from src.data.roster_hosted_controller import recover_retained_controller
    result = recover_retained_controller(store, binding_pin,
        expected_binding_sha256=expected_binding_sha256, custodian=custodian,
        synthetic_helper=synthetic_helper, clock=clock)
    return dict(result, startup_route="existing_retained_controller_recovery",
                startup_inspection=inspection)
