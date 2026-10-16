"""Read explicitly pinned accepted Roster Watch membership for provider work.

The caller supplies the accepted generation's independent pin. Provider profile
status and upcoming-card names are never membership authorities here.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import re

import pandas as pd

from src.data.ufc_active_roster import ACTIVE_ROSTER_SYNC_GENERATION_COLUMN
from src.data.name_utils import normalize_ufcstats_id


def load_accepted_rosterwatch_membership(
    roster_path: Path, *, expected_sha256: str, expected_generation: str,
) -> pd.DataFrame:
    """Validate a caller-pinned accepted CSV without fetching or changing it."""
    path = Path(roster_path)
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Accepted membership path may not be redirected")
    return load_accepted_rosterwatch_membership_bytes(
        path.read_bytes(), expected_sha256=expected_sha256, expected_generation=expected_generation)


def load_accepted_rosterwatch_membership_bytes(
    raw: bytes, *, expected_sha256: str, expected_generation: str,
) -> pd.DataFrame:
    """Validate exactly the same accepted membership in a portable saved carrier."""
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256)) or not expected_generation:
        raise ValueError("Accepted membership requires an independent hash and generation")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Accepted membership bytes differ from their supplied pin")
    frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    required = {ACTIVE_ROSTER_SYNC_GENERATION_COLUMN, "active_roster_status_reason"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("Accepted Roster Watch membership is empty or incomplete")
    if set(frame[ACTIVE_ROSTER_SYNC_GENERATION_COLUMN]) != {expected_generation}:
        raise ValueError("Accepted membership has a different or mixed generation")
    if set(frame["active_roster_status_reason"]) != {"rosterwatch_authoritative_current"}:
        raise ValueError("Provider nominations require authoritative Roster Watch membership")
    identities = []
    for row in frame.to_dict("records"):
        explicit = normalize_ufcstats_id(row.get("stable_ufcstats_id"))
        linked = normalize_ufcstats_id(row.get("ufcstats_url"))
        if (row.get("stable_ufcstats_id") and not explicit) or (row.get("ufcstats_url") and not linked):
            raise ValueError("Accepted membership contains a malformed stable identity")
        if explicit and linked and explicit != linked:
            raise ValueError("Accepted membership contains conflicting stable identities")
        identities.append(explicit or linked or "")
    known = [identity for identity in identities if identity]
    if len(known) != len(set(known)):
        raise ValueError("Accepted membership repeats a stable fighter identity")
    frame["stable_ufcstats_id"] = identities
    # Preserve every member, including pending identities; only resolved members
    # can be attached to provider records. No provider-status intersection.
    frame["coverage_eligible"] = [bool(identity) for identity in identities]
    frame.attrs.update(membership_source="rosterwatch", accepted_sha256=expected_sha256,
                       accepted_generation=expected_generation)
    return frame
