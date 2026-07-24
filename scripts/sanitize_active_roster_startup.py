"""Sanitize the persisted UFC active roster before hosted runtime startup.

The runtime volume is the authoritative existing snapshot.  Image seeding is
only for a missing volume file; row count is not a freshness or quality signal.
This pass removes identities that the latest persisted identity audit verified
as inactive and quarantines rows without a positively verified current status.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.io_utils import write_csv_atomically  # noqa: E402
from src.data.ufc_active_roster import (  # noqa: E402
    ACTIVE_ROSTER_SYNC_GENERATION_COLUMN,
    _apply_profile_status_eligibility,
    _inactive_profile_status,
    _roster_rows_same_identity,
    _validate_cached_roster_snapshot,
    _verified_inactive_profile_status,
    retained_roster_diagnostics,
)

logger = logging.getLogger(__name__)

_EXCLUDED_AUDIT_ACTIONS = frozenset(
    {"excluded_inactive_profile_status", "excluded_test_profile"}
)


def _load_valid_roster_snapshot(path: Path, *, label: str) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"{label} is missing: {path}")
    try:
        roster_df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"{label} is unreadable: {path}: {exc}") from exc
    try:
        _validate_cached_roster_snapshot(roster_df)
    except Exception as exc:
        raise RuntimeError(f"{label} is invalid: {path}: {exc}") from exc
    return roster_df


def _accepted_roster_generation_id(roster_df: pd.DataFrame) -> str:
    if ACTIVE_ROSTER_SYNC_GENERATION_COLUMN not in roster_df.columns:
        return ""
    generation_values = [
        ""
        if pd.isna(value) or str(value).strip().casefold() == "nan"
        else str(value).strip()
        for value in roster_df[ACTIVE_ROSTER_SYNC_GENERATION_COLUMN].tolist()
    ]
    # An accepted write stamps every row. A partially stamped/mixed file is not
    # proof that a sidecar audit belongs to this exact roster artifact.
    if not generation_values or any(not value for value in generation_values):
        return ""
    generations = set(generation_values)
    return next(iter(generations)) if len(generations) == 1 else ""


def _audit_excluded_identity_rows(
    identity_audit_path: Path | None,
    *,
    accepted_generation_id: str,
) -> list[dict[str, object]]:
    if identity_audit_path is None or not identity_audit_path.exists():
        return []
    if not accepted_generation_id:
        return []
    try:
        audit_df = pd.read_csv(identity_audit_path)
    except Exception as exc:
        # The sidecar is auxiliary evidence. An unreadable file cannot prove an
        # exclusion, but it must not make an otherwise valid roster unbootable.
        logger.warning(
            "Ignoring unreadable UFC active-roster identity audit %s: %s",
            identity_audit_path,
            exc,
        )
        return []
    if ACTIVE_ROSTER_SYNC_GENERATION_COLUMN not in audit_df.columns:
        logger.warning(
            "Ignoring UFC active-roster identity audit without %s: %s",
            ACTIVE_ROSTER_SYNC_GENERATION_COLUMN,
            identity_audit_path,
        )
        return []
    excluded: list[dict[str, object]] = []
    for _, row in audit_df.iterrows():
        if (
            str(row.get(ACTIVE_ROSTER_SYNC_GENERATION_COLUMN) or "").strip()
            != accepted_generation_id
        ):
            continue
        if str(row.get("action") or "").strip() not in _EXCLUDED_AUDIT_ACTIONS:
            continue
        excluded.append(row.to_dict())
    return excluded


def sanitize_active_roster_startup(
    *,
    roster_path: Path,
    identity_audit_path: Path | None = None,
    fallback_roster_path: Path | None = None,
) -> dict[str, object]:
    recovered_from_fallback = False
    persisted_roster_load_error = ""
    try:
        roster_df = _load_valid_roster_snapshot(
            roster_path,
            label="Persisted UFC active roster",
        )
    except Exception as persisted_exc:
        persisted_roster_load_error = str(persisted_exc)
        if fallback_roster_path is None:
            raise RuntimeError(
                f"{persisted_roster_load_error}; no fallback roster was configured"
            ) from persisted_exc
        try:
            roster_df = _load_valid_roster_snapshot(
                fallback_roster_path,
                label="Image fallback UFC active roster",
            )
        except Exception as fallback_exc:
            raise RuntimeError(
                f"{persisted_roster_load_error}; image fallback cannot be used: "
                f"{fallback_exc}"
            ) from persisted_exc
        recovered_from_fallback = True
        logger.warning(
            "Recovering invalid persisted UFC active roster %s from validated image "
            "fallback %s: %s",
            roster_path,
            fallback_roster_path,
            persisted_roster_load_error,
        )

    rows_before = int(len(roster_df))
    accepted_generation_id = _accepted_roster_generation_id(roster_df)
    excluded_rows = _audit_excluded_identity_rows(
        identity_audit_path,
        accepted_generation_id=accepted_generation_id,
    )
    keep_mask: list[bool] = []
    removed_verified_inactive = 0
    removed_by_identity_audit = 0
    for _, row in roster_df.iterrows():
        identity_excluded = any(
            _roster_rows_same_identity(row, excluded_row)
            for excluded_row in excluded_rows
        )
        status_inactive = _verified_inactive_profile_status(row)
        keep = not identity_excluded and not status_inactive
        keep_mask.append(keep)
        if not keep:
            if identity_excluded:
                removed_by_identity_audit += 1
            else:
                removed_verified_inactive += 1

    sanitized = roster_df.loc[keep_mask].copy().reset_index(drop=True)
    sanitized = _apply_profile_status_eligibility(sanitized)
    if sanitized.empty:
        raise RuntimeError(
            "Startup roster sanitation would remove every persisted UFC active-roster row"
        )

    unknown_status_rows = int(
        sanitized["profile_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq("status_unknown")
        .sum()
    )
    quarantined_unverified_inactive_rows = int(
        sanitized.apply(
            lambda row: _inactive_profile_status(row)
            and not _verified_inactive_profile_status(row),
            axis=1,
        ).sum()
    )
    retained_rows = retained_roster_diagnostics(sanitized)

    comparable_before = roster_df.fillna("").astype(str)
    comparable_after = sanitized.fillna("").astype(str)
    changed = recovered_from_fallback or not comparable_before.equals(comparable_after)
    if changed:
        write_csv_atomically(sanitized, roster_path, refuse_empty=True)

    action = (
        "recovered_from_fallback"
        if recovered_from_fallback
        else ("updated" if changed else "unchanged")
    )
    return {
        "action": action,
        "path": str(roster_path),
        "rows_before": rows_before,
        "rows_after": int(len(sanitized)),
        "removed_by_identity_audit": removed_by_identity_audit,
        "removed_verified_inactive_status": removed_verified_inactive,
        "quarantined_unverified_inactive_status": quarantined_unverified_inactive_rows,
        "accepted_sync_generation_id": accepted_generation_id,
        "status_unknown_rows": unknown_status_rows,
        "retained_missing_live_rows": int(len(retained_rows)),
        "fallback_recovery_used": recovered_from_fallback,
        "fallback_roster_path": (
            str(fallback_roster_path)
            if recovered_from_fallback and fallback_roster_path is not None
            else ""
        ),
        "persisted_roster_load_error": persisted_roster_load_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--identity-audit", type=Path, default=None)
    parser.add_argument("--fallback-roster", type=Path, default=None)
    args = parser.parse_args()
    summary = sanitize_active_roster_startup(
        roster_path=args.roster,
        identity_audit_path=args.identity_audit,
        fallback_roster_path=args.fallback_roster,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
