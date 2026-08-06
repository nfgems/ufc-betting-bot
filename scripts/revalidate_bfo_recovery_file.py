"""Recompute a BFO recovery CSV with paired, sportsbook-only consensus prices.

Rows that cannot be confirmed from at least three consistent sportsbook pairs
are removed from the recovered output and written to a quarantine CSV.  This
keeps the project's core invariant: an unavailable value becomes NaN rather
than a plausible-looking default or contaminated market price.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.recover_bfo_moneyline_gaps import (
    MIN_PAIRED_SPORTSBOOKS,
    build_bfo_provenance_record,
    default_provenance_path,
    find_odds_from_event,
    sportsbook_consensus_details,
    write_jsonl_atomically,
)
from src.data.io_utils import write_csv_atomically


DEFAULT_MAX_REMOVAL_FRACTION = 0.05
SYSTEMIC_OPERATIONAL_FAILURE_FRACTION = 0.50
CONSENSUS_REJECTION = "insufficient_or_inconsistent_sportsbook_consensus"


class RevalidationSafetyError(RuntimeError):
    """Raised before corrected data is published when a safety gate fails."""


def _row_text(value) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def revalidate_recovery_frame(
    frame: pd.DataFrame,
    *,
    preserve_legacy_observations: bool = False,
    provenance_records: list[dict[str, object]] | None = None,
    input_batch: str = "",
    output_batch: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corrected_rows: list[dict[str, object]] = []
    quarantine_rows: list[dict[str, object]] = []

    for row in frame.to_dict(orient="records"):
        source_url = _row_text(row.get("source_url"))
        fighter_a = _row_text(row.get("fighter_a"))
        fighter_b = _row_text(row.get("fighter_b"))
        raw_event_date = row.get("event_date")
        try:
            event_date = pd.Timestamp(raw_event_date).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            event_date = ""
        observation: dict[str, object] = {
            "requested_fighters": {
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
            }
        }
        paired_quotes: list[dict[str, object]] = []
        consensus = None
        provenance_reason = ""
        error = ""
        try:
            odds = (
                find_odds_from_event(
                    source_url,
                    fighter_a,
                    fighter_b,
                    event_date,
                    observation=observation,
                )
                if source_url and fighter_a and fighter_b and event_date
                else None
            )
            if odds is not None:
                consensus, paired_quotes, provenance_reason = sportsbook_consensus_details(
                    odds[2],
                    odds[3],
                )
            if odds is None:
                error = str(observation.get("match_error") or CONSENSUS_REJECTION)
                provenance_reason = error
            elif consensus is None:
                error = CONSENSUS_REJECTION
        except Exception as exc:
            consensus = None
            error = f"{type(exc).__name__}: {exc}"
            provenance_reason = error

        if consensus is None:
            rejected = dict(row)
            rejected["revalidation_error"] = error
            quarantine_rows.append(rejected)
            original_book_count = pd.to_numeric(
                pd.Series([row.get("num_bookmakers")]),
                errors="coerce",
            ).iloc[0]
            is_manual_source = not source_url.lower().startswith(("http://", "https://"))
            if preserve_legacy_observations and (
                is_manual_source
                or (pd.notna(original_book_count) and int(original_book_count) < MIN_PAIRED_SPORTSBOOKS)
            ):
                corrected_rows.append(dict(row))
                decision = "preserved_legacy"
            else:
                decision = "rejected"
            if provenance_records is not None:
                provenance_records.append(
                    build_bfo_provenance_record(
                        row,
                        input_batch=input_batch,
                        output_batch=output_batch,
                        event_url=source_url,
                        observation=observation,
                        paired_quotes=paired_quotes,
                        consensus=consensus,
                        decision=decision,
                        rejection_reason=provenance_reason or error,
                    )
                )
            continue

        corrected = dict(row)
        corrected.update(
            {
                "a_fair_prob": round(float(consensus["a_fair_prob"]), 6),
                "b_fair_prob": round(float(consensus["b_fair_prob"]), 6),
                "a_decimal_odds": round(float(consensus["a_decimal_odds"]), 6),
                "b_decimal_odds": round(float(consensus["b_decimal_odds"]), 6),
                "num_bookmakers": int(consensus["num_bookmakers"]),
            }
        )
        corrected_rows.append(corrected)
        if provenance_records is not None:
            provenance_records.append(
                build_bfo_provenance_record(
                    corrected,
                    input_batch=input_batch,
                    output_batch=output_batch,
                    event_url=source_url,
                    observation=observation,
                    paired_quotes=paired_quotes,
                    consensus=consensus,
                    decision="accepted",
                    rejection_reason="",
                )
            )

    return (
        pd.DataFrame(corrected_rows, columns=frame.columns),
        pd.DataFrame(quarantine_rows, columns=[*frame.columns, "revalidation_error"]),
    )


def _resolved_distinct_paths(
    input_path: Path,
    output_path: Path,
    quarantine_path: Path,
    provenance_path: Path,
) -> tuple[Path, Path, Path, Path]:
    resolved = tuple(
        path.expanduser().resolve(strict=False)
        for path in (input_path, output_path, quarantine_path, provenance_path)
    )
    if len(set(resolved)) != 4:
        raise RevalidationSafetyError(
            "input, corrected output, quarantine output, and provenance output "
            "must resolve to four "
            "distinct paths; in-place revalidation is not permitted"
        )
    return resolved


def _safety_issues(
    source: pd.DataFrame,
    corrected: pd.DataFrame,
    quarantined: pd.DataFrame,
) -> list[str]:
    source_rows = len(source)
    corrected_rows = len(corrected)
    quarantine_rows = len(quarantined)
    issues: list[str] = []

    if source_rows == 0:
        issues.append("source input is empty")
    elif corrected_rows == 0:
        issues.append("corrected output would be empty")
    else:
        removed_rows = max(0, source_rows - corrected_rows)
        removed_fraction = removed_rows / source_rows
        if removed_fraction > DEFAULT_MAX_REMOVAL_FRACTION:
            issues.append(
                f"corrected output would remove {removed_rows}/{source_rows} rows "
                f"({removed_fraction:.1%}), exceeding the "
                f"{DEFAULT_MAX_REMOVAL_FRACTION:.1%} safety limit"
            )

    if source_rows and quarantine_rows == source_rows:
        issues.append("every source row failed revalidation")

    if source_rows and "revalidation_error" in quarantined.columns:
        errors = quarantined["revalidation_error"].fillna("").astype(str)
        operational_failures = int((errors != CONSENSUS_REJECTION).sum())
        operational_fraction = operational_failures / source_rows
        if operational_fraction >= SYSTEMIC_OPERATIONAL_FAILURE_FRACTION:
            issues.append(
                f"operational fetch/parser failures affected {operational_failures}/"
                f"{source_rows} rows ({operational_fraction:.1%})"
            )

    return issues


def _write_and_verify_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    refuse_empty: bool,
) -> None:
    """Atomically write a CSV and verify its exact serialized table round trip."""
    write_csv_atomically(frame, path, refuse_empty=refuse_empty)
    try:
        expected = pd.read_csv(
            io.StringIO(frame.to_csv(index=False)),
            dtype=str,
            keep_default_na=False,
        )
        observed = pd.read_csv(path, dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(expected, observed, check_dtype=True)
    except Exception as exc:
        raise RevalidationSafetyError(
            f"failed to verify written CSV {path}: {type(exc).__name__}: {exc}"
        ) from exc


def _write_and_verify_provenance(
    records: list[dict[str, object]],
    path: Path,
) -> None:
    try:
        write_jsonl_atomically(records, path)
    except Exception as exc:
        raise RevalidationSafetyError(
            f"failed to write or verify provenance JSONL {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def publish_revalidation(
    input_path: Path,
    output_path: Path,
    quarantine_path: Path,
    *,
    provenance_path: Path | None = None,
    preserve_legacy_observations: bool = False,
    allow_unsafe_output: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Revalidate and safely publish distinct quarantine and corrected artifacts.

    The source input is never overwritten.  Quarantine is durably written and
    read-verified before the corrected output is published.
    """
    provenance_path = provenance_path or default_provenance_path(output_path)
    input_path, output_path, quarantine_path, provenance_path = _resolved_distinct_paths(
        input_path,
        output_path,
        quarantine_path,
        provenance_path,
    )
    source = pd.read_csv(input_path)
    provenance_records: list[dict[str, object]] = []
    corrected, quarantined = revalidate_recovery_frame(
        source,
        preserve_legacy_observations=preserve_legacy_observations,
        provenance_records=provenance_records,
        input_batch=str(input_path),
        output_batch=str(output_path),
    )
    issues = _safety_issues(source, corrected, quarantined)
    if issues and not allow_unsafe_output:
        raise RevalidationSafetyError("; ".join(issues))

    # Quarantine must exist and round-trip successfully before corrected data
    # can be published.  A corrected-output failure therefore never destroys
    # the source, and rejected rows remain available for diagnosis.
    _write_and_verify_csv(quarantined, quarantine_path, refuse_empty=False)
    _write_and_verify_provenance(provenance_records, provenance_path)
    _write_and_verify_csv(
        corrected,
        output_path,
        refuse_empty=not allow_unsafe_output,
    )
    return corrected, quarantined, issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Corrected recovery CSV. Must be distinct from input and quarantine output.",
    )
    parser.add_argument(
        "--quarantine-output",
        type=Path,
        required=True,
        help="Rejected-row CSV. Must be distinct from input and corrected output.",
    )
    parser.add_argument(
        "--provenance-output",
        type=Path,
        help=(
            "Companion JSONL quote ledger. Defaults beside --output with a "
            ".provenance.jsonl suffix."
        ),
    )
    parser.add_argument(
        "--preserve-legacy-observations",
        action="store_true",
        help=(
            "Keep manually sourced rows and old one/two-book observations when they "
            "cannot be revalidated. They are still copied to quarantine for review."
        ),
    )
    parser.add_argument(
        "--allow-unsafe-output",
        action="store_true",
        help=(
            "Explicitly publish despite systemic, empty-output, or material-shrink "
            "safety gates. The source path still cannot be overwritten."
        ),
    )
    args = parser.parse_args(argv)

    try:
        corrected, quarantined, issues = publish_revalidation(
            args.input,
            args.output,
            args.quarantine_output,
            provenance_path=args.provenance_output,
            preserve_legacy_observations=args.preserve_legacy_observations,
            allow_unsafe_output=args.allow_unsafe_output,
        )
    except (OSError, pd.errors.ParserError, RevalidationSafetyError) as exc:
        print(f"Revalidation aborted: {exc}", file=sys.stderr)
        return 1

    if issues:
        print(
            "WARNING: --allow-unsafe-output bypassed safety gates: " + "; ".join(issues),
            file=sys.stderr,
        )
    print(
        f"{args.input}: corrected={len(corrected)} quarantined={len(quarantined)} "
        f"output={args.output} provenance="
        f"{args.provenance_output or default_provenance_path(args.output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
