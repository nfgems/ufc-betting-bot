"""Fail CI on known tracked artifact integrity regressions."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_ODDS_PATH = Path("data/raw/historical_odds/historical_odds.csv")


def _tracked_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line.strip()) for line in completed.stdout.splitlines() if line.strip()]


def _check_json_bom(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for rel_path in paths:
        if rel_path.suffix.lower() != ".json":
            continue
        abs_path = REPO_ROOT / rel_path
        if not abs_path.exists():
            continue
        try:
            if abs_path.read_bytes().startswith(b"\xef\xbb\xbf"):
                errors.append(f"{rel_path}: BOM-prefixed JSON is not allowed")
        except OSError as exc:
            errors.append(f"{rel_path}: unreadable JSON artifact ({exc})")
    return errors


def _check_historical_odds_duplicates(paths: list[Path]) -> list[str]:
    if HISTORICAL_ODDS_PATH not in paths:
        return []

    abs_path = REPO_ROOT / HISTORICAL_ODDS_PATH
    seen: set[tuple[str, str, str, str]] = set()
    dupes: list[tuple[int, tuple[str, str, str, str]]] = []
    with abs_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            key = (
                str(row.get("event_date", "")),
                str(row.get("fighter_a", "")),
                str(row.get("fighter_b", "")),
                str(row.get("offset_days", "")),
            )
            if key in seen:
                dupes.append((row_number, key))
            else:
                seen.add(key)

    return [
        (
            f"{HISTORICAL_ODDS_PATH}:{row_number}: duplicate "
            f"(event_date, fighter_a, fighter_b, offset_days)={key}"
        )
        for row_number, key in dupes
    ]


def _check_candidate_test_sets(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for rel_path in paths:
        if rel_path.parts[:3] != ("data", "processed", "candidates"):
            continue
        if rel_path.name != "test_set.csv":
            continue

        abs_path = REPO_ROOT / rel_path
        if not abs_path.exists():
            continue
        with abs_path.open("r", encoding="utf-8", newline="") as handle:
            row_count = sum(1 for _ in handle) - 1
        if row_count <= 0:
            errors.append(f"{rel_path}: tracked candidate test_set.csv must contain at least one data row")
    return errors


def main() -> int:
    tracked = _tracked_files()
    errors: list[str] = []
    errors.extend(_check_json_bom(tracked))
    errors.extend(_check_historical_odds_duplicates(tracked))
    errors.extend(_check_candidate_test_sets(tracked))

    if errors:
        print("Tracked artifact integrity check failed:", file=sys.stderr)
        for error in errors:
            print(f" - {error}", file=sys.stderr)
        return 1

    print("Tracked artifact integrity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
