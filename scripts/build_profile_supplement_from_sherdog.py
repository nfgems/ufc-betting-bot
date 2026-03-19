"""
Build a local supplemental fighter-profile artifact from Sherdog.

This is an offline enrichment step for static fields that UFCStats leaves blank.
The output is a small CSV keyed by the original UFCStats fighter name and is
safe to merge only into blank fields during training-data construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.fallback_scrapers import scrape_sherdog_page, search_sherdog  # noqa: E402
from src.data.name_utils import normalize_person_name  # noqa: E402


DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "ufc_fighters_scraped.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw" / "ufc_fighters_profile_supplement.csv"
TARGET_FIELDS = ("height", "weight", "dob")
TARGET_GAP_FIELDS = {"height", "weight", "age"}
SEARCH_NAME_OVERRIDES = {
    "felix lee mitchell": "Felix Mitchell",
}


def _blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() in {"", "--", "N/A", "nan"}


def _load_candidates(scraped_fighters_path: Path, gap_audit_csv: Path | None = None) -> pd.DataFrame:
    scraped_df = pd.read_csv(scraped_fighters_path)
    if scraped_df.empty or "name" not in scraped_df.columns:
        raise SystemExit(f"Scraped fighter profile artifact is empty or missing 'name': {scraped_fighters_path}")

    if gap_audit_csv is not None:
        gap_df = pd.read_csv(gap_audit_csv)
        names = sorted(
            {
                name
                for _, row in gap_df.iterrows()
                if row.get("field") in TARGET_GAP_FIELDS and not _blank(row.get("fighter_name"))
                for name in [row.get("fighter_name")]
            }
        )
        return scraped_df[scraped_df["name"].isin(names)].copy()

    mask = pd.Series(False, index=scraped_df.index)
    for field in TARGET_FIELDS:
        if field not in scraped_df.columns:
            continue
        mask = mask | scraped_df[field].apply(_blank)
    return scraped_df[mask].copy()


def _build_supplement_row(scraped_row: pd.Series) -> dict[str, object] | None:
    fighter_name = str(scraped_row.get("name", "") or "").strip()
    if not fighter_name:
        return None

    search_name = SEARCH_NAME_OVERRIDES.get(normalize_person_name(fighter_name), fighter_name)
    sherdog_url = search_sherdog(search_name)
    if not sherdog_url:
        return None

    profile, _ = scrape_sherdog_page(sherdog_url, search_name)
    supplement = {
        "name": fighter_name,
        "source": "sherdog",
        "source_name": profile.get("name", ""),
        "search_name": search_name,
        "fighter_url": sherdog_url,
        "height": "",
        "reach": "",
        "weight": "",
        "stance": "",
        "dob": "",
    }

    recovered_any = False
    if _blank(scraped_row.get("height")) and not _blank(profile.get("height_raw")):
        supplement["height"] = profile.get("height_raw", "")
        recovered_any = True
    if _blank(scraped_row.get("weight")) and not _blank(profile.get("weight_raw")):
        supplement["weight"] = profile.get("weight_raw", "")
        recovered_any = True
    if _blank(scraped_row.get("dob")) and not _blank(profile.get("dob")):
        supplement["dob"] = profile.get("dob", "")
        recovered_any = True

    return supplement if recovered_any else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scraped-fighters-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--gap-audit-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    candidates = _load_candidates(args.scraped_fighters_path, gap_audit_csv=args.gap_audit_csv)
    candidates = candidates.sort_values("name").reset_index(drop=True)
    if args.limit is not None:
        candidates = candidates.head(args.limit).copy()

    existing_rows: list[dict[str, object]] = []
    existing_names: set[str] = set()
    if args.output.exists():
        existing_df = pd.read_csv(args.output)
        existing_rows = existing_df.to_dict(orient="records")
        existing_names = {
            normalize_person_name(row.get("name"))
            for row in existing_rows
            if not _blank(row.get("name"))
        }

    results: list[dict[str, object]] = []
    attempted = 0
    recovered = 0
    skipped_existing = 0
    for _, row in candidates.iterrows():
        name_key = normalize_person_name(row.get("name"))
        if name_key in existing_names:
            skipped_existing += 1
            continue
        attempted += 1
        supplement = _build_supplement_row(row)
        if supplement is None:
            continue
        results.append(supplement)
        recovered += 1

    combined_rows = existing_rows + results
    output_df = pd.DataFrame(combined_rows)
    if not output_df.empty:
        output_df = output_df.sort_values(["name", "source"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)

    summary = {
        "candidate_rows": int(len(candidates)),
        "attempted_rows": attempted,
        "skipped_existing_rows": skipped_existing,
        "recovered_rows": recovered,
        "output_path": str(args.output),
    }
    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Candidates: {summary['candidate_rows']}")
    print(f"Attempted: {summary['attempted_rows']}")
    print(f"Skipped existing: {summary['skipped_existing_rows']}")
    print(f"Recovered: {summary['recovered_rows']}")
    print(f"Output: {summary['output_path']}")


if __name__ == "__main__":
    main()
