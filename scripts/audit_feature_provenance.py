"""Audit feature-by-feature training provenance for a UFC training spec."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.feature_provenance import audit_training_feature_provenance, resolve_spec_from_name  # noqa: E402


def _print_text_report(payload: dict) -> None:
    print(f"Spec: {payload['spec_name']}")
    print(f"Dataset variant: {payload['dataset_variant']}")
    print(f"Train cutoff date: {payload['train_cutoff_date']}")
    print(
        "Rows: "
        f"features_total={payload['features_rows_total']} "
        f"trainable={payload['trainable_rows']} "
        f"train_split={payload['train_split_rows']} "
        f"test_split={payload['test_split_rows']}"
    )
    print("")
    print("Family summary:")
    for row in payload["family_summary"]:
        print(
            "  "
            f"{row['feature_family']}: "
            f"features={row['feature_count']} "
            f"non_null_rows={row['train_rows_with_any_non_null_feature']} "
            f"legacy_dep_rows={row['train_rows_with_any_legacy_dependency']} "
            f"repo_only_rows={row['train_rows_repo_only_all_non_null']} "
            f"default_only_rows={row['train_rows_default_or_seeded_only']}"
        )

    print("")
    print("Top legacy-dependent features:")
    top_rows = sorted(
        payload["inventory"],
        key=lambda row: (row["train_legacy_dependent_rows"], row["train_non_null_rows"]),
        reverse=True,
    )[:20]
    for row in top_rows:
        print(
            "  "
            f"{row['feature']}: "
            f"legacy_dep={row['train_legacy_dependent_rows']} "
            f"repo_only={row['train_repo_only_rows']} "
            f"mixed={row['train_mixed_rows']} "
            f"legacy_only={row['train_legacy_only_rows']} "
            f"default_or_seeded={row['train_default_or_seeded_rows']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="full_live_contract_v5")
    parser.add_argument("--min-fights", type=int, default=2)
    parser.add_argument("--json", action="store_true", help="Emit the full audit payload as JSON.")
    args = parser.parse_args()

    spec = resolve_spec_from_name(args.spec)
    result = audit_training_feature_provenance(spec, min_fights=args.min_fights)
    payload = result.to_dict()

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    _print_text_report(payload)


if __name__ == "__main__":
    main()
