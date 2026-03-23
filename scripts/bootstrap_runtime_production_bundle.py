"""Bootstrap the hosted runtime manifest and canonical processed snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.io_utils import copy_file_atomically
from src.model.production_bundle import (
    reconcile_production_bundle_manifest,
    runtime_manifest_needs_source_bootstrap,
)


def _copy_processed_snapshot(*, source_processed_dir: Path, target_processed_dir: Path) -> None:
    for filename in ("fights_cleaned.csv", "features.csv"):
        copy_file_atomically(
            source_processed_dir / filename,
            target_processed_dir / filename,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-processed-dir", type=Path, required=True)
    parser.add_argument("--target-processed-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--no-odds-model-path", type=Path, required=True)
    parser.add_argument("--logistic-model-path", type=Path, default=None)
    args = parser.parse_args()

    promoted_source_bundle = runtime_manifest_needs_source_bootstrap(
        target_manifest_path=args.target_manifest,
        source_manifest_path=args.source_manifest,
    )
    if promoted_source_bundle:
        _copy_processed_snapshot(
            source_processed_dir=args.source_processed_dir,
            target_processed_dir=args.target_processed_dir,
        )

    summary = reconcile_production_bundle_manifest(
        target_manifest_path=args.target_manifest,
        source_manifest_path=args.source_manifest,
        model_path=args.model_path,
        no_odds_model_path=args.no_odds_model_path,
        logistic_model_path=args.logistic_model_path,
        processed_dir=args.target_processed_dir,
    )
    summary["bootstrap_action"] = (
        "promoted_source_bundle" if promoted_source_bundle else "reused_existing_runtime_bundle"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
