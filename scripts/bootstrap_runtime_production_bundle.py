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
    get_processed_snapshot_max_event_date,
    reconcile_production_bundle_manifest,
    runtime_manifest_needs_source_bootstrap,
)


def _processed_snapshot_exists(processed_dir: Path) -> bool:
    return all((processed_dir / filename).is_file() for filename in ("fights_cleaned.csv", "features.csv"))


def _copy_processed_snapshot(*, source_processed_dir: Path, target_processed_dir: Path) -> None:
    for filename in ("fights_cleaned.csv", "features.csv"):
        copy_file_atomically(
            source_processed_dir / filename,
            target_processed_dir / filename,
        )


def _runtime_snapshot_is_newer(*, source_processed_dir: Path, target_processed_dir: Path) -> bool:
    source_date = get_processed_snapshot_max_event_date(source_processed_dir)
    target_date = get_processed_snapshot_max_event_date(target_processed_dir)
    if not target_date:
        return False
    if not source_date:
        return True
    return target_date > source_date


def bootstrap_runtime_production_bundle(
    *,
    target_manifest: Path,
    source_manifest: Path,
    source_processed_dir: Path,
    target_processed_dir: Path,
    model_path: Path,
    no_odds_model_path: Path,
    logistic_model_path: Path | None = None,
) -> dict[str, object]:
    needs_source_bootstrap = runtime_manifest_needs_source_bootstrap(
        target_manifest_path=target_manifest,
        source_manifest_path=source_manifest,
    )
    bootstrap_action = "reused_existing_runtime_bundle"
    if needs_source_bootstrap:
        source_snapshot_exists = _processed_snapshot_exists(source_processed_dir)
        target_snapshot_exists = _processed_snapshot_exists(target_processed_dir)
        if target_manifest.exists() and target_snapshot_exists:
            bootstrap_action = "adopted_existing_runtime_snapshot"
        elif target_snapshot_exists and _runtime_snapshot_is_newer(
            source_processed_dir=source_processed_dir,
            target_processed_dir=target_processed_dir,
        ):
            bootstrap_action = "adopted_existing_runtime_snapshot"
        elif source_snapshot_exists:
            _copy_processed_snapshot(
                source_processed_dir=source_processed_dir,
                target_processed_dir=target_processed_dir,
            )
            bootstrap_action = "promoted_source_bundle"
        elif target_snapshot_exists:
            bootstrap_action = "adopted_existing_runtime_snapshot"
        else:
            missing_files = [
                str(source_processed_dir / filename)
                for filename in ("fights_cleaned.csv", "features.csv")
            ]
            raise FileNotFoundError(
                "Cannot bootstrap production bundle: canonical processed snapshot is missing "
                f"from image and runtime volume. Missing files: {', '.join(missing_files)}"
            )

    summary = reconcile_production_bundle_manifest(
        target_manifest_path=target_manifest,
        source_manifest_path=source_manifest,
        model_path=model_path,
        no_odds_model_path=no_odds_model_path,
        logistic_model_path=logistic_model_path,
        processed_dir=target_processed_dir,
    )
    summary["bootstrap_action"] = bootstrap_action
    return summary


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

    summary = bootstrap_runtime_production_bundle(
        target_manifest=args.target_manifest,
        source_manifest=args.source_manifest,
        source_processed_dir=args.source_processed_dir,
        target_processed_dir=args.target_processed_dir,
        model_path=args.model_path,
        no_odds_model_path=args.no_odds_model_path,
        logistic_model_path=args.logistic_model_path,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
