"""Write the active production runtime manifest from the current live artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.production_bundle import reconcile_production_bundle_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-manifest",
        type=Path,
        default=None,
        help="Manifest path to write. Defaults to UFC_PRODUCTION_BUNDLE_MANIFEST or the repo default.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="Optional base manifest to preserve promotion metadata from.",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--no-odds-model-path", type=Path, default=None)
    parser.add_argument("--logistic-model-path", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = reconcile_production_bundle_manifest(
        target_manifest_path=args.target_manifest,
        source_manifest_path=args.source_manifest,
        model_path=args.model_path,
        no_odds_model_path=args.no_odds_model_path,
        logistic_model_path=args.logistic_model_path,
        processed_dir=args.processed_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
