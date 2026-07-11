"""Write the active production runtime manifest from the current live artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.io_utils import write_json_atomically
from src.model.production_bundle import (
    DEFAULT_MANIFEST_PATH,
    PRODUCTION_BUNDLE_ENV,
    _runtime_timestamp_now,
    reconcile_production_bundle_manifest,
)


def _resolve_target_manifest_path(target_manifest: Path | None) -> Path:
    import os

    if target_manifest is not None:
        return target_manifest
    env_value = os.environ.get(PRODUCTION_BUNDLE_ENV, "").strip()
    if env_value:
        return Path(env_value)
    return DEFAULT_MANIFEST_PATH


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
    parser.add_argument(
        "--set-built-at-now",
        action="store_true",
        help=(
            "Stamp a fresh built_at after reconciling. Required for same-spec refit "
            "promotions: reconcile intentionally preserves the prior built_at when the "
            "spec name is unchanged, which the runtime freshness guard reads."
        ),
    )
    args = parser.parse_args()

    summary = reconcile_production_bundle_manifest(
        target_manifest_path=args.target_manifest,
        source_manifest_path=args.source_manifest,
        model_path=args.model_path,
        no_odds_model_path=args.no_odds_model_path,
        logistic_model_path=args.logistic_model_path,
        processed_dir=args.processed_dir,
    )

    if args.set_built_at_now:
        manifest_path = _resolve_target_manifest_path(args.target_manifest)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        stamp = _runtime_timestamp_now()
        payload["built_at"] = stamp
        payload["manifest_updated_at"] = stamp
        write_json_atomically(payload, manifest_path)
        summary["built_at"] = stamp
        summary["manifest_updated_at"] = stamp

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
