"""Rebuild processed UFC fight and feature artifacts from the current raw data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.kaggle_loader import load_kaggle_dataset, save_processed
from src.data.ufc_refresh import TRAINING_DATASET_VARIANTS, build_training_dataset_variants
from src.features.build_features import build_features, save_features


def _resolve_output_dirs(values: list[str] | None) -> list[Path | None]:
    if not values:
        return [None]
    outputs: list[Path | None] = []
    for value in values:
        text = str(value or "").strip()
        if text in {"", ".", "base"}:
            outputs.append(None)
        else:
            outputs.append(Path(text))
    return outputs


def run_rebuild(
    *,
    dataset_variant: str = "pulled_all_plus_legacy_market",
    output_subdirs: list[str] | None = None,
) -> dict[str, object]:
    legacy_df = load_kaggle_dataset(RAW_DATA_DIR / "ufc-master.csv")
    variants = build_training_dataset_variants(legacy_df=legacy_df)
    fights_df = variants[dataset_variant].copy()
    features_df = build_features(fights_df)

    output_dirs = _resolve_output_dirs(output_subdirs)
    summaries: list[dict[str, object]] = []
    for subdir in output_dirs:
        fights_filename = Path("fights_cleaned.csv") if subdir is None else subdir / "fights_cleaned.csv"
        features_filename = Path("features.csv") if subdir is None else subdir / "features.csv"
        fights_path = save_processed(fights_df, filename=fights_filename)
        save_features(features_df, filename=features_filename)
        features_path = (PROCESSED_DATA_DIR / features_filename) if not Path(features_filename).is_absolute() else Path(features_filename)
        summaries.append(
            {
                "output_dir": str(PROCESSED_DATA_DIR if subdir is None else PROCESSED_DATA_DIR / subdir),
                "fights_path": str(fights_path),
                "features_path": str(features_path),
                "fight_rows": int(len(fights_df)),
                "feature_rows": int(len(features_df)),
                "feature_cols": int(len(features_df.columns)),
            }
        )

    return {
        "dataset_variant": dataset_variant,
        "outputs": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-variant",
        default="pulled_all_plus_legacy_market",
        choices=sorted(TRAINING_DATASET_VARIANTS),
        help="Which training-row universe to materialize before feature building.",
    )
    parser.add_argument(
        "--output-subdir",
        action="append",
        default=None,
        help="Optional processed output subdir to rebuild as well. Use '.' or 'base' for data/processed itself.",
    )
    args = parser.parse_args()

    print(json.dumps(run_rebuild(dataset_variant=args.dataset_variant, output_subdirs=args.output_subdir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
