"""Rebuild processed UFC fight and feature artifacts from the current raw data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.kaggle_loader import load_kaggle_dataset, save_processed
from src.data.ufc_refresh import TRAINING_DATASET_VARIANTS, build_training_dataset_variants
from src.features.build_features import build_features, save_features
from src.model.production_bundle import (
    _require_processed_identity_integrity,
    load_production_bundle_or_none,
    reconcile_production_bundle_manifest,
)


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
    update_production_manifest: bool = False,
    raw_data_dir: Path | None = None,
    processed_data_dir: Path | None = None,
    base_dataset_path: Path | None = None,
    fight_results_path: Path | None = None,
    fight_stats_path: Path | None = None,
    scraped_fighters_path: Path | None = None,
    supplemental_profiles_path: Path | None = None,
    official_active_roster_path: Path | None = None,
    pierce_fight_identities_path: Path | None = None,
    event_dates_cache_path: Path | None = None,
    historical_odds_dir: Path | None = None,
    rankings_history_path: Path | None = None,
    method_odds_history_path: Path | None = None,
    isolated_staging: bool = False,
) -> dict[str, object]:
    raw_root = Path(raw_data_dir) if raw_data_dir is not None else None
    raw_base = raw_root if raw_root is not None else RAW_DATA_DIR
    from src.data.provider_consumer_references import (
        provider_reference_inventory, snapshot_provider_references, record_provider_consumer_artifact, SUFFIX,
    )
    from src.data.observation_consumer_references import (
        freeze_observation_inputs, observation_reference_inventory, record_observation_consumer_artifact,
    )
    observation_inventory, observation_parents, observation_originals = freeze_observation_inputs(raw_base)
    provider_inventory = provider_reference_inventory(raw_base)
    provider_parents = []
    for relative, expected in provider_inventory.items():
        if relative.endswith(SUFFIX):
            snapshot = snapshot_provider_references(raw_base / relative[:-len(SUFFIX)])
            if snapshot is None or {key: snapshot["pin"][key] for key in ("bytes", "sha256")} != expected:
                raise ValueError("provider references changed while selecting rebuild inputs")
            provider_parents.append(snapshot)
    processed_base = (
        Path(processed_data_dir)
        if processed_data_dir is not None
        else PROCESSED_DATA_DIR
    )
    base_dataset_path = (
        Path(base_dataset_path)
        if base_dataset_path is not None
        else raw_base / "ufc-master.csv"
    )
    output_dirs = _resolve_output_dirs(output_subdirs)
    target_dirs = [
        processed_base
        if subdir is None
        else subdir if subdir.is_absolute() else processed_base / subdir
        for subdir in output_dirs
    ]
    canonical_dir = PROCESSED_DATA_DIR.resolve(strict=False)
    canonical_raw_dir = RAW_DATA_DIR.resolve(strict=False)

    def _same_or_within(path: Path, root: Path) -> bool:
        resolved = path.resolve(strict=False)
        resolved_root = root.resolve(strict=False)
        return resolved == resolved_root or resolved_root in resolved.parents

    if isolated_staging:
        if raw_data_dir is None or processed_data_dir is None:
            raise ValueError(
                "isolated staging requires explicit raw_data_dir and processed_data_dir"
            )
        raw_stage_root = raw_base.resolve(strict=False)
        processed_stage_root = processed_base.resolve(strict=False)
        if _same_or_within(raw_stage_root, canonical_raw_dir) or _same_or_within(
            canonical_raw_dir, raw_stage_root
        ):
            raise ValueError(
                "isolated staging raw_data_dir cannot overlap the canonical raw directory"
            )
        if _same_or_within(processed_stage_root, canonical_dir) or _same_or_within(
            canonical_dir, processed_stage_root
        ):
            raise ValueError(
                "isolated staging processed_data_dir cannot overlap the canonical processed directory"
            )
        if _same_or_within(raw_stage_root, processed_stage_root) or _same_or_within(
            processed_stage_root, raw_stage_root
        ):
            raise ValueError(
                "isolated staging raw_data_dir and processed_data_dir cannot overlap"
            )
        if update_production_manifest:
            raise ValueError(
                "isolated staging cannot update the production bundle manifest"
            )
        if any(
            not _same_or_within(target, processed_stage_root)
            for target in target_dirs
        ):
            raise ValueError(
                "isolated staging outputs must remain within processed_data_dir"
            )
        staged_raw_inputs = (
            base_dataset_path,
            Path(fight_results_path)
            if fight_results_path is not None
            else raw_base / "ufc-fight-results.csv",
            Path(fight_stats_path)
            if fight_stats_path is not None
            else raw_base / "ufc-fight-stats.csv",
            Path(scraped_fighters_path)
            if scraped_fighters_path is not None
            else raw_base / "ufc_fighters_scraped.csv",
            Path(supplemental_profiles_path)
            if supplemental_profiles_path is not None
            else raw_base / "ufc_fighters_profile_supplement.csv",
            Path(official_active_roster_path)
            if official_active_roster_path is not None
            else raw_base / "ufc_active_roster_official.csv",
            Path(pierce_fight_identities_path)
            if pierce_fight_identities_path is not None
            else raw_base / "github_PierceHampton.csv",
            Path(event_dates_cache_path)
            if event_dates_cache_path is not None
            else raw_base / "ufc-event-dates.csv",
            Path(historical_odds_dir)
            if historical_odds_dir is not None
            else raw_base / "historical_odds",
            Path(rankings_history_path)
            if rankings_history_path is not None
            else raw_base / "kaggle_rankings" / "rankings_history_extended.csv",
            Path(method_odds_history_path)
            if method_odds_history_path is not None
            else raw_base / "method_odds" / "historical_method_odds_all.csv",
        )
        if any(
            not _same_or_within(path, raw_stage_root)
            for path in staged_raw_inputs
        ):
            raise ValueError(
                "isolated staging raw inputs must remain within raw_data_dir"
            )
        active_bundle = None
    else:
        active_bundle = load_production_bundle_or_none()
    active_dir = (
        active_bundle.processed_dir.resolve(strict=False)
        if active_bundle is not None
        else None
    )
    if active_dir is not None and active_dir != canonical_dir and any(
        target.resolve(strict=False) == active_dir for target in target_dirs
    ):
        raise RuntimeError(
            "refusing to rebuild directly into the active production processed_dir: "
            f"{active_dir}"
        )
    writes_canonical = any(
        target.resolve(strict=False) == canonical_dir for target in target_dirs
    )

    legacy_df = load_kaggle_dataset(base_dataset_path)
    variants = build_training_dataset_variants(
        legacy_df=legacy_df,
        raw_data_dir=raw_root,
        base_dataset_path=base_dataset_path,
        fight_results_path=fight_results_path,
        fight_stats_path=fight_stats_path,
        scraped_fighters_path=scraped_fighters_path,
        supplemental_profiles_path=supplemental_profiles_path,
        official_active_roster_path=official_active_roster_path,
        pierce_fight_identities_path=pierce_fight_identities_path,
        event_dates_cache_path=event_dates_cache_path,
        historical_odds_dir=historical_odds_dir,
        rankings_history_path=rankings_history_path,
        method_odds_history_path=method_odds_history_path,
    )
    fights_df = variants[dataset_variant].copy()
    features_df = (
        build_features(fights_df)
        if raw_root is None
        else build_features(fights_df, raw_data_dir=raw_root)
    )

    if writes_canonical:
        with TemporaryDirectory(prefix="ufc_processed_identity_prevalidation_") as temp:
            validation_dir = Path(temp)
            save_processed(fights_df, filename=validation_dir / "fights_cleaned.csv")
            save_features(features_df, filename=validation_dir / "features.csv")
            _require_processed_identity_integrity(validation_dir)

    summaries: list[dict[str, object]] = []
    if provider_reference_inventory(raw_base) != provider_inventory:
        raise ValueError("provider reference inputs changed during rebuild")
    if observation_reference_inventory(raw_base) != observation_inventory:
        raise ValueError("observation reference inputs changed during rebuild")
    if observation_parents or observation_originals:
        fights_df["observation_reference_schema"] = 1
        features_df["observation_reference_schema"] = 1
    for target_dir in target_dirs:
        fights_filename = target_dir / "fights_cleaned.csv"
        features_filename = target_dir / "features.csv"
        fights_path = save_processed(fights_df, filename=fights_filename)
        save_features(features_df, filename=features_filename)
        features_path = Path(features_filename)
        if provider_parents:
            for artifact in (fights_path, features_path):
                record_provider_consumer_artifact(artifact, parent_references=provider_parents,
                    history_root=target_dir, kind="rebuilt_artifact")
        if observation_parents or observation_originals:
            for artifact in (fights_path, features_path):
                record_observation_consumer_artifact(artifact, parent_references=observation_parents, evidence_references=observation_originals, kind="rebuilt_artifact")
        summaries.append(
            {
                "output_dir": str(target_dir),
                "fights_path": str(fights_path),
                "features_path": str(features_path),
                "fight_rows": int(len(fights_df)),
                "feature_rows": int(len(features_df)),
                "feature_cols": int(len(features_df.columns)),
            }
        )

    summary = {
        "dataset_variant": dataset_variant,
        "outputs": summaries,
    }
    if update_production_manifest and writes_canonical:
        summary["production_bundle"] = reconcile_production_bundle_manifest(
            processed_dir=PROCESSED_DATA_DIR,
        )
    return summary


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
    parser.add_argument(
        "--update-production-manifest",
        action="store_true",
        help="Refresh the active runtime manifest when rebuilding the canonical data/processed snapshot.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=None,
        help="Optional raw artifact root. All unspecified UFC raw inputs resolve beneath it.",
    )
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        default=None,
        help="Optional processed output root.",
    )
    parser.add_argument(
        "--isolated-staging",
        action="store_true",
        help="Require explicit noncanonical raw/processed roots and avoid production bundle access.",
    )
    args = parser.parse_args()

    print(
        json.dumps(
            run_rebuild(
                dataset_variant=args.dataset_variant,
                output_subdirs=args.output_subdir,
                update_production_manifest=args.update_production_manifest,
                raw_data_dir=args.raw_data_dir,
                processed_data_dir=args.processed_data_dir,
                isolated_staging=args.isolated_staging,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
