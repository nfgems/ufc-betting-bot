"""Train an isolated tuned full_live_contract_v4 candidate and emit audit artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_model_feature_nulls import run_audit as run_null_audit  # noqa: E402
from scripts.audit_v4_profile_gaps import (  # noqa: E402
    _build_gap_details,
    _load_feature_frame,
    _load_profile_lookup,
    _resolve_spec,
    _select_split_frame,
    run_audit as run_profile_gap_audit,
)
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR  # noqa: E402
from src.data.kaggle_loader import load_kaggle_dataset, save_processed  # noqa: E402
from src.data.ufc_refresh import TRAINING_DATASET_VARIANTS, build_training_dataset_variants  # noqa: E402
from src.features.build_features import build_features, save_features  # noqa: E402
from src.model.feature_provenance import audit_training_feature_provenance  # noqa: E402
from src.model.refreshed_v4_eval import evaluate_models_against_test_set, write_evaluation_payload  # noqa: E402
from src.model.train import train_all_models  # noqa: E402
from src.model import training_spec as training_spec_module  # noqa: E402


def _parse_labeled_model_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid value '{value}'. Expected label=path_to_model.pkl"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError(
            f"Invalid value '{value}'. Expected label=path_to_model.pkl"
        )
    return label, path


def _parse_xgb_param(value: str) -> tuple[str, object]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid value '{value}'. Expected key=value"
        )
    key, raw = value.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    if not key:
        raise argparse.ArgumentTypeError(
            f"Invalid value '{value}'. Expected key=value"
        )

    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return key, lowered == "true"

    try:
        if any(ch in raw for ch in (".", "e", "E")):
            return key, float(raw)
        return key, int(raw)
    except ValueError:
        return key, raw


def _load_training_dataframe(spec) -> object:
    dataset_variant = getattr(spec, "dataset_variant", "default")
    if dataset_variant in {None, "", "default"}:
        return load_kaggle_dataset()

    if dataset_variant not in TRAINING_DATASET_VARIANTS:
        known = ", ".join(TRAINING_DATASET_VARIANTS)
        raise ValueError(
            f"Unknown training dataset variant '{dataset_variant}'. Known variants: {known}"
        )

    legacy_df = load_kaggle_dataset(RAW_DATA_DIR / "ufc-master.csv")
    all_variants = build_training_dataset_variants(legacy_df=legacy_df)
    return all_variants[dataset_variant].copy()


def _build_tuned_spec(args) -> object:
    base_spec = training_spec_module.resolve_named_training_spec(args.base_spec)
    xgb_params = dict(base_spec.xgb_params)
    for key, value in args.xgb_param or []:
        xgb_params[key] = value

    return replace(
        base_spec,
        calibration_method=args.calibration_method or base_spec.calibration_method,
        calibration_cv=args.calibration_cv or base_spec.calibration_cv,
        train_cutoff_date=(
            args.train_cutoff_date
            if args.train_cutoff_date is not None
            else base_spec.train_cutoff_date
        ),
        time_decay_half_life=(
            args.time_decay_half_life
            if args.time_decay_half_life is not None
            else base_spec.time_decay_half_life
        ),
        odds_noise_std=(
            args.odds_noise_std
            if args.odds_noise_std is not None
            else base_spec.odds_noise_std
        ),
        train_start_date=(
            args.train_start_date
            if args.train_start_date is not None
            else getattr(base_spec, "train_start_date", "")
        ),
        train_end_date=(
            args.train_end_date
            if args.train_end_date is not None
            else getattr(base_spec, "train_end_date", "")
        ),
        xgb_params=xgb_params,
    )


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_profile_gap_csv(
    *,
    spec_name: str,
    split: str,
    output_csv: Path,
    min_fights: int,
    recent_rows: int,
    scraped_fighters_path: Path,
) -> None:
    spec = _resolve_spec(spec_name)
    raw_features = _load_feature_frame(getattr(spec, "dataset_variant", None))
    materialized = training_spec_module.materialize_spec_transforms(raw_features, spec)
    frame = _select_split_frame(
        materialized,
        spec,
        split=split,
        min_fights=min_fights,
        recent_rows=recent_rows,
    )
    profile_lookup = _load_profile_lookup(scraped_fighters_path)
    gap_details = _build_gap_details(frame, profile_lookup).sort_values(
        ["field", "cause", "fighter_name", "event_date"],
        ascending=[True, True, True, True],
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    gap_details.to_csv(output_csv, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-spec", default="full_live_contract_v4")
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument("--candidate-label", default=None)
    parser.add_argument(
        "--calibration-method",
        choices=["isotonic", "sigmoid", "none"],
        default=None,
    )
    parser.add_argument("--calibration-cv", default=None)
    parser.add_argument("--time-decay-half-life", type=int, default=None)
    parser.add_argument("--odds-noise-std", type=float, default=None)
    parser.add_argument(
        "--train-cutoff-date",
        default=None,
        help="Optional train/test split cutoff (YYYY-MM-DD). Rows on/after this date go to the test set.",
    )
    parser.add_argument(
        "--train-start-date",
        default=None,
        help="Optional lower bound for training rows only (YYYY-MM-DD). Test set remains tied to train_cutoff_date.",
    )
    parser.add_argument(
        "--train-end-date",
        default=None,
        help="Optional exclusive upper bound for training rows only (YYYY-MM-DD). Test set remains tied to train_cutoff_date.",
    )
    parser.add_argument(
        "--xgb-param",
        action="append",
        type=_parse_xgb_param,
        default=None,
        help="XGBoost override in the form key=value. Repeat as needed.",
    )
    parser.add_argument("--test-set", type=Path, default=None)
    parser.add_argument(
        "--compare-model",
        action="append",
        type=_parse_labeled_model_arg,
        default=None,
        help="Comparison artifact in the form label=path_to_model.pkl. Repeat as needed.",
    )
    parser.add_argument("--skip-audits", action="store_true")
    args = parser.parse_args()

    spec = _build_tuned_spec(args)
    candidate_label = args.candidate_label or Path(args.output_subdir).name

    processed_output_dir = PROCESSED_DATA_DIR / args.output_subdir
    models_output_dir = MODELS_DIR / args.output_subdir
    test_set_path = processed_output_dir / "test_set.csv"

    fights_df = _load_training_dataframe(spec)
    save_processed(
        fights_df,
        filename=Path(args.output_subdir) / "fights_cleaned.csv",
    )

    features_df = build_features(fights_df)
    save_features(
        features_df,
        filename=Path(args.output_subdir) / "features.csv",
    )

    train_all_models(
        features_df,
        spec=spec,
        models_dir=models_output_dir,
        test_set_path=test_set_path,
    )

    summary: dict[str, object] = {
        "candidate_label": candidate_label,
        "output_subdir": args.output_subdir,
        "model_dir": models_output_dir.resolve().as_posix(),
        "test_set_path": test_set_path.resolve().as_posix(),
        "spec_name": spec.name,
        "feature_count": len(spec.feature_cols),
        "calibration_method": spec.calibration_method,
        "calibration_cv": spec.calibration_cv,
        "train_cutoff_date": spec.train_cutoff_date,
        "time_decay_half_life": spec.time_decay_half_life,
        "odds_noise_std": spec.odds_noise_std,
        "train_start_date": getattr(spec, "train_start_date", ""),
        "train_end_date": getattr(spec, "train_end_date", ""),
        "xgb_params": spec.xgb_params,
    }

    if not args.skip_audits:
        null_audit = run_null_audit(
            spec_name=spec.name,
            dataset_variant=spec.dataset_variant,
            min_fights=2,
            recent_rows=612,
            null_threshold_pct=0.0,
        )
        null_audit_path = models_output_dir / "null_audit.json"
        null_audit_path.write_text(null_audit.to_json() + "\n", encoding="utf-8")
        summary["null_audit_path"] = null_audit_path.resolve().as_posix()

        profile_gap = run_profile_gap_audit(
            spec_name=spec.name,
            split="train",
            min_fights=2,
            recent_rows=250,
            sample_limit=25,
            scraped_fighters_path=REPO_ROOT / "data" / "raw" / "ufc_fighters_scraped.csv",
        )
        profile_gap_json_path = models_output_dir / "profile_gap_audit_train.json"
        profile_gap_json_path.write_text(profile_gap.to_json() + "\n", encoding="utf-8")
        summary["profile_gap_audit_json_path"] = profile_gap_json_path.resolve().as_posix()

        profile_gap_csv_path = models_output_dir / "profile_gap_audit_train.csv"
        _write_profile_gap_csv(
            spec_name=spec.name,
            split="train",
            output_csv=profile_gap_csv_path,
            min_fights=2,
            recent_rows=250,
            scraped_fighters_path=REPO_ROOT / "data" / "raw" / "ufc_fighters_scraped.csv",
        )
        summary["profile_gap_audit_csv_path"] = profile_gap_csv_path.resolve().as_posix()

        provenance_payload = audit_training_feature_provenance(spec).to_dict()
        provenance_path = models_output_dir / "provenance_audit.json"
        _write_json(provenance_path, provenance_payload)
        summary["provenance_audit_path"] = provenance_path.resolve().as_posix()

    if args.test_set is not None and args.compare_model:
        labeled_models = list(args.compare_model)
        labeled_models.append((candidate_label, str(models_output_dir / "xgboost_model.pkl")))
        eval_payload = evaluate_models_against_test_set(
            test_set_path=args.test_set,
            labeled_models=labeled_models,
        )
        eval_path = models_output_dir / "eval_against_refreshed_v4_test.json"
        write_evaluation_payload(eval_payload, eval_path)
        summary["comparison_path"] = eval_path.resolve().as_posix()
        summary["comparison_metrics"] = eval_payload

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
