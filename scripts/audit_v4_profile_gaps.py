"""
Audit remaining V4 profile-feature gaps at the fighter-row level.

This script answers a narrower follow-up question after the contract-level null
audit: for the remaining physical-profile / stance nulls, are we missing the
entire fighter profile row, or is the underlying UFCStats profile field itself
blank in the canonical scraped artifact?
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.name_utils import normalize_person_name  # noqa: E402
from src.model import training_spec  # noqa: E402
from src.model.train import prepare_train_test  # noqa: E402


PROFILE_FIELD_CONFIG = {
    "height": {
        "a_feature": "a_height",
        "b_feature": "b_height",
        "combined_feature": "diff_height",
        "source_column": "height",
    },
    "reach": {
        "a_feature": "a_reach",
        "b_feature": "b_reach",
        "combined_feature": "diff_reach",
        "source_column": "reach",
    },
    "weight": {
        "a_feature": "a_weight",
        "b_feature": "b_weight",
        "combined_feature": "diff_weight",
        "source_column": "weight",
    },
    "age": {
        "a_feature": "a_age",
        "b_feature": "b_age",
        "combined_feature": "diff_age",
        "source_column": "dob",
    },
    "stance": {
        "a_feature": "a_stance_enc",
        "b_feature": "b_stance_enc",
        "combined_feature": "same_stance",
        "source_column": "stance",
    },
}


@dataclass
class GapAuditResult:
    spec_name: str
    dataset_variant: str | None
    split: str
    fight_rows: int
    fighter_side_rows: int
    gap_rows: int
    field_summary: list[dict[str, object]]
    combined_feature_summary: list[dict[str, object]]
    top_fighters: list[dict[str, object]]
    sample_rows: list[dict[str, object]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def _resolve_spec(spec_name: str):
    try:
        return training_spec.resolve_named_training_spec(spec_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _load_feature_frame(dataset_variant: str | None) -> pd.DataFrame:
    if dataset_variant is None:
        features_path = REPO_ROOT / "data" / "processed" / "features.csv"
        if not features_path.exists():
            raise SystemExit(f"Missing processed feature matrix: {features_path}")
        return pd.read_csv(features_path, parse_dates=["event_date"])

    from src.data.kaggle_loader import load_kaggle_dataset
    from src.data.ufc_refresh import TRAINING_DATASET_VARIANTS, build_training_dataset_variants
    from src.features.build_features import build_features

    if dataset_variant not in TRAINING_DATASET_VARIANTS:
        known = ", ".join(TRAINING_DATASET_VARIANTS)
        raise SystemExit(f"Unknown dataset variant '{dataset_variant}'. Known variants: {known}")

    legacy_df = load_kaggle_dataset(REPO_ROOT / "data" / "raw" / "ufc-master.csv")
    variants = build_training_dataset_variants(legacy_df=legacy_df)
    return build_features(variants[dataset_variant])


def _blank_profile_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "n/a", "--"}


def _load_profile_lookup(scraped_fighters_path: Path) -> dict[str, dict[str, object]]:
    if not scraped_fighters_path.exists():
        raise SystemExit(f"Missing scraped fighter profile artifact: {scraped_fighters_path}")

    fighters_df = pd.read_csv(scraped_fighters_path)
    if fighters_df.empty or "name" not in fighters_df.columns:
        raise SystemExit(
            f"Scraped fighter profile artifact is empty or missing 'name': {scraped_fighters_path}"
        )

    lookup: dict[str, dict[str, object]] = {}
    for _, row in fighters_df.iterrows():
        key = normalize_person_name(row.get("name"))
        if not key:
            continue
        dob_raw = row.get("dob")
        dob_ts = pd.to_datetime(dob_raw, errors="coerce", format="mixed")
        lookup[key] = {
            "name": row.get("name"),
            "fighter_url": row.get("fighter_url", ""),
            "height": row.get("height"),
            "reach": row.get("reach"),
            "weight": row.get("weight"),
            "stance": row.get("stance"),
            "dob": dob_raw,
            "dob_present": bool(pd.notna(dob_ts)),
        }
    return lookup


def _select_split_frame(
    materialized: pd.DataFrame,
    spec,
    *,
    split: str,
    min_fights: int,
    recent_rows: int,
) -> pd.DataFrame:
    trainable = materialized[
        (materialized["a_num_fights"] >= min_fights)
        & (materialized["b_num_fights"] >= min_fights)
        & materialized["target"].notna()
    ].copy()
    train_split, test_split, _ = prepare_train_test(
        materialized.copy(),
        cutoff_date=spec.train_cutoff_date,
        min_fights=min_fights,
        feature_cols=list(spec.feature_cols),
    )

    if split == "trainable":
        return trainable
    if split == "train":
        return train_split
    if split == "test":
        return test_split
    if split == "recent":
        return trainable.sort_values("event_date").tail(recent_rows).copy()
    raise SystemExit(f"Unknown split '{split}'")


def _source_field_present(profile_row: dict[str, object], field_name: str) -> bool:
    if field_name == "age":
        return bool(profile_row.get("dob_present"))
    return not _blank_profile_value(profile_row.get(PROFILE_FIELD_CONFIG[field_name]["source_column"]))


def _source_field_value(profile_row: dict[str, object], field_name: str) -> object:
    if field_name == "age":
        return profile_row.get("dob")
    return profile_row.get(PROFILE_FIELD_CONFIG[field_name]["source_column"])


def _build_gap_details(
    frame: pd.DataFrame,
    profile_lookup: dict[str, dict[str, object]],
) -> pd.DataFrame:
    gap_rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        row_dict = row._asdict()
        for side, fighter_col, opponent_col in (
            ("a", "fighter_a", "fighter_b"),
            ("b", "fighter_b", "fighter_a"),
        ):
            fighter_name = row_dict.get(fighter_col)
            opponent_name = row_dict.get(opponent_col)
            profile_row = profile_lookup.get(normalize_person_name(fighter_name))
            for field_name, config in PROFILE_FIELD_CONFIG.items():
                feature_column = config[f"{side}_feature"]
                feature_value = row_dict.get(feature_column, pd.NA)
                if pd.notna(feature_value):
                    continue

                source_row_present = profile_row is not None
                source_field_present = (
                    _source_field_present(profile_row, field_name)
                    if source_row_present
                    else False
                )
                if not source_row_present:
                    cause = "missing_profile_row"
                elif not source_field_present:
                    cause = "source_blank"
                else:
                    cause = "source_present_but_feature_null"

                gap_rows.append(
                    {
                        "event_date": row_dict.get("event_date"),
                        "weight_class": row_dict.get("weight_class"),
                        "fighter_name": fighter_name,
                        "opponent_name": opponent_name,
                        "side": side,
                        "field": field_name,
                        "feature_column": feature_column,
                        "combined_feature": config["combined_feature"],
                        "cause": cause,
                        "source_row_present": source_row_present,
                        "source_field_present": source_field_present,
                        "source_value": _source_field_value(profile_row, field_name)
                        if source_row_present
                        else None,
                        "fighter_url": profile_row.get("fighter_url", "") if source_row_present else "",
                    }
                )
    return pd.DataFrame(gap_rows)


def run_audit(
    *,
    spec_name: str,
    split: str,
    min_fights: int,
    recent_rows: int,
    sample_limit: int,
    scraped_fighters_path: Path,
) -> GapAuditResult:
    spec = _resolve_spec(spec_name)
    raw_features = _load_feature_frame(getattr(spec, "dataset_variant", None))
    materialized = training_spec.materialize_spec_transforms(raw_features, spec)
    frame = _select_split_frame(
        materialized,
        spec,
        split=split,
        min_fights=min_fights,
        recent_rows=recent_rows,
    )
    profile_lookup = _load_profile_lookup(scraped_fighters_path)
    details_df = _build_gap_details(frame, profile_lookup)
    if details_df.empty:
        return GapAuditResult(
            spec_name=spec.name,
            dataset_variant=getattr(spec, "dataset_variant", None),
            split=split,
            fight_rows=len(frame),
            fighter_side_rows=len(frame) * 2,
            gap_rows=0,
            field_summary=[],
            combined_feature_summary=[],
            top_fighters=[],
            sample_rows=[],
        )

    field_summary: list[dict[str, object]] = []
    for field_name in PROFILE_FIELD_CONFIG:
        field_rows = details_df[details_df["field"] == field_name]
        field_summary.append(
            {
                "field": field_name,
                "gap_rows": int(len(field_rows)),
                "missing_profile_row": int((field_rows["cause"] == "missing_profile_row").sum()),
                "source_blank": int((field_rows["cause"] == "source_blank").sum()),
                "source_present_but_feature_null": int(
                    (field_rows["cause"] == "source_present_but_feature_null").sum()
                ),
                "unique_fighters": int(field_rows["fighter_name"].nunique()),
            }
        )

    combined_feature_summary = []
    for field_name, config in PROFILE_FIELD_CONFIG.items():
        combined_column = config["combined_feature"]
        if combined_column not in frame.columns:
            continue
        null_rows = int(frame[combined_column].isna().sum())
        combined_feature_summary.append(
            {
                "field": field_name,
                "combined_feature": combined_column,
                "null_rows": null_rows,
            }
        )

    top_fighters = (
        details_df.groupby("fighter_name")
        .agg(
            gap_rows=("field", "size"),
            fields=("field", lambda values: ",".join(sorted(set(values)))),
        )
        .reset_index()
        .sort_values(["gap_rows", "fighter_name"], ascending=[False, True])
        .head(25)
        .to_dict(orient="records")
    )

    sample_rows = (
        details_df.sort_values(["field", "fighter_name", "event_date"])
        .head(sample_limit)
        .assign(event_date=lambda df: df["event_date"].astype(str))
        .to_dict(orient="records")
    )

    return GapAuditResult(
        spec_name=spec.name,
        dataset_variant=getattr(spec, "dataset_variant", None),
        split=split,
        fight_rows=len(frame),
        fighter_side_rows=len(frame) * 2,
        gap_rows=len(details_df),
        field_summary=field_summary,
        combined_feature_summary=combined_feature_summary,
        top_fighters=top_fighters,
        sample_rows=sample_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="full_live_contract_v4")
    parser.add_argument(
        "--split",
        choices=["trainable", "train", "test", "recent"],
        default="train",
    )
    parser.add_argument("--min-fights", type=int, default=2)
    parser.add_argument("--recent-rows", type=int, default=250)
    parser.add_argument("--sample-limit", type=int, default=25)
    parser.add_argument(
        "--scraped-fighters-path",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "ufc_fighters_scraped.csv",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_audit(
        spec_name=args.spec,
        split=args.split,
        min_fights=args.min_fights,
        recent_rows=args.recent_rows,
        sample_limit=args.sample_limit,
        scraped_fighters_path=args.scraped_fighters_path,
    )

    if args.output_csv is not None:
        spec = _resolve_spec(args.spec)
        raw_features = _load_feature_frame(getattr(spec, "dataset_variant", None))
        materialized = training_spec.materialize_spec_transforms(raw_features, spec)
        frame = _select_split_frame(
            materialized,
            spec,
            split=args.split,
            min_fights=args.min_fights,
            recent_rows=args.recent_rows,
        )
        profile_lookup = _load_profile_lookup(args.scraped_fighters_path)
        output_df = _build_gap_details(frame, profile_lookup).sort_values(
            ["field", "cause", "fighter_name", "event_date"],
            ascending=[True, True, True, True],
        )
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(args.output_csv, index=False)

    if args.json:
        print(result.to_json())
        return

    print(f"Spec: {result.spec_name}")
    print(f"Dataset variant: {result.dataset_variant}")
    print(f"Split: {result.split}")
    print(f"Fight rows: {result.fight_rows}")
    print(f"Fighter-side rows: {result.fighter_side_rows}")
    print(f"Gap rows: {result.gap_rows}")
    print()
    print("Field summary:")
    for row in result.field_summary:
        print(
            "  "
            f"{row['field']}: gaps={row['gap_rows']} | "
            f"missing_profile_row={row['missing_profile_row']} | "
            f"source_blank={row['source_blank']} | "
            f"source_present_but_feature_null={row['source_present_but_feature_null']} | "
            f"unique_fighters={row['unique_fighters']}"
        )
    print()
    print("Combined feature null rows:")
    for row in result.combined_feature_summary:
        print(f"  {row['combined_feature']}: {row['null_rows']}")


if __name__ == "__main__":
    main()
