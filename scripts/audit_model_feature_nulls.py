"""
Audit promoted-contract feature population before retrain / commit.

This script answers a specific release question:
which declared model features are still absent or effectively dead
(100% null) in the materialized training frame that would feed retraining?
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

from src.model import training_spec  # noqa: E402


@dataclass
class AuditResult:
    spec_name: str
    dataset_variant: str | None
    train_cutoff_date: str
    features_rows_total: int
    trainable_rows: int
    train_split_rows: int
    test_split_rows: int
    recent_rows: int
    raw_contract_columns_missing: list[str]
    materialized_contract_columns_missing: list[str]
    dead_contract_columns_trainable: list[str]
    dead_contract_columns_train_split: list[str]
    dead_contract_columns_recent: list[str]
    high_null_columns_trainable: list[dict]
    high_null_columns_train_split: list[dict]
    high_null_columns_recent: list[dict]
    suspicious_default_fills_trainable: list[dict]
    suspicious_default_fills_recent: list[dict]
    group_null_pct_trainable: dict[str, dict[str, float]]
    group_null_pct_recent: dict[str, dict[str, float]]
    raw_master_recent_rows: int | None = None
    raw_master_recent_missing_weight_class: int | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


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


def _null_summary(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_rows = len(frame)
    for column in feature_cols:
        if column not in frame.columns:
            rows.append(
                {
                    "column": column,
                    "nulls": n_rows,
                    "null_pct": 100.0 if n_rows else 0.0,
                }
            )
            continue
        series = frame[column]
        nulls = int(series.isna().sum())
        rows.append(
            {
                "column": column,
                "nulls": nulls,
                "null_pct": round((nulls / n_rows * 100.0) if n_rows else 0.0, 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["null_pct", "column"], ascending=[False, True])


def _group_null_pct(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    groups = {
        "dead_roll_stats": [
            "a_roll_sapm",
            "b_roll_sapm",
            "diff_roll_sapm",
            "a_roll_str_def",
            "b_roll_str_def",
            "diff_roll_str_def",
            "a_roll_td_def",
            "b_roll_td_def",
            "diff_roll_td_def",
            "a_roll_sig_str_landed",
            "b_roll_sig_str_landed",
            "diff_roll_sig_str_landed",
            "a_roll_td_landed",
            "b_roll_td_landed",
            "diff_roll_td_landed",
            "a_roll_kd",
            "b_roll_kd",
            "diff_roll_kd",
            "a_strike_diff",
            "b_strike_diff",
            "diff_strike_diff",
        ],
        "line_movement": [
            "line_movement",
            "line_abs_movement",
            "line_is_sharp",
            "line_steam_move",
            "line_direction_toward_a",
            "line_direction_toward_b",
        ],
        "moneyline_odds": [
            "a_implied_prob",
            "b_implied_prob",
            "diff_implied_prob",
        ],
        "method_odds": [
            "a_ko_odds_prob",
            "a_sub_odds_prob",
            "a_dec_odds_prob",
            "b_ko_odds_prob",
            "b_sub_odds_prob",
            "b_dec_odds_prob",
        ],
        "rankings": [
            "a_wc_rank_feat",
            "b_wc_rank_feat",
            "diff_wc_rank",
            "a_pfp_rank_feat",
            "b_pfp_rank_feat",
            "diff_pfp_rank",
        ],
        "transform_defaults": [
            "is_rematch",
            "h2h_record_diff",
            "a_elo_momentum",
            "b_elo_momentum",
            "diff_elo_momentum",
            "a_sos",
            "b_sos",
            "diff_sos",
        ],
    }
    return {
        group_name: {
            column: round(float(frame[column].isna().mean() * 100.0), 2)
            for column in columns
            if column in frame.columns
        }
        for group_name, columns in groups.items()
    }


def _default_fill_summary(frame: pd.DataFrame) -> list[dict[str, object]]:
    checks = [
        ("a_wc_rank", "a_wc_rank_feat", 16.0),
        ("b_wc_rank", "b_wc_rank_feat", 16.0),
        ("a_pfp_rank", "a_pfp_rank_feat", 16.0),
        ("b_pfp_rank", "b_pfp_rank_feat", 16.0),
        ("title_bout", "is_title_bout", 0.0),
        ("empty_arena", "is_empty_arena", 0.0),
        ("num_rounds", "num_rounds_feat", 3.0),
    ]
    rows: list[dict[str, object]] = []
    for raw_col, feature_col, default_value in checks:
        if raw_col not in frame.columns or feature_col not in frame.columns:
            continue
        raw_missing = frame[raw_col].isna()
        defaulted = frame[feature_col] == default_value
        count = int((raw_missing & defaulted).sum())
        if count <= 0:
            continue
        rows.append(
            {
                "raw_column": raw_col,
                "feature_column": feature_col,
                "default_value": default_value,
                "rows": count,
                "pct": round(float(count / len(frame) * 100.0), 2) if len(frame) else 0.0,
            }
        )
    return rows


def run_audit(
    *,
    spec_name: str,
    dataset_variant: str | None,
    min_fights: int,
    recent_rows: int,
    null_threshold_pct: float,
) -> AuditResult:
    from src.model.train import prepare_train_test

    spec = _resolve_spec(spec_name)
    effective_dataset_variant = dataset_variant
    if effective_dataset_variant in {None, ""}:
        spec_dataset_variant = getattr(spec, "dataset_variant", None)
        if spec_dataset_variant not in {None, "", "default"}:
            effective_dataset_variant = spec_dataset_variant
    if effective_dataset_variant == "default":
        effective_dataset_variant = None

    raw_features = _load_feature_frame(effective_dataset_variant)
    raw_contract_columns_missing = [c for c in spec.feature_cols if c not in raw_features.columns]
    materialized = training_spec.materialize_spec_transforms(raw_features, spec)
    materialized_contract_columns_missing = [
        column for column in spec.feature_cols if column not in materialized.columns
    ]
    trainable = materialized[
        (materialized["a_num_fights"] >= min_fights)
        & (materialized["b_num_fights"] >= min_fights)
        & materialized["target"].notna()
    ].copy()
    split_ready = materialized.copy()
    for column in materialized_contract_columns_missing:
        split_ready[column] = pd.NA
    train_split, test_split, _ = prepare_train_test(
        split_ready,
        cutoff_date=spec.train_cutoff_date,
        min_fights=min_fights,
        feature_cols=list(spec.feature_cols),
    )
    recent = trainable.sort_values("event_date").tail(recent_rows).copy()

    trainable_nulls = _null_summary(trainable, spec.feature_cols)
    train_split_nulls = _null_summary(train_split, spec.feature_cols)
    recent_nulls = _null_summary(recent, spec.feature_cols)

    master_recent_rows = None
    master_recent_missing_weight_class = None
    master_path = REPO_ROOT / "data" / "raw" / "ufc-master.csv"
    if master_path.exists():
        master = pd.read_csv(master_path, low_memory=False)
        if "Date" in master.columns:
            recent_master = master[pd.to_datetime(master["Date"], errors="coerce") >= pd.Timestamp("2025-01-01")]
            master_recent_rows = int(len(recent_master))
            if "WeightClass" in recent_master.columns:
                master_recent_missing_weight_class = int(recent_master["WeightClass"].isna().sum())

    return AuditResult(
        spec_name=spec.name,
        dataset_variant=effective_dataset_variant,
        train_cutoff_date=spec.train_cutoff_date,
        features_rows_total=int(len(raw_features)),
        trainable_rows=int(len(trainable)),
        train_split_rows=int(len(train_split)),
        test_split_rows=int(len(test_split)),
        recent_rows=int(len(recent)),
        raw_contract_columns_missing=raw_contract_columns_missing,
        materialized_contract_columns_missing=materialized_contract_columns_missing,
        dead_contract_columns_trainable=trainable_nulls.loc[
            trainable_nulls["null_pct"] == 100.0, "column"
        ].tolist(),
        dead_contract_columns_train_split=train_split_nulls.loc[
            train_split_nulls["null_pct"] == 100.0, "column"
        ].tolist(),
        dead_contract_columns_recent=recent_nulls.loc[
            recent_nulls["null_pct"] == 100.0, "column"
        ].tolist(),
        high_null_columns_trainable=trainable_nulls.loc[
            trainable_nulls["null_pct"] >= null_threshold_pct
        ].head(30).to_dict(orient="records"),
        high_null_columns_train_split=train_split_nulls.loc[
            train_split_nulls["null_pct"] >= null_threshold_pct
        ].head(30).to_dict(orient="records"),
        high_null_columns_recent=recent_nulls.loc[
            recent_nulls["null_pct"] >= null_threshold_pct
        ].head(30).to_dict(orient="records"),
        suspicious_default_fills_trainable=_default_fill_summary(trainable),
        suspicious_default_fills_recent=_default_fill_summary(recent),
        group_null_pct_trainable=_group_null_pct(trainable),
        group_null_pct_recent=_group_null_pct(recent),
        raw_master_recent_rows=master_recent_rows,
        raw_master_recent_missing_weight_class=master_recent_missing_weight_class,
    )


def _print_human(result: AuditResult) -> None:
    print(f"Spec: {result.spec_name}")
    print(f"Dataset variant: {result.dataset_variant or 'processed_features_csv'}")
    print(
        "Rows: "
        f"features={result.features_rows_total}, "
        f"trainable={result.trainable_rows}, "
        f"train_split={result.train_split_rows}, "
        f"test_split={result.test_split_rows}, "
        f"recent={result.recent_rows}"
    )
    print(f"Train cutoff date: {result.train_cutoff_date}")
    if result.raw_master_recent_rows is not None:
        print(
            "Raw master recent rows: "
            f"{result.raw_master_recent_rows}, "
            f"missing_weight_class={result.raw_master_recent_missing_weight_class}"
        )
    print()

    print("Raw contract columns missing before materialization:")
    if result.raw_contract_columns_missing:
        print("  " + ", ".join(result.raw_contract_columns_missing))
    else:
        print("  none")
    print()

    print("Contract columns still missing after materialization:")
    if result.materialized_contract_columns_missing:
        print("  " + ", ".join(result.materialized_contract_columns_missing))
    else:
        print("  none")
    print()

    print("Dead contract columns after materialization (trainable rows):")
    if result.dead_contract_columns_trainable:
        print("  " + ", ".join(result.dead_contract_columns_trainable))
    else:
        print("  none")
    print()

    print("Dead contract columns on actual train split:")
    if result.dead_contract_columns_train_split:
        print("  " + ", ".join(result.dead_contract_columns_train_split))
    else:
        print("  none")
    print()

    print("High-null columns (trainable rows):")
    for row in result.high_null_columns_trainable:
        print(f"  {row['column']}: {row['null_pct']}% ({row['nulls']} nulls)")
    print()

    print("High-null columns (actual train split):")
    for row in result.high_null_columns_train_split:
        print(f"  {row['column']}: {row['null_pct']}% ({row['nulls']} nulls)")
    print()

    print("High-null columns (recent rows):")
    for row in result.high_null_columns_recent:
        print(f"  {row['column']}: {row['null_pct']}% ({row['nulls']} nulls)")
    print()

    print("Suspicious default fills (trainable rows):")
    if result.suspicious_default_fills_trainable:
        for row in result.suspicious_default_fills_trainable:
            print(
                f"  {row['feature_column']}: {row['rows']} rows "
                f"({row['pct']}%) from missing {row['raw_column']} -> {row['default_value']}"
            )
    else:
        print("  none")
    print()

    print("Suspicious default fills (recent rows):")
    if result.suspicious_default_fills_recent:
        for row in result.suspicious_default_fills_recent:
            print(
                f"  {row['feature_column']}: {row['rows']} rows "
                f"({row['pct']}%) from missing {row['raw_column']} -> {row['default_value']}"
            )
    else:
        print("  none")
    print()

    print("Group null % (recent rows):")
    for group_name, values in result.group_null_pct_recent.items():
        joined = ", ".join(f"{column}={pct}%" for column, pct in values.items())
        print(f"  {group_name}: {joined}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="full_live_contract_v2")
    parser.add_argument("--dataset-variant", default=None)
    parser.add_argument("--min-fights", type=int, default=2)
    parser.add_argument("--recent-rows", type=int, default=612)
    parser.add_argument("--null-threshold-pct", type=float, default=20.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--allow-dead-features",
        action="store_true",
        help="Report dead features but do not fail the process.",
    )
    args = parser.parse_args()

    result = run_audit(
        spec_name=args.spec,
        dataset_variant=args.dataset_variant,
        min_fights=args.min_fights,
        recent_rows=args.recent_rows,
        null_threshold_pct=args.null_threshold_pct,
    )

    if args.json:
        print(result.to_json())
    else:
        _print_human(result)

    if (
        result.materialized_contract_columns_missing
        or result.dead_contract_columns_trainable
        or result.dead_contract_columns_train_split
        or result.suspicious_default_fills_trainable
    ) and not args.allow_dead_features:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
