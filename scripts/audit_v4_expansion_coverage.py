"""Audit exact fight-level coverage gaps for V4 expansion feature families."""

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

from src.data.kaggle_loader import load_kaggle_dataset  # noqa: E402
from src.data.ufc_refresh import TRAINING_DATASET_VARIANTS, build_training_dataset_variants  # noqa: E402
from src.features.build_features import build_features  # noqa: E402
from src.model import training_spec  # noqa: E402
from src.model.train import prepare_train_test  # noqa: E402


FAMILY_CONFIG = {
    "moneyline_odds": {
        "raw_columns": ["a_odds", "b_odds"],
        "feature_columns": ["a_implied_prob", "b_implied_prob", "diff_implied_prob"],
    },
    "method_odds": {
        "raw_columns": [
            "a_ko_odds",
            "a_sub_odds",
            "a_dec_odds",
            "b_ko_odds",
            "b_sub_odds",
            "b_dec_odds",
        ],
        "feature_columns": [
            "a_ko_odds_prob",
            "a_sub_odds_prob",
            "a_dec_odds_prob",
            "b_ko_odds_prob",
            "b_sub_odds_prob",
            "b_dec_odds_prob",
        ],
    },
    "rankings": {
        "raw_columns": ["a_wc_rank", "b_wc_rank", "a_pfp_rank", "b_pfp_rank"],
        "feature_columns": [
            "a_wc_rank_feat",
            "b_wc_rank_feat",
            "diff_wc_rank",
            "a_pfp_rank_feat",
            "b_pfp_rank_feat",
            "diff_pfp_rank",
        ],
    },
}


@dataclass
class FamilyCoverageSummary:
    family: str
    rows_total: int
    rows_complete_raw: int
    rows_any_raw_present: int
    rows_complete_features: int
    rows_any_feature_present: int
    raw_complete_pct: float
    feature_complete_pct: float
    top_missing_raw_columns: list[dict[str, object]]
    top_missing_feature_columns: list[dict[str, object]]
    missing_fight_rows: int


@dataclass
class ExpansionCoverageAudit:
    spec_name: str
    dataset_variant: str | None
    split: str
    fight_rows: int
    family_summaries: list[FamilyCoverageSummary]
    output_dir: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _resolve_spec(spec_name: str):
    try:
        return training_spec.resolve_named_training_spec(spec_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _load_feature_frame(dataset_variant: str | None) -> pd.DataFrame:
    if dataset_variant in {None, "", "default"}:
        features_path = REPO_ROOT / "data" / "processed" / "features.csv"
        if not features_path.exists():
            raise SystemExit(f"Missing processed feature matrix: {features_path}")
        return pd.read_csv(features_path, parse_dates=["event_date"])

    if dataset_variant not in TRAINING_DATASET_VARIANTS:
        known = ", ".join(TRAINING_DATASET_VARIANTS)
        raise SystemExit(f"Unknown dataset variant '{dataset_variant}'. Known variants: {known}")

    legacy_df = load_kaggle_dataset(REPO_ROOT / "data" / "raw" / "ufc-master.csv")
    variants = build_training_dataset_variants(legacy_df=legacy_df)
    return build_features(variants[dataset_variant])


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


def _rank_missing_columns(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = len(frame)
    for column in columns:
        if column not in frame.columns:
            continue
        nulls = int(frame[column].isna().sum())
        rows.append(
            {
                "column": column,
                "nulls": nulls,
                "null_pct": round(float(nulls / total * 100.0), 2) if total else 0.0,
            }
        )
    return sorted(rows, key=lambda row: (-row["nulls"], row["column"]))


def _event_column(frame: pd.DataFrame) -> str | None:
    for candidate in ("event", "event_name", "event_title"):
        if candidate in frame.columns:
            return candidate
    return None


def _missing_fight_rows(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    config = FAMILY_CONFIG[family]
    raw_columns = [column for column in config["raw_columns"] if column in frame.columns]
    feature_columns = [column for column in config["feature_columns"] if column in frame.columns]
    feature_complete = frame[feature_columns].notna().all(axis=1) if feature_columns else pd.Series(False, index=frame.index)
    raw_complete = frame[raw_columns].notna().all(axis=1) if raw_columns else pd.Series(False, index=frame.index)
    raw_any_present = frame[raw_columns].notna().any(axis=1) if raw_columns else pd.Series(False, index=frame.index)

    event_column = _event_column(frame)
    rows: list[dict[str, object]] = []
    for idx, row in frame.loc[~feature_complete].iterrows():
        rows.append(
            {
                "event_date": row.get("event_date"),
                "event": row.get(event_column) if event_column is not None else "",
                "fighter_a": row.get("fighter_a"),
                "fighter_b": row.get("fighter_b"),
                "weight_class": row.get("weight_class"),
                "family": family,
                "raw_complete": bool(raw_complete.loc[idx]),
                "raw_any_present": bool(raw_any_present.loc[idx]),
                "missing_raw_columns": ",".join(
                    column for column in raw_columns if pd.isna(row.get(column))
                ),
                "missing_feature_columns": ",".join(
                    column for column in feature_columns if pd.isna(row.get(column))
                ),
            }
        )
    return pd.DataFrame(rows)


def run_audit(
    *,
    spec_name: str,
    split: str,
    min_fights: int,
    recent_rows: int,
    output_dir: Path | None,
) -> ExpansionCoverageAudit:
    spec = _resolve_spec(spec_name)
    raw_features = _load_feature_frame(getattr(spec, "dataset_variant", None))
    materialized = training_spec.materialize_spec_transforms(raw_features.copy(), spec)
    frame = _select_split_frame(
        materialized,
        spec,
        split=split,
        min_fights=min_fights,
        recent_rows=recent_rows,
    )

    family_summaries: list[FamilyCoverageSummary] = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for family_name in ("moneyline_odds", "method_odds", "rankings"):
        config = FAMILY_CONFIG[family_name]
        raw_columns = [column for column in config["raw_columns"] if column in frame.columns]
        feature_columns = [column for column in config["feature_columns"] if column in frame.columns]
        if not raw_columns and not feature_columns:
            continue

        raw_complete = frame[raw_columns].notna().all(axis=1) if raw_columns else pd.Series(False, index=frame.index)
        raw_any_present = frame[raw_columns].notna().any(axis=1) if raw_columns else pd.Series(False, index=frame.index)
        feature_complete = frame[feature_columns].notna().all(axis=1) if feature_columns else pd.Series(False, index=frame.index)
        feature_any_present = frame[feature_columns].notna().any(axis=1) if feature_columns else pd.Series(False, index=frame.index)

        missing_rows = _missing_fight_rows(frame, family_name)
        if output_dir is not None:
            missing_rows.to_csv(output_dir / f"{family_name}_missing_fights.csv", index=False)

        total = len(frame)
        family_summaries.append(
            FamilyCoverageSummary(
                family=family_name,
                rows_total=total,
                rows_complete_raw=int(raw_complete.sum()),
                rows_any_raw_present=int(raw_any_present.sum()),
                rows_complete_features=int(feature_complete.sum()),
                rows_any_feature_present=int(feature_any_present.sum()),
                raw_complete_pct=round(float(raw_complete.mean() * 100.0), 2) if total else 0.0,
                feature_complete_pct=round(float(feature_complete.mean() * 100.0), 2) if total else 0.0,
                top_missing_raw_columns=_rank_missing_columns(frame, raw_columns),
                top_missing_feature_columns=_rank_missing_columns(frame, feature_columns),
                missing_fight_rows=int(len(missing_rows)),
            )
        )

    return ExpansionCoverageAudit(
        spec_name=spec.name,
        dataset_variant=getattr(spec, "dataset_variant", None),
        split=split,
        fight_rows=int(len(frame)),
        family_summaries=family_summaries,
        output_dir=str(output_dir) if output_dir is not None else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="full_live_contract_v4_138")
    parser.add_argument(
        "--split",
        default="train",
        choices=["trainable", "train", "test", "recent"],
    )
    parser.add_argument("--min-fights", type=int, default=2)
    parser.add_argument("--recent-rows", type=int, default=612)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    audit = run_audit(
        spec_name=args.spec,
        split=args.split,
        min_fights=args.min_fights,
        recent_rows=args.recent_rows,
        output_dir=args.output_dir,
    )
    if args.json or args.output_dir is None:
        print(audit.to_json())
    else:
        output_path = args.output_dir / "coverage_summary.json"
        output_path.write_text(audit.to_json() + "\n", encoding="utf-8")
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
