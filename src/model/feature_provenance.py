"""Feature provenance audit helpers for UFC training specs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR
from src.data.kaggle_loader import load_kaggle_dataset
from src.data.ufc_refresh import (
    DEFAULT_VARIANT_OVERLAP_START,
    _LEGACY_PREFERRED_FIELDS,
    _PULLED_PREFERRED_FIELDS,
    _fight_key_series,
    _historical_moneyline_overlay,
    _historical_rankings_overlay,
    _keyed_training_frame,
    build_training_dataset_variants,
    build_training_rows_from_pulled_data,
)
from src.features.build_features import build_features
from src.model.train import prepare_train_test
from src.model.training_spec import NamedModelTrainingSpec, materialize_spec_transforms

TRAINING_SOURCE_REPO = "repo_owned_pulled_scraped"
TRAINING_SOURCE_LEGACY = "legacy_only_carried"
TRAINING_SOURCE_DEFAULT = "default_or_seeded"
LIVE_SOURCE_REPO = "repo_owned_pulled_scraped"
LIVE_SOURCE_SNAPSHOT = "live_api_snapshot"

_VALID_TRAINING_SOURCES = {"legacy", "pulled"}
_ROLLING_STAT_BASES = {
    "roll_slpm": "slpm",
    "roll_sapm": "sapm",
    "roll_str_acc": "str_acc",
    "roll_str_def": "str_def",
    "roll_td_avg": "td_avg",
    "roll_td_acc": "td_acc",
    "roll_td_def": "td_def",
    "roll_sub_avg": "sub_avg",
    "roll_sig_str_landed": "sig_str_landed",
    "roll_td_landed": "td_landed",
    "roll_kd": "kd",
    "roll_ctrl_seconds": "ctrl_seconds",
}
_DIRECT_RAW_FEATURE_MAP = {
    "a_height": "a_height",
    "b_height": "b_height",
    "a_reach": "a_reach",
    "b_reach": "b_reach",
    "a_weight": "a_weight",
    "b_weight": "b_weight",
    "a_age": "a_age",
    "b_age": "b_age",
    "a_stance_enc": "a_stance",
    "b_stance_enc": "b_stance",
    "weight_class_enc": "weight_class",
    "a_implied_prob": "a_odds",
    "b_implied_prob": "b_odds",
    "is_title_bout": "title_bout",
    "num_rounds_feat": "num_rounds",
    "is_empty_arena": "empty_arena",
    "a_lose_streak": "a_lose_streak",
    "b_lose_streak": "b_lose_streak",
    "a_longest_win_streak": "a_longest_win_streak",
    "b_longest_win_streak": "b_longest_win_streak",
    "a_total_rounds": "a_total_rounds",
    "b_total_rounds": "b_total_rounds",
    "a_title_bouts": "a_title_bouts",
    "b_title_bouts": "b_title_bouts",
    "a_draws": "a_draws",
    "b_draws": "b_draws",
    "a_wc_rank_feat": "a_wc_rank",
    "b_wc_rank_feat": "b_wc_rank",
    "a_pfp_rank_feat": "a_pfp_rank",
    "b_pfp_rank_feat": "b_pfp_rank",
    "a_ko_odds_prob": "a_ko_odds",
    "a_sub_odds_prob": "a_sub_odds",
    "a_dec_odds_prob": "a_dec_odds",
    "b_ko_odds_prob": "b_ko_odds",
    "b_sub_odds_prob": "b_sub_odds",
    "b_dec_odds_prob": "b_dec_odds",
}
_RAW_HISTORY_FEATURES = {
    "a_elo",
    "b_elo",
    "a_roll_won",
    "b_roll_won",
    "a_current_win_streak",
    "b_current_win_streak",
    "a_elo_momentum",
    "b_elo_momentum",
    "a_sos",
    "b_sos",
}
_PRESENCE_HISTORY_FEATURES = {
    "a_num_fights",
    "b_num_fights",
    "a_days_since_last_fight",
    "b_days_since_last_fight",
}
_CAREER_COUNT_HISTORY_FEATURES = {
    "a_wins",
    "b_wins",
    "a_losses",
    "b_losses",
}
_CAREER_RESULT_HISTORY_FEATURES = {
    "a_draws",
    "b_draws",
    "a_lose_streak",
    "b_lose_streak",
    "a_longest_win_streak",
    "b_longest_win_streak",
}
_CAREER_METHOD_HISTORY_FEATURES = {
    "a_wins_ko",
    "b_wins_ko",
    "a_wins_sub",
    "b_wins_sub",
    "a_wins_dec",
    "b_wins_dec",
}
_CAREER_ROUND_HISTORY_FEATURES = {
    "a_total_rounds",
    "b_total_rounds",
}
_CAREER_TITLE_HISTORY_FEATURES = {
    "a_title_bouts",
    "b_title_bouts",
}


@dataclass
class FeatureProvenanceRow:
    feature: str
    feature_family: str
    training_lineage: str
    observed_training_sources: list[str]
    live_lineage: str
    live_sources: list[str]
    train_non_null_rows: int
    train_repo_only_rows: int
    train_legacy_only_rows: int
    train_mixed_rows: int
    train_default_or_seeded_rows: int
    train_null_rows: int
    train_legacy_dependent_rows: int


@dataclass
class FeatureFamilySummary:
    feature_family: str
    feature_count: int
    train_rows_with_any_non_null_feature: int
    train_rows_with_any_legacy_dependency: int
    train_rows_repo_only_all_non_null: int
    train_rows_default_or_seeded_only: int


@dataclass
class FeatureProvenanceAudit:
    spec_name: str
    dataset_variant: str
    train_cutoff_date: str
    features_rows_total: int
    trainable_rows: int
    train_split_rows: int
    test_split_rows: int
    inventory: list[FeatureProvenanceRow]
    family_summary: list[FeatureFamilySummary]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["inventory"] = [asdict(row) for row in self.inventory]
        payload["family_summary"] = [asdict(row) for row in self.family_summary]
        return payload


def audit_training_feature_provenance(
    spec: NamedModelTrainingSpec,
    *,
    legacy_df: pd.DataFrame | None = None,
    pulled_df: pd.DataFrame | None = None,
    min_fights: int = 2,
) -> FeatureProvenanceAudit:
    """Audit train-split feature provenance for a named UFC training spec."""
    if legacy_df is None:
        legacy_df = load_kaggle_dataset(RAW_DATA_DIR / "ufc-master.csv")
    if pulled_df is None:
        pulled_df = build_training_rows_from_pulled_data(legacy_df=legacy_df)

    training_df, source_df = build_training_dataset_with_sources(
        spec.dataset_variant,
        legacy_df=legacy_df,
        pulled_df=pulled_df,
    )
    features_df = build_features(training_df)
    features_df = materialize_spec_transforms(features_df, spec)

    train_df, test_df, _ = prepare_train_test(
        features_df,
        cutoff_date=spec.train_cutoff_date,
        min_fights=min_fights,
        feature_cols=list(spec.feature_cols),
    )
    train_idx = train_df.index
    legacy_flags, repo_flags = _build_feature_source_flags(features_df, source_df, raw_df=training_df)

    inventory: list[FeatureProvenanceRow] = []
    for feature in spec.feature_cols:
        values = features_df[feature] if feature in features_df.columns else pd.Series(pd.NA, index=features_df.index)
        non_null = values.loc[train_idx].notna().to_numpy()
        legacy_used = legacy_flags.get(feature, pd.Series(False, index=features_df.index)).loc[train_idx].to_numpy()
        repo_used = repo_flags.get(feature, pd.Series(False, index=features_df.index)).loc[train_idx].to_numpy()
        buckets = _count_source_buckets(non_null, legacy_used, repo_used)
        inventory.append(
            FeatureProvenanceRow(
                feature=feature,
                feature_family=classify_feature_family(feature),
                training_lineage=describe_training_lineage(feature),
                observed_training_sources=_observed_training_sources(buckets),
                live_lineage=describe_live_lineage(feature),
                live_sources=live_source_categories(feature),
                train_non_null_rows=buckets["non_null_rows"],
                train_repo_only_rows=buckets["repo_only_rows"],
                train_legacy_only_rows=buckets["legacy_only_rows"],
                train_mixed_rows=buckets["mixed_rows"],
                train_default_or_seeded_rows=buckets["default_rows"],
                train_null_rows=buckets["null_rows"],
                train_legacy_dependent_rows=buckets["legacy_only_rows"] + buckets["mixed_rows"],
            )
        )

    family_summary = _summarize_feature_families(features_df, train_idx, inventory, legacy_flags, repo_flags)
    return FeatureProvenanceAudit(
        spec_name=spec.name,
        dataset_variant=spec.dataset_variant,
        train_cutoff_date=spec.train_cutoff_date,
        features_rows_total=int(len(features_df)),
        trainable_rows=int(len(train_df) + len(test_df)),
        train_split_rows=int(len(train_df)),
        test_split_rows=int(len(test_df)),
        inventory=inventory,
        family_summary=family_summary,
    )


def build_training_dataset_with_sources(
    dataset_variant: str | None,
    *,
    legacy_df: pd.DataFrame,
    pulled_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a dataset variant plus aligned raw-column source metadata."""
    variant_name = dataset_variant or "default"
    if variant_name == "default":
        variant_name = "legacy_only"

    variants = build_training_dataset_variants(legacy_df=legacy_df, pulled_df=pulled_df)
    if variant_name not in variants:
        known = ", ".join(sorted(variants))
        raise ValueError(f"Unknown dataset variant '{variant_name}'. Known variants: {known}")

    if variant_name == "legacy_only":
        dataset = _keyed_training_frame(variants[variant_name])
        source = _uniform_source_frame(dataset, source_label="legacy", fight_source_label="legacy")
        return dataset.drop(columns="fight_key", errors="ignore"), source

    if variant_name == "pulled_all":
        dataset = _keyed_training_frame(variants[variant_name])
        source = _uniform_source_frame(dataset, source_label="pulled", fight_source_label="pulled")
        return dataset.drop(columns="fight_key", errors="ignore"), source

    if variant_name == "pulled_all_plus_legacy_market":
        return _build_pulled_all_plus_legacy_market_with_sources(
            variants[variant_name],
            legacy_df=legacy_df,
            pulled_df=pulled_df,
        )

    if variant_name in {"best_of_both_field_level", "best_of_both_full_history"}:
        return _build_field_level_variant_with_sources(
            variant_name,
            variants[variant_name],
            legacy_df=legacy_df,
            pulled_df=pulled_df,
        )

    return _build_row_selected_variant_with_sources(
        variant_name,
        variants[variant_name],
        legacy_df=legacy_df,
        pulled_df=pulled_df,
    )


def _uniform_source_frame(
    dataset: pd.DataFrame,
    *,
    source_label: str,
    fight_source_label: str,
) -> pd.DataFrame:
    source = pd.DataFrame(index=dataset.index)
    for column in dataset.columns:
        if column == "fight_key":
            source[column] = dataset[column]
            continue
        source[column] = np.where(dataset[column].notna(), source_label, pd.NA)
    source["__fight_source"] = fight_source_label
    source["__fight_present_legacy"] = source_label == "legacy"
    source["__fight_present_pulled"] = source_label == "pulled"
    return source


def _build_row_selected_variant_with_sources(
    variant_name: str,
    dataset: pd.DataFrame,
    *,
    legacy_df: pd.DataFrame,
    pulled_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    legacy_keyed = _keyed_training_frame(legacy_df)
    pulled_keyed = _keyed_training_frame(pulled_df)
    keyed_dataset = _keyed_training_frame(dataset)

    legacy_map = {
        row["fight_key"]: row
        for _, row in legacy_keyed.dropna(subset=["fight_key"]).iterrows()
    }
    pulled_map = {
        row["fight_key"]: row
        for _, row in pulled_keyed.dropna(subset=["fight_key"]).iterrows()
    }

    legacy_max_date = pd.to_datetime(legacy_df["event_date"], errors="coerce", format="mixed").max()
    overlap_start_ts = pd.Timestamp(DEFAULT_VARIANT_OVERLAP_START)
    records: list[dict[str, Any]] = []

    for _, row in keyed_dataset.iterrows():
        fight_key = row["fight_key"]
        pulled_row = pulled_map.get(fight_key)
        legacy_row = legacy_map.get(fight_key)
        source_label = "legacy"
        if variant_name == "hybrid_legacy_first":
            source_label = "legacy" if legacy_row is not None else "pulled"
        elif variant_name == "append_only_2026":
            event_date = pd.to_datetime(row.get("event_date"), errors="coerce", format="mixed")
            source_label = "pulled" if pd.notna(event_date) and event_date > legacy_max_date else "legacy"
        elif variant_name == "hybrid_pulled_2022plus":
            event_date = pd.to_datetime(row.get("event_date"), errors="coerce", format="mixed")
            source_label = "pulled" if pd.notna(event_date) and event_date >= overlap_start_ts else "legacy"
        else:
            raise ValueError(f"Row-selected source audit is not implemented for variant '{variant_name}'")

        chosen_row = pulled_row if source_label == "pulled" else legacy_row
        if chosen_row is None:
            chosen_row = pulled_row if pulled_row is not None else legacy_row
            source_label = "pulled" if pulled_row is not None else "legacy"

        source_record = {"fight_key": fight_key}
        for column in row.index:
            if column == "fight_key":
                continue
            source_record[column] = source_label if column in chosen_row.index and pd.notna(chosen_row.get(column)) else pd.NA
        source_record["__fight_source"] = source_label
        source_record["__fight_present_legacy"] = legacy_row is not None
        source_record["__fight_present_pulled"] = pulled_row is not None
        records.append(source_record)

    source_df = pd.DataFrame(records)
    keyed_source = keyed_dataset[["fight_key"]].merge(source_df, on="fight_key", how="left")
    return keyed_dataset.drop(columns="fight_key", errors="ignore"), keyed_source


def _build_pulled_all_plus_legacy_market_with_sources(
    dataset: pd.DataFrame,
    *,
    legacy_df: pd.DataFrame,
    pulled_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keyed_dataset = _keyed_training_frame(dataset)
    source_df = _uniform_source_frame(
        keyed_dataset,
        source_label="pulled",
        fight_source_label="pulled",
    )

    legacy_keyed = _keyed_training_frame(legacy_df)
    pulled_keyed = _keyed_training_frame(pulled_df)

    source_df["__fight_present_legacy"] = source_df["fight_key"].isin(
        set(legacy_keyed["fight_key"].dropna())
    )
    source_df["__fight_present_pulled"] = source_df["fight_key"].isin(
        set(pulled_keyed["fight_key"].dropna())
    )

    historical_overlay = _historical_moneyline_overlay(keyed_dataset)
    if not historical_overlay.empty:
        source_df = source_df.merge(
            historical_overlay.rename(
                columns={
                    "a_odds__historical_overlay": "a_odds__historical_value",
                    "b_odds__historical_overlay": "b_odds__historical_value",
                }
            ),
            on="fight_key",
            how="left",
        )
        for column in ("a_odds", "b_odds"):
            historical_value_column = f"{column}__historical_value"
            source_df[column] = np.where(
                source_df[historical_value_column].notna(),
                "legacy",
                source_df[column],
            )
        source_df = source_df.drop(
            columns=["a_odds__historical_value", "b_odds__historical_value"],
            errors="ignore",
        )

    rankings_overlay = _historical_rankings_overlay(keyed_dataset)
    if not rankings_overlay.empty:
        source_df = source_df.merge(
            rankings_overlay.rename(
                columns={
                    "a_wc_rank__historical_overlay": "a_wc_rank__historical_value",
                    "b_wc_rank__historical_overlay": "b_wc_rank__historical_value",
                    "a_pfp_rank__historical_overlay": "a_pfp_rank__historical_value",
                    "b_pfp_rank__historical_overlay": "b_pfp_rank__historical_value",
                }
            ),
            on="fight_key",
            how="left",
        )
        for column in ("a_wc_rank", "b_wc_rank", "a_pfp_rank", "b_pfp_rank"):
            historical_value_column = f"{column}__historical_value"
            source_df[column] = np.where(
                source_df[historical_value_column].notna(),
                "legacy",
                source_df[column],
            )
        source_df = source_df.drop(
            columns=[
                "a_wc_rank__historical_value",
                "b_wc_rank__historical_value",
                "a_pfp_rank__historical_value",
                "b_pfp_rank__historical_value",
            ],
            errors="ignore",
        )

    legacy_market_cols = [
        "a_odds",
        "b_odds",
        "a_wc_rank",
        "b_wc_rank",
        "a_pfp_rank",
        "b_pfp_rank",
        "a_ko_odds",
        "a_sub_odds",
        "a_dec_odds",
        "b_ko_odds",
        "b_sub_odds",
        "b_dec_odds",
    ]
    legacy_market_cols = [column for column in legacy_market_cols if column in keyed_dataset.columns]
    if legacy_market_cols:
        legacy_lookup = legacy_keyed[["fight_key", *legacy_market_cols]].copy()
        legacy_lookup = legacy_lookup.dropna(subset=["fight_key"]).drop_duplicates(
            subset="fight_key",
            keep="first",
        )
        source_df = source_df.merge(
            legacy_lookup.rename(
                columns={column: f"{column}__legacy_value" for column in legacy_market_cols}
            ),
            on="fight_key",
            how="left",
        )
        for column in legacy_market_cols:
            legacy_value_column = f"{column}__legacy_value"
            source_df[column] = np.where(
                source_df[legacy_value_column].notna(),
                "legacy",
                source_df[column],
            )
        source_df = source_df.drop(
            columns=[f"{column}__legacy_value" for column in legacy_market_cols],
            errors="ignore",
        )

    return keyed_dataset.drop(columns="fight_key", errors="ignore"), source_df


def _build_field_level_variant_with_sources(
    variant_name: str,
    dataset: pd.DataFrame,
    *,
    legacy_df: pd.DataFrame,
    pulled_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    legacy_keyed = _keyed_training_frame(legacy_df)
    pulled_keyed = _keyed_training_frame(pulled_df)
    keyed_dataset = _keyed_training_frame(dataset)

    if variant_name == "best_of_both_full_history":
        overlap_start_ts = pd.to_datetime(legacy_df["event_date"], errors="coerce", format="mixed").min()
    else:
        overlap_start_ts = pd.Timestamp(DEFAULT_VARIANT_OVERLAP_START)

    legacy_map = {
        row["fight_key"]: row
        for _, row in legacy_keyed.dropna(subset=["fight_key"]).iterrows()
        if pd.notna(row.get("event_date")) and row.get("event_date") >= overlap_start_ts
    }
    pulled_map = {
        row["fight_key"]: row
        for _, row in pulled_keyed.dropna(subset=["fight_key"]).iterrows()
        if pd.notna(row.get("event_date")) and row.get("event_date") >= overlap_start_ts
    }
    legacy_before_map = {
        row["fight_key"]: row
        for _, row in legacy_keyed.dropna(subset=["fight_key"]).iterrows()
        if pd.notna(row.get("event_date")) and row.get("event_date") < overlap_start_ts
    }

    source_records: list[dict[str, Any]] = []
    for _, row in keyed_dataset.iterrows():
        fight_key = row["fight_key"]
        legacy_row = legacy_map.get(fight_key)
        pulled_row = pulled_map.get(fight_key)
        if legacy_row is None and pulled_row is None:
            legacy_row = legacy_before_map.get(fight_key)

        source_record = {
            "fight_key": fight_key,
            "__fight_source": "pulled" if pulled_row is not None else "legacy",
            "__fight_present_legacy": legacy_row is not None,
            "__fight_present_pulled": pulled_row is not None,
        }
        for column in row.index:
            if column == "fight_key":
                continue
            source_record[column] = _resolve_field_source(column, legacy_row, pulled_row)
        source_records.append(source_record)

    source_df = pd.DataFrame(source_records)
    keyed_source = keyed_dataset[["fight_key"]].merge(source_df, on="fight_key", how="left")
    return keyed_dataset.drop(columns="fight_key", errors="ignore"), keyed_source


def _resolve_field_source(column: str, legacy_row: pd.Series | None, pulled_row: pd.Series | None):
    legacy_value = legacy_row.get(column) if legacy_row is not None and column in legacy_row.index else pd.NA
    pulled_value = pulled_row.get(column) if pulled_row is not None and column in pulled_row.index else pd.NA
    legacy_present = pd.notna(legacy_value)
    pulled_present = pd.notna(pulled_value)

    if column in _LEGACY_PREFERRED_FIELDS:
        if legacy_present:
            return "legacy"
        if pulled_present:
            return "pulled"
        return pd.NA
    if column in _PULLED_PREFERRED_FIELDS:
        if pulled_present:
            return "pulled"
        if legacy_present:
            return "legacy"
        return pd.NA
    if legacy_present:
        return "legacy"
    if pulled_present:
        return "pulled"
    return pd.NA


def _build_feature_source_flags(
    features_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    raw_df: pd.DataFrame | None = None,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    source_df = _align_source_rows_to_features(features_df, source_df)
    raw_df = _align_rows_to_features(features_df, raw_df if raw_df is not None else features_df)
    legacy_flags: dict[str, pd.Series] = {}
    repo_flags: dict[str, pd.Series] = {}

    for feature, raw_column in _DIRECT_RAW_FEATURE_MAP.items():
        legacy_flags[feature], repo_flags[feature] = _raw_source_flags(source_df, raw_column)

    legacy_flags["same_stance"], repo_flags["same_stance"] = _union_source_flags(
        legacy_flags["a_stance_enc"],
        repo_flags["a_stance_enc"],
        legacy_flags["b_stance_enc"],
        repo_flags["b_stance_enc"],
    )

    history_legacy_flags, history_repo_flags = _populate_history_source_flags(features_df, raw_df, source_df)

    for feature_name in list(_ROLLING_STAT_BASES):
        for prefix in ("a_", "b_"):
            legacy_flags[f"{prefix}{feature_name}"] = history_legacy_flags[f"{prefix}{feature_name}"]
            repo_flags[f"{prefix}{feature_name}"] = history_repo_flags[f"{prefix}{feature_name}"]

    for feature_name in _RAW_HISTORY_FEATURES | _PRESENCE_HISTORY_FEATURES | {
        "a_wc_move",
        "b_wc_move",
        "is_rematch",
        "h2h_record_diff",
    }:
        legacy_flags[feature_name] = history_legacy_flags[feature_name]
        repo_flags[feature_name] = history_repo_flags[feature_name]

    for feature_name in (
        _CAREER_COUNT_HISTORY_FEATURES
        | _CAREER_RESULT_HISTORY_FEATURES
        | _CAREER_METHOD_HISTORY_FEATURES
        | _CAREER_ROUND_HISTORY_FEATURES
        | _CAREER_TITLE_HISTORY_FEATURES
    ):
        raw_column = _DIRECT_RAW_FEATURE_MAP.get(feature_name, feature_name)
        _merge_history_backfill_flags(
            source_df=source_df,
            raw_df=raw_df,
            feature_name=feature_name,
            raw_column=raw_column,
            history_legacy=history_legacy_flags[feature_name],
            history_repo=history_repo_flags[feature_name],
            legacy_flags=legacy_flags,
            repo_flags=repo_flags,
        )

    for prefix in ("a_", "b_"):
        legacy_flags[f"{prefix}win_pct"], repo_flags[f"{prefix}win_pct"] = _union_source_flags(
            legacy_flags[f"{prefix}wins"],
            repo_flags[f"{prefix}wins"],
            legacy_flags[f"{prefix}losses"],
            repo_flags[f"{prefix}losses"],
        )
        legacy_flags[f"{prefix}ko_rate"], repo_flags[f"{prefix}ko_rate"] = _union_source_flags(
            legacy_flags[f"{prefix}wins_ko"],
            repo_flags[f"{prefix}wins_ko"],
            legacy_flags[f"{prefix}wins"],
            repo_flags[f"{prefix}wins"],
        )
        legacy_flags[f"{prefix}sub_rate"], repo_flags[f"{prefix}sub_rate"] = _union_source_flags(
            legacy_flags[f"{prefix}wins_sub"],
            repo_flags[f"{prefix}wins_sub"],
            legacy_flags[f"{prefix}wins"],
            repo_flags[f"{prefix}wins"],
        )
        legacy_flags[f"{prefix}dec_rate"], repo_flags[f"{prefix}dec_rate"] = _union_source_flags(
            legacy_flags[f"{prefix}wins_dec"],
            repo_flags[f"{prefix}wins_dec"],
            legacy_flags[f"{prefix}wins"],
            repo_flags[f"{prefix}wins"],
        )
        legacy_flags[f"{prefix}strike_diff"], repo_flags[f"{prefix}strike_diff"] = _union_source_flags(
            legacy_flags[f"{prefix}roll_slpm"],
            repo_flags[f"{prefix}roll_slpm"],
            legacy_flags[f"{prefix}roll_sapm"],
            repo_flags[f"{prefix}roll_sapm"],
        )
        legacy_flags[f"{prefix}cage_rust"], repo_flags[f"{prefix}cage_rust"] = (
            legacy_flags[f"{prefix}days_since_last_fight"],
            repo_flags[f"{prefix}days_since_last_fight"],
        )
        legacy_flags[f"{prefix}layoff_log"], repo_flags[f"{prefix}layoff_log"] = (
            legacy_flags[f"{prefix}days_since_last_fight"],
            repo_flags[f"{prefix}days_since_last_fight"],
        )
        legacy_flags[f"{prefix}fight_pace"], repo_flags[f"{prefix}fight_pace"] = _union_source_flags(
            legacy_flags[f"{prefix}roll_slpm"],
            repo_flags[f"{prefix}roll_slpm"],
            legacy_flags[f"{prefix}roll_sapm"],
            repo_flags[f"{prefix}roll_sapm"],
        )
        legacy_flags[f"{prefix}ctrl_efficiency"], repo_flags[f"{prefix}ctrl_efficiency"] = _union_source_flags(
            legacy_flags[f"{prefix}roll_sig_str_landed"],
            repo_flags[f"{prefix}roll_sig_str_landed"],
            legacy_flags[f"{prefix}roll_ctrl_seconds"],
            repo_flags[f"{prefix}roll_ctrl_seconds"],
        )
        legacy_flags[f"{prefix}adj_win_pct"], repo_flags[f"{prefix}adj_win_pct"] = _union_source_flags(
            legacy_flags[f"{prefix}win_pct"],
            repo_flags[f"{prefix}win_pct"],
            legacy_flags[f"{prefix}sos"],
            repo_flags[f"{prefix}sos"],
        )

    legacy_flags["a_striker_edge"], repo_flags["a_striker_edge"] = _union_source_flags(
        legacy_flags["a_ko_rate"],
        repo_flags["a_ko_rate"],
        legacy_flags["b_roll_str_def"],
        repo_flags["b_roll_str_def"],
    )
    legacy_flags["b_striker_edge"], repo_flags["b_striker_edge"] = _union_source_flags(
        legacy_flags["b_ko_rate"],
        repo_flags["b_ko_rate"],
        legacy_flags["a_roll_str_def"],
        repo_flags["a_roll_str_def"],
    )
    legacy_flags["a_grappler_edge"], repo_flags["a_grappler_edge"] = _union_source_flags(
        legacy_flags["a_sub_rate"],
        repo_flags["a_sub_rate"],
        legacy_flags["b_roll_td_def"],
        repo_flags["b_roll_td_def"],
    )
    legacy_flags["b_grappler_edge"], repo_flags["b_grappler_edge"] = _union_source_flags(
        legacy_flags["b_sub_rate"],
        repo_flags["b_sub_rate"],
        legacy_flags["a_roll_td_def"],
        repo_flags["a_roll_td_def"],
    )

    for feature in features_df.columns:
        if not feature.startswith("diff_"):
            continue
        components = _diff_components(feature)
        if components is None:
            continue
        if any(component not in legacy_flags or component not in repo_flags for component in components):
            continue
        legacy_flags[feature], repo_flags[feature] = _union_source_flags(
            legacy_flags[components[0]],
            repo_flags[components[0]],
            legacy_flags[components[1]],
            repo_flags[components[1]],
        )

    return legacy_flags, repo_flags


def _align_rows_to_features(features_df: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    aligned = frame.copy()
    if "fight_key" not in aligned.columns:
        aligned["fight_key"] = _fight_key_series(aligned)
    feature_keys = _fight_key_series(features_df)
    aligned = aligned.drop_duplicates(subset="fight_key", keep="first").set_index("fight_key", drop=False)
    joined = pd.DataFrame({"fight_key": feature_keys}).merge(
        aligned.reset_index(drop=True),
        on="fight_key",
        how="left",
    )
    joined.index = features_df.index
    return joined


def _align_source_rows_to_features(features_df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    return _align_rows_to_features(features_df, source_df)


def _populate_history_source_flags(
    features_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    source_df: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    history_legacy_flags: dict[str, pd.Series] = {}
    history_repo_flags: dict[str, pd.Series] = {}

    for feature_name in list(_ROLLING_STAT_BASES):
        for prefix in ("a_", "b_"):
            history_legacy_flags[f"{prefix}{feature_name}"] = pd.Series(False, index=features_df.index)
            history_repo_flags[f"{prefix}{feature_name}"] = pd.Series(False, index=features_df.index)
    for feature_name in (
        _RAW_HISTORY_FEATURES
        | _PRESENCE_HISTORY_FEATURES
        | _CAREER_COUNT_HISTORY_FEATURES
        | _CAREER_RESULT_HISTORY_FEATURES
        | _CAREER_METHOD_HISTORY_FEATURES
        | _CAREER_ROUND_HISTORY_FEATURES
        | _CAREER_TITLE_HISTORY_FEATURES
        | {"a_wc_move", "b_wc_move", "is_rematch", "h2h_record_diff"}
    ):
        history_legacy_flags[feature_name] = pd.Series(False, index=features_df.index)
        history_repo_flags[feature_name] = pd.Series(False, index=features_df.index)

    fighter_stat_sources = defaultdict(lambda: defaultdict(lambda: {"legacy": 0, "pulled": 0}))
    fighter_result_sources = defaultdict(lambda: {"legacy": 0, "pulled": 0})
    fighter_method_sources = defaultdict(lambda: {"legacy": 0, "pulled": 0})
    fighter_round_sources = defaultdict(lambda: {"legacy": 0, "pulled": 0})
    fighter_title_sources = defaultdict(lambda: {"legacy": 0, "pulled": 0})
    fighter_presence_sources = defaultdict(lambda: {"legacy": 0, "pulled": 0})
    fighter_weight_class_sources = defaultdict(lambda: {"legacy": 0, "pulled": 0})
    pair_result_sources = defaultdict(lambda: {"legacy": 0, "pulled": 0})

    for idx, row in raw_df.iterrows():
        fa = row.get("fighter_a")
        fb = row.get("fighter_b")
        weight_class_source = source_df.at[idx, "weight_class"] if "weight_class" in source_df.columns else pd.NA
        pair = tuple(sorted([str(fa or ""), str(fb or "")]))

        for fighter, prefix in ((fa, "a_"), (fb, "b_")):
            if not fighter:
                continue

            for feature_name, raw_stat in _ROLLING_STAT_BASES.items():
                counter = fighter_stat_sources[fighter][raw_stat]
                history_legacy_flags[f"{prefix}{feature_name}"].at[idx] = counter["legacy"] > 0
                history_repo_flags[f"{prefix}{feature_name}"].at[idx] = counter["pulled"] > 0

            result_counter = fighter_result_sources[fighter]
            for feature_name in (
                f"{prefix}wins",
                f"{prefix}losses",
                f"{prefix}draws",
                f"{prefix}lose_streak",
                f"{prefix}longest_win_streak",
                f"{prefix}roll_won",
                f"{prefix}elo",
                f"{prefix}current_win_streak",
                f"{prefix}elo_momentum",
                f"{prefix}sos",
            ):
                history_legacy_flags[feature_name].at[idx] = result_counter["legacy"] > 0
                history_repo_flags[feature_name].at[idx] = result_counter["pulled"] > 0

            method_counter = fighter_method_sources[fighter]
            for feature_name in (
                f"{prefix}wins_ko",
                f"{prefix}wins_sub",
                f"{prefix}wins_dec",
            ):
                history_legacy_flags[feature_name].at[idx] = method_counter["legacy"] > 0
                history_repo_flags[feature_name].at[idx] = method_counter["pulled"] > 0

            round_counter = fighter_round_sources[fighter]
            history_legacy_flags[f"{prefix}total_rounds"].at[idx] = round_counter["legacy"] > 0
            history_repo_flags[f"{prefix}total_rounds"].at[idx] = round_counter["pulled"] > 0

            title_counter = fighter_title_sources[fighter]
            history_legacy_flags[f"{prefix}title_bouts"].at[idx] = title_counter["legacy"] > 0
            history_repo_flags[f"{prefix}title_bouts"].at[idx] = title_counter["pulled"] > 0

            presence_counter = fighter_presence_sources[fighter]
            for feature_name in (f"{prefix}num_fights", f"{prefix}days_since_last_fight"):
                history_legacy_flags[feature_name].at[idx] = presence_counter["legacy"] > 0
                history_repo_flags[feature_name].at[idx] = presence_counter["pulled"] > 0

            wc_counter = fighter_weight_class_sources[fighter]
            current_legacy = weight_class_source == "legacy"
            current_repo = weight_class_source == "pulled"
            history_legacy_flags[f"{prefix}wc_move"].at[idx] = wc_counter["legacy"] > 0 or current_legacy
            history_repo_flags[f"{prefix}wc_move"].at[idx] = wc_counter["pulled"] > 0 or current_repo

        pair_counter = pair_result_sources[pair]
        history_legacy_flags["is_rematch"].at[idx] = pair_counter["legacy"] > 0
        history_repo_flags["is_rematch"].at[idx] = pair_counter["pulled"] > 0
        history_legacy_flags["h2h_record_diff"].at[idx] = pair_counter["legacy"] > 0
        history_repo_flags["h2h_record_diff"].at[idx] = pair_counter["pulled"] > 0

        fight_source = source_df.at[idx, "__fight_source"] if "__fight_source" in source_df.columns else pd.NA
        for fighter in (fa, fb):
            if fighter and fight_source in _VALID_TRAINING_SOURCES:
                fighter_presence_sources[fighter][fight_source] += 1

        winner_source = source_df.at[idx, "winner"] if "winner" in source_df.columns else pd.NA
        winner_value = row.get("winner")
        if winner_source in _VALID_TRAINING_SOURCES and pd.notna(winner_value):
            for fighter in (fa, fb):
                if fighter:
                    fighter_result_sources[fighter][winner_source] += 1
            pair_result_sources[pair][winner_source] += 1

        method_sources = {
            source
            for source in (
                winner_source,
                source_df.at[idx, "method"] if "method" in source_df.columns else pd.NA,
            )
            if source in _VALID_TRAINING_SOURCES
        }
        method_value = row.get("method")
        if method_sources and pd.notna(method_value) and str(method_value).strip():
            for fighter in (fa, fb):
                if fighter:
                    for source in method_sources:
                        fighter_method_sources[fighter][source] += 1

        finish_round_source = source_df.at[idx, "finish_round"] if "finish_round" in source_df.columns else pd.NA
        if finish_round_source in _VALID_TRAINING_SOURCES and pd.notna(row.get("finish_round")):
            for fighter in (fa, fb):
                if fighter:
                    fighter_round_sources[fighter][finish_round_source] += 1

        title_bout_source = source_df.at[idx, "title_bout"] if "title_bout" in source_df.columns else pd.NA
        if title_bout_source in _VALID_TRAINING_SOURCES and pd.notna(row.get("title_bout")):
            for fighter in (fa, fb):
                if fighter:
                    fighter_title_sources[fighter][title_bout_source] += 1

        if weight_class_source in _VALID_TRAINING_SOURCES and pd.notna(row.get("weight_class")):
            for fighter in (fa, fb):
                if fighter:
                    fighter_weight_class_sources[fighter][weight_class_source] += 1

        for fighter, prefix in ((fa, "a_"), (fb, "b_")):
            if not fighter:
                continue
            for raw_stat in _ROLLING_STAT_BASES.values():
                raw_column = f"{prefix}{raw_stat}"
                if raw_column not in source_df.columns:
                    continue
                source_value = source_df.at[idx, raw_column]
                raw_value = row.get(raw_column)
                if source_value in _VALID_TRAINING_SOURCES and pd.notna(raw_value):
                    fighter_stat_sources[fighter][raw_stat][source_value] += 1

    return history_legacy_flags, history_repo_flags


def _raw_source_flags(source_df: pd.DataFrame, raw_column: str) -> tuple[pd.Series, pd.Series]:
    if raw_column not in source_df.columns:
        empty = pd.Series(False, index=source_df.index)
        return empty, empty.copy()
    series = source_df[raw_column]
    return series.eq("legacy"), series.eq("pulled")


def _raw_union(source_df: pd.DataFrame, *raw_columns: str) -> tuple[pd.Series, pd.Series]:
    flags: list[pd.Series] = []
    for raw_column in raw_columns:
        legacy, repo = _raw_source_flags(source_df, raw_column)
        flags.extend([legacy, repo])
    return _union_source_flags(*flags)


def _merge_history_backfill_flags(
    *,
    source_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    feature_name: str,
    raw_column: str,
    history_legacy: pd.Series,
    history_repo: pd.Series,
    legacy_flags: dict[str, pd.Series],
    repo_flags: dict[str, pd.Series],
) -> None:
    direct_legacy, direct_repo = legacy_flags.get(feature_name), repo_flags.get(feature_name)
    if direct_legacy is None or direct_repo is None:
        direct_legacy, direct_repo = _raw_source_flags(source_df, raw_column)

    merged_legacy = direct_legacy.copy()
    merged_repo = direct_repo.copy()
    if raw_column in raw_df.columns:
        raw_present = raw_df[raw_column].notna()
        merged_legacy.loc[~raw_present] = history_legacy.loc[~raw_present]
        merged_repo.loc[~raw_present] = history_repo.loc[~raw_present]
    else:
        merged_legacy = history_legacy.copy()
        merged_repo = history_repo.copy()

    legacy_flags[feature_name] = merged_legacy
    repo_flags[feature_name] = merged_repo


def _union_source_flags(*parts: pd.Series) -> tuple[pd.Series, pd.Series]:
    if not parts:
        raise ValueError("At least one source flag part is required")
    legacy_parts = parts[0::2]
    repo_parts = parts[1::2]
    legacy = legacy_parts[0].copy()
    repo = repo_parts[0].copy()
    for series in legacy_parts[1:]:
        legacy = legacy | series
    for series in repo_parts[1:]:
        repo = repo | series
    return legacy, repo


def _diff_components(feature: str) -> tuple[str, str] | None:
    special = {
        "diff_implied_prob": ("a_implied_prob", "b_implied_prob"),
        "diff_wc_rank": ("a_wc_rank_feat", "b_wc_rank_feat"),
        "diff_pfp_rank": ("a_pfp_rank_feat", "b_pfp_rank_feat"),
    }
    if feature in special:
        return special[feature]
    suffix = feature.removeprefix("diff_")
    return f"a_{suffix}", f"b_{suffix}"


def _count_source_buckets(
    non_null: np.ndarray,
    legacy_used: np.ndarray,
    repo_used: np.ndarray,
) -> dict[str, int]:
    repo_only = non_null & repo_used & ~legacy_used
    legacy_only = non_null & legacy_used & ~repo_used
    mixed = non_null & legacy_used & repo_used
    default_rows = non_null & ~legacy_used & ~repo_used
    return {
        "non_null_rows": int(non_null.sum()),
        "repo_only_rows": int(repo_only.sum()),
        "legacy_only_rows": int(legacy_only.sum()),
        "mixed_rows": int(mixed.sum()),
        "default_rows": int(default_rows.sum()),
        "null_rows": int((~non_null).sum()),
    }


def _observed_training_sources(buckets: dict[str, int]) -> list[str]:
    sources: list[str] = []
    if buckets["repo_only_rows"] > 0 or buckets["mixed_rows"] > 0:
        sources.append(TRAINING_SOURCE_REPO)
    if buckets["legacy_only_rows"] > 0 or buckets["mixed_rows"] > 0:
        sources.append(TRAINING_SOURCE_LEGACY)
    if buckets["default_rows"] > 0:
        sources.append(TRAINING_SOURCE_DEFAULT)
    return sources


def _summarize_feature_families(
    features_df: pd.DataFrame,
    train_idx: pd.Index,
    inventory: list[FeatureProvenanceRow],
    legacy_flags: dict[str, pd.Series],
    repo_flags: dict[str, pd.Series],
) -> list[FeatureFamilySummary]:
    families: list[FeatureFamilySummary] = []
    for family in sorted({row.feature_family for row in inventory}):
        family_features = [row.feature for row in inventory if row.feature_family == family]
        any_non_null = np.zeros(len(train_idx), dtype=bool)
        any_legacy = np.zeros(len(train_idx), dtype=bool)
        any_repo = np.zeros(len(train_idx), dtype=bool)
        any_default = np.zeros(len(train_idx), dtype=bool)
        for feature in family_features:
            values = features_df.loc[train_idx, feature] if feature in features_df.columns else pd.Series(pd.NA, index=train_idx)
            non_null = values.notna().to_numpy()
            legacy = legacy_flags.get(feature, pd.Series(False, index=features_df.index)).loc[train_idx].to_numpy()
            repo = repo_flags.get(feature, pd.Series(False, index=features_df.index)).loc[train_idx].to_numpy()
            any_non_null |= non_null
            any_legacy |= non_null & legacy
            any_repo |= non_null & repo
            any_default |= non_null & ~legacy & ~repo
        families.append(
            FeatureFamilySummary(
                feature_family=family,
                feature_count=len(family_features),
                train_rows_with_any_non_null_feature=int(any_non_null.sum()),
                train_rows_with_any_legacy_dependency=int(any_legacy.sum()),
                train_rows_repo_only_all_non_null=int((any_non_null & any_repo & ~any_legacy).sum()),
                train_rows_default_or_seeded_only=int((any_non_null & any_default & ~any_legacy & ~any_repo).sum()),
            )
        )
    return families


def classify_feature_family(feature: str) -> str:
    """Return a high-level feature family for provenance summaries."""
    core = feature
    for prefix in ("diff_", "a_", "b_"):
        if core.startswith(prefix):
            core = core[len(prefix):]
            break

    if "implied_prob" in feature or feature.endswith("_odds_prob"):
        return "market_odds"
    if "rank" in feature:
        return "rankings"
    if core in {"height", "reach", "weight", "age", "stance_enc", "same_stance"}:
        return "physical_profile"
    if core in {"weight_class_enc", "is_title_bout", "num_rounds_feat", "is_empty_arena", "wc_move"}:
        return "event_context"
    if core.startswith("roll_") or core in {
        "elo",
        "current_win_streak",
        "num_fights",
        "days_since_last_fight",
        "strike_diff",
        "cage_rust",
        "layoff_log",
        "elo_momentum",
        "sos",
    }:
        return "historical_performance"
    if core in {"ko_rate", "sub_rate", "dec_rate", "win_pct", "lose_streak", "longest_win_streak", "total_rounds", "title_bouts", "draws"}:
        return "career_record"
    if core in {"fight_pace", "ctrl_efficiency", "adj_win_pct"}:
        return "experimental"
    if core in {"striker_edge", "grappler_edge"}:
        return "style_matchup"
    if core in {"is_rematch", "h2h_record_diff"}:
        return "rematch_h2h"
    return "misc"


def describe_training_lineage(feature: str) -> str:
    """Return a short training lineage description for a feature."""
    if feature in _DIRECT_RAW_FEATURE_MAP:
        raw_column = _DIRECT_RAW_FEATURE_MAP[feature]
        if feature in (
            _CAREER_RESULT_HISTORY_FEATURES
            | _CAREER_ROUND_HISTORY_FEATURES
            | _CAREER_TITLE_HISTORY_FEATURES
        ):
            return f"career-record field from raw `{raw_column}` with prior-fight-history backfill when raw is absent"
        if feature.endswith("_odds_prob") or feature in {"a_implied_prob", "b_implied_prob"}:
            return f"direct implied-probability transform from raw `{raw_column}`"
        if feature in {"is_title_bout", "num_rounds_feat", "is_empty_arena"}:
            return f"direct context transform from raw `{raw_column}`"
        return f"direct feature from raw `{raw_column}`"

    if feature == "same_stance":
        return "direct comparison of fighter stance raw columns"
    if feature in {"is_rematch", "h2h_record_diff"}:
        return "pairwise history derived from prior fight results"
    if feature.endswith("elo_momentum"):
        return "slope of prior fighter Elo history"
    if feature.endswith("sos"):
        return "mean prior opponent Elo at fight time"
    if feature.endswith("wc_move"):
        return "current weight class versus prior fighter weight-class history"
    if feature.startswith("diff_"):
        return "difference of fighter A and fighter B feature values"
    if feature.startswith(("a_roll_", "b_roll_")):
        base = feature.split("roll_", 1)[1]
        return f"rolling history of per-fight `{base}` observations"
    if feature.startswith(("a_elo", "b_elo", "diff_elo")) or "current_win_streak" in feature:
        return "chronological fight-result history"
    if feature.endswith(("num_fights", "days_since_last_fight")) or feature.startswith(("diff_num_fights", "diff_days_since_last_fight")):
        return "chronological fight-presence history"
    if feature.endswith(("fight_pace", "ctrl_efficiency", "adj_win_pct", "striker_edge", "grappler_edge", "strike_diff")):
        return "derived from upstream contract features in the same row"
    if feature.endswith(("win_pct", "ko_rate", "sub_rate", "dec_rate")):
        return "career-record columns with prior-fight-history backfill when raw values are absent"
    return "derived during feature materialization"


def describe_live_lineage(feature: str) -> str:
    """Return a short live-inference lineage description for a feature."""
    family = classify_feature_family(feature)
    if family == "market_odds":
        return "live bookmaker snapshot / odds API inputs"
    if family == "rankings":
        return "live rankings snapshot"
    if family == "event_context":
        return "live event context snapshot plus local fight history where applicable"
    if family in {"historical_performance", "career_record", "physical_profile", "experimental", "style_matchup", "rematch_h2h"}:
        return "local processed fight history plus UFCStats fighter/fight scrape"
    return "local processed history and live snapshot inputs"


def live_source_categories(feature: str) -> list[str]:
    """Return the broad live source categories for a feature."""
    family = classify_feature_family(feature)
    if family in {"market_odds", "rankings"}:
        return [LIVE_SOURCE_SNAPSHOT]
    if family == "event_context":
        if feature in {"a_wc_move", "b_wc_move", "diff_wc_move"}:
            return [LIVE_SOURCE_REPO, LIVE_SOURCE_SNAPSHOT]
        return [LIVE_SOURCE_SNAPSHOT]
    return [LIVE_SOURCE_REPO]


def resolve_spec_from_name(spec_name: str) -> NamedModelTrainingSpec:
    """Resolve a known promoted training spec by name."""
    try:
        from src.model.training_spec import resolve_named_training_spec

        return resolve_named_training_spec(spec_name)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
