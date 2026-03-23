import pandas as pd

from src.model.feature_provenance import (
    _build_feature_source_flags,
    audit_training_feature_provenance,
    build_training_dataset_with_sources,
)
from src.model.training_spec import NamedModelTrainingSpec


def _toy_legacy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2020-01-01"),
                "fighter_a": "Alpha",
                "fighter_b": "Bravo",
                "winner": "Alpha",
                "target": 1,
                "weight_class": "Lightweight",
                "title_bout": 0.0,
                "num_rounds": 3.0,
                "a_slpm": 1.1,
                "b_slpm": 0.7,
                "a_str_acc": 45.0,
                "b_str_acc": 38.0,
                "a_td_avg": 1.2,
                "b_td_avg": 0.3,
                "a_td_acc": 40.0,
                "b_td_acc": 25.0,
                "a_sub_avg": 0.4,
                "b_sub_avg": 0.1,
                "a_odds": -140.0,
                "b_odds": 120.0,
                "a_height": 180.0,
                "b_height": 175.0,
            },
            {
                "event_date": pd.Timestamp("2020-06-01"),
                "fighter_a": "Alpha",
                "fighter_b": "Charlie",
                "winner": "Alpha",
                "target": 1,
                "weight_class": "Lightweight",
                "title_bout": 0.0,
                "num_rounds": 3.0,
                "a_slpm": 2.0,
                "b_slpm": 1.0,
                "a_str_acc": 48.0,
                "b_str_acc": 35.0,
                "a_td_avg": 0.8,
                "b_td_avg": 0.4,
                "a_td_acc": 42.0,
                "b_td_acc": 22.0,
                "a_sub_avg": 0.3,
                "b_sub_avg": 0.2,
                "a_odds": -150.0,
                "b_odds": 130.0,
                "a_height": 180.0,
                "b_height": 178.0,
            },
            {
                "event_date": pd.Timestamp("2021-01-01"),
                "fighter_a": "Alpha",
                "fighter_b": "Delta",
                "winner": "Alpha",
                "target": 1,
                "weight_class": "Lightweight",
                "title_bout": 0.0,
                "num_rounds": 3.0,
                "a_slpm": 2.4,
                "b_slpm": 1.1,
                "a_str_acc": 50.0,
                "b_str_acc": 33.0,
                "a_td_avg": 0.9,
                "b_td_avg": 0.3,
                "a_td_acc": 44.0,
                "b_td_acc": 20.0,
                "a_sub_avg": 0.5,
                "b_sub_avg": 0.1,
                "a_odds": -170.0,
                "b_odds": 145.0,
                "a_height": 180.0,
                "b_height": 182.0,
            },
        ]
    )


def _toy_pulled_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2020-06-01"),
                "fighter_a": "Alpha",
                "fighter_b": "Charlie",
                "winner": "Alpha",
                "target": 1,
                "weight_class": "Lightweight",
                "title_bout": 1.0,
                "num_rounds": 5.0,
                "a_slpm": 1.9,
                "b_slpm": 0.9,
                "a_str_acc": 47.0,
                "b_str_acc": 36.0,
                "a_td_avg": 0.75,
                "b_td_avg": 0.45,
                "a_td_acc": 41.0,
                "b_td_acc": 24.0,
                "a_sub_avg": 0.25,
                "b_sub_avg": 0.15,
                "a_sapm": 0.6,
                "b_sapm": 1.4,
                "a_str_def": 59.0,
                "b_str_def": 51.0,
                "a_td_def": 71.0,
                "b_td_def": 46.0,
                "a_sig_str_landed": 31.0,
                "b_sig_str_landed": 19.0,
                "a_td_landed": 2.0,
                "b_td_landed": 1.0,
                "a_kd": 1.0,
                "b_kd": 0.0,
                "a_ctrl_seconds": 95.0,
                "b_ctrl_seconds": 42.0,
            }
        ]
    )


def _toy_pulled_history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2020-01-01"),
                "fighter_a": "Alpha",
                "fighter_b": "Bravo",
                "winner": "Alpha",
                "target": 1,
                "weight_class": "Lightweight",
                "title_bout": 0.0,
                "num_rounds": 3.0,
                "finish_round": 3.0,
                "method": "Decision - Unanimous",
                "a_slpm": 1.2,
                "b_slpm": 0.9,
                "a_sapm": 0.8,
                "b_sapm": 1.1,
                "a_str_acc": 46.0,
                "b_str_acc": 39.0,
                "a_str_def": 58.0,
                "b_str_def": 49.0,
                "a_td_avg": 0.4,
                "b_td_avg": 0.2,
                "a_td_acc": 32.0,
                "b_td_acc": 20.0,
                "a_td_def": 74.0,
                "b_td_def": 52.0,
                "a_sub_avg": 0.1,
                "b_sub_avg": 0.0,
                "a_sig_str_landed": 25.0,
                "b_sig_str_landed": 19.0,
                "a_td_landed": 1.0,
                "b_td_landed": 0.0,
                "a_kd": 0.0,
                "b_kd": 0.0,
                "a_ctrl_seconds": 45.0,
                "b_ctrl_seconds": 12.0,
            },
            {
                "event_date": pd.Timestamp("2020-06-01"),
                "fighter_a": "Alpha",
                "fighter_b": "Charlie",
                "winner": "Charlie",
                "target": 0,
                "weight_class": "Lightweight",
                "title_bout": 1.0,
                "num_rounds": 5.0,
                "finish_round": 2.0,
                "method": "KO/TKO",
                "a_slpm": 1.0,
                "b_slpm": 1.8,
                "a_sapm": 1.4,
                "b_sapm": 0.9,
                "a_str_acc": 40.0,
                "b_str_acc": 48.0,
                "a_str_def": 51.0,
                "b_str_def": 55.0,
                "a_td_avg": 0.3,
                "b_td_avg": 0.1,
                "a_td_acc": 25.0,
                "b_td_acc": 18.0,
                "a_td_def": 63.0,
                "b_td_def": 68.0,
                "a_sub_avg": 0.0,
                "b_sub_avg": 0.2,
                "a_sig_str_landed": 16.0,
                "b_sig_str_landed": 27.0,
                "a_td_landed": 0.0,
                "b_td_landed": 1.0,
                "a_kd": 0.0,
                "b_kd": 1.0,
                "a_ctrl_seconds": 18.0,
                "b_ctrl_seconds": 35.0,
            },
            {
                "event_date": pd.Timestamp("2021-01-01"),
                "fighter_a": "Alpha",
                "fighter_b": "Delta",
                "winner": "Alpha",
                "target": 1,
                "weight_class": "Lightweight",
                "title_bout": 0.0,
                "num_rounds": 3.0,
                "finish_round": 1.0,
                "method": "Submission",
                "a_slpm": 2.0,
                "b_slpm": 1.0,
                "a_sapm": 0.7,
                "b_sapm": 1.3,
                "a_str_acc": 51.0,
                "b_str_acc": 35.0,
                "a_str_def": 60.0,
                "b_str_def": 47.0,
                "a_td_avg": 0.6,
                "b_td_avg": 0.2,
                "a_td_acc": 38.0,
                "b_td_acc": 22.0,
                "a_td_def": 76.0,
                "b_td_def": 50.0,
                "a_sub_avg": 0.5,
                "b_sub_avg": 0.1,
                "a_sig_str_landed": 30.0,
                "b_sig_str_landed": 20.0,
                "a_td_landed": 2.0,
                "b_td_landed": 0.0,
                "a_kd": 1.0,
                "b_kd": 0.0,
                "a_ctrl_seconds": 88.0,
                "b_ctrl_seconds": 16.0,
            },
        ]
    )


def test_build_training_dataset_with_sources_respects_field_level_preferences():
    training_df, source_df = build_training_dataset_with_sources(
        "best_of_both_full_history",
        legacy_df=_toy_legacy_frame(),
        pulled_df=_toy_pulled_frame(),
    )

    overlap_idx = training_df.index[training_df["fighter_b"] == "Charlie"][0]

    assert source_df.loc[overlap_idx, "a_height"] == "legacy"
    assert source_df.loc[overlap_idx, "a_sapm"] == "pulled"
    assert source_df.loc[overlap_idx, "title_bout"] == "pulled"
    assert source_df.loc[overlap_idx, "a_slpm"] == "legacy"


def test_build_training_dataset_with_sources_marks_legacy_market_overlay_on_pulled_all():
    training_df, source_df = build_training_dataset_with_sources(
        "pulled_all_plus_legacy_market",
        legacy_df=_toy_legacy_frame(),
        pulled_df=_toy_pulled_frame(),
    )

    overlap_idx = training_df.index[training_df["fighter_b"] == "Charlie"][0]

    assert source_df.loc[overlap_idx, "__fight_source"] == "pulled"
    assert bool(source_df.loc[overlap_idx, "__fight_present_legacy"]) is True
    assert bool(source_df.loc[overlap_idx, "__fight_present_pulled"]) is True
    assert source_df.loc[overlap_idx, "a_odds"] == "legacy"
    assert source_df.loc[overlap_idx, "b_odds"] == "legacy"
    assert source_df.loc[overlap_idx, "a_sapm"] == "pulled"
    assert source_df.loc[overlap_idx, "title_bout"] == "pulled"


def test_feature_source_flags_align_by_fight_key_not_row_order():
    training_df, source_df = build_training_dataset_with_sources(
        "best_of_both_full_history",
        legacy_df=_toy_legacy_frame(),
        pulled_df=_toy_pulled_frame(),
    )
    features_df = training_df.copy()
    features_df["event_date"] = pd.to_datetime(features_df["event_date"])
    from src.features.build_features import build_features

    built = build_features(features_df)
    shuffled_source = source_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    legacy_flags, repo_flags = _build_feature_source_flags(built, shuffled_source, raw_df=training_df)

    assert legacy_flags["a_height"][built["a_height"].notna()].all()
    assert not repo_flags["a_height"][built["a_height"].notna()].any()


def test_audit_training_feature_provenance_reports_expected_source_buckets():
    spec = NamedModelTrainingSpec(
        name="toy_provenance",
        feature_cols=[
            "a_height",
            "a_implied_prob",
            "a_roll_slpm",
            "a_roll_sapm",
            "is_title_bout",
        ],
        dataset_variant="best_of_both_full_history",
        train_cutoff_date="2022-01-01",
    )

    result = audit_training_feature_provenance(
        spec,
        legacy_df=_toy_legacy_frame(),
        pulled_df=_toy_pulled_frame(),
        min_fights=0,
    )
    inventory = {row.feature: row for row in result.inventory}

    assert inventory["a_height"].train_legacy_only_rows > 0
    assert inventory["a_height"].train_repo_only_rows == 0
    assert inventory["a_implied_prob"].train_legacy_only_rows > 0
    assert inventory["a_roll_sapm"].train_repo_only_rows > 0
    assert inventory["a_roll_sapm"].train_legacy_dependent_rows == 0
    assert inventory["a_roll_slpm"].train_legacy_dependent_rows > 0
    assert inventory["is_title_bout"].train_repo_only_rows > 0


def test_audit_provenance_marks_history_backfilled_career_features_repo_owned_on_pulled_all():
    spec = NamedModelTrainingSpec(
        name="toy_pulled_career_history",
        feature_cols=[
            "a_lose_streak",
            "a_total_rounds",
            "a_title_bouts",
            "a_win_pct",
            "a_ko_rate",
        ],
        dataset_variant="pulled_all",
        train_cutoff_date="2022-01-01",
    )

    result = audit_training_feature_provenance(
        spec,
        legacy_df=_toy_legacy_frame(),
        pulled_df=_toy_pulled_history_frame(),
        min_fights=0,
    )
    inventory = {row.feature: row for row in result.inventory}

    for feature in inventory.values():
        assert feature.train_legacy_dependent_rows == 0

    assert inventory["a_lose_streak"].train_repo_only_rows > 0
    assert inventory["a_total_rounds"].train_repo_only_rows > 0
    assert inventory["a_title_bouts"].train_repo_only_rows > 0
    assert inventory["a_win_pct"].train_repo_only_rows > 0
    assert inventory["a_ko_rate"].train_repo_only_rows > 0
