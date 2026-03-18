"""
NamedModelTrainingSpec — explicit, reproducible model training contracts.

Defines the feature contract, hyperparameters, dataset variant, and
imputation strategy in a single, serializable specification.  Every
promoted model artifact MUST be accompanied by its spec so that the
auto-retrain path can reproduce it exactly.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Explicit live-compatible feature contract
# ---------------------------------------------------------------------------
# ONLY features that can be honestly produced at live inference time
# from UFCStats.com scraping + The Odds API should appear here.
#
# Hierarchy:  real observed value  >  NaN  >  NEVER median/default
# ---------------------------------------------------------------------------

LIVE_FEATURE_COLS: list[str] = [
    # ---- Differential features (A minus B) ----
    "diff_roll_slpm", "diff_roll_sapm", "diff_roll_str_acc", "diff_roll_str_def",
    "diff_roll_td_avg", "diff_roll_td_acc", "diff_roll_td_def", "diff_roll_sub_avg",
    "diff_roll_sig_str_landed", "diff_roll_td_landed", "diff_roll_kd",
    "diff_roll_won",
    "diff_elo",
    "diff_current_win_streak", "diff_num_fights", "diff_days_since_last_fight",
    "diff_height", "diff_reach", "diff_weight", "diff_age",
    "diff_strike_diff",
    "diff_ko_rate", "diff_sub_rate", "diff_dec_rate", "diff_win_pct",
    "diff_lose_streak", "diff_longest_win_streak",
    "diff_total_rounds", "diff_title_bouts", "diff_draws",
    "diff_cage_rust", "diff_layoff_log",
    "diff_fight_pace", "diff_ctrl_efficiency", "diff_adj_win_pct",
    "diff_striker_edge", "diff_grappler_edge",
    "diff_wc_move",

    # ---- Individual rolling / Elo (both fighters) ----
    "a_roll_slpm", "b_roll_slpm",
    "a_roll_sapm", "b_roll_sapm",
    "a_roll_str_acc", "b_roll_str_acc",
    "a_roll_str_def", "b_roll_str_def",
    "a_roll_td_avg", "b_roll_td_avg",
    "a_roll_td_acc", "b_roll_td_acc",
    "a_roll_td_def", "b_roll_td_def",
    "a_roll_sub_avg", "b_roll_sub_avg",
    "a_roll_sig_str_landed", "b_roll_sig_str_landed",
    "a_roll_td_landed", "b_roll_td_landed",
    "a_roll_kd", "b_roll_kd",
    "a_roll_won", "b_roll_won",
    "a_elo", "b_elo",
    "a_current_win_streak", "b_current_win_streak",
    "a_num_fights", "b_num_fights",
    "a_days_since_last_fight", "b_days_since_last_fight",
    "a_strike_diff", "b_strike_diff",

    # ---- Encoded categoricals ----
    "a_stance_enc", "b_stance_enc",
    "same_stance",
    "weight_class_enc",

    # ---- Physical attributes ----
    "a_height", "b_height",
    "a_reach", "b_reach",
    "a_weight", "b_weight",
    "a_age", "b_age",

    # ---- Finish method rates / win pct ----
    "a_ko_rate", "b_ko_rate",
    "a_sub_rate", "b_sub_rate",
    "a_dec_rate", "b_dec_rate",
    "a_win_pct", "b_win_pct",

    # ---- Odds-derived (from live API) ----
    "a_implied_prob", "b_implied_prob", "diff_implied_prob",

    # ---- Event context ----
    "is_title_bout", "num_rounds_feat", "is_empty_arena",

    # ---- Career records ----
    "a_lose_streak", "b_lose_streak",
    "a_longest_win_streak", "b_longest_win_streak",
    "a_total_rounds", "b_total_rounds",
    "a_title_bouts", "b_title_bouts",
    "a_draws", "b_draws",

    # ---- Cage rust / layoff ----
    "a_cage_rust", "b_cage_rust",
    "a_layoff_log", "b_layoff_log",

    # ---- Weight class moves ----
    "a_wc_move", "b_wc_move",

    # ---- Style matchup interactions ----
    "a_striker_edge", "b_striker_edge",
    "a_grappler_edge", "b_grappler_edge",

    # ---- Experimental (derivable live) ----
    "a_fight_pace", "b_fight_pace",
    "a_ctrl_efficiency", "b_ctrl_efficiency",
    "a_adj_win_pct", "b_adj_win_pct",

    # ---- Rematch / H2H (derivable from scraped fight history) ----
    "is_rematch", "h2h_record_diff",

    # ---- Elo momentum (slope over last 5 Elo values — from features.csv history) ----
    "a_elo_momentum", "b_elo_momentum", "diff_elo_momentum",

    # ---- Strength of Schedule (mean opponent Elo, last 5 — from features.csv history) ----
    "a_sos", "b_sos", "diff_sos",

    # ---- Line movement (from line_tracker snapshots / opening odds cache) ----
    "line_movement", "line_abs_movement", "line_is_sharp",
    "line_steam_move", "line_direction_toward_a", "line_direction_toward_b",

    # ---- Rankings (scraped from UFC.com / ESPN / Tapology with weekly cache) ----
    "a_wc_rank_feat", "b_wc_rank_feat", "diff_wc_rank",
    "a_pfp_rank_feat", "b_pfp_rank_feat", "diff_pfp_rank",

    # ---- Method odds (from The Odds API / BestFightOdds) ----
    "a_ko_odds_prob", "a_sub_odds_prob", "a_dec_odds_prob",
    "b_ko_odds_prob", "b_sub_odds_prob", "b_dec_odds_prob",
]

REMATCH_CONTRACT_FEATURE_COLS = [
    "is_rematch",
    "h2h_record_diff",
]

ELO_MOMENTUM_CONTRACT_FEATURE_COLS = [
    "a_elo_momentum",
    "b_elo_momentum",
    "diff_elo_momentum",
]

STRENGTH_OF_SCHEDULE_CONTRACT_FEATURE_COLS = [
    "a_sos",
    "b_sos",
    "diff_sos",
]

LINE_MOVEMENT_CONTRACT_FEATURE_COLS = [
    "line_movement",
    "line_abs_movement",
    "line_is_sharp",
    "line_steam_move",
    "line_direction_toward_a",
    "line_direction_toward_b",
]

_OPTIONAL_CONTRACT_FEATURE_COLS = set(
    REMATCH_CONTRACT_FEATURE_COLS
    + ELO_MOMENTUM_CONTRACT_FEATURE_COLS
    + STRENGTH_OF_SCHEDULE_CONTRACT_FEATURE_COLS
    + LINE_MOVEMENT_CONTRACT_FEATURE_COLS
)
PROVENANCE_STRICT_EXCLUDED_FEATURE_COLS = {
    "diff_height",
    "diff_reach",
    "diff_weight",
    "diff_age",
    "a_stance_enc",
    "b_stance_enc",
    "same_stance",
    "a_height",
    "b_height",
    "a_reach",
    "b_reach",
    "a_weight",
    "b_weight",
    "a_age",
    "b_age",
    "a_implied_prob",
    "b_implied_prob",
    "diff_implied_prob",
    "is_empty_arena",
    "a_wc_rank_feat",
    "b_wc_rank_feat",
    "diff_wc_rank",
    "a_pfp_rank_feat",
    "b_pfp_rank_feat",
    "diff_pfp_rank",
    "a_ko_odds_prob",
    "a_sub_odds_prob",
    "a_dec_odds_prob",
    "b_ko_odds_prob",
    "b_sub_odds_prob",
    "b_dec_odds_prob",
}


# Features that are in the training DataFrame but NOT available live.
# Phase 2 complete: ALL features now have live acquisition paths.
# This set should be EMPTY — any remaining entries indicate a gap.
TRAINING_ONLY_FEATURES: set[str] = set()


def contract_feature_columns(
    *,
    add_rematch_features: bool = False,
    add_elo_momentum: bool = False,
    add_strength_of_schedule: bool = False,
    add_line_movement: bool = False,
) -> list[str]:
    """Return the ordered feature contract for a spec's enabled feature groups."""
    enabled = set(LIVE_FEATURE_COLS) - _OPTIONAL_CONTRACT_FEATURE_COLS
    if add_rematch_features:
        enabled.update(REMATCH_CONTRACT_FEATURE_COLS)
    if add_elo_momentum:
        enabled.update(ELO_MOMENTUM_CONTRACT_FEATURE_COLS)
    if add_strength_of_schedule:
        enabled.update(STRENGTH_OF_SCHEDULE_CONTRACT_FEATURE_COLS)
    if add_line_movement:
        enabled.update(LINE_MOVEMENT_CONTRACT_FEATURE_COLS)
    return [column for column in LIVE_FEATURE_COLS if column in enabled]


def provenance_strict_feature_columns(
    *,
    add_rematch_features: bool = False,
    add_elo_momentum: bool = False,
    add_strength_of_schedule: bool = False,
    add_line_movement: bool = False,
) -> list[str]:
    """Return the repo-owned/live-snapshot-only contract subset."""
    feature_cols = contract_feature_columns(
        add_rematch_features=add_rematch_features,
        add_elo_momentum=add_elo_momentum,
        add_strength_of_schedule=add_strength_of_schedule,
        add_line_movement=add_line_movement,
    )
    return [column for column in feature_cols if column not in PROVENANCE_STRICT_EXCLUDED_FEATURE_COLS]


@dataclass
class NamedModelTrainingSpec:
    """
    Immutable specification for a named model configuration.

    Every field that affects the trained artifact MUST be captured here
    so that auto-retrain can reproduce the promoted model exactly.
    """

    name: str
    description: str = ""

    # ---- Feature contract ----
    feature_cols: list[str] = field(default_factory=lambda: list(LIVE_FEATURE_COLS))

    # ---- Dataset ----
    dataset_variant: str = "default"  # "default", "append_only_2026", etc.
    train_cutoff_date: str = "2022-01-01"

    # ---- Imputation ----
    # "native_nan" = let XGBoost handle NaN natively (PREFERRED)
    # "median"     = legacy median imputation (for LogisticRegression only)
    impute_strategy: str = "native_nan"
    impute_with_indicators: bool = False

    # ---- XGBoost hyperparameters ----
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
        "use_label_encoder": False,
    })

    # ---- Calibration ----
    calibration_method: str = "isotonic"  # "isotonic", "sigmoid", "none"
    calibration_cv: str = "timeseries_5fold"

    # ---- Training process ----
    time_decay_half_life: int = 730
    odds_noise_std: float = 0.04

    # ---- Extra feature flags ----
    add_rematch_features: bool = True
    add_elo_momentum: bool = False
    add_strength_of_schedule: bool = False
    add_line_movement: bool = False

    # ---- Metadata (populated at save time) ----
    git_hash: str = ""
    trained_at: str = ""

    def to_json(self) -> str:
        """Serialize to JSON for persistence alongside model artifact."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "NamedModelTrainingSpec":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save spec to a JSON file."""
        path.write_text(self.to_json())
        logger.info(f"Saved training spec to {path}")

    @classmethod
    def load(cls, path: Path) -> "NamedModelTrainingSpec":
        """Load spec from a JSON file."""
        return cls.from_json(path.read_text())

    def validate_feature_contract(self) -> list[str]:
        """
        Check that no training-only features leaked into the contract.
        Returns list of violations (empty = clean).
        """
        violations = []
        for col in self.feature_cols:
            if col in TRAINING_ONLY_FEATURES:
                violations.append(f"TRAINING_ONLY feature in contract: {col}")
        return violations


# ---------------------------------------------------------------------------
# Pre-defined specs for promoted models
# ---------------------------------------------------------------------------

def rematch_features_spec() -> NamedModelTrainingSpec:
    """Spec for the promoted rematch_features model (2026-03-16)."""
    return NamedModelTrainingSpec(
        name="rematch_features_v1",
        description="Promoted model with rematch detection, native NaN, live-compatible contract",
        feature_cols=contract_feature_columns(add_rematch_features=True),
        dataset_variant="default",
        impute_strategy="native_nan",
        impute_with_indicators=False,
        calibration_method="isotonic",
        calibration_cv="timeseries_5fold",
        add_rematch_features=True,
    )


def full_live_contract_v1_spec() -> NamedModelTrainingSpec:
    """
    Historical Phase 2 contract retained for explicit audit/compatibility use.

    This is the line-movement-inclusive contract that still relies on the
    legacy/default dataset path and therefore fails the honest pre-2022 audit.
    """
    return NamedModelTrainingSpec(
        name="full_live_contract_v1",
        description=(
            "Historical Phase 2 contract with rematch, Elo momentum, SoS, "
            "and line movement enabled on the default legacy dataset path."
        ),
        feature_cols=contract_feature_columns(
            add_rematch_features=True,
            add_elo_momentum=True,
            add_strength_of_schedule=True,
            add_line_movement=True,
        ),
        dataset_variant="default",
        impute_strategy="native_nan",
        impute_with_indicators=False,
        calibration_method="isotonic",
        calibration_cv="timeseries_5fold",
        add_rematch_features=True,
        add_elo_momentum=True,
        add_strength_of_schedule=True,
        add_line_movement=True,
    )


def full_live_contract_spec() -> NamedModelTrainingSpec:
    """
    Current promoted training contract.

    Uses the full-history best-of-both dataset variant so the real pre-2022
    train split can be populated honestly from local UFCStats data. Line
    movement stays out of the contract until honest pre-2022 history exists.
    """
    return NamedModelTrainingSpec(
        name="full_live_contract_v2",
        description=(
            "Honest promoted training contract: rematch + Elo momentum + SoS "
            "on the best_of_both_full_history dataset variant. Line movement "
            "remains out of contract pending real pre-2022 historical coverage."
        ),
        feature_cols=contract_feature_columns(
            add_rematch_features=True,
            add_elo_momentum=True,
            add_strength_of_schedule=True,
        ),
        dataset_variant="best_of_both_full_history",
        impute_strategy="native_nan",
        impute_with_indicators=False,
        calibration_method="isotonic",
        calibration_cv="timeseries_5fold",
        add_rematch_features=True,
        add_elo_momentum=True,
        add_strength_of_schedule=True,
    )


def full_live_contract_v3_spec() -> NamedModelTrainingSpec:
    """
    Provenance-strict candidate contract.

    Trains only on the pulled UFCStats history and excludes every feature family
    that still depends on legacy-carried historical values.
    """
    return NamedModelTrainingSpec(
        name="full_live_contract_v3",
        description=(
            "Provenance-strict candidate contract: pulled_all dataset with "
            "rematch + Elo momentum + SoS, excluding legacy-only physical, "
            "rankings, moneyline, method-odds, and empty-arena fields."
        ),
        feature_cols=provenance_strict_feature_columns(
            add_rematch_features=True,
            add_elo_momentum=True,
            add_strength_of_schedule=True,
        ),
        dataset_variant="pulled_all",
        impute_strategy="native_nan",
        impute_with_indicators=False,
        calibration_method="isotonic",
        calibration_cv="timeseries_5fold",
        add_rematch_features=True,
        add_elo_momentum=True,
        add_strength_of_schedule=True,
    )


def append_only_2026_spec() -> NamedModelTrainingSpec:
    """Spec for append-only 2026 dataset variant."""
    return NamedModelTrainingSpec(
        name="rematch_features_ao2026",
        description="Rematch features trained on append-only 2026 dataset",
        feature_cols=contract_feature_columns(add_rematch_features=True),
        dataset_variant="append_only_2026",
        impute_strategy="native_nan",
        impute_with_indicators=False,
        calibration_method="isotonic",
        calibration_cv="timeseries_5fold",
        add_rematch_features=True,
    )


def named_training_spec_factories() -> dict[str, Callable[[], NamedModelTrainingSpec]]:
    """Return the named training-spec registry used by CLI and audits."""
    return {
        "full_live_contract": full_live_contract_spec,
        "full_live_contract_v1": full_live_contract_v1_spec,
        "full_live_contract_v2": full_live_contract_spec,
        "full_live_contract_v3": full_live_contract_v3_spec,
        "rematch_features_v1": rematch_features_spec,
        "rematch_features_ao2026": append_only_2026_spec,
    }


def resolve_named_training_spec(spec_name: str) -> NamedModelTrainingSpec:
    """Resolve a known named training spec."""
    try:
        return named_training_spec_factories()[spec_name]()
    except KeyError as exc:
        known = ", ".join(sorted(named_training_spec_factories()))
        raise ValueError(f"Unknown spec '{spec_name}'. Known specs: {known}") from exc


def materialize_contract_transforms(
    features_df: pd.DataFrame,
    *,
    add_rematch_features: bool = False,
    add_elo_momentum: bool = False,
    add_strength_of_schedule: bool = False,
    add_line_movement: bool = False,
) -> pd.DataFrame:
    """
    Materialize any requested contract transforms onto a training features frame.

    Missing external data degrades to NaN columns rather than fabricated values.
    """
    transformed = features_df.copy()

    from src.features.build_features import materialize_honest_context_features
    from src.strategy.model_variants import (
        add_elo_momentum as add_elo_momentum_features,
        add_rematch_features as add_rematch_features_fn,
        add_strength_of_schedule as add_strength_of_schedule_features,
    )

    # Correct stale processed snapshots by rebuilding these fields from raw columns
    # whenever they are available, preserving unknowns as NaN.
    transformed = materialize_honest_context_features(transformed)

    if add_rematch_features:
        transformed = add_rematch_features_fn(transformed)
    if add_elo_momentum:
        transformed = add_elo_momentum_features(transformed)
    if add_strength_of_schedule:
        transformed = add_strength_of_schedule_features(transformed)
    if add_line_movement:
        from src.data.historical_backfill import merge_line_movement_features
        transformed = merge_line_movement_features(transformed)

    return transformed


def materialize_spec_transforms(
    features_df: pd.DataFrame,
    spec: NamedModelTrainingSpec,
) -> pd.DataFrame:
    """Apply the transforms requested by a NamedModelTrainingSpec."""
    return materialize_contract_transforms(
        features_df,
        add_rematch_features=spec.add_rematch_features,
        add_elo_momentum=spec.add_elo_momentum,
        add_strength_of_schedule=spec.add_strength_of_schedule,
        add_line_movement=spec.add_line_movement,
    )


def materialize_and_validate_spec_features(
    features_df: pd.DataFrame,
    spec: NamedModelTrainingSpec,
) -> pd.DataFrame:
    """
    Apply the spec's transforms and require every declared feature column to exist.

    Specs are structural contracts: if a declared column is absent after
    materialization, training/evaluation should fail rather than silently drop it.
    """
    transformed = materialize_spec_transforms(features_df, spec)
    missing_cols = [column for column in spec.feature_cols if column not in transformed.columns]
    if missing_cols:
        raise ValueError(
            f"Spec features missing from DataFrame after transforms: {missing_cols}"
        )
    return transformed
