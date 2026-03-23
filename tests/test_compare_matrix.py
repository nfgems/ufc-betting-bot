import pandas as pd
import pytest

from src.features.build_features import (
    BETSAPI_CHALLENGER_FEATURE_NAMES,
    BETSAPI_HISTORICAL_FEATURE_NAMES,
    UFCSTATS_REFRESH_ONLY_FEATURE_NAMES,
)
from src.strategy.compare_matrix import (
    ALL_FEATURE_FAMILIES,
    _select_feature_columns,
    _slice_predictions,
)


def test_slice_predictions_deduplicates_feature_rows_for_betsapi_coverage():
    predictions_df = pd.DataFrame(
        [
            {
                "event_date": "2025-01-01",
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "target": 1,
                "baseline_prob": 0.60,
                "challenger_prob": 0.62,
            }
        ]
    )
    features_df = pd.DataFrame(
        [
            {
                "event_date": "2025-01-01",
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "betsapi_hist_opening_a_implied_prob": 0.55,
            },
            {
                "event_date": "2025-01-01",
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "betsapi_hist_opening_a_implied_prob": 0.55,
            },
        ]
    )

    slices = _slice_predictions(predictions_df, features_df)

    assert len(slices["high_betsapi_coverage"]) == 1
    assert slices["high_betsapi_coverage"].iloc[0]["fighter_a"] == "Fighter A"
    assert slices["low_betsapi_coverage"].empty


# ---------------------------------------------------------------------------
# T1 — _select_feature_columns parametrised smoke test
# ---------------------------------------------------------------------------

def _build_representative_features_df() -> pd.DataFrame:
    """Build a minimal DataFrame with representative columns from every family."""
    row: dict = {
        "event_date": "2025-01-01",
        "fighter_a": "A",
        "fighter_b": "B",
        "target": 1,
        # Core production columns (differential, rolling, num_fights, implied_prob)
        "diff_skill": 0.1,
        "a_roll_slpm": 3.5,
        "b_roll_slpm": 3.2,
        "a_num_fights": 8,
        "b_num_fights": 7,
        "a_implied_prob": 0.55,
        "b_implied_prob": 0.45,
        "diff_implied_prob": 0.10,
    }
    # UFCSTATS refresh-only columns
    for col in UFCSTATS_REFRESH_ONLY_FEATURE_NAMES:
        row[col] = 0.5
    # All BetsAPI challenger columns (superset of historical)
    for col in BETSAPI_CHALLENGER_FEATURE_NAMES:
        row[col] = 0.5
    return pd.DataFrame([row])


@pytest.mark.parametrize("family", list(ALL_FEATURE_FAMILIES))
def test_select_feature_columns_returns_valid_split_for_each_family(family: str):
    df = _build_representative_features_df()
    baseline_cols, challenger_cols = _select_feature_columns(df, family)

    # Baseline is always production (no refresh-only, no betsapi)
    assert not any(c in UFCSTATS_REFRESH_ONLY_FEATURE_NAMES for c in baseline_cols)
    assert not any(c.startswith("betsapi_") for c in baseline_cols)

    if family == "production":
        assert challenger_cols == baseline_cols

    elif family == "production_betsapi":
        # Historical BetsAPI cols present, but NOT snapshot/expanded-only
        hist_in_challenger = [c for c in challenger_cols if c in BETSAPI_HISTORICAL_FEATURE_NAMES]
        assert len(hist_in_challenger) == len(BETSAPI_HISTORICAL_FEATURE_NAMES)
        snapshot_only = set(BETSAPI_CHALLENGER_FEATURE_NAMES) - set(BETSAPI_HISTORICAL_FEATURE_NAMES)
        assert not any(c in snapshot_only for c in challenger_cols)

    elif family == "production_betsapi_expanded":
        # Full challenger BetsAPI cols present
        for col in BETSAPI_CHALLENGER_FEATURE_NAMES:
            assert col in challenger_cols

    elif family == "ufcstats_refreshed":
        # Includes refresh-only, no betsapi
        for col in UFCSTATS_REFRESH_ONLY_FEATURE_NAMES:
            assert col in challenger_cols
        assert not any(c.startswith("betsapi_") for c in challenger_cols)

    elif family == "ufcstats_betsapi_expanded":
        # Refreshed base + full challenger BetsAPI
        for col in UFCSTATS_REFRESH_ONLY_FEATURE_NAMES:
            assert col in challenger_cols
        for col in BETSAPI_CHALLENGER_FEATURE_NAMES:
            assert col in challenger_cols

    elif family == "no_odds":
        # No market-derived columns
        assert "a_implied_prob" not in challenger_cols
        assert "b_implied_prob" not in challenger_cols
        assert not any(c.startswith("betsapi_") for c in challenger_cols)


def test_select_feature_columns_raises_on_unknown_family():
    df = _build_representative_features_df()
    with pytest.raises(ValueError, match="Unknown feature family"):
        _select_feature_columns(df, "nonexistent_family")


# ---------------------------------------------------------------------------
# Fix #3 test — family-aware BetsAPI coverage slicing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "family,expected_coverage",
    [
        # production_betsapi: only historical cols → fight with ONLY historical signal is "high"
        ("production_betsapi", True),
        # production_betsapi_expanded: full challenger cols → fight with ONLY historical signal
        # still counts as covered (historical is subset of challenger)
        ("production_betsapi_expanded", True),
    ],
)
def test_slice_predictions_uses_correct_feature_names_for_family(family: str, expected_coverage: bool):
    """_slice_predictions should use HISTORICAL feature names for production_betsapi
    and CHALLENGER feature names for expanded families."""
    # Create a fight with ONLY historical BetsAPI signal (no snapshot cols)
    hist_col = "betsapi_hist_opening_a_implied_prob"  # in BOTH historical and challenger
    snapshot_col = "betsapi_snapshot_count"  # ONLY in challenger, NOT in historical

    predictions_df = pd.DataFrame(
        [
            {
                "event_date": "2025-06-01",
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "target": 1,
                "baseline_prob": 0.60,
                "challenger_prob": 0.62,
            }
        ]
    )
    features_df = pd.DataFrame(
        [
            {
                "event_date": "2025-06-01",
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                hist_col: 0.55,
                snapshot_col: float("nan"),  # no snapshot signal
            }
        ]
    )

    slices = _slice_predictions(predictions_df, features_df, feature_family=family)

    if expected_coverage:
        assert len(slices["high_betsapi_coverage"]) == 1
        assert slices["low_betsapi_coverage"].empty
    else:
        assert slices["high_betsapi_coverage"].empty
        assert len(slices["low_betsapi_coverage"]) == 1


@pytest.mark.parametrize(
    "family,expected_coverage",
    [
        # production_betsapi uses HISTORICAL names — snapshot_count is NOT in that
        # list, so a fight with ONLY snapshot signal is NOT covered.
        ("production_betsapi", False),
        # production_betsapi_expanded uses CHALLENGER names — snapshot_count IS in
        # that list, so the same fight IS covered.
        ("production_betsapi_expanded", True),
    ],
)
def test_slice_predictions_snapshot_only_signal_diverges_by_family(family: str, expected_coverage: bool):
    """A fight with signal ONLY in a challenger-exclusive column must be classified
    differently depending on the feature family.  This is the discriminating case
    that fails if family-aware dispatch is removed."""
    snapshot_col = "betsapi_snapshot_count"  # in CHALLENGER, NOT in HISTORICAL
    hist_col = "betsapi_hist_opening_a_implied_prob"  # in BOTH lists

    predictions_df = pd.DataFrame(
        [
            {
                "event_date": "2025-06-01",
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "target": 1,
                "baseline_prob": 0.60,
                "challenger_prob": 0.62,
            }
        ]
    )
    features_df = pd.DataFrame(
        [
            {
                "event_date": "2025-06-01",
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                snapshot_col: 3.0,  # snapshot signal present
                hist_col: float("nan"),  # NO historical signal
            }
        ]
    )

    slices = _slice_predictions(predictions_df, features_df, feature_family=family)

    if expected_coverage:
        assert len(slices["high_betsapi_coverage"]) == 1
        assert slices["low_betsapi_coverage"].empty
    else:
        assert slices["high_betsapi_coverage"].empty
        assert len(slices["low_betsapi_coverage"]) == 1
