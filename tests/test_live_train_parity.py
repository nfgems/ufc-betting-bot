"""
Live/train feature parity regression tests (E1).

Replays recent historical fights through ``build_fight_features`` on the
date-aware processed path (no network) and asserts the produced features
match the stored training rows in ``features.csv``.

Two oracles:

- exact mode (as_of = event_date): the snapshot serves the training row, so
  any divergence isolates logic computed on top of the lookup (opp_strength
  strength-of-schedule, pre-UFC/amateur summaries, diffs, encodings).
- prefight mode (as_of = event_date - 1): the lookup must reconstruct the
  same pre-fight state without seeing the fight row, exercising the
  roll-forward path that live falls back to when scraping fails. Time-aged
  fields (age/layoff families) legitimately differ by the one-day shift and
  are excluded; everything else must match.

These tests use the real processed artifacts (data/processed). They are
skipped when the artifacts are absent (e.g. a bare checkout).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import PROCESSED_DATA_DIR
from src.data.fighter_lookup import build_fight_features
from src.model.training_spec import resolve_named_training_spec

PARITY_PROCESSED_DATA_DIR = Path(
    os.environ.get("UFC_PARITY_PROCESSED_DATA_DIR", str(PROCESSED_DATA_DIR))
).resolve(strict=False)
PARITY_TRAINING_SPEC = os.environ.get(
    "UFC_PARITY_TRAINING_SPEC", "full_live_contract_v6_fullfit"
)
PARITY_OVERRIDE = "UFC_PARITY_PROCESSED_DATA_DIR" in os.environ
FEATURES_PATH = PARITY_PROCESSED_DATA_DIR / "features.csv"
FIGHTS_PATH = PARITY_PROCESSED_DATA_DIR / "fights_cleaned.csv"

if PARITY_OVERRIDE and not (FEATURES_PATH.is_file() and FIGHTS_PATH.is_file()):
    raise RuntimeError(
        "explicit UFC_PARITY_PROCESSED_DATA_DIR is missing features.csv or "
        "fights_cleaned.csv"
    )

pytestmark = pytest.mark.skipif(
    not PARITY_OVERRIDE and not (FEATURES_PATH.exists() and FIGHTS_PATH.exists()),
    reason="processed artifacts not available",
)

N_FIGHTS = 25
TOLERANCE = 1e-6

# Live values of these fields depend on the reference timestamp, so the
# one-day prefight shift (and snapshot aging) legitimately moves them.
TIME_AGED_ROOTS = (
    "days_since_last_fight", "layoff_log", "cage_rust",
    "age", "age_squared", "age_over_35", "age_under_25",
)

ODDS_PASSTHROUGH = [
    "a_implied_prob", "b_implied_prob", "diff_implied_prob",
    "a_ko_odds_prob", "a_sub_odds_prob", "a_dec_odds_prob",
    "b_ko_odds_prob", "b_sub_odds_prob", "b_dec_odds_prob",
]


def _is_time_aged(col: str) -> bool:
    return any(col == root or col.endswith(root) for root in TIME_AGED_ROOTS)


def _load_replay_sample():
    spec = resolve_named_training_spec(PARITY_TRAINING_SPEC)
    df = pd.read_csv(FEATURES_PATH, parse_dates=["event_date"])
    df = df.sort_values("event_date")

    # Prefight mode needs both fighters to have prior processed history
    # (debutants are served by the live scrape path, not the snapshot).
    seen: set[str] = set()
    has_history = []
    for row in df.itertuples(index=False):
        has_history.append(row.fighter_a in seen and row.fighter_b in seen)
        seen.add(row.fighter_a)
        seen.add(row.fighter_b)
    df = df[pd.Series(has_history, index=df.index)]

    recent = df[df["event_date"] >= df["event_date"].max() - pd.Timedelta(days=270)]
    rng = np.random.RandomState(13)
    if len(recent) > N_FIGHTS:
        recent = recent.iloc[sorted(rng.choice(len(recent), size=N_FIGHTS, replace=False))]
    return spec, recent


def _replay_fight(row, spec, as_of: str) -> dict:
    odds_features = {col: row[col] for col in ODDS_PASSTHROUGH if col in row.index}
    return build_fight_features(
        row["fighter_a"],
        row["fighter_b"],
        odds_features=odds_features,
        weight_class=row.get("weight_class") if isinstance(row.get("weight_class"), str) else None,
        is_title_bout=bool(row.get("is_title_bout", 0)),
        num_rounds=int(row["num_rounds_feat"]) if pd.notna(row.get("num_rounds_feat")) else 3,
        is_empty_arena=row.get("is_empty_arena"),
        as_of_date=as_of,
        training_spec=spec,
        processed_data_dir=PARITY_PROCESSED_DATA_DIR,
    )


def _collect_mismatches(live: dict, row, feature_cols, *, skip_time_aged: bool) -> list[str]:
    mismatches = []
    for col in feature_cols:
        if col not in row.index:
            continue
        if skip_time_aged and _is_time_aged(col):
            continue
        live_val = pd.to_numeric(pd.Series([live.get(col)]), errors="coerce").iloc[0]
        train_val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        if pd.isna(live_val) and pd.isna(train_val):
            continue
        if pd.isna(live_val) or pd.isna(train_val) or abs(live_val - train_val) > TOLERANCE:
            mismatches.append(
                f"{col}: live={live_val!r} train={train_val!r}"
            )
    return mismatches


def test_exact_replay_matches_training_rows():
    """as_of = event_date must reproduce the stored training row exactly."""
    spec, sample = _load_replay_sample()
    failures = []
    for _, row in sample.iterrows():
        as_of = row["event_date"].strftime("%Y-%m-%d")
        live = _replay_fight(row, spec, as_of)
        mismatches = _collect_mismatches(live, row, spec.feature_cols, skip_time_aged=False)
        if mismatches:
            failures.append(
                f"{row['fighter_a']} vs {row['fighter_b']} ({as_of}): " + "; ".join(mismatches[:8])
            )
    assert not failures, (
        f"{len(failures)}/{len(sample)} fights diverged from training rows:\n"
        + "\n".join(failures[:10])
    )


def test_prefight_replay_matches_training_rows():
    """as_of = event_date - 1 must reconstruct the same pre-fight state."""
    spec, sample = _load_replay_sample()
    failures = []
    for _, row in sample.iterrows():
        as_of = (row["event_date"] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        live = _replay_fight(row, spec, as_of)
        mismatches = _collect_mismatches(live, row, spec.feature_cols, skip_time_aged=True)
        if mismatches:
            failures.append(
                f"{row['fighter_a']} vs {row['fighter_b']} ({as_of}): " + "; ".join(mismatches[:8])
            )
    assert not failures, (
        f"{len(failures)}/{len(sample)} fights diverged from pre-fight training state:\n"
        + "\n".join(failures[:10])
    )
