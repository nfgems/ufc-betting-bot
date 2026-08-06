"""
Live/train feature parity replay harness (E1).

Replays historical fights through ``build_fight_features`` using the date-aware
processed path (no network) and diffs the result against the stored training
rows in ``features.csv``.

Two oracle modes:

- ``exact``    : as_of_date = event_date. The processed snapshot serves the
                 training row itself, so any divergence isolates logic that
                 ``build_fight_features`` computes on top of the lookup
                 (opp_strength, pre-UFC/amateur summaries, diffs, encodings).
- ``prefight`` : as_of_date = event_date - 1 day. The snapshot must reconstruct
                 the same pre-fight state without seeing the fight row, which
                 exercises the fallback roll-forward path. Stored pre-fight
                 values of the fight are the oracle; time-aged fields are
                 allowlisted.

Usage:
    python scripts/parity_replay.py --processed-dir data/processed/candidates/run \
        --spec full_live_contract_v6_durability --mode exact --start 2025-01-01 --limit 250
    python scripts/parity_replay.py --processed-dir data/processed/candidates/run \
        --spec full_live_contract_v6_durability --mode prefight --established-only \
        --start 2025-01-01 --limit 250

The command exits nonzero for a missing requested feature, a replay build
failure, or any forbidden mismatch. In prefight mode only numeric differences
in explicitly time-aged fields are allowlisted, and those differences remain
visible in both the console summary and JSON report.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DATA_DIR
from src.data.fighter_lookup import build_fight_features
from src.model.training_spec import resolve_named_training_spec

logger = logging.getLogger("parity_replay")

# Fields whose live value legitimately depends on the reference timestamp
# rather than fight history. In prefight mode these differ by the 1-day shift
# (and by snapshot aging) and are reported separately, not as failures.
TIME_AGED_FIELDS = {
    "days_since_last_fight", "layoff_log", "cage_rust",
    "age", "age_squared", "age_over_35", "age_under_25",
}

MISMATCH_VERDICTS = (
    "live_feature_missing",
    "live_nan_train_value",
    "live_value_train_nan",
    "value_diverge",
)


def _expand_time_aged(cols: list[str]) -> set[str]:
    out = set()
    for col in cols:
        for root in TIME_AGED_FIELDS:
            if col == root or col.endswith(root):
                out.add(col)
    return out


ODDS_PASSTHROUGH = [
    "a_implied_prob", "b_implied_prob", "diff_implied_prob",
    "a_ko_odds_prob", "a_sub_odds_prob", "a_dec_odds_prob",
    "b_ko_odds_prob", "b_sub_odds_prob", "b_dec_odds_prob",
]


def classify(live_val, train_val, tol: float) -> str:
    live_num = pd.to_numeric(pd.Series([live_val]), errors="coerce").iloc[0]
    train_num = pd.to_numeric(pd.Series([train_val]), errors="coerce").iloc[0]
    live_nan = pd.isna(live_num)
    train_nan = pd.isna(train_num)
    if live_nan and train_nan:
        return "match"
    if live_nan and not train_nan:
        return "live_nan_train_value"
    if not live_nan and train_nan:
        return "live_value_train_nan"
    if abs(float(live_num) - float(train_num)) <= tol:
        return "match"
    return "value_diverge"


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")


def _established_history_mask(df: pd.DataFrame) -> pd.Series:
    """Select fights where both fighters have earlier snapshot history.

    Historical prefight replay cannot reconstruct a debutant from the processed
    snapshot alone; live inference serves those fighters from contemporaneous
    profile sources. Filtering that source-limited case matches the established
    regression oracle while leaving every requested feature under comparison.
    """
    ordered = df.sort_values("event_date", kind="mergesort")
    seen: set[str] = set()
    eligible: dict[object, bool] = {}
    for index, row in ordered.iterrows():
        fighter_a = str(row.get("fighter_a") or "").strip()
        fighter_b = str(row.get("fighter_b") or "").strip()
        eligible[index] = bool(
            fighter_a and fighter_b and fighter_a in seen and fighter_b in seen
        )
        if fighter_a:
            seen.add(fighter_a)
        if fighter_b:
            seen.add(fighter_b)
    return pd.Series(eligible, dtype=bool).reindex(df.index, fill_value=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["exact", "prefight"], default="exact")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--spec", default="full_live_contract_v6_fullfit")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Processed snapshot directory containing features.csv and fights_cleaned.csv.",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--established-only",
        action="store_true",
        help=(
            "Require both fighters to have earlier processed history. This is "
            "the strict historical prefight oracle; debutants use live profile "
            "sources that are unavailable to a snapshot-only replay."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    spec = resolve_named_training_spec(args.spec)
    feature_cols = list(spec.feature_cols)
    processed_dir = args.processed_dir.resolve(strict=False)
    out_path = Path(args.out or f"logs/parity_replay_{args.mode}.json")

    features_path = processed_dir / "features.csv"
    if not features_path.exists():
        print(f"Processed feature snapshot not found: {features_path}", file=sys.stderr)
        return 1
    df = pd.read_csv(features_path, parse_dates=["event_date"])
    missing_feature_cols = [column for column in feature_cols if column not in df.columns]
    duplicate_feature_cols = sorted(
        {column for column in feature_cols if feature_cols.count(column) > 1}
    )
    if missing_feature_cols or duplicate_feature_cols:
        report = {
            "mode": args.mode,
            "n_fights": 0,
            "start": args.start,
            "spec": args.spec,
            "processed_dir": str(processed_dir),
            "features_path": str(features_path),
            "established_only": bool(args.established_only),
            "requested_feature_count": len(feature_cols),
            "compared_feature_count": 0,
            "missing_feature_cols": missing_feature_cols,
            "duplicate_spec_feature_cols": duplicate_feature_cols,
            "allowlisted_mismatch_count": 0,
            "forbidden_mismatch_count": len(missing_feature_cols)
            + len(duplicate_feature_cols),
            "replay_build_failures": [],
            "per_feature": [],
            "failures": [],
            "passed": False,
        }
        _write_report(out_path, report)
        if missing_feature_cols:
            print(
                f"Requested spec features missing from {features_path}: "
                f"{missing_feature_cols}",
                file=sys.stderr,
            )
        if duplicate_feature_cols:
            print(
                f"Requested spec repeats feature columns: {duplicate_feature_cols}",
                file=sys.stderr,
            )
        print(f"Detail written to {out_path}")
        return 1

    df = df.sort_values("event_date", kind="mergesort")
    established_eligible_count = None
    if args.established_only:
        established_mask = _established_history_mask(df)
        established_eligible_count = int(established_mask.sum())
        df = df[established_mask]
    df = df[df["event_date"] >= pd.Timestamp(args.start)]
    eligible_fight_count_before_sampling = int(len(df))
    if args.limit and len(df) > args.limit:
        rng = np.random.RandomState(args.seed)
        df = df.iloc[sorted(rng.choice(len(df), size=args.limit, replace=False))]
    logger.warning("Replaying %d fights (%s mode)", len(df), args.mode)

    compare_cols = feature_cols
    time_aged = _expand_time_aged(compare_cols)

    per_feature: dict[str, dict[str, int]] = {
        c: {
            "match": 0,
            "live_feature_missing": 0,
            "live_nan_train_value": 0,
            "live_value_train_nan": 0,
            "value_diverge": 0,
            "max_abs_diff": 0.0,
        }
        for c in compare_cols
    }
    n_fights = 0
    failures: list[dict] = []
    replay_build_failures: list[dict] = []

    for _, row in df.iterrows():
        event_date = row["event_date"]
        if args.mode == "exact":
            as_of = event_date.strftime("%Y-%m-%d")
        else:
            as_of = (event_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        odds_features = {}
        for col in ODDS_PASSTHROUGH:
            if col in row.index:
                odds_features[col] = row[col]

        try:
            live = build_fight_features(
                row["fighter_a"],
                row["fighter_b"],
                odds_features=odds_features,
                weight_class=row.get("weight_class") if isinstance(row.get("weight_class"), str) else None,
                is_title_bout=bool(row.get("is_title_bout", 0)),
                num_rounds=int(row["num_rounds_feat"]) if pd.notna(row.get("num_rounds_feat")) else 3,
                is_empty_arena=row.get("is_empty_arena"),
                as_of_date=as_of,
                training_spec=spec,
                processed_data_dir=processed_dir,
            )
            if not isinstance(live, dict):
                raise TypeError(
                    "build_fight_features returned a non-mapping result: "
                    f"{type(live).__name__}"
                )
        except Exception as exc:  # pragma: no cover - harness diagnostics
            logger.warning("build failed for %s vs %s @ %s: %s",
                           row["fighter_a"], row["fighter_b"], as_of, exc)
            replay_build_failures.append({
                "fight": f"{row['fighter_a']} vs {row['fighter_b']}",
                "event_date": str(event_date.date()),
                "as_of_date": as_of,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        n_fights += 1
        for col in compare_cols:
            verdict = (
                "live_feature_missing"
                if col not in live
                else classify(live[col], row[col], args.tol)
            )
            per_feature[col][verdict] += 1
            if verdict != "match":
                live_num = pd.to_numeric(
                    pd.Series([live.get(col)]), errors="coerce"
                ).iloc[0]
                train_num = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
                if pd.notna(live_num) and pd.notna(train_num):
                    diff = abs(float(live_num) - float(train_num))
                    per_feature[col]["max_abs_diff"] = max(per_feature[col]["max_abs_diff"], diff)
                allowlisted = bool(
                    args.mode == "prefight"
                    and col in time_aged
                    and verdict == "value_diverge"
                )
                if len(failures) < 4000:
                    failures.append({
                        "fight": f"{row['fighter_a']} vs {row['fighter_b']}",
                        "event_date": str(event_date.date()),
                        "feature": col,
                        "verdict": verdict,
                        "allowlisted_time_aging": allowlisted,
                        "live": None if pd.isna(live_num) else float(live_num),
                        "train": None if pd.isna(train_num) else float(train_num),
                    })

    # Summarize
    rows = []
    for col, counts in per_feature.items():
        mismatches = sum(int(counts[verdict]) for verdict in MISMATCH_VERDICTS)
        if mismatches:
            allowlisted_mismatches = (
                int(counts["value_diverge"])
                if args.mode == "prefight" and col in time_aged
                else 0
            )
            rows.append({
                "feature": col,
                "time_aged": col in time_aged,
                "mismatch_pct": 100.0 * mismatches / max(n_fights, 1),
                "mismatch_count": mismatches,
                "allowlisted_mismatch_count": allowlisted_mismatches,
                "forbidden_mismatch_count": mismatches - allowlisted_mismatches,
                **{k: v for k, v in counts.items()},
            })
    rows.sort(key=lambda r: (-r["mismatch_pct"], r["feature"]))

    allowlisted_mismatch_count = sum(
        int(row["allowlisted_mismatch_count"]) for row in rows
    )
    forbidden_mismatch_count = sum(
        int(row["forbidden_mismatch_count"]) for row in rows
    )
    structural = [row for row in rows if row["forbidden_mismatch_count"]]
    passed = bool(
        n_fights > 0
        and not replay_build_failures
        and forbidden_mismatch_count == 0
    )

    print(f"\n=== Parity replay ({args.mode}) — {n_fights} fights, "
          f"{len(compare_cols)} features compared ===")
    clean = len(compare_cols) - len(rows)
    print(f"Features with perfect parity: {clean}/{len(compare_cols)}")
    print(
        f"Features with mismatches: {len(rows)} "
        f"({len(structural)} with forbidden mismatches)"
    )
    print(
        f"Mismatch values: {forbidden_mismatch_count} forbidden, "
        f"{allowlisted_mismatch_count} prefight time-aged/allowlisted"
    )
    print(f"Replay build failures: {len(replay_build_failures)}\n")
    print(
        f"{'feature':<38}{'mism%':>7}{'missing':>9}{'nan|val':>9}"
        f"{'val|nan':>9}{'diverge':>9}{'forbid':>8}{'allowed':>9}  aged"
    )
    for r in rows[:80]:
        print(f"{r['feature']:<38}{r['mismatch_pct']:>6.1f}%"
              f"{r['live_feature_missing']:>9}"
              f"{r['live_nan_train_value']:>9}{r['live_value_train_nan']:>9}"
              f"{r['value_diverge']:>9}{r['forbidden_mismatch_count']:>8}"
              f"{r['allowlisted_mismatch_count']:>9}  {'Y' if r['time_aged'] else ''}")

    _write_report(
        out_path,
        {
            "mode": args.mode,
            "n_fights": n_fights,
            "start": args.start,
            "spec": args.spec,
            "processed_dir": str(processed_dir),
            "features_path": str(features_path),
            "established_only": bool(args.established_only),
            "established_eligible_count_all_dates": established_eligible_count,
            "eligible_fight_count_before_sampling": eligible_fight_count_before_sampling,
            "requested_feature_count": len(feature_cols),
            "compared_feature_count": len(compare_cols),
            "missing_feature_cols": [],
            "duplicate_spec_feature_cols": [],
            "allowlisted_time_aged_features": sorted(time_aged)
            if args.mode == "prefight"
            else [],
            "allowlisted_mismatch_count": allowlisted_mismatch_count,
            "forbidden_mismatch_count": forbidden_mismatch_count,
            "replay_build_failures": replay_build_failures,
            "per_feature": rows,
            "failures": failures[:1500],
            "passed": passed,
        },
    )
    print(f"\nDetail written to {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
