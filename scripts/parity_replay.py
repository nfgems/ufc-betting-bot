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
    python scripts/parity_replay.py --mode exact --start 2025-01-01 --limit 250
    python scripts/parity_replay.py --mode prefight --start 2025-01-01 --limit 250
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["exact", "prefight"], default="exact")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--spec", default="full_live_contract_v6_fullfit")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    spec = resolve_named_training_spec(args.spec)
    feature_cols = list(spec.feature_cols)

    features_path = PROCESSED_DATA_DIR / "features.csv"
    df = pd.read_csv(features_path, parse_dates=["event_date"])
    df = df[df["event_date"] >= pd.Timestamp(args.start)].sort_values("event_date")
    if args.limit and len(df) > args.limit:
        rng = np.random.RandomState(args.seed)
        df = df.iloc[sorted(rng.choice(len(df), size=args.limit, replace=False))]
    logger.warning("Replaying %d fights (%s mode)", len(df), args.mode)

    compare_cols = [c for c in feature_cols if c in df.columns]
    time_aged = _expand_time_aged(compare_cols)

    per_feature: dict[str, dict[str, int]] = {
        c: {"match": 0, "live_nan_train_value": 0, "live_value_train_nan": 0,
            "value_diverge": 0, "max_abs_diff": 0.0}
        for c in compare_cols
    }
    n_fights = 0
    failures: list[dict] = []

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
                processed_data_dir=PROCESSED_DATA_DIR,
            )
        except Exception as exc:  # pragma: no cover - harness diagnostics
            logger.warning("build failed for %s vs %s @ %s: %s",
                           row["fighter_a"], row["fighter_b"], as_of, exc)
            continue

        n_fights += 1
        for col in compare_cols:
            verdict = classify(live.get(col), row[col], args.tol)
            per_feature[col][verdict] += 1
            if verdict != "match":
                live_num = pd.to_numeric(pd.Series([live.get(col)]), errors="coerce").iloc[0]
                train_num = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
                if pd.notna(live_num) and pd.notna(train_num):
                    diff = abs(float(live_num) - float(train_num))
                    per_feature[col]["max_abs_diff"] = max(per_feature[col]["max_abs_diff"], diff)
                if len(failures) < 4000:
                    failures.append({
                        "fight": f"{row['fighter_a']} vs {row['fighter_b']}",
                        "event_date": str(event_date.date()),
                        "feature": col,
                        "verdict": verdict,
                        "live": None if pd.isna(live_num) else float(live_num),
                        "train": None if pd.isna(train_num) else float(train_num),
                    })

    # Summarize
    rows = []
    for col, counts in per_feature.items():
        mismatches = counts["live_nan_train_value"] + counts["live_value_train_nan"] + counts["value_diverge"]
        if mismatches:
            rows.append({
                "feature": col,
                "time_aged": col in time_aged,
                "mismatch_pct": 100.0 * mismatches / max(n_fights, 1),
                **{k: v for k, v in counts.items()},
            })
    rows.sort(key=lambda r: (-r["mismatch_pct"], r["feature"]))

    print(f"\n=== Parity replay ({args.mode}) — {n_fights} fights, "
          f"{len(compare_cols)} features compared ===")
    clean = len(compare_cols) - len(rows)
    print(f"Features with perfect parity: {clean}/{len(compare_cols)}")
    structural = [r for r in rows if not r["time_aged"]]
    print(f"Features with mismatches: {len(rows)} ({len(structural)} structural, "
          f"{len(rows) - len(structural)} time-aged/allowlisted)\n")
    print(f"{'feature':<38}{'mism%':>7}{'nan|val':>9}{'val|nan':>9}{'diverge':>9}{'maxdiff':>10}  aged")
    for r in rows[:80]:
        print(f"{r['feature']:<38}{r['mismatch_pct']:>6.1f}%"
              f"{r['live_nan_train_value']:>9}{r['live_value_train_nan']:>9}"
              f"{r['value_diverge']:>9}{r['max_abs_diff']:>10.4f}  {'Y' if r['time_aged'] else ''}")

    out_path = args.out or f"logs/parity_replay_{args.mode}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({
            "mode": args.mode, "n_fights": n_fights, "start": args.start,
            "spec": args.spec, "per_feature": rows, "failures": failures[:1500],
        }, fh, indent=1, default=str)
    print(f"\nDetail written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
