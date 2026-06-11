"""
E1 scoring experiment: production-model metrics on replayed live features.

Replays historical fights through the live feature builder (processed path),
predicts with the promoted production model (symmetric inference), and scores
log-loss/Brier against actual outcomes. Comparing runs isolates the cost of
each live/train parity defect:

  - --mode exact            : training-parity features (the reference)
  - --mode exact --variant proxy_sos : opp_strength replaced by the old
                              "opponent's roll_won" proxy (pre-fix live logic)
  - --mode prefight         : the snapshot-fallback path one day before the
                              fight (run on old code via `git stash` to score
                              the stale-fallback behavior)

Usage:
    python scripts/parity_scoring.py --mode exact --out logs/scoring_exact_new.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DATA_DIR
from src.data.fighter_lookup import build_fight_features
from src.model.predict import predict_fight
from src.model.train import load_model
from src.model.training_spec import resolve_named_training_spec

logger = logging.getLogger("parity_scoring")

ODDS_PASSTHROUGH = [
    "a_implied_prob", "b_implied_prob", "diff_implied_prob",
    "a_ko_odds_prob", "a_sub_odds_prob", "a_dec_odds_prob",
    "b_ko_odds_prob", "b_sub_odds_prob", "b_dec_odds_prob",
]


def apply_proxy_sos(features: dict) -> dict:
    """Reapply the pre-fix opp_strength proxy (opponent's roll_won)."""
    a_roll_won = features.get("a_roll_won", np.nan)
    b_roll_won = features.get("b_roll_won", np.nan)
    features = dict(features)
    features["a_opp_strength"] = b_roll_won if isinstance(b_roll_won, (int, float)) else np.nan
    features["b_opp_strength"] = a_roll_won if isinstance(a_roll_won, (int, float)) else np.nan
    a_os, b_os = features["a_opp_strength"], features["b_opp_strength"]
    if (isinstance(a_os, (int, float)) and isinstance(b_os, (int, float))
            and not (np.isnan(a_os) or np.isnan(b_os))):
        features["diff_opp_strength"] = a_os - b_os
    else:
        features.pop("diff_opp_strength", None)
    return features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["exact", "prefight"], default="exact")
    parser.add_argument("--variant", choices=["fixed", "proxy_sos"], default="fixed")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--spec", default="full_live_contract_v6_fullfit")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)

    spec = resolve_named_training_spec(args.spec)
    model_result = load_model("xgboost")

    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv", parse_dates=["event_date"])
    df = df.sort_values("event_date")

    seen: set[str] = set()
    has_history = []
    for row in df.itertuples(index=False):
        has_history.append(row.fighter_a in seen and row.fighter_b in seen)
        seen.add(row.fighter_a)
        seen.add(row.fighter_b)
    df = df[pd.Series(has_history, index=df.index)]
    df = df[(df["event_date"] >= pd.Timestamp(args.start)) & df["target"].notna()]
    if args.limit:
        df = df.iloc[: args.limit]

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        event_date = row["event_date"]
        as_of = (
            event_date if args.mode == "exact" else event_date - pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")
        odds_features = {c: row[c] for c in ODDS_PASSTHROUGH if c in row.index}
        try:
            features = build_fight_features(
                row["fighter_a"], row["fighter_b"],
                odds_features=odds_features,
                weight_class=row.get("weight_class") if isinstance(row.get("weight_class"), str) else None,
                is_title_bout=bool(row.get("is_title_bout", 0)),
                num_rounds=int(row["num_rounds_feat"]) if pd.notna(row.get("num_rounds_feat")) else 3,
                is_empty_arena=row.get("is_empty_arena"),
                as_of_date=as_of,
                training_spec=spec,
                processed_data_dir=PROCESSED_DATA_DIR,
            )
        except Exception as exc:
            logger.error("build failed %s vs %s: %s", row["fighter_a"], row["fighter_b"], exc)
            continue
        if args.variant == "proxy_sos":
            features = apply_proxy_sos(features)
        pred = predict_fight(features, model_result=model_result)
        records.append({
            "event_date": event_date,
            "fighter_a": row["fighter_a"],
            "fighter_b": row["fighter_b"],
            "target": float(row["target"]),
            "prob_a": pred["prob_a"],
            "market_prob_a": row.get("a_implied_prob", np.nan),
        })
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df)} fights scored", flush=True)

    out = pd.DataFrame(records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    eps = 1e-12
    p = out["prob_a"].clip(eps, 1 - eps)
    y = out["target"]
    log_loss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    brier = float(((p - y) ** 2).mean())
    acc = float(((p > 0.5).astype(float) == y).mean())
    print(f"mode={args.mode} variant={args.variant} n={len(out)} "
          f"log_loss={log_loss:.5f} brier={brier:.5f} acc={acc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
