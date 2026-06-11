"""
E9: fitted logit-space stacking of model and market probabilities.

The production blend uses five hand-set constants and is anti-shrinkage (it
up-weights the model at extreme probabilities where tree overconfidence is
worst). This fits, walk-forward (train on prior folds only), a logistic
regression on [logit(p_model), logit(p_market), logit(p_noodds)] and compares
the resulting probability quality against the production heuristic blend on
the same out-of-fold fights.

Symmetry: each training fight is duplicated with the mirrored orientation
(1-p, target flipped), which forces the intercept toward 0 and makes the
stack A/B-symmetric by construction.

Usage:
    python scripts/e9_logit_stack.py --preds-cache logs/fold_predictions_baseline.pkl
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BLEND_WEIGHT
from src.strategy.value import compute_independent_blend_probs

logger = logging.getLogger("e9_logit_stack")
EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        total += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(total)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-cache", default="logs/fold_predictions_baseline.pkl")
    parser.add_argument("--min-train-folds", type=int, default=3)
    parser.add_argument("--out", default="logs/e9_logit_stack.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with Path(args.preds_cache).open("rb") as fh:
        fold_predictions = pickle.load(fh)

    frames = []
    for fold_num, frame in fold_predictions:
        f = frame.copy()
        f["fold"] = fold_num
        frames.append(f)
    data = pd.concat(frames, ignore_index=True)
    needed = ["prob_a", "a_market_prob", "no_odds_prob_a", "target", "fold"]
    missing_cols = [c for c in needed if c not in data.columns]
    if missing_cols:
        raise SystemExit(f"fold predictions missing columns: {missing_cols}")
    data = data.dropna(subset=["prob_a", "a_market_prob", "target"])
    has_noodds = data["no_odds_prob_a"].notna()
    logger.info("Fights with model+market: %d (no-odds present on %d)",
                len(data), int(has_noodds.sum()))

    folds = sorted(data["fold"].unique())
    rows = []
    oof_records = []

    for k_idx in range(args.min_train_folds, len(folds)):
        train_folds = folds[:k_idx]
        test_fold = folds[k_idx]
        train = data[data["fold"].isin(train_folds)]
        test = data[data["fold"] == test_fold]
        if train.empty or test.empty:
            continue

        # Mirror-duplicated training set forces orientation symmetry.
        def _design(frame: pd.DataFrame, flipped: bool) -> tuple[np.ndarray, np.ndarray]:
            p_model = frame["prob_a"].to_numpy()
            p_market = frame["a_market_prob"].to_numpy()
            p_noodds = frame["no_odds_prob_a"].to_numpy()
            y = frame["target"].to_numpy().astype(float)
            if flipped:
                p_model, p_market, p_noodds, y = 1 - p_model, 1 - p_market, 1 - p_noodds, 1 - y
            # no_odds enters as a deviation term only when observed; never imputed.
            noodds_logit = np.where(
                np.isnan(p_noodds), 0.0, _logit(np.nan_to_num(p_noodds, nan=0.5))
            )
            noodds_seen = (~np.isnan(p_noodds)).astype(float)
            X = np.column_stack([
                _logit(p_model), _logit(p_market), noodds_logit, noodds_seen,
            ])
            return X, y

        X1, y1 = _design(train, flipped=False)
        X2, y2 = _design(train, flipped=True)
        X = np.vstack([X1, X2])
        y = np.concatenate([y1, y2])

        stack = LogisticRegression(C=1e6, max_iter=1000)
        stack.fit(X, y)

        X_test, y_test = _design(test, flipped=False)
        p_stack = stack.predict_proba(X_test)[:, 1]

        # Production heuristic blend on the same fights.
        p_heur = np.full(len(test), np.nan)
        for i, (_, r) in enumerate(test.iterrows()):
            try:
                p_heur[i], _ = compute_independent_blend_probs(
                    float(r["prob_a"]), float(r["a_market_prob"]),
                    float(r["no_odds_prob_a"]) if pd.notna(r["no_odds_prob_a"]) else None,
                    float(1 - r["prob_a"]), float(1 - r["a_market_prob"]),
                    float(1 - r["no_odds_prob_a"]) if pd.notna(r["no_odds_prob_a"]) else None,
                    base_weight=BLEND_WEIGHT,
                )
            except Exception:
                continue
        valid = np.isfinite(p_heur)

        y_v = y_test[valid].astype(int)
        market_v = test["a_market_prob"].to_numpy()[valid]
        edge_stack = np.abs(p_stack[valid] - market_v)
        edge_heur = np.abs(p_heur[valid] - market_v)

        rows.append({
            "test_fold": test_fold,
            "n": int(valid.sum()),
            "logloss_stack": log_loss(y_v, np.clip(p_stack[valid], EPS, 1 - EPS)),
            "logloss_heuristic": log_loss(y_v, np.clip(p_heur[valid], EPS, 1 - EPS)),
            "logloss_market": log_loss(y_v, np.clip(market_v, EPS, 1 - EPS)),
            "brier_stack": brier_score_loss(y_v, p_stack[valid]),
            "brier_heuristic": brier_score_loss(y_v, p_heur[valid]),
            "ece_stack": _ece(y_v, p_stack[valid]),
            "ece_heuristic": _ece(y_v, p_heur[valid]),
            "bets_proxy_stack": int((edge_stack >= 0.025).sum()),
            "bets_proxy_heuristic": int((edge_heur >= 0.025).sum()),
            "coef_model": stack.coef_[0][0],
            "coef_market": stack.coef_[0][1],
            "coef_noodds": stack.coef_[0][2],
            "intercept": stack.intercept_[0],
        })
        oof_records.append(pd.DataFrame({
            "fold": test_fold, "target": y_v,
            "p_stack": p_stack[valid], "p_heuristic": p_heur[valid],
            "p_market": market_v,
        }))

    result = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)

    pooled = pd.concat(oof_records, ignore_index=True)
    print("\n=== E9 logit stack vs heuristic blend (out-of-fold) ===")
    print(result.to_string(index=False))
    y = pooled["target"].to_numpy().astype(int)
    print(f"\nPooled OOF n={len(pooled)}")
    for name in ("p_stack", "p_heuristic", "p_market"):
        p = np.clip(pooled[name].to_numpy(), EPS, 1 - EPS)
        print(f"  {name:<12} logloss={log_loss(y, p):.5f} brier={brier_score_loss(y, p):.5f} "
              f"ece={_ece(y, p):.5f}")
    print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
