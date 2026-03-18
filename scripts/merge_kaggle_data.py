"""
Merge Kaggle odds (moneyline) and rankings into the backfilled ufc-master.csv.

Uses fuzzy name matching to handle spelling differences between UFCStats
and the Kaggle odds source (BestFightOdds).
"""

import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR


def norm(name):
    if pd.isna(name):
        return ""
    return str(name).strip().lower().replace(".", "").replace("'", "").replace("-", " ")


def last_name(name):
    """Extract last name for fallback matching."""
    parts = norm(name).split()
    return parts[-1] if parts else ""


def fuzzy_score(a, b):
    return SequenceMatcher(None, a, b).ratio()


def decimal_to_american(dec):
    if pd.isna(dec) or dec <= 1.0:
        return np.nan
    if dec >= 2.0:
        return round((dec - 1) * 100, 1)
    else:
        return round(-100 / (dec - 1), 1)


def build_odds_lookup(odds_path):
    """Build a lookup of median closing odds per fight."""
    odds_raw = pd.read_csv(odds_path, low_memory=False)
    odds_raw["event_date"] = pd.to_datetime(odds_raw["event_date"], errors="coerce")
    odds_raw["adding_date"] = pd.to_datetime(odds_raw["adding_date"], errors="coerce")

    gap = odds_raw[odds_raw["event_date"] >= "2024-12-15"].copy()
    gap = gap.dropna(subset=["adding_date"])
    gap["f1_norm"] = gap["fighter_1"].apply(norm)
    gap["f2_norm"] = gap["fighter_2"].apply(norm)

    # Latest snapshot per source, then median across sources
    idx_latest = gap.groupby(["event_date", "f1_norm", "f2_norm", "source"])["adding_date"].idxmax()
    closing = gap.loc[idx_latest]

    fight_odds = (
        closing.groupby(["event_date", "f1_norm", "f2_norm"])
        .agg(odds_1_median=("odds_1", "median"), odds_2_median=("odds_2", "median"))
        .reset_index()
    )

    fight_odds["american_1"] = fight_odds["odds_1_median"].apply(decimal_to_american)
    fight_odds["american_2"] = fight_odds["odds_2_median"].apply(decimal_to_american)

    # Build two lookups:
    # 1. Exact: (date, frozenset(names)) -> {name: odds}
    # 2. By date: date -> list of (f1_norm, f2_norm, odds_dict)
    exact = {}
    by_date = {}

    for _, row in fight_odds.iterrows():
        d = row["event_date"].date()
        f1 = row["f1_norm"]
        f2 = row["f2_norm"]
        odds_dict = {f1: row["american_1"], f2: row["american_2"]}

        key = (d, frozenset([f1, f2]))
        exact[key] = odds_dict

        by_date.setdefault(d, []).append((f1, f2, odds_dict))

    return exact, by_date


def fuzzy_match_fight(red_norm, blue_norm, date_fights, threshold=0.75):
    """Fuzzy match a fight against available odds fights for a date."""
    best_score = 0
    best_match = None

    for f1, f2, odds_dict in date_fights:
        # Try both orientations
        for fa, fb in [(f1, f2), (f2, f1)]:
            score_r = fuzzy_score(red_norm, fa)
            score_b = fuzzy_score(blue_norm, fb)
            combined = (score_r + score_b) / 2
            if combined > best_score:
                best_score = combined
                # Map odds to red/blue
                best_match = {
                    "red_odds": odds_dict.get(fa, np.nan),
                    "blue_odds": odds_dict.get(fb, np.nan),
                    "score": combined,
                }

    if best_match and best_match["score"] >= threshold:
        return best_match
    return None


def merge_odds(master, needs_mask):
    """Merge Kaggle moneyline odds using exact + fuzzy matching."""
    odds_path = RAW_DATA_DIR / "kaggle_odds" / "UFC_betting_odds.csv"
    exact, by_date = build_odds_lookup(odds_path)

    matched_exact = 0
    matched_fuzzy = 0
    unmatched = 0

    for idx in master.index[needs_mask]:
        row = master.loc[idx]
        d = row["Date"].date() if pd.notna(row["Date"]) else None
        if d is None:
            continue

        rn = norm(row["RedFighter"])
        bn = norm(row["BlueFighter"])

        # Try exact match first
        key = (d, frozenset([rn, bn]))
        if key in exact:
            o = exact[key]
            master.loc[idx, "RedOdds"] = o.get(rn, np.nan)
            master.loc[idx, "BlueOdds"] = o.get(bn, np.nan)
            matched_exact += 1
            continue

        # Try fuzzy match
        if d in by_date:
            match = fuzzy_match_fight(rn, bn, by_date[d])
            if match:
                master.loc[idx, "RedOdds"] = match["red_odds"]
                master.loc[idx, "BlueOdds"] = match["blue_odds"]
                matched_fuzzy += 1
                continue

        unmatched += 1

    print(f"  Exact matches: {matched_exact}")
    print(f"  Fuzzy matches: {matched_fuzzy}")
    print(f"  Unmatched: {unmatched}")
    return matched_exact + matched_fuzzy


def merge_rankings(master, needs_mask):
    """Merge Kaggle rankings with fuzzy name matching."""
    rankings_path = RAW_DATA_DIR / "kaggle_rankings" / "rankings_history.csv"
    rankings_raw = pd.read_csv(rankings_path)
    rankings_raw["date"] = pd.to_datetime(rankings_raw["date"], errors="coerce")
    rankings_raw["fighter_norm"] = rankings_raw["fighter"].apply(norm)

    rankings_dates = sorted(rankings_raw["date"].dropna().unique())

    rank_col_map = {
        "Flyweight": ("RFlyweightRank", "BFlyweightRank"),
        "Bantamweight": ("RBantamweightRank", "BBantamweightRank"),
        "Featherweight": ("RFeatherweightRank", "BFeatherweightRank"),
        "Lightweight": ("RLightweightRank", "BLightweightRank"),
        "Welterweight": ("RWelterweightRank", "BWelterweightRank"),
        "Middleweight": ("RMiddleweightRank", "BMiddleweightRank"),
        "Light Heavyweight": ("RLightHeavyweightRank", "BLightHeavyweightRank"),
        "Heavyweight": ("RHeavyweightRank", "BHeavyweightRank"),
        "Women's Strawweight": ("RWStrawweightRank", "BWStrawweightRank"),
        "Women's Flyweight": ("RWFlyweightRank", "BWFlyweightRank"),
        "Women's Bantamweight": ("RWBantamweightRank", "BWBantamweightRank"),
        "Women's Featherweight": ("RWFeatherweightRank", "BWFeatherweightRank"),
    }

    def get_rankings_at(fight_date):
        fd = pd.Timestamp(fight_date)
        candidates = [d for d in rankings_dates if d <= fd]
        if not candidates:
            return {}, set()
        snap = rankings_raw[rankings_raw["date"] == candidates[-1]]
        lookup = {}
        all_names = set()
        for _, r in snap.iterrows():
            lookup[(r["fighter_norm"], r["weightclass"])] = r["rank"]
            all_names.add(r["fighter_norm"])
        return lookup, all_names

    cache = {}
    matched = 0

    for idx in master.index[needs_mask]:
        row = master.loc[idx]
        d = row["Date"]
        if pd.isna(d):
            continue

        d_key = d.date()
        if d_key not in cache:
            cache[d_key] = get_rankings_at(d)
        rlookup, all_names = cache[d_key]

        rn = norm(row["RedFighter"])
        bn = norm(row["BlueFighter"])
        wc = row["WeightClass"]

        # Try fuzzy name matching for rankings
        def find_fighter(fn):
            if any(k[0] == fn for k in rlookup):
                return fn
            # Fuzzy match
            best = None
            best_score = 0
            for name in all_names:
                s = fuzzy_score(fn, name)
                if s > best_score:
                    best_score = s
                    best = name
            return best if best_score >= 0.75 else None

        rn_matched = find_fighter(rn)
        bn_matched = find_fighter(bn)

        fight_matched = False

        if wc in rank_col_map:
            rcol, bcol = rank_col_map[wc]
            if rn_matched:
                r_rank = rlookup.get((rn_matched, wc))
                if r_rank is not None:
                    master.loc[idx, rcol] = r_rank
                    master.loc[idx, "RMatchWCRank"] = r_rank
                    fight_matched = True
            if bn_matched:
                b_rank = rlookup.get((bn_matched, wc))
                if b_rank is not None:
                    master.loc[idx, bcol] = b_rank
                    master.loc[idx, "BMatchWCRank"] = b_rank
                    fight_matched = True

        for pfp_wc in ["Pound-for-Pound", "Men's Pound-for-Pound", "Women's Pound-for-Pound"]:
            if rn_matched:
                r_pfp = rlookup.get((rn_matched, pfp_wc))
                if r_pfp is not None:
                    master.loc[idx, "RPFPRank"] = r_pfp
                    fight_matched = True
            if bn_matched:
                b_pfp = rlookup.get((bn_matched, pfp_wc))
                if b_pfp is not None:
                    master.loc[idx, "BPFPRank"] = b_pfp
                    fight_matched = True

        if fight_matched:
            matched += 1

    return matched


def main():
    master_path = RAW_DATA_DIR / "ufc-master.csv"

    # Re-read original to start fresh (undo partial merge)
    master = pd.read_csv(master_path, low_memory=False)
    master["Date"] = pd.to_datetime(master["Date"], errors="coerce")

    # Clear any previous partial merge for new rows
    needs = master["Date"] >= "2025-01-01"
    total_new = needs.sum()
    for col in ["RedOdds", "BlueOdds"]:
        master.loc[needs, col] = np.nan
    rank_cols = [c for c in master.columns if "Rank" in c]
    for col in rank_cols:
        master.loc[needs, col] = np.nan

    print(f"New rows needing enrichment: {total_new}")

    print("\n--- Merging moneyline odds (exact + fuzzy) ---")
    n_odds = merge_odds(master, needs)
    print(f"Total matched odds: {n_odds}/{total_new}")

    print("\n--- Merging rankings (exact + fuzzy) ---")
    n_ranks = merge_rankings(master, needs)
    print(f"Total matched rankings: {n_ranks}/{total_new}")

    new = master[needs]
    print(f"\nPost-merge verification:")
    print(f"  RedOdds non-null: {new['RedOdds'].notna().sum()}/{total_new}")
    print(f"  BlueOdds non-null: {new['BlueOdds'].notna().sum()}/{total_new}")
    print(f"  RMatchWCRank non-null: {new['RMatchWCRank'].notna().sum()}/{total_new}")
    print(f"  BMatchWCRank non-null: {new['BMatchWCRank'].notna().sum()}/{total_new}")

    # Show coverage by month
    print("\nOdds coverage by month:")
    new_copy = new.copy()
    new_copy["month"] = new_copy["Date"].dt.to_period("M")
    for month, group in new_copy.groupby("month"):
        has = group["RedOdds"].notna().sum()
        print(f"  {month}: {has}/{len(group)}")

    master.to_csv(master_path, index=False)
    print(f"\nSaved to {master_path} ({len(master)} rows)")


if __name__ == "__main__":
    main()
