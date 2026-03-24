"""Analyze whether remaining unmatched rows already have source candidates."""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.fix_tennis_odds_matching import names_match_flexible, normalize_text  # noqa: E402


def load_missing():
    df = pd.read_csv(ROOT / "data" / "processed" / "tennis" / "matches_with_odds.csv", low_memory=False)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    mask = df["b365_a"].isna() & df["ps_a"].isna() & df["max_a"].isna() & df["avg_a"].isna()
    return df[mask].copy()


def find_candidates(missing_df, source_df, label):
    print(f"\n=== {label} ===")
    found = 0
    samples = []
    for _, row in missing_df.iterrows():
        candidates = source_df[
            (source_df["tour"] == row["tour"]) &
            ((source_df["Date"] - row["event_date"]).abs().dt.days <= 14)
        ]
        for _, cand in candidates.iterrows():
            forward = (
                names_match_flexible(row["winner"], cand["Winner"]) and
                names_match_flexible(row["loser_name"], cand["Loser"])
            )
            reverse = (
                names_match_flexible(row["winner"], cand["Loser"]) and
                names_match_flexible(row["loser_name"], cand["Winner"])
            )
            if forward or reverse:
                found += 1
                if len(samples) < 10:
                    samples.append({
                        "event_date": str(row["event_date"].date()),
                        "tourney_name": row["tourney_name"],
                        "winner": row["winner"],
                        "loser": row["loser_name"],
                        "src_date": str(cand["Date"].date()),
                        "src_winner": cand["Winner"],
                        "src_loser": cand["Loser"],
                        "src_tournament": cand.get("tournament", ""),
                    })
                break
    print(f"rows={len(missing_df)} candidate_matches={found}")
    for sample in samples:
        print(sample)


def match_tourney_to_be(tourney_name):
    norm = normalize_text(tourney_name)
    norm = re.sub(r"\b125\b", "", norm).strip()
    num_match = re.search(r"#(\d+)", str(tourney_name))
    suffix = f"-{num_match.group(1)}" if num_match else ""
    base = re.sub(r"\d+", "", norm).strip()
    base = base.replace(" ", "-") if " " in base else base
    return base + suffix if suffix else base


def analyze_wta125_failures(missing_df, source_df):
    print("\n=== WTA125 failure analysis ===")
    source_df = source_df.copy()
    source_df["tournament_norm"] = source_df["tournament"].astype(str).map(normalize_text)

    buckets = {
        "name_match_within_14d": 0,
        "name_match_within_7d": 0,
        "blocked_by_tournament_filter": 0,
        "blocked_by_7d_window_only": 0,
        "no_name_match_within_14d": 0,
    }
    samples = []

    for _, row in missing_df.iterrows():
        candidates_14 = source_df[
            (source_df["tour"] == row["tour"]) &
            ((source_df["Date"] - row["event_date"]).abs().dt.days <= 14)
        ]
        candidates_7 = candidates_14[(candidates_14["Date"] - row["event_date"]).abs().dt.days <= 7]

        name_match_14 = None
        for _, cand in candidates_14.iterrows():
            forward = (
                names_match_flexible(row["winner"], cand["Winner"]) and
                names_match_flexible(row["loser_name"], cand["Loser"])
            )
            reverse = (
                names_match_flexible(row["winner"], cand["Loser"]) and
                names_match_flexible(row["loser_name"], cand["Winner"])
            )
            if forward or reverse:
                name_match_14 = cand
                break

        if name_match_14 is None:
            buckets["no_name_match_within_14d"] += 1
            continue

        buckets["name_match_within_14d"] += 1

        name_match_7 = None
        for _, cand in candidates_7.iterrows():
            forward = (
                names_match_flexible(row["winner"], cand["Winner"]) and
                names_match_flexible(row["loser_name"], cand["Loser"])
            )
            reverse = (
                names_match_flexible(row["winner"], cand["Loser"]) and
                names_match_flexible(row["loser_name"], cand["Winner"])
            )
            if forward or reverse:
                name_match_7 = cand
                break

        if name_match_7 is not None:
            buckets["name_match_within_7d"] += 1

        mapped = match_tourney_to_be(str(row["tourney_name"]))
        strict = candidates_7[candidates_7["tournament_norm"] == normalize_text(mapped)]
        if strict.empty:
            base = mapped.split("-")[0]
            strict = candidates_7[candidates_7["tournament_norm"].str.contains(base, na=False)]

        strict_name_match = None
        for _, cand in strict.iterrows():
            forward = (
                names_match_flexible(row["winner"], cand["Winner"]) and
                names_match_flexible(row["loser_name"], cand["Loser"])
            )
            reverse = (
                names_match_flexible(row["winner"], cand["Loser"]) and
                names_match_flexible(row["loser_name"], cand["Winner"])
            )
            if forward or reverse:
                strict_name_match = cand
                break

        if name_match_7 is None:
            buckets["blocked_by_7d_window_only"] += 1
        elif strict_name_match is None:
            buckets["blocked_by_tournament_filter"] += 1
            if len(samples) < 12:
                samples.append({
                    "event_date": str(row["event_date"].date()),
                    "tourney_name": row["tourney_name"],
                    "winner": row["winner"],
                    "loser": row["loser_name"],
                    "mapped_tournament": mapped,
                    "src_date": str(name_match_7["Date"].date()),
                    "src_tournament": name_match_7["tournament"],
                    "src_winner": name_match_7["Winner"],
                    "src_loser": name_match_7["Loser"],
                    "strict_candidates": strict["tournament"].astype(str).head(5).tolist(),
                })

    print(buckets)
    for sample in samples:
        print(sample)


def main():
    missing = load_missing()

    oddsportal = pd.read_csv(ROOT / "data" / "raw" / "tennis" / "oddsportal_team_events_odds.csv", low_memory=False)
    oddsportal["Date"] = pd.to_datetime(oddsportal["Date"], errors="coerce")

    wta125 = pd.read_csv(ROOT / "data" / "raw" / "tennis" / "betexplorer_wta125_odds.csv", low_memory=False)
    wta125["Date"] = pd.to_datetime(wta125["Date"], errors="coerce")
    if "tour" not in wta125.columns:
        wta125["tour"] = "wta"

    oddsportal_wta125_path = ROOT / "data" / "raw" / "tennis" / "oddsportal_wta125_odds.csv"
    oddsportal_wta125 = None
    if oddsportal_wta125_path.exists():
        oddsportal_wta125 = pd.read_csv(oddsportal_wta125_path, low_memory=False)
        oddsportal_wta125["Date"] = pd.to_datetime(oddsportal_wta125["Date"], errors="coerce")
        if "tour" not in oddsportal_wta125.columns:
            oddsportal_wta125["tour"] = "wta"

    for tourney in ["Atp Cup", "United Cup", "BJK Cup Qualifiers", "BJK Cup Playoffs", "BJK Cup Finals"]:
        subset = missing[missing["tourney_name"] == tourney].copy()
        find_candidates(subset, oddsportal, f"OddsPortal -> {tourney}")

    wta125_2025 = missing[(missing["tourney_name"].str.contains("125", na=False)) & (missing["event_date"].dt.year == 2025)].copy()
    find_candidates(wta125_2025, wta125, "BetExplorer WTA125 -> 2025 missing")
    analyze_wta125_failures(wta125_2025, wta125)

    if oddsportal_wta125 is not None:
        for tourney in ["Oeiras 125 Indoor #2", "Mumbai 125", "Newport 125", "Antalya 125 #3"]:
            subset = missing[missing["tourney_name"] == tourney].copy()
            find_candidates(subset, oddsportal_wta125, f"OddsPortal WTA125 -> {tourney}")


if __name__ == "__main__":
    main()
