"""Fix post-2014 moneyline and method odds gaps via name matching.

Reads source datasets, finds fights by date + name variants, extracts odds,
and appends to the respective output CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RAW = REPO_ROOT / "data" / "raw"
HIST_ODDS = RAW / "historical_odds"
METHOD_ODDS = RAW / "method_odds"

# ── Name alias map: UFCStats name → source dataset name(s) ──
NAME_ALIASES = {
    "Tiago dos Santos e Silva": ["Tiago Trator", "Tiago Dos Santos"],
    "Tecia Pennington": ["Tecia Torres"],
    "Michelle Waterson-Gomez": ["Michelle Waterson"],
    "Katlyn Cerminara": ["Katlyn Chookagian"],
    "Joanne Wood": ["Joanne Calderwood", "Joanne Wood", "JoJo Wood"],
    "Wu Yanan": ["Yanan Wu", "Wu Yanan"],
    "Mizuki": ["Mizuki Inoue", "Mizuki"],
    "Alatengheili": ["Heili Alateng", "Alatengheili"],
    "Rong Zhu": ["Rongzhu", "Rong Zhu"],
    "Gustavo Lopez": ["Gustavo Lopez"],
    "Mike de la Torre": ["Mike de la Torre", "Mike De La Torre"],
    "Clay Collard": ["Clay Collard"],
    "Shane Burgos": ["Shane Burgos"],
    "Brandon Jenkins": ["Brandon Jenkins"],
    "Cristiane Justino": ["Cris Cyborg", "Cristiane Justino", "Cris Justino"],
    "Nina Nunes": ["Nina Ansaroff", "Nina Nunes"],
    "Rafael Cavalcante": ["Rafael Cavalcante", "Feijao Cavalcante"],
    "Norifumi Yamamoto": ["Norifumi Yamamoto", "Kid Yamamoto"],
    "Mirko Filipovic": ["Mirko Cro Cop", "Mirko Filipovic"],
    "Rogerio Nogueira": ["Rogerio Nogueira", "Little Nog", "Antonio Rogerio Nogueira"],
    "Antonio Carlos Junior": ["Antonio Carlos Junior", "Antonio Carlos Jr", "Antonio Carlos Jr."],
    "Jacare Souza": ["Jacare Souza", "Ronaldo Souza"],
    "Marcos Rogerio de Lima": ["Marcos Rogerio de Lima", "Marcos Rogerio De Lima"],
    "Lando Vannata": ["Lando Vannata"],
    "Song Yadong": ["Song Yadong", "Yadong Song"],
    "Khalil Rountree Jr.": ["Khalil Rountree", "Khalil Rountree Jr", "Khalil Rountree Jr."],
    "Li Jingliang": ["Li Jingliang", "Jingliang Li"],
    "Song Kenan": ["Song Kenan", "Kenan Song"],
    "Da Woon Jung": ["Da Woon Jung", "Da-Woon Jung", "Dawoon Jung"],
    "Zubaira Tukhugov": ["Zubaira Tukhugov"],
    "Joselyne Edwards": ["Joselyne Edwards"],
    "Mayra Bueno Silva": ["Mayra Bueno Silva"],
    "Montana De La Rosa": ["Montana De La Rosa", "Montana de la Rosa"],
    "Alan Patrick": ["Alan Patrick"],
    "King Green": ["King Green", "Bobby Green"],
}


def _normalize(name: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return " ".join(str(name).strip().lower().split())


def _last_name(name: str) -> str:
    parts = _normalize(name).split()
    return parts[-1] if parts else ""


def _get_aliases(name: str) -> list[str]:
    """Return all known variants of a name, including the original."""
    aliases = [name]
    for key, vals in NAME_ALIASES.items():
        if _normalize(name) == _normalize(key):
            aliases.extend(vals)
        for v in vals:
            if _normalize(name) == _normalize(v):
                aliases.append(key)
                aliases.extend(vals)
    # Deduplicate preserving order
    seen = set()
    result = []
    for a in aliases:
        n = _normalize(a)
        if n not in seen:
            seen.add(n)
            result.append(a)
    return result


def _american_to_prob(odds) -> float | None:
    """Convert American odds to implied probability."""
    try:
        odds = float(odds)
    except (ValueError, TypeError):
        return None
    if odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def _find_in_master_by_date(master: pd.DataFrame, date: str, fighter_a: str, fighter_b: str):
    """Find a fight in ufc-master.csv by date and name variants."""
    date_mask = master["Date"] == date
    date_rows = master[date_mask]
    if date_rows.empty:
        return None

    a_aliases = [_normalize(a) for a in _get_aliases(fighter_a)]
    b_aliases = [_normalize(b) for b in _get_aliases(fighter_b)]

    for _, row in date_rows.iterrows():
        red = _normalize(str(row["RedFighter"]))
        blue = _normalize(str(row["BlueFighter"]))

        # Check both orientations
        if (red in a_aliases or _last_name(red) in [_last_name(a) for a in a_aliases]) and \
           (blue in b_aliases or _last_name(blue) in [_last_name(b) for b in b_aliases]):
            return row, False  # Not reversed
        if (blue in a_aliases or _last_name(blue) in [_last_name(a) for a in a_aliases]) and \
           (red in b_aliases or _last_name(red) in [_last_name(b) for b in b_aliases]):
            return row, True  # Reversed

    # Fallback: try matching just last names more aggressively
    a_lasts = set(_last_name(a) for a in a_aliases)
    b_lasts = set(_last_name(b) for b in b_aliases)
    for _, row in date_rows.iterrows():
        red_last = _last_name(str(row["RedFighter"]))
        blue_last = _last_name(str(row["BlueFighter"]))
        if red_last in a_lasts and blue_last in b_lasts:
            return row, False
        if blue_last in a_lasts and red_last in b_lasts:
            return row, True

    return None


def _find_odds_in_sources(sources: list[pd.DataFrame], source_configs: list[dict],
                           date: str, fighter_a: str, fighter_b: str):
    """Search multiple source datasets for moneyline odds."""
    a_aliases = [_normalize(a) for a in _get_aliases(fighter_a)]
    b_aliases = [_normalize(b) for b in _get_aliases(fighter_b)]

    for src_df, cfg in zip(sources, source_configs):
        date_col = cfg["date_col"]
        fa_col = cfg["fa_col"]
        fb_col = cfg["fb_col"]
        odds_a_col = cfg["odds_a_col"]
        odds_b_col = cfg["odds_b_col"]
        is_decimal = cfg.get("is_decimal", True)

        date_mask = src_df[date_col] == date
        date_rows = src_df[date_mask]

        for _, row in date_rows.iterrows():
            f1 = _normalize(str(row[fa_col]))
            f2 = _normalize(str(row[fb_col]))

            reversed_match = False
            if (f1 in a_aliases and f2 in b_aliases):
                reversed_match = False
            elif (f2 in a_aliases and f1 in b_aliases):
                reversed_match = True
            elif (_last_name(f1) in [_last_name(a) for a in a_aliases] and
                  _last_name(f2) in [_last_name(b) for b in b_aliases]):
                reversed_match = False
            elif (_last_name(f2) in [_last_name(a) for a in a_aliases] and
                  _last_name(f1) in [_last_name(b) for b in b_aliases]):
                reversed_match = True
            else:
                continue

            try:
                odds_1 = float(row[odds_a_col])
                odds_2 = float(row[odds_b_col])
            except (ValueError, TypeError):
                continue

            if not is_decimal:
                # Convert American to decimal
                if odds_1 > 0:
                    odds_1 = 1 + odds_1 / 100
                else:
                    odds_1 = 1 + 100 / abs(odds_1)
                if odds_2 > 0:
                    odds_2 = 1 + odds_2 / 100
                else:
                    odds_2 = 1 + 100 / abs(odds_2)

            if reversed_match:
                odds_1, odds_2 = odds_2, odds_1

            # Convert to fair probabilities (remove vig)
            raw_a = 1.0 / odds_1
            raw_b = 1.0 / odds_2
            total = raw_a + raw_b
            fair_a = raw_a / total
            fair_b = raw_b / total

            return {
                "a_fair_prob": round(fair_a, 6),
                "b_fair_prob": round(fair_b, 6),
                "a_decimal_odds": round(odds_1, 6),
                "b_decimal_odds": round(odds_2, 6),
            }

    return None


def fix_moneyline_gaps():
    """Fix the 8 unique missing moneyline fights."""
    print("=" * 60)
    print("TASK 1: Fixing missing moneyline fights")
    print("=" * 60)

    missing = pd.read_csv(HIST_ODDS / "pre2022_odds_still_missing.csv")
    existing = pd.read_csv(HIST_ODDS / "historical_odds_pre2022_from_cleaned.csv")
    print(f"Currently {len(existing)} rows in pre2022 moneyline file")
    print(f"Missing fights: {len(missing)}")

    # Load source datasets
    master = pd.read_csv(RAW / "ufc-master.csv")
    master["Date"] = pd.to_datetime(master["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    jansen = pd.read_csv(RAW / "jansen88_ufc_data.csv")
    jansen.columns = jansen.columns.str.strip()
    for c in ["fighter1", "fighter2"]:
        if c in jansen.columns:
            jansen[c] = jansen[c].str.strip()
    if "date" in jansen.columns:
        jansen["date"] = pd.to_datetime(jansen["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    bfo = pd.read_csv(RAW / "bfo_iankotliar_odds.csv")
    bfo.columns = bfo.columns.str.strip()
    for c in ["fighter1", "fighter2"]:
        if c in bfo.columns:
            bfo[c] = bfo[c].str.strip()
    if "date" in bfo.columns:
        bfo["date"] = pd.to_datetime(bfo["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Figure out correct column names for each source
    # ufc-master: American odds
    sources = []
    configs = []

    sources.append(master)
    configs.append({
        "date_col": "Date",
        "fa_col": "RedFighter",
        "fb_col": "BlueFighter",
        "odds_a_col": "RedOdds",
        "odds_b_col": "BlueOdds",
        "is_decimal": False,
    })

    # Jansen: find odds columns
    jansen_odds_cols = [c for c in jansen.columns if "odds" in c.lower() or "line" in c.lower()]
    print(f"Jansen odds cols: {jansen_odds_cols}")
    if "fighter1_odds" in jansen.columns:
        sources.append(jansen)
        configs.append({
            "date_col": "date",
            "fa_col": "fighter1",
            "fb_col": "fighter2",
            "odds_a_col": "fighter1_odds",
            "odds_b_col": "fighter2_odds",
            "is_decimal": True,
        })

    # BFO
    bfo_odds_cols = [c for c in bfo.columns if "odds" in c.lower() or "line" in c.lower() or "avg" in c.lower()]
    print(f"BFO odds cols (first 10): {bfo_odds_cols[:10]}")
    # Probably has average odds or specific bookmaker columns
    if "Avg" in bfo.columns:
        sources.append(bfo)
        configs.append({
            "date_col": "date",
            "fa_col": "fighter1",
            "fb_col": "fighter2",
            "odds_a_col": "fighter1_Avg",  # guess
            "odds_b_col": "fighter2_Avg",
            "is_decimal": True,
        })

    new_rows = []
    still_missing = []

    for _, row in missing.iterrows():
        date = str(row["event_date"])
        fa = str(row["fighter_a"])
        fb = str(row["fighter_b"])

        result = _find_odds_in_sources(sources, configs, date, fa, fb)
        if result:
            new_rows.append({
                "event_date": date,
                "fighter_a": fa,
                "fighter_b": fb,
                "query_date": date,
                "offset_days": 1,
                "a_fair_prob": result["a_fair_prob"],
                "b_fair_prob": result["b_fair_prob"],
                "a_decimal_odds": result["a_decimal_odds"],
                "b_decimal_odds": result["b_decimal_odds"],
                "num_bookmakers": 1,
            })
            print(f"  FOUND: {date} {fa} vs {fb} -> a={result['a_fair_prob']:.3f} b={result['b_fair_prob']:.3f}")
        else:
            still_missing.append(f"{date}: {fa} vs {fb}")
            print(f"  MISS:  {date} {fa} vs {fb}")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["event_date", "fighter_a", "fighter_b"], keep="last")
        combined.to_csv(HIST_ODDS / "historical_odds_pre2022_from_cleaned.csv", index=False)
        print(f"\nAppended {len(new_rows)} rows. Total now: {len(combined)}")

    if still_missing:
        print(f"\nStill missing {len(still_missing)} fights:")
        for m in still_missing:
            print(f"  {m}")

    return len(new_rows), still_missing


def fix_method_odds_gaps():
    """Fix the 121 missing method odds fights."""
    print("\n" + "=" * 60)
    print("TASK 2: Fixing missing method odds fights")
    print("=" * 60)

    missing = pd.read_csv(METHOD_ODDS / "post2014_method_missing.csv")
    existing = pd.read_csv(METHOD_ODDS / "historical_method_odds_all.csv")
    print(f"Currently {len(existing)} rows in method odds file")
    print(f"Missing fights: {len(missing)}")

    # Load source
    master = pd.read_csv(RAW / "ufc-master.csv")
    master["Date"] = pd.to_datetime(master["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Also try PierceHampton for KO/Sub odds
    try:
        pierce = pd.read_csv(RAW / "github_PierceHampton.csv")
        pierce.columns = pierce.columns.str.strip()
        if "event_date" in pierce.columns:
            pierce["event_date"] = pd.to_datetime(pierce["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        elif "date" in pierce.columns:
            pierce["date"] = pd.to_datetime(pierce["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        print(f"PierceHampton loaded: {len(pierce)} rows")
        pierce_ko_cols = [c for c in pierce.columns if "ko" in c.lower() or "sub" in c.lower()]
        print(f"  KO/Sub cols: {pierce_ko_cols}")
    except Exception as e:
        print(f"  PierceHampton load failed: {e}")
        pierce = None

    new_rows = []
    still_missing_method = []

    for _, row in missing.iterrows():
        date = str(row["event_date"])
        fa = str(row["fighter_a"])
        fb = str(row["fighter_b"])

        result = _find_in_master_by_date(master, date, fa, fb)
        if result is not None:
            master_row, reversed_match = result

            # Extract method odds (American format)
            try:
                r_ko = float(master_row.get("RKOOdds", np.nan))
                b_ko = float(master_row.get("BKOOdds", np.nan))
                r_sub = float(master_row.get("RSubOdds", np.nan))
                b_sub = float(master_row.get("BSubOdds", np.nan))
                r_dec = float(master_row.get("RedDecOdds", np.nan))
                b_dec = float(master_row.get("BlueDecOdds", np.nan))
            except (ValueError, TypeError):
                still_missing_method.append(f"{date}: {fa} vs {fb} (odds parse error)")
                print(f"  PARSE ERR: {date} {fa} vs {fb}")
                continue

            # Check if any odds are actually present
            odds_vals = [r_ko, b_ko, r_sub, b_sub, r_dec, b_dec]
            if all(pd.isna(v) for v in odds_vals):
                still_missing_method.append(f"{date}: {fa} vs {fb} (all NaN in master)")
                print(f"  ALL NaN: {date} {fa} vs {fb}")
                continue

            # Convert American odds to implied probabilities
            a_ko_prob = _american_to_prob(r_ko if not reversed_match else b_ko)
            b_ko_prob = _american_to_prob(b_ko if not reversed_match else r_ko)
            a_sub_prob = _american_to_prob(r_sub if not reversed_match else b_sub)
            b_sub_prob = _american_to_prob(b_sub if not reversed_match else r_sub)
            a_dec_prob = _american_to_prob(r_dec if not reversed_match else b_dec)
            b_dec_prob = _american_to_prob(b_dec if not reversed_match else r_dec)

            new_rows.append({
                "event_date": date,
                "fighter_a": fa,
                "fighter_b": fb,
                "event_title": "",
                "source": "kaggle_mdabbert_bestfightodds",
                "source_url": "",
                "captured_at": "2026-03-18T00:00:00Z",
                "a_ko_odds_prob": round(a_ko_prob, 6) if a_ko_prob else "",
                "a_sub_odds_prob": round(a_sub_prob, 6) if a_sub_prob else "",
                "a_dec_odds_prob": round(a_dec_prob, 6) if a_dec_prob else "",
                "b_ko_odds_prob": round(b_ko_prob, 6) if b_ko_prob else "",
                "b_sub_odds_prob": round(b_sub_prob, 6) if b_sub_prob else "",
                "b_dec_odds_prob": round(b_dec_prob, 6) if b_dec_prob else "",
            })
            matched_name = f"{master_row['RedFighter']} vs {master_row['BlueFighter']}"
            print(f"  FOUND: {date} {fa} vs {fb} -> matched {matched_name}")
        else:
            still_missing_method.append(f"{date}: {fa} vs {fb}")
            print(f"  MISS:  {date} {fa} vs {fb}")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["event_date", "fighter_a", "fighter_b"], keep="last"
        )
        combined.to_csv(METHOD_ODDS / "historical_method_odds_all.csv", index=False)
        print(f"\nAppended {len(new_rows)} rows. Total now: {len(combined)}")

    if still_missing_method:
        print(f"\nStill missing {len(still_missing_method)} method odds fights:")
        for m in still_missing_method:
            print(f"  {m}")

    return len(new_rows), still_missing_method


if __name__ == "__main__":
    mono_found, mono_missing = fix_moneyline_gaps()
    method_found, method_missing = fix_method_odds_gaps()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Moneyline: found {mono_found}, still missing {len(mono_missing)}")
    print(f"Method odds: found {method_found}, still missing {len(method_missing)}")
