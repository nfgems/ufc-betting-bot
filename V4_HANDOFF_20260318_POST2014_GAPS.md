# V4 Handoff: Close All Post-2014 Data Gaps

## Mission

Get post-2014 pre-2022 data to 100% coverage for moneyline and method odds. Every gap remaining is a **name-matching failure**, not a missing source. The data exists — the fighter names in the training dataset don't match the fighter names in the source datasets. Fix the matching, then wire everything into the pipeline.

## Current State

### Post-2014 pre-2022: 3,622 training fights

| Family | Covered | Missing | Coverage | Root cause |
|---|---|---|---|---|
| Moneyline | 3,613 | **9** | 99.8% | Name mismatches (8 unique fights) |
| Method odds | 3,501 | **121** | 96.7% | Name mismatches (all 121 on dates WITH data) |
| Rankings | 3,596 | 26 | 99.3% | Snapshot availability (not fixable) |
| Rolling stats | 2,249 | 1,373 | 62.1% | Debut fighters — honest NaN, don't touch |

### What "name mismatch" means

The training dataset (`data/processed/fights_cleaned.csv`) uses names from UFCStats.com. The odds source datasets use names from BestFightOdds.com. These differ for:

1. **Married name changes:** `Tecia Pennington` (UFCStats) = `Tecia Torres` (BFO)
2. **Name transliterations:** `Rongzhu` vs `Rong Zhu`, `Wu Yanan` vs `Yanan Wu`
3. **Long compound names:** `Tiago dos Santos e Silva` = `Thiago Santos`
4. **Alternate spellings:** `Katlyn Cerminara` vs `Katlyn Chookagian`, `Michelle Waterson-Gomez` vs `Michelle Waterson`
5. **Chinese/Asian name order:** `Guangyou Ning` vs `Ning Guangyou`

## Task 1: Fix the 9 missing moneyline fights

These 8 unique fights (one is a duplicate row) have odds in the source datasets but the name matching failed:

```
2015-02-22: Tiago dos Santos e Silva vs Mike de la Torre
2015-09-05: Clay Collard vs Tiago dos Santos e Silva
2016-12-09: Tiago dos Santos e Silva vs Shane Burgos
2017-12-02: Tecia Pennington vs Michelle Waterson-Gomez
2019-06-08: Katlyn Cerminara vs Joanne Wood
2019-08-31: Wu Yanan vs Mizuki
2021-09-18: Gustavo Lopez vs Alatengheili
2021-09-18: Rong Zhu vs Brandon Jenkins
```

**Approach:** Manually look up each fighter's name in the source datasets and add explicit aliases. The data definitely exists — these are all UFC main/prelim card fights with full BFO odds coverage.

**Source datasets to search** (all in `data/raw/`):
- `ufc-master.csv` — `RedFighter`/`BlueFighter` columns, American odds
- `jansen88_ufc_data.csv` — `fighter1`/`fighter2` columns, decimal odds
- `bfo_iankotliar_odds.csv` — `fighter1`/`fighter2` columns, multi-bookmaker decimal odds
- `github_PierceHampton.csv` — `f_1_name`/`f_2_name` columns, decimal odds

**Output:** Add recovered rows to `data/raw/historical_odds/historical_odds_pre2022_from_cleaned.csv`

## Task 2: Fix the 121 missing method odds fights

**Critical fact:** ALL 121 missing fights are on dates where `ufc-master.csv` has method odds data for OTHER fights on the same card. The data exists — only the name matching failed.

Full list saved at: `data/raw/method_odds/post2014_method_missing.csv`

By year: 2015: 16, 2016: 28, 2017: 24, 2018: 8, 2019: 16, 2020: 12, 2021: 17

**Approach:**
1. For each missing fight, search `ufc-master.csv` on that date for a row with matching last names or partial name overlap
2. When a match is found, extract `RKOOdds`, `BKOOdds`, `RSubOdds`, `BSubOdds`, `RedDecOdds`, `BlueDecOdds`
3. Convert American odds to implied probability
4. Add to `data/raw/method_odds/historical_method_odds_all.csv`

**Common name aliases to build** (identified from the 121 gaps):
- `Tiago dos Santos e Silva` → `Thiago Santos` or truncated variants
- `Cristiane Justino` → `Cris Cyborg`
- `Joanne Wood` → `Joanne Calderwood` or `JoJo Wood`
- Any fighter with accent marks, hyphens, or compound surnames

**Also check** `github_PierceHampton.csv` for KO/Sub odds (has `f_1_ko_odds`, `f_1_sub_odds`, `f_2_ko_odds`, `f_2_sub_odds`).

## Task 3: Wire the new data into the pipeline

After fixing the name matches, the scraped data needs to be wired into `src/data/ufc_refresh.py`:

### Moneyline overlay
The `_historical_moneyline_overlay()` function reads from `data/raw/historical_odds/historical_odds.csv` (2022+ only). It needs to also read from `historical_odds_pre2022_from_cleaned.csv`:

```python
# In _historical_moneyline_overlay(), after loading historical_odds.csv:
pre2022_path = BACKFILL_DIR / "historical_odds_pre2022_from_cleaned.csv"
if pre2022_path.exists():
    pre2022_df = pd.read_csv(pre2022_path, parse_dates=["event_date"])
    hist_df = pd.concat([pre2022_df, hist_df]).drop_duplicates(
        subset=["event_date", "fighter_a", "fighter_b"], keep="last"
    )
```

### Method odds overlay
The `_historical_method_odds_overlay()` function reads from `data/raw/method_odds/historical_method_odds.csv` (12 rows). Point it at the combined file instead:

```python
# Change the path to:
METHOD_ODDS_HISTORY_PATH = RAW_DATA_DIR / "method_odds" / "historical_method_odds_all.csv"
```

### Rankings overlay
The `_historical_rankings_overlay()` function reads from `data/raw/kaggle_rankings/rankings_history.csv` (480 dates). Point it at the extended file:

```python
# Change the path to:
RANKINGS_CSV = RAW_DATA_DIR / "kaggle_rankings" / "rankings_history_extended.csv"
```

## Task 4: Rebuild and evaluate

After wiring, rebuild the 138-feature dataset and check:
1. Post-2014 NaN rate should drop from ~5.6% to <2% (only rolling stats for debut fighters)
2. Moneyline should be ~0% NaN post-2014
3. Method odds should be ~0% NaN post-2014
4. Rankings should be available for ~99% of post-2014 fights (but still NaN for unranked fighters)

Then retrain the 138 model and compare against the current best candidates.

## Task 5: Wayback Machine for remaining stubborn gaps

If any fights still can't be matched after building aliases, use the Wayback Machine to fetch their BFO event pages. The infrastructure is already built:

- `scripts/scrape_wayback_bfo_targeted.py` — searches BFO, finds event URLs, fetches Wayback snapshots, parses odds
- BFO search works (confirmed): e.g., searching "BJ Penn Jon Fitch" returns `/events/ufc-127-penn-vs-fitch-344`
- Wayback has archived BFO pages WITH odds intact (confirmed: UFC 110, UFC 114 had multi-bookmaker odds)
- **Rate limit:** archive.org blocks after ~15-20 requests. Wait 30-60 minutes between batches or use 3-second delays

## Files inventory

### Source datasets (downloaded this session)
```
data/raw/ufc-master.csv                    — Kaggle mdabbert (BFO moneyline + method odds, 6,541 rows)
data/raw/jansen88_ufc_data.csv             — GitHub (BetMMA.tips odds, 7,340 rows)
data/raw/bfo_iankotliar_odds.csv           — GitHub (BFO 12-bookmaker, 4,702 rows)
data/raw/github_PierceHampton.csv          — GitHub (comprehensive + KO/Sub odds, 8,142 rows)
```

### Collected/extracted data
```
data/raw/historical_odds/historical_odds_pre2022_from_cleaned.csv  — 5,428 pre-2022 moneyline rows
data/raw/historical_odds/pre2022_odds_still_missing.csv            — 13 remaining gaps (8 unique)
data/raw/method_odds/historical_method_odds_all.csv                — 5,841 combined method odds rows
data/raw/method_odds/post2014_method_missing.csv                   — 121 post-2014 method gaps
data/raw/kaggle_rankings/rankings_history_extended.csv             — 527 ranking snapshots
```

### Scripts
```
scripts/scrape_wayback_bfo_targeted.py     — Wayback BFO scraper (ready to use)
scripts/scrape_wayback_rankings.py         — Wayback rankings scraper (done)
scripts/scrape_bfo_moneyline.py            — BFO live scraper (BFO has no historical data)
```

## Absolute rules

- **NEVER estimate, impute, or fabricate** any odds value. Every number must come from a real source.
- Do NOT overwrite existing data files — append to or create new files.
- Do NOT touch the BetsAPI ingest if it's still running.
- Do NOT modify rolling stats NaNs — those are honest undefineds for debut fighters.
- Read `V4_HANDOFF_20260318_PARALLEL_DATA.md` and `V4_HANDOFF_20260318_MODEL_QUALITY.md` for full context.
