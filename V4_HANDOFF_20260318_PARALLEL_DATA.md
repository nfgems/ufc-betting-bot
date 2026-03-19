# V4 Handoff 2026-03-18: Parallel Data Scraping Session

## Mission

**Get as close to 100% data coverage as possible.** Exhaust every available online source. Nothing should EVER be estimated, imputed, or fabricated. If a value can't be found from a real source, it stays NaN. But the goal is to leave no stone unturned — scrape every site, download every dataset, check every archive until the gaps are filled.

## Situation Right Now

There is a **long-running BetsAPI ingest process already executing** in another terminal. Do NOT interfere with it, restart it, or duplicate its work.

### BetsAPI ingest status

- Script: `from src.data.betsapi_mma import backfill_betsapi_mma_history`
- Phase 1 (events_ended): **COMPLETE** — 1,545 day-files covering 2022-01-01 through 2026-03-07, containing 8,531 total MMA events
- Phase 2 (odds_summary): **IN PROGRESS** — ~1,272 / 8,531 events downloaded as of this handoff (~15%)
- Rate: ~66 events/minute
- ETA: ~1.5–2 hours remaining
- Output dir: `data/raw/betsapi/mma/`
- The process has auto-retry logic and handles intermittent HTTP 502s correctly

### What BetsAPI covers (post-2022 only)

The odds_summary files contain multi-bookmaker odds across these markets:
- `162_1`: **moneyline** (home/away decimal odds)
- `162_2`: handicap
- `162_3`: totals

The existing ETL in `src/data/betsapi_mma.py` extracts:
- **Moneyline implied probabilities** (`a_implied_prob` / `b_implied_prob` / `diff_implied_prob`)
- **Method odds** (KO/sub/decision probabilities per fighter)
- Additional market-structure features (overround, cross-market agreement, coverage flags)

**BetsAPI only covers 2022+.** It will NOT help with pre-2022 gaps.

## Your Job: Scrape The Missing Data In Parallel

While the BetsAPI ingest runs, aggressively scrape online sources for the data gaps it does NOT cover. There are **6,425 pre-2022 fights** in the training dataset that currently have **zero** moneyline and method odds coverage. There are also major rankings gaps.

### Task 1: Moneyline Odds — Pre-2022 Backfill (6,425 fights, currently 100% null)

This is the highest-impact gap. The training dataset has 8,571 total rows. 6,425 are pre-2022 and have NO moneyline odds at all. The existing `data/raw/historical_odds/historical_odds.csv` only has 3,776 rows starting from 2021-02-06.

**Current schema** (`data/raw/historical_odds/historical_odds.csv`):
```
event_date,fighter_a,fighter_b,query_date,offset_days,a_fair_prob,b_fair_prob,a_decimal_odds,b_decimal_odds,num_bookmakers
```

**Sources to scrape — exhaust ALL of these:**

1. **BestFightOdds.com** — The most comprehensive historical MMA odds archive. Has fight odds going back to the early UFC days. Scrape opening/closing moneylines for every UFC event.

2. **OddsPortal.com** — Historical odds archive with multiple bookmaker consensus odds. Has MMA/UFC history.

3. **Kaggle UFC datasets** — Several datasets exist with historical odds:
   - Search for "UFC odds", "MMA betting odds", "UFC historical odds"
   - The repo already has a `scripts/merge_kaggle_data.py` and `src/data/kaggle_loader.py` — check what they expect

4. **The Odds API** (historical endpoint) — If there's an API key available, pull historical MMA odds

5. **Action Network / ESPN odds archives** — May have historical UFC odds

6. **Web Archive (archive.org)** — Wayback Machine snapshots of BestFightOdds or other odds sites for older events

**Output rules:**
- Write to `data/raw/historical_odds/historical_odds_pre2022.csv` (or split by source)
- Match the existing schema above, or document any schema differences clearly
- Do NOT overwrite `historical_odds.csv`
- Every row must have a real source — never estimate fair probabilities

### Task 2: Method Odds — Pre-2022 Backfill (6,425 fights, currently 100% null)

Method odds (KO/sub/decision probabilities) are even harder to find historically, but they may exist on the same sources.

**The 6 features:**
- `a_ko_odds_prob`, `b_ko_odds_prob`
- `a_sub_odds_prob`, `b_sub_odds_prob`
- `a_dec_odds_prob`, `b_dec_odds_prob`

**Sources to try:**

1. **BestFightOdds.com** — Often has method-of-victory prop odds for UFC events
2. **OddsPortal.com** — Check for method/prop markets
3. **BetMMA.tips** — Historical MMA prop odds
4. **Kaggle** — Some UFC datasets include prop odds
5. **BetsAPI historical** — The current ingest covers 2022+, but check if older MMA data exists in their API

**Output:** Write to `data/raw/method_odds/` directory (currently empty). Document the schema.

### Task 3: Rankings — Fill the Gaps (currently 60-98% null in features)

**You already have data here.** `data/raw/rankings/rankings_history.json` contains **480 weekly snapshots** from 2013-02-04 to 2025-05-06. Coverage by year:

```
2013: 31    2014: 46    2015: 36    2016: 36    2017: 37
2018: 37    2019: 41    2020: 38    2021: 41    2022: 39
2023: 43    2024: 40    2025: 15
```

**Problems to solve:**
1. **Gap from 2025-05-06 to 2026-03-18** — ~10 months of missing snapshots. Scrape current and recent rankings to fill this.
2. **Pre-2013 gap** — No ranking data before Feb 2013. UFC rankings started in Feb 2012. Try to recover the first year.
3. **Despite having 480 snapshots, features are still 60-98% null** — This likely means the merge path from `rankings_history.json` into the training features is broken or not wired up. Investigate why.
4. **Women's divisions** — Check if the existing snapshots include women's weight classes (strawweight, flyweight, bantamweight, featherweight). These were added over time.

**Sources to scrape for missing ranking snapshots:**

1. **UFC.com/rankings** — Current rankings (use existing `src/data/rankings_scraper.py`)
2. **Web Archive** — Wayback Machine snapshots of UFC.com/rankings for the 2025-05 to 2026-03 gap and pre-2013
3. **Sherdog rankings archives**
4. **Wikipedia UFC rankings history** — Sometimes has dated ranking tables
5. **MMA Junkie / MMA Fighting** — Historical ranking articles with dated snapshots

**Output:** Extend `data/raw/rankings/rankings_history.json` with new snapshots using the same format:
```json
{
  "YYYY-MM-DD": {
    "POUND-FOR-POUND": [{"fighter": "NAME", "rank": 1}, ...],
    "FLYWEIGHT": [...],
    ...
  }
}
```

### Task 4: TD Rate Nulls — DO NOT TOUCH

These are honest undefineds from fighters with zero takedown attempts. They are **not** a data gap.

- `diff_roll_td_acc`: 313 nulls, `diff_roll_td_def`: 222, `b_roll_td_acc`: 196, `b_roll_td_def`: 134, `a_roll_td_acc`: 132, `a_roll_td_def`: 94

Do NOT impute, fill, or try to fix these. They represent real missing denominators.

## Priority Order

1. **Pre-2022 moneyline** — Biggest bang for the buck. 6,425 fights with zero coverage.
2. **Rankings gap-fill** — Already have partial data, need to extend and verify the merge path works.
3. **Pre-2022 method odds** — Hardest to find but very valuable if available.

## Key Context

### Dataset shape

- Total training rows: **8,571**
- Pre-2022: **6,425** (75% of data — this is the bulk of training)
- Post-2022: **2,146** (25%)
- Train cutoff: `2022-01-01`
- Train split: 3,374 rows | Test split: 1,420 rows

### Active model line: 138-feature overlay

- Base: audited 129-feature V4 contract
- Overlay: +9 features from moneyline (3) and method odds (6)
- Rankings (6) currently excluded — if coverage improves enough, they can be added (would make it 144)

### Current best candidates (benchmarks, do not retrain yet)

| model | features | accuracy | log_loss | brier |
|---|---:|---:|---:|---:|
| `v4_138_tuned02_nocal_noise002` | 138 | 0.6465 | 0.6320 | 0.2209 |
| `v4_138_tuned03_nocal_noise002_d6m8` | 138 | 0.6373 | 0.6290 | 0.2195 |
| `v4_129_tuned02_altxgb` | 129 | 0.6366 | 0.6451 | 0.2269 |
| `old_v4_122` | 122 | 0.6169 | 0.6538 | 0.2309 |

### Important files

- Historical odds: `data/raw/historical_odds/historical_odds.csv` (3,776 rows, 2021-02 to 2026-03)
- Rankings data: `data/raw/rankings/rankings_history.json` (480 snapshots, 2013-02 to 2025-05)
- Rankings scraper: `src/data/rankings_scraper.py`
- Method odds dir: `data/raw/method_odds/` (empty)
- BetsAPI ETL: `src/data/betsapi_mma.py`
- Kaggle loader: `src/data/kaggle_loader.py`
- Kaggle merge script: `scripts/merge_kaggle_data.py`
- Feature building: `src/features/build_features.py`
- Data loading: `src/data/ufc_refresh.py`
- Null audit: `scripts/audit_model_feature_nulls.py`
- Spec: `src/model/training_spec.py`

### Absolute rules

- **NEVER estimate, impute, interpolate, or fabricate any data value.** Every number must come from a real source.
- Do NOT interfere with the running BetsAPI ingest process
- Do NOT overwrite existing data files — write new data to new files
- Do NOT retrain models — this session is data collection only
- Do NOT commit or push without explicit approval
- Read any file before modifying it — the worktree is dirty with many uncommitted changes

### Repo state

- Branch: `recovery/repo-cleanup-20260317`
- Dirty worktree with many uncommitted changes
- Full test suite last passed: `418 passed, 1 skipped`
