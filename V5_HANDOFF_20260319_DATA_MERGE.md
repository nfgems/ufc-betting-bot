# V5 Handoff — Data State & Recovery Plan (2026-03-19)

## What happened this session

- Merged 265 moneyline rows from BetsAPI into `historical_odds.csv`
- Matched via exact name + fuzzy name matching (>= 0.75 similarity threshold)
- BetsAPI rows marked with `num_bookmakers=1` to distinguish from multi-book BFO data
- Updated `v5_missing_moneyline.csv` from 524 to 318 remaining

## Coverage table (current)

| Year | Fights | Moneyline | Method Odds |
|------|--------|-----------|-------------|
| 2014-2021 | 3,881 | 100% (0 missing) | 99% (37 missing) |
| 2022 | 511 | 97% (16 missing) | 98% (12 missing) |
| 2023 | 520 | 98% (11 missing) | 90% (50 missing) |
| 2024 | 517 | 96% (21 missing) | 74% (135 missing) |
| 2025 | 522 | 60% (208 missing) | 0% (522 missing) |
| 2026 | 76 | 18% (62 missing) | 0% (76 missing) |
| **TOTAL** | **6,027** | **95% (318 missing)** | **86% (832 missing)** |

---

## File map

### Primary data files (what the model reads)

| File | What it is | Rows | Range |
|------|-----------|------|-------|
| `data/raw/historical_odds/historical_odds.csv` | Moneyline odds | 4,605 | 2021-09 to 2026-03 |
| `data/raw/method_odds/historical_method_odds_all.csv` | Method/finish prop odds | 6,059 | 2012-05 to 2024-12 |
| `data/raw/ufc_fighters_scraped.csv` | Fighter profiles (height/reach/stance/stats) | 4,455 | — |
| `data/raw/ufc_fighters_profile_supplement.csv` | Gap-fill profiles from external sources | 71 | — |
| `data/raw/jansen88_ufc_data.csv` | Base fight results dataset | 7,340 | 1994 to Sep 2023 |

### Gap tracking files (what's still missing)

| File | Rows | What it tracks |
|------|------|---------------|
| `data/raw/v5_missing_moneyline.csv` | 318 | Fights needing moneyline odds |
| `data/raw/v5_missing_method_odds.csv` | 832 | Fights needing method odds |

### BetsAPI data (already extracted what we can)

| File | Rows | Notes |
|------|------|-------|
| `data/processed/betsapi/mma/ended_events.csv` | 8,531 | Event metadata 2022-2026 |
| `data/processed/betsapi/mma/odds_summary_rows.csv` | 12,266 | ML/rounds/totals only — **no method props** |

BetsAPI markets: `162_1` = moneyline, `162_2` = rounds, `162_3` = over/under. It does NOT carry KO/Sub/Decision prop markets. BetsAPI is tapped out for our purposes.

---

## Schemas for appending recovered data

### Moneyline — append to `data/raw/historical_odds/historical_odds.csv`

```
event_date,fighter_a,fighter_b,query_date,offset_days,a_fair_prob,b_fair_prob,a_decimal_odds,b_decimal_odds,num_bookmakers,event_date_dt,fight_key
```

- `a_decimal_odds` / `b_decimal_odds`: decimal format (e.g. 1.5, 2.8)
- `a_fair_prob` = implied prob with vig removed: `(1/a_decimal_odds) / (1/a_decimal_odds + 1/b_decimal_odds)`
- `fight_key` = `{event_date}|{name_a_lower}|{name_b_lower}` (names alphabetically sorted)
- `event_date_dt` = same as `event_date`

### Method odds — append to `data/raw/method_odds/historical_method_odds_all.csv`

```
event_date,fighter_a,fighter_b,event_title,source,source_url,captured_at,a_ko_odds_prob,a_sub_odds_prob,a_dec_odds_prob,b_ko_odds_prob,b_sub_odds_prob,b_dec_odds_prob
```

- All `*_prob` values are implied probabilities (0 to 1), NOT American odds
- Conversion from American: positive `100 / (odds + 100)`, negative `abs(odds) / (abs(odds) + 100)`
- Conversion from decimal: `1 / decimal_odds`

---

## Recovery plan

### Priority 1: Method odds 2025-2026 (598 missing) — CRITICAL

This is the single biggest gap. Zero method odds exist for any fight after Dec 2024.

**Source: The Odds API (historical endpoint)**
- You have an API key
- Query for `fight_winner` market won't help — need `method_of_victory` or equivalent prop market
- Check what historical prop markets are available for MMA/UFC

**Source: BestFightOdds scrape**
- `src/data/method_odds.py` has existing scraper logic for BFO prop pages
- Would need to hit each 2025+ event page and extract KO/Sub/Dec lines

**Source: FanDuel/DraftKings articles**
- Fight preview articles often list method odds
- Tedious but some coverage exists

### Priority 2: Moneyline 2025-2026 (270 missing) — MEDIUM

These are fights where BetsAPI tracked the event but had no odds snapshot. Mostly DWCS and international prelims.

**Source: The Odds API (historical endpoint)**
- Should have moneyline for most UFC events

**Source: BFO scrape**
- Moneyline pages are simpler to scrape than props

### Priority 3: Method odds pre-2025 (234 missing) — LOW

Scattered old gaps across 2014-2024. Multiple recovery scrapes already hit diminishing returns. Most are undercard fights where props were never offered.

### Priority 4: Moneyline pre-2025 (48 missing) — LOW

Likely cancelled bouts or events no bookmaker covered. Probably unrecoverable.

---

## Rules

- **Append-only** to data files — never overwrite existing rows
- **No invented or interpolated values** — every number needs a verifiable source
- **Deduplicate on fight_key** before appending moneyline rows (file already has multiple snapshots per fight at different offsets, that's by design)
- **Check the missing CSVs** for exact fighter name spellings to match against

## After recovery: train V5

```bash
python scripts/train_v4_tuned_candidate.py \
  --base-spec full_live_contract_v5 \
  --output-subdir candidates/full_live_contract_v5_production \
  --candidate-label v5_production
```
