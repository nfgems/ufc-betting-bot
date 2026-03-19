## Update After Web-Curated Recovery Pass (2026-03-19 evening)

- Ran scripts/backfill_v5_web_curated.py
- Recovered all 6 remaining UFC Shanghai moneylines that had public opening odds
- Recovered 31 method-of-victory rows from web-indexed article and prop-history sources
- Current state:
  - Moneyline: **4,953 / 4,953 = 100%** with **0** remaining gaps
  - Method odds: **6,860 / 6,887 = 99.61%** with **27** remaining gaps
- Remaining method gaps now split into:
  - **25 older fights (2014-2020)** that still look genuinely unrecoverable
  - **2 recent fights** still worth probing if needed:
    - 2024-05-11 Waldo Cortes Acosta vs Robelis Despaigne
    - 2025-08-22 Nyamjargal Tumendemberel vs Terrance Saeteurn
# V5 Handoff — Recovery Progress & Next Steps (2026-03-19)

## Recovery Results This Session

### Moneyline: 318 → 7 remaining (311 recovered, 97.8%)
- Recovered 265 from BetsAPI (prior session)
- Recovered 23 from Odds API (smart name matching: reversed names, aliases like Tecia Torres/Pennington, Ariane Lipski/da Silva)
- Recovered 23 more from Odds API after discovering `v5_still_missing_moneyline.csv` had 128 unreconciled rows
- **Both missing files now synced and verified against master CSV**
- **Coverage: 4,946 / 4,953 = 99.9%**

### Method Odds: 832 → 58 remaining (774 recovered, 93.0%)
- Recovered 126 from BFO Playwright checkpoint (data was sitting in checkpoint, never appended)
- Recovered 157 from BFO static HTML scraper (new `backfill_v5_method_bfo_fast.py`)
- Recovered 6 from BFO aggressive name matching retry
- Recovered 2 from BFO direct fight-page search
- **Coverage: 6,829 / 6,887 = 99.2%**

---

## Current Coverage Table

| Year | Fights | Moneyline | Method Odds |
|------|--------|-----------|-------------|
| 2014-2021 | 3,881 | 100% | 99.5% (20 missing) |
| 2022 | 511 | 100% | 100% |
| 2023 | 520 | 99.8% (1 miss) | 99.4% (3 missing) |
| 2024 | 517 | 100% | 98.8% (6 missing) |
| 2025 | 522 | 98.9% (6 miss) | 95.4% (24 missing) |
| 2026 | 76 | 100% | 93.4% (5 missing) |
| **TOTAL** | **6,027** | **99.9% (7 missing)** | **99.0% (58 missing)** |

---

## 7 Remaining Moneyline Gaps (unrecoverable)

| Date | Fight | Why |
|------|-------|-----|
| 2024-03-23 | Igor Severino vs Andre Lima | Not in Odds API |
| 2025-08-23 | Maheshate vs Gauge Young | UFC Shanghai — Chinese fighters not in any API |
| 2025-08-23 | Rongzhu vs Austin Hubbard | UFC Shanghai |
| 2025-08-23 | Sergei Pavlovich vs Waldo Cortes Acosta | UFC Shanghai |
| 2025-08-23 | Sumudaerji vs Kevin Borjas | UFC Shanghai |
| 2025-08-23 | Xiao Long vs SuYoung You | UFC Shanghai |
| 2025-08-23 | Yizha vs Westin Wilson | UFC Shanghai |

---

## 58 Remaining Method Odds Gaps

### Tier 1: Recoverable (33 fights, 2023+)

These are past events where method props WERE offered but we couldn't scrape them:

**9 DWCS/undercard fights (2023-2024)** — BFO event pages had props but name matching failed or props were sparse:
- 2023-08-26: Waldo Cortes Acosta vs Lukasz Brzeski
- 2023-11-11: Nazim Sadykhov vs Viacheslav Borshchev
- 2023-12-16: Martin Buday vs Shamil Gaziev
- 2024-02-03: Aliaskhab Khizriev vs Makhmud Muradov
- 2024-03-16: Bryan Battle vs Ange Loosa
- 2024-03-23: Montse Rendon vs Daria Zhelezniakova
- 2024-05-11: Waldo Cortes Acosta vs Robelis Despaigne
- 2024-06-01: Mickey Gall vs Bassil Hafez
- 2024-08-24: Zach Reese vs Jose Daniel Medina

**12 UFC Nashville (2025-07-12)** — BFO event pages `3748` and `3749` return NO method props (only 27-29KB of moneyline data). Correct event URL may exist under a different ID, OR BFO never listed props for this card. Web searches found partial odds (e.g. Lewis KO +275, Teixeira KO -175 from gambling911.com; Bonfim sub +125). Need to:
1. Try Playwright with broader event ID search (3740-3760)
2. Fetch articles from gambling911.com, docsports.com, oddschecker.com that mention specific method odds
3. Use WebFetch on fight-specific preview URLs from search results

**10 UFC 318 (2025-07-19)** — BFO page `3706` only had Holloway/Poirier props (3 fighters). Page `3707` was a **homepage redirect** (not a real event). The correct event page with full card props hasn't been found yet. Web searches found partial odds (e.g. Zellhuber KO -125 from DraftKings). Same recovery approach as Nashville.

**2 UFC Shanghai (2025-08-22)** — BFO pages `3818`/`3819` had no props / homepage redirect. These are Chinese-fighter events, harder to find.

### Tier 2: Unrecoverable (25 fights, pre-2019)

BFO did not track method-of-victory props before ~2019. The Odds API doesn't support MMA method markets. No other accessible source has historical method props from this era. These 25 fights (2014-2020) are genuinely unrecoverable.

---

## Recovery Strategy for Remaining 33 Method Odds

### Approach 1: Web article scraping (RECOMMENDED)
Many fight preview sites publish method odds in article text. Web searches already found:
- **gambling911.com** — Published structured method odds (Lewis KO +275, Teixeira KO -175, etc.)
- **docsports.com** — Has method picks with odds
- **oddschecker.com** — Has method markets
- **clutchpoints.com** — Mentions specific method odds in predictions
- **betmgm.com blog** — Has method odds in preview articles

**Strategy:** For each missing fight, WebSearch for `"FighterA" "FighterB" method victory odds`, then WebFetch the top results and parse American odds from article text. Convert to implied probabilities using standard formula.

### Approach 2: FightOdds.io / Sofascore
These sites aggregate historical method odds but block scraping (403). Playwright may work.

### Approach 3: DraftKings/FanDuel historical
DraftKings `sportsbook.draftkings.com/leagues/mma/ufc?category=winning-method` has method-of-victory markets. These are live sportsbook pages that require JS rendering. Playwright could potentially navigate to historical fight pages.

---

## File Map

| File | Rows | Status |
|------|------|--------|
| `data/raw/historical_odds/historical_odds.csv` | 4,946 | Master moneyline |
| `data/raw/method_odds/historical_method_odds_all.csv` | 6,829 | Master method odds |
| `data/raw/v5_missing_moneyline.csv` | 7 | Verified against master |
| `data/raw/v5_still_missing_moneyline.csv` | 7 | Synced with above |
| `data/raw/v5_missing_method_odds.csv` | 58 | 33 recoverable, 25 not |

## Scripts Created This Session

| Script | Purpose |
|--------|---------|
| `scripts/backfill_v5_smart_recovery.py` | Odds API ML with improved name matching (reversed names, aliases) |
| `scripts/backfill_v5_method_bfo_fast.py` | BFO method odds via static HTML (no Playwright needed) |
| `scripts/backfill_v5_bfo_direct.py` | BFO individual fight page search for method odds |
| `scripts/backfill_v5_bfo_discover.py` | Playwright-based BFO event URL discovery via fighter pages |
| `scripts/backfill_v5_google_method.py` | Google search for method odds (blocked by CAPTCHA) |

## Known Issues / Gotchas

1. **BFO event ID `+1` trap**: Some events have two BFO pages (e.g. `3706` and `3707`). The `+1` page sometimes redirects to the BFO homepage (3.5MB, 800+ fighters). The Playwright script recovered 8 rows from wrong-event data that had to be **rolled back**. Always validate page title !== "UFC & MMA Odds & Betting Lines | Best Fight Odds" and content size < 2MB.

2. **Name aliases in data**: `Tecia Pennington` = Tecia Torres, `Ariane da Silva` = Ariane Lipski, `King Green` = Bobby Green. These are married/ring name changes.

3. **Odds API has no MMA method markets**: `method_of_victory`, `fighter_method_of_victory`, `winning_method` all return 422. Only `h2h` works for MMA historical.

4. **v5_still_missing_moneyline.csv was stale**: Had 128 rows that were never reconciled after Odds API recovery. Both files are now synced.

---

## After Method Odds Recovery: Train V5

```bash
python scripts/train_v4_tuned_candidate.py \
  --base-spec full_live_contract_v5 \
  --output-subdir candidates/full_live_contract_v5_production \
  --candidate-label v5_production
```

Check that the `full_live_contract_v5` spec exists in `src/model/training_spec.py` before running. If not, create it based on the best v4 candidate with updated data paths.
