# Tennis Odds Handoff - 2026-03-21

## Continuing from

- Previous handoff: `data/processed/tennis/CLAUDE_HANDOFF_2026-03-20.md`
- Starting coverage: `23,883 / 24,572` (97.2%)

## What happened this session

Ran parallel web searches across PokerStars, Betfair, Smarkets, Matchstat, SportyTrader, SignalOdds, Dexwin, nb-bet, OddsPortal, PickDawgz, scores24, checklive, Betmonitor, BetUS, TennisTonic, and Bleacher Nation for all 75 remaining `source_absent` rows dated 2025-2026.

**No rows were added to `manual_search_snippets_odds.csv` yet** — the session was interrupted before appending. All finds below need to be verified and appended.

## Verified finds (page content directly confirmed)

These odds were confirmed by fetching the actual page and reading explicit odds text:

### 1. J. Engel vs A. Blockx — Next Gen ATP Finals 2025 (RR, Dec 17)
- **Blockx -286 / Engel +175** (American) → **Blockx 1.35 / Engel 2.75** (decimal)
- Source: PickDawgz
- URL: `https://pickdawgz.com/tennis-picks/blockx-vs-engel-prediction-12-17-2025-todays-tennis-picks/`
- Exact quote: "The Line: Betting Odds: Alexander Blockx -286 / Justin Engel +175"

### 2. Tatjana Maria vs Irene Burillo — Quito 125 2025 (R32, Dec 1)
- **Maria -200 / Burillo +150** (American) → **Maria 1.50 / Burillo 2.50** (decimal)
- Source: PickDawgz
- URL: `https://pickdawgz.com/tennis-picks/quito-prediction-maria-vs-burillo-escorihuela-12-2-2025-todays-tennis-picks/`
- Exact quote: "The Line: Betting Odds: Tatjana Maria -200 / Irene Burillo Escorihuela +150"

## Agent-reported finds (search snippets, NOT independently page-verified)

These were found by search agents reading search result snippets or pages that returned 403 when I tried to verify directly. They should be verified before adding.

### 3. T. Barrios Vera vs T. Seyboth Wild — Argentina Open 2026 (R32, Feb 10)
- **Barrios Vera 1.84 / Seyboth Wild 1.96** (from TennisTonic search snippet)
- Also: **Barrios Vera -160 / Seyboth Wild +125** (from Bleacher Nation snippet) → decimal ~1.625 / ~2.25
- URLs: `https://tennistonic.com/tennis-news/957595/...` (403 on fetch), `https://www.bleachernation.com/picks/2026/02/09/...` (403 on fetch)
- ⚠️ Two sources give different odds. Need to pick one or verify.

### 4. A. Zolotareva vs J. Riera — Antalya 125 #1 2026 (R32, Feb 24)
- **Riera 1.34 / Zolotareva 3.25** (from SportyTrader search snippet)
- URL: `https://www.sportytrader.com/en/odds/julia-riera-anastasia-zolotareva-8336684/` (403 on fetch)

### 5. L. Fruhvirtova vs B. Haddad Maia — Austin 125 #1 2026 (R32, Mar 9)
- **Fruhvirtova 1.83 / Haddad Maia 2.01** (from scores24.live search snippet)
- URL: `https://scores24.live/en/tennis/m-09-03-2026-fruhvirtova-linda-haddad-maia-beatriz-prediction` (403 on fetch)

### 6. Carla Markus vs Laura Pigossi — Tucuman 125 2025 (R32, Nov 3)
- **Pigossi 1.11 / Markus 6.05** (from Betmonitor search snippet)
- URL: betmonitor.com odds comparison (403 on fetch)

### 7. Jazmin Ortenzi vs Candela Vazquez — Tucuman 125 2025 (R32, Nov 3)
- **Ortenzi 1.14 / Vazquez 4.3** (from Smarkets search snippet)
- URL: `https://smarkets.com/listing/sport/tennis/wta-125k/wta-125k-tucuman-argentina-women-singles` (JS-rendered)

### 8. Amandine Monnot vs Yuriko Miyazaki — Angers 125 2025 (R32, Dec 1)
- **Miyazaki 1.48 / Monnot 2.75** (from SportyTrader search snippet, LSbet/Betway sourced)
- URL: `https://www.sportytrader.com/en/odds/yuriko-lily-miyazaki-amandine-monnot-7933940/` (403 on fetch)

### 9. Guiomar Maristany vs Irene Burillo — Mallorca 125 2025 (R32, Oct 5)
- **Maristany 1.59 / Burillo 2.16** (from Smarkets search snippet)
- URL: smarkets.com WTA 125K Mallorca (JS-rendered, confirmed page exists but odds not in static HTML)

## Matches confirmed NOT to have occurred

### ATP Finals 2025
- **Shelton vs Alcaraz**: Did NOT occur. They were in different groups (Borg vs Connors).
- **Auger-Aliassime vs de Minaur**: Did NOT occur. Same reason — different groups.
- These are likely upstream match-data issues, not odds recovery targets.

### Next Gen ATP Finals 2025
- **Engel vs Blockx Final**: Did NOT occur. The actual final was **Learner Tien vs Alexander Blockx** (Tien won).
- The RR match (Engel vs Blockx) DID occur and odds were found (see verified find #1).
- **Blockx vs Jodar (RR)**: Match occurred but no odds found from any source.

## Existing local data that agents flagged as potentially already loaded

These were found in existing scraped source files. They are probably already being picked up by the matcher — if they're still showing as `source_absent`, the matcher may have a name-matching issue:

- Bernarda Pera vs Fernanda Contreras (Puerto Vallarta 125): in `betexplorer_odds.csv` — Pera 1.07 / Contreras 7.37
- Alizé Cornet vs Irene Burillo (La Bisbal 125): in `betexplorer_odds.csv` — Cornet 2.14 / Burillo 1.66
- Renata Zarazua vs Irene Burillo (Vic 125): in `betexplorer_odds.csv` — Zarazua 1.12 / Burillo 5.13
- Elisabetta Cocciaretto vs Irene Burillo (Bastad 125): in `betexplorer_odds.csv` — Cocciaretto 1.13 / Burillo 5.42
- Simona Waltert vs Irene Burillo (Iasi 2025): in `tennis_data_co_uk_odds.csv` — Waltert 1.30 / Burillo 3.60

⚠️ If these are genuinely in the local source files but still showing `source_absent`, investigate whether the matcher is failing to match them (name normalization, tournament name mismatch, date offset, etc.).

## Not found (exhaustive search, no results)

These matches had no recoverable odds from any searched source:

### WTA 125 2025
- Francesca Jones vs Noma Noha Akugue (Oeiras 125, Apr 14)
- Victoria Jimenez Kasintseva vs Yue Yuan (Oeiras 125, Apr 14)
- Fiona Ferro vs Noma Noha Akugue (Saint Malo 125, Apr 28)
- Renata Zarazua vs Patricia Maria Tig (Parma 125, May 12)
- Greet Minnen vs Kimberly Birrell (Birmingham 125, Jun 2)
- Emily Appleton vs Simona Waltert (Ilkley 125, Jun 9)
- Viktorija Golubic vs Lucrezia Stefanini (Ilkley 125, Jun 9)
- Sofia Costoulas vs Martyna Kubka (Warsaw 125, Jul 28)
- Xinyu Gao vs Tamara Korpatsch (Warsaw 125, Jul 28)
- Alexandra Eala vs Varvara Lepchenko (Guadalajara 125, Sep 1)
- Francesca Jones vs Victoria Rodriguez (Guadalajara 125, Sep 1)
- Carson Branstine vs Anouk Koevermans (Montreux 125, Sep 1)
- Darya Astakhova vs Nuria Brancaccio (Montreux 125, Sep 1)
- Sinja Kraus vs Silvia Ambrosio (Ljubljana 125, Sep 8)
- Tiphanie Lemaitre vs Leonie Kung (Cosenza 125, Sep 29)
- Kaitlin Quevedo vs Carol Young Suh Lee (Samsun 125, Sep 29)
- Leolia Jeanjean vs Maddison Inglis (Suzhou 125, Sep 29)
- Linda Fruhvirtova vs Anastasia Zakharova (Suzhou 125, Sep 29)
- Sinja Kraus vs Julia Konishi Camargo Silva (Rio 125, Oct 13)
- Irene Burillo vs Maribella Zamarripa (Florianopolis 125, Oct 20)
- Leyre Romero Gormaz vs Julia Konishi Camargo Silva (Florianopolis 125, Oct 20)
- Alicia Herrero Linana vs Marianne Angel Gonzalez (Queretaro 125, Oct 20)
- Maja Chwalinska vs Tania Varela (Quito 125, Dec 1)
- Miriam Bulgaru vs Irene Burillo (Buenos Aires 125, Nov 24) — partial find only

### ATP Metz 2025 (all 9 matches)
- No odds found from any source. OddsPortal has results but JS-rendered. SportyTrader returned 403.

### 2026 WTA 125
- Mai Hontama vs Elizabeth Abarquez (Manila 125, Jan 26)
- Tatiana Prozorova vs Kaye Ann Emana (Manila 125, Jan 26)
- Misaki Matsuda vs Ankita Raina (Mumbai 125, Feb 2)
- Lucrezia Stefanini vs Angelina Voloshchuk (Oeiras 125 Indoor #1, Feb 9)
- Carol Young Suh Lee vs Anna-Lena Friedsam (Les Sables 125, Feb 16)
- Mona Barthel vs Tiphanie Lemaitre (Les Sables 125, Feb 16)
- Katarina Jokic vs Elizabeth Jones (Midland 125, Feb 16)
- Viktorija Golubic vs Ana Filipa Santos (Oeiras 125 Indoor #2, Feb 16)
- Moyuka Uchijima vs Katarzyna Kawa (Antalya 125 #1, Feb 24)
- Leyre Romero Gormaz vs Carole Monnet (Antalya 125 #2, Mar 3)
- Oksana Selekhmeteva vs Yue Yuan (Austin 125 #1, Mar 9)
- Ekaterine Gorgodze vs Arantxa Rus (Antalya 125 #3, Mar 10)

## What Claude should do next

### Immediate: Append verified rows
Add verified finds #1 and #2 to `manual_search_snippets_odds.csv`, then rerun `python scripts/fix_tennis_odds_matching.py`.

### High-value: Verify agent-reported finds
Finds #3–#9 need page-level verification. The search snippets are promising but the actual pages were 403 from shell. Options:
1. Try fetching at different times (some sites have rate limits)
2. Use browser-based access to confirm snippet content
3. If user can confirm the snippet quotes are correct, add them directly

### Investigate: Local data not matching
The 5 matches flagged as already in local source files but still `source_absent` suggest a matcher bug. Check `scripts/fix_tennis_odds_matching.py` for:
- Tournament name normalization (e.g., "La Bisbal D'Empordá" vs BetExplorer's slug)
- Date offsets (BetExplorer dates may differ by ±1 day)
- Player name matching (accents, abbreviations)

### Low-value / skip
- ATP Finals Shelton/Alcaraz and FAA/de Minaur — upstream data issues, not odds gaps
- Next Gen Finals final (Engel/Blockx) — match didn't happen
- Metz 2025 — all 9 matches exhaustively searched with no results
- Older 2022-2024 source_absent rows — not searched this session

## Guardrails reminder
- No data was fabricated
- No rows were added to any CSV this session
- All finds above have explicit source attribution
- American odds conversion: +X → 1 + X/100; -X → 1 + 100/X
