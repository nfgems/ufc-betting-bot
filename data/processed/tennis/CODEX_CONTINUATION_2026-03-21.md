# Tennis Odds Continuation - 2026-03-21

## Cleanup update after quarantining bad match rows

- Patched `scripts/fix_tennis_odds_matching.py` to write a modeling-safe export and quarantine files:
  - `data/processed/tennis/matches_with_odds_modeling_safe.csv`
  - `data/processed/tennis/known_bad_match_rows.csv`
  - `data/processed/tennis/suspected_bad_match_rows.csv`
- The raw output `matches_with_odds.csv` is still written for provenance, but the safe export now excludes:
  - `5` confirmed bad rows
  - `15` suspected bad rows with explicit reasons
- Reran `.\.venv\Scripts\python.exe scripts\fix_tennis_odds_matching.py`
  - raw rows: `24,572`
  - modeling-safe rows: `24,552`
  - rows with any odds stayed `23,983` in both files
- The confirmed bad rows are the impossible self-vs-self `Metz 2025` records:
  - `J. Choinski vs J. Choinski`
  - `K. Jacquet vs K. Jacquet`
  - `F. Passaro vs F. Passaro`
  - `V. Sachko vs V. Sachko`
  - `L. Van Assche vs L. Van Assche`
- The suspected bad quarantine currently includes:
  - `Leylah Fernandez vs Lena Rueffer` at `Bad Homburg 2023`
  - the corrupted `Metz 2025` non-self rows
  - the two `Nitto ATP Finals 2025` rows (`Shelton/Alcaraz`, `FAA/de Minaur`)
  - the two `Next Gen ATP Finals 2025` rows (`Blockx/Jodar`, bogus `Engel/Blockx` final)
- Patched `scripts/tennis_backtest.py` to prefer `matches_with_odds_modeling_safe.csv` automatically when it exists.
- Extended the same cleanup to the base match table:
  - wrote `data/processed/tennis/matches_modeling_safe.csv`
  - wrote `data/processed/tennis/known_bad_base_match_rows.csv`
  - wrote `data/processed/tennis/suspected_bad_base_match_rows.csv`
  - base raw rows: `351,743`
  - base modeling-safe rows: `351,720`
  - base confirmed bad rows: `8`
  - base suspected bad rows: `15`
- The extra 3 confirmed-bad base rows are old `U Unknown vs U Unknown` records:
  - `Barcelona 1969` x2
  - `Surbiton 1974` x1
- Patched `src/data/tennis_data.py` so `load_processed_tennis_data()` now prefers `matches_modeling_safe.csv` automatically when it exists.
- Also patched `save_processed_tennis_data()` so if raw `matches.csv` is overwritten later, the stale safe base exports are removed and must be regenerated cleanly.

## What changed in this continuation

- Patched `scripts/fix_tennis_odds_matching.py` to:
  - match shortened player surnames against compound source surnames
    - examples fixed: `Fernanda Contreras` vs `Contreras Gomez F.`, `Irene Burillo` vs `Burillo Escorihuela I.`
  - preserve optional source `round` metadata and require round compatibility when present
    - this prevents the verified `J. Engel` vs `A. Blockx` round-robin odds from being applied to the bogus final row
- Updated `data/raw/tennis/manual_search_snippets_odds.csv`
  - added optional `round` column to the header
  - appended 9 verified rows

## Verified rows appended

- `2025-10-06` Guiomar Maristany Zuleta De Reales vs Irene Burillo
- `2025-11-04` Carla Markus vs Laura Pigossi
- `2025-11-04` Jazmin Ortenzi vs Candela Vazquez
- `2025-12-02` Tatjana Maria vs Irene Burillo
- `2025-12-02` Amandine Monnot vs Yuriko Miyazaki
- `2025-12-17` J. Engel vs A. Blockx
- `2026-02-10` T. Barrios Vera vs T. Seyboth Wild
- `2026-02-24` Anastasia Zolotareva vs Julia Riera
- `2026-03-09` Linda Fruhvirtova vs Beatriz Haddad Maia

## Matcher rerun result

- Command run: `.\.venv\Scripts\python.exe scripts\fix_tennis_odds_matching.py`
- New coverage: `23,904 / 24,572`
- Previous coverage from handoff: `23,883 / 24,572`
- Net improvement this continuation: `+21` matched rows

## Confirmed recovered targets

- Manual-source recoveries now matched:
  - Guiomar Maristany Zuleta De Reales vs Irene Burillo
  - Carla Markus vs Laura Pigossi
  - Jazmin Ortenzi vs Candela Vazquez
  - Tatjana Maria vs Irene Burillo
  - Amandine Monnot vs Yuriko Miyazaki
  - J. Engel vs A. Blockx (`RR` row only)
  - T. Barrios Vera vs T. Seyboth Wild
  - Anastasia Zolotareva vs Julia Riera
  - Linda Fruhvirtova vs Beatriz Haddad Maia
- Previously flagged local-source matcher misses now recovered:
  - Bernarda Pera vs Fernanda Contreras
  - Alizé Cornet vs Irene Burillo
  - Renata Zarazua vs Irene Burillo
  - Elisabetta Cocciaretto vs Irene Burillo
  - Simona Waltert vs Irene Burillo

## Important guardrail outcome

- `J. Engel` vs `A. Blockx` still exists twice in `matches.csv`
  - `RR` row is now matched
  - bogus `F` row remains unmatched
  - this is intentional and was the reason for adding round-aware source matching

## Remaining state after rerun

- `248` rows classified as `source_absent`
- `415` rows classified as `source_has_no_numeric_odds`
- `5` rows classified as `upstream_match_record_issue`

## Notes for the next continuation

- The authoritative post-rerun files are:
  - `data/processed/tennis/matches_with_odds.csv`
  - `data/processed/tennis/unmatched_diagnostics.csv`
  - `data/processed/tennis/odds_matching_report.txt`
- `odds_matching_report.txt` still includes some stale pre-Pass-3 name-mismatch lines in the narrative section.
  - Example: it still mentions `T. Barrios Vera vs T. Seyboth Wild` under name mismatches even though the row is matched in `matches_with_odds.csv`.
  - Trust `matches_with_odds.csv` and `unmatched_diagnostics.csv` over that narrative subsection.

## Further continuation on 2026-03-21

- Ran additional targeted BetExplorer scrapes and reruns after the first note above.
  - `.\.venv\Scripts\python.exe scripts\scrape_be_targeted.py bjk_g1`
  - `.\.venv\Scripts\python.exe scripts\scrape_be_targeted.py davis_wg`
  - `.\.venv\Scripts\python.exe scripts\scrape_be_targeted.py nextgen`
  - `.\.venv\Scripts\python.exe scripts\scrape_be_targeted.py wta125 oeiras saint-malo parma birmingham ilkley warsaw guadalajara montreux ljubljana cosenza samsun suzhou queretaro quito manila mumbai les-sables midland antalya austin`
- Those team-event scrapes were the big mover.
  - Coverage moved from `23,904 / 24,572` to `23,934 / 24,572`
  - `source_absent` dropped from `248` to `81`
  - `source_has_no_numeric_odds` rose from `415` to `552`
  - Interpretation: many former true absences became real source pages with no numeric odds
- The targeted WTA 125 scrape refreshed `betexplorer_wta125_odds.csv` but did not recover any of the remaining `source_absent` rows.

## One more verified manual recovery

- Appended 1 more verified row to `data/raw/tennis/manual_search_snippets_odds.csv`
  - `2025-11-07` Aryna Sabalenka vs Amanda Anisimova
  - source: `https://www.goonersguide.com/tennis-pick-57655-Aryna-Sabalenka-vs-Amanda-Anisimova.htm`
  - verified fixed odds on page: Bet365 `1.40 / 2.75`
  - round: `SF`
- Reran `.\.venv\Scripts\python.exe scripts\fix_tennis_odds_matching.py`
  - Coverage moved from `23,934 / 24,572` to `23,935 / 24,572`
  - `source_absent` dropped from `81` to `80`
  - The row matched correctly in `matches_with_odds.csv` with `odds_date_diff = 6`
  - This confirms some late-season official rows use tournament-start dates rather than actual match dates, but the matcher's date window still recovers them

## Verified dead ends from this pass

- Exhaustive local-source check:
  - loaded all additional source tables used in Pass 3
  - for the remaining `source_absent` rows, no row had any pair-level source match within `+/-14` days under the current name matcher
  - conclusion: the remaining `source_absent` rows are not just another local normalization miss
- Oddspedia US investigation:
  - several residual matches do have public odds-comparison pages
  - examples checked: `Emily Appleton vs Simona Waltert`, `Noma Akugue Noha vs Francesca Jones`, `Yue Yuan vs Victoria Jimenez Kasintseva`
  - those pages exposed moneyline compare blocks with different books on each side
    - example pattern: home best from one book, away best from another
  - I did not append those because that would create synthetic mixed-book rows
- Sportsbook/direct-page probes that did not yield usable single-book winner odds:
  - FanDuel direct event pages
  - DraftKings direct event pages
  - Bovada/Bodog
  - Oddschecker
  - Odds.am
  - BettingExpert
  - Matchstat H2H/odds pages
  - Betimate
  - DexWin AI picks pages

## Current baseline after all work in this continuation

- Coverage: `23,935 / 24,572`
- Remaining unmatched classifications:
  - `552` `source_has_no_numeric_odds`
  - `80` `source_absent`
  - `5` `upstream_match_record_issue`
- Biggest remaining `source_absent` buckets:
  - `10` `Metz 2025`
  - `3` `Atp Cup 2022`
  - `3` `Davis Cup WG2 R1: ESA vs IRL 2023`
  - `2` each in `Oeiras 125 2025`, `Ilkley 125 2025`, `Warsaw 125 2025`, `Guadalajara 125 2025`, `Montreux 125 2025`, `Suzhou 125 2025`, `Nitto ATP Finals 2025`, `Next Gen ATP Finals 2025`, `Manila 125 2026`, `Les Sables D'Olonne 125 2026`

## Further continuation later on 2026-03-21

### Public GitHub dataset sweep

- Checked local repo for tennis-related GitHub source mirrors.
  - only relevant GitHub-derived local file was UFC-only `data/raw/github_PierceHampton.csv`
- Checked public GitHub repos that looked plausible for tennis odds backfill:
  - `Tennismylife/TML-Database`
    - contains match result CSVs only
    - no odds columns
    - useful only for sanity-checking whether a match happened, not for odds recovery
  - `DanielSzakacs/Tennis_odds`
    - contains historical tennis-data-style ATP/WTA CSV mirrors with `B365W/B365L`, `PSW/PSL`, etc.
    - loaded the 2025 ATP/WTA files and compared them against the remaining 2025 `source_absent` rows
    - result: `0` hits for the remaining 2025 absences
    - no 2026 data in that repo
  - `alienorsutinn/tennis-odds-mvp`
    - cloned sparse checkout of the `data/` folder into `tmp/alienorsutinn-tennis-odds-mvp`
    - repo contains real per-bookmaker snapshot CSVs with schema:
      - `event_id,start,player1,player2,odds1,odds2,prob1,prob2,bookmaker,sport_key,region`
    - scanned relevant windows for the remaining `source_absent` rows using the existing flexible name matcher
    - result: `0` exact hits for the current residual `source_absent` rows
    - direct grep of the repo showed coverage is limited to a small set of top-level events such as Australian Open, Indian Wells, Miami, Qatar, Dubai, Wuhan, China Open, Shanghai, and Paris Masters
    - direct text checks were `False` for the residual hard-gap tournament tokens:
      - `metz`, `finals`, `nextgen`, `jeddah`, `turin`, `quito`, `manila`, `mumbai`, `oeiras`, `suzhou`, `warsaw`, `montreux`, `guadalajara`, `saint-malo`, `antalya`, `birmingham`, `ilkley`
- conclusion:
  - public GitHub was worth checking
  - it did not produce any valid rows for the current hard gaps
  - the best-looking GitHub snapshot repo does not cover the residual tournaments

### Automated source passes after the GitHub sweep

- `.\.venv\Scripts\python.exe scripts\scrape_welcome_bet_tennis.py ...`
  - used cached `welcome_bet_post_urls.txt` (`139,699` post URLs)
  - targeted the current 2025-2026 unresolved buckets
  - result: `0` rows recovered
- `.\.venv\Scripts\python.exe scripts\scrape_signalodds_tennis.py ...`
  - targeted the same recent unresolved buckets
  - result: `0` rows recovered
- `.\.venv\Scripts\python.exe scripts\scrape_wayback_tennis.py ...`
  - targeted 2026 WTA 125 / Next Gen residual buckets
  - result: `0` rows recovered
  - important detail: the Wayback CDX API returned `0 snapshots` for every attempted OddsPortal / BetExplorer results URL in that pass
    - examples: `Antalya 125 #1 2026`, `Antalya 125 #2 2026`, `Antalya 125 #3 2026`, `Austin 125 #1 2026`, `Les Sables D'Olonne 125 2026`, `Manila 125 2026`, `Midland 125 2026`, `Mumbai 125 2026`, `Oeiras 125 Indoor #1 2026`, `Oeiras 125 Indoor #2 2026`, `Next Gen ATP Finals 2025`

### SportsbookWire scraper bug fix

- The long-running `scrape_manual_search_articles.py` pass initially wrote `data/raw/tennis/manual_search_articles_odds.csv`, but inspection showed it could create false positives.
- Root cause:
  - it was stamping recovered rows with the diagnostics row's `event_date` and `tourney_name`
  - it was not requiring the article page's actual tournament or round to match the unresolved row
- Verified false-positive examples:
  - `F. Auger-Aliassime` vs `A. de Minaur`
    - article page was clearly a `US Open` quarterfinal page
    - it had been incorrectly mapped into `Nitto ATP Finals 2025`
  - `C. Tabur` vs `V. Sachko`
    - article page was clearly `Moselle Open` `Quarterfinal`
    - the unresolved row in `Metz 2025` is `SF`, so it was the wrong match instance
  - `A. Blockx` vs `J. Engel`
    - article page was clearly `Next Gen ATP Finals presented by PIF` `Round Robin`
    - that row is already covered by the verified manual snippet; the remaining unmatched duplicate is the bogus `F` row
- Patched `scripts/scrape_manual_search_articles.py` to:
  - parse and use the article page's actual match date
  - parse and store the article page's round
  - validate tournament compatibility before writing a row
  - validate round compatibility before writing a row
  - reject page-date mismatches beyond `14` days
- Deleted the buggy generated `data/raw/tennis/manual_search_articles_odds.csv`
- Reran the patched SportsbookWire pass on the recent buckets.
  - result: `0` clean rows recovered
  - this is the correct outcome; the earlier apparent hits were not valid recoveries

### Long-running OddsPortal pass still in progress when this note was updated

- Started:
  - `.\.venv\Scripts\python.exe scripts\scrape_oddsportal_manual_pages.py`
- Purpose:
  - enrich `data/raw/tennis/oddsportal_team_events_odds.csv` from rendered OddsPortal pages for ATP Cup / United Cup / BJK Cup
- Status when this note was written:
  - process was still alive
  - `data/raw/tennis/oddsportal_team_events_odds.csv` had not changed yet
  - file remained at `987` rows, last write time `2026-03-20 18:44:33`
- If the process eventually finishes and updates the file, rerun:
  - `.\.venv\Scripts\python.exe scripts\fix_tennis_odds_matching.py`

## Final continuation update later on 2026-03-21

### What actually moved coverage after the above note

- Widened `welcome.bet` recovery from recent-only buckets to older unresolved team-event buckets:
  - `.\.venv\Scripts\python.exe scripts\scrape_welcome_bet_tennis.py united cup bjk davis atp cup olympics guadalajara`
- That pass recovered 2 valid rows in `BJK Cup Qualifiers 2024`:
  - `2024-04-12` Elina Svitolina vs Jaqueline Cristian — `1.33 / 3.25`
  - `2024-04-13` Jaqueline Cristian vs Lesia Tsurenko — `2.30 / 1.62`
- Reran `.\.venv\Scripts\python.exe scripts\fix_tennis_odds_matching.py`
  - coverage moved from `23,935 / 24,572` to `23,937 / 24,572`
  - unmatched classifications moved to:
    - `550` `source_has_no_numeric_odds`
    - `80` `source_absent`
    - `5` `upstream_match_record_issue`

### Welcome.Bet scraper provenance fix

- Found the same provenance bug pattern in `scripts/scrape_welcome_bet_tennis.py` that had existed in the SportsbookWire scraper:
  - it validated the page date but still wrote the diagnostics row date into `welcome_bet_odds.csv`
- Patched `scripts/scrape_welcome_bet_tennis.py` to:
  - write the actual parsed page date
  - write the page tournament when available
  - skip timed-out page fetches instead of crashing the whole run
  - flush each recovered row to disk immediately via `merge_rows([row])`
- Audited the existing `data/raw/tennis/welcome_bet_odds.csv` file and found `19` rows where the stored `Date` disagreed with the date encoded in the saved `source_url`.
  - normalized those `Date` values from the source URLs
  - reran the matcher afterward
  - coverage stayed exactly the same at `23,937 / 24,572`
  - conclusion: the welcome.bet source file now has better provenance without changing recovery totals

### Full broad sweeps exhausted cleanly

- After the scraper hardening above, reran:
  - `.\.venv\Scripts\python.exe scripts\scrape_welcome_bet_tennis.py`
  - `.\.venv\Scripts\python.exe scripts\scrape_manual_search_articles.py`
- Result for both full unmatched sweeps: `0` additional rows recovered
- Interpretation:
  - the obvious broad source families are now exhausted on the current queue
  - remaining gaps are not because those scrapers were failing fast or failing to checkpoint

### OddsPortal manual-pages pass outcome

- The previously long-running `scrape_oddsportal_manual_pages.py` process eventually touched `data/raw/tennis/oddsportal_team_events_odds.csv`, but produced no net row-count increase.
  - file remained `987` rows
  - no downstream coverage change after the next matcher rerun

### Local-source audit after all of the above

- Ran `.\.venv\Scripts\python.exe scripts\analyze_remaining_candidates.py`
- Important conclusions:
  - `OddsPortal -> Atp Cup / United Cup / BJK Cup Qualifiers / BJK Cup Playoffs / BJK Cup Finals`
    - `0` hidden local candidate matches for the remaining unmatched rows
  - `BetExplorer WTA125 -> 2025 missing`
    - found `8` pair/date candidate rows
    - but every one of those rows has `NaN` odds in `betexplorer_wta125_odds.csv`
    - they are not matcher misses
    - they are genuine `source_has_no_numeric_odds` cases

### Current authoritative baseline at end of this continuation

- Coverage: `23,937 / 24,572`
- Remaining unmatched:
  - `550` `source_has_no_numeric_odds`
  - `80` `source_absent`
  - `5` `upstream_match_record_issue`
- Output files:
  - `data/processed/tennis/matches_with_odds.csv`
  - `data/processed/tennis/unmatched_diagnostics.csv`
  - `data/processed/tennis/odds_matching_report.txt`

## Later continuation update on 2026-03-21 after direct article recovery

### Direct page families that actually worked

- `Bleacher Nation / Data Skrive`
  - verified and appended `2024-10-28` Antonia Ruzic vs Laura Samson
  - odds: `1.526316 / 2.45`
- `Khel Now`
  - verified and appended `2024-11-08` Coco Gauff vs Aryna Sabalenka
  - odds: `3.0 / 1.47619`
  - important: this matched the local `Riyadh Finals 2024` semifinal row dated `2024-11-04` with `odds_date_diff = 4`, confirming again that some season-ending event rows use tournament-start dates instead of actual match dates
- `TennisTonic`
  - verified and appended 11 additional rows with exact odds from article bodies:
    - `2022-09-12` Eugenie Bouchard vs Joanne Zuger — `1.93 / 1.87`
    - `2023-01-30` Xin Yu Wang vs Joanne Zuger — `1.23 / 4.2`
    - `2023-06-12` Liudmila Samsonova vs Lena Rueffer / Lena Papadakis — `1.047 / 10.25`
    - `2023-07-04` Kaja Juvan vs Margarita Gasparyan / Margarita Betova — `1.076 / 8.1`
    - `2023-07-18` Yulia Putintseva vs Kateryna Bondarenko / Kateryna Volodko — `1.133 / 5.85`
    - `2023-08-28` Yuriko Miyazaki vs Margarita Gasparyan / Margarita Betova — `1.33 / 3.32`
    - `2023-10-09` Qinwen Zheng vs Kateryna Bondarenko / Kateryna Volodko — `1.08 / 7.9`
    - `2023-11-05` Iga Swiatek vs Aryna Sabalenka — `1.57 / 2.41`
    - `2024-07-22` Laura Samson vs Tara Wurth — `1.88 / 1.93`
    - `2024-07-23` Laura Samson vs Katerina Siniakova — `5.9 / 1.134`
    - `2024-10-28` Han Shi vs Margarita Gasparyan / Margarita Betova — `1.22 / 4.3`

### Coverage change from the article batch

- `manual_search_articles_odds.csv` now contains `13` verified rows
- after rerunning `.\.venv\Scripts\python.exe scripts\fix_tennis_odds_matching.py`:
  - coverage moved from `23,937 / 24,572` to `23,950 / 24,572`
  - remaining unmatched moved to:
    - `550` `source_has_no_numeric_odds`
    - `67` `source_absent`
    - `5` `upstream_match_record_issue`

### Important remaining shape after the batch

- Remaining WTA `source_absent` one-offs are now much smaller:
  - `2023-06-26` Bad Homburg — Leylah Fernandez vs Lena Rueffer
  - `2023-11-07` BJK Cup Finals — Yulia Putintseva vs Tamara Zidansek
- Most remaining WTA `source_absent` rows are now clustered `125` events in 2025-2026
- Biggest non-WTA bucket still blocking `source_absent` is `Metz 2025` with `10` rows

## Later continuation update on 2026-03-21 after targeted snippet/article recovery

### Verified rows appended in this batch

- Added 8 exact snippet rows to `data/raw/tennis/manual_search_snippets_odds.csv`:
  - `2025-07-28` Sofia Costoulas vs Martyna Kubka — PokerStars snippet `1.44 / 2.50`
  - `2025-09-01` Francesca Jones vs Victoria Rodriguez — PokerStars snippet `1.05 / 7.50`
  - `2025-09-01` Carson Branstine vs Anouk Koevermans — Betfair snippet `1.57 / 2.25`
  - `2025-09-01` Darya Astakhova vs Nuria Brancaccio — Betfair snippet `2.50 / 1.50`
  - `2025-09-29` Tiphanie Lemaitre vs Leonie Kung — PokerStars snippet `1.57 / 2.20`
  - `2025-09-29` Carol Young Suh Lee vs Kaitlin Quevedo — PokerStars snippet `1.83 / 1.83`
  - `2025-09-28` Linda Fruhvirtova vs Anastasia Zakharova — PokerStars snippet `1.87 / 1.80`
  - `2025-10-20` Alicia Herrero Linana vs Marianne Angel Gonzalez — PokerStars snippet `1.01 / 10.00`
- Added 1 verified SportyTrader article row to `data/raw/tennis/manual_search_articles_odds.csv`:
  - `2025-06-09` Viktorija Golubic vs Lucrezia Stefanini — `1.41 / 3.2`

### Matcher rerun result after this batch

- Reran `.\.venv\Scripts\python.exe scripts\fix_tennis_odds_matching.py`
- Coverage moved from `23,961 / 24,572` to `23,970 / 24,572`
- `source_absent` dropped from `56` to `47`
- `source_has_no_numeric_odds` stayed `550`
- `upstream_match_record_issue` stayed `5`

### Confirmed matched rows from this batch

- `Viktorija Golubic` vs `Lucrezia Stefanini`
- `Sofia Costoulas` vs `Martyna Kubka`
- `Francesca Jones` vs `Victoria Rodriguez`
- `Carson Branstine` vs `Anouk Koevermans`
- `Darya Astakhova` vs `Nuria Brancaccio`
- `Tiphanie Lemaitre` vs `Leonie Kung`
- `Kaitlin Quevedo` vs `Carol Young Suh Lee`
- `Linda Fruhvirtova` vs `Anastasia Zakharova`
  - matched with `odds_date_diff = 1` because the snippet page was dated `2025-09-28` and the local match row is `2025-09-29`
- `Alicia Herrero Linana` vs `Marianne Angel Gonzalez`

### Current residual `source_absent` shape

- Biggest buckets:
  - `10` `Metz 2025`
  - `3` `Atp Cup 2022`
  - `3` `Davis Cup WG2 R1: ESA vs IRL 2023`
  - `2` each in `Davis Cup WG2 R1: BAR vs IRL 2022`, `Nitto ATP Finals 2025`, `Next Gen ATP Finals 2025`, `Manila 125 2026`
- Remaining WTA one-offs:
  - `2025-04-14` Oeiras 125 — Francesca Jones vs Noma Noha Akugue
  - `2025-04-28` Saint Malo 125 — Fiona Ferro vs Noma Noha Akugue
  - `2025-09-08` Ljubljana 125 — Sinja Kraus vs Silvia Ambrosio
  - `2025-09-29` Suzhou 125 — Leolia Jeanjean vs Maddison Inglis
  - `2025-12-01` Quito 125 — Maja Chwalinska vs Tania Varela
  - `2026-01-26` Manila 125 — Mai Hontama vs Elizabeth Abarquez
  - `2026-01-26` Manila 125 — Tatiana Prozorova vs Kaye Ann Emana
  - `2026-02-02` Mumbai 125 — Misaki Matsuda vs Ankita Raina
  - `2026-02-16` Les Sables D'Olonne 125 — Mona Barthel vs Tiphanie Lemaitre
  - `2026-02-16` Midland 125 — Katarina Jokic vs Elizabeth Jones
  - `2026-02-24` Antalya 125 #1 — Moyuka Uchijima vs Katarzyna Kawa
  - `2026-03-03` Antalya 125 #2 — Leyre Romero Gormaz vs Carole Monnet

## Later continuation update on 2026-03-21 after two more 2026 recoveries

### Verified rows appended in this batch

- Added 2 rows to `data/raw/tennis/manual_search_snippets_odds.csv`:
  - `2026-02-24` Moyuka Uchijima vs Katarzyna Kawa
    - source: Betfair Sportsbook page
    - page listed `4/11` and `17/10`, stored as exact decimal equivalents `1.36 / 2.70`
  - `2026-03-03` Leyre Romero Gormaz vs Carole Monnet
    - source: Betfair Exchange page
    - stored the leading visible back prices `1.54 / 2.56`

### Matcher rerun result after this batch

- Reran `.\.venv\Scripts\python.exe scripts\fix_tennis_odds_matching.py`
- Coverage moved from `23,970 / 24,572` to `23,972 / 24,572`
- `source_absent` dropped from `47` to `45`
- `source_has_no_numeric_odds` stayed `550`
- `upstream_match_record_issue` stayed `5`
- `2026` coverage moved from `1575 / 1583` earlier in the turn to `1577 / 1583`

### Confirmed matched rows from this batch

- `Moyuka Uchijima` vs `Katarzyna Kawa`
- `Leyre Romero Gormaz` vs `Carole Monnet`

### Rechecked broad source families after the manual adds

- Ran `.\.venv\Scripts\python.exe scripts\analyze_remaining_candidates.py`
  - still no hidden local-candidate recoveries for the current residual buckets
  - the earlier WTA 125 local candidate findings remain `source_has_no_numeric_odds`, not matcher misses
- Ran targeted `welcome.bet` pass on the remaining hard buckets:
  - `.\.venv\Scripts\python.exe scripts\scrape_welcome_bet_tennis.py "atp cup" davis lyon hamburg "bad homburg" "bjk cup finals" oeiras "saint malo" ljubljana suzhou quito metz nitto "next gen" manila mumbai "les sables" midland`
  - result: `0` rows recovered
- Targeted SportsbookWire and SignalOdds passes on the same buckets hit the command timeout ceiling and wrote no new rows before timing out.
  - `manual_search_articles_odds.csv` remained at `24` rows
  - `signalodds_odds.csv` remained at `23` rows

### Current authoritative baseline

- Coverage: `23,972 / 24,572`
- Remaining unmatched:
  - `550` `source_has_no_numeric_odds`
  - `45` `source_absent`
  - `5` `upstream_match_record_issue`
- Remaining `source_absent` WTA rows:
  - `2023-06-26` Bad Homburg — Leylah Fernandez vs Lena Rueffer
  - `2023-11-07` BJK Cup Finals — Yulia Putintseva vs Tamara Zidansek
  - `2025-04-14` Oeiras 125 — Francesca Jones vs Noma Noha Akugue
  - `2025-04-28` Saint Malo 125 — Fiona Ferro vs Noma Noha Akugue
  - `2025-09-08` Ljubljana 125 — Sinja Kraus vs Silvia Ambrosio
  - `2025-09-29` Suzhou 125 — Leolia Jeanjean vs Maddison Inglis
  - `2025-12-01` Quito 125 — Maja Chwalinska vs Tania Varela
  - `2026-01-26` Manila 125 — Mai Hontama vs Elizabeth Abarquez
  - `2026-01-26` Manila 125 — Tatiana Prozorova vs Kaye Ann Emana
  - `2026-02-02` Mumbai 125 — Misaki Matsuda vs Ankita Raina
  - `2026-02-16` Les Sables D'Olonne 125 — Mona Barthel vs Tiphanie Lemaitre
  - `2026-02-16` Midland 125 — Katarina Jokic vs Elizabeth Jones

## Later continuation update on 2026-03-21 after Midland 125 and Suzhou 125 recovery

### Verified rows appended in this batch

- Added 1 row to `data/raw/tennis/manual_search_snippets_odds.csv`:
  - `2026-02-16` Katarina Jokic vs Elizabeth Jones
    - source: PokerStars search result snippet
    - exact snippet odds stored as `1.85 / 1.85`
- Added 1 row to `data/raw/tennis/manual_search_articles_odds.csv`:
  - `2025-09-29` Leolia Jeanjean vs Maddison Inglis
    - source: SportyTrader odds page
    - page listed best odds `Leolia Jeanjean 1.99` and `Maddison Inglis 1.84`

### Matcher rerun result after this batch

- Reran `.\.venv\Scripts\python.exe scripts\fix_tennis_odds_matching.py`
- Coverage moved from `23,972 / 24,572` to `23,974 / 24,572`
- `source_absent` dropped from `45` to `43`
- `source_has_no_numeric_odds` stayed `550`
- `upstream_match_record_issue` stayed `5`
- `2026` coverage moved from `1577 / 1583` to `1578 / 1583`

### Confirmed matched rows from this batch

- `2025-09-29` Leolia Jeanjean vs Maddison Inglis
  - present in `matches_with_odds.csv` with `b365_a = 1.99`, `b365_b = 1.84`, `odds_date_diff = 0.0`
- `2026-02-16` Katarina Jokic vs Elizabeth Jones
  - present in `matches_with_odds.csv` with `b365_a = 1.85`, `b365_b = 1.85`, `odds_date_diff = 0.0`

### Current authoritative baseline

- Coverage: `23,974 / 24,572`
- Remaining unmatched:
  - `550` `source_has_no_numeric_odds`
  - `43` `source_absent`
  - `5` `upstream_match_record_issue`
- Largest `source_absent` buckets:
  - `10` `Metz 2025`
  - `3` `Atp Cup 2022`
  - `3` `Davis Cup WG2 R1: ESA vs IRL 2023`
  - `2` each in `Nitto ATP Finals 2025`, `Next Gen ATP Finals 2025`, `Manila 125 2026`, `Davis Cup WG2 R1: BAR vs IRL 2022`

### Remaining `source_absent` WTA rows

- `2023-06-26` Bad Homburg — Leylah Fernandez vs Lena Rueffer
- `2023-11-07` BJK Cup Finals — Yulia Putintseva vs Tamara Zidansek
- `2025-04-14` Oeiras 125 — Francesca Jones vs Noma Noha Akugue
- `2025-04-28` Saint Malo 125 — Fiona Ferro vs Noma Noha Akugue
- `2025-09-08` Ljubljana 125 — Sinja Kraus vs Silvia Ambrosio
- `2025-12-01` Quito 125 — Maja Chwalinska vs Tania Varela
- `2026-01-26` Manila 125 — Mai Hontama vs Elizabeth Abarquez
- `2026-01-26` Manila 125 — Tatiana Prozorova vs Kaye Ann Emana
- `2026-02-02` Mumbai 125 — Misaki Matsuda vs Ankita Raina
- `2026-02-16` Les Sables D'Olonne 125 — Mona Barthel vs Tiphanie Lemaitre

## Later continuation update on 2026-03-21 after Sofascore API recovery pass

### New source path used

- Used public Sofascore event pages and archived odds endpoint:
  - match page JSON gave exact `event.id`
  - archived odds endpoint: `https://api.sofascore.com/api/v1/event/{event_id}/odds/1/all`
- Only appended rows when:
  - player names matched the exact event
  - the event date was consistent with the bucketed match date
  - the API returned a real `Full time` home/away market with numeric fractional odds

### Verified rows appended in the first Sofascore batch

- Added 5 rows to `data/raw/tennis/manual_search_articles_odds.csv`:
  - `2022-01-03` Hubert Hurkacz vs Aleksandre Metreveli
  - `2023-02-02` Aleksandre Bakshi vs Juan Carlos Prado Angelo
  - `2023-11-10` Yulia Putintseva vs Tamara Zidansek
  - `2024-02-02` Kris Van Wyk vs Linh Giang Trinh
  - `2025-04-17` Francesca Jones vs Noma Noha Akugue

### Matcher rerun result after the first Sofascore batch

- Reran `.\.venv\Scripts\python.exe scripts\fix_tennis_odds_matching.py`
- Coverage moved from `23,974 / 24,572` to `23,979 / 24,572`
- `source_absent` dropped from `43` to `38`
- All 5 appended rows matched
- Confirmed examples in `matches_with_odds.csv`:
  - `Yulia Putintseva` vs `Tamara Zidansek` matched with `odds_date_diff = 3.0`
  - `Francesca Jones` vs `Noma Noha Akugue` matched with `odds_date_diff = 3.0`

### Verified rows appended in the second Sofascore batch

- Added 4 more rows to `data/raw/tennis/manual_search_articles_odds.csv`:
  - `2022-03-04` Coleman Wong vs Delmas Ntcha
  - `2023-02-03` Arklon Huertas Del Pino vs Osgar Ohoisin
  - `2023-09-16` Marcelo Arevalo vs Osgar Ohoisin
  - `2024-02-03` Sebastian Ofner vs Osgar Ohoisin

### Matcher rerun result after the second Sofascore batch

- Reran `.\.venv\Scripts\python.exe scripts\fix_tennis_odds_matching.py`
- Coverage moved from `23,979 / 24,572` to `23,983 / 24,572`
- `source_absent` dropped from `38` to `34`
- All 4 appended rows matched

### Current authoritative baseline

- Coverage: `23,983 / 24,572`
- Remaining unmatched:
  - `550` `source_has_no_numeric_odds`
  - `34` `source_absent`
  - `5` `upstream_match_record_issue`
- Coverage by year after the latest rerun:
  - `2022: 5108 / 5306`
  - `2023: 5352 / 5598`
  - `2024: 5613 / 5723`
  - `2025: 6332 / 6362`
  - `2026: 1578 / 1583`

### Remaining `source_absent` WTA rows after Sofascore audit

- `2023-06-26` Bad Homburg — Leylah Fernandez vs Lena Rueffer
- `2025-04-28` Saint Malo 125 — Fiona Ferro vs Noma Noha Akugue
- `2025-09-08` Ljubljana 125 — Sinja Kraus vs Silvia Ambrosio
- `2025-12-01` Quito 125 — Maja Chwalinska vs Tania Varela
- `2026-01-26` Manila 125 — Mai Hontama vs Elizabeth Abarquez
- `2026-01-26` Manila 125 — Tatiana Prozorova vs Kaye Ann Emana
- `2026-02-02` Mumbai 125 — Misaki Matsuda vs Ankita Raina
- `2026-02-16` Les Sables D'Olonne 125 — Mona Barthel vs Tiphanie Lemaitre

### Sofascore audit findings for the remaining WTA rows

- Exact event found but archived odds endpoint returned `404` / no market:
  - `2025-04-29` Fiona Ferro vs Noma Noha Akugue at Saint Malo, France
  - `2025-09-09` Sinja Kraus vs Silvia Ambrosio at Ljubljana, Slovenia
  - `2025-12-02` Tania Varela-Alvarado vs Maja Chwalinska at Quito, Ecuador
  - `2026-01-26` Elizabeth Abarquez vs Mai Hontama at Manila, Philippines
  - `2026-01-27` Kaye Ann Emana vs Tatiana Prozorova at Manila, Philippines
  - `2026-02-02` Ankita Raina vs Misaki Matsuda at Mumbai, India
  - `2026-02-16` Tiphanie Lemaitre vs Mona Barthel at Les Sables d Olonne, France
- `Leylah Fernandez` vs `Lena Rueffer` looks like an upstream record issue, not a missing odds source:
  - Sofascore Bad Homburg on `2023-06-26` shows `Lena Papadakis vs Leylah Fernandez` and `Alycia Parks vs Leylah Fernandez`
  - no `Leylah Fernandez vs Lena Rueffer` event was found in the nearby Bad Homburg history pages

### Sofascore audit findings for the remaining ATP `source_absent` rows

- Exact event found but archived odds endpoint returned no market:
  - `2022-01-01` Aristotelis Thanos vs Hubert Hurkacz
  - `2022-01-02` Jannik Sinner vs Max Purcell
  - `2022-03-05` Benjamin Bonzi vs Antonio Cayetano March
  - `2022-09-16` Illya Beloborodko vs Marton Fucsovics
  - `2022-09-16` Darian King vs O. O'Hoisin
  - `2022-09-17` Kaipo Marshall vs O. O'Hoisin
  - `2023-09-17` Jose Flores vs Freddy Murray
  - `2023-09-17` Diego Duran vs O. O'Hoisin
  - `2024-07-16` Luciano Darderi vs Nick Hardt
  - `2024-09-14` Faris Zakaria vs Mate Valkusz
  - `2024-09-14` Martyn Pawelski vs Min-Kyu Song
- Rows that now look upstream-bad rather than genuinely source-missing:
  - `Metz 2025`:
    - no exact Metz main-draw candidates were found for most of the bucket
    - the one real `Sachko vs Tabur` Metz match found was on `2025-11-06` and had the winner reversed relative to the dataset row
    - a `2025-11-02` `Sachko vs Tabur` qualifying final also exists and is a false positive for the main-draw row
  - `Nitto ATP Finals 2025`:
    - Sofascore Tour Finals group schedule shows `Ben Shelton` in the `Bjorn Borg` group against `Alexander Zverev`, `Felix Auger-Aliassime`, and `Jannik Sinner`
    - `Felix Auger-Aliassime` also appears in that group and later against `Carlos Alcaraz` in the semifinal
    - no `Ben Shelton vs Carlos Alcaraz` or `Felix Auger-Aliassime vs Alex de Minaur` round-robin events were found
  - `Next Gen ATP Finals 2025`:
    - `Justin Engel vs Alexander Blockx` exists on `2025-12-17` as a `Red Group` match, not a final
    - no `Alexander Blockx vs Rafael Jodar` event was found

## Later continuation update on 2026-03-21 after widened archive and live-site fallback passes

### Wayback pass expansion

- Patched `scripts/scrape_wayback_tennis.py` to widen target coverage for:
  - `Atp Cup`
  - generic `Davis Cup` buckets via year-aware URL templates
  - `Metz`
  - `Nitto ATP Finals`
  - `Bad Homburg`
  - `Saint Malo 125`
  - `Ljubljana 125`
  - `Quito 125`
  - extra `Next Gen ATP Finals` URL variants
- Added `{year}` URL formatting support in the archive runner.
- Verified syntax with:
  - `.\.venv\Scripts\python.exe -m py_compile scripts\scrape_wayback_tennis.py`

### Wayback run result

- Ran `.\.venv\Scripts\python.exe scripts\scrape_wayback_tennis.py`
- Result:
  - `0` recovered rows
  - `0` appended rows
  - no `wayback_tennis_odds.csv` data was produced for the remaining hard buckets
- Important negative finding:
  - the archive path has now been checked not just for `Next Gen ATP Finals` and the WTA 125 defaults, but also for the remaining ATP Cup, Davis Cup, Metz, ATP Finals, Bad Homburg, Saint Malo, Ljubljana, Quito, Manila, Mumbai, and Les Sables buckets
  - it produced no numeric recovery rows

### Live BetExplorer targeted reruns after the archive pass

- Ran WTA 125 targeted live scrape:
  - `.\.venv\Scripts\python.exe scripts\scrape_be_targeted.py wta125 saint-malo ljubljana quito manila mumbai les-sables`
  - refreshed `betexplorer_wta125_odds.csv`
  - scraper found additional rows in some nearby seasons (`Saint Malo 2022/2024`, `Ljubljana 2023/2024`, `Mumbai 2024/2025`)
  - none of the remaining hard-match player pairs appeared in the refreshed file
- Ran Davis Cup live scrape:
  - `.\.venv\Scripts\python.exe scripts\scrape_be_targeted.py davis_wg`
  - refreshed `betexplorer_team_events_odds.csv` with a large number of Davis Cup rows
  - reran matcher after the refresh
  - coverage did not move
- Ran Davis Cup qualifiers live scrape:
  - `.\.venv\Scripts\python.exe scripts\scrape_be_targeted.py davis_quals`
  - returned `0` rows for `2022-2025`
- Ran Next Gen live scrape:
  - `.\.venv\Scripts\python.exe scripts\scrape_be_targeted.py nextgen`
  - pulled `2025` Jeddah rows plus older seasons into `betexplorer_team_events_odds.csv`
  - reran matcher after the refresh
  - coverage did not move

### Current hard-stop baseline after these negative passes

- Coverage remains `23,983 / 24,572`
- Remaining unmatched remains:
  - `550` `source_has_no_numeric_odds`
  - `34` `source_absent`
  - `5` `upstream_match_record_issue`
- Interpretation:
  - the remaining `34` are no longer low-effort misses from the archive path or the standard BetExplorer target pages
  - the easy structured source families are exhausted for the current residue
