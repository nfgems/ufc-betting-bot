# Tennis Odds Handoff - 2026-03-20

## Current state

- Repo: `C:\Users\Evan\betting-bot\ufc-betting-bot`
- Latest full matcher run: `python scripts\fix_tennis_odds_matching.py`
- Current coverage: `23,883 / 24,572` rows with odds
- Coverage percent: `97.2%`
- Unmatched total: `689`
- Unmatched classification:
  - `410` `source_has_no_numeric_odds`
  - `274` `source_absent`
  - `5` `upstream_match_record_issue`

## Key outputs

- Processed matches with odds:
  - `data/processed/tennis/matches_with_odds.csv`
- Final report:
  - `data/processed/tennis/odds_matching_report.txt`
- Exact unmatched diagnostics:
  - `data/processed/tennis/unmatched_diagnostics.csv`

## Important scripts

- Main matcher:
  - `scripts/fix_tennis_odds_matching.py`
- Manual web source scripts:
  - `scripts/scrape_signalodds_tennis.py`
  - `scripts/scrape_manual_search_articles.py`
  - `scripts/scrape_wayback_tennis.py`
- Existing source scrapers already in use:
  - `scripts/scrape_be_targeted.py`
  - `scripts/scrape_oddsportal_wta125.py`
  - `scripts/scrape_oddsportal_team_events.py`
  - `scripts/scrape_flashscore_odds.py`

## Raw source files currently loaded by the matcher

- `data/raw/tennis/betexplorer_team_events_odds.csv`
- `data/raw/tennis/oddsportal_team_events_odds.csv`
- `data/raw/tennis/betexplorer_wta125_odds.csv`
- `data/raw/tennis/oddsportal_wta125_odds.csv`
- `data/raw/tennis/flashscore_odds.csv`
- `data/raw/tennis/betexplorer_extra_odds.csv`
- `data/raw/tennis/welcome_bet_odds.csv`
- `data/raw/tennis/manual_search_snippets_odds.csv`
- `data/raw/tennis/manual_search_articles_odds.csv`
- `data/raw/tennis/signalodds_odds.csv`
- `data/raw/tennis/wayback_tennis_odds.csv`
- `data/raw/tennis/masterscup_odds.csv`

## What changed in this session

- Improved `scripts/scrape_signalodds_tennis.py`:
  - added Yahoo-based URL discovery when direct `signalodds` slug guessing fails
  - this recovered the final missing `Miami 2026` row: `Camila Osorio vs Katerina Siniakova`
- Improved `scripts/scrape_manual_search_articles.py`:
  - switched to Yahoo search parsing
  - targeted SportsbookWire article discovery
- Added many new auditable manual rows to:
  - `data/raw/tennis/manual_search_snippets_odds.csv`
- `manual_search_snippets_odds.csv` is now `51` rows
- `signalodds_odds.csv` is now `23` rows
- Coverage moved from `23,859` to `23,883` during this session

## Manual web sources that actually worked

- `SignalOdds`
- `Matchstat`
- `SportyTrader`
- `Smarkets`
- `Betfair`
- `PokerStars`
- `nb-bet`
- `Dexwin`

These sources were used only when they exposed explicit numeric pre-match odds on the page itself or in a search result snippet. No rows were fabricated.

## Sources that were mostly dead ends or blocked

- `Sportsbet`: blocked by Akamai / 403 from shell
- `SportyTrader` list pages: useful sometimes, but many generic event pages have no match rows
- `365scores`: JS shell, no usable odds in raw HTML here
- `SportyTrader` / `OddsChecker` / some bookmakers: Cloudflare or anti-bot on some pages
- `TonyBet`: country blocked
- `Bovada`: many old event pages rolled over to "no upcoming odds"
- `Wayback`: targeted attempts mostly returned no snapshots or archive-empty

## Important caveat

The `Name Mismatch Details` section inside `data/processed/tennis/odds_matching_report.txt` is noisy and includes pass-1 misses that were later recovered by extra sources. For final unresolved work, trust `data/processed/tennis/unmatched_diagnostics.csv`, not that section.

## What is still missing

### 1. `source_has_no_numeric_odds` rows

These already have a source candidate, but the source row/page does not contain usable numeric odds. These are lower priority unless a new source family is found.

Top buckets:

- `United Cup 2023`: `10`
- `BJK Cup Qualifiers 2024`: `9`
- `United Cup 2024`: `7`
- `BJK Cup Playoffs 2023`: `6`

Large older team-event buckets also remain here because pages were found but only names/results were available.

### 2. `source_absent` rows

These are the real remaining recovery targets.

Top buckets:

- `Metz 2025`: `9`
- `Atp Cup 2022`: `3`
- `Davis Cup WG2 PO: INA vs VIE 2023`: `3`
- `Davis Cup WG2 R1: ESA vs IRL 2023`: `3`
- `Davis Cup WG2 PO: VIE vs RSA 2024`: `3`
- `Next Gen ATP Finals 2025`: `3`

### 3. `upstream_match_record_issue`

All `5` are malformed `Metz 2025` rows where winner/loser/player fields are broken:

- `F. Passaro vs F. Passaro`
- `J. Choinski vs J. Choinski`
- `K. Jacquet vs K. Jacquet`
- `L. Van Assche vs L. Van Assche`
- `V. Sachko vs V. Sachko`

These need upstream match-data cleanup or a special-case repair layer. They are not normal source recovery misses.

## Recent `source_absent` rows worth targeting next

These are the most promising because they are 2024+ and many similar rows were recoverable from bookmaker snippets or match pages.

### ATP / WTA 2024

- `2024-02-02` `Davis Cup WG2 PO: VIE vs RSA 2024`
  - `Kris Van Wyk vs Linh Giang Trinh`
  - `Nam Hoang Ly vs Kris Van Wyk`
  - `Nam Hoang Ly vs Philip Henning`
- `2024-02-03` `Davis Cup QLS R1: TPE vs FRA 2024`
  - `Adrian Mannarino vs Tung Lin Wu`
- `2024-02-03` `Davis Cup WG1 PO: IRL vs AUT 2024`
  - `Sebastian Ofner vs Osgar Ohoisin`
- `2024-02-03` `Davis Cup WG2 PO: TOG vs INA 2024`
  - `Fitriadi M Rifqi vs Komlavi Loglo`
  - `Fitriadi M Rifqi vs Thomas Yaka Kofi Setodji`
- `2024-07-15` `Hamburg 2024`
  - `Luciano Darderi vs Nick Hardt`
- `2024-07-22` `Prague 2024`
  - `Laura Samson vs Katerina Siniakova`
  - `Laura Samson vs Tara Wurth`
- `2024-09-13` `Davis Cup WG1 R1: EGY vs HUN 2024`
  - `Mate Valkusz vs Faris Zakaryia`
- `2024-09-13` `Davis Cup WG1 R1: POL vs KOR 2024`
  - `Martyn Pawelski vs Min Kyu Song`
- `2024-09-14` `Davis Cup WG1 R1: TPE vs BIH 2024`
  - `Tung Lin Wu vs Damir Dzumhur`
  - `Tung Lin Wu vs Mirza Basic`
- `2024-10-28` `Hong Kong 2024`
  - `Han Shi vs Margarita Gasparyan`
- `2024-10-28` `Merida 2024`
  - `Antonia Ruzic vs Laura Samson`

### WTA 125 2025

- `2025-03-24` `Puerto Vallarta 125 2025`
  - `Bernarda Pera vs Fernanda Contreras`
- `2025-03-31` `La Bisbal D'Empordá 125 2025`
  - `Alizé Cornet vs Irene Burillo`
- `2025-04-14` `Oeiras 125 2025`
  - `Francesca Jones vs Noma Noha Akugue`
  - `Victoria Jimenez Kasintseva vs Yue Yuan`
- `2025-04-28` `Saint Malo 125 2025`
  - `Fiona Ferro vs Noma Noha Akugue`
- `2025-04-28` `Vic 125 2025`
  - `Renata Zarazua vs Irene Burillo`
- `2025-05-12` `Parma 125 2025`
  - `Renata Zarazua vs Patricia Maria Tig`
- `2025-06-02` `Birmingham 125 2025`
  - `Greet Minnen vs Kimberly Birrell`
- `2025-06-09` `Ilkley 125 2025`
  - `Emily Appleton vs Simona Waltert`
  - `Viktorija Golubic vs Lucrezia Stefanini`
- `2025-07-07` `Bastad 125 2025`
  - `Elisabetta Cocciaretto vs Irene Burillo`
- `2025-07-14` `Iasi 2025`
  - `Simona Waltert vs Irene Burillo`
- `2025-07-28` `Warsaw 125 2025`
  - `Sofia Costoulas vs Martyna Kubka`
  - `Xinyu Gao vs Tamara Korpatsch`
- `2025-09-01` `Guadalajara 125 2025`
  - `Alexandra Eala vs Varvara Lepchenko`
  - `Francesca Jones vs Victoria Rodriguez`
- `2025-09-01` `Montreux 125 2025`
  - `Carson Branstine vs Anouk Koevermans`
  - `Darya Astakhova vs Nuria Brancaccio`
- `2025-09-08` `Ljubljana 125 2025`
  - `Sinja Kraus vs Silvia Ambrosio`
- `2025-09-08` `San Sebastian 125 2025`
  - `Darja Semenistaja vs Irene Burillo`
- `2025-09-29` `Cosenza 125 2025`
  - `Tiphanie Lemaitre vs Leonie Kung`
- `2025-09-29` `Samsun 125 2025`
  - `Kaitlin Quevedo vs Carol Young Suh Lee`
- `2025-09-29` `Suzhou 125 2025`
  - `Leolia Jeanjean vs Maddison Inglis`
  - `Linda Fruhvirtova vs Anastasia Zakharova`
- `2025-10-05` `Mallorca 125 2025`
  - `Guiomar Maristany Zuleta De Reales vs Irene Burillo`
- `2025-10-13` `Rio De Janeiro 125 2025`
  - `Sinja Kraus vs Julia Konishi Camargo Silva`
- `2025-10-20` `Florianopolis 125 2025`
  - `Irene Burillo vs Maribella Zamarripa`
  - `Leyre Romero Gormaz vs Julia Konishi Camargo Silva`
- `2025-10-20` `Queretaro 125 2025`
  - `Alicia Herrero Linana vs Marianne Angel Gonzalez`
- `2025-11-03` `Tucuman 125 2025`
  - `Carla Markus vs Laura Pigossi`
  - `Jazmin Ortenzi vs Candela Vazquez`
- `2025-11-24` `Buenos Aires 125 2025`
  - `Miriam Bulgaru vs Irene Burillo`
- `2025-12-01` `Angers 125 2025`
  - `Amandine Monnot vs Yuriko Miyazaki`
- `2025-12-01` `Quito 125 2025`
  - `Maja Chwalinska vs Tania Varela`
  - `Tatjana Maria vs Irene Burillo`

### ATP 2025

- `2025-11-02` `Metz 2025`
  - `A. Cazaux vs K. Jacquet`
  - `C. Tabur vs A. Rinderknech`
  - `C. Tabur vs D. Altmaier`
  - `G. Mpetshi Perricard vs A. Blockx`
  - `T. Atmane vs A. Cazaux`
  - `T. Atmane vs C. Norrie`
  - `T. Atmane vs L. Sonego`
  - `V. Sachko vs A. Vukic`
  - `V. Sachko vs T. Atmane`
- `2025-11-09` `Nitto ATP Finals 2025`
  - `B. Shelton vs C. Alcaraz`
  - `F. Auger-Aliassime vs A. de Minaur`
- `2025-12-17` `Next Gen ATP Finals 2025`
  - `A. Blockx vs R. Jodar`
  - `J. Engel vs A. Blockx`
  - `J. Engel vs A. Blockx` final

### 2026 rows still missing

- `2026-01-26` `Manila 125 2026`
  - `Mai Hontama vs Elizabeth Abarquez`
  - `Tatiana Prozorova vs Kaye Ann Emana`
- `2026-02-02` `Mumbai 125 2026`
  - `Misaki Matsuda vs Ankita Raina`
- `2026-02-09` `IEB+ Argentina Open 2026`
  - `T. Barrios Vera vs T. Seyboth Wild`
- `2026-02-09` `Oeiras 125 Indoor #1 2026`
  - `Lucrezia Stefanini vs Angelina Voloshchuk`
- `2026-02-16` `Les Sables D'Olonne 125 2026`
  - `Carol Young Suh Lee vs Anna-Lena Friedsam`
  - `Mona Barthel vs Tiphanie Lemaitre`
- `2026-02-16` `Midland 125 2026`
  - `Katarina Jokic vs Elizabeth Jones`
- `2026-02-16` `Oeiras 125 Indoor #2 2026`
  - `Viktorija Golubic vs Ana Filipa Santos`
- `2026-02-24` `Antalya 125 #1 2026`
  - `Anastasia Zolotareva vs Julia Riera`
  - `Moyuka Uchijima vs Katarzyna Kawa`
- `2026-03-03` `Antalya 125 #2 2026`
  - `Leyre Romero Gormaz vs Carole Monnet`
- `2026-03-09` `Austin 125 #1 2026`
  - `Linda Fruhvirtova vs Beatriz Haddad Maia`
  - `Oksana Selekhmeteva vs Yue Yuan`
- `2026-03-10` `Antalya 125 #3 2026`
  - `Ekaterine Gorgodze vs Arantxa Rus`

## What worked best for recent rows

### Search/result-snippet pattern

The most productive pattern was:

1. Search exact player pair plus tournament and year.
2. Favor source families that expose prices in search snippets or static HTML:
   - `PokerStars`
   - `Betfair`
   - `Smarkets`
   - `SportyTrader`
   - `Matchstat`
   - `SignalOdds`
   - `nb-bet`
   - `Dexwin`
3. Only add rows when both sides of the moneyline are explicitly visible.
4. Preserve the exact source URL and a short provenance note in `manual_search_snippets_odds.csv`.

### Good examples already in `manual_search_snippets_odds.csv`

- `Simona Waltert vs Dominika Salkova` from `Smarkets`
- `Greet Minnen vs Rebeka Masarova` from `SportyTrader`
- `Solana Sierra vs Carla Markus` from `Matchstat`
- `Nishesh Basavareddy vs Dino Prizmic` from `Dexwin`
- many `PokerStars` and `Betfair` rows for Oeiras, Newport, San Sebastian, Rio, Cali, Canberra

## What Claude should do next

### Highest-value next work

1. Keep targeting the remaining 2025-2026 `source_absent` rows above.
2. Continue using exact manual search for:
   - `PokerStars`
   - `Betfair`
   - `Smarkets`
   - `SportyTrader`
   - `Matchstat`
   - `SignalOdds`
3. If a page is accessible in browser/open search but blocked in shell, use search snippets if both odds are present.
4. Do not spend much time on `source_has_no_numeric_odds` team-event rows unless a brand new source family is found.

### Lower-value / harder work

- `Metz 2025`:
  - still the largest real-absence bucket
  - also polluted by `5` upstream-bad duplicate rows
- old `ATP Cup` / `Davis Cup` / `BJK` tails:
  - many are either truly absent or already demoted to `source_has_no_numeric_odds`

## Guardrails

- Do not fabricate anything.
- Only add a row if both sides of the moneyline are explicit.
- Keep provenance in `manual_search_snippets_odds.csv`.
- After every batch, rerun:
  - `python scripts\fix_tennis_odds_matching.py`
- Use `data/processed/tennis/unmatched_diagnostics.csv` as the source of truth for what is still missing.
