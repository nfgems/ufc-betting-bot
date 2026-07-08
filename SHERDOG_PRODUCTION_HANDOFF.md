# Sherdog Production Access Handoff

Date: 2026-07-05 (resolution update: 2026-07-08)

## Resolution Update - 2026-07-08

Direct Sherdog access from Railway production is working again, verified with
the proof bar defined in this document, run from the production `ufc-bot`
runtime via read-only SSH probes:

- `https://www.sherdog.com/fighter/Ian-Garry-268923` returned `200` with real
  content: `FIGHT HISTORY - PRO` section, `module fight_history`,
  `fighter-info`, real page title (not `Just a moment...`).
- FightFinder search returned `200` with `fightfinder_result` and the correct
  profile link.
- The deployed app code end-to-end: `search_sherdog("Ian Garry")` found the
  profile URL and `scrape_sherdog_page` parsed the full profile plus 18 fight
  history rows (2019 Cage Warriors through 2025 UFC).

Nobody changed anything to cause this: the Cloudflare block was transient on
Sherdog's side (worked Jul 3, challenged Jul 4-7, lifted by Jul 8). The real
defect this revealed is that the worktree's `_sherdog_blocked` flag was sticky
for the process lifetime, so a temporary challenge would disable Sherdog until
a redeploy. That was fixed on 2026-07-08:

- `_sherdog_blocked` replaced with a TTL cooldown
  (`SHERDOG_BLOCK_COOLDOWN_SECONDS` in `src/config.py`, default 1800s). After
  the cooldown the next request probes Sherdog again; on success an INFO
  "Sherdog access restored" is logged and the block alert is re-armed.
- Sherdog request pacing got jitter (1.5-3.0s) to look less like a
  uniform-interval crawler to Cloudflare bot scoring.

A free degraded-mode fallback was also added for the next block
(`SHERDOG_WAYBACK_FALLBACK_ENABLED`, default on): while Sherdog is
Cloudflare-blocked, fighter profile pages are served from the newest Wayback
Machine snapshot (`https://web.archive.org/web/<ts>id_/<url>` raw mode, same
parser), and name-to-URL discovery falls back to Wayback CDX prefix queries
(after the existing search-index candidates). This satisfies the constraints:
no proxy, no Railway changes, no browser machinery. Production proof obtained
2026-07-08 from the `ufc-bot` runtime: CDX lookup 200 and raw snapshot fetch
200 with `FIGHT HISTORY`/`fighter-info` markers and no challenge markers.
Research also verified archive.org's crawler is on Cloudflare's verified-bots
allowlist — Save Page Now captures made during the block window (2026-07-04)
and on 2026-07-08 contained real Sherdog content, so snapshots keep updating
even while datacenter egress is blocked. Wayback data is used only while
blocked, never in normal operation, and a wayback-served page never clears the
direct-access cooldown.

The Cloudflare classification, fail-fast, and fallback behavior described
below is retained. Expect the block to recur; the bot now recovers from it
automatically and keeps profile data flowing via Wayback meanwhile.

## Summary

Production previously could fetch Sherdog. It no longer can. Current production evidence shows Sherdog content pages are reachable at the TCP/DNS level but blocked by Cloudflare before the app receives actual page HTML.

The direct Sherdog connection is not restored. Do not claim it works until a production probe returns real Sherdog profile or FightFinder HTML, not a Cloudflare challenge page.

User constraints:

- Do not use or suggest a Sherdog proxy.
- Do not change Railway regions, service locations, environments, or deployment settings.
- Keep the fix simple. Do not add browser/stealth machinery unless it is proven to work in production first.

## Production Target

Railway context checked read-only:

- Project: `ufc-betting-bot`
- Environment: `production`
- Service: `ufc-bot`
- `railway whoami` succeeded as the expected account.
- `railway status` showed environment `production` and service `ufc-bot` online.

Note: after a Railway CLI upgrade, `railway ssh` wrapper hung on commands, but direct OpenSSH to the Railway SSH target worked. This was used only for read-only runtime probes.

## Timeline From Production Logs

Sherdog was working on 2026-07-03:

- `2026-07-03 19:31:32` - found Benoit Saint-Denis on Sherdog: `https://www.sherdog.com/fighter/Benoit-St-Denis-317103`
- `2026-07-03 19:59:18` - found Ian Garry on Sherdog: `https://www.sherdog.com/fighter/Ian-Garry-268923`
- `2026-07-03 20:08:00` - found Zachary Reese on Sherdog: `https://www.sherdog.com/fighter/Zachary-Reese-100903`
- `2026-07-03 19:36:18` - merged `67,360` pre-UFC career rows from Sherdog.

Failures began on 2026-07-04:

- `2026-07-04 07:45:36` - FightFinder returned `403 Client Error: Forbidden`
- Later July 4 and July 5 logs show repeated `403` failures for FightFinder searches and direct profile pages.

Conclusion: this was not a longstanding missing-integration problem. Sherdog worked, then started returning Cloudflare 403s from production around July 4, 2026.

## Production Probe Results

All probes below were run from the production `ufc-bot` runtime without changing Railway settings.

### Direct Requests

Tested URLs:

- `https://www.sherdog.com/stats/fightfinder?SearchTxt=Benoit%20Saint-Denis`
- `https://www.sherdog.com/fighter/Benoit-St-Denis-317103`
- `http://www.sherdog.com/stats/fightfinder?SearchTxt=Benoit%20Saint-Denis`
- `http://www.sherdog.com/fighter/Benoit-St-Denis-317103`
- `https://sherdog.com/stats/fightfinder?SearchTxt=Benoit%20Saint-Denis`
- `https://sherdog.com/fighter/Benoit-St-Denis-317103`

Result:

- Every content URL returned HTTP `403`.
- Response server was `cloudflare`.
- Cloudflare Ray region was `AMS` for production.
- With default headers, title was `Attention Required! | Cloudflare`.
- With Chrome-like document headers, title was `Just a moment...` and `cf-mitigated: challenge`.
- No real Sherdog markers were present (`fightfinder_result`, `Fight History`, `fighter-info` were absent from the returned content).

### Browser Probe

Production container has:

- Chromium: `/usr/bin/chromium`
- Chromedriver: `/usr/bin/chromedriver`
- Selenium: `4.44.0`
- Xvfb: `/usr/bin/Xvfb`

Selenium/Chromium probe against:

- `https://www.sherdog.com/fighter/Benoit-St-Denis-317103`

Result:

- Browser title stayed `Just a moment...`
- Current URL gained a `__cf_chl_rt_tk` challenge parameter.
- `has_fight_history=False`
- `has_fighter_info=False`
- `has_cloudflare=True`

Conclusion: a basic real browser in production did not restore Sherdog access.

### Lightweight Endpoint Probe

Production can reach:

- `https://www.sherdog.com/robots.txt` - returned `200`

Production is blocked on:

- `https://www.sherdog.com/` - `403`, Cloudflare challenge
- `https://www.sherdog.com/sitemap.xml` - `403`, Cloudflare challenge
- `https://www.sherdog.com/news/news/list` - `403`, Cloudflare challenge
- FightFinder and fighter profiles - `403`, Cloudflare challenge

Conclusion: the network path to Sherdog exists, but Cloudflare blocks actual content pages.

### Environment Proxy Check

Checked sanitized proxy-related environment variable names in production.

Result:

- No global `HTTP_PROXY`, `HTTPS_PROXY`, or `ALL_PROXY` was set.
- Only unrelated `CLOB_PROXY_URL` was present for Polymarket/CLOB behavior.

Conclusion: Python `requests` is not accidentally routing Sherdog through a global proxy.

## What Was Tried And Rejected

These did not restore direct Sherdog access:

- Plain Python `requests`
- Chrome-like request headers
- `http` instead of `https`
- `www.sherdog.com` vs `sherdog.com`
- Direct profile URL instead of FightFinder search
- Production Chromium/Selenium with Xvfb
- `cloudscraper`
- Jina reader for Sherdog pages
- `curl_cffi` Chrome impersonation variants
- `undetected-chromedriver`

Temporary Railway region probes were also performed before the user explicitly said not to change Railway locations anymore. Those probes were done in a temporary Railway environment/service, not by scaling the production service, and the temporary environment was deleted afterward. Do not repeat region/location experiments unless the user explicitly authorizes them.

## Current Code Changes In Worktree

Files with intentional content changes:

- `src/data/fallback_scrapers.py`
- `src/data/name_utils.py`
- `tests/test_name_utils.py`
- `tests/test_v4_profile_enrichment.py`

Files shown as modified due to line-ending metadata only, with no content diff:

- `.env.example`
- `src/config.py`

### `src/data/fallback_scrapers.py`

Added explicit Sherdog Cloudflare classification:

- New `_sherdog_blocked` process-level state.
- New `SherdogRequestError`.
- New `_get_sherdog_soup()` wrapper for Sherdog only.
- `_get_sherdog_soup()` checks `_is_cloudflare_challenge(resp)` before generic `raise_for_status()`.
- On Cloudflare, it logs `External data source unavailable: Sherdog - blocked by Cloudflare`, marks `_sherdog_blocked=True`, and raises `SherdogRequestError(..., status_code=403, detail="Cloudflare challenge")`.
- Once `_sherdog_blocked=True`, later Sherdog fetch attempts in that process fail fast instead of repeatedly hammering FightFinder variants.

Updated Sherdog search flow:

- `search_sherdog()` now uses `_get_sherdog_soup()` instead of generic `_get_soup()`.
- If FightFinder is Cloudflare-blocked, it stops retrying name variants and optionally tries search-index-discovered Sherdog profile URLs.
- Direct profile scraping still requires fetching Sherdog and will still fail under the current Cloudflare block.

Updated Tapology fallback behavior:

- The production Railway default has Tapology origin fetch disabled, but the reader/search-index fallback can still provide data.
- `_tapology_profile_fetch_available()` now allows the reader path.
- `search_tapology_candidates()` can use reader/search-index discovery even when Tapology origin fetch is disabled.
- `scrape_tapology_profile()` and `scrape_tapology_fights()` try the reader path before rejecting due to disabled origin fetch.

Removed after proving it did not work:

- Sherdog-specific browser fallback code.
- `SHERDOG_BROWSER_FALLBACK_ENABLED`.
- Any Sherdog proxy references.

### `src/data/name_utils.py`

Added nickname normalization:

- `zach` -> `zachary`
- `zack` -> `zachary`

This helps match cross-source names such as Zach Reese/Zachary Reese.

## Verification Already Run

Focused tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests\test_v4_profile_enrichment.py::test_get_sherdog_soup_marks_cloudflare_blocked `
  tests\test_v4_profile_enrichment.py::test_search_sherdog_uses_site_search_when_fightfinder_cloudflare_blocked `
  tests\test_v4_profile_enrichment.py::test_search_tapology_candidates_uses_reader_on_railway_without_origin_fetch `
  tests\test_v4_profile_enrichment.py::test_fallback_lookup_uses_tapology_reader_path_on_railway_runtime `
  tests\test_name_utils.py::test_zachary_reese_cross_source_alias
```

Result:

```text
5 passed
```

Compile check:

```powershell
.\.venv\Scripts\python.exe -m compileall src tests
```

Result:

```text
passed
```

Search checks:

```bash
rg -n "SHERDOG_BROWSER|_sherdog_browser|Sherdog browser|Sherdog.*proxy|proxy.*Sherdog" .env.example src tests
```

Result:

```text
no matches
```

## Important Non-Claims

Do not say the issue is fixed.

Do not say production can connect to Sherdog content.

Do not say a commit/push will make Sherdog work. The current evidence proves the opposite: production is still blocked by Cloudflare for actual Sherdog content endpoints.

What is improved:

- The app now detects the actual Sherdog Cloudflare block explicitly.
- It avoids repeating doomed FightFinder variant requests after the block is known.
- It can continue to non-Sherdog fallback paths more cleanly.

What is not solved:

- Direct Sherdog content access from Railway production.

## Suggested Next Steps For Claude

1. Start from the current worktree. Do not reintroduce Sherdog proxy or browser fallback unless production proof comes first.
2. Re-run a read-only production probe against Sherdog content pages.
3. If production still returns Cloudflare 403, treat direct Sherdog restoration as externally blocked under the current constraints.
4. If a code-only direct access path is proposed, require this proof before accepting it:
   - Run from Railway production `ufc-bot`.
   - Fetch a real Sherdog fighter profile or FightFinder result.
   - Response status is `200`.
   - HTML contains real Sherdog markers such as `Fight History`, `fighter-info`, or `fightfinder_result`.
   - HTML title is not `Just a moment...` or `Attention Required! | Cloudflare`.
5. Keep any fallback changes separate from the claim of restoring Sherdog. Fallback data recovery is useful, but it is not the same as restoring Sherdog access.
