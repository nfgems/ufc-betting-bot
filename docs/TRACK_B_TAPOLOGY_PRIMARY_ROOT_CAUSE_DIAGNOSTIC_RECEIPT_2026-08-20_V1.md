# Track B Tapology primary root-cause diagnostic receipt — 2026-08-20 v1

Status: `TAPOLOGY_PRIMARY_NOT_READY`

This receipt records the authorized fail-closed investigation of Weekly Tapology Profile Refresh run `31927852134`. It does not authorize or record a deployment, commit, push, workflow run, production configuration change, schedule change, model action, or order action.

## Scope and immutable anchors

- Track B checkout: `C:/Users/Evan/betting-bot/ufc-betting-bot-main-production-warning-repair-20260813`
- Exact base and `HEAD`: `3eacba726d5f0fd6c497492a18c74a85d8001044`
- Prior Track B handoff SHA-256: `8b6d3d72e12f2f86cfc8221dc3ba84a9e9db2e2a6a53a5fc7775756aad4caed4`
- Track A living-handoff resumption SHA-256 before the authorized append: `67e92bd05157d06daa7de06f28cc43037188cb59dbfbd92db4859683783f1d93`
- Frozen Track A candidate receipt SHA-256: `17359ba860f7698c369d29b83b4197734404e93eb7e20d1883e5bb992d1704d8`
- Frozen Track A candidate manifest SHA-256: `6b96536d5ea9e34a362b1feffebf746888bc1b31e44af47127762148006fa8dc`
- Frozen production pointer SHA-256: `a26179e9afdb1f0125c2dcec94c2595ceb3a59cb8364b9133c66743ad883a404`

The pre-investigation Track B status contained exactly the expected five local changes: the wallet source/test pair, the one-line Tapology classification edit and its test, and the prior Track B handoff. Those changes were inspected completely before this receipt was written.

## Failed hosted run evidence

- Repository/run: `nfgems/ufc-betting-bot`, run `31927852134`, run number 20, attempt 1
- Trigger/ref: scheduled, `main`
- Run SHA: `3eacba726d5f0fd6c497492a18c74a85d8001044`
- Created: `2026-08-16T04:57:19Z`
- Completed: `2026-08-16T05:01:53Z`
- Result: failure
- Primary job: `95118090348`, GitHub-hosted Ubuntu 24.04/X64 in Azure `northcentralus`
- Alternate job: `95118500604`, skipped before runner assignment
- Finalizer job: `95118500701`, failed because no validated attempt existed
- Primary artifact: ID `9258419475`, `tapology-primary-31927852134-1`, 4,516 bytes, ZIP SHA-256 `8f67e5361928de0291307769fd7caed7634bcc6d3421fd9ee3faa1856dd5f04c`
- Complete GitHub job-log archive: 32,935 bytes, SHA-256 `b075f958ce5e0a7761dfc491ce027f719fc8887bb75b3970351b432db0039cb0`

The artifact contained no supplement result because packaging was skipped. Its complete diagnostic entries were:

- `attempt.json`: 8,505 bytes, SHA-256 `a6a7b64ae3cecd730eb4688f533cde1dd0d3ce6dd2814b5b2ca80a871a77554d`
- `attempt.log`: 14,804 bytes, SHA-256 `9592c84c4a3546a99a0b69814225fb65224cb3dc0d4922346dee190ec6367f8b`
- `runtime.txt`: 483 bytes, SHA-256 `b106280d85f5cf561074302530d76cbb3c55b1897efbddecb3ebe8ca2cdffc6d`

Recorded runtime configuration was `route=direct`, `proxy_configured=no`, and `brave_search_configured=no`. The runner log also proved that the reader/Jina API key was blank. The primary explicitly did not use the optional configured proxy.

## Exact route and failure layer

| Layer | Run evidence | Current bounded evidence | Judgment |
| --- | --- | --- | --- |
| Tapology origin | `https://www.tapology.com/search?term=Feng+Pengchao` hit a Cloudflare challenge. | Plain origin searches for Feng and Harrison and the known Feng profile still returned HTTP 403 Cloudflare challenge pages. | Origin access is externally WAF-blocked from both tested egresses. |
| Hosted browser | Chrome/Xvfb launched successfully, but the Feng search still returned HTTP 403 `Cloudflare challenge from browser fallback`. | No bypass or evasive browser probe was attempted. | Browser availability was healthy; Tapology access through it was not. |
| Configured reader | Feng search/profile, Felix, and Frank succeeded. Harrison's exact reader search returned HTTP 403 twice. | The same unauthenticated reader URL now returns HTTP 200 and a valid Tapology zero-result search page. | The run's terminal failing layer was the external reader HTTP route. The deeper reader-side reason is unproven. |
| Search/discovery | Harrison never produced a profile URL. | Tapology currently reports `Search Results (0)` for Harrison; bounded Tapology-only name variants found no profile. | No current Harrison Tapology identity/profile exists to parse. |
| Site-search fallback | DuckDuckGo HTML search returned an anti-bot challenge. No Brave API key was configured. | No different provider was treated as Tapology. | The label `unavailable_or_no_results` hid an unavailable/challenged route in the failed run. |
| Optional proxy | No proxy secret was configured; the primary disables it even when present. | No proxy was used or configured. | Not tested and not proven necessary; changing it would be a hosted configuration decision. |
| GitHub runner egress | The reader and other public routes worked immediately before Harrison. | Local reader access works now. | A total runner-network outage is contradicted. A route-specific transient restriction remains possible. |
| Parser/schema | Feng and three candidate profiles reached and passed the existing parser. Harrison returned HTTP 403 before parsing. | The existing parser recognizes both Feng's current search/profile and Harrison's explicit zero-result search. | Parser drift did not cause Harrison's recorded failure. |

The exact failing request was:

`https://r.jina.ai/https://www.tapology.com/search?term=Harrison+Garcia`

At `2026-08-16T05:01:22.977Z`, the default reader variant returned HTTP 403. The repository retried once with only `x-no-cache: true`; at `05:01:27.943Z`, the second HTTP 403 opened the reader circuit for 900 seconds. No Authorization header was present. The preserved run did not capture either 403 response body, safe response headers, redirect chain, cookies, response length, or response hash. It is therefore impossible to distinguish reader policy/quota, shared-reader rate or abuse controls, target-specific cache behavior, relayed upstream blocking, or another reader-side rule from the archived evidence.

## Classification chain

1. Reader 403 became `TapologyRequestError(..., status_code=403, detail="reader search unavailable")`.
2. The reader circuit opened and discovery returned `runtime_block`.
3. Challenged site search became the ambiguous `unavailable_or_no_results` label.
4. Candidate refresh persisted `Tapology discovery unavailable` but omitted the underlying HTTP 403.
5. The aggregate classifier selected `discovery_failed`.
6. The workflow accepts only `hosted_egress_blocked` or `network_unavailable` as alternate-runner outcomes, so the primary output was `failed` and the alternate job was skipped.
7. The finalizer failed and no data commit occurred.

The existing local one-line edit in `scripts/refresh_tapology_profile_supplement.py` maps the persisted word `runtime_block` to `hosted_egress_blocked`. Its test is mock-based. This corrects the alternate-runner decision only; it does not restore origin, reader, search, parsing, or Harrison data and is not the Tapology primary fix.

## Input and checkpoint freshness

At run start, the three repository inputs all contained bytes originating from commit `972911d11cf9f7349b5349d6fbde6c6ca935f3a4` at `2026-08-04T10:56:08Z`, an age of 11 days, 18 hours, 1 minute, 11 seconds:

| Input | Git blob | Current SHA-256 |
| --- | --- | --- |
| `data/raw/ufc_active_roster_official.csv` | `be85d5dd780b305d1e34d03eb229dff287880fd2` | `d83792d089241b3ba31cee4a37360f9793d54207dd2d416d2496f0ec0a003e2f` |
| `data/raw/ufc_fighters_scraped.csv` | `89d7d2c91da3fa68b746f2347ba8fc5baa3d9c01` | `321ee0e66918d75de1e44d72993787901fcd0759a9b80c51881aef5babc0723a` |
| `data/raw/ufc_fighters_profile_supplement.csv` | `679245e5ce9eca4d9c3dafbedb629ee61b820cce` | `8571feb42fff37d443e5944309d330494c3e669c2b7ab13af25d983eb8cbf23b` |

The scheduled run did not sync the active roster, but it did fetch current UFC.com cards during execution: 64 fight contexts and 128 fighter names. Harrison was already an active, coverage-eligible roster candidate with no stored Tapology URL and no Tapology supplement/scraped row. Candidate selection used the run-number rotation, not a permanent Tapology failure checkpoint. Therefore neither a stale Harrison URL nor a consumed checkpoint caused the reader 403. The older inputs remain a broader continuity concern, but they do not explain this target-specific HTTP response.

## Current sanitized probes

Probe window: `2026-08-20T19:19:19Z` through `2026-08-20T19:22:07Z`. Requests were bounded, read-only, used no credentials or proxy, and followed no access-control bypass. All recorded responses had no redirect.

| Public endpoint | Status | Bytes | SHA-256 | Safe classification |
| --- | ---: | ---: | --- | --- |
| Origin Feng search | 403 | 5,838 | `6ac3b3bd2ff68aea14375bd8b0c026b2cb90c44d8c04a4ea0e46c25ae35bfe93` | Cloudflare challenge |
| Origin Harrison search | 403 | 5,588 | `c9f134f096c5568d486bff3c8f5497b2722361892ca46df605336d06f590b391` | Cloudflare challenge |
| Origin Feng profile | 403 | 5,956 | `c9bf0e2bacdd7c2b0f1ae390a9c9a8013a5cdd537785982309a18b9a456a8866` | Cloudflare challenge |
| Reader Feng search, default | 200 | 4,989 | `085ad64ac7cac6110eb99b5e0ffc1c0e0e9f76ba8906a627c430c1ed1d830915` | Valid Tapology search, one fighter result |
| Reader Harrison search, default | 200 | 4,734 | `984b25f62a2c1d845a98e0c9eb134f2e4b03fdfa94060c21a7782b41e181a7d1` | Valid Tapology search, zero fighter results |
| Reader Feng profile, default | 200 | 65,120 | `7297469eb5044affded100f2e1b573fb049b3d91e273f6dfac92843ec13c501b` | Valid Tapology fighter profile |
| Reader Harrison search, `x-no-cache`, one observation | 200 | 4,734 | `fafb5d1591b453cc03d4e72f88e933a24491fa3dc2b9a5d61be7029b8074ab0f` | Valid zero-result search |
| Reader Harrison search, `x-no-cache`, adjacent observation | 200 | 134 | `509e5e588d68f266dd4afc343f31988f5851590e87f1a789f536b22abc323eef` | Unrelated 1x1 tracker content, not Tapology search markup |

Safe response headers identified the origin as `text/html` served by Cloudflare and the reader as `text/plain` served by Cloudflare via `1.1 google`. The reader exposed a public rate-limit budget of 20 requests per 60 seconds during the default probes. No `Retry-After` or redirect was observed. The changing `x-no-cache` result is additional evidence of reader/cache instability; it is not proof of a repository parser defect.

The actual repository parser currently produces:

- Feng Pengchao: healthy reader discovery with the known profile URL; parsed identity `Pengchao Feng`, record `15-11-1`, height `173 cm`, reach `173 cm`, weight `146 lbs`, DOB unavailable.
- Harrison Garcia: healthy reader discovery state `no_results`, no candidate URL, and therefore no identity or profile fields to parse.

## Hosted configuration and Railway read-only evidence

GitHub run logs proved no configured Tapology proxy, reader/Jina key, or Brave key. These are optional routes; their absence is a fact, not proof that adding one would fix the failure.

Railway identity was verified read-only. The isolated Track B checkout itself is intentionally unlinked, so target verification used explicit project/environment arguments. Project `ufc-betting-bot`, environment `production`, service `ufc-bot` currently reports rollback deployment `ba12c1d7-3f0e-4c3a-9c51-7e678558f786` as `SUCCESS`/running on commit `3eacba726d5f0fd6c497492a18c74a85d8001044`. A sanitized variable-name inspection found no Railway `TAPOLOGY_*`, `JINA_*`, or `BRAVE_*` variable; the only matching proxy-named variable was the unrelated `CLOB_PROXY_URL`. No raw variable value was printed or retained. Railway is not the failed scheduled workflow path, and its direct Tapology fetch remains intentionally disabled/delegated.

## Root-cause decision and minimal-repair decision

The proven proximate cause of run `31927852134` is an external HTTP 403 at the configured unauthenticated reader layer for the exact Harrison discovery URL. Origin and browser access were separately Cloudflare-blocked. The deeper reader-side reason is not recoverable from the retained run and is not reproducible from local egress today.

Independently, current Tapology content returns a healthy zero-result search for Harrison Garcia. The real configured primary path therefore cannot currently return Harrison identity plus expected profile fields. This fails the stated acceptance condition even though Feng remains healthy.

No repository-code cause of the primary access/content failure was proven. Consequently, no speculative source, workflow, parser, proxy, provider-substitution, or silent-green repair was made. The exact minimal honest repository repair for this session is **none**. The pre-existing classifier edit remains only a fallback-routing improvement.

## Validation and files changed

- Focused Tapology classification, workflow-contract, and diagnostic-hardening tests: `82 passed in 1.51s` using only `C:/Users/Evan/betting-bot/_ci_venv/Scripts/python.exe`.
- Relevant compilation: passed with the mandated interpreter.
- A broader profile-enrichment diagnostic produced `313 passed, 1 failed`; the sole failure was `ModuleNotFoundError: selenium` in a browser-timeout test because Selenium is absent from the mandated CI virtual environment. The failed GitHub run itself had Selenium/Chrome/Xvfb and passed its browser smoke test, so this local dependency absence does not explain the hosted incident. No test was weakened or skipped to claim a green broad suite.
- No Tapology source, workflow, parser, configuration, or test file was changed by this investigation.
- The only new Track B file is this versioned receipt. The existing five local changes remain otherwise untouched.

## Remaining blocker and exact next authorization

Primary readiness cannot be proved locally or from the archived run. If the owner wants to distinguish current GitHub-runner behavior from the local result, the next permissible step requires separate authorization for one diagnostic-only GitHub-hosted probe. It should request only the exact Feng and Harrison origin/reader search/profile routes, write no supplement, make no commit, and retain sanitized route, status, redirect, safe-header, byte-count, hash, and parser classification evidence. It must not rerun run `31927852134` or launch the full refresh.

Even a green hosted reader probe would not meet acceptance while Tapology itself reports zero Harrison results. Readiness also requires a genuine current Tapology Harrison profile identity and expected fields; no alternate provider or fabricated/inferred URL may substitute for it. Configuring a reader credential, proxy, or search API would be a separate hosted/service decision and is not justified as a repair by this evidence alone.

## Final outcome

`TAPOLOGY_PRIMARY_NOT_READY`

No deployment, restart, Railway mutation, workflow run/rerun, production configuration change, schedule activation, order action, model action, pointer change, commit, or push occurred.
