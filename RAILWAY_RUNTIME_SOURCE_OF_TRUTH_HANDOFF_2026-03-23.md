# Railway Runtime Source-Of-Truth Handoff

Date: 2026-03-23

Purpose: document the production-runtime data/model/provenance problem for the UFC system, explain the current code path, and define the cleanup plan so a follow-up instance can implement it without redoing the investigation.

## Executive Summary

The main problem is not local-machine execution. The real production risk is Railway runtime drift caused by:

1. persistent-volume precedence for models and data,
2. startup seeding that only copies files when missing,
3. runtime path resolution based on naming conventions instead of an explicit production bundle,
4. multiple artifacts in the repo that look authoritative but are not actually used by the hosted bot.

In the current design, Railway can keep using stale model/data files from its mounted volume even after a new deploy, because startup does not overwrite existing volume artifacts. Separately, `fighter_lookup` can select processed data based on `spec.name` and candidate-folder naming, which is fragile and can silently fall back.

The fix is to establish one real production source of truth for:

- active models,
- active processed snapshot,
- active training spec / bundle metadata,
- operator/bot/runtime provenance.

Recommended direction:

1. stop defaulting to persistent-volume models on Railway,
2. make `data/processed/*.csv` the only production processed snapshot,
3. introduce a real production bundle manifest,
4. fail closed on bundle/spec/model mismatch,
5. expose provenance in logs/UI/API,
6. demote legacy and candidate artifacts so hosted runtime never touches them.

## Important Constraint

The user explicitly stated that nothing should be treated as running on the local machine; production is Railway only.

That means:

- local repo inspection is useful for code-path truth,
- but it does not prove the current live Railway filesystem contents,
- and the largest risk is whatever is already sitting on the Railway mounted volume.

This handoff therefore focuses on runtime architecture and deployment semantics, not local ad hoc behavior.

## Current Production Runtime Path

### 1. Railway startup mounts persistent dirs and exports runtime paths

Startup sets the runtime persistent dirs in:

- `entrypoint.sh:6`
- `src/config.py:63`
- `src/config.py:64`

Relevant behavior:

- `PERSISTENT_DATA_DIR` comes from `UFC_DATA_DIR` or `RAILWAY_VOLUME_MOUNT_PATH`
- `PROCESSED_DATA_DIR` resolves under `DATA_DIR`
- `MODELS_DIR` is selected by `_resolve_default_models_dir()`

Key files:

- `entrypoint.sh`
- `src/config.py`

### 2. Startup seeds models and processed data into the volume only if missing

Current startup migration behavior:

- `entrypoint.sh:70` copies `/app/models/xgboost_model.pkl` into persistent models dir only if missing
- `entrypoint.sh:72` copies `/app/models/xgboost_no_odds_model.pkl` only if missing
- `entrypoint.sh:76` copies `/app/data/processed` into persistent processed dir only if files are missing

This is the core stale-deploy risk.

If the Railway volume already contains:

- an old `xgboost_model.pkl`,
- an old `xgboost_no_odds_model.pkl`,
- old `data/processed/features.csv`,
- old `data/processed/fights_cleaned.csv`,

then a fresh deploy does not replace them.

### 3. Hosted runtime starts the web service in Railway

Hosted entrypoint:

- `entrypoint.sh` final `web` branch launches `python -m src.web.serve`
- hosted UFC refresh loop lives in `src/web/serve.py:166`

Important implication:

- the always-on Railway service uses the mounted volume,
- scheduled refresh also runs against that mounted volume,
- so stale volume contents can persist across deploys and blend with new code.

### 4. Bot loads active models from canonical aliases

Runtime loads model artifacts through:

- `src/model/train.py:656` `load_model()`

The live bot resolves embedded training spec from the loaded artifact in:

- `src/bot.py:131` `_training_spec_from_model_result()`

Important implication:

- runtime truth is the actual `xgboost_model.pkl` loaded from `MODELS_DIR`,
- not `models/current_production_model.json`,
- not candidate-dir names,
- not README statements.

### 5. Bot builds fight features, then operator receives those features

Live feature flow:

- `src/bot.py:2390` `build_fight_features(...)`
- `src/bot.py:2401` passes `training_spec=inference_spec`
- `src/strategy/llm_operator.py:1026` operator reads `features_by_fight`

Important implication:

- operator is downstream of the bot feature builder,
- operator does not directly choose between raw files / processed snapshots / live scrape,
- it sees the resolved feature payload the bot already built.

### 6. fighter_lookup currently resolves processed snapshot via spec-name inference

Relevant code:

- `src/data/fighter_lookup.py:300` `_resolve_processed_data_dir()`
- `src/data/fighter_lookup.py:310` candidate-dir path built as `PROCESSED_DATA_DIR / "candidates" / spec.name`
- `src/data/fighter_lookup.py:314` fallback to base `PROCESSED_DATA_DIR`
- `src/data/fighter_lookup.py:2005` `lookup_fighter()`
- `src/data/fighter_lookup.py:2152` `build_fight_features()`
- `src/data/fighter_lookup.py:2067` live-refresh stale warning

Important implication:

- if candidate folder naming does not exactly match `spec.name`, runtime silently falls back,
- if candidate folder exists but is old, runtime may use it until live-staleness threshold forces scrape,
- there is no explicit production processed snapshot manifest.

## Current Artifact Taxonomy

This is the clean mental model that should be carried into the refactor.

### A. Raw rebuild inputs

These are rebuild-time inputs, not direct live-runtime feature sources:

- `data/raw/ufc-fight-results.csv`
- `data/raw/ufc-fight-stats.csv`
- `data/raw/ufc_fighters_scraped.csv`
- `data/raw/ufc_fighters_profile_supplement.csv`
- `data/raw/ufc_active_roster_official.csv`

Relevant rebuild code:

- `src/data/ufc_refresh.py:685` `build_training_rows_from_pulled_data()`
- `src/data/ufc_refresh.py:1828` `_load_scraped_fighter_lookup()`
- `src/data/ufc_refresh.py:839` `build_training_dataset_variants()`
- `scripts/rebuild_ufc_processed_artifacts.py:33` `run_rebuild()`

### B. Legacy raw dataset

Legacy-only artifact:

- `data/raw/ufc-master.csv`

This is still used by rebuild/merge/training utilities, but it should not be treated as live production truth for hosted inference.

### C. Production processed snapshot

These should be the only processed files the hosted runtime depends on:

- `data/processed/fights_cleaned.csv`
- `data/processed/features.csv`

These are the normalized and model-ready outputs after rebuild.

### D. Offline experiment artifacts

These should remain offline only:

- `models/candidates/*`
- `data/processed/candidates/*`

Hosted runtime should never need to guess from or inspect these.

### E. Production model aliases

These should be the only hosted model artifacts:

- `models/xgboost_model.pkl`
- `models/xgboost_no_odds_model.pkl`
- optional `models/logistic_model.pkl`

### F. Metadata / manifest

Currently ambiguous:

- `models/current_production_model.json`

Observed issue:

- this file reads like a production manifest,
- but runtime model loading is driven by the actual alias `.pkl` files,
- so the JSON can drift and confuse debugging.

## Core Problems To Fix

### Problem 1: Persistent volume can override fresh deploy contents

Root cause:

- startup uses copy-if-missing semantics for models and processed data,
- config prefers persistent models on hosted deployments if they exist.

Effect:

- Railway may keep old model/data after a redeploy,
- image contents may never take effect,
- deploy status can look healthy while runtime bundle is stale.

### Problem 2: Runtime chooses processed snapshot by naming convention

Root cause:

- `_resolve_processed_data_dir()` infers candidate processed dir from `spec.name`

Effect:

- candidate-dir naming drift breaks deterministic resolution,
- runtime behavior depends on folder names instead of explicit promotion metadata,
- debugging becomes guesswork.

### Problem 3: Multiple “authoritative” artifacts exist

Confusing artifacts:

- `ufc-master.csv`
- `data/processed/*.csv`
- `data/processed/candidates/*`
- `models/current_production_model.json`
- `models/*.pkl`

Effect:

- humans cannot quickly tell which files drive Railway,
- promotion/debugging/handoffs become error-prone,
- old docs continue to mislead after promotion changes.

### Problem 4: Operator provenance is too opaque

Current operator logs decisions and rationale, but not enough runtime provenance to answer:

- which bundle was active,
- which processed snapshot was used,
- whether a fighter came from processed snapshot or live scrape,
- which embedded model spec was active.

Effect:

- impossible to distinguish stale data vs weird real data quickly,
- UI surfaces outputs without enough operational context.

### Problem 5: Legacy files remain in the critical path conceptually

Even if runtime does not directly read `ufc-master.csv`, it is still mentally mixed into discussions of “the fighter data” and “the model data.”

Effect:

- team confusion,
- wrong debugging assumptions,
- incorrect blame when live inference behaves oddly.

## Recommended Target Design

### Production Source Of Truth Contract

Hosted runtime should use exactly one production bundle contract:

- active model paths,
- active processed-data path,
- active spec name,
- bundle id,
- bundle build timestamp,
- processed snapshot max event date.

The bundle should be explicit, not inferred.

### Recommended production bundle shape

One acceptable approach:

`data/production_bundle/current/manifest.json`

plus:

- `data/production_bundle/current/models/xgboost_model.pkl`
- `data/production_bundle/current/models/xgboost_no_odds_model.pkl`
- `data/production_bundle/current/processed/fights_cleaned.csv`
- `data/production_bundle/current/processed/features.csv`
- `data/production_bundle/current/spec.json`

Alternative acceptable approach:

- keep canonical aliases in existing places,
- but add a real manifest that explicitly points to them,
- and make runtime validate against that manifest at startup.

Either is fine. The important thing is that runtime must not guess.

## Recommended Fixes

These are listed in implementation order.

### Fix 1: Stop using persistent-volume models as the default on Railway

Recommendation:

- production models should come from the deploy image by default,
- not from the mounted volume,
- unless an explicit override env var is intentionally set.

Why:

- model artifacts should change only when a deploy/promotion happens,
- they should not silently persist across deploys because of old volume contents.

Implementation direction:

- update `src/config.py` so hosted default `MODELS_DIR` prefers `/app/models`
- remove or greatly narrow model-copy seeding in `entrypoint.sh`
- keep the volume for logs and mutable data, not the active model aliases

Acceptance criteria:

- redeploying a new model artifact always changes Railway runtime model without manual volume cleanup
- startup logs clearly print the loaded model file path and embedded spec name

### Fix 2: Make `data/processed/*.csv` the only production processed snapshot

Recommendation:

- hosted runtime should only read:
  - `data/processed/fights_cleaned.csv`
  - `data/processed/features.csv`
- candidate processed dirs should be offline-only

Why:

- runtime should not select between candidate dirs,
- production processed data should be a single canonical snapshot,
- refresh jobs already operate against the canonical hosted volume path.

Implementation direction:

- remove production candidate-dir inference from `fighter_lookup`
- in production/hosted mode, resolve processed snapshot directly to canonical `PROCESSED_DATA_DIR`
- keep candidate dirs for offline evaluation only

Acceptance criteria:

- no hosted runtime code path touches `data/processed/candidates/*`
- live bot and operator always reuse the same canonical processed snapshot when not scraping live

### Fix 3: Introduce one real production bundle manifest

Recommendation:

- add a manifest that runtime actually uses, not just a descriptive JSON sidecar

Required manifest fields:

- `bundle_id`
- `model_spec_name`
- `model_path`
- `no_odds_model_path`
- `processed_dir`
- `snapshot_max_event_date`
- `built_at`
- `git_sha`

Why:

- model, processed snapshot, and metadata must be tied together atomically,
- runtime should use explicit paths, not name-derived guesses.

Implementation direction:

- runtime startup loads manifest first
- model loader and fighter lookup both consume manifest-backed paths
- operator provenance includes manifest `bundle_id`

Acceptance criteria:

- startup can print a single canonical bundle summary
- model path and processed dir are explicit and reproducible

### Fix 4: Fail closed on mismatch

Recommendation:

- if manifest, embedded model spec, and processed snapshot do not line up, startup should stop

Examples of mismatch that should hard-fail:

- manifest `model_spec_name` != embedded model `training_spec.name`
- manifest `processed_dir` missing required files
- processed snapshot max event date older than manifest expectation
- alias paths point somewhere different than manifest

Why:

- silent fallback is the main source of bad production ambiguity

Acceptance criteria:

- Railway service refuses to start with mismatched bundle contents
- failures are obvious in startup logs

### Fix 5: Expose provenance in logs and UI/API

Recommendation:

Log at startup:

- bundle id
- loaded model paths
- embedded model spec name
- processed dir
- processed `features.csv` max event date

Log per operator decision:

- bundle id
- model spec name
- processed dir
- processed snapshot max date
- fighter A source (`processed` / `ufcstats` / fallback)
- fighter B source (`processed` / `ufcstats` / fallback)

Why:

- this makes debugging operator cards trivial,
- it removes guesswork around stale snapshot vs live scrape.

Implementation direction:

- add provenance object to operator decision log
- expose provenance fields through operator API
- optionally surface in Operator UI under “matchup details”

Acceptance criteria:

- operator page/API can answer “where did this stat come from?” without code inspection

### Fix 6: Demote legacy and offline artifacts by naming and usage

Recommendation:

- treat `ufc-master.csv` as legacy-only
- treat candidate dirs as offline-only
- either remove `models/current_production_model.json` from the critical path or make it the real manifest

Why:

- too many files look authoritative right now,
- hosted runtime needs one obvious production path.

Naming / policy direction:

- `data/raw/ufc-master.csv` => legacy training base only
- `models/current_production_model.json` => either delete from runtime discussions or upgrade into the real manifest
- `models/candidates/*` / `data/processed/candidates/*` => evaluation only

Acceptance criteria:

- no future handoff should need to ask “which file is actually live?”

## Immediate Railway Operational Fix

Even before the refactor lands, the dangerous production issue is the mounted volume.

Safest operational sequence:

1. patch runtime so production models come from the image, not the persistent volume
2. patch runtime so production processed data comes from one canonical location only
3. redeploy Railway with a clean volume, or explicitly remove stale persisted model/data files once
4. verify startup logs print:
   - exact model path
   - embedded spec name
   - processed path
   - processed snapshot max event date
   - bundle id / manifest path

This is the cleanest way to stop stale-file ambiguity fast.

## Concrete Code Areas To Change

### Railway startup / hosted path selection

Primary files:

- `entrypoint.sh`
- `src/config.py`

Expected changes:

- stop model-volume precedence by default
- change data seeding behavior so production deploys do not leave stale aliases behind
- add startup provenance logging

### Runtime bundle loading / validation

Primary files:

- `src/model/train.py`
- `src/bot.py`
- possibly a new module, e.g. `src/model/production_bundle.py`

Expected changes:

- load explicit production manifest
- validate loaded model artifact against manifest
- expose manifest info to bot and operator

### fighter_lookup production processed-dir resolution

Primary file:

- `src/data/fighter_lookup.py`

Expected changes:

- remove or gate candidate-dir inference for hosted runtime
- use explicit production processed dir from bundle/manifest
- keep live-refresh stale check behavior, but only against canonical processed snapshot

### Operator provenance

Primary file:

- `src/strategy/llm_operator.py`

Expected changes:

- attach provenance to decision log entries
- optionally surface through API/UI

### Refresh integration

Primary files:

- `src/web/serve.py`
- `scripts/rebuild_ufc_processed_artifacts.py`
- `src/data/ufc_refresh.py`

Expected changes:

- refresh should write into canonical production processed snapshot
- rebuild metadata should update bundle manifest / snapshot metadata if needed

## Proposed Implementation Phases

### Phase 1: Stop the bleeding

Goal:

- eliminate stale-model ambiguity on Railway

Tasks:

- change hosted `MODELS_DIR` default to image models
- remove model-copy-if-missing behavior from startup
- add startup log of actual model path + spec name

### Phase 2: Canonicalize processed production snapshot

Goal:

- make canonical `data/processed/*.csv` the only production processed source

Tasks:

- remove hosted candidate-dir inference
- add startup log of processed path + max event date
- ensure hosted refresh writes canonical processed files only

### Phase 3: Add explicit production bundle manifest

Goal:

- make model/spec/processed snapshot atomic and explicit

Tasks:

- introduce manifest structure
- wire manifest into runtime loading and validation
- fail closed on mismatch

### Phase 4: Add operator provenance

Goal:

- let UI/API reveal runtime bundle and feature source

Tasks:

- extend operator decision logging schema
- surface provenance through API
- optionally add UI section

### Phase 5: Cleanup / demotion of legacy artifacts

Goal:

- prevent future confusion

Tasks:

- document runtime-vs-offline artifact policy
- rename or clearly annotate legacy/offline paths
- remove misleading metadata from critical-path discussions

## Acceptance Criteria

The work is complete when all of the following are true:

- Railway hosted runtime always uses the intended active model after a deploy
- Railway hosted runtime always uses one canonical processed snapshot
- runtime no longer guesses production processed dir from `spec.name`
- startup fails if model/spec/processed bundle mismatch
- operator decision logs expose provenance
- team can answer “what model/data was active?” from logs/API alone
- candidate and legacy artifacts are clearly out of the hosted runtime path

## Risks / Things To Watch

- changing hosted model-path behavior can alter deploy expectations if anyone relies on persistent-volume models today
- changing processed-dir resolution can break tests that assume candidate-dir inference
- startup hard-fail validation will surface hidden inconsistencies that current code tolerates
- Railway volume cleanup must be coordinated so old files do not continue shadowing new deploys

## Open Questions

These do not block the architectural direction, but should be decided during implementation:

1. Should the real production manifest live under `models/`, `data/processed/`, or a dedicated production-bundle directory?
2. Should processed data remain on the volume permanently, or should deploys overwrite it from the image when the bundled snapshot is newer?
3. Do we want bundle ids tied to git SHA, timestamp, candidate label, or all three?
4. Do we want operator provenance only in JSONL/API, or surfaced directly in the web UI too?

## Suggested Next Prompt For Another Instance

Use this handoff to implement the production source-of-truth cleanup for Railway.

Requirements:

- make Railway runtime use image-bundled production models by default
- make canonical `data/processed/*.csv` the only hosted processed snapshot
- remove hosted production reliance on candidate-dir inference from `fighter_lookup`
- introduce a real production bundle manifest used by runtime
- fail closed on model/spec/processed mismatch
- add startup provenance logging and operator decision provenance

Please inspect and modify:

- `entrypoint.sh`
- `src/config.py`
- `src/model/train.py`
- `src/bot.py`
- `src/data/fighter_lookup.py`
- `src/strategy/llm_operator.py`
- `src/web/serve.py`

Then provide:

- the patch,
- rollout notes for Railway,
- and a verification checklist proving the hosted bot/operator are reading the intended model and processed snapshot.
