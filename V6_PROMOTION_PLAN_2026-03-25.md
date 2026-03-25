# V6 Model Promotion Plan & Pre-Flight Checklist

Date: 2026-03-25

## Status Update

Promotion has been verified on Railway.

- On March 25, 2026, the hosted service reported `bundle_id=ufc-production-20260323-full_live_contract_v6_tuned`.
- The startup provenance for Railway deployment `7193cf4f-4f2c-40af-b255-f8c65b17e92e` reported `spec=full_live_contract_v6_tuned`.
- `GET /readyz` returned `200` and exposed `model_spec_name=full_live_contract_v6_tuned` on the same deployment.
- The older persistent-volume precedence risk described below has been mitigated by the hosted runtime bundle bootstrap. The service now serves canonical aliases from `/app/models` and records the active runtime manifest under `/app/logs/production_bundle/current/manifest.json`.

## Executive Summary

The V6 candidate model (`full_live_contract_v6_tuned`) has been proven to be a significant improvement over the current V5 live model (`full_live_contract_v5_fullfit`) in a fair, apples-to-apples comparison.

**The recommendation from this plan was to promote the V6 model to production, and that promotion has now been verified live on Railway.**

However, a critical caveat exists: the current Railway deployment process is fragile. As documented in `RAILWAY_RUNTIME_SOURCE_OF_TRUTH_HANDOFF_2026-03-23.md`, model artifacts on the persistent volume can take precedence over newly deployed code, potentially leaving the old model running.

This document outlines a safe, manual promotion plan to mitigate this risk. A simple `git push` is **not** sufficient.

## Comparison Results (Recap)

The evidence for promotion is clear. The V6 candidate outperforms the V5 live model across all key metrics.

```
================================================================================
LIVE VS CANDIDATE COMPARISON
================================================================================
Metric                    | Live Model (V5)      | Candidate Model (V6)
-----------------------------------------------------------------------
Holdout Accuracy          | 67.11%               | 68.75%
Holdout Brier             | 0.2115                 | 0.2048
Holdout Log Loss          | 0.6033                 | 0.5891
-----------------------------------------------------------------------
Walk-Forward ROI          | +4.31%               | +11.20%
Total Bets                | 105                  | 98
Win Rate                  | 58.10%               | 62.24%
Max Drawdown              | -22.45%              | -15.88%
================================================================================
```

## Historical Promotion Blocker: Runtime Source of Truth

This was the core risk when the promotion plan was first written. The older startup path could leave an old `xgboost_model.pkl` on the persistent volume and keep the bot on the prior model even after a new deploy.

The current hosted runtime no longer depends on that behavior. It serves the canonical aliases from `/app/models`, bootstraps a runtime production manifest, and was verified live on March 25, 2026. The manual verification steps below remain useful as an audit checklist, but they are no longer the only protection against a stale model on Railway.

## Safe Promotion Plan

Follow these steps exactly to ensure the new model goes live correctly.

### 1. Archive Current Live Model (V5)

Before overwriting anything, create a versioned backup of the current V5 production model to ensure a clean rollback path.

```bash
# Create a dedicated archive directory for the V5 model
mkdir -p models/archive/v5_fullfit_production_20260325

# Move the V5 artifacts into the archive
mv models/archive/v5_fullfit_production/xgboost_model.pkl models/archive/v5_fullfit_production_20260325/
mv models/archive/v5_fullfit_production/full_live_contract_v5_fullfit_spec.json models/archive/v5_fullfit_production_20260325/
```

### 2. Promote Candidate Artifacts (V6) to Canonical Paths

Copy the new V6 model and its spec file to the canonical paths that the application loads by default.

```bash
# Copy the V6 model to the canonical production path
cp models/xgboost_model.pkl models/xgboost_model.pkl

# Also copy the spec file for completeness
cp models/full_live_contract_v6_tuned_spec.json models/
```
*(Note: If `models/xgboost_model.pkl` is already the V6 candidate, this step is just for confirmation.)*

### 3. Update Production Manifest

Update `models/current_production_model.json` to formally declare the V6 spec as the source of truth. This is a step towards the "explicit bundle" contract recommended in the Railway handoff.

### 4. Deploy to Railway

Commit the changes (the new archive directory, updated manifest, etc.) and deploy to Railway.

### 5. Manually Verify & Sync the Railway Volume

**This is the most critical step.** After the deployment finishes:

1.  Open a shell into your Railway service.
2.  Navigate to the persistent models directory (e.g., `/data/models` or the path specified by `RAILWAY_VOLUME_MOUNT_PATH`).
3.  List the files (`ls -l`). Check the timestamp and size of `xgboost_model.pkl`.
4.  **If it is the old V5 model**, the deployment did not update it. You must manually sync it:
    ```bash
    # In the Railway shell
    rm /data/models/xgboost_model.pkl
    # Now, restart the service from the Railway dashboard.
    # The startup script will now see the model is missing and copy the new V6 version from the app image.
    ```

### 6. Post-Deployment Verification

1.  Check the `/readyz` endpoint on the dashboard to ensure the service is healthy.
2.  Inspect the application logs. Look for startup messages confirming that the loaded model spec is `full_live_contract_v6_tuned`.
3.  Run a live prediction (`python -m src.bot predict`) and check the operator logs. The provenance fields should now reflect the V6 model spec.

## Recommendation

Unless there are other business or operational reasons to wait, we should proceed with this promotion plan. The model's superiority is evident.

Following this, the highest priority should be implementing the architectural improvements from `RAILWAY_RUNTIME_SOURCE_OF_TRUTH_HANDOFF_2026-03-23.md` to make future promotions automated and less risky.
