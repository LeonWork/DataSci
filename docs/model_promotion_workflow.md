# Model Promotion Workflow

## Why This Is Today's Focus

ChurnGuard can already accept tenant-specific schemas, store learning rows, review training data, and retrain models. The next SaaS-grade step is to make retraining safe.

Companies should not have their live churn model replaced automatically every time new data is uploaded. A new training run should create a candidate model, show whether it is better than production, and only become active after an owner promotes it.

## Product Goal

Build a review gate between retraining and production serving:

1. The current production model keeps serving predictions.
2. A retraining run writes candidate artifacts.
3. The owner compares production metrics against candidate metrics.
4. The owner promotes the candidate, or rejects it.
5. Promotion replaces the active production artifacts and reloads the predictor.

This gives the project a credible MLOps story for companies and interviews: model updates are measured, reversible, and tenant-aware.

## User Experience

In the Improvement Lab, owners should see a Model Promotion card with:

- current production model family
- production ROC-AUC, PR-AUC, F1, precision, recall, and calibration/Brier score where available
- candidate model family
- candidate metrics beside production metrics
- visible metric deltas
- candidate status: no candidate, training, ready, promoted, rejected, or failed
- Promote Candidate and Reject Candidate actions

Analysts and viewers may inspect the status, but only owners can promote or reject.

## Backend Design

Production artifacts stay in their current locations:

- `models/{company_id}_model.joblib`
- `models/{company_id}_pipeline.joblib`
- `models/{company_id}_model_meta.json`

Candidate artifacts use separate names:

- `models/{company_id}_candidate_model.joblib`
- `models/{company_id}_candidate_pipeline.joblib`
- `models/{company_id}_candidate_model_meta.json`

The training run table stores the candidate lifecycle:

- `running`
- `candidate_ready`
- `promoted`
- `rejected`
- `failed`

The existing `model_training_runs` fields are enough for this first version:

- `company_id`
- `status`
- `model_family`
- `metrics_json`
- `artifact_paths_json`
- `error_message`
- `started_at`
- `finished_at`

## API Design

Add these owner-aware endpoints:

- `GET /admin/model/candidate`
  - returns production metadata, latest candidate metadata, latest training run, and whether a candidate can be promoted

- `POST /admin/model/promote`
  - copies candidate artifacts into production artifact paths
  - marks the run as `promoted`
  - clears the in-memory predictor for that company so the next prediction reloads the promoted model

- `POST /admin/model/reject`
  - marks the candidate run as `rejected`
  - removes candidate artifacts when they exist
  - leaves production untouched

## Training Script Behavior

The training script should support a candidate mode:

```bash
python scripts/train_and_save.py company-a --candidate
python scripts/train_and_save.py all --candidate
```

Candidate mode writes only candidate artifacts and marks the run `candidate_ready`. Production mode remains available for local/global baseline training.

## Acceptance Criteria

- A candidate training run does not overwrite production model files.
- `GET /admin/model/candidate` shows production and candidate metrics.
- Promoting a candidate replaces production artifacts.
- Rejecting a candidate leaves production artifacts unchanged.
- Failed training runs leave the current production model serving.
- Only owners can promote or reject candidates.
- Tests cover candidate creation metadata, promote, reject, and permissions.

## Later Follow-Up

## Used-In-Model Tracking

This workflow also closes the loop from uploaded outcomes to reviewed rows to promoted models:

- approved learning rows remain `approved_for_training` while a candidate is under review
- candidate metadata stores the exact `learning_row_ids` included in the training run
- promotion marks only those included rows as `used_in_model`
- used rows store the promoted `model_training_run_id`
- rejected candidates leave approved rows available for a future candidate

This keeps the training audit trail honest. If a company asks why a model changed, we can point to the specific reviewed learning rows that were included.

## Promotion Quality Rules

Candidate promotion now includes a quality gate. The gate compares production metrics against candidate metrics and blocks promotion when the candidate has a material regression.

The first version checks:

- ROC-AUC cannot drop by more than `0.010`
- PR-AUC cannot drop by more than `0.020`
- F1 cannot drop by more than `0.020`
- Brier score cannot worsen by more than `0.020`
- balanced quality score cannot drop by more than `0.010`

The balanced quality score uses the same product-oriented idea as training selection:

- ROC-AUC for ranking quality
- PR-AUC for churn-class usefulness
- F1 for decision balance
- Brier score for calibration

If a candidate fails, the API returns `409 Conflict` with the failed quality checks. Owners can still force-promote when there is a business reason, but the UI makes that an explicit confirmation step.

## Drift Monitoring

ChurnGuard now records the input payload for each prediction event so the active company can compare recent scored customers against the model's training profile.

New model metadata includes a `training_profile` with:

- numeric feature means, standard deviations, and missing rates
- categorical top-value distributions and missing rates
- training row count

The drift monitor checks recent prediction inputs and reports:

- `stable`
- `watch`
- `high`
- `warming_up`
- `unavailable`

The first version uses simple, explainable drift signals:

- numeric mean shift measured in training standard deviations
- categorical distribution distance
- top-feature drift scores
- overall score from the highest-drift features

When drift is high, the API returns `retrain_recommended: true`. This is intentionally conservative and transparent for pilot customers; later versions can add PSI, KL divergence, label drift, and outcome-based model performance drift.
