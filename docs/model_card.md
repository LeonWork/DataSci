# Customer Churn Prediction Model Card

## Model Overview

| Field | Value |
|-------|-------|
| Model type | XGBoost selected from Logistic Regression, Random Forest, XGBoost, and LightGBM comparison |
| Task | Binary classification: churn vs. no churn |
| Dataset | IBM Telco Customer Churn plus compatible labeled Telco rows |
| Target | `Churn` |
| Version | 1.0.0 |
| Current status | Pilot model with tenant-specific adaptation support |

## Intended Use

ChurnGuard scores telecom-style customer accounts so retention teams can prioritize outreach, inspect the strongest churn drivers, and export a focused list of high-risk customers.

This model is appropriate for retention triage and portfolio demonstrations. It is not appropriate for credit, employment, insurance, healthcare, or other high-stakes eligibility decisions.

## Training Data

| Property | Value |
|----------|-------|
| Source | IBM Telco Customer Churn sample dataset |
| Size | 57,043 compatible Telco-style rows in the current global training run |
| Positive class | `Churn = Yes` |
| Churn rate | About 26.5% |
| Feedback rows | Optional reviewed company rows from Postgres learning rows and `data/external/labeled_churn/` |

## Features

The model uses original customer attributes plus engineered account signals:

| Feature | Description |
|---------|-------------|
| `clv` | `tenure * MonthlyCharges` |
| `avg_monthly_charge` | `TotalCharges / max(tenure, 1)` |
| `charge_increase` | `MonthlyCharges - avg_monthly_charge` |
| `contract_stability` | Month-to-month=0, One year=1, Two year=2 |
| `service_bundle_score` | Count of subscribed add-on services |
| `has_internet` | Whether the customer has internet service |
| `tenure_band` | Lifecycle bucket for tenure |
| `is_high_value` | Whether CLV is in a high-value range |

## Current Performance

The current global artifact in `models/global_model_meta.json` reports:

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.8467 |
| PR-AUC | 0.6340 |
| Brier score | 0.1573 |
| F1 | 0.6153 |
| Accuracy | 0.7690 |
| Precision | 0.5117 |
| Recall | 0.7716 |

The training script now saves a model comparison report and threshold report. The current production selection uses a balanced pilot score across ROC-AUC, PR-AUC, F1, and calibration/Brier quality rather than chasing one metric alone.

## Explainability

Every prediction includes the top feature impacts used by the dashboard. The current implementation uses model explanation values to show whether each factor increases or reduces the churn score for that customer.

Typical risk drivers in this dataset include short tenure, month-to-month contracts, fiber optic service, high monthly charges, and missing support/security add-ons.

## Limitations

- The global model is trained on public/compatible telecom-style data and is not yet calibrated to a specific real company.
- Prediction quality may drift if pricing, products, contracts, or customer behavior change.
- Demographic fields such as gender and senior-citizen status require fairness review before production use.
- Uploaded company feedback is queued for retraining, and the app records training status, but it still needs approval and production/candidate promotion controls before automated production retraining.

## Roadmap

1. Add a schema editor for correcting inferred tenant schemas before retraining.
2. Expand the model registry so production and candidate models are separated.
3. Add reviewed learning-row states: `queued`, `approved_for_training`, and `used_in_model`.
4. Add drift monitoring and a `retrain_recommended` signal.
5. Add company-level calibration after enough labeled outcomes are collected.
