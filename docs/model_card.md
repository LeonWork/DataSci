# Customer Churn Prediction Model Card

## Model Overview

| Field | Value |
|-------|-------|
| Model type | RandomForestClassifier |
| Task | Binary classification: churn vs. no churn |
| Dataset | IBM Telco Customer Churn |
| Target | `Churn` |
| Version | 1.0.0 |
| Current status | Pilot model, not yet company-calibrated |

## Intended Use

ChurnGuard scores telecom-style customer accounts so retention teams can prioritize outreach, inspect the strongest churn drivers, and export a focused list of high-risk customers.

This model is appropriate for retention triage and portfolio demonstrations. It is not appropriate for credit, employment, insurance, healthcare, or other high-stakes eligibility decisions.

## Training Data

| Property | Value |
|----------|-------|
| Source | IBM Telco Customer Churn sample dataset |
| Size | 7,043 customers |
| Positive class | `Churn = Yes` |
| Churn rate | About 26.5% |
| Feedback rows | Optional reviewed company rows from `data/company_feedback.csv` and `data/external/labeled_churn/` |

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

The deployed artifact in `models/model_meta.json` reports:

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.7897 |
| F1 | 0.5428 |
| Accuracy | 0.7381 |
| Precision | 0.4611 |
| Recall | 0.6596 |

The next ML upgrade should compare Random Forest, Logistic Regression, XGBoost, and LightGBM under the same reproducible split, then choose a threshold based on the business goal. A retention team may prefer higher recall even if precision falls, because missing true churners can be more expensive than contacting extra customers.

## Explainability

Every prediction includes the top feature impacts used by the dashboard. The current implementation uses model explanation values to show whether each factor increases or reduces the churn score for that customer.

Typical risk drivers in this dataset include short tenure, month-to-month contracts, fiber optic service, high monthly charges, and missing support/security add-ons.

## Limitations

- The model is trained on a public telecom dataset and is not yet calibrated to a specific company.
- Prediction quality may drift if pricing, products, contracts, or customer behavior change.
- Demographic fields such as gender and senior-citizen status require fairness review before production use.
- Uploaded company feedback is queued for retraining, but the current app still needs review, approval, registry, and promotion controls before automated production retraining.

## Roadmap

1. Add a model comparison report with ROC-AUC, PR-AUC, F1, precision, recall, and calibration.
2. Add a model registry that separates production and candidate models.
3. Add reviewed learning-row states: `queued`, `approved_for_training`, and `used_in_model`.
4. Add drift monitoring and a `retrain_recommended` signal.
5. Add company-level calibration after enough labeled outcomes are collected.
