# Customer Churn Prediction — Model Card

## Model Overview

| Field | Value |
|-------|-------|
| **Model type** | XGBoost Classifier (gradient boosted trees) |
| **Task** | Binary classification — churn (1) vs. no-churn (0) |
| **Dataset** | IBM Telco Customer Churn (7,043 rows, 21 features) |
| **Target** | `Churn` — whether a customer cancelled service within the billing period |
| **Version** | 1.0.0 |
| **Last trained** | Week 3–4 (see MLflow `churn-baseline` experiment) |

---

## Intended Use

**Primary use case:** Identify telecom customers at high risk of churning so retention teams can intervene proactively.

**Intended users:** Customer success teams, data analysts, product managers.

**Out-of-scope uses:** Credit scoring, employment decisions, or any high-stakes decisions where model errors carry significant harm.

---

## Training Data

| Property | Value |
|----------|-------|
| Source | [Kaggle: blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Size | 7,043 customers |
| Churn rate | ~26.5% (imbalanced) |
| Train / test split | 80% / 20% (stratified) |
| Missing values | 11 rows with blank `TotalCharges` → imputed as 0 |

---

## Features

### Engineered Features (Week 2)

| Feature | Description |
|---------|-------------|
| `clv` | Customer Lifetime Value = tenure × MonthlyCharges |
| `avg_monthly_charge` | TotalCharges / max(tenure, 1) |
| `charge_increase` | MonthlyCharges − avg_monthly_charge |
| `contract_stability` | Month-to-month=0, One year=1, Two year=2 |
| `service_bundle_score` | Count of add-on services subscribed |
| `has_internet` | 1 if any internet service, else 0 |
| `tenure_band` | new / mid / loyal / champion lifecycle bucket |
| `is_high_value` | 1 if CLV ≥ 75th percentile |

### Preprocessing
- Numerical: median imputation → StandardScaler
- Categorical: mode imputation → OneHotEncoder (drop_first=True)

---

## Model Performance

| Metric | Score (test set) |
|--------|-----------------|
| ROC-AUC | ~0.84–0.86 |
| F1 Score | ~0.60–0.65 |
| Accuracy | ~0.80–0.82 |
| Precision | ~0.66–0.72 |
| Recall | ~0.55–0.60 |

> **Note:** Due to class imbalance (~27% churn), AUC and F1 are the primary metrics. Accuracy alone is misleading.

---

## Limitations & Risks

- **Static snapshot:** The model was trained on historical data. Customer behaviour and pricing plans change over time — retrain quarterly.
- **Class imbalance:** Churn is the minority class (~27%). The model may underestimate churn in extreme cases.
- **Feature drift:** If the product catalogue changes (new service types, new contract structures), features like `service_bundle_score` may drift.
- **Proxy discrimination:** `SeniorCitizen`, `gender`, and `Partner` are included as features. Monitor for disparate impact across demographic groups.

---

## Explainability

Every prediction is accompanied by SHAP values computed via `shap.TreeExplainer`. The top contributing factors are returned in the API response and displayed in the Streamlit dashboard.

**Key drivers identified in the training data (typical ranking):**
1. `tenure` — shorter tenure → higher churn risk
2. `Contract` — Month-to-month significantly higher risk than Two year
3. `InternetService_Fiber optic` — higher cost, higher churn rate
4. `TotalCharges` / `clv` — lower total spend → newer, riskier customers
5. `TechSupport` / `OnlineSecurity` — lack of protective services → higher risk

---

## API

```
POST /predict
Content-Type: application/json

{
  "customerID": "C-123",
  "tenure": 2,
  "Contract": "Month-to-month",
  "MonthlyCharges": 85.5,
  ...
}
```

Response includes: `churn_probability`, `risk_level`, `top_factors` (SHAP).

---

## Reproducibility

```bash
# 1. Download dataset
kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip

# 2. Train and save model
python scripts/train_and_save.py

# 3. Run API
uvicorn src.api.main:app --reload --port 8000

# 4. Run dashboard
streamlit run dashboard/app.py
```

All experiment runs are tracked in MLflow (`mlruns/` directory).
