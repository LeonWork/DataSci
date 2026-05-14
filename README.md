# 📉 Customer Churn Prediction — MLOps Pipeline

> **Predict which telecom customers will churn — before they do.**  
> End-to-end ML project with feature engineering, model explainability, a REST API, and a live dashboard.

---

## 🎯 Problem Statement

Customer churn is a critical business problem in the telecom industry. Losing a customer costs 5–25× more than retaining one. This project builds a production-grade ML pipeline to:

1. **Predict** which customers are at risk of churning (binary classification)
2. **Explain** *why* they're likely to churn using SHAP values
3. **Serve** predictions through a FastAPI REST endpoint
4. **Monitor** model performance over time via a Streamlit dashboard

**Target metric:** F1-Score (balancing precision and recall for an imbalanced dataset)

---

## 📊 Dataset

**Telco Customer Churn** — IBM Sample Dataset via Kaggle  
- ~7,000 customer records  
- 21 features (demographics, account info, services used)  
- Target: `Churn` (Yes / No) — ~26% positive rate  

---

## 🏗️ Project Architecture

```
DataSci/
├── data/
│   ├── raw/            # Original, immutable Kaggle data
│   ├── processed/      # Cleaned & feature-engineered datasets
│   └── external/       # Supplementary data sources
├── notebooks/          # Jupyter notebooks (EDA, modeling, interpretation)
├── src/
│   ├── data/           # Data loading & preprocessing pipeline
│   ├── models/         # Training, evaluation, and saving logic
│   ├── api/            # FastAPI application
│   └── utils/          # Shared helpers (logging, config, metrics)
├── tests/              # Unit and integration tests
├── models/             # Serialized model artifacts (.pkl, .joblib)
├── docs/               # Architecture diagrams, model cards
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🗓️ 6-Week Roadmap

| Week | Focus | Status |
|------|-------|--------|
| 1 | EDA & Data Quality | ✅ Complete |
| 2 | Feature Engineering & Data Pipeline | 🔄 In Progress |
| 3 | Model Development & Experiment Tracking | ⏳ Upcoming |
| 4 | Model Interpretation & Hyperparameter Tuning | ⏳ Upcoming |
| 5 | FastAPI + Docker Deployment | ⏳ Upcoming |
| 6 | Streamlit Dashboard + Portfolio Polish | ⏳ Upcoming |

---

## ⚙️ Setup

### 1. Clone & create environment

```bash
git clone <your-repo-url>
cd DataSci
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
# Option A: Kaggle CLI
kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip

# Option B: Manual download
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place WA_Fn-UseC_-Telco-Customer-Churn.csv in data/raw/
```

### 3. Run EDA notebook

```bash
jupyter lab notebooks/01_eda.ipynb
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Core** | Python 3.11, Pandas, NumPy, Scikit-learn |
| **ML** | XGBoost, LightGBM, SHAP, Imbalanced-learn |
| **MLOps** | MLflow, DVC (optional) |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Dashboard** | Streamlit |
| **Infrastructure** | Docker, Docker Compose |
| **Deployment** | Render / Railway / Hugging Face Spaces |

---

## 📈 Results

*(Updated after Week 3)*

| Model | F1-Score | ROC-AUC | Precision | Recall |
|-------|----------|---------|-----------|--------|
| Logistic Regression (baseline) | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |
| **LightGBM (champion)** | — | — | — | — |

---

## 🔍 Key Insights

*(Updated after Week 4)*

Top churn drivers (from SHAP analysis):
- TBD after model interpretation

---

## 📬 API Reference

*(Available after Week 5)*

```
POST /predict        — Single customer churn probability
POST /batch_predict  — Batch CSV predictions
GET  /health         — API health check
```

---

## 📝 License

MIT — free to use for learning and portfolio purposes.
