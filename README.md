# 📉 Customer Churn Prediction — MLOps Pipeline

> **Predict which telecom customers will churn — before they do.**  
> End-to-end ML project with feature engineering, model explainability, a FastAPI backend, and a custom web app.

---

## 🎯 Problem Statement

Customer churn is a critical business problem in the telecom industry. Losing a customer costs 5–25× more than retaining one. This project builds a production-grade ML pipeline to:

1. **Predict** which customers are at risk of churning (binary classification)
2. **Explain** *why* they're likely to churn using SHAP values
3. **Serve** predictions through a FastAPI REST endpoint
4. **Analyze** single customers and CSV uploads in a custom browser UI

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
├── web/                # Custom HTML/CSS/JS app served by FastAPI
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
| 2 | Feature Engineering & Data Pipeline | ✅ Complete |
| 3 | Model Development & Experiment Tracking | ✅ Complete |
| 4 | Model Interpretation & Hyperparameter Tuning | ✅ Complete |
| 5 | FastAPI + Docker Deployment | ✅ Complete |
| 6 | Custom Web App + Portfolio Polish | ✅ Complete |

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
| **Web App** | FastAPI StaticFiles, HTML, CSS, JavaScript |
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
GET  /               — Web app
POST /auth/signup    — Create local user
POST /auth/login     — Authenticate local user
POST /predict        — Single customer churn probability
POST /predict-csv    — Score customer CSV uploads
POST /learning/upload — Store labeled CSV rows for future retraining
GET  /health         — API health check
```

## 🔐 Web App Login

Run the custom web app:

```bash
uvicorn src.api.main:app --reload --port 8000
```

Then open:

```text
http://localhost:8000
```

Development fallback credentials:

```text
Username: admin
Password: admin123
```

The app also supports self-service sign up. New local accounts are saved in
`data/app_users.json` with bcrypt password hashes. That file is ignored by Git.

Override them with environment variables before starting the app:

```bash
export CHURNGUARD_USERNAME="admin"
export CHURNGUARD_PASSWORD="choose-a-local-password"
uvicorn src.api.main:app --reload --port 8000
```

For deployments, prefer `CHURNGUARD_PASSWORD_HASH` with a bcrypt hash instead of a plain-text password.

## 🧠 Learning From Company CSVs

Unlabeled company CSVs can be scored immediately through the Batch CSV analysis panel.
To improve the model, upload labeled CSVs that include a known `Churn` column through
the Learning Queue panel. Those rows are stored in `data/company_feedback.csv`, which is
ignored by Git. The next run of `python scripts/train_and_save.py` automatically includes
that labeled feedback in training.

You can also add reviewed labeled datasets as CSV files in:

```text
data/external/labeled_churn/
```

Those files must use the same customer columns as the IBM Telco dataset plus `Churn`.
They are included automatically the next time you run `python scripts/train_and_save.py`.

---

## 📝 License

MIT — free to use for learning and portfolio purposes.
