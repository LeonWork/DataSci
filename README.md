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
pip install -r requirements-dev.txt  # optional: notebooks, tests, training tools
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
| **ML** | Scikit-learn Random Forest, XGBoost/LightGBM comparison track, SHAP-style explanations |
| **MLOps** | MLflow, DVC (optional) |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Web App** | FastAPI StaticFiles, HTML, CSS, JavaScript |
| **Infrastructure** | Docker, Docker Compose |
| **Deployment** | Vercel-ready FastAPI/static app |

---

## 📈 Results

| Model | F1-Score | ROC-AUC | Precision | Recall |
|-------|----------|---------|-----------|--------|
| **RandomForestClassifier (current)** | 0.5428 | 0.7897 | 0.4611 | 0.6596 |
| Logistic Regression (next comparison) | — | — | — | — |
| XGBoost (next comparison) | — | — | — | — |
| LightGBM (next comparison) | — | — | — | — |

---

## 🔍 Key Insights

*(Updated after Week 4)*

Current top churn drivers usually include:
- Month-to-month contracts
- Short tenure
- Fiber optic service
- High monthly charges
- Missing support, security, or backup services

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
GET  /admin/summary  — Protected admin metrics for the workspace
GET  /health         — API health check
```

## SaaS Pilot Architecture

```mermaid
flowchart LR
    A["Admin user"] --> B["FastAPI app"]
    B --> C["RandomForest model artifact"]
    B --> D["Pilot product store"]
    D --> E["Prediction events"]
    D --> F["CSV upload batches"]
    D --> G["Queued learning rows"]
    B --> H["Static dashboard UI"]
```

The pilot now records prediction events, CSV scoring batches, and learning rows in a local SQLite product store. This is intentionally shaped like the future Postgres schema so we can move to Neon when real company uploads need durable cloud storage. Until then, local storage avoids unnecessary spending.

## 🔐 Web App Login

Run the custom web app:

```bash
uvicorn src.api.main:app --reload --port 8000
```

Then open:

```text
http://localhost:8000
```

By default the app is a private admin workspace. Configure credentials in `.env`
or with environment variables before using the dashboard:

```text
CHURNGUARD_USERNAME="admin"
CHURNGUARD_PASSWORD="choose-a-local-password"
CHURNGUARD_SESSION_SECRET="choose-a-long-random-secret"
CHURNGUARD_ENABLE_SIGNUP=false
```

For invite-only account creation, set:

```text
CHURNGUARD_SIGNUP_CODE="private-invite-code"
```

For open local signup, set:

```bash
export CHURNGUARD_ENABLE_SIGNUP=true
export CHURNGUARD_PASSWORD="choose-a-local-password"
uvicorn src.api.main:app --reload --port 8000
```

New accounts are saved in the application database with bcrypt password hashes.
Local development uses `data/churnguard.sqlite3`; production can use Neon/Postgres
by setting `DATABASE_URL`.

```text
DATABASE_URL="postgresql://USER:PASSWORD@HOST/dbname?sslmode=require"
CHURNGUARD_COMPANY_ID="your-company"
CHURNGUARD_COMPANY_NAME="Your Company"
```

For legacy local JSON users only, set `CHURNGUARD_USE_LEGACY_JSON_USERS=true`.

The old local demo fallback (`admin` / `admin123`) is disabled unless
`CHURNGUARD_ENABLE_DEFAULT_ADMIN=true` is explicitly set.

For deployments, prefer `CHURNGUARD_PASSWORD_HASH` with a bcrypt hash instead of a plain-text password.
Set `CHURNGUARD_SESSION_SECRET` in production so browser sessions are signed with
a deployment-specific secret.

After sign-in, the app sends a bearer token to protected API endpoints. Prediction,
batch scoring, and learning queue routes require that signed session token.

## 🧠 Learning From Company CSVs

Unlabeled company CSVs can be scored immediately through the Batch CSV analysis panel.
To improve the model, upload labeled CSVs that include a known `Churn` column through
the Learning Queue panel. Those rows are stored in `data/company_feedback.csv`, which is
ignored by Git. The next run of `python scripts/train_and_save.py` automatically includes
that labeled feedback in training. The API validates `Churn` labels before saving rows and
exposes the current queue at:

```text
GET /learning/status
```

The admin dashboard also shows stored workspace totals: predictions, high-risk accounts,
CSV upload batches, queued learning rows, current model type, and current training AUC.

On Vercel, local SQLite and `/tmp` training CSV files are temporary function storage.
For production learning, connect durable Postgres/Neon first, then add Vercel Blob for
raw CSV archives or model artifacts when file retention becomes necessary.

You can also add reviewed labeled datasets as CSV files in:

```text
data/external/labeled_churn/
```

Those files must use the same customer columns as the IBM Telco dataset plus `Churn`.
They are included automatically the next time you run `python scripts/train_and_save.py`.

## Portfolio Demo Flow

1. Sign in to the private workspace.
2. Use **Load demo file** in Batch CSV analysis to score sample customers instantly.
3. Export high-risk accounts with **Export high-risk**.
4. Upload labeled rows with a `Churn` column to the Learning Queue.
5. Show the admin summary metrics as proof that the app stores operational history.

## Privacy and Security Notes

- Do not upload real company PII until durable storage, access controls, and deletion flows are configured.
- Keep `.env`, `data/app_users.json`, feedback CSVs, and local database files out of Git.
- Use hashed passwords in production via `CHURNGUARD_PASSWORD_HASH`.
- Early pilots should be manually onboarded; open self-serve signup and billing should wait until tenant isolation and audit logs are complete.

## Business Case

Retention teams can use ChurnGuard to turn raw customer exports into a prioritized save list. The dashboard explains why each customer is risky, supports CSV batch scoring, exports high-risk accounts for outreach, and queues known outcomes so the model can improve as company-specific data arrives.

## Product Roadmap

### Current Release: Private Pilot and Portfolio

This release is designed to look and behave like a real internal company tool:

- Private admin login with bearer-token protected APIs
- Single-customer churn scoring with feature explanations
- CSV batch scoring for customer exports
- High-risk customer export for retention outreach
- Learning queue for labeled company rows
- Admin summary cards for prediction volume, high-risk accounts, uploads, queued learning rows, and model metrics
- SQLAlchemy storage layer with local SQLite and Neon/Postgres support
- Database-backed app users with bcrypt password hashes
- Workspace-aware sessions with `company_id` and role claims
- Tenant-scoped prediction, upload, learning, and admin-summary records
- Workspace member tracking for owner/analyst/viewer readiness
- CSV validation reports with row-level errors for scoring and learning uploads
- Corrected model documentation for the current RandomForestClassifier artifact

### Immediate Next Steps

The app is Postgres-ready and now has upload validation/error reporting. The next
step is to create the Neon database and connect it to local/Vercel environments.

Recommended order:

1. Create a Neon Postgres project.
2. Set `DATABASE_URL` locally and in Vercel.
3. Run the app once so SQLAlchemy creates the production tables.
4. Seed your owner account into Postgres using the existing admin environment credentials.
5. Add admin-created company onboarding instead of environment-variable workspace setup.
6. Add deployment smoke tests for protected APIs, database connection, and dashboard loading.
7. Add basic audit logs for login, upload, prediction, export, and learning-row review events.

### Neon Setup Checklist

You do not need the Neon CLI. When ready:

1. Create a Neon account and project in the browser.
2. Copy the pooled Postgres connection string.
3. Add it to local `.env`:

```text
DATABASE_URL="postgresql://USER:PASSWORD@HOST/dbname?sslmode=require"
```

4. Add the same `DATABASE_URL` in Vercel environment variables.
5. Run a local smoke test, then a Vercel production smoke test.

The app can keep using local SQLite until this checklist is done.

### ML Upgrade Track

The model should improve through disciplined evaluation before spending on heavy AI infrastructure:

1. Compare Random Forest, Logistic Regression, XGBoost, and LightGBM on the same split.
2. Track ROC-AUC, PR-AUC, F1, precision, recall, and calibration.
3. Tune decision thresholds for business goals, especially recall-heavy retention workflows.
4. Add a model registry with production and candidate model states.
5. Require reviewed learning rows before retraining.
6. Monitor prediction distribution drift, feature drift, label balance, and post-outcome performance.

### Later SaaS and Enterprise Work

- Stripe billing by prediction volume, upload size, and workspace tier
- Password reset and optional OAuth through Clerk/Auth0
- Audit logs and rate limiting
- Company data deletion and retention controls
- Vercel Blob for raw CSV archives and model artifacts
- Monitoring for API errors, latency, and uptime
- Onboarding checklist, downloadable CSV template, saved customer segments, and cohort insights

### Spending Guidance

Spend first on Neon/Postgres when storing real company uploads and prediction history in production. Buy a domain once the product name is final. Add Vercel Blob only when raw CSV/model artifact retention matters. Do not pay for heavy AI compute yet; the biggest near-term gains come from better evaluation, threshold tuning, cleaner company feedback, and a retraining approval workflow.

---

## 📝 License

MIT — free to use for learning and portfolio purposes.
