# Dynamic Schema (Schema-Agnostic) Implementation Plan

Transitioning ChurnGuard from a fixed IBM Telco schema to a platform capable of ingesting *any* company's data (e.g., Verizon, AT&T) requires moving from a "Global Model" architecture to a "Tenant-Specific Model" architecture.

Here is the step-by-step roadmap to implement this massive architectural upgrade.

## Phase 1: Storage and Schema Registry
Before we can train models on different schemas, we need to know what those schemas are.
- [x] **Create a Schema Table:** Add a new table in Postgres (`company_schemas`) that stores a JSON definition of each company's expected columns, categorized into `numerical` and `categorical`.
- [x] **Artifact Isolation:** Change the `models/` directory structure to store tenant-specific models. For example, `models/{company_id}_model.joblib` and `models/{company_id}_pipeline.joblib`.
- [x] **Update Upload Endpoints:** When a new client uploads a CSV via the `/predict-csv` or `/learning/upload` routes, parse the CSV headers dynamically. If it's their first upload, automatically infer the schema, separate numeric vs categorical columns, and save it to the database.

## Phase 2: Dynamic Training Pipeline (`src/data/pipeline.py`)
Currently, `NUMERICAL_FEATURES` and `CATEGORICAL_FEATURES` are hardcoded lists.
- [x] **Remove Hardcoded Lists:** Update `pipeline.py` to accept dynamic lists of numerical and categorical feature names when building the `ColumnTransformer`.
- [x] **Tenant-Specific Training Script:** Update `train_and_save.py` so it loops through all `company_id`s in the database. For each company:
    1. Fetch their specific `learning_rows`.
    2. Fetch their stored schema JSON.
    3. Generate a dynamic `ColumnTransformer` specifically for their data shape.
    4. Train a Logistic Regression model.
    5. Save the output to `models/{company_id}_model.joblib`.

## Phase 3: The Model Router (`src/api/model_loader.py`)
Currently, the FastAPI app loads a single `churn_model.joblib` into memory on startup as a Singleton.
- [x] **Lazy Loading:** Refactor `model_loader.py` so it holds a dictionary of models in memory, keyed by `company_id` (e.g., `{"verizon": ModelObj, "acme": ModelObj}`).
- [x] **Model Hot-Swapping:** When a user hits `/predict`, read their `company_id` from their JWT session token, and route the prediction request to their specific model. If the model isn't in memory yet, load it from disk dynamically.

## Phase 4: Dynamic Frontend UI (`web/app.js` & `web/index.html`)
The sidebar form is currently hardcoded with HTML `<fieldset>` tags for IBM's specific features (e.g., "Streaming TV", "Multiple Lines").
- [x] **New `/schema` API:** Add a `GET /admin/schema` endpoint that returns the logged-in company's schema JSON.
- [x] **Javascript Form Generator:** On page load, delete the hardcoded sidebar HTML. Have `app.js` call `/schema` and dynamically construct the form:
    - If a feature is `categorical` (and unique values < 10), render an HTML `<select>` dropdown populated with the possible values.
    - If a feature is `numerical`, render an HTML `<input type="number">`.
- [x] **Dynamic Feature Importance:** Ensure the SHAP-style local feature importance charts gracefully handle the new, dynamically generated feature names.

## Phase 5: Client Onboarding Workflow
- [x] **Upload First CSV:** When an Admin uses the "Platform Admin" panel to create a new workspace, force the new client to upload a "Seed CSV" upon their first login. This seed file will define their schema and cold-start their personalized model.

## Phase 6: Production Scaling & Advanced MLOps (Pending / Next Steps)
- [ ] **Automated Background Model Retraining:**
    - Transition manual training triggers (`python scripts/train_and_save.py`) to an automated background process (e.g., Celery, Cron) that triggers once a tenant has collected 500+ new records in the learning queue database table.
- [ ] **Dynamic "What-If" Simulation Tooling:**
    - Build a baseline comparison tool inside the What-If tab. Allows users to load an existing scored client, clone their profile, modify critical factors (e.g., contract type, tech support) and view risk-reduction side-by-side.
- [ ] **Data Drift Detection & Monitoring:**
    - Add a pipeline step to calculate PSI (Population Stability Index) or KS test on numerical features by comparing live inference query logs against initial training datasets, rendering drift warning indicators in the Dashboard.
- [ ] **Multi-Tenant Model Performance Comparison:**
    - Design a centralized administrative visual panel comparing the ROC-AUC, F1-scores, and training history across all tenants to help admins identify sub-performing models.

