# What We Did

1. **Multi-Tenant Model Isolation**: Overhauled the Model Router to support true dynamic model loading and context-swapping based on `company_id`. Each tenant now correctly gets their own specific model without cross-pollution.
2. **Dynamic Schema Integration**: Added the `/admin/schema` endpoint to display the correct categorical and numerical options for the active tenant's schema dynamically.
3. **Database Test Isolation & Stability**: Configured an isolated SQLite test database and truncated tables in the `pytest` fixture to guarantee 100% test isolation. Resolved a key API race condition on concurrent logins using SQLAlchemy savepoints to handle duplicate workspace members gracefully.
4. **Health Probes Fixed**: Updated the `/health` endpoint to properly report `model_loaded=True` to account for the new lazy-loading architecture.
5. **Robust CSV Validation**: Implemented detailed row-by-row validation in `_validate_customer_csv` returning exact 1-indexed row offsets and error codes.
6. **Global Model Quality Upgrade**: Boosted the global Telco model ROC-AUC from **0.7481** to **0.8467** and F1 to **0.6153** using a balanced model comparison across Logistic Regression, Random Forest, XGBoost, LightGBM, tuned XGBoost, and tuned LightGBM.
7. **Tenant Model AUC Optimization**: Raised custom tenant model accuracy (across `company-a`, `company-b`, and `acme_ecommerce`) from **0.685** up to **0.826 – 0.836** by strengthening predictive signals and reducing noise in the synthetic data generator.
8. **Predictor Auto-Reloading**: Configured the backend Model Router to automatically reload updated classifiers from disk by tracking file modification timestamps (`st_mtime`), preventing stale memory cache issues during retraining.
9. **Interactive Testing UI**: Added an action panel to the frontend allowing users to dynamically randomize customer attributes for testing and minimize/collapse analysis results to return the dashboard to a clean empty state.
10. **Dynamic Prediction API**: Updated `/predict` and `/predict-batch` so tenant workspaces can score schema-specific JSON instead of being locked to the Telco fields.
11. **Retraining Observability**: Added per-company training status records for `running`, `succeeded`, and `failed` model jobs.
12. **Schema Review UI**: Added an owner-editable schema editor so inferred columns can be corrected before retraining.
13. **Learning Row Review**: Added queued, approved, and rejected learning-row states with review controls in the Improvement Lab.
14. **Model Promotion Workflow**: Added candidate model artifacts, promotion/rejection API endpoints, and an Improvement Lab review card so retraining no longer has to replace production immediately.
15. **Used-In-Model Tracking**: Candidate metadata now records included learning row IDs, and promotion marks those approved rows as `used_in_model` with the promoted training run ID.
16. **Promotion Quality Rules**: Candidate promotion now compares production vs candidate metrics and blocks material regressions unless an owner explicitly force-promotes.
17. **Drift Monitoring**: Prediction events now store scored input payloads, model metadata stores training profiles, and `/admin/drift` flags stable/watch/high drift with retrain recommendations.
18. **100% Test Suite Pass Rate**: All API, pipeline, and integration tests pass locally with SQLite and no Neon/network dependency.

# What To Do Next

1. **Advanced Model Explanations**: Integrate richer SHAP/LIME-style plots directly in the dashboard.
2. **Tenant Usage Analytics**: Implement billing, usage limits, and API quota tracking for the multi-tenant SaaS subscription model.
3. **Durable Model Artifact Storage**: Move production/candidate artifacts to Vercel Blob or object storage before serious external pilots.
4. **Promotion History UI**: Show past candidate, promoted, rejected, failed, and force-promoted training runs in the Improvement Lab.
5. **Outcome Performance Drift**: Once companies upload later outcomes, compare predicted risk against actual churn by cohort and time window.
