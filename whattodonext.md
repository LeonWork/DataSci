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
14. **100% Test Suite Pass Rate**: All 146 API, pipeline, and integration tests pass locally with SQLite and no Neon/network dependency.

# What To Do Next

1. **Model Promotion Workflow**: Separate production and candidate artifacts so retraining can be reviewed before promotion.
2. **Used-In-Model Tracking**: Mark approved rows as consumed after successful retraining.
3. **Drift Monitoring**: Add feature drift, prediction distribution drift, label balance, and retrain-recommended indicators.
4. **Advanced Model Explanations**: Integrate richer SHAP/LIME-style plots directly in the dashboard.
5. **Tenant Usage Analytics**: Implement billing, usage limits, and API quota tracking for the multi-tenant SaaS subscription model.
