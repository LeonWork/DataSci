"""
dashboard/app.py
----------------
ChurnGuard AI — Customer Churn Prediction Dashboard

Loads the trained model directly (no API server needed) and provides
a premium, interactive interface for churn risk analysis.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import bcrypt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib

from src.data.features import engineer_all_features
from src.data.preprocess import drop_ids, handle_missing
from src.utils.logger import get_logger

logger = get_logger("dashboard")

MODEL_PATH    = ROOT / "models" / "churn_model.joblib"
PIPELINE_PATH = ROOT / "models" / "churn_pipeline.joblib"
META_PATH     = ROOT / "models" / "model_meta.json"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    }

    /* Header */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0;
    }
    .sub-header {
        color: #8b949e;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 2rem;
    }

    /* Cards */
    .metric-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 20px 24px;
        backdrop-filter: blur(10px);
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: rgba(139, 92, 246, 0.5); }

    .risk-high   { border-left: 4px solid #ef4444 !important; }
    .risk-medium { border-left: 4px solid #f59e0b !important; }
    .risk-low    { border-left: 4px solid #10b981 !important; }

    .risk-badge-high   { background: rgba(239,68,68,0.15);   color: #ef4444;   border: 1px solid rgba(239,68,68,0.3);   border-radius: 6px; padding: 4px 12px; font-weight: 600; }
    .risk-badge-medium { background: rgba(245,158,11,0.15);  color: #f59e0b;   border: 1px solid rgba(245,158,11,0.3);  border-radius: 6px; padding: 4px 12px; font-weight: 600; }
    .risk-badge-low    { background: rgba(16,185,129,0.15);  color: #10b981;   border: 1px solid rgba(16,185,129,0.3);  border-radius: 6px; padding: 4px 12px; font-weight: 600; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(13, 17, 23, 0.95) !important;
        border-right: 1px solid rgba(48, 54, 61, 0.6);
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #c9d1d9 !important;
        font-size: 0.82rem !important;
    }

    /* Divider */
    hr { border-color: rgba(48, 54, 61, 0.6) !important; }

    /* Factor bars */
    .factor-bar-pos { background: linear-gradient(90deg, #ef4444, #f87171); height: 8px; border-radius: 4px; }
    .factor-bar-neg { background: linear-gradient(90deg, #10b981, #34d399); height: 8px; border-radius: 4px; }

    /* Section titles */
    .section-title {
        color: #e6edf3;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(48,54,61,0.6);
    }

    /* Recommendation card */
    .reco-card {
        background: rgba(139, 92, 246, 0.08);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 10px;
        padding: 16px;
        margin-top: 8px;
    }
    .reco-item { color: #c9d1d9; font-size: 0.9rem; margin: 6px 0; }
    /* Login Screen */
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 80vh;
    }
    .login-card {
        background: rgba(22, 27, 34, 0.9);
        border: 1px solid rgba(48, 54, 61, 1);
        border-radius: 16px;
        padding: 40px;
        width: 400px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
        text-align: center;
    }
    .login-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .login-subtitle {
        color: #8b949e;
        font-size: 0.9rem;
        margin-bottom: 32px;
    }
</style>
""", unsafe_allow_html=True)


# ── Authentication ────────────────────────────────────────────────────────────

ADMIN_HASH = b"$2b$12$amCWoXmqjip9GRVhRnmNJ.DBvO1ayDKDMK7aOceeiXAXP4kWdmS4m"

def check_password(password: str) -> bool:
    return bcrypt.checkpw(password.encode(), ADMIN_HASH)

def login_screen():
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<p class="login-title">🔮 ChurnGuard AI</p>', unsafe_allow_html=True)
        st.markdown('<p class="login-subtitle">Enterprise Intelligence Platform</p>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            user = st.text_input("Username", placeholder="admin")
            pwd  = st.text_input("Password", type="password", placeholder="••••••••")
            submit = st.form_submit_button("Sign In", use_container_width=True)
            
            if submit:
                if user == "admin" and check_password(pwd):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    if not MODEL_PATH.exists():
        return None, None, [], {}
    model    = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    meta     = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    features = meta.get("feature_names", [])
    metrics  = meta.get("metrics", {})
    return model, pipeline, features, metrics


# ── Inference ─────────────────────────────────────────────────────────────────

def run_prediction(customer_dict: dict, model, pipeline, feature_names: list):
    df = pd.DataFrame([customer_dict])
    df = handle_missing(df)
    df = engineer_all_features(df)
    df = drop_ids(df)
    df = df.drop(columns=["Churn"], errors="ignore")
    X = pipeline.transform(df)

    prob = float(model.predict_proba(X)[0, 1])

    # SHAP
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        vals = sv[0]
        indices = np.argsort(np.abs(vals))[::-1][:8]
        factors = []
        for i in indices:
            name = feature_names[i] if i < len(feature_names) else f"f{i}"
            factors.append({"feature": name, "value": float(vals[i])})
    except Exception:
        factors = []

    return prob, factors


# ── Gauge chart ───────────────────────────────────────────────────────────────

def make_gauge(prob: float, risk: str) -> go.Figure:
    color_map = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
    color = color_map[risk]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        domain={"x": [0, 1], "y": [0, 1]},
        number={"suffix": "%", "font": {"size": 52, "color": color, "family": "Inter"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#8b949e",
                "tickfont": {"color": "#8b949e", "size": 11},
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],  "color": "rgba(16,185,129,0.12)"},
                {"range": [30, 65], "color": "rgba(245,158,11,0.12)"},
                {"range": [65, 100],"color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": round(prob * 100, 1),
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=10, l=20, r=20),
        height=260,
    )
    return fig


# ── Factor chart ──────────────────────────────────────────────────────────────

def make_factor_chart(factors: list[dict]) -> go.Figure:
    if not factors:
        return None
    
    # Feature name cleanup and better truncation
    names = [f["feature"].replace("_", " ").title() for f in factors]
    vals  = [f["value"] for f in factors]
    
    # Deep, premium colors
    colors = ["#f87171" if v > 0 else "#34d399" for v in vals] # Softer red/green

    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        text=[f" {v:+.3f}" if v > 0 else f"{v:+.3f} " for v in vals],
        textposition="outside",
        textfont={"color": "#c9d1d9", "size": 13, "family": "Inter Semibold"},
        cliponaxis=False,
    ))

    # Calculate x-axis range with a bit of padding for the labels
    max_val = max([abs(v) for v in vals]) if vals else 1
    padding = max_val * 0.25
    x_range = [-max_val - padding, max_val + padding]

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=10, l=10, r=10), # Tighter margins
        height=max(320, len(factors) * 40),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(48,54,61,0.3)",
            zeroline=True,
            zerolinecolor="rgba(139,92,246,0.5)",
            tickfont={"color": "#8b949e", "size": 10},
            title=dict(text="Impact on Churn Risk", font={"color": "#8b949e", "size": 11}),
            range=x_range,
            anchor="free",
            position=0,
        ),
        yaxis=dict(
            tickfont={"color": "#e6edf3", "size": 12},
            autorange="reversed",
            side="left",
            automargin=True, # Handles long labels automatically
        ),
        bargap=0.35,
        showlegend=False,
        font=dict(family="Inter", color="#c9d1d9"),
    )

    # Add background shapes for red/green zones
    fig.add_vrect(x0=0, x1=x_range[1], fillcolor="rgba(239,68,68,0.03)", line_width=0)
    fig.add_vrect(x0=x_range[0], x1=0, fillcolor="rgba(16,185,129,0.03)", line_width=0)

    return fig


# ── Recommendations ───────────────────────────────────────────────────────────

def get_recommendations(customer: dict, risk: str) -> list[str]:
    recos = []
    if risk == "High":
        if customer["Contract"] == "Month-to-month":
            recos.append("💼 Offer a discounted 1-year or 2-year contract")
        if customer["tenure"] < 12:
            recos.append("🎁 Send a loyalty welcome package within 30 days")
        if customer["TechSupport"] == "No" and customer["InternetService"] != "No":
            recos.append("🛠️ Proactively enroll in free Tech Support trial")
        if customer["MonthlyCharges"] > 80:
            recos.append("💰 Review billing — consider a bundle discount")
        recos.append("📞 Assign a dedicated account manager for proactive outreach")
    elif risk == "Medium":
        recos.append("📧 Send personalised engagement email with loyalty rewards")
        recos.append("⭐ Invite to customer feedback program")
        if customer["StreamingTV"] == "No" and customer["InternetService"] != "No":
            recos.append("📺 Offer a free 3-month StreamingTV trial")
    else:
        recos.append("✅ Customer is stable — maintain quality of service")
        recos.append("📈 Candidate for upsell (premium bundle or fibre upgrade)")
    return recos


# ── Demo presets ──────────────────────────────────────────────────────────────

HIGH_RISK_PRESET = {
    "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 171.0,
}
LOW_RISK_PRESET = {
    "gender": "Male", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 79.35, "TotalCharges": 4760.1,
}


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar_form() -> dict:
    with st.sidebar:
        st.markdown("## 👤 Customer Profile")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔴 High Risk", use_container_width=True):
                st.session_state.preset = "high"
        with col2:
            if st.button("🟢 Low Risk", use_container_width=True):
                st.session_state.preset = "low"

        preset = st.session_state.get("preset", None)
        p = HIGH_RISK_PRESET if preset == "high" else (LOW_RISK_PRESET if preset == "low" else {})

        st.divider()
        st.markdown("**Demographics**")
        gender        = st.selectbox("Gender",           ["Male", "Female"],          index=["Male","Female"].index(p.get("gender","Male")))
        senior        = st.selectbox("Senior Citizen",   ["No", "Yes"],               index=["No","Yes"].index(p.get("SeniorCitizen","No")))
        partner       = st.selectbox("Partner",          ["Yes", "No"],               index=["Yes","No"].index(p.get("Partner","No")))
        dependents    = st.selectbox("Dependents",       ["Yes", "No"],               index=["Yes","No"].index(p.get("Dependents","No")))

        st.divider()
        st.markdown("**Account**")
        tenure        = st.slider("Tenure (months)",     0, 72,                       value=p.get("tenure", 12))
        contract      = st.selectbox("Contract",         ["Month-to-month","One year","Two year"], index=["Month-to-month","One year","Two year"].index(p.get("Contract","Month-to-month")))
        paperless     = st.selectbox("Paperless Billing",["Yes","No"],                index=["Yes","No"].index(p.get("PaperlessBilling","Yes")))
        payment       = st.selectbox("Payment Method",   ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"], index=["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"].index(p.get("PaymentMethod","Electronic check")))

        st.divider()
        st.markdown("**Services**")
        phone         = st.selectbox("Phone Service",    ["Yes","No"],                index=["Yes","No"].index(p.get("PhoneService","Yes")))
        multiline     = st.selectbox("Multiple Lines",   ["Yes","No","No phone service"], index=["Yes","No","No phone service"].index(p.get("MultipleLines","No")))
        internet      = st.selectbox("Internet Service", ["DSL","Fiber optic","No"],  index=["DSL","Fiber optic","No"].index(p.get("InternetService","DSL")))

        no_inet = "No internet service" if internet == "No" else None
        sec     = st.selectbox("Online Security",  ["Yes","No"] if not no_inet else ["No internet service"], index=0 if no_inet else ["Yes","No"].index(p.get("OnlineSecurity","No") if p.get("OnlineSecurity","No") in ["Yes","No"] else "No"))
        backup  = st.selectbox("Online Backup",    ["Yes","No"] if not no_inet else ["No internet service"], index=0 if no_inet else ["Yes","No"].index(p.get("OnlineBackup","No") if p.get("OnlineBackup","No") in ["Yes","No"] else "No"))
        protect = st.selectbox("Device Protection",["Yes","No"] if not no_inet else ["No internet service"], index=0 if no_inet else ["Yes","No"].index(p.get("DeviceProtection","No") if p.get("DeviceProtection","No") in ["Yes","No"] else "No"))
        tech    = st.selectbox("Tech Support",     ["Yes","No"] if not no_inet else ["No internet service"], index=0 if no_inet else ["Yes","No"].index(p.get("TechSupport","No") if p.get("TechSupport","No") in ["Yes","No"] else "No"))
        tv      = st.selectbox("Streaming TV",     ["Yes","No"] if not no_inet else ["No internet service"], index=0 if no_inet else ["Yes","No"].index(p.get("StreamingTV","No") if p.get("StreamingTV","No") in ["Yes","No"] else "No"))
        movies  = st.selectbox("Streaming Movies", ["Yes","No"] if not no_inet else ["No internet service"], index=0 if no_inet else ["Yes","No"].index(p.get("StreamingMovies","No") if p.get("StreamingMovies","No") in ["Yes","No"] else "No"))

        st.divider()
        st.markdown("**Charges**")
        monthly = st.number_input("Monthly Charges ($)", 0.0, 999.0, value=float(p.get("MonthlyCharges", 65.0)), step=0.5)
        total   = st.number_input("Total Charges ($)",   0.0, 99999.0, value=float(p.get("TotalCharges", monthly * tenure if tenure else monthly)), step=1.0)




        return {
            "customerID": "DASHBOARD-USER",
            "gender": gender, "SeniorCitizen": senior,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": multiline,
            "InternetService": internet,
            "OnlineSecurity": no_inet or sec, "OnlineBackup": no_inet or backup,
            "DeviceProtection": no_inet or protect, "TechSupport": no_inet or tech,
            "StreamingTV": no_inet or tv, "StreamingMovies": no_inet or movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }


# ── Batch analysis ───────────────────────────────────────────────────────────

BATCH_REQUIRED_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

def batch_tab(model, pipeline, feature_names: list, train_metrics: dict):
    st.markdown('<p class="section-title">📂 Batch Customer Analysis</p>', unsafe_allow_html=True)
    st.write(
        "Upload a CSV containing multiple customer records. "
        "ChurnGuard AI will score every row and return a ranked risk table."
    )

    # Template download
    template_df = pd.DataFrame([HIGH_RISK_PRESET, LOW_RISK_PRESET])
    template_df.insert(0, "customerID", ["CUST-001", "CUST-002"])
    st.download_button(
        "⬇️ Download CSV Template",
        data=template_df.to_csv(index=False),
        file_name="churnguard_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is None:
        st.info("👆 Upload a CSV to begin batch scoring.")
        return

    raw = pd.read_csv(uploaded)
    missing = [c for c in BATCH_REQUIRED_COLS if c not in raw.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        return

    st.success(f"✅ Loaded **{len(raw):,}** customer records.")

    with st.spinner(f"Scoring {len(raw):,} customers…"):
        results = []
        for _, row in raw.iterrows():
            cust = row.to_dict()
            cust.setdefault("customerID", "UNKNOWN")
            try:
                prob, factors = run_prediction(cust, model, pipeline, feature_names)
                risk = "High" if prob >= 0.65 else ("Medium" if prob >= 0.30 else "Low")
                top = factors[0]["feature"] if factors else ""
                results.append({
                    "customerID": cust.get("customerID", ""),
                    "churn_probability_pct": round(prob * 100, 1),
                    "risk_tier": risk,
                    "top_driver": top,
                    "tenure": cust.get("tenure"),
                    "contract": cust.get("Contract"),
                    "monthly_charges": cust.get("MonthlyCharges"),
                })
            except Exception:
                pass

    results_df = pd.DataFrame(results).sort_values("churn_probability_pct", ascending=False)

    # KPI strip
    n_high   = (results_df["risk_tier"] == "High").sum()
    n_medium = (results_df["risk_tier"] == "Medium").sum()
    n_low    = (results_df["risk_tier"] == "Low").sum()
    avg_prob = results_df["churn_probability_pct"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔴 High Risk",   f"{n_high:,}",   f"{n_high/len(results_df)*100:.0f}% of base")
    k2.metric("🟡 Medium Risk", f"{n_medium:,}", f"{n_medium/len(results_df)*100:.0f}% of base")
    k3.metric("🟢 Low Risk",    f"{n_low:,}",    f"{n_low/len(results_df)*100:.0f}% of base")
    k4.metric("Avg Churn Prob", f"{avg_prob:.1f}%")

    # Distribution histogram
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">📊 Risk Score Distribution</p>', unsafe_allow_html=True)
    hist_colors = [
        "#10b981" if p < 30 else ("#f59e0b" if p < 65 else "#ef4444")
        for p in results_df["churn_probability_pct"]
    ]
    fig_hist = go.Figure(go.Histogram(
        x=results_df["churn_probability_pct"],
        nbinsx=20,
        marker_color="#667eea",
        marker_line_width=0,
        opacity=0.85,
    ))
    fig_hist.add_vrect(x0=0,  x1=30, fillcolor="rgba(16,185,129,0.07)",  line_width=0)
    fig_hist.add_vrect(x0=30, x1=65, fillcolor="rgba(245,158,11,0.07)",  line_width=0)
    fig_hist.add_vrect(x0=65, x1=100,fillcolor="rgba(239,68,68,0.07)",   line_width=0)
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(t=10,b=10,l=10,r=10),
        xaxis=dict(title="Churn Probability (%)", tickfont={"color":"#8b949e"}, gridcolor="rgba(48,54,61,0.5)"),
        yaxis=dict(title="# Customers", tickfont={"color":"#8b949e"}, gridcolor="rgba(48,54,61,0.5)"),
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    # Ranked table
    st.markdown('<p class="section-title">🏷️ Ranked Customer Risk Table</p>', unsafe_allow_html=True)
    def color_risk(val):
        c = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}.get(val, "")
        return f"color: {c}; font-weight:600"
    styled = results_df.style.applymap(color_risk, subset=["risk_tier"])
    st.dataframe(styled, use_container_width=True, height=400)

    # Bulk export
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        "⬇️ Download Scored Results (CSV)",
        data=results_df.to_csv(index=False),
        file_name="churnguard_batch_results.csv",
        mime="text/csv",
        use_container_width=False,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not st.session_state.get("authenticated"):
        login_screen()
        return

    # Header
    head_col, out_col = st.columns([5, 1])
    with head_col:
        st.markdown('<p class="main-header">🔮 ChurnGuard AI</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Real-time Customer Churn Risk Analysis & Explainability</p>', unsafe_allow_html=True)
    with out_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    # Load model
    model, pipeline, feature_names, train_metrics = load_artifacts()

    if model is None:
        st.error(
            "⚠️ **Model not found.** Run the training script first:\n\n"
            "```bash\npython scripts/train_and_save.py\n```",
        )
        st.stop()

    tab_single, tab_batch = st.tabs(["🔮 Single Customer", "📂 Batch Analysis"])

    # Sidebar (outside tabs — Streamlit sidebars are global)
    customer = sidebar_form()
    with st.sidebar:
        st.divider()
        predict_btn = st.button("🔮 Analyse Churn Risk", use_container_width=True, type="primary")
    auto_predict = "preset" in st.session_state

    with tab_batch:
        batch_tab(model, pipeline, feature_names, train_metrics)

    with tab_single:
        if predict_btn or auto_predict:
            with st.spinner("Analysing…"):
                prob, factors = run_prediction(customer, model, pipeline, feature_names)

            risk = "High" if prob >= 0.65 else ("Medium" if prob >= 0.30 else "Low")
            risk_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

            col_gauge, col_detail = st.columns([1, 1.6], gap="large")

            with col_gauge:
                st.markdown('<p class="section-title">Churn Probability</p>', unsafe_allow_html=True)
                st.plotly_chart(make_gauge(prob, risk), use_container_width=True, config={"displayModeBar": False})
                badge_class = f"risk-badge-{risk.lower()}"
                st.markdown(
                    f'<div style="text-align:center;margin-top:-10px;">'
                    f'<span class="{badge_class}">{risk_emoji[risk]} {risk} Risk</span></div>',
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
                recos = get_recommendations(customer, risk)
                st.markdown('<p class="section-title">💡 Recommended Actions</p>', unsafe_allow_html=True)
                st.markdown('<div class="reco-card">', unsafe_allow_html=True)
                for r in recos:
                    st.markdown(f'<p class="reco-item">{r}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_detail:
                st.markdown('<p class="section-title">Feature Impact (SHAP)</p>', unsafe_allow_html=True)
                if factors:
                    st.plotly_chart(make_factor_chart(factors), use_container_width=True, config={"displayModeBar": False})
                    st.caption("🔴 Red bars push toward churn · 🟢 Green bars reduce churn risk")
                else:
                    st.info("SHAP values unavailable.")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<p class="section-title">📊 Model Performance</p>', unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("AUC",       f"{train_metrics.get('roc_auc', 0):.3f}")
                m2.metric("F1",        f"{train_metrics.get('f1', 0):.3f}")
                m3.metric("Precision", f"{train_metrics.get('precision', 0):.3f}")
                m4.metric("Recall",    f"{train_metrics.get('recall', 0):.3f}")
                st.caption(
                    "**AUC** = ability to rank churners above non-churners (1.0 = perfect, 0.5 = random). "
                    "**F1** = balance of precision and recall. "
                    "Industry baseline AUC for telco churn: ~0.70–0.80."
                )

            with st.expander("📋 Full Customer Profile", expanded=False):
                display = {k: v for k, v in customer.items() if k != "customerID"}
                st.dataframe(pd.DataFrame([display]).T.rename(columns={0: "Value"}), use_container_width=True)

            # Export
            import json as _json
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-title">📥 Export Prediction Report</p>', unsafe_allow_html=True)
            export_data = {
                "customer_profile": {k: v for k, v in customer.items() if k != "customerID"},
                "churn_probability_pct": round(prob * 100, 2),
                "risk_tier": risk,
                "top_shap_factors": factors[:5],
                "recommended_actions": recos,
                "model_performance": train_metrics,
            }
            export_df = pd.DataFrame([{
                **{k: v for k, v in customer.items() if k != "customerID"},
                "churn_probability_pct": round(prob * 100, 2),
                "risk_tier": risk,
                "top_factor_1": factors[0]["feature"] if len(factors) > 0 else "",
                "top_factor_2": factors[1]["feature"] if len(factors) > 1 else "",
                "top_factor_3": factors[2]["feature"] if len(factors) > 2 else "",
                "recommendations": " | ".join(recos),
            }])
            dl1, dl2, _ = st.columns([1, 1, 1.5]) # Slightly better spacing
            with dl1:
                st.download_button(
                    label="⬇️ CSV Report",
                    data=export_df.to_csv(index=False),
                    file_name="churnguard_prediction.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with dl2:
                st.download_button(
                    label="⬇️ JSON Report",
                    data=_json.dumps(export_data, indent=2),
                    file_name="churnguard_prediction.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("👈 **Fill in the customer profile** in the sidebar, then click **Analyse Churn Risk** — or use the demo presets (🔴 High Risk / 🟢 Low Risk).")
            c1, c2, c3 = st.columns(3)
            c1.metric("Model", "XGBoost", "Tuned")
            c2.metric("Dataset", "IBM Telco", "7,043 customers")
            c3.metric("Features", "30+", "Engineered")

            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📖 How to Use ChurnGuard AI", expanded=True):
                st.markdown("""
### Getting Started in 3 Steps

**Step 1 — Enter a Customer Profile**
Use the sidebar on the left to fill in a customer's details:
- **Demographics**: Gender, senior citizen status, partner, dependents.
- **Account**: Tenure, contract type, billing method, payment method.
- **Services**: Phone, internet, streaming, security, support add-ons.
- **Charges**: Monthly and total charges billed to date.

> 💡 **Tip:** Use the **🔴 High Risk** or **🟢 Low Risk** preset buttons to instantly load a demo customer.

---

**Step 2 — Run the Analysis**
Click **🔮 Analyse Churn Risk** in the sidebar. The model will:
1. Preprocess and engineer features from the profile.
2. Run an XGBoost classifier trained on 7,043 IBM Telco customers.
3. Return a **Churn Probability (0–100%)** and assign a **Risk Tier**.

| Risk Tier | Probability | Meaning |
|-----------|-------------|---------|
| 🟢 **Low** | < 30% | Customer is stable. Low intervention needed. |
| 🟡 **Medium** | 30–65% | Early churn signals. Engage proactively. |
| 🔴 **High** | ≥ 65% | Significant risk. Immediate action recommended. |

---

**Step 3 — Interpret the Results**

- **Gauge** — Exact churn probability, colour-coded by risk tier.
- **SHAP Chart** — 🔴 Red = churn driver · 🟢 Green = protective factor. Longer bar = stronger influence.
- **Recommended Actions** — CRM-ready retention steps tailored to this customer.
- **Export** — Download CSV or JSON to share with your team.
- **Batch Analysis tab** — Upload a CSV to score hundreds of customers at once.

---

> ⚠️ Predictions are probabilistic and should supplement human judgment, not replace it.
                """)

    # Footer
    st.divider()
    st.markdown(
        '<p style="color:#484f58; font-size:0.78rem; text-align:center;">'
        "ChurnGuard AI · Customer Retention Intelligence Platform · "
        "Powered by XGBoost + SHAP · Built on FastAPI &amp; Streamlit"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
