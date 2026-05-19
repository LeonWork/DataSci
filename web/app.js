const authView = document.querySelector("#authView");
const dashboardView = document.querySelector("#dashboardView");
const signinTab = document.querySelector("#signinTab");
const signupTab = document.querySelector("#signupTab");
const signinForm = document.querySelector("#signinForm");
const signupForm = document.querySelector("#signupForm");
const inviteCodeField = document.querySelector("#inviteCodeField");
const authMessage = document.querySelector("#authMessage");
const signedInUser = document.querySelector("#signedInUser");
const workspaceBadge = document.querySelector("#workspaceBadge");
const logoutButton = document.querySelector("#logoutButton");
const toggleSidebar = document.querySelector("#toggleSidebar");
const openLab = document.querySelector("#openLab");
const dashboardPage = document.querySelector("#dashboardPage");
const labPage = document.querySelector("#labPage");
const backToDashboard = document.querySelector("#backToDashboard");
const profileForm = document.querySelector("#profileForm");
const emptyState = document.querySelector("#emptyState");
const results = document.querySelector("#results");
const probabilityValue = document.querySelector("#probabilityValue");
const riskBadge = document.querySelector("#riskBadge");
const probabilityMarker = document.querySelector("#probabilityMarker");
const riskMargin = document.querySelector("#riskMargin");
const oddsValue = document.querySelector("#oddsValue");
const factorList = document.querySelector("#factorList");
const recommendations = document.querySelector("#recommendations");
const featureDetail = document.querySelector("#featureDetail");
const detailTitle = document.querySelector("#detailTitle");
const detailValue = document.querySelector("#detailValue");
const detailDirection = document.querySelector("#detailDirection");
const detailImpactChart = document.querySelector("#detailImpactChart");
const scenarioDemo = document.querySelector("#scenarioDemo");
const detailNotes = document.querySelector("#detailNotes");
const closeDetail = document.querySelector("#closeDetail");
const modelType = document.querySelector("#modelType");
const modelAuc = document.querySelector("#modelAuc");
const featureCount = document.querySelector("#featureCount");
const driverCount = document.querySelector("#driverCount");
const protectorCount = document.querySelector("#protectorCount");
const strongestFactor = document.querySelector("#strongestFactor");
const batchCsv = document.querySelector("#batchCsv");
const batchSummary = document.querySelector("#batchSummary");
const riskChart = document.querySelector("#riskChart");
const batchTable = document.querySelector("#batchTable");
const learningCsv = document.querySelector("#learningCsv");
const learningStatus = document.querySelector("#learningStatus");
const totalPredictions = document.querySelector("#totalPredictions");
const highRiskPredictions = document.querySelector("#highRiskPredictions");
const queuedLearningRows = document.querySelector("#queuedLearningRows");
const loadDemoCsv = document.querySelector("#loadDemoCsv");
const exportHighRisk = document.querySelector("#exportHighRisk");
const adminSection = document.querySelector("#adminSection");
const onboardForm = document.querySelector("#onboardForm");
const onboardMessage = document.querySelector("#onboardMessage");
let lastCustomer = null;
let lastPrediction = null;
let lastBatchRows = [];

const highRiskPreset = {
  gender: "Female",
  SeniorCitizen: "No",
  Partner: "No",
  Dependents: "No",
  tenure: 2,
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "Yes",
  StreamingMovies: "Yes",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: 85.5,
  TotalCharges: 171,
};

const lowRiskPreset = {
  gender: "Male",
  SeniorCitizen: "No",
  Partner: "Yes",
  Dependents: "Yes",
  tenure: 60,
  PhoneService: "Yes",
  MultipleLines: "Yes",
  InternetService: "DSL",
  OnlineSecurity: "Yes",
  OnlineBackup: "Yes",
  DeviceProtection: "Yes",
  TechSupport: "Yes",
  StreamingTV: "Yes",
  StreamingMovies: "Yes",
  Contract: "Two year",
  PaperlessBilling: "No",
  PaymentMethod: "Bank transfer (automatic)",
  MonthlyCharges: 79.35,
  TotalCharges: 4760.1,
};

function setAuthMode(mode) {
  const signup = mode === "signup";
  signinTab.classList.toggle("active", !signup);
  signupTab.classList.toggle("active", signup);
  signinForm.classList.toggle("hidden", signup);
  signupForm.classList.toggle("hidden", !signup);
  authMessage.textContent = "";
}

function setSession(username) {
  localStorage.setItem("churnguard_user", username);
  signedInUser.textContent = `Signed in as ${username}`;
  authView.classList.add("hidden");
  dashboardView.classList.remove("hidden");
  loadSchema();
  loadModelInfo();
  loadLearningStatus();
  loadAdminSummary();
  loadWorkspace();
  loadTenants();
}

function setAuthenticatedSession(username, token, companyId, role) {
  localStorage.setItem("churnguard_token", token);
  localStorage.setItem("churnguard_user", username);
  if (companyId) localStorage.setItem("churnguard_company_id", companyId);
  if (role) localStorage.setItem("churnguard_role", role);
  signedInUser.textContent = `Signed in as ${username}`;
  authView.classList.add("hidden");
  dashboardView.classList.remove("hidden");
  loadSchema();
  loadModelInfo();
  loadLearningStatus();
  loadAdminSummary();
  loadWorkspace();
  loadTenants();
}

function clearSession() {
  localStorage.removeItem("churnguard_user");
  localStorage.removeItem("churnguard_token");
  localStorage.removeItem("churnguard_company_id");
  localStorage.removeItem("churnguard_role");
  dashboardView.classList.add("hidden");
  authView.classList.remove("hidden");
  lastBatchRows = [];
  if (exportHighRisk) exportHighRisk.disabled = true;
  if (workspaceBadge) workspaceBadge.textContent = "";
}

function setSidebarHidden(hidden) {
  document.body.classList.toggle("sidebar-hidden", hidden);
  toggleSidebar.textContent = hidden ? "›" : "‹";
  toggleSidebar.setAttribute("aria-label", hidden ? "Show sidebar" : "Hide sidebar");
  localStorage.setItem("churnguard_sidebar_hidden", hidden ? "1" : "0");
}

function showPage(page) {
  const lab = page === "lab";
  dashboardPage.classList.toggle("hidden", lab);
  labPage.classList.toggle("hidden", !lab);
}

function formToObject(form) {
  return Object.fromEntries(new FormData(form).entries());
}

function authHeaders(extra = {}) {
  const token = localStorage.getItem("churnguard_token");
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) clearSession();
    throw new Error(formatApiError(data.detail, "Request failed."));
  }
  return data;
}

async function postFile(url, file) {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(url, { method: "POST", headers: authHeaders(), body });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) clearSession();
    const error = new Error(formatApiError(data.detail, "Upload failed."));
    error.detail = data.detail;
    throw error;
  }
  return data;
}

function formatApiError(detail, fallback) {
  if (!detail) return fallback;
  if (typeof detail === "string") return detail;
  if (detail.message) return detail.message;
  return fallback;
}

function renderValidationErrors(container, detail) {
  const errors = Array.isArray(detail?.errors) ? detail.errors : [];
  if (!errors.length) {
    container.textContent = formatApiError(detail, "Upload failed.");
    return;
  }
  container.innerHTML = `
    <div class="validation-report">
      <strong>${detail.message || "CSV validation failed."}</strong>
      <span>${detail.error_count || errors.length} issue${(detail.error_count || errors.length) === 1 ? "" : "s"} found${detail.truncated ? "; showing first 50" : ""}.</span>
      <ul>
        ${errors.map((issue) => `
          <li>
            <span>${issue.row ? `Row ${issue.row}` : "File"}${issue.column ? ` · ${issue.column}` : ""}</span>
            ${issue.message}
          </li>
        `).join("")}
      </ul>
    </div>
  `;
}

async function loadLearningStatus() {
  if (!learningStatus) return;
  try {
    const response = await fetch("/learning/status", { headers: authHeaders() });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (response.status === 401) clearSession();
      throw new Error(data.detail || "Could not load learning queue.");
    }
    learningStatus.textContent = `${data.stored_rows} labeled rows queued for retraining. Run ${data.retraining_command} after review.`;
  } catch (error) {
    learningStatus.textContent = error.message;
  }
}

async function loadAdminSummary() {
  if (!totalPredictions || !highRiskPredictions || !queuedLearningRows) return;
  try {
    const response = await fetch("/admin/summary", { headers: authHeaders() });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (response.status === 401) clearSession();
      throw new Error(data.detail || "Could not load admin summary.");
    }
    totalPredictions.textContent = data.total_predictions ?? "--";
    highRiskPredictions.textContent = data.high_risk_predictions ?? "--";
    queuedLearningRows.textContent = data.learning_rows_queued ?? "--";
  } catch {
    totalPredictions.textContent = "--";
    highRiskPredictions.textContent = "--";
    queuedLearningRows.textContent = "--";
  }
}

async function loadWorkspace() {
  if (!workspaceBadge) return;
  try {
    const response = await fetch("/admin/workspace", { headers: authHeaders() });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (response.status === 401) clearSession();
      throw new Error(data.detail || "Could not load workspace.");
    }
    const role = localStorage.getItem("churnguard_role") || "member";
    workspaceBadge.textContent = `${data.company_name} · ${role}`;
  } catch {
    const companyId = localStorage.getItem("churnguard_company_id");
    const role = localStorage.getItem("churnguard_role");
    workspaceBadge.textContent = companyId && role ? `${companyId} · ${role}` : "";
  }
  const role = localStorage.getItem("churnguard_role");
  if (adminSection) {
    adminSection.classList.toggle("hidden", role !== "owner");
  }
}

async function loadAuthConfig() {
  try {
    const response = await fetch("/auth/config");
    const data = await response.json().catch(() => ({}));
    const signupEnabled = Boolean(data.signup_enabled);
    const inviteRequired = Boolean(data.signup_requires_invite);
    signupTab.classList.toggle("hidden", !signupEnabled);
    inviteCodeField.classList.toggle("hidden", !inviteRequired);
    inviteCodeField.querySelector("input").required = inviteRequired;
    if (!signupEnabled) {
      setAuthMode("signin");
      authMessage.textContent = data.message || "";
    } else {
      setAuthMode("signup");
    }
  } catch {
    setAuthMode("signin");
  }
}

function applyPreset(preset) {
  Object.entries(preset).forEach(([key, value]) => {
    const field = profileForm.elements[key];
    if (field) field.value = value;
  });
}

function customerFromForm() {
  const customer = formToObject(profileForm);
  return {
    customerID: "WEB-USER",
    ...customer,
    tenure: Number(customer.tenure),
    MonthlyCharges: Number(customer.MonthlyCharges),
    TotalCharges: Number(customer.TotalCharges),
  };
}

function recommendationText(customer, risk) {
  const recs = [];
  if (risk === "High") {
    if (customer.Contract === "Month-to-month") recs.push("Offer a discounted annual contract migration.");
    if (customer.tenure < 12) recs.push("Send an early-tenure loyalty offer within 30 days.");
    if (customer.TechSupport === "No" && customer.InternetService !== "No") recs.push("Enroll the customer in a free technical support trial.");
    if (customer.MonthlyCharges > 80) recs.push("Review billing and test a bundle discount.");
    recs.push("Assign proactive outreach to a retention owner.");
  } else if (risk === "Medium") {
    recs.push("Send a personalized loyalty reward sequence.");
    recs.push("Invite the customer to a feedback and service review.");
    if (customer.StreamingTV === "No" && customer.InternetService !== "No") recs.push("Offer a short streaming add-on trial.");
  } else {
    recs.push("Maintain service quality and monitor at the normal cadence.");
    recs.push("Consider a premium bundle or fiber upgrade offer.");
  }
  return recs;
}

function renderPrediction(customer, prediction) {
  lastCustomer = customer;
  lastPrediction = prediction;
  const probability = Math.round(prediction.churn_probability * 1000) / 10;
  probabilityValue.textContent = `${probability}%`;
  probabilityValue.style.color = riskColor(prediction.risk_level);
  riskBadge.textContent = `${prediction.risk_level} risk`;
  riskBadge.className = `risk-badge risk-${prediction.risk_level.toLowerCase()}`;
  probabilityMarker.style.left = `calc(${probability}% - 2px)`;

  const nextTier = probability < 30 ? 30 : probability < 65 ? 65 : 100;
  const margin = Math.max(0, nextTier - probability);
  riskMargin.textContent = prediction.risk_level === "High" ? "Top tier" : `${margin.toFixed(1)} pts`;
  const odds = prediction.churn_probability >= 0.999
    ? "999:1"
    : `${(prediction.churn_probability / Math.max(0.001, 1 - prediction.churn_probability)).toFixed(2)}:1`;
  oddsValue.textContent = odds;

  const factors = prediction.top_factors || [];
  const drivers = factors.filter((factor) => factor.shap_value > 0);
  const protectors = factors.filter((factor) => factor.shap_value <= 0);
  const strongest = factors[0]?.feature?.replaceAll("_", " ") || "--";
  driverCount.textContent = drivers.length;
  protectorCount.textContent = protectors.length;
  strongestFactor.textContent = strongest;

  const maxImpact = Math.max(...factors.map((factor) => Math.abs(factor.shap_value)), 0.01);
  factorList.innerHTML = factors.map((factor) => {
    const width = Math.max(8, Math.abs(factor.shap_value) / maxImpact * 100);
    const positive = factor.shap_value > 0;
    const label = factor.feature.replaceAll("_", " ");
    return `
      <button class="factor-row" type="button" data-feature="${factor.feature}">
        <div class="factor-meta">
          <span>${label}</span>
          <span>${factor.shap_value > 0 ? "+" : ""}${factor.shap_value.toFixed(4)}</span>
        </div>
        <div class="factor-track">
          <div class="factor-fill" style="width:${width}%;background:${positive ? "var(--red)" : "var(--green)"}"></div>
        </div>
      </button>
    `;
  }).join("");

  recommendations.innerHTML = recommendationText(customer, prediction.risk_level)
    .map((item) => `<li>${item}</li>`)
    .join("");

  emptyState.classList.add("hidden");
  results.classList.remove("hidden");
  featureDetail.classList.add("hidden");
}

function showFeatureDetail(featureName) {
  if (!lastPrediction || !lastCustomer) return;
  const factor = (lastPrediction.top_factors || []).find((item) => item.feature === featureName);
  if (!factor) return;

  const label = featureName.replaceAll("_", " ");
  const rawKey = Object.keys(lastCustomer).find((key) => {
    const compactFeature = featureName.toLowerCase().replace(/[^a-z0-9]/g, "");
    const compactKey = key.toLowerCase().replace(/[^a-z0-9]/g, "");
    return compactFeature.includes(compactKey) || compactKey.includes(compactFeature);
  });
  const currentValue = rawKey ? lastCustomer[rawKey] : "Encoded model feature";
  const positive = factor.shap_value > 0;
  const absImpact = Math.abs(factor.shap_value);

  detailTitle.textContent = label;
  detailValue.textContent = currentValue;
  detailDirection.textContent = positive
    ? "This factor is pushing the model toward a higher churn probability for this customer."
    : "This factor is reducing the model's churn probability for this customer.";

  detailImpactChart.innerHTML = `
    <div class="detail-impact-row">
      <span>Direction</span>
      <div class="detail-impact-track">
        <div class="detail-impact-fill" style="width:100%;background:${positive ? "var(--red)" : "var(--green)"}"></div>
      </div>
      <span>${positive ? "Risk" : "Protective"}</span>
    </div>
    <div class="detail-impact-row">
      <span>SHAP value</span>
      <div class="detail-impact-track">
        <div class="detail-impact-fill" style="width:${Math.min(100, absImpact * 100)}%;background:var(--blue)"></div>
      </div>
      <span>${factor.shap_value > 0 ? "+" : ""}${factor.shap_value.toFixed(4)}</span>
    </div>
  `;

  const baseProbability = Math.round(lastPrediction.churn_probability * 1000) / 10;
  const simulatedLift = Math.min(18, Math.max(3, absImpact * 35));
  const improved = positive ? Math.max(1, baseProbability - simulatedLift) : Math.min(99, baseProbability + simulatedLift);
  scenarioDemo.innerHTML = `
    <div class="scenario-row">
      <span>Current</span>
      <div class="scenario-track"><div class="scenario-fill" style="width:${baseProbability}%;background:${riskColor(lastPrediction.risk_level)}"></div></div>
      <span>${baseProbability}%</span>
    </div>
    <div class="scenario-row">
      <span>${positive ? "If mitigated" : "If removed"}</span>
      <div class="scenario-track"><div class="scenario-fill" style="width:${improved}%;background:${positive ? "var(--green)" : "var(--amber)"}"></div></div>
      <span>${improved.toFixed(1)}%</span>
    </div>
  `;

  detailNotes.innerHTML = [
    "SHAP values explain this prediction relative to the model's baseline, not a universal business rule.",
    "Categorical inputs may appear as encoded feature names after preprocessing.",
    positive
      ? "Use this factor to design a focused retention action or billing/service review."
      : "This factor is currently helping retention; protect it before changing the account."
  ].map((note) => `<li>${note}</li>`).join("");

  featureDetail.classList.remove("hidden");
  featureDetail.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderBatchResults(data) {
  const rows = data.rows || [];
  lastBatchRows = rows;
  if (exportHighRisk) exportHighRisk.disabled = !rows.some((row) => row.risk_level === "High");
  const counts = {
    High: rows.filter((row) => row.risk_level === "High").length,
    Medium: rows.filter((row) => row.risk_level === "Medium").length,
    Low: rows.filter((row) => row.risk_level === "Low").length,
  };
  batchSummary.textContent = `${data.total} customers scored. ${data.high_risk_count} high-risk accounts found.`;
  riskChart.innerHTML = Object.entries(counts).map(([risk, count]) => {
    const width = data.total ? Math.round(count / data.total * 100) : 0;
    const color = riskColor(risk);
    return `
      <div class="chart-row">
        <span>${risk}</span>
        <div class="chart-track"><div class="chart-fill" style="width:${width}%;background:${color}"></div></div>
        <span>${count}</span>
      </div>
    `;
  }).join("");

  const topRows = [...rows]
    .sort((a, b) => b.churn_probability - a.churn_probability)
    .slice(0, 20);
  batchTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Customer</th>
          <th>Risk</th>
          <th>Probability</th>
          <th>Top driver</th>
          <th>Contract</th>
        </tr>
      </thead>
      <tbody>
        ${topRows.map((row) => `
          <tr>
            <td>${row.customerID}</td>
            <td>${row.risk_level}</td>
            <td>${Math.round(row.churn_probability * 1000) / 10}%</td>
            <td>${row.top_driver.replaceAll("_", " ")}</td>
            <td>${row.contract || ""}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function rowsToCsv(rows) {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  return [
    headers.join(","),
    ...rows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")),
  ].join("\n");
}

function demoRows() {
  return [
    { customerID: "DEMO-001", ...highRiskPreset },
    { customerID: "DEMO-002", ...lowRiskPreset },
    { customerID: "DEMO-003", ...highRiskPreset, tenure: 7, MonthlyCharges: 94.2, TotalCharges: 659.4 },
    { customerID: "DEMO-004", ...lowRiskPreset, tenure: 38, Contract: "One year", MonthlyCharges: 62.1, TotalCharges: 2359.8 },
    { customerID: "DEMO-005", ...highRiskPreset, Partner: "Yes", tenure: 14, MonthlyCharges: 73.4, TotalCharges: 1027.6 },
  ];
}

async function scoreDemoCsv() {
  const csv = rowsToCsv(demoRows());
  const file = new File([csv], "churnguard_demo_customers.csv", { type: "text/csv" });
  batchSummary.textContent = "Scoring demo CSV...";
  riskChart.innerHTML = "";
  batchTable.innerHTML = "";
  const data = await postFile("/predict-csv", file);
  renderBatchResults(data);
  await loadAdminSummary();
}

function exportHighRiskCsv() {
  const rows = lastBatchRows.filter((row) => row.risk_level === "High");
  if (!rows.length) return;
  const blob = new Blob([rowsToCsv(rows)], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "churnguard_high_risk_customers.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function loadSchema() {
  const container = document.getElementById("dynamicFormFields");
  if (!container) return;
  const role = localStorage.getItem("churnguard_role");
  if (role === "platform_admin") {
    container.innerHTML = '<p style="color:var(--gray-400); font-size:14px;">Platform Admins manage workspaces and do not have an active prediction schema.</p>';
    const btn = document.querySelector("#profileForm .primary");
    if (btn) btn.disabled = true;
    return;
  }

  try {
    const data = await fetch("/admin/schema", { headers: authHeaders() }).then(r => {
      if (!r.ok) throw new Error("Failed to load schema");
      return r.json();
    });
    
    let html = "";
    const numCols = data.numerical || [];
    const catCols = data.categorical || {};
    
    if (numCols.length === 0 && Object.keys(catCols).length === 0) {
      container.innerHTML = '<p style="color:var(--gray-400); font-size:14px;">Upload a Seed CSV in the Improvement Lab to initialize your dynamic schema and generate your form.</p>';
      return;
    }

    if (numCols.length > 0) {
      html += '<fieldset><legend>Numerical Attributes</legend>';
      numCols.forEach(col => {
        html += `<label>${col}<input name="${col}" type="number" step="0.1" value="0" /></label>`;
      });
      html += '</fieldset>';
    }

    if (Object.keys(catCols).length > 0) {
      html += '<fieldset><legend>Categorical Attributes</legend>';
      for (const [col, values] of Object.entries(catCols)) {
        if (Array.isArray(values) && values.length > 0 && values.length < 15) {
          html += `<label>${col}<select name="${col}">`;
          values.forEach(v => {
            html += `<option value="${v}">${v}</option>`;
          });
          html += `</select></label>`;
        } else {
          html += `<label>${col}<input name="${col}" type="text" placeholder="Value" /></label>`;
        }
      }
      html += '</fieldset>';
    }
    
    container.innerHTML = html;
  } catch (error) {
    container.innerHTML = '<p style="color:var(--red); font-size:14px;">Failed to load dynamic schema.</p>';
  }
}

async function loadTenants() {
  const table = document.getElementById("tenantTable");
  const role = localStorage.getItem("churnguard_role");
  if (!table || role !== "owner") return;
  try {
    const response = await fetch("/admin/tenants", { headers: authHeaders() });
    if (!response.ok) return;
    const tenants = await response.json();

    if (tenants.length === 0) {
      table.innerHTML = '<p style="color:var(--gray-400)">No tenants onboarded yet.</p>';
      return;
    }

    // Update KPI cards
    const totalTenants = document.getElementById("adminTotalTenants");
    const totalModels = document.getElementById("adminTotalModels");
    const totalUsers = document.getElementById("adminTotalUsers");
    const totalPredictions = document.getElementById("adminTotalPredictions");
    if (totalTenants) totalTenants.textContent = tenants.length;
    if (totalModels) totalModels.textContent = tenants.filter(t => t.has_model).length;
    if (totalUsers) totalUsers.textContent = tenants.reduce((s, t) => s + t.user_count, 0);
    if (totalPredictions) totalPredictions.textContent = tenants.reduce((s, t) => s + t.predictions, 0).toLocaleString();

    // Build table
    let html = `<table><thead><tr>
      <th>Company</th><th>Users</th><th>Schema</th><th>Model</th>
      <th>Learning Rows</th><th>Predictions</th><th>Created</th><th>Actions</th>
    </tr></thead><tbody>`;
    for (const t of tenants) {
      const schemaStatus = t.has_schema ? '✅' : '⚠️ None';
      const modelStatus = t.has_model ? '✅ Trained' : '❌ Not trained';
      const created = t.created_at ? new Date(t.created_at).toLocaleDateString() : '—';
      html += `<tr>
        <td><strong>${t.company_name || t.company_id}</strong><br><small style="color:var(--gray-400)">${t.company_id}</small></td>
        <td>${t.user_count}</td>
        <td>${schemaStatus}</td>
        <td>${modelStatus}</td>
        <td>${t.learning_rows.toLocaleString()}</td>
        <td>${t.predictions.toLocaleString()}</td>
        <td>${created}</td>
        <td>
          <button class="impersonate-btn" data-id="${t.company_id}" style="padding:0.35rem 0.65rem;font-size:0.75rem;background:linear-gradient(135deg, var(--cyan) 0%, #0891b2 100%);border:none;color:#111827;border-radius:6px;cursor:pointer;font-weight:700;box-shadow:0 2px 6px rgba(8,145,178,0.15);">🔬 View in Lab</button>
          ${t.company_id !== 'default' ? `<button class="delete-tenant-btn" data-id="${t.company_id}" style="padding:0.35rem 0.65rem;font-size:0.75rem;background:#fee2e2;border:1px solid #fca5a5;color:#dc2626;border-radius:6px;margin-left:0.35rem;cursor:pointer;font-weight:700;">🗑️ Delete</button>` : ''}
        </td>
      </tr>`;
    }
    html += '</tbody></table>';
    table.innerHTML = html;

    // Remove existing event listener if any (by replacing table with clone, or just adding once)
    if (!table.dataset.delegated) {
      table.dataset.delegated = "true";
      table.addEventListener("click", async (e) => {
        const impBtn = e.target.closest(".impersonate-btn");
        const delBtn = e.target.closest(".delete-tenant-btn");
        
        if (impBtn) {
          const companyId = impBtn.dataset.id;
          try {
            const data = await postJson("/admin/impersonate", { company_id: companyId });
            localStorage.setItem("churnguard_token", data.access_token);
            localStorage.setItem("churnguard_company_id", data.company_id);
            
            // Reload all context!
            await loadWorkspace();
            await loadSchema();
            showPage("lab");
            await loadLabData();
          } catch (err) {
            alert("Failed to switch context: " + err.message);
          }
        }
        
        if (delBtn) {
          const companyId = delBtn.dataset.id;
          if (!confirm(`Are you absolutely sure you want to delete tenant '${companyId}'? All predictions and model artifacts will be destroyed.`)) return;
          try {
            const data = await postJson("/admin/delete-tenant", { company_id: companyId });
            alert(data.message);
            loadTenants();
          } catch (err) {
            alert("Failed to delete tenant: " + err.message);
          }
        }
      });
    }
  } catch (error) {
    table.innerHTML = '<p style="color:var(--red)">Failed to load tenants.</p>';
  }
}

async function loadModelInfo() {
  try {
    const response = await fetch("/model-info", { headers: authHeaders() });
    if (!response.ok) {
      if (response.status === 401) clearSession();
      return;
    }
    const data = await response.json();
    modelType.textContent = data.model_type || "Model";
    modelAuc.textContent = data.training_metrics?.roc_auc?.toFixed?.(3) || "--";
    featureCount.textContent = data.n_features || "--";
  } catch {
    modelAuc.textContent = "--";
  }
}

function riskColor(risk) {
  if (risk === "High") return "var(--red)";
  if (risk === "Medium") return "var(--amber)";
  return "var(--green)";
}

signinTab.addEventListener("click", () => setAuthMode("signin"));
signupTab.addEventListener("click", () => setAuthMode("signup"));

signinForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  authMessage.textContent = "";
  try {
    const data = await postJson("/auth/login", formToObject(signinForm));
    setAuthenticatedSession(data.username, data.access_token, data.company_id, data.role);
  } catch (error) {
    authMessage.textContent = error.message;
  }
});

signupForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  authMessage.textContent = "";
  try {
    const data = await postJson("/auth/signup", formToObject(signupForm));
    setAuthenticatedSession(data.username, data.access_token, data.company_id, data.role);
  } catch (error) {
    authMessage.textContent = error.message;
  }
});

logoutButton.addEventListener("click", clearSession);
toggleSidebar.addEventListener("click", () => {
  setSidebarHidden(!document.body.classList.contains("sidebar-hidden"));
});
openLab.addEventListener("click", () => { showPage("lab"); loadLabData(); });
backToDashboard.addEventListener("click", () => showPage("dashboard"));

// ── Lab Tab Switching ──────────────────────────────────────────────
document.querySelector(".lab-tabs")?.addEventListener("click", (e) => {
  const tab = e.target.closest(".lab-tab");
  if (!tab) return;
  document.querySelectorAll(".lab-tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".lab-panel").forEach(p => p.classList.add("hidden"));
  tab.classList.add("active");
  const panel = document.getElementById(tab.dataset.tab);
  if (panel) panel.classList.remove("hidden");
});

async function loadLabData() {
  loadLabSchema();
  loadLabMetrics();
}

async function loadLabSchema() {
  const view = document.getElementById("labSchemaView");
  if (!view) return;
  try {
    const data = await fetch("/admin/schema", { headers: authHeaders() }).then(r => r.json());
    const numCols = data.numerical || [];
    const catCols = data.categorical || {};
    const catKeys = typeof catCols === "object" && !Array.isArray(catCols) ? Object.keys(catCols) : (Array.isArray(catCols) ? catCols : []);

    if (numCols.length === 0 && catKeys.length === 0) {
      view.innerHTML = '<p style="color:var(--muted);font-size:14px;">No schema detected yet. Upload a seed CSV in the Train & Learn tab.</p>';
      return;
    }

    let html = '<div class="schema-chips">';
    numCols.forEach(c => { html += `<span class="schema-chip schema-chip--num">🔢 ${c}</span>`; });
    catKeys.forEach(c => { html += `<span class="schema-chip schema-chip--cat">🏷️ ${c}</span>`; });
    html += '</div>';
    html += `<p style="margin-top:.6rem;font-size:.82rem;color:var(--muted)">${numCols.length} numerical · ${catKeys.length} categorical</p>`;
    view.innerHTML = html;
  } catch {
    view.innerHTML = '<p style="color:var(--red)">Failed to load schema.</p>';
  }
}

async function loadLabMetrics() {
  const metricsDiv = document.getElementById("labModelMetrics");
  const heroAuc = document.getElementById("labModelAuc");
  const heroQueued = document.getElementById("labQueuedRows");
  const heroPreds = document.getElementById("labTotalPreds");
  try {
    const info = await fetch("/model-info", { headers: authHeaders() }).then(r => r.ok ? r.json() : null);
    if (info) {
      const m = info.training_metrics || {};
      if (heroAuc) heroAuc.textContent = m.roc_auc ? Number(m.roc_auc).toFixed(3) : "--";
      if (metricsDiv) {
        let html = "";
        const metrics = [
          ["AUC", m.roc_auc], ["F1", m.f1], ["Accuracy", m.accuracy],
          ["Precision", m.precision], ["Recall", m.recall], ["Features", info.n_features]
        ];
        metrics.forEach(([label, val]) => {
          const display = typeof val === "number" ? (val < 1 ? (val * 100).toFixed(1) + "%" : val) : (val || "--");
          html += `<div class="lab-metric"><span class="lab-metric__value">${display}</span><span class="lab-metric__label">${label}</span></div>`;
        });
        metricsDiv.innerHTML = html;
      }
    }
  } catch {}
  try {
    const summary = await fetch("/admin/summary", { headers: authHeaders() }).then(r => r.ok ? r.json() : null);
    if (summary) {
      if (heroQueued) heroQueued.textContent = summary.learning_rows_queued?.toLocaleString() || "0";
      if (heroPreds) heroPreds.textContent = summary.total_predictions?.toLocaleString() || "0";
    }
  } catch {}
}

// ── Lab Learning CSV Upload ────────────────────────────────────────
document.getElementById("labLearningCsv")?.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  const status = document.getElementById("labLearningStatus");
  if (status) status.textContent = "Uploading labeled rows...";
  try {
    const data = await postFile("/learning/upload", file);
    if (status) status.textContent = `✅ ${data.accepted_rows} rows accepted. ${data.stored_rows} total queued.`;
    loadLabMetrics();
  } catch (error) {
    if (status) status.textContent = `❌ ${error.message}`;
  }
});

// ── Lab Seed CSV Upload ────────────────────────────────────────────
document.getElementById("labSeedCsv")?.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  const status = document.getElementById("labSeedStatus");
  if (status) status.textContent = "Uploading seed CSV...";
  try {
    const data = await postFile("/learning/upload", file);
    if (status) {
      status.textContent = `✅ Schema inferred! ${data.accepted_rows} rows queued.`;
      status.style.color = "var(--green)";
    }
    loadSchema();
    loadLabSchema();
  } catch (error) {
    if (status) status.textContent = `❌ ${error.message}`;
  }
});

// ── Lab Export Buttons ─────────────────────────────────────────────
document.getElementById("labExportSchema")?.addEventListener("click", async () => {
  const msg = document.getElementById("labExportMessage");
  try {
    const data = await fetch("/admin/schema", { headers: authHeaders() }).then(r => r.json());
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "churnguard_schema.json";
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    if (msg) { msg.textContent = "✅ Schema exported."; msg.style.color = "var(--green)"; }
  } catch {
    if (msg) { msg.textContent = "❌ Failed to export schema."; msg.style.color = "var(--red)"; }
  }
});

document.getElementById("labExportHighRisk")?.addEventListener("click", () => {
  exportHighRiskCsv();
  const msg = document.getElementById("labExportMessage");
  if (msg) { msg.textContent = lastBatchRows.length ? "✅ High-risk CSV downloaded." : "⚠️ Run a batch prediction first."; msg.style.color = lastBatchRows.length ? "var(--green)" : "var(--amber)"; }
});

document.getElementById("labExportTemplate")?.addEventListener("click", async () => {
  const msg = document.getElementById("labExportMessage");
  try {
    const data = await fetch("/admin/schema", { headers: authHeaders() }).then(r => r.json());
    const numCols = data.numerical || [];
    const catCols = data.categorical || {};
    const catKeys = typeof catCols === "object" && !Array.isArray(catCols) ? Object.keys(catCols) : (Array.isArray(catCols) ? catCols : []);
    const headers = [...numCols, ...catKeys, "Churn"];
    const csv = headers.join(",") + "\n";
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "churnguard_template.csv";
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    if (msg) { msg.textContent = "✅ Template CSV downloaded."; msg.style.color = "var(--green)"; }
  } catch {
    if (msg) { msg.textContent = "❌ No schema available yet."; msg.style.color = "var(--red)"; }
  }
});
closeDetail.addEventListener("click", () => featureDetail.classList.add("hidden"));
factorList.addEventListener("click", (event) => {
  const row = event.target.closest(".factor-row");
  if (row?.dataset.feature) showFeatureDetail(row.dataset.feature);
});

profileForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const customer = customerFromForm();
  try {
    const prediction = await postJson("/predict", customer);
    renderPrediction(customer, prediction);
    await loadAdminSummary();
  } catch (error) {
    emptyState.classList.remove("hidden");
    results.classList.add("hidden");
    emptyState.innerHTML = `<h2>Prediction unavailable</h2><p>${error.message}</p>`;
  }
});

batchCsv.addEventListener("change", async () => {
  const file = batchCsv.files?.[0];
  if (!file) return;
  batchSummary.textContent = "Scoring CSV...";
  riskChart.innerHTML = "";
  batchTable.innerHTML = "";
  try {
    const data = await postFile("/predict-csv", file);
    renderBatchResults(data);
    await loadAdminSummary();
  } catch (error) {
    renderValidationErrors(batchSummary, error.detail || error.message);
  }
});

learningCsv.addEventListener("change", async () => {
  const file = learningCsv.files?.[0];
  if (!file) return;
  learningStatus.textContent = "Uploading labeled rows...";
  try {
    const data = await postFile("/learning/upload", file);
    learningStatus.textContent = `${data.accepted_rows} rows added to the learning queue. ${data.stored_rows} rows stored total.`;
    await loadLearningStatus();
    await loadAdminSummary();
  } catch (error) {
    renderValidationErrors(learningStatus, error.detail || error.message);
  }
});

loadDemoCsv.addEventListener("click", async () => {
  try {
    await scoreDemoCsv();
  } catch (error) {
    renderValidationErrors(batchSummary, error.detail || error.message);
  }
});

exportHighRisk.addEventListener("click", exportHighRiskCsv);

onboardForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  onboardMessage.textContent = "Provisioning workspace...";
  onboardMessage.style.color = "inherit";
  try {
    const data = await postJson("/admin/onboard", formToObject(onboardForm));
    onboardMessage.textContent = data.message;
    onboardMessage.style.color = "var(--green)";
    onboardForm.reset();
    loadTenants();
  } catch (error) {
    onboardMessage.textContent = error.message;
    onboardMessage.style.color = "var(--red)";
  }
});

document.getElementById("adminRefreshTenants")?.addEventListener("click", () => {
  loadTenants();
});

document.getElementById("adminRetrainAll")?.addEventListener("click", async () => {
  const msg = document.getElementById("adminActionMessage");
  if (msg) {
    msg.textContent = "⏳ Triggering background model retraining...";
    msg.style.color = "var(--amber)";
  }
  try {
    const data = await postJson("/admin/retrain", {});
    if (msg) {
      msg.textContent = "✅ " + data.message;
      msg.style.color = "var(--green)";
    }
  } catch (error) {
    if (msg) {
      msg.textContent = "❌ " + error.message;
      msg.style.color = "var(--red)";
    }
  }
});

document.body.addEventListener("click", (e) => {
  const btn = e.target.closest("#loadTestTrialBtn");
  if (!btn) return;
  const container = document.getElementById("dynamicFormFields");
  if (!container) return;
  const inputs = container.querySelectorAll("input, select");
  inputs.forEach(input => {
    if (input.tagName.toLowerCase() === "select") {
      if (input.options.length > 0) {
        input.selectedIndex = 0;
      }
    } else if (input.type === "number") {
      const name = input.name.toLowerCase();
      if (name.includes("tenure")) input.value = 12;
      else if (name.includes("charge")) input.value = 65;
      else if (name.includes("total")) input.value = 780;
      else input.value = 15;
    } else {
      input.value = "1";
    }
  });
  // Auto-submit the profileForm
  document.getElementById("profileForm")?.dispatchEvent(new Event("submit", { cancelable: true }));
});

const existingUser = localStorage.getItem("churnguard_user");
const existingToken = localStorage.getItem("churnguard_token");
if (existingUser && existingToken) {
  setSession(existingUser);
} else {
  clearSession();
  loadAuthConfig();
}
setSidebarHidden(localStorage.getItem("churnguard_sidebar_hidden") === "1");
