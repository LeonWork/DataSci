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
const resultsActions = document.querySelector("#resultsActions");
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
const schemaEditorForm = document.querySelector("#schemaEditorForm");
const schemaNumerical = document.querySelector("#schemaNumerical");
const schemaCategorical = document.querySelector("#schemaCategorical");
const schemaEditorMessage = document.querySelector("#schemaEditorMessage");
const learningReviewSummary = document.querySelector("#learningReviewSummary");
const learningReviewTable = document.querySelector("#learningReviewTable");
const modelPromotionPanel = document.querySelector("#modelPromotionPanel");
const promotionHistoryPanel = document.querySelector("#promotionHistoryPanel");
const modelPromotionMessage = document.querySelector("#modelPromotionMessage");
const trainModelCandidate = document.querySelector("#trainModelCandidate");
const promoteModelCandidate = document.querySelector("#promoteModelCandidate");
const rejectModelCandidate = document.querySelector("#rejectModelCandidate");
const whatIfUseCurrent = document.querySelector("#whatIfUseCurrent");
const whatIfGeneratePlan = document.querySelector("#whatIfGeneratePlan");
const whatIfRunComparison = document.querySelector("#whatIfRunComparison");
const whatIfBaseline = document.querySelector("#whatIfBaseline");
const whatIfScenario = document.querySelector("#whatIfScenario");
const whatIfResults = document.querySelector("#whatIfResults");
const whatIfMessage = document.querySelector("#whatIfMessage");
let lastCustomer = null;
let lastPrediction = null;
let lastBatchRows = [];
let currentSchema = null;
let whatIfState = {
  baseline: null,
  scenario: null,
  baselinePrediction: null,
  scenarioPrediction: null,
};

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
    const error = new Error(formatApiError(data.detail, "Request failed."));
    error.detail = data.detail;
    throw error;
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
  const schemaNumbers = currentSchema?.numerical || [];
  schemaNumbers.forEach((field) => {
    if (field in customer) customer[field] = Number(customer[field]);
  });
  return { customerID: "WEB-USER", ...customer };
}

function cloneCustomer(customer) {
  return JSON.parse(JSON.stringify(customer || {}));
}

function summarizeProfile(customer) {
  if (!customer) return [];
  const preferred = [
    "Contract",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "InternetService",
    "TechSupport",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "PaymentMethod",
    "PaperlessBilling",
  ].filter((key) => key in customer);
  const fallback = Object.keys(customer).filter((key) => key !== "customerID").slice(0, 8);
  const keys = preferred.length ? preferred : fallback;
  return keys.map((key) => [key, customer[key]]);
}

function renderProfileSummary(container, customer, changedFrom = null) {
  if (!container) return;
  if (!customer) {
    container.innerHTML = '<p style="color:var(--muted);font-size:14px;">No profile selected.</p>';
    return;
  }
  container.innerHTML = `
    <div class="whatif-profile-grid">
      ${summarizeProfile(customer).map(([key, value]) => {
        const changed = changedFrom && String(changedFrom[key]) !== String(value);
        return `
          <div class="${changed ? "changed" : ""}">
            <span>${key}</span>
            <strong>${value ?? "--"}</strong>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function buildRetentionScenario(customer) {
  const scenario = cloneCustomer(customer);
  if ("Contract" in scenario && scenario.Contract === "Month-to-month") scenario.Contract = "One year";
  if ("TechSupport" in scenario && scenario.InternetService !== "No") scenario.TechSupport = "Yes";
  if ("OnlineSecurity" in scenario && scenario.InternetService !== "No") scenario.OnlineSecurity = "Yes";
  if ("OnlineBackup" in scenario && scenario.InternetService !== "No") scenario.OnlineBackup = "Yes";
  if ("DeviceProtection" in scenario && scenario.InternetService !== "No") scenario.DeviceProtection = "Yes";
  if ("PaperlessBilling" in scenario) scenario.PaperlessBilling = "No";
  if ("PaymentMethod" in scenario) scenario.PaymentMethod = "Bank transfer (automatic)";
  if ("tenure" in scenario) scenario.tenure = Math.min(72, Number(scenario.tenure || 0) + 12);
  if ("MonthlyCharges" in scenario) scenario.MonthlyCharges = Math.max(0, Number((Number(scenario.MonthlyCharges || 0) * 0.9).toFixed(2)));
  if ("TotalCharges" in scenario && "MonthlyCharges" in scenario && "tenure" in scenario) {
    scenario.TotalCharges = Number((Number(scenario.MonthlyCharges || 0) * Number(scenario.tenure || 0)).toFixed(2));
  }
  scenario.customerID = "WHAT-IF-B";
  return scenario;
}

function changedFields(base, scenario) {
  return Object.keys(scenario || {})
    .filter((key) => key !== "customerID" && String(base?.[key]) !== String(scenario?.[key]))
    .map((key) => ({ key, from: base?.[key], to: scenario?.[key] }));
}

function factorMap(prediction) {
  const map = new Map();
  (prediction?.top_factors || []).forEach((item) => map.set(item.feature, item.shap_value));
  return map;
}

function renderWhatIfComparison() {
  if (!whatIfResults) return;
  const { baseline, scenario, baselinePrediction, scenarioPrediction } = whatIfState;
  if (!baseline || !scenario || !baselinePrediction || !scenarioPrediction) {
    whatIfResults.innerHTML = `
      <div class="promotion-empty">
        <strong>No comparison yet</strong>
        <span>Capture a baseline, generate a scenario, then score the comparison.</span>
      </div>
    `;
    return;
  }

  const baseProb = baselinePrediction.churn_probability * 100;
  const scenarioProb = scenarioPrediction.churn_probability * 100;
  const delta = scenarioProb - baseProb;
  const improved = delta < 0;
  const changes = changedFields(baseline, scenario);
  const beforeFactors = factorMap(baselinePrediction);
  const afterFactors = factorMap(scenarioPrediction);
  const factorRows = Array.from(new Set([
    ...(baselinePrediction.top_factors || []).map((item) => item.feature),
    ...(scenarioPrediction.top_factors || []).map((item) => item.feature),
  ])).slice(0, 6).map((feature) => {
    const before = beforeFactors.get(feature) || 0;
    const after = afterFactors.get(feature) || 0;
    return { feature, before, after, delta: after - before };
  });

  whatIfResults.innerHTML = `
    <div class="whatif-score-grid">
      <div>
        <span>Profile A</span>
        <strong>${baseProb.toFixed(1)}%</strong>
        <em>${baselinePrediction.risk_level} risk</em>
      </div>
      <div>
        <span>Profile B</span>
        <strong>${scenarioProb.toFixed(1)}%</strong>
        <em>${scenarioPrediction.risk_level} risk</em>
      </div>
      <div class="${improved ? "good" : "bad"}">
        <span>Delta</span>
        <strong>${delta >= 0 ? "+" : ""}${delta.toFixed(1)} pts</strong>
        <em>${improved ? "Lower churn risk" : "Higher churn risk"}</em>
      </div>
    </div>
    <div class="whatif-two-col">
      <div>
        <h4>Changed inputs</h4>
        ${changes.length ? `
          <ul class="whatif-change-list">
            ${changes.map((item) => `<li><strong>${item.key}</strong><span>${item.from ?? "--"} → ${item.to ?? "--"}</span></li>`).join("")}
          </ul>
        ` : '<p class="history-muted">No input changes detected.</p>'}
      </div>
      <div>
        <h4>Recommended action</h4>
        <ul class="recommendations">
          ${recommendationText(scenario, scenarioPrediction.risk_level).slice(0, 4).map((item) => `<li>${item}</li>`).join("")}
        </ul>
      </div>
    </div>
    <div class="whatif-driver-table">
      <h4>Driver movement</h4>
      <table>
        <thead><tr><th>Feature</th><th>Profile A</th><th>Profile B</th><th>Move</th></tr></thead>
        <tbody>
          ${factorRows.map((row) => `
            <tr>
              <td>${row.feature.replaceAll("_", " ")}</td>
              <td>${row.before.toFixed(4)}</td>
              <td>${row.after.toFixed(4)}</td>
              <td class="${row.delta <= 0 ? "good" : "bad"}">${row.delta >= 0 ? "+" : ""}${row.delta.toFixed(4)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function setWhatIfMessage(text, color = "inherit") {
  if (!whatIfMessage) return;
  whatIfMessage.textContent = text;
  whatIfMessage.style.color = color;
}

function recommendationText(customer, risk) {
  const telcoFields = ["Contract", "tenure", "TechSupport", "InternetService", "MonthlyCharges", "StreamingTV"];
  const hasTelcoShape = telcoFields.some((field) => field in customer);
  if (!hasTelcoShape) {
    if (risk === "High") {
      return [
        "Prioritize this account for retention review.",
        "Inspect the strongest risk drivers and compare them against recent customer activity.",
        "Assign a specific owner and track the outcome for future model learning.",
      ];
    }
    if (risk === "Medium") {
      return [
        "Monitor this account and add it to a proactive check-in segment.",
        "Review the top drivers before the next renewal or usage milestone.",
      ];
    }
    return [
      "Keep this customer in the normal monitoring cadence.",
      "Use future outcomes to confirm whether the model remains calibrated.",
    ];
  }
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
  resultsActions.classList.remove("hidden");
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
            <td><span class="risk-badge risk-${row.risk_level.toLowerCase()}">${row.risk_level}</span></td>
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
    currentSchema = data;
    
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
      const schemaStatus = t.has_schema ? '<span class="badge badge--success">Active</span>' : '<span class="badge badge--warn">None</span>';
      const modelStatus = t.has_model ? '<span class="badge badge--success">Trained</span>' : '<span class="badge badge--error">Not trained</span>';
      const created = t.created_at ? new Date(t.created_at).toLocaleDateString() : '—';
      html += `<tr>
        <td><strong>${t.company_name || t.company_id}</strong><br><small>${t.company_id}</small></td>
        <td>${t.user_count}</td>
        <td>${schemaStatus}</td>
        <td>${modelStatus}</td>
        <td>${t.learning_rows.toLocaleString()}</td>
        <td>${t.predictions.toLocaleString()}</td>
        <td>${created}</td>
        <td>
          <button class="impersonate-btn" data-id="${t.company_id}">🔬 View in Lab</button>
          ${t.company_id !== 'default' ? `<button class="delete-tenant-btn" data-id="${t.company_id}">🗑️ Delete</button>` : ''}
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
  loadDriftMonitor();
  loadLearningReview();
  loadModelPromotion();
  loadPromotionHistory();
}

async function loadLabSchema() {
  const view = document.getElementById("labSchemaView");
  if (!view) return;
  try {
    const data = await fetch("/admin/schema", { headers: authHeaders() }).then(r => r.json());
    const numCols = data.numerical || [];
    const catCols = data.categorical || {};
    const catKeys = typeof catCols === "object" && !Array.isArray(catCols) ? Object.keys(catCols) : (Array.isArray(catCols) ? catCols : []);
    currentSchema = data;
    populateSchemaEditor(data);

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

function populateSchemaEditor(schema) {
  if (!schemaNumerical || !schemaCategorical) return;
  const numCols = schema.numerical || [];
  const catCols = schema.categorical || {};
  schemaNumerical.value = numCols.join("\n");
  const catLines = Object.entries(catCols).map(([column, values]) => {
    const suffix = Array.isArray(values) && values.length ? `: ${values.join(", ")}` : "";
    return `${column}${suffix}`;
  });
  schemaCategorical.value = catLines.join("\n");
}

function schemaFromEditor() {
  const numerical = (schemaNumerical?.value || "")
    .split(/\n|,/)
    .map((item) => item.trim())
    .filter(Boolean);
  const categorical = {};
  (schemaCategorical?.value || "").split("\n").forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    const [column, ...rest] = trimmed.split(":");
    const name = column.trim();
    if (!name) return;
    const values = rest.join(":").split(",").map((item) => item.trim()).filter(Boolean);
    categorical[name] = values;
  });
  return { numerical, categorical };
}

async function saveSchemaEditor(event) {
  event.preventDefault();
  if (!schemaEditorMessage) return;
  schemaEditorMessage.textContent = "Saving schema...";
  schemaEditorMessage.style.color = "inherit";
  try {
    const data = await fetch("/admin/schema", {
      method: "PUT",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(schemaFromEditor()),
    }).then(async (response) => {
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(formatApiError(body.detail, "Could not save schema."));
      return body;
    });
    currentSchema = data.schema;
    schemaEditorMessage.textContent = "Schema saved.";
    schemaEditorMessage.style.color = "var(--green)";
    await loadSchema();
    await loadLabSchema();
  } catch (error) {
    schemaEditorMessage.textContent = error.message;
    schemaEditorMessage.style.color = "var(--red)";
  }
}

async function loadLearningReview() {
  if (!learningReviewSummary || !learningReviewTable) return;
  try {
    const data = await fetch("/learning/review", { headers: authHeaders() }).then(async (response) => {
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(formatApiError(body.detail, "Could not load review queue."));
      return body;
    });
    const counts = data.counts || {};
    const rows = data.rows || [];
    learningReviewSummary.textContent = `${counts.queued || 0} queued · ${counts.approved_for_training || 0} approved · ${counts.used_in_model || 0} used · ${counts.rejected || 0} rejected`;
    if (!rows.length) {
      learningReviewTable.innerHTML = '<p style="color:var(--muted);font-size:14px;">No queued learning rows need review.</p>';
      return;
    }
    learningReviewTable.innerHTML = `
      <table>
        <thead><tr><th></th><th>Customer</th><th>Churn</th><th>Preview</th></tr></thead>
        <tbody>
          ${rows.map((item) => {
            const row = item.row || {};
            const preview = Object.entries(row)
              .filter(([key]) => !["Churn", "source_file", "uploaded_at"].includes(key))
              .slice(0, 3)
              .map(([key, value]) => `${key}: ${value}`)
              .join(" · ");
            return `
              <tr>
                <td><input type="checkbox" data-learning-row-id="${item.id}" /></td>
                <td>${item.customer_id}</td>
                <td>${item.churn}</td>
                <td>${preview}</td>
              </tr>
            `;
          }).join("")}
        </tbody>
      </table>
    `;
  } catch (error) {
    learningReviewSummary.textContent = error.message;
    learningReviewSummary.style.color = "var(--red)";
  }
}

async function reviewSelectedLearningRows(status) {
  const ids = Array.from(document.querySelectorAll("[data-learning-row-id]:checked"))
    .map((input) => Number(input.dataset.learningRowId));
  if (!ids.length) {
    if (learningReviewSummary) learningReviewSummary.textContent = "Select at least one row first.";
    return;
  }
  try {
    await postJson("/learning/review", { row_ids: ids, status });
    await loadLearningReview();
    await loadAdminSummary();
    await loadLabMetrics();
  } catch (error) {
    if (learningReviewSummary) {
      learningReviewSummary.textContent = error.message;
      learningReviewSummary.style.color = "var(--red)";
    }
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

function driftStatusLabel(status) {
  if (status === "high") return "High drift";
  if (status === "watch") return "Watch";
  if (status === "stable") return "Stable";
  if (status === "warming_up") return "Warming up";
  return "Unavailable";
}

async function loadDriftMonitor() {
  const container = document.getElementById("labDriftMonitor");
  if (!container) return;
  try {
    const response = await fetch("/admin/drift", { headers: authHeaders() });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(formatApiError(data.detail, "Could not load drift status."));

    const features = data.features || [];
    const score = typeof data.overall_score === "number" ? `${Math.round(data.overall_score * 100)}%` : "--";
    container.innerHTML = `
      <div class="drift-summary drift-summary--${data.status}">
        <div>
          <span>Status</span>
          <strong>${driftStatusLabel(data.status)}</strong>
        </div>
        <div>
          <span>Drift score</span>
          <strong>${score}</strong>
        </div>
        <div>
          <span>Recent predictions</span>
          <strong>${data.sample_size || 0}</strong>
        </div>
      </div>
      <p class="drift-message">${data.message || ""}</p>
      ${features.length ? `
        <div class="drift-feature-list">
          ${features.slice(0, 6).map((item) => `
            <div class="drift-feature">
              <div>
                <strong>${item.feature}</strong>
                <span>${item.type} · ${item.detail}</span>
              </div>
              <span class="drift-score drift-score--${item.status}">${Math.round(item.score * 100)}%</span>
            </div>
          `).join("")}
        </div>
      ` : ""}
    `;
  } catch (error) {
    container.innerHTML = `<p style="color:var(--red);font-size:14px;">${error.message}</p>`;
  }
}

function metricValue(metrics, key) {
  const value = metrics?.[key];
  if (typeof value !== "number") return "--";
  return value <= 1 ? (value * 100).toFixed(1) + "%" : value.toFixed(3);
}

function metricDelta(candidateMetrics, productionMetrics, key) {
  const candidate = candidateMetrics?.[key];
  const production = productionMetrics?.[key];
  if (typeof candidate !== "number" || typeof production !== "number") return "";
  const delta = candidate - production;
  const good = key === "brier" ? delta < 0 : delta > 0;
  const neutral = Math.abs(delta) < 0.0001;
  const cls = neutral ? "promotion-delta" : `promotion-delta ${good ? "good" : "bad"}`;
  const formatted = Math.abs(delta) <= 1 ? (delta * 100).toFixed(1) + " pts" : delta.toFixed(3);
  return `<span class="${cls}">${delta >= 0 ? "+" : ""}${formatted}</span>`;
}

function renderQualityGate(gate) {
  if (!gate) return "";
  const passed = Boolean(gate.passed);
  const blockers = gate.blockers || [];
  const warnings = gate.warnings || [];
  const score = typeof gate.score_delta === "number"
    ? `${gate.score_delta >= 0 ? "+" : ""}${(gate.score_delta * 100).toFixed(1)} pts balanced score`
    : "No balanced-score comparison";
  return `
    <div class="quality-gate ${passed ? "quality-gate--pass" : "quality-gate--block"}">
      <strong>${passed ? "Quality gate passed" : "Quality gate blocked promotion"}</strong>
      <span>${score}</span>
      ${blockers.length ? `<ul>${blockers.map((item) => `<li>${item}</li>`).join("")}</ul>` : ""}
      ${warnings.length ? `<ul>${warnings.map((item) => `<li>${item}</li>`).join("")}</ul>` : ""}
    </div>
  `;
}

function formatRunDate(value) {
  if (!value) return "In progress";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function runStatusLabel(status) {
  return String(status || "unknown").replaceAll("_", " ");
}

function renderRunMetrics(metrics = {}) {
  const metricKeys = [
    ["ROC-AUC", "roc_auc"],
    ["PR-AUC", "pr_auc"],
    ["F1", "f1"],
    ["Brier", "brier"],
  ];
  const items = metricKeys.filter(([, key]) => typeof metrics[key] === "number");
  if (!items.length) return '<span class="history-muted">No metric snapshot</span>';
  return items.map(([label, key]) => (
    `<span><strong>${metricValue(metrics, key)}</strong>${label}</span>`
  )).join("");
}

async function loadPromotionHistory() {
  if (!promotionHistoryPanel) return;
  try {
    const response = await fetch("/admin/retrain/status", { headers: authHeaders() });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(formatApiError(data.detail, "Could not load promotion history."));

    const runs = data.runs || [];
    if (!runs.length) {
      promotionHistoryPanel.innerHTML = `
        <div class="promotion-empty">
          <strong>No training runs yet</strong>
          <span>Train a candidate model to start the promotion timeline.</span>
        </div>
      `;
      return;
    }

    promotionHistoryPanel.innerHTML = runs.slice(0, 10).map((run) => {
      const metrics = run.metrics || {};
      const forced = Boolean(metrics.force_promoted || run.artifact_paths?.force_promoted);
      const status = forced && run.status === "promoted" ? "force-promoted" : run.status;
      const statusClass = String(run.status || "unknown").replaceAll("_", "-");
      return `
        <div class="history-run history-run--${statusClass}">
          <div class="history-run__main">
            <div>
              <span class="history-status">${runStatusLabel(status)}</span>
              <strong>Run #${run.id}${run.model_family ? ` · ${run.model_family}` : ""}</strong>
            </div>
            <time>${formatRunDate(run.finished_at || run.started_at)}</time>
          </div>
          <div class="history-run__metrics">
            ${renderRunMetrics(metrics)}
          </div>
          ${run.error_message ? `<p class="history-error">${run.error_message}</p>` : ""}
        </div>
      `;
    }).join("");
  } catch (error) {
    promotionHistoryPanel.innerHTML = `<p style="color:var(--red);font-size:14px;">${error.message}</p>`;
  }
}

async function loadModelPromotion() {
  if (!modelPromotionPanel) return;
  try {
    const response = await fetch("/admin/model/candidate", { headers: authHeaders() });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(formatApiError(data.detail, "Could not load model promotion status."));

    const production = data.production?.metadata || {};
    const candidate = data.candidate?.metadata || {};
    const productionMetrics = production.metrics || {};
    const candidateMetrics = candidate.metrics || {};
    const run = data.candidate?.run;
    const canPromote = Boolean(data.candidate?.can_promote);
    const qualityGate = data.candidate?.quality_gate;
    const metrics = ["roc_auc", "pr_auc", "f1", "precision", "recall", "brier"];

    promoteModelCandidate.disabled = !canPromote;
    rejectModelCandidate.disabled = !run;

    if (!run && !data.candidate?.exists) {
      modelPromotionPanel.innerHTML = `
        <div class="promotion-empty">
          <strong>No candidate model yet</strong>
          <span>Run retraining after approving learning rows. The live production model will keep serving while the candidate trains.</span>
        </div>
      `;
      return;
    }

    modelPromotionPanel.innerHTML = `
      <div class="promotion-summary">
        <div>
          <span>Production</span>
          <strong>${production.model_family || "Current model"}</strong>
        </div>
        <div>
          <span>Candidate</span>
          <strong>${candidate.model_family || run?.model_family || "Candidate model"}</strong>
        </div>
        <div>
          <span>Status</span>
          <strong>${run?.status || "artifact found"}</strong>
        </div>
      </div>
      <div class="promotion-metrics">
        <table>
          <thead><tr><th>Metric</th><th>Production</th><th>Candidate</th><th>Delta</th></tr></thead>
          <tbody>
            ${metrics.map((key) => `
              <tr>
                <td>${key.replace("_", " ").toUpperCase()}</td>
                <td>${metricValue(productionMetrics, key)}</td>
                <td>${metricValue(candidateMetrics, key)}</td>
                <td>${metricDelta(candidateMetrics, productionMetrics, key)}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
      ${renderQualityGate(qualityGate)}
    `;
  } catch (error) {
    modelPromotionPanel.innerHTML = `<p style="color:var(--red);font-size:14px;">${error.message}</p>`;
    if (promoteModelCandidate) promoteModelCandidate.disabled = true;
    if (rejectModelCandidate) rejectModelCandidate.disabled = true;
  }
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
    loadLearningReview();
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
    loadLearningReview();
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
    await loadDriftMonitor();
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
    await loadDriftMonitor();
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
      msg.textContent = `✅ ${data.message} Run #${data.run_id || ""}`;
      msg.style.color = "var(--green)";
    }
    loadModelPromotion();
    loadPromotionHistory();
  } catch (error) {
    if (msg) {
      msg.textContent = "❌ " + error.message;
      msg.style.color = "var(--red)";
    }
  }
});

schemaEditorForm?.addEventListener("submit", saveSchemaEditor);
document.getElementById("approveLearningRows")?.addEventListener("click", () => {
  reviewSelectedLearningRows("approved_for_training");
});
document.getElementById("rejectLearningRows")?.addEventListener("click", () => {
  reviewSelectedLearningRows("rejected");
});

trainModelCandidate?.addEventListener("click", async () => {
  if (modelPromotionMessage) {
    modelPromotionMessage.textContent = "Starting candidate training...";
    modelPromotionMessage.style.color = "inherit";
  }
  try {
    const data = await postJson("/admin/retrain", {});
    if (modelPromotionMessage) {
      modelPromotionMessage.textContent = `${data.message} Run #${data.run_id || ""}`;
      modelPromotionMessage.style.color = "var(--green)";
    }
    await loadModelPromotion();
    await loadPromotionHistory();
  } catch (error) {
    if (modelPromotionMessage) {
      modelPromotionMessage.textContent = error.message;
      modelPromotionMessage.style.color = "var(--red)";
    }
  }
});

promoteModelCandidate?.addEventListener("click", async () => {
  if (modelPromotionMessage) {
    modelPromotionMessage.textContent = "Promoting candidate...";
    modelPromotionMessage.style.color = "inherit";
  }
  try {
    const data = await postJson("/admin/model/promote", {});
    if (modelPromotionMessage) {
      modelPromotionMessage.textContent = `Candidate promoted from run #${data.run_id}. ${data.used_learning_rows || 0} learning rows marked used.`;
      modelPromotionMessage.style.color = "var(--green)";
    }
    await loadModelInfo();
    await loadLabMetrics();
    await loadModelPromotion();
    await loadPromotionHistory();
  } catch (error) {
    const gate = error.detail?.quality_gate;
    if (gate && confirm("This candidate failed quality checks. Promote it anyway?")) {
      try {
        const forced = await postJson("/admin/model/promote", { force: true });
        if (modelPromotionMessage) {
          modelPromotionMessage.textContent = `Candidate force-promoted from run #${forced.run_id}. ${forced.used_learning_rows || 0} learning rows marked used.`;
          modelPromotionMessage.style.color = "var(--amber)";
        }
        await loadModelInfo();
        await loadLabMetrics();
        await loadModelPromotion();
        await loadPromotionHistory();
        return;
      } catch (forcedError) {
        if (modelPromotionMessage) {
          modelPromotionMessage.textContent = forcedError.message;
          modelPromotionMessage.style.color = "var(--red)";
        }
        return;
      }
    }
    if (modelPromotionMessage) {
      modelPromotionMessage.textContent = error.message;
      modelPromotionMessage.style.color = "var(--red)";
    }
  }
});

rejectModelCandidate?.addEventListener("click", async () => {
  if (modelPromotionMessage) {
    modelPromotionMessage.textContent = "Rejecting candidate...";
    modelPromotionMessage.style.color = "inherit";
  }
  try {
    const data = await postJson("/admin/model/reject", {});
    if (modelPromotionMessage) {
      modelPromotionMessage.textContent = `Candidate rejected from run #${data.run_id}.`;
      modelPromotionMessage.style.color = "var(--green)";
    }
    await loadModelPromotion();
    await loadPromotionHistory();
  } catch (error) {
    if (modelPromotionMessage) {
      modelPromotionMessage.textContent = error.message;
      modelPromotionMessage.style.color = "var(--red)";
    }
  }
});

whatIfUseCurrent?.addEventListener("click", () => {
  whatIfState.baseline = cloneCustomer(customerFromForm());
  whatIfState.baseline.customerID = "WHAT-IF-A";
  whatIfState.baselinePrediction = null;
  whatIfState.scenarioPrediction = null;
  renderProfileSummary(whatIfBaseline, whatIfState.baseline);
  renderWhatIfComparison();
  setWhatIfMessage("Profile A captured from the sidebar.", "var(--green)");
});

whatIfGeneratePlan?.addEventListener("click", () => {
  if (!whatIfState.baseline) {
    whatIfState.baseline = cloneCustomer(customerFromForm());
    whatIfState.baseline.customerID = "WHAT-IF-A";
    renderProfileSummary(whatIfBaseline, whatIfState.baseline);
  }
  whatIfState.scenario = buildRetentionScenario(whatIfState.baseline);
  whatIfState.baselinePrediction = null;
  whatIfState.scenarioPrediction = null;
  renderProfileSummary(whatIfScenario, whatIfState.scenario, whatIfState.baseline);
  renderWhatIfComparison();
  const changes = changedFields(whatIfState.baseline, whatIfState.scenario).length;
  setWhatIfMessage(`${changes} scenario change${changes === 1 ? "" : "s"} generated.`, "var(--green)");
});

whatIfRunComparison?.addEventListener("click", async () => {
  if (!whatIfState.baseline) {
    whatIfState.baseline = cloneCustomer(customerFromForm());
    whatIfState.baseline.customerID = "WHAT-IF-A";
    renderProfileSummary(whatIfBaseline, whatIfState.baseline);
  }
  if (!whatIfState.scenario) {
    whatIfState.scenario = buildRetentionScenario(whatIfState.baseline);
    renderProfileSummary(whatIfScenario, whatIfState.scenario, whatIfState.baseline);
  }
  setWhatIfMessage("Scoring both profiles...");
  if (whatIfRunComparison) whatIfRunComparison.disabled = true;
  try {
    const [baselinePrediction, scenarioPrediction] = await Promise.all([
      postJson("/predict", whatIfState.baseline),
      postJson("/predict", whatIfState.scenario),
    ]);
    whatIfState.baselinePrediction = baselinePrediction;
    whatIfState.scenarioPrediction = scenarioPrediction;
    renderWhatIfComparison();
    await loadAdminSummary();
    await loadDriftMonitor();
    setWhatIfMessage("Comparison scored.", "var(--green)");
  } catch (error) {
    setWhatIfMessage(error.message, "var(--red)");
  } finally {
    if (whatIfRunComparison) whatIfRunComparison.disabled = false;
  }
});

document.body.addEventListener("click", (e) => {
  // Handle minimizing results
  const minimizeBtn = e.target.closest("#resultsMinimizeBtn");
  if (minimizeBtn) {
    resultsActions.classList.add("hidden");
    results.classList.add("hidden");
    featureDetail.classList.add("hidden");
    emptyState.classList.remove("hidden");
    return;
  }

  // Handle generating/testing a new customer
  const tryNewBtn = e.target.closest("#loadTestTrialBtn") || e.target.closest("#resultsTryNewBtn");
  if (!tryNewBtn) return;
  
  const container = document.getElementById("dynamicFormFields");
  if (!container) return;
  const inputs = container.querySelectorAll("input, select");
  inputs.forEach(input => {
    if (input.tagName.toLowerCase() === "select") {
      if (input.options.length > 0) {
        // Pick a random option index
        const randIdx = Math.floor(Math.random() * input.options.length);
        input.selectedIndex = randIdx;
      }
    } else if (input.type === "number") {
      const name = input.name.toLowerCase();
      if (name.includes("tenure")) {
        input.value = Math.floor(Math.random() * 71) + 1; // 1 to 72 months
      } else if (name.includes("charge") || name.includes("spent")) {
        input.value = Math.floor(Math.random() * 101) + 20; // 20 to 120
      } else if (name.includes("total")) {
        const tenureInput = Array.from(inputs).find(i => i.name.toLowerCase().includes("tenure"));
        const chargeInput = Array.from(inputs).find(i => i.name.toLowerCase().includes("charge"));
        const tVal = tenureInput ? Number(tenureInput.value) : 12;
        const cVal = chargeInput ? Number(chargeInput.value) : 65;
        input.value = (tVal * cVal).toFixed(2);
      } else {
        input.value = Math.floor(Math.random() * 10) + 1;
      }
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
