const authView = document.querySelector("#authView");
const dashboardView = document.querySelector("#dashboardView");
const signinTab = document.querySelector("#signinTab");
const signupTab = document.querySelector("#signupTab");
const signinForm = document.querySelector("#signinForm");
const signupForm = document.querySelector("#signupForm");
const authMessage = document.querySelector("#authMessage");
const signedInUser = document.querySelector("#signedInUser");
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
let lastCustomer = null;
let lastPrediction = null;

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
}

function clearSession() {
  localStorage.removeItem("churnguard_user");
  dashboardView.classList.add("hidden");
  authView.classList.remove("hidden");
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

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "Request failed.");
  }
  return data;
}

async function postFile(url, file) {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(url, { method: "POST", body });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "Upload failed.");
  }
  return data;
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

async function loadModelInfo() {
  try {
    const response = await fetch("/model-info");
    if (!response.ok) return;
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
    setSession(data.username);
  } catch (error) {
    authMessage.textContent = error.message;
  }
});

signupForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  authMessage.textContent = "";
  try {
    const data = await postJson("/auth/signup", formToObject(signupForm));
    setSession(data.username);
  } catch (error) {
    authMessage.textContent = error.message;
  }
});

logoutButton.addEventListener("click", clearSession);
toggleSidebar.addEventListener("click", () => {
  setSidebarHidden(!document.body.classList.contains("sidebar-hidden"));
});
openLab.addEventListener("click", () => showPage("lab"));
backToDashboard.addEventListener("click", () => showPage("dashboard"));
document.querySelector("#highPreset").addEventListener("click", () => applyPreset(highRiskPreset));
document.querySelector("#lowPreset").addEventListener("click", () => applyPreset(lowRiskPreset));
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
  } catch (error) {
    batchSummary.textContent = error.message;
  }
});

learningCsv.addEventListener("change", async () => {
  const file = learningCsv.files?.[0];
  if (!file) return;
  learningStatus.textContent = "Uploading labeled rows...";
  try {
    const data = await postFile("/learning/upload", file);
    learningStatus.textContent = `${data.accepted_rows} rows added to the learning queue. ${data.stored_rows} rows stored total.`;
  } catch (error) {
    learningStatus.textContent = error.message;
  }
});

const existingUser = localStorage.getItem("churnguard_user");
if (existingUser) {
  setSession(existingUser);
} else {
  clearSession();
}
loadModelInfo();
setSidebarHidden(localStorage.getItem("churnguard_sidebar_hidden") === "1");
