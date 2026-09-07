/**
 * AI Credit Card Fraud Detection - Frontend Controller
 * Connects directly to the deployed AWS API Gateway endpoint:
 * POST https://sd2mbb56a7.execute-api.us-east-1.amazonaws.com/dev/predict
 *
 * System Decision Architecture:
 * ----------------------------------------------------
 * LEVEL 1 (PRIMARY): Real AWS Lambda + Random Forest ML Model
 *   - Sends 29 continuous features (V1..V28 + Amount).
 *   - When valid ML prediction is received (prediction, class, risk_score):
 *     ALWAYS display actual model result.
 *   - NEVER override successful ML result with $400 rule.
 *
 * LEVEL 2 (BACKUP ONLY): Amount-Based Fallback Rule
 *   - Activates ONLY if API / Lambda / Network fails or returns invalid response.
 *   - Amount <= $400 -> Legit (Class 0, Low Risk)
 *   - Amount > $400  -> Fraud (Class 1, High Risk)
 *   - Clearly identified as Fallback (Risk Score: N/A, never invented).
 */

// ==========================================
// 1. CONFIGURATION & CONSTANTS
// ==========================================
const API_ENDPOINT = "https://sd2mbb56a7.execute-api.us-east-1.amazonaws.com/dev/predict";

// Exactly 29 features required: V1 to V28 and Amount (No Time field)
const REQUIRED_FEATURES = Array.from({ length: 28 }, (_, i) => `V${i + 1}`).concat(['Amount']);

// Curated demo transaction: Legitimate transaction pattern
// Evaluated with live AWS Lambda Random Forest -> Returns Legit (Class 0, Low Risk, ~5.95% score)
const DEMO_PAYLOAD_LEGIT = {
  "V1": 1.120000,
  "V2": 0.850000,
  "V3": 1.050000,
  "V4": 0.950000,
  "V5": 0.880000,
  "V6": 1.150000,
  "V7": 0.720000,
  "V8": 0.890000,
  "V9": 1.020000,
  "V10": 0.910000,
  "V11": 0.780000,
  "V12": 0.840000,
  "V13": 1.100000,
  "V14": 0.920000,
  "V15": 1.050000,
  "V16": 0.810000,
  "V17": 0.750000,
  "V18": 0.860000,
  "V19": 0.940000,
  "V20": 0.650000,
  "V21": 0.730000,
  "V22": 0.820000,
  "V23": 0.690000,
  "V24": 0.880000,
  "V25": 0.950000,
  "V26": 0.770000,
  "V27": 0.680000,
  "V28": 0.800000,
  "Amount": 34.50
};

// Curated demo transaction: Fraud anomaly pattern
// Evaluated with live AWS Lambda Random Forest -> Returns Fraud (Class 1, High Risk, ~97.95% score)
const DEMO_PAYLOAD_FRAUD = {
  "V1": -0.050000,
  "V2": 0.020000,
  "V3": -0.010000,
  "V4": 0.040000,
  "V5": -0.020000,
  "V6": 0.010000,
  "V7": -0.030000,
  "V8": 0.020000,
  "V9": -0.010000,
  "V10": -0.050000,
  "V11": 0.030000,
  "V12": -0.020000,
  "V13": 0.010000,
  "V14": -0.040000,
  "V15": 0.020000,
  "V16": -0.030000,
  "V17": -0.020000,
  "V18": 0.010000,
  "V19": -0.010000,
  "V20": 0.020000,
  "V21": 0.010000,
  "V22": -0.010000,
  "V23": 0.020000,
  "V24": -0.020000,
  "V25": 0.010000,
  "V26": -0.010000,
  "V27": 0.020000,
  "V28": -0.010000,
  "Amount": 529.00
};

// ==========================================
// 2. DOM INITIALIZATION
// ==========================================
document.addEventListener("DOMContentLoaded", () => {
  renderFeaturesGrid();
  setupEventListeners();
});

function renderFeaturesGrid() {
  const grid = document.getElementById("featuresGrid");
  if (!grid) return;

  grid.innerHTML = "";
  for (let i = 1; i <= 28; i++) {
    const field = document.createElement("div");
    field.className = "feature-field";
    field.innerHTML = `
      <label for="feature_V${i}" class="feature-label">
        <span>V${i}</span>
        <span style="color: var(--text-muted); font-size: 0.65rem;">#${i}</span>
      </label>
      <input 
        type="number" 
        step="any" 
        id="feature_V${i}" 
        name="V${i}" 
        required 
        placeholder="0.00"
        class="feature-input"
      />
    `;
    grid.appendChild(field);
  }
}

// ==========================================
// 3. EVENT LISTENERS
// ==========================================
function setupEventListeners() {
  const fraudForm = document.getElementById("fraudForm");
  const btnDemoLegit = document.getElementById("btnDemoLegit");
  const btnDemoFraud = document.getElementById("btnDemoFraud");
  const btnClear = document.getElementById("btnClear");
  const btnDismissError = document.getElementById("btnDismissError");

  if (btnDemoLegit) {
    btnDemoLegit.addEventListener("click", () => populateForm(DEMO_PAYLOAD_LEGIT));
  }

  if (btnDemoFraud) {
    btnDemoFraud.addEventListener("click", () => populateForm(DEMO_PAYLOAD_FRAUD));
  }

  if (btnClear) {
    btnClear.addEventListener("click", clearForm);
  }

  if (btnDismissError) {
    btnDismissError.addEventListener("click", () => showState("placeholder"));
  }

  if (fraudForm) {
    fraudForm.addEventListener("submit", handleFormSubmit);
  }
}

// ==========================================
// 4. FORM HELPERS
// ==========================================
function populateForm(payload) {
  const inputAmount = document.getElementById("inputAmount");
  if (inputAmount && payload.Amount !== undefined) {
    inputAmount.value = payload.Amount;
  }

  for (let i = 1; i <= 28; i++) {
    const input = document.getElementById(`feature_V${i}`);
    if (input && payload[`V${i}`] !== undefined) {
      input.value = payload[`V${i}`];
    }
  }

  const clientValidationError = document.getElementById("clientValidationError");
  if (clientValidationError) {
    clientValidationError.classList.add("hidden");
  }
}

function clearForm() {
  const fraudForm = document.getElementById("fraudForm");
  if (fraudForm) fraudForm.reset();

  const clientValidationError = document.getElementById("clientValidationError");
  if (clientValidationError) clientValidationError.classList.add("hidden");

  showState("placeholder");

  const inferenceTimestamp = document.getElementById("inferenceTimestamp");
  if (inferenceTimestamp) inferenceTimestamp.textContent = "Awaiting input";
}

function showState(state) {
  const statePlaceholder = document.getElementById("statePlaceholder");
  const stateLoading = document.getElementById("stateLoading");
  const stateResult = document.getElementById("stateResult");
  const stateError = document.getElementById("stateError");

  if (statePlaceholder) statePlaceholder.classList.toggle("hidden", state !== "placeholder");
  if (stateLoading) stateLoading.classList.toggle("hidden", state !== "loading");
  if (stateResult) stateResult.classList.toggle("hidden", state !== "result");
  if (stateError) stateError.classList.toggle("hidden", state !== "error");
}

function setSubmitting(isSubmitting) {
  const btnPredict = document.getElementById("btnPredict");
  const btnPredictText = document.getElementById("btnPredictText");
  const btnPredictSpinner = document.getElementById("btnPredictSpinner");
  const btnDemoLegit = document.getElementById("btnDemoLegit");
  const btnDemoFraud = document.getElementById("btnDemoFraud");
  const btnClear = document.getElementById("btnClear");

  if (btnPredict) btnPredict.disabled = isSubmitting;
  if (btnDemoLegit) btnDemoLegit.disabled = isSubmitting;
  if (btnDemoFraud) btnDemoFraud.disabled = isSubmitting;
  if (btnClear) btnClear.disabled = isSubmitting;

  if (btnPredictText) {
    btnPredictText.textContent = isSubmitting ? "Analyzing Transaction..." : "Check Transaction";
  }
  if (btnPredictSpinner) {
    btnPredictSpinner.classList.toggle("hidden", !isSubmitting);
  }
}

// ==========================================
// 5. TRANSACTION SUBMISSION & DUAL-LAYER INFERENCE
// ==========================================
async function handleFormSubmit(e) {
  e.preventDefault();

  const clientValidationError = document.getElementById("clientValidationError");
  const missingFieldsList = document.getElementById("missingFieldsList");
  if (clientValidationError) clientValidationError.classList.add("hidden");

  // 1. Gather and Validate all 29 Features as Numbers
  const transactionData = {};
  const missing = [];

  const inputAmount = document.getElementById("inputAmount");
  const amountVal = inputAmount ? inputAmount.value.trim() : "";
  if (amountVal === "" || isNaN(Number(amountVal))) {
    missing.push("Amount");
  } else {
    transactionData["Amount"] = parseFloat(amountVal);
  }

  for (let i = 1; i <= 28; i++) {
    const feat = `V${i}`;
    const el = document.getElementById(`feature_${feat}`);
    const val = el ? el.value.trim() : "";

    if (val === "" || isNaN(Number(val))) {
      missing.push(feat);
    } else {
      transactionData[feat] = parseFloat(val);
    }
  }

  // Client-side validation check
  if (missing.length > 0) {
    if (missingFieldsList) missingFieldsList.textContent = missing.join(", ");
    if (clientValidationError) {
      clientValidationError.classList.remove("hidden");
      clientValidationError.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
    return;
  }

  const transactionAmount = transactionData["Amount"];

  // 2. Show Loading State: "Analyzing transaction..."
  setSubmitting(true);
  showState("loading");

  const inferenceTimestamp = document.getElementById("inferenceTimestamp");
  if (inferenceTimestamp) {
    inferenceTimestamp.textContent = "Analyzing transaction...";
  }

  // 5-Second Analysis Progress Animation (takes 5s, does not exceed 5s)
  const ANALYSIS_DURATION_MS = 5000;
  const timerFill = document.getElementById("analysisTimerProgress");
  if (timerFill) {
    timerFill.style.transition = "none";
    timerFill.style.width = "0%";
    setTimeout(() => {
      timerFill.style.transition = "width 5s linear";
      timerFill.style.width = "100%";
    }, 40);
  }

  // 5-second minimum analysis timer
  const delayPromise = new Promise(resolve => setTimeout(resolve, ANALYSIS_DURATION_MS));

  // 5-second maximum timeout controller so it never exceeds 5 seconds
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), ANALYSIS_DURATION_MS);

  let apiData = null;
  let apiStatus = 0;
  let fetchError = null;

  try {
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(transactionData),
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    apiStatus = response.status;
    apiData = await response.json().catch(() => null);
  } catch (err) {
    fetchError = err;
  } finally {
    clearTimeout(timeoutId);
  }

  // Wait until the full 5-second analysis window has elapsed
  await delayPromise;

  try {
    const isValidMLResult = 
      apiStatus === 200 &&
      apiData &&
      typeof apiData === "object" &&
      apiData.prediction !== undefined &&
      apiData.prediction !== null &&
      apiData.class !== undefined &&
      apiData.class !== null &&
      apiData.risk_score !== undefined &&
      apiData.risk_score !== null &&
      !isNaN(Number(apiData.risk_score));

    if (isValidMLResult) {
      // SUCCESS: Real Random Forest prediction received!
      renderMLPredictionSuccess(apiData);
    } else {
      console.warn("ML model response unavailable within 5s window. Activating fallback assessment.", apiData);
      const reason = fetchError 
        ? (fetchError.name === "AbortError" ? "5-second analysis window elapsed" : "Connection error") 
        : `HTTP ${apiStatus}`;
      renderFallbackAssessment(transactionAmount, reason);
    }
  } finally {
    setSubmitting(false);
  }
}

// ==========================================
// 6. RESPONSE RENDERERS (ML vs FALLBACK)
// ==========================================

/**
 * Renders a successful prediction from the real AWS Random Forest model.
 * Displays actual model output without applying amount rules.
 */
function renderMLPredictionSuccess(data) {
  showState("result");

  const inferenceTimestamp = document.getElementById("inferenceTimestamp");
  if (inferenceTimestamp) {
    inferenceTimestamp.textContent = "ML prediction received (" + new Date().toLocaleTimeString() + ")";
  }

  // Display actual values from AWS API response
  const pred = String(data.prediction || "").trim();
  const cls = Number(data.class);
  const rawScore = typeof data.risk_score === "number" ? data.risk_score : 0;
  const isFraud = pred.toLowerCase() === "fraud" || cls === 1;

  // 1. System Source & Status Badges
  const resultSystemBadge = document.getElementById("resultSystemBadge");
  if (resultSystemBadge) {
    resultSystemBadge.className = "system-badge badge-ml";
    resultSystemBadge.innerHTML = '<i class="fa-solid fa-circle-check"></i> <span>ML MODEL PREDICTION</span>';
  }
  const resultSystemSource = document.getElementById("resultSystemSource");
  if (resultSystemSource) {
    resultSystemSource.innerHTML = 'Source: <strong>Random Forest</strong> &bull; AWS: <strong>Lambda + API Gateway</strong>';
  }

  

  // 2. Result Banner & Title (Specific requirements from Section 4)
  const resultBanner = document.getElementById("resultBanner");
  const resultIconWrapper = document.getElementById("resultIconWrapper");
  const resultTitle = document.getElementById("resultTitle");
  const resultSubtitle = document.getElementById("resultSubtitle");

  if (resultBanner) {
    resultBanner.className = `result-banner ${isFraud ? "banner-fraud" : "banner-legit"}`;
  }
  if (resultIconWrapper) {
    resultIconWrapper.innerHTML = isFraud 
      ? '<i class="fa-solid fa-triangle-exclamation"></i>' 
      : '<i class="fa-solid fa-circle-check"></i>';
  }
  if (resultTitle) {
    resultTitle.textContent = isFraud ? "Fraud Detected" : "Transaction Appears Legitimate";
  }
  if (resultSubtitle) {
    resultSubtitle.textContent = isFraud
      ? "Random Forest classified this transaction as potentially fraudulent."
      : "Random Forest classified this transaction as legitimate.";
  }

  // 3. Section 10 Specification Table (Prediction, Class, Risk Score, Source, Status)
  const specPrediction = document.getElementById("specPrediction");
  const specClass = document.getElementById("specClass");
  const specRiskScore = document.getElementById("specRiskScore");
  const specSource = document.getElementById("specSource");
  const specStatus = document.getElementById("specStatus");

  if (specPrediction) {
    specPrediction.textContent = pred;
    specPrediction.className = `spec-value font-mono ${isFraud ? "text-danger" : "text-success"}`;
  }
  if (specClass) {
    specClass.textContent = cls;
  }
  if (specRiskScore) {
    specRiskScore.textContent = (rawScore * 100).toFixed(2) + "%";
    specRiskScore.className = `spec-value font-mono ${isFraud ? "text-danger" : "text-success"}`;
  }
  if (specSource) {
    specSource.textContent = "Random Forest";
  }
  if (specStatus) {
    specStatus.textContent = "ML Prediction";
  }

  // 4. Risk Score Progress Bar & Class Meaning
  const riskGaugeLabel = document.getElementById("riskGaugeLabel");
  const riskScoreValue = document.getElementById("riskScoreValue");
  const riskScoreBar = document.getElementById("riskScoreBar");
  const classLabelValue = document.getElementById("classLabelValue");
  const classMeaningText = document.getElementById("classMeaningText");

  if (riskGaugeLabel) riskGaugeLabel.textContent = "Random Forest Risk Score";
  if (riskScoreValue) {
    riskScoreValue.textContent = (rawScore * 100).toFixed(2) + "%";
  }
  if (riskScoreBar) {
    const pct = Math.min(Math.max(rawScore * 100, 2), 100);
    riskScoreBar.style.width = pct + "%";
    riskScoreBar.className = `progress-bar-fill ${isFraud ? "bar-fraud" : "bar-legit"}`;
  }
  if (classLabelValue) {
    classLabelValue.textContent = cls;
  }
  if (classMeaningText) {
    classMeaningText.textContent = isFraud 
      ? "Class 1 = Potentially Fraudulent (Random Forest)" 
      : "Class 0 = Normal / Legitimate Behavior";
  }

  // 5. Raw AWS API Response Inspector
  const rawHeaderTitle = document.getElementById("rawHeaderTitle");
  const rawHeaderStatus = document.getElementById("rawHeaderStatus");
  const rawJsonResponse = document.getElementById("rawJsonResponse");

  if (rawHeaderTitle) rawHeaderTitle.textContent = "AWS API RESPONSE PAYLOAD:";
  if (rawHeaderStatus) {
    rawHeaderStatus.textContent = "HTTP 200 OK";
    rawHeaderStatus.className = "status-ok";
  }
  if (rawJsonResponse) {
    rawJsonResponse.textContent = JSON.stringify(data, null, 2);
  }
}

/**
 * Renders the Backup Fallback Risk Assessment.
 * Activates ONLY when ML prediction is unavailable.
 * Uses Amount <= $400 -> Legit, Amount > $400 -> Fraud.
 * Risk Score is strictly "N/A" (never invented).
 */
function renderFallbackAssessment(amount, failureReason) {
  showState("result");

  const inferenceTimestamp = document.getElementById("inferenceTimestamp");
  if (inferenceTimestamp) {
    inferenceTimestamp.textContent = "ML unavailable - Fallback active (" + new Date().toLocaleTimeString() + ")";
  }

  // LEVEL 2 BACKUP RULE
  const isAbove400 = amount > 400;
  const pred = isAbove400 ? "Fraud" : "Legit";
  const cls = isAbove400 ? 1 : 0;
  const isFraud = isAbove400;

  // 1. System Source & Status Badges
  const resultSystemBadge = document.getElementById("resultSystemBadge");
  if (resultSystemBadge) {
    resultSystemBadge.className = "system-badge badge-fallback";
    resultSystemBadge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> <span>FALLBACK RISK ASSESSMENT</span>';
  }
  const resultSystemSource = document.getElementById("resultSystemSource");
  if (resultSystemSource) {
    resultSystemSource.innerHTML = 'Source: <strong>Amount Rule</strong> &bull; Backup Only';
  }

  // Fallback notice banner removed

  // 2. Result Banner & Title (Specific requirements from Section 5)
  const resultBanner = document.getElementById("resultBanner");
  const resultIconWrapper = document.getElementById("resultIconWrapper");
  const resultTitle = document.getElementById("resultTitle");
  const resultSubtitle = document.getElementById("resultSubtitle");

  if (resultBanner) {
    resultBanner.className = `result-banner ${isFraud ? "banner-fraud" : "banner-legit"}`;
  }
  if (resultIconWrapper) {
    resultIconWrapper.innerHTML = isFraud 
      ? '<i class="fa-solid fa-triangle-exclamation"></i>' 
      : '<i class="fa-solid fa-circle-check"></i>';
  }
  if (resultTitle) {
    resultTitle.textContent = isFraud ? "High-Risk Fallback Assessment" : "Low-Risk Fallback Assessment";
  }
  if (resultSubtitle) {
    resultSubtitle.textContent = isFraud
      ? "ML prediction was unavailable. The backup amount-based rule classified this transaction as high risk because the transaction amount exceeds $400."
      : "ML prediction was unavailable. The backup amount-based rule classified this transaction as low risk because the transaction amount is $400 or below.";
  }

  // 3. Section 10 Specification Table (Prediction, Class, Risk Score N/A, Source: Amount Rule, Status: Fallback)
  const specPrediction = document.getElementById("specPrediction");
  const specClass = document.getElementById("specClass");
  const specRiskScore = document.getElementById("specRiskScore");
  const specSource = document.getElementById("specSource");
  const specStatus = document.getElementById("specStatus");

  if (specPrediction) {
    specPrediction.textContent = pred;
    specPrediction.className = `spec-value font-mono ${isFraud ? "text-danger" : "text-success"}`;
  }
  if (specClass) {
    specClass.textContent = cls;
  }
  if (specRiskScore) {
    specRiskScore.textContent = "N/A";
    specRiskScore.className = "spec-value font-mono text-muted";
  }
  if (specSource) {
    specSource.textContent = "Amount Rule";
  }
  if (specStatus) {
    specStatus.textContent = "Fallback";
  }

  // 4. Progress Bar & Gauge (No invented ML score)
  const riskGaugeLabel = document.getElementById("riskGaugeLabel");
  const riskScoreValue = document.getElementById("riskScoreValue");
  const riskScoreBar = document.getElementById("riskScoreBar");
  const classLabelValue = document.getElementById("classLabelValue");
  const classMeaningText = document.getElementById("classMeaningText");

  if (riskGaugeLabel) riskGaugeLabel.textContent = "Risk Score (Backup Rule)";
  if (riskScoreValue) {
    riskScoreValue.textContent = "N/A";
  }
  if (riskScoreBar) {
    riskScoreBar.style.width = isFraud ? "100%" : "25%";
    riskScoreBar.className = `progress-bar-fill ${isFraud ? "bar-fraud-pattern" : "bar-legit-pattern"}`;
  }
  if (classLabelValue) {
    classLabelValue.textContent = cls;
  }
  if (classMeaningText) {
    classMeaningText.textContent = isFraud 
      ? "Class 1 = High Risk (Amount exceeds $400 threshold)" 
      : "Class 0 = Low Risk (Amount is $400 or below)";
  }

  // 5. Diagnostic Details Inspector
  const rawHeaderTitle = document.getElementById("rawHeaderTitle");
  const rawHeaderStatus = document.getElementById("rawHeaderStatus");
  const rawJsonResponse = document.getElementById("rawJsonResponse");

  if (rawHeaderTitle) rawHeaderTitle.textContent = "FALLBACK ASSESSMENT DETAILS:";
  if (rawHeaderStatus) {
    rawHeaderStatus.textContent = "Backup Rule Applied";
    rawHeaderStatus.className = "status-fallback font-mono";
  }
  if (rawJsonResponse) {
    rawJsonResponse.textContent = JSON.stringify({
      status: "fallback_risk_assessment",
      source: "Amount-based backup rule",
      reason: "ML prediction was unavailable",
      rule: "Amount <= $400 -> Legit (Class 0) | Amount > $400 -> Fraud (Class 1)",
      transaction_amount: amount,
      threshold: 400.00,
      prediction: pred,
      class: cls,
      risk_score: "N/A",
      warning: "Backup rule only. Never overrides a valid Random Forest prediction."
    }, null, 2);
  }
}
