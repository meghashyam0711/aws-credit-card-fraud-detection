/**
 * Architecture & Animated Workflow Controller
 * Drives the real-time inference transaction simulation, playback controls,
 * Dual-Layer Decision Logic (Level 1 ML vs Level 2 Backup), Lambda internal sequence,
 * and component inspection.
 */

// Simulation Mode State: 'ml' (Primary Flow) or 'fallback' (Backup Flow)
let SIMULATION_MODE = "ml";

// ===========================================================================
// 1. WORKFLOW STEP DEFINITIONS (Dual-Layer Decision Architecture)
// ===========================================================================
const WORKFLOW_STEPS = [
  {
    stepNum: "01",
    stepTotal: "09",
    nodeId: "node-user",
    packetId: "packet-1",
    name: "USER",
    phase: "Transaction Ingestion",
    description: "Enters transaction information (Amount + 28 PCA numerical features) on the frontend dashboard.",
    progressPct: "11%",
    durationMs: 1400,
    sideHighlight: null,
    lambdaSubstep: null,
    showSimResult: false,
    returnStepId: null
  },
  {
    stepNum: "02",
    stepTotal: "09",
    nodeId: "node-frontend",
    packetId: "packet-2",
    name: "FRONTEND DASHBOARD",
    phase: "Client Validation",
    description: "Collects and validates 29 numerical features. Verifies non-empty numbers and serializes into JSON.",
    progressPct: "22%",
    durationMs: 1400,
    sideHighlight: null,
    lambdaSubstep: null,
    showSimResult: false,
    returnStepId: null
  },
  {
    stepNum: "03",
    stepTotal: "09",
    nodeId: "node-gateway",
    packetId: "packet-3",
    name: "AMAZON API GATEWAY",
    phase: "HTTP API Gateway Ingress",
    description: "Receives POST /predict request, enforces CORS (Access-Control-Allow-Origin: *), terminates SSL, and invokes Lambda.",
    progressPct: "33%",
    durationMs: 1400,
    sideHighlight: null,
    lambdaSubstep: null,
    showSimResult: false,
    returnStepId: null
  },
  {
    stepNum: "04",
    stepTotal: "09",
    nodeId: "node-lambda",
    packetId: "packet-4",
    name: "AWS LAMBDA",
    phase: "Serverless Preprocessing",
    description: "Validates request, standardizes Amount via scaler, and arranges pandas DataFrame in model.feature_names_in_ order.",
    progressPct: "50%",
    durationMs: 3000,
    sideHighlight: "side-s3",
    lambdaSubstep: "animate-lambda",
    showSimResult: false,
    returnStepId: null
  },
  {
    stepNum: "05",
    stepTotal: "09",
    nodeId: "node-model",
    packetId: "packet-5",
    name: "RANDOM FOREST CLASSIFIER",
    phase: "Machine Learning Scoring",
    description: "Performs inference across 50 decision trees using in-memory model weights cached in Lambda.",
    progressPct: "65%",
    durationMs: 1600,
    sideHighlight: "side-cloudwatch",
    lambdaSubstep: null,
    showSimResult: false,
    returnStepId: null
  },
  {
    stepNum: "06",
    stepTotal: "09",
    nodeId: "node-decision",
    packetId: null,
    name: "DECISION GATE: VALID PREDICTION?",
    phase: "Dual-Layer Risk Evaluation",
    description: "Evaluates prediction availability: YES -> Use Random Forest Result; NO -> Activate Backup Fallback Rule ($400 threshold).",
    progressPct: "78%",
    durationMs: 2000,
    sideHighlight: null,
    lambdaSubstep: null,
    showSimResult: true,
    returnStepId: "r-step-pred"
  },
  {
    stepNum: "07",
    stepTotal: "09",
    nodeId: "node-gateway",
    packetId: "packet-return",
    name: "API GATEWAY RESPONSE",
    phase: "HTTP Response Return",
    description: "Returns the structured JSON response payload (HTTP 200 OK) back across the network boundary.",
    progressPct: "88%",
    durationMs: 1300,
    sideHighlight: null,
    lambdaSubstep: null,
    showSimResult: true,
    returnStepId: "r-step-gateway"
  },
  {
    stepNum: "08",
    stepTotal: "09",
    nodeId: "node-frontend",
    packetId: "packet-return",
    name: "FRONTEND RESULT",
    phase: "UI Presentation",
    description: "Displays prediction banner, risk probability gauge, and class label, clearly identifying ML vs Fallback source.",
    progressPct: "95%",
    durationMs: 1400,
    sideHighlight: null,
    lambdaSubstep: null,
    showSimResult: true,
    returnStepId: "r-step-frontend"
  },
  {
    stepNum: "09",
    stepTotal: "09",
    nodeId: "node-user",
    packetId: null,
    name: "USER",
    phase: "Assessment Received",
    description: "User inspects real-time fraud assessment. Primary ML completes in < 350 ms; Fallback acts as continuous safety net.",
    progressPct: "100%",
    durationMs: 2000,
    sideHighlight: null,
    lambdaSubstep: null,
    showSimResult: true,
    returnStepId: "r-step-user"
  }
];

// ===========================================================================
// 2. COMPONENT ENCYCLOPEDIA (Detailed inspector descriptions)
// ===========================================================================
const COMPONENT_DETAILS = {
  lambda: {
    badge: "AWS COMPUTE &bull; SERVERLESS",
    title: "AWS Lambda (Python 3.12)",
    type: "Serverless Execution Engine",
    desc: "AWS Lambda performs serverless inference by parsing the incoming request, validating that all 29 numerical features exist, preparing the feature vector in exact sequence from model.feature_names_in_, standardizing ONLY the Amount column with StandardScaler, loading the ML artifacts when required, running the Scikit-Learn Random Forest model, and returning the prediction and calibrated risk score.",
    chips: [
      "Runtime: Python 3.12 (x86_64)",
      "Memory: 512 MB Provisioned",
      "Warm Start Duration: ~45ms",
      "Preprocessing: Amount-Only Scaling",
      "Model Cache: Global Memory Reuse"
    ]
  },
  gateway: {
    badge: "API INGRESS &bull; GATEWAY",
    title: "Amazon API Gateway (HTTP API v2.0)",
    type: "Managed Regional HTTP Endpoint",
    desc: "Provides a managed HTTPS entry point receiving 'POST /dev/predict'. Terminates SSL/TLS 1.3 connections, validates HTTP request length, enforces CORS headers ('Access-Control-Allow-Origin: *'), buffers surge traffic with automatic throttling, and performs synchronous Lambda proxy integration invoking the inference engine.",
    chips: [
      "Route: POST /dev/predict",
      "Payload Format: 2.0",
      "CORS: Allowed All Origins (*)",
      "SSL/TLS: Forced TLS 1.3"
    ]
  },
  model: {
    badge: "MACHINE LEARNING &bull; CLASSIFIER",
    title: "Random Forest Classifier (Primary ML)",
    type: "Supervised Ensemble Classifier",
    desc: "Primary decision-maker trained on 284,807 creditcard.csv transactions. Combines 50 decision trees with balanced class weighting to detect non-linear fraud signatures. If this model successfully produces a prediction, its output is ALWAYS used as the primary source of truth without applying amount thresholds.",
    chips: [
      "Estimators: 50 Decision Trees",
      "Class Weight: Balanced",
      "Features: 28 PCA + Scaled Amount",
      "Priority: Level 1 Primary",
      "Output: Fraud/Legit + Risk Score"
    ]
  },
  fallback: {
    badge: "SAFETY SYSTEM &bull; BACKUP ONLY",
    title: "Amount-Based Fallback Rule",
    type: "High-Availability Backup Mechanism",
    desc: "Activates ONLY when the AWS Lambda API or Random Forest prediction cannot be obtained (network error, timeout, HTTP 500, or missing payload fields). Evaluates the transaction Amount: Amount <= $400 classified as Legit (Class 0, Low Risk); Amount > $400 classified as Fraud (Class 1, High Risk). This backup rule never overrides a successful Random Forest prediction.",
    chips: [
      "Priority: Level 2 Backup Only",
      "Threshold: $400.00 USD",
      "Risk Score: N/A (Never invented)",
      "Rule: <= $400 Legit, > $400 Fraud",
      "Trigger: API / Network Failure Only"
    ]
  },
  s3: {
    badge: "AWS STORAGE &bull; MODEL ARTIFACTS",
    title: "Amazon S3 (Simple Storage Service)",
    type: "Secure Object Storage",
    desc: "Private S3 bucket (shyam-fraud-ml-model-2026) hosting serialized machine learning artifacts: 'model/model.joblib' and 'model/scaler.joblib'. Lambda downloads these artifacts on cold start and reuses them in memory for warm invocations. S3 stores only model weights; customer transaction data is NEVER uploaded to S3.",
    chips: [
      "Bucket: shyam-fraud-ml-model-2026",
      "Artifact 1: model/model.joblib",
      "Artifact 2: model/scaler.joblib",
      "Access: Private (IAM Role)",
      "Customer Data Stored: Zero"
    ]
  },
  cloudwatch: {
    badge: "AWS OBSERVABILITY &bull; TELEMETRY",
    title: "Amazon CloudWatch",
    type: "Logging & Telemetry Service",
    desc: "Collects invocation logs, cold/warm start latencies, memory utilization, and error rates from AWS Lambda asynchronously. Operates out-of-band without adding inline request latency to the fraud evaluation pipeline.",
    chips: [
      "Log Stream: /aws/lambda/fraudDetection",
      "Warm Latency: ~45 ms",
      "Provisioned Memory: 512 MB",
      "Error Metric: 0.00% Under Normal Load"
    ]
  },
  iam: {
    badge: "AWS SECURITY &bull; IDENTITY",
    title: "AWS Identity and Access Management",
    type: "Least-Privilege Role Authorization",
    desc: "Provides runtime authorization via 'LambdaExecutionRole' with scoped permission strictly limited to 's3:GetObject' on the specific model bucket. S3 Block Public Access is 100% enforced; the browser has zero direct access to S3 or AWS credentials.",
    chips: [
      "Role: LambdaExecutionRole",
      "Permission: s3:GetObject (Bucket Only)",
      "Public Access: Fully Blocked",
      "Browser Credentials: None"
    ]
  },
  frontend: {
    badge: "CLIENT APPLICATION &bull; UI",
    title: "Single-Page Application (SPA)",
    type: "Static Client Dashboard",
    desc: "Lightweight, responsive dashboard providing interactive 29-feature entry, demo transaction loaders, live API invocation, and clear visual distinction between Primary ML Model predictions and Backup Fallback assessments.",
    chips: [
      "Stack: Vanilla HTML5 / CSS3 / ES6",
      "Payload: 29 Numeric Features",
      "Design: Non-flashy Academic Theme",
      "Accessibility: Prefers-Reduced-Motion"
    ]
  }
};

// ===========================================================================
// 3. ANIMATION CONTROLLER STATE
// ===========================================================================
let currentStepIndex = 0;
let isPlaying = false;
let stepTimer = null;
let lambdaSubstepTimer = null;
let isReducedMotion = false;

// ===========================================================================
// 4. INITIALIZATION
// ===========================================================================
document.addEventListener("DOMContentLoaded", () => {
  // Check prefers-reduced-motion
  isReducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  setupWorkflowControls();
  setupTrainingSimulation();
  setupComponentInspector();
  updateSimulationDrawerMode();

  if (!isReducedMotion) {
    startWorkflow();
  } else {
    // Show static complete state for reduced motion
    renderStaticReducedMotion();
  }
});

// ===========================================================================
// 5. WORKFLOW STEP EXECUTION & LOOP
// ===========================================================================
function startWorkflow() {
  if (isPlaying) return;
  isPlaying = true;
  updatePlayPauseUI();
  executeCurrentStep();
}

function pauseWorkflow() {
  isPlaying = false;
  clearTimeout(stepTimer);
  clearTimeout(lambdaSubstepTimer);
  updatePlayPauseUI();
}

function restartWorkflow() {
  pauseWorkflow();
  currentStepIndex = 0;
  resetAllVisuals();
  startWorkflow();
}

function executeCurrentStep() {
  if (!isPlaying) return;

  const step = WORKFLOW_STEPS[currentStepIndex];
  renderStepVisuals(step);

  stepTimer = setTimeout(() => {
    // Advance to next step
    currentStepIndex++;
    if (currentStepIndex >= WORKFLOW_STEPS.length) {
      // Loop after short delay
      currentStepIndex = 0;
      setTimeout(() => {
        resetAllVisuals();
        executeCurrentStep();
      }, 1500);
    } else {
      executeCurrentStep();
    }
  }, step.durationMs);
}

function renderStepVisuals(step) {
  // 1. Update Step Counter & Banner Text
  const stepCounter = document.getElementById("stepCounter");
  const stepName = document.getElementById("stepName");
  const stepPhaseBadge = document.getElementById("stepPhaseBadge");
  const stepDescription = document.getElementById("stepDescription");
  const stepProgressFill = document.getElementById("stepProgressFill");

  if (stepCounter) stepCounter.textContent = `${step.stepNum} / ${step.stepTotal}`;
  if (stepName) stepName.textContent = step.name;
  if (stepPhaseBadge) stepPhaseBadge.textContent = step.phase;
  if (stepDescription) {
    if (step.nodeId === "node-decision" && SIMULATION_MODE === "fallback") {
      stepDescription.textContent = "Decision Gate: ML API unavailable. Routing data to Level 2 Backup Fallback Rule ($400 threshold).";
    } else {
      stepDescription.textContent = step.description;
    }
  }
  if (stepProgressFill) stepProgressFill.style.width = step.progressPct;

  // 2. Clear previous active nodes & packets
  document.querySelectorAll(".pipe-node").forEach(n => n.classList.remove("node-active"));
  document.querySelectorAll(".packet").forEach(p => p.classList.remove("packet-active"));
  document.querySelectorAll(".side-card").forEach(s => s.classList.remove("side-active"));
  document.querySelectorAll(".r-step").forEach(r => r.classList.remove("r-step-active"));

  // 3. Highlight current node
  const activeNode = document.getElementById(step.nodeId);
  if (activeNode) activeNode.classList.add("node-active");

  // Also highlight decision branches at step 6
  const branchCardYes = document.getElementById("branchCardYes");
  const branchCardNo = document.getElementById("branchCardNo");
  if (step.nodeId === "node-decision") {
    if (branchCardYes) branchCardYes.classList.toggle("active-branch", SIMULATION_MODE === "ml");
    if (branchCardNo) branchCardNo.classList.toggle("active-branch", SIMULATION_MODE === "fallback");
    const nodePred = document.getElementById("node-prediction");
    if (nodePred && SIMULATION_MODE === "ml") nodePred.classList.add("node-active");
  }

  // 4. Animate data packet along connector
  if (step.packetId) {
    const packet = document.getElementById(step.packetId);
    if (packet) packet.classList.add("packet-active");
  }

  // 5. Highlight side branch if applicable
  if (step.sideHighlight) {
    const side = document.getElementById(step.sideHighlight);
    if (side) side.classList.add("side-active");
  }

  // 6. Animate Lambda internal workflow if active
  if (step.lambdaSubstep === "animate-lambda") {
    animateLambdaInternal();
  } else {
    resetLambdaInternal();
  }

  // 7. Update return pipeline steps
  if (step.returnStepId) {
    const rStep = document.getElementById(step.returnStepId);
    if (rStep) rStep.classList.add("r-step-active");
  }

  // 8. Show Simulation Result Drawer at Step 6 - 9
  const simResultCard = document.getElementById("simResultCard");
  if (simResultCard) {
    simResultCard.classList.toggle("hidden", !step.showSimResult);
  }
}

function updateSimulationDrawerMode() {
  const simBadge = document.getElementById("simBadge");
  const simModeNote = document.getElementById("simModeNote");
  const simHttpStatus = document.getElementById("simHttpStatus");
  const simPredVal = document.getElementById("simPredVal");
  const simClassVal = document.getElementById("simClassVal");
  const simScoreVal = document.getElementById("simScoreVal");
  const simSourceVal = document.getElementById("simSourceVal");
  const simJsonCode = document.getElementById("simJsonCode");
  const simModeText = document.getElementById("simModeText");

  if (SIMULATION_MODE === "ml") {
    if (simBadge) {
      simBadge.className = "badge badge-accent font-mono";
      simBadge.textContent = "✓ ML MODEL PREDICTION";
    }
    if (simModeNote) simModeNote.textContent = "• Primary Random Forest Model (AWS Lambda)";
    if (simHttpStatus) simHttpStatus.textContent = "HTTP 200 OK";
    if (simPredVal) {
      simPredVal.textContent = "Fraud";
      simPredVal.className = "sim-badge-val text-danger font-mono";
    }
    if (simClassVal) simClassVal.textContent = "1";
    if (simScoreVal) {
      simScoreVal.textContent = "92.34%";
      simScoreVal.className = "sim-badge-val text-danger font-mono";
    }
    if (simSourceVal) simSourceVal.textContent = "Random Forest";
    if (simJsonCode) {
      simJsonCode.textContent = JSON.stringify({
        prediction: "Fraud",
        class: 1,
        risk_score: 0.923406,
        source: "Random Forest",
        note: "Primary Level 1 ML result"
      }, null, 2);
    }
    if (simModeText) simModeText.textContent = "Mode: Primary ML Flow";
  } else {
    // Fallback Simulation Mode
    if (simBadge) {
      simBadge.className = "badge badge-fallback font-mono";
      simBadge.textContent = "⚠ FALLBACK RISK ASSESSMENT";
    }
    if (simModeNote) simModeNote.textContent = "• Amount-Based Backup Rule (ML Unavailable)";
    if (simHttpStatus) simHttpStatus.textContent = "Backup Applied";
    if (simPredVal) {
      simPredVal.textContent = "Fraud";
      simPredVal.className = "sim-badge-val text-danger font-mono";
    }
    if (simClassVal) simClassVal.textContent = "1";
    if (simScoreVal) {
      simScoreVal.textContent = "N/A";
      simScoreVal.className = "sim-badge-val text-muted font-mono";
    }
    if (simSourceVal) simSourceVal.textContent = "Amount Rule (Backup)";
    if (simJsonCode) {
      simJsonCode.textContent = JSON.stringify({
        prediction: "Fraud",
        class: 1,
        risk_score: "N/A",
        source: "Amount-based backup rule",
        threshold: 400.00,
        condition: "Amount > $400 -> Fraud",
        note: "Backup only. Never replaces valid Random Forest prediction."
      }, null, 2);
    }
    if (simModeText) simModeText.textContent = "Mode: Backup Fallback Demo";
  }
}

function resetAllVisuals() {
  document.querySelectorAll(".pipe-node").forEach(n => n.classList.remove("node-active"));
  document.querySelectorAll(".packet").forEach(p => p.classList.remove("packet-active"));
  document.querySelectorAll(".side-card").forEach(s => s.classList.remove("side-active"));
  document.querySelectorAll(".r-step").forEach(r => r.classList.remove("r-step-active"));
  resetLambdaInternal();
  const simResultCard = document.getElementById("simResultCard");
  if (simResultCard) simResultCard.classList.add("hidden");
}

function animateLambdaInternal() {
  const section = document.getElementById("lambdaInternalSection");
  const status = document.getElementById("lambdaActiveStatus");
  if (section) section.classList.add("lambda-active");
  if (status) status.textContent = "Inference active (Warm start ~45ms)";

  const substeps = document.querySelectorAll(".l-substep");
  substeps.forEach((el, idx) => {
    setTimeout(() => {
      el.classList.add("substep-highlight");
      setTimeout(() => el.classList.remove("substep-highlight"), 450);
    }, idx * 280);
  });
}

function resetLambdaInternal() {
  const section = document.getElementById("lambdaInternalSection");
  const status = document.getElementById("lambdaActiveStatus");
  if (section) section.classList.remove("lambda-active");
  if (status) status.textContent = "Standing by";
  document.querySelectorAll(".l-substep").forEach(el => el.classList.remove("substep-highlight"));
}

function renderStaticReducedMotion() {
  const animStatusPill = document.getElementById("animStatusPill");
  if (animStatusPill) animStatusPill.textContent = "Static Specification Mode";
  document.querySelectorAll(".pipe-node").forEach(n => n.classList.add("node-static-highlight"));
  const simResultCard = document.getElementById("simResultCard");
  if (simResultCard) simResultCard.classList.remove("hidden");
}

// ===========================================================================
// 6. CONTROLS TOOLBAR & PLAYBACK
// ===========================================================================
function setupWorkflowControls() {
  const btnPlay = document.getElementById("btnPlay");
  const btnPause = document.getElementById("btnPause");
  const btnRestart = document.getElementById("btnRestart");
  const btnToggleSimMode = document.getElementById("btnToggleSimMode");

  if (btnPlay) {
    btnPlay.addEventListener("click", () => {
      if (!isPlaying) startWorkflow();
    });
  }

  if (btnPause) {
    btnPause.addEventListener("click", pauseWorkflow);
  }

  if (btnRestart) {
    btnRestart.addEventListener("click", restartWorkflow);
  }

  if (btnToggleSimMode) {
    btnToggleSimMode.addEventListener("click", () => {
      SIMULATION_MODE = SIMULATION_MODE === "ml" ? "fallback" : "ml";
      updateSimulationDrawerMode();
      // If currently at step 6, update visual immediately
      const branchCardYes = document.getElementById("branchCardYes");
      const branchCardNo = document.getElementById("branchCardNo");
      if (branchCardYes) branchCardYes.classList.toggle("active-branch", SIMULATION_MODE === "ml");
      if (branchCardNo) branchCardNo.classList.toggle("active-branch", SIMULATION_MODE === "fallback");
    });
  }
}

function updatePlayPauseUI() {
  const btnPlay = document.getElementById("btnPlay");
  const btnPause = document.getElementById("btnPause");
  const animStatusPill = document.getElementById("animStatusPill");

  if (btnPlay) btnPlay.classList.toggle("btn-active", isPlaying);
  if (btnPause) btnPause.classList.toggle("btn-active", !isPlaying);
  if (animStatusPill) {
    animStatusPill.textContent = isPlaying ? "Live Animation Active" : "Simulation Paused";
    animStatusPill.className = isPlaying ? "badge badge-soft font-mono" : "badge badge-secondary font-mono";
  }
}

// ===========================================================================
// 7. OFFLINE TRAINING SIMULATION
// ===========================================================================
function setupTrainingSimulation() {
  const btnAnimate = document.getElementById("btnAnimateTraining");
  if (!btnAnimate) return;

  btnAnimate.addEventListener("click", () => {
    btnAnimate.disabled = true;
    const steps = document.querySelectorAll(".t-step");

    steps.forEach((step, idx) => {
      setTimeout(() => {
        steps.forEach(s => s.classList.remove("t-step-active"));
        step.classList.add("t-step-active");

        if (idx === steps.length - 1) {
          setTimeout(() => {
            steps.forEach(s => s.classList.remove("t-step-active"));
            btnAnimate.disabled = false;
          }, 1200);
        }
      }, idx * 450);
    });
  });
}

// ===========================================================================
// 8. COMPONENT INSPECTOR
// ===========================================================================
function setupComponentInspector() {
  const buttons = document.querySelectorAll(".comp-btn");

  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      buttons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      const compKey = btn.getAttribute("data-component");
      const details = COMPONENT_DETAILS[compKey];
      if (!details) return;

      const compBadge = document.getElementById("compBadge");
      const compTitle = document.getElementById("compTitle");
      const compType = document.getElementById("compType");
      const compBody = document.getElementById("compBody");
      const compChips = document.getElementById("compChips");

      if (compBadge) compBadge.innerHTML = details.badge;
      if (compTitle) compTitle.textContent = details.title;
      if (compType) compType.textContent = details.type;
      if (compBody) compBody.textContent = details.desc;

      if (compChips) {
        compChips.innerHTML = details.chips.map(chip => `<span class="chip">${chip}</span>`).join("");
      }
    });
  });

  // Trigger click on default component (lambda)
  const defaultBtn = document.querySelector('.comp-btn[data-component="lambda"]');
  if (defaultBtn) defaultBtn.click();
}
