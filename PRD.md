# Product Requirements Document (PRD)

## Project Title: “Innovate Faster with Machine Learning (ML) on AWS”
### System: AWS-based Credit Card Fraud Detection System
**Document Version:** 1.0.0  
**Target AWS Region:** `us-east-1`  
**Repository:** `https://github.com/meghashyam0711/ai-credit-card-fraud-detection`

---

## 1. Product Overview

The **AWS-based Credit Card Fraud Detection System** is a real-time, cloud-native machine learning solution designed to identify fraudulent financial transactions instantaneously. Utilizing a serverless architecture consisting of Amazon API Gateway, AWS Lambda, Amazon S3, and Amazon CloudWatch, the system executes low-latency inference using a trained `RandomForestClassifier`. The platform delivers an end-to-end user experience through an interactive browser dashboard capable of simulating legitimate and fraudulent transactions, observing risk scores, and exploring system telemetry.

---

## 2. Problem Statement

Financial institutions and payment processors face severe financial losses and erosion of user trust due to unauthorized credit card charges. Fraudulent patterns evolve rapidly and represent a tiny fraction (often < 0.2%) of overall transaction volume. Traditional rule-based engines fail to detect subtle multivariate anomalies and introduce excessive friction through false positives. Organizations require a machine learning system capable of handling severe class imbalance, processing transactions with minimal latency, scaling seamlessly to absorb transaction spikes, and operating with zero infrastructure maintenance overhead.

---

## 3. Project Objective

The key objectives of the project are:
- Deliver real-time credit card fraud classification with sub-second end-to-end response times and sub-20ms warm Lambda inference execution.
- Leverage AWS Serverless architecture to eliminate server management, idle compute costs, and provisioning delays.
- Host and maintain serialized machine learning artifacts (`model.joblib` and `scaler.joblib`) securely within a private Amazon S3 bucket.
- Provide a robust API Gateway HTTP API supporting Cross-Origin Resource Sharing (CORS) and standard JSON payloads.
- Guarantee strict input validation to prevent model failure caused by missing features, non-numeric inputs, or schema drifts.
- Offer an intuitive frontend dashboard providing instant evaluation feedback and transparency into the decision-making process.

---

## 4. Target Users

- **Fraud Analysts & Risk Officers:** Individuals who monitor transaction streams, evaluate high-risk events, and verify fraud patterns.
- **FinTech Application Developers:** Engineers integrating automated fraud detection hooks into checkout funnels and payment gateways.
- **System Administrators & Auditors:** Personnel tracking cloud operational telemetry, latency distributions, and security compliance.
- **Academic Evaluators & Cloud Assessors:** Individuals reviewing high-performance serverless cloud ML patterns and distributed system designs.

---

## 5. Functional Requirements

- **FR-1 (Transaction Ingestion):** Ingest JSON payloads containing 29 continuous numerical transaction features (`V1` to `V28` and `Amount`) via HTTPS `POST /predict`.
- **FR-2 (CORS Preflight Support):** Accurately process HTTP `OPTIONS` preflight requests, returning necessary headers allowing cross-origin web browser invocation.
- **FR-3 (Schema & Numerical Validation):** Inspect incoming request keys against model expectations (`model.feature_names_in_`). Return HTTP 400 with a detailed error array if features are omitted or non-numeric.
- **FR-4 (Selective Feature Scaling):** Standardize strictly the `Amount` feature using the pre-fitted `StandardScaler` while preserving PCA features `V1`..`V28` without alteration.
- **FR-5 (Inference Execution):** Compute binary classification (`0` = Legit, `1` = Fraud) using `RandomForestClassifier.predict()` and continuous fraud probability using `predict_proba()`.
- **FR-6 (Artifact Caching):** Automatically download model artifacts from Amazon S3 to `/tmp` during cold starts and maintain them in global memory across warm executions.
- **FR-7 (Response Delivery):** Format and return standardized JSON containing classification label, numerical class, and a 4-decimal risk score.
- **FR-8 (Interactive Frontend):** Render an interactive web interface offering automated demo data loading (`Legit` and `Fraud`), single-click evaluation, real-time risk gauges, and architecture visualization.
- **FR-9 (Client-Side Fallback):** In the event of network disruption or backend unreachability, the client interface provides an explicit fallback indicator without falsifying ML risk scores.

---

## 6. Non-Functional Requirements

- **NFR-1 (Latency):** End-to-end API response time under 150ms for warm invocations; Lambda execution compute under 20ms.
- **NFR-2 (Availability & Scalability):** Serverless architecture scaling automatically on demand via AWS Lambda and Amazon API Gateway HTTP API without manual intervention.
- **NFR-3 (Security & Isolation):** Private S3 bucket hosting model artifacts, accessible only via IAM role execution credentials (`FraudModelLambdaRole`). Zero credentials in code or repository.
- **NFR-4 (Fault Tolerance):** Graceful degradation with explicit HTTP status codes (400 for bad input, 500 for unrecoverable backend failures) preventing unhandled exceptions or lambda timeouts.
- **NFR-5 (Maintainability):** Dynamic extraction of required feature names directly from `model.feature_names_in_`, eliminating brittle hard-coded feature indices.

---

## 7. System Architecture

```text
┌────────────────────────────────────────────────────────┐
│                   Frontend Client                      │
│   (index.html, script.js, style.css, architecture.html)│
└───────────────────────────┬────────────────────────────┘
                            │ HTTPS (POST /predict)
                            ▼
┌────────────────────────────────────────────────────────┐
│             Amazon API Gateway (HTTP API)             │
│   Endpoint: .../dev/predict | Stage: dev               │
└───────────────────────────┬────────────────────────────┘
                            │ Payload Format 2.0
                            ▼
┌────────────────────────────────────────────────────────┐
│                  AWS Lambda Function                   │
│   Name: FraudDetectionLambda (Python 3.12, 512MB)      │
│   Handler: lambda_function.lambda_handler              │
└─────────────┬────────────────────────────┬─────────────┘
              │ Cold-start                 │ Telemetry
              │ S3 GetObject               │ & Logs
              ▼                            ▼
┌───────────────────────────┐  ┌─────────────────────────┐
│      Amazon S3 Bucket     │  │    Amazon CloudWatch    │
│ shyam-fraud-ml-model-2026 │  │ Logs: Execution duration│
│  - model/model.joblib     │  │ & inference outcomes    │
│  - model/scaler.joblib    │  └─────────────────────────┘
└───────────────────────────┘
              │ Artifacts loaded
              ▼
┌────────────────────────────────────────────────────────┐
│              Random Forest Model Engine                │
│   - Feature Alignment & Amount Scaling                 │
│   - Binary Prediction + Continuous Risk Score          │
└────────────────────────────────────────────────────────┘
```

---

## 8. Backend Requirements

- **Runtime Environment:** Python 3.12 running on `x86_64` AWS Lambda architecture.
- **Dependencies:** `joblib`, `numpy`, `scipy`, `pandas`, `scikit-learn`, and `boto3`.
- **Payload Handling:** Native handling of API Gateway HTTP API Payload Format 2.0, REST API v1.0, Base64-encoded strings, and direct dictionary events.
- **Artifact Management:** Lazy-loading mechanism that checks `/tmp` and pulls from S3 bucket `shyam-fraud-ml-model-2026` only when not cached in memory.
- **CORS Support:** Universal origins (`*`), supporting `POST` and `OPTIONS` methods with standard API headers.
- **Inference Pipeline:** Features arranged in a single-row `pandas.DataFrame` preserving columns matching `model.feature_names_in_`.

---

## 9. Frontend Integration Requirements

- **Decoupled Architecture:** Pure client-side static web application requiring no Node.js backend server.
- **Form Controls:** 28 PCA numerical input fields (`V1` to `V28`) and one `Amount` input field.
- **Pre-configured Profiles:**
  - `Load Demo (Legit)`: Populates verified legitimate transaction parameters (`Amount = 34.50`).
  - `Load Demo (Fraud)`: Populates verified fraudulent anomaly parameters (`Amount = 529.00`).
  - `Clear`: Resets all form fields and returns UI to ready state.
- **Status Indicators:** Real-time API connection badge, live latency counter, visual risk bar gauge, and classification banner (`Legit` / `Fraud`).
- **Telemetry Display:** Specifications panel detailing classification, class index, risk percentage, source (`AWS Random Forest` vs `Fallback`), and request status.

---

## 10. ML Model Requirements

- **Model Architecture:** `RandomForestClassifier` from `scikit-learn`.
- **Model Parameters:**
  - `n_estimators = 50`
  - `max_depth = 10`
  - `class_weight = "balanced"`
  - `random_state = 42`
- **Output Classes:**
  - `Class 0`: Legitimate transaction.
  - `Class 1`: Fraudulent transaction.
- **Metric Formulation:**
  - Binary classification: `prediction = "Fraud" if class == 1 else "Legit"`.
  - Risk probability: Calculated from `predict_proba()[:, 1]` and rounded to four decimal places.
- **Preprocessing Pipeline:**
  - Input: 29 features (`V1`..`V28`, `Amount`).
  - Preprocessing: `StandardScaler` applied exclusively to `Amount`.
  - Format: Single-row `pandas.DataFrame` structured according to `model.feature_names_in_`.

---

## 11. AWS Infrastructure Requirements

- **Region:** `us-east-1` (US East, N. Virginia).
- **S3 Bucket:** `shyam-fraud-ml-model-2026`
  - Block Public Access: Enabled (Bucket strictly private).
  - Objects: `model/model.joblib` and `model/scaler.joblib`.
- **Lambda Function:** `FraudDetectionLambda`
  - Memory: 512 MB.
  - Timeout: 60 seconds.
  - Handler: `lambda_function.lambda_handler`.
- **API Gateway:** `FraudDetectionAPI`
  - Type: HTTP API.
  - Stage: `dev`.
  - Route: `POST /predict`.
- **IAM Role:** `FraudModelLambdaRole`
  - Permissions: Scoped `s3:GetObject` on the model bucket keys, plus CloudWatch log stream publishing.

---

## 12. API Specification

- **Protocol:** HTTPS
- **Base Endpoint:** `https://sd2mbb56a7.execute-api.us-east-1.amazonaws.com/dev/predict`
- **Supported Methods:** `POST`, `OPTIONS`

---

## 13. Request Format

### Headers:
```http
Content-Type: application/json
Accept: application/json
```

### Body (JSON):
The request requires 29 numerical features (`V1` through `V28` and `Amount`).

```json
{
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
}
```

---

## 14. Response Format

### Successful Prediction (HTTP 200 OK):
```json
{
  "prediction": "Fraud",
  "class": 1,
  "risk_score": 0.9234
}
```
*(Or `"prediction": "Legit"`, `"class": 0`, `"risk_score": 0.0595"`)*

### Preflight CORS Response (HTTP 200 / 204):
```json
{
  "message": "CORS preflight OK"
}
```

---

## 15. Error Handling

### 15.1 Missing Features (HTTP 400 Bad Request):
Returned when any required feature from `model.feature_names_in_` is absent:
```json
{
  "error": "Missing features",
  "missing_features": [
    "V1",
    "Amount"
  ]
}
```

### 15.2 Non-Numeric Feature Value (HTTP 400 Bad Request):
Returned when a feature cannot be converted to a floating-point number:
```json
{
  "error": "Invalid numeric value",
  "message": "Feature 'Amount' must be numeric, got: abc"
}
```

### 15.3 Malformed JSON Payload (HTTP 400 Bad Request):
Returned when payload cannot be parsed as valid JSON:
```json
{
  "error": "Bad Request",
  "message": "Invalid JSON format in request body: ..."
}
```

### 15.4 Internal Execution / Artifact Failure (HTTP 500 Internal Server Error):
Returned when S3 access fails or model unpickling encounters an issue:
```json
{
  "error": "Internal Server Error",
  "message": "Failed to load model artifacts from S3."
}
```

---

## 16. Security Requirements

- **S3 Bucket Security:** `shyam-fraud-ml-model-2026` is private with all public access blocked. No public URLs exist for model artifacts.
- **IAM Principle of Least Privilege:** `FraudModelLambdaRole` permissions are restricted strictly to `s3:GetObject` on the specific model keys and `logs:*` on CloudWatch.
- **Zero Embedded Credentials:** No AWS access keys, secret keys, or passwords exist within the codebase or repository.
- **PII Protection:** The system accepts anonymized PCA features (`V1`..`V28`) and transaction amount. No cardholder names, primary account numbers (PAN), or CVVs are processed or logged.

---

## 17. Logging and Monitoring

- **Service:** Amazon CloudWatch Logs.
- **Log Group:** Automatically generated under `/aws/lambda/FraudDetectionLambda`.
- **Captured Telemetry:**
  - Invocation start and HTTP method.
  - S3 download and `/tmp` caching events.
  - Cold vs. warm start execution branch.
  - Prediction outcome, predicted class, and computed risk score.
  - Sanitized error traces for debugging.
- **Sanitization:** Raw transaction payload dictionaries are intentionally omitted from persistent logs to adhere to data minimization principles.

---

## 18. Testing Requirements

- **Unit Test Coverage:** Automated test suite `backend/tests/test_validation.py` executing 9 distinct test assertions:
  - Dynamic discovery of feature names via `model.feature_names_in_`.
  - Rejection of missing features with HTTP 400.
  - Verification that only `Amount` is modified by `StandardScaler`.
  - CORS header validation on `OPTIONS` calls.
  - HTTP API Payload Format 2.0 end-to-end simulation.
  - Base64-encoded body decoding.
  - Direct dictionary event execution.
  - Rejection of non-numeric values with HTTP 400.
  - Malformed JSON handling with HTTP 400.
- **Live Integration Testing:** Automated script `backend/test_api.py` validating live connectivity to API Gateway endpoint for positive, negative, and CORS requests.

---

## 19. Deployment Requirements

- **Deployment Mechanism:** AWS Lambda direct package deployment or Lambda Layer containing scikit-learn, NumPy, SciPy, pandas, and joblib.
- **Artifact Placement:** `model.joblib` and `scaler.joblib` must be uploaded to `s3://shyam-fraud-ml-model-2026/model/`.
- **Zero Resource Recreation:** Deployment utilizes existing AWS resources without creating redundant buckets or endpoints.

---

## 20. Project Constraints

- **Strict Model Fixation:** The existing trained model `model.joblib` and scaler `scaler.joblib` cannot be modified or replaced.
- **No Algorithm Alteration:** The machine learning algorithm is strictly fixed to the trained `RandomForestClassifier`.
- **Infrastructure Exclusivity:** All requests must route through the single existing API Gateway `FraudDetectionAPI` and Lambda function `FraudDetectionLambda`.
- **Framework Constraint:** No heavyweight web frameworks (Flask, FastAPI, Django) within the Lambda runtime; execution relies on the native `lambda_handler` interface.

---

## 21. Future Enhancements

The following items are not implemented in the current production baseline and are reserved for future phases:

- `[Future Enhancement]` **Custom Domain & TLS:** Configure Amazon Route 53 and AWS Certificate Manager (ACM) to attach a custom branded domain (e.g., `api.fraud-detection.domain.com`).
- `[Future Enhancement]` **API Rate Limiting & Web Application Firewall:** Integrate AWS WAF with Amazon API Gateway to mitigate DDoS attacks and enforce IP rate limiting.
- `[Future Enhancement]` **API Key Authentication:** Implement AWS API Gateway API keys or Amazon Cognito JWT authorizers to restrict prediction calls to authorized partners.
- `[Future Enhancement]` **Automated Model Retraining Pipeline:** Implement an AWS Step Functions workflow using Amazon SageMaker to periodically retrain models on new transaction batches.
- `[Future Enhancement]` **Model Drift & Telemetry Alarms:** Configure Amazon CloudWatch Alarms and Amazon SNS notifications to alert engineers when fraud risk distributions shift significantly.
- `[Future Enhancement]` **Bulk / Batch Inference Endpoint:** Develop an asynchronous `POST /predict/batch` route to process multi-thousand transaction CSV files via AWS Step Functions or Amazon SQS.
