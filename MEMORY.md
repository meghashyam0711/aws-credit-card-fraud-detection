# MEMORY.md — Persistent Technical Memory

> **Project Title:** “Innovate Faster with Machine Learning (ML) on AWS”  
> **Project Name:** AWS-based Credit Card Fraud Detection System  
> **Repository:** `https://github.com/meghashyam0711/ai-credit-card-fraud-detection`  
> **Role of this Document:** System source-of-truth and persistent technical memory for future development and maintenance agents.

---

## 1. Project Objective

The primary objective of this project is to provide real-time, highly reliable, and low-latency credit card fraud inference by deploying a pre-trained scikit-learn Machine Learning (Random Forest) model on an AWS serverless architecture.

Key operational characteristics:
- **Zero permanent server overhead**: Fully serverless compute using AWS Lambda and Amazon API Gateway HTTP API.
- **Fast cold-start recovery and sub-20ms warm-start inference**: Model artifacts stored in Amazon S3 and cached in Lambda `/tmp` disk and memory.
- **Strict schema validation**: Incoming payloads must provide all required numerical features dynamically identified by the model.
- **Decoupled interactive frontend**: Pure client-side browser application communicating over HTTPS with automated fallback handling.

---

## 2. Current Architecture

```text
┌────────────────────────────────────────────────────────┐
│               Frontend Client Layer                    │
│   (HTML5, Vanilla CSS, JS Dashboard & Architecture)   │
└───────────────────────────┬────────────────────────────┘
                            │ HTTPS (POST /predict)
                            ▼
┌────────────────────────────────────────────────────────┐
│             Amazon API Gateway (HTTP API)             │
│   - API Name: FraudDetectionAPI                        │
│   - Stage: dev                                         │
│   - Route: POST /predict                               │
│   - Payload Format: 2.0                                │
│   - Direct OPTIONS CORS Preflight Handling             │
└───────────────────────────┬────────────────────────────┘
                            │ Lambda Proxy Event (v2.0)
                            ▼
┌────────────────────────────────────────────────────────┐
│                  AWS Lambda Function                   │
│   - Name: FraudDetectionLambda (Python 3.12, x86_64)   │
│   - Memory: 512 MB | Timeout: 60s                      │
│   - Handler: lambda_function.lambda_handler            │
│   - In-Memory Cache: _MODEL, _SCALER, _REQUIRED_FEATS  │
└─────────────┬────────────────────────────┬─────────────┘
              │ Cold-start                 │ Telemetry
              │ S3 GetObject               │ & Logs
              ▼                            ▼
┌───────────────────────────┐  ┌─────────────────────────┐
│      Amazon S3 Bucket     │  │    Amazon CloudWatch    │
│ shyam-fraud-ml-model-2026 │  │  - Execution Metrics    │
│  - model/model.joblib     │  │  - Cold/Warm Status     │
│  - model/scaler.joblib    │  │  - Sanitized Logs       │
└───────────────────────────┘  └─────────────────────────┘
              │ Artifacts loaded
              ▼
┌────────────────────────────────────────────────────────┐
│                Inference Engine                        │
│   - Dynamic feature alignment (model.feature_names_in_)│
│   - Amount scaled via StandardScaler.transform()       │
│   - RandomForestClassifier.predict() & predict_proba() │
│   - Returns: prediction ("Fraud"|"Legit"), class (1|0),│
│     risk_score (0.0000 - 1.0000)                       │
└────────────────────────────────────────────────────────┘
```

### Supporting Services:
- **AWS IAM**: `FraudModelLambdaRole` execution role granting strictly scoped `s3:GetObject` access to model artifacts and `logs:*` for CloudWatch.
- **Amazon CloudWatch**: Gathers Lambda system telemetry, execution durations, memory utilization, cold/warm start status, and inference outcomes without logging sensitive transaction data.

---

## 3. AWS Resource Specifications

All infrastructure resides in `us-east-1`:

| Component | Resource Identifier / Value | Details & Configuration | Status |
| :--- | :--- | :--- | :--- |
| **AWS Region** | `us-east-1` | Primary deployment region | IMPLEMENTED |
| **S3 Bucket** | `shyam-fraud-ml-model-2026` | Private bucket storing trained model and scaler | IMPLEMENTED |
| **S3 Model Object** | `model/model.joblib` | Serialized scikit-learn `RandomForestClassifier` | IMPLEMENTED |
| **S3 Scaler Object**| `model/scaler.joblib` | Serialized scikit-learn `StandardScaler` | IMPLEMENTED |
| **IAM Execution Role**| `FraudModelLambdaRole` | Attached to Lambda with least-privilege policy | IMPLEMENTED |
| **Lambda Function** | `FraudDetectionLambda` | Serverless Python inference backend | IMPLEMENTED |
| **Lambda Runtime** | `Python 3.12` | Architecture: `x86_64` | IMPLEMENTED |
| **Lambda Compute** | `512 MB` Memory, `60s` Timeout | Optimized for NumPy/scikit-learn cold-start | IMPLEMENTED |
| **Lambda Handler** | `lambda_function.lambda_handler` | Entry point handling routing & execution | IMPLEMENTED |
| **API Gateway** | `FraudDetectionAPI` | Amazon API Gateway HTTP API | IMPLEMENTED |
| **API Stage** | `dev` | Active deployment stage | IMPLEMENTED |
| **API Route** | `POST /predict` | Triggers `FraudDetectionLambda` integration | IMPLEMENTED |
| **Live Endpoint** | `https://sd2mbb56a7.execute-api.us-east-1.amazonaws.com/dev/predict` | Publicly reachable HTTPS endpoint | IMPLEMENTED |

---

## 4. Machine Learning & Preprocessing Details

### 4.1 ML Algorithm & Hyperparameters
- **Model Type:** `RandomForestClassifier` (scikit-learn)
- **Number of Estimators (`n_estimators`):** `50`
- **Maximum Depth (`max_depth`):** `10`
- **Class Balancing (`class_weight`):** `balanced` (compensates for severe fraud class imbalance)
- **Random State (`random_state`):** `42`
- **Target Label:** `Class` (`0` = Legit, `1` = Fraud)

### 4.2 Training & Feature Specifications
- **Dataset Source:** Kaggle Credit Card Fraud Detection dataset (`creditcard.csv`).
- **Feature Exclusion:** Target column `Class` and identifier column `id` (or irrelevant metadata) were dropped during training:
  ```python
  X = df.drop(columns=["Class", "id"])
  ```
- **Active Model Input Features (29 continuous numerical attributes):**
  - PCA transformed features: `V1`, `V2`, `V3`, `V4`, `V5`, `V6`, `V7`, `V8`, `V9`, `V10`, `V11`, `V12`, `V13`, `V14`, `V15`, `V16`, `V17`, `V18`, `V19`, `V20`, `V21`, `V22`, `V23`, `V24`, `V25`, `V26`, `V27`, `V28`
  - Transaction amount: `Amount`
  *(Note: The model was trained without `Time`; only `V1`-`V28` and `Amount` are used in production inference).*
- **Standardization Preprocessing:**
  - `StandardScaler()` was fitted **strictly on the `Amount` feature**.
  - All PCA features `V1` through `V28` were already standardized during PCA extraction and are passed **unmodified**.
- **Dynamic Inspection Rule:**
  - The model was trained and fitted on a pandas `DataFrame`.
  - The backend dynamically inspects `model.feature_names_in_` at runtime.
  - Features are mapped into a single-row `pd.DataFrame` matching `model.feature_names_in_` exactly, guaranteeing compatibility with scikit-learn without hardcoded positional ordering.

---

## 5. Backend Behavior & Implementation Logic

Implemented in `backend/lambda_function.py`:

1. **Protocol Support:**
   - Handles API Gateway HTTP API Payload Format 2.0 (`event["requestContext"]["http"]["method"]` and `event["body"]`).
   - Supports backward-compatible REST API / Payload Format 1.0.
   - Accepts direct Lambda console test events (dictionary directly passed).
   - Handles Base64-encoded request payloads (`event["isBase64Encoded"] == True`).
2. **CORS Handling:**
   - Detects `OPTIONS` HTTP requests and immediately responds with HTTP 200/204 and headers:
     - `Access-Control-Allow-Origin: *`
     - `Access-Control-Allow-Methods: POST,OPTIONS`
     - `Access-Control-Allow-Headers: Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token`
3. **Artifact Caching (Cold vs. Warm Invocations):**
   - **Cold Start:** Checks `/tmp` disk cache. If missing, downloads `model/model.joblib` and `model/scaler.joblib` from `shyam-fraud-ml-model-2026` via `boto3`. Loads artifacts into global variables `_MODEL`, `_SCALER`, and extracts `_REQUIRED_FEATURES = list(model.feature_names_in_)`.
   - **Warm Start:** Reuses in-memory `_MODEL`, `_SCALER`, and `_REQUIRED_FEATURES` directly. Invocations execute in sub-20ms.
4. **Validation Logic:**
   - Ensures payload is valid JSON.
   - Validates that every feature in `model.feature_names_in_` exists in the incoming transaction dictionary.
   - If features are missing, raises `InputValidationError` and returns HTTP 400:
     ```json
     {
       "error": "Missing features",
       "missing_features": ["V1", "Amount"]
     }
     ```
   - Validates that all feature values are numeric (`float` convertible). Returns HTTP 400 for non-numeric inputs.
5. **Preprocessing & Inference:**
   - Standardizes `Amount` using `scaler.transform([[raw_amount]])[0][0]`.
   - Assembles an ordered `pd.DataFrame` matching `model.feature_names_in_`.
   - Generates binary prediction `model.predict(input_df)` (0 or 1).
   - Computes continuous risk probability using `model.predict_proba(input_df)[0]`.
   - Maps class `1` probability to `risk_score` (rounded to 4 decimal places).
6. **Prediction Response Format (HTTP 200):**
   ```json
   {
     "prediction": "Fraud",
     "class": 1,
     "risk_score": 0.9234
   }
   ```
   *(Or `"prediction": "Legit"`, `"class": 0`, `"risk_score": 0.0595"`)*

---

## 6. Frontend Integration & Architecture

Implemented in `frontend/`:
- **Dashboard (`frontend/index.html`, `style.css`, `script.js`):**
  - Interactive web interface with a 4-column responsive grid for features `V1` to `V28` and an `Amount` input.
  - **Demo Profiles:**
    - `Load Demo (Legit)`: Populates typical non-fraud transaction values (`Amount = $34.50`) yielding Legit (`class: 0`, `risk_score ~5.95%`).
    - `Load Demo (Fraud)`: Populates anomalous transaction values (`Amount = $529.00`) yielding Fraud (`class: 1`, `risk_score ~97.95%`).
    - `Clear`: Resets all 29 fields.
  - Real-time gauge and visual indicator reflecting classification and risk score.
  - **Two-Tier Decision Logic:**
    - **Tier 1 (Primary):** Real AWS Lambda + Random Forest API prediction.
    - **Tier 2 (Fallback Only):** If the network/API is unreachable, triggers a local fallback rule (`Amount > $400` -> Fraud, `<= $400` -> Legit) and explicitly flags it as a fallback with `Risk Score: N/A`.
- **System Architecture Visualizer (`frontend/architecture.html`, `architecture.css`, `architecture.js`):**
  - Interactive walkthrough showing end-to-end cloud pipeline, request flow, and AWS service configurations.

---

## 7. Current Repository Structure

```text
ai-credit-card-fraud-detection/
├── backend/
│   ├── README.md               # Backend-specific architecture and setup guide
│   ├── lambda_function.py      # Core Lambda handler (Payload 2.0, caching, CORS, ML inference)
│   ├── requirements.txt        # Production dependencies (boto3, joblib, numpy, pandas, scikit-learn)
│   ├── test_api.py             # Integration test script against live AWS API Gateway
│   └── tests/
│       ├── __init__.py
│       └── test_validation.py  # Unit test suite (9 test cases covering preprocessing & CORS)
├── frontend/
│   ├── architecture.css        # Architecture viewer styles
│   ├── architecture.html       # Visual architecture explorer page
│   ├── architecture.js         # Interactive diagram controls
│   ├── index.html              # Main interactive fraud detection dashboard
│   ├── script.js               # Frontend controller, API client, demo loader, and fallback
│   └── style.css               # Main dashboard UI stylesheet
├── lambda_function.py          # Root-level mirror of the Lambda handler
├── MEMORY.md                   # This persistent technical memory document
├── PRD.md                      # Product Requirements Document
├── README.md                   # Primary project documentation and repository overview
├── RULES.md                    # Immutable developer rules and operational constraints
└── TASKS.md                    # Project task status and testing checklist
```

---

## 8. Verification & Testing Status

### Unit Validation Suite (`backend/tests/test_validation.py`)
Status: **9 passed, 0 failed**
Covered scenarios:
1. `test_dynamic_feature_inspection`: Confirms dynamic feature extraction via `model.feature_names_in_`.
2. `test_missing_features_returns_400`: Asserts HTTP 400 when any required feature is missing.
3. `test_amount_scaling_only`: Asserts only `Amount` is transformed while `V1`..`V28` remain raw, returning `pd.DataFrame`.
4. `test_cors_preflight_options`: Verifies HTTP 200 with complete CORS headers for `OPTIONS` requests.
5. `test_api_gateway_v2_post_success`: Asserts end-to-end inference flow using Payload Format 2.0.
6. `test_base64_encoded_body`: Verifies decoding and parsing of base64-encoded request bodies.
7. `test_direct_lambda_test_event`: Asserts direct dictionary payloads (Lambda console) are processed correctly.
8. `test_invalid_numeric_amount_returns_400`: Asserts HTTP 400 when `Amount` or features cannot be cast to float.
9. `test_malformed_json_returns_400`: Asserts HTTP 400 when the JSON body syntax is invalid.

### Live API Gateway Integration Testing (`backend/test_api.py`)
Endpoint: `https://sd2mbb56a7.execute-api.us-east-1.amazonaws.com/dev/predict`
Status: **Verified and Operational**
1. **Valid Transaction:** Returns HTTP 200 with keys `prediction`, `class`, `risk_score`.
2. **Missing Features (del V1, Amount):** Returns HTTP 400 with `missing_features: ["V1", "Amount"]`.
3. **CORS Preflight:** `OPTIONS` request returns HTTP 200/204 with required CORS headers.

---

## 9. Immutable Constraints & Architecture Decisions

Future developers and agents MUST abide by the following technical decisions:

1. **Model Immutability:**
   - The trained `model.joblib` and `scaler.joblib` stored in `s3://shyam-fraud-ml-model-2026/model/` MUST NOT be replaced, overwritten, or retrained unless explicitly authorized.
   - The model algorithm is fixed to `RandomForestClassifier` with 50 estimators and max depth 10.
2. **Dynamic Feature Ordering:**
   - Never hardcode positional feature order. Always extract feature order from `model.feature_names_in_`.
3. **Selective Preprocessing:**
   - `Amount` is the **only** feature passed through `StandardScaler`.
   - Never apply scalers or normalization to `V1` through `V28`.
4. **AWS Infrastructure Permanence:**
   - Do NOT create duplicate S3 buckets, Lambda functions, or API Gateways.
   - Do NOT alter the existing bucket name (`shyam-fraud-ml-model-2026`) or endpoint path (`/dev/predict`).
   - S3 bucket must remain strictly **private**; public access is forbidden.
5. **Stateless Serverless Execution:**
   - The Lambda function must remain a single, lightweight entry point (`lambda_function.lambda_handler`).
   - Do NOT introduce web frameworks (Flask, FastAPI, Django) into the Lambda runtime.

---

## 10. Current vs. Future Implementation Status

| Feature / Capability | Category | Implementation Status |
| :--- | :--- | :--- |
| AWS S3 model artifact storage | Infrastructure | **CURRENT / IMPLEMENTED** |
| Lambda function with Python 3.12 & x86_64 | Infrastructure | **CURRENT / IMPLEMENTED** |
| API Gateway HTTP API v2.0 (`POST /predict`) | Infrastructure | **CURRENT / IMPLEMENTED** |
| IAM Role with least-privilege `s3:GetObject` | Security | **CURRENT / IMPLEMENTED** |
| In-memory & `/tmp` artifact warm-start caching | Performance | **CURRENT / IMPLEMENTED** |
| Dynamic feature alignment via `feature_names_in_` | ML Backend | **CURRENT / IMPLEMENTED** |
| `StandardScaler` transformation on `Amount` only | Preprocessing | **CURRENT / IMPLEMENTED** |
| Random Forest inference with continuous `risk_score` | ML Backend | **CURRENT / IMPLEMENTED** |
| Missing feature & non-numeric payload validation | Validation | **CURRENT / IMPLEMENTED** |
| CORS headers & OPTIONS preflight response | Web Protocol | **CURRENT / IMPLEMENTED** |
| CloudWatch logging of inference telemetry | Monitoring | **CURRENT / IMPLEMENTED** |
| Web frontend dashboard with live demo profiles | Frontend | **CURRENT / IMPLEMENTED** |
| Visual architecture explorer | Frontend | **CURRENT / IMPLEMENTED** |
| API integration test script (`test_api.py`) | Testing | **CURRENT / IMPLEMENTED** |
| Preprocessing unit test suite (9 tests) | Testing | **CURRENT / IMPLEMENTED** |
| CloudWatch Custom Alarm for inference errors | Monitoring | **FUTURE / NOT IMPLEMENTED** |
| AWS WAF rate limiting on API Gateway | Security | **FUTURE / NOT IMPLEMENTED** |
| Custom Route53 domain for API Gateway | Networking | **FUTURE / NOT IMPLEMENTED** |
| Model drift detection & automated shadow retraining | ML Ops | **FUTURE / NOT IMPLEMENTED** |
| Batch transaction inference endpoint (`POST /batch`) | Feature | **FUTURE / NOT IMPLEMENTED** |
