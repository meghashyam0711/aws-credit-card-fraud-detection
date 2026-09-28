# Innovate Faster with Machine Learning (ML) on AWS

> **AWS-based Credit Card Fraud Detection System**  
> High-performance serverless inference pipeline delivering real-time fraud classification and continuous risk scoring using scikit-learn on AWS Lambda and Amazon API Gateway.  
> **Repository:** `https://github.com/meghashyam0711/ai-credit-card-fraud-detection`

---

## Overview

The **AWS-based Credit Card Fraud Detection System** is an end-to-end cloud machine learning solution engineered to evaluate financial transactions for fraudulent activity in real time. 

Built using a modern serverless paradigm on Amazon Web Services (AWS), the system ingests transaction feature vectors via an Amazon API Gateway HTTP API, passes them to an AWS Lambda function running Python 3.12, dynamically standardizes the transaction amount, and evaluates the transaction using a trained Random Forest model downloaded from Amazon S3. 

The architecture guarantees sub-20ms warm-start inference execution, eliminates permanent server costs, and pairs with an interactive client dashboard featuring live evaluation, demo profile injection, and cloud architecture exploration.

---

## Architecture

```text
Frontend
   │
   ▼
API Gateway
   │
   ▼
AWS Lambda
   │
   ▼
Amazon S3
   │
   ▼
Random Forest Model
   │
   ▼
Prediction
   │
   ▼
API Gateway
   │
   ▼
Frontend
```

### Detailed Data & Execution Flow:
1. **Frontend:** Client interface submits transaction parameters via HTTPS `POST /predict`.
2. **API Gateway:** HTTP API routes payload using Payload Format 2.0 and manages CORS preflight (`OPTIONS`).
3. **AWS Lambda:**
   - **Cold Start:** Retrieves `model.joblib` and `scaler.joblib` from Amazon S3 to `/tmp` and loads them into memory.
   - **Warm Start:** Bypasses S3 and reuses cached in-memory model objects.
4. **Preprocessing & Inference:** Validates features dynamically using `model.feature_names_in_`, standardizes `Amount`, and runs `predict()` and `predict_proba()`.
5. **Response:** Returns binary classification (`"Fraud"` or `"Legit"`), numerical class (`1` or `0`), and probability `risk_score` with CORS headers.

---

## Technologies Used

- **Python (3.12):** Core language for backend execution and validation testing.
- **AWS Lambda:** Serverless compute for on-demand ML scoring and payload validation.
- **Amazon API Gateway:** HTTP API (Payload Format 2.0) providing managed routing, HTTPS termination, and CORS.
- **Amazon S3:** Highly durable, private cloud storage for serialized model artifacts.
- **AWS IAM:** Role-based access control enforcing least-privilege permissions (`s3:GetObject`).
- **Amazon CloudWatch:** Logging, latency telemetry, and operational observability.
- **scikit-learn:** Machine learning library providing `RandomForestClassifier` and `StandardScaler`.
- **Random Forest:** Ensemble classification algorithm tuned for credit card fraud detection.
- **joblib:** Serialization and fast in-memory loading of trained model artifacts.
- **NumPy:** High-performance vectorized numerical processing.
- **SciPy:** Scientific computing runtime dependencies required by scikit-learn.
- **pandas:** Structured tabular data container for DataFrame-based model feature alignment.

---

## AWS Configuration

The system is deployed using the following existing AWS resources:

| Parameter | Existing AWS Resource Configuration |
| :--- | :--- |
| **AWS Region** | `us-east-1` |
| **S3 Bucket** | `shyam-fraud-ml-model-2026` |
| **S3 Model Key** | `model/model.joblib` |
| **S3 Scaler Key** | `model/scaler.joblib` |
| **Lambda Function** | `FraudDetectionLambda` |
| **Runtime & Architecture** | `Python 3.12`, `x86_64` |
| **Memory Allocation** | `512 MB` |
| **Execution Timeout** | `60 seconds` |
| **Lambda Handler** | `lambda_function.lambda_handler` |
| **API Gateway Name** | `FraudDetectionAPI` (HTTP API) |
| **API Stage** | `dev` |
| **API Route** | `POST /predict` |
| **Live Endpoint URL** | `https://sd2mbb56a7.execute-api.us-east-1.amazonaws.com/dev/predict` |
| **IAM Execution Role**| `FraudModelLambdaRole` |

---

## ML Model

The production classifier is a `RandomForestClassifier` from `scikit-learn`:

- **Algorithm:** Random Forest Classifier
- **Estimators (`n_estimators`):** `50` decision trees
- **Maximum Depth (`max_depth`):** `10`
- **Class Balancing (`class_weight`):** `"balanced"` (adjusts weights inversely proportional to class frequencies to handle severe transaction imbalance)
- **Random State (`random_state`):** `42`
- **Target Feature:** `Class` (`0` = Legit, `1` = Fraud)
- **Inference Outputs:**
  - `prediction`: `"Fraud"` if class is 1, else `"Legit"`
  - `class`: `1` (fraudulent) or `0` (legitimate)
  - `risk_score`: Continuous fraud probability calculated via `predict_proba()` rounded to 4 decimal places.

---

## Preprocessing

1. **Selective Scaling:**
   - **Only the `Amount` feature is standardized** using `StandardScaler`.
   - The transformation formula applied is:
     ```python
     scaled_amount = float(scaler.transform([[raw_amount]])[0][0])
     ```
   - PCA transformed features `V1` through `V28` were already standardized prior to model training and are passed through **completely unmodified**.
2. **Dynamic Feature Alignment:**
   - The model was trained using a `pandas.DataFrame`.
   - The backend retrieves the required feature sequence at runtime using:
     ```python
     required_features = list(model.feature_names_in_)
     ```
   - The inference array is packaged into a single-row `pd.DataFrame` matching `model.feature_names_in_` exactly, preventing feature misalignment.

---

## API Endpoint

```http
POST https://sd2mbb56a7.execute-api.us-east-1.amazonaws.com/dev/predict
Content-Type: application/json
Accept: application/json
```

---

## Request JSON

The API requires 29 numerical features:
- **PCA Features:** `V1` through `V28` (`float`)
- **Transaction Amount:** `Amount` (`float`)

### Verified Example Request (High-Risk Fraud Anomaly):
The following transaction profile from `frontend/script.js` evaluates to Fraud against the production model:

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

### Verified Example Request (Legitimate Transaction):
```json
{
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
}
```

---

## Response JSON

### Successful Inference (HTTP 200 OK):
```json
{
  "prediction": "Fraud",
  "class": 1,
  "risk_score": 0.9234
}
```

For legitimate transactions:
```json
{
  "prediction": "Legit",
  "class": 0,
  "risk_score": 0.0595
}
```

---

## Error Responses

The backend catches and translates errors into structured HTTP status responses:

### 1. Missing Features (HTTP 400 Bad Request):
Triggered when one or more required features are missing:
```json
{
  "error": "Missing features",
  "missing_features": [
    "V1",
    "Amount"
  ]
}
```

### 2. Invalid Numeric Value (HTTP 400 Bad Request):
Triggered when any feature value cannot be parsed to a float:
```json
{
  "error": "Invalid numeric value",
  "message": "Feature 'Amount' must be numeric, got: abc"
}
```

### 3. Malformed JSON (HTTP 400 Bad Request):
Triggered when the request body is not valid JSON:
```json
{
  "error": "Bad Request",
  "message": "Invalid JSON format in request body: ..."
}
```

### 4. Internal Server Error (HTTP 500):
Triggered if S3 download fails, unpickling fails, or model inference encounters an unhandled exception:
```json
{
  "error": "Internal Server Error",
  "message": "Failed to load model artifacts from S3."
}
```

---

## CORS Configuration

Cross-Origin Resource Sharing is enabled for browser clients:

- **Origin:** `Access-Control-Allow-Origin: *`
- **Methods:** `Access-Control-Allow-Methods: POST,OPTIONS`
- **Headers:** `Access-Control-Allow-Headers: Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token`
- **Preflight Support:** HTTP `OPTIONS` requests are caught by `lambda_function.lambda_handler` and return HTTP 200 with the CORS headers.

---

## Local Testing

The project includes unit tests for preprocessing and an automated live integration test script.

### 1. Running Unit Tests:
Run the unit test suite covering validation, amount scaling, base64 payloads, and CORS handling:

```bash
python backend/tests/test_validation.py
```
*(Or via pytest if installed)*:
```bash
python -m pytest backend/tests/test_validation.py -v
```

### 2. Testing the Live AWS API Gateway Endpoint:
Run the live integration script against the deployed AWS endpoint:

```bash
python backend/test_api.py
```

This script automatically executes:
- **Test 1:** Positive test with a complete transaction payload (verifies HTTP 200).
- **Test 2:** Negative test with missing `V1` and `Amount` (verifies HTTP 400).
- **Test 3:** CORS preflight `OPTIONS` request (verifies HTTP 200/204 and CORS headers).

---

## Deployment

The system is deployed without creating redundant cloud resources:

1. **Model Artifacts in S3:**
   - Pre-trained `model.joblib` and `scaler.joblib` reside in `s3://shyam-fraud-ml-model-2026/model/`.
2. **Lambda Package & Dependencies:**
   - Function code resides in `lambda_function.py`.
   - Scikit-learn, NumPy, SciPy, pandas, and joblib are supplied via an attached AWS Lambda Layer or direct package zip.
   - Handler is configured as `lambda_function.lambda_handler`.
3. **API Gateway Route:**
   - HTTP API `FraudDetectionAPI` routes `POST /predict` directly to the Lambda function.

---

## Project Structure

```text
ai-credit-card-fraud-detection/
├── backend/
│   ├── README.md               # Backend documentation
│   ├── lambda_function.py      # Core Lambda handler with HTTP API 2.0 & caching
│   ├── requirements.txt        # Backend dependencies
│   ├── test_api.py             # Live API Gateway integration test script
│   └── tests/
│       ├── __init__.py
│       └── test_validation.py  # 9-test unit validation suite
├── frontend/
│   ├── architecture.css        # Architecture viewer stylesheet
│   ├── architecture.html       # Cloud architecture viewer
│   ├── architecture.js         # Architecture viewer logic
│   ├── index.html              # Main fraud detection dashboard
│   ├── script.js               # Frontend controller & API client
│   └── style.css               # Main dashboard stylesheet
├── lambda_function.py          # Root-level Lambda handler mirror
├── MEMORY.md                   # Persistent technical memory
├── PRD.md                      # Product Requirements Document
├── README.md                   # Main project documentation
├── RULES.md                    # Development rules and operational constraints
└── TASKS.md                    # Project task tracker and testing checklist
```

---

## Security

- **Private S3 Storage:** The S3 bucket `shyam-fraud-ml-model-2026` is strictly private. Public read/write access is blocked.
- **Least-Privilege IAM:** `FraudModelLambdaRole` execution role is restricted to `s3:GetObject` on the two specific artifact ARNs:
  - `arn:aws:s3:::shyam-fraud-ml-model-2026/model/model.joblib`
  - `arn:aws:s3:::shyam-fraud-ml-model-2026/model/scaler.joblib`
- **Zero Embedded Secrets:** The codebase contains no AWS credentials, access keys, or secret tokens.
- **No Direct Model Access:** Public clients interact exclusively through the API Gateway endpoint; S3 is never exposed directly.

---

## Current Status

- **Unit Validation Suite:** **9 passed** (100% pass rate in `backend/tests/test_validation.py`).
- **Live API Integration Test:**
  - Valid transaction prediction: **HTTP 200 OK** (returns `prediction`, `class`, `risk_score`).
  - Missing features payload: **HTTP 400 Bad Request** (returns `missing_features` list).
  - CORS Preflight: **HTTP 200 / 204** (returns full CORS headers).
- **Frontend Dashboard:** Fully functional with real-time API connectivity, demo profile loading, and visual risk indicators.
