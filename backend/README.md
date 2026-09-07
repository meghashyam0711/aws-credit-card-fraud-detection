# AWS-Based Credit Card Fraud Detection System - Backend

> **Project Title:** “Innovate faster with machine learning (ML) on AWS”  
> **Domain:** Serverless Machine Learning Inference & Cloud Security (HPCC Project)

---

## 1. System Architecture

```text
User / Frontend
      │
      ▼  (POST /predict - JSON Payload)
API Gateway (HTTP API: FraudDetectionAPI)
      │
      ▼  (Payload Format 2.0)
AWS Lambda (FraudDetectionLambda, Python 3.12, x86_64, 512MB)
      │
      ├── First Invocation (Cold Start): Download & Cache from S3
      │       ▼
      │   Amazon S3 (shyam-fraud-ml-model-2026)
      │       ├── model/model.joblib
      │       └── model/scaler.joblib
      │
      ├── Subsequent Invocations (Warm Start): In-Memory Cache (Sub-20ms)
      │
      ▼
Preprocessing & Inference
      ├── Validate required features (from model.feature_names_in_)
      ├── Standardize ONLY 'Amount' using StandardScaler
      └── Run RandomForestClassifier.predict() & predict_proba()
      │
      ▼  (JSON Response with CORS Headers)
API Gateway
      │
      ▼
Frontend / Client Application
```

### Supporting Cloud Services:
- **AWS IAM**: `FraudModelLambdaRole` grants least-privilege `s3:GetObject` access strictly to the model artifacts.
- **Amazon CloudWatch**: Captures operational telemetry, execution duration, cold/warm start status, and inference outcomes without leaking sensitive customer transaction details.

---

## 2. Existing AWS Resources Reference

| Parameter | Configuration |
| :--- | :--- |
| **AWS Region** | `us-east-1` |
| **S3 Bucket** | `shyam-fraud-ml-model-2026` |
| **S3 Model Key** | `model/model.joblib` |
| **S3 Scaler Key** | `model/scaler.joblib` |
| **Lambda Function** | `FraudDetectionLambda` |
| **Runtime & Arch** | `Python 3.12`, `x86_64` |
| **Memory & Timeout** | `512 MB`, `60 seconds` |
| **Lambda Handler** | `lambda_function.lambda_handler` |
| **API Gateway** | `FraudDetectionAPI` (HTTP API) |
| **Stage & Route** | Stage: `dev`, Route: `POST /predict` |
| **Prediction URL** | `https://sd2mbb56a7.execute-api.us-east-1.amazonaws.com/dev/predict` |
| **IAM Execution Role**| `FraudModelLambdaRole` |

---

## 3. Directory Structure

```text
backend/
├── lambda_function.py       # Core Lambda handler with HTTP API 2.0, caching, & CORS
├── requirements.txt         # Production runtime dependencies
├── test_api.py              # Automated live API Gateway test script
├── README.md                # System documentation & setup guide
└── tests/
    ├── __init__.py
    └── test_validation.py   # Unit tests for preprocessing, scaling, & CORS
```

---

## 4. API Specification

### Endpoint:
```http
POST https://sd2mbb56a7.execute-api.us-east-1.amazonaws.com/dev/predict
Content-Type: application/json
```

### Request JSON Format:
All 30 features from `creditcard.csv` (`Time`, `V1` to `V28`, and `Amount`) are required.

```json
{
  "Time": 406.0,
  "V1": -2.312226542,
  "V2": 1.951992011,
  "V3": -1.609850732,
  "V4": 3.997905588,
  "V5": -0.522187865,
  "V6": -1.426545319,
  "V7": -2.537387304,
  "V8": 1.391657248,
  "V9": -2.770089277,
  "V10": -2.772272145,
  "V11": 3.202033207,
  "V12": -2.899907388,
  "V13": -0.595221881,
  "V14": -4.289253782,
  "V15": 0.389724120,
  "V16": -1.140747180,
  "V17": -2.830055675,
  "V18": -0.016822468,
  "V19": 0.416955705,
  "V20": 0.126910559,
  "V21": 0.517232371,
  "V22": -0.035049369,
  "V23": -0.465211076,
  "V24": 0.320198198,
  "V25": 0.044519167,
  "V26": 0.177839798,
  "V27": 0.261145003,
  "V28": -0.143275875,
  "Amount": 149.62
}
```

### Successful Response (HTTP 200 OK):
```json
{
  "prediction": "Fraud",
  "class": 1,
  "risk_score": 0.9234
}
```
*(Or `"prediction": "Legit"`, `"class": 0`, `"risk_score": 0.0125`)*

### Validation Error Response (HTTP 400 Bad Request):
When any required model features are missing:
```json
{
  "error": "Missing features",
  "missing_features": [
    "V1",
    "Amount"
  ]
}
```

### Preflight CORS Response (HTTP 200 OK):
`OPTIONS` requests receive immediate preflight approval:
```http
HTTP/1.1 200 OK
Access-Control-Allow-Origin: *
Access-Control-Allow-Headers: Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token
Access-Control-Allow-Methods: POST,OPTIONS
```

---

## 5. Preprocessing & ML Rules

1. **Feature Ordering**:
   Features are dynamically mapped using `model.feature_names_in_` extracted from `model.joblib`. Features are never hard-coded in arbitrary order.
2. **Amount Scaling**:
   `Amount` is the ONLY column scaled using `scaler.transform([[Amount]])[0][0]`. Features `Time` and `V1`..`V28` are preserved in their raw values.
3. **In-Memory Caching**:
   `model.joblib` and `scaler.joblib` are saved locally to `/tmp` and kept in global Python variables `_MODEL` and `_SCALER`. Warm invocations bypass S3 completely.

---

## 6. Testing Guide

### Running Local Preprocessing & Validation Unit Tests:
```bash
python backend/tests/test_validation.py
```
*Validates feature validation, dynamic extraction, amount standardization, CORS headers, and API Gateway payload handling.*

### Testing the Live AWS API Gateway Endpoint:
```bash
python backend/test_api.py
```
*Sends live HTTP requests (Positive test, Negative missing-features test, and CORS OPTIONS test) directly to the API Gateway URL.*

---

## 7. Least-Privilege IAM Policy (`FraudModelLambdaRole`)

Attach this inline policy to `FraudModelLambdaRole` in AWS IAM:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowS3ModelArtifactDownload",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": [
        "arn:aws:s3:::shyam-fraud-ml-model-2026/model/model.joblib",
        "arn:aws:s3:::shyam-fraud-ml-model-2026/model/scaler.joblib"
      ]
    },
    {
      "Sid": "AllowCloudWatchLogging",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-east-1:*:*"
    }
  ]
}
```
