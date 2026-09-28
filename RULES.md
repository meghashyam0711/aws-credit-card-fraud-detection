# Development & Operational Rules (RULES.md)

> **Project Title:** “Innovate Faster with Machine Learning (ML) on AWS”  
> **System:** AWS-based Credit Card Fraud Detection System  
> **Repository:** `https://github.com/meghashyam0711/ai-credit-card-fraud-detection`  
> **Scope:** Mandatory guidelines and hard operational constraints for any developer, automated agent, or AI assistant working on this codebase.

---

## 1. AWS Infrastructure Rules

- **Rule 1.1 — Resource Exclusivity:** Never create duplicate or alternative S3 buckets, Lambda functions, API Gateways, or CloudWatch log groups. All changes must utilize the existing defined infrastructure:
  - Bucket: `shyam-fraud-ml-model-2026`
  - Lambda: `FraudDetectionLambda`
  - API Gateway: `FraudDetectionAPI` (`dev` stage, route `POST /predict`)
- **Rule 1.2 — Zero Deletions:** Never delete, rename, or teardown existing AWS cloud resources or S3 objects.
- **Rule 1.3 — Strict S3 Privacy:** Never modify the access policy of S3 bucket `shyam-fraud-ml-model-2026` to make it public. "Block Public Access" must remain fully enabled.
- **Rule 1.4 — Zero Hardcoded Credentials:** Never store AWS Access Keys, Secret Keys, Session Tokens, or sensitive credentials anywhere in the repository, frontend, or Lambda environment variables.
- **Rule 1.5 — Least-Privilege IAM:** Maintain least-privilege IAM policies on `FraudModelLambdaRole`. Access is restricted strictly to `s3:GetObject` on `model/model.joblib` and `model/scaler.joblib`, plus standard CloudWatch logging.

---

## 2. Machine Learning & Preprocessing Rules

- **Rule 2.1 — Artifact Immutability:** Never overwrite, replace, modify, or retrain the existing production model artifacts (`model/model.joblib` and `model/scaler.joblib`) unless explicitly requested and approved by the project owner.
- **Rule 2.2 — Fixed ML Configuration:** Never alter the Random Forest algorithm or its production hyperparameters:
  - `n_estimators = 50`
  - `max_depth = 10`
  - `class_weight = "balanced"`
  - `random_state = 42`
- **Rule 2.3 — Preprocessing Integrity:** Never alter the preprocessing pipeline:
  - **Only `Amount` is standardized** using `StandardScaler`.
  - PCA features `V1` through `V28` must remain **unscaled and unmodified**.
  - Do not introduce min-max scaling, log scaling, or clipping.
- **Rule 2.4 — Dynamic Feature Alignment:** The backend must dynamically determine the expected feature list and feature ordering directly from `model.feature_names_in_`.
- **Rule 2.5 — No Positional Hardcoding:** Never assume a hardcoded positional order for features. Featurized data must be structured as a `pandas.DataFrame` whose columns match `model.feature_names_in_` exactly.

---

## 3. Backend Implementation Rules

- **Rule 3.1 — Preserved Entry Point:** Keep `lambda_function.lambda_handler` as the primary execution entry point.
- **Rule 3.2 — Preserved API Route:** Keep the HTTP API route as `POST /predict` on stage `dev`.
- **Rule 3.3 — Protocol Compatibility:** Always maintain support for:
  - API Gateway HTTP API Payload Format 2.0.
  - REST API / Payload Format 1.0 proxy events.
  - Direct Lambda test events (plain dictionary).
  - Base64-encoded request bodies.
- **Rule 3.4 — Persistent CORS Compliance:** All backend responses (success HTTP 200, client error HTTP 400, internal error HTTP 500, and preflight `OPTIONS` HTTP 200/204) must include standard CORS headers allowing origin `*`.
- **Rule 3.5 — Strict Validation & Safe Error Responses:**
  - Validate that every feature in `model.feature_names_in_` is present.
  - Validate that all feature values are numeric (`float` convertible).
  - Return HTTP 400 for bad input with explicit error details.
  - Never allow unhandled exceptions or stack traces to leak to the client.
- **Rule 3.6 — No Heavyweight Web Frameworks:** Do not introduce Flask, FastAPI, Django, or Mangum into the Lambda runtime. The handler must remain lightweight and serverless-native.
- **Rule 3.7 — Warm Cache Preservation:** Retain the global in-memory caching mechanism (`_MODEL`, `_SCALER`, `_REQUIRED_FEATURES`) and `/tmp` local disk cache to ensure sub-20ms warm executions.

---

## 4. Frontend Integration Rules

- **Rule 4.1 — Existing Endpoint Alignment:** The frontend must communicate strictly with the existing live API endpoint:
  `https://sd2mbb56a7.execute-api.us-east-1.amazonaws.com/dev/predict`
- **Rule 4.2 — No Backend Architectural Shifts:** Do not alter the backend architecture or request schema to cater to cosmetic UI requirements.
- **Rule 4.3 — Zero Credential Exposure:** Never add AWS SDKs with embedded credentials or access tokens into frontend HTML or JavaScript files.
- **Rule 4.4 — Honest Fallback Reporting:** If the network or API is down, the frontend may display an amount-based rule fallback, but it must explicitly mark the outcome as `Fallback` with `Risk Score: N/A`. Never fake or invent an ML model risk score.

---

## 5. Repository & Workflow Rules

- **Rule 5.1 — File Preservation:** Do not delete, rename, or move existing working files in `backend/` or `frontend/`.
- **Rule 5.2 — Inspect Before Modifying:** Always inspect existing code, directory structures, and configurations before proposing or making changes.
- **Rule 5.3 — Test After Every Change:** Run the unit validation suite (`python backend/tests/test_validation.py`) and verify that all 9 tests pass whenever backend code is adjusted.
- **Rule 5.4 — No Architectural Deviations:** Do not introduce Docker containers, Kubernetes, EC2 instances, SageMaker endpoints, or alternative databases unless explicitly instructed by the user.
- **Rule 5.5 — Safe Version Control:** Do not commit or push to Git automatically. Do not edit `.gitignore` unless explicitly requested.
