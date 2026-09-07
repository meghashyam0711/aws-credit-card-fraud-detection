import os
import json
import base64
import logging
from typing import Any, Dict, List, Tuple

import boto3
from botocore.exceptions import ClientError
import joblib
import numpy as np
import pandas as pd

# Configure CloudWatch logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# S3 Configuration matching existing AWS resources
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "shyam-fraud-ml-model-2026")
MODEL_KEY = os.environ.get("MODEL_KEY", "model/model.joblib")
SCALER_KEY = os.environ.get("SCALER_KEY", "model/scaler.joblib")
CACHE_DIR = os.environ.get("CACHE_DIR", "/tmp")

# Global in-memory cache across warm Lambda invocations
_MODEL = None
_SCALER = None
_REQUIRED_FEATURES: List[str] = []

# S3 client using IAM role credentials (no hard-coded keys)
s3_client = boto3.client("s3")

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
    "Access-Control-Allow-Methods": "POST,OPTIONS"
}


class InputValidationError(Exception):
    """Exception raised for client-side input validation errors (HTTP 400)."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


def build_response(status_code: int, body_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format standard API Gateway HTTP API response with CORS headers.
    """
    return {
        "statusCode": status_code,
        "headers": CORS_HEADERS,
        "body": json.dumps(body_dict)
    }


def download_from_s3_if_needed(bucket: str, key: str, destination_path: str) -> str:
    """
    Download S3 object to local /tmp cache if not already present.
    """
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    if not os.path.exists(destination_path):
        logger.info(f"Downloading s3://{bucket}/{key} to {destination_path}...")
        try:
            s3_client.download_file(bucket, key, destination_path)
            logger.info(f"Successfully downloaded {key}.")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            logger.error(f"S3 ClientError downloading s3://{bucket}/{key}: [{error_code}] {error_msg}")
            raise RuntimeError(f"Could not download model artifact from S3: {error_msg}")
        except Exception as e:
            logger.error(f"Unexpected error downloading s3://{bucket}/{key}: {str(e)}")
            raise RuntimeError(f"Unexpected S3 download failure: {str(e)}")
    else:
        logger.info(f"Model artifact already cached in /tmp at {destination_path}.")
    return destination_path


def load_artifacts() -> Tuple[Any, Any, List[str]]:
    """
    Load model and scaler into global memory cache.
    Reuses cached objects on warm starts for sub-20ms latency.
    """
    global _MODEL, _SCALER, _REQUIRED_FEATURES

    if _MODEL is not None and _SCALER is not None and _REQUIRED_FEATURES:
        logger.info("Reusing cached model and scaler from memory (warm start).")
        return _MODEL, _SCALER, _REQUIRED_FEATURES

    logger.info("Initializing cold start: downloading/loading model and scaler artifacts...")
    model_local_path = os.path.join(CACHE_DIR, os.path.basename(MODEL_KEY))
    scaler_local_path = os.path.join(CACHE_DIR, os.path.basename(SCALER_KEY))

    # Download from S3 if missing from /tmp disk
    download_from_s3_if_needed(MODEL_BUCKET, MODEL_KEY, model_local_path)
    download_from_s3_if_needed(MODEL_BUCKET, SCALER_KEY, scaler_local_path)

    # Load artifacts using joblib
    try:
        _MODEL = joblib.load(model_local_path)
        _SCALER = joblib.load(scaler_local_path)
        logger.info("Successfully loaded model.joblib and scaler.joblib via joblib.")
    except Exception as e:
        logger.error(f"Failed to load joblib artifacts: {str(e)}")
        raise RuntimeError(f"Model artifact unpickling failure: {str(e)}")

    # Dynamically extract feature names and order from model.feature_names_in_
    if hasattr(_MODEL, "feature_names_in_"):
        _REQUIRED_FEATURES = [str(f) for f in _MODEL.feature_names_in_]
        logger.info(f"Dynamically discovered {len(_REQUIRED_FEATURES)} features from model.feature_names_in_.")
    else:
        logger.error("model.joblib does not contain feature_names_in_ attribute.")
        raise RuntimeError("Model does not have feature_names_in_ attribute. Ensure model was fitted on a DataFrame.")

    return _MODEL, _SCALER, _REQUIRED_FEATURES


def parse_event(event: Any) -> Dict[str, Any]:
    """
    Extract transaction JSON payload from API Gateway HTTP API 2.0 or direct Lambda test event.
    """
    if event is None:
        raise InputValidationError("Empty request payload received.")

    data = event

    # Handle API Gateway HTTP API (v2.0) or REST API proxy event
    if isinstance(event, dict) and "body" in event:
        body = event.get("body")
        if body is None or body == "":
            raise InputValidationError("Request body is empty.")

        # Decode base64 if required
        if event.get("isBase64Encoded", False) and isinstance(body, str):
            try:
                body = base64.b64decode(body).decode("utf-8")
            except Exception as e:
                raise InputValidationError(f"Invalid base64 encoding in body: {str(e)}")

        # Parse string body as JSON
        if isinstance(body, str):
            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                raise InputValidationError(f"Invalid JSON format in request body: {str(e)}")
        elif isinstance(body, dict):
            data = body
        else:
            raise InputValidationError(f"Unsupported request body type: {type(body).__name__}")

    # Unwrap optional envelope key e.g. {"transaction": {...}}
    if isinstance(data, dict) and "transaction" in data and isinstance(data["transaction"], dict):
        data = data["transaction"]

    if not isinstance(data, dict):
        raise InputValidationError(f"Expected JSON object (dictionary) for transaction, got {type(data).__name__}")

    return data


def validate_and_prepare_features(transaction: Dict[str, Any],
                                   required_features: List[str],
                                   scaler: Any) -> pd.DataFrame:
    """
    1. Validates all model features are present.
    2. Scales ONLY the 'Amount' feature using scaler.joblib.
    3. Keeps all other features unchanged.
    4. Arranges features in the exact order of model.feature_names_in_.
    5. Returns a pandas DataFrame preserving feature names.
    """
    # 1. Validate required features
    missing = [feat for feat in required_features if feat not in transaction]
    if missing:
        logger.warning(f"Validation failed: missing {len(missing)} required features.")
        raise InputValidationError(
            "Missing features",
            details={
                "error": "Missing features",
                "missing_features": missing
            }
        )

    # 2. Scale ONLY Amount feature
    raw_amount = transaction.get("Amount")
    try:
        raw_amount_float = float(raw_amount)
    except (TypeError, ValueError):
        raise InputValidationError(
            "Invalid numeric value",
            details={
                "error": "Invalid numeric value",
                "message": f"Feature 'Amount' must be numeric, got: {raw_amount}"
            }
        )

    try:
        scaled_amount = float(scaler.transform([[raw_amount_float]])[0][0])
    except Exception as e:
        logger.error(f"Error transforming Amount with scaler: {str(e)}")
        raise RuntimeError(f"Scaler transformation error for Amount: {str(e)}")

    # 3. Arrange features in exact model.feature_names_in_ order
    ordered_values: List[float] = []
    for feat in required_features:
        if feat == "Amount":
            ordered_values.append(scaled_amount)
        else:
            val = transaction[feat]
            try:
                ordered_values.append(float(val))
            except (TypeError, ValueError):
                raise InputValidationError(
                    "Invalid numeric value",
                    details={
                        "error": "Invalid numeric value",
                        "message": f"Feature '{feat}' must be numeric, got: {val}"
                    }
                )

    # 5. Create a pandas DataFrame
    input_df = pd.DataFrame(
        [ordered_values],
        columns=required_features
    )

    # 6. Return input_df instead of a NumPy array
    return input_df


def lambda_handler(event: Any, context: Any = None) -> Dict[str, Any]:
    """
    Main AWS Lambda entry point for Credit Card Fraud Detection inference.
    Handles CORS preflight (OPTIONS), validation, S3 caching, and ML scoring.
    """
    # Determine HTTP method from API Gateway HTTP API v2.0 or v1.0
    http_method = "POST"
    if isinstance(event, dict):
        # Payload Format 2.0: event["requestContext"]["http"]["method"]
        http_method = event.get("requestContext", {}).get("http", {}).get("method") or event.get("httpMethod", "POST")

    logger.info(f"Lambda invocation started: method={http_method}")

    # Handle CORS preflight OPTIONS request for browser integration
    if http_method.upper() == "OPTIONS":
        logger.info("CORS preflight OPTIONS request handled.")
        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps({"message": "CORS preflight OK"})
        }

    # 1. Load model & scaler (warm starts reuse in-memory cache)
    try:
        model, scaler, required_features = load_artifacts()
        feature_names = model.feature_names_in_
    except Exception as e:
        logger.error(f"Artifact loading error: {str(e)}", exc_info=True)
        return build_response(500, {
            "error": "Internal Server Error",
            "message": "Failed to load model artifacts from S3."
        })

    # 2. Parse request payload
    try:
        transaction_data = parse_event(event)
    except InputValidationError as e:
        logger.warning(f"Payload parse validation error: {e.message}")
        error_body = e.details if e.details else {"error": "Bad Request", "message": e.message}
        return build_response(400, error_body)
    except Exception as e:
        logger.error(f"Unexpected error parsing payload: {str(e)}", exc_info=True)
        return build_response(500, {
            "error": "Internal Server Error",
            "message": "Error parsing request payload."
        })

    # 3. Validate features and scale Amount
    try:
        input_df = validate_and_prepare_features(transaction_data, required_features, scaler)
    except InputValidationError as e:
        logger.warning(f"Feature validation error: {e.message}")
        error_body = e.details if e.details else {"error": "Bad Request", "message": e.message}
        return build_response(400, error_body)
    except Exception as e:
        logger.error(f"Feature preparation error: {str(e)}", exc_info=True)
        return build_response(500, {
            "error": "Internal Server Error",
            "message": "Feature preprocessing failed."
        })

    # 4. Execute ML Prediction
    try:
        preds = model.predict(input_df)
        pred_class = int(preds[0])

        probs = model.predict_proba(input_df)[0]
        classes_list = list(model.classes_)
        if 1 in classes_list:
            class_1_idx = classes_list.index(1)
            risk_score = float(probs[class_1_idx])
        else:
            risk_score = float(probs[-1])

        prediction_label = "Fraud" if pred_class == 1 else "Legit"

        logger.info(f"Prediction completed: class={pred_class}, prediction={prediction_label}, risk_score={risk_score:.4f}")

        return build_response(200, {
            "prediction": prediction_label,
            "class": pred_class,
            "risk_score": round(risk_score, 4)
        })

    except Exception as e:
        logger.error(f"Model prediction error: {str(e)}", exc_info=True)
        return build_response(500, {
            "error": "Internal Server Error",
            "message": "Model inference execution failed."
        })
