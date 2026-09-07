import os
import json
import base64
import logging
import warnings
from typing import Any, Dict, List, Tuple

import boto3
from botocore.exceptions import ClientError
import joblib
import numpy as np

# Configure logging for AWS CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Suppress sklearn feature name warning when predicting on numpy array
warnings.filterwarnings("ignore", category=UserWarning)

# S3 Configuration from environment variables with defaults
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "shyam-fraud-ml-model-2026")
MODEL_KEY = os.environ.get("MODEL_KEY", "model/model.joblib")
SCALER_KEY = os.environ.get("SCALER_KEY", "model/scaler.joblib")
CACHE_DIR = os.environ.get("CACHE_DIR", "/tmp")

# Global memory cache across warm Lambda invocations
_MODEL = None
_SCALER = None
_REQUIRED_FEATURES: List[str] = []

# Initialize S3 client outside handler for reuse
s3_client = boto3.client("s3")


class InputValidationError(Exception):
    """Exception raised for client-side input validation errors (HTTP 400)."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


def build_response(status_code: int, body_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format standard API Gateway response with CORS headers.
    """
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods": "POST,OPTIONS"
        },
        "body": json.dumps(body_dict)
    }


def download_from_s3_if_needed(bucket: str, key: str, destination_path: str) -> str:
    """
    Download an S3 object to local cache (/tmp) if not already present.
    """
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    if not os.path.exists(destination_path):
        logger.info(f"Downloading s3://{bucket}/{key} to {destination_path}...")
        try:
            s3_client.download_file(bucket, key, destination_path)
            logger.info(f"Successfully downloaded {key}.")
        except ClientError as e:
            logger.error(f"Failed to download s3://{bucket}/{key}: {str(e)}")
            raise RuntimeError(f"Could not download s3://{bucket}/{key}: {e.response.get('Error', {}).get('Message', str(e))}")
        except Exception as e:
            logger.error(f"Unexpected error downloading s3://{bucket}/{key}: {str(e)}")
            raise RuntimeError(f"Unexpected error downloading s3://{bucket}/{key}: {str(e)}")
    else:
        logger.info(f"File already cached at {destination_path}.")
    return destination_path


def load_artifacts() -> Tuple[Any, Any, List[str]]:
    """
    Load model and scaler into global memory cache.
    Reuses cached objects on warm starts to achieve low latency.
    """
    global _MODEL, _SCALER, _REQUIRED_FEATURES

    if _MODEL is not None and _SCALER is not None and _REQUIRED_FEATURES:
        return _MODEL, _SCALER, _REQUIRED_FEATURES

    model_local_path = os.path.join(CACHE_DIR, os.path.basename(MODEL_KEY))
    scaler_local_path = os.path.join(CACHE_DIR, os.path.basename(SCALER_KEY))

    # Download from S3 if missing in local disk cache
    download_from_s3_if_needed(MODEL_BUCKET, MODEL_KEY, model_local_path)
    download_from_s3_if_needed(MODEL_BUCKET, SCALER_KEY, scaler_local_path)

    # Load artifacts using joblib
    logger.info("Loading model and scaler using joblib...")
    try:
        _MODEL = joblib.load(model_local_path)
        _SCALER = joblib.load(scaler_local_path)
    except Exception as e:
        logger.error(f"Error loading joblib artifacts: {str(e)}")
        raise RuntimeError(f"Failed to load model or scaler artifacts: {str(e)}")

    # Dynamically extract feature names from the trained model
    if hasattr(_MODEL, "feature_names_in_"):
        _REQUIRED_FEATURES = [str(f) for f in _MODEL.feature_names_in_]
        logger.info(f"Dynamically discovered {len(_REQUIRED_FEATURES)} features from model.feature_names_in_.")
    else:
        raise RuntimeError("Model does not have feature_names_in_ attribute. Ensure model was trained with feature names.")

    return _MODEL, _SCALER, _REQUIRED_FEATURES


def parse_event(event: Any) -> Dict[str, Any]:
    """
    Parse transaction payload from direct Lambda invocations or API Gateway events.
    """
    if event is None:
        raise InputValidationError("Empty event received.")

    data = event

    # Handle API Gateway proxy integration where payload is in 'body'
    if isinstance(event, dict) and "body" in event:
        body = event.get("body")
        if body is None:
            raise InputValidationError("Request body is empty.")

        # Decode base64 if flag is present
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
            raise InputValidationError(f"Unsupported body type: {type(body).__name__}")

    # Unwrap optional wrapper key e.g. {"transaction": {...}}
    if isinstance(data, dict) and "transaction" in data and isinstance(data["transaction"], dict):
        data = data["transaction"]

    if not isinstance(data, dict):
        raise InputValidationError(f"Expected a JSON object (dictionary) for transaction, got {type(data).__name__}")

    return data


def validate_and_prepare_features(transaction: Dict[str, Any],
                                   required_features: List[str],
                                   scaler: Any) -> np.ndarray:
    """
    1. Validates that all model features are present.
    2. Scales ONLY the 'Amount' feature using scaler.joblib.
    3. Keeps all other features unchanged.
    4. Arranges features in the exact order expected by the trained model.
    """
    # Check for missing features
    missing = [feat for feat in required_features if feat not in transaction]
    if missing:
        raise InputValidationError(
            f"Transaction payload is missing {len(missing)} required feature(s).",
            details={
                "missing_features": missing,
                "required_features_count": len(required_features),
                "provided_features_count": len(transaction)
            }
        )

    # Scale ONLY the Amount feature using the pre-fitted scaler
    raw_amount = transaction.get("Amount")
    try:
        raw_amount_float = float(raw_amount)
    except (TypeError, ValueError):
        raise InputValidationError(f"Invalid value for 'Amount': {raw_amount}. Must be a valid numeric value.")

    try:
        # StandardScaler expects 2D array [[value]]
        scaled_amount = float(scaler.transform([[raw_amount_float]])[0][0])
    except Exception as e:
        logger.error(f"Failed to scale 'Amount': {str(e)}")
        raise RuntimeError(f"Error during Amount feature scaling: {str(e)}")

    # Arrange features in the exact order defined by model.feature_names_in_
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
                    f"Feature '{feat}' must be numeric, got: {val}"
                )

    # 2D numpy array with shape (1, n_features)
    return np.array([ordered_values], dtype=np.float64)


def lambda_handler(event: Any, context: Any = None) -> Dict[str, Any]:
    """
    Main AWS Lambda entry point for serverless fraud detection.
    """
    logger.info("Received request for fraud detection inference.")

    # 1. Load artifacts (warm start uses memory cache)
    try:
        model, scaler, required_features = load_artifacts()
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {str(e)}", exc_info=True)
        return build_response(500, {
            "error": "Internal Server Error",
            "message": f"Artifact loading failure: {str(e)}"
        })

    # 2. Parse incoming event payload
    try:
        transaction_data = parse_event(event)
    except InputValidationError as e:
        logger.warning(f"Payload validation error: {e.message}")
        response_data = {
            "error": "Bad Request",
            "message": e.message
        }
        if e.details:
            response_data.update(e.details)
        return build_response(400, response_data)
    except Exception as e:
        logger.error(f"Unexpected error parsing payload: {str(e)}", exc_info=True)
        return build_response(500, {
            "error": "Internal Server Error",
            "message": f"Error parsing request payload: {str(e)}"
        })

    # 3. Validate features and perform Amount scaling
    try:
        X = validate_and_prepare_features(transaction_data, required_features, scaler)
    except InputValidationError as e:
        logger.warning(f"Feature validation failed: {e.message}")
        response_data = {
            "error": "Bad Request",
            "message": e.message
        }
        if e.details:
            response_data.update(e.details)
        return build_response(400, response_data)
    except Exception as e:
        logger.error(f"Feature preparation error: {str(e)}", exc_info=True)
        return build_response(500, {
            "error": "Internal Server Error",
            "message": f"Feature preparation failure: {str(e)}"
        })

    # 4. Run model inference: predict() and predict_proba()
    try:
        predictions = model.predict(X)
        pred_class = int(predictions[0])

        probabilities = model.predict_proba(X)[0]

        # Determine fraud probability (Class 1)
        classes_list = list(model.classes_)
        if 1 in classes_list:
            class_1_idx = classes_list.index(1)
            risk_score = float(probabilities[class_1_idx])
        else:
            # Fallback to last class if binary 0/1 not explicitly numbered
            risk_score = float(probabilities[-1])

        prediction_label = "Fraud" if pred_class == 1 else "Legit"

        result = {
            "prediction": prediction_label,
            "class": pred_class,
            "risk_score": round(risk_score, 6)
        }

        logger.info(f"Inference result: class={pred_class}, prediction={prediction_label}, risk_score={risk_score:.4f}")
        return build_response(200, result)

    except Exception as e:
        logger.error(f"Inference execution failure: {str(e)}", exc_info=True)
        return build_response(500, {
            "error": "Internal Server Error",
            "message": f"Model inference execution failure: {str(e)}"
        })
