"""
Unit tests for AWS Lambda backend preprocessing, validation, CORS, and Payload Format 2.0 handling.
"""

import sys
import os
import json
import base64
import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Add parent directory to sys.path to import lambda_function
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import lambda_function


class TestFraudDetectionBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 30 creditcard.csv model features (Time, V1..V28, Amount)
        cls.feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

        # Train a mock RandomForestClassifier with DataFrame to set feature_names_in_
        rng = np.random.RandomState(42)
        X_df = pd.DataFrame(rng.randn(20, len(cls.feature_names)), columns=cls.feature_names)
        y_dummy = np.array([0] * 18 + [1] * 2)

        cls.mock_model = RandomForestClassifier(
            n_estimators=10,
            max_depth=5,
            class_weight="balanced",
            random_state=42
        )
        cls.mock_model.fit(X_df, y_dummy)

        # Train a mock StandardScaler on Amount
        cls.mock_scaler = StandardScaler()
        amount_dummy = np.array([[10.0], [50.0], [100.0], [250.0], [1000.0]])
        cls.mock_scaler.fit(amount_dummy)

        # Base valid transaction dictionary
        cls.valid_transaction = {f: 0.25 for f in cls.feature_names}
        cls.valid_transaction["Time"] = 405.0
        cls.valid_transaction["Amount"] = 125.50

    def setUp(self):
        # Inject mocks into module cache
        lambda_function._MODEL = self.mock_model
        lambda_function._SCALER = self.mock_scaler
        lambda_function._REQUIRED_FEATURES = list(self.mock_model.feature_names_in_)

    def test_dynamic_feature_inspection(self):
        """Verify model.feature_names_in_ correctly dictates required features and ordering."""
        self.assertEqual(len(lambda_function._REQUIRED_FEATURES), 30)
        self.assertEqual(lambda_function._REQUIRED_FEATURES[0], "Time")
        self.assertEqual(lambda_function._REQUIRED_FEATURES[-1], "Amount")

    def test_missing_features_returns_400(self):
        """Verify HTTP 400 when required features are missing."""
        incomplete = dict(self.valid_transaction)
        del incomplete["V1"]
        del incomplete["Amount"]

        event = {
            "version": "2.0",
            "requestContext": {"http": {"method": "POST"}},
            "body": json.dumps(incomplete)
        }
        res = lambda_function.lambda_handler(event)
        self.assertEqual(res["statusCode"], 400)
        body = json.loads(res["body"])
        self.assertEqual(body.get("error"), "Missing features")
        self.assertIn("missing_features", body)
        self.assertIn("V1", body["missing_features"])
        self.assertIn("Amount", body["missing_features"])

    def test_amount_scaling_only(self):
        """Verify ONLY Amount is transformed by the scaler while other features stay intact; returns DataFrame."""
        raw_amount = self.valid_transaction["Amount"]
        expected_scaled = float(self.mock_scaler.transform([[raw_amount]])[0][0])

        X = lambda_function.validate_and_prepare_features(
            self.valid_transaction,
            lambda_function._REQUIRED_FEATURES,
            self.mock_scaler
        )

        self.assertIsInstance(X, pd.DataFrame)
        self.assertEqual(list(X.columns), lambda_function._REQUIRED_FEATURES)
        self.assertAlmostEqual(float(X["Amount"].iloc[0]), expected_scaled, places=4)
        self.assertEqual(float(X["V1"].iloc[0]), self.valid_transaction["V1"])

    def test_cors_preflight_options(self):
        """Verify OPTIONS preflight returns 200 with all required CORS headers."""
        event = {
            "version": "2.0",
            "requestContext": {"http": {"method": "OPTIONS"}}
        }
        res = lambda_function.lambda_handler(event)
        self.assertEqual(res["statusCode"], 200)
        headers = res["headers"]
        self.assertEqual(headers["Access-Control-Allow-Origin"], "*")
        self.assertIn("POST,OPTIONS", headers["Access-Control-Allow-Methods"])
        self.assertIn("Content-Type", headers["Access-Control-Allow-Headers"])

    def test_api_gateway_v2_post_success(self):
        """Verify successful prediction via API Gateway HTTP API v2.0 POST request."""
        event = {
            "version": "2.0",
            "requestContext": {"http": {"method": "POST"}},
            "body": json.dumps(self.valid_transaction)
        }
        res = lambda_function.lambda_handler(event)
        self.assertEqual(res["statusCode"], 200)
        body = json.loads(res["body"])
        self.assertIn("prediction", body)
        self.assertIn("class", body)
        self.assertIn("risk_score", body)
        self.assertIn(body["prediction"], ["Fraud", "Legit"])
        self.assertIn(body["class"], [0, 1])
        self.assertTrue(0.0 <= body["risk_score"] <= 1.0)

    def test_base64_encoded_body(self):
        """Verify support for base64-encoded request bodies."""
        raw_json = json.dumps(self.valid_transaction)
        b64_body = base64.b64encode(raw_json.encode("utf-8")).decode("utf-8")
        event = {
            "version": "2.0",
            "requestContext": {"http": {"method": "POST"}},
            "isBase64Encoded": True,
            "body": b64_body
        }
        res = lambda_function.lambda_handler(event)
        self.assertEqual(res["statusCode"], 200)
        body = json.loads(res["body"])
        self.assertIn("prediction", body)
        self.assertIn("risk_score", body)

    def test_direct_lambda_test_event(self):
        """Verify direct Lambda console test event (transaction dict directly passed)."""
        res = lambda_function.lambda_handler(self.valid_transaction)
        self.assertEqual(res["statusCode"], 200)
        body = json.loads(res["body"])
        self.assertIn("prediction", body)
        self.assertIn("risk_score", body)

    def test_invalid_numeric_amount_returns_400(self):
        """Verify invalid non-numeric amount returns HTTP 400."""
        invalid_txn = dict(self.valid_transaction)
        invalid_txn["Amount"] = "abc_invalid"

        event = {
            "requestContext": {"http": {"method": "POST"}},
            "body": json.dumps(invalid_txn)
        }
        res = lambda_function.lambda_handler(event)
        self.assertEqual(res["statusCode"], 400)
        body = json.loads(res["body"])
        self.assertEqual(body.get("error"), "Invalid numeric value")

    def test_malformed_json_returns_400(self):
        """Verify malformed JSON string returns HTTP 400."""
        event = {
            "requestContext": {"http": {"method": "POST"}},
            "body": "{ invalid json string }"
        }
        res = lambda_function.lambda_handler(event)
        self.assertEqual(res["statusCode"], 400)
        body = json.loads(res["body"])
        self.assertEqual(body.get("error"), "Bad Request")


if __name__ == "__main__":
    unittest.main()
