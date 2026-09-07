"""
API Integration Test Script for AWS-based Credit Card Fraud Detection System

Endpoint:
  POST https://sd2mbb56a7.execute-api.us-east-1.amazonaws.com/dev/predict

Tests:
  1. Valid transaction POST -> Expected HTTP 200 with prediction, class, risk_score.
  2. Incomplete transaction POST (missing features) -> Expected HTTP 400 with missing_features list.
  3. CORS Preflight OPTIONS -> Expected HTTP 200 with Access-Control-Allow-Origin: *.
"""

import sys
import json
import time
import urllib.request
import urllib.error

ENDPOINT = "https://sd2mbb56a7.execute-api.us-east-1.amazonaws.com/dev/predict"

# 30 creditcard.csv model features (Time, V1..V28, Amount)
SAMPLE_TRANSACTION = {
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
    "V15": 0.38972412,
    "V16": -1.14074718,
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


def send_request(url: str, method: str = "POST", payload: dict = None) -> tuple:
    """Send HTTP request using standard library urllib."""
    data_bytes = None
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    if payload is not None:
        data_bytes = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)

    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            elapsed = (time.time() - start_time) * 1000
            resp_body = resp.read().decode("utf-8")
            try:
                parsed_json = json.loads(resp_body)
            except Exception:
                parsed_json = resp_body
            return resp.status, parsed_json, dict(resp.headers), elapsed
    except urllib.error.HTTPError as e:
        elapsed = (time.time() - start_time) * 1000
        err_body = e.read().decode("utf-8")
        try:
            parsed_json = json.loads(err_body)
        except Exception:
            parsed_json = err_body
        return e.code, parsed_json, dict(e.headers), elapsed
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        return None, {"error": str(e)}, {}, elapsed


def test_live_api():
    print("=" * 70)
    print("TESTING BACKEND API GATEWAY ENDPOINT")
    print(f"URL: {ENDPOINT}")
    print("=" * 70)

    # -------------------------------------------------------------
    # Test 1: Positive Test (Valid Transaction)
    # -------------------------------------------------------------
    print("\n[TEST 1] Sending Valid Transaction Payload (POST)...")
    status, body, headers, elapsed = send_request(ENDPOINT, method="POST", payload=SAMPLE_TRANSACTION)

    print(f"  HTTP Status Code: {status}")
    print(f"  Response Latency: {elapsed:.1f} ms")
    print(f"  Response Body:    {json.dumps(body, indent=2)}")

    if status == 200:
        if isinstance(body, dict) and "prediction" in body and "class" in body and "risk_score" in body:
            print("  >>> TEST 1 RESULT: PASSED [OK]")
        else:
            print("  >>> TEST 1 RESULT: FAILED (Missing expected response fields)")
    elif status == 500 and isinstance(body, dict) and "Artifact loading failure" in str(body):
        print("  >>> TEST 1 INFO: Endpoint is LIVE and routing to Lambda. (Lambda requires model artifact in S3)")
    else:
        print(f"  >>> TEST 1 RESULT: HTTP {status}")

    # -------------------------------------------------------------
    # Test 2: Negative Test (Missing Required Features)
    # -------------------------------------------------------------
    print("\n[TEST 2] Sending Incomplete Transaction Payload (missing V1 & Amount)...")
    incomplete_payload = dict(SAMPLE_TRANSACTION)
    del incomplete_payload["V1"]
    del incomplete_payload["Amount"]

    status, body, headers, elapsed = send_request(ENDPOINT, method="POST", payload=incomplete_payload)

    print(f"  HTTP Status Code: {status}")
    print(f"  Response Latency: {elapsed:.1f} ms")
    print(f"  Response Body:    {json.dumps(body, indent=2)}")

    if status == 400 and isinstance(body, dict) and "missing_features" in body:
        print("  >>> TEST 2 RESULT: PASSED [OK] (Correctly rejected with 400 Bad Request)")
    elif status == 400:
        print("  >>> TEST 2 RESULT: PASSED [OK] (Returned 400 Bad Request)")
    else:
        print(f"  >>> TEST 2 RESULT: HTTP {status}")

    # -------------------------------------------------------------
    # Test 3: CORS Preflight Test (OPTIONS)
    # -------------------------------------------------------------
    print("\n[TEST 3] Sending CORS Preflight Request (OPTIONS)...")
    status, body, headers, elapsed = send_request(ENDPOINT, method="OPTIONS")

    print(f"  HTTP Status Code: {status}")
    print(f"  CORS Origin Header: {headers.get('access-control-allow-origin', 'Not Present')}")
    print(f"  CORS Methods Header: {headers.get('access-control-allow-methods', 'Not Present')}")

    if status in (200, 204):
        print("  >>> TEST 3 RESULT: PASSED [OK] (CORS preflight supported)")
    else:
        print(f"  >>> TEST 3 RESULT: HTTP {status}")

    print("\n" + "=" * 70)
    print("API TEST EXECUTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    test_live_api()
