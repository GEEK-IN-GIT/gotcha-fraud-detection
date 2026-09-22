"""Serve fraud risk predictions through a small Flask API."""

import json
import logging
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from src.config import get_feature_metadata_file, get_preprocessor_file, get_xgboost_model_file
from src.score_transaction import APPROVE_BELOW, BLOCK_AT_OR_ABOVE, apply_decision_policy


logger = logging.getLogger(__name__)

MODEL_VARIANT = os.getenv("MODEL_VARIANT", "full")
API_KEY = os.getenv("FRAUD_API_KEY")

if not API_KEY:
    raise RuntimeError("FRAUD_API_KEY environment variable must be set before starting the API.")


def load_artifacts(variant: str) -> tuple[object, object, dict[str, object]]:
    """Load the preprocessor, model, and feature metadata for one variant."""
    try:
        preprocessor = joblib.load(get_preprocessor_file(variant))
        model = joblib.load(get_xgboost_model_file(variant))

        with get_feature_metadata_file(variant).open("r", encoding="utf-8") as metadata_file:
            feature_metadata = json.load(metadata_file)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model artifacts for variant '{variant}': {exc}") from exc

    return preprocessor, model, feature_metadata


PREPROCESSOR, MODEL, FEATURE_METADATA = load_artifacts(MODEL_VARIANT)
# Freeze the training-time column contract at startup so every request uses the same ordering.
NUMERIC_COLUMNS = FEATURE_METADATA["numeric_columns"]
CATEGORICAL_COLUMNS = FEATURE_METADATA["categorical_columns"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

app = Flask(__name__)


def require_api_key() -> tuple | None:
    """Validate the caller API key from the request headers."""
    provided_api_key = request.headers.get("X-API-Key")
    if provided_api_key is None:
        return jsonify({"error": "Missing API key", "detail": "Provide X-API-Key header."}), 401
    if provided_api_key != API_KEY:
        return jsonify({"error": "Invalid API key", "detail": "The provided X-API-Key is not valid."}), 401
    return None


def validate_request_payload(request_json: dict) -> tuple[list[str], dict[str, str]]:
    """Validate request field types before preprocessing and inference."""
    missing_fields = [column for column in FEATURE_COLUMNS if request_json.get(column) is None]
    invalid_fields: dict[str, str] = {}

    for column in NUMERIC_COLUMNS:
        value = request_json.get(column)
        if value is None:
            continue
        if isinstance(value, bool):
            invalid_fields[column] = "Expected numeric value, received boolean."
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            invalid_fields[column] = "Expected numeric value."

    for column in CATEGORICAL_COLUMNS:
        value = request_json.get(column)
        if value is None:
            continue
        if isinstance(value, (list, dict)):
            invalid_fields[column] = "Expected scalar categorical value."

    return missing_fields, invalid_fields


@app.get("/health")
def health() -> tuple:
    return jsonify(
        {
            "status": "ok",
            "model_variant": MODEL_VARIANT,
            "threshold": BLOCK_AT_OR_ABOVE,
            "approve_below": APPROVE_BELOW,
            "block_at_or_above": BLOCK_AT_OR_ABOVE,
        }
    ), 200


@app.post("/score")
def score() -> tuple:
    try:
        unauthorized_response = require_api_key()
        if unauthorized_response is not None:
            return unauthorized_response

        request_json = request.get_json(silent=True)
        if not isinstance(request_json, dict):
            return jsonify({"error": "Missing required fields", "missing_fields": FEATURE_COLUMNS}), 400

        missing_fields, invalid_fields = validate_request_payload(request_json)
        if missing_fields:
            logger.warning("Missing request fields filled with defaults: %s", missing_fields)
        if invalid_fields:
            return jsonify({"error": "Invalid input", "invalid_fields": invalid_fields}), 400

        row_data = {}
        for column in NUMERIC_COLUMNS:
            # Numeric defaults keep scoring tolerant of partially populated upstream events.
            value = request_json.get(column, 0)
            row_data[column] = 0.0 if value is None else float(value)

        for column in CATEGORICAL_COLUMNS:
            # Unknown-category fallback lets the shared encoder safely handle missing labels.
            value = request_json.get(column, "unknown")
            row_data[column] = "unknown" if value in (None, "") else str(value)

        if row_data.get("amt", 0) < 0:
            return jsonify({"error": "Invalid input", "detail": "amt cannot be negative"}), 400

        # Preserve training-time column order before applying the fitted preprocessor.
        input_frame = pd.DataFrame([row_data], columns=FEATURE_COLUMNS)
        transformed_features = PREPROCESSOR.transform(input_frame)
        fraud_probability = float(MODEL.predict_proba(transformed_features)[0, 1])

        decision_payload = apply_decision_policy(fraud_probability)
        response_payload = {
            "fraud_probability": round(fraud_probability, 4),
            "decision": decision_payload["decision"],
            "risk_band": decision_payload["risk_band"],
            "action": decision_payload["action"],
            "model_variant": MODEL_VARIANT,
        }
        return jsonify(response_payload), 200
    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed", "detail": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)
