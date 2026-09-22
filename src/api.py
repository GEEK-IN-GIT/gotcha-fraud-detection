"""Production Flask API for serving fraud risk predictions.

Run with gunicorn using the app factory:
    gunicorn --factory "src.api:create_app" --bind 0.0.0.0:5001

Or directly for local development:
    python -m src.api
"""

import hmac
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, g, jsonify, request

from src.config import (
    get_feature_metadata_file,
    get_preprocessor_file,
    get_selected_threshold_file,
    get_tuned_test_metrics_file,
    get_xgboost_model_file,
)
from src.features import DERIVABLE_FIELDS, REQUIRED_FIELDS, derive_features
from src.score_transaction import APPROVE_BELOW, BLOCK_AT_OR_ABOVE, apply_decision_policy


logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 100
DEFAULT_RATE_LIMIT_PER_MIN = 120
DEFAULT_MAX_CONTENT_LENGTH = 1 * 1024 * 1024


class ScoreValidationError(Exception):
    """Raised when a transaction payload fails validation before scoring."""

    def __init__(self, message: str, missing_fields: list[str] | None = None, invalid_fields: dict[str, str] | None = None):
        super().__init__(message)
        self.message = message
        self.missing_fields = missing_fields
        self.invalid_fields = invalid_fields

    def to_payload(self) -> dict:
        payload: dict[str, object] = {"error": self.message}
        if self.missing_fields:
            payload["missing_fields"] = self.missing_fields
        if self.invalid_fields:
            payload["invalid_fields"] = self.invalid_fields
        return payload


class ModelArtifacts:
    """Loads and holds everything needed to score transactions for one model variant."""

    def __init__(self, variant: str):
        self.variant = variant
        self.preprocessor = joblib.load(get_preprocessor_file(variant))
        self.model = joblib.load(get_xgboost_model_file(variant))
        self.booster = self.model.get_booster()

        with get_feature_metadata_file(variant).open("r", encoding="utf-8") as handle:
            self.feature_metadata = json.load(handle)

        self.numeric_columns: list[str] = self.feature_metadata["numeric_columns"]
        self.categorical_columns: list[str] = self.feature_metadata["categorical_columns"]
        self.feature_columns: list[str] = self.numeric_columns + self.categorical_columns

        self.transformed_feature_names = self._build_transformed_feature_names()
        self.categorical_choices = self._build_categorical_choices()

    def _build_transformed_feature_names(self) -> list[str]:
        try:
            return list(self.preprocessor.get_feature_names_out())
        except Exception:
            return []

    def _build_categorical_choices(self) -> dict[str, list[str]]:
        encoder = self.preprocessor.named_transformers_["categorical"].named_steps["encoder"]
        return {
            column: [str(value) for value in categories]
            for column, categories in zip(self.categorical_columns, encoder.categories_)
        }


class RateLimiter:
    """Simple in-memory sliding-window rate limiter, keyed by caller API key."""

    def __init__(self, limit_per_min: int):
        self.limit_per_min = limit_per_min
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        window_start = now - 60
        hits = self._hits[key]
        while hits and hits[0] < window_start:
            hits.popleft()
        if len(hits) >= self.limit_per_min:
            return False
        hits.append(now)
        return True


def _load_metrics_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _raw_feature_name(transformed_name: str, categorical_columns: list[str]) -> str:
    """Map a transformed (post-ColumnTransformer) column name back to its raw feature."""
    if transformed_name.startswith("numeric__"):
        return transformed_name[len("numeric__"):]
    if transformed_name.startswith("categorical__"):
        remainder = transformed_name[len("categorical__"):]
        for column in categorical_columns:
            if remainder == column or remainder.startswith(column + "_"):
                return column
        return remainder
    return transformed_name


def _build_model_row(payload: dict, artifacts: ModelArtifacts) -> tuple[dict, list[str], list[str]]:
    """Derive, validate, and prepare one transaction row for scoring.

    Missing numeric/categorical fields are passed through as NaN so the
    fitted preprocessor's imputers fill them the same way they did at
    training time, instead of silently defaulting to 0/"unknown".

    Raises:
        ScoreValidationError: missing required fields or bad value types.
    """
    features, derived_fields = derive_features(payload)

    missing_required = sorted(
        column for column in REQUIRED_FIELDS if features.get(column) in (None, "")
    )
    if missing_required:
        raise ScoreValidationError("Missing required fields", missing_fields=missing_required)

    invalid_fields: dict[str, str] = {}
    row_data: dict[str, object] = {}
    imputed_fields: list[str] = []

    for column in artifacts.numeric_columns:
        value = features.get(column)
        if value in (None, ""):
            row_data[column] = np.nan
            imputed_fields.append(column)
            continue
        if isinstance(value, bool):
            invalid_fields[column] = "Expected numeric value, received boolean."
            continue
        try:
            row_data[column] = float(value)
        except (TypeError, ValueError):
            invalid_fields[column] = "Expected numeric value."

    for column in artifacts.categorical_columns:
        value = features.get(column)
        if value in (None, ""):
            row_data[column] = np.nan
            imputed_fields.append(column)
            continue
        if isinstance(value, (list, dict)):
            invalid_fields[column] = "Expected scalar categorical value."
            continue
        row_data[column] = str(value)

    if invalid_fields:
        raise ScoreValidationError("Invalid input", invalid_fields=invalid_fields)

    amt_value = row_data.get("amt")
    if isinstance(amt_value, float) and not np.isnan(amt_value) and amt_value < 0:
        raise ScoreValidationError("Invalid input", invalid_fields={"amt": "amt cannot be negative"})

    return row_data, derived_fields, imputed_fields


def _compute_top_factors(transformed_features, artifacts: ModelArtifacts, top_n: int = 5) -> list[dict]:
    if not artifacts.transformed_feature_names:
        return []

    matrix = transformed_features.tocsr() if hasattr(transformed_features, "tocsr") else transformed_features
    dmatrix = xgb.DMatrix(matrix)
    contributions = artifacts.booster.predict(dmatrix, pred_contribs=True)[0]

    raw_contributions: dict[str, float] = defaultdict(float)
    for name, value in zip(artifacts.transformed_feature_names, contributions[:-1]):
        raw_contributions[_raw_feature_name(name, artifacts.categorical_columns)] += float(value)

    ranked = sorted(raw_contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    return [{"feature": name, "contribution": round(value, 6)} for name, value in ranked[:top_n]]


def _score_row(row_data: dict, artifacts: ModelArtifacts) -> tuple[float, list[dict]]:
    input_frame = pd.DataFrame([row_data], columns=artifacts.feature_columns)
    transformed = artifacts.preprocessor.transform(input_frame)
    fraud_probability = float(artifacts.model.predict_proba(transformed)[0, 1])
    top_factors = _compute_top_factors(transformed, artifacts)
    return fraud_probability, top_factors


def _build_openapi_spec(artifacts: ModelArtifacts) -> dict:
    feature_properties: dict[str, dict] = {}
    required: list[str] = []

    for column in artifacts.numeric_columns:
        feature_properties[column] = {"type": "number"}
        if column in REQUIRED_FIELDS:
            required.append(column)

    for column in artifacts.categorical_columns:
        feature_properties[column] = {"type": "string", "enum": artifacts.categorical_choices.get(column)}
        if column in REQUIRED_FIELDS:
            required.append(column)

    feature_properties["timestamp"] = {"type": "string", "format": "date-time"}
    feature_properties["dob"] = {"type": "string", "format": "date"}

    transaction_schema = {"type": "object", "properties": feature_properties, "required": sorted(required)}

    score_response_schema = {
        "type": "object",
        "properties": {
            "fraud_probability": {"type": "number"},
            "decision": {"type": "string", "enum": ["approve", "step_up_verification", "block"]},
            "risk_band": {"type": "string", "enum": ["low_risk", "medium_risk", "high_risk"]},
            "action": {"type": "string"},
            "model_variant": {"type": "string"},
            "derived_fields": {"type": "array", "items": {"type": "string"}},
            "imputed_fields": {"type": "array", "items": {"type": "string"}},
            "request_id": {"type": "string"},
            "latency_ms": {"type": "number"},
            "top_factors": {"type": "array"},
        },
    }

    return {
        "openapi": "3.0.3",
        "info": {"title": "Gotcha Fraud Detection API", "version": "1.0.0"},
        "servers": [{"url": "/"}],
        "security": [{"ApiKeyAuth": []}],
        "components": {
            "securitySchemes": {"ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}},
            "schemas": {"TransactionInput": transaction_schema, "ScoreResponse": score_response_schema},
        },
        "paths": {
            "/v1/health": {"get": {"summary": "Health check", "security": [], "responses": {"200": {"description": "OK"}}}},
            "/v1/model": {
                "get": {"summary": "Model metadata and metrics", "security": [], "responses": {"200": {"description": "OK"}}}
            },
            "/v1/schema": {
                "get": {"summary": "Feature schema", "security": [], "responses": {"200": {"description": "OK"}}}
            },
            "/v1/score": {
                "post": {
                    "summary": "Score a single transaction",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/TransactionInput"}}},
                    },
                    "responses": {
                        "200": {
                            "description": "Fraud score",
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ScoreResponse"}}},
                        },
                        "400": {"description": "Validation error"},
                        "401": {"description": "Missing or invalid API key"},
                        "429": {"description": "Rate limit exceeded"},
                    },
                }
            },
            "/v1/score/batch": {
                "post": {
                    "summary": f"Score up to {MAX_BATCH_SIZE} transactions",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "transactions": {
                                            "type": "array",
                                            "maxItems": MAX_BATCH_SIZE,
                                            "items": {"$ref": "#/components/schemas/TransactionInput"},
                                        }
                                    },
                                }
                            }
                        },
                    },
                    "responses": {"200": {"description": "Batch scoring results"}},
                }
            },
        },
    }


def create_app() -> Flask:
    """Build and configure the Flask application.

    FRAUD_API_KEY is validated here, not at import time, so importing this
    module (e.g. for tests or tooling) never fails just because the
    environment isn't fully configured yet.
    """
    api_key = os.getenv("FRAUD_API_KEY")
    if not api_key:
        raise RuntimeError("FRAUD_API_KEY environment variable must be set before starting the API.")

    model_variant = os.getenv("MODEL_VARIANT", "full")
    allowed_origins = {origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()}
    rate_limit_per_min = int(os.getenv("RATE_LIMIT_PER_MIN", str(DEFAULT_RATE_LIMIT_PER_MIN)))
    max_content_length = int(os.getenv("MAX_CONTENT_LENGTH", str(DEFAULT_MAX_CONTENT_LENGTH)))

    artifacts = ModelArtifacts(model_variant)
    rate_limiter = RateLimiter(rate_limit_per_min)
    openapi_spec = _build_openapi_spec(artifacts)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_content_length

    def json_error(status_code: int, error_name: str, detail: str):
        payload = {"error": error_name, "detail": detail}
        request_id = getattr(g, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        return jsonify(payload), status_code

    def check_auth():
        provided = request.headers.get("X-API-Key")
        if provided is None:
            return json_error(401, "Unauthorized", "Provide X-API-Key header.")
        if not hmac.compare_digest(provided, api_key):
            return json_error(401, "Unauthorized", "The provided X-API-Key is not valid.")
        return None

    def enforce_rate_limit():
        if not rate_limiter.allow(api_key):
            return json_error(429, "Too Many Requests", f"Rate limit of {rate_limit_per_min} requests per minute exceeded.")
        return None

    @app.before_request
    def assign_request_id():
        g.request_id = str(uuid.uuid4())

    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get("Origin")
        if origin and origin in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key"
        return response

    def health() -> tuple:
        return jsonify(
            {
                "status": "ok",
                "model_variant": artifacts.variant,
                "threshold": BLOCK_AT_OR_ABOVE,
                "approve_below": APPROVE_BELOW,
                "block_at_or_above": BLOCK_AT_OR_ABOVE,
            }
        ), 200

    app.add_url_rule("/v1/health", endpoint="health_v1", view_func=health, methods=["GET"])
    app.add_url_rule("/health", endpoint="health_legacy", view_func=health, methods=["GET"])

    @app.get("/v1/model")
    def model_info():
        return jsonify(
            {
                "model_variant": artifacts.variant,
                "approve_below": APPROVE_BELOW,
                "block_at_or_above": BLOCK_AT_OR_ABOVE,
                "tuned_test_metrics": _load_metrics_file(get_tuned_test_metrics_file(artifacts.variant)),
                "selected_threshold": _load_metrics_file(get_selected_threshold_file(artifacts.variant)),
            }
        ), 200

    @app.get("/v1/schema")
    def schema():
        fields = []
        for column in artifacts.numeric_columns:
            fields.append(
                {
                    "name": column,
                    "type": "number",
                    "required": column in REQUIRED_FIELDS,
                    "derivable": column in DERIVABLE_FIELDS,
                    "allowed_values": None,
                }
            )
        for column in artifacts.categorical_columns:
            fields.append(
                {
                    "name": column,
                    "type": "string",
                    "required": column in REQUIRED_FIELDS,
                    "derivable": column in DERIVABLE_FIELDS,
                    "allowed_values": artifacts.categorical_choices.get(column),
                }
            )
        return jsonify(
            {
                "model_variant": artifacts.variant,
                "fields": fields,
                "derivation_inputs": {
                    "timestamp": (
                        "ISO 8601 transaction time. Used to derive txn_hour, txn_dayofweek, "
                        "and as the reference time for customer_age."
                    ),
                    "dob": "ISO 8601 date of birth. Used with timestamp to derive customer_age.",
                },
            }
        ), 200

    @app.get("/openapi.json")
    def openapi():
        return jsonify(openapi_spec), 200

    def score() -> tuple:
        start = time.monotonic()

        auth_error = check_auth()
        if auth_error is not None:
            return auth_error
        rate_limit_error = enforce_rate_limit()
        if rate_limit_error is not None:
            return rate_limit_error

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return json_error(400, "Bad Request", "Request body must be a JSON object.")

        try:
            row_data, derived_fields, imputed_fields = _build_model_row(payload, artifacts)
        except ScoreValidationError as exc:
            return jsonify(exc.to_payload()), 400

        fraud_probability, top_factors = _score_row(row_data, artifacts)
        decision_payload = apply_decision_policy(fraud_probability)

        response_payload = {
            "fraud_probability": round(fraud_probability, 4),
            "decision": decision_payload["decision"],
            "risk_band": decision_payload["risk_band"],
            "action": decision_payload["action"],
            "model_variant": artifacts.variant,
            "derived_fields": sorted(derived_fields),
            "imputed_fields": sorted(imputed_fields),
            "request_id": g.request_id,
            "latency_ms": round((time.monotonic() - start) * 1000, 2),
            "top_factors": top_factors,
        }
        return jsonify(response_payload), 200

    app.add_url_rule("/v1/score", endpoint="score_v1", view_func=score, methods=["POST"])
    app.add_url_rule("/score", endpoint="score_legacy", view_func=score, methods=["POST"])

    @app.post("/v1/score/batch")
    def score_batch():
        start = time.monotonic()

        auth_error = check_auth()
        if auth_error is not None:
            return auth_error
        rate_limit_error = enforce_rate_limit()
        if rate_limit_error is not None:
            return rate_limit_error

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("transactions"), list):
            return json_error(400, "Bad Request", "Request body must be a JSON object with a 'transactions' array.")

        transactions = payload["transactions"]
        if len(transactions) == 0:
            return json_error(400, "Bad Request", "The 'transactions' array must contain at least one item.")
        if len(transactions) > MAX_BATCH_SIZE:
            return json_error(400, "Bad Request", f"Batch size exceeds the maximum of {MAX_BATCH_SIZE} transactions.")

        results = []
        for index, item in enumerate(transactions):
            if not isinstance(item, dict):
                results.append({"index": index, "error": "Each transaction must be a JSON object."})
                continue
            try:
                row_data, derived_fields, imputed_fields = _build_model_row(item, artifacts)
            except ScoreValidationError as exc:
                error_payload = exc.to_payload()
                error_payload["index"] = index
                results.append(error_payload)
                continue

            fraud_probability, top_factors = _score_row(row_data, artifacts)
            decision_payload = apply_decision_policy(fraud_probability)
            results.append(
                {
                    "index": index,
                    "fraud_probability": round(fraud_probability, 4),
                    "decision": decision_payload["decision"],
                    "risk_band": decision_payload["risk_band"],
                    "action": decision_payload["action"],
                    "derived_fields": sorted(derived_fields),
                    "imputed_fields": sorted(imputed_fields),
                    "top_factors": top_factors,
                }
            )

        return jsonify(
            {
                "model_variant": artifacts.variant,
                "count": len(results),
                "request_id": g.request_id,
                "latency_ms": round((time.monotonic() - start) * 1000, 2),
                "results": results,
            }
        ), 200

    @app.errorhandler(400)
    def handle_bad_request(error):
        return json_error(400, "Bad Request", getattr(error, "description", None) or "The request could not be understood.")

    @app.errorhandler(401)
    def handle_unauthorized(error):
        return json_error(401, "Unauthorized", getattr(error, "description", None) or "Authentication required.")

    @app.errorhandler(404)
    def handle_not_found(error):
        return json_error(404, "Not Found", "The requested resource does not exist.")

    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        return json_error(405, "Method Not Allowed", "The HTTP method is not allowed for this endpoint.")

    @app.errorhandler(413)
    def handle_payload_too_large(error):
        return json_error(413, "Payload Too Large", "The request body exceeds the maximum allowed size.")

    @app.errorhandler(429)
    def handle_rate_limited(error):
        return json_error(429, "Too Many Requests", getattr(error, "description", None) or "Rate limit exceeded.")

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        logger.exception("Unhandled exception while processing request")
        return json_error(500, "Internal Server Error", "An unexpected error occurred.")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=False, host="0.0.0.0", port=5001)
