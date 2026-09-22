import importlib

import pytest

import src.api as api_module
from src.api import create_app


VALID_TRANSACTION = {
    "amt": 125.50,
    "zip": 90210,
    "lat": 34.0522,
    "long": -118.2437,
    "city_pop": 500000,
    "merch_lat": 34.05,
    "merch_long": -118.25,
    "category": "grocery_pos",
    "state": "CA",
    "gender": "F",
    "timestamp": "2026-06-15T14:30:00",
}


def auth_headers(key: str = "test-key") -> dict:
    return {"X-API-Key": key}


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("FRAUD_API_KEY", "test-key")
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "3")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://example.com")
    application = create_app()
    application.config.update(TESTING=True)
    return application


@pytest.fixture
def client(app):
    return app.test_client()


# --- Import safety ---


def test_import_does_not_require_api_key(monkeypatch):
    # Importing the module must never fail just because the environment
    # isn't configured yet - only create_app() should check FRAUD_API_KEY.
    monkeypatch.delenv("FRAUD_API_KEY", raising=False)
    importlib.reload(api_module)


def test_create_app_requires_api_key(monkeypatch):
    monkeypatch.delenv("FRAUD_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        create_app()


# --- Auth ---


def test_score_requires_api_key(client):
    response = client.post("/v1/score", json=VALID_TRANSACTION)
    assert response.status_code == 401


def test_score_rejects_wrong_api_key(client):
    response = client.post("/v1/score", json=VALID_TRANSACTION, headers=auth_headers("wrong-key"))
    assert response.status_code == 401


def test_score_accepts_correct_api_key(client):
    response = client.post("/v1/score", json=VALID_TRANSACTION, headers=auth_headers())
    assert response.status_code == 200


# --- Validation ---


def test_score_missing_required_field_returns_400(client):
    payload = dict(VALID_TRANSACTION)
    del payload["amt"]
    response = client.post("/v1/score", json=payload, headers=auth_headers())
    assert response.status_code == 400
    assert "amt" in response.get_json()["missing_fields"]


def test_score_rejects_negative_amount(client):
    payload = dict(VALID_TRANSACTION, amt=-5)
    response = client.post("/v1/score", json=payload, headers=auth_headers())
    assert response.status_code == 400


def test_score_rejects_non_numeric_amount(client):
    payload = dict(VALID_TRANSACTION, amt="not-a-number")
    response = client.post("/v1/score", json=payload, headers=auth_headers())
    assert response.status_code == 400


def test_score_rejects_non_json_body(client):
    response = client.post("/v1/score", data="not json", content_type="text/plain", headers=auth_headers())
    assert response.status_code == 400


# --- Derivation ---


def test_score_derives_missing_fields(client):
    response = client.post("/v1/score", json=VALID_TRANSACTION, headers=auth_headers())
    body = response.get_json()
    for field in (
        "txn_hour",
        "txn_dayofweek",
        "is_weekend",
        "is_night_transaction",
        "amount_log",
        "is_high_amount",
        "is_very_high_amount",
        "merchant_customer_distance_km",
        "merchant_category_risk_tier",
        "merchant_category_risk_score",
    ):
        assert field in body["derived_fields"]


def test_score_caller_value_overrides_derivation(client):
    payload = dict(VALID_TRANSACTION, txn_hour=3)
    response = client.post("/v1/score", json=payload, headers=auth_headers())
    body = response.get_json()
    assert "txn_hour" not in body["derived_fields"]


def test_score_missing_history_fields_are_imputed(client):
    response = client.post("/v1/score", json=VALID_TRANSACTION, headers=auth_headers())
    body = response.get_json()
    assert "transactions_last_10min" in body["imputed_fields"]
    assert "synthetic_ip_risk_score" in body["imputed_fields"]


def test_score_response_contains_top_factors(client):
    response = client.post("/v1/score", json=VALID_TRANSACTION, headers=auth_headers())
    body = response.get_json()
    assert isinstance(body["top_factors"], list)
    assert 0 < len(body["top_factors"]) <= 5
    for factor in body["top_factors"]:
        assert "feature" in factor and "contribution" in factor


def test_score_response_contains_request_metadata(client):
    response = client.post("/v1/score", json=VALID_TRANSACTION, headers=auth_headers())
    body = response.get_json()
    assert body["request_id"]
    assert body["latency_ms"] >= 0


# --- Legacy aliases ---


def test_legacy_score_alias_matches_v1(client):
    response = client.post("/score", json=VALID_TRANSACTION, headers=auth_headers())
    assert response.status_code == 200
    assert "fraud_probability" in response.get_json()


def test_legacy_health_alias(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_v1_health(client):
    response = client.get("/v1/health")
    assert response.status_code == 200


# --- Model / schema / openapi ---


def test_model_endpoint(client):
    response = client.get("/v1/model")
    assert response.status_code == 200
    body = response.get_json()
    assert body["model_variant"] == "full"
    assert "approve_below" in body and "block_at_or_above" in body


def test_schema_endpoint_lists_categorical_choices(client):
    response = client.get("/v1/schema")
    body = response.get_json()
    fields_by_name = {field["name"]: field for field in body["fields"]}
    assert set(fields_by_name["gender"]["allowed_values"]) == {"F", "M"}
    assert fields_by_name["amt"]["required"] is True
    assert fields_by_name["txn_hour"]["derivable"] is True


def test_openapi_endpoint(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    body = response.get_json()
    assert body["openapi"].startswith("3.")
    assert "/v1/score" in body["paths"]


# --- Batch ---


def test_batch_scores_multiple_transactions(client):
    payload = {"transactions": [VALID_TRANSACTION, VALID_TRANSACTION]}
    response = client.post("/v1/score/batch", json=payload, headers=auth_headers())
    assert response.status_code == 200
    body = response.get_json()
    assert body["count"] == 2
    assert all("fraud_probability" in result for result in body["results"])


def test_batch_reports_per_item_errors_without_failing_whole_batch(client):
    bad_transaction = dict(VALID_TRANSACTION)
    del bad_transaction["amt"]
    payload = {"transactions": [VALID_TRANSACTION, bad_transaction]}
    response = client.post("/v1/score/batch", json=payload, headers=auth_headers())
    assert response.status_code == 200
    body = response.get_json()
    assert "fraud_probability" in body["results"][0]
    assert "error" in body["results"][1]


def test_batch_rejects_over_max_size(client):
    payload = {"transactions": [VALID_TRANSACTION] * 101}
    response = client.post("/v1/score/batch", json=payload, headers=auth_headers())
    assert response.status_code == 400


def test_batch_rejects_empty_list(client):
    response = client.post("/v1/score/batch", json={"transactions": []}, headers=auth_headers())
    assert response.status_code == 400


# --- CORS ---


def test_cors_header_present_for_allowed_origin(client):
    response = client.post(
        "/v1/score",
        json=VALID_TRANSACTION,
        headers={**auth_headers(), "Origin": "https://example.com"},
    )
    assert response.headers.get("Access-Control-Allow-Origin") == "https://example.com"


def test_cors_header_absent_for_disallowed_origin(client):
    response = client.post(
        "/v1/score",
        json=VALID_TRANSACTION,
        headers={**auth_headers(), "Origin": "https://evil.example"},
    )
    assert "Access-Control-Allow-Origin" not in response.headers


def test_cors_preflight_options_request(client):
    response = client.options(
        "/v1/score",
        headers={"Origin": "https://example.com", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 200
    assert response.headers.get("Access-Control-Allow-Origin") == "https://example.com"


# --- Rate limiting (fixture sets RATE_LIMIT_PER_MIN=3) ---


def test_rate_limit_blocks_after_threshold(client):
    for _ in range(3):
        response = client.post("/v1/score", json=VALID_TRANSACTION, headers=auth_headers())
        assert response.status_code == 200
    response = client.post("/v1/score", json=VALID_TRANSACTION, headers=auth_headers())
    assert response.status_code == 429


# --- Error handlers ---


def test_404_returns_json(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.get_json()["error"]


def test_405_returns_json(client):
    response = client.get("/v1/score")
    assert response.status_code == 405
    assert response.get_json()["error"]
