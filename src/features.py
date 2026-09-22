"""Derive model-input features from a raw scoring request payload.

Mirrors the formulas in src/enrich_data.py so a single live transaction can
be scored without needing the full historical dataset. Fields that require
transaction history (velocity/aggregate features, synthetic risk signals)
have no formula here and are left for the caller or the fitted
preprocessor's imputers to fill.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.enrich_data import haversine_distance_km


NIGHT_HOURS = {22, 23, 0, 1, 2, 3, 4, 5}

HIGH_RISK_CATEGORIES = {"shopping_net", "misc_net", "grocery_pos"}
MEDIUM_RISK_CATEGORIES = {
    "shopping_pos",
    "misc_pos",
    "grocery_net",
    "food_dining",
    "entertainment",
    "travel",
    "gas_transport",
}
LOW_RISK_CATEGORIES = {"health_fitness", "kids_pets", "personal_care", "home"}

RISK_TIER_SCORES = {"high": 2, "medium": 1, "low": 0}

# Raw inputs with no derivation formula and no safe default - a request
# missing one of these is rejected rather than silently imputed.
REQUIRED_FIELDS = {
    "amt",
    "zip",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long",
    "category",
    "state",
    "gender",
}

# Fields derive_features() can compute from other request fields.
DERIVABLE_FIELDS = {
    "txn_hour",
    "txn_dayofweek",
    "is_weekend",
    "is_night_transaction",
    "customer_age",
    "amount_log",
    "is_high_amount",
    "is_very_high_amount",
    "merchant_customer_distance_km",
    "travel_speed_kmh",
    "merchant_category_risk_tier",
    "merchant_category_risk_score",
}


def _is_missing(value: object) -> bool:
    return value is None or value == ""


def _parse_iso_datetime(value: object) -> datetime | None:
    if _is_missing(value):
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _as_float(value: object) -> float | None:
    if _is_missing(value) or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def derive_features(payload: dict) -> tuple[dict, list[str]]:
    """Fill in derivable feature values the caller did not supply.

    Values the caller explicitly sends always win over derived ones. Fields
    that cannot be derived (missing formula inputs) are left absent.

    Args:
        payload: Raw request JSON body.

    Returns:
        (features, derived_fields): the enriched feature dict, and the list
        of field names that were filled in by this function.
    """
    features = dict(payload)
    derived_fields: list[str] = []

    txn_time = _parse_iso_datetime(features.get("timestamp"))

    if _is_missing(features.get("txn_hour")) and txn_time is not None:
        features["txn_hour"] = txn_time.hour
        derived_fields.append("txn_hour")

    if _is_missing(features.get("txn_dayofweek")) and txn_time is not None:
        features["txn_dayofweek"] = txn_time.weekday()
        derived_fields.append("txn_dayofweek")

    if _is_missing(features.get("is_weekend")) and not _is_missing(features.get("txn_dayofweek")):
        features["is_weekend"] = int(int(features["txn_dayofweek"]) in (5, 6))
        derived_fields.append("is_weekend")

    if _is_missing(features.get("is_night_transaction")) and not _is_missing(features.get("txn_hour")):
        features["is_night_transaction"] = int(int(features["txn_hour"]) in NIGHT_HOURS)
        derived_fields.append("is_night_transaction")

    if _is_missing(features.get("customer_age")):
        dob = _parse_iso_datetime(features.get("dob"))
        if dob is not None:
            reference_time = txn_time or datetime.now(timezone.utc)
            if dob.tzinfo is None:
                dob = dob.replace(tzinfo=timezone.utc)
            if reference_time.tzinfo is None:
                reference_time = reference_time.replace(tzinfo=timezone.utc)
            age_days = (reference_time - dob).days
            features["customer_age"] = float(np.floor(age_days / 365.25))
            derived_fields.append("customer_age")

    amt = _as_float(features.get("amt"))
    if amt is not None and amt >= 0:
        if _is_missing(features.get("amount_log")):
            features["amount_log"] = float(np.log1p(amt))
            derived_fields.append("amount_log")
        if _is_missing(features.get("is_high_amount")):
            features["is_high_amount"] = int(amt > 200)
            derived_fields.append("is_high_amount")
        if _is_missing(features.get("is_very_high_amount")):
            features["is_very_high_amount"] = int(amt > 500)
            derived_fields.append("is_very_high_amount")

    if _is_missing(features.get("merchant_customer_distance_km")):
        lat = _as_float(features.get("lat"))
        lon = _as_float(features.get("long"))
        merch_lat = _as_float(features.get("merch_lat"))
        merch_long = _as_float(features.get("merch_long"))
        if None not in (lat, lon, merch_lat, merch_long):
            distance = haversine_distance_km(
                pd.Series([lat]), pd.Series([lon]), pd.Series([merch_lat]), pd.Series([merch_long])
            )
            features["merchant_customer_distance_km"] = float(distance.iloc[0])
            derived_fields.append("merchant_customer_distance_km")

    if _is_missing(features.get("travel_speed_kmh")):
        distance = _as_float(features.get("distance_from_last_txn_km"))
        elapsed_seconds = _as_float(features.get("time_since_last_txn_sec"))
        if distance is not None and elapsed_seconds is not None and elapsed_seconds > 0:
            speed_kmh = distance / (elapsed_seconds / 3600.0)
            features["travel_speed_kmh"] = float(min(speed_kmh, 50000))
            derived_fields.append("travel_speed_kmh")

    category = features.get("category")
    if not _is_missing(category):
        category_name = str(category).lower()
        if _is_missing(features.get("merchant_category_risk_tier")):
            if category_name in HIGH_RISK_CATEGORIES:
                tier = "high"
            elif category_name in LOW_RISK_CATEGORIES:
                tier = "low"
            else:
                tier = "medium"
            features["merchant_category_risk_tier"] = tier
            derived_fields.append("merchant_category_risk_tier")

        if _is_missing(features.get("merchant_category_risk_score")):
            tier = features["merchant_category_risk_tier"]
            features["merchant_category_risk_score"] = RISK_TIER_SCORES.get(tier, 1)
            derived_fields.append("merchant_category_risk_score")

    return features, derived_fields
