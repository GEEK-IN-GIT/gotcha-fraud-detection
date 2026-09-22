"""Create enriched fraud features from the development sample dataset."""

import numpy as np
import pandas as pd

from src.config import DEV_SAMPLE_FILE, ENRICHED_SAMPLE_FILE, PROCESSED_DIR, RANDOM_SEED, TARGET_COL


def haversine_distance_km(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Return great-circle distance in kilometers for two coordinate pairs.

    This method uses the haversine formula to compute the distance between
    two latitude and longitude pairs.

    Args:
        lat1 (pd.Series): Starting latitude values.
        lon1 (pd.Series): Starting longitude values.
        lat2 (pd.Series): Ending latitude values.
        lon2 (pd.Series): Ending longitude values.

    Returns:
        pd.Series: Distance between each coordinate pair in kilometers.
    """
    earth_radius_km = 6371.0

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    haversine_term = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    )
    central_angle = 2.0 * np.arctan2(np.sqrt(haversine_term), np.sqrt(1.0 - haversine_term))
    return earth_radius_km * central_angle


def add_synthetic_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add seeded operational risk indicators with mild fraud correlation.

    This method creates synthetic operational fraud-risk signals that are
    reproducible and only mildly correlated with the fraud label.

    Args:
        df (pd.DataFrame): Input dataframe containing the base transaction data.

    Returns:
        pd.DataFrame: Dataframe with additional synthetic fraud-risk columns.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    amount_signal = df["amount_vs_customer_mean"].rank(pct=True).fillna(0.5)
    speed_signal = df["travel_speed_kmh"].rank(pct=True).fillna(0.5)
    burst_signal = df["transactions_last_10min"].rank(pct=True).fillna(0.5)
    distance_signal = df["distance_from_last_txn_km"].rank(pct=True).fillna(0.5)
    night_signal = df["is_night_transaction"].fillna(0).astype(float)

    suspicion_score = (
        0.28 * amount_signal
        + 0.24 * speed_signal
        + 0.18 * burst_signal
        + 0.18 * distance_signal
        + 0.12 * night_signal
    )
    suspicion_score = np.clip(suspicion_score.to_numpy(dtype=float), 0.0, 1.0)

    adjusted_suspicion = suspicion_score

    # Binary device changes rise with suspicious transaction context, not the label itself.
    device_change_prob = np.clip(0.08 + 0.30 * adjusted_suspicion, 0.05, 0.75)
    df["synthetic_device_changed"] = rng.binomial(1, device_change_prob)

    # Failed logins increase under bursty or unusual behavior, but remain noisy.
    failed_login_lambda = np.clip(0.5 + 2.8 * adjusted_suspicion, 0.1, 6.0)
    df["synthetic_failed_logins_24h"] = np.clip(rng.poisson(failed_login_lambda), 0, 12)

    # IP risk tracks suspicious context with enough variance to keep overlap realistic.
    ip_risk_center = 18 + 58 * adjusted_suspicion
    ip_risk_score = rng.normal(loc=ip_risk_center, scale=14)
    df["synthetic_ip_risk_score"] = np.clip(np.round(ip_risk_score, 2), 0, 100)

    # Suspicious traffic leans slightly newer, but with broad overlap across the population.
    account_age_days = rng.normal(loc=980 - 420 * adjusted_suspicion, scale=320)
    df["synthetic_account_age_days"] = np.clip(np.round(account_age_days), 1, 3650).astype(int)

    email_age_days = rng.normal(loc=760 - 300 * adjusted_suspicion, scale=260)
    df["synthetic_email_age_days"] = np.clip(np.round(email_age_days), 1, 3650).astype(int)

    mismatch_prob = np.clip(0.06 + 0.24 * adjusted_suspicion, 0.03, 0.65)
    df["synthetic_billing_shipping_mismatch"] = rng.binomial(1, mismatch_prob)

    return df


def count_prior_transactions_within_window(transaction_times: pd.Series, window_seconds: int) -> pd.Series:
    """Count prior transactions that fall inside a trailing time window.

    This method counts how many earlier transactions occur within a fixed
    trailing window for each timestamp in an already sorted series.

    Args:
        transaction_times (pd.Series): Sorted transaction timestamps for one card.
        window_seconds (int): Size of the trailing window in seconds.

    Returns:
        pd.Series: Count of prior transactions inside the trailing window.
    """
    transaction_timestamps = transaction_times.astype("datetime64[s]").astype("int64")
    window_start_indices = np.searchsorted(
        transaction_timestamps,
        transaction_timestamps - window_seconds,
        side="left",
    )
    prior_transaction_counts = np.arange(len(transaction_timestamps)) - window_start_indices
    return pd.Series(prior_transaction_counts, index=transaction_times.index)


def sum_prior_amounts_within_window(
    transaction_times: pd.Series,
    transaction_amounts: pd.Series,
    window_seconds: int,
) -> pd.Series:
    """Sum prior transaction amounts inside a trailing time window.

    Args:
        transaction_times (pd.Series): Sorted transaction timestamps for one card.
        transaction_amounts (pd.Series): Sorted transaction amounts for one card.
        window_seconds (int): Size of the trailing window in seconds.

    Returns:
        pd.Series: Sum of prior amounts inside the trailing window.
    """
    transaction_timestamps = transaction_times.astype("datetime64[s]").astype("int64")
    window_start_indices = np.searchsorted(
        transaction_timestamps,
        transaction_timestamps - window_seconds,
        side="left",
    )
    cumulative_amounts = np.cumsum(transaction_amounts.to_numpy(dtype=float))
    # Shift the cumulative totals so the current amount is excluded from its own trailing sum.
    prior_cumulative_amounts = np.concatenate(([0.0], cumulative_amounts[:-1]))
    window_amount_starts = np.where(
        window_start_indices > 0,
        cumulative_amounts[window_start_indices - 1],
        0.0,
    )
    prior_window_amounts = prior_cumulative_amounts - window_amount_starts
    return pd.Series(prior_window_amounts, index=transaction_times.index)


def count_distinct_prior_categories_within_window(
    transaction_times: pd.Series,
    categories: pd.Series,
    window_seconds: int,
) -> pd.Series:
    """Count distinct prior categories inside a trailing time window.

    Args:
        transaction_times (pd.Series): Sorted transaction timestamps for one card.
        categories (pd.Series): Sorted transaction categories for one card.
        window_seconds (int): Size of the trailing window in seconds.

    Returns:
        pd.Series: Count of distinct prior categories inside the trailing window.
    """
    transaction_timestamps = transaction_times.astype("datetime64[s]").astype("int64").to_numpy()
    category_values = categories.fillna("unknown").to_numpy()
    distinct_counts = np.zeros(len(transaction_timestamps), dtype=int)

    window_start = 0
    category_counts: dict[str, int] = {}

    for current_index, current_timestamp in enumerate(transaction_timestamps):
        # Remove expired rows before counting the current window so only prior in-window categories remain.
        while transaction_timestamps[window_start] < current_timestamp - window_seconds:
            expired_category = category_values[window_start]
            category_counts[expired_category] -= 1
            if category_counts[expired_category] == 0:
                del category_counts[expired_category]
            window_start += 1

        distinct_counts[current_index] = len(category_counts)
        current_category = category_values[current_index]
        category_counts[current_category] = category_counts.get(current_category, 0) + 1

    return pd.Series(distinct_counts, index=transaction_times.index)


def main() -> None:
    """Build derived fraud features and save the enriched sample dataset.

    This method loads the development sample, creates derived fraud features,
    writes the enriched dataset to the processed-data directory, and prints
    a short summary of the generated output.

    Args:
        None

    Returns:
        None
    """
    df = pd.read_csv(DEV_SAMPLE_FILE)
    new_columns: list[str] = []

    # Parse timestamps once so the downstream feature logic stays vectorized.
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    # Basic temporal signals often separate routine and unusual transaction patterns.
    df["txn_hour"] = df["trans_date_trans_time"].dt.hour
    df["txn_dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
    df["is_weekend"] = df["txn_dayofweek"].isin([5, 6]).astype(int)
    df["is_night_transaction"] = df["txn_hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    new_columns.extend(["txn_hour", "txn_dayofweek", "is_weekend", "is_night_transaction"])

    # Age is measured at transaction time, not using the current date.
    age_in_days = (df["trans_date_trans_time"] - df["dob"]).dt.days
    df["customer_age"] = np.floor(age_in_days / 365.25)
    new_columns.append("customer_age")

    # Distance from cardholder to merchant helps capture anomalous purchase context.
    df["merchant_customer_distance_km"] = haversine_distance_km(
        df["lat"],
        df["long"],
        df["merch_lat"],
        df["merch_long"],
    )
    new_columns.append("merchant_customer_distance_km")

    # Merchant history is built in global time order so only prior merchant behavior is used.
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    merchant_groups = df.groupby("merchant", sort=False)
    # cumcount gives the number of earlier transactions for each merchant without looking forward.
    df["merchant_prior_txn_count"] = merchant_groups.cumcount()
    df["merchant_prior_fraud_count"] = merchant_groups[TARGET_COL].transform(
        # shift(fill_value=0) removes the current label before the cumulative fraud count is built.
        lambda fraud_values: fraud_values.shift(fill_value=0).cumsum()
    )
    df["merchant_prior_fraud_rate"] = np.where(
        df["merchant_prior_txn_count"] > 0,
        df["merchant_prior_fraud_count"] / df["merchant_prior_txn_count"],
        0.0,
    )
    new_columns.extend(
        [
            "merchant_prior_txn_count",
            "merchant_prior_fraud_count",
            "merchant_prior_fraud_rate",
        ]
    )

    # Historical features must be computed in customer-time order.
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)

    customer_groups = df.groupby("cc_num", sort=False)
    historical_mean = customer_groups["amt"].transform(lambda s: s.shift().expanding().mean())
    historical_std = customer_groups["amt"].transform(lambda s: s.shift().expanding().std())

    # Use only prior transactions so the current row does not leak into its own history.
    df["amount_log"] = np.log1p(df["amt"])
    # Explicit flags help the model capture sharp fraud-rate jumps at higher transaction amounts.
    df["is_high_amount"] = (df["amt"] > 200).astype(int)
    df["is_very_high_amount"] = (df["amt"] > 500).astype(int)
    df["amount_vs_customer_mean"] = np.where(
        historical_mean.notna() & historical_mean.ne(0),
        df["amt"] / historical_mean,
        1.0,
    )
    df["amount_zscore_customer"] = np.where(
        historical_std.notna() & historical_std.gt(0),
        (df["amt"] - historical_mean) / historical_std,
        0.0,
    )
    new_columns.extend(
        [
            "amount_log",
            "is_high_amount",
            "is_very_high_amount",
            "amount_vs_customer_mean",
            "amount_zscore_customer",
        ]
    )

    # Previous transaction context supports velocity, travel, and burstiness features.
    df["prev_txn_time"] = customer_groups["trans_date_trans_time"].shift(1)
    df["prev_lat"] = customer_groups["lat"].shift(1)
    df["prev_long"] = customer_groups["long"].shift(1)
    df["time_since_last_txn_sec"] = (
        df["trans_date_trans_time"] - df["prev_txn_time"]
    ).dt.total_seconds()
    df["distance_from_last_txn_km"] = haversine_distance_km(
        df["prev_lat"],
        df["prev_long"],
        df["lat"],
        df["long"],
    )
    # Invalid or missing gaps stay null so we avoid impossible speeds.
    valid_time_gap = df["time_since_last_txn_sec"].notna() & df["time_since_last_txn_sec"].gt(0)
    df["travel_speed_kmh"] = np.nan
    df.loc[valid_time_gap, "travel_speed_kmh"] = (
        df.loc[valid_time_gap, "distance_from_last_txn_km"]
        / (df.loc[valid_time_gap, "time_since_last_txn_sec"] / 3600.0)
    )
    df["travel_speed_kmh"] = df["travel_speed_kmh"].clip(upper=50000)
    # Count only earlier transactions from the same card in the last 10 minutes.
    df["transactions_last_10min"] = customer_groups["trans_date_trans_time"].transform(
        lambda timestamps: count_prior_transactions_within_window(timestamps, window_seconds=600)
    )
    df["transactions_last_1h"] = customer_groups["trans_date_trans_time"].transform(
        lambda timestamps: count_prior_transactions_within_window(timestamps, window_seconds=3600)
    )
    df["transactions_last_6h"] = customer_groups["trans_date_trans_time"].transform(
        lambda timestamps: count_prior_transactions_within_window(timestamps, window_seconds=21600)
    )
    df["transactions_last_24h"] = customer_groups["trans_date_trans_time"].transform(
        lambda timestamps: count_prior_transactions_within_window(timestamps, window_seconds=86400)
    )
    df["amount_spent_last_1h"] = customer_groups.apply(
        lambda group: sum_prior_amounts_within_window(
            group["trans_date_trans_time"],
            group["amt"],
            window_seconds=3600,
        )
    ).reset_index(level=0, drop=True)
    df["amount_spent_last_6h"] = customer_groups.apply(
        lambda group: sum_prior_amounts_within_window(
            group["trans_date_trans_time"],
            group["amt"],
            window_seconds=21600,
        )
    ).reset_index(level=0, drop=True)
    df["amount_spent_last_24h"] = customer_groups.apply(
        lambda group: sum_prior_amounts_within_window(
            group["trans_date_trans_time"],
            group["amt"],
            window_seconds=86400,
        )
    ).reset_index(level=0, drop=True)
    df["distinct_categories_last_24h"] = customer_groups.apply(
        lambda group: count_distinct_prior_categories_within_window(
            group["trans_date_trans_time"],
            group["category"],
            window_seconds=86400,
        )
    ).reset_index(level=0, drop=True)

    category_names = df["category"].fillna("").str.lower()
    high_risk_categories = {"shopping_net", "misc_net", "grocery_pos"}
    medium_risk_categories = {
        "shopping_pos",
        "misc_pos",
        "grocery_net",
        "food_dining",
        "entertainment",
        "travel",
        "gas_transport",
    }
    low_risk_categories = {
        "health_fitness",
        "kids_pets",
        "personal_care",
        "home",
    }
    high_risk_mask = category_names.isin(high_risk_categories)
    medium_risk_mask = category_names.isin(medium_risk_categories)
    low_risk_mask = category_names.isin(low_risk_categories)
    df["merchant_category_risk_tier"] = np.select(
        [high_risk_mask, low_risk_mask, medium_risk_mask],
        ["high", "low", "medium"],
        default="medium",
    )
    df["merchant_category_risk_score"] = np.select(
        [df["merchant_category_risk_tier"].eq("high"), df["merchant_category_risk_tier"].eq("medium")],
        [2, 1],
        default=0,
    )
    new_columns.extend(
        [
            "prev_txn_time",
            "prev_lat",
            "prev_long",
            "time_since_last_txn_sec",
            "distance_from_last_txn_km",
            "travel_speed_kmh",
            "transactions_last_10min",
            "transactions_last_1h",
            "transactions_last_6h",
            "transactions_last_24h",
            "amount_spent_last_1h",
            "amount_spent_last_6h",
            "amount_spent_last_24h",
            "distinct_categories_last_24h",
            "merchant_category_risk_tier",
            "merchant_category_risk_score",
        ]
    )

    df = add_synthetic_risk_features(df)
    new_columns.extend(
        [
            "synthetic_device_changed",
            "synthetic_failed_logins_24h",
            "synthetic_ip_risk_score",
            "synthetic_account_age_days",
            "synthetic_email_age_days",
            "synthetic_billing_shipping_mismatch",
        ]
    )

    # Helper columns are dropped after the final features have been derived.
    df = df.drop(columns=["prev_txn_time", "prev_lat", "prev_long"])
    new_columns = [column for column in new_columns if column not in {"prev_txn_time", "prev_lat", "prev_long"}]

    # Writing under data/processed keeps feature engineering outputs separate from raw inputs.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ENRICHED_SAMPLE_FILE, index=False)

    print(f"Output shape: {df.shape}")
    print(f"Newly added columns: {new_columns}")
    print(f"Output file path: {ENRICHED_SAMPLE_FILE}")


if __name__ == "__main__":
    main()
