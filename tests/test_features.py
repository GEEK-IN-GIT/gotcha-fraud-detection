import numpy as np
import pandas as pd

from src.enrich_data import (
    count_distinct_prior_categories_within_window,
    count_prior_transactions_within_window,
    haversine_distance_km,
    sum_prior_amounts_within_window,
)
from src.score_transaction import apply_decision_policy, score_band_summary


def test_haversine_same_location():
    # Identical coordinates should produce zero distance.
    lat1 = pd.Series([40.7128])
    lon1 = pd.Series([-74.0060])
    lat2 = pd.Series([40.7128])
    lon2 = pd.Series([-74.0060])

    distance = haversine_distance_km(lat1, lon1, lat2, lon2)

    assert distance.iloc[0] == 0.0


def test_haversine_known_distance_nyc_to_london():
    # Use a real-world city pair to sanity-check the haversine implementation.
    lat1 = pd.Series([40.7128])
    lon1 = pd.Series([-74.0060])
    lat2 = pd.Series([51.5074])
    lon2 = pd.Series([-0.1278])

    distance = haversine_distance_km(lat1, lon1, lat2, lon2)

    assert 5500 <= distance.iloc[0] <= 5600


def test_haversine_known_distance_la_to_chicago():
    # A second known route helps confirm the formula across different regions.
    lat1 = pd.Series([34.0522])
    lon1 = pd.Series([-118.2437])
    lat2 = pd.Series([41.8781])
    lon2 = pd.Series([-87.6298])

    distance = haversine_distance_km(lat1, lon1, lat2, lon2)

    assert 2800 <= distance.iloc[0] <= 3000


def test_haversine_returns_series():
    # The helper should preserve vectorized behavior and return one distance per row.
    lat1 = pd.Series([0.0, 10.0, 20.0])
    lon1 = pd.Series([0.0, 10.0, 20.0])
    lat2 = pd.Series([1.0, 11.0, 21.0])
    lon2 = pd.Series([1.0, 11.0, 21.0])

    distance = haversine_distance_km(lat1, lon1, lat2, lon2)

    assert isinstance(distance, pd.Series)
    assert len(distance) == 3


def test_haversine_all_nonnegative():
    # Great-circle distances should never be negative for any valid coordinates.
    lat1 = pd.Series([0.0, 15.0, -22.0, 48.0, -10.0])
    lon1 = pd.Series([0.0, 45.0, 80.0, -122.0, 130.0])
    lat2 = pd.Series([5.0, 20.0, -10.0, 50.0, -12.0])
    lon2 = pd.Series([5.0, 50.0, 90.0, -120.0, 120.0])

    distance = haversine_distance_km(lat1, lon1, lat2, lon2)

    assert (distance >= 0).all()


def test_count_first_row_is_zero():
    # The first transaction has no prior history inside any trailing window.
    timestamps = pd.Series(pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:05:00", "2026-01-01 00:10:00"]))

    counts = count_prior_transactions_within_window(timestamps, window_seconds=600)

    assert counts.iloc[0] == 0


def test_count_within_window():
    # Closely spaced transactions should accumulate prior counts within the window.
    timestamps = pd.Series(pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:05:00", "2026-01-01 00:10:00"]))

    counts = count_prior_transactions_within_window(timestamps, window_seconds=600)

    assert counts.tolist() == [0, 1, 2]


def test_count_outside_window():
    # Widely separated transactions should not count earlier rows once they are outside the window.
    timestamps = pd.Series(pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:25:00", "2026-01-01 00:50:00"]))

    counts = count_prior_transactions_within_window(timestamps, window_seconds=600)

    assert counts.tolist() == [0, 0, 0]


def test_count_mixed_window():
    # Mix in-window and out-of-window gaps to confirm the rolling count resets correctly.
    timestamps = pd.Series(pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:05:00", "2026-01-01 00:20:00", "2026-01-01 00:35:00"]))

    counts = count_prior_transactions_within_window(timestamps, window_seconds=600)

    assert counts.tolist() == [0, 1, 0, 0]


def test_count_single_transaction():
    # A one-row series should still produce a valid zero count.
    timestamps = pd.Series(pd.to_datetime(["2026-01-01 00:00:00"]))

    counts = count_prior_transactions_within_window(timestamps, window_seconds=600)

    assert counts.tolist() == [0]


def test_count_preserves_index():
    # The returned series should align back to the original row labels.
    timestamps = pd.Series(
        pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:05:00"]),
        index=[10, 20],
    )

    counts = count_prior_transactions_within_window(timestamps, window_seconds=600)

    assert counts.index.tolist() == [10, 20]


def test_sum_first_row_is_zero():
    # The first row has no prior spend, so its trailing sum should be zero.
    timestamps = pd.Series(pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:10:00", "2026-01-01 00:20:00"]))
    amounts = pd.Series([100.0, 200.0, 300.0])

    prior_amounts = sum_prior_amounts_within_window(timestamps, amounts, window_seconds=3600)

    assert prior_amounts.iloc[0] == 0.0


def test_sum_within_window():
    # Prior amounts inside the window should accumulate while excluding the current row.
    timestamps = pd.Series(pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:10:00", "2026-01-01 00:20:00"]))
    amounts = pd.Series([100.0, 200.0, 300.0])

    prior_amounts = sum_prior_amounts_within_window(timestamps, amounts, window_seconds=3600)

    assert prior_amounts.tolist() == [0.0, 100.0, 300.0]


def test_sum_outside_window():
    # Amounts from rows strictly outside the window should not contribute to the trailing sum.
    timestamps = pd.Series(pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 02:00:01", "2026-01-01 04:00:02"]))
    amounts = pd.Series([100.0, 200.0, 300.0])

    prior_amounts = sum_prior_amounts_within_window(timestamps, amounts, window_seconds=3600)

    assert prior_amounts.tolist() == [0.0, 0.0, 0.0]


def test_sum_partial_window():
    # Boundary handling should match the production helper's inclusive window logic.
    timestamps = pd.Series(pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:05:00", "2026-01-01 00:10:00", "2026-01-01 01:10:00"]))
    amounts = pd.Series([50.0, 75.0, 100.0, 200.0])

    prior_amounts = sum_prior_amounts_within_window(timestamps, amounts, window_seconds=3600)

    # Row 3 is at 01:10:00. ts[3] - 3600s = 00:10:00.
    # searchsorted side="left" finds index 2 (ts[2] == 00:10:00).
    # Only row 2 (amount=100.0) falls inside the window.
    # Rows 0 and 1 are 70 and 65 min prior — both outside.
    # Prior sum = 100.0
    assert prior_amounts.tolist() == [0.0, 50.0, 125.0, 100.0]


def test_sum_excludes_current_row():
    # The current transaction amount must never be included in its own trailing total.
    timestamps = pd.Series(pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:01:00", "2026-01-01 00:02:00"]))
    amounts = pd.Series([500.0, 500.0, 500.0])

    prior_amounts = sum_prior_amounts_within_window(timestamps, amounts, window_seconds=3600)

    assert prior_amounts.iloc[2] == 1000.0


def test_sum_single_transaction():
    # A single transaction should still yield a valid zero trailing spend.
    timestamps = pd.Series(pd.to_datetime(["2026-01-01 00:00:00"]))
    amounts = pd.Series([100.0])

    prior_amounts = sum_prior_amounts_within_window(timestamps, amounts, window_seconds=3600)

    assert prior_amounts.tolist() == [0.0]


def test_distinct_first_row_is_zero():
    # The first row has no prior categories to count.
    timestamps = pd.Series(pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:05:00", "2026-01-01 00:10:00"]))
    categories = pd.Series(["A", "B", "C"])

    distinct_counts = count_distinct_prior_categories_within_window(timestamps, categories, window_seconds=3600)

    assert distinct_counts.iloc[0] == 0


def test_distinct_counts_correctly():
    # Repeated categories should not inflate the count of distinct prior categories.
    timestamps = pd.Series(pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:05:00", "2026-01-01 00:10:00", "2026-01-01 00:15:00"]))
    categories = pd.Series(["A", "B", "A", "C"])

    distinct_counts = count_distinct_prior_categories_within_window(timestamps, categories, window_seconds=3600)

    assert distinct_counts.tolist() == [0, 1, 2, 2]


def test_distinct_outside_window():
    # Categories from rows outside the trailing window should be dropped from the distinct count.
    timestamps = pd.Series(pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 02:00:01", "2026-01-01 04:00:02"]))
    categories = pd.Series(["A", "B", "C"])

    distinct_counts = count_distinct_prior_categories_within_window(timestamps, categories, window_seconds=3600)

    assert distinct_counts.tolist() == [0, 0, 0]


def test_distinct_same_category_repeated():
    # Multiple prior rows with the same category should still count as one distinct category.
    timestamps = pd.Series(pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:05:00", "2026-01-01 00:10:00"]))
    categories = pd.Series(["A", "A", "A"])

    distinct_counts = count_distinct_prior_categories_within_window(timestamps, categories, window_seconds=3600)

    assert distinct_counts.tolist() == [0, 1, 1]


def test_distinct_single_transaction():
    # A one-row input should produce a zero distinct-count result.
    timestamps = pd.Series(pd.to_datetime(["2026-01-01 00:00:00"]))
    categories = pd.Series(["A"])

    distinct_counts = count_distinct_prior_categories_within_window(timestamps, categories, window_seconds=3600)

    assert distinct_counts.tolist() == [0]


def test_approve_low_probability():
    # Scores in the low band should auto-approve without added friction.
    result = apply_decision_policy(0.15)

    assert result["decision"] == "approve"
    assert result["risk_band"] == "low_risk"


def test_step_up_mid_probability():
    # Mid-band scores should route to step-up verification instead of approve/block.
    result = apply_decision_policy(0.55)

    assert result["decision"] == "step_up_verification"
    assert result["risk_band"] == "medium_risk"


def test_block_high_probability():
    # High-risk scores should be blocked outright.
    result = apply_decision_policy(0.85)

    assert result["decision"] == "block"
    assert result["risk_band"] == "high_risk"


def test_boundary_just_below_approve():
    # Values just below the lower threshold should remain in the approve band.
    result = apply_decision_policy(0.399)

    assert result["decision"] == "approve"


def test_boundary_exactly_approve_threshold():
    # The lower boundary is inclusive for the medium-risk band.
    result = apply_decision_policy(0.40)

    assert result["decision"] == "step_up_verification"


def test_boundary_just_below_block():
    # Values just below the block threshold should still require step-up verification.
    result = apply_decision_policy(0.749)

    assert result["decision"] == "step_up_verification"


def test_boundary_exactly_block_threshold():
    # The upper boundary is inclusive for the high-risk block band.
    result = apply_decision_policy(0.75)

    assert result["decision"] == "block"


def test_probability_zero():
    # The smallest possible score should still map cleanly to approve.
    result = apply_decision_policy(0.0)

    assert result["decision"] == "approve"


def test_probability_one():
    # The largest possible score should always map to block.
    result = apply_decision_policy(1.0)

    assert result["decision"] == "block"


def test_output_contains_required_keys():
    # The policy helper should always return the full response contract used by the API.
    result = apply_decision_policy(0.55)

    assert set(result.keys()) == {"decision", "risk_band", "action", "probability"}


def test_probability_rounded_in_output():
    # Output probabilities should be rounded consistently for stable downstream display.
    result = apply_decision_policy(0.123456789)

    assert result["probability"] == 0.1235


def test_action_is_string():
    # Every decision band should provide a non-empty human-readable action message.
    low_result = apply_decision_policy(0.1)
    medium_result = apply_decision_policy(0.5)
    high_result = apply_decision_policy(0.9)

    assert isinstance(low_result["action"], str) and low_result["action"]
    assert isinstance(medium_result["action"], str) and medium_result["action"]
    assert isinstance(high_result["action"], str) and high_result["action"]


def test_band_summary_structure():
    # The summary helper should always return all three bands with the expected fields.
    probabilities = np.array([0.1, 0.2, 0.3, 0.45, 0.55, 0.65, 0.8, 0.85, 0.9, 0.95])
    y_true = np.array([0, 0, 1, 0, 1, 0, 1, 1, 0, 1])

    summary = score_band_summary(probabilities, y_true)

    assert set(summary.keys()) == {"low_risk", "medium_risk", "high_risk"}
    for band_summary in summary.values():
        assert set(band_summary.keys()) == {"transaction_count", "fraud_count", "fraud_rate", "action"}


def test_band_summary_counts_correct():
    # Transactions should be bucketed into low, medium, and high bands using the shared policy thresholds.
    probabilities = np.array([0.1, 0.2, 0.5, 0.6, 0.8, 0.9])
    y_true = np.array([0, 0, 1, 0, 1, 1])

    summary = score_band_summary(probabilities, y_true)

    assert summary["low_risk"]["transaction_count"] == 2
    assert summary["medium_risk"]["transaction_count"] == 2
    assert summary["high_risk"]["transaction_count"] == 2


def test_band_summary_fraud_rate_zero():
    # A band with no fraud labels should report a zero fraud rate.
    probabilities = np.array([0.1, 0.2, 0.3])
    y_true = np.array([0, 0, 0])

    summary = score_band_summary(probabilities, y_true)

    assert summary["low_risk"]["fraud_rate"] == 0.0


def test_band_summary_fraud_rate_one():
    # A band containing only fraud labels should report a fraud rate of one.
    probabilities = np.array([0.8, 0.85, 0.9])
    y_true = np.array([1, 1, 1])

    summary = score_band_summary(probabilities, y_true)

    assert summary["high_risk"]["fraud_rate"] == 1.0


def test_band_summary_empty_band():
    # Empty bands should keep a zero count and zero fraud rate instead of failing on division.
    probabilities = np.array([0.1, 0.2, 0.3])
    y_true = np.array([0, 1, 0])

    summary = score_band_summary(probabilities, y_true)

    assert summary["high_risk"]["transaction_count"] == 0
    assert summary["high_risk"]["fraud_rate"] == 0.0
