"""Apply the bank-style fraud decision policy to model probability scores."""


# Thresholds define three risk bands reflecting real bank fraud operations.
# Low risk: auto-approve. Review band: step-up verification required.
# High risk: transaction blocked pending investigation.
# Lower bound set at 0.40 to capture borderline cases without flooding
# the review queue. Upper bound matches the tuned model threshold (0.75).
APPROVE_BELOW = 0.40
BLOCK_AT_OR_ABOVE = 0.75


def apply_decision_policy(probability: float) -> dict:
    """Apply the three-band fraud decision policy to a fraud probability score.

    Bands:
        Low risk  (score < 0.40):  Auto-approve. No friction added.
        Medium risk (0.40–0.75):   Step-up verification required.
                                   Examples: OTP push, card holder call,
                                   CVV re-entry, or 3D Secure challenge.
        High risk (score >= 0.75): Transaction blocked. Card flagged for
                                   review by fraud operations team.

    Args:
        probability: Fraud probability score from XGBoost model (0.0–1.0).

    Returns:
        dict with keys:
            decision:     "approve" | "step_up_verification" | "block"
            risk_band:    "low_risk" | "medium_risk" | "high_risk"
            action:       Human-readable action string
            probability:  The input probability rounded to 4 decimal places
    """
    # Round once here so API responses, CLI demos, and tests all apply the same boundary behavior.
    rounded_probability = round(float(probability), 4)

    if rounded_probability < APPROVE_BELOW:
        return {
            "decision": "approve",
            "risk_band": "low_risk",
            "action": "Transaction approved automatically.",
            "probability": rounded_probability,
        }

    if rounded_probability < BLOCK_AT_OR_ABOVE:
        return {
            "decision": "step_up_verification",
            "risk_band": "medium_risk",
            "action": (
                "Step-up verification required. Cardholder must complete additional "
                "authentication before transaction is processed."
            ),
            "probability": rounded_probability,
        }

    return {
        "decision": "block",
        "risk_band": "high_risk",
        "action": (
            "Transaction blocked. Card flagged for review by fraud operations "
            "team. Cardholder notified."
        ),
        "probability": rounded_probability,
    }


def score_band_summary(test_probabilities, y_true) -> dict:
    """Compute fraud rate and transaction count for each decision band.

    Args:
        test_probabilities: Array of fraud probabilities.
        y_true: Array of true fraud labels.

    Returns:
        dict with band names as keys, each containing:
            transaction_count, fraud_count, fraud_rate, action
    """
    summary = {
        "low_risk": {
            "transaction_count": 0,
            "fraud_count": 0,
            "fraud_rate": 0.0,
            "action": "Transaction approved automatically.",
        },
        "medium_risk": {
            "transaction_count": 0,
            "fraud_count": 0,
            "fraud_rate": 0.0,
            "action": (
                "Step-up verification required. Cardholder must complete additional "
                "authentication before transaction is processed."
            ),
        },
        "high_risk": {
            "transaction_count": 0,
            "fraud_count": 0,
            "fraud_rate": 0.0,
            "action": (
                "Transaction blocked. Card flagged for review by fraud operations "
                "team. Cardholder notified."
            ),
        },
    }

    for probability, fraud_label in zip(test_probabilities, y_true):
        decision = apply_decision_policy(probability)
        risk_band = decision["risk_band"]
        summary[risk_band]["transaction_count"] += 1
        summary[risk_band]["fraud_count"] += int(fraud_label)

    for risk_band, band_summary in summary.items():
        transaction_count = band_summary["transaction_count"]
        if transaction_count > 0:
            # Leave empty bands at 0.0 instead of raising a division-by-zero error.
            band_summary["fraud_rate"] = band_summary["fraud_count"] / transaction_count

    return summary


if __name__ == "__main__":
    for example_probability in [0.15, 0.55, 0.85]:
        print(apply_decision_policy(example_probability))
