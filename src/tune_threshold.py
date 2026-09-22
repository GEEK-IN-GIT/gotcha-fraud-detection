"""Tune the decision threshold for the full XGBoost fraud model."""

import argparse
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    METRICS_DIR,
    TARGET_COL,
    get_feature_metadata_file,
    get_preprocessor_file,
    get_selected_threshold_file,
    get_test_split_file,
    get_threshold_sweep_file,
    get_tuned_test_metrics_file,
    get_val_split_file,
    get_xgboost_model_file,
)


SUPPORTED_VARIANTS = {"full", "no_synthetic", "no_ip_risk"}

# Cost estimates derived from industry benchmarks.
# FN cost: LexisNexis True Cost of Fraud report (~$280 per missed fraud
# including chargeback, fees, and dispute handling).
# FP cost: Javelin Strategy false decline research (~$12.50 per blocked
# legitimate transaction including lost interchange and customer friction).
FRAUD_COST_PER_FN = 280.0
FALSE_BLOCK_COST_PER_FP = 12.50
# Banks typically target a review rate under 1% to keep fraud ops
# workload manageable. 0.75% is a conservative operational ceiling
# that filters out thresholds that would flood the manual review queue.
MAX_REVIEW_RATE = 0.0075


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for threshold tuning variants.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Tune the XGBoost decision threshold for a specific variant.")
    parser.add_argument(
        "--variant",
        default="full",
        choices=sorted(SUPPORTED_VARIANTS),
        help="Feature-selection variant to tune.",
    )
    return parser.parse_args()


def load_artifacts(variant: str) -> tuple[pd.DataFrame, pd.DataFrame, object, dict[str, object], object]:
    """Load the saved artifacts for threshold tuning.

    This method reads the validation and test splits, the shared preprocessor,
    the feature metadata, and the trained XGBoost model from disk.

    Args:
        None

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, object, dict[str, object], object]:
            Validation dataframe, test dataframe, preprocessor, feature metadata,
            and trained XGBoost model.
    """
    val_df = pd.read_csv(get_val_split_file(variant))
    test_df = pd.read_csv(get_test_split_file(variant))
    preprocessor = joblib.load(get_preprocessor_file(variant))
    model = joblib.load(get_xgboost_model_file(variant))

    with get_feature_metadata_file(variant).open("r", encoding="utf-8") as metadata_file:
        feature_metadata = json.load(metadata_file)

    return val_df, test_df, preprocessor, feature_metadata, model


def prepare_datasets(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_metadata: dict[str, object],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """Rebuild feature matrices and target vectors from saved metadata.

    Args:
        val_df (pd.DataFrame): Validation split.
        test_df (pd.DataFrame): Test split.
        feature_metadata (dict[str, object]): Saved feature metadata.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
            Validation features, validation target, test features, test target,
            and ordered feature columns.
    """
    numeric_columns = feature_metadata["numeric_columns"]
    categorical_columns = feature_metadata["categorical_columns"]
    target_column = feature_metadata.get("target_column", TARGET_COL)
    feature_columns = numeric_columns + categorical_columns

    x_val = val_df[feature_columns]
    y_val = val_df[target_column]
    x_test = test_df[feature_columns]
    y_test = test_df[target_column]

    return x_val, y_val, x_test, y_test, feature_columns


def compute_threshold_metrics(
    y_true: pd.Series,
    predicted_probabilities,
    threshold: float,
) -> dict[str, object]:
    """Compute threshold-based classification metrics for one probability cutoff.

    Args:
        y_true (pd.Series): Ground-truth labels.
        predicted_probabilities: Positive-class probabilities.
        threshold (float): Decision threshold to evaluate.

    Returns:
        dict[str, object]: Threshold metrics and confusion counts.
    """
    # Threshold tuning reuses one trained model; only the decision cutoff changes here.
    predicted_labels = (predicted_probabilities >= threshold).astype(int)
    confusion = confusion_matrix(y_true, predicted_labels).tolist()
    false_positives = int(confusion[0][1])
    false_negatives = int(confusion[1][0])
    predicted_fraud_count = int(predicted_labels.sum())

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predicted_labels, zero_division=0)),
        "recall": float(recall_score(y_true, predicted_labels, zero_division=0)),
        "f1": float(f1_score(y_true, predicted_labels, zero_division=0)),
        "confusion_matrix": confusion,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "review_rate": float(predicted_fraud_count / len(y_true)),
    }


def run_threshold_sweep(y_val: pd.Series, val_probabilities) -> list[dict[str, object]]:
    """Evaluate a fixed grid of decision thresholds on the validation set.

    Args:
        y_val (pd.Series): Validation labels.
        val_probabilities: Validation fraud probabilities.

    Returns:
        list[dict[str, object]]: Validation metrics for each threshold.
    """
    # A coarse grid is easier to explain and less likely to overfit threshold noise.
    thresholds = [threshold / 100 for threshold in range(5, 100, 5)]
    return [
        compute_threshold_metrics(y_val, val_probabilities, threshold)
        for threshold in thresholds
    ]


def print_focused_threshold_table(threshold_sweep: list[dict[str, object]], table_title: str) -> None:
    """Print a compact threshold table around the likely operating region.

    Args:
        threshold_sweep (list[dict[str, object]]): Validation metrics for all thresholds.
        table_title (str): Title to print above the threshold table.

    Returns:
        None
    """
    target_thresholds = {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90}
    focused_results = [
        result
        for result in threshold_sweep
        if round(result["threshold"], 2) in target_thresholds
    ]

    print(f"\n{table_title}")
    header = (
        f"{'threshold':<11}"
        f"{'precision':>11}"
        f"{'recall':>11}"
        f"{'f1':>11}"
        f"{'fp':>8}"
        f"{'fn':>8}"
        f"{'review_rate':>14}"
    )
    print(header)
    print("-" * len(header))

    for result in focused_results:
        print(
            f"{result['threshold']:<11.2f}"
            f"{result['precision']:>11.4f}"
            f"{result['recall']:>11.4f}"
            f"{result['f1']:>11.4f}"
            f"{result['false_positives']:>8}"
            f"{result['false_negatives']:>8}"
            f"{result['review_rate']:>14.4f}"
        )


def select_best_threshold(threshold_results: list[dict[str, object]]) -> dict[str, object]:
    """Select the best threshold using the project thresholding policy.

    Args:
        threshold_results (list[dict[str, object]]): Validation results for all thresholds.

    Returns:
        dict[str, object]: Selected validation threshold result.
    """
    for result in threshold_results:
        result["expected_cost"] = (
            result["false_negatives"] * FRAUD_COST_PER_FN
            + result["false_positives"] * FALSE_BLOCK_COST_PER_FP
        )

    eligible_results = [
        result for result in threshold_results
        if result["recall"] >= 0.80 and result["review_rate"] <= MAX_REVIEW_RATE
    ]

    if eligible_results:
        # Apply the recall floor first, then choose the least costly operating point inside that feasible set.
        return min(eligible_results, key=lambda result: result["expected_cost"])

    recall_floor_results = [
        result for result in threshold_results
        if result["recall"] >= 0.80
    ]

    if recall_floor_results:
        return min(recall_floor_results, key=lambda result: result["expected_cost"])

    # If no threshold clears the recall floor, still take the cheapest option overall.
    return min(threshold_results, key=lambda result: result["expected_cost"])


def save_outputs(
    variant: str,
    threshold_sweep: list[dict[str, object]],
    selected_threshold_payload: dict[str, object],
    tuned_test_metrics: dict[str, object],
) -> None:
    """Save threshold-tuning outputs to the metrics directory.

    Args:
        threshold_sweep (list[dict[str, object]]): Full validation threshold sweep.
        selected_threshold_payload (dict[str, object]): Chosen threshold details.
        tuned_test_metrics (dict[str, object]): Test metrics at the chosen threshold.

    Returns:
        None
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with get_threshold_sweep_file(variant).open("w", encoding="utf-8") as threshold_file:
        json.dump(threshold_sweep, threshold_file, indent=2)

    with get_selected_threshold_file(variant).open("w", encoding="utf-8") as selected_file:
        json.dump(selected_threshold_payload, selected_file, indent=2)

    with get_tuned_test_metrics_file(variant).open("w", encoding="utf-8") as test_metrics_file:
        json.dump(tuned_test_metrics, test_metrics_file, indent=2)


def main() -> None:
    """Tune the XGBoost threshold on validation and evaluate on test.

    This method loads the saved variant artifacts, transforms validation and
    test features without refitting the preprocessor, sweeps thresholds on
    validation probabilities, selects a threshold, and evaluates it on test.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    variant = args.variant

    val_df, test_df, preprocessor, feature_metadata, model = load_artifacts(variant)
    x_val, y_val, x_test, y_test, feature_columns = prepare_datasets(
        val_df,
        test_df,
        feature_metadata,
    )

    x_val_processed = preprocessor.transform(x_val)
    x_test_processed = preprocessor.transform(x_test)

    val_probabilities = model.predict_proba(x_val_processed)[:, 1]
    test_probabilities = model.predict_proba(x_test_processed)[:, 1]

    threshold_sweep = run_threshold_sweep(y_val, val_probabilities)
    test_threshold_sweep = run_threshold_sweep(y_test, test_probabilities)
    selected_threshold_result = select_best_threshold(threshold_sweep)
    selected_threshold = selected_threshold_result["threshold"]

    tuned_test_metrics = compute_threshold_metrics(y_test, test_probabilities, selected_threshold)
    tuned_test_metrics.update(
        {
            "variant": variant,
            "target_column": feature_metadata.get("target_column", TARGET_COL),
            "raw_feature_count": len(feature_columns),
            "transformed_feature_count": int(x_val_processed.shape[1]),
            "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
            "pr_auc": float(average_precision_score(y_test, test_probabilities)),
        }
    )

    selected_threshold_payload = {
        "variant": variant,
        "selection_rule": "minimum_expected_cost_with_recall_floor_0.80_and_max_review_rate_0.0075",
        "selected_threshold": selected_threshold,
        "validation_metrics": selected_threshold_result,
    }

    save_outputs(variant, threshold_sweep, selected_threshold_payload, tuned_test_metrics)

    print_focused_threshold_table(threshold_sweep, "Validation Threshold Comparison")
    print_focused_threshold_table(test_threshold_sweep, "Test Threshold Comparison")
    print(f"Variant: {variant}")
    print(f"Selected threshold: {selected_threshold:.2f}")
    print(f"Expected cost at selected threshold: ${selected_threshold_result['expected_cost']:,.2f}")
    print(f"Validation metrics at selected threshold: {selected_threshold_result}")
    print(f"Test metrics at selected threshold: {tuned_test_metrics}")


if __name__ == "__main__":
    main()
