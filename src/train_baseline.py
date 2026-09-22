"""Train and evaluate the baseline Logistic Regression fraud model."""

import argparse
import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
    RANDOM_SEED,
    TARGET_COL,
    get_baseline_metrics_file,
    get_baseline_model_file,
    get_feature_metadata_file,
    get_preprocessor_file,
    get_test_split_file,
    get_train_split_file,
    get_val_split_file,
)


SUPPORTED_VARIANTS = {"full", "no_synthetic", "no_ip_risk", "full_random"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for baseline training variants."""
    parser = argparse.ArgumentParser(description="Train the Logistic Regression baseline for a specific variant.")
    parser.add_argument(
        "--variant",
        default="full",
        choices=sorted(SUPPORTED_VARIANTS),
        help="Feature-selection variant to train.",
    )
    return parser.parse_args()


def load_artifacts(variant: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object, dict[str, object]]:
    """Load split datasets and shared preprocessing artifacts.

    This method reads the saved train-validation-test splits, the fitted
    preprocessing transformer, and the feature metadata from disk.

    Args:
        None

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object, dict[str, object]]:
            Train dataframe, validation dataframe, test dataframe,
            preprocessing transformer, and feature metadata.
    """
    # These artifacts were produced in Phase 2 and are reused here as-is.
    train_df = pd.read_csv(get_train_split_file(variant))
    val_df = pd.read_csv(get_val_split_file(variant))
    test_df = pd.read_csv(get_test_split_file(variant))
    preprocessor = joblib.load(get_preprocessor_file(variant))

    with get_feature_metadata_file(variant).open("r", encoding="utf-8") as metadata_file:
        feature_metadata = json.load(metadata_file)

    return train_df, val_df, test_df, preprocessor, feature_metadata


def prepare_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_metadata: dict[str, object],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """Split raw datasets into feature matrices and target vectors.

    This method reconstructs the feature columns from the saved metadata and
    returns model inputs for training, validation, and test evaluation.

    Args:
        train_df (pd.DataFrame): Training split.
        val_df (pd.DataFrame): Validation split.
        test_df (pd.DataFrame): Test split.
        feature_metadata (dict[str, object]): Saved feature metadata.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
            Feature matrices, target vectors, and the ordered feature list.
    """
    # Rebuild the exact feature ordering used when the preprocessor was saved.
    numeric_columns = feature_metadata["numeric_columns"]
    categorical_columns = feature_metadata["categorical_columns"]
    target_column = feature_metadata.get("target_column", TARGET_COL)
    feature_columns = numeric_columns + categorical_columns

    x_train = train_df[feature_columns]
    y_train = train_df[target_column]
    x_val = val_df[feature_columns]
    y_val = val_df[target_column]
    x_test = test_df[feature_columns]
    y_test = test_df[target_column]

    return x_train, y_train, x_val, y_val, x_test, y_test, feature_columns


def compute_metrics(y_true: pd.Series, predicted_labels, predicted_probabilities) -> dict[str, object]:
    """Compute the core evaluation metrics for a binary fraud model.

    This method calculates threshold-based and ranking-based metrics along
    with the confusion matrix in a JSON-friendly format.

    Args:
        y_true (pd.Series): Ground-truth labels.
        predicted_labels: Binary model predictions.
        predicted_probabilities: Fraud probabilities for the positive class.

    Returns:
        dict[str, object]: Evaluation metrics for one dataset split.
    """
    # Keep the confusion matrix JSON-friendly so it can be written directly to disk.
    return {
        "precision": float(precision_score(y_true, predicted_labels, zero_division=0)),
        "recall": float(recall_score(y_true, predicted_labels, zero_division=0)),
        "f1": float(f1_score(y_true, predicted_labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, predicted_probabilities)),
        "pr_auc": float(average_precision_score(y_true, predicted_probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predicted_labels).tolist(),
    }


def save_outputs(variant: str, model: LogisticRegression, metrics_payload: dict[str, object]) -> None:
    """Persist the trained baseline model and evaluation outputs.

    This method writes the model artifact and the metrics report so later
    model versions can be compared against the baseline.

    Args:
        model (LogisticRegression): Trained baseline model.
        metrics_payload (dict[str, object]): Validation and test metrics.

    Returns:
        None
    """
    # Save model and metrics separately so future models can be compared easily.
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_model_file = get_baseline_model_file(variant)
    baseline_metrics_file = get_baseline_metrics_file(variant)
    baseline_model_file.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, baseline_model_file)

    with baseline_metrics_file.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics_payload, metrics_file, indent=2)


def main() -> None:
    """Train and evaluate the baseline Logistic Regression model.

    This method loads the prepared splits and shared preprocessor, transforms
    each split without refitting the preprocessor, trains an imbalance-aware
    Logistic Regression baseline, evaluates it, and saves the outputs.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    variant = args.variant

    train_df, val_df, test_df, preprocessor, feature_metadata = load_artifacts(variant)
    x_train, y_train, x_val, y_val, x_test, y_test, feature_columns = prepare_datasets(
        train_df,
        val_df,
        test_df,
        feature_metadata,
    )

    # The preprocessor was already fit in Phase 2, so this phase only transforms.
    x_train_processed = preprocessor.transform(x_train)
    x_val_processed = preprocessor.transform(x_val)
    x_test_processed = preprocessor.transform(x_test)

    # Use class weighting so the baseline is less biased toward the majority class.
    baseline_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    baseline_model.fit(x_train_processed, y_train)

    # Keep the default threshold for now and use probabilities for ranking metrics.
    val_probabilities = baseline_model.predict_proba(x_val_processed)[:, 1]
    val_predictions = baseline_model.predict(x_val_processed)
    test_probabilities = baseline_model.predict_proba(x_test_processed)[:, 1]
    test_predictions = baseline_model.predict(x_test_processed)

    validation_metrics = compute_metrics(y_val, val_predictions, val_probabilities)
    test_metrics = compute_metrics(y_test, test_predictions, test_probabilities)

    metrics_payload = {
        "model_name": "baseline_logistic_regression",
        "variant": variant,
        "target_column": feature_metadata.get("target_column", TARGET_COL),
        "feature_count": len(feature_columns),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    save_outputs(variant, baseline_model, metrics_payload)

    baseline_model_file = get_baseline_model_file(variant)
    baseline_metrics_file = get_baseline_metrics_file(variant)
    print(f"Variant: {variant}")
    print(f"Train matrix shape: {x_train_processed.shape}")
    print(f"Validation matrix shape: {x_val_processed.shape}")
    print(f"Test matrix shape: {x_test_processed.shape}")
    print(f"Validation metrics: {validation_metrics}")
    print(f"Test metrics: {test_metrics}")
    print(f"Baseline model file: {baseline_model_file}")
    print(f"Baseline metrics file: {baseline_metrics_file}")


if __name__ == "__main__":
    main()
