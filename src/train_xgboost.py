"""Train and evaluate the main XGBoost fraud detection model."""

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
from xgboost import XGBClassifier

from src.config import (
    METRICS_DIR,
    RANDOM_SEED,
    TARGET_COL,
    get_feature_metadata_file,
    get_preprocessor_file,
    get_test_split_file,
    get_train_split_file,
    get_val_split_file,
    get_xgboost_feature_importance_file,
    get_xgboost_metrics_file,
    get_xgboost_model_file,
)

SUPPORTED_VARIANTS = {"full", "no_synthetic", "no_ip_risk", "full_random"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for XGBoost training variants.

    This method reads the requested ablation variant from the command line.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train the XGBoost fraud model for a specific variant.")
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
    train_split_file = get_train_split_file(variant)
    val_split_file = get_val_split_file(variant)
    test_split_file = get_test_split_file(variant)
    preprocessor_file = get_preprocessor_file(variant)
    feature_metadata_file = get_feature_metadata_file(variant)

    # These artifacts were produced in Phase 2 and are reused here as-is.
    train_df = pd.read_csv(train_split_file)
    val_df = pd.read_csv(val_split_file)
    test_df = pd.read_csv(test_split_file)
    preprocessor = joblib.load(preprocessor_file)

    with feature_metadata_file.open("r", encoding="utf-8") as metadata_file:
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


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Compute the positive-class weight for imbalanced XGBoost training.

    This method calculates the ratio of negative to positive training labels
    so XGBoost gives more weight to the minority fraud class.

    Args:
        y_train (pd.Series): Training targets.

    Returns:
        float: Negative-to-positive class ratio.
    """
    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    if positive_count == 0:
        return 1.0
    return negative_count / positive_count


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


def extract_feature_importance(model: XGBClassifier, preprocessor) -> list[dict[str, float | str]]:
    """Extract transformed feature importances in descending order.

    This method aligns XGBoost feature importances with transformed feature
    names when available and falls back to indexed names otherwise.

    Args:
        model (XGBClassifier): Trained XGBoost model.
        preprocessor: Shared fitted preprocessing transformer.

    Returns:
        list[dict[str, float | str]]: Feature importance records.
    """
    importances = model.feature_importances_

    try:
        transformed_feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        transformed_feature_names = [f"feature_{index}" for index in range(len(importances))]

    importance_records = [
        {
            "feature": str(feature_name),
            "importance": float(importance),
        }
        for feature_name, importance in zip(transformed_feature_names, importances)
    ]
    importance_records.sort(key=lambda record: record["importance"], reverse=True)
    return importance_records


def save_outputs(
    variant: str,
    model: XGBClassifier,
    metrics_payload: dict[str, object],
    feature_importance_payload: list[dict[str, float | str]],
) -> None:
    """Persist the trained XGBoost model and evaluation outputs.

    This method writes the model artifact, metrics report, and feature
    importance report so the model can be compared against the baseline.

    Args:
        model (XGBClassifier): Trained XGBoost model.
        metrics_payload (dict[str, object]): Validation and test metrics.
        feature_importance_payload (list[dict[str, float | str]]): Feature importance data.

    Returns:
        None
    """
    xgboost_model_file = get_xgboost_model_file(variant)
    xgboost_metrics_file = get_xgboost_metrics_file(variant)
    xgboost_feature_importance_file = get_xgboost_feature_importance_file(variant)

    # Save model, metrics, and importance outputs as separate reusable artifacts.
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    xgboost_model_file.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, xgboost_model_file)

    with xgboost_metrics_file.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics_payload, metrics_file, indent=2)

    with xgboost_feature_importance_file.open("w", encoding="utf-8") as importance_file:
        json.dump(feature_importance_payload, importance_file, indent=2)


def main() -> None:
    """Train and evaluate the shared-preprocessing XGBoost model.

    This method loads the prepared splits and shared preprocessor, transforms
    each split without refitting the preprocessor, trains an imbalance-aware
    XGBoost model, evaluates it, and saves the outputs.

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

    scale_pos_weight = compute_scale_pos_weight(y_train)
    xgboost_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric=["logloss", "aucpr"],
        early_stopping_rounds=20,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    # Early stopping watches the held-out validation period so the model does not overfit older transactions.
    xgboost_model.fit(
        x_train_processed,
        y_train,
        eval_set=[(x_val_processed, y_val)],
        verbose=False,
    )

    # Keep the default threshold for now and use probabilities for ranking metrics.
    val_probabilities = xgboost_model.predict_proba(x_val_processed)[:, 1]
    val_predictions = xgboost_model.predict(x_val_processed)
    test_probabilities = xgboost_model.predict_proba(x_test_processed)[:, 1]
    test_predictions = xgboost_model.predict(x_test_processed)

    validation_metrics = compute_metrics(y_val, val_predictions, val_probabilities)
    test_metrics = compute_metrics(y_test, test_predictions, test_probabilities)
    # Feature importances are reported on transformed columns because one-hot encoding expands categoricals.
    feature_importance = extract_feature_importance(xgboost_model, preprocessor)
    top_10_features = feature_importance[:10]

    metrics_payload = {
        "model_name": "xgboost_fraud_model",
        "variant": variant,
        "target_column": feature_metadata.get("target_column", TARGET_COL),
        "raw_feature_count": len(feature_columns),
        "transformed_feature_count": int(x_train_processed.shape[1]),
        "scale_pos_weight": float(scale_pos_weight),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "top_10_features": top_10_features,
    }
    save_outputs(variant, xgboost_model, metrics_payload, feature_importance)

    xgboost_model_file = get_xgboost_model_file(variant)
    xgboost_metrics_file = get_xgboost_metrics_file(variant)
    xgboost_feature_importance_file = get_xgboost_feature_importance_file(variant)

    print(f"Variant: {variant}")
    print(f"Train matrix shape: {x_train_processed.shape}")
    print(f"Validation matrix shape: {x_val_processed.shape}")
    print(f"Test matrix shape: {x_test_processed.shape}")
    print(f"Scale pos weight: {scale_pos_weight:.6f}")
    print(f"Validation metrics: {validation_metrics}")
    print(f"Test metrics: {test_metrics}")
    print(f"Top 10 important features: {top_10_features}")
    print(f"XGBoost model file: {xgboost_model_file}")
    print(f"XGBoost metrics file: {xgboost_metrics_file}")
    print(f"XGBoost feature importance file: {xgboost_feature_importance_file}")


if __name__ == "__main__":
    main()
