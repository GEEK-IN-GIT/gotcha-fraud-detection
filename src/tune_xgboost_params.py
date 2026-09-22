"""Run a compact hyperparameter search for the full XGBoost fraud model."""

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
    get_xgboost_best_metrics_file,
    get_xgboost_best_model_file,
    get_xgboost_best_params_file,
    get_xgboost_param_search_file,
)


VARIANT = "full"


def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object, dict[str, object]]:
    """Load the saved full-variant artifacts for XGBoost parameter tuning.

    Args:
        None

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object, dict[str, object]]:
            Train dataframe, validation dataframe, test dataframe,
            preprocessor, and feature metadata.
    """
    train_df = pd.read_csv(get_train_split_file(VARIANT))
    val_df = pd.read_csv(get_val_split_file(VARIANT))
    test_df = pd.read_csv(get_test_split_file(VARIANT))
    preprocessor = joblib.load(get_preprocessor_file(VARIANT))

    with get_feature_metadata_file(VARIANT).open("r", encoding="utf-8") as metadata_file:
        feature_metadata = json.load(metadata_file)

    return train_df, val_df, test_df, preprocessor, feature_metadata


def prepare_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_metadata: dict[str, object],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """Rebuild feature matrices and target vectors from saved metadata.

    Args:
        train_df (pd.DataFrame): Training split.
        val_df (pd.DataFrame): Validation split.
        test_df (pd.DataFrame): Test split.
        feature_metadata (dict[str, object]): Saved feature metadata.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
            Train, validation, and test features and targets, plus ordered feature columns.
    """
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
    """Compute the XGBoost positive-class weight from training labels.

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
    """Compute classification metrics for one evaluation split.

    Args:
        y_true (pd.Series): Ground-truth labels.
        predicted_labels: Binary model predictions.
        predicted_probabilities: Positive-class probabilities.

    Returns:
        dict[str, object]: Evaluation metrics and confusion counts.
    """
    confusion = confusion_matrix(y_true, predicted_labels).tolist()
    return {
        "precision": float(precision_score(y_true, predicted_labels, zero_division=0)),
        "recall": float(recall_score(y_true, predicted_labels, zero_division=0)),
        "f1": float(f1_score(y_true, predicted_labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, predicted_probabilities)),
        "pr_auc": float(average_precision_score(y_true, predicted_probabilities)),
        "confusion_matrix": confusion,
        "false_positives": int(confusion[0][1]),
        "false_negatives": int(confusion[1][0]),
    }


def get_param_grid() -> list[dict[str, float | int]]:
    """Return a compact manual hyperparameter search space for XGBoost.

    Args:
        None

    Returns:
        list[dict[str, float | int]]: Candidate XGBoost parameter sets.
    """
    return [
        {"max_depth": 4, "learning_rate": 0.03, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 1.0},
        {"max_depth": 5, "learning_rate": 0.03, "min_child_weight": 3, "subsample": 1.0, "colsample_bytree": 0.8},
        {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 5, "learning_rate": 0.1, "min_child_weight": 5, "subsample": 1.0, "colsample_bytree": 1.0},
        {"max_depth": 6, "learning_rate": 0.03, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 1.0},
        {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 3, "subsample": 1.0, "colsample_bytree": 0.8},
        {"max_depth": 6, "learning_rate": 0.1, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 1.0},
    ]


def run_param_search(
    x_train_processed,
    y_train: pd.Series,
    x_val_processed,
    y_val: pd.Series,
    scale_pos_weight: float,
) -> list[dict[str, object]]:
    """Train and evaluate each candidate parameter set on the validation split.

    Args:
        x_train_processed: Transformed training matrix.
        y_train (pd.Series): Training targets.
        x_val_processed: Transformed validation matrix.
        y_val (pd.Series): Validation targets.
        scale_pos_weight (float): Imbalance weight for XGBoost.

    Returns:
        list[dict[str, object]]: Validation results for all candidate parameter sets.
    """
    search_results: list[dict[str, object]] = []

    for params in get_param_grid():
        # Keep the search intentionally compact so validation comparisons stay readable and reproducible.
        candidate_model = XGBClassifier(
            n_estimators=300,
            objective="binary:logistic",
            eval_metric=["logloss", "aucpr"],
            early_stopping_rounds=20,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            **params,
        )
        candidate_model.fit(
            x_train_processed,
            y_train,
            eval_set=[(x_val_processed, y_val)],
            verbose=False,
        )

        val_probabilities = candidate_model.predict_proba(x_val_processed)[:, 1]
        val_predictions = candidate_model.predict(x_val_processed)
        validation_metrics = compute_metrics(y_val, val_predictions, val_probabilities)

        search_results.append(
            {
                "params": params,
                "validation_metrics": validation_metrics,
            }
        )

    return search_results


def select_best_result(search_results: list[dict[str, object]]) -> dict[str, object]:
    """Select the best validation result using the project ranking policy.

    Args:
        search_results (list[dict[str, object]]): Validation results for all candidates.

    Returns:
        dict[str, object]: Best result entry.
    """
    return max(
        search_results,
        key=lambda result: (
            result["validation_metrics"]["pr_auc"],
            result["validation_metrics"]["f1"],
            result["validation_metrics"]["precision"],
            -result["validation_metrics"]["false_positives"],
        ),
    )


def save_outputs(
    search_results: list[dict[str, object]],
    best_params_payload: dict[str, object],
    best_model_metrics_payload: dict[str, object],
    best_model: XGBClassifier,
) -> None:
    """Persist XGBoost tuning outputs and the selected best model.

    Args:
        search_results (list[dict[str, object]]): Full parameter search results.
        best_params_payload (dict[str, object]): Selected best parameter summary.
        best_model_metrics_payload (dict[str, object]): Validation/test metrics for the best model.
        best_model (XGBClassifier): Trained best model.

    Returns:
        None
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    get_xgboost_best_model_file(VARIANT).parent.mkdir(parents=True, exist_ok=True)

    with get_xgboost_param_search_file(VARIANT).open("w", encoding="utf-8") as search_file:
        json.dump(search_results, search_file, indent=2)

    with get_xgboost_best_params_file(VARIANT).open("w", encoding="utf-8") as params_file:
        json.dump(best_params_payload, params_file, indent=2)

    with get_xgboost_best_metrics_file(VARIANT).open("w", encoding="utf-8") as metrics_file:
        json.dump(best_model_metrics_payload, metrics_file, indent=2)

    joblib.dump(best_model, get_xgboost_best_model_file(VARIANT))


def main() -> None:
    """Run a compact XGBoost hyperparameter search for the full variant.

    This method loads the full-variant artifacts, transforms the saved splits
    without refitting the preprocessor, compares a small set of XGBoost
    parameter combinations on validation PR-AUC, and saves the best result.

    Args:
        None

    Returns:
        None
    """
    train_df, val_df, test_df, preprocessor, feature_metadata = load_artifacts()
    x_train, y_train, x_val, y_val, x_test, y_test, feature_columns = prepare_datasets(
        train_df,
        val_df,
        test_df,
        feature_metadata,
    )

    x_train_processed = preprocessor.transform(x_train)
    x_val_processed = preprocessor.transform(x_val)
    x_test_processed = preprocessor.transform(x_test)

    scale_pos_weight = compute_scale_pos_weight(y_train)
    search_results = run_param_search(
        x_train_processed,
        y_train,
        x_val_processed,
        y_val,
        scale_pos_weight,
    )
    best_result = select_best_result(search_results)

    # Retrain only the winning configuration so the saved artifact matches the reported best result.
    best_model = XGBClassifier(
        n_estimators=300,
        objective="binary:logistic",
        eval_metric=["logloss", "aucpr"],
        early_stopping_rounds=20,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        **best_result["params"],
    )
    best_model.fit(
        x_train_processed,
        y_train,
        eval_set=[(x_val_processed, y_val)],
        verbose=False,
    )

    test_probabilities = best_model.predict_proba(x_test_processed)[:, 1]
    test_predictions = best_model.predict(x_test_processed)
    test_metrics = compute_metrics(y_test, test_predictions, test_probabilities)

    best_params_payload = {
        "variant": VARIANT,
        "selection_rule": "highest_validation_pr_auc_then_f1_then_precision_then_lower_false_positives",
        "scale_pos_weight": float(scale_pos_weight),
        "best_params": best_result["params"],
        "best_validation_metrics": best_result["validation_metrics"],
    }

    best_model_metrics_payload = {
        "variant": VARIANT,
        "target_column": feature_metadata.get("target_column", TARGET_COL),
        "raw_feature_count": len(feature_columns),
        "transformed_feature_count": int(x_train_processed.shape[1]),
        "scale_pos_weight": float(scale_pos_weight),
        "best_params": best_result["params"],
        "validation_metrics": best_result["validation_metrics"],
        "test_metrics": test_metrics,
    }

    save_outputs(search_results, best_params_payload, best_model_metrics_payload, best_model)

    print(f"Parameter combinations tested: {len(search_results)}")
    print(f"Best parameter set: {best_result['params']}")
    print(f"Best validation metrics: {best_result['validation_metrics']}")
    print(f"Final test metrics: {test_metrics}")
    print(f"Parameter search file: {get_xgboost_param_search_file(VARIANT)}")
    print(f"Best params file: {get_xgboost_best_params_file(VARIANT)}")
    print(f"Best model metrics file: {get_xgboost_best_metrics_file(VARIANT)}")
    print(f"Best tuned model file: {get_xgboost_best_model_file(VARIANT)}")


if __name__ == "__main__":
    main()
