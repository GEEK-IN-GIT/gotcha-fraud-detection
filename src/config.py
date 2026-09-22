"""Project paths and shared constants for the fraud detection workflow."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
SPLITS_DIR = PROCESSED_DIR / "splits"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

RAW_DATA_FILE = RAW_DIR / "fraudTrain.csv"
DEV_SAMPLE_FILE = SAMPLES_DIR / "fraud_dev_sample.csv"
ENRICHED_SAMPLE_FILE = PROCESSED_DIR / "fraud_enriched_sample.csv"
TRAIN_SPLIT_FILE = SPLITS_DIR / "train.csv"
VAL_SPLIT_FILE = SPLITS_DIR / "validation.csv"
TEST_SPLIT_FILE = SPLITS_DIR / "test.csv"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
FEATURE_METADATA_FILE = MODELS_DIR / "feature_metadata.json"
BASELINE_MODEL_FILE = MODELS_DIR / "baseline_logistic_regression.joblib"
BASELINE_METRICS_FILE = METRICS_DIR / "baseline_logistic_regression_metrics.json"
XGBOOST_MODEL_FILE = MODELS_DIR / "xgboost_fraud_model.joblib"
XGBOOST_METRICS_FILE = METRICS_DIR / "xgboost_metrics.json"
XGBOOST_FEATURE_IMPORTANCE_FILE = METRICS_DIR / "xgboost_feature_importance.json"
TARGET_COL = "is_fraud"
DEV_SAMPLE_ROWS = 1000000
RANDOM_SEED = 42


def get_preprocessor_file(variant: str) -> Path:
    """Return the preprocessor artifact path for a feature-selection variant."""
    # Variant-scoped paths keep ablation and comparison runs from clobbering each other.
    return MODELS_DIR / f"preprocessor_{variant}.joblib"


def get_feature_metadata_file(variant: str) -> Path:
    """Return the feature metadata path for a feature-selection variant."""
    return MODELS_DIR / f"feature_metadata_{variant}.json"


def get_train_split_file(variant: str) -> Path:
    """Return the training split path for a feature-selection variant."""
    return SPLITS_DIR / f"train_{variant}.csv"


def get_val_split_file(variant: str) -> Path:
    """Return the validation split path for a feature-selection variant."""
    return SPLITS_DIR / f"validation_{variant}.csv"


def get_test_split_file(variant: str) -> Path:
    """Return the test split path for a feature-selection variant."""
    return SPLITS_DIR / f"test_{variant}.csv"


def get_xgboost_model_file(variant: str) -> Path:
    """Return the XGBoost model artifact path for a feature-selection variant."""
    return MODELS_DIR / f"xgboost_fraud_model_{variant}.joblib"


def get_xgboost_metrics_file(variant: str) -> Path:
    """Return the XGBoost metrics path for a feature-selection variant."""
    return METRICS_DIR / f"xgboost_metrics_{variant}.json"


def get_xgboost_feature_importance_file(variant: str) -> Path:
    """Return the XGBoost feature importance path for a feature-selection variant."""
    return METRICS_DIR / f"xgboost_feature_importance_{variant}.json"


def get_baseline_model_file(variant: str) -> Path:
    """Return the baseline Logistic Regression model artifact path for a variant."""
    return MODELS_DIR / f"baseline_logistic_regression_{variant}.joblib"


def get_baseline_metrics_file(variant: str) -> Path:
    """Return the baseline Logistic Regression metrics path for a variant."""
    return METRICS_DIR / f"baseline_logistic_regression_metrics_{variant}.json"


def get_threshold_sweep_file(variant: str) -> Path:
    """Return the threshold sweep output path for a feature-selection variant."""
    return METRICS_DIR / f"threshold_sweep_{variant}.json"


def get_selected_threshold_file(variant: str) -> Path:
    """Return the selected threshold output path for a feature-selection variant."""
    return METRICS_DIR / f"selected_threshold_{variant}.json"


def get_tuned_test_metrics_file(variant: str) -> Path:
    """Return the tuned test metrics output path for a feature-selection variant."""
    return METRICS_DIR / f"xgboost_tuned_test_metrics_{variant}.json"


def get_xgboost_param_search_file(variant: str) -> Path:
    """Return the XGBoost hyperparameter search output path for a feature-selection variant."""
    return METRICS_DIR / f"xgboost_param_search_{variant}.json"


def get_xgboost_best_params_file(variant: str) -> Path:
    """Return the best XGBoost parameter output path for a feature-selection variant."""
    return METRICS_DIR / f"xgboost_best_params_{variant}.json"


def get_xgboost_best_metrics_file(variant: str) -> Path:
    """Return the best tuned XGBoost metrics output path for a feature-selection variant."""
    return METRICS_DIR / f"xgboost_best_model_metrics_{variant}.json"


def get_xgboost_best_model_file(variant: str) -> Path:
    """Return the best tuned XGBoost model artifact path for a feature-selection variant."""
    return MODELS_DIR / f"xgboost_best_tuned_{variant}.joblib"
