"""Prepare model-ready fraud datasets and preprocessing artifacts."""

import argparse
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ENRICHED_SAMPLE_FILE,
    MODELS_DIR,
    SPLITS_DIR,
    TARGET_COL,
    get_feature_metadata_file,
    get_preprocessor_file,
    get_test_split_file,
    get_train_split_file,
    get_val_split_file,
)


DROPPED_COLUMNS = [
    "Unnamed: 0",
    "trans_date_trans_time",
    "dob",
    "cc_num",
    "first",
    "last",
    "street",
    "trans_num",
    "unix_time",
    "merchant",
]


# Keep the first model version simple and broadly useful.
# Raw merchant names are intentionally excluded because they create very
# high-cardinality one-hot features that tend to be sparse and noisy.
# The current feature set includes multi-window velocity signals and
# merchant-category risk features plus prior merchant-history signals to
# better reflect realistic fraud patterns.
NUMERIC_FEATURE_CANDIDATES = [
    "amt",
    "zip",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long",
    "txn_hour",
    "txn_dayofweek",
    "is_weekend",
    "is_night_transaction",
    "customer_age",
    "merchant_customer_distance_km",
    "amount_log",
    "amount_vs_customer_mean",
    "amount_zscore_customer",
    "is_high_amount",
    "is_very_high_amount",
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
    "merchant_category_risk_score",
    "merchant_prior_txn_count",
    "merchant_prior_fraud_count",
    "merchant_prior_fraud_rate",
    "synthetic_device_changed",
    "synthetic_failed_logins_24h",
    "synthetic_ip_risk_score",
    "synthetic_account_age_days",
    "synthetic_email_age_days",
    "synthetic_billing_shipping_mismatch",
]

CATEGORICAL_FEATURE_CANDIDATES = [
    "category",
    "state",
    "gender",
    "merchant_category_risk_tier",
]

SYNTHETIC_FEATURE_COLUMNS = [
    "synthetic_device_changed",
    "synthetic_failed_logins_24h",
    "synthetic_ip_risk_score",
    "synthetic_account_age_days",
    "synthetic_email_age_days",
    "synthetic_billing_shipping_mismatch",
]

TOP_SYNTHETIC_FEATURE = "synthetic_ip_risk_score"
SUPPORTED_VARIANTS = {"full", "no_synthetic", "no_ip_risk"}


def load_dataset() -> pd.DataFrame:
    """Load the enriched dataset used for downstream model preparation.

    This method reads the enriched fraud sample that was generated in the
    previous phase, parses transaction timestamps, and returns it as a dataframe.

    Args:
        None

    Returns:
        pd.DataFrame: Enriched fraud dataset ready for feature selection.
    """
    dataset = pd.read_csv(ENRICHED_SAMPLE_FILE)
    dataset["trans_date_trans_time"] = pd.to_datetime(dataset["trans_date_trans_time"])
    return dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for feature-building variants.

    This method reads the requested ablation variant from the command line.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Build fraud-model features for a specific variant.")
    parser.add_argument(
        "--variant",
        default="full",
        choices=sorted(SUPPORTED_VARIANTS),
        help="Feature-selection variant to build.",
    )
    return parser.parse_args()


def get_feature_lists(df: pd.DataFrame, variant: str = "full") -> tuple[list[str], list[str], list[str]]:
    """Build numeric, categorical, and dropped feature lists.

    This method selects the modeling columns to keep, filters them to columns
    that actually exist in the dataset, and records which raw columns are
    intentionally excluded from the first model version. Broad categorical
    fields such as category and state are preferred here because they usually
    generalize better than raw merchant or city names. Multi-window velocity
    features, merchant-category risk features, and prior merchant-history
    features are included to improve fraud realism, while variants can still
    remove synthetic features for ablations.

    Args:
        df (pd.DataFrame): Enriched fraud dataset.
        variant (str): Feature-selection variant to build.

    Returns:
        tuple[list[str], list[str], list[str]]: Numeric features, categorical
        features, and dropped columns.
    """
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unsupported variant '{variant}'. Expected one of: {sorted(SUPPORTED_VARIANTS)}."
        )

    available_columns = set(df.columns)
    dropped_columns = [column for column in DROPPED_COLUMNS if column in available_columns]
    numeric_columns = [column for column in NUMERIC_FEATURE_CANDIDATES if column in available_columns]
    categorical_columns = [column for column in CATEGORICAL_FEATURE_CANDIDATES if column in available_columns]

    if variant == "no_synthetic":
        # This ablation removes every synthetic operational-risk feature at once.
        numeric_columns = [column for column in numeric_columns if column not in SYNTHETIC_FEATURE_COLUMNS]
    elif variant == "no_ip_risk":
        # This variant isolates the lift from the strongest synthetic signal alone.
        numeric_columns = [column for column in numeric_columns if column != TOP_SYNTHETIC_FEATURE]

    return numeric_columns, categorical_columns, dropped_columns


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train, validation, and test partitions.

    This method sorts transactions chronologically and assigns the oldest 70%
    to training, the next 15% to validation, and the newest 15% to test.
    Fraud systems are better evaluated this way because they predict future
    behavior from past observations, while random splits can leak future
    behavioral patterns into training.

    Args:
        df (pd.DataFrame): Full enriched dataset.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation,
        and test dataframes.
    """
    sorted_df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    total_rows = len(sorted_df)
    train_end_index = int(total_rows * 0.70)
    val_end_index = int(total_rows * 0.85)

    # Preserve chronological ordering within each split so evaluation mirrors deployment.
    train_df = sorted_df.iloc[:train_end_index].copy()
    val_df = sorted_df.iloc[train_end_index:val_end_index].copy()
    test_df = sorted_df.iloc[val_end_index:].copy()
    return train_df, val_df, test_df


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    """Build the preprocessing transformer for the baseline model stage.

    This method creates separate preprocessing pipelines for numeric and
    categorical features and combines them in a single ColumnTransformer.

    Args:
        numeric_columns (list[str]): Numeric feature names.
        categorical_columns (list[str]): Categorical feature names.

    Returns:
        ColumnTransformer: Configured preprocessing transformer.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def save_artifacts(
    variant: str,
    preprocessor: ColumnTransformer,
    feature_metadata: dict[str, object],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Persist preprocessing artifacts and reusable dataset splits.

    This method writes the fitted preprocessing object metadata file and the
    raw split CSVs so the next phase can train models reproducibly.

    Args:
        variant (str): Feature-selection variant being saved.
        preprocessor (ColumnTransformer): Preprocessing transformer to save.
        feature_metadata (dict[str, object]): Metadata about selected features.
        train_df (pd.DataFrame): Training split.
        val_df (pd.DataFrame): Validation split.
        test_df (pd.DataFrame): Test split.

    Returns:
        None
    """
    preprocessor_file = get_preprocessor_file(variant)
    feature_metadata_file = get_feature_metadata_file(variant)
    train_split_file = get_train_split_file(variant)
    val_split_file = get_val_split_file(variant)
    test_split_file = get_test_split_file(variant)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, preprocessor_file)

    with feature_metadata_file.open("w", encoding="utf-8") as metadata_file:
        json.dump(feature_metadata, metadata_file, indent=2)

    train_df.to_csv(train_split_file, index=False)
    val_df.to_csv(val_split_file, index=False)
    test_df.to_csv(test_split_file, index=False)


def fraud_rate(df: pd.DataFrame) -> float:
    """Calculate the fraud prevalence for a dataset split.

    This method computes the mean of the target column and returns it as a
    simple fraud-rate metric for summary output.

    Args:
        df (pd.DataFrame): Dataset split containing the target column.

    Returns:
        float: Fraud rate for the split.
    """
    return float(df[TARGET_COL].mean())


def main() -> None:
    """Prepare model-ready splits and preprocessing artifacts.

    This method loads the enriched dataset, selects model features, creates
    reusable train-validation-test splits, fits the preprocessing pipeline,
    and saves artifacts needed for baseline modeling.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    variant = args.variant

    dataset = load_dataset()
    numeric_columns, categorical_columns, dropped_columns = get_feature_lists(dataset, variant=variant)

    # Keep the timestamp only long enough to build the chronological split.
    split_dataset_with_timestamp = dataset[["trans_date_trans_time"] + numeric_columns + categorical_columns + [TARGET_COL]].copy()

    train_df, val_df, test_df = split_dataset(split_dataset_with_timestamp)
    # Saved modeling splits should contain only model inputs plus the target.
    train_df = train_df.drop(columns=["trans_date_trans_time"])
    val_df = val_df.drop(columns=["trans_date_trans_time"])
    test_df = test_df.drop(columns=["trans_date_trans_time"])
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)

    # Fit on the training data only so validation and test data stay untouched.
    preprocessor.fit(train_df[numeric_columns + categorical_columns], train_df[TARGET_COL])

    feature_metadata = {
        "variant": variant,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "dropped_columns": dropped_columns,
        "target_column": TARGET_COL,
    }
    save_artifacts(variant, preprocessor, feature_metadata, train_df, val_df, test_df)

    preprocessor_file = get_preprocessor_file(variant)
    feature_metadata_file = get_feature_metadata_file(variant)
    train_split_file = get_train_split_file(variant)
    val_split_file = get_val_split_file(variant)
    test_split_file = get_test_split_file(variant)

    print(f"Input shape: {dataset.shape}")
    print(f"Variant: {variant}")
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Train fraud rate: {fraud_rate(train_df):.6f}")
    print(f"Validation fraud rate: {fraud_rate(val_df):.6f}")
    print(f"Test fraud rate: {fraud_rate(test_df):.6f}")
    print(f"Numeric columns: {len(numeric_columns)}")
    print(f"Categorical columns: {len(categorical_columns)}")
    print(f"Preprocessor file: {preprocessor_file}")
    print(f"Feature metadata file: {feature_metadata_file}")
    print(f"Train split file: {train_split_file}")
    print(f"Validation split file: {val_split_file}")
    print(f"Test split file: {test_split_file}")
    print("Added ablation-ready feature selection with variants: full, no_synthetic, and no_ip_risk.")
    print("Example commands:")
    print("- python -m src.build_features --variant full")
    print("- python -m src.build_features --variant no_synthetic")
    print("- python -m src.build_features --variant no_ip_risk")


if __name__ == "__main__":
    main()
