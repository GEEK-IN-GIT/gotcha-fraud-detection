"""Prepare model-ready fraud datasets using the original random split strategy."""

from sklearn.model_selection import train_test_split

from src.build_features import (
    build_preprocessor,
    get_feature_lists,
    load_dataset,
    parse_args,
    save_artifacts,
    fraud_rate,
)
from src.config import (
    RANDOM_SEED,
    TARGET_COL,
    get_feature_metadata_file,
    get_preprocessor_file,
    get_test_split_file,
    get_train_split_file,
    get_val_split_file,
)


VARIANT = "full_random"


def split_dataset_random(df):
    """Split the dataset into train, validation, and test with stratified randomness.

    Args:
        df: Full modeling dataset.

    Returns:
        tuple: Train, validation, and test dataframes.
    """
    # Match the original 70/15/15 split so random-vs-time comparisons stay apples-to-apples.
    train_temp_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=df[TARGET_COL],
    )
    train_df, val_df = train_test_split(
        train_temp_df,
        test_size=0.17647058823529413,
        random_state=RANDOM_SEED,
        stratify=train_temp_df[TARGET_COL],
    )
    return train_df, val_df, test_df


def main() -> None:
    """Build full-feature artifacts using the original random split strategy.

    Args:
        None

    Returns:
        None
    """
    parse_args()  # Reuse the existing CLI shape even though this script always builds the full feature set.

    dataset = load_dataset()
    numeric_columns, categorical_columns, dropped_columns = get_feature_lists(dataset, variant="full")

    # The random-split benchmark reuses the same full feature set as the main pipeline.
    modeling_dataset = dataset[numeric_columns + categorical_columns + [TARGET_COL]].copy()
    train_df, val_df, test_df = split_dataset_random(modeling_dataset)

    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    preprocessor.fit(train_df[numeric_columns + categorical_columns], train_df[TARGET_COL])

    feature_metadata = {
        "variant": VARIANT,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "dropped_columns": dropped_columns,
        "target_column": TARGET_COL,
    }
    save_artifacts(VARIANT, preprocessor, feature_metadata, train_df, val_df, test_df)

    print(f"Input shape: {dataset.shape}")
    print(f"Variant: {VARIANT}")
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Train fraud rate: {fraud_rate(train_df):.6f}")
    print(f"Validation fraud rate: {fraud_rate(val_df):.6f}")
    print(f"Test fraud rate: {fraud_rate(test_df):.6f}")
    print(f"Preprocessor file: {get_preprocessor_file(VARIANT)}")
    print(f"Feature metadata file: {get_feature_metadata_file(VARIANT)}")
    print(f"Train split file: {get_train_split_file(VARIANT)}")
    print(f"Validation split file: {get_val_split_file(VARIANT)}")
    print(f"Test split file: {get_test_split_file(VARIANT)}")


if __name__ == "__main__":
    main()
