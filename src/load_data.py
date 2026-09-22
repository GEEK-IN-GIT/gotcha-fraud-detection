"""Load the raw fraud dataset and create a reusable development sample."""

import pandas as pd

from src.config import (
    DEV_SAMPLE_FILE,
    DEV_SAMPLE_ROWS,
    RANDOM_SEED,
    RAW_DATA_FILE,
    SAMPLES_DIR,
    TARGET_COL,
)


def main() -> None:
    """Inspect the raw dataset and persist a reproducible sample.

    This method loads the raw fraud dataset, prints basic dataset diagnostics,
    creates a development sample, and saves that sample for downstream work.

    Args:
        None

    Returns:
        None
    """
    df = pd.read_csv(RAW_DATA_FILE)

    print(f"Dataset shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")

    print("\nDtypes:")
    print(df.dtypes)

    # Quick data quality check before sampling.
    print("\nMissing value counts:")
    print(df.isna().sum())

    counts = df[TARGET_COL].value_counts(dropna=False)
    ratios = df[TARGET_COL].value_counts(normalize=True, dropna=False)

    print("\nFraud class counts:")
    print(counts)

    print("\nFraud class percentages:")
    print(ratios)

    # The sample directory may not exist on a fresh clone.
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Cap the sample size so smaller source files still work.
    sample_size = min(DEV_SAMPLE_ROWS, len(df))
    sample_df = df.sample(n=sample_size, random_state=RANDOM_SEED)

    # Compare the sample target mix with the source data.
    sample_counts = sample_df[TARGET_COL].value_counts(dropna=False)
    sample_ratios = sample_df[TARGET_COL].value_counts(normalize=True, dropna=False)

    print("\nSample fraud class counts:")
    print(sample_counts)

    print("\nSample fraud class percentages:")
    print(sample_ratios)

    sample_df.to_csv(DEV_SAMPLE_FILE, index=False)

    print(f"\nDevelopment sample saved to: {DEV_SAMPLE_FILE}")


if __name__ == "__main__":
    main()
