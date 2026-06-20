from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = Path("data/processed/xnli_train.csv")
OUTPUT_DIR = Path("data/processed")

RANDOM_STATE = 42


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_columns = {"text", "label_id", "label_name"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "language" not in df.columns:
        df["language"] = "unknown"

    if "source_split" in df.columns:
        df = df.rename(columns={"source_split": "original_source_split"})

    df["stratify_key"] = df["language"].astype(str) + "_" + df["label_id"].astype(str)

    # Some language-label combinations may be too small for stratification.
    counts = df["stratify_key"].value_counts()
    rare_keys = counts[counts < 3].index

    if len(rare_keys) > 0:
        print("Warning: Some stratification groups are too small. Falling back to label-only stratification.")
        stratify = df["label_id"]
    else:
        stratify = df["stratify_key"]

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    temp_counts = temp_df["stratify_key"].value_counts()
    temp_rare_keys = temp_counts[temp_counts < 2].index

    if len(temp_rare_keys) > 0:
        temp_stratify = temp_df["label_id"]
    else:
        temp_stratify = temp_df["stratify_key"]

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_stratify,
    )

    train_df = train_df.drop(columns=["stratify_key"])
    validation_df = validation_df.drop(columns=["stratify_key"])
    test_df = test_df.drop(columns=["stratify_key"])

    train_df["source_split"] = "train"
    validation_df["source_split"] = "validation"
    test_df["source_split"] = "test"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "xnli_train.csv", index=False)
    validation_df.to_csv(OUTPUT_DIR / "xnli_validation.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "xnli_test.csv", index=False)

    print("Saved cleaned splits:")
    print(f"Train:      {train_df.shape}")
    print(f"Validation: {validation_df.shape}")
    print(f"Test:       {test_df.shape}")

    print("\nTrain label distribution:")
    print(train_df["label_name"].value_counts())

    print("\nValidation label distribution:")
    print(validation_df["label_name"].value_counts())

    print("\nTest label distribution:")
    print(test_df["label_name"].value_counts())

    print("\nTrain language distribution:")
    print(train_df["language"].value_counts())


if __name__ == "__main__":
    main()
