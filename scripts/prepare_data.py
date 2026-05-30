from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.load_xnli import load_xnli_project_splits
from src.data.preprocess import save_dataframe


def _print_split_summary(name: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print(f"{name.upper()} SPLIT")
    print("=" * 80)
    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(list(df.columns))

    print("\nLanguage distribution:")
    print(df["language"].value_counts())

    print("\nLabel distribution:")
    print(df["label_name"].value_counts())

    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare XNLI train / validation / test CSV files."
    )

    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "tr"],
        help="Languages to load. Example: --languages en tr",
    )

    parser.add_argument(
        "--source_splits",
        "--source-splits",
        dest="source_splits",
        nargs="+",
        default=["validation", "test"],
        help="Original XNLI splits used as data pool.",
    )

    parser.add_argument(
        "--max_samples_per_language",
        "--max-samples-per-language",
        dest="max_samples_per_language",
        type=int,
        default=500,
        help="Maximum number of samples per language before project split.",
    )

    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="data/processed",
        help="Directory where processed CSV files will be saved.",
    )

    parser.add_argument(
        "--split",
        dest="single_split",
        choices=["train", "validation", "test", "dev"],
        default=None,
        help="Backward-compatible alias for selecting a single source split.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Train split ratio.",
    )

    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.single_split is not None:
        args.source_splits = [_normalize_source_split(args.single_split)]
    else:
        args.source_splits = [_normalize_source_split(split) for split in args.source_splits]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    languages = [lang.lower() for lang in args.languages]
    language_tag = "_".join(languages)

    print("Preparing XNLI dataset...")
    print(f"Languages: {languages}")
    print(f"Source splits: {args.source_splits}")
    print(f"Max samples per language: {args.max_samples_per_language}")
    print(f"Output directory: {output_dir}")

    splits = load_xnli_project_splits(
        languages=languages,
        source_splits=args.source_splits,
        max_samples_per_language=args.max_samples_per_language,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )

    train_df = splits["train"]
    validation_df = splits["validation"]
    test_df = splits["test"]

    # Standard filenames expected by the rest of the project
    save_dataframe(train_df, output_dir / "xnli_train.csv")
    save_dataframe(validation_df, output_dir / "xnli_validation.csv")
    save_dataframe(validation_df, output_dir / "xnli_valid.csv")
    save_dataframe(test_df, output_dir / "xnli_test.csv")

    # Language-tagged filenames for clarity and reproducibility
    save_dataframe(train_df, output_dir / f"xnli_train_{language_tag}.csv")
    save_dataframe(validation_df, output_dir / f"xnli_validation_{language_tag}.csv")
    save_dataframe(validation_df, output_dir / f"xnli_valid_{language_tag}.csv")
    save_dataframe(test_df, output_dir / f"xnli_test_{language_tag}.csv")

    print("\nSaved processed files:")
    print(output_dir / "xnli_train.csv")
    print(output_dir / "xnli_validation.csv")
    print(output_dir / "xnli_valid.csv")
    print(output_dir / "xnli_test.csv")
    print(output_dir / f"xnli_train_{language_tag}.csv")
    print(output_dir / f"xnli_validation_{language_tag}.csv")
    print(output_dir / f"xnli_test_{language_tag}.csv")

    _print_split_summary("train", train_df)
    _print_split_summary("validation", validation_df)
    _print_split_summary("test", test_df)


def _normalize_source_split(split: str) -> str:
    split = split.strip().lower()
    if split == "dev":
        return "validation"
    return split


if __name__ == "__main__":
    main()
