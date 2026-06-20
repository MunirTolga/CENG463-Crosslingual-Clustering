from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.load_xnli import load_xnli_dataframe, load_xnli_project_splits
from src.data.preprocess import save_dataframe


EXPECTED_COLUMNS = [
    "id",
    "language",
    "premise",
    "hypothesis",
    "text",
    "label_id",
    "label_name",
    "source_split",
]


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
    print(_value_counts_text(df, "label_name"))

    print("\nSource split distribution:")
    print(_value_counts_text(df, "source_split"))

    _warn_if_source_split_is_not_clean(name, df)

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
        default=["train", "validation"],
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

    parser.add_argument(
        "--random_split_fallback",
        "--random-split-fallback",
        action="store_true",
        help=(
            "Use the older fallback behavior: combine requested source splits and "
            "randomly split them into project train/validation/test files."
        ),
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

    if args.random_split_fallback:
        print("Using explicit random split fallback mode.")
        splits = _load_random_split_fallback(
            languages=languages,
            source_splits=args.source_splits,
            max_samples_per_language=args.max_samples_per_language,
            seed=args.seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
        )
    else:
        splits = _load_clean_source_splits(
            languages=languages,
            source_splits=args.source_splits,
            max_samples_per_language=args.max_samples_per_language,
            seed=args.seed,
        )

    train_df = splits["train"]
    validation_df = splits["validation"]
    test_df = splits["test"]

    _save_processed_splits(
        output_dir=output_dir,
        language_tag=language_tag,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
    )

    _print_split_summary("train", train_df)
    _print_split_summary("validation", validation_df)
    _print_split_summary("test", test_df)


def _load_clean_source_splits(
    languages: list[str],
    source_splits: list[str],
    max_samples_per_language: int | None,
    seed: int,
) -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}

    for output_split in ["train", "validation", "test"]:
        if output_split not in source_splits:
            print(
                f"Source split '{output_split}' was not requested. "
                f"Saving an empty {output_split} CSV."
            )
            splits[output_split] = _empty_xnli_dataframe()
            continue

        splits[output_split] = load_xnli_dataframe(
            languages=languages,
            source_splits=[output_split],
            max_samples_per_language=max_samples_per_language,
            seed=seed,
        )

    return splits


def _save_processed_splits(
    output_dir: Path,
    language_tag: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
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


def _empty_xnli_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=EXPECTED_COLUMNS)


def _value_counts_text(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return f"Column '{column}' is missing."
    if df.empty:
        return "Empty split."
    return df[column].value_counts().to_string()


def _warn_if_source_split_is_not_clean(name: str, df: pd.DataFrame) -> None:
    expected_source_split = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }.get(name)

    if expected_source_split is None or df.empty or "source_split" not in df.columns:
        return

    actual_values = set(df["source_split"].dropna().astype(str).str.lower())
    unexpected_values = sorted(actual_values - {expected_source_split})

    if unexpected_values:
        print(
            "WARNING: "
            f"{name} output contains source_split values other than "
            f"'{expected_source_split}': {unexpected_values}"
        )


def _load_random_split_fallback(
    languages: list[str],
    source_splits: list[str],
    max_samples_per_language: int | None,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> dict[str, pd.DataFrame]:
    return load_xnli_project_splits(
        languages=languages,
        source_splits=source_splits,
        max_samples_per_language=max_samples_per_language,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )


def _normalize_source_split(split: str) -> str:
    split = split.strip().lower()
    if split == "dev":
        return "validation"
    return split


if __name__ == "__main__":
    main()
