from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.data.preprocess import build_pair_text, encode_label


DATASET_NAME = "facebook/xnli"


def _normalize_language_code(language: str) -> str:
    """
    Normalize language code for XNLI.
    """
    language = language.strip().lower()

    aliases = {
        "english": "en",
        "turkish": "tr",
        "german": "de",
        "french": "fr",
        "arabic": "ar",
        "spanish": "es",
        "russian": "ru",
        "chinese": "zh",
        "hindi": "hi",
        "urdu": "ur",
        "bulgarian": "bg",
        "greek": "el",
        "swahili": "sw",
        "thai": "th",
        "vietnamese": "vi",
    }

    return aliases.get(language, language)


def _extract_text(value: Any, language: str) -> str:
    """
    Extract text robustly from different XNLI schemas.

    Some XNLI versions return strings directly.
    Some Hub parquet versions may return dictionaries like:
      {"en": "...", "tr": "..."}
    or:
      {"language": ["en", "tr"], "translation": ["...", "..."]}
    """
    language = _normalize_language_code(language)

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        if language in value and isinstance(value[language], str):
            return value[language]

        if "translation" in value and "language" in value:
            languages = value["language"]
            translations = value["translation"]

            if isinstance(languages, list) and isinstance(translations, list):
                for lang, translation in zip(languages, translations):
                    if str(lang).lower() == language:
                        return str(translation)

            if isinstance(languages, str) and languages.lower() == language:
                return str(translations)

        if "translation" in value and isinstance(value["translation"], str):
            return value["translation"]

    return str(value)


def _load_dataset_with_fallback(language: str, split: str):
    """
    Load XNLI from the stable namespaced repo first.

    If this fails for any reason, try the older short-name version as fallback.
    """
    language = _normalize_language_code(language)

    try:
        return load_dataset(DATASET_NAME, language, split=split)
    except Exception as first_error:
        print(
            f"Warning: Could not load {DATASET_NAME}, language={language}, split={split}. "
            f"Trying fallback dataset name 'xnli'. Reason: {first_error}"
        )

        try:
            return load_dataset("xnli", language, split=split)
        except Exception as second_error:
            raise RuntimeError(
                f"Could not load XNLI for language={language}, split={split}. "
                f"First error with {DATASET_NAME}: {first_error}. "
                f"Second error with xnli: {second_error}"
            ) from second_error


def _load_xnli_split_for_language(language: str, split: str) -> pd.DataFrame:
    """
    Load one language and one split from XNLI.
    """
    language = _normalize_language_code(language)

    dataset = _load_dataset_with_fallback(language=language, split=split)

    rows = []

    for idx, item in enumerate(dataset):
        premise = _extract_text(item.get("premise"), language)
        hypothesis = _extract_text(item.get("hypothesis"), language)

        if premise == "" or hypothesis == "":
            continue

        label_id = int(item["label"])

        rows.append(
            {
                "id": f"{language}_{split}_{idx}",
                "language": language,
                "premise": premise,
                "hypothesis": hypothesis,
                "text": build_pair_text(premise, hypothesis),
                "label_id": label_id,
                "label_name": encode_label(label_id),
                "source_split": split,
            }
        )

    if not rows:
        raise RuntimeError(
            f"Loaded dataset but extracted zero rows for language={language}, split={split}."
        )

    return pd.DataFrame(rows)


def _sample_per_language(
    df: pd.DataFrame,
    max_samples_per_language: int | None,
    seed: int,
) -> pd.DataFrame:
    """
    Sample each language while keeping labels approximately balanced.
    """
    if max_samples_per_language is None or max_samples_per_language <= 0:
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    sampled_parts = []

    for language, lang_df in df.groupby("language"):
        if len(lang_df) <= max_samples_per_language:
            sampled_parts.append(lang_df)
            continue

        label_count = lang_df["label_id"].nunique()
        per_label = max(1, max_samples_per_language // label_count)

        label_parts = []

        for _, label_df in lang_df.groupby("label_id"):
            take_n = min(len(label_df), per_label)
            label_parts.append(label_df.sample(n=take_n, random_state=seed))

        sampled_lang_df = pd.concat(label_parts, ignore_index=False)

        remaining = max_samples_per_language - len(sampled_lang_df)

        if remaining > 0:
            rest = lang_df.drop(index=sampled_lang_df.index)
            if len(rest) > 0:
                sampled_lang_df = pd.concat(
                    [
                        sampled_lang_df,
                        rest.sample(n=min(remaining, len(rest)), random_state=seed),
                    ],
                    ignore_index=False,
                )

        sampled_parts.append(sampled_lang_df)

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    return sampled_df.sample(frac=1, random_state=seed).reset_index(drop=True)


def _safe_stratify_key(df: pd.DataFrame):
    """
    Build stratification key using language + label.
    Return None if stratification is unsafe.
    """
    if df.empty:
        return None

    key = df["language"].astype(str) + "_" + df["label_id"].astype(str)
    counts = key.value_counts()

    if counts.min() < 2:
        return None

    return key


def _split_dataframe(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """
    Split multilingual XNLI pool into project-level train / validation / test.
    """
    total = train_ratio + validation_ratio + test_ratio

    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total}. "
            f"train={train_ratio}, validation={validation_ratio}, test={test_ratio}"
        )

    if df.empty:
        raise ValueError("Cannot split an empty dataframe.")

    stratify_key = _safe_stratify_key(df)

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        shuffle=True,
        stratify=stratify_key,
    )

    temp_stratify_key = _safe_stratify_key(temp_df)

    validation_size_inside_temp = validation_ratio / (validation_ratio + test_ratio)

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - validation_size_inside_temp),
        random_state=seed,
        shuffle=True,
        stratify=temp_stratify_key,
    )

    return {
        "train": train_df.reset_index(drop=True),
        "validation": validation_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def load_xnli_dataframe(
    languages: Iterable[str] = ("en", "tr"),
    source_splits: Iterable[str] = ("validation", "test"),
    max_samples_per_language: int | None = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load multilingual XNLI data into one dataframe.
    """
    all_parts = []

    for language in languages:
        language = _normalize_language_code(language)

        for split in source_splits:
            try:
                split_df = _load_xnli_split_for_language(language, split)
                all_parts.append(split_df)
                print(
                    f"Loaded XNLI dataset={DATASET_NAME}, "
                    f"language={language}, split={split}, rows={len(split_df)}"
                )
            except Exception as exc:
                print(
                    f"Warning: Could not load language={language}, split={split}. "
                    f"Reason: {exc}"
                )

    if not all_parts:
        raise RuntimeError(
            "No XNLI data was loaded. "
            "Check internet connection, Hugging Face packages, and dataset name."
        )

    df = pd.concat(all_parts, ignore_index=True)

    df = _sample_per_language(
        df=df,
        max_samples_per_language=max_samples_per_language,
        seed=seed,
    )

    expected_columns = [
        "id",
        "language",
        "premise",
        "hypothesis",
        "text",
        "label_id",
        "label_name",
        "source_split",
    ]

    return df[expected_columns].reset_index(drop=True)


def load_xnli_project_splits(
    languages: Iterable[str] = ("en", "tr"),
    source_splits: Iterable[str] = ("validation", "test"),
    max_samples_per_language: int | None = 500,
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """
    Load XNLI and create project-level train / validation / test CSV-ready splits.
    """
    df = load_xnli_dataframe(
        languages=languages,
        source_splits=source_splits,
        max_samples_per_language=max_samples_per_language,
        seed=seed,
    )

    return _split_dataframe(
        df=df,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )