from pathlib import Path
from typing import Any

import pandas as pd


LABEL_ID_TO_NAME = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}


def clean_text(text: Any) -> str:
    """
    Clean text while preserving multilingual characters.
    Do not lowercase because mBERT is cased.
    """
    if text is None:
        return ""

    text = str(text)
    text = " ".join(text.split())
    return text.strip()


def build_pair_text(premise: Any, hypothesis: Any) -> str:
    """
    Combine premise and hypothesis into one sentence-pair text.
    """
    premise = clean_text(premise)
    hypothesis = clean_text(hypothesis)
    return f"{premise} [SEP] {hypothesis}"


def encode_label(label_id: Any) -> str:
    """
    Convert XNLI numeric label into readable label name.
    """
    label_id = int(label_id)

    if label_id not in LABEL_ID_TO_NAME:
        raise ValueError(f"Unknown XNLI label id: {label_id}")

    return LABEL_ID_TO_NAME[label_id]


def encode_labels(label_id: Any) -> str:
    """
    Backward-compatible alias for encode_label.
    """
    return encode_label(label_id)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def load_dataframe(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
