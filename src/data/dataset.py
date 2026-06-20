from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocess import build_pair_text


class XNLISupConDataset(Dataset):
    """PyTorch dataset for supervised contrastive XNLI training."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._validate_columns()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]
        text = build_pair_text(row["premise"], row["hypothesis"])
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label_id": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "language": row["language"],
            "text": text,
        }

    def _validate_columns(self) -> None:
        required_columns = {"premise", "hypothesis", "label_id"}
        missing_columns = sorted(required_columns - set(self.dataframe.columns))
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        if "language" not in self.dataframe.columns:
            self.dataframe["language"] = "unknown"

def xnli_supcon_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate XNLI supervised contrastive dataset samples."""

    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "label_id": torch.stack([item["label_id"] for item in batch]),
        "language": [item["language"] for item in batch],
        "text": [item["text"] for item in batch],
    }
