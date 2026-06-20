from __future__ import annotations

import torch
from torch import nn


class MBertEncoder(nn.Module):
    """mBERT encoder that returns sentence embeddings with simple pooling."""

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        pooling: str = "mean",
    ) -> None:
        super().__init__()

        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be either 'mean' or 'cls'.")

        from transformers import AutoModel

        self.model_name = model_name
        self.pooling = pooling
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling == "cls":
            return outputs.last_hidden_state[:, 0]

        return self._mean_pool(outputs.last_hidden_state, attention_mask)

    @staticmethod
    def _mean_pool(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed_embeddings = torch.sum(token_embeddings * mask, dim=1)
        token_counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed_embeddings / token_counts
