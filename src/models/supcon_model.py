from __future__ import annotations

import torch
from torch import nn

from src.models.mbert_encoder import MBertEncoder
from src.models.projection_head import ProjectionHead


class SupConMBertModel(nn.Module):
    """mBERT encoder plus projection head for supervised contrastive learning."""

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        pooling: str = "mean",
        encoder_hidden_size: int = 768,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = MBertEncoder(model_name=model_name, pooling=pooling)
        self.projection_head = ProjectionHead(
            input_dim=encoder_hidden_size,
            projection_dim=projection_dim,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_embeddings = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        projected_embeddings = self.projection_head(encoder_embeddings)
        return encoder_embeddings, projected_embeddings
