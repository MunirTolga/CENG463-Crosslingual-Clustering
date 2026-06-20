from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    """Small projection head for supervised contrastive learning."""

    def __init__(
        self,
        input_dim: int = 768,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        projections = self.layers(embeddings)
        return F.normalize(projections, dim=-1)
