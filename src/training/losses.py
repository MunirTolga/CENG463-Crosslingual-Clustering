from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss for one embedding view per sample."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [batch_size, projection_dim].")
        if labels.ndim != 1:
            raise ValueError("labels must have shape [batch_size].")
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must have the same batch size.")

        features = F.normalize(features, dim=1)
        labels = labels.view(-1, 1)
        batch_size = features.shape[0]

        # Positive pairs share the same label. The diagonal is removed because a
        # sample should not be treated as a positive pair with itself.
        positive_mask = torch.eq(labels, labels.T).float().to(features.device)
        logits_mask = torch.ones_like(positive_mask) - torch.eye(batch_size, device=features.device)
        positive_mask = positive_mask * logits_mask

        similarity = torch.matmul(features, features.T) / self.temperature
        similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()

        exp_similarity = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_similarity.sum(dim=1, keepdim=True).clamp_min(1e-12))

        positive_counts = positive_mask.sum(dim=1)
        valid_anchors = positive_counts > 0

        if not torch.any(valid_anchors):
            return features.sum() * 0.0

        mean_log_prob_positive = (
            (positive_mask * log_prob).sum(dim=1)[valid_anchors]
            / positive_counts[valid_anchors]
        )

        return -mean_log_prob_positive.mean()
