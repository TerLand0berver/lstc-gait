from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    """
    Batch-hard triplet loss (Hermans et al., 2017):
    - For each anchor, choose the hardest positive (max distance within same class)
      and the hardest negative (min distance across different classes),
      then compute hinge loss: max(0, d(ap) - d(an) + margin).
    Expects embeddings normalized or not; cosine or Euclidean distance supported.
    """

    def __init__(self, margin: float = 0.3, metric: str = "euclidean") -> None:
        super().__init__()
        assert metric in {"euclidean", "cosine"}
        self.margin = margin
        self.metric = metric

    @staticmethod
    def _pairwise_distances(emb: torch.Tensor, metric: str) -> torch.Tensor:
        if metric == "euclidean":
            # squared Euclidean distance
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
            sq = torch.sum(emb ** 2, dim=1, keepdim=True)
            dist = sq + sq.t() - 2.0 * emb @ emb.t()
            dist = torch.clamp(dist, min=0.0)
            return torch.sqrt(dist + 1e-12)
        else:  # cosine distance = 1 - cosine similarity
            emb = F.normalize(emb, dim=1)
            sim = emb @ emb.t()
            return 1.0 - sim

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: (N, D), labels: (N)
        with torch.no_grad():
            mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
            mask_neg = ~mask_pos
            # Exclude self-comparisons from positives
            eye = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
            mask_pos = mask_pos & ~eye

        dist = self._pairwise_distances(embeddings, metric=self.metric)

        # hardest positive per anchor: max distance among positives
        dist_pos = dist.clone()
        dist_pos[~mask_pos] = -1.0  # ignore non-positives
        hardest_pos, _ = dist_pos.max(dim=1)

        # hardest negative per anchor: min distance among negatives
        dist_neg = dist.clone()
        dist_neg[~mask_neg] = float("inf")
        hardest_neg, _ = dist_neg.min(dim=1)

        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()
