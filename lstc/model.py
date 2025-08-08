from __future__ import annotations

from typing import Tuple
import torch
from torch import nn

from .modules import AsymmetricSpatioTemporalBlock, LocalSpatioTemporalPooling


class LSTCBackbone(nn.Module):
    """
    Reference backbone to learn gait features with local spatio-temporal priors.

    Input: (N, C_in, T, H, W)
    Output: dict with
      - feat_map: (N, C_feat, T', H', W') feature map
      - embedding: (N, D) sequence-level descriptor
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_stripes: int = 8,
        embedding_dim: int = 256,
        use_temporal: bool = True,
        use_spatial: bool = True,
        use_joint: bool = True,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # Save init kwargs for potential EMA cloning
        self.init_kwargs = dict(
            in_channels=in_channels,
            base_channels=base_channels,
            num_stripes=num_stripes,
            embedding_dim=embedding_dim,
            use_temporal=use_temporal,
            use_spatial=use_spatial,
            use_joint=use_joint,
        )

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
        )
        self.block1 = AsymmetricSpatioTemporalBlock(
            c1, c1, kT=3, kH=7, kW=3, num_stripes=num_stripes,
            use_temporal=use_temporal, use_spatial=use_spatial, use_joint=use_joint,
        )
        self.down1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.block2 = AsymmetricSpatioTemporalBlock(
            c1, c2, kT=3, kH=5, kW=3, num_stripes=num_stripes,
            use_temporal=use_temporal, use_spatial=use_spatial, use_joint=use_joint,
        )
        self.down2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.block3 = AsymmetricSpatioTemporalBlock(
            c2, c3, kT=3, kH=3, kW=3, num_stripes=num_stripes,
            use_temporal=use_temporal, use_spatial=use_spatial, use_joint=use_joint,
        )

        self.head = nn.Sequential(
            nn.Conv3d(c3, c3, kernel_size=1, bias=False),
            nn.BatchNorm3d(c3),
            nn.ReLU(inplace=True),
        )

        self.pool = LocalSpatioTemporalPooling(num_stripes=num_stripes, topk=2)
        self.fc = nn.Linear(c3 * num_stripes, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> dict:
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        fmap = self.block3(x)
        fmap = self.head(fmap)
        emb = self.pool(fmap)
        emb = self.fc(emb)
        emb = self.bn(emb)
        return {"feat_map": fmap, "embedding": emb}
