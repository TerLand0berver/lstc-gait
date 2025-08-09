from __future__ import annotations

from typing import Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F


class LocalSpatioTemporalConv(nn.Module):
    """
    Local spatio-temporal convolution (LSTC).

    Goal:
    - Inject a stripe-local prior (height-wise local receptive field) while
      jointly convolving along temporal and spatial dimensions.
    - Support per-branch asymmetric kernels for time (T), width (W), height (H).

    Input shape: (N, C, T, H, W)
    Output shape: (N, C_out, T', H', W')

    Key ideas:
    - Factorize a 3D conv into depthwise local stripes along H, then pointwise mix.
    - Use group conv with groups=C to do channelwise localized 3D conv inside stripes.
    - Stripes are implemented via unfolding and masked conv to avoid overlap bleeding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_t: int = 3,
        kernel_h: int = 7,
        kernel_w: int = 3,
        stride_t: int = 1,
        stride_h: int = 1,
        stride_w: int = 1,
        padding_t: Optional[int] = None,
        padding_h: Optional[int] = None,
        padding_w: Optional[int] = None,
        num_stripes: int = 8,
        bias: bool = False,
        norm: bool = True,
        activation: bool = True,
    ) -> None:
        super().__init__()
        assert in_channels > 0 and out_channels > 0
        assert num_stripes >= 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stripes = num_stripes

        # Default to 'same' padding for odd kernels
        kt = kernel_t
        kh = kernel_h
        kw = kernel_w
        pt = (kt // 2) if padding_t is None else padding_t
        ph = (kh // 2) if padding_h is None else padding_h
        pw = (kw // 2) if padding_w is None else padding_w

        self.kernel_size: Tuple[int, int, int] = (kt, kh, kw)
        self.stride: Tuple[int, int, int] = (stride_t, stride_h, stride_w)
        self.padding: Tuple[int, int, int] = (pt, ph, pw)

        # Depthwise 3D conv per-channel, will be applied per-stripe via masking
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=(kt, kh, kw),
            stride=(stride_t, stride_h, stride_w),
            padding=(pt, ph, pw),
            groups=in_channels,
            bias=bias,
        )
        # Pointwise mixing across channels
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.bn = nn.BatchNorm3d(out_channels) if norm else nn.Identity()
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, h, w = x.shape
        stripe_height = max(1, h // self.num_stripes)
        outputs = []

        for s in range(self.num_stripes):
            h_start = s * stripe_height
            h_end = h if s == self.num_stripes - 1 else min(h, h_start + stripe_height)
            # Crop local stripe region
            x_local = x[:, :, :, h_start:h_end, :]
            # Apply depthwise conv within stripe
            y_local = self.depthwise(x_local)
            # Place back into a zero canvas matching the conv output height
            # Compute output spatial/temporal sizes using conv formula
            t_out = (t + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
            h_local = x_local.shape[3]
            h_out_local = (h_local + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
            w_out = (w + 2 * self.padding[2] - self.kernel_size[2]) // self.stride[2] + 1

            if s == 0:
                canvas = x.new_zeros((n, c, t_out, 0, w_out))
            # Concat along H to reconstruct
            outputs.append(y_local)

        y_depthwise = torch.cat(outputs, dim=3)
        # Pointwise channel mixing
        y = self.pointwise(y_depthwise)
        y = self.bn(y)
        y = self.act(y)
        return y


class AsymmetricSpatioTemporalBlock(nn.Module):
    """
    Three-branch asymmetric conv block:
    - Temporal-only: kernel (kT,1,1)
    - Spatial-only: kernel (1,kH,kW)
    - Joint LSTC: full (kT,kH,kW) with local stripes
    Outputs are fused by concatenation + 1x1x1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kT: int = 3,
        kH: int = 7,
        kW: int = 3,
        num_stripes: int = 8,
        bias: bool = False,
        use_temporal: bool = True,
        use_spatial: bool = True,
        use_joint: bool = True,
    ) -> None:
        super().__init__()
        enabled = [use_temporal, use_spatial, use_joint]
        num_enabled = max(1, sum(1 for e in enabled if e))
        mid_per = max(out_channels // (2 * num_enabled), 8)

        self.use_temporal = use_temporal
        self.use_spatial = use_spatial
        self.use_joint = use_joint

        if use_temporal:
            self.branch_temporal = nn.Sequential(
                nn.Conv3d(in_channels, mid_per, kernel_size=(kT, 1, 1), padding=(kT // 2, 0, 0), bias=bias),
                nn.BatchNorm3d(mid_per),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch_temporal = None
        if use_spatial:
            self.branch_spatial = nn.Sequential(
                nn.Conv3d(in_channels, mid_per, kernel_size=(1, kH, kW), padding=(0, kH // 2, kW // 2), bias=bias),
                nn.BatchNorm3d(mid_per),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch_spatial = None
        if use_joint:
            self.branch_joint = LocalSpatioTemporalConv(
                in_channels=in_channels,
                out_channels=mid_per,
                kernel_t=kT,
                kernel_h=kH,
                kernel_w=kW,
                num_stripes=num_stripes,
                bias=bias,
            )
        else:
            self.branch_joint = None
        fuse_in = mid_per * num_enabled
        self.fuse = nn.Sequential(
            nn.Conv3d(fuse_in, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        if self.use_temporal and self.branch_temporal is not None:
            feats.append(self.branch_temporal(x))
        if self.use_spatial and self.branch_spatial is not None:
            feats.append(self.branch_spatial(x))
        if self.use_joint and self.branch_joint is not None:
            feats.append(self.branch_joint(x))
        y = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]
        return self.fuse(y)


class LocalSpatioTemporalPooling(nn.Module):
    """
    Local Spatio-Temporal Pooling (LSTP):
    - Split feature map into height stripes
    - For each stripe, perform top-k pooling across time to capture most discriminative frames
    - Then spatial average within stripe to get a compact descriptor per stripe
    - Finally concatenate all stripes to form sequence-level representation

    Input: (N, C, T, H, W)
    Output: (N, C * num_stripes)
    """

    def __init__(self, num_stripes: int = 8, topk: int = 2, eps: float = 1e-6, soft: bool = False, temperature: float = 1.0) -> None:
        super().__init__()
        assert num_stripes >= 1 and topk >= 1
        self.num_stripes = num_stripes
        self.topk = topk
        self.eps = eps
        self.soft = soft
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, h, w = x.shape
        stripe_height = max(1, h // self.num_stripes)
        stripe_features = []

        for s in range(self.num_stripes):
            h_start = s * stripe_height
            h_end = h if s == self.num_stripes - 1 else min(h, h_start + stripe_height)
            x_local = x[:, :, :, h_start:h_end, :]
            # Spatial average per frame in this stripe -> (N, C, T)
            frame_feats = x_local.mean(dim=[3, 4])
            # Score frames by L2 magnitude over channels
            scores = torch.sqrt(torch.clamp((frame_feats ** 2).sum(dim=1), min=self.eps))  # (N, T)
            if self.soft:
                # Soft top-k: temperature-scaled softmax weights across time
                weights = torch.softmax(scores / max(self.eps, self.temperature), dim=1)  # (N, T)
                stripe_vec = torch.bmm(frame_feats, weights.unsqueeze(2)).squeeze(2)  # (N, C)
            else:
                k = min(self.topk, t)
                topk_vals, topk_idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=False)
                gather_idx = topk_idx.unsqueeze(1).expand(-1, c, -1)  # (N, C, k)
                topk_frames = torch.gather(frame_feats, dim=2, index=gather_idx)
                stripe_vec = topk_frames.mean(dim=2)  # (N, C)
            stripe_features.append(stripe_vec)

        out = torch.cat(stripe_features, dim=1)  # (N, C * num_stripes)
        return out
