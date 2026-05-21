# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Dental lesion ablation modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class SPDConv(nn.Module):
    """Space-to-depth downsampling followed by stride-1 Conv."""

    def __init__(self, c1: int, c2: int, k: int = 3):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2))
        x = torch.cat(
            (
                x[..., 0::2, 0::2],
                x[..., 1::2, 0::2],
                x[..., 0::2, 1::2],
                x[..., 1::2, 1::2],
            ),
            dim=1,
        )
        return self.conv(x)
