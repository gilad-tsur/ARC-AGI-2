"""Model architectures for ARC-AGI-2."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from utils import PAD_VALUE


class ARCModel(nn.Module):
    """Simple convolutional model for ARC tasks."""

    def __init__(self, num_colors: int = 11, embed_dim: int = 32, task_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.color_emb = nn.Embedding(num_colors, embed_dim)
        in_channels = embed_dim + task_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )
        self.out_grid = nn.Conv2d(hidden_dim, num_colors, 1)
        self.out_mask = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, inp: torch.Tensor, task_emb: torch.Tensor, input_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        inp : torch.Tensor
            Tensor of shape ``(B, H, W)`` containing integer colors.
        task_emb : torch.Tensor
            Tensor of shape ``(B, task_dim)``.
        input_mask : torch.Tensor
            Binary tensor of shape ``(B, H, W)`` indicating valid input cells.
        """
        x = self.color_emb(inp.long())  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        input_mask = input_mask.unsqueeze(1).float()
        x = x * input_mask  # zero-out padded areas
        task = task_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, task], dim=1)
        x = self.conv(x)
        grid_logits = self.out_grid(x)
        mask_logits = self.out_mask(x)
        mask = torch.sigmoid(mask_logits)
        return grid_logits, mask.squeeze(1)


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross entropy loss that ignores padded regions."""
    loss = F.cross_entropy(logits.permute(0, 2, 3, 1).reshape(-1, logits.size(1)), target.view(-1), reduction="none")
    loss = loss * mask.view(-1).float()
    return loss.sum() / mask.sum().clamp(min=1)
