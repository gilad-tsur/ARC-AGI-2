"""Utility functions for ARC-AGI-2.

This module provides helper functions for padding and
visualization of ARC grids.
"""
from __future__ import annotations

from typing import List, Tuple
import matplotlib.pyplot as plt
import torch

PAD_VALUE = 10
MAX_SIZE = 30


def pad_grid(grid: List[List[int]], pad_value: int = PAD_VALUE, size: int = MAX_SIZE) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a grid to ``size`` using ``pad_value``.

    Parameters
    ----------
    grid: List of rows containing integer color values.
    pad_value: Value to use for padding.
    size: Final size of each side of the padded grid.

    Returns
    -------
    tuple
        ``(padded_grid, mask)`` where ``padded_grid`` is a ``size x size``
        tensor and ``mask`` is a binary tensor with ``1`` in valid cells.
    """
    h, w = len(grid), len(grid[0])
    if h > size or w > size:
        raise ValueError(f"Grid too large: {h}x{w} > {size}x{size}")

    padded = torch.full((size, size), pad_value, dtype=torch.long)
    mask = torch.zeros((size, size), dtype=torch.bool)
    padded[:h, :w] = torch.tensor(grid, dtype=torch.long)
    mask[:h, :w] = 1
    return padded, mask


def unpad_grid(grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Remove padding from a grid using the provided mask."""
    indices = mask.nonzero(as_tuple=False)
    if indices.numel() == 0:
        return torch.empty(0, 0, dtype=grid.dtype)
    h = indices[:, 0].max().item() + 1
    w = indices[:, 1].max().item() + 1
    return grid[:h, :w]


def visualize_grid(grid: torch.Tensor) -> None:
    """Display an ARC grid using matplotlib."""
    plt.imshow(grid.cpu().numpy(), interpolation="nearest", vmin=0, vmax=10)
    plt.axis("off")
    plt.show()
