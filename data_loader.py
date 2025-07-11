"""Data loading utilities for the ARC-AGI-2 dataset."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from utils import pad_grid, PAD_VALUE, MAX_SIZE


class ARCDataset(Dataset):
    """Dataset for ARC-AGI-2 tasks.

    Parameters
    ----------
    root : str
        Root directory of the repository containing the ``data`` folder.
    split : {"training", "evaluation"}
        Dataset split to load.
    mode : {"train", "test"}
        Whether to load the training or test examples for each task.
    preload : bool
        If ``True``, all task files are loaded in memory at initialization.
    """

    def __init__(self, root: str, split: str = "training", mode: str = "train", *, preload: bool = False) -> None:
        self.data_dir = Path(root) / "data" / split
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.data_dir}")
        self.mode = mode
        self.task_files = sorted(self.data_dir.glob("*.json"))
        self.preload = preload
        self.tasks: Dict[str, Dict] = {}
        if preload:
            for path in self.task_files:
                with open(path) as f:
                    self.tasks[path.stem] = json.load(f)

        # Build index of (task_id, pair_index)
        self.index: List[Tuple[str, int]] = []
        for path in self.task_files:
            task_id = path.stem
            data = self.tasks.get(task_id)
            if data is None:
                with open(path) as f:
                    data = json.load(f)
            pairs = data[self.mode]
            for i in range(len(pairs)):
                self.index.append((task_id, i))

    def __len__(self) -> int:  # noqa: D401
        """Return dataset length."""
        return len(self.index)

    def _load_task(self, task_id: str) -> Dict:
        if task_id in self.tasks:
            return self.tasks[task_id]
        with open(self.data_dir / f"{task_id}.json") as f:
            data = json.load(f)
        if self.preload:
            self.tasks[task_id] = data
        return data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        task_id, pair_idx = self.index[idx]
        data = self._load_task(task_id)
        pair = data[self.mode][pair_idx]
        inp, in_mask = pad_grid(pair["input"], pad_value=PAD_VALUE, size=MAX_SIZE)
        out, out_mask = pad_grid(pair["output"], pad_value=PAD_VALUE, size=MAX_SIZE)
        return inp, out, in_mask, out_mask, task_id


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]]):
    """Collate a batch of samples."""
    inputs, outputs, in_masks, out_masks, task_ids = zip(*batch)
    return (
        torch.stack(inputs),
        torch.stack(outputs),
        torch.stack(in_masks),
        torch.stack(out_masks),
        list(task_ids),
    )
