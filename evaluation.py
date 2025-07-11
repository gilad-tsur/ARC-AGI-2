"""Evaluation utilities."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict

import json

import torch

from data_loader import ARCDataset
from models import ARCModel, masked_cross_entropy
from task_embeddings import TaskEmbeddingModule
from utils import pad_grid


def evaluate_task(
    model: ARCModel,
    embeddings: TaskEmbeddingModule,
    task_file: Path,
    *,
    adapt_steps: int = 50,
    adapt_lr: float = 1e-2,
    device: torch.device | None = None,
) -> Dict[str, float]:
    """Evaluate a single task file with adaptation."""
    device = device or torch.device("cpu")
    with open(task_file) as f:
        data = json.load(f)
    examples = []
    for pair in data["train"]:
        inp, in_mask = pad_grid(pair["input"])
        out, out_mask = pad_grid(pair["output"])
        examples.append((inp, out, in_mask, out_mask))
    task_id = task_file.stem
    adapted = embeddings.adapt_embedding(task_id, model, examples, steps=adapt_steps, lr=adapt_lr, device=device)
    metrics = defaultdict(float)
    count = 0
    for pair in data["test"]:
        inp, in_mask = pad_grid(pair["input"])
        out, out_mask = pad_grid(pair["output"])
        inp = inp.unsqueeze(0).to(device)
        in_mask = in_mask.unsqueeze(0).to(device)
        out = out.unsqueeze(0).to(device)
        out_mask = out_mask.unsqueeze(0).to(device)
        logits, pred_mask = model(inp, adapted, in_mask)
        loss_grid = masked_cross_entropy(logits, out, out_mask)
        loss_mask = torch.nn.functional.binary_cross_entropy(pred_mask, out_mask.float())
        metrics["loss"] += (loss_grid + loss_mask).item()
        count += 1
    metrics = {k: v / count for k, v in metrics.items()}
    return metrics
