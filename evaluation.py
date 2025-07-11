"""Evaluation utilities."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import json

import torch

from models import ARCModel, masked_cross_entropy
from task_embeddings import TaskEmbeddingModule
from utils import pad_grid, unpad_grid


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


def predict_task(
    model: ARCModel,
    embeddings: TaskEmbeddingModule,
    task_file: Path,
    *,
    adapt_steps: int = 50,
    adapt_lr: float = 1e-2,
    device: torch.device | None = None,
) -> List[List[List[int]]]:
    """Return predictions for the test pairs of ``task_file`` after adaptation."""
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
    preds = []
    for pair in data["test"]:
        inp, in_mask = pad_grid(pair["input"])
        inp = inp.unsqueeze(0).to(device)
        in_mask = in_mask.unsqueeze(0).to(device)
        logits, pred_mask = model(inp, adapted, in_mask)
        grid = logits.argmax(dim=1).squeeze(0)
        mask = pred_mask.squeeze(0) > 0.5
        pred = unpad_grid(grid, mask)
        preds.append(pred.cpu().tolist())
    return preds


def evaluate_split(
    model: ARCModel,
    embeddings: TaskEmbeddingModule,
    root: str,
    *,
    split: str = "evaluation",
    adapt_steps: int = 50,
    adapt_lr: float = 1e-2,
    device: torch.device | None = None,
) -> Dict[str, float]:
    """Evaluate ``model`` on all tasks of ``split``."""
    device = device or torch.device("cpu")
    data_dir = Path(root) / "data" / split
    task_files = sorted(data_dir.glob("*.json"))
    agg = defaultdict(float)
    for task in task_files:
        metrics = evaluate_task(
            model,
            embeddings,
            task,
            adapt_steps=adapt_steps,
            adapt_lr=adapt_lr,
            device=device,
        )
        for k, v in metrics.items():
            agg[k] += v
    return {k: v / len(task_files) for k, v in agg.items()}


def score_split(
    model: ARCModel,
    embeddings: TaskEmbeddingModule,
    root: str,
    *,
    split: str = "evaluation",
    adapt_steps: int = 50,
    adapt_lr: float = 1e-2,
    device: torch.device | None = None,
) -> float:
    """Return exact prediction accuracy on ``split``."""
    device = device or torch.device("cpu")
    data_dir = Path(root) / "data" / split
    task_files = sorted(data_dir.glob("*.json"))
    correct = 0
    total = 0
    for task in task_files:
        preds = predict_task(
            model,
            embeddings,
            task,
            adapt_steps=adapt_steps,
            adapt_lr=adapt_lr,
            device=device,
        )
        with open(task) as f:
            data = json.load(f)
        targets = [pair["output"] for pair in data["test"]]
        for p, t in zip(preds, targets):
            total += 1
            if p == t:
                correct += 1
    if total == 0:
        return 0.0
    return correct / total

