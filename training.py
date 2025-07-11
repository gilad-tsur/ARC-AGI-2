"""Training utilities."""
from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_loader import collate_fn
from data_loader import ARCDataset
from models import ARCModel, masked_cross_entropy
from task_embeddings import TaskEmbeddingModule


def train(
    model: ARCModel,
    task_embeddings: TaskEmbeddingModule,
    dataset: torch.utils.data.Dataset,
    *,
    epochs: int = 10,
    batch_size: int = 16,
    model_lr: float = 1e-3,
    emb_lr: float = 1e-2,
    device: torch.device | None = None,
) -> None:
    """Train ``model`` and ``task_embeddings`` jointly."""
    device = device or torch.device("cpu")
    model.to(device)
    task_embeddings.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    opt_model = Adam(model.parameters(), lr=model_lr)
    opt_emb = Adam(task_embeddings.parameters(), lr=emb_lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, outputs, in_masks, out_masks, task_ids in loader:
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            in_masks = in_masks.to(device)
            out_masks = out_masks.to(device)
            task_emb = task_embeddings(task_ids)
            opt_model.zero_grad()
            opt_emb.zero_grad()
            logits, pred_mask = model(inputs, task_emb, in_masks)
            loss_grid = masked_cross_entropy(logits, outputs, out_masks)
            loss_mask = torch.nn.functional.binary_cross_entropy(pred_mask, out_masks.float())
            loss = loss_grid + loss_mask
            loss.backward()
            opt_model.step()
            opt_emb.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg:.4f}")


def main() -> None:
    """Entry point for command line training."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Train ARC model")
    parser.add_argument("--root", default=".", help="Repository root containing data folder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset = ARCDataset(args.root, split="training", mode="train")
    task_ids = [p.stem for p in dataset.task_files]
    embeddings = TaskEmbeddingModule(task_ids)
    model = ARCModel()
    device = torch.device(args.device)
    train(
        model,
        embeddings,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
