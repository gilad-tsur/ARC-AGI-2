"""Training utilities."""
from __future__ import annotations

from typing import Iterable, Optional
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_loader import collate_fn
from data_loader import ARCDataset
from models import ARCModel, masked_cross_entropy
from task_embeddings import TaskEmbeddingModule
from evaluation import score_split


def train(
    model: ARCModel,
    task_embeddings: TaskEmbeddingModule,
    dataset: torch.utils.data.Dataset,
    val_dataset: Optional[torch.utils.data.Dataset] = None,
    *,
    epochs: int = 10,
    batch_size: int = 16,
    model_lr: float = 1e-3,
    emb_lr: float = 1e-2,
    device: torch.device | None = None,
    save_path: Optional[str] = None,
    eval_root: Optional[str] = None,
    eval_interval: int = 10,
) -> None:
    """Train ``model`` and ``task_embeddings`` jointly."""
    device = device or torch.device("cpu")
    model.to(device)
    task_embeddings.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
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
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, outputs, in_masks, out_masks, task_ids in val_loader:
                    inputs = inputs.to(device)
                    outputs = outputs.to(device)
                    in_masks = in_masks.to(device)
                    out_masks = out_masks.to(device)
                    task_emb = task_embeddings(task_ids)
                    logits, pred_mask = model(inputs, task_emb, in_masks)
                    loss_grid = masked_cross_entropy(logits, outputs, out_masks)
                    loss_mask = torch.nn.functional.binary_cross_entropy(pred_mask, out_masks.float())
                    val_loss += (loss_grid + loss_mask).item()
            val_avg = val_loss / len(val_loader)
            msg = f"Epoch {epoch+1}/{epochs} - loss: {avg:.4f} - val_loss: {val_avg:.4f}"
        else:
            msg = f"Epoch {epoch+1}/{epochs} - loss: {avg:.4f}"

        if eval_root is not None and (epoch + 1) % eval_interval == 0:
            eval_score = score_split(model, task_embeddings, eval_root, device=device)
            msg += f" - eval_score: {eval_score:.4f}"

        print(msg)

    if save_path is not None:
        torch.save(
            {
                "model": model.state_dict(),
                "embeddings": task_embeddings.state_dict(),
                "task_ids": task_embeddings.task_to_idx,
            },
            save_path,
        )


def main() -> None:
    """Entry point for command line training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train ARC model")
    parser.add_argument("--root", default=".", help="Repository root containing data folder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-to", default="arc_model.pt", help="Path to save trained model")
    args = parser.parse_args()

    dataset = ARCDataset(args.root, split="training", mode="train")
    val_dataset = ARCDataset(args.root, split="training", mode="test")
    train_ids = [p.stem for p in dataset.task_files]
    eval_ids = [p.stem for p in (Path(args.root) / "data" / "evaluation").glob("*.json")]
    task_ids = sorted(set(train_ids + eval_ids))
    embeddings = TaskEmbeddingModule(task_ids)
    model = ARCModel()
    device = torch.device(args.device)
    train(
        model,
        embeddings,
        dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        save_path=args.save_to,
        eval_root=args.root,
        eval_interval=10,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
