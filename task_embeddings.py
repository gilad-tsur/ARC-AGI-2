"""Task embedding management."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.optim import Adam

from models import masked_cross_entropy, ARCModel


class TaskEmbeddingModule(nn.Module):
    """Container for learnable embeddings for each task."""

    def __init__(self, task_ids: Iterable[str], embedding_dim: int = 32) -> None:
        super().__init__()
        self.task_to_idx: Dict[str, int] = {t: i for i, t in enumerate(sorted(task_ids))}
        self.embeddings = nn.Embedding(len(self.task_to_idx), embedding_dim)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)

    def forward(self, task_ids: List[str]) -> torch.Tensor:
        idx = torch.tensor([self.task_to_idx[t] for t in task_ids], dtype=torch.long, device=self.embeddings.weight.device)
        return self.embeddings(idx)

    def adapt_embedding(
        self,
        task_id: str,
        model: ARCModel,
        examples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        *,
        steps: int = 50,
        lr: float = 1e-2,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Adapt a task embedding on a few-shot set."""
        device = device or next(model.parameters()).device
        idx = self.task_to_idx[task_id]
        embedding = self.embeddings.weight[idx : idx + 1].clone().detach().to(device)
        embedding.requires_grad_(True)
        opt = Adam([embedding], lr=lr)
        model.eval()
        for _ in range(steps):
            opt.zero_grad()
            loss = 0.0
            for inp, out, in_mask, out_mask in examples:
                inp = inp.unsqueeze(0).to(device)
                out = out.unsqueeze(0).to(device)
                in_mask = in_mask.unsqueeze(0).to(device)
                out_mask = out_mask.unsqueeze(0).to(device)
                logits, pred_mask = model(inp, embedding, in_mask)
                ce = masked_cross_entropy(logits, out, out_mask)
                mask_loss = nn.functional.binary_cross_entropy(pred_mask, out_mask.float())
                loss = loss + ce + mask_loss
            loss.backward()
            opt.step()
        return embedding.detach()

    def set_embedding(self, task_id: str, value: torch.Tensor) -> None:
        idx = self.task_to_idx[task_id]
        with torch.no_grad():
            self.embeddings.weight[idx].copy_(value)
