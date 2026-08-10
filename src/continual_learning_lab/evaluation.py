from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    accuracy: float
    examples: int


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> EvaluationResult:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        total_loss += criterion(logits, targets).item()
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += targets.size(0)

    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return EvaluationResult(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
        examples=total_examples,
    )

