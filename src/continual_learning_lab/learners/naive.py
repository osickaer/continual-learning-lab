from __future__ import annotations

import torch
from torch import nn
from torch.optim import AdamW, Optimizer, SGD
from torch.utils.data import DataLoader

from continual_learning_lab.config import TrainingConfig
from continual_learning_lab.learners.base import EpochResult


class NaiveSequentialLearner:
    """Train on the current task only; retain no samples or old-model state."""

    def __init__(self, model: nn.Module, config: TrainingConfig, device: torch.device) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> Optimizer:
        if self.config.optimizer == "sgd":
            return SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        if self.config.optimizer == "adamw":
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def train_epoch(self, task_id: int, epoch: int, loader: DataLoader) -> EpochResult:
        # task_id is intentionally unused: the naive model is never told which task it sees.
        del task_id
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for inputs, targets in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # PyTorch accumulates gradients by default, so clear values left by
            # the preceding batch before building this batch's computation graph.
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            # backward() writes gradients into each model parameter's .grad;
            # step() reads those gradients and updates the same parameters.
            loss.backward()
            self.optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_examples += batch_size

        if total_examples == 0:
            raise ValueError("Cannot train on an empty dataset")

        return EpochResult(
            epoch=epoch,
            loss=total_loss / total_examples,
            accuracy=total_correct / total_examples,
            examples=total_examples,
        )
