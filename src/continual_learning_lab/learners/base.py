from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    loss: float
    accuracy: float
    examples: int


class ContinualLearner(Protocol):
    """The narrow boundary between experiment infrastructure and a CL method."""

    model: nn.Module

    def train_epoch(
        self,
        task_id: int,
        epoch: int,
        loader: DataLoader,
    ) -> EpochResult: ...
