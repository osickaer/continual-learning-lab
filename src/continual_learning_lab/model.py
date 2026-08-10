from __future__ import annotations

import torch
from torch import nn

from continual_learning_lab.config import ModelConfig


class SmallCNN(nn.Module):
    """A deliberately small CIFAR-10 CNN with one shared classification head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        first, second = config.channels
        self.features = nn.Sequential(
            nn.Conv2d(3, first, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(first, first, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(first, second, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(second, second, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # CIFAR-10 inputs are fixed at 32x32, so features are 8x8 here.
            # A fixed pool preserves the intended 4x4 shape and has a
            # deterministic CUDA backward implementation; AdaptiveAvgPool2d
            # does not when strict PyTorch determinism is enabled.
            nn.AvgPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(second * 4 * 4, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
