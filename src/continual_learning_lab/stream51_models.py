from __future__ import annotations

import torch
from torch import nn

from continual_learning_lab.config import Stream51ModelConfig


class Stream51RecurrentClassifier(nn.Module):
    """One recurrent classifier used for every state/tick condition."""

    def __init__(self, config: Stream51ModelConfig) -> None:
        super().__init__()
        self.feature_size = config.feature_size
        self.hidden_size = config.hidden_size
        self.cell = nn.GRUCell(config.feature_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.num_classes)

    def initial_state(self, features: torch.Tensor) -> torch.Tensor:
        return features.new_zeros(features.size(0), self.hidden_size)

    def step(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor | None,
        *,
        ticks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.size(1) != self.feature_size:
            raise ValueError(
                f"features must have shape [batch, {self.feature_size}]"
            )
        if ticks <= 0:
            raise ValueError("ticks must be positive")
        if hidden is None:
            hidden = self.initial_state(features)
        if hidden.shape != (features.size(0), self.hidden_size):
            raise ValueError(
                f"hidden must have shape [batch, {self.hidden_size}]"
            )

        blank_input = torch.zeros_like(features)
        next_hidden = hidden
        for tick in range(ticks):
            tick_input = features if tick == 0 else blank_input
            next_hidden = self.cell(tick_input, next_hidden)
        return self.output(next_hidden), next_hidden


def build_stream51_model(config: Stream51ModelConfig) -> Stream51RecurrentClassifier:
    return Stream51RecurrentClassifier(config)
