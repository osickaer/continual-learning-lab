from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F

from continual_learning_lab.config import (
    AddingProblemModelConfig,
    DelayedRecallModelConfig,
)


class RecallModel(Protocol):
    model_type: str

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor: ...

    def hidden_states(self, inputs: torch.Tensor) -> torch.Tensor: ...


def initialize_recurrent_model(model: nn.Module, hidden_size: int) -> None:
    """Apply the same initialization rule to every recurrent architecture."""
    bound = 1.0 / math.sqrt(hidden_size)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "bias" in name:
                parameter.zero_()
            else:
                parameter.uniform_(-bound, bound)


def hidden_state_at_recall_cue(
    inputs: torch.Tensor, states: torch.Tensor
) -> torch.Tensor:
    """Select each sequence's hidden state when its recall cue arrives."""
    if inputs.ndim != 3 or states.ndim != 3:
        raise ValueError("Inputs and states must have shape [batch, time, feature]")
    cues = inputs[:, :, 2]
    if not torch.all(cues.eq(1.0).sum(dim=1).eq(1)):
        raise ValueError("Each sequence must contain exactly one recall cue")
    # Multiplying by the one-hot cue mask is mathematically equivalent to
    # states[batch_index, cue_index]. Unlike that advanced-index operation, its
    # backward pass has a deterministic MPS implementation.
    return (states * cues.unsqueeze(-1)).sum(dim=1)


def readout_hidden_state(
    inputs: torch.Tensor, states: torch.Tensor, readout: str
) -> torch.Tensor:
    if readout == "cue":
        return hidden_state_at_recall_cue(inputs, states)
    if readout == "final":
        return states[:, -1]
    raise ValueError(f"Unsupported recurrent readout: {readout}")


class VanillaRNNClassifier(nn.Module):
    model_type = "vanilla_rnn"

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        readout: str,
    ) -> None:
        super().__init__()
        self.readout = readout
        self.recurrent = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, output_size)
        initialize_recurrent_model(self, hidden_size)

    def hidden_states(self, inputs: torch.Tensor) -> torch.Tensor:
        states, _final_state = self.recurrent(inputs)
        return states

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        states = self.hidden_states(inputs)
        return self.output(readout_hidden_state(inputs, states, self.readout))


class GRUClassifier(nn.Module):
    model_type = "gru"

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        readout: str,
    ) -> None:
        super().__init__()
        self.readout = readout
        self.recurrent = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, output_size)
        initialize_recurrent_model(self, hidden_size)

    def hidden_states(self, inputs: torch.Tensor) -> torch.Tensor:
        states, _final_state = self.recurrent(inputs)
        return states

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        states = self.hidden_states(inputs)
        return self.output(readout_hidden_state(inputs, states, self.readout))


class LeakyRNNCell(nn.Module):
    """One explicit dense recurrent update with fixed per-neuron leak rates."""

    def __init__(self, input_size: int, hidden_size: int, alpha: torch.Tensor) -> None:
        super().__init__()
        if alpha.shape != (hidden_size,):
            raise ValueError(f"alpha must have shape ({hidden_size},)")
        if torch.any(alpha <= 0) or torch.any(alpha > 1):
            raise ValueError("alpha values must be in (0, 1]")

        # W_ih maps the current input x_t into hidden space. W_hh is a dense
        # recurrent matrix, so every value in h_prev can affect every candidate
        # hidden neuron at the next timestep.
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        # A buffer moves with the model and is saved in its state_dict, but it
        # receives no gradient and is never changed by the optimizer.
        self.register_buffer("alpha", alpha.detach().float().clone())

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # x_t is this batch's input at one timestep; h_prev is the memory carried
        # from the preceding timestep. The candidate is the state a vanilla tanh
        # RNN would adopt after applying its input and recurrent weights.
        candidate = torch.tanh(
            F.linear(x_t, self.weight_ih, self.bias_ih)
            + F.linear(h_prev, self.weight_hh, self.bias_hh)
        )
        # alpha is an update rate. Small values retain more of h_prev; large
        # values move more quickly toward the newly computed candidate.
        alpha = self.alpha.unsqueeze(0)
        return (1.0 - alpha) * h_prev + alpha * candidate


class LeakyRNNClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        alpha: torch.Tensor,
        *,
        model_type: str,
        readout: str,
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.readout = readout
        self.hidden_size = hidden_size
        self.cell = LeakyRNNCell(input_size, hidden_size, alpha)
        self.output = nn.Linear(hidden_size, num_classes)
        initialize_recurrent_model(self, hidden_size)

    def hidden_states(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("Recurrent inputs must have shape [batch, time, feature]")
        batch_size, timesteps, _features = inputs.shape
        h_t = inputs.new_zeros(batch_size, self.hidden_size)
        states: list[torch.Tensor] = []
        # Keeping this loop visible is deliberate: h_t becomes h_prev on the
        # next iteration, and autograd connects the entire unrolled sequence.
        for timestep in range(timesteps):
            h_t = self.cell(inputs[:, timestep], h_t)
            states.append(h_t)
        return torch.stack(states, dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        states = self.hidden_states(inputs)
        return self.output(readout_hidden_state(inputs, states, self.readout))


def build_recurrent_model(
    model_type: str,
    config: DelayedRecallModelConfig | AddingProblemModelConfig,
    *,
    alpha: float | None = None,
    readout: str = "cue",
) -> nn.Module:
    output_size = (
        config.num_classes
        if isinstance(config, DelayedRecallModelConfig)
        else config.output_size
    )
    if model_type == "vanilla_rnn":
        return VanillaRNNClassifier(
            config.input_size,
            config.hidden_size,
            output_size,
            readout=readout,
        )
    if model_type == "gru":
        return GRUClassifier(
            config.input_size,
            config.hidden_size,
            output_size,
            readout=readout,
        )
    if model_type == "homogeneous_leaky":
        if alpha is None:
            raise ValueError("A homogeneous leaky model requires alpha")
        alpha_vector = torch.full((config.hidden_size,), float(alpha))
        return LeakyRNNClassifier(
            config.input_size,
            config.hidden_size,
            output_size,
            alpha_vector,
            model_type=model_type,
            readout=readout,
        )
    if model_type == "heterogeneous_leaky":
        if alpha is not None:
            raise ValueError("The heterogeneous alpha vector comes from model config")
        alpha_vector = torch.repeat_interleave(
            torch.tensor(config.heterogeneous_alphas, dtype=torch.float32),
            torch.tensor(config.heterogeneous_group_sizes),
        )
        return LeakyRNNClassifier(
            config.input_size,
            config.hidden_size,
            output_size,
            alpha_vector,
            model_type=model_type,
            readout=readout,
        )
    raise ValueError(f"Unsupported recurrent model type: {model_type}")
