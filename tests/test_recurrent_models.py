import torch
from torch import nn

from continual_learning_lab.config import AddingProblemConfig, DelayedRecallConfig, load_config
from continual_learning_lab.recurrent_models import (
    LeakyRNNCell,
    build_recurrent_model,
    hidden_state_at_recall_cue,
)


def _config() -> DelayedRecallConfig:
    config = load_config("configs/heterogeneous_leaky_delayed_recall.yaml")
    assert isinstance(config, DelayedRecallConfig)
    return config


def test_recurrent_models_share_output_and_hidden_state_interfaces() -> None:
    config = _config()
    inputs = torch.randn(4, 82, 3)
    inputs[:, :, 2] = 0
    inputs[:, -1, 2] = 1
    settings = (
        ("vanilla_rnn", None),
        ("gru", None),
        ("homogeneous_leaky", 0.1),
        ("heterogeneous_leaky", None),
    )

    for model_type, alpha in settings:
        model = build_recurrent_model(model_type, config.model, alpha=alpha)
        assert model(inputs).shape == (4, 2)
        assert model.hidden_states(inputs).shape == (4, 82, 96)


def test_leaky_cell_matches_hand_computed_coordinate_updates() -> None:
    cell = LeakyRNNCell(2, 2, torch.tensor([1.0, 0.25]))
    with torch.no_grad():
        cell.weight_ih.copy_(torch.eye(2))
        cell.weight_hh.zero_()
        cell.bias_ih.zero_()
        cell.bias_hh.zero_()
    x_t = torch.tensor([[0.5, -0.5]])
    h_prev = torch.tensor([[0.2, 0.8]])

    actual = cell(x_t, h_prev)
    candidate = torch.tanh(x_t)
    expected = torch.stack(
        (candidate[:, 0], 0.75 * h_prev[:, 1] + 0.25 * candidate[:, 1]), dim=1
    )

    assert torch.allclose(actual, expected)
    assert "alpha" not in dict(cell.named_parameters())
    assert "alpha" in dict(cell.named_buffers())


def test_alpha_one_custom_rnn_matches_vanilla_rnn_after_weight_copy() -> None:
    config = _config()
    vanilla = build_recurrent_model("vanilla_rnn", config.model)
    leaky = build_recurrent_model("homogeneous_leaky", config.model, alpha=1.0)
    with torch.no_grad():
        leaky.cell.weight_ih.copy_(vanilla.recurrent.weight_ih_l0)
        leaky.cell.weight_hh.copy_(vanilla.recurrent.weight_hh_l0)
        leaky.cell.bias_ih.copy_(vanilla.recurrent.bias_ih_l0)
        leaky.cell.bias_hh.copy_(vanilla.recurrent.bias_hh_l0)
        leaky.output.load_state_dict(vanilla.output.state_dict())
    inputs = torch.randn(3, 12, 3)
    inputs[:, :, 2] = 0
    inputs[:, -1, 2] = 1

    # nn.RNN uses a fused implementation, so operation ordering can differ by a
    # few last-place bits from the explicit Python timestep loop.
    assert torch.allclose(
        leaky.hidden_states(inputs), vanilla.hidden_states(inputs), atol=1e-6, rtol=1e-5
    )
    assert torch.allclose(leaky(inputs), vanilla(inputs), atol=1e-6, rtol=1e-5)


def test_heterogeneous_alpha_groups_are_fixed_and_exact() -> None:
    config = _config()
    model = build_recurrent_model("heterogeneous_leaky", config.model)

    assert torch.equal(
        model.cell.alpha,
        torch.tensor([0.5] * 32 + [0.1] * 32 + [0.01] * 32),
    )
    assert model.cell.weight_hh.shape == (96, 96)
    assert model.cell.alpha.requires_grad is False


def test_leaky_model_learns_with_alpha_unchanged() -> None:
    config = _config()
    model = build_recurrent_model("heterogeneous_leaky", config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    inputs = torch.randn(8, 6, 3)
    inputs[:, :, 2] = 0
    inputs[:, -1, 2] = 1
    targets = torch.tensor([0, 1] * 4)
    initial_weight = model.cell.weight_hh.detach().clone()
    initial_alpha = model.cell.alpha.detach().clone()

    loss = nn.CrossEntropyLoss()(model(inputs), targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    clipped_norm = torch.linalg.vector_norm(
        torch.stack(
            [parameter.grad.norm() for parameter in model.parameters() if parameter.grad is not None]
        )
    )
    optimizer.step()

    assert not torch.equal(model.cell.weight_hh, initial_weight)
    assert torch.equal(model.cell.alpha, initial_alpha)
    assert clipped_norm <= 1.000001


def test_logits_are_read_at_cue_and_ignore_later_padding() -> None:
    config = _config()
    model = build_recurrent_model("heterogeneous_leaky", config.model)
    inputs = torch.randn(4, 12, 3)
    inputs[:, :, 2] = 0
    inputs[:, 6, 2] = 1
    changed_padding = inputs.clone()
    changed_padding[:, 7:, :2] = torch.randn_like(changed_padding[:, 7:, :2]) * 10

    assert torch.allclose(model(inputs), model(changed_padding))


def test_cue_mask_selection_matches_indexing_and_backpropagates() -> None:
    states = torch.randn(3, 8, 5, requires_grad=True)
    inputs = torch.zeros(3, 8, 3)
    cue_indices = torch.tensor([1, 4, 6])
    inputs[torch.arange(3), cue_indices, 2] = 1

    selected = hidden_state_at_recall_cue(inputs, states)
    expected = states.detach()[torch.arange(3), cue_indices]
    selected.sum().backward()

    assert torch.equal(selected.detach(), expected)
    expected_gradient = inputs[:, :, 2].unsqueeze(-1).expand_as(states)
    assert torch.equal(states.grad, expected_gradient)


def test_paired_leaky_models_begin_with_identical_trainable_parameters() -> None:
    config = _config()
    torch.manual_seed(123)
    homogeneous = build_recurrent_model("homogeneous_leaky", config.model, alpha=0.1)
    torch.manual_seed(123)
    heterogeneous = build_recurrent_model("heterogeneous_leaky", config.model)

    for homogeneous_parameter, heterogeneous_parameter in zip(
        homogeneous.parameters(), heterogeneous.parameters(), strict=True
    ):
        assert torch.equal(homogeneous_parameter, heterogeneous_parameter)


def test_adding_problem_models_use_final_state_for_scalar_regression() -> None:
    config = load_config("configs/adding_problem.yaml")
    assert isinstance(config, AddingProblemConfig)
    inputs = torch.randn(4, 20, 2)

    for model_type, alpha in (
        ("vanilla_rnn", None),
        ("gru", None),
        ("homogeneous_leaky", 0.1),
        ("heterogeneous_leaky", None),
    ):
        model = build_recurrent_model(
            model_type, config.model, alpha=alpha, readout="final"
        )
        assert model(inputs).shape == (4, 1)
        loss = nn.MSELoss()(model(inputs), torch.rand(4, 1))
        loss.backward()
        assert all(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )
