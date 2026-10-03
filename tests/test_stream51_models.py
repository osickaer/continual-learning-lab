import torch

from continual_learning_lab.config import Stream51Config, load_config
from continual_learning_lab.model import count_trainable_parameters
from continual_learning_lab.stream51_models import build_stream51_model


def _config() -> Stream51Config:
    config = load_config("configs/stream51_temporal_state.yaml")
    assert isinstance(config, Stream51Config)
    return config


def test_all_conditions_use_the_same_model_parameters() -> None:
    config = _config()
    torch.manual_seed(123)
    first = build_stream51_model(config.model)
    torch.manual_seed(123)
    second = build_stream51_model(config.model)

    assert count_trainable_parameters(first) == count_trainable_parameters(second)
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)


def test_only_the_first_internal_tick_receives_the_observation() -> None:
    model = build_stream51_model(_config().model)
    received: list[torch.Tensor] = []
    hook = model.cell.register_forward_pre_hook(
        lambda _module, values: received.append(values[0].detach().clone())
    )
    features = torch.randn(1, model.feature_size)

    model.step(features, None, ticks=4)
    hook.remove()

    assert torch.equal(received[0], features)
    assert all(torch.count_nonzero(value) == 0 for value in received[1:])


def test_persistent_state_changes_the_next_observation_and_can_be_detached() -> None:
    model = build_stream51_model(_config().model)
    first = torch.randn(1, model.feature_size)
    second = torch.randn(1, model.feature_size)
    _logits, hidden = model.step(first, None, ticks=1)

    persistent_logits, _ = model.step(second, hidden.detach(), ticks=1)
    stateless_logits, _ = model.step(second, None, ticks=1)

    assert hidden.detach().grad_fn is None
    assert not torch.allclose(persistent_logits, stateless_logits)


def test_gradients_flow_through_all_four_internal_ticks() -> None:
    model = build_stream51_model(_config().model)
    features = torch.randn(2, model.feature_size)
    logits, _hidden = model.step(features, None, ticks=4)

    logits.square().mean().backward()

    assert model.cell.weight_ih.grad is not None
    assert model.cell.weight_hh.grad is not None
    assert torch.count_nonzero(model.cell.weight_hh.grad) > 0
