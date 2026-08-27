import torch

from continual_learning_lab.config import DelayedRecallConfig, load_config
from continual_learning_lab.delayed_recall_data import (
    build_delayed_recall_loaders,
    make_delayed_recall_dataset,
)


def test_delayed_recall_tensor_encodes_value_distractor_and_cue() -> None:
    dataset = make_delayed_recall_dataset(
        delay=5,
        max_delay=80,
        examples=10,
        seed=7,
        distractor_low=-1.0,
        distractor_high=1.0,
    )
    inputs, targets = dataset.tensors

    assert inputs.shape == (10, 82, 3)
    assert torch.equal(targets.bincount(), torch.tensor([5, 5]))
    assert torch.equal(inputs[:, 75, 0], targets.float().mul(2).sub(1))
    assert torch.count_nonzero(inputs[:, :75, 0]) == 0
    assert torch.count_nonzero(inputs[:, 76:, 0]) == 0
    assert torch.all(inputs[:, 81, 2] == 1)
    assert torch.count_nonzero(inputs[:, :81, 2]) == 0
    assert torch.count_nonzero(inputs[:, 75, 1]) == 0
    assert torch.count_nonzero(inputs[:, 81, 1]) == 0
    assert torch.all(inputs[:, :, 1] >= -1.0)
    assert torch.all(inputs[:, :, 1] <= 1.0)


def test_paired_seed_rebuilds_identical_data_and_first_epoch_order() -> None:
    config = load_config("configs/heterogeneous_leaky_delayed_recall.yaml")
    assert isinstance(config, DelayedRecallConfig)
    first = build_delayed_recall_loaders(config.data, seed=42, use_pin_memory=False)
    second = build_delayed_recall_loaders(config.data, seed=42, use_pin_memory=False)

    first_inputs, first_targets = next(iter(first.train))
    second_inputs, second_targets = next(iter(second.train))
    assert torch.equal(first_inputs, second_inputs)
    assert torch.equal(first_targets, second_targets)
    assert torch.equal(
        first.test_datasets[80].tensors[0], second.test_datasets[80].tensors[0]
    )


def test_corrected_protocol_uses_exact_delay_and_zero_padding() -> None:
    dataset = make_delayed_recall_dataset(
        delay=40,
        max_delay=120,
        examples=10,
        seed=7,
        distractor_low=-1.0,
        distractor_high=1.0,
        timing_protocol="fixed_initial_value",
        distractor_mode="same_content_channel",
    )
    inputs, targets = dataset.tensors

    assert inputs.shape == (10, 122, 3)
    assert torch.equal(targets.bincount(), torch.tensor([5, 5]))
    assert torch.equal(inputs[:, 0, 0], targets.float().mul(2).sub(1))
    assert torch.all(inputs[:, 0, 1] == 1)
    assert torch.count_nonzero(inputs[:, 1:41, 0]) == 10 * 40
    assert torch.all(inputs[:, 1:41, 0] >= -1.0)
    assert torch.all(inputs[:, 1:41, 0] <= 1.0)
    assert torch.all(inputs[:, 41, 2] == 1)
    assert torch.count_nonzero(inputs[:, :41, 2]) == 0
    assert torch.count_nonzero(inputs[:, 42:]) == 0
    assert torch.count_nonzero(inputs[:, 1:, 1]) == 0
