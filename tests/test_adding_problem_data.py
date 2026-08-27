import torch

from continual_learning_lab.adding_problem_data import (
    build_adding_problem_loaders,
    make_adding_problem_dataset,
)
from continual_learning_lab.config import AddingProblemConfig, load_config


def test_adding_problem_has_two_split_markers_and_exact_sum_target() -> None:
    dataset = make_adding_problem_dataset(
        sequence_length=20,
        examples=32,
        seed=7,
        value_low=0.0,
        value_high=1.0,
    )
    inputs, targets = dataset.tensors
    markers = inputs[:, :, 1]

    assert inputs.shape == (32, 20, 2)
    assert targets.shape == (32, 1)
    assert torch.all(markers.sum(dim=1) == 2)
    assert torch.all(markers[:, :10].sum(dim=1) == 1)
    assert torch.all(markers[:, 10:].sum(dim=1) == 1)
    assert torch.allclose(targets.squeeze(1), (inputs[:, :, 0] * markers).sum(dim=1))
    assert torch.all((0 <= targets) & (targets <= 2))


def test_adding_problem_paired_seed_rebuilds_data_and_batch_order() -> None:
    config = load_config("configs/adding_problem.yaml")
    assert isinstance(config, AddingProblemConfig)
    first = build_adding_problem_loaders(config.data, seed=42, use_pin_memory=False)
    second = build_adding_problem_loaders(config.data, seed=42, use_pin_memory=False)

    first_inputs, first_targets = next(iter(first.train[40]))
    second_inputs, second_targets = next(iter(second.train[40]))
    assert torch.equal(first_inputs, second_inputs)
    assert torch.equal(first_targets, second_targets)
    assert set(first.validation) == set(config.data.train_lengths)
    assert set(first.test) == set(config.data.evaluation_lengths)
    assert torch.equal(
        first.test_datasets[320].tensors[0], second.test_datasets[320].tensors[0]
    )
