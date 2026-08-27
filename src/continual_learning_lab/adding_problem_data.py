from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from continual_learning_lab.config import AddingProblemDataConfig
from continual_learning_lab.reproducibility import make_generator, seed_worker


@dataclass(frozen=True)
class AddingProblemLoaders:
    train: dict[int, DataLoader[Any]]
    validation: dict[int, DataLoader[Any]]
    test: dict[int, DataLoader[Any]]
    test_datasets: dict[int, TensorDataset]


def make_adding_problem_dataset(
    *,
    sequence_length: int,
    examples: int,
    seed: int,
    value_low: float,
    value_high: float,
) -> TensorDataset:
    """Create classic Adding Problem sequences with one marker in each half."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if examples <= 0:
        raise ValueError("examples must be positive")
    if value_low >= value_high:
        raise ValueError("value_low must be less than value_high")

    generator = make_generator(seed)
    values = torch.empty(examples, sequence_length).uniform_(
        value_low, value_high, generator=generator
    )
    split = sequence_length // 2
    first_positions = torch.randint(0, split, (examples,), generator=generator)
    second_positions = torch.randint(
        split, sequence_length, (examples,), generator=generator
    )
    rows = torch.arange(examples)
    markers = torch.zeros(examples, sequence_length)
    markers[rows, first_positions] = 1.0
    markers[rows, second_positions] = 1.0
    targets = (
        values[rows, first_positions] + values[rows, second_positions]
    ).unsqueeze(1)
    inputs = torch.stack((values, markers), dim=2)
    return TensorDataset(inputs, targets)


def build_adding_problem_loaders(
    config: AddingProblemDataConfig,
    *,
    seed: int,
    use_pin_memory: bool,
) -> AddingProblemLoaders:
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory and use_pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": False,
    }

    def datasets(lengths: tuple[int, ...], examples: int, offset: int):
        return {
            length: make_adding_problem_dataset(
                sequence_length=length,
                examples=examples,
                seed=seed + offset + length,
                value_low=config.value_low,
                value_high=config.value_high,
            )
            for length in lengths
        }

    train_datasets = datasets(config.train_lengths, config.train_examples_per_length, 0)
    validation_datasets = datasets(
        config.train_lengths,
        config.validation_examples_per_length,
        10_000,
    )
    test_datasets = datasets(
        config.evaluation_lengths,
        config.test_examples_per_length,
        20_000,
    )
    train = {
        length: DataLoader(
            dataset,
            shuffle=True,
            generator=make_generator(seed + 30_000 + length),
            **common,
        )
        for length, dataset in train_datasets.items()
    }
    validation = {
        length: DataLoader(dataset, shuffle=False, **common)
        for length, dataset in validation_datasets.items()
    }
    test = {
        length: DataLoader(dataset, shuffle=False, **common)
        for length, dataset in test_datasets.items()
    }
    return AddingProblemLoaders(
        train=train,
        validation=validation,
        test=test,
        test_datasets=test_datasets,
    )
