from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from continual_learning_lab.config import DelayedRecallDataConfig
from continual_learning_lab.reproducibility import make_generator, seed_worker


@dataclass(frozen=True)
class DelayedRecallLoaders:
    train: DataLoader[Any]
    validation: dict[int, DataLoader[Any]]
    test: dict[int, DataLoader[Any]]
    test_datasets: dict[int, TensorDataset]


def make_delayed_recall_dataset(
    *,
    delay: int,
    max_delay: int,
    examples: int,
    seed: int,
    distractor_low: float,
    distractor_high: float,
    timing_protocol: str = "fixed_final_cue",
    distractor_mode: str = "separate_channel",
) -> TensorDataset:
    """Generate a balanced delayed-recall split for one delay.

    All sequences use ``max_delay + 2`` timesteps so mixed delays can share a
    DataLoader. The corrected protocol puts the value at t=0, the cue at
    t=delay+1, and zero padding after the cue. Its three features mean content,
    store marker, and recall cue. The historical protocol remains available so
    the completed first run stays reproducible.
    """
    if examples <= 0 or examples % 2:
        raise ValueError("examples must be a positive even integer")
    if not 0 < delay <= max_delay:
        raise ValueError("delay must be in [1, max_delay]")

    generator = make_generator(seed)
    sequence_length = max_delay + 2
    if timing_protocol == "fixed_final_cue":
        cue_index = sequence_length - 1
        value_index = cue_index - delay - 1
    elif timing_protocol == "fixed_initial_value":
        value_index = 0
        cue_index = delay + 1
    else:
        raise ValueError(f"Unsupported timing protocol: {timing_protocol}")
    if distractor_mode not in {"separate_channel", "same_content_channel"}:
        raise ValueError(f"Unsupported distractor mode: {distractor_mode}")

    inputs = torch.zeros(examples, sequence_length, 3)
    historical_protocol = (
        timing_protocol == "fixed_final_cue"
        and distractor_mode == "separate_channel"
    )
    if historical_protocol:
        # Retain the exact random-number order and full distractor prefix used
        # by the completed first run, including distractors before the value.
        inputs[:, :, 1].uniform_(
            distractor_low, distractor_high, generator=generator
        )

    # Classes 0 and 1 mean recall -1 and +1. Shuffle them so every split is
    # exactly balanced without presenting all examples of one sign together.
    targets = torch.cat(
        (
            torch.zeros(examples // 2, dtype=torch.long),
            torch.ones(examples // 2, dtype=torch.long),
        )
    )
    permutation = torch.randperm(examples, generator=generator)
    targets = targets[permutation]
    values = targets.float().mul(2.0).sub(1.0)

    if historical_protocol:
        inputs[:, value_index, 0] = values
        inputs[:, value_index, 1] = 0.0
        inputs[:, cue_index, 1] = 0.0
    else:
        # Exactly `delay` distractors occur strictly between value and cue.
        distractors = torch.empty(examples, delay).uniform_(
            distractor_low, distractor_high, generator=generator
        )
        if distractor_mode == "separate_channel":
            inputs[:, value_index, 0] = values
            inputs[:, value_index + 1 : cue_index, 1] = distractors
        else:
            # Content and distractors share a channel, while the marker tells
            # the model which item must be stored. It cannot solve the task by
            # ignoring an always-irrelevant distractor feature.
            inputs[:, value_index, 0] = values
            inputs[:, value_index, 1] = 1.0
            inputs[:, value_index + 1 : cue_index, 0] = distractors
    inputs[:, cue_index, 2] = 1.0
    return TensorDataset(inputs, targets)


def build_delayed_recall_loaders(
    config: DelayedRecallDataConfig,
    *,
    seed: int,
    use_pin_memory: bool,
) -> DelayedRecallLoaders:
    max_delay = max(config.delays)
    common_loader_args = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory and use_pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": False,
    }

    def build_split(examples: int, split_offset: int) -> dict[int, TensorDataset]:
        return {
            delay: make_delayed_recall_dataset(
                delay=delay,
                max_delay=max_delay,
                examples=examples,
                seed=seed + split_offset + delay,
                distractor_low=config.distractor_low,
                distractor_high=config.distractor_high,
                timing_protocol=config.timing_protocol,
                distractor_mode=config.distractor_mode,
            )
            for delay in config.delays
        }

    train_datasets = build_split(config.train_examples_per_delay, 0)
    validation_datasets = build_split(config.validation_examples_per_delay, 10_000)
    test_datasets = build_split(config.test_examples_per_delay, 20_000)

    train = DataLoader(
        ConcatDataset([train_datasets[delay] for delay in config.delays]),
        shuffle=True,
        generator=make_generator(seed + 30_000),
        **common_loader_args,
    )
    validation = {
        delay: DataLoader(
            dataset,
            shuffle=False,
            generator=make_generator(seed + 40_000 + delay),
            **common_loader_args,
        )
        for delay, dataset in validation_datasets.items()
    }
    test = {
        delay: DataLoader(
            dataset,
            shuffle=False,
            generator=make_generator(seed + 50_000 + delay),
            **common_loader_args,
        )
        for delay, dataset in test_datasets.items()
    }
    return DelayedRecallLoaders(
        train=train,
        validation=validation,
        test=test,
        test_datasets=test_datasets,
    )
