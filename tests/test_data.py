from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, TensorDataset

import continual_learning_lab.data as data_module
from continual_learning_lab.config import load_config
from continual_learning_lab.data import ClassSubset, build_joint_cifar10


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "joint_cifar10_oracle.yaml"


class TinyCIFAR(Dataset):
    def __init__(self, targets: list[int]) -> None:
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.tensor([index], dtype=torch.float32), self.targets[index]


def test_class_subset_keeps_global_labels_and_caps_each_class() -> None:
    inputs = torch.arange(8).unsqueeze(1)
    targets = torch.tensor([0, 1, 2, 1, 2, 2, 3, 2])
    dataset = TensorDataset(inputs, targets)

    subset = ClassSubset(dataset, targets.tolist(), classes=(1, 2), max_samples_per_class=2)

    labels = [int(subset[index][1]) for index in range(len(subset))]
    assert labels == [1, 2, 1, 2]


def test_joint_loaders_include_all_classes_and_global_pair_labels(monkeypatch) -> None:
    train_dataset = TinyCIFAR([class_id for class_id in range(10) for _ in range(4)])
    test_dataset = TinyCIFAR([class_id for class_id in range(10) for _ in range(3)])
    monkeypatch.setattr(
        data_module,
        "_build_cifar10_datasets",
        lambda config: (train_dataset, test_dataset),
    )
    config = replace(
        load_config(CONFIG_PATH).data,
        num_workers=0,
        pin_memory=False,
        max_train_samples_per_class=2,
        max_eval_samples_per_class=1,
    )

    loaders = build_joint_cifar10(config, seed=42, use_pin_memory=False)

    train_labels = [int(label) for _, labels in loaders.train for label in labels]
    overall_labels = [int(label) for _, labels in loaders.overall_test for label in labels]
    pair_labels = [
        [int(label) for _, labels in loader for label in labels]
        for loader in loaders.pair_tests
    ]

    assert Counter(train_labels) == Counter({class_id: 2 for class_id in range(10)})
    assert Counter(overall_labels) == Counter({class_id: 1 for class_id in range(10)})
    assert [set(labels) for labels in pair_labels] == [
        {0, 1},
        {2, 3},
        {4, 5},
        {6, 7},
        {8, 9},
    ]
    assert isinstance(loaders.train.sampler, RandomSampler)
    assert isinstance(loaders.overall_test.sampler, SequentialSampler)
    assert all(isinstance(loader.sampler, SequentialSampler) for loader in loaders.pair_tests)
