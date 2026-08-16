from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from continual_learning_lab.config import DataConfig
from continual_learning_lab.reproducibility import make_generator, seed_worker


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class ClassSubset(Dataset[Any]):
    """A dataset view containing only selected classes, with labels unchanged."""

    def __init__(
        self,
        dataset: Dataset[Any],
        targets: Sequence[int],
        classes: Sequence[int],
        max_samples_per_class: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.classes = tuple(classes)
        allowed = set(classes)
        counts = {class_id: 0 for class_id in classes}
        self.indices: list[int] = []

        for index, target in enumerate(targets):
            target = int(target)
            if target not in allowed:
                continue
            if max_samples_per_class is not None and counts[target] >= max_samples_per_class:
                continue
            self.indices.append(index)
            counts[target] += 1

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]


@dataclass(frozen=True)
class TaskLoaders:
    task_id: int
    classes: tuple[int, ...]
    train: DataLoader[Any]
    test: DataLoader[Any]


@dataclass(frozen=True)
class JointLoaders:
    train: DataLoader[Any]
    overall_test: DataLoader[Any]
    pair_tests: tuple[DataLoader[Any], ...]


def build_cifar10_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Build augmented training transforms and stable evaluation transforms."""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_transform, test_transform


def _build_cifar10_datasets(
    config: DataConfig,
) -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    train_transform, test_transform = build_cifar10_transforms()
    train_dataset = datasets.CIFAR10(
        root=config.root,
        train=True,
        transform=train_transform,
        download=config.download,
    )
    test_dataset = datasets.CIFAR10(
        root=config.root,
        train=False,
        transform=test_transform,
        download=config.download,
    )
    return train_dataset, test_dataset


def build_split_cifar10(config: DataConfig, seed: int, use_pin_memory: bool) -> list[TaskLoaders]:
    train_dataset, test_dataset = _build_cifar10_datasets(config)

    tasks: list[TaskLoaders] = []
    for task_id, classes in enumerate(config.task_classes):
        train_subset = ClassSubset(
            train_dataset,
            train_dataset.targets,
            classes,
            config.max_train_samples_per_class,
        )
        test_subset = ClassSubset(
            test_dataset,
            test_dataset.targets,
            classes,
            config.max_eval_samples_per_class,
        )
        common_loader_args = {
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory and use_pin_memory,
            "worker_init_fn": seed_worker,
            # Every task owns train/test loaders. Keeping all of their workers
            # alive would accumulate 2 * num_tasks worker pools after evaluation.
            "persistent_workers": False,
        }
        tasks.append(
            TaskLoaders(
                task_id=task_id,
                classes=classes,
                train=DataLoader(
                    train_subset,
                    batch_size=config.train_batch_size,
                    shuffle=True,
                    generator=make_generator(seed + task_id),
                    **common_loader_args,
                ),
                test=DataLoader(
                    test_subset,
                    batch_size=config.eval_batch_size,
                    shuffle=False,
                    generator=make_generator(seed + 10_000 + task_id),
                    **common_loader_args,
                ),
            )
        )
    return tasks
