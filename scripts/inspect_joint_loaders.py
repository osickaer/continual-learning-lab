"""Quick, local contract check for the Experiment 001 joint data loaders.

This is a development tool, not an experiment. It uses the downloaded CIFAR-10
files with tiny per-class caps and ``num_workers=0`` so loader bugs produce fast,
readable tracebacks in the main process.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from continual_learning_lab.config import load_config
from continual_learning_lab.data import JointLoaders, build_joint_cifar10


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "joint_cifar10_oracle.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a tiny, real-CIFAR instance of the joint loaders."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-per-class", type=int, default=3)
    parser.add_argument("--eval-per-class", type=int, default=2)
    return parser.parse_args()


def _read_loader(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor, Counter[int]]:
    first_images: torch.Tensor | None = None
    first_labels: torch.Tensor | None = None
    counts: Counter[int] = Counter()

    for images, labels in loader:
        if first_images is None:
            first_images = images
            first_labels = labels
        counts.update(int(label) for label in labels.tolist())

    if first_images is None or first_labels is None:
        raise AssertionError("Loader unexpectedly produced no batches")
    return first_images, first_labels, counts


def _assert_equal(actual: object, expected: object, description: str) -> None:
    if actual != expected:
        raise AssertionError(f"{description}: expected {expected!r}, got {actual!r}")


def inspect_loaders(
    loaders: JointLoaders,
    task_classes: tuple[tuple[int, ...], ...],
    train_per_class: int,
    eval_per_class: int,
) -> None:
    all_classes = tuple(class_id for classes in task_classes for class_id in classes)
    expected_train_counts = Counter({class_id: train_per_class for class_id in all_classes})
    expected_eval_counts = Counter({class_id: eval_per_class for class_id in all_classes})

    _assert_equal(
        len(loaders.train.dataset),
        len(all_classes) * train_per_class,
        "training subset size",
    )
    _assert_equal(
        len(loaders.overall_test.dataset),
        len(all_classes) * eval_per_class,
        "overall test subset size",
    )
    _assert_equal(len(loaders.pair_tests), len(task_classes), "pair-loader count")
    if not isinstance(loaders.train.sampler, RandomSampler):
        raise AssertionError("Training loader must shuffle with a RandomSampler")
    if not isinstance(loaders.overall_test.sampler, SequentialSampler):
        raise AssertionError("Overall test loader must use deterministic sequential sampling")

    train_images, train_labels, train_counts = _read_loader(loaders.train)
    overall_images, overall_labels, overall_counts = _read_loader(loaders.overall_test)
    _assert_equal(train_counts, expected_train_counts, "training label counts")
    _assert_equal(overall_counts, expected_eval_counts, "overall test label counts")
    _assert_equal(tuple(train_images.shape[1:]), (3, 32, 32), "training image shape")
    _assert_equal(tuple(overall_images.shape[1:]), (3, 32, 32), "test image shape")
    _assert_equal(train_labels.dtype, torch.int64, "training label dtype")
    _assert_equal(overall_labels.dtype, torch.int64, "test label dtype")
    if not train_images.is_floating_point() or not overall_images.is_floating_point():
        raise AssertionError("Normalized CIFAR-10 images must be floating-point tensors")

    pair_summaries: list[tuple[int, ...]] = []
    for pair_id, (classes, loader) in enumerate(zip(task_classes, loaders.pair_tests)):
        if not isinstance(loader.sampler, SequentialSampler):
            raise AssertionError(f"Pair test loader {pair_id} must not shuffle")
        _, _, counts = _read_loader(loader)
        expected_counts = Counter({class_id: eval_per_class for class_id in classes})
        _assert_equal(counts, expected_counts, f"pair {pair_id} global-label counts")
        pair_summaries.append(tuple(sorted(counts)))

    print("Joint loader contract passed")
    print(f"  train:   {len(loaders.train.dataset):3d} examples, labels={dict(train_counts)}")
    print(f"  overall: {len(loaders.overall_test.dataset):3d} examples, labels={dict(overall_counts)}")
    print(f"  pairs:   {pair_summaries}")
    print(f"  train batch tensors: images={tuple(train_images.shape)}, labels={tuple(train_labels.shape)}")


def main() -> None:
    args = parse_args()
    if args.train_per_class <= 0 or args.eval_per_class <= 0:
        raise SystemExit("Per-class debug limits must be positive")

    config = load_config(args.config)
    debug_data = replace(
        config.data,
        download=False,
        num_workers=0,
        max_train_samples_per_class=args.train_per_class,
        max_eval_samples_per_class=args.eval_per_class,
    )
    loaders = build_joint_cifar10(
        debug_data,
        seed=config.experiment.seed,
        use_pin_memory=False,
    )
    if loaders is None:
        raise SystemExit(
            "build_joint_cifar10 returned None. Finish constructing and returning "
            "JointLoaders, then rerun this command."
        )
    inspect_loaders(
        loaders,
        task_classes=debug_data.task_classes,
        train_per_class=args.train_per_class,
        eval_per_class=args.eval_per_class,
    )


if __name__ == "__main__":
    main()
