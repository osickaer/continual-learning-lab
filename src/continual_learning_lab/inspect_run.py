from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import datasets

from continual_learning_lab.config import Config, load_config
from continual_learning_lab.data import CIFAR10_MEAN, CIFAR10_STD, build_cifar10_transforms
from continual_learning_lab.experiment import resolve_device
from continual_learning_lab.inspection import CIFAR10_CLASSES, capture_activations, write_inspection
from continual_learning_lab.model import SmallCNN
from continual_learning_lab.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one CIFAR-10 image as it passes through a trained SmallCNN"
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML configuration")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Model state file; defaults to the newest final model under experiment.output_dir",
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--class-id",
        type=int,
        choices=range(len(CIFAR10_CLASSES)),
        help="Optional CIFAR-10 class ID (0-9) used to select an example",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Dataset index, or index within --class-id when a class is supplied",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=16,
        help="Maximum feature maps shown per layer; raw tensors retain all channels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to inspections/<run-and-sample>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_index < 0:
        raise ValueError("--sample-index must be non-negative")
    if args.max_channels <= 0:
        raise ValueError("--max-channels must be positive")

    config = load_config(args.config)
    seed_everything(config.experiment.seed, config.experiment.deterministic)
    device = resolve_device(config.experiment.device)
    checkpoint = resolve_checkpoint(config, args.checkpoint)
    image, target, dataset_index = load_sample(
        config,
        split=args.split,
        sample_index=args.sample_index,
        class_id=args.class_id,
    )

    model = SmallCNN(config.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    result = capture_activations(model, image.unsqueeze(0).to(device))
    output_dir = choose_output_dir(
        args.output_dir,
        config,
        checkpoint,
        args.split,
        dataset_index,
    )
    report = write_inspection(
        output_dir,
        model,
        image,
        result,
        checkpoint=checkpoint,
        split=args.split,
        dataset_index=dataset_index,
        target=target,
        max_channels=args.max_channels,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
    )

    prediction = int(result.probabilities[0].argmax().item())
    confidence = result.probabilities[0, prediction].item()
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint.resolve()}")
    print(
        f"Input: {args.split} dataset index {dataset_index}, "
        f"true class {target} ({CIFAR10_CLASSES[target]})"
    )
    print(
        f"Prediction: class {prediction} ({CIFAR10_CLASSES[prediction]}), "
        f"probability={confidence:.3f}"
    )
    print(f"Inspection report: {report.resolve()}")
    print(f"Raw tensors: {(output_dir / 'intermediate_values.pt').resolve()}")


def resolve_checkpoint(config: Config, requested: Path | None) -> Path:
    if requested is not None:
        if not requested.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {requested}")
        return requested

    output_root = Path(config.experiment.output_dir)
    candidates = list(output_root.glob("*/final_model_state.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No final_model_state.pt found under {output_root}. "
            "Run an experiment or pass --checkpoint explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_sample(
    config: Config,
    *,
    split: str,
    sample_index: int,
    class_id: int | None,
) -> tuple[torch.Tensor, int, int]:
    _, evaluation_transform = build_cifar10_transforms()
    dataset = datasets.CIFAR10(
        root=config.data.root,
        train=split == "train",
        transform=evaluation_transform,
        download=config.data.download,
    )

    dataset_index = sample_index
    if class_id is not None:
        matching = [
            index for index, target in enumerate(dataset.targets) if int(target) == class_id
        ]
        if sample_index >= len(matching):
            raise IndexError(
                f"Class {class_id} has {len(matching)} samples, so index {sample_index} is invalid"
            )
        dataset_index = matching[sample_index]
    if dataset_index >= len(dataset):
        raise IndexError(
            f"Dataset has {len(dataset)} samples, so index {dataset_index} is invalid"
        )

    image, target = dataset[dataset_index]
    return image, int(target), dataset_index


def choose_output_dir(
    requested: Path | None,
    config: Config,
    checkpoint: Path,
    split: str,
    dataset_index: int,
) -> Path:
    if requested is not None:
        candidate = requested
    else:
        run_name = checkpoint.parent.name
        candidate = Path("inspections") / f"{run_name}-{split}-{dataset_index}"

    if not candidate.exists():
        return candidate
    for suffix in range(2, 10_000):
        alternative = candidate.with_name(f"{candidate.name}-{suffix}")
        if not alternative.exists():
            return alternative
    raise RuntimeError(f"Could not choose an unused inspection directory near {candidate}")


if __name__ == "__main__":
    main()
