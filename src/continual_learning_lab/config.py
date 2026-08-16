from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    run_name: str | None
    seed: int
    device: str
    output_dir: str
    tracking_uri: str
    artifact_dir: str
    deterministic: bool


@dataclass(frozen=True)
class DataConfig:
    root: str
    download: bool
    train_batch_size: int
    eval_batch_size: int
    num_workers: int
    pin_memory: bool
    max_train_samples_per_class: int | None
    max_eval_samples_per_class: int | None
    task_classes: tuple[tuple[int, ...], ...]
    protocol: str


@dataclass(frozen=True)
class ModelConfig:
    channels: tuple[int, int]
    hidden_dim: int
    dropout: float
    num_classes: int


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    epochs_per_task: int
    optimizer: str
    learning_rate: float
    momentum: float
    weight_decay: float


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_sections(raw: dict[str, Any]) -> None:
    expected = {"experiment", "data", "model", "training"}
    missing = expected - raw.keys()
    extra = raw.keys() - expected
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unknown configuration sections: {sorted(extra)}")


def _check_keys(section: str, values: dict[str, Any], expected: set[str]) -> None:
    missing = expected - values.keys()
    extra = values.keys() - expected
    if missing:
        raise ValueError(f"Missing keys in '{section}': {sorted(missing)}")
    if extra:
        raise ValueError(f"Unknown keys in '{section}': {sorted(extra)}")


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    _require_sections(raw)

    experiment = raw["experiment"]
    data = raw["data"]
    model = raw["model"]
    training = raw["training"]
    if not all(isinstance(section, dict) for section in (experiment, data, model, training)):
        raise ValueError("Every configuration section must be a mapping")

    _check_keys("experiment", experiment, set(ExperimentConfig.__dataclass_fields__))
    _check_keys("data", data, set(DataConfig.__dataclass_fields__))
    _check_keys("model", model, set(ModelConfig.__dataclass_fields__))
    _check_keys("training", training, set(TrainingConfig.__dataclass_fields__))

    config = Config(
        experiment=ExperimentConfig(**experiment),
        data=DataConfig(
            **{key: value for key, value in data.items() if key != "task_classes"},
            task_classes=tuple(tuple(classes) for classes in data["task_classes"]),
        ),
        model=ModelConfig(
            **{key: value for key, value in model.items() if key != "channels"},
            channels=tuple(model["channels"]),
        ),
        training=TrainingConfig(**training),
    )
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    if config.experiment.seed < 0:
        raise ValueError("experiment.seed must be non-negative")
    if config.experiment.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("experiment.device must be auto, cpu, cuda, or mps")
    if not config.data.task_classes:
        raise ValueError("data.task_classes cannot be empty")
    if config.data.protocol not in {"sequential", "joint"}:
        raise ValueError("data.protocol must be sequential or joint")

    flat_classes = [class_id for task in config.data.task_classes for class_id in task]
    expected_classes = list(range(config.model.num_classes))
    if sorted(flat_classes) != expected_classes:
        raise ValueError(
            "data.task_classes must contain every class from 0 to "
            f"{config.model.num_classes - 1} exactly once"
        )
    if any(not task for task in config.data.task_classes):
        raise ValueError("Each task must contain at least one class")
    if len(config.model.channels) != 2 or any(value <= 0 for value in config.model.channels):
        raise ValueError("model.channels must contain two positive integers")
    if config.model.hidden_dim <= 0:
        raise ValueError("model.hidden_dim must be positive")
    if not 0.0 <= config.model.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if config.training.method != "naive":
        raise ValueError("Only training.method='naive' is implemented")
    if config.training.optimizer not in {"sgd", "adamw"}:
        raise ValueError("training.optimizer must be sgd or adamw")
    if config.training.epochs_per_task <= 0:
        raise ValueError("training.epochs_per_task must be positive")
    if config.training.learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive")
    if config.data.train_batch_size <= 0 or config.data.eval_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if config.data.num_workers < 0:
        raise ValueError("data.num_workers cannot be negative")
    for name, value in (
        ("max_train_samples_per_class", config.data.max_train_samples_per_class),
        ("max_eval_samples_per_class", config.data.max_eval_samples_per_class),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"data.{name} must be positive or null")
