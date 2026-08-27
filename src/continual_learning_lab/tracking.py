from __future__ import annotations

import platform
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow
import torch
import torchvision

from continual_learning_lab.config import Config


def _experiment_description(config: Config) -> str:
    if config.data.protocol == "joint":
        return (
            "Joint-training CIFAR-10 capacity oracle. All ten classes remain "
            "available throughout training; this is a privileged control, not a "
            "continual-learning solution."
        )
    return (
        "Naive sequential Split CIFAR-10 baseline. Five class pairs arrive in "
        "sequence and the shared ten-class model retains no old data or model state."
    )


def normalize_tracking_uri(uri: str) -> str:
    """Make a relative SQLite URI stable regardless of MLflow internals."""
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return uri
    database = Path(uri[len(prefix) :])
    if database.is_absolute():
        return uri
    return f"{prefix}{database.resolve().as_posix()}"


def flatten_mapping(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_mapping(value, name))
        elif isinstance(value, (list, tuple)):
            flat[name] = str(value)
        elif value is None:
            flat[name] = "null"
        else:
            flat[name] = value
    return flat


def get_or_create_experiment(
    *,
    name: str,
    artifact_dir: str,
    description: str,
) -> str:
    """Resolve one MLflow experiment for either a single run or a run suite."""
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = client.create_experiment(
            name,
            artifact_location=Path(artifact_dir).resolve().as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    client.set_experiment_tag(experiment_id, "mlflow.note.content", description)
    return experiment_id


@contextmanager
def start_run(config: Config) -> Iterator[mlflow.ActiveRun]:
    mlflow.set_tracking_uri(normalize_tracking_uri(config.experiment.tracking_uri))
    experiment_id = get_or_create_experiment(
        name=config.experiment.name,
        artifact_dir=config.experiment.artifact_dir,
        description=_experiment_description(config),
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=config.experiment.run_name,
    ) as active_run:
        mlflow.log_params(flatten_mapping(config.to_dict()))
        mlflow.set_tags(
            {
                "protocol": config.data.protocol,
                "dataset": "CIFAR-10",
                "method": config.training.method,
                "run.role": (
                    "joint-training capacity oracle"
                    if config.data.protocol == "joint"
                    else "continual-learning baseline"
                ),
                "evaluation": "shared-head ten-class classification",
                "python.version": platform.python_version(),
                "torch.version": torch.__version__,
                "torchvision.version": torchvision.__version__,
                "device.requested": config.experiment.device,
            }
        )
        yield active_run
