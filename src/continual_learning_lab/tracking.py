from __future__ import annotations

import platform
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow
import torch
import torchvision

from continual_learning_lab.config import Config


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


@contextmanager
def start_run(config: Config) -> Iterator[mlflow.ActiveRun]:
    mlflow.set_tracking_uri(normalize_tracking_uri(config.experiment.tracking_uri))
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(config.experiment.name)
    if experiment is None:
        experiment_id = client.create_experiment(
            config.experiment.name,
            artifact_location=Path(config.experiment.artifact_dir).resolve().as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=config.experiment.run_name,
    ) as active_run:
        mlflow.log_params(flatten_mapping(config.to_dict()))
        mlflow.set_tags(
            {
                "protocol": "class-incremental",
                "dataset": "Split CIFAR-10",
                "method": config.training.method,
                "python.version": platform.python_version(),
                "torch.version": torch.__version__,
                "torchvision.version": torchvision.__version__,
                "device.requested": config.experiment.device,
            }
        )
        yield active_run

