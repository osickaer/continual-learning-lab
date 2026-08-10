from __future__ import annotations

import json
from pathlib import Path

import mlflow
import torch

from continual_learning_lab.config import Config
from continual_learning_lab.data import TaskLoaders, build_split_cifar10
from continual_learning_lab.evaluation import evaluate
from continual_learning_lab.learners.base import ContinualLearner
from continual_learning_lab.learners.naive import NaiveSequentialLearner
from continual_learning_lab.metrics import AccuracyMatrix
from continual_learning_lab.model import SmallCNN, count_trainable_parameters
from continual_learning_lab.reproducibility import seed_everything
from continual_learning_lab.tracking import start_run


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def build_learner(config: Config, device: torch.device) -> ContinualLearner:
    model = SmallCNN(config.model).to(device)
    if config.training.method == "naive":
        return NaiveSequentialLearner(model, config.training, device)
    raise ValueError(f"Unsupported continual-learning method: {config.training.method}")


def _write_json(path: Path, values: object) -> None:
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")


def _run_tasks(
    config: Config,
    device: torch.device,
    learner: ContinualLearner,
    tasks: list[TaskLoaders],
    run_dir: Path,
) -> AccuracyMatrix:
    matrix = AccuracyMatrix(len(tasks))
    for task in tasks:
        print(f"\nTraining task {task.task_id}: classes {task.classes}")
        epoch_results = learner.train_task(task.task_id, task.train)
        for result in epoch_results:
            global_epoch = task.task_id * config.training.epochs_per_task + result.epoch
            mlflow.log_metrics(
                {
                    "train_loss": result.loss,
                    "train_accuracy": result.accuracy,
                    "train_task": float(task.task_id),
                },
                step=global_epoch,
            )
            print(
                f"  epoch {result.epoch + 1:02d}/{config.training.epochs_per_task}: "
                f"loss={result.loss:.4f}, accuracy={result.accuracy:.3f}"
            )

        print("  evaluation:")
        for evaluated_task in tasks:
            result = evaluate(learner.model, evaluated_task.test, device)
            matrix.update(task.task_id, evaluated_task.task_id, result.accuracy)
            mlflow.log_metrics(
                {
                    f"eval_task_{evaluated_task.task_id}_accuracy": result.accuracy,
                    f"eval_task_{evaluated_task.task_id}_loss": result.loss,
                },
                step=task.task_id,
            )
            print(f"    task {evaluated_task.task_id}: accuracy={result.accuracy:.3f}")

        summary = matrix.stage_summary(task.task_id)
        mlflow.log_metrics(summary, step=task.task_id)
        print(
            f"  seen-task average accuracy={summary['average_accuracy']:.3f}, "
            f"average forgetting={summary['average_forgetting']:.3f}"
        )
        matrix.write_csv(run_dir / "accuracy_matrix.csv")

    return matrix


def run_experiment(config: Config, config_path: str | Path) -> Path:
    seed_everything(config.experiment.seed, config.experiment.deterministic)
    device = resolve_device(config.experiment.device)

    device_description = str(device)
    if device.type == "cuda":
        device_description += f" ({torch.cuda.get_device_name(device)})"
    print(f"Using device: {device_description}", flush=True)
    print("Preparing Split CIFAR-10 data...", flush=True)

    tasks = build_split_cifar10(
        config.data,
        config.experiment.seed,
        use_pin_memory=device.type == "cuda",
    )
    learner = build_learner(config, device)

    print(f"Trainable parameters: {count_trainable_parameters(learner.model):,}")

    with start_run(config) as active_run:
        run_dir = Path(config.experiment.output_dir) / active_run.info.run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        mlflow.set_tag("device.resolved", str(device))
        mlflow.log_param("model.trainable_parameters", count_trainable_parameters(learner.model))
        mlflow.log_artifact(str(Path(config_path).resolve()), artifact_path="configuration")

        matrix = _run_tasks(config, device, learner, tasks, run_dir)
        final_stage = len(tasks) - 1
        summary = {
            "run_id": active_run.info.run_id,
            "final_average_accuracy": matrix.average_accuracy(final_stage),
            "final_average_forgetting": matrix.average_forgetting(final_stage),
            "device": str(device),
        }
        summary_path = run_dir / "summary.json"
        config_copy_path = run_dir / "resolved_config.json"
        model_path = run_dir / "final_model_state.pt"
        _write_json(summary_path, summary)
        _write_json(config_copy_path, config.to_dict())
        torch.save(learner.model.state_dict(), model_path)

        mlflow.log_artifacts(str(run_dir), artifact_path="results")
        print(f"\nRun complete: {active_run.info.run_id}")
        print(f"Artifacts: {run_dir.resolve()}")
        return run_dir
