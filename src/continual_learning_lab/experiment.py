from __future__ import annotations

import json
from pathlib import Path

import mlflow
import torch

from continual_learning_lab.config import (
    AddingProblemConfig,
    Config,
    DelayedRecallConfig,
    LoadedConfig,
)
from continual_learning_lab.data import (
    JointLoaders,
    TaskLoaders,
    build_joint_cifar10,
    build_split_cifar10,
)
from continual_learning_lab.evaluation import EvaluationResult, evaluate
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


def _run_sequential(
    config: Config,
    device: torch.device,
    learner: ContinualLearner,
    tasks: list[TaskLoaders],
    run_dir: Path,
) -> AccuracyMatrix:
    matrix = AccuracyMatrix(len(tasks))
    for task in tasks:
        print(f"\nTraining task {task.task_id}: classes {task.classes}")
        for epoch in range(config.training.epochs_per_task):
            result = learner.train_epoch(task.task_id, epoch, task.train)
            global_epoch = task.task_id * config.training.epochs_per_task + result.epoch
            mlflow.log_metrics(
                {
                    "train_loss": result.loss,
                    "train_accuracy": result.accuracy,
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
                    f"test_task_{evaluated_task.task_id}_accuracy": result.accuracy,
                },
                step=task.task_id,
            )
            print(f"    task {evaluated_task.task_id}: accuracy={result.accuracy:.3f}")

        summary = matrix.stage_summary(task.task_id)
        mlflow.log_metrics(
            {
                "summary_average_accuracy": summary["average_accuracy"],
                "summary_average_forgetting": summary["average_forgetting"],
            },
            step=task.task_id,
        )
        print(
            f"  seen-task average accuracy={summary['average_accuracy']:.3f}, "
            f"average forgetting={summary['average_forgetting']:.3f}"
        )
        matrix.write_csv(run_dir / "accuracy_matrix.csv")

    return matrix


def _run_joint(
    config: Config,
    device: torch.device,
    learner: ContinualLearner,
    loaders: JointLoaders,
) -> dict[str, object]:
    """Train one stage whose loader contains all ten CIFAR-10 classes.

    ``task_id=0`` is only a stable interface value for the learner. It does not
    make this a two-class task: every epoch iterates over the all-class loader.
    """
    total_example_exposure = 0
    final_overall_result: EvaluationResult | None = None

    for epoch in range(config.training.epochs_per_task):
        train_result = learner.train_epoch(
            task_id=0,
            epoch=epoch,
            loader=loaders.train,
        )
        total_example_exposure += train_result.examples

        # Evaluate the same ten-class problem after each epoch so the training
        # and generalization curves can be interpreted together.
        final_overall_result = evaluate(
            learner.model,
            loaders.overall_test,
            device,
        )
        mlflow.log_metrics(
            {
                "train_loss": train_result.loss,
                "train_accuracy": train_result.accuracy,
                "test_overall_loss": final_overall_result.loss,
                "test_overall_accuracy": final_overall_result.accuracy,
            },
            step=epoch,
        )
        print(
            f"Joint epoch {epoch + 1:02d}/{config.training.epochs_per_task}: "
            f"train_loss={train_result.loss:.4f}, "
            f"train_accuracy={train_result.accuracy:.3f}, "
            f"eval_loss={final_overall_result.loss:.4f}, "
            f"eval_accuracy={final_overall_result.accuracy:.3f}"
        )

    if final_overall_result is None:
        raise RuntimeError("Joint training completed without running an epoch")

    if len(loaders.pair_tests) != len(config.data.task_classes):
        raise ValueError(
            "Joint pair-test loader count must match data.task_classes: "
            f"got {len(loaders.pair_tests)} loaders for "
            f"{len(config.data.task_classes)} class pairs"
        )

    # Each pair loader is only a diagnostic view. The unchanged model still
    # produces ten logits and the labels keep their original global IDs.
    pair_results: list[dict[str, object]] = []
    pair_accuracies: list[float] = []
    final_step = config.training.epochs_per_task - 1
    for pair_id, (classes, loader) in enumerate(
        zip(config.data.task_classes, loaders.pair_tests, strict=True)
    ):
        result = evaluate(learner.model, loader, device)
        pair_accuracies.append(result.accuracy)
        pair_results.append(
            {
                "pair_id": pair_id,
                "classes": list(classes),
                "loss": result.loss,
                "accuracy": result.accuracy,
                "examples": result.examples,
            }
        )
        mlflow.log_metrics(
            {
                f"test_pair_{pair_id}_accuracy": result.accuracy,
            },
            step=final_step,
        )
        print(
            f"Final pair {pair_id} classes={classes}: "
            f"loss={result.loss:.4f}, accuracy={result.accuracy:.3f}"
        )

    mean_pair_accuracy = sum(pair_accuracies) / len(pair_accuracies)
    mlflow.log_param("training.total_example_exposure", total_example_exposure)

    return {
        "total_example_exposure": total_example_exposure,
        "final_overall_loss": final_overall_result.loss,
        "final_overall_accuracy": final_overall_result.accuracy,
        "final_mean_pair_accuracy": mean_pair_accuracy,
        "pair_results": pair_results,
    }


def _build_run_report(config: Config, summary: dict[str, object]) -> str:
    """Create the human-readable explanation shown in MLflow and artifacts."""
    if config.data.protocol == "joint":
        lines = [
            "# Experiment 001 — Joint-training CIFAR-10 oracle",
            "",
            "## Purpose",
            "",
            "Tests whether the unchanged ten-class CNN can learn CIFAR-10 when all "
            "classes remain available. This is a privileged capacity control, not a "
            "continual-learning solution.",
            "",
            "## Final held-out test results",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Overall accuracy | {float(summary['final_overall_accuracy']):.2%} |",
            f"| Overall loss | {float(summary['final_overall_loss']):.4f} |",
            f"| Mean pair accuracy | {float(summary['final_mean_pair_accuracy']):.2%} |",
            f"| Training example presentations | {int(summary['total_example_exposure']):,} |",
            "",
            "| Pair | Classes | Accuracy | Loss |",
            "|---:|---|---:|---:|",
        ]
        pair_results = summary.get("pair_results", [])
        if isinstance(pair_results, list):
            for pair in pair_results:
                if not isinstance(pair, dict):
                    continue
                classes = pair.get("classes", [])
                class_text = ", ".join(str(value) for value in classes)
                lines.append(
                    f"| {int(pair['pair_id'])} | {class_text} | "
                    f"{float(pair['accuracy']):.2%} | {float(pair['loss']):.4f} |"
                )
        lines.extend(
            [
                "",
                "## How to read the metrics",
                "",
                "- `train_*` is measured on augmented training batches while parameters "
                "are being updated.",
                "- `test_overall_*` is measured after each epoch on the unaugmented, "
                "held-out ten-class test set with dropout disabled and gradients off.",
                "- The train and overall-test curves both use epoch steps 0–14 so "
                "each test point shows generalization after that epoch's updates.",
                "- Each `test_pair_*_accuracy` metric contains one point at step 14. "
                "The pair views are evaluated only after the final epoch.",
                "- The final test result is the step-14 `test_overall_*` point and is "
                "also recorded in `summary.json`.",
                "",
                "No forgetting metric is reported because old classes are never "
                "withdrawn during joint training.",
            ]
        )
        return "\n".join(lines)

    return "\n".join(
        [
            "# Experiment 000 — Naive sequential Split CIFAR-10 baseline",
            "",
            "## Purpose",
            "",
            "Measures catastrophic forgetting when one shared ten-class CNN learns "
            "five disjoint class pairs sequentially without replay or another "
            "continual-learning mechanism.",
            "",
            "## Final held-out test results",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Final average accuracy | {float(summary['final_average_accuracy']):.2%} |",
            f"| Final average forgetting | {float(summary['final_average_forgetting']):.2%} |",
            "",
            "## How to read the metrics",
            "",
            "- `train_*` is measured on the current task's augmented training batches "
            "while parameters are being updated.",
            "- `test_task_*_accuracy` is held-out test performance for every class pair after "
            "each sequential training stage.",
            "- `summary_*` aggregates only tasks seen by that stage.",
            "- `train_*` uses global epoch steps 0–74; test and summary metrics use "
            "sequential task-stage steps 0–4.",
            "- The step-4 summary is the final conclusion; task-0 forgetting is "
            "correctly zero because no earlier performance exists to lose.",
            "",
            "All evaluations use one shared ten-output classifier with global labels.",
        ]
    )


def run_experiment(config: LoadedConfig, config_path: str | Path) -> Path:
    if isinstance(config, AddingProblemConfig):
        from continual_learning_lab.adding_problem_experiment import (
            run_adding_problem_experiment,
        )

        device = resolve_device(config.experiment.device)
        print(f"Using device: {device}", flush=True)
        return run_adding_problem_experiment(config, config_path, device)

    if isinstance(config, DelayedRecallConfig):
        # Import locally so the delayed-recall runner can reuse resolve_device
        # without creating a module-import cycle for the legacy CIFAR path.
        from continual_learning_lab.delayed_recall_experiment import (
            run_delayed_recall_experiment,
        )

        device = resolve_device(config.experiment.device)
        print(f"Using device: {device}", flush=True)
        return run_delayed_recall_experiment(config, config_path, device)

    seed_everything(config.experiment.seed, config.experiment.deterministic)
    device = resolve_device(config.experiment.device)

    device_description = str(device)
    if device.type == "cuda":
        device_description += f" ({torch.cuda.get_device_name(device)})"
    print(f"Using device: {device_description}", flush=True)
    if config.data.protocol == "sequential":
        print("Preparing sequential Split CIFAR-10 data...", flush=True)
        loaders: list[TaskLoaders] | JointLoaders = build_split_cifar10(
            config.data,
            config.experiment.seed,
            use_pin_memory=device.type == "cuda",
        )
    elif config.data.protocol == "joint":
        print("Preparing joint CIFAR-10 data...", flush=True)
        loaders = build_joint_cifar10(
            config.data,
            config.experiment.seed,
            use_pin_memory=device.type == "cuda",
        )
    else:
        # Config validation should make this unreachable. Keeping the error
        # here gives a local failure if a future protocol is wired incompletely.
        raise ValueError(f"Unsupported data protocol: {config.data.protocol!r}")
    learner = build_learner(config, device)

    print(f"Trainable parameters: {count_trainable_parameters(learner.model):,}")

    with start_run(config) as active_run:
        run_dir = Path(config.experiment.output_dir) / active_run.info.run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        mlflow.set_tag("device.resolved", str(device))
        mlflow.log_param("model.trainable_parameters", count_trainable_parameters(learner.model))
        mlflow.log_artifact(str(Path(config_path).resolve()), artifact_path="configuration")

        if config.data.protocol == "sequential":
            if not isinstance(loaders, list):
                raise TypeError("Sequential protocol requires task loaders")
            matrix = _run_sequential(config, device, learner, loaders, run_dir)
            final_stage = len(loaders) - 1
            summary: dict[str, object] = {
                "run_id": active_run.info.run_id,
                "final_average_accuracy": matrix.average_accuracy(final_stage),
                "final_average_forgetting": matrix.average_forgetting(final_stage),
                "device": str(device),
            }
        else:
            if not isinstance(loaders, JointLoaders):
                raise TypeError("Joint protocol requires joint loaders")
            joint_summary = _run_joint(config, device, learner, loaders)
            summary = {
                "run_id": active_run.info.run_id,
                "device": str(device),
                **joint_summary,
            }
        summary_path = run_dir / "summary.json"
        config_copy_path = run_dir / "resolved_config.json"
        model_path = run_dir / "final_model_state.pt"
        _write_json(summary_path, summary)
        _write_json(config_copy_path, config.to_dict())
        torch.save(learner.model.state_dict(), model_path)

        run_report = _build_run_report(config, summary)
        mlflow.set_tag("mlflow.note.content", run_report)
        mlflow.log_text(run_report, "reports/run_summary.md")

        mlflow.log_artifacts(str(run_dir), artifact_path="results")
        print(f"\nRun complete: {active_run.info.run_id}")
        print(f"Artifacts: {run_dir.resolve()}")
        return run_dir
