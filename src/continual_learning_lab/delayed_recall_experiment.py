from __future__ import annotations

import csv
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from continual_learning_lab.config import DelayedRecallConfig
from continual_learning_lab.delayed_recall_data import (
    DelayedRecallLoaders,
    build_delayed_recall_loaders,
)
from continual_learning_lab.model import count_trainable_parameters
from continual_learning_lab.recurrent_models import build_recurrent_model
from continual_learning_lab.reproducibility import seed_everything
from continual_learning_lab.tracking import (
    flatten_mapping,
    get_or_create_experiment,
    normalize_tracking_uri,
)


@dataclass(frozen=True)
class RecallMetrics:
    loss: float
    accuracy: float
    examples: int


@dataclass(frozen=True)
class RecallRunResult:
    run_id: str
    seed: int
    model_type: str
    alpha: float | None
    trainable_parameters: int
    training_seconds: float
    validation: dict[int, RecallMetrics]
    test: dict[int, RecallMetrics]
    model_state_path: Path


def _write_json(path: Path, values: object) -> None:
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def evaluate_recall(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> RecallMetrics:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        total_loss += criterion(logits, targets).item()
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += targets.size(0)
    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty delayed-recall dataset")
    return RecallMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def _evaluate_delays(
    model: nn.Module,
    loaders: dict[int, DataLoader[Any]],
    device: torch.device,
) -> tuple[dict[int, RecallMetrics], RecallMetrics]:
    by_delay = {
        delay: evaluate_recall(model, loader, device)
        for delay, loader in loaders.items()
    }
    total_examples = sum(result.examples for result in by_delay.values())
    aggregate = RecallMetrics(
        loss=sum(result.loss * result.examples for result in by_delay.values())
        / total_examples,
        accuracy=sum(result.accuracy * result.examples for result in by_delay.values())
        / total_examples,
        examples=total_examples,
    )
    return by_delay, aggregate


def _model_setting_name(model_type: str, alpha: float | None) -> str:
    if alpha is None:
        return model_type
    return f"{model_type}_alpha_{alpha:g}"


def model_settings(config: DelayedRecallConfig) -> list[tuple[str, float | None]]:
    settings: list[tuple[str, float | None]] = [("vanilla_rnn", None)]
    settings.extend(
        ("homogeneous_leaky", alpha) for alpha in config.model.homogeneous_alphas
    )
    settings.append(("heterogeneous_leaky", None))
    if config.model.include_gru:
        settings.append(("gru", None))
    return settings


def _run_model(
    config: DelayedRecallConfig,
    *,
    model_type: str,
    alpha: float | None,
    seed: int,
    device: torch.device,
    parent_dir: Path,
) -> RecallRunResult:
    # Reset before every child so models in a paired seed receive reproducible
    # initialization, generated data, and shuffled batch order.
    seed_everything(seed, config.experiment.deterministic)
    loaders = build_delayed_recall_loaders(
        config.data,
        seed=seed,
        use_pin_memory=device.type == "cuda",
    )
    model = build_recurrent_model(model_type, config.model, alpha=alpha).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    parameter_count = count_trainable_parameters(model)
    setting_name = _model_setting_name(model_type, alpha)
    run_name = f"{config.experiment.run_name}-{setting_name}-seed-{seed}"
    parent_run = mlflow.active_run()
    if parent_run is None:
        raise RuntimeError("Delayed-recall child run requires an active parent run")

    with mlflow.start_run(
        experiment_id=parent_run.info.experiment_id,
        run_name=run_name,
        nested=True,
    ) as active_run:
        child_dir = parent_dir / "children" / active_run.info.run_id
        child_dir.mkdir(parents=True, exist_ok=False)
        child_params = flatten_mapping(config.to_dict())
        child_params.update(
            {
                "seed": seed,
                "model_type": model_type,
                "alpha": "none" if alpha is None else alpha,
                "model.trainable_parameters": parameter_count,
                "training.total_example_exposure": (
                    len(loaders.train.dataset) * config.training.epochs
                ),
            }
        )
        mlflow.log_params(child_params)
        mlflow.set_tags(
            {
                "dataset": "synthetic_delayed_recall",
                "protocol": "mixed_delay",
                "run.role": "architecture_seed_comparison",
                "device.resolved": str(device),
            }
        )

        total_training_seconds = 0.0
        final_validation: dict[int, RecallMetrics] = {}
        for epoch in range(config.training.epochs):
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_examples = 0
            _synchronize_device(device)
            epoch_start = time.perf_counter()
            for inputs, targets in loaders.train:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm
                )
                optimizer.step()

                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (logits.argmax(dim=1) == targets).sum().item()
                total_examples += batch_size
            _synchronize_device(device)
            epoch_seconds = time.perf_counter() - epoch_start
            total_training_seconds += epoch_seconds

            final_validation, validation_aggregate = _evaluate_delays(
                model, loaders.validation, device
            )
            metrics = {
                "train_loss": total_loss / total_examples,
                "train_recall_accuracy": total_correct / total_examples,
                "validation_loss": validation_aggregate.loss,
                "validation_recall_accuracy": validation_aggregate.accuracy,
                "training_epoch_seconds": epoch_seconds,
            }
            for delay, result in final_validation.items():
                metrics[f"validation_delay_{delay}_loss"] = result.loss
                metrics[f"validation_delay_{delay}_recall_accuracy"] = result.accuracy
            mlflow.log_metrics(metrics, step=epoch)
            print(
                f"  {setting_name} seed={seed} epoch "
                f"{epoch + 1:02d}/{config.training.epochs}: "
                f"train_acc={metrics['train_recall_accuracy']:.3f}, "
                f"val_acc={validation_aggregate.accuracy:.3f}",
                flush=True,
            )

        final_test, _test_aggregate = _evaluate_delays(model, loaders.test, device)
        final_step = config.training.epochs - 1
        test_metrics: dict[str, float] = {
            "training_total_seconds": total_training_seconds
        }
        for delay, result in final_test.items():
            test_metrics[f"test_delay_{delay}_loss"] = result.loss
            test_metrics[f"test_delay_{delay}_recall_accuracy"] = result.accuracy
        mlflow.log_metrics(test_metrics, step=final_step)

        model_path = child_dir / "final_model_state.pt"
        summary_path = child_dir / "summary.json"
        torch.save(model.state_dict(), model_path)
        _write_json(
            summary_path,
            {
                "run_id": active_run.info.run_id,
                "seed": seed,
                "model_type": model_type,
                "alpha": alpha,
                "trainable_parameters": parameter_count,
                "training_seconds": total_training_seconds,
                "validation": {
                    str(delay): asdict(result)
                    for delay, result in final_validation.items()
                },
                "test": {
                    str(delay): asdict(result) for delay, result in final_test.items()
                },
            },
        )
        mlflow.log_artifacts(str(child_dir), artifact_path="results")
        return RecallRunResult(
            run_id=active_run.info.run_id,
            seed=seed,
            model_type=model_type,
            alpha=alpha,
            trainable_parameters=parameter_count,
            training_seconds=total_training_seconds,
            validation=final_validation,
            test=final_test,
            model_state_path=model_path,
        )


def select_homogeneous_alpha(
    results: list[RecallRunResult], config: DelayedRecallConfig
) -> float:
    candidates: list[tuple[tuple[float, float, float], float]] = []
    for alpha in config.model.homogeneous_alphas:
        matching = [
            result
            for result in results
            if result.model_type == "homogeneous_leaky"
            and result.alpha is not None
            and math.isclose(result.alpha, alpha)
        ]
        if len(matching) != len(config.experiment.seeds):
            raise ValueError(f"Expected one homogeneous alpha={alpha} run per seed")
        selection_score = statistics.mean(
            result.validation[delay].accuracy
            for result in matching
            for delay in config.evaluation.selection_delays
        )
        all_delay_score = statistics.mean(
            result.validation[delay].accuracy
            for result in matching
            for delay in config.data.delays
        )
        # max() chooses the lower alpha only after both validation scores tie.
        candidates.append(((selection_score, all_delay_score, -alpha), alpha))
    return max(candidates)[1]


def _official_results(
    results: list[RecallRunResult], selected_alpha: float
) -> dict[str, list[RecallRunResult]]:
    return {
        "vanilla_rnn": [r for r in results if r.model_type == "vanilla_rnn"],
        "homogeneous_leaky": [
            r
            for r in results
            if r.model_type == "homogeneous_leaky"
            and r.alpha is not None
            and math.isclose(r.alpha, selected_alpha)
        ],
        "heterogeneous_leaky": [
            r for r in results if r.model_type == "heterogeneous_leaky"
        ],
        "gru": [r for r in results if r.model_type == "gru"],
    }


def evaluate_hypothesis(
    results: list[RecallRunResult],
    config: DelayedRecallConfig,
    selected_alpha: float,
) -> dict[str, object]:
    official = _official_results(results, selected_alpha)
    checks: list[dict[str, object]] = []

    def mean_accuracy(model_name: str, delay: int) -> float:
        return statistics.mean(r.test[delay].accuracy for r in official[model_name])

    heterogeneous = {r.seed: r for r in official["heterogeneous_leaky"]}
    for delay in config.evaluation.long_delays:
        for baseline in ("vanilla_rnn", "homogeneous_leaky"):
            advantage = mean_accuracy("heterogeneous_leaky", delay) - mean_accuracy(
                baseline, delay
            )
            checks.append(
                {
                    "kind": "long_delay_advantage",
                    "delay": delay,
                    "baseline": baseline,
                    "observed": advantage,
                    "required": config.evaluation.min_long_advantage,
                    "passed": advantage >= config.evaluation.min_long_advantage,
                }
            )
            baseline_by_seed = {r.seed: r for r in official[baseline]}
            wins = sum(
                heterogeneous[seed].test[delay].accuracy
                > baseline_by_seed[seed].test[delay].accuracy
                for seed in config.experiment.seeds
            )
            checks.append(
                {
                    "kind": "paired_seed_wins",
                    "delay": delay,
                    "baseline": baseline,
                    "observed": wins,
                    "required": config.evaluation.min_paired_seed_wins,
                    "passed": wins >= config.evaluation.min_paired_seed_wins,
                }
            )

    for delay in config.evaluation.short_delays:
        best_baseline = max(
            mean_accuracy("vanilla_rnn", delay),
            mean_accuracy("homogeneous_leaky", delay),
        )
        penalty = best_baseline - mean_accuracy("heterogeneous_leaky", delay)
        checks.append(
            {
                "kind": "short_delay_penalty",
                "delay": delay,
                "baseline": "best_primary_baseline",
                "observed": penalty,
                "maximum": config.evaluation.max_short_penalty,
                "passed": penalty <= config.evaluation.max_short_penalty,
            }
        )

    return {
        "supported": all(bool(check["passed"]) for check in checks),
        "selected_homogeneous_alpha": selected_alpha,
        "checks": checks,
        "note": "GRU is a secondary reference and does not affect this verdict.",
    }


def _write_results_csv(path: Path, results: list[RecallRunResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "run_id",
                "seed",
                "model_type",
                "alpha",
                "delay",
                "validation_loss",
                "validation_accuracy",
                "test_loss",
                "test_accuracy",
                "trainable_parameters",
                "training_seconds",
            ),
        )
        writer.writeheader()
        for result in results:
            for delay in sorted(result.test):
                writer.writerow(
                    {
                        "run_id": result.run_id,
                        "seed": result.seed,
                        "model_type": result.model_type,
                        "alpha": "" if result.alpha is None else result.alpha,
                        "delay": delay,
                        "validation_loss": result.validation[delay].loss,
                        "validation_accuracy": result.validation[delay].accuracy,
                        "test_loss": result.test[delay].loss,
                        "test_accuracy": result.test[delay].accuracy,
                        "trainable_parameters": result.trainable_parameters,
                        "training_seconds": result.training_seconds,
                    }
                )


def _aggregate_rows(
    results: list[RecallRunResult],
    config: DelayedRecallConfig,
    selected_alpha: float,
) -> list[dict[str, object]]:
    official = _official_results(results, selected_alpha)
    rows: list[dict[str, object]] = []
    for model_name, model_results in official.items():
        if not model_results:
            continue
        for delay in config.data.delays:
            values = [result.test[delay].accuracy for result in model_results]
            rows.append(
                {
                    "model_type": model_name,
                    "delay": delay,
                    "mean_test_accuracy": statistics.mean(values),
                    "sample_std_test_accuracy": statistics.stdev(values)
                    if len(values) > 1
                    else 0.0,
                    "seeds": len(values),
                }
            )
    return rows


def _write_aggregate_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_accuracy_curves(path: Path, rows: list[dict[str, object]]) -> None:
    labels = {
        "vanilla_rnn": "Vanilla RNN",
        "homogeneous_leaky": "Best homogeneous leak",
        "heterogeneous_leaky": "Heterogeneous leak",
        "gru": "GRU (secondary)",
    }
    fig, axis = plt.subplots(figsize=(8, 5))
    for model_name in labels:
        model_rows = [row for row in rows if row["model_type"] == model_name]
        if not model_rows:
            continue
        axis.errorbar(
            [int(row["delay"]) for row in model_rows],
            [float(row["mean_test_accuracy"]) for row in model_rows],
            yerr=[float(row["sample_std_test_accuracy"]) for row in model_rows],
            marker="o",
            capsize=3,
            label=labels[model_name],
        )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")
    axis.set_xlabel("Delay (intervening timesteps)")
    axis.set_ylabel("Recall accuracy")
    axis.set_ylim(0.0, 1.01)
    axis.set_title("Experiment 002: delayed recall by recurrent architecture")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_per_seed_accuracy_curves(
    path: Path,
    results: list[RecallRunResult],
    selected_alpha: float,
) -> None:
    labels = {
        "vanilla_rnn": "Vanilla RNN",
        "homogeneous_leaky": "Best homogeneous leak",
        "heterogeneous_leaky": "Heterogeneous leak",
        "gru": "GRU (secondary)",
    }
    official = _official_results(results, selected_alpha)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, (model_name, title) in zip(axes.ravel(), labels.items(), strict=True):
        model_results = official[model_name]
        for result in sorted(model_results, key=lambda item: item.seed):
            delays = sorted(result.test)
            axis.plot(
                delays,
                [result.test[delay].accuracy for delay in delays],
                marker="o",
                label=f"seed {result.seed}",
            )
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_ylim(0.0, 1.01)
        axis.grid(alpha=0.25)
        if model_results:
            axis.legend()
        else:
            axis.text(0.5, 0.5, "Not included", ha="center", va="center")
    fig.supxlabel("Delay (intervening timesteps)")
    fig.supylabel("Test recall accuracy")
    fig.suptitle("Experiment 002: individual-seed recall curves")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _log_parent_test_metric_curves(
    results: list[RecallRunResult], selected_alpha: float
) -> None:
    """Expose delay curves on the parent run; MLflow step means delay here."""
    official = _official_results(results, selected_alpha)
    delays = sorted(next(iter(official.values()))[0].test)
    for delay in delays:
        metrics: dict[str, float] = {}
        for model_name, model_results in official.items():
            if not model_results:
                continue
            values = [result.test[delay].accuracy for result in model_results]
            metrics[f"test_mean_{model_name}_recall_accuracy"] = statistics.mean(values)
            for result in model_results:
                metrics[
                    f"test_seed_{result.seed}_{model_name}_recall_accuracy"
                ] = result.test[delay].accuracy
        mlflow.log_metrics(metrics, step=delay)


def _log_parent_learning_metric_curves(
    results: list[RecallRunResult], selected_alpha: float
) -> None:
    """Expose aggregate and seed validation curves; MLflow step means epoch."""
    official = _official_results(results, selected_alpha)
    client = mlflow.MlflowClient()
    by_step: dict[int, dict[str, float]] = {}
    for model_name, model_results in official.items():
        if not model_results:
            continue
        validation_histories: dict[int, dict[int, float]] = {}
        train_histories: dict[int, dict[int, float]] = {}
        for result in model_results:
            validation_histories[result.seed] = {
                point.step: point.value
                for point in client.get_metric_history(
                    result.run_id, "validation_recall_accuracy"
                )
            }
            train_histories[result.seed] = {
                point.step: point.value
                for point in client.get_metric_history(
                    result.run_id, "train_recall_accuracy"
                )
            }
        common_steps = set.intersection(
            *(set(history) for history in validation_histories.values()),
            *(set(history) for history in train_histories.values()),
        )
        for step in sorted(common_steps):
            step_metrics = by_step.setdefault(step, {})
            step_metrics[
                f"epoch_mean_{model_name}_validation_recall_accuracy"
            ] = statistics.mean(
                history[step] for history in validation_histories.values()
            )
            step_metrics[f"epoch_mean_{model_name}_train_recall_accuracy"] = (
                statistics.mean(history[step] for history in train_histories.values())
            )
            for seed, history in validation_histories.items():
                step_metrics[
                    f"epoch_seed_{seed}_{model_name}_validation_recall_accuracy"
                ] = history[step]
    for step, metrics in sorted(by_step.items()):
        mlflow.log_metrics(metrics, step=step)


def _inspection_result(
    results: list[RecallRunResult],
    *,
    model_type: str,
    seed: int,
    alpha: float | None = None,
) -> RecallRunResult:
    matches = [
        result
        for result in results
        if result.model_type == model_type
        and result.seed == seed
        and (
            (alpha is None and result.alpha is None)
            or (
                alpha is not None
                and result.alpha is not None
                and math.isclose(result.alpha, alpha)
            )
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one inspection run for {model_type}, seed={seed}")
    return matches[0]


def _write_hidden_state_artifacts(
    output_dir: Path,
    results: list[RecallRunResult],
    config: DelayedRecallConfig,
    selected_alpha: float,
    device: torch.device,
) -> None:
    seed = config.evaluation.inspection_seed
    delay = config.evaluation.inspection_delay
    loaders = build_delayed_recall_loaders(
        config.data, seed=seed, use_pin_memory=device.type == "cuda"
    )
    inputs, targets = loaders.test_datasets[delay].tensors
    selected_indices: list[int] = []
    for target in (0, 1):
        indices = torch.nonzero(targets == target, as_tuple=False).flatten()
        selected_indices.extend(
            indices[: config.evaluation.inspection_examples_per_class].tolist()
        )
    selected_inputs = inputs[selected_indices]
    selected_targets = targets[selected_indices]
    torch.save(
        {"inputs": selected_inputs, "targets": selected_targets, "delay": delay},
        output_dir / "inspection_examples.pt",
    )

    settings = (
        ("vanilla_rnn", None, "vanilla_rnn"),
        ("homogeneous_leaky", selected_alpha, "homogeneous_leaky"),
        ("heterogeneous_leaky", None, "heterogeneous_leaky"),
        ("gru", None, "gru"),
    )
    value_index, cue_index = _value_and_cue_indices(config, delay)
    for model_type, alpha, label in settings:
        if model_type == "gru" and not config.model.include_gru:
            continue
        result = _inspection_result(
            results, model_type=model_type, seed=seed, alpha=alpha
        )
        seed_everything(seed, config.experiment.deterministic)
        model = build_recurrent_model(model_type, config.model, alpha=alpha).to(device)
        state = torch.load(result.model_state_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            states = model.hidden_states(selected_inputs.to(device)).cpu()
        torch.save(
            {
                "inputs": selected_inputs,
                "targets": selected_targets,
                "hidden_states": states,
                "model_type": model_type,
                "alpha": alpha,
                "delay": delay,
                "seed": seed,
            },
            output_dir / f"hidden_states_{label}.pt",
        )
        _plot_hidden_heatmap(
            output_dir / f"hidden_states_{label}.png",
            states,
            selected_targets,
            delay=delay,
            value_index=value_index,
            cue_index=cue_index,
            title=label.replace("_", " ").title(),
            group_sizes=config.model.heterogeneous_group_sizes
            if model_type == "heterogeneous_leaky"
            else None,
        )
        if model_type == "heterogeneous_leaky":
            _plot_heterogeneous_group_traces(
                output_dir / "heterogeneous_group_traces.png",
                states,
                selected_targets,
                config,
                value_index=value_index,
                cue_index=cue_index,
            )


def _value_and_cue_indices(
    config: DelayedRecallConfig, delay: int
) -> tuple[int, int]:
    if config.data.timing_protocol == "fixed_initial_value":
        return 0, delay + 1
    max_delay = max(config.data.delays)
    return max_delay - delay, max_delay + 1


def _plot_hidden_heatmap(
    path: Path,
    states: torch.Tensor,
    targets: torch.Tensor,
    *,
    delay: int,
    value_index: int,
    cue_index: int,
    title: str,
    group_sizes: tuple[int, ...] | None,
) -> None:
    columns = states.size(0)
    fig, axes = plt.subplots(
        1,
        columns,
        figsize=(5 * columns, 5),
        squeeze=False,
        constrained_layout=True,
    )
    limit = max(float(states.abs().max()), 1e-6)
    for index, axis in enumerate(axes[0]):
        image = axis.imshow(
            states[index].T.numpy(),
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axis.axvline(value_index, color="lime", linewidth=1.5, label="value")
        axis.axvline(cue_index, color="yellow", linewidth=1.5, label="cue")
        if group_sizes is not None:
            boundary = 0
            for size in group_sizes[:-1]:
                boundary += size
                axis.axhline(boundary - 0.5, color="black", linewidth=1)
        recalled_value = -1 if int(targets[index]) == 0 else 1
        axis.set_title(f"Target {recalled_value:+d}")
        axis.set_xlabel("Timestep")
        axis.set_ylabel("Hidden neuron")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(f"{title}: delay-{delay} hidden activity")
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.8,
        pad=0.03,
        label="Activation",
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_heterogeneous_group_traces(
    path: Path,
    states: torch.Tensor,
    targets: torch.Tensor,
    config: DelayedRecallConfig,
    *,
    value_index: int,
    cue_index: int,
) -> None:
    fig, axes = plt.subplots(states.size(0), 1, figsize=(9, 3 * states.size(0)), squeeze=False)
    start = 0
    group_slices: list[tuple[float, slice]] = []
    for alpha, size in zip(
        config.model.heterogeneous_alphas,
        config.model.heterogeneous_group_sizes,
        strict=True,
    ):
        group_slices.append((alpha, slice(start, start + size)))
        start += size
    for example_index, axis in enumerate(axes[:, 0]):
        for alpha, group_slice in group_slices:
            axis.plot(
                states[example_index, :, group_slice].abs().mean(dim=1).numpy(),
                label=f"alpha={alpha:g}",
            )
        axis.axvline(value_index, color="lime", linewidth=1.5, label="value")
        axis.axvline(cue_index, color="gold", linewidth=1.5, label="cue")
        recalled_value = -1 if int(targets[example_index]) == 0 else 1
        axis.set_title(
            f"Target {recalled_value:+d}: mean absolute activation by leak group"
        )
        axis.set_xlabel("Timestep")
        axis.set_ylabel("Mean |hidden activation|")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _build_parent_report(
    rows: list[dict[str, object]], verdict: dict[str, object]
) -> str:
    supported = bool(verdict["supported"])
    lines = [
        "# Experiment 002 — Heterogeneous leaky RNN delayed recall",
        "",
        "## Hypothesis verdict",
        "",
        (
            "The preregistered exploratory support criteria were satisfied."
            if supported
            else "The preregistered exploratory support criteria were not all satisfied."
        ),
        "",
        f"Selected homogeneous alpha: `{verdict['selected_homogeneous_alpha']}`",
        "",
        "GRU is shown as a secondary learned-gating reference and does not affect the verdict.",
        "",
        "In parent-run `test_*` metric series, MLflow step is the delay length. "
        "In `epoch_*` series, step is the zero-based training epoch.",
        "",
        "## Mean test recall accuracy",
        "",
        "| Model | Delay | Mean | Sample SD |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {str(row['model_type']).replace('_', ' ')} | {row['delay']} | "
            f"{float(row['mean_test_accuracy']):.2%} | "
            f"{float(row['sample_std_test_accuracy']):.2%} |"
        )
    lines.extend(["", "## Preregistered checks", ""])
    for check in verdict["checks"]:
        assert isinstance(check, dict)
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(
            f"- **{status}:** {check['kind']} at delay {check['delay']} "
            f"against {check['baseline']} — observed `{check['observed']}`."
        )
    return "\n".join(lines)


def run_delayed_recall_experiment(
    config: DelayedRecallConfig,
    config_path: str | Path,
    device: torch.device,
) -> Path:
    mlflow.set_tracking_uri(normalize_tracking_uri(config.experiment.tracking_uri))
    experiment_id = get_or_create_experiment(
        name=config.experiment.name,
        artifact_dir=config.experiment.artifact_dir,
        description=(
            "Tests whether fixed heterogeneous leak rates improve long-delay recall "
            "over vanilla and validation-selected homogeneous recurrent dynamics."
        ),
    )
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=config.experiment.run_name,
    ) as parent_run:
        parent_dir = Path(config.experiment.output_dir) / parent_run.info.run_id
        aggregate_dir = parent_dir / "aggregate"
        aggregate_dir.mkdir(parents=True, exist_ok=False)
        mlflow.log_params(flatten_mapping(config.to_dict()))
        mlflow.set_tags(
            {
                "dataset": "synthetic_delayed_recall",
                "protocol": "mixed_delay",
                "run.role": "architecture_benchmark_parent",
                "hypothesis.owner": "learner",
                "device.requested": config.experiment.device,
                "device.resolved": str(device),
                "python.version": platform.python_version(),
                "torch.version": torch.__version__,
            }
        )
        mlflow.log_artifact(str(Path(config_path).resolve()), artifact_path="configuration")

        settings = model_settings(config)

        results: list[RecallRunResult] = []
        print(
            f"Running {len(settings) * len(config.experiment.seeds)} paired "
            f"delayed-recall child runs on {device}...",
            flush=True,
        )
        for seed in config.experiment.seeds:
            for model_type, alpha in settings:
                results.append(
                    _run_model(
                        config,
                        model_type=model_type,
                        alpha=alpha,
                        seed=seed,
                        device=device,
                        parent_dir=parent_dir,
                    )
                )

        selected_alpha = select_homogeneous_alpha(results, config)
        verdict = evaluate_hypothesis(results, config, selected_alpha)
        rows = _aggregate_rows(results, config, selected_alpha)
        _write_results_csv(aggregate_dir / "per_seed_results.csv", results)
        _write_aggregate_csv(aggregate_dir / "aggregate_results.csv", rows)
        _write_json(aggregate_dir / "aggregate_results.json", rows)
        _write_json(aggregate_dir / "hypothesis_verdict.json", verdict)
        _plot_accuracy_curves(aggregate_dir / "recall_accuracy_vs_delay.png", rows)
        _plot_per_seed_accuracy_curves(
            aggregate_dir / "per_seed_recall_accuracy_vs_delay.png",
            results,
            selected_alpha,
        )
        _log_parent_test_metric_curves(results, selected_alpha)
        _log_parent_learning_metric_curves(results, selected_alpha)
        _write_hidden_state_artifacts(
            aggregate_dir, results, config, selected_alpha, device
        )
        report = _build_parent_report(rows, verdict)
        (aggregate_dir / "run_summary.md").write_text(report, encoding="utf-8")
        _write_json(aggregate_dir / "resolved_config.json", config.to_dict())

        mlflow.log_param("selected_homogeneous_alpha", selected_alpha)
        mlflow.log_metric("hypothesis_supported", float(bool(verdict["supported"])))
        mlflow.set_tag("hypothesis.verdict", "supported" if verdict["supported"] else "not_supported")
        mlflow.set_tag("mlflow.note.content", report)
        mlflow.log_artifacts(str(aggregate_dir), artifact_path="aggregate")
        print(f"\nExperiment 002 suite complete: {parent_run.info.run_id}")
        print(f"Artifacts: {parent_dir.resolve()}")
        return parent_dir
