from __future__ import annotations

import csv
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from continual_learning_lab.adding_problem_data import (
    AddingProblemLoaders,
    build_adding_problem_loaders,
)
from continual_learning_lab.config import AddingProblemConfig
from continual_learning_lab.model import count_trainable_parameters
from continual_learning_lab.recurrent_models import build_recurrent_model
from continual_learning_lab.reproducibility import make_generator, seed_everything
from continual_learning_lab.tracking import (
    flatten_mapping,
    get_or_create_experiment,
    normalize_tracking_uri,
)


@dataclass(frozen=True)
class AddingMetrics:
    mse: float
    mae: float
    examples: int


@dataclass(frozen=True)
class AddingRunResult:
    run_id: str
    seed: int
    model_type: str
    alpha: float | None
    trainable_parameters: int
    training_seconds: float
    validation: dict[int, AddingMetrics]
    test: dict[int, AddingMetrics]
    model_state_path: Path


def _write_json(path: Path, values: object) -> None:
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _setting_name(model_type: str, alpha: float | None) -> str:
    return model_type if alpha is None else f"{model_type}_alpha_{alpha:g}"


def model_settings(config: AddingProblemConfig) -> list[tuple[str, float | None]]:
    settings: list[tuple[str, float | None]] = [("vanilla_rnn", None)]
    settings.extend(
        ("homogeneous_leaky", alpha) for alpha in config.model.homogeneous_alphas
    )
    settings.append(("heterogeneous_leaky", None))
    if config.model.include_gru:
        settings.append(("gru", None))
    return settings


@torch.no_grad()
def evaluate_adding(
    model: nn.Module, loader: DataLoader[Any], device: torch.device
) -> AddingMetrics:
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    examples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = model(inputs)
        errors = predictions - targets
        squared_error += errors.square().sum().item()
        absolute_error += errors.abs().sum().item()
        examples += targets.size(0)
    if examples == 0:
        raise ValueError("Cannot evaluate an empty Adding Problem dataset")
    return AddingMetrics(
        mse=squared_error / examples,
        mae=absolute_error / examples,
        examples=examples,
    )


def _evaluate_lengths(
    model: nn.Module,
    loaders: dict[int, DataLoader[Any]],
    device: torch.device,
) -> tuple[dict[int, AddingMetrics], AddingMetrics]:
    by_length = {
        length: evaluate_adding(model, loader, device)
        for length, loader in loaders.items()
    }
    examples = sum(metric.examples for metric in by_length.values())
    return by_length, AddingMetrics(
        mse=sum(metric.mse * metric.examples for metric in by_length.values()) / examples,
        mae=sum(metric.mae * metric.examples for metric in by_length.values()) / examples,
        examples=examples,
    )


def _epoch_batches(
    loaders: dict[int, DataLoader[Any]], *, seed: int, epoch: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Interleave fixed-length batches in a paired deterministic order."""
    lengths = sorted(loaders)
    iterators = {length: iter(loader) for length, loader in loaders.items()}
    schedule = [length for length in lengths for _ in range(len(loaders[length]))]
    permutation = torch.randperm(
        len(schedule), generator=make_generator(seed + 60_000 + epoch)
    )
    for index in permutation.tolist():
        yield next(iterators[schedule[index]])


def _run_model(
    config: AddingProblemConfig,
    *,
    model_type: str,
    alpha: float | None,
    seed: int,
    device: torch.device,
    parent_dir: Path,
) -> AddingRunResult:
    seed_everything(seed, config.experiment.deterministic)
    loaders = build_adding_problem_loaders(
        config.data, seed=seed, use_pin_memory=device.type == "cuda"
    )
    model = build_recurrent_model(
        model_type, config.model, alpha=alpha, readout="final"
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    criterion = nn.MSELoss()
    parameter_count = count_trainable_parameters(model)
    setting_name = _setting_name(model_type, alpha)
    parent_run = mlflow.active_run()
    if parent_run is None:
        raise RuntimeError("Adding Problem child run requires an active parent run")

    with mlflow.start_run(
        experiment_id=parent_run.info.experiment_id,
        run_name=f"{config.experiment.run_name}-{setting_name}-seed-{seed}",
        nested=True,
    ) as active_run:
        child_dir = parent_dir / "children" / active_run.info.run_id
        child_dir.mkdir(parents=True, exist_ok=False)
        params = flatten_mapping(config.to_dict())
        params.update(
            {
                "seed": seed,
                "model_type": model_type,
                "alpha": "none" if alpha is None else alpha,
                "model.trainable_parameters": parameter_count,
                "training.total_example_exposure": (
                    len(config.data.train_lengths)
                    * config.data.train_examples_per_length
                    * config.training.epochs
                ),
            }
        )
        mlflow.log_params(params)
        mlflow.set_tags(
            {
                "dataset": "synthetic_adding_problem",
                "protocol": "mixed_length_regression",
                "run.role": "architecture_seed_comparison",
                "device.resolved": str(device),
            }
        )

        total_training_seconds = 0.0
        final_validation: dict[int, AddingMetrics] = {}
        for epoch in range(config.training.epochs):
            model.train()
            total_squared_error = 0.0
            total_absolute_error = 0.0
            total_examples = 0
            _synchronize_device(device)
            epoch_start = time.perf_counter()
            for inputs, targets in _epoch_batches(loaders.train, seed=seed, epoch=epoch):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm
                )
                optimizer.step()

                errors = predictions.detach() - targets
                batch_examples = targets.size(0)
                total_squared_error += errors.square().sum().item()
                total_absolute_error += errors.abs().sum().item()
                total_examples += batch_examples
            _synchronize_device(device)
            epoch_seconds = time.perf_counter() - epoch_start
            total_training_seconds += epoch_seconds

            final_validation, validation = _evaluate_lengths(
                model, loaders.validation, device
            )
            metrics = {
                "train_mse": total_squared_error / total_examples,
                "train_mae": total_absolute_error / total_examples,
                "validation_mse": validation.mse,
                "validation_mae": validation.mae,
                "training_epoch_seconds": epoch_seconds,
            }
            for length, result in final_validation.items():
                metrics[f"validation_length_{length}_mse"] = result.mse
                metrics[f"validation_length_{length}_mae"] = result.mae
            mlflow.log_metrics(metrics, step=epoch)
            print(
                f"  {setting_name} seed={seed} epoch "
                f"{epoch + 1:02d}/{config.training.epochs}: "
                f"train_mse={metrics['train_mse']:.6f}, "
                f"val_mse={validation.mse:.6f}",
                flush=True,
            )

        final_test, _aggregate_test = _evaluate_lengths(model, loaders.test, device)
        final_step = config.training.epochs - 1
        test_metrics: dict[str, float] = {
            "training_total_seconds": total_training_seconds
        }
        for length, result in final_test.items():
            test_metrics[f"test_length_{length}_mse"] = result.mse
            test_metrics[f"test_length_{length}_mae"] = result.mae
        mlflow.log_metrics(test_metrics, step=final_step)

        model_path = child_dir / "final_model_state.pt"
        torch.save(model.state_dict(), model_path)
        _write_json(
            child_dir / "summary.json",
            {
                "run_id": active_run.info.run_id,
                "seed": seed,
                "model_type": model_type,
                "alpha": alpha,
                "trainable_parameters": parameter_count,
                "training_seconds": total_training_seconds,
                "validation": {
                    str(length): asdict(result)
                    for length, result in final_validation.items()
                },
                "test": {
                    str(length): asdict(result)
                    for length, result in final_test.items()
                },
            },
        )
        mlflow.log_artifacts(str(child_dir), artifact_path="results")
        return AddingRunResult(
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
    results: list[AddingRunResult], config: AddingProblemConfig
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
        selection_mse = statistics.mean(
            result.validation[length].mse
            for result in matching
            for length in config.evaluation.selection_lengths
        )
        all_train_mse = statistics.mean(
            result.validation[length].mse
            for result in matching
            for length in config.data.train_lengths
        )
        candidates.append(((selection_mse, all_train_mse, alpha), alpha))
    return min(candidates)[1]


def _official_results(
    results: list[AddingRunResult], selected_alpha: float
) -> dict[str, list[AddingRunResult]]:
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
    results: list[AddingRunResult],
    config: AddingProblemConfig,
    selected_alpha: float,
) -> dict[str, object]:
    official = _official_results(results, selected_alpha)
    checks: list[dict[str, object]] = []

    def mean_mse(model_name: str, length: int) -> float:
        return statistics.mean(r.test[length].mse for r in official[model_name])

    heterogeneous = {r.seed: r for r in official["heterogeneous_leaky"]}
    for length in config.evaluation.long_lengths:
        for baseline in ("vanilla_rnn", "homogeneous_leaky"):
            baseline_mse = mean_mse(baseline, length)
            heterogeneous_mse = mean_mse("heterogeneous_leaky", length)
            relative_improvement = (
                (baseline_mse - heterogeneous_mse) / baseline_mse
                if baseline_mse > 0
                else 0.0
            )
            checks.append(
                {
                    "kind": "long_length_relative_mse_improvement",
                    "length": length,
                    "baseline": baseline,
                    "observed": relative_improvement,
                    "required": config.evaluation.min_long_relative_improvement,
                    "passed": relative_improvement
                    >= config.evaluation.min_long_relative_improvement,
                }
            )
            baseline_by_seed = {r.seed: r for r in official[baseline]}
            wins = sum(
                heterogeneous[seed].test[length].mse
                < baseline_by_seed[seed].test[length].mse
                for seed in config.experiment.seeds
            )
            checks.append(
                {
                    "kind": "paired_seed_mse_wins",
                    "length": length,
                    "baseline": baseline,
                    "observed": wins,
                    "required": config.evaluation.min_paired_seed_wins,
                    "passed": wins >= config.evaluation.min_paired_seed_wins,
                }
            )

    for length in config.evaluation.short_lengths:
        best_baseline = min(
            mean_mse("vanilla_rnn", length),
            mean_mse("homogeneous_leaky", length),
        )
        penalty = mean_mse("heterogeneous_leaky", length) - best_baseline
        checks.append(
            {
                "kind": "short_length_mse_penalty",
                "length": length,
                "baseline": "best_primary_baseline",
                "observed": penalty,
                "maximum": config.evaluation.max_short_mse_penalty,
                "passed": penalty <= config.evaluation.max_short_mse_penalty,
            }
        )
    return {
        "supported": all(bool(check["passed"]) for check in checks),
        "selected_homogeneous_alpha": selected_alpha,
        "checks": checks,
        "note": "GRU is a secondary reference and does not affect this verdict.",
    }


def _write_per_seed_csv(
    path: Path, results: list[AddingRunResult], config: AddingProblemConfig
) -> None:
    fields = (
        "run_id",
        "seed",
        "model_type",
        "alpha",
        "sequence_length",
        "seen_in_training",
        "validation_mse",
        "validation_mae",
        "test_mse",
        "test_mae",
        "trainable_parameters",
        "training_seconds",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for length in sorted(result.test):
                writer.writerow(
                    {
                        "run_id": result.run_id,
                        "seed": result.seed,
                        "model_type": result.model_type,
                        "alpha": "" if result.alpha is None else result.alpha,
                        "sequence_length": length,
                        "seen_in_training": length in config.data.train_lengths,
                        "validation_mse": result.validation[length].mse
                        if length in result.validation
                        else "",
                        "validation_mae": result.validation[length].mae
                        if length in result.validation
                        else "",
                        "test_mse": result.test[length].mse,
                        "test_mae": result.test[length].mae,
                        "trainable_parameters": result.trainable_parameters,
                        "training_seconds": result.training_seconds,
                    }
                )


def _aggregate_rows(
    results: list[AddingRunResult],
    config: AddingProblemConfig,
    selected_alpha: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name, model_results in _official_results(results, selected_alpha).items():
        if not model_results:
            continue
        for length in config.data.evaluation_lengths:
            mse_values = [result.test[length].mse for result in model_results]
            mae_values = [result.test[length].mae for result in model_results]
            rows.append(
                {
                    "model_type": model_name,
                    "sequence_length": length,
                    "seen_in_training": length in config.data.train_lengths,
                    "mean_test_mse": statistics.mean(mse_values),
                    "sample_std_test_mse": statistics.stdev(mse_values)
                    if len(mse_values) > 1
                    else 0.0,
                    "mean_test_mae": statistics.mean(mae_values),
                    "sample_std_test_mae": statistics.stdev(mae_values)
                    if len(mae_values) > 1
                    else 0.0,
                    "seeds": len(mse_values),
                }
            )
    return rows


def _write_aggregate_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_mse_curves(path: Path, rows: list[dict[str, object]]) -> None:
    labels = {
        "vanilla_rnn": "Vanilla RNN",
        "homogeneous_leaky": "Best homogeneous leak",
        "heterogeneous_leaky": "Heterogeneous leak",
        "gru": "GRU (secondary)",
    }
    fig, axis = plt.subplots(figsize=(8, 5))
    for model_name, label in labels.items():
        model_rows = [row for row in rows if row["model_type"] == model_name]
        if not model_rows:
            continue
        axis.errorbar(
            [int(row["sequence_length"]) for row in model_rows],
            [float(row["mean_test_mse"]) for row in model_rows],
            yerr=[float(row["sample_std_test_mse"]) for row in model_rows],
            marker="o",
            capsize=3,
            label=label,
        )
    axis.axhline(1.0 / 6.0, color="black", linestyle="--", linewidth=1, label="Predict-1 baseline")
    axis.axvline(80, color="gray", linestyle=":", linewidth=1, label="Longest training length")
    axis.set_yscale("log")
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Test mean squared error (log scale)")
    axis.set_title("Experiment 003: Adding Problem error by sequence length")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_per_seed_curves(
    path: Path, results: list[AddingRunResult], selected_alpha: float
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
        for result in sorted(official[model_name], key=lambda item: item.seed):
            lengths = sorted(result.test)
            axis.plot(
                lengths,
                [result.test[length].mse for length in lengths],
                marker="o",
                label=f"seed {result.seed}",
            )
        axis.axhline(1.0 / 6.0, color="black", linestyle="--", linewidth=1)
        axis.axvline(80, color="gray", linestyle=":", linewidth=1)
        axis.set_yscale("log")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        if official[model_name]:
            axis.legend()
    fig.supxlabel("Sequence length")
    fig.supylabel("Test MSE (log scale)")
    fig.suptitle("Experiment 003: individual-seed Adding Problem curves")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _log_parent_curves(
    results: list[AddingRunResult], selected_alpha: float
) -> None:
    official = _official_results(results, selected_alpha)
    lengths = sorted(next(iter(official.values()))[0].test)
    for length in lengths:
        metrics: dict[str, float] = {}
        for model_name, model_results in official.items():
            if not model_results:
                continue
            metrics[f"test_mean_{model_name}_mse"] = statistics.mean(
                result.test[length].mse for result in model_results
            )
            for result in model_results:
                metrics[f"test_seed_{result.seed}_{model_name}_mse"] = result.test[
                    length
                ].mse
        mlflow.log_metrics(metrics, step=length)

    client = mlflow.MlflowClient()
    by_epoch: dict[int, dict[str, float]] = {}
    for model_name, model_results in official.items():
        if not model_results:
            continue
        histories = {
            result.seed: {
                point.step: point.value
                for point in client.get_metric_history(result.run_id, "validation_mse")
            }
            for result in model_results
        }
        for epoch in sorted(set.intersection(*(set(history) for history in histories.values()))):
            metrics = by_epoch.setdefault(epoch, {})
            metrics[f"epoch_mean_{model_name}_validation_mse"] = statistics.mean(
                history[epoch] for history in histories.values()
            )
            for seed, history in histories.items():
                metrics[f"epoch_seed_{seed}_{model_name}_validation_mse"] = history[
                    epoch
                ]
    for epoch, metrics in sorted(by_epoch.items()):
        mlflow.log_metrics(metrics, step=epoch)


def _inspection_result(
    results: list[AddingRunResult],
    *,
    model_type: str,
    seed: int,
    alpha: float | None,
) -> AddingRunResult:
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
                and math.isclose(alpha, result.alpha)
            )
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one inspection run for {model_type}, seed={seed}")
    return matches[0]


def _write_hidden_artifacts(
    output_dir: Path,
    results: list[AddingRunResult],
    config: AddingProblemConfig,
    selected_alpha: float,
    device: torch.device,
) -> None:
    seed = config.evaluation.inspection_seed
    length = config.evaluation.inspection_length
    loaders = build_adding_problem_loaders(
        config.data, seed=seed, use_pin_memory=device.type == "cuda"
    )
    inputs, targets = loaders.test_datasets[length].tensors
    selected_inputs = inputs[: config.evaluation.inspection_examples]
    selected_targets = targets[: config.evaluation.inspection_examples]
    torch.save(
        {"inputs": selected_inputs, "targets": selected_targets, "length": length},
        output_dir / "inspection_examples.pt",
    )
    settings = (
        ("vanilla_rnn", None),
        ("homogeneous_leaky", selected_alpha),
        ("heterogeneous_leaky", None),
        ("gru", None),
    )
    for model_type, alpha in settings:
        if model_type == "gru" and not config.model.include_gru:
            continue
        result = _inspection_result(
            results, model_type=model_type, seed=seed, alpha=alpha
        )
        seed_everything(seed, config.experiment.deterministic)
        model = build_recurrent_model(
            model_type, config.model, alpha=alpha, readout="final"
        ).to(device)
        model.load_state_dict(
            torch.load(result.model_state_path, map_location=device, weights_only=True)
        )
        model.eval()
        with torch.no_grad():
            states = model.hidden_states(selected_inputs.to(device)).cpu()
            predictions = model(selected_inputs.to(device)).cpu()
        label = model_type
        torch.save(
            {
                "inputs": selected_inputs,
                "targets": selected_targets,
                "predictions": predictions,
                "hidden_states": states,
                "model_type": model_type,
                "alpha": alpha,
                "length": length,
                "seed": seed,
            },
            output_dir / f"hidden_states_{label}.pt",
        )
        _plot_hidden_heatmap(
            output_dir / f"hidden_states_{label}.png",
            states,
            selected_inputs,
            selected_targets,
            predictions,
            title=label.replace("_", " ").title(),
            group_sizes=config.model.heterogeneous_group_sizes
            if model_type == "heterogeneous_leaky"
            else None,
        )
        if model_type == "heterogeneous_leaky":
            _plot_group_traces(
                output_dir / "heterogeneous_group_traces.png",
                states,
                selected_inputs,
                config,
            )


def _plot_hidden_heatmap(
    path: Path,
    states: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    *,
    title: str,
    group_sizes: tuple[int, ...] | None,
) -> None:
    fig, axes = plt.subplots(
        1,
        states.size(0),
        figsize=(6 * states.size(0), 5),
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
        marker_positions = torch.nonzero(
            inputs[index, :, 1], as_tuple=False
        ).flatten()
        for marker_number, position in enumerate(marker_positions.tolist()):
            axis.axvline(
                position,
                color="lime" if marker_number == 0 else "yellow",
                linewidth=1.5,
                label=f"marked value {marker_number + 1}",
            )
        if group_sizes is not None:
            boundary = 0
            for size in group_sizes[:-1]:
                boundary += size
                axis.axhline(boundary - 0.5, color="black", linewidth=1)
        axis.set_title(
            f"target={targets[index].item():.3f}, "
            f"prediction={predictions[index].item():.3f}"
        )
        axis.set_xlabel("Timestep")
        axis.set_ylabel("Hidden neuron")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(f"{title}: length-{states.size(1)} hidden activity")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="Activation")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_group_traces(
    path: Path,
    states: torch.Tensor,
    inputs: torch.Tensor,
    config: AddingProblemConfig,
) -> None:
    fig, axes = plt.subplots(
        states.size(0), 1, figsize=(10, 3 * states.size(0)), squeeze=False
    )
    start = 0
    groups: list[tuple[float, slice]] = []
    for alpha, size in zip(
        config.model.heterogeneous_alphas,
        config.model.heterogeneous_group_sizes,
        strict=True,
    ):
        groups.append((alpha, slice(start, start + size)))
        start += size
    for example, axis in enumerate(axes[:, 0]):
        for alpha, group in groups:
            axis.plot(
                states[example, :, group].abs().mean(dim=1).numpy(),
                label=f"alpha={alpha:g}",
            )
        for position in torch.nonzero(inputs[example, :, 1], as_tuple=False).flatten():
            axis.axvline(int(position), color="black", linestyle=":", linewidth=1)
        axis.set_xlabel("Timestep")
        axis.set_ylabel("Mean |hidden activation|")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Heterogeneous activity; dotted lines are marked values")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _build_report(
    rows: list[dict[str, object]], verdict: dict[str, object]
) -> str:
    lines = [
        "# Experiment 003 — Adding Problem",
        "",
        "## Hypothesis verdict",
        "",
        (
            "The preregistered exploratory support criteria were satisfied."
            if verdict["supported"]
            else "The preregistered exploratory support criteria were not all satisfied."
        ),
        "",
        f"Selected homogeneous alpha: `{verdict['selected_homogeneous_alpha']}`",
        "",
        "GRU is a secondary reference and does not affect the verdict.",
        "",
        "## Mean test error",
        "",
        "| Model | Length | Trained length? | Mean MSE | Sample SD | Mean MAE |",
        "|---|---:|:---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {str(row['model_type']).replace('_', ' ')} | "
            f"{row['sequence_length']} | "
            f"{'yes' if row['seen_in_training'] else 'no'} | "
            f"{float(row['mean_test_mse']):.6f} | "
            f"{float(row['sample_std_test_mse']):.6f} | "
            f"{float(row['mean_test_mae']):.6f} |"
        )
    lines.extend(["", "## Preregistered checks", ""])
    for check in verdict["checks"]:
        assert isinstance(check, dict)
        lines.append(
            f"- **{'PASS' if check['passed'] else 'FAIL'}:** {check['kind']} "
            f"at length {check['length']} against {check['baseline']} — "
            f"observed `{check['observed']}`."
        )
    return "\n".join(lines)


def run_adding_problem_experiment(
    config: AddingProblemConfig,
    config_path: str | Path,
    device: torch.device,
) -> Path:
    mlflow.set_tracking_uri(normalize_tracking_uri(config.experiment.tracking_uri))
    experiment_id = get_or_create_experiment(
        name=config.experiment.name,
        artifact_dir=config.experiment.artifact_dir,
        description=(
            "Tests heterogeneous recurrent timescales on selective continuous-value "
            "memory and unseen-length extrapolation in the classic Adding Problem."
        ),
    )
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=config.experiment.run_name
    ) as parent_run:
        parent_dir = Path(config.experiment.output_dir) / parent_run.info.run_id
        aggregate_dir = parent_dir / "aggregate"
        aggregate_dir.mkdir(parents=True, exist_ok=False)
        mlflow.log_params(flatten_mapping(config.to_dict()))
        mlflow.set_tags(
            {
                "dataset": "synthetic_adding_problem",
                "protocol": "mixed_length_with_unseen_extrapolation",
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
        print(
            f"Running {len(settings) * len(config.experiment.seeds)} paired "
            f"Adding Problem child runs on {device}...",
            flush=True,
        )
        results: list[AddingRunResult] = []
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
        _write_per_seed_csv(
            aggregate_dir / "per_seed_results.csv", results, config
        )
        _write_aggregate_csv(aggregate_dir / "aggregate_results.csv", rows)
        _write_json(aggregate_dir / "aggregate_results.json", rows)
        _write_json(aggregate_dir / "hypothesis_verdict.json", verdict)
        _plot_mse_curves(aggregate_dir / "mse_vs_sequence_length.png", rows)
        _plot_per_seed_curves(
            aggregate_dir / "per_seed_mse_vs_sequence_length.png",
            results,
            selected_alpha,
        )
        _log_parent_curves(results, selected_alpha)
        _write_hidden_artifacts(
            aggregate_dir, results, config, selected_alpha, device
        )
        report = _build_report(rows, verdict)
        (aggregate_dir / "run_summary.md").write_text(report, encoding="utf-8")
        _write_json(aggregate_dir / "resolved_config.json", config.to_dict())

        mlflow.log_param("selected_homogeneous_alpha", selected_alpha)
        mlflow.log_metric("hypothesis_supported", float(bool(verdict["supported"])))
        mlflow.set_tag(
            "hypothesis.verdict",
            "supported" if verdict["supported"] else "not_supported",
        )
        mlflow.set_tag("mlflow.note.content", report)
        mlflow.log_artifacts(str(aggregate_dir), artifact_path="aggregate")
        print(f"\nExperiment 003 suite complete: {parent_run.info.run_id}")
        print(f"Artifacts: {parent_dir.resolve()}")
        return parent_dir
