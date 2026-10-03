from __future__ import annotations

import csv
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import torch
from torch import nn
from torch.optim import AdamW

from continual_learning_lab.config import Stream51Config
from continual_learning_lab.experiment_utils import synchronize_device, write_json
from continual_learning_lab.model import count_trainable_parameters
from continual_learning_lab.reproducibility import seed_everything
from continual_learning_lab.stream51_data import (
    PreparedStream51Data,
    Stream51Record,
    prepare_stream51_data,
    validate_matched_orderings,
)
from continual_learning_lab.stream51_models import (
    Stream51RecurrentClassifier,
    build_stream51_model,
)
from continual_learning_lab.tracking import (
    flatten_mapping,
    get_or_create_experiment,
    normalize_tracking_uri,
)


@dataclass(frozen=True)
class Stream51RunResult:
    run_id: str
    seed: int
    ordering: str
    persistent: bool
    ticks: int
    trainable_parameters: int
    optimizer_steps: int
    training_seconds: float
    prequential_macro_accuracy: float
    prequential_micro_accuracy: float
    prequential_cross_entropy: float
    heldout_macro_accuracy: float
    heldout_micro_accuracy: float
    heldout_cross_entropy: float
    windows: tuple[dict[str, float | int], ...]
    position_metrics: dict[str, dict[str, float | int]]
    model_state_path: Path


def model_settings(config: Stream51Config) -> list[tuple[bool, int]]:
    return [
        (persistent, ticks)
        for persistent in (False, True)
        for ticks in config.model.tick_counts
    ]


def setting_name(persistent: bool, ticks: int) -> str:
    state = "persistent" if persistent else "stateless"
    return f"{state}_{ticks}_tick" if ticks == 1 else f"{state}_{ticks}_ticks"


def _macro_accuracy(correct: torch.Tensor, counts: torch.Tensor) -> float:
    present = counts.gt(0)
    if not present.any():
        raise ValueError("Cannot compute macro accuracy without examples")
    return float((correct[present] / counts[present]).mean().item())


def _position_bin(position: int) -> str:
    if position == 0:
        return "start"
    if position < 5:
        return "positions_1_4"
    if position < 20:
        return "positions_5_19"
    return "positions_20_plus"


@torch.no_grad()
def evaluate_heldout(
    model: Stream51RecurrentClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    ticks: int,
    batch_size: int,
    num_classes: int,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    class_correct = torch.zeros(num_classes, dtype=torch.float64)
    class_counts = torch.zeros(num_classes, dtype=torch.float64)
    total_correct = 0
    total_loss = 0.0
    for start in range(0, labels.numel(), batch_size):
        stop = min(start + batch_size, labels.numel())
        batch_features = features[start:stop].to(device, non_blocking=True)
        batch_labels = labels[start:stop].to(device, non_blocking=True)
        # A fresh zero state for the whole batch is equivalent to resetting each
        # independent static test image.
        logits, _hidden = model.step(batch_features, None, ticks=ticks)
        total_loss += nn.functional.cross_entropy(
            logits, batch_labels, reduction="sum"
        ).item()
        predictions = logits.argmax(dim=1)
        correct = predictions.eq(batch_labels)
        total_correct += int(correct.sum().item())
        for class_id in batch_labels.unique().tolist():
            class_mask = batch_labels.eq(class_id)
            class_counts[class_id] += int(class_mask.sum().item())
            class_correct[class_id] += int((correct & class_mask).sum().item())
    examples = int(labels.numel())
    if examples == 0:
        raise ValueError("Cannot evaluate an empty Stream-51 held-out set")
    return (
        _macro_accuracy(class_correct, class_counts),
        total_correct / examples,
        total_loss / examples,
    )


def train_prequential(
    model: Stream51RecurrentClassifier,
    records: tuple[Stream51Record, ...],
    data: PreparedStream51Data,
    config: Stream51Config,
    *,
    persistent: bool,
    ticks: int,
    device: torch.device,
    log_windows: bool = True,
) -> tuple[
    float,
    float,
    float,
    tuple[dict[str, float | int], ...],
    dict[str, dict[str, float | int]],
    int,
]:
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    model.train()
    num_classes = config.model.num_classes
    class_correct = torch.zeros(num_classes, dtype=torch.float64)
    class_counts = torch.zeros(num_classes, dtype=torch.float64)
    window_correct = torch.zeros(num_classes, dtype=torch.float64)
    window_counts = torch.zeros(num_classes, dtype=torch.float64)
    position_totals: dict[str, list[int]] = {}
    total_correct = 0
    total_loss = 0.0
    window_total_correct = 0
    window_total_loss = 0.0
    window_examples = 0
    windows: list[dict[str, float | int]] = []
    hidden: torch.Tensor | None = None

    for observation_index, record in enumerate(records):
        feature = data.train_features[record.observation_id].unsqueeze(0).to(
            device, non_blocking=True
        )
        target = data.train_labels[record.observation_id].view(1).to(
            device, non_blocking=True
        )
        if int(target.item()) != record.class_id:
            raise ValueError("Feature-cache label does not match Stream-51 metadata")
        if not persistent or record.reset_before:
            hidden = None

        optimizer.zero_grad(set_to_none=True)
        logits, next_hidden = model.step(feature, hidden, ticks=ticks)
        loss = nn.functional.cross_entropy(logits, target)

        # All metrics above this update are prequential: the prediction was
        # produced before the current label changed any parameter.
        prediction = int(logits.detach().argmax(dim=1).item())
        target_id = int(target.item())
        correct = int(prediction == target_id)
        loss_value = float(loss.detach().item())
        class_counts[target_id] += 1
        class_correct[target_id] += correct
        window_counts[target_id] += 1
        window_correct[target_id] += correct
        total_correct += correct
        total_loss += loss_value
        window_total_correct += correct
        window_total_loss += loss_value
        window_examples += 1
        bin_name = _position_bin(record.position_in_segment)
        bin_values = position_totals.setdefault(bin_name, [0, 0])
        bin_values[0] += correct
        bin_values[1] += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.gradient_clip_norm
        )
        optimizer.step()
        hidden = next_hidden.detach() if persistent else None

        at_window_end = window_examples == config.evaluation.window_size
        at_stream_end = observation_index == len(records) - 1
        if at_window_end or at_stream_end:
            window = {
                "observation": observation_index + 1,
                "examples": window_examples,
                "macro_accuracy": _macro_accuracy(window_correct, window_counts),
                "micro_accuracy": window_total_correct / window_examples,
                "cross_entropy": window_total_loss / window_examples,
            }
            windows.append(window)
            if log_windows:
                mlflow.log_metrics(
                    {
                        "prequential_window_macro_accuracy": float(
                            window["macro_accuracy"]
                        ),
                        "prequential_window_micro_accuracy": float(
                            window["micro_accuracy"]
                        ),
                        "prequential_window_cross_entropy": float(
                            window["cross_entropy"]
                        ),
                    },
                    step=observation_index + 1,
                )
            window_correct.zero_()
            window_counts.zero_()
            window_total_correct = 0
            window_total_loss = 0.0
            window_examples = 0

    examples = len(records)
    position_metrics = {
        name: {
            "accuracy": correct / count,
            "correct": correct,
            "examples": count,
        }
        for name, (correct, count) in position_totals.items()
    }
    return (
        _macro_accuracy(class_correct, class_counts),
        total_correct / examples,
        total_loss / examples,
        tuple(windows),
        position_metrics,
        examples,
    )


def _write_windows(path: Path, windows: tuple[dict[str, float | int], ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "observation",
                "examples",
                "macro_accuracy",
                "micro_accuracy",
                "cross_entropy",
            ),
        )
        writer.writeheader()
        writer.writerows(windows)


def _run_child(
    config: Stream51Config,
    data: PreparedStream51Data,
    *,
    seed: int,
    ordering: str,
    persistent: bool,
    ticks: int,
    device: torch.device,
    parent_dir: Path,
) -> Stream51RunResult:
    seed_everything(seed, config.experiment.deterministic)
    model = build_stream51_model(config.model).to(device)
    parameter_count = count_trainable_parameters(model)
    name = setting_name(persistent, ticks)
    parent_run = mlflow.active_run()
    if parent_run is None:
        raise RuntimeError("Stream-51 child run requires an active parent run")

    with mlflow.start_run(
        experiment_id=parent_run.info.experiment_id,
        run_name=f"{config.experiment.run_name}-{ordering}-{name}-seed-{seed}",
        nested=True,
    ) as active_run:
        child_dir = parent_dir / "children" / active_run.info.run_id
        child_dir.mkdir(parents=True, exist_ok=False)
        params = flatten_mapping(config.to_dict())
        params.update(
            {
                "seed": seed,
                "ordering": ordering,
                "persistent": persistent,
                "ticks": ticks,
                "model.trainable_parameters": parameter_count,
                "data.reset_schedule_hash": data.reset_schedule_hash,
                "training.total_example_exposure": len(data.orderings[ordering]),
            }
        )
        mlflow.log_params(params)
        mlflow.set_tags(
            {
                "dataset": "Stream-51_frozen_resnet18_features",
                "protocol": "single_pass_prequential",
                "run.role": "state_order_tick_seed_comparison",
                "device.resolved": str(device),
            }
        )

        synchronize_device(device)
        started = time.perf_counter()
        (
            prequential_macro,
            prequential_micro,
            prequential_loss,
            windows,
            position_metrics,
            optimizer_steps,
        ) = train_prequential(
            model,
            data.orderings[ordering],
            data,
            config,
            persistent=persistent,
            ticks=ticks,
            device=device,
        )
        synchronize_device(device)
        training_seconds = time.perf_counter() - started
        heldout_macro, heldout_micro, heldout_loss = evaluate_heldout(
            model,
            data.test_features,
            data.test_labels,
            ticks=ticks,
            batch_size=config.evaluation.eval_batch_size,
            num_classes=config.model.num_classes,
            device=device,
        )
        mlflow.log_metrics(
            {
                "prequential_final_macro_accuracy": prequential_macro,
                "prequential_final_micro_accuracy": prequential_micro,
                "prequential_final_cross_entropy": prequential_loss,
                "heldout_macro_accuracy": heldout_macro,
                "heldout_micro_accuracy": heldout_micro,
                "heldout_cross_entropy": heldout_loss,
                "training_total_seconds": training_seconds,
            },
            step=optimizer_steps,
        )

        model_path = child_dir / "final_model_state.pt"
        torch.save(model.state_dict(), model_path)
        _write_windows(child_dir / "prequential_windows.csv", windows)
        write_json(child_dir / "position_metrics.json", position_metrics)
        summary = {
            "run_id": active_run.info.run_id,
            "seed": seed,
            "ordering": ordering,
            "persistent": persistent,
            "ticks": ticks,
            "trainable_parameters": parameter_count,
            "optimizer_steps": optimizer_steps,
            "training_seconds": training_seconds,
            "prequential_macro_accuracy": prequential_macro,
            "prequential_micro_accuracy": prequential_micro,
            "prequential_cross_entropy": prequential_loss,
            "heldout_macro_accuracy": heldout_macro,
            "heldout_micro_accuracy": heldout_micro,
            "heldout_cross_entropy": heldout_loss,
            "reset_schedule_hash": data.reset_schedule_hash,
        }
        write_json(child_dir / "summary.json", summary)
        mlflow.log_artifacts(str(child_dir), artifact_path="results")
        print(
            f"  {ordering} {name} seed={seed}: "
            f"prequential_macro={prequential_macro:.3f}, "
            f"heldout_macro={heldout_macro:.3f}",
            flush=True,
        )
        return Stream51RunResult(
            run_id=active_run.info.run_id,
            seed=seed,
            ordering=ordering,
            persistent=persistent,
            ticks=ticks,
            trainable_parameters=parameter_count,
            optimizer_steps=optimizer_steps,
            training_seconds=training_seconds,
            prequential_macro_accuracy=prequential_macro,
            prequential_micro_accuracy=prequential_micro,
            prequential_cross_entropy=prequential_loss,
            heldout_macro_accuracy=heldout_macro,
            heldout_micro_accuracy=heldout_micro,
            heldout_cross_entropy=heldout_loss,
            windows=windows,
            position_metrics=position_metrics,
            model_state_path=model_path,
        )


def _result_index(
    results: list[Stream51RunResult],
) -> dict[tuple[int, str, bool, int], Stream51RunResult]:
    return {
        (result.seed, result.ordering, result.persistent, result.ticks): result
        for result in results
    }


def evaluate_hypothesis(
    results: list[Stream51RunResult], config: Stream51Config
) -> dict[str, object]:
    indexed = _result_index(results)
    seeds = config.experiment.seeds

    def gains(ordering: str, ticks: int) -> list[float]:
        return [
            indexed[(seed, ordering, True, ticks)].prequential_macro_accuracy
            - indexed[(seed, ordering, False, ticks)].prequential_macro_accuracy
            for seed in seeds
        ]

    natural = gains("natural", 1)
    global_values = gains("global_shuffle", 1)
    interactions = [
        natural_gain - global_gain
        for natural_gain, global_gain in zip(natural, global_values, strict=True)
    ]
    heldout_penalties = [
        indexed[(seed, "natural", False, 1)].heldout_macro_accuracy
        - indexed[(seed, "natural", True, 1)].heldout_macro_accuracy
        for seed in seeds
    ]
    primary_checks = {
        "natural_mean_gain": statistics.mean(natural)
        >= config.evaluation.min_natural_persistence_gain,
        "natural_paired_wins": sum(value > 0 for value in natural)
        >= config.evaluation.min_paired_seed_wins,
        "natural_over_global_mean_gain": statistics.mean(interactions)
        >= config.evaluation.min_natural_over_global_gain,
        "natural_over_global_paired_wins": sum(value > 0 for value in interactions)
        >= config.evaluation.min_paired_seed_wins,
        "heldout_safeguard": statistics.mean(heldout_penalties)
        <= config.evaluation.max_heldout_macro_penalty,
    }

    one_tick = {
        persistent: [
            indexed[(seed, "natural", persistent, 1)].prequential_macro_accuracy
            for seed in seeds
        ]
        for persistent in (False, True)
    }
    four_tick = {
        persistent: [
            indexed[(seed, "natural", persistent, 4)].prequential_macro_accuracy
            for seed in seeds
        ]
        for persistent in (False, True)
    }
    tick_interactions = [
        (four_tick[True][index] - one_tick[True][index])
        - (four_tick[False][index] - one_tick[False][index])
        for index in range(len(seeds))
    ]
    secondary_checks = {
        "mean_interaction_gain": statistics.mean(tick_interactions)
        >= config.evaluation.min_multitick_interaction_gain,
        "paired_wins": sum(value > 0 for value in tick_interactions)
        >= config.evaluation.min_paired_seed_wins,
    }
    return {
        "primary_supported": all(primary_checks.values()),
        "secondary_multitick_supported": all(secondary_checks.values()),
        "primary_checks": primary_checks,
        "secondary_checks": secondary_checks,
        "natural_persistence_gains": natural,
        "global_persistence_gains": global_values,
        "natural_over_global_interactions": interactions,
        "heldout_macro_penalties": heldout_penalties,
        "multitick_interactions": tick_interactions,
    }


def _aggregate_rows(results: list[Stream51RunResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ordering in ("natural", "local_shuffle", "global_shuffle"):
        for persistent in (False, True):
            for ticks in (1, 4):
                matching = [
                    result
                    for result in results
                    if result.ordering == ordering
                    and result.persistent == persistent
                    and result.ticks == ticks
                ]
                row: dict[str, object] = {
                    "ordering": ordering,
                    "persistent": persistent,
                    "ticks": ticks,
                    "setting": setting_name(persistent, ticks),
                    "seeds": len(matching),
                }
                for field in (
                    "prequential_macro_accuracy",
                    "prequential_micro_accuracy",
                    "prequential_cross_entropy",
                    "heldout_macro_accuracy",
                    "heldout_micro_accuracy",
                    "heldout_cross_entropy",
                    "training_seconds",
                ):
                    values = [float(getattr(result, field)) for result in matching]
                    row[f"{field}_mean"] = statistics.mean(values)
                    row[f"{field}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
                rows.append(row)
    return rows


def _write_aggregate_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_aggregate(path: Path, rows: list[dict[str, object]]) -> None:
    fig, axis = plt.subplots(figsize=(10, 5.5))
    orderings = ("natural", "local_shuffle", "global_shuffle")
    settings = (
        (False, 1),
        (False, 4),
        (True, 1),
        (True, 4),
    )
    width = 0.18
    x = list(range(len(orderings)))
    for offset_index, (persistent, ticks) in enumerate(settings):
        matching = [
            next(
                row
                for row in rows
                if row["ordering"] == ordering
                and row["persistent"] == persistent
                and row["ticks"] == ticks
            )
            for ordering in orderings
        ]
        positions = [value + (offset_index - 1.5) * width for value in x]
        axis.bar(
            positions,
            [float(row["prequential_macro_accuracy_mean"]) for row in matching],
            width=width,
            yerr=[float(row["prequential_macro_accuracy_sd"]) for row in matching],
            label=setting_name(persistent, ticks),
            capsize=3,
        )
    axis.set_xticks(x, orderings)
    axis.set_ylabel("Prequential macro accuracy")
    axis.set_title("Experiment 004: persistent-state benefit by training order")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _build_report(
    rows: list[dict[str, object]], verdict: dict[str, object]
) -> str:
    lines = [
        "# Experiment 004 — Persistent state on Stream-51",
        "",
        "## Preregistered verdict",
        "",
        f"Primary persistence hypothesis supported: `{verdict['primary_supported']}`.",
        f"Secondary multi-tick hypothesis supported: "
        f"`{verdict['secondary_multitick_supported']}`.",
        "",
        "## Aggregate results",
        "",
        "| Ordering | Setting | Prequential macro | Held-out macro |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['ordering']} | {row['setting']} | "
            f"{float(row['prequential_macro_accuracy_mean']):.2%} ± "
            f"{float(row['prequential_macro_accuracy_sd']):.2%} | "
            f"{float(row['heldout_macro_accuracy_mean']):.2%} ± "
            f"{float(row['heldout_macro_accuracy_sd']):.2%} |"
        )
    lines.extend(
        [
            "",
            "Prequential metrics score every observation before its label updates "
            "the model. Held-out images reset state independently.",
        ]
    )
    return "\n".join(lines)


def run_stream51_experiment(
    config: Stream51Config,
    config_path: str | Path,
    device: torch.device,
    *,
    prepared_data: PreparedStream51Data | None = None,
) -> Path:
    data = prepared_data or prepare_stream51_data(config, device)
    validate_matched_orderings(data.orderings, data.segment_lengths)
    mlflow.set_tracking_uri(normalize_tracking_uri(config.experiment.tracking_uri))
    experiment_id = get_or_create_experiment(
        name=config.experiment.name,
        artifact_dir=config.experiment.artifact_dir,
        description=(
            "Tests whether persistent recurrent activation state exploits temporal "
            "coherence in a single-pass Stream-51 observation stream."
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
                "dataset": "Stream-51_frozen_resnet18_features",
                "protocol": "matched_reset_prequential_stream",
                "run.role": "architecture_order_benchmark_parent",
                "hypothesis.owner": "learner",
                "device.requested": config.experiment.device,
                "device.resolved": str(device),
                "python.version": platform.python_version(),
                "torch.version": torch.__version__,
            }
        )
        mlflow.log_artifact(str(Path(config_path).resolve()), artifact_path="configuration")

        diagnostics = {
            name: asdict(values) for name, values in data.diagnostics.items()
        }
        write_json(aggregate_dir / "ordering_diagnostics.json", diagnostics)
        write_json(
            aggregate_dir / "reset_schedule.json",
            {
                "segment_lengths": list(data.segment_lengths),
                "reset_schedule_hash": data.reset_schedule_hash,
            },
        )
        write_json(aggregate_dir / "feature_cache_manifest.json", data.cache_manifest)

        settings = model_settings(config)
        total_runs = len(settings) * len(config.data.orderings) * len(config.experiment.seeds)
        print(f"Running {total_runs} paired Stream-51 child runs on {device}...", flush=True)
        results: list[Stream51RunResult] = []
        for seed in config.experiment.seeds:
            for ordering in config.data.orderings:
                for persistent, ticks in settings:
                    results.append(
                        _run_child(
                            config,
                            data,
                            seed=seed,
                            ordering=ordering,
                            persistent=persistent,
                            ticks=ticks,
                            device=device,
                            parent_dir=parent_dir,
                        )
                    )

        rows = _aggregate_rows(results)
        verdict = evaluate_hypothesis(results, config)
        _write_aggregate_csv(aggregate_dir / "aggregate_results.csv", rows)
        write_json(aggregate_dir / "aggregate_results.json", rows)
        write_json(aggregate_dir / "hypothesis_verdict.json", verdict)
        _plot_aggregate(aggregate_dir / "prequential_macro_by_order.png", rows)
        report = _build_report(rows, verdict)
        (aggregate_dir / "run_summary.md").write_text(report, encoding="utf-8")
        write_json(aggregate_dir / "resolved_config.json", config.to_dict())

        mlflow.log_metric(
            "primary_hypothesis_supported", float(bool(verdict["primary_supported"]))
        )
        mlflow.log_metric(
            "secondary_multitick_supported",
            float(bool(verdict["secondary_multitick_supported"])),
        )
        mlflow.set_tag(
            "hypothesis.verdict",
            "supported" if verdict["primary_supported"] else "not_supported",
        )
        mlflow.set_tag("mlflow.note.content", report)
        mlflow.log_artifacts(str(aggregate_dir), artifact_path="aggregate")
        print(f"\nExperiment 004 suite complete: {parent_run.info.run_id}")
        print(f"Artifacts: {parent_dir.resolve()}")
        return parent_dir
