from dataclasses import replace
from pathlib import Path

import torch
import mlflow

from continual_learning_lab.config import DelayedRecallConfig, load_config
from continual_learning_lab.delayed_recall_experiment import (
    RecallMetrics,
    RecallRunResult,
    evaluate_hypothesis,
    model_settings,
    run_delayed_recall_experiment,
    select_homogeneous_alpha,
)


def _config() -> DelayedRecallConfig:
    config = load_config("configs/heterogeneous_leaky_delayed_recall.yaml")
    assert isinstance(config, DelayedRecallConfig)
    return config


def _metrics(accuracy: float) -> dict[int, RecallMetrics]:
    return {
        delay: RecallMetrics(loss=1.0 - accuracy, accuracy=accuracy, examples=20)
        for delay in (5, 10, 20, 40, 80)
    }


def _result(
    model_type: str,
    seed: int,
    *,
    validation_accuracy: float,
    test_accuracies: dict[int, float],
    alpha: float | None = None,
) -> RecallRunResult:
    return RecallRunResult(
        run_id=f"{model_type}-{alpha}-{seed}",
        seed=seed,
        model_type=model_type,
        alpha=alpha,
        trainable_parameters=10,
        training_seconds=1.0,
        validation=_metrics(validation_accuracy),
        test={
            delay: RecallMetrics(loss=0.5, accuracy=accuracy, examples=20)
            for delay, accuracy in test_accuracies.items()
        },
        model_state_path=Path("unused.pt"),
    )


def _supporting_results() -> list[RecallRunResult]:
    results: list[RecallRunResult] = []
    for seed in (42, 43, 44):
        results.extend(
            (
                _result(
                    "vanilla_rnn",
                    seed,
                    validation_accuracy=0.7,
                    test_accuracies={5: 0.95, 10: 0.94, 20: 0.8, 40: 0.60, 80: 0.55},
                ),
                _result(
                    "homogeneous_leaky",
                    seed,
                    alpha=0.5,
                    validation_accuracy=0.6,
                    test_accuracies={5: 0.94, 10: 0.93, 20: 0.8, 40: 0.58, 80: 0.54},
                ),
                _result(
                    "homogeneous_leaky",
                    seed,
                    alpha=0.1,
                    validation_accuracy=0.85,
                    test_accuracies={5: 0.96, 10: 0.95, 20: 0.85, 40: 0.64, 80: 0.59},
                ),
                _result(
                    "homogeneous_leaky",
                    seed,
                    alpha=0.01,
                    validation_accuracy=0.75,
                    test_accuracies={5: 0.8, 10: 0.82, 20: 0.84, 40: 0.65, 80: 0.60},
                ),
                _result(
                    "heterogeneous_leaky",
                    seed,
                    validation_accuracy=0.9,
                    test_accuracies={5: 0.94, 10: 0.93, 20: 0.9, 40: 0.72, 80: 0.68},
                ),
                # Deliberately much stronger: GRU must not affect the verdict.
                _result(
                    "gru",
                    seed,
                    validation_accuracy=1.0,
                    test_accuracies={5: 1.0, 10: 1.0, 20: 1.0, 40: 1.0, 80: 1.0},
                ),
            )
        )
    return results


def test_official_config_expands_to_eighteen_paired_child_runs() -> None:
    config = _config()

    assert len(model_settings(config)) == 6
    assert len(model_settings(config)) * len(config.experiment.seeds) == 18


def test_alpha_selection_uses_validation_and_gru_does_not_gate_support() -> None:
    config = _config()
    results = _supporting_results()

    selected_alpha = select_homogeneous_alpha(results, config)
    verdict = evaluate_hypothesis(results, config, selected_alpha)

    assert selected_alpha == 0.1
    assert verdict["supported"] is True
    assert all(check["passed"] for check in verdict["checks"])
    assert "secondary reference" in verdict["note"]


def test_hypothesis_is_not_supported_when_long_delay_effect_is_too_small() -> None:
    config = _config()
    results = _supporting_results()
    weakened: list[RecallRunResult] = []
    for result in results:
        if result.model_type != "heterogeneous_leaky":
            weakened.append(result)
            continue
        test = dict(result.test)
        test[40] = RecallMetrics(loss=0.5, accuracy=0.66, examples=20)
        test[80] = RecallMetrics(loss=0.5, accuracy=0.61, examples=20)
        weakened.append(replace(result, test=test))

    verdict = evaluate_hypothesis(weakened, config, selected_alpha=0.1)

    assert verdict["supported"] is False
    assert any(not check["passed"] for check in verdict["checks"])


def test_reduced_cpu_suite_writes_comparison_and_hidden_state_artifacts(
    tmp_path: Path,
) -> None:
    config = _config()
    config = replace(
        config,
        experiment=replace(
            config.experiment,
            name=f"exp002-test-{tmp_path.name}",
            run_name="reduced-test",
            seeds=(42,),
            output_dir=str(tmp_path / "outputs"),
            tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
            artifact_dir=str(tmp_path / "artifacts"),
        ),
        data=replace(
            config.data,
            delays=(1, 2),
            timing_protocol="fixed_initial_value",
            distractor_mode="same_content_channel",
            train_examples_per_delay=4,
            validation_examples_per_delay=2,
            test_examples_per_delay=2,
            batch_size=4,
        ),
        model=replace(
            config.model,
            hidden_size=6,
            homogeneous_alphas=(0.1,),
            heterogeneous_group_sizes=(2, 2, 2),
            include_gru=False,
        ),
        training=replace(config.training, epochs=2),
        evaluation=replace(
            config.evaluation,
            selection_delays=(2,),
            long_delays=(2,),
            short_delays=(1,),
            min_paired_seed_wins=1,
            inspection_delay=2,
        ),
    )

    output_dir = run_delayed_recall_experiment(
        config,
        "configs/heterogeneous_leaky_delayed_recall.yaml",
        torch.device("cpu"),
    )

    aggregate = output_dir / "aggregate"
    assert len(list((output_dir / "children").iterdir())) == 3
    assert (aggregate / "per_seed_results.csv").is_file()
    assert (aggregate / "recall_accuracy_vs_delay.png").is_file()
    assert (aggregate / "per_seed_recall_accuracy_vs_delay.png").is_file()
    assert (aggregate / "hypothesis_verdict.json").is_file()
    assert (aggregate / "hidden_states_heterogeneous_leaky.pt").is_file()
    assert (aggregate / "heterogeneous_group_traces.png").is_file()

    mlflow.set_tracking_uri(config.experiment.tracking_uri)
    experiment = mlflow.MlflowClient().get_experiment_by_name(config.experiment.name)
    assert experiment is not None
    client = mlflow.MlflowClient()
    tracked_runs = client.search_runs([experiment.experiment_id])
    assert len(tracked_runs) == 4
    parent_id = output_dir.name
    child_runs = [run for run in tracked_runs if run.info.run_id != parent_id]
    assert all(run.data.tags.get("mlflow.parentRunId") == parent_id for run in child_runs)
    for child in child_runs:
        assert len(client.get_metric_history(child.info.run_id, "validation_delay_1_loss")) == 2
        assert len(client.get_metric_history(child.info.run_id, "test_delay_1_loss")) == 1
    parent_test_curve = client.get_metric_history(
        parent_id, "test_mean_vanilla_rnn_recall_accuracy"
    )
    parent_epoch_curve = client.get_metric_history(
        parent_id, "epoch_mean_vanilla_rnn_validation_recall_accuracy"
    )
    assert [point.step for point in parent_test_curve] == [1, 2]
    assert [point.step for point in parent_epoch_curve] == [0, 1]
