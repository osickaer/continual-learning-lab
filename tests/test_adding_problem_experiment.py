from dataclasses import replace
from pathlib import Path

import mlflow
import torch

from continual_learning_lab.adding_problem_experiment import (
    AddingMetrics,
    AddingRunResult,
    evaluate_hypothesis,
    model_settings,
    run_adding_problem_experiment,
    select_homogeneous_alpha,
)
from continual_learning_lab.config import AddingProblemConfig, load_config


def _config() -> AddingProblemConfig:
    config = load_config("configs/adding_problem.yaml")
    assert isinstance(config, AddingProblemConfig)
    return config


def _result(model_type: str, seed: int, alpha: float | None = None) -> AddingRunResult:
    validation_mse = 0.02 if alpha != 0.1 else 0.01
    test_mse = {20: 0.015, 40: 0.015, 80: 0.02, 160: 0.05, 320: 0.05}
    if model_type == "vanilla_rnn":
        test_mse.update({20: 0.01, 40: 0.01, 160: 0.10, 320: 0.10})
    elif model_type == "homogeneous_leaky" and alpha == 0.1:
        test_mse.update({20: 0.01, 40: 0.01, 160: 0.08, 320: 0.08})
    elif model_type == "gru":
        test_mse = {length: 0.001 for length in test_mse}
    return AddingRunResult(
        run_id=f"{model_type}-{alpha}-{seed}",
        seed=seed,
        model_type=model_type,
        alpha=alpha,
        trainable_parameters=10,
        training_seconds=1.0,
        validation={
            length: AddingMetrics(validation_mse, 0.1, 10)
            for length in (20, 40, 80)
        },
        test={length: AddingMetrics(mse, 0.1, 10) for length, mse in test_mse.items()},
        model_state_path=Path("unused.pt"),
    )


def test_adding_problem_suite_has_eighteen_runs_and_gru_does_not_gate() -> None:
    config = _config()
    results = []
    for seed in config.experiment.seeds:
        results.extend(
            [
                _result("vanilla_rnn", seed),
                _result("homogeneous_leaky", seed, 0.5),
                _result("homogeneous_leaky", seed, 0.1),
                _result("homogeneous_leaky", seed, 0.01),
                _result("heterogeneous_leaky", seed),
                _result("gru", seed),
            ]
        )

    assert len(model_settings(config)) * len(config.experiment.seeds) == 18
    selected = select_homogeneous_alpha(results, config)
    verdict = evaluate_hypothesis(results, config, selected)
    assert selected == 0.1
    assert verdict["supported"] is True


def test_reduced_adding_problem_suite_runs_without_downloads(tmp_path: Path) -> None:
    config = _config()
    config = replace(
        config,
        experiment=replace(
            config.experiment,
            name=f"exp003-test-{tmp_path.name}",
            run_name="reduced-test",
            seeds=(42,),
            output_dir=str(tmp_path / "outputs"),
            tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
            artifact_dir=str(tmp_path / "artifacts"),
        ),
        data=replace(
            config.data,
            train_lengths=(4, 6),
            evaluation_lengths=(4, 6, 8),
            train_examples_per_length=8,
            validation_examples_per_length=4,
            test_examples_per_length=4,
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
            selection_lengths=(6,),
            long_lengths=(8,),
            short_lengths=(4,),
            min_paired_seed_wins=1,
            inspection_length=8,
            inspection_examples=1,
        ),
    )

    output_dir = run_adding_problem_experiment(
        config, "configs/adding_problem.yaml", torch.device("cpu")
    )
    aggregate = output_dir / "aggregate"
    assert len(list((output_dir / "children").iterdir())) == 3
    assert (aggregate / "mse_vs_sequence_length.png").is_file()
    assert (aggregate / "per_seed_mse_vs_sequence_length.png").is_file()
    assert (aggregate / "hypothesis_verdict.json").is_file()
    assert (aggregate / "hidden_states_heterogeneous_leaky.pt").is_file()
    assert (aggregate / "heterogeneous_group_traces.png").is_file()

    mlflow.set_tracking_uri(config.experiment.tracking_uri)
    experiment = mlflow.MlflowClient().get_experiment_by_name(config.experiment.name)
    assert experiment is not None
    runs = mlflow.MlflowClient().search_runs([experiment.experiment_id])
    assert len(runs) == 4
