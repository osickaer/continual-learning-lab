from dataclasses import replace
from pathlib import Path

import mlflow
import torch
from torch import nn

from continual_learning_lab.config import Stream51Config, load_config
from continual_learning_lab.stream51_data import (
    PreparedStream51Data,
    Stream51Image,
    build_stream_orderings,
    ordering_diagnostics,
    reset_schedule_hash,
)
from continual_learning_lab.stream51_experiment import (
    model_settings,
    run_stream51_experiment,
    train_prequential,
)


def _config() -> Stream51Config:
    config = load_config("configs/stream51_temporal_state.yaml")
    assert isinstance(config, Stream51Config)
    return config


def _prepared(feature_size: int = 512) -> PreparedStream51Data:
    images = tuple(
        Stream51Image(
            observation_id=index,
            class_id=class_id,
            original_trajectory_id=trajectory,
            frame_num=frame,
            bbox=(10.0, 0.0, 10.0, 0.0),
            file_location=f"unused-{index}.jpg",
        )
        for index, (class_id, trajectory, frame) in enumerate(
            (
                (0, "0:0:0", 0),
                (0, "0:0:0", 1),
                (0, "0:0:0", 2),
                (1, "1:0:0", 0),
                (1, "1:0:0", 1),
                (1, "1:0:0", 2),
            )
        )
    )
    orderings, lengths = build_stream_orderings(images, seed=42)
    return PreparedStream51Data(
        train_features=torch.randn(len(images), feature_size),
        train_labels=torch.tensor([image.class_id for image in images]),
        test_features=torch.randn(4, feature_size),
        test_labels=torch.tensor([0, 0, 1, 1]),
        orderings=orderings,
        diagnostics={
            name: ordering_diagnostics(records) for name, records in orderings.items()
        },
        segment_lengths=lengths,
        reset_schedule_hash=reset_schedule_hash(lengths),
        cache_manifest={"encoder": "fixture"},
    )


class _SpyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.score = nn.Parameter(torch.tensor(-1.0))
        self.hidden_was_none: list[bool] = []

    def step(
        self, features: torch.Tensor, hidden: torch.Tensor | None, *, ticks: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del features, ticks
        self.hidden_was_none.append(hidden is None)
        logits = torch.stack((self.score, -self.score)).view(1, 2)
        return logits, self.score.view(1, 1)


def test_prequential_prediction_is_scored_before_update_and_state_uses_reset_segment() -> None:
    config = _config()
    config = replace(
        config,
        model=replace(config.model, num_classes=2),
        evaluation=replace(config.evaluation, window_size=10),
    )
    data = _prepared()
    records = data.orderings["global_shuffle"][:2]
    # Force two unrelated original trajectories into one experimental segment.
    second = replace(
        records[1],
        original_trajectory_id="different:trajectory:id",
        class_id=0,
        reset_segment_id=records[0].reset_segment_id,
        position_in_segment=1,
        reset_before=False,
    )
    first = replace(records[0], class_id=0, reset_before=True, position_in_segment=0)
    data = replace(data, train_labels=torch.zeros_like(data.train_labels))
    model = _SpyModel()

    _macro, micro, _loss, _windows, _positions, steps = train_prequential(
        model,
        (first, second),
        data,
        config,
        persistent=True,
        ticks=1,
        device=torch.device("cpu"),
        log_windows=False,
    )

    assert micro == 0.0  # Initial logits predict class 1 before either update.
    assert steps == 2
    assert model.hidden_was_none == [True, False]


def test_official_config_expands_to_thirty_six_paired_runs() -> None:
    config = _config()

    assert len(model_settings(config)) == 4
    assert (
        len(model_settings(config))
        * len(config.data.orderings)
        * len(config.experiment.seeds)
        == 36
    )


def test_reduced_stream51_suite_writes_twelve_children_and_aggregate_artifacts(
    tmp_path: Path,
) -> None:
    config = _config()
    config = replace(
        config,
        experiment=replace(
            config.experiment,
            name=f"exp004-test-{tmp_path.name}",
            run_name="reduced-test",
            seeds=(42,),
            output_dir=str(tmp_path / "outputs"),
            tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
            artifact_dir=str(tmp_path / "artifacts"),
        ),
        model=replace(config.model, hidden_size=4),
        evaluation=replace(
            config.evaluation,
            window_size=2,
            eval_batch_size=2,
            min_paired_seed_wins=1,
        ),
    )

    output_dir = run_stream51_experiment(
        config,
        "configs/stream51_temporal_state.yaml",
        torch.device("cpu"),
        prepared_data=_prepared(),
    )
    aggregate = output_dir / "aggregate"

    assert len(list((output_dir / "children").iterdir())) == 12
    assert (aggregate / "aggregate_results.csv").is_file()
    assert (aggregate / "hypothesis_verdict.json").is_file()
    assert (aggregate / "ordering_diagnostics.json").is_file()
    assert (aggregate / "reset_schedule.json").is_file()
    assert (aggregate / "prequential_macro_by_order.png").is_file()

    mlflow.set_tracking_uri(config.experiment.tracking_uri)
    experiment = mlflow.MlflowClient().get_experiment_by_name(config.experiment.name)
    assert experiment is not None
    runs = mlflow.MlflowClient().search_runs([experiment.experiment_id])
    assert len(runs) == 13
