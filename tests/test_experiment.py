import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import continual_learning_lab.experiment as experiment_module
from continual_learning_lab.config import DelayedRecallConfig, Stream51Config, load_config
from continual_learning_lab.data import JointLoaders, TaskLoaders
from continual_learning_lab.evaluation import EvaluationResult
from continual_learning_lab.experiment import _build_run_report, _run_joint, _run_sequential
from continual_learning_lab.learners.base import EpochResult


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "naive_split_cifar10.yaml"
JOINT_CONFIG_PATH = Path(__file__).parents[1] / "configs" / "joint_cifar10_oracle.yaml"
DELAYED_CONFIG_PATH = (
    Path(__file__).parents[1] / "configs" / "heterogeneous_leaky_delayed_recall.yaml"
)
STREAM51_CONFIG_PATH = (
    Path(__file__).parents[1] / "configs" / "stream51_temporal_state.yaml"
)


class RecordingLearner:
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.model = nn.Identity()
        self.events = events

    def train_epoch(
        self,
        task_id: int,
        epoch: int,
        loader: DataLoader,
    ) -> EpochResult:
        del loader
        self.events.append(("train", task_id, epoch))
        return EpochResult(epoch=epoch, loss=1.0, accuracy=0.5, examples=4)


def _loader() -> DataLoader:
    dataset = TensorDataset(torch.zeros(4, 1), torch.zeros(4, dtype=torch.long))
    return DataLoader(dataset, batch_size=2)


def test_sequential_runner_trains_each_epoch_before_task_evaluation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG_PATH)
    config = replace(
        config,
        training=replace(config.training, epochs_per_task=2),
    )
    events: list[tuple[object, ...]] = []
    learner = RecordingLearner(events)
    tasks = [
        TaskLoaders(task_id=0, classes=(0, 1), train=_loader(), test=_loader()),
        TaskLoaders(task_id=1, classes=(2, 3), train=_loader(), test=_loader()),
    ]
    test_loader_ids = {id(task.test): task.task_id for task in tasks}
    logged_training_steps: list[int] = []

    def fake_evaluate(model, loader, device) -> EvaluationResult:
        del model, device
        events.append(("evaluate", test_loader_ids[id(loader)]))
        return EvaluationResult(loss=1.0, accuracy=0.5, examples=4)

    def fake_log_metrics(metrics, step) -> None:
        if "train_loss" in metrics:
            logged_training_steps.append(step)

    monkeypatch.setattr(experiment_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(experiment_module.mlflow, "log_metrics", fake_log_metrics)

    _run_sequential(config, torch.device("cpu"), learner, tasks, tmp_path)

    assert events == [
        ("train", 0, 0),
        ("train", 0, 1),
        ("evaluate", 0),
        ("evaluate", 1),
        ("train", 1, 0),
        ("train", 1, 1),
        ("evaluate", 0),
        ("evaluate", 1),
    ]
    assert logged_training_steps == [0, 1, 2, 3]


def test_joint_runner_evaluates_overall_each_epoch_and_pairs_at_end(monkeypatch) -> None:
    config = load_config(JOINT_CONFIG_PATH)
    config = replace(
        config,
        training=replace(config.training, epochs_per_task=2),
    )
    events: list[tuple[object, ...]] = []
    learner = RecordingLearner(events)
    overall_loader = _loader()
    pair_loaders = tuple(_loader() for _ in config.data.task_classes)
    loaders = JointLoaders(
        train=_loader(),
        overall_test=overall_loader,
        pair_tests=pair_loaders,
    )
    loader_names = {id(overall_loader): "overall"}
    loader_names.update(
        {id(loader): f"pair_{index}" for index, loader in enumerate(pair_loaders)}
    )
    pair_accuracies = [0.4, 0.5, 0.6, 0.7, 0.8]
    logged_metric_groups: list[tuple[dict[str, float], int]] = []
    logged_single_metrics: list[tuple[str, float, int]] = []
    logged_params: dict[str, object] = {}

    def fake_evaluate(model, loader, device) -> EvaluationResult:
        del model, device
        loader_name = loader_names[id(loader)]
        events.append(("evaluate", loader_name))
        if loader_name == "overall":
            return EvaluationResult(loss=1.2, accuracy=0.55, examples=4)
        pair_id = int(loader_name.split("_")[1])
        return EvaluationResult(
            loss=1.0 - pair_id / 10,
            accuracy=pair_accuracies[pair_id],
            examples=4,
        )

    monkeypatch.setattr(experiment_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        experiment_module.mlflow,
        "log_metrics",
        lambda metrics, step: logged_metric_groups.append((metrics, step)),
    )
    monkeypatch.setattr(
        experiment_module.mlflow,
        "log_metric",
        lambda name, value, step: logged_single_metrics.append((name, value, step)),
    )
    monkeypatch.setattr(
        experiment_module.mlflow,
        "log_param",
        lambda name, value: logged_params.update({name: value}),
    )

    summary = _run_joint(config, torch.device("cpu"), learner, loaders)

    assert events == [
        ("train", 0, 0),
        ("evaluate", "overall"),
        ("train", 0, 1),
        ("evaluate", "overall"),
        ("evaluate", "pair_0"),
        ("evaluate", "pair_1"),
        ("evaluate", "pair_2"),
        ("evaluate", "pair_3"),
        ("evaluate", "pair_4"),
    ]
    assert summary["total_example_exposure"] == 8
    assert summary["final_overall_accuracy"] == 0.55
    assert summary["final_mean_pair_accuracy"] == 0.6
    assert len(summary["pair_results"]) == 5
    assert all("test_overall_accuracy" in metrics for metrics, _ in logged_metric_groups[:2])
    assert logged_single_metrics == []
    assert logged_metric_groups[-1] == ({"test_pair_4_accuracy": 0.8}, 1)
    assert all(
        "/" not in metric_name
        for metrics, _ in logged_metric_groups
        for metric_name in metrics
    )
    assert logged_params["training.total_example_exposure"] == 8
    assert "average_forgetting" not in summary


def test_run_reports_explain_final_metrics() -> None:
    sequential_report = _build_run_report(
        load_config(CONFIG_PATH),
        {
            "final_average_accuracy": 0.168,
            "final_average_forgetting": 0.858875,
        },
    )
    assert "85.89%" in sequential_report
    assert "task-0 forgetting" in sequential_report

    joint_report = _build_run_report(
        load_config(JOINT_CONFIG_PATH),
        {
            "final_overall_accuracy": 0.7819,
            "final_overall_loss": 0.6245,
            "final_mean_pair_accuracy": 0.7819,
            "total_example_exposure": 750_000,
            "pair_results": [],
        },
    )
    assert "78.19%" in joint_report
    assert "privileged capacity control" in joint_report


def test_run_experiment_dispatches_joint_protocol(monkeypatch, tmp_path: Path) -> None:
    config = load_config(JOINT_CONFIG_PATH)
    config = replace(
        config,
        experiment=replace(config.experiment, output_dir=str(tmp_path)),
    )
    learner = RecordingLearner([])
    loaders = JointLoaders(
        train=_loader(),
        overall_test=_loader(),
        pair_tests=tuple(_loader() for _ in config.data.task_classes),
    )
    calls: list[str] = []

    def fake_build_joint(*args, **kwargs) -> JointLoaders:
        del args, kwargs
        calls.append("build_joint")
        return loaders

    def fail_build_sequential(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Sequential data builder should not be called")

    def fake_run_joint(received_config, device, received_learner, received_loaders):
        del received_config, device
        assert received_learner is learner
        assert received_loaders is loaders
        calls.append("run_joint")
        return {
            "total_example_exposure": 8,
            "final_overall_loss": 1.0,
            "final_overall_accuracy": 0.5,
            "final_mean_pair_accuracy": 0.5,
            "pair_results": [],
        }

    @contextmanager
    def fake_start_run(received_config):
        assert received_config is config
        yield SimpleNamespace(info=SimpleNamespace(run_id="joint-test-run"))

    monkeypatch.setattr(experiment_module, "build_joint_cifar10", fake_build_joint)
    monkeypatch.setattr(experiment_module, "build_split_cifar10", fail_build_sequential)
    monkeypatch.setattr(experiment_module, "build_learner", lambda *args: learner)
    monkeypatch.setattr(experiment_module, "_run_joint", fake_run_joint)
    monkeypatch.setattr(experiment_module, "start_run", fake_start_run)
    monkeypatch.setattr(experiment_module.mlflow, "set_tag", lambda *args: None)
    monkeypatch.setattr(experiment_module.mlflow, "log_param", lambda *args: None)
    monkeypatch.setattr(experiment_module.mlflow, "log_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment_module.mlflow, "log_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment_module.mlflow, "log_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment_module.torch, "save", lambda *args: None)

    run_dir = experiment_module.run_experiment(config, JOINT_CONFIG_PATH)

    assert calls == ["build_joint", "run_joint"]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == "joint-test-run"
    assert summary["final_overall_accuracy"] == 0.5
    assert "final_average_forgetting" not in summary


def test_run_experiment_dispatches_delayed_recall_suite(monkeypatch, tmp_path: Path) -> None:
    import continual_learning_lab.delayed_recall_experiment as delayed_module

    config = load_config(DELAYED_CONFIG_PATH)
    assert isinstance(config, DelayedRecallConfig)
    expected_path = tmp_path / "delayed-suite"
    calls: list[tuple[object, ...]] = []

    def fake_run(received_config, received_path, device):
        calls.append((received_config, received_path, device))
        return expected_path

    monkeypatch.setattr(delayed_module, "run_delayed_recall_experiment", fake_run)
    monkeypatch.setattr(
        experiment_module, "resolve_device", lambda requested: torch.device("cpu")
    )

    result = experiment_module.run_experiment(config, DELAYED_CONFIG_PATH)

    assert result == expected_path
    assert calls == [(config, DELAYED_CONFIG_PATH, torch.device("cpu"))]


def test_run_experiment_dispatches_stream51_suite(monkeypatch, tmp_path: Path) -> None:
    import continual_learning_lab.stream51_experiment as stream51_module

    config = load_config(STREAM51_CONFIG_PATH)
    assert isinstance(config, Stream51Config)
    expected_path = tmp_path / "stream51-suite"
    calls: list[tuple[object, ...]] = []

    def fake_run(received_config, received_path, device):
        calls.append((received_config, received_path, device))
        return expected_path

    monkeypatch.setattr(stream51_module, "run_stream51_experiment", fake_run)
    monkeypatch.setattr(
        experiment_module, "resolve_device", lambda requested: torch.device("cpu")
    )

    result = experiment_module.run_experiment(config, STREAM51_CONFIG_PATH)

    assert result == expected_path
    assert calls == [(config, STREAM51_CONFIG_PATH, torch.device("cpu"))]
