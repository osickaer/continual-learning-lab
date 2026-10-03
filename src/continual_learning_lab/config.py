from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    run_name: str | None
    seed: int
    device: str
    output_dir: str
    tracking_uri: str
    artifact_dir: str
    deterministic: bool


@dataclass(frozen=True)
class DataConfig:
    root: str
    download: bool
    train_batch_size: int
    eval_batch_size: int
    num_workers: int
    pin_memory: bool
    max_train_samples_per_class: int | None
    max_eval_samples_per_class: int | None
    task_classes: tuple[tuple[int, ...], ...]
    protocol: str


@dataclass(frozen=True)
class ModelConfig:
    channels: tuple[int, int]
    hidden_dim: int
    dropout: float
    num_classes: int


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    epochs_per_task: int
    optimizer: str
    learning_rate: float
    momentum: float
    weight_decay: float


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DelayedRecallExperimentConfig:
    kind: str
    name: str
    run_name: str
    seeds: tuple[int, ...]
    device: str
    output_dir: str
    tracking_uri: str
    artifact_dir: str
    deterministic: bool


@dataclass(frozen=True)
class DelayedRecallDataConfig:
    delays: tuple[int, ...]
    timing_protocol: str
    distractor_mode: str
    train_examples_per_delay: int
    validation_examples_per_delay: int
    test_examples_per_delay: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    distractor_low: float
    distractor_high: float


@dataclass(frozen=True)
class DelayedRecallModelConfig:
    input_size: int
    hidden_size: int
    num_classes: int
    homogeneous_alphas: tuple[float, ...]
    heterogeneous_alphas: tuple[float, ...]
    heterogeneous_group_sizes: tuple[int, ...]
    include_gru: bool


@dataclass(frozen=True)
class DelayedRecallTrainingConfig:
    epochs: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float


@dataclass(frozen=True)
class DelayedRecallEvaluationConfig:
    selection_delays: tuple[int, ...]
    long_delays: tuple[int, ...]
    short_delays: tuple[int, ...]
    min_long_advantage: float
    min_paired_seed_wins: int
    max_short_penalty: float
    inspection_seed: int
    inspection_delay: int
    inspection_examples_per_class: int


@dataclass(frozen=True)
class DelayedRecallConfig:
    experiment: DelayedRecallExperimentConfig
    data: DelayedRecallDataConfig
    model: DelayedRecallModelConfig
    training: DelayedRecallTrainingConfig
    evaluation: DelayedRecallEvaluationConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AddingProblemExperimentConfig:
    kind: str
    name: str
    run_name: str
    seeds: tuple[int, ...]
    device: str
    output_dir: str
    tracking_uri: str
    artifact_dir: str
    deterministic: bool


@dataclass(frozen=True)
class AddingProblemDataConfig:
    train_lengths: tuple[int, ...]
    evaluation_lengths: tuple[int, ...]
    train_examples_per_length: int
    validation_examples_per_length: int
    test_examples_per_length: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    value_low: float
    value_high: float
    marker_policy: str


@dataclass(frozen=True)
class AddingProblemModelConfig:
    input_size: int
    hidden_size: int
    output_size: int
    homogeneous_alphas: tuple[float, ...]
    heterogeneous_alphas: tuple[float, ...]
    heterogeneous_group_sizes: tuple[int, ...]
    include_gru: bool


@dataclass(frozen=True)
class AddingProblemTrainingConfig:
    epochs: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float


@dataclass(frozen=True)
class AddingProblemEvaluationConfig:
    selection_lengths: tuple[int, ...]
    long_lengths: tuple[int, ...]
    short_lengths: tuple[int, ...]
    min_long_relative_improvement: float
    min_paired_seed_wins: int
    max_short_mse_penalty: float
    inspection_seed: int
    inspection_length: int
    inspection_examples: int


@dataclass(frozen=True)
class AddingProblemConfig:
    experiment: AddingProblemExperimentConfig
    data: AddingProblemDataConfig
    model: AddingProblemModelConfig
    training: AddingProblemTrainingConfig
    evaluation: AddingProblemEvaluationConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stream51ExperimentConfig:
    kind: str
    name: str
    run_name: str
    seeds: tuple[int, ...]
    device: str
    output_dir: str
    tracking_uri: str
    artifact_dir: str
    deterministic: bool


@dataclass(frozen=True)
class Stream51DataConfig:
    root: str
    train_metadata: str
    test_metadata: str
    feature_cache: str
    orderings: tuple[str, ...]
    order_seed: int
    bbox_crop: bool
    bbox_padding_ratio: float
    feature_batch_size: int
    num_workers: int
    pin_memory: bool
    max_trajectories: int | None


@dataclass(frozen=True)
class Stream51ModelConfig:
    encoder_weights: str
    feature_size: int
    hidden_size: int
    num_classes: int
    tick_counts: tuple[int, ...]


@dataclass(frozen=True)
class Stream51TrainingConfig:
    passes: int
    batch_size: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float


@dataclass(frozen=True)
class Stream51EvaluationConfig:
    window_size: int
    eval_batch_size: int
    min_natural_persistence_gain: float
    min_natural_over_global_gain: float
    max_heldout_macro_penalty: float
    min_multitick_interaction_gain: float
    min_paired_seed_wins: int


@dataclass(frozen=True)
class Stream51Config:
    experiment: Stream51ExperimentConfig
    data: Stream51DataConfig
    model: Stream51ModelConfig
    training: Stream51TrainingConfig
    evaluation: Stream51EvaluationConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LoadedConfig = Config | DelayedRecallConfig | AddingProblemConfig | Stream51Config


def _require_sections(raw: dict[str, Any], expected: set[str]) -> None:
    missing = expected - raw.keys()
    extra = raw.keys() - expected
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unknown configuration sections: {sorted(extra)}")


def _check_keys(section: str, values: dict[str, Any], expected: set[str]) -> None:
    missing = expected - values.keys()
    extra = values.keys() - expected
    if missing:
        raise ValueError(f"Missing keys in '{section}': {sorted(missing)}")
    if extra:
        raise ValueError(f"Unknown keys in '{section}': {sorted(extra)}")


def load_config(path: str | Path) -> LoadedConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    experiment_raw = raw.get("experiment")
    if isinstance(experiment_raw, dict) and experiment_raw.get("kind") == "delayed_recall":
        return _load_delayed_recall_config(raw)
    if isinstance(experiment_raw, dict) and experiment_raw.get("kind") == "adding_problem":
        return _load_adding_problem_config(raw)
    if (
        isinstance(experiment_raw, dict)
        and experiment_raw.get("kind") == "stream51_temporal_state"
    ):
        return _load_stream51_config(raw)

    _require_sections(raw, {"experiment", "data", "model", "training"})

    experiment = raw["experiment"]
    data = raw["data"]
    model = raw["model"]
    training = raw["training"]
    if not all(isinstance(section, dict) for section in (experiment, data, model, training)):
        raise ValueError("Every configuration section must be a mapping")

    _check_keys("experiment", experiment, set(ExperimentConfig.__dataclass_fields__))
    _check_keys("data", data, set(DataConfig.__dataclass_fields__))
    _check_keys("model", model, set(ModelConfig.__dataclass_fields__))
    _check_keys("training", training, set(TrainingConfig.__dataclass_fields__))

    config = Config(
        experiment=ExperimentConfig(**experiment),
        data=DataConfig(
            **{key: value for key, value in data.items() if key != "task_classes"},
            task_classes=tuple(tuple(classes) for classes in data["task_classes"]),
        ),
        model=ModelConfig(
            **{key: value for key, value in model.items() if key != "channels"},
            channels=tuple(model["channels"]),
        ),
        training=TrainingConfig(**training),
    )
    validate_config(config)
    return config


def _load_delayed_recall_config(raw: dict[str, Any]) -> DelayedRecallConfig:
    _require_sections(
        raw,
        {"experiment", "data", "model", "training", "evaluation"},
    )
    sections = tuple(raw[name] for name in ("experiment", "data", "model", "training", "evaluation"))
    if not all(isinstance(section, dict) for section in sections):
        raise ValueError("Every configuration section must be a mapping")

    experiment, data, model, training, evaluation = sections
    _check_keys(
        "experiment", experiment, set(DelayedRecallExperimentConfig.__dataclass_fields__)
    )
    _check_keys("data", data, set(DelayedRecallDataConfig.__dataclass_fields__))
    _check_keys("model", model, set(DelayedRecallModelConfig.__dataclass_fields__))
    _check_keys("training", training, set(DelayedRecallTrainingConfig.__dataclass_fields__))
    _check_keys(
        "evaluation", evaluation, set(DelayedRecallEvaluationConfig.__dataclass_fields__)
    )

    config = DelayedRecallConfig(
        experiment=DelayedRecallExperimentConfig(
            **{key: value for key, value in experiment.items() if key != "seeds"},
            seeds=tuple(experiment["seeds"]),
        ),
        data=DelayedRecallDataConfig(
            **{key: value for key, value in data.items() if key != "delays"},
            delays=tuple(data["delays"]),
        ),
        model=DelayedRecallModelConfig(
            **{
                key: value
                for key, value in model.items()
                if key
                not in {
                    "homogeneous_alphas",
                    "heterogeneous_alphas",
                    "heterogeneous_group_sizes",
                }
            },
            homogeneous_alphas=tuple(model["homogeneous_alphas"]),
            heterogeneous_alphas=tuple(model["heterogeneous_alphas"]),
            heterogeneous_group_sizes=tuple(model["heterogeneous_group_sizes"]),
        ),
        training=DelayedRecallTrainingConfig(**training),
        evaluation=DelayedRecallEvaluationConfig(
            **{
                key: value
                for key, value in evaluation.items()
                if key not in {"selection_delays", "long_delays", "short_delays"}
            },
            selection_delays=tuple(evaluation["selection_delays"]),
            long_delays=tuple(evaluation["long_delays"]),
            short_delays=tuple(evaluation["short_delays"]),
        ),
    )
    validate_delayed_recall_config(config)
    return config


def _load_adding_problem_config(raw: dict[str, Any]) -> AddingProblemConfig:
    _require_sections(raw, {"experiment", "data", "model", "training", "evaluation"})
    sections = tuple(
        raw[name] for name in ("experiment", "data", "model", "training", "evaluation")
    )
    if not all(isinstance(section, dict) for section in sections):
        raise ValueError("Every configuration section must be a mapping")

    experiment, data, model, training, evaluation = sections
    _check_keys("experiment", experiment, set(AddingProblemExperimentConfig.__dataclass_fields__))
    _check_keys("data", data, set(AddingProblemDataConfig.__dataclass_fields__))
    _check_keys("model", model, set(AddingProblemModelConfig.__dataclass_fields__))
    _check_keys("training", training, set(AddingProblemTrainingConfig.__dataclass_fields__))
    _check_keys("evaluation", evaluation, set(AddingProblemEvaluationConfig.__dataclass_fields__))

    config = AddingProblemConfig(
        experiment=AddingProblemExperimentConfig(
            **{key: value for key, value in experiment.items() if key != "seeds"},
            seeds=tuple(experiment["seeds"]),
        ),
        data=AddingProblemDataConfig(
            **{
                key: value
                for key, value in data.items()
                if key not in {"train_lengths", "evaluation_lengths"}
            },
            train_lengths=tuple(data["train_lengths"]),
            evaluation_lengths=tuple(data["evaluation_lengths"]),
        ),
        model=AddingProblemModelConfig(
            **{
                key: value
                for key, value in model.items()
                if key
                not in {
                    "homogeneous_alphas",
                    "heterogeneous_alphas",
                    "heterogeneous_group_sizes",
                }
            },
            homogeneous_alphas=tuple(model["homogeneous_alphas"]),
            heterogeneous_alphas=tuple(model["heterogeneous_alphas"]),
            heterogeneous_group_sizes=tuple(model["heterogeneous_group_sizes"]),
        ),
        training=AddingProblemTrainingConfig(**training),
        evaluation=AddingProblemEvaluationConfig(
            **{
                key: value
                for key, value in evaluation.items()
                if key not in {"selection_lengths", "long_lengths", "short_lengths"}
            },
            selection_lengths=tuple(evaluation["selection_lengths"]),
            long_lengths=tuple(evaluation["long_lengths"]),
            short_lengths=tuple(evaluation["short_lengths"]),
        ),
    )
    validate_adding_problem_config(config)
    return config


def _load_stream51_config(raw: dict[str, Any]) -> Stream51Config:
    _require_sections(raw, {"experiment", "data", "model", "training", "evaluation"})
    sections = tuple(
        raw[name] for name in ("experiment", "data", "model", "training", "evaluation")
    )
    if not all(isinstance(section, dict) for section in sections):
        raise ValueError("Every configuration section must be a mapping")

    experiment, data, model, training, evaluation = sections
    _check_keys("experiment", experiment, set(Stream51ExperimentConfig.__dataclass_fields__))
    _check_keys("data", data, set(Stream51DataConfig.__dataclass_fields__))
    _check_keys("model", model, set(Stream51ModelConfig.__dataclass_fields__))
    _check_keys("training", training, set(Stream51TrainingConfig.__dataclass_fields__))
    _check_keys("evaluation", evaluation, set(Stream51EvaluationConfig.__dataclass_fields__))

    config = Stream51Config(
        experiment=Stream51ExperimentConfig(
            **{key: value for key, value in experiment.items() if key != "seeds"},
            seeds=tuple(experiment["seeds"]),
        ),
        data=Stream51DataConfig(
            **{key: value for key, value in data.items() if key != "orderings"},
            orderings=tuple(data["orderings"]),
        ),
        model=Stream51ModelConfig(
            **{key: value for key, value in model.items() if key != "tick_counts"},
            tick_counts=tuple(model["tick_counts"]),
        ),
        training=Stream51TrainingConfig(**training),
        evaluation=Stream51EvaluationConfig(**evaluation),
    )
    validate_stream51_config(config)
    return config


def validate_config(config: Config) -> None:
    if config.experiment.seed < 0:
        raise ValueError("experiment.seed must be non-negative")
    if config.experiment.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("experiment.device must be auto, cpu, cuda, or mps")
    if not config.data.task_classes:
        raise ValueError("data.task_classes cannot be empty")
    if config.data.protocol not in {"sequential", "joint"}:
        raise ValueError("data.protocol must be sequential or joint")

    flat_classes = [class_id for task in config.data.task_classes for class_id in task]
    expected_classes = list(range(config.model.num_classes))
    if sorted(flat_classes) != expected_classes:
        raise ValueError(
            "data.task_classes must contain every class from 0 to "
            f"{config.model.num_classes - 1} exactly once"
        )
    if any(not task for task in config.data.task_classes):
        raise ValueError("Each task must contain at least one class")
    if len(config.model.channels) != 2 or any(value <= 0 for value in config.model.channels):
        raise ValueError("model.channels must contain two positive integers")
    if config.model.hidden_dim <= 0:
        raise ValueError("model.hidden_dim must be positive")
    if not 0.0 <= config.model.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if config.training.method != "naive":
        raise ValueError("Only training.method='naive' is implemented")
    if config.training.optimizer not in {"sgd", "adamw"}:
        raise ValueError("training.optimizer must be sgd or adamw")
    if config.training.epochs_per_task <= 0:
        raise ValueError("training.epochs_per_task must be positive")
    if config.training.learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive")
    if config.data.train_batch_size <= 0 or config.data.eval_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if config.data.num_workers < 0:
        raise ValueError("data.num_workers cannot be negative")
    for name, value in (
        ("max_train_samples_per_class", config.data.max_train_samples_per_class),
        ("max_eval_samples_per_class", config.data.max_eval_samples_per_class),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"data.{name} must be positive or null")


def validate_delayed_recall_config(config: DelayedRecallConfig) -> None:
    experiment = config.experiment
    data = config.data
    model = config.model
    training = config.training
    evaluation = config.evaluation

    if experiment.kind != "delayed_recall":
        raise ValueError("experiment.kind must be delayed_recall")
    if experiment.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("experiment.device must be auto, cpu, cuda, or mps")
    if not experiment.seeds or len(set(experiment.seeds)) != len(experiment.seeds):
        raise ValueError("experiment.seeds must contain unique values")
    if any(seed < 0 for seed in experiment.seeds):
        raise ValueError("experiment.seeds must be non-negative")

    if not data.delays or tuple(sorted(set(data.delays))) != data.delays:
        raise ValueError("data.delays must be positive, unique, and sorted")
    if any(delay <= 0 for delay in data.delays):
        raise ValueError("data.delays must be positive, unique, and sorted")
    for name, count in (
        ("train_examples_per_delay", data.train_examples_per_delay),
        ("validation_examples_per_delay", data.validation_examples_per_delay),
        ("test_examples_per_delay", data.test_examples_per_delay),
    ):
        if count <= 0 or count % 2 != 0:
            raise ValueError(f"data.{name} must be a positive even integer")
    if data.batch_size <= 0:
        raise ValueError("data.batch_size must be positive")
    if data.num_workers < 0:
        raise ValueError("data.num_workers cannot be negative")
    if data.distractor_low >= data.distractor_high:
        raise ValueError("data.distractor_low must be less than distractor_high")
    if data.timing_protocol not in {"fixed_final_cue", "fixed_initial_value"}:
        raise ValueError(
            "data.timing_protocol must be fixed_final_cue or fixed_initial_value"
        )
    if data.distractor_mode not in {"separate_channel", "same_content_channel"}:
        raise ValueError(
            "data.distractor_mode must be separate_channel or same_content_channel"
        )

    if model.input_size != 3:
        raise ValueError("model.input_size must be 3 for the delayed-recall inputs")
    if model.num_classes != 2:
        raise ValueError("model.num_classes must be 2 for -1/+1 recall")
    if model.hidden_size <= 0:
        raise ValueError("model.hidden_size must be positive")
    if not model.homogeneous_alphas:
        raise ValueError("model.homogeneous_alphas cannot be empty")
    all_alphas = model.homogeneous_alphas + model.heterogeneous_alphas
    if any(not 0.0 < alpha <= 1.0 for alpha in all_alphas):
        raise ValueError("All alpha values must be in (0, 1]")
    if len(model.heterogeneous_alphas) != len(model.heterogeneous_group_sizes):
        raise ValueError("Each heterogeneous alpha requires one group size")
    if any(size <= 0 for size in model.heterogeneous_group_sizes):
        raise ValueError("Heterogeneous group sizes must be positive")
    if sum(model.heterogeneous_group_sizes) != model.hidden_size:
        raise ValueError("Heterogeneous group sizes must sum to model.hidden_size")

    if training.epochs <= 0:
        raise ValueError("training.epochs must be positive")
    if training.optimizer != "adamw":
        raise ValueError("Delayed recall currently requires training.optimizer='adamw'")
    if training.learning_rate <= 0 or training.gradient_clip_norm <= 0:
        raise ValueError("Learning rate and gradient clip norm must be positive")
    if training.weight_decay < 0:
        raise ValueError("training.weight_decay cannot be negative")

    delay_set = set(data.delays)
    for name, delays in (
        ("selection_delays", evaluation.selection_delays),
        ("long_delays", evaluation.long_delays),
        ("short_delays", evaluation.short_delays),
    ):
        if not delays or not set(delays).issubset(delay_set):
            raise ValueError(f"evaluation.{name} must be a non-empty subset of data.delays")
    if not 0.0 <= evaluation.min_long_advantage <= 1.0:
        raise ValueError("evaluation.min_long_advantage must be in [0, 1]")
    if not 0.0 <= evaluation.max_short_penalty <= 1.0:
        raise ValueError("evaluation.max_short_penalty must be in [0, 1]")
    if not 1 <= evaluation.min_paired_seed_wins <= len(experiment.seeds):
        raise ValueError("evaluation.min_paired_seed_wins must fit experiment.seeds")
    if evaluation.inspection_seed not in experiment.seeds:
        raise ValueError("evaluation.inspection_seed must be one of experiment.seeds")
    if evaluation.inspection_delay not in data.delays:
        raise ValueError("evaluation.inspection_delay must be one of data.delays")
    if evaluation.inspection_examples_per_class <= 0:
        raise ValueError("evaluation.inspection_examples_per_class must be positive")


def validate_adding_problem_config(config: AddingProblemConfig) -> None:
    experiment = config.experiment
    data = config.data
    model = config.model
    training = config.training
    evaluation = config.evaluation

    if experiment.kind != "adding_problem":
        raise ValueError("experiment.kind must be adding_problem")
    if experiment.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("experiment.device must be auto, cpu, cuda, or mps")
    if not experiment.seeds or len(set(experiment.seeds)) != len(experiment.seeds):
        raise ValueError("experiment.seeds must contain unique values")
    if any(seed < 0 for seed in experiment.seeds):
        raise ValueError("experiment.seeds must be non-negative")

    for name, lengths in (
        ("train_lengths", data.train_lengths),
        ("evaluation_lengths", data.evaluation_lengths),
    ):
        if not lengths or tuple(sorted(set(lengths))) != lengths:
            raise ValueError(f"data.{name} must be positive, unique, and sorted")
        if any(length < 2 for length in lengths):
            raise ValueError(f"data.{name} values must be at least 2")
    if not set(data.train_lengths).issubset(data.evaluation_lengths):
        raise ValueError("data.train_lengths must be a subset of evaluation_lengths")
    for name, count in (
        ("train_examples_per_length", data.train_examples_per_length),
        ("validation_examples_per_length", data.validation_examples_per_length),
        ("test_examples_per_length", data.test_examples_per_length),
    ):
        if count <= 0:
            raise ValueError(f"data.{name} must be positive")
    if data.batch_size <= 0 or data.num_workers < 0:
        raise ValueError("Adding-problem batch size must be positive and workers non-negative")
    if data.value_low >= data.value_high:
        raise ValueError("data.value_low must be less than value_high")
    if data.marker_policy != "split_halves":
        raise ValueError("data.marker_policy must be split_halves")

    if model.input_size != 2 or model.output_size != 1:
        raise ValueError("Adding Problem requires model input_size=2 and output_size=1")
    if model.hidden_size <= 0:
        raise ValueError("model.hidden_size must be positive")
    if not model.homogeneous_alphas:
        raise ValueError("model.homogeneous_alphas cannot be empty")
    all_alphas = model.homogeneous_alphas + model.heterogeneous_alphas
    if any(not 0.0 < alpha <= 1.0 for alpha in all_alphas):
        raise ValueError("All alpha values must be in (0, 1]")
    if len(model.heterogeneous_alphas) != len(model.heterogeneous_group_sizes):
        raise ValueError("Each heterogeneous alpha requires one group size")
    if any(size <= 0 for size in model.heterogeneous_group_sizes):
        raise ValueError("Heterogeneous group sizes must be positive")
    if sum(model.heterogeneous_group_sizes) != model.hidden_size:
        raise ValueError("Heterogeneous group sizes must sum to model.hidden_size")

    if training.epochs <= 0 or training.optimizer != "adamw":
        raise ValueError("Adding Problem requires positive epochs and optimizer='adamw'")
    if training.learning_rate <= 0 or training.gradient_clip_norm <= 0:
        raise ValueError("Learning rate and gradient clip norm must be positive")
    if training.weight_decay < 0:
        raise ValueError("training.weight_decay cannot be negative")

    evaluation_set = set(data.evaluation_lengths)
    train_set = set(data.train_lengths)
    for name, lengths in (
        ("selection_lengths", evaluation.selection_lengths),
        ("short_lengths", evaluation.short_lengths),
    ):
        if not lengths or not set(lengths).issubset(train_set):
            raise ValueError(f"evaluation.{name} must be a non-empty training-length subset")
    if not evaluation.long_lengths or not set(evaluation.long_lengths).issubset(evaluation_set - train_set):
        raise ValueError("evaluation.long_lengths must be unseen evaluation lengths")
    if not 0.0 <= evaluation.min_long_relative_improvement <= 1.0:
        raise ValueError("min_long_relative_improvement must be in [0, 1]")
    if evaluation.max_short_mse_penalty < 0:
        raise ValueError("max_short_mse_penalty cannot be negative")
    if not 1 <= evaluation.min_paired_seed_wins <= len(experiment.seeds):
        raise ValueError("min_paired_seed_wins must fit experiment.seeds")
    if evaluation.inspection_seed not in experiment.seeds:
        raise ValueError("inspection_seed must be one of experiment.seeds")
    if evaluation.inspection_length not in data.evaluation_lengths:
        raise ValueError("inspection_length must be an evaluation length")
    if evaluation.inspection_examples <= 0:
        raise ValueError("inspection_examples must be positive")


def validate_stream51_config(config: Stream51Config) -> None:
    experiment = config.experiment
    data = config.data
    model = config.model
    training = config.training
    evaluation = config.evaluation

    if experiment.kind != "stream51_temporal_state":
        raise ValueError("experiment.kind must be stream51_temporal_state")
    if experiment.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("experiment.device must be auto, cpu, cuda, or mps")
    if not experiment.seeds or len(set(experiment.seeds)) != len(experiment.seeds):
        raise ValueError("experiment.seeds must contain unique values")
    if any(seed < 0 for seed in experiment.seeds):
        raise ValueError("experiment.seeds must be non-negative")

    expected_orderings = ("natural", "local_shuffle", "global_shuffle")
    if data.orderings != expected_orderings:
        raise ValueError(f"data.orderings must be exactly {expected_orderings}")
    if data.order_seed < 0:
        raise ValueError("data.order_seed must be non-negative")
    if data.bbox_padding_ratio < 1.0:
        raise ValueError("data.bbox_padding_ratio must be at least 1.0")
    if data.feature_batch_size <= 0 or data.num_workers < 0:
        raise ValueError("Feature batch size must be positive and workers non-negative")
    if data.max_trajectories is not None and data.max_trajectories <= 0:
        raise ValueError("data.max_trajectories must be positive or null")

    if model.encoder_weights != "IMAGENET1K_V1":
        raise ValueError("model.encoder_weights must be IMAGENET1K_V1")
    if model.feature_size != 512:
        raise ValueError("ResNet-18 requires model.feature_size=512")
    if model.hidden_size <= 0 or model.num_classes != 51:
        raise ValueError("Stream-51 requires positive hidden_size and num_classes=51")
    if model.tick_counts != (1, 4):
        raise ValueError("model.tick_counts must be exactly [1, 4]")

    if training.passes != 1 or training.batch_size != 1:
        raise ValueError("Stream-51 requires one pass with batch_size=1")
    if training.optimizer != "adamw":
        raise ValueError("Stream-51 requires training.optimizer='adamw'")
    if training.learning_rate <= 0 or training.gradient_clip_norm <= 0:
        raise ValueError("Learning rate and gradient clip norm must be positive")
    if training.weight_decay < 0:
        raise ValueError("training.weight_decay cannot be negative")

    if evaluation.window_size <= 0 or evaluation.eval_batch_size <= 0:
        raise ValueError("Evaluation window and batch sizes must be positive")
    for name, value in (
        ("min_natural_persistence_gain", evaluation.min_natural_persistence_gain),
        ("min_natural_over_global_gain", evaluation.min_natural_over_global_gain),
        ("max_heldout_macro_penalty", evaluation.max_heldout_macro_penalty),
        ("min_multitick_interaction_gain", evaluation.min_multitick_interaction_gain),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"evaluation.{name} must be in [0, 1]")
    if not 1 <= evaluation.min_paired_seed_wins <= len(experiment.seeds):
        raise ValueError("evaluation.min_paired_seed_wins must fit experiment.seeds")
