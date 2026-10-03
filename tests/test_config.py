from pathlib import Path

import pytest

from continual_learning_lab.config import (
    AddingProblemConfig,
    DelayedRecallConfig,
    Stream51Config,
    load_config,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "naive_split_cifar10.yaml"
JOINT_CONFIG_PATH = Path(__file__).parents[1] / "configs" / "joint_cifar10_oracle.yaml"
DELAYED_CONFIG_PATH = (
    Path(__file__).parents[1] / "configs" / "heterogeneous_leaky_delayed_recall.yaml"
)
CORRECTED_DELAYED_CONFIG_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "heterogeneous_leaky_delayed_recall_corrected.yaml"
)
ADDING_CONFIG_PATH = Path(__file__).parents[1] / "configs" / "adding_problem.yaml"
STREAM51_CONFIG_PATH = (
    Path(__file__).parents[1] / "configs" / "stream51_temporal_state.yaml"
)


def test_baseline_config_loads() -> None:
    config = load_config(CONFIG_PATH)

    assert config.training.method == "naive"
    assert config.data.protocol == "sequential"
    assert config.data.task_classes == ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
    assert config.model.num_classes == 10


def test_joint_config_changes_only_identity_and_data_protocol() -> None:
    baseline = load_config(CONFIG_PATH).to_dict()
    joint = load_config(JOINT_CONFIG_PATH).to_dict()

    # Construct the complete expected config from the baseline. This catches a
    # future accidental change to any controlled condition, not just the fields
    # we happen to remember to assert individually.
    baseline["experiment"]["name"] = "experiment-001-joint-cifar10-oracle"
    baseline["experiment"]["run_name"] = "exp001-joint-seed-42"
    baseline["data"]["protocol"] = "joint"

    assert joint == baseline


def test_duplicate_or_missing_task_classes_are_rejected(tmp_path: Path) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8").replace("- [8, 9]", "- [8, 8]")
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="every class"):
        load_config(invalid_path)


def test_unknown_data_protocol_is_rejected(tmp_path: Path) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "protocol: sequential", "protocol: unsupported"
    )
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="data.protocol must be sequential or joint"):
        load_config(invalid_path)


def test_delayed_recall_config_loads_full_experiment_contract() -> None:
    config = load_config(DELAYED_CONFIG_PATH)

    assert isinstance(config, DelayedRecallConfig)
    assert config.experiment.seeds == (42, 43, 44)
    assert config.data.delays == (5, 10, 20, 40, 80)
    assert config.model.heterogeneous_group_sizes == (32, 32, 32)
    assert config.evaluation.min_long_advantage == 0.05
    assert config.data.timing_protocol == "fixed_final_cue"
    assert config.data.distractor_mode == "separate_channel"


def test_corrected_delayed_recall_config_loads_timing_contract() -> None:
    config = load_config(CORRECTED_DELAYED_CONFIG_PATH)

    assert isinstance(config, DelayedRecallConfig)
    assert config.data.delays == (5, 10, 20, 40, 80, 120)
    assert config.data.timing_protocol == "fixed_initial_value"
    assert config.data.distractor_mode == "same_content_channel"
    assert config.evaluation.selection_delays == (80, 120)
    assert config.evaluation.inspection_delay == 120


def test_adding_problem_config_separates_training_and_unseen_lengths() -> None:
    config = load_config(ADDING_CONFIG_PATH)

    assert isinstance(config, AddingProblemConfig)
    assert config.data.train_lengths == (20, 40, 80)
    assert config.data.evaluation_lengths == (20, 40, 80, 160, 320)
    assert config.evaluation.long_lengths == (160, 320)
    assert config.model.output_size == 1


def test_stream51_config_encodes_the_matched_reset_experiment_contract() -> None:
    config = load_config(STREAM51_CONFIG_PATH)

    assert isinstance(config, Stream51Config)
    assert config.experiment.seeds == (42, 43, 44)
    assert config.data.orderings == ("natural", "local_shuffle", "global_shuffle")
    assert config.data.order_seed == 42
    assert config.model.tick_counts == (1, 4)
    assert config.training.passes == 1
    assert config.training.batch_size == 1


def test_delayed_recall_rejects_groups_that_do_not_fill_hidden_state(
    tmp_path: Path,
) -> None:
    text = DELAYED_CONFIG_PATH.read_text(encoding="utf-8").replace(
        "heterogeneous_group_sizes: [32, 32, 32]",
        "heterogeneous_group_sizes: [32, 32, 31]",
    )
    invalid_path = tmp_path / "invalid-delayed.yaml"
    invalid_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="sum to model.hidden_size"):
        load_config(invalid_path)
